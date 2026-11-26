#!/usr/bin/env python3
"""
去除视频中的人声（保留击球声 / 环境声 / 观众声）。

四种后端:
    demucs       AI 人声分离（htdemucs, torch 版），把音轨拆成 vocals + no_vocals
    demucs-onnx  同一套 HT-Demucs 模型的 ONNX 版（onnxruntime, 不需要 torch）
    center       卡拉OK 中心声道相减，零依赖、秒出结果，但会削掉立体声、只对人声居中有效
    mute         直接删除音轨（纯静音，最彻底）

用法:
    # 默认: demucs 分离 + 音量提升 1 倍(+6dB), 视频流直接复制(不重编码)
    # 输出 <原名>_novoice.mp4
    python src/remove_narration.py --video Videos/fb1080.mp4

    # 只处理前 60 秒做效果验证
    python src/remove_narration.py --video Videos/fb1080.mp4 --duration 60 --output tmp/test_novoice.mp4

    # 音量保持不变 / 提升到 2.5 倍
    python src/remove_narration.py --video in.mp4 --volume-gain 1.0
    python src/remove_narration.py --video in.mp4 --volume-gain 2.5

    # 卡拉OK 相减（秒出，适合快速试听）
    python src/remove_narration.py --video Videos/fb1080.mp4 --method center

    # ONNX 人声分离（不依赖 torch; 首次会自动下载模型）
    python src/remove_narration.py --video Videos/fb1080.mp4 --method demucs-onnx

    # CPU 跑推理 + 更激进地减掉分离残留的人声
    python src/remove_narration.py --video Videos/fb1080.mp4 --device cpu --residual 0.15

参数说明:
    --video       源视频路径
    --output      输出视频路径（默认: 源视频同目录 <原名>_novoice.mp4）
    --method      去人声方式: demucs / demucs-onnx / center / mute（默认 demucs）
    --model       demucs 模型: htdemucs / htdemucs_ft / mdx_extra（默认 htdemucs）
    --device      auto / cuda / cpu / dml（默认 auto；demucs-onnx 的 auto = CPU，
                  因为 DirectML 在部分显卡上会静默输出全零）
    --shifts      demucs 位移平均次数, 越大越干净越慢（默认 1）
    --residual    再减去 vocals 轨的比例, 用于压掉分离残留人声（0 表示不减, 默认 0）
    --onnx-precision    demucs-onnx 权重精度: fp16weights(默认, 下载更小) / fp32
    --onnx-cache-dir    demucs-onnx 模型缓存目录（留空 = HuggingFace 默认缓存）
    --volume-gain 去人声后的音量倍数（默认 2.0, 即提升 1 倍 / +6dB; 1.0 表示不变）
    --no-limiter  不加防削波限幅器（默认在增益后自动接 alimiter, 峰值超 0dBFS 才动作）
    --audio-bitrate  输出音频码率（默认 192k）
    --start       起始时间（秒, 默认 0）
    --duration    处理时长（秒, 默认整段）
    --reencode    强制重编码视频流（默认只在 --start > 0 时重编码, 否则直接复制）
    --keep-temp   保留下载/分离出的中间 wav（默认处理完删除）
"""

import json
import os
import subprocess
import sys
import shutil
from argparse import ArgumentParser
from pathlib import Path

# 中间文件统一放项目根目录 tmp/（可用环境变量覆盖，打包后指向可写目录）
_TMP_ROOT = Path(os.environ.get("PPH_NARRATION_TMP")
                 or Path(__file__).resolve().parent.parent / "tmp" / "narration")


# ────────────────────────────────────────────────────────────────────────────
#  基础工具
# ────────────────────────────────────────────────────────────────────────────

def run(cmd, quiet=False):
    """执行外部命令, 失败时打印完整命令并退出"""
    if not quiet:
        print("> " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        print(f"! 命令失败 (exit={proc.returncode})")
        sys.exit(proc.returncode)
    return proc


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("! 未找到 ffmpeg / ffprobe, 请先安装并加入 PATH")
        sys.exit(1)


def probe_media(path):
    """读取视频/音频流信息"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"! 无法读取媒体信息: {path}")
        print(out.stderr.strip())
        sys.exit(1)
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "video": video,
        "audio": audio,
    }


def resolve_device(requested):
    """auto -> 有可用 torch 加速就 cuda, 否则 cpu"""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_output_path(video, output, suffix):
    if output:
        return Path(output)
    src = Path(video)
    return src.with_name(f"{src.stem}{suffix}{src.suffix}")


# ────────────────────────────────────────────────────────────────────────────
#  音频处理
# ────────────────────────────────────────────────────────────────────────────

def extract_audio(video, wav_path, start, duration, sample_rate=None):
    """把视频音轨抽成 wav（demucs 只吃音频文件）"""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if duration and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-c:a", "pcm_f32le"]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd += [str(wav_path)]
    run(cmd, quiet=True)
    return wav_path


def separate_demucs(wav_path, model, device, shifts, out_dir):
    """调用 demucs 分离人声, 返回 (no_vocals, vocals) 路径"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-d", device,
        "--shifts", str(shifts),
        "-o", str(out_dir),
        str(wav_path),
    ]
    print(f"\n[demucs] 模型={model} 设备={device} shifts={shifts}")
    run(cmd)

    stem_dir = out_dir / model / wav_path.stem
    no_vocals = stem_dir / "no_vocals.wav"
    vocals = stem_dir / "vocals.wav"
    if not no_vocals.exists():
        print(f"! demucs 未生成伴奏轨: {no_vocals}")
        sys.exit(1)
    return no_vocals, vocals


# demucs-onnx 支持的模型（同一套 HT-Demucs 权重的 ONNX 导出）
_ONNX_MODELS = {"htdemucs", "htdemucs_ft", "htdemucs_6s"}
# GPU 后端一旦失败就记住，后续直接用 CPU（避免每个视频都白等一次）
_ONNX_GPU_FAILED = False


def resolve_onnx_model(name):
    """把 --model 映射到 demucs-onnx 认识的模型名（不认识就回退 htdemucs）"""
    if name in _ONNX_MODELS:
        return name
    print(f"! demucs-onnx 不支持模型 {name}, 回退到 htdemucs")
    return "htdemucs"


def _onnx_default_providers(device):
    """auto 默认走 CPU

    DirectML 在部分 AMD 核显/驱动组合上不会报错，但会**静默输出全零**（实测
    gfx1151 + onnxruntime-directml 1.24.4），所以 auto 不冒险上 GPU；
    需要 GPU 时显式传 --device dml / cuda。
    """
    if device in ("", None, "auto"):
        print("  demucs-onnx 默认使用 CPU（DirectML 可能静默输出全零；"
              "需要 GPU 请显式 --device dml）")
        return "cpu"
    return device


def _wav_peak(path, seconds=10):
    """读取 wav 中段最多 seconds 秒，返回峰值（用于识别“全静音”输出）"""
    import numpy as np
    import soundfile as sf
    info = sf.info(str(path))
    frames = int(min(info.frames - info.frames // 3,
                     max(1, int(info.samplerate * seconds))))
    if frames <= 0:
        return 0.0
    data, _ = sf.read(str(path), start=info.frames // 3, frames=frames,
                      dtype="float32", always_2d=True)
    return float(np.max(np.abs(data))) if data.size else 0.0


def separate_demucs_onnx(wav_path, model, providers, precision, out_dir,
                         cache_dir=None):
    """用 demucs-onnx（onnxruntime, 不需要 torch）分离人声

    返回 (no_vocals, vocals) 两个 wav 路径。
    GPU 后端失败**或输出全静音**时自动回退 CPU，并把失败状态记在进程内。
    """
    global _ONNX_GPU_FAILED
    try:
        from demucs_onnx import separate
    except ImportError:
        print("! 未安装 demucs-onnx（ONNX 人声分离后端）")
        print("  pip install demucs-onnx")
        print("  或改用 --method center / --method mute")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    providers = _onnx_default_providers(providers)
    if _ONNX_GPU_FAILED and providers != "cpu":
        providers = "cpu"
    print(f"\n[demucs-onnx] 模型={model} providers={providers} "
          f"precision={precision}")

    def _run(prov):
        return separate(
            str(wav_path), str(out_dir), model=model, providers=prov,
            precision=precision, cache_dir=str(cache_dir) if cache_dir else None,
            mix_stems=("drums", "bass", "other"), mix_output_name="no_vocals",
        )

    source_peak = _wav_peak(wav_path)
    try:
        stems = _run(providers)
        if source_peak > 1e-3 and _wav_peak(out_dir / "no_vocals.wav") < 1e-6:
            raise RuntimeError("分离结果全为静音（后端输出异常）")
    except Exception as e:
        if providers == "cpu":
            print(f"! demucs-onnx 分离失败: {type(e).__name__}: {str(e)[:180]}")
            sys.exit(1)
        # DirectML 在部分核显/驱动上会直接报错或静默输出全零
        _ONNX_GPU_FAILED = True
        print(f"! demucs-onnx GPU 后端失败 ({type(e).__name__}), 回退 CPU 重试")
        print(f"  {str(e)[:180]}")
        print("  之后本进程内将直接使用 CPU；也可显式 --device cpu")
        stems = _run("cpu")

    no_vocals = out_dir / "no_vocals.wav"
    vocals = out_dir / "vocals.wav"
    # mix_stems 是 v0.2+ 的功能；老版本没写混音文件时自己加
    if not no_vocals.exists():
        import numpy as np
        import soundfile as sf
        sr = sf.info(str(wav_path)).samplerate
        parts = [stems[k] for k in ("drums", "bass", "other") if k in stems]
        if not parts:
            print("! demucs-onnx 未返回可混音的伴奏轨")
            sys.exit(1)
        sf.write(str(no_vocals), np.sum(parts, axis=0).T, sr)
    if not vocals.exists() and "vocals" in stems:
        import soundfile as sf
        sr = sf.info(str(wav_path)).samplerate
        sf.write(str(vocals), stems["vocals"].T, sr)
    return no_vocals, vocals


def subtract_residual(no_vocals, vocals, out_wav, ratio):
    """从伴奏轨里再减掉 ratio 倍人声, 压制分离残留"""
    if ratio <= 0:
        return no_vocals
    print(f"[residual] 伴奏 - {ratio:.2f} × 人声")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(no_vocals), "-i", str(vocals),
        "-filter_complex",
        f"[1:a]volume={ratio:.4f}[v];"
        f"[0:a][v]amix=inputs=2:duration=first:weights=1 -1:normalize=0[a]",
        "-map", "[a]", "-c:a", "pcm_f32le", str(out_wav),
    ]
    run(cmd, quiet=True)
    return out_wav


def build_audio_filter(gain, limiter=True):
    """构造音量和防削波滤镜链; gain=1.0 时返回空串"""
    parts = []
    if gain and abs(gain - 1.0) > 1e-6:
        parts.append(f"volume={gain:.4f}")
    # 增益后若某段峰值超过 0dBFS 会削波, alimiter 只在超限时动作
    if limiter and gain and gain > 1.0:
        parts.append("alimiter=limit=0.98:level=false")
    return ",".join(parts)


def mux_audio(video, audio_path, output, start, duration, bitrate,
              reencode, encoder="libx264", crf=18, audio_filter=""):
    """把处理后的音轨合回视频, 视频流默认直接复制"""
    output.parent.mkdir(parents=True, exist_ok=True)
    need_reencode = reencode or (start and start > 0)
    cmd = ["ffmpeg", "-v", "warning", "-stats", "-y"]
    if start and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video), "-i", str(audio_path)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    # 时长必须作为输出选项, 否则只作用于紧邻的输入(音频), 视频会整段复制
    if duration and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    if audio_filter:
        cmd += ["-af", audio_filter]
    if need_reencode:
        cmd += ["-c:v", encoder]
        if encoder == "libx264":
            cmd += ["-crf", str(crf), "-preset", "fast"]
        else:
            cmd += ["-b:v", "8M"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart"]
    cmd += [str(output)]
    run(cmd)


# ────────────────────────────────────────────────────────────────────────────
#  三种后端
# ────────────────────────────────────────────────────────────────────────────

def do_demucs(args, video, output, info):
    """AI 人声分离"""
    stem = Path(video).stem
    work = _TMP_ROOT / stem
    work.mkdir(parents=True, exist_ok=True)

    wav = work / "source.wav"
    print("[1/3] 提取音轨 ...")
    extract_audio(video, wav, args.start, args.duration)

    print("[2/3] demucs 人声分离 ...")
    device = resolve_device(args.device)
    no_vocals, vocals = separate_demucs(wav, args.model, device, args.shifts, work / "separated")

    audio = no_vocals
    if args.residual and args.residual > 0:
        audio = subtract_residual(no_vocals, vocals, work / "residual.wav", args.residual)

    print("[3/3] 合成输出视频 ...")
    af = build_audio_filter(args.volume_gain, not args.no_limiter)
    if af:
        print(f"      音频滤镜: {af}")
    mux_audio(video, audio, output, args.start, args.duration,
              args.audio_bitrate, args.reencode, args.encoder, args.crf,
              audio_filter=af)

    if not args.keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    return output


def do_demucs_onnx(args, video, output, info):
    """AI 人声分离（ONNX 后端，不需要 torch）"""
    stem = Path(video).stem
    work = _TMP_ROOT / f"{stem}_onnx"
    work.mkdir(parents=True, exist_ok=True)

    wav = work / "source.wav"
    print("[1/3] 提取音轨 ...")
    extract_audio(video, wav, args.start, args.duration)

    print("[2/3] demucs-onnx 人声分离 ...")
    model = resolve_onnx_model(args.model)
    no_vocals, vocals = separate_demucs_onnx(
        wav, model, args.device, args.onnx_precision,
        work / "separated", cache_dir=args.onnx_cache_dir or None)

    audio = no_vocals
    if args.residual and args.residual > 0:
        audio = subtract_residual(no_vocals, vocals, work / "residual.wav",
                                  args.residual)

    print("[3/3] 合成输出视频 ...")
    af = build_audio_filter(args.volume_gain, not args.no_limiter)
    if af:
        print(f"      音频滤镜: {af}")
    mux_audio(video, audio, output, args.start, args.duration,
              args.audio_bitrate, args.reencode, args.encoder, args.crf,
              audio_filter=af)

    if not args.keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    return output


def do_center(args, video, output, info):
    """卡拉OK 中心声道相减: 人声通常在正中, 左右声道同相 -> 相减后消失"""
    audio = info.get("audio") or {}
    channels = int(audio.get("channels") or 0)
    if channels < 2:
        print("! center 方法需要立体声音轨, 当前视频是单声道")
        print("  可改用 --method mute 或 --method demucs")
        sys.exit(1)

    af = "pan=stereo|c0=0.5*c0-0.5*c1|c1=0.5*c1-0.5*c0"
    gain_f = build_audio_filter(args.volume_gain, not args.no_limiter)
    if gain_f:
        af = f"{af},{gain_f}"
    print(f"[center] 立体声反相相减: {af}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-v", "warning", "-stats", "-y"]
    if args.start and args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", str(video)]
    if args.duration and args.duration > 0:
        cmd += ["-t", f"{args.duration:.3f}"]
    cmd += ["-map", "0:v:0", "-map", "0:a:0", "-af", af]
    if args.reencode or (args.start and args.start > 0):
        cmd += ["-c:v", args.encoder, "-crf", str(args.crf), "-preset", "fast"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate, "-movflags", "+faststart"]
    cmd += [str(output)]
    run(cmd)
    return output


def do_mute(args, video, output, info):
    """直接删除音轨"""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "warning", "-stats", "-y"]
    if args.start and args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", str(video)]
    if args.duration and args.duration > 0:
        cmd += ["-t", f"{args.duration:.3f}"]
    cmd += ["-map", "0:v:0", "-an"]
    if args.reencode or (args.start and args.start > 0):
        cmd += ["-c:v", args.encoder, "-crf", str(args.crf), "-preset", "fast"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-movflags", "+faststart", str(output)]
    run(cmd)
    return output


# ────────────────────────────────────────────────────────────────────────────
#  入口
# ────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = ArgumentParser(
        description="去除视频人声, 保留击球声/环境声",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", help="源视频路径")
    p.add_argument("--output", help="输出视频路径")
    p.add_argument("--method", choices=["demucs", "demucs-onnx", "center", "mute"],
                   default="demucs")
    p.add_argument("--model", default="htdemucs", help="demucs 模型名")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "dml"],
                   help="auto/cuda/cpu/dml（demucs 用 cuda/cpu；demucs-onnx 的 auto = CPU，"
                        "DirectML 可能静默输出全零，需显式 dml）")
    p.add_argument("--shifts", type=int, default=1)
    p.add_argument("--onnx-precision", default="fp16weights",
                   choices=["fp16weights", "fp32"],
                   help="demucs-onnx 权重精度, fp16weights 下载更小（默认）")
    p.add_argument("--onnx-cache-dir", default="",
                   help="demucs-onnx 模型缓存目录（留空 = HuggingFace 默认缓存）")
    p.add_argument("--residual", type=float, default=0.0, help="额外减掉的 vocals 比例")
    p.add_argument("--volume-gain", type=float, default=2.0,
                   help="去人声后的音量倍数, 默认 2.0 (提升 1 倍 / +6dB)")
    p.add_argument("--no-limiter", action="store_true",
                   help="增益后不加防削波限幅器 (默认 auto 加 alimiter)")
    p.add_argument("--audio-bitrate", default="192k")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--reencode", action="store_true", help="强制重编码视频流")
    p.add_argument("--encoder", default="libx264", help="重编码时的视频编码器")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--keep-temp", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    check_ffmpeg()

    if not args.video:
        args.video = input("源视频路径: ").strip().strip('"')
    video = Path(args.video)
    if not video.is_file():
        print(f"! 视频不存在: {video}")
        sys.exit(1)

    info = probe_media(video)
    if not info["video"]:
        print("! 输入文件里没有视频流")
        sys.exit(1)
    if not info["audio"]:
        print("! 输入文件里没有音轨, 无需去除人声")
        sys.exit(1)

    audio = info["audio"]
    print(f"输入: {video}")
    print(f"  视频: {info['video'].get('width')}x{info['video'].get('height')} "
          f"{info['video'].get('codec_name')}  时长 {info['duration']:.1f}s")
    print(f"  音频: {audio.get('codec_name')} {audio.get('sample_rate')}Hz "
          f"{audio.get('channels')}ch  方法 {args.method}")

    output = build_output_path(video, args.output, "_novoice")
    if output.resolve() == video.resolve():
        print("! 输出路径不能与输入相同")
        sys.exit(1)

    handlers = {"demucs": do_demucs, "demucs-onnx": do_demucs_onnx,
                "center": do_center, "mute": do_mute}
    result = handlers[args.method](args, video, output, info)

    size_mb = result.stat().st_size / 1024 / 1024
    print(f"\n完成: {result}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
