#!/usr/bin/env python3
"""
去除视频中的静态水印 / 叠加文字（台标、赛事条、比分板）。

两种算法:
    delogo   ffmpeg delogo 滤镜, 用区域周边像素插值填补, 速度快, 区域呈模糊块
    inpaint  OpenCV 逐帧 TELEA/NS 修复, 边缘过渡更自然, 速度慢

内置预设区域（实测于 1920x1080 的 WTT 转播画面, 可直接改 _PRESETS）:
    logo      右上角「咪咕 独播」台标
    overlay   logo + 底部「ISTANBUL | 百岁山 | WTT CONTENDER」赛事条
    all       overlay + 左下角比分板
    none      不用预设, 只保留 --region 手动指定的区域

用法:
    # 默认: 去全部静态叠加 (preset=all), delogo 算法, 视频重编码
    python src/remove_watermark.py --video Videos/fb1080.mp4

    # 只去右上角咪咕台标
    python src/remove_watermark.py --video Videos/fb1080.mp4 --preset logo

    # 自定义区域(可重复), 不用预设
    python src/remove_watermark.py --video in.mp4 --preset none --region 100,100,200,50 --region 800,900,300,80

    # 精修算法 + 前 30 秒试跑
    python src/remove_watermark.py --video in.mp4 --method inpaint --duration 30

    # 只导出处理前后对比静帧(不编码视频), 先看效果
    python src/remove_watermark.py --video in.mp4 --preview 3,900,2100 --dry-run

参数说明:
    --video       源视频路径
    --output      输出视频路径（默认: 源视频同目录 <原名>_nowatermark.mp4）
    --method      delogo / inpaint（默认 delogo）
    --preset      logo / overlay / all / none（默认 all）
    --region      额外区域 "x,y,w,h", 可重复
    --expand      区域外扩像素, 盖住水印抗锯齿边（默认 3）
    --crf         libx264 质量, 越小越清晰（默认 18）
    --encoder     视频编码器（默认 libx264; AMD 可用 h264_amf 加速）
    --x264-preset  x264 速度档（默认 medium, 可选 fast/slower 等）
    --inpaint-radius  OpenCV inpaint 采样半径（默认 4）
    --inpaint-method  telea / ns（默认 telea）
    --start / --duration  只处理指定时间区间
    --preview     逗号分隔的时间点(秒), 导出前后对比图到 tmp/
    --dry-run     只打印/导出预览, 不编码视频
    --list-presets  打印内置预设坐标后退出
"""

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

_TMP_ROOT = Path(__file__).resolve().parent.parent / "tmp"

# 预设区域 (x, y, w, h), 基于 1920x1080。
# 若换成别的分辨率, 用 --region 手动指定, 或按比例换算。
_PRESETS = {
    "logo": [
        (1588, 38, 292, 74),      # 右上角「咪咕 独播」
    ],
    "overlay": [
        (1588, 38, 292, 74),      # 右上角「咪咕 独播」
        (1170, 950, 650, 78),     # 底部 ISTANBUL | 百岁山 | WTT CONTENDER
    ],
    "all": [
        (1588, 38, 292, 74),      # 右上角「咪咕 独播」
        (1170, 950, 650, 78),     # 底部赛事条
        (84, 884, 462, 150),      # 左下角比分板
    ],
    "none": [],
}


# ────────────────────────────────────────────────────────────────────────────
#  基础工具
# ────────────────────────────────────────────────────────────────────────────

def run(cmd, quiet=False):
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
    if video is None:
        print("! 输入文件里没有视频流")
        sys.exit(1)

    # 平均帧率优先, 避免容器里的 r_frame_rate 过于乐观
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = video.get(key) or "0/0"
        try:
            num, den = raw.split("/")
            val = float(num) / float(den) if float(den) else 0.0
        except Exception:
            val = 0.0
        if val > 0:
            fps = val
            break
    if fps <= 0:
        fps = 25.0

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "has_audio": audio is not None,
    }


def normalize_region(x, y, w, h, width, height):
    """裁进画面内, 并保证 delogo 要求的边界（不能贴边）"""
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    x = max(1, x)
    y = max(1, y)
    if x + w > width - 1:
        w = width - 1 - x
    if y + h > height - 1:
        h = height - 1 - y
    if w < 2 or h < 2:
        print(f"! 区域无效 (越界): x={x} y={y} w={w} h={h}")
        sys.exit(1)
    return x, y, w, h


def collect_regions(args, width, height):
    """预设 + 手动 --region, 统一外扩并裁边界"""
    regions = list(_PRESETS.get(args.preset, []))
    for raw in args.region or []:
        parts = [p.strip() for p in str(raw).replace(" ", ",").split(",") if p.strip()]
        if len(parts) != 4:
            print(f"! --region 需要 4 个数字 x,y,w,h, 收到: {raw}")
            sys.exit(1)
        regions.append(tuple(float(p) for p in parts))

    if not regions:
        print("! 没有任何待处理区域, 用 --preset 或 --region 指定")
        sys.exit(1)

    out = []
    for (x, y, w, h) in regions:
        x -= args.expand
        y -= args.expand
        w += args.expand * 2
        h += args.expand * 2
        out.append(normalize_region(x, y, w, h, width, height))
    return out


def build_delogo_filter(regions):
    return ",".join(
        f"delogo=x={x}:y={y}:w={w}:h={h}" for (x, y, w, h) in regions
    )


def encoder_args(args):
    """libx264 走 crf, AMF/NVENC 走码率"""
    if args.encoder == "libx264":
        return ["-c:v", "libx264", "-crf", str(args.crf), "-preset", args.x264_preset]
    if args.encoder.endswith("_amf"):
        return ["-c:v", args.encoder, "-b:v", args.video_bitrate, "-quality", "quality"]
    if args.encoder.endswith("_nvenc"):
        return ["-c:v", args.encoder, "-cq", str(args.crf), "-b:v", "0"]
    return ["-c:v", args.encoder, "-crf", str(args.crf)]


# ────────────────────────────────────────────────────────────────────────────
#  delogo 后端
# ────────────────────────────────────────────────────────────────────────────

def do_delogo(args, video, output, info, regions):
    filt = build_delogo_filter(regions)
    print(f"[delogo] 区域: {regions}")
    print(f"[delogo] filter: {filt}")

    if args.dry_run:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "warning", "-stats", "-y"]
    if args.start and args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", str(video)]
    if args.duration and args.duration > 0:
        cmd += ["-t", f"{args.duration:.3f}"]
    cmd += ["-map", "0:v:0"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    cmd += ["-vf", filt]
    cmd += encoder_args(args)
    cmd += ["-movflags", "+faststart", str(output)]
    run(cmd)
    return output


# ────────────────────────────────────────────────────────────────────────────
#  inpaint 后端
# ────────────────────────────────────────────────────────────────────────────

def build_mask(width, height, regions):
    mask = np.zeros((height, width), np.uint8)
    for (x, y, w, h) in regions:
        mask[y:y + h, x:x + w] = 255
    return mask


def do_inpaint(args, video, output, info, regions):
    import cv2

    width, height, fps = info["width"], info["height"], info["fps"]
    mask = build_mask(width, height, regions)
    flag = cv2.INPAINT_TELEA if args.inpaint_method == "telea" else cv2.INPAINT_NS
    frame_bytes = width * height * 3

    print(f"[inpaint] {width}x{height} @ {fps:.3f}fps, 区域 {regions}")
    print(f"[inpaint] radius={args.inpaint_radius} method={args.inpaint_method}")

    if args.dry_run:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)

    # 读: ffmpeg 解码原始帧到管道
    dec = ["ffmpeg", "-v", "error"]
    if args.start and args.start > 0:
        dec += ["-ss", f"{args.start:.3f}"]
    dec += ["-i", str(video)]
    if args.duration and args.duration > 0:
        dec += ["-t", f"{args.duration:.3f}"]
    dec += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    read_proc = subprocess.Popen([str(c) for c in dec],
                                 stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    # 写: 管道喂帧给 ffmpeg 编码, 音轨直接从源视频复制
    enc = ["ffmpeg", "-v", "warning", "-stats", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
    if info["has_audio"]:
        if args.start and args.start > 0:
            enc += ["-ss", f"{args.start:.3f}"]
        enc += ["-i", str(video)]
        if args.duration and args.duration > 0:
            enc += ["-t", f"{args.duration:.3f}"]
        enc += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    enc += encoder_args(args)
    enc += ["-movflags", "+faststart", str(output)]
    write_proc = subprocess.Popen([str(c) for c in enc], stdin=subprocess.PIPE)

    total = int((args.duration if args.duration and args.duration > 0 else info["duration"]) * fps)
    done = 0
    try:
        while True:
            buf = read_proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            fixed = cv2.inpaint(frame, mask, args.inpaint_radius, flag)
            write_proc.stdin.write(fixed.tobytes())
            done += 1
            if done % 500 == 0:
                pct = done / max(total, 1) * 100
                print(f"  inpaint {done}/{total} 帧 ({pct:.1f}%)", flush=True)
    finally:
        read_proc.stdout.close()
        read_proc.wait()
        write_proc.stdin.close()
        write_proc.wait()

    if write_proc.returncode != 0:
        print(f"! 编码器退出码 {write_proc.returncode}")
        sys.exit(write_proc.returncode)
    print(f"  inpaint 完成, 共 {done} 帧")
    return output


# ────────────────────────────────────────────────────────────────────────────
#  静帧预览对比
# ────────────────────────────────────────────────────────────────────────────

def export_preview(args, video, info, regions, times):
    """对每个时间点导出「原图 | 处理后 | 区域放大」对比图"""
    import cv2

    out_dir = _TMP_ROOT / "watermark_preview" / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    filt = build_delogo_filter(regions)
    mask = build_mask(info["width"], info["height"], regions)
    flag = cv2.INPAINT_TELEA if args.inpaint_method == "telea" else cv2.INPAINT_NS

    for t in times:
        orig_png = out_dir / f"t{t:g}_orig.png"
        proc_png = out_dir / f"t{t:g}_proc.png"

        base = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "1"]
        run(base + [str(orig_png)], quiet=True)

        if args.method == "inpaint":
            # inpaint 直接在 python 侧处理单帧
            orig = cv2.imread(str(orig_png))
            if orig is None:
                print(f"! 时间点 {t}s 抽帧失败")
                continue
            cv2.imwrite(str(proc_png),
                        cv2.inpaint(orig, mask, args.inpaint_radius, flag))
        else:
            run(base + ["-vf", filt, str(proc_png)], quiet=True)

        orig = cv2.imread(str(orig_png))
        proc = cv2.imread(str(proc_png))
        if orig is None or proc is None:
            print(f"! 时间点 {t}s 抽帧失败")
            continue

        # 上: 原图 / 中: 处理后
        sheet = np.vstack([orig, proc])
        # 下: 每个区域的处理前后放大并排
        crops = []
        for (x, y, w, h) in regions:
            pad = 12
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(info["width"], x + w + pad)
            y1 = min(info["height"], y + h + pad)
            oc = orig[y0:y1, x0:x1]
            pc = proc[y0:y1, x0:x1]
            scale = max(1.0, 360.0 / max(oc.shape[1], 1))
            oc = cv2.resize(oc, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pc = cv2.resize(pc, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pair = np.vstack([oc, pc])
            if pair.shape[1] < 740:
                pad_w = 740 - pair.shape[1]
                pair = cv2.copyMakeBorder(pair, 0, 0, 0, pad_w,
                                          cv2.BORDER_CONSTANT, value=(20, 20, 20))
            crops.append(pair)
        if crops:
            # 各区域高度不同, hstack 前先补齐到最高
            max_h = max(c.shape[0] for c in crops)
            crops = [
                cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
                for c in crops
            ]
            strip = np.hstack(crops)
            if strip.shape[1] > sheet.shape[1]:
                scale = sheet.shape[1] / strip.shape[1]
                strip = cv2.resize(strip, None, fx=scale, fy=scale)
            elif strip.shape[1] < sheet.shape[1]:
                strip = cv2.copyMakeBorder(strip, 0, 0, 0,
                                           sheet.shape[1] - strip.shape[1],
                                           cv2.BORDER_CONSTANT, value=(20, 20, 20))
            sheet = np.vstack([sheet, strip])

        out_png = out_dir / f"t{t:g}_compare.png"
        cv2.imwrite(str(out_png), sheet)
        print(f"  预览: {out_png}")

    return out_dir


# ────────────────────────────────────────────────────────────────────────────
#  入口
# ────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = ArgumentParser(
        description="去除视频静态水印/叠加文字 (delogo 或 OpenCV inpaint)",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", help="源视频路径")
    p.add_argument("--output", help="输出视频路径")
    p.add_argument("--method", choices=["delogo", "inpaint"], default="delogo")
    p.add_argument("--preset", choices=list(_PRESETS.keys()), default="all")
    p.add_argument("--region", action="append", metavar="x,y,w,h")
    p.add_argument("--expand", type=int, default=3, help="区域外扩像素")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--encoder", default="libx264")
    p.add_argument("--x264-preset", default="medium")
    p.add_argument("--video-bitrate", default="10M")
    p.add_argument("--inpaint-radius", type=int, default=4)
    p.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--preview", default="", help="逗号分隔的时间点(秒)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_presets:
        for name, regs in _PRESETS.items():
            print(f"{name:8s} {regs}")
        return

    check_ffmpeg()

    if not args.video:
        args.video = input("源视频路径: ").strip().strip('"')
    video = Path(args.video)
    if not video.is_file():
        print(f"! 视频不存在: {video}")
        sys.exit(1)

    info = probe_media(video)
    regions = collect_regions(args, info["width"], info["height"])

    print(f"输入: {video}")
    print(f"  画面: {info['width']}x{info['height']} @ {info['fps']:.3f}fps  "
          f"时长 {info['duration']:.1f}s  音轨 {info['has_audio']}")
    print(f"  方法: {args.method}  预设: {args.preset}")
    for (x, y, w, h) in regions:
        print(f"  区域: x={x} y={y} w={w} h={h}")

    times = []
    if args.preview:
        times = [float(v) for v in args.preview.replace(" ", "").split(",") if v]
        export_preview(args, video, info, regions, times)

    output = Path(args.output) if args.output else \
        video.with_name(f"{video.stem}_nowatermark{video.suffix}")
    if output.resolve() == video.resolve():
        print("! 输出路径不能与输入相同")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: 不编码视频")
        if args.method == "delogo":
            print("filter: " + build_delogo_filter(regions))
        return

    handler = do_delogo if args.method == "delogo" else do_inpaint
    result = handler(args, video, output, info, regions)

    if result and result.exists():
        size_mb = result.stat().st_size / 1024 / 1024
        print(f"\n完成: {result}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
