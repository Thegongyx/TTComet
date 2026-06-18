#!/usr/bin/env python3
"""
run_pipeline.py — 一键流水线: YOLO 推理 → 高光特效剪辑

将 inference_video.py (YOLO 推理) 和 pingpong_highlight.py (特效高光)
串联为一条命令, 最少只需指定输入视频和输出名称。

流程:
    Step 1: YOLO 逐帧推理, 生成 raw_detections.jsonl
    Step 2: 球轨迹分析 + 回合检测 + 彗尾/粒子/迷你球台特效 → 高光集锦

默认输出:
    所有文件放在 ./inference_output/pipeline_<视频名>/
    ├── <视频名>_highlight.mp4    ← 最终高光视频
    ├── raw_detections.jsonl      ← YOLO 检测日志
    └── ... (中间文件默认保留)

用法:
    python run_pipeline.py --video <输入视频>        # 单个视频
    python run_pipeline.py --video <视频目录>        # 批量处理目录中所有视频

示例:
    # 最简单: 只指定视频
    python run_pipeline.py --video match.mp4

    # 批量处理目录中所有视频 (串行)
    python run_pipeline.py --video ./videos/

    # 指定模型和参数
    python run_pipeline.py --video match.mp4 --model best.pt --batch-size 8 --imgsz 1280

    # 已有 JSONL, 跳过推理直接做高光
    python run_pipeline.py --video match.mp4 --skip-inference --raw-detections existing_log.jsonl

    # 运行后清理中间文件
    python run_pipeline.py --video match.mp4 --clean

    # 使用 conda 环境
    python run_pipeline.py --video match.mp4 --conda-env yolo26
"""

import os
import sys
import time
import argparse
import subprocess
import platform
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"
_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_SCRIPT = SCRIPT_DIR / "inference_video.py"
HIGHLIGHT_SCRIPT = SCRIPT_DIR / "pingpong_highlight.py"

_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv',
    '.wmv', '.m4v', '.mpg', '.mpeg', '.ts', '.3gp',
}


def get_python_cmd(conda_env=None):
    """返回运行 Python 的命令前缀列表。
    -u: 强制无缓冲 stdout/stderr (Windows 子进程关键)
    --live-stream: conda 下实时输出"""
    if conda_env:
        return ["conda", "run", "--live-stream", "-n", conda_env, "python", "-u"]
    return [sys.executable, "-u"]


def run_step(cmd, step_name, note=None):
    """运行一个子进程步骤, 管道逐行读取实时输出到终端。返回 bool 成功与否。"""
    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}")
    print(f"  → {' '.join(cmd)}")
    if note:
        print(f"  ⏳ {note}")
    print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    t_start = time.time()
    proc = None
    try:
        # Popen + PIPE: 彻底绕过 Windows 控制台 stdout 继承问题, 逐行实时打印
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            creationflags=_SUBPROC_FLAGS,
        )

        for line in proc.stdout:  # type: ignore[union-attr] — PIPE 保证 stdout 非 None
            print(line, end="", flush=True)

        proc.wait()
        elapsed_min = (time.time() - t_start) / 60.0

        if proc.returncode != 0:
            print(f"\n  ❌ {step_name} 失败 (exit code {proc.returncode})")
            return False
        print(f"\n  ✅ {step_name} 完成  ⏱️ 耗时: {elapsed_min:.1f} min")
        return True

    except KeyboardInterrupt:
        if proc:
            proc.kill()
        elapsed_min = (time.time() - t_start) / 60.0
        print(f"\n  ⚠️ 用户中断  ⏱️ 已耗时: {elapsed_min:.1f} min")
        raise
    except Exception as e:
        if proc:
            proc.kill()
        elapsed_min = (time.time() - t_start) / 60.0
        print(f"\n  ❌ {step_name} 异常: {e}  ⏱️ 已耗时: {elapsed_min:.1f} min")
        return False


def find_video_files(directory):
    """在目录中查找所有视频文件（仅当前层级，不递归），按文件名排序。

    Args:
        directory: 目录路径 (str 或 Path)

    Returns:
        list[str]: 视频文件的绝对路径列表
    """
    videos = []
    try:
        for entry in sorted(os.scandir(str(directory)), key=lambda e: e.name.lower()):
            if (entry.is_file(follow_symlinks=False)
                    and Path(entry.name).suffix.lower() in _VIDEO_EXTENSIONS):
                videos.append(entry.path)
    except OSError as e:
        print(f"❌ 无法读取目录: {directory}: {e}")
    return videos


def _process_one_video(args, python_cmd, has_ffmpeg, video_index=None, video_total=None):
    """对单个视频执行完整流水线 (推理 → 高光剪辑)。

    使用 args.video 作为输入视频路径。args.output 若为 None 则自动生成默认路径。
    不会调用 sys.exit，而是返回 bool 表示成功与否。

    Args:
        args: argparse.Namespace (video/output/work_dir/skip_inference 等所有参数)
        python_cmd: Python 命令前缀列表
        has_ffmpeg: ffmpeg 是否可用
        video_index: 批量模式下的当前索引 (1-based)，用于日志显示
        video_total: 批量模式下的视频总数

    Returns:
        bool: 流水线是否成功完成
    """
    video_path = Path(args.video)
    batch_tag = f" [{video_index}/{video_total}]" if video_index else ""

    # === 默认 output ===
    if args.output is None:
        default_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{video_path.stem}")
        output_path = os.path.join(default_dir, f"{video_path.stem}_highlight{video_path.suffix}")
    else:
        output_path = args.output

    # === 参数校验 ===
    if not video_path.exists():
        print(f"❌{batch_tag} 视频不存在: {video_path}")
        return False

    # 确定 raw_detections 路径 (使用局部变量避免修改 args 影响后续视频)
    raw_detections_input = args.raw_detections

    if args.skip_inference:
        if not raw_detections_input:
            # 尝试自动查找 work-dir 下的 JSONL
            auto_jsonl = os.path.join(
                args.work_dir or os.path.join(os.getcwd(), "inference_output", f"pipeline_{video_path.stem}"),
                "raw_detections.jsonl")
            if os.path.exists(auto_jsonl):
                raw_detections_input = auto_jsonl
            else:
                print(f"❌{batch_tag} --skip-inference 需要同时指定 --raw-detections, "
                      f"且未在默认位置找到 ({auto_jsonl})")
                return False

    if raw_detections_input and not os.path.exists(raw_detections_input):
        print(f"❌{batch_tag} 检测日志不存在: {raw_detections_input}")
        return False

    if args.skip_inference and not raw_detections_input:
        print(f"❌{batch_tag} --skip-inference 需要同时指定 --raw-detections")
        return False

    # === 工作目录 ===
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{video_path.stem}")
    os.makedirs(work_dir, exist_ok=True)

    # === 确保输出目录存在 ===
    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # === 打印信息 ===
    print("=" * 60)
    print(f"🏓  PingPong Pipeline{batch_tag}: 推理 → 高光")
    print("=" * 60)
    print(f"  输入视频:    {video_path}")
    print(f"  输出高光:    {output_abs}")
    print(f"  工作目录:    {work_dir}")
    print(f"  Python:      {'conda ' + args.conda_env if args.conda_env else sys.executable}")
    print(f"  ffmpeg:      {'✅ 可用' if has_ffmpeg else '⚠️ 未检测到 (部分功能受限)'}")
    print(f"  推理:        {'⏭️ 跳过' if args.skip_inference else '✅ 执行'}")
    print(f"  高光剪辑:    ✅ 执行")

    # ================================================================
    #  Step 1: YOLO 推理
    # ================================================================
    if args.skip_inference:
        raw_detections = raw_detections_input
        print(f"\n  ⏭️  跳过推理, 使用已有 JSONL: {raw_detections}")
    else:
        # 推理的输出是 work_dir/raw_detections.jsonl
        raw_detections = os.path.join(work_dir, "raw_detections.jsonl")

        infer_cmd = python_cmd + [str(INFERENCE_SCRIPT),
                                   "--video", str(video_path),
                                   "--output", work_dir]

        # 可选推理参数
        if args.model:
            infer_cmd += ["--model", args.model]
        if args.batch_size is not None:
            infer_cmd += ["--batch-size", str(args.batch_size)]
        if args.imgsz is not None:
            infer_cmd += ["--imgsz", str(args.imgsz)]
        if args.device is not None:
            infer_cmd += ["--device", args.device]
        if args.half:
            infer_cmd += ["--half"]
        if args.conf is not None:
            infer_cmd += ["--conf", str(args.conf)]
        if args.out_width is not None:
            infer_cmd += ["--out-width", str(args.out_width)]
        if args.encoder is not None:
            infer_cmd += ["--encoder", args.encoder]
        if args.remove_cfr:
            infer_cmd += ["--remove-cfr"]

        if not run_step(infer_cmd,
                         "Step 1/2: YOLO 推理 (inference_video.py)",
                         note="首次加载模型约需 30-60 秒 (import torch + ultralytics), 请耐心等待..."):
            print(f"\n❌{batch_tag} 流水线终止, 中间文件保留在: {work_dir}")
            return False

        if not os.path.exists(raw_detections):
            print(f"\n❌{batch_tag} 推理完成但未生成检测日志: {raw_detections}")
            return False

    # ================================================================
    #  Step 2: 高光特效剪辑
    # ================================================================
    # 优先使用 CFR 转换后的视频 (时间戳与 JSONL 对齐), 不存在则用原始视频
    cfr_video = os.path.join(work_dir, "video_cfr.mp4")
    if os.path.exists(cfr_video):
        highlight_video = cfr_video
        print(f"\n  ℹ️  高光剪辑使用 CFR 视频: {cfr_video}")
    else:
        highlight_video = str(video_path)

    hl_cmd = python_cmd + [str(HIGHLIGHT_SCRIPT),
                            "--video", highlight_video,
                            "--raw-detections", raw_detections,
                            "--output", output_abs]

    if args.output_height is not None:
        hl_cmd += ["--output-height", str(args.output_height)]
    if args.encoder is not None:
        hl_cmd += ["--encoder", args.encoder]
    if args.no_audio:
        hl_cmd += ["--no-audio"]
    if args.debug:
        hl_cmd += ["--debug"]

    if not run_step(hl_cmd, "Step 2/2: 高光特效剪辑 (pingpong_highlight.py)"):
        print(f"\n❌{batch_tag} 流水线终止, 中间文件保留在: {work_dir}")
        return False

    if not os.path.exists(output_abs):
        print(f"\n❌{batch_tag} 剪辑完成但未生成输出: {output_abs}")
        return False

    # ================================================================
    #  清理
    # ================================================================
    if args.clean and not args.skip_inference:
        for fname in ["pingpong_detection.mp4", "video_cfr.mp4"]:
            fp = os.path.join(work_dir, fname)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    print(f"  🧹 已清理: {fp}")
                except Exception:
                    pass

    # ================================================================
    #  汇总
    # ================================================================
    size_mb = os.path.getsize(output_abs) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"🎉  流水线完成{batch_tag}!")
    print(f"{'=' * 60}")
    print(f"  📄 检测日志: {raw_detections}")
    print(f"  🎬 高光视频: {output_abs} ({size_mb:.1f} MB)")

    if not args.clean:
        print(f"  💡 中间文件保留在: {work_dir}/")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="一键流水线: YOLO 推理 + 高光特效剪辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py --video match.mp4
  python run_pipeline.py --video ./videos/          # 批量处理目录中所有视频
  python run_pipeline.py --video match.mp4 --model best.pt --batch-size 8
  python run_pipeline.py --video match.mp4 --skip-inference --raw-detections log.jsonl
        """,
    )

    # === 必需参数 ===
    parser.add_argument("--video", "-v", required=True,
                        help="输入视频路径, 或包含多个视频的目录路径 (目录模式下串行处理所有视频)")
    parser.add_argument("--output", "-o", default=None,
                        help="输出高光视频路径。单视频模式默认: "
                             "inference_output/pipeline_<视频名>/<视频名>_highlight.<后缀>; "
                             "目录模式默认: 每个视频使用各自的默认路径")
    parser.add_argument("--output-dir", default=None,
                        help="批量模式下输出目录, 每个视频输出为 "
                             "<output-dir>/<视频名>_highlight.<后缀> (仅目录模式生效)")

    # === 推理参数 (传给 inference_video.py) ===
    parser.add_argument("--model", "-m", default=None,
                        help="YOLO 模型路径 (默认使用脚本内置默认值)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="推理批大小 (默认 4)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="推理分辨率 (默认 960)")
    parser.add_argument("--device", default=None,
                        help="推理设备 (默认 0=GPU)")
    parser.add_argument("--half", action="store_true",
                        help="启用 FP16 推理加速")
    parser.add_argument("--conf", type=float, default=None,
                        help="检测置信度阈值 (默认 0.30)")
    parser.add_argument("--out-width", type=int, default=None,
                        help="推理阶段输出的检测视频宽度 (默认 640)")
    parser.add_argument("--remove-cfr", action="store_true",
                        help="VFR→CFR 转换后删除中间文件")

    # === 高光参数 (传给 pingpong_highlight.py) ===
    parser.add_argument("--output-height", type=int, default=None,
                        help="高光视频输出高度 (默认与输入视频一致)")
    parser.add_argument("--encoder", default=None,
                        help="高光视频编码器 (auto/libx264/h264_amf/hevc_amf)")
    parser.add_argument("--no-audio", action="store_true",
                        help="高光视频不保留原声")
    parser.add_argument("--debug", action="store_true",
                        help="输出调试信息 (球台几何/落点坐标等)")

    # === 流水线控制 ===
    parser.add_argument("--work-dir", default=None,
                        help="中间文件目录 (默认: inference_output/pipeline_<视频名>/)")
    parser.add_argument("--clean", action="store_true",
                        help="运行后清理中间检测视频 (默认保留)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="跳过推理步骤, 直接用已有 JSONL 做高光剪辑")
    parser.add_argument("--raw-detections", "-r", default=None,
                        help="已有检测日志路径 (配合 --skip-inference 使用)")
    parser.add_argument("--conda-env", default=None,
                        help="conda 环境名 (默认使用当前 Python 解释器)")

    args = parser.parse_args()

    # === 检查必要脚本 (只需一次) ===
    for script, name in [(INFERENCE_SCRIPT, "inference_video.py"),
                          (HIGHLIGHT_SCRIPT, "pingpong_highlight.py")]:
        if not script.exists():
            print(f"❌ 找不到 {name}: {script}")
            sys.exit(1)

    python_cmd = get_python_cmd(args.conda_env)

    # === ffmpeg 检测 (只需一次) ===
    has_ffmpeg = True
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True,
                       creationflags=_SUBPROC_FLAGS, timeout=5)
    except Exception:
        has_ffmpeg = False

    # === 判断是目录还是文件 ===
    video_input = Path(args.video)

    if video_input.is_dir():
        # ================================================================
        #  批量模式: 扫描目录中所有视频, 串行处理
        # ================================================================
        videos = find_video_files(video_input)
        if not videos:
            print(f"❌ 目录中未找到视频文件 (支持格式: {', '.join(sorted(_VIDEO_EXTENSIONS))})")
            print(f"   目录: {video_input}")
            sys.exit(1)

        print("=" * 70)
        print(f"📂 批量模式: 在目录中找到 {len(videos)} 个视频文件, 将串行处理")
        print("=" * 70)
        for i, vp in enumerate(videos, 1):
            print(f"  [{i}] {Path(vp).name}")
        print()

        # 确定输出目录
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        # 保存原始 raw_detections (用于每个视频重置)
        original_raw_detections = args.raw_detections

        succeeded = 0
        failed_videos = []
        t_batch_start = time.time()

        for i, vp in enumerate(videos, 1):
            video_name = Path(vp).name
            print(f"\n{'#' * 70}")
            print(f"#  [{i}/{len(videos)}] 开始处理: {video_name}")
            print(f"{'#' * 70}")

            # 为当前视频设置参数
            args.video = str(vp)
            args.raw_detections = original_raw_detections  # 重置

            # 计算输出路径
            video_p = Path(vp)
            if args.output_dir:
                args.output = os.path.join(args.output_dir, f"{video_p.stem}_highlight{video_p.suffix}")
            else:
                args.output = None  # 使用默认路径

            try:
                if _process_one_video(args, python_cmd, has_ffmpeg,
                                      video_index=i, video_total=len(videos)):
                    succeeded += 1
                else:
                    failed_videos.append(video_name)
                    print(f"\n  ⚠️  [{i}/{len(videos)}] {video_name} 处理失败, 继续下一个...")
            except KeyboardInterrupt:
                elapsed_min = (time.time() - t_batch_start) / 60.0
                print(f"\n\n⚠️ 用户中断! 已处理 {i - 1}/{len(videos)} 个视频, "
                      f"成功 {succeeded} 个, ⏱️ 已耗时: {elapsed_min:.1f} min")
                sys.exit(1)

        # ================================================================
        #  批量汇总
        # ================================================================
        elapsed_min = (time.time() - t_batch_start) / 60.0
        print(f"\n{'=' * 70}")
        print(f"📊 批量处理完成!")
        print(f"{'=' * 70}")
        print(f"  总计: {len(videos)} 个视频")
        print(f"  成功: {succeeded} 个")
        print(f"  失败: {len(failed_videos)} 个")
        print(f"  总耗时: {elapsed_min:.1f} min")
        if failed_videos:
            print(f"\n  失败视频列表:")
            for name in failed_videos:
                print(f"    ❌ {name}")
        if args.output_dir:
            print(f"\n  输出目录: {args.output_dir}")
        print()

        sys.exit(0 if len(failed_videos) == 0 else 1)

    else:
        # ================================================================
        #  单视频模式 (原始行为)
        # ================================================================
        # 默认 output
        if args.output is None:
            video_path = Path(args.video)
            default_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{video_path.stem}")
            args.output = os.path.join(default_dir, f"{video_path.stem}_highlight{video_path.suffix}")

        if not _process_one_video(args, python_cmd, has_ffmpeg):
            sys.exit(1)


if __name__ == "__main__":
    main()
