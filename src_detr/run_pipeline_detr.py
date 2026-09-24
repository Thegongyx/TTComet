#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline_detr.py — 一键流水线: RF-DETR 推理 → 高光特效剪辑
============================================================
将 src_detr/inference_video_detr.py (RF-DETR 推理) 与
src/pingpong_highlight.py (高光特效, 与 YOLO 版完全共用) 串联为一条命令。

由于 inference_video_detr.py 输出的 raw_detections.jsonl 与 YOLO 版格式一致,
高光剪辑脚本无需任何改动即可使用。

默认输出:
    ./inference_output/pipeline_<视频名>/
    ├── <视频名>_highlight.mp4    ← 最终高光视频
    └── raw_detections.jsonl      ← RF-DETR 检测日志

用法:
    conda run -n rfdetr python src_detr/run_pipeline_detr.py `
        --model runs_detr/detr_ft_v1/checkpoint_best_total.pth `
        --video Videos/match.mp4

    # 批量目录
    conda run -n rfdetr python src_detr/run_pipeline_detr.py `
        --model <ckpt.pth> --video Videos/

    # 已有 JSONL, 跳过推理直接高光
    conda run -n rfdetr python src_detr/run_pipeline_detr.py `
        --video match.mp4 --skip-inference --raw-detections <log.jsonl>
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

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_SCRIPT = Path(__file__).resolve().parent / "inference_video_detr.py"
HIGHLIGHT_SCRIPT = _SRC_DIR / "pingpong_highlight.py"


def _default_model():
    """默认模型: 优先项目根目录的 rf-detr-seg-*.pt。"""
    for name in ("rf-detr-seg-medium.pt", "rf-detr-seg-small.pt", "rf-detr-seg-nano.pt"):
        p = _PROJECT_ROOT / name
        if p.exists():
            return str(p)
    return "rf-detr-seg-medium.pt"

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
                     ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp"}


def get_python_cmd(conda_env=None):
    if conda_env:
        return ["conda", "run", "--live-stream", "-n", conda_env, "python", "-u"]
    return [sys.executable, "-u"]


def run_step(cmd, step_name, note=None):
    print(f"\n{'=' * 60}\n  {step_name}\n{'=' * 60}")
    print(f"  → {' '.join(cmd)}")
    if note:
        print(f"  ⏳ {note}")
    print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1, encoding="utf-8",
                                errors="replace", creationflags=_SUBPROC_FLAGS)
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        mins = (time.time() - t0) / 60.0
        if proc.returncode != 0:
            print(f"\n  ❌ {step_name} 失败 (exit code {proc.returncode})")
            return False
        print(f"\n  ✅ {step_name} 完成  ⏱️ 耗时: {mins:.1f} min")
        return True
    except KeyboardInterrupt:
        if proc:
            proc.kill()
        mins = (time.time() - t0) / 60.0
        print(f"\n  ⚠️ 用户中断  ⏱️ 已耗时: {mins:.1f} min")
        raise
    except Exception as e:
        if proc:
            proc.kill()
        mins = (time.time() - t0) / 60.0
        print(f"\n  ❌ {step_name} 异常: {e}  ⏱️ 已耗时: {mins:.1f} min")
        return False


def find_video_files(directory):
    vids = []
    try:
        for entry in sorted(os.scandir(str(directory)), key=lambda e: e.name.lower()):
            if entry.is_file(follow_symlinks=False) and \
                    Path(entry.name).suffix.lower() in _VIDEO_EXTENSIONS:
                vids.append(entry.path)
    except OSError as e:
        print(f"❌ 无法读取目录: {directory}: {e}")
    return vids


def _build_inference_cmd(args, python_cmd, video_path, work_dir):
    cmd = python_cmd + [str(INFERENCE_SCRIPT),
                        "--video", str(video_path),
                        "--output", work_dir,
                        "--model", args.model]
    if args.size:
        cmd += ["--size", args.size]
    if args.task:
        cmd += ["--task", args.task]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.imgsz is not None:
        cmd += ["--imgsz", str(args.imgsz)]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.half:
        cmd += ["--half"]
    if args.conf is not None:
        cmd += ["--conf", str(args.conf)]
    if args.iou is not None:
        cmd += ["--iou", str(args.iou)]
    if args.conf_ball is not None:
        cmd += ["--conf-ball", str(args.conf_ball)]
    if args.conf_player is not None:
        cmd += ["--conf-player", str(args.conf_player)]
    if args.conf_racket is not None:
        cmd += ["--conf-racket", str(args.conf_racket)]
    if args.conf_table is not None:
        cmd += ["--conf-table", str(args.conf_table)]
    if args.conf_net is not None:
        cmd += ["--conf-net", str(args.conf_net)]
    if args.conf_margin is not None:
        cmd += ["--conf-margin", str(args.conf_margin)]
    if args.out_width is not None:
        cmd += ["--out-width", str(args.out_width)]
    if args.encoder is not None:
        cmd += ["--encoder", args.encoder]
    if args.poly_approx is not None:
        cmd += ["--poly-approx", str(args.poly_approx)]
    if args.log_flush_interval is not None:
        cmd += ["--log-flush-interval", str(args.log_flush_interval)]
    if args.empty_cache_every is not None:
        cmd += ["--empty-cache-every", str(args.empty_cache_every)]
    if args.no_polygon:
        cmd += ["--no-polygon"]
    if args.remove_cfr:
        cmd += ["--remove-cfr"]
    if args.writer_backend is not None:
        cmd += ["--writer-backend", args.writer_backend]
    if args.amf_quality is not None:
        cmd += ["--amf-quality", args.amf_quality]
    if args.amf_qp is not None:
        cmd += ["--amf-qp", str(args.amf_qp)]
    return cmd


def _process_one_video(args, python_cmd, has_ffmpeg, video_index=None, video_total=None):
    video_path = Path(args.video)
    batch_tag = f" [{video_index}/{video_total}]" if video_index else ""

    if args.output is None:
        default_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{video_path.stem}")
        output_path = os.path.join(default_dir, f"{video_path.stem}_highlight{video_path.suffix}")
    else:
        output_path = args.output

    if not video_path.exists():
        print(f"❌{batch_tag} 视频不存在: {video_path}")
        return False

    raw_input = args.raw_detections
    if args.skip_inference and not raw_input:
        auto_jsonl = os.path.join(
            args.work_dir or os.path.join(os.getcwd(), "inference_output",
                                          f"pipeline_{video_path.stem}"),
            "raw_detections.jsonl")
        if os.path.exists(auto_jsonl):
            raw_input = auto_jsonl
        else:
            print(f"❌{batch_tag} --skip-inference 需要 --raw-detections, 且未在 {auto_jsonl} 找到")
            return False
    if raw_input and not os.path.exists(raw_input):
        print(f"❌{batch_tag} 检测日志不存在: {raw_input}")
        return False

    work_dir = args.work_dir or os.path.join(
        os.getcwd(), "inference_output", f"pipeline_{video_path.stem}")
    os.makedirs(work_dir, exist_ok=True)

    output_abs = os.path.abspath(output_path)
    out_dir = os.path.dirname(output_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"🏓  PingPong Pipeline (RF-DETR){batch_tag}: 推理 → 高光")
    print("=" * 60)
    print(f"  输入视频:    {video_path}")
    print(f"  输出高光:    {output_abs}")
    print(f"  工作目录:    {work_dir}")
    print(f"  模型:        {args.model}")
    print(f"  Python:      {'conda ' + args.conda_env if args.conda_env else sys.executable}")
    print(f"  ffmpeg:      {'✅ 可用' if has_ffmpeg else '⚠️ 未检测到 (部分功能受限)'}")
    print(f"  推理:        {'⏭️ 跳过' if args.skip_inference else '✅ 执行'}")
    print(f"  高光剪辑:    ✅ 执行")

    # Step 1: 推理
    if args.skip_inference:
        raw_detections = raw_input
        print(f"\n  ⏭️  跳过推理, 使用已有 JSONL: {raw_detections}")
    else:
        raw_detections = os.path.join(work_dir, "raw_detections.jsonl")
        infer_cmd = _build_inference_cmd(args, python_cmd, video_path, work_dir)
        if not run_step(infer_cmd, "Step 1/2: RF-DETR 推理 (inference_video_detr.py)",
                        note="首次加载模型需要一些时间, 请耐心等待..."):
            print(f"\n❌{batch_tag} 流水线终止, 中间文件保留在: {work_dir}")
            return False
        if not os.path.exists(raw_detections):
            print(f"\n❌{batch_tag} 推理完成但未生成检测日志: {raw_detections}")
            return False

    # Step 2: 高光
    cfr_video = os.path.join(work_dir, "video_cfr.mp4")
    highlight_video = cfr_video if os.path.exists(cfr_video) else str(video_path)
    if os.path.exists(cfr_video):
        print(f"\n  ℹ️ 高光使用 CFR 视频: {cfr_video}")

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
    if args.min_rally_frames is not None:
        hl_cmd += ["--min-rally-frames", str(args.min_rally_frames)]
    if args.min_static_ball_count is not None:
        hl_cmd += ["--min-static-ball-count", str(args.min_static_ball_count)]
    if args.pre_buffer is not None:
        hl_cmd += ["--pre-buffer", str(args.pre_buffer)]
    if args.post_buffer is not None:
        hl_cmd += ["--post-buffer", str(args.post_buffer)]
    if args.merge_gap is not None:
        hl_cmd += ["--merge-gap", str(args.merge_gap)]
    if args.corridor_margin is not None:
        hl_cmd += ["--corridor-margin", str(args.corridor_margin)]
    if args.corridor_extend is not None:
        hl_cmd += ["--corridor-extend", str(args.corridor_extend)]
    if args.max_jump_ratio is not None:
        hl_cmd += ["--max-jump-ratio", str(args.max_jump_ratio)]

    if not run_step(hl_cmd, "Step 2/2: 高光特效剪辑 (pingpong_highlight.py)"):
        print(f"\n❌{batch_tag} 流水线终止, 中间文件保留在: {work_dir}")
        return False
    if not os.path.exists(output_abs):
        print(f"\n❌{batch_tag} 剪辑完成但未生成输出: {output_abs}")
        return False

    if args.clean and not args.skip_inference:
        for fname in ("pingpong_detection.mp4", "video_cfr.mp4"):
            fp = os.path.join(work_dir, fname)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    print(f"  🧹 已清理: {fp}")
                except Exception:
                    pass

    mb = os.path.getsize(output_abs) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"🎉  流水线完成{batch_tag}!")
    print(f"{'=' * 60}")
    print(f"  📄 检测日志: {raw_detections}")
    print(f"  🎬 高光视频: {output_abs} ({mb:.1f} MB)")
    if not args.clean:
        print(f"  💡 中间文件保留在: {work_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="一键流水线: RF-DETR 推理 + 高光特效剪辑",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  conda run -n rfdetr python src_detr/run_pipeline_detr.py --model <ckpt.pth> --video match.mp4
  conda run -n rfdetr python src_detr/run_pipeline_detr.py --model <ckpt.pth> --video Videos/
  conda run -n rfdetr python src_detr/run_pipeline_detr.py --video match.mp4 --skip-inference --raw-detections log.jsonl
        """,
    )
    parser.add_argument("--video", "-v", required=True,
                        help="输入视频路径, 或包含多个视频的目录")
    parser.add_argument("--model", "-m", default=_default_model(),
                        help="模型路径: 训练 checkpoint(.pth) 或官方预训练权重(.pt); "
                             "默认=项目根目录 rf-detr-seg-medium.pt")
    parser.add_argument("--output", "-o", default=None, help="输出高光视频路径")
    parser.add_argument("--output-dir", default=None, help="批量模式输出目录")

    # 推理参数
    parser.add_argument("--size", default=None,
                        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"])
    parser.add_argument("--task", default=None, choices=["seg", "detect"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--conf-ball", type=float, default=None)
    parser.add_argument("--conf-player", type=float, default=None)
    parser.add_argument("--conf-racket", type=float, default=None)
    parser.add_argument("--conf-table", type=float, default=None)
    parser.add_argument("--conf-net", type=float, default=None)
    parser.add_argument("--conf-margin", type=float, default=None)
    parser.add_argument("--out-width", type=int, default=None)
    parser.add_argument("--poly-approx", type=float, default=None)
    parser.add_argument("--no-polygon", action="store_true")
    parser.add_argument("--remove-cfr", action="store_true")
    parser.add_argument("--writer-backend", default=None, choices=["ffmpeg", "opencv"])
    parser.add_argument("--amf-quality", default=None, choices=["speed", "balanced", "quality"])
    parser.add_argument("--amf-qp", type=int, default=None)
    parser.add_argument("--log-flush-interval", type=float, default=None)
    parser.add_argument("--empty-cache-every", type=int, default=None)

    # 高光参数
    parser.add_argument("--output-height", type=int, default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--min-rally-frames", type=int, default=None)
    parser.add_argument("--min-static-ball-count", type=int, default=None)
    parser.add_argument("--pre-buffer", type=float, default=None)
    parser.add_argument("--post-buffer", type=float, default=None)
    parser.add_argument("--merge-gap", type=float, default=None)
    parser.add_argument("--corridor-margin", type=float, default=None)
    parser.add_argument("--corridor-extend", type=float, default=None)
    parser.add_argument("--max-jump-ratio", type=float, default=None)

    # 流水线控制
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--raw-detections", "-r", default=None)
    parser.add_argument("--conda-env", default="rfdetr",
                        help="运行子脚本的 conda 环境 (默认 rfdetr; 传 '' 用当前解释器)")
    args = parser.parse_args()

    for script, name in ((INFERENCE_SCRIPT, "inference_video_detr.py"),
                         (HIGHLIGHT_SCRIPT, "pingpong_highlight.py")):
        if not script.exists():
            print(f"❌ 找不到 {name}: {script}")
            sys.exit(1)

    python_cmd = get_python_cmd(args.conda_env or None)

    has_ffmpeg = True
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True,
                       creationflags=_SUBPROC_FLAGS, timeout=5)
    except Exception:
        has_ffmpeg = False

    video_input = Path(args.video)
    if video_input.is_dir():
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

        os.makedirs("tmp", exist_ok=True)
        with open("tmp/video_list.txt", "w", encoding="utf-8") as f:
            for vp in videos:
                f.write(vp + "\n")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        original_raw = args.raw_detections
        batch_infer_cmd = python_cmd + [str(INFERENCE_SCRIPT),
                                        "--video-list-file", "tmp/video_list.txt",
                                        "--output", "inference_output",
                                        "--model", args.model]
        if args.size:
            batch_infer_cmd += ["--size", args.size]
        if args.task:
            batch_infer_cmd += ["--task", args.task]
        if args.batch_size is not None:
            batch_infer_cmd += ["--batch-size", str(args.batch_size)]
        if args.imgsz is not None:
            batch_infer_cmd += ["--imgsz", str(args.imgsz)]
        if args.device is not None:
            batch_infer_cmd += ["--device", str(args.device)]
        if args.half:
            batch_infer_cmd += ["--half"]
        if args.conf is not None:
            batch_infer_cmd += ["--conf", str(args.conf)]
        if args.iou is not None:
            batch_infer_cmd += ["--iou", str(args.iou)]
        if args.out_width is not None:
            batch_infer_cmd += ["--out-width", str(args.out_width)]
        if args.encoder is not None:
            batch_infer_cmd += ["--encoder", args.encoder]
        if args.log_flush_interval is not None:
            batch_infer_cmd += ["--log-flush-interval", str(args.log_flush_interval)]
        if args.empty_cache_every is not None:
            batch_infer_cmd += ["--empty-cache-every", str(args.empty_cache_every)]
        if args.no_polygon:
            batch_infer_cmd += ["--no-polygon"]
        if args.remove_cfr:
            batch_infer_cmd += ["--remove-cfr"]

        if args.skip_inference:
            infer_args_present = any([
                args.model, args.batch_size is not None, args.imgsz is not None,
                args.device, args.half, args.conf is not None, args.out_width is not None,
                args.remove_cfr, args.iou is not None, args.log_flush_interval is not None,
                args.writer_backend, args.no_polygon, args.empty_cache_every is not None,
                args.conf_margin is not None, args.amf_quality, args.amf_qp is not None,
            ])
            if infer_args_present:
                print("  ℹ️  --skip-inference 已指定, 推理参数将被忽略")
            print(f"\n  ⏭️  跳过批量推理 (--skip-inference), 直接对已有 JSONL 的视频做高光剪辑")
        else:
            if not run_step(batch_infer_cmd,
                            "Step 1/2: RF-DETR 推理 (批处理 via --video-list-file)",
                            note=f"模型仅加载一次, 共 {len(videos)} 个视频..."):
                print("⚠️ 批量推理部分失败, 将基于已有 jsonl 继续处理...")

        # 批量推理输出在 inference_output/pipeline_<stem> (与单视频命名一致)
        succeeded = 0
        failed = []
        t_batch = time.time()
        for i, vp in enumerate(videos, 1):
            print(f"\n{'#' * 70}")
            print(f"#  [{i}/{len(videos)}] 高光处理: {Path(vp).name}")
            print(f"{'#' * 70}")
            args.video = str(vp)
            args.raw_detections = original_raw
            vp_p = Path(vp)
            if args.output_dir:
                args.output = os.path.join(args.output_dir,
                                           f"{vp_p.stem}_highlight{vp_p.suffix}")
            else:
                args.output = None
            args.skip_inference = True

            stem = vp_p.stem
            work_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{stem}")
            raw_path = os.path.join(work_dir, "raw_detections.jsonl")
            if os.path.exists(raw_path):
                try:
                    if _process_one_video(args, python_cmd, has_ffmpeg,
                                          video_index=i, video_total=len(videos)):
                        succeeded += 1
                    else:
                        failed.append(vp_p.name)
                except KeyboardInterrupt:
                    mins = (time.time() - t_batch) / 60.0
                    print(f"\n\n⚠️ 用户中断! 已处理 {i - 1}/{len(videos)} 个视频, "
                          f"成功 {succeeded} 个, ⏱️ 已耗时: {mins:.1f} min")
                    sys.exit(1)
            else:
                print(f"⚠️ 跳过高光: {raw_path} 不存在")
                failed.append(vp_p.name)

        elapsed_min = (time.time() - t_batch) / 60.0
        print(f"\n{'=' * 70}")
        print(f"📊 批量处理完成!")
        print(f"{'=' * 70}")
        print(f"  总计: {len(videos)} 个视频")
        print(f"  成功: {succeeded} 个")
        print(f"  失败: {len(failed)} 个")
        print(f"  总耗时: {elapsed_min:.1f} min")
        if failed:
            print(f"\n  失败视频列表:")
            for name in failed:
                print(f"    ❌ {name}")
        if args.output_dir:
            print(f"\n  输出目录: {args.output_dir}")
        print()
        sys.exit(0 if len(failed) == 0 else 1)
    else:
        if args.output is None:
            vp = Path(args.video)
            default_dir = os.path.join(os.getcwd(), "inference_output", f"pipeline_{vp.stem}")
            args.output = os.path.join(default_dir, f"{vp.stem}_highlight{vp.suffix}")
        if not _process_one_video(args, python_cmd, has_ffmpeg):
            sys.exit(1)


if __name__ == "__main__":
    main()
