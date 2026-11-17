#!/usr/bin/env python3
"""
ONNX 推理 CLI（视频 -> raw_detections.jsonl.gz）
=================================================
薄壳：真正的实现在 `inference_job.infer_video()`，WebUI 调的是同一个函数。

用法:
    webui\.venv\Scripts\python.exe webui\app\run_video.py `
        --video Videos/xxx.mp4 --max-frames 600 --output webui/out/test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_job  # noqa: E402
from onnx_engine import DetectorConfig, OnnxDetector  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ONNX 视频推理 -> raw_detections.jsonl.gz")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--model", "-m", default="webui/models/m960n_fp16_nonms.onnx")
    p.add_argument("--output", "-o", default=None,
                   help="输出目录（默认 webui/out/<视频名>/）")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="0=全部")
    p.add_argument("--ep", default="auto", choices=["auto", "dml", "cpu"])
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--conf-ball", type=float, default=0.10)
    p.add_argument("--conf-player", type=float, default=None)
    p.add_argument("--conf-racket", type=float, default=None)
    p.add_argument("--conf-table", type=float, default=None)
    p.add_argument("--conf-net", type=float, default=None)
    p.add_argument("--no-polygon", action="store_true")
    p.add_argument("--no-cfr", action="store_true",
                   help="跳过 VFR->CFR 转换（默认自动转换，高光阶段依赖 video_cfr.mp4）")
    p.add_argument("--cfr-highlights", type=float, default=10.0)
    p.add_argument("--cfr-contrast", type=float, default=10.0)
    p.add_argument("--cfr-saturation", type=float, default=6.0)
    p.add_argument("--serial", action="store_true", help="（保留兼容，已无意义）")
    p.add_argument("--queue-size", type=int, default=2)
    p.add_argument("--progress-every", type=float, default=2.0)
    return p


def main() -> int:
    args = build_parser().parse_args()

    cfg = DetectorConfig(
        iou=args.iou, conf_ball=args.conf_ball,
        conf_player=args.conf_player, conf_racket=args.conf_racket,
        conf_table=args.conf_table, conf_net=args.conf_net,
        want_polygon=not args.no_polygon, prefer_ep=args.ep,
    )
    det = OnnxDetector(args.model, cfg)
    print("=" * 66)
    print("  ONNX 视频推理")
    print("=" * 66)
    print(f"  模型 : {args.model}")
    print(f"  后端 : {det.providers}  ({det.load_ms:.0f} ms)")
    print(f"  输入 : {det.imgsz}×{det.imgsz} {np.dtype(det.input_dtype).name}")
    print(f"  类别 : {det.raw_names}\n")

    out_dir = args.output or os.path.join(
        "webui", "out", os.path.splitext(os.path.basename(args.video))[0])

    def on_progress(p: dict):
        if p.get("stage") == "inference" and "frames" in p:
            print(f"  [{p['frames']:6d}/{p['total']}] {p['fps']:5.1f} fps | "
                  f"ETA {p['eta_sec']:5.0f}s | pre {p['pre_ms']:.1f} "
                  f"infer {p['infer_ms']:.1f} post {p['post_ms']:.1f} ms")

    result = inference_job.infer_video(
        args.video, out_dir, cfg, model=args.model,
        max_frames=args.max_frames, start_frame=args.start_frame,
        cfr=not args.no_cfr,
        cfr_highlights=args.cfr_highlights,
        cfr_contrast=args.cfr_contrast,
        cfr_saturation=args.cfr_saturation,
        queue_size=args.queue_size, progress_every=args.progress_every,
        log=lambda m: print("  " + m),
        progress=on_progress, detector=det)

    print("\n" + "=" * 66)
    print("  结果")
    print("=" * 66)
    print(f"  处理帧数 : {result['frames']}")
    print(f"  墙钟耗时 : {result['wall_sec']}s -> {result['fps']} fps "
          f"(源 {result['source_fps']} fps)")
    print(f"  输出体积 : {result['size_mb']} MB")
    print(f"  输出文件 : {result['jsonl']}")
    if result["cfr_video"]:
        print(f"  CFR 视频 : {result['cfr_video']}")
    print("\n  --- 各阶段平均耗时 (ms/帧) ---")
    st = result["stages_ms"]
    for k in ("decode", "pre", "infer", "post", "write"):
        print(f"    {k:8s}: {st.get(k, 0):6.2f}")
    print(f"    串行合计 : {sum(st.values()):6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
