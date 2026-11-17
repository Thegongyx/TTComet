#!/usr/bin/env python3
"""
torch/ROCm vs ONNX DirectML 公平对比: 同一模型 (m960n) 同一输入
=============================================================
必须用 yolo26 环境跑 (需要 torch + ultralytics):
    conda run -n yolo26 python webui/bench/bench_torch.py `
        --model models/m960n.pt --batch 8

注意:
  - 这里测的是 model.predict(...) 端到端耗时 (含 ultralytics 的 letterbox/NMS/
    mask 后处理), 与 ONNX 侧"NMS 内嵌在模型里"的 forward 最接近。
  - 不测 CPU 绘图/resize/写 jsonl —— 那些属于管线后处理, 两边相同。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


def pick_frame(video: str | None, idx: int):
    import glob
    if video is None:
        cands = sorted(glob.glob("Videos/*.mp4"))
        if not cands:
            raise SystemExit("找不到测试视频, 用 --video 指定")
        video = cands[0]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读帧失败")
    return frame, video


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/m960n.pt")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batches", default="1,8",
                    help="要测的 batch 大小, 逗号分隔 (默认 1,8)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    frame, video = pick_frame(args.video, args.frame)
    print("=" * 66)
    print("  torch (ROCm) 基准")
    print("=" * 66)
    print(f"  模型    : {args.model}")
    print(f"  视频帧  : {video} 帧 {args.frame}  {frame.shape[1]}x{frame.shape[0]}")

    t0 = time.time()
    model = YOLO(args.model)
    try:
        model.fuse()
    except Exception:
        pass
    print(f"  模型加载: {time.time() - t0:.1f}s")

    kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=0.45, verbose=False,
                  device="0", max_det=300)

    for bs in [int(b) for b in args.batches.split(",") if b.strip()]:
        frames = [frame.copy() for _ in range(bs)]
        for _ in range(args.warmup):
            model.predict(frames, **kwargs)
        lat = []
        for _ in range(args.iters):
            t = time.perf_counter()
            res = model.predict(frames, **kwargs)
            lat.append((time.perf_counter() - t) * 1000)
        mean = statistics.fmean(lat)
        print(f"\n  --- batch={bs} ---")
        print(f"  批次耗时: {mean:8.2f} ms  (p50 {statistics.median(lat):.2f})")
        print(f"  单帧    : {mean / bs:8.2f} ms/帧")
        print(f"  吞吐    : {1000 * bs / mean:8.1f} fps")

        # 打印第 0 帧的检出, 便于与 ONNX 侧人工比对
        if bs == 1:
            r = res[0]
            boxes = r.boxes
            names = model.names
            print(f"  检出    : {0 if boxes is None else len(boxes)} 个")
            if boxes is not None and len(boxes):
                order = np.argsort(-boxes.conf.cpu().numpy())[:10]
                for i in order:
                    c = int(boxes.cls[i].item())
                    s = float(boxes.conf[i].item())
                    b = boxes.xyxy[i].cpu().numpy()
                    print(f"    {names[c]:<6s} conf={s:.4f} "
                          f"xyxy=[{b[0]:7.1f},{b[1]:7.1f},{b[2]:7.1f},{b[3]:7.1f}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
