#!/usr/bin/env python3
"""
两份 raw_detections 对拍
========================
用同一批帧比较 torch 版（src/inference_video.py）与 ONNX 版的检出差异：
  - 每类检出总数
  - 同帧同类的 bbox 匹配率与平均 IoU（贪心匹配, IoU>0.5 视为同一目标）
  - 球的逐帧命中率
  - polygon 存在率与平均点数

用法:
    webui\.venv\Scripts\python.exe webui\bench\compare_jsonl.py `
        inference_output/pipeline_xxx/raw_detections.jsonl `
        webui/out/test600/raw_detections.jsonl.gz --frames 600
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def opener(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load(path: str, n: int) -> dict[int, dict]:
    out = {}
    with opener(path) as f:
        for line in f:
            if len(out) >= n:
                break
            o = json.loads(line)
            out[int(o["frame"])] = o
    return out


def iou(a, b) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="基准 jsonl (torch 版)")
    ap.add_argument("b", help="对比 jsonl (ONNX 版, 可 .gz)")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    A = load(args.a, args.frames)
    B = load(args.b, args.frames)
    common = sorted(set(A) & set(B))
    print("=" * 66)
    print("  两份 raw_detections 对拍")
    print("=" * 66)
    print(f"  A (基准): {args.a}")
    print(f"  B (ONNX): {args.b}")
    print(f"  对比帧数: {len(common)}  (A {len(A)} / B {len(B)})")
    if not common:
        print("❌ 没有共同帧")
        return 1

    # ---- 每类总数 ----
    ca, cb = collections.Counter(), collections.Counter()
    for i in common:
        for d in A[i]["detections"]:
            ca[d["class"]] += 1
        for d in B[i]["detections"]:
            cb[d["class"]] += 1
    print("\n  --- 每类检出总数 ---")
    print(f"    {'类别':<14s}{'A(torch)':>10s}{'B(ONNX)':>10s}{'差异':>10s}")
    for k in sorted(set(ca) | set(cb)):
        print(f"    {k:<14s}{ca[k]:>10d}{cb[k]:>10d}{cb[k] - ca[k]:>+10d}")

    # ---- bbox 匹配 ----
    matched_total = 0
    ious: list[float] = []
    per_cls_match = collections.Counter()
    per_cls_total = collections.Counter()
    for i in common:
        da = A[i]["detections"]
        db = B[i]["detections"]
        used = set()
        for x in da:
            per_cls_total[x["class"]] += 1
            best_j, best_iou = -1, 0.0
            for j, y in enumerate(db):
                if j in used or y["class"] != x["class"]:
                    continue
                v = iou(x["bbox"], y["bbox"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= args.iou_thr:
                used.add(best_j)
                matched_total += 1
                ious.append(best_iou)
                per_cls_match[x["class"]] += 1

    print("\n  --- 同帧同类 bbox 匹配 (IoU>={:.2f}) ---".format(args.iou_thr))
    tot_a = sum(per_cls_total.values())
    print(f"    总体: {matched_total}/{tot_a} = {matched_total / max(tot_a, 1) * 100:.1f}%"
          f"   平均 IoU {statistics.fmean(ious) if ious else 0:.3f}")
    for k in sorted(per_cls_total):
        print(f"    {k:<14s}{per_cls_match[k]:>8d}/{per_cls_total[k]:<8d}"
              f"{per_cls_match[k] / max(per_cls_total[k], 1) * 100:6.1f}%")

    # ---- 球命中帧率 ----
    def ball_frames(D):
        n = 0
        for i in common:
            if any(d["class"] == "pingpangball" for d in D[i]["detections"]):
                n += 1
        return n
    ba, bb = ball_frames(A), ball_frames(B)
    print("\n  --- 球命中帧 ---")
    print(f"    A: {ba}/{len(common)} = {ba / len(common) * 100:.1f}%")
    print(f"    B: {bb}/{len(common)} = {bb / len(common) * 100:.1f}%")

    # ---- polygon ----
    def poly_stats(D):
        n_poly, pts = 0, []
        for i in common:
            for d in D[i]["detections"]:
                if d.get("polygon"):
                    n_poly += 1
                    pts.append(len(d["polygon"]))
        return n_poly, (statistics.fmean(pts) if pts else 0)
    pa, sa = poly_stats(A)
    pb, sb = poly_stats(B)
    print("\n  --- polygon ---")
    print(f"    A: {pa} 个, 平均 {sa:.0f} 点")
    print(f"    B: {pb} 个, 平均 {sb:.0f} 点")

    return 0


if __name__ == "__main__":
    sys.exit(main())
