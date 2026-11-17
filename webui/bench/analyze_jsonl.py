#!/usr/bin/env python3
"""
raw_detections.jsonl 体积/结构分析
==================================
统计: 行数、单行大小、各字段占比、字段裁剪后的体积、gzip 压缩比。
用于评估 WebUI 场景下的磁盘与内存成本, 以及可能的瘦身手段。

用法:
    webui\.venv\Scripts\python.exe webui\bench\analyze_jsonl.py `
        inference_output/pipeline_xxx/raw_detections.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys

# Windows 控制台默认 GBK, 中文/emoji 会抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--sample", type=int, default=3000,
                    help="抽多少行做字段分析/压缩估算 (默认 3000)")
    args = ap.parse_args()

    path = args.path
    size = os.path.getsize(path)

    # ---- 1. 行数 + 字段构成 (流式, 不占内存) ----
    n_lines = 0
    sample_raw: list[bytes] = []
    key_bytes = {"detections": 0, "polygon": 0, "counts": 0, "ball_pos": 0}
    det_count = 0
    with open(path, "rb") as f:
        for raw in f:
            n_lines += 1
            if len(sample_raw) < args.sample:
                sample_raw.append(raw)

    for raw in sample_raw:
        obj = json.loads(raw)
        key_bytes["detections"] += len(json.dumps(
            obj.get("detections"), ensure_ascii=False).encode())
        key_bytes["counts"] += len(json.dumps(
            obj.get("counts"), ensure_ascii=False).encode())
        key_bytes["ball_pos"] += len(json.dumps(
            obj.get("ball_pos"), ensure_ascii=False).encode())
        for d in obj.get("detections") or []:
            det_count += 1
            if d.get("polygon"):
                key_bytes["polygon"] += len(json.dumps(
                    d["polygon"], ensure_ascii=False).encode())

    sample_bytes = sum(len(r) for r in sample_raw)
    avg_line = sample_bytes / max(1, len(sample_raw))

    print("=" * 66)
    print("  raw_detections.jsonl 分析")
    print("=" * 66)
    print(f"  文件        : {path}")
    print(f"  大小        : {size / 1024 / 1024:.1f} MB")
    print(f"  行数        : {n_lines:,}")
    print(f"  平均行大小  : {avg_line:.0f} B")
    print(f"  平均检出数  : {det_count / max(1, len(sample_raw)):.2f} 个/帧")

    if sample_bytes:
        print("\n  --- 字段占比 (按抽样) ---")
        det_total = key_bytes["detections"]
        poly = key_bytes["polygon"]
        print(f"    {'detections':12s}: {det_total / sample_bytes * 100:5.1f}%  "
              f"(全文件约 {det_total / sample_bytes * size / 1024 / 1024:6.1f} MB)")
        print(f"      ├ polygon  : {poly / sample_bytes * 100:5.1f}%  "
              f"(全文件约 {poly / sample_bytes * size / 1024 / 1024:6.1f} MB, "
              f"占 detections 的 {poly / max(1, det_total) * 100:.0f}%)")
        print(f"      └ 其余字段 : {(det_total - poly) / sample_bytes * 100:5.1f}%")
        for k in ("counts", "ball_pos"):
            v = key_bytes[k]
            print(f"    {k:12s}: {v / sample_bytes * 100:5.1f}%  "
                  f"(全文件约 {v / sample_bytes * size / 1024 / 1024:6.1f} MB)")
        rest = sample_bytes - det_total - key_bytes["counts"] - key_bytes["ball_pos"]
        print(f"    {'其它(键名等)':12s}: {rest / sample_bytes * 100:5.1f}%")

    # ---- 2. 瘦身后的估算 ----
    print("\n  --- 瘦身估算 (全文件) ---")
    slim_raw = 0
    for raw in sample_raw:
        obj = json.loads(raw)
        slim = {
            "frame": obj.get("frame"),
            "detections": [
                {"c": d.get("class"), "s": d.get("confidence"),
                 "b": [round(v, 1) for v in d.get("bbox") or []],
                 "p": d.get("polygon")}
                for d in (obj.get("detections") or [])
            ],
        }
        slim_raw += len(json.dumps(slim, ensure_ascii=False,
                                   separators=(",", ":")).encode())
    print(f"    只留 frame+detections(短键名, 去 counts/ball_pos): "
          f"约 {slim_raw / sample_bytes * size / 1024 / 1024:.1f} MB "
          f"({slim_raw / sample_bytes * 100:.0f}% 原体积)")

    no_poly_raw = 0
    for raw in sample_raw:
        obj = json.loads(raw)
        no_poly = {
            "frame": obj.get("frame"),
            "detections": [
                {"c": d.get("class"), "s": d.get("confidence"),
                 "b": [round(v, 1) for v in d.get("bbox") or []]}
                for d in (obj.get("detections") or [])
            ],
        }
        no_poly_raw += len(json.dumps(no_poly, ensure_ascii=False,
                                      separators=(",", ":")).encode())
    print(f"    再去掉 polygon: 约 {no_poly_raw / sample_bytes * size / 1024 / 1024:.1f} MB "
          f"({no_poly_raw / sample_bytes * 100:.0f}% 原体积)")
    print("      ⚠️ 注意: polygon 是高光提取球台/球网几何的唯一来源, 不能去掉")

    # ---- 3. gzip 压缩比 ----
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        for raw in sample_raw:
            gz.write(raw)
    ratio = len(buf.getvalue()) / sample_bytes
    print("\n  --- gzip 压缩 (level 6, 抽样) ---")
    print(f"    压缩比      : {ratio * 100:.1f}%  → 全文件约 "
          f"{ratio * size / 1024 / 1024:.1f} MB")

    # ---- 4. 内存膨胀估算 ----
    import tracemalloc
    tracemalloc.start()
    objs = [json.loads(r) for r in sample_raw]
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del objs
    print("\n  --- 全量读入内存 (pingpong_highlight.load_detections 的做法) ---")
    print(f"    Python 对象膨胀: {cur / sample_bytes:.1f}x 文本体积  → 预估峰值 "
          f"{cur / sample_bytes * size / 1024 / 1024 / 1024:.1f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
