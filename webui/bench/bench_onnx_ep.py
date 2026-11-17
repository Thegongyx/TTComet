#!/usr/bin/env python3
"""
ONNX Runtime 执行后端可用性与速度基准 (DirectML vs CPU)
=====================================================
只做一件事: 验证在 Windows + AMD(ROCm 机器) 上, onnxruntime-directml 能否
真正用 D3D12 跑 YOLO 分割模型, 以及相对 CPU 的加速比。

不改动项目任何现有文件; 依赖装在 webui/.venv (独立 venv, 不含 torch)。

用法:
    webui\.venv\Scripts\python.exe webui\bench\bench_onnx_ep.py `
        --model models/n960n.onnx

输出:
    - 各 EP 的会话初始化耗时 / 首次推理(编译)耗时 / 稳态 ms/帧
    - DML 与 CPU 输出的数值差异 (确认 DML 没有算错)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort


def make_session(model_path: str, ep: str, threads: int | None = None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = threads
    return ort.InferenceSession(model_path, so, providers=[ep])


def bench(sess, input_name: str, shape, iters: int, warmup: int, dtype=np.float32):
    x = np.random.rand(*shape).astype(dtype)

    t0 = time.perf_counter()
    outs = sess.run(None, {input_name: x})
    first_ms = (time.perf_counter() - t0) * 1000

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        outs = sess.run(None, {input_name: x})
        lat.append((time.perf_counter() - t) * 1000)

    return {
        "first_ms": first_ms,
        "mean_ms": statistics.fmean(lat),
        "p50_ms": statistics.median(lat),
        "min_ms": min(lat),
        "max_ms": max(lat),
        "outputs": outs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/n960n.onnx")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print("=" * 66)
    print("  ONNX Runtime 执行后端基准")
    print("=" * 66)
    print(f"  onnxruntime : {ort.__version__}")
    print(f"  可用 EP     : {ort.get_available_providers()}")
    print(f"  模型        : {args.model}")

    # 探测模型输入形状
    probe = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = probe.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    print(f"  输入        : {inp.name} {inp.shape} {inp.type}")
    for o in probe.get_outputs():
        print(f"  输出        : {o.name} {o.shape} {o.type}")
    dynamic = any(not isinstance(d, int) for d in inp.shape)
    print(f"  动态形状    : {dynamic}")

    batch = shape[0] if shape and shape[0] > 0 else 1
    np_dtype = np.float16 if "float16" in inp.type else np.float32
    print(f"  batch       : {batch}   <- 固定形状模型只能按导出时的 batch 跑")
    print(f"  输入 dtype  : {np_dtype.__name__}")
    print()

    results = {}
    targets = []
    if "DmlExecutionProvider" in ort.get_available_providers():
        targets.append(("DmlExecutionProvider", "DirectML (GPU)"))
    else:
        print("  ⚠️ DmlExecutionProvider 不在可用列表, 跳过 GPU 测试\n")
    targets.append(("CPUExecutionProvider", "CPU"))

    for ep, label in targets:
        print(f"--- {label}  [{ep}] ---")
        try:
            t0 = time.perf_counter()
            sess = make_session(args.model, ep)
            init_ms = (time.perf_counter() - t0) * 1000
            used = sess.get_providers()
            r = bench(sess, inp.name, shape, args.iters, args.warmup, np_dtype)
            results[ep] = r
            print(f"  会话初始化  : {init_ms:8.1f} ms")
            print(f"  首次推理    : {r['first_ms']:8.1f} ms   (含 EP 编译/图优化)")
            print(f"  稳态        : {r['mean_ms']:8.2f} ms/帧  "
                  f"(p50 {r['p50_ms']:.2f} / min {r['min_ms']:.2f} / max {r['max_ms']:.2f})")
            print(f"  吞吐        : {1000 / r['mean_ms']:8.1f} fps  "
                  f"({args.iters} 次平均)")
            print(f"  实际使用 EP : {used}")
        except Exception as e:
            print(f"  ❌ 失败: {type(e).__name__}: {e}")
        print()

    # 数值一致性: DML vs CPU
    if "DmlExecutionProvider" in results and "CPUExecutionProvider" in results:
        print("--- 数值一致性 (DirectML vs CPU) ---")
        a = results["DmlExecutionProvider"]["outputs"]
        b = results["CPUExecutionProvider"]["outputs"]
        for i, (oa, ob) in enumerate(zip(a, b)):
            d = np.abs(oa.astype(np.float64) - ob.astype(np.float64))
            print(f"  output{i}: shape {oa.shape}  max|Δ| {d.max():.3e}  "
                  f"mean|Δ| {d.mean():.3e}")
        sp = results["CPUExecutionProvider"]["mean_ms"] / results["DmlExecutionProvider"]["mean_ms"]
        print(f"\n  加速比 (CPU / DML): {sp:.2f}x")

    # 视频时长换算 (按 30fps 素材)
    if "DmlExecutionProvider" in results:
        fps = 1000 / results["DmlExecutionProvider"]["mean_ms"]
        print("\n--- 换算参考 (30fps 素材, 仅模型前向, 不含解码/后处理/编码) ---")
        for mins in (1, 5, 10):
            frames = mins * 60 * 30
            print(f"  {mins:2d} 分钟视频 ({frames:6d} 帧) @ {fps:.1f} fps = {frames / fps / 60:.1f} min")

    return 0


if __name__ == "__main__":
    sys.exit(main())
