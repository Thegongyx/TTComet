#!/usr/bin/env python3
"""
gzip 无损性 + 开销实测
=====================
回答两个问题:
  1. 把 jsonl 改成 .jsonl.gz 会不会影响内容/精度?  -> 用 sha256 逐字节比对
  2. 压缩/解压要花多少 CPU 时间?                  -> 测吞吐

用法:
    webui\.venv\Scripts\python.exe webui\bench\verify_gzip.py `
        inference_output/pipeline_xxx/raw_detections.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def sha256_file(path: str, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
            h.update(b)
    return h.hexdigest(), n


def sha256_stream(read, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    while True:
        b = read(chunk)
        if not b:
            break
        n += len(b)
        h.update(b)
    return h.hexdigest(), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()

    src = args.path
    size = os.path.getsize(src)
    print("=" * 66)
    print("  gzip 无损性 + 开销实测")
    print("=" * 66)
    print(f"  源文件 : {src}")
    print(f"  大小   : {size / 1024 / 1024:.1f} MB")
    print(f"  级别   : {args.level}")

    t0 = time.perf_counter()
    h_src, n_src = sha256_file(src)
    t_hash = time.perf_counter() - t0
    print(f"\n  源文件 sha256: {h_src}")
    print(f"  源文件字节数: {n_src:,}  (哈希耗时 {t_hash:.1f}s)")

    tmpdir = tempfile.mkdtemp(prefix="gzipcheck_")
    gz_path = os.path.join(tmpdir, "out.jsonl.gz")
    try:
        # ---- 压缩 (与推理端 "写 gzip" 等价) ----
        t0 = time.perf_counter()
        with open(src, "rb") as fi, \
                gzip.open(gz_path, "wb", compresslevel=args.level) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        t_comp = time.perf_counter() - t0
        gz_size = os.path.getsize(gz_path)

        # ---- 解压 (与高光端 "读 gzip" 等价) ----
        t0 = time.perf_counter()
        with gzip.open(gz_path, "rb") as fi:
            h_gz, n_gz = sha256_stream(fi.read)
        t_decomp = time.perf_counter() - t0

        print("\n  --- 压缩 ---")
        print(f"  输出        : {gz_size / 1024 / 1024:.1f} MB  "
              f"({gz_size / size * 100:.2f}% 原体积, 压缩比 {size / gz_size:.1f}:1)")
        print(f"  压缩耗时    : {t_comp:.1f}s  ({size / 1024 / 1024 / t_comp:.1f} MB/s)")
        print("\n  --- 解压 ---")
        print(f"  解压耗时    : {t_decomp:.1f}s  "
              f"({size / 1024 / 1024 / t_decomp:.1f} MB/s)")
        print(f"  解压后 sha256: {h_gz}")
        print(f"  解压后字节数 : {n_gz:,}")

        print("\n  --- 一致性 ---")
        if h_gz == h_src and n_gz == n_src:
            print("  ✅ 逐字节完全一致 (sha256 相同) -> gzip 无损, 不影响任何精度")
        else:
            print("  ❌ 不一致!")
            print(f"     期望 {n_src:,} B / {h_src}")
            print(f"     实际 {n_gz:,} B / {h_gz}")
            return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
