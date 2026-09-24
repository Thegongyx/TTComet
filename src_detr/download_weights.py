#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR 官方预训练权重下载器 (多连接分段 + MD5 校验)
============================================================
为什么不用 HuggingFace 镜像:
  * 本项目用的 `rfdetr` pip 包，其官方权重托管在 **Google Cloud Storage**
    (`https://storage.googleapis.com/rfdetr/...`)，是 `.pth` 检查点格式。
  * HuggingFace 上虽有 `Roboflow/rf-detr-*`，但那是 **transformers safetensors** 格式
    (须用 `transformers.RfDetrForObjectDetection` 加载)，与 pip 包不通用；
    且实测 hf-mirror 速度(~24 KiB/s)远低于官方源(~600 KiB/s)。
  * 因此本脚本直接对官方源做 **多连接分段下载**（实测约 2 倍速），
    并把文件放到 rfdetr 缓存目录，使 `RFDETR*/train` 直接命中、不再下载。

用法 (conda 环境 rfdetr, 或任意有标准库的 python):
------------------------------------------------------------
# 默认下载 seg-nano / seg-small / seg-medium
conda run -n rfdetr python src_detr/download_weights.py

# 指定模型
conda run -n rfdetr python src_detr/download_weights.py --models seg-nano seg-small

# 同时下载检测系列的 nano/small
conda run -n rfdetr python src_detr/download_weights.py --models detect-nano detect-small

# 列出全部可下载项
conda run -n rfdetr python src_detr/download_weights.py --list

# 全部下载
conda run -n rfdetr python src_detr/download_weights.py --all

# 强制重下 / 调整并发
conda run -n rfdetr python src_detr/download_weights.py --models seg-small --force --connections 8
"""

import os
import sys
import time
import hashlib
import argparse
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_GCS = "https://storage.googleapis.com/rfdetr/"

# key -> (缓存文件名, 下载URL, MD5)
WEIGHTS = {
    # ---- 分割 (seg): 项目主用 ----
    "seg-preview": ("rf-detr-seg-preview.pt", _GCS + "rf-detr-seg-preview.pt",
                    "e35820c28fb86080558123e47a5e49ca"),
    "seg-nano":    ("rf-detr-seg-nano.pt", _GCS + "rf-detr-seg-n-ft.pth",
                    "9995497791d0ff1664a1d9ddee9cfd20"),
    "seg-small":   ("rf-detr-seg-small.pt", _GCS + "rf-detr-seg-s-ft.pth",
                    "0a2a3006381d0c42853907e700eadd08"),
    "seg-medium":  ("rf-detr-seg-medium.pt", _GCS + "rf-detr-seg-m-ft.pth",
                    "a49af1562c3719227ad43d0ca53b4c7a"),
    "seg-large":   ("rf-detr-seg-large.pt", _GCS + "rf-detr-seg-l-ft.pth",
                    "275f7b094909544ed2841c94a677d07e"),
    "seg-xlarge":  ("rf-detr-seg-xlarge.pt", _GCS + "rf-detr-seg-xl-ft.pth",
                    "3693b35d0eea86ebb3e0444f4a611fba"),
    "seg-2xlarge": ("rf-detr-seg-2xlarge.pt", _GCS + "rf-detr-seg-2xl-ft.pth",
                    "040bc3412af840fa8a47e0ff69b552ba"),
    # ---- 检测 (detect) ----
    "detect-nano":   ("rf-detr-nano.pth", _GCS + "nano_coco/checkpoint_best_regular.pth",
                      "fb6504cce7fbdc783f7a46991f07639f"),
    "detect-small":  ("rf-detr-small.pth", _GCS + "small_coco/checkpoint_best_regular.pth",
                      "fb37061c1af7bace359c91b723a8d5c1"),
    "detect-medium": ("rf-detr-medium.pth", _GCS + "medium_coco/checkpoint_best_regular.pth",
                      "7223f764a87b863f02eb8d52bf0ce2ee"),
    "detect-large":  ("rf-detr-large-2026.pth", _GCS + "rf-detr-large-2026.pth",
                      "5cb72153541cbcb9aa6efa26222acc75"),
    "detect-base":   ("rf-detr-base.pth", _GCS + "rf-detr-base-coco.pth",
                      "b4d3ce46099eaed50626ede388caf979"),
    "keypoint-preview": ("rf-detr-keypoint-preview-xlarge.pth",
                         _GCS + "rf-detr-keypoint-preview-xlarge.pth",
                         "6de511943ee85a547d4c5cb527daf0eb"),
}

_DEFAULT_MODELS = ["seg-nano", "seg-small", "seg-medium"]
_UA = {"User-Agent": "Mozilla/5.0"}
_CHUNK = 8 * 1024 * 1024          # 8 MiB / 分段
_RETRY = 4


def cache_dir() -> Path:
    """与 rfdetr.assets.model_weights.get_model_cache_dir 保持一致。"""
    root = os.environ.get("RF_HOME") or os.environ.get("ROBOFLOW_HOME")
    if root:
        return Path(os.path.expanduser(root)) / "models"
    return Path(os.path.expanduser("~")) / ".roboflow" / "models"


def _head(url):
    req = urllib.request.Request(url, headers=_UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0)), \
            (r.headers.get("Accept-Ranges", "") == "bytes")


def _md5_file(path, bufsize=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class _Progress:
    def __init__(self, total, label):
        self.total = total
        self.done = 0
        self.label = label
        self.lock = threading.Lock()
        self.t0 = time.time()
        self._last = 0.0

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self._last >= 1.0 or self.done >= self.total:
                self._last = now
                pct = self.done / self.total * 100 if self.total else 0
                spd = self.done / max(now - self.t0, 1e-6) / 1024
                sys.stdout.write(
                    f"\r    {self.label}: {pct:5.1f}%  "
                    f"{self.done/1e6:6.1f}/{self.total/1e6:.1f} MB  {spd:6.0f} KiB/s")
                sys.stdout.flush()

    def finish(self):
        with self.lock:
            dt = max(time.time() - self.t0, 1e-6)
            sys.stdout.write(
                f"\r    {self.label}: 100.0%  {self.total/1e6:6.1f} MB  "
                f"{self.total/dt/1024:6.0f} KiB/s (avg)   \n")
            sys.stdout.flush()


def _fetch_range(url, start, end, out_path, prog):
    """下载 url 的 [start, end] 闭区间到 out_path, 支持重试。"""
    want = end - start + 1
    for attempt in range(1, _RETRY + 1):
        written = 0
        try:
            req = urllib.request.Request(url, headers={**_UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                with open(out_path, "wb") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        written += len(buf)
                        prog.add(len(buf))
            if out_path.stat().st_size == want:
                return
        except Exception as e:
            if attempt >= _RETRY:
                raise
            time.sleep(1.5 * attempt)
        # 失败/不完整: 回退本次已计入的进度, 避免重试后重复计数
        prog.add(-written)
    raise RuntimeError(f"分段 {start}-{end} 下载不完整")


def parallel_download(url, dest: Path, connections=8, chunk=_CHUNK):
    total, ranges_ok = _head(url)
    if total <= 0:
        raise RuntimeError(f"无法获取文件大小: {url}")
    print(f"  URL: {url}")
    print(f"  大小: {total/1e6:.1f} MB | 支持分段: {ranges_ok} | 并发: {connections}")

    if not ranges_ok or connections <= 1:
        prog = _Progress(total, dest.name)
        tmp = dest.with_suffix(dest.suffix + ".single")
        _fetch_range(url, 0, total - 1, tmp, prog)
        prog.finish()
        os.replace(tmp, dest)
        return

    parts_dir = dest.parent / (dest.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (total + chunk - 1) // chunk
    jobs = []
    for i in range(n_chunks):
        start = i * chunk
        end = min(total - 1, start + chunk - 1)
        p = parts_dir / f"part{i:04d}"
        jobs.append((i, start, end, p))

    # 已完整的分段可断点续传
    remaining = [j for j in jobs if not (j[3].exists() and j[3].stat().st_size == j[2] - j[1] + 1)]
    prog = _Progress(total, dest.name)
    # 已完成分段计入进度
    for i, s, e, p in jobs:
        if p.exists() and p.stat().st_size == e - s + 1:
            prog.add(e - s + 1)

    if remaining:
        with ThreadPoolExecutor(max_workers=connections) as ex:
            futs = [ex.submit(_fetch_range, url, s, e, p, prog) for _, s, e, p in remaining]
            for f in as_completed(futs):
                f.result()
    prog.finish()

    tmp = dest.with_suffix(dest.suffix + ".merged")
    with open(tmp, "wb") as out:
        for i, s, e, p in jobs:
            with open(p, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
    os.replace(tmp, dest)
    for p in parts_dir.glob("part*"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        parts_dir.rmdir()
    except Exception:
        pass


def download_one(key, cdir: Path, connections, force=False):
    filename, url, md5 = WEIGHTS[key]
    dest = cdir / filename
    prefix = f"[{key}] {filename}"

    if dest.exists() and not force:
        actual = _md5_file(dest)
        if actual == md5:
            print(f"✅ {prefix} 已存在且校验通过 ({dest.stat().st_size/1e6:.1f} MB), 跳过")
            return True
        print(f"⚠️ {prefix} 已存在但 MD5 不匹配, 重新下载")
    print(f"⬇️  {prefix}")
    parallel_download(url, dest, connections=connections)

    actual = _md5_file(dest)
    if actual == md5:
        print(f"✅ {prefix} 下载完成, MD5 校验通过 ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"❌ {prefix} MD5 不匹配! 期望 {md5}, 实际 {actual}")
    return False


def main():
    ap = argparse.ArgumentParser(description="RF-DETR 官方权重多连接下载器")
    ap.add_argument("--models", nargs="+", default=None,
                    help=f"要下载的模型 key, 默认 {_DEFAULT_MODELS}")
    ap.add_argument("--all", action="store_true", help="下载全部")
    ap.add_argument("--list", action="store_true", help="列出所有可下载项")
    ap.add_argument("--connections", type=int, default=8, help="并发连接数 (默认 8)")
    ap.add_argument("--cache-dir", default=None, help="缓存目录 (默认 rfdetr 缓存目录)")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    if args.list:
        print(f"缓存目录: {cache_dir()}")
        print("可下载项:")
        for k, (fn, url, md5) in WEIGHTS.items():
            print(f"  {k:18s} -> {fn}")
        return

    models = list(WEIGHTS) if args.all else (args.models or _DEFAULT_MODELS)
    bad = [m for m in models if m not in WEIGHTS]
    if bad:
        print(f"❌ 未知模型 key: {bad}\n   可用: {list(WEIGHTS)}")
        sys.exit(1)

    cdir = Path(args.cache_dir) if args.cache_dir else cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"RF-DETR 权重下载 | 目标目录: {cdir}")
    print(f"模型: {models} | 并发: {args.connections}")
    print("=" * 70)

    ok = True
    for m in models:
        if not download_one(m, cdir, args.connections, force=args.force):
            ok = False

    print("=" * 70)
    if ok:
        print("🎉 全部完成。后续 RFDETR*/train 会直接命中缓存，不再联网下载。")
    else:
        print("⚠️ 部分文件失败, 请重跑本脚本 (支持断点续传)。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
