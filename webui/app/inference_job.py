#!/usr/bin/env python3
"""
推理任务（可被 CLI 与 WebUI 共用）
==================================
把「VFR 转换 -> 逐帧推理 -> 写 jsonl.gz」抽成一个函数，带进度/日志回调。
`run_video.py` 只是它的命令行外壳，WebUI 也调它，保证行为只有一份实现。

输出目录约定（与 run_pipeline.py 一致）：
    <out_dir>/video_cfr.mp4        VFR 素材才会生成
    <out_dir>/raw_detections.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import os
import queue
import statistics
import threading
import time
from typing import Callable

import cv2
import numpy as np

import video_utils
from onnx_engine import DetectorConfig, OnnxDetector

LogCb = Callable[[str], None]
ProgressCb = Callable[[dict], None]


def _noop(_msg):
    pass


def infer_video(video: str,
                out_dir: str,
                cfg: DetectorConfig | None = None,
                *,
                model: str = "webui/models/m960n_fp16_nonms.onnx",
                max_frames: int = 0,
                start_frame: int = 0,
                cfr: bool = True,
                cfr_highlights: float = 10.0,
                cfr_contrast: float = 10.0,
                cfr_saturation: float = 6.0,
                queue_size: int = 2,
                progress_every: float = 1.0,
                compress: bool = False,
                should_cancel: Callable[[], bool] | None = None,
                log: LogCb | None = None,
                progress: ProgressCb | None = None,
                detector: OnnxDetector | None = None) -> dict:
    """跑完整段推理，返回结果统计。

    detector 可传入已加载的实例以复用（WebUI 常驻进程时避免重复加载模型）。
    """
    log = log or _noop
    emit = progress or (lambda _d: None)

    if not os.path.exists(video):
        raise FileNotFoundError(f"视频不存在: {video}")
    os.makedirs(out_dir, exist_ok=True)

    if detector is None:
        t_load = time.perf_counter()
        detector = OnnxDetector(model, cfg or DetectorConfig())
        log(f"模型已加载: {model} ({detector.load_ms:.0f} ms, 后端 {detector.providers})")
    cfg = detector.cfg

    # ---- VFR -> CFR（与 src/inference_video.py 一致）----
    src_video = video
    cfr_path = os.path.join(out_dir, "video_cfr.mp4")
    is_vfr = False
    if cfr and video_utils.is_variable_frame_rate(video):
        is_vfr = True
        log("⚠️ 检测到可变帧率 (VFR)，正在转成恒定帧率...")
        video_utils.convert_vfr_to_cfr(
            video, cfr_path,
            highlights_pct=cfr_highlights,
            contrast_pct=cfr_contrast,
            saturation_pct=cfr_saturation)
        src_video = cfr_path
        log(f"VFR→CFR 完成: {cfr_path}")

    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {src_video}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_target = max_frames if max_frames > 0 else max(0, total - start_frame)
    out_path = os.path.join(out_dir,
                            "raw_detections.jsonl.gz" if compress else "raw_detections.jsonl")
    fout = (gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6)
            if compress else open(out_path, "w", encoding="utf-8"))

    log(f"视频: {os.path.basename(src_video)}  {w}x{h} @ {fps_src:.2f}fps  "
        f"{total} 帧" + ("  (已转 CFR)" if is_vfr else ""))

    stats = {k: [] for k in ("decode", "pre", "infer", "post", "write")}
    lock = threading.Lock()

    def add(k, v):
        with lock:
            stats[k].append(v)

    def mean(k):
        with lock:
            return statistics.fmean(stats[k]) if stats[k] else 0.0

    t_start = time.perf_counter()
    last = [t_start]
    last_n = [0]

    def report(n_done: int):
        now = time.perf_counter()
        if now - last[0] < progress_every:
            return
        rate = (n_done - last_n[0]) / (now - last[0])
        emit({
            "stage": "inference",
            "frames": n_done,
            "total": n_target,
            "percent": round(n_done / n_target * 100, 1) if n_target else 0.0,
            "fps": round(rate, 1),
            "eta_sec": round((n_target - n_done) / max(rate, 1e-6), 0),
            "pre_ms": round(mean("pre"), 1),
            "infer_ms": round(mean("infer"), 1),
            "post_ms": round(mean("post"), 1),
        })
        last[0], last_n[0] = now, n_done

    def do_post(item):
        idx, out0, proto, scale, pad_x, pad_y, fshape = item
        t0 = time.perf_counter()
        detections, counts, best_ball, _ = detector.postprocess(
            out0, proto, fshape, scale, pad_x, pad_y)
        t1 = time.perf_counter()
        all_counts = {n: counts.get(n, 0) for n in detector.logic_names.values()}
        rec = {
            "frame": idx,
            "timestamp": round(idx / fps_src, 2) if fps_src else 0.0,
            "counts": all_counts,
            "ball_pos": ([round(best_ball["pos"][0], 1), round(best_ball["pos"][1], 1)]
                         if best_ball else None),
            "detections": detections,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        t2 = time.perf_counter()
        add("post", (t1 - t0) * 1000)
        add("write", (t2 - t1) * 1000)

    q_pre: queue.Queue = queue.Queue(maxsize=queue_size)
    q_post: queue.Queue = queue.Queue(maxsize=queue_size)
    SENTINEL = None
    cancelled = threading.Event()

    def reader():
        idx = 0
        while idx < n_target and not cancelled.is_set():
            if should_cancel is not None and should_cancel():
                cancelled.set()
                log("收到取消请求，停止读取帧")
                break
            t_a = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t_b = time.perf_counter()
            blob, scale, pad_x, pad_y = detector.preprocess(frame)
            t_c = time.perf_counter()
            add("decode", (t_b - t_a) * 1000)
            add("pre", (t_c - t_b) * 1000)
            q_pre.put((start_frame + idx, blob, scale, pad_x, pad_y, frame.shape[:2]))
            idx += 1
        q_pre.put(SENTINEL)

    def inferrer():
        while True:
            item = q_pre.get()
            if item is SENTINEL:
                break
            idx, blob, scale, pad_x, pad_y, fshape = item
            t0 = time.perf_counter()
            out0, proto = detector.infer(blob)
            add("infer", (time.perf_counter() - t0) * 1000)
            q_post.put((idx, out0, proto, scale, pad_x, pad_y, fshape))
        q_post.put(SENTINEL)

    n_done = 0
    try:
        t_read = threading.Thread(target=reader, name="reader", daemon=True)
        t_infer = threading.Thread(target=inferrer, name="infer", daemon=True)
        t_read.start()
        t_infer.start()
        while True:
            item = q_post.get()
            if item is SENTINEL:
                break
            do_post(item)
            n_done += 1
            report(n_done)
        t_read.join()
        t_infer.join()
    finally:
        try:
            fout.close()
        finally:
            cap.release()

    wall = time.perf_counter() - t_start
    result = {
        "frames": n_done,
        "wall_sec": round(wall, 1),
        "fps": round(n_done / wall, 1) if wall > 0 else 0.0,
        "source_fps": round(fps_src, 2),
        "was_vfr": is_vfr,
        "cancelled": cancelled.is_set(),
        "jsonl": out_path,
        "video": src_video,
        "cfr_video": cfr_path if is_vfr else None,
        "stages_ms": {k: round(mean(k), 2) for k in stats},
        "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 2),
    }
    log(f"推理完成: {n_done} 帧, {result['fps']} fps, 输出 {result['size_mb']} MB")
    emit({"stage": "inference_done", "percent": 100.0, **result})
    return result
