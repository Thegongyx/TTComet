#!/usr/bin/env python3
"""
PingPong Video Inference v8 — AMD Radeon 8060S / Windows Native 深度优化
==========================================================
v7 → v8 性能优化 (解决 GPU 利用率仅 60% / 45fps 瓶颈):
  [v8 PERF] 核心改动 —— 解耦 GPU 推理与 CPU 后处理:
    - [关键] 后处理 (resize/绘图/mask/JSON) 从主循环剥离到独立 worker 线程池。
        原来: GPU推理 → 主线程串行做完所有后处理 → 下一批 (GPU 空转等 CPU)
        现在: 主线程只负责 喂批 + GPU 推理 + 一次性搬运结果到 CPU;
              后处理由 N 个 worker 并行消费 → GPU 连续满载。
    - [关键] 结果 GPU→CPU 搬运在主线程一次性完成 (extract_result_cpu),
        避免后处理线程里再触发 .cpu() 同步 stall GPU。
    - 多解码线程: 大分辨率视频单线程 cap.read() 是隐藏瓶颈,
        可用 --reader-threads 拆多个解码器分段并行 (可选)。
    - 顺序保证: 后处理结果带 seq 号, 由 ordering buffer 重排后再写盘,
        保证视频帧顺序 100% 正确。
    - read_queue / 后处理队列加深, 吸收抖动。
    - torch.cuda.synchronize 只在必要处; 移除每帧隐式同步。
  v7 全部功能保留 (AMF 硬编 / VFR→CFR 调色 / polygon 选择性提取 等)。
"""

# ---------------------------------------------------------------------------
# MIOpen / HIP 环境变量 (必须在 import torch 之前!)
# ---------------------------------------------------------------------------
import os
import sys
import time
import platform

_IS_WINDOWS = (platform.system() == "Windows")

if _IS_WINDOWS:
    _DEF_MIOPEN_CACHE = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "miopen"
    )
else:
    _DEF_MIOPEN_CACHE = os.path.expanduser("~/.cache/miopen")
os.makedirs(_DEF_MIOPEN_CACHE, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH",       _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR",   _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_FIND_MODE",          "FAST")
os.environ.setdefault("MIOPEN_FIND_ENFORCE",       "NONE")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",    "expandable_segments:True")

from ultralytics import YOLO
import cv2
import numpy as np
import json
import shutil
import subprocess
import argparse
import torch
import threading
from queue import Queue
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

# [v8 PERF] 限制 OpenCV 内部线程数, 避免与我们的后处理线程池抢 CPU 打架
try:
    cv2.setNumThreads(2)
except Exception:
    pass

_SUBPROC_FLAGS = 0
if _IS_WINDOWS:
    _SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

# ==================== 类别配置 ====================
CLASS_CONFIG_TEMPLATE = {
    "pingpangball": {
        "color": (0, 255, 0), "conf": 0.25,
        "aliases": ["pingpangball", "pingpongball", "ppball", "乒乓球"]
    },
    "otherball": {
        "color": (0, 140, 255), "conf": 0.40,
        "aliases": ["otherball", "其他比赛球"]
    },
    "player": {
        "color": (0, 140, 255), "conf": 0.40,
        "aliases": ["player", "players", "球员"]
    },
    "racket": {
        "color": (0, 255, 255), "conf": 0.30,
        "aliases": ["racket", "pingpang paddel", "paddel", "paddle", "bat", "球拍"]
    },
    "table": {
        "color": (255, 0, 255), "conf": 0.30,
        "aliases": ["table", "pingpang table", "pingpong table", "球桌"]
    },
    "net": {
        "color": (255, 255, 255), "conf": 0.30,
        "aliases": ["net", "pingpang net", "pingpong net", "球网"]
    },
}
BALL_LOGIC_NAME = "pingpangball"
POLYGON_NEEDED_LOGIC_NAMES = {"table", "net"}


# ==================== CLI ====================
def _default_video():
    if _IS_WINDOWS:
        return os.path.join(os.path.expanduser("~"), "Videos", "VID_20260429_220054.mp4")
    return "/home/gongy/Videos/VID_20260429_220054.mp4"

def _default_model():
    if _IS_WINDOWS:
        return os.path.join(os.path.expanduser("~"), "yolo26_frames",
                            "runs", "train_pingpong_v2", "pingpong_x_v1",
                            "weights", "best.pt")
    return "/home/gongy/yolo26_frames/runs/train_pingpong_v2/pingpong_x_v1/weights/best.pt"

def _default_output():
    if _IS_WINDOWS:
        return os.path.join(os.path.expanduser("~"), "yolo26_frames", "inference_output")
    return "/home/gongy/yolo26_frames/inference_output"


def parse_args():
    parser = argparse.ArgumentParser(description="PingPong Video Inference v8 (Windows/AMF, perf-tuned)")
    parser.add_argument("--video", "-v", type=str, default=_default_video())
    parser.add_argument("--video-list-file", type=str, default=None,
                        help="批处理模式: UTF-8 文本文件, 每行一个视频绝对路径")
    parser.add_argument("--model", "-m", type=str, default=_default_model())
    parser.add_argument("--output", "-o", type=str, default=_default_output())
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", action="store_true", help="FP16 推理 (ROCm/CUDA 加速)")
    parser.add_argument("--imgsz", type=int, default=960, help="推理分辨率, 默认 960")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou",  type=float, default=0.45)
    parser.add_argument("--conf-ball",   type=float, default=0.30)
    parser.add_argument("--conf-player", type=float, default=None)
    parser.add_argument("--conf-racket", type=float, default=None)
    parser.add_argument("--conf-table",  type=float, default=None)
    parser.add_argument("--conf-net",    type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批大小, ROCm 建议 8-16; 尾批用零帧 padding 保持固定 shape")
    parser.add_argument("--remove-cfr", action="store_true")
    parser.add_argument("--log-flush-interval", type=float, default=0.5)
    parser.add_argument("--writer-backend", type=str, default="ffmpeg",
                        choices=["ffmpeg", "opencv"])
    parser.add_argument("--out-width", type=int, default=640)
    parser.add_argument("--no-polygon", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=0,
                        help="每 N 批清一次 GPU 缓存 (0 关闭; v8 默认关闭, 清缓存会 stall)")
    parser.add_argument("--conf-margin", type=float, default=0.05)
    parser.add_argument("--encoder", type=str, default="auto",
                        choices=["auto", "libx264", "h264_amf", "hevc_amf", "av1_amf"])
    parser.add_argument("--amf-quality", type=str, default="quality",
                        choices=["speed", "balanced", "quality"])
    parser.add_argument("--amf-qp", type=int, default=20)
    parser.add_argument("--end2end", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--cfr-highlights", type=float, default=10.0)
    parser.add_argument("--cfr-contrast", type=float, default=10.0)
    parser.add_argument("--cfr-saturation", type=float, default=6.0)
    # [v8 PERF] 新增性能相关参数
    parser.add_argument("--post-workers", type=int, default=8,
                        help="[v8] CPU 后处理 worker 线程数 (绘图/resize/mask/JSON)。"
                             "GPU 利用率低时调大 (4-8); CPU 核多可到物理核数-2")
    parser.add_argument("--reader-threads", type=int, default=1,
                        help="[v8] 视频解码线程数。大分辨率视频单线程 cap.read() "
                             "可能是瓶颈, 设 2-3 分段并行解码 (需 total_frames 已知)")
    parser.add_argument("--prefetch-batches", type=int, default=6,
                        help="[v8] 预取批数 (read_queue 深度), 吸收解码/GPU 抖动")
    return parser.parse_args()


# ==================== 硬件加速探测 (AMF for AMD on Windows) ====================
def _detect_amf_encoder(prefer="auto"):
    if prefer == "libx264":
        return None, None
    if not shutil.which("ffmpeg"):
        return None, None
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=8,
                           creationflags=_SUBPROC_FLAGS)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None, None
    has_h264 = "h264_amf" in out
    has_hevc = "hevc_amf" in out
    has_av1  = "av1_amf"  in out
    if prefer == "h264_amf" and has_h264:
        enc = "h264_amf"
    elif prefer == "hevc_amf" and has_hevc:
        enc = "hevc_amf"
    elif prefer == "av1_amf" and has_av1:
        enc = "av1_amf"
    elif has_h264:
        enc = "h264_amf"
    elif has_hevc:
        enc = "hevc_amf"
    elif has_av1:
        enc = "av1_amf"
    else:
        return None, None
    try:
        probe = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2:r=25",
            "-c:v", enc, "-f", "null", "-"
        ], capture_output=True, text=True, timeout=15,
           creationflags=_SUBPROC_FLAGS)
        if probe.returncode != 0:
            return None, None
    except Exception:
        return None, None
    return enc, None


def build_conf_thresholds(args, class_config):
    conf_map = {
        "pingpangball": args.conf_ball, "player": args.conf_player,
        "racket": args.conf_racket, "table": args.conf_table, "net": args.conf_net,
    }
    thresholds = {}
    for cls_id, cfg in class_config.items():
        logic_name = cfg["name"]
        thresholds[cls_id] = (conf_map[logic_name]
                              if conf_map.get(logic_name) is not None
                              else cfg["conf"])
    return thresholds


# ==================== VFR -> CFR ====================
def is_variable_frame_rate(video_path):
    if not shutil.which("ffprobe"):
        return False
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate,avg_frame_rate",
               "-of", "default=noprint_wrappers=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, timeout=30,
                                creationflags=_SUBPROC_FLAGS)
        r_fps, avg_fps = None, None
        for line in result.stdout.strip().splitlines():
            if line.startswith("r_frame_rate="):
                r_fps = line.split("=", 1)[1].strip()
            elif line.startswith("avg_frame_rate="):
                avg_fps = line.split("=", 1)[1].strip()
        if not r_fps or not avg_fps:
            return False

        def _to_f(x):
            try:
                if "/" in x:
                    a, b = x.split("/", 1)
                    return float(a) / float(b) if float(b) != 0 else 0.0
                return float(x)
            except Exception:
                return 0.0

        r, a = _to_f(r_fps), _to_f(avg_fps)
        if r <= 0 or a <= 0:
            return False
        return abs(r - a) / max(r, a) > 0.00001
    except Exception:
        return False


def _probe_has_audio(video_path):
    if not shutil.which("ffprobe"):
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=15,
            creationflags=_SUBPROC_FLAGS)
        return "audio" in (r.stdout or "")
    except Exception:
        return False


def convert_vfr_to_cfr(video_path, output_path, target_fps=None,
                       highlights_pct=10.0, contrast_pct=10.0, saturation_pct=6.0):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found.")
    if target_fps is None:
        cap = cv2.VideoCapture(video_path)
        target_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if target_fps <= 0:
            target_fps = 30.0
    target_fps = round(target_fps, 3)
    has_audio = _probe_has_audio(video_path)
    print(f"  Converting VFR to CFR @ {target_fps} fps (audio: {'yes' if has_audio else 'no'}) ...")

    vf_filters = [f"fps={target_fps}"]
    eq_opts = []
    if contrast_pct != 0.0:
        eq_opts.append(f"contrast={1.0 + contrast_pct / 100.0:.4f}")
    if saturation_pct != 0.0:
        eq_opts.append(f"saturation={1.0 + saturation_pct / 100.0:.4f}")
    if eq_opts:
        vf_filters.append("eq=" + ":".join(eq_opts))
    if highlights_pct != 0.0:
        h_max = max(0.0, 1.0 - highlights_pct / 100.0)
        vf_filters.append(f"curves=master='0/0 0.5/0.5 1/{h_max:.4f}'")
    vf_chain = ",".join(vf_filters)

    if highlights_pct or contrast_pct or saturation_pct:
        print(f"  🎨 画面调整: 高光-{highlights_pct}%  对比度+{contrast_pct}%  饱和度+{saturation_pct}%")
        print(f"     -vf chain: {vf_chain}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts",
        "-i", video_path,
        "-vf", vf_chain,
        "-vsync", "cfr",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-async", "1"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, timeout=3600, creationflags=_SUBPROC_FLAGS)
    return output_path


# ==================== ffmpeg 管道写入器 ====================
class FFmpegWriter:
    def __init__(self, out_path, width, height, fps, prefer="auto",
                 amf_quality="quality", amf_qp=24):
        self.encoder = None
        enc, _ = _detect_amf_encoder(prefer=prefer)
        input_args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]
        if enc:
            self.encoder = enc
            qp = int(amf_qp)
            enc_args = [
                "-c:v", enc, "-usage", "transcoding", "-quality", amf_quality,
                "-rc", "cqp", "-qp_i", str(qp), "-qp_p", str(min(51, qp + 2)),
                "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-video_track_timescale", "90000", out_path,
            ]
            if enc == "av1_amf":
                enc_args = [
                    "-c:v", enc, "-usage", "transcoding", "-quality", amf_quality,
                    "-rc", "cqp", "-qp_i", str(qp), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-video_track_timescale", "90000",
                    out_path,
                ]
            cmd = input_args + enc_args
        else:
            cmd = input_args + [
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "0",
                "-movflags", "+faststart", "-video_track_timescale", "90000",
                out_path,
            ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0,
                                     creationflags=_SUBPROC_FLAGS)

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            pass

    def release(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=300)
        except Exception:
            self.proc.kill()


# ==================== 几何/绘图 ====================
def get_bbox_center(bbox):
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def draw_detections(frame, detections_list, colors):
    for det in detections_list:
        b = det["bbox"]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cls_id = det.get("class_id", 0)
        color = colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    return frame


def draw_mask_overlay_inplace(frame, bool_mask, color, alpha=0.4):
    if bool_mask is None or not bool_mask.any():
        return frame
    ys, xs = np.where(bool_mask)
    if ys.size == 0:
        return frame
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub_frame = frame[y0:y1, x0:x1]
    sub_mask  = bool_mask[y0:y1, x0:x1]
    if sub_frame.size == 0 or not sub_mask.any():
        return frame
    color_img = np.empty_like(sub_frame)
    color_img[:] = color
    blended = cv2.addWeighted(sub_frame, 1.0 - alpha, color_img, alpha, 0.0)
    sub_frame[sub_mask] = blended[sub_mask]
    return frame


# ==================== [v8 PERF] 多线程解码 reader ====================
def frame_reader(cap, q, batch_size, stop_evt):
    """单线程顺序解码 (reader_threads=1 时用)。"""
    idx = 0
    try:
        while not stop_evt.is_set():
            batch = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append((idx, frame))
                idx += 1
            if batch:
                q.put(batch)
            if len(batch) < batch_size:
                q.put(None)
                break
    except Exception as e:
        try:
            q.put(None)
        except Exception:
            pass
        print(f"⚠️ frame_reader error: {e}")


def frame_reader_segmented(video_path, q, batch_size, stop_evt,
                           seg_start, seg_end, out_seq_offset):
    """
    [v8 PERF] 分段解码: 每个线程用独立 VideoCapture 打开视频并 seek 到 seg_start,
    只解码 [seg_start, seg_end) 区间, 帧的全局 idx = seg_start + 本地偏移。
    多个这样的线程并行 → 提升总解码吞吐 (大分辨率 CPU 解码瓶颈时有效)。
    """
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        idx = seg_start
        while not stop_evt.is_set() and idx < seg_end:
            batch = []
            for _ in range(batch_size):
                if idx >= seg_end:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append((idx, frame))
                idx += 1
            if batch:
                q.put(batch)
            if idx >= seg_end or len(batch) == 0:
                break
    except Exception as e:
        print(f"⚠️ segmented reader error [{seg_start},{seg_end}): {e}")
    finally:
        cap.release()


def frame_writer(writer, q):
    while True:
        item = q.get()
        if item is None:
            break
        writer.write(item)


def log_writer(log_f, q, flush_interval=0.5):
    buf = []
    last_flush = datetime.now()
    while True:
        item = q.get()
        if item is None:
            if buf:
                log_f.write("".join(buf))
                log_f.flush()
            break
        buf.append(item)
        now = datetime.now()
        if (now - last_flush).total_seconds() >= flush_interval:
            log_f.write("".join(buf))
            log_f.flush()
            buf.clear()
            last_flush = now


# ==================== mask 处理 ====================
class MaskProcessor:
    def resize_masks(self, masks_np, cls_ids, out_hw, table_cls_id, net_cls_id):
        out_h, out_w = out_hw
        table_mask = net_mask = None
        for mask, cls_id in zip(masks_np, cls_ids):
            if cls_id == table_cls_id and table_mask is None:
                m = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                table_mask = (m > 0.5).astype(np.uint8)
            elif cls_id == net_cls_id and net_mask is None:
                m = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                net_mask = (m > 0.5).astype(np.uint8)
            if table_mask is not None and net_mask is not None:
                break
        return table_mask, net_mask


# ==================== [v8 PERF] GPU→CPU 一次性搬运 ====================
def extract_result_cpu(result, want_polygons, polygon_needed_cls_ids):
    """
    [v8 PERF 关键] 在主线程 (GPU 上下文所在线程) 一次性把当前 result 需要的
    张量搬到 CPU numpy。这样后处理 worker 线程里就不再触发任何 .cpu() 同步,
    GPU 不会因后处理线程的懒下载而 stall。
    返回一个纯 CPU 的轻量 dict, 可安全跨线程传递。
    """
    boxes = result.boxes
    masks = result.masks
    payload = {"xyxy": None, "conf": None, "cls": None,
               "masks_np": None, "polygons_by_idx": None}

    if boxes is not None and len(boxes) > 0:
        payload["xyxy"] = boxes.xyxy.cpu().numpy()
        payload["conf"] = boxes.conf.cpu().numpy()
        payload["cls"]  = boxes.cls.cpu().numpy().astype(np.int32)

        if masks is not None and masks.data is not None and len(masks.data) > 0:
            payload["masks_np"] = masks.data.cpu().numpy()

            # polygon 也在这里一次性取出 (masks.xy 内部也走 CPU)
            if want_polygons:
                all_cls = payload["cls"]
                need_idx = [i for i in range(len(all_cls))
                            if int(all_cls[i]) in polygon_needed_cls_ids]
                if need_idx:
                    polys = {}
                    try:
                        xy_list = masks.xy
                        for i in need_idx:
                            if i < len(xy_list):
                                poly = xy_list[i]
                                if poly is not None and len(poly) > 0:
                                    polys[i] = [
                                        [round(float(x), 1), round(float(y), 1)]
                                        for x, y in poly
                                    ]
                    except Exception:
                        pass
                    payload["polygons_by_idx"] = polys
    return payload


# ==================== [v8 PERF] 后处理 (纯 CPU, 在 worker 线程执行) ====================
def process_payload_cpu(
    payload, frame_raw, frame_idx, fps,
    CLASS_CONFIG, CONF_THRESHOLDS, CLASS_NAMES, COLORS,
    BALL_CLASS_ID, TABLE_CLASS_ID, NET_CLASS_ID,
    mask_proc, out_w, out_h, want_polygons
):
    """
    纯 CPU 后处理: 由 extract_result_cpu 产出的 payload (已在 CPU) 出发,
    做绘图/resize/mask/JSON。绝不触碰 GPU 张量 → 可在 worker 线程并行。
    返回 (out_frame, log_line, class_counts_local, ball_present_bool)
    """
    h_raw, w_raw = frame_raw.shape[:2]
    sx = out_w / w_raw
    sy = out_h / h_raw

    frame_class_counts_local = {cls_id: 0 for cls_id in CLASS_CONFIG}
    frame_detections = []
    best_detections = {}

    all_xyxy = payload["xyxy"]
    all_conf = payload["conf"]
    all_cls  = payload["cls"]
    masks_np = payload["masks_np"]
    polygons_by_idx = payload["polygons_by_idx"] or {}
    cls_ids_np = all_cls

    if all_cls is not None:
        n = len(all_cls)
        for i in range(n):
            cls_id = int(all_cls[i])
            conf = float(all_conf[i])
            if cls_id not in CLASS_CONFIG or conf < CONF_THRESHOLDS[cls_id]:
                continue
            x1, y1, x2, y2 = all_xyxy[i]
            name = CLASS_CONFIG[cls_id]["name"]
            det = {
                "class_id": cls_id, "class": name, "confidence": round(conf, 3),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "polygon": polygons_by_idx.get(i) if want_polygons else None
            }
            frame_detections.append(det)
            frame_class_counts_local[cls_id] += 1
            if cls_id not in best_detections or conf > best_detections[cls_id]["confidence"]:
                best_detections[cls_id] = det

    # table/net mask overlay
    if masks_np is not None and cls_ids_np is not None:
        table_mask_u8, net_mask_u8 = mask_proc.resize_masks(
            masks_np, cls_ids_np, (out_h, out_w), TABLE_CLASS_ID, NET_CLASS_ID)
    else:
        table_mask_u8 = net_mask_u8 = None

    ball_det = best_detections.get(BALL_CLASS_ID)
    bx = by = 0.0
    ball_pos = None
    if ball_det:
        bx, by = get_bbox_center(ball_det["bbox"])
        ball_pos = (bx, by)

    out_frame = cv2.resize(frame_raw, (out_w, out_h))
    if table_mask_u8 is not None:
        draw_mask_overlay_inplace(out_frame, table_mask_u8.astype(bool),
                                   (255, 0, 255), alpha=0.3)
    if net_mask_u8 is not None:
        draw_mask_overlay_inplace(out_frame, net_mask_u8.astype(bool),
                                   (255, 255, 255), alpha=0.4)

    scaled_dets = []
    for det in frame_detections:
        b = det["bbox"]
        scaled_dets.append({**det, "bbox": [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]})
    draw_detections(out_frame, scaled_dets, COLORS)

    timestamp = frame_idx / fps
    log_obj = {
        "frame": frame_idx,
        "timestamp": round(timestamp, 2),
        "counts": {CLASS_NAMES[cls_id]: frame_class_counts_local[cls_id]
                   for cls_id in CLASS_NAMES},
        "ball_pos": (round(bx, 1), round(by, 1)) if ball_pos else None,
        "detections": [
            {"class": d["class"], "confidence": d["confidence"],
             "bbox": [round(x, 1) for x in d["bbox"]], "polygon": d.get("polygon")}
            for d in frame_detections
        ]
    }
    log_line = json.dumps(log_obj) + "\n"

    any_det = any(c > 0 for c in frame_class_counts_local.values())
    return out_frame, log_line, frame_class_counts_local, any_det, timestamp


# ==================== 推理上下文 ====================
@dataclass
class InferenceContext:
    model: 'YOLO'
    device: str
    use_half: bool
    CLASS_CONFIG: dict
    CLASS_NAMES: dict
    COLORS: dict
    CONF_THRESHOLDS: dict
    BALL_CLASS_ID: int | None
    TABLE_CLASS_ID: int | None
    NET_CLASS_ID: int | None
    POLYGON_NEEDED_CLASS_IDS: set
    want_polygons: bool
    OUTPUT_DIR: str
    LOG_FILENAME: str
    BATCH_SIZE: int
    IMGSZ: int
    args: argparse.Namespace
    mask_proc: 'MaskProcessor'
    # pad_frame_zero: 'np.ndarray'   # 已废弃: 方形零帧会破坏尾批 shape, 改用 np.zeros_like(frames_in_batch[0])
    min_conf_for_predict: float


def load_model_and_config(args):
    if args.output.endswith(".json"):
        OUTPUT_DIR = os.path.dirname(args.output) or "."
        LOG_FILENAME = os.path.basename(args.output)
    else:
        OUTPUT_DIR = args.output
        LOG_FILENAME = "detection_log.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_PATH = args.model
    BATCH_SIZE = args.batch_size
    IMGSZ      = args.imgsz

    print("=" * 60)
    print(f"🏓 PingPong Video Inference v8 [Platform: {platform.system()}]")
    print("=" * 60)

    device = args.device
    use_half = args.half
    if device != "cpu" and not torch.cuda.is_available():
        print("⚠️ ROCm/GPU not detected, falling back to CPU")
        device = "cpu"
        use_half = False
    if device == "cpu":
        use_half = False

    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            dev_idx = int(device) if device.isdigit() else 0
            print(f"GPU: {torch.cuda.get_device_name(dev_idx)}")
        except Exception:
            pass

    print(f"\nLoading model: {MODEL_PATH} "
          f"[Device: {device}, FP16: {use_half}, Batch: {BATCH_SIZE}, imgsz: {IMGSZ}]")
    model = YOLO(MODEL_PATH)

    try:
        model.fuse()
        print("✅ Model fused (Conv+BN).")
    except Exception as e:
        print(f"⚠️ fuse() failed: {e}")

    _alias_lookup = {}
    for logic_name, cfg in CLASS_CONFIG_TEMPLATE.items():
        for a in cfg["aliases"]:
            _alias_lookup[a.lower()] = (logic_name, cfg)
    CLASS_CONFIG = {}
    for cls_id, raw_name in model.names.items():
        hit = _alias_lookup.get(raw_name.lower())
        if hit is not None:
            logic_name, cfg = hit
            CLASS_CONFIG[cls_id] = {"name": logic_name, "raw_name": raw_name,
                                    "color": cfg["color"], "conf": cfg["conf"]}
        else:
            CLASS_CONFIG[cls_id] = {"name": raw_name, "raw_name": raw_name,
                                    "color": (128, 128, 128), "conf": args.conf}

    BALL_CLASS_ID  = next((k for k, v in CLASS_CONFIG.items()
                           if v["name"] == BALL_LOGIC_NAME), None)
    TABLE_CLASS_ID = next((k for k, v in CLASS_CONFIG.items()
                           if v["name"] == "table"), None)
    NET_CLASS_ID   = next((k for k, v in CLASS_CONFIG.items()
                           if v["name"] == "net"), None)

    CLASS_NAMES = {k: v["name"] for k, v in CLASS_CONFIG.items()}
    COLORS      = {k: v["color"] for k, v in CLASS_CONFIG.items()}
    CONF_THRESHOLDS = build_conf_thresholds(args, CLASS_CONFIG)

    POLYGON_NEEDED_CLASS_IDS = {
        cls_id for cls_id, v in CLASS_CONFIG.items()
        if v["name"] in POLYGON_NEEDED_LOGIC_NAMES
    }
    want_polygons = (not args.no_polygon) and bool(POLYGON_NEEDED_CLASS_IDS)
    if not want_polygons:
        print("ℹ️ Polygon 输出已禁用")
    else:
        print(f"ℹ️ Polygon 仅对类别 {sorted(POLYGON_NEEDED_CLASS_IDS)} 提取")

    mask_proc = MaskProcessor()
    # pad_frame_zero = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)  # 已废弃: 见 process_one_video 尾批 padding
    min_conf_for_predict = max(0.01,
                               min(CONF_THRESHOLDS.values()) - float(args.conf_margin))

    return InferenceContext(
        model=model, device=device, use_half=use_half,
        CLASS_CONFIG=CLASS_CONFIG, CLASS_NAMES=CLASS_NAMES, COLORS=COLORS,
        CONF_THRESHOLDS=CONF_THRESHOLDS, BALL_CLASS_ID=BALL_CLASS_ID,
        TABLE_CLASS_ID=TABLE_CLASS_ID, NET_CLASS_ID=NET_CLASS_ID,
        POLYGON_NEEDED_CLASS_IDS=POLYGON_NEEDED_CLASS_IDS,
        want_polygons=want_polygons, OUTPUT_DIR=OUTPUT_DIR,
        LOG_FILENAME=LOG_FILENAME, BATCH_SIZE=BATCH_SIZE, IMGSZ=IMGSZ,
        args=args, mask_proc=mask_proc,  # pad_frame_zero 已废弃
        min_conf_for_predict=min_conf_for_predict,
    )


# ==================== [v8 PERF] 后处理 worker + 排序写盘 ====================
def _post_worker(ctx, fps, out_w, out_h, in_q, done_dict, done_lock, done_cv):
    """
    后处理 worker: 从 in_q 取 (seq, payload, frame_raw), 做 CPU 后处理,
    结果连同 seq 存进 done_dict, 由 ordering 线程按 seq 顺序取出写盘。
    """
    while True:
        item = in_q.get()
        if item is None:
            in_q.put(None)  # 传染给其他 worker
            break
        seq, payload, frame_raw = item
        out_frame, log_line, counts, any_det, ts = process_payload_cpu(
            payload, frame_raw, seq, fps,
            ctx.CLASS_CONFIG, ctx.CONF_THRESHOLDS, ctx.CLASS_NAMES, ctx.COLORS,
            ctx.BALL_CLASS_ID, ctx.TABLE_CLASS_ID, ctx.NET_CLASS_ID,
            ctx.mask_proc, out_w, out_h, ctx.want_polygons
        )
        with done_lock:
            done_dict[seq] = (out_frame, log_line, counts, any_det, ts)
            done_cv.notify_all()


def _ordering_writer(write_queue, log_queue, done_dict, done_lock, done_cv,
                     total_expected_holder, state):
    """
    按 seq 递增顺序从 done_dict 取结果 → 保证视频帧写入顺序正确。
    total_expected_holder[0] 为总帧数 (推理结束后由主线程写入; -1 表示未知/未完)。
    """
    next_seq = 0
    while True:
        with done_lock:
            while next_seq not in done_dict:
                total = total_expected_holder[0]
                if total >= 0 and next_seq >= total:
                    return
                done_cv.wait(timeout=1.0)
                if next_seq not in done_dict:
                    total = total_expected_holder[0]
                    if total >= 0 and next_seq >= total:
                        return
                    continue
            out_frame, log_line, counts, any_det, ts = done_dict.pop(next_seq)

        write_queue.put(out_frame)
        log_queue.put(log_line)
        for cls_id, c in counts.items():
            if c > 0:
                state["total_class_counts"][cls_id] += c
                state["class_frame_counts"][cls_id].append(ts)
        if not any_det:
            state["empty_frames"] += 1
        next_seq += 1


# ==================== process_one_video ====================
def process_one_video(video_path, args, ctx, output_dir=None):
    try:
        output_dir_or_ctx = output_dir if output_dir is not None else ctx.OUTPUT_DIR

        print(f"\n📹 Video: {video_path}")
        if is_variable_frame_rate(video_path):
            print("⚠️ Variable Frame Rate (VFR) detected!")
            cfr_video_path = os.path.join(output_dir_or_ctx, "video_cfr.mp4")
            video_input_path = convert_vfr_to_cfr(
                video_path, cfr_video_path,
                highlights_pct=args.cfr_highlights,
                contrast_pct=args.cfr_contrast,
                saturation_pct=args.cfr_saturation,
            )
        else:
            print("✅ Constant Frame Rate (CFR) confirmed.")
            video_input_path = video_path

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_width  = args.out_width
        out_height = max(2, int(round(height * out_width / width)))
        if out_height % 2 != 0:
            out_height += 1

        output_video_path = os.path.join(output_dir_or_ctx, "pingpong_detection.mp4")
        if args.writer_backend == "ffmpeg" and shutil.which("ffmpeg"):
            out_video = FFmpegWriter(output_video_path, out_width, out_height, fps,
                                     prefer=args.encoder, amf_quality=args.amf_quality,
                                     amf_qp=args.amf_qp)
            if out_video.encoder:
                print(f"🎬 FFmpeg pipe writer ({out_video.encoder}, "
                      f"quality={args.amf_quality}, qp={args.amf_qp}) "
                      f"@ {out_width}x{out_height} fps={fps}")
            else:
                print(f"🎬 FFmpeg pipe writer (libx264 software fallback) "
                      f"@ {out_width}x{out_height} fps={fps}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps,
                                        (out_width, out_height))
            print(f"🎬 OpenCV writer @ {out_width}x{out_height} fps={fps}")

        state = {
            "total_class_counts": {cls_id: 0 for cls_id in ctx.CLASS_CONFIG},
            "class_frame_counts": defaultdict(list),
            "empty_frames":       0,
            "out_width":          out_width,
            "out_height":         out_height,
        }

        # ---- 队列 ----
        read_queue  = Queue(maxsize=max(8, args.prefetch_batches))
        write_queue = Queue(maxsize=512)
        raw_log_path = os.path.join(output_dir_or_ctx, "raw_detections.jsonl")
        raw_log_f = open(raw_log_path, "w", encoding="utf-8")
        log_queue = Queue(maxsize=8000)
        reader_stop = threading.Event()

        # [v8 PERF] 后处理相关: 输入队列 + 有序结果字典
        post_in_q = Queue(maxsize=args.post_workers * ctx.BATCH_SIZE * 2)
        done_dict = {}
        done_lock = threading.Lock()
        done_cv = threading.Condition(done_lock)
        total_expected_holder = [-1]  # 推理完成后写入实际帧数

        # ---- 解码线程 (单/多) ----
        reader_threads = []
        use_segmented = (args.reader_threads > 1 and total_frames > 0)
        if use_segmented:
            n_seg = args.reader_threads
            seg_size = (total_frames + n_seg - 1) // n_seg
            # 分段并行解码, 每段结果标全局 idx; 顺序由后处理 done_dict 保证
            for s in range(n_seg):
                seg_start = s * seg_size
                seg_end = min(total_frames, seg_start + seg_size)
                if seg_start >= seg_end:
                    break
                t = threading.Thread(
                    target=frame_reader_segmented,
                    args=(video_input_path, read_queue, ctx.BATCH_SIZE,
                          reader_stop, seg_start, seg_end, seg_start),
                    daemon=True)
                reader_threads.append(t)
            # 分段模式下 cap 只用来读元信息, 关掉
            cap.release()
            cap = None
        else:
            t = threading.Thread(target=frame_reader,
                                 args=(cap, read_queue, ctx.BATCH_SIZE, reader_stop),
                                 daemon=True)
            reader_threads.append(t)

        t_writer = threading.Thread(target=frame_writer,
                                    args=(out_video, write_queue), daemon=True)
        t_logger = threading.Thread(target=log_writer,
                                    args=(raw_log_f, log_queue, args.log_flush_interval),
                                    daemon=True)

        # [v8 PERF] 后处理 worker 池 + 排序写盘线程
        post_workers = []
        for _ in range(max(1, args.post_workers)):
            wt = threading.Thread(target=_post_worker,
                                  args=(ctx, fps, out_width, out_height,
                                        post_in_q, done_dict, done_lock, done_cv),
                                  daemon=True)
            post_workers.append(wt)
        t_order = threading.Thread(target=_ordering_writer,
                                   args=(write_queue, log_queue, done_dict,
                                         done_lock, done_cv,
                                         total_expected_holder, state),
                                   daemon=True)

        for t in reader_threads:
            t.start()
        t_writer.start(); t_logger.start(); t_order.start()
        for wt in post_workers:
            wt.start()

        if total_frames > 0:
            print(f"\n🚀 开始推理: {total_frames:,} 帧, batch={ctx.BATCH_SIZE}, "
                  f"half={ctx.use_half}, imgsz={ctx.IMGSZ}, "
                  f"post_workers={args.post_workers}, "
                  f"reader_threads={args.reader_threads}")
        else:
            print(f"\n🚀 开始推理 (总帧数未知), batch={ctx.BATCH_SIZE}")

        frame_idx = 0
        processed_count = 0
        start_time = datetime.now()
        last_log_time = start_time
        last_log_idx  = 0
        ewma_fps = None
        batch_count = 0
        # 记录已 dispatch 到后处理的最大 seq, 用于最终确定 total_expected
        max_seq_seen = -1
        finished_readers = 0
        n_readers = len(reader_threads)

        try:
            while True:
                batch_item = read_queue.get()
                if batch_item is None:
                    finished_readers += 1
                    # 所有 reader 都发了结束信号才退出
                    if finished_readers >= n_readers:
                        break
                    continue

                frames_in_batch = [item[1] for item in batch_item]
                actual_n = len(frames_in_batch)
                pad_n = 0
                if 0 < actual_n < ctx.BATCH_SIZE:
                    pad_n = ctx.BATCH_SIZE - actual_n
                    # 必须用与真实帧同 H,W 的零帧: 方形零帧 (IMGSZ,IMGSZ,3) 会让 ultralytics 的
                    # same_shapes=False → LetterBox auto=False → 补齐成方形 (如 960x960), 与正常批的
                    # 矩形 (如 544x960) 形状不同 → MIOpen 尾批重新编译内核 (卡很久) 且多算 1.76x 像素。
                    frames_in_batch = frames_in_batch + [np.zeros_like(frames_in_batch[0])] * pad_n

                # [v8 PERF] 主线程只做 GPU 推理
                with torch.inference_mode():
                    results = ctx.model.predict(
                        frames_in_batch,
                        conf=ctx.min_conf_for_predict, iou=args.iou,
                        verbose=False, imgsz=ctx.IMGSZ, augment=False,
                        device=ctx.device, quantize=16 if ctx.use_half else None, stream=False,
                    )
                if pad_n > 0:
                    results = results[:actual_n]

                # [v8 PERF] 一次性把 GPU 结果搬到 CPU (在本线程完成同步),
                # 然后把纯 CPU payload 丢给后处理线程池 → GPU 立刻去跑下一批
                for result, (orig_idx, frame_raw) in zip(results, batch_item):
                    payload = extract_result_cpu(
                        result, ctx.want_polygons, ctx.POLYGON_NEEDED_CLASS_IDS)
                    post_in_q.put((orig_idx, payload, frame_raw))
                    if orig_idx > max_seq_seen:
                        max_seq_seen = orig_idx
                    frame_idx = orig_idx
                    processed_count += 1

                batch_count += 1
                if args.empty_cache_every > 0 and ctx.device != "cpu" \
                   and batch_count % args.empty_cache_every == 0:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                now = datetime.now()
                if (now - last_log_time).total_seconds() >= 2.0 and processed_count > 0:
                    dt = (now - last_log_time).total_seconds()
                    instant_fps = (processed_count - last_log_idx) / max(dt, 1e-6)
                    ewma_fps = instant_fps if ewma_fps is None \
                               else 0.7 * ewma_fps + 0.3 * instant_fps
                    total_dets_now = sum(state["total_class_counts"].values())

                    if total_frames > 0:
                        pct = processed_count / total_frames * 100
                        bar_len = 30
                        filled = int(bar_len * processed_count / total_frames)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        eta = (total_frames - processed_count) / max(ewma_fps, 1e-6)
                        print(f"  [{bar}] {pct:5.1f}%  "
                              f"帧 {processed_count:,}/{total_frames:,}  "
                              f"{instant_fps:.1f} fps | ETA {eta:.0f}s  "
                              f"| 检出(已写): {total_dets_now}", flush=True)
                    else:
                        elapsed_total = (now - start_time).total_seconds()
                        print(f"  帧 {processed_count:,}  "
                              f"{instant_fps:.1f} fps (EWMA {ewma_fps:.1f})  "
                              f"已耗时 {elapsed_total:.0f}s", flush=True)

                    last_log_time = now
                    last_log_idx  = processed_count
        finally:
            reader_stop.set()

            # [v8 PERF] 告诉 ordering 线程总共有多少帧 (= 已 dispatch 的最大 seq + 1)
            # 分段模式下 seq 对应全局 idx, 连续覆盖 [0, total_frames)
            with done_lock:
                total_expected_holder[0] = max_seq_seen + 1
                done_cv.notify_all()

            # 结束后处理 worker
            post_in_q.put(None)

            # 等 ordering 线程把所有帧写完 (它会在 next_seq 达到 total 时退出)
            try: t_order.join(timeout=600)
            except Exception: pass

            # 收尾其余线程
            for wt in post_workers:
                try: wt.join(timeout=10)
                except Exception: pass
            for t in reader_threads:
                try: t.join(timeout=5)
                except Exception: pass

            write_queue.put(None)
            log_queue.put(None)
            try: t_writer.join(timeout=300)
            except Exception: pass
            try: t_logger.join(timeout=10)
            except Exception: pass
            try: raw_log_f.close()
            except Exception: pass
            if cap is not None:
                try: cap.release()
                except Exception: pass
            try: out_video.release()
            except Exception: pass

        if video_input_path != video_path and os.path.exists(video_input_path) \
           and args.remove_cfr:
            try: os.remove(video_input_path)
            except Exception: pass

        elapsed = (datetime.now() - start_time).total_seconds()
        avg_fps = processed_count / max(elapsed, 1e-6)
        print(f"\n⏱️  Total: {elapsed:.1f}s | Avg: {avg_fps:.2f} fps "
              f"(processed {processed_count:,} frames)")

        total_detections = sum(state["total_class_counts"].values())
        print(f"🔍 检测统计: 总检测数={total_detections}")
        for cls_id, count in state["total_class_counts"].items():
            if count > 0:
                print(f"    {ctx.CLASS_NAMES[cls_id]}: {count}")

        print(f"✅ Inference Complete! Output: {output_dir_or_ctx}")
        print(f"📄 raw_detections.jsonl: {raw_log_path}")
        print(f"🎬 pingpong_detection.mp4: {output_video_path}")
        return True, raw_log_path
    except Exception as e:
        print(f"❌ process_one_video error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ==================== 主流程 ====================
def main():
    args = parse_args()
    t0 = time.time()
    ctx = load_model_and_config(args)
    dt = time.time() - t0
    print(f"[BENCHMARK] model_loaded id={id(ctx.model)} time={dt:.1f}s")

    if args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            video_list = [line.strip() for line in f if line.strip()]
        if not video_list:
            print("❌ video-list-file 为空")
            sys.exit(1)
        failed = []
        for vp in video_list:
            stem = os.path.splitext(os.path.basename(vp))[0]
            out_dir = os.path.join(ctx.OUTPUT_DIR, f"pipeline_{stem}")
            os.makedirs(out_dir, exist_ok=True)
            try:
                ok, log_path = process_one_video(vp, args, ctx, output_dir=out_dir)
                if not ok:
                    failed.append(vp)
            except Exception as e:
                print(f"⚠️ 视频 {vp} 推理失败: {e}")
                failed.append(vp)
            finally:
                if ctx.device != "cpu":
                    torch.cuda.empty_cache()
        if failed:
            print(f"⚠️ 以下 {len(failed)} 个视频推理失败:")
            for f in failed:
                print(f"  - {f}")
            sys.exit(1)
    else:
        process_one_video(args.video, args, ctx)


if __name__ == "__main__":
    main()
