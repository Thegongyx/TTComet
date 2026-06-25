#!/usr/bin/env python3
"""
PingPong Video Inference v7 — AMD Radeon 8060S / Windows Native 深度优化
==========================================================
v6 (WSL/VAAPI) → v7 (Windows/AMF) 迁移修复:
  严重 Bug fixes:
    - [致命] convert_vfr_to_cfr 在 ffmpeg 8.1 下导致 video_cfr.mp4 播放卡顿:
        ffmpeg 8.x 改了 VFR→CFR 时戳处理, -r + -fps_mode cfr + -c:a copy
        组合会让音频保留原始 VFR PTS, 视频被强制 CFR, A/V 时戳错位 →
        Windows 媒体播放器表现为画面停顿。
        修复: 改用 -vf fps=N filter 重建时戳; 音频改 AAC 重编(不 copy);
        加 -fflags +genpts -vsync cfr -async 1 -video_track_timescale 90000
    - [致命] VAAPI(/dev/dri/renderD128, h264_vaapi) 在 Windows 不存在,
        改用 AMD AMF: h264_amf / hevc_amf / av1_amf (Windows 原生 AMD 硬编)
    - MIOpen cache 路径跨平台化 (Windows 使用 %LOCALAPPDATA%\\miopen)
    - 默认 --video / --model 路径改成 Windows 风格的占位符
  优化:
    - AMF 主路径用 -quality quality -rc cqp 获得最佳质量/体积平衡
    - 软编 fallback: veryfast/crf20
    - frame_writer join timeout 拉到 300s, 大视频不会被截断
    - 进程启动加 CREATE_NO_WINDOW 标志, 避免 Windows 下弹黑窗
  保留 v6 全部优化:
    - polygon 只对 table/net 提取
    - mask overlay cv2.addWeighted ROI
    - inference_mode、batch padding、EWMA fps、周期性显存清理 等
"""

# ---------------------------------------------------------------------------
# MIOpen / HIP 环境变量 (必须在 import torch 之前!)
# Windows: 使用 %LOCALAPPDATA%\miopen; Linux/WSL: 使用 ~/.cache/miopen
# ---------------------------------------------------------------------------
import os
import sys
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

# Windows 子进程不弹黑窗
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

# 这些类别在下游需要 polygon 轮廓 (mini 球台几何提取); 其余只取 bbox.
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
    parser = argparse.ArgumentParser(description="PingPong Video Inference v7 (Windows/AMF)")
    parser.add_argument("--video", "-v", type=str, default=_default_video())
    parser.add_argument("--model", "-m", type=str, default=_default_model())
    parser.add_argument("--output", "-o", type=str, default=_default_output())
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", action="store_true", help="FP16 推理 (ROCm/CUDA 加速)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="推理分辨率, 默认 960")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou",  type=float, default=0.45)
    parser.add_argument("--conf-ball",   type=float, default=0.30)
    parser.add_argument("--conf-player", type=float, default=None)
    parser.add_argument("--conf-racket", type=float, default=None)
    parser.add_argument("--conf-table",  type=float, default=None)
    parser.add_argument("--conf-net",    type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批大小, ROCm 建议 4-8; 尾批用零帧 padding 保持固定 shape")
    parser.add_argument("--remove-cfr", action="store_true",
                        help="VFR 转 CFR 后删除中间文件")
    parser.add_argument("--log-flush-interval", type=float, default=0.5,
                        help="log writer 自动 flush 间隔 (秒)")
    parser.add_argument("--writer-backend", type=str, default="ffmpeg",
                        choices=["ffmpeg", "opencv"],
                        help="视频写入后端")
    parser.add_argument("--out-width", type=int, default=640,
                        help="输出视频宽度 (默认 640, aspect 自动)")
    parser.add_argument("--no-polygon", action="store_true",
                        help="完全跳过 polygon 输出 (mini 球台依赖此特性, 关闭可省 IO)")
    parser.add_argument("--empty-cache-every", type=int, default=200,
                        help="每 N 批清一次 GPU 缓存 (0 关闭)")
    parser.add_argument("--conf-margin", type=float, default=0.05,
                        help="传给模型的 conf = min(各类阈值)-margin, 防止边缘类别被 GPU NMS 提前裁")
    # Windows AMF 编码器
    parser.add_argument("--encoder", type=str, default="auto",
                        choices=["auto", "libx264", "h264_amf", "hevc_amf", "av1_amf"],
                        help="输出视频编码器; auto=优先探测 AMD AMF 硬编")
    parser.add_argument("--amf-quality", type=str, default="quality",
                        choices=["speed", "balanced", "quality"],
                        help="AMF -quality 参数 (speed=最快, quality=最好)")
    parser.add_argument("--amf-qp", type=int, default=20,
                        help="AMF CQP 量化参数 (越小越清晰, 18-28 常用)")
    parser.add_argument("--end2end", action="store_true",
                        help="YOLO26 走 NMS-free end2end 推理（更快，但 --iou 失效，单帧上限 max_det）。"
                             "默认关闭=走 one-to-many+NMS，--iou 生效、召回更高，适合多目标计数")
    parser.add_argument("--max-det", type=int, default=300,
                        help="单帧最大检测数（NMS-free 模式硬上限 300；乒乓球场景目标少，默认 300 足够）")
    return parser.parse_args()


# ==================== 硬件加速探测 (AMF for AMD on Windows) ====================
def _detect_amf_encoder(prefer="auto"):
    """返回 (encoder_name, None) 或 (None, None)。仅检测并返回推荐配置。
    Windows AMD GPU 用 AMF, 不需要设备路径 (内部走 DXGI/D3D11)。
    """
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
    # 探测一次编码器是否真的可用 (避免列表里有但驱动没启用)
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

        # 解析分数 / 浮点; 容忍 1% 误差
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
    """检测视频是否含有音频流; 没有的话 ffmpeg 加 -c:a 会报错。"""
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


def convert_vfr_to_cfr(video_path, output_path, target_fps=None):
    """
    [关键修复 v7]
    ffmpeg 8.1 (Windows) 下原来的 `-r N -fps_mode cfr -c:a copy` 组合会让
      - 视频流被强制 CFR (PTS = idx * 1/fps)
      - 音频流仍保留原始 VFR PTS
    两条流时戳基准不同 → A/V 不同步 → Windows Media Player / 浏览器表现为画面卡顿。

    修复策略:
      1. 用 `-vf fps={fps}` filter 在 filter 层显式重采样到固定帧率,
         比 `-r` 输出选项更可靠 (filter 内部会丢/复帧, 时戳干净)
      2. 加 `-fflags +genpts` 让 ffmpeg 重新生成 demuxer 端 PTS
      3. 加 `-vsync cfr` (新版叫 -fps_mode cfr, 这里两个都给) 显式 CFR
      4. 音频改 `-c:a aac -b:a 192k -ar 48000 -async 1` 重编码 + 时戳软对齐,
         不再 copy 原始时戳 (这是问题根源)
      5. `-video_track_timescale 90000` 固定 MP4 时基 = 90000 Hz,
         避免 Windows 播放器对小时基视频的解码毛刺
    """
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

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts",
        "-i", video_path,
        "-vf", f"fps={target_fps}",
        "-vsync", "cfr",            # 旧名 (ffmpeg <= 7 兼容)
        "-fps_mode", "cfr",         # 新名 (ffmpeg 8.x 推荐)
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
    ]
    if has_audio:
        cmd += [
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-async", "1",
        ]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, timeout=3600, creationflags=_SUBPROC_FLAGS)
    return output_path


# ==================== ffmpeg 管道写入器 ====================
class FFmpegWriter:
    """ffmpeg 管道 BGR raw → 优先 AMD AMF 硬编 / 回退 libx264 软编。
    在 Windows + AMD Radeon 8060S 上默认使用 h264_amf, 写入耗时减 2-4x。
    """
    def __init__(self, out_path, width, height, fps, prefer="auto",
                 amf_quality="quality", amf_qp=24):
        self.encoder = None
        enc, _ = _detect_amf_encoder(prefer=prefer)

        # 通用输入参数 (BGR raw pipe)
        input_args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]

        if enc:
            self.encoder = enc
            # AMF: D3D11 后端, 不需要设备节点
            # -usage transcoding: 高质量转码场景 (vs ultralowlatency / lowlatency)
            # -quality {speed|balanced|quality}
            # -rc cqp + -qp_i/p/b: 固定 QP, 质量稳定
            # -bf 0: 不用 B 帧, 兼容性好 (有些 AMD 驱动 B 帧会绿屏)
            qp = int(amf_qp)
            enc_args = [
                "-c:v", enc,
                "-usage", "transcoding",
                "-quality", amf_quality,
                "-rc", "cqp",
                "-qp_i", str(qp),
                "-qp_p", str(min(51, qp + 2)),
                "-bf", "0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-video_track_timescale", "90000",
                out_path,
            ]
            # av1_amf 不支持 qp_p 参数, 单独处理
            if enc == "av1_amf":
                enc_args = [
                    "-c:v", enc,
                    "-usage", "transcoding",
                    "-quality", amf_quality,
                    "-rc", "cqp",
                    "-qp_i", str(qp),
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-video_track_timescale", "90000",
                    out_path,
                ]
            cmd = input_args + enc_args
        else:
            # 软编 fallback: veryfast/crf20
            cmd = input_args + [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-threads", "0",
                "-movflags", "+faststart",
                "-video_track_timescale", "90000",
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
            # 大视频编码 flush 可能需要很久, 给到 5 分钟
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
    """
    用 cv2.addWeighted 替代 fancy indexing。
    对 ROI 做加权混合 → 整体快 2-5x (依 mask 占比)。
    """
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
    # 仅写入 mask 内像素
    sub_frame[sub_mask] = blended[sub_mask]
    return frame


# ==================== I/O 线程 ====================
def frame_reader(cap, q, batch_size, stop_evt):
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
        # 出错时向主线程传递 None, 避免主循环永久等待
        try:
            q.put(None)
        except Exception:
            pass
        print(f"⚠️ frame_reader error: {e}")


def frame_writer(writer, q):
    while True:
        item = q.get()
        if item is None:
            break
        writer.write(item)


def log_writer(log_f, q, flush_interval=0.5):
    """按时间间隔 flush (不再依赖固定 buffer 大小, 对短视频更友好)。"""
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
    """对 table/net mask 做 resize → uint8 二值。"""
    def resize_masks(self, masks_np, cls_ids,
                     out_hw, table_cls_id, net_cls_id):
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


# ==================== 单帧处理 ====================
def process_single_result(
    result, frame_raw, frame_idx, fps,
    CLASS_CONFIG, CONF_THRESHOLDS, CLASS_NAMES, COLORS,
    BALL_CLASS_ID, TABLE_CLASS_ID, NET_CLASS_ID,
    POLYGON_NEEDED_CLASS_IDS,  # set of cls_id
    mask_proc, state, want_polygons: bool
):
    boxes = result.boxes
    masks = result.masks
    out_h = state["out_height"]
    out_w = state["out_width"]
    h_raw, w_raw = frame_raw.shape[:2]
    sx = out_w / w_raw
    sy = out_h / h_raw

    frame_class_counts_local = {cls_id: 0 for cls_id in CLASS_CONFIG}
    frame_detections = []
    best_detections = {}

    # ---- 一次性下载 boxes + masks ----
    masks_np = None
    cls_ids_np = None
    if boxes is not None and len(boxes) > 0:
        all_xyxy = boxes.xyxy.cpu().numpy()
        all_conf = boxes.conf.cpu().numpy()
        all_cls  = boxes.cls.cpu().numpy().astype(np.int32)
        n = len(all_cls)
        cls_ids_np = all_cls

        if masks is not None and masks.data is not None and len(masks.data) > 0:
            masks_np = masks.data.cpu().numpy()  # 一次下载

        # 仅对 polygon 需求类别提取 polygon — 跳过 ball/player/racket
        polygons_by_idx = {}
        if want_polygons and masks_np is not None:
            need_polygon_idx = [
                i for i in range(n)
                if int(all_cls[i]) in POLYGON_NEEDED_CLASS_IDS
            ]
            if need_polygon_idx:
                try:
                    xy_list = masks.xy
                    for i in need_polygon_idx:
                        if i < len(xy_list):
                            poly = xy_list[i]
                            if poly is not None and len(poly) > 0:
                                polygons_by_idx[i] = [
                                    [round(float(x), 1), round(float(y), 1)]
                                    for x, y in poly
                                ]
                except Exception:
                    pass

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
            state["total_class_counts"][cls_id] += 1
            if cls_id not in best_detections or conf > best_detections[cls_id]["confidence"]:
                best_detections[cls_id] = det

    # ---- table/net mask overlay ----
    if masks_np is not None and cls_ids_np is not None:
        table_mask_u8, net_mask_u8 = mask_proc.resize_masks(
            masks_np, cls_ids_np, (out_h, out_w),
            TABLE_CLASS_ID, NET_CLASS_ID)
    else:
        table_mask_u8 = net_mask_u8 = None

    # ---- ball 中心 (供日志) ----
    ball_det = best_detections.get(BALL_CLASS_ID)
    bx = by = 0.0
    ball_pos = None
    if ball_det:
        bx, by = get_bbox_center(ball_det["bbox"])
        ball_pos = (bx, by)

    # ---- 可视化输出帧 ----
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

    # ---- 日志 ----
    timestamp = frame_idx / fps
    log_obj = {
        "frame": frame_idx,
        "timestamp": round(timestamp, 2),
        "counts": {CLASS_NAMES[cls_id]: frame_class_counts_local[cls_id]
                   for cls_id in CLASS_NAMES},
        "ball_pos": (round(bx, 1), round(by, 1)) if ball_pos else None,
        "detections": [
            {
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox": [round(x, 1) for x in d["bbox"]],
                "polygon": d.get("polygon")
            }
            for d in frame_detections
        ]
    }
    log_line = json.dumps(log_obj) + "\n"

    any_det = False
    for cls_id, count in frame_class_counts_local.items():
        if count > 0:
            state["class_frame_counts"][cls_id].append(timestamp)
            any_det = True
    if not any_det:
        state["empty_frames"] += 1

    return out_frame, log_line


# ==================== 主流程 ====================
def main():
    args = parse_args()

    if args.output.endswith(".json"):
        OUTPUT_DIR = os.path.dirname(args.output) or "."
        LOG_FILENAME = os.path.basename(args.output)
    else:
        OUTPUT_DIR = args.output
        LOG_FILENAME = "detection_log.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    VIDEO_PATH = args.video
    MODEL_PATH = args.model
    BATCH_SIZE = args.batch_size
    IMGSZ      = args.imgsz

    print("=" * 60)
    print(f"🏓 PingPong Video Inference v7 [Platform: {platform.system()}]")
    print("=" * 60)

    # ---- 设备 ----
    device = args.device
    use_half = args.half
    if device != "cpu" and not torch.cuda.is_available():
        print("⚠️ ROCm/GPU not detected, falling back to CPU")
        device = "cpu"
        use_half = False
    if device == "cpu":
        use_half = False

    if device != "cpu":
        # ROCm 不支持 TF32, 设置无害; cuDNN benchmark 在 ROCm 下也能选走最优 kernel
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 打印 GPU 信息便于确认是 Radeon 8060S
    if device != "cpu":
        try:
            dev_idx = int(device) if device.isdigit() else 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            print(f"GPU: {gpu_name}")
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

    # ---- 类别映射 (只在启动时计算一次) ----
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
        print("ℹ️ Polygon 输出已禁用 (--no-polygon 或模型不含 table/net 类别)")
    else:
        print(f"ℹ️ Polygon 仅对类别 {sorted(POLYGON_NEEDED_CLASS_IDS)} 提取, "
              f"跳过其他类别以节省 CPU")

    # ---- VFR 处理 ----
    print(f"\n📹 Video: {VIDEO_PATH}")
    if is_variable_frame_rate(VIDEO_PATH):
        print("⚠️ Variable Frame Rate (VFR) detected!")
        cfr_video_path = os.path.join(OUTPUT_DIR, "video_cfr.mp4")
        video_input_path = convert_vfr_to_cfr(VIDEO_PATH, cfr_video_path)
    else:
        print("✅ Constant Frame Rate (CFR) confirmed.")
        video_input_path = VIDEO_PATH

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

    output_video_path = os.path.join(OUTPUT_DIR, "pingpong_detection.mp4")
    if args.writer_backend == "ffmpeg" and shutil.which("ffmpeg"):
        out_video = FFmpegWriter(output_video_path, out_width, out_height, fps,
                                 prefer=args.encoder,
                                 amf_quality=args.amf_quality,
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

    # ---- 状态 ----
    state = {
        "total_class_counts": {cls_id: 0 for cls_id in CLASS_CONFIG},
        "class_frame_counts": defaultdict(list),
        "empty_frames":       0,
        "out_width":          out_width,
        "out_height":         out_height,
    }
    mask_proc = MaskProcessor()

    # ---- 流水线 ----
    read_queue  = Queue(maxsize=max(8, BATCH_SIZE * 4))
    write_queue = Queue(maxsize=240)
    raw_log_path = os.path.join(OUTPUT_DIR, "raw_detections.jsonl")
    raw_log_f = open(raw_log_path, "w", encoding="utf-8")
    log_queue = Queue(maxsize=4000)
    reader_stop = threading.Event()

    t_reader = threading.Thread(target=frame_reader,
                                args=(cap, read_queue, BATCH_SIZE, reader_stop),
                                daemon=True)
    t_writer = threading.Thread(target=frame_writer,
                                args=(out_video, write_queue), daemon=True)
    t_logger = threading.Thread(target=log_writer,
                                args=(raw_log_f, log_queue, args.log_flush_interval),
                                daemon=True)
    t_reader.start(); t_writer.start(); t_logger.start()

    if total_frames > 0:
        print(f"\n🚀 开始推理: {total_frames:,} 帧, batch={BATCH_SIZE}, "
              f"half={use_half}, imgsz={IMGSZ}")
    else:
        print(f"\n🚀 开始推理 (总帧数未知), batch={BATCH_SIZE}, "
              f"half={use_half}, imgsz={IMGSZ}")
    frame_idx = 0
    start_time = datetime.now()
    last_log_time = start_time
    last_log_idx  = 0
    ewma_fps = None
    batch_count = 0

    # 预分配 padding 用的零帧 —— 小于 imgsz 不影响 letterbox, 使用 imgsz×imgsz 节省传输
    pad_frame_zero = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)

    # 传给 GPU 端 NMS 的 conf: 取最小阈值减 margin, 避免边缘类别被提前刪除
    min_conf_for_predict = max(0.01,
                               min(CONF_THRESHOLDS.values()) - float(args.conf_margin))

    try:
        while True:
            batch_item = read_queue.get()
            if batch_item is None:
                break

            frames_in_batch = [item[1] for item in batch_item]
            actual_n = len(frames_in_batch)
            pad_n = 0
            if 0 < actual_n < BATCH_SIZE:
                pad_n = BATCH_SIZE - actual_n
                frames_in_batch = frames_in_batch + [pad_frame_zero] * pad_n

            with torch.inference_mode():
                results = model.predict(
                    frames_in_batch,
                    conf=min_conf_for_predict, iou=args.iou,
                    verbose=False, imgsz=IMGSZ, augment=False,
                    device=device, half=use_half,
                    stream=False,
                    # end2end=args.end2end,   # 新增：与训练验证保持一致
                    # max_det=args.max_det,   # 新增：避免 NMS-free 模式漏检
                )
            if pad_n > 0:
                results = results[:actual_n]

            for result, (orig_idx, frame_raw) in zip(results, batch_item):
                frame_idx = orig_idx
                out_frame, log_line = process_single_result(
                    result, frame_raw, frame_idx, fps,
                    CLASS_CONFIG, CONF_THRESHOLDS, CLASS_NAMES, COLORS,
                    BALL_CLASS_ID, TABLE_CLASS_ID, NET_CLASS_ID,
                    POLYGON_NEEDED_CLASS_IDS,
                    mask_proc, state, want_polygons
                )
                write_queue.put(out_frame)
                log_queue.put(log_line)

            batch_count += 1
            if args.empty_cache_every > 0 and device != "cpu" \
               and batch_count % args.empty_cache_every == 0:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # ---- 进度条 (每 ~2s 刷新, 含可视化进度条 + EWMA fps + 检测统计) ----
            now = datetime.now()
            if (now - last_log_time).total_seconds() >= 2.0 and frame_idx > 0:
                dt = (now - last_log_time).total_seconds()
                instant_fps = (frame_idx - last_log_idx) / max(dt, 1e-6)
                ewma_fps = instant_fps if ewma_fps is None \
                           else 0.7 * ewma_fps + 0.3 * instant_fps
                total_dets_now = sum(state["total_class_counts"].values())

                if total_frames > 0:
                    pct = frame_idx / total_frames * 100
                    bar_len = 30
                    filled = int(bar_len * frame_idx / total_frames)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    eta = (total_frames - frame_idx) / max(ewma_fps, 1e-6)
                    print(f"  [{bar}] {pct:5.1f}%  "
                          f"帧 {frame_idx:,}/{total_frames:,}  "
                          f"{instant_fps:.1f} fps | ETA {eta:.0f}s  "
                          f"| 检出: {total_dets_now}", flush=True)
                else:
                    elapsed_total = (now - start_time).total_seconds()
                    print(f"  帧 {frame_idx:,}  "
                          f"{instant_fps:.1f} fps (EWMA {ewma_fps:.1f})  "
                          f"已耗时 {elapsed_total:.0f}s  "
                          f"| 检出: {total_dets_now}", flush=True)

                last_log_time = now
                last_log_idx  = frame_idx
    finally:
        # 通知子线程退出
        reader_stop.set()
        write_queue.put(None)
        log_queue.put(None)
        try: t_reader.join(timeout=5)
        except Exception: pass
        try: t_writer.join(timeout=300)   # 大视频 ffmpeg flush 可能久
        except Exception: pass
        try: t_logger.join(timeout=10)
        except Exception: pass
        # 必须在 t_logger 退出后才 close (避免在 logger 还在写时被关)
        try: raw_log_f.close()
        except Exception: pass
        try: cap.release()
        except Exception: pass
        try: out_video.release()
        except Exception: pass

    if video_input_path != VIDEO_PATH and os.path.exists(video_input_path) \
       and args.remove_cfr:
        try: os.remove(video_input_path)
        except Exception: pass

    elapsed = (datetime.now() - start_time).total_seconds()
    avg_fps = frame_idx / max(elapsed, 1e-6)
    print(f"\n⏱️  Total: {elapsed:.1f}s | Avg: {avg_fps:.2f} fps")

    total_detections = sum(state["total_class_counts"].values())
    print(f"🔍 检测统计: 总检测数={total_detections}")
    for cls_id, count in state["total_class_counts"].items():
        if count > 0:
            print(f"    {CLASS_NAMES[cls_id]}: {count}")

    print(f"✅ Inference Complete! Output: {OUTPUT_DIR}")
    print(f"📄 raw_detections.jsonl: {raw_log_path}")
    print(f"🎬 pingpong_detection.mp4: {output_video_path}")


if __name__ == "__main__":
    main()