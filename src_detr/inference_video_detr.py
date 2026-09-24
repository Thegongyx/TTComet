#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR 乒乓球逐帧推理脚本
============================================================
用 RF-DETR (https://github.com/roboflow/rf-detr) 替代 YOLO26 推理。

输出与 src/inference_video.py 完全一致, 因此可直接被 src/pingpong_highlight.py 消费:
    <output>/raw_detections.jsonl   每行:
        {
          "frame": int,
          "timestamp": float,
          "counts": {"乒乓球": n, ...},
          "ball_pos": [x, y] | null,
          "detections": [
             {"class": "球桌", "confidence": 0.93,
              "bbox": [x1, y1, x2, y2],
              "polygon": [[x, y], ...] | null}
          ]
        }
    <output>/pingpong_detection.mp4  可视化检测视频

* 分割模型 (RFDETRSeg*) 会输出球桌/球网的 polygon (由 mask 轮廓提取)。
* 检测模型 (RFDETR*) 只输出 bbox, polygon 为 null。
* 类别名沿用 inference_video.py 的别名表 (含中文), 因此同一套高光逻辑可直接复用。

用法示例 (conda 环境 rfdetr)
------------------------------------------------------------
conda run -n rfdetr python src_detr/inference_video_detr.py `
    --model runs_detr/pingpong_detr_v1/checkpoint_best_total.pth `
    --video Videos/match.mp4 `
    --output inference_output/detr_match

# 批处理: 文本文件每行一个视频绝对路径
conda run -n rfdetr python src_detr/inference_video_detr.py `
    --model <ckpt.pth> --video-list-file tmp/video_list.txt --output inference_output
"""

# ---------------------------------------------------------------------------
# MIOpen / HIP 环境变量 (必须在 import torch 之前)
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
os.environ.setdefault("MIOPEN_USER_DB_PATH",     _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_FIND_MODE",        "FAST")
os.environ.setdefault("MIOPEN_FIND_ENFORCE",     "NONE")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",  "expandable_segments:True")

# AMD ROCm: 启用 Flash / Mem-Efficient Attention 的实验内核。
# 实测可见 ViT attention 从 math 回退路径切到融合 kernel, fp16 推理 ~20.7 → ~28.9 fps。
# 用 --no-aotriton 关闭 (必须在 import torch 之前设置)。
if "--no-aotriton" not in sys.argv:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import cv2
import json
import shutil
import argparse
import subprocess
import numpy as np
import torch
from datetime import datetime

try:
    cv2.setNumThreads(2)
except Exception:
    pass

_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0


# ==================== 类别配置 (与 inference_video.py 保持一致) ====================
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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_output():
    if _IS_WINDOWS:
        return os.path.join(os.path.expanduser("~"), "yolo26_frames", "inference_output_detr")
    return "/home/gongy/yolo26_frames/inference_output_detr"


def _default_model():
    """默认模型: 优先项目根目录的 rf-detr-seg-*.pt (与 yolo26*.pt 同位置)。"""
    for name in ("rf-detr-seg-medium.pt", "rf-detr-seg-small.pt", "rf-detr-seg-nano.pt"):
        p = os.path.join(_PROJECT_ROOT, name)
        if os.path.exists(p):
            return p
    return "rf-detr-seg-medium.pt"


def parse_args():
    p = argparse.ArgumentParser(description="RF-DETR 乒乓球逐帧推理")
    p.add_argument("--video", "-v", type=str, default=None, help="输入视频路径")
    p.add_argument("--video-list-file", type=str, default=None,
                   help="批处理: UTF-8 文本, 每行一个视频绝对路径")
    p.add_argument("--model", "-m", type=str, default=_default_model(),
                   help="模型路径: 训练 checkpoint(.pth) 或官方预训练权重(.pt, "
                        "如 rf-detr-seg-medium.pt / 项目根目录权重); "
                        "默认=项目根目录 rf-detr-seg-medium.pt")
    p.add_argument("--size", type=str, default=None,
                   choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
                   help="当 --model 指向官方权重名/本地路径而非 checkpoint 时, 指定模型规模")
    p.add_argument("--task", type=str, default=None, choices=["seg", "detect"],
                   help="仅当直接构造官方模型时需要; checkpoint 会自动推断")
    p.add_argument("--output", "-o", type=str, default=_default_output())
    p.add_argument("--device", type=str, default="0", help="0 / cuda:0 / cpu")
    p.add_argument("--conf", type=float, default=0.30, help="全局基础置信度阈值")
    p.add_argument("--conf-ball", type=float, default=0.30)
    p.add_argument("--conf-player", type=float, default=None)
    p.add_argument("--conf-racket", type=float, default=None)
    p.add_argument("--conf-table", type=float, default=None)
    p.add_argument("--conf-net", type=float, default=None)
    p.add_argument("--conf-margin", type=float, default=0.05,
                   help="传给模型的 conf 裕量 (低阈值预测, 后处理再按类别过滤)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="推理批大小 (默认 32)。显存充足可到 64/128 (实测 8060S/107GB 平台期 ~53fps); "
                        "1080p 多目标场景可适当调小以省后处理开销")
    p.add_argument("--out-width", type=int, default=640, help="可视化视频宽度")
    p.add_argument("--no-polygon", action="store_true", help="不输出 polygon")
    p.add_argument("--poly-approx", type=float, default=0.002,
                   help="mask->polygon 的 approxPolyDP epsilon 比例 (越小点越多)")
    p.add_argument("--remove-cfr", action="store_true", help="推理后删除 CFR 中间视频")
    p.add_argument("--writer-backend", type=str, default="ffmpeg",
                   choices=["ffmpeg", "opencv"])
    p.add_argument("--encoder", type=str, default="auto",
                   choices=["auto", "libx264", "h264_amf", "hevc_amf", "av1_amf"])
    p.add_argument("--amf-quality", type=str, default="quality",
                   choices=["speed", "balanced", "quality"])
    p.add_argument("--amf-qp", type=int, default=20)
    p.add_argument("--cfr-highlights", type=float, default=10.0)
    p.add_argument("--cfr-contrast", type=float, default=10.0)
    p.add_argument("--cfr-saturation", type=float, default=6.0)
    p.add_argument("--log-flush-interval", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0,
                   help="仅处理前 N 帧 (0=全部; 快速测试用)")
    p.add_argument("--optimize", type=str, default="fp16",
                   choices=["none", "export", "fp16"],
                   help="推理优化 (官方 model.inference): none=基线; export=融合导出(fp32); "
                        "fp16=融合导出+FP16 (最快, 默认)。实测 AMD 8060S 上 fp16+AOTRITON 约 2.3x")
    p.add_argument("--no-aotriton", action="store_true",
                   help="关闭 AMD Flash/Mem-Efficient Attention 实验内核 (默认开启)")
    # ---- YOLO 兼容参数 (同名接受; 能映射的映射, 不支持的打印忽略警告) ----
    p.add_argument("--half", action="store_true",
                   help="YOLO 兼容: 映射为 FP16 推理 (等价 --optimize fp16)")
    p.add_argument("--imgsz", type=int, default=None,
                   help="YOLO 兼容: 推理分辨率 (映射到 RF-DETR model.resolution)")
    p.add_argument("--iou", type=float, default=None,
                   help="[YOLO 兼容, 忽略] NMS IoU 阈值 (RF-DETR NMS-free)")
    p.add_argument("--max-det", type=int, default=None,
                   help="[YOLO 兼容, 忽略] 最大检测数")
    p.add_argument("--empty-cache-every", type=int, default=None,
                   help="[YOLO 兼容, 忽略] 每 N 批清 GPU 缓存")
    p.add_argument("--post-workers", type=int, default=None,
                   help="[YOLO 兼容, 忽略] CPU 后处理线程数")
    p.add_argument("--reader-threads", type=int, default=None,
                   help="[YOLO 兼容, 忽略] 视频解码线程数")
    p.add_argument("--prefetch-batches", type=int, default=None,
                   help="[YOLO 兼容, 忽略] 预取批数")
    p.add_argument("--end2end", action="store_true",
                   help="[YOLO 兼容, 忽略] RF-DETR 始终 NMS-free")
    return p.parse_args()


# 仅接受并忽略的 YOLO 兼容参数 -> (中文描述)
_IGNORED_YOLO_ARGS = [
    ("iou", "NMS IoU 阈值"),
    ("max_det", "最大检测数"),
    ("empty_cache_every", "每 N 批清 GPU 缓存"),
    ("post_workers", "CPU 后处理线程数"),
    ("reader_threads", "视频解码线程数"),
    ("prefetch_batches", "预取批数"),
]


def _warn_ignored_yolo_args(args):
    for name, _desc in _IGNORED_YOLO_ARGS:
        if getattr(args, name, None) is None:
            continue
        print(f"⚠️ RF-DETR 不支持参数 --{name.replace('_', '-')}，已忽略")
    if getattr(args, "end2end", False):
        print("⚠️ RF-DETR 不支持参数 --end2end，已忽略")
    # --half -> 映射为 FP16 (默认已是 fp16)
    if getattr(args, "half", False):
        args.optimize = "fp16"
        print("ℹ️ --half 已映射为 RF-DETR 的 FP16 推理 (--optimize fp16)")


# ==================== 硬件加速探测 (AMF) ====================
def _detect_amf_encoder(prefer="auto"):
    if prefer == "libx264":
        return None
    if not shutil.which("ffmpeg"):
        return None
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=8,
                           creationflags=_SUBPROC_FLAGS)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None
    has_h264 = "h264_amf" in out
    has_hevc = "hevc_amf" in out
    has_av1 = "av1_amf" in out
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
        return None
    try:
        probe = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2:r=25",
            "-c:v", enc, "-f", "null", "-"
        ], capture_output=True, text=True, timeout=15, creationflags=_SUBPROC_FLAGS)
        if probe.returncode != 0:
            return None
    except Exception:
        return None
    return enc


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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                timeout=30, creationflags=_SUBPROC_FLAGS)

        def _to_f(x):
            try:
                if "/" in x:
                    a, b = x.split("/", 1)
                    return float(a) / float(b) if float(b) != 0 else 0.0
                return float(x)
            except Exception:
                return 0.0

        vals = {}
        for line in result.stdout.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
        r = _to_f(vals.get("r_frame_rate", "0"))
        a = _to_f(vals.get("avg_frame_rate", "0"))
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
            capture_output=True, text=True, timeout=15, creationflags=_SUBPROC_FLAGS)
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
        "-fflags", "+genpts", "-i", video_path,
        "-vf", vf_chain, "-vsync", "cfr", "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-video_track_timescale", "90000",
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
        enc = _detect_amf_encoder(prefer=prefer)
        input_args = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]
        if enc:
            self.encoder = enc
            qp = int(amf_qp)
            if enc == "av1_amf":
                enc_args = ["-c:v", enc, "-usage", "transcoding", "-quality", amf_quality,
                            "-rc", "cqp", "-qp_i", str(qp), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-video_track_timescale", "90000", out_path]
            else:
                enc_args = ["-c:v", enc, "-usage", "transcoding", "-quality", amf_quality,
                            "-rc", "cqp", "-qp_i", str(qp), "-qp_p", str(min(51, qp + 2)),
                            "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-video_track_timescale", "90000", out_path]
            cmd = input_args + enc_args
        else:
            cmd = input_args + [
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "0",
                "-movflags", "+faststart", "-video_track_timescale", "90000", out_path,
            ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            self.proc.wait(timeout=60)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ==================== 绘图 ====================
def draw_detections(frame, detections_list, colors):
    for det in detections_list:
        b = det["bbox"]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cls_id = det.get("class_id", 0)
        color = colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}",
                    (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
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
    sub_mask = bool_mask[y0:y1, x0:x1]
    if sub_frame.size == 0 or not sub_mask.any():
        return frame
    color_img = np.empty_like(sub_frame)
    color_img[:] = color
    blended = cv2.addWeighted(sub_frame, 1.0 - alpha, color_img, alpha, 0.0)
    sub_frame[sub_mask] = blended[sub_mask]
    return frame


# ==================== mask -> polygon ====================
def mask_to_polygon(mask_bool, approx_eps_ratio=0.002):
    """由 bool mask 提取最大外轮廓多边形。返回 [[x,y], ...] 或 None。"""
    if mask_bool is None or not mask_bool.any():
        return None
    m = (mask_bool.astype(np.uint8)) * 255
    try:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return None
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4.0:
        return None
    peri = cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, max(0.0, approx_eps_ratio) * peri, True)
    pts = ap.reshape(-1, 2) if ap is not None and len(ap) >= 4 else c.reshape(-1, 2)
    # 下采样避免点数过多 (高光剪辑只需要 >=4 点的轮廓)
    if len(pts) > 64:
        step = int(np.ceil(len(pts) / 64))
        pts = pts[::step]
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


# ==================== 模型加载 ====================
_SIZE_MAP_DETECT = {"nano": "RFDETRNano", "small": "RFDETRSmall",
                    "medium": "RFDETRMedium", "large": "RFDETRLarge",
                    "xlarge": "RFDETRXLarge", "2xlarge": "RFDETR2XLarge"}
_SIZE_MAP_SEG = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall",
                 "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge",
                 "xlarge": "RFDETRSegXLarge", "2xlarge": "RFDETRSeg2XLarge"}


def _read_class_names_from_sidecar(model_path):
    """从 checkpoint 同级的 training_config.json 读取类别名。"""
    try:
        cfg = os.path.join(os.path.dirname(os.path.abspath(model_path)), "training_config.json")
        if os.path.exists(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            names = data.get("class_names")
            if names:
                return list(names)
    except Exception:
        pass
    return None


def _infer_size_task_from_name(name):
    """从文件名推断 (size, task): 如 rf-detr-seg-medium.pt -> ('medium','seg')。"""
    n = name.lower()
    task = "seg" if "seg" in n else "detect"
    size = None
    for s in ("2xlarge", "xxlarge", "xlarge", "large", "medium", "small", "nano"):
        if s in n:
            size = "2xlarge" if s == "xxlarge" else s
            break
    return size, task


def load_detr_model(args, device):
    import rfdetr

    model_path = args.model
    use_half = (getattr(args, "optimize", "fp16") == "fp16")
    batch_size = max(1, int(getattr(args, "batch_size", 1) or 1))
    imgsz_display = args.imgsz if getattr(args, "imgsz", None) else "自动"
    print(f"\nLoading model: {model_path} "
          f"[Device: {device}, FP16: {use_half}, Batch: {batch_size}, imgsz: {imgsz_display}]")
    # 本地存在的权重文件: 可能是训练 checkpoint (有 args), 也可能是官方预训练权重
    local_file = os.path.exists(model_path) and model_path.lower().endswith(
        (".pth", ".pt", ".ckpt"))
    if local_file:
        try:
            model = rfdetr.RFDETR.from_checkpoint(model_path, device=device)
            print("  ✓ 训练 checkpoint (from_checkpoint)")
        except Exception as e:
            # 官方预训练权重没有 args 字段, 不是训练 checkpoint; 按变体类加载
            size, task = _infer_size_task_from_name(os.path.basename(model_path))
            size = args.size or size or "medium"
            task = args.task or task or "seg"
            cls_name = (_SIZE_MAP_SEG if task == "seg" else _SIZE_MAP_DETECT)[size]
            if not hasattr(rfdetr, cls_name):
                print(f"❌ rfdetr 不含模型类 {cls_name}")
                _exit(1)
            print(f"  ✓ 官方预训练权重, 按 {cls_name} 加载 ({type(e).__name__})")
            model = getattr(rfdetr, cls_name)(pretrain_weights=model_path, device=device)
    else:
        if not args.size or not args.task:
            print("⚠️  --model 不是本地文件时需同时指定 --size 与 --task; "
                  "默认按 seg/medium 处理")
        size = args.size or "medium"
        task = args.task or "seg"
        reg = _SIZE_MAP_SEG if task == "seg" else _SIZE_MAP_DETECT
        cls_name = reg[size]
        if not hasattr(rfdetr, cls_name):
            print(f"❌ rfdetr 不含模型类 {cls_name}")
            _exit(1)
        print(f"Loading RF-DETR pretrained: {cls_name} ({args.model})")
        try:
            model = getattr(rfdetr, cls_name)(pretrain_weights=args.model, device=device)
        except Exception:
            model = getattr(rfdetr, cls_name)(device=device)

    # ---- YOLO 兼容: --imgsz 映射到 RF-DETR 推理分辨率 ----
    if getattr(args, "imgsz", None):
        try:
            new_res = int(args.imgsz)
            if hasattr(model.model, "resolution"):
                model.model.resolution = new_res
            try:
                model.model_config.resolution = new_res
            except Exception:
                pass
            try:
                model.model.args.resolution = new_res
            except Exception:
                pass
            print(f"ℹ️ 推理分辨率已设为 {new_res} (--imgsz)")
        except Exception as e:
            print(f"⚠️ RF-DETR 无法应用 --imgsz={args.imgsz}: {e}")

    # ---- 官方推理优化: export 融合 + 可选 FP16 (compile=False, 保留任意 batch) ----
    opt = getattr(args, "optimize", "fp16")
    if opt and opt != "none":
        dtype = torch.float16 if opt == "fp16" else torch.float32
        try:
            t0 = time.time()
            model.inference(compile=False, dtype=dtype, inplace=False)
            print(f"⚡ 推理优化已启用: optimize={opt} dtype={dtype}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"⚠️ 推理优化失败, 回退基线: {e}")

    # 类别名解析
    class_names = None
    try:
        cn = model.class_names
        if cn:
            class_names = list(cn)
    except Exception:
        pass
    if not class_names and local_file:
        class_names = _read_class_names_from_sidecar(model_path)
    if not class_names:
        try:
            class_names = list(model.model.args.class_names)
        except Exception:
            class_names = None
    if not class_names:
        print("⚠️  无法获取类别名, 按 class_id 数字命名")
        class_names = []

    # ---- GPU 遥测 + 预热 (触发 MIOpen 编译并把权重搬到设备, 让 GPU 活动可见) ----
    if device != "cpu" and torch.cuda.is_available():
        try:
            print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}  | device={device}")
        except Exception:
            pass
        try:
            res = int(getattr(model.model, "resolution", 432) or 432)
            bs = max(1, int(getattr(args, "batch_size", 1) or 1))
            dummies = [np.zeros((res, res, 3), dtype=np.uint8) for _ in range(bs)]
            t0 = time.time()
            _predict_batch(model, dummies, 0.5)
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1e6
            print(f"🔥 预热完成 {time.time()-t0:.2f}s (batch={bs}, MIOpen/AOTRITON 已编译, "
                  f"权重已在 GPU) | GPU 显存占用 {alloc:.0f} MB")
        except Exception as e:
            print(f"⚠️  预热失败 (继续): {e}")

    return build_context(model, device, class_names, args)


def build_context(model, device, class_names, args):
    """由模型 + 类别名构造推理上下文 (类别别名映射 / 阈值 / polygon 配置)。"""
    _alias_lookup = {}
    for logic_name, cfg in CLASS_CONFIG_TEMPLATE.items():
        for a in cfg["aliases"]:
            _alias_lookup[a.lower()] = (logic_name, cfg)

    CLASS_CONFIG = {}
    for cls_id, raw_name in enumerate(class_names):
        hit = _alias_lookup.get(str(raw_name).lower())
        if hit is not None:
            logic_name, cfg = hit
            CLASS_CONFIG[cls_id] = {"name": logic_name, "raw_name": str(raw_name),
                                    "color": cfg["color"], "conf": cfg["conf"]}
        else:
            CLASS_CONFIG[cls_id] = {"name": str(raw_name), "raw_name": str(raw_name),
                                    "color": (128, 128, 128), "conf": args.conf}

    BALL_CLASS_ID = next((k for k, v in CLASS_CONFIG.items() if v["name"] == BALL_LOGIC_NAME), None)
    TABLE_CLASS_ID = next((k for k, v in CLASS_CONFIG.items() if v["name"] == "table"), None)
    NET_CLASS_ID = next((k for k, v in CLASS_CONFIG.items() if v["name"] == "net"), None)

    CLASS_NAMES = {k: v["name"] for k, v in CLASS_CONFIG.items()}
    COLORS = {k: v["color"] for k, v in CLASS_CONFIG.items()}
    CONF_THRESHOLDS = build_conf_thresholds(args, CLASS_CONFIG)
    POLYGON_NEEDED_CLASS_IDS = {cls_id for cls_id, v in CLASS_CONFIG.items()
                                if v["name"] in POLYGON_NEEDED_LOGIC_NAMES}
    want_polygons = (not args.no_polygon) and bool(POLYGON_NEEDED_CLASS_IDS)
    min_conf_for_predict = max(0.01, min(CONF_THRESHOLDS.values()) - float(args.conf_margin))

    print(f"类别: {CLASS_NAMES}")
    print(f"阈值: { {CLASS_NAMES[k]: v for k, v in CONF_THRESHOLDS.items()} }")
    if not want_polygons:
        print("ℹ️ Polygon 输出已禁用")
    else:
        print(f"ℹ️ Polygon 仅对类别 {sorted(POLYGON_NEEDED_CLASS_IDS)} 提取")

    return {
        "model": model, "device": device,
        "CLASS_CONFIG": CLASS_CONFIG, "CLASS_NAMES": CLASS_NAMES, "COLORS": COLORS,
        "CONF_THRESHOLDS": CONF_THRESHOLDS,
        "BALL_CLASS_ID": BALL_CLASS_ID, "TABLE_CLASS_ID": TABLE_CLASS_ID,
        "NET_CLASS_ID": NET_CLASS_ID,
        "POLYGON_NEEDED_CLASS_IDS": POLYGON_NEEDED_CLASS_IDS,
        "want_polygons": want_polygons,
        "min_conf_for_predict": min_conf_for_predict,
    }


# ==================== 单视频推理 ====================
def _predict_batch(model, frames_rgb, threshold):
    try:
        return model.predict(frames_rgb, threshold=threshold, include_source_image=False)
    except TypeError:
        return model.predict(frames_rgb, threshold=threshold)


def process_one_video(video_path, args, ctx, output_dir=None):
    out_dir = output_dir or (os.path.dirname(args.output) if args.output.endswith(".json")
                             else args.output)
    os.makedirs(out_dir, exist_ok=True)
    log_path = (args.output if args.output.endswith(".json")
                else os.path.join(out_dir, "raw_detections.jsonl"))

    CLASS_CONFIG = ctx["CLASS_CONFIG"]
    CONF_THRESHOLDS = ctx["CONF_THRESHOLDS"]
    CLASS_NAMES = ctx["CLASS_NAMES"]
    COLORS = ctx["COLORS"]
    BALL_CLASS_ID = ctx["BALL_CLASS_ID"]
    TABLE_CLASS_ID = ctx["TABLE_CLASS_ID"]
    NET_CLASS_ID = ctx["NET_CLASS_ID"]
    POLYGON_NEEDED = ctx["POLYGON_NEEDED_CLASS_IDS"]
    want_polygons = ctx["want_polygons"]
    model = ctx["model"]
    device = ctx["device"]

    try:
        print(f"\n📹 Video: {video_path}")
        if is_variable_frame_rate(video_path):
            print("⚠️ Variable Frame Rate (VFR) detected!")
            cfr_video_path = os.path.join(out_dir, "video_cfr.mp4")
            video_input_path = convert_vfr_to_cfr(
                video_path, cfr_video_path,
                highlights_pct=args.cfr_highlights,
                contrast_pct=args.cfr_contrast,
                saturation_pct=args.cfr_saturation)
        else:
            print("✅ Constant Frame Rate (CFR) confirmed.")
            video_input_path = video_path

        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if args.max_frames and args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames) if total_frames > 0 else args.max_frames
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w = args.out_width
        out_h = max(2, int(round(height * out_w / width)))
        if out_h % 2 != 0:
            out_h += 1

        output_video_path = os.path.join(out_dir, "pingpong_detection.mp4")
        if args.writer_backend == "ffmpeg" and shutil.which("ffmpeg"):
            out_video = FFmpegWriter(output_video_path, out_w, out_h, fps,
                                     prefer=args.encoder, amf_quality=args.amf_quality,
                                     amf_qp=args.amf_qp)
            if out_video.encoder:
                print(f"🎬 FFmpeg pipe writer ({out_video.encoder}, "
                      f"quality={args.amf_quality}, qp={args.amf_qp}) "
                      f"@ {out_w}x{out_h} fps={fps}")
            else:
                print(f"🎬 FFmpeg pipe writer (libx264 software fallback) "
                      f"@ {out_w}x{out_h} fps={fps}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
            print(f"🎬 OpenCV writer @ {out_w}x{out_h} fps={fps}")

        total_counts = {c: 0 for c in CLASS_CONFIG}
        empty_frames = 0
        batch_size = max(1, args.batch_size)
        use_half = (getattr(args, "optimize", "fp16") == "fp16")
        imgsz_now = (args.imgsz if getattr(args, "imgsz", None)
                     else (int(getattr(model.model, "resolution", 0) or 0) or "自动"))
        frame_idx = 0
        processed = 0
        start_time = datetime.now()
        last_log = start_time
        last_idx = 0
        ewma_fps = None
        model_names = list(CLASS_NAMES.values())

        if total_frames > 0:
            print(f"\n🚀 开始推理: {total_frames:,} 帧, batch={batch_size}, "
                  f"half={use_half}, imgsz={imgsz_now}")
        else:
            print(f"\n🚀 开始推理 (总帧数未知), batch={batch_size}")

        with open(log_path, "w", encoding="utf-8") as log_f:
            while True:
                if args.max_frames and args.max_frames > 0 and frame_idx >= args.max_frames:
                    break
                batch_frames = []
                batch_idxs = []
                for _ in range(batch_size):
                    if args.max_frames and args.max_frames > 0 and frame_idx >= args.max_frames:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    batch_idxs.append(frame_idx)
                    frame_idx += 1
                if not batch_frames:
                    break

                # 尾批 padding 到固定 batch: 避免不同 shape 触发 MIOpen/AOTRITON 重复编译
                actual_n = len(batch_frames)
                if 0 < actual_n < batch_size:
                    batch_frames = batch_frames + [np.zeros_like(batch_frames[0])] * (batch_size - actual_n)

                rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
                with torch.inference_mode():
                    results = _predict_batch(model, rgb_list, ctx["min_conf_for_predict"])
                if not isinstance(results, (list, tuple)):
                    results = [results]
                results = results[:actual_n]
                batch_frames = batch_frames[:actual_n]

                for frame_raw, fidx, dets in zip(batch_frames, batch_idxs, results):
                    frame_counts = {c: 0 for c in CLASS_CONFIG}
                    frame_dets = []
                    best = {}

                    n = 0 if dets is None else len(dets)
                    xyxy = getattr(dets, "xyxy", None) if n else None
                    conf = getattr(dets, "confidence", None) if n else None
                    cls_ids = getattr(dets, "class_id", None) if n else None
                    masks = getattr(dets, "mask", None) if n else None

                    for i in range(n):
                        cls_id = int(cls_ids[i])
                        c = float(conf[i])
                        if cls_id not in CLASS_CONFIG or c < CONF_THRESHOLDS.get(cls_id, 0.0):
                            continue
                        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                        name = CLASS_CONFIG[cls_id]["name"]
                        poly = None
                        if want_polygons and cls_id in POLYGON_NEEDED and masks is not None:
                            poly = mask_to_polygon(masks[i], approx_eps_ratio=args.poly_approx)
                        if poly is None and want_polygons and cls_id in POLYGON_NEEDED:
                            # 退化: 用 bbox 四角作 polygon
                            poly = [[round(x1, 1), round(y1, 1)], [round(x2, 1), round(y1, 1)],
                                    [round(x2, 1), round(y2, 1)], [round(x1, 1), round(y2, 1)]]
                        det = {"class_id": cls_id, "class": name, "confidence": round(c, 3),
                               "bbox": [x1, y1, x2, y2], "polygon": poly}
                        frame_dets.append(det)
                        frame_counts[cls_id] += 1
                        if cls_id not in best or c > best[cls_id]["confidence"]:
                            best[cls_id] = det

                    ball_det = best.get(BALL_CLASS_ID)
                    bx = by = 0.0
                    ball_pos = None
                    if ball_det:
                        b = ball_det["bbox"]
                        bx, by = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
                        ball_pos = (bx, by)

                    out_frame = cv2.resize(frame_raw, (out_w, out_h))
                    # 球桌 / 球网 mask 叠加
                    if masks is not None:
                        sx, sy = out_w / frame_raw.shape[1], out_h / frame_raw.shape[0]
                        for i in range(n):
                            cid = int(cls_ids[i])
                            if cid not in (TABLE_CLASS_ID, NET_CLASS_ID):
                                continue
                            mm = cv2.resize(masks[i].astype(np.uint8), (out_w, out_h),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                            color = (255, 0, 255) if cid == TABLE_CLASS_ID else (255, 255, 255)
                            alpha = 0.3 if cid == TABLE_CLASS_ID else 0.4
                            draw_mask_overlay_inplace(out_frame, mm, color, alpha)

                    sxd, syd = out_w / frame_raw.shape[1], out_h / frame_raw.shape[0]
                    scaled = [{**d, "bbox": [d["bbox"][0] * sxd, d["bbox"][1] * syd,
                                            d["bbox"][2] * sxd, d["bbox"][3] * syd]}
                              for d in frame_dets]
                    draw_detections(out_frame, scaled, COLORS)
                    out_video.write(out_frame)

                    timestamp = fidx / fps
                    log_obj = {
                        "frame": fidx,
                        "timestamp": round(timestamp, 2),
                        "counts": {CLASS_NAMES[cid]: frame_counts[cid] for cid in CLASS_NAMES},
                        "ball_pos": (round(bx, 1), round(by, 1)) if ball_pos else None,
                        "detections": [
                            {"class": d["class"], "confidence": d["confidence"],
                             "bbox": [round(x, 1) for x in d["bbox"]], "polygon": d.get("polygon")}
                            for d in frame_dets
                        ],
                    }
                    log_f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

                    for cid, cnum in frame_counts.items():
                        total_counts[cid] += cnum
                    if not any(v > 0 for v in frame_counts.values()):
                        empty_frames += 1
                    processed += 1

                now = datetime.now()
                if (now - last_log).total_seconds() >= 2.0:
                    dt = (now - last_log).total_seconds()
                    fps_now = (processed - last_idx) / max(dt, 1e-6)
                    ewma_fps = fps_now if ewma_fps is None \
                        else 0.7 * ewma_fps + 0.3 * fps_now
                    gpu_info = ""
                    if device != "cpu" and torch.cuda.is_available():
                        try:
                            gpu_info = f" | GPU显存={torch.cuda.memory_allocated()/1e6:.0f}MB"
                        except Exception:
                            pass
                    if total_frames > 0:
                        pct = processed / total_frames * 100
                        bar_len = 30
                        filled = int(bar_len * processed / total_frames)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        eta = (total_frames - processed) / max(ewma_fps, 1e-6)
                        print(f"  [{bar}] {pct:5.1f}%  "
                              f"帧 {processed:,}/{total_frames:,}  "
                              f"{fps_now:.1f} fps | ETA {eta:.0f}s  "
                              f"| 检出(已写): {sum(total_counts.values())}{gpu_info}",
                              flush=True)
                    else:
                        elapsed_total = (now - start_time).total_seconds()
                        print(f"  帧 {processed:,}  "
                              f"{fps_now:.1f} fps (EWMA {ewma_fps:.1f})  "
                              f"已耗时 {elapsed_total:.0f}s{gpu_info}", flush=True)
                    last_log = now
                    last_idx = processed

        cap.release()
        out_video.release()
        if video_input_path != video_path and os.path.exists(video_input_path) and args.remove_cfr:
            try:
                os.remove(video_input_path)
            except Exception:
                pass

        elapsed = (datetime.now() - start_time).total_seconds()
        avg_fps = processed / max(elapsed, 1e-6)
        print(f"\n⏱️  Total: {elapsed:.1f}s | Avg: {avg_fps:.2f} fps "
              f"(processed {processed:,} frames)")

        total_detections = sum(total_counts.values())
        print(f"🔍 检测统计: 总检测数={total_detections}")
        for cid, cnum in total_counts.items():
            if cnum > 0:
                print(f"    {CLASS_NAMES[cid]}: {cnum}")

        print(f"✅ Inference Complete! Output: {out_dir}")
        print(f"📄 raw_detections.jsonl: {log_path}")
        print(f"🎬 pingpong_detection.mp4: {output_video_path}")
        return True, log_path
    except Exception as e:
        print(f"❌ process_one_video error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ==================== 主流程 ====================
def _exit(code):
    """干净退出。

    Windows 上 os._exit()/_exit() 最终走 ExitProcess, 它会逐个调用 DLL 的
    DLL_PROCESS_DETACH 回调; AMD ROCm/MIOpen 的 DLL 在该阶段会卡死, 导致进程
    完成工作后无法退出 (CPU 空转) 且退出码异常。这里改用 TerminateProcess
    (跳过 DLL 卸载) 强制立即结束。"""
    code = int(code)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if _IS_WINDOWS:
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateProcess(k32.GetCurrentProcess(), code)
        except Exception:
            pass
    os._exit(code)


def main():
    args = parse_args()
    _warn_ignored_yolo_args(args)

    if args.device.isdigit():
        device = f"cuda:{args.device}"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ 未检测到 GPU, 回退 CPU")
        device = "cpu"

    if torch.cuda.is_available() and device != "cpu":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    print("=" * 60)
    print(f"🏓 PingPong Video Inference v8 [RF-DETR] [Platform: {platform.system()}]")
    print("=" * 60)
    if device != "cpu":
        try:
            dev_idx = int(args.device) if args.device.isdigit() else 0
            print(f"GPU: {torch.cuda.get_device_name(dev_idx)}")
        except Exception:
            pass
    t0 = time.time()
    ctx = load_detr_model(args, device)
    print(f"[BENCHMARK] model_loaded id={id(ctx['model'])} time={time.time() - t0:.1f}s")

    if args.video_list_file:
        with open(args.video_list_file, "r", encoding="utf-8") as f:
            videos = [ln.strip() for ln in f if ln.strip()]
        if not videos:
            print("❌ video-list-file 为空")
            _exit(1)
        failed = []
        for vp in videos:
            stem = os.path.splitext(os.path.basename(vp))[0]
            od = os.path.join(args.output, f"pipeline_{stem}")
            os.makedirs(od, exist_ok=True)
            ok, _ = process_one_video(vp, args, ctx, output_dir=od)
            if not ok:
                failed.append(vp)
        if failed:
            print(f"⚠️ {len(failed)} 个视频失败:")
            for f in failed:
                print(f"  - {f}")
            _exit(1)
    else:
        if not args.video:
            print("❌ 需指定 --video 或 --video-list-file")
            _exit(1)
        ok, _ = process_one_video(args.video, args, ctx)
        _exit(0 if ok else 1)


if __name__ == "__main__":
    main()
