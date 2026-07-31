#!/usr/bin/env python3
"""
pingpong_highlight_track.py  (优化版 v4.8 - track_id 贯穿全链路, TRACK MODE 跟踪版)
=====================================================================
乒乓球比赛 → 自动特效精华集锦 (跟踪版)

TRACK MODE 跟踪版改动:
  ★ 输入 JSONL 的 detection 可含 "track_id" 字段 (由 inference_video_track.py 生成),
    球候选携带 track_id 组成 5 元组 (cx, cy, conf, bbox, track_id)。
  ★ 新增轨迹级前置过滤: filter_static_tracks (整条静止轨迹剔除) +
    filter_short_lived_tracks (短生命周期轨迹剔除), 替代原网格哈希静止球过滤。
  ★ track_id 贯穿全链路 (不再剥离回 4 元组): 轨迹级过滤 → 走廊豁免 →
    主轨迹构建 (异 tid 衔接容差×0.5 收紧) → 补帧 (拒绝跨 tid 插值, 保留断口) →
    渲染 (拖尾 trail 按 tid 断开重新起笔 + 相邻点距离突变兜底不连线),
    根治残影跨轨迹混画。
  ★ 旧版 JSONL 无 track_id 时自动降级: 跳过轨迹级过滤并给出中文警告,
    其余行为与原版完全一致 (仅多一道渲染层距离突变兜底断线保护)。

v4.8 在 v4.7 基础上的改动 (修复残影跨轨迹混画):
  ★ track_id 从前置过滤一直透传到渲染层: main_track 值扩为
    (x, y, conf, bbox, tid), filled 值扩为 (x, y, is_interp, tid),
    scaled_track 值扩为 (x*scale, y*scale, tid)。
  ★ 衔接匹配 (贪心/匈牙利) 加 tid 约束; 补帧拒绝跨 tid 插值;
    trail 按 tid 断开; SpeedTrailRenderer 加对角线 6% 距离突变兜底。

v4.7 在 v4.6 基础上的改动 (功能/画质不变, 仅精简与提速):
  ★ 渲染热路径去除整帧 frame.copy(): PlayerFX 光环 / 球头辉光改为仅在
    局部 ROI 上 addWeighted, 大幅降低每帧内存分配。
  ★ AMF 编码器探测结果缓存 (避免重复 test-encode); 默认使用 speed 档,
    针对 Ryzen AI Max 395 iGPU (h264_amf) 吞吐更高。
  ★ speed_to_color 结果做 256 级 LUT 缓存, 避免每段拖尾重复插值。
  ★ track↔detection 匈牙利/贪心两条分支合并为单一 greedy-fallback 辅助
    函数, 去除重复代码。
  ★ ffmpeg pipe 直编增加 -threads / 队列参数, 减少管道阻塞。
  ★ 其余几何自愈 / 走廊 / 落点交替过滤 / 速度自适应特效行为完全保留。

v4.6 特效 (脚下光环 / 高透明白色落点圈 / 更淡拖尾 / 按回合重置落点) 全保留。
"""

import os
import sys
import json
import math
import shutil
import argparse
import subprocess
import tempfile
import platform
import time
from collections import defaultdict, deque
from enum import Enum, auto

import numpy as np
import cv2

# ---- 可选依赖 ----
try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
    _HAS_SCIPY_LSA = True
except Exception:
    _linear_sum_assignment = None
    _HAS_SCIPY_LSA = False

_IS_WINDOWS = (platform.system() == "Windows")
_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0
_CPU = os.cpu_count() or 8

_SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float64)
_PLAYER_CLASS_NAMES = ("person", "player", "people", "human", "athlete")

# 编码器探测缓存 (进程内只探测一次)
_ENC_CACHE = {}

TABLE_LEN_M = 2.74
TABLE_WID_M = 1.525
TABLE_RATIO = TABLE_LEN_M / TABLE_WID_M   # ≈ 1.797


# =====================================================================
#  CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="乒乓球比赛自动特效高光剪辑 (pingpong_highlight.py)")
    p.add_argument("--video", "-v", required=True, help="输入视频路径")
    p.add_argument("--raw-detections", "-r", required=True, help="raw_detections.jsonl 路径")
    p.add_argument("--output", "-o", default="highlight_output.mp4", help="输出 mp4 路径")
    p.add_argument("--output-height", type=int, default=None, help="输出高度像素 (默认原分辨率)")
    p.add_argument("--min-rally-frames", type=int, default=None, help="有效回合最小帧数 (默认 0.6s)")
    p.add_argument("--min-track-frames", type=int, default=5,
                   help="短生命周期轨迹阈值: 轨迹出现总帧数低于该值则整体剔除")
    p.add_argument("--track-static-frames", type=int, default=15,
                   help="静止轨迹最小持续帧数: 轨迹持续帧数达到该值才参与静止判定")
    p.add_argument("--track-static-motion", type=float, default=3.0,
                   help="静止轨迹相邻帧最大位移像素: 轨迹相邻点位移均小于该值判定为静止")
    p.add_argument("--pre-buffer", type=float, default=0.5, help="每个回合前缓冲秒数")
    p.add_argument("--post-buffer", type=float, default=1.5, help="每个回合后缓冲秒数")
    p.add_argument("--merge-gap", type=float, default=2.0, help="相邻回合间隔小于该秒数则合并")
    p.add_argument("--corridor-margin", type=float, default=0.5, help="长边走廊横向容差")
    p.add_argument("--corridor-extend", type=float, default=1.2, help="走廊沿长轴向短边外延伸倍数")
    p.add_argument("--out-of-table-timeout", type=float, default=2.0,
                   help="持续出走廊该秒数则回合结束 (默认 2.0, 放宽以容忍放高球滞空时间)")
    p.add_argument("--corridor-max-drop", type=float, default=0.55, help="走廊过滤单次剔除比例上限")
    p.add_argument("--track-corridor-exempt-ratio", type=float, default=0.3,
                   help="轨迹走廊豁免阈值: 某 track 落在走廊内的点占比 >= 该值时"
                        "整条轨迹豁免走廊过滤 (保留放高球飞出走廊的段落)")
    p.add_argument("--ball-conf", type=float, default=0.5,
                   help="乒乓球检测最小置信度, 低于该值的球候选在收集阶段直接丢弃")
    p.add_argument("--no-audio", action="store_true", help="不保留原声")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_amf", "hevc_amf"],
                   help="输出编码器, auto=优先 AMD AMF 硬编")
    p.add_argument("--max-jump-ratio", type=float, default=0.1, help="轨迹最大跳变距离占对角线比例")
    p.add_argument("--match-algo", default="auto", choices=["auto", "hungarian", "greedy"],
                   help="track↔detection 关联算法")
    p.add_argument("--fx-level", default="normal", choices=["minimal", "normal", "max"],
                   help="特效强度")
    p.add_argument("--no-player-fx", action="store_true", help="关闭选手移动特效 (脚下光环)")
    p.add_argument("--debug", action="store_true", help="输出调试信息")
    return p.parse_args()


# =====================================================================
#  工具函数
# =====================================================================
def log(msg):
    print(msg, flush=True)


def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_color(c0, c1, t):
    t = _clamp(t, 0.0, 1.0)
    return (int(_lerp(c0[0], c1[0], t)),
            int(_lerp(c0[1], c1[1], t)),
            int(_lerp(c0[2], c1[2], t)))


def _build_speed_lut():
    """预计算 256 级 速度→BGR 颜色 LUT, 避免运行时反复插值。"""
    stops = [(0.00, (255, 200, 60)), (0.35, (120, 240, 80)),
             (0.65, (60, 240, 255)), (1.00, (40, 90, 255))]
    lut = []
    for k in range(256):
        norm = k / 255.0
        col = stops[-1][1]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if norm <= t1:
                local = (norm - t0) / (t1 - t0 + 1e-9)
                col = _lerp_color(c0, c1, local)
                break
        lut.append(col)
    return lut


_SPEED_LUT = _build_speed_lut()


def speed_to_color(norm):
    return _SPEED_LUT[int(_clamp(norm, 0.0, 1.0) * 255)]


def ffprobe_has_audio(video):
    if not shutil.which("ffprobe"):
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video],
            capture_output=True, text=True, timeout=20, creationflags=_SUBPROC_FLAGS)
        return "audio" in (r.stdout or "")
    except Exception:
        return False


def detect_amf_encoder(prefer="auto"):
    if prefer in _ENC_CACHE:
        return _ENC_CACHE[prefer]
    result = _detect_amf_encoder_impl(prefer)
    _ENC_CACHE[prefer] = result
    return result


def _detect_amf_encoder_impl(prefer):
    if prefer == "libx264":
        return "libx264"
    if not shutil.which("ffmpeg"):
        return "libx264"
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=8,
                           creationflags=_SUBPROC_FLAGS)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return "libx264"
    cand = []
    if prefer in ("auto", "h264_amf") and "h264_amf" in out:
        cand.append("h264_amf")
    if prefer in ("auto", "hevc_amf") and "hevc_amf" in out:
        cand.append("hevc_amf")
    for enc in cand:
        try:
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.2:r=25",
                 "-c:v", enc, "-f", "null", "-"],
                capture_output=True, text=True, timeout=15,
                creationflags=_SUBPROC_FLAGS)
            if probe.returncode == 0:
                return enc
        except Exception:
            continue
    return "libx264"


def _enc_args(enc):
    if enc == "h264_amf":
        return ["-c:v", "h264_amf", "-usage", "transcoding",
                "-quality", "speed", "-rc", "cqp",
                "-qp_i", "18", "-qp_p", "18", "-bf", "0", "-pix_fmt", "yuv420p"]
    if enc == "hevc_amf":
        return ["-c:v", "hevc_amf", "-usage", "transcoding",
                "-quality", "speed", "-rc", "cqp",
                "-qp_i", "18", "-qp_p", "18", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-threads", str(_CPU), "-pix_fmt", "yuv420p"]


# =====================================================================
#  1) 读取日志
# =====================================================================
def load_detections(path):
    frames = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            fidx = obj.get("frame")
            if fidx is None:
                continue
            frames[fidx] = obj
    if not frames:
        raise RuntimeError("raw_detections.jsonl 为空或无法解析")
    return [frames[k] for k in sorted(frames.keys())]


# =====================================================================
#  2) 球台 / 球网几何提取
# =====================================================================
def order_quad(quad):
    quad = np.array(quad, dtype=np.float32)
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    s = q[:, 0] + q[:, 1]
    q = np.roll(q, -int(np.argmin(s)), axis=0)
    v1 = q[1] - q[0]
    v2 = q[2] - q[0]
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        q = q[[0, 3, 2, 1]]
    return q.astype(np.float32)


def _get_polygon_pts_single(fr, target_class, conf_min=0.5, img_hw=None):
    """
    取单帧中指定类别的多边形顶点。

    若同一帧识别到多个同类实例（例如多个球台 / 多条球网），
    按各实例几何中心离画面中心最近的原则挑选单个实例返回，
    避免把多个实例的顶点混成一坨导致拟合畸形。
    """
    insts = []
    for det in fr.get("detections", []):
        if det["class"] != target_class or det.get("confidence", 0) < conf_min:
            continue
        poly = det.get("polygon")
        if poly and len(poly) >= 4:
            insts.append(np.array(poly, dtype=np.float32))
        else:
            b = det["bbox"]
            insts.append(np.array(
                [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                dtype=np.float32))

    if not insts:
        return None
    if len(insts) == 1:
        return insts[0]

    # 多实例：选离画面中心最近的那个
    if img_hw is not None:
        h, w = img_hw
        cx, cy = w * 0.5, h * 0.5
    else:
        # 退化：以所有点的包围盒中心近似画面中心
        all_pts = np.concatenate(insts, axis=0)
        cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()

    best_idx, best_dist = 0, float("inf")
    for i, poly in enumerate(insts):
        icx, icy = poly[:, 0].mean(), poly[:, 1].mean()
        d = math.hypot(icx - cx, icy - cy)
        if d < best_dist:
            best_dist, best_idx = d, i
    return insts[best_idx]


def _fit_quad_single(pts):
    if pts is None or len(pts) < 4:
        return None
    pts = pts.astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps_ratio in (0.003, 0.005, 0.008, 0.012, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


def _quad_is_convex_simple(quad):
    q = np.array(quad, dtype=np.float64)
    signs = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = [s for s in signs if abs(s) > 1e-6]
    if not signs:
        return False
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _quad_quality(quad, img_hw):
    h, w = img_hw
    q = order_quad(quad).astype(np.float64)
    edges = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    min_edge = min(edges)
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]
                         for i in range(4)))
    area_frac = area / (w * h)
    side_a = (edges[0] + edges[2]) * 0.5
    side_b = (edges[1] + edges[3]) * 0.5
    ratio = max(side_a, side_b) / (min(side_a, side_b) + 1e-6)
    ok, reason = True, "ok"
    if not _quad_is_convex_simple(q):
        ok, reason = False, "non_convex_or_selfintersect"
    elif min_edge < min(h, w) * 0.04:
        ok, reason = False, f"edge_too_short({min_edge:.0f}px)"
    elif area_frac < 0.01:
        ok, reason = False, f"area_too_small({area_frac:.4f})"
    elif not (0.8 <= ratio <= 3.5):
        ok, reason = False, f"ratio_implausible({ratio:.2f})"
    return {"ok": ok, "reason": reason, "ratio": ratio,
            "area_frac": area_frac, "min_edge": min_edge}


def _edges_are_adjacent(i, j, n=4):
    diff = abs(i - j)
    return diff == 1 or diff == n - 1


def _extend_line(p1, p2, length):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return p1, p2
    ux, uy = dx / norm, dy / norm
    return (p1[0] - ux * length, p1[1] - uy * length), \
           (p2[0] + ux * length, p2[1] + uy * length)


def _segments_intersect(p1, p2, q1, q2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _net_quad_intersections(quad, net_line, img_hw):
    if net_line is None:
        return []
    h, w = img_hw
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    net_ext = _extend_line(net_line[0], net_line[1], length=max(w, h) * 4)
    return [i for i, (a, b) in enumerate(edges)
            if _segments_intersect(net_ext[0], net_ext[1], a, b)]


def _net_bisects_quad(quad, net_line, img_hw):
    if net_line is None:
        return True, []
    inter_idx = _net_quad_intersections(quad, net_line, img_hw)
    if len(inter_idx) != 2 or _edges_are_adjacent(inter_idx[0], inter_idx[1]):
        return False, inter_idx
    return True, inter_idx


def _robust_median_quad(quad_list, img_hw, debug=False):
    if len(quad_list) < 3:
        return None
    ordered = np.stack([order_quad(q) for q in quad_list], axis=0)
    med = np.median(ordered, axis=0)
    devs = np.linalg.norm(ordered - med[None], axis=2).sum(axis=1)
    mad = np.median(np.abs(devs - np.median(devs))) + 1e-6
    keep = devs <= (np.median(devs) + 3.0 * mad)
    if keep.sum() >= 3:
        med = np.median(ordered[keep], axis=0)
    if debug:
        log(f"  时序聚合: {len(quad_list)} 帧 quad, 保留 {int(keep.sum())} 帧 (MAD 剔除离群)")
    return med.astype(np.float32)


def _fit_net_line_single(pts):
    if pts is None or len(pts) < 4:
        return None
    bl_idx = None
    for i in range(1, min(len(pts) - 4, 30)):
        if (pts[i + 1, 0] > pts[i, 0] and pts[i + 2, 0] > pts[i + 1, 0] and
                pts[i + 3, 0] > pts[i + 2, 0]):
            bl_idx = i
            break
    if bl_idx is None:
        return None
    search_start = min(bl_idx + 10, len(pts) - 5)
    br_idx = None
    for i in range(search_start, len(pts) - 4):
        if (pts[i + 1, 1] < pts[i, 1] and pts[i + 2, 1] < pts[i + 1, 1] and
                pts[i + 3, 1] < pts[i + 2, 1]):
            br_idx = i - 1
            break
    if br_idx is None or br_idx <= bl_idx:
        return None
    bl, br = pts[bl_idx], pts[br_idx]
    net_vec = br - bl
    net_len = float(np.linalg.norm(net_vec))
    if net_len < 10.0:
        return None
    net_dir = net_vec / net_len
    proj = (pts - bl) @ net_dir
    p1 = bl + net_dir * proj.max()
    p2 = bl + net_dir * proj.min()
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    return (tuple(p1.astype(float).tolist()), tuple(p2.astype(float).tolist()))


def _median_net_line(line_list):
    if len(line_list) < 3:
        return None
    p1s = np.array([l[0] for l in line_list])
    p2s = np.array([l[1] for l in line_list])
    return (tuple(np.median(p1s, axis=0).tolist()),
            tuple(np.median(p2s, axis=0).tolist()))


# =====================================================================
#  几何校验 + 基于约束的重建
# =====================================================================
def rebuild_quad_from_constraints(table_pts, net_line, img_hw, debug=False):
    if net_line is not None:
        net_start = np.array(net_line[0], dtype=np.float64)
        net_end = np.array(net_line[1], dtype=np.float64)
        net_vec = net_end - net_start
        net_len = float(np.linalg.norm(net_vec))
        if net_len < 10:
            return None
        net_dir = net_vec / net_len
        long_dir = np.array([-net_dir[1], net_dir[0]], dtype=np.float64)
        if long_dir[1] < 0:
            long_dir = -long_dir
        short_len = net_len * 1.05
        detected_ratio = None
        if table_pts is not None and len(table_pts) >= 4:
            (_, _), (rw, rh), _ = cv2.minAreaRect(table_pts.astype(np.float32))
            if rw > 5 and rh > 5:
                detected_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        if detected_ratio is not None and 1.2 < detected_ratio < 4.0:
            blended_ratio = 0.6 * TABLE_RATIO + 0.4 * detected_ratio
        else:
            blended_ratio = TABLE_RATIO
        long_len = short_len * blended_ratio
        half_long = long_dir * (long_len * 0.5)
        corners = np.array([net_start - half_long, net_end - half_long,
                            net_end + half_long, net_start + half_long],
                           dtype=np.float32)
        q = order_quad(corners)
        if debug:
            ratio_str = f"检测比={detected_ratio:.2f}, " if detected_ratio else ""
            log(f"  ◆ 重建台面四角(球网线约束): 短边={short_len:.0f}px "
                f"长边={long_len:.0f}px({ratio_str}ITTF={TABLE_RATIO:.2f})")
        return q
    if table_pts is None or len(table_pts) < 4:
        return None
    pts = table_pts.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), _ = rect
    if rw < 1 or rh < 1:
        return None
    box = np.array(cv2.boxPoints(rect), dtype=np.float64)
    e0, e1 = box[1] - box[0], box[2] - box[1]
    len0 = np.linalg.norm(e0) + 1e-6
    len1 = np.linalg.norm(e1) + 1e-6
    u0, u1 = e0 / len0, e1 / len1
    if len0 >= len1:
        long_dir, long_len, short_dir, short_len = u0, len0, u1, len1
    else:
        long_dir, long_len, short_dir, short_len = u1, len1, u0, len0
    detected_ratio = long_len / (short_len + 1e-6)
    blended_long = short_len * (0.5 * TABLE_RATIO + 0.5 * detected_ratio)
    cen = np.array([cx, cy], dtype=np.float64)
    hl = long_dir * (blended_long * 0.5)
    hs = short_dir * (short_len * 0.5)
    corners = np.array([cen - hl - hs, cen - hl + hs, cen + hl + hs, cen + hl - hs],
                       dtype=np.float32)
    q = order_quad(corners)
    if debug:
        log(f"  ◆ 重建台面四角(minAreaRect): 检测比={detected_ratio:.2f} "
            f"→ 使用比={blended_long/short_len:.2f} (ITTF={TABLE_RATIO:.2f})")
    return q


def validate_and_refine_quad(quad, table_pts, net_line, img_hw, debug=False):
    qual = _quad_quality(quad, img_hw)
    if qual["ok"]:
        if debug:
            log(f"  ✓ 球台四角校验通过: 长宽比={qual['ratio']:.2f} "
                f"面积占比={qual['area_frac']:.3f} 最短边={qual['min_edge']:.0f}px")
        return order_quad(quad), False
    log(f"  ⚠️ 球台四角不合理 ({qual['reason']}, 长宽比={qual['ratio']:.2f}, "
        f"面积占比={qual['area_frac']:.4f}) → 启动约束重建")
    rebuilt = rebuild_quad_from_constraints(table_pts, net_line, img_hw, debug=debug)
    if rebuilt is None:
        log("  ⚠️ 重建失败, 回退使用原始四角 (走廊将启用尺寸兜底)")
        return order_quad(quad), False
    qual2 = _quad_quality(rebuilt, img_hw)
    if qual2["ok"]:
        log(f"  ✓ 重建后四角合理: 长宽比={qual2['ratio']:.2f} 面积占比={qual2['area_frac']:.3f}")
    else:
        log(f"  ⚠️ 重建后仍不理想 ({qual2['reason']}), 仍采用重建结果")
    return rebuilt, True


# =====================================================================
#  长边走廊
# =====================================================================
def build_corridor(far_left, far_right, near_left, near_right,
                   corridor_margin=0.45, corridor_extend=1.2, img_hw=None,
                   open_top=True, debug=False):
    fl = np.array(far_left, dtype=np.float64)
    fr = np.array(far_right, dtype=np.float64)
    nl = np.array(near_left, dtype=np.float64)
    nr = np.array(near_right, dtype=np.float64)
    left_axis, right_axis = fl - nl, fr - nr

    def _unit(v):
        return v / (np.linalg.norm(v) + 1e-8)
    ul, ur = _unit(left_axis), _unit(right_axis)
    body_len_detected = (np.linalg.norm(left_axis) + np.linalg.norm(right_axis)) * 0.5
    near_width_vec = nr - nl
    near_width = np.linalg.norm(near_width_vec) + 1e-8
    body_len = max(body_len_detected, near_width * TABLE_RATIO)
    if debug and body_len > body_len_detected * 1.3:
        log(f"  ◆ 台身长度兜底: 检测={body_len_detected:.0f}px → 采用={body_len:.0f}px")
    ext = corridor_extend * body_len
    far_left_ext = fl + ul * ext
    far_right_ext = fr + ur * ext
    near_left_ext = nl - ul * ext
    near_right_ext = nr - ur * ext
    side_dir = near_width_vec / near_width
    margin_px = corridor_margin * near_width
    far_left_ext -= side_dir * margin_px
    near_left_ext -= side_dir * margin_px
    far_right_ext += side_dir * margin_px
    near_right_ext += side_dir * margin_px
    # 顶部开放: 沿走廊两侧斜边的射线 (near_ext → far_ext) 延长/回缩到 y=-10
    # (略高于画面顶, 避免边界浮点问题), 保持侧边斜率不变, 覆盖放高球 (lob)
    # 飞出走廊上边界的场景; 射线几乎水平时退化为直接钳 Y, 避免除零
    top_y = -10.0
    if open_top:
        for near_pt, far_pt in ((near_left_ext, far_left_ext),
                                (near_right_ext, far_right_ext)):
            d = far_pt - near_pt
            if abs(d[1]) < 1e-6:
                far_pt[1] = top_y          # 射线几乎水平 → 直接钳 Y
            else:
                t = (top_y - near_pt[1]) / d[1]
                if t > 0:
                    far_pt[0] = near_pt[0] + d[0] * t
                    far_pt[1] = top_y      # 沿射线求与 y=top_y 的交点
                else:
                    far_pt[1] = top_y      # 射线朝下 (几何异常) → 兜底钳 Y
    corridor = np.array([far_left_ext, far_right_ext, near_right_ext, near_left_ext],
                        dtype=np.float32)
    if debug:
        top_note = f", 顶部开放(远端钳至 y={top_y:.0f})" if open_top else ""
        log(f"  ◆ 长边走廊: 横向容差={margin_px:.0f}px, 纵向延伸={ext:.0f}px{top_note}")
    return corridor


def _point_in_corridor(pt, corridor):
    if corridor is None:
        return True
    return cv2.pointPolygonTest(corridor.reshape(-1, 1, 2),
                                (float(pt[0]), float(pt[1])), False) >= 0


def extract_table_geometry(frames, img_hw, debug=False, persp_blend=0.5,
                           corridor_margin=0.45, corridor_extend=1.2):
    h, w = img_hw
    sample_frames = frames[::10]
    all_table_pts, table_quads = [], []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "table", conf_min=0.5, img_hw=img_hw)
        if pts is not None and len(pts) >= 4:
            all_table_pts.append(pts)
            quad = _fit_quad_single(pts)
            if quad is not None and len(quad) == 4:
                table_quads.append(quad)
    table_pts_agg = np.concatenate(all_table_pts, axis=0) if all_table_pts else None

    if len(table_quads) < 3:
        log(f"⚠️ 有效球台拟合帧数不足 ({len(table_quads)}<3), 无法做迷你球台 / 落点分析")
        return None

    quad = _robust_median_quad(table_quads, img_hw, debug=debug)
    if quad is None:
        log("⚠️ 球台四角中位数拟合失败")
        return None

    net_lines = []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "net", conf_min=0.5, img_hw=img_hw)
        if pts is not None and len(pts) >= 2:
            line = _fit_net_line_single(pts)
            if line is not None:
                net_lines.append(line)
    net_line = _median_net_line(net_lines) if len(net_lines) >= 3 else None

    quad, was_rebuilt = validate_and_refine_quad(quad, table_pts_agg, net_line, img_hw, debug=debug)

    if net_line is not None and not was_rebuilt:
        bisects_ok, _ = _net_bisects_quad(quad, net_line, img_hw)
        if not bisects_ok:
            if debug:
                log("  ⚠️ 球网与相邻边相交, 启动约束重建")
            rebuilt = rebuild_quad_from_constraints(table_pts_agg, net_line, img_hw, debug=debug)
            if rebuilt is not None:
                quad, was_rebuilt = rebuilt, True

    if net_line is not None and not was_rebuilt:
        net_len = float(np.hypot(net_line[1][0] - net_line[0][0], net_line[1][1] - net_line[0][1]))
        q = order_quad(quad).astype(np.float64)
        edge_lens = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        side_a = (edge_lens[0] + edge_lens[2]) * 0.5
        side_b = (edge_lens[1] + edge_lens[3]) * 0.5
        if min(side_a, side_b) < net_len * 0.7:
            if debug:
                log(f"  ⚠️ 短边远小于球网长度, 四边形过窄, 启动约束重建")
            rebuilt = rebuild_quad_from_constraints(table_pts_agg, net_line, img_hw, debug=debug)
            if rebuilt is not None:
                quad, was_rebuilt = rebuilt, True

    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    if net_line is not None:
        nd = np.array([net_line[1][0] - net_line[0][0], net_line[1][1] - net_line[0][1]],
                      dtype=np.float32)
        nd /= np.linalg.norm(nd) + 1e-8
    else:
        nd = None

    edge_info = []
    for i, (a, b) in enumerate(edges):
        ev = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)
        ev_n = ev / (np.linalg.norm(ev) + 1e-8)
        par = abs(float(np.dot(ev_n, nd))) if nd is not None else 0.0
        edge_info.append((i, np.linalg.norm(ev), math.degrees(math.acos(min(1.0, par))), a, b))

    def _pick_near_far(end_edge_list):
        near_corner_idx = int(np.argmin([math.hypot(c[0], c[1] - h) for c in quad]))
        near_corner = quad[near_corner_idx]
        e0_a = np.array(end_edge_list[0][3])
        e0_b = np.array(end_edge_list[0][4])
        if (np.linalg.norm(near_corner - e0_a) < 2.0 or np.linalg.norm(near_corner - e0_b) < 2.0):
            return end_edge_list[0][3], end_edge_list[0][4], end_edge_list[1][3], end_edge_list[1][4]
        return end_edge_list[1][3], end_edge_list[1][4], end_edge_list[0][3], end_edge_list[0][4]

    found_end_edges = False
    if net_line is not None:
        inter_idx = _net_quad_intersections(quad, net_line, img_hw)
        if len(inter_idx) == 2 and not _edges_are_adjacent(inter_idx[0], inter_idx[1]):
            found_end_edges = True
            non_inter_idx = [i for i in range(4) if i not in inter_idx]
            near_a, near_b, far_a, far_b = _pick_near_far([edge_info[i] for i in non_inter_idx])
        elif debug:
            log("  ⚠️ 球网仍与相邻边相交, 回退到角度法判定端边")
    if not found_end_edges:
        edge_info.sort(key=lambda x: x[2])
        near_a, near_b, far_a, far_b = _pick_near_far(edge_info[:2])

    if far_a[0] > far_b[0]:
        far_left, far_right = far_b, far_a
    else:
        far_left, far_right = far_a, far_b
    if near_a[0] > near_b[0]:
        near_left, near_right = near_b, near_a
    else:
        near_left, near_right = near_a, near_b

    if persp_blend > 0:
        near_width = float(near_right[0] - near_left[0])
        if near_width > 10:
            near_slope = (near_right[1] - near_left[1]) / near_width
            old_far_left = np.array(far_left, dtype=np.float32).copy()
            old_far_right = np.array(far_right, dtype=np.float32).copy()
            if far_left[1] < far_right[1]:
                target_x = far_right[0] - near_width
                target_y = far_right[1] - near_slope * near_width
                far_left = np.array([far_left[0] + persp_blend * (target_x - far_left[0]),
                                     far_left[1] + persp_blend * (target_y - far_left[1])],
                                    dtype=np.float32)
            else:
                target_x = far_left[0] + near_width
                target_y = far_left[1] + near_slope * near_width
                far_right = np.array([far_right[0] + persp_blend * (target_x - far_right[0]),
                                      far_right[1] + persp_blend * (target_y - far_right[1])],
                                     dtype=np.float32)
            if (not np.allclose(far_left, old_far_left, atol=0.5) or
                    not np.allclose(far_right, old_far_right, atol=0.5)):
                quad_updated = quad.copy()
                for qi in range(4):
                    if np.linalg.norm(quad[qi] - old_far_left) < 3.0:
                        quad_updated[qi] = far_left
                    elif np.linalg.norm(quad[qi] - old_far_right) < 3.0:
                        quad_updated[qi] = far_right
                quad = quad_updated
                if debug:
                    log(f"  ◆ 透视修正(平行 blend={persp_blend})")

    src = np.array([far_left, far_right, near_right, near_left], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [TABLE_WID_M * 1000, 0.0],
                    [TABLE_WID_M * 1000, TABLE_LEN_M * 1000],
                    [0.0, TABLE_LEN_M * 1000]], dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        H = None

    net_side_sign = None
    if net_line is not None and H is not None:
        (nx1, ny1), (nx2, ny2) = net_line
        near_mid_x = (near_left[0] + near_right[0]) * 0.5
        near_mid_y = (near_left[1] + near_right[1]) * 0.5
        cross_near = (nx2 - nx1) * (near_mid_y - ny1) - (ny2 - ny1) * (near_mid_x - nx1)
        net_side_sign = 1 if cross_near > 0 else -1

    corridor = build_corridor(far_left, far_right, near_left, near_right,
                              corridor_margin=corridor_margin,
                              corridor_extend=corridor_extend, img_hw=img_hw, debug=debug)
    if debug:
        log(f"  球台四角: {quad.tolist()}")
        log(f"  球网线: {net_line}")

    return {"quad": quad, "net_line": net_line, "homography": H,
            "net_side_sign": net_side_sign, "corridor": corridor, "rebuilt": was_rebuilt,
            "far_left": np.array(far_left, dtype=np.float32),
            "far_right": np.array(far_right, dtype=np.float32),
            "near_left": np.array(near_left, dtype=np.float32),
            "near_right": np.array(near_right, dtype=np.float32),
            "far_w": float(far_right[0] - far_left[0]),
            "near_w": float(near_right[0] - near_left[0]),
            "far_mean_y": float((far_left[1] + far_right[1]) / 2.0),
            "near_mean_y": float((near_left[1] + near_right[1]) / 2.0)}


def img_to_table(H_or_geo, pt):
    if H_or_geo is None:
        return None
    if isinstance(H_or_geo, dict):
        geo = H_or_geo
        H = geo.get("homography")
        if H is None:
            return None
        out = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], dtype=np.float32), H)
        x_mm, y_mm = float(out[0, 0, 0]), float(out[0, 0, 1])
        net_line = geo.get("net_line")
        net_side_sign = geo.get("net_side_sign")
        if net_line is not None and net_side_sign is not None:
            (nx1, ny1), (nx2, ny2) = net_line
            cross = (nx2 - nx1) * (pt[1] - ny1) - (ny2 - ny1) * (pt[0] - nx1)
            is_near_side = (cross * net_side_sign > 0)
            half_y = TABLE_LEN_M * 1000 / 2.0
            if is_near_side and y_mm < half_y:
                y_mm = half_y + (half_y - y_mm)
            elif (not is_near_side) and y_mm > half_y:
                y_mm = half_y - (y_mm - half_y)
        return (x_mm, y_mm)
    out = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], dtype=np.float32), H_or_geo)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# =====================================================================
#  3) 球检测整理 + 错误球过滤
# =====================================================================
def gather_ball_candidates(frames, fps, conf_min=0.35):
    out = []
    for fr in frames:
        cands = []
        for det in fr.get("detections", []):
            if det["class"] != "pingpangball":
                continue
            if det.get("confidence", 0.0) < conf_min:
                continue
            b = det["bbox"]
            cx, cy = bbox_center(b)
            # 跟踪版: 附带 track_id (旧版 JSONL 无该字段时为 -1), 组成 5 元组
            cands.append((cx, cy, det.get("confidence", 0.0), b, det.get("track_id", -1)))
        out.append((fr["frame"], cands))
    return out


def gather_player_boxes(frames, debug=False):
    class_hits = defaultdict(int)
    for fr in frames:
        for det in fr.get("detections", []):
            c = str(det.get("class", "")).lower()
            if c in _PLAYER_CLASS_NAMES:
                class_hits[c] += 1
    if not class_hits:
        if debug:
            log("  选手特效: 检测日志无 person/player 类, 跳过选手特效")
        return {}, None
    detected_class = max(class_hits, key=lambda k: class_hits[k])
    boxes = {}
    for fr in frames:
        bs = []
        for det in fr.get("detections", []):
            if str(det.get("class", "")).lower() != detected_class:
                continue
            if det.get("confidence", 0) < 0.35:
                continue
            b = det.get("bbox")
            if b and len(b) == 4:
                bs.append(tuple(float(v) for v in b))
        if bs:
            boxes[fr["frame"]] = bs
    if debug:
        log(f"  选手特效: 识别类名='{detected_class}', 覆盖 {len(boxes)} 帧")
    return boxes, detected_class


def filter_static_tracks(ball_cands, min_track_frames=15, static_max_motion=3.0, debug=False):
    """轨迹级静止过滤 (跟踪版):
    按 track_id(≥0) 聚合所有 (fidx, cx, cy), 对持续帧数 ≥ min_track_frames 的轨迹,
    按帧序计算相邻点位移, 若最大位移 < static_max_motion 则判定整条轨迹静止,
    整体剔除。输入/输出均为 5 元组 (cx, cy, conf, bbox, track_id) 结构。"""
    if not ball_cands:
        return ball_cands
    # 按 track_id 聚合轨迹点
    tracks = defaultdict(list)
    for fidx, cands in ball_cands:
        for (cx, cy, _conf, _b, tid) in cands:
            if tid >= 0:
                tracks[tid].append((fidx, cx, cy))
    static_tids = set()
    removed_pts = 0
    for tid, pts in tracks.items():
        if len(pts) < min_track_frames:
            continue
        pts.sort(key=lambda x: x[0])
        # 按帧序计算相邻点位移, 取最大值
        max_motion = 0.0
        for i in range(1, len(pts)):
            d = math.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
            if d > max_motion:
                max_motion = d
        if max_motion < static_max_motion:
            static_tids.add(tid)
            removed_pts += len(pts)
    if not static_tids:
        return ball_cands
    filtered = []
    for fidx, cands in ball_cands:
        remaining = [det for det in cands if det[4] not in static_tids]
        filtered.append((fidx, remaining))
    if debug:
        log(f"  ★ 静止轨迹过滤: 剔除 {len(static_tids)} 条静止轨迹, 共 {removed_pts} 个球坐标")
    return filtered


def filter_short_lived_tracks(ball_cands, min_frames=5, debug=False):
    """短生命周期轨迹过滤 (跟踪版):
    统计每个 track_id(≥0) 出现总帧数, < min_frames 的轨迹整体剔除;
    track_id=-1 的候选一律保留。输入/输出均为 5 元组结构。"""
    if not ball_cands:
        return ball_cands
    # 统计每条轨迹出现的总帧数
    track_count = defaultdict(int)
    for _fidx, cands in ball_cands:
        for det in cands:
            if det[4] >= 0:
                track_count[det[4]] += 1
    short_tids = {tid for tid, cnt in track_count.items() if cnt < min_frames}
    if not short_tids:
        return ball_cands
    removed_pts = 0
    filtered = []
    for fidx, cands in ball_cands:
        remaining = []
        for det in cands:
            if det[4] in short_tids:
                removed_pts += 1
            else:
                remaining.append(det)
        filtered.append((fidx, remaining))
    if debug:
        log(f"  ★ 短轨迹过滤: 剔除 {len(short_tids)} 条短生命周期轨迹, 共 {removed_pts} 个球坐标")
    return filtered


def filter_ball_by_corridor(ball_cands, corridor, max_drop_ratio=0.55,
                            geo_rebuilt=False, exempt_ratio=0.3, debug=False):
    """长边走廊过滤 (跟踪版):
    兼容 4/5 元组输入 (len(det)>=5 时第 5 位为 track_id, 否则视为 -1)。
    第一遍按 track_id(≥0) 聚合, 统计每条轨迹落在走廊内的点占比 in_ratio;
    in_ratio >= exempt_ratio 的轨迹整条豁免走廊检查 (保留放高球飞出走廊的段落)。
    第二遍逐点过滤: 豁免轨迹的点直接保留, 其余按走廊几何判断。
    旧 JSONL 无 track_id 时 (全部 tid=-1) 行为与原版逐点过滤完全一致。"""
    if corridor is None or not ball_cands:
        return ball_cands
    # 第一遍: 按轨迹统计走廊内占比, 决定豁免集合
    track_total = defaultdict(int)
    track_inside = defaultdict(int)
    for _fidx, cands in ball_cands:
        for det in cands:
            tid = det[4] if len(det) >= 5 else -1
            if tid >= 0:
                track_total[tid] += 1
                if _point_in_corridor((det[0], det[1]), corridor):
                    track_inside[tid] += 1
    exempt_tids = {tid for tid, tot in track_total.items()
                   if tot > 0 and track_inside[tid] / tot >= exempt_ratio}
    # 第二遍: 逐点过滤 (豁免轨迹整条保留)
    n_before = n_after = n_exempt_extra = 0
    filtered = []
    for fidx, cands in ball_cands:
        keep = []
        for det in cands:
            n_before += 1
            tid = det[4] if len(det) >= 5 else -1
            inside = _point_in_corridor((det[0], det[1]), corridor)
            if inside or (tid >= 0 and tid in exempt_tids):
                keep.append(det)
                n_after += 1
                if not inside:
                    n_exempt_extra += 1  # 依靠豁免才保留的走廊外点 (高球段)
        filtered.append((fidx, keep))
    if n_before == 0:
        return ball_cands
    drop_ratio = 1.0 - (n_after / n_before)
    effective_max = max_drop_ratio + (0.15 if geo_rebuilt else 0.0)
    if drop_ratio > effective_max:
        log(f"  ⚠️ 走廊过滤将剔除 {drop_ratio*100:.0f}% 的球 (> 上限 "
            f"{effective_max*100:.0f}%), 判定几何不可信 → 跳过走廊过滤")
        return ball_cands
    if debug:
        log(f"  ★ 长边走廊过滤: {n_before} → {n_after} 个球检测")
        if exempt_tids:
            log(f"  ★ 轨迹走廊豁免: {len(exempt_tids)}/{len(track_total)} 条轨迹整条豁免 "
                f"(走廊内占比≥{exempt_ratio:.2f}), 额外保留走廊外点 {n_exempt_extra} 个")
    return filtered


def cluster_ball_by_court(ball_cands, frame_w, frame_h, debug=False):
    if not ball_cands:
        return ball_cands
    pts, pt_refs = [], []
    for bi, (fidx, cands) in enumerate(ball_cands):
        # 兼容 4/5 元组: 聚类只用 cx, cy, det 保持原样输出 (track_id 直通后续链路)
        for ci, det in enumerate(cands):
            pts.append([det[0], det[1]])
            pt_refs.append((bi, ci))
    if len(pts) <= 1:
        return ball_cands
    pts_arr = np.array(pts, dtype=np.float64)
    diag = math.hypot(frame_w, frame_h)
    eps = diag * 0.005
    method = "DBSCAN" if _HAS_SKLEARN else "网格连通"
    if _HAS_SKLEARN:
        labels = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit(pts_arr).labels_
        cluster_counts = {}
        for lb in labels:
            if lb >= 0:
                cluster_counts[lb] = cluster_counts.get(lb, 0) + 1
        if not cluster_counts:
            return ball_cands
        best_label = max(cluster_counts, key=lambda k: cluster_counts[k])
        keep = set(i for i, lb in enumerate(labels) if lb == best_label)
        n_clusters = len(cluster_counts)
    else:
        cell_size = max(15, int(diag / 35))
        cols = max(2, frame_w // cell_size + 1)
        rows = max(2, frame_h // cell_size + 1)
        grid = np.zeros((rows, cols), dtype=np.int32)
        for cx, cy in pts:
            r, c = int(cy / cell_size), int(cx / cell_size)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] += 1
        visited = np.zeros_like(grid, dtype=bool)
        clusters = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] > 0 and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True
                    cells = []
                    while stack:
                        cr, cc = stack.pop()
                        cells.append((cr, cc))
                        for nr, nc in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
                            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    total = int(sum(grid[r2, c2] for r2, c2 in cells))
                    clusters.append((total, set(cells)))
        if len(clusters) <= 1:
            return ball_cands
        clusters.sort(key=lambda x: x[0], reverse=True)
        main_cells = clusters[0][1]
        keep = set(i for i, (cx, cy) in enumerate(pts)
                   if (int(cy / cell_size), int(cx / cell_size)) in main_cells)
        n_clusters = len(clusters)
    filtered_dict = defaultdict(list)
    n_before, n_after = len(pts), 0
    for i in keep:
        bi, ci = pt_refs[i]
        fidx, cands_orig = ball_cands[bi]
        filtered_dict[fidx].append(cands_orig[ci])
        n_after += 1
    filtered = [(fidx, cands) for fidx, cands in sorted(filtered_dict.items())]
    if n_before > n_after:
        log(f"  球台空间聚类 ({method} eps={eps:.0f}px): {n_before} → {n_after} 个检测 ({n_clusters} 个簇)")
    elif not filtered:
        return ball_cands
    return filtered


def _greedy_match(active, cands, used, preds, tols):
    for i, tr in enumerate(active):
        px, py = preds[i]
        best_j, best_d = -1, 1e18
        for j, (cx, cy, _c, _b) in enumerate(cands):
            if used[j]:
                continue
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d < tols[i]:
            cx, cy, conf, b = cands[best_j]
            tr["pts"].append((fidx_of := active[i]["pts"][-1][0], cx, cy, conf, b))  # placeholder
    # note: replaced below by explicit frame passing


def build_main_ball_track(ball_cands, fps, frame_w, frame_h, table_geo=None, debug=False,
                          max_jump_ratio=0.05, match_algo="auto"):
    max_jump = max_jump_ratio * math.hypot(frame_w, frame_h)
    max_gap_frames = max(1, int(fps * 0.5))
    tracks = []
    algo = (match_algo or "auto").lower()
    if algo == "hungarian" and not _HAS_SCIPY_LSA:
        if debug:
            log("  [match] 请求 hungarian 但未安装 scipy, 回退 greedy")
        algo = "greedy"
    if algo == "auto":
        algo = "hungarian" if _HAS_SCIPY_LSA else "greedy"
    if debug:
        log(f"  [match] track↔detection 关联算法 = {algo}")

    def predict(track, fidx):
        pts = track["pts"]
        if len(pts) == 1:
            return pts[-1][1], pts[-1][2]
        (f0, x0, y0, *_), (f1, x1, y1, *_) = pts[-2], pts[-1]
        df = max(1, f1 - f0)
        vx, vy = (x1 - x0) / df, (y1 - y0) / df
        dt = fidx - f1
        decay = 0.85 ** max(0, dt - 1)
        return x1 + vx * dt * decay, y1 + vy * dt * decay

    _BIG = 1e9

    def tid_tol(base_tol, last_tid, cand_tid):
        """tid 约束容差: 轨迹末点与候选点 tid 均有效(≥0)但不同 → 容差乘 0.5 收紧;
        tid 相同或任一为 -1 (降级路径) → 保持原容差, 行为与旧版完全一致。"""
        if last_tid >= 0 and cand_tid >= 0 and last_tid != cand_tid:
            return base_tol * 0.5
        return base_tol

    def greedy_assign(fidx, active, cands, used, preds, tols):
        for i, tr in enumerate(active):
            px, py = preds[i]
            last_tid = tr["pts"][-1][5]
            best_j, best_d = -1, 1e18
            for j, (cx, cy, _c, _b, ctid) in enumerate(cands):
                if used[j]:
                    continue
                d = math.hypot(cx - px, cy - py)
                # 距离必须小于该候选对应的 tid 有效容差才能入选
                if d < tid_tol(tols[i], last_tid, ctid) and d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                cx, cy, conf, b, ctid = cands[best_j]
                tr["pts"].append((fidx, cx, cy, conf, b, ctid))
                used[best_j] = True

    for fidx, cands in ball_cands:
        used = [False] * len(cands)
        active = [tr for tr in tracks if 0 < (fidx - tr["pts"][-1][0]) <= max_gap_frames]
        if active and cands:
            preds, tols = [], []
            for tr in active:
                gap = fidx - tr["pts"][-1][0]
                preds.append(predict(tr, fidx))
                tols.append(max_jump * (1 + 0.1 * (gap - 1)))
            done = False
            if algo == "hungarian":
                n_a, n_c = len(active), len(cands)
                cost = np.full((n_a, n_c), _BIG, dtype=np.float64)
                for i, (px, py) in enumerate(preds):
                    ti = tols[i]
                    last_tid = active[i]["pts"][-1][5]
                    for j, (cx, cy, _c, _b, ctid) in enumerate(cands):
                        d = math.hypot(cx - px, cy - py)
                        if d < tid_tol(ti, last_tid, ctid):
                            cost[i, j] = d
                try:
                    row_ind, col_ind = _linear_sum_assignment(cost)
                except Exception as e:
                    if debug:
                        log(f"  [match] linear_sum_assignment 失败 ({e}), 当帧回退 greedy")
                else:
                    for i, j in zip(row_ind, col_ind):
                        if cost[i, j] >= _BIG:
                            continue
                        cx, cy, conf, b, ctid = cands[j]
                        active[i]["pts"].append((fidx, cx, cy, conf, b, ctid))
                        used[j] = True
                    done = True
            if not done:
                greedy_assign(fidx, active, cands, used, preds, tols)
        for j, (cx, cy, conf, b, ctid) in enumerate(cands):
            if not used[j]:
                tracks.append({"pts": [(fidx, cx, cy, conf, b, ctid)]})

    if not tracks:
        return {}
    net_line = table_geo["net_line"] if table_geo else None

    def net_cross_count(pts):
        if net_line is None:
            return 0
        (nx1, ny1), (nx2, ny2) = net_line
        count, prev_side = 0, None
        for p in pts:
            side = ((nx2 - nx1) * (p[2] - ny1) - (ny2 - ny1) * (p[1] - nx1)) > 0
            if prev_side is not None and side != prev_side:
                count += 1
            prev_side = side
        return count

    def score(tr):
        pts = tr["pts"]
        n = len(pts)
        if n < 3:
            return 0.0
        span = pts[-1][0] - pts[0][0] + 1
        coverage = n / span
        motion = sum(math.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
                     for i in range(1, n))
        avg_conf = float(np.mean([p[3] for p in pts]))
        base = n * (0.4 + coverage) * (1 + math.log1p(motion)) * (0.5 + avg_conf)
        net_boost = 1.0 + min(2.0, net_cross_count(pts) * 0.5)
        return base * net_boost

    tracks.sort(key=score, reverse=True)
    n_before_filter = len(tracks)
    if len(tracks) >= 2 and table_geo is not None:
        anchor = tracks[0]
        anchor_pts = np.array([[p[1], p[2]] for p in anchor["pts"]])
        ax_min, ay_min = anchor_pts.min(axis=0)
        ax_max, ay_max = anchor_pts.max(axis=0)
        anchor_area = max((ax_max - ax_min) * (ay_max - ay_min), 1.0)
        filtered = [anchor]
        for tr in tracks[1:]:
            tr_pts = np.array([[p[1], p[2]] for p in tr["pts"]])
            tx_min, ty_min = tr_pts.min(axis=0)
            tx_max, ty_max = tr_pts.max(axis=0)
            tr_area = (tx_max - tx_min) * (ty_max - ty_min)
            if tr_area <= 0:
                continue
            ix_min, iy_min = max(ax_min, tx_min), max(ay_min, ty_min)
            ix_max, iy_max = min(ax_max, tx_max), min(ay_max, ty_max)
            if ix_min < ix_max and iy_min < iy_max:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                iou = inter_area / (anchor_area + tr_area - inter_area)
                if iou >= 0.15:
                    filtered.append(tr)
        tracks = filtered
        if debug:
            log(f"  空间共域投票: {n_before_filter} → {len(tracks)} 条轨迹 (IoU≥0.15)")

    score_min = 6.0
    main, n_used = {}, 0
    for tr in tracks:
        if score(tr) < score_min:
            continue
        n_used += 1
        # 同帧冲突仍按置信度优先, main 值携带该点 tid (可视混画由补帧/渲染层按 tid 断开)
        for (fidx, x, y, conf, b, tid) in tr["pts"]:
            if fidx not in main or conf > main[fidx][2]:
                main[fidx] = (x, y, conf, b, tid)
    if debug:
        log(f"  候选轨迹数={n_before_filter}, 采用={n_used}, 去重后点数={len(main)}")
    return main


# =====================================================================
#  4) 补帧
# =====================================================================
def interpolate_track(main_track, fps, frame_w, frame_h, debug=False):
    if not main_track:
        return {}
    keys = sorted(main_track.keys())
    max_gap = int(fps * 0.5)
    segments = []
    cur = [keys[0]]
    for k in keys[1:]:
        if k - cur[-1] <= max_gap:
            cur.append(k)
        else:
            segments.append(cur)
            cur = [k]
    segments.append(cur)
    # filled 值扩为 4 元组 (x, y, is_interp, tid), tid 来自 main_track 第 5 位
    filled = {k: (main_track[k][0], main_track[k][1], False, main_track[k][4]) for k in keys}
    n_skip_cross_tid = 0
    for seg in segments:
        if len(seg) < 2:
            continue
        fs = np.array(seg, dtype=np.float64)
        xs = np.array([main_track[k][0] for k in seg], dtype=np.float64)
        ys = np.array([main_track[k][1] for k in seg], dtype=np.float64)
        for i in range(len(seg) - 1):
            f0, f1 = seg[i], seg[i + 1]
            if f1 - f0 <= 1:
                continue
            tid0, tid1 = main_track[f0][4], main_track[f1][4]
            # 拒绝跨 tid 插值: 断口两端 tid 均有效且不同 → 不同球, 保留断口 (渲染层自然断尾)
            if tid0 >= 0 and tid1 >= 0 and tid0 != tid1:
                n_skip_cross_tid += 1
                continue
            lo, hi = max(0, i - 1), min(len(seg), i + 3)
            wf, wx, wy = fs[lo:hi], xs[lo:hi], ys[lo:hi]
            deg = min(2, len(wf) - 1)
            try:
                px = np.polyfit(wf, wx, deg)
                py = np.polyfit(wf, wy, deg)
            except Exception:
                px = np.polyfit([f0, f1], [main_track[f0][0], main_track[f1][0]], 1)
                py = np.polyfit([f0, f1], [main_track[f0][1], main_track[f1][1]], 1)
            for f in range(f0 + 1, f1):
                xi = _clamp(float(np.polyval(px, f)), 0, frame_w - 1)
                yi = _clamp(float(np.polyval(py, f)), 0, frame_h - 1)
                # 插值点 tid 继承左端点
                filled[f] = (xi, yi, True, tid0)
    if debug:
        n_interp = sum(1 for v in filled.values() if v[2])
        log(f"  补帧: 原始 {len(main_track)} 帧 -> 填充后 {len(filled)} 帧 (插值 {n_interp})")
        if n_skip_cross_tid > 0:
            log(f"  ★ 跨 tid 断口拒绝插值: {n_skip_cross_tid} 处 (不同球之间不连线)")
    return filled


# =====================================================================
#  4b) 逐帧球速表
# =====================================================================
def compute_ball_speed_table(filled_track, table_geo, fps, frame_w, frame_h):
    if not filled_track or fps <= 0:
        return {}, (1.0, 1.0)
    keys = sorted(filled_track.keys())
    raw_px = {}
    for i in range(1, len(keys)):
        f0, f1 = keys[i - 1], keys[i]
        df = f1 - f0
        if df <= 0 or df > int(fps * 0.4) + 2:
            continue
        x0, y0 = filled_track[f0][0], filled_track[f0][1]
        x1, y1 = filled_track[f1][0], filled_track[f1][1]
        raw_px[f1] = math.hypot(x1 - x0, y1 - y0) / (df / fps)
    if not raw_px:
        return {}, (1.0, 1.0)
    vals = sorted(raw_px.values())
    p50 = vals[len(vals) // 2]
    p95 = max(vals[min(len(vals) - 1, int(len(vals) * 0.95))], p50 * 1.5, 1.0)
    smoothed = {}
    win = max(1, int(fps * 0.08))
    for i, f in enumerate(keys):
        acc = [raw_px[keys[j]] for j in range(max(0, i - win), min(len(keys), i + win + 1))
               if keys[j] in raw_px]
        if acc:
            smoothed[f] = sum(acc) / len(acc)
    table = {}
    key_index = {f: i for i, f in enumerate(keys)}
    for f in keys:
        px_s = smoothed.get(f, raw_px.get(f))
        if px_s is None:
            continue
        norm = _clamp((px_s - p50 * 0.4) / (p95 - p50 * 0.4 + 1e-6), 0.0, 1.0)
        kmh = None
        if table_geo is not None and table_geo.get("homography") is not None:
            idx = key_index[f]
            if idx >= 1:
                f0 = keys[idx - 1]
                x0, y0 = filled_track[f0][0], filled_track[f0][1]
                x1, y1 = filled_track[f][0], filled_track[f][1]
                t0 = img_to_table(table_geo, (x0, y0))
                t1 = img_to_table(table_geo, (x1, y1))
                df = f - f0
                if t0 and t1 and 0 < df <= int(fps * 0.4) + 2:
                    d_mm = math.hypot(t1[0] - t0[0], t1[1] - t0[1])
                    dt = df / fps
                    if dt > 0:
                        kmh = (d_mm / 1000.0) / dt * 3.6
        table[f] = {"px_s": px_s, "kmh": kmh, "norm": norm}
    return table, (p50, p95)


# =====================================================================
#  5) 回合检测 + 落点/击球事件
# =====================================================================
class RallyState(Enum):
    IDLE = auto()
    IN_PLAY = auto()


def _finalize_rally(rally_periods, current_rally, end_reason, rally_min_duration):
    rally_duration = current_rally["end"] - current_rally["start"]
    if not (current_rally.get("crossed_net", False) and rally_duration >= float(rally_min_duration)):
        return False
    rally_periods.append({
        "start_frame": current_rally["start_frame"],
        "end_frame": current_rally["end_frame"],
        "duration": round(rally_duration, 2),
        "ended_reason": end_reason})
    return True


def _get_net_center_x(raw_log_path):
    table_x = None
    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                det = json.loads(line)
            except Exception:
                continue
            for d in det.get("detections", []):
                if d.get("class") == "net" and len(d.get("bbox", [])) == 4:
                    return (d["bbox"][0] + d["bbox"][2]) / 2.0
                if table_x is None and d.get("class") == "table" and len(d.get("bbox", [])) == 4:
                    table_x = (d["bbox"][0] + d["bbox"][2]) / 2.0
    return table_x


def _estimate_slow_speed_threshold(ball_lookup, fps):
    real_pts = []
    for f in sorted(ball_lookup.keys()):
        entry = ball_lookup.get(f)
        if entry is None:
            continue
        (bx, by), is_real = entry
        if is_real:
            real_pts.append((f, float(bx), float(by)))
    if len(real_pts) < 30 or fps <= 0:
        return 80.0
    speeds = []
    for i in range(1, len(real_pts)):
        f0, x0, y0 = real_pts[i - 1]
        f1, x1, y1 = real_pts[i]
        df = f1 - f0
        if df < 1 or df > int(fps * 0.4) + 2:
            continue
        dt = df / fps
        if dt <= 0:
            continue
        sp = math.hypot(x1 - x0, y1 - y0) / dt
        if sp > 0:
            speeds.append(sp)
    if len(speeds) < 20:
        return 80.0
    speeds.sort()
    p15 = speeds[int(len(speeds) * 0.15)]
    p50 = speeds[len(speeds) // 2]
    return max(min(p15 * 0.6, p50 * 0.35), 80.0)


def detect_rallies(filled_track, fps, total_frames, raw_log_path, table_geo=None,
                   rally_lost_timeout=1.0, rally_min_duration=0.6, pre_roll_sec=1.0,
                   slow_speed_window_sec=1.0, slow_speed_min_in_play_sec=0.8,
                   out_of_table_timeout=1.0, debug=False):
    if not filled_track or fps <= 0:
        return []
    ball_lookup = {f: ((v[0], v[1]), not v[2]) for f, v in filled_track.items()}
    sorted_frames = sorted(ball_lookup.keys())
    net_line = table_geo.get("net_line") if table_geo else None
    net_center_x = _get_net_center_x(raw_log_path) if net_line is None else None
    if net_line is None and net_center_x is None:
        log("  ⚠️ 未提取到球网, 回退到连续性切分")
        keys = sorted_frames
        max_gap = int(fps * 0.45)
        rallies = []
        cur = [keys[0]]
        for k in keys[1:]:
            if k - cur[-1] <= max_gap:
                cur.append(k)
            else:
                rallies.append({"start_frame": cur[0], "end_frame": cur[-1]})
                cur = [k]
        rallies.append({"start_frame": cur[0], "end_frame": cur[-1]})
        return rallies
    slow_thresh = _estimate_slow_speed_threshold(ball_lookup, fps)
    corridor = table_geo.get("corridor") if table_geo else None
    OUT_OF_TABLE_FRAMES = max(1, int(round(fps * out_of_table_timeout)))
    if debug:
        log(f"  慢速阈值: {slow_thresh:.0f} px/s")
    PRE_ROLL_FRAMES = int(round(fps * pre_roll_sec))
    LOST_FRAMES = int(round(fps * rally_lost_timeout))
    SLOW_WIN_FRAMES = max(2, int(round(fps * slow_speed_window_sec)))
    SLOW_MIN_INPLAY_FRAMES = int(round(fps * slow_speed_min_in_play_sec))
    first_frame = sorted_frames[0]
    last_frame = max(sorted_frames[-1], total_frames - 1)
    rally_state = RallyState.IDLE
    rally_num = 0
    current_rally = None
    missing_ball_frames = 0
    prev_side = None
    slow_run_frames = 0
    out_of_table_run = 0
    rallies = []
    traj_buffer = deque(maxlen=max(PRE_ROLL_FRAMES * 3, int(fps * 6)))

    def _ball_side(x, y):
        if net_line is not None:
            (nx1, ny1), (nx2, ny2) = net_line
            return 'A' if (nx2 - nx1) * (y - ny1) - (ny2 - ny1) * (x - nx1) > 0 else 'B'
        if net_center_x is not None:
            return 'L' if x < net_center_x else 'R'
        return None

    def _recent_avg_speed(buf, frame_idx, win_frames):
        lo_frame = frame_idx - win_frames
        pts = [p for p in buf if p[0] >= lo_frame and p[4]]
        if len(pts) < 2:
            return None
        spds = []
        for i in range(1, len(pts)):
            f0, _, x0, y0, _, _ = pts[i - 1]
            f1, _, x1, y1, _, _ = pts[i]
            df = f1 - f0
            if df <= 0 or df > int(fps * 0.4) + 2:
                continue
            spds.append(math.hypot(x1 - x0, y1 - y0) / (df / fps))
        return sum(spds) / len(spds) if spds else None

    for frame_idx in range(first_frame, last_frame + 1):
        entry = ball_lookup.get(frame_idx)
        if entry is not None:
            (bx, by), is_real = entry
            side = _ball_side(bx, by)
            traj_buffer.append((frame_idx, frame_idx / fps, float(bx), float(by), bool(is_real), side))
            missing_ball_frames = 0
        else:
            is_real, side = False, None
            missing_ball_frames += 1
        if rally_state == RallyState.IDLE:
            if entry is not None and prev_side is not None and side is not None and side != prev_side:
                rally_num += 1
                start_frame = max(0, frame_idx - PRE_ROLL_FRAMES)
                current_rally = {"start_frame": start_frame, "end_frame": frame_idx, "crossed_net": True}
                rally_state = RallyState.IN_PLAY
                slow_run_frames = out_of_table_run = 0
                while traj_buffer and traj_buffer[0][0] < start_frame:
                    traj_buffer.popleft()
            if entry is not None:
                prev_side = side
        elif rally_state == RallyState.IN_PLAY:
            if current_rally is None:
                continue
            if entry is not None:
                current_rally["end_frame"] = frame_idx
                if prev_side is not None and side is not None and side != prev_side:
                    current_rally["crossed_net"] = True
                prev_side = side
            end_by_lost = (missing_ball_frames >= LOST_FRAMES)
            end_by_slow = False
            in_play_frames = frame_idx - current_rally["start_frame"]
            if in_play_frames >= SLOW_MIN_INPLAY_FRAMES and entry is not None and is_real:
                avg_sp = _recent_avg_speed(traj_buffer, frame_idx, SLOW_WIN_FRAMES)
                if avg_sp is not None and avg_sp < slow_thresh:
                    slow_run_frames += 1
                else:
                    slow_run_frames = 0
                if slow_run_frames >= SLOW_WIN_FRAMES:
                    end_by_slow = True
            end_by_out = False
            if corridor is not None:
                if entry is not None:
                    if _point_in_corridor((bx, by), corridor):
                        out_of_table_run = 0
                    else:
                        out_of_table_run += 1
                if out_of_table_run >= OUT_OF_TABLE_FRAMES:
                    end_by_out = True
            if end_by_lost or end_by_slow or end_by_out:
                if end_by_lost:
                    end_reason = "ball_lost"
                elif end_by_slow:
                    end_reason = "ball_too_slow"
                else:
                    end_reason = "ball_out_of_table"
                if end_by_lost and missing_ball_frames > 0:
                    end_f = current_rally["start_frame"]
                    for f in range(frame_idx - missing_ball_frames, current_rally["start_frame"] - 1, -1):
                        if ball_lookup.get(f) is not None:
                            end_f = f
                            break
                elif end_by_out:
                    end_f = frame_idx
                    for f in range(frame_idx, current_rally["start_frame"] - 1, -1):
                        e = ball_lookup.get(f)
                        if e is not None and _point_in_corridor((e[0][0], e[0][1]), corridor):
                            end_f = f
                            break
                else:
                    end_f = frame_idx
                current_rally["end_frame"] = end_f
                current_rally["end"] = end_f / fps
                current_rally["start"] = current_rally["start_frame"] / fps
                _finalize_rally(rallies, current_rally, end_reason, rally_min_duration)
                rally_state = RallyState.IDLE
                current_rally = None
                slow_run_frames = out_of_table_run = 0
                prev_side = side
    if rally_state == RallyState.IN_PLAY and current_rally is not None:
        current_rally["start"] = current_rally["start_frame"] / fps
        current_rally["end"] = current_rally["end_frame"] / fps
        _finalize_rally(rallies, current_rally, "video_end", rally_min_duration)
    if debug:
        log(f"  回合检测 (跨网+球速+出台): {len(rallies)} 个回合")
        for ri, r in enumerate(rallies):
            log(f"    回合{ri}: [{r['start_frame']}-{r['end_frame']}] "
                f"{r['duration']}s 结束原因={r['ended_reason']}")
    return rallies


def detect_events(filled_track, table_geo, fps):
    events = []
    if not filled_track:
        return events
    keys = sorted(filled_track.keys())
    pts = [(k, filled_track[k][0], filled_track[k][1]) for k in keys]
    n = len(pts)
    if n < 5:
        return events

    def speed_kmh(k0, k1, p0, p1):
        if table_geo is None:
            return None
        t0 = img_to_table(table_geo, (p0[1], p0[2]))
        t1 = img_to_table(table_geo, (p1[1], p1[2]))
        if t0 is None or t1 is None:
            return None
        d_mm = math.hypot(t1[0] - t0[0], t1[1] - t0[1])
        dt = (k1 - k0) / fps
        if dt <= 0:
            return None
        return (d_mm / 1000.0) / dt * 3.6

    quad = table_geo.get("quad") if table_geo else None

    def _in_table(x, y):
        return True if quad is None else _point_in_corridor((x, y), quad)

    _TX_MAX = TABLE_WID_M * 1000
    _TY_MAX = TABLE_LEN_M * 1000
    _SURFACE_TOL = 80

    def _on_table_surface(x, y):
        if table_geo is None:
            return True
        txy = img_to_table(table_geo, (x, y))
        if txy is None:
            return False
        tx, ty = txy
        return (-_SURFACE_TOL <= tx <= _TX_MAX + _SURFACE_TOL and
                -_SURFACE_TOL <= ty <= _TY_MAX + _SURFACE_TOL)

    last_bounce_frame = -10 ** 9
    bounce_cooldown_frames = max(1, int(round(fps * 0.1)))
    for i in range(2, n - 2):
        f, x, y = pts[i]
        f_p, x_p, y_p = pts[i - 1]
        f_n, x_n, y_n = pts[i + 1]
        ev_type = None
        is_local_y_max = (y >= pts[i - 1][2] and y >= pts[i + 1][2] and
                          y > pts[i - 2][2] and y > pts[i + 2][2])
        vy_prev, vy_next = y - y_p, y_n - y
        if is_local_y_max and vy_prev > 0.1 and vy_next < -0.1:
            if (f - last_bounce_frame >= bounce_cooldown_frames and
                    _in_table(x, y) and _on_table_surface(x, y)):
                ev_type = "bounce"
                last_bounce_frame = f
        if ev_type is None:
            vx_prev, vx_next = x - x_p, x_n - x
            if vx_prev * vx_next < 0 and abs(vx_prev) > 2.0 and abs(vx_next) > 2.0:
                if _in_table(x, y):
                    ev_type = "hit"
        if ev_type:
            spd = speed_kmh(f_p, f_n, pts[i - 1], pts[i + 1])
            txy = img_to_table(table_geo, (x, y)) if table_geo is not None else None
            events.append({"frame": f, "x": x, "y": y, "type": ev_type,
                           "speed_kmh": spd, "table_xy": txy})
    return events


# =====================================================================
#  5b) 落点半台判定 + 两半台交替过滤
# =====================================================================
def _landing_half(landing):
    txy = landing.get("table_xy")
    if txy is None:
        return None
    return 'FAR' if txy[1] < TABLE_LEN_M * 1000 / 2.0 else 'NEAR'


def filter_alternating_landings(landings, debug=False):
    if not landings:
        return []
    ordered = sorted(landings, key=lambda lp: lp["frame"])
    kept, last_half, n_drop = [], None, 0
    for lp in ordered:
        half = _landing_half(lp)
        if half is None or half == last_half:
            n_drop += 1
            continue
        kept.append(lp)
        last_half = half
    if debug and n_drop > 0:
        log(f"    落点交替过滤: {len(ordered)} → {len(kept)} 个落点 (剔除 {n_drop})")
    return kept


def build_landing_rally_index(segments, rally_starts_by_seg, landing_all, debug=False):
    def seg_of_frame(f):
        for i, (s, e) in enumerate(segments):
            if s <= f <= e:
                return i
        return -1

    raw_by_seg = defaultdict(list)
    for lp in landing_all:
        si = seg_of_frame(lp["frame"])
        if si >= 0:
            raw_by_seg[si].append(lp)

    landing_by_seg = defaultdict(list)
    landing_rally_id = {}
    for si, (s, e) in enumerate(segments):
        seg_landings = sorted(raw_by_seg.get(si, []), key=lambda lp: lp["frame"])
        if not seg_landings:
            continue
        rally_starts = sorted(rally_starts_by_seg.get(si, []))
        bounds = rally_starts if rally_starts else [s]
        if debug:
            log(f"    片段{si}: 落点交替过滤 (回合边界帧={bounds})")
        for bi in range(len(bounds)):
            lo = bounds[bi]
            hi = bounds[bi + 1] if bi + 1 < len(bounds) else (e + 1)
            group = [lp for lp in seg_landings if lo <= lp["frame"] < hi]
            if not group:
                continue
            kept = filter_alternating_landings(group, debug=debug)
            for lp in kept:
                landing_rally_id[(si, lp["frame"])] = bi
            landing_by_seg[si].extend(kept)
        landing_by_seg[si].sort(key=lambda lp: lp["frame"])
    return landing_by_seg, landing_rally_id


# =====================================================================
#  6) 剪辑片段计算
# =====================================================================
def build_segments(rallies, fps, total_frames, args):
    min_frames = args.min_rally_frames or int(fps * 2)
    pre = int(args.pre_buffer * fps)
    post = int(args.post_buffer * fps)
    merge_gap = int(args.merge_gap * fps)
    valid_rallies = [r for r in rallies
                     if (r["end_frame"] - r["start_frame"] + 1) >= min_frames]
    if not valid_rallies:
        return [], {}
    buffed = []
    for r in valid_rallies:
        bs = max(0, r["start_frame"] - pre)
        be = min(total_frames - 1, r["end_frame"] + post)
        buffed.append((bs, be, r["start_frame"]))
    buffed.sort(key=lambda x: (x[0], x[1]))
    merged = [[buffed[0][0], buffed[0][1], [buffed[0][2]]]]
    for bs, be, raw_s in buffed[1:]:
        if bs - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], be)
            merged[-1][2].append(raw_s)
        else:
            merged.append([bs, be, [raw_s]])
    segments = [(int(s), int(e)) for s, e, _ in merged]
    rally_starts_by_seg = {si: sorted(set(int(x) for x in raw_starts))
                           for si, (_, _, raw_starts) in enumerate(merged)}
    return segments, rally_starts_by_seg


# =====================================================================
#  7) 渲染 — 特效系统
# =====================================================================
class MiniTable:
    def __init__(self, frame_w, frame_h):
        self.h = int(frame_h * 0.30)
        self.w = int(self.h / TABLE_RATIO)
        self.margin = int(frame_w * 0.015)
        self.x0 = frame_w - self.w - self.margin
        self.y0 = self.margin
        self.sx = self.w / (TABLE_WID_M * 1000)
        self.sy = self.h / (TABLE_LEN_M * 1000)
        self._bg = np.full((self.h, self.w, 3), (60, 90, 60), dtype=np.uint8)
        self._x_max = TABLE_WID_M * 1000
        self._y_max = TABLE_LEN_M * 1000

    def table_to_mini(self, txy):
        if txy is None:
            return None
        X, Y = txy
        _tol = 50
        if not (-_tol <= X <= self._x_max + _tol and -_tol <= Y <= self._y_max + _tol):
            return None
        X = _clamp(X, 0.0, self._x_max)
        Y = _clamp(Y, 0.0, self._y_max)
        return int(self.x0 + X * self.sx), int(self.y0 + Y * self.sy)

    def draw(self, frame, current_rally_points):
        roi = frame[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        cv2.addWeighted(self._bg, 0.55, roi, 0.45, 0, roi)
        cv2.rectangle(frame, (self.x0, self.y0),
                      (self.x0 + self.w, self.y0 + self.h), (255, 255, 255), 2)
        midy = self.y0 + self.h // 2
        cv2.line(frame, (self.x0, midy), (self.x0 + self.w, midy), (255, 255, 255), 2)
        dash_len = max(4, self.w // 12)
        for dx in range(0, self.w, dash_len * 2):
            x_start = self.x0 + dx
            x_end = min(self.x0 + dx + dash_len, self.x0 + self.w)
            cv2.line(frame, (x_start, midy), (x_end, midy), (180, 220, 255), 1)
        cv2.putText(frame, "NET", (self.x0 + self.w + 3, midy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 220, 255), 1)
        midx = self.x0 + self.w // 2
        cv2.line(frame, (midx, self.y0), (midx, self.y0 + self.h), (200, 200, 200), 1)
        cv2.putText(frame, "FAR", (self.x0 + 2, self.y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, "NEAR", (self.x0 + 2, self.y0 + self.h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        mpts = []
        for lp in current_rally_points:
            mp = self.table_to_mini(lp.get("table_xy"))
            if mp:
                mpts.append(mp)
        for i in range(1, len(mpts)):
            cv2.line(frame, mpts[i - 1], mpts[i], (0, 180, 255), 1, cv2.LINE_AA)
        for i, mp in enumerate(mpts):
            if i == len(mpts) - 1:
                cv2.circle(frame, mp, 7, (0, 255, 255), -1)
                cv2.circle(frame, mp, 8, (0, 0, 0), 1)
            else:
                cv2.circle(frame, mp, 5, (0, 215, 255), -1)
                cv2.circle(frame, mp, 6, (0, 0, 0), 1)
        cv2.putText(frame, "TABLE", (self.x0, self.y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if current_rally_points:
            spd = current_rally_points[-1].get("speed_kmh")
            if spd:
                speed_text = f"{spd:.0f} km/h"
                (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                sx = self.x0 + self.w - tw - 4
                sy = self.y0 + self.h + th + 4
                cv2.putText(frame, speed_text, (sx, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


class ParticleSystem:
    def __init__(self, intensity=1.0):
        self.particles = []
        self.intensity = intensity

    def emit(self, x, y, n=18, kind="bounce", speed_norm=0.3):
        n = int(n * self.intensity)
        base_col = speed_to_color(speed_norm)
        spd_scale = 1.0 + speed_norm * 1.6
        for _ in range(n):
            ang = np.random.uniform(0, 2 * math.pi)
            spd = np.random.uniform(1.5, 6.0) * spd_scale
            if kind == "bounce":
                vy = -abs(math.sin(ang) * spd) - 1.0
                vx = math.cos(ang) * spd
                color = _lerp_color((230, 240, 255), base_col, 0.5 * speed_norm)
            else:
                vx, vy = math.cos(ang) * spd, math.sin(ang) * spd
                color = base_col
            life = np.random.randint(8, 20 + int(speed_norm * 12))
            self.particles.append({"x": x, "y": y, "vx": vx, "vy": vy,
                                   "life": life, "maxlife": life,
                                   "size": np.random.uniform(1.0, 3.0 + speed_norm * 1.5),
                                   "color": color})

    def update_and_draw(self, frame):
        alive = []
        h, w = frame.shape[:2]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.35
            p["vx"] *= 0.96
            p["life"] -= 1
            if p["life"] <= 0:
                continue
            a = p["life"] / p["maxlife"]
            col = tuple(int(c * a) for c in p["color"])
            r = max(1, int(p["size"] * a))
            xi, yi = int(p["x"]), int(p["y"])
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(frame, (xi, yi), r, col, -1, cv2.LINE_AA)
            alive.append(p)
        self.particles = alive


class ShockwaveSystem:
    def __init__(self, intensity=1.0):
        self.waves = []
        self.intensity = intensity

    def emit(self, x, y, kind="hit", speed_norm=0.3):
        #base_col = speed_to_color(speed_norm)
        n_rings = 1 + int(round(speed_norm * 2 * self.intensity))
        life = int(10 + speed_norm * 14)
        for k in range(max(1, n_rings)):
            self.waves.append({"x": x, "y": y, "r": 0.01 + k * 0.08,
                               "vr": (0.05 + speed_norm * 0.15) * (1.0 - 0.15 * k),
                               "life": life, "maxlife": life,
                               "color": (255, 255, 255),
                               "thick": 2 if kind == "hit" else 1})

    def update_and_draw(self, frame):
        alive = []
        h, w = frame.shape[:2]
        for wv in self.waves:
            wv["r"] += wv["vr"]
            wv["vr"] *= 0.94
            wv["life"] -= 1
            if wv["life"] <= 0:
                continue
            a = wv["life"] / wv["maxlife"]
            xi, yi, rr = int(wv["x"]), int(wv["y"]), int(wv["r"])
            if rr <= 0 or not (0 <= xi < w and 0 <= yi < h):
                alive.append(wv)
                continue
            #col = tuple(int(c * a) for c in wv["color"])
            cv2.circle(
                        frame, (xi, yi), rr,                 # ← col → (255, 255, 255)
                        (255, 255, 255),                     # ← 固定白色
                        max(1, int(wv["thick"] * a + 0.5)),  # ← thick 随透明度变细
                        cv2.LINE_AA
                        )
            alive.append(wv)
        self.waves = alive


class FloatingLabel:
    def __init__(self):
        self.labels = []

    def emit(self, x, y, text, color=(0, 200, 255), scale=1.0, life=22):
        self.labels.append({"x": x, "y": y, "text": text, "color": color,
                            "scale": scale, "life": life, "maxlife": life})

    def update_and_draw(self, frame):
        alive = []
        h, w = frame.shape[:2]
        for lb in self.labels:
            lb["y"] -= 1.4
            lb["life"] -= 1
            if lb["life"] <= 0:
                continue
            a = lb["life"] / lb["maxlife"]
            fs = 0.6 * lb["scale"] * (0.6 + 0.4 * a)
            col = tuple(int(c * a) for c in lb["color"])
            (tw, th), _ = cv2.getTextSize(lb["text"], cv2.FONT_HERSHEY_DUPLEX, fs, 2)
            px = int(_clamp(lb["x"] - tw / 2, 2, w - tw - 2))
            py = int(_clamp(lb["y"], th + 2, h - 2))
            cv2.putText(frame, lb["text"], (px, py), cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, lb["text"], (px, py), cv2.FONT_HERSHEY_DUPLEX, fs, col, 2, cv2.LINE_AA)
            alive.append(lb)
        self.labels = alive


class SpeedFlash:
    def __init__(self):
        self._t = 0
        self._max = 0
        self._mask_cache = {}

    def trigger(self, strength_frames=8):
        self._t = max(self._t, strength_frames)
        self._max = max(self._max, strength_frames)

    def _get_mask(self, h, w):
        key = (h, w)
        if key not in self._mask_cache:
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w / 2.0, h / 2.0
            d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            m = np.clip((d - 0.55) / 0.6, 0.0, 1.0).astype(np.float32)
            self._mask_cache[key] = m[..., None]
        return self._mask_cache[key]

    def update_and_draw(self, frame):
        if self._t <= 0:
            return
        a = self._t / max(1, self._max)
        h, w = frame.shape[:2]
        mask = self._get_mask(h, w)
        tint = np.array([40, 120, 255], dtype=np.float32)
        strength = 0.5 * a
        f = frame.astype(np.float32)
        f = f * (1.0 - mask * strength) + tint * (mask * strength)
        frame[:] = np.clip(f, 0, 255).astype(np.uint8)
        self._t -= 1


class BounceMarker:
    """主画面落点脉冲标记: 高透明度白色扩散圈 (半径较小) + 细白十字。"""
    def __init__(self):
        self.marks = []

    def emit(self, x, y):
        self.marks.append({"x": x, "y": y, "life": 16, "maxlife": 16})

    def update_and_draw(self, frame):
        alive = []
        h, w = frame.shape[:2]
        for m in self.marks:
            m["life"] -= 1
            if m["life"] <= 0:
                continue
            a = m["life"] / m["maxlife"]
            xi, yi = int(m["x"]), int(m["y"])
            r = int((1 - a) * 12 + 2)
            pad = r + 3
            y0, y1 = max(0, yi - pad), min(h, yi + pad + 1)
            x0, x1 = max(0, xi - pad), min(w, xi + pad + 1)
            if y0 >= y1 or x0 >= x1:
                alive.append(m)
                continue
            roi = frame[y0:y1, x0:x1]
            ov = roi.copy()
            lcx, lcy = xi - x0, yi - y0
            cv2.circle(ov, (lcx, lcy), max(1, r), (255, 255, 255), 1, cv2.LINE_AA)
            cl = int(6 * a + 2)
            cv2.line(ov, (lcx - cl, lcy), (lcx + cl, lcy), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(ov, (lcx, lcy - cl), (lcx, lcy + cl), (255, 255, 255), 1, cv2.LINE_AA)
            alpha = 0.28 * a
            cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)
            alive.append(m)
        self.marks = alive


class SpeedTrailRenderer:
    """速度自适应彗星拖尾 (细/淡/偏白)。"""
    def __init__(self, intensity=1.0):
        self._cache = {}
        self.intensity = intensity

    def draw(self, frame, trail_pts, speed_norm=0.3):
        n = len(trail_pts)
        if n < 2:
            return
        h, w = frame.shape[:2]
        key = (h, w)
        if key not in self._cache:
            self._cache[key] = (np.empty((h, w, 3), dtype=np.uint8),
                                np.zeros((h, w), dtype=np.float32))
        overlay, alpha_mask = self._cache[key]
        xs = [p[0] for p in trail_pts]
        ys = [p[1] for p in trail_pts]
        pad = 16
        x0 = max(0, int(min(xs)) - pad)
        x1 = min(w, int(max(xs)) + pad + 1)
        y0 = max(0, int(min(ys)) - pad)
        y1 = min(h, int(max(ys)) + pad + 1)
        alpha_mask[y0:y1, x0:x1] = 0.0
        overlay[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        trail_col = speed_to_color(speed_norm)
        thick_scale = (1.0 + speed_norm * 1.8) * (2.0 / 3.0)
        # 距离突变兜底: 相邻点距离超过画面对角线 6% 时该段不连线
        # (防 tid 缺失的降级路径跨球连线, 后续段照常继续)
        max_seg_dist = 0.06 * math.hypot(w, h)
        for i in range(1, n):
            a = i / n
            alpha = (0.15 + 0.55 * a) * (0.85 + 0.15 * self.intensity)
            thick = max(1, int((1 + 4 * a) * thick_scale))
            p0 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
            p1 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > max_seg_dist:
                continue
            seg_col = _lerp_color((255, 255, 255), trail_col,
                                  a * (0.55 * 0.6 + 0.45 * speed_norm) * 0.75)
            cv2.line(overlay, p0, p1, seg_col, thick, cv2.LINE_AA)
            cv2.line(alpha_mask, p0, p1, alpha, thick, cv2.LINE_AA)
        roi_a = alpha_mask[y0:y1, x0:x1]
        if (roi_a > 0).any():
            roi_f = frame[y0:y1, x0:x1]
            roi_o = overlay[y0:y1, x0:x1]
            a3 = roi_a[..., np.newaxis]
            blended = roi_f.astype(np.float32) * (1.0 - a3) + roi_o.astype(np.float32) * a3
            roi_f[:] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        cx, cy = int(trail_pts[-1][0]), int(trail_pts[-1][1])
        glow_r = int((3 + speed_norm * 9 * self.intensity) * (2.0 / 3.0))
        if speed_norm > 0.3 and glow_r >= 2:
            gy0, gy1 = max(0, cy - glow_r * 2), min(h, cy + glow_r * 2 + 1)
            gx0, gx1 = max(0, cx - glow_r * 2), min(w, cx + glow_r * 2 + 1)
            if gy0 < gy1 and gx0 < gx1:
                glow = frame[gy0:gy1, gx0:gx1].copy()
                lcx, lcy = cx - gx0, cy - gy0
                glow_col = _lerp_color((255, 255, 255), trail_col, 0.6)
                for layer in range(3):
                    rr = glow_r - layer * (glow_r // 3 + 1)
                    if rr < 1:
                        continue
                    cv2.circle(glow, (lcx, lcy), rr, glow_col, -1, cv2.LINE_AA)
                cv2.addWeighted(glow, 0.28, frame[gy0:gy1, gx0:gx1], 0.72, 0, frame[gy0:gy1, gx0:gx1])
        # 残影末端的实心白点 (球头标记), 已注释掉: 低速/残影淡出时会在原地闪一个白点
        # cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)


class PlayerFX:
    """选手移动特效: 仅脚下光环 (随移动速度变化), 仅在局部 ROI 混合。"""
    def __init__(self, fps, frame_w, frame_h, intensity=1.0):
        self.fps = fps
        self.diag = math.hypot(frame_w, frame_h)
        self.intensity = intensity
        self.fast_thresh = self.diag * 0.012
        self.aura_phase = 0.0

    def draw(self, frame, boxes_prev, boxes_cur):
        if not boxes_cur:
            return
        h, w = frame.shape[:2]
        self.aura_phase += 0.35

        def speed_of(box):
            if not boxes_prev:
                return 0.0
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            best, bp = 1e18, None
            for pb in boxes_prev:
                pcx = (pb[0] + pb[2]) * 0.5
                pcy = (pb[1] + pb[3]) * 0.5
                d = math.hypot(cx - pcx, cy - pcy)
                if d < best:
                    best, bp = d, (pcx, pcy)
            if bp is None or best > self.diag * 0.15:
                return 0.0
            return best

        for box in boxes_cur:
            x1, y1, x2, y2 = [int(v) for v in box]
            cx = (x1 + x2) // 2
            snorm = _clamp(speed_of(box) / (self.fast_thresh + 1e-6), 0.0, 1.0)
            foot_y = y2
            ring_r = int((x2 - x1) * (0.35 + 0.15 * snorm) * 0.5)
            if ring_r <= 3:
                continue
            pulse = 0.6 + 0.4 * math.sin(self.aura_phase)
            ring_col = speed_to_color(snorm)
            ring_h = max(3, ring_r // 3)
            # ★ 仅在光环外接矩形 ROI 上混合 (避免整帧 copy)
            ry0 = max(0, foot_y - ring_h - 2)
            ry1 = min(h, foot_y + ring_h + 3)
            rx0 = max(0, cx - ring_r - 2)
            rx1 = min(w, cx + ring_r + 3)
            if ry0 >= ry1 or rx0 >= rx1:
                continue
            roi = frame[ry0:ry1, rx0:rx1]
            ov = roi.copy()
            cv2.ellipse(ov, (cx - rx0, foot_y - ry0), (ring_r, ring_h),
                        0, 0, 360, ring_col, 2, cv2.LINE_AA)
            blend = 0.35 * (0.5 + 0.5 * pulse) * self.intensity
            cv2.addWeighted(ov, blend, roi, 1 - blend, 0, roi)


def smooth_trail(points):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    xs = np.convolve(pts[:, 0], _SMOOTH_KERNEL, mode="same")
    ys = np.convolve(pts[:, 1], _SMOOTH_KERNEL, mode="same")
    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    return list(zip(xs, ys))


def render(args, segments, rally_starts_by_seg, filled_track, events, table_geo,
           in_w, in_h, fps, total_frames, ball_speed_table, player_boxes):
    out_h = args.output_height or in_h
    out_w = int(round(in_w * out_h / in_h))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    scale = out_w / in_w

    fx_level = args.fx_level
    fx_intensity = {"minimal": 0.7, "max": 1.4}.get(fx_level, 1.0)
    enable_extra_fx = (fx_level != "minimal")
    enable_player_fx = (not args.no_player_fx) and bool(player_boxes) and enable_extra_fx

    ev_by_frame = defaultdict(list)
    landing_all = []
    for ev in events:
        ev_by_frame[ev["frame"]].append(ev)
        if ev["type"] == "bounce":
            landing_all.append(ev)

    kmh_vals = [ev["speed_kmh"] for ev in events if ev.get("speed_kmh")]
    kmh_p95 = 1.0
    if kmh_vals:
        kmh_vals_sorted = sorted(kmh_vals)
        kmh_p95 = max(1.0, kmh_vals_sorted[min(len(kmh_vals_sorted) - 1,
                                               int(len(kmh_vals_sorted) * 0.9))])

    def event_speed_norm(ev):
        s = ev.get("speed_kmh")
        if s and kmh_p95 > 0:
            return _clamp(s / kmh_p95, 0.0, 1.0)
        st = ball_speed_table.get(ev["frame"])
        return st["norm"] if st else 0.3

    scaled_ev_by_frame = defaultdict(list)
    for f, evs in ev_by_frame.items():
        for ev in evs:
            scaled_ev_by_frame[f].append({
                "x": ev["x"] * scale, "y": ev["y"] * scale, "type": ev["type"],
                "snorm": event_speed_norm(ev), "kmh": ev.get("speed_kmh")})

    landing_by_seg, landing_rally_id = build_landing_rally_index(
        segments, rally_starts_by_seg, landing_all, debug=args.debug)

    # scaled_track 同步携带 tid: (x*scale, y*scale, tid), 供渲染层按 tid 断开残影
    scaled_track = {f: (v[0] * scale, v[1] * scale, v[3]) for f, v in filled_track.items()}

    tmp_dir = tempfile.mkdtemp(prefix="pphl_")
    silent_path = os.path.join(tmp_dir, "silent.mp4")
    enc = detect_amf_encoder(args.encoder)
    use_pipe = False
    pipe_proc = None
    writer = None
    cap = None
    try:
        if shutil.which("ffmpeg"):
            try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-thread_queue_size", "512",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
                ] + _enc_args(enc) + ["-pix_fmt", "yuv420p", silent_path]
                pipe_proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE, creationflags=_SUBPROC_FLAGS)
                use_pipe = True
            except Exception:
                use_pipe = False
        if not use_pipe:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError("无法创建临时视频写入器")

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {args.video}")
        mini = MiniTable(out_w, out_h)
        particles = ParticleSystem(intensity=fx_intensity)
        shockwaves = ShockwaveSystem(intensity=fx_intensity)
        labels = FloatingLabel()
        flash = SpeedFlash()
        bmarker = BounceMarker()
        trail_renderer = SpeedTrailRenderer(intensity=fx_intensity)
        player_fx = PlayerFX(fps, out_w, out_h, intensity=fx_intensity) if enable_player_fx else None
        TRAIL_LEN = max(12, int(fps * 0.5))

        total_render = sum(e - s + 1 for s, e in segments)
        pipe_method = "ffmpeg pipe 直编" if use_pipe else "cv2.VideoWriter"
        log(f"🎬 渲染: {len(segments)} 个片段, 约 {total_render} 帧 @ {out_w}x{out_h} "
            f"{fps:.2f}fps [{pipe_method} → {enc}]  fx={fx_level}"
            f"{' +选手光环' if enable_player_fx else ''}")
        written = 0
        t_start = time.time()
        last_progress_time = t_start

        do_resize = (scale != 1.0)
        for si, (s, e) in enumerate(segments):
            cap.set(cv2.CAP_PROP_POS_FRAMES, s)
            trail = deque(maxlen=TRAIL_LEN)
            cur_landings_sorted = sorted(landing_by_seg.get(si, []), key=lambda lp: lp["frame"])
            lp_idx = 0
            cur_lp_so_far = []
            cur_rally_id = None
            prev_player_boxes = None
            for f in range(s, e + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                if do_resize:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                for sev in scaled_ev_by_frame.get(f, []):
                    snorm = sev["snorm"]
                    is_bounce = (sev["type"] == "bounce")
                    particles.emit(sev["x"], sev["y"], n=22 if is_bounce else 16,
                                   kind=sev["type"], speed_norm=snorm)
                    if enable_extra_fx:
                        shockwaves.emit(sev["x"], sev["y"], kind=sev["type"], speed_norm=snorm)
                        if is_bounce:
                            bmarker.emit(sev["x"], sev["y"])
                        if not is_bounce and snorm >= 0.6:
                            if snorm >= 0.85:
                                txt, col = "SMASH!", (60, 80, 255)
                                flash.trigger(int(7 * fx_intensity))
                            else:
                                txt, col = "FAST!", (60, 200, 255)
                            labels.emit(sev["x"], sev["y"] - 20, txt,
                                        color=col, scale=1.0 + snorm * 0.5)

                if player_fx is not None:
                    cur_boxes = player_boxes.get(f, [])
                    cur_boxes_scaled = [(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale)
                                        for b in cur_boxes]
                    player_fx.draw(frame, prev_player_boxes, cur_boxes_scaled)
                    prev_player_boxes = cur_boxes_scaled

                if f in scaled_track:
                    pt = scaled_track[f]
                    # ★ tid 断开: 新点与残影末点 tid 均有效且不同 → 清空残影, 从新球重新起笔
                    # (tid=-1 参与时不触发清空, 降级路径行为不变)
                    if trail and pt[2] >= 0 and trail[-1][2] >= 0 and trail[-1][2] != pt[2]:
                        trail.clear()
                    trail.append(pt)
                elif trail and len(trail) > 1:
                    trail.popleft()
                if len(trail) >= 2:
                    st = ball_speed_table.get(f)
                    snorm = st["norm"] if st else 0.3
                    # 平滑前剥离 tid, 只传坐标给拖尾渲染
                    trail_renderer.draw(frame, smooth_trail([(p[0], p[1]) for p in trail]),
                                        speed_norm=snorm)

                #shockwaves.update_and_draw(frame)
                particles.update_and_draw(frame)
                #bmarker.update_and_draw(frame)
                labels.update_and_draw(frame)
                flash.update_and_draw(frame)

                while lp_idx < len(cur_landings_sorted) and cur_landings_sorted[lp_idx]["frame"] <= f:
                    lp = cur_landings_sorted[lp_idx]
                    rid = landing_rally_id.get((si, lp["frame"]))
                    if cur_rally_id is not None and rid != cur_rally_id:
                        cur_lp_so_far = []
                    cur_rally_id = rid
                    cur_lp_so_far.append(lp)
                    lp_idx += 1
                mini.draw(frame, cur_lp_so_far)

                if use_pipe:
                    try:
                        pipe_proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError):
                        log("⚠️ ffmpeg pipe 断开, 回退到 cv2.VideoWriter")
                        try:
                            pipe_proc.wait()
                        except Exception:
                            pass
                        use_pipe = False
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建回退视频写入器")
                        writer.write(frame)
                else:
                    writer.write(frame)
                written += 1
                now = time.time()
                if written % 100 == 0 or written == total_render or (now - last_progress_time) >= 0.5:
                    elapsed = now - t_start
                    fps_render = written / elapsed if elapsed > 0 else 0
                    pct = written / total_render * 100 if total_render > 0 else 100
                    eta = (total_render - written) / fps_render if fps_render > 0 else 0
                    sys.stdout.write(
                        f"\r  渲染进度: {written}/{total_render} ({pct:5.1f}%)"
                        f" | {fps_render:6.1f} fps | ETA {eta:5.0f}s   ")
                    sys.stdout.flush()
                    last_progress_time = now
        sys.stdout.write("\n")
        sys.stdout.flush()
        elapsed_total = time.time() - t_start
        log(f"✅ 无声视频渲染完成: {written} 帧, 耗时 {elapsed_total:.1f}s "
            f"({written / max(elapsed_total, 1e-6):.1f} fps)")
    finally:
        if cap is not None:
            cap.release()
        if use_pipe and pipe_proc is not None:
            try:
                pipe_proc.stdin.close()
            except Exception:
                pass
            try:
                pipe_proc.wait()
            except Exception:
                pass
        if writer is not None:
            writer.release()

    try:
        finalize_output(args, silent_path, segments, fps, written, tmp_dir,
                        already_encoded=use_pipe, enc=enc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def finalize_output(args, silent_path, segments, fps, written, tmp_dir,
                    already_encoded=False, enc="libx264"):
    has_audio = (not args.no_audio) and ffprobe_has_audio(args.video)
    if not has_audio or not shutil.which("ffmpeg"):
        if already_encoded:
            log(f"🎞️ 无音频, 直接复制已编码视频 ({enc})")
            shutil.copy2(silent_path, args.output)
        else:
            if enc == "libx264":
                enc = detect_amf_encoder(args.encoder)
            log(f"🎞️ 无音频, 编码 ({enc})")
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                   "-i", silent_path] + _enc_args(enc) + [args.output]
            subprocess.run(cmd, check=True, creationflags=_SUBPROC_FLAGS)
        return
    parts, filt = [], []
    for i, (s, e) in enumerate(segments):
        t0 = s / fps
        t1 = (e + 1) / fps
        filt.append(f"[1:a]atrim=start={t0:.4f}:end={t1:.4f},asetpts=PTS-STARTPTS[a{i}]")
        parts.append(f"[a{i}]")
    concat = "".join(parts) + f"concat=n={len(segments)}:v=0:a=1[aout]"
    filter_complex = ";".join(filt + [concat])
    if already_encoded:
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux (视频 copy, 音频 aac)")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
               "-i", silent_path, "-i", args.video,
               "-filter_complex", filter_complex,
               "-map", "0:v:0", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", args.output]
    else:
        if enc == "libx264":
            enc = detect_amf_encoder(args.encoder)
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux ({enc})")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
               "-i", silent_path, "-i", args.video,
               "-filter_complex", filter_complex,
               "-map", "0:v:0", "-map", "[aout]"] + _enc_args(enc) + [
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", args.output]
    try:
        subprocess.run(cmd, check=True, creationflags=_SUBPROC_FLAGS)
    except subprocess.CalledProcessError:
        log("⚠️ 音频 mux 失败, 退回输出无声视频")
        if already_encoded:
            shutil.copy2(silent_path, args.output)
        else:
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                   "-i", silent_path] + _enc_args(enc) + [args.output]
            subprocess.run(cmd, check=True, creationflags=_SUBPROC_FLAGS)


# =====================================================================
#  main
# =====================================================================
def main():
    args = parse_args()
    if not os.path.exists(args.video):
        log(f"❌ 视频不存在: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.raw_detections):
        log(f"❌ 检测日志不存在: {args.raw_detections}")
        sys.exit(1)
    log("=" * 60)
    log("🏓 PingPong Highlight Generator (v4.8 track_id 贯穿, TRACK MODE 跟踪版)")
    log("=" * 60)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log(f"❌ 无法打开视频: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log(f"📹 {in_w}x{in_h} @ {fps:.2f}fps, {total_frames} 帧")

    log("📖 读取检测日志 ...")
    frames = load_detections(args.raw_detections)
    log(f"   {len(frames)} 条帧记录")

    log("📐 提取球台/球网几何 + 校验自愈 + 长边走廊 ...")
    table_geo = extract_table_geometry(
        frames, (in_h, in_w), debug=args.debug,
        corridor_margin=args.corridor_margin, corridor_extend=args.corridor_extend)
    geo_rebuilt = bool(table_geo.get("rebuilt")) if table_geo else False

    if not args.no_player_fx:
        player_boxes, _pc = gather_player_boxes(frames, debug=args.debug)
    else:
        player_boxes = {}

    log("⚪ 提取主球轨迹 (轨迹级过滤 + 走廊软过滤 + 多球过滤) ...")
    ball_cands = gather_ball_candidates(frames, fps, conf_min=args.ball_conf)  # 5 元组
    # 降级检测: 统计所有候选中 track_id≥0 的比例
    n_total_cands = sum(len(cands) for _fidx, cands in ball_cands)
    n_tracked = sum(1 for _fidx, cands in ball_cands for det in cands if det[4] >= 0)
    if n_tracked > 0:
        if args.debug:
            log(f"  ★ 轨迹级过滤: {n_tracked}/{n_total_cands} 个候选含有效 track_id")
        ball_cands = filter_static_tracks(
            ball_cands, args.track_static_frames, args.track_static_motion, debug=args.debug)
        ball_cands = filter_short_lived_tracks(
            ball_cands, args.min_track_frames, debug=args.debug)
    else:
        log("⚠️ 检测日志无 track_id 字段, 轨迹级过滤已跳过 — 请使用 inference_video_track.py 生成的 JSONL")
    # 走廊过滤直接吃 5 元组, 以便按 track_id 做轨迹级豁免 (放高球保留)
    corridor = table_geo.get("corridor") if table_geo else None
    if corridor is not None:
        ball_cands = filter_ball_by_corridor(
            ball_cands, corridor, max_drop_ratio=args.corridor_max_drop,
            geo_rebuilt=geo_rebuilt,
            exempt_ratio=args.track_corridor_exempt_ratio, debug=args.debug)
    # ★ 不再剥离 track_id: 5 元组直通 cluster/build/interpolate/渲染全链路,
    #   由各环节按 tid 约束防止跨轨迹混画 (旧 JSONL 全 -1 时行为与原版一致)
    ball_cands = cluster_ball_by_court(ball_cands, in_w, in_h, debug=args.debug)
    main_track = build_main_ball_track(
        ball_cands, fps, in_w, in_h, table_geo=table_geo, debug=args.debug,
        max_jump_ratio=args.max_jump_ratio, match_algo=args.match_algo)
    log(f"   主轨迹原始点数: {len(main_track)}")

    log("🧩 物理补帧 ...")
    filled = interpolate_track(main_track, fps, in_w, in_h, debug=args.debug)

    log("⚡ 计算逐帧球速表 (速度自适应特效) ...")
    ball_speed_table, (p50, p95) = compute_ball_speed_table(filled, table_geo, fps, in_w, in_h)
    if args.debug:
        log(f"   球速基准: p50={p50:.0f}px/s  p95={p95:.0f}px/s  (共 {len(ball_speed_table)} 帧速度)")

    log("🎾 检测回合与落点/击球事件 ...")
    rallies = detect_rallies(filled, fps, total_frames, args.raw_detections,
                             table_geo=table_geo,
                             out_of_table_timeout=args.out_of_table_timeout, debug=args.debug)
    events = detect_events(filled, table_geo, fps)
    log(f"   回合: {len(rallies)}, 事件: {len(events)}")

    segments, rally_starts_by_seg = build_segments(rallies, fps, total_frames, args)
    if not segments:
        log("❌ 没有检测到有效回合, 无法生成高光。可尝试放宽 --min-rally-frames")
        sys.exit(2)
    total_out = sum(e - s + 1 for s, e in segments)
    log(f"✂️  有效片段: {len(segments)} 个, 共 {total_out} 帧 (~{total_out / fps:.1f}s)")
    if args.debug:
        for si, (s, e) in enumerate(segments):
            log(f"    片段{si}: [{s}-{e}]  内含回合起始帧={rally_starts_by_seg.get(si)}")

    render(args, segments, rally_starts_by_seg, filled, events, table_geo,
           in_w, in_h, fps, total_frames, ball_speed_table, player_boxes)
    log(f"🎉 完成! 输出: {args.output}")


if __name__ == "__main__":
    main()