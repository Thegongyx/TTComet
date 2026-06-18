#!/usr/bin/env python3
"""
pingpong_highlight.py  (修复版)
=====================================================================
乒乓球比赛 → 自动特效精华集锦

本次修复 (基于你的需求):
  1. 【有效区改为"长边走廊"】不再用封闭四边形 quad 裁球。
     利用 extract_table_geometry 已确定好的四点 (far_left/far_right/
     near_left/near_right), 构造一条"两条长边为左右边界、沿长轴方向
     无限延伸"的走廊 (长边延长线覆盖区), 横向留容差。
     → 超出短边(near/far)的球(发球/削球/高抛)轨迹也能正常渲染。
  2. 【台外球剔除不影响特效】走廊外的球被剔除, 但 mini 球台落点、
     彗尾残影、击球/落台特效只对"走廊内的主轨迹"绘制, 互不干扰。
  3. 【不误杀比赛球】走廊横向容差自适应 (用近端短边宽度比例),
     聚类/轨迹构建只在走廊内做, 避免把短边外的真实比赛球当野值。
  4. 【轨迹平滑】保留物理补帧 + smooth_trail, 走廊过滤在补帧前完成,
     不在轨迹中间打洞。
  5. 3D/走廊几何只用于判定, 不渲染到视频。

其余说明同原版 (略)。
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
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

_IS_WINDOWS = (platform.system() == "Windows")
_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0

# smooth_trail 预分配卷积核
_SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float64)


# =====================================================================
#  标准球台常量 (ITTF)
# =====================================================================
TABLE_LEN_M   = 2.74     # 长
TABLE_WID_M   = 1.525    # 宽
TABLE_RATIO   = TABLE_LEN_M / TABLE_WID_M   # ≈ 1.797 (长/宽)


# =====================================================================
#  CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="乒乓球比赛自动特效高光剪辑 (pingpong_highlight.py)")
    p.add_argument("--video", "-v", required=True, help="输入视频路径")
    p.add_argument("--raw-detections", "-r", required=True,
                   help="raw_detections.jsonl 路径")
    p.add_argument("--output", "-o", default="highlight_output.mp4",
                   help="输出 mp4 路径 (默认 highlight_output.mp4)")
    p.add_argument("--output-height", type=int, default=None,
                   help="输出高度像素 (默认与输入视频原分辨率一致)")

    # ---- 进阶可调 ----
    p.add_argument("--min-rally-frames", type=int, default=None,
                   help="有效回合最小帧数 (默认按 fps 自动 = 0.6s)")
    p.add_argument("--min-static-ball-count", type=int, default=10,
                   help="静止球过滤: 600帧窗口内最少静止检测数, 达到才剔除 (默认 10)")
    p.add_argument("--pre-buffer", type=float, default=0.5,
                   help="每个回合前缓冲秒数")
    p.add_argument("--post-buffer", type=float, default=1.5,
                   help="每个回合后缓冲秒数")
    p.add_argument("--merge-gap", type=float, default=2.0,
                   help="相邻回合间隔小于该秒数则合并")
    # ★ 新增: 走廊横向容差 (相对近端短边宽度的比例)
    p.add_argument("--corridor-margin", type=float, default=0.5,
                   help="长边走廊横向容差 (近端短边宽度比例, 默认 0.5)")
    # ★ 新增: 走廊纵向延伸 (相对球台长度的倍数, 覆盖发球/高抛出界)
    p.add_argument("--corridor-extend", type=float, default=1.2,
                   help="走廊沿长轴向短边外延伸倍数 (球台长度的倍数, 默认 1.2)")
    p.add_argument("--no-audio", action="store_true",
                   help="不保留原声")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_amf", "hevc_amf"],
                   help="输出编码器, auto=优先 AMD AMF 硬编")
    p.add_argument("--debug", action="store_true",
                   help="输出调试信息")
    return p.parse_args()


# =====================================================================
#  工具函数
# =====================================================================
def log(msg):
    print(msg, flush=True)


def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


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
    """返回可用编码器名; 失败返回 libx264"""
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


# =====================================================================
#  1) 读取日志
# =====================================================================
def load_detections(path):
    """读取 jsonl, 返回 frames: list[dict], 按 frame 升序。"""
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
    ordered = [frames[k] for k in sorted(frames.keys())]
    return ordered


# =====================================================================
#  2) 球台 / 球网几何提取 (polygon 点云拟合)
# =====================================================================
def aggregate_polygon_points(frames, target_class, conf_min=0.5):
    """收集所有高置信度帧的 polygon 顶点 (用于稳定拟合)。"""
    pts = []
    for fr in frames:
        for det in fr.get("detections", []):
            if det["class"] != target_class:
                continue
            if det.get("confidence", 0) < conf_min:
                continue
            poly = det.get("polygon")
            if poly:
                pts.extend(poly)
            else:
                b = det["bbox"]
                pts.extend([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
    return np.array(pts, dtype=np.float32) if pts else None


def robust_quad_from_points(pts):
    if pts is None or len(pts) < 4:
        return None
    pts = pts.astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps_ratio in (0.02, 0.03, 0.05, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        quad = np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ], dtype=np.float32)
    return order_quad(quad)


def order_quad(quad):
    """把四个角点排序为 [TL, TR, BR, BL] (图像坐标, y向下)。"""
    quad = np.array(quad, dtype=np.float32)
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    order = np.argsort(ang)
    q = quad[order]
    s = q[:, 0] + q[:, 1]
    start = int(np.argmin(s))
    q = np.roll(q, -start, axis=0)
    v1 = q[1] - q[0]
    v2 = q[2] - q[0]
    if (v1[0] * v2[1] - v1[1] * v2[0]) < 0:
        q = q[[0, 3, 2, 1]]
    return q.astype(np.float32)


# ---- 逐帧拟合辅助函数 ----
def _get_polygon_pts_single(fr, target_class, conf_min=0.5):
    pts = []
    for det in fr.get("detections", []):
        if det["class"] != target_class:
            continue
        if det.get("confidence", 0) < conf_min:
            continue
        poly = det.get("polygon")
        if poly:
            pts.extend(poly)
        else:
            b = det["bbox"]
            pts.extend([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
    return np.array(pts, dtype=np.float32) if pts else None


def _fit_quad_single(pts):
    if pts is None or len(pts) < 4:
        return None
    pts = pts.astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps_ratio in (0.003, 0.005, 0.008, 0.012, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        quad = np.array([
            pts[np.argmin(s)],
            pts[np.argmax(d)],
            pts[np.argmax(s)],
            pts[np.argmin(d)],
        ], dtype=np.float32)
    return quad.astype(np.float32)


def _median_quad(quad_list):
    if len(quad_list) < 3:
        return None
    ordered = []
    for quad in quad_list:
        q = order_quad(quad)
        ordered.append(q)
    stacked = np.stack(ordered, axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def _fit_net_line_single(pts):
    if pts is None or len(pts) < 2:
        return None
    mean_all = pts.mean(axis=0)
    centered_all = pts - mean_all
    _, eigvecs_all = np.linalg.eigh(np.cov(centered_all.T))
    rough_dir = eigvecs_all[:, -1]
    proj_all = centered_all @ rough_dir
    if len(pts) >= 30:
        lo = np.percentile(proj_all, 5.0)
        hi = np.percentile(proj_all, 95.0)
        keep = (proj_all >= lo) & (proj_all <= hi)
        pts_trim = pts[keep]
    else:
        pts_trim = pts
    if len(pts_trim) < 2:
        return None
    mean = pts_trim.mean(axis=0)
    centered = pts_trim - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]
    proj = centered @ principal
    t_min, t_max = proj.min(), proj.max()
    p1 = mean + principal * t_max
    p2 = mean + principal * t_min
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    return (tuple(p1.astype(float).tolist()),
            tuple(p2.astype(float).tolist()))


def _median_net_line(line_list):
    if len(line_list) < 3:
        return None
    p1s = np.array([l[0] for l in line_list])
    p2s = np.array([l[1] for l in line_list])
    return (tuple(np.median(p1s, axis=0).tolist()),
            tuple(np.median(p2s, axis=0).tolist()))


def _point_in_quad(pt, quad, margin=0.10):
    quad_arr = np.array(quad, dtype=np.float32)
    center = quad_arr.mean(axis=0)
    expanded = center + (quad_arr - center) * (1.0 + margin)
    result = cv2.pointPolygonTest(expanded.astype(np.float32),
                                  (float(pt[0]), float(pt[1])), False)
    return result >= 0


# =====================================================================
#  ★★★ 新增: 长边走廊 (有效渲染区) 构造 ★★★
# =====================================================================
def build_corridor(far_left, far_right, near_left, near_right,
                   corridor_margin=0.45, corridor_extend=1.2, debug=False):
    """
    用已确定好的四点坐标构造"长边走廊"多边形 (有效渲染区)。

    物理依据: 球可飞出短边(发球/削球/高抛), 但极少横向飞出长边太多。
    所以有效区 = 以两条长边(far_left↔near_left, far_right↔near_right)
    为左右边界, 沿长轴方向向 far/near 两端各延伸 corridor_extend 倍,
    并横向外扩 corridor_margin (相对近端短边宽度) 的容差带。

    返回 (4,2) float32 多边形 [扩展后远左, 远右, 近右, 近左], 顺时针闭合。
    完全复用上游确定好的四点, 不重做几何识别逻辑。
    """
    fl = np.array(far_left, dtype=np.float64)
    fr = np.array(far_right, dtype=np.float64)
    nl = np.array(near_left, dtype=np.float64)
    nr = np.array(near_right, dtype=np.float64)

    # 长轴方向: 从近端指向远端 (左/右两条长边各自的方向, 取平均更稳)
    left_axis = fl - nl     # 近左 -> 远左
    right_axis = fr - nr    # 近右 -> 远右
    # 单位向量
    def _unit(v):
        n = np.linalg.norm(v) + 1e-8
        return v / n
    ul = _unit(left_axis)
    ur = _unit(right_axis)

    # 球台一个台身的长度 (像素, 取左右长边均值)
    body_len = (np.linalg.norm(left_axis) + np.linalg.norm(right_axis)) * 0.5

    # 沿长轴向两端延伸 (far 端沿 +轴, near 端沿 -轴)
    ext = corridor_extend * body_len
    far_left_ext  = fl + ul * ext
    far_right_ext = fr + ur * ext
    near_left_ext  = nl - ul * ext
    near_right_ext = nr - ur * ext

    # 横向容差: 以近端短边宽度为基准向两侧外扩
    near_width_vec = nr - nl
    near_width = np.linalg.norm(near_width_vec) + 1e-8
    side_dir = near_width_vec / near_width    # 指向右
    margin_px = corridor_margin * near_width

    # 左边界整体向左推, 右边界整体向右推
    far_left_ext  = far_left_ext  - side_dir * margin_px
    near_left_ext = near_left_ext - side_dir * margin_px
    far_right_ext  = far_right_ext  + side_dir * margin_px
    near_right_ext = near_right_ext + side_dir * margin_px

    corridor = np.array([far_left_ext, far_right_ext,
                         near_right_ext, near_left_ext], dtype=np.float32)
    if debug:
        log(f"  ◆ 长边走廊: 横向容差={margin_px:.0f}px (近宽{near_width:.0f}px×{corridor_margin}), "
            f"纵向延伸={ext:.0f}px (台身{body_len:.0f}px×{corridor_extend})")
        log(f"    走廊四角: {corridor.tolist()}")
    return corridor


def _point_in_corridor(pt, corridor):
    """判断像素点是否在走廊多边形内 (走廊已含容差, 不再额外扩张)。"""
    if corridor is None:
        return True
    res = cv2.pointPolygonTest(corridor.reshape(-1, 1, 2),
                               (float(pt[0]), float(pt[1])), False)
    return res >= 0


def extract_table_geometry(frames, img_hw, debug=False, persp_blend=0.5,
                           corridor_margin=0.45, corridor_extend=1.2):
    """
    返回 dict:
       quad        : (4,2) 球台四角 [TL,TR,BR,BL]
       net_line    : ((x1,y1),(x2,y2))
       homography  : 图像->俯视标准台
       corridor    : (4,2) ★ 长边走廊有效渲染区
       far_left/far_right/near_left/near_right : 已确定好的四点
     若无法提取返回 None。
    """
    h, w = img_hw
    sample_frames = frames[::10]

    table_quads = []
    table_fail = 0
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "table", conf_min=0.5)
        if pts is not None and len(pts) >= 4:
            quad = _fit_quad_single(pts)
            if quad is not None and len(quad) == 4:
                table_quads.append(quad)
            else:
                table_fail += 1
        else:
            table_fail += 1

    if len(table_quads) < 3:
        log(f"⚠️ 有效球台拟合帧数不足 ({len(table_quads)}<3), 无法做迷你球台 / 落点分析")
        return None

    quad = _median_quad(table_quads)
    if quad is None:
        log("⚠️ 球台四角中位数拟合失败")
        return None

    # ---- 球网 ----
    net_lines = []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "net", conf_min=0.5)
        if pts is not None and len(pts) >= 2:
            line = _fit_net_line_single(pts)
            if line is not None:
                net_lines.append(line)
    net_line = _median_net_line(net_lines) if len(net_lines) >= 3 else None

    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]

    if net_line is not None:
        nd = np.array([net_line[1][0] - net_line[0][0],
                       net_line[1][1] - net_line[0][1]], dtype=np.float32)
        nd /= np.linalg.norm(nd) + 1e-8
    else:
        nd = None

    edge_info = []
    for i, (a, b) in enumerate(edges):
        ev = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float32)
        elen = np.linalg.norm(ev) + 1e-8
        ev_n = ev / elen
        par = abs(float(np.dot(ev_n, nd))) if nd is not None else 0.0
        angle_deg = math.degrees(math.acos(min(1.0, par)))
        edge_info.append((i, elen, angle_deg, a, b))

    found_end_edges = False
    if net_line is not None:
        def _extend_line(p1, p2, length):
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                return p1, p2
            ux, uy = dx / norm, dy / norm
            return (p1[0] - ux*length, p1[1] - uy*length), \
                   (p2[0] + ux*length, p2[1] + uy*length)

        def _segments_intersect(p1, p2, q1, q2):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(q1, q2, p1)
            d2 = cross(q1, q2, p2)
            d3 = cross(p1, p2, q1)
            d4 = cross(p1, p2, q2)
            return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

        net_ext = _extend_line(net_line[0], net_line[1], length=max(w, h) * 4)
        intersect_mask = [_segments_intersect(net_ext[0], net_ext[1], a, b)
                          for a, b in edges]
        if sum(intersect_mask) == 2:
            found_end_edges = True
            inter_idx = [i for i, v in enumerate(intersect_mask) if v]
            non_inter_idx = [i for i, v in enumerate(intersect_mask) if not v]
            end_edge_list = [edge_info[i] for i in non_inter_idx]
            side_edges = [edge_info[i] for i in inter_idx]
            near_corner_idx = int(np.argmin([math.hypot(c[0], c[1] - h) for c in quad]))
            near_corner = quad[near_corner_idx]
            e0_a, e0_b = np.array(end_edge_list[0][3]), np.array(end_edge_list[0][4])
            e1_a, e1_b = np.array(end_edge_list[1][3]), np.array(end_edge_list[1][4])
            if (np.linalg.norm(near_corner - e0_a) < 2.0
                    or np.linalg.norm(near_corner - e0_b) < 2.0):
                near_a, near_b = end_edge_list[0][3], end_edge_list[0][4]
                far_a, far_b = end_edge_list[1][3], end_edge_list[1][4]
            else:
                near_a, near_b = end_edge_list[1][3], end_edge_list[1][4]
                far_a, far_b = end_edge_list[0][3], end_edge_list[0][4]
            end_edges = end_edge_list

    if not found_end_edges:
        edge_info.sort(key=lambda x: x[2])
        end_edges = edge_info[:2]
        side_edges = edge_info[2:]
        near_corner_idx = int(np.argmin([math.hypot(c[0], c[1] - h) for c in quad]))
        near_corner = quad[near_corner_idx]
        e0_a, e0_b = np.array(end_edges[0][3]), np.array(end_edges[0][4])
        e1_a, e1_b = np.array(end_edges[1][3]), np.array(end_edges[1][4])
        if (np.linalg.norm(near_corner - e0_a) < 2.0
                or np.linalg.norm(near_corner - e0_b) < 2.0):
            near_a, near_b = end_edges[0][3], end_edges[0][4]
            far_a, far_b = end_edges[1][3], end_edges[1][4]
        else:
            near_a, near_b = end_edges[1][3], end_edges[1][4]
            far_a, far_b = end_edges[0][3], end_edges[0][4]

    if far_a[0] > far_b[0]:
        far_left, far_right = far_b, far_a
    else:
        far_left, far_right = far_a, far_b
    if near_a[0] > near_b[0]:
        near_left, near_right = near_b, near_a
    else:
        near_left, near_right = near_a, near_b

    # ---- 透视修正 ----
    if persp_blend > 0:
        near_width = float(near_right[0] - near_left[0])
        if near_width > 10:
            near_slope = (near_right[1] - near_left[1]) / near_width
            old_far_left = np.array(far_left, dtype=np.float32).copy()
            old_far_right = np.array(far_right, dtype=np.float32).copy()
            if far_left[1] < far_right[1]:
                target_x = far_right[0] - near_width
                target_y = far_right[1] - near_slope * near_width
                far_left = np.array([
                    far_left[0] + persp_blend * (target_x - far_left[0]),
                    far_left[1] + persp_blend * (target_y - far_left[1]),
                ], dtype=np.float32)
            else:
                target_x = far_left[0] + near_width
                target_y = far_left[1] + near_slope * near_width
                far_right = np.array([
                    far_right[0] + persp_blend * (target_x - far_right[0]),
                    far_right[1] + persp_blend * (target_y - far_right[1]),
                ], dtype=np.float32)
            if (not np.allclose(far_left, old_far_left, atol=0.5)
                    or not np.allclose(far_right, old_far_right, atol=0.5)):
                quad_updated = quad.copy()
                for qi in range(4):
                    if np.linalg.norm(quad[qi] - old_far_left) < 3.0:
                        quad_updated[qi] = far_left
                    elif np.linalg.norm(quad[qi] - old_far_right) < 3.0:
                        quad_updated[qi] = far_right
                quad = quad_updated
                if debug:
                    d_fl = np.linalg.norm(far_left - old_far_left)
                    d_fr = np.linalg.norm(far_right - old_far_right)
                    log(f"  ◆ 透视修正(平行 blend={persp_blend}): "
                        f"far_left位移={d_fl:.1f}px  far_right位移={d_fr:.1f}px")

    # homography
    src = np.array([far_left, far_right, near_right, near_left], dtype=np.float32)
    dst = np.array([[0.0,                   0.0],
                    [TABLE_WID_M * 1000,    0.0],
                    [TABLE_WID_M * 1000,    TABLE_LEN_M * 1000],
                    [0.0,                   TABLE_LEN_M * 1000]], dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        H = None

    net_side_sign = None
    if net_line is not None and H is not None:
        (nx1, ny1), (nx2, ny2) = net_line
        near_mid_x = (near_left[0] + near_right[0]) * 0.5
        near_mid_y = (near_left[1] + near_right[1]) * 0.5
        cross_near = ((nx2 - nx1) * (near_mid_y - ny1)
                      - (ny2 - ny1) * (near_mid_x - nx1))
        net_side_sign = 1 if cross_near > 0 else -1

    # ---- ★ 用已确定好的四点构造长边走廊 ----
    corridor = build_corridor(far_left, far_right, near_left, near_right,
                              corridor_margin=corridor_margin,
                              corridor_extend=corridor_extend, debug=debug)

    if debug:
        log(f"  球台四角: {quad.tolist()}")
        log(f"  球网线: {net_line}")
        for ei in edge_info:
            idx, elen, ang, a, b = ei
            mid_y = (a[1] + b[1]) * 0.5
            tag = "端" if ei in end_edges else "侧"
            log(f"  边{idx}({tag}): 长={elen:.0f}px 与网夹角={ang:.1f}度 Y中点={mid_y:.0f}"
                f"  ({a[0]:.0f},{a[1]:.0f})->({b[0]:.0f},{b[1]:.0f})")
        method_tag = "网线延长不相交" if found_end_edges else "与网夹角最小"
        log(f"  端边({method_tag}): {end_edges[0][0]},{end_edges[1][0]}"
            f" {end_edges[0][2]:.1f}度,{end_edges[1][2]:.1f}度")
        log(f"  远端短边: {np.array(far_left).tolist()} -> {np.array(far_right).tolist()}")
        log(f"  近端短边: {np.array(near_left).tolist()} -> {np.array(near_right).tolist()}")
        if H is not None:
            far_mid = ((far_left[0] + far_right[0]) * 0.5, (far_left[1] + far_right[1]) * 0.5)
            near_mid = ((near_left[0] + near_right[0]) * 0.5, (near_left[1] + near_right[1]) * 0.5)
            fm = img_to_table(H, far_mid)
            nm = img_to_table(H, near_mid)
            if fm and nm:
                log(f"  H自检: 远端中点 -> 台坐标 {fm[0]:.0f},{fm[1]:.0f} mm (期望 Y≈0)")
                log(f"  H自检: 近端中点 -> 台坐标 {nm[0]:.0f},{nm[1]:.0f} mm (期望 Y≈2740)")

    return {"quad": quad, "net_line": net_line, "homography": H,
            "net_side_sign": net_side_sign,
            "corridor": corridor,
            "far_left": np.array(far_left, dtype=np.float32),
            "far_right": np.array(far_right, dtype=np.float32),
            "near_left": np.array(near_left, dtype=np.float32),
            "near_right": np.array(near_right, dtype=np.float32),
            "far_w": float(far_right[0] - far_left[0]),
            "near_w": float(near_right[0] - near_left[0]),
            "far_mean_y": float((far_left[1] + far_right[1]) / 2.0),
            "near_mean_y": float((near_left[1] + near_right[1]) / 2.0)}


def img_to_table(H_or_geo, pt):
    """图像点 -> 标准俯视台坐标 (mm)。"""
    if H_or_geo is None:
        return None
    if isinstance(H_or_geo, dict):
        geo = H_or_geo
        H = geo.get("homography")
        if H is None:
            return None
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(src, H)
        x_mm = float(out[0, 0, 0])
        y_mm = float(out[0, 0, 1])
        net_line = geo.get("net_line")
        net_side_sign = geo.get("net_side_sign")
        if net_line is not None and net_side_sign is not None:
            (nx1, ny1), (nx2, ny2) = net_line
            px, py = pt[0], pt[1]
            cross = (nx2 - nx1) * (py - ny1) - (ny2 - ny1) * (px - nx1)
            is_near_side = (cross * net_side_sign > 0)
            half_y = TABLE_LEN_M * 1000 / 2.0
            if is_near_side and y_mm < half_y:
                y_mm = half_y + (half_y - y_mm)
            elif (not is_near_side) and y_mm > half_y:
                y_mm = half_y - (y_mm - half_y)
        return (x_mm, y_mm)
    else:
        H = H_or_geo
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(src, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])


# =====================================================================
#  3) 球检测整理 + 错误球过滤
# =====================================================================
def gather_ball_candidates(frames, fps):
    out = []
    for fr in frames:
        fidx = fr["frame"]
        cands = []
        for det in fr.get("detections", []):
            if det["class"] != "pingpangball":
                continue
            b = det["bbox"]
            cx, cy = bbox_center(b)
            cands.append((cx, cy, det.get("confidence", 0.0), b))
        out.append((fidx, cands))
    return out


def filter_static_ball_candidates(ball_cands, fps, frame_w, frame_h,
                                   min_static_count=10, debug=False):
    if not ball_cands:
        return ball_cands
    window_frames = 600
    img_diag = math.hypot(frame_w, frame_h)
    static_threshold = max(img_diag * 0.0008, 2.0)
    ball_cands_sorted = sorted(ball_cands, key=lambda x: x[0])
    flat_dets = []
    for fidx, cands in ball_cands_sorted:
        for di, (cx, cy, _conf, _bbox) in enumerate(cands):
            flat_dets.append((fidx, di, cx, cy))
    if not flat_dets:
        return ball_cands
    removed_set = set()
    idx = 0
    n_dets = len(flat_dets)
    while idx < n_dets:
        fidx_anchor = flat_dets[idx][0]
        win_end = fidx_anchor + window_frames
        window_dets = []
        while idx < n_dets and flat_dets[idx][0] <= win_end:
            window_dets.append(flat_dets[idx])
            idx += 1
        nw = len(window_dets)
        if nw < min_static_count:
            continue
        processed = [False] * nw
        for a in range(nw):
            if processed[a]:
                continue
            fidx_a, di_a, cx_a, cy_a = window_dets[a]
            if (fidx_a, di_a) in removed_set:
                continue
            cluster = [(a, fidx_a, di_a)]
            for b in range(a + 1, nw):
                fidx_b, di_b, cx_b, cy_b = window_dets[b]
                if (fidx_b, di_b) in removed_set:
                    continue
                if math.hypot(cx_a - cx_b, cy_a - cy_b) <= static_threshold:
                    cluster.append((b, fidx_b, di_b))
            if len(cluster) >= min_static_count:
                for _, f, d in cluster:
                    removed_set.add((f, d))
                for b_idx, _, _ in cluster:
                    processed[b_idx] = True
    if not removed_set:
        return ball_cands
    filtered = []
    n_removed = len(removed_set)
    for fidx, cands in ball_cands:
        remaining = []
        for di, det in enumerate(cands):
            if (fidx, di) not in removed_set:
                remaining.append(det)
        filtered.append((fidx, remaining))
    if debug:
        log(f"  ★ 静止球过滤: 剔除 {n_removed} 个静止球坐标 "
            f"(窗口={window_frames}帧, 静止阈值={static_threshold:.0f}px, "
            f"最少静止数={min_static_count})")
    return filtered


def filter_ball_by_corridor(ball_cands, corridor, debug=False):
    """
    ★ 新增: 用长边走廊剔除台外球 (替代/前置于原空间聚类)。
    走廊外的球直接丢弃 → 不进入主轨迹, 不影响特效。
    保留走廊内的所有球 (不误杀短边外的发球/高抛)。
    """
    if corridor is None or not ball_cands:
        return ball_cands
    filtered = []
    n_before = 0
    n_after = 0
    for fidx, cands in ball_cands:
        keep = []
        for det in cands:
            n_before += 1
            cx, cy = det[0], det[1]
            if _point_in_corridor((cx, cy), corridor):
                keep.append(det)
                n_after += 1
        filtered.append((fidx, keep))
    if debug:
        log(f"  ★ 长边走廊过滤: {n_before} → {n_after} 个球检测 "
            f"(剔除走廊外 {n_before - n_after} 个)")
    return filtered


def cluster_ball_by_court(ball_cands, frame_w, frame_h, debug=False):
    """空间聚类: 只保留最大簇的球 (走廊过滤后的二次保险, 分离旁桌残余)。"""
    import math
    if not ball_cands:
        return ball_cands
    pts = []
    pt_refs = []
    for bi, (fidx, cands) in enumerate(ball_cands):
        for ci, (cx, cy, conf, bbox) in enumerate(cands):
            pts.append([cx, cy])
            pt_refs.append((bi, ci))
    if len(pts) <= 1:
        return ball_cands
    pts_arr = np.array(pts, dtype=np.float64)
    diag = math.hypot(frame_w, frame_h)
    eps = diag * 0.005
    method = "DBSCAN" if _HAS_SKLEARN else "网格连通"
    if _HAS_SKLEARN:
        db = DBSCAN(eps=eps, min_samples=5, n_jobs=-1).fit(pts_arr)
        labels = db.labels_
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
                        for nr, nc in [(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)]:
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr, nc] > 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    total = int(sum(grid[r2, c2] for r2, c2 in cells))
                    clusters.append((total, set(cells)))
        if len(clusters) <= 1:
            return ball_cands
        clusters.sort(key=lambda x: x[0], reverse=True)
        main_cells = clusters[0][1]
        keep = set()
        for i, (cx, cy) in enumerate(pts):
            if (int(cy / cell_size), int(cx / cell_size)) in main_cells:
                keep.add(i)
        n_clusters = len(clusters)
    filtered_dict = {}
    n_before = len(pts)
    n_after = 0
    for i in keep:
        bi, ci = pt_refs[i]
        fidx, cands_orig = ball_cands[bi]
        if fidx not in filtered_dict:
            filtered_dict[fidx] = []
        filtered_dict[fidx].append(cands_orig[ci])
        n_after += 1
    filtered = [(fidx, cands) for fidx, cands in sorted(filtered_dict.items())]
    if n_before > n_after:
        log(f"  球台空间聚类 ({method} eps={eps:.0f}px): "
            f"{n_before} → {n_after} 个检测 ({n_clusters} 个簇)")
    elif not filtered:
        return ball_cands
    return filtered


def build_main_ball_track(ball_cands, fps, frame_w, frame_h, table_geo=None, debug=False):
    max_jump = 0.05 * math.hypot(frame_w, frame_h)
    tracks = []

    def predict(track, fidx):
        pts = track["pts"]
        if len(pts) == 1:
            return pts[-1][1], pts[-1][2]
        (f0, x0, y0, *_), (f1, x1, y1, *_) = pts[-2], pts[-1]
        df = max(1, f1 - f0)
        vx, vy = (x1 - x0) / df, (y1 - y0) / df
        dt = fidx - f1
        return x1 + vx * dt, y1 + vy * dt

    for fidx, cands in ball_cands:
        used = [False] * len(cands)
        for tr in tracks:
            gap = fidx - tr["pts"][-1][0]
            if gap <= 0 or gap > fps * 0.5:
                continue
            px, py = predict(tr, fidx)
            best_j, best_d = -1, 1e18
            for j, (cx, cy, conf, b) in enumerate(cands):
                if used[j]:
                    continue
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d, best_j = d, j
            tol = max_jump * (1 + 0.1 * (gap - 1))
            if best_j >= 0 and best_d < tol:
                cx, cy, conf, b = cands[best_j]
                tr["pts"].append((fidx, cx, cy, conf, b))
                used[best_j] = True
        for j, (cx, cy, conf, b) in enumerate(cands):
            if not used[j]:
                tracks.append({"pts": [(fidx, cx, cy, conf, b)]})

    if not tracks:
        return {}

    net_line = table_geo["net_line"] if table_geo else None

    def net_cross_count(pts):
        if net_line is None:
            return 0
        (nx1, ny1), (nx2, ny2) = net_line
        count = 0
        prev_side = None
        for p in pts:
            x, y = p[1], p[2]
            side = ((nx2 - nx1) * (y - ny1) - (ny2 - ny1) * (x - nx1)) > 0
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
        motion = sum(math.hypot(pts[i][1]-pts[i-1][1], pts[i][2]-pts[i-1][2])
                     for i in range(1, n))
        avg_conf = np.mean([p[3] for p in pts])
        base = n * (0.4 + coverage) * (1 + math.log1p(motion)) * (0.5 + avg_conf)
        nc = net_cross_count(pts)
        net_boost = 1.0 + min(2.0, nc * 0.5)
        return base * net_boost

    tracks.sort(key=score, reverse=True)

    n_before_filter = len(tracks)
    if len(tracks) >= 2 and table_geo is not None:
        anchor = tracks[0]
        anchor_pts = np.array([[p[1], p[2]] for p in anchor["pts"]])
        ax_min, ay_min = anchor_pts.min(axis=0)
        ax_max, ay_max = anchor_pts.max(axis=0)
        anchor_area = (ax_max - ax_min) * (ay_max - ay_min)
        if anchor_area <= 0:
            anchor_area = 1.0
        filtered = [anchor]
        for tr in tracks[1:]:
            tr_pts = np.array([[p[1], p[2]] for p in tr["pts"]])
            tx_min, ty_min = tr_pts.min(axis=0)
            tx_max, ty_max = tr_pts.max(axis=0)
            tr_area = (tx_max - tx_min) * (ty_max - ty_min)
            if tr_area <= 0:
                continue
            ix_min = max(ax_min, tx_min)
            iy_min = max(ay_min, ty_min)
            ix_max = min(ax_max, tx_max)
            iy_max = min(ay_max, ty_max)
            if ix_min < ix_max and iy_min < iy_max:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                iou = inter_area / (anchor_area + tr_area - inter_area)
                if iou >= 0.15:
                    filtered.append(tr)
        tracks = filtered
        if debug:
            log(f"  空间共域投票: {n_before_filter} → {len(tracks)} 条轨迹 (IoU≥0.15)")

    score_min = 6.0
    main = {}
    n_used = 0
    for tr in tracks:
        if score(tr) < score_min:
            continue
        n_used += 1
        for (fidx, x, y, conf, b) in tr["pts"]:
            if fidx not in main or conf > main[fidx][2]:
                main[fidx] = (x, y, conf, b)
    if debug:
        log(f"  候选轨迹数={n_before_filter}, 采用={n_used} (score>={score_min:.0f}), "
            f"去重后点数={len(main)}")
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
    filled = {}
    for k in keys:
        filled[k] = (main_track[k][0], main_track[k][1], False)
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
            lo = max(0, i - 1)
            hi = min(len(seg), i + 3)
            wf = fs[lo:hi]
            wx = xs[lo:hi]
            wy = ys[lo:hi]
            try:
                px = np.polyfit(wf, wx, min(2, len(wf) - 1))
            except Exception:
                px = np.polyfit([f0, f1], [main_track[f0][0], main_track[f1][0]], 1)
            try:
                py = np.polyfit(wf, wy, min(2, len(wf) - 1))
            except Exception:
                py = np.polyfit([f0, f1], [main_track[f0][1], main_track[f1][1]], 1)
            for f in range(f0 + 1, f1):
                xi = float(np.polyval(px, f))
                yi = float(np.polyval(py, f))
                xi = min(max(xi, 0), frame_w - 1)
                yi = min(max(yi, 0), frame_h - 1)
                filled[f] = (xi, yi, True)
    if debug:
        n_interp = sum(1 for v in filled.values() if v[2])
        log(f"  补帧: 原始 {len(main_track)} 帧 -> 填充后 {len(filled)} 帧 "
            f"(插值 {n_interp})")
    return filled


# =====================================================================
#  5) 回合检测 + 落点/击球事件
# =====================================================================
class RallyState(Enum):
    IDLE = auto()
    IN_PLAY = auto()


def _finalize_rally(rally_periods, current_rally, rally_state, end_reason,
                    rally_min_duration, rally_num):
    rally_duration = current_rally["end"] - current_rally["start"]
    is_valid = (current_rally.get("crossed_net", False)
                and rally_duration >= float(rally_min_duration))
    if not is_valid and rally_duration < 0.5:
        return False
    rally_periods.append({
        "start_frame": current_rally["start_frame"],
        "end_frame": current_rally["end_frame"],
        "duration": round(rally_duration, 2),
        "ended_reason": end_reason,
    })
    return True


def _get_net_center_x(raw_log_path):
    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            det = json.loads(line)
            for d in det.get("detections", []):
                if d.get("class") == "net" and len(d.get("bbox", [])) == 4:
                    return (d["bbox"][0] + d["bbox"][2]) / 2.0
            for d in det.get("detections", []):
                if d.get("class") == "table" and len(d.get("bbox", [])) == 4:
                    return (d["bbox"][0] + d["bbox"][2]) / 2.0
    return None


def _estimate_slow_speed_threshold(ball_lookup, fps):
    real_pts = []
    for f in sorted(ball_lookup.keys()):
        entry = ball_lookup.get(f)
        if entry is None:
            continue
        (bx, by), is_real = entry
        if not is_real:
            continue
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
    slow_thresh = min(p15 * 0.6, p50 * 0.35)
    return max(slow_thresh, 80.0)


def detect_rallies(filled_track, fps, total_frames, raw_log_path,
                   rally_lost_timeout=1.0, rally_min_duration=0.6,
                   pre_roll_sec=1.0, slow_speed_window_sec=1.0,
                   slow_speed_min_in_play_sec=0.8, debug=False):
    if not filled_track or fps <= 0:
        return []
    ball_lookup = {f: ((x, y), not interp) for f, (x, y, interp) in filled_track.items()}
    sorted_frames = sorted(ball_lookup.keys())
    rallies = []
    net_center_x = _get_net_center_x(raw_log_path)
    if net_center_x is None:
        log("  ⚠️ 未提取到球网中心 x, 回退到连续性切分")
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
    if debug:
        log(f"  慢速阈值: {slow_thresh:.0f} px/s  net_center_x={net_center_x:.0f}")
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
    traj_buffer = deque(maxlen=max(PRE_ROLL_FRAMES * 3, int(fps * 6)))

    def _ball_side(x):
        return 'L' if x < net_center_x else 'R'

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
            side = _ball_side(bx)
            traj_buffer.append((frame_idx, frame_idx / fps, float(bx), float(by), bool(is_real), side))
            missing_ball_frames = 0
        else:
            is_real = False
            side = None
            missing_ball_frames += 1
        if rally_state == RallyState.IDLE:
            if entry is not None and prev_side is not None and side is not None and side != prev_side:
                rally_num += 1
                start_frame = max(0, frame_idx - PRE_ROLL_FRAMES)
                current_rally = {
                    "start_frame": start_frame, "end_frame": frame_idx,
                    "crossed_net": True,
                }
                rally_state = RallyState.IN_PLAY
                slow_run_frames = 0
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
            if end_by_lost or end_by_slow:
                end_reason = "ball_lost" if end_by_lost else "ball_too_slow"
                if end_by_lost and missing_ball_frames > 0:
                    end_f = current_rally["start_frame"]
                    for f in range(frame_idx - missing_ball_frames, current_rally["start_frame"] - 1, -1):
                        if ball_lookup.get(f) is not None:
                            end_f = f
                            break
                else:
                    end_f = frame_idx
                current_rally["end_frame"] = end_f
                current_rally["end"] = end_f / fps
                current_rally["start"] = current_rally["start_frame"] / fps
                _finalize_rally(rallies, current_rally, rally_state,
                                end_reason, rally_min_duration, rally_num)
                rally_state = RallyState.IDLE
                current_rally = None
                slow_run_frames = 0
                prev_side = side
    if rally_state == RallyState.IN_PLAY and current_rally is not None:
        current_rally["start"] = current_rally["start_frame"] / fps
        current_rally["end"] = current_rally["end_frame"] / fps
        _finalize_rally(rallies, current_rally, rally_state,
                        "video_end", rally_min_duration, rally_num)
    if debug:
        log(f"  回合检测 (跨网+球速): {len(rallies)} 个回合")
    return rallies


def detect_events(filled_track, table_geo, fps):
    """
    检测落点(bounce)与击球(hit)事件。
    ★ 修复: 台面判定改用"长边走廊", 而非封闭 quad。
       → 超出短边的击球/落点(发球/高抛/削球)也能正常触发特效。
    """
    events = []
    if not filled_track:
        return events
    keys = sorted(filled_track.keys())
    pts = [(k, filled_track[k][0], filled_track[k][1]) for k in keys]

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
        v_mps = (d_mm / 1000.0) / dt
        return v_mps * 3.6

    # ★ 用走廊替代 quad 做台面判定
    corridor = table_geo.get("corridor") if table_geo else None

    def _in_table(x, y):
        # 走廊已含横向容差 + 纵向延伸, 直接判定即可
        if corridor is None:
            return True
        return _point_in_corridor((x, y), corridor)

    last_bounce_frame = -10 ** 9
    bounce_cooldown_frames = max(1, int(round(fps * 0.1)))

    for i in range(2, len(pts) - 2):
        f, x, y = pts[i]
        f_p, x_p, y_p = pts[i - 1]
        f_n, x_n, y_n = pts[i + 1]
        vy_prev = (y - y_p)
        vy_next = (y_n - y)
        vx_prev = (x - x_p)
        vx_next = (x_n - x)
        ev_type = None
        if vy_prev > 0.1 and vy_next < -0.1:
            if f - last_bounce_frame < bounce_cooldown_frames:
                continue
            if not _in_table(x, y):
                continue
            ev_type = "bounce"
            last_bounce_frame = f
        elif vx_prev * vx_next < 0 and abs(vx_prev) > 2.0 and abs(vx_next) > 2.0:
            # ★ 击球事件也限制在走廊内 (但走廊已纵向延伸, 短边外照样算)
            if not _in_table(x, y):
                continue
            ev_type = "hit"
        if ev_type:
            spd = speed_kmh(f_p, f_n, pts[i - 1], pts[i + 1])
            txy = img_to_table(table_geo, (x, y)) if table_geo is not None else None
            events.append({"frame": f, "x": x, "y": y, "type": ev_type,
                           "speed_kmh": spd, "table_xy": txy})
    return events


# =====================================================================
#  6) 剪辑片段计算
# =====================================================================
def build_segments(rallies, fps, total_frames, args):
    min_frames = args.min_rally_frames or int(fps * 0.6)
    pre = int(args.pre_buffer * fps)
    post = int(args.post_buffer * fps)
    merge_gap = int(args.merge_gap * fps)
    valid = [(r["start_frame"], r["end_frame"]) for r in rallies
             if (r["end_frame"] - r["start_frame"] + 1) >= min_frames]
    if not valid:
        return []
    buffed = [(max(0, s - pre), min(total_frames - 1, e + post)) for (s, e) in valid]
    buffed.sort()
    merged = [list(buffed[0])]
    for s, e in buffed[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged]


# =====================================================================
#  7) 渲染
# =====================================================================
class MiniTable:
    def __init__(self, frame_w, frame_h, table_geo):
        self.geo = table_geo
        self.h = int(frame_h * 0.30)
        self.w = int(self.h / TABLE_RATIO)
        self.margin = int(frame_w * 0.015)
        self.x0 = frame_w - self.w - self.margin
        self.y0 = self.margin
        self.sx = self.w / (TABLE_WID_M * 1000)
        self.sy = self.h / (TABLE_LEN_M * 1000)
        self._bg = np.full((self.h, self.w, 3), (60, 90, 60), dtype=np.uint8)

    def table_to_mini(self, txy):
        if txy is None:
            return None
        X, Y = txy
        x_max = TABLE_WID_M * 1000
        y_max = TABLE_LEN_M * 1000
        margin = 0.15
        if not (-x_max * margin <= X <= x_max * (1 + margin)
                and -y_max * margin <= Y <= y_max * (1 + margin)):
            return None
        X = max(0.0, min(x_max, X))
        Y = max(0.0, min(y_max, Y))
        mx = self.x0 + X * self.sx
        my = self.y0 + Y * self.sy
        return int(mx), int(my)

    def draw(self, frame, current_rally_points):
        roi = frame[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        cv2.addWeighted(self._bg, 0.55, roi, 0.45, 0, roi)
        cv2.rectangle(frame, (self.x0, self.y0),
                      (self.x0 + self.w, self.y0 + self.h), (255, 255, 255), 2)
        midy = self.y0 + self.h // 2
        cv2.line(frame, (self.x0, midy), (self.x0 + self.w, midy),
                 (255, 255, 255), 2)
        dash_len = max(4, self.w // 12)
        for dx in range(0, self.w, dash_len * 2):
            x_start = self.x0 + dx
            x_end = min(self.x0 + dx + dash_len, self.x0 + self.w)
            cv2.line(frame, (x_start, midy), (x_end, midy),
                     (180, 220, 255), 1)
        cv2.putText(frame, "NET", (self.x0 + self.w + 3, midy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 220, 255), 1)
        midx = self.x0 + self.w // 2
        cv2.line(frame, (midx, self.y0), (midx, self.y0 + self.h),
                 (200, 200, 200), 1)
        cv2.putText(frame, "FAR", (self.x0 + 2, self.y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(frame, "NEAR", (self.x0 + 2, self.y0 + self.h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        for lp in current_rally_points:
            mp = self.table_to_mini(lp.get("table_xy"))
            if mp:
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
    def __init__(self):
        self.particles = []

    def emit(self, x, y, n=18, kind="bounce"):
        for _ in range(n):
            ang = np.random.uniform(0, 2 * math.pi)
            spd = np.random.uniform(1.5, 6.0)
            if kind == "bounce":
                vy = -abs(math.sin(ang) * spd) - 1.0
                vx = math.cos(ang) * spd
                color = (210, 230, 255)
            else:
                vx = math.cos(ang) * spd
                vy = math.sin(ang) * spd
                color = (200, 200, 200)
            life = np.random.randint(8, 20)
            self.particles.append({
                "x": x, "y": y, "vx": vx, "vy": vy,
                "life": life, "maxlife": life,
                "size": np.random.uniform(1.0, 3.0), "color": color})

    def update_and_draw(self, frame):
        alive = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.35
            p["vx"] *= 0.96
            p["life"] -= 1
            if p["life"] <= 0:
                continue
            a = p["life"] / p["maxlife"]
            col = tuple(int(c * a + 255 * (1 - a) * 0.0) for c in p["color"])
            r = max(1, int(p["size"] * a))
            xi, yi = int(p["x"]), int(p["y"])
            if 0 <= xi < frame.shape[1] and 0 <= yi < frame.shape[0]:
                cv2.circle(frame, (xi, yi), r, col, -1)
            alive.append(p)
        self.particles = alive


def draw_comet_trail(frame, trail_pts, _cache={}):
    n = len(trail_pts)
    if n < 2:
        return
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _cache:
        _cache[key] = (np.empty((h, w, 3), dtype=np.uint8),
                       np.zeros((h, w), dtype=np.float32))
    overlay, alpha_mask = _cache[key]
    xs = [p[0] for p in trail_pts]
    ys = [p[1] for p in trail_pts]
    pad = 14
    x0 = max(0, int(min(xs)) - pad)
    x1 = min(w, int(max(xs)) + pad + 1)
    y0 = max(0, int(min(ys)) - pad)
    y1 = min(h, int(max(ys)) + pad + 1)
    alpha_mask[y0:y1, x0:x1] = 0.0
    overlay[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    for i in range(1, n):
        a = i / n
        alpha = 0.10 + 0.6 * a
        thick = max(1, int(1 + 6 * a))
        p0 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
        p1 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
        cv2.line(overlay, p0, p1, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.line(alpha_mask, p0, p1, alpha, thick, cv2.LINE_AA)
    roi_a = alpha_mask[y0:y1, x0:x1]
    active = roi_a > 0
    if active.any():
        roi_f = frame[y0:y1, x0:x1]
        roi_o = overlay[y0:y1, x0:x1]
        a3 = roi_a[..., np.newaxis]
        blended = roi_f.astype(np.float32) * (1.0 - a3) + roi_o.astype(np.float32) * a3
        roi_f[:] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    cx, cy = int(trail_pts[-1][0]), int(trail_pts[-1][1])
    r = 4
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y0 < y1 and x0 < x1:
        ball_roi = frame[y0:y1, x0:x1].copy()
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        alpha = 0.5
        cv2.addWeighted(frame[y0:y1, x0:x1], alpha, ball_roi, 1.0 - alpha, 0,
                        frame[y0:y1, x0:x1])


def smooth_trail(points):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    xs = np.convolve(pts[:, 0], _SMOOTH_KERNEL, mode="same")
    ys = np.convolve(pts[:, 1], _SMOOTH_KERNEL, mode="same")
    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    return list(zip(xs, ys))


def render(args, segments, filled_track, events, table_geo,
           in_w, in_h, fps, total_frames):
    out_h = args.output_height or in_h
    out_w = int(round(in_w * out_h / in_h))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    scale = out_w / in_w
    ev_by_frame = defaultdict(list)
    landing_all = []
    for ev in events:
        ev_by_frame[ev["frame"]].append(ev)
        if ev["type"] == "bounce":
            landing_all.append(ev)
    scaled_ev_by_frame = defaultdict(list)
    for f, evs in ev_by_frame.items():
        for ev in evs:
            scaled_ev_by_frame[f].append({
                "x": ev["x"] * scale,
                "y": ev["y"] * scale,
                "type": ev["type"],
            })

    def seg_of_frame(f):
        for i, (s, e) in enumerate(segments):
            if s <= f <= e:
                return i
        return -1

    landing_by_seg = defaultdict(list)
    for lp in landing_all:
        si = seg_of_frame(lp["frame"])
        if si >= 0:
            landing_by_seg[si].append(lp)
    scaled_track = {f: (x * scale, y * scale)
                    for f, (x, y, _) in filled_track.items()}
    tmp_dir = tempfile.mkdtemp(prefix="pphl_")
    silent_path = os.path.join(tmp_dir, "silent.mp4")
    enc = detect_amf_encoder(args.encoder)
    use_pipe = False
    pipe_proc = None
    writer = None
    if shutil.which("ffmpeg"):
        try:
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{out_w}x{out_h}", "-r", str(fps),
                "-i", "-",
            ] + _enc_args(enc) + [
                "-pix_fmt", "yuv420p",
                silent_path,
            ]
            pipe_proc = subprocess.Popen(
                ffmpeg_cmd, stdin=subprocess.PIPE,
                creationflags=_SUBPROC_FLAGS)
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
    mini = MiniTable(out_w, out_h, table_geo)
    particles = ParticleSystem()
    TRAIL_LEN = max(12, int(fps * 0.5))
    render_frames = set()
    for s, e in segments:
        for f in range(s, e + 1):
            render_frames.add(f)
    total_render = len(render_frames)
    pipe_method = "ffmpeg pipe 直编" if use_pipe else "cv2.VideoWriter"
    log(f"🎬 渲染: {len(segments)} 个片段, 约 {total_render} 帧 "
        f"@ {out_w}x{out_h} {fps:.2f}fps [{pipe_method} → {enc}]")
    written = 0
    t_start = time.time()
    last_progress_time = t_start
    for si, (s, e) in enumerate(segments):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        trail = deque(maxlen=TRAIL_LEN)
        cur_landings = landing_by_seg.get(si, [])
        cur_landings_sorted = sorted(cur_landings, key=lambda lp: lp["frame"])
        lp_idx = 0
        cur_lp_so_far = []
        for f in range(s, e + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            for sev in scaled_ev_by_frame.get(f, []):
                particles.emit(sev["x"], sev["y"],
                               n=22 if sev["type"] == "bounce" else 16,
                               kind=sev["type"])
            if f in scaled_track:
                trail.append(scaled_track[f])
            else:
                if trail:
                    trail.popleft() if len(trail) > 1 else None
            if len(trail) >= 2:
                draw_comet_trail(frame, smooth_trail(list(trail)))
            particles.update_and_draw(frame)
            while lp_idx < len(cur_landings_sorted) and cur_landings_sorted[lp_idx]["frame"] <= f:
                cur_lp_so_far.append(cur_landings_sorted[lp_idx])
                lp_idx += 1
            mini.draw(frame, cur_lp_so_far)
            if use_pipe:
                try:
                    pipe_proc.stdin.write(frame.ravel().data)
                except (BrokenPipeError, OSError):
                    log("⚠️ ffmpeg pipe 断开, 回退到 cv2.VideoWriter")
                    pipe_proc.wait()
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
    cap.release()
    if use_pipe:
        try:
            pipe_proc.stdin.close()
        except Exception:
            pass
        pipe_proc.wait()
    if writer is not None:
        writer.release()
    elapsed_total = time.time() - t_start
    log(f"✅ 无声视频渲染完成: {written} 帧, 耗时 {elapsed_total:.1f}s "
        f"({written / max(elapsed_total, 1e-6):.1f} fps)")
    finalize_output(args, silent_path, segments, fps, written, tmp_dir,
                    already_encoded=use_pipe, enc=enc)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def finalize_output(args, silent_path, segments, fps, written, tmp_dir,
                    already_encoded=False, enc="libx264"):
    has_audio = (not args.no_audio) and ffprobe_has_audio(args.video)
    if not has_audio or not shutil.which("ffmpeg"):
        if already_encoded:
            log(f"🎞️ 无音频, 直接复制已编码视频 ({enc})")
            shutil.copy2(silent_path, args.output)
        else:
            if enc == "libx264" and already_encoded is False:
                enc = detect_amf_encoder(args.encoder)
            log(f"🎞️ 无音频, 编码 ({enc})")
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                   "-i", silent_path] + _enc_args(enc) + [args.output]
            subprocess.run(cmd, check=True, creationflags=_SUBPROC_FLAGS)
        return
    parts = []
    filt = []
    for i, (s, e) in enumerate(segments):
        t0 = s / fps
        t1 = (e + 1) / fps
        filt.append(
            f"[1:a]atrim=start={t0:.4f}:end={t1:.4f},asetpts=PTS-STARTPTS[a{i}]")
        parts.append(f"[a{i}]")
    concat = "".join(parts) + f"concat=n={len(segments)}:v=0:a=1[aout]"
    filter_complex = ";".join(filt + [concat])
    if already_encoded:
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux (视频 copy, 音频 aac)")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", silent_path,
            "-i", args.video,
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            args.output,
        ]
    else:
        if enc == "libx264" and already_encoded is False:
            enc = detect_amf_encoder(args.encoder)
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux ({enc})")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", silent_path,
            "-i", args.video,
            "-filter_complex", filter_complex,
            "-map", "0:v:0", "-map", "[aout]",
        ] + _enc_args(enc) + [
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart",
            args.output,
        ]
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


def _enc_args(enc):
    if enc == "h264_amf":
        return ["-c:v", "h264_amf", "-usage", "transcoding",
                "-quality", "quality", "-rc", "cqp",
                "-qp_i", "22", "-qp_p", "24", "-bf", "0", "-pix_fmt", "yuv420p"]
    if enc == "hevc_amf":
        return ["-c:v", "hevc_amf", "-usage", "transcoding",
                "-quality", "quality", "-rc", "cqp",
                "-qp_i", "22", "-qp_p", "24", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p"]


# =====================================================================
#  main
# =====================================================================
def main():
    args = parse_args()
    if not os.path.exists(args.video):
        log(f"❌ 视频不存在: {args.video}"); sys.exit(1)
    if not os.path.exists(args.raw_detections):
        log(f"❌ 检测日志不存在: {args.raw_detections}"); sys.exit(1)
    log("=" * 60)
    log("🏓 PingPong Highlight Generator")
    log("=" * 60)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        log(f"❌ 无法打开视频: {args.video}"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log(f"📹 {in_w}x{in_h} @ {fps:.2f}fps, {total_frames} 帧")

    log("📖 读取检测日志 ...")
    frames = load_detections(args.raw_detections)
    log(f"   {len(frames)} 条帧记录")

    log("📐 提取球台/球网几何 + 长边走廊 ...")
    table_geo = extract_table_geometry(
        frames, (in_h, in_w), debug=args.debug,
        corridor_margin=args.corridor_margin,
        corridor_extend=args.corridor_extend)

    log("⚪ 提取主球轨迹 (走廊过滤 + 多球过滤) ...")
    ball_cands = gather_ball_candidates(frames, fps)
    ball_cands = filter_static_ball_candidates(
        ball_cands, fps, in_w, in_h,
        min_static_count=args.min_static_ball_count, debug=args.debug)
    # ★ 长边走廊过滤: 优先剔除台外球 (短边外的发球/高抛保留)
    corridor = table_geo.get("corridor") if table_geo else None
    if corridor is not None:
        ball_cands = filter_ball_by_corridor(ball_cands, corridor, debug=args.debug)
    # 二次保险: 空间聚类分离旁桌残余
    ball_cands = cluster_ball_by_court(ball_cands, in_w, in_h, debug=args.debug)
    main_track = build_main_ball_track(
        ball_cands, fps, in_w, in_h, table_geo=table_geo, debug=args.debug)
    log(f"   主轨迹原始点数: {len(main_track)}")

    log("🧩 物理补帧 ...")
    filled = interpolate_track(main_track, fps, in_w, in_h, debug=args.debug)

    log("🎾 检测回合与落点/击球事件 ...")
    rallies = detect_rallies(filled, fps, total_frames, args.raw_detections, debug=args.debug)
    events = detect_events(filled, table_geo, fps)
    log(f"   回合: {len(rallies)}, 事件: {len(events)}")

    segments = build_segments(rallies, fps, total_frames, args)
    if not segments:
        log("❌ 没有检测到有效回合, 无法生成高光。可尝试放宽 --min-rally-frames")
        sys.exit(2)
    total_out = sum(e - s + 1 for s, e in segments)
    log(f"✂️  有效片段: {len(segments)} 个, 共 {total_out} 帧 "
        f"(~{total_out / fps:.1f}s)")

    render(args, segments, filled, events, table_geo,
           in_w, in_h, fps, total_frames)
    log(f"🎉 完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
