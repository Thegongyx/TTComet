#!/usr/bin/env python3
"""
pingpong_highlight.py  (优化版 v4.2 - 在 v4.1 基础上把 track↔detection 关联改为匈牙利算法)
=====================================================================
乒乓球比赛 → 自动特效精华集锦

v4.2 在 v4.1 基础上的改动 (默认行为可能略有差异, 但可通过 --match-algo greedy 复现):
  ★ build_main_ball_track: 把原来的「按 active track 顺序逐个 argmin」贪心关联,
    改为基于代价矩阵的匈牙利算法 (scipy.optimize.linear_sum_assignment) 全局最优分配,
    仍沿用原有 tol = max_jump * (1 + 0.1*(gap-1)) 门限对每条配对做事后过滤.
  ★ 新增 CLI: --match-algo {auto, hungarian, greedy}, 默认 auto:
      - 有 scipy 时使用匈牙利 (全局最小总代价)
      - 无 scipy 或显式 greedy 时使用原来的贪心实现 (与 v4.1 完全等价)
  ★ 其他逻辑 (predict / score / 球网穿越加成 / IoU 空间共域过滤 / score_min) 全部保持不变.

v4.1 在 v4 基础上的修复 (行为基本兼容):
  ★ 移除死代码: 未使用的 scipy 导入、MiniTable.geo、side_edges 变量
  ★ 去重: extract_table_geometry 中内联的球网相交判定改为复用 _net_bisects_quad
  ★ 修复 _finalize_rally: 之前 is_valid 计算后被忽略, 现在真正生效
  ★ 修复 _get_net_center_x: 之前永远返回 table_x (net_x 死变量), 已简化
  ★ 修正若干 CLI 帮助文案与实际默认值不符

v4 关键能力全部保留:
  1. 几何合理性校验 validate_and_refine_quad
  2. 基于真实尺寸+球网约束的重建 rebuild_quad_from_constraints
  3. 走廊最小尺寸兜底 + 软过滤自愈
  4. 时序鲁棒聚合 (RANSAC 式中位数 + MAD 剔除)
  5. 静止球过滤(网格哈希)、主轨迹追踪剪枝、彗星拖尾缓存
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

# smooth_trail 预分配卷积核
_SMOOTH_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float64)


# =====================================================================
#  标准球台常量 (ITTF)
# =====================================================================
TABLE_LEN_M = 2.74     # 长
TABLE_WID_M = 1.525    # 宽
TABLE_RATIO = TABLE_LEN_M / TABLE_WID_M   # ≈ 1.797 (长/宽)


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
    p.add_argument("--min-static-ball-count", type=int, default=20,
                   help="静止球过滤: 600帧窗口内最少静止检测数, 达到才剔除 (默认 20)")
    p.add_argument("--pre-buffer", type=float, default=0.5,
                   help="每个回合前缓冲秒数")
    p.add_argument("--post-buffer", type=float, default=1.5,
                   help="每个回合后缓冲秒数")
    p.add_argument("--merge-gap", type=float, default=2.0,
                   help="相邻回合间隔小于该秒数则合并")
    p.add_argument("--corridor-margin", type=float, default=0.5,
                   help="长边走廊横向容差 (近端短边宽度比例, 默认 0.5)")
    p.add_argument("--corridor-extend", type=float, default=1.2,
                   help="走廊沿长轴向短边外延伸倍数 (球台长度的倍数, 默认 1.2)")
    p.add_argument("--out-of-table-timeout", type=float, default=1.0,
                   help="球(含插值)持续出走廊该秒数则回合结束 (默认 1.0s)")
    p.add_argument("--corridor-max-drop", type=float, default=0.55,
                   help="走廊过滤单次剔除比例上限, 超过视为几何不可信并放宽 (默认 0.55)")
    p.add_argument("--no-audio", action="store_true",
                   help="不保留原声")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_amf", "hevc_amf"],
                   help="输出编码器, auto=优先 AMD AMF 硬编")
    p.add_argument("--max-jump-ratio", type=float, default=0.1,
                   help="轨迹追踪最大跳变距离相对图像对角线的比例 (默认 0.1)")
    p.add_argument("--match-algo", default="auto",
                   choices=["auto", "hungarian", "greedy"],
                   help="track↔detection 关联算法: auto=有 scipy 用匈牙利, "
                        "hungarian=强制匈牙利 (需要 scipy), greedy=原 v4.1 贪心 (默认 auto)")
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


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


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
#  2) 球台 / 球网几何提取
# =====================================================================
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
        rect = cv2.minAreaRect(pts)
        quad = cv2.boxPoints(rect)
    return quad.astype(np.float32)


def _quad_is_convex_simple(quad):
    """检测四边形是否凸且无自交 (排除退化/蝴蝶形)。"""
    q = np.array(quad, dtype=np.float64)
    n = 4
    signs = []
    for i in range(n):
        a = q[i]
        b = q[(i + 1) % n]
        c = q[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = [s for s in signs if abs(s) > 1e-6]
    if not signs:
        return False
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _quad_quality(quad, img_hw):
    """评估一个 quad 作为乒乓球台的合理性。"""
    h, w = img_hw
    q = order_quad(quad).astype(np.float64)
    edges = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    min_edge = min(edges)
    area = 0.5 * abs(sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1]
                         for i in range(4)))
    area_frac = area / (w * h)
    side_a = (edges[0] + edges[2]) * 0.5
    side_b = (edges[1] + edges[3]) * 0.5
    long_side = max(side_a, side_b)
    short_side = min(side_a, side_b) + 1e-6
    ratio = long_side / short_side

    ok = True
    reason = "ok"
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
    """判断 n 边形中边 i 和边 j 是否相邻 (共享一个顶点)。"""
    diff = abs(i - j)
    return diff == 1 or diff == n - 1


def _extend_line(p1, p2, length):
    """沿 p1->p2 方向两端各延长 length。"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
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
    """返回球网延长线与四边形各边的相交索引列表。"""
    if net_line is None:
        return []
    h, w = img_hw
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    net_ext = _extend_line(net_line[0], net_line[1], length=max(w, h) * 4)
    return [i for i, (a, b) in enumerate(edges)
            if _segments_intersect(net_ext[0], net_ext[1], a, b)]


def _net_bisects_quad(quad, net_line, img_hw):
    """
    检查球网线是否正确地将四边形切分为两半 (与两条对边相交)。
    返回 (bisects: bool, intersect_indices: list)。
    """
    if net_line is None:
        return True, []  # 无球网时不阻断
    inter_idx = _net_quad_intersections(quad, net_line, img_hw)
    if len(inter_idx) != 2:
        return False, inter_idx
    if _edges_are_adjacent(inter_idx[0], inter_idx[1]):
        return False, inter_idx
    return True, inter_idx


def _robust_median_quad(quad_list, img_hw, debug=False):
    """时序鲁棒聚合: 逐角点中位数 + MAD 剔除离群帧。"""
    if len(quad_list) < 3:
        return None
    ordered = np.stack([order_quad(q) for q in quad_list], axis=0)  # (N,4,2)
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
    """用球网多边形底边(BL→BR)方向拟合球网线, 并投影到最远端点.

    多边形走线顺序: TL(左上) → BL(左下) → BR(右下) → TR(右上) → TL
    取 BL→BR (球网接触桌面的底边) 作为球网线方向,
    再将全部多边形点沿该方向投影到最远两端.
    """
    if pts is None or len(pts) < 4:
        return None

    # 1. 找 BL (左下角): 经过左边缘(垂直段)后, x 开始持续增大
    bl_idx = None
    for i in range(1, min(len(pts) - 4, 30)):
        if (pts[i + 1, 0] > pts[i, 0] and
            pts[i + 2, 0] > pts[i + 1, 0] and
            pts[i + 3, 0] > pts[i + 2, 0]):
            bl_idx = i
            break

    if bl_idx is None:
        return None

    # 2. 找 BR (右下角): 跳过至少 10 点, 找到 y 开始持续下降的位置
    search_start = min(bl_idx + 10, len(pts) - 5)
    br_idx = None
    for i in range(search_start, len(pts) - 4):
        if (pts[i + 1, 1] < pts[i, 1] and
            pts[i + 2, 1] < pts[i + 1, 1] and
            pts[i + 3, 1] < pts[i + 2, 1]):
            br_idx = i - 1
            break

    if br_idx is None or br_idx <= bl_idx:
        return None

    bl = pts[bl_idx]
    br = pts[br_idx]

    # 3. 用 BL→BR 方向作为球网线方向
    net_vec = br - bl
    net_len = float(np.linalg.norm(net_vec))
    if net_len < 10.0:
        return None
    net_dir = net_vec / net_len

    # 4. 把所有点沿该方向投影, 取最远两端
    proj = (pts - bl) @ net_dir
    t_min, t_max = proj.min(), proj.max()

    p1 = bl + net_dir * t_max
    p2 = bl + net_dir * t_min

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


# =====================================================================
#  几何校验 + 基于约束的重建
# =====================================================================
def rebuild_quad_from_constraints(table_pts, net_line, img_hw, debug=False):
    """当检测四边形不可信时, 用球网线 + ITTF比例约束重建四角。

    ★ 核心: 当有球网线时, 直接用球网线作为主约束:
       - 球网方向 = 短边方向 (球网始终沿短边延伸)
       - 球网长度 = 短边长度 (球网跨越整个短边, 球网两端≈长边中点)
       - 长边方向 = 球网方向的垂直方向
       - 长边长度 = 短边长度 × ITTF比例 (可混合检测比例)
       - 四角 = 球网两端 ± 长边方向 × (长边长度/2)
    """
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

        # 短边长度 = 球网长度 (球网跨越整个短边), 加 5% 余量
        short_len = net_len * 1.05

        detected_ratio = None
        if table_pts is not None and len(table_pts) >= 4:
            pts = table_pts.astype(np.float32)
            rect = cv2.minAreaRect(pts)
            (_, _), (rw, rh), _ = rect
            if rw > 5 and rh > 5:
                detected_long = max(rw, rh)
                detected_short = min(rw, rh)
                detected_ratio = detected_long / (detected_short + 1e-6)

        if detected_ratio is not None and 1.2 < detected_ratio < 4.0:
            blended_ratio = 0.6 * TABLE_RATIO + 0.4 * detected_ratio
        else:
            blended_ratio = TABLE_RATIO

        long_len = short_len * blended_ratio

        half_long = long_dir * (long_len * 0.5)
        far_left = net_start - half_long
        far_right = net_end - half_long
        near_right = net_end + half_long
        near_left = net_start + half_long

        corners = np.array([far_left, far_right, near_right, near_left],
                           dtype=np.float32)
        q = order_quad(corners)

        if debug:
            ratio_str = f"检测比={detected_ratio:.2f}, " if detected_ratio else ""
            log(f"  ◆ 重建台面四角(球网线约束): "
                f"短边={short_len:.0f}px(网长{net_len:.0f}×1.05) "
                f"长边={long_len:.0f}px({ratio_str}ITTF={TABLE_RATIO:.2f})")
        return q

    # 无球网线时, 回退到 minAreaRect
    if table_pts is None or len(table_pts) < 4:
        return None
    pts = table_pts.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), _ang = rect
    if rw < 1 or rh < 1:
        return None

    box = np.array(cv2.boxPoints(rect), dtype=np.float64)
    e0 = box[1] - box[0]
    e1 = box[2] - box[1]
    len0 = np.linalg.norm(e0) + 1e-6
    len1 = np.linalg.norm(e1) + 1e-6
    u0 = e0 / len0
    u1 = e1 / len1

    if len0 >= len1:
        long_dir, long_len, short_dir, short_len = u0, len0, u1, len1
    else:
        long_dir, long_len, short_dir, short_len = u1, len1, u0, len0
    detected_ratio = long_len / (short_len + 1e-6)
    blended_long = short_len * (0.5 * TABLE_RATIO + 0.5 * detected_ratio)

    cen = np.array([cx, cy], dtype=np.float64)
    hl = long_dir * (blended_long * 0.5)
    hs = short_dir * (short_len * 0.5)
    corners = np.array([cen - hl - hs,
                        cen - hl + hs,
                        cen + hl + hs,
                        cen + hl - hs], dtype=np.float32)
    q = order_quad(corners)
    if debug:
        log(f"  ◆ 重建台面四角(minAreaRect): "
            f"检测比={detected_ratio:.2f} → 使用比={blended_long/short_len:.2f} "
            f"(ITTF={TABLE_RATIO:.2f})")
    return q


def validate_and_refine_quad(quad, table_pts, net_line, img_hw, debug=False):
    """校验 quad 合理性, 不合理则重建。返回 (quad_final, was_rebuilt)。"""
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
        log(f"  ✓ 重建后四角合理: 长宽比={qual2['ratio']:.2f} "
            f"面积占比={qual2['area_frac']:.3f}")
        return rebuilt, True
    log(f"  ⚠️ 重建后仍不理想 ({qual2['reason']}), 仍采用重建结果(优于原始退化四边形)")
    return rebuilt, True


# =====================================================================
#  长边走廊 (有效渲染区) 构造 — 含最小尺寸兜底
# =====================================================================
def build_corridor(far_left, far_right, near_left, near_right,
                   corridor_margin=0.45, corridor_extend=1.2,
                   img_hw=None, debug=False):
    fl = np.array(far_left, dtype=np.float64)
    fr = np.array(far_right, dtype=np.float64)
    nl = np.array(near_left, dtype=np.float64)
    nr = np.array(near_right, dtype=np.float64)

    left_axis = fl - nl
    right_axis = fr - nr

    def _unit(v):
        n = np.linalg.norm(v) + 1e-8
        return v / n
    ul = _unit(left_axis)
    ur = _unit(right_axis)

    body_len_detected = (np.linalg.norm(left_axis) + np.linalg.norm(right_axis)) * 0.5

    near_width_vec = nr - nl
    near_width = np.linalg.norm(near_width_vec) + 1e-8
    body_len_expected = near_width * TABLE_RATIO
    body_len = max(body_len_detected, body_len_expected)

    if debug and body_len > body_len_detected * 1.3:
        log(f"  ◆ 台身长度兜底: 检测={body_len_detected:.0f}px → "
            f"采用={body_len:.0f}px (按近宽{near_width:.0f}×{TABLE_RATIO:.2f})")

    ext = corridor_extend * body_len
    far_left_ext = fl + ul * ext
    far_right_ext = fr + ur * ext
    near_left_ext = nl - ul * ext
    near_right_ext = nr - ur * ext

    side_dir = near_width_vec / near_width
    margin_px = corridor_margin * near_width

    far_left_ext = far_left_ext - side_dir * margin_px
    near_left_ext = near_left_ext - side_dir * margin_px
    far_right_ext = far_right_ext + side_dir * margin_px
    near_right_ext = near_right_ext + side_dir * margin_px

    corridor = np.array([far_left_ext, far_right_ext,
                         near_right_ext, near_left_ext], dtype=np.float32)
    if debug:
        log(f"  ◆ 长边走廊: 横向容差={margin_px:.0f}px (近宽{near_width:.0f}px×{corridor_margin}), "
            f"纵向延伸={ext:.0f}px (台身{body_len:.0f}px×{corridor_extend})")
    return corridor


def _point_in_corridor(pt, corridor):
    if corridor is None:
        return True
    res = cv2.pointPolygonTest(corridor.reshape(-1, 1, 2),
                               (float(pt[0]), float(pt[1])), False)
    return res >= 0


def extract_table_geometry(frames, img_hw, debug=False, persp_blend=0.5,
                           corridor_margin=0.45, corridor_extend=1.2):
    h, w = img_hw
    sample_frames = frames[::10]

    all_table_pts = []
    table_quads = []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "table", conf_min=0.5)
        if pts is not None and len(pts) >= 4:
            all_table_pts.append(pts)
            quad = _fit_quad_single(pts)
            if quad is not None and len(quad) == 4:
                table_quads.append(quad)
    table_pts_agg = (np.concatenate(all_table_pts, axis=0)
                     if all_table_pts else None)

    if len(table_quads) < 3:
        log(f"⚠️ 有效球台拟合帧数不足 ({len(table_quads)}<3), 无法做迷你球台 / 落点分析")
        return None

    quad = _robust_median_quad(table_quads, img_hw, debug=debug)
    if quad is None:
        log("⚠️ 球台四角中位数拟合失败")
        return None

    net_lines = []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "net", conf_min=0.5)
        if pts is not None and len(pts) >= 2:
            line = _fit_net_line_single(pts)
            if line is not None:
                net_lines.append(line)
    net_line = _median_net_line(net_lines) if len(net_lines) >= 3 else None

    quad, was_rebuilt = validate_and_refine_quad(
        quad, table_pts_agg, net_line, img_hw, debug=debug)

    # ★ 检查球网是否正确切分球台 (与两条对边相交), 否则重建
    if net_line is not None and not was_rebuilt:
        bisects_ok, _ = _net_bisects_quad(quad, net_line, img_hw)
        if not bisects_ok:
            if debug:
                log("  ⚠️ 球网与相邻边相交 (四边形可能为菱形/非矩形), 启动约束重建")
            rebuilt = rebuild_quad_from_constraints(
                table_pts_agg, net_line, img_hw, debug=debug)
            if rebuilt is not None:
                quad = rebuilt
                was_rebuilt = True
                bisects_ok2, _ = _net_bisects_quad(quad, net_line, img_hw)
                if debug:
                    if bisects_ok2:
                        log("  ✓ 重建后球网正确切分球台 (与对边相交)")
                    else:
                        log("  ⚠️ 重建后球网仍与相邻边相交, 继续使用重建结果")

    # ★ 检查短边长度是否与球网长度匹配 (球网跨越整个短边)
    if net_line is not None and not was_rebuilt:
        net_len = float(np.hypot(net_line[1][0] - net_line[0][0],
                                  net_line[1][1] - net_line[0][1]))
        q = order_quad(quad).astype(np.float64)
        edge_lens = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        side_a = (edge_lens[0] + edge_lens[2]) * 0.5
        side_b = (edge_lens[1] + edge_lens[3]) * 0.5
        short_edge = min(side_a, side_b)
        if short_edge < net_len * 0.7:
            if debug:
                log(f"  ⚠️ 短边({short_edge:.0f}px)远小于球网长度({net_len:.0f}px), "
                    f"四边形过窄, 启动约束重建")
            rebuilt = rebuild_quad_from_constraints(
                table_pts_agg, net_line, img_hw, debug=debug)
            if rebuilt is not None:
                quad = rebuilt
                was_rebuilt = True

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

    # ---- 确定端边 (near/far) ----
    found_end_edges = False
    if net_line is not None:
        inter_idx = _net_quad_intersections(quad, net_line, img_hw)
        if len(inter_idx) == 2 and not _edges_are_adjacent(inter_idx[0], inter_idx[1]):
            found_end_edges = True
            non_inter_idx = [i for i in range(4) if i not in inter_idx]
            end_edge_list = [edge_info[i] for i in non_inter_idx]
            near_corner_idx = int(np.argmin([math.hypot(c[0], c[1] - h) for c in quad]))
            near_corner = quad[near_corner_idx]
            e0_a = np.array(end_edge_list[0][3])
            e0_b = np.array(end_edge_list[0][4])
            if (np.linalg.norm(near_corner - e0_a) < 2.0
                    or np.linalg.norm(near_corner - e0_b) < 2.0):
                near_a, near_b = end_edge_list[0][3], end_edge_list[0][4]
                far_a, far_b = end_edge_list[1][3], end_edge_list[1][4]
            else:
                near_a, near_b = end_edge_list[1][3], end_edge_list[1][4]
                far_a, far_b = end_edge_list[0][3], end_edge_list[0][4]
        elif debug:
            log("  ⚠️ 球网仍与相邻边相交, 回退到角度法判定端边")

    if not found_end_edges:
        edge_info.sort(key=lambda x: x[2])
        end_edges = edge_info[:2]
        near_corner_idx = int(np.argmin([math.hypot(c[0], c[1] - h) for c in quad]))
        near_corner = quad[near_corner_idx]
        e0_a = np.array(end_edges[0][3])
        e0_b = np.array(end_edges[0][4])
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
    dst = np.array([[0.0, 0.0],
                    [TABLE_WID_M * 1000, 0.0],
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
        cross_near = ((nx2 - nx1) * (near_mid_y - ny1)
                      - (ny2 - ny1) * (near_mid_x - nx1))
        net_side_sign = 1 if cross_near > 0 else -1

    corridor = build_corridor(far_left, far_right, near_left, near_right,
                              corridor_margin=corridor_margin,
                              corridor_extend=corridor_extend,
                              img_hw=img_hw, debug=debug)

    if debug:
        log(f"  球台四角: {quad.tolist()}")
        log(f"  球网线: {net_line}")
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
            "rebuilt": was_rebuilt,
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
    """
    用空间网格哈希替代窗口内 O(n²) 两两比较。
    同一 600 帧窗口内, 若某网格 cell (及其 8 邻域) 聚集了 >= min_static_count
    个空间高度集中的检测, 视为静止球予以剔除。
    """
    if not ball_cands:
        return ball_cands
    window_frames = 600
    img_diag = math.hypot(frame_w, frame_h)
    static_threshold = max(img_diag * 0.0008, 2.0)
    cell = max(static_threshold, 1.0)

    flat = []  # (fidx, di, cx, cy)
    for fidx, cands in ball_cands:
        for di, (cx, cy, _c, _b) in enumerate(cands):
            flat.append((fidx, di, cx, cy))
    if not flat:
        return ball_cands
    flat.sort(key=lambda x: x[0])

    removed_set = set()
    n = len(flat)
    idx = 0
    while idx < n:
        anchor_f = flat[idx][0]
        win_end = anchor_f + window_frames
        win = []
        while idx < n and flat[idx][0] <= win_end:
            win.append(flat[idx])
            idx += 1
        if len(win) < min_static_count:
            continue
        grid = defaultdict(list)  # (gx,gy) -> [(fidx,di,cx,cy)]
        for rec in win:
            _f, _d, cx, cy = rec
            grid[(int(cx / cell), int(cy / cell))].append(rec)
        handled = set()
        for (gx, gy), bucket in grid.items():
            if (gx, gy) in handled:
                continue
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neigh.extend(grid.get((gx + dx, gy + dy), ()))
            if len(neigh) < min_static_count:
                continue
            cxs = np.array([r[2] for r in neigh])
            cys = np.array([r[3] for r in neigh])
            mcx, mcy = float(np.median(cxs)), float(np.median(cys))
            d = np.hypot(cxs - mcx, cys - mcy)
            close = [neigh[i] for i in range(len(neigh)) if d[i] <= static_threshold]
            if len(close) >= min_static_count:
                for _f, _d, _cx, _cy in close:
                    removed_set.add((_f, _d))
                handled.add((gx, gy))

    if not removed_set:
        return ball_cands
    filtered = []
    for fidx, cands in ball_cands:
        remaining = [det for di, det in enumerate(cands)
                     if (fidx, di) not in removed_set]
        filtered.append((fidx, remaining))
    if debug:
        log(f"  ★ 静止球过滤(网格哈希): 剔除 {len(removed_set)} 个静止球坐标 "
            f"(窗口={window_frames}帧, 阈值={static_threshold:.0f}px, "
            f"最少静止数={min_static_count})")
    return filtered


def filter_ball_by_corridor(ball_cands, corridor, max_drop_ratio=0.55,
                            geo_rebuilt=False, debug=False):
    """走廊软过滤 + 自愈: 过滤比例过高则判定几何不可信, 跳过过滤。"""
    if corridor is None or not ball_cands:
        return ball_cands
    n_before = 0
    n_after = 0
    filtered = []
    for fidx, cands in ball_cands:
        keep = []
        for det in cands:
            n_before += 1
            if _point_in_corridor((det[0], det[1]), corridor):
                keep.append(det)
                n_after += 1
        filtered.append((fidx, keep))
    if n_before == 0:
        return ball_cands
    drop_ratio = 1.0 - (n_after / n_before)
    effective_max = max_drop_ratio + (0.15 if geo_rebuilt else 0.0)
    if drop_ratio > effective_max:
        log(f"  ⚠️ 走廊过滤将剔除 {drop_ratio*100:.0f}% 的球 "
            f"(> 上限 {effective_max*100:.0f}%), 判定几何不可信 → 跳过走廊过滤, "
            f"改由空间聚类分离台外球")
        return ball_cands
    if debug:
        log(f"  ★ 长边走廊过滤: {n_before} → {n_after} 个球检测 "
            f"(剔除走廊外 {n_before - n_after} 个, 占比 {drop_ratio*100:.0f}%)")
    return filtered


def cluster_ball_by_court(ball_cands, frame_w, frame_h, debug=False):
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
                        for nr, nc in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
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


def build_main_ball_track(ball_cands, fps, frame_w, frame_h, table_geo=None, debug=False,
                          max_jump_ratio=0.05, match_algo="auto"):
    """
    活跃 track 集合 + 提前剪枝; predict 在大 gap 时速度衰减外推。

    v4.2: track ↔ detection 关联支持匈牙利算法 (全局最优), 失败/缺失依赖时回退到贪心.
        match_algo:
          - "auto"      : 有 scipy.optimize.linear_sum_assignment 时使用匈牙利, 否则贪心
          - "hungarian" : 强制匈牙利 (无 scipy 则警告并回退贪心)
          - "greedy"    : 强制原 v4.1 贪心
    """
    max_jump = max_jump_ratio * math.hypot(frame_w, frame_h)
    max_gap_frames = max(1, int(fps * 0.5))
    tracks = []

    # ---- 决定本次使用的关联算法 ----
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

    # 不可达哨兵: 必须远大于任何可能的 tol, 用于在代价矩阵中屏蔽超距配对.
    # 距离上界约为 hypot(frame_w, frame_h), 取 1e9 远超此值.
    _BIG = 1e9

    for fidx, cands in ball_cands:
        used = [False] * len(cands)
        active = [tr for tr in tracks
                  if 0 < (fidx - tr["pts"][-1][0]) <= max_gap_frames]

        if active and cands:
            # 预先算好每个 active 的预测点和 gap-aware 容差, 两种算法共用
            preds = []
            tols = []
            for tr in active:
                gap = fidx - tr["pts"][-1][0]
                preds.append(predict(tr, fidx))
                tols.append(max_jump * (1 + 0.1 * (gap - 1)))

            if algo == "hungarian":
                # ---- 匈牙利: 构造 (n_active x n_cands) 代价矩阵, 超过 tol 的位置置为 _BIG ----
                n_a, n_c = len(active), len(cands)
                cost = np.full((n_a, n_c), _BIG, dtype=np.float64)
                for i, (px, py) in enumerate(preds):
                    ti = tols[i]
                    for j, (cx, cy, _conf, _b) in enumerate(cands):
                        d = math.hypot(cx - px, cy - py)
                        if d < ti:
                            cost[i, j] = d
                # linear_sum_assignment 在矩形矩阵上也能工作: 返回 min(n_a, n_c) 对
                try:
                    row_ind, col_ind = _linear_sum_assignment(cost)
                except Exception as e:
                    if debug:
                        log(f"  [match] linear_sum_assignment 失败 ({e}), 当帧回退 greedy")
                    row_ind, col_ind = (), ()
                    # 触发下方 greedy 分支
                    algo_this_frame = "greedy"
                else:
                    algo_this_frame = "hungarian"

                if algo_this_frame == "hungarian":
                    for i, j in zip(row_ind, col_ind):
                        # 过滤掉哨兵配对 (本就是超距的)
                        if cost[i, j] >= _BIG:
                            continue
                        cx, cy, conf, b = cands[j]
                        active[i]["pts"].append((fidx, cx, cy, conf, b))
                        used[j] = True
                else:
                    # 退路: 与下面 greedy 分支同实现
                    for i, tr in enumerate(active):
                        px, py = preds[i]
                        ti = tols[i]
                        best_j, best_d = -1, 1e18
                        for j, (cx, cy, _conf, _b) in enumerate(cands):
                            if used[j]:
                                continue
                            d = math.hypot(cx - px, cy - py)
                            if d < best_d:
                                best_d, best_j = d, j
                        if best_j >= 0 and best_d < ti:
                            cx, cy, conf, b = cands[best_j]
                            tr["pts"].append((fidx, cx, cy, conf, b))
                            used[best_j] = True
            else:
                # ---- 贪心 (与 v4.1 完全等价): 每个 active 独立挑最近的未占用 detection ----
                for i, tr in enumerate(active):
                    px, py = preds[i]
                    ti = tols[i]
                    best_j, best_d = -1, 1e18
                    for j, (cx, cy, _conf, _b) in enumerate(cands):
                        if used[j]:
                            continue
                        d = math.hypot(cx - px, cy - py)
                        if d < best_d:
                            best_d, best_j = d, j
                    if best_j >= 0 and best_d < ti:
                        cx, cy, conf, b = cands[best_j]
                        tr["pts"].append((fidx, cx, cy, conf, b))
                        used[best_j] = True

        # 未被任何 track 匹配上的 detection → 新开 track
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
        motion = sum(math.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
                     for i in range(1, n))
        avg_conf = float(np.mean([p[3] for p in pts]))
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


def _finalize_rally(rally_periods, current_rally, end_reason, rally_min_duration):
    """仅当回合有效 (已跨网 且 时长 >= 最小时长) 时记录。
    修复: 旧版计算 is_valid 后被忽略, 任何 >=0.5s 的回合都会被记录。
    """
    rally_duration = current_rally["end"] - current_rally["start"]
    is_valid = (current_rally.get("crossed_net", False)
                and rally_duration >= float(rally_min_duration))
    if not is_valid:
        return False
    rally_periods.append({
        "start_frame": current_rally["start_frame"],
        "end_frame": current_rally["end_frame"],
        "duration": round(rally_duration, 2),
        "ended_reason": end_reason,
    })
    return True


def _get_net_center_x(raw_log_path):
    """回退用: 取球网 (或球台) 包围盒中心 x。
    修复: 旧版的 net_x 为死变量, 永远返回 table_x。
    """
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


def detect_rallies(filled_track, fps, total_frames, raw_log_path, table_geo=None,
                   rally_lost_timeout=1.0, rally_min_duration=0.6,
                   pre_roll_sec=1.0, slow_speed_window_sec=1.0,
                   slow_speed_min_in_play_sec=0.8,
                   out_of_table_timeout=1.0, debug=False):
    if not filled_track or fps <= 0:
        return []
    ball_lookup = {f: ((x, y), not interp) for f, (x, y, interp) in filled_track.items()}
    sorted_frames = sorted(ball_lookup.keys())
    rallies = []
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
        if net_line is not None:
            log(f"  慢速阈值: {slow_thresh:.0f} px/s  球网线判定 (cross product)")
        else:
            log(f"  慢速阈值: {slow_thresh:.0f} px/s  net_center_x={net_center_x:.0f}")
        if corridor is not None:
            log(f"  出台判定: 持续 {OUT_OF_TABLE_FRAMES} 帧 (~{out_of_table_timeout}s) 出走廊则结束")
        else:
            log("  出台判定: 无走廊几何, 跳过该规则")
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
    traj_buffer = deque(maxlen=max(PRE_ROLL_FRAMES * 3, int(fps * 6)))

    def _ball_side(x, y):
        """使用球网线 cross product 判定球在哪一侧 (任意角度均正确)。"""
        if net_line is not None:
            (nx1, ny1), (nx2, ny2) = net_line
            cross = (nx2 - nx1) * (y - ny1) - (ny2 - ny1) * (x - nx1)
            return 'A' if cross > 0 else 'B'
        elif net_center_x is not None:
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
                out_of_table_run = 0
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
                slow_run_frames = 0
                out_of_table_run = 0
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
    """
    bounce 用窗口极值判定 (局部 y 最大值), 抗噪;
    hit 用 vx 反向且幅度足够。
    """
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
        v_mps = (d_mm / 1000.0) / dt
        return v_mps * 3.6

    corridor = table_geo.get("corridor") if table_geo else None

    def _in_table(x, y):
        if corridor is None:
            return True
        return _point_in_corridor((x, y), corridor)

    last_bounce_frame = -10 ** 9
    bounce_cooldown_frames = max(1, int(round(fps * 0.1)))

    for i in range(2, n - 2):
        f, x, y = pts[i]
        f_p, x_p, y_p = pts[i - 1]
        f_n, x_n, y_n = pts[i + 1]
        ev_type = None

        # bounce: 局部 y 极大值 (图像 y 向下, 落点为最低位置即 y 最大)
        is_local_y_max = (y >= pts[i - 1][2] and y >= pts[i + 1][2]
                          and y > pts[i - 2][2] and y > pts[i + 2][2])
        vy_prev = y - y_p
        vy_next = y_n - y
        if is_local_y_max and vy_prev > 0.1 and vy_next < -0.1:
            if f - last_bounce_frame >= bounce_cooldown_frames and _in_table(x, y):
                ev_type = "bounce"
                last_bounce_frame = f

        if ev_type is None:
            vx_prev = x - x_p
            vx_next = x_n - x
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
#  6) 剪辑片段计算
# =====================================================================
def build_segments(rallies, fps, total_frames, args):
    min_frames = args.min_rally_frames or int(fps * 0.6)
    pre = int(args.pre_buffer * fps)
    post = int(args.post_buffer * fps)
    merge_gap = int(args.merge_gap * fps)

    valid_rallies = [r for r in rallies
                     if (r["end_frame"] - r["start_frame"] + 1) >= min_frames]
    if not valid_rallies:
        return [], {}

    buffed = []
    for r in valid_rallies:
        s = r["start_frame"]
        e = r["end_frame"]
        bs = max(0, s - pre)
        be = min(total_frames - 1, e + post)
        buffed.append((bs, be, s))
    buffed.sort(key=lambda x: (x[0], x[1]))

    merged = []
    first = buffed[0]
    merged.append([first[0], first[1], [first[2]]])
    for bs, be, raw_s in buffed[1:]:
        if bs - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], be)
            merged[-1][2].append(raw_s)
        else:
            merged.append([bs, be, [raw_s]])

    segments = [(int(s), int(e)) for s, e, _ in merged]
    rally_starts_by_seg = {}
    for si, (_, _, raw_starts) in enumerate(merged):
        rs = sorted(set(int(x) for x in raw_starts))
        rally_starts_by_seg[si] = rs
    return segments, rally_starts_by_seg


# =====================================================================
#  7) 渲染
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
        margin = 0.15
        if not (-self._x_max * margin <= X <= self._x_max * (1 + margin)
                and -self._y_max * margin <= Y <= self._y_max * (1 + margin)):
            return None
        X = _clamp(X, 0.0, self._x_max)
        Y = _clamp(Y, 0.0, self._y_max)
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
            col = tuple(int(c * a) for c in p["color"])
            r = max(1, int(p["size"] * a))
            xi, yi = int(p["x"]), int(p["y"])
            if 0 <= xi < frame.shape[1] and 0 <= yi < frame.shape[0]:
                cv2.circle(frame, (xi, yi), r, col, -1)
            alive.append(p)
        self.particles = alive


class CometRenderer:
    """彗星拖尾渲染器 (按分辨率缓存复用 overlay/alpha 缓冲)。"""
    def __init__(self):
        self._cache = {}

    def draw(self, frame, trail_pts):
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
        pad = 14
        x0 = max(0, int(min(xs)) - pad)
        x1 = min(w, int(max(xs)) + pad + 1)
        y0 = max(0, int(min(ys)) - pad)
        y1 = min(h, int(max(ys)) + pad + 1)
        alpha_mask[y0:y1, x0:x1] = 0.0
        overlay[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        for i in range(1, n):
            a = i / n
            alpha = 0.07 + 0.5 * a
            thick = max(1, int(1 + 6 * a))
            p0 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
            p1 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
            cv2.line(overlay, p0, p1, (255, 255, 255), thick, cv2.LINE_AA)
            cv2.line(alpha_mask, p0, p1, alpha, thick, cv2.LINE_AA)
        roi_a = alpha_mask[y0:y1, x0:x1]
        if (roi_a > 0).any():
            roi_f = frame[y0:y1, x0:x1]
            roi_o = overlay[y0:y1, x0:x1]
            a3 = roi_a[..., np.newaxis]
            blended = roi_f.astype(np.float32) * (1.0 - a3) + roi_o.astype(np.float32) * a3
            roi_f[:] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        cx, cy = int(trail_pts[-1][0]), int(trail_pts[-1][1])
        r = 4
        yy0, yy1 = max(0, cy - r), min(h, cy + r + 1)
        xx0, xx1 = max(0, cx - r), min(w, cx + r + 1)
        if yy0 < yy1 and xx0 < xx1:
            ball_roi = frame[yy0:yy1, xx0:xx1].copy()
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
            alpha = 0.5
            cv2.addWeighted(frame[yy0:yy1, xx0:xx1], alpha, ball_roi, 1.0 - alpha, 0,
                            frame[yy0:yy1, xx0:xx1])


def smooth_trail(points):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    xs = np.convolve(pts[:, 0], _SMOOTH_KERNEL, mode="same")
    ys = np.convolve(pts[:, 1], _SMOOTH_KERNEL, mode="same")
    # 端点保护: convolve 'same' 会在两端引入零填充偏移, 恢复原始端点
    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    return list(zip(xs, ys))


def render(args, segments, rally_starts_by_seg, filled_track, events, table_geo,
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
    cap = None
    try:
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
        mini = MiniTable(out_w, out_h)
        particles = ParticleSystem()
        comet = CometRenderer()
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
            rally_starts = sorted(rally_starts_by_seg.get(si, []))
            rs_ptr = 0
            for f in range(s, e + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                if scale != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                while rs_ptr < len(rally_starts) and rally_starts[rs_ptr] <= f:
                    rstart = rally_starts[rs_ptr]
                    if rstart > s:
                        cur_lp_so_far = []
                    rs_ptr += 1

                for sev in scaled_ev_by_frame.get(f, []):
                    particles.emit(sev["x"], sev["y"],
                                   n=22 if sev["type"] == "bounce" else 16,
                                   kind=sev["type"])
                if f in scaled_track:
                    trail.append(scaled_track[f])
                elif trail and len(trail) > 1:
                    trail.popleft()
                if len(trail) >= 2:
                    comet.draw(frame, smooth_trail(list(trail)))
                particles.update_and_draw(frame)
                while lp_idx < len(cur_landings_sorted) and cur_landings_sorted[lp_idx]["frame"] <= f:
                    cur_lp_so_far.append(cur_landings_sorted[lp_idx])
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
        if enc == "libx264":
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
                "-qp_i", "20", "-qp_p", "20", "-bf", "0", "-pix_fmt", "yuv420p"]
    if enc == "hevc_amf":
        return ["-c:v", "hevc_amf", "-usage", "transcoding",
                "-quality", "quality", "-rc", "cqp",
                "-qp_i", "20", "-qp_p", "20", "-pix_fmt", "yuv420p"]
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
    log("🏓 PingPong Highlight Generator (v4.1 几何自愈 + 性能优化 + 逻辑修复)")
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

    log("📐 提取球台/球网几何 + 校验自愈 + 长边走廊 ...")
    table_geo = extract_table_geometry(
        frames, (in_h, in_w), debug=args.debug,
        corridor_margin=args.corridor_margin,
        corridor_extend=args.corridor_extend)

    geo_rebuilt = bool(table_geo.get("rebuilt")) if table_geo else False

    log("⚪ 提取主球轨迹 (走廊软过滤 + 多球过滤) ...")
    ball_cands = gather_ball_candidates(frames, fps)
    ball_cands = filter_static_ball_candidates(
        ball_cands, fps, in_w, in_h,
        min_static_count=args.min_static_ball_count, debug=args.debug)
    corridor = table_geo.get("corridor") if table_geo else None
    if corridor is not None:
        ball_cands = filter_ball_by_corridor(
            ball_cands, corridor,
            max_drop_ratio=args.corridor_max_drop,
            geo_rebuilt=geo_rebuilt, debug=args.debug)
    ball_cands = cluster_ball_by_court(ball_cands, in_w, in_h, debug=args.debug)
    main_track = build_main_ball_track(
        ball_cands, fps, in_w, in_h, table_geo=table_geo, debug=args.debug,
        max_jump_ratio=args.max_jump_ratio, match_algo=args.match_algo)
    log(f"   主轨迹原始点数: {len(main_track)}")

    log("🧩 物理补帧 ...")
    filled = interpolate_track(main_track, fps, in_w, in_h, debug=args.debug)

    log("🎾 检测回合与落点/击球事件 ...")
    rallies = detect_rallies(filled, fps, total_frames, args.raw_detections,
                             table_geo=table_geo,
                             out_of_table_timeout=args.out_of_table_timeout,
                             debug=args.debug)
    events = detect_events(filled, table_geo, fps)
    log(f"   回合: {len(rallies)}, 事件: {len(events)}")

    segments, rally_starts_by_seg = build_segments(rallies, fps, total_frames, args)
    if not segments:
        log("❌ 没有检测到有效回合, 无法生成高光。可尝试放宽 --min-rally-frames")
        sys.exit(2)
    total_out = sum(e - s + 1 for s, e in segments)
    log(f"✂️  有效片段: {len(segments)} 个, 共 {total_out} 帧 "
        f"(~{total_out / fps:.1f}s)")
    if args.debug:
        for si, (s, e) in enumerate(segments):
            log(f"    片段{si}: [{s}-{e}]  内含回合起始帧={rally_starts_by_seg.get(si)}")

    render(args, segments, rally_starts_by_seg, filled, events, table_geo,
           in_w, in_h, fps, total_frames)
    log(f"🎉 完成! 输出: {args.output}")


if __name__ == "__main__":
    main()