#!/usr/bin/env python3
"""
pingpong_highlight.py  (优化版 v4.16 - 物理抛物线轨迹净化与平滑 + 脚本缺陷修复)
=====================================================================
乒乓球比赛 → 自动特效精华集锦

v4.16 在 v4.15 基础上的改动:
  ★ 重力抛物线轨迹净化与平滑 + 多球台邻台球隔离 (消除误检折线与隔壁台串球):
    - 隔壁球台干扰空域提取 (_extract_neighbor_table_zones): 自动识别非主球台的常驻背景台/侧边台打球区,
      结合主球台表面亲和度、景深球框视直径比例、过网有效高度窗及恢复严格 IoU>=0.15 主判据, 彻底过滤邻台独立轨迹。
    - 防跨台吸球与时空交汇切断 (assoc_cost / _split_switched_subsegs / _ballistic_fill_gap):
      收紧多帧缺口预测门限并拦截非弹跳反重力上窜; 对跨缺口两侧子弧做时空交汇最小距离检验 (δ_meet),
      一旦前景落网球误吸咐到背景隔壁台球立即切断并剔除悬空邻台段; 补帧阶段拒绝强反重力缺口硬连斜直线。
    - 物理重力标定 (estimate_gravity_px_per_frame2): 由球台透视四角反推中台比例尺与俯仰角,
      结合实测干净弧段中位数自动标定像素重力加速度 g_frame (px/帧²)。
    - 关联与主轨迹合并升级 (build_main_ball_track): 多点速度+重力预测、运动学方向代价惩罚、
      按得分时序排他合并 (杜绝多条重叠轨迹逐帧交替跳变)。
    - 多轮物理抛物线离群点剔除 (clean_ball_track_parabolic): 先裁首尾误检折钩, 再用双侧支撑
      (各≥2点) Leave-Block-Out 抛物线桥接剔除 1~3 帧空中误检尖刺, 100% 保护真实台面弹跳与球拍击球顶点。
    - 分段自由飞行抛物线平滑 + 弹跳缺口解析补帧 (interpolate_track):
      在击球/弹跳断点及跨事件缺口处切分自由飞行子弧, 段内做带重力上下界约束的抛物线平滑消除检测框抖动;
      缺口补帧限制重力上界防过冲跑飞, 支持缺口内单次台面弹跳的双段抛物线闭式重建, 拒绝跨击球反向缺口硬连直线。
    - 渲染拖尾断续保护 (render): 修复缺口期间 trail 残留最后 1 个幽灵点导致跨屏拉直线的 bug,
      帧号不连续或跨段时自动清空旧残影。
  ★ 脚本层面其他问题修复:
    - 修复侧拍梯形球台 extract_table_geometry 长边对角扭曲(沙漏化)及 build_corridor 顶部开放导致右侧走廊坍缩的问题
    - 修复 detect_bounce_events 击球判定误限桌面 quad (改为走廊区域)、跨长缺口伪落点/伪击球及 1 帧过短弹跳冷却的问题
    - 修复渲染在分段剪辑切换时粒子/文字/闪屏残留串场、前置集锦无进度输出及未启用特效列表内存泄漏
    - load_detections 支持 .jsonl.gz 透明读取并裁剪无用字段省内存, 主流程提取后及时释放 frames
    - 修复静止球过滤固定窗口边界漏检、回合回退分支缺 duration 字段、Windows 超长音频滤镜命令行上限等问题

v4.15 在 v4.14 基础上的改动:
  ★ 残影 (球拖尾) 透明度与粗细整体 ×0.77:
    TRAIL_ALPHA_SCALE = 0.77  (alpha = (0.15+0.55a) · (0.85+0.15·intensity) · 0.77)
    TRAIL_THICK_SCALE = 0.77  (thick_scale = (1+1.8·snorm) · 2/3 · 0.77)
    只缩放整体浓淡/粗细, 尾部淡 → 球头浓的渐变形状与颜色插值保持不变;
    想再调只需改这两个常数 (1.0 = v4.14 及以前的观感)。

v4.14 在 v4.13 基础上的改动:
  ★ 补帧视野由 0.2s 放宽到 0.3s (interpolate_track):
    max_gap = max(1, int(fps * 0.3))    (30fps → 9 帧, 60fps → 18 帧)
    (v4.13 曾收到 0.2s; 0.3s 回收一部分中等缺口, 保留"超长缺口不猜"的原则)
    关联阶段的 max_gap_frames 仍是 0.5s, 不受影响。

v4.13 在 v4.12 基础上的改动:
  ★ 补帧视野由 0.5s 收窄到 0.2s (interpolate_track):
    max_gap = max(1, int(fps * 0.2))    (30fps → 6 帧, 60fps → 12 帧)
    0.2s 以上的缺口不再用曲线猜 (0.2s 高速球可飞出数米, 平滑曲线注定是错的),
    这些缺口留空; 关联阶段的 max_gap_frames 仍是 0.5s, 不受影响。

v4.12 在 v4.11 基础上的改动:
  ★ 球拖尾 (残影) 时长由 0.15s 调回 0.2s:
    TRAIL_LEN = max(4, int(fps * 0.2))
    (30fps → 6 帧, 60fps → 12 帧; 下限 4 帧用于极低 fps 场景)

v4.11 在 v4.10 基础上的改动:
  ★ 补帧算法升级 (interpolate_track): 无约束多项式拟合 → 端点钉死的弹道插值
    - x 取缺口两端点连线 (静止机位下 x(t) 近似线性, 避免二次项横向过冲)
    - y 取 y0 + v0'·τ + ½·a·τ², a 由缺口两侧实测速度 v0/v1 反解后取均值,
      v0' 再反解一次 → 曲线严格过两端真实检测点
    - 效果: 缺口边界不再有台阶/折角 (旧 polyfit 不过端点),
      只在两端点之间取值 → 无窗口外过冲/跑飞 (旧法用 4 点撑 15 帧窗口)
    - 退化保护: 取不到速度 / a 非有限 / a<=0 (反重力) → 该缺口 y 退回线性弦
    - 调试输出新增: 走弹道的缺口数与退化为弦的缺口数

v4.10 在 v4.9 基础上的改动:
  ★ 球拖尾 (残影) 时长由 0.3s 缩短为 0.15s:
    TRAIL_LEN = max(4, int(fps * 0.15))
    (原为 max(12, int(fps * 0.3)), 12 帧下限使 30fps 下实际为 0.4s,
     现下限降为 4 帧以保证 0.15s 真正生效, 同时仍够 smooth_trail 平滑)

v4.9 在 v4.8 基础上的改动:
  ★ 精彩回合候选池增加"或"条件:
    - 时长 >= --highlight-min-duration (秒) 或
      落点较少半台的落点数 > --highlight-min-bounce-count (默认 5)
    - 满足任一条件即可进入球速排序候选池

v4.8 在 v4.7 基础上的改动:
  ★ 前置精彩回合功能完善:
    - 合并前回合时长 > --highlight-min-duration (默认 10s) 的回合参与排序
    - 按平均球速 (px/s) 降序取前 --highlight-top-n (默认 7) 个回合
    - 复制一份放到视频最前面, 用 --highlight-black-gap (默认 1s) 黑幕隔开
    - 新增 --no-highlight-prepend 可关闭该功能
  ★ _pick_top_speed_segments 效率优化:
    - 将 ball_speed_table 预展平为 numpy 数组 (按帧号索引)
    - 每个回合用切片 + 布尔掩码一次性求均值, 替代逐帧 dict.get 循环
    - 对数千帧的回合列表, 球速计算从 O(n*python_loop) 降为 O(n*numpy_slice)
  ★ 修复前置片段渲染中落点追踪 bug:
    - rid 从硬编码 None 改为 prepend_landing_rally_id 查表
    - 修复前置片段内多回合落点不会正确重置的问题

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

本次移植跟踪版已验证的高球 (lob) 修复: 走廊顶部开放 (open_top) +
out-of-table-timeout 默认放宽至 2.0s。
"""

import os
import sys
import gzip
import json
import math
import shutil
import bisect
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

# 落点宽松采纳 (长边延长线放宽):
# 弹跳点即使不在 quad/±80mm 台面内, 只要投影落在台面纵向两端各外延
# _EDGE_EXTEND_RATIO×台长的延长带内 (横向仍限 ±80mm), 且与前一个已采纳落点
# 在 _ALT_WINDOW_SEC 秒内、半台相反(交替), 也计为落点 (供回合结束/精彩球判定用)
_EDGE_EXTEND_RATIO = 0.2
_ALT_WINDOW_SEC = 2.0


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
    p.add_argument("--min-rally-frames", type=int, default=90, help="有效回合最小帧数")
    p.add_argument("--min-rally-bounce-count", type=int, default=2,
                   help="短回合豁免: 时长<--min-rally-frames 的回合, 若落点较少半台的"
                        "落点数 >= 该值则不判无效 (与时长条件取或, 默认 2)")
    p.add_argument("--min-rally-speed-top-n", type=int, default=10,
                   help="球速豁免: 平均球速前 N 名的回合无条件视为有效 (0 关闭, 默认 10)")
    p.add_argument("--min-static-ball-count", type=int, default=20, help="静止球过滤最少静止检测数")
    p.add_argument("--pre-buffer", type=float, default=1.5, help="每个回合前缓冲秒数")
    p.add_argument("--post-buffer", type=float, default=2.5, help="每个回合后缓冲秒数")
    p.add_argument("--merge-gap", type=float, default=2.0, help="相邻回合间隔小于该秒数则合并")
    p.add_argument("--corridor-margin", type=float, default=0.5, help="长边走廊横向容差")
    p.add_argument("--table-edge-tol", type=float, default=30.0,
                   help="球台四边等距带容差(px): 落点允许在 quad 四边形外该像素内通过 (排查用, 默认 30)")
    p.add_argument("--corridor-extend", type=float, default=5.0, help="走廊沿长轴向短边外延伸倍数")
    p.add_argument("--out-of-table-timeout", type=float, default=3.0,
                   help="持续出走廊该秒数则回合结束 (默认 3.0, 放宽以容忍放高球滞空时间)")
    p.add_argument("--rally-lost-timeout", type=float, default=1.5,
                   help="球连续丢失该秒数则回合以 ball_lost 结束 (默认 1.5, 高球漏检多可调大)")
    p.add_argument("--no-bounce-timeout", type=float, default=3.0,
                   help="回合内球台连续该秒数无新落点则回合结束 (默认 3.0)")
    p.add_argument("--corridor-max-drop", type=float, default=0.55, help="走廊过滤单次剔除比例上限")
    p.add_argument("--court-cluster", action="store_true",
                   help="启用球台空间聚类过滤 (DBSCAN 只保留最大簇)。"
                        "实测会误删约 47%% 的真球检测 (被删点置信度与保留点同级, "
                        "且只保留最大簇会让球台一侧完全无点), 默认关闭")
    p.add_argument("--center-dist-ratio", type=float, default=0.2,
                   help="实例中心离画面中心超过该比例×长边时丢弃 (0=关闭)")
    p.add_argument("--ball-conf", type=float, default=0.50,
                   help="乒乓球检测最小置信度, 低于该值的球候选在收集阶段直接丢弃")
    p.add_argument("--no-audio", action="store_true", help="不保留原声")
    p.add_argument("--cut", action=argparse.BooleanOptionalAction, default=True,
                   help="是否按回合剪辑（默认开启）。--no-cut 时整段视频加特效，"
                        "不裁掉捡球/短回合片段；此时回合仍会用于落点标注")
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_amf", "hevc_amf"],
                   help="输出编码器, auto=优先 AMD AMF 硬编")
    p.add_argument("--max-jump-ratio", type=float, default=0.10, help="轨迹最大跳变距离占对角线比例")
    p.add_argument("--match-algo", default="auto", choices=["auto", "hungarian", "greedy"],
                   help="track↔detection 关联算法")
    p.add_argument("--fx-level", default="normal", choices=["minimal", "normal", "max"],
                   help="特效强度")
    p.add_argument("--no-player-fx", action="store_true", help="关闭选手移动特效 (脚下光环)")
    p.add_argument("--highlight-top-n", type=int, default=7,
                   help="前置精彩回合数量 (按平均球速降序取前 N 个, 默认 7)")
    p.add_argument("--highlight-min-duration", type=float, default=6.0,
                   help="前置精彩回合的最小合并前时长 (秒, 默认 6.0)")
    p.add_argument("--highlight-min-bounce-count", type=int, default=4,
                   help="前置精彩回合的落点条件: 落点较少半台的落点数 > 该值"
                        " (与时长条件取或, 满足任一即可入选, 默认 4)")
    p.add_argument("--highlight-black-gap", type=float, default=1.0,
                   help="前置精彩回合与全顺序视频之间的黑幕间隔 (秒, 默认 1.0)")
    p.add_argument("--no-highlight-prepend", action="store_true",
                   help="关闭前置精彩回合功能 (只输出全顺序视频)")
    p.add_argument("--no-track-smooth", action="store_true",
                   help="关闭分段重力抛物线平滑 (仅保留离群误检剔除与弹道补帧, 排查对比用)")
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
def _open_text_maybe_gz(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def load_detections(path):
    frames = {}
    keep_classes = {"pingpangball", "table", "net"} | set(_PLAYER_CLASS_NAMES)
    with _open_text_maybe_gz(path) as f:
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
            raw_dets = obj.get("detections") or []
            slim_dets = []
            for d in raw_dets:
                cls_name = str(d.get("class", ""))
                if cls_name not in keep_classes and cls_name.lower() not in _PLAYER_CLASS_NAMES:
                    continue
                item = {
                    "class": cls_name,
                    "confidence": float(d.get("confidence", 0.0)),
                    "bbox": d.get("bbox"),
                }
                if cls_name in ("table", "net") and d.get("polygon"):
                    item["polygon"] = d["polygon"]
                slim_dets.append(item)
            frames[int(fidx)] = {"frame": int(fidx), "detections": slim_dets}
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


def _get_polygon_pts_single(fr, target_class, conf_min=0.5, img_hw=None,
                            center_dist_max=None):
    """
    取单帧中指定类别的多边形顶点。

    若同一帧识别到多个同类实例（例如多个球台 / 多条球网），
    按各实例几何中心离画面中心最近的原则挑选单个实例返回，
    避免把多个实例的顶点混成一坨导致拟合畸形。

    若指定 center_dist_max（像素），则选中的实例离画面中心超距时
    丢弃该帧（返回 None），防止远端误检污染中位数统计。
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

    # 选实例：单实例直接取；多实例取离画面中心最近的那个
    if len(insts) == 1:
        best = insts[0]
    else:
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
        best = insts[best_idx]

    # 离画面中心太远的实例视为误检，直接丢弃
    if center_dist_max is not None and img_hw is not None:
        h, w = img_hw
        cx, cy = w * 0.5, h * 0.5
        bc = best.mean(axis=0)
        d = math.hypot(bc[0] - cx, bc[1] - cy)
        if d > center_dist_max:
            return None

    return best


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
    """
    从球网分割轮廓拟合"球网与台面的接触线"(球网下边缘)。

    旧实现靠"找连续 4 点 x 递增 / y 递减"猜左右角点, 在 66~80 点的密集轮廓上
    极不稳定: 轮廓末端常有网柱/夹子的凸起, 会让右角点落在网的上边中段, 从而把
    下边缘拟合成**对角线**(实测成功率仅 2.9%, 角度分布 -35°~+10° 双峰)。

    改为下边缘包络拟合, 不依赖轮廓点序:
      1) PCA 求网的长轴 u (朝右) 与垂直方向 v (朝下)
      2) 沿 u 分箱, 每箱取 v 投影最大的点 → 下边缘采样点
      3) 对采样点做 TLS 直线拟合
      4) 端点取采样点在拟合方向上的投影极值
    返回 ((x1, y1), (x2, y2)) 或 None。
    """
    if pts is None or len(pts) < 4:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    cen = pts.mean(axis=0)
    X = pts - cen
    try:
        u = np.linalg.svd(X, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        return None
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]], dtype=np.float64)
    if v[1] < 0:
        v = -v
    t = X @ u
    s = X @ v
    span = float(t.max() - t.min())
    if span < 10.0:
        return None
    n_bins = 20
    bin_idx = np.clip(((t - t.min()) / span * n_bins).astype(int), 0, n_bins - 1)
    env = [pts[bin_idx == b][int(np.argmax(s[bin_idx == b]))]
           for b in range(n_bins) if (bin_idx == b).any()]
    if len(env) < 3:
        return None
    env = np.array(env, dtype=np.float64)
    ecen = env.mean(axis=0)
    try:
        d = np.linalg.svd(env - ecen, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        return None
    if d[0] < 0:
        d = -d
    proj = (env - ecen) @ d
    p1 = ecen + d * float(proj.min())
    p2 = ecen + d * float(proj.max())
    if float(np.linalg.norm(p2 - p1)) < 10.0:
        return None
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    return (tuple(p1.tolist()), tuple(p2.tolist()))


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
# [默认关闭约束重建] True = 禁用全部约束重建。
# 3 处调用点(validate_and_refine_quad / 网线二分失败 / 四边形过窄)都会拿到
# None 并自动回退到检测到的中位数四角, 不再用球网线合成台面。
#
# 关闭原因(2026-11 实测 VID_20261104_223026): 重建基于两个不成立的假设 ——
#   ① 球网投影长度 ≈ 台面短边投影长度; ② 台面长轴 ⊥ 球网投影方向。
# 平视机位下二者都失真(网投影 180px vs 台面短边投影 76px), 会把正确的台面
# 四角重建到台面下方并旋转 ~65°, 走廊随之变成一条竖直窄条, 真球被判出台。
# 同一份日志: 启用重建 = 2 片段/10.3s, 关闭重建 = 28 片段/196.5s。
# 需要恢复原来的"校验自愈"行为时改回 False。
DISABLE_QUAD_REBUILD = True


def rebuild_quad_from_constraints(table_pts, net_line, img_hw, debug=False):
    if DISABLE_QUAD_REBUILD:
        return None
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
    if DISABLE_QUAD_REBUILD:
        log(f"  ⚠️ 球台四角校验告警 ({qual['reason']}, 长宽比={qual['ratio']:.2f}, "
            f"面积占比={qual['area_frac']:.4f}) → 约束重建已禁用, 沿用检测四角")
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
        u_long = _unit(ul + ur)
        # 纵拍/端线视角 (far 显著在 near 上方): 沿左右两条斜边射线向上延伸到 y=top_y
        if u_long[1] < -0.5:
            for near_pt, far_pt in ((near_left_ext, far_left_ext),
                                    (near_right_ext, far_right_ext)):
                d = far_pt - near_pt
                if d[1] < -1e-6:
                    t = (top_y - near_pt[1]) / d[1]
                    if t > 0:
                        far_pt[0] = near_pt[0] + d[0] * t
                        far_pt[1] = top_y
        else:
            # 横拍/侧视或平视角 (near/far 分处左右两端): 将走廊上侧长边的两个端点抬至 y=top_y,
            # 保留下侧长边在球台下方以继续拦截地面误检, 避免误把右侧上下角同时钳到 y=-10 造成走廊坍缩成斜楔形
            mean_y_left = 0.5 * (near_left_ext[1] + far_left_ext[1])
            mean_y_right = 0.5 * (near_right_ext[1] + far_right_ext[1])
            if mean_y_left <= mean_y_right:
                near_left_ext[1] = min(near_left_ext[1], top_y)
                far_left_ext[1] = min(far_left_ext[1], top_y)
            else:
                near_right_ext[1] = min(near_right_ext[1], top_y)
                far_right_ext[1] = min(far_right_ext[1], top_y)
    corridor = np.array([far_left_ext, far_right_ext, near_right_ext, near_left_ext],
                        dtype=np.float32)
    if debug:
        top_note = f", 顶部开放(远端钳至 y={top_y:.0f})" if open_top else ""
        log(f"  ◆ 长边走廊: 横向容差={margin_px:.0f}px, 纵向延伸={ext:.0f}px{top_note}")
    return corridor


def _extract_neighbor_table_zones(sample_frames, main_quad, img_hw, debug=False):
    """从采样帧检出的其他非主球台中聚合常驻邻台干扰区 [(x1, y1, x2, y2), ...]。

    用于在多球台球馆场景下识别背景球台/侧边球台的打球空域, 防止隔壁台乒乓球混入主球轨迹与残影。
    """
    if main_quad is None or not sample_frames:
        return []
    h, w = img_hw
    diag = math.hypot(w, h)
    mq_x1, mq_y1 = float(main_quad[:, 0].min()), float(main_quad[:, 1].min())
    mq_x2, mq_y2 = float(main_quad[:, 0].max()), float(main_quad[:, 1].max())
    mq_w = max(1.0, mq_x2 - mq_x1)
    mq_h = max(1.0, mq_y2 - mq_y1)
    mq_area = mq_w * mq_h
    min_sep = max(15.0, 0.08 * min(mq_w, mq_h))

    raw_boxes = []
    for fr in sample_frames:
        for det in fr.get("detections", []):
            if det.get("class") != "table" or det.get("confidence", 0.0) < 0.40:
                continue
            b = det.get("bbox")
            if not b or len(b) != 4:
                continue
            bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            bw, bh = bx2 - bx1, by2 - by1
            if bw < 20.0 or bh < 6.0:
                continue
            bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
            # 中心在主球台内或与主球台包围盒高度重叠的视为主球台自身
            dist_mq = cv2.pointPolygonTest(main_quad.reshape(-1, 1, 2), (bcx, bcy), True)
            if dist_mq >= -min_sep:
                continue
            ix1, iy1 = max(mq_x1, bx1), max(mq_y1, by1)
            ix2, iy2 = min(mq_x2, bx2), min(mq_y2, by2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / (mq_area + bw * bh - inter + 1e-6)
                if iou >= 0.15:
                    continue
            raw_boxes.append((bcx, bcy, bx1, by1, bx2, by2))

    if not raw_boxes:
        return []

    # 按邻台中心聚类, 仅保留跨多帧稳定出现的真实隔壁球台
    cluster_rad = diag * 0.08
    clusters = []  # [[box_tuple, ...], ...]
    for item in raw_boxes:
        placed = False
        for cl in clusters:
            mcx = sum(x[0] for x in cl) / len(cl)
            mcy = sum(x[1] for x in cl) / len(cl)
            if math.hypot(item[0] - mcx, item[1] - mcy) <= cluster_rad:
                cl.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    min_hits = max(2, len(sample_frames) // 30)
    zones = []
    for cl in clusters:
        if len(cl) < min_hits:
            continue
        arr = np.array([[x[2], x[3], x[4], x[5]] for x in cl], dtype=np.float64)
        nx1, ny1, nx2, ny2 = np.median(arr, axis=0).tolist()
        nw = max(10.0, nx2 - nx1)
        nh = max(nw * 0.18, ny2 - ny1)
        # 覆盖邻台桌面及其上方击球空域、两侧挥拍区
        zx1 = max(0.0, nx1 - 0.25 * nw)
        zy1 = max(0.0, ny1 - 1.15 * nh)
        zx2 = min(float(w), nx2 + 0.25 * nw)
        zy2 = min(float(h), ny2 + 0.20 * nh)
        zones.append((zx1, zy1, zx2, zy2))

    if debug and zones:
        log(f"  ◆ 识别到 {len(zones)} 个隔壁球台干扰空域, 已启用邻台球隔离")
    return zones


def _point_in_neighbor_zones(x, y, neighbor_zones, main_quad=None, safe_main_tol=22.0):
    if not neighbor_zones:
        return False
    if main_quad is not None:
        if cv2.pointPolygonTest(main_quad.reshape(-1, 1, 2), (float(x), float(y)), True) >= -safe_main_tol:
            return False
    for zx1, zy1, zx2, zy2 in neighbor_zones:
        if zx1 <= x <= zx2 and zy1 <= y <= zy2:
            return True
    return False


def _point_in_corridor(pt, corridor):
    if corridor is None:
        return True
    return cv2.pointPolygonTest(corridor.reshape(-1, 1, 2),
                                (float(pt[0]), float(pt[1])), False) >= 0


def extract_table_geometry(frames, img_hw, debug=False, persp_blend=0.0,
                           corridor_margin=0.45, corridor_extend=1.2,
                           center_dist_ratio=0.3):
    """
    参数
    ----
    center_dist_ratio : float, default=0.3
        实例中心离画面中心超过 center_dist_ratio × max(w,h) 时丢弃该帧。
        设为 None 或 0 关闭过滤。
    """
    h, w = img_hw
    sample_frames = frames[::10]
    center_dist_max = center_dist_ratio * max(w, h) if center_dist_ratio else None
    all_table_pts, table_quads = [], []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "table", conf_min=0.5, img_hw=img_hw,
                                       center_dist_max=center_dist_max)
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
    if debug:
        log(f"  检测中位数四角(调整前): {quad.tolist()}")

    net_lines = []
    for fr in sample_frames:
        pts = _get_polygon_pts_single(fr, "net", conf_min=0.5, img_hw=img_hw,
                                       center_dist_max=center_dist_max)
        if pts is not None and len(pts) >= 2:
            line = _fit_net_line_single(pts)
            if line is not None:
                net_lines.append(line)
    net_line = _median_net_line(net_lines) if len(net_lines) >= 3 else None

    quad, was_rebuilt = validate_and_refine_quad(quad, table_pts_agg, net_line, img_hw, debug=debug)
    if debug:
        log(f"  关卡1校验后四角({'已重建' if was_rebuilt else '未改动'}): {quad.tolist()}")

    if net_line is not None and not was_rebuilt and not DISABLE_QUAD_REBUILD:
        bisects_ok, _ = _net_bisects_quad(quad, net_line, img_hw)
        if not bisects_ok:
            if debug:
                log("  ⚠️ 球网与相邻边相交, 启动约束重建")
            rebuilt = rebuild_quad_from_constraints(table_pts_agg, net_line, img_hw, debug=debug)
            if rebuilt is not None:
                quad, was_rebuilt = rebuilt, True

    if net_line is not None and not was_rebuilt and not DISABLE_QUAD_REBUILD:
        net_len = float(np.hypot(net_line[1][0] - net_line[0][0], net_line[1][1] - net_line[0][1]))
        q = order_quad(quad).astype(np.float64)
        edge_lens = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        side_a = (edge_lens[0] + edge_lens[2]) * 0.5
        side_b = (edge_lens[1] + edge_lens[3]) * 0.5
        if min(side_a, side_b) < net_len * 0.5:
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
    long_edge_idx = None          # 脚本判定的长边（网线穿过的两条边）索引, 仅用于 debug 输出
    if net_line is not None:
        inter_idx = _net_quad_intersections(quad, net_line, img_hw)
        if len(inter_idx) == 2 and not _edges_are_adjacent(inter_idx[0], inter_idx[1]):
            found_end_edges = True
            long_edge_idx = inter_idx
            non_inter_idx = [i for i in range(4) if i not in inter_idx]
            near_a, near_b, far_a, far_b = _pick_near_far([edge_info[i] for i in non_inter_idx])
        elif debug:
            log("  ⚠️ 球网仍与相邻边相交, 回退到角度法判定端边")
    if not found_end_edges:
        # 端边必为四边形的一组对边 (0,2) 或 (1,3), 严禁按单边排序选出相邻两条边!
        if nd is not None:
            ang_02 = 0.5 * (edge_info[0][2] + edge_info[2][2])
            ang_13 = 0.5 * (edge_info[1][2] + edge_info[3][2])
            pair_idx = (0, 2) if ang_02 <= ang_13 else (1, 3)
        else:
            len_02 = 0.5 * (edge_info[0][1] + edge_info[2][1])
            len_13 = 0.5 * (edge_info[1][1] + edge_info[3][1])
            pair_idx = (0, 2) if len_02 <= len_13 else (1, 3)
        near_a, near_b, far_a, far_b = _pick_near_far([edge_info[pair_idx[0]], edge_info[pair_idx[1]]])

    if far_a[0] > far_b[0]:
        far_left, far_right = far_b, far_a
    else:
        far_left, far_right = far_a, far_b
    if near_a[0] > near_b[0]:
        near_left, near_right = near_b, near_a
    else:
        near_left, near_right = near_a, near_b
    # 防止平视侧拍梯形球台因左右短边反向倾斜导致独立按 x 排序把两条长边扭成对角线沙漏(bowtie)
    if _segments_intersect(far_left, near_left, far_right, near_right):
        near_left, near_right = near_right, near_left

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
    neighbor_zones = _extract_neighbor_table_zones(sample_frames, quad, img_hw, debug=debug)
    if debug:
        log(f"  球台四角: {quad.tolist()}")
        log(f"  球网线: {net_line}")
        if long_edge_idx:
            long_desc = " | ".join(
                f"e{i} ({edges[i][0][0]:.0f},{edges[i][0][1]:.0f})-"
                f"({edges[i][1][0]:.0f},{edges[i][1][1]:.0f})" for i in long_edge_idx)
            log(f"  长边: {long_desc}")
        log(f"  短边: near ({near_left[0]:.0f},{near_left[1]:.0f})-"
            f"({near_right[0]:.0f},{near_right[1]:.0f})  "
            f"far ({far_left[0]:.0f},{far_left[1]:.0f})-"
            f"({far_right[0]:.0f},{far_right[1]:.0f})")

    return {"quad": quad, "net_line": net_line, "homography": H,
            "net_side_sign": net_side_sign, "corridor": corridor, "rebuilt": was_rebuilt,
            "neighbor_zones": neighbor_zones,
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
        # 网侧镜像校正暂时注释 (排查落点, 观察效果后决定恢复/删除)
        # if net_line is not None and net_side_sign is not None:
        #     (nx1, ny1), (nx2, ny2) = net_line
        #     cross = (nx2 - nx1) * (pt[1] - ny1) - (ny2 - ny1) * (pt[0] - nx1)
        #     is_near_side = (cross * net_side_sign > 0)
        #     half_y = TABLE_LEN_M * 1000 / 2.0
        #     if is_near_side and y_mm < half_y:
        #         y_mm = half_y + (half_y - y_mm)
        #     elif (not is_near_side) and y_mm > half_y:
        #         y_mm = half_y - (y_mm - half_y)
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
            cands.append((cx, cy, det.get("confidence", 0.0), b))
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


def filter_static_ball_candidates(ball_cands, fps, frame_w, frame_h,
                                  min_static_count=10, debug=False):
    if not ball_cands:
        return ball_cands
    window_frames = max(300, int(round(fps * 15.0)))
    step_frames = max(150, window_frames // 2)
    img_diag = math.hypot(frame_w, frame_h)
    # 兼容 FP16 / 缩放还原后静止假球 2~4px 的检测框中心微抖
    static_threshold = max(img_diag * 0.002, 3.5)
    cell = max(static_threshold, 1.0)
    flat = []
    for fidx, cands in ball_cands:
        for di, (cx, cy, _c, _b) in enumerate(cands):
            flat.append((fidx, di, cx, cy))
    if not flat:
        return ball_cands
    flat.sort(key=lambda x: x[0])
    flat_frames = [r[0] for r in flat]
    removed_set = set()
    min_f, max_f = flat_frames[0], flat_frames[-1]
    win_start = min_f
    while win_start <= max_f:
        win_end = win_start + window_frames
        lo = bisect.bisect_left(flat_frames, win_start)
        hi = bisect.bisect_right(flat_frames, win_end)
        win = flat[lo:hi]
        win_start += step_frames
        if len(win) < min_static_count:
            continue
        grid = defaultdict(list)
        for rec in win:
            _f, _d, cx, cy = rec
            grid[(int(cx / cell), int(cy / cell))].append(rec)
        handled = set()
        for (gx, gy), _bucket in grid.items():
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
            distinct_frames = {r[0] for r in close}
            if len(distinct_frames) >= min_static_count:
                for _f, _d, _cx, _cy in close:
                    removed_set.add((_f, _d))
                handled.add((gx, gy))
        if win_end >= max_f:
            break
    if not removed_set:
        return ball_cands
    filtered = []
    for fidx, cands in ball_cands:
        remaining = [det for di, det in enumerate(cands) if (fidx, di) not in removed_set]
        filtered.append((fidx, remaining))
    if debug:
        log(f"  ★ 静止球过滤(滑动窗口网格哈希): 剔除 {len(removed_set)} 个静止球坐标")
    return filtered


def filter_ball_by_corridor(ball_cands, corridor, max_drop_ratio=0.55,
                            geo_rebuilt=False, debug=False):
    if corridor is None or not ball_cands:
        return ball_cands
    n_before = n_after = 0
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
        log(f"  ⚠️ 走廊过滤将剔除 {drop_ratio*100:.0f}% 的球 (> 上限 "
            f"{effective_max*100:.0f}%), 判定几何不可信 → 跳过走廊过滤")
        return ball_cands
    if debug:
        log(f"  ★ 长边走廊过滤: {n_before} → {n_after} 个球检测")
    return filtered


def cluster_ball_by_court(ball_cands, frame_w, frame_h, debug=False):
    if not ball_cands:
        return ball_cands
    pts, pt_refs = [], []
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


def estimate_gravity_px_per_frame2(table_geo, track_dict, fps, frame_w, frame_h, debug=False):
    """结合球台透视几何与干净轨迹弧段标定图像竖直向下重力加速度 g_frame (px/帧²)。"""
    diag = math.hypot(frame_w, frame_h)
    fps_sq = max(float(fps) * float(fps), 1.0)
    if table_geo is not None and table_geo.get("near_left") is not None:
        fl = np.asarray(table_geo["far_left"], dtype=np.float64)
        fr = np.asarray(table_geo["far_right"], dtype=np.float64)
        nl = np.asarray(table_geo["near_left"], dtype=np.float64)
        nr = np.asarray(table_geo["near_right"], dtype=np.float64)
        ew = 0.5 * ((nr - nl) + (fr - fl))
        el = 0.5 * ((nl - fl) + (nr - fr))
        px = float(ew[0]) / TABLE_WID_M
        qx = float(el[0]) / TABLE_LEN_M
        s_horiz = math.hypot(px, qx)
        s_edge = max(float(np.linalg.norm(ew)) / TABLE_WID_M,
                     float(np.linalg.norm(el)) / TABLE_LEN_M)
        s_px_m = _clamp(max(s_horiz, s_edge), diag * 0.05, diag * 0.65)
        py = float(ew[1]) / TABLE_WID_M
        qy = float(el[1]) / TABLE_LEN_M
        sin_pitch = _clamp(math.hypot(py, qy) / (s_px_m + 1e-6), 0.0, 0.80)
        cos_pitch = math.sqrt(max(0.12, 1.0 - sin_pitch * sin_pitch))
        g_geo = 9.81 * s_px_m * cos_pitch / fps_sq
    else:
        s_px_m = diag * 0.16
        g_geo = 9.81 * s_px_m * 0.92 / fps_sq

    g_geo = _clamp(g_geo, 0.04, max(0.5, diag * 0.025))

    # 从实测单向无弹跳弧段中提取中位数重力做自适应校准
    a_samples = []
    if isinstance(track_dict, dict) and len(track_dict) >= 10:
        keys = sorted(track_dict.keys())
        runs, cur = [], [keys[0]]
        for k in keys[1:]:
            if k - cur[-1] <= 2:
                cur.append(k)
            else:
                if len(cur) >= 6:
                    runs.append(cur)
                cur = [k]
        if len(cur) >= 6:
            runs.append(cur)

        win = max(6, min(12, int(round(fps * 0.20))))
        step = max(3, win // 2)
        rms_lim = max(4.5, diag * 0.0035)
        for r in runs:
            if len(r) < win:
                continue
            rf = np.array(r, dtype=np.float64)
            rx = np.array([track_dict[k][0] for k in r], dtype=np.float64)
            ry = np.array([track_dict[k][1] for k in r], dtype=np.float64)
            for s in range(0, len(r) - win + 1, step):
                wf = rf[s:s + win]
                wx = rx[s:s + win]
                wy = ry[s:s + win]
                dx = np.diff(wx)
                if not ((dx > 0.5).all() or (dx < -0.5).all()):
                    continue
                # 排除窗口内部含台面弹跳极值点
                dy = np.diff(wy)
                if any(dy[j] > 1.0 and dy[j + 1] < -1.0 for j in range(len(dy) - 1)):
                    continue
                tau = wf - wf.mean()
                try:
                    coef = np.polyfit(tau, wy, 2)
                except Exception:
                    continue
                a_emp = 2.0 * float(coef[0])
                rms = float(np.sqrt(np.mean((wy - np.polyval(coef, tau)) ** 2)))
                if rms <= rms_lim and (0.18 * g_geo <= a_emp <= 3.0 * g_geo):
                    a_samples.append(a_emp)

    if len(a_samples) >= 3:
        med_a = float(np.median(a_samples))
        g_final = (0.4 * g_geo + 0.6 * med_a) if table_geo is not None else med_a
    else:
        g_final = g_geo

    g_final = _clamp(g_final, 0.04, max(0.5, diag * 0.025))
    if debug:
        emp_note = (f", 实测弧段样本={len(a_samples)} 中位={float(np.median(a_samples)):.3f}"
                    if len(a_samples) >= 3 else f", 实测样本不足({len(a_samples)})")
        log(f"  ◆ 重力标定: 比例尺≈{s_px_m:.0f}px/m, 几何g={g_geo:.3f}px/帧²"
            f"{emp_note} → 采用 g={g_final:.3f}px/帧²")
    return g_final, s_px_m


def _fit_constrained_parabola(fs, xs, ys, weights=None,
                              ay_min=0.0, ay_max=5.0, ax_max=2.0):
    """闭式加权有界抛物线拟合:
        x(τ) = cx + vx·τ + ½·ax·τ²,  ax ∈ [-ax_max, +ax_max]
        y(τ) = cy + vy·τ + ½·ay·τ²,  ay ∈ [ay_min,  ay_max]
    目标函数关于 (a, v, c) 严格凸二次: 无约束最优越界时, 精确全局约束最优必落在边界!
    返回 dict(t0, ax, vx, cx, ay, vy, cy)。
    """
    fs = np.asarray(fs, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    t0 = float(fs.mean())
    tau = fs - t0
    sw = np.sqrt(np.clip(weights, 1e-4, None)) if weights is not None else None

    def _solve_1d(vals, a_lo, a_hi):
        if len(tau) >= 3 and float(np.ptp(tau)) >= 1.8:
            M = np.column_stack([0.5 * tau * tau, tau, np.ones_like(tau)])
            try:
                if sw is not None:
                    theta, *_ = np.linalg.lstsq(M * sw[:, None], vals * sw, rcond=None)
                else:
                    theta, *_ = np.linalg.lstsq(M, vals, rcond=None)
                a_hat, v_hat, c_hat = float(theta[0]), float(theta[1]), float(theta[2])
                if a_lo <= a_hat <= a_hi and math.isfinite(a_hat):
                    return a_hat, v_hat, c_hat
                a_c = _clamp(a_hat if math.isfinite(a_hat) else 0.0, a_lo, a_hi)
            except Exception:
                a_c = _clamp(0.0, a_lo, a_hi)
        else:
            a_c = _clamp(0.5 * (a_lo + a_hi) if a_lo > 0 else 0.0, a_lo, a_hi)

        rem = vals - 0.5 * a_c * tau * tau
        if len(tau) >= 2 and float(np.ptp(tau)) >= 0.5:
            M2 = np.column_stack([tau, np.ones_like(tau)])
            try:
                if sw is not None:
                    th2, *_ = np.linalg.lstsq(M2 * sw[:, None], rem * sw, rcond=None)
                else:
                    th2, *_ = np.linalg.lstsq(M2, rem, rcond=None)
                return a_c, float(th2[0]), float(th2[1])
            except Exception:
                pass
        return a_c, 0.0, float(np.mean(rem))

    ax, vx, cx = _solve_1d(xs, -abs(ax_max), abs(ax_max))
    ay, vy, cy = _solve_1d(ys, ay_min, ay_max)
    return {"t0": t0, "ax": ax, "vx": vx, "cx": cx, "ay": ay, "vy": vy, "cy": cy}


def _eval_parabola(model, f_eval):
    dt = np.asarray(f_eval, dtype=np.float64) - model["t0"]
    xh = model["cx"] + model["vx"] * dt + 0.5 * model["ax"] * dt * dt
    yh = model["cy"] + model["vy"] * dt + 0.5 * model["ay"] * dt * dt
    return xh, yh


def build_main_ball_track(ball_cands, fps, frame_w, frame_h, table_geo=None, debug=False,
                          max_jump_ratio=0.05, match_algo="auto"):
    diag = math.hypot(frame_w, frame_h)
    max_jump = max_jump_ratio * diag
    max_gap_frames = max(1, int(fps * 0.5))
    g_prior, _ = estimate_gravity_px_per_frame2(table_geo, None, fps, frame_w, frame_h, debug=False)
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
        f1, x1, y1 = pts[-1][0], pts[-1][1], pts[-1][2]
        if len(pts) == 1:
            return x1, y1, None, None
        f0, x0, y0 = pts[-2][0], pts[-2][1], pts[-2][2]
        df1 = max(1, f1 - f0)
        # 弦速度对应区间中点时刻, 加上 ½·g·df1 还原 f1 时刻瞬时竖直速度
        vx = (x1 - x0) / df1
        vy = (y1 - y0) / df1 + 0.5 * g_prior * df1
        if len(pts) >= 3:
            fp, xp, yp = pts[-3][0], pts[-3][1], pts[-3][2]
            df0 = max(1, f0 - fp)
            vx0 = (x0 - xp) / df0
            vy0_at_f1 = (y0 - yp) / df0 + g_prior * (0.5 * df0 + df1)
            # 同向且未发生台面弹跳翻转时, 融合前一步速度抑制单帧检测框抖动
            if vx0 * vx > 0 and not (vy0_at_f1 > 1.5 and vy < -1.0) and df0 <= 4 and df1 <= 4:
                vx = 0.65 * vx + 0.35 * vx0
                vy = 0.65 * vy + 0.35 * vy0_at_f1
        dt = fidx - f1
        decay = 0.88 ** max(0, dt - 1)
        dt_eff = dt * decay
        px = x1 + vx * dt_eff
        py = y1 + vy * dt_eff + 0.5 * g_prior * dt_eff * dt_eff
        return px, py, vx, vy

    main_quad = table_geo.get("quad") if table_geo else None
    neighbor_zones = (table_geo.get("neighbor_zones") or []) if table_geo else []
    if main_quad is not None:
        mq_x1, mq_y1 = float(main_quad[:, 0].min()), float(main_quad[:, 1].min())
        mq_x2, mq_y2 = float(main_quad[:, 0].max()), float(main_quad[:, 1].max())
        mq_w = max(1.0, mq_x2 - mq_x1)
        mq_h = max(1.0, mq_y2 - mq_y1)
    else:
        mq_x1 = mq_y1 = mq_x2 = mq_y2 = mq_w = mq_h = 0.0

    def _bbox_minor(b):
        if not b or len(b) != 4:
            return None
        return max(1.0, min(abs(float(b[2]) - float(b[0])), abs(float(b[3]) - float(b[1]))))

    def _dist_main_quad(x, y):
        if main_quad is None:
            return 0.0
        # pointPolygonTest >= 0 在四边形内; 返回到主球台的距离 (>=0, 台内为 0)
        sd = cv2.pointPolygonTest(main_quad.reshape(-1, 1, 2), (float(x), float(y)), True)
        return 0.0 if sd >= 0.0 else -float(sd)

    def assoc_cost(tr, pred_info, tol, gap, cx, cy, cand_bbox=None):
        px, py, vx, vy = pred_info
        pts = tr["pts"]
        x1, y1 = pts[-1][1], pts[-1][2]
        # 限制跨缺口的绝对位移上限, 防止速度外推在长缺口上瞬移抓到远端/隔壁台噪点
        d_last = math.hypot(cx - x1, cy - y1)
        max_abs_jump = max_jump * (1.0 + 0.30 * min(gap - 1, 3))
        if d_last > max_abs_jump:
            return None
        d_pred = math.hypot(cx - px, cy - py)
        if d_pred >= tol:
            return None

        # 邻台空域隔离: 从非邻台轨迹跨入邻台空域时, 除非严格落在当前前景抛物线预测点上(如前景高球掠过背景台前方), 否则拒绝串联!
        if neighbor_zones and _point_in_neighbor_zones(cx, cy, neighbor_zones, main_quad):
            prev_in_nb = _point_in_neighbor_zones(x1, y1, neighbor_zones, main_quad)
            if (not prev_in_nb) or gap >= 2:
                if d_pred > max(18.0, diag * 0.012):
                    return None

        # 反重力上窜与景深突变拦截(含 gap==1 单帧遮挡瞬间!):
        # 防止前景球擦网/被遮挡仅 1 帧时, 活跃轨迹直接跳上 100+px 外的背景球台球而被劫持跑偏
        if len(pts) >= 2 and gap >= 1:
            if vy is not None and vy >= -1.0 and cy < y1 - max(26.0, diag * 0.020):
                on_bounce_surface = (main_quad is not None and _dist_main_quad(x1, y1) <= max(24.0, 0.15 * mq_h))
                if not on_bounce_surface and d_pred > max(28.0, diag * 0.020):
                    return None
            # 景深视直径突变拦截: 偏离预测较大且球框短轴尺寸骤缩(远端背景台小球)或骤增(近景邻台大球)时拒绝串联
            if cand_bbox is not None and d_pred > max(24.0, diag * 0.016):
                recent_b = [_bbox_minor(p[4]) for p in pts[-4:] if len(p) > 4 and p[4]]
                recent_b = [v for v in recent_b if v is not None]
                cb = _bbox_minor(cand_bbox)
                if recent_b and cb is not None:
                    med_b = float(np.median(recent_b))
                    ratio_b = cb / max(med_b, 1.0)
                    if ratio_b < 0.54 or ratio_b > 1.90:
                        return None

        cost_val = d_pred
        # 已建立明确横向速度时, 对反向跳变施加软惩罚(同帧有真球时优先选顺向抛物线点, 无真球时仍允许真实击球折返)
        if vx is not None and abs(vx) > 3.0:
            dx = cx - x1
            if dx * vx < 0 and abs(dx) > 8.0:
                cost_val += 0.30 * abs(dx)
        return cost_val

    _BIG = 1e9

    def greedy_assign(fidx, active, cands, used, preds, tols, gaps):
        for i, tr in enumerate(active):
            best_j, best_c = -1, _BIG
            for j, (cx, cy, _c, b) in enumerate(cands):
                if used[j]:
                    continue
                cval = assoc_cost(tr, preds[i], tols[i], gaps[i], cx, cy, b)
                if cval is not None and cval < best_c:
                    best_c, best_j = cval, j
            if best_j >= 0:
                cx, cy, conf, b = cands[best_j]
                tr["pts"].append((fidx, cx, cy, conf, b))
                used[best_j] = True

    for fidx, cands in ball_cands:
        used = [False] * len(cands)
        active = [tr for tr in tracks if 0 < (fidx - tr["pts"][-1][0]) <= max_gap_frames]
        if active and cands:
            preds, tols, gaps = [], [], []
            for tr in active:
                gap = fidx - tr["pts"][-1][0]
                preds.append(predict(tr, fidx))
                if len(tr["pts"]) == 1:
                    tols.append(max_jump * (0.90 if gap == 1 else 0.65))
                elif gap == 1:
                    tols.append(max_jump * 0.90)
                else:
                    # 已建立速度矢量的轨迹在多帧缺口上收紧预测门限, 防止球死网后跨空抓取隔壁台球
                    last_x, last_y = tr["pts"][-1][1], tr["pts"][-1][2]
                    near_endline = (main_quad is None or _dist_main_quad(last_x, last_y) <= max(45.0, 0.35 * mq_w))
                    base_pred_tol = max(42.0, diag * (0.042 if (near_endline and gap <= 3) else 0.032))
                    tols.append(min(max_jump * 0.85, base_pred_tol * (1.0 + 0.14 * (gap - 1))))
                gaps.append(gap)
            done = False
            if algo == "hungarian":
                n_a, n_c = len(active), len(cands)
                cost = np.full((n_a, n_c), _BIG, dtype=np.float64)
                for i, tr in enumerate(active):
                    for j, (cx, cy, _c, b) in enumerate(cands):
                        cval = assoc_cost(tr, preds[i], tols[i], gaps[i], cx, cy, b)
                        if cval is not None:
                            cost[i, j] = cval
                try:
                    row_ind, col_ind = _linear_sum_assignment(cost)
                except Exception as e:
                    if debug:
                        log(f"  [match] linear_sum_assignment 失败 ({e}), 当帧回退 greedy")
                else:
                    for i, j in zip(row_ind, col_ind):
                        if cost[i, j] >= _BIG:
                            continue
                        cx, cy, conf, b = cands[j]
                        active[i]["pts"].append((fidx, cx, cy, conf, b))
                        used[j] = True
                    done = True
            if not done:
                greedy_assign(fidx, active, cands, used, preds, tols, gaps)
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
        count, prev_side, prev_pt = 0, None, None
        for p in pts:
            side = ((nx2 - nx1) * (p[2] - ny1) - (ny2 - ny1) * (p[1] - nx1)) > 0
            if prev_side is not None and side != prev_side and prev_pt is not None:
                mx, my = 0.5 * (prev_pt[1] + p[1]), 0.5 * (prev_pt[2] + p[2])
                # 仅统计主球台有效上空范围内的过网(排除球网直线无限延伸穿过背景球台造成的伪过网加分)
                valid_cross = True
                if main_quad is not None:
                    if not (mq_x1 - 0.30 * mq_w <= mx <= mq_x2 + 0.30 * mq_w
                            and mq_y1 - 1.60 * mq_h <= my <= mq_y2 + 0.25 * mq_h):
                        valid_cross = False
                    elif _point_in_neighbor_zones(mx, my, neighbor_zones, main_quad):
                        valid_cross = False
                if valid_cross:
                    count += 1
            prev_side, prev_pt = side, p
        return count

    def _track_main_table_stats(pts):
        """返回 (min_dist_to_main_quad, frac_in_neighbor_zones, med_bbox_minor)。"""
        if not pts:
            return 1e9, 0.0, None
        min_d = min(_dist_main_quad(p[1], p[2]) for p in pts) if main_quad is not None else 0.0
        nb_cnt = (sum(1 for p in pts if _point_in_neighbor_zones(p[1], p[2], neighbor_zones, main_quad))
                  if neighbor_zones else 0)
        frac_nb = nb_cnt / len(pts)
        bmins = [_bbox_minor(p[4]) for p in pts if len(p) > 4 and p[4]]
        bmins = [v for v in bmins if v is not None]
        med_b = float(np.median(bmins)) if bmins else None
        return min_d, frac_nb, med_b

    def score(tr):
        pts = tr["pts"]
        n = len(pts)
        if n < 3:
            return 0.0
        min_d, frac_nb, _ = _track_main_table_stats(pts)
        # 绝大部分点落在隔壁球台空域且从未接触主球台的轨迹直接判 0 分
        if main_quad is not None:
            if frac_nb >= 0.45 and min_d > max(20.0, 0.12 * mq_h):
                return 0.0
            # 全程远离主球台面(如悬在挡板后方背景台或前景角落邻台)的孤立轨迹大幅降权/淘汰
            if min_d > max(55.0, 0.55 * mq_h):
                return 0.0
        span = pts[-1][0] - pts[0][0] + 1
        coverage = n / span
        motion = sum(math.hypot(pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2])
                     for i in range(1, n))
        avg_conf = float(np.mean([p[3] for p in pts]))
        base = n * (0.4 + coverage) * (1 + math.log1p(motion)) * (0.5 + avg_conf)
        net_boost = 1.0 + min(2.0, net_cross_count(pts) * 0.5)
        # 紧贴/落入主球台面的轨迹给予主台亲和加成, 确保 anchor 必为主球台回合
        table_affinity = 1.25 if (main_quad is not None and min_d <= max(18.0, 0.10 * mq_h)) else 1.0
        return base * net_boost * table_affinity

    tracks = [tr for tr in tracks if score(tr) >= 6.0]
    if not tracks:
        return {}
    tracks.sort(key=score, reverse=True)
    n_before_filter = len(tracks)
    if len(tracks) >= 2 and table_geo is not None:
        anchor = tracks[0]
        anchor_pts = np.array([[p[1], p[2]] for p in anchor["pts"]])
        ax_min, ay_min = anchor_pts.min(axis=0)
        ax_max, ay_max = anchor_pts.max(axis=0)
        anchor_area = max((ax_max - ax_min) * (ay_max - ay_min), 1.0)
        _, _, anchor_bmin = _track_main_table_stats(anchor["pts"])
        filtered = [anchor]
        n_nb_rejected = 0
        for tr in tracks[1:]:
            tr_pts = np.array([[p[1], p[2]] for p in tr["pts"]])
            tx_min, ty_min = tr_pts.min(axis=0)
            tx_max, ty_max = tr_pts.max(axis=0)
            tr_area = (tx_max - tx_min) * (ty_max - ty_min)
            if tr_area <= 0:
                continue
            min_d, frac_nb, tr_bmin = _track_main_table_stats(tr["pts"])
            if frac_nb >= 0.40 and min_d > max(18.0, 0.10 * mq_h):
                n_nb_rejected += 1
                continue
            # 景深球体尺寸比校验: 排除远端背景小台球(<0.55x)与超近景邻台大球(>1.85x)
            if anchor_bmin and tr_bmin:
                sz_ratio = tr_bmin / max(anchor_bmin, 1.0)
                if (sz_ratio < 0.55 or sz_ratio > 1.85) and min_d > max(15.0, 0.08 * mq_h):
                    n_nb_rejected += 1
                    continue
            ix_min, iy_min = max(ax_min, tx_min), max(ay_min, ty_min)
            ix_max, iy_max = min(ax_max, tx_max), min(ay_max, ty_max)
            if ix_min < ix_max and iy_min < iy_max:
                inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                iou = inter_area / (anchor_area + tr_area - inter_area)
                # 主判据恢复严格 IoU>=0.15(防止背景台小包络框落入 anchor 上空大框被误放行);
                # 短快球豁免仅限真实触达主球台桌面(min_d<=20px)且跨度合理的片段
                touches_main_surface = (
                    min_d <= max(20.0, 0.12 * mq_h)
                    and len(tr["pts"]) >= 4
                    and (tx_max - tx_min) >= 0.18 * mq_w
                )
                if iou >= 0.15 or touches_main_surface:
                    filtered.append(tr)
                else:
                    n_nb_rejected += 1
        tracks = filtered
        if debug:
            log(f"  空间共域+主台亲和过滤: {n_before_filter} → {len(tracks)} 条轨迹 "
                f"(剔除邻台/非共域 {n_nb_rejected})")

    # 时序排他合并: 按轨迹得分由高到低认领时间区间, 禁止并发误检轨迹与主球轨迹逐帧交替穿插造成折线
    score_min = 6.0
    main, n_used = {}, 0
    claimed_spans = []  # [(start_f, end_f), ...]

    def _in_claimed(f, margin=1):
        for s_f, e_f in claimed_spans:
            if (s_f - margin) <= f <= (e_f + margin):
                return True
        return False

    for tr in tracks:
        if score(tr) < score_min:
            continue
        free_pts = [p for p in tr["pts"] if not _in_claimed(p[0], margin=1)]
        if len(free_pts) < 3:
            continue
        # 若被已认领区间截断成多段, 仅保留点数 >= 3 的有效子段
        sub_runs, cur_sub = [], [free_pts[0]]
        for p in free_pts[1:]:
            if p[0] - cur_sub[-1][0] <= max_gap_frames:
                cur_sub.append(p)
            else:
                if len(cur_sub) >= 3:
                    sub_runs.append(cur_sub)
                cur_sub = [p]
        if len(cur_sub) >= 3:
            sub_runs.append(cur_sub)
        if not sub_runs:
            continue
        n_used += 1
        for sub in sub_runs:
            # 校验与紧邻已采用点的边界跳变是否物理相容, 不相容则裁掉边界冲突帧留出干净缺口
            while len(sub) >= 3:
                f_head, x_head, y_head = sub[0][0], sub[0][1], sub[0][2]
                conflict = False
                for df_chk in range(1, max_gap_frames + 1):
                    prev_pt = main.get(f_head - df_chk)
                    if prev_pt is not None:
                        lim = max_jump * (1.0 + 0.12 * (df_chk - 1))
                        if math.hypot(x_head - prev_pt[0], y_head - prev_pt[1]) > lim:
                            conflict = True
                        break
                if conflict and df_chk <= int(fps * 0.3):
                    sub.pop(0)
                else:
                    break
            while len(sub) >= 3:
                f_tail, x_tail, y_tail = sub[-1][0], sub[-1][1], sub[-1][2]
                conflict = False
                for df_chk in range(1, max_gap_frames + 1):
                    next_pt = main.get(f_tail + df_chk)
                    if next_pt is not None:
                        lim = max_jump * (1.0 + 0.12 * (df_chk - 1))
                        if math.hypot(next_pt[0] - x_tail, next_pt[1] - y_tail) > lim:
                            conflict = True
                        break
                if conflict and df_chk <= int(fps * 0.3):
                    sub.pop()
                else:
                    break
            if len(sub) < 3:
                continue
            claimed_spans.append((sub[0][0], sub[-1][0]))
            for (fidx, x, y, conf, b) in sub:
                main[fidx] = (x, y, conf, b)

    if debug:
        log(f"  候选轨迹数={n_before_filter}, 采用={n_used}, 去重后点数={len(main)}")
    return main


# =====================================================================
#  3b) 物理抛物线离群误检净化
# =====================================================================
def clean_ball_track_parabolic(main_track, fps, frame_w, frame_h, g_frame,
                               table_geo=None, max_jump_ratio=0.10, debug=False):
    """剔除主轨迹中破坏重力抛物线的误检折线点 (1~3 帧离群尖刺、双反转折角、首尾误检折钩)。

    核心判据 (Leave-Block-Out 抛物线桥接检验):
      对可疑块 B (长度 1~3 帧), 取其前后紧邻支撑点集 P_pre ∪ P_post 拟合带重力约束的抛物线。
      - 若 P_pre ∪ P_post 自身高度吻合同一条自由飞行抛物线 (残差 <= tol_support),
        而 B 中各点显著偏离该抛物线 (残差 > tol_outlier), 则 B 必为空中误检 (如球拍/白鞋/球网高光)!
      - 真实台面弹跳 (下降转上升, 需强反重力才能单弧拟合) 与真实击球折返 (vx 反向)
        均无法用单条重力抛物线同时拟合 P_pre ∪ P_post, 因此真实弹跳/击球顶点 100% 免疫误删!
    """
    if not main_track or len(main_track) < 3:
        return dict(main_track) if main_track else {}

    diag = math.hypot(frame_w, frame_h)
    max_assoc_gap = max(2, int(round(fps * 0.50)))
    short_gap = max(4, int(round(fps * 0.20)))
    bridge_max_gap = max(6, int(round(fps * 0.35)))
    tol_support = max(10.0, diag * 0.007)
    tol_outlier = max(18.0, diag * 0.013)
    tol_dx_rev = max(9.0, diag * 0.006)
    quad = table_geo.get("quad") if table_geo else None
    neighbor_zones = (table_geo.get("neighbor_zones") or []) if table_geo else []
    mq_h = (max(1.0, float(quad[:, 1].max() - quad[:, 1].min()))
            if quad is not None else 0.0)

    def _near_table(x, y, margin_px):
        if quad is None:
            return True
        return cv2.pointPolygonTest(quad.reshape(-1, 1, 2),
                                    (float(x), float(y)), True) >= -margin_px

    def _split_switched_subsegs(seg_frames):
        """时空交汇检验 (δ_meet): 若同一轨迹段在缺口/大跳变两侧的子弧在 [f_i, f_{i+1}] 任意时刻均无法在空间相交,
        说明发生了跨球串轨(典型如前景球死网消失后吸咐到背景隔壁台正在打的球), 立即在此处切断!
        """
        m_s = len(seg_frames)
        if m_s < 6:
            return [seg_frames]
        s_f = np.array(seg_frames, dtype=np.float64)
        s_x = np.array([main_track[k][0] for k in seg_frames], dtype=np.float64)
        s_y = np.array([main_track[k][1] for k in seg_frames], dtype=np.float64)
        s_c = np.array([main_track[k][2] for k in seg_frames], dtype=np.float64)
        cut_after = []
        for idx in range(2, m_s - 3):
            df = float(s_f[idx + 1] - s_f[idx])
            step_d = math.hypot(float(s_x[idx + 1] - s_x[idx]), float(s_y[idx + 1] - s_y[idx]))
            if df < 2.0 and step_d < max(45.0, diag * 0.032):
                continue
            # 取左侧最多 4 点与右侧最多 4 点各自拟合局部抛物线
            l_idx = [idx]
            j = idx
            while len(l_idx) < 4 and j - 1 >= 0 and (s_f[j] - s_f[j - 1]) <= short_gap:
                j -= 1
                l_idx.append(j)
            l_idx.reverse()
            r_idx = [idx + 1]
            j = idx + 1
            while len(r_idx) < 4 and j + 1 < m_s and (s_f[j + 1] - s_f[j]) <= short_gap:
                j += 1
                r_idx.append(j)
            if len(l_idx) < 3 or len(r_idx) < 3:
                continue
            m_l = _fit_constrained_parabola(
                s_f[l_idx], s_x[l_idx], s_y[l_idx], weights=np.clip(s_c[l_idx], 0.25, 1.0) ** 2,
                ay_min=-0.25 * g_frame, ay_max=3.2 * g_frame, ax_max=0.85 * g_frame)
            m_r = _fit_constrained_parabola(
                s_f[r_idx], s_x[r_idx], s_y[r_idx], weights=np.clip(s_c[r_idx], 0.25, 1.0) ** 2,
                ay_min=-0.25 * g_frame, ay_max=3.2 * g_frame, ax_max=0.85 * g_frame)
            err_l = float(np.max(np.hypot(s_x[l_idx] - _eval_parabola(m_l, s_f[l_idx])[0],
                                          s_y[l_idx] - _eval_parabola(m_l, s_f[l_idx])[1])))
            err_r = float(np.max(np.hypot(s_x[r_idx] - _eval_parabola(m_r, s_f[r_idx])[0],
                                          s_y[r_idx] - _eval_parabola(m_r, s_f[r_idx])[1])))
            if err_l > tol_support * 1.4 or err_r > tol_support * 1.4:
                continue
            t_grid = np.linspace(float(s_f[idx]), float(s_f[idx + 1]), max(7, int(df) * 2 + 1))
            xl, yl = _eval_parabola(m_l, t_grid)
            xr, yr = _eval_parabola(m_r, t_grid)
            d_meet = float(np.min(np.hypot(xl - xr, yl - yr)))
            in_nb = (neighbor_zones and (
                _point_in_neighbor_zones(s_x[idx], s_y[idx], neighbor_zones, quad)
                or _point_in_neighbor_zones(s_x[idx + 1], s_y[idx + 1], neighbor_zones, quad)))
            meet_lim = (max(26.0, diag * 0.018) + 1.8 * df) if in_nb else (max(38.0, diag * 0.026) + 3.0 * df)
            if d_meet > meet_lim:
                cut_after.append(idx)
        if not cut_after:
            return [seg_frames]
        out_s, prev = [], 0
        for c_i in cut_after:
            out_s.append(seg_frames[prev:c_i + 1])
            prev = c_i + 1
        out_s.append(seg_frames[prev:])
        return out_s

    keys = sorted(main_track.keys())
    raw_segs, cur = [], [keys[0]]
    for k in keys[1:]:
        if k - cur[-1] <= max_assoc_gap:
            cur.append(k)
        else:
            raw_segs.append(cur)
            cur = [k]
    raw_segs.append(cur)

    cleaned = {}
    n_removed_bridge = 0
    n_removed_end = 0
    n_removed_orphan = 0

    for seg in raw_segs:
        cur_seg = list(seg)
        # 多轮迭代: 先裁首尾孤立折钩, 再用双侧(各≥2点)抛物线桥接摘除段内 1~3 帧误检块
        for _pass in range(3):
            changed = False
            # A) 首尾孤立折钩裁剪 (例如球被遮挡后首/尾单帧抓到球员白鞋/护腕/球拍)
            for _end_pass in range(2):
                m = len(cur_seg)
                if m < 5:
                    break
                fs = np.array(cur_seg, dtype=np.float64)
                xs = np.array([main_track[k][0] for k in cur_seg], dtype=np.float64)
                ys = np.array([main_track[k][1] for k in cur_seg], dtype=np.float64)
                cs = np.array([main_track[k][2] for k in cur_seg], dtype=np.float64)
                dropped_end = False

                # 检查末端点 m - 1
                if fs[-2] - fs[-5] <= bridge_max_gap:
                    sup = slice(m - 5, m - 1)
                    model = _fit_constrained_parabola(
                        fs[sup], xs[sup], ys[sup], weights=np.clip(cs[sup], 0.25, 1.0) ** 2,
                        ay_min=-0.25 * g_frame, ay_max=3.2 * g_frame, ax_max=0.85 * g_frame)
                    sxh, syh = _eval_parabola(model, fs[sup])
                    if float(np.max(np.hypot(xs[sup] - sxh, ys[sup] - syh))) <= tol_support:
                        px_t, py_t = _eval_parabola(model, fs[-1])
                        dt_t = max(1.0, float(fs[-1] - fs[-2]))
                        err_t = math.hypot(float(xs[-1] - px_t), float(ys[-1] - py_t))
                        vx_end = model["vx"] + model["ax"] * (float(fs[-2]) - model["t0"])
                        dx_last = float(xs[-1] - xs[-2])
                        dy_dev = float(ys[-1] - py_t)
                        lim_t = max(24.0, diag * 0.017) * (1.0 + 0.15 * (dt_t - 1.0))
                        is_rev = (dx_last * vx_end < 0 and abs(dx_last) > tol_dx_rev)
                        is_dive = (dy_dev > max(22.0, diag * 0.015))
                        is_sky_jump = (dy_dev < -max(32.0, diag * 0.022)
                                       and not _near_table(xs[-2], ys[-2], max(45.0, diag * 0.03)))
                        if err_t > lim_t and (is_rev or is_dive or is_sky_jump):
                            cur_seg.pop()
                            n_removed_end += 1
                            dropped_end = True
                            changed = True

                # 检查首端点 0
                m = len(cur_seg)
                if m >= 5:
                    fs = np.array(cur_seg, dtype=np.float64)
                    if fs[4] - fs[1] <= bridge_max_gap:
                        sup = slice(1, 5)
                        xs = np.array([main_track[k][0] for k in cur_seg], dtype=np.float64)
                        ys = np.array([main_track[k][1] for k in cur_seg], dtype=np.float64)
                        cs = np.array([main_track[k][2] for k in cur_seg], dtype=np.float64)
                        model = _fit_constrained_parabola(
                            fs[sup], xs[sup], ys[sup], weights=np.clip(cs[sup], 0.25, 1.0) ** 2,
                            ay_min=-0.25 * g_frame, ay_max=3.2 * g_frame, ax_max=0.85 * g_frame)
                        sxh, syh = _eval_parabola(model, fs[sup])
                        if float(np.max(np.hypot(xs[sup] - sxh, ys[sup] - syh))) <= tol_support:
                            px_h, py_h = _eval_parabola(model, fs[0])
                            dt_h = max(1.0, float(fs[1] - fs[0]))
                            err_h = math.hypot(float(xs[0] - px_h), float(ys[0] - py_h))
                            vx_beg = model["vx"] + model["ax"] * (float(fs[1]) - model["t0"])
                            dx_first = float(xs[1] - xs[0])
                            dy_dev = float(ys[0] - py_h)
                            lim_h = max(24.0, diag * 0.017) * (1.0 + 0.15 * (dt_h - 1.0))
                            is_rev = (dx_first * vx_beg < 0 and abs(dx_first) > tol_dx_rev)
                            is_dive = (dy_dev > max(22.0, diag * 0.015))
                            is_sky_jump = (dy_dev < -max(32.0, diag * 0.022)
                                           and not _near_table(xs[1], ys[1], max(45.0, diag * 0.03)))
                            if err_h > lim_h and (is_rev or is_dive or is_sky_jump):
                                cur_seg.pop(0)
                                n_removed_end += 1
                                dropped_end = True
                                changed = True

                if not dropped_end:
                    break

            # B) 段内 Leave-Block-Out 双侧支撑抛物线桥接检验 (严格要求左右两侧各 ≥2 个支撑点)
            m = len(cur_seg)
            if m < 5:
                break
            fs = np.array(cur_seg, dtype=np.float64)
            xs = np.array([main_track[k][0] for k in cur_seg], dtype=np.float64)
            ys = np.array([main_track[k][1] for k in cur_seg], dtype=np.float64)
            cs = np.array([main_track[k][2] for k in cur_seg], dtype=np.float64)

            candidates = []  # (score, start_idx, end_idx_excl)
            for w in (1, 2, 3):
                if m < w + 4:
                    continue
                for i in range(2, m - w - 1):
                    # 收集左侧紧邻支撑点 (2~4 个)
                    if fs[i] - fs[i - 1] > bridge_max_gap:
                        continue
                    pre_idx = [i - 1]
                    j = i - 1
                    while len(pre_idx) < 4 and j - 1 >= 0:
                        if fs[j] - fs[j - 1] > short_gap:
                            break
                        j -= 1
                        pre_idx.append(j)
                    if len(pre_idx) < 2:
                        continue
                    pre_idx.reverse()

                    # 收集右侧紧邻支撑点 (2~4 个)
                    r0 = i + w
                    if fs[r0] - fs[i + w - 1] > bridge_max_gap:
                        continue
                    if fs[r0] - fs[pre_idx[-1]] > bridge_max_gap + 2:
                        continue
                    post_idx = [r0]
                    j = r0
                    while len(post_idx) < 4 and j + 1 < m:
                        if fs[j + 1] - fs[j] > short_gap:
                            break
                        j += 1
                        post_idx.append(j)
                    if len(post_idx) < 2:
                        continue

                    sup_idx = pre_idx + post_idx
                    sf = fs[sup_idx]
                    sx = xs[sup_idx]
                    sy = ys[sup_idx]
                    sw = np.clip(cs[sup_idx], 0.25, 1.0) ** 2
                    model = _fit_constrained_parabola(
                        sf, sx, sy, weights=sw,
                        ay_min=-0.25 * g_frame, ay_max=3.2 * g_frame,
                        ax_max=0.85 * g_frame)
                    sxh, syh = _eval_parabola(model, sf)
                    err_s = np.hypot(sx - sxh, sy - syh)
                    sup_max = float(err_s.max())
                    sup_rms = float(np.sqrt(np.mean(err_s ** 2)))
                    if sup_max > tol_support:
                        continue

                    blk_slice = slice(i, i + w)
                    bxh, byh = _eval_parabola(model, fs[blk_slice])
                    err_b = np.hypot(xs[blk_slice] - bxh, ys[blk_slice] - byh)
                    eff_tol = tol_outlier
                    if w == 1:
                        dx1 = xs[i] - xs[i - 1]
                        dx2 = xs[i + 1] - xs[i]
                        if dx1 * dx2 < 0 and abs(xs[i] - bxh[0]) > tol_dx_rev:
                            eff_tol = max(12.0, tol_outlier * 0.72)
                    thresh = max(eff_tol, 3.0 * sup_rms)
                    if float(err_b.min()) > thresh:
                        sc = float(err_b.mean()) / (sup_max + 2.0)
                        candidates.append((sc, i, i + w))

            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                to_drop = set()
                locked = set()
                for _sc, bi, bj in candidates:
                    if any(k in locked for k in range(max(0, bi - 1), min(m, bj + 1))):
                        continue
                    for k in range(bi, bj):
                        to_drop.add(k)
                    for k in range(max(0, bi - 2), min(m, bj + 2)):
                        locked.add(k)
                if to_drop:
                    n_removed_bridge += len(to_drop)
                    cur_seg = [cur_seg[k] for k in range(m) if k not in to_drop]
                    changed = True

            if not changed:
                break

        # 再次按短缺口 + 跨球串轨时空交汇检验切分, 剔除孤立残片及脱离主球台的邻台尾段
        if len(cur_seg) < 3:
            n_removed_orphan += len(cur_seg)
            continue
        sub_segs, sub_cur = [], [cur_seg[0]]
        for k in cur_seg[1:]:
            if k - sub_cur[-1] <= max_assoc_gap:
                sub_cur.append(k)
            else:
                sub_segs.append(sub_cur)
                sub_cur = [k]
        sub_segs.append(sub_cur)
        final_sub_segs = []
        for ss in sub_segs:
            final_sub_segs.extend(_split_switched_subsegs(ss))
        for ss in final_sub_segs:
            if len(ss) < 3:
                n_removed_orphan += len(ss)
                continue
            if quad is not None:
                # 切断后若某一段全程悬在背景/侧边邻台而从未靠近主球台面, 直接整段丢弃
                near_any = any(_near_table(main_track[k][0], main_track[k][1], max(50.0, 0.50 * mq_h))
                               for k in ss)
                nb_frac = (sum(1 for k in ss if _point_in_neighbor_zones(
                               main_track[k][0], main_track[k][1], neighbor_zones, quad)) / len(ss)
                           if neighbor_zones else 0.0)
                if (not near_any) or (nb_frac >= 0.45 and not any(
                        _near_table(main_track[k][0], main_track[k][1], max(18.0, 0.10 * mq_h)) for k in ss)):
                    n_removed_orphan += len(ss)
                    continue
            for k in ss:
                cleaned[k] = main_track[k]

    if debug:
        log(f"  ★ 抛物线离群净化: {len(main_track)} → {len(cleaned)} 帧 "
            f"(剔除桥接尖刺 {n_removed_bridge}, 首尾折钩 {n_removed_end}, 孤立残片 {n_removed_orphan})")
    return cleaned


# =====================================================================
#  4) 分段自由飞行抛物线平滑 + 物理弹道补帧
# =====================================================================
def _local_diff_velocity(frames, vals, idx, direction, span=3, max_step=2):
    """用 idx 一侧 (含 idx, 相邻间隔 <= max_step) 最多 span 个点做中值差分, 返回 px/帧。"""
    n = len(frames)
    js = [idx]
    j = idx
    for _ in range(span - 1):
        j += direction
        if not (0 <= j < n):
            break
        if abs(frames[js[-1]] - frames[j]) > max_step:
            break
        js.append(j)
    js.sort()
    if len(js) < 2:
        return None
    dv = []
    for a, b in zip(js, js[1:]):
        df = frames[b] - frames[a]
        if df > 0:
            dv.append((vals[b] - vals[a]) / df)
    return float(np.median(dv)) if dv else None


def _detect_arc_breakpoints(seg_f, seg_x, seg_y, g_frame, diag):
    """在连续检测段内识别物理接触断点 (击球折返 'hit' / 台面弹跳 'bounce')。
    返回 {idx: 'hit' | 'bounce'}。
    """
    m = len(seg_f)
    if m < 3:
        return {}
    raw_bps = []  # (idx, kind, strength)
    dvx_thresh = max(9.0, diag * 0.007)
    dvy_thresh = max(2.5, 1.6 * g_frame)
    for i in range(1, m - 1):
        vx_l = _local_diff_velocity(seg_f, seg_x, i, -1, span=3, max_step=3)
        vx_r = _local_diff_velocity(seg_f, seg_x, i, +1, span=3, max_step=3)
        vy_l = _local_diff_velocity(seg_f, seg_y, i, -1, span=3, max_step=3)
        vy_r = _local_diff_velocity(seg_f, seg_y, i, +1, span=3, max_step=3)

        if vx_l is not None and vx_r is not None:
            if (vx_l * vx_r < 0 and (abs(vx_l) > 1.5 or abs(vx_r) > 1.5)) or abs(vx_r - vx_l) > dvx_thresh:
                raw_bps.append((i, "hit", abs(vx_r - vx_l)))
                continue

        if vy_l is not None and vy_r is not None:
            is_y_peak = (seg_y[i] >= seg_y[i - 1] - 1.5 and seg_y[i] >= seg_y[i + 1] - 1.5)
            if is_y_peak and vy_l > 0.8 and vy_r < -0.8 and (vy_r - vy_l) < -dvy_thresh:
                raw_bps.append((i, "bounce", abs(vy_r - vy_l)))

    # 合并相邻 1 帧内的重复极值点, 取强度最大者
    bps = {}
    for idx, kind, strength in raw_bps:
        if (idx - 1) in bps:
            prev_kind, prev_str = bps[idx - 1]
            if strength > prev_str:
                del bps[idx - 1]
                bps[idx] = (kind, strength)
        else:
            bps[idx] = (kind, strength)
    return {k: v[0] for k, v in bps.items()}


def _smooth_segment_parabolic(seg_f, seg_x, seg_y, seg_c, bps, g_frame, fps, frame_w, frame_h):
    """在接触断点切分出的每一段自由飞行弧内执行带重力约束的抛物线平滑, 并严格钉住弹跳/击球顶点。"""
    m = len(seg_f)
    if m < 3:
        return seg_x.copy(), seg_y.copy()

    diag = math.hypot(frame_w, frame_h)
    sx = seg_x.copy()
    sy = seg_y.copy()
    # 构建自由飞行子弧区间列表 [(b0, b1), ...]:
    #   - 显式接触断点 idx 处左右两侧共享顶点: (..., idx) 与 (idx, ...)
    #   - 若缺口 (i, i+1) 内部发生了弹跳(下降→上升)或击球折返(vx 变号), 则从缺口两侧断开为 (..., i) 与 (i+1, ...),
    #     严禁跨弹跳/跨击球缺口强行拟合单条抛物线把两侧真实速度抹平!
    gap_splits = set()
    for i in range(m - 1):
        if seg_f[i + 1] - seg_f[i] >= 2:
            vx0 = _local_diff_velocity(seg_f, seg_x, i, -1, span=3, max_step=2)
            vx1 = _local_diff_velocity(seg_f, seg_x, i + 1, +1, span=3, max_step=2)
            vy0 = _local_diff_velocity(seg_f, seg_y, i, -1, span=3, max_step=2)
            vy1 = _local_diff_velocity(seg_f, seg_y, i + 1, +1, span=3, max_step=2)
            if (vx0 is not None and vx1 is not None and vx0 * vx1 < 0
                    and (abs(vx0) > 1.5 or abs(vx1) > 1.5)):
                gap_splits.add(i)
            elif vy0 is not None and vy1 is not None and vy0 > 1.0 and vy1 < -1.0:
                gap_splits.add(i)

    arc_ranges = []
    cur_start = 0
    for i in range(m - 1):
        if i in bps and i > cur_start:
            arc_ranges.append((cur_start, i))
            cur_start = i
        if i in gap_splits:
            if i >= cur_start:
                arc_ranges.append((cur_start, i))
            cur_start = i + 1
    if (m - 1) >= cur_start:
        arc_ranges.append((cur_start, m - 1))

    max_single_span = max(8, int(round(fps * 0.38)))
    hw_frames = max(4.0, fps * 0.16)
    max_dev_allow = max(20.0, diag * 0.014)

    for b0, b1 in arc_ranges:
        if b1 - b0 < 2:
            continue
        af = seg_f[b0:b1 + 1]
        ax_in = seg_x[b0:b1 + 1]
        ay_in = seg_y[b0:b1 + 1]
        aw = np.clip(seg_c[b0:b1 + 1], 0.25, 1.0) ** 2
        if b0 in bps or (b0 - 1) in gap_splits:
            aw[0] *= 4.0
        if b1 in bps or b1 in gap_splits:
            aw[-1] *= 4.0

        span_f = float(af[-1] - af[0])
        use_local = (span_f > max_single_span)
        if not use_local:
            model = _fit_constrained_parabola(
                af, ax_in, ay_in, weights=aw,
                ay_min=-0.18 * g_frame, ay_max=3.0 * g_frame, ax_max=0.75 * g_frame)
            xh, yh = _eval_parabola(model, af)
            if float(np.max(np.hypot(ax_in - xh, ay_in - yh))) <= max_dev_allow:
                sx[b0:b1 + 1] = xh
                sy[b0:b1 + 1] = yh
            else:
                use_local = True

        if use_local:
            for k in range(len(af)):
                dt = np.abs(af - af[k])
                mask = dt <= hw_frames
                if int(mask.sum()) < 3:
                    order = np.argsort(dt)[:min(5, len(af))]
                    mask = np.zeros(len(af), dtype=bool)
                    mask[order] = True
                u = np.clip(dt[mask] / ( float(dt[mask].max()) + 1.0 ), 0.0, 0.999)
                tricube = (1.0 - u ** 3) ** 3
                wk = aw[mask] * tricube
                model = _fit_constrained_parabola(
                    af[mask], ax_in[mask], ay_in[mask], weights=wk,
                    ay_min=-0.18 * g_frame, ay_max=3.0 * g_frame, ax_max=0.75 * g_frame)
                xk, yk = _eval_parabola(model, af[k])
                if math.hypot(float(ax_in[k] - xk), float(ay_in[k] - yk)) <= max_dev_allow:
                    sx[b0 + k] = float(xk)
                    sy[b0 + k] = float(yk)

    # 钉住真实接触断点, 确保台面弹跳极值不被削平而漏触发落点事件
    for idx, kind in bps.items():
        sx[idx] = seg_x[idx]
        sy[idx] = seg_y[idx]
        if kind == "bounce" and 0 < idx < m - 1:
            floor_y = max(sy[idx - 1], sy[idx + 1]) + 0.25
            if sy[idx] < floor_y:
                sy[idx] = max(seg_y[idx], floor_y)

    sx = np.clip(sx, 0.0, frame_w - 1.0)
    sy = np.clip(sy, 0.0, frame_h - 1.0)
    return sx, sy


def _ballistic_fill_gap(seg_f, seg_x, seg_y, i, frame_w, frame_h,
                        g_frame=0.5, table_geo=None, bps=None):
    """缺口 (seg_f[i], seg_f[i+1]) 的物理弹道插值:
      1) 跨击球反向缺口 (D>=3 且两侧 vx 异号): 拒绝用直线硬连以免产生跨空折线, 返回空列表留白。
      2) 缺口内含单次台面弹跳 (左侧下降 vy0>0、右侧上升 vy1<0): 闭式求解精确弹跳时刻 τ_b,
         拼接落地前/弹起后两段重力抛物线 (消除旧版弹跳缺口退化为悬空水平弦的问题)。
      3) 常规自由飞行缺口: x 走弦, y 走端点钉死且重力加速度有界的抛物线
         a ∈ [0.20·g_frame, 2.8·g_frame] (无实测速度时采用标定重力 g_frame), 彻底杜绝反重力直弦与失控过冲。
    返回 (pts, mode): mode ∈ {"ballistic", "bounce_2arc", "skipped"}。
    """
    f0, f1 = int(seg_f[i]), int(seg_f[i + 1])
    x0, y0 = float(seg_x[i]), float(seg_y[i])
    x1, y1 = float(seg_x[i + 1]), float(seg_y[i + 1])
    D = float(f1 - f0)
    if D <= 1:
        return [], "skipped"

    bps = bps or {}
    # 若 i 是断点, 左侧不再跨越 i 向前取速度; 若 i+1 是断点, 右侧同理
    vx0 = None if i in bps else _local_diff_velocity(seg_f, seg_x, i, -1, span=3, max_step=2)
    vx1 = None if (i + 1) in bps else _local_diff_velocity(seg_f, seg_x, i + 1, +1, span=3, max_step=2)
    vy0 = None if i in bps else _local_diff_velocity(seg_f, seg_y, i, -1, span=3, max_step=2)
    vy1 = None if (i + 1) in bps else _local_diff_velocity(seg_f, seg_y, i + 1, +1, span=3, max_step=2)

    # 1) 缺口内发生球拍击球横向折返: D>=3 时不硬连斜弦
    if D >= 3:
        dx_gap = x1 - x0
        if vx0 is not None and vx1 is not None and vx0 * vx1 < 0 and (abs(vx0) > 2.0 or abs(vx1) > 2.0):
            return [], "skipped"
        if vx0 is not None and abs(vx0) > 4.0 and dx_gap * vx0 < 0 and abs(dx_gap) > 10.0:
            return [], "skipped"
        if vx1 is not None and abs(vx1) > 4.0 and dx_gap * vx1 < 0 and abs(dx_gap) > 10.0:
            return [], "skipped"

    # 2) 缺口内含单次台面弹跳 (下降 -> 弹起): 双段重力抛物线闭式重建
    if vy0 is not None and vy1 is not None and vy0 > 1.5 and vy1 < -1.5:
        denom = vy0 - vy1 + g_frame * D
        if denom > 1e-6:
            tau_b = ((y1 - y0) - vy1 * D + 0.5 * g_frame * D * D) / denom
            if 0.4 <= tau_b <= (D - 0.4):
                xb = x0 + (x1 - x0) * (tau_b / D)
                yb = y0 + vy0 * tau_b + 0.5 * g_frame * tau_b * tau_b
                quad = table_geo.get("quad") if table_geo else None
                diag = math.hypot(frame_w, frame_h)
                near_tbl = (True if quad is None else
                            cv2.pointPolygonTest(quad.reshape(-1, 1, 2),
                                                 (float(xb), float(yb)), True) >= -max(60.0, diag * 0.04))
                if yb > max(y0, y1) and yb < frame_h and near_tbl:
                    vy0_in = (yb - y0 - 0.5 * g_frame * tau_b * tau_b) / tau_b
                    d2 = D - tau_b
                    vy_out = (y1 - yb - 0.5 * g_frame * d2 * d2) / d2
                    pts = []
                    for f in range(f0 + 1, f1):
                        t = float(f - f0)
                        xi = x0 + (x1 - x0) * (t / D)
                        if t <= tau_b:
                            yi = y0 + vy0_in * t + 0.5 * g_frame * t * t
                        else:
                            u = t - tau_b
                            yi = yb + vy_out * u + 0.5 * g_frame * u * u
                        pts.append([_clamp(xi, 0, frame_w - 1), _clamp(yi, 0, frame_h - 1)])
                    return pts, "bounce_2arc"
        if D >= 4:
            return [], "skipped"

    # 3) 常规同弧自由飞行缺口: 端点钉死 + 有界物理重力
    a_cands = []
    if vy0 is not None:
        a_cands.append(2.0 * (y1 - y0 - vy0 * D) / (D * D))
    if vy1 is not None:
        a_cands.append(2.0 * (y0 - y1 + vy1 * D) / (D * D))
    if a_cands:
        a_raw = sum(a_cands) / len(a_cands)
        if not math.isfinite(a_raw):
            return [], "skipped"
        # 若跨缺口需要强反重力向上加速(且未通过上方台面弹跳双弧检验, 如从前景落网球跳到高处背景台球), 拒绝硬连直线!
        if D >= 2 and a_raw < -0.65 * g_frame:
            return [], "skipped"
        a = _clamp(a_raw, 0.20 * g_frame, 2.8 * g_frame)
    else:
        a = g_frame
    vy0_eff = (y1 - y0 - 0.5 * a * D * D) / D

    pts = []
    for f in range(f0 + 1, f1):
        t = float(f - f0)
        xi = x0 + (x1 - x0) * t / D
        yi = y0 + vy0_eff * t + 0.5 * a * t * t
        pts.append([_clamp(xi, 0, frame_w - 1), _clamp(yi, 0, frame_h - 1)])
    return pts, "ballistic"


def interpolate_track(main_track, fps, frame_w, frame_h, debug=False,
                      table_geo=None, g_frame=None, smooth=True):
    if not main_track:
        return {}
    if g_frame is None:
        g_frame, _ = estimate_gravity_px_per_frame2(
            table_geo, main_track, fps, frame_w, frame_h, debug=False)
    diag = math.hypot(frame_w, frame_h)
    keys = sorted(main_track.keys())
    # 补帧视野: 只补 ≤0.3s 的缺口; 更长的缺口物理上无法可靠恢复, 直接留空
    max_gap = max(1, int(fps * 0.3))
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
    n_ballistic = n_bounce = n_skipped = 0
    for seg in segments:
        if len(seg) == 1:
            k = seg[0]
            filled[k] = (float(main_track[k][0]), float(main_track[k][1]), False)
            continue
        seg_f = np.array(seg, dtype=np.float64)
        seg_x = np.array([main_track[k][0] for k in seg], dtype=np.float64)
        seg_y = np.array([main_track[k][1] for k in seg], dtype=np.float64)
        seg_c = np.array([main_track[k][2] if len(main_track[k]) > 2 else 0.8 for k in seg],
                         dtype=np.float64)

        bps = _detect_arc_breakpoints(seg_f, seg_x, seg_y, g_frame, diag)
        if smooth:
            seg_x, seg_y = _smooth_segment_parabolic(
                seg_f, seg_x, seg_y, seg_c, bps, g_frame, fps, frame_w, frame_h)

        for idx_k, k in enumerate(seg):
            filled[k] = (float(seg_x[idx_k]), float(seg_y[idx_k]), False)

        for i in range(len(seg) - 1):
            gs, ge = seg[i], seg[i + 1]
            if ge - gs <= 1:
                continue
            pts, mode = _ballistic_fill_gap(
                seg_f, seg_x, seg_y, i, frame_w, frame_h,
                g_frame=g_frame, table_geo=table_geo, bps=bps)
            if mode == "ballistic":
                n_ballistic += 1
            elif mode == "bounce_2arc":
                n_bounce += 1
            else:
                n_skipped += 1
                continue
            for k, (xi, yi) in zip(range(gs + 1, ge), pts):
                filled[k] = (xi, yi, True)
    if debug:
        n_interp = sum(1 for v in filled.values() if v[2])
        log(f"  补帧+分段抛物线平滑(smooth={smooth}): 原始 {len(main_track)} 帧 -> 填充后 {len(filled)} 帧 "
            f"(插值 {n_interp}; 弹道缺口 {n_ballistic}, 弹跳双弧缺口 {n_bounce}, 跨反向留白 {n_skipped})")
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
        x0, y0, _ = filled_track[f0]
        x1, y1, _ = filled_track[f1]
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
                x0, y0, _ = filled_track[f0]
                x1, y1, _ = filled_track[f]
                t0 = img_to_table(table_geo, (x0, y0))
                t1 = img_to_table(table_geo, (x1, y1))
                df = f - f0
                if t0 and t1 and 0 < df <= int(fps * 0.4) + 2:
                    d_mm = math.hypot(t1[0] - t0[0], t1[1] - t0[1])
                    dt = df / fps
                    if dt > 0:
                        raw_kmh = (d_mm / 1000.0) / dt * 3.6
                        # 抑制高抛球经台面单应性透视放大产生的极端虚高球速
                        if 0.0 < raw_kmh <= 165.0:
                            kmh = raw_kmh
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


def _get_net_center_x(frames_or_path):
    """稳健获取球网(兜底球台)水平中心坐标: 支持直接传入已加载的 frames 列表或日志路径(.jsonl/.gz)。"""
    net_xs = []
    table_xs = []
    if isinstance(frames_or_path, list):
        sample = frames_or_path[::max(1, len(frames_or_path) // 200)]
        for det in sample:
            for d in det.get("detections", []):
                if d.get("confidence", 0.0) < 0.45:
                    continue
                b = d.get("bbox")
                if not b or len(b) != 4:
                    continue
                cls = d.get("class")
                if cls == "net":
                    net_xs.append((b[0] + b[2]) * 0.5)
                elif cls == "table":
                    table_xs.append((b[0] + b[2]) * 0.5)
    elif frames_or_path and os.path.exists(str(frames_or_path)):
        with _open_text_maybe_gz(frames_or_path) as f:
            for li, line in enumerate(f):
                if li % 10 != 0:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    det = json.loads(line)
                except Exception:
                    continue
                for d in det.get("detections", []):
                    if d.get("confidence", 0.0) < 0.45:
                        continue
                    b = d.get("bbox")
                    if not b or len(b) != 4:
                        continue
                    if d.get("class") == "net":
                        net_xs.append((b[0] + b[2]) * 0.5)
                    elif d.get("class") == "table":
                        table_xs.append((b[0] + b[2]) * 0.5)
                if len(net_xs) >= 50:
                    break
    if net_xs:
        return float(np.median(net_xs))
    if table_xs:
        return float(np.median(table_xs))
    return None


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
    p5 = speeds[int(len(speeds) * 0.05)]
    p50 = speeds[len(speeds) // 2]
    return max(min(p5 * 0.3, p50 * 0.1), 80.0)


def detect_bounce_events(filled_track, table_geo, fps, table_edge_tol=30.0):
    """统一落点/击球检测 (合并原 compute_bounce_halves + detect_events)。

    落点 (bounce) 两级采纳 (保留原①语义):
      strict: 图像在 quad 内 且 投影落在台面 ±_SURFACE_TOL 内;
      edge:   投影在长边延长带 (纵向 ±_EDGE_EXTEND, 横向 ±_SURFACE_TOL)
              且与前一个已采纳落点在 _ALT_WINDOW_SEC 秒内且半台相反(交替)。
    击球 (hit): 当前帧未采纳为 bounce 时, vx 变号(|Δ|>2px) + 台面内
                (保留原②语义, 且不受 bounce cooldown 限制)。
    半台: 落点直接判定并携带 ('FAR'/'NEAR'); 无台面几何时 half=None。

    几何缺失 (table_geo=None) 退化行为 (保留原②语义):
      quad/台面判定恒真 → 候选门 + cooldown 通过即收 bounce, vx 变号即收 hit;
      speed_kmh/table_xy/half/adopted 均为 None。

    返回 (events, bounce_halves):
      events: [{frame, x, y, type('bounce'/'hit'), half, adopted,
                table_xy, speed_kmh}, ...]    # 按帧升序
      bounce_halves: {frame: 'FAR'/'NEAR'}    # 仅含 half 非空的落点, 供回合逻辑
    """
    events = []
    if not filled_track or fps <= 0:
        return events, {}
    keys = sorted(filled_track.keys())
    pts = [(k, filled_track[k][0], filled_track[k][1], filled_track[k][2]) for k in keys]
    n = len(pts)
    if n < 3:
        return events, {}
    has_geo = table_geo is not None
    quad = table_geo.get("quad") if has_geo else None
    corridor = table_geo.get("corridor") if has_geo else None
    _TX_MAX = TABLE_WID_M * 1000
    _TY_MAX = TABLE_LEN_M * 1000
    _SURFACE_TOL = 80
    _TY_HALF = TABLE_LEN_M * 1000 / 2.0
    _EDGE_EXTEND = _TY_MAX * _EDGE_EXTEND_RATIO   # 长边延长带深度 (mm)
    _ALT_WIN_FRAMES = max(1, int(round(fps * _ALT_WINDOW_SEC)))
    _TABLE_EDGE_TOL = table_edge_tol   # 落点允许在 quad 四边形外该像素内 (CLI: --table-edge-tol)

    def _in_table(x, y):
        if quad is None:
            return True
        # 测距模式: 点在四边形外但距边界 ≤ _TABLE_EDGE_TOL 像素也算通过
        return cv2.pointPolygonTest(quad.reshape(-1, 1, 2),
                                    (float(x), float(y)), True) >= -_TABLE_EDGE_TOL

    def _in_hit_zone(x, y):
        # 选手常在端线外/台面斜上方挥拍击球, 击球判定应使用走廊区域而非仅限桌面四边形 quad
        if corridor is not None:
            return _point_in_corridor((x, y), corridor)
        return _in_table(x, y)

    def _on_table_surface(txy):
        tx, ty = txy
        return (-_SURFACE_TOL <= tx <= _TX_MAX + _SURFACE_TOL and
                -_SURFACE_TOL <= ty <= _TY_MAX + _SURFACE_TOL)

    def _on_edge_band(txy):
        """长边延长线区域: 纵向(长边方向)允许超出台面 _EDGE_EXTEND, 横向保持 ±_SURFACE_TOL。"""
        tx, ty = txy
        return (-_SURFACE_TOL <= tx <= _TX_MAX + _SURFACE_TOL and
                -_EDGE_EXTEND <= ty <= _TY_MAX + _EDGE_EXTEND)

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
        v_kmh = (d_mm / 1000.0) / dt * 3.6
        return v_kmh if 0.0 < v_kmh <= 165.0 else None

    bounce_halves = {}
    last_bounce_frame = -10 ** 9
    # 物理上乒乓球不可能在 <0.12s 内于台面弹跳两次; 避免弹跳峰值轻微平顶在 2 帧内重复计点误触 three_same_half
    bounce_cooldown_frames = max(3, int(round(fps * 0.12)))
    max_adj_gap = max(2, int(round(fps * 0.08)))
    last_ok_frame = None    # 最近一次已采纳落点帧号 (严格或宽松)
    last_ok_half = None     # 其半台
    for i in range(1, n - 1):
        f, x, y, is_interp = pts[i]
        f_p, x_p, y_p, _ = pts[i - 1]
        f_n, x_n, y_n, _ = pts[i + 1]
        # 禁止跨未补帧长缺口(如两回合间隔数秒)做局部极值与变号差分, 防止回合首尾产生伪落点/伪击球
        if (f - f_p) > max_adj_gap or (f_n - f) > max_adj_gap:
            continue
        ev_type = None
        is_local_y_max = (y >= pts[i - 1][2] and y >= pts[i + 1][2])
        vy_prev, vy_next = y - y_p, y_n - y
        # vy 变号门槛: 插值帧(球漏检期间被填充, 平滑会压低帧间位移)用低门槛 0.01,
        # 真实检测帧保持 0.1 以挡亚像素噪声
        vy_gate = 0.01 if is_interp else 0.1
        if is_local_y_max and vy_prev > vy_gate and vy_next < -vy_gate:
            if f - last_bounce_frame >= bounce_cooldown_frames:
                txy = img_to_table(table_geo, (x, y)) if has_geo else None
                half = (None if (not has_geo or txy is None)
                        else ('FAR' if txy[1] < _TY_HALF else 'NEAR'))
                accepted = False
                adopted = None
                if not has_geo:
                    accepted = True                     # 退化照收 (原②语义)
                elif txy is not None:
                    # _on_table_surface 过滤暂时注释 (排查落点丢失, 观察效果后决定恢复/删除)
                    if _in_table(x, y):  # and _on_table_surface(txy):
                        accepted = True                 # ① 严格落点
                        adopted = 'strict'
                    elif (_on_edge_band(txy)            # ② 长边延长线放宽
                          and last_ok_frame is not None
                          and f - last_ok_frame <= _ALT_WIN_FRAMES
                          and half != last_ok_half):    # 2s 内 + 半台交替
                        accepted = True
                        adopted = 'edge'
                if accepted:
                    ev_type = "bounce"
                    last_bounce_frame = f
                    last_ok_frame = f
                    last_ok_half = half
                    spd = speed_kmh(f_p, f_n, pts[i - 1], pts[i + 1])
                    events.append({"frame": f, "x": x, "y": y, "type": ev_type,
                                   "half": half, "adopted": adopted,
                                   "table_xy": txy, "speed_kmh": spd})
                    if half is not None:
                        bounce_halves[f] = half
        if ev_type is None:
            vx_prev, vx_next = x - x_p, x_n - x
            if vx_prev * vx_next < 0 and abs(vx_prev) > 2.0 and abs(vx_next) > 2.0:
                if _in_hit_zone(x, y):
                    ev_type = "hit"
                    spd = speed_kmh(f_p, f_n, pts[i - 1], pts[i + 1])
                    txy = img_to_table(table_geo, (x, y)) if has_geo else None
                    events.append({"frame": f, "x": x, "y": y, "type": ev_type,
                                   "half": None, "adopted": None,
                                   "table_xy": txy, "speed_kmh": spd})
    return events, bounce_halves


def detect_rallies(filled_track, fps, total_frames, raw_log_path, table_geo=None,
                   rally_lost_timeout=1.0, rally_min_duration=0.6, pre_roll_sec=1.0,
                   slow_speed_window_sec=1.0, slow_speed_min_in_play_sec=0.8,
                   out_of_table_timeout=1.0, no_bounce_timeout=2.0,
                   bounce_halves=None, net_center_x_hint=None, debug=False):
    if not filled_track or fps <= 0:
        return []
    ball_lookup = {f: ((x, y), not interp) for f, (x, y, interp) in filled_track.items()}
    sorted_frames = sorted(ball_lookup.keys())
    net_line = table_geo.get("net_line") if table_geo else None
    net_center_x = (net_center_x_hint if net_center_x_hint is not None
                    else (_get_net_center_x(raw_log_path) if net_line is None else None))
    if net_line is None and net_center_x is None and not bounce_halves:
        log("  ⚠️ 无球网且无落点信息, 回退到连续性切分")
        keys = sorted_frames
        max_gap = int(fps * 0.45)
        rallies = []
        cur = [keys[0]]
        for k in keys[1:]:
            if k - cur[-1] <= max_gap:
                cur.append(k)
            else:
                rallies.append({
                    "start_frame": cur[0],
                    "end_frame": cur[-1],
                    "duration": round((cur[-1] - cur[0]) / fps, 2),
                    "ended_reason": "gap_split",
                })
                cur = [k]
        rallies.append({
            "start_frame": cur[0],
            "end_frame": cur[-1],
            "duration": round((cur[-1] - cur[0]) / fps, 2),
            "ended_reason": "gap_split",
        })
        return rallies
    slow_thresh = _estimate_slow_speed_threshold(ball_lookup, fps)
    corridor = table_geo.get("corridor") if table_geo else None
    OUT_OF_TABLE_FRAMES = max(1, int(round(fps * out_of_table_timeout)))
    NO_BOUNCE_FRAMES = max(1, int(round(fps * no_bounce_timeout)))
    BOUNCE_START_WINDOW_FRAMES = max(1, int(round(fps * 2.0)))
    if debug:
        log(f"  落点半台检测: {len(bounce_halves)} 个落点")
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
    same_half_count = 0
    last_bounce_half = None
    no_bounce_last_frame = None
    rallies = []
    traj_buffer = deque(maxlen=max(PRE_ROLL_FRAMES * 3, int(fps * 6)))
    idle_bounces = deque()

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
            rally_start = None
            if entry is not None:
                if bounce_halves:
                    # 落点驱动: 2 秒窗口内两半台(FAR/NEAR)都有落点 → 以倒数第二个落点开始回合
                    bh = bounce_halves.get(frame_idx)
                    if bh is not None:
                        idle_bounces.append((frame_idx, bh))
                        cutoff = frame_idx - BOUNCE_START_WINDOW_FRAMES
                        while idle_bounces and idle_bounces[0][0] < cutoff:
                            idle_bounces.popleft()
                        halves = {h for _, h in idle_bounces}
                        if len(idle_bounces) >= 2 and len(halves) == 2:
                            rally_start = idle_bounces[-2][0]
                elif prev_side is not None and side is not None and side != prev_side:
                    # 无落点信息时跨网兜底
                    rally_start = max(0, frame_idx - PRE_ROLL_FRAMES)
            if rally_start is not None:
                rally_num += 1
                current_rally = {"start_frame": rally_start, "end_frame": frame_idx, "crossed_net": True}
                rally_state = RallyState.IN_PLAY
                slow_run_frames = out_of_table_run = 0
                same_half_count, last_bounce_half = 0, None
                no_bounce_last_frame = frame_idx
                idle_bounces.clear()
                while traj_buffer and traj_buffer[0][0] < rally_start:
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
                elif out_of_table_run > 0:
                    # 球丢失前已在走廊外 → 视为朝走廊外飞走(出台), 丢失期间继续累计
                    out_of_table_run += 1
                if out_of_table_run >= OUT_OF_TABLE_FRAMES:
                    end_by_out = True
            # 落点状态机: ① 同半台连续落点 3 次 → 结束; ② 球台连续无新落点超时 → 结束
            end_by_same_half = False
            end_by_no_bounce = False
            if entry is not None:
                bh = bounce_halves.get(frame_idx)
                if bh is not None:
                    no_bounce_last_frame = frame_idx
                    if bh == last_bounce_half:
                        same_half_count += 1
                    else:
                        same_half_count, last_bounce_half = 1, bh
                    if same_half_count >= 3:
                        end_by_same_half = True
            # 无落点超时不依赖球是否被检测到: 球丢失(漏检/出画)期间同样计时
            if (bounce_halves and no_bounce_last_frame is not None and
                    frame_idx - no_bounce_last_frame >= NO_BOUNCE_FRAMES):
                end_by_no_bounce = True
            if end_by_lost or end_by_slow or end_by_out or end_by_same_half or end_by_no_bounce:
                if end_by_same_half:
                    end_reason = "three_same_half"
                elif end_by_no_bounce:
                    end_reason = "no_bounce"
                elif end_by_out:
                    end_reason = "ball_out_of_table"
                elif end_by_lost:
                    end_reason = "ball_lost"
                else:
                    end_reason = "ball_too_slow"
                if end_by_same_half:
                    end_f = frame_idx
                elif end_by_no_bounce:
                    end_f = no_bounce_last_frame  # 回退到最后一次落点帧, 裁掉无落点空窗
                elif end_by_out:
                    end_f = frame_idx
                    for f in range(frame_idx, current_rally["start_frame"] - 1, -1):
                        e = ball_lookup.get(f)
                        if e is not None and _point_in_corridor((e[0][0], e[0][1]), corridor):
                            end_f = f
                            break
                elif end_by_lost and missing_ball_frames > 0:
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
                _finalize_rally(rallies, current_rally, end_reason, rally_min_duration)
                rally_state = RallyState.IDLE
                current_rally = None
                slow_run_frames = out_of_table_run = 0
                same_half_count, last_bounce_half = 0, None
                no_bounce_last_frame = None
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
        half = lp.get("half") or _landing_half(lp)
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
def _min_side_bounce_count(r, bounce_far, bounce_near):
    """回合 [start,end] 区间内两半台各自落点数, 取较少一侧; 无落点信息返回 0。"""
    if not bounce_far and not bounce_near:
        return 0
    s, e = r["start_frame"], r["end_frame"]
    return min(bisect.bisect_right(bounce_far, e) - bisect.bisect_left(bounce_far, s),
               bisect.bisect_right(bounce_near, e) - bisect.bisect_left(bounce_near, s))


def _rally_avg_speeds_px(rallies, ball_speed_table):
    """计算每个回合的平均球速 (px/s, 仅非零帧), 返回 [(avg_sp, r)] 按球速降序。

    效率: ball_speed_table 预展平为 numpy 数组 (按帧号索引), 每个回合切片+掩码求一次均值。
    回合区间外/缺失速度数据帧不参与平均; 无速度数据的回合返回 0.0。"""
    if ball_speed_table:
        max_f = max(ball_speed_table.keys()) + 1
        px_s_arr = np.zeros(max_f, dtype=np.float64)
        for f, st in ball_speed_table.items():
            v = st.get("px_s")
            if v and v > 0:
                px_s_arr[f] = v
    else:
        px_s_arr = np.zeros(0, dtype=np.float64)
    arr_len = px_s_arr.shape[0]
    out = []
    for r in rallies:
        s = r["start_frame"]
        e = min(r["end_frame"] + 1, arr_len)
        if s >= arr_len or e <= s:
            out.append((0.0, r))
            continue
        slice_sp = px_s_arr[s:e]
        mask = slice_sp > 0
        out.append((float(slice_sp[mask].mean()) if mask.any() else 0.0, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def build_segments(rallies, fps, total_frames, args, bounce_halves=None,
                   min_bounce_count=0, ball_speed_table=None, speed_top_n=0):
    min_frames = args.min_rally_frames or int(fps * 2)
    pre = int(args.pre_buffer * fps)
    post = int(args.post_buffer * fps)
    merge_gap = int(args.merge_gap * fps)
    # 落点半台索引 (短回合豁免用)
    bounce_far = []
    bounce_near = []
    if bounce_halves:
        bounce_far = sorted(f for f, h in bounce_halves.items() if h == "FAR")
        bounce_near = sorted(f for f, h in bounce_halves.items() if h == "NEAR")
    # 球速豁免: 平均球速前 speed_top_n 名的回合无条件视为有效
    speed_top_ids = set()
    if ball_speed_table and speed_top_n > 0 and rallies:
        speed_top_ids = {id(r) for _, r in
                         _rally_avg_speeds_px(rallies, ball_speed_table)[:speed_top_n]}
    valid_rallies = []
    for r in rallies:
        dur_frames = r["end_frame"] - r["start_frame"] + 1
        if dur_frames >= min_frames:
            valid_rallies.append(r)                       # 时长达标
        elif _min_side_bounce_count(r, bounce_far, bounce_near) >= min_bounce_count:
            valid_rallies.append(r)                       # 短回合豁免: 落点充分
        elif id(r) in speed_top_ids:
            valid_rallies.append(r)                       # 球速 top-N 豁免
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


def _pick_top_speed_segments(rallies, ball_speed_table, fps, total_frames,
                              top_n=7, min_duration=10.0,
                              pre_buffer=0.5, post_buffer=1.5,
                              bounce_halves=None, min_bounce_count=0):
    """筛选候选回合: 合并前时长 >= min_duration 或 落点数较少半台的落点数 > min_bounce_count
    (两条件取或, 满足任一即可), 按平均球速降序取 top_n, 返回片段列表 + rally_starts_by_seg。

    效率优化: 将 ball_speed_table 预展平为 numpy 数组 (按帧号索引),
    每个回合用切片 + 布尔掩码一次性求均值, 避免逐帧 dict.get + 字段访问的 Python 循环开销。
    落点计数: 两半台落点帧号预先排序, 用 bisect 在 O(log n) 内统计回合区间内落点数。
    """
    # 落点半台索引: 两半台各自的落点帧号升序列表
    bounce_far = []
    bounce_near = []
    if bounce_halves:
        bounce_far = sorted(f for f, h in bounce_halves.items() if h == "FAR")
        bounce_near = sorted(f for f, h in bounce_halves.items() if h == "NEAR")

    valid = [r for r in rallies if (r.get("duration", 0) >= min_duration
                                    or _min_side_bounce_count(r, bounce_far, bounce_near) > min_bounce_count)]
    if not valid:
        return [], {}

    # 平均球速 (px/s, 非零帧) 降序 — 复用模块级实现
    rally_speeds = _rally_avg_speeds_px(valid, ball_speed_table)
    top = rally_speeds[:top_n]
    pre = int(pre_buffer * fps)
    post = int(post_buffer * fps)
    segments = []
    rally_starts_by_seg = {}
    for si, (sp, r) in enumerate(top):
        bs = max(0, r["start_frame"] - pre)
        be = min(total_frames - 1, r["end_frame"] + post)
        segments.append((int(bs), int(be)))
        rally_starts_by_seg[si] = [r["start_frame"]]
    n_above = len(valid)
    avg_top = [sp for sp, _ in top]
    if avg_top:
        log(f"🎯 前置精彩回合: {len(rallies)}回合中{n_above}个通过筛选 "
            f"(时长>={min_duration}s 或 少落点半台落点>{min_bounce_count}), "
            f"球速排序取top{len(top)}, 速度{min(avg_top):.0f}~{max(avg_top):.0f} px/s")
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


# 残影 (球拖尾) 整体浓淡/粗细系数
# 1.0 = v4.14 及以前的观感; <1 表示整体更透明 / 更细 (形状与渐变比例不变)
TRAIL_ALPHA_SCALE = 0.77
TRAIL_THICK_SCALE = 0.77


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
        thick_scale = (1.0 + speed_norm * 1.8) * (2.0 / 3.0) * TRAIL_THICK_SCALE
        for i in range(1, n):
            a = i / n
            alpha = (0.15 + 0.55 * a) * (0.85 + 0.15 * self.intensity) * TRAIL_ALPHA_SCALE
            thick = max(1, int((1 + 4 * a) * thick_scale))
            p0 = (int(trail_pts[i - 1][0]), int(trail_pts[i - 1][1]))
            p1 = (int(trail_pts[i][0]), int(trail_pts[i][1]))
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
        # 按帧率归一化单帧位移阈值, 保证 30fps / 60fps / 120fps 光环响应一致
        self.fast_thresh = self.diag * 0.012 * (30.0 / max(float(fps), 1.0))
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


_TRAIL_POLISH_KERNEL = np.array([0.15, 0.70, 0.15], dtype=np.float64)


def smooth_trail(points):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float64)
    xs = np.convolve(pts[:, 0], _TRAIL_POLISH_KERNEL, mode="same")
    ys = np.convolve(pts[:, 1], _TRAIL_POLISH_KERNEL, mode="same")
    xs[0], ys[0] = pts[0]
    xs[-1], ys[-1] = pts[-1]
    # 保护真实台面弹跳峰值与球拍击球折返顶点不被二次圆角削平而悬空
    for i in range(1, len(pts) - 1):
        dx1 = pts[i, 0] - pts[i - 1, 0]
        dx2 = pts[i + 1, 0] - pts[i, 0]
        dy1 = pts[i, 1] - pts[i - 1, 1]
        dy2 = pts[i + 1, 1] - pts[i, 1]
        if (dx1 * dx2 < 0 and (abs(dx1) > 1.5 or abs(dx2) > 1.5)) or (dy1 > 1.0 and dy2 < -1.0):
            xs[i], ys[i] = pts[i, 0], pts[i, 1]
    return list(zip(xs, ys))


def render(args, segments, rally_starts_by_seg, filled_track, events, table_geo,
           in_w, in_h, fps, total_frames, ball_speed_table, player_boxes,
           prepend_segments=None, prepend_rally_starts_by_seg=None,
           black_gap_sec=1.0):
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

    scaled_track = {f: (x * scale, y * scale) for f, (x, y, _) in filled_track.items()}

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
                ] + _enc_args(enc) + [silent_path]
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
        labels = FloatingLabel()
        flash = SpeedFlash()
        trail_renderer = SpeedTrailRenderer(intensity=fx_intensity)
        player_fx = PlayerFX(fps, out_w, out_h, intensity=fx_intensity) if enable_player_fx else None
        # 残影 (球拖尾) 时长 ≈ 0.2s; 下限 4 帧保证可平滑/绘制
        TRAIL_LEN = max(4, int(fps * 0.2))
        max_trail_step = getattr(args, "max_jump_ratio", 0.10) * math.hypot(out_w, out_h) * 0.85

        prepend_frame_count = sum(e - ps + 1 for ps, e in (prepend_segments or []))
        black_frame_count = int(round(fps * black_gap_sec)) if prepend_segments else 0
        total_render = (prepend_frame_count + black_frame_count +
                        sum(e - s + 1 for s, e in segments))

        # ---- 淡入淡出参数 ----
        fade_duration = 0.3  # 淡入淡出时长（秒）
        fade_frames = int(round(fps * fade_duration))
        main_start = prepend_frame_count + black_frame_count
        black_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        def _apply_fade(frame, fidx):
            """对帧应用淡入淡出 (in-place), fidx 为 0-based 全局帧序号。"""
            if total_render <= fade_frames * 2 or fade_frames <= 0:
                return
            alpha = 1.0
            if fidx < fade_frames:
                # ① 开头淡入
                alpha = fidx / fade_frames
            elif prepend_segments and prepend_frame_count - fade_frames <= fidx < prepend_frame_count:
                # ② 前置→黑幕 淡出
                alpha = (prepend_frame_count - fidx) / fade_frames
            elif prepend_segments and main_start <= fidx < main_start + fade_frames:
                # ③ 黑幕→主视频 淡入
                alpha = (fidx - main_start) / fade_frames
            elif fidx >= total_render - fade_frames:
                # ④ 末尾淡出
                alpha = (total_render - fidx) / fade_frames
            if alpha < 1.0:
                cv2.addWeighted(frame, alpha, black_img, 1.0 - alpha, 0, dst=frame)

        pipe_method = "ffmpeg pipe 直编" if use_pipe else "cv2.VideoWriter"
        log(f"🎬 渲染: {len(segments)} 个片段, 约 {total_render} 帧 @ {out_w}x{out_h} "
            f"{fps:.2f}fps [{pipe_method} → {enc}]  fx={fx_level}"
            f"{' +选手光环' if enable_player_fx else ''}")
        written = 0
        t_start = time.time()
        last_progress_time = t_start
        do_resize = (scale != 1.0)

        def _write_frame_and_progress(frm):
            nonlocal use_pipe, writer, written, last_progress_time
            if use_pipe:
                try:
                    pipe_proc.stdin.write(frm.tobytes())
                except (BrokenPipeError, OSError):
                    log("⚠️ ffmpeg pipe 断开, 回退到 cv2.VideoWriter")
                    try:
                        pipe_proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        pipe_proc.wait(timeout=5)
                    except Exception:
                        pass
                    use_pipe = False
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建回退视频写入器")
                    writer.write(frm)
            else:
                writer.write(frm)
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

        def _render_segment_batch(seg_list, seg_landings_map, seg_rally_id_map):
            for idx_seg, (s_f, e_f) in enumerate(seg_list):
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_f)
                trail = deque(maxlen=TRAIL_LEN)
                last_trail_f = None
                # 切换剪辑片段时清空上一段残留的粒子/文字/闪屏, 防止跨镜头特效串场
                particles.particles.clear()
                labels.labels.clear()
                flash._t = 0
                flash._max = 0
                cur_landings_sorted = sorted(seg_landings_map.get(idx_seg, []),
                                             key=lambda lp: lp["frame"])
                lp_idx = 0
                cur_lp_so_far = []
                cur_rally_id = None
                prev_player_boxes = None
                for f in range(s_f, e_f + 1):
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
                        if enable_extra_fx and not is_bounce and snorm >= 0.6:
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
                        # 跨越未补帧缺口或极端跳变时立即清空旧残影, 防止旧尾点与新位置跨屏连成折线/直线
                        if last_trail_f is not None and (
                            (f - last_trail_f) > 1
                            or (trail and math.hypot(pt[0] - trail[-1][0],
                                                     pt[1] - trail[-1][1]) > max_trail_step)
                        ):
                            trail.clear()
                        trail.append(pt)
                        last_trail_f = f
                    elif trail:
                        # 彻底排空到 0 (修复旧版 len>1 留 1 个幽灵点导致跨缺口拉直线的问题)
                        trail.popleft()
                        if not trail:
                            last_trail_f = None

                    if len(trail) >= 2:
                        st = ball_speed_table.get(f)
                        snorm = st["norm"] if st else 0.3
                        trail_renderer.draw(frame, smooth_trail(list(trail)), speed_norm=snorm)

                    particles.update_and_draw(frame)
                    labels.update_and_draw(frame)
                    flash.update_and_draw(frame)

                    while lp_idx < len(cur_landings_sorted) and cur_landings_sorted[lp_idx]["frame"] <= f:
                        lp = cur_landings_sorted[lp_idx]
                        rid = seg_rally_id_map.get((idx_seg, lp["frame"]))
                        if cur_rally_id is not None and rid != cur_rally_id:
                            cur_lp_so_far = []
                        cur_rally_id = rid
                        cur_lp_so_far.append(lp)
                        lp_idx += 1
                    mini.draw(frame, cur_lp_so_far)
                    _apply_fade(frame, written)
                    _write_frame_and_progress(frame)

        # — 前置精彩回合 (球速最快) —
        if prepend_segments:
            prepend_landing_by_seg, prepend_landing_rally_id = build_landing_rally_index(
                prepend_segments, prepend_rally_starts_by_seg or {},
                landing_all, debug=False)
            log(f"🎬 渲染前置精彩回合: {len(prepend_segments)} 段")
            _render_segment_batch(prepend_segments, prepend_landing_by_seg,
                                  prepend_landing_rally_id)
            # 黑幕间隔
            for _ in range(black_frame_count):
                _write_frame_and_progress(black_img)
            sys.stdout.write("\n")
            sys.stdout.flush()
            log(f"⏯️  前置 {len(prepend_segments)} 段 + 黑幕 {black_gap_sec}s 完成, "
                f"进入全顺序视频")

        _render_segment_batch(segments, landing_by_seg, landing_rally_id)
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
                        already_encoded=use_pipe, enc=enc,
                        prepend_segments=prepend_segments,
                        black_gap_sec=black_gap_sec if prepend_segments else 0.0)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def finalize_output(args, silent_path, segments, fps, written, tmp_dir,
                    already_encoded=False, enc="libx264",
                    prepend_segments=None, black_gap_sec=0.0):
    has_ffmpeg = bool(shutil.which("ffmpeg"))
    if not has_ffmpeg:
        log("🎞️ 未检测到 ffmpeg, 直接输出渲染视频 (mp4v, 无音频)")
        shutil.copy2(silent_path, args.output)
        return
    has_audio = (not args.no_audio) and ffprobe_has_audio(args.video)
    if not has_audio:
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
    # 构建完整音频滤镜: 前置精彩剪辑 + 静音间隔 + 全顺序剪辑
    parts, filt = [], []
    aidx = 0
    # 前置片段
    if prepend_segments:
        for ps, pe in prepend_segments:
            t0 = ps / fps
            t1 = (pe + 1) / fps
            filt.append(
                f"[1:a]atrim=start={t0:.4f}:end={t1:.4f},"
                f"asetpts=PTS-STARTPTS[a{aidx}]")
            parts.append(f"[a{aidx}]")
            aidx += 1
    # 黑幕静音 (与实际渲染黑幕帧数严格对齐, 消除浮点取整累计的音画微偏)
    if black_gap_sec > 0:
        exact_black_sec = int(round(fps * black_gap_sec)) / fps
        if exact_black_sec > 0:
            filt.append(
                f"anullsrc=r=48000:cl=stereo,atrim=0:{exact_black_sec:.4f},"
                f"asetpts=PTS-STARTPTS[a{aidx}]")
            parts.append(f"[a{aidx}]")
            aidx += 1
    # 主片段
    for i, (s, e) in enumerate(segments):
        t0 = s / fps
        t1 = (e + 1) / fps
        filt.append(
            f"[1:a]atrim=start={t0:.4f}:end={t1:.4f},"
            f"asetpts=PTS-STARTPTS[a{aidx}]")
        parts.append(f"[a{aidx}]")
        aidx += 1
    concat = "".join(parts) + f"concat=n={aidx}:v=0:a=1[aout]"
    filter_complex = ";".join(filt + [concat])
    # Windows CreateProcess 命令行上限 32767 字符: 多片段长视频改走脚本文件传递滤镜图
    if len(filter_complex) > 8000:
        fc_script = os.path.join(tmp_dir, "audio_filter.txt")
        with open(fc_script, "w", encoding="utf-8") as f_fc:
            f_fc.write(filter_complex)
        fc_args = ["-filter_complex_script", fc_script]
    else:
        fc_args = ["-filter_complex", filter_complex]

    if already_encoded:
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux (视频 copy, 音频 aac)")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
               "-i", silent_path, "-i", args.video] + fc_args + [
               "-map", "0:v:0", "-map", "[aout]",
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", args.output]
    else:
        if enc == "libx264":
            enc = detect_amf_encoder(args.encoder)
        log(f"🔊 重建音频 ({len(segments)} 段) 并 mux ({enc})")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
               "-i", silent_path, "-i", args.video] + fc_args + [
               "-map", "0:v:0", "-map", "[aout]"] + _enc_args(enc) + [
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
               "-shortest", "-movflags", "+faststart", args.output]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, creationflags=_SUBPROC_FLAGS)
    except subprocess.CalledProcessError as e:
        log("⚠️ 音频 mux 失败, 退回输出无声视频")
        log(f"⚠️ FFmpeg 返回码: {e.returncode}")
        if e.stderr:
            log(f"⚠️ FFmpeg 错误详情:\n{e.stderr}")
        else:
            log(f"⚠️ FFmpeg 无 Stderr 输出, Stdout 详情:\n{e.stdout}")
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
        gz_alt = str(args.raw_detections) + ".gz"
        if os.path.exists(gz_alt):
            log(f"ℹ️ 未找到 {args.raw_detections}, 自动读取压缩日志: {gz_alt}")
            args.raw_detections = gz_alt
        else:
            log(f"❌ 检测日志不存在: {args.raw_detections}")
            sys.exit(1)
    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    log("=" * 60)
    log("🏓 PingPong Highlight Generator (v4.16 物理抛物线净化+平滑)")
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
        corridor_margin=args.corridor_margin, corridor_extend=args.corridor_extend,
        center_dist_ratio=args.center_dist_ratio)
    geo_rebuilt = bool(table_geo.get("rebuilt")) if table_geo else False
    net_center_x_hint = (_get_net_center_x(frames)
                         if (table_geo is None or table_geo.get("net_line") is None) else None)

    if not args.no_player_fx:
        player_boxes, _pc = gather_player_boxes(frames, debug=args.debug)
    else:
        player_boxes = {}

    log("⚪ 提取主球轨迹 (走廊软过滤 + 多球过滤 + 重力抛物线净化) ...")
    ball_cands = gather_ball_candidates(frames, fps, conf_min=args.ball_conf)
    # 及时释放全量帧检测字典, 降低长视频渲染阶段常驻内存占用
    del frames

    ball_cands = filter_static_ball_candidates(
        ball_cands, fps, in_w, in_h,
        min_static_count=args.min_static_ball_count, debug=args.debug)
    corridor = table_geo.get("corridor") if table_geo else None
    if corridor is not None:
        ball_cands = filter_ball_by_corridor(
            ball_cands, corridor, max_drop_ratio=args.corridor_max_drop,
            geo_rebuilt=geo_rebuilt, debug=args.debug)
    if args.court_cluster:
        ball_cands = cluster_ball_by_court(ball_cands, in_w, in_h, debug=args.debug)
    main_track = build_main_ball_track(
        ball_cands, fps, in_w, in_h, table_geo=table_geo, debug=args.debug,
        max_jump_ratio=args.max_jump_ratio, match_algo=args.match_algo)
    g_frame, _ = estimate_gravity_px_per_frame2(
        table_geo, main_track, fps, in_w, in_h, debug=args.debug)
    main_track = clean_ball_track_parabolic(
        main_track, fps, in_w, in_h, g_frame,
        table_geo=table_geo, max_jump_ratio=args.max_jump_ratio, debug=args.debug)
    log(f"   主轨迹净化后点数: {len(main_track)}")

    log("🧩 分段重力抛物线平滑 + 物理弹道补帧 ...")
    filled = interpolate_track(
        main_track, fps, in_w, in_h, debug=args.debug,
        table_geo=table_geo, g_frame=g_frame, smooth=(not args.no_track_smooth))

    log("⚡ 计算逐帧球速表 (速度自适应特效) ...")
    ball_speed_table, (p50, p95) = compute_ball_speed_table(filled, table_geo, fps, in_w, in_h)
    if args.debug:
        log(f"   球速基准: p50={p50:.0f}px/s  p95={p95:.0f}px/s  (共 {len(ball_speed_table)} 帧速度)")

    log("🎾 检测回合与落点/击球事件 ...")
    # 统一落点/击球检测: 一次遍历产出渲染事件流 + 回合用落点半台分布
    events, hl_bounce_halves = detect_bounce_events(filled, table_geo, fps,
                                                    table_edge_tol=args.table_edge_tol)
    rallies = detect_rallies(filled, fps, total_frames, args.raw_detections,
                             table_geo=table_geo, bounce_halves=hl_bounce_halves,
                             net_center_x_hint=net_center_x_hint,
                             out_of_table_timeout=args.out_of_table_timeout,
                             rally_lost_timeout=args.rally_lost_timeout,
                             no_bounce_timeout=args.no_bounce_timeout, debug=args.debug)
    log(f"   回合: {len(rallies)}, 事件: {len(events)}")

    if args.cut:
        segments, rally_starts_by_seg = build_segments(
            rallies, fps, total_frames, args,
            bounce_halves=hl_bounce_halves,
            min_bounce_count=args.min_rally_bounce_count,
            ball_speed_table=ball_speed_table,
            speed_top_n=args.min_rally_speed_top_n)
        if not segments:
            log("❌ 没有检测到有效回合, 无法生成高光。可尝试放宽 --min-rally-frames, "
                "或用 --no-cut 输出整段加特效视频")
            sys.exit(2)
    else:
        # --no-cut: 不裁剪，整段视频加特效（保留捡球/短回合/走动片段）。
        # 回合仍然保留，只用于落点标注的分组边界，让特效与原剪辑模式一致。
        if total_frames <= 0:
            log("❌ --no-cut 需要已知总帧数，但当前视频帧数读取失败")
            sys.exit(1)
        segments = [(0, int(total_frames) - 1)]
        rally_starts_by_seg = {0: sorted(int(r["start_frame"]) for r in rallies)}
        log("✂️ 已关闭剪辑 (--no-cut): 整段视频加特效, 不剔除捡球/短回合片段")

    total_out = sum(e - s + 1 for s, e in segments)
    log(f"✂️  有效片段: {len(segments)} 个, 共 {total_out} 帧 (~{total_out / fps:.1f}s)")
    if args.debug:
        for si, (s, e) in enumerate(segments):
            log(f"    片段{si}: [{s}-{e}]  内含回合起始帧={rally_starts_by_seg.get(si)}")

    # 选取球速最快的精彩回合作为前置
    if args.no_highlight_prepend or not args.cut:
        top_segs, top_rally_starts = [], {}
        if not args.cut:
            log("ℹ️ --no-cut 模式下不做前置精彩集锦（整段即完整内容）")
        else:
            log("ℹ️ 已通过 --no-highlight-prepend 关闭前置精彩回合功能")
    else:
        top_segs, top_rally_starts = _pick_top_speed_segments(
            rallies, ball_speed_table, fps, total_frames,
            top_n=args.highlight_top_n, min_duration=args.highlight_min_duration,
            pre_buffer=args.pre_buffer, post_buffer=args.post_buffer,
            bounce_halves=hl_bounce_halves,
            min_bounce_count=args.highlight_min_bounce_count)

    render(args, segments, rally_starts_by_seg, filled, events, table_geo,
           in_w, in_h, fps, total_frames, ball_speed_table, player_boxes,
           prepend_segments=top_segs or None,
           prepend_rally_starts_by_seg=top_rally_starts,
           black_gap_sec=args.highlight_black_gap)
    log(f"🎉 完成! 输出: {args.output}")


if __name__ == "__main__":
    main()