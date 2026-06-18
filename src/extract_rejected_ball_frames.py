#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_rejected_ball_frames.py
=====================================================================
从 pingpong_highlight.py 的过滤逻辑反向收集"错误球"帧，用于标注负样本。

复用 pingpong_highlight.py 的 TableModel 和球提取逻辑，收集以下两类错误球：
    类型A - 整帧被拒绝: 该帧所有球候选都被过滤掉
            (球台外 / otherball / 跳变离群 / 权重过低)
    类型B - 多球中被拒绝的候选: 一帧有多个球检测，highlight.py 选了一个
            但丢弃了另一个（邻台错误球 / 误检测）

输出: 按真实时间顺序命名的 jpg 图片 (rejected_ball_000_fXXXXXX_tsXX.XXXs.jpg)

用法:
    python extract_rejected_ball_frames.py --count 50
"""

import os
import sys
import json
import math
import argparse
import subprocess
import platform
from collections import defaultdict

import numpy as np
import cv2

# --------------------------------------------------------------------------- #
#  常量 (与 pingpong_highlight.py 一致)
# --------------------------------------------------------------------------- #
TABLE_LEN_M  = 2.74
TABLE_WID_M  = 1.525
BALL_MAX_JUMP_FACTOR = 0.45

CLASS_BALL   = "pingpangball"
CLASS_OTHER  = "otherball"
CLASS_TABLE  = "table"
CLASS_NET    = "net"

_IS_WINDOWS = (platform.system() == "Windows")
_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0


# --------------------------------------------------------------------------- #
#  工具函数 (从 pingpong_highlight.py 复制)
# --------------------------------------------------------------------------- #
def bbox_center(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def bbox_size(b):
    return (abs(b[2] - b[0]), abs(b[3] - b[1]))

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------- #
#  TableModel (从 pingpong_highlight.py 精简复制)
# --------------------------------------------------------------------------- #
class TableModel:
    def __init__(self):
        self.table_poly = None
        self.table_bbox = None
        self.net_line = None

    @staticmethod
    def _quad_from_polygon(poly):
        pts = np.array(poly, dtype=np.float32)
        if len(pts) < 4:
            return None
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05, 0.08, 0.1):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    @staticmethod
    def _order_quad(q):
        q = np.array(q, dtype=np.float32)
        s = q.sum(axis=1)
        d = (q[:, 0] - q[:, 1])
        tl = q[np.argmin(s)]
        br = q[np.argmax(s)]
        tr = q[np.argmax(d)]
        bl = q[np.argmin(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _diamond_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        top_y, bot_y = y1, y2
        top_half = (x2 - x1) * 0.30
        bot_half = (x2 - x1) * 0.50
        tl = [cx - top_half, top_y]
        tr = [cx + top_half, top_y]
        br = [cx + bot_half, bot_y]
        bl = [cx - bot_half, bot_y]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def build(self, frames, frame_w, frame_h):
        quads = []
        bboxes = []
        net_bboxes = []
        for rec in frames.values():
            for det in rec.get("detections", []):
                c = det.get("class")
                conf = det.get("confidence", 0)
                if c == CLASS_TABLE and conf >= 0.5:
                    bboxes.append((det["bbox"], conf))
                    poly = det.get("polygon")
                    if poly and len(poly) >= 4:
                        q = self._quad_from_polygon(poly)
                        if q is not None:
                            quads.append((self._order_quad(q), conf))
                elif c == CLASS_NET and conf >= 0.5:
                    net_bboxes.append((det["bbox"], conf))

        if bboxes:
            arr = np.array([b for b, c in bboxes], dtype=np.float32)
            self.table_bbox = tuple(np.median(arr, axis=0).tolist())
        else:
            self.table_bbox = (frame_w * 0.1, frame_h * 0.45,
                               frame_w * 0.9, frame_h * 0.95)

        if quads:
            stacked = np.stack([q for q, c in quads], axis=0)
            self.table_poly = np.median(stacked, axis=0).astype(np.float32)
        else:
            self.table_poly = self._diamond_from_bbox(self.table_bbox)

        if net_bboxes:
            arr = np.array([b for b, c in net_bboxes], dtype=np.float32)
            nb = np.median(arr, axis=0)
            self.net_line = ((float(nb[0]), float(nb[3])),
                             (float(nb[2]), float(nb[1])))
        else:
            tl, tr, br, bl = self.table_poly
            left_mid = (tl + bl) * 0.5
            right_mid = (tr + br) * 0.5
            mid = (left_mid + right_mid) * 0.5
            self.net_line = ((float(mid[0]), float(min(tl[1], tr[1]))),
                             (float(mid[0]), float(max(bl[1], br[1]))))

    def point_in_table(self, pt, margin=0.35):
        if self.table_poly is None:
            return True
        cx = self.table_poly.mean(axis=0)
        expanded = []
        for p in self.table_poly:
            expanded.append(cx + (p - cx) * (1.0 + margin))
        poly = np.array(expanded, dtype=np.float32)
        miny = poly[:, 1].min()
        h = poly[:, 1].max() - miny
        up_pad = h * 1.6
        test_poly = poly.copy()
        for i, p in enumerate(test_poly):
            test_poly[i][1] = p[1] - up_pad if p[1] <= cx[1] else p[1]
        r = cv2.pointPolygonTest(test_poly.astype(np.float32),
                                 (float(pt[0]), float(pt[1])), False)
        return r >= 0


# --------------------------------------------------------------------------- #
#  核心：反向收集被过滤的错误球帧 (两类型)
# --------------------------------------------------------------------------- #
def find_rejected_ball_frames(frames, table, frame_w, frame_h):
    """
    运行与 pingpong_highlight.py 相同的球提取逻辑，收集两类被拒绝的错误球：

    类型A (fully_rejected): 整帧所有球候选都被过滤
    类型B (multi_ball_rejected): 多候选帧中被放弃的候选（邻台干扰球）

    返回 rejected_frames: list of dict
    """
    # ---- 第一遍：收集所有帧的球候选 ----
    # raw[fi] = [(cx, cy, conf, rad, base, class_name, in_table), ...]
    raw = defaultdict(list)
    for fi, rec in frames.items():
        for det in rec.get("detections", []):
            c = det.get("class")
            if c not in (CLASS_BALL, CLASS_OTHER):
                continue
            b = det["bbox"]
            cx, cy = bbox_center(b)
            w, h = bbox_size(b)
            rad = max(2.0, (w + h) * 0.25)
            conf = det.get("confidence", 0)
            base = conf if c == CLASS_BALL else conf * 0.5
            in_table = table.point_in_table((cx, cy), margin=0.35)
            if not in_table:
                base *= 0.15
            raw[fi].append((cx, cy, conf, rad, base, c, in_table))

    # ---- 第二遍：时序贪心匹配 ----
    obs = {}
    accepted_set = set()
    sorted_fi = sorted(raw.keys())
    last_pt = None
    last_fi = None
    last_vel = (0.0, 0.0)
    diag = math.hypot(frame_w, frame_h)
    max_jump = diag * BALL_MAX_JUMP_FACTOR

    # 记录每帧的选中/未选中候选
    accepted_per_frame = {}  # fi -> cand index
    rejected_per_frame = defaultdict(list)  # fi -> [cand, ...]

    for fi in sorted_fi:
        cands = raw[fi]
        if not cands:
            continue
        if last_pt is None:
            best_idx = max(range(len(cands)), key=lambda i: cands[i][4])
        else:
            gap = fi - last_fi
            pred = (last_pt[0] + last_vel[0] * gap,
                    last_pt[1] + last_vel[1] * gap)
            def score(idx):
                c = cands[idx]
                d = dist((c[0], c[1]), pred)
                return c[4] * 2.0 - (d / diag) * 3.0
            best_idx = max(range(len(cands)), key=score)
            best = cands[best_idx]
            if dist((best[0], best[1]), last_pt) > max_jump and best[2] < 0.5:
                # 整帧跳过
                for i, c in enumerate(cands):
                    rejected_per_frame[fi].append((c, "jump_outlier"))
                continue

        best = cands[best_idx]
        cx, cy, conf, rad = best[0], best[1], best[2], best[3]
        obs[fi] = (cx, cy, conf, rad)
        accepted_set.add(fi)
        accepted_per_frame[fi] = best_idx

        # 记录被放弃的其他候选
        for i, c in enumerate(cands):
            if i != best_idx:
                rejected_per_frame[fi].append((c, "not_selected"))

        if last_pt is not None and (fi - last_fi) > 0:
            g = fi - last_fi
            nv = ((cx - last_pt[0]) / g, (cy - last_pt[1]) / g)
            last_vel = (0.6 * last_vel[0] + 0.4 * nv[0],
                        0.6 * last_vel[1] + 0.4 * nv[1])
        last_pt = (cx, cy)
        last_fi = fi

    # ---- 构建结果 ----
    rejected = []

    # 类型A: 整帧被拒绝
    for fi in raw:
        if fi in accepted_set:
            continue
        rec = frames[fi]
        cands = raw[fi]
        timestamp = rec.get("timestamp", 0.0)

        has_pingpang = any(c[5] == CLASS_BALL for c in cands)
        has_other = any(c[5] == CLASS_OTHER for c in cands)
        any_in_table = any(c[6] for c in cands)

        reasons = []
        if not has_pingpang and has_other:
            reasons.append("only_otherball")
        if not any_in_table:
            reasons.append("out_of_table")

        rp = rejected_per_frame.get(fi, [])
        for cand, r in rp:
            if r == "jump_outlier":
                reasons.append("jump_outlier")

        if not reasons:
            reasons.append("low_base")

        rejected.append({
            "category": "A_fully_rejected",
            "frame_idx": rec.get("frame", fi),
            "timestamp": timestamp,
            "reasons": reasons,
            "cand_details": [
                {"class": c[5], "conf": round(c[2], 3), "base": round(c[4], 4),
                 "in_table": c[6], "pos": (round(c[0], 1), round(c[1], 1))}
                for c in cands
            ]
        })

    # 类型B: 多球帧中被放弃的候选（邻台错误球）
    for fi in accepted_set:
        rp = rejected_per_frame.get(fi, [])
        if not rp:
            continue
        rec = frames[fi]
        timestamp = rec.get("timestamp", 0.0)
        # 被接受的候选
        acc_idx = accepted_per_frame.get(fi)
        acc_cand = raw[fi][acc_idx] if acc_idx is not None else None

        for cand, reject_reason in rp:
            cx, cy, conf, rad, base, cls, in_table = cand

            # 细化拒绝原因
            detail_reasons = []
            if cls == CLASS_OTHER:
                detail_reasons.append("only_otherball")
            if not in_table:
                detail_reasons.append("out_of_table")
            if reject_reason == "not_selected":
                detail_reasons.append("not_selected")

            rejected.append({
                "category": "B_multi_ball_rejected",
                "frame_idx": rec.get("frame", fi),
                "timestamp": timestamp,
                "reasons": detail_reasons,
                "cand_details": [{
                    "class": cls, "conf": round(conf, 3), "base": round(base, 4),
                    "in_table": in_table, "pos": (round(cx, 1), round(cy, 1))
                }],
                "accepted_cand": {
                    "class": acc_cand[5], "conf": round(acc_cand[2], 3),
                    "in_table": acc_cand[6],
                    "pos": (round(acc_cand[0], 1), round(acc_cand[1], 1))
                } if acc_cand else None
            })

    # 按时间排序
    rejected.sort(key=lambda x: x["timestamp"])
    return rejected


# --------------------------------------------------------------------------- #
#  选择帧
# --------------------------------------------------------------------------- #
def select_frames(rejected, count):
    """
    从被拒绝的帧中均匀选择指定数量。
    优先从类型A和类型B中按比例选取，保持时间均匀分布。
    """
    if not rejected:
        return []

    if count >= len(rejected):
        return list(range(len(rejected)))

    # 先分类
    type_a = [(i, item) for i, item in enumerate(rejected) if item["category"] == "A_fully_rejected"]
    type_b = [(i, item) for i, item in enumerate(rejected) if item["category"] == "B_multi_ball_rejected"]

    # 按比例分配名额
    if type_a and type_b:
        a_ratio = len(type_a) / len(rejected)
        a_count = max(1, min(len(type_a), int(count * a_ratio)))
        b_count = count - a_count
        b_count = min(b_count, len(type_b))
        a_count = count - b_count
    elif type_a:
        a_count = min(count, len(type_a))
        b_count = 0
    else:
        a_count = 0
        b_count = min(count, len(type_b))

    selected = set()

    # 从类型A均匀选
    if a_count > 0 and type_a:
        step = (len(type_a) - 1) / max(1, a_count - 1) if a_count > 1 else 0
        for i in range(a_count):
            idx = min(int(round(i * step)), len(type_a) - 1)
            selected.add(type_a[idx][0])

    # 从类型B均匀选
    if b_count > 0 and type_b:
        step = (len(type_b) - 1) / max(1, b_count - 1) if b_count > 1 else 0
        for i in range(b_count):
            idx = min(int(round(i * step)), len(type_b) - 1)
            selected.add(type_b[idx][0])

    return sorted(selected)


# --------------------------------------------------------------------------- #
#  ffmpeg 提取帧
# --------------------------------------------------------------------------- #
def extract_frames(rejected, selected_indices, video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, creationflags=_SUBPROC_FLAGS)
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    print(f"  视频时长: {duration:.2f} 秒")

    extracted = []
    for order_i, idx in enumerate(selected_indices):
        item = rejected[idx]
        ts = item["timestamp"]
        frame_idx = item["frame_idx"]
        cat = item["category"]

        # 文件名: rejected_ball_{序号}_{类型}_{帧号}_ts{时间}.jpg
        cat_short = "A" if cat == "A_fully_rejected" else "B"
        output_path = os.path.join(
            output_dir,
            f"rejected_ball_{order_i:03d}_{cat_short}_f{frame_idx:06d}_ts{ts:.3f}s.jpg"
        )

        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            reasons_str = ", ".join(item["reasons"])
            print(f"  [{order_i+1}/{len(selected_indices)}] SKIP (exists) {os.path.basename(output_path)} "
                  f"({size_kb:.1f} KB)")
            extracted.append(output_path)
            continue

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(ts),
            "-frames:v", "1",
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                                creationflags=_SUBPROC_FLAGS)
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            reasons_str = ", ".join(item["reasons"])
            print(f"  [{order_i+1}/{len(selected_indices)}] OK {os.path.basename(output_path)} "
                  f"({size_kb:.1f} KB) | {cat_short}: {reasons_str}")
            extracted.append(output_path)
        else:
            print(f"  [{order_i+1}/{len(selected_indices)}] FAIL (f={frame_idx}, ts={ts:.3f}s)")

    return extracted


# --------------------------------------------------------------------------- #
#  生成说明文件
# --------------------------------------------------------------------------- #
def generate_info(output_dir, rejected, selected_indices):
    info_path = os.path.join(output_dir, "REJECTED_BALL_INFO.txt")

    # 统计
    cat_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    for idx in selected_indices:
        item = rejected[idx]
        cat_counts[item["category"]] += 1
        for r in item["reasons"]:
            reason_counts[r] += 1

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("错误球帧提取记录 (用于标注负样本)\n")
        f.write(f"{'='*60}\n")
        f.write(f"说明: 这些帧包含的球检测会被 pingpong_highlight.py 的过滤逻辑拒绝\n")
        f.write(f"  - 类型A (fully_rejected): 整帧所有球候选都被过滤\n")
        f.write(f"  - 类型B (multi_ball_rejected): 多候选帧中被放弃的错误球\n")
        f.write(f"\n候选总数: {len(rejected)}\n")
        f.write(f"  - 类型A: {sum(1 for x in rejected if x['category']=='A_fully_rejected')}\n")
        f.write(f"  - 类型B: {sum(1 for x in rejected if x['category']=='B_multi_ball_rejected')}\n")
        f.write(f"实际提取: {len(selected_indices)}\n")
        f.write(f"  - 类型A: {cat_counts.get('A_fully_rejected', 0)}\n")
        f.write(f"  - 类型B: {cat_counts.get('B_multi_ball_rejected', 0)}\n")
        f.write(f"\n被拒绝原因分布:\n")
        reason_desc = {
            "out_of_table": "球在球台扩展范围外 (台外误检)",
            "only_otherball": "仅 otherball 类别 (非目标球)",
            "jump_outlier": "位移跳变过大 (时序离群/跟踪错误)",
            "low_base": "置信度/权重过低",
            "not_selected": "多候选中被优先候选压制 (邻台干扰)",
        }
        for r, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {reason_desc.get(r, r)}: {cnt}\n")

        f.write(f"\n帧列表 (按时间顺序):\n")
        f.write(f"{'-'*60}\n")
        for order_i, idx in enumerate(selected_indices):
            item = rejected[idx]
            cat_short = "A" if item["category"] == "A_fully_rejected" else "B"
            fname = f"rejected_ball_{order_i:03d}_{cat_short}_f{item['frame_idx']:06d}_ts{item['timestamp']:.3f}s.jpg"
            f.write(f"{fname}\n")
            f.write(f"  category={item['category']}\n")
            f.write(f"  reasons={item['reasons']}\n")
            f.write(f"  rejected cands={item['cand_details']}\n")
            if "accepted_cand" in item and item["accepted_cand"]:
                f.write(f"  accepted cand={item['accepted_cand']}\n")
            f.write("\n")

    return info_path


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="收集 pingpong_highlight.py 会过滤掉的错误球帧")
    parser.add_argument("--log", "-l", type=str,
                        default="inference_output/raw_detections.jsonl")
    parser.add_argument("--video", "-v", type=str,
                        default="inference_output/video_cfr.mp4")
    parser.add_argument("--count", "-c", type=int, default=50)
    parser.add_argument("--output", "-o", type=str,
                        default="rejected_ball_frames")
    args = parser.parse_args()

    for p, name in [(args.log, "日志"), (args.video, "视频")]:
        if not os.path.isfile(p):
            print(f"ERROR: {name}不存在: {p}")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"  错误球帧提取工具")
    print(f"  (收集 pingpong_highlight.py 会过滤掉的球)")
    print(f"{'='*60}")
    print(f"  日志: {args.log}")
    print(f"  视频: {args.video}")
    print(f"  数量: {args.count}")
    print(f"  输出: {args.output}/")
    print(f"{'='*60}")

    # ---- 读视频元信息 ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频: {args.video}")
        sys.exit(1)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  视频尺寸: {src_w}x{src_h}")

    # ---- 加载日志 ----
    print(f"\n[1/4] 加载检测日志 ...")
    frames = {}
    total_lines = 0
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                rec = json.loads(line)
                fi = int(rec.get("frame", -1))
                if fi >= 0:
                    frames[fi] = rec
            except Exception:
                continue
    print(f"       总行数: {total_lines}, 有效帧: {len(frames)}")

    # ---- 构建球台模型 ----
    print(f"\n[2/4] 构建球台/球网模型 ...")
    table = TableModel()
    table.build(frames, src_w, src_h)
    print(f"       球台 bbox: {tuple(round(v, 1) for v in table.table_bbox)}")

    # ---- 找被拒绝的错误球帧 ----
    print(f"\n[3/4] 扫描错误球帧 (被过滤的球检测) ...")
    rejected = find_rejected_ball_frames(frames, table, src_w, src_h)
    type_a = sum(1 for x in rejected if x["category"] == "A_fully_rejected")
    type_b = sum(1 for x in rejected if x["category"] == "B_multi_ball_rejected")
    print(f"       找到 {len(rejected)} 个被过滤的错误球")
    print(f"         - 类型A (整帧被拒): {type_a}")
    print(f"         - 类型B (多球中被弃): {type_b}")

    if not rejected:
        print("\nWARNING: 未找到被过滤的错误球帧！")
        sys.exit(0)

    # 原因分布统计
    reason_counts = defaultdict(int)
    for item in rejected:
        for r in item["reasons"]:
            reason_counts[r] += 1
    print(f"       原因分布:")
    desc = {
        "out_of_table": "球在球台外",
        "only_otherball": "仅 otherball",
        "jump_outlier": "位移跳变离群",
        "low_base": "权重过低",
        "not_selected": "被优先候选压制",
    }
    for r, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"         {desc.get(r, r)}: {cnt}")

    # ---- 选择 + 提取 ----
    print(f"\n[4/4] 选择 {args.count} 帧并提取 ...")
    actual_count = min(args.count, len(rejected))
    selected = select_frames(rejected, actual_count)
    print(f"       类型A: {sum(1 for i in selected if rejected[i]['category'] == 'A_fully_rejected')}")
    print(f"       类型B: {sum(1 for i in selected if rejected[i]['category'] == 'B_multi_ball_rejected')}")

    extracted = extract_frames(rejected, selected, args.video, args.output)

    # ---- 生成说明 ----
    info_path = generate_info(args.output, rejected, selected)

    print(f"\n{'='*60}")
    print(f"  完成!")
    print(f"  成功提取: {len(extracted)}/{actual_count} 张")
    print(f"  保存位置: {os.path.abspath(args.output)}/")
    print(f"  标注说明: {info_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
