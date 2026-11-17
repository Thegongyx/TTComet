#!/usr/bin/env python3
"""
真实帧一致性对比: DirectML vs CPU
================================
用视频里的真实帧(而不是随机噪声)跑一次前向, 对比两个 EP 实际检出的框/置信度,
确认 DML 不是"能跑但算错"。

随机输入下 NMS 输出槽位本身就是乱的, 所以必须用真实帧才能判断一致性。

用法:
    webui\.venv\Scripts\python.exe webui\bench\bench_real_frame.py `
        --model models/n960n.onnx --video Videos/xxx.mp4 --frame 300
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img: np.ndarray, new: int = 960, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


CLASS_NAMES = {0: "乒乓球", 1: "其他比赛球", 2: "球员", 3: "球拍", 4: "球桌", 5: "球网"}


def decode(out0: np.ndarray, conf_thr: float = 0.25, max_det: int = 300):
    """output0: [1, 300, 4+1+1+32] (NMS 已内嵌)

    列布局 (实测确认):
        [x1, y1, x2, y2, conf, class_id, 32 × mask_coef]
    注意不是"每类一个分数"。
    """
    pred = out0[0]
    if pred.shape[0] == 0:
        return []
    boxes = pred[:, :4]
    confs = pred[:, 4]
    cls_ids = pred[:, 5].astype(np.int32)
    keep = confs >= conf_thr
    out = []
    for b, c, s in sorted(
            zip(boxes[keep], cls_ids[keep], confs[keep]),
            key=lambda t: -t[2])[:max_det]:
        out.append((int(c), float(s), [float(v) for v in b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/n960n.onnx")
    ap.add_argument("--video", default=None, help="不传则自动找 Videos/ 下第一个视频")
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    video = args.video
    if video is None:
        import glob
        cands = sorted(glob.glob("Videos/*.mp4"))
        if not cands:
            print("❌ 找不到测试视频, 请用 --video 指定")
            return 1
        video = cands[0]

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"❌ 打不开视频: {video}")
        return 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("❌ 读帧失败")
        return 2

    print("=" * 66)
    print("  真实帧一致性对比")
    print("=" * 66)
    print(f"  视频      : {video}  帧 {args.frame}  尺寸 {frame.shape[1]}x{frame.shape[0]}")

    lb, r, left, top = letterbox(frame, 960)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    print(f"  letterbox : 缩放 {r:.4f}, 偏移 ({left},{top})")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    names = {i.name: i for i in sess.get_inputs()}
    inp_name = list(names)[0]
    inp_dtype = np.float16 if "float16" in names[inp_name].type else np.float32
    x = x.astype(inp_dtype)
    print(f"  输入 dtype: {inp_dtype.__name__}")

    outs = {}
    for ep in ["DmlExecutionProvider", "CPUExecutionProvider"]:
        if ep not in ort.get_available_providers():
            continue
        s = ort.InferenceSession(args.model, providers=[ep])
        outs[ep] = s.run(None, {inp_name: x})

    for ep, o in outs.items():
        dets = decode(o[0], args.conf)
        print(f"\n--- {ep}: {len(dets)} 个框 (conf>={args.conf}) ---")
        for c, s, b in dets[:10]:
            nm = CLASS_NAMES.get(c, str(c))
            print(f"    {nm:<6s} conf={s:.4f} xyxy=[{b[0]:7.1f},{b[1]:7.1f},{b[2]:7.1f},{b[3]:7.1f}]")

    if len(outs) == 2:
        d = list(outs.values())
        print("\n--- 原始输出差异 ---")
        for i in range(len(d[0])):
            a = np.asarray(d[0][i], dtype=np.float64)
            b = np.asarray(d[1][i], dtype=np.float64)
            delta = np.abs(a - b)
            print(f"  output{i}: max|Δ| {delta.max():.4e}  mean|Δ| {delta.mean():.4e}")

            if i == 0:
                # 只比较有意义的槽位 (置信度和 > 0.05 的)
                n_cls = a.shape[2] - 4 - 32
                conf_a = a[0, :, 4:4 + n_cls].max(axis=1)
                conf_b = b[0, :, 4:4 + n_cls].max(axis=1)
                active = (conf_a > 0.05) | (conf_b > 0.05)
                if active.any():
                    da = np.abs(a[0][active] - b[0][active])
                    print(f"          有效槽位 {active.sum()} 个, 其中 max|Δ| {da.max():.4e}")
                else:
                    print("          无有效槽位 (两路都没有可信检测)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
