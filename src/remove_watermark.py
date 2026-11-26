#!/usr/bin/env python3
"""
去除视频中的静态水印 / 叠加文字（台标、赛事条、比分板）。

两种算法:
    delogo   ffmpeg delogo 滤镜, 用区域周边像素插值填补, 速度快, 区域呈模糊块
    inpaint  OpenCV 逐帧 TELEA/NS 修复, 边缘过渡更自然, 速度慢

内置预设区域（基于 1920x1080 的常见转播布局, 可直接改 _PRESETS）:
    logo      右上角台标
    overlay   logo + 底部赛事信息条（默认）
    all       overlay + 左下角比分牌（比分是比赛信息, 非必要别动）
    none      不用预设, 只保留 --region 手动指定的区域

用法:
    # 默认: preset=overlay(右上台标 + 底部赛事条), delogo 算法, 保留比分牌
    # 预设坐标基于 1920x1080, 换成 720p/4K 会自动等比缩放
    python src/remove_watermark.py --video Videos/fb1080.mp4

    # 只去右上角台标
    python src/remove_watermark.py --video Videos/fb1080.mp4 --preset logo

    # 连比分牌一起去掉(比赛信息会消失)
    python src/remove_watermark.py --video Videos/fb1080.mp4 --preset all

    # 自动检测水印位置(适合换平台/换赛事的未知水印)
    # --detect 只输出候选框和可疑度, 不动视频; 确认后加 --detect-apply 才处理
    python src/remove_watermark.py --video in.mp4 --detect
    python src/remove_watermark.py --video in.mp4 --detect --detect-apply

    # 自定义区域(可重复), 不用预设; 坐标按实际像素
    python src/remove_watermark.py --video in.mp4 --preset none --region 100,100,200,50
    # 若坐标是按 1920x1080 量的, 加 --region-in-ref 让它跟着缩放
    python src/remove_watermark.py --video in.mp4 --preset none --region-in-ref --region 100,100,200,50

    # 精修算法 + 前 30 秒试跑
    python src/remove_watermark.py --video in.mp4 --method inpaint --duration 30

    # 只导出处理前后对比静帧(不编码视频), 先看效果
    python src/remove_watermark.py --video in.mp4 --preview 3,900,2100 --dry-run

参数说明:
    --video       源视频路径
    --output      输出视频路径（默认: 源视频同目录 <原名>_nowatermark.mp4）
    --method      delogo / inpaint（默认 delogo）
    --preset      logo / overlay / all / none（默认 overlay, 保留比分牌）
    --region      额外区域 "x,y,w,h", 可重复
    --ref-size    预设坐标的参考分辨率（默认 1920x1080）, 自动按实际分辨率等比缩放
    --region-in-ref  --region 的坐标也按参考分辨率缩放（默认按实际像素）
    --detect      多帧时序统计自动检测静态叠加区域（只输出候选, 不处理视频; 忽略 --preset）
    --detect-apply   直接把检测结果用于去水印（可用 --region 追加手动区域）
    --detect-frames / --detect-width    抽帧数量 / 检测用灰度图宽度（默认 48 / 960）
    --detect-std / --detect-bright      时间标准差上限 / 最小亮度下限（默认 10 / 140）
    --detect-min-area / --detect-max-area   连通域面积下限(检测分辨率) / 占画面比例上限
    --detect-min-fill / --detect-max-fill   连通域填充率区间（默认 0.08~0.78, 排掉纯色块）
    --expand      区域外扩像素, 盖住水印抗锯齿边（默认 3）
    --crf         libx264 质量, 越小越清晰（默认 18）
    --encoder     视频编码器（默认 libx264; AMD 可用 h264_amf 加速）
    --x264-preset  x264 速度档（默认 medium, 可选 fast/slower 等）
    --inpaint-radius  OpenCV inpaint 采样半径（默认 4）
    --inpaint-method  telea / ns（默认 telea）
    --start / --duration  只处理指定时间区间
    --preview     逗号分隔的时间点(秒), 导出前后对比图到 tmp/
    --dry-run     只打印/导出预览, 不编码视频
    --list-presets  打印内置预设坐标后退出

自动检测的适用范围:
    能检出 —— 纯色/不透明的台标、字幕条、角标（时间上位置和亮度都恒定）
    检不出 —— 半透明叠加（如比分牌）、滚动/闪烁/漂移的水印、与背景同亮度的水印
    检不出时用 --region 手动指定, 或放宽 --detect-bright / --detect-std
"""

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

_TMP_ROOT = Path(__file__).resolve().parent.parent / "tmp"

# 预设区域 (x, y, w, h, 说明), 基于 1920x1080。
# 若换成别的分辨率, 用 --region 手动指定, 或按比例换算。
# 注意: 比分牌属于比赛信息, 默认不处理; 需要抹掉时显式用 --preset all
_PRESETS = {
    "logo": [
        (1588, 38, 292, 74, "右上角台标"),
    ],
    "overlay": [
        (1588, 38, 292, 74, "右上角台标"),
        (1170, 950, 650, 78, "底部赛事信息条"),
    ],
    "all": [
        (1588, 38, 292, 74, "右上角台标"),
        (1170, 950, 650, 78, "底部赛事信息条"),
        (84, 884, 462, 150, "左下角比分牌（比赛信息, 谨慎去除）"),
    ],
    "none": [],
}


# ────────────────────────────────────────────────────────────────────────────
#  基础工具
# ────────────────────────────────────────────────────────────────────────────

def run(cmd, quiet=False):
    if not quiet:
        print("> " + " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        print(f"! 命令失败 (exit={proc.returncode})")
        sys.exit(proc.returncode)
    return proc


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("! 未找到 ffmpeg / ffprobe, 请先安装并加入 PATH")
        sys.exit(1)


def probe_media(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(f"! 无法读取媒体信息: {path}")
        print(out.stderr.strip())
        sys.exit(1)
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        print("! 输入文件里没有视频流")
        sys.exit(1)

    # 平均帧率优先, 避免容器里的 r_frame_rate 过于乐观
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = video.get(key) or "0/0"
        try:
            num, den = raw.split("/")
            val = float(num) / float(den) if float(den) else 0.0
        except Exception:
            val = 0.0
        if val > 0:
            fps = val
            break
    if fps <= 0:
        fps = 25.0

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "has_audio": audio is not None,
    }


def normalize_region(x, y, w, h, width, height):
    """裁进画面内, 并保证 delogo 要求的边界（不能贴边）"""
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    x = max(1, x)
    y = max(1, y)
    if x + w > width - 1:
        w = width - 1 - x
    if y + h > height - 1:
        h = height - 1 - y
    if w < 2 or h < 2:
        print(f"! 区域无效 (越界): x={x} y={y} w={w} h={h}")
        sys.exit(1)
    return x, y, w, h


def parse_ref_size(text):
    """解析 '1920x1080' 形式的参考分辨率"""
    raw = str(text).lower().replace(" ", "")
    for sep in ("x", "*", ",", ":"):
        if sep in raw:
            a, b = raw.split(sep, 1)
            try:
                w, h = int(a), int(b)
                if w > 0 and h > 0:
                    return w, h
            except ValueError:
                pass
            break
    print(f"! --ref-size 格式应为 WxH（如 1920x1080）, 收到: {text}")
    sys.exit(1)


def merge_boxes(boxes, gap):
    """把同一行里水平相邻的连通域合并成一条（文字被逐字拆分时用）"""
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            used[i] = True
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_y <= 0 or overlap_y < 0.5 * min(h1, h2):
                    continue
                gap_x = max(x2 - (x1 + w1), x1 - (x2 + w2))
                if gap_x > gap:
                    continue
                nx1, ny1 = min(x1, x2), min(y1, y2)
                nx2, ny2 = max(x1 + w1, x2 + w2), max(y1 + h1, y2 + h2)
                x1, y1, w1, h1 = nx1, ny1, nx2 - nx1, ny2 - ny1
                used[j] = True
                changed = True
            merged.append((x1, y1, w1, h1))
        boxes = merged
    return boxes


def drop_contained_boxes(boxes):
    """去掉被更大的框基本包含掉的框（合并后常出现）"""
    out = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        area1 = w1 * h1
        contained = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j or w2 * h2 <= area1:
                continue
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if ix * iy >= 0.9 * area1:
                contained = True
                break
        if not contained:
            out.append((x1, y1, w1, h1))
    return out


def detect_overlay_regions(video, info, args):
    """用多帧时序统计自动检测静态不透明叠加区域（台标 / 字幕条）

    原理：水印在时间上是恒定的 —— 逐像素对多帧取最小值仍然偏亮（叠加了白色），
    同时时间标准差极低（不随画面变化）。场景里静止的深色物体达不到这个亮度。
    检测在缩小的灰度图上做，最后把坐标换算回原始分辨率。

    局限：半透明叠加（如比分牌）和动态/滚动水印检测不到。
    """
    import cv2

    width, height = info["width"], info["height"]
    duration = info["duration"]
    n = max(int(args.detect_frames), 4)

    det_w = max(2, min(int(width), int(args.detect_width)))
    det_w -= det_w % 2
    det_h = max(2, int(round(height * det_w / width)))
    det_h -= det_h % 2
    scale = width / det_w

    fps = n / duration if duration > 0 else 1.0
    print(f"[detect] 抽取 {n} 帧做时序统计（灰度 {det_w}x{det_h}, fps={fps:.4f}）")
    cmd = ["ffmpeg", "-v", "error", "-i", str(video),
           "-vf", f"fps={fps:.6f},scale={det_w}:{det_h}:flags=bilinear,format=gray",
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print("! 抽帧失败")
        print(proc.stderr.decode("utf-8", "ignore")[:600])
        sys.exit(1)

    frame_size = det_w * det_h
    count = len(proc.stdout) // frame_size
    if count < 3:
        print(f"! 有效帧只有 {count} 帧, 不足以做时序统计")
        sys.exit(1)
    frames = np.frombuffer(proc.stdout[:count * frame_size], np.uint8)
    frames = frames.reshape(count, det_h, det_w)

    mn = frames.min(0)
    std = frames.std(0)

    cand = ((std < args.detect_std) & (mn > args.detect_bright)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, det_w // 160), 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)

    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    max_area = det_w * det_h * args.detect_max_area
    boxes = []
    for i in range(1, n_lab):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if area < args.detect_min_area or w < 8 or h < 6:
            continue
        if w * h > max_area:
            continue
        # 填充率: 文字类水印通常 15%~70%, 纯色块接近 100%, 噪点很低
        fill = area / float(w * h)
        if fill < args.detect_min_fill or fill > args.detect_max_fill:
            continue
        boxes.append((x, y, w, h))

    print(f"[detect] 候选连通域 {len(boxes)} 个（std<{args.detect_std} "
          f"min>{args.detect_bright} 填充率 {args.detect_min_fill}~{args.detect_max_fill}）")
    if not boxes:
        return []

    boxes = drop_contained_boxes(merge_boxes(boxes, args.detect_merge_gap))

    scored = []
    for (x, y, w, h) in boxes:
        sub = cand[y:y + h, x:x + w] > 0
        fill = float(sub.mean())
        r_std = float(std[y:y + h, x:x + w][sub].mean()) if sub.any() else 99.0
        r_min = float(mn[y:y + h, x:x + w][sub].mean()) if sub.any() else 0.0
        aspect = w / float(max(h, 1))
        # 越"时间恒定 + 亮度高 + 横向长条"越像叠加文字; 竖长/方形多半是场景物体
        aspect_factor = min(max(aspect / 3.0, 0.35), 1.6)
        score = (1.0 / (1.0 + r_std)) * (r_min / 255.0) * aspect_factor
        scored.append((score, x, y, w, h, r_std, r_min, fill, aspect))

    # 按可疑度排序, 同时把遮挡相近的重复框压掉
    scored.sort(key=lambda t: -t[0])
    out = []
    for (score, x, y, w, h, r_std, r_min, fill, aspect) in scored:
        out.append((x * scale, y * scale, w * scale, h * scale, score,
                    f"score={score:.3f} std={r_std:.1f} min={r_min:.0f} "
                    f"fill={fill:.0%} aspect={aspect:.1f}"))
        if len(out) >= args.detect_max_regions:
            break
    return out


def report_detections(detected, score_ratio=0.5):
    """打印候选区域表, 并给出可直接复制的命令"""
    print(f"\n[detect] 候选区域 {len(detected)} 个（按可疑度从高到低）:")
    best_score = detected[0][4] if detected else 0.0
    for idx, (x, y, w, h, score, extra) in enumerate(detected, 1):
        mark = ""
        if best_score > 0:
            mark = "  <= 会处理" if score >= best_score * score_ratio else "  (疑似场景物体, 跳过)"
        print(f"  #{idx}  x={int(x):4d} y={int(y):4d} w={int(w):4d} h={int(h):4d}  {extra}{mark}")
    best = detected[0]
    print("\n最像叠加文字的是 #1, 只处理它:")
    print(f"  python src/remove_watermark.py --video <视频> --preset none "
          f"--region {int(best[0])},{int(best[1])},{int(best[2])},{int(best[3])}")
    print(f"确认候选都对再加 --detect-apply（只处理 score >= {score_ratio:.0%} × 最高分的候选）。")


def collect_regions(args, width, height, detected=None):
    """汇总待处理区域: 自动检测 / 预设 / 手动 --region

    预设坐标基于 --ref-size（默认 1920x1080）, 会按实际分辨率等比缩放;
    --region 默认按实际像素解释, 加 --region-in-ref 则同样按参考分辨率缩放。
    """
    ref_w, ref_h = parse_ref_size(args.ref_size)
    sx, sy = width / ref_w, height / ref_h
    scaled = abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6

    regions = []
    if detected:
        # 自动检测结果已是实际像素坐标
        regions.extend(detected)
    else:
        for (x, y, w, h, name) in _PRESETS.get(args.preset, []):
            regions.append((x * sx, y * sy, w * sx, h * sy, name))

    for raw in args.region or []:
        parts = [p.strip() for p in str(raw).replace(" ", ",").split(",") if p.strip()]
        if len(parts) != 4:
            print(f"! --region 需要 4 个数字 x,y,w,h, 收到: {raw}")
            sys.exit(1)
        x, y, w, h = (float(p) for p in parts)
        if args.region_in_ref:
            x, y, w, h = x * sx, y * sy, w * sx, h * sy
        regions.append((x, y, w, h, f"自定义区域 {raw}"))

    if not regions:
        print("! 没有任何待处理区域, 用 --detect / --preset / --region 指定")
        sys.exit(1)

    if scaled:
        print(f"  参考分辨率 {ref_w}x{ref_h} -> 实际 {width}x{height}, "
              f"坐标按 x{sx:.4f} / y{sy:.4f} 缩放")

    out = []
    for (x, y, w, h, name) in regions:
        x -= args.expand
        y -= args.expand
        w += args.expand * 2
        h += args.expand * 2
        out.append(normalize_region(x, y, w, h, width, height) + (name,))
    return out


def build_delogo_filter(regions):
    return ",".join(
        f"delogo=x={x}:y={y}:w={w}:h={h}" for (x, y, w, h, _) in regions
    )


def encoder_args(args):
    """libx264 走 crf, AMF/NVENC 走码率"""
    if args.encoder == "libx264":
        return ["-c:v", "libx264", "-crf", str(args.crf), "-preset", args.x264_preset]
    if args.encoder.endswith("_amf"):
        return ["-c:v", args.encoder, "-b:v", args.video_bitrate, "-quality", "quality"]
    if args.encoder.endswith("_nvenc"):
        return ["-c:v", args.encoder, "-cq", str(args.crf), "-b:v", "0"]
    return ["-c:v", args.encoder, "-crf", str(args.crf)]


# ────────────────────────────────────────────────────────────────────────────
#  delogo 后端
# ────────────────────────────────────────────────────────────────────────────

def do_delogo(args, video, output, info, regions):
    filt = build_delogo_filter(regions)
    print(f"[delogo] 区域: {regions}")
    print(f"[delogo] filter: {filt}")

    if args.dry_run:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "warning", "-stats", "-y"]
    if args.start and args.start > 0:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", str(video)]
    if args.duration and args.duration > 0:
        cmd += ["-t", f"{args.duration:.3f}"]
    cmd += ["-map", "0:v:0"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    cmd += ["-vf", filt]
    cmd += encoder_args(args)
    cmd += ["-movflags", "+faststart", str(output)]
    run(cmd)
    return output


# ────────────────────────────────────────────────────────────────────────────
#  inpaint 后端
# ────────────────────────────────────────────────────────────────────────────

def build_mask(width, height, regions):
    mask = np.zeros((height, width), np.uint8)
    for (x, y, w, h, _) in regions:
        mask[y:y + h, x:x + w] = 255
    return mask


def do_inpaint(args, video, output, info, regions):
    import cv2

    width, height, fps = info["width"], info["height"], info["fps"]
    mask = build_mask(width, height, regions)
    flag = cv2.INPAINT_TELEA if args.inpaint_method == "telea" else cv2.INPAINT_NS
    frame_bytes = width * height * 3

    print(f"[inpaint] {width}x{height} @ {fps:.3f}fps, 区域 {regions}")
    print(f"[inpaint] radius={args.inpaint_radius} method={args.inpaint_method}")

    if args.dry_run:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)

    # 读: ffmpeg 解码原始帧到管道
    dec = ["ffmpeg", "-v", "error"]
    if args.start and args.start > 0:
        dec += ["-ss", f"{args.start:.3f}"]
    dec += ["-i", str(video)]
    if args.duration and args.duration > 0:
        dec += ["-t", f"{args.duration:.3f}"]
    dec += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    read_proc = subprocess.Popen([str(c) for c in dec],
                                 stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    # 写: 管道喂帧给 ffmpeg 编码, 音轨直接从源视频复制
    enc = ["ffmpeg", "-v", "warning", "-stats", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-"]
    if info["has_audio"]:
        if args.start and args.start > 0:
            enc += ["-ss", f"{args.start:.3f}"]
        enc += ["-i", str(video)]
        if args.duration and args.duration > 0:
            enc += ["-t", f"{args.duration:.3f}"]
        enc += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"]
    enc += encoder_args(args)
    enc += ["-movflags", "+faststart", str(output)]
    write_proc = subprocess.Popen([str(c) for c in enc], stdin=subprocess.PIPE)

    total = int((args.duration if args.duration and args.duration > 0
                 else max(info["duration"] - args.start, 0.0)) * fps)
    done = 0
    try:
        while True:
            buf = read_proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            fixed = cv2.inpaint(frame, mask, args.inpaint_radius, flag)
            write_proc.stdin.write(fixed.tobytes())
            done += 1
            if done % 500 == 0:
                pct = done / max(total, 1) * 100
                print(f"  inpaint {done}/{total} 帧 ({pct:.1f}%)", flush=True)
    finally:
        read_proc.stdout.close()
        read_proc.wait()
        write_proc.stdin.close()
        write_proc.wait()

    if write_proc.returncode != 0:
        print(f"! 编码器退出码 {write_proc.returncode}")
        sys.exit(write_proc.returncode)
    print(f"  inpaint 完成, 共 {done} 帧")
    return output


# ────────────────────────────────────────────────────────────────────────────
#  静帧预览对比
# ────────────────────────────────────────────────────────────────────────────

def export_preview(args, video, info, regions, times):
    """对每个时间点导出「原图 | 处理后 | 区域放大」对比图"""
    import cv2

    out_dir = _TMP_ROOT / "watermark_preview" / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    filt = build_delogo_filter(regions)
    mask = build_mask(info["width"], info["height"], regions)
    flag = cv2.INPAINT_TELEA if args.inpaint_method == "telea" else cv2.INPAINT_NS

    for t in times:
        orig_png = out_dir / f"t{t:g}_orig.png"
        proc_png = out_dir / f"t{t:g}_proc.png"

        base = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "1"]
        run(base + [str(orig_png)], quiet=True)

        if args.method == "inpaint":
            # inpaint 直接在 python 侧处理单帧
            orig = cv2.imread(str(orig_png))
            if orig is None:
                print(f"! 时间点 {t}s 抽帧失败")
                continue
            cv2.imwrite(str(proc_png),
                        cv2.inpaint(orig, mask, args.inpaint_radius, flag))
        else:
            run(base + ["-vf", filt, str(proc_png)], quiet=True)

        orig = cv2.imread(str(orig_png))
        proc = cv2.imread(str(proc_png))
        if orig is None or proc is None:
            print(f"! 时间点 {t}s 抽帧失败")
            continue

        # 上: 原图 / 中: 处理后
        sheet = np.vstack([orig, proc])
        # 下: 每个区域的处理前后放大并排
        crops = []
        for (x, y, w, h, _name) in regions:
            pad = 12
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(info["width"], x + w + pad)
            y1 = min(info["height"], y + h + pad)
            oc = orig[y0:y1, x0:x1]
            pc = proc[y0:y1, x0:x1]
            scale = max(1.0, 360.0 / max(oc.shape[1], 1))
            oc = cv2.resize(oc, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pc = cv2.resize(pc, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            pair = np.vstack([oc, pc])
            if pair.shape[1] < 740:
                pad_w = 740 - pair.shape[1]
                pair = cv2.copyMakeBorder(pair, 0, 0, 0, pad_w,
                                          cv2.BORDER_CONSTANT, value=(20, 20, 20))
            crops.append(pair)
        if crops:
            # 各区域高度不同, hstack 前先补齐到最高
            max_h = max(c.shape[0] for c in crops)
            crops = [
                cv2.copyMakeBorder(c, 0, max_h - c.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
                for c in crops
            ]
            strip = np.hstack(crops)
            if strip.shape[1] > sheet.shape[1]:
                scale = sheet.shape[1] / strip.shape[1]
                strip = cv2.resize(strip, None, fx=scale, fy=scale)
            elif strip.shape[1] < sheet.shape[1]:
                strip = cv2.copyMakeBorder(strip, 0, 0, 0,
                                           sheet.shape[1] - strip.shape[1],
                                           cv2.BORDER_CONSTANT, value=(20, 20, 20))
            sheet = np.vstack([sheet, strip])

        out_png = out_dir / f"t{t:g}_compare.png"
        cv2.imwrite(str(out_png), sheet)
        print(f"  预览: {out_png}")

    return out_dir


# ────────────────────────────────────────────────────────────────────────────
#  入口
# ────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = ArgumentParser(
        description="去除视频静态水印/叠加文字 (delogo 或 OpenCV inpaint)",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--video", help="源视频路径")
    p.add_argument("--output", help="输出视频路径")
    p.add_argument("--method", choices=["delogo", "inpaint"], default="delogo")
    p.add_argument("--preset", choices=list(_PRESETS.keys()), default="overlay")
    p.add_argument("--region", action="append", metavar="x,y,w,h")
    p.add_argument("--ref-size", default="1920x1080",
                   help="预设坐标的参考分辨率, 实际分辨率不同时自动等比缩放")
    p.add_argument("--region-in-ref", action="store_true",
                   help="--region 的坐标也按 --ref-size 缩放（默认按实际像素）")
    p.add_argument("--expand", type=int, default=3, help="区域外扩像素")
    p.add_argument("--detect", action="store_true",
                   help="自动检测静态叠加区域（只输出候选, 不处理视频）")
    p.add_argument("--detect-apply", action="store_true",
                   help="直接把检测到的区域用于去水印（可配合 --detect 或单独用）")
    p.add_argument("--detect-frames", type=int, default=48, help="检测抽帧数")
    p.add_argument("--detect-width", type=int, default=960, help="检测时的灰度图宽度")
    p.add_argument("--detect-std", type=float, default=10.0, help="时间标准差上限")
    p.add_argument("--detect-bright", type=float, default=140.0, help="最小亮度下限")
    p.add_argument("--detect-min-area", type=int, default=300, help="连通域最小面积(检测分辨率下)")
    p.add_argument("--detect-max-area", type=float, default=0.06, help="连通域最大面积占比")
    p.add_argument("--detect-min-fill", type=float, default=0.08, help="连通域最小填充率")
    p.add_argument("--detect-max-fill", type=float, default=0.78, help="连通域最大填充率")
    p.add_argument("--detect-max-regions", type=int, default=6, help="最多输出几个区域")
    p.add_argument("--detect-score-ratio", type=float, default=0.5,
                   help="--detect-apply 时只处理 score >= 该比例 × 最高分 的候选")
    p.add_argument("--detect-merge-gap", type=int, default=24, help="同行连通域合并的最大水平间距")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--encoder", default="libx264")
    p.add_argument("--x264-preset", default="medium")
    p.add_argument("--video-bitrate", default="10M")
    p.add_argument("--inpaint-radius", type=int, default=4)
    p.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--preview", default="", help="逗号分隔的时间点(秒)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_presets:
        ref_w, ref_h = parse_ref_size(args.ref_size)
        print(f"# 坐标基于参考分辨率 {ref_w}x{ref_h}, 实际分辨率不同时自动等比缩放")
        for name, regs in _PRESETS.items():
            print(f"[{name}]")
            if not regs:
                print("  (无)")
            for (x, y, w, h, desc) in regs:
                print(f"  x={x:4d} y={y:4d} w={w:4d} h={h:4d}  {desc}")
        return

    check_ffmpeg()

    if not args.video:
        args.video = input("源视频路径: ").strip().strip('"')
    video = Path(args.video)
    if not video.is_file():
        print(f"! 视频不存在: {video}")
        sys.exit(1)

    info = probe_media(video)

    detected = None
    if args.detect or args.detect_apply:
        candidates = detect_overlay_regions(video, info, args)
        if not candidates:
            print("! 没检出静态叠加区域")
            print("  可放宽阈值: --detect-bright 110 / --detect-std 16 / --detect-min-area 150")
            print("  或改用 --preset / --region 手动指定")
            sys.exit(1)
        report_detections(candidates, args.detect_score_ratio)
        if not args.detect_apply and not args.preview:
            print("\n(仅检测, 未处理视频；确认后加 --detect-apply 或按上面命令手动指定区域)")
            return

        # 只保留相对可疑度够高的候选, 避免把背景物体一起抹掉
        best_score = candidates[0][4]
        kept = [c for c in candidates if c[4] >= best_score * args.detect_score_ratio]
        if len(kept) < len(candidates):
            print(f"[detect] 按可疑度过滤: 保留 {len(kept)}/{len(candidates)} 个候选"
                  f"（阈值 {args.detect_score_ratio:.0%} × {best_score:.3f}）")
        detected = [(x, y, w, h, f"自动检测 {extra}")
                    for (x, y, w, h, _score, extra) in kept]

    regions = collect_regions(args, info["width"], info["height"], detected=detected)

    print(f"输入: {video}")
    print(f"  画面: {info['width']}x{info['height']} @ {info['fps']:.3f}fps  "
          f"时长 {info['duration']:.1f}s  音轨 {info['has_audio']}")
    print(f"  方法: {args.method}  "
          f"{'来源: 自动检测' if detected else '预设: ' + args.preset}")
    for (x, y, w, h, name) in regions:
        print(f"  区域: x={x} y={y} w={w} h={h}  {name}")

    times = []
    if args.preview:
        times = [float(v) for v in args.preview.replace(" ", "").split(",") if v]
        export_preview(args, video, info, regions, times)

    output = Path(args.output) if args.output else \
        video.with_name(f"{video.stem}_nowatermark{video.suffix}")
    if output.resolve() == video.resolve():
        print("! 输出路径不能与输入相同")
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: 不编码视频")
        if args.method == "delogo":
            print("filter: " + build_delogo_filter(regions))
        return

    handler = do_delogo if args.method == "delogo" else do_inpaint
    result = handler(args, video, output, info, regions)

    if result and result.exists():
        size_mb = result.stat().st_size / 1024 / 1024
        print(f"\n完成: {result}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
