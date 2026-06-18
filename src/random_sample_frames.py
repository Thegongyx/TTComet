#!/usr/bin/env python3
"""
通用随机抽帧脚本 — 支持两种模式:
  A. 基于检测日志: 按条件筛选并均匀随机抽取视频帧 (原有功能)
  B. 直接抽帧 (--no-log): 不依赖检测日志, 根据视频时长均匀分段随机抽帧

支持批量处理: --video 可以是目录, 自动处理目录中所有视频文件。

用法:
    # 直接抽帧 (无需检测日志) — 最简用法
    conda run -n yolo26 python random_sample_frames.py --video match.mp4 --no-log

    # 批量处理目录中所有视频, 每视频抽 20 帧
    conda run -n yolo26 python random_sample_frames.py --video ./Videos/ --no-log --count 20

    # 基于检测日志的原有模式
    conda run -n yolo26 python random_sample_frames.py \
        --log detection_log.json \
        --video match.mp4 \
        --count 50 --ball-only

参数说明:
    --log         detection_log.json 路径 (--no-log 模式下不需要)
    --video       源视频路径, 或包含多个视频的目录
    --count       每个视频抽取数量, 默认 20
    --no-log      直接抽帧模式: 不依赖检测日志, 按视频时长均匀分段随机抽取
    --margin      直接抽帧时首尾跳过秒数 (默认 2.0, 避免黑屏/片尾)
    --ball-only   只抽有球场景 (需要 --log)
    --no-player   只抽无人无球背景帧 (需要 --log)
    --output      输出文件夹名 (默认: random_frames/)
"""

import json
import os
import subprocess
import sys
import random
from pathlib import Path
from argparse import ArgumentParser

_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv',
    '.wmv', '.m4v', '.mpg', '.mpeg', '.ts', '.3gp',
}


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def find_video_files(directory):
    """在目录中查找所有视频文件（仅当前层级，不递归），按文件名排序。"""
    videos = []
    try:
        for entry in sorted(os.scandir(str(directory)), key=lambda e: e.name.lower()):
            if (entry.is_file(follow_symlinks=False)
                    and Path(entry.name).suffix.lower() in _VIDEO_EXTENSIONS):
                videos.append(entry.path)
    except OSError as e:
        print(f"❌ 无法读取目录: {directory}: {e}")
    return videos


def resolve_video_list(video_arg):
    """解析 --video 参数, 返回视频文件路径列表。
    如果是目录则扫描视频文件, 如果是文件则返回单元素列表, 否则报错退出。"""
    p = Path(video_arg)
    if p.is_dir():
        videos = find_video_files(p)
        if not videos:
            print(f"❌ 目录中未找到视频文件 (支持: {', '.join(sorted(_VIDEO_EXTENSIONS))})")
            print(f"   目录: {p}")
            sys.exit(1)
        return videos
    elif p.is_file():
        return [str(p)]
    else:
        print(f"❌ 路径不存在: {video_arg}")
        sys.exit(1)


def get_video_duration(video_file):
    """用 ffprobe 获取视频时长（秒）。失败返回 0.0。"""
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            capture_output=True, text=True, timeout=30
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return float(probe.stdout.strip())
    except Exception:
        pass
    return 0.0


def sample_random_timestamps(duration, count, margin=2.0):
    """将视频时长均匀分为 count 段, 每段内随机选一个时间点。
    返回升序排列的时间戳列表 (float 秒)。
    margin: 首尾各跳过的秒数 (避免黑屏帧/片尾)。"""
    if duration <= 0:
        return []
    if duration <= margin * 2:
        # 视频太短, 就在中间取
        return [duration / 2.0] * min(count, 1)

    effective_start = margin
    effective_end = max(effective_start + 0.1, duration - margin)
    effective_duration = effective_end - effective_start

    if count <= 0:
        return []

    step = effective_duration / count
    timestamps = []
    for i in range(count):
        seg_start = effective_start + i * step
        seg_end = min(effective_start + (i + 1) * step, effective_end)
        if seg_start >= seg_end:
            ts = seg_start
        else:
            ts = random.uniform(seg_start, seg_end)
        timestamps.append(ts)

    return sorted(timestamps)


# ═══════════════════════════════════════════════════════════════
#  基于检测日志的抽帧 (原有逻辑, 基本不变)
# ═══════════════════════════════════════════════════════════════

def load_log(log_path):
    """加载检测日志，返回帧列表"""
    print(f"[1/4] 加载检测日志: {log_path}")
    with open(log_path, 'r') as f:
        data = json.load(f)

    if 'detections_all' not in data:
        for key in ['frames', 'detections']:
            if key in data:
                detections = data[key]
                break
        else:
            print("   ❌ 日志中未找到帧数据（期望 'detections_all'）")
            sys.exit(1)
    else:
        detections = data['detections_all']

    total_frames = len(detections)
    print(f"       总帧数: {total_frames}")

    ball_only = sum(1 for d in detections if d.get('ball_count', 0) > 0)
    player_only = sum(1 for d in detections
                      if d.get('player_count', 0) > 0 and d.get('ball_count', 0) == 0)
    empty = sum(1 for d in detections
                if d.get('ball_count', 0) == 0 and d.get('player_count', 0) == 0)

    print(f"       有球帧: {ball_only} | 仅有人: {player_only} | 空白帧: {empty}")
    return detections


def filter_frames(detections, mode='all'):
    """根据模式过滤候选帧"""
    if mode == 'ball-only':
        candidates = [(i, d) for i, d in enumerate(detections)
                      if d.get('ball_count', 0) > 0]
    elif mode == 'no-player':
        candidates = [(i, d) for i, d in enumerate(detections)
                      if d.get('player_count', 0) == 0 and d.get('ball_count', 0) == 0]
    else:
        candidates = list(enumerate(detections))

    print(f"[2/4] 筛选模式: {mode}")
    print(f"       候选帧数: {len(candidates)}")
    return candidates


def select_random_frames(candidates, count):
    """均匀分段随机抽样，返回选中帧在 candidates 中的索引列表"""
    if not candidates:
        return []

    n = len(candidates)
    if count >= n:
        print(f"[3/4] 候选数 ({n}) <= 请求数 ({count})，全部选中")
        selected_indices = list(range(n))
    else:
        step = n / count
        selected_indices = []
        for bucket in range(count):
            start = int(bucket * step)
            end = min(int((bucket + 1) * step), n)
            if start >= end:
                selected_indices.append(start)
            else:
                chosen = random.randint(start, end - 1)
                selected_indices.append(chosen)
        selected_indices = sorted(set(selected_indices))[:count]

    print(f"       实际选取: {len(selected_indices)} 帧")
    return selected_indices


def extract_frames_log(candidates, selected_indices, video_file, output_dir,
                       video_prefix=None):
    """用 ffmpeg 从视频中提取日志指定的帧。
    video_prefix: 批量模式下的视频名前缀, 用于输出文件名。"""
    os.makedirs(output_dir, exist_ok=True)

    # 获取视频总时长
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    print(f"[4/4] 视频时长: {duration:.2f} 秒")

    prefix = f"{video_prefix}_" if video_prefix else ""

    extracted = []
    for i, idx in enumerate(selected_indices):
        frame_idx, det = candidates[idx]
        ts = det.get('timestamp', 0.0)
        ball_count = det.get('ball_count', 0)
        player_count = det.get('player_count', 0)

        tag_parts = []
        if ball_count > 0:
            tag_parts.append(f"b{ball_count}")
        if player_count == 0 and ball_count == 0:
            tag_parts.append("empty")

        tag_str = "_" + "_".join(tag_parts) if tag_parts else ""
        output_path = os.path.join(
            output_dir,
            f"{prefix}rand_{frame_idx:06d}_ts{ts:.3f}s{tag_str}.jpg")

        if os.path.exists(output_path):
            print(f"  [{i+1}/{len(selected_indices)}] ⏭️ 已存在: {os.path.basename(output_path)}")
            extracted.append((output_path, det))
            continue

        cmd = [
            'ffmpeg', '-y', '-ss', str(ts),
            '-i', video_file,
            '-frames:v', '1', '-q:v', '2',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  [{i+1}/{len(selected_indices)}] ✅ {os.path.basename(output_path)} ({size_kb:.1f} KB)")
            extracted.append((output_path, det))
        else:
            stderr_short = result.stderr[-300:] if result.stderr else "(empty)"
            print(f"  [{i+1}/{len(selected_indices)}] ❌ 失败 (frame={frame_idx}, ts={ts:.3f}s)")

    return extracted


# ═══════════════════════════════════════════════════════════════
#  直接抽帧 (--no-log 模式, 不依赖检测日志)
# ═══════════════════════════════════════════════════════════════

def extract_frames_direct(video_file, timestamps, output_dir, video_prefix,
                          video_index=None, video_total=None):
    """从视频文件中提取指定时间戳的帧 (直接抽帧模式)。

    Args:
        video_file: 视频文件路径
        timestamps: 时间戳列表 (float 秒), 已排序
        output_dir: 输出目录
        video_prefix: 视频名前缀 (用于输出文件名)
        video_index: 批量模式下的当前索引 (1-based), 用于日志
        video_total: 批量模式下的视频总数

    Returns:
        int: 成功提取的帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_tag = f" [{video_index}/{video_total}]" if video_index else ""

    total = len(timestamps)
    extracted = 0

    for i, ts in enumerate(timestamps):
        seq = i + 1
        output_path = os.path.join(
            output_dir,
            f"{video_prefix}_{seq:04d}_ts{ts:.3f}s.jpg")

        if os.path.exists(output_path):
            print(f"  [{seq}/{total}] ⏭️ 已存在: {os.path.basename(output_path)}")
            extracted += 1
            continue

        # ffmpeg: -ss 放在 -i 之前更快 (seek mode)
        cmd = [
            'ffmpeg', '-y', '-ss', str(ts),
            '-i', video_file,
            '-frames:v', '1', '-q:v', '2',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  [{seq}/{total}] ✅ {os.path.basename(output_path)} ({size_kb:.1f} KB)")
            extracted += 1
        else:
            stderr_short = result.stderr[-200:] if result.stderr else "(empty)"
            print(f"  [{seq}/{total}] ❌ 失败 (ts={ts:.3f}s): {stderr_short}")

    return extracted


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = ArgumentParser(description="通用随机抽帧脚本 — 支持检测日志模式 & 直接抽帧模式")

    # 核心参数
    parser.add_argument("--log", "-l", type=str, default=None,
                        help="detection_log.json 路径 (直接抽帧模式不需要)")
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="源视频路径, 或包含多个视频的目录")
    parser.add_argument("--count", "-c", type=int, default=20,
                        help="每个视频抽取数量, 默认 20")

    # 直接抽帧模式
    parser.add_argument("--no-log", action="store_true",
                        help="直接抽帧模式: 不依赖检测日志, 按视频时长均匀分段随机抽取")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="直接抽帧时首尾跳过的秒数 (默认 2.0, 避免黑屏/片尾)")

    # 检测日志模式筛选
    parser.add_argument("--ball-only", action="store_true",
                        help="只抽有球场景 (需要 --log)")
    parser.add_argument("--no-player", action="store_true",
                        help="只抽无人无球背景帧 (需要 --log)")

    # 输出
    parser.add_argument("--output", "-o", type=str, default="random_frames",
                        help="输出文件夹名, 默认: random_frames/")

    # 交互式回退
    parser.add_argument("--interactive", action="store_true",
                        help="交互式询问参数")

    args = parser.parse_args()

    # 交互模式
    if args.interactive:
        args.video = input("🎥 源视频路径 (或目录): ").strip()
        args.no_log = input("🔧 直接抽帧模式? [y/N]: ").strip().lower() == 'y'
        if not args.no_log:
            args.log = input("📁 detection_log.json 路径: ").strip()
        args.count = int(input(f"📊 每视频抽取数量 [{args.count}]: ").strip() or str(args.count))
        if not args.no_log:
            mode = input("🔍 筛选模式 [all/ball-only/no-player] [all]: ").strip().lower() or "all"
            if mode == "ball-only":
                args.ball_only = True
            elif mode == "no-player":
                args.no_player = True
        args.output = input(f"📂 输出文件夹 [{args.output}]: ").strip() or args.output

    # 验证: --no-log 模式下不需要 --log
    if not args.no_log and not args.log:
        parser.error("需要指定 --log (或使用 --no-log 直接抽帧模式)")

    return args


def main():
    args = parse_args()

    # 解析视频列表
    video_list = resolve_video_list(args.video)
    is_batch = len(video_list) > 1

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── 打印信息 ──
    print(f"\n{'=' * 60}")
    print(f"  随机抽帧工具")
    print(f"{'=' * 60}")
    if args.no_log:
        print(f"  模式:     直接抽帧 (无检测日志)")
        print(f"  首尾跳过: {args.margin:.1f} 秒")
    else:
        print(f"  模式:     基于检测日志")
        print(f"  日志:     {args.log}")
        if args.ball_only:
            print(f"  筛选:     ball-only (有球场景)")
        elif args.no_player:
            print(f"  筛选:     no-player (无人无球)")
        else:
            print(f"  筛选:     all (全量)")
    print(f"  每视频:   {args.count} 帧")
    print(f"  输出:     {os.path.abspath(output_dir)}/")
    if is_batch:
        print(f"  视频数:   {len(video_list)} 个 (批量模式)")
    else:
        print(f"  视频:     {video_list[0]}")
    print(f"{'=' * 60}\n")

    # ── 处理每个视频 ──
    total_extracted = 0
    total_requested = 0
    failed_list = []

    for vi, video_path in enumerate(video_list, 1):
        video_stem = Path(video_path).stem

        if is_batch:
            print(f"\n{'─' * 60}")
            print(f"  [{vi}/{len(video_list)}] {Path(video_path).name}")
            print(f"{'─' * 60}")
            prefix = video_stem
            batch_vi, batch_vt = vi, len(video_list)
        else:
            prefix = None
            batch_vi, batch_vt = None, None

        total_requested += args.count

        if args.no_log:
            # ──── 直接抽帧模式 ────
            duration = get_video_duration(video_path)
            if duration <= 0:
                print(f"  ❌ 无法获取视频时长: {Path(video_path).name}")
                failed_list.append(Path(video_path).name)
                continue

            print(f"  视频时长: {duration:.1f} 秒")
            timestamps = sample_random_timestamps(duration, args.count, margin=args.margin)
            if not timestamps:
                print(f"  ❌ 无法生成时间戳")
                failed_list.append(Path(video_path).name)
                continue

            extracted = extract_frames_direct(
                video_path, timestamps, output_dir, video_stem,
                video_index=batch_vi, video_total=batch_vt)
            total_extracted += extracted

            if extracted < args.count:
                failed_list.append(Path(video_path).name)

        else:
            # ──── 基于检测日志模式 ────
            detections = load_log(args.log)

            mode = "all"
            if args.ball_only:
                mode = "ball-only"
            elif args.no_player:
                mode = "no-player"

            candidates = filter_frames(detections, mode)
            if not candidates:
                print("  ⚠️ 未找到符合条件的候选帧！")
                failed_list.append(Path(video_path).name)
                continue

            selected_indices = select_random_frames(candidates, args.count)
            if not selected_indices:
                print("  ⚠️ 无帧可选取！")
                failed_list.append(Path(video_path).name)
                continue

            extracted_frames = extract_frames_log(
                candidates, selected_indices, video_path, output_dir,
                video_prefix=prefix)
            total_extracted += len(extracted_frames)

            if len(extracted_frames) < args.count:
                failed_list.append(Path(video_path).name)

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    if total_extracted == total_requested:
        print(f"  ✅ 全部完成! 共提取 {total_extracted} 张帧")
    else:
        print(f"  ⚠️ 完成, 共提取 {total_extracted}/{total_requested} 张帧")
    print(f"  📁 保存位置: {os.path.abspath(output_dir)}/")
    if failed_list:
        print(f"\n  ⚠️ 部分视频未完全成功:")
        for name in failed_list:
            print(f"    - {name}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
