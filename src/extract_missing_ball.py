#!/usr/bin/env python3
"""
漏检帧提取工具 — 从 YOLO 检测日志中找出"球运动跳变"的帧并提取图片。

查找条件：上一帧 ball_count > 0 AND 当前帧 ball_count == 0 AND 下一帧 ball_count > 0
即球在连续两帧之间"消失了一帧又出现"，属于典型的漏检样本。

用法:
    # 交互式（不传任何参数）
    conda run -n yolo26 python3 extract_missing_ball.py
    
    # 命令行参数
    conda run -n yolo26 python3 extract_missing_ball.py \
        --log /path/to/detection_log.jsonl \
        --video /path/to/video.mp4 \
        --count 50 \
        --output custom_frames/

参数说明:
    --log       detection_log.jsonl 路径（JSONL 格式，必需）
    --video     源视频文件路径（必需）
    --count     提取数量，默认 20；--all 表示全部
    --output    输出文件夹名（默认: missing_ball_frames/）
"""

import json
import os
import subprocess
import sys


def parse_args():
    """解析命令行参数，支持交互式询问"""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="漏检帧提取工具")
    parser.add_argument("--log", "-l", type=str, help="detection_log.jsonl 路径（JSONL 格式，必需）")
    parser.add_argument("--video", "-v", type=str, help="源视频文件路径（必需）")
    parser.add_argument("--count", "-c", type=int, default=20, help="提取数量，默认20")
    parser.add_argument("--all", action="store_true", dest="use_all", help="提取所有候选帧")
    parser.add_argument("--output", "-o", type=str, default="missing_ball_frames", help="输出文件夹名")
    
    args = parser.parse_args()
    
    # 如果没传 --log 和 --video，进入交互式询问
    if not args.log or not args.video:
        log_path = input("📁 detection_log.jsonl 路径: ").strip()
        video_path = input("🎥 源视频文件路径: ").strip()
        
        if not os.path.exists(log_path):
            print(f"❌ 日志不存在: {log_path}")
            sys.exit(1)
        if not os.path.exists(video_path):
            print(f"❌ 视频不存在: {video_path}")
            sys.exit(1)
        
        count_input = input("📊 提取数量（默认20，输入 all 全部）[20]: ").strip()
        output_dir = input("📂 输出文件夹名 [missing_ball_frames]: ").strip() or "missing_ball_frames"
        
        if count_input == 'all':
            with open(log_path) as f:
                total_lines = sum(1 for _ in f)
                count = total_lines * 2
        elif count_input:
            count = int(count_input)
    else:
        log_path = args.log
        video_path = args.video
        output_dir = args.output
        
        if args.use_all:
            with open(log_path) as f:
                total_lines = sum(1 for _ in f)
                count = total_lines * 2
        else:
            count = args.count
    
    # 验证路径存在
    if not os.path.exists(log_path):
        print(f"❌ 日志不存在: {log_path}")
        sys.exit(1)
    if not os.path.exists(video_path):
        print(f"❌ 视频不存在: {video_path}")
        sys.exit(1)
    
    return log_path, video_path, count, output_dir


def load_log(log_path):
    """加载检测日志（JSONL 格式，每行一个 JSON）"""
    print(f"[1/3] 加载检测日志: {log_path}")
    detections = []
    total_lines = 0
    skipped = 0
    ball_frames = 0
    empty_frames = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 5:
                    print(f"       ⚠️ 跳过第 {total_lines} 行（JSON 解析失败）")
                continue
            
            # ball_count 从 counts.pingpangball 获取，不存在则默认 0
            bc = d.get('counts', {}).get('pingpangball', 0)
            if bc > 0:
                ball_frames += 1
            else:
                empty_frames += 1
            
            detections.append({
                'ball_count': bc,
                'timestamp': d.get('timestamp', 0.0),
                'frame': d.get('frame', 0),
            })
    
    print(f"       总行数: {total_lines} (跳过 {skipped} 行)")
    print(f"       有效帧: {len(detections)}")
    print(f"       有球帧: {ball_frames} | 无球帧: {empty_frames}")
    
    return detections


def find_missing_ball_frames(detections):
    """找出所有漏检帧：上一帧有球，当前帧无球，下一帧也有球"""
    print("[2/3] 扫描漏检帧...")
    candidates = []
    
    for i in range(1, len(detections) - 1):
        prev_has_ball = detections[i - 1]['ball_count'] > 0
        curr_no_ball = detections[i]['ball_count'] == 0
        next_has_ball = detections[i + 1]['ball_count'] > 0
        
        if prev_has_ball and curr_no_ball and next_has_ball:
            candidates.append({
                'frame_idx': i,
                'frame': detections[i]['frame'],
                'timestamp': detections[i]['timestamp'],
                'prev_ball': detections[i - 1]['ball_count'],
                'next_ball': detections[i + 1]['ball_count']
            })
    
    print(f"       找到 {len(candidates)} 个漏检候选帧")
    return candidates


def select_frames(candidates, count):
    """从候选帧中均匀选择指定数量的帧"""
    if not candidates:
        return []
    
    if count >= len(candidates):
        print(f"       候选数 ({len(candidates)}) <= 请求数 ({count})，全部选中")
        selected = list(range(len(candidates)))
    elif count <= 1:
        selected = [0]
        print(f"       选取第一帧")
    else:
        # 均匀分布抽样
        step = (len(candidates) - 1) / (count - 1)
        selected = []
        for i in range(count):
            idx = int(round(i * step))
            selected.append(idx)
    
    print(f"       实际选取: {len(selected)} 帧")
    return selected


def extract_frames(candidates, selected_indices, video_file, output_dir):
    """用 ffmpeg 从视频中提取指定帧"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频总时长（用于验证）
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    print(f"[3/3] 视频时长: {duration:.2f} 秒")
    
    extracted = []
    for i, idx in enumerate(selected_indices):
        item = candidates[idx]
        frame_idx = item['frame_idx']
        ts = item['timestamp']
        
            # 文件名格式：missing_ball_{帧号}_{时间戳}.jpg
        jsonl_frame = item.get('frame', 0)
        output_path = os.path.join(output_dir, f"missing_ball_{jsonl_frame:06d}_ts{ts:.3f}s.jpg")
        
        if os.path.exists(output_path):
            print(f"  [{i+1}/{len(selected_indices)}] ⏭️ 已存在")
            extracted.append((output_path, item))
            continue
        
        # ffmpeg -ss 精确定位到时间戳，-frames:v 1 提取一帧，-q:v 2 高质量 JPEG
        cmd = [
            'ffmpeg', '-y',
            '-i', video_file,
            '-ss', str(ts),
            '-frames:v', '1',
            '-q:v', '2',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  [{i+1}/{len(selected_indices)}] ✅ {os.path.basename(output_path)} ({size_kb:.1f} KB)")
            extracted.append((output_path, item))
        else:
            stderr_short = result.stderr[-300:] if result.stderr else "(empty)"
            print(f"  [{i+1}/{len(selected_indices)}] ❌ 失败 (frame={frame_idx}, ts={ts:.3f}s)")
    
    return extracted


def generate_info(output_dir, candidates, selected_indices):
    """生成标注说明文件"""
    info_path = os.path.join(output_dir, "MISSING_BALL_INFO.txt")
    
    with open(info_path, 'w') as f:
        f.write(f"漏检帧提取记录\n")
        f.write(f"{'='*60}\n")
        f.write(f"筛选条件: ball_count(prev>0, curr==0, next>0)\n")
        f.write(f"候选总数: {len(candidates)}\n")
        f.write(f"实际提取: {len(selected_indices)}\n\n")
        
        if selected_indices:
            f.write(f"帧列表 (文件名 | JSONL帧号 | 数组索引 | 时间戳 | 前后球数量):\n")
            f.write(f"{'-'*60}\n")
            for i, idx in enumerate(selected_indices):
                item = candidates[idx]
                name = f"missing_ball_{item['frame']:06d}_ts{item['timestamp']:.3f}s.jpg"
                f.write(f"{name}  |  frame={item['frame']}  |  idx={item['frame_idx']}  |  "
                       f"{item['timestamp']:.3f}s  |  "
                       f"prev={item['prev_ball']}, next={item['next_ball']}\n")
    
    return info_path


def main():
    log_path, video_path, count, output_dir = parse_args()
    
    print(f"\n{'='*60}")
    print(f"  漏检帧提取工具")
    print(f"{'='*60}")
    print(f"  日志:   {log_path}")
    print(f"  视频:   {video_path}")
    print(f"  输出:   {output_dir}/")
    print(f"  数量:   {'全部' if count > 10000 else count} 帧")
    print(f"{'='*60}\n")
    
    # Step 1: 加载日志
    detections = load_log(log_path)
    
    # Step 2: 查找漏检帧
    candidates = find_missing_ball_frames(detections)
    
    if not candidates:
        print("\n⚠️  未找到符合条件的漏检帧！")
        print("   可能原因：模型检测很完整，没有连续有球→无球→有球的跳变")
        sys.exit(0)
    
    # Step 3: 选择帧并提取
    selected_indices = select_frames(candidates, count)
    extracted = extract_frames(candidates, selected_indices, video_path, output_dir)
    
    # 生成说明文件
    info_path = generate_info(output_dir, candidates, selected_indices)
    
    # 总结
    print(f"\n{'='*60}")
    if len(extracted) == count or (count >= 10000 and len(extracted) == len(candidates)):
        print(f"  ✅ 成功提取 {len(extracted)} 张漏检帧")
        print(f"  📁 保存位置: {os.path.abspath(output_dir)}/")
        print(f"  📋 标注说明: {info_path}")
    else:
        print(f"  ⚠️  仅成功提取 {len(extracted)}/{count} 张")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
