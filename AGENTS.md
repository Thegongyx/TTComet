# PROJECT KNOWLEDGE BASE

**Generated:** 2025-06-19
**No git repo** — data+scripts workbench

## ENVIRONMENT
- **Shell**: PowerShell（Windows 原生）。所有命令使用 PowerShell 语法：换行符用 `` ` ``（反引号），不用 `^` 或 `\`。
- **语言**: 项目内全部使用中文沟通和思考（文档、注释、交互）。
- **Python**: conda 环境 `yolo26`，Windows 用 `python`，WSL/Linux 用 `python3`。

## OVERVIEW
YOLO 视频帧提取、推理与模型微调工具集。从检测日志中抽帧、运行 YOLO 推理、自动生成高光集锦、训练微调模型（支持 ROCm / CUDA）。

## STRUCTURE
```
./
├── src/                                 # 所有 Python 脚本
│   ├── extract_missing_ball.py          #   漏检帧提取（球跳变帧）
│   ├── extract_rejected_ball_frames.py  #   错误球帧收集（标注负样本）
│   ├── random_sample_frames.py          #   通用随机抽帧（支持无日志模式）
│   ├── inference_video.py              #   YOLO 逐帧推理 → 检测日志 + 可视化视频
│   ├── pingpong_highlight.py           #   高光特效剪辑（球轨迹 + 彗尾 + 迷你球台）
│   ├── run_pipeline.py                 #   一键流水线: 推理 → 高光 (串联上述两个)
│   ├── train_pingpong_v2.py            #   YOLO 分割训练启动（bbox→polygon 转换）
│   └── split_all1.py                   #   数据集拆分 train/val
├── yolo26x-seg.pt                       # YOLO 分割预训练权重 (26M params)
├── yolo26n.pt                           # YOLO nano 预训练权重
├── train_data/                          # 标注数据集 (images/ + labels/ + data.yaml)
│   ├── all9/                            #   6 类全量数据集 (最新版本)
│   └── all9_split/                      #   同上, 已拆分 80/20 train/val
├── runs/                                # 训练输出 (weights, metrics, plots)
├── Videos/                              # 原始输入视频
├── random_frames/                       # 输出: random_sample_frames.py 抽帧结果
├── rejected_ball_frames/                # 输出: extract_rejected_ball_frames.py 结果
├── inference_output/                    # 输出: inference_video.py 推理结果
├── old/                                 # 归档旧版本文件
└── .omo/                                # 规划文件 (内部使用)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| 提取漏检帧（球跳变帧） | `src/extract_missing_ball.py` | `--log` + `--video` 必需 |
| 收集错误球帧（标注负样本） | `src/extract_rejected_ball_frames.py` | 复用 pingpong_highlight 过滤逻辑, 输出 A/B 两类错误球 |
| 随机抽帧（支持无日志） | `src/random_sample_frames.py` | `--ball-only` / `--no-player` / `--no-log` 三种模式 |
| YOLO 逐帧推理 | `src/inference_video.py` | 输入视频 → 输出 raw_detections.jsonl + 检测可视化 mp4 |
| 高光特效剪辑 | `src/pingpong_highlight.py` | 需要 `--video` + `--raw-detections` |
| 一键流水线（推理+高光） | `src/run_pipeline.py` | 最少只需 `--video`；支持批量目录模式 |
| YOLO 分割模型训练 | `src/train_pingpong_v2.py` | bbox→polygon 自动转换, best_class AP/mAP50 双追踪 |
| 数据集拆分 train/val | `src/split_all1.py` | `--src` / `--dst` CLI 参数 |
| 查找训练结果 | `runs/<name>/weights/best.pt` | 同时保存 best_class_*_ap.pt 和 best_class_*_map50.pt |
| 训练配置 | `train_data/*/data.yaml` | nc, names, path 字段 |

## CONVENTIONS
- **项目内沟通和思考使用中文** — 所有文档、注释、交互均使用中文
- **Shell 为 PowerShell** — 换行符用 `` ` ``（反引号），路径可用 `/` 或 `\`。不要用 bash 语法（`^`、`&&`、`\` 换行等）
- **所有脚本在 conda 环境 `yolo26` 中运行**: `conda run -n yolo26 python src/<script>.py`
- **Windows 用 `python`，WSL/Linux 用 `python3`**
- **Scripts are self-contained**: each has `main()` + `if __name__ == "__main__"`, no cross-file imports（`extract_rejected_ball_frames.py` 和 `run_pipeline.py` 除外 — 前者复制了 TableModel，后者通过子进程调其他脚本）
- **Labels are YOLO format**: one `.txt` per image, each line: `class_id x_center y_center width height` (normalized [0,1])
- **polygon labels also accepted**: 9+ values per line (class + 4 corner points)
- **data.yaml convention**: `path` → parent of `images/` and `labels/`; `train`/`val` are subdirs
- **Output dirs auto-created** by scripts — `random_frames/`, `runs/train_<name>/` 等
- **ffmpeg required** for frame extraction / inference / highlight scripts
- **Interactive fallback**: scripts prompt for missing CLI args instead of erroring
- **临时文件统一放在 `tmp/` 目录**：所有脚本运行中产生的中间文件、缓存、临时输出等，均写入项目根目录下的 `tmp/`（自动创建），避免污染源码目录

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER write new Python files for one-off tasks** — pass args to existing scripts instead
- **NEVER monitor training process continuously** — confirm launch, then wait for user to ask for progress
- **data.yaml `path` MUST point to the parent of images/labels** — wrong path = training silently fails
- **Always split data into train/val** before training — `split_all1.py` does this; don't train on unsplit data

## COMMANDS
```powershell
# 漏检帧提取
conda run -n yolo26 python src/extract_missing_ball.py `
    --log /path/to/detection_log.jsonl `
    --video /path/to/video.mp4 `
    --count 50

# 错误球帧收集（标注负样本）
conda run -n yolo26 python src/extract_rejected_ball_frames.py `
    --log inference_output/raw_detections.jsonl `
    --video inference_output/video_cfr.mp4 `
    --count 50

# 随机抽帧（有球场景）
conda run -n yolo26 python src/random_sample_frames.py `
    --log /path/to/detection_log.json `
    --video /path/to/video.mp4 `
    --count 50 --ball-only

# 随机抽帧（直接模式，无需检测日志）
conda run -n yolo26 python src/random_sample_frames.py `
    --video /path/to/video.mp4 `
    --no-log --count 20

# YOLO 逐帧推理
conda run -n yolo26 python src/inference_video.py `
    --video /path/to/video.mp4 `
    --model yolo26x-seg.pt `
    --output inference_output/

# 高光特效剪辑
conda run -n yolo26 python src/pingpong_highlight.py `
    --video /path/to/video.mp4 `
    --raw-detections inference_output/raw_detections.jsonl `
    --output highlight.mp4

# 一键流水线: 推理 + 高光 (单视频)
conda run -n yolo26 python src/run_pipeline.py `
    --video /path/to/video.mp4

# 一键流水线: 批量处理目录中所有视频
conda run -n yolo26 python src/run_pipeline.py `
    --video Videos/

# 数据集拆分 80/20
conda run -n yolo26 python src/split_all1.py `
    --src train_data/all9 --dst train_data/all9_split

# YOLO 分割训练
conda run -n yolo26 python src/train_pingpong_v2.py `
    --data train_data/all9_split/data.yaml `
    --pretrained yolo26x-seg.pt `
    --name my_run --epochs 100 --batch 4
```

## NOTES
- **GPU**: AMD Radeon via ROCm (`device: "0"` in ultralytics) 或 NVIDIA CUDA 均可，ultralytics 自动检测
- **Python 命令**: Windows 用 `python`，WSL/Linux 用 `python3`
- **Pre-trained weights**: `yolo26x-seg.pt` (26M params, segmentation) 和 `yolo26n.pt` (nano) 在项目根目录
- **Label format is mixed**: some files are polygon (9 values), some are bbox (5 values) — `train_pingpong_v2.py` auto-converts bbox→polygon before training
- **OOM workaround**: reduce `--batch` (default 4) if GPU runs out of memory
- **mAP@50-95 >0.60** is target for production models
- Training results in `runs/<name>/results.csv` for post-hoc analysis
