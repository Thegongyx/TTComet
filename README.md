# 🏓 乒乓彗星 TTComet · 乒乓球视频高光工具集

[中文](README.md) | [English](#english)

把整场乒乓球录像自动剪成带特效的高光集锦。仓库覆盖完整链路：抽帧 → 标注 → 训练 → 推理 → 剪辑。

| 用法 | 适合谁 | 入口 |
|---|---|---|
| **A. 桌面安装包**（推荐） | 只想剪视频 | `TTComet-0.3.1-setup.exe` |
| **B. 命令行脚本** | 调算法 / 训模型 / 批量处理 | `src/` |

> 💗 支持项目：<https://ifdian.net/a/TTComet>

---

## A. 桌面版（乒乓彗星 TTComet）

本地 WebUI：选视频 → 自动推理 + 剪高光 → 浏览器里播放/下载。无需 Python、无需联网，内置 ffmpeg。

### 下载安装

- 从 [最新版本](https://github.com/TheGongyx/TTComet/releases/latest) 下载 `TTComet-*-setup.exe`（约 374 MB）
- 安装到 `%LOCALAPPDATA%\Programs\TTComet`，不需要管理员权限
- 数据（任务、预设、成品）在 `%LOCALAPPDATA%\TTComet\jobs\`

### 功能

- 自动高光剪辑 + 球轨迹 / 落点 / 迷你球台 / 球员特效
- 可选清理：去水印（渲染特效前处理源视频）、去人声（成片后仅处理音频），默认关闭
- 批量处理、中英双语、两档精度模型、参数预设、处理记录、在线更新
- 不剪辑模式：保留完整时长，只加特效

### 已知限制

- 未做代码签名，首次运行可能提示“未知发布者”（更多信息 → 仍要运行）
- 同一时间只允许一个实例；建议独显，纯 CPU 也可但较慢
- 日志在 `%LOCALAPPDATA%\TTComet\logs\app.log`

---

## B. 命令行脚本

### 环境

- Python 3.10+（建议 conda / venv）
- PyTorch + ultralytics（推理 / 训练）
- ffmpeg（必需）
- 去人声：`demucs`（torch）或 `demucs-onnx`（无需 torch，首次下载约 158 MB）
- GPU：AMD ROCm / NVIDIA CUDA 自动检测，CPU 也可

> 命令在 PowerShell 中执行；Windows 用 `python`，Linux/WSL 用 `python3`。

### 常用脚本

| 脚本 | 用途 |
|---|---|
| `run_pipeline.py` | 一键：推理 → 高光（单视频或目录） |
| `inference_video.py` | 逐帧推理 → `raw_detections.jsonl` |
| `pingpong_highlight.py` | 高光剪辑（video + jsonl） |
| `remove_watermark.py` | 静态水印去除（delogo / inpaint） |
| `remove_narration.py` | 人声去除（demucs / demucs-onnx / center / mute） |
| `extract_missing_ball.py` / `random_sample_frames.py` | 抽帧用于标注 |
| `train_pingpong_v2.py` | 分割模型训练 |
| `export_onnx.py` | `.pt` → `.onnx` |
| `split_all1.py` | 数据集拆分 train/val |

### 快速开始

```powershell
# 一键高光（单个 / 整个目录）
python src/run_pipeline.py --video Videos/match.mp4
python src/run_pipeline.py --video Videos/

# 分步：推理 → 剪辑
python src/inference_video.py --video match.mp4 --model models/m960n.pt --output inference_output/
python src/pingpong_highlight.py --video match.mp4 `
    --raw-detections inference_output/raw_detections.jsonl --output highlight.mp4

# 去水印（可选）+ 去人声（可选）
python src/remove_watermark.py --video Videos/match.mp4 --preset overlay
python src/remove_narration.py --video Videos/match_nowatermark.mp4 --method demucs-onnx

# 训练
python src/split_all1.py --src train_data/all22 --dst train_data/all22_split
python src/train_pingpong_v2.py --data train_data/all22_split/data.yaml `
    --pretrained yolo26x-seg.pt --name my_run --epochs 100 --batch 4
```

### 说明

- 标注为 YOLO txt；`data.yaml` 的 `path` 指向 `images/` `labels/` 的父目录，训练前必须拆分 train/val
- `train_data/`、模型权重、素材视频不入库，只在本地准备
- 临时文件统一放 `tmp/`
- 更多参数、打包与架构说明见 [`webui/app/README.md`](webui/app/README.md)

## 许可

未声明开源许可证（默认保留所有权利）。第三方组件遵循各自许可：Ultralytics（AGPL-3.0）、ONNX Runtime（MIT）、three.js（MIT）。

---

<a id="english"></a>

# 🏓 TTComet · Table Tennis Video Highlight Toolkit

[中文](README.md) | [English](#english)

Turn a full table tennis match recording into a highlight reel with effects. The repository covers the whole workflow: frame extraction → labeling → training → inference → editing.

| Usage | For | Entry |
|---|---|---|
| **A. Desktop installer** (recommended) | Just want to cut videos | `TTComet-0.3.1-setup.exe` |
| **B. Command-line scripts** | Tune algorithms / train / batch-process | `src/` |

> 💗 Support the project: <https://ifdian.net/a/TTComet>

---

## A. Desktop app (TTComet)

A local WebUI: pick a video → automatic inference + highlight editing → play/download in the browser. No Python or internet required, ffmpeg is bundled.

### Install

- Download `TTComet-*-setup.exe` (about 374 MB) from the [latest release](https://github.com/TheGongyx/TTComet/releases/latest)
- Installs to `%LOCALAPPDATA%\Programs\TTComet`; no administrator rights needed
- Jobs, presets and outputs live in `%LOCALAPPDATA%\TTComet\jobs\`

### Features

- Automatic highlight editing with ball trails, bounce markers, a mini table and player effects
- Optional cleanup: watermark removal (on the source video before effects) and vocal removal (audio only, after rendering), both off by default
- Batch processing, bilingual UI, two model tiers, presets, job history, update check
- No-cut mode: keep the full duration and only add effects

### Known limitations

- Not code-signed; Windows may warn about an unknown publisher on first run (More info → Run anyway)
- One instance at a time; a discrete GPU is recommended, CPU-only works but is slow
- Logs: `%LOCALAPPDATA%\TTComet\logs\app.log`

---

## B. Command-line scripts

### Requirements

- Python 3.10+ (conda / venv recommended)
- PyTorch + ultralytics (inference / training)
- ffmpeg (required)
- Vocal removal: `demucs` (torch) or `demucs-onnx` (no torch, ~158 MB model on first use)
- GPU: AMD ROCm / NVIDIA CUDA auto-detected; CPU works too

> Commands use PowerShell; use `python` on Windows and `python3` on Linux/WSL.

### Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | One command: inference → highlight (single video or folder) |
| `inference_video.py` | Frame-by-frame inference → `raw_detections.jsonl` |
| `pingpong_highlight.py` | Highlight editing (video + jsonl) |
| `remove_watermark.py` | Static watermark removal (delogo / inpaint) |
| `remove_narration.py` | Vocal removal (demucs / demucs-onnx / center / mute) |
| `extract_missing_ball.py` / `random_sample_frames.py` | Frame sampling for labeling |
| `train_pingpong_v2.py` | Segmentation training |
| `export_onnx.py` | `.pt` → `.onnx` |
| `split_all1.py` | Train/val split |

### Quick start

```powershell
# One-command highlight (single video / whole folder)
python src/run_pipeline.py --video Videos/match.mp4
python src/run_pipeline.py --video Videos/

# Step by step: inference → editing
python src/inference_video.py --video match.mp4 --model models/m960n.pt --output inference_output/
python src/pingpong_highlight.py --video match.mp4 `
    --raw-detections inference_output/raw_detections.jsonl --output highlight.mp4

# Optional watermark removal + vocal removal
python src/remove_watermark.py --video Videos/match.mp4 --preset overlay
python src/remove_narration.py --video Videos/match_nowatermark.mp4 --method demucs-onnx

# Training
python src/split_all1.py --src train_data/all22 --dst train_data/all22_split
python src/train_pingpong_v2.py --data train_data/all22_split/data.yaml `
    --pretrained yolo26x-seg.pt --name my_run --epochs 100 --batch 4
```

### Notes

- Labels are YOLO txt; `data.yaml` `path` must point to the parent of `images/` and `labels/`; split train/val before training
- `train_data/`, model weights and source videos are not tracked; prepare them locally
- Put temporary files in `tmp/`
- Parameters, packaging and architecture: [`webui/app/README.md`](webui/app/README.md)

## License

No open-source license is declared (all rights reserved by default). Third-party components keep their licenses: Ultralytics (AGPL-3.0), ONNX Runtime (MIT), three.js (MIT).
