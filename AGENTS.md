# PROJECT KNOWLEDGE BASE

**Generated:** 2025-06-19
**Git 仓库** — 使用 git 版本控制

## ENVIRONMENT
- **Shell**: PowerShell（Windows 原生）。所有命令使用 PowerShell 语法：换行符用 `` ` ``（反引号），不用 `^` 或 `\`。
- **语言**: 项目内全部使用中文沟通和思考（文档、注释、交互）。
- **Python**: Windows 用 `python`，WSL/Linux 用 `python3`。
- **Conda 环境**（Windows 原生，均已从 WSL2 迁移到 Windows）:
  - `yolo26` — YOLO 推理 / 训练 / ML 后端（`src_yls`）
  - `label-studio` — Label Studio 标注服务本体（Python 3.12，label-studio 1.23.0）
  - `rfdetr` — RF-DETR 训练 / 推理 / 流水线（`src_detr`）。由 `yolo26` 克隆而来以复用 ROCm torch，再 `pip install rfdetr[train]==1.10.1`

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
│   ├── inference_video_track.py        #   YOLO 逐帧推理（track 追踪模式）
│   ├── pingpong_highlight.py           #   高光特效剪辑（球轨迹 + 彗尾 + 迷你球台）
│   ├── pingpong_highlight_track.py     #   高光特效剪辑（track 追踪模式）
│   ├── run_pipeline.py                 #   一键流水线: 推理 → 高光 (串联上述两个)
│   ├── run_pipeline_track.py           #   一键流水线（track 追踪模式）
│   ├── convert_yolo_to_labelstudio.py  #   YOLO 标注转 Label Studio 格式
│   ├── train_pingpong_v2.py            #   YOLO 分割训练启动（bbox→polygon 转换）
│   └── split_all1.py                   #   数据集拆分 train/val
├── src_detr/                            # RF-DETR 队列（训练/推理/流水线, 复用 src/pingpong_highlight.py）
│   ├── train_detr.py                    #   RF-DETR 微调（detect/seg; 自动搭 YOLO→RF-DETR 数据集视图）
│   ├── inference_video_detr.py          #   RF-DETR 逐帧推理 → 与 YOLO 版同格式的 raw_detections.jsonl
│   ├── run_pipeline_detr.py             #   一键流水线: RF-DETR 推理 + src/pingpong_highlight.py
│   ├── download_weights.py              #   官方权重多连接下载器 (MD5 校验 + 断点续传)
│   └── README.md                        #   环境搭建与用法
├── src_yls/                             # Label Studio ML 后端
│   ├── model.py                         #   ML 后端模型入口
│   ├── control_models/                  #   控制模型实现
│   ├── utils/                           #   工具函数
│   ├── cli.py                           #   CLI 入口
│   ├── start.py                         #   启动脚本
│   └── requirements.txt                 #   依赖清单
├── models/                              # 自定义训练权重
│   ├── m960n.pt                         #   自定义模型权重
│   └── m960nold.pt                      #   旧版自定义模型权重
├── yolo26x-seg.pt                       # YOLO 分割预训练权重 (26M params)
├── yolo26l-seg.pt                       # YOLO large 分割预训练权重
├── yolo26m-seg.pt                       # YOLO medium 分割预训练权重
├── yolo26s-seg.pt                       # YOLO small 分割预训练权重
├── yolo26n.pt                           # YOLO nano 预训练权重
├── rf-detr-seg-nano.pt                  # RF-DETR seg 预训练权重 (nano)
├── rf-detr-seg-small.pt                 # RF-DETR seg 预训练权重 (small)
├── rf-detr-seg-medium.pt                # RF-DETR seg 预训练权重 (medium)
├── rf-detr-seg-2xlarge.pt               # RF-DETR seg 预训练权重 (2xlarge, 768²)
├── train_data/                          # 标注数据集 (images/ + labels/ + data.yaml)
│   ├── all12_labelstudio/               #   从 WSL2 迁移的数据集 (labelstudio_tasks.json 导出)
│   ├── all18/                           #   全量数据集 (最新版本)
│   ├── all18_split/                     #   同上, 已拆分 80/20 train/val
│   └── all17_split/                     #   历史版本, 已拆分
├── runs/                                # YOLO 训练输出 (weights, metrics, plots)
├── runs_detr/                           # RF-DETR 训练输出 (checkpoint_*.pth)
├── Videos/                              # 原始输入视频
├── webui/                               # 乒乓彗星 TTComet —— 本地 WebUI + ONNX 推理（可打包成 exe）
│   ├── run_server.py                    #   启动器（exe 入口）：起本地服务 + 开浏览器
│   ├── app/                             #   后端与前端
│   │   ├── onnx_engine.py               #     ONNX 推理引擎（预处理/NMS/mask→polygon）
│   │   ├── inference_job.py             #     推理主流程（VFR→CFR、三段流水线）
│   │   ├── highlight_bridge.py          #     以模块方式加载并调用 src/pingpong_highlight.py
│   │   ├── jobs.py / server.py          #     串行任务队列 / FastAPI + SSE
│   │   ├── paths.py                     #     资源路径（兼容 PyInstaller）
│   │   └── static/                      #     单页界面 + three.js 场景 + favicon
│   ├── models/                          #   两档 ONNX 模型（高精度 m960 / 低精度 s960）
│   ├── design/                          #   图标设计与生成脚本（make_icons.py / make_ico.py）
│   ├── packaging/                       #   打包：app.spec (PyInstaller) / app.iss (Inno Setup)
│   └── bench/                           #   引擎基准与结果对拍脚本（RESULTS.md 有实测数据）
├── random_frames/                       # 输出: random_sample_frames.py 抽帧结果
├── missing_ball_frames/                 # 输出: extract_missing_ball.py 漏检帧结果
├── inference_output/                    # 输出: inference_video.py 推理结果
├── tmp/                                 # 临时文件（脚本自动创建）
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
| YOLO .pt→onnx 导出 | `src/export_onnx.py` | `--model` + 可选 `--imgsz`/`--half`/`--dynamic`/`--simplify`/`--nms`/`--opset`（`--half` 内部映射为 ultralytics `quantize=16`） |
| YOLO→Label Studio 转换 | `src/convert_yolo_to_labelstudio.py` | YOLO 标注格式转 Label Studio |
| Track 逐帧推理 | `src/inference_video_track.py` | 带目标追踪的推理模式 |
| Track 高光剪辑 | `src/pingpong_highlight_track.py` | 带目标追踪的高光模式 |
| Track 一键流水线 | `src/run_pipeline_track.py` | track 模式推理+高光 |
| RF-DETR 训练（detect/seg） | `src_detr/train_detr.py` | 复用 YOLO 数据集; 自动在 `tmp/` 搭 RF-DETR 数据集视图; 输出 `runs_detr/<name>/checkpoint_best_total.pth` |
| RF-DETR 逐帧推理 | `src_detr/inference_video_detr.py` | 输出与 YOLO 版同 schema 的 `raw_detections.jsonl`（含 table/net polygon） |
| RF-DETR 一键流水线 | `src_detr/run_pipeline_detr.py` | RF-DETR 推理 + 共用 `src/pingpong_highlight.py` |
| RF-DETR 权重预下载 | `src_detr/download_weights.py` | 官方 GCS 源多连接分段 + MD5 校验; 缓存到 `~/.roboflow/models/` |
| Label Studio ML 后端 | `src_yls/model.py` | Label Studio 集成，`src_yls/start.py` 启动 |
| Label Studio 标注服务 | conda 环境 `label-studio` | Windows 原生运行, `http://localhost:8080`, 数据在 `%LOCALAPPDATA%\label-studio\label-studio\` |
| 本地 WebUI（乒乓彗星） | `webui/run_server.py` | 起本地服务 + 开浏览器；用 `webui/.venv` |
| ONNX 推理引擎 | `webui/app/onnx_engine.py` | 预处理 + NMS + mask→polygon，DirectML 加速 |
| 打包 exe / 安装包 | `webui/packaging/build.ps1` | `-WithFFmpeg -Installer`，产物在 `webui/dist` |
| 界面文案（中/英） | `webui/app/static/i18n.js` | 所有 UI 与参数说明的翻译源 |
| 图标设计 | `webui/design/make_ico.py` | 生成 `ttComet.ico` / `favicon.ico` |
| 查找训练结果 | `runs/<name>/weights/best.pt` | 同时保存 best_class_*_ap.pt 和 best_class_*_map50.pt |
| 训练配置 | `train_data/*/data.yaml` | nc, names, path 字段 |

## CONVENTIONS
- **项目内沟通和思考使用中文** — 所有文档、注释、交互均使用中文
- **Shell 为 PowerShell** — 换行符用 `` ` ``（反引号），路径可用 `/` 或 `\`。不要用 bash 语法（`^`、`&&`、`\` 换行等）
- **所有脚本在 conda 环境 `yolo26` 中运行**: `conda run -n yolo26 python src/<script>.py`
- **`src_detr/` 脚本在 conda 环境 `rfdetr` 中运行**: `conda run -n rfdetr python src_detr/<script>.py`（`rfdetr` 由 `yolo26` 克隆，含相同 ROCm torch，可直接跑 `src/pingpong_highlight.py`）
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

# YOLO .pt → onnx 导出
conda run -n yolo26 python src/export_onnx.py `
    --model models/m960n.pt `
    --half --dynamic --simplify

# YOLO .pt → onnx 批量导出 (递归扫描目录下所有 *.pt, 可加 --pattern / --no-recurse)
conda run -n yolo26 python src/export_onnx.py `
    --model-dir runs --pattern "*.pt" --output inference_output/

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
    --src train_data/all18 --dst train_data/all18_split

# YOLO 分割训练
conda run -n yolo26 python src/train_pingpong_v2.py `
    --data train_data/all18_split/data.yaml `
    --pretrained yolo26x-seg.pt `
    --name my_run --epochs 100 --batch 4

# RF-DETR 微调训练 (分割, 复用 YOLO 数据集; 输出 runs_detr/<name>/checkpoint_best_total.pth)
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --size medium --task seg --name pingpong_detr_v1

# RF-DETR 官方权重预下载 (多连接加速 + MD5; 缓存到 ~/.roboflow/models/)
conda run -n rfdetr python src_detr/download_weights.py --models seg-nano seg-small seg-medium

# RF-DETR 逐帧推理 (输出与 YOLO 版同格式的 raw_detections.jsonl, 可被 pingpong_highlight.py 使用)
conda run -n rfdetr python src_detr/inference_video_detr.py `
    --model runs_detr/pingpong_detr_v1/checkpoint_best_total.pth `
    --video Videos/match.mp4 `
    --output inference_output/detr_match

# RF-DETR 一键流水线: 推理 + 高光 (单视频 / 批量目录)
conda run -n rfdetr python src_detr/run_pipeline_detr.py `
    --model runs_detr/pingpong_detr_v1/checkpoint_best_total.pth `
    --video Videos/

# Label Studio 标注服务 (Windows 原生, conda 环境 label-studio)
conda run -n label-studio label-studio start --port 8080

# Label Studio ML 后端 (Windows 原生, conda 环境 yolo26)
conda run -n yolo26 python src_yls/_wsgi.py --port 9090 --host 0.0.0.0
```

## NOTES
- **GPU**: AMD Radeon via ROCm (`device: "0"` in ultralytics) 或 NVIDIA CUDA 均可，ultralytics 自动检测
- **yolo26 环境栈 (2026-10)**: torch `2.13.0+rocm10.1.0` / torchvision `0.28.0+rocm10.1.0` / torchaudio `2.11.0.3+rocm10.1.0` / ROCm SDK `10.1.0`，来自 AMD 预发布通道 `--index-url https://stable.repo.amd.com/rocm/whl-next/`（示例：`"torch[device-gfx1151]==2.13.0+rocm10.1.0"`）。注意 `rocm_is_available()` 只认 Linux，Windows+ROCm 仍被 ultralytics 误判为 CUDA。换 ROCm 版本会使 MIOpen 缓存整体失效，首次推理/训练需重建（ROCm 10.1 在 gfx1151 上首批实测约 14 分钟，7.14 约 4 分钟）。`HIP_PATH`（用户变量，指向独立的 `C:\xiangmu\lemonade\rocm-7.14`）与 pip 版 ROCm 10.1 并存未冲突，暂未改动。原 7.14 组合快照见历史记录；`rfdetr` 环境未升级（仍 torch 2.12.0+rocm7.14.0），`src_detr/` 不受影响
- **ultralytics 版本**: `yolo26` 环境固定为 `8.4.173`（2026-10 升级，前为 8.4.60）。新版新增硬依赖 `nvidia-ml-py` / `ultralytics-platform` / `ultralytics-thop>=2.2.2`，并支持 AMD ROCm / MIGraphX 导出。API 上 `half`/`int8` 已废弃 → 用 `quantize`（`16`=FP16），`end2end` 已废弃 → 用 `nms`（`False`=NMS-free 头）。脚本已适配，勿再直接传 `half=`/`end2end=`（否则每次 `predict` 都会刷废弃警告）。`rfdetr` 环境仍为 8.4.60，`src_detr/` 不 import ultralytics，无需同步升级
- **Python 命令**: Windows 用 `python`，WSL/Linux 用 `python3`
- **Pre-trained weights**: `yolo26x-seg.pt` (26M params, segmentation)、`yolo26l-seg.pt`、`yolo26m-seg.pt`、`yolo26s-seg.pt` 和 `yolo26n.pt` (nano) 在项目根目录；`models/` 下有自定义训练权重 `m960n.pt`
- **Label Studio 已从 WSL2 迁移到 Windows**: 2026-08-23 迁移。数据在 `%LOCALAPPDATA%\label-studio\label-studio\`（sqlite + media + .env）。`.env` 中的 `SECRET_KEY` 必须与 WSL2 原数据一致，否则用户密码/API token 无法解密。WSL2 原数据保留在 `~/.local/share/label-studio/` 作为备份
- **Label format is mixed**: some files are polygon (9 values), some are bbox (5 values) — `train_pingpong_v2.py` auto-converts bbox→polygon before training
- **OOM workaround**: reduce `--batch` (default 4) if GPU runs out of memory
- **mAP@50-95 >0.60** is target for production models
- Training results in `runs/<name>/results.csv` for post-hoc analysis
- **RF-DETR (`src_detr/`)**: 用 RF-DETR (DINOv2 backbone, NMS-free) 替代 YOLO。数据集仍是 YOLO 格式，`train_detr.py` 在 `tmp/detr_dataset_<name>/` 用 **junction/软链接** 搭 `train/{images,labels}` + `valid/{images,labels}` 视图（不复制图片），RF-DETR 原生支持 YOLO 格式与 bbox/polygon 混合标签。推理输出的 `raw_detections.jsonl` 与 YOLO 版逐字段一致，`src/pingpong_highlight.py` 直接可用。分割 checkpoint 会输出 table/net 的 `polygon`。详见 `src_detr/README.md`
- **RF-DETR 权重**: 官方权重在 Google Storage（非 HuggingFace；HF 上 `Roboflow/rf-detr-*` 是 transformers safetensors，与 pip 包不通用，且 hf-mirror 实测更慢）。用 `src_detr/download_weights.py` 多连接预下载（约 2 倍速 + MD5），缓存到 `~/.roboflow/models/`，之后训练/推理直接命中不再联网
