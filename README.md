# YOLO 视频帧提取与模型微调工具集

从 YOLO 检测日志中智能筛选并提取特定场景的视频帧,用于数据标注、模型训练与高光集锦生成。全流程覆盖:抽帧 → 标注 → 训练 → 推理 → 高光特效剪辑。

## 📁 项目结构

```
yolo_frame_extractor/
├── src/                                  # 所有主脚本
│   ├── extract_missing_ball.py           #   漏检帧提取(球跳变帧)
│   ├── extract_rejected_ball_frames.py   #   错误球帧收集(标注负样本)
│   ├── random_sample_frames.py           #   通用随机抽帧(支持无日志模式)
│   ├── inference_video.py                #   YOLO 逐帧推理 → 检测日志 + 可视化视频
│   ├── pingpong_highlight.py             #   高光特效剪辑(球轨迹 + 彗尾 + 迷你球台)
│   ├── run_pipeline.py                   #   一键流水线: 推理 → 高光
│   ├── convert_yolo_to_labelstudio.py    #   YOLO 标注转 Label Studio 格式
│   ├── train_pingpong_v2.py              #   YOLO 分割训练启动(bbox→polygon 自动转换)
│   ├── export_onnx.py                    #   YOLO .pt → .onnx 导出
│   ├── split_all1.py                     #   通用数据集拆分 train/val
│   ├── split_all19.py                    #   数据集增量合并拆分(专用)
│   └── track/                            #   目标追踪模式脚本
│       ├── inference_video_track.py      #     track 逐帧推理
│       ├── pingpong_highlight_track.py   #     track 高光剪辑
│       └── run_pipeline_track.py         #     track 一键流水线
├── src_yls/                              # Label Studio ML 后端
│   ├── model.py                          #   ML 后端模型入口
│   ├── control_models/                   #   控制模型实现
│   ├── utils/                            #   工具函数
│   ├── cli.py / start.py / _wsgi.py      #   入口脚本
│   └── requirements.txt                  #   依赖清单
├── models/                               # 自定义训练权重(仅目录结构入库)
│   ├── m960n.pt                          #   自定义模型权重
│   └── m960nold.pt                       #   旧版自定义模型权重
├── train_data/                           # 标注数据集(images/ + labels/ + data.yaml)
│   ├── all21_split/                      #   全量数据集 v21,已拆分 80/20
│   ├── all22/                            #   全量数据集 v22(未拆分)
│   └── all22_split/                      #   全量数据集 v22,已拆分 80/20
├── yolo26x-seg.pt / yolo26l-seg.pt       # YOLO 分割预训练权重(根目录)
├── yolo26m-seg.pt / yolo26s-seg.pt / yolo26n.pt
├── Videos/                               # 原始输入视频(不入库)
├── transition_videos/                    # 过渡期临时视频
├── runs/                                 # 训练输出(weights, metrics, plots)
├── random_frames/                        # 输出: random_sample_frames.py 结果
├── missing_ball_frames/                  # 输出: extract_missing_ball.py 结果
├── inference_output/                     # 输出: inference_video.py 结果
├── tmp/                                  # 临时文件(脚本自动创建)
└── old/                                  # 归档旧版本文件
```

## 🚀 环境要求

- Python 3.10+,conda 环境 `yolo26`(含 PyTorch + ultralytics)
- ffmpeg(视频抽帧 / 推理 / 剪辑必需)
- GPU: AMD Radeon(ROCm)或 NVIDIA(CUDA)均可,ultralytics 自动检测
- Label Studio 标注服务:conda 环境 `label-studio`

---

## 🔧 可用脚本一览

| 脚本 | 用途 | 位置 |
|------|------|------|
| `extract_missing_ball.py` | 漏检帧提取(球跳变帧) | `src/` |
| `extract_rejected_ball_frames.py` | 错误球帧收集(标注负样本) | `src/` |
| `random_sample_frames.py` | 通用随机抽帧(支持无日志) | `src/` |
| `inference_video.py` | YOLO 逐帧推理 → raw_detections.jsonl + 可视化 mp4 | `src/` |
| `pingpong_highlight.py` | 高光特效剪辑(轨迹 + 彗尾 + 迷你球台) | `src/` |
| `run_pipeline.py` | 一键流水线: 推理 → 高光 | `src/` |
| `convert_yolo_to_labelstudio.py` | YOLO 标注转 Label Studio 格式 | `src/` |
| `train_pingpong_v2.py` | YOLO 分割训练启动 | `src/` |
| `export_onnx.py` | YOLO .pt → .onnx 导出 | `src/` |
| `split_all1.py` | 通用数据集拆分 train/val | `src/` |
| `split_all19.py` | 数据集增量合并拆分(专用) | `src/` |
| `inference_video_track.py` | track 追踪模式逐帧推理 | `src/track/` |
| `pingpong_highlight_track.py` | track 追踪模式高光剪辑 | `src/track/` |
| `run_pipeline_track.py` | track 模式一键流水线 | `src/track/` |

---

## 📋 快速上手

> 所有命令在 PowerShell 中执行,换行用反引号 `` ` ``;Windows 用 `python`,Linux/WSL 用 `python3`。

### 1️⃣ 漏检帧提取

从检测日志中找出**球运动跳变**的帧:上一帧有球、当前帧无球、下一帧又有球。典型的高速球运动中 YOLO 漏检的场景。

```powershell
conda run -n yolo26 python src/extract_missing_ball.py `
    --log /path/to/detection_log.jsonl `
    --video /path/to/video.mp4 `
    --count 50
```

**筛选条件:** `ball_count(prev>0, curr==0, next>0)`

### 2️⃣ 随机抽帧

从检测日志中按时间区间均匀随机抽取 N 帧,支持过滤模式;也可无日志直接抽帧。

```powershell
# 只抽有球场景
conda run -n yolo26 python src/random_sample_frames.py `
    --log /path/to/detection_log.json `
    --video /path/to/video.mp4 `
    --count 50 --ball-only

# 直接模式(无检测日志)
conda run -n yolo26 python src/random_sample_frames.py `
    --video /path/to/video.mp4 --no-log --count 20
```

**筛选选项:**
- `--ball-only`:只抽有球场景(ball_count > 0)
- `--no-player`:只抽无人无球背景帧
- 不传过滤参数 = 全量候选随机抽

### 3️⃣ YOLO 逐帧推理

输入视频 → 输出检测日志 + 检测可视化视频,下游供高光剪辑 / 抽帧使用。

```powershell
conda run -n yolo26 python src/inference_video.py `
    --video /path/to/video.mp4 `
    --model yolo26x-seg.pt `
    --output inference_output/
```

### 4️⃣ 高光特效剪辑

基于推理产出的 raw_detections.jsonl 生成高光集锦(球轨迹 + 彗尾 + 迷你球台)。

```powershell
conda run -n yolo26 python src/pingpong_highlight.py `
    --video /path/to/video.mp4 `
    --raw-detections inference_output/raw_detections.jsonl `
    --output highlight.mp4
```

### 5️⃣ 一键流水线

推理 + 高光一键串联;支持单视频或批量目录。

```powershell
# 单视频
conda run -n yolo26 python src/run_pipeline.py --video /path/to/video.mp4

# 批量处理目录中所有视频
conda run -n yolo26 python src/run_pipeline.py --video Videos/
```

---

## 🏋️ 模型训练

### 数据集准备

数据集必须拆分为 train/val 后才能训练:

```powershell
conda run -n yolo26 python src/split_all1.py `
    --src train_data/all22 --dst train_data/all22_split
```

### 启动训练

```powershell
conda run -n yolo26 python src/train_pingpong_v2.py `
    --data train_data/all22_split/data.yaml `
    --pretrained yolo26x-seg.pt `
    --name my_run --epochs 100 --batch 4
```

**参数说明:**
- `--data`:data.yaml 路径(必需)
- `--pretrained`:预训练权重,默认根目录 `yolo26x-seg.pt`;自定义权重用 `models/m960n.pt`
- `--name`:运行名称,输出到 `runs/train_<name>/`
- `--epochs`:训练轮数,默认 100
- `--batch`:批次大小,默认 4(显存不足时调低)

### 类别定义(6 类)

当前数据集 `train_data/*/data.yaml` 定义:

| class_id | 类别 |
|----------|------|
| 0 | 乒乓球 |
| 1 | 其他比赛球 |
| 2 | 球员 |
| 3 | 球拍 |
| 4 | 球桌 |
| 5 | 球网 |

---

## 🔖 Label Studio 标注

数据集标注通过 Label Studio 完成,分为标注服务与 ML 后端两部分:

```powershell
# 标注服务(conda 环境 label-studio,Windows 原生)
conda run -n label-studio label-studio start --port 8080
# 访问 http://localhost:8080

# ML 后端(自动标注/预标注,conda 环境 yolo26)
conda run -n yolo26 python src_yls/_wsgi.py --port 9090 --host 0.0.0.0
```

标注数据导出为 YOLO 格式后放入 `train_data/<name>/`(images/ + labels/ + data.yaml)。也可用 `convert_yolo_to_labelstudio.py` 将已有 YOLO 标注转换为 Label Studio 格式继续标注。

---

## 📊 训练结果解读

### 关键指标

| 指标 | 说明 | 优秀值 |
|------|------|--------|
| `mAP@50` | IoU=0.5 时的平均精度 | >0.85 |
| `mAP@50-95` | IoU 从 0.5-0.95 的平均精度(更严格) | >0.60 |
| `Precision` | 预测为正的样本中真正为正的比例 | >0.80 |
| `Recall` | 真实正样本中被正确找出的比例 | >0.75 |

### 输出文件说明

```
runs/train_<name>/
├── results.csv              ← 每轮指标(可导入 Excel/Python 画图)
├── confusion_matrix.png     ← 混淆矩阵可视化
├── labels.jpg               ← 标注分布图
├── train_batch0-2.jpg       ← 训练批次可视化
└── weights/
    ├── best.pt              ← 🏆 最佳模型(推荐部署用)
    ├── last.pt              ← 最后一轮模型
    └── best_class_*_ap.pt   ← 按单类 AP 追踪的模型(可选)
```

---

## 💡 使用技巧

### 1. 数据集量级建议

| 数据量 | 适用场景 | 推荐 epochs |
|--------|----------|-------------|
| <50 张 | 快速验证 / 原型测试 | 50-100 |
| 50-200 张 | 小规模微调 | 100-200 |
| >200 张 | 生产级模型 | 200-500 |

### 2. GPU 内存(OOM)调整

训练报显存不足时降低 batch size:

```powershell
conda run -n yolo26 python src/train_pingpong_v2.py `
    --data train_data/all22_split/data.yaml `
    --batch 2           # 默认 4,显存不足降到 2
```

### 3. 查看训练曲线

训练完成后用 Python 读取 results.csv 画图:

```python
import pandas as pd
df = pd.read_csv('runs/train_my_run_1/results.csv')
print(df[['epoch', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)']].tail(20))
```

---

## ⚠️ 注意事项

1. **所有脚本必须在 `yolo26` conda 环境中运行**:`conda run -n yolo26 python src/<script>.py`
2. **训练启动后不要持续监控进程**,会占用 GPU 资源拖慢速度。确认启动正常后等用户主动询问进度
3. **标注格式为 YOLO 格式**(txt 每行:`class_id x_center y_center width height`,归一化 [0,1]);也接受 polygon 格式(9+ 值),训练脚本自动转换
4. **data.yaml 的 `path` 字段必须指向 images/labels 所在的父目录**,且训练前必须拆分 train/val —— 否则训练会静默失败
5. **临时文件统一放 `tmp/`**:所有脚本运行中的中间产物写入 `tmp/`,避免污染源码目录

## 📚 更多

项目内部约定、详细命令与 AI 协作规范见 [AGENTS.md](AGENTS.md)。