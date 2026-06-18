# YOLO 视频帧提取与模型微调工具集

从 YOLO 检测日志中智能筛选并提取特定场景的视频帧，用于数据标注和模型训练。

## 📁 项目结构

```
yolo_frame_extractor/
├── README.md                          # 本文件（完整工作流程文档）
├── AGENTS.md                          # AI 知识库（项目约定和命令）
├── extract_missing_ball.py            # 漏检帧提取脚本
├── random_sample_frames.py            # 通用随机抽帧脚本
├── train_pingpong_v2.py               # YOLO 微调训练启动脚本（完整版）
├── train_all1.py                      # YOLO 训练快速入口（CLI 参数）
├── split_all1.py                      # 数据集拆分工具（CLI 参数）
├── yolo26x-seg.pt                     # YOLO 分割预训练权重
├── train_data/                        # 标注数据集
│   ├── all5/                          #   全量标注，5 类
│   └── all5_split/                    #   已拆分 train/val
├── runs/                              # 训练输出目录
├── missing_ball_frames/               # 漏检帧输出目录（自动创建）
├── random_frames/                     # 随机抽帧输出目录（自动创建）
└── pingpong_dataset_1080p/            # 原始视频帧数据集
```

## 🚀 环境要求

- Python 3.10+
- conda 环境: `yolo26`（已安装 PyTorch + ROCm + ultralytics）
- ffmpeg（视频抽帧）
- GPU: AMD Radeon GPU（ROCm）

---

## 🔧 可用脚本一览

### 1️⃣ 漏检帧提取 — `extract_missing_ball.py`

从检测日志中找出**球运动跳变**的帧：上一帧有球、当前帧无球、下一帧又有球。典型的高速球运动中 YOLO 漏检的场景。

```bash
conda run -n yolo26 python extract_missing_ball.py ^
    --log /path/to/detection_log.json ^
    --video /path/to/video.mp4 ^
    --count 50 ^
    --output custom_frames/
```

**筛选条件：** `ball_count(prev>0, curr==0, next>0)`

---

### 2️⃣ 通用随机抽帧 — `random_sample_frames.py`

从检测日志中按时间区间均匀随机抽取 N 帧，支持有球场景/无人背景等过滤模式。

```bash
conda run -n yolo26 python random_sample_frames.py ^
    --log /path/to/detection_log.json ^
    --video /path/to/video.mp4 ^
    --count 50 ^
    --ball-only ^
    --output my_random_frames/
```

**筛选选项：**
- `--ball-only`：只抽有球场景（ball_count > 0）
- `--no-player`：只抽无人无球背景帧
- 不传过滤参数 = 全量候选随机抽

---

### 3️⃣ YOLO 微调训练 — `train_pingpong_v2.py`

启动 YOLO 模型微调，支持任意 data.yaml + 预训练权重。一定要把训练数据拆分成训练集和测试集之后再进行后续的工作

```bash
conda run -n yolo26 python train_pingpong_v2.py ^
    --data /path/to/data.yaml ^
    --pretrained /path/to/best.pt ^
    --name my_run ^
    --epochs 100 ^
    --batch 4
```

**参数说明：**
- `--data`：data.yaml 路径（必需）
- `--pretrained`：预训练模型权重，默认 `pingpong_project_test/models/best.pt`
- `--name`：运行名称，用于输出目录命名
- `--epochs`：训练轮数，默认 100
- `--batch`：批次大小，默认 4（根据 GPU 内存调整）

---

## 📋 完整工作流程（推荐流程）

> **核心原则：** AI 只负责启动脚本和监控进程，**不写新 Python 文件**。所有操作通过传参 + 执行现有 py 文件完成。

### Step 1: 提供视频 → AI 抽帧

```
用户：我有视频 /path/to/my_video.mp4 和检测日志 /path/to/detection_log.json，帮我抽 50 张漏检帧用于训练标注
```

**AI 执行：**
```bash
conda run -n yolo26 python extract_missing_ball.py ^
    --log /path/to/detection_log.json ^
    --video /path/to/my_video.mp4 ^
    --count 50
```

> 如果用户选择随机抽帧，则执行 `random_sample_frames.py`，参数类似。

**AI 汇报：**
```
✅ 抽帧完成！文件保存在：
📁 missing_ball_frames/
共 50 张图，标注说明在 MISSING_BALL_INFO.txt

请去 YOLO Label Studio（或你习惯的标注工具）进行标注。
```

---

### Step 2: 用户标注 → 导出 YOLO 格式训练文件

用户在标注工具中标注图片，导出为 YOLO 格式：
- `images/` 目录存放标注后的 `.jpg` 文件
- `labels/` 目录存放对应的 `.txt` 标签文件（YOLO 格式）
- `classes.txt` 列出所有类别名称
- 可选：`notes.json` 记录标注元信息

**用户操作：**
```
我已经标注好了，文件在 /path/to/labeled_data/
```

---

### Step 3: AI 编写 data.yaml → 启动训练

**AI 自动创建 `data.yaml`（不写 Python）：**
```yaml
# /path/to/labeled_data/data.yaml
path: /path/to/labeled_data
train: images/
val: images/

nc: 5                    # 类别数量
names:
  0: pingpangball        # 乒乓球
  1: player              # 球员
  2: racket              # 球拍
  3: table               # 球桌
  4: net                 # 球网
```

**AI 执行训练（后台启动，不持续监控）：**
```bash
conda run -n yolo26 python train_pingpong_v2.py ^
    --data /path/to/labeled_data/data.yaml ^
    --pretrained /path/to/best.pt ^
    --name my_run_1 ^
    --epochs 100
```

**AI 汇报：**
```
✅ 训练已启动！进程运行中...

📊 训练配置:
   - 数据集: /path/to/labeled_data (5 类)
   - 预训练权重: best.pt
   - Epochs: 100 | Batch: 4
   - GPU: 自动检测（CUDA 或 ROCm）

📁 输出目录: runs/train_my_run_1/
🏆 最佳模型将保存在: runs/train_my_run_1/weights/best.pt

⏱️ 预计完成时间: ~9-15 分钟（取决于数据量和 GPU）

我会监控进程是否正常启动。完成后请告诉我「训练完成了吗？」我来汇报结果。
```

---

### Step 4: 用户过一段时间询问进度

**用户：**
```
训练完成了吗？
```

**AI 检查：**
- ✅ **已完成**：读取 `results.csv`，提取 mAP@50-95 等指标，输出最终模型路径和各类别精度表。
- ❌ **未完成**：估算剩余时间（基于已跑 epoch 的平均耗时），告知用户继续等待即可。

---

## 📊 训练结果解读

### 关键指标

| 指标 | 说明 | 优秀值 |
|------|------|--------|
| `mAP@50` | IoU=0.5 时的平均精度 | >0.85 |
| `mAP@50-95` | IoU 从 0.5-0.95 的平均精度（更严格） | >0.60 |
| `Precision` | 预测为正的样本中真正为正的比例 | >0.80 |
| `Recall` | 真实正样本中被正确找出的比例 | >0.75 |

### 输出文件说明

```
runs/train_<name>/
├── results.csv              ← 每轮指标（可导入 Excel/Python 画图）
├── confusion_matrix.png     ← 混淆矩阵可视化
├── labels.jpg               ← 标注分布图
├── train_batch0-2.jpg       ← 训练批次可视化
└── weights/
    ├── best.pt              ← 🏆 最佳模型（推荐部署用）
    └── last.pt              ← 最后一轮模型
```

---

## 💡 使用技巧

### 1. 数据集量级建议

| 数据量 | 适用场景 | 推荐 epochs |
|--------|----------|-------------|
| <50 张 | 快速验证 / 原型测试 | 50-100 |
| 50-200 张 | 小规模微调 | 100-200 |
| >200 张 | 生产级模型 | 200-500 |

### 2. GPU 内存调整

如果训练时 OOM（显存不足），减小 batch size：
```bash
conda run -n yolo26 python train_pingpong_v2.py ^
    --data /path/to/data.yaml ^
    --batch 2           # 从 4 降到 2
```

### 3. 加速训练

减少 workers（如果 CPU 瓶颈）或增加 batch size（如果 GPU 空闲）：
```bash
conda run -n yolo26 python train_pingpong_v2.py ^
    --data /path/to/data.yaml ^
    --workers 4          # 增加数据加载线程
    --batch 8            # 增大批次（GPU 内存允许时）
```

### 4. 查看训练曲线

训练完成后，用 Python 读取 results.csv 画图：
```python
import pandas as pd
df = pd.read_csv('runs/train_my_run_1/results.csv')
print(df[['epoch', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)']].tail(20))
```

---

## ⚠️ 注意事项

1. **所有脚本必须在 `yolo26` conda 环境中运行**：`conda run -n yolo26 python xxx.py`（Windows 用 `python`，WSL/Linux 用 `python3`）
2. **训练启动后不要持续监控进程**，会占用 GPU 资源拖慢速度。AI 确认启动正常后即可汇报，等用户主动询问进度。
3. **标注格式必须是 YOLO 格式**（txt 文件每行：`class_id x_center y_center width height`，归一化到 [0,1]）。
4. **data.yaml 的 `path` 字段必须指向 images/labels 所在的父目录，且一定要拆分训练集和测试集**。
