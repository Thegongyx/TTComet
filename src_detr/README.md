# src_detr — RF-DETR 乒乓球队列 (训练 / 推理 / 流水线)

用 [RF-DETR](https://github.com/roboflow/rf-detr) 替代 YOLO26 完成 **训练 → 推理 → 高光剪辑**。
推理输出 (`raw_detections.jsonl`) 与 `src/inference_video.py` **完全一致**，
因此 `src/pingpong_highlight.py` 无需任何修改即可直接消费。

```
src_detr/
├── train_detr.py            # RF-DETR 微调训练 (检测 / 实例分割)
├── inference_video_detr.py  # 逐帧推理 → raw_detections.jsonl + 可视化视频
├── run_pipeline_detr.py     # 一键流水线: RF-DETR 推理 + src/pingpong_highlight.py
├── download_weights.py      # 官方权重多连接下载器 (MD5 校验 + 断点续传)
└── README.md
```

## 模型路径参数（各脚本）

| 脚本 | 参数 | 默认值 |
|------|------|--------|
| `train_detr.py` | `--pretrained` / `--model` | 无（自动下载对应规模的官方权重） |
| `inference_video_detr.py` | `--model` / `-m` | 项目根目录 `rf-detr-seg-medium.pt` |
| `run_pipeline_detr.py` | `--model` / `-m` | 项目根目录 `rf-detr-seg-medium.pt` |

- `--model` 既可传**训练 checkpoint**（`runs_detr/<name>/checkpoint_best_total.pth`），
  也可传**官方预训练权重**（项目根目录的 `rf-detr-seg-*.pt`）；脚本会自动判别并用对应方式加载。
- 训练时 `--pretrained` 用于指定微调的起点（默认官方权重）。

## 与 YOLO 脚本的对齐（CLI + 日志）

为让用户"只感觉到换了模型"，`src_detr/` 脚本在**参数名**与**控制台日志**上对齐了 `src/` 的 YOLO 脚本：

- **参数**：接受与 YOLO 对应脚本相同的参数名。能映射的做别名（如 `--imgsz`→分辨率、`--lr0`→学习率、
  `--freeze`→冻结 backbone、`--amp/--no-amp`、`--half`）；RF-DETR 不支持的（`--mosaic/--mixup/--iou/
  --max-det/--p2/--end2end` 等）**仍接受但打印一行忽略警告**，保证原有 YOLO 命令行不会报错。
- **日志**：banner、`Loading model: ...`、VFR/CFR、进度条与 ETA、`⏱️  Total ...`、`🔍 检测统计 ...`、
  `✅ Inference Complete!`、`📄 raw_detections.jsonl`、`🎬 pingpong_detection.mp4`；训练侧
  `🏓 ... - 启动`、`📋 训练配置:`、`🚀 开始训练...`、`✅ 训练完成!`、每类指标表（`★` 标记）、
  `📦 所有保存模型一览`。
- **best_class 追踪**：`--best-class`（-2=自动识别乒乓球）会在每轮验证后按该类 AP 保存
  `best_class_<id>_ap.pth` 并打印同款表格。
  > 限制：RF-DETR 只暴露每类 `AP@50-95`，**没有每类 AP50**，故不生成 `best_class_*_map50.pth`
  > （启动日志会提示 `ℹ️ RF-DETR 无每类 AP50，跳过 best_class_*_map50`）。
- **输出命名**：流水线输出与 YOLO 版一致（`inference_output/pipeline_<视频名>/`、
  `<视频名>_highlight.mp4`、`raw_detections.jsonl`、`pingpong_detection.mp4`、`video_cfr.mp4`）。
- **类别加权**：`--class-weight W`（默认 1.0）与 YOLO 语义一致 —— 在**分类损失**上按类乘权重。
  实现：补丁 `SetCriterion.loss_labels` 的默认 IA-BCE 分支，等价 `loss_ce *= class_weights`
  （YOLO 是 `bce_loss *= class_weights`），训练结束自动恢复。
  默认对「乒乓球」加权，也跟随 `--best-class` 指定类别；仅当 `ia_bce_loss=True`（RF-DETR 默认）时生效。

---

## 1. 环境准备 (conda env: `rfdetr`)

项目已从 `yolo26` 环境 **克隆** 出 `rfdetr` 环境（复用其中的 ROCm torch / torchvision /
opencv / numpy 等大库，无需重新下载），并在其中安装了 `rfdetr[train]`。

已完成的搭建命令（如需重建）：

```powershell
# 1) 克隆 yolo26 环境 (硬链接, 不复制/不下载大库)
conda create -n rfdetr --clone yolo26 --yes

# 2) 安装 RF-DETR 及训练依赖 (torch 已存在, 不会被重装)
conda run -n rfdetr python -m pip install "rfdetr[train]==1.10.1"
```

验证：

```powershell
conda run -n rfdetr python -c "import torch, rfdetr; print(torch.__version__, torch.cuda.is_available(), rfdetr.__version__)"
```

> GPU 说明：`yolo26` 使用 AMD ROCm（`torch 2.12.0+rocm`）。克隆后 `rfdetr` 复用同一套
> ROCm torch，`torch.cuda.is_available()` 为 `True`，脚本用 `--device 0` / `cuda:0` 即可。
> 若无 GPU，传 `--device cpu`。

### 权重预下载（多连接加速）

RF-DETR 官方权重托管在 **Google Cloud Storage**（`storage.googleapis.com/rfdetr/...`），
首次训练/推理会自动下载（每个约 136 MB）。HuggingFace 上虽有 `Roboflow/rf-detr-*`，
但那是 **transformers safetensors** 格式（须用 `transformers.RfDetrForObjectDetection` 加载），
与本项目用的 `rfdetr` pip 包的 `.pth` 检查点**不通用**，且实测 hf-mirror 速度更慢。
因此提供一个多连接分段下载器（实测约 2 倍速 + MD5 校验 + 断点续传）：

```powershell
# 默认下载 seg-nano / seg-small / seg-medium
conda run -n rfdetr python src_detr/download_weights.py

# 指定模型 (可选 seg-nano/seg-small/seg-medium/seg-large/seg-xlarge/seg-2xlarge/
#                detect-nano/detect-small/detect-medium/detect-large/detect-base/keypoint-preview)
conda run -n rfdetr python src_detr/download_weights.py --models seg-nano seg-small

# 查看全部可下载项 / 全部下载 / 强制重下
conda run -n rfdetr python src_detr/download_weights.py --list
conda run -n rfdetr python src_detr/download_weights.py --all
conda run -n rfdetr python src_detr/download_weights.py --models seg-small --force
```

下载完成后文件落在 rfdetr 缓存目录（默认 `~/.roboflow/models/`），
`RFDETR*/train` 会直接命中、不再联网。缓存目录可用 `RF_HOME` 环境变量覆盖。

> **本项目根目录已放置** `rf-detr-seg-nano.pt` / `rf-detr-seg-small.pt` / `rf-detr-seg-medium.pt`
> （与 `yolo26*-seg.pt` 同位置），可直接用：
> ```powershell
> conda run -n rfdetr python src_detr/inference_video_detr.py --model rf-detr-seg-medium.pt ...
> ```
> 脚本会自动识别「官方预训练权重」并按其变体类加载（若是训练 checkpoint 则走 `from_checkpoint`）。

---

## 2. 训练 `train_detr.py`

RF-DETR 原生支持 YOLO 数据集格式，但要求目录布局为 `train/{images,labels}` +
`valid/{images,labels}`。项目现有数据集是 `images/train` + `labels/train` 布局，
因此脚本会在 `tmp/` 下自动搭一个 **视图**（用 junction/软链接指向原始图片和标签，
**不复制图片**），并写一份 `data.yaml` 指向该视图。

```powershell
# 默认: RF-DETR Seg-Medium (实例分割, 高光剪辑需要球桌/球网 polygon)
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --name pingpong_detr_v1

# 更大模型 / 更高分辨率 / 更多轮次
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --size large --resolution 672 `
    --epochs 150 --batch 2 --grad-accum 8 `
    --name pingpong_detr_l

# 纯检测 (不需要分割 mask)
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --task detect --size medium --name pingpong_detr_det

# 断点续训
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --resume runs_detr/pingpong_detr_v1/checkpoint.pth
```

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | 必需 | YOLO `data.yaml` 路径 |
| `--size` | `medium` | `nano/small/medium/large/xlarge/2xlarge` |
| `--task` | `seg` | `seg`=实例分割，`detect`=纯检测 |
| `--epochs` | `100` | 训练轮数 |
| `--batch` / `--grad-accum` | `16` / `1` | 单卡批大小与梯度累积（等效 batch=16）。显存大可直接调大 `--batch`（去梯度累积） |
| `--progress-bar` | `tqdm` | 控制台训练进度条风格：`tqdm` / `rich` / `none`（RF-DETR 默认关闭，这里默认开启） |
| `--workers` / `--prefetch-factor` / `--eval-batch-size` | `8` / `3` / `32` | 数据加载与验证并行度，减少 GPU 空转 |
| `--eval-interval` | `1` | 每 N epoch 验证一次（默认每轮都验证；调大可省时间，同时决定 `best_class_*.pth` 的更新频率） |
| `--class-weight` | `1.0` | **乒乓球类别的分类损失权重**（>1 给乒乓球加权，等价 YOLO 的 `--class-weight`） |
| `--resolution` | 模型自带 | 需能被 `patch_size*num_windows` 整除（seg/medium 为 24） |
| `--lr` | `1e-4` | 基础学习率 |
| `--no-pretrain` | 关 | 从零训练（无网络时可用） |
| `--resume` | 关 | 从 `checkpoint.pth` 续训 |
| `--device` | 自动 | `0` / `cuda:0` / `cpu` |
| `--name` / `--output` | `runs_detr/<name>` | 输出目录 |

训练输出（`runs_detr/<name>/`）常见 checkpoint：

- `checkpoint_best_total.pth` — **推理推荐使用**
- `checkpoint_best_ema.pth` / `checkpoint_best_regular.pth` — EMA / 原始权重最优
- `checkpoint.pth` — 最近一次（可续训）

OOM 时优先降低 `--batch` 并提高 `--grad-accum` 保持等效 batch=16，或加
`--gradient-checkpointing`。训练结束会打印**峰值显存**便于调参。

### 验证指标（每个 epoch）

RF-DETR 每轮验证后打印并记录（≈ YOLO 的 `mAP50` / `results.csv`）：

- `val/mAP_50`（**mAP50**）、`val/mAP_50_95`、`val/mAP_75`、`val/mAR`
- `val/precision` / `val/recall` / `val/F1`
- 分割模型额外：`val/segm_mAP_50` / `val/segm_mAP_50_95`
- **每类指标**（默认开启，`--no-log-per-class` 可关）：控制台每类表格 + `val/AP/<类名>`（如 `乒乓球`）

写入 `runs_detr/<name>/metrics.csv`（≈ YOLO 的 `results.csv`）与 TensorBoard。

> 与 YOLO 训练脚本的差异：RF-DETR 只按**总体** mAP 保存 `checkpoint_best_total.pth` /
> `checkpoint_best_ema.pth`，**不会**额外保存 `best_class_<id>_ap.pt` / `_map50.pt`。
> 若要「只按乒乓球 mAP 存最优」，需要另加回调（可按需实现）。

### 训练性能 / AMD ROCm 说明

`train_detr.py` 在 `import torch` 之前设置了 ROCm 环境变量（与 YOLO 训练脚本一致），
**之前缺失这些会导致 kernel 选择慢、显存碎片化**：

- `MIOPEN_FIND_MODE=FAST`、MIOpen 缓存目录、`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`（AMD Flash/高效 Attention；`--no-aotriton` 关闭）

AMP 精度用 `--amp-dtype {auto,bf16,fp16,none}` 控制（默认 `auto`；本机实测 auto 选 **bf16**，
比 fp16 更稳）。已用小数据集（6 train / 3 val，seg-nano，1 epoch）验证训练前向+反向正常。

### 训练加速（新增开关 + 社区结论）

- `--group-detr N`：训练时 `group_detr`（默认 13）。**调小（4/1）可大幅降低 decoder 计算与
  匹配开销**，但可能影响收敛。
- `--compile` / `--cuda-graphs`：`ModelConfig.compile` / `cuda_graphs`（实验，建议配 `--no-multi-scale`；
  ROCm 上可能不稳定）。
- `--fused-optimizer / --no-fused-optimizer`。
- `--no-amp`（= `--amp-dtype none`，fp32）：**ROCm 8060S 上混合精度可能反而更慢**
  （roc-torch 讨论 #5541），值得对比测试。

社区已知导致"分割训练慢"的点：
1. 验证阶段 mask→RLE 转换（pycocotools）是 CPU 瓶颈（官方 issue #616）→ 用
   `--eval-masks-head-resolution`（默认开）+ `--eval-max-dets 100` 缓解；
2. Windows/ROCm 无 Triton，匈牙利匹配回退到 CPU（SciPy）；
3. gfx1151/Windows 上 `hipblasGemmStridedBatchedEx` 缺 kernel（ROCm #6517）会走慢速路径；
4. RF-DETR 用 `num_queries×group_detr` 集合预测，训练天然比 YOLO 重。

**训练 batch / 显存实测**（seg-nano，312px，bf16-AMP，64 train 图，1 epoch）：

| batch | 峰值显存 | 结果 |
|-------|---------|------|
| 4 | ~2 GB（推算） | ✅（旧默认） |
| **16（默认）** | **8.07 GB** / 107 GB | ✅ |
| 32 | 15.65 GB / 107 GB | ✅ |
| 64 | — | ❌ `hipErrorInvalidConfiguration`（ROCm kernel 限制，非显存不足） |

结论：显存远非瓶颈，但 **ROCm 在 batch≥64 时会因 kernel 配置报错**，训练 `--batch` 建议 **≤32**。
默认 `--batch 16 --grad-accum 1`（等效 batch=16，与官方推荐一致）。seg-medium（432px）显存约为上表 **1.9×**。
训练结束会打印峰值显存，便于调参。

---

## 3. 推理 `inference_video_detr.py`

```powershell
conda run -n rfdetr python src_detr/inference_video_detr.py `
    --model runs_detr/pingpong_detr_v1/checkpoint_best_total.pth `
    --video Videos/match.mp4 `
    --output inference_output/detr_match
```

输出：

```
<output>/
├── raw_detections.jsonl      # 与 YOLO 版格式一致, 可被 pingpong_highlight.py 使用
└── pingpong_detection.mp4    # 可视化检测视频
```

要点：

- 分割模型会把球桌 / 球网的 mask 轮廓转成 `polygon` 字段；检测模型 `polygon` 为 `null`。
- 类别别名沿用 `inference_video.py` 的表（含中文 `乒乓球/球桌/球网`），语义与高光脚本一致。
- 批处理：`--video-list-file tmp/video_list.txt`（每行一个视频路径）。
- `--conf-ball/--conf-table/...` 可按类别设阈值；`--batch-size` 按显存调整。
- 快速测试：`--max-frames 8` 只跑前 N 帧。

### GPU / 性能说明（AMD ROCm 实测）

- 脚本启动时会打印设备与显存，并做一次**预热**：
  ```
  🖥️ GPU: AMD Radeon(TM) 8060S Graphics  | device=cuda:0
  🔥 预热完成 13.70s (MIOpen 已编译, 权重已在 GPU) | GPU 显存占用 224 MB
  ```
  这 ~10-14s 是 **MIOpen 一次性编译 kernel + 权重上卡**；首次遇到新 batch shape 会更久
  （如 batch=16 首次 ~58s），但结果会缓存到 `%LOCALAPPDATA%\miopen`，**第二次同 batch 降到 ~14s**。
  进度行也会显示 `GPU显存=xxxMB`，可据此确认 GPU 在工作。
- **Windows 任务管理器看不到 AMD 核显的 ROCm/HIP 计算**（它只统计 D3D）。要看占用请用
  AMD Adrenalin 的性能页，或安装 `amd-smi` / `pyrsmi`。

#### 推理优化（默认已开启，实测 2.3x 提速）

脚本默认启用两项官方/AMD 优化：

1. **`model.inference(compile=False, dtype=torch.float16)`** —— 官方 export 融合 + FP16
   （`--optimize {none,export,fp16}`，默认 `fp16`）；
2. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** —— 让 ViT attention 走 AMD 融合
   kernel，而非慢速 math 回退（`--no-aotriton` 可关闭）。

AMD Radeon 8060S（batch=4，fp32/fp16）实测：

| 配置 | ms/帧 | fps |
|------|-------|-----|
| baseline fp32（未优化） | 80 | 12.5 |
| export fp32 | 66 | 15.1 |
| export fp16 | 48 | 20.7 |
| export fp32 + AOTRITON | 64 | 15.6 |
| **export fp16 + AOTRITON（默认）** | **35** | **28.9** |

> 注意：`compile=True`（`torch.jit.trace`）在本仓库/ROCm 上会因 trace 分支固化产生
> **数值不一致**，故脚本固定使用 `compile=False`。
> `--optimize none` 可关掉全部优化做对照。

- **batch 选择**：优化后吞吐随 batch 上升，平台期约 **53 fps**（本机显存 107 GB，占用极小）：

  | batch | ms/帧 | fps | 峰值显存 |
  |------|-------|-----|---------|
  | 4 | 38.5 | 26.0 | 0.4 GB |
  | 16 | 21.5 | 46.6 | 0.8 GB |
  | 32（默认） | 20.6 | 48.4 | 1.3 GB |
  | 64 | 19.3 | 51.8 | 2.3 GB |
  | 128 | 18.8 | 53.3 | 4.3 GB |

  显存大可放心用 `--batch-size 64/128`。1080p 多目标场景可适当调小以省后处理开销。
  脚本会对**尾批 padding 到固定 batch**，避免不同 shape 触发 MIOpen/AOTRITON 重复编译。

> 上表为 `--optimize fp16` + AOTRITON 下的纯前向吞吐；真实视频还会叠加 1080p mask 上采样
> 与 CPU 绘制/写盘开销，实际帧率会低一些。

#### 进程退出防护（重要）

Windows + ROCm 上，即使 `os._exit()` 也会触发 `ExitProcess` → 逐个 DLL 的
`DLL_PROCESS_DETACH`；**AMD ROCm/MIOpen 的 DLL 在该阶段会卡死**，导致推理/训练
**任务完成后进程不退出**（CPU 空转、退出码非 0、残留僵尸进程）。

`inference_video_detr.py` 与 `train_detr.py` 的 `_exit()` 改用 Win32
`TerminateProcess` 强制结束（跳过 DLL 卸载），可立即退出。
**请勿把结尾的 `_exit(...)` 改回 `sys.exit(...)`。**

> 注意 `ctypes` 调用必须声明签名：`GetCurrentProcess()` 返回伪句柄 `-1`，
> 若未设 `restype=c_void_p` / `argtypes=[c_void_p, c_uint]`，句柄会被截断为 32 位而
> 变成无效句柄，`TerminateProcess` 静默失败并回落 `os._exit` → 仍会卡死。脚本已正确处理。

---

## 4. 一键流水线 `run_pipeline_detr.py`

```powershell
# 单视频
conda run -n rfdetr python src_detr/run_pipeline_detr.py `
    --model runs_detr/pingpong_detr_v1/checkpoint_best_total.pth `
    --video Videos/match.mp4

# 批量目录
conda run -n rfdetr python src_detr/run_pipeline_detr.py `
    --model <ckpt.pth> --video Videos/

# 已有 JSONL, 只做高光
conda run -n rfdetr python src_detr/run_pipeline_detr.py `
    --video Videos/match.mp4 --skip-inference --raw-detections inference_output/detr_match/raw_detections.jsonl
```

默认输出 `inference_output/pipeline_<视频名>/<视频名>_highlight.mp4`（与 YOLO 版 `run_pipeline.py` 完全一致）。

> **CLI 兼容**: 三个 RF-DETR 脚本均接受与对应 YOLO 脚本同名的参数。
> 语义可映射的别名会映射到 RF-DETR 等价参数（如 `--imgsz` → 分辨率、`--lr0` → 学习率、
> `--half` → FP16 推理、`--freeze` → 冻结 backbone、`--no-amp` → `--amp-dtype none`、
> `--freeze`/`--resume` 等）；RF-DETR 不支持的 YOLO 参数（如 `--mosaic/--mixup/--iou/--max-det`
> 等）仍会被接受，但会打印一行 `⚠️ RF-DETR 不支持参数 --xxx，已忽略` 的中文警告，不会报错。
> 训练侧 `--best-class`（-2=自动识别乒乓球）会保存 `best_class_<id>_ap.pth`；
> 因 RF-DETR 不提供每类 AP50，会打印 `ℹ️ RF-DETR 无每类 AP50，跳过 best_class_*_map50`。

---

## 5. 与 YOLO 版的对应关系

| YOLO (src/) | RF-DETR (src_detr/) |
|-------------|---------------------|
| `train_pingpong_v2.py` | `train_detr.py` |
| `inference_video.py` | `inference_video_detr.py` |
| `run_pipeline.py` | `run_pipeline_detr.py` |
| `pingpong_highlight.py` | 直接共用 `src/pingpong_highlight.py` |

> RF-DETR 的 `raw_detections.jsonl` schema 与 YOLO 版保持逐字段一致
> （`frame / timestamp / counts / ball_pos / detections[class, confidence, bbox, polygon]`），
> 所以两条技术路线的下游高光管线完全通用。
