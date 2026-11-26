# 🏓 乒乓彗星 TTComet · 乒乓球视频高光工具集

把一场乒乓球比赛的录像，自动剪成带特效的高光集锦。

全流程覆盖：**抽帧 → 标注 → 训练 → 推理 → 高光剪辑**。仓库里同时提供两种用法：

| 用法 | 适合谁 | 入口 |
|---|---|---|
| **A. 桌面安装包**（推荐） | 只想剪视频，不想碰命令行 | `TTComet-0.3.0-setup.exe` |
| **B. 命令行脚本链** | 要改算法、训模型、批量跑数据 | `src/` 下的脚本 |

> ### 💗 支持项目
>
> **爱发电主页：<https://ifdian.net/a/TTComet>**
>
> 如果这个工具帮你省了时间，可以到上面请作者喝瓶快乐水。
> 不打赏也完全没关系 —— 反馈问题、提建议、提 PR 一样是支持。

---

## A. 桌面版（乒乓彗星 TTComet）

本地运行的 WebUI 工具：选视频 → 自动推理 + 剪高光 → 网页里直接播放/下载。
不依赖 Python、不需要联网、**内置 ffmpeg**。

### 下载安装

从 [最新版本](https://github.com/TheGongyx/TTComet/releases/latest) 下载 `TTComet-*-setup.exe`（约 240 MB），
双击安装即可。安装包已内置 ffmpeg，目标机不需要装 Python 或任何运行环境。

> 这个链接始终指向最新正式版；已安装旧版的话，程序启动时也会自动检查更新，可直接在界面里下载安装。

安装**不需要管理员权限**（装到 `%LOCALAPPDATA%\Programs\TTComet`）。
数据（任务、预设、成品）在 `%LOCALAPPDATA%\TTComet\jobs\`。

### 功能

- **自动高光剪辑**：回合检测 + 切片，自带球轨迹彗尾、落点标记、迷你球台、球员光环
- **批量处理**：选文件夹，按文件名顺序跑完，模型只加载一次
- **中英双语界面**：右上角切换
- **两档精度**：高精度（准）/ 低精度（快约 2.8 倍），可随时切换
- **参数可调**：9 组 38 项，每项标注"调高/调低"的效果；支持保存多套预设并重命名
- **不剪辑模式**：只加特效、保留完整时长
- **实时反馈**：进度条 + 日志流，结果在线播放
- **输出目录可选**：留空就用默认目录，也可以指定任意文件夹（同名视频自动加序号，不覆盖）
- **处理记录**：每个任务自动落盘（`job.json`），刷新页面或重启程序后仍能在「③ 处理」里看到历史，点「查看」重新载入结果；被强行中断的任务会标「已中断」
- **无控制台窗口**：双击启动不弹黑框；任务栏托盘图标可「打开界面 / 退出程序」
- **检查更新**：启动时自动查一次 GitHub 最新版本（失败静默），也可点「检查更新」手动查；发现新版可一键下载并启动安装程序

### 已知限制

- **未做代码签名**：首次运行 Windows 可能提示"未知发布者"，点"更多信息 → 仍要运行"即可
- 运行日志在 `%LOCALAPPDATA%\TTComet\logs\app.log`（启动失败会弹提示框，日志里有详细原因）
- 同一时间只允许一个实例（避免抢 GPU）
- 建议有独立显卡（AMD / NVIDIA 走 DirectML）；纯 CPU 能跑但很慢

---

## 📁 项目结构

```
.
├── src/                                  # 命令行脚本链
│   ├── inference_video.py                #   逐帧推理 → raw_detections.jsonl
│   ├── pingpong_highlight.py             #   高光特效剪辑（轨迹 + 彗尾 + 迷你球台）
│   ├── run_pipeline.py                   #   推理 + 高光 一键串联（支持目录批量）
│   ├── extract_missing_ball.py           #   漏检帧提取（球跳变帧）
│   ├── extract_rejected_ball_frames.py   #   错误球帧收集（负样本）
│   ├── random_sample_frames.py           #   通用随机抽帧（支持无日志）
│   ├── remove_narration.py               #   去人声（demucs / ONNX / 卡拉OK相减 / 静音）
│   ├── remove_watermark.py               #   去静态水印（delogo / OpenCV inpaint）
│   ├── train_pingpong_v2.py              #   分割模型训练（bbox→polygon 自动转换）
│   ├── export_onnx.py                    #   .pt → .onnx 导出（支持 FP16 / NMS）
│   ├── split_all1.py / split_all19.py    #   数据集拆分 / 增量合并
│   └── convert_yolo_to_labelstudio.py    #   YOLO 标注 → Label Studio
│
├── src_yls/                              # Label Studio ML 后端（自动预标注）
│
├── webui/                                # 乒乓彗星 TTComet 桌面版
│   ├── run_server.py                     #   启动器（exe 入口）
│   ├── app/                              #   ONNX 推理引擎 + FastAPI + 前端
│   ├── models/                           #   两档 ONNX 模型（高精度 / 低精度）
│   ├── design/                           #   图标生成脚本
│   ├── packaging/                        #   PyInstaller + Inno Setup 打包
│   └── bench/                            #   引擎基准与结果对拍
│
├── models/                               # 自定义训练权重（.pt 不入库）
├── train_data/                           # 标注数据集（内容不入库，见下方说明）
├── yolo26*-seg.pt                        # 预训练权重（不入库）
├── runs/                                 # 训练输出
├── Videos/  transition_videos/           # 素材视频（不入库）
└── tmp/                                  # 脚本临时文件
```

> **关于 `train_data/`**：目录结构保留，但**图片与标签内容不进版本库**
> （数据集动辄数十 GB，不适合放 Git）。本地放好 `images/ + labels/ + data.yaml` 即可正常训练。

---

## 🚀 环境要求

> 下面的命令都假设你已经激活了**装好依赖的 Python 环境**（conda / venv 均可），
> 也就是 `python` 指向该环境。

| 依赖 | 说明 |
|---|---|
| Python 3.10+ | 建议用 conda 或 venv 隔离 |
| PyTorch + ultralytics | 抽帧推理 / 训练 |
| `demucs` | 可选，只有 `remove_narration.py --method demucs` 才需要（`pip install --no-deps demucs` + `julius einops omegaconf pyyaml tqdm lameenc huggingface-hub safetensors sphn`，用 `--no-deps` 避免替换已有 torch） |
| `label-studio` | 可选，只有做标注时才需要 |
| ffmpeg | 抽帧 / 推理 / 剪辑必需（桌面安装包已内置） |
| GPU | AMD（ROCm）或 NVIDIA（CUDA），框架自动检测；也可纯 CPU 跑 |

---

## 🔧 脚本一览

| 脚本 | 用途 |
|---|---|
| `inference_video.py` | 逐帧推理 → `raw_detections.jsonl` + 可视化 mp4 |
| `pingpong_highlight.py` | 高光特效剪辑（需要 video + jsonl） |
| `run_pipeline.py` | 一键流水线：推理 → 高光（单视频或目录批量） |
| `extract_missing_ball.py` | 漏检帧提取（球跳变帧） |
| `extract_rejected_ball_frames.py` | 错误球帧收集（A/B 两类负样本） |
| `random_sample_frames.py` | 通用随机抽帧（`--ball-only` / `--no-player` / `--no-log`） |
| `remove_narration.py` | 去人声，保留击球声/环境声（`--method demucs` / `demucs-onnx` / `center` / `mute`），默认再提音量 1 倍 |
| `remove_watermark.py` | 去静态水印（`--method delogo` / `inpaint`；`logo`/`overlay`/`all` 预设默认 `overlay` 保留比分显示；`--ref-size` 分辨率自适应；`--detect` 自动检测位置） |
| `train_pingpong_v2.py` | 分割模型训练（best_class AP / mAP50 双追踪） |
| `export_onnx.py` | `.pt` → `.onnx`（`--half` / `--dynamic` / `--simplify` / `--nms`） |
| `split_all1.py` | 数据集拆分 train/val |
| `split_all19.py` | 数据集增量合并拆分（专用） |
| `convert_yolo_to_labelstudio.py` | YOLO 标注 → Label Studio 格式 |

---

## 📋 快速上手

> 命令在 PowerShell 中执行，换行用反引号 `` ` ``；Windows 用 `python`，Linux/WSL 用 `python3`。

### 1️⃣ 一键出高光（最常用）

```powershell
# 单个视频
python src/run_pipeline.py --video Videos/match.mp4

# 批量处理整个目录
python src/run_pipeline.py --video Videos/
```

产物默认在 `inference_output/pipeline_<视频名>/`。

### 2️⃣ 分步执行（要调参时）

```powershell
# 推理
python src/inference_video.py `
    --video Videos/match.mp4 --model models/m960n.pt --output inference_output/

# 剪辑（可反复调参，不用重跑推理）
python src/pingpong_highlight.py `
    --video Videos/match.mp4 `
    --raw-detections inference_output/raw_detections.jsonl `
    --output highlight.mp4
```

### 3️⃣ 抽帧标注

```powershell
# 漏检帧（上一帧有球、当前无球、下一帧又有球）
python src/extract_missing_ball.py `
    --log inference_output/raw_detections.jsonl --video Videos/match.mp4 --count 50

# 随机抽帧（只抽有球场景 / 直接抽帧）
python src/random_sample_frames.py `
    --log inference_output/raw_detections.jsonl --video Videos/match.mp4 `
    --count 50 --ball-only
```

---

## ✂️ 音视频清理（去水印 / 去人声）

两个独立的后期处理脚本，和上面的高光流程互不干扰，可以任意组合。
都只需要 **ffmpeg**；`remove_narration.py` 的 AI 分离额外需要 `demucs`（见「环境要求」）。

### 去静态水印 `remove_watermark.py`

针对画面里**时间上位置固定**的叠加层（角标、信息条、文字）。

内置预设（坐标基于 1920×1080 的常见画面布局，其他分辨率会**自动等比缩放**）：

| 预设 | 处理内容 | 适用场景 |
|---|---|---|
| `logo` | 右上角角标 | 只想去角标 |
| `overlay`（默认） | 角标 + 底部信息条 | 保留比赛信息 |
| `all` | 再加左下角比分显示 | 比分也不需要时 |
| `none` | 不用预设，只处理 `--region` 指定的区域 | 其他来源的水印 |

```powershell
# 默认：去角标 + 底部信息条，保留比分显示
python src/remove_watermark.py --video Videos/match.mp4

# 只去角标 / 连比分显示一起去
python src/remove_watermark.py --video Videos/match.mp4 --preset logo
python src/remove_watermark.py --video Videos/match.mp4 --preset all

# 手动指定区域（可重复），x,y,w,h 按实际画面像素
python src/remove_watermark.py --video in.mp4 --preset none --region 100,100,200,50
```

**自动检测未知水印位置。** 换来源、换素材的角标位置不一样时，不用手量像素：

```powershell
# 只输出候选框和可疑度，不动视频
python src/remove_watermark.py --video in.mp4 --detect

# 确认候选无误后直接处理
python src/remove_watermark.py --video in.mp4 --detect --detect-apply
```

检测原理是对多帧做逐像素时序统计：水印叠加使得该区域**最小亮度偏高**、**时间标准差极低**，
再结合填充率和宽高比打分。默认只报告不处理，`--detect-apply` 时也只处理
`score ≥ 50% × 最高分` 的候选，避免把白球衣、广告板这类静止亮色物体一起抹掉。

**两种算法：**

| 方法 | 速度 | 效果 |
|---|---|---|
| `delogo`（默认） | 快 | 区域被周边像素插值成模糊块，复杂背景上有拖影 |
| `inpaint` | 慢 | OpenCV TELEA 逐帧修复，纯色背景（地板）更好，复杂背景拖影更明显 |

**先看效果再动手**（导出前后对比静帧，不编码视频）：

```powershell
python src/remove_watermark.py --video in.mp4 --preview 3,900,2100 --dry-run
```

实测：1920×1080 / 46 分钟影片，**约 4.8 分钟**（9.75x 实时）。

> **边界**：只处理矩形区域、只对时间上静止的水印有效。滚动/闪烁/位置漂移的水印、
> 与背景亮度接近的水印、以及异形 logo（会被多抹一点背景）都不适用。

### 去人声 `remove_narration.py`

把视频中的人声去掉，保留击球声、观众声和场馆环境音。

| 方法 | 说明 | 速度 | 效果 |
|---|---|---|---|
| `demucs`（默认） | AI 人声分离（htdemucs），只保留 `no_vocals` 轨 | 慢，需 GPU 更佳 | 最彻底 |
| `center` | 立体声反相相减（人声通常居中） | 秒出 | 人声有残留 |
| `mute` | 直接删除音轨 | 秒出 | 完全静音 |

```powershell
# 默认：demucs 分离 + 音量提升 1 倍
python src/remove_narration.py --video Videos/match.mp4

# 卡拉OK 相减（零依赖，快速试听）
python src/remove_narration.py --video in.mp4 --method center

# 音量不变 / 提到 2.5 倍 / 再压一点分离残留
python src/remove_narration.py --video in.mp4 --volume-gain 1.0
python src/remove_narration.py --video in.mp4 --volume-gain 2.5
python src/remove_narration.py --video in.mp4 --residual 0.15
```

**音量处理**：分离后人声没了，整体响度会明显下降，所以默认 `--volume-gain 2.0`
（+6dB）把音轨提回来，并在增益后接 `alimiter` 限幅 —— 只在峰值超过 0dBFS 时压一下，
正常段落不受影响，不会整体变闷。

**视频流默认直接复制**（`-c:v copy`），只重编码音频，所以几乎不影响画质、速度也快。

实测：46 分钟影片的音频分离约 **2 分 15 秒**（22x 实时），合成 45 秒。

> 首次运行会从 HuggingFace 下载 htdemucs 权重（约 80 MB）。国内网络慢可以设
> `$env:HF_ENDPOINT = "https://hf-mirror.com"`。
>
> **注意**：demucs 会把人声连同部分环境声一起划进 vocals，所以去人声后击球声会变弱。
> 觉得太闷就用 `--method center` 对比，或调小 `--volume-gain`。

### 组合使用

先去水印（视频重编码），再去人声（视频流复制，很快）：

```powershell
python src/remove_watermark.py --video Videos/match.mp4
python src/remove_narration.py `
    --video Videos/match_nowatermark.mp4 --output Videos/match_clean.mp4
```

---

## 🧠 模型

### 分割模型（6 类）

| class_id | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 类别 | 乒乓球 | 其他比赛球 | 球员 | 球拍 | 球桌 | 球网 |

### 桌面版用的两档 ONNX

| 档位 | 文件 | 大小 | 模型前向 | 端到端 |
|---|---|---|---|---|
| 高精度 | `m960n_fp16_nonms.onnx` | 45 MB | 20.5 ms/帧 | ~40 fps |
| 低精度 | `s960n_fp16_nonms.onnx` | 20 MB | 7.4 ms/帧 | ~65 fps |

导出方式：

```powershell
python src/export_onnx.py `
    --model models/m960n.pt --output webui/models/m960n_fp16_nonms.onnx `
    --imgsz 960 --simplify --half --opset 17 --device 0
```

> 注意不要加 `--nms`：内嵌 NMS 会把 0.25 阈值烧进模型，导致 `--conf-ball 0.10` 的低分球召回失效。

---

## 🏋️ 模型训练

```powershell
# 1. 拆分 train/val（必须）
python src/split_all1.py `
    --src train_data/all22 --dst train_data/all22_split

# 2. 训练
python src/train_pingpong_v2.py `
    --data train_data/all22_split/data.yaml `
    --pretrained yolo26x-seg.pt `
    --name my_run --epochs 100 --batch 4
```

| 参数 | 说明 |
|---|---|
| `--data` | `data.yaml` 路径（必需） |
| `--pretrained` | 预训练权重，默认 `yolo26x-seg.pt`，自定义用 `models/m960n.pt` |
| `--name` | 输出到 `runs/train_<name>/` |
| `--epochs` / `--batch` | 训练轮数与批大小（显存不足调低 batch） |

---

## 🔖 Label Studio 标注

```powershell
# 标注服务（Windows 原生）
label-studio start --port 8080   # http://localhost:8080

# ML 后端（自动预标注）
python src_yls/_wsgi.py --port 9090 --host 0.0.0.0
```

标注数据导出为 YOLO 格式后放入 `train_data/<name>/`（`images/ + labels/ + data.yaml`）。
也可用 `convert_yolo_to_labelstudio.py` 把已有 YOLO 标注转成 Label Studio 任务继续标注。

---

## 📊 训练结果解读

| 指标 | 说明 | 优秀值 |
|---|---|---|
| `mAP@50` | IoU=0.5 时的平均精度 | >0.85 |
| `mAP@50-95` | IoU 0.5~0.95 的平均精度（更严格） | >0.60 |
| `Precision` | 预测为正的样本中真正为正的比例 | >0.80 |
| `Recall` | 真实正样本中被正确找出的比例 | >0.75 |

```
runs/train_<name>/
├── results.csv              ← 每轮指标
├── confusion_matrix.png     ← 混淆矩阵
├── labels.jpg               ← 标注分布
└── weights/
    ├── best.pt              ← 最佳模型（部署用）
    ├── last.pt
    └── best_class_*_ap.pt   ← 按单类 AP 追踪的模型
```

---

## 💡 使用技巧

**数据量建议**

| 数据量 | 场景 | 推荐 epochs |
|---|---|---|
| <50 张 | 快速验证 | 50-100 |
| 50-200 张 | 小规模微调 | 100-200 |
| >200 张 | 生产级 | 200-500 |

**显存不足（OOM）**：降低 `--batch`（默认 4，可降到 2 或 1）。

**查看训练曲线**：

```python
import pandas as pd
df = pd.read_csv('runs/train_my_run/results.csv')
print(df[['epoch', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)']].tail(20))
```

---

## ⚠️ 注意事项

1. **先激活装好依赖的环境**，再执行 `python src/<script>.py`（conda 用 `conda activate <环境名>`）
2. **训练启动后不要持续监控进程**：确认启动正常即可，持续查看会拖慢训练
3. **标注格式**：YOLO txt（`class_id x_center y_center width height`，归一化）；也接受 polygon（9+ 值），训练脚本会自动转换
4. **`data.yaml` 的 `path` 必须指向 `images/` `labels/` 的父目录**，且训练前必须拆分 train/val，否则会静默失败
5. **临时文件统一放 `tmp/`**
6. **`train_data/` 内容不入库**：只保留目录结构，数据集请在本地准备

---

## 📄 许可

本仓库目前**未声明开源许可证**（默认保留所有权利）。如需引用或二次分发，请先联系仓库所有者。

第三方组件遵循各自许可：Ultralytics（AGPL-3.0）、ONNX Runtime（MIT）、three.js（MIT）。

## 📚 更多

构建、打包、参数与架构细节见 [`webui/app/README.md`](webui/app/README.md)。
