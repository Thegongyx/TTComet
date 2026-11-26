# 乒乓彗星 TTComet —— WebUI 与 ONNX 推理引擎

把 `src/inference_video.py`（torch + ultralytics）的推理链路替换为纯
`onnxruntime` 实现，输出与其**同 schema** 的检测日志，供 `src/pingpong_highlight.py`
直接消费。不依赖 torch / ultralytics，为打包成 exe 做准备。

## 文件

| 文件 | 作用 |
|---|---|
| `onnx_engine.py` | 模型加载 + letterbox 预处理 + NMS + mask→polygon + 记录组装 |
| `inference_job.py` | 推理主流程（VFR→CFR、三段流水线、进度/日志回调、取消检查） |
| `run_video.py` | 推理命令行（薄壳，实现在 `inference_job`） |
| `video_utils.py` | VFR 检测 / CFR 转换（与 `src/inference_video.py` 逐行对齐） |
| `highlight_bridge.py` | **以模块方式加载 `src/pingpong_highlight.py` 并调用其 `main()`** |
| `postprocess_bridge.py` | 后处理：加载 `src/remove_watermark.py` / `src/remove_narration.py`（去水印、去人声，默认关闭） |
| `jobs.py` | 串行任务队列 + 编排（推理 → 高光 → 后处理 → 收尾压缩） |
| `server.py` | FastAPI：上传 / SSE 进度 / 结果下载 / 环境自检 |
| `static/index.html` | 单页界面 |
| `static/pingpong-scene.js` | three.js 场景：两只小动物打乒乓球（纯程序化建模） |
| `static/vendor/three.module.min.js` | three.js 0.160.0（本地化，离线可用，655 KB） |
| `../run_server.py` | 启动器：本地起服务 + 打开浏览器（`TTComet.exe` 的入口） |
| `../models/m960n_fp16_nonms.onnx` | 高精度档（fp16、稠密输出、batch=1，45.2 MB） |
| `../models/s960n_fp16_nonms.onnx` | 高速档（同上格式，20.1 MB） |

## 启动 WebUI

```powershell
# 开发模式（工作目录在 webui/workspace/）
$env:PPT_DEV="1"; $env:PPT_PORT="8123"
webui\.venv\Scripts\python.exe webui\run_server.py

# 正常模式（工作目录在 %LOCALAPPDATA%\TTComet\jobs\）
webui\.venv\Scripts\python.exe webui\run_server.py
```

浏览器会自动打开 `http://127.0.0.1:<port>/`。服务只监听回环地址，不对外网开放。

### 输入方式（对齐 `run_pipeline.py`）

- **选一个视频文件** → 单视频处理
- **选一个文件夹** → 扫描该目录下所有视频（只扫当前层，按文件名排序），批量串行处理
- 目录批量时 **模型只冷启动一次**；跨任务也会复用已加载的会话（日志里会写
  "首次加载模型" 或 "复用已加载的模型"）
- 每个视频一个独立子目录：`<job>/00_<名>/`，内含 `video_cfr.mp4`（VFR 时）、
  `raw_detections.jsonl.gz`、`highlight.mp4`

界面内置本地文件浏览器（盘符 / 常用目录 / 双击进入 / 面包屑跳转），不需要把大文件
再上传一遍。

### 模型档位（高精度 / 低精度）

两个模型导出参数完全一致（fp16、稠密输出、batch=1、imgsz 960），用户只看到"精度档位"，
不暴露文件名和体积：

| 档位 | 内部文件 | 大小 | 模型前向 | 端到端（12s 片段） |
|---|---|---|---|---|
| **高精度**（默认） | `m960n_fp16_nonms.onnx` | 45.2 MB | 20.5 ms/帧（48.8 fps） | 40.2 fps |
| **低精度** | `s960n_fp16_nonms.onnx` | 20.1 MB | 7.4 ms/帧（134 fps） | 65.5 fps |

- 模型层快 **2.76 倍**，端到端快 **1.63 倍**（后处理与写盘是固定开销，会吃掉一部分收益）。
- 两个模型各自常驻缓存：切回已加载过的档位只替换检测阈值，不重新初始化 DirectML
  （日志里会写"加载模型[x]"或"复用已加载模型[x]"）。
- 档位是普通参数，会跟着预设一起保存，所以可以存"高速-快速出片""高精度-比赛成片"两套。
- 接口：`GET /api/models` 返回可用档位、默认档位与已加载列表。

### 托盘图标与启动方式

打包版是**窗口子系统程序**，双击不弹控制台：

- 任务栏右下角出现托盘图标：**左键**打开界面，**右键**菜单可选「打开界面 / 退出程序」
- 日志写到 `%LOCALAPPDATA%\TTComet\logs\app.log`（超过 2MB 自动滚动，保留 5 份）
- 启动失败（端口被占、模型缺失等）会弹原生提示框，避免"双击没反应"
- 默认端口 `8760`，被占用时自动换随机端口；实际地址写入 `%LOCALAPPDATA%\TTComet\last_url.txt`

> 源码运行（`python run_server.py`）时仍是控制台程序，日志同时打到终端和文件。

### 输出目录

界面「① 选择视频」里多了一行**输出目录**：

- 留空 → 成品放在 `jobs/<任务>/<序号>_<视频名>/highlight.mp4`（默认行为）
- 选一个文件夹 → 成品直接写到该目录，命名 `<原视频名>_highlight.mp4`；同名自动加 `_2`、`_3`，不覆盖
- 后端 `/api/jobs` 会校验：目录可创建、是目录、可写，否则返回 400

### 处理记录（刷新/重启不丢）

以前刷新页面，`③ 处理` 里的结果和日志就没了（任务只存在内存里）。现在：

- 任务创建、开始、每个视频完成、全部结束时都会把快照写到
  `jobs/<任务ID>/job.json`（JSON，约 2 KB，不含日志全文）
- **刷新页面** → 自动把最近一次任务的结果铺回结果区，日志也重新拉一遍（`/api/jobs/{id}/logs`）
- **重启程序** → 启动时扫描 `jobs/*/job.json` 恢复历史（最多 80 条），出现在「③ 处理 → 处理记录」里
- 老版本留下的任务目录（没有 `job.json`）也能反推：只要 `<序号>_<视频名>/highlight.mp4` 还在，就会生成一条历史
- 状态判定：
  | 落盘状态 | 恢复后 |
  |---|---|
  | `done` / `failed` / `cancelled` | 原样保留 |
  | `queued` / `running`（说明当时被强杀） | `interrupted`（已中断），未完成的视频也标中断 |
  | 视频有 `output` 且文件存在 | 一律算 `done`（成品在就是跑完了） |
  | 视频有 `output` 但文件被删/挪走 | `failed`，「成品文件已不存在」，不再给一个点不开的播放按钮 |

每行显示：时间、来源（视频/目录名）、`共 N 个 · 成功 x · 失败 y`、状态、`查看` 按钮。
`查看` 只重新载入结果，不会重新处理；「收起」可以把列表折起来。

### 退出程序

页面右上角有「退出程序」按钮，会弹确认框，确认后本地服务停止、进程退出（打包版的控制台窗口一起关掉）。

- 有任务在排队或运行时会被拒绝（返回 409），需要先等它完成或手动取消
- 接口要求自定义头 `X-TTComet-Request: 1`：浏览器对跨域自定义头会先发 preflight，
  本服务不放行 CORS，所以其它网页无法偷偷调用这个接口关掉程序
- 也可以直接关控制台窗口（等价，但不会清理得更干净）

### 在线更新

启动时自动检查一次（静默，失败不提示），也可以点状态栏右侧「检查更新」手动查。

- **只查询 GitHub Releases**（`TheGongyx/TTComet` 的 `/releases/latest`），Gitee 仓库不公开，不作为更新源
- 查询成功缓存 6 小时，失败缓存 30 分钟；超时 5 秒，断网/被墙都只是静默失败，不影响出片
- 发现新版本时状态栏下方弹出提示：**下载并安装** / 更新说明 / 关闭
- 点「下载并安装」后后台线程下载安装包到 `%TEMP%\TTComet-update\`，前端显示百分比进度；
  下载地址限定 GitHub 域名，若 Release 提供 `sha256` 摘要会自动校验，不匹配则删除并报错
- 下载完成后点「立即安装」：启动安装包（`/CLOSEAPPLICATIONS /NORESTART`），本程序随即退出，
  避免安装程序无法覆盖正在运行的文件；安装完成后重新打开即可
- 有任务在运行时拒绝安装（409），与「退出程序」一致

接口：`GET /api/update?force=1`、`POST /api/update/download`、`GET /api/update/download`、
`POST /api/update/install`（后两个 POST 需要 `X-TTComet-Request: 1`）。

### 后处理：去水印 / 去人声（默认关闭）

高光成片之后可以再走两步可选后处理，**默认都不启用**，在参数区「后处理」组里勾选才执行：

```text
推理 -> 高光剪辑 -> 去水印（默认方案 / 自动检测）-> 去人声（demucs-onnx / center / mute）-> 最终 mp4
```

- **去水印**：复用 `src/remove_watermark.py`，界面上只提供两种方式：
  - **默认方案**：处理右上角 + 底部的固定叠加区域（内置坐标，按 1920×1080 基准自动缩放），保留比分显示；
  - **自动检测水印位置**：先统计多帧再处理，适合位置不确定的素材。
  两种方式都使用 delogo 算法；需要自定义区域或改用 inpaint 时，请直接运行命令行脚本。
- **去人声**：复用 `src/remove_narration.py`。WebUI 提供三种方式：
  - `demucs-onnx`（默认推荐）：HT-Demucs 的 ONNX 版，`onnxruntime` 推理，**不需要 torch**；
    模型缓存在 `webui/models/demucs-onnx-cache/`（约 158 MB，随安装包一起分发，离线可用）。
  - `center`：立体声反相相减，秒出，但音场会变薄。
  - `mute`：直接静音。
- 成片没有音轨、或参数里勾了「去掉原声」时，自动跳过去人声。
- 中间结果先写成 `highlight.wm.tmp.mp4` / `highlight.narr.tmp.mp4`，全部成功后才替换
  `highlight.mp4`；任一步失败会保留原高光成片，并把任务标为失败。
- `demucs`（torch 版）不在 WebUI 里提供：打包版没有 torch，源码版 `.venv` 也没装。
  需要它时用命令行流水线 `--narration-method demucs`（`yolo26` 环境）。

已知限制：

- DirectML 在部分核显/驱动组合上会直接报错（`DmlFusedNode` 非零状态）**或静默输出全零**，
  所以 demucs-onnx 的 `auto` 固定走 CPU，并在分离后检查是否全静音（异常则回退 CPU）。
  本机实测 CPU 分离约 **2 倍实时**。
- 自动检测只能找「一直亮且不随时间变化」的静态叠加，检不出半透明比分显示、滚动水印。

### 参数预设

调好一套参数后可以存下来反复用：

- **保存当前参数**：把当前"生效中"的参数（即未被模式禁用的项）存为一套预设，弹窗里填名字
- **多套共存**：下拉列表显示 `名称（N 项）`，最多 50 套
- **应用**：先恢复全部默认值，再套用预设，同时刷新参数的可达性（比如预设里是"不剪辑"，
  那三个"仅剪辑"分组会自动置灰）
- **重命名 / 删除**：带输入校验（空名、超长、重名都会拦下）和删除确认

存储位置：`<workspace>/presets.json`（跟着工作目录走，换浏览器或清缓存都不会丢）。
接口：`GET/POST /api/presets`、`PUT/DELETE /api/presets/{id}`，
重名返回 409，不存在返回 404。

### 可调参数（10 组，默认全部折叠）
| 分组 | 业务含义 | 参数 |
|---|---|---|
| **剪辑与输出** | 成片形态 | 是否按回合剪辑、输出高度、编码器、去原声 |
| **检测阈值** | 模型识别各类目标的置信度下限 | 乒乓球、球桌、球网、球员、NMS IoU |
| **球轨迹与回合判定** | 轨迹鲁棒性 + 回合起止规则 | 回合起点判定、高光阶段球置信度、静止球过滤、最大跳变、球丢失/出台/无落点超时 |
| **回合有效性**（仅剪辑） | 哪些回合值得剪进成片 | 最小回合帧数、短回合豁免落点数、球速前 N 豁免 |
| **片段边界**（仅剪辑） | 回合前后的留白与合并 | 前缓冲、后缓冲、合并间隔 |
| **前置精彩集锦**（仅剪辑） | 成片开头的集锦 | 关闭开关、取前 N、最小时长、落点条件、黑幕间隔 |
| **视觉效果** | 轨迹彗尾/迷你球台/光环 | 特效强度、球残影时长、集锦/正片特效组合 |
| **后处理**（默认关闭） | 成片后去水印 / 去人声 | 去水印开关、处理方式（默认方案 / 自动检测）；去人声开关、方式、音量倍数 |
| **高级 · 几何与调试** | 球台几何与轨迹关联 | 走廊容差/延伸/剔除上限、中心距离、出台容差、关联算法、调试输出 |

标注"仅剪辑"的三组在关闭"按回合剪辑"后**整组置灰且参数不提交**；
"关闭前置集锦"勾选后，集锦组内除开关本身外也全部置灰。

参数默认值与源脚本 argparse 完全一致；留空的数值项表示"用脚本默认值"。

### 相比 torch 版，哪些参数被删掉了

迁移到 ONNX 后，`src/inference_video.py` 里这些参数已经**没有意义**，WebUI 不提供：

| 已移除 | 原因 |
|---|---|
| `--device` / `--half` | 模型已是 fp16，"设备"由执行后端自动选择（DirectML → CPU） |
| `--imgsz` / `--batch-size` | 导出时固定为 960 / batch=1，改不了（DirectML 不适合动态维度） |
| `--post-workers` / `--reader-threads` / `--prefetch-batches` / `--overlap*` | torch/ultralytics 专属的性能调优，与新实现无关 |
| `--writer-backend` / `--out-width` / `--no-detection-video` | 这些只影响"检测可视化视频"，流水线本来就不生成 |
| `--empty-cache-every` / `--conf-margin` / `--end2end` / `--max-det` / `--log-flush-interval` / `--amf-quality` / `--amf-qp` | 实现细节或被新后处理取代 |
| `--no-polygon` | 球台/球网几何依赖 polygon，关掉等于废掉落点判定 |
| **`conf_racket`**（本次删除） | 高光脚本完全不使用 `racket` 类，调了不会有任何效果 |
| **`court_cluster`**（本次删除） | 源脚本自述"实测会误删约 47% 的真球检测"，默认关闭，暴露给用户只有害处 |


### 剪辑开关（`--cut / --no-cut`，默认剪辑）

关闭剪辑（`--no-cut` / WebUI 里取消勾选"按回合剪辑"）时：

- 输出**整段视频**，不剔除捡球、走动、短回合等片段
- 特效照常渲染（球轨迹、彗尾、迷你球台、落点标记、球员光环）
- 回合仍会检测，但只用于落点标注的分组边界，让特效与剪辑模式保持同一套逻辑
- 自动关闭"前置精彩集锦"（整段本身就是完整内容）
- 不会因为"没有有效回合"而失败退出（这正是需要整段加特效的场景）


### 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 后端/ffmpeg/模型/磁盘自检 |
| GET | `/api/fs/roots` | 盘符与常用目录 |
| GET | `/api/fs/list?path=` | 目录浏览（子目录 + 视频） |
| POST | `/api/jobs` | `{path, params}`，path 为文件或目录 |
| GET | `/api/jobs` | 任务列表 |
| GET | `/api/jobs/{id}` | 任务详情（含每个视频状态） |
| GET | `/api/jobs/{id}/logs` | 完整日志 |
| GET | `/api/jobs/{id}/events` | SSE：进度 / 日志 / 状态 / 视频列表 |
| GET | `/api/jobs/{id}/videos/{idx}` | 单个结果视频（支持 Range） |
| POST | `/api/jobs/{id}/cancel` | 取消 |
| POST | `/api/shutdown` | 退出程序（需 `X-TTComet-Request: 1` 头；有任务在跑时返回 409） |
| GET | `/api/update?force=1` | 检查 GitHub 最新版本（带缓存，失败静默） |
| POST | `/api/update/download` | 下载新版安装包（需确认头；后台线程） |
| GET | `/api/update/download` | 下载进度 `{state, percent, path, error}` |
| POST | `/api/update/install` | 启动安装程序并退出本程序（需确认头；有任务时 409） |


## 高光逻辑一致性

`highlight_bridge.py` **不复制**任何剪辑逻辑，而是：

1. 用 `importlib` 从 `src/pingpong_highlight.py` 加载模块；
2. 构造 argv，临时替换 `sys.argv` / `sys.stdout`；
3. 调用其 `main()`，把 stdout 逐行转发成任务日志；
4. 捕获 `SystemExit`（源脚本用 0/1/2 三种退出码，2 = 没有有效回合，属业务结果）。

因此参数默认值、回合判定、特效渲染全部沿用源脚本，改动源脚本 WebUI 自动跟随。
打包时只需把该 `.py` 作为资源文件带上。


## 用法

```powershell
# 完整视频
webui\.venv\Scripts\python.exe webui\app\run_video.py `
    --video Videos/xxx.mp4 --output webui/out/xxx

# 只跑前 600 帧（调试）
webui\.venv\Scripts\python.exe webui\app\run_video.py `
    --video Videos/xxx.mp4 --max-frames 600
```

参数较多，常用的：
`--ep auto|dml|cpu`、`--conf-ball`、`--conf-table`、`--no-polygon`、`--serial`（关流水线）。

## 设计要点

1. **模型是稠密输出（导出时未带 `--nms`）**，NMS 在 Python 侧做，因此可以按类别设不同
   置信度阈值。带 `--nms` 的导出会把 0.25 阈值烧进计算图，`--conf-ball 0.10` 会失效。
2. **batch 固定为 1**。实测 batch=8 单帧慢 22%，集成 GPU 没有吞吐摊薄。
3. **三段流水线**：解码+预处理 / 推理 / 后处理+写盘 各占一个线程，墙钟由最慢阶段决定。
4. **polygon 只对 table/net 计算**（与旧实现一致），其余类别 `polygon=None`。
5. **mask→polygon 必须合并多块连通域**。proto 只有 240×240，球台远端细边会断成独立小块，
   只取最大轮廓会让拟合出的四边形缺边（实测球台 polygon 上半部分缺 46px，直接导致
   落点/回合判定失效）。下游用 `cv2.convexHull` 拟合，点序无关，所以按面积过滤后全部拼接。

## 实测（40 秒片段，1920×1080 @58.5fps，2353 帧）

| 指标 | torch/ROCm（基准） | ONNX/DirectML |
|---|---|---|
| 端到端吞吐 | — | **42.8 fps**（1.37x 实时） |
| 各阶段 | — | decode 3.9 / **pre 19.5** / **infer 21.3** / post 10.5 / write 0.6 ms |
| 主轨迹点数 | 417 | 407 |
| 回合 / 事件 | 3 / 40 | 4 / 36 |
| 高光输出帧数 | 892 | 1093 |
| bbox 匹配率 (IoU≥0.5) | — | 91.1%，平均 IoU 0.945 |
| table / net / player | — | 99.5% / 99.9% / 98.9% |
| pingpangball | — | 85.6%（多检出 688 个，低分球召回更高） |
| racket | — | 70.5%（高光流程不使用该类，无影响） |

## WebUI 端到端实测（同一个 40 秒片段，走完整 HTTP 流程）

```
上传 32.8 MB -> 排队 -> 推理 2353 帧 (42.1 fps, 自动 VFR→CFR)
             -> 高光渲染 1112 帧 (源脚本, 7.2s)
             -> jsonl 压缩 21.0 MB -> 1.4 MB
             -> 总耗时约 65 秒，产出 highlight.mp4 27.1 MB
```

## 界面与动效

### 配色（参考"渐变 + 玻璃卡片"风格）

取自参考示例的设计语言：**深海军蓝渐变底 + 玻璃拟态卡片 + 蓝紫渐变主色 + 青色霓虹点缀**。

| 用途 | 值 | 来源 |
|---|---|---|
| 页面背景 | `linear-gradient(160deg, #141e30 → #1a2942 → #243b55)` + 三处极淡光斑 | 参考里的深色渐变示例 |
| 卡片 | `rgba(26,39,62,.72)` + `backdrop-filter: blur(12px)` + 1px 白 10% 描边 | 玻璃拟态 |
| 卡片圆角 / 投影 | `16px` / `0 12px 34px rgba(6,12,24,.38)` | 参考的 15~20px 圆角与柔和投影 |
| 主按钮 | `linear-gradient(135deg, #7287fe → #8258ba)` + 蓝紫外发光 | 参考的蓝紫按钮渐变 |
| 强调色 | `#45f3ff`（青） | 参考里的青色霓虹，用于进度条与聚焦环 |
| 正文 / 次要 | `#eef3ff` / `#93a4c0` | 保证深色底上的可读性 |

细节：进度条是"青→蓝紫"渐变并带发光；输入框聚焦时青色描边 + 3px 光晕；
自定义滚动条；卡片悬停时描边和投影同时收紧。three.js 场景的雾、地面、网格也调成海军蓝，
与页面渐变衔接。

### 顶部 three.js 场景

`static/pingpong-scene.js` 用程序化几何体拼出两只对打的小动物（熊猫 / 小兔 / 企鹅 / 柴犬），
点"换一组对手"会缩放切换。不依赖任何外部模型文件。

动画里复刻了产品高光视频的三个核心特效：

| 特效 | 实现 |
|---|---|
| **速度残影轨迹** | 22 个点的位置历史，按速度着色（青蓝→黄→橙红），尺寸与不透明度沿轨迹衰减 |
| **落点标记** | 球落台时在桌面生成环形光晕 + 中心点，0.75 秒内扩散淡出（对象池复用 6 个） |
| **球速数值** | 跟随球浮动的 Sprite 标签，Canvas 每 0.12 秒重绘一次（节流，避免每帧上传纹理） |

背景元素：上浮尘埃粒子（240 点）、地面网格、两盏游走的彩色补光（蓝/紫）、
接地阴影（半透明圆片，不用 shadow map）。球体带自发光 + 加色混合光晕。

**GPU 占用是刻意控制的**（推理走 DirectML，同一块 GPU）：

| 场景 | 帧率上限 | 触发时机 |
|---|---|---|
| `full` | 45 fps | 空闲时 |
| `calm` | 15 fps | 任务运行中（提交后自动切换，结束后恢复） |
| `pause` | 停帧 | 标签页隐藏 / `prefers-reduced-motion` |

其他动效：面板淡入上浮、展开分组的内容渐显、进度条流光、运行中状态标签呼吸、
日志逐行淡入、按钮按压缩放。全部遵循 `prefers-reduced-motion`。

> 调试：页面挂了 `window.__pp`（scene/camera/ball/label/marks），排查渲染问题时可查。
> 标签页隐藏时渲染循环会停（省 GPU），此时点"换对手"会直接切换并渲染一帧。

## 打包 exe

```powershell
cd webui

# 完整包：内置 ffmpeg/ffprobe，目标机无需安装任何东西（约 736 MB）
.\packaging\build.ps1 -WithFFmpeg

# 轻量包：不含 ffmpeg，要求目标机已装 ffmpeg（约 303 MB）
.\packaging\build.ps1

# 强制全量重建
.\packaging\build.ps1 -WithFFmpeg -Clean
```

产物：`webui/dist/TTComet/`（onedir，整个文件夹一起分发），
内含 `TTComet.exe`、`使用说明.txt`、`_internal/`（运行时、模型、静态页、ffmpeg）。

打包相关说明：

| 项目 | 处理方式 |
|---|---|
| 用 onedir 而非 onefile | onefile 每次启动要把 ~300MB 解压到临时目录，冷启动几十秒 |
| 资源定位 | `paths.bundle_root()`：打包后取 `sys._MEIPASS`，源码运行取项目根 |
| 高光脚本 | `src/pingpong_highlight.py` 作为 data 打包，由 `highlight_bridge` 动态加载 |
| 后处理脚本 | `src/remove_watermark.py`、`src/remove_narration.py` 作为 data 打包，由 `postprocess_bridge` 动态加载 |
| 去人声模型 | `webui/models/demucs-onnx-cache/`（HT-Demucs fp16weights，约 158 MB）随包分发，离线可用 |
| 静态页/模型 | `webui/app/static`、`webui/models` 一并打进 `_internal/` |
| ffmpeg | 内置时放到 `_internal/ffbin/`，启动时前置到 PATH |
| 控制台编码 | 打包后默认 GBK，启动即把 stdout 切 UTF-8（否则日志 emoji 会抛异常） |
| 单实例 | Windows 命名互斥量 `TTComet_SingleInstance`，重复双击会提示"已在运行" |
| 工作目录 | `%LOCALAPPDATA%\TTComet\jobs`（不放安装目录，避免无写权限） |
| 默认浏览起点 | 打包后回落到用户"视频"目录（不是 `_internal`） |

#### 去人声依赖与模型准备（构建机）

```powershell
# 1. 安装 ONNX 去人声后端（约 50 MB，不需要 torch）
webui\.venv\Scripts\python.exe -m pip install demucs-onnx==0.3.4

# 2. 装 demucs-onnx 会把 onnxruntime-directml 覆盖成普通 onnxruntime，
#    必须恢复，否则 WebUI 推理掉到 CPU
webui\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
webui\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4

# 3. 下载 HT-Demucs fp16 权重到随包目录（约 158 MB，构建机需要本地存在；已 gitignore）
$env:HF_ENDPOINT="https://hf-mirror.com"   # 国内可加速
webui\.venv\Scripts\python.exe -c "from demucs_onnx._hub import download_single_model; download_single_model('htdemucs', precision='fp16weights', cache_dir=r'webui/models/demucs-onnx-cache')"
```

验证：`python -c "import onnxruntime as ort; print(ort.get_available_providers())"` 必须包含
`DmlExecutionProvider`。

### 打包实测（本机）

| 项目 | 结果 |
|---|---|
| 构建耗时 | 约 16 秒（不含 ffmpeg 复制） |
| 完整包体积 | 736.4 MB / 160 个文件 |
| 其中 ffmpeg+ffprobe | 433.6 MB（WinGet 的 full build，静态链接） |
| 冷启动到可访问 | 约 6-8 秒 |
| 端到端任务 | 703 帧片段 → 推理 63 fps → 高光 307 帧 → 9.2 MB，全流程通过 |
| 视频拖动 | `Range` 请求返回 206 |

### 发版流程

```powershell
# 1. 改三处版本号：webui/app/server.py(APP_VERSION) / webui/packaging/app.iss / README.md
# 2. 重写 webui/packaging/RELEASE_NOTES.md（顶部版本号、变更内容、新 SHA256）
# 3. 构建
cd webui; .\packaging\build.ps1 -WithFFmpeg -Installer

# 4. 打标签并推两端
git tag -a v0.x.y -m "乒乓彗星 TTComet v0.x.y"; git push origin refs/tags/v0.x.y

# 5. 两端发 Release
#    GitHub：整包 243 MB 直接上传
#    Gitee ：单文件上限 100 MB，切 3 卷上传，说明里保留 copy /b 合并命令
```

**版本策略（2026-11 起）**：**保留全部历史 Release，不再删除旧版本**。
用户升级时可能想回退，历史包也便于对照问题（v0.1.0 及更早的版本在发版时曾被清空过）。

## 已知缺口

1. **取消只对推理阶段生效**。高光渲染/后处理都是同步调用源脚本，只能等当前步骤结束。
2. **处理记录的日志不落盘**。任务快照存在 `job.json`，但日志正文只在内存，
   重启后历史记录能点开看结果，日志区是空的。
3. **无鉴权**。服务只监听 127.0.0.1，单机单用户场景可接受；若要局域网共享必须加认证
   和路径白名单。
4. **预处理 19.5ms 偏慢**（占整帧约 1/3）。`cv2.dnn.blobFromImage` + `astype(float16)`
   有两次大块拷贝，可考虑导出 fp32 输入模型或零拷贝填 tensor。
5. **jsonl 内存问题未解**。已默认 gzip（体积 1/20）并在任务结束后压缩归档，但
   `pingpong_highlight.load_detections()` 仍是全量读入（491MB → 约 4GB）。长视频要改流式。
6. **ONNX 去人声目前走 CPU**。本机 DirectML 跑 HT-Demucs 会报 `DmlFusedNode` 错误，
   脚本已自动回退 CPU（约 2 倍实时，10 分钟成片约 5 分钟）。
