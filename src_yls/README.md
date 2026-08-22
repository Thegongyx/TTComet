# YOLO ML Backend：Label Studio 自动标注后端

基于 Ultralytics YOLO 模型的 Label Studio 自动标注后端。输入乒乓球比赛视频帧，自动识别并标注 6 类目标，直接为 Label Studio 提供预标注结果。**无 Docker，Windows 原生运行。**

## 目录

- [概述](#概述)
- [环境要求](#环境要求)
- [快速安装](#快速安装)
- [启动服务](#启动服务)
- [在 Label Studio 中连接](#在-label-studio-中连接)
- [Label Studio 标注配置](#label-studio-标注配置)
- [环境变量配置](#环境变量配置)
- [缓存清理机制](#缓存清理机制)
- [文件结构说明](#文件结构说明)
- [故障排查](#故障排查)

---

## 概述

本项目是一个 Label Studio ML Backend，封装了 YOLO 模型（`m960n.pt`），为 Label Studio 提供实时的自动标注能力。当你打开一张图片或一帧视频画面时，后端自动运行 YOLO 推理，将检测到的目标以边界框或分割掩膜的形式返回给 Label Studio 前端。

**核心能力：**

- **边界框标注（RectangleLabels）**：检测目标并用矩形框标注，支持普通框和旋转框（OBB）
- **分割标注（PolygonLabels）**：检测目标并用多边形轮廓标注
- **中文类别名自动映射**：Label Studio 中的标签可以通过 `alias` 属性自动对接到模型的类别名，无需手动配置 `predicted_values`
- **缓存自动清理**：启动时清空 LS 缓存，提交标注时清理单任务缓存，确保模型始终使用最新数据

**6 个检测类别：**

| 类别 | 说明 |
|------|------|
| 乒乓球 | 比赛用乒乓球 |
| 其他比赛球 | 非乒乓球的其他球类 |
| 球员 | 乒乓球运动员 |
| 球拍 | 乒乓球拍 |
| 球桌 | 乒乓球桌 |
| 球网 | 乒乓球网 |

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 |
| Python | 3.10 或更高版本 |
| Conda 环境 | `yolo26`（预装 PyTorch + ultralytics） |
| Label Studio | 运行在 Windows 原生（conda 环境 `label-studio`），地址 `http://localhost:8080` |
| GPU（推荐） | NVIDIA CUDA 或 AMD ROCm，ultralytics 自动检测 |
| 磁盘空间 | 模型文件约 50 MB，标注缓存按需增长 |

**为什么不用 Docker：**

后端直接运行在 Windows 的 conda 环境中，与主项目的 YOLO 推理环境保持一致。避免 Docker 带来的网络映射、GPU 透传、文件路径等额外复杂度。

**Label Studio 本体（Windows 原生）：**

Label Studio 标注服务也已从 WSL2 迁移到 Windows 原生运行（conda 环境 `label-studio`，版本 1.23.0）：

```powershell
conda run -n label-studio label-studio start --port 8080
```

- 数据目录：`%LOCALAPPDATA%\label-studio\label-studio\`（sqlite 数据库 + media 上传图片 + `.env`）
- `.env` 中的 `SECRET_KEY` 必须与迁移前一致，否则用户密码 / API token 无法解密
- 迁移来源：WSL2 `~/.local/share/label-studio/`（原数据保留作为备份）
- 与 ML 后端同机运行，图片下载走本地回环（loopback），不再跨 WSL2 网络栈，也无需再向 WSL2 复制图片

---

## 快速安装

双击 `install.bat` 即可一键安装：

```
install.bat
```

安装脚本会执行以下操作：

1. 激活 conda 环境 `yolo26`
2. 使用清华 PyPI 镜像（`https://pypi.tuna.tsinghua.edu.cn/simple`）安装 `requirements.txt` 中的依赖
3. 依赖包括：`label-studio-ml`、`label-studio-sdk`、`ultralytics`、`flask-cors` 等

安装完成后控制台会显示 `Installation complete`。

如果网络速度慢，也可以手动运行：

```powershell
conda run -n yolo26 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

---

## 启动服务

### 方式一：直接运行（最推荐）

激活 conda 环境后直接运行，日志最清晰：

```powershell
conda activate yolo26
python _wsgi.py --port 9090 --host 0.0.0.0
```

或从项目根目录：

```powershell
conda run -n yolo26 python src_yls/_wsgi.py --port 9090 --host 0.0.0.0
```

`_wsgi.py` 支持的完整参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p` / `--port` | 监听端口 | 9090 |
| `--host` | 监听地址 | 0.0.0.0 |
| `-d` / `--debug` | Flask 调试模式 | off |
| `--log-level` | 日志级别 | INFO |
| `--model-dir` | 模型/作业存储目录 | 脚本所在目录 |
| `--check` | 启动前验证模型加载 | — |
| `--basic-auth-user` | 基础认证用户名 | — |
| `--basic-auth-pass` | 基础认证密码 | — |

### 方式二：start.py 一键启动

封装了配置加载、缓存清理、模型检查，适合快速启动：

```powershell
conda run -n yolo26 python start.py
```

可选参数：

| 参数 | 说明 |
|------|------|
| `--port 9091` | 自定义端口（默认 9090） |
| `--host 0.0.0.0` | 监听地址（默认 0.0.0.0） |
| `--no-cache` | 跳过缓存清理 |
| `--check-only` | 只做模型检查，不启动服务器 |
| `--debug` | Flask 调试模式 |
| `--log-level DEBUG` | 日志级别（默认 INFO） |

### 方式三：start.bat 双击启动

双击 `start.bat` 一键启动（Windows 批处理）。

---

无论哪种方式，启动过程均包括：

**1. 加载配置** — 从 `config.example.json` 读取基础配置，再用 `config.json` 覆盖（token 等敏感信息）。

**2. 清理缓存** — 删除 `%LOCALAPPDATA%\label-studio\Cache` 目录下的旧图片缓存。

**3. 模型检查** — 加载 `models/m960n.pt`，验证 YOLO 模型可用。检查失败则不会启动服务。

**4. 启动 Flask 服务器** — 在 `http://0.0.0.0:9090` 监听，等待 Label Studio 连接。

启动成功后控制台会显示模型加载信息、类别列表，以及后续的请求日志。

---

## 在 Label Studio 中连接

### 前提：获取 Access Token

1. 打开 Label Studio（`http://localhost:8080`）
2. 进入 **Account & Settings** → **Access Token**
3. 创建一个新 Token，复制它
4. 写入 `src_yls/config.json`：

```json
{
    "LABEL_STUDIO_URL": "http://localhost:8080",
    "LABEL_STUDIO_API_KEY": "你的Token"
}
```

> 本项目已配置 `c853009a78af44de56d1172afe945c947ebcd8e9`，过期后请按上述步骤重新生成。

### 添加 ML 后端

1. 进入某个项目的 **Settings** → **Model**
2. 点击 **Add Model**
3. 填写：

| 字段 | 值 |
|------|-----|
| Title | `YOLO Backend`（可自定义） |
| URL | `http://localhost:9090` |

4. 勾选 **Use for interactive pre-annotations**（打开任务时自动标注）
5. 点击 **Validate and Save**

连接成功后，打开任意标注任务，YOLO 预标注结果自动显示。

---

## Label Studio 标注配置

在 Label Studio 的 **Labeling Interface**（标注配置）中，你需要定义与模型对应的标签。后端提供了两种映射方式。

### Option A（推荐）：使用 `predicted_values`

通过 `predicted_values` 属性显式指定每个标签对应的模型类别名：

```xml
<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <Label value="乒乓球" predicted_values="乒乓球" background="#FF0000"/>
    <Label value="其他球" predicted_values="其他比赛球" background="#FF8800"/>
    <Label value="球员" predicted_values="球员" background="#00FF00"/>
    <Label value="球拍" predicted_values="球拍" background="#0000FF"/>
    <Label value="球桌" predicted_values="球桌" background="#FFFF00"/>
    <Label value="球网" predicted_values="球网" background="#FF00FF"/>
  </RectangleLabels>
</View>
```

这种方式最直观，每个 LS 标签明确关联到模型的哪个类别。

### Option B：使用 `alias`

如果觉得 `predicted_values` 写起来太长，可以用 `alias` 属性。后端会自动把 `alias` 的值与模型的类别名做匹配：

```xml
<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <Label value="乒乓球" alias="乒乓球" background="#FF0000"/>
    <Label value="比赛球" alias="其他比赛球" background="#FF8800"/>
    <Label value="人" alias="球员" background="#00FF00"/>
    <Label value="拍子" alias="球拍" background="#0000FF"/>
    <Label value="球台" alias="球桌" background="#FFFF00"/>
    <Label value="网" alias="球网" background="#FF00FF"/>
  </RectangleLabels>
</View>
```

当 `predicted_values` 未设置且默认标签映射无法匹配时，后端会遍历每个 `Label` 标签的 `alias` 属性，将其与模型类别名列表逐一比对。匹配成功则建立映射关系。

这意味着你在 Label Studio 界面上看到的标签名可以和模型内部的类别名不同，只要 `alias` 指向正确的模型类别名即可。

### 分割标注配置

如果使用多边形分割标注，将 `RectangleLabels` 替换为 `PolygonLabels`：

```xml
<View>
  <Image name="image" value="$image"/>

  <PolygonLabels name="label" toName="image">
    <Label value="乒乓球" predicted_values="乒乓球" background="#FF0000"/>
    <Label value="球员" predicted_values="球员" background="#00FF00"/>
    <Label value="球拍" predicted_values="球拍" background="#0000FF"/>
    <Label value="球桌" predicted_values="球桌" background="#FFFF00"/>
    <Label value="球网" predicted_values="球网" background="#FF00FF"/>
  </PolygonLabels>
</View>
```

### 同时使用边界框和分割

可以在同一个标注配置中同时定义 `RectangleLabels` 和 `PolygonLabels`，后端会自动匹配对应的控制模型：

```xml
<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="bbox" toName="image">
    <Label value="乒乓球" predicted_values="乒乓球" background="#FF0000"/>
    <Label value="球员" predicted_values="球员" background="#00FF00"/>
    <Label value="球拍" predicted_values="球拍" background="#0000FF"/>
    <Label value="球桌" predicted_values="球桌" background="#FFFF00"/>
    <Label value="球网" predicted_values="球网" background="#FF00FF"/>
  </RectangleLabels>

  <PolygonLabels name="seg" toName="image">
    <Label value="乒乓球" predicted_values="乒乓球" background="#FF0000"/>
    <Label value="球桌" predicted_values="球桌" background="#FFFF00"/>
    <Label value="球网" predicted_values="球网" background="#FF00FF"/>
  </PolygonLabels>
</View>
```

### 高级属性

在后端的 `Label` 或控制标签上可以设置以下额外属性：

| 属性 | 说明 | 示例 |
|------|------|------|
| `model_score_threshold` | 每个检测框的最低置信度，低于此值的预测会被过滤 | `<RectangleLabels model_score_threshold="0.6">` |
| `model_skip` | 跳过此控制标签，不生成预测 | `<RectangleLabels model_skip="true">` |
| `model_obb` | 使用旋转边界框（Oriented Bounding Box） | `<RectangleLabels model_obb="true">` |

---

## 环境变量配置

后端通过环境变量与 Label Studio 通信。配置分两个层次：

### config.example.json（模板，提交到版本库）

```json
{
    "LABEL_STUDIO_URL": "http://localhost:8080",
    "LABEL_STUDIO_API_KEY": "REPLACE_WITH_YOUR_LS_ACCESS_TOKEN"
}
```

此文件包含默认值，提交到 Git。`API_KEY` 设置为占位符，防止误用。

### config.json（用户本地配置，不提交到版本库）

```json
{
    "LABEL_STUDIO_URL": "http://localhost:8080",
    "LABEL_STUDIO_API_KEY": "你的真实 Access Token"
}
```

此文件已被 `.gitignore` 排除，用于存放你本机的真实 Token。没有此文件时，`_wsgi.py` 也能正常工作（返回空配置）。

### 变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LABEL_STUDIO_URL` | Label Studio 的访问地址 | `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | 用于后端访问 LS 资源的认证 Token | 无默认值 |
| `MODEL_SCORE_THRESHOLD` | 全局置信度阈值 | `0.5` |
| `MODEL_ROOT` | 模型文件搜索路径 | `src_yls/models/` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `DEBUG_PLOT` | 是否显示调试图像（仅交互环境） | `false` |
| `ALLOW_CUSTOM_MODEL_PATH` | 是否允许在标注配置中指定自定义模型路径 | `true` |

### 加载顺序

`start.bat` 按以下顺序加载变量：

1. 从 `config.example.json` 读取，设置初始值
2. 如果 `config.json` 存在，读取并覆盖同名变量
3. 最终值传递给 Flask 进程的环境变量

### 在标注配置中覆盖阈值

除了环境变量，你还可以在每个 `Label` 或控制标签上单独设置 `model_score_threshold`，粒度更高。

---

## 缓存清理机制

### 启动时全量清理

每次通过 `start.bat` 启动后端时，脚本自动删除 `%LOCALAPPDATA%\label-studio\Cache` 下的全部缓存文件。这样可以确保：

- 模型不会使用旧的、过时的图片缓存
- Label Studio 更新了图片源时，后端获取到的是最新版本
- 多人协作时避免缓存冲突

### 提交标注时单任务清理

当你提交标注时，后端的 `fit()` 方法会被 Label Studio 调用。此时会：

1. 从标注数据中提取图片 URL
2. 计算 URL 的 MD5 哈希，拼接出缓存文件名
3. 如果缓存文件存在，删除它

这样，下次模型需要读取同一张图片时，会从 Label Studio 重新下载，而不是使用过期的缓存。

### 清理范围说明

两种清理方式各司其职：

| 时机 | 清理范围 | 目的 |
|------|----------|------|
| 服务启动 | 全部缓存 | 确保全新开始，无残留 |
| 标注提交 | 单张图片 | 确保重新训练或推理时使用最新图片 |

---

## 文件结构说明

```
src_yls/
├── __init__.py                  # 空文件，包标记
├── _wsgi.py                     # WSGI 启动入口，解析命令行参数，启动 Flask 服务
├── model.py                     # 核心 YOLO ML Backend 类，继承 LabelStudioMLBase
│                                #   - predict(): 运行推理，返回预标注结果
│                                #   - fit(): 提交标注时触发，清理单任务缓存
│                                #   - build_label_map(): 构建标签映射，支持 alias
│                                #   - detect_control_models(): 自动检测标注配置中的控制标签
├── cli.py                       # 命令行客户端，通过 SDK 向项目批量推送预测结果
├── config.json                  # 本地配置（不提交到 Git），覆盖 config.example.json
├── config.example.json          # 配置模板，包含 LABEL_STUDIO_URL 和 LABEL_STUDIO_API_KEY
├── install.bat                  # 一键安装脚本，conda 环境 + 清华镜像安装依赖
├── start.bat                    # 一键启动脚本，加载配置 → 清理缓存 → 检查模型 → 启动服务
├── requirements.txt             # Python 依赖列表
│
├── control_models/              # 控制模型层，每种标注类型对应一个模型类
│   ├── __init__.py
│   ├── base.py                  # ControlModel 基类：缓存、路径解析、标签映射、工厂方法
│   ├── rectangle_labels.py      # RectangleLabels 模型：边界框标注（普通框）
│   └── polygon_labels.py        # PolygonLabels 模型：多边形分割标注
│
├── models/                      # YOLO 模型权重文件
│   ├── .gitkeep                 # 保持目录结构
│   └── m960n.pt                 # YOLO 模型文件（6 类检测 + 分割）
│
└── utils/                       # 工具函数
    ├── __init__.py
    └── converter.py             # 标签映射转换（timelinelabels ↔ numpy 概率数组）
```

### 关键文件职责速查

| 文件 | 你要用它做什么 |
|------|---------------|
| `start.bat` | 日常启动后端 |
| `install.bat` | 首次部署或升级依赖 |
| `config.json` | 填写你的 LS Access Token |
| `config.example.json` | 查看配置格式，一般不改 |
| `model.py` | 了解推理逻辑和缓存清理机制 |
| `_wsgi.py` | 了解命令行参数（端口、调试模式等） |
| `control_models/*.py` | 了解每种标注类型的具体实现 |
| `requirements.txt` | 查看依赖版本 |

---

## 故障排查

### 后端无法启动

**现象**：双击 `start.bat` 后窗口一闪而过，或显示错误信息。

**排查步骤**：

1. 是否已安装依赖？运行 `install.bat` 或手动执行：
   ```powershell
   conda run -n yolo26 pip install -r requirements.txt
   ```

2. conda 环境 `yolo26` 是否存在？
   ```powershell
   conda env list
   ```

3. 模型文件是否存在？检查 `models/m960n.pt`。

4. 端口 9090 是否被占用？
   ```powershell
   netstat -ano | findstr :9090
   ```
   如果被占用，用 `--port` 换端口。

5. 单独运行模型检查：
   ```powershell
   conda run -n yolo26 python _wsgi.py --check
   ```

6. 如果报 `AttributeError: 'YOLO' object has no attribute 'xxx'`，可能是 Python 字节码缓存（`__pycache__`）污染。删除项目下所有 `__pycache__` 目录后重启。


### Label Studio 连接失败

**现象**：在 LS Model 页面点击 Validate 后提示连接失败。

**排查步骤**：

1. 确认后端已启动：访问 `http://localhost:9090/health` 应返回 `{"status":"UP"}`。

2. 确认 LS 在 `http://localhost:8080` 正常运行。

3. LS 与 ML 后端均运行在 Windows 上，通过 `localhost:8080` 访问 LS，纯本地回环，无跨系统网络开销。

4. 确认 `config.json` 中的 `LABEL_STUDIO_API_KEY` 是有效的 LS Access Token（不是 JWT，是 LS 后台生成的 API Token，格式如 `c853009a...`）。


### 标注没有预标注结果

**现象**：打开标注任务后，画面干净，没有 YOLO 的预标注。

**排查步骤**：

1. 确认后端已连接成功（Model 页面显示绿色状态）。

2. 确认模型添加时勾选了 **Use for interactive pre-annotations**。

3. 检查标签名是否与模型类别（乒乓球、其他比赛球、球员、球拍、球桌、球网）匹配。支持两种方式：
   - `<Label value="乒乓球">` — 标签名直接匹配
   - `<Label value="pingpangball" alias="乒乓球">` — alias 匹配

4. 后端日志中若有 `No label map built` 警告，说明标签没有匹配上。检查标注配置中的标签名或 alias。

5. 降低置信度阈值测试：
   ```xml
   <RectangleLabels model_score_threshold="0.1">
   ```


### 模型推理速度慢

**现象**：打开标注任务后，需要等很久才显示预标注。

**可能原因**：

- **首次加载较慢**：模型第一次加载需要初始化，后续推理快很多。这是正常现象。
- **GPU 未启用**：检查 GPU 可用性：
  ```powershell
  conda run -n yolo26 python -c "import torch; print(torch.cuda.is_available())"
  ```
  返回 `False` 说明使用 CPU 推理，速度较慢。
- **大图推理慢**：首帧可能较慢，后续帧会利用缓存加速。


### 标注提交后结果没有变化

**现象**：提交标注后，下次推理仍使用旧数据。

**排查**：检查后端日志是否有 `Removed cached image:`。如果没有，`fit()` 的缓存清理未触发。确认 `config.json` 中的 token 正确。


### SDK 版本兼容性

本项目基于 `label-studio-ml==1.0.9` + `label-studio-sdk>=1.0`。新版本 SDK API 与旧模板有差异，以下改动已内置：

- `super().build_label_map()` → 改用 `LabelInterface` 新 API
- `get_local_path(task_id=)` → 改为 `project_dir=`
- `ModelResponse` 返回值 → 改为直接返回 `dict` 列表
- `label_interface` → 在 `YOLO.__init__` 中从 `label_config` 自动创建
- `project_id` → 添加 `@property` 和类属性兜底

如果你升级 `label-studio-ml`，请注意这些 API 变化。

---

*本项目是 YOLO 视频帧提取与模型微调工具集的一部分。更多信息请参见项目根目录的 `README.md`。*
