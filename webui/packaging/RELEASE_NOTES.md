## PingPong Studio v0.0.1

首个可分发版本：**本地运行的乒乓球视频自动高光剪辑工具**。

### 主要功能

- **自动高光剪辑**：逐帧推理 → 球轨迹分析 → 回合检测 → 切片输出，自带球轨迹彗尾、落点标记、迷你球台、球员光环等特效
- **本地网页界面**：选择单个视频或整个文件夹批量处理，实时进度与日志，结果在线播放/下载
- **两档精度模型**：高精度（识别更准）/ 低精度（快约 2.8 倍），可随时切换
- **参数可调**：按业务分 8 组共 35 项，每项都标注"调高/调低"的效果；支持保存多套预设并重命名
- **不剪辑模式**：可只给视频加特效、保留完整时长（不剔除捡球与空档）
- **自动化处理**：VFR 自动转 CFR，检测日志自动压缩（约 20:1）

### 安装

- **无需管理员权限**（默认安装到当前用户目录）
- **已内置 ffmpeg**，目标机不需要安装 Python 或任何运行环境
- 安装完成后自动启动，浏览器会自动打开操作界面

### 📦 安装包下载（重要）

**Gitee 附件单个限制 100 MB**，所以安装包分成了 3 个分卷，需要下载后合并：

1. 把 `PingPongStudio-0.0.1-setup.exe.part1 / .part2 / .part3` 三个文件下载到**同一目录**
2. 在该目录打开命令行（地址栏输入 `cmd` 回车），执行：

```
copy /b PingPongStudio-0.0.1-setup.exe.part1 + PingPongStudio-0.0.1-setup.exe.part2 + PingPongStudio-0.0.1-setup.exe.part3 PingPongStudio-0.0.1-setup.exe
```

3. 得到完整的 `PingPongStudio-0.0.1-setup.exe`（239 MB），双击安装

> **更省事的方式**：直接从 GitHub 下载完整安装包（不用合并）
> https://github.com/Thegongyx/tt_ylov/releases/download/v0.0.1/PingPongStudio-0.0.1-setup.exe

### 环境要求

- Windows 10 / 11（64 位）
- 推荐带独立显卡（AMD / NVIDIA 均可，走 DirectML 加速）；无独显也能运行，但速度明显更慢

### 已知限制

- 程序**未做代码签名**，首次运行 Windows Defender / 杀软可能拦截，选"仍要运行"即可
- 同一时间只允许运行一个实例（避免抢占 GPU）
- 处理数据保存在 `%LOCALAPPDATA%\PingPongStudio\jobs\`，卸载不会删除

### 技术说明

- 推理基于 ONNX Runtime（DirectML），不依赖 PyTorch / ultralytics
- 高光剪辑逻辑与仓库中的 `src/pingpong_highlight.py` 完全同源（运行时直接加载该脚本）
