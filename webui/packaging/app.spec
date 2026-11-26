# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置（onedir）
==============================
入口：webui/run_server.py

构建：
    webui\\.venv\\Scripts\\pyinstaller.exe webui\\packaging\\app.spec --noconfirm ^
        --distpath webui\\dist --workpath webui\\build

产物：webui\dist\\TTComet\\TTComet.exe

说明：
  - 用 onedir 而不是 onefile：onefile 每次启动都要把 ~300MB 解压到临时目录，冷启动几十秒。
  - console=False：双击不弹控制台；日志写 %LOCALAPPDATA%\TTComet\logs\app.log，
    启动失败会弹原生提示框。托盘图标提供「打开界面 / 退出程序」。
  - 资源（静态页 / three.js / 模型 / 高光脚本）作为 data 随包，路径由 paths.resource() 解析。
  - ffmpeg / ffprobe 默认不内置；设置环境变量 PPT_BUNDLE_FFMPEG=1 可从 PATH 复制进包。
"""

import os
import shutil
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))   # 项目根
WEBUI = os.path.join(ROOT, "webui")
APP = os.path.join(WEBUI, "app")

# ---------------------------------------------------------------- 资源
datas = [
    (os.path.join(APP, "static"), "webui/app/static"),
    (os.path.join(WEBUI, "models"), "webui/models"),
    (os.path.join(ROOT, "src", "pingpong_highlight.py"), "src"),
    (os.path.join(ROOT, "src", "remove_watermark.py"), "src"),
    (os.path.join(ROOT, "src", "remove_narration.py"), "src"),
]

# ---------------------------------------------------------------- ffmpeg（可选内置）
if os.environ.get("PPT_BUNDLE_FFMPEG") == "1":
    found_any = False
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        found = shutil.which(exe)
        if found:
            datas.append((found, "ffbin"))     # 作为数据文件放进 <资源>/ffbin
            found_any = True
        else:
            print(f"[spec] 警告: 找不到 {exe}，将不会内置")
    if found_any:
        print("[spec] 已内置 ffmpeg/ffprobe")
else:
    print("[spec] 未内置 ffmpeg（设置 PPT_BUNDLE_FFMPEG=1 可内置）")

# ---------------------------------------------------------------- 隐式依赖
hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "pystray",
    "pystray._win32",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    # 后处理：AI 去人声（ONNX，无需 torch）
    "demucs_onnx",
    "demucs_onnx._audio",
    "demucs_onnx._hub",
    "demucs_onnx.inference",
    "demucs_onnx.providers",
    "soundfile",
    "soxr",
    "huggingface_hub",
] + collect_submodules("uvicorn")

# onnxruntime 的 DLL（含 DirectML.dll）默认收集不全，显式抓一遍
binaries = collect_dynamic_libs("onnxruntime")

# soundfile 的 libsndfile DLL 放在 _soundfile_data/ 下，显式随包
try:
    import soundfile as _sf
    _sf_data = os.path.join(os.path.dirname(_sf.__file__), "_soundfile_data")
    if os.path.isdir(_sf_data):
        datas.append((_sf_data, "soundfile/_soundfile_data"))
        print("[spec] 已收集 soundfile/_soundfile_data")
except Exception as _e:
    print(f"[spec] 警告: 未能收集 soundfile 运行库: {_e}")

a = Analysis(
    [os.path.join(WEBUI, "run_server.py")],
    pathex=[APP, ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "pandas", "scipy", "IPython",
        "pytest", "torch", "ultralytics",     # 本工程不需要 torch 栈
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TTComet",
    icon=os.path.join(WEBUI, "packaging", "TTComet.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                # 窗口子系统：双击不弹黑框（日志写文件，见 run_server.py）
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TTComet",
)
