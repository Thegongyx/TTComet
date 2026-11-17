#!/usr/bin/env python3
"""
资源路径解析（兼容源码运行与 PyInstaller 打包）
==============================================
打包后 `__file__` 指向临时解压目录里的模块，不能再用它推项目根。
统一从这里取：

    resource("webui", "models", "m960n_fp16_nonms.onnx")

- 源码运行：返回 <项目根>/webui/models/...
- PyInstaller onedir/onefile：返回 <_MEIPASS>/webui/models/...
"""

from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> str:
    """资源根目录（打包后是 exe 内置资源目录，开发时是项目根）。"""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def resource(*parts: str) -> str:
    return os.path.join(bundle_root(), *parts)
