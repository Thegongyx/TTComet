#!/usr/bin/env python3
"""
启动器：本地起服务 + 打开默认浏览器
====================================
只监听 127.0.0.1（随机空闲端口），不对外网开放。
打包成 exe 后，这个脚本就是入口。

用法:
    webui\.venv\Scripts\python.exe webui\run_server.py            # 用 %LOCALAPPDATA% 作工作目录
    set PPT_DEV=1 && webui\.venv\Scripts\python.exe webui\run_server.py   # 用 webui/workspace
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import webbrowser

# 打包成 exe 后控制台默认是 GBK，日志里的 emoji/中文会抛 UnicodeEncodeError。
# 必须在任何模块打印之前把标准输出切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, APP_DIR)

_MUTEX = None


def ensure_ffmpeg_path() -> str | None:
    """打包时若内置了 ffmpeg（<资源>/ffbin），把它加进 PATH 供 subprocess 调用。"""
    try:
        from paths import resource
        d = resource("ffbin")
    except Exception:
        return None
    if os.path.isdir(d):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        return os.path.join(d, "ffmpeg.exe")
    return None


def single_instance() -> bool:
    """Windows 命名互斥量：避免重复启动抢 GPU。非 Windows 直接放行。"""
    global _MUTEX
    if os.name != "nt":
        return True
    kernel32 = ctypes.windll.kernel32
    _MUTEX = kernel32.CreateMutexW(None, False, "PingPongStudio_SingleInstance")
    ERROR_ALREADY_EXISTS = 183
    return kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    import uvicorn

    if not single_instance():
        print("=" * 60)
        print("  PingPong Studio 已经在运行了")
        print("  请使用已打开的那个窗口/浏览器标签页。")
        print("=" * 60)
        try:
            input("按回车退出…")
        except Exception:
            pass
        return 1

    ff = ensure_ffmpeg_path()

    port = int(os.environ.get("PPT_PORT", "0")) or _free_port()
    url = f"http://127.0.0.1:{port}/"

    def open_browser():
        webbrowser.open(url)

    threading.Timer(1.2, open_browser).start()
    print("=" * 60)
    print("  PingPong Studio")
    print("=" * 60)
    print(f"  地址: {url}")
    if ff:
        print(f"  ffmpeg: 内置 ({ff})")
    print("  关闭本窗口即退出服务")
    print()

    from server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
