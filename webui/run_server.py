#!/usr/bin/env python3
r"""
启动器：本地起服务 + 托盘图标 + 打开浏览器
============================================
- 只监听 127.0.0.1，不对外网开放
- 打包版无控制台窗口：日志写文件，启动失败弹原生提示框
- 托盘图标（任务栏右下角）：左键/双击打开界面，右键可退出程序

用法:
    webui\.venv\Scripts\python.exe webui\run_server.py                   # 数据目录 %LOCALAPPDATA%\TTComet
    set PPT_DEV=1 && webui\.venv\Scripts\python.exe webui\run_server.py  # 数据目录 webui/workspace
环境变量:
    PPT_PORT      固定端口（默认 8760，被占用时自动换随机端口）
    PPT_WORKSPACE 自定义数据目录
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
import webbrowser

# ---------------------------------------------------------------------------
# 1. 标准输出：打包成窗口程序后 sys.stdout 是 None，任何 print 都会崩。
#    必须在 import 其它模块之前把 stdout/stderr 接到日志文件。
# ---------------------------------------------------------------------------
APP_NAME = "TTComet"
DEFAULT_PORT = 8760


def _data_root() -> str:
    p = os.environ.get("PPT_WORKSPACE")
    if p:
        return os.path.dirname(os.path.abspath(p))
    local = os.environ.get("LOCALAPPDATA")
    if local and not os.environ.get("PPT_DEV"):
        return os.path.join(local, APP_NAME)
    # 开发模式：webui/
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))


DATA_ROOT = _data_root()
LOG_DIR = os.path.join(DATA_ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "app.log")


def _rotate_logs(keep: int = 5) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 2 * 1024 * 1024:
            for i in range(keep - 1, 0, -1):
                src = f"{LOG_PATH}.{i}"
                if os.path.exists(src):
                    os.replace(src, f"{LOG_PATH}.{i + 1}")
            os.replace(LOG_PATH, f"{LOG_PATH}.1")
    except Exception:
        pass


def _setup_streams() -> None:
    """把 stdout/stderr 指到日志文件（仅当没有控制台时）。"""
    _rotate_logs()
    if sys.stdout is not None and sys.stderr is not None:
        # 有控制台（源码运行）：正常输出，顺带写一份日志
        for st in (sys.stdout, sys.stderr):
            try:
                st.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        return

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        f = open(LOG_PATH, "a", encoding="utf-8", buffering=1, errors="replace")
    except Exception:
        f = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 =====", flush=True)


_setup_streams()

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, APP_DIR)

_MUTEX = None
_TRAY = None


def msgbox(text: str, title: str = "乒乓彗星 TTComet", error: bool = False) -> None:
    """原生消息框（无控制台时唯一的提示手段）。"""
    try:
        flags = 0x10 if error else 0x40          # MB_ICONERROR / MB_ICONINFORMATION
        ctypes.windll.user32.MessageBoxW(None, text, title, flags | 0x40000)
    except Exception:
        pass


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


def set_console_title(title: str) -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except Exception:
        pass


def single_instance() -> bool:
    """Windows 命名互斥量：避免重复启动抢 GPU。非 Windows 直接放行。"""
    global _MUTEX
    if os.name != "nt":
        return True
    kernel32 = ctypes.windll.kernel32
    _MUTEX = kernel32.CreateMutexW(None, False, "TTComet_SingleInstance")
    return kernel32.GetLastError() != 183        # ERROR_ALREADY_EXISTS


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _write_last_url(url: str) -> None:
    try:
        os.makedirs(DATA_ROOT, exist_ok=True)
        with open(os.path.join(DATA_ROOT, "last_url.txt"), "w", encoding="utf-8") as f:
            f.write(url + "\n")
    except Exception:
        pass


def start_tray(url: str) -> None:
    """任务栏托盘图标：左键打开界面，右键菜单可退出。失败不影响主流程。"""
    global _TRAY
    try:
        import pystray
        from PIL import Image
        from paths import resource

        icon_path = resource("webui", "app", "static", "favicon.ico")
        if not os.path.exists(icon_path):
            print(f"[tray] 找不到图标: {icon_path}")
            return
        image = Image.open(icon_path)

        def on_open(_icon=None, _item=None):
            webbrowser.open(url)

        def on_quit(_icon=None, _item=None):
            print("[tray] 用户从托盘退出")
            try:
                if _TRAY is not None:
                    _TRAY.stop()
            except Exception:
                pass
            os._exit(0)

        menu = pystray.Menu(
            pystray.MenuItem("打开界面", on_open, default=True),
            pystray.MenuItem("退出程序", on_quit),
        )
        _TRAY = pystray.Icon("TTComet", image, "乒乓彗星 TTComet", menu)
        threading.Thread(target=_TRAY.run, name="tray", daemon=True).start()
        print("[tray] 托盘图标已创建")
    except Exception as e:
        print(f"[tray] 创建失败（不影响使用）: {e}")


def main() -> int:
    try:
        return _main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        msgbox(f"启动失败：\n\n{e}\n\n日志：{LOG_PATH}", error=True)
        return 1


def _main() -> int:
    import uvicorn

    if not single_instance():
        msgbox("乒乓彗星 TTComet 已经在运行了。\n\n请使用任务栏右下角的图标，"
               "或浏览器里已经打开的页面。")
        return 1

    set_console_title("乒乓彗星 TTComet")
    ff = ensure_ffmpeg_path()

    want = (os.environ.get("PPT_PORT") or "").strip()
    want = int(want) if want else DEFAULT_PORT
    port = want if _port_available(want) else _free_port()
    url = f"http://127.0.0.1:{port}/"
    _write_last_url(url)

    print("=" * 60)
    print("  乒乓彗星 TTComet")
    print("=" * 60)
    print(f"  地址: {url}")
    print(f"  数据: {DATA_ROOT}")
    if ff:
        print(f"  ffmpeg: 内置 ({ff})")
    if not sys.stdout.isatty():
        print(f"  日志: {LOG_PATH}")
    print("  退出: 托盘图标右键 → 退出程序，或界面右上角按钮")
    print()

    start_tray(url)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    from server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
