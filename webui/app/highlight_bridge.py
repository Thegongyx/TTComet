#!/usr/bin/env python3
"""
高光剪辑桥接层
==============
**不重写高光逻辑**，而是以模块方式加载 `src/pingpong_highlight.py` 并调用它的
`main()`，只负责构造 argv、把它的 stdout 转发成日志行、捕获 SystemExit。

这样业务逻辑永远与源脚本一致（参数默认值、回合判定、特效渲染全部沿用），
以后改源脚本，WebUI 自动跟着变。

用法:
    from highlight_bridge import run_highlight
    r = run_highlight(video, jsonl, output, {"highlight_top_n": 10}, log=print)
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import time
import traceback

# 允许通过环境变量指定脚本位置（打包后脚本会随程序一起放进资源目录）
_ENV_PATH = "PPH_HIGHLIGHT_SCRIPT"

_PARAM_FLAGS = {
    # 通用
    "output_height": "--output-height",
    "no_audio": "--no-audio",
    "encoder": "--encoder",
    "debug": "--debug",
    # 剪辑开关（值为 True 用 --cut，False 用 --no-cut）
    "cut": "--cut",
    # 回合判定
    "min_rally_frames": "--min-rally-frames",
    "min_rally_bounce_count": "--min-rally-bounce-count",
    "min_rally_speed_top_n": "--min-rally-speed-top-n",
    "min_static_ball_count": "--min-static-ball-count",
    "ball_conf": "--ball-conf",
    "table_edge_tol": "--table-edge-tol",
    "out_of_table_timeout": "--out-of-table-timeout",
    "rally_lost_timeout": "--rally-lost-timeout",
    "no_bounce_timeout": "--no-bounce-timeout",
    "max_jump_ratio": "--max-jump-ratio",
    "corridor_margin": "--corridor-margin",
    "corridor_extend": "--corridor-extend",
    "corridor_max_drop": "--corridor-max-drop",
    "center_dist_ratio": "--center-dist-ratio",
    "court_cluster": "--court-cluster",
    "match_algo": "--match-algo",
    "camera_mode": "--camera-mode",
    # 片段与缓冲
    "pre_buffer": "--pre-buffer",
    "post_buffer": "--post-buffer",
    "merge_gap": "--merge-gap",
    # 前置精彩回合
    "highlight_top_n": "--highlight-top-n",
    "highlight_min_duration": "--highlight-min-duration",
    "highlight_min_bounce_count": "--highlight-min-bounce-count",
    "highlight_black_gap": "--highlight-black-gap",
    "no_highlight_prepend": "--no-highlight-prepend",
    # 特效
    "fx_level": "--fx-level",
    "trail_sec": "--trail-sec",
    "no_player_fx": "--no-player-fx",
}

_BOOL_FLAGS = {"no_audio", "debug", "court_cluster", "no_highlight_prepend", "no_player_fx"}
# 这些参数本身是"布尔对"：True 走 --flag，False 走 --no-flag
_NEGATED_FLAGS = {"cut"}


def _candidate_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    # webui/app -> 项目根
    root = os.path.abspath(os.path.join(here, "..", ".."))
    cands = [os.environ.get(_ENV_PATH, "")]
    try:
        from paths import resource  # 打包后由 PyInstaller 资源目录提供
        cands.append(resource("src", "pingpong_highlight.py"))
        cands.append(resource("pingpong_highlight.py"))
    except Exception:
        pass
    cands.append(os.path.join(root, "src", "pingpong_highlight.py"))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.insert(0, os.path.join(meipass, "pingpong_highlight.py"))
        cands.insert(0, os.path.join(meipass, "src", "pingpong_highlight.py"))
    cands.insert(0, os.path.join(here, "pingpong_highlight.py"))
    return [c for c in cands if c and os.path.exists(c)]


def find_script() -> str:
    cands = _candidate_paths()
    if not cands:
        raise FileNotFoundError(
            "找不到 pingpong_highlight.py；可用环境变量 "
            f"{_ENV_PATH} 指定路径")
    return cands[0]


_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = find_script()
    spec = importlib.util.spec_from_file_location("_pp_highlight", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)      # 顶层只定义函数/常量，main 在 __main__ 保护下
    _MODULE = mod
    return mod


class _LineWriter(io.TextIOBase):
    """把脚本的 stdout（含 \\r 刷新的进度行）切成整行回调出去。"""

    def __init__(self, cb):
        super().__init__()
        self._cb = cb
        self._buf = ""

    def write(self, s):  # noqa: D102
        if not s:
            return 0
        self._buf += s
        while True:
            positions = [p for p in (self._buf.find("\n"), self._buf.find("\r")) if p >= 0]
            if not positions:
                break
            i = min(positions)
            line, self._buf = self._buf[:i], self._buf[i + 1:]
            line = line.strip()
            if line:
                self._cb(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._cb(self._buf.strip())
            self._buf = ""


def build_argv(video: str, jsonl: str, output: str, params: dict | None = None) -> list[str]:
    argv = ["pingpong_highlight.py",
            "--video", os.path.abspath(video),
            "--raw-detections", os.path.abspath(jsonl),
            "--output", os.path.abspath(output)]
    for key, val in (params or {}).items():
        flag = _PARAM_FLAGS.get(key)
        if not flag or val is None:
            continue
        if key in _NEGATED_FLAGS:
            argv.append(flag if val else flag.replace("--", "--no-", 1))
        elif key in _BOOL_FLAGS:
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    return argv


def run_highlight(video: str, jsonl: str, output: str,
                  params: dict | None = None,
                  log=None) -> dict:
    """调用源脚本完成高光剪辑。返回 {ok, returncode, output, duration_sec}。"""
    log = log or (lambda _m: None)
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    argv = build_argv(video, jsonl, output, params)
    log("高光剪辑: " + " ".join(argv))

    mod = _load_module()
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    writer = _LineWriter(log)
    t0 = time.perf_counter()
    code = 0
    err = None
    try:
        sys.argv = argv
        sys.stdout = writer  # type: ignore[assignment]
        sys.stderr = writer  # type: ignore[assignment]
        mod.main()
    except SystemExit as e:
        code = int(e.code or 0)
    except Exception as e:  # 让脚本内部的异常变成任务失败而不是进程崩溃
        code = 1
        err = f"{type(e).__name__}: {e}"
        log("❌ 高光脚本异常: " + err)
        log(traceback.format_exc())
    finally:
        try:
            writer.flush()
        finally:
            sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr

    dur = time.perf_counter() - t0
    exists = os.path.exists(output)
    ok = (code == 0) and exists
    # 源脚本的退出码语义: 0=成功, 1=参数/IO问题, 2=没有有效回合(业务结果不是故障)
    if code == 2:
        log("ℹ️ 未检测到有效回合（源脚本返回 2），这是业务结果而非故障")
    if ok:
        log(f"高光完成: {output} ({os.path.getsize(output) / 1024 / 1024:.1f} MB, "
            f"{dur:.1f}s)")
    return {
        "ok": ok,
        "returncode": code,
        "no_rally": code == 2,
        "output": output if exists else None,
        "duration_sec": round(dur, 1),
        "error": err,
    }
