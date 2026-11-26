#!/usr/bin/env python3
"""
后处理桥接层（去水印 / 去人声）
==============================
和 `highlight_bridge.py` 同样的思路：**不重写业务逻辑**，以模块方式加载
`src/remove_watermark.py` / `src/remove_narration.py`，只负责构造 argv、
把 stdout 转成 WebUI 日志、捕获 SystemExit。

默认两个功能都不启用；用户在参数里勾选后：

    推理 -> 去水印（可选，在源视频上执行）-> 高光渲染 -> 去人声（可选，在成片上执行）

去水印必须放在渲染特效之前，否则会把迷你球台 / 雷达 / 连击面板等 HUD 一起抹掉。
去人声只处理音频，放在成片之后，不影响画面。

去人声的 `demucs`（torch 版）不在这里支持：打包版没有 torch，
源码版的 .venv 也没有装；需要 torch demucs 时请用命令行流水线。
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import time
import traceback

# 允许用环境变量指定脚本位置（打包后脚本会随程序一起放进资源目录）
_ENV_WATERMARK = "PPH_WATERMARK_SCRIPT"
_ENV_NARRATION = "PPH_NARRATION_SCRIPT"

# WebUI 支持的去人声方式（不带 torch 版 demucs）
NARRATION_METHODS = ("demucs-onnx", "center", "mute")


def _candidate_paths(script_name: str, env_var: str) -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    cands = [os.environ.get(env_var, "")]
    try:
        from paths import resource          # 打包后由 PyInstaller 资源目录提供
        cands.append(resource("src", script_name))
        cands.append(resource(script_name))
    except Exception:
        pass
    cands.append(os.path.join(root, "src", script_name))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.insert(0, os.path.join(meipass, script_name))
        cands.insert(0, os.path.join(meipass, "src", script_name))
    cands.insert(0, os.path.join(here, script_name))
    return [c for c in cands if c and os.path.exists(c)]


def find_script(script_name: str, env_var: str) -> str:
    cands = _candidate_paths(script_name, env_var)
    if not cands:
        raise FileNotFoundError(f"找不到 {script_name}；可用环境变量 {env_var} 指定路径")
    return cands[0]


_LOADED: dict[str, object] = {}


def _load_module(script_name: str, env_var: str):
    if script_name in _LOADED:
        return _LOADED[script_name]
    path = find_script(script_name, env_var)
    mod_name = "_pp_" + os.path.splitext(script_name)[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)        # 顶层只定义函数/常量，main 在 __main__ 保护下
    _LOADED[script_name] = mod
    return mod


class _LineWriter(io.TextIOBase):
    """把脚本 stdout（含 \\r 刷新的进度行）切成整行回调出去。"""

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


def _run_script(script_name: str, env_var: str, argv: list[str], log) -> tuple[int, str | None]:
    mod = _load_module(script_name, env_var)
    old_argv, old_stdout, old_stderr = sys.argv, sys.stdout, sys.stderr
    writer = _LineWriter(log)
    code, err = 0, None

    def emit(line: str) -> None:
        # 调用方的 log 可能会 print（例如测试时传 print）；先恢复真实 stdout，
        # 否则会写回 writer 造成无限递归。
        sys.stdout = old_stdout
        try:
            log(line)
        finally:
            sys.stdout = writer

    writer._cb = emit        # type: ignore[attr-defined]
    try:
        sys.argv = argv
        sys.stdout = writer      # type: ignore[assignment]
        sys.stderr = writer      # type: ignore[assignment]
        mod.main()
    except SystemExit as e:
        code = int(e.code or 0)
    except Exception as e:       # 把脚本内部异常变成任务失败，而不是整个进程崩溃
        code = 1
        err = f"{type(e).__name__}: {e}"
        emit("❌ 脚本异常: " + err)
        emit(traceback.format_exc())
    finally:
        try:
            writer.flush()
        finally:
            sys.argv, sys.stdout, sys.stderr = old_argv, old_stdout, old_stderr
    return code, err


def _parse_regions(text: str) -> list[str]:
    """'x,y,w,h;x,y,w,h' -> ['x,y,w,h', ...]"""
    out = []
    for part in str(text or "").replace("，", ",").split(";"):
        part = part.strip()
        if not part:
            continue
        nums = [p.strip() for p in part.split(",") if p.strip()]
        if len(nums) != 4:
            raise ValueError(f"水印区域格式应为 x,y,w,h（用 ; 分隔多个）: {part}")
        for n in nums:
            int(n)            # 提前暴露非法数字
        out.append(",".join(nums))
    return out


def _watermark_argv(video: str, output: str, params: dict) -> list[str]:
    # WebUI 只暴露两种方式: default（内置区域） / detect（自动检测）
    mode = str(params.get("watermark_mode") or "").strip()
    detect = mode == "detect" or bool(params.get("watermark_detect"))
    # 保留旧参数兼容（预设 / 自定义区域 / 算法仍可来自 CLI 或旧预设）
    argv = ["remove_watermark.py", "--video", video, "--output", output,
            "--preset", str(params.get("watermark_preset") or "overlay"),
            "--method", str(params.get("watermark_method") or "delogo")]
    for region in _parse_regions(params.get("watermark_region") or ""):
        argv += ["--region", region]
    if detect:
        argv += ["--detect", "--detect-apply"]
    return argv


def _narration_argv(video: str, output: str, params: dict,
                    cache_dir: str | None) -> list[str]:
    method = str(params.get("narration_method") or "demucs-onnx")
    if method not in NARRATION_METHODS:
        raise ValueError(f"WebUI 不支持的去人声方式: {method}"
                         f"（可选 {', '.join(NARRATION_METHODS)}；torch 版 demucs 请用命令行流水线）")
    argv = ["remove_narration.py", "--video", video, "--output", output,
            "--method", method]
    gain = params.get("narration_volume_gain")
    if gain not in (None, ""):
        argv += ["--volume-gain", str(gain)]
    if method == "demucs-onnx":
        argv += ["--onnx-precision", str(params.get("narration_onnx_precision") or "fp16weights")]
        if cache_dir:
            argv += ["--onnx-cache-dir", cache_dir]
        if params.get("narration_device"):
            argv += ["--device", str(params["narration_device"])]
    return argv


def resolve_onnx_cache_dir() -> str | None:
    """内置模型缓存目录：webui/models/demucs-onnx-cache（打包后同路径）。"""
    cands = []
    try:
        from paths import resource
        cands.append(resource("webui", "models", "demucs-onnx-cache"))
        cands.append(resource("models", "demucs-onnx-cache"))
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.abspath(os.path.join(here, "..", "models", "demucs-onnx-cache")))
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return None


def has_audio_stream(video: str) -> bool:
    """ffprobe 判断是否有音频流；探测失败时保守返回 True。"""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.returncode == 0 and "audio" in (out.stdout or "")
    except Exception:
        return True


def run_watermark(video: str, output: str, params: dict, log=None) -> dict:
    """在源视频上去水印（必须在渲染特效之前执行，避免把迷你球台等 HUD 一起抹掉）。

    输出写到临时文件，成功后替换 output；失败时 output 保持原样。
    """
    log = log or (lambda _m: None)
    result = {"ok": True, "error": None, "duration_sec": 0.0, "output": None}
    if not params.get("remove_watermark"):
        return result

    output = os.path.abspath(output)
    tmp = f"{output}.tmp.mp4"
    t0 = time.perf_counter()
    try:
        mode = str(params.get("watermark_mode") or "default")
        log(f"去水印（{mode}）: 在源视频上处理，不经过渲染特效")
        argv = _watermark_argv(video, tmp, params)
        log("去水印: " + " ".join(argv))
        code, err = _run_script("remove_watermark.py", _ENV_WATERMARK, argv, log)
        if code != 0 or not os.path.exists(tmp):
            result.update(ok=False, error=err or f"去水印失败（返回码 {code}）")
            log("❌ " + result["error"])
            return result
        os.replace(tmp, output)
        result["output"] = output
        log(f"✅ 去水印完成: {os.path.basename(output)} "
            f"({os.path.getsize(output) / 1024 / 1024:.1f} MB)")
        return result
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        result["duration_sec"] = round(time.perf_counter() - t0, 1)
        if result["ok"]:
            log(f"去水印耗时 {result['duration_sec']}s")


def run_narration(video: str, output: str, params: dict, log=None) -> dict:
    """在成片上执行去人声（音频后处理，不影响画面特效）。"""
    log = log or (lambda _m: None)
    result = {"ok": True, "error": None, "duration_sec": 0.0, "output": None}
    if not params.get("remove_narration"):
        return result
    if params.get("no_audio"):
        log("ℹ️ 已选择“去掉原声”，跳过去人声")
        return result
    if not has_audio_stream(video):
        log("ℹ️ 当前成片没有音轨，跳过去人声")
        return result

    # 打包后脚本在 _MEIPASS/安装目录里，未必可写；中间 wav 统一放系统临时目录
    import tempfile
    os.environ.setdefault("PPH_NARRATION_TMP",
                          os.path.join(tempfile.gettempdir(), "TTComet", "narration"))
    onnx_cache = resolve_onnx_cache_dir()
    if onnx_cache:
        # 模型随包分发，直接走本地缓存，避免每任务联网解析
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    output = os.path.abspath(output)
    tmp = f"{output}.narr.tmp.mp4"
    t0 = time.perf_counter()
    try:
        method = params.get("narration_method") or "demucs-onnx"
        log(f"去人声（{method}）")
        argv = _narration_argv(video, tmp, params, onnx_cache)
        log("去人声: " + " ".join(argv))
        code, err = _run_script("remove_narration.py", _ENV_NARRATION, argv, log)
        if code != 0 or not os.path.exists(tmp):
            result.update(ok=False, error=err or f"去人声失败（返回码 {code}）")
            log("❌ " + result["error"])
            return result
        os.replace(tmp, output)
        result["output"] = output
        log(f"✅ 去人声完成: {os.path.basename(output)} "
            f"({os.path.getsize(output) / 1024 / 1024:.1f} MB)")
        return result
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        result["duration_sec"] = round(time.perf_counter() - t0, 1)
        if result["ok"]:
            log(f"去人声耗时 {result['duration_sec']}s")
