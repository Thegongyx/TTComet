#!/usr/bin/env python3
"""
视频探测与 VFR→CFR 转换
=======================
逻辑逐行对齐 `src/inference_video.py` 的 `is_variable_frame_rate` /
`_probe_has_audio` / `convert_vfr_to_cfr`，保证两条链路对同一素材的处理结果一致
（包括高光/对比度/饱和度那套画面调整参数）。

为什么不直接 import 原脚本：原脚本顶层会 `import torch/ultralytics`，打包后没有
这些依赖；这里只复制这 3 个纯 ffmpeg/opencv 函数，行为保持一致。
"""

from __future__ import annotations

import platform
import shutil
import subprocess

import cv2

_IS_WINDOWS = platform.system() == "Windows"
_SUBPROC_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if _IS_WINDOWS else 0


def is_variable_frame_rate(video_path: str) -> bool:
    """r_frame_rate 与 avg_frame_rate 不一致即视为 VFR（阈值 1e-5）。"""
    if not shutil.which("ffprobe"):
        return False
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=r_frame_rate,avg_frame_rate",
               "-of", "default=noprint_wrappers=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, timeout=30,
                                creationflags=_SUBPROC_FLAGS)
        r_fps = avg_fps = None
        for line in result.stdout.strip().splitlines():
            if line.startswith("r_frame_rate="):
                r_fps = line.split("=", 1)[1].strip()
            elif line.startswith("avg_frame_rate="):
                avg_fps = line.split("=", 1)[1].strip()
        if not r_fps or not avg_fps:
            return False

        def _to_f(x: str) -> float:
            try:
                if "/" in x:
                    a, b = x.split("/", 1)
                    return float(a) / float(b) if float(b) != 0 else 0.0
                return float(x)
            except Exception:
                return 0.0

        r, a = _to_f(r_fps), _to_f(avg_fps)
        if r <= 0 or a <= 0:
            return False
        return abs(r - a) / max(r, a) > 0.00001
    except Exception:
        return False


def probe_has_audio(video_path: str) -> bool:
    if not shutil.which("ffprobe"):
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=15,
            creationflags=_SUBPROC_FLAGS)
        return "audio" in (r.stdout or "")
    except Exception:
        return False


def convert_vfr_to_cfr(video_path: str, output_path: str, target_fps: float | None = None,
                       highlights_pct: float = 10.0, contrast_pct: float = 10.0,
                       saturation_pct: float = 6.0) -> str:
    """转成恒定帧率，并与原脚本一样做轻微画面调整。"""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found.")
    if target_fps is None:
        cap = cv2.VideoCapture(video_path)
        target_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if target_fps <= 0:
            target_fps = 30.0
    target_fps = round(target_fps, 3)
    has_audio = probe_has_audio(video_path)
    print(f"  Converting VFR to CFR @ {target_fps} fps "
          f"(audio: {'yes' if has_audio else 'no'}) ...")

    vf_filters = [f"fps={target_fps}"]
    eq_opts = []
    if contrast_pct != 0.0:
        eq_opts.append(f"contrast={1.0 + contrast_pct / 100.0:.4f}")
    if saturation_pct != 0.0:
        eq_opts.append(f"saturation={1.0 + saturation_pct / 100.0:.4f}")
    if eq_opts:
        vf_filters.append("eq=" + ":".join(eq_opts))
    if highlights_pct != 0.0:
        h_max = max(0.0, 1.0 - highlights_pct / 100.0)
        vf_filters.append(f"curves=master='0/0 0.5/0.5 1/{h_max:.4f}'")
    vf_chain = ",".join(vf_filters)

    if highlights_pct or contrast_pct or saturation_pct:
        print(f"  🎨 画面调整: 高光-{highlights_pct}%  对比度+{contrast_pct}%  饱和度+{saturation_pct}%")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts",
        "-i", video_path,
        "-vf", vf_chain,
        "-vsync", "cfr",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
    ]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-async", "1"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", output_path]

    subprocess.run(cmd, check=True, timeout=3600, creationflags=_SUBPROC_FLAGS)
    return output_path
