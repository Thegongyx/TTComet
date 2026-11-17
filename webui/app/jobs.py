#!/usr/bin/env python3
"""
任务管理与编排
==============
语义对齐 `src/run_pipeline.py`：
  - 输入可以是单个视频文件，也可以是目录（目录只扫当前层，按文件名排序）
  - 目录批量时**模型只加载一次**，所有视频复用同一个 OnnxDetector
  - 每个视频走：推理 -> 高光（调用 src/pingpong_highlight.py 本体）-> 日志压缩

单 GPU，任务队列串行；一个"任务"可包含多个视频。
"""

from __future__ import annotations

import gzip
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import inference_job
from highlight_bridge import run_highlight
from onnx_engine import DetectorConfig, OnnxDetector

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
              ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".3gp"}


def _now() -> float:
    return time.time()


def find_video_files(directory: str) -> list[str]:
    """与 run_pipeline.find_video_files 一致：只扫当前层，按文件名排序。"""
    out = []
    try:
        for entry in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
            if entry.is_file(follow_symlinks=False) and \
                    Path(entry.name).suffix.lower() in VIDEO_EXTS:
                out.append(entry.path)
    except OSError:
        pass
    return out


def list_dir(path: str) -> dict:
    """给前端文件浏览器用：子目录 + 视频文件。"""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(p):
        raise NotADirectoryError(p)
    dirs, videos = [], []
    try:
        for e in sorted(os.scandir(p), key=lambda x: x.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir(follow_symlinks=False):
                    dirs.append(e.name)
                elif Path(e.name).suffix.lower() in VIDEO_EXTS:
                    videos.append({"name": e.name,
                                   "mb": round(e.stat().st_size / 1024 / 1024, 1)})
            except OSError:
                continue
    except PermissionError:
        pass
    parent = os.path.dirname(p)
    return {"path": p, "parent": parent if parent != p else None,
            "dirs": dirs, "videos": videos}


@dataclass
class VideoItem:
    index: int
    name: str
    src_path: str
    out_dir: str
    status: str = "pending"      # pending / running / done / failed / cancelled
    result: dict = field(default_factory=dict)
    error: str = ""

    def brief(self) -> dict:
        out = self.result.get("output")
        return {
            "index": self.index,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "size_mb": out["size_mb"] if out else None,
            "frames": self.result.get("inference", {}).get("frames"),
            "fps": self.result.get("inference", {}).get("fps"),
            "video_url": (f"/api/jobs/{self.result.get('job_id')}/videos/{self.index}"
                          if out else None),
        }


class Job:
    def __init__(self, videos: list[VideoItem], params: dict, source: str,
                 job_id: str | None = None):
        self.id = job_id or uuid.uuid4().hex[:12]
        self.videos = videos
        self.params = params or {}
        self.source = source              # 显示用的输入（文件或目录路径）
        self.status = "queued"
        self.stage = ""
        self.progress: dict = {}
        self.logs: deque[str] = deque(maxlen=4000)
        self.result: dict = {}
        self.error = ""
        self.created_at = _now()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.cancel_requested = False
        self.current_index = -1
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        for v in videos:
            v.result.setdefault("job_id", self.id)

    # ---- 对外 ----
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "source": self.source,
                "status": self.status,
                "stage": self.stage,
                "progress": dict(self.progress),
                "result": dict(self.result),
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "current_index": self.current_index,
                "videos": [v.brief() for v in self.videos],
                "log_tail": list(self.logs)[-40:],
            }

    def summary(self) -> dict:
        done = sum(1 for v in self.videos if v.status == "done")
        failed = sum(1 for v in self.videos if v.status == "failed")
        return {"total": len(self.videos), "done": done, "failed": failed,
                "pending": len(self.videos) - done - failed}

    # ---- 事件广播 ----
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=800)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _broadcast(self, msg: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def add_log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self.logs.append(line)
        self._broadcast({"type": "log", "data": line})

    def update_progress(self, patch: dict) -> None:
        with self._lock:
            self.progress.update(patch)
            snap = dict(self.progress)
        self._broadcast({"type": "progress", "data": snap})

    def set_status(self, status: str, stage: str | None = None) -> None:
        with self._lock:
            self.status = status
            if stage is not None:
                self.stage = stage
        self._broadcast({"type": "status", "data": {"status": status,
                                                    "stage": self.stage}})

    def video_changed(self, index: int) -> None:
        with self._lock:
            self.current_index = index
            snap = [v.brief() for v in self.videos]
        self._broadcast({"type": "videos", "data": snap})


class JobManager:
    """串行任务队列；模型按档位分别常驻，批量视频共用。"""

    def __init__(self, workspace: str, models: dict[str, str], default_model: str):
        self.workspace = os.path.abspath(workspace)
        self.models = dict(models)
        self.default_model = default_model
        os.makedirs(self.workspace, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._detectors: dict[str, OnnxDetector] = {}
        self._worker = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._worker.start()

    @property
    def model_loaded(self) -> bool:
        return bool(self._detectors)

    @property
    def loaded_models(self) -> list[str]:
        return sorted(self._detectors.keys())

    def has_active_jobs(self) -> bool:
        """是否有排队中或正在跑的任务（退程序前要拦一下）。"""
        with self._lock:
            return any(j.status in ("queued", "running") for j in self.jobs.values())

    # ---- 提交 ----
    def create(self, source: str, params: dict) -> Job:
        """source 可以是视频文件或目录（目录只扫当前层）。"""
        source = os.path.abspath(os.path.expanduser(source))
        if not os.path.exists(source):
            raise FileNotFoundError(source)

        if os.path.isdir(source):
            paths = find_video_files(source)
            if not paths:
                raise ValueError(f"目录中没有视频文件: {source}")
        else:
            if Path(source).suffix.lower() not in VIDEO_EXTS:
                raise ValueError(f"不支持的视频格式: {source}")
            paths = [source]

        job_id = uuid.uuid4().hex[:12]
        job_root = os.path.join(self.workspace, job_id)
        os.makedirs(job_root, exist_ok=True)

        items: list[VideoItem] = []
        used: dict[str, int] = {}
        for i, p in enumerate(paths):
            stem = Path(p).stem
            used[stem] = used.get(stem, 0) + 1
            safe = stem if used[stem] == 1 else f"{stem}_{used[stem]}"
            out_dir = os.path.join(job_root, f"{i:02d}_{safe}")
            items.append(VideoItem(index=i, name=Path(p).name,
                                   src_path=p, out_dir=out_dir))

        job = Job(items, params, source, job_id=job_id)
        job.add_log(f"任务创建: {'目录' if os.path.isdir(source) else '单文件'} "
                    f"-> {len(items)} 个视频")
        for v in items:
            job.add_log(f"  [{v.index + 1}/{len(items)}] {v.name}")
        with self._lock:
            self.jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self.jobs.values(), key=lambda j: -j.created_at)
        out = []
        for j in jobs:
            d = j.to_dict()
            d.pop("log_tail", None)
            d["summary"] = j.summary()
            out.append(d)
        return out

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status in ("done", "failed", "cancelled"):
            return False
        job.cancel_requested = True
        job.add_log("收到取消请求（当前视频的推理会立即停止；高光渲染需等当前帧结束）")
        return True

    def queue_position(self, job: Job) -> int:
        with self._lock:
            pending = [j for j in self.jobs.values() if j.status == "queued"]
        pending.sort(key=lambda j: j.created_at)
        try:
            return pending.index(job) + 1
        except ValueError:
            return 0

    # ---- worker ----
    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                self._run(job)
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                job.add_log("❌ 任务异常: " + job.error)
                job.add_log(traceback.format_exc())
                job.set_status("failed", "done")
                job.finished_at = _now()
            finally:
                job._broadcast({"type": "done", "data": job.to_dict()})

    def _ensure_detector(self, job: Job):
        """按任务选择的精度档位取（或懒加载）对应的 detector。"""
        mid = str(job.params.get("model") or self.default_model)
        if mid not in self.models:
            job.add_log(f"未知模型档位 {mid}，回退到 {self.default_model}")
            mid = self.default_model

        cfg = DetectorConfig(
            conf_ball=float(job.params.get("conf_ball", 0.10)),
            conf_player=_opt_float(job.params, "conf_player"),
            conf_racket=_opt_float(job.params, "conf_racket"),
            conf_table=_opt_float(job.params, "conf_table"),
            conf_net=_opt_float(job.params, "conf_net"),
            iou=float(job.params.get("iou", 0.45)),
        )
        det = self._detectors.get(mid)
        if det is None:
            job.add_log(f"加载模型[{mid}]: {os.path.basename(self.models[mid])} ...")
            det = OnnxDetector(self.models[mid], cfg)
            self._detectors[mid] = det
            job.add_log(f"模型就绪[{mid}]: 后端 {det.providers}, "
                        f"输入 {det.imgsz}, dtype {det.input_dtype.__name__}")
        else:
            # 复用已加载的会话，只替换检测参数（阈值/NMS）
            cfg.imgsz = det.imgsz
            det.cfg = cfg
            job.add_log(f"复用已加载模型[{mid}]（不重新冷启动）")
        return det, mid

    def _run(self, job: Job) -> None:
        job.started_at = _now()
        job.set_status("running", "inference")
        detector, model_id = self._ensure_detector(job)
        total = len(job.videos)

        for item in job.videos:
            if job.cancel_requested:
                item.status = "cancelled"
                continue
            item.status = "running"
            item.result.setdefault("job_id", job.id)
            job.video_changed(item.index)
            job.add_log(f"===== [{item.index + 1}/{total}] {item.name} =====")
            job.set_status("running", "inference")
            job.update_progress({"video_index": item.index, "video_total": total,
                                 "video_name": item.name, "percent": 0.0})

            try:
                inf = inference_job.infer_video(
                    item.src_path, item.out_dir, detector.cfg,
                    model=self.models[model_id], compress=False,
                    should_cancel=lambda: job.cancel_requested,
                    log=job.add_log, progress=job.update_progress,
                    detector=detector)
                inf["model"] = model_id
            except Exception as e:
                item.status = "failed"
                item.error = f"推理失败: {type(e).__name__}: {e}"
                job.add_log("❌ " + item.error)
                job.add_log(traceback.format_exc())
                job.video_changed(item.index)
                continue

            if inf.get("cancelled") or job.cancel_requested:
                item.status = "cancelled"
                job.add_log("已取消")
                job.video_changed(item.index)
                continue

            item.result["inference"] = inf
            job.set_status("running", "highlight")
            job.add_log("推理完成，开始高光剪辑")

            highlight_video = inf.get("cfr_video") or inf["video"]
            output = os.path.join(item.out_dir, "highlight.mp4")
            params = {k: v for k, v in job.params.items()
                      if k not in ("model", "conf_ball", "iou", "conf_player",
                                   "conf_racket", "conf_table", "conf_net")}
            hl = run_highlight(highlight_video, inf["jsonl"], output, params,
                               log=job.add_log)
            item.result["highlight"] = hl

            self._compress_jsonl(job, inf["jsonl"])

            if hl["ok"]:
                item.status = "done"
                item.result["output"] = {
                    "path": output,
                    "size_mb": round(os.path.getsize(output) / 1024 / 1024, 1),
                }
                job.add_log(f"✅ {item.name} 完成")
            elif hl.get("no_rally"):
                item.status = "done"
                item.result["no_rally"] = True
                job.add_log(f"ℹ️ {item.name}: 有效回合不足，未生成高光（非故障）")
            else:
                item.status = "failed"
                item.error = hl.get("error") or f"高光脚本返回码 {hl['returncode']}"
                job.add_log(f"❌ {item.name} 高光失败: {item.error}")
            job.video_changed(item.index)

        s = job.summary()
        job.result = {"summary": s,
                      "videos": [v.brief() for v in job.videos]}
        if job.cancel_requested:
            job.set_status("cancelled", "done")
        elif s["failed"] and s["done"] == 0:
            job.error = "全部视频失败"
            job.set_status("failed", "done")
        else:
            job.set_status("done", "done")
        job.add_log(f"任务结束: 成功 {s['done']} / 失败 {s['failed']} / 共 {s['total']}")
        job.finished_at = _now()

    @staticmethod
    def _compress_jsonl(job: Job, path: str) -> None:
        if not path or not os.path.exists(path) or path.endswith(".gz"):
            return
        try:
            t0 = time.perf_counter()
            with open(path, "rb") as fi, gzip.open(path + ".gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, 1 << 20)
            before = os.path.getsize(path) / 1024 / 1024
            after = os.path.getsize(path + ".gz") / 1024 / 1024
            os.remove(path)
            job.add_log(f"检测日志已压缩: {before:.0f} MB -> {after:.1f} MB "
                        f"({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            job.add_log(f"⚠️ 日志压缩失败（不影响结果）: {e}")


def _opt_float(params: dict, key: str) -> float | None:
    v = params.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
