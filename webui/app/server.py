#!/usr/bin/env python3
"""
本地 Web 服务（只监听 127.0.0.1）
================================
不是"服务器部署"，而是 exe 内部的界面通道：浏览器渲染页面，服务与推理/渲染跑在
同一个进程组里，不对外网开放。

输入语义对齐 src/run_pipeline.py：选一个视频文件，或选一个目录（目录下所有视频
按文件名排序批量处理，模型只冷启动一次）。

接口:
    GET  /                            页面
    GET  /api/health                  环境自检
    GET  /api/update                  检查 GitHub 最新版本（缓存）
    POST /api/update/download         下载新版安装包（后台线程）
    GET  /api/update/download         下载进度
    POST /api/update/install          启动安装程序并退出本程序
    GET  /api/fs/roots                盘符 / 常用起点
    GET  /api/fs/list?path=...        目录浏览（子目录 + 视频文件）
    POST /api/jobs                    {path, params} -> 建任务（文件或目录）
    GET  /api/jobs                    任务列表
    GET  /api/jobs/{id}               任务详情
    GET  /api/jobs/{id}/events        SSE: 进度/日志/状态/视频状态
    GET  /api/jobs/{id}/videos/{idx}  单个结果视频（支持 Range）
    POST /api/jobs/{id}/cancel        取消
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 与 run_server.py 一致：保证进程内所有日志都能写 UTF-8（打包后控制台默认 GBK）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import jobs as jobs_mod  # noqa: E402
from jobs import JobManager, list_dir  # noqa: E402
from paths import bundle_root, resource  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = bundle_root()
STATIC_DIR = resource("webui", "app", "static")


def _resolve_workspace() -> str:
    p = os.environ.get("PPT_WORKSPACE")
    if p:
        return p
    local = os.environ.get("LOCALAPPDATA")
    if local and not os.environ.get("PPT_DEV"):
        return os.path.join(local, "TTComet", "jobs")
    return os.path.join(PROJECT_ROOT, "webui", "workspace")


# 模型档位：高精度 / 低精度（导出参数相同：fp16、稠密输出、batch=1、imgsz 960）
MODEL_CATALOG = [
    ("m960", "高精度", "m960n_fp16_nonms.onnx", "识别更准，速度较慢（推荐正式成片）"),
    ("s960", "低精度", "s960n_fp16_nonms.onnx", "速度快约 2.8 倍，识别略粗（适合快速预览）"),
]


def _resolve_models() -> tuple[dict[str, str], list[dict], str]:
    models: dict[str, str] = {}
    meta: list[dict] = []
    for mid, name, filename, note in MODEL_CATALOG:
        p = resource("webui", "models", filename)
        if os.path.exists(p):
            models[mid] = p
            meta.append({
                "id": mid, "name": name, "note": note,
                "file": filename,
                "mb": round(os.path.getsize(p) / 1024 / 1024, 1),
            })
    custom = os.environ.get("PPT_MODEL")
    if custom and os.path.exists(custom):
        models["custom"] = custom
        meta.append({"id": "custom", "name": "自定义", "note": "来自 PPT_MODEL",
                     "file": os.path.basename(custom),
                     "mb": round(os.path.getsize(custom) / 1024 / 1024, 1)})
    if not models:
        raise RuntimeError("找不到任何模型文件；用环境变量 PPT_MODEL 指定 .onnx 路径")
    default_id = "custom" if "custom" in models else (
        "m960" if "m960" in models else next(iter(models)))
    return models, meta, default_id


MODELS, MODEL_META, DEFAULT_MODEL = _resolve_models()

APP_VERSION = "0.2.2"
PROJECT_URL = "https://github.com/Thegongyx/TTComet"
SUPPORT_URL = "https://ifdian.net/a/TTComet"

# ---- 在线更新（只查 GitHub Releases；Gitee 仓库不公开，不作为更新源）----
UPDATE_REPO = "TheGongyx/TTComet"
UPDATE_API = f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest"
UPDATE_USER_AGENT = f"TTComet/{APP_VERSION}"
UPDATE_OK_TTL = 6 * 3600          # 查询成功缓存 6 小时
UPDATE_FAIL_TTL = 30 * 60         # 失败也缓存 30 分钟，避免频繁打接口
UPDATE_ALLOWED_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
)


app = FastAPI(title="TTComet")
manager = JobManager(_resolve_workspace(), MODELS, DEFAULT_MODEL)
# three.js 与场景脚本走静态目录（已本地化，不依赖 CDN）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class CreateJobBody(BaseModel):
    path: str
    params: dict = {}


class PresetBody(BaseModel):
    name: str
    params: dict = {}


# ----------------------------------------------------------------------
# 参数预设（存文件，换浏览器/清缓存都不丢）
# ----------------------------------------------------------------------
PRESET_PATH = os.path.join(manager.workspace, "presets.json")
_preset_lock = threading.Lock()
MAX_PRESETS = 50
MAX_NAME_LEN = 40


def _load_presets() -> list[dict]:
    if not os.path.exists(PRESET_PATH):
        return []
    try:
        with open(PRESET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("presets", [])
        return items if isinstance(items, list) else []
    except Exception:
        return []


def _save_presets(items: list[dict]) -> None:
    tmp = PRESET_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"presets": items}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PRESET_PATH)


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "预设名称不能为空")
    if len(name) > MAX_NAME_LEN:
        raise HTTPException(400, f"名称太长了（最多 {MAX_NAME_LEN} 字）")
    return name


@app.get("/api/presets")
def list_presets():
    with _preset_lock:
        return {"presets": _load_presets()}


# ----------------------------------------------------------------------
# 退出程序
# ----------------------------------------------------------------------
@app.post("/api/shutdown")
def shutdown(x_ttcomet_request: str = Header(default="")):
    """关闭整个程序（进程退出，控制台窗口随之关闭）。

    防跨站误触：要求自定义请求头 `X-TTComet-Request: 1`。
    浏览器对跨域自定义头的请求会先发 preflight，而本服务不放行 CORS，
    所以其它网页无法偷偷调用这个接口。
    """
    if x_ttcomet_request != "1":
        raise HTTPException(403, "missing confirmation header")
    if manager.has_active_jobs():
        raise HTTPException(409, "job running")

    def _exit() -> None:
        time.sleep(0.4)     # 先让 HTTP 响应写完
        os._exit(0)         # 直接退出进程：线程/子进程随之结束

    threading.Thread(target=_exit, name="shutdown", daemon=True).start()
    return {"ok": True}


# ----------------------------------------------------------------------
# 在线更新（只查 GitHub；用户确认后才下载，安装时退出本程序）
# ----------------------------------------------------------------------
_update_lock = threading.Lock()
_update_cache: dict = {"ts": 0.0, "ok": False, "data": None}

_dl_lock = threading.Lock()
_dl_state: dict = {
    "state": "idle",      # idle / downloading / done / error
    "downloaded": 0,
    "total": 0,
    "percent": 0.0,
    "path": "",
    "error": "",
    "version": "",
    "digest": "",
}


def _ver_tuple(v: str) -> tuple[int, ...]:
    """把 "v0.2.2" / "0.2.2-beta" 解析成可比较的整数三元组。"""
    out: list[int] = []
    for part in (v or "").strip().lstrip("vV").split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            break
        out.append(int(num))
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def _pick_update_asset(rel: dict) -> dict | None:
    assets = rel.get("assets") or []
    exes = [a for a in assets if str(a.get("name", "")).lower().endswith(".exe")]
    setup = [a for a in exes if "setup" in str(a.get("name", "")).lower()]
    picked = setup or exes
    return picked[0] if picked else None


def _fetch_latest_release() -> dict:
    req = urllib.request.Request(UPDATE_API, headers={
        "User-Agent": UPDATE_USER_AGENT,
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_update(force: bool = False) -> dict:
    """查询 GitHub 最新 Release（带缓存）。失败不抛异常，仅返回 ok=False。"""
    now = time.time()
    with _update_lock:
        age = now - float(_update_cache["ts"] or 0)
        ttl = UPDATE_OK_TTL if _update_cache["ok"] else UPDATE_FAIL_TTL
        cached = _update_cache["data"]
        if not force and cached is not None and age < ttl:
            return cached

    data: dict = {
        "current": APP_VERSION,
        "latest": None,
        "has_update": False,
        "ok": False,
        "error": "",
        "url": "",
        "published_at": "",
        "asset_url": "",
        "asset_name": "",
        "asset_size": 0,
        "digest": "",
    }
    try:
        rel = _fetch_latest_release()
        latest = str(rel.get("tag_name") or "").strip().lstrip("vV")
        asset = _pick_update_asset(rel)
        data.update({
            "ok": True,
            "latest": latest or None,
            "has_update": bool(latest) and _ver_tuple(latest) > _ver_tuple(APP_VERSION),
            "url": rel.get("html_url") or "",
            "published_at": rel.get("published_at") or "",
        })
        if asset:
            data.update({
                "asset_url": asset.get("browser_download_url") or "",
                "asset_name": asset.get("name") or "",
                "asset_size": int(asset.get("size") or 0),
                "digest": str(asset.get("digest") or ""),
            })
    except urllib.error.HTTPError as e:
        data["error"] = f"GitHub HTTP {e.code}"
    except Exception as e:  # 断网/超时/DNS/证书
        data["error"] = f"{type(e).__name__}"

    with _update_lock:
        _update_cache.update({"ts": time.time(), "ok": data["ok"], "data": data})
    return data


def _download_target(name: str, latest: str) -> str:
    base = os.path.basename(name or "") or f"TTComet-{latest or 'update'}-setup.exe"
    base = "".join(ch for ch in base if ch.isalnum() or ch in "._-") or "TTComet-update.exe"
    folder = os.path.join(tempfile.gettempdir(), "TTComet-update")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, base)


def _download_worker(url: str, name: str, latest: str, digest: str) -> None:
    tmp_path = ""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in UPDATE_ALLOWED_HOSTS:
            raise ValueError(f"不允许的下载地址: {parsed.hostname or url}")
        target = _download_target(name, latest)
        tmp_path = target + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": UPDATE_USER_AGENT})
        done = 0
        with urllib.request.urlopen(req, timeout=30) as resp:
            final_host = urllib.parse.urlparse(resp.geturl()).hostname or ""
            if final_host not in UPDATE_ALLOWED_HOSTS:
                raise ValueError(f"重定向到非 GitHub 域名: {final_host}")
            total = int(resp.headers.get("Content-Length") or 0)
            with _dl_lock:
                _dl_state.update({"total": total, "downloaded": 0, "percent": 0.0,
                                  "path": "", "error": ""})
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    with _dl_lock:
                        _dl_state["downloaded"] = done
                        _dl_state["percent"] = round(done * 100.0 / total, 1) if total else 0.0

        if digest.lower().startswith("sha256:"):
            want = digest.split(":", 1)[1].strip().lower()
            got = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    got.update(block)
            if got.hexdigest() != want:
                raise ValueError("安装包校验失败（SHA256 不匹配），请重试")

        os.replace(tmp_path, target)
        with _dl_lock:
            _dl_state.update({"state": "done", "path": target, "percent": 100.0})
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        with _dl_lock:
            _dl_state.update({"state": "error", "error": f"{type(e).__name__}: {e}"})


@app.get("/api/update")
def check_update(force: int = 0):
    return _check_update(force=bool(force))


@app.post("/api/update/download")
def start_update_download(x_ttcomet_request: str = Header(default="")):
    if x_ttcomet_request != "1":
        raise HTTPException(403, "missing confirmation header")
    with _dl_lock:
        if _dl_state["state"] == "downloading":
            return dict(_dl_state)

    info = _check_update(force=True)
    if not info.get("has_update"):
        raise HTTPException(400, "当前已是最新版本，没有可下载的更新")
    if not info.get("asset_url"):
        raise HTTPException(502, "最新版本没有可下载的安装包")

    with _dl_lock:
        _dl_state.update({
            "state": "downloading", "downloaded": 0, "percent": 0.0,
            "total": int(info.get("asset_size") or 0), "path": "", "error": "",
            "version": info.get("latest") or "", "digest": info.get("digest") or "",
        })
    threading.Thread(
        target=_download_worker,
        args=(info["asset_url"], info.get("asset_name") or "",
              info.get("latest") or "", info.get("digest") or ""),
        name="update-download", daemon=True,
    ).start()
    with _dl_lock:
        return dict(_dl_state)


@app.get("/api/update/download")
def update_download_status():
    with _dl_lock:
        return dict(_dl_state)


@app.post("/api/update/install")
def install_update(x_ttcomet_request: str = Header(default="")):
    if x_ttcomet_request != "1":
        raise HTTPException(403, "missing confirmation header")
    if manager.has_active_jobs():
        raise HTTPException(409, "有任务正在运行，请等待完成或取消后再安装")
    with _dl_lock:
        state = _dl_state["state"]
        path = _dl_state["path"]
    if state != "done" or not path or not os.path.exists(path):
        raise HTTPException(400, "安装包尚未下载完成")

    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen([path, "/CLOSEAPPLICATIONS", "/NORESTART"],
                         close_fds=True, creationflags=flags)
    except Exception as e:
        raise HTTPException(500, f"无法启动安装程序: {e}")

    def _exit() -> None:
        time.sleep(1.5)   # 等 HTTP 响应写完，再释放文件句柄
        os._exit(0)

    threading.Thread(target=_exit, name="update-exit", daemon=True).start()
    return {"ok": True, "path": path}


@app.post("/api/presets")
def create_preset(body: PresetBody):
    name = _clean_name(body.name)
    if not isinstance(body.params, dict):
        raise HTTPException(400, "params 必须是对象")
    with _preset_lock:
        items = _load_presets()
        if len(items) >= MAX_PRESETS:
            raise HTTPException(400, f"预设数量已达上限（{MAX_PRESETS} 个），请先删除一些")
        if any(p.get("name") == name for p in items):
            raise HTTPException(409, "已存在同名预设")
        item = {
            "id": uuid.uuid4().hex[:10],
            "name": name,
            "params": body.params,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        items.append(item)
        _save_presets(items)
        return item


@app.put("/api/presets/{pid}")
def update_preset(pid: str, body: PresetBody):
    name = _clean_name(body.name)
    with _preset_lock:
        items = _load_presets()
        idx = next((i for i, p in enumerate(items) if p.get("id") == pid), None)
        if idx is None:
            raise HTTPException(404, "预设不存在")
        if any(p.get("name") == name and p.get("id") != pid for p in items):
            raise HTTPException(409, "已存在同名预设")
        items[idx]["name"] = name
        if isinstance(body.params, dict) and body.params:
            items[idx]["params"] = body.params
        items[idx]["updated_at"] = time.time()
        _save_presets(items)
        return items[idx]


@app.delete("/api/presets/{pid}")
def delete_preset(pid: str):
    with _preset_lock:
        items = _load_presets()
        left = [p for p in items if p.get("id") != pid]
        if len(left) == len(items):
            raise HTTPException(404, "预设不存在")
        _save_presets(left)
        return {"ok": True}



# ----------------------------------------------------------------------
# 页面 & 自检
# ----------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _default_browse_path() -> str:
    """文件浏览器的默认起点：优先项目 Videos（开发），打包后回落到用户「视频」目录。"""
    bundled_videos = resource("Videos")
    if os.path.isdir(bundled_videos):
        return bundled_videos
    home = os.path.expanduser("~")
    for name in ("Videos", "Desktop", "Downloads"):
        p = os.path.join(home, name)
        if os.path.isdir(p):
            return p
    return home if os.path.isdir(home) else PROJECT_ROOT


@app.get("/api/health")
def health():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    du = shutil.disk_usage(manager.workspace)
    default_meta = next((m for m in MODEL_META if m["id"] == DEFAULT_MODEL), None)
    return {
        "ok": True,
        "providers": providers,
        "backend": "DirectML" if "DmlExecutionProvider" in providers else "CPU",
        "ffmpeg": shutil.which("ffmpeg"),
        "model": default_meta["file"] if default_meta else "-",
        "model_mb": default_meta["mb"] if default_meta else 0,
        "model_loaded": manager.model_loaded,
        "models": MODEL_META,
        "default_model": DEFAULT_MODEL,
        "disk_free_gb": round(du.free / 1024 ** 3, 1),
        "workspace": manager.workspace,
        "default_path": _default_browse_path(),
        "version": APP_VERSION,
        "project_url": PROJECT_URL,
        "support_url": SUPPORT_URL,
    }


@app.get("/api/models")
def list_models():
    return {
        "models": MODEL_META,
        "default": DEFAULT_MODEL,
        "loaded": manager.loaded_models,
    }


# ----------------------------------------------------------------------
# 文件浏览
# ----------------------------------------------------------------------
@app.get("/api/fs/roots")
def fs_roots():
    drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    quick = []
    home = os.path.expanduser("~")
    for name, p in (("视频", os.path.join(home, "Videos")),
                    ("桌面", os.path.join(home, "Desktop")),
                    ("下载", os.path.join(home, "Downloads")),
                    ("项目 Videos", os.path.join(PROJECT_ROOT, "Videos")),
                    ("项目根目录", PROJECT_ROOT)):
        if os.path.isdir(p):
            quick.append({"name": name, "path": p})
    return {"drives": drives, "quick": quick}


@app.get("/api/fs/list")
def fs_list(path: str):
    try:
        return list_dir(path)
    except NotADirectoryError:
        raise HTTPException(400, f"不是目录: {path}")
    except Exception as e:
        raise HTTPException(400, f"无法读取: {e}")


# ----------------------------------------------------------------------
# 任务
# ----------------------------------------------------------------------
def _validate_output_dir(params: dict) -> None:
    """用户指定输出目录时做基本校验：存在或可创建、必须是目录、可写。"""
    out = (params.get("output_dir") or "").strip()
    if not out:
        return
    p = os.path.abspath(os.path.expanduser(out))
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        raise HTTPException(400, f"无法创建输出目录: {e}")
    if not os.path.isdir(p):
        raise HTTPException(400, f"输出路径不是目录: {p}")
    probe = os.path.join(p, ".ttcomet_write_test")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except Exception:
        raise HTTPException(400, f"输出目录不可写: {p}")
    params["output_dir"] = p


@app.post("/api/jobs")
def create_job(body: CreateJobBody):
    params = dict(body.params or {})
    _validate_output_dir(params)
    try:
        job = manager.create(body.path, params)
    except FileNotFoundError:
        raise HTTPException(404, f"路径不存在: {body.path}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job.id, "status": job.status,
            "videos": len(job.videos),
            "queue_position": manager.queue_position(job)}


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    d = job.to_dict()
    d["queue_position"] = manager.queue_position(job)
    d["summary"] = job.summary()
    return d


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(400, "任务无法取消")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/logs")
def get_logs(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"logs": list(job.logs)}


@app.get("/api/jobs/{job_id}/videos/{index}")
def get_video(job_id: str, index: int):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if not (0 <= index < len(job.videos)):
        raise HTTPException(404, "视频序号越界")
    out = (job.videos[index].result.get("output") or {}).get("path")
    if not out or not os.path.exists(out):
        raise HTTPException(404, "结果视频尚未生成")
    return FileResponse(out, media_type="video/mp4",
                        filename=os.path.basename(out))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在")

    async def gen():
        q = job.subscribe()
        try:
            snap = {"type": "snapshot",
                    "data": {"job": job.to_dict(),
                             "summary": job.summary(),
                             "queue_position": manager.queue_position(job)}}
            yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.4)
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("type") == "done":
                    break
        finally:
            job.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
