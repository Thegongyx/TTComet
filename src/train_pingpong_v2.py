#!/usr/bin/env python3
"""
YOLO 分割模型微调训练脚本 v3.2（v3.1 → v3.2 增强）
==============================================
v3.2 在 v3.1 基础上的功能增强（针对乒乓球目标识别）：
  新功能 [任务2]：
    - **保留两个最优 ball 模型**，分别针对两种业界常用指标：
        ① best_class_<id>_ap.pt    — 类别 mAP (mAP50-95) 最高时的 checkpoint
                                     （目标检测准确率最优）
        ② best_class_<id>_map50.pt — 类别 mAP50 最高时的 checkpoint
                                     （宽松 IoU 下的检测能力最优）
      原来的"按一个类别 AP 选最优"被扩展为同时跟踪两个指标。
    - **训练完成后总览**：列出所有保存的模型（best.pt / last.pt / best_class_*）
      并打印它们在最后一次 val 上的指标（含 Box P/R/mAP50/mAP），便于直观对比。

  保留 v3.1 全部修复（参考原文件头部）。
  默认值不变：imgsz=960, lr0=0.005, optimizer=auto, cos_lr=True,
              mosaic=1.0, mixup=0.15, copy_paste=0.30, amp=True

CLI 兼容性：
    - 原 `--best-class <id>` 行为不变（默认 1=乒乓球）。
    - 设为 -1 时同时关闭两个最优追踪。
"""

# ---------------------------------------------------------------------------
# 必须在 import torch / ultralytics 之前设置 MIOpen 环境变量；
# 否则首次 conv 会卡在 "Compiling " 阶段且每次进程重启都重新编译。
# ---------------------------------------------------------------------------
import os
_DEF_MIOPEN_CACHE = os.path.expanduser("~/.cache/miopen")
os.makedirs(_DEF_MIOPEN_CACHE, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH",       _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR",   _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_FIND_MODE",          "FAST")
os.environ.setdefault("MIOPEN_FIND_ENFORCE",       "NONE")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",    "expandable_segments:True")

from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch
import yaml
import argparse
import sys
import shutil
from pathlib import Path

# [debug] 写入文件，避免被 ultralytics 输出系统吞掉
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"
_TMP_DIR.mkdir(exist_ok=True)
_DEBUG_LOG = _TMP_DIR / "_debug_bestclass.log"

def _debug(*args):
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(" ".join(str(a) for a in args) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 分割模型训练脚本 v3.2")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 路径（必需）")
    parser.add_argument("--pretrained", type=str, default="yolo26x-seg.pt",
                        help="预训练分割模型权重路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录名（默认: runs/train_<name>）")
    parser.add_argument("--name", type=str, default="seg_ft_v1", help="运行名称")

    # 训练时长 / 数据
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="训练分辨率（默认 960，小目标乒乓球更友好）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=str, default="ram",
                        choices=["ram", "disk", "false"],
                        help="数据缓存策略（ram 最快但占内存；disk 折衷；false 不缓存）")
    parser.add_argument("--rect", action="store_true",
                        help="启用矩形训练（同分组同形状，速度更快但 mosaic 关闭）")
    parser.add_argument("--multi-scale", action="store_true",
                        help="启用 multi-scale 训练（imgsz ± 50% 随机），提升对不同分辨率视频鲁棒性")
    parser.add_argument("--resume", action="store_true",
                        help="从 last.pt 续训（项目目录必须存在）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（提升可复现性）")

    # 学习率 / 优化器
    parser.add_argument("--lr0", type=float, default=0.05,
                        help="初始学习率（finetune 默认 0.05；from-scratch 用 0.01）")
    parser.add_argument("--lrf", type=float, default=0.001,
                        help="最终学习率因子（终值 = lr0 * lrf）")
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["auto", "SGD", "AdamW", "Adam", "RMSProp"])
    try:
        _BoolAction = argparse.BooleanOptionalAction
        parser.add_argument("--cos-lr", action=_BoolAction, default=True,
                            help="余弦退火 LR（默认开启；--no-cos-lr 禁用）")
    except AttributeError:
        parser.add_argument("--cos-lr", dest="cos_lr", action="store_true",
                            help="余弦退火 LR（默认开启）")
        parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false",
                            help="禁用余弦退火")
        parser.set_defaults(cos_lr=True)
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="在最后 N 个 epoch 关闭 mosaic（自动 clamp <= epochs/2）")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--freeze", type=int, default=0,
                        help="冻结前 N 层（>0 表示冻结 backbone 前 N 层；0 表示不冻结）")
    parser.add_argument("--dropout", type=float, default=0.0)

    # 数据增强
    parser.add_argument("--mosaic",     type=float, default=1.0)
    parser.add_argument("--mixup",      type=float, default=0.15)
    parser.add_argument("--copy-paste", type=float, default=0.30)
    parser.add_argument("--degrees",    type=float, default=0.0)
    parser.add_argument("--scale",      type=float, default=0.5)
    parser.add_argument("--fliplr",     type=float, default=0.5)

    # AMP
    try:
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="自动混合精度（默认开启；--no-amp 禁用）")
    except AttributeError:
        parser.add_argument("--amp", dest="amp", action="store_true",
                            help="自动混合精度（默认开启）")
        parser.add_argument("--no-amp", dest="amp", action="store_false")
        parser.set_defaults(amp=True)

    # 最优类别追踪 (v3.2: 同时保留两个最优)
    parser.add_argument("--best-class", type=int, default=-2,
                        help="按指定类别ID跟踪最优模型，会同时保存："
                             "best_class_<id>_ap.pt (类别 mAP50-95 最高) 与 "
                             "best_class_<id>_map50.pt (类别 mAP50 最高)。"
                             "默认 -2=自动识别乒乓球类；设为 -1 关闭追踪")
    # 类别损失加权 (v3.3: 自动识别乒乓球类, 其余类权重=1.0)
    parser.add_argument("--class-weight", type=float, default=5.0,
                        help="乒乓球分类损失权重倍率（自动识别数据集中'乒乓球'类；"
                             "其余类权重=1.0；设为 1.0 则关闭加权）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 标签转换：检测框 -> 多边形（兼容混合标签）
# ---------------------------------------------------------------------------
def convert_bbox_to_polygon(label_dir):
    label_paths = list(Path(label_dir).glob("*.txt"))
    converted_count = 0
    skipped_count = 0
    for lp in label_paths:
        new_lines = []
        with open(lp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    cls_id = parts[0]
                    x_c, y_c, w, h = map(float, parts[1:])
                except ValueError:
                    skipped_count += 1
                    continue
                if w <= 1e-4 or h <= 1e-4:
                    skipped_count += 1
                    continue
                x1, y1 = x_c - w / 2, y_c - h / 2
                x2, y2 = x_c + w / 2, y_c - h / 2
                x3, y3 = x_c + w / 2, y_c + h / 2
                x4, y4 = x_c - w / 2, y_c + h / 2
                coords = [max(0.0, min(1.0, v))
                          for v in (x1, y1, x2, y2, x3, y3, x4, y4)]
                new_lines.append(
                    f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords))
                converted_count += 1
            else:
                new_lines.append(line)
        with open(lp, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    print(f"✅ 标签转换: {converted_count} 个检测框→多边形, 跳过 {skipped_count} 个非法框 | 目录: {label_dir}")


def _resolve_label_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return Path(str(images_dir).replace("images", "labels"))


# ---------------------------------------------------------------------------
# best_class 追踪（进程级，DDP 仅 rank0 写）
# v3.2: 同时维护两个 best state: "ap" (mAP50-95) 与 "map50"
# ---------------------------------------------------------------------------
_BEST_METRICS = ("ap", "map50")

def _new_best_state():
    return {"score": 0.0, "epoch": -1, "box": None, "seg": None,
            "snapshot_path": None}

_best_states = {m: _new_best_state() for m in _BEST_METRICS}
_track_class_id = None
_model = None
_names = []
_nc = 0
_last_val_metrics = None  # 训练结束时给 last.pt 提供一份近似 metrics


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _safe_get_list(obj, attr):
    v = getattr(obj, attr, None)
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except Exception:
        return None


def _safe_get_int_list(obj, attr):
    v = getattr(obj, attr, None)
    if v is None:
        return None
    try:
        return [int(x) for x in v]
    except Exception:
        return None


def _snapshot_metrics(metrics):
    """从 validator.metrics 摘取 box/seg 的 P/R/mAP50/mAP，返回 (box_data, seg_data)."""
    if metrics is None:
        _debug("_snapshot_metrics: metrics is None")
        return None, None
    if not hasattr(metrics, "box"):
        _debug(f"_snapshot_metrics: metrics has no 'box' attr (type={type(metrics).__name__})")
        return None, None
    box_ap   = _safe_get_list(metrics.box, "ap")
    box_idx  = _safe_get_int_list(metrics.box, "ap_class_index")
    if box_ap is None or box_idx is None:
        _debug(f"_snapshot_metrics: box.ap={box_ap}, box.ap_class_index={box_idx}")
        return None, None
    box_data = {
        "p":    _safe_get_list(metrics.box, "p")    or [],
        "r":    _safe_get_list(metrics.box, "r")    or [],
        "ap50": _safe_get_list(metrics.box, "ap50") or [],
        "ap":   box_ap,
        "idx":  box_idx,
    }
    seg = getattr(metrics, "seg", None)
    seg_ap = _safe_get_list(seg, "ap") if seg is not None else None
    if seg is not None and seg_ap is not None:
        seg_data = {
            "p":    _safe_get_list(seg, "p")    or [],
            "r":    _safe_get_list(seg, "r")    or [],
            "ap50": _safe_get_list(seg, "ap50") or [],
            "ap":   seg_ap,
            "idx":  _safe_get_int_list(seg, "ap_class_index") or [],
        }
    else:
        seg_data = None
    return box_data, seg_data


def _format_metrics_table(box_data, seg_data, highlight_class_id=None):
    """打印 per-class 指标表。highlight_class_id 处加 ★ 标记。"""
    if box_data is None:
        return "    (无可用 metrics)"
    lines = []
    header = f"{'cls':>16s}  {'Box P':>7s} {'Box R':>7s} {'mAP50':>7s} {'mAP':>7s}"
    if seg_data is not None:
        header += f"  {'Seg P':>7s} {'Seg R':>7s} {'mAP50':>7s} {'mAP':>7s}"
    lines.append(header)
    lines.append("-" * len(header))

    idx_map = {ci: i for i, ci in enumerate(box_data["idx"])}
    seg_idx_map = {ci: i for i, ci in enumerate(seg_data["idx"])} if seg_data else {}

    for ci in range(_nc):
        name = _names[ci] if ci < len(_names) else f"cls{ci}"
        marker = " ★" if ci == highlight_class_id else ""
        bi = idx_map.get(ci)
        if bi is not None:
            row = (f"{name+marker:>16s}  "
                   f"{box_data['p'][bi]:7.3f} {box_data['r'][bi]:7.3f} "
                   f"{box_data['ap50'][bi]:7.3f} {box_data['ap'][bi]:7.3f}")
        else:
            row = f"{name+marker:>16s}  {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}"
        if seg_data is not None:
            si = seg_idx_map.get(ci)
            if si is not None:
                row += (f"  {seg_data['p'][si]:7.3f} {seg_data['r'][si]:7.3f} "
                        f"{seg_data['ap50'][si]:7.3f} {seg_data['ap'][si]:7.3f}")
            else:
                row += f"  {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}"
        lines.append(row)
    return "\n".join(lines)


def _extract_class_value(data, class_id, key, _caller_ctx=""):
    """从 _snapshot_metrics 返回的 dict 中取出指定类别的某项指标。"""
    if data is None or class_id is None or class_id < 0:
        _debug(f"_extract_class_value{_caller_ctx}: data={data is not None}, class_id={class_id}")
        return None
    try:
        i = data["idx"].index(class_id)
    except (ValueError, AttributeError):
        _debug(f"_extract_class_value{_caller_ctx}: class_id={class_id} NOT in idx={data.get('idx')}")
        return None
    arr = data.get(key)
    if not arr or i >= len(arr):
        _debug(f"_extract_class_value{_caller_ctx}: arr={arr}, i={i}, len(arr)={len(arr) if arr else 0}")
        return None
    return float(arr[i])


def _save_checkpoint(trainer, save_path: Path) -> bool:
    """把 last.pt 复制到 save_path；若 last.pt 不存在则 fallback 到 torch.save。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(trainer.last) if trainer.last and Path(trainer.last).exists() else None
    if src is None:
        try:
            torch.save({"model": trainer.model}, save_path)
            return True
        except Exception as e:
            print(f"⚠️ best 模型保存失败（fallback 也失败）: {e}")
            return False
    try:
        shutil.copy2(str(src), str(save_path))
        return True
    except Exception as e:
        print(f"⚠️ best 模型复制失败: {e}")
        return False


def _on_val_end(validator):
    """每次 val 后回调：尝试同时更新 ap / map50 两个 best_class checkpoint。"""
    _debug("_on_val_end: CALLBACK FIRED!")
    global _last_val_metrics
    if _track_class_id is None or _track_class_id < 0 or _model is None:
        _debug(f"_on_val_end: guard1 FAIL (_track_class_id={_track_class_id}, _model={'set' if _model else 'None'})")
        return
    if not _is_main_process():
        _debug("_on_val_end: guard2 FAIL (not main process)")
        return

    trainer = _model.trainer
    if trainer is None:
        _debug("_on_val_end: guard3 FAIL (_model.trainer is None)")
        return

    _debug(f"_on_val_end: calling _snapshot_metrics, type(validator)={type(validator).__name__}")
    box_data, seg_data = _snapshot_metrics(validator.metrics)
    if box_data is None:
        _debug("_on_val_end: _snapshot_metrics returned None")
        return

    _last_val_metrics = {"box": box_data, "seg": seg_data, "epoch": trainer.epoch}

    cls_ap    = _extract_class_value(box_data, _track_class_id, "ap", _caller_ctx=".ap")
    cls_map50 = _extract_class_value(box_data, _track_class_id, "ap50", _caller_ctx=".map50")
    if cls_ap is None and cls_map50 is None:
        _debug(f"_on_val_end: cls_ap and cls_map50 both None after epoch {trainer.epoch}")
        _debug(f"_on_val_end: box_data[idx]={box_data.get('idx')}, box_data[ap]={box_data.get('ap')}")
        return

    cname = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
    _debug(f"_on_val_end: epoch={trainer.epoch}, cls={_track_class_id} ({cname}), "
           f"ap={cls_ap}, map50={cls_map50}, best_ap={_best_states['ap']['score']}, best_map50={_best_states['map50']['score']}")

    triggered_save = {"done": False}
    def _ensure_save():
        if triggered_save["done"]:
            return
        try:
            trainer.save_model()
        except Exception as e:
            print(f"⚠️ trainer.save_model() 失败: {e}")
        triggered_save["done"] = True

    weights_dir = Path(trainer.save_dir) / "weights"

    # --- 跟踪 mAP50-95 (即 "ap") ---
    if cls_ap is not None and cls_ap > _best_states["ap"]["score"]:
        _ensure_save()
        save_path = weights_dir / f"best_class_{_track_class_id}_ap.pt"
        if _save_checkpoint(trainer, save_path):
            _best_states["ap"].update({
                "score": cls_ap, "epoch": trainer.epoch,
                "box": box_data, "seg": seg_data,
                "snapshot_path": save_path,
            })
            # ===== 新增：打印 per-class 指标表 =====
            cname = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
            print(f"\n{'='*70}")
            print(f"🏓 [类别{_track_class_id} mAP50-95 ↑]  "
                  f"epoch={trainer.epoch}  AP={cls_ap:.4f}  →  {save_path}")
            print(f"   类别: {_track_class_id} ({cname})")
            print(_format_metrics_table(box_data, seg_data,
                                         highlight_class_id=_track_class_id))
            print(f"{'='*70}")

    # --- 跟踪 mAP50 ---
    if cls_map50 is not None and cls_map50 > _best_states["map50"]["score"]:
        _ensure_save()
        save_path = weights_dir / f"best_class_{_track_class_id}_map50.pt"
        if _save_checkpoint(trainer, save_path):
            _best_states["map50"].update({
                "score": cls_map50, "epoch": trainer.epoch,
                "box": box_data, "seg": seg_data,
                "snapshot_path": save_path,
            })
            # ===== 新增：打印 per-class 指标表 =====
            cname = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
            print(f"\n{'='*70}")
            print(f"🏓 [类别{_track_class_id} mAP50    ↑]  "
                  f"epoch={trainer.epoch}  mAP50={cls_map50:.4f}  →  {save_path}")
            print(f"   类别: {_track_class_id} ({cname})")
            print(_format_metrics_table(box_data, seg_data,
                                         highlight_class_id=_track_class_id))
            print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 训练完成总览：列出所有保存模型并打印指标
# ---------------------------------------------------------------------------
def _print_final_summary(output_dir: Path, track_class_id):
    weights_dir = output_dir / "weights"
    print("\n" + "=" * 100)
    print("  📦 所有保存模型一览")
    print("=" * 100)

    # [(label, path, info_dict_or_None)]
    entries = []

    # best.pt - ultralytics 内部综合 fitness 最优
    entries.append(("best.pt", weights_dir / "best.pt", None))

    # last.pt - 最后一次 epoch；附最后一次 val 指标
    last_info = None
    if _last_val_metrics is not None:
        last_info = {
            "label": f"最后一次 val (epoch={_last_val_metrics['epoch']})",
            "box": _last_val_metrics["box"],
            "seg": _last_val_metrics["seg"],
        }
    entries.append(("last.pt", weights_dir / "last.pt", last_info))

    # best_class_<id>_ap.pt / best_class_<id>_map50.pt
    if track_class_id is not None and track_class_id >= 0:
        ap_info = _best_states["ap"]
        if ap_info["epoch"] >= 0:
            entries.append((
                f"best_class_{track_class_id}_ap.pt",
                weights_dir / f"best_class_{track_class_id}_ap.pt",
                {
                    "label": (f"类别{track_class_id} mAP50-95 最高 "
                              f"(epoch={ap_info['epoch']}, score={ap_info['score']:.4f})"),
                    "box": ap_info["box"],
                    "seg": ap_info["seg"],
                },
            ))
        map50_info = _best_states["map50"]
        if map50_info["epoch"] >= 0:
            entries.append((
                f"best_class_{track_class_id}_map50.pt",
                weights_dir / f"best_class_{track_class_id}_map50.pt",
                {
                    "label": (f"类别{track_class_id} mAP50 最高 "
                              f"(epoch={map50_info['epoch']}, score={map50_info['score']:.4f})"),
                    "box": map50_info["box"],
                    "seg": map50_info["seg"],
                },
            ))

    cname = (_names[track_class_id] if (track_class_id is not None
                                          and 0 <= track_class_id < len(_names)) else "?")

    for label, path, info in entries:
        print()
        print("─" * 100)
        exists = path.exists() if path else False
        size_str = ""
        if exists:
            try:
                sz = path.stat().st_size / (1024 * 1024)
                size_str = f"  [{sz:.1f} MB]"
            except Exception:
                pass
        status = "✅" if exists else "❌ (未找到)"
        print(f"{status} {label}{size_str}")
        print(f"    路径: {path}")
        if info is not None:
            print(f"    场景: {info['label']}")
            print(_format_metrics_table(info["box"], info["seg"],
                                         highlight_class_id=track_class_id))
        else:
            print("    场景: ultralytics 内部综合 fitness 最优（精确指标见 results.csv）")

    # ap vs map50 并列对比
    if (track_class_id is not None and track_class_id >= 0
            and _best_states["ap"]["epoch"] >= 0
            and _best_states["map50"]["epoch"] >= 0):
        print()
        print("=" * 100)
        print(f"  🆚 类别 {track_class_id} ({cname}) 两个最优模型并列对比")
        print("=" * 100)
        ap_state = _best_states["ap"]
        m50_state = _best_states["map50"]
        ap_cls_ap50 = _extract_class_value(ap_state["box"], track_class_id, "ap50")
        m50_cls_ap  = _extract_class_value(m50_state["box"], track_class_id, "ap")
        print(f"{'指标':<14s} {'best_class_*_ap.pt':>24s} {'best_class_*_map50.pt':>24s}")
        print("-" * 66)
        print(f"{'选优 epoch':<14s} {ap_state['epoch']:>24d} {m50_state['epoch']:>24d}")
        print(f"{'cls mAP50-95':<14s} {ap_state['score']:>24.4f} "
              f"{(m50_cls_ap if m50_cls_ap is not None else float('nan')):>24.4f}")
        print(f"{'cls mAP50':<14s} "
              f"{(ap_cls_ap50 if ap_cls_ap50 is not None else float('nan')):>24.4f} "
              f"{m50_state['score']:>24.4f}")

        if ap_state["epoch"] == m50_state["epoch"]:
            print("\nℹ️  两个最优来自同一 epoch；文件内容相同，文件名不同（建议保留一份即可）。")
        else:
            print("\nℹ️  两个最优来自不同 epoch，可分别用于不同场景：")
            print("    - 高 IoU 要求（打点定位 / 落台精确像素）→ best_class_<id>_ap.pt")
            print("    - 宽松 IoU 检测（多目标计数 / 召回优先）→ best_class_<id>_map50.pt")
    print("=" * 100)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global _track_class_id
    _track_class_id = args.best_class

    if not os.path.exists(args.data):
        print(f"❌ data.yaml 不存在: {args.data}")
        sys.exit(1)
    if not os.path.exists(args.pretrained):
        print(f"⚠️ 预训练权重不存在: {args.pretrained} -- 将尝试由 ultralytics 自动下载")

    with open(args.data, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # ---- 标签格式自动转换（检测→多边形） ----
    dataset_path = Path(data_config.get("path", "")).expanduser()
    for subset in ("train", "val"):
        img_dir_rel = data_config.get(subset, "")
        if not img_dir_rel:
            continue
        img_dir = Path(img_dir_rel)
        if not img_dir.is_absolute():
            img_dir = dataset_path / img_dir_rel
        label_dir = _resolve_label_dir(img_dir)
        if label_dir.exists():
            convert_bbox_to_polygon(label_dir)
        else:
            print(f"⚠️ 未找到标签目录: {label_dir}，跳过转换")

    # ---- 输出目录（须与 ultralytics 实际输出的 project/name 一致） ----
    # _PROJECT_ROOT 在文件顶部定义（src/ 的父目录 = 项目根）
    if args.output:
        output_dir = _PROJECT_ROOT / args.output
    else:
        output_dir = _PROJECT_ROOT / "runs" / args.name
    _debug(f"output_dir={output_dir}")

    # ---- close_mosaic clamp ----
    close_mosaic = max(0, min(args.close_mosaic, max(0, args.epochs // 2)))
    if close_mosaic != args.close_mosaic:
        print(f"ℹ️  close_mosaic 自动 clamp: {args.close_mosaic} → {close_mosaic} "
              f"(epochs={args.epochs})")

    cache_arg = False if args.cache == "false" else args.cache

    TRAIN_ARGS = {
        "model":         args.pretrained,
        "data":          args.data,
        "epochs":        args.epochs,
        "imgsz":         args.imgsz,
        "batch":         args.batch,
        "lr0":           args.lr0,
        "lrf":           args.lrf,
        "momentum":      0.937,
        "weight_decay":  0.0005,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic":  close_mosaic,
        "patience":      args.patience,
        "optimizer":     args.optimizer,
        "device":        "0" if torch.cuda.is_available() else "cpu",
        "workers":       args.workers,
        "project":       str(output_dir.parent),
        "name":          args.name,
        "exist_ok":      True,
        "plots":         True,
        "save":          True,
        "seed":          args.seed,
        "cos_lr":        args.cos_lr,
        "cache":         cache_arg,
        "rect":          args.rect,
        "multi_scale":   args.multi_scale,
        "resume":        args.resume,
        "amp":           args.amp,
        "freeze":        args.freeze if args.freeze > 0 else 0,
        "dropout":       args.dropout,
        "mosaic":        args.mosaic,
        "mixup":         args.mixup,
        "copy_paste":    args.copy_paste,
        "degrees":       args.degrees,
        "scale":         args.scale,
        "fliplr":        args.fliplr,
        "val":           True,  # 每个 epoch 验证，便于 best_class 追踪
    }

    # ---- 配置打印 ----
    print("=" * 70)
    print("🏓 YOLO 分割模型训练 v3.2 - 启动")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    model = YOLO(args.pretrained)
    global _model, _names, _nc
    _model = model

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n预训练模型: {args.pretrained}  | 参数量: {n_params/1e6:.1f}M")

    nc = data_config.get("nc", "N/A")
    raw_names = data_config.get("names", [])
    names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
    _names = names
    _nc = int(nc) if isinstance(nc, (int, float)) else len(_names)

    # --best-class 自动检测：在 names 中查找"乒乓球"
    sentinel_not_set = -2
    if _track_class_id == sentinel_not_set:
        try:
            auto_id = _names.index("乒乓球")
            _track_class_id = auto_id
            print(f"🎯 --best-class 自动检测: 乒乓球 = class {auto_id}")
        except ValueError:
            _track_class_id = 0
            print(f"ℹ️  数据集 names 中未找到'乒乓球'，--best-class 默认回退到 class 0")

    # --class-weight: 自动加权乒乓球类 (v3.3)
    if args.class_weight != 1.0 and _nc > 0:
        # 在 names 中查找"乒乓球"
        auto_ball_id = None
        try:
            auto_ball_id = _names.index("乒乓球")
        except ValueError:
            pass
        # 如果 names 中没找到，尝试复用 --best-class 自动检测的结果
        if auto_ball_id is None and _track_class_id is not None and _track_class_id >= 0:
            auto_ball_id = _track_class_id
        if auto_ball_id is not None:
            class_weights = torch.ones(_nc, dtype=torch.float32)
            class_weights[auto_ball_id] = args.class_weight
            model.model.class_weights = class_weights
            cname = _names[auto_ball_id] if auto_ball_id < len(_names) else "?"
            print(f"⚖️  类别加权: cls={auto_ball_id} ({cname}) 权重 x{args.class_weight:.1f}  "
                  f"(其余类 x1.0)")
        else:
            print(f"⚠️  --class-weight={args.class_weight} 但 names 中未找到'乒乓球'，跳过加权")

    print(f"数据集: {dataset_path}  | 类别数: {nc} - {names}")

    # 注册 best_class 回调（须在自动检测之后，此时 _track_class_id 已确定）
    if _track_class_id is not None and _track_class_id >= 0:
        model.add_callback("on_val_end", _on_val_end)
        cb_list = model.callbacks.get("on_val_end", [])
        _debug(f"add_callback done: on_val_end has {len(cb_list)} callbacks, "
               f"_on_val_end registered={_on_val_end in cb_list}")

    if _track_class_id is not None and _track_class_id >= 0:
        cname_ = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
        print(f"🎯 best_class 追踪：cls={_track_class_id} ({cname_})  "
              f"→ 同时保存 mAP50-95 最高与 mAP50 最高 两份 checkpoint")
    else:
        print("ℹ️  best_class 已禁用（--best-class=-1），仅保存默认的 best.pt / last.pt")

    print("\n" + "=" * 70)
    print("📋 训练配置:")
    print("=" * 70)
    for k, v in TRAIN_ARGS.items():
        if k not in ("data", "model"):
            print(f"  {k:18s}: {v}")

    print("\n" + "=" * 70)
    print("🚀 开始训练...")
    print("=" * 70)

    model.train(**TRAIN_ARGS)

    print("\n" + "=" * 80)
    print("  ✅ 训练完成!")
    print("=" * 80)

    # v3.2 新增：列出所有保存的模型及其指标
    _print_final_summary(output_dir, _track_class_id)
    print()


if __name__ == "__main__":
    main()