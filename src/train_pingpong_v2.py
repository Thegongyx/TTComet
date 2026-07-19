#!/usr/bin/env python3
"""
YOLO26-seg 乒乓球微调训练脚本 v3.3（针对小目标 + AMD ROCm 优化）
==============================================
v3.2 → v3.3 关键改动（均针对"识别乒乓球"这一核心目标）：
  [小目标架构] 新增 --p2：一键加载 yolo26-p2 架构（额外 P2 高分辨率检测头）
  [增强策略]   mixup 0.15→0（对极小目标有害，易把球与背景混淆）
               cutmix 保持 0（同理）
               copy_paste 0.30→0.10（分割任务保留少量，避免粘贴不合理小目标）
               新增 hsv_h/s/v 暴露（室内灯光/球桌反光变化大）
               degrees/translate/shear/perspective 全部暴露为 CLI
  [类别加权]   class_weight 默认 10.0→4.0（10x 过高会让球损失主导、其他类欠拟合）
  [E2E 一致性] 新增 --end2end，控制验证时是否走 NMS-free（须与推理一致）
  [预训练]     默认 yolo26x-seg.pt → yolo26m-seg.pt（8060S 上更实用）
  保留 v3.2 全部 best_class 双指标追踪 / 标签转换 / 最终总览功能。

注意：YOLO26 默认是 end-to-end (NMS-free) 模型。验证与推理必须用同样的
      end2end 设置，否则训练时看到的指标和部署表现不一致。
"""

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
    parser = argparse.ArgumentParser(description="YOLO26-seg 乒乓球训练脚本 v3.3")
    parser.add_argument("--data", type=str, required=True, help="data.yaml 路径（必需）")
    parser.add_argument("--pretrained", type=str, default="yolo26m-seg.pt",
                        help="预训练分割模型权重（默认 yolo26m-seg.pt）")
    parser.add_argument("--p2", action="store_true",
                        help="使用 yolo26-p2 架构（额外 P2 小目标检测头，乒乓球友好但更慢/更吃显存）。"
                             "启用时 --pretrained 仍用于加载权重做迁移。")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录名（默认: runs/train_<name>）")
    parser.add_argument("--name", type=str, default="seg_ft_v1", help="运行名称")

    # 训练时长 / 数据
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8,
                        help="批大小")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="训练分辨率（默认 960）")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=str, default="disk",
                        choices=["ram", "disk", "false"],
                        help="数据缓存（imgsz=1280 时 ram 易爆内存，默认 disk）")
    parser.add_argument("--rect", action="store_true",
                        help="矩形训练（关闭 mosaic，慎用）")
    parser.add_argument("--multi-scale", action="store_true",
                        help="multi-scale 训练（imgsz ± 50%）")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # 学习率 / 优化器
    parser.add_argument("--lr0", type=float, default=0.005)
    parser.add_argument("--lrf", type=float, default=0.005,
                        help="最终学习率因子（lr_final = lr0 * lrf）")
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["auto", "SGD", "AdamW", "Adam", "RMSProp"])
    try:
        _BoolAction = argparse.BooleanOptionalAction
        parser.add_argument("--cos-lr", action=_BoolAction, default=True)
    except AttributeError:
        parser.add_argument("--cos-lr", dest="cos_lr", action="store_true")
        parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
        parser.set_defaults(cos_lr=True)
    parser.add_argument("--close-mosaic", type=int, default=15,
                        help="最后 N epoch 关闭 mosaic（让模型在干净图上收尾，小目标定位更准）")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # 数据增强（针对乒乓球场景调优后的默认值）
    parser.add_argument("--mosaic",      type=float, default=1.0,
                        help="马赛克（对小目标有益，保留 1.0）")
    parser.add_argument("--mixup",       type=float, default=0.0,
                        help="MixUp（对极小目标有害，默认关闭）")
    parser.add_argument("--cutmix",      type=float, default=0.0,
                        help="CutMix（对小目标定位有害，默认关闭）")
    parser.add_argument("--copy-paste",  type=float, default=0.10,
                        help="Copy-Paste（分割专用，保留少量增加球样本多样性）")
    parser.add_argument("--copy-paste-mode", type=str, default="flip",
                        choices=["flip", "mixup"])
    parser.add_argument("--degrees",     type=float, default=0.0,
                        help="旋转角度（球桌场景一般保持 0；手持拍摄可设 5-10）")
    parser.add_argument("--translate",   type=float, default=0.1)
    parser.add_argument("--scale",       type=float, default=0.5,
                        help="尺度抖动（让模型见到不同大小的球，重要）")
    parser.add_argument("--shear",       type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0)
    parser.add_argument("--fliplr",      type=float, default=0.5)
    parser.add_argument("--flipud",      type=float, default=0.0,
                        help="上下翻转（球桌有固定上下结构，保持 0）")
    parser.add_argument("--hsv-h",       type=float, default=0.015)
    parser.add_argument("--hsv-s",       type=float, default=0.7)
    parser.add_argument("--hsv-v",       type=float, default=0.4,
                        help="亮度抖动（室内灯光/球桌反光变化大，保留较高值）")

    # AMP
    try:
        parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    except AttributeError:
        parser.add_argument("--amp", dest="amp", action="store_true")
        parser.add_argument("--no-amp", dest="amp", action="store_false")
        parser.set_defaults(amp=True)

    # E2E 一致性
    try:
        parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True,
                            help="验证时是否走 NMS-free end2end（须与推理脚本一致，默认 True）")
    except AttributeError:
        parser.add_argument("--end2end", dest="end2end", action="store_true")
        parser.add_argument("--no-end2end", dest="end2end", action="store_false")
        parser.set_defaults(end2end=True)

    # 最优类别追踪
    parser.add_argument("--best-class", type=int, default=-2,
                        help="按类别ID追踪最优（同时存 _ap.pt 与 _map50.pt）；"
                             "-2=自动识别乒乓球；-1=关闭")
    # 类别损失加权
    parser.add_argument("--class-weight", type=float, default=4.0,
                        help="乒乓球分类损失权重倍率（默认 4.0；过高(>8)会让其他类欠拟合；1.0=关闭）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 标签转换：检测框 -> 多边形
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
# best_class 追踪（双指标）
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
_last_val_metrics = None
_best_val_metrics = None


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


def _extract_f1_conf(metric_obj):
    f1_conf = {}
    if metric_obj is None:
        return f1_conf
    try:
        f1_curve = getattr(metric_obj, "f1_curve", None)
        px       = getattr(metric_obj, "px", None)
        idx      = getattr(metric_obj, "ap_class_index", None)
        if f1_curve is None or px is None or idx is None:
            return f1_conf
        for i, ci in enumerate(idx):
            if i < len(f1_curve):
                best_pos = int(f1_curve[i].argmax())
                f1_conf[int(ci)] = float(px[best_pos])
    except Exception as e:
        _debug(f"_extract_f1_conf 异常: {e}")
    return f1_conf


def _snapshot_metrics(metrics):
    if metrics is None:
        _debug("_snapshot_metrics: metrics is None")
        return None, None
    if not hasattr(metrics, "box"):
        _debug(f"_snapshot_metrics: no 'box' attr (type={type(metrics).__name__})")
        return None, None
    box_ap   = _safe_get_list(metrics.box, "ap")
    box_idx  = _safe_get_int_list(metrics.box, "ap_class_index")
    if box_ap is None or box_idx is None:
        _debug(f"_snapshot_metrics: box.ap={box_ap}, box.idx={box_idx}")
        return None, None
    box_data = {
        "p":    _safe_get_list(metrics.box, "p")    or [],
        "r":    _safe_get_list(metrics.box, "r")    or [],
        "ap50": _safe_get_list(metrics.box, "ap50") or [],
        "ap":   box_ap,
        "idx":  box_idx,
        "f1_conf": _extract_f1_conf(metrics.box),
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
            "f1_conf": _extract_f1_conf(seg),
        }
    else:
        seg_data = None
    return box_data, seg_data


def _format_metrics_table(box_data, seg_data, highlight_class_id=None):
    if box_data is None:
        return "    (无可用 metrics)"
    lines = []
    header = f"{'cls':>16s}  {'Box P':>7s} {'Box R':>7s} {'mAP50':>7s} {'mAP':>7s}  {'BF1阈':>7s}"
    if seg_data is not None:
        header += f"  {'Seg P':>7s} {'Seg R':>7s} {'mAP50':>7s} {'mAP':>7s}  {'SF1阈':>7s}"
    lines.append(header)
    lines.append("-" * len(header))

    idx_map = {ci: i for i, ci in enumerate(box_data["idx"])}
    seg_idx_map = {ci: i for i, ci in enumerate(seg_data["idx"])} if seg_data else {}

    box_f1 = box_data.get("f1_conf", {})
    seg_f1 = seg_data.get("f1_conf", {}) if seg_data else {}

    for ci in range(_nc):
        name = _names[ci] if ci < len(_names) else f"cls{ci}"
        marker = " ★" if ci == highlight_class_id else ""
        bi = idx_map.get(ci)
        if bi is not None:
            bf1 = box_f1.get(ci)
            row = (f"{name+marker:>16s}  "
                   f"{box_data['p'][bi]:7.3f} {box_data['r'][bi]:7.3f} "
                   f"{box_data['ap50'][bi]:7.3f} {box_data['ap'][bi]:7.3f}  "
                   f"{bf1:7.3f}" if bf1 is not None else
                   f"{name+marker:>16s}  "
                   f"{box_data['p'][bi]:7.3f} {box_data['r'][bi]:7.3f} "
                   f"{box_data['ap50'][bi]:7.3f} {box_data['ap'][bi]:7.3f}  "
                   f"{'':>7s}")
        else:
            bf1 = box_f1.get(ci)
            row = (f"{name+marker:>16s}  {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}  "
                   f"{bf1:7.3f}" if bf1 is not None else
                   f"{name+marker:>16s}  {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}  "
                   f"{'':>7s}")
        if seg_data is not None:
            si = seg_idx_map.get(ci)
            if si is not None:
                sf1 = seg_f1.get(ci)
                row += (f"  {seg_data['p'][si]:7.3f} {seg_data['r'][si]:7.3f} "
                        f"{seg_data['ap50'][si]:7.3f} {seg_data['ap'][si]:7.3f}  "
                        f"{sf1:7.3f}" if sf1 is not None else
                        f"  {seg_data['p'][si]:7.3f} {seg_data['r'][si]:7.3f} "
                        f"{seg_data['ap50'][si]:7.3f} {seg_data['ap'][si]:7.3f}  "
                        f"{'':>7s}")
            else:
                row += f"  {'-':>7s} {'-':>7s} {'-':>7s} {'-':>7s}  {'':>7s}"
        lines.append(row)
    return "\n".join(lines)


def _extract_class_value(data, class_id, key, _caller_ctx=""):
    if data is None or class_id is None or class_id < 0:
        _debug(f"_extract_class_value{_caller_ctx}: data={data is not None}, cls={class_id}")
        return None
    try:
        i = data["idx"].index(class_id)
    except (ValueError, AttributeError):
        _debug(f"_extract_class_value{_caller_ctx}: cls={class_id} NOT in idx={data.get('idx')}")
        return None
    arr = data.get(key)
    if not arr or i >= len(arr):
        return None
    return float(arr[i])


def _save_checkpoint(trainer, save_path: Path) -> bool:
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
    _debug("_on_val_end: CALLBACK FIRED!")
    global _last_val_metrics
    if _track_class_id is None or _track_class_id < 0 or _model is None:
        return
    if not _is_main_process():
        return
    trainer = _model.trainer
    if trainer is None:
        return
    box_data, seg_data = _snapshot_metrics(validator.metrics)
    if box_data is None:
        return
    _last_val_metrics = {"box": box_data, "seg": seg_data, "epoch": trainer.epoch}

    global _best_val_metrics
    if box_data and len(box_data.get("ap", [])) > 0:
        _mean_ap   = float(sum(box_data["ap"])   / len(box_data["ap"]))
        _mean_ap50 = float(sum(box_data["ap50"]) / len(box_data["ap50"]))
        _fit = 0.1 * _mean_ap50 + 0.9 * _mean_ap
        if _best_val_metrics is None or _fit > _best_val_metrics["fitness"]:
            _best_val_metrics = {"fitness": _fit, "epoch": trainer.epoch,
                                 "box": box_data, "seg": seg_data}

    cls_ap    = _extract_class_value(box_data, _track_class_id, "ap", ".ap")
    cls_map50 = _extract_class_value(box_data, _track_class_id, "ap50", ".map50")
    if cls_ap is None and cls_map50 is None:
        return

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

    if cls_ap is not None and cls_ap > _best_states["ap"]["score"]:
        _ensure_save()
        save_path = weights_dir / f"best_class_{_track_class_id}_ap.pt"
        if _save_checkpoint(trainer, save_path):
            _best_states["ap"].update({"score": cls_ap, "epoch": trainer.epoch,
                                       "box": box_data, "seg": seg_data,
                                       "snapshot_path": save_path})
            cname = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
            print(f"\n{'='*70}")
            print(f"🏓 [类别{_track_class_id} mAP50-95 ↑]  "
                  f"epoch={trainer.epoch}  AP={cls_ap:.4f}  →  {save_path}")
            print(f"   类别: {_track_class_id} ({cname})")
            print(_format_metrics_table(box_data, seg_data, _track_class_id))
            print(f"{'='*70}")

    if cls_map50 is not None and cls_map50 > _best_states["map50"]["score"]:
        _ensure_save()
        save_path = weights_dir / f"best_class_{_track_class_id}_map50.pt"
        if _save_checkpoint(trainer, save_path):
            _best_states["map50"].update({"score": cls_map50, "epoch": trainer.epoch,
                                          "box": box_data, "seg": seg_data,
                                          "snapshot_path": save_path})
            cname = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
            print(f"\n{'='*70}")
            print(f"🏓 [类别{_track_class_id} mAP50    ↑]  "
                  f"epoch={trainer.epoch}  mAP50={cls_map50:.4f}  →  {save_path}")
            print(f"   类别: {_track_class_id} ({cname})")
            print(_format_metrics_table(box_data, seg_data, _track_class_id))
            print(f"{'='*70}")


def _print_final_summary(output_dir: Path, track_class_id):
    weights_dir = output_dir / "weights"
    print("\n" + "=" * 100)
    print("  📦 所有保存模型一览")
    print("=" * 100)
    entries = []
    best_pt_info = None
    if _best_val_metrics is not None:
        best_pt_info = {"label": f"综合 fitness 最优 (epoch={_best_val_metrics['epoch']})",
                        "box": _best_val_metrics["box"], "seg": _best_val_metrics["seg"]}
    entries.append(("best.pt", weights_dir / "best.pt", best_pt_info))
    last_info = None
    if _last_val_metrics is not None:
        last_info = {"label": f"最后一次 val (epoch={_last_val_metrics['epoch']})",
                     "box": _last_val_metrics["box"], "seg": _last_val_metrics["seg"]}
    entries.append(("last.pt", weights_dir / "last.pt", last_info))

    if track_class_id is not None and track_class_id >= 0:
        ap_info = _best_states["ap"]
        if ap_info["epoch"] >= 0:
            entries.append((f"best_class_{track_class_id}_ap.pt",
                            weights_dir / f"best_class_{track_class_id}_ap.pt",
                            {"label": (f"类别{track_class_id} mAP50-95 最高 "
                                       f"(epoch={ap_info['epoch']}, score={ap_info['score']:.4f})"),
                             "box": ap_info["box"], "seg": ap_info["seg"]}))
        map50_info = _best_states["map50"]
        if map50_info["epoch"] >= 0:
            entries.append((f"best_class_{track_class_id}_map50.pt",
                            weights_dir / f"best_class_{track_class_id}_map50.pt",
                            {"label": (f"类别{track_class_id} mAP50 最高 "
                                       f"(epoch={map50_info['epoch']}, score={map50_info['score']:.4f})"),
                             "box": map50_info["box"], "seg": map50_info["seg"]}))

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
            print(_format_metrics_table(info["box"], info["seg"], track_class_id))
        else:
            print("    场景: ultralytics 内部综合 fitness 最优（精确指标见 results.csv）")

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
            print("\nℹ️  两个最优来自同一 epoch；文件内容相同。")
        else:
            print("\nℹ️  两个最优来自不同 epoch，可分别用于不同场景：")
            print("    - 高 IoU（落台精确像素）→ best_class_<id>_ap.pt")
            print("    - 宽松 IoU（多目标计数 / 召回优先）→ best_class_<id>_map50.pt")
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
    if not args.p2 and not os.path.exists(args.pretrained):
        print(f"⚠️ 预训练权重不存在: {args.pretrained} -- 将尝试由 ultralytics 自动下载")

    with open(args.data, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # ---- 标签格式自动转换 ----
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

    if args.output:
        output_dir = _PROJECT_ROOT / args.output
    else:
        output_dir = _PROJECT_ROOT / "runs" / args.name
    _debug(f"output_dir={output_dir}")

    close_mosaic = max(0, min(args.close_mosaic, max(0, args.epochs // 2)))
    if close_mosaic != args.close_mosaic:
        print(f"ℹ️  close_mosaic 自动 clamp: {args.close_mosaic} → {close_mosaic}")

    cache_arg = False if args.cache == "false" else args.cache

    # ---- 模型加载：--p2 时用架构 yaml + 迁移权重 ----
    if args.p2:
        # 选用与 m 规模对应的 p2 架构；ultralytics 会按 scale 后缀缩放
        p2_cfg = "yolo26-p2.yaml"
        print(f"🧩 使用 P2 小目标架构: {p2_cfg}（迁移权重: {args.pretrained}）")
        model = YOLO(p2_cfg)
        try:
            model.load(args.pretrained)
            print(f"✅ 已加载迁移权重: {args.pretrained}")
        except Exception as e:
            print(f"⚠️ P2 迁移权重加载失败（将从头训）: {e}")
        model_arg_for_train = p2_cfg
    else:
        model = YOLO(args.pretrained)
        model_arg_for_train = args.pretrained

    global _model, _names, _nc
    _model = model

    TRAIN_ARGS = {
        "model":         model_arg_for_train,
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
        # 增强
        "mosaic":        args.mosaic,
        "mixup":         args.mixup,
        "cutmix":        args.cutmix,
        "copy_paste":    args.copy_paste,
        "copy_paste_mode": args.copy_paste_mode,
        "degrees":       args.degrees,
        "translate":     args.translate,
        "scale":         args.scale,
        "shear":         args.shear,
        "perspective":   args.perspective,
        "fliplr":        args.fliplr,
        "flipud":        args.flipud,
        "hsv_h":         args.hsv_h,
        "hsv_s":         args.hsv_s,
        "hsv_v":         args.hsv_v,
        # 验证 / E2E
        "val":           True,
    }
    # end2end 只在 ultralytics 支持时传递（兼容旧版本）
    try:
        import inspect
        if "end2end" in inspect.signature(model.train).parameters or True:
            TRAIN_ARGS["end2end"] = args.end2end
    except Exception:
        pass

    print("=" * 70)
    print("🏓 YOLO26-seg 乒乓球训练 v3.3 - 启动")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n模型: {model_arg_for_train}  | 参数量: {n_params/1e6:.1f}M")

    nc = data_config.get("nc", "N/A")
    raw_names = data_config.get("names", [])
    names = list(raw_names.values()) if isinstance(raw_names, dict) else list(raw_names)
    _names = names
    _nc = int(nc) if isinstance(nc, (int, float)) else len(_names)

    sentinel_not_set = -2
    if _track_class_id == sentinel_not_set:
        try:
            auto_id = _names.index("乒乓球")
            _track_class_id = auto_id
            print(f"🎯 --best-class 自动检测: 乒乓球 = class {auto_id}")
        except ValueError:
            _track_class_id = 0
            print(f"ℹ️  names 中未找到'乒乓球'，--best-class 回退到 class 0")

    if args.class_weight != 1.0 and _nc > 0:
        auto_ball_id = None
        try:
            auto_ball_id = _names.index("乒乓球")
        except ValueError:
            pass
        if auto_ball_id is None and _track_class_id is not None and _track_class_id >= 0:
            auto_ball_id = _track_class_id
        if auto_ball_id is not None:
            class_weights = torch.ones(_nc, dtype=torch.float32)
            class_weights[auto_ball_id] = args.class_weight
            model.model.class_weights = class_weights
            cname = _names[auto_ball_id] if auto_ball_id < len(_names) else "?"
            print(f"⚖️  类别加权: cls={auto_ball_id} ({cname}) 权重 x{args.class_weight:.1f}")
        else:
            print(f"⚠️  --class-weight={args.class_weight} 但未找到'乒乓球'，跳过加权")

    print(f"数据集: {dataset_path}  | 类别数: {nc} - {names}")

    if _track_class_id is not None and _track_class_id >= 0:
        model.add_callback("on_val_end", _on_val_end)
        cb_list = model.callbacks.get("on_val_end", [])
        _debug(f"add_callback done: on_val_end has {len(cb_list)} callbacks")
        cname_ = (_names[_track_class_id] if _track_class_id < len(_names) else "?")
        print(f"🎯 best_class 追踪：cls={_track_class_id} ({cname_})")
    else:
        print("ℹ️  best_class 已禁用")

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
    _print_final_summary(output_dir, _track_class_id)
    print()


if __name__ == "__main__":
    main()