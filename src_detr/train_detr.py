#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR 乒乓球微调训练脚本
============================================================
用 RF-DETR (https://github.com/roboflow/rf-detr) 替代 YOLO26 训练乒乓球检测/分割模型。

设计要点
--------
* 直接复用项目现有的 YOLO 数据集 (data.yaml + images/{train,val} + labels/{train,val})，
  不复制任何图片: 在 tmp/ 下用「目录联接(junction)/软链接」搭出一个 RF-DETR 认识的视图:
      tmp/detr_dataset_<name>/
          data.yaml            # train: train/images, val: valid/images
          train/images -> <原图目录>     (junction)
          train/labels -> <原标签目录>   (junction)
          valid/images -> <原图目录>
          valid/labels -> <原标签目录>
  RF-DETR 既支持 COCO 也原生支持 YOLO 格式; 其格式探测要求 train/ valid/ 下各有 images/ labels/。
* 标签兼容 bbox(5 列) 与 polygon(9+ 列) 两种 YOLO 写法 (RF-DETR 自动把 bbox 当 4 角点多边形)。
* 支持检测 (detect) 与实例分割 (seg) 两种任务; 项目高光剪辑需要球桌/球网的 polygon，
  因此默认使用 **分割 (seg)** 模型。

用法示例 (conda 环境 rfdetr)
------------------------------------------------------------
# 默认: RF-DETR Seg-Medium 微调
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --name pingpong_detr_v1

# 更大模型 / 自定义分辨率 / 更多 epoch
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --size large --resolution 672 `
    --epochs 150 --batch 2 --grad-accum 8 `
    --name pingpong_detr_l

# 纯检测 (不需要分割 polygon)
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --task detect --size medium --name pingpong_detr_det

# 断点续训
conda run -n rfdetr python src_detr/train_detr.py `
    --data train_data/all22_split/data.yaml `
    --resume runs_detr/pingpong_detr_v1/checkpoint.pth
"""

import os
import sys
import json
import math
import shutil
import argparse
import platform
import subprocess
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# AMD ROCm / MIOpen 环境变量 (必须在 import torch 之前设置!)
# ---------------------------------------------------------------------------
_IS_WINDOWS = platform.system() == "Windows"
if _IS_WINDOWS:
    _DEF_MIOPEN_CACHE = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "miopen"
    )
else:
    _DEF_MIOPEN_CACHE = os.path.expanduser("~/.cache/miopen")
os.makedirs(_DEF_MIOPEN_CACHE, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH",     _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_FIND_MODE",        "FAST")
os.environ.setdefault("MIOPEN_FIND_ENFORCE",     "NONE")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",  "expandable_segments:True")
# 某些 PyTorch/ROCm 版本只认 CUDA 命名的分配器配置, 两个都设以启用 expandable_segments (抗碎片)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# AMD ROCm: 启用 Flash / Mem-Efficient Attention 实验内核 (训练前向/反向同受益)。
# 用 --no-aotriton 关闭 (必须在 import torch 之前设置)。
if "--no-aotriton" not in sys.argv:
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"
_TMP_DIR.mkdir(exist_ok=True)


def _exit(code):
    """干净退出 (Windows: 用 TerminateProcess 跳过会卡死的 ROCm/MIOpen DLL 卸载)。
    详见 src_detr/inference_video_detr.py 中的同名函数。"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if _IS_WINDOWS:
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            k32.TerminateProcess(k32.GetCurrentProcess(), int(code))
        except Exception:
            pass
    os._exit(int(code))


# ---------------------------------------------------------------------------
# 支持的 RF-DETR 模型规模
# ---------------------------------------------------------------------------
# size -> (检测类名, 分割类名)
_MODEL_REGISTRY = {
    "nano":    ("RFDETRNano",    "RFDETRSegNano"),
    "small":   ("RFDETRSmall",   "RFDETRSegSmall"),
    "medium":  ("RFDETRMedium",  "RFDETRSegMedium"),
    "large":   ("RFDETRLarge",   "RFDETRSegLarge"),
    "xlarge":  ("RFDETRXLarge",  "RFDETRSegXLarge"),   # 需 rfdetr[plus]
    "2xlarge": ("RFDETR2XLarge", "RFDETRSeg2XLarge"),  # 需 rfdetr[plus]
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="RF-DETR 乒乓球微调训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据 / 输出
    ap.add_argument("--data", required=True,
                    help="YOLO data.yaml 路径 (如 train_data/all22_split/data.yaml)")
    ap.add_argument("--size", default="medium",
                    choices=list(_MODEL_REGISTRY.keys()),
                    help="RF-DETR 模型规模 (默认 medium)")
    ap.add_argument("--task", default="seg", choices=["seg", "detect"],
                    help="seg=实例分割 (默认, 高光剪辑需要 table/net polygon); detect=纯检测")
    ap.add_argument("--name", default="detr_ft_v1", help="运行名称")
    ap.add_argument("--output", default=None,
                    help="输出目录 (默认: runs_detr/<name>)")
    ap.add_argument("--dataset-view", default=None,
                    help="RF-DETR 数据集视图目录 (默认: tmp/detr_dataset_<name>); "
                         "设为 'keep' 可复用已存在的视图而不重建")
    # 训练
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16,
                    help="单卡 batch size (默认 16; 显存大可到 32/64)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="YOLO 兼容别名: 训练分辨率 (映射到 RF-DETR resolution; "
                         "必须能被 patch_size*num_windows 整除, 如 medium/seg 为 24)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="梯度累积步数 (默认 1)。等效 batch = batch × grad_accum；"
                         "显存不足时减小 batch 并增大此值维持 16")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr0", type=float, default=None,
                    help="YOLO 兼容别名: 初始学习率 (映射到 --lr)")
    ap.add_argument("--lr-encoder", type=float, default=None,
                    help="backbone 学习率 (默认沿用 RF-DETR 默认)")
    ap.add_argument("--resolution", type=int, default=None,
                    help="自定义训练分辨率; 必须能被 patch_size*num_windows 整除 "
                         "(medium/seg 为 24)。默认使用模型自带分辨率")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers (默认 8)")
    ap.add_argument("--eval-interval", type=int, default=1,
                    help="每 N epoch 验证一次 (默认 1=每轮都验证; 调大可省时间)")
    ap.add_argument("--eval-batch-size", type=int, default=32,
                    help="验证批大小 (默认 32; 验证无梯度, 调大可减少 GPU 空转)")
    ap.add_argument("--eval-masks-head-resolution", action=argparse.BooleanOptionalAction, default=True,
                    help="验证时跳过 mask 上采样到原图分辨率 (默认开, 分割验证明显更快; "
                         "segm mAP 变成 mask head 分辨率下, 与全分辨率不可比)")
    ap.add_argument("--eval-max-dets", type=int, default=100,
                    help="验证 COCO 评估每图最大检测数 (默认 100; 越小越快)")
    ap.add_argument("--num-sanity-val-steps", type=int, default=0,
                    help="训练前 sanity 验证步数 (默认 0=跳过, 省时间)")
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,
                    help="dataloader pin_memory (默认开, 有 GPU 时略快)")
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True,
                    help="dataloader persistent_workers (默认开, 避免每 epoch 重建 worker)")
    ap.add_argument("--prefetch-factor", type=int, default=3,
                    help="每 worker 预取批数 (默认 3, 减少 GPU 空转)")
    ap.add_argument("--early-stopping", action="store_true", help="启用早停")
    ap.add_argument("--patience", type=int, default=15, help="早停耐心值")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True,
                    help="是否使用 EMA 权重 (默认开启)")
    ap.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True,
                    help="多尺度训练 (默认开启)")
    ap.add_argument("--gpu-postprocess", action="store_true",
                    help="使用 GPU (kornia) 数据增强后端 (默认 CPU torchvision)")
    # 权重
    ap.add_argument("--pretrained", "--model", dest="pretrained", default=None,
                    help="预训练权重路径 (本地 .pt 如 rf-detr-seg-medium.pt, 或官方权重名); "
                         "默认自动下载对应规模的官方权重")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="不使用预训练权重 (从零训练, 无网络时可用)")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="从 checkpoint(.pth) 恢复训练; 不带值时自动使用输出目录下最新 checkpoint")
    ap.add_argument("--freeze", type=int, default=None,
                    help="YOLO 兼容: >0 时冻结 backbone (= --freeze-encoder)")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="冻结 backbone (省显存)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="梯度检查点 (省显存, 变慢)")
    ap.add_argument("--compile", action="store_true",
                    help="[实验] torch.compile 编译模型 (ModelConfig.compile); "
                         "固定输入形状时收益最大, 建议同时 --no-multi-scale; ROCm 上可能不稳定")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="[实验] 用 CUDA/HIP graph 捕获训练前向 (ModelConfig.cuda_graphs); 小 batch 减少启动开销")
    ap.add_argument("--group-detr", type=int, default=None,
                    help="[加速] 训练时的 group_detr (默认模型自带 13)。调小(如 4/1)可显著降低 "
                         "decoder 计算与匹配开销, 但可能影响收敛")
    ap.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None,
                    help="使用 fuse 优化器 (ModelConfig.fused_optimizer; 默认模型自带 True)")
    # 设备 / 日志
    ap.add_argument("--device", default=None,
                    help="设备: 0 / cuda:0 / cpu (默认自动)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="YOLO 兼容: --no-amp 等价于 --amp-dtype none")
    ap.add_argument("--amp-dtype", default="auto",
                    choices=["auto", "bf16", "fp16", "none"],
                    help="训练 AMP 精度: auto(默认) / bf16 / fp16 / none(fp32)。"
                         "ROCm 上 auto 通常选 fp16")
    ap.add_argument("--no-aotriton", action="store_true",
                    help="关闭 AMD Flash/Mem-Efficient Attention 实验内核 (默认开启)")
    ap.add_argument("--log-per-class", action=argparse.BooleanOptionalAction, default=True,
                    help="打印/记录每类 AP (含乒乓球) 与每类表 (默认开; 关闭可省少量计算/显存)")
    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--progress-bar", default="tqdm", choices=["tqdm", "rich", "none"],
                    help="控制台训练进度条风格 (默认 tqdm; rich 更详细; none 关闭)")
    ap.add_argument("--seed", type=int, default=0)
    # best_class 追踪
    ap.add_argument("--best-class", type=int, default=-2,
                    help="按类别ID追踪最优并保存 best_class_<id>_ap.pth; "
                         "-2=自动识别乒乓球; -1=关闭")
    # YOLO 兼容参数 (RF-DETR 不支持, 仅接受并打印忽略警告, 避免原有命令行报错)
    ap.add_argument("--p2", action="store_true", help="[YOLO 兼容, 忽略] P2 小目标检测头")
    ap.add_argument("--cache", default=None, help="[YOLO 兼容, 忽略] 数据缓存")
    ap.add_argument("--rect", action="store_true", help="[YOLO 兼容, 忽略] 矩形训练")
    ap.add_argument("--lrf", type=float, default=None, help="[YOLO 兼容, 忽略] 最终学习率因子")
    ap.add_argument("--optimizer", default=None, help="[YOLO 兼容, 忽略] 优化器")
    ap.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True,
                    help="[YOLO 兼容, 忽略] 余弦学习率 (RF-DETR 固定 cosine)")
    ap.add_argument("--close-mosaic", type=int, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--warmup-epochs", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--dropout", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--class-weight", type=float, default=1.0,
                    help="乒乓球类别的分类损失权重 (默认 1.0=不加权; 等价 YOLO --class-weight)")
    ap.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True,
                    help="[YOLO 兼容, 忽略] RF-DETR 始终 NMS-free")
    ap.add_argument("--mosaic", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--mixup", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--cutmix", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--copy-paste", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--copy-paste-mode", default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--degrees", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--translate", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--scale", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--shear", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--perspective", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--fliplr", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--flipud", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--hsv-h", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--hsv-s", type=float, default=None, help="[YOLO 兼容, 忽略]")
    ap.add_argument("--hsv-v", type=float, default=None, help="[YOLO 兼容, 忽略]")
    return ap.parse_args()


# YOLO 兼容参数 -> 中文描述 (仅当用户显式提供时打印忽略警告)
_UNSUPPORTED_YOLO_ARGS = [
    ("p2", "P2 小目标检测头"),
    ("cache", "数据缓存"),
    ("rect", "矩形训练"),
    ("lrf", "最终学习率因子"),
    ("optimizer", "优化器"),
    ("close_mosaic", "close-mosaic"),
    ("warmup_epochs", "warmup-epochs"),
    ("dropout", "dropout"),
    ("mosaic", "mosaic"),
    ("mixup", "mixup"),
    ("cutmix", "cutmix"),
    ("copy_paste", "copy-paste"),
    ("copy_paste_mode", "copy-paste-mode"),
    ("degrees", "degrees"),
    ("translate", "translate"),
    ("scale", "scale"),
    ("shear", "shear"),
    ("perspective", "perspective"),
    ("fliplr", "fliplr"),
    ("flipud", "flipud"),
    ("hsv_h", "hsv-h"),
    ("hsv_s", "hsv-s"),
    ("hsv_v", "hsv-v"),
]


def _warn_unsupported_args(args):
    """对用户显式传入但 RF-DETR 不支持的 YOLO 参数打印中文忽略警告。"""
    for name, _desc in _UNSUPPORTED_YOLO_ARGS:
        val = getattr(args, name, None)
        if val is None or val is False:
            continue
        print(f"⚠️ RF-DETR 不支持参数 --{name.replace('_', '-')}，已忽略")
    if getattr(args, "cos_lr", True) is False:
        print("⚠️ RF-DETR 不支持参数 --no-cos-lr，已忽略")
    if getattr(args, "end2end", True) is False:
        print("⚠️ RF-DETR 不支持参数 --no-end2end，已忽略")


def _install_class_weight_hook(class_weights):
    """等价 YOLO 的 model.model.class_weights (ultralytics: bce_loss *= class_weights)。

    RF-DETR 无原生每类权重, 这里补丁 SetCriterion.loss_labels, 在默认的 IA-BCE 分支里
    对逐元素分类损失乘以每类权重后再求和。仅当 ia_bce_loss=True 且类别数匹配时生效,
    其余情况回退原实现。返回 restore 回调 (训练结束在 finally 里恢复)。
    """
    try:
        from rfdetr.models import criterion as _crit
        SetCriterion = _crit.SetCriterion
        orig = SetCriterion.loss_labels
        _F = _crit.F
        _box_ops = _crit.box_ops
        _accuracy = _crit.accuracy

        def _patched(self, outputs, targets, indices, num_boxes, log=True, matched_targets=None):
            src_logits = outputs.get("pred_logits")
            if (not getattr(self, "ia_bce_loss", False) or src_logits is None
                    or src_logits.ndim < 3 or src_logits.shape[-1] != class_weights.numel()):
                return orig(self, outputs, targets, indices, num_boxes,
                            log=log, matched_targets=matched_targets)
            if matched_targets is None:
                idx = self._get_src_permutation_idx(indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                target_boxes = None
            else:
                idx = matched_targets.source_indices
                target_classes_o = matched_targets.labels
                target_boxes = matched_targets.boxes
            if target_boxes is None:
                target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            alpha = self.focal_alpha
            gamma = 2
            src_boxes = outputs["pred_boxes"][idx]
            iou_targets, _ = _box_ops.elementwise_box_iou(
                _box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                _box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob ** gamma
            pos_ind = list(idx)
            pos_ind.append(target_classes_o)
            t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
            t = torch.clamp(t, 0.01).detach()
            pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
            neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
            loss_ce = neg_weights * src_logits - _F.logsigmoid(src_logits) * (pos_weights + neg_weights)
            # ↓ 每类加权 (等价 YOLO bce_loss *= class_weights)
            loss_ce = loss_ce * class_weights.to(loss_ce.device, loss_ce.dtype).view(1, -1)
            losses = {"loss_ce": loss_ce.sum() / num_boxes}
            if log:
                losses["class_error"] = 100 - _accuracy(src_logits[idx], target_classes_o)[0]
            return losses

        SetCriterion.loss_labels = _patched

        def _restore():
            try:
                SetCriterion.loss_labels = orig
            except Exception:
                pass
        return _restore
    except Exception as e:
        print(f"⚠️ 类别加权安装失败, 已跳过: {e}")
        return None


# ---------------------------------------------------------------------------
# 数据集: YOLO data.yaml -> RF-DETR 视图 (junction/symlink, 不复制图片)
# ---------------------------------------------------------------------------
def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _names_list(raw_names):
    """data.yaml 的 names 既可能是 list 也可能是 {0:..} dict, 统一成有序 list。"""
    if isinstance(raw_names, dict):
        items = sorted(raw_names.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in items]
    if isinstance(raw_names, (list, tuple)):
        return [str(v) for v in raw_names]
    raise ValueError(f"无法解析 data.yaml 的 names 字段: {raw_names!r}")


def _resolve_images_labels(root: Path, rel: str):
    """由 split 的 images 相对路径推导 images / labels 目录。"""
    sp = Path(rel)
    if not sp.is_absolute():
        sp = root / rel
    parts = list(sp.parts)
    if "images" in parts:
        i = parts.index("images")
        lb = Path(*parts[:i], "labels", *parts[i + 1:])
    else:
        lb = sp.parent / "labels"
    return sp, lb


def _link_or_copy(link: Path, target: Path):
    """优先 junction -> symlink -> copytree。返回使用的方式。"""
    link = Path(link)
    target = Path(target).resolve()
    if os.path.lexists(str(link)):
        return "exists"
    link.parent.mkdir(parents=True, exist_ok=True)

    if _IS_WINDOWS:
        # mklink /J 创建目录联接, 普通用户即可, 无需开发者模式
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        if r.returncode == 0 and os.path.lexists(str(link)):
            return "junction"
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return "symlink"
    except OSError:
        pass
    print(f"⚠️  无法创建链接, 改为复制 (可能较慢): {link} <- {target}")
    shutil.copytree(str(target), str(link))
    return "copy"


def build_dataset_view(data_yaml: Path, view_dir: Path):
    """在 view_dir 下搭建 RF-DETR 可识别的 YOLO 数据集视图。返回 (view_dir, names)。"""
    if not data_yaml.exists():
        print(f"❌ data.yaml 不存在: {data_yaml}")
        sys.exit(1)

    raw = _read_yaml(data_yaml)
    base = data_yaml.parent

    # 解析 path 字段: 可能是相对项目根、相对 data.yaml 目录, 或绝对路径
    path_field = raw.get("path")
    root = base
    if path_field:
        p = Path(os.path.expanduser(str(path_field)))
        if p.is_absolute():
            root = p
        else:
            cand_base = (base / p)
            cand_proj = (_PROJECT_ROOT / p)
            root = cand_base if cand_base.exists() else (cand_proj if cand_proj.exists() else cand_base)

    train_rel = raw.get("train")
    val_rel = raw.get("val") or raw.get("valid")
    if not train_rel or not val_rel:
        print(f"❌ data.yaml 缺少 train / val 字段: {raw}")
        sys.exit(1)
    train_img, train_lb = _resolve_images_labels(root, str(train_rel))
    val_img, val_lb = _resolve_images_labels(root, str(val_rel))

    for label, d in (("train images", train_img), ("train labels", train_lb),
                     ("val images", val_img), ("val labels", val_lb)):
        if not d.is_dir():
            print(f"❌ {label} 目录不存在: {d}")
            sys.exit(1)

    names = _names_list(raw.get("names", []))
    nc = raw.get("nc", len(names))

    view_dir = Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)

    modes = []
    for split, imgd, lbd in (("train", train_img, train_lb),
                             ("valid", val_img, val_lb)):
        modes.append((split, "images", _link_or_copy(view_dir / split / "images", imgd)))
        modes.append((split, "labels", _link_or_copy(view_dir / split / "labels", lbd)))

    out_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "nc": int(nc),
        "names": names,
    }
    with open(view_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"📂 RF-DETR 数据集视图: {view_dir}")
    for split, kind, mode in modes:
        print(f"    {split}/{kind:<6s} [{mode}]")
    print(f"    类别 ({nc}): {names}")
    return view_dir, names


# ---------------------------------------------------------------------------
# best_class 追踪 (对齐 YOLO train_pingpong_v2.py 的 _on_val_end / 指标表)
# ---------------------------------------------------------------------------
try:
    from pytorch_lightning import Callback as _PLCallback
except Exception:  # pragma: no cover - 训练环境一定装了 PTL
    class _PLCallback:  # type: ignore[no-redef]
        pass


def _format_metrics_table(names, per_class_ap, highlight_class_id=None):
    """按 YOLO train_pingpong_v2.py `_format_metrics_table` 的风格打印每类指标表。

    RF-DETR 不提供每类 AP50 / P / R, 因此相应列以 '-' 占位, 仅填 mAP(mAP50-95)。
    """
    header = f"{'cls':>16s}  {'Box P':>7s} {'Box R':>7s} {'mAP50':>7s} {'mAP':>7s}"
    lines = [header, "-" * len(header)]
    for ci in range(len(names)):
        name = names[ci]
        marker = " ★" if ci == highlight_class_id else ""
        ap = per_class_ap.get(ci)
        ap_s = f"{ap:7.3f}" if ap is not None else f"{'-':>7s}"
        lines.append(f"{name + marker:>16s}  {'-':>7s} {'-':>7s} {'-':>7s} {ap_s}")
    return "\n".join(lines)


class BestClassTracker(_PLCallback):
    """在每次验证后按指定类别 AP 追踪最优模型 (RF-DETR / PTL 版本)。

    RF-DETR 的 ``model.train()`` 不暴露 callbacks, 但 ``rfdetr.training.build_trainer``
    可被猴子补丁: 返回的 PTL Trainer 在 ``fit`` 前向其 ``callbacks`` 追加本实例。
    """

    def __init__(self, class_id, names, out_dir):
        super().__init__()
        self.class_id = int(class_id)
        self.names = list(names)
        self.class_name = (self.names[self.class_id]
                           if 0 <= self.class_id < len(self.names) else str(self.class_id))
        self.out_dir = Path(out_dir)
        self.best_ap = -1.0
        self.best_epoch = -1
        self.saved_path = None
        self.last_per_class_ap = {}

    # ---- 指标收集 ----
    @staticmethod
    def _collect_metrics(trainer):
        merged = {}
        for attr in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
            try:
                merged.update(getattr(trainer, attr, {}) or {})
            except Exception:
                continue
        return merged

    def _read_per_class_ap(self, metrics):
        per_class = {}
        for i, nm in enumerate(self.names):
            for key in (f"val/AP/{nm}", f"val/ema_AP/{nm}"):
                v = metrics.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    per_class[i] = fv
                break
        return per_class

    # ---- 保存 ----
    def _build_payload(self, trainer, pl_module):
        """尽量复刻 RF-DETR 官方 checkpoint (checkpoint_best_total.pth) 的结构。"""
        import torch
        from rfdetr.training.callbacks.best_model import BestModelCallback

        state = BestModelCallback._get_live_model_state_dict(pl_module)
        train_config = getattr(pl_module, "train_config", None)
        dataset_class_names = getattr(getattr(trainer, "datamodule", None), "class_names", None)
        if (dataset_class_names is not None and train_config is not None
                and hasattr(train_config, "model_copy")
                and getattr(train_config, "class_names", None) is None):
            train_config = train_config.model_copy(update={"class_names": dataset_class_names})
        args_dict = (train_config.model_dump()
                     if hasattr(train_config, "model_dump") else train_config)
        model_name = BestModelCallback._resolve_model_name(pl_module)
        model_config_dict = BestModelCallback._serialize_model_config(pl_module, state)
        return BestModelCallback._build_checkpoint_payload(
            state, args_dict, trainer,
            model_name=model_name, model_config_dict=model_config_dict)

    def _save(self, trainer, pl_module, dest: Path) -> bool:
        import torch
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = self._build_payload(trainer, pl_module)
            torch.save(payload, dest)
            return True
        except Exception as e:
            print(f"⚠️ best_class 模型保存失败 (fallback 复制 best_model_path): {e}")
        # fallback: 复制官方 BestModelCallback 产出的最新 checkpoint
        try:
            ckpt_cb = getattr(trainer, "checkpoint_callback", None)
            src = getattr(ckpt_cb, "best_model_path", None) if ckpt_cb else None
            if src and Path(src).exists():
                shutil.copy2(str(src), str(dest))
                return True
        except Exception as e:
            print(f"⚠️ best_class 模型复制失败: {e}")
        return False

    # ---- PTL hook ----
    def on_validation_end(self, trainer, pl_module):
        try:
            if getattr(trainer, "sanity_checking", False):
                return
        except Exception:
            pass
        if not getattr(trainer, "is_global_zero", True):
            return

        metrics = self._collect_metrics(trainer)
        per_class = self._read_per_class_ap(metrics)
        self.last_per_class_ap = per_class

        cls_ap = per_class.get(self.class_id)
        if cls_ap is None:
            return
        if cls_ap <= self.best_ap:
            return

        save_path = self.out_dir / f"best_class_{self.class_id}_ap.pth"
        if not self._save(trainer, pl_module, save_path):
            return

        self.best_ap = cls_ap
        self.best_epoch = int(getattr(trainer, "current_epoch", -1))
        self.saved_path = save_path

        print(f"\n{'=' * 70}")
        print(f"🏓 [类别{self.class_id} mAP50-95 ↑]  "
              f"epoch={self.best_epoch}  AP={cls_ap:.4f}  →  {save_path}")
        print(f"   类别: {self.class_id} ({self.class_name})")
        print(_format_metrics_table(self.names, per_class, self.class_id))
        print(f"{'=' * 70}")


def _print_final_summary(out_dir: Path, tracker):
    """对齐 YOLO `_print_final_summary` 的 '📦 所有保存模型一览' 总览。"""
    print("\n" + "=" * 100)
    print("  📦 所有保存模型一览")
    print("=" * 100)
    entries = []
    for fname in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                  "checkpoint_best_regular.pth", "checkpoint.pth"):
        entries.append((fname, out_dir / fname, None))
    if tracker is not None and tracker.best_epoch >= 0 and tracker.saved_path is not None:
        entries.append((f"best_class_{tracker.class_id}_ap.pth", tracker.saved_path, tracker))

    for label, path, info in entries:
        print()
        print("─" * 100)
        exists = path is not None and Path(path).exists()
        size_str = ""
        if exists:
            try:
                size_str = f"  [{Path(path).stat().st_size / (1024 * 1024):.1f} MB]"
            except Exception:
                pass
        status = "✅" if exists else "❌ (未找到)"
        print(f"{status} {label}{size_str}")
        print(f"    路径: {path}")
        if info is not None:
            print(f"    场景: 类别{info.class_id} mAP50-95 最高 "
                  f"(epoch={info.best_epoch}, score={info.best_ap:.4f})")
            print(_format_metrics_table(info.names, info.last_per_class_ap, info.class_id))
        else:
            print("    场景: RF-DETR 内部综合指标最优（精确指标见 metrics.csv / TensorBoard）")
    print("=" * 100)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    _warn_unsupported_args(args)

    # ---- YOLO 兼容别名映射 ----
    eff_lr = args.lr0 if args.lr0 is not None else args.lr
    if args.lr0 is not None:
        print(f"ℹ️ --lr0 已映射为 RF-DETR 学习率 {eff_lr}")
    eff_resolution = args.imgsz if args.imgsz is not None else args.resolution
    if args.imgsz is not None:
        print(f"ℹ️ --imgsz 已映射为 RF-DETR 训练分辨率 {eff_resolution}")
    eff_amp_dtype = "none" if not args.amp else args.amp_dtype
    if not args.amp:
        print("ℹ️ --no-amp 已映射为 --amp-dtype none")
    eff_freeze_encoder = bool(args.freeze_encoder or (args.freeze and args.freeze > 0))
    if args.freeze and args.freeze > 0 and not args.freeze_encoder:
        print(f"ℹ️ --freeze={args.freeze} 已映射为 RF-DETR --freeze-encoder")

    try:
        import torch
        import rfdetr
    except ModuleNotFoundError as e:
        print(f"❌ 缺少依赖: {e}\n   请在 rfdetr conda 环境中运行: "
              f"conda run -n rfdetr python src_detr/train_detr.py ...")
        sys.exit(1)

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = (_PROJECT_ROOT / data_yaml).resolve()

    # ---- 输出目录 ----
    out_dir = Path(args.output) if args.output else (_PROJECT_ROOT / "runs_detr" / args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume: YOLO 风格 (--resume 不带值) 自动挑最新 checkpoint ----
    if args.resume is True:
        preferred = [out_dir / n for n in ("checkpoint.pth", "checkpoint_best_total.pth",
                                           "last.ckpt", "checkpoint_best_regular.pth")]
        cand = next((p for p in preferred if p.exists()), None)
        if cand is None:
            all_ck = sorted(out_dir.glob("checkpoint*.pth")) + sorted(out_dir.glob("*.ckpt"))
            cand = all_ck[-1] if all_ck else None
        if cand is None:
            print(f"❌ --resume 未指定路径, 且 {out_dir} 下未找到 checkpoint")
            sys.exit(1)
        args.resume = str(cand)
        print(f"ℹ️ --resume 自动选择: {args.resume}")

    # ---- 数据集视图 ----
    if args.dataset_view == "keep":
        view_dir = _TMP_DIR / f"detr_dataset_{args.name}"
        if not (view_dir / "data.yaml").exists():
            print(f"❌ --dataset-view keep 但视图不存在: {view_dir}")
            sys.exit(1)
        names = _names_list(_read_yaml(view_dir / "data.yaml").get("names", []))
        print(f"📂 复用已存在的数据集视图: {view_dir}")
    else:
        view_dir = Path(args.dataset_view) if args.dataset_view else \
            (_TMP_DIR / f"detr_dataset_{args.name}")
        # 重建时先清掉旧的链接/文件, 避免 mklink 报 "已存在"
        if view_dir.exists() and args.dataset_view is None:
            shutil.rmtree(view_dir, ignore_errors=True)
        view_dir, names = build_dataset_view(data_yaml, view_dir)

    # ---- 模型类 ----
    detect_cls, seg_cls = _MODEL_REGISTRY[args.size]
    cls_name = seg_cls if args.task == "seg" else detect_cls
    if not hasattr(rfdetr, cls_name):
        print(f"❌ 当前 rfdetr 版本不含模型类 {cls_name}。"
              f"xlarge/2xlarge 需安装 rfdetr[plus]。")
        sys.exit(1)
    model_cls = getattr(rfdetr, cls_name)

    # ---- 设备 ----
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  未检测到 GPU, 回退 CPU")
        device = "cpu"

    # ---- 预训练权重 ----
    pretrain_weights = None if args.no_pretrain else args.pretrained  # None => 用官方默认

    print("=" * 70)
    print(f"🏓 RF-DETR-{args.task} 乒乓球训练 - 启动")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}  HIP: {getattr(torch.version, 'hip', None)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"设备    : {device}")
    print(f"数据集  : {view_dir}")
    print(f"输出    : {out_dir}")

    # ---- best_class 追踪 ----
    tracker = None
    track_class_id = args.best_class
    if track_class_id == -2:
        if "乒乓球" in names:
            track_class_id = names.index("乒乓球")
            print(f"🎯 --best-class 自动检测: 乒乓球 = class {track_class_id}")
        else:
            track_class_id = 0
            print("ℹ️  names 中未找到'乒乓球'，--best-class 回退到 class 0")
    if track_class_id is not None and track_class_id >= 0:
        if track_class_id >= len(names):
            print(f"⚠️  --best-class={track_class_id} 超出类别范围 (共 {len(names)} 类)，已禁用追踪")
            track_class_id = -1
        else:
            tracker = BestClassTracker(track_class_id, names, out_dir)
            print(f"🎯 best_class 追踪：cls={track_class_id} ({names[track_class_id]})")
            print("ℹ️ RF-DETR 无每类 AP50，跳过 best_class_*_map50")
    if tracker is None:
        print("ℹ️  best_class 已禁用")

    print(f"数据集: {view_dir}  | 类别数: {len(names)} - {names}")
    print("=" * 70)

    # ---- 构造模型 ----
    ctor_kwargs = {"device": device}
    if pretrain_weights is not None:
        ctor_kwargs["pretrain_weights"] = pretrain_weights
    if args.no_pretrain:
        ctor_kwargs["pretrain_weights"] = None
    if eff_freeze_encoder:
        ctor_kwargs["freeze_encoder"] = True
    if args.gradient_checkpointing:
        ctor_kwargs["gradient_checkpointing"] = True
    if getattr(args, "compile", False):
        ctor_kwargs["compile"] = True
    if getattr(args, "cuda_graphs", False):
        ctor_kwargs["cuda_graphs"] = True
    if getattr(args, "group_detr", None) is not None:
        ctor_kwargs["group_detr"] = int(args.group_detr)
    if getattr(args, "fused_optimizer", None) is not None:
        ctor_kwargs["fused_optimizer"] = bool(args.fused_optimizer)

    print(f"\n🧩 构造模型 {cls_name} (pretrain_weights={ctor_kwargs.get('pretrain_weights', '<官方默认>')}) ...")
    model = model_cls(**ctor_kwargs)

    # ---- 类别加权 (等价 YOLO 的 model.model.class_weights) ----
    _restore_cw = None
    _cw = float(getattr(args, "class_weight", 1.0) or 1.0)
    if _cw != 1.0 and len(names) > 0:
        if track_class_id is not None and 0 <= track_class_id < len(names):
            _ball_id = track_class_id
        elif "乒乓球" in names:
            _ball_id = names.index("乒乓球")
        else:
            _ball_id = None
        if _ball_id is None:
            print(f"⚠️  --class-weight={_cw} 但未找到可加权类别 (无 乒乓球 / best-class)，跳过加权")
        else:
            _wt = torch.ones(len(names), dtype=torch.float32)
            _wt[_ball_id] = _cw
            _restore_cw = _install_class_weight_hook(_wt)
            if _restore_cw is not None:
                print(f"⚖️  类别加权: cls={_ball_id} ({names[_ball_id]}) 权重 x{_cw:.1f} "
                      f"(等价 YOLO: 分类损失逐元素 * class_weights)")

    # ---- train() 参数 ----
    train_kwargs = dict(
        dataset_dir=str(view_dir),
        output_dir=str(out_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        grad_accum_steps=int(args.grad_accum),
        lr=float(eff_lr),
        num_workers=int(args.workers),
        device=device,
        tensorboard=bool(args.tensorboard),
        eval_interval=int(args.eval_interval),
        class_names=list(names),
        use_ema=bool(args.ema),
        multi_scale=bool(args.multi_scale),
        run_test=False,
    )
    if args.lr_encoder is not None:
        train_kwargs["lr_encoder"] = float(args.lr_encoder)
    if eff_resolution is not None:
        train_kwargs["resolution"] = int(eff_resolution)
    if args.early_stopping:
        train_kwargs["early_stopping"] = True
        train_kwargs["early_stopping_patience"] = int(args.patience)
    if args.resume:
        train_kwargs["resume"] = str(args.resume)
    if args.gpu_postprocess:
        train_kwargs["augmentation_backend"] = "kornia"

    # 仅当该版本 TrainConfig 支持时传入可选字段 (extra="forbid" 下未知字段会报错)
    try:
        from rfdetr.config import TrainConfig
        fields = set(TrainConfig.model_fields)
        if "seed" in fields:
            train_kwargs["seed"] = int(args.seed)
        # 显式走 "roboflow" 构建器 (自动探测 COCO/YOLO); 我们的视图是 YOLO 布局
        if "dataset_file" in fields:
            train_kwargs["dataset_file"] = "roboflow"
        if "amp_dtype" in fields:
            train_kwargs["amp_dtype"] = None if eff_amp_dtype == "none" else eff_amp_dtype
        if "log_per_class_metrics" in fields:
            train_kwargs["log_per_class_metrics"] = bool(args.log_per_class)
        if "progress_bar" in fields:
            train_kwargs["progress_bar"] = None if args.progress_bar == "none" else args.progress_bar
        if "eval_batch_size" in fields and args.eval_batch_size is not None:
            train_kwargs["eval_batch_size"] = int(args.eval_batch_size)
        if "eval_masks_head_resolution" in fields:
            train_kwargs["eval_masks_head_resolution"] = bool(args.eval_masks_head_resolution)
        if "eval_max_dets" in fields:
            train_kwargs["eval_max_dets"] = int(args.eval_max_dets)
        if "num_sanity_val_steps" in fields:
            train_kwargs["num_sanity_val_steps"] = int(args.num_sanity_val_steps)
        if "pin_memory" in fields:
            train_kwargs["pin_memory"] = bool(args.pin_memory)
        if "persistent_workers" in fields:
            train_kwargs["persistent_workers"] = bool(args.persistent_workers and int(args.workers) > 0)
        if "prefetch_factor" in fields and args.prefetch_factor is not None and int(args.workers) > 0:
            train_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("📋 训练配置:")
    print("=" * 70)
    for k, v in train_kwargs.items():
        print(f"  {k:20s}: {v}")

    print("\n" + "=" * 70)
    print("🚀 开始训练...")
    print("=" * 70)

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # ---- 猴子补丁 build_trainer: 向 PTL Trainer 追加 best_class 回调 ----
    # RF-DETR 的 model.train() 不暴露 callbacks; build_trainer 返回 PTL Trainer,
    # 可在 fit 前 append 自定义 Callback。训练结束后必须在 finally 中恢复原函数。
    _orig_build_trainer = None
    try:
        import rfdetr.training as _rt
        if tracker is not None:
            _orig_build_trainer = _rt.build_trainer

            def _patched_build_trainer(*a, **kw):
                _trainer = _orig_build_trainer(*a, **kw)
                try:
                    _trainer.callbacks.append(tracker)
                except Exception as e:
                    print(f"⚠️ 无法挂载 best_class 回调: {e}")
                return _trainer

            _rt.build_trainer = _patched_build_trainer
        model.train(**train_kwargs)
    finally:
        if _orig_build_trainer is not None:
            try:
                _rt.build_trainer = _orig_build_trainer
            except Exception:
                pass
        if _restore_cw is not None:
            _restore_cw()

    # ---- 峰值显存 ----
    if torch.cuda.is_available():
        try:
            peak = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n📊 训练峰值显存: {peak:.2f} GB / {total:.1f} GB")
        except Exception:
            pass

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("  ✅ 训练完成!")
    print("=" * 80)
    print(f"输出目录: {out_dir}")
    _print_final_summary(out_dir, tracker)
    print("\n推理推荐使用 checkpoint_best_total.pth:")
    print(f"  conda run -n rfdetr python src_detr/inference_video_detr.py "
          f"--model {out_dir / 'checkpoint_best_total.pth'} --video <视频>")
    print("=" * 80)

    _exit(0)


if __name__ == "__main__":
    main()
