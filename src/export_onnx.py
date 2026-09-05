#!/usr/bin/env python3
"""
YOLO .pt → .onnx 导出脚本 (支持单文件 / 目录批量)
==================================================
用 ultralytics 加载训练好的 .pt 模型并导出为 ONNX，便于 ONNX Runtime / OpenVINO
等部署框架推理。支持 FP16 / 动态输入尺寸 / 简化 / NMS 内嵌 等导出选项。
支持两种输入模式:
  - 单文件:        --model <xx.pt>
  - 目录批量:      --model-dir <目录>  (递归扫描 *.pt, 可用 --pattern / --no-recurse 控制)

用法 (PowerShell, conda 环境 yolo26):
    # 单文件
    conda run -n yolo26 python src/export_onnx.py --model models/m960n.pt `
        --imgsz 960 --half --dynamic --simplify --nms

    # 目录批量 (递归扫描所有 *.pt)
    conda run -n yolo26 python src/export_onnx.py --model-dir models `
        --half --dynamic --simplify

    # 目录批量, 只转 best*.pt, 不递归子目录, 输出到指定目录
    conda run -n yolo26 python src/export_onnx.py --model-dir runs `
        --pattern "best*.pt" --no-recurse --output inference_output/

说明:
  - 首次导出可能自动安装 onnx / onnxslim（ultralytics 自带 auto-install）。
  - YOLO26 默认是 NMS-free (end2end) 模型；--nms 会把后处理 NMS 内嵌进模型，
    便于不支持端到端的推理框架直接使用。
  - 批量模式下单个模型失败不影响其余, 末尾统一汇总成功/失败列表。
"""

# ---------------------------------------------------------------------------
# MIOpen / HIP 环境变量 (必须在 import torch 之前!)
# ---------------------------------------------------------------------------
import os
import sys
import shutil
import platform


_IS_WINDOWS = (platform.system() == "Windows")

if _IS_WINDOWS:
    _DEF_MIOPEN_CACHE = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "miopen"
    )
else:
    _DEF_MIOPEN_CACHE = os.path.expanduser("~/.cache/miopen")
os.makedirs(_DEF_MIOPEN_CACHE, exist_ok=True)
os.environ.setdefault("MIOPEN_USER_DB_PATH",       _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR",   _DEF_MIOPEN_CACHE)
os.environ.setdefault("MIOPEN_FIND_MODE",          "FAST")
os.environ.setdefault("MIOPEN_FIND_ENFORCE",       "NONE")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF",    "expandable_segments:True")

from ultralytics import YOLO
import torch
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _PROJECT_ROOT / "tmp"
_TMP_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="将 YOLO .pt 模型导出为 .onnx (ultralytics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入: --model (单文件) 与 --model-dir (目录批量) 互斥, 至少给一个
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--model", "-m", type=str, default=None,
                       help="待转换的 .pt 权重路径 (单文件模式)")
    group.add_argument("--model-dir", type=str, default=None,
                       help="扫描目录下所有 *.pt 批量转换 (批量模式)")
    parser.add_argument("--pattern", type=str, default="*.pt",
                        help="批量模式下的文件过滤 pattern (如 '*.pt' / 'best*.pt')")
    parser.add_argument("--no-recurse", dest="recurse", action="store_false", default=True,
                        help="批量模式: 不递归子目录 (默认递归)")

    parser.add_argument("--output", "-o", type=str, default=None,
                        help="导出路径。单文件: 具体 .onnx 路径或目录; "
                             "批量: 输出目录。默认: 与各 --model 同目录、同名、扩展名 .onnx")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="导出模型的推理分辨率 (输入尺寸)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset 版本 (YOLO26 推荐 17-19)")
    parser.add_argument("--half", action="store_true",
                        help="导出 FP16 模型 (GPU 部署加速; CPU 需 FP32)")
    parser.add_argument("--dynamic", action="store_true",
                        help="导出动态 batch/输入尺寸 (部署时更灵活)")
    parser.add_argument("--simplify", action="store_true",
                        help="用 onnxslim 简化模型 (清理冗余算子, 需联网/安装 onnxslim)")
    parser.add_argument("--nms", action="store_true",
                        help="内嵌 NMS 后处理到模型 (对非端到端框架友好)")
    parser.add_argument("--batch", type=int, default=1,
                        help="静态 batch 大小 (仅 --dynamic 未开启时生效)")
    parser.add_argument("--device", type=str, default="0",
                        help="导出设备: '0'=GPU(ROCm/CUDA), 'cpu'=CPU")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ONNX 结果打印 (可选, onnx 未安装时静默跳过)
# ---------------------------------------------------------------------------
def _inspect_onnx(onnx_path: str):
    try:
        import onnx
    except ImportError:
        print("ℹ️  (可选) 未安装 onnx 包, 跳过模型结构校验。"
              "可用 `pip install onnx` 后重跑查看张量信息。")
        return
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("\n📐 ONNX 模型结构校验通过:")
        for inp in model.graph.input:
            print(f"   输入 : {inp.name}  shape={[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")
        for out in model.graph.output:
            print(f"   输出 : {out.name}  shape={[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")
        n_ops = len(model.graph.node)
        print(f"   算子节点数: {n_ops}")
    except Exception as e:
        print(f"⚠️  ONNX 校验异常 (模型可能已导出, 但结构检查失败): {e}")


# ---------------------------------------------------------------------------
# 解析 --output 是"文件"还是"目录" → 得到某模型的最终 .onnx 路径
# ---------------------------------------------------------------------------
def _resolve_out_path(model_path: Path, args_output) -> Path:
    if not args_output:
        return model_path.with_suffix(".onnx")
    out = Path(args_output)
    is_dir = (str(args_output).endswith(("/", "\\"))
              or out.is_dir()
              or out.suffix == "")
    if is_dir:
        return out / (model_path.stem + ".onnx")
    return out


# ---------------------------------------------------------------------------
# 导出单个模型; 返回 (是否成功, 实际输出的 .onnx 路径 或 None)
# ---------------------------------------------------------------------------
def export_one(model_path: Path, out_path: Path, args, device: str,
               verbose: bool = True):
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        return False, None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"❌ 加载模型失败 [{model_path}]: {e}")
        return False, None

    if verbose:
        task = getattr(model, "task", "?")
        try:
            nc = len(model.names)
            names = list(model.names.values())
            print(f"模型任务: {task}  | 类别数: {nc}  | {names}")
        except Exception:
            pass
        try:
            n_params = sum(p.numel() for p in model.model.parameters())
            print(f"参数量  : {n_params / 1e6:.1f}M")
        except Exception:
            pass

    export_args = {
        "format":   "onnx",
        "imgsz":    args.imgsz,
        "opset":    args.opset,
        "half":     args.half,
        "dynamic":  args.dynamic,
        "simplify": args.simplify,
        "nms":      args.nms,
        "device":   device,
        "batch":    args.batch,
    }

    print("🚀 开始导出... (首次可能自动安装 onnx/onnxslim, 请稍候)")
    try:
        exported = model.export(**export_args)
    except Exception as e:
        print(f"❌ 导出失败 [{model_path.name}]: {e}")
        import traceback
        traceback.print_exc()
        return False, None

    # model.export 返回导出的文件路径; 若与我们预期不同则以实际为准
    actual = Path(exported) if exported else out_path
    if actual.suffix.lower() != ".onnx":
        actual = out_path
    # 若模型落到了别的路径 (ultralytics 基于模型名命名), 而我们期望 out_path,
    # 始终把产物移动/复制到 out_path (覆盖旧文件, 兼容 --output 已存在的重复运行)
    if os.path.abspath(str(actual)) != os.path.abspath(str(out_path)):
        try:
            os.replace(str(actual), str(out_path))
            actual = out_path
        except OSError:
            try:
                shutil.copy2(str(actual), str(out_path))
                actual = out_path
            except Exception:
                pass  # 移动失败则保留 ultralytics 原始输出位置

    if not actual.exists():
        print(f"⚠️ 未找到输出文件: {actual}")
        return False, actual

    size_mb = actual.stat().st_size / (1024 * 1024)
    print(f"✅ 导出完成 → {actual}  ({size_mb:.1f} MB)")
    if verbose:
        _inspect_onnx(str(actual))
        print("\n💡 下一步: 用 onnxruntime 校验推理结果, 例如:")
        print(f"   import onnxruntime as ort; ort.InferenceSession(str(r'{actual}'))")
    return True, actual


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if not args.model and not args.model_dir:
        print("❌ 请提供输入: --model <xx.pt> 或 --model-dir <目录> (二者互斥, 至少一个)")
        sys.exit(1)

    # ---- 设备策略 (只解析一次) ----
    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        print("⚠️ GPU 不可用 (ROCm/CUDA), 回退到 CPU")
        device = "cpu"
    if device == "cpu" and args.half:
        print("⚠️  CPU 不支持 FP16, 已自动关闭 --half")
        args.half = False

    print("=" * 62)
    print(f"🤖 YOLO .pt → .onnx 导出  [Platform: {platform.system()}]")
    print("=" * 62)
    print(f"设备    : {device}  | imgsz: {args.imgsz}  | opset: {args.opset}")
    print(f"FP16    : {args.half}  | dynamic: {args.dynamic}  | simplify: {args.simplify}  | nms: {args.nms}")

    # ====================== 目录批量模式 ======================
    if args.model_dir:
        base = Path(args.model_dir)
        if not base.is_dir():
            print(f"❌ --model-dir 不是有效目录: {base}")
            sys.exit(1)
        if args.recurse:
            files = sorted(base.rglob(args.pattern))
        else:
            files = sorted(base.glob(args.pattern))
        # 过滤掉目录
        files = [f for f in files if f.is_file()]
        if not files:
            print(f"❌ 目录中未找到匹配 '{args.pattern}' 的 .pt 文件: {base}")
            sys.exit(1)

        print(f"\n📁 目录批量: 共 {len(files)} 个模型")
        success, failed = 0, []
        for i, fp in enumerate(files, 1):
            print("\n" + "-" * 62)
            print(f"[{i}/{len(files)}] 转换: {fp}")
            o = _resolve_out_path(fp, args.output)
            ok, _ = export_one(fp, o, args, device, verbose=False)
            if ok:
                success += 1
            else:
                failed.append(str(fp))
            if device != "cpu":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        print("\n" + "=" * 62)
        print(f"📊 批量完成: ✅ 成功 {success}  | ❌ 失败 {len(failed)} / 共 {len(files)}")
        if failed:
            print("失败列表:")
            for f in failed:
                print(f"  - {f}")
        print("=" * 62)
        sys.exit(1 if failed else 0)

    # ====================== 单文件模式 ======================
    model_path = Path(args.model)
    out_path = _resolve_out_path(model_path, args.output)
    print(f"模型    : {model_path}")
    print(f"导出到  : {out_path}\n")
    ok, _ = export_one(model_path, out_path, args, device, verbose=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
