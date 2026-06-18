#!/usr/bin/env python3
"""
将数据集拆分为 train/val，并生成新的 data.yaml。

用法:
    conda run -n yolo26 python split_all1.py
    conda run -n yolo26 python split_all1.py --src train_data/all5 --dst train_data/all5_split
    conda run -n yolo26 python split_all1.py --src /path/to/dataset --dst /path/to/output --ratio 0.8 --seed 42
"""

import os, shutil, random, argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="将数据集拆分为 train/val")
    parser.add_argument("--src", type=str, default=None,
                        help="源数据集路径（包含 images/ 和 labels/，默认: <脚本目录>/train_data/all5）")
    parser.add_argument("--dst", type=str, default=None,
                        help="输出路径（拆分后的数据集，默认: <脚本目录>/train_data/all5_split）")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="训练集比例，默认 0.8")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认 42")
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # src/ 的父目录 = 项目根
    if args.src is None:
        args.src = str(PROJECT_ROOT / "train_data" / "all5")
    if args.dst is None:
        args.dst = str(PROJECT_ROOT / "train_data" / "all5_split")

    return args


def main():
    args = parse_args()
    random.seed(args.seed)

    SRC = str(Path(args.src).resolve())
    DST = str(Path(args.dst).resolve())
    TRAIN_RATIO = args.ratio

    # 获取所有图片（排除 Zone.Identifier）
    img_dir = Path(SRC) / "images"
    img_files = sorted([
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() == '.jpg' and ':Zone' not in f.name
    ])

    print(f"总计 {len(img_files)} 张图片")

    # 随机打乱并拆分
    random.shuffle(img_files)
    n_train = int(len(img_files) * TRAIN_RATIO)
    train_imgs, val_imgs = img_files[:n_train], img_files[n_train:]

    print(f"训练集: {len(train_imgs)}, 验证集: {len(val_imgs)}")

    # 创建目录结构
    for split in ["train", "val"]:
        (Path(DST) / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (Path(DST) / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    # 复制图片和标签
    src_labels = Path(SRC) / "labels"
    for img_name in train_imgs:
        shutil.copy2(img_dir / img_name, Path(DST) / f"images/train/{img_name}")
        base = Path(img_name).stem
        lbl = src_labels / f"{base}.txt"
        if lbl.exists():
            shutil.copy2(lbl, Path(DST) / f"labels/train/{base}.txt")

    for img_name in val_imgs:
        shutil.copy2(img_dir / img_name, Path(DST) / f"images/val/{img_name}")
        base = Path(img_name).stem
        lbl = src_labels / f"{base}.txt"
        if lbl.exists():
            shutil.copy2(lbl, Path(DST) / f"labels/val/{base}.txt")

    # 统计标注率
    train_labeled = len(list((Path(DST) / "labels/train").glob("*.txt")))
    val_labeled = len(list((Path(DST) / "labels/val").glob("*.txt")))
    print(f"\n标注情况:")
    print(f"  Train: {train_labeled}/{len(train_imgs)} ({train_labeled*100//len(train_imgs)}%)")
    print(f"  Val:   {val_labeled}/{len(val_imgs)} ({val_labeled*100//len(val_imgs)}%)")

    # 从 classes.txt 读取类别（自动适应任意数量）
    classes_file = Path(SRC) / "classes.txt"
    if classes_file.exists():
        class_names = [line.strip() for line in classes_file.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    else:
        class_names = ["player", "pingpangball", "racket", "table", "net"]  # 回退默认
    nc = len(class_names)
    names_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))

    # 生成新的 data.yaml
    data_yaml = f"""path: {DST}
train: images/train
val: images/val

nc: {nc}
names:
{names_lines}
"""
    with open(Path(DST) / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml)

    print(f"\n=== 拆分完成: {DST} ===")
    print("已生成 data.yaml")


if __name__ == "__main__":
    main()
