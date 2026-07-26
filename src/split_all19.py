#!/usr/bin/env python3
"""
按 all18_split 的分组拆分 all19：
  - all18_split 中已有的图：保持原 train/val 分组不变
  - all19 新增的图：随机分到 train/val (seed=42, ratio=0.8)

用法:
    python src/split_all19.py
    python src/split_all19.py --src train_data/all19 --from-split train_data/all18_split --dst train_data/all19_split
"""

import os, shutil, random, argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="保持 all18 分组拆分 all19")
    parser.add_argument("--src", type=str, default="train_data/all19",
                        help="源数据集 (包含 images/ labels/ classes.txt)")
    parser.add_argument("--from-split", type=str, default="train_data/all18_split",
                        help="参考分组 (images/train/ images/val/)")
    parser.add_argument("--dst", type=str, default="train_data/all19_split",
                        help="输出路径")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="新增图片的 train 比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    SRC     = Path(args.src).resolve()
    FROM    = Path(args.from_split).resolve()
    DST     = Path(args.dst).resolve()

    # ── 1. 读取参考分组 ──
    old_train = {f.name for f in (FROM / "images" / "train").glob("*.jpg")}
    old_val   = {f.name for f in (FROM / "images" / "val").glob("*.jpg")}
    old_all   = old_train | old_val
    print(f"[1/4] 参考分组: 训练 {len(old_train)} / 验证 {len(old_val)} / 共 {len(old_all)}")

    # ── 2. 读取 all19 全部图片 ──
    src_images = SRC / "images"
    src_labels = SRC / "labels"
    all_imgs = sorted([f.name for f in src_images.iterdir()
                       if f.suffix.lower() == '.jpg' and ':Zone' not in f.name])
    print(f"[2/4] all19 总图数: {len(all_imgs)}")

    existing = [f for f in all_imgs if f in old_all]
    new_imgs = [f for f in all_imgs if f not in old_all]
    print(f"       已有: {len(existing)}, 新增: {len(new_imgs)}")

    # ── 3. 分配 ──
    # 已有图: 沿用 all18 分组
    # 新增图: 随机分
    random.shuffle(new_imgs)
    n_new_train = int(len(new_imgs) * args.ratio)
    new_train = new_imgs[:n_new_train]
    new_val   = new_imgs[n_new_train:]

    train_set = list(old_train) + new_train
    val_set   = list(old_val)   + new_val

    print(f"[3/4] 分配完成:")
    print(f"       训练集: {len(train_set)} (已有 {len(old_train)} + 新增 {len(new_train)})")
    print(f"       验证集: {len(val_set)}   (已有 {len(old_val)} + 新增 {len(new_val)})")

    # ── 4. 复制文件 ──
    for split_name, file_list in [("train", train_set), ("val", val_set)]:
        img_out = DST / "images" / split_name
        lbl_out = DST / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for fname in file_list:
            shutil.copy2(src_images / fname, img_out / fname)
            lbl = src_labels / Path(fname).with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, lbl_out / Path(fname).with_suffix(".txt").name)

    # ── 5. 生成 data.yaml ──
    classes_file = SRC / "classes.txt"
    if classes_file.exists():
        class_names = [l.strip() for l in classes_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    else:
        class_names = ["player", "pingpangball", "racket", "table", "net"]
    nc = len(class_names)
    names_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))

    yaml_content = f"""path: {DST}
train: images/train
val: images/val

nc: {nc}
names:
{names_lines}
"""
    (DST / "data.yaml").write_text(yaml_content, encoding="utf-8")

    # ── 统计标注率 ──
    train_labeled = len(list((DST / "labels" / "train").glob("*.txt")))
    val_labeled   = len(list((DST / "labels" / "val").glob("*.txt")))
    print(f"\n标注情况:")
    print(f"  Train: {train_labeled}/{len(train_set)} ({train_labeled*100//max(len(train_set),1)}%)")
    print(f"  Val:   {val_labeled}/{len(val_set)} ({val_labeled*100//max(len(val_set),1)}%)")
    print(f"\n=== 拆分完成: {DST} ===")


if __name__ == "__main__":
    main()
