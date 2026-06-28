"""
Convert YOLO format dataset to Label Studio importable format.

Usage:
    conda run -n yolo26 python src/convert_yolo_to_labelstudio.py
        --src train_data/all12
        --dst train_data/all12_labelstudio

Output:
    dst/
    ├── images/                    # 复制原图
    ├── labels/                    # 复制 YOLO 标签（原样保留）
    ├── classes.txt                # 修正为 0-indexed 一行一类名格式
    ├── data.yaml                  # YOLO data.yaml
    └── labelstudio_tasks.json     # ← Label Studio 直接导入的标注文件
"""

import json
import os
import sys
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO → Label Studio 格式转换")
    parser.add_argument("--src", default=r"train_data/all12",
                        help="源数据集目录（含 images/ labels/ classes.txt）")
    parser.add_argument("--dst", default=r"train_data/all12_labelstudio",
                        help="输出目录")
    parser.add_argument("--classes", default=None,
                        help="classes.txt 路径（默认取 src/classes.txt）")
    return parser.parse_args()


def parse_yolo_classes(classes_path: Path):
    """解析 classes.txt，返回 [class_name, ...] 列表（0-indexed）"""
    lines = []
    with open(classes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)

    # 检测格式：普通列表 or "id: name"
    if lines and ":" in lines[0]:
        # "1: 乒乓球" → 提取 name，按 id 排序
        pairs = []
        for line in lines:
            parts = line.split(":", 1)
            if len(parts) == 2:
                idx = int(parts[0].strip())
                name = parts[1].strip()
                pairs.append((idx, name))
        pairs.sort(key=lambda x: x[0])
        return [name for _, name in pairs]
    else:
        # 普通列表，0-indexed
        return lines


def get_image_size(img_path: Path):
    """获取图片宽高"""
    from PIL import Image
    img = Image.open(img_path)
    return img.size  # (width, height)


def yolo_bbox_to_ls_rect(cls_id, x_center, y_center, w, h, img_w, img_h, class_names):
    """YOLO bbox → Label Studio rectanglelabels"""
    # YOLO: normalized center+size → LS: percentage left/top/width/height
    left_pct = (x_center - w / 2) * 100
    top_pct = (y_center - h / 2) * 100
    width_pct = w * 100
    height_pct = h * 100

    return {
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": left_pct,
            "y": top_pct,
            "width": width_pct,
            "height": height_pct,
            "rotation": 0,
            "rectanglelabels": [class_names[cls_id]],
        },
        "type": "rectanglelabels",
    }


def yolo_polygon_to_ls_polygon(cls_id, points, img_w, img_h, class_names):
    """YOLO polygon → Label Studio polygonlabels"""
    # YOLO: [(x1,y1), (x2,y2), ...] normalized [0,1]
    # LS: [[x1,y1], [x2,y2], ...] in percentage (0-100)
    ls_points = [[x * 100, y * 100] for x, y in points]

    return {
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "points": ls_points,
            "polygonlabels": [class_names[cls_id]],
        },
        "type": "polygonlabels",
    }


def parse_yolo_label(label_path: Path):
    """解析 YOLO .txt 标签，返回 list of (class_id, format, data)

    Returns:
        list of (cls_id: int, fmt: str, data)
        - fmt="bbox":  data = (x_center, y_center, w, h)
        - fmt="polygon": data = [(x1,y1), (x2,y2), ...]
    """
    results = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [float(x) for x in line.strip().split()]
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            coords = parts[1:]

            if len(coords) == 4:
                # bbox: x_center y_center width height
                results.append((cls_id, "bbox", tuple(coords)))
            else:
                # polygon: x1 y1 x2 y2 ... xn yn (must be even length)
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                    results.append((cls_id, "polygon", points))

    return results


def main():
    args = parse_args()
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    classes_path = Path(args.classes) if args.classes else src_dir / "classes.txt"

    if not src_dir.exists():
        print(f"❌ 源目录不存在: {src_dir}")
        sys.exit(1)
    if not classes_path.exists():
        print(f"❌ classes.txt 不存在: {classes_path}")
        sys.exit(1)

    # ---- 1. 读取 classes ----
    class_names = parse_yolo_classes(classes_path)
    print(f"📋 类别 ({len(class_names)}): {class_names}")

    # ---- 2. 创建目标目录 ----
    dst_images = dst_dir / "images"
    dst_labels = dst_dir / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    # ---- 3. 扫描图片文件 ----
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_files = sorted([
        f for f in (src_dir / "images").iterdir()
        if f.suffix.lower() in img_exts
    ])
    print(f"🖼️  图片: {len(img_files)} 张")
    print(f"📁 输出目录: {dst_dir}")

    # ---- 4. 复制 images + labels ----
    print("🔄 复制 images/ 和 labels/ ...")
    for img_path in img_files:
        shutil.copy2(img_path, dst_images / img_path.name)

    src_labels = src_dir / "labels"
    if src_labels.exists():
        for lbl_path in src_labels.iterdir():
            if lbl_path.suffix.lower() == ".txt":
                shutil.copy2(lbl_path, dst_labels / lbl_path.name)

    # ---- 5. 写入 classes.txt（修正格式：0-indexed 一行一类）----
    with open(dst_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✅ classes.txt 已修正（{len(class_names)} 类，0-indexed）")

    # ---- 6. 写入 data.yaml ----
    yaml_content = f"""# Label Studio / YOLO 数据集配置
path: {dst_dir.resolve().as_posix()}
train: images/
val: images/

nc: {len(class_names)}
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(dst_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content.strip() + "\n")
    print(f"✅ data.yaml 已生成")

    # ---- 7. 生成 Label Studio JSON 任务文件（含标注）----
    print("🔄 转换标注为 Label Studio JSON 格式...")
    tasks = []
    failed = 0

    for img_path in img_files:
        label_path = src_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            tasks.append({
                "data": {"image": f"images/{img_path.name}"},
                "annotations": [{"result": []}],
            })
            continue

        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            print(f"  ⚠️ 无法读取图片尺寸: {img_path.name} — {e}，跳过")
            failed += 1
            continue

        annotations = parse_yolo_label(label_path)
        results = []

        for cls_id, fmt, data in annotations:
            if cls_id < 0 or cls_id >= len(class_names):
                continue
            if fmt == "bbox":
                x_center, y_center, w, h = data
                results.append(
                    yolo_bbox_to_ls_rect(cls_id, x_center, y_center, w, h, img_w, img_h, class_names)
                )
            elif fmt == "polygon":
                results.append(
                    yolo_polygon_to_ls_polygon(cls_id, data, img_w, img_h, class_names)
                )

        tasks.append({
            "data": {"image": f"images/{img_path.name}"},
            "annotations": [{"result": results}],
        })

    # ---- 8. 写入 JSON ----
    json_path = dst_dir / "labelstudio_tasks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"✅ labelstudio_tasks.json 已生成（{len(tasks)} 个任务）")
    if failed:
        print(f"  ⚠️  失败: {failed} 张")

    # ---- 9. 统计摘要 ----
    bbox_count = 0
    polygon_count = 0
    for task in tasks:
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r["type"] == "rectanglelabels":
                    bbox_count += 1
                elif r["type"] == "polygonlabels":
                    polygon_count += 1

    print(f"\n{'='*50}")
    print(f"  🎯 转换完成！")
    print(f"  📂 输出: {dst_dir.resolve()}")
    print(f"  🖼️  图片: {len(tasks)} 张")
    print(f"  📦 矩形框: {bbox_count} 个")
    print(f"  📦 多边形: {polygon_count} 个")
    print(f"\n  📄 导入 Label Studio 方式：")
    print(f"    方式 A — 直接导入 JSON：")
    print(f"      打开 Label Studio → Import → 上传 labelstudio_tasks.json")
    print(f"    方式 B — YOLO 格式导入：")
    print(f"      打开 Label Studio → Import → YOLO → 选择 {dst_dir} 目录")
    print(f"      （需要 data.yaml 或 classes.txt）")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
