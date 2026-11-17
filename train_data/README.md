# train_data —— 本地数据集目录

这个目录**只保留结构，内容不入版本库**（数据集动辄数十 GB，不适合放 Git）。

## 需要放什么

```
train_data/
└── <数据集名>/
    ├── images/
    │   ├── train/   *.jpg / *.png
    │   └── val/     *.jpg / *.png
    ├── labels/
    │   ├── train/   *.txt（与 images 同名）
    │   └── val/     *.txt
    └── data.yaml
```

`data.yaml` 示例：

```yaml
path: ./train_data/<数据集名>     # 必须指向 images/ labels/ 的父目录
train: images/train
val: images/val
nc: 6
names: ['乒乓球', '其他比赛球', '球员', '球拍', '球桌', '球网']
```

## 标注格式

YOLO 格式，每行一个目标：

```
class_id x_center y_center width height      # 归一化到 [0,1]
```

也支持 polygon（分割）标注，每行 9+ 个值：

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

`src/train_pingpong_v2.py` 会自动把 bbox 转成 polygon。

## 怎么获得数据

| 方式 | 脚本 |
|---|---|
| 从检测日志挑漏检帧 | `src/extract_missing_ball.py` |
| 收集误检帧当负样本 | `src/extract_rejected_ball_frames.py` |
| 随机抽帧 | `src/random_sample_frames.py` |
| Label Studio 导出 | `src/convert_yolo_to_labelstudio.py`（反向转换） |

## 训练前

```powershell
# 必须拆成 train/val，否则训练会静默失败
conda run -n yolo26 python src/split_all1.py `
    --src train_data/<数据集名> --dst train_data/<数据集名>_split
```
