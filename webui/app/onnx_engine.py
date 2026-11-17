#!/usr/bin/env python3
"""
ONNX Runtime 推理引擎（YOLO26-seg / DirectML + CPU 回落）
=========================================================
把 ultralytics 的推理链路替换为纯 onnxruntime，输出与
`src/inference_video.py` 完全一致的 jsonl 结构：

    {"frame", "timestamp", "counts", "ball_pos",
     "detections": [{"class", "confidence", "bbox", "polygon"}]}

设计约束:
  - 模型是 fp16 / 稠密输出（导出时不带 --nms），NMS 在这里用 numpy 做，
    这样能按类别设不同 conf 阈值（对比内嵌 NMS 会把 0.25 烧死）。
  - batch 固定为 1（导出时定死；实测 batch>1 更慢）。
  - polygon 只对 table/net 计算（与现有实现一致，其余类别 polygon=None）。

用法:
    from onnx_engine import OnnxDetector, DetectorConfig

    det = OnnxDetector("webui/models/m960n_fp16_nonms.onnx")
    rec = det.detect(frame, frame_idx=0, fps=30.0)
    print(rec["detections"])
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

# 模型默认类别表（onnx 元数据缺失时的兜底）
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "乒乓球", 1: "其他比赛球", 2: "球员", 3: "球拍", 4: "球桌", 5: "球网",
}

# ---------------------------------------------------------------------------
# 类别别名: 模型里的中文类名 -> 现有 jsonl 使用的逻辑名
# (与 src/inference_video.py 的 CLASS_CONFIG_TEMPLATE 保持一致)
# ---------------------------------------------------------------------------
CLASS_ALIASES: dict[str, str] = {
    "pingpangball": "pingpangball",
    "pingpongball": "pingpangball",
    "ppball": "pingpangball",
    "乒乓球": "pingpangball",
    "otherball": "otherball",
    "其他比赛球": "otherball",
    "player": "player",
    "players": "player",
    "球员": "player",
    "racket": "racket",
    "paddel": "racket",
    "paddle": "racket",
    "bat": "racket",
    "球拍": "racket",
    "table": "table",
    "pingpang table": "table",
    "pingpong table": "table",
    "球桌": "table",
    "net": "net",
    "pingpang net": "net",
    "pingpong net": "net",
    "球网": "net",
}

BALL_LOGIC_NAME = "pingpangball"
POLYGON_NEEDED = {"table", "net"}


@dataclass
class DetectorConfig:
    """推理参数。默认值对齐 src/inference_video.py 的默认行为。"""

    imgsz: int = 960
    iou: float = 0.45
    max_det: int = 300
    # 每类置信度阈值（逻辑名）。对应 CLI 的 --conf-ball / --conf-player / ...
    conf_ball: float = 0.10
    conf_player: float | None = None
    conf_racket: float | None = None
    conf_table: float | None = None
    conf_net: float | None = None
    # 未显式指定的类别走这里（与 CLASS_CONFIG_TEMPLATE 的 conf 一致）
    default_conf: dict[str, float] = field(default_factory=lambda: {
        "pingpangball": 0.25,
        "otherball": 0.40,
        "player": 0.40,
        "racket": 0.30,
        "table": 0.30,
        "net": 0.30,
    })
    want_polygon: bool = True
    polygon_thr: float = 0.5
    # 模型输入是 fp16 时，DirectML 需要 fp16 输入
    prefer_ep: str = "auto"          # auto | dml | cpu


class OnnxDetector:
    """YOLO26-seg 的 onnxruntime 实现（含预处理 / NMS / mask->polygon）。"""

    def __init__(self, model_path: str, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.model_path = model_path

        avail = ort.get_available_providers()
        if self.cfg.prefer_ep == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "DmlExecutionProvider" in avail:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        t0 = time.perf_counter()
        self.sess = ort.InferenceSession(model_path, so, providers=providers)
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.providers = self.sess.get_providers()

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        self.batch = shape[0]
        if self.batch != 1:
            raise ValueError(
                f"模型 batch={self.batch}，本引擎只支持 batch=1（请用 --batch 1 重新导出）")
        self.imgsz = shape[2] if len(shape) >= 4 else self.cfg.imgsz
        self.cfg.imgsz = self.imgsz

        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.raw_names = self._load_class_names()
        self.logic_names = {
            cid: CLASS_ALIASES.get(str(name).lower().strip(), str(name))
            for cid, name in self.raw_names.items()
        }

    # ------------------------------------------------------------------
    # 类别表: 优先读 onnx 元数据里的 names, 读不到则用默认 6 类
    # ------------------------------------------------------------------
    def _load_class_names(self) -> dict[int, str]:
        try:
            meta = self.sess.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if raw:
                d = ast.literal_eval(raw)
                return {int(k): str(v) for k, v in d.items()}
        except Exception:
            pass
        return dict(DEFAULT_CLASS_NAMES)

    # ------------------------------------------------------------------
    # 预处理
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray):
        """letterbox 到 (imgsz, imgsz)，返回 (tensor, scale, pad_x, pad_y)。

        用 cv2.dnn.blobFromImage 一次完成 BGR->RGB + HWC->CHW + /255，
        避免 numpy 里连续 4 次全图拷贝（实测从 14ms 降到 ~3ms）。
        """
        h, w = frame.shape[:2]
        size = self.imgsz
        scale = min(size / h, size / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        if nw == size and nh == size:
            canvas = frame
            pad_x = pad_y = 0
        else:
            canvas = np.full((size, size, 3), 114, dtype=np.uint8)
            pad_x = (size - nw) // 2
            pad_y = (size - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(
                frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (size, size),
                                     swapRB=True, crop=False,
                                     ddepth=cv2.CV_32F)
        if self.input_dtype == np.float16:
            blob = blob.astype(np.float16)
        return blob, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def infer(self, tensor: np.ndarray):
        outs = self.sess.run(self.out_names, {self.input_name: tensor})
        return outs[0], (outs[1] if len(outs) > 1 else None)

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def _conf_threshold(self, logic_name: str, cls_id: int) -> float:
        explicit = {
            "pingpangball": self.cfg.conf_ball,
            "player": self.cfg.conf_player,
            "racket": self.cfg.conf_racket,
            "table": self.cfg.conf_table,
            "net": self.cfg.conf_net,
        }.get(logic_name)
        if explicit is not None:
            return explicit
        return self.cfg.default_conf.get(logic_name, 0.25)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def _mask_to_polygon(self, coef: np.ndarray, proto_flat: np.ndarray,
                         proto_hw: tuple[int, int], box: np.ndarray,
                         scale: float, pad_x: int, pad_y: int,
                         frame_w: int, frame_h: int):
        """由 mask 系数 + proto 恢复分割轮廓（letterbox 空间 -> 原图坐标）。

        关键优化: 先在 proto 分辨率 (240) 上裁剪 bbox 区域，再插值到 bbox 大小，
        而不是把整张 mask 上采样到 960 再裁 —— 像素量差 10 倍以上。
        """
        mh, mw = proto_hw
        z = coef.astype(np.float32) @ proto_flat
        m = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        m = m.reshape(mh, mw)

        size = self.imgsz
        sx, sy = mw / size, mh / size
        x1 = int(max(0, np.floor(box[0]) - 1))
        y1 = int(max(0, np.floor(box[1]) - 1))
        x2 = int(min(size, np.ceil(box[2]) + 1))
        y2 = int(min(size, np.ceil(box[3]) + 1))
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            return None

        px1 = int(max(0, np.floor(x1 * sx)))
        py1 = int(max(0, np.floor(y1 * sy)))
        px2 = int(min(mw, np.ceil(x2 * sx)))
        py2 = int(min(mh, np.ceil(y2 * sy)))
        if px2 - px1 < 2 or py2 - py1 < 2:
            return None

        crop = cv2.resize(m[py1:py2, px1:px2], (bw, bh),
                          interpolation=cv2.INTER_LINEAR)
        binary = (crop > self.cfg.polygon_thr).astype(np.uint8)
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return None
        # proto 只有 240×240, 球台远端细边这类结构容易断成多块; 只取最大块会丢边,
        # 导致拟合出的四边形偏小 (实测球台 polygon 上半部分缺 46px)。
        # 下游用 cv2.convexHull 拟合, 点序无关, 所以把面积达标的块全部拼接。
        areas = [cv2.contourArea(c) for c in contours]
        amax = max(areas) if areas else 0.0
        thr_area = max(4.0, amax * 0.05)
        pieces = [c.reshape(-1, 2) for c, a in zip(contours, areas)
                  if a >= thr_area and len(c) >= 3]
        if not pieces:
            return None
        pts = np.concatenate(pieces).astype(np.float64)
        pts[:, 0] += x1
        pts[:, 1] += y1

        # letterbox -> 原图
        pts[:, 0] = (pts[:, 0] - pad_x) / scale
        pts[:, 1] = (pts[:, 1] - pad_y) / scale
        pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)
        return np.round(pts, 1).tolist()

    def postprocess(self, out0: np.ndarray, proto: np.ndarray | None,
                    frame_shape: tuple[int, int],
                    scale: float, pad_x: int, pad_y: int):
        """返回 (detections, counts, ball_pos, timing_ms)。"""
        t_nms = time.perf_counter()
        frame_h, frame_w = frame_shape[:2]
        proto_flat = None
        proto_hw = (0, 0)
        if proto is not None and self.cfg.want_polygon:
            if proto.ndim == 4:
                proto = proto[0]               # [1,32,240,240] -> [32,240,240]
            # 每帧只转换一次 (7.4 MB), 而不是每个检测转一次
            proto_flat = proto.astype(np.float32).reshape(proto.shape[0], -1)
            proto_hw = (proto.shape[1], proto.shape[2])
        pred = out0[0]                      # [300, 4+1+1+32]
        boxes_lb = pred[:, :4]
        conf = pred[:, 4]
        cls_ids = pred[:, 5].astype(np.int32)
        coefs = pred[:, 6:]

        # 1) 按类别阈值筛选
        keep_idx: list[int] = []
        for i in range(len(pred)):
            if conf[i] <= 0:
                continue
            logic = self.logic_names.get(int(cls_ids[i]))
            if logic is None:
                continue
            if conf[i] < self._conf_threshold(logic, int(cls_ids[i])):
                continue
            keep_idx.append(i)

        # 2) 逐类别 NMS
        kept: list[int] = []
        by_cls: dict[int, list[int]] = {}
        for i in keep_idx:
            by_cls.setdefault(int(cls_ids[i]), []).append(i)
        for c, idxs in by_cls.items():
            idxs_arr = np.asarray(idxs)
            sel = self._nms(boxes_lb[idxs_arr], conf[idxs_arr], self.cfg.iou)
            kept.extend(int(idxs_arr[s]) for s in sel)

        kept.sort(key=lambda i: -float(conf[i]))
        kept = kept[:self.cfg.max_det]

        t_filt = time.perf_counter()

        # 3) 组装 detections（bbox/polygon 换算回原图）
        detections = []
        counts: dict[str, int] = {}
        best_ball = None
        for i in kept:
            cid = int(cls_ids[i])
            logic = self.logic_names[cid]
            counts[logic] = counts.get(logic, 0) + 1

            b = boxes_lb[i]
            x1 = float(np.clip((b[0] - pad_x) / scale, 0, frame_w - 1))
            y1 = float(np.clip((b[1] - pad_y) / scale, 0, frame_h - 1))
            x2 = float(np.clip((b[2] - pad_x) / scale, 0, frame_w - 1))
            y2 = float(np.clip((b[3] - pad_y) / scale, 0, frame_h - 1))

            polygon = None
            if proto_flat is not None and logic in POLYGON_NEEDED:
                polygon = self._mask_to_polygon(
                    coefs[i], proto_flat, proto_hw, b,
                    scale, pad_x, pad_y, frame_w, frame_h)

            det = {
                "class": logic,
                "confidence": round(float(conf[i]), 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "polygon": polygon,
            }
            detections.append(det)
            if logic == BALL_LOGIC_NAME:
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                if best_ball is None or det["confidence"] > best_ball["confidence"]:
                    best_ball = {"confidence": det["confidence"], "pos": (cx, cy)}

        t_poly = time.perf_counter()
        timing = {
            "nms_ms": (t_filt - t_nms) * 1000,
            "poly_ms": (t_poly - t_filt) * 1000,
        }
        return detections, counts, best_ball, timing

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, frame_idx: int = 0, fps: float = 30.0):
        """完整一帧: 预处理 -> 推理 -> 后处理 -> jsonl 记录（dict）。"""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        tensor, scale, pad_x, pad_y = self.preprocess(frame)
        t1 = time.perf_counter()
        out0, proto = self.infer(tensor)
        t2 = time.perf_counter()
        detections, counts, best_ball, tms = self.postprocess(
            out0, proto, (h, w), scale, pad_x, pad_y)
        t3 = time.perf_counter()

        all_counts = {name: counts.get(name, 0) for name in self.logic_names.values()}
        rec = {
            "frame": frame_idx,
            "timestamp": round(frame_idx / fps, 2) if fps else 0.0,
            "counts": all_counts,
            "ball_pos": ([round(best_ball["pos"][0], 1), round(best_ball["pos"][1], 1)]
                         if best_ball else None),
            "detections": detections,
        }
        rec_timing = {
            "pre_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_ms": (t3 - t2) * 1000,
            **tms,
        }
        return rec, rec_timing
