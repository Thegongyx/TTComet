# NO-BYTECODE: 必须在所有 import 之前
import sys as _sys
if not getattr(_sys, 'dont_write_bytecode', False):
    _sys.dont_write_bytecode = True

import os
import logging
import hashlib
from urllib.parse import urlparse
from appdirs import user_cache_dir

# CONFIRM: this file is loaded (stderr to bypass buffering)
print(f"[model.py v4] Loaded from {__file__}", file=_sys.stderr, flush=True)

from label_studio_ml.model import LabelStudioMLBase

from control_models.base import ControlModel
from control_models.choices import ChoicesModel
from control_models.rectangle_labels import RectangleLabelsModel
from control_models.rectangle_labels_obb import RectangleLabelsObbModel
from control_models.polygon_labels import PolygonLabelsModel
from control_models.brush_labels import BrushLabelsModel
from control_models.keypoint_labels import KeypointLabelsModel
from control_models.video_rectangle import VideoRectangleModel
from control_models.timeline_labels import TimelineLabelsModel
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

# Register available model classes
available_model_classes = [
    ChoicesModel,
    RectangleLabelsModel,
    RectangleLabelsObbModel,
    PolygonLabelsModel,
    BrushLabelsModel,
    KeypointLabelsModel,
    VideoRectangleModel,
    TimelineLabelsModel,
]


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""
    
    # 类属性兜底：防止子进程/multiprocessing 中 __init__ 被跳过
    _project_id = None

    @property
    def project_id(self):
        return self._project_id

    @project_id.setter
    def project_id(self, value):
        self._project_id = value

    def __init__(self, project_id=None, **kwargs):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.model_version = "yolo"
        # 从 label_config 创建 label_interface（新版本 SDK 不再自动设置）
        if self.label_config:
            from label_studio_sdk.label_interface import LabelInterface
            self.label_interface = LabelInterface(self.label_config)

    def setup(self):
        """Configure any parameters of your model here"""
        self.set("model_version", "yolo")

    def detect_control_models(self) -> List[ControlModel]:
        """Detect control models based on the labeling config.
        Control models are used to predict regions for different control tags in the labeling config.
        """
        control_models = []

        for control in self.label_interface.controls:
            # skipping tags without toName
            if not control.to_name:
                logger.warning(
                    f'{control.tag} {control.name} has no "toName" attribute, skipping it'
                )
                continue

            # match control tag with available control models
            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        logger.debug(
                            f"No instance created for {control.tag} {control.name}"
                        )
                        continue
                    if not instance.label_map:
                        logger.error(
                            f"No label map built for the '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"This indicates that your Label Studio config labels do not match the model's labels.\n"
                            f"To fix this, ensure that the 'value' or 'predicted_values' attribute "
                            f"in your Label Studio config matches one or more of these model labels.\n"
                            f"If you don't want to use this control tag for predictions, "
                            f'add `model_skip="true"` to it.\n'
                            f"Examples:\n"
                            f'  <Label value="Car"/>\n'
                            f'  <Label value="YourLabel" predicted_values="label1,label2"/>\n'
                            f"Labels provided in your labeling config:\n"
                            f"  {str(control.labels_attrs)}\n"
                            f"Available '{instance.model_path}' model labels:\n"
                            f"  {list(instance.model.names.values())}"
                        )
                        continue

                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance}")
                    break

        if not control_models:
            control_tags = ", ".join([c.type for c in available_model_classes])
            raise ValueError(
                f"No suitable control tags (e.g. {control_tags} connected to Image or Video object tags) "
                f"detected in the label config:\n{self.label_config}"
            )

        return control_models

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Run YOLO predictions on the tasks
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create)
        :return model_response
            ModelResponse(predictions=predictions) with
            predictions [Predictions array in JSON format]
            (https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        logger.info(
            f"Run prediction on {len(tasks)} tasks"
        )
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:
            regions = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(path)

            # calculate final score
            all_scores = [region["score"] for region in regions if "score" in region]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            # compose final prediction
            prediction = {
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            }
            predictions.append(prediction)

        return predictions

    def build_label_map(self, tag_name, names):
        """Build label map using LabelInterface (new SDK API).
        
        Matches model class names to LS label values via direct name match or alias.
        """
        label_map = {}
        
        if hasattr(self, "label_interface"):
            for control in self.label_interface.controls:
                if control.name != tag_name:
                    continue
                # control.labels_attrs: dict of {label_value: label_tag}
                for label_value, label_tag in getattr(control, "labels_attrs", {}).items():
                    # Check alias first
                    alias = (label_tag.alias or "").strip() if hasattr(label_tag, "alias") else ""
                    if alias and alias in names:
                        label_map[alias] = label_value
                    # Direct name match
                    if label_value in names:
                        label_map[label_value] = label_value
                break
        
        # Fallback: direct 1:1 name matching
        if not label_map:
            for name in names:
                label_map[name] = name
        
        return label_map

    def fit(self, event, data, **additional_params):
        """Extend fit with cache cleanup for training images.

        Removes cached images associated with annotated tasks to ensure
        that subsequent training uses fresh data rather than stale cache.

        :param event: event type (e.g. 'ANNOTATION_CREATED')
        :param data: annotation payload dictionary
        :param additional_params: extra keyword arguments
        :return: dict with cache_cleanup status
        """
        try:
            url = data.get("annotation", {}).get("task", {}).get("data", {}).get("image", "")
        except KeyError:
            return {"cache_cleanup": "no_image_url"}

        if not url:
            logger.warning("No image URL found in annotation data")
            return {"cache_cleanup": False}

        cache_dir = getattr(self, 'cache_dir', user_cache_dir("label-studio"))
        md5_prefix = hashlib.md5(url.encode()).hexdigest()[:8]
        basename = os.path.basename(urlparse(url).path)
        cache_path = os.path.join(cache_dir, f"{md5_prefix}__{basename}")

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed cached image: {cache_path}")

        return {"cache_cleanup": True}
