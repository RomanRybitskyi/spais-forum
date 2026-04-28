from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


class PersonDetector:
    def __init__(
        self,
        model_path: Path = config.PERSON_MODEL_PATH,
        input_size: int = config.PERSON_INPUT_SIZE,
        conf_threshold: float = config.PERSON_CONF_THRESHOLD,
        iou_threshold: float = config.PERSON_IOU_THRESHOLD,
        class_id: int = config.PERSON_CLASS_ID,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_id = class_id
        self.session: Optional["ort.InferenceSession"] = None
        self.input_name: Optional[str] = None
        self.is_ready = False

        if ort is None:
            print("[PersonDetector] onnxruntime not available — running in stub mode.")
            return

        if not Path(model_path).is_file():
            print(f"[PersonDetector] model not found at {model_path} — stub mode.")
            return

        providers = []
        if config.USE_OPENVINO_EP and "OpenVINOExecutionProvider" in ort.get_available_providers():
            providers.append("OpenVINOExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        n_threads = max(1, (os.cpu_count() or 4) // 2)
        sess_opts.intra_op_num_threads = n_threads
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.is_ready = True
        print(f"[PersonDetector] ONNX model loaded ({providers[0]}).")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob), scale, pad_x, pad_y

    def _postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_shape: tuple[int, int],
    ) -> List[List[float]]:
        preds = outputs[0]
        preds = preds.transpose(1, 0)

        boxes = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        mask = (class_ids == self.class_id) & (confidences >= self.conf_threshold)
        boxes = boxes[mask]
        confidences = confidences[mask]
        if boxes.shape[0] == 0:
            return []

        cx, cy, w, h = boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        H, W = orig_shape
        x1 = np.clip(x1, 0, W - 1)
        y1 = np.clip(y1, 0, H - 1)
        x2 = np.clip(x2, 0, W - 1)
        y2 = np.clip(y2, 0, H - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        keep = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(),
                                self.conf_threshold, self.iou_threshold)
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()

        results = []
        for i in keep:
            results.append([
                float(boxes_xyxy[i, 0]),
                float(boxes_xyxy[i, 1]),
                float(boxes_xyxy[i, 2]),
                float(boxes_xyxy[i, 3]),
                float(confidences[i]),
            ])
        return results

    def detect(self, frame: np.ndarray) -> List[List[float]]:
        if not self.is_ready:
            return []
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(outputs, scale, pad_x, pad_y, frame.shape[:2])
