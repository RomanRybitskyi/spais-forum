from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


class HelmetDetector:
    def __init__(self) -> None:
        self.session: Optional["ort.InferenceSession"] = None
        self.input_name: Optional[str] = None
        self.is_ready = False

        if ort is None:
            print("[HelmetDetector] onnxruntime not available — stub mode.")
            return
        if not config.HELMET_MODEL_PATH.is_file():
            print(f"[HelmetDetector] model not found at {config.HELMET_MODEL_PATH} — stub mode.")
            return

        sess_opts = ort.SessionOptions()
        n = max(1, (os.cpu_count() or 4) // 2)
        sess_opts.intra_op_num_threads = n
        sess_opts.inter_op_num_threads = n
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if config.USE_OPENVINO_EP and "OpenVINOExecutionProvider" in ort.get_available_providers():
            providers.append("OpenVINOExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(config.HELMET_MODEL_PATH), sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.is_ready = True
        print(f"[HelmetDetector] ONNX model loaded ({providers[0]}).")

    @staticmethod
    def head_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        h = y2 - y1
        head_y2 = y1 + max(1, int(h * config.HEAD_CROP_RATIO))
        H, W = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W - 1, x2); head_y2 = min(H - 1, head_y2)
        if x2 <= x1 or head_y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:head_y2, x1:x2]

    def predict(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> str:
        if not self.is_ready:
            return "unknown"
        crop = self.head_crop(frame, box)
        if crop.size == 0:
            return "unknown"

        size = config.HELMET_INPUT_SIZE
        h, w = crop.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        preds = outputs[0].transpose(1, 0)
        if preds.shape[1] < 5:
            return "unknown"
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = confidences >= config.HELMET_CONF_THRESHOLD
        if not np.any(mask):
            return "unknown"

        best_idx_in_filtered = int(np.argmax(confidences[mask]))
        best_class_id = int(class_ids[mask][best_idx_in_filtered])
        return config.HELMET_CLASS_NAMES.get(best_class_id, "unknown")
