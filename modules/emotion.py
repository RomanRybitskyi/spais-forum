from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import config


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


class EmotionClassifier:
    def __init__(
        self,
        model_path: Path = config.EMOTION_MODEL_PATH,
        input_size: int = config.EMOTION_INPUT_SIZE,
        labels: Tuple[str, ...] = config.EMOTION_LABELS,
    ) -> None:
        self.input_size = int(input_size)
        self.labels = tuple(labels)
        self.session: Optional["ort.InferenceSession"] = None
        self.input_name: Optional[str] = None

        if ort is None:
            print("[EmotionClassifier] onnxruntime not available — stub mode.")
            return
        if not Path(model_path).is_file():
            print(f"[EmotionClassifier] model missing at {model_path} — stub mode.")
            return
        try:
            sess_opts = ort.SessionOptions()
            n = max(1, (os.cpu_count() or 4) // 2)
            sess_opts.intra_op_num_threads = n
            sess_opts.inter_op_num_threads = 1

            providers = ["CPUExecutionProvider"]
            if getattr(config, "USE_OPENVINO_EP", False):
                providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]

            self.session = ort.InferenceSession(
                str(model_path), sess_options=sess_opts, providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"[EmotionClassifier] loaded {Path(model_path).name} "
                  f"({self.input_size}×{self.input_size}, {len(self.labels)} classes).")
        except Exception as e:
            print(f"[EmotionClassifier] load failed: {e}")
            self.session = None

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        size = self.input_size
        img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(img, (2, 0, 1))[None].astype(np.float32)
        return np.ascontiguousarray(blob)

    def predict(
        self, face_crop_bgr: np.ndarray,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        if (
            self.session is None
            or face_crop_bgr is None
            or face_crop_bgr.size == 0
            or face_crop_bgr.shape[0] < 4
            or face_crop_bgr.shape[1] < 4
        ):
            return "Unknown", 0.0, []

        blob = self._preprocess(face_crop_bgr)
        try:
            out = self.session.run(None, {self.input_name: blob})[0]
        except Exception as e:
            print(f"[EmotionClassifier] inference error: {e}")
            return "Unknown", 0.0, []

        logits = np.asarray(out).reshape(-1)
        probs = _softmax(logits)

        n = min(len(self.labels), probs.shape[0])
        pairs = [(self.labels[i], float(probs[i])) for i in range(n)]
        pairs.sort(key=lambda p: p[1], reverse=True)
        top_label, top_score = pairs[0]
        return top_label, top_score, pairs
