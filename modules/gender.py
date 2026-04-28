from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

import config


_CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


class GenderClassifier:
    def __init__(self) -> None:
        self.backend: Optional[str] = None
        self.session = None
        self.input_name: Optional[str] = None
        self.caffe_net = None
        self.face_app = None

        requested = getattr(config, "GENDER_BACKEND", "auto").lower()

        if requested == "insightface":
            if not self._try_load_insightface():
                print("[GenderClassifier] requested 'insightface' backend failed — stub mode.")
            return
        if requested == "yolo":
            if not self._try_load_yolo():
                print("[GenderClassifier] requested 'yolo' backend failed — stub mode.")
            return
        if requested == "onnx":
            if not self._try_load_onnx():
                print("[GenderClassifier] requested 'onnx' backend failed — stub mode.")
            return
        if requested == "caffe":
            if not self._try_load_caffe():
                print("[GenderClassifier] requested 'caffe' backend failed — stub mode.")
            return

        if getattr(config, "USE_INSIGHTFACE_GENDER", False) and self._try_load_insightface():
            return
        if self._try_load_yolo():
            return
        if config.USE_CAFFE_GENDER and self._try_load_caffe():
            return
        if self._try_load_onnx():
            return
        if self._try_load_caffe():
            return
        print("[GenderClassifier] no model available — running in stub mode.")

    def _try_load_insightface(self) -> bool:
        if FaceAnalysis is None:
            print("[GenderClassifier] insightface not installed — skipping.")
            return False
        try:
            name = getattr(config, "INSIGHTFACE_MODEL_NAME", "buffalo_s")
            det_size = getattr(config, "INSIGHTFACE_DET_SIZE", 256)
            self.face_app = FaceAnalysis(
                name=name,
                allowed_modules=["detection", "genderage"],
                providers=["CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
            self.backend = "insightface"
            print(f"[GenderClassifier] InsightFace '{name}' loaded "
                  f"(det_size={det_size}).")
            return True
        except Exception as e:
            print(f"[GenderClassifier] InsightFace load failed: {e}")
            self.face_app = None
            return False

    def _try_load_yolo(self) -> bool:
        if ort is None:
            return False
        yolo_path = getattr(config, "GENDER_YOLO_MODEL_PATH", None)
        if yolo_path is None or not yolo_path.is_file():
            return False
        sess_opts = ort.SessionOptions()
        n = max(1, (os.cpu_count() or 4) // 2)
        sess_opts.intra_op_num_threads = n
        sess_opts.inter_op_num_threads = n
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        if getattr(config, "USE_OPENVINO_EP", False) and \
                "OpenVINOExecutionProvider" in ort.get_available_providers():
            providers.append("OpenVINOExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            str(yolo_path), sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.backend = "yolo"
        print(f"[GenderClassifier] YOLOv8 ONNX model loaded ({providers[0]}).")
        return True

    def _try_load_onnx(self) -> bool:
        if ort is None or not config.GENDER_MODEL_PATH.is_file():
            return False
        sess_opts = ort.SessionOptions()
        n = max(1, (os.cpu_count() or 4) // 2)
        sess_opts.intra_op_num_threads = n
        sess_opts.inter_op_num_threads = n
        self.session = ort.InferenceSession(
            str(config.GENDER_MODEL_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.backend = "onnx"
        print("[GenderClassifier] ONNX model loaded.")
        return True

    def _try_load_caffe(self) -> bool:
        if not (config.GENDER_CAFFE_PROTO.is_file() and config.GENDER_CAFFE_WEIGHTS.is_file()):
            return False
        self.caffe_net = cv2.dnn.readNetFromCaffe(
            str(config.GENDER_CAFFE_PROTO),
            str(config.GENDER_CAFFE_WEIGHTS),
        )
        self.backend = "caffe"
        print("[GenderClassifier] Caffe model loaded.")
        return True

    def predict(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr is None or crop_bgr.size == 0:
            return "?"

        if self.backend == "insightface":
            return self._predict_insightface(crop_bgr)
        if self.backend == "yolo":
            return self._predict_yolo(crop_bgr)
        if self.backend == "onnx":
            return self._predict_onnx(crop_bgr)
        if self.backend == "caffe":
            return self._predict_caffe(crop_bgr)
        return "?"

    def predict_face(self, face_crop_bgr: np.ndarray) -> str:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return "?"
        if self.backend == "insightface":
            try:
                faces = self.face_app.get(face_crop_bgr)
            except Exception:
                return "?"
            if not faces:
                return "?"
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            return self._face_to_label(face)
        if self.backend == "yolo":
            return self._predict_yolo(face_crop_bgr)
        if self.backend == "onnx":
            return self._predict_onnx(face_crop_bgr)
        if self.backend == "caffe":
            return self._predict_caffe(face_crop_bgr)
        return "?"

    @staticmethod
    def _face_to_label(face) -> str:
        sex = getattr(face, "sex", None)
        if sex == "M":
            label = config.GENDER_LABELS[0]
        elif sex == "F":
            label = config.GENDER_LABELS[1]
        else:
            g = int(getattr(face, "gender", -1))
            if g == 1:
                label = config.GENDER_LABELS[0]
            elif g == 0:
                label = config.GENDER_LABELS[1]
            else:
                return "?"
        if getattr(config, "SHOW_AGE", False):
            age = getattr(face, "age", None)
            if age is not None:
                label = f"{label} {int(age)}"
        return label

    def _predict_insightface(self, crop: np.ndarray) -> str:
        h = crop.shape[0]
        head_h = max(1, int(h * max(0.4, getattr(config, "HEAD_CROP_RATIO", 0.3) * 1.5)))
        head = crop[:head_h]
        if head.size == 0:
            return "?"
        try:
            faces = self.face_app.get(head)
        except Exception:
            return "?"
        if not faces:
            return "?"
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return self._face_to_label(face)

    def _predict_yolo(self, crop: np.ndarray) -> str:
        size = getattr(config, "GENDER_YOLO_INPUT_SIZE", 224)
        img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        blob = img.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: blob})[0]
        e = np.exp(out - out.max())
        probs = (e / e.sum()).flatten()
        idx = int(np.argmax(probs))
        conf_thresh = getattr(config, "GENDER_CONF_THRESHOLD", 0.0)
        if probs[idx] < conf_thresh or idx >= len(config.GENDER_LABELS):
            return "?"
        return config.GENDER_LABELS[idx]

    def _predict_onnx(self, crop: np.ndarray) -> str:
        size = config.GENDER_INPUT_SIZE
        img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        blob = img.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: blob})[0]
        e = np.exp(out - out.max())
        probs = (e / e.sum()).flatten()
        idx = int(np.argmax(probs))
        conf_thresh = getattr(config, "GENDER_CONF_THRESHOLD", 0.0)
        if probs[idx] < conf_thresh or idx >= len(config.GENDER_LABELS):
            return "?"
        return config.GENDER_LABELS[idx]

    def _predict_caffe(self, crop: np.ndarray) -> str:
        blob = cv2.dnn.blobFromImage(
            crop, scalefactor=1.0, size=(227, 227),
            mean=_CAFFE_MEAN, swapRB=False, crop=False,
        )
        self.caffe_net.setInput(blob)
        out = self.caffe_net.forward()
        probs = out.flatten()
        idx = int(np.argmax(probs))
        conf_thresh = getattr(config, "GENDER_CONF_THRESHOLD", 0.0)
        if probs[idx] < conf_thresh or idx >= len(config.GENDER_LABELS):
            return "?"
        return config.GENDER_LABELS[idx]
