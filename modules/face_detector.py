from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config


_REF_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceDetection:

    box: Tuple[int, int, int, int]
    landmarks: np.ndarray
    score: float

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


class FaceDetector:

    def __init__(
        self,
        model_path: Path = config.FACE_MODEL_PATH,
        input_size: Tuple[int, int] = config.FACE_INPUT_SIZE,
        conf_threshold: float = config.FACE_CONF_THRESHOLD,
        nms_threshold: float = config.FACE_NMS_THRESHOLD,
        top_k: int = config.FACE_TOP_K,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self._detector: Optional["cv2.FaceDetectorYN"] = None

        if not Path(model_path).is_file():
            print(f"[FaceDetector] model missing at {model_path} — stub mode.")
            return
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                input_size,
                self.conf_threshold,
                self.nms_threshold,
                self.top_k,
            )
            print(f"[FaceDetector] YuNet loaded ({model_path.name}, "
                  f"input={input_size}).")
        except Exception as e:
            print(f"[FaceDetector] YuNet load failed: {e}")
            self._detector = None

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        if self._detector is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        in_w, in_h = self.input_size
        if (w, h) != (in_w, in_h):
            small = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            sx, sy = w / in_w, h / in_h
        else:
            small = frame
            sx, sy = 1.0, 1.0

        self._detector.setInputSize((in_w, in_h))
        try:
            _, faces = self._detector.detect(small)
        except cv2.error:
            return []
        if faces is None:
            return []

        out: List[FaceDetection] = []
        for row in faces:
            x, y, fw, fh = row[0:4]
            score = float(row[14])
            x1 = int(round(x * sx))
            y1 = int(round(y * sy))
            x2 = int(round((x + fw) * sx))
            y2 = int(round((y + fh) * sy))
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            lmk = row[4:14].reshape(5, 2).astype(np.float32).copy()
            lmk[:, 0] *= sx
            lmk[:, 1] *= sy

            out.append(FaceDetection(box=(x1, y1, x2, y2), landmarks=lmk, score=score))
        return out

    @staticmethod
    def align_crop(
        frame: np.ndarray,
        det: FaceDetection,
        output_size: int = config.FACE_ALIGN_SIZE,
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        try:
            src = det.landmarks.astype(np.float32)
            dst = _REF_LANDMARKS_112 * (output_size / 112.0)
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if M is None:
                raise ValueError("estimateAffinePartial2D returned None")
            aligned = cv2.warpAffine(
                frame, M, (output_size, output_size),
                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
            )
            return aligned
        except Exception:
            x1, y1, x2, y2 = det.box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                return np.empty((0, 0, 3), dtype=np.uint8)
            crop = frame[y1:y2, x1:x2]
            return cv2.resize(crop, (output_size, output_size),
                              interpolation=cv2.INTER_LINEAR)
