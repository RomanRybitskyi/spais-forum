from __future__ import annotations

import numpy as np

from modules.face_detector import FaceDetection, FaceDetector


def test_face_detection_dataclass_fields():
    lm = np.zeros((5, 2), dtype=np.float32)
    d = FaceDetection(box=(10, 20, 50, 80), landmarks=lm, score=0.9)
    assert d.box == (10, 20, 50, 80)
    assert d.landmarks.shape == (5, 2)
    assert d.score == 0.9
    assert d.width == 40
    assert d.height == 60
    assert d.area == 40 * 60


def test_detect_blank_frame_returns_list():
    fd = FaceDetector()
    out = fd.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(out, list)
    for d in out:
        assert isinstance(d, FaceDetection)


def test_detect_handles_none_and_empty():
    fd = FaceDetector()
    assert fd.detect(None) == []
    assert fd.detect(np.empty((0, 0, 3), dtype=np.uint8)) == []


def test_align_crop_output_shape():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    lm = np.array(
        [[60, 80], [110, 80], [85, 100], [65, 130], [105, 130]],
        dtype=np.float32,
    )
    det = FaceDetection(box=(40, 50, 140, 160), landmarks=lm, score=0.99)
    out = FaceDetector.align_crop(frame, det, output_size=112)
    assert out.shape == (112, 112, 3)
    assert out.dtype == np.uint8


def test_align_crop_fallback_on_empty_landmarks():
    frame = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    lm = np.zeros((5, 2), dtype=np.float32)
    det = FaceDetection(box=(10, 10, 80, 80), landmarks=lm, score=0.5)
    out = FaceDetector.align_crop(frame, det, output_size=64)
    assert out.shape[:2] == (64, 64)
