from __future__ import annotations

import numpy as np

import config
from modules.emotion import EmotionClassifier


def test_preprocess_output_shape():
    em = EmotionClassifier()
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    blob = em._preprocess(img)
    assert blob.shape == (1, 3, em.input_size, em.input_size)
    assert blob.dtype == np.float32


def test_predict_empty_returns_unknown():
    em = EmotionClassifier()
    label, score, scores = em.predict(np.empty((0, 0, 3), dtype=np.uint8))
    assert label == "Unknown"
    assert score == 0.0
    assert scores == []


def test_predict_none_returns_unknown():
    em = EmotionClassifier()
    label, score, scores = em.predict(None)
    assert label == "Unknown"
    assert score == 0.0
    assert scores == []


def test_predict_real_or_stub_contract():
    em = EmotionClassifier()
    crop = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    label, score, scores = em.predict(crop)
    if em.session is None:
        assert (label, score, scores) == ("Unknown", 0.0, [])
        return
    assert label in config.EMOTION_LABELS
    assert 0.0 <= score <= 1.0
    assert len(scores) == len(config.EMOTION_LABELS)
    vals = [s for _, s in scores]
    assert vals == sorted(vals, reverse=True)
    assert abs(sum(vals) - 1.0) < 1e-3
    for lbl, _ in scores:
        assert lbl in config.EMOTION_LABELS
