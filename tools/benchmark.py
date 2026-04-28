from __future__ import annotations

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np

import config
from modules.detector import PersonDetector
from modules.tracker import SortTracker
from modules.gender import GenderClassifier
from modules.helmet import HelmetDetector
from modules.zone import ZoneAnalytics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video file path or camera index")
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    detector = PersonDetector()
    tracker = SortTracker()
    gender = GenderClassifier()
    helmet = HelmetDetector()
    zone = ZoneAnalytics(polygon=config.ROI_POLYGON)

    timings: dict[str, list[float]] = defaultdict(list)
    n = 0
    t_start = time.perf_counter()
    while n < args.frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        timings["det"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tracks = tracker.update(detections)
        timings["trk"].append(time.perf_counter() - t0)

        if n % config.CLASSIFIER_FRAME_STRIDE == 0:
            t0 = time.perf_counter()
            for trk in tracks:
                x1, y1, x2, y2 = trk.box
                gender.predict(frame[max(0, y1):y2, max(0, x1):x2])
                helmet.predict(frame, trk.box)
            timings["cls"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        zone.update(tracks)
        zone.stats()
        timings["zone"].append(time.perf_counter() - t0)

        n += 1

    elapsed = time.perf_counter() - t_start
    cap.release()

    print(f"\n=== Benchmark over {n} frames ({elapsed:.2f}s) ===")
    print(f"End-to-end FPS: {n / elapsed:.1f}")
    print(f"{'stage':<8}{'avg ms':>10}{'p50':>10}{'p95':>10}")
    for stage, vals in timings.items():
        arr = np.array(vals) * 1000
        if arr.size == 0:
            continue
        print(f"{stage:<8}{arr.mean():>10.2f}{np.percentile(arr, 50):>10.2f}{np.percentile(arr, 95):>10.2f}")


if __name__ == "__main__":
    main()

