from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

import config
from modules.detector import PersonDetector
from modules.tracker import SortTracker
from modules.gender import GenderClassifier
from modules.helmet import HelmetDetector
from modules.zone import ZoneAnalytics
from modules.renderer import Renderer


def crop_person(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.CAMERA_INDEX}")

    detector = PersonDetector()
    tracker = SortTracker()
    gender = GenderClassifier()
    helmet = HelmetDetector()
    zone = ZoneAnalytics(polygon=config.ROI_POLYGON)
    renderer = Renderer()

    gender_cache: dict[int, str] = {}
    helmet_cache: dict[int, str] = {}

    frame_times: deque[float] = deque(maxlen=30)
    last_t = time.perf_counter()
    frame_idx = 0

    print("Pipeline started. 'q'=quit, 'a'=toggle anonymise, 'd'=toggle debug HUD.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed; exiting.")
            break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tracks = tracker.update(detections)
        t_trk = (time.perf_counter() - t0) * 1000

        t_cls = 0.0
        if frame_idx % config.CLASSIFIER_FRAME_STRIDE == 0:
            t0 = time.perf_counter()
            active_ids = set()
            for trk in tracks:
                active_ids.add(trk.id)
                person_crop = crop_person(frame, trk.box)
                gender_cache[trk.id] = gender.predict(person_crop)
                helmet_cache[trk.id] = helmet.predict(frame, trk.box)
            for stale in set(gender_cache) - active_ids:
                gender_cache.pop(stale, None)
            for stale in set(helmet_cache) - active_ids:
                helmet_cache.pop(stale, None)
            t_cls = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        zone.update(tracks)
        stats = zone.stats()
        t_zone = (time.perf_counter() - t0) * 1000

        renderer.draw_zone(frame, zone.contour)
        for trk in tracks:
            renderer.draw_track(
                frame, trk,
                in_zone=zone.is_inside(trk.bottom_centre),
                gender_label=gender_cache.get(trk.id, "?"),
                helmet_label=helmet_cache.get(trk.id, "unknown"),
            )

        now = time.perf_counter()
        frame_times.append(now - last_t)
        last_t = now
        fps = 1.0 / (np.mean(frame_times) if frame_times else 1.0)

        renderer.draw_hud(frame, stats, fps, {
            "det": t_det, "trk": t_trk, "cls": t_cls, "zone": t_zone,
        })

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            config.ANONYMISE_OVERLAY = not config.ANONYMISE_OVERLAY
            print(f"Anonymise overlay: {config.ANONYMISE_OVERLAY}")
        elif key == ord('d'):
            config.SHOW_DEBUG_HUD = not config.SHOW_DEBUG_HUD

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
