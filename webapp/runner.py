from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import config
from modules.detector import PersonDetector
from modules.tracker import SortTracker
from modules.gender import GenderClassifier
from modules.helmet import HelmetDetector
from modules.zone import ZoneAnalytics
from modules.renderer import Renderer
from modules.face_detector import FaceDetector
from modules.emotion import EmotionClassifier


def _crop_person(frame: np.ndarray, box) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


class PipelineRunner:
    SCENARIOS = {
        1: "Presence + Gender + Helmet",
        2: "Face: Gender + Emotion",
        3: "Scenario 3 (coming soon)",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_event = threading.Event()

        self._latest_jpeg: Optional[bytes] = None
        self._stats_dict: dict = {}
        self._fps: float = 0.0
        self._latencies: dict = {}
        self._frame_w = config.CAPTURE_WIDTH
        self._frame_h = config.CAPTURE_HEIGHT

        self._scenario = 1
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._models_loaded = False
        self.detector: Optional[PersonDetector] = None
        self.tracker: Optional[SortTracker] = None
        self.gender: Optional[GenderClassifier] = None
        self.helmet: Optional[HelmetDetector] = None
        self.zone: Optional[ZoneAnalytics] = None
        self.renderer: Optional[Renderer] = None
        self.face_detector: Optional[FaceDetector] = None
        self.emotion: Optional[EmotionClassifier] = None

        self._zone_polygon = list(config.ROI_POLYGON)

        self._emotion_history: deque = deque(maxlen=config.EMOTION_HISTORY_LEN)
        self._face_gender_cache: dict = {}
        self._face_emotion_cache: dict = {}
        self._prev_scenario: int = 1
        self._scenario_changed = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._models_loaded:
                self._load_models()
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        with self._lock:
            self._running = False
            self._thread = None

    def set_scenario(self, scenario: int) -> None:
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
        with self._lock:
            if self._scenario != int(scenario):
                self._scenario = int(scenario)
                self._scenario_changed.set()

    def set_zone(self, polygon) -> None:
        if not polygon or len(polygon) < 3:
            raise ValueError("polygon must have at least 3 points")
        poly = [(int(x), int(y)) for x, y in polygon]
        with self._lock:
            self._zone_polygon = poly
            if self.zone is not None:
                self.zone.set_polygon(poly)

    def set_anonymise(self, value: bool) -> None:
        config.ANONYMISE_OVERLAY = bool(value)

    def set_debug(self, value: bool) -> None:
        config.SHOW_DEBUG_HUD = bool(value)

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "scenario": self._scenario,
                "scenarios": self.SCENARIOS,
                "anonymise": config.ANONYMISE_OVERLAY,
                "debug": config.SHOW_DEBUG_HUD,
                "frame_size": [self._frame_w, self._frame_h],
                "zone": self._zone_polygon,
                "fps": round(self._fps, 1),
                "stats": self._stats_dict,
                "latencies": self._latencies,
            }

    def get_latest_jpeg(self, wait: bool = True, timeout: float = 1.0) -> Optional[bytes]:
        if wait:
            self._frame_event.wait(timeout)
            self._frame_event.clear()
        with self._lock:
            return self._latest_jpeg

    def _load_models(self) -> None:
        print("[Runner] loading models...")
        self.detector = PersonDetector()
        self.tracker = SortTracker()
        self.gender = GenderClassifier()
        self.helmet = HelmetDetector()
        self.zone = ZoneAnalytics(polygon=self._zone_polygon)
        self.renderer = Renderer()
        self.face_detector = FaceDetector()
        self.emotion = EmotionClassifier()
        self._models_loaded = True
        print("[Runner] models loaded.")

    def _run_loop(self) -> None:
        cap = self._open_capture(self._scenario)
        if cap is None:
            self._publish_error_frame(f"Cannot open camera {config.CAMERA_INDEX}")
            with self._lock:
                self._running = False
            return

        gender_cache: dict = {}
        helmet_cache: dict = {}
        frame_times: deque = deque(maxlen=30)
        last_t = time.perf_counter()
        frame_idx = 0
        active_scenario = self._scenario

        try:
            while not self._stop_event.is_set():
                if self._scenario_changed.is_set():
                    self._scenario_changed.clear()
                    with self._lock:
                        active_scenario = self._scenario
                    w, h = self._scenario_resolution(active_scenario)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    frame_times.clear()
                    self._emotion_history.clear()
                    self._face_gender_cache.clear()
                    self._face_emotion_cache.clear()
                    last_t = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    self._frame_w = frame.shape[1]
                    self._frame_h = frame.shape[0]
                    scenario = self._scenario
                active_scenario = scenario

                if scenario == 1:
                    annotated, fps_now, lat = self._process_full(
                        frame, frame_idx, gender_cache, helmet_cache,
                        frame_times, last_t,
                    )
                elif scenario == 2:
                    annotated, fps_now, lat = self._process_face(
                        frame, frame_idx, frame_times, last_t,
                    )
                else:
                    annotated = frame
                    title = self.SCENARIOS.get(scenario, f"Scenario {scenario}")
                    cv2.putText(annotated, title, (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
                    fps_now, lat = 0.0, {}

                last_t = time.perf_counter()

                ok, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._fps = fps_now
                        self._latencies = lat
                    self._frame_event.set()

                frame_idx += 1
        finally:
            cap.release()
            with self._lock:
                self._running = False
            print("[Runner] loop stopped.")

    @staticmethod
    def _scenario_resolution(scenario: int) -> tuple[int, int]:
        if scenario == 2:
            return config.SCENARIO2_CAPTURE_WIDTH, config.SCENARIO2_CAPTURE_HEIGHT
        return config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT

    def _open_capture(self, scenario: int) -> Optional[cv2.VideoCapture]:
        backend = getattr(config, 'CAMERA_BACKEND', 0)
        cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        if not cap.isOpened():
            print(f"[Runner] cannot open camera index {config.CAMERA_INDEX}")
            return None
        w, h = self._scenario_resolution(scenario)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    def _process_full(self, frame, frame_idx, gender_cache, helmet_cache,
                      frame_times, last_t):
        assert self.detector and self.tracker and self.gender and self.helmet
        assert self.zone and self.renderer

        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        tracks = self.tracker.update(detections)
        t_trk = (time.perf_counter() - t0) * 1000

        t_cls = 0.0
        if frame_idx % config.CLASSIFIER_FRAME_STRIDE == 0:
            t0 = time.perf_counter()
            active_ids = set()
            for trk in tracks:
                active_ids.add(trk.id)
                person_crop = _crop_person(frame, trk.box)
                gender_cache[trk.id] = self.gender.predict(person_crop)
                helmet_cache[trk.id] = self.helmet.predict(frame, trk.box)
            for stale in set(gender_cache) - active_ids:
                gender_cache.pop(stale, None)
            for stale in set(helmet_cache) - active_ids:
                helmet_cache.pop(stale, None)
            t_cls = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.zone.update(tracks)
        stats = self.zone.stats()
        t_zone = (time.perf_counter() - t0) * 1000

        flash_ids = self.zone.recently_entered_ids()
        self.renderer.draw_zone(frame, self.zone.contour, flash=bool(flash_ids))
        for trk in tracks:
            self.renderer.draw_track(
                frame, trk,
                in_zone=self.zone.is_inside(trk.bottom_centre),
                gender_label=gender_cache.get(trk.id, "?"),
                helmet_label=helmet_cache.get(trk.id, "unknown"),
                flash=trk.id in flash_ids,
            )

        now = time.perf_counter()
        frame_times.append(now - last_t)
        fps = 1.0 / (float(np.mean(frame_times)) if frame_times else 1.0)

        latencies = {
            "det": round(t_det, 1), "trk": round(t_trk, 1),
            "cls": round(t_cls, 1), "zone": round(t_zone, 1),
        }

        self.renderer.draw_hud(frame, stats, fps, latencies)

        with self._lock:
            self._stats_dict = {
                "total_count": stats.total_count,
                "currently_inside": stats.currently_inside,
                "average_dwell_seconds": round(stats.average_dwell_seconds, 1),
                "longest_dwell_seconds": round(stats.longest_dwell_seconds, 1),
            }

        return frame, fps, latencies

    def _process_face(self, frame, frame_idx, frame_times, last_t):
        assert self.face_detector and self.emotion and self.gender and self.renderer

        t0 = time.perf_counter()
        faces = self.face_detector.detect(frame)
        t_face = (time.perf_counter() - t0) * 1000

        faces.sort(key=lambda f: f.area, reverse=True)

        t_emo = 0.0
        t_gen = 0.0
        n_faces = len(faces)
        primary_label = "Unknown"

        run_gender = (frame_idx % max(1, config.CLASSIFIER_FRAME_STRIDE) == 0)
        run_emotion = (frame_idx % max(1, getattr(config, 'EMOTION_FRAME_STRIDE', 6)) == 0)
        active_slots: set = set()

        for slot, det in enumerate(faces):
            crop = self.face_detector.align_crop(
                frame, det, output_size=config.FACE_ALIGN_SIZE,
            )

            if run_emotion or slot not in self._face_emotion_cache:
                te0 = time.perf_counter()
                label, score, all_scores = self.emotion.predict(crop)
                t_emo += (time.perf_counter() - te0) * 1000
                self._face_emotion_cache[slot] = (label, score, all_scores)
            else:
                label, score, all_scores = self._face_emotion_cache[slot]

            if score < config.EMOTION_CONF_THRESHOLD:
                label_disp = "Unknown"
            else:
                label_disp = label
            if slot == 0:
                primary_label = label_disp

            active_slots.add(slot)
            if run_gender or slot not in self._face_gender_cache:
                tg0 = time.perf_counter()
                gender_label = self._gender_for_face(frame, det)
                t_gen += (time.perf_counter() - tg0) * 1000
                self._face_gender_cache[slot] = gender_label
            gender_label = self._face_gender_cache.get(slot, "?")

            self.renderer.draw_face_track(
                frame, det,
                gender_label=gender_label,
                emotion_label=label_disp,
                emotion_score=score,
                all_scores=all_scores,
                face_id=slot,
            )

        for stale in set(self._face_gender_cache) - active_slots:
            self._face_gender_cache.pop(stale, None)
        for stale in set(self._face_emotion_cache) - active_slots:
            self._face_emotion_cache.pop(stale, None)

        self._emotion_history.append(primary_label)

        from collections import Counter
        hist = list(self._emotion_history)
        non_unknown = [x for x in hist if x != "Unknown"]
        if non_unknown:
            dominant = Counter(non_unknown).most_common(1)[0][0]
        elif hist:
            dominant = Counter(hist).most_common(1)[0][0]
        else:
            dominant = "Unknown"

        now = time.perf_counter()
        frame_times.append(now - last_t)
        fps = 1.0 / (float(np.mean(frame_times)) if frame_times else 1.0)

        latencies = {
            "face": round(t_face, 1),
            "emo":  round(t_emo, 1),
            "gen":  round(t_gen, 1),
        }

        self.renderer.draw_face_hud(
            frame, hist, dominant, n_faces, fps, latencies,
        )

        with self._lock:
            self._stats_dict = {
                "n_faces": n_faces,
                "dominant_emotion": dominant,
                "primary_emotion": primary_label,
                "emotion_history": hist,
            }

        return frame, fps, latencies

    def _gender_for_face(self, frame: np.ndarray, det) -> str:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        mw = int((x2 - x1) * 0.25)
        mh = int((y2 - y1) * 0.25)
        cx1 = max(0, x1 - mw)
        cy1 = max(0, y1 - mh)
        cx2 = min(W - 1, x2 + mw)
        cy2 = min(H - 1, y2 + mh)
        if cx2 <= cx1 or cy2 <= cy1:
            return "?"
        crop = frame[cy1:cy2, cx1:cx2]
        try:
            return self.gender.predict_face(crop)
        except Exception:
            return "?"

    def _publish_error_frame(self, msg: str) -> None:
        img = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
        cv2.putText(img, msg, (40, self._frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 230), 2)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            with self._lock:
                self._latest_jpeg = buf.tobytes()
            self._frame_event.set()
