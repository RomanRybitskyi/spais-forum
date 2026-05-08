from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from modules.tracker import Track
from modules.zone import ZoneStats


COLOUR_GREEN = (60, 220, 60)
COLOUR_WHITE = (240, 240, 240)
COLOUR_RED = (40, 40, 230)
COLOUR_YELLOW = (40, 220, 230)
COLOUR_BLACK = (0, 0, 0)


EMOTION_COLOURS: dict = {
    "Happiness": (60, 220, 230),
    "Surprise":  (60, 180, 240),
    "Neutral":   (200, 200, 200),
    "Sadness":   (220, 140, 60),
    "Anger":     (40, 40, 230),
    "Fear":      (180, 60, 200),
    "Disgust":   (60, 180, 60),
    "Contempt":  (120, 120, 200),
    "Unknown":   (160, 160, 160),
}

EMOTION_GLYPHS: dict = {
    "Happiness": ":)",
    "Surprise":  ":O",
    "Neutral":   ":|",
    "Sadness":   ":(",
    "Anger":     ">:(",
    "Fear":      "D:",
    "Disgust":   ":P",
    "Contempt":  ":/",
    "Unknown":   "??",
}


class Renderer:
    def __init__(self) -> None:
        self.thumbs_up: Optional[np.ndarray] = self._load_icon(config.THUMBS_UP_PATH, 40)
        self.male_icon: Optional[np.ndarray] = self._load_icon(
            getattr(config, "MALE_ICON_PATH", None), 22)
        self.female_icon: Optional[np.ndarray] = self._load_icon(
            getattr(config, "FEMALE_ICON_PATH", None), 22)
        self.helmet_green_icon: Optional[np.ndarray] = self._load_icon(
            getattr(config, "HELMET_GREEN_ICON_PATH", None), 40)
        self.helmet_red_icon: Optional[np.ndarray] = self._load_icon(
            getattr(config, "HELMET_RED_ICON_PATH", None), 40)

    @staticmethod
    def _load_icon(path: Optional[Path], size: int) -> Optional[np.ndarray]:
        if path is None or not Path(path).is_file():
            if path is not None:
                print(f"[Renderer] icon missing at {path} — text fallback will be used.")
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img = np.dstack([img, alpha])
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _load_thumbs_up(path: Path) -> Optional[np.ndarray]:
        return Renderer._load_icon(path, 40)

    @staticmethod
    def _alpha_paste(frame: np.ndarray, overlay_rgba: np.ndarray, x: int, y: int) -> None:
        H, W = frame.shape[:2]
        h, w = overlay_rgba.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return

        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        roi = frame[y1:y2, x1:x2]
        ov = overlay_rgba[oy1:oy2, ox1:ox2]
        alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (alpha * ov[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)

    def draw_zone(self, frame: np.ndarray, contour: np.ndarray, flash: bool = False) -> None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [contour], (60, 180, 60))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, dst=frame)
        border_colour = COLOUR_RED if flash else COLOUR_GREEN
        border_thickness = 4 if flash else 2
        cv2.polylines(frame, [contour], isClosed=True, color=border_colour, thickness=border_thickness)

    @staticmethod
    def _square_box(x1: int, y1: int, x2: int, y2: int,
                    frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        side = max(x2 - x1, y2 - y1)
        half = side // 2
        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(frame_w - 1, cx + half)
        sy2 = min(frame_h - 1, cy + half)
        return sx1, sy1, sx2, sy2

    def draw_track(
        self,
        frame: np.ndarray,
        track: Track,
        in_zone: bool,
        gender_label: str,
        helmet_label: str,
        flash: bool = False,
    ) -> None:
        H, W = frame.shape[:2]
        ox1, oy1, ox2, oy2 = track.box
        colour = COLOUR_RED if flash else (COLOUR_GREEN if in_zone else COLOUR_WHITE)

        if helmet_label == "helmet":
            x1, y1, x2, y2 = self._square_box(ox1, oy1, ox2, oy2, W, H)
        else:
            x1, y1, x2, y2 = ox1, oy1, ox2, oy2
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if helmet_label in ("helmet", "no_helmet"):
            icon = self.helmet_green_icon if helmet_label == "helmet" else self.helmet_red_icon
            if icon is not None:
                icon_w, icon_h = icon.shape[1], icon.shape[0]
                ix = min(W - icon_w, x2 - icon_w // 2)
                iy = max(0, y1 - icon_h - 4)
                self._alpha_paste(frame, icon, ix, iy)
            else:
                fb_colour = COLOUR_GREEN if helmet_label == "helmet" else COLOUR_RED
                fb_text   = "H" if helmet_label == "helmet" else "NH"
                cv2.putText(frame, fb_text, (x2 - 20, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, fb_colour, 2)

        if config.ANONYMISE_OVERLAY:
            gender_token = "?"
            extra = ""
        else:
            parts = gender_label.split(maxsplit=1)
            gender_token = parts[0] if parts else "?"
            extra = parts[1] if len(parts) > 1 else ""

        icon = None
        if gender_token == "Male":
            icon = self.male_icon
        elif gender_token == "Female":
            icon = self.female_icon

        if icon is not None:
            text = f"#{track.id}  {extra}".rstrip()
        else:
            text = f"#{track.id}  {gender_token}"
            if extra:
                text += f" {extra}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        icon_w = icon.shape[1] + 4 if icon is not None else 0
        icon_h = icon.shape[0] if icon is not None else 0
        bar_w = tw + 8 + icon_w
        bar_h = max(th + 10, icon_h + 4)
        bx1, by1 = x1, y1
        bx2, by2 = min(W - 1, x1 + bar_w), y1 + bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), colour, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

        text_x = x1 + 4
        if icon is not None:
            iy = by1 + (bar_h - icon_h) // 2
            self._alpha_paste(frame, icon, text_x, iy)
            text_x += icon_w

        text_y = by1 + (bar_h + th) // 2 - 1
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_BLACK, 1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        stats: ZoneStats,
        fps: float,
        latencies_ms: dict[str, float] | None = None,
    ) -> None:
        H, W = frame.shape[:2]
        panel_w = 320
        panel_h = 110 if not config.SHOW_DEBUG_HUD else 170
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOUR_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        lines = [
            f"Total entered:   {stats.total_count}",
            f"Inside now:      {stats.currently_inside}",
            f"Avg dwell:       {stats.average_dwell_seconds:.1f}s",
            f"Longest dwell:   {stats.longest_dwell_seconds:.1f}s",
        ]
        if config.SHOW_DEBUG_HUD:
            lines.append(f"FPS:             {fps:.1f}")
            if latencies_ms:
                joined = " ".join(f"{k}:{v:.0f}ms" for k, v in latencies_ms.items())
                lines.append(joined)

        y = 32
        for line in lines:
            cv2.putText(frame, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_WHITE, 1, cv2.LINE_AA)
            y += 22

    def draw_face_track(
        self,
        frame: np.ndarray,
        det,
        gender_label: str,
        emotion_label: str,
        emotion_score: float,
        all_scores: list[tuple[str, float]] | None = None,
        face_id: int | None = None,
    ) -> None:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        colour = EMOTION_COLOURS.get(emotion_label, COLOUR_WHITE)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        if config.ANONYMISE_OVERLAY:
            gender_token = "?"
            extra = ""
        else:
            parts = (gender_label or "?").split(maxsplit=1)
            gender_token = parts[0] if parts else "?"
            extra = parts[1] if len(parts) > 1 else ""

        icon = None
        if gender_token == "Male":
            icon = self.male_icon
        elif gender_token == "Female":
            icon = self.female_icon

        id_part = f"#{face_id}  " if face_id is not None else ""
        emo_part = f"{emotion_label} {int(round(emotion_score * 100))}%" \
            if emotion_label != "Unknown" else "—"
        if icon is not None:
            text = f"{id_part}{extra}  {emo_part}".strip()
        else:
            text = f"{id_part}{gender_token}  {emo_part}".strip()

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        icon_w = icon.shape[1] + 4 if icon is not None else 0
        icon_h = icon.shape[0] if icon is not None else 0
        bar_w = tw + 12 + icon_w
        bar_h = max(th + 12, icon_h + 6)
        bx1, by1 = x1, max(0, y1 - bar_h - 2)
        bx2, by2 = min(W - 1, x1 + bar_w), max(0, y1 - 2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), colour, -1)

        text_x = bx1 + 6
        if icon is not None:
            iy = by1 + (bar_h - icon_h) // 2
            self._alpha_paste(frame, icon, text_x, iy)
            text_x += icon_w
        text_y = by1 + (bar_h + th) // 2 - 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_BLACK, 1, cv2.LINE_AA)

        if (
            getattr(config, "SHOW_EMOTION_BARS", True)
            and all_scores
            and (x2 - x1) > 100
        ):
            self._draw_emotion_bars(frame, all_scores, x2 + 6, y1, height=y2 - y1)

    @staticmethod
    def _draw_emotion_bars(
        frame: np.ndarray,
        all_scores: list[tuple[str, float]],
        x: int,
        y: int,
        height: int,
        width: int = 110,
    ) -> None:
        H, W = frame.shape[:2]
        n = len(all_scores)
        if n == 0:
            return
        bar_h = max(8, min(18, height // n - 2))
        panel_h = (bar_h + 2) * n + 4
        panel_w = width
        if x + panel_w > W:
            x = max(0, W - panel_w)
        if y + panel_h > H:
            y = max(0, H - panel_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), COLOUR_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        cy = y + 4
        max_bar = panel_w - 60
        ordered = sorted(all_scores, key=lambda p: p[0])
        for label, score in ordered:
            colour = EMOTION_COLOURS.get(label, COLOUR_WHITE)
            cv2.putText(frame, label[:4], (x + 4, cy + bar_h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_WHITE, 1, cv2.LINE_AA)
            bw = int(max_bar * max(0.0, min(1.0, score)))
            cv2.rectangle(frame, (x + 38, cy + 2),
                          (x + 38 + bw, cy + bar_h - 1), colour, -1)
            cy += bar_h + 2

    def draw_face_hud(
        self,
        frame: np.ndarray,
        emotion_history: list[str],
        dominant_emotion: str,
        n_faces: int,
        fps: float,
        latencies_ms: dict[str, float] | None = None,
    ) -> None:
        H, W = frame.shape[:2]
        panel_w = 320
        panel_h = 160 if not config.SHOW_DEBUG_HUD else 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                      COLOUR_BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        glyph = EMOTION_GLYPHS.get(dominant_emotion, "")
        lines = [
            f"Faces:           {n_faces}",
            f"Dominant:        {dominant_emotion} {glyph}",
        ]
        if emotion_history:
            from collections import Counter
            counts = Counter(emotion_history)
            total = sum(counts.values())
            top3 = counts.most_common(3)
            for lbl, cnt in top3:
                pct = 100 * cnt / total
                lines.append(f"  {lbl:<10} {pct:5.1f}%")
        if config.SHOW_DEBUG_HUD:
            lines.append(f"FPS:             {fps:.1f}")
            if latencies_ms:
                joined = " ".join(f"{k}:{v:.0f}ms" for k, v in latencies_ms.items())
                lines.append(joined)

        y = 32
        for line in lines:
            cv2.putText(frame, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_WHITE, 1, cv2.LINE_AA)
            y += 22
