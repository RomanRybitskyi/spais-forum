from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from modules.tracker import Track


@dataclass
class ZoneStats:
    total_count: int = 0
    currently_inside: int = 0
    average_dwell_seconds: float = 0.0
    longest_dwell_seconds: float = 0.0


@dataclass
class ZoneAnalytics:
    polygon: list[tuple[int, int]]
    entered_ids: set[int] = field(default_factory=set)
    entry_times: dict[int, float] = field(default_factory=dict)
    completed_dwells: list[float] = field(default_factory=list)
    last_seen: dict[int, float] = field(default_factory=dict)
    stale_timeout: float = 2.0

    def __post_init__(self) -> None:
        self._contour = np.array(self.polygon, dtype=np.int32).reshape(-1, 1, 2)

    def set_polygon(self, polygon: list[tuple[int, int]]) -> None:
        if not polygon or len(polygon) < 3:
            return
        self.polygon = [(int(x), int(y)) for x, y in polygon]
        self._contour = np.array(self.polygon, dtype=np.int32).reshape(-1, 1, 2)

    def is_inside(self, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self._contour, (float(point[0]), float(point[1])), False) >= 0

    def bbox_in_zone(self, bbox: tuple[int, int, int, int], threshold: float = 0.5) -> bool:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bbox_area = max((x2 - x1) * (y2 - y1), 1)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        shifted = self._contour - np.array([[[x1, y1]]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 1)
        intersection = int(mask.sum())
        return (intersection / bbox_area) >= threshold

    def update(self, tracks: Iterable[Track]) -> None:
        now = time.time()
        seen_ids: set[int] = set()

        for trk in tracks:
            seen_ids.add(trk.id)
            self.last_seen[trk.id] = now
            inside = self.bbox_in_zone(trk.box)

            if inside:
                if trk.id not in self.entry_times:
                    self.entry_times[trk.id] = now
                    if trk.id not in self.entered_ids:
                        self.entered_ids.add(trk.id)
            else:
                if trk.id in self.entry_times:
                    dwell = now - self.entry_times.pop(trk.id)
                    self.completed_dwells.append(dwell)

        for tid in list(self.entry_times.keys()):
            if (now - self.last_seen.get(tid, 0)) > self.stale_timeout:
                dwell = now - self.entry_times.pop(tid)
                self.completed_dwells.append(dwell)

    def stats(self) -> ZoneStats:
        now = time.time()
        ongoing = [now - t for t in self.entry_times.values()]
        all_dwells = self.completed_dwells + ongoing
        avg = float(np.mean(all_dwells)) if all_dwells else 0.0
        longest = float(np.max(all_dwells)) if all_dwells else 0.0
        return ZoneStats(
            total_count=len(self.entered_ids),
            currently_inside=len(self.entry_times),
            average_dwell_seconds=avg,
            longest_dwell_seconds=longest,
        )

    @property
    def contour(self) -> np.ndarray:
        return self._contour

