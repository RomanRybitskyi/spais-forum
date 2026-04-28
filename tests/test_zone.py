from __future__ import annotations

import time

from modules.tracker import Track
from modules.zone import ZoneAnalytics


SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def _trk(tid: int, x: int, y: int) -> Track:
    box = (x - 5, y - 20, x + 5, y)
    return Track(id=tid, box=box)


def test_inside_outside_basic():
    zone = ZoneAnalytics(polygon=SQUARE)
    assert zone.is_inside((50, 50))
    assert not zone.is_inside((200, 50))


def test_entry_count_increments_once_per_track():
    zone = ZoneAnalytics(polygon=SQUARE)
    for _ in range(5):
        zone.update([_trk(1, 50, 50)])
    assert zone.stats().total_count == 1


def test_two_tracks_counted():
    zone = ZoneAnalytics(polygon=SQUARE)
    zone.update([_trk(1, 30, 30), _trk(2, 60, 60)])
    assert zone.stats().total_count == 2


def test_dwell_time_recorded_on_exit():
    zone = ZoneAnalytics(polygon=SQUARE)
    zone.update([_trk(1, 50, 50)])
    time.sleep(0.05)
    zone.update([_trk(1, 50, 50)])
    zone.update([_trk(1, 200, 200)])
    s = zone.stats()
    assert s.total_count == 1
    assert s.average_dwell_seconds >= 0.04


def test_stale_track_closes_dwell():
    zone = ZoneAnalytics(polygon=SQUARE, stale_timeout=0.05)
    zone.update([_trk(1, 50, 50)])
    time.sleep(0.1)
    zone.update([])
    assert zone.stats().currently_inside == 0
    assert zone.stats().average_dwell_seconds > 0
