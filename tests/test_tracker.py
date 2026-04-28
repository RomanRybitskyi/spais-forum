from __future__ import annotations

from modules.tracker import SortTracker


def test_track_id_persists_across_frames():
    tracker = SortTracker(min_hits=1, max_age=5, iou_threshold=0.1)
    box = [100.0, 100.0, 200.0, 300.0, 0.9]
    ids = []
    for _ in range(5):
        tracks = tracker.update([box])
        ids.append(tracks[0].id)
    assert len(set(ids)) == 1


def test_two_distinct_objects_get_distinct_ids():
    tracker = SortTracker(min_hits=1, max_age=5, iou_threshold=0.1)
    a = [100.0, 100.0, 200.0, 300.0, 0.9]
    b = [400.0, 100.0, 500.0, 300.0, 0.9]
    tracks = tracker.update([a, b])
    tracks = tracker.update([a, b])
    ids = {t.id for t in tracks}
    assert len(ids) == 2


def test_track_drops_after_max_age():
    tracker = SortTracker(min_hits=1, max_age=2, iou_threshold=0.1)
    box = [100.0, 100.0, 200.0, 300.0, 0.9]
    tracker.update([box])
    for _ in range(5):
        tracker.update([])
    assert len(tracker.trackers) == 0
