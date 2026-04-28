from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

import config


class TrackState(Enum):
    Tracked = auto()
    Lost    = auto()


def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(bb_test[:, None, 0], bb_gt[None, :, 0])
    yy1 = np.maximum(bb_test[:, None, 1], bb_gt[None, :, 1])
    xx2 = np.minimum(bb_test[:, None, 2], bb_gt[None, :, 2])
    yy2 = np.minimum(bb_test[:, None, 3], bb_gt[None, :, 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt   = (bb_gt[:, 2]  - bb_gt[:, 0])  * (bb_gt[:, 3]  - bb_gt[:, 1])
    union = area_test[:, None] + area_gt[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _iou_cost(dets: np.ndarray, trks: np.ndarray) -> np.ndarray:
    return 1.0 - _iou(dets, trks)


def _hungarian(cost: np.ndarray, thresh: float):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    accept_thresh = 1.0 - thresh
    matches, matched_r, matched_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= accept_thresh:
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
    u_rows = [i for i in range(cost.shape[0]) if i not in matched_r]
    u_cols = [j for j in range(cost.shape[1]) if j not in matched_c]
    return matches, u_rows, u_cols


def _xyxy_to_z(box: np.ndarray) -> np.ndarray:
    w  = box[2] - box[0]
    h  = box[3] - box[1]
    cx = box[0] + w / 2
    cy = box[1] + h / 2
    s  = w * h
    r  = w / max(h, 1e-6)
    return np.array([cx, cy, s, r]).reshape(4, 1)


def _x_to_xyxy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).flatten()
    cx, cy, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    w = np.sqrt(max(s * r, 1e-6))
    h = s / max(w, 1e-6)
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2]).flatten()


class _KalmanBoxTracker:
    _next_id = 1

    def __init__(self, box: np.ndarray, score: float = 1.0) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _xyxy_to_z(box)

        self.id    = _KalmanBoxTracker._next_id
        _KalmanBoxTracker._next_id += 1
        self.score             = score
        self.state             = TrackState.Tracked
        self.time_since_update = 0
        self.hits              = 1
        self.hit_streak        = 1
        self.age               = 0

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _x_to_xyxy(self.kf.x)

    def update(self, box: np.ndarray, score: float = 1.0) -> None:
        self.time_since_update = 0
        self.hits       += 1
        self.hit_streak += 1
        self.score       = score
        self.state       = TrackState.Tracked
        self.kf.update(_xyxy_to_z(box))

    def get_state(self) -> np.ndarray:
        return _x_to_xyxy(self.kf.x)


@dataclass
class Track:
    id:    int
    box:   tuple[int, int, int, int]
    score: float = 1.0

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def bottom_centre(self) -> tuple[int, int]:
        x1, _, x2, y2 = self.box
        return (x1 + x2) // 2, y2

    @property
    def head_centre(self) -> tuple[int, int]:
        x1, y1, x2, _ = self.box
        return (x1 + x2) // 2, y1 + (self.box[3] - y1) // 8


class ByteTracker:

    def __init__(
        self,
        track_high_thresh:   float = config.BYTETRACK_TRACK_HIGH_THRESH,
        track_low_thresh:    float = config.BYTETRACK_TRACK_LOW_THRESH,
        new_track_thresh:    float = config.BYTETRACK_NEW_TRACK_THRESH,
        match_thresh:        float = config.BYTETRACK_MATCH_THRESH,
        second_match_thresh: float = config.BYTETRACK_SECOND_MATCH_THRESH,
        max_age:             int   = config.BYTETRACK_MAX_AGE,
        min_hits:            int   = config.BYTETRACK_MIN_HITS,
        iou_threshold:       float | None = None,
    ) -> None:
        self.track_high_thresh   = track_high_thresh
        self.track_low_thresh    = track_low_thresh
        self.new_track_thresh    = new_track_thresh
        self.match_thresh        = iou_threshold if iou_threshold is not None else match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_age             = max_age
        self.min_hits            = min_hits
        self.trackers:    list[_KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(self, detections: list[list[float]]) -> List[Track]:
        self.frame_count += 1

        dets = np.array(detections, dtype=float) if detections else np.empty((0, 5))

        if dets.size:
            high_mask = dets[:, 4] >= self.track_high_thresh
            low_mask  = (dets[:, 4] >= self.track_low_thresh) & ~high_mask
            high_dets = dets[high_mask]
            low_dets  = dets[low_mask]
        else:
            high_dets = np.empty((0, 5))
            low_dets  = np.empty((0, 5))

        predicted_boxes: list[np.ndarray] = []
        valid_idx: list[int] = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if not np.any(np.isnan(pos)):
                predicted_boxes.append(pos)
                valid_idx.append(i)
        self.trackers = [self.trackers[i] for i in valid_idx]
        pred_arr = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        if high_dets.size and pred_arr.size:
            cost1 = _iou_cost(high_dets[:, :4], pred_arr)
            matches1, u_high, u_trks1 = _hungarian(cost1, self.match_thresh)
        else:
            matches1 = []
            u_high   = list(range(len(high_dets)))
            u_trks1  = list(range(len(self.trackers)))

        for d_i, t_i in matches1:
            self.trackers[t_i].update(high_dets[d_i, :4], float(high_dets[d_i, 4]))

        unmatched_pred = pred_arr[u_trks1] if (pred_arr.size and u_trks1) else np.empty((0, 4))

        if low_dets.size and unmatched_pred.size:
            cost2 = _iou_cost(low_dets[:, :4], unmatched_pred)
            matches2, _, still_u_trks = _hungarian(cost2, self.second_match_thresh)
        else:
            matches2     = []
            still_u_trks = list(range(len(u_trks1)))

        for d_i, idx_in_u in matches2:
            t_i = u_trks1[idx_in_u]
            self.trackers[t_i].update(low_dets[d_i, :4], float(low_dets[d_i, 4]))

        for idx_in_u in still_u_trks:
            self.trackers[u_trks1[idx_in_u]].state = TrackState.Lost

        for d_i in u_high:
            score = float(high_dets[d_i, 4])
            if score >= self.new_track_thresh:
                self.trackers.append(_KalmanBoxTracker(high_dets[d_i, :4], score))

        results: list[Track] = []
        live: list[_KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue
            live.append(trk)
            confirmed = (trk.hits >= self.min_hits or
                         self.frame_count <= self.min_hits)
            if confirmed and trk.time_since_update == 0:
                box = trk.get_state().astype(int)
                results.append(Track(
                    id=trk.id,
                    box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    score=trk.score,
                ))
        self.trackers = live
        return results


SortTracker = ByteTracker
