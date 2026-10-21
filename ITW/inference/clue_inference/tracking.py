"""Turn per-frame person detections into smooth per-person box tracks.

A greedy IoU tracker is enough here: it gives every person a stable identity, drops detections that do not persist
(spurious boxes), bridges short detector misses by interpolation, and lightly smooths the boxes so the crops fed to
the model do not jitter. The MODEL OUTPUT is never smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    boxes: dict[int, np.ndarray] = field(default_factory=dict)   # frame index -> xyxy

    @property
    def last_frame(self) -> int:
        return max(self.boxes)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def link(detections: list[np.ndarray], *, iou_threshold: float = 0.3, max_gap: int = 15) -> list[Track]:
    """Greedy frame-to-frame association: best IoU first, a track survives ``max_gap`` frames without a detection."""
    tracks: list[Track] = []
    for frame, dets in enumerate(detections):
        live = [t for t in tracks if frame - t.last_frame <= max_gap]
        pairs = sorted(((iou(t.boxes[t.last_frame], d[:4]), ti, di) for ti, t in enumerate(live) for di, d in enumerate(dets)), reverse=True)
        used_tracks, used_dets = set(), set()
        for score, ti, di in pairs:
            if score < iou_threshold:
                break
            if ti in used_tracks or di in used_dets:
                continue
            live[ti].boxes[frame] = dets[di, :4]; used_tracks.add(ti); used_dets.add(di)
        for di, d in enumerate(dets):
            if di not in used_dets:
                tracks.append(Track(len(tracks), {frame: d[:4]}))
    return tracks


def densify(track: Track, window: int = 5) -> dict[int, np.ndarray]:
    """Boxes for every frame between the track's first and last detection: gaps linearly interpolated, then a
    centred moving average (zero lag) over ``window`` frames."""
    frames = np.asarray(sorted(track.boxes)); boxes = np.stack([track.boxes[f] for f in frames])
    every = np.arange(frames[0], frames[-1] + 1)
    dense = np.stack([np.interp(every, frames, boxes[:, c]) for c in range(4)], 1)
    if window > 1 and len(dense) >= window:
        pad = window // 2
        padded = np.pad(dense, ((pad, pad), (0, 0)), mode="edge")
        dense = np.stack([np.convolve(padded[:, c], np.ones(window) / window, mode="valid") for c in range(4)], 1)
    return {int(f): dense[i].astype(np.float32) for i, f in enumerate(every)}


def people_per_frame(detections: list[np.ndarray], *, min_length: int = 30, smooth_window: int = 5, **link_kwargs) -> list[list[tuple[int, np.ndarray]]]:
    """For every frame, the (track id, xyxy box) of each person who is detected in at least ``min_length`` frames."""
    per_frame: list[list[tuple[int, np.ndarray]]] = [[] for _ in detections]
    for track in link(detections, **link_kwargs):
        if len(track.boxes) < min_length:
            continue
        for frame, box in densify(track, smooth_window).items():
            per_frame[frame].append((track.track_id, box))
    return per_frame
