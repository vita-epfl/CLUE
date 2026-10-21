"""Letterboxed sources: curated compilations put 16:9 and 9:16 clips on one square canvas with black bars.

The ACTIVE PICTURE is what the camera saw, so it defines the camera approximation, and the meshes are only drawn
inside it — a person cut off by the clip edge is cut off in the overlay too, not painted over the bars.
"""

from __future__ import annotations

import numpy as np

BLACK_LEVEL = 24        # above compression noise in the bars, below any real picture content


def content_rect(frame_bgr: np.ndarray) -> np.ndarray:
    """(x0, y0, x1, y1) of the active picture: the bounding box of the non-black pixels (the full frame if none)."""
    bright = frame_bgr.max(axis=2) > BLACK_LEVEL
    rows, cols = np.flatnonzero(bright.any(axis=1)), np.flatnonzero(bright.any(axis=0))
    height, width = frame_bgr.shape[:2]
    if len(rows) < 2 or len(cols) < 2:
        return np.asarray([0, 0, width, height], np.int32)
    return np.asarray([cols[0], rows[0], cols[-1] + 1, rows[-1] + 1], np.int32)


def stabilise(rects: np.ndarray, min_share: float = 0.01, slack: int = 2) -> np.ndarray:
    """Snap every frame's rectangle to its clip's true layout. Dark picture edges can only SHRINK a detected
    rectangle, so each frame takes the smallest of the video's common layouts (those seen in at least ``min_share``
    of the frames) that contains its detection; a frame that fits none keeps its own. All frames of a clip then have
    the same size."""
    layouts, counts = np.unique(rects, axis=0, return_counts=True)
    layouts = layouts[counts >= max(min_share * len(rects), 1)]
    areas = (layouts[:, 2] - layouts[:, 0]) * (layouts[:, 3] - layouts[:, 1])
    layouts = layouts[np.argsort(areas)]                                     # smallest first
    out = rects.copy()
    for i, r in enumerate(rects):
        inside = (layouts[:, 0] <= r[0] + slack) & (layouts[:, 1] <= r[1] + slack) & (layouts[:, 2] >= r[2] - slack) & (layouts[:, 3] >= r[3] - slack)
        if inside.any():
            out[i] = layouts[np.flatnonzero(inside)[0]]
    return out


def content_intrinsics(rect: np.ndarray) -> np.ndarray:
    """The CLIFF approximation of the evaluations (focal = picture diagonal), for the active picture: principal
    point at its centre, in full-frame pixel coordinates."""
    x0, y0, x1, y1 = (float(v) for v in rect)
    focal = float(np.hypot(x1 - x0, y1 - y0))
    return np.asarray([[focal, 0.0, (x0 + x1) / 2.0], [0.0, focal, (y0 + y1) / 2.0], [0.0, 0.0, 1.0]], np.float32)
