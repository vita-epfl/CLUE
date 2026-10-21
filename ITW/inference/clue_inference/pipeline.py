"""Image and video pipelines: detect people, (videos) track them, run SAM 3D Body, SAM 3D Body (CLUE) and/or CLUE,
colour every vertex of the CLUE models' meshes by its uncertainty percentile, render
Input | SAM 3D Body (grey) | SAM 3D Body (CLUE) | CLUE side by side.

Cameras follow the CLUE evaluations: focal length = picture diagonal, principal point = picture centre, taken on the
active picture of letterboxed frames. Video frames are processed independently: only the detection boxes are smoothed
(over 5 frames) by the tracker. A video is read twice, once to detect and track people and once to run the models and
render, so long videos need no more memory than short ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from clue_inference.detect import PersonDetector
from clue_inference.framing import content_rect, stabilise
from clue_inference.model import ASSETS, MODELS, TITLES, Models, cliff_intrinsics
from clue_inference.percentiles import TABLES, PercentileTable
from clue_inference.render import Layout, Rasterizer
from clue_inference.tracking import people_per_frame

VIDEO_BOXES = {"score": 0.7, "overlap": 0.4}       # detect.py: clean boxes for the tracker
DUPLICATE = 0.15                                    # drop_duplicates: same person below this mesh distance


@dataclass
class Report:
    frames: int = 0
    people: int = 0
    tracks: int = 0
    seconds: dict = field(default_factory=dict)

    def summary(self) -> str:
        timing = ", ".join(f"{k} {v:.1f} s" for k, v in self.seconds.items())
        what = f"{self.frames} frames, {self.people} person-frames, {self.tracks} people" if self.frames > 1 else f"{self.people} people"
        return f"{what}; {timing}"


def _project(vertices: np.ndarray, cam: np.ndarray) -> np.ndarray:
    z = np.maximum(-vertices[:, 2], 1e-6)
    return np.stack([cam[0, 0] * vertices[:, 0] / z + cam[0, 2], cam[1, 1] * -vertices[:, 1] / z + cam[1, 2]], 1)


def drop_duplicates(persons: list, cam: np.ndarray, threshold: float = DUPLICATE) -> list:
    """Two boxes on one person give two meshes in the same place: the median image distance between corresponding vertices
    (every 7th), over the mean mesh height, is below 0.05 for such pairs and above 0.3 for two people grappling (the judo
    example). People are kept in order of box area; one whose mesh is within ``threshold`` of a kept one is dropped."""
    uv = [_project(p.vertices[::7], cam) for p in persons]
    areas = [float((p.box[2] - p.box[0]) * (p.box[3] - p.box[1])) for p in persons]
    kept = []
    for i in np.argsort(-np.asarray(areas, dtype=np.float64)):
        if all(np.median(np.linalg.norm(uv[i] - uv[j], axis=1)) >= threshold * 0.5 * (np.ptp(uv[i][:, 1]) + np.ptp(uv[j][:, 1]))
               for j in kept):
            kept.append(int(i))
    return [persons[i] for i in sorted(kept)]


def _fit(image: np.ndarray, max_side: int | None) -> np.ndarray:
    h, w = image.shape[:2]
    scale = 1.0 if not max_side else max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2), interpolation=cv2.INTER_AREA)


def _frames(path: str | Path, limit: int | None, max_side: int | None):
    cap = cv2.VideoCapture(str(path))
    count = 0
    while limit is None or count < limit:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        yield _fit(frame, max_side)
    cap.release()


class Pipeline:
    """``models``: a subset of ("sam3dbody", "sam3dbody_clue", "clue"); ``device``: default CUDA when available;
    ``assets``: the folder with the model files (assets.py)."""

    def __init__(self, models: tuple[str, ...] = MODELS, device: str | torch.device | None = None, *,
                 assets: str | Path = ASSETS) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = Models(models, self.device, assets=Path(assets))
        self.detector = PersonDetector(self.device)
        self.tables = {name: PercentileTable(TABLES[name], self.device) for name in self.models.models if name in TABLES}
        self.raster = Rasterizer(self.models.faces, self.device)
        self.titles = ("Input",) + tuple(TITLES[m] for m in self.models.models)

    def _panels(self, frame_bgr: np.ndarray, cam: np.ndarray, rect: np.ndarray, persons) -> list[np.ndarray]:
        """One panel per model: SAM 3D Body's meshes in grey; SAM 3D Body (CLUE)'s (the same meshes) and CLUE's coloured by
        uncertainty percentile, each against its own reference."""
        panels = []
        for name in self.models.models:
            if not persons:
                panels.append(frame_bgr.copy())
            elif name == "sam3dbody":
                panels.append(self.raster(frame_bgr, cam, [(p.sam3d, None) for p in persons], rect))
            else:
                mesh, sigma = ("sam3d", "sam3d_sigma_cm") if name == "sam3dbody_clue" else ("clue", "clue_sigma_cm")
                u = self.tables[name](np.stack([getattr(p, sigma) for p in persons]))
                panels.append(self.raster(frame_bgr, cam, [(getattr(p, mesh), u[i]) for i, p in enumerate(persons)], rect))
        return panels

    def image(self, image_rgb: np.ndarray, max_side: int | None = 1600) -> tuple[np.ndarray, Report]:
        """RGB image -> RGB panels (Input | SAM 3D Body | SAM 3D Body (CLUE) | CLUE, or the chosen models) + report."""
        report, start = Report(frames=1), time.perf_counter()
        rgb = _fit(np.ascontiguousarray(image_rgb[..., :3]), max_side)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = rgb.shape[:2]
        rect = content_rect(bgr)
        cam = cliff_intrinsics(w, h, rect)
        boxes = self.detector([rgb])[0]
        report.seconds["detect"] = time.perf_counter() - start
        t = time.perf_counter()
        persons = self.models.predict([rgb], [[(i, b[:4]) for i, b in enumerate(boxes)]], [cam])[0]
        persons = drop_duplicates(persons, cam)
        report.seconds["models"] = time.perf_counter() - t
        t = time.perf_counter()
        panels = self._panels(bgr, cam, rect, persons)
        report.seconds["render"] = time.perf_counter() - t
        report.people = report.tracks = len(persons)
        return cv2.cvtColor(Layout(w, h, self.titles)(bgr, *panels), cv2.COLOR_BGR2RGB), report

    def video(self, path: str | Path, out_path: str | Path, *, max_seconds: float | None = None, max_side: int | None = 1280,
              crf: int = 20, chunk: int = 64) -> Report:
        """Video file -> H.264 panel video at ``out_path`` (no audio) + report; only the first ``max_seconds`` when given."""
        import imageio_ffmpeg

        report, start = Report(), time.perf_counter()
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        limit = int(round(max_seconds * fps)) if max_seconds else None

        detections, rects, batch, size = [], [], [], None
        for frame in _frames(path, limit, max_side):
            size = frame.shape[1], frame.shape[0]
            rects.append(content_rect(frame))
            batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(batch) == 8:
                detections += self.detector(batch, **VIDEO_BOXES); batch.clear()
        if batch:
            detections += self.detector(batch, **VIDEO_BOXES)
        if not rects:
            raise ValueError(f"could not read any frame from {path}")
        people = people_per_frame(detections, min_length=min(30, max(len(rects) // 2, 1)), smooth_window=5)
        pictures = stabilise(np.stack(rects))
        w, h = size
        report.frames, report.people = len(rects), sum(len(p) for p in people)
        report.tracks = len({t for p in people for t, _ in p})
        report.seconds["detect + track"] = time.perf_counter() - start

        layout = Layout(w, h, self.titles)
        writer = imageio_ffmpeg.write_frames(str(out_path), (layout.canvas.shape[1], layout.canvas.shape[0]), fps=fps, codec="libx264",
                                             pix_fmt_in="bgr24", pix_fmt_out="yuv420p", macro_block_size=2,
                                             output_params=["-crf", str(crf), "-preset", "veryfast"])
        writer.send(None)
        report.seconds["models"] = report.seconds["render + encode"] = 0.0
        frames, first = [], 0

        def flush() -> None:
            t = time.perf_counter()
            cams = [cliff_intrinsics(w, h, rects[first + k]) for k in range(len(frames))]
            persons = self.models.predict([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], people[first:first + len(frames)], cams)
            report.seconds["models"] += time.perf_counter() - t
            t = time.perf_counter()
            for k, frame in enumerate(frames):
                writer.send(np.ascontiguousarray(layout(frame, *self._panels(frame, cams[k], pictures[first + k], persons[k]))).tobytes())
            report.seconds["render + encode"] += time.perf_counter() - t

        try:
            for frame in _frames(path, len(rects), max_side):
                frames.append(frame)
                if len(frames) == chunk:
                    flush()
                    first += len(frames); frames.clear()
            if frames:
                flush()
        finally:
            writer.close()
        return report
