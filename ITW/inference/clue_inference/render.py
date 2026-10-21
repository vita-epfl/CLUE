"""Rendering: a z-buffer triangle rasteriser in torch (GPU or CPU) and the panel layout
(Input | SAM 3D Body | SAM 3D Body (CLUE) | CLUE).

Faces are flat-shaded and coloured by their uncertainty percentile on the Turbo scale, with the face colouring of the
CLUE videos (mean percentile of the face's vertices, diffuse light from (0.2, -0.4, -1), shading 0.75 + 0.25 |n.l|);
meshes without uncertainty (SAM 3D Body) are neutral grey, so they never read as a Turbo colour.
The mesh is rasterised at ``supersample`` x the frame size; the share of covered sub-pixels then blends it over the
untouched frame, so edges are anti-aliased and the background stays exactly the input.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SHADE = (0.75, 0.25)
GREY_BGR = (185, 185, 185)
INK, MUTED, PAPER = (30, 30, 30), (105, 105, 105), (255, 255, 255)
TITLE_HEIGHT, GAP, BAR_COLUMN = 64, 6, 170
_FACE_BITS = 24                                  # packed z-buffer key: depth (0.1 mm steps) << 24 | face index
_MAX_CANDIDATES = 1 << 24                        # (face, pixel) candidates per chunk


def _turbo_lut_bgr() -> np.ndarray:
    """256-entry BGR LUT of Google's Turbo colormap (quintic approximation)."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    r = 0.13572138 + 4.61539260 * x - 42.66032258 * x**2 + 132.13108234 * x**3 - 152.94239396 * x**4 + 59.28637943 * x**5
    g = 0.09140261 + 2.19418839 * x + 4.84296658 * x**2 - 14.18503333 * x**3 + 4.27729857 * x**4 + 2.82956604 * x**5
    b = 0.10667330 + 12.64194608 * x - 60.58204836 * x**2 + 110.36276771 * x**3 - 89.90310912 * x**4 + 27.34824973 * x**5
    return np.round(np.clip(np.stack([b, g, r], axis=1), 0.0, 1.0) * 255.0).astype(np.uint8)


TURBO = _turbo_lut_bgr()


def _edge(a: torch.Tensor, b: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Signed area of (a, b, p) for p = (px, py); a, b: (n, 2), px, py: (n, k)."""
    return (b[:, 0:1] - a[:, 0:1]) * (py - a[:, 1:2]) - (b[:, 1:2] - a[:, 1:2]) * (px - a[:, 0:1])


class Rasterizer:
    def __init__(self, faces: np.ndarray, device: torch.device) -> None:
        self.device = torch.device(device)
        self.faces = torch.as_tensor(np.asarray(faces), dtype=torch.long, device=self.device)
        self.lut = torch.as_tensor(TURBO, dtype=torch.float32, device=self.device)
        self.grey = torch.tensor(GREY_BGR, dtype=torch.float32, device=self.device)
        light = torch.tensor([0.2, -0.4, -1.0], device=self.device)
        self.light = light / light.norm()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, cam_int: np.ndarray, people: list[tuple[np.ndarray, np.ndarray | None]],
                 rect: np.ndarray | None = None, supersample: int = 2) -> np.ndarray:
        """``people``: (vertices (V, 3) metres, y up, visible depth at negative z; percentile u (V,) in [0, 1], or None for
        all people: grey) per person. Meshes are drawn only inside ``rect`` (x0, y0, x1, y1), the active picture of
        letterboxed frames."""
        if not people:
            return image_bgr.copy()
        dev, s = self.device, int(supersample)
        h, w = image_bgr.shape[:2]
        sh, sw = h * s, w * s
        verts = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v, _ in people]).to(dev)       # (P, V, 3)
        grey = people[0][1] is None
        count, num_v = verts.shape[:2]
        faces = (self.faces[None] + num_v * torch.arange(count, device=dev)[:, None, None]).reshape(-1, 3)
        verts = verts.reshape(-1, 3)
        cam = torch.as_tensor(np.asarray(cam_int, np.float32), device=dev) * torch.tensor([[s], [s], [1.0]], device=dev)
        depth = -verts[:, 2]
        safe = depth.clamp_min(1e-6)
        xy = torch.stack([cam[0, 0] * verts[:, 0] / safe + cam[0, 2], cam[1, 1] * -verts[:, 1] / safe + cam[1, 2]], 1)

        tri, tri_depth, tri3 = xy[faces], depth[faces], verts[faces]                                # (F, 3, 2), (F, 3), (F, 3, 3)
        normals = torch.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0], dim=1)
        normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
        shading = SHADE[0] + SHADE[1] * (normals @ self.light).abs()
        if grey:
            colour = self.grey[None] * shading[:, None]                                               # (F, 3) BGR
        else:
            u = torch.stack([torch.as_tensor(x, dtype=torch.float32) for _, x in people]).to(dev).reshape(-1)
            colour = self.lut[(u[faces].mean(1) * 255.0).round().clamp(0, 255).long()] * shading[:, None]

        lo, hi = tri.min(1).values, tri.max(1).values
        x0, y0 = lo[:, 0].floor().clamp(0, sw - 1).long(), lo[:, 1].floor().clamp(0, sh - 1).long()
        x1, y1 = hi[:, 0].ceil().clamp(0, sw - 1).long(), hi[:, 1].ceil().clamp(0, sh - 1).long()
        area = _edge(tri[:, 0], tri[:, 1], tri[:, 2, 0:1], tri[:, 2, 1:2])[:, 0]
        valid = (tri_depth > 1e-6).all(1) & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < sw) & (lo[:, 1] < sh) & (area.abs() > 1e-9)
        size = torch.maximum(x1 - x0, y1 - y0) + 1
        zbuf = torch.full((sh * sw,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=dev)
        previous = 0
        box = 2
        while previous < max(sh, sw):
            index = torch.nonzero(valid & (size > previous) & (size <= box))[:, 0]
            if len(index):
                oy, ox = torch.meshgrid(torch.arange(box, device=dev), torch.arange(box, device=dev), indexing="ij")
                oy, ox = oy.reshape(1, -1), ox.reshape(1, -1)
                for start in range(0, len(index), max(1, _MAX_CANDIDATES // (box * box))):
                    f = index[start:start + _MAX_CANDIDATES // (box * box)]
                    px, py = x0[f, None] + ox, y0[f, None] + oy                                       # (n, box * box)
                    inside = (px <= x1[f, None]) & (py <= y1[f, None])
                    cx, cy = px.float() + 0.5, py.float() + 0.5
                    a, b, c = tri[f, 0], tri[f, 1], tri[f, 2]
                    l0, l1, l2 = (_edge(b, c, cx, cy) / area[f, None], _edge(c, a, cx, cy) / area[f, None],
                                  _edge(a, b, cx, cy) / area[f, None])
                    inside &= (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                    z = l0 * tri_depth[f, 0:1] + l1 * tri_depth[f, 1:2] + l2 * tri_depth[f, 2:3]
                    key = ((z * 1e4).long() << _FACE_BITS) | f[:, None]
                    zbuf.scatter_reduce_(0, (py * sw + px)[inside], key[inside], reduce="amin")
            previous, box = box, box * 2

        covered = zbuf != torch.iinfo(torch.int64).max
        face_at = (zbuf & ((1 << _FACE_BITS) - 1)).clamp_max(len(faces) - 1)
        mesh = torch.where(covered[:, None], colour[face_at], torch.zeros((), device=dev)).reshape(sh, sw, 3)
        mesh = mesh.permute(2, 0, 1)[None]
        alpha = F.avg_pool2d(covered.reshape(1, 1, sh, sw).float(), s)                                # (1, 1, h, w)
        mesh = F.avg_pool2d(mesh, s) / alpha.clamp_min(1e-6)
        frame = torch.as_tensor(image_bgr, device=dev).permute(2, 0, 1)[None].float()
        out = frame * (1 - alpha) + mesh * alpha
        if rect is not None:                                                                           # draw inside the picture only
            x_lo, y_lo, x_hi, y_hi = (int(v) for v in rect)
            keep = torch.zeros((1, 1, h, w), dtype=torch.bool, device=dev)
            keep[..., y_lo:y_hi, x_lo:x_hi] = True
            out = torch.where(keep, out, frame)
        return out[0].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()


def _text(canvas: np.ndarray, text: str, cx: int, y: int, scale: float, colour, thickness: int = 2) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(canvas, text, (int(cx - tw / 2), y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def colorbar_column(height: int, label=("uncertainty", "percentile")) -> np.ndarray:
    column = np.full((height, BAR_COLUMN, 3), 255, np.uint8)
    top, bottom, x0, x1 = int(0.12 * height), int(0.88 * height), 36, 76
    ramp = np.linspace(1.0, 0.0, max(bottom - top, 1))
    column[top:bottom, x0:x1] = TURBO[np.round(ramp * 255).astype(np.uint8)][:, None, :]
    cv2.rectangle(column, (x0, top), (x1, bottom), MUTED, 1)
    for value in (0, 25, 50, 75, 100):
        y = int(bottom - value / 100.0 * (bottom - top))
        cv2.line(column, (x1, y), (x1 + 8, y), MUTED, 1, cv2.LINE_AA)
        cv2.putText(column, str(value), (x1 + 14, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 1, cv2.LINE_AA)
    _text(column, label[0], BAR_COLUMN // 2, max(top - 46, 20), 0.7, INK, 1)
    _text(column, label[1], BAR_COLUMN // 2, max(top - 18, 44), 0.7, INK, 1)
    return column


class Layout:
    """Panels side by side with titles (Input | SAM 3D Body | SAM 3D Body (CLUE) | CLUE by default) and, when a panel shows
    uncertainty, the percentile colorbar; the static parts are drawn once per frame size."""

    def __init__(self, width: int, height: int, titles=("Input", "SAM 3D Body", "SAM 3D Body (CLUE)", "CLUE")) -> None:
        self.width, self.height, self.count = width, height, len(titles)
        panels_w = self.count * width + (self.count - 1) * GAP
        colorbar = any("CLUE" in title for title in titles)
        out_w = panels_w + (BAR_COLUMN if colorbar else 0)
        out_w += out_w % 2
        out_h = TITLE_HEIGHT + height + (TITLE_HEIGHT + height) % 2                              # even sizes for yuv420p
        self.canvas = np.full((out_h, out_w, 3), 255, np.uint8)
        scale = min(1.1, max(0.6, width / 900))
        for i, title in enumerate(titles):
            _text(self.canvas, title, i * (width + GAP) + width // 2, TITLE_HEIGHT - 20, scale, INK)
        if colorbar:
            self.canvas[TITLE_HEIGHT:TITLE_HEIGHT + height, panels_w:panels_w + BAR_COLUMN] = colorbar_column(height)

    def __call__(self, *panels_bgr: np.ndarray) -> np.ndarray:
        out = self.canvas.copy()
        for i, panel in enumerate(panels_bgr):
            x = i * (self.width + GAP)
            out[TITLE_HEIGHT:TITLE_HEIGHT + self.height, x:x + self.width] = panel
        return out
