"""Generate FLAME parameter-sweep videos with CLUE uncertainty overlays.

This script renders short MP4 clips that vary one FLAME parameter at a time.
For each shape, expression, jaw, and neck parameter, the exporter activates the
corresponding closed-form uncertainty term in :class:`algorithm.CLUE`, animates
the parameter through ``0 -> +max -> -max -> 0``, and writes a cropped video
named ``<group>_<index>.mp4``.

The module is intentionally headless: it builds a pyrender scene, renders frames
offscreen, and saves videos to ``results/videos/`` by default. Use
``--quality=default`` for the fast export path or ``--quality=high`` for
higher-quality MP4s.
"""
from __future__ import annotations

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
from dataclasses import dataclass
import io
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _RESAMPLE = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
    try:
        _CBAR_FONT = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        _CBAR_FONT = ImageFont.load_default(size=11)
    except TypeError:                           # Pillow < 10
        _CBAR_FONT = ImageFont.load_default()
except ImportError as exc:
    raise ImportError("Pillow is required: pip install Pillow") from exc

# Try to load the `turbo` colormap for amplitude coloring.
try:
    import matplotlib as _mpl

    try:
        _TURBO = _mpl.colormaps["turbo"]
    except Exception:
        import matplotlib.cm as _mpl_cm2
        _TURBO = _mpl_cm2.get_cmap("turbo")
except Exception:
    _TURBO = None


# ---------------------------------------------------------------------------
# Uncertainty colour blend
# ---------------------------------------------------------------------------
# These must match the PBR material's baseColorFactor so that the vertex-colour
# mesh starts at the same visual colour as the deterministic mesh.
# Use the low end of `turbo` as the base colour when available.
try:
    if _TURBO is not None:
        _BASE_COLOR = np.array((_TURBO(0.0)[:3]), dtype=np.float32)
    else:
        _BASE_COLOR = np.array(
            [0.05, 0.14, 0.28], dtype=np.float32
        )   # dark blue - matches material
except NameError:
    _BASE_COLOR = np.array([0.05, 0.14, 0.28], dtype=np.float32)

_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)   # red – max uncertainty


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    """Colour vertices by uncertainty amplitude.

    t=0 → base mesh colour.
    t=1 → red.
    """
    # Prefer the turbo colormap for amplitude coloring.
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    if _TURBO is not None:
        # _TURBO returns RGBA floats in [0,1]
        return (_TURBO(tt) * 255).astype(np.uint8)
    # Fallback: blend base colour → red
    t_col = tt[:, np.newaxis]
    rgb   = (1.0 - t_col) * _BASE_COLOR + t_col * _AMPLITUDE_HIGH
    alpha = np.ones((len(tt), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


# pyrender 0.1.x uses np.infty which was removed in NumPy 2.0
if not hasattr(np, "infty"):
    np.infty = np.inf

import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config
from algorithm import CLUE


@dataclass(frozen=True)
class ParameterSweep:
    """One parameter animation and its matching CLUE variance entry."""

    group: str
    index: int
    value_min: float
    value_max: float
    variance_attr: str
    variance_max: float

    @property
    def filename(self) -> str:
        return f"{self.group}_{self.index:02d}.mp4"


@dataclass(frozen=True)
class VideoQualityProfile:
    """Render and encoder settings for one export quality level."""

    frames_per_phase: int
    fps: int
    supersample: int
    output_scale: int
    crop_pad: int
    color_boost: float
    contrast_boost: float
    crf: int
    preset: str


QUALITY_PROFILES = {
    "default": VideoQualityProfile(
        frames_per_phase=8,
        fps=12,
        supersample=1,
        output_scale=1,
        crop_pad=8,
        color_boost=1.25,
        contrast_boost=1.05,
        crf=28,
        preset="veryfast",
    ),
    "high": VideoQualityProfile(
        frames_per_phase=12,
        fps=12,
        supersample=3,
        output_scale=2,
        crop_pad=8,
        color_boost=1.25,
        contrast_boost=1.05,
        crf=18,
        preset="slow",
    ),
}


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Fixed orbit camera used for all exported frames."""

    def __init__(self, distance: float, azimuth: float = 0.0, elevation: float = 20.0):
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self._target = np.zeros(3)

    def get_pose(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        cam_pos = self._target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])

        forward = self._target - cam_pos
        flen = np.linalg.norm(forward)
        if flen < 1e-8:
            return np.eye(4)
        forward /= flen

        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        return pose


# ---------------------------------------------------------------------------
# Video exporter
# ---------------------------------------------------------------------------

class FlameVideoExporter:
    SHAPE_PARAMETERS = 10
    EXPRESSION_PARAMETERS = 10

    FRAME_WIDTH = 800
    FRAME_HEIGHT = 800
    BETA_VARIANCE_MAX = 0.5
    POSE_VARIANCE_MAX = 0.1

    def __init__(self):
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log(f"Using device: {self.device.type.upper()}")
        self._log("Loading FLAME model...")
        self.flame = FLAME(self.config).to(self.device)
        self.flame.eval()
        self._log("FLAME model ready.")

        self.shape_params = torch.zeros(
            1, self.config.shape_params, dtype=torch.float32, device=self.device
        )
        self.expression_params = torch.zeros(
            1, self.config.expression_params, dtype=torch.float32, device=self.device
        )
        self.pose_params = torch.zeros(
            1, self.config.pose_params, dtype=torch.float32, device=self.device
        )
        self.neck_pose = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.eye_pose  = torch.zeros(1, 6, dtype=torch.float32, device=self.device)

        self.faces = self.flame.faces
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.55, 0.64, 0.88, 1.0],
            metallicFactor=0.3,
            roughnessFactor=0.52,
            alphaMode="OPAQUE",
            doubleSided=True,
        )
        # Same PBR properties but neutral white base colour so that vertex
        # colours are used directly as the albedo (GPU multiplies vertex colour
        # by baseColorFactor; white = identity → vertex colour passes through).
        self.vc_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.3,
            roughnessFactor=0.52,
            alphaMode="OPAQUE",
            doubleSided=True,
        )

        # Variance propagation
        self.clue = CLUE(self.flame)

        # Total beta counts (from flame.py constants: 300 shape, 100 expression).
        self._shape_beta_total: int = (
            self.flame.shape_betas.shape[1] + self.config.shape_params
        )
        self._expr_beta_total: int = (
            self.flame.expression_betas.shape[1] + self.config.expression_params
        )

        self.shape_var_params = torch.zeros(
            1, self.SHAPE_PARAMETERS, dtype=torch.float32, device=self.device
        )
        self.expr_var_params = torch.zeros(
            1, self.EXPRESSION_PARAMETERS, dtype=torch.float32, device=self.device
        )
        # pose_vars ordering mirrors full_pose: [global(3) | neck(3) | jaw(3) | eye(6)]
        # neck → joints slot 1, jaw → joints slot 2
        self.neck_var_params = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.jaw_var_params  = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self._var_range: float | None = None         # std_ref_max for colorbar

        # Populated by _precompute_param_data().
        self._param_std_ref: dict = {}
        self._param_group_max_std: dict = {}

        # Scene
        self._log("Building render scene...")
        vertices, joints = self._compute_vertices()
        self.scene, self.mesh_node, self.center, self.cam_node, extent = (
            self._make_scene(vertices, joints)
        )
        self.orbit = OrbitCamera(distance=2.1 * extent)

        # Pre-compute the CLUE ranges used to normalize uncertainty colours.
        self._log("Precomputing CLUE uncertainty ranges...")
        self._precompute_param_data()
        self._log("Exporter initialization complete.")

    # FLAME helpers

    def _log(self, message: str) -> None:
        print(f"[video] {message}", flush=True)

    def _compute_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            vertices, _, joints = self.flame(
                self.shape_params, self.expression_params,
                self.pose_params, self.neck_pose, self.eye_pose,
            )
        return (
            vertices[0].detach().cpu().numpy(),
            joints[0].detach().cpu().numpy(),   # (J, 3)
        )

    def _build_trimesh(self, vertices: np.ndarray, center: np.ndarray) -> trimesh.Trimesh:
        tri_mesh = trimesh.Trimesh(vertices - center, self.faces, process=False)
        for remove_method, face_selector in [
            ("remove_degenerate_faces", "nondegenerate_faces"),
            ("remove_duplicate_faces",  "unique_faces"),
        ]:
            if hasattr(tri_mesh, remove_method):
                getattr(tri_mesh, remove_method)()
            elif hasattr(tri_mesh, face_selector):
                tri_mesh.update_faces(getattr(tri_mesh, face_selector)())
        if hasattr(tri_mesh, "remove_unreferenced_vertices"):
            tri_mesh.remove_unreferenced_vertices()
        tri_mesh.fix_normals()
        return tri_mesh

    def _make_scene(self, vertices: np.ndarray, joints: np.ndarray):
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = max(float(np.max(bounds[1] - bounds[0])), 1e-3)

        tri_mesh = self._build_trimesh(vertices, center)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)

        # Reduce ambient light for a less bright scene
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.12, 0.13, 0.14])
        mesh_node = scene.add(mesh)
        cam_node  = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=np.eye(4))

        def _look_at_light(direction, intensity):
            d = np.asarray(direction, dtype=np.float64)
            d /= np.linalg.norm(d) + 1e-8
            z = -d
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z)
            x /= np.linalg.norm(x) + 1e-8
            y = np.cross(z, x)
            lpose = np.eye(4)
            lpose[:3, 0] = x
            lpose[:3, 1] = y
            lpose[:3, 2] = z
            lpose[:3, 3] = -d * 2.5 * extent
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            scene.add(light, pose=lpose)

        # Directional lights with reduced intensities (dimmer scene)
        _look_at_light([-0.4, -0.55, -1.0], 3.5)
        _look_at_light([ 0.8, -0.2,  -0.6], 1.75)
        _look_at_light([ 0.0,  1.0,  -0.5], 0.75)

        return scene, mesh_node, center, cam_node, extent

    # CLUE uncertainty helpers

    def _variance_active(self) -> bool:
        return bool(
            (self.shape_var_params > 0).any().item()
            or (self.expr_var_params > 0).any().item()
            or (self.neck_var_params > 0).any().item()
            or (self.jaw_var_params  > 0).any().item()
        )

    def _build_beta_vars(self) -> torch.Tensor:
        """Construct the 400-dim beta_vars vector matching FLAME's betas layout:
           [shape(100) | shape_pad(200) | expr(50) | expr_pad(50)]
        """
        n_total   = self.flame.shapedirs.shape[2]          # 400
        beta_vars = torch.zeros(1, n_total, dtype=torch.float32, device=self.device)
        n_sv = self.shape_var_params.shape[1]
        beta_vars[0, :n_sv] = self.shape_var_params[0]
        n_ev = self.expr_var_params.shape[1]
        off  = self._shape_beta_total                       # 300
        beta_vars[0, off : off + n_ev] = self.expr_var_params[0]
        return beta_vars

    def _build_beta_vars_max(self) -> torch.Tensor:
        """Beta-var tensor with every active entry set to BETA_VARIANCE_MAX.

        Used to compute the per-vertex uncertainty ceiling so that actual
        colours can be expressed as a fraction of that ceiling.
        """
        n_total  = self.flame.shapedirs.shape[2]
        bv_max   = torch.zeros(1, n_total, dtype=torch.float32, device=self.device)
        n_sv     = self.shape_var_params.shape[1]
        bv_max[0, :n_sv] = (
            (self.shape_var_params[0] > 0).float() * self.BETA_VARIANCE_MAX
        )
        n_ev     = self.expr_var_params.shape[1]
        off      = self._shape_beta_total
        bv_max[0, off : off + n_ev] = (
            (self.expr_var_params[0] > 0).float() * self.BETA_VARIANCE_MAX
        )
        return bv_max

    def _build_pose_vars(self) -> torch.Tensor:
        """Construct the 15-dim pose_vars vector matching full_pose ordering:
           [global(3) | neck(3) | jaw(3) | eye(6)]
        """
        pose_vars = torch.zeros(1, 15, dtype=torch.float32, device=self.device)
        pose_vars[0, 3:6] = self.neck_var_params[0]   # joint slot 1
        pose_vars[0, 6:9] = self.jaw_var_params[0]    # joint slot 2
        return pose_vars

    def _build_pose_vars_max(self) -> torch.Tensor:
        """Pose-var tensor with every active entry set to POSE_VARIANCE_MAX."""
        pose_vars_max = torch.zeros(1, 15, dtype=torch.float32, device=self.device)
        pose_vars_max[0, 3:6] = (
            (self.neck_var_params[0] > 0).float() * self.POSE_VARIANCE_MAX
        )
        pose_vars_max[0, 6:9] = (
            (self.jaw_var_params[0] > 0).float() * self.POSE_VARIANCE_MAX
        )
        return pose_vars_max

    def _precompute_param_data(self) -> None:
        """Precompute per-vertex sensitivity patterns and per-group max stds.

        For each parameter, CLUE is called at var_max to get the
        spatial std pattern.  The per-group max std is the highest max-vertex-std
        found across all parameters in that group, used to anchor the colorbar.
        """
        n_total = self.flame.shapedirs.shape[2]   # 400 beta blendweights
        n_pose  = 15                               # 5 joints × 3 angles

        # (group_name, n_params, beta_offset_or_None, pose_offset_or_None, var_max)
        groups = [
            (
                "shape",
                self.SHAPE_PARAMETERS,
                0,
                None,
                self.BETA_VARIANCE_MAX,
            ),
            (
                "expr",
                self.EXPRESSION_PARAMETERS,
                self._shape_beta_total,
                None,
                self.BETA_VARIANCE_MAX,
            ),
            ("neck", 3, None, 3, self.POSE_VARIANCE_MAX),
            ("jaw", 3, None, 6, self.POSE_VARIANCE_MAX),
        ]

        total_params = sum(n_params for _, n_params, _, _, _ in groups)
        completed = 0

        for group_name, n_params, beta_off, pose_off, var_max in groups:
            for idx in range(n_params):
                key = (group_name, idx)
                completed += 1
                self._log(
                    f"CLUE range {completed}/{total_params}: {group_name}[{idx}]"
                )

                beta_vars = torch.zeros(1, n_total, dtype=torch.float32, device=self.device)
                pose_vars = torch.zeros(1, n_pose,  dtype=torch.float32, device=self.device)
                if beta_off is not None:
                    beta_vars[0, beta_off + idx] = var_max
                if pose_off is not None:
                    pose_vars[0, pose_off + idx] = var_max

                total_var, _ = self.clue.compute_variance(
                    beta_vars, pose_vars,
                    shape_params=self.shape_params,
                    expression_params=self.expression_params,
                    pose_params=self.pose_params,
                    neck_pose=self.neck_pose,
                    eye_pose=self.eye_pose,
                )
                var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)   # (V, 3)
                std     = np.sqrt(var_xyz.sum(axis=-1))                  # (V,)

                std_max = float(std.max())
                self._param_std_ref[key] = std_max * 1000  # mm

            # Highest max-vertex-std across all params in this group (anchors colorbar top)
            self._param_group_max_std[group_name] = max(
                self._param_std_ref.get((group_name, i), 0.0)
                for i in range(n_params)
            )

    def _compute_per_param_colors(self) -> tuple[np.ndarray, float | None]:
        """Per-group amplitude coloring for CLUE uncertainty.

        Computes actual per-vertex std via CLUE, normalises by the
        precomputed highest-variance group max std, then maps through the
        amplitude colormap.

        Returns
        -------
        vertex_colors : (V, 4) RGBA uint8
        std_ref_max   : colorbar ceiling in mm, or None when no variance is active.
        """
        if not self._variance_active():
            # All uncertainty terms are zero; show the base mesh colour.
            V = int(self.flame.v_template.shape[0])
            t = np.zeros(V, dtype=np.float32)
            return _amplitude_colors(t), None

        # Compute actual per-vertex variance
        common_args = dict(
            shape_params=self.shape_params,
            expression_params=self.expression_params,
            pose_params=self.pose_params,
            neck_pose=self.neck_pose,
            eye_pose=self.eye_pose,
        )
        total_var, _ = self.clue.compute_variance(
            self._build_beta_vars(), self._build_pose_vars(), **common_args
        )
        var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)   # (V, 3)
        std = np.sqrt(var_xyz.sum(axis=-1)) * 1000               # (V,) mm

        # Colorbar ceiling: highest group max std across all active groups
        active_groups: set[str] = set()
        if (self.shape_var_params > 0).any():
            active_groups.add("shape")
        if (self.expr_var_params > 0).any():
            active_groups.add("expr")
        if (self.jaw_var_params > 0).any():
            active_groups.add("jaw")
        if (self.neck_var_params > 0).any():
            active_groups.add("neck")

        std_ref_max = max(
            (self._param_group_max_std.get(g, 1e-12) for g in active_groups),
            default=1e-12,
        )

        t = std / std_ref_max if std_ref_max > 1e-12 else np.zeros_like(std)
        return _amplitude_colors(t), float(std_ref_max)

    # Colorbar

    def _draw_colorbar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
        """Vertical gradient colorbar: base colour (bottom/low) → red (top/high)."""
        draw     = ImageDraw.Draw(img)
        iw, ih   = img.size
        bar_w    = 36
        bar_h    = 240
        margin_r = 14
        margin_t = 14
        pad      = 10
        label_w  = 70   # space for tick labels to the left of the bar

        panel_w = bar_w + label_w + pad * 2
        panel_h = bar_h + pad * 2 + 20   # +20 for title

        px = iw - margin_r - panel_w
        py = margin_t

        draw.rectangle(
            [px, py, px + panel_w, py + panel_h],
            fill=(245, 245, 245), outline=(190, 190, 190),
        )
        draw.text((px + pad, py + pad), "uncertainty", fill=(50, 50, 50), font=_CBAR_FONT)

        bx = px + pad + label_w   # left edge of gradient bar
        by = py + pad + 20        # top edge of gradient bar

        # Draw gradient row by row (top = high, bottom = low) using turbo when available
        for row in range(bar_h):
            t = 1.0 - row / max(bar_h - 1, 1)
            if _TURBO is not None:
                rgba = (_TURBO(np.array([t]))[0] * 255).astype(int)
                r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
            else:
                rgb = (1.0 - t) * _BASE_COLOR + t * _AMPLITUDE_HIGH
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            draw.line([(bx, by + row), (bx + bar_w, by + row)], fill=(r, g, b))
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(80, 80, 80))

        # Tick labels: σ_max at top, 0 mm at bottom
        draw.text(
            (px + pad, by),
            f"{std_ref_max:.2f}mm",
            fill=(30, 30, 30), font=_CBAR_FONT,
        )
        draw.text(
            (px + pad, by + bar_h - 14),
            "0 mm",
            fill=(30, 30, 30), font=_CBAR_FONT,
        )
        return img

    # Mesh update

    def _update_mesh(self) -> None:
        vertices, joints = self._compute_vertices()
        tri_mesh = self._build_trimesh(vertices, self.center)

        # Always render with vertex colours so CLUE uncertainty is visible.
        vertex_colors, std_ref_max = self._compute_per_param_colors()
        self._var_range = std_ref_max

        vc_mat  = self.vc_material
        fallback = self.material

        if len(vertex_colors) == len(tri_mesh.vertices):
            # Build the primitive directly so we can pass vertex colours AND
            # the PBR material simultaneously, preserving the metallic sheen.
            prim = pyrender.Primitive(
                positions=tri_mesh.vertices.astype(np.float32),
                normals=tri_mesh.vertex_normals.astype(np.float32),
                indices=tri_mesh.faces.astype(np.uint32),
                color_0=vertex_colors,          # (V, 4) RGBA uint8
                material=vc_mat,
            )
            mesh = pyrender.Mesh(primitives=[prim])
        else:
            # Fallback for unexpected vertex-count mismatch (shouldn't occur for FLAME)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=fallback, smooth=True)

        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)

    def _render_offscreen_image(
        self,
        renderer: pyrender.OffscreenRenderer,
        target_size: tuple[int, int] | None = None,
        draw_overlays: bool = True,
    ) -> Image.Image:
        """Render the current scene offscreen and return a PIL Image with overlays."""
        try:
            self.scene.set_pose(self.cam_node, self.orbit.get_pose())
            color, _ = renderer.render(self.scene)
        except Exception as exc:
            raise RuntimeError("Offscreen render failed; check the EGL/OpenGL setup.") from exc

        img = Image.fromarray(color)

        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, _RESAMPLE)

        if draw_overlays:
            if self._var_range is not None:
                img = self._draw_colorbar(img, self._var_range)

        return img

    def _tight_crop(self, img: Image.Image, pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
        """Crop a uniform background border (bg) tightly around the content, with padding."""
        arr = np.asarray(img.convert("RGB"))
        # Mask where any channel differs from background beyond small tolerance
        tol = 6
        mask = np.any(np.abs(arr - np.array(bg, dtype=np.uint8)) > tol, axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # PIL crop box: (left, upper, right, lower)
        left = max(0, x0 - pad)
        upper = max(0, y0 - pad)
        right = min(img.width, x1 + pad + 1)
        lower = min(img.height, y1 + pad + 1)
        return img.crop((left, upper, right, lower))

    def _sweep_specs(self) -> list[ParameterSweep]:
        groups = [
            (
                "shape",
                min(self.SHAPE_PARAMETERS, self.config.shape_params),
                -3.0,
                3.0,
                "shape_var_params",
                self.BETA_VARIANCE_MAX,
            ),
            (
                "expr",
                min(self.EXPRESSION_PARAMETERS, self.config.expression_params),
                -3.0,
                3.0,
                "expr_var_params",
                self.BETA_VARIANCE_MAX,
            ),
            ("jaw", 3, -0.25, 0.25, "jaw_var_params", self.POSE_VARIANCE_MAX),
            ("neck", 3, -0.25, 0.25, "neck_var_params", self.POSE_VARIANCE_MAX),
        ]
        return [
            ParameterSweep(group, idx, value_min, value_max, variance_attr, variance_max)
            for group, count, value_min, value_max, variance_attr, variance_max in groups
            for idx in range(count)
        ]

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            "shape_params": self.shape_params.clone(),
            "expression_params": self.expression_params.clone(),
            "pose_params": self.pose_params.clone(),
            "neck_pose": self.neck_pose.clone(),
            "eye_pose": self.eye_pose.clone(),
            "shape_var_params": self.shape_var_params.clone(),
            "expr_var_params": self.expr_var_params.clone(),
            "jaw_var_params": self.jaw_var_params.clone(),
            "neck_var_params": self.neck_var_params.clone(),
        }

    def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
        self.shape_params.copy_(state["shape_params"])
        self.expression_params.copy_(state["expression_params"])
        self.pose_params.copy_(state["pose_params"])
        self.neck_pose.copy_(state["neck_pose"])
        self.eye_pose.copy_(state["eye_pose"])
        self.shape_var_params.copy_(state["shape_var_params"])
        self.expr_var_params.copy_(state["expr_var_params"])
        self.jaw_var_params.copy_(state["jaw_var_params"])
        self.neck_var_params.copy_(state["neck_var_params"])

    def _reset_sweep_state(self) -> None:
        self.shape_params.zero_()
        self.expression_params.zero_()
        self.pose_params.zero_()
        self.neck_pose.zero_()
        self.eye_pose.zero_()
        self.shape_var_params.zero_()
        self.expr_var_params.zero_()
        self.jaw_var_params.zero_()
        self.neck_var_params.zero_()

    def _activate_sweep_uncertainty(self, sweep: ParameterSweep) -> None:
        getattr(self, sweep.variance_attr)[0, sweep.index] = sweep.variance_max

    def _set_sweep_value(self, sweep: ParameterSweep, value: float) -> None:
        if sweep.group == "shape":
            self.shape_params[0, sweep.index] = value
        elif sweep.group == "expr":
            self.expression_params[0, sweep.index] = value
        elif sweep.group == "jaw":
            self.pose_params[0, 3 + sweep.index] = value
        elif sweep.group == "neck":
            self.neck_pose[0, sweep.index] = value
        else:
            raise ValueError(f"Unknown sweep group: {sweep.group}")

    @staticmethod
    def _sweep_values(sweep: ParameterSweep, frames_per_phase: int) -> np.ndarray:
        positive = np.linspace(0.0, sweep.value_max, frames_per_phase, endpoint=False)
        through_negative = np.linspace(
            sweep.value_max, sweep.value_min, frames_per_phase * 2, endpoint=False
        )
        back_to_zero = np.linspace(sweep.value_min, 0.0, frames_per_phase, endpoint=True)
        return np.concatenate([positive, through_negative, back_to_zero])

    def _render_sweep_frames(
        self,
        sweep: ParameterSweep,
        values: np.ndarray,
        renderer: pyrender.OffscreenRenderer,
        target_size: tuple[int, int] | None,
        crop_pad: int,
    ) -> list[Image.Image]:
        frames: list[Image.Image] = []
        pad_scale = (target_size[0] if target_size else self.FRAME_WIDTH) / self.FRAME_WIDTH
        pad = int(max(1, crop_pad * pad_scale))

        for value in values:
            self._set_sweep_value(sweep, float(value))
            self._update_mesh()
            img = self._render_offscreen_image(
                renderer=renderer,
                target_size=target_size,
                draw_overlays=False,
            )
            frames.append(self._tight_crop(img, pad=pad).convert("RGB"))

        return frames

    @staticmethod
    def _normalize_frame_sizes(frames: list[Image.Image]) -> list[Image.Image]:
        max_w = max(frame.width for frame in frames)
        max_h = max(frame.height for frame in frames)

        # libx264 requires even dimensions.
        max_w += max_w % 2
        max_h += max_h % 2

        normalized: list[Image.Image] = []
        for frame in frames:
            if frame.size == (max_w, max_h):
                normalized.append(frame)
                continue

            canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            offset_x = (max_w - frame.width) // 2
            offset_y = (max_h - frame.height) // 2
            canvas.paste(frame, (offset_x, offset_y))
            normalized.append(canvas)

        return normalized

    @staticmethod
    def _validate_export_settings(
        frames_per_phase: int,
        fps: int,
        supersample: int,
        output_scale: int,
        crf: int,
    ) -> None:
        if frames_per_phase < 1:
            raise ValueError("frames_per_phase must be at least 1.")
        if fps < 1:
            raise ValueError("fps must be at least 1.")
        if supersample < 1:
            raise ValueError("supersample must be at least 1.")
        if output_scale < 1:
            raise ValueError("output_scale must be at least 1.")
        if not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51.")

    @staticmethod
    def _resolve_ffmpeg() -> str:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg is required to export MP4 videos. Install it with apt "
                "or provide an ffmpeg binary on PATH."
            )

        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or "libx264" not in result.stdout:
            raise RuntimeError(
                "ffmpeg was found, but the libx264 encoder is unavailable. "
                "Install an ffmpeg build compiled with H.264/libx264 support."
            )

        return ffmpeg_path

    @staticmethod
    def _encode_mp4(
        frames: list[Image.Image],
        output_path: Path,
        fps: int,
        color_boost: float,
        contrast_boost: float,
        ffmpeg_path: str,
        crf: int,
        preset: str,
    ) -> None:
        from PIL import ImageEnhance

        cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            if process.stdin is None:
                raise RuntimeError("Failed to open ffmpeg stdin.")

            for frame in frames:
                image = frame
                if color_boost != 1.0:
                    image = ImageEnhance.Color(image).enhance(color_boost)
                if contrast_boost != 1.0:
                    image = ImageEnhance.Contrast(image).enhance(contrast_boost)

                buf = io.BytesIO()
                image.save(buf, format="PNG")
                process.stdin.write(buf.getvalue())

            process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed while writing {output_path}.")
        except Exception:
            if process.poll() is None:
                process.kill()
            raise

    def create_videos(
        self,
        out_dir: str | Path = "results/videos",
        quality: str = "default",
    ) -> None:
        """Generate one MP4 parameter sweep for each configured FLAME parameter."""
        if quality not in QUALITY_PROFILES:
            valid = ", ".join(sorted(QUALITY_PROFILES))
            raise ValueError(f"quality must be one of: {valid}.")
        profile = QUALITY_PROFILES[quality]
        self._validate_export_settings(
            profile.frames_per_phase,
            profile.fps,
            profile.supersample,
            profile.output_scale,
            profile.crf,
        )

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = self._resolve_ffmpeg()
        self._log(f"Writing videos to: {output_dir}")
        self._log(f"Using ffmpeg: {ffmpeg_path}")
        self._log(
            "Export settings: "
            f"quality={quality}, "
            f"frames_per_phase={profile.frames_per_phase}, "
            f"fps={profile.fps}, "
            f"supersample={profile.supersample}, "
            f"output_scale={profile.output_scale}, "
            f"crf={profile.crf}, preset={profile.preset}"
        )

        target_size = (
            self.FRAME_WIDTH * profile.output_scale,
            self.FRAME_HEIGHT * profile.output_scale,
        )
        render_scale = profile.supersample * profile.output_scale
        renderer = pyrender.OffscreenRenderer(
            self.FRAME_WIDTH * render_scale,
            self.FRAME_HEIGHT * render_scale,
        )
        resize_target = target_size if render_scale > 1 else None

        sweeps = self._sweep_specs()
        self._log(f"Generating {len(sweeps)} MP4 files...")

        state = self._snapshot_state()
        try:
            for sweep_number, sweep in enumerate(sweeps, start=1):
                output_path = output_dir / sweep.filename
                self._log(
                    f"Rendering {sweep_number}/{len(sweeps)}: {output_path.name}"
                )
                self._reset_sweep_state()
                self._activate_sweep_uncertainty(sweep)
                values = self._sweep_values(sweep, profile.frames_per_phase)
                frames = self._render_sweep_frames(
                    sweep,
                    values,
                    renderer=renderer,
                    target_size=resize_target,
                    crop_pad=profile.crop_pad,
                )
                frames = self._normalize_frame_sizes(frames)
                self._encode_mp4(
                    frames,
                    output_path,
                    fps=profile.fps,
                    color_boost=profile.color_boost,
                    contrast_boost=profile.contrast_boost,
                    ffmpeg_path=ffmpeg_path,
                    crf=profile.crf,
                    preset=profile.preset,
                )
                self._log(f"Saved: {output_path}")
        finally:
            self._restore_state(state)
            self._update_mesh()
            renderer.delete()
        self._log("Video export complete.")

    def close(self) -> None:
        """Release exporter resources.

        Renderers are scoped to `create_videos()` so there is no persistent EGL
        context to clean up here.
        """
        return None

if __name__ == "__main__":
    import sys

    _video_parser = argparse.ArgumentParser(
        description="Generate FLAME parameter-sweep MP4 videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _video_parser.add_argument(
        "--out-dir",
        default="results/videos",
        help="Directory for MP4 outputs.",
    )
    _video_parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PROFILES),
        default="default",
        help="Export quality profile: default is faster, high restores slower high-quality MP4s.",
    )
    _video_args, _remaining = _video_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining   # hide exporter flags from get_config()

    print("[video] Starting FLAME video exporter.", flush=True)
    exporter = FlameVideoExporter()
    try:
        exporter.create_videos(
            out_dir=_video_args.out_dir,
            quality=_video_args.quality,
        )
    finally:
        exporter.close()
