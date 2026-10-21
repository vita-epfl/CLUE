"""Generate SMPL-X parameter-sweep videos with CLUE uncertainty overlays.

This script renders short MP4 clips that vary one SMPL-X parameter sweep at a
time. For each configured shape and body-pose sweep, the exporter activates the
corresponding closed-form uncertainty term in :class:`algorithm.CLUE`, animates
the parameter through ``0 -> +max -> -max -> 0``, and writes a cropped video
named ``<sweep>.mp4``.

The module is intentionally headless: it builds a pyrender scene, renders
frames offscreen, and saves videos to ``results/videos/`` by default. Use
``--quality=default`` for the fast export path or ``--quality=high`` for
higher-quality MP4s. Pass ``--direction`` to color uncertainty by the dominant
variance axis instead of amplitude only.
"""

from __future__ import annotations

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
_DEFAULT_MODEL_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
)

import argparse
from dataclasses import dataclass
import io
import math
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    _RESAMPLE = (
        getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
        or Image.LANCZOS
    )
    try:
        _CBAR_FONT = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        _CBAR_FONT = ImageFont.load_default(size=11)
    except TypeError:
        _CBAR_FONT = ImageFont.load_default()
except ImportError as exc:
    raise ImportError("Pillow is required: pip install Pillow") from exc

try:
    import matplotlib as _mpl

    try:
        _TURBO = _mpl.colormaps["turbo"]
    except Exception:
        import matplotlib.cm as _mpl_cm

        _TURBO = _mpl_cm.get_cmap("turbo")
except Exception:
    _TURBO = None


if _TURBO is not None:
    _BASE_COLOR = np.array(_TURBO(0.0)[:3], dtype=np.float32)
else:
    _BASE_COLOR = np.array([0.05, 0.14, 0.28], dtype=np.float32)

_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)
_DIR_COLORS = np.array(
    [
        [0.85, 0.05, 0.05],
        [0.85, 0.80, 0.05],
        [0.05, 0.65, 0.05],
    ],
    dtype=np.float32,
)


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    """Color vertices by uncertainty amplitude."""

    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    if _TURBO is not None:
        return (_TURBO(tt) * 255).astype(np.uint8)

    t_col = tt[:, np.newaxis]
    rgb = (1.0 - t_col) * _BASE_COLOR + t_col * _AMPLITUDE_HIGH
    alpha = np.ones((len(tt), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _dir_colors(var_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Color vertices by dominant variance axis, scaled by total uncertainty."""

    dominant = var_xyz.argmax(axis=-1)
    target = _DIR_COLORS[dominant]
    t_col = np.clip(t, 0.0, 1.0).astype(np.float32)[:, np.newaxis]
    rgb = (1.0 - t_col) * _BASE_COLOR + t_col * target
    alpha = np.ones((len(t), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


if not hasattr(np, "infty"):
    np.infty = np.inf

import pyrender
import smplx
import torch
import trimesh

from algorithm import CLUE

__all__ = ["SmplxVideoExporter", "QUALITY_PROFILES", "parse_args", "main"]


@dataclass(frozen=True)
class TensorTarget:
    """One tensor entry that participates in a sweep."""

    attr: str
    index: int


@dataclass(frozen=True)
class VarianceTarget:
    """One variance tensor entry and the uncertainty value applied to it."""

    attr: str
    index: int
    value: float


@dataclass(frozen=True)
class ParameterSweep:
    """One SMPL-X animation and its matching CLUE variance entries."""

    name: str
    value_min: float
    value_max: float
    parameter_targets: tuple[TensorTarget, ...]
    variance_targets: tuple[VarianceTarget, ...]

    @property
    def filename(self) -> str:
        return f"{self.name}.mp4"


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

        cam_pos = self._target + self.distance * np.array(
            [
                math.cos(el) * math.sin(az),
                math.sin(el),
                math.cos(el) * math.cos(az),
            ]
        )

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


class SmplxVideoExporter:
    """Headless SMPL-X exporter for CLUE uncertainty sweep videos."""

    SHAPE_PARAMETERS = 10
    EXPRESSION_PARAMETERS = 10

    FRAME_WIDTH = 800
    FRAME_HEIGHT = 800
    BETA_VARIANCE_MAX = 0.5
    POSE_VARIANCE_MAX = 0.1

    def __init__(
        self,
        model_folder: str | Path,
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
        direction: bool = False,
    ) -> None:
        self._direction_mode = direction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log(f"Using device: {self.device.type.upper()}")
        self._log("Loading SMPL-X model...")
        self.model = smplx.create(
            str(model_folder),
            model_type="smplx",
            gender="neutral",
            num_betas=max(num_betas, self.SHAPE_PARAMETERS),
            num_expression_coeffs=max(
                num_expression_coeffs, self.EXPRESSION_PARAMETERS
            ),
            use_pca=False,
            ext="npz",
            batch_size=1,
        ).to(self.device)
        self.model.eval()
        self._log("SMPL-X model ready.")

        self.n_shape_betas = self.model.shapedirs.shape[2]
        self.n_expr_betas = self.model.expr_dirs.shape[2]
        self.n_body_joints = self.model.NUM_BODY_JOINTS
        self.n_hand_joints = self.model.NUM_HAND_JOINTS

        self._fp_global_start = 0
        self._fp_body_start = 3
        self._fp_jaw_start = self._fp_body_start + self.n_body_joints * 3
        self._fp_leye_start = self._fp_jaw_start + 3
        self._fp_reye_start = self._fp_leye_start + 3
        self._fp_lhand_start = self._fp_reye_start + 3
        self._fp_rhand_start = self._fp_lhand_start + self.n_hand_joints * 3
        self.n_pose_total = self._fp_rhand_start + self.n_hand_joints * 3

        self.betas = torch.zeros(
            1, self.n_shape_betas, dtype=torch.float32, device=self.device
        )
        self.expression = torch.zeros(
            1, self.n_expr_betas, dtype=torch.float32, device=self.device
        )
        self.global_orient = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.body_pose = torch.zeros(
            1, self.n_body_joints * 3, dtype=torch.float32, device=self.device
        )
        self.jaw_pose = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.leye_pose = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.reye_pose = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.left_hand_pose = torch.zeros(
            1, self.n_hand_joints * 3, dtype=torch.float32, device=self.device
        )
        self.right_hand_pose = torch.zeros(
            1, self.n_hand_joints * 3, dtype=torch.float32, device=self.device
        )

        self.shape_var_params = torch.zeros(
            1, self.SHAPE_PARAMETERS, dtype=torch.float32, device=self.device
        )
        self.expr_var_params = torch.zeros(
            1, self.EXPRESSION_PARAMETERS, dtype=torch.float32, device=self.device
        )
        self.pose_var_params = torch.zeros(
            1, self.n_pose_total, dtype=torch.float32, device=self.device
        )
        self._var_range: float | None = None

        self._param_std_ref: dict[tuple[str, int], float] = {}

        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[
                float(_BASE_COLOR[0]),
                float(_BASE_COLOR[1]),
                float(_BASE_COLOR[2]),
                1.0,
            ],
            metallicFactor=0.3,
            roughnessFactor=0.52,
            alphaMode="OPAQUE",
            doubleSided=True,
        )
        self.vc_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.3,
            roughnessFactor=0.52,
            alphaMode="OPAQUE",
            doubleSided=True,
        )

        self.clue = CLUE(self.model)

        self._log("Building render scene...")
        vertices, joints = self._compute_vertices()
        self.scene, self.mesh_node, self.center, self.cam_node, extent = (
            self._make_scene(vertices, joints)
        )
        self.orbit = OrbitCamera(distance=2.1 * extent)

        self._log("Precomputing CLUE uncertainty ranges...")
        self._precompute_param_data()
        self._log("Exporter initialization complete.")

    def _log(self, message: str) -> None:
        print(f"[video] {message}", flush=True)

    def _pose_kwargs(self) -> dict[str, torch.Tensor]:
        """Return the full set of SMPL-X pose keyword arguments."""

        return {
            "global_orient": self.global_orient,
            "body_pose": self.body_pose,
            "jaw_pose": self.jaw_pose,
            "leye_pose": self.leye_pose,
            "reye_pose": self.reye_pose,
            "left_hand_pose": self.left_hand_pose,
            "right_hand_pose": self.right_hand_pose,
        }

    def _compute_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            output = self.model(
                betas=self.betas,
                expression=self.expression,
                **self._pose_kwargs(),
                return_verts=True,
            )
        return (
            output.vertices[0].detach().cpu().numpy(),
            output.joints[0].detach().cpu().numpy(),
        )

    def _build_trimesh(
        self, vertices: np.ndarray, center: np.ndarray
    ) -> trimesh.Trimesh:
        tri_mesh = trimesh.Trimesh(vertices - center, self.model.faces, process=False)
        for remove_method, face_selector in [
            ("remove_degenerate_faces", "nondegenerate_faces"),
            ("remove_duplicate_faces", "unique_faces"),
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
        del joints

        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = max(float(np.max(bounds[1] - bounds[0])), 1e-3)

        tri_mesh = self._build_trimesh(vertices, center)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)

        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.12, 0.13, 0.14],
        )
        mesh_node = scene.add(mesh)
        cam_node = scene.add(
            pyrender.PerspectiveCamera(yfov=np.pi / 4.0),
            pose=np.eye(4),
        )

        def _look_at_light(direction: Sequence[float], intensity: float) -> None:
            d = np.asarray(direction, dtype=np.float64)
            d /= np.linalg.norm(d) + 1e-8
            z = -d
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z)
            x /= np.linalg.norm(x) + 1e-8
            y = np.cross(z, x)
            light_pose = np.eye(4)
            light_pose[:3, 0] = x
            light_pose[:3, 1] = y
            light_pose[:3, 2] = z
            light_pose[:3, 3] = -d * 2.5 * extent
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                pose=light_pose,
            )

        _look_at_light([-0.4, -0.55, -1.0], 3.5)
        _look_at_light([0.8, -0.2, -0.6], 1.75)
        _look_at_light([0.0, 1.0, -0.5], 0.75)

        return scene, mesh_node, center, cam_node, extent

    def _variance_active(self) -> bool:
        return bool(
            (self.shape_var_params > 0).any().item()
            or (self.expr_var_params > 0).any().item()
            or (self.pose_var_params[:, self._fp_body_start :] > 0).any().item()
        )

    def _build_beta_vars(self) -> torch.Tensor:
        """Construct the SMPL-X beta variance vector."""

        beta_vars = torch.zeros(
            1,
            self.n_shape_betas + self.n_expr_betas,
            dtype=torch.float32,
            device=self.device,
        )
        beta_vars[0, : self.shape_var_params.shape[1]] = self.shape_var_params[0]
        expr_start = self.n_shape_betas
        expr_end = expr_start + self.expr_var_params.shape[1]
        beta_vars[0, expr_start:expr_end] = self.expr_var_params[0]
        return beta_vars

    def _build_pose_vars(self) -> torch.Tensor:
        """Return the flat SMPL-X pose variance tensor without root rotation."""

        pose_vars = self.pose_var_params.clone()
        pose_vars[:, : self._fp_body_start] = 0.0
        return pose_vars

    def _sweep_specs(self) -> list[ParameterSweep]:
        """Return the configured SMPL-X parameter sweeps."""

        shape_last = min(self.SHAPE_PARAMETERS, self.n_shape_betas) - 1

        spine1_body = 2 * 3
        rhip_body = 1 * 3
        rshoulder_body = 16 * 3
        fp_body = self._fp_body_start

        return [
            ParameterSweep(
                name="shape_0",
                value_min=-3.0,
                value_max=3.0,
                parameter_targets=(TensorTarget("betas", 0),),
                variance_targets=(
                    VarianceTarget(
                        "shape_var_params",
                        0,
                        self.BETA_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="shape_1",
                value_min=-3.0,
                value_max=3.0,
                parameter_targets=(TensorTarget("betas", 1),),
                variance_targets=(
                    VarianceTarget(
                        "shape_var_params",
                        1,
                        self.BETA_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="shape_last",
                value_min=-3.0,
                value_max=3.0,
                parameter_targets=(TensorTarget("betas", shape_last),),
                variance_targets=(
                    VarianceTarget(
                        "shape_var_params",
                        shape_last,
                        self.BETA_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="spine1_x",
                value_min=-0.5,
                value_max=0.5,
                parameter_targets=(TensorTarget("body_pose", spine1_body),),
                variance_targets=(
                    VarianceTarget(
                        "pose_var_params",
                        fp_body + spine1_body,
                        self.POSE_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="spine1_y",
                value_min=-0.5,
                value_max=0.5,
                parameter_targets=(TensorTarget("body_pose", spine1_body + 1),),
                variance_targets=(
                    VarianceTarget(
                        "pose_var_params",
                        fp_body + spine1_body + 1,
                        self.POSE_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="spine1_z",
                value_min=-0.5,
                value_max=0.5,
                parameter_targets=(TensorTarget("body_pose", spine1_body + 2),),
                variance_targets=(
                    VarianceTarget(
                        "pose_var_params",
                        fp_body + spine1_body + 2,
                        self.POSE_VARIANCE_MAX,
                    ),
                ),
            ),
            ParameterSweep(
                name="rshoulder_z_rhip_x",
                value_min=-0.5,
                value_max=0.5,
                parameter_targets=(
                    TensorTarget("body_pose", rshoulder_body + 2),
                    TensorTarget("body_pose", rhip_body),
                ),
                variance_targets=(
                    VarianceTarget(
                        "pose_var_params",
                        fp_body + rshoulder_body + 2,
                        self.POSE_VARIANCE_MAX,
                    ),
                    VarianceTarget(
                        "pose_var_params",
                        fp_body + rhip_body,
                        self.POSE_VARIANCE_MAX,
                    ),
                ),
            ),
        ]

    def _precompute_param_data(self) -> None:
        """Precompute the uncertainty ceiling for each active sweep target."""

        unique_targets: list[VarianceTarget] = []
        seen: set[tuple[str, int]] = set()
        for sweep in self._sweep_specs():
            for target in sweep.variance_targets:
                key = (target.attr, target.index)
                if key not in seen:
                    seen.add(key)
                    unique_targets.append(target)

        beta_width = self.n_shape_betas + self.n_expr_betas
        pose_width = self.n_pose_total
        common_args = {
            "betas": self.betas,
            "expression": self.expression,
            **self._pose_kwargs(),
        }

        for idx, target in enumerate(unique_targets, start=1):
            self._log(
                f"CLUE range {idx}/{len(unique_targets)}: "
                f"{target.attr}[{target.index}]"
            )

            beta_vars = torch.zeros(
                1, beta_width, dtype=torch.float32, device=self.device
            )
            pose_vars = torch.zeros(
                1, pose_width, dtype=torch.float32, device=self.device
            )

            if target.attr == "shape_var_params":
                beta_vars[0, target.index] = target.value
            elif target.attr == "expr_var_params":
                beta_vars[0, self.n_shape_betas + target.index] = target.value
            elif target.attr == "pose_var_params":
                pose_vars[0, target.index] = target.value
            else:
                raise ValueError(f"Unsupported variance target: {target.attr}")

            total_var, _ = self.clue.compute_variance(
                beta_vars,
                pose_vars,
                **common_args,
            )
            var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)
            std = np.sqrt(var_xyz.sum(axis=-1))
            self._param_std_ref[(target.attr, target.index)] = float(std.max()) * 1000.0

    def _active_std_reference(self) -> float | None:
        refs: list[float] = []

        for idx in torch.nonzero(
            self.shape_var_params[0] > 0, as_tuple=False
        ).flatten().tolist():
            ref = self._param_std_ref.get(("shape_var_params", int(idx)))
            if ref is not None:
                refs.append(ref)

        for idx in torch.nonzero(
            self.expr_var_params[0] > 0, as_tuple=False
        ).flatten().tolist():
            ref = self._param_std_ref.get(("expr_var_params", int(idx)))
            if ref is not None:
                refs.append(ref)

        for idx in torch.nonzero(
            self.pose_var_params[0, self._fp_body_start :] > 0, as_tuple=False
        ).flatten().tolist():
            pose_idx = int(idx) + self._fp_body_start
            ref = self._param_std_ref.get(("pose_var_params", pose_idx))
            if ref is not None:
                refs.append(ref)

        return max(refs) if refs else None

    def _compute_vertex_colors(self) -> tuple[np.ndarray, float | None]:
        """Propagate active variances to per-vertex RGBA colors."""

        if not self._variance_active():
            vertex_count = int(self.model.v_template.shape[0])
            return _amplitude_colors(np.zeros(vertex_count, dtype=np.float32)), None

        total_var, _ = self.clue.compute_variance(
            self._build_beta_vars(),
            self._build_pose_vars(),
            betas=self.betas,
            expression=self.expression,
            **self._pose_kwargs(),
        )
        var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)
        std = np.sqrt(var_xyz.sum(axis=-1)) * 1000.0

        std_ref_max = self._active_std_reference()
        if std_ref_max is None or std_ref_max <= 1e-12:
            t = np.zeros_like(std)
            std_ref_max = None
        else:
            t = std / std_ref_max

        if self._direction_mode:
            return _dir_colors(var_xyz, t), std_ref_max
        return _amplitude_colors(t), std_ref_max

    def _draw_colorbar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
        """Render a vertical uncertainty legend in the top-right corner."""

        draw = ImageDraw.Draw(img)
        iw, ih = img.size
        bar_w = 36
        bar_h = 240
        margin_r = 14
        margin_t = 14
        pad = 10
        label_w = 70

        panel_w = bar_w + label_w + pad * 2
        panel_h = bar_h + pad * 2 + 20

        px = iw - margin_r - panel_w
        py = margin_t

        draw.rectangle(
            [px, py, px + panel_w, py + panel_h],
            fill=(245, 245, 245),
            outline=(190, 190, 190),
        )
        draw.text(
            (px + pad, py + pad),
            "uncertainty",
            fill=(50, 50, 50),
            font=_CBAR_FONT,
        )

        bx = px + pad + label_w
        by = py + pad + 20
        for row in range(bar_h):
            t = 1.0 - row / max(bar_h - 1, 1)
            if _TURBO is not None:
                rgba = (_TURBO(np.array([t]))[0] * 255).astype(int)
                color = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
            else:
                rgb = (1.0 - t) * _BASE_COLOR + t * _AMPLITUDE_HIGH
                color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            draw.line([(bx, by + row), (bx + bar_w, by + row)], fill=color)
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(80, 80, 80))

        draw.text(
            (px + pad, by),
            f"{std_ref_max:.2f}mm",
            fill=(30, 30, 30),
            font=_CBAR_FONT,
        )
        draw.text(
            (px + pad, by + bar_h - 14),
            "0 mm",
            fill=(30, 30, 30),
            font=_CBAR_FONT,
        )
        return img

    def _update_mesh(self) -> None:
        vertices, _ = self._compute_vertices()
        tri_mesh = self._build_trimesh(vertices, self.center)
        vertex_colors, std_ref_max = self._compute_vertex_colors()
        self._var_range = std_ref_max

        if len(vertex_colors) == len(tri_mesh.vertices):
            primitive = pyrender.Primitive(
                positions=tri_mesh.vertices.astype(np.float32),
                normals=tri_mesh.vertex_normals.astype(np.float32),
                indices=tri_mesh.faces.astype(np.uint32),
                color_0=vertex_colors,
                material=self.vc_material,
            )
            mesh = pyrender.Mesh(primitives=[primitive])
        else:
            mesh = pyrender.Mesh.from_trimesh(
                tri_mesh, material=self.material, smooth=True
            )

        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)

    def _render_offscreen_image(
        self,
        renderer: pyrender.OffscreenRenderer,
        target_size: tuple[int, int] | None = None,
        draw_overlays: bool = True,
    ) -> Image.Image:
        """Render the current scene offscreen and return a PIL image."""

        try:
            self.scene.set_pose(self.cam_node, self.orbit.get_pose())
            color, _ = renderer.render(self.scene)
        except Exception as exc:
            raise RuntimeError(
                "Offscreen render failed; check the EGL/OpenGL setup."
            ) from exc

        img = Image.fromarray(color)
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, _RESAMPLE)

        if draw_overlays and self._var_range is not None:
            img = self._draw_colorbar(img, self._var_range)

        return img

    def _tight_crop(
        self,
        img: Image.Image,
        pad: int = 8,
        bg: tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """Crop a uniform background border tightly around the content."""

        arr = np.asarray(img.convert("RGB"))
        tol = 6
        mask = np.any(np.abs(arr - np.array(bg, dtype=np.uint8)) > tol, axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        left = max(0, x0 - pad)
        upper = max(0, y0 - pad)
        right = min(img.width, x1 + pad + 1)
        lower = min(img.height, y1 + pad + 1)
        return img.crop((left, upper, right, lower))

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            "betas": self.betas.clone(),
            "expression": self.expression.clone(),
            "global_orient": self.global_orient.clone(),
            "body_pose": self.body_pose.clone(),
            "jaw_pose": self.jaw_pose.clone(),
            "leye_pose": self.leye_pose.clone(),
            "reye_pose": self.reye_pose.clone(),
            "left_hand_pose": self.left_hand_pose.clone(),
            "right_hand_pose": self.right_hand_pose.clone(),
            "shape_var_params": self.shape_var_params.clone(),
            "expr_var_params": self.expr_var_params.clone(),
            "pose_var_params": self.pose_var_params.clone(),
        }

    def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
        self.betas.copy_(state["betas"])
        self.expression.copy_(state["expression"])
        self.global_orient.copy_(state["global_orient"])
        self.body_pose.copy_(state["body_pose"])
        self.jaw_pose.copy_(state["jaw_pose"])
        self.leye_pose.copy_(state["leye_pose"])
        self.reye_pose.copy_(state["reye_pose"])
        self.left_hand_pose.copy_(state["left_hand_pose"])
        self.right_hand_pose.copy_(state["right_hand_pose"])
        self.shape_var_params.copy_(state["shape_var_params"])
        self.expr_var_params.copy_(state["expr_var_params"])
        self.pose_var_params.copy_(state["pose_var_params"])

    def _reset_sweep_state(self) -> None:
        self.betas.zero_()
        self.expression.zero_()
        self.global_orient.zero_()
        self.body_pose.zero_()
        self.jaw_pose.zero_()
        self.leye_pose.zero_()
        self.reye_pose.zero_()
        self.left_hand_pose.zero_()
        self.right_hand_pose.zero_()
        self.shape_var_params.zero_()
        self.expr_var_params.zero_()
        self.pose_var_params.zero_()

    def _activate_sweep_uncertainty(self, sweep: ParameterSweep) -> None:
        for target in sweep.variance_targets:
            getattr(self, target.attr)[0, target.index] = target.value

    def _set_sweep_value(self, sweep: ParameterSweep, value: float) -> None:
        for target in sweep.parameter_targets:
            getattr(self, target.attr)[0, target.index] = value

    @staticmethod
    def _sweep_values(sweep: ParameterSweep, frames_per_phase: int) -> np.ndarray:
        positive = np.linspace(0.0, sweep.value_max, frames_per_phase, endpoint=False)
        through_negative = np.linspace(
            sweep.value_max,
            sweep.value_min,
            frames_per_phase * 2,
            endpoint=False,
        )
        back_to_zero = np.linspace(
            sweep.value_min,
            0.0,
            frames_per_phase,
            endpoint=True,
        )
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

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                process.stdin.write(buffer.getvalue())

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
        """Generate one MP4 parameter sweep for each configured SMPL-X sweep."""

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
        """Release exporter resources."""

        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the SMPL-X video exporter."""

    parser = argparse.ArgumentParser(
        description="Generate SMPL-X parameter-sweep MP4 videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-folder",
        required=False,
        default=_DEFAULT_MODEL_FOLDER,
        help="Path to the folder containing smplx/SMPLX_NEUTRAL.npz.",
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=10,
        help="Number of shape coefficients to request from SMPL-X.",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        type=int,
        default=10,
        help="Number of expression coefficients to request from SMPL-X.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/videos",
        help="Directory for MP4 outputs.",
    )
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PROFILES),
        default="default",
        help="Export quality profile: default is faster, high restores slower high-quality MP4s.",
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        default=False,
        help="Color-code uncertainty by dominant variance axis instead of amplitude only.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SMPL-X video exporter."""

    args = parse_args(argv)
    print("[video] Starting SMPL-X video exporter.", flush=True)
    exporter = SmplxVideoExporter(
        model_folder=Path(os.path.expandvars(args.model_folder)).expanduser(),
        num_betas=args.num_betas,
        num_expression_coeffs=args.num_expression_coeffs,
        direction=args.direction,
    )
    try:
        exporter.create_videos(
            out_dir=Path(args.out_dir).expanduser(),
            quality=args.quality,
        )
    finally:
        exporter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
