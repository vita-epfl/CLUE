"""Generate MHR parameter-sweep videos with CLUE uncertainty overlays.

This mirrors the FLAME and SMPL-X ``video.py`` experiments: it renders short
MP4 clips that sweep one deterministic parameter while activating the matching
CLUE variance term. Outputs are written under ``results/videos/`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("USER", f"uid{os.getuid()}")

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    _RESAMPLE = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
    try:
        _CBAR_FONT = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        _CBAR_FONT = ImageFont.load_default(size=11)
    except TypeError:
        _CBAR_FONT = ImageFont.load_default()
except ImportError as exc:
    raise ImportError("Pillow is required for MHR video export.") from exc

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
_MODEL_STD_TO_MM = 10.0
_ROOT_MODEL_PARAM_COUNT = 6


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    if _TURBO is not None:
        return (_TURBO(tt) * 255).astype(np.uint8)
    rgb = (1.0 - tt[:, None]) * _BASE_COLOR + tt[:, None] * _AMPLITUDE_HIGH
    alpha = np.ones((len(tt), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _dir_colors(var_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    dominant = var_xyz.argmax(axis=-1)
    target = _DIR_COLORS[dominant]
    t_col = np.clip(t, 0.0, 1.0).astype(np.float32)[:, None]
    rgb = (1.0 - t_col) * _BASE_COLOR + t_col * target
    alpha = np.ones((len(t), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


if not hasattr(np, "infty"):
    np.infty = np.inf

import pyrender
import torch
import trimesh

from algorithm import CLUE
from model_loader import (
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_IDENTITY_BLENDSHAPES,
    get_default_asset_folder,
    load_mhr_model,
)

__all__ = ["MhrVideoExporter", "QUALITY_PROFILES", "parse_args", "main"]


@dataclass(frozen=True)
class TensorTarget:
    attr: str
    index: int


@dataclass(frozen=True)
class VarianceTarget:
    attr: str
    index: int
    value: float


@dataclass(frozen=True)
class ParameterSweep:
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
        azimuth = math.radians(self.azimuth)
        elevation = math.radians(self.elevation)
        cam_pos = self._target + self.distance * np.array(
            [
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
                math.cos(elevation) * math.cos(azimuth),
            ]
        )
        forward = self._target - cam_pos
        length = np.linalg.norm(forward)
        if length < 1e-8:
            return np.eye(4)
        forward /= length
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


class MhrVideoExporter:
    """Headless MHR exporter for CLUE uncertainty sweep videos."""

    IDENTITY_PARAMETERS = 10
    EXPRESSION_PARAMETERS = 10
    MODEL_PARAMETERS = 204 - _ROOT_MODEL_PARAM_COUNT

    FRAME_WIDTH = 800
    FRAME_HEIGHT = 800
    IDENTITY_EXPR_VARIANCE_MAX = 0.5
    MODEL_VARIANCE_MAX = 0.1

    def __init__(
        self,
        asset_folder: str | Path,
        lod: int = 1,
        num_identity_coeffs: int = 10,
        num_expression_coeffs: int = 10,
        num_model_params: int = 204,
        apply_correctives: bool = False,
        direction: bool = False,
    ) -> None:
        self._direction_mode = direction
        self.apply_correctives = apply_correctives
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log(f"Using device: {self.device.type.upper()}")
        self._log("Loading MHR model...")
        self.model = load_mhr_model(
            folder=Path(asset_folder),
            device=self.device,
            lod=lod,
            wants_pose_correctives=apply_correctives,
        )
        self.model.eval()
        self._log("MHR model ready.")

        self.n_identity = NUM_IDENTITY_BLENDSHAPES
        self.n_expression = NUM_FACE_EXPRESSION_BLENDSHAPES
        self.n_model_total = (
            self.model.character.parameter_transform.size
            - self.n_identity
            - self.n_expression
        )
        self.n_identity_visible = min(num_identity_coeffs, self.n_identity)
        self.n_expression_visible = min(num_expression_coeffs, self.n_expression)
        self.model_param_names = list(
            self.model.character.parameter_transform.names[: self.n_model_total]
        )
        model_stop = min(
            self.n_model_total,
            _ROOT_MODEL_PARAM_COUNT + max(num_model_params, 0),
        )
        self.model_param_indices = list(range(_ROOT_MODEL_PARAM_COUNT, model_stop))
        self.model_param_index_to_local = {
            model_idx: local_idx
            for local_idx, model_idx in enumerate(self.model_param_indices)
        }
        self.n_model_visible = len(self.model_param_indices)

        self.identity_coeffs = torch.zeros(1, self.n_identity, dtype=torch.float32, device=self.device)
        self.face_expr_coeffs = torch.zeros(1, self.n_expression, dtype=torch.float32, device=self.device)
        self.model_parameters = torch.zeros(1, self.n_model_total, dtype=torch.float32, device=self.device)

        self.identity_var_params = torch.zeros(1, self.n_identity_visible, dtype=torch.float32, device=self.device)
        self.expression_var_params = torch.zeros(1, self.n_expression_visible, dtype=torch.float32, device=self.device)
        self.model_var_params = torch.zeros(1, self.n_model_visible, dtype=torch.float32, device=self.device)
        self._var_range: float | None = None
        self._param_std_ref: dict[tuple[str, int], float] = {}
        self._param_group_max_std: dict[str, float] = {}

        self.faces = self.model.character.mesh.faces
        self.clue = CLUE(self.model)

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

        self._log("Building render scene...")
        vertices, _ = self._compute_vertices()
        self.scene, self.mesh_node, self.center, self.cam_node, extent = self._make_scene(vertices)
        self.orbit = OrbitCamera(distance=2.1 * extent)

        self._log("Precomputing CLUE uncertainty ranges...")
        self._precompute_param_data()
        self._log("Exporter initialization complete.")

    def _log(self, message: str) -> None:
        print(f"[video] {message}", flush=True)

    def _compute_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            vertices, skel_state = self.model(
                self.identity_coeffs,
                self.model_parameters,
                self.face_expr_coeffs,
                apply_correctives=self.apply_correctives,
            )
        return (
            vertices[0].detach().cpu().numpy(),
            skel_state[0, :, :3].detach().cpu().numpy(),
        )

    def _build_trimesh(self, vertices: np.ndarray, center: np.ndarray) -> trimesh.Trimesh:
        tri_mesh = trimesh.Trimesh(vertices - center, self.faces, process=False)
        tri_mesh.fix_normals()
        return tri_mesh

    def _make_scene(self, vertices: np.ndarray):
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
        cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=np.eye(4))

        def _look_at_light(direction: Sequence[float], intensity: float) -> None:
            light_dir = np.asarray(direction, dtype=np.float64)
            light_dir /= np.linalg.norm(light_dir) + 1e-8
            z_axis = -light_dir
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z_axis, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis) + 1e-8
            y_axis = np.cross(z_axis, x_axis)
            pose = np.eye(4)
            pose[:3, 0] = x_axis
            pose[:3, 1] = y_axis
            pose[:3, 2] = z_axis
            pose[:3, 3] = -light_dir * 2.5 * extent
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose)

        _look_at_light([-0.4, -0.55, -1.0], 3.5)
        _look_at_light([0.8, -0.2, -0.6], 1.75)
        _look_at_light([0.0, 1.0, -0.5], 0.75)
        return scene, mesh_node, center, cam_node, extent

    def _variance_active(self) -> bool:
        return bool(
            (self.identity_var_params > 0).any().item()
            or (self.expression_var_params > 0).any().item()
            or (self.model_var_params > 0).any().item()
        )

    def _build_identity_vars(self) -> torch.Tensor:
        values = torch.zeros(1, self.n_identity, dtype=torch.float32, device=self.device)
        values[0, : self.n_identity_visible] = self.identity_var_params[0]
        return values

    def _build_expression_vars(self) -> torch.Tensor:
        values = torch.zeros(1, self.n_expression, dtype=torch.float32, device=self.device)
        values[0, : self.n_expression_visible] = self.expression_var_params[0]
        return values

    def _build_model_vars(self) -> torch.Tensor:
        values = torch.zeros(1, self.n_model_total, dtype=torch.float32, device=self.device)
        if self.model_param_indices:
            values[0, self.model_param_indices] = self.model_var_params[0]
        return values

    def _variance_common_args(self) -> dict:
        return {
            "identity_coeffs": self.identity_coeffs,
            "model_parameters": self.model_parameters,
            "face_expr_coeffs": self.face_expr_coeffs,
            "apply_correctives": self.apply_correctives,
        }

    def _sweep_specs(self) -> list[ParameterSweep]:
        shape_last = min(self.IDENTITY_PARAMETERS, self.n_identity_visible) - 1
        expr_last = min(self.EXPRESSION_PARAMETERS, self.n_expression_visible) - 1

        sweeps: list[ParameterSweep] = []
        if shape_last >= 0:
            sweeps.extend(
                [
                    ParameterSweep(
                        name="identity_0",
                        value_min=-3.0,
                        value_max=3.0,
                        parameter_targets=(TensorTarget("identity_coeffs", 0),),
                        variance_targets=(
                            VarianceTarget("identity_var_params", 0, self.IDENTITY_EXPR_VARIANCE_MAX),
                        ),
                    ),
                    ParameterSweep(
                        name="identity_1",
                        value_min=-3.0,
                        value_max=3.0,
                        parameter_targets=(TensorTarget("identity_coeffs", min(1, shape_last)),),
                        variance_targets=(
                            VarianceTarget(
                                "identity_var_params",
                                min(1, shape_last),
                                self.IDENTITY_EXPR_VARIANCE_MAX,
                            ),
                        ),
                    ),
                ]
            )
        if expr_last >= 0:
            sweeps.append(
                ParameterSweep(
                    name="expression_0",
                    value_min=-1.0,
                    value_max=1.0,
                    parameter_targets=(TensorTarget("face_expr_coeffs", 0),),
                    variance_targets=(
                        VarianceTarget("expression_var_params", 0, self.IDENTITY_EXPR_VARIANCE_MAX),
                    ),
                ),
            )

        spine1_global = 16
        spine1_local = self.model_param_index_to_local.get(spine1_global)
        if spine1_local is not None:
            sweeps.append(
                ParameterSweep(
                    name="spine1_rz",
                    value_min=-0.5,
                    value_max=0.5,
                    parameter_targets=(TensorTarget("model_parameters", spine1_global),),
                    variance_targets=(
                        VarianceTarget("model_var_params", spine1_local, self.MODEL_VARIANCE_MAX),
                    ),
                ),
            )

        shoulder_global = 132
        hip_global = 134
        shoulder_local = self.model_param_index_to_local.get(shoulder_global)
        hip_local = self.model_param_index_to_local.get(hip_global)
        if shoulder_local is not None and hip_local is not None:
            sweeps.append(
                ParameterSweep(
                    name="shoulder_hip_width",
                    value_min=-0.5,
                    value_max=0.5,
                    parameter_targets=(
                        TensorTarget("model_parameters", shoulder_global),
                        TensorTarget("model_parameters", hip_global),
                    ),
                    variance_targets=(
                        VarianceTarget("model_var_params", shoulder_local, self.MODEL_VARIANCE_MAX),
                        VarianceTarget("model_var_params", hip_local, self.MODEL_VARIANCE_MAX),
                    ),
                ),
            )
        return sweeps

    def _precompute_param_data(self) -> None:
        groups = [
            (
                "identity",
                "identity_var_params",
                self.n_identity_visible,
                self.IDENTITY_EXPR_VARIANCE_MAX,
            ),
            (
                "expression",
                "expression_var_params",
                self.n_expression_visible,
                self.IDENTITY_EXPR_VARIANCE_MAX,
            ),
            (
                "model",
                "model_var_params",
                self.n_model_visible,
                self.MODEL_VARIANCE_MAX,
            ),
        ]
        total_params = sum(n_params for _group, _attr, n_params, _var_max in groups)
        completed = 0

        for group_name, attr, n_params, var_max in groups:
            if n_params <= 0:
                self._param_group_max_std[group_name] = 1e-12
                continue

            for param_idx in range(n_params):
                completed += 1
                self._log(
                    f"CLUE range {completed}/{total_params}: "
                    f"{group_name}[{param_idx}]"
                )
                identity_vars = torch.zeros(1, self.n_identity, dtype=torch.float32, device=self.device)
                expression_vars = torch.zeros(1, self.n_expression, dtype=torch.float32, device=self.device)
                model_vars = torch.zeros(1, self.n_model_total, dtype=torch.float32, device=self.device)
                if attr == "identity_var_params":
                    identity_vars[0, param_idx] = var_max
                elif attr == "expression_var_params":
                    expression_vars[0, param_idx] = var_max
                elif attr == "model_var_params":
                    model_vars[0, self.model_param_indices[param_idx]] = var_max
                else:
                    raise ValueError(f"Unsupported variance target: {attr}")

                with torch.no_grad():
                    total_var, _ = self.clue.compute_variance(
                        identity_vars,
                        model_vars,
                        expression_vars,
                        **self._variance_common_args(),
                    )
                var_xyz = np.maximum(total_var[0].detach().cpu().numpy(), 0.0)
                std = np.sqrt(var_xyz.sum(axis=-1))
                self._param_std_ref[(attr, param_idx)] = float(std.max()) * _MODEL_STD_TO_MM

            self._param_group_max_std[group_name] = max(
                self._param_std_ref.get((attr, param_idx), 0.0)
                for param_idx in range(n_params)
            )

    def _active_std_reference(self) -> float | None:
        active_groups: set[str] = set()
        if (self.identity_var_params > 0).any():
            active_groups.add("identity")
        if (self.expression_var_params > 0).any():
            active_groups.add("expression")
        if (self.model_var_params > 0).any():
            active_groups.add("model")
        if not active_groups:
            return None
        return max(
            self._param_group_max_std.get(group_name, 1e-12)
            for group_name in active_groups
        )

    def _compute_vertex_colors(self) -> tuple[np.ndarray, float | None]:
        if not self._variance_active():
            vertex_count = int(self.model.character.mesh.n_vertices)
            return _amplitude_colors(np.zeros(vertex_count, dtype=np.float32)), None

        total_var, _ = self.clue.compute_variance(
            self._build_identity_vars(),
            self._build_model_vars(),
            self._build_expression_vars(),
            **self._variance_common_args(),
        )
        var_xyz = np.maximum(total_var[0].detach().cpu().numpy(), 0.0)
        std = np.sqrt(var_xyz.sum(axis=-1)) * _MODEL_STD_TO_MM
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
        draw = ImageDraw.Draw(img)
        iw, _ih = img.size
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
        draw.rectangle([px, py, px + panel_w, py + panel_h], fill=(245, 245, 245), outline=(190, 190, 190))
        draw.text((px + pad, py + pad), "uncertainty", fill=(50, 50, 50), font=_CBAR_FONT)
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
        draw.text((px + pad, by), f"{std_ref_max:.2f}mm", fill=(30, 30, 30), font=_CBAR_FONT)
        draw.text((px + pad, by + bar_h - 14), "0 mm", fill=(30, 30, 30), font=_CBAR_FONT)
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
            mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)
        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)

    def _render_offscreen_image(
        self,
        renderer: pyrender.OffscreenRenderer,
        target_size: tuple[int, int] | None = None,
        draw_overlays: bool = True,
    ) -> Image.Image:
        try:
            self.scene.set_pose(self.cam_node, self.orbit.get_pose())
            color, _ = renderer.render(self.scene)
        except Exception as exc:
            raise RuntimeError("Offscreen render failed; check the EGL/OpenGL setup.") from exc
        img = Image.fromarray(color)
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, _RESAMPLE)
        if draw_overlays and self._var_range is not None:
            img = self._draw_colorbar(img, self._var_range)
        return img

    @staticmethod
    def _tight_crop(
        img: Image.Image,
        pad: int = 8,
        bg: tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        arr = np.asarray(img.convert("RGB"))
        tol = 6
        mask = np.any(np.abs(arr - np.array(bg, dtype=np.uint8)) > tol, axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return img.crop(
            (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(img.width, x1 + pad + 1),
                min(img.height, y1 + pad + 1),
            )
        )

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            "identity_coeffs": self.identity_coeffs.clone(),
            "face_expr_coeffs": self.face_expr_coeffs.clone(),
            "model_parameters": self.model_parameters.clone(),
            "identity_var_params": self.identity_var_params.clone(),
            "expression_var_params": self.expression_var_params.clone(),
            "model_var_params": self.model_var_params.clone(),
        }

    def _restore_state(self, state: dict[str, torch.Tensor]) -> None:
        for key, value in state.items():
            getattr(self, key).copy_(value)

    def _reset_sweep_state(self) -> None:
        self.identity_coeffs.zero_()
        self.face_expr_coeffs.zero_()
        self.model_parameters.zero_()
        self.identity_var_params.zero_()
        self.expression_var_params.zero_()
        self.model_var_params.zero_()

    def _activate_sweep_uncertainty(self, sweep: ParameterSweep) -> None:
        for target in sweep.variance_targets:
            getattr(self, target.attr)[0, target.index] = target.value

    def _set_sweep_value(self, sweep: ParameterSweep, value: float) -> None:
        for target in sweep.parameter_targets:
            getattr(self, target.attr)[0, target.index] = value

    @staticmethod
    def _sweep_values(sweep: ParameterSweep, frames_per_phase: int) -> np.ndarray:
        positive = np.linspace(0.0, sweep.value_max, frames_per_phase, endpoint=False)
        through_negative = np.linspace(sweep.value_max, sweep.value_min, frames_per_phase * 2, endpoint=False)
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
            img = self._render_offscreen_image(renderer=renderer, target_size=target_size, draw_overlays=False)
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
            canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            canvas.paste(frame, ((max_w - frame.width) // 2, (max_h - frame.height) // 2))
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
            raise RuntimeError("ffmpeg is required to export MP4 videos.")
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or "libx264" not in result.stdout:
            raise RuntimeError("ffmpeg was found, but libx264 is unavailable.")
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
            f"quality={quality}, frames_per_phase={profile.frames_per_phase}, "
            f"fps={profile.fps}, supersample={profile.supersample}, "
            f"output_scale={profile.output_scale}, crf={profile.crf}, preset={profile.preset}"
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
                self._log(f"Rendering {sweep_number}/{len(sweeps)}: {output_path.name}")
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
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MHR parameter-sweep MP4 videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asset-folder",
        type=str,
        default=str(get_default_asset_folder()),
        help="Path to the MHR assets folder.",
    )
    parser.add_argument("--lod", type=int, default=1, choices=list(range(7)), help="MHR level of detail.")
    parser.add_argument("--num-identity-coeffs", type=int, default=10, help="Identity coefficients to expose.")
    parser.add_argument("--num-expression-coeffs", type=int, default=10, help="Expression coefficients to expose.")
    parser.add_argument(
        "--num-model-params",
        type=int,
        default=204 - _ROOT_MODEL_PARAM_COUNT,
        help="Non-root compact model parameters to expose.",
    )
    parser.add_argument("--out-dir", default="results/videos", help="Directory for MP4 outputs.")
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_PROFILES),
        default="default",
        help="Export quality profile.",
    )
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        default=False,
        help="Render deterministic pose correctives; CLUE does not propagate corrective-network uncertainty.",
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        default=False,
        help=(
            "Color-code uncertainty by dominant variance axis "
            "(x=red, y=yellow, z=green). "
            "Without this flag, only the uncertainty amplitude is shown."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print("[video] Starting MHR video exporter.", flush=True)
    exporter = MhrVideoExporter(
        asset_folder=Path(args.asset_folder).expanduser(),
        lod=args.lod,
        num_identity_coeffs=args.num_identity_coeffs,
        num_expression_coeffs=args.num_expression_coeffs,
        num_model_params=args.num_model_params,
        apply_correctives=args.apply_correctives,
        direction=args.direction,
    )
    try:
        exporter.create_videos(out_dir=Path(args.out_dir).expanduser(), quality=args.quality)
    finally:
        exporter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
