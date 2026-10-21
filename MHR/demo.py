"""Interactive MHR viewer with CLUE uncertainty overlays.

This mirrors the FLAME and SMPL-X demos in the parent CLUE experiments. The
left panel edits deterministic MHR identity, expression, and compact model
parameters while paired variance sliders show how parameter uncertainty affects
the mesh vertices.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("USER", f"uid{os.getuid()}")

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk

    _RESAMPLE = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
    try:
        _CBAR_FONT = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        _CBAR_FONT = ImageFont.load_default(size=11)
    except TypeError:
        _CBAR_FONT = ImageFont.load_default()
except ImportError as exc:
    raise ImportError("Pillow is required for the interactive MHR demo.") from exc

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

_DIR_COLORS = np.array(
    [
        [0.85, 0.05, 0.05],
        [0.85, 0.80, 0.05],
        [0.05, 0.65, 0.05],
    ],
    dtype=np.float32,
)
_DIR_LABELS = ["x", "y", "z"]
_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)

# MHR vertices are in centimeters in the released assets.
_MODEL_STD_TO_MM = 10.0
_ROOT_MODEL_PARAM_COUNT = 6

_JOINT_RADIUS = 0.8
_JOINT_LINE_RADIUS = 0.18


def _dir_colors(var_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    dominant = var_xyz.argmax(axis=-1)
    target = _DIR_COLORS[dominant]
    t_col = np.clip(t, 0.0, 1.0)[:, np.newaxis]
    rgb = (1.0 - t_col) * _BASE_COLOR + t_col * target
    alpha = np.ones((len(t), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    if _TURBO is not None:
        return (_TURBO(tt) * 255).astype(np.uint8)
    rgb = (1.0 - tt[:, None]) * _BASE_COLOR + tt[:, None] * _AMPLITUDE_HIGH
    alpha = np.ones((len(tt), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _cylinder_between(p1: np.ndarray, p2: np.ndarray, radius: float):
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return None
    direction = vec / length
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_len = float(np.linalg.norm(cross))
    dot = float(np.dot(z_axis, direction))
    if cross_len > 1e-8:
        rotation = trimesh.transformations.rotation_matrix(
            np.arctan2(cross_len, dot),
            cross / cross_len,
        )
    elif dot < 0.0:
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        rotation = np.eye(4)
    translation = trimesh.transformations.translation_matrix((p1 + p2) * 0.5)
    return trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=8,
        transform=translation @ rotation,
    )


if not hasattr(np, "infty"):
    np.infty = np.inf

try:
    import pyrender
except ImportError as exc:
    raise ImportError("pyrender is required for the interactive MHR demo.") from exc

import torch
import trimesh

from algorithm import CLUE
from model_loader import (
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_IDENTITY_BLENDSHAPES,
    get_default_asset_folder,
    load_mhr_model,
)


class OrbitCamera:
    """Orbit camera that circles around a target point."""

    def __init__(self, distance: float, azimuth: float = 0.0, elevation: float = 20.0):
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self._target = np.zeros(3)

    def drag_rotate(self, dx: float, dy: float, sensitivity: float = 0.3) -> None:
        self.azimuth -= dx * sensitivity
        self.elevation = float(np.clip(self.elevation + dy * sensitivity, -85.0, 85.0))

    def drag_pan(self, dx: float, dy: float) -> None:
        pose = self.get_pose()
        scale = self.distance * 0.001
        self._target -= dx * scale * pose[:3, 0]
        self._target += dy * scale * pose[:3, 1]

    def zoom(self, delta: float) -> None:
        self.distance = max(0.01, self.distance * (0.9 ** delta))

    def reset_target(self) -> None:
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


class MhrInteractiveApp:
    """Interactive MHR viewer with deterministic and uncertainty controls."""

    CTRL_WIDTH = 700
    VIEWER_W = 900
    VIEWER_H = 900
    RENDER_INTERVAL_MS = 15

    IDENTITY_PARAM_MIN = -3.0
    IDENTITY_PARAM_MAX = 3.0
    EXPRESSION_PARAM_MIN = -1.0
    EXPRESSION_PARAM_MAX = 1.0
    MODEL_PARAM_MIN = -0.8
    MODEL_PARAM_MAX = 0.8

    IDENTITY_EXPR_VAR_MAX = 0.5
    MODEL_VAR_MAX = 0.1

    def __init__(
        self,
        asset_folder: str,
        lod: int,
        num_identity_coeffs: int,
        num_expression_coeffs: int,
        num_model_params: int,
        apply_correctives: bool,
        direction: bool,
    ) -> None:
        self._direction_mode = direction
        self.apply_correctives = apply_correctives
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_mhr_model(
            folder=Path(asset_folder),
            device=self.device,
            lod=lod,
            wants_pose_correctives=apply_correctives,
        )
        self.model.eval()

        self.n_identity = min(max(num_identity_coeffs, 0), NUM_IDENTITY_BLENDSHAPES)
        self.n_expression = min(
            max(num_expression_coeffs, 0),
            NUM_FACE_EXPRESSION_BLENDSHAPES,
        )
        self.n_model_params_total = (
            self.model.character.parameter_transform.size
            - NUM_IDENTITY_BLENDSHAPES
            - NUM_FACE_EXPRESSION_BLENDSHAPES
        )
        self.model_param_names = list(
            self.model.character.parameter_transform.names[: self.n_model_params_total]
        )
        model_stop = min(
            self.n_model_params_total,
            _ROOT_MODEL_PARAM_COUNT + max(num_model_params, 0),
        )
        self.model_param_indices = list(range(_ROOT_MODEL_PARAM_COUNT, model_stop))
        self.n_model = len(self.model_param_indices)

        self.identity_coeffs = torch.zeros(
            1,
            NUM_IDENTITY_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        self.face_expr_coeffs = torch.zeros(
            1,
            NUM_FACE_EXPRESSION_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        self.model_parameters = torch.zeros(
            1,
            self.n_model_params_total,
            dtype=torch.float32,
            device=self.device,
        )

        self.identity_var_params = torch.zeros(
            1,
            self.n_identity,
            dtype=torch.float32,
            device=self.device,
        )
        self.expression_var_params = torch.zeros(
            1,
            self.n_expression,
            dtype=torch.float32,
            device=self.device,
        )
        self.model_var_params = torch.zeros(
            1,
            self.n_model,
            dtype=torch.float32,
            device=self.device,
        )
        self._var_range: float | None = None
        self._param_std_ref: dict[tuple[str, int], float] = {}
        self._param_group_max_std: dict[str, float] = {}

        self.faces = self.model.character.mesh.faces
        self.parents = np.asarray(self.model.character.skeleton.joint_parents, dtype=np.int64)
        self.joint_names = list(self.model.character.skeleton.joint_names)
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

        self._joint_nodes: list[pyrender.Node] = []
        self._last_joints: np.ndarray | None = None

        vertices, joints = self._compute_vertices()
        self._last_joints = joints
        self.scene, self.mesh_node, self.center, self.cam_node, extent = self._make_scene(
            vertices
        )
        self.orbit = OrbitCamera(distance=2.1 * extent)
        self.renderer = pyrender.OffscreenRenderer(self.VIEWER_W, self.VIEWER_H)
        self._precompute_param_data()

        self.root = tk.Tk()
        self.root.title("MHR Interactive Viewer")
        self._show_joints = tk.BooleanVar(value=False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#2c3e50")
        style.configure("Section.TLabel", font=("Segoe UI", 9, "bold"), foreground="#34495e", padding=(0, 6, 0, 2))
        style.configure("Variance.TLabel", font=("Segoe UI", 9, "bold"), foreground="#8e44ad", padding=(0, 6, 0, 2))
        style.configure("Control.TLabel", font=("Segoe UI", 9), foreground="#555555")
        style.configure("Hint.TLabel", font=("Segoe UI", 8), foreground="#888888")
        style.configure("TScale", background="#ecf0f1")

        self._ctrl_frame = ttk.Frame(self.root, width=self.CTRL_WIDTH)
        self._ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._ctrl_frame.pack_propagate(False)
        ttk.Separator(self.root, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))

        view_frame = ttk.Frame(self.root)
        view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(view_frame, bg="white", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        hint_bar = ttk.Frame(view_frame, padding=(8, 2, 8, 4))
        hint_bar.pack(fill=tk.X)
        self._angle_label = ttk.Label(hint_bar, text="az=0.0  el=20.0", style="Hint.TLabel")
        self._angle_label.pack(side=tk.LEFT)
        ttk.Label(
            hint_bar,
            text="Drag: rotate  |  Right-drag: pan  |  Scroll: zoom  |  Middle-click: reset pan",
            style="Hint.TLabel",
        ).pack(side=tk.RIGHT)

        self._drag_start: tuple[int, int] | None = None
        self._drag_button: int | None = None
        self.canvas.bind("<ButtonPress-1>", lambda e: self._on_mouse_press(e, 1))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_mouse_press(e, 3))
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<B3-Motion>", self._on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-2>", self._on_middle_click)
        self.canvas.bind("<Configure>", lambda _e: self._mark_dirty())

        self._scheduled_update: str | None = None
        self._dirty = True
        self._photo_image = None

        self._identity_param_sliders: list[ttk.Scale] = []
        self._identity_var_sliders: list[ttk.Scale] = []
        self._expression_param_sliders: list[ttk.Scale] = []
        self._expression_var_sliders: list[ttk.Scale] = []
        self._model_param_sliders: list[ttk.Scale] = []
        self._model_var_sliders: list[ttk.Scale] = []

        self._build_controls()

        init_w = self.CTRL_WIDTH + 2 + self.VIEWER_W
        self.root.geometry(f"{init_w}x{self.VIEWER_H}+50+50")
        self.root.minsize(820, 460)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.RENDER_INTERVAL_MS, self._render_loop)

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

    def _variance_active(self) -> bool:
        return bool(
            (self.identity_var_params > 0).any().item()
            or (self.expression_var_params > 0).any().item()
            or (self.model_var_params > 0).any().item()
        )

    def _build_identity_vars(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            NUM_IDENTITY_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        values[0, : self.n_identity] = self.identity_var_params[0]
        return values

    def _build_identity_vars_max(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            NUM_IDENTITY_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        values[0, : self.n_identity] = (
            (self.identity_var_params[0] > 0).float() * self.IDENTITY_EXPR_VAR_MAX
        )
        return values

    def _build_expression_vars(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            NUM_FACE_EXPRESSION_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        values[0, : self.n_expression] = self.expression_var_params[0]
        return values

    def _build_expression_vars_max(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            NUM_FACE_EXPRESSION_BLENDSHAPES,
            dtype=torch.float32,
            device=self.device,
        )
        values[0, : self.n_expression] = (
            (self.expression_var_params[0] > 0).float() * self.IDENTITY_EXPR_VAR_MAX
        )
        return values

    def _build_model_vars(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            self.n_model_params_total,
            dtype=torch.float32,
            device=self.device,
        )
        if self.model_param_indices:
            values[0, self.model_param_indices] = self.model_var_params[0]
        return values

    def _build_model_vars_max(self) -> torch.Tensor:
        values = torch.zeros(
            1,
            self.n_model_params_total,
            dtype=torch.float32,
            device=self.device,
        )
        if self.model_param_indices:
            values[0, self.model_param_indices] = (
                (self.model_var_params[0] > 0).float() * self.MODEL_VAR_MAX
            )
        return values

    def _variance_common_args(self) -> dict:
        return dict(
            identity_coeffs=self.identity_coeffs,
            model_parameters=self.model_parameters,
            face_expr_coeffs=self.face_expr_coeffs,
            apply_correctives=self.apply_correctives,
        )

    def _precompute_param_data(self) -> None:
        """Precompute group-level colorbar ceilings for amplitude mode."""
        groups = [
            ("identity", self.n_identity, self.IDENTITY_EXPR_VAR_MAX),
            ("expression", self.n_expression, self.IDENTITY_EXPR_VAR_MAX),
            ("model", self.n_model, self.MODEL_VAR_MAX),
        ]

        for group_name, n_params, var_max in groups:
            if n_params <= 0:
                self._param_group_max_std[group_name] = 1e-12
                continue

            for idx in range(n_params):
                identity_vars = torch.zeros(
                    1,
                    NUM_IDENTITY_BLENDSHAPES,
                    dtype=torch.float32,
                    device=self.device,
                )
                expression_vars = torch.zeros(
                    1,
                    NUM_FACE_EXPRESSION_BLENDSHAPES,
                    dtype=torch.float32,
                    device=self.device,
                )
                model_vars = torch.zeros(
                    1,
                    self.n_model_params_total,
                    dtype=torch.float32,
                    device=self.device,
                )

                if group_name == "identity":
                    identity_vars[0, idx] = var_max
                elif group_name == "expression":
                    expression_vars[0, idx] = var_max
                elif group_name == "model":
                    model_vars[0, self.model_param_indices[idx]] = var_max

                with torch.no_grad():
                    total_var, _ = self.clue.compute_variance(
                        identity_vars,
                        model_vars,
                        expression_vars,
                        **self._variance_common_args(),
                    )
                std = (
                    np.sqrt(
                        np.maximum(
                            total_var[0].sum(dim=-1).detach().cpu().numpy(),
                            0.0,
                        )
                    )
                    * _MODEL_STD_TO_MM
                )
                self._param_std_ref[(group_name, idx)] = float(std.max())

            self._param_group_max_std[group_name] = max(
                self._param_std_ref.get((group_name, idx), 0.0)
                for idx in range(n_params)
            )

    def _active_variance_groups(self) -> set[str]:
        active_groups: set[str] = set()
        if (self.identity_var_params > 0).any():
            active_groups.add("identity")
        if (self.expression_var_params > 0).any():
            active_groups.add("expression")
        if (self.model_var_params > 0).any():
            active_groups.add("model")
        return active_groups

    def _compute_variance_colors(self) -> tuple[np.ndarray, float | None]:
        if not self._variance_active():
            return _amplitude_colors(
                np.zeros(self.model.character.mesh.n_vertices, dtype=np.float32)
            ), None

        common_args = self._variance_common_args()
        total_var, _ = self.clue.compute_variance(
            self._build_identity_vars(),
            self._build_model_vars(),
            self._build_expression_vars(),
            **common_args,
        )
        var_xyz = np.maximum(total_var[0].detach().cpu().numpy(), 0.0)
        std = np.sqrt(var_xyz.sum(axis=-1)) * _MODEL_STD_TO_MM

        total_var_max, _ = self.clue.compute_variance(
            self._build_identity_vars_max(),
            self._build_model_vars_max(),
            self._build_expression_vars_max(),
            **common_args,
        )
        std_max = (
            np.sqrt(
                np.maximum(
                    total_var_max[0].sum(dim=-1).detach().cpu().numpy(),
                    0.0,
                )
            )
            * _MODEL_STD_TO_MM
        )
        std_ref_max = float(std_max.max())
        t = std / std_ref_max if std_ref_max > 0.0 else np.zeros_like(std)

        if self._direction_mode:
            return _dir_colors(var_xyz, t), std_ref_max
        return _amplitude_colors(t), std_ref_max

    def _compute_per_param_colors(self) -> tuple[np.ndarray, float | None]:
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

        std_ref_max = max(
            (
                self._param_group_max_std.get(group_name, 1e-12)
                for group_name in self._active_variance_groups()
            ),
            default=1e-12,
        )
        t = std / std_ref_max if std_ref_max > 1e-12 else np.zeros_like(std)
        return _amplitude_colors(t), float(std_ref_max)

    def _build_trimesh(self, vertices: np.ndarray, center: np.ndarray) -> trimesh.Trimesh:
        tri_mesh = trimesh.Trimesh(vertices - center, self.faces, process=False)
        tri_mesh.fix_normals()
        return tri_mesh

    def _make_scene(
        self,
        vertices: np.ndarray,
    ) -> tuple[pyrender.Scene, pyrender.Node, np.ndarray, pyrender.Node, float]:
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = max(float(np.max(bounds[1] - bounds[0])), 1e-3)

        tri_mesh = self._build_trimesh(vertices, center)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.12, 0.13, 0.14])
        mesh_node = scene.add(mesh)
        cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=np.eye(4))

        def _look_at_light(direction, intensity):
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

    def _create_joint_overlay(
        self,
        joints: np.ndarray,
        center: np.ndarray,
        scene: pyrender.Scene,
    ) -> None:
        palette = np.array(
            [
                [0.92, 0.34, 0.34],
                [0.31, 0.60, 0.88],
                [0.29, 0.74, 0.49],
                [0.95, 0.73, 0.23],
                [0.67, 0.43, 0.91],
                [0.96, 0.55, 0.29],
            ],
            dtype=np.float32,
        )
        joint_positions = joints - center
        for joint_idx, joint_position in enumerate(joint_positions):
            sphere = trimesh.creation.uv_sphere(radius=_JOINT_RADIUS)
            color = palette[joint_idx % len(palette)]
            sphere.visual.vertex_colors = np.tile(
                np.array([color[0], color[1], color[2], 1.0]),
                (len(sphere.vertices), 1),
            )
            node = scene.add(
                pyrender.Mesh.from_trimesh(sphere),
                pose=trimesh.transformations.translation_matrix(joint_position),
            )
            self._joint_nodes.append(node)

        for child_idx in range(1, len(joint_positions)):
            parent_idx = int(self.parents[child_idx])
            if parent_idx < 0 or parent_idx >= len(joint_positions):
                continue
            cylinder = _cylinder_between(
                joint_positions[parent_idx],
                joint_positions[child_idx],
                _JOINT_LINE_RADIUS,
            )
            if cylinder is None:
                continue
            color = palette[child_idx % len(palette)]
            cylinder.visual.vertex_colors = np.tile(
                np.array([color[0], color[1], color[2], 1.0]),
                (len(cylinder.vertices), 1),
            )
            self._joint_nodes.append(scene.add(pyrender.Mesh.from_trimesh(cylinder)))

    def _update_joint_overlay(self, joints: np.ndarray) -> None:
        for node in self._joint_nodes:
            self.scene.remove_node(node)
        self._joint_nodes.clear()
        self._last_joints = joints
        if self._show_joints.get():
            self._create_joint_overlay(joints, self.center, self.scene)

    def _draw_colorbar(self, img: Image.Image, std_ref_max: float):
        if self._direction_mode:
            return self._draw_direction_legend(img, std_ref_max)
        return self._draw_amplitude_bar(img, std_ref_max), None

    def _draw_direction_legend(self, img: Image.Image, std_ref_max: float):
        draw = ImageDraw.Draw(img)
        iw, _ih = img.size
        swatch_w = 40
        swatch_h = 26
        margin_r = 14
        margin_t = 14
        row_gap = 6
        pad = 10
        panel_w = 170
        swatches_h = len(_DIR_LABELS) * swatch_h + (len(_DIR_LABELS) - 1) * row_gap
        panel_h = pad + 20 + swatches_h + 8 + 1 + 6 + 14 + pad
        px = iw - margin_r - panel_w
        py = margin_t
        draw.rectangle([px, py, px + panel_w, py + panel_h], fill=(245, 245, 245), outline=(190, 190, 190))
        draw.text((px + pad, py + pad), "Axis of largest variation", fill=(50, 50, 50), font=_CBAR_FONT)
        for idx, (label, color) in enumerate(zip(_DIR_LABELS, _DIR_COLORS)):
            sx = px + pad
            sy = py + pad + 20 + idx * (swatch_h + row_gap)
            rgb = tuple((color * 255).astype(np.uint8))
            draw.rectangle([sx, sy, sx + swatch_w, sy + swatch_h], fill=rgb, outline=(80, 80, 80))
            draw.text((sx + swatch_w + 6, sy + 5), label, fill=(30, 30, 30), font=_CBAR_FONT)
        sep_y = py + pad + 20 + swatches_h + 8
        draw.line([(px + pad, sep_y), (px + panel_w - pad, sep_y)], fill=(190, 190, 190), width=1)
        draw.text((px + pad, sep_y + 6), f"max std = {std_ref_max:.1f} mm", fill=(80, 80, 80), font=_CBAR_FONT)
        return img, (px, py, panel_w, panel_h)

    def _draw_amplitude_bar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
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
                r, g, b = int(rgba[0]), int(rgba[1]), int(rgba[2])
            else:
                rgb = (1.0 - t) * _BASE_COLOR + t * _AMPLITUDE_HIGH
                r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            draw.line([(bx, by + row), (bx + bar_w, by + row)], fill=(r, g, b))
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(80, 80, 80))
        draw.text((px + pad, by), f"{std_ref_max:.2f}mm", fill=(30, 30, 30), font=_CBAR_FONT)
        draw.text((px + pad, by + bar_h - 14), "0 mm", fill=(30, 30, 30), font=_CBAR_FONT)
        return img

    def _draw_axis_gizmo(
        self,
        img: Image.Image,
        center: tuple[int, int] | None = None,
    ) -> Image.Image:
        draw = ImageDraw.Draw(img)
        _iw, ih = img.size
        arm_len = 50
        if center is None:
            ox, oy = 60, ih - 60
        else:
            ox, oy = int(center[0]), int(center[1])

        view_rotation = self.orbit.get_pose()[:3, :3].T
        axes = [
            (np.array([1.0, 0.0, 0.0]), _DIR_COLORS[0], "x"),
            (np.array([0.0, 1.0, 0.0]), _DIR_COLORS[1], "y"),
            (np.array([0.0, 0.0, 1.0]), _DIR_COLORS[2], "z"),
        ]
        for world_dir, color, label in sorted(axes, key=lambda item: -(view_rotation @ item[0])[2]):
            cam_dir = view_rotation @ world_dir
            tx = int(ox + arm_len * cam_dir[0])
            ty = int(oy - arm_len * cam_dir[1])
            rgb = tuple((color * 255).astype(np.uint8))
            draw.line([(ox, oy), (tx, ty)], fill=rgb, width=3)
            draw.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=rgb)
            draw.text((tx + 5, ty - 7), label, fill=rgb, font=_CBAR_FONT)
        return img

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _render_loop(self) -> None:
        if self._dirty:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                self.scene.set_pose(self.cam_node, pose=self.orbit.get_pose())
                color, _ = self.renderer.render(self.scene)
                img = Image.fromarray(color)
                scale = min(canvas_w / self.VIEWER_W, canvas_h / self.VIEWER_H)
                if abs(scale - 1.0) > 0.005:
                    img = img.resize(
                        (
                            max(1, int(self.VIEWER_W * scale)),
                            max(1, int(self.VIEWER_H * scale)),
                        ),
                        _RESAMPLE,
                    )
                gizmo_center = None
                if self._var_range is not None:
                    img, panel_rect = self._draw_colorbar(img, self._var_range)
                    if panel_rect is not None:
                        px, py, panel_w, panel_h = panel_rect
                        gizmo_center = (px + panel_w // 2, py + panel_h + 65)
                if self._direction_mode:
                    img = self._draw_axis_gizmo(img, center=gizmo_center)
                self._photo_image = ImageTk.PhotoImage(image=img)
                self.canvas.delete("all")
                self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._photo_image, anchor="center")
                self._angle_label.configure(
                    text=f"az={self.orbit.azimuth:.1f}  el={self.orbit.elevation:.1f}"
                )
                self._dirty = False
        self.root.after(self.RENDER_INTERVAL_MS, self._render_loop)

    def _on_mouse_press(self, event: tk.Event, button: int) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_button = button

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        if self._drag_button == 1:
            self.orbit.drag_rotate(dx, dy)
        elif self._drag_button == 3:
            self.orbit.drag_pan(dx, dy)
        self._mark_dirty()

    def _on_scroll(self, event: tk.Event) -> None:
        self.orbit.zoom(event.delta / 120)
        self._mark_dirty()

    def _on_middle_click(self, _event: tk.Event) -> None:
        self.orbit.reset_target()
        self._mark_dirty()

    def _schedule_mesh_update(self) -> None:
        if self._scheduled_update is not None:
            self.root.after_cancel(self._scheduled_update)
        self._scheduled_update = self.root.after(15, self._update_mesh)

    def _update_mesh(self) -> None:
        self._scheduled_update = None
        vertices, joints = self._compute_vertices()
        tri_mesh = self._build_trimesh(vertices, self.center)
        if self._direction_mode:
            vertex_colors, std_ref_max = self._compute_variance_colors()
            self._var_range = std_ref_max if self._variance_active() else None
        else:
            vertex_colors, std_ref_max = self._compute_per_param_colors()
            self._var_range = std_ref_max

        if len(vertex_colors) == len(tri_mesh.vertices):
            primitive = pyrender.Primitive(
                positions=tri_mesh.vertices.astype(np.float32),
                normals=tri_mesh.vertex_normals.astype(np.float32),
                indices=tri_mesh.faces.astype(np.uint32),
                color_0=vertex_colors,
                material=self.vc_material,
            )
            new_mesh = pyrender.Mesh(primitives=[primitive])
        else:
            new_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)

        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(new_mesh)
        self._update_joint_overlay(joints)
        self._mark_dirty()

    def _build_scrollable_frame(self) -> ttk.Frame:
        container = ttk.Frame(self._ctrl_frame, padding=0)
        container.pack(fill=tk.BOTH, expand=True)
        inner_canvas = tk.Canvas(container, highlightthickness=0, bg="white", borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=inner_canvas.yview)
        scrollable = ttk.Frame(inner_canvas, style="TFrame")
        scrollable.bind(
            "<Configure>",
            lambda _: inner_canvas.configure(scrollregion=inner_canvas.bbox("all")),
        )
        inner_canvas.create_window(
            (0, 0),
            window=scrollable,
            anchor="nw",
            width=self.CTRL_WIDTH - 20,
        )
        inner_canvas.configure(yscrollcommand=scrollbar.set)
        inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return scrollable

    def _make_section_header(
        self,
        parent: ttk.Frame,
        left_text: str,
        right_text: str = "",
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=(10, 2))
        ttk.Label(row, text=left_text, style="Section.TLabel").pack(side=tk.LEFT)
        if right_text:
            ttk.Label(row, text=right_text, style="Variance.TLabel").pack(side=tk.RIGHT)

    def _slider_changed(
        self,
        value: str,
        label: ttk.Label,
        callback: Callable[[str], None],
    ) -> None:
        label.configure(text=f"{float(value):+.3f}")
        callback(value)

    def _make_slider_pair(
        self,
        parent: ttk.Frame,
        param_text: str,
        param_from: float,
        param_to: float,
        param_cb: Callable[[str], None],
        var_cb: Callable[[str], None],
        var_to: float,
        param_list: list[ttk.Scale],
        var_list: list[ttk.Scale],
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=2)
        left = ttk.Frame(row)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right = ttk.Frame(row)
        right.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        ttk.Label(left, text=param_text, width=28).pack(side=tk.LEFT)
        value_label = ttk.Label(left, text="+0.000", width=8)
        value_label.pack(side=tk.LEFT)
        param_slider = ttk.Scale(
            left,
            from_=param_from,
            to=param_to,
            orient=tk.HORIZONTAL,
            command=lambda v: self._slider_changed(v, value_label, param_cb),
        )
        param_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        var_label = ttk.Label(right, text="+0.000", width=8)
        var_label.pack(side=tk.LEFT)
        var_slider = ttk.Scale(
            right,
            from_=0.0,
            to=var_to,
            orient=tk.HORIZONTAL,
            command=lambda v: self._slider_changed(v, var_label, var_cb),
        )
        var_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        param_list.append(param_slider)
        var_list.append(var_slider)

    def _build_controls(self) -> None:
        content = self._build_scrollable_frame()
        ttk.Label(content, text="MHR Controls", style="Header.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 4))

        self._make_section_header(content, "Identity Parameters", "variance")
        for idx in range(self.n_identity):
            self._make_slider_pair(
                content,
                f"identity[{idx}]",
                self.IDENTITY_PARAM_MIN,
                self.IDENTITY_PARAM_MAX,
                lambda v, i=idx: self._on_identity_change(i, v),
                lambda v, i=idx: self._on_identity_var_change(i, v),
                self.IDENTITY_EXPR_VAR_MAX,
                self._identity_param_sliders,
                self._identity_var_sliders,
            )

        self._make_section_header(content, "Expression Parameters", "variance")
        for idx in range(self.n_expression):
            self._make_slider_pair(
                content,
                f"expr[{idx}]",
                self.EXPRESSION_PARAM_MIN,
                self.EXPRESSION_PARAM_MAX,
                lambda v, i=idx: self._on_expression_change(i, v),
                lambda v, i=idx: self._on_expression_var_change(i, v),
                self.IDENTITY_EXPR_VAR_MAX,
                self._expression_param_sliders,
                self._expression_var_sliders,
            )

        self._make_section_header(content, "Compact Model Parameters", "variance")
        for local_idx, model_idx in enumerate(self.model_param_indices):
            name = (
                self.model_param_names[model_idx]
                if model_idx < len(self.model_param_names)
                else f"model[{model_idx}]"
            )
            self._make_slider_pair(
                content,
                f"{model_idx:03d}: {name}",
                self.MODEL_PARAM_MIN,
                self.MODEL_PARAM_MAX,
                lambda v, i=local_idx: self._on_model_change(i, v),
                lambda v, i=local_idx: self._on_model_var_change(i, v),
                self.MODEL_VAR_MAX,
                self._model_param_sliders,
                self._model_var_sliders,
            )

        toggles = ttk.Frame(content)
        toggles.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Checkbutton(
            toggles,
            text="Show all MHR joints",
            variable=self._show_joints,
            command=self._on_joint_toggle,
        ).pack(side=tk.LEFT)

        buttons = ttk.Frame(content)
        buttons.pack(fill=tk.X, padx=10, pady=(2, 10))
        ttk.Button(buttons, text="Reset all", command=self._reset_all).pack(side=tk.LEFT)

    def _on_identity_change(self, index: int, value: str) -> None:
        self.identity_coeffs[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expression_change(self, index: int, value: str) -> None:
        self.face_expr_coeffs[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_model_change(self, index: int, value: str) -> None:
        self.model_parameters[0, self.model_param_indices[index]] = float(value)
        self._schedule_mesh_update()

    def _on_identity_var_change(self, index: int, value: str) -> None:
        self.identity_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expression_var_change(self, index: int, value: str) -> None:
        self.expression_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_model_var_change(self, index: int, value: str) -> None:
        self.model_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_joint_toggle(self) -> None:
        if self._last_joints is not None:
            self._update_joint_overlay(self._last_joints)
        self._schedule_mesh_update()

    def _reset_all(self) -> None:
        for slider_group in [
            self._identity_param_sliders,
            self._identity_var_sliders,
            self._expression_param_sliders,
            self._expression_var_sliders,
            self._model_param_sliders,
            self._model_var_sliders,
        ]:
            for slider in slider_group:
                slider.set(0.0)

    def _on_close(self) -> None:
        try:
            self.renderer.delete()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MHR viewer with CLUE uncertainty overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asset-folder",
        type=str,
        default=str(get_default_asset_folder()),
        help="Path to the MHR assets folder containing lod*.fbx and compact_v6_1.model.",
    )
    parser.add_argument("--lod", type=int, default=1, choices=list(range(7)), help="MHR level of detail.")
    parser.add_argument(
        "--num-identity-coeffs",
        type=int,
        default=10,
        help="Number of identity coefficients to expose in the viewer.",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        type=int,
        default=10,
        help="Number of facial expression coefficients to expose in the viewer.",
    )
    parser.add_argument(
        "--num-model-params",
        type=int,
        default=204 - _ROOT_MODEL_PARAM_COUNT,
        help="Number of non-root compact model parameters to expose in the viewer.",
    )
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        default=False,
        help=(
            "Render deterministic MHR pose correctives. CLUE does not propagate "
            "uncertainty through the non-linear corrective network."
        ),
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        default=False,
        help=(
            "Colour-code uncertainty by dominant variance axis "
            "(x=red, y=yellow, z=green). "
            "Without this flag, only the uncertainty amplitude is shown."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = MhrInteractiveApp(
        asset_folder=os.path.expanduser(os.path.expandvars(args.asset_folder)),
        lod=args.lod,
        num_identity_coeffs=args.num_identity_coeffs,
        num_expression_coeffs=args.num_expression_coeffs,
        num_model_params=args.num_model_params,
        apply_correctives=args.apply_correctives,
        direction=args.direction,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
