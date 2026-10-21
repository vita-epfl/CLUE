"""Interactive SMPL-X viewer with CLUE uncertainty overlays.

The viewer combines a Tkinter control panel with a pyrender-based mesh preview.
It supports deterministic SMPL-X edits and uncertainty amplitude overlays.
"""

from __future__ import annotations

import argparse
import math
import os
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
_DEFAULT_MODEL_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
)

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk

    _RESAMPLE = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or Image.LANCZOS
    try:
        _CBAR_FONT = ImageFont.truetype("segoeui.ttf", 11)
    except OSError:
        _CBAR_FONT = ImageFont.load_default(size=11)
    except TypeError:                           # Pillow < 10
        _CBAR_FONT = ImageFont.load_default()
except ImportError as exc:
    raise ImportError("Pillow is required: pip install Pillow") from exc

# Try to load the `turbo` colormap used by the evaluation and video exports.
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
# These must match the PBR material's baseColorFactor so that the vertex-
# colour mesh starts at the same visual colour as the deterministic mesh,
# producing a seamless transition.
# Use the same base colour as the batch exports when `turbo` is available.
try:
    if _TURBO is not None:
        _BASE_COLOR = np.array((_TURBO(0.0)[:3]), dtype=np.float32)
    else:
        _BASE_COLOR = np.array([0.05, 0.14, 0.28], dtype=np.float32)   # dark blue – matches material
except NameError:
    _BASE_COLOR = np.array([0.05, 0.14, 0.28], dtype=np.float32)

_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)   # red – max uncertainty

# ---------------------------------------------------------------------------
# Joint-overlay constants
# ---------------------------------------------------------------------------
_JOINT_RADIUS      = 0.008   # 8 mm spheres
_JOINT_LINE_RADIUS = 0.002   # 2 mm connection cylinders


def _cylinder_between(p1: np.ndarray, p2: np.ndarray, radius: float):
    """Return a trimesh cylinder connecting p1 to p2, or None if degenerate."""
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return None
    direction = vec / length
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, direction)
    cross_len = float(np.linalg.norm(cross))
    dot = float(np.dot(z, direction))
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


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    """Colour vertices by uncertainty amplitude only (no direction info).

    t=0 → base mesh colour.
    t=1 → red.
    """
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    if _TURBO is not None:
        return (_TURBO(tt) * 255).astype(np.uint8)
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

import smplx
from smplx.joint_names import JOINT_NAMES

from algorithm import CLUE


def _parse_bool_flag(value: str) -> bool:
    """Parse common true/false strings for CLI flags."""
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes"}:
        return True
    if normalised in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: true, false, 1, 0, yes, no."
    )


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Orbit camera that circles around a target point.

    Azimuth  – yaw around the world Y-axis (degrees)
    Elevation – pitch up/down (degrees, clamped ±85°)
    """

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
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)

        cam_pos = self._target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])

        forward = self._target - cam_pos
        flen    = np.linalg.norm(forward)
        if flen < 1e-8:
            return np.eye(4)
        forward /= flen

        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up    = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos
        return pose


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SmplxInteractiveApp:
    """Interactive SMPL-X viewer with deterministic and uncertainty controls."""

    SHAPE_SLIDERS      = 10
    EXPRESSION_SLIDERS = 10

    CTRL_WIDTH         = 620
    VIEWER_W           = 900
    VIEWER_H           = 900
    RENDER_INTERVAL_MS = 15

    PARAM_SLIDER_MIN   = -3.0
    PARAM_SLIDER_MAX   =  3.0
    SHAPE_EXPR_VAR_MAX =  0.5
    POSE_VAR_MAX       =  0.1

    def __init__(
        self,
        model_folder: str,
        num_betas: int,
        num_expression_coeffs: int,
        use_face_contour: bool,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = smplx.create(
            model_folder,
            model_type="smplx",
            gender="neutral",
            use_face_contour=use_face_contour,
            num_betas=max(num_betas, self.SHAPE_SLIDERS),
            num_expression_coeffs=max(num_expression_coeffs, self.EXPRESSION_SLIDERS),
            use_pca=False,
            ext="npz",
            batch_size=1,
        ).to(self.device)
        self.model.eval()

        self.n_shape        = min(self.SHAPE_SLIDERS,      self.model.num_betas)
        self.n_expr         = min(self.EXPRESSION_SLIDERS, self.model.num_expression_coeffs)
        self.n_pose_joints  = int(self.model.parents.numel())

        self._idx_global     = 0
        self._idx_body_start = 1
        self._idx_body_end   = self._idx_body_start + self.model.NUM_BODY_JOINTS
        self._idx_jaw        = self._idx_body_end
        self._idx_leye       = self._idx_jaw + 1
        self._idx_reye       = self._idx_leye + 1
        self._idx_lhand_start = self._idx_reye + 1
        self._idx_lhand_end   = self._idx_lhand_start + self.model.NUM_HAND_JOINTS
        self._idx_rhand_start = self._idx_lhand_end
        self._idx_rhand_end   = self._idx_rhand_start + self.model.NUM_HAND_JOINTS

        self._joint_names = list(JOINT_NAMES[: self.n_pose_joints])
        if len(self._joint_names) < self.n_pose_joints:
            self._joint_names.extend(
                [f"joint_{i}" for i in range(len(self._joint_names), self.n_pose_joints)]
            )

        self.neck_joint_idx = self._joint_names.index("neck") if "neck" in self._joint_names else 12
        self.jaw_joint_idx  = self._joint_names.index("jaw")  if "jaw"  in self._joint_names else 22
        self._pose_control_joint_indices = [
            joint_idx for joint_idx in range(self.n_pose_joints)
            if joint_idx != self._idx_global
        ]

        # Pose and shape parameter tensors
        self.betas       = torch.zeros(1, self.model.num_betas,              dtype=torch.float32, device=self.device)
        self.expression  = torch.zeros(1, self.model.num_expression_coeffs,  dtype=torch.float32, device=self.device)
        self.joint_params = torch.zeros(1, self.n_pose_joints, 3,            dtype=torch.float32, device=self.device)

        # Variance parameter tensors
        self.shape_var_params = torch.zeros(1, self.n_shape,      dtype=torch.float32, device=self.device)
        self.expr_var_params  = torch.zeros(1, self.n_expr,       dtype=torch.float32, device=self.device)
        self.joint_var_params = torch.zeros(1, self.n_pose_joints, 3, dtype=torch.float32, device=self.device)
        self._var_range: float | None = None

        # Per-parameter precomputed stds (populated in _precompute_param_data)
        self._param_std_ref:       dict = {}
        self._param_group_max_std: dict = {}

        self.clue = CLUE(self.model)

        # PBR materials — OPAQUE only (no transparency)
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[float(_BASE_COLOR[0]), float(_BASE_COLOR[1]), float(_BASE_COLOR[2]), 1.0],
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

        self._joint_nodes: list = []
        self._last_joints: np.ndarray | None = None

        vertices, joints = self._compute_vertices()
        self._last_joints = joints
        self.scene, self.mesh_node, self.center, self.cam_node, extent = self._make_scene(vertices, joints)

        self.orbit    = OrbitCamera(distance=2.1 * extent)
        self.renderer = pyrender.OffscreenRenderer(self.VIEWER_W, self.VIEWER_H)

        # Pre-compute per-parameter sensitivity patterns
        self._precompute_param_data()

        # ── Tkinter window ──────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("SMPL-X Interactive Viewer")
        self._show_joints = tk.BooleanVar(value=False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Header.TLabel",   font=("Segoe UI", 11, "bold"),  foreground="#2c3e50")
        style.configure("Section.TLabel",  font=("Segoe UI", 9,  "bold"),  foreground="#34495e", padding=(0, 6, 0, 2))
        style.configure("Variance.TLabel", font=("Segoe UI", 9,  "bold"),  foreground="#8e44ad", padding=(0, 6, 0, 2))
        style.configure("Control.TLabel",  font=("Segoe UI", 9),           foreground="#555555")
        style.configure("Hint.TLabel",     font=("Segoe UI", 8),           foreground="#888888")
        style.configure("TScale",          background="#ecf0f1")
        style.configure("Reset.TButton",   font=("Segoe UI", 9),           padding=6)

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
        self._angle_label = ttk.Label(hint_bar, text="az=0.0°  el=20.0°", style="Hint.TLabel")
        self._angle_label.pack(side=tk.LEFT)
        ttk.Label(
            hint_bar,
            text="Drag: rotate  |  Right-drag: pan  |  Scroll: zoom  |  Middle-click: reset pan",
            style="Hint.TLabel",
        ).pack(side=tk.RIGHT)

        self._drag_start:  tuple[int, int] | None = None
        self._drag_button: int | None = None
        self.canvas.bind("<ButtonPress-1>", lambda e: self._on_mouse_press(e, 1))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_mouse_press(e, 3))
        self.canvas.bind("<B1-Motion>",     self._on_mouse_drag)
        self.canvas.bind("<B3-Motion>",     self._on_mouse_drag)
        self.canvas.bind("<MouseWheel>",    self._on_scroll)
        self.canvas.bind("<Button-2>",      self._on_middle_click)
        self.canvas.bind("<Configure>",     lambda _e: self._mark_dirty())

        self._scheduled_update = None
        self._dirty       = True
        self._photo_image = None

        self._shape_param_sliders: list[ttk.Scale] = []
        self._expr_param_sliders: list[ttk.Scale] = []
        self._shape_var_sliders: list[ttk.Scale] = []
        self._expr_var_sliders: list[ttk.Scale] = []
        self._joint_param_sliders: list[ttk.Scale] = []
        self._joint_var_sliders: list[ttk.Scale] = []

        self._build_controls()

        init_w = self.CTRL_WIDTH + 2 + self.VIEWER_W
        self.root.geometry(f"{init_w}x{self.VIEWER_H}+50+50")
        self.root.minsize(760, 460)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.RENDER_INTERVAL_MS, self._render_loop)

    # ── SMPL-X helpers ───────────────────────────────────────────────────────

    def _variance_active(self) -> bool:
        """Return whether any uncertainty slider is currently active."""
        return bool(
            (self.shape_var_params > 0).any().item()
            or (self.expr_var_params > 0).any().item()
            or (self.joint_var_params[:, self._pose_control_joint_indices, :] > 0).any().item()
        )

    def _joint_pose_kwargs(self) -> dict[str, torch.Tensor]:
        """Build the SMPL-X pose kwargs expected by the model and CLUE."""
        return dict(
            global_orient=self.joint_params[:, self._idx_global, :],
            body_pose=self.joint_params[:, self._idx_body_start : self._idx_body_end, :].reshape(1, -1),
            jaw_pose=self.joint_params[:, self._idx_jaw, :],
            leye_pose=self.joint_params[:, self._idx_leye, :],
            reye_pose=self.joint_params[:, self._idx_reye, :],
            left_hand_pose=self.joint_params[:, self._idx_lhand_start : self._idx_lhand_end, :].reshape(1, -1),
            right_hand_pose=self.joint_params[:, self._idx_rhand_start : self._idx_rhand_end, :].reshape(1, -1),
        )

    def _compute_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        """Run a single SMPL-X forward pass and return vertices and joints."""
        with torch.no_grad():
            output = self.model(
                betas=self.betas,
                expression=self.expression,
                **self._joint_pose_kwargs(),
                return_verts=True,
            )
        return output.vertices[0].detach().cpu().numpy(), output.joints[0].detach().cpu().numpy()

    def _build_beta_vars(self) -> torch.Tensor:
        """Construct the concatenated shape+expression variance vector."""
        beta_vars = torch.zeros(
            1,
            self.model.shapedirs.shape[2] + self.model.expr_dirs.shape[2],
            dtype=torch.float32,
            device=self.device,
        )
        beta_vars[0, : self.n_shape] = self.shape_var_params[0]
        beta_vars[0, self.model.shapedirs.shape[2] : self.model.shapedirs.shape[2] + self.n_expr] = (
            self.expr_var_params[0]
        )
        return beta_vars

    def _build_beta_vars_max(self) -> torch.Tensor:
        """Construct the max-slider binary shape+expression variance vector."""
        beta_vars = torch.zeros(
            1,
            self.model.shapedirs.shape[2] + self.model.expr_dirs.shape[2],
            dtype=torch.float32,
            device=self.device,
        )
        beta_vars[0, : self.n_shape] = (self.shape_var_params[0] > 0).float() * self.SHAPE_EXPR_VAR_MAX
        beta_vars[0, self.model.shapedirs.shape[2] : self.model.shapedirs.shape[2] + self.n_expr] = (
            (self.expr_var_params[0] > 0).float() * self.SHAPE_EXPR_VAR_MAX
        )
        return beta_vars

    def _build_pose_vars(self) -> torch.Tensor:
        """Flatten non-pelvis per-joint pose variances to full-pose order."""
        pose_vars = self.joint_var_params.clone()
        pose_vars[:, self._idx_global, :] = 0.0
        return pose_vars.reshape(1, -1)

    def _build_pose_vars_max(self) -> torch.Tensor:
        """Flatten a max-slider binary pose-variance mask to full-pose order."""
        pose_vars = self.joint_var_params.clone()
        pose_vars[:, self._idx_global, :] = 0.0
        return (pose_vars > 0).float().reshape(1, -1) * self.POSE_VAR_MAX

    def _precompute_param_data(self) -> None:
        """Precompute per-vertex sensitivity patterns and per-group max stds.

        For each parameter group, CLUE is called at var_max to get
        the spatial std pattern.  The per-group max std anchors the colorbar.
        """
        n_beta_total = self.model.shapedirs.shape[2] + self.model.expr_dirs.shape[2]
        n_pose_total = self.n_pose_joints * 3

        # (group_name, n_params, beta_offset_or_None, pose_offset_or_None, var_max)
        groups = [
            ("shape", self.n_shape, 0,                            None, self.SHAPE_EXPR_VAR_MAX),
            ("expr",  self.n_expr,  self.model.shapedirs.shape[2], None, self.SHAPE_EXPR_VAR_MAX),
        ]

        for group_name, n_params, beta_off, pose_off, var_max in groups:
            for idx in range(n_params):
                key = (group_name, idx)

                beta_vars = torch.zeros(1, n_beta_total, dtype=torch.float32, device=self.device)
                pose_vars = torch.zeros(1, n_pose_total, dtype=torch.float32, device=self.device)
                if beta_off is not None:
                    beta_vars[0, beta_off + idx] = var_max
                if pose_off is not None:
                    pose_vars[0, pose_off + idx] = var_max

                total_var, _ = self.clue.compute_variance(
                    beta_vars, pose_vars,
                    betas=self.betas,
                    expression=self.expression,
                    **self._joint_pose_kwargs(),
                )
                var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)
                std     = np.sqrt(var_xyz.sum(axis=-1))
                self._param_std_ref[key] = float(std.max()) * 1000  # mm

            self._param_group_max_std[group_name] = max(
                self._param_std_ref.get((group_name, i), 0.0)
                for i in range(n_params)
            )

        # Pose group: all exposed pose controls, excluding pelvis/global rotation.
        pose_sample_indices = [
            joint_idx * 3 + axis_idx
            for joint_idx in self._pose_control_joint_indices
            for axis_idx in range(3)
        ]
        pose_stds = []
        for flat_idx in pose_sample_indices:
            pose_vars = torch.zeros(1, n_pose_total, dtype=torch.float32, device=self.device)
            pose_vars[0, flat_idx] = self.POSE_VAR_MAX
            total_var, _ = self.clue.compute_variance(
                torch.zeros(1, n_beta_total, dtype=torch.float32, device=self.device),
                pose_vars,
                betas=self.betas,
                expression=self.expression,
                **self._joint_pose_kwargs(),
            )
            var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)
            pose_stds.append(float(np.sqrt(var_xyz.sum(axis=-1)).max()) * 1000)
        self._param_group_max_std["pose"] = max(pose_stds) if pose_stds else 1e-12

    def _compute_per_param_colors(self) -> tuple[np.ndarray, float | None]:
        """Per-group amplitude coloring: each group normalised by its own peak σ.

        Returns
        -------
        vertex_colors : (V, 4) RGBA uint8
        std_ref_max   : colorbar ceiling in mm, or None when no variance active.
        """
        if not self._variance_active():
            V = int(self.model.v_template.shape[0])
            return _amplitude_colors(np.zeros(V, dtype=np.float32)), None

        common = dict(betas=self.betas, expression=self.expression, **self._joint_pose_kwargs())
        total_var, _ = self.clue.compute_variance(
            self._build_beta_vars(), self._build_pose_vars(), **common
        )
        var_xyz = np.maximum(total_var[0].detach().cpu().numpy(), 0.0)
        std     = np.sqrt(var_xyz.sum(axis=-1)) * 1000.0

        active_groups: set[str] = set()
        if (self.shape_var_params > 0).any():
            active_groups.add("shape")
        if (self.expr_var_params > 0).any():
            active_groups.add("expr")
        if (self.joint_var_params[:, self._pose_control_joint_indices, :] > 0).any():
            active_groups.add("pose")

        std_ref_max = max(
            (self._param_group_max_std.get(g, 1e-12) for g in active_groups),
            default=1e-12,
        )

        t = std / std_ref_max if std_ref_max > 1e-12 else np.zeros_like(std)
        return _amplitude_colors(t), float(std_ref_max)

    def _build_trimesh(
        self,
        vertices: np.ndarray,
        center: np.ndarray,
    ) -> trimesh.Trimesh:
        """Build a clean trimesh centered at the current scene origin."""
        tri_mesh = trimesh.Trimesh(vertices - center, self.model.faces, process=False)
        if hasattr(tri_mesh, "remove_degenerate_faces"):
            tri_mesh.remove_degenerate_faces()
        if hasattr(tri_mesh, "remove_duplicate_faces"):
            tri_mesh.remove_duplicate_faces()
        if hasattr(tri_mesh, "remove_unreferenced_vertices"):
            tri_mesh.remove_unreferenced_vertices()
        tri_mesh.fix_normals()
        return tri_mesh

    def _create_joint_overlay(
        self,
        joints: np.ndarray,
        center: np.ndarray,
        scene: pyrender.Scene,
    ) -> None:
        """Add coloured spheres and connection cylinders for SMPL-X joints."""
        joints_core = joints[: self.n_pose_joints] - center
        palette = np.array([
            [0.92, 0.34, 0.34],
            [0.31, 0.60, 0.88],
            [0.29, 0.74, 0.49],
            [0.95, 0.73, 0.23],
            [0.67, 0.43, 0.91],
            [0.96, 0.55, 0.29],
        ], dtype=np.float32)

        for j in range(self.n_pose_joints):
            sph   = trimesh.creation.uv_sphere(radius=_JOINT_RADIUS)
            color = palette[j % len(palette)]
            sph.visual.vertex_colors = np.array([color[0], color[1], color[2], 1.0])
            node  = scene.add(
                pyrender.Mesh.from_trimesh(sph),
                pose=trimesh.transformations.translation_matrix(joints_core[j]),
            )
            self._joint_nodes.append(node)

        parents = self.model.parents.detach().cpu().numpy().astype(np.int64)
        for child_idx in range(1, self.n_pose_joints):
            parent_idx = int(parents[child_idx])
            if parent_idx < 0 or parent_idx >= self.n_pose_joints:
                continue
            cyl = _cylinder_between(
                joints_core[parent_idx],
                joints_core[child_idx],
                _JOINT_LINE_RADIUS,
            )
            if cyl is None:
                continue
            color = palette[child_idx % len(palette)]
            cyl.visual.vertex_colors = np.array([color[0], color[1], color[2], 1.0])
            self._joint_nodes.append(scene.add(pyrender.Mesh.from_trimesh(cyl)))

    def _update_joint_overlay(self, joints: np.ndarray) -> None:
        """Replace joint overlay nodes to match updated joint positions."""
        for node in self._joint_nodes:
            self.scene.remove_node(node)
        self._joint_nodes.clear()
        self._last_joints = joints
        if self._show_joints.get() and self._last_joints is not None:
            self._create_joint_overlay(self._last_joints, self.center, self.scene)

    def _make_scene(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
    ) -> tuple[pyrender.Scene, pyrender.Node, np.ndarray, pyrender.Node, float]:
        """Create the initial viewer scene, mesh node, and camera node."""
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = max(float(np.max(bounds[1] - bounds[0])), 1e-3)

        tri_mesh  = self._build_trimesh(vertices, center)
        mesh      = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)
        scene     = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.12, 0.13, 0.14])
        mesh_node = scene.add(mesh)
        cam_node  = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=np.eye(4))

        def _look_at_light(direction, intensity):
            d = np.asarray(direction, dtype=np.float64)
            d /= np.linalg.norm(d) + 1e-8
            z  = -d
            up = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(z, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z);  x /= np.linalg.norm(x) + 1e-8
            y = np.cross(z, x)
            lp = np.eye(4)
            lp[:3, 0], lp[:3, 1], lp[:3, 2], lp[:3, 3] = x, y, z, -d * 2.5 * extent
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=lp)

        _look_at_light([-0.4, -0.55, -1.0], 3.5)
        _look_at_light([ 0.8, -0.2,  -0.6], 1.75)
        _look_at_light([ 0.0,  1.0,  -0.5], 0.75)

        self._create_joint_overlay(joints, center, scene)

        return scene, mesh_node, center, cam_node, extent

    # ── Colorbar ──────────────────────────────────────────────────────────────

    def _draw_colorbar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
        """Render the uncertainty colour bar in the top-right corner of img."""
        return self._draw_amplitude_bar(img, std_ref_max)

    def _draw_amplitude_bar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
        """Vertical gradient colorbar: base colour (bottom/low) → red (top/high)."""
        draw     = ImageDraw.Draw(img)
        iw, ih   = img.size
        bar_w    = 36
        bar_h    = 240
        margin_r = 14
        margin_t = 14
        pad      = 10
        label_w  = 70

        panel_w = bar_w + label_w + pad * 2
        panel_h = bar_h + pad * 2 + 20

        px = iw - margin_r - panel_w
        py = margin_t

        draw.rectangle([px, py, px + panel_w, py + panel_h],
                       fill=(245, 245, 245), outline=(190, 190, 190))
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

        draw.text((px + pad, by),              f"{std_ref_max:.2f}mm", fill=(30, 30, 30), font=_CBAR_FONT)
        draw.text((px + pad, by + bar_h - 14), "0 mm",                 fill=(30, 30, 30), font=_CBAR_FONT)
        return img

    # ── Render loop ──────────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        """Request a viewer redraw on the next render-loop iteration."""
        self._dirty = True

    def _render_loop(self) -> None:
        if self._dirty:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                self.scene.set_pose(self.cam_node, pose=self.orbit.get_pose())
                color, _ = self.renderer.render(self.scene)

                img   = Image.fromarray(color)
                scale = min(cw / self.VIEWER_W, ch / self.VIEWER_H)
                if abs(scale - 1.0) > 0.005:
                    img = img.resize(
                        (max(1, int(self.VIEWER_W * scale)),
                         max(1, int(self.VIEWER_H * scale))),
                        _RESAMPLE,
                    )

                if self._var_range is not None:
                    img = self._draw_colorbar(img, self._var_range)

                self._photo_image = ImageTk.PhotoImage(image=img)
                self.canvas.delete("all")
                self.canvas.create_image(cw // 2, ch // 2, image=self._photo_image, anchor="center")
                self._angle_label.configure(
                    text=f"az={self.orbit.azimuth:.1f}°  el={self.orbit.elevation:.1f}°"
                )
                self._dirty = False

        self.root.after(self.RENDER_INTERVAL_MS, self._render_loop)

    # ── Mouse interaction ────────────────────────────────────────────────────

    def _on_mouse_press(self, event: tk.Event, button: int) -> None:
        self._drag_start  = (event.x, event.y)
        self._drag_button = button

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        dx, dy           = event.x - self._drag_start[0], event.y - self._drag_start[1]
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

    # ── Mesh update ──────────────────────────────────────────────────────────

    def _schedule_mesh_update(self) -> None:
        """Debounce slider-driven mesh recomputation onto the Tk event loop."""
        if self._scheduled_update is not None:
            self.root.after_cancel(self._scheduled_update)
        self._scheduled_update = self.root.after(15, self._update_mesh)

    def _update_mesh(self) -> None:
        self._scheduled_update = None
        vertices, joints = self._compute_vertices()
        tri_mesh         = self._build_trimesh(vertices, self.center)

        # Always OPAQUE — colormaps normalised per group, no transparency
        vertex_colors, std_ref_max = self._compute_per_param_colors()
        self._var_range = std_ref_max

        if len(vertex_colors) == len(tri_mesh.vertices):
            prim = pyrender.Primitive(
                positions=tri_mesh.vertices.astype(np.float32),
                normals=tri_mesh.vertex_normals.astype(np.float32),
                indices=tri_mesh.faces.astype(np.uint32),
                color_0=vertex_colors,
                material=self.vc_material,
            )
            new_mesh = pyrender.Mesh(primitives=[prim])
        else:
            new_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=self.material, smooth=True)

        self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(new_mesh)
        self._update_joint_overlay(joints)
        self._mark_dirty()

    # ── Controls ─────────────────────────────────────────────────────────────

    def _build_scrollable_frame(self) -> ttk.Frame:
        """Create the scrollable control container used by the left panel."""
        container    = ttk.Frame(self._ctrl_frame, padding=0)
        container.pack(fill=tk.BOTH, expand=True)
        inner_canvas = tk.Canvas(container, highlightthickness=0, bg="white", borderwidth=0)
        scrollbar    = ttk.Scrollbar(container, orient=tk.VERTICAL, command=inner_canvas.yview)
        scrollable   = ttk.Frame(inner_canvas, style="TFrame")
        scrollable.bind(
            "<Configure>",
            lambda _: inner_canvas.configure(scrollregion=inner_canvas.bbox("all")),
        )
        inner_canvas.create_window((0, 0), window=scrollable, anchor="nw", width=self.CTRL_WIDTH - 20)
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
        """Render a full-width section header with an optional right label."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=(10, 2))
        ttk.Label(row, text=left_text,  style="Section.TLabel").pack(side=tk.LEFT)
        if right_text:
            ttk.Label(row, text=right_text, style="Variance.TLabel").pack(side=tk.RIGHT)

    def _slider_changed(
        self,
        value: str,
        label: ttk.Label,
        callback: Callable[[str], None],
    ) -> None:
        """Update the numeric label beside a slider and forward the callback."""
        label.configure(text=f"{float(value):+.3f}")
        callback(value)

    def _make_slider_pair(
        self,
        parent,
        param_text,
        param_from,
        param_to,
        param_cb,
        var_cb,
        var_to,
        param_list,
        var_list,
    ) -> None:
        """Create a paired parameter/variance slider row."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=2)

        left  = ttk.Frame(row)
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right = ttk.Frame(row)
        right.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        ttk.Label(left, text=param_text, width=20).pack(side=tk.LEFT)
        val_l = ttk.Label(left, text="+0.000", width=8)
        val_l.pack(side=tk.LEFT)
        s_l = ttk.Scale(
            left, from_=param_from, to=param_to, orient=tk.HORIZONTAL,
            command=lambda v: self._slider_changed(v, val_l, param_cb),
        )
        s_l.pack(side=tk.LEFT, fill=tk.X, expand=True)

        val_r = ttk.Label(right, text="+0.000", width=8)
        val_r.pack(side=tk.LEFT)
        s_r = ttk.Scale(
            right, from_=0.0, to=var_to, orient=tk.HORIZONTAL,
            command=lambda v: self._slider_changed(v, val_r, var_cb),
        )
        s_r.pack(side=tk.LEFT, fill=tk.X, expand=True)

        param_list.append(s_l)
        var_list.append(s_r)

    def _build_controls(self) -> None:
        """Populate the left control panel with parameter and variance sliders."""
        content = self._build_scrollable_frame()

        ttk.Label(content, text="SMPL-X Controls", style="Header.TLabel").pack(anchor=tk.W, padx=10, pady=(10, 4))

        self._make_section_header(content, "Shape Parameters", "σ²")
        for i in range(self.n_shape):
            self._make_slider_pair(
                content,
                f"shape[{i}]",
                self.PARAM_SLIDER_MIN,
                self.PARAM_SLIDER_MAX,
                lambda v, idx=i: self._on_shape_change(idx, v),
                lambda v, idx=i: self._on_shape_var_change(idx, v),
                self.SHAPE_EXPR_VAR_MAX,
                self._shape_param_sliders,
                self._shape_var_sliders,
            )

        self._make_section_header(content, "Expression Parameters", "σ²")
        for i in range(self.n_expr):
            self._make_slider_pair(
                content,
                f"expr[{i}]",
                self.PARAM_SLIDER_MIN,
                self.PARAM_SLIDER_MAX,
                lambda v, idx=i: self._on_expression_change(idx, v),
                lambda v, idx=i: self._on_expr_var_change(idx, v),
                self.SHAPE_EXPR_VAR_MAX,
                self._expr_param_sliders,
                self._expr_var_sliders,
            )

        self._make_section_header(content, "Joint Pose Parameters", "σ²")
        for joint_idx in self._pose_control_joint_indices:
            joint_name = self._joint_names[joint_idx]
            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                self._make_slider_pair(
                    content,
                    f"{joint_name}[{axis_name}]",
                    -0.8,
                    0.8,
                    lambda v, j=joint_idx, a=axis_idx: self._on_joint_change(j, a, v),
                    lambda v, j=joint_idx, a=axis_idx: self._on_joint_var_change(j, a, v),
                    self.POSE_VAR_MAX,
                    self._joint_param_sliders,
                    self._joint_var_sliders,
                )

        toggles = ttk.Frame(content)
        toggles.pack(fill=tk.X, padx=10, pady=(10, 4))
        ttk.Checkbutton(
            toggles,
            text="Show all SMPL-X joints",
            variable=self._show_joints,
            command=self._on_joint_toggle,
        ).pack(side=tk.LEFT)

        btns = ttk.Frame(content)
        btns.pack(fill=tk.X, padx=10, pady=(2, 10))
        ttk.Button(btns, text="Reset all", command=self._reset_all).pack(side=tk.LEFT)

    # ── Slider callbacks ──────────────────────────────────────────────────────

    def _on_shape_change(self, index: int, value: str) -> None:
        self.betas[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expression_change(self, index: int, value: str) -> None:
        self.expression[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_joint_change(self, joint_index: int, axis_index: int, value: str) -> None:
        self.joint_params[0, joint_index, axis_index] = float(value)
        self._schedule_mesh_update()

    def _on_shape_var_change(self, index: int, value: str) -> None:
        self.shape_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expr_var_change(self, index: int, value: str) -> None:
        self.expr_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_joint_var_change(
        self,
        joint_index: int,
        axis_index: int,
        value: str,
    ) -> None:
        self.joint_var_params[0, joint_index, axis_index] = float(value)
        self._schedule_mesh_update()

    def _on_joint_toggle(self) -> None:
        if self._last_joints is not None:
            self._update_joint_overlay(self._last_joints)
        self._schedule_mesh_update()

    def _reset_all(self) -> None:
        for seq in [
            self._shape_param_sliders,
            self._expr_param_sliders,
            self._shape_var_sliders,
            self._expr_var_sliders,
            self._joint_param_sliders,
            self._joint_var_sliders,
        ]:
            for s in seq:
                s.set(0.0)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        try:
            self.renderer.delete()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the public SMPL-X viewer entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive SMPL-X viewer with CLUE uncertainty overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-folder",
        required=False,
        default=_DEFAULT_MODEL_FOLDER,
        type=str,
        help="Path to the folder containing smplx/SMPLX_NEUTRAL.npz.",
    )
    parser.add_argument(
        "--num-betas",
        default=10,
        type=int,
        help="Number of shape coefficients to load into the viewer.",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        default=10,
        type=int,
        help="Number of expression coefficients to load into the viewer.",
    )
    parser.add_argument(
        "--use-face-contour",
        default=False,
        type=_parse_bool_flag,
        help="Enable dynamic face-contour landmarks in the SMPL-X forward pass.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_folder = os.path.expanduser(os.path.expandvars(args.model_folder))

    app = SmplxInteractiveApp(
        model_folder=model_folder,
        num_betas=args.num_betas,
        num_expression_coeffs=args.num_expression_coeffs,
        use_face_contour=args.use_face_contour,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
