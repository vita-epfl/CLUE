"""Interactive FLAME viewer with CLUE uncertainty overlays.

The viewer combines a Tkinter control panel with a pyrender-based mesh preview.
It supports deterministic FLAME edits, uncertainty amplitude overlays, and an
optional ``--direction`` mode that colors vertices by dominant variance axis.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Iterator, Sequence

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

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

# Per-axis target colours (x=red, y=yellow, z=green)
_DIR_COLORS = np.array([
    [0.85, 0.05, 0.05],   # x → red
    [0.85, 0.80, 0.05],   # y → yellow
    [0.05, 0.65, 0.05],   # z → green
], dtype=np.float32)
_DIR_LABELS = ["x", "y", "z"]

# ---------------------------------------------------------------------------
# Joint-sphere constants  (FLAME kintree order: global, neck, jaw, l_eye, r_eye)
# ---------------------------------------------------------------------------
_JOINT_NAMES  = ["global", "neck",        "jaw",        "l_eye",      "r_eye"]
_JOINT_COLORS = [
    [0.85, 0.85, 0.85],   # global  – light grey
    [0.00, 0.85, 0.85],   # neck    – cyan
    [1.00, 0.50, 0.00],   # jaw     – orange
    [0.30, 1.00, 0.30],   # l_eye   – green
    [1.00, 0.30, 1.00],   # r_eye   – magenta
]
_JOINT_RADIUS      = 0.004   # 4 mm spheres
_JOINT_LINE_RADIUS = 0.001   # 1 mm connection cylinders


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
        R = trimesh.transformations.rotation_matrix(
            np.arctan2(cross_len, dot), cross / cross_len
        )
    elif dot < 0.0:
        R = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        R = np.eye(4)
    T = trimesh.transformations.translation_matrix((p1 + p2) * 0.5)
    return trimesh.creation.cylinder(radius=radius, height=length, sections=8, transform=T @ R)


def _dir_colors(var_xyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Colour vertices by dominant variance axis, scaled by total uncertainty.

    Parameters
    ----------
    var_xyz : (V, 3) per-vertex variance in x, y, z
    t       : (V,) normalised total uncertainty in [0, 1]

    Returns
    -------
    (V, 4) RGBA uint8
        t=0 → base mesh colour regardless of dominant axis.
        t=1 → pure axis colour (red / yellow / green).
    """
    dominant  = var_xyz.argmax(axis=-1)              # (V,) in {0, 1, 2}
    target    = _DIR_COLORS[dominant]                 # (V, 3)
    t_col     = t[:, np.newaxis]                      # (V, 1)
    rgb       = (1.0 - t_col) * _BASE_COLOR + t_col * target
    alpha     = np.ones((len(t), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)   # red – max uncertainty


def _amplitude_colors(t: np.ndarray) -> np.ndarray:
    """Colour vertices by uncertainty amplitude only (no direction info).

    t=0 → base mesh colour.
    t=1 → red.
    """
    # Prefer the shared turbo amplitude colormap when available.
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


@contextmanager
def _temporary_cli_args(argv: Sequence[str]) -> Iterator[None]:
    """Temporarily replace ``sys.argv`` for libraries that parse CLI flags."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *argv]
        yield
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Orbit camera that circles around a target point.

    Azimuth  – yaw  around the world Y-axis (degrees)
    Elevation – pitch up/down               (degrees, clamped ±85°)
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
# Main application
# ---------------------------------------------------------------------------

class FlameInteractiveApp:
    SHAPE_SLIDERS      = 10
    EXPRESSION_SLIDERS = 10

    CTRL_WIDTH         = 560    # fixed width of the control panel (px)
    VIEWER_W           = 800    # offscreen render resolution
    VIEWER_H           = 800
    RENDER_INTERVAL_MS = 15
    VAR_SLIDER_MAX      = 0.5   # maximum value of shape/expression σ² sliders
    POSE_VAR_SLIDER_MAX = 0.1   # maximum value of jaw/neck σ² sliders

    def __init__(self, direction: bool = True):
        self._direction_mode = direction
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.flame = FLAME(self.config).to(self.device)
        self.flame.eval()

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

        # ── Variance ─────────────────────────────────────────────────────────
        self.clue = CLUE(self.flame)

        # Total beta counts (300 shape + 100 expression in the default FLAME model).
        self._shape_beta_total: int = (
            self.flame.shape_betas.shape[1] + self.config.shape_params
        )

        self.shape_var_params = torch.zeros(
            1, self.SHAPE_SLIDERS, dtype=torch.float32, device=self.device
        )
        self.expr_var_params = torch.zeros(
            1, self.EXPRESSION_SLIDERS, dtype=torch.float32, device=self.device
        )
        # pose_vars ordering mirrors full_pose: [global(3) | neck(3) | jaw(3) | eye(6)]
        # neck → joints slot 1, jaw → joints slot 2
        self.neck_var_params = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self.jaw_var_params  = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self._var_range: float | None = None         # std_ref_max for colorbar

        # Per-parameter reference standard deviations populated by
        # _precompute_param_data() at the end of __init__.
        self._param_std_ref:       dict = {}   # (group, idx) → float, max σ in mm at var_max input
        self._param_group_max_std: dict = {}   # group → max σ in mm across all params in group

        # ── Joint visualization state (must exist before scene) ──────────────
        self._joint_nodes: list = []
        self._last_joints: np.ndarray | None = None

        # ── Scene ────────────────────────────────────────────────────────────
        vertices, joints = self._compute_vertices()
        self._last_joints = joints
        self.scene, self.mesh_node, self.center, self.cam_node, extent = (
            self._make_scene(vertices, joints)
        )
        self.orbit   = OrbitCamera(distance=2.1 * extent)
        self.renderer = pyrender.OffscreenRenderer(self.VIEWER_W, self.VIEWER_H)

        # Pre-compute the CLUE ranges used to normalize uncertainty colours.
        self._precompute_param_data()

        # ── Unified Tkinter window ──────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("FLAME Interactive Viewer")
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

        # Left panel: controls (fixed width)
        self._ctrl_frame = ttk.Frame(self.root, width=self.CTRL_WIDTH)
        self._ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._ctrl_frame.pack_propagate(False)

        ttk.Separator(self.root, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))

        # Right panel: viewer
        view_frame = ttk.Frame(self.root)
        view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            view_frame, bg="white", highlightthickness=0, cursor="crosshair"
        )
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

        # Mouse bindings
        self._drag_start: tuple[int, int] | None = None
        self._drag_button: int | None = None
        self.canvas.bind("<ButtonPress-1>", lambda e: self._on_mouse_press(e, 1))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._on_mouse_press(e, 3))
        self.canvas.bind("<B1-Motion>",     self._on_mouse_drag)
        self.canvas.bind("<B3-Motion>",     self._on_mouse_drag)
        self.canvas.bind("<MouseWheel>",    self._on_scroll)
        self.canvas.bind("<Button-2>",      self._on_middle_click)
        self.canvas.bind("<Configure>",     lambda _e: self._mark_dirty())

        self._scheduled_update: str | None = None
        self._dirty       = True
        self._photo_image = None

        self._build_controls()

        init_w = self.CTRL_WIDTH + 2 + self.VIEWER_W
        self.root.geometry(f"{init_w}x{self.VIEWER_H}+50+50")
        self.root.minsize(700, 400)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.RENDER_INTERVAL_MS, self._render_loop)

    # ── FLAME helpers ────────────────────────────────────────────────────────

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

    def _create_joint_spheres(
        self, joints: np.ndarray, center: np.ndarray, scene: pyrender.Scene
    ) -> None:
        """Add coloured spheres and connection cylinders for every FLAME joint."""
        n = min(len(_JOINT_NAMES), len(joints))
        # Spheres at each joint position
        for j in range(n):
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=_JOINT_RADIUS)
            rgba = np.array([*_JOINT_COLORS[j], 1.0], dtype=np.float32)
            sphere.visual.vertex_colors = np.tile(rgba, (len(sphere.vertices), 1))
            sphere.apply_translation(joints[j] - center)
            self._joint_nodes.append(
                scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=False))
            )
        # Cylinders following the actual kinematic chain (parents[0]=-1, rest from kintree)
        parents_np = self.flame.parents.cpu().numpy()
        for j in range(1, n):
            parent_j = int(parents_np[j])
            if parent_j < 0 or parent_j >= n:
                continue
            cyl = _cylinder_between(joints[parent_j] - center, joints[j] - center, _JOINT_LINE_RADIUS)
            if cyl is None:
                continue
            rgba = np.array([*_JOINT_COLORS[j], 1.0], dtype=np.float32)
            cyl.visual.vertex_colors = np.tile(rgba, (len(cyl.vertices), 1))
            self._joint_nodes.append(
                scene.add(pyrender.Mesh.from_trimesh(cyl, smooth=False))
            )

    def _update_joint_spheres(self, joints: np.ndarray) -> None:
        """Replace joint sphere nodes to match updated joint positions."""
        for node in self._joint_nodes:
            self.scene.remove_node(node)
        self._joint_nodes.clear()
        self._last_joints = joints
        if self._show_joints.get():
            self._create_joint_spheres(joints, self.center, self.scene)

    def _on_joint_toggle(self) -> None:
        if self._last_joints is not None:
            self._update_joint_spheres(self._last_joints)
        self._schedule_mesh_update()

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
            x = np.cross(up, z);  x /= np.linalg.norm(x) + 1e-8
            y = np.cross(z, x)
            lpose = np.eye(4)
            lpose[:3, 0], lpose[:3, 1], lpose[:3, 2], lpose[:3, 3] = x, y, z, -d * 2.5 * extent
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=lpose)

        # Directional lights with reduced intensities (dimmer scene)
        _look_at_light([-0.4, -0.55, -1.0], 3.5)
        _look_at_light([ 0.8, -0.2,  -0.6], 1.75)
        _look_at_light([ 0.0,  1.0,  -0.5], 0.75)

        self._create_joint_spheres(joints, center, scene)

        return scene, mesh_node, center, cam_node, extent

    # ── Variance helpers ─────────────────────────────────────────────────────

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
        """Beta-var tensor with every *active* (> 0) entry set to VAR_SLIDER_MAX.

        Used to compute the per-vertex uncertainty ceiling so that actual
        colours can be expressed as a fraction of that ceiling.
        """
        n_total  = self.flame.shapedirs.shape[2]
        bv_max   = torch.zeros(1, n_total, dtype=torch.float32, device=self.device)
        n_sv     = self.shape_var_params.shape[1]
        bv_max[0, :n_sv]               = (self.shape_var_params[0] > 0).float() * self.VAR_SLIDER_MAX
        n_ev     = self.expr_var_params.shape[1]
        off      = self._shape_beta_total
        bv_max[0, off : off + n_ev]    = (self.expr_var_params[0] > 0).float() * self.VAR_SLIDER_MAX
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
        """Pose-var ceiling: active (> 0) entries → POSE_VAR_SLIDER_MAX, rest → 0."""
        pose_vars_max = torch.zeros(1, 15, dtype=torch.float32, device=self.device)
        pose_vars_max[0, 3:6] = (self.neck_var_params[0] > 0).float() * self.POSE_VAR_SLIDER_MAX
        pose_vars_max[0, 6:9] = (self.jaw_var_params[0] > 0).float() * self.POSE_VAR_SLIDER_MAX
        return pose_vars_max

    def _precompute_param_data(self) -> None:
        """Precompute per-vertex sensitivity patterns and per-group max stds.

        For each parameter, CLUE is evaluated at var_max to get the
        spatial std pattern.  The per-group max std is the highest max-vertex-std
        found across all parameters in that group, used to anchor the colorbar.
        """
        n_total = self.flame.shapedirs.shape[2]   # 400 beta blendweights
        n_pose  = 15                               # 5 joints × 3 angles

        # (group_name, n_params, beta_offset_or_None, pose_offset_or_None, var_max)
        groups = [
            ("shape", self.SHAPE_SLIDERS,      0,                       None, self.VAR_SLIDER_MAX),
            ("expr",  self.EXPRESSION_SLIDERS, self._shape_beta_total,  None, self.VAR_SLIDER_MAX),
            ("neck",  3,                       None,                    3,    self.POSE_VAR_SLIDER_MAX),
            ("jaw",   3,                       None,                    6,    self.POSE_VAR_SLIDER_MAX),
        ]

        for group_name, n_params, beta_off, pose_off, var_max in groups:
            for idx in range(n_params):
                key = (group_name, idx)

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

    def _compute_variance_colors(self) -> tuple[np.ndarray, float]:
        """Propagate σ² → per-vertex σ → RGBA uint8 colours.

        Normalization uses a *global* ceiling: the maximum σ any vertex can
        reach when all currently-active sliders are set to VAR_SLIDER_MAX.

            t[v] = std_actual[v] / std_ref_max_global

        This preserves per-vertex spatial variation (vertices differ in how
        sensitive they are to each shape / expression parameter) while keeping
        colours proportional to slider values — a slider at 0.3 tints vertices
        at 30 % of their maximum red intensity.

        When no slider is active (all σ² = 0) std_actual = 0 → t = 0
        everywhere → _unc_colors returns the base mesh colour, giving a
        seamless visual match with the zero-variance state.

        Returns
        -------
        vertex_colors : (V, 4) RGBA uint8
        std_ref_max   : global σ ceiling (max σ at full slider=VAR_SLIDER_MAX uncertainty),
                        used for colorbar tick labels.
        """
        common_args = dict(
            shape_params=self.shape_params,
            expression_params=self.expression_params,
            pose_params=self.pose_params,
            neck_pose=self.neck_pose,
            eye_pose=self.eye_pose,
        )

        # Actual per-vertex variance (V, 3) and total σ (V,)
        total_var, _ = self.clue.compute_variance(
            self._build_beta_vars(), self._build_pose_vars(), **common_args
        )  # (1, V, 3)
        var_xyz = np.maximum(total_var[0].cpu().numpy(), 0.0)           # (V, 3)
        std     = np.sqrt(var_xyz.sum(axis=-1)) * 1000                  # (V,) mm

        # Ceiling: global max σ when all active sliders → 1
        total_var_max, _ = self.clue.compute_variance(
            self._build_beta_vars_max(), self._build_pose_vars_max(), **common_args
        )
        std_max_vx = np.sqrt(np.maximum(
            total_var_max[0].sum(dim=-1).cpu().numpy(), 0.0
        )) * 1000
        std_ref_max = float(std_max_vx.max())

        t = std / std_ref_max if std_ref_max > 0.0 else np.zeros_like(std)

        if self._direction_mode:
            return _dir_colors(var_xyz, t), std_ref_max
        else:
            return _amplitude_colors(t), std_ref_max

    def _compute_per_param_colors(self) -> tuple[np.ndarray, float | None]:
        """Per-group amplitude coloring matching the export scripts.

        Computes actual per-vertex std via CLUE, normalises by the
        precomputed highest-variance group max std, then maps through the same
        base-blue → red amplitude colormap used by the evaluation and video
        exports.

        Returns
        -------
        vertex_colors : (V, 4) RGBA uint8
        std_ref_max   : colorbar ceiling in mm, or None when no variance is active.
        """
        if not self._variance_active():
            # All sliders at zero — show base mesh colour (t=0 everywhere)
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

    # ── Colorbar ──────────────────────────────────────────────────────────────

    def _draw_colorbar(self, img: Image.Image, std_ref_max: float):
        """Render a legend in the top-right corner of img.

        Direction mode  → dominant-axis swatch legend (x/y/z colours).
        Amplitude mode  → vertical gradient bar from base colour (low) to red (high).

        Returns (img, panel_rect) where panel_rect is (px, py, panel_w, panel_h) for
        direction mode (used to position the axis gizmo) or None for amplitude mode.
        """
        if self._direction_mode:
            return self._draw_direction_legend(img, std_ref_max)
        else:
            return self._draw_amplitude_bar(img, std_ref_max), None

    def _draw_direction_legend(self, img: Image.Image, std_ref_max: float):
        """Dominant-axis direction legend with std dev text inside the panel."""
        draw       = ImageDraw.Draw(img)
        iw, ih     = img.size
        swatch_w   = 40
        swatch_h   = 26
        margin_r   = 14
        margin_t   = 14
        row_gap    = 6
        pad        = 10

        n = len(_DIR_LABELS)
        panel_w  = 170   # wide enough for "Axis of largest variation"
        swatches_h = n * swatch_h + (n - 1) * row_gap
        # top-pad + title + swatches + gap + separator + gap + std-text + bottom-pad
        panel_h  = pad + 20 + swatches_h + 8 + 1 + 6 + 14 + pad

        px = iw - margin_r - panel_w
        py = margin_t

        draw.rectangle(
            [px, py, px + panel_w, py + panel_h],
            fill=(245, 245, 245), outline=(190, 190, 190),
        )
        draw.text((px + pad, py + pad), "Axis of largest variation",
                  fill=(50, 50, 50), font=_CBAR_FONT)

        for i, (lbl, col) in enumerate(zip(_DIR_LABELS, _DIR_COLORS)):
            sx = px + pad
            sy = py + pad + 20 + i * (swatch_h + row_gap)
            rgb_u8 = tuple((col * 255).astype(np.uint8))
            draw.rectangle([sx, sy, sx + swatch_w, sy + swatch_h],
                           fill=rgb_u8, outline=(80, 80, 80))
            draw.text((sx + swatch_w + 6, sy + 5), lbl, fill=(30, 30, 30), font=_CBAR_FONT)

        # Thin separator above the std dev line
        sep_y = py + pad + 20 + swatches_h + 8
        draw.line([(px + pad, sep_y), (px + panel_w - pad, sep_y)],
                  fill=(190, 190, 190), width=1)

        draw.text(
            (px + pad, sep_y + 6),
            f"max std = {std_ref_max:.1f} mm",
            fill=(80, 80, 80), font=_CBAR_FONT,
        )
        return img, (px, py, panel_w, panel_h)

    def _draw_amplitude_bar(self, img: Image.Image, std_ref_max: float) -> Image.Image:
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

    # ── Axis gizmo ───────────────────────────────────────────────────────────

    def _draw_axis_gizmo(self, img: Image.Image, center=None) -> Image.Image:
        """Draw a small XYZ axis gizmo.

        When *center* is None the gizmo is placed in the bottom-left corner.
        Otherwise *center* is an (ox, oy) pixel coordinate for the gizmo origin
        (used to position it adjacent to the direction legend in direction mode).

        Axes are projected using the current camera orientation so the gizmo
        rotates in sync with the mesh.  Colours match the direction colorbar:
        x=red, y=yellow, z=green.
        """
        draw    = ImageDraw.Draw(img)
        iw, ih  = img.size
        arm_len = 50          # pixels
        if center is None:
            margin = 60       # distance of gizmo centre from corner
            ox, oy = margin, ih - margin   # gizmo centre (bottom-left)
        else:
            ox, oy = int(center[0]), int(center[1])

        # Camera rotation matrix (columns: right, up, -forward in world space).
        # Transposing gives the view rotation (world → camera).
        pose   = self.orbit.get_pose()
        view_R = pose[:3, :3].T

        axes = [
            (np.array([1.0, 0.0, 0.0]), _DIR_COLORS[0], "x"),
            (np.array([0.0, 1.0, 0.0]), _DIR_COLORS[1], "y"),
            (np.array([0.0, 0.0, 1.0]), _DIR_COLORS[2], "z"),
        ]

        # Draw back-to-front (sort by camera-space depth, largest first)
        axes_sorted = sorted(
            axes, key=lambda a: -(view_R @ a[0])[2]
        )

        for world_dir, color, label in axes_sorted:
            cam_dir = view_R @ world_dir
            # Project onto screen: cam x→right, cam y→up (flip to image down)
            tx = int(ox + arm_len * cam_dir[0])
            ty = int(oy - arm_len * cam_dir[1])
            rgb_u8 = tuple((color * 255).astype(np.uint8))
            draw.line([(ox, oy), (tx, ty)], fill=rgb_u8, width=3)
            # Small filled circle at the tip
            r = 4
            draw.ellipse([tx - r, ty - r, tx + r, ty + r], fill=rgb_u8)
            draw.text((tx + 5, ty - 7), label, fill=rgb_u8, font=_CBAR_FONT)

        return img

    # ── Render loop ──────────────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _render_loop(self) -> None:
        if self._dirty:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                self.scene.set_pose(self.cam_node, self.orbit.get_pose())
                color, _ = self.renderer.render(self.scene)

                img   = Image.fromarray(color)
                scale = min(cw / self.VIEWER_W, ch / self.VIEWER_H)
                if abs(scale - 1.0) > 0.005:
                    img = img.resize(
                        (max(1, int(self.VIEWER_W * scale)),
                         max(1, int(self.VIEWER_H * scale))),
                        _RESAMPLE,
                    )

                # Colorbar / legend overlay (drawn on the final, scaled image)
                gizmo_center = None
                if self._var_range is not None:
                    img, panel_rect = self._draw_colorbar(img, self._var_range)
                    if panel_rect is not None:
                        # Place gizmo centred below the direction legend panel.
                        px, py, panel_w, panel_h = panel_rect
                        gizmo_center = (px + panel_w // 2, py + panel_h + 65)

                # Axis gizmo (direction mode only)
                if self._direction_mode:
                    img = self._draw_axis_gizmo(img, center=gizmo_center)

                self._photo_image = ImageTk.PhotoImage(image=img)
                self.canvas.delete("all")
                self.canvas.create_image(
                    cw // 2, ch // 2, image=self._photo_image, anchor="center"
                )
                self._angle_label.config(
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
        dx, dy = event.x - self._drag_start[0], event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        if self._drag_button == 1:
            self.orbit.drag_rotate(dx, dy)
        elif self._drag_button == 3:
            self.orbit.drag_pan(dx, dy)
        self._dirty = True

    def _on_scroll(self, event: tk.Event) -> None:
        self.orbit.zoom(event.delta / 120)
        self._dirty = True

    def _on_middle_click(self, _event: tk.Event) -> None:
        self.orbit.reset_target()
        self._dirty = True

    # ── Mesh update ──────────────────────────────────────────────────────────

    def _schedule_mesh_update(self) -> None:
        if self._scheduled_update is not None:
            self.root.after_cancel(self._scheduled_update)
        self._scheduled_update = self.root.after(15, self._update_mesh)

    def _update_mesh(self) -> None:
        self._scheduled_update = None
        vertices, joints = self._compute_vertices()
        tri_mesh = self._build_trimesh(vertices, self.center)

        # Always render with vertex colours in the active uncertainty mode.
        if self._direction_mode:
            vertex_colors, std_ref_max = self._compute_variance_colors()
            self._var_range = std_ref_max if self._variance_active() else None
        else:
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
        self._update_joint_spheres(joints)
        self._dirty = True

    # ── Controls ─────────────────────────────────────────────────────────────

    def _build_scrollable_frame(self) -> ttk.Frame:
        container    = ttk.Frame(self._ctrl_frame, padding=0)
        container.pack(fill=tk.BOTH, expand=True)
        inner_canvas = tk.Canvas(container, highlightthickness=0, bg="white", borderwidth=0)
        scrollbar    = ttk.Scrollbar(container, orient=tk.VERTICAL, command=inner_canvas.yview)
        scrollable   = ttk.Frame(inner_canvas, style="TFrame")
        scrollable.bind(
            "<Configure>",
            lambda _: inner_canvas.configure(scrollregion=inner_canvas.bbox("all")),
        )
        inner_canvas.create_window(
            (0, 0), window=scrollable, anchor="nw", width=self.CTRL_WIDTH - 20
        )
        inner_canvas.configure(yscrollcommand=scrollbar.set)
        inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return scrollable

    def _make_section_header(
        self, parent, left_text: str, right_text: str = ""
    ) -> None:
        """Full-width header row; optional right label marks the σ² column."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=12, pady=(10, 2))
        ttk.Label(row, text=left_text,  style="Section.TLabel").pack(side=tk.LEFT)
        if right_text:
            ttk.Label(row, text=right_text, style="Variance.TLabel").pack(side=tk.RIGHT)

    def _make_slider(self, parent, label_text, from_, to_, callback) -> ttk.Scale:
        """Single-column slider (used for jaw / neck parameters)."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=12, pady=2)

        lf = ttk.Frame(frame)
        lf.pack(side=tk.LEFT)
        ttk.Label(lf, text=label_text, width=10, anchor="w",
                  style="Control.TLabel").pack(side=tk.LEFT)
        val_lbl = ttk.Label(lf, text="0.00", width=5, anchor="e",
                             style="Control.TLabel")
        val_lbl.pack(side=tk.LEFT, padx=(2, 8))

        slider = ttk.Scale(
            frame, from_=from_, to=to_, orient=tk.HORIZONTAL,
            command=lambda v, l=val_lbl, c=callback: self._slider_changed(v, l, c),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        slider.set(0.0)
        return slider

    def _make_slider_pair(
        self,
        parent,
        param_text: str,
        param_from: float,
        param_to: float,
        param_cb,
        var_cb,
        var_to: float | None = None,
    ) -> tuple[ttk.Scale, ttk.Scale]:
        """One full-width row: left = parameter slider, right = σ² slider.

        Pack order: variance side + divider from the RIGHT first so that the
        parameter side can fill all remaining space with expand=True.
        """
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)

        # ── σ² side (fixed width, packed from right) ──────────────────────
        var_frame = ttk.Frame(row, width=128)
        var_frame.pack(side=tk.RIGHT, fill=tk.Y)
        var_frame.pack_propagate(False)

        v_val = ttk.Label(var_frame, text="0.00", width=5, anchor="e",
                          style="Control.TLabel")
        v_val.pack(side=tk.LEFT, padx=(0, 4))
        v_slider = ttk.Scale(
            var_frame, from_=0.0, to=(var_to if var_to is not None else self.VAR_SLIDER_MAX), orient=tk.HORIZONTAL,
            command=lambda v, l=v_val, c=var_cb: self._slider_changed(v, l, c),
        )
        v_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        v_slider.set(0.0)

        # ── divider ───────────────────────────────────────────────────────
        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 4))

        # ── parameter side (fills remaining space) ────────────────────────
        par_frame = ttk.Frame(row)
        par_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        lf = ttk.Frame(par_frame)
        lf.pack(side=tk.LEFT)
        ttk.Label(lf, text=param_text, width=9, anchor="w",
                  style="Control.TLabel").pack(side=tk.LEFT)
        p_val = ttk.Label(lf, text="0.00", width=5, anchor="e",
                          style="Control.TLabel")
        p_val.pack(side=tk.LEFT, padx=(2, 6))

        p_slider = ttk.Scale(
            par_frame, from_=param_from, to=param_to, orient=tk.HORIZONTAL,
            command=lambda v, l=p_val, c=param_cb: self._slider_changed(v, l, c),
        )
        p_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        p_slider.set(0.0)

        return p_slider, v_slider

    def _slider_changed(self, value: str, label: ttk.Label, callback) -> None:
        label.config(text=f"{float(value):.2f}")
        callback(value)

    def _build_controls(self) -> None:
        # ── Header ───────────────────────────────────────────────────────────
        header = ttk.Frame(self._ctrl_frame, padding=(15, 12, 15, 8))
        header.pack(fill=tk.X)
        ttk.Label(header, text="FLAME Parameter Controls",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"Device: {self.device.type.upper()}",
                  style="Control.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Separator(self._ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(0, 4))

        content = self._build_scrollable_frame()

        self.shape_sliders:      list[ttk.Scale] = []
        self.shape_var_sliders:  list[ttk.Scale] = []
        self.expression_sliders: list[ttk.Scale] = []
        self.expr_var_sliders:   list[ttk.Scale] = []
        self.pose_sliders:       list[ttk.Scale] = []
        self.neck_sliders:       list[ttk.Scale] = []
        self.jaw_var_sliders:    list[ttk.Scale] = []
        self.neck_var_sliders:   list[ttk.Scale] = []

        # ── Shape (paired) ───────────────────────────────────────────────────
        self._make_section_header(
            content,
            f"Shape Parameters (first {self.SHAPE_SLIDERS})",
            "σ²",
        )
        for idx in range(min(self.SHAPE_SLIDERS, self.config.shape_params)):
            p_s, v_s = self._make_slider_pair(
                content,
                f"shape[{idx}]", -3.0, 3.0,
                lambda v, i=idx: self._on_shape_change(i, v),
                lambda v, i=idx: self._on_shape_var_change(i, v),
            )
            self.shape_sliders.append(p_s)
            self.shape_var_sliders.append(v_s)

        # ── Expression (paired) ──────────────────────────────────────────────
        self._make_section_header(
            content,
            f"Expression Parameters (first {self.EXPRESSION_SLIDERS})",
            "σ²",
        )
        for idx in range(min(self.EXPRESSION_SLIDERS, self.config.expression_params)):
            p_s, v_s = self._make_slider_pair(
                content,
                f"expr[{idx}]", -3.0, 3.0,
                lambda v, i=idx: self._on_expression_change(i, v),
                lambda v, i=idx: self._on_expr_var_change(i, v),
            )
            self.expression_sliders.append(p_s)
            self.expr_var_sliders.append(v_s)

        # ── Jaw pose ─────────────────────────────────────────────────────────
        self._make_section_header(content, "Jaw Pose", "σ²")
        # Jaw rotation: paired sliders (pose_params indices 3,4,5 → jaw_var indices 0,1,2)
        for j, lbl in enumerate(["jaw_x", "jaw_y", "jaw_z"]):
            p_s, v_s = self._make_slider_pair(
                content, lbl, -0.25, 0.25,
                lambda v, i=j + 3: self._on_pose_change(i, v),
                lambda v, i=j:     self._on_jaw_var_change(i, v),
                var_to=self.POSE_VAR_SLIDER_MAX,
            )
            self.pose_sliders.append(p_s)
            self.jaw_var_sliders.append(v_s)

        # ── Neck pose ────────────────────────────────────────────────────────
        self._make_section_header(content, "Neck Pose", "σ²")
        for idx in range(3):
            p_s, v_s = self._make_slider_pair(
                content, f"neck[{idx}]", -0.25, 0.25,
                lambda v, i=idx: self._on_neck_change(i, v),
                lambda v, i=idx: self._on_neck_var_change(i, v),
                var_to=self.POSE_VAR_SLIDER_MAX,
            )
            self.neck_sliders.append(p_s)
            self.neck_var_sliders.append(v_s)

        ttk.Frame(content, height=10).pack()
        ttk.Separator(self._ctrl_frame, orient="horizontal").pack(fill=tk.X, pady=(5, 0))

        footer = ttk.Frame(self._ctrl_frame, padding=(15, 10, 15, 12))
        footer.pack(fill=tk.X)
        ttk.Button(
            footer, text="Reset All Parameters", command=self._reset_all,
            style="Reset.TButton",
        ).pack(side=tk.RIGHT)
        ttk.Checkbutton(
            footer, text="Show Joints", variable=self._show_joints,
            command=self._on_joint_toggle,
        ).pack(side=tk.LEFT)

    # ── Slider callbacks ──────────────────────────────────────────────────────

    def _on_shape_change(self, index: int, value: str) -> None:
        self.shape_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expression_change(self, index: int, value: str) -> None:
        self.expression_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_pose_change(self, index: int, value: str) -> None:
        self.pose_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_neck_change(self, index: int, value: str) -> None:
        self.neck_pose[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_jaw_var_change(self, index: int, value: str) -> None:
        self.jaw_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_neck_var_change(self, index: int, value: str) -> None:
        self.neck_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_shape_var_change(self, index: int, value: str) -> None:
        self.shape_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _on_expr_var_change(self, index: int, value: str) -> None:
        self.expr_var_params[0, index] = float(value)
        self._schedule_mesh_update()

    def _reset_all(self) -> None:
        for s in (
            self.shape_sliders + self.shape_var_sliders
            + self.expression_sliders + self.expr_var_sliders
            + self.pose_sliders + self.neck_sliders
            + self.jaw_var_sliders + self.neck_var_sliders
        ):
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse viewer-specific CLI flags and preserve FLAME args."""
    parser = argparse.ArgumentParser(
        description="Interactive FLAME viewer with CLUE uncertainty overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--direction",
        action="store_true",
        default=False,
        help=(
            "Colour-code uncertainty by dominant variance axis "
            "(x=red, y=yellow, z=green). "
            "Without this flag, only the uncertainty amplitude is shown: "
            "base mesh colour for low uncertainty, red for high."
        ),
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    viewer_args, flame_args = parse_args(argv)
    with _temporary_cli_args(flame_args):
        app = FlameInteractiveApp(direction=viewer_args.direction)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
