"""Evaluate SMPL-X variance propagation methods against shared baselines.

The script compares CLUE, Monte-Carlo sampling, and a Jacobian baseline.
It renders qualitative error maps and a ground-truth variance figure
(on the neutral body, or with ``--set_parameters`` on seeded random bodies),
and optionally measures runtime, peak memory and FLOPs per method,
saved as JSON/CSV tables under ``results/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Any, Callable, Sequence, TypedDict

import numpy as np
import torch

import matplotlib

# matplotlib's non-interactive backend must be set before importing pyplot.
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})
import matplotlib.pyplot as plt
import cmasher  # registers cmr.* colormaps with matplotlib

from PIL import Image, ImageDraw, ImageFont

def _get_pil_font(size: int = 14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

if not hasattr(np, "infty"):
    np.infty = np.inf

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless / no-display rendering
import pyrender
import trimesh
import smplx
from smplx.joint_names import JOINT_NAMES
from smplx.lbs import lbs as smplx_lbs

# Make algorithm importable when running from the script directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import CLUE


# ─── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = "results"
ERROR_MAPS_DIR = os.path.join(RESULTS_DIR, "error_maps")
TIME_DIR = os.path.join(RESULTS_DIR, "time")
MEMORY_DIR = os.path.join(RESULTS_DIR, "memory")
FLOPS_DIR = os.path.join(RESULTS_DIR, "flops")
GT_CACHE_DIR = os.path.join(ERROR_MAPS_DIR, "ground_truth_variance")
_DEFAULT_MODEL_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
)


# ─── Experiment constants ─────────────────────────────────────────────────────
SAMPLE_COUNTS   = [5, 10, 100, 1000]
GT_SAMPLE_COUNT = 10_000
N_TIMING_WARMUP = 3
N_TIMING_TRIALS = 10
SHAPE_VAR_MAX   = 0.5
POSE_VAR_MAX    = 0.1
RENDER_W        = 400
RENDER_H        = 400
LABEL_H         = 20
PANEL_CROP      = 48
PANEL_CROP_V    = 8


class ParamGroup(TypedDict):
    beta_idx: int | None
    pose_fp_indices: list[int]   # each index carries its own, independent variance
    var_max: float               # σ² when the group is uncertain
    value_max: float             # |value| when a randomised body sets the group


class EvaluationSetting(TypedDict):
    name: str
    display: str
    values: dict[str, float]     # operating point: value per group (on each of its indices); {} = neutral body
    variances: dict[str, float]  # σ² on each index of each uncertain group


# ─── Runtime state initialised by main() ──────────────────────────────────────
device = torch.device("cpu")
model: Any = None
clue: CLUE | None = None
_renderer: pyrender.OffscreenRenderer | None = None

N_SHAPE_BETAS = 0
N_EXPR_BETAS = 0
N_BETAS_TOTAL = 0
N_BODY_JOINTS = 0
N_HAND_JOINTS = 0
N_POSE_TOTAL = 0

_FP_GLOBAL_START = 0
_FP_BODY_START = 0
_FP_JAW_START = 0
_FP_LEYE_START = 0
_FP_REYE_START = 0
_FP_LHAND_START = 0
_FP_RHAND_START = 0

_SPINE1_FP_IDX_Z = 0
_RHIP_FP_IDX_Z = 0
_RSHOULDER_FP_IDX_Z = 0

SHAPEDIRS_FULL = torch.empty(0)
BASE_BETAS = torch.empty(0)
BASE_EXPR = torch.empty(0)
BASE_GLOBAL = torch.empty(0)
BASE_BODY = torch.empty(0)
BASE_JAW = torch.empty(0)
BASE_LEYE = torch.empty(0)
BASE_REYE = torch.empty(0)
BASE_LHAND = torch.empty(0)
BASE_RHAND = torch.empty(0)
BASE_MODEL_KWARGS: dict[str, torch.Tensor] = {}
FACES = np.empty((0, 3), dtype=np.int32)
PARAM_GROUPS: dict[str, ParamGroup] = {}
SETTINGS: list[EvaluationSetting] = []


def _ensure_output_dirs() -> None:
    os.makedirs(ERROR_MAPS_DIR, exist_ok=True)
    os.makedirs(TIME_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(FLOPS_DIR, exist_ok=True)
    os.makedirs(GT_CACHE_DIR, exist_ok=True)


def _body_axis_groups() -> list[str]:
    """One group per body-joint rotation axis ("<joint>_<x|y|z>", flat-pose
    order) except spine1 z, which is the "spine" group."""
    return [f"{joint}_{axis}" for joint in JOINT_NAMES[1:1 + N_BODY_JOINTS] for axis in "xyz"
            if (joint, axis) != ("spine1", "z")]


def _build_param_groups() -> dict[str, ParamGroup]:
    body_joints = JOINT_NAMES[1:1 + N_BODY_JOINTS]
    assert (body_joints[1], body_joints[2], body_joints[16]) == ("right_hip", "spine1", "right_shoulder")
    groups: dict[str, ParamGroup] = {
        "shape":        {"beta_idx": 0,    "pose_fp_indices": [],
                         "var_max": SHAPE_VAR_MAX, "value_max": 0.5},
        "shape_2nd":    {"beta_idx": 1,    "pose_fp_indices": [],
                         "var_max": SHAPE_VAR_MAX, "value_max": 0.5},
        "spine":        {"beta_idx": None, "pose_fp_indices": [_SPINE1_FP_IDX_Z],
                         "var_max": POSE_VAR_MAX,  "value_max": 0.3},
        "hip_shoulder": {"beta_idx": None, "pose_fp_indices": [_RHIP_FP_IDX_Z, _RSHOULDER_FP_IDX_Z],
                         "var_max": POSE_VAR_MAX,  "value_max": 0.3},
    }
    # --set_parameters: every body-joint axis is its own parameter.
    for j, joint in enumerate(body_joints):
        for a, axis in enumerate("xyz"):
            if (joint, axis) != ("spine1", "z"):
                groups[f"{joint}_{axis}"] = {"beta_idx": None, "pose_fp_indices": [_FP_BODY_START + 3 * j + a],
                                             "var_max": POSE_VAR_MAX, "value_max": 0.3}
    return groups


def _build_settings() -> list[EvaluationSetting]:
    return [
        {"name": "shape_0",          "display": "shape, 1$^{\\mathrm{st}}$ principal component",
         "values": {}, "variances": {"shape": SHAPE_VAR_MAX}},
        {"name": "shape_1",          "display": "shape, 2$^{\\mathrm{nd}}$ component",
         "values": {}, "variances": {"shape_2nd": SHAPE_VAR_MAX}},
        {"name": "spine1_z",         "display": "spine",
         "values": {}, "variances": {"spine": POSE_VAR_MAX}},
        {"name": "rhip_rshoulder_z", "display": "hip and shoulder",
         "values": {}, "variances": {"hip_shoulder": POSE_VAR_MAX}},
    ]


# Spine sign (× value_max) of each randomised pose, one pose per entry. Fixed
# rather than drawn, as the neck in the FLAME study: the spine variance
# dominates every row where the spine is uncertain, so it is limited to the
# first and last rows. Everything else is drawn per pose.
RANDOM_BODY_SPINE_SIGNS = (-1, 0, 0, +1)


def random_pose_settings(seed: int) -> list[EvaluationSetting]:
    """Randomised poses for ``--set_parameters`` (deterministic given *seed*).

    spine1 z follows RANDOM_BODY_SPINE_SIGNS; shape[0] and every other
    body-joint rotation axis are drawn independently and uniformly from {0,
    +value_max, -value_max}. Each parameter that is set (non-zero) is also
    uncertain, with its var_max. Draws repeating an earlier pose are redrawn.
    """
    rng = np.random.default_rng(seed)
    drawn = ["shape", *_body_axis_groups()]
    groups = ["shape", "spine", *_body_axis_groups()]
    bodies: list[EvaluationSetting] = []
    for spine_sign in RANDOM_BODY_SPINE_SIGNS:
        while True:
            signs = dict(zip(drawn, rng.integers(-1, 2, size=len(drawn))))   # -1, 0 or +1
            signs["spine"] = spine_sign
            values = {g: float(signs[g]) * PARAM_GROUPS[g]["value_max"] for g in groups}
            if any(signs.values()) and all(values != body["values"] for body in bodies):
                break
        variances = {g: PARAM_GROUPS[g]["var_max"] for g in groups if signs[g]}
        n_axes = sum(1 for g in drawn[1:] if signs[g])
        shape = f"shape {values['shape']:+.1f}" if values["shape"] else "shape 0"
        spine = f"spine {values['spine']:+.1f}" if values["spine"] else "spine 0"
        bodies.append({
            "name": f"pose_{len(bodies)}",
            "display": f"{shape}, {spine} rad\n{n_axes} of {len(drawn) - 1} joint axes ±0.3 rad",
            "values": values,
            "variances": variances,
        })
    return bodies


def _require_model() -> Any:
    if model is None:
        raise RuntimeError("SMPL-X evaluation has not been initialised.")
    return model


def _require_clue() -> CLUE:
    if clue is None:
        raise RuntimeError("SMPL-X evaluation has not been initialised.")
    return clue


def _require_renderer() -> pyrender.OffscreenRenderer:
    if _renderer is None:
        raise RuntimeError("SMPL-X evaluation renderer has not been initialised.")
    return _renderer


def _base_model_kwargs() -> dict[str, torch.Tensor]:
    return dict(BASE_MODEL_KWARGS)


def _initialise_experiment(args: argparse.Namespace) -> None:
    global device, model, clue, _renderer
    global N_SHAPE_BETAS, N_EXPR_BETAS, N_BETAS_TOTAL, N_BODY_JOINTS, N_HAND_JOINTS, N_POSE_TOTAL
    global _FP_GLOBAL_START, _FP_BODY_START, _FP_JAW_START, _FP_LEYE_START, _FP_REYE_START
    global _FP_LHAND_START, _FP_RHAND_START, _SPINE1_FP_IDX_Z, _RHIP_FP_IDX_Z, _RSHOULDER_FP_IDX_Z
    global SHAPEDIRS_FULL, BASE_BETAS, BASE_EXPR, BASE_GLOBAL, BASE_BODY, BASE_JAW
    global BASE_LEYE, BASE_REYE, BASE_LHAND, BASE_RHAND, BASE_MODEL_KWARGS, FACES, PARAM_GROUPS, SETTINGS

    _ensure_output_dirs()

    if _renderer is not None:
        _renderer.delete()
        _renderer = None

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = smplx.create(
        os.path.expanduser(os.path.expandvars(args.model_folder)),
        model_type="smplx",
        gender="neutral",
        num_betas=args.num_betas,
        num_expression_coeffs=args.num_expression_coeffs,
        use_pca=False,
        ext="npz",
        batch_size=1,
    ).to(device)
    model.eval()
    clue = CLUE(model)

    N_SHAPE_BETAS = model.shapedirs.shape[2]
    N_EXPR_BETAS = model.expr_dirs.shape[2]
    N_BETAS_TOTAL = N_SHAPE_BETAS + N_EXPR_BETAS

    N_BODY_JOINTS = model.NUM_BODY_JOINTS
    N_HAND_JOINTS = model.NUM_HAND_JOINTS

    _FP_GLOBAL_START = 0
    _FP_BODY_START = 3
    _FP_JAW_START = _FP_BODY_START + N_BODY_JOINTS * 3
    _FP_LEYE_START = _FP_JAW_START + 3
    _FP_REYE_START = _FP_LEYE_START + 3
    _FP_LHAND_START = _FP_REYE_START + 3
    _FP_RHAND_START = _FP_LHAND_START + N_HAND_JOINTS * 3
    N_POSE_TOTAL = _FP_RHAND_START + N_HAND_JOINTS * 3

    _SPINE1_FP_IDX_Z = _FP_BODY_START + 2 * 3 + 2
    _RHIP_FP_IDX_Z = _FP_BODY_START + 1 * 3 + 2
    _RSHOULDER_FP_IDX_Z = _FP_BODY_START + 16 * 3 + 2

    SHAPEDIRS_FULL = torch.cat([model.shapedirs, model.expr_dirs], dim=-1)

    BASE_BETAS = torch.zeros(1, N_SHAPE_BETAS, dtype=torch.float32, device=device)
    BASE_EXPR = torch.zeros(1, N_EXPR_BETAS, dtype=torch.float32, device=device)
    BASE_GLOBAL = torch.zeros(1, 3, dtype=torch.float32, device=device)
    BASE_BODY = torch.zeros(1, N_BODY_JOINTS * 3, dtype=torch.float32, device=device)
    BASE_JAW = torch.zeros(1, 3, dtype=torch.float32, device=device)
    BASE_LEYE = torch.zeros(1, 3, dtype=torch.float32, device=device)
    BASE_REYE = torch.zeros(1, 3, dtype=torch.float32, device=device)
    BASE_LHAND = torch.zeros(1, N_HAND_JOINTS * 3, dtype=torch.float32, device=device)
    BASE_RHAND = torch.zeros(1, N_HAND_JOINTS * 3, dtype=torch.float32, device=device)

    BASE_MODEL_KWARGS = {
        "betas": BASE_BETAS,
        "expression": BASE_EXPR,
        "global_orient": BASE_GLOBAL,
        "body_pose": BASE_BODY,
        "jaw_pose": BASE_JAW,
        "leye_pose": BASE_LEYE,
        "reye_pose": BASE_REYE,
        "left_hand_pose": BASE_LHAND,
        "right_hand_pose": BASE_RHAND,
    }
    FACES = model.faces
    PARAM_GROUPS = _build_param_groups()
    SETTINGS = _build_settings()
    _renderer = pyrender.OffscreenRenderer(RENDER_W, RENDER_H)

# ─── Experiment constants ─────────────────────────────────────────────────────
SAMPLE_COUNTS   = [5, 10, 100, 1000]
GT_SAMPLE_COUNT = 10_000
N_TIMING_WARMUP = 3
N_TIMING_TRIALS = 10
SHAPE_VAR_MAX   = 0.5
POSE_VAR_MAX    = 0.1
RENDER_W        = 400
RENDER_H        = 400
# Reduce vertical label/header space for tighter layouts
LABEL_H         = 20
# slightly smaller panel crop so panels sit tighter together
PANEL_CROP      = 48
PANEL_CROP_V    = 8


# ─── Colormap helpers ─────────────────────────────────────────────────────────
def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.cm as _mpl_cm
        return _mpl_cm.get_cmap(name)


_BASE_COLOR     = np.array(_get_cmap("turbo")(0.0)[:3], dtype=np.float32)
_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)  # main.py peak red
_ORANGE_LOW     = np.array([1.0,  0.55, 0.0 ], dtype=np.float32)  # orange (low uncertainty)
_GT_COLOR_LOW   = np.array([0.55, 0.10, 0.75], dtype=np.float32)  # purple (low GT std)
_GT_COLOR_HIGH  = np.array([0.10, 0.65, 0.15], dtype=np.float32)  # green  (high GT std)


def _main_amplitude_cmap(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0,1] using main.py amplitude colors: blue base → red."""
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    rgb = (1.0 - tt[:, None]) * _BASE_COLOR[None, :] + tt[:, None] * _AMPLITUDE_HIGH[None, :]
    alpha = np.ones((tt.shape[0], 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _orange_red_cmap(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0,1]: orange (low) → red (high)."""
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    rgb = (1.0 - tt[:, None]) * _ORANGE_LOW[None, :] + tt[:, None] * _AMPLITUDE_HIGH[None, :]
    alpha = np.ones((tt.shape[0], 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _base_to_peak_cmap(t: np.ndarray, peak_color: np.ndarray) -> np.ndarray:
    """Blend _BASE_COLOR (t=0) → peak_color (t=1), mirroring main.py _amplitude_colors."""
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)[:, None]
    rgb = (1.0 - tt) * _BASE_COLOR[None, :] + tt * peak_color[None, :]
    alpha = np.ones((len(t), 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def _peak_alpha_colors(t: np.ndarray, peak_color: np.ndarray) -> np.ndarray:
    """Fixed peak_color RGB with alpha=t.

    Renders group peak colour at full saturation; opacity encodes the error
    magnitude.  Designed for use with alphaMode="BLEND" so that the peak hue
    is always visible — low-error regions simply appear more translucent.
    """
    rgb   = np.tile(peak_color.astype(np.float32), (len(t), 1))
    alpha = np.clip(t, 0.0, 1.0).astype(np.float32)[:, None]
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def apply_cmap(t: np.ndarray, cmap_name: str = "cmr.guppy") -> np.ndarray:
    """Map t ∈ [0,1] → RGBA uint8, shape (N, 4)."""
    if cmap_name == "main_amplitude":
        return _main_amplitude_cmap(t)
    if cmap_name == "orange_red":
        return _orange_red_cmap(t)
    cmap = _get_cmap(cmap_name)
    return (cmap(np.clip(t, 0.0, 1.0)) * 255).astype(np.uint8)


# ─── Pyrender helpers ──────────────────────────────────────────────────────────
_vc_material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    metallicFactor=0.5,
    roughnessFactor=0.52,
    alphaMode="OPAQUE",
    doubleSided=True,
)

_vc_material_blend = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    metallicFactor=0.5,
    roughnessFactor=0.52,
    alphaMode="BLEND",
    doubleSided=True,
)

_renderer = pyrender.OffscreenRenderer(RENDER_W, RENDER_H)


def _camera_pose(distance: float, elevation_deg: float = 15.0) -> np.ndarray:
    el  = np.deg2rad(elevation_deg)
    pos = np.array([0.0, np.sin(el) * distance, np.cos(el) * distance])
    fwd = -pos / (np.linalg.norm(pos) + 1e-8)
    up  = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, up);  right /= np.linalg.norm(right) + 1e-8
    up    = np.cross(right, fwd)
    pose  = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd
    pose[:3, 3] = pos
    return pose


def render_scalar_mesh(v_np: np.ndarray,
                       faces: np.ndarray,
                       values: np.ndarray,
                       vmax: float,
                       cmap_name: str = "cmr.guppy",
                       peak_color: np.ndarray | None = None,
                       use_alpha: bool = False) -> Image.Image:
    """Render SMPL-X mesh coloured by scalar per-vertex values.

    Parameters
    ----------
    v_np      : (V, 3) vertex positions
    faces     : (F, 3) triangle indices
    values    : (V,) scalar value per vertex
    vmax      : normalization ceiling
    cmap_name : matplotlib colormap name (ignored when peak_color is given)
    peak_color: if given, use group peak-colour mode
    use_alpha : if True (requires peak_color), render peak_color with alpha=t
                using BLEND material so the hue is always visible; otherwise
                blend _BASE_COLOR→peak_color opaquely.

    Returns
    -------
    PIL.Image (RENDER_W × RENDER_H, RGB)
    """
    t = values / vmax if vmax > 1e-12 else np.zeros_like(values)
    if peak_color is not None:
        if use_alpha:
            colors   = _peak_alpha_colors(t, peak_color)
            material = _vc_material_blend
        else:
            colors   = _base_to_peak_cmap(t, peak_color)
            material = _vc_material
    else:
        colors   = apply_cmap(t, cmap_name=cmap_name)
        material = _vc_material

    bbox_min = v_np.min(axis=0)
    bbox_max = v_np.max(axis=0)
    center  = (bbox_min + bbox_max) * 0.5          # bounding-box centre, not vertex mean
    v_c     = (v_np - center).astype(np.float32)
    extent  = float(np.max(bbox_max - bbox_min))
    dist    = 2.0 * extent                          # generous distance → full body fits in frame

    tm = trimesh.Trimesh(v_c, faces, process=False)
    tm.fix_normals()
    normals = tm.vertex_normals.astype(np.float32)

    prim = pyrender.Primitive(
        positions=v_c,
        normals=normals,
        indices=faces.astype(np.uint32),
        color_0=colors,
        material=material,
    )
    mesh_pr = pyrender.Mesh(primitives=[prim])

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.22, 0.24, 0.27])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=_camera_pose(dist))
    scene.add(mesh_pr)

    def _dir_light(direction, intensity):
        d  = np.asarray(direction, dtype=np.float64)
        d /= np.linalg.norm(d) + 1e-8
        z  = -d
        up = np.array([0., 1., 0.])
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1., 0., 0.])
        x = np.cross(up, z);  x /= np.linalg.norm(x) + 1e-8
        y = np.cross(z, x)
        lp = np.eye(4)
        lp[:3, 0], lp[:3, 1], lp[:3, 2] = x, y, z
        lp[:3, 3] = -d * 2.0 * extent
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=lp)

    _dir_light([-0.4, -0.55, -1.0], 7.0)
    _dir_light([ 0.8, -0.2,  -0.6], 3.5)
    _dir_light([ 0.0,  1.0,  -0.5], 1.5)

    color, _ = _require_renderer().render(scene)
    return Image.fromarray(color)


def render_variance_mesh(v_np: np.ndarray,
                         faces: np.ndarray,
                         var_xyz: np.ndarray,
                         vmax_std: float,
                         cmap_name: str = "cmr.guppy") -> Image.Image:
    """Render SMPL-X mesh coloured by per-vertex total std."""
    std = np.sqrt(np.maximum(var_xyz, 0.0).sum(axis=-1))
    return render_scalar_mesh(v_np, faces, std, vmax_std, cmap_name=cmap_name)


def _make_colorbar(height: int, vmax_value: float, cmap_name: str, unit_label: str) -> Image.Image:
    """Vertical colorbar image (width=70px)."""
    W, H   = 70, height
    img    = Image.new("RGB", (W, H), color=(30, 30, 30))
    draw   = ImageDraw.Draw(img)
    bar_x0, bar_x1 = 20, 50
    bar_y0, bar_y1 = 10, H - 25
    bar_h  = bar_y1 - bar_y0

    for row in range(bar_h):
        t    = 1.0 - row / max(bar_h - 1, 1)
        rgba = apply_cmap(np.array([t]), cmap_name=cmap_name)[0]
        draw.line([(bar_x0, bar_y0 + row), (bar_x1, bar_y0 + row)],
                  fill=(int(rgba[0]), int(rgba[1]), int(rgba[2])))

    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline=(200, 200, 200))
    font = _get_pil_font(9)

    draw.text((bar_x0 - 2, bar_y0 - 2), f"{vmax_value:.2f}", fill=(220, 220, 220), font=font)
    draw.text((bar_x0 - 2, bar_y1 + 2), f"0 {unit_label}",   fill=(220, 220, 220), font=font)
    draw.text((5, H // 2 - 4),          unit_label,           fill=(180, 180, 180), font=font)
    return img


# ─── Single SMPL-X forward pass ───────────────────────────────────────────────
@torch.no_grad()
def smplx_forward(**override) -> np.ndarray:
    """Return vertices (V, 3) numpy array for the given (or base) parameters."""
    active_model = _require_model()
    kwargs = _base_model_kwargs()
    kwargs.update(override)
    output = active_model(**kwargs, return_verts=True)
    return output.vertices[0].cpu().numpy()


# ─── Variance methods ─────────────────────────────────────────────────────────

_POSE_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
              "left_hand_pose", "right_hand_pose")   # flat full-pose order


def operating_kwargs(setting: EvaluationSetting) -> dict[str, torch.Tensor]:
    """SMPL-X model kwargs (fresh tensors) at *setting*'s operating point: each
    group's value is set on its coefficient and on each of its pose indices."""
    kwargs = {key: value.clone() for key, value in BASE_MODEL_KWARGS.items()}
    for group, value in setting["values"].items():
        p = PARAM_GROUPS[group]
        if p["beta_idx"] is not None:
            _apply_beta_noise(p["beta_idx"], value, kwargs["betas"], kwargs["expression"])
        for fp_idx in p["pose_fp_indices"]:
            _apply_fp_noise(fp_idx, value, *(kwargs[key] for key in _POSE_KEYS))
    return kwargs


def _variance_vectors(setting: EvaluationSetting) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal input variances: betas (N_BETAS_TOTAL,) and flat pose (N_POSE_TOTAL,)."""
    bv = torch.zeros(N_BETAS_TOTAL, dtype=torch.float32, device=device)
    pv = torch.zeros(N_POSE_TOTAL,  dtype=torch.float32, device=device)
    for group, sigma2 in setting["variances"].items():
        p = PARAM_GROUPS[group]
        if p["beta_idx"] is not None:
            bv[p["beta_idx"]] = sigma2
        for fp_idx in p["pose_fp_indices"]:
            pv[fp_idx] = sigma2
    return bv, pv


def variance_approx(setting: EvaluationSetting) -> np.ndarray:
    """Closed-form CLUE variance → (V, 3) numpy."""
    active_clue = _require_clue()
    bv, pv = _variance_vectors(setting)

    var, _ = active_clue.compute_variance(bv[None], pv[None], **operating_kwargs(setting))
    return var[0].cpu().numpy()   # (V, 3)


def variance_jacobian(setting: EvaluationSetting) -> np.ndarray:
    """Full-Jacobian variance via torch.autograd.functional.jacobian.

    Var[V_{v,k}] = Σ_i (∂V_{v,k}/∂beta_i)²  · σ²_i
                 + Σ_j (∂V_{v,k}/∂pose_j)²   · σ²_j

    betas layout : [shape_betas | expression_betas]  (N_BETAS_TOTAL,)
    pose layout  : [global(3) | body(63) | jaw(3) | leye(3) | reye(3) | lhand(45) | rhand(45)]
    """
    active_model = _require_model()
    bv, pv = _variance_vectors(setting)

    # Nominal operating point
    kwargs = operating_kwargs(setting)
    betas0 = torch.cat([kwargs["betas"][0], kwargs["expression"][0]])  # (N_BETAS_TOTAL,)

    raw_pose0 = torch.cat([kwargs[key][0] for key in _POSE_KEYS])  # (N_POSE_TOTAL,)
    # Add pose_mean so the operating point matches the mean-posed body,
    # consistent with CLUE which adds pose_mean before rodrigues.
    pose0 = raw_pose0 + active_model.pose_mean   # (N_POSE_TOTAL,)

    template = active_model.v_template   # (V, 3)

    def _model(betas: torch.Tensor, full_pose: torch.Tensor) -> torch.Tensor:
        verts, _ = smplx_lbs(
            betas.unsqueeze(0),
            full_pose.unsqueeze(0),
            template.unsqueeze(0),
            SHAPEDIRS_FULL,
            active_model.posedirs,
            active_model.J_regressor,
            active_model.parents,
            active_model.lbs_weights,
        )
        return verts[0]   # (V, 3)

    J_betas, J_pose = torch.autograd.functional.jacobian(
        _model, (betas0, pose0),
        strategy="forward-mode",
        vectorize=True,
    )   # (V, 3, N_BETAS_TOTAL), (V, 3, N_POSE_TOTAL)

    # Var[V_{v,k}] = Σ_i J²_{v,k,i} · σ²_i  (matrix-vector product over the
    # parameter axis). Stays in torch; the single NumPy conversion at the end
    # is for the rendering/plotting code and is a zero-copy view on CPU.
    with torch.no_grad():
        var = J_betas.detach().square() @ bv + J_pose.detach().square() @ pv   # (V, 3)
    return var.cpu().numpy()


def _apply_beta_noise(beta_idx: int, noise: float, betas_s, expr_s) -> None:
    """Perturb the coefficient at index beta_idx of [shape_betas | expression_betas]."""
    if beta_idx < N_SHAPE_BETAS:
        betas_s[0, beta_idx] += noise
    else:
        expr_s[0, beta_idx - N_SHAPE_BETAS] += noise


def _apply_fp_noise(fp_idx: int, noise: float,
                    global_s, body_s, jaw_s, leye_s, reye_s, lhand_s, rhand_s) -> None:
    """Perturb the pose parameter at flat full-pose index fp_idx."""
    if fp_idx < _FP_BODY_START:
        global_s[0, fp_idx - _FP_GLOBAL_START] += noise
    elif fp_idx < _FP_JAW_START:
        body_s[0, fp_idx - _FP_BODY_START] += noise
    elif fp_idx < _FP_LEYE_START:
        jaw_s[0, fp_idx - _FP_JAW_START] += noise
    elif fp_idx < _FP_REYE_START:
        leye_s[0, fp_idx - _FP_LEYE_START] += noise
    elif fp_idx < _FP_LHAND_START:
        reye_s[0, fp_idx - _FP_REYE_START] += noise
    elif fp_idx < _FP_RHAND_START:
        lhand_s[0, fp_idx - _FP_LHAND_START] += noise
    else:
        rhand_s[0, fp_idx - _FP_RHAND_START] += noise


def variance_sampling(setting: EvaluationSetting, n_samples: int) -> np.ndarray:
    """Sequential Monte-Carlo empirical variance → (V, 3) numpy.

    Every uncertain coefficient and pose index gets its own independent noise.
    The samples are kept as torch tensors and the (population) variance is
    taken in torch; the single NumPy conversion at the end is for the
    rendering/plotting code and is a zero-copy view on CPU.
    """
    active_model = _require_model()
    stds     = {group: float(np.sqrt(sigma2)) for group, sigma2 in setting["variances"].items()}
    template = active_model.v_template.unsqueeze(0)   # (1, V, 3)
    base     = operating_kwargs(setting)

    verts_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            betas_s  = base["betas"].clone()
            expr_s   = base["expression"].clone()
            global_s = base["global_orient"].clone()
            body_s   = base["body_pose"].clone()
            jaw_s    = base["jaw_pose"].clone()
            leye_s   = base["leye_pose"].clone()
            reye_s   = base["reye_pose"].clone()
            lhand_s  = base["left_hand_pose"].clone()
            rhand_s  = base["right_hand_pose"].clone()

            for group, std in stds.items():
                p = PARAM_GROUPS[group]
                if p["beta_idx"] is not None:
                    noise = torch.randn(1, device=device) * std
                    _apply_beta_noise(p["beta_idx"], noise.item(), betas_s, expr_s)
                for fp_idx in p["pose_fp_indices"]:
                    noise = torch.randn(1, device=device) * std
                    _apply_fp_noise(fp_idx, noise.item(),
                                    global_s, body_s, jaw_s, leye_s, reye_s, lhand_s, rhand_s)

            # Concatenate into LBS inputs
            betas_full = torch.cat([betas_s, expr_s], dim=1)   # (1, N_BETAS_TOTAL)
            full_pose  = torch.cat([global_s, body_s, jaw_s, leye_s, reye_s,
                                    lhand_s, rhand_s], dim=1)  # (1, N_POSE_TOTAL)
            full_pose  = full_pose + active_model.pose_mean.unsqueeze(0)

            verts, _ = smplx_lbs(
                betas_full, full_pose, template,
                SHAPEDIRS_FULL, active_model.posedirs,
                active_model.J_regressor, active_model.parents, active_model.lbs_weights,
            )
            verts_list.append(verts[0])   # (V, 3) torch

        # Population variance (unbiased=False), the same estimator as
        # np.var(axis=0) with its default ddof=0.
        var = torch.stack(verts_list, dim=0).var(dim=0, unbiased=False)   # (V, 3)
    return var.cpu().numpy()


def std_error_mm(gt_var_xyz: np.ndarray, pred_var_xyz: np.ndarray) -> np.ndarray:
    gt_std   = np.sqrt(np.maximum(gt_var_xyz,   0.0).sum(axis=-1))
    pred_std = np.sqrt(np.maximum(pred_var_xyz, 0.0).sum(axis=-1))
    return np.abs(pred_std - gt_std) * 1000.0


# ─── GT variance cache ─────────────────────────────────────────────────────────

def _gt_cache_path(setting: EvaluationSetting) -> str:
    if setting["values"]:   # randomised body: key on the operating point and the variances
        values = "_".join(f"{g}{v:+.2f}" for g, v in setting["values"].items())
        sigma2 = "_".join(f"{g}{s2:g}" for g, s2 in setting["variances"].items())
        stem = f"body_{values}_s2_{sigma2}".replace(".", "p")
        if len(stem) > 150:   # many parameters: hash the key (values/variances are stored in the file)
            stem = f"body_{hashlib.sha1(stem.encode()).hexdigest()[:16]}"
    else:                   # neutral body, one uncertain group
        (sigma2,) = setting["variances"].values()
        stem = f"{setting['name']}_s2_{sigma2:.8f}".replace(".", "p")
    return os.path.join(GT_CACHE_DIR, f"{stem}_N{GT_SAMPLE_COUNT}.npz")


def load_or_compute_gt_variance(
    setting: EvaluationSetting,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load cached GT variance if available; otherwise compute and cache it."""
    cache_path = _gt_cache_path(setting)
    if os.path.exists(cache_path) and not force_recompute:
        data = np.load(cache_path)
        print(f"    GT cache hit: {cache_path}")
        return data["var_xyz"]

    print(f"    GT cache miss -> computing sampling N={GT_SAMPLE_COUNT} ...")
    gt_var = variance_sampling(setting, GT_SAMPLE_COUNT)
    np.savez_compressed(
        cache_path,
        var_xyz=gt_var,
        setting_name=setting["name"],
        setting_display=setting["display"],
        values=str(setting["values"]),
        variances=str(setting["variances"]),
        n_samples=np.array([GT_SAMPLE_COUNT], dtype=np.int32),
    )
    print(f"    GT saved: {cache_path}")
    return gt_var


# ─── Qualitative experiment ────────────────────────────────────────────────────

def run_qualitative(setting: EvaluationSetting) -> None:
    print(f"  [qualitative] computing baseline variances + GT (N={GT_SAMPLE_COUNT}) ...")
    v_base = smplx_forward(**operating_kwargs(setting))

    gt_var     = load_or_compute_gt_variance(setting)
    var_approx = variance_approx(setting)
    print("    approx done")
    var_jac    = variance_jacobian(setting)
    print("    jacobian done")

    var_samples = []
    for n in SAMPLE_COUNTS:
        var_samples.append(variance_sampling(setting, n))
        print(f"    sampling N={n} done")

    methods   = [var_approx] + var_samples + [var_jac]
    meth_lbls = ["CLUE"] + [f"Sample N={n}" for n in SAMPLE_COUNTS] + ["Jacobian"]
    errors_mm = [std_error_mm(gt_var, m) for m in methods]
    vmax_err  = max(float(e.max()) for e in errors_mm)
    if vmax_err < 1e-12:
        vmax_err = 1e-12

    imgs = []
    for lbl, err_mm in zip(meth_lbls, errors_mm):
        img = render_scalar_mesh(v_base, FACES, err_mm, vmax_err, cmap_name="main_amplitude")
        imgs.append(img)
        print(f"    rendered '{lbl}'")

    n_panels = len(imgs)
    cbar_w   = 70
    panel_w  = RENDER_W - 2 * PANEL_CROP
    panel_h  = RENDER_H - 2 * PANEL_CROP_V
    total_w  = panel_w * n_panels + cbar_w
    total_h  = panel_h + LABEL_H

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    font = _get_pil_font(14)

    for i, (img, lbl) in enumerate(zip(imgs, meth_lbls)):
        cropped = img.crop((PANEL_CROP, PANEL_CROP_V, RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
        canvas.paste(cropped, (i * panel_w, LABEL_H))
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw   = bbox[2] - bbox[0]
        tx   = i * panel_w + (panel_w - tw) // 2
        draw.text((tx, 8), lbl, fill=(240, 240, 240), font=font)

    cbar = _make_colorbar(panel_h, vmax_err, cmap_name="main_amplitude", unit_label="mm")
    canvas.paste(cbar, (panel_w * n_panels, LABEL_H))

    title = (f"Variance error vs GT (N={GT_SAMPLE_COUNT}) - "
             f"{setting['display']}   (s2_in = {setting['variances']})")
    draw.text((8, 8), title, fill=(200, 200, 200), font=font)

    out = os.path.join(ERROR_MAPS_DIR, f"{setting['name']}.png")
    canvas.save(out)
    print(f"  Saved {out}")


# ─── Combined qualitative experiment ──────────────────────────────────────────

def run_qualitative_combined(
    settings: Sequence[EvaluationSetting] | None = None,
    out_name: str = "combined.png",
    subtitle: str = "Shared scale (global vmax)",
) -> None:
    """Render all settings × methods in one figure with a shared blue→red colorbar.

    Each row's mesh is posed at that setting's operating point. Defaults
    (SETTINGS) give results/error_maps/combined.png (neutral body, one
    uncertain group per row).

    All panels share a single normalisation (global vmax across all settings and
    methods) and a single colormap:

        vertex_rgb = (1 - t) * _BASE_COLOR  +  t * _AMPLITUDE_HIGH
        where t = |Δσ|_vertex / vmax_global

    The colorbar therefore spans metallic blue (zero error) → red (peak error),
    matching every rendered panel exactly.  No transparency is used.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    print(f"  [qualitative-combined] computing all variances + GT (N={GT_SAMPLE_COUNT}) ...")
    settings = SETTINGS if settings is None else settings
    meth_lbls  = [f"Sample N={n}" for n in SAMPLE_COUNTS] + ["Jacobian"] + [r"$\mathbf{CLUE}$"]
    n_settings = len(settings)
    n_methods  = len(meth_lbls)

    # ── Pass 1: compute per-vertex |Δσ| (mm) vs GT for each method ──────────
    all_errors: list[list[np.ndarray]] = []
    for setting in settings:
        gt_var  = load_or_compute_gt_variance(setting)
        methods = (
            [variance_sampling(setting, n) for n in SAMPLE_COUNTS]
            + [variance_jacobian(setting)]
            + [variance_approx(setting)]
        )
        all_errors.append([std_error_mm(gt_var, m) for m in methods])
        detail = (f"values {setting['values']}, σ² {setting['variances']}" if len(setting["values"]) <= 6
                  else f"{len(setting['variances'])} uncertain parameters")
        print(f"    {setting['name']} errors computed  ({detail})")
        for lbl, err_mm in zip(meth_lbls, all_errors[-1]):
            print(f"      {lbl:<20s} mean {err_mm.mean():.3f}  max {err_mm.max():.3f} mm")

    all_errors_flat = np.concatenate([e for row in all_errors for e in row])
    vmax_err_mm = float(np.percentile(all_errors_flat, 99.5))
    if vmax_err_mm < 1e-12:
        vmax_err_mm = 1e-12
    print(f"  Global vmax_err (95th pct) = {vmax_err_mm:.4f} mm  "
          f"[abs max = {float(all_errors_flat.max()):.4f} mm]")

    # ── Pass 2: render all panels ──────────────────────────────────────────────
    # All panels use the same global vmax and the cmr.guppy cmap.
    # No transparency; alphaMode is OPAQUE throughout.
    all_imgs: list[list[np.ndarray]] = []
    for setting, errors in zip(settings, all_errors):
        v_base = smplx_forward(**operating_kwargs(setting))
        row_imgs = []
        for err_mm in errors:
            img = render_scalar_mesh(v_base, FACES, err_mm, vmax_err_mm,
                                     cmap_name="cmr.guppy")
            cropped = img.crop((PANEL_CROP, PANEL_CROP_V,
                                RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
            row_imgs.append(np.asarray(cropped))
        all_imgs.append(row_imgs)
        print(f"    {setting['name']} rendered")

    # ── Pass 3: assemble matplotlib figure ────────────────────────────────────
    turbo_mpl = _get_cmap("cmr.guppy")

    panel_w = RENDER_W - 2 * PANEL_CROP    # 240 px
    panel_h = RENDER_H - 2 * PANEL_CROP_V  # 300 px
    dpi      = 100
    cell_w   = panel_w / dpi
    cell_h   = panel_h / dpi
    fig_w    = cell_w * n_methods + 1.6    # +1.6 in for colorbar + row labels
    fig_h    = cell_h * n_settings + 0.6  # +0.6 in for column titles + suptitle

    fig, axes = plt.subplots(
        n_settings, n_methods,
        figsize=(fig_w, fig_h),
        dpi=dpi,
    )

    for s_idx in range(n_settings):
        for m_idx in range(n_methods):
            ax = axes[s_idx, m_idx]
            ax.imshow(all_imgs[s_idx][m_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if s_idx == 0:
                ax.set_title(meth_lbls[m_idx], fontsize=8, pad=3)
            if m_idx == 0:
                ax.set_ylabel(settings[s_idx]["display"], fontsize=9, labelpad=5)

    # Colorbar: guppy, matching the rendered vertex colours exactly.
    norm = Normalize(vmin=0.0, vmax=vmax_err_mm)
    sm   = ScalarMappable(cmap=turbo_mpl, norm=norm)
    sm.set_array([])

    # Reserve right margin for the colorbar, then position it explicitly.
    # tighten margins so title/subtitle sit closer to the panels
    fig.subplots_adjust(right=0.92, top=0.93, bottom=0.06, hspace=-0.15, wspace=0.02)
    cbar_ax = fig.add_axes([0.935, 0.10, 0.015, 0.68])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.0, vmax_err_mm])
    cbar.set_ticklabels(["0", f"≥{vmax_err_mm:.2f}"], fontsize=11)
    cbar.set_label("|Δσ| (mm)", fontsize=11)

    fig.suptitle("Variance Error  Δσ  vs Ground Truth", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.945,
        f"{subtitle}  —  GT: N={GT_SAMPLE_COUNT} samples",
        ha="center", va="top", fontsize=12, color="#334155",
    )

    out = os.path.join(ERROR_MAPS_DIR, out_name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─── Timing experiment ────────────────────────────────────────────────────────

def _time_fn(fn: Callable[[], object], warmup: int, trials: int) -> float:
    """Return median wall-clock time (seconds) of fn() over *trials* runs,
    preceded by *warmup* un-timed calls.  CUDA synchronisation is applied
    before every measurement when a GPU is available."""
    import time

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        fn()
        _sync()

    times = []
    for _ in range(trials):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)

    return float(np.median(times))


def _method_names() -> list[str]:
    """Row order of the cost tables (timing / memory / FLOPs): sampling, Jacobian, CLUE."""
    return [f"sample_N{n}" for n in SAMPLE_COUNTS] + ["jacobian", "clue"]


def _machine_info() -> dict:
    """Where the numbers were measured (goes into every table's metadata)."""
    import platform
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "num_vertices": int(np.asarray(FACES).max()) + 1,
    }


def _method_fns(setting: EvaluationSetting) -> dict:
    """The variance methods for one setting, keyed by _method_names()."""
    fns = {f"sample_N{n}": (lambda nn=n: variance_sampling(setting, nn)) for n in SAMPLE_COUNTS}
    fns["jacobian"] = lambda: variance_jacobian(setting)
    fns["clue"] = lambda: variance_approx(setting)
    return fns


def _save_benchmark_table(out_dir: str, name: str, values: dict, columns: list[str], meta: dict) -> None:
    """Write ``values[setting][method] -> {column: number}`` as
    ``<out_dir>/<name>.json`` (nested, with metadata) and ``<out_dir>/<name>.csv``
    (long format: setting, method, columns...) for later tabulation."""
    import csv
    import datetime
    import json

    meta = {"measured_at": datetime.datetime.now().isoformat(timespec="seconds"), **meta, **_machine_info()}
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w") as fh:
        json.dump({"meta": meta, "settings": [st["name"] for st in SETTINGS],
                   "methods": _method_names(), "columns": columns, "values": values}, fh, indent=2)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["setting", "method", *columns])
        for st in SETTINGS:
            for method in _method_names():
                writer.writerow([st["name"], method, *[values[st["name"]][method][c] for c in columns]])
    print(f"  Saved {json_path} and {csv_path}")


def run_timing() -> None:
    """Time every variance method for every setting; save results/time/timing.{json,csv}."""
    print("  [timing] warming up + measuring ...")
    values = {}
    for setting in SETTINGS:
        row = {}
        for method, fn in _method_fns(setting).items():
            seconds = _time_fn(fn, N_TIMING_WARMUP, N_TIMING_TRIALS)
            row[method] = {"ms": seconds * 1e3}
            print(f"    {setting['name']:18s} {method:14s} {seconds * 1e3:9.2f} ms")
        values[setting["name"]] = row
    _save_benchmark_table(
        TIME_DIR, "timing", values, ["ms"],
        meta={"quantity": "wall-clock time of one variance evaluation",
              "statistic": f"median of {N_TIMING_TRIALS} trials after {N_TIMING_WARMUP} warm-up runs",
              "sigma2_per_setting": {st["name"]: s2 for st in SETTINGS for s2 in st["variances"].values()}},
    )


# ─── Memory experiment ────────────────────────────────────────────────────────

def _torch_cpu_peak_bytes(prof) -> int:
    """Peak live torch CPU allocation (bytes) inside a torch.profiler region
    run with profile_memory=True.

    The raw kineto ``[memory]`` events carry one signed byte delta per
    allocator call (allocation > 0, free < 0) with a timestamp; the peak of
    their running sum, in time order, is the allocator's high-water mark for
    the region. (``prof.events()`` cannot be used for this: it folds the
    allocations that happen inside an op into that op's ``cpu_memory_usage``
    and keeps only the stray ones as ``[memory]`` rows.)"""
    from torch.autograd import DeviceType

    deltas = []
    for k in prof.profiler.kineto_results.events():
        if k.name() != "[memory]":
            continue
        try:
            if k.device_type() != DeviceType.CPU:
                continue
        except Exception:
            pass
        start = k.start_ns() if hasattr(k, "start_ns") else int(k.start_us() * 1000)
        deltas.append((start, int(k.nbytes())))
    deltas.sort()
    running = peak = 0
    for _, nbytes in deltas:
        running += nbytes
        peak = max(peak, running)
    return peak


_PROFILER_WARMED_UP = False


def _measure_memory_breakdown(fn: Callable[[], object]) -> dict:
    """Return the peak *incremental* memory consumed by fn(), in MB, as
    ``{"total_mb", "torch_mb", "numpy_mb"}``.

    GPU path  : uses torch.cuda.max_memory_allocated() delta.
    CPU path  : torch CPU tensors come from the c10 allocator, which
                tracemalloc does NOT trace (tracemalloc only sees Python /
                NumPy buffers; a ``tensor.numpy()`` view costs 0 there). So the
                CPU path records torch allocations with the torch profiler
                (peak of the running sum of its ``[memory]`` events) and NumPy
                allocations with tracemalloc, and reports the sum of the two
                peaks (an upper bound on the true combined peak; exact when the
                torch tensors stay alive through the NumPy phase, as here).

    A garbage-collection cycle is triggered before each measurement so that
    stale tensors from previous calls do not inflate the baseline.
    """
    import gc
    import tracemalloc

    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        mb = max(torch.cuda.max_memory_allocated() - baseline, 0) / 1e6   # bytes -> MB
        return {"total_mb": mb, "torch_mb": mb, "numpy_mb": None}

    from torch.profiler import profile, ProfilerActivity
    global _PROFILER_WARMED_UP
    if not _PROFILER_WARMED_UP:
        # The profiler's first use allocates its own Python-side state,
        # which tracemalloc would otherwise bill to the first method.
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True):
            torch.zeros(1)
        _PROFILER_WARMED_UP = True
        gc.collect()
    tracemalloc.start()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    _, numpy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    torch_peak = _torch_cpu_peak_bytes(prof)
    return {"total_mb": (torch_peak + numpy_peak) / 1e6, "torch_mb": torch_peak / 1e6,
            "numpy_mb": numpy_peak / 1e6}   # bytes -> MB


def run_memory() -> None:
    """Measure peak incremental memory of every variance method for every
    setting; save results/memory/memory.{json,csv}."""
    print("  [memory] measuring peak incremental memory ...")
    values = {}
    for setting in SETTINGS:
        row = {}
        for method, fn in _method_fns(setting).items():
            row[method] = _measure_memory_breakdown(fn)
            mb = row[method]
            numpy_mb = mb["numpy_mb"] if mb["numpy_mb"] is not None else float("nan")
            print(f"    {setting['name']:18s} {method:14s} {mb['total_mb']:9.3f} MB"
                  f"  (torch {mb['torch_mb']:.3f}, numpy {numpy_mb:.3f})")
        values[setting["name"]] = row
    _save_benchmark_table(
        MEMORY_DIR, "memory", values, ["total_mb", "torch_mb", "numpy_mb"],
        meta={"quantity": "peak incremental memory of one variance evaluation",
              "method": ("CPU: torch allocator peak (running sum of the profiler's raw [memory] events) "
                         "plus NumPy peak (tracemalloc); total = sum of the two peaks (upper bound). "
                         "CUDA: torch.cuda.max_memory_allocated delta."),
              "sigma2_per_setting": {st["name"]: s2 for st in SETTINGS for s2 in st["variances"].values()}},
    )


# ─── FLOP count ───────────────────────────────────────────────────────────────

def _count_flops(fn: Callable[[], object]) -> int:
    """FLOPs executed by fn() as counted by torch.utils.flop_counter.

    The counter sees every aten op after autograd and functorch have run, so
    forward-mode/vmap Jacobians are counted at their batched sizes. It counts
    matmul-class ops (mm, bmm, addmm, baddbmm, einsum lowered to these,
    convolutions) at 2 FLOPs per multiply-add; elementwise ops are not
    counted, which is the standard convention.
    """
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def _count_flops_all(fn) -> int:
    """All-arithmetic FLOPs of fn(): matmul-class ops plus elementwise
    arithmetic, see _all_arithmetic_flop_counter. Sparse-dense matmuls are not
    seen by the counter and are added by the caller where they occur."""
    with _all_arithmetic_flop_counter() as counter:
        fn()
    return int(counter.get_total_flops())


def _all_arithmetic_flop_counter():
    """A FlopCounterMode that also counts elementwise arithmetic.

    The stock counter registers only matmul-class ops (mm, bmm, addmm,
    baddbmm, convolutions, attention), which is fine for rigs whose forward is
    matmul-based but under-counts a rig such as MHR whose skinning is written
    as gathers, cross products and index-adds. For a like-for-like comparison
    every method on every rig is counted with the same extended rule set:
    matmul-class ops at 2 FLOPs per multiply-add (stock formulas), plus
      1 FLOP per output element   for add/sub/mul/div/neg/square/sqrt/rsqrt/
                                  reciprocal/exp/log/sin/cos/atan2/pow/sigmoid/tanh (+ in-place forms),
      2 per output element        for addcmul/addcdiv,  3 for lerp,
      3 per output element        for linalg_cross (9 per 3-vector),
      1 per input element         for sum/mean/cumsum,  2 for norm (square + add),
      3 per input element         for var/std,
      1 per source element        for index_add/scatter_add (the accumulation),
      2 per stored nonzero        for sparse-dense matmul (per dense column).
    Data movement (view, permute, gather, index_select, cat, clone, expand)
    and comparisons (clamp, where, abs, max, min, sign) count 0.
    """
    from torch.utils.flop_counter import FlopCounterMode
    aten = torch.ops.aten

    def numel(shape) -> int:
        n = 1
        for d in shape:
            n *= int(d)
        return n

    def out_elems(k):
        def f(*args, out_shape=None, **kwargs):
            return k * numel(out_shape)
        return f

    def in_elems(k):
        def f(*args, out_shape=None, **kwargs):
            return k * numel(args[0])
        return f

    def index_add(*args, out_shape=None, **kwargs):
        return numel(args[3])            # (self, dim, index, source)

    mapping = {}
    for name in ("add", "sub", "mul", "div", "neg", "square", "sqrt", "rsqrt", "reciprocal", "exp", "log",
                 "sin", "cos", "atan2", "pow", "sigmoid", "tanh", "rsub"):
        for variant in (name, name + "_"):
            op = getattr(aten, variant, None)
            if op is not None:
                mapping[op] = out_elems(1)
    for name, k in (("addcmul", 2), ("addcdiv", 2), ("lerp", 3), ("linalg_cross", 3)):
        for variant in (name, name + "_"):
            op = getattr(aten, variant, None)
            if op is not None:
                mapping[op] = out_elems(k)
    for name, k in (("sum", 1), ("mean", 1), ("cumsum", 1), ("norm", 2), ("linalg_vector_norm", 2), ("var", 3), ("std", 3)):
        op = getattr(aten, name, None)
        if op is not None:
            mapping[op] = in_elems(k)
    for name in ("index_add", "index_add_", "scatter_add", "scatter_add_"):
        op = getattr(aten, name, None)
        if op is not None:
            mapping[op] = index_add
    return FlopCounterMode(display=False, custom_mapping=mapping)


def _flop_counter_vmap_check() -> bool:
    """vmap over B copies of an (n,n)@(n,n) product must count B*2*n^3 FLOPs;
    if the counter only saw the un-batched op, the vectorised Jacobian below
    would be under-counted by its batch size."""
    B, n = 8, 16
    a, b = torch.randn(B, n, n, device=device), torch.randn(B, n, n, device=device)
    return _count_flops(lambda: torch.vmap(torch.mm)(a, b)) == B * 2 * n ** 3


def _uncounted_flops(method: str) -> int:
    """FLOPs of a method that the dispatcher-level counter cannot see: CLUE's
    sparse-dense matmuls (2 FLOPs per stored nonzero per dense column, batch 1)."""
    if method == "clue" and clue is not None:
        return int(clue.sparse_matmul_flops(batch=1))
    return 0


def run_flops() -> None:
    """Count FLOPs of every variance method for every setting; save
    results/flops/flops.{json,csv}."""
    print("  [flops] counting floating-point operations ...")
    vmap_ok = _flop_counter_vmap_check()
    forward_flops = _count_flops_all(lambda: smplx_forward())
    forward_flops_matmul = _count_flops(lambda: smplx_forward())
    print(f"    counter sees vmap-batched ops: {vmap_ok}; one plain forward: {forward_flops / 1e9:.4f} GFLOP "
          f"(matmul-class only {forward_flops_matmul / 1e9:.4f})")
    values = {}
    for setting in SETTINGS:
        row = {}
        for method, fn in _method_fns(setting).items():
            flops = _count_flops_all(fn) + _uncounted_flops(method)
            flops_matmul = _count_flops(fn)
            row[method] = {"flops": flops, "gflops": flops / 1e9, "gflops_matmul_only": flops_matmul / 1e9}
            print(f"    {setting['name']:18s} {method:14s} {flops / 1e9:12.4f} GFLOP  (matmul-class only {flops_matmul / 1e9:.4f})")
        values[setting["name"]] = row
    _save_benchmark_table(
        FLOPS_DIR, "flops", values, ["flops", "gflops", "gflops_matmul_only"],
        meta={"quantity": "floating-point operations of one variance evaluation",
              "counter": "torch.utils.flop_counter.FlopCounterMode with elementwise formulas (_all_arithmetic_flop_counter)",
              "convention": ("gflops = all arithmetic: matmul-class ops at 2 FLOPs per multiply-add plus elementwise "
                             "arithmetic at 1 FLOP per output element (cross products 3, norms 2, var/std 3 per input "
                             "element, index_add 1 per source element, sparse matmul 2 per nonzero per column); "
                             "data movement and comparisons count 0. gflops_matmul_only = the stock counter."),
              "vmap_batched_ops_counted": vmap_ok,
              "flops_one_forward": forward_flops,
              "flops_one_forward_matmul_only": forward_flops_matmul,
              "sigma2_per_setting": {st["name"]: s2 for st in SETTINGS for s2 in st["variances"].values()}},
    )


# ─── GT variance visualisation ────────────────────────────────────────────────

def run_qualitative_gt(
    settings: Sequence[EvaluationSetting] | None = None,
    out_name: str = "gt_variance.png",
) -> None:
    """Render per-vertex GT variance std (mm) for each setting.

    Saved to ``results/error_maps/gt_variance.png`` (default) so it sits
    alongside the other qualitative comparison outputs.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    print(f"  [qualitative-gt] rendering GT variance maps (N={GT_SAMPLE_COUNT}) ...")
    settings   = SETTINGS if settings is None else settings
    n_settings = len(settings)
    gt_stds    = []
    gt_vmaxes  = []
    for setting in settings:
        gt_var    = load_or_compute_gt_variance(setting)
        gt_std_mm = np.sqrt(np.maximum(gt_var, 0.0).sum(axis=-1)) * 1000.0
        gt_stds.append(gt_std_mm)
        gt_vmaxes.append(float(gt_std_mm.max()))

    all_gt_flat  = np.concatenate(gt_stds)
    global_vmax  = float(np.percentile(all_gt_flat, 80)) or 1e-12

    imgs = []
    for s_idx, (setting, gt_std_mm) in enumerate(zip(settings, gt_stds)):
        v_base  = smplx_forward(**operating_kwargs(setting))
        img     = render_scalar_mesh(v_base, FACES, gt_std_mm, global_vmax, cmap_name="turbo")
        cropped = img.crop((PANEL_CROP, PANEL_CROP_V,
                            RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
        imgs.append(np.asarray(cropped))
        print(f"    {setting['name']}  GT max σ = {gt_vmaxes[s_idx]:.3f} mm")

    panel_w = RENDER_W - 2 * PANEL_CROP
    panel_h = RENDER_H - 2 * PANEL_CROP_V
    dpi     = 100
    cell_w  = panel_w / dpi
    cell_h  = panel_h / dpi
    fig_w   = cell_w + 1.4
    fig_h   = cell_h * n_settings + 1.0

    fig, axes = plt.subplots(n_settings, 1, figsize=(fig_w, fig_h), dpi=dpi)
    if n_settings == 1:
        axes = [axes]

    for ax, img, setting in zip(axes, imgs, settings):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(setting["display"], fontsize=11, labelpad=5)

    turbo_mpl = _get_cmap("turbo")
    norm = Normalize(vmin=0.0, vmax=global_vmax)
    sm   = ScalarMappable(cmap=turbo_mpl, norm=norm)
    sm.set_array([])

    # tighter margins for the ground-truth panels
    fig.subplots_adjust(right=0.94, top=0.92, bottom=0.06, hspace=-0.15, wspace=0.02)
    cbar_ax = fig.add_axes([0.955, 0.09, 0.035, 0.78])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.0, global_vmax])
    cbar.set_ticklabels(["0", f"≥{global_vmax:.2f}"], fontsize=11)
    cbar.set_label("Max vertex σ (mm)", fontsize=11)

    fig.suptitle("Ground Truth Variance", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.945,
        f"GT sampling  N={GT_SAMPLE_COUNT}  —  per-vertex total σ (mm), normalised per setting",
        ha="center", va="top", fontsize=12, color="#334155",
    )

    out = os.path.join(ERROR_MAPS_DIR, out_name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the public evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="SMPL-X variance comparison experiment",
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
        type=int,
        default=10,
        help="Number of shape blendshape coefficients.",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        type=int,
        default=10,
        help="Number of expression blendshape coefficients.",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Run only the qualitative mesh-rendering outputs.",
    )
    parser.add_argument(
        "--set_parameters",
        action="store_true",
        help="Error maps on randomised poses instead of the neutral body: spine = -max, 0, "
             "0, +max by row; shape[0] and every other body-joint rotation axis each drawn "
             "from {0, +max, -max} (each its own parameter); every parameter that is set is "
             "uncertain with its maximum variance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the randomised poses of --set_parameters.",
    )
    return parser.parse_args(argv)


def _print_section(title: str) -> None:
    rule = "-" * 60
    print(f"\n{rule}")
    print(f"  {title}")
    print(rule)


def main(argv: Sequence[str] | None = None) -> int:
    global _renderer

    args = parse_args(argv)
    _initialise_experiment(args)
    active_model = _require_model()

    print("=" * 60)
    print("  SMPL-X Variance Comparison Experiment")
    print(f"  model   : {args.model_folder}  (neutral)")
    print(f"  betas   : {N_SHAPE_BETAS} shape + {N_EXPR_BETAS} expression = {N_BETAS_TOTAL}")
    print(f"  pose    : {N_POSE_TOTAL} dims  (body={N_BODY_JOINTS}j, hand={N_HAND_JOINTS}j/side)")
    print(f"  vertices: {active_model.get_num_verts()}")
    print("=" * 60)

    if args.set_parameters:
        settings = random_pose_settings(args.seed)
        suffix   = f"_random_seed{args.seed}"
        spine    = " / ".join(f"{s * PARAM_GROUPS['spine']['value_max']:g}" for s in RANDOM_BODY_SPINE_SIGNS)
        subtitle = (f"Random poses (seed {args.seed}): shape[0] ∈ {{0, ±0.5}}, every other body-joint axis ∈ "
                    f"{{0, ±0.3}} rad, spine = {spine} by row; every set parameter uncertain at its max σ²")
    else:
        settings, suffix, subtitle = SETTINGS, "", "Shared scale (global vmax)"

    try:
        _print_section("Qualitative (all settings, shared colorbar)")
        run_qualitative_combined(settings, f"combined{suffix}.png", subtitle)

        _print_section("Qualitative GT variance (turbo ramp)")
        run_qualitative_gt(settings, f"gt_variance{suffix}.png")

        if not args.qualitative:
            _print_section("Timing benchmark")
            run_timing()

            _print_section("Memory benchmark")
            run_memory()

            _print_section("FLOP count")
            run_flops()
    finally:
        if _renderer is not None:
            _renderer.delete()
            _renderer = None

    print("\nAll done.")
    print(f"  Error maps         -> {ERROR_MAPS_DIR}/")
    if not args.qualitative:
        print(f"  Timing table       -> {TIME_DIR}/timing.json|csv")
        print(f"  Memory table       -> {MEMORY_DIR}/memory.json|csv")
        print(f"  FLOP table         -> {FLOPS_DIR}/flops.json|csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
