"""Evaluate FLAME variance propagation methods against shared baselines.

The script compares CLUE, Monte-Carlo sampling, and a Jacobian baseline. It renders qualitative
error maps and a ground-truth variance figure (on the neutral face, or with
``--set_parameters`` on seeded random faces), and optionally measures
runtime, peak memory and FLOPs per method, saved as JSON/CSV tables under
``results/``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import sys
from typing import Iterator, Sequence, TypedDict

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

# FLAME models use np.infty which was removed in NumPy 2.0
if not hasattr(np, "infty"):
    np.infty = np.inf

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless / no-display rendering
import pyrender
import trimesh
from smplx.lbs import lbs as smplx_lbs


@contextmanager
def _temporary_cli_args(argv: Sequence[str]) -> Iterator[None]:
    """Temporarily replace ``sys.argv`` for libraries that parse CLI flags."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *argv]
        yield
    finally:
        sys.argv = original_argv

from flame_pytorch import FLAME, get_config
from algorithm import CLUE


def _load_flame_config():
    with _temporary_cli_args([]):
        return get_config()

# ─── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR = "results"
ERROR_MAPS_DIR = os.path.join(RESULTS_DIR, "error_maps")
TIME_DIR = os.path.join(RESULTS_DIR, "time")
MEMORY_DIR = os.path.join(RESULTS_DIR, "memory")
FLOPS_DIR = os.path.join(RESULTS_DIR, "flops")

os.makedirs(ERROR_MAPS_DIR, exist_ok=True)
os.makedirs(TIME_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(FLOPS_DIR, exist_ok=True)
GT_CACHE_DIR = os.path.join(ERROR_MAPS_DIR, "ground_truth_variance")
os.makedirs(GT_CACHE_DIR, exist_ok=True)

# ─── Model setup ──────────────────────────────────────────────────────────────
config = _load_flame_config()
device = torch.device("cpu")
print(f"Device: {device}")

flame     = FLAME(config).to(device)
flame.eval()
clue = CLUE(flame)

N_BETAS_TOTAL    = flame.shapedirs.shape[2]                             # 400
SHAPE_BETA_TOTAL = config.shape_params + flame.shape_betas.shape[1]    # 300

# Base parameters (mean face, neutral pose, zero translation)
BASE_SHAPE = torch.zeros(1, config.shape_params,      dtype=torch.float32, device=device)
BASE_EXPR  = torch.zeros(1, config.expression_params, dtype=torch.float32, device=device)
BASE_POSE  = torch.zeros(1, config.pose_params,       dtype=torch.float32, device=device)
BASE_NECK  = torch.zeros(1, 3,                        dtype=torch.float32, device=device)
BASE_EYE   = torch.zeros(1, 6,                        dtype=torch.float32, device=device)

FACES = flame.faces  # (F, 3) int array

# ─── Experiment constants ─────────────────────────────────────────────────────
SAMPLE_COUNTS   = [5, 10, 100, 1000]
# Row order of the cost tables (timing / memory / FLOPs): sampling, Jacobian, CLUE.
METHOD_NAMES    = [f"sample_N{n}" for n in SAMPLE_COUNTS] + ["jacobian", "clue"]
GT_SAMPLE_COUNT = 10_000
N_TIMING_WARMUP = 3        # warm-up runs before timing (flushes CUDA JIT / caches)
N_TIMING_TRIALS = 10       # timed repetitions; median is reported
SHAPE_VAR_MAX = 0.5
POSE_VAR_MAX  = 0.1
RENDER_W      = 400
RENDER_H      = 400
# tighten header space and panel crops for denser figures
LABEL_H       = 20       # pixel height of the label strip above each panel
PANEL_CROP    = 48       # pixels to crop from left/right of a panel before stitching
PANEL_CROP_V  = 10       # pixels to crop from top/bottom of a panel before stitching

# ─── Settings ─────────────────────────────────────────────────────────────────
# Component 0 of each evaluated parameter group.
# betas layout    : [shape_params | shape_betas | expr_params | expr_betas]
# full_pose layout: [global(3) | neck(3) | jaw(3) | eye(6)]
#   neck[0] → full_pose index 3
#   jaw[0]  → full_pose index 6
class ParamGroup(TypedDict):
    beta_idx: int | None
    pose_fp_idx: int | None
    var_max: float           # σ² when the group is uncertain
    value_max: float         # |value| when a randomised face sets the group


PARAM_GROUPS: dict[str, ParamGroup] = {
    "shape":      {"beta_idx": 0,                "pose_fp_idx": None, "var_max": SHAPE_VAR_MAX, "value_max": 0.5},
    "expression": {"beta_idx": SHAPE_BETA_TOTAL, "pose_fp_idx": None, "var_max": SHAPE_VAR_MAX, "value_max": 0.5},
    "jaw":        {"beta_idx": None,             "pose_fp_idx": 6,    "var_max": POSE_VAR_MAX,  "value_max": 0.3},
    "neck":       {"beta_idx": None,             "pose_fp_idx": 3,    "var_max": POSE_VAR_MAX,  "value_max": 0.3},
}


class EvaluationSetting(TypedDict):
    name: str
    display: str
    values: dict[str, float]     # operating point: component-0 value per group; {} = neutral face
    variances: dict[str, float]  # σ² on component 0 of each uncertain group


SETTINGS: list[EvaluationSetting] = [
    {"name": "shape_0",      "display": "shape, 1$^{\\mathrm{st}}$ principal component",
     "values": {}, "variances": {"shape": SHAPE_VAR_MAX}},
    {"name": "expression_0", "display": "expression[0]",
     "values": {}, "variances": {"expression": SHAPE_VAR_MAX}},
    {"name": "jaw_0",        "display": "jaw[0]",
     "values": {}, "variances": {"jaw": POSE_VAR_MAX}},
    {"name": "neck_0",       "display": "neck[0]",
     "values": {}, "variances": {"neck": POSE_VAR_MAX}},
]

# Neck sign (× value_max) of each randomised face, one face per entry. Fixed
# rather than drawn: the neck variance dominates every row where the neck is
# uncertain, so it is limited to the first and last rows.
RANDOM_FACE_NECK_SIGNS = (-1, 0, 0, +1)


def random_face_settings(seed: int) -> list[EvaluationSetting]:
    """Randomised faces for ``--set_parameters`` (deterministic given *seed*).

    neck[0] follows RANDOM_FACE_NECK_SIGNS; component 0 of every other group
    in PARAM_GROUPS is drawn uniformly from {0, +value_max, -value_max}. Each
    group that is set (non-zero) is also uncertain, with its var_max. Faces
    with no group set (nothing to propagate) or repeating an earlier face are
    redrawn.
    """
    rng = np.random.default_rng(seed)
    drawn = [g for g in PARAM_GROUPS if g != "neck"]
    faces: list[EvaluationSetting] = []
    for neck_sign in RANDOM_FACE_NECK_SIGNS:
        while True:
            signs = dict(zip(drawn, rng.integers(-1, 2, size=len(drawn))))   # -1, 0 or +1
            signs["neck"] = neck_sign
            values = {g: float(signs[g]) * p["value_max"] for g, p in PARAM_GROUPS.items()}
            if any(signs.values()) and all(values != face["values"] for face in faces):
                break
        variances = {g: p["var_max"] for g, p in PARAM_GROUPS.items() if signs[g]}
        shown = [f"{g} {v:+.1f}" if v else f"{g} 0" for g, v in values.items()]
        faces.append({
            "name": f"face_{len(faces)}",
            "display": f"{', '.join(shown[:2])}\n{', '.join(shown[2:])} rad",
            "values": values,
            "variances": variances,
        })
    return faces


def _add_to_param(group: str, delta: float, shape: torch.Tensor, expr: torch.Tensor,
                  pose: torch.Tensor, neck: torch.Tensor) -> None:
    """Add *delta* in place to component 0 of *group* in the FLAME inputs."""
    p = PARAM_GROUPS[group]
    if p["beta_idx"] is not None:
        idx = p["beta_idx"]
        if idx < config.shape_params:
            shape[0, idx] += delta
        else:
            expr[0, idx - SHAPE_BETA_TOTAL] += delta
    else:
        fp = p["pose_fp_idx"]
        if 3 <= fp <= 5:
            neck[0, fp - 3] += delta
        else:
            pose[0, fp - 3] += delta


def operating_params(
    setting: EvaluationSetting,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(shape, expression, pose_params, neck) at *setting*'s operating point.

    pose_params layout: [global(3) | jaw(3)].
    """
    shape, expr = BASE_SHAPE.clone(), BASE_EXPR.clone()
    pose,  neck = BASE_POSE.clone(),  BASE_NECK.clone()
    for group, value in setting["values"].items():
        _add_to_param(group, value, shape, expr, pose, neck)
    return shape, expr, pose, neck


def _variance_vectors(setting: EvaluationSetting) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal input variances: betas (N_BETAS_TOTAL,) and full_pose (15,)."""
    bv = torch.zeros(N_BETAS_TOTAL, dtype=torch.float32, device=device)
    pv = torch.zeros(15,            dtype=torch.float32, device=device)
    for group, sigma2 in setting["variances"].items():
        p = PARAM_GROUPS[group]
        if p["beta_idx"] is not None:
            bv[p["beta_idx"]] = sigma2
        else:
            pv[p["pose_fp_idx"]] = sigma2
    return bv, pv

# ─── Colormap helpers ─────────────────────────────────────────────────────────
def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.cm as _mpl_cm
        return _mpl_cm.get_cmap(name)


_BASE_COLOR     = np.array(_get_cmap("turbo")(0.0)[:3], dtype=np.float32)  # turbo t=0
_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)  # shared peak red
_ORANGE_LOW     = np.array([1.0,  0.55, 0.0 ], dtype=np.float32)  # orange (low uncertainty)
_GT_COLOR_LOW   = np.array([0.55, 0.10, 0.75], dtype=np.float32)  # purple (low GT std)
_GT_COLOR_HIGH  = np.array([0.10, 0.65, 0.15], dtype=np.float32)  # green  (high GT std)


def _main_amplitude_cmap(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0,1] using the shared amplitude ramp: blue base → red."""
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
    """Blend _BASE_COLOR (t=0) → peak_color (t=1) for per-group overlays."""
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
    """Render FLAME mesh coloured by scalar per-vertex values.

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
            colors  = _peak_alpha_colors(t, peak_color)
            material = _vc_material_blend
        else:
            colors  = _base_to_peak_cmap(t, peak_color)
            material = _vc_material
    else:
        colors   = apply_cmap(t, cmap_name=cmap_name)              # (V, 4) uint8
        material = _vc_material

    center   = v_np.mean(axis=0)
    v_c      = (v_np - center).astype(np.float32)
    extent   = float(np.max(v_np.max(axis=0) - v_np.min(axis=0)))
    dist     = 1.5 * extent

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

    _dir_light([-0.4, -0.55, -1.0], 3.5)
    _dir_light([ 0.8, -0.2,  -0.6], 1.5)
    _dir_light([ 0.0,  1.0,  -0.5], 0.8)

    color, _ = _renderer.render(scene)
    return Image.fromarray(color)


def render_variance_mesh(v_np: np.ndarray,
                         faces: np.ndarray,
                         var_xyz: np.ndarray,
                         vmax_std: float,
                         cmap_name: str = "cmr.guppy") -> Image.Image:
    """Render FLAME mesh coloured by per-vertex total std."""
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

    draw.text((bar_x0 - 2, bar_y0 - 2),     f"{vmax_value:.2f}", fill=(220, 220, 220), font=font)
    draw.text((bar_x0 - 2, bar_y1 + 2),     f"0 {unit_label}",    fill=(220, 220, 220), font=font)
    draw.text((5, H // 2 - 4),              unit_label,            fill=(180, 180, 180), font=font)
    return img


# ─── Single FLAME forward pass ────────────────────────────────────────────────
@torch.no_grad()
def flame_forward(shape=None, expr=None, pose=None, neck=None, eye=None) -> np.ndarray:
    """Return vertices (V, 3) numpy array for one set of parameters."""
    v, _, _ = flame(
        shape if shape is not None else BASE_SHAPE,
        expr  if expr  is not None else BASE_EXPR,
        pose  if pose  is not None else BASE_POSE,
        neck  if neck  is not None else BASE_NECK,
        eye   if eye   is not None else BASE_EYE,
    )
    return v[0].cpu().numpy()


# ─── Variance methods ─────────────────────────────────────────────────────────

def variance_approx(setting: EvaluationSetting) -> np.ndarray:
    """Closed-form CLUE → (V, 3) numpy."""
    bv, pv = _variance_vectors(setting)
    shape, expr, pose, neck = operating_params(setting)

    var, _ = clue.compute_variance(
        bv[None], pv[None],
        shape_params=shape, expression_params=expr,
        pose_params=pose,   neck_pose=neck, eye_pose=BASE_EYE,
    )
    return var[0].cpu().numpy()   # (V, 3)


def variance_jacobian(setting: EvaluationSetting) -> np.ndarray:
    """Full-Jacobian variance via torch.autograd.functional.jacobian.

    Computes J = (∂V/∂betas, ∂V/∂full_pose) over all inputs and propagates
    the per-parameter variances (same parameterisation as variance_approx):

        Var[V_{v,k}] = Σ_i (∂V_{v,k}/∂beta_i)²   · σ²_beta_i
                     + Σ_j (∂V_{v,k}/∂pose_j)²   · σ²_pose_j

    betas layout    : [shape_params | shape_betas | expr_params | expr_betas]
    full_pose layout: [global(3) | neck(3) | jaw(3) | eye(6)]
    """
    # Variance vectors — same layout and indexing as variance_approx
    bv, pv = _variance_vectors(setting)

    # Nominal operating point (full concatenated vectors)
    shape, expr, pose, neck = operating_params(setting)
    betas0 = torch.cat([
        shape[0],
        flame.shape_betas[0],
        expr[0],
        flame.expression_betas[0],
    ])  # (N_BETAS_TOTAL,)

    pose0 = torch.cat([
        pose[0, :3],        # global
        neck[0],            # neck
        pose[0, 3:],        # jaw
        BASE_EYE[0],        # eye
    ])  # (15,)

    template = flame.v_template  # (V, 3)

    def _model(betas: torch.Tensor, full_pose: torch.Tensor) -> torch.Tensor:
        verts, _ = smplx_lbs(
            betas.unsqueeze(0),
            full_pose.unsqueeze(0),
            template.unsqueeze(0),
            flame.shapedirs,
            flame.posedirs,
            flame.J_regressor,
            flame.parents,
            flame.lbs_weights,
        )
        return verts[0]   # (V, 3)

    J_betas, J_pose = torch.autograd.functional.jacobian(
        _model, (betas0, pose0),
        strategy="forward-mode",
        vectorize=True,
    )   # (V, 3, N_BETAS_TOTAL), (V, 3, 15)

    # Var[V_{v,k}] = Σ_i J²_{v,k,i} · σ²_i  (matrix-vector product over the
    # parameter axis). Stays in torch; the single NumPy conversion at the end
    # is for the rendering/plotting code and is a zero-copy view on CPU.
    with torch.no_grad():
        var = J_betas.detach().square() @ bv + J_pose.detach().square() @ pv   # (V, 3)
    return var.cpu().numpy()


def variance_sampling(setting: EvaluationSetting, n_samples: int) -> np.ndarray:
    """Sequential Monte-Carlo empirical variance → (V, 3) numpy.

    Each sample is run as a separate single-item LBS call, matching the
    sequential (batch_size=1) processing used by the other baselines. Every
    uncertain group gets its own independent noise. The samples are kept as
    torch tensors and the (population) variance is taken in torch; the single
    NumPy conversion at the end is for the rendering/plotting code and is a
    zero-copy view on CPU.
    """
    stds = {group: float(np.sqrt(sigma2)) for group, sigma2 in setting["variances"].items()}
    template = flame.v_template.unsqueeze(0)   # (1, V, 3)
    shape0, expr0, pose0, neck0 = operating_params(setting)

    verts_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            shape_s = shape0.clone()
            expr_s  = expr0.clone()
            pose_s  = pose0.clone()
            neck_s  = neck0.clone()
            eye_s   = BASE_EYE.clone()

            for group, std in stds.items():
                noise = torch.randn(1, device=device) * std   # scalar per sample and group
                _add_to_param(group, noise.item(), shape_s, expr_s, pose_s, neck_s)

            betas     = torch.cat([shape_s, flame.shape_betas, expr_s, flame.expression_betas], dim=1)
            full_pose = torch.cat([pose_s[:, :3], neck_s, pose_s[:, 3:], eye_s], dim=1)

            verts, _ = smplx_lbs(
                betas, full_pose, template,
                flame.shapedirs, flame.posedirs,
                flame.J_regressor, flame.parents, flame.lbs_weights,
            )
            verts_list.append(verts[0])   # (V, 3) torch

        # Population variance (unbiased=False), the same estimator as
        # np.var(axis=0) with its default ddof=0.
        var = torch.stack(verts_list, dim=0).var(dim=0, unbiased=False)   # (V, 3)
    return var.cpu().numpy()


def std_error_mm(gt_var_xyz: np.ndarray, pred_var_xyz: np.ndarray) -> np.ndarray:
    """Per-vertex |Δσ| in mm, where σ = sqrt(Var[x]+Var[y]+Var[z])."""
    gt_std = np.sqrt(np.maximum(gt_var_xyz, 0.0).sum(axis=-1))
    pred_std = np.sqrt(np.maximum(pred_var_xyz, 0.0).sum(axis=-1))
    return np.abs(pred_std - gt_std) * 1000.0


def _gt_cache_path(setting: EvaluationSetting) -> str:
    if setting["values"]:   # randomised face: key on the operating point and the variances
        values = "_".join(f"{g}{v:+.2f}" for g, v in setting["values"].items())
        sigma2 = "_".join(f"{g}{s2:g}" for g, s2 in setting["variances"].items())
        stem = f"face_{values}_s2_{sigma2}".replace(".", "p")
    else:                   # neutral face, one uncertain group
        (group, sigma2), = setting["variances"].items()
        stem = f"{group}_s2_{sigma2:.8f}".replace(".", "p")
    return os.path.join(GT_CACHE_DIR, f"{stem}_N{GT_SAMPLE_COUNT}.npz")


def load_or_compute_gt_variance(
    setting: EvaluationSetting,
    force_recompute: bool = False,
) -> np.ndarray:
    """Load cached GT variance if available; otherwise compute with N=GT_SAMPLE_COUNT and cache."""
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
    v_base = flame_forward(*operating_params(setting))

    gt_var = load_or_compute_gt_variance(setting)

    var_approx = variance_approx(setting)
    print(f"    approx done")

    var_jac = variance_jacobian(setting)
    print(f"    jacobian done")

    var_samples = []
    for n in SAMPLE_COUNTS:
        var_samples.append(variance_sampling(setting, n))
        print(f"    sampling N={n} done")

    # Shared error scale: max |Δσ| across all baselines (mm)
    methods = [var_approx] + var_samples + [var_jac]
    meth_lbls = ["CLUE"] + [f"Sample N={n}" for n in SAMPLE_COUNTS] + ["Jacobian"]
    errors_mm = [std_error_mm(gt_var, m) for m in methods]
    vmax_err_mm = max(float(e.max()) for e in errors_mm)
    if vmax_err_mm < 1e-12:
        vmax_err_mm = 1e-12

    imgs = []
    for lbl, err_mm in zip(meth_lbls, errors_mm):
        img = render_scalar_mesh(v_base, FACES, err_mm, vmax_err_mm, cmap_name="main_amplitude")
        imgs.append(img)
        print(f"    rendered error '{lbl}'")

    # Stitch panels horizontally + label strip
    n_panels  = len(imgs)
    cbar_w    = 70
    panel_w   = RENDER_W - 2 * PANEL_CROP    # width after side-cropping
    panel_h   = RENDER_H - 2 * PANEL_CROP_V  # height after top/bottom cropping
    total_w   = panel_w * n_panels + cbar_w
    total_h   = panel_h + LABEL_H

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    font = _get_pil_font(14)

    for i, (img, lbl) in enumerate(zip(imgs, meth_lbls)):
        cropped = img.crop((PANEL_CROP, PANEL_CROP_V, RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
        canvas.paste(cropped, (i * panel_w, LABEL_H))
        # Centred label
        bbox = draw.textbbox((0, 0), lbl, font=font)
        tw   = bbox[2] - bbox[0]
        tx   = i * panel_w + (panel_w - tw) // 2
        draw.text((tx, 8), lbl, fill=(240, 240, 240), font=font)

    # Colorbar on the right
    cbar = _make_colorbar(panel_h, vmax_err_mm, cmap_name="main_amplitude", unit_label="mm")
    canvas.paste(cbar, (panel_w * n_panels, LABEL_H))

    # Title
    title = f"Variance error vs GT (N={GT_SAMPLE_COUNT}) - {setting['display']}   (s2_in = {setting['variances']})"
    draw.text((8, 8), title, fill=(200, 200, 200), font=font)

    out = os.path.join(ERROR_MAPS_DIR, f"{setting['name']}.png")
    canvas.save(out)
    print(f"  Saved {out}")


# ─── Combined qualitative experiment (shared blue→red colorbar) ──────────────

def run_qualitative_combined(
    settings: Sequence[EvaluationSetting] = SETTINGS,
    out_name: str = "combined.png",
    subtitle: str = "Shared scale (global vmax)",
) -> None:
    """Render all settings × methods in one figure with a shared blue→red colorbar.

    Each row's mesh is posed at that setting's operating point. Defaults give
    results/error_maps/combined.png (neutral face, one uncertain group per row).

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
        print(f"    {setting['name']} errors computed  (values {setting['values']}, σ² {setting['variances']})")
        for lbl, err_mm in zip(meth_lbls, all_errors[-1]):
            print(f"      {lbl:<20s} mean {err_mm.mean():.3f}  max {err_mm.max():.3f} mm")

    all_errors_flat = np.concatenate([e for row in all_errors for e in row])
    vmax_err_mm = float(np.percentile(all_errors_flat, 99.5))
    if vmax_err_mm < 1e-12:
        vmax_err_mm = 1e-12
    print(f"  Global vmax_err (95th pct) = {vmax_err_mm:.4f} mm  "
          f"[abs max = {float(all_errors_flat.max()):.4f} mm]")

    # ── Pass 2: render all panels ──────────────────────────────────────────────
    # All panels use the same global vmax and the main_amplitude (blue→red) cmap.
    # No transparency; alphaMode is OPAQUE throughout.
    all_imgs: list[list[np.ndarray]] = []
    for setting, errors in zip(settings, all_errors):
        v_base = flame_forward(*operating_params(setting))
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

def _time_fn(fn, warmup: int, trials: int) -> float:
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
        "num_vertices": int(FACES.max()) + 1,
    }


def _method_fns(setting: EvaluationSetting) -> dict:
    """The six variance methods for one setting, keyed by METHOD_NAMES."""
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
                   "methods": METHOD_NAMES, "columns": columns, "values": values}, fh, indent=2)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["setting", "method", *columns])
        for st in SETTINGS:
            for method in METHOD_NAMES:
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
            print(f"    {setting['name']:14s} {method:14s} {seconds * 1e3:9.2f} ms")
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


def _measure_memory_breakdown(fn) -> dict:
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
    import gc, tracemalloc
    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        peak_delta = torch.cuda.max_memory_allocated() - baseline
        mb = max(peak_delta, 0) / 1e6   # bytes -> MB
        return {"total_mb": mb, "torch_mb": mb, "numpy_mb": None}
    else:
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
            print(f"    {setting['name']:14s} {method:14s} {mb['total_mb']:9.3f} MB"
                  f"  (torch {mb['torch_mb']:.3f}, numpy {mb['numpy_mb'] if mb['numpy_mb'] is not None else float('nan'):.3f})")
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

def _count_flops(fn) -> int:
    """FLOPs executed by fn() as counted by torch.utils.flop_counter.

    The counter sees every aten op after autograd and functorch have run, so
    forward-mode/vmap Jacobians are counted at their batched sizes. It counts
    matmul-class ops (mm, bmm, addmm, baddbmm, einsum lowered to these,
    convolutions) at 2 FLOPs per multiply-add; elementwise ops are not
    counted, which is the standard convention and dominates all methods here.
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
    """FLOPs of a method that the dispatcher-level counter cannot see (none by default)."""
    return 0


def run_flops() -> None:
    """Count FLOPs of every variance method for every setting; save
    results/flops/flops.{json,csv}."""
    print("  [flops] counting floating-point operations ...")
    vmap_ok = _flop_counter_vmap_check()
    forward_flops = _count_flops_all(lambda: flame_forward())
    forward_flops_matmul = _count_flops(lambda: flame_forward())
    print(f"    counter sees vmap-batched ops: {vmap_ok}; one plain forward: {forward_flops / 1e9:.4f} GFLOP "
          f"(matmul-class only {forward_flops_matmul / 1e9:.4f})")
    values = {}
    for setting in SETTINGS:
        row = {}
        for method, fn in _method_fns(setting).items():
            flops = _count_flops_all(fn) + _uncounted_flops(method)
            flops_matmul = _count_flops(fn)
            row[method] = {"flops": flops, "gflops": flops / 1e9, "gflops_matmul_only": flops_matmul / 1e9}
            print(f"    {setting['name']:14s} {method:14s} {flops / 1e9:12.4f} GFLOP  (matmul-class only {flops_matmul / 1e9:.4f})")
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
    settings: Sequence[EvaluationSetting] = SETTINGS,
    out_name: str = "gt_variance.png",
) -> None:
    """Render per-vertex GT variance std (mm) for each setting using a
    light-green → dark-green colour ramp.  Saved to results/error_maps/gt_variance.png
    (default) so it does not overwrite the baseline-comparison figure."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    print(f"  [qualitative-gt] rendering GT variance maps (N={GT_SAMPLE_COUNT}) ...")
    n_settings = len(settings)
    # Pass 1: compute per-vertex GT std for each setting and find global vmax
    gt_stds   = []
    gt_vmaxes = []
    for setting in settings:
        gt_var    = load_or_compute_gt_variance(setting)
        gt_std_mm = np.sqrt(np.maximum(gt_var, 0.0).sum(axis=-1)) * 1000.0
        gt_stds.append(gt_std_mm)
        gt_vmaxes.append(float(gt_std_mm.max()))

    all_gt_stds_flat = np.concatenate(gt_stds)
    global_vmax = float(np.percentile(all_gt_stds_flat, 80)) or 1e-12

    # Pass 2: render all panels with the shared global vmax
    imgs = []
    for s_idx, (setting, gt_std_mm) in enumerate(zip(settings, gt_stds)):
        v_base  = flame_forward(*operating_params(setting))
        img     = render_scalar_mesh(v_base, FACES, gt_std_mm, global_vmax,
                                     cmap_name="turbo")
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

    # tighter margins for ground-truth figure
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
        description="FLAME variance comparison experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        help="Run only the qualitative mesh-rendering outputs.",
    )
    parser.add_argument(
        "--set_parameters",
        action="store_true",
        help="Error maps on randomised faces instead of the neutral face: shape[0], "
             "expression[0] and jaw[0] each drawn from {0, +max, -max}, neck[0] = -max, "
             "0, 0, +max by row; every group that is set is uncertain with its maximum "
             "variance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the randomised faces of --set_parameters.",
    )
    return parser.parse_args(argv)


def _print_section(title: str) -> None:
    rule = "-" * 60
    print(f"\n{rule}")
    print(f"  {title}")
    print(rule)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 60)
    print("  FLAME Variance Comparison Experiment")
    print("=" * 60)

    if args.set_parameters:
        settings = random_face_settings(args.seed)
        suffix   = f"_random_seed{args.seed}"
        ranges   = ", ".join(f"{g}[0] ∈ {{0, ±{p['value_max']}}}"
                             for g, p in PARAM_GROUPS.items() if g != "neck")
        neck     = " / ".join(f"{s * PARAM_GROUPS['neck']['value_max']:g}" for s in RANDOM_FACE_NECK_SIGNS)
        subtitle = (f"Random faces (seed {args.seed}): {ranges}, neck[0] = {neck} by row; "
                    f"every set group uncertain at its max σ²")
    else:
        settings, suffix, subtitle = SETTINGS, "", "Shared scale (global vmax)"

    try:
        _print_section("Qualitative (all settings, shared colorbar)")
        run_qualitative_combined(settings, f"combined{suffix}.png", subtitle)

        _print_section("Qualitative GT variance (green ramp)")
        run_qualitative_gt(settings, f"gt_variance{suffix}.png")

        if not args.qualitative:
            _print_section("Timing benchmark")
            run_timing()

            _print_section("Memory benchmark")
            run_memory()

            _print_section("FLOP count")
            run_flops()
    finally:
        _renderer.delete()

    print("\nAll done.")
    print(f"  Error maps         -> {ERROR_MAPS_DIR}/")
    if not args.qualitative:
        print(f"  Timing table       -> {TIME_DIR}/timing.json|csv")
        print(f"  Memory table       -> {MEMORY_DIR}/memory.json|csv")
        print(f"  FLOP table         -> {FLOPS_DIR}/flops.json|csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
