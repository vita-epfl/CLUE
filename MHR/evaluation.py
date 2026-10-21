"""Evaluate MHR variance propagation methods against shared baselines.

This mirrors the FLAME and SMPL-X evaluation scripts. It compares CLUE,
Monte-Carlo sampling, and a Jacobian baseline on identity,
expression, and compact MHR model-parameter settings. Error-map figures (on the
neutral body, or with ``--set_parameters`` on seeded random bodies) are written under
``results/``; runtime, peak memory and FLOPs per method are saved as JSON/CSV
tables there.
"""

from __future__ import annotations

import argparse
import hashlib
import os

os.environ.setdefault("USER", f"uid{os.getuid()}")

from pathlib import Path
import time
import tracemalloc
from typing import Callable, Sequence, TypedDict

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import cmasher  # registers cmr.* colormaps with matplotlib

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)
import matplotlib.pyplot as plt

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

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import trimesh

from algorithm import CLUE
from model_loader import (
    MHR_MODEL_PARAMETER_NAMES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_IDENTITY_BLENDSHAPES,
    get_default_asset_folder,
    load_mhr_model,
)


RESULTS_DIR = "results"
ERROR_MAPS_DIR = os.path.join(RESULTS_DIR, "error_maps")
TIME_DIR = os.path.join(RESULTS_DIR, "time")
MEMORY_DIR = os.path.join(RESULTS_DIR, "memory")
FLOPS_DIR = os.path.join(RESULTS_DIR, "flops")
GT_CACHE_DIR = os.path.join(ERROR_MAPS_DIR, "ground_truth_variance")

SAMPLE_COUNTS = [5, 10, 100, 1000]
GT_SAMPLE_COUNT = 10_000
N_TIMING_WARMUP = 3
N_TIMING_TRIALS = 10
IDENTITY_EXPR_VAR_MAX = 0.5
MODEL_VAR_MAX = 0.1
RENDER_W = 400
RENDER_H = 400
LABEL_H = 20
PANEL_CROP = 48
PANEL_CROP_V = 8
MODEL_STD_TO_MM = 10.0


class ParamGroup(TypedDict):
    identity_idx: int | None
    expression_idx: int | None
    model_indices: list[int]     # each index carries its own, independent variance
    var_max: float               # σ² when the group is uncertain
    value_max: float | None      # |value| when a randomised body sets the group (None: never randomised)


class EvaluationSetting(TypedDict):
    name: str
    display: str
    values: dict[str, float]     # operating point: value per group (on each of its indices); {} = neutral body
    variances: dict[str, float]  # σ² on each index of each uncertain group


_SPINE1_RZ = MHR_MODEL_PARAMETER_NAMES.index("spine1_rz_flexible")
_HIP_WIDTH = MHR_MODEL_PARAMETER_NAMES.index("hip_width_flexible")
_R_UPLEG_RZ = MHR_MODEL_PARAMETER_NAMES.index("r_upleg_rz")
_R_UPARM_RZ = MHR_MODEL_PARAMETER_NAMES.index("r_uparm_rz")
BODY_MODEL_PARAMS = range(6, 68)   # the 62 body rotations: spine, neck, head, arms, legs, feet


def _body_axis_groups() -> list[str]:
    """One group per body rotation (named after its model parameter) except
    spine1 z, which is the "spine" group."""
    return [MHR_MODEL_PARAMETER_NAMES[i] for i in BODY_MODEL_PARAMS if i != _SPINE1_RZ]


PARAM_GROUPS: dict[str, ParamGroup] = {
    "identity":        {"identity_idx": 0,    "expression_idx": None, "model_indices": [],
                        "var_max": IDENTITY_EXPR_VAR_MAX, "value_max": 0.5},
    "expression":      {"identity_idx": None, "expression_idx": 0,    "model_indices": [],
                        "var_max": IDENTITY_EXPR_VAR_MAX, "value_max": 0.5},
    "spine_hip_width": {"identity_idx": None, "expression_idx": None, "model_indices": [_SPINE1_RZ, _HIP_WIDTH],
                        "var_max": MODEL_VAR_MAX, "value_max": None},
    "spine":           {"identity_idx": None, "expression_idx": None, "model_indices": [_SPINE1_RZ],
                        "var_max": MODEL_VAR_MAX, "value_max": 0.3},
    "hip_shoulder":    {"identity_idx": None, "expression_idx": None, "model_indices": [_R_UPLEG_RZ, _R_UPARM_RZ],
                        "var_max": MODEL_VAR_MAX, "value_max": 0.3},
    # --set_parameters: every body rotation is its own parameter.
    **{name: {"identity_idx": None, "expression_idx": None,
              "model_indices": [MHR_MODEL_PARAMETER_NAMES.index(name)],
              "var_max": MODEL_VAR_MAX, "value_max": 0.3} for name in _body_axis_groups()},
}


device = torch.device("cpu")
model: object | None = None
clue: CLUE | None = None
_renderer: pyrender.OffscreenRenderer | None = None

N_MODEL_PARAMS = 0
BASE_IDENTITY = torch.empty(0)
BASE_EXPR = torch.empty(0)
BASE_MODEL_PARAMS = torch.empty(0)
FACES = np.empty((0, 3), dtype=np.int32)
SETTINGS: list[EvaluationSetting] = []
APPLY_CORRECTIVES = False


def _ensure_output_dirs() -> None:
    os.makedirs(ERROR_MAPS_DIR, exist_ok=True)
    os.makedirs(TIME_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(FLOPS_DIR, exist_ok=True)
    os.makedirs(GT_CACHE_DIR, exist_ok=True)


def _build_settings() -> list[EvaluationSetting]:
    return [
        {"name": "identity_0",      "display": "identity, 1st component",
         "values": {}, "variances": {"identity": IDENTITY_EXPR_VAR_MAX}},
        {"name": "expression_0",    "display": "expression[0]",
         "values": {}, "variances": {"expression": IDENTITY_EXPR_VAR_MAX}},
        {"name": "spine_hip_width", "display": "spine and hip width",
         "values": {}, "variances": {"spine_hip_width": MODEL_VAR_MAX}},
    ]


# Spine sign (× value_max) of each randomised pose, one pose per entry. Fixed
# rather than drawn, as in the FLAME and SMPL-X studies: the spine variance
# dominates every row where the spine is uncertain, so it is limited to the
# first and last rows. Everything else is drawn per pose.
RANDOM_BODY_SPINE_SIGNS = (-1, 0, 0, +1)


def random_pose_settings(seed: int) -> list[EvaluationSetting]:
    """Randomised poses for ``--set_parameters`` (deterministic given *seed*).

    spine1 z follows RANDOM_BODY_SPINE_SIGNS; identity[0] and every other body
    rotation are drawn independently and uniformly from {0, +value_max,
    -value_max}. Each parameter that is set (non-zero) is also uncertain, with
    its var_max. Draws repeating an earlier pose are redrawn.
    """
    rng = np.random.default_rng(seed)
    drawn = ["identity", *_body_axis_groups()]
    groups = ["identity", "spine", *_body_axis_groups()]
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
        identity = f"identity {values['identity']:+.1f}" if values["identity"] else "identity 0"
        spine = f"spine {values['spine']:+.1f}" if values["spine"] else "spine 0"
        bodies.append({
            "name": f"pose_{len(bodies)}",
            "display": f"{identity}, {spine} rad\n{n_axes} of {len(drawn) - 1} body rotations ±0.3 rad",
            "values": values,
            "variances": variances,
        })
    return bodies


def _require_model():
    if model is None:
        raise RuntimeError("MHR evaluation has not been initialised.")
    return model


def _require_clue() -> CLUE:
    if clue is None:
        raise RuntimeError("MHR evaluation has not been initialised.")
    return clue


def _require_renderer() -> pyrender.OffscreenRenderer:
    if _renderer is None:
        raise RuntimeError("MHR evaluation renderer has not been initialised.")
    return _renderer


def _initialise_experiment(args: argparse.Namespace) -> None:
    global device, model, clue, _renderer
    global N_MODEL_PARAMS, BASE_IDENTITY, BASE_EXPR, BASE_MODEL_PARAMS, FACES, SETTINGS
    global GT_SAMPLE_COUNT, SAMPLE_COUNTS, APPLY_CORRECTIVES

    _ensure_output_dirs()
    if _renderer is not None:
        _renderer.delete()
        _renderer = None

    device = torch.device("cpu")
    APPLY_CORRECTIVES = bool(args.apply_correctives)
    GT_SAMPLE_COUNT = int(args.gt_sample_count)
    SAMPLE_COUNTS = list(args.sample_counts)

    print(f"Device: {device}")
    model = load_mhr_model(
        folder=Path(os.path.expanduser(os.path.expandvars(args.asset_folder))),
        device=device,
        lod=args.lod,
        wants_pose_correctives=APPLY_CORRECTIVES,
    )
    model.eval()
    clue = CLUE(model)

    N_MODEL_PARAMS = (
        model.character.parameter_transform.size
        - NUM_IDENTITY_BLENDSHAPES
        - NUM_FACE_EXPRESSION_BLENDSHAPES
    )
    BASE_IDENTITY = torch.zeros(1, NUM_IDENTITY_BLENDSHAPES, dtype=torch.float32, device=device)
    BASE_EXPR = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES, dtype=torch.float32, device=device)
    BASE_MODEL_PARAMS = torch.zeros(1, N_MODEL_PARAMS, dtype=torch.float32, device=device)
    FACES = model.character.mesh.faces
    SETTINGS = _build_settings()

    _renderer = pyrender.OffscreenRenderer(RENDER_W, RENDER_H)


def _get_cmap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.cm as _mpl_cm

        return _mpl_cm.get_cmap(name)


_BASE_COLOR = np.array(_get_cmap("turbo")(0.0)[:3], dtype=np.float32)
_AMPLITUDE_HIGH = np.array([0.85, 0.05, 0.05], dtype=np.float32)


def _main_amplitude_cmap(t: np.ndarray) -> np.ndarray:
    tt = np.clip(t, 0.0, 1.0).astype(np.float32)
    rgb = (1.0 - tt[:, None]) * _BASE_COLOR[None, :] + tt[:, None] * _AMPLITUDE_HIGH[None, :]
    alpha = np.ones((tt.shape[0], 1), dtype=np.float32)
    return (np.concatenate([rgb, alpha], axis=-1) * 255).astype(np.uint8)


def apply_cmap(t: np.ndarray, cmap_name: str = "cmr.guppy") -> np.ndarray:
    if cmap_name == "main_amplitude":
        return _main_amplitude_cmap(t)
    cmap = _get_cmap(cmap_name)
    return (cmap(np.clip(t, 0.0, 1.0)) * 255).astype(np.uint8)


_vc_material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    metallicFactor=0.5,
    roughnessFactor=0.52,
    alphaMode="OPAQUE",
    doubleSided=True,
)


def _camera_pose(distance: float, elevation_deg: float = 15.0) -> np.ndarray:
    elevation = np.deg2rad(elevation_deg)
    pos = np.array([0.0, np.sin(elevation) * distance, np.cos(elevation) * distance])
    fwd = -pos / (np.linalg.norm(pos) + 1e-8)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -fwd
    pose[:3, 3] = pos
    return pose


def render_scalar_mesh(
    v_np: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    vmax: float,
    cmap_name: str = "main_amplitude",
) -> Image.Image:
    vmax = max(float(vmax), 1e-12)
    colors = apply_cmap(values / vmax, cmap_name=cmap_name)

    bounds = np.array([v_np.min(axis=0), v_np.max(axis=0)])
    center = bounds.mean(axis=0)
    extent = max(float(np.max(bounds[1] - bounds[0])), 1e-3)
    tri_mesh = trimesh.Trimesh(v_np - center, faces, process=False)
    tri_mesh.visual.vertex_colors = colors
    tri_mesh.fix_normals()

    scene = pyrender.Scene(ambient_light=[0.12, 0.13, 0.14])
    primitive = pyrender.Primitive(
        positions=tri_mesh.vertices.astype(np.float32),
        normals=tri_mesh.vertex_normals.astype(np.float32),
        indices=tri_mesh.faces.astype(np.uint32),
        color_0=colors,
        material=_vc_material,
    )
    scene.add(pyrender.Mesh(primitives=[primitive]))
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=_camera_pose(2.1 * extent))

    def _light(direction, intensity):
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

    _light([-0.4, -0.55, -1.0], 3.5)
    _light([0.8, -0.2, -0.6], 1.75)
    _light([0.0, 1.0, -0.5], 0.75)
    color, _ = _require_renderer().render(scene, flags=pyrender.RenderFlags.RGBA)
    fg = Image.fromarray(color, mode="RGBA")
    bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    bg.paste(fg, mask=fg.split()[3])
    return bg.convert("RGB")


def render_variance_mesh(v_np: np.ndarray, var_xyz: np.ndarray, vmax_mm: float) -> Image.Image:
    std_mm = np.sqrt(np.maximum(var_xyz, 0.0).sum(axis=-1)) * MODEL_STD_TO_MM
    return render_scalar_mesh(v_np, FACES, std_mm, vmax_mm, cmap_name="turbo")


def _make_colorbar(height: int, vmax_value: float, cmap_name: str, unit_label: str) -> Image.Image:
    width = 70
    bar_w = 18
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_pil_font(11)
    for row in range(height):
        t = 1.0 - row / max(height - 1, 1)
        color = tuple(apply_cmap(np.array([t]), cmap_name=cmap_name)[0, :3].astype(int))
        draw.line([(8, row), (8 + bar_w, row)], fill=color)
    draw.rectangle([8, 0, 8 + bar_w, height - 1], outline=(80, 80, 80))
    draw.text((32, 0), f"{vmax_value:.2f}", fill=(20, 20, 20), font=font)
    draw.text((32, height - 14), f"0 {unit_label}", fill=(20, 20, 20), font=font)
    return img


def mhr_forward(
    identity_coeffs: torch.Tensor | None = None,
    model_parameters: torch.Tensor | None = None,
    face_expr_coeffs: torch.Tensor | None = None,
) -> np.ndarray:
    active_model = _require_model()
    with torch.no_grad():
        verts, _ = active_model(
            identity_coeffs if identity_coeffs is not None else BASE_IDENTITY,
            model_parameters if model_parameters is not None else BASE_MODEL_PARAMS,
            face_expr_coeffs if face_expr_coeffs is not None else BASE_EXPR,
            apply_correctives=APPLY_CORRECTIVES,
        )
    return verts[0].cpu().numpy()


def operating_params(setting: EvaluationSetting) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(identity, model parameters, expression) at *setting*'s operating point:
    each group's value is set on each of its indices."""
    identity, model_params, expr = BASE_IDENTITY.clone(), BASE_MODEL_PARAMS.clone(), BASE_EXPR.clone()
    for group, value in setting["values"].items():
        p = PARAM_GROUPS[group]
        if p["identity_idx"] is not None:
            identity[0, p["identity_idx"]] += value
        if p["expression_idx"] is not None:
            expr[0, p["expression_idx"]] += value
        for idx in p["model_indices"]:
            model_params[0, idx] += value
    return identity, model_params, expr


def _variance_tensors(setting: EvaluationSetting) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity_vars = torch.zeros_like(BASE_IDENTITY)
    expression_vars = torch.zeros_like(BASE_EXPR)
    model_vars = torch.zeros_like(BASE_MODEL_PARAMS)
    for group, sigma2 in setting["variances"].items():
        p = PARAM_GROUPS[group]
        if p["identity_idx"] is not None:
            identity_vars[0, p["identity_idx"]] = sigma2
        if p["expression_idx"] is not None:
            expression_vars[0, p["expression_idx"]] = sigma2
        for idx in p["model_indices"]:
            model_vars[0, idx] = sigma2
    return identity_vars, model_vars, expression_vars


def variance_approx(setting: EvaluationSetting) -> np.ndarray:
    identity_vars, model_vars, expression_vars = _variance_tensors(setting)
    identity, model_params, expr = operating_params(setting)
    var, _ = _require_clue().compute_variance(
        identity_vars,
        model_vars,
        expression_vars,
        identity_coeffs=identity,
        model_parameters=model_params,
        face_expr_coeffs=expr,
        apply_correctives=APPLY_CORRECTIVES,
    )
    return var[0].cpu().numpy()


def variance_jacobian(setting: EvaluationSetting) -> np.ndarray:
    """Full-Jacobian variance via torch.autograd.functional.jacobian.

    This mirrors the FLAME and SMPL-X baselines: compute the dense Jacobian of
    vertices with respect to the full MHR parameter vectors, then propagate the
    selected per-parameter variances with a squared-Jacobian matrix-vector
    product.

    Var[V_{v,k}] = sum_i (dV_{v,k}/didentity_i)^2 * sigma_i^2
                 + sum_j (dV_{v,k}/dmodel_j)^2    * sigma_j^2
                 + sum_l (dV_{v,k}/dexpr_l)^2     * sigma_l^2
    """
    active_model = _require_model()

    identity_vars, model_vars, expression_vars = (t[0] for t in _variance_tensors(setting))
    identity0, model0, expression0 = (t[0] for t in operating_params(setting))

    def _model(
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        verts, _ = active_model(
            identity_coeffs.unsqueeze(0),
            model_parameters.unsqueeze(0),
            face_expr_coeffs.unsqueeze(0),
            apply_correctives=APPLY_CORRECTIVES,
        )
        return verts[0]

    J_identity, J_model, J_expression = torch.autograd.functional.jacobian(
        _model,
        (identity0, model0, expression0),
        strategy="forward-mode",
        vectorize=True,
    )

    # Squared-Jacobian matrix-vector products over each parameter axis. Stays
    # in torch; the single NumPy conversion at the end is for the
    # rendering/plotting code and is a zero-copy view on CPU.
    with torch.no_grad():
        var = (
            J_identity.detach().square() @ identity_vars
            + J_model.detach().square() @ model_vars
            + J_expression.detach().square() @ expression_vars
        )
    return var.cpu().numpy()


def variance_sampling(setting: EvaluationSetting, n_samples: int) -> np.ndarray:
    """Sequential Monte-Carlo empirical variance -> (V, 3) numpy.

    Every uncertain coefficient and model-parameter index gets its own
    independent noise. The samples are kept as torch tensors and the
    (population) variance is taken in torch; the single NumPy conversion at the
    end is for the rendering/plotting code and is a zero-copy view on CPU.
    """
    active_model = _require_model()
    stds = {group: float(np.sqrt(sigma2)) for group, sigma2 in setting["variances"].items()}
    base_identity, base_model_params, base_expr = operating_params(setting)
    verts_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            identity = base_identity.clone()
            expr = base_expr.clone()
            model_params = base_model_params.clone()
            for group, std in stds.items():
                p = PARAM_GROUPS[group]
                if p["identity_idx"] is not None:
                    identity[0, p["identity_idx"]] += torch.randn((), device=device) * std
                if p["expression_idx"] is not None:
                    expr[0, p["expression_idx"]] += torch.randn((), device=device) * std
                for idx in p["model_indices"]:
                    model_params[0, idx] += torch.randn((), device=device) * std
            verts, _ = active_model(identity, model_params, expr, apply_correctives=APPLY_CORRECTIVES)
            verts_list.append(verts[0])   # (V, 3) torch
        # Population variance (unbiased=False), the same estimator as
        # np.var(axis=0) with its default ddof=0.
        var = torch.stack(verts_list, dim=0).var(dim=0, unbiased=False)
    return var.cpu().numpy()


def std_error_mm(gt_var_xyz: np.ndarray, pred_var_xyz: np.ndarray) -> np.ndarray:
    gt_std = np.sqrt(np.maximum(gt_var_xyz, 0.0).sum(axis=-1))
    pred_std = np.sqrt(np.maximum(pred_var_xyz, 0.0).sum(axis=-1))
    return np.abs(pred_std - gt_std) * MODEL_STD_TO_MM


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


def run_qualitative(setting: EvaluationSetting) -> None:
    print(f"  [qualitative] computing baseline variances + GT (N={GT_SAMPLE_COUNT}) ...")
    v_base = mhr_forward(*operating_params(setting))
    gt_var = load_or_compute_gt_variance(setting)
    var_approx = variance_approx(setting)
    print("    approx done")
    var_jac = variance_jacobian(setting)
    print("    jacobian done")
    var_samples = []
    for n in SAMPLE_COUNTS:
        var_samples.append(variance_sampling(setting, n))
        print(f"    sampling N={n} done")

    methods = [var_approx] + var_samples + [var_jac]
    meth_lbls = ["CLUE"] + [f"Sample N={n}" for n in SAMPLE_COUNTS] + ["Jacobian"]
    errors_mm = [std_error_mm(gt_var, method) for method in methods]
    vmax_err = max(float(error.max()) for error in errors_mm)
    vmax_err = max(vmax_err, 1e-12)

    imgs = [
        render_scalar_mesh(v_base, FACES, err_mm, vmax_err, cmap_name="main_amplitude")
        for err_mm in errors_mm
    ]

    n_panels = len(imgs)
    cbar_w = 70
    panel_w = RENDER_W - 2 * PANEL_CROP
    panel_h = RENDER_H - 2 * PANEL_CROP_V
    canvas = Image.new("RGB", (panel_w * n_panels + cbar_w, panel_h + LABEL_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _get_pil_font(14)
    for idx, (img, label) in enumerate(zip(imgs, meth_lbls)):
        cropped = img.crop((PANEL_CROP, PANEL_CROP_V, RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
        canvas.paste(cropped, (idx * panel_w, LABEL_H))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (idx * panel_w + (panel_w - (bbox[2] - bbox[0])) // 2, 3),
            label,
            fill=(30, 30, 30),
            font=font,
        )
    cbar = _make_colorbar(panel_h, vmax_err, cmap_name="main_amplitude", unit_label="mm")
    canvas.paste(cbar, (panel_w * n_panels, LABEL_H))
    out = os.path.join(ERROR_MAPS_DIR, f"{setting['name']}.png")
    canvas.save(out)
    print(f"  Saved {out}")


def run_qualitative_combined(
    settings: Sequence[EvaluationSetting] | None = None,
    out_name: str = "combined.png",
    subtitle: str = "Shared scale (global vmax)",
) -> None:
    """Render all settings x methods in one figure with a shared colorbar.

    Each row's mesh is posed at that setting's operating point. Defaults
    (SETTINGS) give results/error_maps/combined.png (neutral body, one
    uncertain group per row).
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    print(f"  [qualitative-combined] computing all variances + GT (N={GT_SAMPLE_COUNT}) ...")
    settings = SETTINGS if settings is None else settings
    meth_lbls = [f"Sample N={n}" for n in SAMPLE_COUNTS] + ["Jacobian"] + [r"$\mathbf{CLUE}$"]
    n_settings = len(settings)
    n_methods = len(meth_lbls)

    all_errors: list[list[np.ndarray]] = []
    for setting in settings:
        gt_var = load_or_compute_gt_variance(setting)
        methods = (
            [variance_sampling(setting, n) for n in SAMPLE_COUNTS]
            + [variance_jacobian(setting)]
            + [variance_approx(setting)]
        )
        all_errors.append([std_error_mm(gt_var, method) for method in methods])
        detail = (f"values {setting['values']}, σ² {setting['variances']}" if len(setting["values"]) <= 6
                  else f"{len(setting['variances'])} uncertain parameters")
        print(f"    {setting['name']} errors computed  ({detail})")
        for label, error_mm in zip(meth_lbls, all_errors[-1]):
            print(f"      {label:<20s} mean {error_mm.mean():.3f}  max {error_mm.max():.3f} mm")

    all_errors_flat = np.concatenate([error for row in all_errors for error in row])
    vmax_err_mm = float(np.percentile(all_errors_flat, 99.5))
    vmax_err_mm = max(vmax_err_mm, 1e-12)
    print(
        f"  Global vmax_err (99.5th pct) = {vmax_err_mm:.4f} mm  "
        f"[abs max = {float(all_errors_flat.max()):.4f} mm]"
    )

    all_imgs: list[list[np.ndarray]] = []
    for setting, errors in zip(settings, all_errors):
        v_base = mhr_forward(*operating_params(setting))
        row_imgs = []
        for error_mm in errors:
            img = render_scalar_mesh(v_base, FACES, error_mm, vmax_err_mm, cmap_name="cmr.guppy")
            cropped = img.crop((PANEL_CROP, PANEL_CROP_V, RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
            row_imgs.append(np.asarray(cropped))
        all_imgs.append(row_imgs)
        print(f"    {setting['name']} rendered")

    panel_w = RENDER_W - 2 * PANEL_CROP
    panel_h = RENDER_H - 2 * PANEL_CROP_V
    dpi = 100
    cell_w = panel_w / dpi
    cell_h = panel_h / dpi
    fig, axes = plt.subplots(
        n_settings,
        n_methods,
        figsize=(cell_w * n_methods + 1.6, cell_h * n_settings + 0.6),
        dpi=dpi,
    )
    fig.patch.set_facecolor("white")
    if n_settings == 1:
        axes = np.asarray([axes])

    for setting_idx in range(n_settings):
        for method_idx in range(n_methods):
            ax = axes[setting_idx, method_idx]
            ax.set_facecolor("white")
            ax.imshow(all_imgs[setting_idx][method_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if setting_idx == 0:
                ax.set_title(meth_lbls[method_idx], fontsize=8, pad=3)
            if method_idx == 0:
                ax.set_ylabel(settings[setting_idx]["display"], fontsize=9, labelpad=5)

    norm = Normalize(vmin=0.0, vmax=vmax_err_mm)
    sm = ScalarMappable(cmap=_get_cmap("cmr.guppy"), norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.92, top=0.93, bottom=0.06, hspace=-0.15, wspace=0.02)
    cbar_ax = fig.add_axes([0.935, 0.10, 0.015, 0.68])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.0, vmax_err_mm])
    cbar.set_ticklabels(["0", rf"$\geq {vmax_err_mm:.2f}$"], fontsize=11)
    cbar.set_label(r"$|\Delta\sigma|$ (mm)", fontsize=11)
    fig.suptitle(r"Variance Error  $\Delta\sigma$  vs Ground Truth", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.945,
        f"{subtitle} - GT: N={GT_SAMPLE_COUNT} samples",
        ha="center",
        va="top",
        fontsize=12,
        color="#334155",
    )
    out = os.path.join(ERROR_MAPS_DIR, out_name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def _time_fn(fn: Callable[[], object], warmup: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
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
    counted, which is the standard convention. The MHR rig is a TorchScript module;
    ``flops_one_forward`` in the saved metadata records how much of a plain
    forward the counter attributes to it (sparse ops, if any, are not counted).
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
    forward_flops = _count_flops_all(lambda: mhr_forward())
    forward_flops_matmul = _count_flops(lambda: mhr_forward())
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


def run_qualitative_gt(
    settings: Sequence[EvaluationSetting] | None = None,
    out_name: str = "gt_variance.png",
) -> None:
    """Render per-vertex GT variance std (mm) for every setting."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    print(f"  [qualitative-gt] rendering GT variance maps (N={GT_SAMPLE_COUNT}) ...")
    settings = SETTINGS if settings is None else settings

    gt_stds = []
    for setting in settings:
        gt_var = load_or_compute_gt_variance(setting)
        gt_std_mm = np.sqrt(np.maximum(gt_var, 0.0).sum(axis=-1)) * MODEL_STD_TO_MM
        gt_stds.append(gt_std_mm)

    all_gt_flat = np.concatenate(gt_stds)
    global_vmax = float(np.percentile(all_gt_flat, 80)) or 1e-12

    all_imgs = []
    for setting, gt_std_mm in zip(settings, gt_stds):
        v_base = mhr_forward(*operating_params(setting))
        img = render_scalar_mesh(v_base, FACES, gt_std_mm, global_vmax, cmap_name="turbo")
        cropped = img.crop((PANEL_CROP, PANEL_CROP_V, RENDER_W - PANEL_CROP, RENDER_H - PANEL_CROP_V))
        all_imgs.append(np.asarray(cropped))
        print(f"    {setting['name']} GT rendered")

    panel_w = RENDER_W - 2 * PANEL_CROP
    panel_h = RENDER_H - 2 * PANEL_CROP_V
    dpi = 100
    n_settings = len(settings)
    fig, axes = plt.subplots(
        1,
        n_settings,
        figsize=(panel_w * n_settings / dpi + 1.6, panel_h / dpi + 0.6),
        dpi=dpi,
    )
    fig.patch.set_facecolor("white")
    if n_settings == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        ax.set_facecolor("white")
        ax.imshow(all_imgs[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(settings[idx]["display"], fontsize=9, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)

    norm = Normalize(vmin=0.0, vmax=global_vmax)
    sm = ScalarMappable(cmap=_get_cmap("turbo"), norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.90, top=0.82, bottom=0.04, wspace=0.02)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.58])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.0, global_vmax])
    cbar.set_ticklabels(["0", f"{global_vmax:.2f}"], fontsize=11)
    cbar.set_label("GT std (mm)", fontsize=11)
    fig.suptitle("Ground Truth Variance", fontsize=20, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.89,
        f"Shared turbo scale - N={GT_SAMPLE_COUNT} samples",
        ha="center",
        va="top",
        fontsize=12,
        color="#334155",
    )

    out = os.path.join(ERROR_MAPS_DIR, out_name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MHR variance comparison experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--asset-folder",
        type=str,
        default=str(get_default_asset_folder()),
        help="Path to the MHR assets folder.",
    )
    parser.add_argument("--lod", type=int, default=1, choices=list(range(7)), help="MHR LOD for evaluation.")
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=SAMPLE_COUNTS,
        help="Monte-Carlo sample counts used in plots.",
    )
    parser.add_argument("--gt-sample-count", type=int, default=GT_SAMPLE_COUNT, help="Ground-truth sample count.")
    parser.add_argument(
        "--apply-correctives",
        action="store_true",
        default=False,
        help="Render deterministic pose correctives; CLUE does not propagate corrective-network uncertainty.",
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
             "0, +max by row; identity[0] and every other body rotation each drawn from "
             "{0, +max, -max} (each its own parameter); every parameter that is set is "
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
    print("  MHR Variance Comparison Experiment")
    print(f"  assets  : {args.asset_folder}  lod={args.lod}")
    print(f"  params  : {N_MODEL_PARAMS} compact model + {NUM_IDENTITY_BLENDSHAPES} identity + {NUM_FACE_EXPRESSION_BLENDSHAPES} expression")
    print(f"  vertices: {active_model.character.mesh.n_vertices}")
    print("=" * 60)

    spine_values = " / ".join(f"{s * PARAM_GROUPS['spine']['value_max']:g}" for s in RANDOM_BODY_SPINE_SIGNS)
    if args.set_parameters:
        settings = random_pose_settings(args.seed)
        suffix = f"_random_seed{args.seed}"
        subtitle = (f"Random poses (seed {args.seed}): identity[0] ∈ {{0, ±0.5}}, every other body rotation ∈ "
                    f"{{0, ±0.3}} rad, spine = {spine_values} by row; every set parameter uncertain at its max σ²")
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
