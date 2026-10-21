# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MHR_PATH = ROOT_DIR / "assets" / "mhr" / "mhr_model.pt"

HAND_REFINEMENT_PREFIXES = (
    "head_pose_hand.",
    "init_pose_hand.",
    "head_camera_hand.",
    "init_camera_hand.",
    "init_to_token_mhr_hand.",
    "prev_to_token_mhr_hand.",
    "decoder_hand.",
    "hand_pe_layer.",
    "ray_cond_emb_hand.",
    "keypoint_embedding_hand.",
    "keypoint_posemb_linear_hand.",
    "keypoint_feat_linear_hand.",
    "keypoint3d_embedding_hand.",
    "keypoint3d_posemb_linear_hand.",
    "hand_box_embedding.",
    "hand_cls_embed.",
    "bbox_embed.",
)
EXPECTED_BODY_ONLY_MISSING_KEYS = {
    "backbone.encoder.mask_token",
    "head_pose.hand_pose_comps_ori",
}
EXPECTED_BODY_ONLY_MISSING_PREFIXES = (
    "head_pose.mhr.",
)
CHECKPOINT_LOAD_MODES = {
    "full": "full",
    "sam": "full",
    "sam3d": "full",
    "sam3d_body": "full",
    "backbone": "backbone_only",
    "backbone_only": "backbone_only",
    "dino": "backbone_only",
    "dino_only": "backbone_only",
    "none": "none",
    "scratch": "none",
    "random": "none",
}


def _find_model_config(
    checkpoint_path: Path, config_path: str | os.PathLike | None
) -> Path:
    if config_path is not None:
        config = Path(config_path)
        if not config.exists():
            raise FileNotFoundError(f"SAM 3D Body config does not exist: {config}")
        return config

    # Check the current directory, and if not present check the parent dir.
    candidates = (
        checkpoint_path.parent / "model_config.yaml",
        checkpoint_path.parent.parent / "model_config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No SAM 3D Body model_config.yaml found. Checked:\n{checked}"
    )


def _body_only_state_dict(
    state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(HAND_REFINEMENT_PREFIXES)
    }


def _backbone_only_state_dict(
    state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }


def _is_expected_body_only_missing_key(key: str) -> bool:
    return key in EXPECTED_BODY_ONLY_MISSING_KEYS or key.startswith(
        EXPECTED_BODY_ONLY_MISSING_PREFIXES
    )


def _summarize_keys(keys: list[str], *, limit: int = 12) -> str:
    if len(keys) <= limit:
        return ", ".join(keys)
    shown = ", ".join(keys[:limit])
    return f"{shown}, ... ({len(keys)} total)"


def _load_body_regressor_weights(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not _is_expected_body_only_missing_key(key)
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(HAND_REFINEMENT_PREFIXES)
    ]
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append(f"missing keys: {_summarize_keys(missing)}")
    if unexpected:
        details.append(f"unexpected keys: {_summarize_keys(unexpected)}")
    warnings.warn(
        "SAM 3D Body checkpoint had non-body-regressor mismatches: "
        + "; ".join(details),
        stacklevel=2,
    )


def _load_backbone_weights(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    state_dict = _backbone_only_state_dict(_body_only_state_dict(state_dict))
    if not state_dict:
        warnings.warn(
            "checkpoint_load_mode='backbone_only' found no backbone.* tensors.",
            stacklevel=2,
        )
        return
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_backbone = [
        key for key in incompatible.missing_keys if key.startswith("backbone.")
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(HAND_REFINEMENT_PREFIXES)
    ]
    if missing_backbone or unexpected:
        details = []
        if missing_backbone:
            details.append(f"missing backbone keys: {_summarize_keys(missing_backbone)}")
        if unexpected:
            details.append(f"unexpected keys: {_summarize_keys(unexpected)}")
        warnings.warn(
            "SAM 3D Body backbone-only checkpoint had mismatches: "
            + "; ".join(details),
            stacklevel=2,
        )
    print(f"Loaded {len(state_dict)} DINO/backbone tensors from SAM 3D Body checkpoint.")


def _normalize_checkpoint_load_mode(mode: str | None) -> str:
    key = str(mode or "full").lower()
    if key not in CHECKPOINT_LOAD_MODES:
        valid = ", ".join(sorted(set(CHECKPOINT_LOAD_MODES.values())))
        raise ValueError(f"Unsupported checkpoint_load_mode={mode!r}. Use one of: {valid}")
    return CHECKPOINT_LOAD_MODES[key]


def load_sam_3d_body(
    checkpoint_path: str | os.PathLike,
    device: str | torch.device | None = None,
    mhr_path: str | os.PathLike | None = None,
    config_path: str | os.PathLike | None = None,
    checkpoint_load_mode: str = "full",
):
    """Load the SAM 3D Body image-to-full-body regressor.

    This vendored loader intentionally keeps the body regression path only. The
    optional second-stage hand decoder and hand-box tokens are disabled.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM 3D Body checkpoint does not exist: {checkpoint_path}"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    mhr_path = Path(mhr_path) if mhr_path else DEFAULT_MHR_PATH
    if not mhr_path.exists():
        raise FileNotFoundError(
            f"SAM 3D Body MHR model does not exist: {mhr_path}"
        )

    checkpoint_load_mode = _normalize_checkpoint_load_mode(checkpoint_load_mode)
    print("Loading SAM 3D Body body regressor...")

    model_cfg = get_config(str(_find_model_config(checkpoint_path, config_path)))

    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(mhr_path)
    if "DECODER" in model_cfg.MODEL:
        model_cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS = False
    if (
        "PROMPT_ENCODER" in model_cfg.MODEL
        and "SAMPLE_HAND" in model_cfg.MODEL.PROMPT_ENCODER
    ):
        model_cfg.MODEL.PROMPT_ENCODER.SAMPLE_HAND = False
    model_cfg.freeze()

    model = SAM3DBody(model_cfg)

    if checkpoint_load_mode != "none":
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if checkpoint_load_mode == "full":
            _load_body_regressor_weights(model, _body_only_state_dict(state_dict))
        elif checkpoint_load_mode == "backbone_only":
            _load_backbone_weights(model, state_dict)
    else:
        print("Skipping SAM 3D Body checkpoint weight loading.")

    model = model.to(device)
    model.device = device
    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id)
    return (
        os.path.join(local_dir, "model.ckpt"),
        os.path.join(local_dir, "assets", "mhr_model.pt"),
    )


def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    kwargs.setdefault("mhr_path", mhr_path)
    return load_sam_3d_body(checkpoint_path=ckpt_path, **kwargs)
