# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

import os


def _ensure_uid_safe_environment() -> None:
    """Make UID-less containers safe for libraries that call getpass.getuser."""

    fallback_user = f"uid_{os.getuid()}"
    for key in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        os.environ.setdefault(key, fallback_user)
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join("/tmp", f"torchinductor_{fallback_user}"),
    )


_ensure_uid_safe_environment()

__version__ = "1.0.0"

from .build_models import load_sam_3d_body, load_sam_3d_body_hf

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]


def __getattr__(name):
    if name == "SAM3DBodyEstimator":
        from .sam_3d_body_estimator import SAM3DBodyEstimator

        return SAM3DBodyEstimator
    raise AttributeError(name)
