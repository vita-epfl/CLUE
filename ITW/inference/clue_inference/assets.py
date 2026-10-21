"""The model files, kept in the assets folder and laid out like their Hugging Face repositories:

    assets/clue.safetensors                         CLUE (meghshukla/clue)
    assets/sam3dbody_clue.safetensors               SAM 3D Body (CLUE): the uncertainty head for the released SAM 3D Body
                                                    (meghshukla/clue)
    assets/sam-3d-body-dinov3/model.ckpt            SAM 3D Body (Meta, facebook/sam-3d-body-dinov3)
    assets/sam-3d-body-dinov3/assets/mhr_model.pt   MHR rig (the same release)

Missing files are downloaded there; both repositories are gated (accept their terms on Hugging Face once and log in with
``hf auth login``). This module does not touch ``sys.path``, so it can be imported next to Meta's own sam_3d_body package.
"""

from __future__ import annotations

from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CLUE_REPO = "meghshukla/clue"                                    # gated, automatic approval
SAM3D_REPO = "facebook/sam-3d-body-dinov3"                       # gated: Meta's licence
FILES = {                                                        # key -> (repository, file in it, folder under assets)
    "checkpoint": (SAM3D_REPO, "model.ckpt", "sam-3d-body-dinov3"),
    "rig": (SAM3D_REPO, "assets/mhr_model.pt", "sam-3d-body-dinov3"),
    "clue": (CLUE_REPO, "clue.safetensors", ""),
    "sam3dbody_clue": (CLUE_REPO, "sam3dbody_clue.safetensors", ""),
}


def model_file(key: str, assets: Path = ASSETS) -> Path:
    """The path of one model file (a key of FILES), downloaded into ``assets`` when missing."""
    repo, name, folder = FILES[key]
    path = Path(assets) / folder / name
    if not path.exists():
        from huggingface_hub import hf_hub_download

        print(f"{path} not found: downloading {name} from {repo}", flush=True)
        path = Path(hf_hub_download(repo, name, local_dir=Path(assets) / folder))
    return path
