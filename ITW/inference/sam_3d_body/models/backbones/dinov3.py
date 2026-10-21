# Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
import os
import sys
from pathlib import Path

import torch
from torch import nn


def _candidate_dinov3_repos() -> list[Path]:
    candidates = []
    if os.environ.get("SAM3D_DINOV3_REPO"):
        candidates.append(Path(os.environ["SAM3D_DINOV3_REPO"]))

    root_dir = Path(__file__).resolve().parents[3]
    candidates.append(root_dir / "dinov3")

    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    candidates.append(torch_home / "hub" / "facebookresearch_dinov3_main")

    hub_dir = Path(torch.hub.get_dir())
    candidates.append(hub_dir / "facebookresearch_dinov3_main")

    # Useful for sandboxed smoke tests that redirect Torch Hub to /tmp.
    candidates.append(Path("/tmp/torchhub/hub/facebookresearch_dinov3_main"))
    return candidates


def _load_cached_backbone(name: str) -> nn.Module | None:
    for repo_dir in _candidate_dinov3_repos():
        if not (repo_dir / "dinov3" / "hub" / "backbones.py").exists():
            continue
        repo_parent = str(repo_dir)
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)
        backbones = importlib.import_module("dinov3.hub.backbones")
        # The DINOv3 hub builders pin drop_path_rate=0.0 themselves (a
        # drop-path kwarg is either rejected or ignored), so none is passed.
        return getattr(backbones, name)(pretrained=False)
    return None


class Dinov3Backbone(nn.Module):
    def __init__(
        self, name="dinov2_vitb14", pretrained_weight=None, cfg=None, *args, **kwargs
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg

        self.encoder = _load_cached_backbone(self.name)
        if self.encoder is None:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov3",
                self.name,
                source="github",
                pretrained=False,
            )
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.embed_dims = self.encoder.embed_dim

    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.encoder.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
