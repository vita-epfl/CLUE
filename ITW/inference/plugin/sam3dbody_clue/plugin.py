"""SAM 3D Body (CLUE): per-vertex uncertainty for a SAM 3D Body model you already run, without changing it.

The head was trained on top of the released SAM 3D Body, which stayed frozen. It reads the decoder's pose token and the
parameter predictions of SAM 3D Body's last forward through a hook on the MHR head; SAM 3D Body's own outputs do not
change. Attached to Meta's ``SAM3DBodyEstimator``, it adds each person's per-vertex uncertainty to the results of
``process_one_image``. See README.md.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import torch

from .head import VarianceHead, VertexUncertainty

WEIGHTS_REPO, WEIGHTS_FILE = "meghshukla/clue", "sam3dbody_clue.safetensors"     # gated, automatic approval


class SAM3DBodyCLUE:
    """Attach once to Meta's ``SAM3DBodyEstimator``. ``process_one_image`` stays exactly Meta's (same arguments, defaults
    and outputs); calls with SAM 3D Body's body decoder also return ``"vertex_sigma_cm"`` for each person:

        clue = SAM3DBodyCLUE(estimator)
        outputs = estimator.process_one_image(image, inference_type="body")   # outputs[i]["vertex_sigma_cm"]: (18439,) cm

    The body decoder (``inference_type="body"``) predicts the whole body, hands and face included; the head was trained
    on it, so the uncertainty describes exactly the returned mesh. Meta's default, ``"full"``, re-estimates the hands with
    a second decoder and replaces the hands and wrists of the mesh, which the head has not seen: such calls return SAM 3D
    Body's outputs unchanged, without uncertainty (with a one-time warning).

    ``estimator`` can also be a bare SAM 3D Body model: nothing is wrapped then, and ``sigma_cm()`` / ``variance_cm2()``
    give the uncertainty of its last forward. ``weights``: sam3dbody_clue.safetensors; downloaded from Hugging Face
    (meghshukla/clue, into the Hugging Face cache) when not given. ``remove()`` detaches the plugin."""

    def __init__(self, estimator, weights: str | Path | None = None) -> None:
        from safetensors.torch import load_file

        model = estimator.model if hasattr(estimator, "process_one_image") else estimator

        if weights is None:
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(WEIGHTS_REPO, WEIGHTS_FILE)
        head_pose = model.head_pose
        self.device = next(model.parameters()).device
        state = load_file(str(weights), device="cpu")
        state = {k[len("variance_head."):]: v for k, v in state.items() if k.startswith("variance_head.")}
        num_scale_comps = int(head_pose.num_scale_comps)
        self.head = VarianceHead(token_dim=state["fc1.weight"].shape[1] - (45 + 3 + 133 + num_scale_comps),
                                 num_scale_comps=num_scale_comps, hidden_dim=state["fc1.weight"].shape[0])
        self.head.load_state_dict(state)
        self.head.to(self.device).eval()
        rig = head_pose.mhr
        if not isinstance(rig, torch.jit.ScriptModule):           # the MHR Python package instead of the TorchScript rig
            rig = torch.jit.load(str(head_pose.model_data_dir), map_location=self.device)
        self.uncertainty = VertexUncertainty(rig, self.device)
        self._last = None
        self._hook = head_pose.register_forward_hook(self._capture)
        self._estimator = estimator if hasattr(estimator, "process_one_image") else None
        if self._estimator is not None:
            self._process = estimator.process_one_image
            self._signature = inspect.signature(self._process)
            self._warned = False
            estimator.process_one_image = self._process_one_image       # this estimator only, until remove()

    def _capture(self, module: torch.nn.Module, args: tuple, output: dict) -> None:
        self._last = (args[0], output)            # the pose token and the head's predictions; the last decoder layer wins

    def _process_one_image(self, *args, **kwargs) -> list[dict]:
        """Meta's ``process_one_image``, called as given; with the body decoder, "vertex_sigma_cm" (V,) for each person."""
        call = self._signature.bind(*args, **kwargs)
        call.apply_defaults()
        self._last = None
        outputs = self._process(*args, **kwargs)
        if call.arguments.get("inference_type") != "body":
            if not self._warned:
                warnings.warn('SAM 3D Body (CLUE) adds uncertainty to inference_type="body" calls only (the body decoder, '
                              'hands and face included); this call returns SAM 3D Body\'s outputs without it', stacklevel=2)
                self._warned = True
        elif outputs:
            sigma = self.sigma_cm()
            if len(sigma) != len(outputs):
                raise RuntimeError(f"{len(outputs)} people returned, the SAM 3D Body forward had {len(sigma)}")
            for person, s in zip(outputs, sigma):
                person["vertex_sigma_cm"] = s
        return outputs

    def remove(self) -> None:
        """Detach the plugin: SAM 3D Body (and the estimator's ``process_one_image``) as before."""
        self._hook.remove()
        if self._estimator is not None and self._estimator.__dict__.get("process_one_image") == self._process_one_image:
            del self._estimator.process_one_image

    @torch.no_grad()
    def variance_cm2(self) -> torch.Tensor:
        """(P, V, 3): x/y/z variance in cm^2 of every vertex of SAM 3D Body's meshes from its last forward."""
        from sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body

        if self._last is None:
            raise RuntimeError("run SAM 3D Body first (inference_type='body')")
        token, output = self._last
        # The head was trained on the body joint angles with the hand angles as predicted: SAM 3D Body's "body_pose"
        # output has them zeroed (its hand branch drives the mesh's hands), so they are rebuilt from the raw prediction.
        body_pose = compact_cont_to_model_params_body(output["pred_pose_raw"][:, 6:]).clone()
        body_pose[:, -3:] = 0                                              # the jaw, fixed as in SAM 3D Body
        mhr = {"shape": output["shape"], "global_rot": output["global_rot"], "body_pose": body_pose, "scale": output["scale"],
               "mhr_model_params": output["mhr_model_params"]}
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.uncertainty(mhr, self.head(token, mhr))

    def sigma_cm(self) -> np.ndarray:
        """(P, V): sigma_total = sqrt(var_x + var_y + var_z) in cm per vertex, root rotation / translation excluded, in the
        order of the people of SAM 3D Body's last call."""
        return self.variance_cm2().float().clamp_min(0).sum(-1).sqrt().cpu().numpy()
