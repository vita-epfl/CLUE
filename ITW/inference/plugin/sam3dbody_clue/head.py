"""CLUE's uncertainty for the MHR body.

``VarianceHead`` predicts, per person, the variance of the MHR parameters from the decoder's pose token and the predicted
parameters. ``VertexUncertainty`` propagates the pose variances through the MHR rig in closed form (CLUE for MHR,
``clue_mhr/``) to a variance per vertex of the predicted mesh, as posed; root rotation and translation carry no variance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clue_mhr import algorithm, model_loader

NUM_MODEL_PARAMS = 204                  # MHR model parameters: [0:3] root translation, [3:6] root rotation,
BODY_PARAMS = slice(6, 136)             # [6:136] body joint angles + lengths, [136:204] scale


class VarianceHead(nn.Module):
    """A hidden layer on [pose token, shape, root rotation, body pose, scale], then a sigmoid-bounded variance per parameter
    times one softplus scale per parameter group."""

    GROUPS = ("shape", "root_rot", "body_pose", "scale_comps", "cam_t")

    def __init__(self, token_dim: int, num_scale_comps: int = 28, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.widths = {"shape": 45, "root_rot": 3, "body_pose": 133, "scale_comps": int(num_scale_comps), "cam_t": 3}
        inputs = int(token_dim) + 45 + 3 + 133 + int(num_scale_comps)
        self.fc1 = nn.Linear(inputs, hidden_dim)
        self.var_fc = nn.Linear(hidden_dim, sum(self.widths.values()))
        self.scale_fc = nn.Linear(hidden_dim, len(self.GROUPS))

    def forward(self, pose_token: torch.Tensor, mhr: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if pose_token.ndim == 3:
            pose_token = pose_token[:, 0]
        features = torch.cat([pose_token.float(), mhr["shape"].float(), mhr["global_rot"].float(), mhr["body_pose"].float(),
                              mhr["scale"].float()], dim=1)
        hidden = torch.relu(self.fc1(features))
        bounded = torch.sigmoid(self.var_fc(hidden))
        scales = F.softplus(self.scale_fc(hidden))
        variances, start = {}, 0
        for index, name in enumerate(self.GROUPS):
            width = self.widths[name]
            variances[name] = bounded[:, start:start + width] * scales[:, index:index + 1]
            start += width
        return variances


class VertexUncertainty:
    """Per-vertex x/y/z variance in cm^2 of the predicted mesh, from the variance head's output. Uses the SAM 3D Body head's
    own TorchScript MHR rig."""

    def __init__(self, rig: torch.jit.ScriptModule, device: torch.device) -> None:
        self.clue = algorithm.CLUE(model_loader.TorchScriptMHR(rig, torch.device(device), wants_pose_correctives=True))

    def __call__(self, mhr: dict[str, torch.Tensor], variances: dict[str, torch.Tensor]) -> torch.Tensor:
        var_body = variances["body_pose"][:, :130].float()       # joint angles and lengths; the jaw stays fixed
        var_params = var_body.new_zeros(var_body.shape[0], NUM_MODEL_PARAMS)
        var_params[:, BODY_PARAMS] = var_body
        variance, _ = self.clue.compute_variance_differentiable(
            variances["shape"].float(), var_params, None, identity_coeffs=mhr["shape"].float(),
            model_parameters=mhr["mhr_model_params"].float(), face_expr_coeffs=None, apply_correctives=True,
            vertex_indices=None, return_joint_variance=False, euler_variance_override=None)
        return variance
