"""Lightweight MHR model loading for the CLUE experiments.

The interactive and batch CLUE experiments should not require the full MHR
Python package at runtime. The released TorchScript asset contains the LOD 1
model plus the constants needed by :mod:`algorithm`, so this module exposes a
small adapter around ``assets/mhr_model.pt``. If a non-LOD1 model is requested,
it falls back to the upstream MHR package when available.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
MHR_MODEL_PARAMETER_NAMES: tuple[str, ...] = (
    "root_tx",
    "root_ty",
    "root_tz",
    "root_rx",
    "root_ry",
    "root_rz",
    "spine0_rx_flexible",
    "spine_twist0",
    "spine0_ry_flexible",
    "spine_lean0",
    "spine0_rz_flexible",
    "spine_bend0",
    "spine1_rx_flexible",
    "spine_twist1",
    "spine1_ry_flexible",
    "spine_lean1",
    "spine1_rz_flexible",
    "spine_bend1",
    "spine2_rx_flexible",
    "spine2_ry_flexible",
    "spine2_rz_flexible",
    "spine3_rx_flexible",
    "spine3_ry_flexible",
    "spine3_rz_flexible",
    "neck_twist",
    "neck_lean",
    "neck_bend",
    "head_twist",
    "head_lean",
    "head_bend",
    "r_clavicle_rx",
    "r_clavicle_ry",
    "r_clavicle_rz",
    "r_uparm_twist",
    "r_uparm_ry",
    "r_uparm_rz",
    "r_elbow_bend",
    "r_lowarm_twist",
    "r_wrist_ry",
    "r_wrist_rz",
    "l_clavicle_rx",
    "l_clavicle_ry",
    "l_clavicle_rz",
    "l_uparm_twist",
    "l_uparm_ry",
    "l_uparm_rz",
    "l_elbow_bend",
    "l_lowarm_twist",
    "l_wrist_ry",
    "l_wrist_rz",
    "r_upleg_twist",
    "r_upleg_ry",
    "r_upleg_rz",
    "r_knee_bend",
    "r_lowleg_twist",
    "r_foot_bend",
    "r_foot_lean0",
    "r_foot_lean1",
    "r_ball_bend",
    "l_upleg_twist",
    "l_upleg_ry",
    "l_upleg_rz",
    "l_knee_bend",
    "l_lowleg_twist",
    "l_foot_bend",
    "l_foot_lean0",
    "l_foot_lean1",
    "l_ball_bend",
    "r_thumb0_ry",
    "r_thumb0_rz",
    "r_thumb1_rx",
    "r_thumb1_ry",
    "r_thumb1_rz",
    "r_thumb2_rz",
    "r_thumb3_rz",
    "r_index1_ry",
    "r_ring1_ry",
    "r_pinky1_ry",
    "r_middle1_ry",
    "r_index1_rz",
    "r_index2_rz",
    "r_index3_rz",
    "r_middle1_rz",
    "r_middle2_rz",
    "r_middle3_rz",
    "r_ring1_rz",
    "r_ring2_rz",
    "r_ring3_rz",
    "r_pinky1_rz",
    "r_pinky2_rz",
    "r_pinky3_rz",
    "r_index1_rx",
    "r_ring1_rx",
    "r_pinky1_rx",
    "r_middle1_rx",
    "l_thumb0_ry",
    "l_thumb0_rz",
    "l_thumb1_rx",
    "l_thumb1_ry",
    "l_thumb1_rz",
    "l_thumb2_rz",
    "l_thumb3_rz",
    "l_index1_ry",
    "l_ring1_ry",
    "l_pinky1_ry",
    "l_middle1_ry",
    "l_index1_rz",
    "l_index2_rz",
    "l_index3_rz",
    "l_middle1_rz",
    "l_middle2_rz",
    "l_middle3_rz",
    "l_ring1_rz",
    "l_ring2_rz",
    "l_ring3_rz",
    "l_pinky1_rz",
    "l_pinky2_rz",
    "l_pinky3_rz",
    "l_index1_rx",
    "l_ring1_rx",
    "l_pinky1_rx",
    "l_middle1_rx",
    "l_foot_ry_flexible",
    "l_subtalar_rz_flexible",
    "l_talocrural_rx_flexible",
    "l_ball_rx_flexible",
    "r_foot_ry_flexible",
    "r_subtalar_rz_flexible",
    "r_talocrural_rx_flexible",
    "r_ball_rx_flexible",
    "spine_length_flexible",
    "neck_length_flexible",
    "shoulder_width_flexible",
    "arm_length_flexible",
    "hip_width_flexible",
    "leg_length_flexible",
    "scale_eye_width",
    "scale_eye_height",
    "scale_eye_depth",
    "scale_spine_length",
    "scale_neck_length",
    "scale_shoulder_width",
    "scale_uparms",
    "scale_lowarms",
    "scale_r_hands",
    "scale_l_hands",
    "scale_hip_width",
    "scale_hip_height",
    "scale_hip_depth",
    "scale_uplegs",
    "scale_lowlegs",
    "scale_knee_knock",
    "scale_ankle_height",
    "scale_foot_length",
    "scale_r_index1_length",
    "scale_r_middle1_length",
    "scale_r_ring1_length",
    "scale_r_pinky1_length",
    "scale_r_thumb1_length",
    "scale_r_index1_offset",
    "scale_r_middle1_offset",
    "scale_r_ring1_offset",
    "scale_r_pinky1_offset",
    "scale_r_thumb1_offset",
    "scale_r_index2_length",
    "scale_r_middle2_length",
    "scale_r_ring2_length",
    "scale_r_pinky2_length",
    "scale_r_thumb2_length",
    "scale_r_index3_length",
    "scale_r_middle3_length",
    "scale_r_ring3_length",
    "scale_r_pinky3_length",
    "scale_r_thumb3_length",
    "scale_r_index_null_tx",
    "scale_r_middle_null_tx",
    "scale_r_ring_null_tx",
    "scale_r_pinky_null_tx",
    "scale_r_thumb_null_tx",
    "scale_l_index1_length",
    "scale_l_middle1_length",
    "scale_l_ring1_length",
    "scale_l_pinky1_length",
    "scale_l_thumb1_length",
    "scale_l_index1_offset",
    "scale_l_middle1_offset",
    "scale_l_ring1_offset",
    "scale_l_pinky1_offset",
    "scale_l_thumb1_offset",
    "scale_l_index2_length",
    "scale_l_middle2_length",
    "scale_l_ring2_length",
    "scale_l_pinky2_length",
    "scale_l_thumb2_length",
    "scale_l_index3_length",
    "scale_l_middle3_length",
    "scale_l_ring3_length",
    "scale_l_pinky3_length",
    "scale_l_thumb3_length",
    "scale_l_index_null_tx",
    "scale_l_middle_null_tx",
    "scale_l_ring_null_tx",
    "scale_l_pinky_null_tx",
    "scale_l_thumb_null_tx",
)
assert len(MHR_MODEL_PARAMETER_NAMES) == 204


def get_default_asset_folder() -> Path:
    return Path(__file__).parent / "assets"


class _SkinWeightsAdapter:
    def __init__(
        self,
        num_vertices: int,
        skin_indices: torch.Tensor,
        skin_weights: torch.Tensor,
        vertex_indices: torch.Tensor,
    ) -> None:
        self.num_vertices = int(num_vertices)
        self._skin_indices = skin_indices.detach().cpu().long()
        self._skin_weights = skin_weights.detach().cpu().float()
        self._vertex_indices = vertex_indices.detach().cpu().long()

    def to_dense(self, num_joints: int) -> torch.Tensor:
        dense = torch.zeros(self.num_vertices, int(num_joints), dtype=torch.float32)
        dense[self._vertex_indices, self._skin_indices] = self._skin_weights
        return dense.numpy()


class _CombinedBlendShape:
    def __init__(
        self,
        base_shape: torch.Tensor,
        identity_vectors: torch.Tensor,
        expression_vectors: torch.Tensor,
    ) -> None:
        self.base_shape = base_shape
        self.shape_vectors = torch.cat([identity_vectors, expression_vectors], dim=0)

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        identity = coeffs[:, :NUM_IDENTITY_BLENDSHAPES]
        expression = coeffs[
            :,
            NUM_IDENTITY_BLENDSHAPES : NUM_IDENTITY_BLENDSHAPES
            + NUM_FACE_EXPRESSION_BLENDSHAPES,
        ]
        vertices = self.base_shape.unsqueeze(0).expand(coeffs.shape[0], -1, -1).clone()
        vertices = vertices + torch.einsum(
            "bn,nvd->bvd",
            identity,
            self.shape_vectors[:NUM_IDENTITY_BLENDSHAPES],
        )
        vertices = vertices + torch.einsum(
            "bn,nvd->bvd",
            expression,
            self.shape_vectors[NUM_IDENTITY_BLENDSHAPES:],
        )
        return vertices


class _TorchCharacterAdapter:
    def __init__(self, scripted_model: torch.jit.ScriptModule, state: dict[str, torch.Tensor]) -> None:
        self._scripted_character = scripted_model.character_torch
        self._parameter_transform_width = int(
            state["character_torch.parameter_transform.parameter_transform"].shape[1]
        )
        self.skeleton = SimpleNamespace(
            joint_parents=state["character_torch.skeleton.joint_parents"],
            joint_prerotations=state["character_torch.skeleton.joint_prerotations"],
        )
        self.mesh = SimpleNamespace(
            rest_vertices=state["character_torch.mesh.rest_vertices"],
            faces=state["character_torch.mesh.faces"],
        )
        self.parameter_transform = SimpleNamespace(
            parameter_transform=state[
                "character_torch.parameter_transform.parameter_transform"
            ]
        )
        self.linear_blend_skinning = SimpleNamespace(
            inverse_bind_pose=state[
                "character_torch.linear_blend_skinning.inverse_bind_pose"
            ]
        )
        self.blend_shape = _CombinedBlendShape(
            state["character_torch.blend_shape.base_shape"],
            state["character_torch.blend_shape.shape_vectors"],
            state["face_expressions_model.shape_vectors"],
        )

    def model_parameters_to_joint_parameters(self, model_parameters: torch.Tensor) -> torch.Tensor:
        return self._scripted_character.model_parameters_to_joint_parameters(
            model_parameters[:, : self._parameter_transform_width]
        )

    def joint_parameters_to_skeleton_state(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        return self._scripted_character.joint_parameters_to_skeleton_state(joint_parameters)


class TorchScriptMHR:
    """Adapter exposing the subset of MHR used by the CLUE experiments."""

    def __init__(
        self,
        scripted_model: torch.jit.ScriptModule,
        device: torch.device,
        wants_pose_correctives: bool,
    ) -> None:
        self.scripted_model = scripted_model.to(device)
        self.scripted_model.eval()
        self.wants_pose_correctives = wants_pose_correctives
        state = {
            key: value.to(device) if value.is_floating_point() else value.to(device)
            for key, value in self.scripted_model.state_dict().items()
        }

        faces = state["character_torch.mesh.faces"].detach().cpu().numpy()
        n_vertices = int(state["character_torch.mesh.rest_vertices"].shape[0])
        n_joints = int(state["character_torch.skeleton.joint_parents"].numel())
        self.character_torch = _TorchCharacterAdapter(self.scripted_model, state)
        self.character = SimpleNamespace(
            mesh=SimpleNamespace(
                faces=faces,
                n_vertices=n_vertices,
            ),
            skeleton=SimpleNamespace(
                size=n_joints,
                joint_parents=state[
                    "character_torch.skeleton.joint_parents"
                ].detach().cpu().tolist(),
                joint_names=[f"joint_{idx}" for idx in range(n_joints)],
            ),
            parameter_transform=SimpleNamespace(
                size=204 + NUM_IDENTITY_BLENDSHAPES + NUM_FACE_EXPRESSION_BLENDSHAPES,
                names=list(MHR_MODEL_PARAMETER_NAMES)
                + [f"identity[{idx}]" for idx in range(NUM_IDENTITY_BLENDSHAPES)]
                + [
                    f"expression[{idx}]"
                    for idx in range(NUM_FACE_EXPRESSION_BLENDSHAPES)
                ],
            ),
            skin_weights=_SkinWeightsAdapter(
                n_vertices,
                state["character_torch.linear_blend_skinning.skin_indices_flattened"],
                state["character_torch.linear_blend_skinning.skin_weights_flattened"],
                state["character_torch.linear_blend_skinning.vert_indices_flattened"],
            ),
        )
        self.pose_correctives_model = (
            self.scripted_model.pose_correctives_model if wants_pose_correctives else None
        )

    @staticmethod
    def from_files(
        folder: Path = get_default_asset_folder(),
        device: torch.device | str = "cpu",
        lod: int = 1,
        wants_pose_correctives: bool = True,
    ) -> "TorchScriptMHR":
        if lod != 1:
            raise ValueError(
                "The lightweight TorchScript MHR loader supports LOD 1 only. "
                "Use the upstream MHR package with PyMomentum for other LODs."
            )
        model_path = Path(folder) / "mhr_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"MHR TorchScript model not found: {model_path}")
        torch_device = torch.device(device)
        scripted_model = torch.jit.load(str(model_path), map_location=torch_device)
        return TorchScriptMHR(scripted_model, torch_device, wants_pose_correctives)

    def eval(self) -> "TorchScriptMHR":
        self.scripted_model.eval()
        return self

    def __call__(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None,
        apply_correctives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if face_expr_coeffs is None:
            face_expr_coeffs = torch.zeros(
                model_parameters.shape[0],
                NUM_FACE_EXPRESSION_BLENDSHAPES,
                dtype=model_parameters.dtype,
                device=model_parameters.device,
            )
        return self.scripted_model(
            identity_coeffs,
            model_parameters,
            face_expr_coeffs,
            bool(apply_correctives and self.wants_pose_correctives),
        )


def load_mhr_model(
    folder: str | Path = get_default_asset_folder(),
    device: torch.device | str = "cpu",
    lod: int = 1,
    wants_pose_correctives: bool = True,
):
    """Load MHR for CLUE experiments.

    LOD 1 uses the TorchScript adapter and therefore works without PyMomentum.
    Other LODs use the upstream MHR package when it is installed.
    """
    folder = Path(folder)
    if lod == 1:
        return TorchScriptMHR.from_files(
            folder=folder,
            device=device,
            lod=lod,
            wants_pose_correctives=wants_pose_correctives,
        )

    try:
        from mhr.mhr import MHR
    except Exception as exc:
        raise RuntimeError(
            "Non-LOD1 MHR loading requires the upstream MHR package and "
            "PyMomentum; without them, use --lod 1."
        ) from exc
    return MHR.from_files(
        folder=folder,
        device=torch.device(device),
        lod=lod,
        wants_pose_correctives=wants_pose_correctives,
    )
