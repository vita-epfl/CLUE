"""Closed-form uncertainty propagation for FLAME mesh parameters.

The :class:`CLUE` class takes a FLAME model instance and analytically propagates
diagonal parameter variances to per-vertex XYZ variance. It separates the
blendshape contribution from the pose contribution and returns both so callers
can render or benchmark them independently.
"""

from __future__ import annotations

import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

__all__ = ["CLUE"]


class CLUE:
    """Closed-form linear uncertainty estimator for FLAME."""

    def __init__(self, flame) -> None:
        self.flame = flame

        self.template_vertices = self.flame.v_template.unsqueeze(0)
        self.shapedirs = self.flame.shapedirs
        self.shapedirs_outer = self._outer6(self.shapedirs.transpose(1, 2)).transpose(1, 2).contiguous()
        self.j_regressor = self.flame.J_regressor
        self.lbs_weights = self.flame.lbs_weights
        self.lbs_path_weights = self._build_path_weights()
        self.parent_indices = self.flame.parents.clamp_min(0)
        self.root_mask = self.flame.parents < 0

    @staticmethod
    def _outer6(vectors: torch.Tensor) -> torch.Tensor:
        """(..., 3) vectors -> (..., 6) unique entries of v v^T, ordered
        [xx, yy, zz, xy, xz, yz]."""
        x, y, z = vectors.unbind(dim=-1)
        return torch.stack([x * x, y * y, z * z, x * y, x * z, y * z], dim=-1)

    @staticmethod
    def _left_jacobian_so3(omega: torch.Tensor) -> torch.Tensor:
        """Left Jacobian J_l of SO(3) at the axis-angle vectors ``omega``
        (..., 3) -> (..., 3, 3): exp((omega + eps)^) = exp((J_l eps)^) exp(omega^)
        to first order, so a Gaussian on the axis-angle coordinates with
        covariance diag(var) is a rotation perturbation with covariance
        J_l diag(var) J_l^T. J_l = I + (1 - cos t)/t^2 [omega]x
        + (t - sin t)/t^3 [omega]x^2 with t = |omega|; the series is used
        below t = 1e-3 (the closed form loses digits to cancellation there). J_l omega = omega: the Jacobian only acts on the
        components orthogonal to the rotation axis, and J_l -> I as t -> 0."""
        t = omega.norm(dim=-1, keepdim=True)
        t2 = t.square()
        small = t < 1e-3
        safe_t = torch.where(small, torch.ones_like(t), t)
        a = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(safe_t)) / safe_t.square())
        b = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (safe_t - torch.sin(safe_t)) / safe_t.pow(3))
        zero = torch.zeros_like(t[..., 0])
        wx, wy, wz = omega.unbind(-1)
        skew = torch.stack(
            [torch.stack([zero, -wz, wy], dim=-1),
             torch.stack([wz, zero, -wx], dim=-1),
             torch.stack([-wy, wx, zero], dim=-1)],
            dim=-2,
        )
        eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(skew.shape)
        return eye + a[..., None] * skew + b[..., None] * (skew @ skew)

    @staticmethod
    def _transformed_diagonal(linear: torch.Tensor, cov6: torch.Tensor) -> torch.Tensor:
        """Diagonal of ``E C E^T`` for linear maps ``E`` (..., 3, 3) and
        symmetric covariances ``C`` given as (..., 6) = [c00, c11, c22, c01,
        c02, c12] (the layout of :meth:`_outer6`)."""
        e0, e1, e2 = linear.unbind(dim=-1)  # columns of E, each (..., 3)
        c00, c11, c22, c01, c02, c12 = (c.unsqueeze(-1) for c in cov6.unbind(dim=-1))
        return (
            e0.square() * c00 + e1.square() * c11 + e2.square() * c22
            + 2.0 * (e0 * e1 * c01 + e0 * e2 * c02 + e1 * e2 * c12)
        )

    def _build_path_weights(self) -> torch.Tensor:
        """Precompute ancestor-aware LBS weights for each joint path."""
        parents = self.flame.parents
        num_joints = len(parents)
        ancestry = torch.zeros(
            num_joints,
            num_joints,
            dtype=self.lbs_weights.dtype,
            device=self.lbs_weights.device,
        )

        for joint_idx in range(num_joints):
            ancestor_idx = joint_idx
            while ancestor_idx >= 0:
                ancestry[joint_idx, ancestor_idx] = 1.0
                ancestor_idx = parents[ancestor_idx].item()

        # Blend lever arms with linear LBS weights first, then square.
        return self.lbs_weights[:, :, None] * ancestry[None, :, :]

    def _default_param_tensor(
        self,
        batch_size: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, width, dtype=dtype, device=device)

    def _expand_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast FLAME default tensors across the requested batch size."""
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)
        raise ValueError(
            f"Expected batch dimension 1 or {batch_size}, got {tensor.shape[0]}."
        )

    def _build_full_pose(
        self,
        batch_size: int,
        pose_params: torch.Tensor,
        neck_pose: torch.Tensor | None,
        eye_pose: torch.Tensor | None,
    ) -> torch.Tensor:
        neck = neck_pose if neck_pose is not None else self.flame.neck_pose
        eyes = eye_pose if eye_pose is not None else self.flame.eye_pose
        neck = self._expand_batch(neck, batch_size)
        eyes = self._expand_batch(eyes, batch_size)
        return torch.cat([pose_params[:, :3], neck, pose_params[:, 3:], eyes], dim=1)

    @torch.no_grad()
    def compute_variance(
        self,
        beta_vars: torch.Tensor,
        pose_vars: torch.Tensor | None = None,
        shape_params: torch.Tensor | None = None,
        expression_params: torch.Tensor | None = None,
        pose_params: torch.Tensor | None = None,
        neck_pose: torch.Tensor | None = None,
        eye_pose: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate FLAME parameter variances to per-vertex XYZ variance.

        Parameters
        ----------
        beta_vars:
            Tensor of shape ``(batch, num_beta_blendweights)``.
        pose_vars:
            Tensor of shape ``(batch, 15)`` in full-pose order
            ``[global(3) | neck(3) | jaw(3) | eye(6)]``: per joint, the
            variances of its three axis-angle coordinates (a Gaussian on the
            axis-angle vector, in the parent joint's axes); the left Jacobian
            of SO(3) at the mean pose maps them to the covariance of the
            rotation perturbation. When omitted, the pose contribution is
            assumed to be zero.
        transl:
            Accepted for API compatibility with FLAME call sites. Translation
            does not affect the propagated variance and is intentionally unused.

        Returns
        -------
        total_variance, pose_variance:
            Both shaped ``(batch, num_vertices, 3)``.
        """
        del transl

        batch_size = beta_vars.shape[0]
        device = beta_vars.device
        dtype = beta_vars.dtype

        if shape_params is None:
            shape_params = self._default_param_tensor(
                batch_size,
                self.flame.shape_params,
                device=device,
                dtype=dtype,
            )
        if expression_params is None:
            expression_params = self._default_param_tensor(
                batch_size,
                self.flame.expression_params,
                device=device,
                dtype=dtype,
            )
        if pose_params is None:
            pose_params = self._default_param_tensor(
                batch_size,
                self.flame.pose_params,
                device=device,
                dtype=dtype,
            )

        # Unposed 3x3 covariance of each vertex, as its 6 unique entries.
        shape_and_expression_covariance = torch.einsum(
            "bl,mkl->bmk",
            beta_vars,
            self.shapedirs_outer,
        )

        betas = torch.cat(
            [
                shape_params,
                self._expand_batch(self.flame.shape_betas, batch_size),
                expression_params,
                self._expand_batch(self.flame.expression_betas, batch_size),
            ],
            dim=1,
        )
        full_pose = self._build_full_pose(batch_size, pose_params, neck_pose, eye_pose)

        if pose_vars is None:
            pose_vars = self._default_param_tensor(
                batch_size,
                full_pose.shape[1],
                device=device,
                dtype=dtype,
            )

        template_vertices = self.template_vertices.expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)
        joints = vertices2joints(self.j_regressor, v_shaped)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        _, joint_transforms = batch_rigid_transform(
            rot_mats,
            joints,
            self.flame.parents,
            dtype=dtype,
        )

        rotation = joint_transforms[:, :, :3, :3]
        translation = joint_transforms[:, :, :3, 3]

        posed_joint_centers = (
            torch.einsum("bjpq,bjq->bjp", rotation, joints) + translation
        )

        transformed_vertices = (
            torch.einsum("bjpq,bvq->bvjp", rotation, v_shaped) + translation[:, None]
        )

        pose_vars = pose_vars.view(batch_size, -1, 3)
        lever_arms = (
            transformed_vertices[:, :, :, None, :]
            - posed_joint_centers[:, None, None, :, :]
        )
        blended_lever_arms = torch.einsum(
            "vkj,bvkjd->bvjd",
            self.lbs_path_weights,
            lever_arms,
        )

        # The pose variances are those of joint a's axis-angle coordinates in
        # the parent's axes. The left Jacobian J_l(omega_a) turns them into the
        # covariance of the rotation perturbation composed with the mean
        # rotation, and the parent's rotation R_p(a) takes that into world
        # axes: Cov = R_p(a) J_l diag(var) J_l^T R_p(a)^T.
        identity = torch.eye(3, dtype=dtype, device=device)
        parent_rotation = torch.where(
            self.root_mask[None, :, None, None],
            identity,
            rotation[:, self.parent_indices],
        )
        left = parent_rotation @ self._left_jacobian_so3(full_pose.view(batch_size, -1, 3))
        pose_covariance = torch.einsum(
            "bjpk,bjk,bjqk->bjpq",
            left,
            pose_vars,
            left,
        )

        x_comp = blended_lever_arms[..., 0]
        y_comp = blended_lever_arms[..., 1]
        z_comp = blended_lever_arms[..., 2]
        var_x = pose_covariance[:, None, :, 0, 0]
        var_y = pose_covariance[:, None, :, 1, 1]
        var_z = pose_covariance[:, None, :, 2, 2]
        cov_xy = pose_covariance[:, None, :, 0, 1]
        cov_xz = pose_covariance[:, None, :, 0, 2]
        cov_yz = pose_covariance[:, None, :, 1, 2]

        # Diagonal of [l]x Sigma [l]x^T.
        pose_variance = torch.stack(
            [
                (var_y * z_comp.square() + var_z * y_comp.square()
                 - 2.0 * cov_yz * y_comp * z_comp).sum(dim=-1),
                (var_x * z_comp.square() + var_z * x_comp.square()
                 - 2.0 * cov_xz * x_comp * z_comp).sum(dim=-1),
                (var_x * y_comp.square() + var_y * x_comp.square()
                 - 2.0 * cov_xy * x_comp * y_comp).sum(dim=-1),
            ],
            dim=-1,
        )

        effective_rotation = torch.einsum(
            "vj,bjpk->bvpk",
            self.lbs_weights,
            rotation,
        )
        # Diagonal of R_eff C R_eff^T.
        shape_and_expression_variance = self._transformed_diagonal(
            effective_rotation,
            shape_and_expression_covariance,
        )

        total_variance = shape_and_expression_variance + pose_variance
        return total_variance, pose_variance
