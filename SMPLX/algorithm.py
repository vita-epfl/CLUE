"""Closed-form uncertainty propagation for SMPL-X mesh parameters (CLUE).

The :class:`CLUE` class takes a SMPL-X model instance and analytically
propagates diagonal shape, expression, and pose variances to per-vertex XYZ
variance. The pose variances of each joint are those of its axis-angle
coordinates in its parent's axes; the left Jacobian of SO(3) at the mean pose
maps them to the rotation perturbation, which is rotated into world axes and
applied to the lever arms of the ancestor joints in world coordinates;
the blendshape term keeps the full 3x3 covariance of each vertex's unposed offset
and returns the diagonal of that covariance after the vertex's effective rotation.

The evaluation exploits the two sparsities of the body: a vertex is skinned by
only a few of the 55 joints, and its lever-arm / path-sum terms are nonzero
only for the ancestors of those joints. Everything is computed on two
precomputed index sets, the ``nnz`` (vertex, joint, weight) skinning
influences and the unique (vertex, ancestor) pairs they reach, with sparse CSR
matmuls for the accumulations, so one call costs ``O(nnz + |pairs|)`` instead
of ``O(V * J^2)``. :meth:`compute_variance_differentiable` exposes the same
computation with gradients to the variance inputs.
"""

from __future__ import annotations

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

__all__ = ["CLUE"]


class CLUE:
    """Closed-form linear uncertainty estimator for SMPL-X (sparse evaluation)."""

    def __init__(self, model) -> None:
        self.model = model

        self.template_vertices = self.model.v_template.unsqueeze(0)
        self.shapedirs = torch.cat([self.model.shapedirs, self.model.expr_dirs], dim=-1)
        self.shapedirs_outer = self._outer6(self.shapedirs.transpose(1, 2)).transpose(1, 2).contiguous()
        self.j_regressor = self.model.J_regressor
        self.lbs_weights = self.model.lbs_weights
        self.ancestry = self._build_ancestry()
        self.parent_indices = self.model.parents.clamp_min(0)
        self.root_mask = self.model.parents < 0
        self._build_sparse_structures()

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

    def _build_ancestry(self) -> torch.Tensor:
        """Precompute the joint ancestry matrix used by pose propagation."""
        parents = self.model.parents
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

        return ancestry

    def _zeros(
        self,
        batch_size: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, width, dtype=dtype, device=device)

    def _expand_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast model defaults across the requested batch size."""
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f"Expected batch dimension 1 or {batch_size}, got {tensor.shape[0]}."
        )

    def _build_shape_components(
        self,
        batch_size: int,
        betas: torch.Tensor | None,
        expression: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = betas if betas is not None else self.model.betas
        expression_params = expression if expression is not None else self.model.expression
        return torch.cat(
            [
                self._expand_batch(shape, batch_size),
                self._expand_batch(expression_params, batch_size),
            ],
            dim=1,
        )

    def _build_full_pose(
        self,
        batch_size: int,
        global_orient: torch.Tensor | None,
        body_pose: torch.Tensor | None,
        left_hand_pose: torch.Tensor | None,
        right_hand_pose: torch.Tensor | None,
        jaw_pose: torch.Tensor | None,
        leye_pose: torch.Tensor | None,
        reye_pose: torch.Tensor | None,
    ) -> torch.Tensor:
        global_pose = self._expand_batch(
            global_orient if global_orient is not None else self.model.global_orient,
            batch_size,
        )
        body = self._expand_batch(
            body_pose if body_pose is not None else self.model.body_pose,
            batch_size,
        )
        left_hand = self._expand_batch(
            left_hand_pose if left_hand_pose is not None else self.model.left_hand_pose,
            batch_size,
        )
        right_hand = self._expand_batch(
            right_hand_pose if right_hand_pose is not None else self.model.right_hand_pose,
            batch_size,
        )
        jaw = self._expand_batch(
            jaw_pose if jaw_pose is not None else self.model.jaw_pose,
            batch_size,
        )
        leye = self._expand_batch(
            leye_pose if leye_pose is not None else self.model.leye_pose,
            batch_size,
        )
        reye = self._expand_batch(
            reye_pose if reye_pose is not None else self.model.reye_pose,
            batch_size,
        )

        if self.model.use_pca:
            left_hand = left_hand @ self.model.left_hand_components
            right_hand = right_hand @ self.model.right_hand_components

        full_pose = torch.cat(
            [
                global_pose.reshape(batch_size, 1, 3),
                body.reshape(batch_size, self.model.NUM_BODY_JOINTS, 3),
                jaw.reshape(batch_size, 1, 3),
                leye.reshape(batch_size, 1, 3),
                reye.reshape(batch_size, 1, 3),
                left_hand.reshape(batch_size, self.model.NUM_HAND_JOINTS, 3),
                right_hand.reshape(batch_size, self.model.NUM_HAND_JOINTS, 3),
            ],
            dim=1,
        ).reshape(batch_size, -1)

        pose_mean = self._expand_batch(self.model.pose_mean.reshape(1, -1), batch_size)
        return full_pose + pose_mean

    # ------------------------------------------------------------------ setup
    def _build_sparse_structures(self) -> None:
        device = self.lbs_weights.device
        dtype = self.lbs_weights.dtype
        weights = self.lbs_weights.detach().cpu().numpy()                      # (V, J)
        ancestry = self.ancestry.detach().cpu().numpy() > 0                    # (J, J): [k, a] a is ancestor-or-self of k
        num_vertices, num_joints = weights.shape

        infl_v, infl_j = np.nonzero(weights)
        infl_w = weights[infl_v, infl_j].astype(np.float64)
        nnz = infl_v.shape[0]

        anc_lists = [np.nonzero(ancestry[k])[0] for k in range(num_joints)]
        depth = np.array([len(a) for a in anc_lists])
        tri_infl = np.repeat(np.arange(nnz), depth[infl_j])
        tri_a = np.concatenate([anc_lists[k] for k in infl_j]) if nnz else np.zeros(0, np.int64)
        tri_v = infl_v[tri_infl]

        pair_key = tri_v.astype(np.int64) * num_joints + tri_a.astype(np.int64)
        uniq, tri_pair = np.unique(pair_key, return_inverse=True)
        pair_v = (uniq // num_joints).astype(np.int64)
        pair_a = (uniq % num_joints).astype(np.int64)
        num_pairs = uniq.shape[0]
        path_sum = np.zeros(num_pairs, np.float64)
        np.add.at(path_sum, tri_pair, infl_w[tri_infl])

        def csr(rows, cols, vals, shape):
            idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
            return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=dtype), shape).coalesce().to_sparse_csr().to(device)

        self.num_vertices, self.num_joints = int(num_vertices), int(num_joints)
        self.infl_v = torch.tensor(infl_v, dtype=torch.long, device=device)
        self.infl_j = torch.tensor(infl_j, dtype=torch.long, device=device)
        self.pair_v = torch.tensor(pair_v, dtype=torch.long, device=device)
        self.pair_a = torch.tensor(pair_a, dtype=torch.long, device=device)
        self.pair_path_sum = torch.tensor(path_sum, dtype=dtype, device=device)
        self.lever_matrix = csr(tri_pair, tri_infl, infl_w[tri_infl], (num_pairs, nnz))          # S[p, e] = w_e
        self.pair_to_vertex = csr(pair_v, np.arange(num_pairs), np.ones(num_pairs), (num_vertices, num_pairs))
        self.infl_to_vertex = csr(infl_v, np.arange(nnz), infl_w, (num_vertices, nnz))
        self._sparse_nnz = {"lever": int(tri_infl.shape[0]), "pair_to_vertex": int(num_pairs), "infl_to_vertex": int(nnz)}
        self.stats = {
            "num_vertices": int(num_vertices), "num_joints": int(num_joints), "nnz_weights": int(nnz),
            "num_triples": int(tri_infl.shape[0]), "num_pairs": int(num_pairs),
            "mean_influences_per_vertex": float(nnz / num_vertices),
            "mean_ancestors_per_vertex": float(num_pairs / num_vertices),
        }

    @staticmethod
    def _accumulate(matrix: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """``source`` (B, N, *rest) -> (B, M, *rest) = matrix (M x N, sparse CSR) applied
        along the second axis, with the batch and trailing dims folded into the
        dense columns of one sparse-dense matmul."""
        batch, n = source.shape[0], source.shape[1]
        rest = source.shape[2:]
        flat = source.permute(1, 0, *range(2, source.dim())).reshape(n, -1)
        out = torch.sparse.mm(matrix, flat)
        return out.reshape(matrix.shape[0], batch, *rest).permute(1, 0, *range(2, source.dim()))

    def sparse_matmul_flops(self, batch: int = 1) -> int:
        """FLOPs of the sparse-dense matmuls of one call (2 per stored nonzero per
        dense column), which dispatcher-level FLOP counters do not see."""
        return int(2 * (self._sparse_nnz["lever"] * 3 * batch + self._sparse_nnz["pair_to_vertex"] * 3 * batch
                        + self._sparse_nnz["infl_to_vertex"] * 9 * batch))

    # -------------------------------------------------------------- main entry
    def compute_variance_differentiable(
        self,
        beta_vars: torch.Tensor,
        pose_vars: torch.Tensor | None = None,
        betas: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        left_hand_pose: torch.Tensor | None = None,
        right_hand_pose: torch.Tensor | None = None,
        jaw_pose: torch.Tensor | None = None,
        leye_pose: torch.Tensor | None = None,
        reye_pose: torch.Tensor | None = None,
        *,
        vertex_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate SMPL-X parameter variances to per-vertex XYZ variance.

        Gradients flow from the returned variances back to the variance
        inputs; :meth:`compute_variance` is the same call under ``no_grad``.

        Parameters
        ----------
        beta_vars:
            Tensor of shape ``(batch, num_shape_betas + num_expression_betas)``.
        pose_vars:
            Tensor of shape ``(batch, 165)`` in full-pose order
            ``[global | body | jaw | leye | reye | left_hand | right_hand]``:
            per joint, the variances of its three axis-angle coordinates (a
            Gaussian on the axis-angle vector, in the parent joint's axes);
            the left Jacobian of SO(3) at the mean pose maps them to the
            covariance of the rotation perturbation. When omitted, the pose
            contribution is assumed to be zero.
        betas, expression, global_orient, body_pose, left_hand_pose,
        right_hand_pose, jaw_pose, leye_pose, reye_pose:
            Mean parameters of the linearization point; the model's own
            values when omitted.
        vertex_indices:
            Optional ``(num_subset,)`` long tensor restricting the returned
            tensors to a subset of the vertices.

        Returns
        -------
        total_variance, pose_variance:
            Both shaped ``(batch, num_vertices, 3)`` (``num_subset`` with
            ``vertex_indices``).
        """
        batch_size = beta_vars.shape[0]
        device, dtype = beta_vars.device, beta_vars.dtype

        shape_components = self._build_shape_components(batch_size, betas, expression)
        full_pose = self._build_full_pose(batch_size, global_orient, body_pose, left_hand_pose, right_hand_pose,
                                          jaw_pose, leye_pose, reye_pose)
        if pose_vars is None:
            pose_vars = self._zeros(batch_size, full_pose.shape[1], device=device, dtype=dtype)
        else:
            pose_vars = self._expand_batch(pose_vars, batch_size)

        # ---- blendshape covariance in the unposed frame, 6 unique entries per vertex
        shape_and_expression_covariance = torch.einsum("bl,mkl->bmk", beta_vars, self.shapedirs_outer)    # (B, V, 6)

        # ---- mean pose: v_shaped, global joint rotations, rigid-transform translations, posed joints
        template_vertices = self.template_vertices.expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(shape_components, self.shapedirs)
        joints = vertices2joints(self.j_regressor, v_shaped)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        _, joint_transforms = batch_rigid_transform(rot_mats, joints, self.model.parents, dtype=dtype)
        rotation = joint_transforms[:, :, :3, :3]                                                      # (B, J, 3, 3)
        translation = joint_transforms[:, :, :3, 3]                                                    # (B, J, 3)
        posed_joint_centers = torch.einsum("bjpq,bjq->bjp", rotation, joints) + translation            # (B, J, 3)

        # ---- per-influence rigidly skinned positions, only for the nnz (vertex, joint) pairs
        rot_e = rotation[:, self.infl_j]                                                               # (B, nnz, 3, 3)
        skinned = torch.einsum("bepq,beq->bep", rot_e, v_shaped[:, self.infl_v]) + translation[:, self.infl_j]

        # ---- per-joint pose covariance in world axes: the variances are those of the joint's
        #      axis-angle coordinates in the parent's axes; the left Jacobian J_l(omega) maps them to
        #      the rotation perturbation and R_p to world axes, Cov = R_p J_l diag(var) J_l^T R_p^T;
        #      only its 6 unique entries are gathered per pair
        identity = torch.eye(3, dtype=dtype, device=device)
        parent_rotation = torch.where(self.root_mask[None, :, None, None], identity, rotation[:, self.parent_indices])
        left = parent_rotation @ self._left_jacobian_so3(full_pose.view(batch_size, -1, 3))
        cov = torch.einsum("bjpk,bjk,bjqk->bjpq", left, pose_vars.view(batch_size, -1, 3), left)
        cov6 = torch.stack([cov[..., 0, 0], cov[..., 1, 1], cov[..., 2, 2],
                            cov[..., 0, 1], cov[..., 0, 2], cov[..., 1, 2]], dim=-1)               # (B, J, 6)

        # ---- lever arms of the (vertex, ancestor) pairs and the diagonal of [l]x Sigma [l]x^T
        lever = self._accumulate(self.lever_matrix, skinned) - self.pair_path_sum[None, :, None] * posed_joint_centers[:, self.pair_a]
        x, y, z = lever.unbind(-1)
        x2, y2, z2 = lever.square().unbind(-1)
        var_x, var_y, var_z, cov_xy, cov_xz, cov_yz = cov6[:, self.pair_a].unbind(-1)                # (B, pairs) each
        per_pair = torch.stack([var_y * z2 + var_z * y2 - 2.0 * cov_yz * y * z,
                                var_x * z2 + var_z * x2 - 2.0 * cov_xz * x * z,
                                var_x * y2 + var_y * x2 - 2.0 * cov_xy * x * y], dim=-1)
        pose_variance = self._accumulate(self.pair_to_vertex, per_pair)                                # (B, V, 3)

        # ---- blendshape term: diagonal of R_eff C R_eff^T, R_eff = sum_k w_vk R_k the effective rotation
        effective_rotation = self._accumulate(self.infl_to_vertex, rot_e)                              # (B, V, 3, 3)
        shape_and_expression_variance = self._transformed_diagonal(effective_rotation, shape_and_expression_covariance)

        total_variance = shape_and_expression_variance + pose_variance
        if vertex_indices is not None:
            vertex_indices = torch.as_tensor(vertex_indices, dtype=torch.long, device=device)
            total_variance = total_variance.index_select(1, vertex_indices)
            pose_variance = pose_variance.index_select(1, vertex_indices)
        return total_variance, pose_variance

    @torch.no_grad()
    def compute_variance(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`compute_variance_differentiable` under ``no_grad``."""
        return self.compute_variance_differentiable(*args, **kwargs)
