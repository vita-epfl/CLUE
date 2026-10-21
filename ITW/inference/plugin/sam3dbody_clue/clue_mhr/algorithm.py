"""Closed-form uncertainty propagation for MHR mesh parameters (CLUE).

The :class:`CLUE` class propagates diagonal uncertainty from MHR identity,
compact model, and facial-expression parameters to per-vertex XYZ variance.
The implementation follows the pseudo-code, with these MHR-specific details:

* the compact 204-D model-parameter variance is propagated every call through
  the cached squared parameter-transform matrix;
* MHR stores joint scale as a log2-scale parameter, so scale variance uses the
  chain-rule factor ``log(2) ** 2``;
* the blendshape term keeps the full 3x3 covariance of each vertex's unposed
  offset (a blendshape moves a vertex along one direction, so its x/y/z
  components are correlated) and returns the diagonal of that covariance
  after the blended skinning transform — exact per axis at any pose.

The evaluation exploits the two sparsities of the rig: a vertex is skinned by
only a handful of the 127 joints, and its lever-arm / path-sum terms are
nonzero only for the ancestors of those joints. Everything is computed on two
precomputed index sets:

``influences``  the ``nnz`` (vertex, joint, weight) triples of the LBS weights;
``pairs``       the unique (vertex, ancestor) pairs reachable from them, with
                the path-sum weight ``P[v, a] = sum_{k in infl(v), a in anc*(k)} w_vk``.

Per call the work is ``O(nnz + |pairs|)`` instead of ``O(V * J^2)``: the
skinned position of every influence (``nnz`` 3x4 affine products), sparse
matrix products that accumulate them into the pairs' lever arms and the pairs'
variance terms into the vertices, and a few gathers per pair for the rotation,
translation and scale terms. The mean-pose forward that provides the
linearisation point (blendshapes, skeleton state, optional pose correctives)
can be supplied precomputed when the caller already has it. Gradients flow to
the variance inputs; the sparse structures are constants.

The returned variances are in MHR model units squared. The released MHR assets
use centimeters, so callers that display standard deviation in millimeters
should multiply ``sqrt(variance)`` by 10.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .model_loader import NUM_FACE_EXPRESSION_BLENDSHAPES, NUM_IDENTITY_BLENDSHAPES

__all__ = ["CLUE"]


class CLUE:
    """Closed-form linear uncertainty estimator for the Momentum Human Rig (sparse evaluation)."""

    PARAMETERS_PER_JOINT = 7
    TRANSLATION_SLICE = slice(0, 3)
    ROTATION_SLICE = slice(3, 6)
    SCALE_INDEX = 6

    def __init__(self, model) -> None:
        self.model = model
        self.character_torch = model.character_torch
        self.device = self.character_torch.mesh.rest_vertices.device

        self.num_identity = NUM_IDENTITY_BLENDSHAPES
        self.num_face_expression = NUM_FACE_EXPRESSION_BLENDSHAPES
        self.num_model_parameters = (
            model.character.parameter_transform.size
            - self.num_identity
            - self.num_face_expression
        )

        self.num_joints = int(self.character_torch.skeleton.joint_parents.numel())
        expected_joint_params = self.num_joints * self.PARAMETERS_PER_JOINT
        if self.character_torch.parameter_transform.parameter_transform.shape[0] != expected_joint_params:
            raise ValueError(
                "Expected MHR joint parameters to be laid out as "
                f"{self.num_joints} x {self.PARAMETERS_PER_JOINT}."
            )

        # Per blendshape and vertex, the outer product of its offset with
        # itself (6 unique entries): summed with the coefficient variances
        # it gives each vertex's unposed 3x3 covariance.
        blendshape_vectors = self.character_torch.blend_shape.shape_vectors.to(self.device)
        self.identity_shape_outer = self._outer6(blendshape_vectors[: self.num_identity])
        self.face_expression_outer = self._outer6(
            blendshape_vectors[self.num_identity : self.num_identity + self.num_face_expression]
        )

        parameter_transform = self.character_torch.parameter_transform.parameter_transform.to(
            self.device
        )
        self.model_parameter_transform_squared = parameter_transform[
            :, : self.num_model_parameters
        ].square()

        self.parents = self.character_torch.skeleton.joint_parents.to(
            device=self.device, dtype=torch.long
        )
        self.root_mask = self.parents < 0
        self.ancestry = self._build_ancestry()

        lbs_weights = torch.as_tensor(
            model.character.skin_weights.to_dense(self.num_joints),
            dtype=self.character_torch.mesh.rest_vertices.dtype,
            device=self.device,
        )
        self.lbs_weights = lbs_weights

        self.inverse_bind_pose = self.character_torch.linear_blend_skinning.inverse_bind_pose.to(
            self.device
        )
        self.pre_rotation_mats = self._quat_to_matrix(
            self.character_torch.skeleton.joint_prerotations.to(self.device)
        )
        self._build_sparse_structures()

    def _build_ancestry(self) -> torch.Tensor:
        """Build ``ancestry[k, j] = 1`` iff joint ``j`` is an ancestor of ``k``."""
        ancestry = torch.zeros(
            self.num_joints,
            self.num_joints,
            dtype=self.lbs_weights.dtype if hasattr(self, "lbs_weights") else torch.float32,
            device=self.device,
        )
        for joint_idx in range(self.num_joints):
            ancestor_idx = joint_idx
            while ancestor_idx >= 0:
                ancestry[joint_idx, ancestor_idx] = 1.0
                ancestor_idx = int(self.parents[ancestor_idx].item())
        return ancestry

    @staticmethod
    def _outer6(vectors: torch.Tensor) -> torch.Tensor:
        """(..., 3) vectors -> (..., 6) unique entries of v v^T, ordered
        [xx, yy, zz, xy, xz, yz]."""
        x, y, z = vectors.unbind(dim=-1)
        return torch.stack([x * x, y * y, z * z, x * y, x * z, y * z], dim=-1)

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

    @staticmethod
    def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
        """Convert x/y/z/w quaternions to rotation matrices."""
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        x, y, z, w = quat.unbind(dim=-1)
        two = 2.0
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        xw, yw, zw = x * w, y * w, z * w
        return torch.stack(
            [
                1.0 - two * (yy + zz),
                two * (xy - zw),
                two * (xz + yw),
                two * (xy + zw),
                1.0 - two * (xx + zz),
                two * (yz - xw),
                two * (xz - yw),
                two * (yz + xw),
                1.0 - two * (xx + yy),
            ],
            dim=-1,
        ).reshape(quat.shape[:-1] + (3, 3))

    @staticmethod
    def _euler_xyz_matrix_and_derivatives(
        euler: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MHR XYZ-Euler rotation matrices and derivatives.

        PyMomentum's ``batch6DFromXYZ`` convention is
        ``R = Rz(z) @ Ry(y) @ Rx(x)`` for column vectors. The derivatives are
        returned in x/y/z parameter order.
        """
        x, y, z = euler.unbind(dim=-1)
        sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
        cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        def mat(entries: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(entries, dim=-1).reshape(euler.shape[:-1] + (3, 3))

        rx = mat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx])
        ry = mat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy])
        rz = mat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones])

        drx = mat([zeros, zeros, zeros, zeros, -sx, -cx, zeros, cx, -sx])
        dry = mat([-sy, zeros, cy, zeros, zeros, zeros, -cy, zeros, -sy])
        drz = mat([-sz, -cz, zeros, cz, -sz, zeros, zeros, zeros, zeros])

        rotation = rz @ ry @ rx
        derivatives = torch.stack(
            [
                rz @ ry @ drx,
                rz @ dry @ rx,
                drz @ ry @ rx,
            ],
            dim=-3,
        )
        return rotation, derivatives

    @staticmethod
    def _state_to_linear_translation(
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert PyMomentum skeleton states to affine linear/translation terms."""
        rotation = CLUE._quat_to_matrix(state[..., 3:7])
        scale = state[..., 7][..., None, None]
        return rotation * scale, state[..., :3]

    @staticmethod
    def _batch_size(*tensors: torch.Tensor | None) -> int:
        batch_size = 1
        for tensor in tensors:
            if tensor is not None:
                batch_size = max(batch_size, int(tensor.shape[0]))
        return batch_size

    @staticmethod
    def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f"{name} has batch dimension {tensor.shape[0]}, expected 1 or {batch_size}."
        )

    def _prepare_tensor(
        self,
        tensor: torch.Tensor | None,
        batch_size: int,
        width: int,
        name: str,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, width, dtype=dtype, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(f"{name} must have shape (batch, {width}), got {tuple(tensor.shape)}.")
        tensor = tensor.to(device=self.device, dtype=dtype)
        return self._expand_batch(tensor, batch_size, name)

    def _joint_parameter_variance(self, model_parameter_vars: torch.Tensor) -> torch.Tensor:
        """Propagate compact 204-D variances to full per-joint parameter variances."""
        joint_vars = model_parameter_vars @ self.model_parameter_transform_squared.to(
            dtype=model_parameter_vars.dtype
        ).T
        return joint_vars.reshape(
            model_parameter_vars.shape[0],
            self.num_joints,
            self.PARAMETERS_PER_JOINT,
        )

    def _joint_parameters(self, model_parameters: torch.Tensor) -> torch.Tensor:
        padding = torch.zeros(
            model_parameters.shape[0],
            self.num_identity + self.num_face_expression,
            dtype=model_parameters.dtype,
            device=model_parameters.device,
        )
        return self.character_torch.model_parameters_to_joint_parameters(
            torch.cat([model_parameters, padding], dim=1)
        )

    def _skin_affines(
        self,
        skel_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return skinning affine terms and global skeleton linear terms."""
        global_linear, global_translation = self._state_to_linear_translation(skel_state)
        inverse_linear, inverse_translation = self._state_to_linear_translation(
            self.inverse_bind_pose.to(dtype=skel_state.dtype)
        )

        skin_linear = torch.einsum("bjpr,jrq->bjpq", global_linear, inverse_linear)
        skin_translation = (
            torch.einsum("bjpq,jq->bjp", global_linear, inverse_translation)
            + global_translation
        )
        return skin_linear, skin_translation, global_linear, global_translation

    def _parent_terms(
        self,
        global_rotation: torch.Tensor,
        global_linear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_idx = self.parents.clamp_min(0)
        parent_rotation = global_rotation[:, parent_idx].clone()
        parent_linear = global_linear[:, parent_idx].clone()
        if bool(self.root_mask.any()):
            eye = torch.eye(
                3,
                dtype=global_rotation.dtype,
                device=global_rotation.device,
            )
            parent_rotation[:, self.root_mask] = eye
            parent_linear[:, self.root_mask] = eye
        return parent_rotation, parent_linear

    def _rotation_covariance(
        self,
        euler_mean: torch.Tensor,
        euler_vars: torch.Tensor,
        skel_state: torch.Tensor,
        parent_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Map per-Euler variance to global angular covariance per joint."""
        global_rotation = self._quat_to_matrix(skel_state[..., 3:7])
        _, local_derivatives = self._euler_xyz_matrix_and_derivatives(euler_mean)

        left = parent_rotation @ self.pre_rotation_mats.to(
            dtype=euler_mean.dtype
        ).unsqueeze(0)
        global_derivatives = torch.einsum(
            "bjpr,bjarq->bjapq",
            left,
            local_derivatives,
        )
        skew = torch.einsum(
            "bjapq,bjrq->bjapr",
            global_derivatives,
            global_rotation,
        )

        omega_by_param = torch.stack(
            [
                0.5 * (skew[..., 2, 1] - skew[..., 1, 2]),
                0.5 * (skew[..., 0, 2] - skew[..., 2, 0]),
                0.5 * (skew[..., 1, 0] - skew[..., 0, 1]),
            ],
            dim=-1,
        )
        omega_jacobian = omega_by_param.transpose(-1, -2)
        return torch.einsum(
            "bjpa,bja,bjqa->bjpq",
            omega_jacobian,
            euler_vars,
            omega_jacobian,
        )

    # ------------------------------------------------------------------ setup
    def _build_sparse_structures(self) -> None:
        device = self.device
        dtype = self.lbs_weights.dtype
        weights = self.lbs_weights.detach().cpu().numpy()                      # (V, J)
        ancestry = self.ancestry.detach().cpu().numpy() > 0                    # (J, J): [k, a] a is ancestor-or-self of k
        num_vertices, num_joints = weights.shape

        # ---- influences: the nonzero LBS weights
        infl_v, infl_j = np.nonzero(weights)
        infl_w = weights[infl_v, infl_j].astype(np.float64)
        nnz = infl_v.shape[0]

        # ---- triples: every (influence, ancestor-or-self of its joint)
        anc_lists = [np.nonzero(ancestry[k])[0] for k in range(num_joints)]
        depth = np.array([len(a) for a in anc_lists])
        tri_infl = np.repeat(np.arange(nnz), depth[infl_j])
        tri_a = np.concatenate([anc_lists[k] for k in infl_j]) if nnz else np.zeros(0, np.int64)
        tri_v = infl_v[tri_infl]

        # ---- pairs: unique (vertex, ancestor), with the path-sum weight P[v, a]
        pair_key = tri_v.astype(np.int64) * num_joints + tri_a.astype(np.int64)
        uniq, tri_pair = np.unique(pair_key, return_inverse=True)
        pair_v = (uniq // num_joints).astype(np.int64)
        pair_a = (uniq % num_joints).astype(np.int64)
        num_pairs = uniq.shape[0]
        path_sum = np.zeros(num_pairs, np.float64)
        np.add.at(path_sum, tri_pair, infl_w[tri_infl])

        # ---- S: (pairs x influences) sparse matrix, S[p, e] = w_e for each triple -> lever_sum = S @ (skinned positions)
        indices = torch.tensor(np.stack([tri_pair, tri_infl]), dtype=torch.long)
        values = torch.tensor(infl_w[tri_infl], dtype=dtype)
        lever_matrix = torch.sparse_coo_tensor(indices, values, (num_pairs, nnz)).coalesce().to_sparse_csr()

        # ---- accumulation matrices (CSR): pairs -> vertices (0/1) and influences -> vertices (weights)
        pair_to_vertex = torch.sparse_coo_tensor(
            torch.tensor(np.stack([pair_v, np.arange(num_pairs)]), dtype=torch.long),
            torch.ones(num_pairs, dtype=dtype), (num_vertices, num_pairs)).coalesce().to_sparse_csr()
        infl_to_vertex = torch.sparse_coo_tensor(
            torch.tensor(np.stack([infl_v, np.arange(nnz)]), dtype=torch.long),
            torch.tensor(infl_w, dtype=dtype), (num_vertices, nnz)).coalesce().to_sparse_csr()

        # ---- joint centres as virtual vertices: (joint, ancestor-or-self) pairs, unit path sum
        jp_j, jp_a = np.nonzero(ancestry)
        joint_pair_to_joint = torch.sparse_coo_tensor(
            torch.tensor(np.stack([jp_j, np.arange(jp_j.shape[0])]), dtype=torch.long),
            torch.ones(jp_j.shape[0], dtype=dtype), (num_joints, jp_j.shape[0])).coalesce().to_sparse_csr()

        self.num_vertices = num_vertices
        self.infl_v = torch.tensor(infl_v, dtype=torch.long, device=device)
        self.infl_j = torch.tensor(infl_j, dtype=torch.long, device=device)
        self.infl_w = torch.tensor(infl_w, dtype=dtype, device=device)
        self.pair_v = torch.tensor(pair_v, dtype=torch.long, device=device)
        self.pair_a = torch.tensor(pair_a, dtype=torch.long, device=device)
        self.pair_path_sum = torch.tensor(path_sum, dtype=dtype, device=device)
        self.lever_matrix = lever_matrix.to(device)
        self.pair_to_vertex = pair_to_vertex.to(device)
        self.infl_to_vertex = infl_to_vertex.to(device)
        self.joint_pair_j = torch.tensor(jp_j, dtype=torch.long, device=device)
        self.joint_pair_a = torch.tensor(jp_a, dtype=torch.long, device=device)
        self.joint_pair_to_joint = joint_pair_to_joint.to(device)
        self._sparse_nnz = {"lever": int(tri_infl.shape[0]), "pair_to_vertex": int(num_pairs),
                            "infl_to_vertex": int(nnz), "joint_pair_to_joint": int(jp_j.shape[0])}
        self.stats = {
            "num_vertices": int(num_vertices), "num_joints": int(num_joints), "nnz_weights": int(nnz),
            "num_triples": int(tri_infl.shape[0]), "num_pairs": int(num_pairs),
            "mean_influences_per_vertex": float(nnz / num_vertices),
            "mean_ancestors_per_vertex": float(num_pairs / num_vertices),
        }

    # ------------------------------------------------------------ core pieces
    @staticmethod
    def _cov_components(cov: torch.Tensor) -> torch.Tensor:
        """(..., 3, 3) symmetric covariances -> (..., 6) = [s00, s11, s22, 2*s01, 2*s02, 2*s12],
        the only entries the diagonal of [r]x Sigma [r]x^T needs (gathered per pair
        instead of the full block)."""
        return torch.stack(
            [cov[..., 0, 0], cov[..., 1, 1], cov[..., 2, 2],
             2.0 * cov[..., 0, 1], 2.0 * cov[..., 0, 2], 2.0 * cov[..., 1, 2]],
            dim=-1,
        )

    @staticmethod
    def _rotational_variance(lever: torch.Tensor, cov6: torch.Tensor) -> torch.Tensor:
        """Diagonal of [r]x Sigma [r]x^T for lever arms ``lever`` (..., 3) and the
        covariance components of :meth:`_cov_components` (..., 6)."""
        a, b, c = lever.unbind(-1)
        a2, b2, c2 = lever.square().unbind(-1)
        s00, s11, s22, t01, t02, t12 = cov6.unbind(-1)
        return torch.stack(
            [
                s11 * c2 + s22 * b2 - t12 * (b * c),
                s00 * c2 + s22 * a2 - t02 * (a * c),
                s00 * b2 + s11 * a2 - t01 * (a * b),
            ],
            dim=-1,
        )

    @staticmethod
    def _accumulate(matrix: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """``source`` (B, N, *rest) -> (B, M, *rest) = matrix (M x N, sparse CSR) applied
        along the second axis: one sparse-dense matmul with the batch and trailing
        dims folded into the dense columns. This replaces index_add_ along a
        non-leading axis, which is an order of magnitude slower on CPU."""
        batch, n = source.shape[0], source.shape[1]
        rest = source.shape[2:]
        flat = source.permute(1, 0, *range(2, source.dim())).reshape(n, -1)           # (N, B*prod(rest))
        out = torch.sparse.mm(matrix, flat)                                            # (M, B*prod(rest))
        return out.reshape(matrix.shape[0], batch, *rest).permute(1, 0, *range(2, source.dim()))

    def sparse_matmul_flops(self, batch: int = 1, joints: bool = False) -> int:
        """FLOPs of the sparse-dense matmuls of one call (2 per stored nonzero per
        dense column), which dispatcher-level FLOP counters do not see."""
        cols_lever, cols_pairs, cols_blend, cols_joint = 3 * batch, 3 * batch, 9 * batch, 3 * batch
        total = 2 * (self._sparse_nnz["lever"] * cols_lever + self._sparse_nnz["pair_to_vertex"] * cols_pairs
                     + self._sparse_nnz["infl_to_vertex"] * cols_blend)
        if joints:
            total += 2 * self._sparse_nnz["joint_pair_to_joint"] * cols_joint
        return int(total)

    # -------------------------------------------------------------- main entry
    def compute_variance_differentiable(
        self,
        identity_vars: torch.Tensor,
        model_parameter_vars: torch.Tensor | None = None,
        face_expr_vars: torch.Tensor | None = None,
        *,
        identity_coeffs: torch.Tensor | None = None,
        model_parameters: torch.Tensor | None = None,
        face_expr_coeffs: torch.Tensor | None = None,
        apply_correctives: bool = True,
        vertex_indices: torch.Tensor | None = None,
        return_joint_variance: bool = False,
        euler_variance_override: torch.Tensor | None = None,
        skel_state: torch.Tensor | None = None,
        unposed_vertices: torch.Tensor | None = None,
    ):
        """Propagate MHR parameter variances to per-vertex XYZ variance.

        Gradients flow from the returned variances back to the input variance
        tensors. The linearization point follows the supplied mean parameters,
        so pass detached means when the propagation should act as a per-sample
        fixed linear map. :meth:`compute_variance` is the same call under
        ``no_grad``.

        Parameters
        ----------
        identity_vars:
            ``(batch, 45)`` identity coefficient variances.
        model_parameter_vars:
            ``(batch, 204)`` compact model parameter variances. When omitted,
            model-parameter uncertainty is zero.
        face_expr_vars:
            ``(batch, 72)`` face-expression variances. When omitted,
            expression uncertainty is zero.
        identity_coeffs, model_parameters, face_expr_coeffs:
            Mean parameters used for the forward kinematic linearization.
        apply_correctives:
            If true and the loaded MHR model has pose correctives, deterministic
            mean pose-corrective offsets are included in the mean unposed mesh.
            The non-linear uncertainty of the corrective network itself is not
            propagated.
        vertex_indices:
            Optional ``(num_subset,)`` long tensor restricting the returned
            per-vertex tensors to a subset, ``(batch, num_subset, 3)``.
            Joint-level terms are unaffected by the subset.
        return_joint_variance:
            Additionally return the XYZ variance of the ``num_joints``
            skeleton joint centers, ``(batch, num_joints, 3)``. A joint center
            is the translation of the joint's global transform, i.e. exactly
            what skinning gives a point with LBS weight 1 on that joint placed
            at the joint's bind position, so it is evaluated as such a virtual
            vertex (lever arm to ancestor ``a`` = ``t_j - t_a``, unit path
            sums). Identity and expression blendshapes do not move the
            skeleton in this rig, so their contribution is zero by
            construction.
        euler_variance_override:
            ``(batch, num_joints, 3)`` rotation variance supplied directly
            (e.g. converted from matrix-Fisher concentrations) instead of
            propagated from the compact parameter variances.
        skel_state, unposed_vertices:
            ``(batch, num_joints, 8)`` and ``(batch, num_vertices, 3)``, the
            mean-pose forward when the caller has already run it. They must
            correspond to ``model_parameters`` / the blend coefficients and
            already include any pose correctives (``apply_correctives`` only
            acts when ``unposed_vertices`` is computed here). The joint
            parameters are still derived from ``model_parameters`` for the
            rotation Jacobians.

        Returns
        -------
        total_variance, pose_variance[, joint_variance]:
            ``(batch, num_vertices, 3)`` in model units squared;
            ``pose_variance`` is the sum of rotation, translation, and scale
            contributions. ``joint_variance`` only with
            ``return_joint_variance``.
        """
        dtype = identity_vars.dtype
        batch = self._batch_size(identity_vars, model_parameter_vars, face_expr_vars,
                                 identity_coeffs, model_parameters, face_expr_coeffs)
        identity_vars = self._prepare_tensor(identity_vars, batch, self.num_identity, "identity_vars", dtype=dtype)
        model_parameter_vars = self._prepare_tensor(model_parameter_vars, batch, self.num_model_parameters,
                                                    "model_parameter_vars", dtype=dtype)
        face_expr_vars = self._prepare_tensor(face_expr_vars, batch, self.num_face_expression, "face_expr_vars", dtype=dtype)
        identity_coeffs = self._prepare_tensor(identity_coeffs, batch, self.num_identity, "identity_coeffs", dtype=dtype)
        model_parameters = self._prepare_tensor(model_parameters, batch, self.num_model_parameters,
                                                "model_parameters", dtype=dtype)
        face_expr_coeffs = self._prepare_tensor(face_expr_coeffs, batch, self.num_face_expression,
                                                "face_expr_coeffs", dtype=dtype)

        # ---- linearisation point (the mean pose)
        joint_parameters = self._joint_parameters(model_parameters)
        if skel_state is None:
            skel_state = self.character_torch.joint_parameters_to_skeleton_state(joint_parameters)
        if unposed_vertices is None:
            unposed_vertices = self.character_torch.blend_shape.forward(torch.cat([identity_coeffs, face_expr_coeffs], dim=1))
            if apply_correctives and self.model.pose_correctives_model is not None:
                unposed_vertices = unposed_vertices + self.model.pose_correctives_model(joint_parameters)

        # ---- per-joint variances of translation / Euler angles / log2-scale
        joint_parameter_vars = self._joint_parameter_variance(model_parameter_vars)
        trans_vars = joint_parameter_vars[..., self.TRANSLATION_SLICE]
        euler_vars = joint_parameter_vars[..., self.ROTATION_SLICE]
        if euler_variance_override is not None:
            euler_vars = euler_variance_override.to(dtype=euler_vars.dtype)
        scale_vars = (math.log(2.0) ** 2) * joint_parameter_vars[..., self.SCALE_INDEX]      # (B, J)

        skin_linear, skin_translation, global_linear, global_translation = self._skin_affines(skel_state)
        global_rotation = self._quat_to_matrix(skel_state[..., 3:7])
        parent_rotation, parent_linear = self._parent_terms(global_rotation, global_linear)
        euler_mean = joint_parameters.reshape(batch, self.num_joints, self.PARAMETERS_PER_JOINT)[..., self.ROTATION_SLICE]
        rotation_cov = self._rotation_covariance(euler_mean, euler_vars, skel_state, parent_rotation)   # (B, J, 3, 3)
        cov6 = self._cov_components(rotation_cov)                                                       # (B, J, 6)
        trans_var_global = torch.einsum("bjpa,bja->bjp", parent_linear.square(), trans_vars)          # (B, J, 3)

        # ---- per-influence skinned positions (only the nnz vertex-joint pairs)
        x = unposed_vertices[:, self.infl_v]                                                          # (B, nnz, 3)
        lin = skin_linear[:, self.infl_j]                                                             # (B, nnz, 3, 3)
        skinned = torch.einsum("bepq,beq->bep", lin, x) + skin_translation[:, self.infl_j]            # (B, nnz, 3)

        # ---- lever arms of the (vertex, ancestor) pairs
        lever = self._accumulate(self.lever_matrix, skinned) - self.pair_path_sum[None, :, None] * global_translation[:, self.pair_a]

        # ---- the three pose terms per pair, accumulated per vertex
        per_pair = (
            self._rotational_variance(lever, cov6[:, self.pair_a])
            + self.pair_path_sum[None, :, None].square() * trans_var_global[:, self.pair_a]
            + lever.square() * scale_vars[:, self.pair_a, None]
        )
        pose_variance = self._accumulate(self.pair_to_vertex, per_pair)                                # (B, V, 3)

        # ---- blendshape term: diagonal of E C E^T, E = sum_k w_vk G_k, C the unposed 3x3 covariance (6 entries)
        blend_unposed = (
            torch.einsum("bn,nvk->bvk", identity_vars, self.identity_shape_outer.to(dtype=dtype))
            + torch.einsum("bn,nvk->bvk", face_expr_vars, self.face_expression_outer.to(dtype=dtype))
        )
        effective_linear = self._accumulate(self.infl_to_vertex, lin)                                  # (B, V, 3, 3)
        blend_variance = self._transformed_diagonal(effective_linear, blend_unposed)

        total_variance = torch.clamp(blend_variance + pose_variance, min=0.0)
        pose_variance = torch.clamp(pose_variance, min=0.0)
        if vertex_indices is not None:
            vertex_indices = torch.as_tensor(vertex_indices, dtype=torch.long, device=self.device)
            total_variance = total_variance.index_select(1, vertex_indices)
            pose_variance = pose_variance.index_select(1, vertex_indices)
        if not return_joint_variance:
            return total_variance, pose_variance

        # ---- joint centres: virtual vertices with unit weight on their own joint, lever arm t_j - t_a
        lever_j = global_translation[:, self.joint_pair_j] - global_translation[:, self.joint_pair_a]
        per_joint_pair = (
            self._rotational_variance(lever_j, cov6[:, self.joint_pair_a])
            + trans_var_global[:, self.joint_pair_a]
            + lever_j.square() * scale_vars[:, self.joint_pair_a, None]
        )
        joint_variance = self._accumulate(self.joint_pair_to_joint, per_joint_pair)                    # (B, J, 3)
        return total_variance, pose_variance, torch.clamp(joint_variance, min=0.0)

    @torch.no_grad()
    def compute_variance(self, *args, **kwargs):
        """:meth:`compute_variance_differentiable` under ``no_grad``."""
        return self.compute_variance_differentiable(*args, **kwargs)
