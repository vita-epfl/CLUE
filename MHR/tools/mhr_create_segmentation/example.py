# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
pixi run python example.py
"""


import re
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
import trimesh

import pymomentum.geometry as pym_geometry

from mhr.mhr import MHR

# Joint name substrings for grouping
ZERO_WEIGHT_PARTS = {"eye", "tongue", "teeth", "brow", "cheek", "lip", "ear", "nose", "body_world"}
HAND_PARTS = {"thumb", "index", "middle", "ring", "pinky", "wrist"}
FOOT_PARTS = {"foot", "talocrural", "subtalar", "transversetarsal", "ball"}

# matplotlib tab20 colormap (20 distinct colors)
TAB20_COLORS = np.array([
    [0.122, 0.467, 0.706], [0.682, 0.780, 0.910],
    [1.000, 0.498, 0.055], [1.000, 0.733, 0.471],
    [0.173, 0.627, 0.173], [0.596, 0.875, 0.541],
    [0.839, 0.153, 0.157], [1.000, 0.596, 0.588],
    [0.580, 0.404, 0.741], [0.773, 0.690, 0.835],
    [0.549, 0.337, 0.294], [0.769, 0.612, 0.580],
    [0.890, 0.467, 0.761], [0.969, 0.714, 0.824],
    [0.498, 0.498, 0.498], [0.780, 0.780, 0.780],
    [0.737, 0.741, 0.133], [0.859, 0.859, 0.553],
    [0.090, 0.745, 0.812], [0.620, 0.855, 0.898],
])


def joint_group_key(name: str) -> str:
    """Map a joint name to its semantic group key.

    Groups joints by:
    - Zero-weight facial/body joints (eye, tongue, etc.)
    - Hand joints (fingers + wrist, prefixed by side)
    - Foot joints (foot bones, prefixed by side)
    - Otherwise, strip trailing indices/suffixes to merge subparts
      (e.g. "p_l_upleg_twist2" -> "p_l_upleg")
    """
    lower = name.lower()

    if any(part in lower for part in ZERO_WEIGHT_PARTS):
        return "zero_weight"

    # Check after the 2-char side prefix (e.g. "l_" or "r_")
    suffix = lower[2:]
    if any(part in suffix for part in HAND_PARTS):
        return f"{lower[:2]}hand"
    if any(part in suffix for part in FOOT_PARTS):
        return f"{lower[:2]}foot"

    # Strip trailing subpart indices: _null, _twist2, _twist1_proc, or bare digits
    key = re.sub(r"(_null|_twist\d*|\d+|_twist\d_proc)$", "", name).rstrip("_")
    return key


def build_joint_groups(joint_names: list[str]) -> dict[str, list[int]]:
    """Group joint indices by their semantic group key."""
    groups = defaultdict(list)
    for idx, name in enumerate(joint_names):
        groups[joint_group_key(name)].append(idx)
    return groups


def compute_part_weights(
    groups: dict[str, list[int]],
    skin_weights: pym_geometry.SkinWeights,
    hard_assignment: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute per-vertex weight for each joint group by summing member weights.

    If hard_assignment is True, only the primary influence (first column) is
    used: the weight matrix is replaced by ones in the first column and zeros
    elsewhere, so each vertex contributes exactly 1.0 to its dominant joint.

    Returns:
        weights: (N_groups, N_vertices) array of per-vertex weights.
        vertex_indices: dict mapping each group key to the vertex indices
            with non-zero weight in that group.
    """
    if hard_assignment:
        effective_weights = np.zeros_like(skin_weights.weight)
        effective_weights[:, 0] = 1.0
    else:
        effective_weights = skin_weights.weight

    weights = []
    vertex_indices = {}
    for group_key, joint_indices in groups.items():
        masks = [skin_weights.index == j for j in joint_indices]
        combined_mask = reduce(np.logical_or, masks)
        part_weight = (combined_mask * effective_weights).sum(axis=1)
        weights.append(part_weight)
        vertex_indices[group_key] = np.nonzero(part_weight)[0]

    return np.asarray(weights), vertex_indices



if __name__ == "__main__":
    model = MHR.from_files(lod=1, device=torch.device("cpu"))
    character = model.character

    mesh = trimesh.Trimesh(vertices=character.mesh.vertices, faces=character.mesh.faces)

    groups = build_joint_groups(character.skeleton.joint_names)
    part_weights, vertex_indices = compute_part_weights(groups, character.skin_weights, hard_assignment=True)
    mesh.visual.vertex_colors = (part_weights.T @ TAB20_COLORS * 255).astype(np.uint8)

    mesh.export("/tmp/segmented_template.ply")
    np.savez("/tmp/part_segments.npz", **vertex_indices)
