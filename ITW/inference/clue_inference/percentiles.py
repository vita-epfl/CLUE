"""Per-vertex sigma -> percentile against a fixed reference, one per model.

percentiles/clue.npz and percentiles/sam3dbody_clue.npz hold, for every vertex, that model's sigma at 401 evenly spaced
quantile levels over the people of a curated in-the-wild video (6,118 person-frames), a vertex and its left-right mirror
twin pooled into one distribution so both sides are ranked on the same yardstick. A new sigma is placed in that
distribution by linear interpolation between levels: 0 = as certain as this vertex ever gets in the reference for that
model, 100 = the most uncertain it gets. Each model is ranked against its own reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TABLES = {name: Path(__file__).resolve().parent.parent / f"percentiles/{name}.npz" for name in ("clue", "sam3dbody_clue")}


class PercentileTable:
    def __init__(self, path: Path, device: torch.device | str = "cuda") -> None:
        z = np.load(path, allow_pickle=False)
        self.levels = torch.as_tensor(z["levels"].astype(np.float32), device=device)                       # (Q,)
        self.sigma = torch.as_tensor(z["sigma_at_levels"].astype(np.float32), device=device).contiguous()  # (V, Q)

    @torch.no_grad()
    def __call__(self, sigma_cm: torch.Tensor | np.ndarray) -> torch.Tensor:
        """(P, V) or (V,) sigma -> percentile in [0, 1], same shape."""
        s = torch.as_tensor(np.asarray(sigma_cm) if not torch.is_tensor(sigma_cm) else sigma_cm, dtype=torch.float32,
                            device=self.sigma.device)
        single = s.dim() == 1
        s = s.reshape(-1, s.shape[-1]).T.contiguous()                                                       # (V, P)
        q = len(self.levels)
        idx = torch.searchsorted(self.sigma, s)                         # first level whose sigma >= s (count of levels below)
        lo, hi = (idx - 1).clamp(0, q - 1), idx.clamp(0, q - 1)
        s_lo, s_hi = self.sigma.gather(1, lo), self.sigma.gather(1, hi)
        frac = torch.where(s_hi > s_lo, (s - s_lo) / (s_hi - s_lo).clamp_min(1e-9), torch.ones_like(s))
        u = self.levels[lo] + frac.clamp(0, 1) * (self.levels[hi] - self.levels[lo])
        u = torch.where(idx == 0, torch.zeros_like(u), torch.where(idx >= q, torch.ones_like(u), u)).T
        return u[0] if single else u
