"""
The matcher told what it is looking at: FiLM on the keypoint descriptors.

Each view's descriptors d (B, N, D) are scaled and shifted by amounts learnt
for the view's modality (modality.py) and for the pair's pixel size:

    d' = d * (1 + g) + b,   (g, b) = E[modality] + R(log2(px_m / 30) / 4)

E is an embedding (a row per modality), R a small MLP. Both start at zero,
so a fine-tune starts from exactly the official matcher and moves away from
it only as far as the tie-point pairs pay for. Another tool applies the
same transform before calling LightGlue; it needs only the view's modality
and the pixel size.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from modality import MODALITIES  # noqa: E402


class Condition(nn.Module):
    def __init__(self, dim: int, n_modalities: int = len(MODALITIES), hidden: int = 32):
        super().__init__()
        self.modality = nn.Embedding(n_modalities, 2 * dim)
        self.resolution = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.modality.weight)
        nn.init.zeros_(self.resolution[2].weight)
        nn.init.zeros_(self.resolution[2].bias)

    def forward(self, desc: torch.Tensor, modality: torch.Tensor, px_m: torch.Tensor) -> torch.Tensor:
        """desc (B, N, D); modality (B,) ints; px_m (B,) metres."""
        r = torch.log2(px_m.reshape(-1, 1).float().clamp(min=0.01) / 30.0) / 4.0     # about -1..1 for 2-500 m
        g, b = (self.modality(modality.reshape(-1).long()) + self.resolution(r)).to(desc.dtype).chunk(2, dim=-1)
        return desc * (1.0 + g[:, None]) + b[:, None]
