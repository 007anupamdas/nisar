"""
Glue Factory's LightGlue, told each view's modality and the pair's pixel
size (condition.py) before it matches. Matcher name gf_cond_lightglue, with
this folder on PYTHONPATH:

    matcher:
        name: gf_cond_lightglue
        condition: true            # false: plain LightGlue
        weights: aliked            # the official weights it starts from

The samples carry modality0, modality1 and px_m (dataset.py). Without them
(another dataset) it matches as plain LightGlue.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gluefactory.models.matchers.lightglue import LightGlue  # noqa: E402

from condition import Condition  # noqa: E402


class CondLightGlue(LightGlue):
    default_conf = {**LightGlue.default_conf, 'condition': True}

    def __init__(self, conf) -> None:
        super().__init__(conf)                   # loads the official weights (strict=False)
        self.condition = Condition(self.conf.input_dim)

    def forward(self, data: dict) -> dict:
        if self.conf.condition and 'modality0' in data and 'modality1' in data:
            px = data.get('px_m')
            if px is None:
                px = torch.full(data['modality0'].shape, 30.0, device=data['descriptors0'].device)
            data = {**data,
                    'descriptors0': self.condition(data['descriptors0'], data['modality0'], px),
                    'descriptors1': self.condition(data['descriptors1'], data['modality1'], px)}
        return super().forward(data)


__main_model__ = CondLightGlue
