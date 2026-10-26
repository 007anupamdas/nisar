"""
Glue Factory dataset for the tie-point pairs (data.name: gf_gcp, with this
folder on PYTHONPATH). Glue Factory finds it by module path and uses its
BaseDataset loaders; the samples come from dataset.PairDataset.

    data:
        name: gf_gcp
        index: /path/to/pairs/pairs.csv
        channels: 3
        sensors: null          # or a list, e.g. [MicroSat, G1A_MV]
        batch_size: 16
        num_workers: 8
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gluefactory.datasets.base_dataset import BaseDataset  # noqa: E402

from dataset import PairDataset  # noqa: E402


class GcpPairs(BaseDataset):
    default_conf = {
        'index': '???',
        'channels': 3,
        'sensors': None,
        'max_pairs': None,
        'augment': True,
        'train_batch_size': 16,
        'val_batch_size': 16,
        'test_batch_size': 16,
        'num_workers': 8,
    }

    def _init(self, conf):
        if not os.path.exists(conf.index):
            raise FileNotFoundError(f'gf_gcp: index {conf.index} not found (run pairs.py first)')

    def get_dataset(self, split):
        sensors = list(self.conf.sensors) if self.conf.sensors else None
        return PairDataset(self.conf.index, split, channels=self.conf.channels,
                           augment=(self.conf.augment if split == 'train' else False),
                           swap=(split == 'train'), seed=self.conf.seed, sensors=sensors,
                           max_pairs=self.conf.max_pairs)


__main_dataset__ = GcpPairs
