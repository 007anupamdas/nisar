"""
The training pairs as a PyTorch dataset, in the sample layout Glue Factory's
two-view pipeline reads:

    {'name', 'H_0to1' (3 x 3 float32),
     'view0': {'image': (C, H, W) float32 in 0..1, 'image_size': (W, H)},
     'view1': {...}, 'scene', 'sensor',
     'modality0', 'modality1' (int: modality.MODALITIES), 'px_m' (float)}

C is 3 (the band repeated) for ALIKED / DISK, which expect RGB, or 1.
Training samples are lightly augmented (brightness, contrast, gamma, noise,
blur, each view on its own) and, half of the time, swapped (view1 first,
with H inverted and the modalities exchanged), so the matcher does not
learn which side is the reference.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import modality as M


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2
    x = img.astype(np.float32)
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-0.1, 0.1)           # contrast, brightness
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)                     # gamma
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
    x = x + rng.normal(0, rng.uniform(0, 0.03), x.shape).astype(np.float32)
    return np.clip(x, 0, 1)


class PairDataset:
    """pairs.csv of one split. Works as a torch.utils.data.Dataset (no torch
    import needed until an item is read)."""

    def __init__(self, index: str, split: str = 'train', channels: int = 3, augment: Optional[bool] = None,
                 swap: Optional[bool] = None, seed: int = 0, sensors: Optional[List[str]] = None,
                 max_pairs: Optional[int] = None, modalities: Optional[List[str]] = None):
        self.root = os.path.dirname(os.path.abspath(index))
        df = pd.read_csv(index)
        df = df[df['split'].astype(str) == split]
        if sensors:
            df = df[df['sensor'].astype(str).isin([str(s) for s in sensors])]
        if modalities and 'modality0' in df:
            df = df[df['modality0'].astype(str).isin([M.resolve(m) for m in modalities])]
        if max_pairs:
            df = df.head(int(max_pairs))
        self.rows = df.to_dict('records')
        self.channels = int(channels)
        self.augment = (split == 'train') if augment is None else bool(augment)
        self.swap = (split == 'train') if swap is None else bool(swap)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, rel: str) -> np.ndarray:
        import cv2
        a = cv2.imread(os.path.join(self.root, rel), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise FileNotFoundError(os.path.join(self.root, rel))
        return a.astype(np.float32) / 255.0

    def __getitem__(self, i: int) -> Dict:
        import torch
        r = self.rows[i]
        rng = np.random.default_rng(self.seed * 1000003 + i + (np.random.randint(0, 1 << 30) if self.augment else 0))
        im0, im1 = self._image(r['img0']), self._image(r['img1'])
        H = np.asarray([float(v) for v in str(r['H']).split()], np.float64).reshape(3, 3)
        m0, m1 = M.index(r.get('modality0')), M.index(r.get('modality1'))
        px = float(r.get('px_m') or 0)
        px = px if np.isfinite(px) and px > 0 else 30.0
        if self.augment:
            im0, im1 = _augment(im0, rng), _augment(im1, rng)
        if self.swap and rng.random() < 0.5:
            im0, im1, H, m0, m1 = im1, im0, np.linalg.inv(H), m1, m0
        H = H / H[2, 2]

        def view(a: np.ndarray) -> Dict:
            t = torch.from_numpy(np.ascontiguousarray(a))[None]
            if self.channels == 3:
                t = t.repeat(3, 1, 1)
            return {'image': t.float(), 'image_size': torch.tensor([a.shape[1], a.shape[0]], dtype=torch.float32)}
        return {'name': f"{r['scene_id']}/{os.path.basename(r['img0'])}", 'scene': str(r['scene_id']),
                'sensor': str(r.get('sensor', '')), 'H_0to1': torch.from_numpy(H.astype(np.float32)),
                'view0': view(im0), 'view1': view(im1), 'modality0': m0, 'modality1': m1, 'px_m': px}
