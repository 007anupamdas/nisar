"""
Points inside featureless areas: the middle of a lake, river, sea or a
uniform blob (snow, cloud top, a smooth slope in shadow) has nothing to match
on, so a match there is a coincidence even when several detectors agree.
Their edges (shore lines, ridge lines) are where the good matches are.

interior_mask() marks, on the input band, the pixels farther than
edge_px pixels from any textured pixel: class 1 (reject), 0 (keep), 255
(no data). A pixel is textured when the local standard deviation (window
win_px) is above k_noise x the image's noise, the noise estimated robustly
(the median absolute Laplacian, and the flattest 5 % of the local standard
deviations, whichever is lower). Computed on an overview of at most
max_side pixels; the distance is measured at full resolution (edge_px is in
input pixels).

The point filter (automatch_cloudfilter, kinds 'interior') samples the mask
at each match's input position.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

DEFAULTS = {'edge_px': 4.0, 'win_px': 5, 'k_noise': 3.0, 'max_side': 4096}


def interior_mask(path: str, band: int, out_path: str, edge_px: float = DEFAULTS['edge_px'],
                  win_px: int = DEFAULTS['win_px'], k_noise: float = DEFAULTS['k_noise'],
                  max_side: int = DEFAULTS['max_side']) -> Dict:
    """Write the interior mask of one band (module docstring); its stats."""
    import cv2
    import rasterio as rio
    from rasterio.enums import Resampling
    with rio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / float(max_side))
        h, w = max(1, int(round(src.height / scale))), max(1, int(round(src.width / scale)))
        a = src.read(int(band), out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
        nod = src.nodatavals[int(band) - 1] if src.nodatavals else None
        tr = src.transform * src.transform.scale(src.width / float(w), src.height / float(h))
        crs = src.crs
    valid = np.isfinite(a) & (a != 0)
    if nod is not None:
        valid &= a != nod
    if not valid.any():
        raise ValueError(f'{os.path.basename(path)} band {band}: no valid pixels')
    fill = float(np.median(a[valid]))
    x = np.where(valid, a, fill)
    # local standard deviation over win_px (overview pixels)
    k = max(3, int(win_px) | 1)
    m = cv2.blur(x, (k, k))
    m2 = cv2.blur(x * x, (k, k))
    sd = np.sqrt(np.maximum(m2 - m * m, 0.0))
    lap = cv2.Laplacian(x, cv2.CV_32F, ksize=1)                     # 4-neighbour Laplacian: std = sqrt(20) sigma
    inner = valid & (cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    sel = inner if inner.sum() > 100 else valid
    noise_lap = 1.4826 * float(np.median(np.abs(lap[sel]))) / np.sqrt(20.0)
    noise_flat = float(np.percentile(sd[sel], 5))
    noise = max(min(noise_lap, noise_flat) if noise_flat > 0 else noise_lap, 1e-6)
    textured = valid & (sd > k_noise * noise)
    # distance (overview pixels) to the nearest textured pixel; edge_px is in input pixels
    dist = cv2.distanceTransform((~textured).astype(np.uint8), cv2.DIST_L2, 3)
    interior = valid & (dist * scale > float(edge_px))
    cls = np.zeros((h, w), np.uint8)
    cls[interior] = 1
    cls[~valid] = 255
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with rio.open(out_path, 'w', driver='GTiff', height=h, width=w, count=1, dtype='uint8', crs=crs, transform=tr,
                  nodata=255, compress='deflate') as dst:
        dst.write(cls, 1)
    nv = max(int(valid.sum()), 1)
    return {'mask': out_path, 'interior_pct': round(100.0 * float(interior.sum()) / nv, 2),
            'noise': round(noise, 4), 'threshold': round(k_noise * noise, 4), 'overview_scale': round(scale, 2),
            'edge_px': edge_px, 'band': int(band)}


def options(job: Dict) -> Dict:
    """interior_* settings of a job (defaults where unset)."""
    o = dict(DEFAULTS)
    for key in DEFAULTS:
        v = job.get(f'interior_{key}')
        if v not in (None, ''):
            o[key] = type(DEFAULTS[key])(v)
    return o


def band_number(channel: Optional[str]) -> int:
    import re
    m = re.search(r'(\d+)', str(channel or ''))
    return int(m.group(1)) if m else 1
