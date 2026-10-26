"""
One scene's mapping: input pixel (col, row) -> true ground (E, N), from its
tie points.

The tie points say where each feature lies in the image (its pixel, from the
image's own lon/lat) and where it really is (the reference lon/lat). An affine
model (6 numbers) or, where the error changes across the scene (G1A job 347:
+49 km in the west to +77 km in the east), a quadratic one (12 numbers) maps
pixels to true ground. With 10-15 points per scene the model is chosen by
leave-one-out error: the quadratic only when it predicts left-out points
clearly better (by a factor 0.8). Points whose leave-one-out residual stands
out (> 3 x the median and > 1 px) are dropped, at most three, keeping at least
6. A scene is used for training only when its leave-one-out RMSE is at most
max_loo_px: supervision any worse would teach the matcher wrong positions.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import geo


def _design(c: np.ndarray, r: np.ndarray, quad: bool, c0: float, r0: float, s: float) -> np.ndarray:
    u, v = (c - c0) / s, (r - r0) / s
    cols = [np.ones_like(u), u, v]
    if quad:
        cols += [u * u, u * v, v * v]
    return np.column_stack(cols)


class SceneModel:
    """Input pixel -> true ground (E, N) in the scene's UTM zone."""

    def __init__(self, coef: np.ndarray, quad: bool, c0: float, r0: float, s: float, crs: str):
        self.coef, self.quad, self.c0, self.r0, self.s, self.crs = coef, quad, c0, r0, s, crs

    def __call__(self, col, row):
        col, row = np.asarray(col, float), np.asarray(row, float)
        A = _design(col.ravel(), row.ravel(), self.quad, self.c0, self.r0, self.s)
        en = A @ self.coef
        return en[:, 0].reshape(col.shape), en[:, 1].reshape(col.shape)

    def to_dict(self) -> Dict:
        return {'coef': self.coef.tolist(), 'quad': self.quad, 'c0': self.c0, 'r0': self.r0, 's': self.s,
                'crs': self.crs}

    @classmethod
    def from_dict(cls, d: Dict) -> 'SceneModel':
        return cls(np.asarray(d['coef'], float), bool(d['quad']), d['c0'], d['r0'], d['s'], d['crs'])


def _solve(c, r, E, N, quad, c0, r0, s):
    A = _design(c, r, quad, c0, r0, s)
    coef = np.linalg.lstsq(A, np.column_stack([E, N]), rcond=None)[0]
    return coef


def _loo(c, r, E, N, quad, c0, r0, s) -> np.ndarray:
    """Leave-one-out residual (m) of every point."""
    n = len(c)
    res = np.full(n, np.inf)
    need = 6 if quad else 3
    if n - 1 < need + 1:                       # nothing left to check the fit
        return res
    for i in range(n):
        m = np.arange(n) != i
        coef = _solve(c[m], r[m], E[m], N[m], quad, c0, r0, s)
        p = _design(c[i:i + 1], r[i:i + 1], quad, c0, r0, s) @ coef
        res[i] = float(np.hypot(p[0, 0] - E[i], p[0, 1] - N[i]))
    return res


def fit_scene(points: pd.DataFrame, input_path: str, max_loo_px: float = 1.5,
              quad_gain: float = 0.8, max_drop: int = 3) -> Dict:
    """{'ok', 'model' (SceneModel), 'type', 'loo_rmse_px', 'n_used', 'n_points',
    'dropped', 'pixel_m', 'bounds_px' (col0, row0, col1, row1 of the points),
    'mean_error_m' (dE, dN), 'why'} for one scene."""
    import rasterio as rio
    with rio.open(input_path) as src:
        col, row = geo.lonlat_to_pixel(src, points['in_lon'].values, points['in_lat'].values)
        px = geo.ground_pixel_m(src)
        width, height = src.width, src.height
        ix, iy = src.transform * (col + 0.5, row + 0.5)
    lon0, lat0 = float(points['ref_lon'].mean()), float(points['ref_lat'].mean())
    crs = f'EPSG:{geo.utm_epsg(lon0, lat0)}'
    E, N = geo.transformer('EPSG:4326', crs).transform(points['ref_lon'].values, points['ref_lat'].values)
    E, N = np.asarray(E, float), np.asarray(N, float)
    inside = (col >= -0.5) & (row >= -0.5) & (col < width - 0.5) & (row < height - 0.5)
    pid = points['point_id'].values
    out = {'ok': False, 'n_points': int(len(points)), 'pixel_m': px, 'dropped': [], 'why': ''}
    if (~inside).any():
        out['outside_image'] = [int(p) for p in pid[~inside]]
    c, r, E, N, pid = col[inside], row[inside], E[inside], N[inside], pid[inside]
    if len(c) < 5:
        out['why'] = f'{len(c)} point(s) inside the image: too few for a checked fit'
        return out
    c0, r0 = float(c.mean()), float(r.mean())
    s = float(max(np.ptp(c), np.ptp(r), 1.0))
    keep = np.ones(len(c), bool)
    for _ in range(max_drop + 1):
        quad_ok = keep.sum() >= 9                  # 6 coefficients + 3 to spare
        la = _loo(c[keep], r[keep], E[keep], N[keep], False, c0, r0, s)
        lq = _loo(c[keep], r[keep], E[keep], N[keep], True, c0, r0, s) if quad_ok else None
        rms_a = float(np.sqrt(np.mean(la ** 2)))
        rms_q = float(np.sqrt(np.mean(lq ** 2))) if lq is not None else np.inf
        quad = rms_q < quad_gain * rms_a
        loo = lq if quad else la
        worst = int(np.argmax(loo))
        med = float(np.median(loo))
        if keep.sum() > 6 and loo[worst] > 3 * med and loo[worst] > px and len(out['dropped']) < max_drop:
            idx = np.flatnonzero(keep)[worst]
            keep[idx] = False
            out['dropped'].append(int(pid[idx]))
            continue
        break
    coef = _solve(c[keep], r[keep], E[keep], N[keep], quad, c0, r0, s)
    model = SceneModel(coef, quad, c0, r0, s, crs)
    loo_px = float(np.sqrt(np.mean(loo ** 2))) / px
    ixu, iyu = geo.transformer(src_crs_of(input_path), crs).transform(ix[inside][keep], iy[inside][keep])
    out.update({'model': model, 'type': 'quadratic' if quad else 'affine', 'loo_rmse_px': round(loo_px, 3),
                'n_used': int(keep.sum()), 'crs': crs,
                'bounds_px': [float(c[keep].min()), float(r[keep].min()), float(c[keep].max()), float(r[keep].max())],
                'mean_error_m': [round(float(np.mean(np.asarray(ixu) - E[keep])), 1),
                                 round(float(np.mean(np.asarray(iyu) - N[keep])), 1)]})
    if not np.isfinite(loo_px):
        out['why'] = 'leave-one-out not possible with these points'
    elif loo_px > max_loo_px:
        out['why'] = f'leave-one-out RMSE {loo_px:.2f} px > {max_loo_px:g} px'
    else:
        out['ok'] = True
    return out


def src_crs_of(path: str):
    import rasterio as rio
    with rio.open(path) as s:
        return s.crs
