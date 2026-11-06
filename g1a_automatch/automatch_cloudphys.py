"""
Cloud masks from physics, for the bands OmniCloudMask cannot use (it needs
red, green and NIR): G1A's MX-LWIR (ML, six thermal bands) and HyS-SWIR (HS).

ML -- infrared tests, as the operational geostationary cloud masks (GOES
ABI, Himawari AHI, Meteosat SEVIRI) use on the same channels; day and night.
  radiance   value x radiance_scale (G1A: scaled radiance, 0.01). The unit is
             not stated, so the one giving Earth-like brightness temperatures
             is taken (W/m2/sr/um, mW/cm2/sr/um, mW/m2/sr/um or per
             wavenumber) and named in the log.
  BT         inverse Planck at each band's centre: 7.35, 8.5, 9.6, 10.8, 12.0,
             13.25 um (MX-LWIR B1-B6) unless the band descriptions say others.
  cold       BT10.8 colder than clear sky by ir_dt_k (8 K). Clear sky: from
             earlier scenes of the same ground (cloud_ir_clear: the warmest
             value of each pixel over them, best at the same hour -- the
             geostationary advantage), else from the scene itself (the 90th
             percentile of ~100 km blocks, never below the scene's 60th).
             Colder than ir_cold_k (240 K) is cloud whatever.
  split      BT10.8 - BT12.0 above the split-window threshold for its BT10.8
             (about Saunders & Kriebel 1988, nadir; x ir_split_scale): thin
             cirrus.
  ice        BT8.5 - BT10.8 > 0.5 K: ice cloud (clear ground is negative there).
  high       BT13.25 - BT10.8 or BT7.35 - BT10.8 above -8 K while colder than
             clear by 3 K: high, opaque cloud (over clear warm ground the CO2
             and water-vapour bands are 20-40 K colder than the window).
  edges      3 x 3 standard deviation of BT10.8 (valid pixels only) > 1.5 K
             where colder than clear by half of ir_dt_k, next to a pixel the
             cold test found: the ragged edges of cloud (textured ground away
             from cloud does not pass).
  thick (1): cold by twice ir_dt_k, colder than ir_cold_k, or high; thin (2):
  the other tests; blobs smaller than min_blob_km2 are dropped.

HS -- SWIR tests, relative to the scene (any radiance unit), daylight only
(night: the 99th percentile at 1.25 um below 0.05 after the radiance scale,
where daylit ground gives about 1-20 W/m2/sr/um).
  cirrus     L1380 / L1250 above its clear level (the scene's 20th
             percentile) by swir_k (4) robust standard deviations of the clear
             half: water vapour hides the ground at 1.38 um and high or thin
             cloud shows (Landsat 8 band 9, Sentinel-2 B10). High terrain shows
             through too (less vapour above it).
  bright     L1250 brighter than the clear ground by swir_bright_sd (3)
             spreads -- the median plus that many times the median's distance
             down to the 16th percentile, the dark half being ground even when
             much of the scene is cloud -- with L1650 above its median plus one
             such spread (snow is dark at 1.65 um): thick cloud, in blobs.
             Low cloud over
             bright ground stays hard from SWIR alone (visible + SWIR does much
             better: CloudAHSI), so with cloud_mask_from the VNIR pass's mask
             is added where it covers this image.

The thresholds are starting points from the literature; the cloud sweep
varies them (automatch_cloudsweep) against reference outlines.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ML_UM = [7.35, 8.5, 9.6, 10.8, 12.0, 13.25]       # MX-LWIR B1-B6 centres
C1 = 1.191042e8                                    # 2hc^2, W um^4 m^-2 sr^-1
C2 = 1.4387752e4                                   # hc/k, um K
# split-window threshold (K) by BT10.8 (K): about Saunders & Kriebel (1988), nadir
SPLIT_T = [250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0]
SPLIT_D = [0.45, 0.55, 0.80, 1.10, 1.55, 2.40, 3.50]
IR_DEFAULTS = {'dt_k': 8.0, 'split_scale': 1.0, 'cold_k': 240.0, 'min_blob_km2': 1.0}
SWIR_DEFAULTS = {'k': 4.0, 'bright_sd': 3.0, 'min_blob_km2': 1.0}
NIGHT_L1250 = 0.05            # below this (99th percentile, after the radiance scale): no sunlight

# radiance unit -> W m-2 sr-1 um-1, as a function of the value and the band centre (um)
UNITS = (('W/m2/sr/um', lambda v, um: v),
         ('mW/cm2/sr/um', lambda v, um: v * 10.0),
         ('mW/m2/sr/um', lambda v, um: v * 1e-3),
         ('mW/m2/sr/cm-1', lambda v, um: v * 10.0 / um ** 2),
         ('W/m2/sr/cm-1', lambda v, um: v * 1e4 / um ** 2))


def planck_bt(L: np.ndarray, um: float) -> np.ndarray:
    """Brightness temperature (K) of spectral radiance L (W m-2 sr-1 um-1) at um."""
    L = np.maximum(np.asarray(L, np.float64), 1e-9)
    return C2 / (um * np.log1p(C1 / (um ** 5 * L)))


def to_bt(rad: Dict[float, np.ndarray], valid: np.ndarray) -> Tuple[Dict[float, np.ndarray], str, float]:
    """({um: BT}, unit taken, median BT10.8): the radiance unit whose BT at the
    window band lies nearest 285 K (Earth scenes: 200-330 K)."""
    win = min(rad, key=lambda um: abs(um - 10.8))
    best = None
    for name, conv in UNITS:
        bt = planck_bt(conv(rad[win][valid], win), win)
        med = float(np.median(bt)) if bt.size else np.nan
        score = abs(med - 285.0) + (0 if 200.0 <= med <= 330.0 else 1000.0)
        if best is None or score < best[0]:
            best = (score, name, conv, med)
    _, name, conv, med = best
    return {um: planck_bt(conv(r, um), um).astype(np.float32) for um, r in rad.items()}, name, med


def _local_std(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """3 x 3 standard deviation over the valid pixels only."""
    import cv2
    w = valid.astype(np.float32)
    v = np.where(valid, x, 0).astype(np.float64)
    n = np.maximum(cv2.blur(w, (3, 3)).astype(np.float64), 1e-6)
    m = cv2.blur(v, (3, 3)) / n
    return np.sqrt(np.maximum(cv2.blur(v * v, (3, 3)) / n - m * m, 0)).astype(np.float32)


def clear_sky_spatial(b108: np.ndarray, valid: np.ndarray, px_m: float, block_km: float = 100.0) -> np.ndarray:
    """Clear-sky BT10.8 from the scene itself: the 90th percentile of blocks
    of about block_km (clouds are colder than the ground around them), never
    below the scene's 60th percentile (a block under cloud), smoothed."""
    import cv2
    H, W = b108.shape
    bs = max(8, int(round(block_km * 1000.0 / max(px_m, 1.0))))
    ny, nx = max(1, int(np.ceil(H / bs))), max(1, int(np.ceil(W / bs)))
    floor = float(np.percentile(b108[valid], 60)) if valid.any() else 0.0
    grid = np.full((ny, nx), np.nan, np.float32)
    for j in range(ny):
        for i in range(nx):
            v = valid[j * bs:(j + 1) * bs, i * bs:(i + 1) * bs]
            if v.mean() >= 0.2:
                grid[j, i] = np.percentile(b108[j * bs:(j + 1) * bs, i * bs:(i + 1) * bs][v], 90)
    grid = np.where(np.isfinite(grid), np.maximum(grid, floor), floor).astype(np.float32)
    if ny * nx == 1:
        return np.full((H, W), grid[0, 0], np.float32)
    return cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)


def _blobs(mask: np.ndarray, px_m: float, min_km2: float) -> np.ndarray:
    import cv2
    if min_km2 <= 0 or not mask.any():
        return mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= max(1, int(round(min_km2 * 1e6 / max(px_m, 1e-6) ** 2)))
    keep[0] = False
    return keep[lab]


def ir_mask(bt: Dict[float, np.ndarray], valid: np.ndarray, px_m: float, clear: Optional[np.ndarray] = None,
            dt_k: float = 8.0, split_scale: float = 1.0, cold_k: float = 240.0, min_blob_km2: float = 1.0
            ) -> Tuple[np.ndarray, Dict]:
    """Classes (0 clear, 1 thick, 2 thin, 255 no data) from brightness
    temperatures {band centre um: BT}; module docstring. Returns (classes,
    the share of the valid pixels each test flags)."""
    near = (lambda um: next((k for k in bt if abs(k - um) < 0.6), None))
    k108, k12, k85, k133, k73 = near(10.8), near(12.0), near(8.5), near(13.25), near(7.35)
    if k108 is None:
        raise ValueError('infrared tests need the 10.8 um window band')
    b108 = bt[k108]
    if clear is None:
        clear = clear_sky_spatial(b108, valid, px_m)
    cold_by = clear - b108
    tests = {'cold': cold_by > dt_k, 'very cold': (cold_by > 2.0 * dt_k) | (b108 < cold_k)}
    if k12 is not None:
        tests['split'] = (b108 - bt[k12]) > np.interp(b108, SPLIT_T, SPLIT_D) * float(split_scale)
    if k85 is not None:
        tests['ice'] = (bt[k85] - b108) > 0.5
    high = np.zeros_like(valid)
    for k in (k133, k73):
        if k is not None:
            high |= (bt[k] - b108) > -8.0
    tests['high'] = high & (cold_by > 3.0)
    import cv2
    core = cv2.dilate((tests['cold'] & valid).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    tests['edges'] = (_local_std(b108, valid) > 1.5) & (cold_by > dt_k / 2.0) & core
    thick = valid & (tests['very cold'] | tests['high'])
    thin = valid & ~thick & (tests['cold'] | tests.get('split', False) | tests.get('ice', False) | tests['edges'])
    cloud = _blobs(thick | thin, px_m, min_blob_km2)
    cls = np.zeros(b108.shape, np.uint8)
    cls[thin & cloud] = 2
    cls[thick & cloud] = 1
    cls[~valid] = 255
    nv = max(int(valid.sum()), 1)
    return cls, {f'test_{k.replace(" ", "_")}_pct': round(100.0 * float((v & valid).sum()) / nv, 2)
                 for k, v in tests.items()}


def _above_ground(x: np.ndarray, k: float) -> float:
    """The median plus k times its distance down to the 16th percentile:
    brighter than the ground by k spreads (the dark half is ground)."""
    med, p16 = np.percentile(x, [50, 16])
    return float(med + k * max(med - p16, 1e-6 * max(abs(med), 1e-9)))


def swir_mask(L: Dict[int, np.ndarray], valid: np.ndarray, px_m: float, k: float = 4.0, bright_sd: float = 3.0,
              min_blob_km2: float = 1.0) -> Tuple[np.ndarray, Dict]:
    """Classes from SWIR radiances {1250, 1380, 1650, 2200 nm: band}
    (module docstring); (classes, what each test flags, or {'night': True})."""
    l125, l138 = L[1250], L[1380]
    nv = max(int(valid.sum()), 1)
    cls = np.zeros(l125.shape, np.uint8)
    cls[~valid] = 255
    top = float(np.percentile(l125[valid], 99)) if valid.any() else 0.0
    if top < NIGHT_L1250 or float(np.median(l125[valid])) <= 1e-3 * top:
        return cls, {'night': True}                        # no sunlight in the SWIR
    r = np.where(valid, l138 / np.maximum(l125, 1e-6), 0.0)
    rv = r[valid]
    r_clear = float(np.percentile(rv, 20))
    low = rv[rv <= np.percentile(rv, 50)]
    sigma = 1.4826 * float(np.median(np.abs(low - np.median(low)))) or 1e-6
    cirrus = valid & (r > r_clear + k * sigma) & (r > 1.5 * r_clear)
    bright = valid & (l125 > _above_ground(l125[valid], bright_sd))
    if 1650 in L:
        bright &= L[1650] > _above_ground(L[1650][valid], 1.0)
    thick = _blobs(bright, px_m, min_blob_km2)
    thin = _blobs(cirrus & ~thick, px_m, min_blob_km2)
    cls[thin] = 2
    cls[thick] = 1
    return cls, {'test_cirrus_pct': round(100.0 * float(cirrus.sum()) / nv, 2),
                 'test_bright_pct': round(100.0 * float(bright.sum()) / nv, 2),
                 'cirrus_ratio_clear': round(r_clear, 4)}


def nearest_bands(wl: Sequence[Optional[float]], want: Sequence[float], tol: float) -> Dict[float, int]:
    """{wanted wavelength: 1-based band} for the bands within tol of each."""
    out = {}
    for w in want:
        d = [abs((x if x is not None else -1e9) - w) for x in wl]
        i = int(np.argmin(d))
        if d[i] <= tol:
            out[w] = i + 1
    return out


def ml_band_centres(path: str) -> List[float]:
    """The band centres (um) of a thermal image: its band descriptions when
    they state them ('10.8 um', '10800 nm'), else MX-LWIR's B1-B6."""
    import re
    import rasterio as rio
    with rio.open(path) as src:
        n, desc = src.count, list(src.descriptions or [])
    got = []
    for d in desc:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(um|µm|micron|nm)', str(d or ''), re.I)
        got.append(None if not m else float(m.group(1)) / (1000.0 if m.group(2).lower() == 'nm' else 1.0))
    if len(got) == n and all(g is not None for g in got):
        return got
    return ML_UM[:n]


def clear_from_scenes(paths: Sequence[str], onto: str, scale: float = 0.01) -> Optional[np.ndarray]:
    """Clear-sky BT10.8 on the grid of `onto`: the warmest value of each pixel
    over earlier thermal scenes of the same ground (each converted to BT with
    its own unit check), or None without any."""
    import rasterio as rio
    from rasterio.warp import reproject, Resampling
    with rio.open(onto) as o:
        shape, tr, crs = (o.height, o.width), o.transform, o.crs
    best = None
    for p in paths:
        try:
            um = ml_band_centres(p)
            b = int(np.argmin([abs(u - 10.8) for u in um])) + 1
            with rio.open(p) as s:
                nd = s.nodata if s.nodata is not None else 0
                a = s.read(b).astype(np.float64)
                v = np.isfinite(a) & (a != nd)
                bt, _, _ = to_bt({um[b - 1]: a * scale}, v)
                src = np.where(v, bt[um[b - 1]], np.nan).astype(np.float32)
                out = np.full(shape, np.nan, np.float32)
                reproject(src, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                          dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
            best = out if best is None else np.fmax(best, out)
        except Exception as e:
            print(f'[Cloud] clear-sky scene {p} not used: {type(e).__name__}: {e}')
    return best
