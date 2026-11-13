"""
Cloud masks from physics: for the bands OmniCloudMask cannot use (it needs
red, green and NIR) -- G1A's MX-LWIR (ML, six thermal bands) and HyS-SWIR
(HS) -- and, as OmniCloudMask did poorly on G1A MV as well, for MX-VNIR (MV)
and HyS-VNIR (HV).

ML -- infrared tests, as the operational geostationary cloud masks (GOES
ABI, Himawari AHI, Meteosat SEVIRI) use on the same channels; day and night.
  radiance   value x radiance_scale (G1A: scaled radiance, 0.001) in
             radiance_unit (G1A: W/cm2/sr/um). If that unit gives no Earth-
             like brightness temperatures (median 200-330 K at 10.8 um), the
             unit that does (W/m2/sr/um, mW/cm2/sr/um, mW/m2/sr/um or per
             wavenumber) is taken instead, with a warning in the log; 'auto'
             always takes the one nearest 285 K.
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
  the other tests. Every pixel is judged on its own: clouds come in every
  shape and size (broken cumulus, cirrus streaks, haze), so there is no
  blob-size limit unless min_blob_km2 asks for one (0 by default).

HS -- SWIR tests, relative to the scene, daylight only (night: the 99th
percentile at 1.25 um below 0.05 W/m2/sr/um, after radiance_scale and
radiance_unit; daylit ground gives about 1-20).
  cirrus     L1380 / L1250 above its clear level (the scene's 20th
             percentile) by swir_k (4) robust standard deviations of the clear
             half: water vapour hides the ground at 1.38 um and high or thin
             cloud shows (Landsat 8 band 9, Sentinel-2 B10). High terrain shows
             through too (less vapour above it).
  bright     L1250 brighter than the clear ground by swir_bright_sd (3)
             spreads -- the median plus that many times the median's distance
             down to the 16th percentile, the dark half being ground even when
             much of the scene is cloud -- with L1650 above its median plus one
             such spread (snow is dark at 1.65 um): thick cloud.
             Low cloud over
             bright ground stays hard from SWIR alone (visible + SWIR does much
             better: CloudAHSI), so with cloud_mask_from the VNIR pass's mask
             is added where it covers this image.

MV / HV -- VNIR tests (Fmask-like, without SWIR or thermal), daylight only.
  reflectance  relative: radiance / solar irradiance at the band centre
             (the unknown unit, sun elevation and Earth-Sun distance scale
             every band alike, so ratios between bands are true).
  bright     the mean of blue, green, red brighter than the clear ground by
             vis_bright_sd (3) spreads (median plus that many times its
             distance down to the 16th percentile, at least a tenth of the
             median). The ground is what fails the colour tests below
             (vegetation, soil, water), so a scene mostly under cloud still
             has a ground to compare with. With earlier scenes of
             the same ground (cloud_vis_clear: each pixel's darkest, or second
             darkest from three scenes on, every scene normalised by its own
             30th percentile), brighter than that clear sky x vis_clear_factor
             (1.3) instead -- snow and bright desert then stay clear.
  white      spectrally flat: mean |R - mean| / mean over blue, green, red,
             NIR below vis_white (0.25). Vegetation (NIR >> red), water (NIR
             << red) and coloured soil fail; cloud, snow and haze pass.
  blue       R_blue / R_red at least VIS_BLUE_RED (0.8), the colour side of
             Fmask's haze-optimised transform: cloud and haze are as bright
             in blue as in red (or brighter), bright soil, sand and roofs are
             not. Skipped without a blue band.
  thick (1): bright and white; thin (2): bright by half as much and white
             within 1.5 x vis_white; and on HV, where a band near 940 nm
             exists, high cloud by water vapour: L940 / L865 above its clear
             level by 4 robust sd (less vapour above a cloud top).
  shadow (3): NIR darker than the ground by 2 spreads, not water (NDVI above
             -0.05), within shadow_near_km (10) of a cloud blob.
  Snow is white and bright: from one scene it is taken for cloud (no SWIR);
  earlier scenes fix that.

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
IR_DEFAULTS = {'dt_k': 8.0, 'split_scale': 1.0, 'cold_k': 240.0, 'min_blob_km2': 0.0}
SWIR_DEFAULTS = {'k': 4.0, 'bright_sd': 3.0, 'min_blob_km2': 0.0}
NIGHT_L1250 = 0.05            # below this (99th percentile, after the radiance scale): no sunlight

VIS_DEFAULTS = {'bright_sd': 3.0, 'white': 0.25, 'clear_factor': 1.3, 'min_blob_km2': 0.0, 'shadow_near_km': 10.0}
VIS_NM = {'blue': 485.0, 'green': 555.0, 'red': 650.0, 'nir': 860.0}
VIS_BLUE_RED = 0.8            # cloud: R_blue / R_red at least this (bright soil, sand, roofs are redder)
# extraterrestrial solar irradiance (W m-2 um-1) by wavelength (nm), about Thuillier 2003, smoothed
SOLAR_NM = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
SOLAR_E0 = [1050, 1680, 2050, 1950, 1860, 1760, 1580, 1420, 1270, 1130, 1000, 900, 830, 740]

# radiance unit -> W m-2 sr-1 um-1, as a function of the value and the band centre (um)
UNITS = (('W/m2/sr/um', lambda v, um: v),
         ('W/cm2/sr/um', lambda v, um: v * 1e4),
         ('mW/cm2/sr/um', lambda v, um: v * 10.0),
         ('mW/m2/sr/um', lambda v, um: v * 1e-3),
         ('mW/m2/sr/cm-1', lambda v, um: v * 10.0 / um ** 2),
         ('W/m2/sr/cm-1', lambda v, um: v * 1e4 / um ** 2))


def planck_bt(L: np.ndarray, um: float) -> np.ndarray:
    """Brightness temperature (K) of spectral radiance L (W m-2 sr-1 um-1) at um."""
    L = np.maximum(np.asarray(L, np.float64), 1e-9)
    return C2 / (um * np.log1p(C1 / (um ** 5 * L)))


UNIT_ALIASES = {'w/cm2/str/um': 'W/cm2/sr/um', 'w/cm^2/sr/um': 'W/cm2/sr/um', 'w cm-2 sr-1 um-1': 'W/cm2/sr/um',
                'w/m2/str/um': 'W/m2/sr/um', 'w m-2 sr-1 um-1': 'W/m2/sr/um', 'mw/cm2/str/um': 'mW/cm2/sr/um'}


def unit_name(unit: str) -> str:
    """A radiance unit as UNITS names it ('auto' stays), or ValueError."""
    u = str(unit or 'auto').strip()
    if u.lower() == 'auto':
        return 'auto'
    u = UNIT_ALIASES.get(u.lower().replace('µ', 'u').replace('μ', 'u'), u.replace('µ', 'u').replace('μ', 'u'))
    for name, _ in UNITS:
        if name.lower() == u.lower():
            return name
    raise ValueError(f'radiance unit {unit!r}: one of auto, ' + ', '.join(n for n, _ in UNITS))


def to_si(v: np.ndarray, um: float, unit: str) -> np.ndarray:
    """Radiance in `unit` -> W m-2 sr-1 um-1 at band centre um."""
    return dict(UNITS)[unit_name(unit)](v, um)


def pick_unit(rad: Dict[float, np.ndarray], valid: np.ndarray, stated: str = 'auto') -> Tuple[str, float, str]:
    """(unit, median BT10.8, note): the stated unit when its window-band
    temperatures are Earth-like (median 200-330 K), else -- or with 'auto' --
    the unit whose median lies nearest 285 K; the note says when the stated
    unit was overruled."""
    win = min(rad, key=lambda um: abs(um - 10.8))
    x = rad[win][valid]
    med_of = (lambda conv: float(np.median(planck_bt(conv(x, win), win))) if x.size else float('nan'))
    stated = unit_name(stated)
    if stated != 'auto':
        med = med_of(dict(UNITS)[stated])
        if 200.0 <= med <= 330.0:
            return stated, med, ''
    best = None
    for name, conv in UNITS:
        med = med_of(conv)
        score = abs(med - 285.0) + (0 if 200.0 <= med <= 330.0 else 1000.0)
        if best is None or score < best[0]:
            best = (score, name, med)
    note = ''
    if stated != 'auto':
        note = (f'WARNING: read as {stated} the 10.8 um band gives a median of {med_of(dict(UNITS)[stated]):.0f} K, '
                f'not an Earth temperature; {best[1]} gives {best[2]:.0f} K and is used. Check radiance_scale and '
                f'radiance_unit.')
    return best[1], best[2], note


def to_bt(rad: Dict[float, np.ndarray], valid: np.ndarray, unit: str = 'auto'
          ) -> Tuple[Dict[float, np.ndarray], str, float]:
    """({um: BT}, unit taken, median BT10.8); the unit by pick_unit()."""
    name, med, _ = pick_unit(rad, valid, unit)
    conv = dict(UNITS)[name]
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
            dt_k: float = 8.0, split_scale: float = 1.0, cold_k: float = 240.0, min_blob_km2: float = 0.0
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
              min_blob_km2: float = 0.0) -> Tuple[np.ndarray, Dict]:
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


DESTRIPE = {'mode': 'rows', 'window': 31}     # set from the job (cloud_destripe, cloud_destripe_window)


def destripe(a: np.ndarray, valid: np.ndarray, mode: Optional[str] = None, window: Optional[int] = None) -> np.ndarray:
    """Banding removed: each image row (mode 'rows'; 'cols': each column;
    'both'; 'off') scaled to the running median, over `window` rows, of the
    row medians. Full-width bright lines (detector / line calibration) go;
    clouds taller than about half the window stay."""
    import pandas as pd
    mode = (mode or DESTRIPE['mode'] or 'off').lower()
    window = int(window or DESTRIPE['window'] or 31)
    if mode == 'off' or not valid.any():
        return a
    out = a.astype(np.float64, copy=True)
    for ax in ((1,) if mode == 'rows' else (0,) if mode == 'cols' else (1, 0)):
        x = np.where(valid, out, np.nan)
        import warnings
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # rows that are all no-data
            med = np.nanmedian(x, axis=ax)
            share = valid.mean(axis=ax)
        med = np.where(share >= 0.2, med, np.nan)
        ref = pd.Series(med).rolling(window, center=True, min_periods=max(3, window // 4)).median().to_numpy()
        with np.errstate(all='ignore'):
            g = np.clip(ref / med, 0.1, 10.0)
        g = np.where(np.isfinite(g), g, 1.0)
        out = out * (g[:, None] if ax == 1 else g[None, :])
    return np.where(valid, out, a).astype(a.dtype if a.dtype.kind == 'f' else np.float64)


def solar_e0(nm: float) -> float:
    """Extraterrestrial solar irradiance (W m-2 um-1) at nm (350-1000 nm)."""
    return float(np.interp(nm, SOLAR_NM, SOLAR_E0))


def _below_ground(x: np.ndarray, k: float) -> float:
    """The median minus k times its distance down to the 16th percentile."""
    med, p16 = np.percentile(x, [50, 16])
    return float(med - k * max(med - p16, 0.0))


def _ratio_test(num: np.ndarray, den: np.ndarray, valid: np.ndarray, k: float, factor: float) -> np.ndarray:
    """Where num / den stands above its clear level (20th percentile) by k
    robust sd of the lower half and by factor x that level."""
    r = np.where(valid, num / np.maximum(den, 1e-9), 0.0)
    rv = r[valid]
    if rv.size < 10:
        return np.zeros_like(valid)
    rc = float(np.percentile(rv, 20))
    low = rv[rv <= np.percentile(rv, 50)]
    sigma = 1.4826 * float(np.median(np.abs(low - np.median(low)))) or 1e-6
    return valid & (r > rc + k * sigma) & (r > factor * rc)


def vis_brightness(R: Dict[str, np.ndarray]) -> np.ndarray:
    """Mean relative reflectance of the visible bands there are."""
    vis = [R[k] for k in ('blue', 'green', 'red') if k in R]
    return np.mean(np.stack(vis), axis=0)


def vnir_mask(R: Dict[str, np.ndarray], valid: np.ndarray, px_m: float, clear: Optional[np.ndarray] = None,
              bright_sd: float = 3.0, white: float = 0.25, clear_factor: float = 1.3, min_blob_km2: float = 0.0,
              shadow_near_km: float = 10.0, blue_red: float = VIS_BLUE_RED) -> Tuple[np.ndarray, Dict]:
    """Classes (0 clear, 1 thick, 2 thin, 3 shadow, 255 no data) from relative
    reflectances {'blue' (optional), 'green', 'red', 'nir', 'vapour' and
    'vapour_ref' (optional: ~940 and ~865 nm)}; module docstring. clear: the
    clear-sky visible brightness from earlier scenes, normalised by its 30th
    percentile (vis_clear_from_scenes), or None. (classes, tests)."""
    import cv2
    nv = max(int(valid.sum()), 1)
    cls = np.zeros(valid.shape, np.uint8)
    cls[~valid] = 255
    m = vis_brightness(R)
    mv = m[valid]
    top = float(np.percentile(mv, 99)) if mv.size else 0.0
    if top <= 0 or float(np.median(mv)) <= 1e-3 * top:
        return cls, {'night': True}
    keys = [k for k in ('blue', 'green', 'red', 'nir') if k in R]
    stack = np.stack([R[k] for k in keys])
    mean4 = np.maximum(stack.mean(0), 1e-12)
    w = np.abs(stack - mean4).mean(0) / mean4
    blue_ok = (R['blue'] / np.maximum(R['red'], 1e-12) >= blue_red) if 'blue' in R else np.ones_like(valid)
    ground = valid & ~((w < 1.5 * white) & blue_ok)             # not cloud-coloured: vegetation, soil, water
    gv = m[ground] if ground.sum() >= max(100, 0.05 * nv) else mv

    def above(k):
        med, p16 = np.percentile(gv, [50, 16])
        return float(med + k * max(med - p16, 0.1 * med))
    if clear is not None:
        p30 = float(np.percentile(mv, 30)) or 1e-12
        mn = m / p30
        ok = np.isfinite(clear)
        floor = above(1.0)
        bright = valid & np.where(ok, mn > np.where(ok, clear, 0) * clear_factor, m > above(bright_sd))
        bright &= m > floor
        half = 1.0 + (clear_factor - 1.0) / 2.0
        thin_b = valid & np.where(ok, mn > np.where(ok, clear, 0) * half, m > above(bright_sd / 2.0)) & (m > floor)
    else:
        bright = valid & (m > above(bright_sd))
        thin_b = valid & (m > above(bright_sd / 2.0))
    is_white = (w < white) & blue_ok
    thick = bright & is_white
    thin = thin_b & (w < 1.5 * white) & blue_ok & ~thick
    tests = {'bright': bright, 'white': valid & (w < white), 'blue': valid & blue_ok, 'thin_bright': thin_b}
    if 'vapour' in R and 'vapour_ref' in R:
        high = _ratio_test(R['vapour'], R['vapour_ref'], valid, 4.0, 1.3)
        tests['vapour'] = high
        thin |= high & ~thick
    thick = _blobs(thick, px_m, min_blob_km2)
    thin = _blobs(thin & ~thick, px_m, min_blob_km2)
    cloud = thick | thin
    shadow = np.zeros_like(valid)
    if 'nir' in R and 'red' in R and cloud.any() and shadow_near_km > 0:
        ndvi = (R['nir'] - R['red']) / np.maximum(R['nir'] + R['red'], 1e-12)
        dark = valid & ~cloud & (R['nir'] < _below_ground(R['nir'][valid], 2.0)) & (ndvi > -0.05)
        dist = cv2.distanceTransform((~cloud).astype(np.uint8), cv2.DIST_L2, 5) * px_m
        shadow = dark & (dist <= shadow_near_km * 1000.0)
        tests['dark'] = dark
    cls[shadow] = 3
    cls[thin] = 2
    cls[thick] = 1
    cls[~valid] = 255
    out = {f'test_{k}_pct': round(100.0 * float((v & valid).sum()) / nv, 2) for k, v in tests.items()}
    out['whiteness_cloud_median'] = round(float(np.median(w[thick])), 3) if thick.any() else None
    return cls, out


def vnir_bands(wl: Sequence[Optional[float]]) -> Dict[str, Tuple[int, float]]:
    """{'blue'?, 'green', 'red', 'nir', 'vapour'?, 'vapour_ref'?: (1-based band,
    its centre nm)}; green, red and NIR within 40 nm or ValueError."""
    out = {}
    for k, nm in VIS_NM.items():
        d = nearest_bands(wl, [nm], 40.0)
        if d:
            b = d[nm]
            out[k] = (b, float(wl[b - 1]))
    miss = [k for k in ('green', 'red', 'nir') if k not in out]
    if miss:
        raise ValueError(f'VNIR tests need green, red and NIR bands (none within 40 nm of {", ".join(miss)})')
    v = nearest_bands(wl, [940.0], 12.0)
    r = nearest_bands(wl, [865.0], 15.0)
    if v and r:
        out['vapour'] = (v[940.0], float(wl[v[940.0] - 1]))
        out['vapour_ref'] = (r[865.0], float(wl[r[865.0] - 1]))
    return out


def read_vnir(path: str, wl: Sequence[Optional[float]]) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
    """(relative reflectances, valid, {key: (band, nm)}) of a VNIR image."""
    import rasterio as rio
    sel = vnir_bands(wl)
    with rio.open(path) as src:
        nd = src.nodata if src.nodata is not None else 0
        raw = {k: src.read(b).astype(np.float64) for k, (b, _) in sel.items()}
    valid = np.all(np.stack([np.isfinite(a) & (a != nd) for a in raw.values()]), axis=0)
    return {k: destripe(raw[k], valid) / solar_e0(sel[k][1]) for k in raw}, valid, sel


def vis_clear_from_scenes(scenes: Sequence[Tuple[str, Sequence[Optional[float]]]], onto: str
                          ) -> Optional[np.ndarray]:
    """Clear-sky visible brightness on the grid of `onto`, from earlier VNIR
    scenes [(path, band wavelengths)]: each normalised by its own 30th
    percentile, then each pixel's darkest (second darkest from three scenes
    on: a cloud shadow is darker than the ground). None without any."""
    import rasterio as rio
    from rasterio.warp import reproject, Resampling
    with rio.open(onto) as o:
        shape, tr, crs = (o.height, o.width), o.transform, o.crs
    got = []
    for p, wl in scenes:
        try:
            R, v, _ = read_vnir(p, wl)
            m = vis_brightness(R)
            p30 = float(np.percentile(m[v], 30)) if v.any() else 0.0
            if p30 <= 0:
                raise ValueError('no daylight')
            src = np.where(v, m / p30, np.nan).astype(np.float32)
            with rio.open(p) as s:
                out = np.full(shape, np.nan, np.float32)
                reproject(src, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                          dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
            got.append(out)
        except Exception as e:
            print(f'[Cloud] clear-sky scene {p} not used: {type(e).__name__}: {e}')
    if not got:
        return None
    st = np.sort(np.stack(got), axis=0)              # NaN last
    n = np.sum(np.isfinite(st), axis=0)
    second = np.where(n >= 3, st[min(1, len(got) - 1)], st[0])
    return np.where(n > 0, second, np.nan).astype(np.float32)


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


def clear_from_scenes(paths: Sequence[str], onto: str, scale: float = 0.001, unit: str = 'auto'
                      ) -> Optional[np.ndarray]:
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
                bt, _, _ = to_bt({um[b - 1]: a * scale}, v, unit)
                src = np.where(v, bt[um[b - 1]], np.nan).astype(np.float32)
                out = np.full(shape, np.nan, np.float32)
                reproject(src, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                          dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
            best = out if best is None else np.fmax(best, out)
        except Exception as e:
            print(f'[Cloud] clear-sky scene {p} not used: {type(e).__name__}: {e}')
    return best
