"""
Cloud masking with OmniCloudMask (Wright et al., Remote Sensing of Environment
2025; sensor-agnostic, MIT): thick cloud, thin cloud and cloud shadow from a
Red, Green and NIR band, at the image's own resolution (trained 10-50 m; not
upsampled, as its documentation asks).

Bands per G1A sensor (nearest centre wavelength to 650 / 560 / 860 nm):

  MV  MX-VNIR      B1 485, B2 555, B3 650, B4 815, B5 725, B6 860 nm -> B3, B2, B6
  HV  Hysis-VNIR   375-1000 nm, bands evenly spaced           -> nearest bands
  HS  Hysis-SWIR   900-2500 nm: no Red / Green / NIR          -> fallback below
  ML  MX-LWIR      thermal                                    -> infrared tests
  HS                                                          -> SWIR tests

Band centres come from the raster's band descriptions when they state them
('650.3 nm'), else from this table. cloud_bands names three bands explicitly.

cloud_method 'auto' (default) masks ML by the infrared tests and HS by the
SWIR tests of automatch_cloudphys (physics, no model: brightness
temperatures, split window, ice, CO2 / water-vapour bands; the 1.38 um
cirrus band and bright SWIR blobs), the other sensors by OmniCloudMask;
'omnicloudmask', 'ir' or 'swir' choose one for every image. Radiance =
pixel value x radiance_scale (0.01: G1A stores scaled radiance). Each
payload covers its own ground, so every image gets its own mask; with
cloud_mask_from (the VNIR product of the same pass), that mask is added to
an ML / HS image's own where it covers it.

HS (SWIR only) cannot be masked from its own bands: on two cloudy Sentinel-2
scenes over India, OmniCloudMask fed the bands inside 900-2500 nm (945, 1610,
2190 nm) found 0-3 % cloud where there was 52-59 % (IoU 0.00-0.04), while
red / green / NIR at the same 180 m reached IoU 0.68-0.81. HS and HV are
acquired together (same time stamp in the product name), so cloud_mask_from
takes the VNIR product of the same pass: its mask is computed there and laid
onto this image by map coordinates (grown by cloud_dilate_px plus
cloud_from_margin_px for the few km their geolocation differs).

The mask (CLOUD_MASK.tif: 0 clear, 1 thick, 2 thin, 3 shadow, 255 no data)
and its statistics (CLOUD_INFO.json) are kept in the job's cache and reused
while the image, bands and settings are the same. With cloud masking on, each
band the job matches is copied with the masked pixels (cloud_mask_classes,
grown by cloud_dilate_px) set to no data, and the job matches that copy, so no
keypoint comes from a cloud. By default only thick cloud is masked
(cloud_mask_classes [1]).

Bright blobs only (cloud_refine, on by default): the model standardises each
band per 1000 px tile, so in a cloud-free tile the brightest ground looks
like cloud to it; on G1A MV it took water and vegetation for cloud at every
confidence from 0.2 to 0.9. Its clouds are kept only where they are bright
-- above the cloud_bright_pct (90) percentile of the clear ground in both
visible bands -- and form a blob of at least cloud_min_blob_km2 (1 km2);
shadows only within cloud_shadow_near_km (10 km) of such a blob (refine()).

Testing the mask alone, without a matching run (the model runs once; each
confidence threshold gets its own mask and quicklook):

    python automatch_cloud.py <image> --out <folder> [--conf 0.5 0.7 0.85]
        [--sensor MV] [--bands 3 2 6 | --choice quality] [--from <VNIR image>]
        [--model-dir <weights>] [--device cpu] [--model-only]
        [--bright-pct 90 --min-blob-km2 1 --shadow-near-km 10]
        [--method auto|omnicloudmask|ir|swir] [--radiance-scale 0.01]
        [--ir-dt 8 --ir-split 1 --ir-cold 240 --clear <earlier ML scenes>]
        [--swir-k 4 --swir-bright 3]
    (also: automatch_job.py cloud ..., and the GPU service's mode "cloud")

Every run gets its own folder, named by when it ran and its bands, so runs
never overwrite each other: <folder>/20261028_101500_bands_3-2-6/conf_<c>/
CLOUD_MASK.tif, CLOUD_QUICKLOOK.png (and CLOUD_QUICKLOOK_MODEL.png, the
model's own result) and CLOUD_PARAMS_20261028_101500.txt (how it was made:
image, bands, confidence, the bright-blob settings, model, code version, and
what it found), CLOUD_TEST.csv there, and <folder>/CLOUD_TEST_ALL.csv
listing every run made in the folder: cloud /
thick / thin / shadow %, what the model alone said, what the bright-blob
check removed, and dark_cloud_pct (the share of "cloud" darker than the
median clear ground: high means ground taken for cloud).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

SENSOR_BANDS_NM = {
    'MV': [485.0, 555.0, 650.0, 815.0, 725.0, 860.0],
    'HV': (375.0, 1000.0),
    'HS': (900.0, 2500.0),
    'ML': None,                       # thermal (7.35-13.25 um)
}
TARGET_NM = {'red': 650.0, 'green': 560.0, 'nir': 860.0}
MAX_OFFSET_NM = 40.0                  # a band further than this from the target does not stand in for it
CLASS_NAMES = {0: 'clear', 1: 'thick cloud', 2: 'thin cloud', 3: 'cloud shadow'}
MASK_VERSION = 3                      # 2: confidence threshold, brightness check, quicklook; 3: bright blobs


def band_wavelengths(path: str, sensor: str) -> Optional[List[float]]:
    """Centre wavelength (nm) of every band: from the band descriptions when
    they state one, else SENSOR_BANDS_NM (a range is spread evenly over the
    raster's bands). None when unknown."""
    import rasterio as rio
    with rio.open(path) as src:
        n = src.count
        desc = list(src.descriptions or [])
    got = []
    for d in desc:
        m = re.search(r'(\d{3,4}(?:\.\d+)?)\s*nm', str(d or ''), re.I)
        got.append(float(m.group(1)) if m else None)
    if len(got) == n and all(v is not None for v in got):
        return got
    spec = SENSOR_BANDS_NM.get(sensor)
    if spec is None:
        return None
    if isinstance(spec, list):
        return list(spec) if len(spec) == n else None
    lo, hi = spec
    return list(np.linspace(lo, hi, n)) if n > 1 else [lo]


def rgn_from_wavelengths(wl: Optional[List[float]]) -> Optional[List[int]]:
    """1-based Red, Green, NIR bands nearest 650 / 560 / 860 nm, or None when
    one of them is further than MAX_OFFSET_NM away."""
    if not wl:
        return None
    out = []
    for key in ('red', 'green', 'nir'):
        d = [abs(w - TARGET_NM[key]) for w in wl]
        i = int(np.argmin(d))
        if d[i] > MAX_OFFSET_NM:
            return None
        out.append(i + 1)
    return out


def choose_bands(path: str, sensor: str, override: Optional[List[str]] = None,
                 choice: str = 'spec') -> Tuple[List[int], str, List]:
    """(the three bands, how they were chosen, their wavelengths).
    choice 'spec': red / green / nir by the sensor's band centres (default);
    'quality': the three crispest bands (band_quality), short to long
    wavelength, whatever their wavelength. OmniCloudMask takes exactly three
    bands (it was trained on red, green, nir), so not four. cloud_bands
    (override) wins over both."""
    import automatch_engine as E
    wl = band_wavelengths(path, sensor)
    if override:
        b = [int(re.sub(r'\D', '', str(x)) or 0) for x in override]
        if len(b) != 3 or min(b) < 1:
            raise ValueError(f'cloud_bands must name three bands, e.g. ["band3", "band2", "band6"]; got {override}')
        return b, 'cloud_bands', [wl[i - 1] if wl else None for i in b]
    if str(choice).lower() == 'quality':
        picked = [int(c[4:]) for c in E.pick_bands(E.band_quality(path), 3)]
        if len(picked) < 3:
            raise ValueError(f'no three usable bands for cloud masking in {os.path.basename(path)}')
        picked.sort(key=lambda b: (wl[b - 1] if wl else b))
        out_wl = [wl[i - 1] if wl else None for i in picked]
        how = 'the three crispest bands'
        if wl and rgn_from_wavelengths(wl) is None:
            how += ' (no red / green / nir here: the model was not trained on such bands)'
        return picked, how, out_wl
    rgn = rgn_from_wavelengths(wl)
    if rgn:
        return rgn, 'red, green, nir by wavelength', [wl[i - 1] for i in rgn]
    raise ValueError(f'{os.path.basename(path)} ({sensor}) has no red / green / nir bands, and OmniCloudMask '
                     f'finds no clouds from SWIR-only bands (IoU 0.00-0.04 on Sentinel-2). Set cloud_mask_from '
                     f'to the VNIR (HV / MV) product of the same pass, cloud_band_choice to quality (the three crispest '
                     f'bands; untested there) or cloud_bands to force three bands.')


def weight_files() -> List[str]:
    """File names of the OmniCloudMask weights its newest model needs."""
    import csv
    from omnicloudmask import download_models as D
    ver = D.get_latest_model_version()
    with D.model_index_path.open() as f:
        return [r['file_name'] for r in csv.DictReader(f) if float(r['version']) == float(ver)]


def model_dir_for(job: Dict) -> str:
    """Where the weights are: cloud_model_dir, else <weights folder>/omnicloudmask, else ''
    (OmniCloudMask's own folder)."""
    if job.get('cloud_model_dir'):
        return job['cloud_model_dir']
    if job.get('weights_cache_dir'):
        return os.path.join(job['weights_cache_dir'], 'omnicloudmask')
    return ''


def check(job: Dict) -> List[str]:
    """Problems that would stop cloud masking (for preflight)."""
    try:
        import omnicloudmask  # noqa: F401
    except Exception as e:
        return [f'cloud_mask: omnicloudmask is not installed here ({type(e).__name__}: {e}); '
                f'pip install omnicloudmask==1.8.0']
    md = model_dir_for(job)
    if not md:
        return []
    missing = [n for n in weight_files() if not os.path.exists(os.path.join(md, n))]
    return [f'cloud_mask: weights missing in {md}: {", ".join(missing)} '
            f'(https://huggingface.co/NickWright/OmniCloudMask)'] if missing else []


def predict(rgn: np.ndarray, nodata: float, model_dir: str = '', device: Optional[str] = None) -> np.ndarray:
    """OmniCloudMask class probabilities (4, H, W: clear, thick, thin, shadow)
    for a (3, H, W) array."""
    from omnicloudmask import predict_from_array
    kw = {'no_data_value': nodata, 'inference_device': device, 'export_confidence': True, 'softmax_output': True}
    if model_dir:
        kw['destination_model_dir'] = model_dir
    return np.asarray(predict_from_array(rgn.astype(np.float32), **kw))


def classify(out: np.ndarray, min_conf: float = 0.7) -> np.ndarray:
    """Classes from the model's output: probabilities (4, H, W) or classes
    (H, W). With probabilities a pixel is cloud only when P(thick) + P(thin)
    >= min_conf (thick or thin by the larger), shadow only when P(shadow) >=
    min_conf; otherwise clear. Thin cloud took haze and bright ground for
    cloud on G1A MV with the plain most-likely class."""
    out = np.asarray(out)
    if out.ndim == 2:
        return out.astype(np.uint8)
    p = out.astype(np.float32)
    cloud = (p[1] + p[2]) >= min_conf
    cls = np.zeros(p.shape[1:], np.uint8)
    cls[cloud & (p[1] >= p[2])] = 1
    cls[cloud & (p[1] < p[2])] = 2
    cls[~cloud & (p[3] >= min_conf)] = 3
    return cls


def pixel_m(transform, crs, height: int) -> float:
    """Ground size of a raster pixel in metres (degrees converted at the
    raster's middle latitude)."""
    if crs is not None and crs.is_projected:
        return float(abs(transform.a))
    lat = transform.f + transform.e * height / 2.0
    return float(abs(transform.a) * 111320.0 * np.cos(np.radians(lat)))


def refine(cls: np.ndarray, arr: np.ndarray, valid: np.ndarray, px_m: float, wl: Optional[List] = None,
           bright_pct: float = 90.0, min_blob_km2: float = 1.0, shadow_near_km: float = 10.0
           ) -> Tuple[np.ndarray, Dict]:
    """The model's clouds kept only where they are what a thick cloud looks
    like: bright -- above the bright_pct percentile of the clear ground in
    both visible bands (the two shortest wavelengths of the three) -- and in
    a connected blob of at least min_blob_km2. The model standardises each
    band per 1000 px tile, so in a cloud-free tile the brightest ground looks
    like cloud to it: on G1A MV water and vegetation were taken for cloud at
    every confidence. Shadows are kept only within shadow_near_km of a kept
    cloud (water far from any cloud is not a shadow). Returns (classes,
    what was removed, in % of the valid pixels)."""
    import cv2
    out = cls.copy()
    nv = max(int(valid.sum()), 1)
    model_cloud = np.isin(cls, (1, 2)) & valid
    order = list(np.argsort([w if w is not None else i for i, w in enumerate(wl)])) if wl and len(wl) == len(arr) \
        else list(range(len(arr)))
    vis = order[:2]
    clear = valid & (cls == 0)
    ref = clear if clear.sum() >= 1000 else valid
    bright = valid.copy()
    for i in vis:
        bright &= arr[i] > np.percentile(arr[i][ref], bright_pct)
    cand = model_cloud & bright
    min_px = max(1, int(round(min_blob_km2 * 1e6 / max(px_m, 1e-6) ** 2)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False
    cloud = keep[lab] & cand
    out[model_cloud & ~cloud] = 0
    shadow = (cls == 3) & valid
    near = np.zeros_like(shadow)
    if cloud.any() and shadow_near_km > 0:
        dist = cv2.distanceTransform((~cloud).astype(np.uint8), cv2.DIST_L2, 5) * px_m
        near = dist <= shadow_near_km * 1000.0
    elif shadow_near_km <= 0:
        near[:] = True
    out[shadow & ~near] = 0
    return out, {'removed_dark_cloud_pct': round(100.0 * float((model_cloud & ~bright).sum()) / nv, 2),
                 'removed_small_blob_pct': round(100.0 * float((cand & ~cloud).sum()) / nv, 2),
                 'removed_far_shadow_pct': round(100.0 * float((shadow & ~near).sum()) / nv, 2),
                 'blobs': int(keep.sum())}


def quicklook(arr: np.ndarray, cls: np.ndarray, valid: np.ndarray, path: str, max_side: int = 1600) -> None:
    """False colour (NIR, red, green) with thick cloud in purple, thin cloud in
    yellow and shadow in blue, for looking at locally."""
    import cv2
    step = max(1, int(np.ceil(max(cls.shape) / float(max_side))))
    a, c, v = arr[:, ::step, ::step], cls[::step, ::step], valid[::step, ::step]
    chans = []
    for b in (2, 0, 1):                       # NIR, red, green -> R, G, B
        x = a[b][v]
        lo, hi = (np.percentile(x, [2, 98]) if x.size else (0.0, 1.0))
        chans.append(np.clip((a[b] - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8))
    rgb = np.dstack(chans[::-1])              # OpenCV wants B, G, R
    for k, colour in ((1, (200, 0, 200)), (2, (0, 255, 255)), (3, (255, 0, 0))):   # B, G, R: purple, yellow, blue
        m = c == k
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array(colour)).astype(np.uint8)
    rgb[~v] = (40, 40, 40)
    cv2.imwrite(path, rgb)


def compute(path: str, bands: List[int], out_dir: str, model_dir: str = '', device: Optional[str] = None,
            predictor=None, min_conf: float = 0.7, refine_opts: Optional[Dict] = None,
            wl: Optional[List] = None, stamp: Optional[str] = None) -> Dict:
    """CLOUD_MASK.tif and CLOUD_QUICKLOOK.png in out_dir for the image, and its
    statistics, with a check that the clouds are bright: dark_cloud_pct is the
    share of cloud pixels darker in red than the median clear pixel (real
    clouds are brighter; a large share means ground taken for cloud). With
    refine_opts (refine()'s settings) only bright cloud blobs are kept; the
    model's own result is then also given (model_*_pct,
    CLOUD_QUICKLOOK_MODEL.png)."""
    import rasterio as rio
    with rio.open(path) as src:
        prof = src.profile.copy()
        nd = src.nodata if src.nodata is not None else 0
        arr = np.stack([src.read(b).astype(np.float32) for b in bands])
        px = pixel_m(src.transform, src.crs, src.height)
    valid = np.all(np.isfinite(arr) & (arr != nd), axis=0)
    cls = classify((predictor or predict)(np.where(valid, arr, nd), float(nd), model_dir, device), min_conf)
    cls = np.where(valid, cls, 255).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    nv = max(int(valid.sum()), 1)
    extra = {}
    if refine_opts is not None:
        mp = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
        extra = {'model_thick_pct': mp[1], 'model_thin_pct': mp[2], 'model_shadow_pct': mp[3],
                 'model_cloud_pct': round(mp[1] + mp[2], 2)}
        try:
            quicklook(arr, cls, valid, os.path.join(out_dir, 'CLOUD_QUICKLOOK_MODEL.png'))
        except Exception as e:
            print(f'[Cloud] model quicklook not written: {type(e).__name__}: {e}')
        cls, removed = refine(cls, arr, valid, px, wl, **refine_opts)
        cls = np.where(valid, cls, 255).astype(np.uint8)
        extra.update(removed)
        extra['refine'] = dict(refine_opts)
    prof.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    mask_path = os.path.join(out_dir, 'CLOUD_MASK.tif')
    with rio.open(mask_path, 'w', **prof) as dst:
        dst.write(cls, 1)
    pct = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
    cloud = np.isin(cls, (1, 2))
    clear = cls == 0
    dark = None
    if cloud.any() and clear.any():
        dark = round(100.0 * float((arr[0][cloud] < np.median(arr[0][clear])).mean()), 1)
    try:
        quicklook(arr, cls, valid, os.path.join(out_dir, 'CLOUD_QUICKLOOK.png'))
    except Exception as e:
        print(f'[Cloud] quicklook not written: {type(e).__name__}: {e}')
    res = {'mask': mask_path, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
           'cloud_pct': round(pct[1] + pct[2], 2), 'dark_cloud_pct': dark, 'min_confidence': min_conf, **extra}
    stamp = stamp or run_stamp()
    res['params'] = write_params(out_dir, {
        'image': os.path.abspath(path), 'bands': ', '.join(f'band{b}' for b in bands),
        'wavelengths_nm': ', '.join(f'{w:.0f}' if w else '?' for w in (wl or [])), 'pixel_m': round(px, 2),
        'confidence': min_conf, 'check': _check_text(refine_opts),
        'model_weights': model_dir or "OmniCloudMask's own folder", 'device': device or 'GPU when there is one',
        **{k: v for k, v in res.items() if k not in ('mask', 'min_confidence', 'refine')}}, stamp=stamp)
    res['stamp'] = stamp
    return res


def run_stamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


def _check_text(refine_opts: Optional[Dict]) -> str:
    if refine_opts is None:
        return "none: the model's result as it is"
    return (f"bright blobs only: brighter than the {refine_opts['bright_pct']:g}th percentile of the clear ground "
            f"in both visible bands, blobs of {refine_opts['min_blob_km2']:g} km2 or more, shadows within "
            f"{refine_opts['shadow_near_km']:g} km of a blob")


def write_params(folder: str, fields: Dict, merge: bool = False, stamp: Optional[str] = None) -> str:
    """CLOUD_PARAMS_<stamp>.txt (stamp: when the mask was made,
    YYYYmmdd_HHMMSS): how the mask in folder was made and what it found, one
    'name: value' per line, so several runs never overwrite each other's.
    merge: add to that run's file instead of starting it afresh."""
    path = os.path.join(folder, f'CLOUD_PARAMS_{stamp or run_stamp()}.txt')
    head = {'made': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': 'OmniCloudMask'}
    try:
        import automatch_job as J
        head['automatch'] = J.version_line()
    except Exception:
        pass
    keep = {}
    if merge and os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    k, sep, v = line.rstrip('\n').partition(': ')
                    if sep:
                        keep[k] = v
        except OSError:
            keep = {}
    out = {**keep, **head, **{k: v for k, v in fields.items() if v is not None and v != ''}}
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(f'{k}: {v}\n' for k, v in out.items())
    return path


REFINE_DEFAULTS = {'bright_pct': 90.0, 'min_blob_km2': 1.0, 'shadow_near_km': 10.0}


def refine_options(job: Dict) -> Optional[Dict]:
    """refine()'s settings from a job, None when cloud_refine is off."""
    if not job.get('cloud_refine', True):
        return None
    get = (lambda k, d: float(job[k]) if job.get(k) is not None else d)
    return {'bright_pct': get('cloud_bright_pct', REFINE_DEFAULTS['bright_pct']),
            'min_blob_km2': get('cloud_min_blob_km2', REFINE_DEFAULTS['min_blob_km2']),
            'shadow_near_km': get('cloud_shadow_near_km', REFINE_DEFAULTS['shadow_near_km'])}


def masked_band(path: str, band: int, mask_path: str, out_path: str, classes: List[int], dilate_px: int) -> str:
    """One band with the masked classes (grown by dilate_px) set to no data."""
    import cv2
    import rasterio as rio
    with rio.open(path) as src, rio.open(mask_path) as m:
        prof = src.profile.copy()
        nd = src.nodata if src.nodata is not None else 0
        a = src.read(band)
        cls = m.read(1)
    bad = np.isin(cls, classes)
    if dilate_px > 0:
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        bad = cv2.dilate(bad.astype(np.uint8), k).astype(bool)
    a = np.where(bad, nd, a)
    prof.update(count=1, nodata=nd, compress='deflate')
    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rio.open(out_path, 'w', **prof) as dst:
        dst.write(a, 1)
    return out_path


def transfer(mask_path: str, onto: str, out_path: str, grow_px: int = 0) -> str:
    """A cloud mask laid onto another image's grid by map coordinates (nearest
    class; masked classes grown by grow_px there)."""
    import cv2
    import rasterio as rio
    from rasterio.warp import reproject, Resampling
    with rio.open(mask_path) as m, rio.open(onto) as dst_src:
        prof = dst_src.profile.copy()
        out = np.full((dst_src.height, dst_src.width), 255, np.uint8)
        reproject(m.read(1), out, src_transform=m.transform, src_crs=m.crs, src_nodata=255,
                  dst_transform=dst_src.transform, dst_crs=dst_src.crs, dst_nodata=255,
                  resampling=Resampling.nearest)
    if grow_px > 0:
        k = np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)
        for c in (3, 2, 1):                 # thick cloud wins over thin, thin over shadow
            grown = cv2.dilate((out == c).astype(np.uint8), k).astype(bool)
            out = np.where(grown & (out != 255) & ((out == 0) | (out > c)), c, out)
    prof.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    with rio.open(out_path, 'w', **prof) as d:
        d.write(out, 1)
    return out_path


PHYS_NAMES = {'ir': 'infrared tests', 'swir': 'SWIR tests'}


def method_for(job: Dict, sensor: str, path: str = '') -> str:
    """'ocm' (OmniCloudMask, red / green / NIR), 'ir' (MX-LWIR: infrared
    tests) or 'swir' (HyS-SWIR: SWIR tests), from cloud_method; 'auto' by
    sensor (and, for an unknown sensor, by its band wavelengths)."""
    m = str(job.get('cloud_method') or 'auto').strip().lower()
    if m in ('omnicloudmask', 'ocm'):
        return 'ocm'
    if m in ('ir', 'swir'):
        return m
    if sensor == 'ML':
        return 'ir'
    if sensor == 'HS':
        return 'swir'
    if path:
        try:
            wl = band_wavelengths(path, sensor)
            if wl and all(w and w >= 3000 for w in wl):
                return 'ir'
        except Exception:
            pass
    return 'ocm'


def phys_options(job: Dict, method: str) -> Dict:
    """The settings of the infrared or SWIR tests from a job."""
    import automatch_cloudphys as P
    get = (lambda k, d: float(job[k]) if job.get(k) not in (None, '') else d)
    blob = get('cloud_min_blob_km2', 1.0)
    if method == 'ir':
        return {'dt_k': get('cloud_ir_dt_k', P.IR_DEFAULTS['dt_k']),
                'split_scale': get('cloud_ir_split_scale', P.IR_DEFAULTS['split_scale']),
                'cold_k': get('cloud_ir_cold_k', P.IR_DEFAULTS['cold_k']), 'min_blob_km2': blob}
    return {'k': get('cloud_swir_k', P.SWIR_DEFAULTS['k']),
            'bright_sd': get('cloud_swir_bright_sd', P.SWIR_DEFAULTS['bright_sd']), 'min_blob_km2': blob}


def _phys_text(method: str, opts: Dict) -> str:
    if method == 'ir':
        return (f"infrared tests: colder than clear sky by {opts['dt_k']:g} K (thick by twice that, or colder than "
                f"{opts['cold_k']:g} K), split window x {opts['split_scale']:g}, ice (8.5-10.8 um), high cloud "
                f"(13.25 / 7.35 um), edges; blobs of {opts['min_blob_km2']:g} km2 or more")
    return (f"SWIR tests: 1.38 um cirrus ratio above clear by {opts['k']:g} robust sd, thick where brighter than the "
            f"clear ground at 1.25 um by {opts['bright_sd']:g} spreads (and above it at 1.65 um); blobs of "
            f"{opts['min_blob_km2']:g} km2 or more")


def load_phys(path: str, method: str, sensor: str, scale: float = 0.01, clear_paths=()) -> Dict:
    """What the infrared or SWIR tests read from an image: {'valid', 'px',
    'prof', 'show' (3 bands for the quicklook), and 'bt', 'clear', 'unit',
    'bt108_median_k' (ir) or 'L' (swir)}."""
    import rasterio as rio
    import automatch_cloudphys as P
    with rio.open(path) as src:
        prof = src.profile.copy()
        nd = src.nodata if src.nodata is not None else 0
        px = pixel_m(src.transform, src.crs, src.height)
        if method == 'ir':
            um = P.ml_band_centres(path)
            raw = {u: src.read(i + 1).astype(np.float64) for i, u in enumerate(um)}
        else:
            wl = band_wavelengths(path, sensor) or band_wavelengths(path, 'HS')
            idx = P.nearest_bands(wl or [], [1250.0, 1380.0, 1650.0, 2200.0], 40.0)
            if 1250.0 not in idx or 1380.0 not in idx:
                raise ValueError(f'{os.path.basename(path)}: SWIR tests need bands near 1250 and 1380 nm '
                                 f'(found {sorted(idx)})')
            raw = {int(w): src.read(b).astype(np.float64) for w, b in idx.items()}
    valid = np.all(np.stack([np.isfinite(a) & (a != nd) for a in raw.values()]), axis=0)
    out = {'valid': valid, 'px': px, 'prof': prof}
    if method == 'ir':
        bt, unit, med = P.to_bt({u: a * scale for u, a in raw.items()}, valid)
        clear = P.clear_from_scenes(list(clear_paths), path, scale) if clear_paths else None
        if clear is not None:
            k108 = min(bt, key=lambda u: abs(u - 10.8))
            gaps = ~np.isfinite(clear)
            if gaps.any():
                clear = np.where(gaps, P.clear_sky_spatial(bt[k108], valid, px), clear)
        k108 = min(bt, key=lambda u: abs(u - 10.8))
        inv = np.where(valid, -bt[k108], np.nan)                   # cold (cloud) shows bright
        out.update({'bt': bt, 'clear': clear, 'unit': unit, 'bt108_median_k': round(med, 1),
                    'show': np.stack([inv, inv, inv])})
    else:
        L = {w: a * scale for w, a in raw.items()}
        out.update({'L': L, 'show': np.stack([L.get(1650, L[1250]), L[1250], L.get(2200, L[1250])])})
    return out


def compute_phys(path: str, method: str, out_dir: str, opts: Dict, sensor: str, scale: float = 0.01,
                 clear_paths=(), stamp: Optional[str] = None, data: Optional[Dict] = None) -> Dict:
    """CLOUD_MASK.tif, CLOUD_QUICKLOOK.png and CLOUD_PARAMS_<time>.txt of the
    infrared (ML) or SWIR (HS) tests (automatch_cloudphys); info as compute()."""
    import rasterio as rio
    import automatch_cloudphys as P
    d = data or load_phys(path, method, sensor, scale, clear_paths)
    valid, px = d['valid'], d['px']
    if method == 'ir':
        cls, tests = P.ir_mask(d['bt'], valid, px, d['clear'], **opts)
        extra = {'radiance_unit': d['unit'], 'bt108_median_k': d['bt108_median_k'],
                 'clear_sky': (f'warmest of {len(clear_paths)} earlier scene(s)' if d['clear'] is not None
                               else 'from the scene (90th percentile of ~100 km blocks)')}
    else:
        cls, tests = P.swir_mask(d['L'], valid, px, **opts)
        extra = {}
    os.makedirs(out_dir, exist_ok=True)
    prof = d['prof']
    prof.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    mask_path = os.path.join(out_dir, 'CLOUD_MASK.tif')
    with rio.open(mask_path, 'w', **prof) as dst:
        dst.write(cls, 1)
    nv = max(int(valid.sum()), 1)
    pct = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
    try:
        quicklook(np.nan_to_num(d['show'], nan=0.0), cls, valid, os.path.join(out_dir, 'CLOUD_QUICKLOOK.png'))
    except Exception as e:
        print(f'[Cloud] quicklook not written: {type(e).__name__}: {e}')
    res = {'mask': mask_path, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
           'cloud_pct': round(pct[1] + pct[2], 2), 'dark_cloud_pct': None, 'method': method,
           'model': PHYS_NAMES[method], **tests, **extra}
    stamp = stamp or run_stamp()
    res['params'] = write_params(out_dir, {'image': os.path.abspath(path), 'model': PHYS_NAMES[method],
                                           'check': _phys_text(method, opts), 'radiance_scale': scale,
                                           'pixel_m': round(px, 1),
                                           **{k: v for k, v in res.items() if k not in ('mask', 'model')}},
                                 stamp=stamp)
    res['stamp'] = stamp
    return res


def redraw_quicklook(data: Dict, mask_path: str) -> None:
    """CLOUD_QUICKLOOK.png next to mask_path, of load_phys() data and that mask."""
    import rasterio as rio
    with rio.open(mask_path) as m:
        cls = m.read(1)
    try:
        quicklook(np.nan_to_num(data['show'], nan=0.0), cls, data['valid'],
                  os.path.join(os.path.dirname(mask_path), 'CLOUD_QUICKLOOK.png'))
    except Exception as e:
        print(f'[Cloud] quicklook not written: {type(e).__name__}: {e}')


def union_masks(own: str, laid: str, out_path: str) -> Dict:
    """own's classes, and where the laid-on mask covers (not 255) the more
    severe of the two (thick over thin over shadow over clear); percentages."""
    import rasterio as rio
    with rio.open(own) as a, rio.open(laid) as b:
        prof, ca, cb = a.profile.copy(), a.read(1), b.read(1)
    rank = np.array([0, 3, 2, 1] + [0] * 252, np.uint8)            # severity of classes 0, 1, 2, 3
    take = (cb != 255) & (ca != 255) & (rank[cb] > rank[ca])
    out = np.where(take, cb, ca).astype(np.uint8)
    with rio.open(out_path, 'w', **prof) as d:
        d.write(out, 1)
    nv = max(int((out != 255).sum()), 1)
    pct = {k: round(100.0 * int((out == k).sum()) / nv, 2) for k in (1, 2, 3)}
    return {'mask': out_path, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
            'cloud_pct': round(pct[1] + pct[2], 2), 'covered_by_laid_pct': round(100.0 * float((cb != 255).mean()), 1)}


def prepare(job: Dict, scene, channels: List[str], cache_dir: str, out_dir: str, sensor: str,
            predictor=None) -> Tuple[Dict[str, List], Dict]:
    """({channel: [masked raster, 1]}, CLOUD_INFO) for a raster job with cloud
    masking on; reused from cache_dir while image, bands and settings match.
    With cloud_mask_from, the mask is made on that (VNIR) product and laid
    onto this image."""
    path = scene.raster_path
    src = job.get('cloud_mask_from') or ''
    method = method_for(job, sensor, path)
    if method in ('ir', 'swir'):
        return _prepare_phys(job, path, method, channels, cache_dir, out_dir, sensor, src, predictor)
    if src:
        import automatch_bench as B
        import rasterio as rio
        with rio.open(src) as s:
            sres = abs(s.transform.a) if s.crs is not None and s.crs.is_projected else None
        ssensor = B.sensor_of(os.path.splitext(os.path.basename(src))[0], sres)
        sub = dict(job, cloud_mask_from='')

        class _S:
            raster_path = src
        _, sinfo = prepare(sub, _S(), [], os.path.join(cache_dir, 'from_' + hashlib.sha1(
            os.path.abspath(src).encode()).hexdigest()[:8]), '', ssensor, predictor)
        grow = int(job.get('cloud_from_margin_px') or 0)
        os.makedirs(cache_dir, exist_ok=True)
        mask = transfer(sinfo['mask'], path, os.path.join(cache_dir, 'CLOUD_MASK.tif'), grow)
        import rasterio as rio
        with rio.open(mask) as m:
            cls = m.read(1)
        nv = max(int((cls != 255).sum()), 1)
        pct = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
        info = {'mask': mask, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
                'cloud_pct': round(pct[1] + pct[2], 2), 'bands': sinfo['bands'], 'sensor': sensor,
                'band_choice': f"from {os.path.basename(src)} ({ssensor}: {sinfo['band_choice']}), "
                               f"laid on by map coordinates, grown {grow} px",
                'wavelengths_nm': sinfo.get('wavelengths_nm'), 'model': 'OmniCloudMask',
                'key': hashlib.sha1(json.dumps([sinfo['key'], os.path.abspath(path), grow]).encode()).hexdigest()}
        return _finish(job, path, info, channels, cache_dir, out_dir)
    bands, how, wl = choose_bands(path, sensor, job.get('cloud_bands') or None,
                                  job.get('cloud_band_choice') or 'spec')
    st = os.stat(path)
    conf = float(job.get('cloud_min_confidence') or 0.7)
    ropts = refine_options(job)
    key = hashlib.sha1(json.dumps({'v': MASK_VERSION, 'path': os.path.abspath(path), 'size': st.st_size,
                                   'mtime': int(st.st_mtime), 'bands': bands, 'conf': conf, 'refine': ropts},
                                  sort_keys=True).encode()).hexdigest()
    info_path = os.path.join(cache_dir, 'CLOUD_INFO.json')
    info = None
    try:
        with open(info_path, encoding='utf-8') as f:
            info = json.load(f)
        if info.get('key') != key or not os.path.exists(info.get('mask', '')):
            info = None
    except (OSError, ValueError):
        info = None
    if info is None:
        print(f'[Cloud] OmniCloudMask on bands {bands} ({how}; {", ".join(f"{w:.0f} nm" for w in wl if w)}) ...',
              flush=True)
        info = compute(path, bands, cache_dir, model_dir_for(job), job.get('cloud_device') or None, predictor, conf,
                       ropts, wl)
        info.update({'key': key, 'bands': bands, 'band_choice': how, 'wavelengths_nm': wl, 'sensor': sensor,
                     'model': 'OmniCloudMask'})
        os.makedirs(cache_dir, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
    else:
        print(f'[Cloud] mask reused from {cache_dir}')
    return _finish(job, path, info, channels, cache_dir, out_dir)


def _prepare_phys(job: Dict, path: str, method: str, channels: List[str], cache_dir: str, out_dir: str, sensor: str,
                  src: str, predictor=None) -> Tuple[Dict[str, List], Dict]:
    """prepare() for ML (infrared tests) and HS (SWIR tests); with
    cloud_mask_from, that product's mask is added where it covers this image
    (the payloads cover different ground)."""
    opts = phys_options(job, method)
    scale = float(job.get('radiance_scale') or 0.01)
    clear_paths = [p for p in (job.get('cloud_ir_clear') or []) if p] if method == 'ir' else []
    st = os.stat(path)
    stamps = {p: int(os.stat(p).st_mtime) for p in clear_paths + ([src] if src else []) if os.path.exists(p)}
    key = hashlib.sha1(json.dumps({'v': MASK_VERSION, 'path': os.path.abspath(path), 'size': st.st_size,
                                   'mtime': int(st.st_mtime), 'method': method, 'opts': opts, 'scale': scale,
                                   'clear': clear_paths, 'from': src, 'stamps': stamps,
                                   'grow': int(job.get('cloud_from_margin_px') or 0)}, sort_keys=True).encode()
                       ).hexdigest()
    info_path = os.path.join(cache_dir, 'CLOUD_INFO.json')
    try:
        with open(info_path, encoding='utf-8') as f:
            info = json.load(f)
        if info.get('key') != key or not os.path.exists(info.get('mask', '')):
            info = None
    except (OSError, ValueError):
        info = None
    if info is not None:
        print(f'[Cloud] mask reused from {cache_dir}')
        return _finish(job, path, info, channels, cache_dir, out_dir)
    print(f'[Cloud] {PHYS_NAMES[method]} on {os.path.basename(path)} ({sensor}) ...', flush=True)
    data = load_phys(path, method, sensor, scale, clear_paths)
    info = compute_phys(path, method, cache_dir, opts, sensor, scale, clear_paths, data=data)
    how = _phys_text(method, opts)
    if method == 'ir':
        print(f"[Cloud] radiance read as {info['radiance_unit']} (x {scale:g}): median BT10.8 "
              f"{info['bt108_median_k']} K; clear sky {info['clear_sky']}", flush=True)
    if info.get('night'):
        print('[Cloud] no sunlight in the SWIR bands (night): no SWIR mask', flush=True)
    if src:
        import automatch_bench as B
        import rasterio as rio
        with rio.open(src) as s:
            sres = abs(s.transform.a) if s.crs is not None and s.crs.is_projected else None
        ssensor = B.sensor_of(os.path.splitext(os.path.basename(src))[0], sres)

        class _S:
            raster_path = src
        _, sinfo = prepare(dict(job, cloud_mask_from='', cloud_method='auto'), _S(), [],
                           os.path.join(cache_dir, 'from_' + hashlib.sha1(os.path.abspath(src).encode()).hexdigest()[:8]),
                           '', ssensor, predictor)
        grow = int(job.get('cloud_from_margin_px') or 0)
        laid = transfer(sinfo['mask'], path, os.path.join(cache_dir, 'CLOUD_MASK_LAID_ON.tif'), grow)
        u = union_masks(info['mask'], laid, info['mask'])
        info.update(u)
        redraw_quicklook(data, info['mask'])
        how += (f" + the mask from {os.path.basename(src)} ({sinfo.get('model', 'mask')}, laid on by map "
                f"coordinates, grown {grow} px) where it covers this image ({u['covered_by_laid_pct']} %)")
    info.update({'key': key, 'band_choice': how, 'sensor': sensor, 'bands': [], 'wavelengths_nm': []})
    os.makedirs(cache_dir, exist_ok=True)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1, default=str)
    return _finish(job, path, info, channels, cache_dir, out_dir)


def _finish(job: Dict, path: str, info: Dict, channels: List[str], cache_dir: str, out_dir: str
            ) -> Tuple[Dict[str, List], Dict]:
    classes = [int(c) for c in (job.get('cloud_mask_classes') or [1])]
    dilate = int(job.get('cloud_dilate_px') or 0)
    key = info['key']
    sure = (f"; confidence >= {info.get('min_confidence')}" if info.get('min_confidence') is not None else '')
    print(f"[Cloud] {info['cloud_pct']:.1f} % cloud (thick {info['thick_pct']:.1f}, thin {info['thin_pct']:.1f}), "
          f"{info['shadow_pct']:.1f} % shadow ({info.get('band_choice')}{sure}); masking classes {classes}, "
          f"grown by {dilate} px", flush=True)
    if info.get('model_cloud_pct') is not None:
        print(f"[Cloud] the model said {info['model_cloud_pct']:.1f} % cloud, {info['model_shadow_pct']:.1f} % shadow; "
              f"kept only bright blobs: {info.get('removed_dark_cloud_pct')} % not bright, "
              f"{info.get('removed_small_blob_pct')} % in small specks, {info.get('removed_far_shadow_pct')} % shadow "
              f"far from cloud dropped ({info.get('blobs')} cloud blob(s); CLOUD_QUICKLOOK_MODEL.png is the "
              f"model's own)", flush=True)
    if info.get('dark_cloud_pct') is not None and info['dark_cloud_pct'] > 20:
        print(f"[Cloud] CHECK: {info['dark_cloud_pct']:.0f} % of the 'cloud' pixels are darker than the median "
              f"clear ground -- ground is probably taken for cloud. Raise cloud_min_confidence (e.g. 0.85) or "
              f"mask thick cloud only (cloud_mask_classes [1]); see CLOUD_QUICKLOOK.png", flush=True)
    ql = os.path.join(os.path.dirname(info['mask']), 'CLOUD_QUICKLOOK.png')
    if out_dir and os.path.exists(ql):
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(ql, os.path.join(out_dir, 'CLOUD_QUICKLOOK.png'))
    if out_dir:
        pp = info.get('params') or ''
        stamp = info.get('stamp') or run_stamp()
        if os.path.exists(pp) and os.path.abspath(os.path.dirname(pp)) != os.path.abspath(out_dir):
            import shutil
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(pp, os.path.join(out_dir, os.path.basename(pp)))
        write_params(out_dir, {'image': os.path.abspath(path), 'sensor': info.get('sensor'),
                               'band_choice': info.get('band_choice'), 'mask_from': job.get('cloud_mask_from') or '',
                               'masked_classes': ', '.join({1: 'thick cloud', 2: 'thin cloud', 3: 'shadow'}.get(c, str(c))
                                                           for c in classes),
                               'grown_px': dilate,
                               'cloud_pct': info.get('cloud_pct'), 'thick_pct': info.get('thick_pct'),
                               'shadow_pct': info.get('shadow_pct')}, merge=True, stamp=stamp)
    out = {}
    for ch in channels:
        b = int(re.sub(r'\D', '', ch) or 1)
        tag = hashlib.sha1(json.dumps([key, b, classes, dilate]).encode()).hexdigest()[:10]
        mp = os.path.join(cache_dir, f'{ch}_cloudmasked_{tag}.tif')
        if not os.path.exists(mp):
            masked_band(path, b, info['mask'], mp, classes, dilate)
        out[ch] = [mp, 1]
    run_info = dict(info, classes_masked=classes, dilate_px=dilate)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'CLOUD_INFO.json'), 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=1)
    return out, run_info


def standalone(path: str, out_dir: str, sensor: str = '', bands: Optional[List[str]] = None, choice: str = 'spec',
               confidences=(0.5, 0.7, 0.85), model_dir: str = '', device: Optional[str] = None,
               mask_from: str = '', margin_px: int = 10, predictor=None,
               refine_opts: Optional[Dict] = 'default', stamp: Optional[str] = None, method: str = 'auto',
               phys_opts: Optional[Dict] = None, scale: float = 0.01, clear_paths=()) -> List[Dict]:
    """The cloud mask of one image at each confidence threshold, without a
    matching run (module docstring). refine_opts: refine()'s settings
    ('default': REFINE_DEFAULTS; None: the model alone). Every run goes into
    its own folder <out_dir>/<YYYYmmdd_HHMMSS>_bands_<b>-<b>-<b>/ (the time it
    ran, then the bands), so no run overwrites another; every run made in
    out_dir is listed in CLOUD_TEST_ALL.csv. Rows of CLOUD_TEST.csv.
    method (method_for: 'auto' by sensor): 'ir' / 'swir' make one mask by
    the infrared or SWIR tests (phys_opts: phys_options()'s; scale: the
    radiance scale factor; clear_paths: earlier thermal scenes) in
    <stamp>_<method>/, and with mask_from, also that mask with the VNIR
    image's at each confidence added where it covers this image."""
    if refine_opts == 'default':
        refine_opts = dict(REFINE_DEFAULTS)
    stamp = stamp or run_stamp()

    def run_folder(variant):
        name, k = f'{stamp}_{variant}', 1
        while os.path.exists(os.path.join(out_dir, name)):       # two runs in one second
            k += 1
            name = f'{stamp}_{variant}_{k}'
        return name
    import rasterio as rio
    import pandas as pd
    sensor = sensor or sensor_for(path)
    os.makedirs(out_dir, exist_ok=True)
    confs = sorted({round(float(c), 3) for c in confidences}) or [0.7]
    meth = method_for({'cloud_method': method}, sensor, path)
    if meth in ('ir', 'swir'):
        rows, folder = _standalone_phys(path, out_dir, sensor, meth, phys_opts, scale, clear_paths, mask_from,
                                        margin_px, stamp, run_folder,
                                        lambda: standalone(mask_from, out_dir, '', bands, choice, confs, model_dir,
                                                           device, '', 0, predictor, refine_opts, stamp))
        _save_rows(out_dir, folder, rows)
        print(f"[Cloud] {folder}:")
        print(f"[Cloud] {'mask':<34} {'cloud %':>8} {'thick':>6} {'thin':>6}")
        for r in rows:
            what = 'own' if r.get('confidence') in (None, '') else f"+ VNIR mask @ {r['confidence']:g}"
            print(f"[Cloud] {what:<34} {r['cloud_pct']:>8.1f} {r['thick_pct']:>6.1f} {r['thin_pct']:>6.1f}")
        print(f"[Cloud] mask, quicklook and CLOUD_PARAMS_{stamp}.txt: {os.path.join(out_dir, folder)} ; "
              f"every run in this folder: CLOUD_TEST_ALL.csv", flush=True)
        return rows
    if mask_from:              # HS: the mask is made on the VNIR product of the same pass and laid on
        src_rows = standalone(mask_from, out_dir, '', bands, choice, confs, model_dir, device, '', 0, predictor,
                              refine_opts, stamp)
        variant = f"from_{os.path.splitext(os.path.basename(mask_from))[0]}_{src_rows[0]['variant']}"
        folder = run_folder(variant)
        rows = []
        for r in src_rows:
            sub = os.path.join(out_dir, folder, f"conf_{r['confidence']:g}")
            os.makedirs(sub, exist_ok=True)
            mask = transfer(r['mask'], path, os.path.join(sub, 'CLOUD_MASK.tif'), int(margin_px))
            with rio.open(mask) as m:
                cls = m.read(1)
            nv = max(int((cls != 255).sum()), 1)
            pct = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
            write_params(sub, {'image': os.path.abspath(path), 'sensor': sensor, 'variant': variant,
                               'mask_from': os.path.abspath(mask_from),
                               'mask_made_as': os.path.join(os.path.dirname(r['mask']), f'CLOUD_PARAMS_{stamp}.txt'),
                               'laid_on': f'by map coordinates, grown {int(margin_px)} px',
                               'confidence': r['confidence'], 'thick_pct': pct[1], 'thin_pct': pct[2],
                               'shadow_pct': pct[3], 'cloud_pct': round(pct[1] + pct[2], 2)}, stamp=stamp)
            rows.append(dict(r, image=path, sensor=sensor, variant=variant, folder=folder, mask=mask,
                             quicklook=r['quicklook'],
                             thick_pct=pct[1], thin_pct=pct[2], shadow_pct=pct[3],
                             cloud_pct=round(pct[1] + pct[2], 2), dark_cloud_pct=None,
                             band_choice=f"from {os.path.basename(mask_from)} ({r['band_choice']}), "
                                         f"grown {int(margin_px)} px"))
    else:
        bsel, how, wl = choose_bands(path, sensor, bands or None, choice)
        # each band choice in its own folder: a second test (other bands) does not overwrite the first
        variant = f"bands_{'-'.join(str(b) for b in bsel)}" + ('' if refine_opts is not None else '_model_only')
        folder = run_folder(variant)
        print(f'[Cloud] {os.path.basename(path)} ({sensor}): OmniCloudMask on bands {bsel} ({how}; '
              f'{", ".join(f"{w:.0f} nm" for w in wl if w)})'
              + ('; bright blobs only' if refine_opts is not None else '; the model alone'), flush=True)
        memo: Dict = {}

        def once(arr, nd, md, dev):          # the model runs once; each threshold reads its output
            if 'p' not in memo:
                t = time.time()
                memo['p'] = (predictor or predict)(arr, nd, md, dev)
                memo['s'] = round(time.time() - t, 1)
            return memo['p']
        rows = []
        for c in confs:
            sub = os.path.join(out_dir, folder, f'conf_{c:g}')
            info = compute(path, bsel, sub, model_dir, device, once, c, refine_opts, wl, stamp)
            write_params(sub, {'sensor': sensor, 'variant': variant, 'band_choice': how,
                               'model_seconds': memo.get('s')}, merge=True, stamp=stamp)
            rows.append({'run': stamp, 'image': path, 'sensor': sensor, 'variant': variant, 'folder': folder,
                         'confidence': c,
                         'cloud_pct': info['cloud_pct'], 'thick_pct': info['thick_pct'],
                         'thin_pct': info['thin_pct'], 'shadow_pct': info['shadow_pct'],
                         'model_cloud_pct': info.get('model_cloud_pct'),
                         'model_shadow_pct': info.get('model_shadow_pct'),
                         'removed_dark_cloud_pct': info.get('removed_dark_cloud_pct'),
                         'removed_small_blob_pct': info.get('removed_small_blob_pct'),
                         'removed_far_shadow_pct': info.get('removed_far_shadow_pct'), 'blobs': info.get('blobs'),
                         'dark_cloud_pct': info['dark_cloud_pct'],
                         'bands': ' '.join(str(b) for b in bsel), 'band_choice': how,
                         'model_seconds': memo.get('s'), 'mask': info['mask'],
                         'quicklook': os.path.join(sub, 'CLOUD_QUICKLOOK.png'),
                         'quicklook_model': (os.path.join(sub, 'CLOUD_QUICKLOOK_MODEL.png')
                                             if refine_opts is not None else '')})
    _save_rows(out_dir, folder, rows)
    print(f"[Cloud] {folder}:")
    print(f"[Cloud] {'confidence':>10} {'cloud %':>8} {'thick':>6} {'thin':>6} {'shadow':>7} {'dark cloud %':>13}"
          f"{'   model said cloud %':>21}")
    for r in rows:
        dk = '-' if r['dark_cloud_pct'] is None or pd.isna(r['dark_cloud_pct']) else f"{r['dark_cloud_pct']:.0f}"
        ms = '' if r.get('model_cloud_pct') is None else f"{r['model_cloud_pct']:>21.1f}"
        print(f"[Cloud] {r['confidence']:>10g} {r['cloud_pct']:>8.1f} {r['thick_pct']:>6.1f} {r['thin_pct']:>6.1f} "
              f"{r['shadow_pct']:>7.1f} {dk:>13}{ms}" + ('   <- ground taken for cloud?' if r['dark_cloud_pct']
                                                         and r['dark_cloud_pct'] > 20 else ''))
    print(f"[Cloud] masks, quicklooks and CLOUD_PARAMS_{stamp}.txt: {os.path.join(out_dir, folder)}/conf_<c>/ ; "
          f"every run in this folder: CLOUD_TEST_ALL.csv", flush=True)
    return rows


def sensor_for(path: str) -> str:
    """MV / HS / HV / ML of an image, from its G1A name or pixel size."""
    import rasterio as rio
    import automatch_bench as B
    with rio.open(path) as s:
        px = abs(s.transform.a) if s.crs is not None and s.crs.is_projected else None
    return B.sensor_of(os.path.splitext(os.path.basename(path))[0], px)


def _save_rows(out_dir: str, folder: str, rows: List[Dict]) -> None:
    """CLOUD_TEST.csv of a run, and the run added to CLOUD_TEST_ALL.csv."""
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, folder, 'CLOUD_TEST.csv'), index=False)
    allp = os.path.join(out_dir, 'CLOUD_TEST_ALL.csv')          # every run made in this folder
    try:
        old = pd.read_csv(allp)
    except Exception:
        old = pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(allp, index=False)


def _standalone_phys(path: str, out_dir: str, sensor: str, method: str, opts: Optional[Dict], scale: float,
                     clear_paths, mask_from: str, margin_px: int, stamp: str, run_folder, vnir_rows
                     ) -> Tuple[List[Dict], str]:
    """standalone() for the infrared / SWIR tests: (rows, run folder)."""
    opts = dict(opts) if opts else phys_options({}, method)
    clear_paths = [p for p in (clear_paths or []) if p] if method == 'ir' else []
    stem = (lambda p: os.path.splitext(os.path.basename(p))[0])
    variant = method + (f'_plus_{stem(mask_from)}' if mask_from else '')
    folder = run_folder(variant)
    base = os.path.join(out_dir, folder)
    print(f'[Cloud] {os.path.basename(path)} ({sensor}): {PHYS_NAMES[method]}', flush=True)
    data = load_phys(path, method, sensor, scale, clear_paths)
    own = compute_phys(path, method, os.path.join(base, 'own'), opts, sensor, scale, clear_paths, stamp, data)
    if method == 'ir':
        print(f"[Cloud] radiance read as {own['radiance_unit']} (x {scale:g}): median BT10.8 "
              f"{own['bt108_median_k']} K; clear sky {own['clear_sky']}", flush=True)
    if own.get('night'):
        print('[Cloud] no sunlight in the SWIR bands (night): no SWIR mask', flush=True)
    how = _phys_text(method, opts)
    tests = {k: v for k, v in own.items() if k.startswith('test_') or k in ('radiance_unit', 'bt108_median_k',
                                                                             'clear_sky', 'night')}
    common = {'run': stamp, 'image': path, 'sensor': sensor, 'variant': variant, 'folder': folder, 'method': method,
              'band_choice': how, **{f'opt_{k}': v for k, v in opts.items()}, **tests}
    rows = [dict(common, confidence=None, cloud_pct=own['cloud_pct'], thick_pct=own['thick_pct'],
                 thin_pct=own['thin_pct'], shadow_pct=own['shadow_pct'], dark_cloud_pct=None, mask=own['mask'],
                 quicklook=os.path.join(base, 'own', 'CLOUD_QUICKLOOK.png'), quicklook_model='')]
    if mask_from:              # the VNIR image's mask, at each threshold, added where it covers this one
        for r in vnir_rows():
            sub = os.path.join(base, f"plus_vnir_conf_{r['confidence']:g}")
            os.makedirs(sub, exist_ok=True)
            laid = transfer(r['mask'], path, os.path.join(sub, 'CLOUD_MASK_LAID_ON.tif'), int(margin_px))
            u = union_masks(own['mask'], laid, os.path.join(sub, 'CLOUD_MASK.tif'))
            redraw_quicklook(data, u['mask'])
            write_params(sub, {'image': os.path.abspath(path), 'sensor': sensor, 'variant': variant,
                               'own_mask': own['mask'], 'own_made_as': own['params'],
                               'mask_from': os.path.abspath(mask_from),
                               'mask_from_made_as': os.path.join(os.path.dirname(r['mask']),
                                                                 f'CLOUD_PARAMS_{stamp}.txt'),
                               'laid_on': f'by map coordinates, grown {int(margin_px)} px, where it covers '
                                          f"({u['covered_by_laid_pct']} % of this image)",
                               'confidence': r['confidence'], **{k: u[k] for k in ('thick_pct', 'thin_pct',
                                                                                   'shadow_pct', 'cloud_pct')}},
                         stamp=stamp)
            rows.append(dict(common, confidence=r['confidence'], cloud_pct=u['cloud_pct'], thick_pct=u['thick_pct'],
                             thin_pct=u['thin_pct'], shadow_pct=u['shadow_pct'], dark_cloud_pct=None,
                             mask=u['mask'], quicklook=os.path.join(sub, 'CLOUD_QUICKLOOK.png'),
                             quicklook_model=r.get('quicklook') or '', covered_by_vnir_pct=u['covered_by_laid_pct'],
                             band_choice=f"{how} + {os.path.basename(mask_from)}'s mask ({r['band_choice']}) "
                                         f"at confidence {r['confidence']:g}, grown {int(margin_px)} px"))
    return rows, folder


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Test the cloud mask of one image, without a matching run')
    ap.add_argument('image')
    ap.add_argument('--out', default='', help='output folder (default: <image folder>/cloud_test_<image>)')
    ap.add_argument('--conf', type=float, nargs='+', default=[0.5, 0.7, 0.85], help='confidence thresholds')
    ap.add_argument('--sensor', default='', help='MV / HV / HS / ...; default from the file name or pixel size')
    ap.add_argument('--bands', nargs=3, default=None, help='three bands (red green nir), e.g. 3 2 6')
    ap.add_argument('--choice', default='spec', choices=['spec', 'quality'])
    ap.add_argument('--from', dest='mask_from', default='', help='VNIR image of the same pass (for HS)')
    ap.add_argument('--margin', type=int, default=10, help='growth (px) of a mask laid on from --from')
    ap.add_argument('--model-dir', default='', help='OmniCloudMask weights folder')
    ap.add_argument('--device', default=None,
                    help='cpu / cuda / cuda:N (that GPU only, numbered as nvidia-smi lists them); default: the first GPU')
    ap.add_argument('--model-only', action='store_true', help="the model's clouds as they are (no bright-blob check)")
    ap.add_argument('--bright-pct', type=float, default=REFINE_DEFAULTS['bright_pct'],
                    help='a cloud is brighter than this percentile of the clear ground, in both visible bands')
    ap.add_argument('--min-blob-km2', type=float, default=REFINE_DEFAULTS['min_blob_km2'],
                    help='smallest cloud blob kept')
    ap.add_argument('--shadow-near-km', type=float, default=REFINE_DEFAULTS['shadow_near_km'],
                    help='shadows kept within this of a cloud blob')
    import automatch_cloudphys as P
    ap.add_argument('--method', default='auto', choices=['auto', 'omnicloudmask', 'ir', 'swir'],
                    help='auto: infrared tests for ML, SWIR tests for HS, OmniCloudMask for the rest')
    ap.add_argument('--radiance-scale', type=float, default=0.01, help='radiance = pixel value x this (G1A: 0.01)')
    ap.add_argument('--ir-dt', type=float, default=P.IR_DEFAULTS['dt_k'], help='ML: cloud is this much (K) '
                    'colder than clear sky at 10.8 um (thick: twice)')
    ap.add_argument('--ir-split', type=float, default=P.IR_DEFAULTS['split_scale'],
                    help='ML: split-window (10.8 - 12 um) thresholds x this (higher: less thin cirrus)')
    ap.add_argument('--ir-cold', type=float, default=P.IR_DEFAULTS['cold_k'],
                    help='ML: colder than this (K) at 10.8 um is thick cloud')
    ap.add_argument('--clear', nargs='+', default=[], help='ML: earlier thermal scenes of the same ground, for '
                    'the clear sky (the warmest of each pixel)')
    ap.add_argument('--swir-k', type=float, default=P.SWIR_DEFAULTS['k'],
                    help='HS: cirrus where L1380/L1250 is this many robust sd above clear')
    ap.add_argument('--swir-bright', type=float, default=P.SWIR_DEFAULTS['bright_sd'],
                    help='HS: thick cloud brighter than the clear ground at 1250 nm by this many spreads')
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.image)),
                                'cloud_test_' + os.path.splitext(os.path.basename(a.image))[0])
    import automatch_job as J
    device = J.use_gpu(a.device) if a.device else None      # before omnicloudmask loads torch
    sensor = a.sensor or sensor_for(a.image)
    meth = method_for({'cloud_method': a.method}, sensor, a.image)
    if meth == 'ocm' or a.mask_from:
        problems = check({'cloud_model_dir': a.model_dir})
        if problems:
            print('\n'.join(problems))
            return 2
    ropts = None if a.model_only else {'bright_pct': a.bright_pct, 'min_blob_km2': a.min_blob_km2,
                                       'shadow_near_km': a.shadow_near_km}
    popts = ({'dt_k': a.ir_dt, 'split_scale': a.ir_split, 'cold_k': a.ir_cold, 'min_blob_km2': a.min_blob_km2}
             if meth == 'ir' else {'k': a.swir_k, 'bright_sd': a.swir_bright, 'min_blob_km2': a.min_blob_km2})
    standalone(a.image, out, sensor, a.bands, a.choice, a.conf, a.model_dir, device, a.mask_from, a.margin,
               refine_opts=ropts, method=a.method, phys_opts=popts, scale=a.radiance_scale, clear_paths=a.clear)
    return 0


if __name__ == '__main__':
    sys.exit(main())
