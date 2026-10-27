"""
Cloud masking with OmniCloudMask (Wright et al., Remote Sensing of Environment
2025; sensor-agnostic, MIT): thick cloud, thin cloud and cloud shadow from a
Red, Green and NIR band, at the image's own resolution (trained 10-50 m; not
upsampled, as its documentation asks).

Bands per G1A sensor (nearest centre wavelength to 650 / 560 / 860 nm):

  MV  MX-VNIR      B1 485, B2 555, B3 650, B4 815, B5 725, B6 860 nm -> B3, B2, B6
  HV  Hysis-VNIR   375-1000 nm, bands evenly spaced           -> nearest bands
  HS  Hysis-SWIR   900-2500 nm: no Red / Green / NIR          -> fallback below
  ML  MX-LWIR      thermal                                    -> not applicable

Band centres come from the raster's band descriptions when they state them
('650.3 nm'), else from this table. cloud_bands names three bands explicitly.

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
keypoint comes from a cloud.

Testing the mask alone, without a matching run (the model runs once; each
confidence threshold gets its own mask and quicklook):

    python automatch_cloud.py <image> --out <folder> [--conf 0.5 0.7 0.85]
        [--sensor MV] [--bands 3 2 6 | --choice quality] [--from <VNIR image>]
        [--model-dir <weights>] [--device cpu]
    (also: automatch_job.py cloud ..., and the GPU service's mode "cloud")

writes <folder>/conf_<c>/CLOUD_MASK.tif and CLOUD_QUICKLOOK.png per threshold,
and CLOUD_TEST.csv: cloud / thick / thin / shadow %, and dark_cloud_pct (the
share of "cloud" darker than the median clear ground: high means ground
taken for cloud).
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
MASK_VERSION = 2                      # 2: confidence threshold, brightness check, quicklook


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


def quicklook(arr: np.ndarray, cls: np.ndarray, valid: np.ndarray, path: str, max_side: int = 1600) -> None:
    """False colour (NIR, red, green) with thick cloud in red, thin cloud in
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
    for k, colour in ((1, (0, 0, 255)), (2, (0, 255, 255)), (3, (255, 0, 0))):
        m = c == k
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array(colour)).astype(np.uint8)
    rgb[~v] = (40, 40, 40)
    cv2.imwrite(path, rgb)


def compute(path: str, bands: List[int], out_dir: str, model_dir: str = '', device: Optional[str] = None,
            predictor=None, min_conf: float = 0.7) -> Dict:
    """CLOUD_MASK.tif and CLOUD_QUICKLOOK.png in out_dir for the image, and its
    statistics, with a check that the clouds are bright: dark_cloud_pct is the
    share of cloud pixels darker in red than the median clear pixel (real
    clouds are brighter; a large share means ground taken for cloud)."""
    import rasterio as rio
    with rio.open(path) as src:
        prof = src.profile.copy()
        nd = src.nodata if src.nodata is not None else 0
        arr = np.stack([src.read(b).astype(np.float32) for b in bands])
    valid = np.all(np.isfinite(arr) & (arr != nd), axis=0)
    cls = classify((predictor or predict)(np.where(valid, arr, nd), float(nd), model_dir, device), min_conf)
    cls = np.where(valid, cls, 255).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    prof.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    prof.pop('blockxsize', None)
    prof.pop('blockysize', None)
    mask_path = os.path.join(out_dir, 'CLOUD_MASK.tif')
    with rio.open(mask_path, 'w', **prof) as dst:
        dst.write(cls, 1)
    nv = max(int(valid.sum()), 1)
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
    return {'mask': mask_path, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
            'cloud_pct': round(pct[1] + pct[2], 2), 'dark_cloud_pct': dark, 'min_confidence': min_conf}


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


def prepare(job: Dict, scene, channels: List[str], cache_dir: str, out_dir: str, sensor: str,
            predictor=None) -> Tuple[Dict[str, List], Dict]:
    """({channel: [masked raster, 1]}, CLOUD_INFO) for a raster job with cloud
    masking on; reused from cache_dir while image, bands and settings match.
    With cloud_mask_from, the mask is made on that (VNIR) product and laid
    onto this image."""
    path = scene.raster_path
    src = job.get('cloud_mask_from') or ''
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
    key = hashlib.sha1(json.dumps({'v': MASK_VERSION, 'path': os.path.abspath(path), 'size': st.st_size,
                                   'mtime': int(st.st_mtime), 'bands': bands, 'conf': conf},
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
        info = compute(path, bands, cache_dir, model_dir_for(job), job.get('cloud_device') or None, predictor, conf)
        info.update({'key': key, 'bands': bands, 'band_choice': how, 'wavelengths_nm': wl, 'sensor': sensor,
                     'model': 'OmniCloudMask'})
        os.makedirs(cache_dir, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=1)
    else:
        print(f'[Cloud] mask reused from {cache_dir}')
    return _finish(job, path, info, channels, cache_dir, out_dir)


def _finish(job: Dict, path: str, info: Dict, channels: List[str], cache_dir: str, out_dir: str
            ) -> Tuple[Dict[str, List], Dict]:
    classes = [int(c) for c in (job.get('cloud_mask_classes') or [1, 2, 3])]
    dilate = int(job.get('cloud_dilate_px') or 0)
    key = info['key']
    print(f"[Cloud] {info['cloud_pct']:.1f} % cloud (thick {info['thick_pct']:.1f}, thin {info['thin_pct']:.1f}), "
          f"{info['shadow_pct']:.1f} % shadow ({info.get('band_choice')}; confidence >= "
          f"{info.get('min_confidence', '?')}); masking classes {classes}, grown by {dilate} px", flush=True)
    if info.get('dark_cloud_pct') is not None and info['dark_cloud_pct'] > 20:
        print(f"[Cloud] CHECK: {info['dark_cloud_pct']:.0f} % of the 'cloud' pixels are darker than the median "
              f"clear ground -- ground is probably taken for cloud. Raise cloud_min_confidence (e.g. 0.85) or "
              f"mask thick cloud only (cloud_mask_classes [1]); see CLOUD_QUICKLOOK.png", flush=True)
    ql = os.path.join(os.path.dirname(info['mask']), 'CLOUD_QUICKLOOK.png')
    if out_dir and os.path.exists(ql):
        import shutil
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(ql, os.path.join(out_dir, 'CLOUD_QUICKLOOK.png'))
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
               mask_from: str = '', margin_px: int = 10, predictor=None) -> List[Dict]:
    """The cloud mask of one image at each confidence threshold, without a
    matching run (module docstring). Rows of CLOUD_TEST.csv."""
    import rasterio as rio
    import automatch_bench as B
    stem = os.path.splitext(os.path.basename(path))[0]
    if not sensor:
        with rio.open(path) as s:
            px = abs(s.transform.a) if s.crs is not None and s.crs.is_projected else None
        sensor = B.sensor_of(stem, px)
    os.makedirs(out_dir, exist_ok=True)
    confs = sorted({round(float(c), 3) for c in confidences}) or [0.7]
    if mask_from:              # HS: the mask is made on the VNIR product of the same pass and laid on
        src_rows = standalone(mask_from, os.path.join(out_dir, 'from_' + os.path.splitext(
            os.path.basename(mask_from))[0]), '', bands, choice, confs, model_dir, device, '', 0, predictor)
        rows = []
        for r in src_rows:
            sub = os.path.join(out_dir, f"conf_{r['confidence']:g}")
            os.makedirs(sub, exist_ok=True)
            mask = transfer(r['mask'], path, os.path.join(sub, 'CLOUD_MASK.tif'), int(margin_px))
            with rio.open(mask) as m:
                cls = m.read(1)
            nv = max(int((cls != 255).sum()), 1)
            pct = {k: round(100.0 * int((cls == k).sum()) / nv, 2) for k in (1, 2, 3)}
            rows.append(dict(r, image=path, sensor=sensor, mask=mask, quicklook=r['quicklook'],
                             thick_pct=pct[1], thin_pct=pct[2], shadow_pct=pct[3],
                             cloud_pct=round(pct[1] + pct[2], 2), dark_cloud_pct=None,
                             band_choice=f"from {os.path.basename(mask_from)} ({r['band_choice']}), "
                                         f"grown {int(margin_px)} px"))
    else:
        bsel, how, wl = choose_bands(path, sensor, bands or None, choice)
        print(f'[Cloud] {os.path.basename(path)} ({sensor}): OmniCloudMask on bands {bsel} ({how}; '
              f'{", ".join(f"{w:.0f} nm" for w in wl if w)})', flush=True)
        memo: Dict = {}

        def once(arr, nd, md, dev):          # the model runs once; each threshold reads its output
            if 'p' not in memo:
                t = time.time()
                memo['p'] = (predictor or predict)(arr, nd, md, dev)
                memo['s'] = round(time.time() - t, 1)
            return memo['p']
        rows = []
        for c in confs:
            sub = os.path.join(out_dir, f'conf_{c:g}')
            info = compute(path, bsel, sub, model_dir, device, once, c)
            rows.append({'image': path, 'sensor': sensor, 'confidence': c, 'cloud_pct': info['cloud_pct'],
                         'thick_pct': info['thick_pct'], 'thin_pct': info['thin_pct'],
                         'shadow_pct': info['shadow_pct'], 'dark_cloud_pct': info['dark_cloud_pct'],
                         'bands': ' '.join(str(b) for b in bsel), 'band_choice': how,
                         'model_seconds': memo.get('s'), 'mask': info['mask'],
                         'quicklook': os.path.join(sub, 'CLOUD_QUICKLOOK.png')})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'CLOUD_TEST.csv'), index=False)
    print(f"[Cloud] {'confidence':>10} {'cloud %':>8} {'thick':>6} {'thin':>6} {'shadow':>7} {'dark cloud %':>13}")
    for r in rows:
        dk = '-' if r['dark_cloud_pct'] is None else f"{r['dark_cloud_pct']:.0f}"
        print(f"[Cloud] {r['confidence']:>10g} {r['cloud_pct']:>8.1f} {r['thick_pct']:>6.1f} {r['thin_pct']:>6.1f} "
              f"{r['shadow_pct']:>7.1f} {dk:>13}" + ('   <- ground taken for cloud?' if r['dark_cloud_pct']
                                                     and r['dark_cloud_pct'] > 20 else ''))
    print(f"[Cloud] masks and quicklooks: {out_dir}/conf_<c>/ ; table: CLOUD_TEST.csv", flush=True)
    return rows


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
    ap.add_argument('--device', default=None, help='cpu / cuda; default: GPU when there is one')
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.image)),
                                'cloud_test_' + os.path.splitext(os.path.basename(a.image))[0])
    problems = check({'cloud_model_dir': a.model_dir})
    if problems:
        print('\n'.join(problems))
        return 2
    standalone(a.image, out, a.sensor, a.bands, a.choice, a.conf, a.model_dir, a.device, a.mask_from, a.margin)
    return 0


if __name__ == '__main__':
    sys.exit(main())
