"""
Sensor-wise default settings for a production run (DPQED_automatch_run.py):
one preset per G1A sensor (MV / HV / HS / ML), taken from the benchmark of
finished jobs. DPQED_automatch.py stays the tool for benchmarking; these are
the settings found there, ready to run.

    python automatch_presets.py                 # print the presets
    python automatch_presets.py --write         # write automatch_presets.json to edit

automatch_presets.json beside this file, when present, replaces / adds to the
built-in presets (per sensor, key by key), so a new benchmark result is one
edit away: {"HS": {"detectors": [...], "window_sizes": [512]}, ...}.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_FILE = os.path.join(HERE, 'automatch_presets.json')

# What the presets rest on, shown in the run window. They are provisional:
# updated here as the logs and rankings of finished jobs come in.
BASIS = 'provisional, as of 2026-11-17: MV 4 jobs (1066-1069), HV 2-11 jobs per detector, HS 3 jobs, ML and C3 none yet'

# For every sensor
COMMON: Dict = {
    'num_features': [None],                  # keypoints per window: auto
    'ransac_methods': ['magsac', 'lmeds'],
    'subcell_px': [512],
    'coarse_method': 'auto',
    'max_error_from_truth': True,            # with a truth file: the search range from it
    'channels_from_truth': True,             # ... and the band from its name (pts_band28.csv)
    'top_n': 5,                              # the first N detectors of the sensor's list run (ranked best first)
    'agree': 3,                              # the error is where at least this many of them agree (cross-check)
    'cross_max_points': 500,
    'cross_cell_px': [64, 128],              # agreement levels: the best keypoint where the top detectors agree
    'cloud_mask': 'on', 'cloud_mask_classes': [1],
    'stretch_percentiles': [1, 99], 'window_prep': 'auto',
    'device': 'all', 'resume': True,
}

# Per sensor: what differs. The band(s) are the user's, per run (band_hint: a
# suggestion shown beside the field). 'note' and 'band_hint' are shown in the
# GUI, not sent to the job.
INFO_KEYS = ('note', 'band_hint', 'top_n', 'agree', 'fast_detectors', 'slow_detectors')   # read by the window

# Two ways to run each sensor. accurate: its ranked list, both RANSAC
# estimators, 512 px sub-cells, the error where 3 of the top 5 agree.
# fastest: its fast list (1-3 min per pass), the same RANSAC and sub-cells, the
# error where 2 of the top 3 agree: about a fifth of the time.
# most accurate: the ranked list plus the slow ones that ranked well but were
# left out for their time (DKM, RoMa, KeyNet, OmniGlue, ...), all of them run, 4
# agreeing: several times the accurate preset's time.
MODES = {'fastest': {'top_n': 3, 'agree': 2},       # same RANSAC / sub-cells: results shared between presets
         'accurate': {},
         'most accurate': {'top_n': 13, 'agree': 4}}
# Auto: the fastest preset first; its rigid share (the consistent points that
# fit one offset for the whole scene) says how distorted the product is.
AUTO_LOW, AUTO_HIGH = 0.8, 0.4


def rigid_share(out_dir: str) -> Optional[float]:
    """The rigid share a finished run's cross-check found (CROSS_SUMMARY.csv:
    the coarsest agreement level), or None."""
    import pandas as pd
    try:
        c = pd.read_csv(os.path.join(out_dir, 'CROSS_SUMMARY.csv'))
    except (OSError, ValueError):
        return None
    cols = sorted([k for k in c.columns if k.startswith('rigid_share')],
                  key=lambda k: int(re.search(r'(\d+)px', k).group(1)) if re.search(r'(\d+)px', k) else 0)
    vals = [float(c[k].dropna().iloc[0]) for k in reversed(cols) if c[k].notna().any()]
    return vals[0] if vals else None


def auto_next(share: Optional[float]) -> Tuple[Optional[str], str]:
    """(the next preset after the fastest, or None to stop; why) by the rigid share."""
    if share is None:
        return 'accurate', 'no rigid share (no cross-check): accurate to be safe'
    if share >= AUTO_LOW:
        return None, (f'{100 * share:.0f} % of the points fit one offset: little or no distortion -- the fastest '
                      f'result stands (load its _128px_rigid or _128px final points)')
    if share >= AUTO_HIGH:
        return 'accurate', f'{100 * share:.0f} % fit one offset: moderate distortion -- accurate'
    return 'most accurate', f'only {100 * share:.0f} % fit one offset: strong distortion -- most accurate'


SENSORS: Dict[str, Dict] = {
    'MV': {'detectors': ['disk', 'imw-minima-roma', 'imw-minima-loftr', 'loftr', 'imw-sfd2-nn', 'dog',
                         'imw-aliked-lg', 'imw-xfeat-lg'],
           'fast_detectors': ['loftr', 'imw-xfeat-lg', 'imw-aliked-lg', 'imw-sfd2-nn', 'imw-minima-loftr'],
           'slow_detectors': ['imw-dkm', 'imw-roma', 'keynet', 'imw-omniglue', 'imw-d2net-nn'],
           'window_sizes': [1024], 'band_hint': 'band3 (650 nm)',
           'cloud_method_MV': 'omnicloudmask', 'cloud_bands': ['band5', 'band4', 'band6'],
           'note': 'MX-VNIR: 1024 px windows (better than 2048 in every MV job); clouds by OmniCloudMask on bands '
                   '5, 4, 6. From jobs 1066-1069: disk, MINIMA-RoMa / -LoFTR and loftr were in the top 10 most '
                   'often; the fast list is the ones at 1-3 min per pass among the top 20.'},
    'HV': {'detectors': ['imw-lanet-nn', 'imw-ripe-nn', 'imw-rdd-nn', 'imw-minima-loftr', 'disk'],
           'fast_detectors': ['imw-lanet-nn', 'imw-ripe-nn', 'imw-minima-loftr', 'loftr', 'imw-xfeat-lg'],
           'slow_detectors': ['imw-roma', 'imw-minima-roma', 'imw-dkm', 'keynet', 'imw-omniglue'],
           'window_sizes': [512], 'band_hint': 'the band you benchmarked (e.g. band30)',
           'note': 'HyS-VNIR: 512 px windows; clouds by the VNIR tests.'},
    'HS': {'detectors': ['imw-minima-roma', 'imw-roma', 'imw-eloftr', 'imw-topicfm', 'imw-lanet-nn'],
           'fast_detectors': ['imw-minima-loftr', 'imw-eloftr', 'imw-lanet-nn', 'loftr', 'imw-topicfm'],
           'slow_detectors': ['imw-dkm', 'imw-gim-roma', 'keynet', 'imw-omniglue', 'disk'],
           'window_sizes': [512], 'band_hint': 'near 1250, 1650 or 2200 nm (e.g. band28), not 1380 / 1900 nm',
           'note': 'HyS-SWIR: 512 px windows; clouds by the SWIR tests. RoMa / MINIMA-RoMa reached 16 of 16 truth '
                   'points.'},
    'C3': {'detectors': ['disk', 'imw-minima-roma', 'imw-minima-loftr', 'loftr', 'imw-sfd2-nn'],
           'fast_detectors': ['loftr', 'imw-xfeat-lg', 'imw-aliked-lg', 'imw-sfd2-nn', 'imw-minima-loftr'],
           'slow_detectors': ['imw-dkm', 'imw-roma', 'keynet', 'imw-omniglue', 'imw-d2net-nn'],
           'window_sizes': [1024], 'band_hint': 'the MX band you benchmarked (PAN: band1)',
           'cloud_method_MV': 'omnicloudmask',
           'note': 'Cartosat-3 (MX / PAN): no benchmark yet; the detectors that did best on G1A MV as a start. '
                   'References: C1 over India, L8 elsewhere (the run window chooses by the scene centre).'},
    'ML': {'detectors': ['imw-minima-roma', 'imw-roma', 'imw-eloftr', 'imw-topicfm', 'imw-lanet-nn'],
           'fast_detectors': ['imw-minima-loftr', 'imw-eloftr', 'imw-lanet-nn', 'loftr', 'imw-topicfm'],
           'slow_detectors': ['imw-dkm', 'imw-gim-roma', 'keynet', 'imw-omniglue', 'disk'],
           'window_sizes': [512], 'band_hint': 'not band1 (7.35 um) or band6 (13.25 um)',
           'note': 'MX-LWIR: no benchmark yet, the HS list as a start; gradient window preparation (thermal against '
                   'optical); clouds by the infrared tests.'},
}


def load(path: Optional[str] = None) -> Dict[str, Dict]:
    """{sensor: preset} = COMMON + SENSORS + the presets file (per sensor, key by key)."""
    out = {s: {**copy.deepcopy(COMMON), **copy.deepcopy(v)} for s, v in SENSORS.items()}
    path = PRESETS_FILE if path is None else path
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            user = json.load(f)
        for s, v in (user or {}).items():
            if isinstance(v, dict):
                out[s] = {**out.get(s, copy.deepcopy(COMMON)), **v}
    return out


INDIA_REGION = (68.0, 6.0, 98.0, 38.0)          # lon_min, lat_min, lon_max, lat_max: where the C1 references apply
META_SENSORS = {'MX-VNIR': 'MV', 'HYS-VNIR': 'HV', 'HYS-SWIR': 'HS', 'MX-LWIR': 'ML', 'MX': 'C3', 'PAN': 'C3',
                'CARTOSAT-3': 'C3', 'C3': 'C3'}


def sensor_of(path: str) -> str:
    """MV / HV / HS / ML / C3 of an input: from its G1A file name, else from
    the product's metadata beside it (BAND_META.txt: Sensor / SatID; a
    Cartosat-3 product -> C3); '' when neither says."""
    m = re.search(r'G1A_([A-Z]{2})_', os.path.basename(path or ''))
    if m:
        return m.group(1)
    if re.search(r'(^|[_\-/\\])C3[A-Z]{2}', os.path.basename(path or '').upper()):
        return 'C3'
    meta = product_meta(path)
    for key in ('Sensor', 'SensorName', 'SatID', 'SatelliteID', 'Satellite', 'Mission'):
        v = str(meta.get(key, '')).strip().upper()
        if v in META_SENSORS:
            return META_SENSORS[v]
        if v in ('MV', 'HV', 'HS', 'ML'):
            return v
    return ''


def product_meta(path: str) -> Dict[str, str]:
    """BAND_META.txt (or *_metaFile.txt) of the product folder an input lies
    in, as key: value ({} when there is none)."""
    folder = os.path.dirname(os.path.abspath(path)) if path else ''
    if not folder or not os.path.isdir(folder):
        return {}
    import glob
    cands = [os.path.join(folder, 'BAND_META.txt')] + sorted(glob.glob(os.path.join(folder, '*meta*.txt'))) + sorted(
        glob.glob(os.path.join(folder, '*META*.txt')))
    for p in cands:
        if os.path.isfile(p):
            try:
                import automatch_target as TG
                return TG.read_band_meta(p)
            except Exception:
                continue
    return {}


def scene_centre(path: str) -> Optional[Tuple[float, float]]:
    """(lon, lat) of the input's centre: from its metadata, else its raster bounds."""
    meta = product_meta(path)
    for lon_k, lat_k in (('StripCenterLon', 'StripCenterLat'), ('StripCentreLon', 'StripCentreLat'),
                         ('SceneCenterLon', 'SceneCenterLat'), ('ProdCenterLon', 'ProdCenterLat')):
        try:
            return float(meta[lon_k]), float(meta[lat_k])
        except (KeyError, ValueError):
            pass
    try:
        import rasterio as rio
        from pyproj import CRS, Transformer
        with rio.open(path) as src:
            b, crs = src.bounds, src.crs
        x, y = (b.left + b.right) / 2.0, (b.top + b.bottom) / 2.0
        if crs is None:
            return None
        if CRS.from_user_input(crs).is_geographic:
            return x, y
        return Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True).transform(x, y)
    except Exception:
        return None


def reference_for(path: str, ref_l8: str, ref_c1: str = '', region=INDIA_REGION) -> Tuple[str, str]:
    """(reference folder, why): the C1 references when the scene centre lies
    in the Indian region and a C1 folder is given, else the L8 ones."""
    if not ref_c1:
        return ref_l8, 'L8 (no C1 folder given)'
    c = scene_centre(path)
    if c is None:
        return ref_l8, 'L8 (scene centre unknown)'
    lon, lat = c
    if region[0] <= lon <= region[2] and region[1] <= lat <= region[3]:
        return ref_c1, f'C1 (scene centre {lon:.2f}, {lat:.2f} in the Indian region)'
    return ref_l8, f'L8 (scene centre {lon:.2f}, {lat:.2f} outside the Indian region)'


def parse_levels(text: str) -> List[int]:
    """'64, 128' -> [64, 128]; '' or '0' -> [0] (off)."""
    out = [int(float(t)) for t in re.split(r'[,\s;]+', str(text or '').strip()) if t]
    return out or [0]


def parse_bands(text: str) -> List[str]:
    """'28, band30 5' -> ['band28', 'band30', 'band5']."""
    out = []
    for t in re.split(r'[,\s;]+', str(text or '').strip()):
        if t:
            out.append(f'band{int(t)}' if t.isdigit() else t)
    return out


def truth_band(truth_csv: str) -> str:
    """The band a truth file's name carries (pts_band28.csv / pts_B28.csv), else ''."""
    m = re.search(r'[_\-. ](?:band_?|b)(\d+)$', os.path.splitext(os.path.basename(truth_csv or ''))[0], re.I)
    return f'band{int(m.group(1))}' if m else ''


def job_for(sensor: str, input_path: str, reference_dir: str, output_dir: str, truth_csv: str = '',
            bands: Optional[List[str]] = None, max_error_km: Optional[float] = None, device: str = '',
            cloud: Optional[bool] = None, weights_dir: str = '', presets: Optional[Dict] = None,
            crispest: int = 0, top_n: Optional[int] = None, agree: Optional[int] = None,
            mode: str = 'accurate', final_points: Optional[int] = None, temp_dir: str = '',
            cell_px: Optional[int] = None) -> Tuple[Dict, str]:
    """(job, note): the sensor's preset with this run's paths and the few
    things chosen in the window (search range without truth, device, clouds
    on / off) and the band(s): the user's; else, when asked, the crispest N;
    else the band in the truth file's name. No band at all is an error."""
    presets = presets or load()
    if sensor not in presets:
        raise ValueError(f'no preset for sensor {sensor!r}: one of {", ".join(presets)}')
    job = copy.deepcopy(presets[sensor])
    if mode not in MODES:
        raise ValueError(f'mode {mode!r}: one of {", ".join(MODES)}')
    fast, slow = job.pop('fast_detectors', None), job.pop('slow_detectors', None)
    if mode == 'fastest' and fast:
        job['detectors'] = list(fast)
    elif mode == 'most accurate' and slow:
        job['detectors'] = list(job['detectors']) + [d for d in slow if d not in job['detectors']]
    job.update(copy.deepcopy(MODES[mode]))
    note = job.pop('note', '')
    job.pop('band_hint', None)
    n = int(top_n or job.pop('top_n', 5) or 5)
    k = int(agree or job.pop('agree', 3) or 3)
    job.pop('top_n', None)
    job.pop('agree', None)
    job['detectors'] = list(job.get('detectors') or [])[:n]       # the sensor's list is ranked, best first
    job['cross_top_n'] = min(n, len(job['detectors'])) if job['detectors'] else n
    job['cross_min_agree'] = max(1, min(k, job['cross_top_n']))
    job.update(input_path=input_path, reference_dir=reference_dir, output_dir=output_dir)
    if truth_csv:
        job['truth_csv'] = truth_csv
    if bands:                                # the user's bands
        job.update(channels=list(bands), auto_bands=0, channels_from_truth=False)
    elif crispest:                           # asked for: the N bands with the most detail
        job.update(channels=[], auto_bands=int(crispest), channels_from_truth=False)
    elif truth_csv and truth_band(truth_csv):
        job.update(channels=[], auto_bands=0, channels_from_truth=True)
    else:
        raise ValueError('name the band(s) to match (or tick crispest, or use a truth file named ..._band<n>.csv)')
    if max_error_km:
        job['max_expected_error_m'] = float(max_error_km) * 1000.0
    if device:
        job['device'] = device
    if cloud is not None:
        job['cloud_mask'] = 'on' if cloud else 'none'
    if weights_dir:
        job['weights_cache_dir'] = weights_dir
    if final_points is not None:
        job['final_points'] = int(final_points)
    if cell_px is not None:
        job['cross_cell_px'] = [int(v) for v in (cell_px if isinstance(cell_px, (list, tuple)) else [cell_px])]
    if temp_dir:
        job['temp_dir'] = temp_dir
    return job, note


def describe(p: Dict) -> str:
    """A preset in a few lines, for the window."""
    bands = 'yours' + (f" (suggested: {p['band_hint']})" if p.get('band_hint') else '')
    cloud = 'off' if p.get('cloud_mask') in (None, '', 'none') else 'on'
    return '\n'.join([
        f"Detectors (ranked; the first {p.get('top_n', 5)} run, the error where {p.get('agree', 3)} agree): "
        f"{', '.join(p.get('detectors') or [])}",
        f"Window: {', '.join(str(w) for w in p.get('window_sizes') or [])} px    Band(s): {bands}    "
        f"RANSAC: {', '.join(p.get('ransac_methods') or [])}    Sub-cells: "
        f"{', '.join(str(s) for s in p.get('subcell_px') or []) or 'off'}",
        f"Points for RIVAL: {p.get('cross_max_points')}    Clouds: {cloud}",
        p.get('note', '')])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--write' in argv:
        if os.path.exists(PRESETS_FILE):
            print(f'{PRESETS_FILE} exists; edit it (delete it to start again from the built-in presets)')
            return 1
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(load(''), f, indent=2)
        print(f'written: {PRESETS_FILE}')
        return 0
    for s, p in load().items():
        print(f'== {s}\n{describe(p)}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
