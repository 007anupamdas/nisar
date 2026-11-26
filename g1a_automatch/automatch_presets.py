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
BASIS = ('as of 2026-11-25: MV 38 truth jobs (up to 1121), HS 19, HV 12; only methods under 1 s per window; '
         'ML uses the HS list and C3 the MV list')

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
INFO_KEYS = ('note', 'band_hint', 'top_n', 'agree')   # read by the window

# Four tiers per sensor. Each sensor's detector list holds only the methods
# under 1 s per window (median over its truth jobs), ranked best first by
# accuracy: the mean per-job percentile of truth RMSE x (truth points /
# reached)^2 (a detector's best variant per job). A tier runs the first top_n:
# fastest the top 5, fast the top 10, medium the top 20, slow all of them.
# Same RANSAC and sub-cells, so a tier reuses the results of the tiers below it.
MODES = {'fastest': {'top_n': 5, 'agree': 3},
         'fast': {'top_n': 10, 'agree': 3},
         'medium': {'top_n': 20, 'agree': 4},
         'slow': {'top_n': 30, 'agree': 4}}
MODE_ALIASES = {'accurate': 'fast', 'most accurate': 'medium'}      # the earlier names
# Auto: a ladder up the ranked list. Each step runs the top N and asks K of
# them to agree; it passes when the cross-check is reliable and the final
# points fill at least AUTO_TILE_SHARE of the sub-tiles (finest level), else
# the next step runs (the same cache: only the new detectors run).
AUTO_STEPS = [(3, 2), (5, 3), (10, 5), (20, 5)]       # (top N, K agreeing)
AUTO_TILE_SHARE = 0.8
AUTO_LOW, AUTO_HIGH = 0.8, 0.4                         # rigid share: little / strong distortion (reported)


def auto_step_name(i: int) -> str:
    n, k = AUTO_STEPS[i]
    return f'auto_top{n}_agree{k}'


def step_passes(out_dir: str, tile_share: float = AUTO_TILE_SHARE) -> Tuple[bool, str]:
    """(passed, why) of a finished auto step from its CROSS_SUMMARY.csv."""
    import pandas as pd
    try:
        c = pd.read_csv(os.path.join(out_dir, 'CROSS_SUMMARY.csv'))
    except (OSError, ValueError):
        return False, 'no cross-check result'
    if not len(c):
        return False, 'no cross-check result'
    r = c.iloc[0].to_dict()
    lv = sorted([int(m.group(1)) for k in r for m in [re.match(r'final_tiles_ok_(\d+)px$', k)] if m])
    sfx = f'_{lv[0]}px' if lv else ''
    ok, tot = r.get(f'final_tiles_ok{sfx}'), r.get(f'final_tiles{sfx}')
    if not bool(r.get('reliable')):
        return False, 'the top detectors do not agree on enough of the scene (not reliable)'
    if ok is None or tot is None or pd.isna(ok) or not tot:
        return False, 'no final points'
    share = float(ok) / float(tot)
    if share < tile_share:
        return False, f'final points in {int(ok)} of {int(tot)} sub-tiles ({100 * share:.0f} % < {100 * tile_share:.0f} %)'
    return True, f'final points in {int(ok)} of {int(tot)} sub-tiles ({100 * share:.0f} %), reliable'


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
        return 'fast', 'no rigid share (no cross-check): fast to be safe'
    if share >= AUTO_LOW:
        return None, (f'{100 * share:.0f} % of the points fit one offset: little or no distortion -- the fastest '
                      f'result stands (load its _128px_rigid or _128px final points)')
    if share >= AUTO_HIGH:
        return 'fast', f'{100 * share:.0f} % fit one offset: moderate distortion -- fast (top 10)'
    return 'medium', f'only {100 * share:.0f} % fit one offset: strong distortion -- medium (top 20)'


SENSORS: Dict[str, Dict] = {
    'MV': {'detectors': ['imw-topicfm', 'imw-sfd2-nn', 'imw-xoftr', 'disk', 'imw-dkm', 'imw-eloftr', 'imw-liftfeat-nn',
                   'imw-minima-loftr', 'imw-aliked-lg', 'dog', 'imw-darkfeat-nn', 'imw-xfeat-lg', 'keynet', 'sift',
                   'loftr', 'imw-xfeat-dense', 'hessian', 'gftt', 'imw-sp-lg', 'imw-lanet-nn', 'dedode',
                   'imw-ripe-nn', 'loftr-windoornew', 'imw-sp-nn', 'imw-alike-nn', 'loftr-windoor'],
           'window_sizes': [1024], 'band_hint': 'band3 (650 nm)',
           'cloud_method_MV': 'omnicloudmask', 'cloud_bands': ['band5', 'band4', 'band6'],
           'note': 'MX-VNIR: 1024 px windows (better than 2048 in every MV job); clouds by OmniCloudMask on bands '
                   '5, 4, 6. From jobs 1066-1069: disk, MINIMA-RoMa / -LoFTR and loftr were in the top 10 most '
                   'often; the fast list is the ones at 1-3 min per pass among the top 20.'},
    'HV': {'detectors': ['disk', 'dog', 'imw-darkfeat-nn', 'imw-ripe-nn', 'imw-rdd-nn', 'dedode', 'imw-sfd2-nn',
                   'imw-minima-loftr', 'keynet', 'imw-topicfm', 'imw-aspanformer', 'sift', 'gftt', 'imw-lanet-nn',
                   'hessian', 'imw-eloftr', 'imw-aliked-lg', 'imw-d2net-nn', 'imw-xfeat-lg', 'loftr', 'imw-xoftr',
                   'imw-rord-nn', 'imw-r2d2-nn', 'imw-liftfeat-nn', 'imw-alike-nn', 'imw-xfeat-dense',
                   'loftr-windoor', 'loftr-windoornew', 'imw-sp-lg', 'imw-sp-nn'],
           'window_sizes': [512], 'band_hint': 'the band you benchmarked (e.g. band30)',
           'note': 'HyS-VNIR: 512 px windows; clouds by the VNIR tests.'},
    'HS': {'detectors': ['imw-minima-roma', 'imw-minima-loftr', 'imw-topicfm', 'disk', 'imw-gim-roma', 'imw-xoftr',
                   'dedode', 'imw-d2net-nn', 'dog', 'imw-ripe-nn', 'imw-sfd2-nn', 'imw-aspanformer', 'imw-eloftr',
                   'imw-darkfeat-nn', 'imw-liftfeat-nn', 'sift', 'loftr', 'imw-rord-nn', 'keynet', 'imw-lanet-nn',
                   'imw-aliked-lg', 'imw-rdd-nn', 'imw-xfeat-dense', 'hessian', 'gftt', 'imw-dkm', 'imw-xfeat-lg',
                   'imw-r2d2-nn', 'imw-alike-nn', 'loftr-windoor'],
           'window_sizes': [512], 'band_hint': 'near 1250, 1650 or 2200 nm (e.g. band28), not 1380 / 1900 nm',
           'note': 'HyS-SWIR: 512 px windows; clouds by the SWIR tests. RoMa / MINIMA-RoMa reached 16 of 16 truth '
                   'points.'},
    'C3': {'detectors': ['imw-topicfm', 'imw-sfd2-nn', 'imw-xoftr', 'disk', 'imw-dkm', 'imw-eloftr', 'imw-liftfeat-nn',
                   'imw-minima-loftr', 'imw-aliked-lg', 'dog', 'imw-darkfeat-nn', 'imw-xfeat-lg', 'keynet', 'sift',
                   'loftr', 'imw-xfeat-dense', 'hessian', 'gftt', 'imw-sp-lg', 'imw-lanet-nn', 'dedode',
                   'imw-ripe-nn', 'loftr-windoornew', 'imw-sp-nn', 'imw-alike-nn', 'loftr-windoor'],
           'window_sizes': [1024], 'band_hint': 'the MX band you benchmarked (PAN: band1)',
           'cloud_method_MV': 'omnicloudmask',
           'note': 'Cartosat-3 (MX / PAN): no benchmark yet; the detectors that did best on G1A MV as a start. '
                   'References: C1 over India, L8 elsewhere (the run window chooses by the scene centre).'},
    'ML': {'detectors': ['imw-minima-roma', 'imw-minima-loftr', 'imw-topicfm', 'disk', 'imw-gim-roma', 'imw-xoftr',
                   'dedode', 'imw-d2net-nn', 'dog', 'imw-ripe-nn', 'imw-sfd2-nn', 'imw-aspanformer', 'imw-eloftr',
                   'imw-darkfeat-nn', 'imw-liftfeat-nn', 'sift', 'loftr', 'imw-rord-nn', 'keynet', 'imw-lanet-nn',
                   'imw-aliked-lg', 'imw-rdd-nn', 'imw-xfeat-dense', 'hessian', 'gftt', 'imw-dkm', 'imw-xfeat-lg',
                   'imw-r2d2-nn', 'imw-alike-nn', 'loftr-windoor'],
           'window_sizes': [512], 'band_hint': 'not band1 (7.35 um) or band6 (13.25 um)',
           'note': 'MX-LWIR: no benchmark yet, the HS list as a start; gradient window preparation (thermal against '
                   'optical); clouds by the infrared tests.'},
}


# Per detector, the matcher and variant that did best for the sensor (truth jobs:
# the best mean per-job percentile), so a run tries that one method per detector
# instead of every matcher (disk ran smnn, lgm, ada, mnn, ... before).
BEST_METHOD: Dict[str, Dict] = {
    'MV': {'matchers': {'dedode': 'lgm', 'disk': 'smnn', 'dog': 'ada', 'gftt': 'smnn', 'hessian': 'ada', 'keynet': 'lgm', 'sift': 'fginn'},
           'detector_params': {'dedode': {'detector_weights': ['L-upright'], 'descriptor_weights': ['B-upright']}, 'disk': {'checkpoint': ['epipolar']}, 'dog': {'descriptor': ['sosnet']}}},
    'HS': {'matchers': {'dedode': 'lgm', 'disk': 'smnn', 'dog': 'smnn', 'gftt': 'smnn', 'hessian': 'smnn', 'keynet': 'lgm', 'sift': 'fginn'},
           'detector_params': {'dedode': {'detector_weights': ['L-C4-v2'], 'descriptor_weights': ['B-upright']}, 'disk': {'checkpoint': ['epipolar']}, 'dog': {'descriptor': ['hardnet']}}},
    'HV': {'matchers': {'dedode': 'lgm', 'disk': 'ada', 'dog': 'mnn', 'gftt': 'mnn', 'hessian': 'smnn', 'keynet': 'ada', 'sift': 'smnn'},
           'detector_params': {'dedode': {'detector_weights': ['L-C4'], 'descriptor_weights': ['B-upright']}, 'disk': {'checkpoint': ['epipolar']}, 'dog': {'descriptor': ['hardnet8']}}},
    'C3': {'matchers': {'dedode': 'lgm', 'disk': 'smnn', 'dog': 'ada', 'gftt': 'smnn', 'hessian': 'ada', 'keynet': 'lgm', 'sift': 'fginn'},
           'detector_params': {'dedode': {'detector_weights': ['L-upright'], 'descriptor_weights': ['B-upright']}, 'disk': {'checkpoint': ['epipolar']}, 'dog': {'descriptor': ['sosnet']}}},
    'ML': {'matchers': {'dedode': 'lgm', 'disk': 'smnn', 'dog': 'smnn', 'gftt': 'smnn', 'hessian': 'smnn', 'keynet': 'lgm', 'sift': 'fginn'},
           'detector_params': {'dedode': {'detector_weights': ['L-C4-v2'], 'descriptor_weights': ['B-upright']}, 'disk': {'checkpoint': ['epipolar']}, 'dog': {'descriptor': ['hardnet']}}},
}


# The ranking the built-in lists come from; a presets file written for an
# earlier one keeps its own settings but not its detector lists.
PRESETS_VERSION = '2026-11-25'
RANKED_KEYS = ('detectors',)
GONE_KEYS = ('fast_detectors', 'slow_detectors')       # no longer used
LOAD_NOTE = ''


def load(path: Optional[str] = None) -> Dict[str, Dict]:
    """{sensor: preset} = COMMON + SENSORS + the presets file (per sensor, key
    by key). A file from an earlier ranking (no or another _version) keeps its
    settings, but the detector lists are the current ones (LOAD_NOTE says so)."""
    global LOAD_NOTE
    LOAD_NOTE = ''
    out = {s: {**copy.deepcopy(COMMON), **copy.deepcopy(v)} for s, v in SENSORS.items()}
    path = PRESETS_FILE if path is None else path
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            user = json.load(f)
        current = isinstance(user, dict) and user.get('_version') == PRESETS_VERSION
        dropped = []
        for s, v in (user or {}).items():
            if isinstance(v, dict):
                v = {k: x for k, x in v.items() if k not in GONE_KEYS}
                if not current and s in SENSORS:          # a sensor of the file's own keeps its list
                    dropped += [s for k in RANKED_KEYS if k in v and v[k] != out.get(s, {}).get(k)]
                    v = {k: x for k, x in v.items() if k not in RANKED_KEYS}
                out[s] = {**out.get(s, copy.deepcopy(COMMON)), **v}
        if dropped:
            LOAD_NOTE = (f'{os.path.basename(path)} is from an earlier ranking: its detector lists for '
                         f'{", ".join(sorted(set(dropped)))} are replaced by the current ones ({PRESETS_VERSION}); '
                         f'its other settings are kept. Write it again (Presets file…) to edit the current lists.')
    return out


INDIA_REGION = (68.0, 6.0, 98.0, 38.0)          # lon_min, lat_min, lon_max, lat_max: where the C1 references apply
META_SENSORS = {'MX-VNIR': 'MV', 'HYS-VNIR': 'HV', 'HYS-SWIR': 'HS', 'MX-LWIR': 'ML', 'MX': 'C3', 'PAN': 'C3',
                'CARTOSAT-3': 'C3', 'C3': 'C3'}


def sensor_of(path: str) -> str:
    """MV / HV / HS / ML / C3 of an input: from its G1A file name, else from
    the product's metadata beside it (BAND_META.txt: Sensor / SatID; a
    Cartosat-3 product -> C3); '' when neither says."""
    m = re.search(r'G1A_?(MV|HV|HS|ML)(?=[_0-9A-Z])', os.path.basename(path or ''))
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


_RES_KEY = re.compile(r'(resolution|gsd|pixel_?size|pixel_?spacing|spatial_?res|sampling)', re.I)


def meta_resolution(path: str) -> Tuple[Optional[float], str]:
    """(pixel size in metres, the metadata key(s) it came from) from the
    product's own metadata (BAND_META.txt / *meta*.txt beside the input):
    keys naming resolution / GSD / pixel size / spacing with a value in
    metres (along and across are averaged); (None, '') when none."""
    meta = product_meta(path)
    vals, keys = [], []
    for k, v in meta.items():
        if not _RES_KEY.search(str(k)):
            continue
        m = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(m|metres?|meters?|km)?\s*$', str(v), re.I)
        if not m:
            continue
        x = float(m.group(1)) * (1000.0 if (m.group(2) or '').lower() == 'km' else 1.0)
        if 1.0 <= x <= 5000.0:              # metres; a value in degrees is left out
            vals.append(x)
            keys.append(k)
    if not vals:
        return None, ''
    return round(float(sum(vals) / len(vals)), 3), ', '.join(keys)


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


# Pixel size (m) per G1A sensor, as set in RIVAL: the working resolution of a
# lon/lat input (an _ort / _sys product), whose pixel size is in degrees.
# MV: 6 bands 485-860 nm; HS: 256 bands 900-2500 nm; HV: 158 bands 375-1000 nm;
# ML: 6 bands 7350-13250 nm.
SENSOR_RES_M = {'MV': 42.0, 'HS': 191.0, 'HV': 320.0, 'ML': 1180.0}


# The window per sensor, and the larger one once the largest truth error passes a level:
# sensor: (window, larger window, switch above this error in km)
SENSOR_WINDOWS = {'MV': (1024, 2048, 20.0), 'HS': (512, 1024, 40.0), 'HV': (512, 1024, 80.0),
                  'ML': (512, 1024, 30.0), 'C3': (1024, 2048, 20.0)}


def window_for_truth(sensor: str, truth_csv: str, default: List[int]) -> Tuple[List[int], str]:
    """With a truth file: the sensor's window, or its larger one when the
    largest error of the manual points is above the sensor's level
    (SENSOR_WINDOWS); else the preset's. (windows, why)"""
    rule = SENSOR_WINDOWS.get(sensor)
    if not truth_csv or not rule or not os.path.exists(truth_csv):
        return list(default), ''
    try:
        import automatch_report as RP
        from automatch_job import truth_files_beside             # every band's file
        worst = max(float(RP.manual_stats(f).get('max_dist_m') or 0.0)
                    for f in (truth_files_beside(truth_csv) or [truth_csv])) / 1000.0
    except Exception:
        return list(default), ''
    small, big, level = rule
    if worst > level:
        return [big], f'window {big} px: the largest truth error {worst:.1f} km is above {level:g} km ({sensor})'
    return [small], f'window {small} px: the largest truth error {worst:.1f} km is within {level:g} km ({sensor})'


def truth_band(truth_csv: str) -> str:
    """The band a truth file's name carries (pts_band28.csv / pts_B28.csv), else ''."""
    m = re.search(r'[_\-. ](?:band_?|b)(\d+)$', os.path.splitext(os.path.basename(truth_csv or ''))[0], re.I)
    return f'band{int(m.group(1))}' if m else ''


def job_for(sensor: str, input_path: str, reference_dir: str, output_dir: str, truth_csv: str = '',
            bands: Optional[List[str]] = None, max_error_km: Optional[float] = None, device: str = '',
            cloud: Optional[bool] = None, weights_dir: str = '', presets: Optional[Dict] = None,
            crispest: int = 0, top_n: Optional[int] = None, agree: Optional[int] = None,
            mode: str = 'fast', final_points: Optional[int] = None, temp_dir: str = '',
            cell_px: Optional[int] = None, cloud_conf: Optional[float] = None,
            keypoint_test: float = 0) -> Tuple[Dict, str]:
    """(job, note): the sensor's preset with this run's paths and the few
    things chosen in the window (search range without truth, device, clouds
    on / off) and the band(s): the user's; else, when asked, the crispest N;
    else the band in the truth file's name. No band at all is an error."""
    presets = presets or load()
    if sensor not in presets:
        raise ValueError(f'no preset for sensor {sensor!r}: one of {", ".join(presets)}')
    job = copy.deepcopy(presets[sensor])
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f'mode {mode!r}: one of {", ".join(MODES)}')
    job.update(copy.deepcopy(MODES[mode]))
    note = job.pop('note', '')
    job.pop('band_hint', None)
    n = int(top_n or job.pop('top_n', 5) or 5)
    k = int(agree or job.pop('agree', 3) or 3)
    job.pop('top_n', None)
    job.pop('agree', None)
    job['detectors'] = list(job.get('detectors') or [])[:n]       # the sensor's list is ranked, best first
    bm = BEST_METHOD.get(sensor, {})              # one method per detector: its best matcher and variant
    dets = set(job['detectors'])
    if not job.get('matchers'):
        job['matchers'] = {d: [m] for d, m in bm.get('matchers', {}).items() if d in dets}
    if not job.get('detector_params'):
        job['detector_params'] = {d: v for d, v in bm.get('detector_params', {}).items() if d in dets}
    job['cross_top_n'] = min(n, len(job['detectors'])) if job['detectors'] else n
    job['cross_min_agree'] = max(1, min(k, job['cross_top_n']))
    job.update(input_path=input_path, reference_dir=reference_dir, output_dir=output_dir)
    if truth_csv:
        job['truth_csv'] = truth_csv
        job['window_sizes'], why = window_for_truth(sensor, truth_csv, job.get('window_sizes') or [1024])
        if why:
            note = (note + ' ' if note else '') + f'[{why}]'
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
    if keypoint_test:                        # each detector also at keypoint_test keypoints per megapixel
        w = int((job.get('window_sizes') or [1024])[0])
        job['window_sizes'] = [w]            # one window: a keypoint count is for one window size
        job['num_features'] = [None, int(round(float(keypoint_test) * w * w / 1e6))]
        note = (note + ' ' if note else '') + (f'[keypoint test: {w} px windows at the usual density and at '
                                               f'{keypoint_test:g} per megapixel ({job["num_features"][1]} per window)]')
    if cloud_conf:                           # a pixel is cloud only when the mask is this sure
        job['cloud_min_confidence'] = round(float(cloud_conf), 2)
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
