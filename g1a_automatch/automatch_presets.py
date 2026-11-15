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
BASIS = 'provisional, as of 2026-11-15: MV 2 jobs, HV 2-11 jobs per detector, HS 3 jobs, ML none yet'

# For every sensor
COMMON: Dict = {
    'num_features': [None],                  # keypoints per window: auto
    'ransac_methods': ['magsac', 'lmeds'],
    'subcell_px': [512],
    'coarse_method': 'auto',
    'max_error_from_truth': True,            # with a truth file: the search range from it
    'channels_from_truth': True,             # ... and the band from its name (pts_band28.csv)
    'cross_top_n': 5, 'cross_min_agree': 3, 'cross_max_points': 500,
    'cloud_mask': 'on', 'cloud_mask_classes': [1],
    'stretch_percentiles': [1, 99], 'window_prep': 'auto',
    'device': 'all', 'resume': True,
}

# Per sensor: what differs. The band(s) are the user's, per run (band_hint: a
# suggestion shown beside the field). 'note' and 'band_hint' are shown in the
# GUI, not sent to the job.
INFO_KEYS = ('note', 'band_hint')
SENSORS: Dict[str, Dict] = {
    'MV': {'detectors': ['imw-eloftr', 'imw-xoftr', 'imw-xfeat-dense', 'imw-rdd-nn', 'imw-darkfeat-nn'],
           'window_sizes': [1024], 'band_hint': 'band3 (650 nm)',
           'cloud_method_MV': 'omnicloudmask', 'cloud_bands': ['band5', 'band4', 'band6'],
           'note': 'MX-VNIR: 1024 px windows (2048 above ~20 km error); clouds by OmniCloudMask '
                   'on bands 5, 4, 6. From 2 benchmarked jobs (best about 4 px): the weakest basis of the four.'},
    'HV': {'detectors': ['imw-lanet-nn', 'imw-ripe-nn', 'imw-rdd-nn', 'imw-minima-loftr', 'disk'],
           'window_sizes': [512], 'band_hint': 'the band you benchmarked (e.g. band30)',
           'note': 'HyS-VNIR: 512 px windows; clouds by the VNIR tests.'},
    'HS': {'detectors': ['imw-minima-roma', 'imw-roma', 'imw-eloftr', 'imw-topicfm', 'imw-lanet-nn'],
           'window_sizes': [512], 'band_hint': 'near 1250, 1650 or 2200 nm (e.g. band28), not 1380 / 1900 nm',
           'note': 'HyS-SWIR: 512 px windows; clouds by the SWIR tests. RoMa / MINIMA-RoMa reached 16 of 16 truth '
                   'points.'},
    'ML': {'detectors': ['imw-minima-roma', 'imw-roma', 'imw-eloftr', 'imw-topicfm', 'imw-lanet-nn'],
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


def sensor_of(path: str) -> str:
    """MV / HV / HS / ML from a G1A file name ('' when it does not say)."""
    m = re.search(r'G1A_([A-Z]{2})_', os.path.basename(path or ''))
    return m.group(1) if m else ''


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
            crispest: int = 0) -> Tuple[Dict, str]:
    """(job, note): the sensor's preset with this run's paths and the few
    things chosen in the window (search range without truth, device, clouds
    on / off) and the band(s): the user's; else, when asked, the crispest N;
    else the band in the truth file's name. No band at all is an error."""
    presets = presets or load()
    if sensor not in presets:
        raise ValueError(f'no preset for sensor {sensor!r}: one of {", ".join(presets)}')
    job = copy.deepcopy(presets[sensor])
    note = job.pop('note', '')
    job.pop('band_hint', None)
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
    return job, note


def describe(p: Dict) -> str:
    """A preset in a few lines, for the window."""
    bands = 'yours' + (f" (suggested: {p['band_hint']})" if p.get('band_hint') else '')
    cloud = 'off' if p.get('cloud_mask') in (None, '', 'none') else 'on'
    return '\n'.join([
        f"Detectors: {', '.join(p.get('detectors') or [])}",
        f"Window: {', '.join(str(w) for w in p.get('window_sizes') or [])} px    Band(s): {bands}    "
        f"RANSAC: {', '.join(p.get('ransac_methods') or [])}    Sub-cells: "
        f"{', '.join(str(s) for s in p.get('subcell_px') or []) or 'off'}",
        f"Cross-check: top {p.get('cross_top_n')}, {p.get('cross_min_agree')} agreeing, "
        f"{p.get('cross_max_points')} points for RIVAL    Clouds: {cloud}",
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
