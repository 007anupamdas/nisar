#!/usr/bin/env python3
"""
automatch_job -- run an automatch job from a JSON file (the GUI writes the same
file and runs this script as a subprocess, so GUI and command line are one path).

    python automatch_job.py template  > job.json     # annotated defaults
    python automatch_job.py preflight job.json       # validate, no matching
    python automatch_job.py run       job.json       # match + RIVAL CSVs
    python automatch_job.py detectors                # what can be selected
    python automatch_job.py weights   [job.json]     # which weight files are on this machine
    python automatch_job.py compare   <output_dir> --truth manual.csv   # rank vs ground truth
    python automatch_job.py pack      job.json --to <dir> --server-dir <dir>  # bundle for the GPU server
    python automatch_job.py inspect   <input>        # channels, CRS, footprint

Any job key can be overridden on the command line, e.g.
    python automatch_job.py run job.json --set window_sizes=[1024,2048] --set detectors=["sift"]

Outputs (under output_dir):
    win<W>_nf<N>/<channel>_to<ref>/...   engine outputs per sweep point
    rival/RIVAL_*.csv                    one per channel x detector x sweep point,
                                         loadable in DPQED_rival.py ('Load CSV')
    RIVAL_BEST_<scene>_<channel>.csv     highest consensus score per channel
    RUN_MANIFEST.csv                     every run: status, offsets, RMSE, CE90, paths
    TRUTH_BY_DETECTOR_MATCHER.csv        with truth_csv: detector + matcher ranked vs ground truth
    TRUTH_RANKING.csv, TRUTH_POINTS.csv  ... every configuration, every truth point
    automatch.log                        full log
Progress is printed as lines starting with '@@AUTOMATCH ' followed by JSON.
"""

import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PROGRESS_PREFIX = '@@AUTOMATCH '

# Printed at the start of every run log and in PACK_REPORT.txt, so a log says
# which code made it. Not a job key: settings files stay readable by other
# versions.
AUTOMATCH_VERSION = '2026.10.05'
CODE_FILES = ('automatch_engine.py', 'automatch_job.py', 'automatch_imcui.py', 'automatch_refs.py',
              'automatch_rival.py', 'automatch_truth.py', 'automatch_weights.py', 'automatch_gpuaas.py',
              'automatch_native.py', 'imw_configs.py')


def code_fingerprint() -> str:
    """Short hash of the modules a run uses (line endings ignored): the same
    on the workstation and the server only when both have the same code."""
    h = hashlib.sha1()
    for name in CODE_FILES:
        try:
            with open(os.path.join(HERE, name), 'rb') as f:
                data = f.read().replace(b'\r\n', b'\n')
        except OSError:
            data = b'missing'
        h.update(name.encode() + b'\0' + data + b'\0')
    return h.hexdigest()[:8]


def version_line() -> str:
    return f'automatch {AUTOMATCH_VERSION}, code {code_fingerprint()} ({HERE})'

# ─────────────────────────────────────────────────────────────────────────────
# Job specification
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_JOB: Dict = {
    # inputs / outputs
    'input_path': '',            # G1A raster (GeoTIFF/VRT/JP2...) or NISAR .h5 / scene dir
    'channels': [],              # [] = all; raster: "band1".., NISAR: "HH"..
    'reference_dir': '',         # L8_ref / C1 / any RIVAL-readable collection
    'reference_label': '',       # tag in file names; default from folder name
    'reference_mode': 'auto',    # auto | index-shp | sidecar | degree-tile
    'reference_band': 1,         # band used from multi-band references
    'reference_band_map': {},    # per channel, e.g. {"band1": 4}
    'reference_fill_values': [0, 3],
    'reference_scale': 1.0,      # DN scale on the reference (0.003162 for S1 GRD)
    'output_dir': '',
    'temp_dir': '',              # default <output_dir>/_cache
    'resume': True,              # reuse sweep points finished with identical settings

    # algorithms (kornia first; imcui only for algorithms kornia lacks)
    'detectors': ['sift', 'disk'],   # or "all" / "all-kornia" / "all-imcui" (what this machine offers)
    'matchers': {},              # per detector, e.g. {"disk": ["lgm"]}; {} = each detector's defaults;
                                 # "all" = every matcher each detector offers
    # Per-detector parameters, each a list of values to try, e.g.
    #   {"sift": {"rootsift": [true, false]},
    #    "dedode": {"detector_weights": ["L-C4-v2"], "descriptor_weights": ["B-upright", "G-upright"]},
    #    "disk": {"checkpoint": ["depth", "epipolar"], "lgm.filter_threshold": [0.1, 0.2]}}
    # Detector parameters: every combination runs as its own named variant and
    # competes for RIVAL_BEST. 'lgm.*' / 'ada.*' matcher parameters: every
    # combination is an extra matching pass. `detectors` lists what is settable.
    'detector_params': {},
    'smnn_thresholds': [0.9, 0.95],
    'weights_cache_dir': '',     # offline imcui weights (see prefetch_imw_weights.py)

    # sweep
    'window_sizes': [1024],
    'num_features': [None],      # None = from keypoint_density and window size
    'keypoint_density': 9000,    # keypoints per megapixel of window
    'max_num_features': 32000,   # cap per window; raise it to keep the density at large windows
                                 # (e.g. 90000 for 3072 px on a 40 GB GPU)
    'gpu_window_px': 0,          # largest window a detector matches at once; larger windows are
                                 # matched in tiles, pooled. 0 = each detector finds it on this GPU
    'target_resolution': None,   # None = input's native resolution (metres)

    # large offsets (coarse-to-fine)
    'max_expected_error_m': 50000,
    'coarse_method': 'auto',     # auto | matcher | phasecorr | manual | none
    'coarse_resolution_m': 60,
    'initial_offset_m': None,    # [dE, dN] metres, input minus reference
    'search_margin_m': 1500,
    'coarse_local': True,        # per-cell coarse offsets (internal distortion)
    'coarse_cell_km': 0,         # cell size; 0 = one window wide

    # RANSAC (thresholds in pixels of the working resolution unless *_m given)
    'ransac_methods': ['magsac'],
    'ransac_thresholds_px': [1.0, 2.0],
    'ransac_thresholds_m': [],
    'ransac_confidences': [0.99],

    # chip consensus (tolerance in pixels unless *_m given)
    'consensus_tolerance_px': 3.0,
    'consensus_tolerance_m': None,
    'consensus_mode_bin_px': 0.5,
    'min_inliers_per_chip': 6,
    'min_surviving_chips': 1,
    'consensus_model': 'surface',   # surface | constant | none (see README)
    'consensus_surface': 'auto',    # auto | affine | bilinear | quadratic | biquadratic
    'manual_gcp_csv': '',

    # ground truth (evaluation only: it does not steer matching or consensus)
    'truth_csv': '',             # RIVAL CSV of manually measured points (In/Ref lon/lat)
    'truth_radius_m': 5000,      # tool points this close to a truth point estimate its error

    # RIVAL export
    'rival_max_points_per_chip': 200,   # 0 = all inliers

    # misc
    'device': 'auto',            # auto | cuda | cuda:0 | cpu
    'use_amp': True,
    'min_valid_fraction': 0.3,
    'min_area_km2': 50.0,
    'min_gpu_free_gb': 1.5,
    'save_match_images': False,
    'nisar_band': 'auto',
    'nisar_frequency': 'A',
    'debug': False,
}

JOB_HELP = {
    'input_path': 'Image to assess: G1A raster (any rasterio format) or NISAR .h5 / scene folder',
    'channels': 'Channels to process ([] = all): band1.. for rasters, HH/HV.. for NISAR',
    'reference_dir': 'Reference collection (L8_ref, C1, ...) - discovered like DPQED_rival.py',
    'max_expected_error_m': 'Worst-case geolocation error; search buffer for references and coarse alignment',
    'coarse_method': 'auto = matcher at coarse resolution, phase correlation if weak',
    'initial_offset_m': '[dE, dN] metres (input minus reference) if known, e.g. from RIVAL',
    'truth_csv': 'Manually measured points (RIVAL CSV): every detector + matcher is ranked against them',
}


_WARNED_KEYS = set()


def unknown_keys(job: Dict) -> List[str]:
    """Settings this version does not have ('_' keys are internal notes)."""
    return sorted(k for k in job if k not in DEFAULT_JOB and not str(k).startswith('_'))


def unknown_key_note(key: str) -> str:
    near = difflib.get_close_matches(key, list(DEFAULT_JOB), n=1, cutoff=0.75)
    why = (f'did you mean {near[0]!r}?' if near else
           'written by a newer version? copy the current g1a_automatch folder here')
    return f'setting {key!r} is not used by this version (automatch {AUTOMATCH_VERSION}), ignored -- {why}'


def normalize(job: Dict) -> Dict:
    """The job with every default filled in. Settings this version does not
    know are ignored with a warning (once per setting), so a settings file
    from another version still runs; '_' keys are dropped silently."""
    out = copy.deepcopy(DEFAULT_JOB)
    for k in unknown_keys(job):
        if k not in _WARNED_KEYS:
            _WARNED_KEYS.add(k)
            print(f'[Job] WARNING: {unknown_key_note(k)}', flush=True)
    out.update(copy.deepcopy({k: v for k, v in job.items() if k in DEFAULT_JOB}))
    if not out['temp_dir'] and out['output_dir']:
        out['temp_dir'] = os.path.join(out['output_dir'], '_cache')
    for k in ('window_sizes', 'num_features', 'detectors', 'smnn_thresholds',
              'ransac_methods', 'ransac_thresholds_px', 'ransac_thresholds_m', 'ransac_confidences'):
        if not isinstance(out[k], list):
            out[k] = [out[k]]
    out['num_features'] = [None if v in (None, '', 'auto', 0) else int(v) for v in out['num_features']] or [None]
    dp = out.get('detector_params') or {}
    if not isinstance(dp, dict) or not all(isinstance(v, dict) for v in dp.values()):
        raise ValueError('detector_params must map detector -> {parameter: [values]}')
    out['detector_params'] = {d: {k: (list(v) if isinstance(v, (list, tuple)) else [v])
                                  for k, v in p.items()} for d, p in dp.items()}
    if out['initial_offset_m'] in ([], '', None):
        out['initial_offset_m'] = None
    return out


def load_job(path: str, overrides: Optional[List[str]] = None) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        job = json.load(f)
    for ov in overrides or []:
        if '=' not in ov:
            raise ValueError(f'--set expects key=value, got {ov!r}')
        k, v = ov.split('=', 1)
        if k.strip() not in DEFAULT_JOB:     # typed just now, for this version: a typo
            raise ValueError(f'--set {k.strip()}: {unknown_key_note(k.strip())}')
        try:
            job[k.strip()] = json.loads(v)
        except ValueError:
            job[k.strip()] = v
    return normalize(job)


def emit(event: Dict):
    if os.environ.get('AUTOMATCH_EVENTS', '1') == '0':   # e.g. batch services: log only
        return
    try:
        print(PROGRESS_PREFIX + json.dumps(event, default=str), flush=True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Detector catalog
# ─────────────────────────────────────────────────────────────────────────────
def detector_catalog(weights_cache: str = '') -> Dict:
    """All selectable detectors (kornia + imcui-only), and why imcui ones were
    skipped. Registers imcui detectors as a side effect."""
    import automatch_engine as E
    import automatch_imcui as IM
    imw = IM.register_imcui_detectors(weights_cache or None)
    return {'detectors': E.available_detectors(), 'imcui': imw,
            'kornia_unavailable': E.kornia_unavailable()}


def format_catalog(cat: Dict) -> str:
    """The catalogue as text: one line per selectable detector (default and
    optional matchers, settings), then every imcui row not offered and why."""
    lines = []
    for det in cat['detectors']:
        defaults = det.get('default_matchers', det['matchers'])
        extra = [m for m in det['matchers'] if m not in defaults]
        ms = ', '.join(defaults) + (f" (+ {', '.join(extra)} when selected)" if extra else '')
        ps = ', '.join(p['name'] for p in det.get('params', [])
                       if p.get('scope', 'detector') == 'detector' and '.' not in p['name']) or '-'
        if det['source'] != 'kornia' and det.get('params'):
            ps = ', '.join(p['name'] for p in det['params'])   # imcui: thresholds + model keys
        lines.append(f"{det['name']:<22} {det['source']:<7} matchers: {ms:<52} settings: {ps}")
    for name, why in (cat.get('kornia_unavailable') or {}).items():
        lines.append(f'{name:<22} kornia  not available here: {why}')
    if cat['imcui'].get('error'):
        lines.append(f"(imcui: {cat['imcui']['error']})")
    for tag, why in cat['imcui'].get('skipped', []):
        lines.append(f'{"imw-" + tag:<22} not offered: {why}')
    return '\n'.join(lines)


# 'all' tokens for job['detectors']: everything selectable ON THIS MACHINE
# (the imcui part depends on the installed imcui), or one source of it.
ALL_DETECTORS = {'all': None, 'all-kornia': 'kornia', 'all-imcui': 'imcui'}


def _is_all(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == 'all'


def expand_selection(job: Dict) -> Dict:
    """Replace 'all' / 'all-kornia' / 'all-imcui' in detectors and "all" as
    matchers (for every detector, or as one detector's value) by explicit
    lists from this machine's catalogue. A job without them is returned as is."""
    dets = list(job.get('detectors') or [])
    m = job.get('matchers')
    want_d = any(isinstance(d, str) and d.strip().lower() in ALL_DETECTORS for d in dets)
    want_m = _is_all(m) or (isinstance(m, dict) and any(_is_all(v) for v in m.values()))
    if not (want_d or want_m):
        return job
    import automatch_engine as E
    cat = detector_catalog(job.get('weights_cache_dir', ''))
    info = {d['name']: d for d in cat['detectors']}
    names: List[str] = []
    for d in dets:
        key = d.strip().lower() if isinstance(d, str) else d
        if key in ALL_DETECTORS:
            names += [n for n, x in info.items() if ALL_DETECTORS[key] in (None, x['source'])]
        else:
            names.append(d)
    names = list(dict.fromkeys(names))
    out = dict(job, detectors=names)
    if want_m:
        per = {}
        for d in names:
            offered = info.get(E.resolve_detector_name(d)[0], {}).get('matchers')
            given = m if _is_all(m) else (m or {}).get(d)
            if _is_all(given) and offered:
                per[d] = list(offered)
            elif isinstance(given, list):
                per[d] = given
        out['matchers'] = per
    print(f"[Job] selection expanded on this machine: {len(names)} detector(s): {', '.join(names)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────
GCP_COLUMNS = {'scan', 'pix', 'Map_X', 'Map_Y', 'Map_X_ref', 'Map_Y_ref'}


def _check_gcp_file(path: str) -> List[str]:
    """Errors for a manual GCP CSV the consensus cannot use."""
    import csv as _csv
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            cols = {c.strip() for c in next(_csv.reader(f), [])}
    except Exception as e:
        return [f'manual_gcp_csv unreadable: {type(e).__name__}: {e}']
    missing = GCP_COLUMNS - cols
    if not missing:
        return []
    if {'In_Lon', 'In_Lat', 'Ref_Lon', 'Ref_Lat'} <= cols:
        return ['manual_gcp_csv is a RIVAL CSV. Manual GCP CSV takes NISAR-style GCPs (columns scan, pix, '
                'Map_X, Map_Y, Map_X_ref, Map_Y_ref) that steer the chip consensus. To rank detectors and '
                'matchers against manually measured RIVAL points, give the file as Ground truth CSV '
                '(truth_csv) and leave Manual GCP CSV empty.']
    return [f'manual_gcp_csv lacks the columns {sorted(missing)} (needs {sorted(GCP_COLUMNS)})']


WINDOW_KM = 92.0     # 2048 px at 45 m (G1A MX-VNIR): the window size the ranking runs were made with


def _scale_warnings(job: Dict, scene, res: float, E) -> List[str]:
    """Window sizes that do not suit this image's pixel size. A window is
    best judged on the ground: 2048 px is 92 km at 45 m (MX-VNIR) but 369 km
    at 180 m (HS) and 645 km at 315 m, where a scene gives one chip per
    reference (run 300, jobs 312-317)."""
    out = []
    if scene.kind != 'raster' or not scene.native_res:
        return out
    with E.rt.open(scene.raster_path) as src:
        w_km = src.width * abs(src.transform.a) / 1000.0
        h_km = src.height * abs(src.transform.e) / 1000.0
    short_px = min(w_km, h_km) * 1000.0 / res
    suggest = max(256, int(short_px / 3) // 64 * 64)
    ground = max(256, int(round(WINDOW_KM * 1000.0 / res / 64.0)) * 64)
    for w in sorted(set(int(v) for v in job['window_sizes'])):
        if w > short_px:
            out.append(f'{w} px windows are {w * res / 1000:.0f} km across at {res:g} m, larger than the image '
                       f'({w_km:.0f} x {h_km:.0f} km): each reference gives at most one chip and one model '
                       f'covers the whole scene. Use about a third of the image, here {suggest} px')
        elif w * res / 1000.0 > 2 * WINDOW_KM:
            out.append(f'{w} px windows are {w * res / 1000:.0f} km across at {res:g} m: few chips per scene. '
                       f'About {WINDOW_KM:.0f} km, here {ground} px, gives as many chips as 2048 px windows on '
                       f'45 m images')
    return out


def preflight(job: Dict, check_weights: bool = True) -> Dict:
    """Validate a job without matching. {'errors', 'warnings', 'info'}.
    check_weights: also look for every model weight file the job will load
    (missing ones are warnings: they download on first use when online)."""
    errors, warnings, info = [], [], {}
    warnings += [unknown_key_note(k) for k in
                 dict.fromkeys(list(job.get('_ignored_keys') or []) + unknown_keys(job))]
    job = normalize(job)

    if not job['input_path'] or not os.path.exists(job['input_path']):
        errors.append(f"input_path not found: {job['input_path']!r}")
    if not job['reference_dir'] or not os.path.isdir(job['reference_dir']):
        errors.append(f"reference_dir not found: {job['reference_dir']!r}")
    if not job['output_dir']:
        errors.append('output_dir is empty')
    else:
        try:
            os.makedirs(job['output_dir'], exist_ok=True)
            probe = os.path.join(job['output_dir'], '.write_test')
            open(probe, 'w').close()
            os.remove(probe)
        except OSError as e:
            errors.append(f'output_dir not writable: {e}')
    if job['manual_gcp_csv'] and not os.path.exists(job['manual_gcp_csv']):
        errors.append(f"manual_gcp_csv not found: {job['manual_gcp_csv']}")
    elif job['manual_gcp_csv']:
        errors += _check_gcp_file(job['manual_gcp_csv'])
        if job['truth_csv'] and os.path.exists(job['truth_csv']) and \
                os.path.samefile(job['manual_gcp_csv'], job['truth_csv']):
            warnings.append('the same file is both Manual GCP CSV (it steers the chip consensus) and Ground '
                            'truth CSV (it scores the result): the ranking would favour configurations picked '
                            'to agree with it. Leave Manual GCP CSV empty when choosing a detector + matcher.')
    if job['truth_csv']:
        try:
            import automatch_truth as T
            info['truth_points'] = len(T.load_truth(job['truth_csv']))
        except Exception as e:
            errors.append(f"truth_csv: {type(e).__name__}: {e}")
    if job['consensus_model'] not in ('surface', 'constant', 'none'):
        errors.append(f"consensus_model must be surface, constant or none (got {job['consensus_model']!r})")
    if job['consensus_surface'] not in ('auto', 'affine', 'bilinear', 'quadratic', 'biquadratic'):
        errors.append(f"consensus_surface must be auto, affine, bilinear, quadratic or biquadratic "
                      f"(got {job['consensus_surface']!r})")
    if job['coarse_method'] == 'manual' and job['initial_offset_m'] is None:
        errors.append("coarse_method 'manual' needs initial_offset_m [dE, dN]")
    if any(w < 256 for w in job['window_sizes']):
        errors.append('window_sizes must be >= 256 px')
    if errors:
        return {'errors': errors, 'warnings': warnings, 'info': info}

    import automatch_engine as E
    import automatch_refs as refs

    # input
    try:
        cfg = E.PipelineConfig(nisar_band=job['nisar_band'], nisar_frequency=job['nisar_frequency'])
        scene = E.InputScene(job['input_path'], cfg)
        info['input'] = scene.describe()
        bad = [c for c in job['channels'] if c not in scene.channels]
        if bad:
            errors.append(f'channels {bad} not in input; available {scene.channels}')
        res = job['target_resolution'] or scene.native_res
        if res is None and scene.kind == 'raster':
            warnings.append('input is in lon/lat; set target_resolution (metres) explicitly')
        info['working_resolution_m'] = res
        if res:
            warnings += _scale_warnings(job, scene, float(res), E)
    except Exception as e:
        errors.append(f'input unreadable: {type(e).__name__}: {e}')
        scene = None

    # references
    try:
        mode = None if job['reference_mode'] in ('', 'auto', None) else job['reference_mode']
        cat = refs.scan_reference_folder(job['reference_dir'], mode)
        info['references'] = {'mode': cat['mode'], 'n_rasters': cat['n_rasters'],
                              'n_footprints': len(cat['footprints']),
                              'errors': cat['errors'][:10]}
        if not cat['footprints']:
            errors.append('no reference footprints could be read')
        elif scene is not None:
            from shapely.geometry import Polygon
            ch = (job['channels'] or scene.channels)[0]
            try:
                if scene.kind == 'raster':
                    path, band = scene.channel_raster(ch, '')
                    foot, src = scene.footprint_lonlat(path, band)
                else:
                    foot, src = E.NISARH5Reader.footprint_lonlat(scene.info)
                grow_deg = job['max_expected_error_m'] / 111000.0
                grown = foot.buffer(grow_deg)
                hits = [os.path.basename(p) for p, r in cat['footprints'].items()
                        if len(r['ring']) >= 3 and grown.intersects(Polygon(r['ring']))]
                info['footprint_source'] = src
                info['footprint_bounds'] = [round(v, 4) for v in foot.bounds]
                info['candidate_references'] = hits[:50]
                if not hits:
                    errors.append('no reference footprint lies within max_expected_error_m of the input')
            except Exception as e:
                warnings.append(f'footprint check skipped: {type(e).__name__}: {e}')
    except Exception as e:
        errors.append(f'reference folder: {type(e).__name__}: {e}')

    # detectors
    try:
        job = expand_selection(job)
        cat = detector_catalog(job['weights_cache_dir'])
        names = {d['name'] for d in cat['detectors']}
        wanted = list(dict.fromkeys(job['detectors']))
        unknown = [d for d in wanted if E.resolve_detector_name(d)[0] not in names]
        if unknown:
            skipped = dict(cat['imcui'].get('skipped') or [])
            lacking = E.kornia_unavailable()
            for d in unknown:
                tag = d[len('imw-'):] if d.startswith('imw-') else d
                why = (lacking.get(E.resolve_detector_name(d)[0]) or skipped.get(tag)
                       or cat['imcui'].get('error') or 'not available')
                errors.append(f'detector {d!r}: {why}')
        supported = {x['name']: x['matchers'] for x in cat['detectors']}
        defaults = {x['name']: x.get('default_matchers', x['matchers']) for x in cat['detectors']}
        for det, ms in (job['matchers'] or {}).items():
            sup = supported.get(E.resolve_detector_name(det)[0])
            if sup is None:
                errors.append(f'matchers given for unknown detector {det!r}')
            elif set(ms) - set(sup):
                errors.append(f'{det}: unsupported matcher(s) {sorted(set(ms) - set(sup))}; supported {sup}')
        # parameters: validate, and count what the job will actually run
        runs, total = [], 0
        n_channels = len(job['channels']) or len((info.get('input') or {}).get('channels') or [1])
        n_sweep = len(job['window_sizes']) * len(job['num_features'])
        for d in wanted:
            base = E.resolve_detector_name(d)[0]
            if base not in supported:
                continue
            try:
                ms = (job['matchers'] or {}).get(d) or (job['matchers'] or {}).get(base) or defaults[base]
                nv, npass = E.count_runs(d, job['detector_params'].get(d), ms, len(job['smnn_thresholds']))
                runs.append({'detector': d, 'variants': nv, 'passes': npass})
                total += npass
            except ValueError as e:
                errors.append(str(e))
        for d in job['detector_params']:
            if d not in wanted:
                warnings.append(f'parameters given for {d!r}, which is not selected (ignored)')
        info['detectors'] = wanted
        cfgk = E.PipelineConfig(keypoint_density=job['keypoint_density'],
                                max_num_features=int(job['max_num_features']))
        info['keypoints_per_window'] = {
            int(w): (int(nf) if nf else cfgk.compute_num_features(int(w)))
            for w in job['window_sizes'] for nf in job['num_features']}
        for w, n in info['keypoints_per_window'].items():
            want = int(job['keypoint_density'] * w * w / 1e6)
            if n < want:
                warnings.append(f'{w} px windows get {n} keypoints, the cap (max_num_features); '
                                f'{want} would keep the density of {job["keypoint_density"]} per megapixel')
        info['runs'] = runs
        info['matching_passes'] = total * n_channels * n_sweep
        info['ransac_sets_per_pass'] = (len(job['ransac_methods']) * len(job['ransac_confidences'])
                                        * len(job['ransac_thresholds_m'] or job['ransac_thresholds_px']))
    except Exception as e:
        errors.append(f'detector catalog: {type(e).__name__}: {e}')

    # model weights
    if check_weights and not errors:
        try:
            import automatch_weights as W
            wrep = W.check(job=job, quiet=True)
            info['weights'] = wrep['summary']
            for m in wrep['missing']:
                warnings.append(f"weights: {m['file']} ({', '.join(m['detectors'])}) is not on this machine "
                                f"-- it is downloaded on first use if this machine is online; offline, that "
                                f"detector is skipped. Source: {m['source']}. See 'Check weights'.")
            for d in wrep['detectors']:
                for v in d['variants']:
                    if v['status'] in ('error', 'partial'):
                        warnings.append(f"weights: {v['variant']}: {v['status']} ({v['error']})")
        except Exception as e:
            warnings.append(f'weights check skipped: {type(e).__name__}: {e}')

    # device
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info['gpu'] = f'{p.name} ({p.total_memory / 1024 ** 3:.1f} GB)'
            if p.total_memory / 1024 ** 3 < 20 and max(job['window_sizes']) > 2048:
                warnings.append('window sizes > 2048 px may not fit a GPU under 20 GB')
        else:
            info['gpu'] = 'none (CPU)'
            if job['device'].startswith('cuda'):
                errors.append('device=cuda requested but CUDA is not available')
    except Exception:
        pass
    return {'errors': errors, 'warnings': warnings, 'info': info}


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def _config_for(job: Dict, win: int, nf: Optional[int], res: float, out_dir: str):
    import automatch_engine as E
    thr_m = job['ransac_thresholds_m'] or [round(t * res, 3) for t in job['ransac_thresholds_px']]
    tol = job['consensus_tolerance_m'] or job['consensus_tolerance_px'] * res
    cfg = E.PipelineConfig(
        window_size=win, window_size_small=max(256, win // 2), loftr_max_window=win,
        gpu_window_px=int(job['gpu_window_px'] or 0),
        keypoint_density=job['keypoint_density'], max_num_features=int(job['max_num_features']),
        target_resolution=job['target_resolution'],
        use_disk_cache=True, check_existing_pairs=True, cleanup_after_pair=False,
        use_amp=job['use_amp'], debug_mode=job['debug'],
        smnn_thresholds=job['smnn_thresholds'], ransac_methods=job['ransac_methods'],
        ransac_thresholds=thr_m, ransac_confidences=job['ransac_confidences'],
        reference_dir=job['reference_dir'], reference_label=job['reference_label'],
        reference_mode=None if job['reference_mode'] in ('', 'auto') else job['reference_mode'],
        reference_band=int(job['reference_band']), reference_band_map=job['reference_band_map'] or None,
        reference_fill_values=tuple(job['reference_fill_values'] or ()),
        s1_calibration_factor=float(job['reference_scale']),
        output_base_dir=out_dir, temp_dir=job['temp_dir'], min_area=job['min_area_km2'],
        min_valid_fraction=job['min_valid_fraction'],
        consensus_tolerance_m=tol, consensus_mode_bin_m=max(0.1, job['consensus_mode_bin_px'] * res),
        min_inliers_per_chip=job['min_inliers_per_chip'],
        min_surviving_chips=job['min_surviving_chips'], manual_gcp_csv=job['manual_gcp_csv'],
        pols=job['channels'] or None, detector_matchers=job['matchers'] or None,
        detector_params=job['detector_params'] or None,
        max_expected_error_m=job['max_expected_error_m'], coarse_method=job['coarse_method'],
        coarse_resolution_m=job['coarse_resolution_m'],
        initial_offset_m=tuple(job['initial_offset_m']) if job['initial_offset_m'] else None,
        search_margin_m=job['search_margin_m'], min_gpu_free_gb=job['min_gpu_free_gb'],
        coarse_local=bool(job['coarse_local']),
        coarse_cell_m=(float(job['coarse_cell_km']) * 1000.0) if job['coarse_cell_km'] else None,
        consensus_model=job['consensus_model'], consensus_surface_degree=job['consensus_surface'],
        save_match_images=job['save_match_images'], weights_cache_dir=job['weights_cache_dir'],
        nisar_band=job['nisar_band'], nisar_frequency=job['nisar_frequency'],
    )
    cfg.num_features = nf if nf else cfg.compute_num_features(win)
    return cfg


# Job keys that cannot change what the matcher computes for a sweep point.
_RESULT_NEUTRAL_KEYS = {'output_dir', 'temp_dir', 'resume', 'debug', 'rival_max_points_per_chip',
                        'save_match_images', 'weights_cache_dir', 'truth_csv', 'truth_radius_m'}


def _sweep_key(job: Dict, win: int, nf: Optional[int]) -> str:
    """Fingerprint of everything that shapes one sweep point's results, so
    'resume' only reuses results produced with the same settings and input."""
    blob = {k: v for k, v in job.items() if k not in _RESULT_NEUTRAL_KEYS}
    blob.update({'_window': win, '_num_features': nf, '_v': 1})
    try:
        st = os.stat(job['input_path'])
        blob['_input'] = [st.st_size, int(st.st_mtime)]
    except OSError:
        pass
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _best_score(record: Dict) -> Optional[float]:
    p = record.get('best_final_summary_csv')
    if not p or not isinstance(p, str) or not os.path.exists(p):
        return None
    import pandas as pd
    try:
        return float(pd.read_csv(p)['score'].iloc[0])
    except Exception:
        return None


def _matcher_label(method, param) -> str:
    """Same label as the file names / PASS_TIMING / truth ranking."""
    import pandas as pd
    if str(method) == 'smnn' and param is not None and not pd.isna(param):
        return f'smnn{float(param):g}'
    return str(method)


def _performance_table(out_root: str, manifest: List[Dict]) -> Optional[str]:
    """PERFORMANCE.csv: per channel x sweep x detector x matcher -- speed
    (seconds per window, detection included), what its best RANSAC setting
    kept in the chip consensus, and its agreement with the ground truth when
    there is one. Sorted by truth RMSE, else by chips kept."""
    import glob as _glob
    import pandas as pd
    def read(path):         # an empty or cut-off file is skipped, not fatal (job 313)
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    truth = {}
    tr = os.path.join(out_root, 'TRUTH_RANKING.csv')
    if os.path.exists(tr):
        t = read(tr)
        if not t.empty:     # each detector + matcher at its best-ranked RANSAC setting
            t = t.sort_values('rank') if 'rank' in t.columns else t.sort_values('truth_rmse_m', na_position='last')
            for key, g in t.groupby(['channel', 'sweep', 'detector', 'matcher'], sort=False):
                truth[tuple(str(k) for k in key)] = g.iloc[0]
    rows = []
    for row in manifest:
        fd = row.get('final_dir')
        if not isinstance(fd, str) or not os.path.isdir(fd):
            continue
        tp = os.path.join(fd, 'PASS_TIMING.csv')
        timing = read(tp).to_dict('records') if os.path.exists(tp) else []
        best = {}
        for sp in _glob.glob(os.path.join(fd, 'CONSENSUS_SCORES*.csv')):
            sc = read(sp)
            for r in sc.to_dict('records'):
                lab = _matcher_label(r.get('match_method'), r.get('match_parameter'))
                cur = best.get(lab)
                if cur is None or (r['surviving_chips'], r['agg_inliers']) > (cur['surviving_chips'],
                                                                               cur['agg_inliers']):
                    best[lab] = r
        for t in timing:
            lab = t['matcher']
            b = best.get(lab) or {}
            k = (str(row.get('channel')), str(row.get('sweep')), str(row.get('detector')), lab)
            tru = truth.get(k)
            rows.append({
                'channel': row.get('channel'), 'sweep': row.get('sweep'), 'detector': row.get('detector'),
                'matcher': lab, 'sec_per_window': t.get('sec_per_window'), 'pass_seconds': t.get('seconds'),
                'windows': t.get('windows'),
                'chips_kept': b.get('surviving_chips'), 'inliers': b.get('agg_inliers'),
                'surface_rmse_m': b.get('surface_rmse_m'),
                'best_ransac': (f"{b.get('ransac_method')}{float(b['ransac_threshold']):g}"
                                if b.get('ransac_threshold') is not None else None),
                'truth_rank': None if tru is None else tru.get('rank'),
                'truth_rmse_m': None if tru is None else tru.get('truth_rmse_m'),
                'truth_reached': None if tru is None else tru.get('truth_reached'),
                'truth_total': None if tru is None else tru.get('truth_total'),
                'truth_ransac': None if tru is None else tru.get('ransac'),
                'detector_seconds': row.get('detector_seconds'), 'gpu_peak_gb': row.get('gpu_peak_gb'),
                'gpu_window_px': row.get('gpu_window_px')})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df['truth_rank'].notna().any():     # the order of TRUTH_RANKING.csv
        df = df.sort_values(['truth_rank', 'sec_per_window'], na_position='last')
    else:
        df = df.sort_values(['chips_kept', 'sec_per_window'], ascending=[False, True], na_position='last')
    path = os.path.join(out_root, 'PERFORMANCE.csv')
    df.to_csv(path, index=False)
    print('[Job] detector + matcher performance (PERFORMANCE.csv):')
    print(f"  {'channel':<8} {'detector':<26} {'matcher':<12} {'s/window':>9} {'chips':>6} {'inliers':>8} "
          f"{'truth RMSE m':>13} {'reached':>8} {'GPU GB':>7}")
    for r in df.head(40).itertuples():
        f = (lambda v, w, p=1: f'{v:{w}.{p}f}' if v is not None and not pd.isna(v) else f"{'-':>{w}}")
        reached = ('-' if r.truth_reached is None or pd.isna(r.truth_reached)
                   else f'{int(r.truth_reached)}/{int(r.truth_total)}')
        print(f"  {str(r.channel):<8} {str(r.detector):<26} {str(r.matcher):<12} {f(r.sec_per_window, 9, 2)} "
              f"{f(r.chips_kept, 6, 0)} {f(r.inliers, 8, 0)} {f(r.truth_rmse_m, 13)} {reached:>8} "
              f"{f(r.gpu_peak_gb, 7, 1)}")
    return path


def refresh_after_compare(output_dir: str, res: Dict) -> List[str]:
    """After a finished run is ranked again (compare, e.g. a run made by an
    older version): PERFORMANCE.csv and RIVAL_BEST_* follow the new ranking,
    as a run writes them. Returns the files rewritten."""
    import glob as _glob
    import pandas as pd
    written = []
    man = os.path.join(output_dir, 'RUN_MANIFEST.csv')
    if os.path.exists(man):
        try:
            p = _performance_table(output_dir, pd.read_csv(man).to_dict('records'))
            if p:
                written.append(p)
        except Exception as e:
            print(f'[Job] performance table not rewritten: {type(e).__name__}: {e}')
    for ch, b in (res.get('best') or {}).items():
        for dst in _glob.glob(os.path.join(output_dir, f'RIVAL_BEST_*_{ch}.csv')):
            shutil.copyfile(b['csv'], dst)
            written.append(dst)
            print(f"[Job] {os.path.basename(dst)} is now {b['detector']} + {b['matcher']} {b['ransac']} "
                  f"(truth RMSE {b['truth_rmse_m']:.1f} m)")
    return written


def _compare_with_truth(job: Dict, out_root: str, working_crs: str, manifest: List[Dict]) -> Dict:
    """Rank every detector + matcher configuration against the ground truth;
    add each run's own truth metrics to its manifest row."""
    import automatch_truth as T
    runs = [{'filtered_dir': r['filtered_dir'], 'final_dir': r['final_dir'], 'sweep': r['sweep']}
            for r in manifest if isinstance(r.get('filtered_dir'), str) and isinstance(r.get('final_dir'), str)
            and os.path.isdir(r['filtered_dir']) and os.path.isdir(r['final_dir'])]
    try:
        res = T.compare(out_root, job['truth_csv'], float(job['truth_radius_m']), 'consensus', working_crs,
                        runs=runs)
    except Exception as e:
        print(f'[Truth] comparison failed: {type(e).__name__}: {e}')
        traceback.print_exc()
        return {}
    for row in manifest:
        if row.get('detail_csv') and os.path.exists(row['detail_csv']):
            try:
                m = T.score_detail_csv(row['detail_csv'], res['truth'], float(job['truth_radius_m']))
                row.update({'truth_rmse_m': m.get('truth_rmse_m'), 'truth_reached': m.get('truth_reached'),
                            'truth_mean_dE_m': m.get('truth_mean_dE_m'),
                            'truth_mean_dN_m': m.get('truth_mean_dN_m')})
                emit({'event': 'truth_result', 'sweep': row['sweep'], 'channel': row['channel'],
                      'detector': row['detector'], 'truth_rmse_m': m.get('truth_rmse_m'),
                      'truth_reached': m.get('truth_reached')})
            except Exception as e:
                print(f"[Truth] {row.get('detector')}: {type(e).__name__}: {e}")
    files = {k: os.path.join(out_root, f'{k}.csv')
             for k in ('TRUTH_BY_DETECTOR_MATCHER', 'TRUTH_RANKING', 'TRUTH_POINTS')}
    by = res['by_detector_matcher']
    emit({'event': 'truth', 'files': files, 'n_truth': len(res['truth']),
          'top': json.loads(by.head(30).to_json(orient='records')) if not by.empty else []})
    files['best'] = res.get('best') or {}
    return files


def run_job(job: Dict) -> Dict:
    job = expand_selection(normalize(job))
    pf = preflight(job, check_weights=False)   # warmup stops a detector whose weights are missing
    for w in pf['warnings']:
        print(f'[Preflight] WARNING: {w}')
    if pf['errors']:
        for e in pf['errors']:
            print(f'[Preflight] ERROR: {e}')
        emit({'event': 'failed', 'errors': pf['errors']})
        raise SystemExit(2)

    import pandas as pd
    import automatch_engine as E
    import automatch_rival as R

    out_root = job['output_dir']
    E.setup_logging(out_root, 'automatch.log')
    print(f'[Job] {version_line()}')
    with open(os.path.join(out_root, 'job_used.json'), 'w', encoding='utf-8') as f:
        json.dump(job, f, indent=2)
    E.set_device(job['device'])
    E.PROGRESS_HOOK = emit

    scene = E.InputScene(job['input_path'], E.PipelineConfig(nisar_band=job['nisar_band'],
                                                             nisar_frequency=job['nisar_frequency']))
    res = job['target_resolution'] or scene.native_res or 10.0
    detectors = job['detectors']
    sweep = [(w, nf) for w in job['window_sizes'] for nf in job['num_features']]
    emit({'event': 'job', 'scene': scene.name, 'channels': job['channels'] or scene.channels,
          'detectors': detectors, 'sweep': sweep, 'resolution_m': res})
    print(f'[Job] {scene.name}: {len(sweep)} sweep point(s), working resolution {res} m, '
          f'RANSAC thresholds {job["ransac_thresholds_m"] or [t * res for t in job["ransac_thresholds_px"]]} m')

    manifest = []
    rival_dir = os.path.join(out_root, 'rival')
    t_job = time.time()
    for k, (win, nf) in enumerate(sweep):
        tag = f'win{win}_nf{nf or "auto"}'
        out_dir = os.path.join(out_root, tag)
        emit({'event': 'sweep', 'done': k, 'total': len(sweep), 'point': tag})
        summary_csv = os.path.join(out_dir, 'POL_RUN_SUMMARY.csv')
        key_path = os.path.join(out_dir, 'SWEEP_KEY.txt')
        key = _sweep_key(job, win, nf)
        prev = None
        if os.path.exists(key_path):
            with open(key_path) as fh:
                prev = fh.read().strip()
        t0 = time.time()
        if job['resume'] and os.path.exists(summary_csv) and prev == key:
            print(f'[Job] {tag}: reusing finished results ({summary_csv})')
            records = pd.read_csv(summary_csv).to_dict('records')
        else:
            if job['resume'] and os.path.exists(summary_csv):
                print(f'[Job] {tag}: settings or input changed since the finished run -- running again')
            if os.path.exists(key_path):
                os.remove(key_path)
            cfg = _config_for(job, win, nf, res, out_dir)
            print(f'[Job] >>> {tag}: window {win} px, {cfg.num_features} keypoints per window '
              f'({cfg.num_features / (win * win / 1e6):.0f} per megapixel)')
            try:
                records = E.AutoMatchPipeline(cfg).run(job['input_path'], detectors)
                with open(key_path, 'w') as fh:
                    fh.write(key)
            except Exception as e:
                print(f'[Job] {tag} FAILED: {type(e).__name__}: {e}')
                traceback.print_exc()
                records = [{'status': 'failed', 'error': f'{type(e).__name__}: {e}'}]
            E.safe_cuda_empty_cache()
        for rec in records:
            row = {'sweep': tag, 'window_size': win, 'num_features': nf or 'auto',
                   'channel': rec.get('nisar_pol'), 'reference': rec.get('s1_ref_tag'),
                   'detector': rec.get('detector_tag'), 'status': rec.get('status'),
                   'error': rec.get('error') if isinstance(rec.get('error'), str) else '',
                   'coarse_offsets': rec.get('coarse_offsets'), 'consensus_score': _best_score(rec),
                   'minutes': round((time.time() - t0) / 60.0, 2), 'working_crs': scene.working_crs,
                   'detector_seconds': rec.get('seconds'), 'gpu_peak_gb': rec.get('gpu_peak_gb'),
                   'gpu_window_px': rec.get('gpu_window_px'),
                   'filtered_dir': rec.get('filtered_dir'), 'final_dir': rec.get('final_dir')}
            try:
                exp = R.export_run(rec, scene.working_crs, rival_dir, f'{scene.name}_{tag}',
                                   job['rival_max_points_per_chip'])
            except Exception as e:
                exp = None
                row['error'] = (row['error'] + f' | export: {type(e).__name__}: {e}').strip(' |')
            if exp:
                row.update({'rival_csv': exp['rival_csv'], 'detail_csv': exp['detail_csv'],
                            'validated_by_consensus': exp['validated'], 'n_points': exp['n'],
                            'n_inliers_all': exp['n_inliers_all'], 'n_chips': exp.get('n_chips'),
                            **(exp.get('distortion') or {}),
                            'mean_dx_m': exp.get('mean_dx'), 'mean_dy_m': exp.get('mean_dy'),
                            'rmse_x_m': exp['rmse_x'], 'rmse_y_m': exp['rmse_y'], 'ce90_m': exp['ce90'],
                            'matcher_config': exp['matcher_config']})
                emit({'event': 'result', **{k2: row.get(k2) for k2 in (
                    'sweep', 'channel', 'detector', 'status', 'n_points', 'n_chips', 'mean_dx_m',
                    'mean_dy_m', 'dE_min_m', 'dE_max_m', 'dN_min_m', 'dN_max_m', 'affine_rot_deg',
                    'affine_resid_rmse_m', 'rmse_x_m', 'rmse_y_m', 'ce90_m', 'rival_csv')}})
            else:
                emit({'event': 'result', 'sweep': tag, 'channel': row['channel'],
                      'detector': row['detector'], 'status': row['status'], 'error': row['error']})
            manifest.append(row)

    truth_files = {}
    if job['truth_csv']:
        truth_files = _compare_with_truth(job, out_root, scene.working_crs, manifest)
    man = pd.DataFrame(manifest)
    man_path = os.path.join(out_root, 'RUN_MANIFEST.csv')
    man.to_csv(man_path, index=False)
    if 'gpu_window_px' in man.columns:      # what fitted the GPU, detector by detector
        gw = pd.to_numeric(man['gpu_window_px'], errors='coerce')
        tiled = man[gw < pd.to_numeric(man['window_size'], errors='coerce')]
        if len(tiled):
            print('[Job] matched in tiles to fit the GPU: ' + ', '.join(
                f'{r.detector} {int(r.window_size)} px in {int(r.gpu_window_px)} px tiles'
                for r in tiled.drop_duplicates(['detector', 'window_size']).itertuples()))
    try:
        _performance_table(out_root, manifest)
    except Exception as e:
        print(f'[Job] performance table not written: {type(e).__name__}: {e}')

    best_files = {}
    truth_best = (truth_files or {}).get('best') or {}
    if not man.empty and 'rival_csv' in man.columns:
        ok = man[man['rival_csv'].notna()].copy()
        if not ok.empty:
            ok['_rank'] = ok['consensus_score'].fillna(-1.0)
            ok['_v'] = ok.get('validated_by_consensus', True).fillna(False).astype(bool)
            for ch, g in ok.groupby('channel'):
                best = g.sort_values(['_v', '_rank', 'n_points'], ascending=False).iloc[0]
                # with ground truth, the consensus pick is only kept for reference:
                # many consistent but wrong matches can win it
                name = 'RIVAL_CONSENSUS_BEST' if str(ch) in truth_best else 'RIVAL_BEST'
                dst = os.path.join(out_root, f'{name}_{scene.name}_{ch}.csv')
                shutil.copyfile(best['rival_csv'], dst)
                if name == 'RIVAL_BEST':
                    best_files[ch] = dst
                print(f'[Job] {"best" if name == "RIVAL_BEST" else "consensus pick"} for {ch}: '
                      f'{best["detector"]} {best["sweep"]} (mean dE {best["mean_dx_m"]:.1f} m, '
                      f'dN {best["mean_dy_m"]:.1f} m, CE90 {best["ce90_m"]:.1f} m) -> {dst}')
    for ch, b in truth_best.items():
        dst = os.path.join(out_root, f'RIVAL_BEST_{scene.name}_{ch}.csv')
        shutil.copyfile(b['csv'], dst)
        best_files[ch] = dst
        print(f"[Job] best for {ch} by ground truth: {b['detector']} + {b['matcher']} {b['ransac']} "
              f"{b['sweep']} (truth RMSE {b['truth_rmse_m']:.1f} m) -> {dst}")
    emit({'event': 'done', 'manifest': man_path, 'best': best_files, 'truth': truth_files,
          'minutes': round((time.time() - t_job) / 60.0, 2)})
    print(f'[Job] done in {(time.time() - t_job) / 60.0:.1f} min; manifest {man_path}')
    return {'manifest': man_path, 'best': best_files}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='G1A / NISAR automatic geolocation matching')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('template', help='print a job file with the defaults')
    for name in ('preflight', 'run'):
        p = sub.add_parser(name)
        p.add_argument('job')
        p.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    d = sub.add_parser('detectors', help='list selectable detectors')
    d.add_argument('--weights-cache', default='')
    i = sub.add_parser('inspect', help='describe an input image')
    i.add_argument('input')
    w = sub.add_parser('weights', help='check which model weight files are on this machine '
                                       '(options: python automatch_weights.py -h)')
    w.add_argument('rest', nargs=argparse.REMAINDER)
    c = sub.add_parser('compare', help='rank the detector + matcher configurations of an output folder '
                                       'against ground truth (options: python automatch_truth.py -h)')
    c.add_argument('rest', nargs=argparse.REMAINDER)
    k = sub.add_parser('pack', help='copy a scene, the references it needs and server settings for the '
                                    'GPU server (options: python automatch_pack.py -h)')
    k.add_argument('rest', nargs=argparse.REMAINDER)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == 'compare':
        import automatch_truth as T
        return T.main(list(argv[1:]))
    if argv and argv[0] == 'pack':
        import automatch_pack as P
        return P.main(list(argv[1:]))
    if argv and argv[0] == 'weights':
        # a job file as first argument, everything else as automatch_weights takes it
        import automatch_weights as W
        rest = list(argv[1:])
        if rest and not rest[0].startswith('-'):
            rest = ['--job'] + rest
        return W.main(rest)
    a = ap.parse_args(argv)

    if a.cmd == 'template':
        print(json.dumps(DEFAULT_JOB, indent=2))
        return 0
    if a.cmd == 'detectors':
        cat = detector_catalog(a.weights_cache)
        print(format_catalog(cat))
        emit({'event': 'detectors', **cat})
        return 0
    if a.cmd == 'inspect':
        import automatch_engine as E
        sc = E.InputScene(a.input, E.PipelineConfig())
        desc = sc.describe()
        print(json.dumps(desc, indent=2, default=str))
        emit({'event': 'inspect', **desc})
        return 0
    job = load_job(a.job, a.set)
    if a.cmd == 'preflight':
        res = preflight(job)
        print(json.dumps(res, indent=2, default=str))
        emit({'event': 'preflight', **res})
        return 1 if res['errors'] else 0
    try:
        run_job(job)
        return 0
    except SystemExit as e:
        return int(e.code or 1)


if __name__ == '__main__':
    sys.exit(main())
