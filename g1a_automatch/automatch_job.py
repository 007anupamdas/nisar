#!/usr/bin/env python3
"""
automatch_job -- run an automatch job from a JSON file (the GUI writes the same
file and runs this script as a subprocess, so GUI and command line are one path).

    python automatch_job.py template  > job.json     # annotated defaults
    python automatch_job.py preflight job.json       # validate, no matching
    python automatch_job.py run       job.json       # match + RIVAL CSVs
    python automatch_job.py detectors                # what can be selected
    python automatch_job.py inspect   <input>        # channels, CRS, footprint

Any job key can be overridden on the command line, e.g.
    python automatch_job.py run job.json --set window_sizes=[1024,2048] --set detectors=["sift"]

Outputs (under output_dir):
    win<W>_nf<N>/<channel>_to<ref>/...   engine outputs per sweep point
    rival/RIVAL_*.csv                    one per channel x detector x sweep point,
                                         loadable in DPQED_rival.py ('Load CSV')
    RIVAL_BEST_<scene>_<channel>.csv     highest consensus score per channel
    RUN_MANIFEST.csv                     every run: status, offsets, RMSE, CE90, paths
    automatch.log                        full log
Progress is printed as lines starting with '@@AUTOMATCH ' followed by JSON.
"""

import argparse
import copy
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
    'detectors': ['sift', 'disk'],
    'matchers': {},              # per detector, e.g. {"disk": ["lgm"]}; {} = all
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
}


def normalize(job: Dict) -> Dict:
    out = copy.deepcopy(DEFAULT_JOB)
    unknown = sorted(set(job) - set(DEFAULT_JOB))
    if unknown:
        raise ValueError(f'Unknown job key(s): {unknown}')
    out.update(copy.deepcopy(job))
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
        try:
            job[k.strip()] = json.loads(v)
        except ValueError:
            job[k.strip()] = v
    return normalize(job)


def emit(event: Dict):
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
    return {'detectors': E.available_detectors(), 'imcui': imw}


# ─────────────────────────────────────────────────────────────────────────────
# Preflight
# ─────────────────────────────────────────────────────────────────────────────
def preflight(job: Dict) -> Dict:
    """Validate a job without matching. {'errors', 'warnings', 'info'}."""
    errors, warnings, info = [], [], {}
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
        cat = detector_catalog(job['weights_cache_dir'])
        names = {d['name'] for d in cat['detectors']}
        wanted = list(dict.fromkeys(job['detectors']))
        unknown = [d for d in wanted if E.resolve_detector_name(d)[0] not in names]
        if unknown:
            skipped = dict(cat['imcui'].get('skipped') or [])
            for d in unknown:
                tag = d[len('imw-'):] if d.startswith('imw-') else d
                why = skipped.get(tag) or cat['imcui'].get('error') or 'not available'
                errors.append(f'detector {d!r}: {why}')
        supported = {x['name']: x['matchers'] for x in cat['detectors']}
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
                ms = (job['matchers'] or {}).get(d) or (job['matchers'] or {}).get(base) or supported[base]
                nv, npass = E.count_runs(d, job['detector_params'].get(d), ms, len(job['smnn_thresholds']))
                runs.append({'detector': d, 'variants': nv, 'passes_per_variant': npass})
                total += nv * npass
            except ValueError as e:
                errors.append(str(e))
        for d in job['detector_params']:
            if d not in wanted:
                warnings.append(f'parameters given for {d!r}, which is not selected (ignored)')
        info['detectors'] = wanted
        info['runs'] = runs
        info['matching_passes'] = total * n_channels * n_sweep
        info['ransac_sets_per_pass'] = (len(job['ransac_methods']) * len(job['ransac_confidences'])
                                        * len(job['ransac_thresholds_m'] or job['ransac_thresholds_px']))
    except Exception as e:
        errors.append(f'detector catalog: {type(e).__name__}: {e}')

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
        keypoint_density=job['keypoint_density'],
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
                        'save_match_images', 'weights_cache_dir'}


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


def run_job(job: Dict) -> Dict:
    job = normalize(job)
    pf = preflight(job)
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
            print(f'[Job] >>> {tag}: window {win} px, {cfg.num_features} features')
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
                   'minutes': round((time.time() - t0) / 60.0, 2)}
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

    man = pd.DataFrame(manifest)
    man_path = os.path.join(out_root, 'RUN_MANIFEST.csv')
    man.to_csv(man_path, index=False)

    best_files = {}
    if not man.empty and 'rival_csv' in man.columns:
        ok = man[man['rival_csv'].notna()].copy()
        if not ok.empty:
            ok['_rank'] = ok['consensus_score'].fillna(-1.0)
            ok['_v'] = ok.get('validated_by_consensus', True).fillna(False).astype(bool)
            for ch, g in ok.groupby('channel'):
                best = g.sort_values(['_v', '_rank', 'n_points'], ascending=False).iloc[0]
                dst = os.path.join(out_root, f'RIVAL_BEST_{scene.name}_{ch}.csv')
                shutil.copyfile(best['rival_csv'], dst)
                best_files[ch] = dst
                print(f'[Job] best for {ch}: {best["detector"]} {best["sweep"]} '
                      f'(mean dE {best["mean_dx_m"]:.1f} m, dN {best["mean_dy_m"]:.1f} m, '
                      f'CE90 {best["ce90_m"]:.1f} m) -> {dst}')
    emit({'event': 'done', 'manifest': man_path, 'best': best_files,
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
    a = ap.parse_args(argv)

    if a.cmd == 'template':
        print(json.dumps(DEFAULT_JOB, indent=2))
        return 0
    if a.cmd == 'detectors':
        cat = detector_catalog(a.weights_cache)
        for det in cat['detectors']:
            ps = ', '.join(p['name'] for p in det.get('params', [])) or '-'
            print(f"{det['name']:<18} {det['source']:<7} matchers: {', '.join(det['matchers']):<16} params: {ps}")
        if cat['imcui'].get('error'):
            print(f"(imcui: {cat['imcui']['error']})")
        for tag, why in cat['imcui'].get('skipped', []):
            print(f'  imcui {tag}: not offered ({why})')
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
