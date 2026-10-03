#!/usr/bin/env python3
"""
Tests for automatch. Plain script, like tests_rival_*.py:

    python tests_automatch.py                     # fast checks (no matching)
    python tests_automatch.py --e2e               # + synthetic end-to-end runs
    python tests_automatch.py --rival /path/DPQED_rival.py   # + parity with RIVAL

The end-to-end runs build a G1A-like 3-band raster whose georeferencing is off
by a known (dE, dN) of kilometres, plus C1 (degree tiles, lon/lat) and L8_ref
(UTM + Meta/index.shp) references, run automatch_job with kornia SIFT, and
require the RIVAL CSV to report that offset to within half a pixel.
"""

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(name, got, want=True):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ('' if ok else f'  (got {got!r}, want {want!r})'))
    if not ok:
        FAILS.append(name)


# ── reference discovery ──────────────────────────────────────────────────────
def test_refs(tmp):
    import automatch_refs as R
    check('degree tile N16E78', R.parse_degree_tile('N16E78.tif')['ring'][0], (78.0, 17.0))
    check('degree tile unpadded N8E76', R.parse_degree_tile('N8E76_ortho.tif')['ring'][3], (76.0, 8.0))
    check('NISAR name is not a tile', R.parse_degree_tile('NISAR_L2_PR_GSLC.tif'), None)
    check('mode index-shp', R.detect_reference_mode(['a.tif', 'index.shp']), R.REF_MODE_INDEX)
    check('mode sidecar', R.detect_reference_mode(['a.tif', 'a_meta.txt']), R.REF_MODE_SIDECAR)
    check('mode degree-tile', R.detect_reference_mode(['N16E78.tif']), R.REF_MODE_TILE)
    check('meta stem .h5.iso.xml', R.meta_base_stem('X.h5.iso.xml'), 'X')
    check('meta stem _META.txt', R.meta_base_stem('scene_META.txt'), 'scene')
    check('label L8_ref', R.reference_label('/d/L8_ref'), 'L8ref')

    # sidecar folder: gdalinfo-style _meta.txt beside the tif
    d = os.path.join(tmp, 'side')
    os.makedirs(d)
    open(os.path.join(d, 'S1A_X.tif'), 'wb').close()
    with open(os.path.join(d, 'S1A_X_meta.txt'), 'w') as f:
        f.write('Upper Left  ( 78.0, 17.0)\nLower Left  ( 78.0, 16.0)\n'
                'Upper Right ( 79.0, 17.0)\nLower Right ( 79.0, 16.0)\n')
    cat = R.scan_reference_folder(d)
    check('sidecar footprint paired', list(os.path.basename(p) for p in cat['footprints']), ['S1A_X.tif'])

    # index shapefile via the built-in reader (projected .prj, reprojected)
    from synthetic_data import _write_index_shp
    d = os.path.join(tmp, 'idx')
    os.makedirs(os.path.join(d, 'Meta'))
    open(os.path.join(d, 'LC08_A.tif'), 'wb').close()
    _write_index_shp(os.path.join(d, 'Meta', 'index.shp'),
                     [('LC08_A.tif', [(78.0, 17.0), (79.0, 17.0), (79.0, 16.0), (78.0, 16.0)]),
                      ('NOT_HERE.tif', [(70.0, 10.0), (71.0, 10.0), (71.0, 9.0), (70.0, 9.0)])])
    names, feats, reader = R.read_index_features(os.path.join(d, 'Meta', 'index.shp'))
    check('index reader is built-in', reader, 'builtin')
    check('index attributes', names, ['FILENAME'])
    check('index ring', [tuple(round(v, 6) for v in p) for p in feats[0][1][:2]], [(78.0, 17.0), (79.0, 17.0)])
    cat = R.scan_reference_folder(d)
    check('index footprint matched (unmatched entry ignored)', len(cat['footprints']), 1)


# ── RIVAL parity ─────────────────────────────────────────────────────────────
PORTED = ['band_from_name', 'band_from_frequency', 'close_ring', 'is_data_value', 'parse_pos_list',
          'parse_meta_json', 'parse_meta_text', '_iso_text', 'parse_meta_iso_xml', 'meta_base_stem',
          'match_raster', 'parse_degree_tile', 'rank_name_fields', 'build_name_lookup',
          'resolve_index_name', 'pick_index_shapefile', 'detect_reference_mode', 'is_meta_file',
          'accuracy_stats', 'csv_record']


def _functions(path):
    src = open(path, encoding='utf-8').read()
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef):
            out.setdefault(node.name, ast.dump(node))
    return out, src


def test_rival_parity(rival_path):
    if not rival_path or not os.path.exists(rival_path):
        print('SKIP  RIVAL parity (pass --rival /path/DPQED_rival.py)')
        return
    theirs, src = _functions(rival_path)
    ours, _ = _functions(os.path.join(HERE, 'automatch_refs.py'))
    for fn in PORTED:
        check(f'ported {fn} identical to DPQED_rival.py', ours.get(fn) == theirs.get(fn))
    import automatch_rival as AR
    i = src.index('def save_csv(self)')
    header = ast.literal_eval(src[src.index('w.writerow([', i) + len('w.writerow('):
                                  src.index('])', src.index('w.writerow([', i)) + 1])
    check('RIVAL CSV header identical', AR.RIVAL_HEADER, header)


# ── engine pure helpers ──────────────────────────────────────────────────────
def test_engine_helpers():
    import automatch_engine as E
    P = E.MatchStatistics._parse_filename
    r = P('sift_band1_toC1_pair001_scan0_pix2048_smnn_0.95_aff_magsac_20.0_0.99_.csv')
    check('parse sift', (r['detector'], r['nisar_pol'], r['s1_ref_tag'], r['pix'], r['match_parameter'],
                         r['ransac_threshold']), ('sift', 'band1', 'C1', 2048, 0.95, 20.0))
    r = P('disk_depth_band2_toL8ref_pair003_scan5_pix6_lgm_aff_4_2_0.99_.csv')
    check('parse disk_depth lgm', (r['disk_mode'], r['match_method'], r['match_parameter']), ('depth', 'lgm', None))
    r = P('dedode_L-C4_G-C4_HH_toS1VV_pair001_scan1_pix2_smnn_0.9_aff_4_1_0.91_.csv')
    check('parse dedode weights', (r['det_wt'], r['desc_wt'], r['s1_ref_tag']), ('L-C4', 'G-C4', 'S1VV'))
    r = P('loftr_HH_toC1_pair004_scan7_pix8_loftr_internal_aff_4_2_0.95_.csv')
    check('parse loftr', r['match_method'], 'loftr_internal')
    r = P('imw-sp-lg_band1_toC1_pair002_scan1_pix2_internal_aff_ransac_2_0.99_.csv')
    check('parse imcui', (r['detector'], r['desc_wt'], r['ransac_method']), ('imw-sp-lg', 'sp-lg', 'ransac'))
    check('raw csv not parsed', P('sift_band1_toC1_pair001_scan0_pix0_smnn_0.95_raw.csv'), None)
    check('ransac token 4 = MAGSAC', E.ransac_flag(4), E.ransac_flag('magsac'))
    try:
        E.PipelineConfig(ransac_methods=['bogus'])
        check('bad RANSAC method rejected', False)
    except ValueError:
        check('bad RANSAC method rejected', True)
    check('kornia refuses imcui disk', E.register_detector('imw-disk-lg', lambda c: None, ['disk'], 'imcui'), False)
    check('imcui superpoint allowed', E.register_detector('imw-test-sp', lambda c: None, ['superpoint'], 'imcui'), True)
    E.EXTERNAL_DETECTORS.pop('imw-test-sp', None)
    check('utm zone 78.5E 16N', E.utm_epsg_for(78.5, 16.4), 32644)


# ── end to end ───────────────────────────────────────────────────────────────
def _run_job(job, tmp, want_output=False):
    path = os.path.join(tmp, 'job.json')
    with open(path, 'w') as f:
        json.dump(job, f)
    out = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_job.py'), 'run', path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
    return (out.returncode, out.stdout) if want_output else out.returncode


def test_e2e_rerun(tmp):
    """Rerun in the same output folder: identical settings reuse the results;
    changed settings run again on the cached preprocessing, and the previous
    run's files do not leak into the new consensus."""
    import glob
    import synthetic_data as S
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'G1A_SYNTH_L1.tif')):
        S.make_all(data, 4013.0, -2487.0)
    out = os.path.join(tmp, 'out_rerun')
    job = {'input_path': os.path.join(data, 'G1A_SYNTH_L1.tif'), 'reference_dir': os.path.join(data, 'C1'),
           'output_dir': out, 'channels': ['band1'], 'detectors': ['sift'], 'window_sizes': [1024],
           'max_expected_error_m': 10000, 'smnn_thresholds': [0.95], 'use_amp': False}
    rc, log1 = _run_job(job, tmp, True)
    check('rerun: first run', rc, 0)
    rc, log2 = _run_job(job, tmp, True)
    check('rerun: same settings -> results reused', rc == 0 and 'reusing finished results' in log2)
    rc, log3 = _run_job({**job, 'smnn_thresholds': [0.9]}, tmp, True)
    check('rerun: changed settings -> run again', rc == 0 and 'running again' in log3)
    check('rerun: preprocessing reused from the cache', '[Cache] All 1 pairs loaded' in log3)
    files = [os.path.basename(f) for f in glob.glob(
        os.path.join(out, 'win1024_nfauto', 'band1_toC1', 'filtered_same-res_sift', '*.csv'))]
    check('rerun: new run\'s files written', any('_smnn_0.9_' in f for f in files))
    check('rerun: no file left from the previous run', not any('_smnn_0.95_' in f for f in files))


def test_e2e_variants(tmp):
    """SIFT rootsift on/off + DISK LightGlue filter sweep: every variant and
    matcher pass is run and named, and RIVAL_BEST picks one of them."""
    import csv as _csv
    import glob
    import synthetic_data as S
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'G1A_SYNTH_L1.tif')):
        S.make_all(data, 4013.0, -2487.0)
    out = os.path.join(tmp, 'out_variants')
    rc = _run_job({'input_path': os.path.join(data, 'G1A_SYNTH_L1.tif'),
                   'reference_dir': os.path.join(data, 'C1'), 'output_dir': out,
                   'channels': ['band1'], 'detectors': ['sift', 'disk', 'dedode'],
                   'matchers': {'disk': ['lgm'], 'dedode': ['ada', 'lgm']},
                   'detector_params': {'sift': {'rootsift': [True, False]},
                                       'disk': {'lgm.filter_threshold': [0.1, 0.2]},
                                       'dedode': {'detector_weights': ['L-C4-v2'],
                                                  'descriptor_weights': ['B-upright']}},
                   'window_sizes': [1024], 'max_expected_error_m': 10000,
                   'smnn_thresholds': [0.95], 'use_amp': False}, tmp)
    check('variants: job exit code', rc, 0)
    man = os.path.join(out, 'RUN_MANIFEST.csv')
    rows = list(_csv.DictReader(open(man, encoding='utf-8'))) if os.path.exists(man) else []
    check('variants: one manifest row per variant', sorted(r['detector'] for r in rows),
          ['dedode_L-C4-v2_B-upright', 'disk_depth', 'sift', 'sift-rs0'])
    ded = {os.path.basename(p).split('_pix0_')[1].split('_aff')[0]
           for p in glob.glob(os.path.join(out, '*', 'band1_toC1', 'filtered_*dedode*', '*.csv'))}
    check('variants: DeDoDe matched with AdaLAM and LightGlue', ded >= {'ada', 'lgm'})
    sift = {os.path.basename(p).split('_pix0_')[1].split('_aff')[0]
            for p in glob.glob(os.path.join(out, '*', 'band1_toC1', 'filtered_*_sift', '*.csv'))}
    check('variants: SIFT matched with SMNN and AdaLAM', {'ada', 'smnn_0.95'} <= sift)
    check('variants: all ran', {r['status'] for r in rows}, {'ok'})
    lgm = {os.path.basename(p).split('_pix0_')[1].split('_aff')[0]
           for p in glob.glob(os.path.join(out, '*', 'band1_toC1', 'filtered_*disk_depth', '*.csv'))}
    check('variants: both LightGlue passes filtered', lgm >= {'lgm', 'lgm_lf0p2'})
    best = os.path.join(out, 'RIVAL_BEST_G1A_SYNTH_L1_band1.csv')
    check('variants: RIVAL_BEST chosen among them', os.path.exists(best))
    for r in rows:
        if r.get('mean_dx_m'):
            ok = abs(float(r['mean_dx_m']) - 4013.0) < 10 and abs(float(r['mean_dy_m']) + 2487.0) < 10
            print(f"      {r['detector']:<10} dE {float(r['mean_dx_m']):.1f}  dN {float(r['mean_dy_m']):.1f}  "
                  f"CE90 {float(r['ce90_m']):.1f}")
            check(f"variants: {r['detector']} recovers the offset", ok)


def test_e2e_distortion(tmp):
    """Error that varies across the scene by kilometres (6 % east-west scale,
    curvature): every exported point must carry the error of ITS location,
    chips must survive across the whole scene, and the constant rule shows
    why it is not used."""
    import csv as _csv
    import glob
    import numpy as np
    import pandas as pd
    import synthetic_data as S
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'C1')):
        S.make_all(data, 4013.0, -2487.0)
    world, wtf = S._world()
    inp, (Ec, Nc) = S.make_input_distorted(data, world, wtf)
    base = {'input_path': inp, 'reference_dir': os.path.join(data, 'C1'), 'channels': ['band1'],
            'detectors': ['sift'], 'window_sizes': [512], 'max_expected_error_m': 10000,
            'smnn_thresholds': [0.95], 'use_amp': False}
    runs = {}
    for model in ('surface', 'constant'):
        out = os.path.join(tmp, f'out_dist_{model}')
        rc = _run_job({**base, 'output_dir': out, 'consensus_model': model}, tmp)
        check(f'distortion ({model}): job exit code', rc, 0)
        man = os.path.join(out, 'RUN_MANIFEST.csv')
        runs[model] = (out, list(_csv.DictReader(open(man, encoding='utf-8')))[0] if os.path.exists(man) else {})
    out, row = runs['surface']
    field_csv = glob.glob(os.path.join(out, '*', 'band1_toC1', 'raw_matches_*', 'COARSE_FIELD_pair001.csv'))
    if field_csv:
        f = pd.read_csv(field_csv[0])
        print(f"      coarse field: {len(f)} cells, dE {f.dE_m.min():.0f}..{f.dE_m.max():.0f} m")
        check('distortion: coarse field follows the varying error', f.dE_m.max() - f.dE_m.min() > 2500)
    else:
        check('distortion: coarse field written', False)
    det = row.get('detail_csv') or ''
    if os.path.exists(det):
        d = pd.read_csv(det, encoding='utf-8-sig')
        tE, tN = S.distortion_field(d.In_X.values, d.In_Y.values, Ec, Nc)
        eE, eN = np.abs(d.DX_Err.values - tE), np.abs(d.DY_Err.values - tN)
        print(f"      {len(d)} points from {d.source_file.nunique()} chips; |error - truth| median "
              f"{np.median(eE):.1f} / {np.median(eN):.1f} m, 95% {np.percentile(eE, 95):.1f} / "
              f"{np.percentile(eN, 95):.1f} m; true dE spans {tE.min():.0f}..{tE.max():.0f} m")
        check('distortion: each point carries its own location\'s error (median < 10 m)',
              np.median(eE) < 10 and np.median(eN) < 10)
        check('distortion: 95% of points within 30 m of the truth',
              np.percentile(eE, 95) < 30 and np.percentile(eN, 95) < 30)
        check('distortion: chips kept across the scene (>= 20 of 25)', d.source_file.nunique() >= 20)
        # The summary regresses the error on reference position, so compare it
        # with the same summary of the TRUE errors at the same points.
        import automatch_rival as AR
        truth = AR.distortion_summary(d.In_X, d.In_Y, d.In_X - tE, d.In_Y - tN)
        sc, sc_t = float(row.get('affine_scale_E_ppm') or 0), truth['affine_scale_E_ppm']
        rr, rr_t = float(row.get('affine_resid_rmse_m') or 0), truth['affine_resid_rmse_m']
        print(f"      affine scale E {sc:.0f} ppm (truth {sc_t:.0f}), rotation "
              f"{float(row.get('affine_rot_deg') or 0):.3f} deg (truth {truth['affine_rot_deg']:.3f}), "
              f"residual beyond affine {rr:.0f} m (truth {rr_t:.0f})")
        check('distortion: scale and residual match the truth',
              abs(sc - sc_t) < 1000 and abs(rr - rr_t) < 20)
    else:
        check('distortion: RIVAL detail written', False)
    n_const = int(float(runs['constant'][1].get('n_chips') or 0))
    n_surf = int(float(row.get('n_chips') or 0))
    print(f"      chips kept: surface {n_surf}, constant {n_const}")
    check('distortion: constant rule keeps far fewer chips', n_const < n_surf / 2)


def test_e2e_missing_weights(tmp):
    """A detector whose weights are missing is skipped at once with a clear
    message; the other detectors still run."""
    import csv as _csv
    import synthetic_data as S
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'G1A_SYNTH_L1.tif')):
        S.make_all(data, 4013.0, -2487.0)
    out = os.path.join(tmp, 'out_missing')
    # imcui-free, offline-simulated: point loftr at weights this machine never had
    import automatch_engine as E
    import time as _t
    job = os.path.join(tmp, 'job_missing.py')
    with open(job, 'w') as f:
        f.write(textwrap.dedent(f"""
            import sys, json
            sys.path.insert(0, {HERE!r})
            import automatch_engine as E
            def offline(url, *a, **k):
                raise OSError('offline: ' + url)
            E._ORIG_LOAD_STATE_DICT = offline
            E._default_hub_dir = lambda: {os.path.join(tmp, 'no_weights_here')!r}
            import torch
            torch.hub.set_dir({os.path.join(tmp, 'no_weights_here')!r})
            import automatch_job as J
            sys.exit(J.main(['run', {os.path.join(tmp, 'job_missing.json')!r}]))
        """))
    with open(os.path.join(tmp, 'job_missing.json'), 'w') as f:
        json.dump({'input_path': os.path.join(data, 'G1A_SYNTH_L1.tif'),
                   'reference_dir': os.path.join(data, 'C1'), 'output_dir': out,
                   'channels': ['band1'], 'detectors': ['loftr', 'sift'], 'matchers': {'sift': ['smnn']},
                   'detector_params': {'loftr': {'pretrained': ['indoor_new']}},
                   'window_sizes': [1024], 'max_expected_error_m': 10000,
                   'smnn_thresholds': [0.95], 'use_amp': False}, f)
    t0 = _t.time()
    res = subprocess.run([sys.executable, job], capture_output=True, text=True)
    rows = list(_csv.DictReader(open(os.path.join(out, 'RUN_MANIFEST.csv'), encoding='utf-8'))) \
        if os.path.exists(os.path.join(out, 'RUN_MANIFEST.csv')) else []
    by = {r['detector']: r for r in rows}
    check('missing weights: job completes', res.returncode, 0)
    check('missing weights: detector reported as failed with the reason',
          by.get('loftr-windoornew', {}).get('status') == 'failed'
          and 'model not available' in by.get('loftr-windoornew', {}).get('error', ''))
    check('missing weights: other detectors still run', by.get('sift', {}).get('status'), 'ok')
    check('missing weights: skipped before any window was matched',
          'loftr-windoornew SKIPPED' in res.stdout and 'loftr-windoornew: Processing pair' not in res.stdout)
    print(f'      job took {_t.time() - t0:.0f} s')


def test_e2e_truth(tmp):
    """A run with truth_csv ranks every detector + matcher against the manual
    points: right truth -> metres, truth off by 400 m -> ~400 m."""
    import csv as _csv
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    import synthetic_data as S
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'G1A_SYNTH_L1.tif')):
        S.make_all(data, 4013.0, -2487.0)
    src = rasterio.open(os.path.join(data, 'G1A_SYNTH_L1.tif'))
    b = src.bounds
    to_ll = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
    xs = np.array([0.25, 0.5, 0.75, 0.3, 0.7]) * (b.right - b.left) + b.left
    ys = np.array([0.3, 0.5, 0.7, 0.75, 0.25]) * (b.top - b.bottom) + b.bottom

    def truth_file(name, dE, dN):
        ilo, ila = to_ll.transform(xs, ys)
        rlo, rla = to_ll.transform(xs - dE, ys - dN)
        p = os.path.join(tmp, name)
        pd.DataFrame({'In_X': xs, 'In_Y': ys, 'Ref_X': xs - dE, 'Ref_Y': ys - dN, 'DX_Err': dE, 'DY_Err': dN,
                      'Row': range(1, 6), 'In_Lon': ilo, 'In_Lat': ila, 'Ref_Lon': rlo,
                      'Ref_Lat': rla}).to_csv(p, index=False, encoding='utf-8-sig')
        return p
    out = os.path.join(tmp, 'out_truth')
    job = {'input_path': os.path.join(data, 'G1A_SYNTH_L1.tif'), 'reference_dir': os.path.join(data, 'C1'),
           'output_dir': out, 'channels': ['band1'], 'detectors': ['sift'],
           'matchers': {'sift': ['smnn', 'mnn', 'ada']}, 'window_sizes': [1024],
           'max_expected_error_m': 10000, 'smnn_thresholds': [0.95], 'use_amp': False,
           'truth_csv': truth_file('truth_ok.csv', 4013.0, -2487.0), 'truth_radius_m': 5000}
    code, stdout = _run_job(job, tmp, want_output=True)
    check('truth e2e: job completes', code, 0)
    by_path = os.path.join(out, 'TRUTH_BY_DETECTOR_MATCHER.csv')
    by = pd.read_csv(by_path) if os.path.exists(by_path) else pd.DataFrame()
    check('truth e2e: every detector + matcher ranked', sorted(by.get('matcher_family', pd.Series()).tolist()),
          ['ada', 'mnn', 'smnn'])
    if not by.empty:
        top = by.iloc[0]
        check('truth e2e: best agrees with the truth to a few metres',
              (int(top['truth_reached']) >= 4, float(top['truth_rmse_m']) < 15.0), (True, True))
        print(f"      best: {top['detector']} + {top['best_matcher_setting']} RMSE {top['truth_rmse_m']:.1f} m")
    man = list(_csv.DictReader(open(os.path.join(out, 'RUN_MANIFEST.csv'), encoding='utf-8')))
    check('truth e2e: manifest carries the run\'s truth RMSE',
          bool(man) and man[0].get('truth_rmse_m') not in (None, '') and float(man[0]['truth_rmse_m']) < 15.0)
    check('truth e2e: ranking printed', '[Truth]' in stdout and 'Best detector + matcher' in stdout)
    tb = os.path.join(out, 'RIVAL_TRUTH_BEST_band1.csv')
    rb = os.path.join(out, 'RIVAL_BEST_G1A_SYNTH_L1_band1.csv')
    check('truth e2e: RIVAL_BEST is the configuration that agrees best with the truth; consensus pick kept aside',
          (os.path.exists(tb), os.path.exists(rb) and open(rb, 'rb').read() == open(tb, 'rb').read(),
           os.path.exists(os.path.join(out, 'RIVAL_CONSENSUS_BEST_G1A_SYNTH_L1_band1.csv'))), (True, True, True))
    # the same results against a truth that is 400 m off: re-scored without matching again
    res = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_job.py'), 'compare', out, '--truth',
                          truth_file('truth_off.csv', 4413.0, -2487.0)], capture_output=True, text=True)
    by2 = pd.read_csv(by_path)
    check('truth e2e: compare on existing results measures a 400 m disagreement',
          (res.returncode, round(float(by2.iloc[0]['truth_rmse_m']) / 50) * 50), (0, 400))
    # the GPU service's compare mode re-scores a finished run, nothing matched again
    st = os.path.join(tmp, 'cmp_settings.json')
    md = os.path.join(tmp, 'mode_compare.json')
    with open(st, 'w') as f:
        json.dump({'truth_csv': job['truth_csv']}, f)
    with open(md, 'w') as f:
        json.dump({'mode': 'compare', 'compare_dir': out}, f)
    o2 = os.path.join(tmp, 'o_compare')
    res = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_gpuaas.py'), 'id', o2, f'{st},{md}'],
                         capture_output=True, text=True)
    check('truth e2e: GPU-service compare mode re-scores a finished run into its output folder',
          (res.returncode, os.path.exists(os.path.join(o2, 'TRUTH_BY_DETECTOR_MATCHER.csv')),
           os.path.exists(os.path.join(o2, 'RIVAL_TRUTH_BEST_band1.csv')), 'Processing pair' in res.stdout),
          (0, True, True, False))
    # a run whose consensus never ran (e.g. stopped by an unusable GCP file):
    # compare rebuilds it from the saved matches instead of matching again
    import glob as _glob
    for p in _glob.glob(os.path.join(out, '**', 'final_*', '*.csv'), recursive=True):
        os.remove(p)
    res = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_job.py'), 'compare', out, '--truth',
                          job['truth_csv']], capture_output=True, text=True)
    by3 = pd.read_csv(by_path)
    check('truth e2e: missing consensus rebuilt from the saved matches, same ranking',
          (res.returncode, 'rebuilding' in res.stdout, float(by3.iloc[0]['truth_rmse_m']) < 15.0),
          (0, True, True))
    bad = dict(job, output_dir=os.path.join(tmp, 'out_bad_gcp'), manual_gcp_csv=job['truth_csv'])
    code, stdout = _run_job(bad, tmp, want_output=True)
    check('truth e2e: RIVAL file as manual GCP stops the job in preflight, before matching',
          (code, 'manual_gcp_csv is a RIVAL CSV' in stdout, 'Processing pair' in stdout), (2, True, False))


def test_pack(tmp):
    """pack: only the references the scene can need, with their index; paths
    rewritten for the server; the bundle runs through the GPU-service entry."""
    import synthetic_data as S
    import automatch_pack as P
    import automatch_job as J
    data = os.path.join(tmp, 'synth')
    if not os.path.exists(os.path.join(data, 'G1A_SYNTH_L1.tif')):
        S.make_all(data, 4013.0, -2487.0)
    for ref, want_files in (('L8_ref', {'Meta/index.shp', 'Meta/index.dbf', 'Meta/index.shx'}), ('C1', set())):
        job = J.normalize({'input_path': os.path.join(data, 'G1A_SYNTH_L1.tif'),
                           'reference_dir': os.path.join(data, ref), 'output_dir': os.path.join(tmp, 'x'),
                           'channels': ['band1'], 'detectors': ['sift'], 'max_expected_error_m': 10000})
        dest = os.path.join(tmp, f'bundle_{ref}')
        res = P.pack(job, dest, dest, server_weights='/srv/weights')
        got = {os.path.relpath(os.path.join(dp, f), os.path.join(dest, 'reference', ref)).replace(os.sep, '/')
               for dp, _, fs in os.walk(os.path.join(dest, 'reference', ref)) for f in fs}
        n_all = len([f for f in os.listdir(os.path.join(data, ref)) if f.endswith('.tif')])
        check(f'pack {ref}: selected references copied with their footprint source, check OK',
              (want_files <= got, len(res['references']) >= 1, res['check'].get('lost'),
               len([g for g in got if g.endswith('.tif')]) == len(res['references']) <= n_all),
              (True, True, [], True))
        st = json.load(open(os.path.join(dest, 'automatch_settings.json')))
        check(f'pack {ref}: settings point at the server folder',
              (st['input_path'], st['reference_dir'], st['weights_cache_dir']),
              (f'{dest}/input/G1A_SYNTH_L1.tif', f'{dest}/reference/{ref}', '/srv/weights'))
        again = P.pack(job, dest, dest)
        check(f'pack {ref}: a second pack copies nothing again', (again['copied'], again['skipped'] > 0), (0, True))
    sh = open(os.path.join(dest, 'submit_gpuaas.sh')).read()
    check('pack: curl script for env / weights / preflight / run', 'submit_job' in sh and 'mode_$1.json' in sh)
    inputs = again['inputs']
    check('pack: input_path holds only existing files (the service checks each item)',
          all(os.path.exists(p) for m in ('env', 'weights', 'preflight', 'run') for p in inputs[m].split(',')))
    for m in ('preflight', 'env'):
        res = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_gpuaas.py'), 'id',
                              os.path.join(tmp, f'o_{m}'), inputs[m]], capture_output=True, text=True)
        check(f'pack: mode file selects {m}', (res.returncode, os.path.exists(os.path.join(
            tmp, f'o_{m}', {'preflight': 'PREFLIGHT.json', 'env': 'ENVIRONMENT.txt'}[m]))), (0, True))
    pf = json.load(open(os.path.join(tmp, 'o_preflight', 'PREFLIGHT.json')))
    check('pack: the bundle passes preflight through the GPU-service entry', pf['errors'], [])
    # V:\X\Y <-> /maintenance/X/Y : server paths checked through the share
    lf = P.local_for('/maintenance/ICIGDev/GPUPOC/exe/a.py', '/mnt/v/ICIGDev/GPUPOC/input/g1a/set1',
                     '/maintenance/ICIGDev/GPUPOC/input/g1a/set1')
    check('pack: server path mapped to the local share', lf, '/mnt/v/ICIGDev/GPUPOC/exe/a.py')
    d = os.path.join(tmp, 'comp')
    os.makedirs(d, exist_ok=True)
    for n in ('X.tif', 'X.tif.aux.xml', 'X.tfw', 'X_meta.txt', 'X_B5.tif', 'Y.tif', 'X.TIF.ovr', 'XX.tfw'):
        open(os.path.join(d, n), 'w').close()
    check('pack: companions of a raster (aux, world file, sidecar; not other rasters)',
          sorted(os.path.basename(c) for c in P.companions(os.path.join(d, 'X.tif'))),
          ['X.TIF.ovr', 'X.tfw', 'X.tif.aux.xml', 'X_meta.txt'])


def test_e2e(tmp):
    import synthetic_data as S
    import automatch_rival as AR
    dE, dN = 4013.0, -2487.0
    data = S.make_all(os.path.join(tmp, 'synth'), dE, dN)
    for ref in ('C1', 'L8_ref'):
        out = os.path.join(tmp, f'out_{ref}')
        rc = _run_job({'input_path': data['input'], 'reference_dir': data[ref], 'output_dir': out,
                       'channels': ['band1'], 'detectors': ['sift'], 'window_sizes': [1024],
                       'max_expected_error_m': 10000, 'smnn_thresholds': [0.95], 'use_amp': False}, tmp)
        check(f'{ref}: job exit code', rc, 0)
        best = os.path.join(out, 'RIVAL_BEST_G1A_SYNTH_L1_band1.csv')
        check(f'{ref}: RIVAL_BEST written', os.path.exists(best))
        if not os.path.exists(best):
            continue
        with open(best, encoding='utf-8-sig') as f:
            check(f'{ref}: header', next(csv.reader(f)), AR.RIVAL_HEADER)
        pts = AR.read_rival_csv(best)
        mdx, mdy = float((pts.ix - pts.rx).mean()), float((pts.iy - pts.ry).mean())
        print(f'      {ref}: {len(pts)} points, mean dE {mdx:.1f} m (true {dE}), dN {mdy:.1f} m (true {dN})')
        check(f'{ref}: dE within half a pixel', abs(mdx - dE) < 10.0)
        check(f'{ref}: dN within half a pixel', abs(mdy - dN) < 10.0)


def test_nisar_h5(tmp):
    """NISAR L-band GSLC: compound complex, NaN fill, boundingPolygon, no .met."""
    import numpy as np
    import rasterio
    import synthetic_data as S
    import automatch_engine as E
    world, wtf = S._world()
    h5 = S.make_nisar_h5(tmp, world, wtf)
    sc = E.InputScene(os.path.dirname(h5), E.PipelineConfig())
    check('NISAR: detected', sc.kind, 'nisar')
    check('NISAR: L-band grid found', (sc.info['band'], sc.info['product']), ('LSAR', 'GSLC'))
    check('NISAR: channels', sc.channels, ['HH'])
    check('NISAR: working CRS from projection', sc.working_crs, 'EPSG:32644')
    path, band = sc.channel_raster('HH', os.path.join(tmp, 'cache'))
    with rasterio.open(path) as src:
        a = src.read(1)
        check('NISAR: amplitude finite, fill -> 0', bool(np.isfinite(a).all() and (a == 0).any() and a.max() > 0))
        import h5py
        with h5py.File(h5, 'r') as f:
            x0 = float(f['science/LSAR/GSLC/grids/frequencyA/xCoordinates'][0])
            y0 = float(f['science/LSAR/GSLC/grids/frequencyA/yCoordinates'][0])
        check('NISAR: pixel-centre coords -> corner origin',
              (round(src.transform.c - (x0 - 10.0), 6), round(src.transform.f - (y0 + 10.0), 6)), (0.0, 0.0))
    foot, src = sc.footprint_lonlat(path, band)
    check('NISAR: footprint from boundingPolygon (no .met)', src, 'h5-boundingPolygon')


def test_detector_params():
    import automatch_engine as E
    cfg = E.PipelineConfig(smnn_thresholds=[0.9, 0.95])
    names = lambda d, p=None: [m.get_filename_prefix() for m in E.build_variants(d, cfg, p)]
    check('params: sift rootsift on/off -> 2 named variants', names('sift', {'rootsift': [True, False]}),
          ['sift', 'sift-rs0'])
    check('params: dedode weights in the prefix',
          names('dedode', {'detector_weights': ['L-C4-v2', 'L-C4'], 'descriptor_weights': ['B-upright']}),
          ['dedode_L-C4-v2_B-upright', 'dedode_L-C4_B-upright'])
    check('params: non-default single value is named', names('aliked', {'detection_threshold': 0.3}),
          ['aliked_aliked-n16-dt0p3'])
    check('params: legacy disk_epipolar name', names('disk_epipolar'), ['disk_epipolar'])
    check('params: AdaLAM offered for every sparse kornia detector',
          {d: 'ada' in E.KORNIA_MATCHERS[d] for d in ('sift', 'disk', 'dedode', 'aliked', 'xfeat', 'keynet')},
          {d: True for d in ('sift', 'disk', 'dedode', 'aliked', 'xfeat', 'keynet')})
    check('params: AdaLAM defaults are kornia\'s', (cfg.adalam_search_expansion, cfg.adalam_ransac_iters,
                                                   cfg.adalam_min_confidence), (4, 128, 200))
    check('params: DeDoDe LightGlue weights follow the descriptor',
          [E.build_detector('dedode', cfg, {'descriptor_weights': [w]})._lightglue_feature_name()
           for w in ('B-upright', 'G-C4')], ['dedodeb', 'dedodeg'])
    m = E.build_detector('disk', cfg, {'lgm.filter_threshold': [0.1, 0.2], 'ada.search_expansion': [4, 2]})
    runs = [(n, m._matcher_param_str(n, p)) for n, p in m.matcher_runs()]
    check('params: matcher values add passes, not variants', runs,
          [('smnn', '0.9'), ('smnn', '0.95'), ('lgm', ''), ('lgm', 'lf0p2'), ('ada', ''), ('ada', 'as2')])
    check('params: run count (variants, passes in total)',
          E.count_runs('disk', {'checkpoint': ['depth', 'epipolar'], 'lgm.filter_threshold': [0.1, 0.2]},
                       ['smnn', 'lgm'], 2), (2, 8))
    check('params: a matcher a variant cannot run is not counted (lgm: DoG-HardNet only)',
          E.count_runs('dog', {'descriptor': ['hardnet', 'sosnet']}, ['smnn', 'lgm'], 1), (2, 3))
    check('params: new kornia detectors and names',
          (names('dog', {'descriptor': ['hardnet', 'sosnet'], 'affnet': [True]}),
           names('keynet', {'affnet': [False]}), names('gftt'), names('hessian')),
          (['dog_hardnet-aff1', 'dog_sosnet-aff1'], ['keynet-aff0'], ['gftt'], ['hessian']))
    check('params: LightGlue for SIFT, KeyNet and DoG-HardNet (not DoG-SOSNet)',
          [('lgm' in E.build_detector(d, cfg, p).get_available_matchers(),
            E.build_detector(d, cfg, p)._lightglue_feature_name())
           for d, p in (('sift', {}), ('keynet', {}), ('dog', {}), ('dog', {'affnet': [True]}))]
          + ['lgm' in E.build_detector('dog', cfg, {'descriptor': ['sosnet']}).get_available_matchers()],
          [(True, 'sift'), (True, 'keynet_affnet_hardnet'), (True, 'doghardnet'),
           (True, 'dog_affnet_hardnet'), False])
    check('params: default matchers unchanged by the optional ones',
          (E.build_detector('sift', cfg).selected_matchers(), E.OPTIONAL_MATCHERS['sift']),
          (['smnn', 'lgm', 'ada'], ['mnn', 'snn', 'nn', 'fginn']))
    opt = E.build_detector('sift', E.PipelineConfig(smnn_thresholds=[0.9], detector_matchers={
        'sift': ['smnn', 'snn', 'fginn', 'mnn']}), {'snn.th': [0.7, 0.8]})
    runs = [(n, opt._matcher_param_str(n, p)) for n, p in opt.matcher_runs()]
    check('params: optional matchers run when selected, with their parameter passes', runs,
          [('smnn', '0.9'), ('mnn', ''), ('snn', 'sn0p7'), ('snn', ''), ('fginn', '')])
    r = P0 = E.MatchStatistics._parse_filename('dog_sosnet-aff1_band1_toC1_pair001_scan0_pix0_snn_sn0p7'
                                               '_aff_magsac_20_0.99_.csv')
    check('params: new names parse', (r['detector'], r['disk_mode'], r['match_method']),
          ('dog', 'sosnet-aff1', 'snn_sn0p7'))
    cat = {d['name']: d for d in E.available_detectors()}
    check('params: catalogue lists defaults and optional matchers',
          (cat['sift']['default_matchers'], cat['sift']['matchers'][-4:], cat['loftr']['matchers']),
          (['smnn', 'lgm', 'ada'], ['mnn', 'snn', 'nn', 'fginn'], ['loftr_internal']))
    for bad, why in ((('sift', {'rootsift': ['maybe']}), 'bad bool'),
                     (('dedode', {'detector_weights': ['L-X']}), 'bad choice'),
                     (('sift', {'nope': [1]}), 'unknown name'),
                     (('aliked', {'nms_radius': [2.5]}), 'non-integer')):
        try:
            E.expand_variants(*bad)
            check(f'params: {why} rejected', False)
        except ValueError:
            check(f'params: {why} rejected', True)
    P = E.MatchStatistics._parse_filename
    check('params: variant name parses', P('sift-rs0_band1_toC1_pair001_scan0_pix0_smnn_0.95_aff_magsac_20_0.99_.csv')
          ['detector'], 'sift-rs0')
    r = P('disk_depth_band1_toC1_pair001_scan0_pix0_lgm_lf0p2_aff_magsac_20_0.99_.csv')
    check('params: matcher variant parses', (r['detector'], r['disk_mode'], r['match_method']),
          ('disk', 'depth', 'lgm_lf0p2'))


def test_distortion_helpers():
    """Chip consistency on an error field shaped like the manual RIVAL points
    (6.8 % east-west scale + curvature over ~280 x 540 km) and the coarse
    field interpolation."""
    import numpy as np
    import automatch_engine as E
    C = E.ChipConsensusSelector
    X, Y = np.meshgrid(np.linspace(-150e3, 130e3, 5), np.linspace(-270e3, 268e3, 5))
    X, Y = X.ravel(), Y.ravel()
    u, v = X / 140e3, Y / 270e3
    dE = 27000 + 0.068 * X + 12000 * v ** 2 * (u + 1) / 2
    dN = 6400 + 0.0024 * Y + 1200 * v ** 2
    rng = np.random.default_rng(1)
    a, c = dN + rng.normal(0, 30, 25), dE + rng.normal(0, 30, 25)
    a[7] += 6000
    c[18] -= 9000
    keep, _, _, deg, _ = C._surface_keep(X, Y, a, c, tol=150.0)
    check('surface: every good chip kept, both wrong chips rejected',
          (int(keep.sum()), bool(keep[7]), bool(keep[18])), (23, False, False))
    for forced in ('affine', 'bilinear', 'quadratic'):
        keep, *_ = C._surface_keep(X, Y, a, c, tol=150.0, degree=forced)
        check(f'surface ({forced}, cannot fit the curvature): wrong chips still rejected',
              (bool(keep[7]), bool(keep[18])), (False, False))
    am, cm = C._mode_center(a, 50), C._mode_center(c, 50)
    check('constant rule on a varying field keeps almost nothing',
          int(((abs(a - am) <= 150) & (abs(c - cm) <= 150)).sum()) <= 2)
    f = {'x0': 0.0, 'y1': 100.0, 'cw': 50.0, 'ch': 50.0, 'nx': 2, 'ny': 2,
         'dx': [[0.0, 100.0], [0.0, 100.0]], 'dy': [[10.0, 10.0], [30.0, 30.0]]}
    check('field: bilinear between cell centres', E.field_offset(f, 50.0, 50.0), (50.0, 20.0))
    check('field: constant beyond the outer centres', E.field_offset(f, -500.0, 500.0), (0.0, 10.0))


def test_weights_and_failfast(tmp):
    """The user's log: a torch-hub folder pointed elsewhere made kornia miss
    weights it already had and retry a download in every window. Weights must
    be found in kornia's default folder; a model that cannot be loaded must
    fail once, fast, and not be retried."""
    import torch as th
    import automatch_engine as E
    cfg = E.PipelineConfig(num_features=500, use_amp=False)
    default_ckpt = os.path.join(E._default_hub_dir(), 'checkpoints', 'depth-save.pth')
    old = th.hub.get_dir()
    empty = os.path.join(tmp, 'empty_hub')
    os.makedirs(os.path.join(empty, 'checkpoints'), exist_ok=True)
    calls = []
    orig = E._ORIG_LOAD_STATE_DICT
    default_hub = E._default_hub_dir

    def offline(url, *a, **k):  # no network: every download fails
        calls.append(url)
        raise OSError(f'offline: {url}')
    try:
        th.hub.set_dir(empty)
        E._ORIG_LOAD_STATE_DICT = offline
        if os.path.exists(default_ckpt):
            m = E.build_detector('disk', cfg)
            try:
                m.warmup()
                check('weights: kornia default folder used when the hub folder points elsewhere',
                      not calls and m.disk is not None)
            finally:
                m.unload_model()
        else:
            print('SKIP  weights: DISK depth weights not in the default folder here')
        E.FAILED_MODELS.clear()
        # from here on the default folder is not searched either, so the
        # weights count as missing whatever this machine has cached
        E._default_hub_dir = lambda: empty
        m = E.build_detector('loftr', cfg, {'pretrained': ['indoor_new']})
        n_before = len(calls)
        try:
            m.warmup()
            check('fail-fast: missing weights raise ModelLoadError', False)
        except E.ModelLoadError as e:
            check('fail-fast: missing weights raise ModelLoadError', True)
            check('fail-fast: message names the folders searched', 'Looked for its weights in' in str(e))
        m2 = E.build_detector('loftr', cfg, {'pretrained': ['indoor_new']})
        try:
            m2.warmup()
        except E.ModelLoadError:
            pass
        check('fail-fast: download tried once for the whole job, not per window', len(calls) - n_before, 1)
    finally:
        E._ORIG_LOAD_STATE_DICT = orig
        E._default_hub_dir = default_hub
        th.hub.set_dir(old)
        E.FAILED_MODELS.clear()

    ests = {1: {'method': 'failed', 'dx': 0.0, 'dy': 0.0},
            2: {'method': 'matcher@60m', 'dx': 1000.0, 'dy': 2000.0, 'field': None},
            3: {'method': 'matcher@60m', 'dx': 9000.0, 'dy': 9000.0, 'field': None}}
    pairs = [{'pair_id': 1, 'bounds': [0, 0, 10, 10]}, {'pair_id': 2, 'bounds': [10, 0, 20, 10]},
             {'pair_id': 3, 'bounds': [500, 500, 510, 510]}]
    E.AutoMatchPipeline._borrow_failed_offsets(pairs, ests)
    check('coarse: failed pair borrows the nearest pair\'s offset',
          (ests[1]['method'], ests[1]['dx'], ests[1]['dy'], ests[1]['margin_scale']),
          ('from-pair-2', 1000.0, 2000.0, 2.0))


def test_gui_dialog():
    """Configure… dialog, headless: values stored, summary, job round trip."""
    try:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        import DPQED_automatch as G
    except Exception as e:
        print(f'SKIP  GUI dialog ({type(e).__name__}: {e})')
        return
    import automatch_engine as E
    app = G.QtWidgets.QApplication.instance() or G.QtWidgets.QApplication([])
    w = G.AutoMatchWindow()
    try:
        w._fill_detector_table(E.available_detectors())
        dlg = w.open_param_dialog('sift', run=False)
        box = dlg.widgets['rootsift']
        for cb, opt in box._items:
            cb.setChecked(True)                       # rootsift: on AND off
        dlg._accept()
        check('gui: dialog stores the ticked values', w.detector_params.get('sift'), {'rootsift': [True, False]})
        r = w._row_of('sift')
        text = w.det_table.cellWidget(r, 3).findChild(G.QtWidgets.QLabel, 'summary').text()
        check('gui: summary shows the variants', '2 variants' in text)
        w.det_table.item(r, 0).setCheckState(G.CHECKED)
        job = w.get_job()
        check('gui: job carries detector_params', job['detector_params'].get('sift'), {'rootsift': [True, False]})
        dlg = w.open_param_dialog('aliked', run=False)
        dlg.widgets['nms_radius'].setText('2, x')
        try:
            dlg.values()
            check('gui: bad number rejected', False)
        except ValueError:
            check('gui: bad number rejected', True)
        dlg = w.open_param_dialog('disk', run=False)
        dlg.reset_defaults()
        dlg._accept()
        check('gui: all defaults -> nothing stored', 'disk' in w.detector_params, False)
        w.set_job({**G.DEFAULT_JOB, 'detectors': ['disk_depth', 'disk_epipolar']})
        check('gui: legacy disk names -> disk with both checkpoints',
              (w.detector_params.get('disk'), w.det_table.item(w._row_of('disk'), 0).checkState() == G.CHECKED),
              ({'checkpoint': ['depth', 'epipolar']}, True))
        # the whole catalogue: rows not offered are shown greyed and never selected
        w._fill_detector_table(E.available_detectors(), [
            {'tag': 'disk-lg', 'kind': 'kornia', 'use': "'disk' with lgm",
             'reason': "disk is provided by kornia: use 'disk' with lgm"}])
        r = w._row_of('imw-disk-lg')
        check('gui: not-offered row shown, greyed, with the kornia replacement',
              (r is not None, w._selectable(r), w.det_table.cellWidget(r, 2).text()),
              (True, False, "use kornia 'disk' with lgm"))
        w.set_job({**G.DEFAULT_JOB, 'detectors': ['sift', 'imw-disk-lg']})
        job = w.get_job()
        check('gui: default matchers ticked, optional ones not; nothing stored when unchanged',
              (job['detectors'], job['matchers']), (['sift'], {}))
        r = w._row_of('sift')
        for cb in w.det_table.cellWidget(r, 2).findChildren(G.QtWidgets.QCheckBox):
            if cb.text() == 'mnn':
                cb.setChecked(True)
        check('gui: ticking an optional matcher stores the selection',
              w.get_job()['matchers'].get('sift'), ['smnn', 'lgm', 'ada', 'mnn'])
        ev = {'event': 'weights', 'mode': 'check', 'torch_folders': ['/a/checkpoints'], 'hf_caches': [],
              'put_in': '/a/checkpoints', 'summary': {'files': 2, 'found': 1, 'missing': 1, 'detectors': 1},
              'missing': [{'file': 'x.pth', 'source': 'http://h/x.pth', 'kind': 'torch-hub', 'detectors': ['sift']}],
              'detectors': [{'detector': 'sift', 'status': 'missing', 'error': '', 'variants': [
                  {'variant': 'sift', 'status': 'missing', 'error': ''}], 'files': [
                  {'file': 'x.pth', 'status': 'MISSING', 'model': 'LightGlue (sift)', 'source': 'http://h/x.pth',
                   'folder': '', 'needed_by': ['sift'], 'all_variants': True},
                  {'file': 'y.pth', 'status': 'ok', 'model': 'SIFT', 'source': 'http://h/y.pth',
                   'folder': '/a/checkpoints', 'needed_by': ['sift'], 'all_variants': True}]}]}
        w._show_weights(ev)
        dlg = w._last_weights_dialog
        check('gui: weights dialog lists the files and the detector column says what is missing',
              (dlg.table.rowCount(), w.det_table.item(w._row_of('sift'), 4).text(),
               'x.pth\thttp://h/x.pth' in dlg.missing_text()), (2, '1 missing', True))
        dlg.close()
        w.windows.setText('1024, 3072')
        w.max_feat.setValue(32000)
        check('gui: keypoints per window shown, capped sizes flagged',
              ('1024 px: 9,437' in w.lbl_kp.text(), '3072 px: 32,000 (capped; 84,934' in w.lbl_kp.text()), (True, True))
        w.max_feat.setValue(90000)
        check('gui: max keypoints reaches the job', w.get_job()['max_num_features'], 90000)
        w.settings = G.QtCore.QSettings(os.path.join(tempfile.mkdtemp(), 'gui_test.ini'), G.QtCore.QSettings.IniFormat
                                        if G.QT_API == 'PyQt5' else G.QtCore.QSettings.Format.IniFormat)
        pd_ = w.pack_dialog(run=False)
        pd_.local.setText('V:\\ICIGDev\\GPUPOC\\input\\dqe\\g1a\\set9')
        pd_.windows.setText('3072')
        pd_.all_det.setChecked(True)
        a, job = pd_.args()
        check('gui pack: server folder from the share mapping, remembered defaults, overrides',
              (pd_.server.text(), a[a.index('--server-weights') + 1], a[a.index('--gpu-mb') + 1],
               'window_sizes=[3072]' in a, 'detectors=all' in a),
              ('/maintenance/ICIGDev/GPUPOC/input/dqe/g1a/set9',
               '/maintenance/ICIGDev/GPUPOC/input/dqe/imw_runtime/imw_cache', '40000', True, True))
        pd_.close()
        w._show_truth({'event': 'truth', 'n_truth': 3, 'files': {}, 'top': [
            {'rank': 1, 'channel': 'band1', 'detector': 'sift', 'matcher_family': 'lgm',
             'best_matcher_setting': 'lgm', 'ransac': 'magsac2', 'truth_rmse_m': 12.5,
             'truth_reached': 3, 'truth_total': 3}]})
        check('gui: truth ranking shown and summarised',
              (w._last_truth_dialog.table.rowCount(), 'sift + lgm' in w.lbl_status.text()), (1, True))
        w._last_truth_dialog.close()
    finally:
        if w.proc is not None:
            w.proc.kill()
            w.proc.waitForFinished(5000)
        w.close()


def test_imcui_mock():
    """imcui bridge against a mock imcui (real registry shapes) and the full
    catalogue: kornia-covered rows refused with the kornia replacement named,
    rows the installed imcui lacks reported, the rest registered and runnable."""
    import types
    import numpy as np
    sp = {'output': 'f-sp', 'model': {'name': 'superpoint', 'max_keypoints': 4096, 'nms_radius': 4},
          'preprocessing': {'grayscale': True}}
    feat = {'superpoint_max': sp, 'aliked-n16': {'output': 'f-a', 'model': {'name': 'aliked'}, 'preprocessing': {}},
            'disk': {'output': 'f-d', 'model': {'name': 'disk'}, 'preprocessing': {}},
            'sift': {'output': 'f-s', 'model': {'name': 'sift', 'rootsift': True}, 'preprocessing': {}},
            'dedode': {'output': 'f-dd', 'model': {'name': 'dedode'}, 'preprocessing': {}},
            'rootsift': {'output': 'f-r', 'model': {'name': 'dog', 'descriptor': 'rootsift'}, 'preprocessing': {}},
            'hardnet': {'output': 'f-h', 'model': {'name': 'dog', 'descriptor': 'hardnet'}, 'preprocessing': {}},
            'r2d2': {'output': 'f-r2', 'model': {'name': 'r2d2'}, 'preprocessing': {}}}

    def lg(f):
        return {'output': 'm', 'model': {'name': 'lightglue', 'features': f}, 'preprocessing': {}}
    match = {'superpoint-lightglue': lg('superpoint'), 'aliked-lightglue': lg('aliked'),
             'disk-lightglue': lg('disk'), 'sift-lightglue': lg('sift'),
             'superglue': {'output': 'm', 'model': {'name': 'superglue'}, 'preprocessing': {}},
             'NN-mutual': {'output': 'm', 'model': {'name': 'nearest_neighbor', 'do_mutual_check': True},
                           'preprocessing': {}},
             'adalam': {'output': 'm', 'model': {'name': 'adalam'}, 'preprocessing': {}}}
    dense = {n: {'output': 'm', 'model': {'name': n}, 'preprocessing': {}}
             for n in ('loftr', 'eloftr', 'aspanformer', 'roma', 'dkm', 'xfeat_dense', 'xfeat_lightglue')}
    dense['minima_loftr'] = {'output': 'm', 'model': {'name': 'loftr', 'model_name': 'minima_loftr.ckpt'},
                             'preprocessing': {}}

    class API:
        def __init__(self, conf, device, detect_threshold, max_keypoints, match_threshold):
            pass

        def __call__(self, a, b):
            ys, xs = np.mgrid[20:a.shape[0] - 20:25, 20:a.shape[1] - 20:25]
            k0 = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
            return {'mkeypoints0_orig': k0, 'mkeypoints1_orig': k0 + [7.0, -4.0]}

    mods = {'imcui': types.ModuleType('imcui'), 'imcui.api': types.ModuleType('imcui.api'),
            'imcui.hloc': types.ModuleType('imcui.hloc')}
    mods['imcui.api'].ImageMatchingAPI = API
    for sub, confs in (('extract_features', feat), ('match_features', match), ('match_dense', dense)):
        m = types.ModuleType('imcui.hloc.' + sub)
        m.confs = confs
        mods['imcui.hloc.' + sub] = m
        setattr(mods['imcui.hloc'], sub, m)
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        import automatch_engine as E
        import automatch_imcui as IM
        IM._API = None
        res = IM.register_imcui_detectors()
        check('imcui: registered only what kornia lacks (MINIMA-LoFTR exempt: other weights)',
              sorted(res['registered']),
              ['imw-aspanformer', 'imw-dkm', 'imw-eloftr', 'imw-minima-loftr', 'imw-r2d2-nn', 'imw-roma',
               'imw-sp-lg', 'imw-sp-nn', 'imw-sp-sg', 'imw-xfeat-lg'])
        by_kind = {}
        for row in res['not_offered']:
            by_kind.setdefault(row['kind'], {})[row['tag']] = row
        check('imcui: kornia-covered rows refused', sorted(by_kind.get('kornia', {})),
              ['aliked-lg', 'dedode-nn', 'disk-adalam', 'disk-lg', 'hardnet-nn', 'loftr',
               'rootsift-nn', 'sift-lg', 'xfeat-dense'])
        use = {t: r['use'] for t, r in by_kind.get('kornia', {}).items()}
        check('imcui: refusals name the kornia replacement',
              (use['hardnet-nn'], use['rootsift-nn'], use['disk-lg'], use['sift-lg'], use['disk-adalam'],
               use['xfeat-dense']),
              ("'dog' (descriptor hardnet) with mnn", "'sift' (RootSIFT on) with mnn", "'disk' with lgm",
               "'sift' with lgm", "'disk' with ada", "'xfeatstar'"))
        unavailable = by_kind.get('unavailable', {})
        check('imcui: rows the installed imcui lacks are reported',
              ('rord-nn' in unavailable and 'omniglue' in unavailable
               and 'not in the installed imcui' in unavailable['rord-nn']['reason']), True)
        check('imcui: every catalogue row is either offered or explained',
              len(res['registered']) + len(res['not_offered']), len(__import__('imw_configs').catalog_rows()))
        m = E.build_detector('imw-sp-lg', E.PipelineConfig(num_features=100))
        rng = np.random.default_rng(0)
        img = rng.random((300, 300)).astype('float32') * 100 + 1
        meta = {'x01': 1000.0, 'y01': 5000.0, 'xres1': 10.0, 'yres1': -10.0, 'x02': 0.0, 'y02': 5000.0,
                'xres2': 10.0, 'yres2': -10.0, 'nisar_crop_row_offset': 0, 'nisar_crop_col_offset': 0,
                'pair_id': 1, 'nisar_pol': 'band1', 's1_ref_tag': 'C1'}
        rec = m._process_single_window(img, img, 0, 0, 0, 0, meta, 'internal', None, 0.0, 0.0)
        check('imcui: window record built', rec is not None and len(rec['X1']) > 10)
        if rec:
            dx = float(np.median(np.array(rec['X1']) - np.array(rec['X2'])))
            dy = float(np.median(np.array(rec['Y1']) - np.array(rec['Y2'])))
            check('imcui: map offset from pixel shift', (round(dx, 3), round(dy, 3)), (930.0, -40.0))
            check('imcui: file id parses', E.MatchStatistics._parse_filename(
                rec['file_id'] + '_aff_magsac_2_0.99_.csv')['detector'], 'imw-sp-lg')
        # per-model parameters come from the registry conf; overridden keys hidden
        names = [p.name for p in E.detector_param_specs('imw-sp-lg')]
        check('imcui params: thresholds + registry keys, API-owned keys hidden',
              (names[:2], 'feature.nms_radius' in names, 'feature.max_keypoints' in names),
              (['api.detect_threshold', 'api.match_threshold'], True, False))
        vs = E.build_variants('imw-sp-lg', E.PipelineConfig(num_features=100),
                              {'feature.nms_radius': [3, 4], 'api.match_threshold': [0.3]})
        check('imcui variants: one per combination, distinct names',
              len({v.get_filename_prefix() for v in vs}), 2)
        v0 = vs[0]
        check('imcui override reaches the conf',
              (v0.imw_conf['feature']['model']['nms_radius'], v0.match_threshold,
               v0.imw_conf['matcher']['model']['match_threshold']), (3, 0.3, 0.3))
        check("imcui conf carries both 'dense' and 'standalone'",
              (v0.imw_conf['dense'], v0.imw_conf['standalone']), (False, False))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        try:
            import automatch_engine as E
            for k in [k for k in E.EXTERNAL_DETECTORS if k.startswith('imw-')]:
                E.EXTERNAL_DETECTORS.pop(k)
        except Exception:
            pass


def _texture(size=320, seed=3):
    import numpy as np
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    for scale, amp in ((64, 1.0), (16, 0.6), (4, 0.35)):
        small = rng.random((size // scale + 2, size // scale + 2)).astype(np.float32)
        img += amp * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(size * size // 600):
        cv2.circle(img, (int(rng.random() * size * 16), int(rng.random() * size * 16)),
                   int(rng.integers(2, 9)) * 16, float(rng.random() * 2), -1, lineType=cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = 50 + 1000 * (img - img.min()) / (img.max() - img.min())
    shifted = cv2.warpAffine(img, np.float32([[1, 0, 7], [0, 1, -4]]), (size, size), borderValue=0)
    return img, shifted


_META = {'x01': 0.0, 'y01': 0.0, 'xres1': 1.0, 'yres1': -1.0, 'x02': 0.0, 'y02': 0.0, 'xres2': 1.0,
         'yres2': -1.0, 'nisar_crop_row_offset': 0, 'nisar_crop_col_offset': 0, 'pair_id': 1,
         'nisar_pol': 'band1', 's1_ref_tag': 'C1'}


def _shift_of(rec):
    import numpy as np
    if rec is None:
        return None
    d = np.array(rec['X1']) - np.array(rec['X2'])
    e = np.array(rec['Y1']) - np.array(rec['Y2'])
    return len(d), round(float(np.median(d))), round(float(np.median(e)))


def test_truth_helpers(tmp):
    """Ground-truth scoring of one configuration: local median near a truth
    point, error surface elsewhere, extrapolation flagged and not scored."""
    import numpy as np
    import pandas as pd
    import automatch_truth as T
    rng = np.random.default_rng(1)
    E0, N0 = 500000.0, 3000000.0
    field = lambda e, n: (4000.0 + 0.002 * (e - E0), -2500.0 + 0.001 * (n - N0))  # 2 / 1 m per km
    rows = []
    for ci, (ce, cn) in enumerate([(E0 + x, N0 + y) for x in range(0, 60001, 15000) for y in range(0, 60001, 15000)]):
        e = ce + rng.uniform(-3000, 3000, 40)
        n = cn + rng.uniform(-3000, 3000, 40)
        de, dn = field(e, n)
        rows.append(pd.DataFrame({'E': e, 'N': n, 'dE': de + rng.normal(0, 3, 40),
                                  'dN': dn + rng.normal(0, 3, 40), 'chip': ci}))
    pts = pd.concat(rows, ignore_index=True)
    te = np.array([E0 + 15000, E0 + 37500, E0 + 250000])
    tn = np.array([N0 + 30000, N0 + 22500, N0 + 30000])
    de, dn = field(te, tn)
    truth = pd.DataFrame({'truth_id': [1, 2, 3], 'E': te, 'N': tn, 'dE': de, 'dN': dn})
    summ, per = T.score_config({'points': pts}, truth, 5000.0)
    check('truth: local, surface, extrapolated', [r['method'] for r in per], ['local', 'surface', 'extrapolated'])
    check('truth: agreeing configuration scores a few metres',
          (summ['truth_reached'], summ['truth_rmse_m'] < 5.0), (2, True))
    truth2 = truth.assign(dE=truth['dE'] + 300.0)
    summ2, _ = T.score_config({'points': pts}, truth2, 5000.0)
    check('truth: a 300 m disagreement is measured', round(summ2['truth_rmse_m'] / 10) * 10, 300)
    # a steep east-west gradient (60 m/km, as on G1A scenes) and a truth point at
    # the edge of the matches: the median of the neighbours is pulled by the
    # gradient, a plane through them is not
    ge, gn = rng.uniform(0, 5000, 400), rng.uniform(-2500, 2500, 400)
    steep = pd.DataFrame({'E': ge, 'N': gn, 'dE': 4000 + 0.06 * ge + rng.normal(0, 3, 400),
                          'dN': -2500 + 0.01 * gn + rng.normal(0, 3, 400), 'chip': 0})
    med = T.estimate(steep, None, 0.0, 0.0, 5000.0, local='median')
    pla = T.estimate(steep, None, 0.0, 0.0, 5000.0, local='plane')
    check('truth: with a steep gradient the plane removes the median\'s bias',
          (abs(med[0] - 4000) > 100, abs(pla[0] - 4000) < 5, abs(pla[1] + 2500) < 5), (True, True, True))
    line = steep.assign(N=0.0)
    check('truth: neighbours on a line -> slope along it only', abs(T.estimate(line, None, 0.0, 0.0, 5000.0)[0] - 4000) < 5)
    ranked = T.rank(pd.DataFrame([{**summ, 'detector': 'a'}, {**summ2, 'detector': 'b'},
                                  {**summ, 'truth_reached': 0, 'truth_rmse_m': None, 'detector': 'c'}]))
    check('truth: ranking by coverage then RMSE', ranked['detector'].tolist(), ['a', 'b', 'c'])
    # a RIVAL truth file: errors recomputed in the working CRS from lon/lat
    from pyproj import Transformer
    to_ll = Transformer.from_crs('EPSG:32640', 'EPSG:4326', always_xy=True)
    ilo, ila = to_ll.transform(te, tn)
    rlo, rla = to_ll.transform(te - de, tn - dn)
    path = os.path.join(tmp, 'truth.csv')
    pd.DataFrame({'In_X': te, 'In_Y': tn, 'Ref_X': te - de, 'Ref_Y': tn - dn, 'DX_Err': de, 'DY_Err': dn,
                  'Row': [1, 2, 3], 'In_Lon': ilo, 'In_Lat': ila, 'Ref_Lon': rlo, 'Ref_Lat': rla}).to_csv(
        path, index=False, encoding='utf-8-sig')
    back = T.truth_in_crs(T.load_truth(path), 'EPSG:32640')
    check('truth: RIVAL file read back in the working CRS',
          (np.allclose(back['dE'], de, atol=0.01), np.allclose(back['N'], tn, atol=0.01)), (True, True))
    import automatch_job as J
    errs = J._check_gcp_file(path)
    check('truth: a RIVAL file given as manual GCP CSV is refused with the fix named',
          len(errs) == 1 and 'Ground truth CSV' in errs[0])
    try:
        pd.DataFrame({'a': [1]}).to_csv(os.path.join(tmp, 'bad.csv'), index=False)
        T.load_truth(os.path.join(tmp, 'bad.csv'))
        check('truth: file without lon/lat refused', False)
    except ValueError as e:
        check('truth: file without lon/lat refused', 'In_Lon' in str(e))


def test_server(tmp):
    """HTTP front end: health, token, job state, file download confined to the
    output folder (no subprocess: a finished job is faked)."""
    import threading
    import urllib.request
    import urllib.error
    from http.server import ThreadingHTTPServer
    import automatch_server as SV
    srv = SV.Server(os.path.join(tmp, 'srv_jobs'))
    out = os.path.join(tmp, 'srv_out')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'TRUTH_BY_DETECTOR_MATCHER.csv'), 'w').write('rank,detector\n1,sift\n')
    open(os.path.join(tmp, 'secret.txt'), 'w').write('x')
    j = SV.Job('j1', {'output_dir': out, 'detectors': ['sift']}, os.path.join(tmp, 'srv_jobs'))
    j.state = 'done'
    srv.jobs['j1'] = j
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), SV.make_handler(srv, 'tok'))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'

    def get(path, token='tok'):
        req = urllib.request.Request(base + path, headers={'Authorization': f'Bearer {token}'} if token else {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    try:
        check('server: health without token', get('/health', None)[0], 200)
        check('server: token required', get('/jobs', 'bad')[0], 401)
        code, body = get('/jobs/j1')
        check('server: job state', (code, json.loads(body)['state']), (200, 'done'))
        check('server: output file download', get('/jobs/j1/files/TRUTH_BY_DETECTOR_MATCHER.csv'),
              (200, 'rank,detector\n1,sift\n'))
        check('server: no file outside the output folder', get('/jobs/j1/files/..%2Fsecret.txt')[0], 404)
        check('server: unknown job', get('/jobs/nope')[0], 404)
        try:
            srv.submit({'detectors': ['sift']})
            check('server: job without output_dir refused', False)
        except ValueError:
            check('server: job without output_dir refused', True)
    finally:
        httpd.shutdown()


def test_gpuaas_args(tmp):
    """GPU-service entry: sys.argv[2] = output folder, sys.argv[3] = the
    comma-separated input_path (as the NISAR scripts take it)."""
    import automatch_gpuaas as G
    settings = os.path.join(tmp, 'settings.json')
    with open(settings, 'w') as f:
        json.dump({'detectors': ['sift', 'dog'], 'window_sizes': [3072], 'mode': 'run'}, f)
    ref = os.path.join(tmp, 'refdir')
    os.makedirs(ref, exist_ok=True)
    mode, job = G.build_job(['s.py', 'id', '/out/x', f'/in/scene.tif,{ref},/in/truth.csv,{settings}'])
    check('gpuaas: positional paths, settings file, output folder from the service',
          (mode, job['input_path'], job['reference_dir'], job['truth_csv'], job['detectors'],
           job['window_sizes'], job['output_dir'], job['temp_dir']),
          ('run', '/in/scene.tif', ref, '/in/truth.csv', ['sift', 'dog'], [3072], '/out/x',
           os.path.join('/out/x', '_cache')))
    mode, job = G.build_job(['s.py', 'id', '/out/y', 'weights,/in/scene.tif,window_sizes=2048|3072,coarse_local=false'])
    check('gpuaas: mode token and key=value overrides (| for lists)',
          (mode, job['window_sizes'], job['coarse_local'], job['input_path']),
          ('weights', [2048, 3072], False, '/in/scene.tif'))
    try:
        G.build_job(['s.py', 'id', '/o', '/a,/b,/c'])
        check('gpuaas: three folders refused', False)
    except ValueError:
        check('gpuaas: three folders refused', True)
    envfile = os.path.join(tmp, 'envmode.json')
    with open(envfile, 'w') as f:
        json.dump({'mode': 'env', 'env': {'AUTOMATCH_TEST_VAR': '42'}}, f)
    old = os.environ.pop('AUTOMATCH_TEST_VAR', None)
    try:
        mode, job = G.build_job(['s.py', 'id', '/o', envfile])
        check('gpuaas: env mode, and "env" settings applied before imports',
              (mode, os.environ.get('AUTOMATCH_TEST_VAR'), 'env' in job), ('env', '42', False))
    finally:
        os.environ.pop('AUTOMATCH_TEST_VAR', None)
        if old is not None:
            os.environ['AUTOMATCH_TEST_VAR'] = old
    text = G.environment_report({'weights_cache_dir': ''})
    check('gpuaas: environment report has packages, GPU, catalogue and weight folders',
          all(k in text for k in ('== Packages', 'kornia', '== GPU', '== Detector catalogue', 'dog ',
                                  '== Weight folders searched')))
    import automatch_job as J
    ex = J.expand_selection(J.normalize({'detectors': ['all-kornia'], 'matchers': 'all'}))
    import automatch_engine as E
    check('selection: "all-kornia" and matchers "all" expand to this machine\'s catalogue',
          (ex['detectors'] == list(E.KORNIA_DETECTORS), ex['matchers'].get('dog'),
           ex['matchers'].get('loftr')),
          (True, ['smnn', 'lgm', 'ada', 'mnn', 'snn', 'nn', 'fginn'], ['loftr_internal']))
    ex = J.expand_selection(J.normalize({'detectors': ['sift', 'dog'], 'matchers': {'sift': 'all', 'dog': ['lgm']}}))
    check('selection: per-detector "all" next to an explicit list',
          (ex['detectors'], ex['matchers']['sift'][-1], ex['matchers']['dog']), (['sift', 'dog'], 'fginn', ['lgm']))
    plain = J.normalize({'detectors': ['sift']})
    check('selection: a job without "all" is left as it is', J.expand_selection(plain) is plain)


def test_kornia_versions(tmp):
    """A kornia without ALIKED / XFeat (the GPU node has 0.8.1): those detectors
    are not offered, imcui's ALIKED / XFeat are; native libraries load in the
    order that keeps conda GDAL working next to pip wheels."""
    import automatch_engine as E
    import kornia.feature as KF
    saved = {n: getattr(KF, n) for n in ('ALIKED', 'XFeat')}
    try:
        for n in saved:
            delattr(KF, n)
        names = [d['name'] for d in E.available_detectors()]
        check('kornia 0.8.1: aliked / xfeat / xfeatstar not offered, the rest are',
              (sorted(E.kornia_unavailable()), 'aliked' in names, 'xfeat' in names, 'dog' in names),
              (['aliked', 'xfeat', 'xfeatstar'], False, False, True))
        check('kornia 0.8.1: imcui may provide ALIKED / XFeat, not DISK',
              ('aliked' in E.kornia_families(), 'xfeat_dense' in E.kornia_families(), 'disk' in E.kornia_families()),
              (False, False, True))
        try:
            E.build_detector('aliked', E.PipelineConfig())
            check('kornia 0.8.1: selecting aliked explains why it cannot run', False)
        except ValueError as e:
            check('kornia 0.8.1: selecting aliked explains why it cannot run', 'has no ALIKED' in str(e))
    finally:
        for n, v in saved.items():
            setattr(KF, n, v)
    check('kornia: restored', sorted(E.kornia_unavailable()), [])
    # a model that reads its checkpoint itself (imcui's r2d2): files found -> partial, not error
    import torch as th
    import automatch_weights as W
    hub = os.path.join(tmp, 'kv_hub')
    os.makedirs(os.path.join(hub, 'checkpoints'), exist_ok=True)
    th.save({'net': 'Net()'}, os.path.join(hub, 'checkpoints', 'reads_ckpt.pth'))

    class Reads(E.DiskBasedMatcher):
        def __init__(self, c):
            super().__init__(c)
            self.m = None

        def _build_models(self):
            self.m = self._load_model('Reads', lambda: 'net = ' + th.hub.load_state_dict_from_url(
                'http://example.invalid/reads_ckpt.pth')['net'])

        def detect_and_describe(self, a, b):
            return None

        def get_detector_name(self):
            return 'ext-reads'

        def get_available_matchers(self):
            return ['internal']
    old_dir, default_hub = th.hub.get_dir(), E._default_hub_dir
    try:
        th.hub.set_dir(hub)
        E._default_hub_dir = lambda: hub
        E.register_detector('ext-reads', lambda c, kw: Reads(c), families=('reads',), source='test')
        rep = W.check(detectors=['ext-reads'], imcui=False, quiet=True)
        v = rep['detectors'][0]['variants'][0]
        check('weights: files found but the model reads them itself -> partial, not error',
              (v['status'], 'files are found' in v['error'], rep['detectors'][0]['files'][0]['status']),
              ('partial', True, 'ok'))
    finally:
        th.hub.set_dir(old_dir)
        E._default_hub_dir = default_hub
        E.EXTERNAL_DETECTORS.pop('ext-reads', None)
    order = {}
    for mod in ('automatch_truth', 'automatch_engine', 'automatch_gpuaas', 'automatch_imcui'):
        out = subprocess.run([sys.executable, '-c', textwrap.dedent(f"""
            import sys; sys.path.insert(0, {HERE!r})
            import importlib, ctypes
            loaded = []
            import builtins
            real = builtins.__import__
            def spy(name, *a, **k):
                top = name.split('.')[0]
                if top in ('rasterio', 'pandas', 'torch', 'cv2', 'h5py') and top not in loaded:
                    loaded.append(top)
                return real(name, *a, **k)
            builtins.__import__ = spy
            import {mod}
            print(','.join(loaded))
        """)], capture_output=True, text=True)
        order[mod] = (out.stdout.strip().splitlines() or [''])[-1].split(',')[0]
    check('native: rasterio is the first compiled package each entry module imports',
          order, {m: 'rasterio' for m in order})


def test_warmup_and_padding():
    """A model failing on the synthetic warm-up pair no longer stops its
    detector (imcui SuperPoint found no keypoints in noise and was lost);
    missing weights still do. imcui windows are padded to multiples of 32."""
    import numpy as np
    import automatch_engine as E
    import automatch_imcui as IM
    import imw_configs

    class NoiseHater(E.DenseWindowMatcher):
        def match_images(self, a, b):
            raise IndexError('max(): Expected reduction dim 1 to have non-zero size.')

        def get_detector_name(self):
            return 'noisehater'

    class NoWeights(NoiseHater):
        def _build_models(self):
            self._load_model('nowhere', lambda: (_ for _ in ()).throw(OSError('offline')))
    cfg = E.PipelineConfig(num_features=100, use_amp=False)
    try:
        NoiseHater(cfg).warmup()
        check('warmup: a failure on the synthetic pair is reported, not fatal', True)
    except Exception as e:
        check('warmup: a failure on the synthetic pair is reported, not fatal', f'{type(e).__name__}: {e}')
    E.FAILED_MODELS.pop('nowhere', None)
    try:
        NoWeights(cfg).warmup()
        check('warmup: missing weights still stop the detector', False)
    except E.ModelLoadError:
        check('warmup: missing weights still stop the detector', True)
    finally:
        E.FAILED_MODELS.pop('nowhere', None)
    a = np.zeros((100, 70), np.uint8)
    old = imw_configs.RESIZE_MAX
    try:
        shp = IM._pad_for_model(a).shape
        imw_configs.RESIZE_MAX = 64
        shp_big = IM._pad_for_model(a).shape
    finally:
        imw_configs.RESIZE_MAX = old
    check('imcui padding: multiples of 32; a square when imcui will shrink it', (shp, shp_big), ((128, 96), (128, 128)))


def test_sliced_matching():
    """Matching in slices (for keypoint counts whose full distance table
    would not fit in memory) gives kornia's matches; the keypoint cap is a
    setting."""
    import numpy as np
    import torch as th
    import kornia.feature as KF
    import automatch_engine as E
    import automatch_job as J
    th.manual_seed(0)
    n = 2000
    d1 = th.nn.functional.normalize(th.randn(n, 128), dim=1)
    perm = th.randperm(n)
    d2 = th.nn.functional.normalize(d1[perm][:1700] + 0.08 * th.randn(1700, 128), dim=1)
    old = E.MATCH_SLICE_ELEMS
    E.MATCH_SLICE_ELEMS = 1700 * 97          # ~20 slices
    try:
        same = {}
        for name, ref in (('nn', KF.match_nn(d1, d2)), ('mnn', KF.match_mnn(d1, d2)),
                          ('snn', KF.match_snn(d1, d2, 0.8)), ('smnn', KF.match_smnn(d1, d2, 0.95))):
            got = E.match_sliced(name, d1, d2, 0.8 if name == 'snn' else 0.95)
            same[name] = ({tuple(r) for r in ref[1].tolist()} == {tuple(r) for r in got[1].tolist()}
                          and len(ref[1]) > 0)
        check('slices: nn / mnn / snn / smnn give kornia\'s matches', same, {k: True for k in same})
        img, img2 = _texture(384)
        n1, n2 = E.BaseMatcher._norm_img(img, 0.0), E.BaseMatcher._norm_img(img2, 0.0)
        t1, t2 = th.from_numpy(n1)[None, None].float(), th.from_numpy(n2)[None, None].float()
        m = E.build_detector('sift', E.PipelineConfig(num_features=1500, use_amp=False))
        with th.inference_mode():
            l1, f1, l2, f2 = m.detect_and_describe(t1, t2)
        f1, f2, hw = f1.squeeze(0), f2.squeeze(0), th.tensor(t1.shape[2:])
        th.manual_seed(3)
        ref = KF.match_adalam(f1, f2, l1, l2, config=dict(m.adalam_config), hw1=hw, hw2=hw)
        E.MATCH_SLICE_ELEMS = f2.shape[0] * 61
        th.manual_seed(3)
        got = E.match_adalam_sliced(f1, f2, l1, l2, dict(m.adalam_config), hw, hw)
        check('slices: AdaLAM gives kornia\'s matches', ({tuple(r) for r in ref[1].tolist()}
                                                        == {tuple(r) for r in got[1].tolist()}, len(ref[1]) > 50),
              (True, True))
        fg = E.match_fginn_sliced(f1, f2, l2, 0.8, 10.0, False)
        c1, c2 = KF.get_laf_center(l1)[0], KF.get_laf_center(l2)[0]
        d = (c1[fg[1][:, 0]] - c2[fg[1][:, 1]]).numpy()
        check('slices: FGINN (row by row) finds the shift',
              (len(d) > 50, float(np.mean((np.abs(d[:, 0] + 7) < 1.5) & (np.abs(d[:, 1] - 4) < 1.5))) > 0.85),
              (True, True))
    finally:
        E.MATCH_SLICE_ELEMS = old
    big = E.PipelineConfig(num_features=40000, use_amp=False)
    m = E.build_detector('sift', big)
    a = th.zeros(1, 40000, 128)
    check('slices: used above the table limit only', (E._needs_slices(a[0], a[0]),
                                                      E._needs_slices(a[0, :3000], a[0, :3000])), (True, False))
    cfg = J._config_for(J.normalize({'max_num_features': 90000}), 3072, None, 45.0, '/tmp/x')
    cfg2 = J._config_for(J.normalize({}), 3072, None, 45.0, '/tmp/x')
    check('keypoints: the cap is a job setting (density kept at 3072 px when raised)',
          (cfg.num_features, cfg2.num_features), (int(9000 * 3072 * 3072 / 1e6), 32000))


def test_kornia_catalogue():
    """The matchers on a synthetic pair shifted by (7, -4) px: every kornia
    matcher recovers it, and LightGlue gets matches for SIFT / DoG-HardNet
    (their (1, N, D) descriptors once made it return nothing)."""
    import automatch_engine as E
    img, img2 = _texture()
    cfg = E.PipelineConfig(num_features=600, use_amp=False, smnn_thresholds=[0.95])
    m = E.build_detector('sift', cfg)
    got = {mn: _shift_of(m._process_single_window(img, img2, 0, 0, 0, 0, dict(_META), mn,
                                                  0.95 if mn == 'smnn' else None, 0.0, 0.0))
           for mn in ['smnn', 'ada'] + E.GENERIC_MATCHERS}
    check('matchers: smnn / ada / mnn / snn / nn / fginn recover the shift',
          {k: (v[1:] if v else None) for k, v in got.items()}, {k: (-7, -4) for k in got})
    for det, params in (('sift', {}), ('dog', {})):
        m = E.build_detector(det, cfg, params)
        fname = os.path.join(E._default_hub_dir(), 'checkpoints', {
            'sift': 'sift_lightglue_v0-1_arxiv-pth', 'dog': 'doghardnet_v0-1_arxiv-pth'}[det])
        if det == 'dog' and not os.path.exists(os.path.join(E._default_hub_dir(), 'checkpoints',
                                                            'checkpoint_liberty_with_aug.pth')):
            print('SKIP  DoG-HardNet: HardNet weights not cached here')
            continue
        if not os.path.exists(fname):
            print(f'SKIP  {det} LightGlue: weights not cached here')
            continue
        rec = m._process_single_window(img, img2, 0, 0, 0, 0, dict(_META), 'lgm', None, 0.0, 0.0)
        res = _shift_of(rec)
        check(f'LightGlue: {det} gets matches and the shift', (res[0] > 50, res[1:]) if res else None,
              (True, (-7, -4)))
        m.unload_model()


def test_weights_check(tmp):
    """automatch_weights against a private weights folder: found / missing /
    truncated reported with folder and URL, nothing downloaded, nothing read
    in quick mode; --load reads; the HuggingFace lookup finds imcui files in
    other caches."""
    import types
    import torch as th
    import automatch_engine as E
    import automatch_imcui as IM
    import automatch_weights as W
    hub = os.path.join(tmp, 'wt_hub')
    ck = os.path.join(hub, 'checkpoints')
    os.makedirs(ck, exist_ok=True)
    th.save({'not': th.zeros(1)}, os.path.join(ck, 'depth-save.pth'))       # wrong content on purpose
    th.save({'x': th.zeros(1000)}, os.path.join(tmp, 'full.pth'))
    with open(os.path.join(tmp, 'full.pth'), 'rb') as f:
        head = f.read(200)
    with open(os.path.join(ck, 'disk_lightglue_v0-1_arxiv-pth'), 'wb') as f:
        f.write(head)                                                    # an interrupted download
    old_dir, default_hub, orig = th.hub.get_dir(), E._default_hub_dir, E._ORIG_LOAD_STATE_DICT
    calls = []

    def offline(url, *a, **k):
        calls.append(url)
        raise OSError('offline')
    try:
        th.hub.set_dir(hub)
        E._default_hub_dir = lambda: hub
        E._ORIG_LOAD_STATE_DICT = offline
        rep = W.check(detectors=['disk'], imcui=False, quiet=True)
        files = {f['file']: f for f in rep['detectors'][0]['files']}
        check('weights: found / missing / truncated',
              (files['depth-save.pth']['status'], files['epipolar-save.pth']['status'],
               files['disk_lightglue_v0-1_arxiv-pth']['status']), ('ok', 'MISSING', 'TRUNCATED'))
        check('weights: folder and URL reported',
              (files['depth-save.pth']['folder'] == ck, files['epipolar-save.pth']['source'].endswith(
                  'epipolar-save.pth'), files['epipolar-save.pth']['needed_by']), (True, True, ['disk_epipolar']))
        check('weights: quick check downloads nothing and reads nothing (the bad file passed)',
              (calls, [v['status'] for v in rep['detectors'][0]['variants']]), ([], ['missing', 'missing']))
        text = W.format_report(rep)
        check('weights: report says where to put the missing files',
              'epipolar-save.pth' in text and 'UNDER THE NAME SHOWN' in text and ck in text)
        rep = W.check(detectors=['keynet'], imcui=False, quiet=True)
        ori = {f['file']: f for f in rep['detectors'][0]['files']}.get('OriNet.pth', {})
        check('weights: OriNet needed only by the non-upright KeyNet variants',
              sorted(ori.get('needed_by', [])), ['keynet-aff0-up0', 'keynet-up0'])
        rep = W.check(job={'detectors': ['disk'], 'detector_params': {'disk': {'checkpoint': ['epipolar']}},
                           'matchers': {'disk': ['smnn']}}, imcui=False, quiet=True)
        check('weights: a job checks only what it runs (no LightGlue without lgm)',
              [f['file'] for f in rep['detectors'][0]['files']], ['epipolar-save.pth'])
        rep = W.check(detectors=['disk'], imcui=False, quiet=True, load=True)
        st = {v['variant']: v for v in rep['detectors'][0]['variants']}
        check('weights: --load reads the files (the wrong content is caught)',
              (st['disk_depth']['status'], 'extractor' in st['disk_depth']['error'],
               st['disk_epipolar']['status'], calls), ('error', True, 'missing', []))
        check('weights: tracing leaves nothing behind',
              (E.WEIGHTS_TRACE_HOOK, dict(E.FAILED_MODELS), E.MODEL_LABELS,
               th.nn.Module.load_state_dict.__name__), (None, {}, [], 'load_state_dict'))
    finally:
        th.hub.set_dir(old_dir)
        E._default_hub_dir = default_hub
        E._ORIG_LOAD_STATE_DICT = orig

    # HuggingFace: a file cached in the default HF cache is used although
    # HF_HOME points at the (empty) weights folder -- as for kornia's weights
    fake = types.ModuleType('huggingface_hub')
    fd = types.ModuleType('huggingface_hub.file_download')
    consts = types.ModuleType('huggingface_hub.constants')
    consts.HF_HUB_CACHE = os.path.join(tmp, 'wf', 'huggingface', 'hub')
    hf_calls = []

    def try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        p = os.path.join(cache_dir or '', repo_id.replace('/', '--'), filename)
        return p if os.path.isfile(p) else None

    def hf_hub_download(repo_id, filename, **kw):
        hf_calls.append(filename)
        raise OSError('offline')
    fake.try_to_load_from_cache, fake.hf_hub_download, fake.constants = try_to_load_from_cache, hf_hub_download, consts
    fd.hf_hub_download = hf_hub_download
    saved = {k: sys.modules.get(k) for k in ('huggingface_hub', 'huggingface_hub.file_download',
                                             'huggingface_hub.constants')}
    old_xdg, old_orig = os.environ.get('XDG_CACHE_HOME'), IM._ORIG_HF
    try:
        sys.modules.update({'huggingface_hub': fake, 'huggingface_hub.file_download': fd,
                            'huggingface_hub.constants': consts})
        os.environ['XDG_CACHE_HOME'] = os.path.join(tmp, 'xdg')
        target = os.path.join(tmp, 'xdg', 'huggingface', 'hub', 'Realcat--imcui_checkpoints', 'roma')
        os.makedirs(target, exist_ok=True)
        open(os.path.join(target, 'roma_outdoor.pth'), 'wb').write(b'x' * 10)
        IM._ORIG_HF = None
        check('weights: HF lookup installed', IM.install_hf_lookup(), True)
        got = fake.hf_hub_download('Realcat/imcui_checkpoints', 'roma/roma_outdoor.pth')
        check('weights: imcui file found in the default HuggingFace cache, no download',
              (got == os.path.join(target, 'roma_outdoor.pth'), hf_calls), (True, []))
        tr = W.Tracer()
        with tr.active():
            miss = fake.hf_hub_download('Realcat/imcui_checkpoints', 'dkm/dkm_outdoor.pth')
        check('weights: traced HF request, missing file answered with a stand-in',
              (tr.requests[-1]['kind'], tr.requests[-1]['status'], tr.requests[-1]['repo_file'],
               miss.startswith(W.STUB_PREFIX), hf_calls), ('hf', 'MISSING', 'dkm/dkm_outdoor.pth', True, []))
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if old_xdg is None:
            os.environ.pop('XDG_CACHE_HOME', None)
        else:
            os.environ['XDG_CACHE_HOME'] = old_xdg
        IM._ORIG_HF = old_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2e', action='store_true')
    ap.add_argument('--gui', action='store_true', help='also test the Configure… dialog (needs Qt)')
    ap.add_argument('--rival', default='')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='automatch_test_')
    try:
        test_refs(tmp)
        test_rival_parity(a.rival)
        test_engine_helpers()
        test_nisar_h5(tmp)
        test_detector_params()
        test_distortion_helpers()
        test_weights_and_failfast(tmp)
        test_imcui_mock()
        test_weights_check(tmp)
        test_truth_helpers(tmp)
        test_server(tmp)
        test_gpuaas_args(tmp)
        test_kornia_versions(tmp)
        test_warmup_and_padding()
        test_sliced_matching()
        test_kornia_catalogue()
        if a.gui:
            test_gui_dialog()
        if a.e2e:
            test_e2e(tmp)
            test_e2e_variants(tmp)
            test_e2e_rerun(tmp)
            test_e2e_distortion(tmp)
            test_e2e_missing_weights(tmp)
            test_e2e_truth(tmp)
            test_pack(tmp)
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(FAILS)} failure(s)' + (': ' + ', '.join(FAILS) if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
