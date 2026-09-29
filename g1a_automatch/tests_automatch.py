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
def _run_job(job, tmp):
    path = os.path.join(tmp, 'job.json')
    with open(path, 'w') as f:
        json.dump(job, f)
    out = subprocess.run([sys.executable, os.path.join(HERE, 'automatch_job.py'), 'run', path],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
    return out.returncode


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


def test_imcui_mock():
    """imcui bridge against a mock imcui (real registry shapes): kornia-covered
    models refused, the rest registered and runnable through DenseWindowMatcher."""
    import types
    import numpy as np
    sp = {'output': 'f-sp', 'model': {'name': 'superpoint', 'max_keypoints': 4096}, 'preprocessing': {'grayscale': True}}
    feat = {'superpoint_max': sp, 'aliked-n16': {'output': 'f-a', 'model': {'name': 'aliked'}, 'preprocessing': {}},
            'disk': {'output': 'f-d', 'model': {'name': 'disk'}, 'preprocessing': {}},
            'xfeat': {'output': 'f-x', 'model': {'name': 'xfeat'}, 'preprocessing': {}},
            'sift': {'output': 'f-s', 'model': {'name': 'sift'}, 'preprocessing': {}}}
    def lg(f):
        return {'output': 'm', 'model': {'name': 'lightglue', 'features': f}, 'preprocessing': {}}
    match = {'superpoint-lightglue': lg('superpoint'), 'aliked-lightglue': lg('aliked'),
             'disk-lightglue': lg('disk'), 'xfeat_lightglue': lg('xfeat'), 'sift-lightglue': lg('sift'),
             'superglue': {'output': 'm', 'model': {'name': 'superglue'}, 'preprocessing': {}}}
    for n in ('loftr', 'eloftr', 'aspanformer', 'roma', 'dkm', 'xfeat_dense'):
        match[n] = {'output': 'm', 'model': {'name': n}, 'preprocessing': {}}

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
    for sub, confs in (('extract_features', feat), ('match_features', match), ('match_dense', match)):
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
        check('imcui: registered only non-kornia algorithms', sorted(res['registered']),
              ['imw-aspanformer', 'imw-dkm', 'imw-eloftr', 'imw-roma', 'imw-sp-lg', 'imw-sp-sg'])
        check('imcui: refused kornia-covered', sorted(t for t, _ in res['skipped']),
              ['aliked-lg', 'disk-lg', 'loftr', 'sift-lg', 'xfeat-dense', 'xfeat-lg'])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e2e', action='store_true')
    ap.add_argument('--rival', default='')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='automatch_test_')
    try:
        test_refs(tmp)
        test_rival_parity(a.rival)
        test_engine_helpers()
        test_nisar_h5(tmp)
        test_imcui_mock()
        if a.e2e:
            test_e2e(tmp)
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(FAILS)} failure(s)' + (': ' + ', '.join(FAILS) if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
