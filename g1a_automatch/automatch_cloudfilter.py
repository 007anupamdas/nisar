"""
Cloud masking after the fact: reject the match points of finished jobs that
fall on cloud, and score again what is left.

    python automatch_job.py cloudfilter <job folder | folder of jobs>
        [--jobs 335-366] [--method vnir|omnicloudmask|csmask|ir|swir] [--mv-bands 5 4 6]
        [--classes 1] [--dilate 2] [--device cuda:0]
        [--share-local V:\\ --share-server /maintenance/]
        [--top 5 --min-agree 3 --cell-km 25]

For each job, the cloud mask of its input image is made with the job's own
cloud settings (method per sensor, radiance scale and unit, ...; --method
overrides, for every image), as a job with cloud masking would have made it.
Where the job chose no method, it goes by the sensor: MV OmniCloudMask on
bands 5, 4, 6 (--mv-bands), HV the VNIR tests, HS the SWIR tests, ML the
infrared tests. Every match
whose input-image position lies on a masked class (--classes, default thick
cloud; grown by --dilate px) is then left out wherever the job's matches
are read, and the job is scored again, without touching its own files:

    <job>/CLOUDFREE/CLOUD_MASK.tif, CLOUD_QUICKLOOK.png, CLOUD_PARAMS_*.txt
    <job>/CLOUDFREE/TRUTH_RANKING.csv, TRUTH_BY_DETECTOR_MATCHER.csv,
                    TRUTH_MATCH_ACCURACY.csv   (with ground truth)
    <job>/CLOUDFREE/RIVAL_CROSS_<band>.csv, CROSS_SUMMARY.csv   (the cross-check)
    <job>/CLOUDFREE/CLOUDFREE_SUMMARY.csv      cloud %, share of matches
                    rejected, best configuration and RMSE before and after
    <folder>/CLOUDFREE_ALL.csv                 one row per job (a folder of jobs)

Without ground truth only the cross-check is redone. The mask is kept in
CLOUDFREE/ and reused while the input and the cloud settings are the same.

    python automatch_job.py filter <job folder | folder of jobs> [--no-cloud] [--no-interior]
        [--edge-px 4] (and the options above)

does the same with two masks: cloud, and the interior of featureless areas
(automatch_interior: the middle of water bodies and uniform blobs, their
edges kept), into <job>/FILTERED/ (FILTERED_SUMMARY.csv, FILTERED_ALL.csv),
and ends with the report (REPORT/: shapefiles, look vectors, statistics).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# What a sensor gets when neither the job nor the command line chose a cloud
# method for it: MV: OmniCloudMask on bands 5, 4, 6 (red, green, NIR). The
# other sensors keep the physical tests of their kind (HV VNIR, HS SWIR, ML infrared).
SENSOR_DEFAULTS: Dict[str, Dict] = {'MV': {'cloud_method_MV': 'omnicloudmask',
                                           'cloud_bands': ['band5', 'band4', 'band6']}}


def sensor_settings(cfg: Dict, sensor: str, overrides: Optional[Dict] = None) -> Dict:
    """cfg with SENSOR_DEFAULTS of this sensor, unless a method was chosen
    (the job's cloud_method / cloud_method_<sensor>, or an override)."""
    chosen = any(str(v or 'auto').lower() != 'auto' for v in (
        cfg.get('cloud_method'), cfg.get(f'cloud_method_{sensor}'), (overrides or {}).get('cloud_method')))
    if chosen or sensor not in SENSOR_DEFAULTS:
        return cfg
    d = dict(SENSOR_DEFAULTS[sensor])
    if cfg.get('cloud_bands'):
        d.pop('cloud_bands', None)
    return dict(cfg, **d)


def job_input(job_dir: str, job: Dict) -> str:
    """The job's input image as this machine sees it ('' when unreachable)."""
    import automatch_cross as X
    p = job.get('input_path') or ''
    for loc in (p, X.local_path(p, job_dir, job.get('output_dir') or ''), X._by_share(p)):
        if loc and os.path.exists(loc):
            return loc
    return ''


def cloud_mask_for(job_dir: str, job: Dict, overrides: Optional[Dict] = None, out: str = '') -> Dict:
    """CLOUD_INFO of the job's input (automatch_cloud.prepare) in
    <job>/CLOUDFREE/ (or out); raises when the input cannot be read here."""
    import automatch_cloud as C
    import automatch_job as J
    inp = job_input(job_dir, job)
    if not inp:
        raise FileNotFoundError(f"input {job.get('input_path')!r} not reachable from here (set --share-local / "
                                f"--share-server)")
    if os.path.isdir(inp) or inp.lower().endswith(('.h5', '.hdf5')):
        raise ValueError('cloud masking is for raster inputs (NISAR is radar)')
    cfg = dict(J.normalize({k: v for k, v in job.items() if k in J.DEFAULT_JOB}), cloud_mask='on',
               **(overrides or {}))
    out = out or os.path.join(job_dir, 'CLOUDFREE')

    class _Scene:
        raster_path = inp
    sensor = C.sensor_for(inp)
    cfg = sensor_settings(cfg, sensor, overrides)
    if C.needs_model(cfg, sensor, inp):
        probs = C.check(cfg)
        if probs:
            raise RuntimeError('; '.join(probs))
    _, info = C.prepare(cfg, _Scene(), [], os.path.join(out, 'cloud'), out, sensor)
    info['sensor'] = sensor
    return info


def point_sampler(mask_path: str, classes: List[int], dilate_px: int, working_crs: str):
    """f(E, N) -> keep (bool array): False where a point (working CRS) lies on
    a masked class of the mask, grown by dilate_px; points outside the mask
    or on its no data are kept. f.stats counts what it saw and rejected."""
    import cv2
    import rasterio as rio
    from pyproj import CRS, Transformer
    with rio.open(mask_path) as m:
        cls, tr, mcrs = m.read(1), m.transform, m.crs
    bad = np.isin(cls, [int(c) for c in classes])
    if dilate_px > 0 and bad.any():
        k = np.ones((2 * int(dilate_px) + 1,) * 2, np.uint8)
        bad = cv2.dilate(bad.astype(np.uint8), k).astype(bool)
    to_mask = None
    try:
        if working_crs and mcrs is not None and not CRS.from_user_input(working_crs).equals(mcrs):
            to_mask = Transformer.from_crs(CRS.from_user_input(working_crs), mcrs, always_xy=True)
    except Exception:
        to_mask = None
    inv = ~tr
    H, W = bad.shape

    def keep(E: np.ndarray, N: np.ndarray) -> np.ndarray:
        x, y = (to_mask.transform(E, N) if to_mask is not None else (E, N))
        col, row = inv * (np.asarray(x, float), np.asarray(y, float))
        r, c = np.floor(row).astype(int), np.floor(col).astype(int)
        inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        hit = np.zeros(len(r), bool)
        hit[inside] = bad[r[inside], c[inside]]
        keep.stats['seen'] += len(r)
        keep.stats['rejected'] += int(hit.sum())
        return ~hit
    keep.stats = {'seen': 0, 'rejected': 0}
    return keep


def _best(ranking_csv: str) -> Dict:
    try:
        r = pd.read_csv(ranking_csv)
        r = r[r['truth_rmse_m'].notna()].sort_values('rank')
        b = r.iloc[0]
        return {'config': f"{b['channel']} {b['detector']} + {b['matcher']} {b['ransac']} {b['sweep']}",
                'rmse_m': round(float(b['truth_rmse_m']), 1),
                'reached': f"{int(b['truth_reached'])}/{int(b['truth_total'])}"}
    except Exception:
        return {}


def interior_mask_for(job_dir: str, job: Dict, out: str, opts: Optional[Dict] = None) -> Dict:
    """The interior mask (automatch_interior) of the band the job matched,
    in out; raises when the input cannot be read here."""
    import automatch_bench as B
    import automatch_interior as IM
    inp = job_input(job_dir, job)
    if not inp:
        raise FileNotFoundError(f"input {job.get('input_path')!r} not reachable from here (set --share-local / "
                                f"--share-server)")
    if os.path.isdir(inp) or inp.lower().endswith(('.h5', '.hdf5')):
        raise ValueError('the interior filter is for raster inputs')
    band = B._band_run(job_dir, job)
    o = dict(IM.options(job), **(opts or {}))
    info = IM.interior_mask(inp, IM.band_number(band), os.path.join(out, f'INTERIOR_MASK_{band}.tif'),
                            o['edge_px'], o['win_px'], o['k_noise'], o['max_side'])
    return dict(info, band=band)


def filter_job(job_dir: str, overrides: Optional[Dict] = None, classes=None, dilate_px: Optional[int] = None,
               cross: Optional[Dict] = None, mask_path: str = '', kinds=('cloud',), out_name: str = '',
               interior_opts: Optional[Dict] = None, interior_mask_path: str = '') -> Dict:
    """Filter one finished job (module docstring): kinds 'cloud' and / or
    'interior'; into <job>/CLOUDFREE (cloud only) or <job>/FILTERED; its
    <folder>_SUMMARY row."""
    import automatch_cross as X
    import automatch_rival as R
    import automatch_truth as T
    kinds = tuple(kinds)
    name = os.path.basename(os.path.normpath(job_dir))
    with open(os.path.join(job_dir, 'job_used.json'), encoding='utf-8') as f:
        job = json.load(f)
    out_name = out_name or ('CLOUDFREE' if kinds == ('cloud',) else 'FILTERED')
    out = os.path.join(job_dir, out_name)
    os.makedirs(out, exist_ok=True)
    classes = [int(c) for c in (classes or job.get('cloud_mask_classes') or [1])]
    dilate_px = int(job.get('cloud_dilate_px') or 2) if dilate_px is None else int(dilate_px)
    row: Dict = {'job': name, 'filters': '+'.join(kinds)}
    crs = T.working_crs_of(job_dir)
    samplers = []
    if 'cloud' in kinds:
        if mask_path:
            info = {'mask': mask_path, 'cloud_pct': None, 'model': 'given'}
        else:
            info = cloud_mask_for(job_dir, job, overrides, out)
        row.update({'classes': ' '.join(str(c) for c in classes), 'dilate_px': dilate_px,
                    'sensor': info.get('sensor'), 'cloud_pct': info.get('cloud_pct'),
                    'thick_pct': info.get('thick_pct'), 'cloud_method': info.get('model'), 'mask': info.get('mask')})
        samplers.append(('cloud', point_sampler(info['mask'], classes, dilate_px, crs)))
    if 'interior' in kinds:
        if interior_mask_path:
            ii = {'mask': interior_mask_path, 'interior_pct': None}
        else:
            ii = interior_mask_for(job_dir, job, out, interior_opts)
        row.update({'interior_pct': ii.get('interior_pct'), 'interior_mask': ii.get('mask'),
                    'interior_edge_px': ii.get('edge_px')})
        samplers.append(('interior', point_sampler(ii['mask'], [1], 0, crs)))

    def keep(E, N):
        ok = np.ones(len(np.asarray(E)), bool)
        for _, s in samplers:
            ok &= s(E, N)
        keep.stats['seen'] += len(ok)
        keep.stats['rejected'] += int((~ok).sum())
        return ok
    keep.stats = {'seen': 0, 'rejected': 0}
    truth_csv = X.job_truth_csv(job_dir)
    before = _best(os.path.join(job_dir, 'TRUTH_RANKING.csv'))
    R.POINT_MASK = keep
    try:
        if truth_csv:
            res = T.compare(job_dir, truth_csv, float(job.get('truth_radius_m') or 5000.0), 'consensus', crs,
                            write=False, quiet=True, rebuild=False)
            for key, fname in (('ranking', 'TRUTH_RANKING.csv'), ('by_detector_matcher', 'TRUTH_BY_DETECTOR_MATCHER.csv'),
                               ('accuracy', 'TRUTH_MATCH_ACCURACY.csv'), ('points', 'TRUTH_POINTS.csv')):
                df = res.get(key)
                if isinstance(df, pd.DataFrame):
                    df.to_csv(os.path.join(out, fname), index=False)
        c = dict(cross or {})
        if int(c.get('top_n', 5)) >= 2:
            xr = X.run(job_dir, int(c.get('top_n', 5)), int(c.get('min_agree', 0)), float(c.get('cell_km', 25.0)),
                       truth_csv=truth_csv, working_crs=crs, quiet=True, out_dir=out,
                       max_points=X._job_max_points(job_dir),
                       ranking_csv=os.path.join(out, 'TRUTH_RANKING.csv'))
            for ch, r in xr.items():
                row[f'cross_{ch}_points'] = r.get('points')
                if r.get('truth_rmse_m') is not None:
                    row[f'cross_{ch}_rmse_m'] = round(float(r['truth_rmse_m']), 1)
    finally:
        R.POINT_MASK = None
    try:
        import automatch_report as RP
        RP.report(out, quiet=True, truth_csv=truth_csv)
    except Exception as e:
        print(f'[Filter] {name}: REPORT not written ({type(e).__name__}: {e})')
    after = _best(os.path.join(out, 'TRUTH_RANKING.csv')) if truth_csv else {}
    seen = keep.stats['seen']
    for kind, s in samplers:
        row[f'{kind}_rejected_pct'] = round(100.0 * s.stats['rejected'] / s.stats['seen'], 2) if s.stats['seen'] else None
    row.update({'matches_checked': seen, 'rejected_pct': round(100.0 * keep.stats['rejected'] / seen, 2) if seen else None,
                'best_before': before.get('config'), 'rmse_before_m': before.get('rmse_m'),
                'reached_before': before.get('reached'), 'best_after': after.get('config'),
                'rmse_after_m': after.get('rmse_m'), 'reached_after': after.get('reached'),
                'truth': bool(truth_csv)})
    pd.DataFrame([row]).to_csv(os.path.join(out, f'{out_name}_SUMMARY.csv'), index=False)
    print(f"[Filter] {name}: " + '; '.join(
              ([f"{row.get('cloud_pct')} % cloud ({row.get('cloud_method')})"] if 'cloud' in kinds else [])
              + ([f"{row.get('interior_pct')} % featureless interior"] if 'interior' in kinds else []))
          + f"; {row['rejected_pct']} % of the matches rejected"
          + (f"; best {before.get('rmse_m')} m ({before.get('reached')}) -> {after.get('rmse_m')} m "
             f"({after.get('reached')}): {after.get('config')}" if truth_csv else '; no ground truth: cross-check only')
          + f' -> {out}', flush=True)
    return row


def batch(folder: str, jobs: str = '', **kw) -> pd.DataFrame:
    """filter_job for one job folder, or every finished job under folder
    (CLOUDFREE_ALL.csv / FILTERED_ALL.csv there)."""
    kinds = tuple(kw.get('kinds') or ('cloud',))
    all_name = ('CLOUDFREE' if kinds == ('cloud',) else 'FILTERED') + '_ALL.csv'
    import automatch_cross as X
    if os.path.exists(os.path.join(folder, 'job_used.json')):
        try:
            return pd.DataFrame([filter_job(folder, **kw)])
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f'[CloudFilter] {os.path.basename(os.path.normpath(folder))}: not done ({type(e).__name__}: {e})',
                  flush=True)
            return pd.DataFrame([{'job': os.path.basename(os.path.normpath(folder)), 'error': str(e)}])
    found, skipped = X.find_jobs(folder, X.parse_jobs(jobs))
    rows = []
    for d in found:
        try:
            rows.append(filter_job(d, **kw))
        except Exception as e:
            print(f'[CloudFilter] {os.path.basename(d)}: skipped ({type(e).__name__}: {e})', flush=True)
            rows.append({'job': os.path.basename(d), 'error': f'{type(e).__name__}: {e}'})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(folder, all_name), index=False)
    print(f'[Filter] {len(rows)} job(s); {all_name} in {folder}', flush=True)
    return df


def main(argv: Optional[List[str]] = None, kinds=('cloud',)) -> int:
    ap = argparse.ArgumentParser(description='Reject the match points of finished jobs that fall on cloud'
                                 + (' or inside featureless areas' if 'interior' in kinds else ''))
    ap.add_argument('folder', help='a job output folder, or a folder of them')
    ap.add_argument('--jobs', default='', help="only these job numbers, e.g. '335-366, 370'")
    ap.add_argument('--method', default='', help='cloud method for every image (default: the job\'s; auto: by sensor)')
    ap.add_argument('--mv-bands', type=int, nargs=3, default=None, metavar=('RED', 'GREEN', 'NIR'),
                    help='MV with OmniCloudMask: its three bands (default 5 4 6)')
    ap.add_argument('--classes', type=int, nargs='+', default=None, help='masked classes (default the job\'s, else 1)')
    ap.add_argument('--dilate', type=int, default=None, help='grow the mask by px (default the job\'s, else 2)')
    ap.add_argument('--device', default='', help='cpu / cuda / cuda:N for a cloud model')
    ap.add_argument('--share-local', default='', help="where the server's share is mounted here, e.g. V:\\")
    ap.add_argument('--share-server', default='', help='the same share on the server, e.g. /maintenance/')
    ap.add_argument('--top', type=int, default=5, help='cross-check: configurations compared (0: off)')
    ap.add_argument('--min-agree', type=int, default=3, help='cross-check: how many must agree')
    ap.add_argument('--cell-km', type=float, default=25.0)
    if 'interior' in kinds:
        ap.add_argument('--no-cloud', action='store_true', help='only the interior filter')
        ap.add_argument('--no-interior', action='store_true', help='only the cloud filter')
        ap.add_argument('--edge-px', type=float, default=None,
                        help='interior: farther than this from any textured pixel (input pixels, default 4)')
    a = ap.parse_args(argv)
    if 'interior' in kinds:
        kinds = tuple(k for k in ('cloud', 'interior') if not getattr(a, f'no_{k}'))
        if not kinds:
            ap.error('--no-cloud and --no-interior leave nothing to filter')
    import automatch_cross as X
    import automatch_job as J
    X.SHARE = (a.share_local, a.share_server)
    over = {}
    if a.mv_bands:
        SENSOR_DEFAULTS['MV']['cloud_bands'] = [f'band{b}' for b in a.mv_bands]
    if a.method:
        over['cloud_method'] = a.method
    if a.device:
        over['cloud_device'] = J.use_gpu(a.device)
    io = {'edge_px': a.edge_px} if getattr(a, 'edge_px', None) is not None else None
    batch(a.folder, a.jobs, overrides=over, classes=a.classes, dilate_px=a.dilate,
          cross={'top_n': a.top, 'min_agree': a.min_agree, 'cell_km': a.cell_km}, kinds=kinds,
          interior_opts=io)
    return 0


if __name__ == '__main__':
    sys.exit(main())
