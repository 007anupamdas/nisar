"""
Benchmark: how each detector does on which kind of image, from finished jobs.

For every finished automatch job under a folder (output/output359 ...) it
records, without matching anything again:

  scene     sensor (MV / HS / HV from the G1A name), band run, pixel size
  error     the image's geolocation error from the job's own ground truth
            (mean In - Ref, and how much it changes across the scene)
  clouds    an estimate of the cloud fraction of the band that ran (below),
            with a quicklook SCENE_CLOUD.png to check it by eye
  texture   how much ground detail the band shows
  detectors every detector's best result against the ground truth
            (TRUTH_RANKING.csv), in metres and in pixels

Of several jobs on one scene only the newest counts: the newest automatch
version in its log, then the latest finish, then the highest job number; the
others are listed as superseded (--all-jobs counts them all).

and writes, in that folder:

  BENCH_JOBS.csv       one row per job: the scene, its error, clouds, best detector
  BENCH_DETECTORS.csv  one row per job x detector
  BENCH_SUMMARY.csv    per detector: jobs, success rate (reached half of the
                       truth points within --success-px pixels), median error;
                       overall, per sensor, per cloud class and per error class

Cloud estimate (no cloud mask comes with G1A): on an overview of the band,
the valid pixels' brightness is split in two (Otsu); when the two groups are
well apart (4 standard deviations; one normal hump splits at about 2.7), the
bright pixels at least twice as smooth as the scene's land are clouds, cleaned with a small opening / closing. Bright
smooth ground (sand, snow, salt flats) can pass for cloud: check
SCENE_CLOUD.png. The result is kept in SCENE_INFO.json and read back on later
runs (--recompute redoes it).

    python automatch_job.py bench <folder of job outputs> [--jobs 335-366] [--success-px 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLOUD_CLASSES = ((10.0, 'clear (<10 %)'), (30.0, 'partly (10-30 %)'), (101.0, 'cloudy (>30 %)'))
ERROR_CLASSES = ((20.0, 'small (<20 km)'), (80.0, 'medium (20-80 km)'), (1e9, 'large (>80 km)'))


# ─────────────────────────────────────────────────────────────────────────────
# scene measures
# ─────────────────────────────────────────────────────────────────────────────
def _otsu(v: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2.0
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.argmax(between))])


def cloud_mask(a: np.ndarray, valid: np.ndarray, min_separation: float = 4.0) -> Tuple[np.ndarray, Dict]:
    """(cloud mask, details) of one band overview; see the module notes."""
    import cv2
    out = np.zeros_like(valid, dtype=bool)
    v = a[valid]
    info = {'bright_threshold': None, 'separation': 0.0}
    if v.size < 500:
        return out, info
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return out, info
    vc = np.clip(v, lo, hi)
    t = _otsu(vc)
    dark, bright = vc[vc <= t], vc[vc > t]
    if dark.size < 50 or bright.size < 50:
        return out, info
    sep = float((bright.mean() - dark.mean()) / math.sqrt((dark.var() + bright.var()) / 2.0 + 1e-12))
    info.update({'bright_threshold': float(t), 'separation': round(sep, 2)})
    if sep < min_separation:
        return out, info                     # not two-humped (one normal hump splits at ~2.7): no clouds
    af = np.where(valid, a, 0.0).astype(np.float32)
    mean = cv2.blur(af, (7, 7))
    sq = cv2.blur(af * af, (7, 7))
    sd = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    cv = sd / np.maximum(np.abs(mean), 1e-6)
    land = valid & (a <= t)
    land_cv = float(np.median(cv[land])) if land.any() else 0.0
    cand = valid & (a > t) & (cv < max(0.5 * land_cv, 1e-6))      # clearly smoother than land
    k3, k5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5).astype(bool) & valid
    info['land_cv'] = round(land_cv, 4)
    return m, info


def scene_info(path: str, band: int, max_side: int = 1024, png: str = '') -> Dict:
    """Valid share, cloud fraction and ground texture of one band (overview read)."""
    import cv2
    import rasterio as rio
    from rasterio.enums import Resampling
    with rio.open(path) as src:
        scale = max(1.0, max(src.height, src.width) / float(max_side))
        oh, ow = max(8, int(src.height / scale)), max(8, int(src.width / scale))
        a = src.read(band, out_shape=(oh, ow), resampling=Resampling.average).astype(np.float64)
        nd = src.nodata
        pixel = abs(src.transform.a) if src.crs is not None and src.crs.is_projected else None
    valid = np.isfinite(a) & (a != (nd if nd is not None else 0))
    mask, info = cloud_mask(a, valid)
    nv = int(valid.sum())
    lo, hi = (np.percentile(a[valid], [2, 98]) if nv else (0.0, 1.0))
    gy, gx = np.gradient(np.where(valid, a, 0.0))
    land = valid & ~mask
    tex = float(np.median(np.hypot(gx, gy)[land]) / max(hi - lo, 1e-9)) if land.any() else 0.0
    out = {'valid_pct': round(100.0 * nv / valid.size, 2),
           'cloud_pct': round(100.0 * mask.sum() / max(nv, 1), 2),
           'texture': round(tex, 4), 'pixel_m': pixel, **info}
    if png:
        g = np.clip((a - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        rgb[mask] = (0.5 * rgb[mask] + np.array([0, 0, 127])).astype(np.uint8)     # clouds in red
        rgb[~valid] = (60, 60, 60)
        cv2.imwrite(png, rgb)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# one job
# ─────────────────────────────────────────────────────────────────────────────
def _job_used(d: str) -> Dict:
    p = os.path.join(d, 'job_used.json')
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _band_run(d: str, job: Dict) -> str:
    man = os.path.join(d, 'RUN_MANIFEST.csv')
    if os.path.exists(man):
        try:
            ch = pd.read_csv(man)['channel'].dropna().astype(str).unique()
            if len(ch):
                return str(ch[0])
        except Exception:
            pass
    ch = job.get('channels') or []
    return str(ch[0]) if ch else 'band1'


def truth_error(truth_csv: str) -> Dict:
    """The image's error from its ground truth: mean In - Ref (km) and how far
    the points' errors spread (km), without needing the working CRS."""
    import automatch_truth as T
    t = T.load_truth(truth_csv)
    lat0 = np.radians((t['in_lat'] + t['ref_lat']) / 2.0)
    de = (t['in_lon'] - t['ref_lon']) * 111.32 * np.cos(lat0)
    dn = (t['in_lat'] - t['ref_lat']) * 110.57
    spread = float(np.hypot(de - de.mean(), dn - dn.mean()).max()) if len(t) > 1 else 0.0
    return {'error_km': round(float(math.hypot(de.mean(), dn.mean())), 2), 'error_dE_km': round(float(de.mean()), 2),
            'error_dN_km': round(float(dn.mean()), 2), 'error_spread_km': round(spread, 2), 'truth_points': len(t)}


def detectors_of(d: str) -> pd.DataFrame:
    """Each detector's best configuration in TRUTH_RANKING.csv (its order)."""
    p = os.path.join(d, 'TRUTH_RANKING.csv')
    if not os.path.exists(p):
        return pd.DataFrame()
    r = pd.read_csv(p)
    if r.empty or 'detector' not in r.columns:
        return pd.DataFrame()
    r = r.sort_values('rank') if 'rank' in r.columns else r
    best = r.drop_duplicates(['channel', 'detector']).copy()
    best['detector_rank'] = range(1, len(best) + 1)
    return best


def bench_job(d: str, recompute: bool = False) -> Tuple[Dict, pd.DataFrame]:
    """(job row, detector rows) for one finished job."""
    import automatch_cross as X
    job = _job_used(d)
    name = os.path.basename(os.path.normpath(d))
    inp = job.get('input_path') or ''
    loc_in = X.local_path(inp, d, job.get('output_dir') or '')
    if inp and not os.path.exists(loc_in):
        loc_in = X._by_share(inp)
    scene = os.path.splitext(os.path.basename(str(inp).replace('\\', '/')))[0]
    m = re.search(r'G1A_([A-Z]{2})_', scene)
    band = _band_run(d, job)
    row = {'job': name, 'scene': scene, 'sensor': m.group(1) if m else 'other', 'band': band,
           'max_expected_error_km': (job.get('max_expected_error_m') or 0) / 1000.0}
    # scene measures: read back, or measured on the input as this machine sees it
    info_p = os.path.join(d, 'SCENE_INFO.json')
    info = None
    if not recompute and os.path.exists(info_p):
        try:
            with open(info_p, encoding='utf-8') as f:
                info = json.load(f)
            if info.get('band') != band:
                info = None
        except (OSError, ValueError):
            info = None
    if info is None and loc_in and os.path.exists(loc_in):
        try:
            bnum = int(re.sub(r'\D', '', band) or 1)
            info = {'band': band, **scene_info(loc_in, bnum, png=os.path.join(d, 'SCENE_CLOUD.png'))}
            try:
                with open(info_p, 'w', encoding='utf-8') as f:
                    json.dump(info, f, indent=1)
            except OSError:
                pass
        except Exception as e:
            print(f'[Bench] {name}: scene measures failed ({type(e).__name__}: {e})')
    elif info is None:
        print(f'[Bench] {name}: input {inp} not reachable from here (set the share mapping); no cloud estimate')
    if info:
        row.update({k: info.get(k) for k in ('cloud_pct', 'valid_pct', 'texture', 'pixel_m', 'separation')})
    tcsv = X.job_truth_csv(d)
    if tcsv:
        try:
            row.update(truth_error(tcsv))
        except Exception as e:
            print(f'[Bench] {name}: ground truth unreadable ({type(e).__name__}: {e})')
    cs = os.path.join(d, 'CROSS_SUMMARY.csv')
    if os.path.exists(cs):
        try:
            c = pd.read_csv(cs)
            if 'reliable' in c.columns:
                row['cross_reliable'] = bool(c['reliable'].astype(str).str.lower().eq('true').all())
            if 'kept_share' in c.columns:
                row['cross_kept_share'] = float(c['kept_share'].min())
        except Exception:
            pass
    det = detectors_of(d)
    if not det.empty:
        px = row.get('pixel_m')
        det = det.assign(job=name, rmse_px=(det['truth_rmse_m'] / px) if px else np.nan)
        top = det.iloc[0]
        row.update({'best_detector': f"{top['detector']} + {top['matcher']}", 'best_rmse_m': top.get('truth_rmse_m'),
                    'best_rmse_px': top.get('rmse_px'), 'detectors_ranked': len(det)})
    return row, det


# ─────────────────────────────────────────────────────────────────────────────
# across jobs
# ─────────────────────────────────────────────────────────────────────────────
def _class(v, classes) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'unknown'
    for lim, name in classes:
        if v < lim:
            return name
    return classes[-1][1]


def summarize(dets: pd.DataFrame, success_px: float = 3.0) -> pd.DataFrame:
    """Per detector, overall and per sensor / cloud class / error class: jobs,
    success rate (reached at least half of the truth points within
    success_px pixels), median RMSE (m and px), times ranked first."""
    if dets.empty:
        return pd.DataFrame()
    d = dets.copy()
    d['success'] = (d['truth_reached'] >= d['truth_total'] / 2.0) & (d['rmse_px'] <= success_px)
    d['first'] = d['detector_rank'] == 1
    rows = []
    groups = [('all', None), ('sensor', 'sensor'), ('cloud', 'cloud_class'), ('error', 'error_class')]
    for gname, col in groups:
        parts = [('all', d)] if col is None else list(d.groupby(col))
        for gval, g in parts:
            for det, h in g.groupby('detector'):
                ok = h[h['truth_reached'] >= h['truth_total'] / 2.0]
                rows.append({'group_by': gname, 'group': gval, 'detector': det, 'jobs': int(h['job'].nunique()),
                             'success_rate': round(float(h['success'].mean()), 3),
                             'median_rmse_px': round(float(ok['rmse_px'].median()), 2) if len(ok) else None,
                             'median_rmse_m': round(float(ok['truth_rmse_m'].median()), 1) if len(ok) else None,
                             'times_first': int(h['first'].sum())})
    out = pd.DataFrame(rows)
    return out.sort_values(['group_by', 'group', 'success_rate', 'median_rmse_px'],
                           ascending=[True, True, False, True])


def job_meta(d: str) -> Dict:
    """What decides which of several jobs on one scene is the newest: the
    automatch version in its log's first lines, when it finished
    (RUN_MANIFEST.csv), its number."""
    job = _job_used(d)
    name = os.path.basename(os.path.normpath(d))
    scene = os.path.splitext(os.path.basename(str(job.get('input_path') or '').replace('\\', '/')))[0]
    version, code = '', ''
    lp = os.path.join(d, 'automatch.log')
    if os.path.exists(lp):
        try:
            with open(lp, encoding='utf-8', errors='replace') as f:
                head = f.read(20000)
            m = re.search(r'automatch (\d{4}\.\d{2}\.\d{2}), code (\w+)', head)
            if m:
                version, code = m.group(1), m.group(2)
        except OSError:
            pass
    done = 0.0
    for n in ('RUN_MANIFEST.csv', 'TRUTH_RANKING.csv', 'job_used.json'):
        if os.path.exists(os.path.join(d, n)):
            done = os.path.getmtime(os.path.join(d, n))
            break
    num = re.search(r'(\d+)\D*$', name)
    return {'job': name, 'scene': scene or name, 'version': version, 'code': code, 'finished': done,
            'number': int(num.group(1)) if num else 0}


def newest_per_scene(metas: List[Dict]) -> Dict[str, Optional[str]]:
    """{job: None when it is the one used for its scene, else the job that
    supersedes it}: per scene the newest automatch version, then the latest
    finish, then the highest job number."""
    out: Dict[str, Optional[str]] = {}
    by: Dict[str, List[Dict]] = {}
    for m in metas:
        by.setdefault(m['scene'], []).append(m)
    for ms in by.values():
        best = max(ms, key=lambda m: (m['version'], m['finished'], m['number']))
        for m in ms:
            out[m['job']] = None if m is best else best['job']
    return out


def batch(parent: str, jobs: str = '', success_px: float = 3.0, recompute: bool = False,
          all_jobs: bool = False) -> int:
    """Benchmark the finished jobs under parent; of several jobs on one scene
    only the newest counts (newest_per_scene) unless all_jobs."""
    import automatch_cross as X
    found, skipped = X.find_jobs(parent, X.parse_jobs(jobs))
    for n, why in skipped.items():
        print(f'[Bench] {n}: skipped, {why}')
    print(f'[Bench] {len(found)} finished job(s) under {parent}')
    metas = {d: job_meta(d) for d in found}
    sup = {} if all_jobs else newest_per_scene(list(metas.values()))
    jrows, drows = [], []
    for d in found:
        m = metas[d]
        if sup.get(m['job']):
            print(f"[Bench] {m['job']}: {m['scene']} superseded by {sup[m['job']]} (newer job on the same scene)")
            jrows.append({'job': m['job'], 'scene': m['scene'], 'automatch': m['version'], 'code': m['code'],
                          'used': False, 'superseded_by': sup[m['job']]})
            continue
        row, det = bench_job(d, recompute)
        row.update({'automatch': m['version'], 'code': m['code'], 'used': True, 'superseded_by': ''})
        jrows.append(row)
        print(f"[Bench] {row['job']}: {row['sensor']} {row['band']}, error {row.get('error_km', '?')} km, "
              f"clouds {row.get('cloud_pct', '?')} %, best {row.get('best_detector', '-')} "
              f"{row.get('best_rmse_m') if row.get('best_rmse_m') is not None else '-'} m")
        if not det.empty:
            drows.append(det.assign(sensor=row['sensor'], band=row['band'], cloud_pct=row.get('cloud_pct'),
                                    error_km=row.get('error_km'), pixel_m=row.get('pixel_m')))
    if not jrows:
        print(f'[Bench] no finished automatch job under {parent}')
        return 1
    jobs_df = pd.DataFrame(jrows)
    jobs_df.to_csv(os.path.join(parent, 'BENCH_JOBS.csv'), index=False)
    if drows:
        dets = pd.concat(drows, ignore_index=True)
        dets['cloud_class'] = [_class(v, CLOUD_CLASSES) for v in dets['cloud_pct']]
        dets['error_class'] = [_class(v, ERROR_CLASSES) for v in dets['error_km']]
        keep = ['job', 'sensor', 'band', 'pixel_m', 'cloud_pct', 'cloud_class', 'error_km', 'error_class',
                'detector', 'matcher', 'sweep', 'ransac', 'truth_rmse_m', 'rmse_px', 'truth_reached', 'truth_total',
                'detector_rank']
        dets = dets[[c for c in keep if c in dets.columns]]
        dets.to_csv(os.path.join(parent, 'BENCH_DETECTORS.csv'), index=False)
        s = summarize(dets, success_px)
        s.to_csv(os.path.join(parent, 'BENCH_SUMMARY.csv'), index=False)
        top = s[s['group_by'] == 'all'].head(10)
        print(f'[Bench] best detectors over {jobs_df["job"].nunique()} job(s) (success: half of the truth '
              f'points within {success_px:g} px):')
        for r in top.itertuples():
            print(f'         {r.detector:<28} success {100 * r.success_rate:5.1f} % of {r.jobs} job(s), '
                  f'median {r.median_rmse_px} px, first {r.times_first}x')
    print(f'[Bench] written: BENCH_JOBS.csv' + (', BENCH_DETECTORS.csv, BENCH_SUMMARY.csv' if drows else '')
          + f' in {parent}')
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    import automatch_cross as X
    ap = argparse.ArgumentParser(description='How each detector does on which kind of image, from finished jobs')
    ap.add_argument('folder', help='a job output folder, or a folder of them (output/output359 ...)')
    ap.add_argument('--jobs', default='', help="only these job numbers, e.g. '335-366, 370'")
    ap.add_argument('--success-px', type=float, default=3.0, help='success: half the truth points within this (px)')
    ap.add_argument('--recompute', action='store_true', help='measure the scenes again (SCENE_INFO.json)')
    ap.add_argument('--all-jobs', action='store_true',
                    help='count every job; by default only the newest job of each scene counts')
    ap.add_argument('--share-local', default='', help="where the server's share is mounted here, e.g. V:\\")
    ap.add_argument('--share-server', default='', help='the same share on the server, e.g. /maintenance/')
    a = ap.parse_args(argv)
    X.SHARE = (a.share_local, a.share_server)
    return batch(a.folder, a.jobs, a.success_px, a.recompute, a.all_jobs)


if __name__ == '__main__':
    sys.exit(main())
