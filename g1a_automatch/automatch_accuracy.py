"""
Match accuracy of finished jobs, again, without matching: how many of each
window's matches are right (automatch_truth.match_accuracy), per window size,
keypoint count and RANSAC setting -- from the matches every job saved
(raw_matches_* and filtered_*) and its own ground truth.

Sub-cells afterwards (subcell_px): the same selection a run makes with
subcell_px (automatch_engine.subcell_select), made from the saved matches:
for each window and RANSAC setting, its RANSAC inliers give the window's
error plane, each sub-cell corrects it and keeps its least distorted
matches. Written into <job>/SUBCELLS_<px>/ (filtered_* and their chip
consensus, final_*), never into the job's own folders, and scored beside the
plain results as RANSAC variant <method>-sc<px>.

Per job: TRUTH_MATCH_ACCURACY.csv -- one file for everything: window size,
keypoint count and sub-cell size (subcell_px, 0 = plain) are columns -- and
with sub-cells TRUTH_RANKING_SUBCELLS.csv, the truth ranking with the
sub-cell variants; the job's TRUTH_RANKING.csv is left as it is. Sizes made
before (SUBCELLS_<px>/) stay in the file when another size is asked for.
Results made with the same settings are read back.

Over many jobs this runs inside the benchmark (automatch_job.py bench
<folder> --accuracy [--subcell-px 256 512]); it writes ACCURACY_ALL.csv and
ACCURACY_SUMMARY.csv there.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VERSION = 1
RAW_COLS = ['x1', 'y1', 'x2', 'y2', 'x1_map', 'y1_map', 'x2_map', 'y2_map', 'distance', 'along', 'across']
OUT_COLS = ['x1', 'y1', 'x2', 'y2', 'x1-map', 'y1-map', 'x2-map', 'y2-map', 'Distance', 'along', 'across',
            'model-type', 'threshold', 'confidence', 'inliers']


def _raw_dir(filtered_dir: str) -> str:
    return os.path.join(os.path.dirname(filtered_dir),
                        'raw_matches_' + os.path.basename(filtered_dir)[len('filtered_'):])


def subcell_runs(job_dir: str, px, keep: int, res: float, settings: Optional[Dict],
                 recompute: bool = False, label: str = '') -> List[Dict]:
    """The sub-cell variants (px: one size or a list) of every run of a
    finished job, made from its saved matches into <job>/SUBCELLS_<px>/;
    [{'filtered_dir', 'final_dir', 'sweep', 'raw_dir'}]. A size already made
    with the same keep is reused; the others are made in one pass over the
    windows (each raw match file read once for every size and RANSAC
    setting), with a line per run."""
    import automatch_engine as E
    import automatch_truth as T
    sizes = sorted({int(v) for v in (px if isinstance(px, (list, tuple)) else [px])})
    todo = []
    for s in sizes:
        marker = os.path.join(job_dir, f'SUBCELLS_{s}', 'SUBCELLS.json')
        want = {'px': s, 'keep': int(keep), 'res': float(res), 'v': VERSION}
        try:
            with open(marker, encoding='utf-8') as f:
                done = json.load(f) == want and not recompute
        except (OSError, ValueError):
            done = False
        if not done:
            todo.append(s)
    runs = [r for r in T.find_runs(job_dir) if os.path.isdir(_raw_dir(r['filtered_dir']))]
    out = []
    for k, run in enumerate(runs, 1):
        raw_dir = _raw_dir(run['filtered_dir'])
        rel = os.path.relpath(os.path.dirname(run['filtered_dir']), job_dir)
        dirs = {s: (os.path.join(job_dir, f'SUBCELLS_{s}', rel, os.path.basename(run['filtered_dir'])),
                    os.path.join(job_dir, f'SUBCELLS_{s}', rel, os.path.basename(run['final_dir']))) for s in sizes}
        if todo:
            t0 = time.time()
            n = _subcell_run(run['filtered_dir'], raw_dir, {s: dirs[s][0] for s in todo}, res, int(keep))
            print(f"[Accuracy] {label}sub-cells {', '.join(str(s) for s in todo)} px: run {k}/{len(runs)} "
                  f"{rel}/{os.path.basename(run['filtered_dir'])}: {n} window(s) in {time.time() - t0:.0f} s",
                  flush=True)
            for s in todo:
                dfilt, dfinal = dirs[s]
                if settings is not None and glob.glob(os.path.join(dfilt, '*.csv')):
                    os.makedirs(dfinal, exist_ok=True)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):     # its "Saved: ..." lines, per run
                            E.ChipConsensusSelector.select_configs(csv_dir=dfilt, output_dir=dfinal,
                                                                   manual_gcp_csv='', **settings)
                    except Exception as e:
                        print(f'[Accuracy] sub-cell consensus of {rel}: {type(e).__name__}: {e}')
        for s in sizes:
            dfilt, dfinal = dirs[s]
            if os.path.isdir(dfilt):
                os.makedirs(dfinal, exist_ok=True)
                out.append({'filtered_dir': dfilt, 'final_dir': dfinal, 'sweep': run['sweep'], 'raw_dir': raw_dir})
    for s in todo:
        base = os.path.join(job_dir, f'SUBCELLS_{s}')
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, 'SUBCELLS.json'), 'w', encoding='utf-8') as f:
            json.dump({'px': s, 'keep': int(keep), 'res': float(res), 'v': VERSION}, f)
    return out


def _inliers(raw: pd.DataFrame, filt: pd.DataFrame) -> np.ndarray:
    """Which raw matches the RANSAC file kept (by their coordinates, vectorised)."""
    if filt.empty:
        return np.zeros(len(raw), bool)
    rk = pd.MultiIndex.from_frame(raw[['x1_map', 'y1_map', 'x2_map', 'y2_map']].astype(float).round(4))
    fk = pd.MultiIndex.from_frame(filt[['x1-map', 'y1-map', 'x2-map', 'y2-map']].astype(float).round(4)
                                  .set_axis(['x1_map', 'y1_map', 'x2_map', 'y2_map'], axis=1))
    return np.asarray(rk.isin(fk), bool)


def _subcell_run(filtered_dir: str, raw_dir: str, out_dirs: Dict[int, str], res: float, keep: int) -> int:
    """Sub-cell picks of every window and RANSAC setting of one run, for every
    size in out_dirs ({px: folder}); each raw file read once. Windows done."""
    import automatch_engine as E
    raws = {os.path.basename(p)[:-len('_raw.csv')]: p for p in glob.glob(os.path.join(raw_dir, '*_raw.csv'))}
    by_win: Dict[str, List[str]] = {}
    for path in sorted(glob.glob(os.path.join(filtered_dir, '*.csv'))):
        name = os.path.basename(path)
        if '_aff_' not in name or '-sc' in name:
            continue
        win = name[:name.index('_aff_')]
        if win in raws:
            by_win.setdefault(win, []).append(path)
    n = 0
    for win, paths in by_win.items():
        try:
            raw = pd.read_csv(raws[win])
        except Exception:
            continue
        if raw.empty or not set(RAW_COLS) <= set(raw.columns):
            continue
        p1 = raw[['x1_map', 'y1_map']].to_numpy(float)
        p2 = raw[['x2_map', 'y2_map']].to_numpy(float)
        for path in paths:
            name = os.path.basename(path)
            p = E.MatchStatistics._parse_filename(name) or {}
            if not p:
                continue
            try:
                filt = pd.read_csv(path)
            except Exception:
                continue
            inl = _inliers(raw, filt)
            thr = float(p['ransac_threshold'])
            tail = name[name.index('_aff_') + len('_aff_'):].split('_')       # <method>_<thr>_<conf>_.csv
            for px, out_dir in out_dirs.items():
                sel = E.subcell_select(p1, p2, inl, float(px) * float(res), thr, keep)
                if not sel.any():
                    continue
                label = f'{tail[0]}-sc{px}'
                r = raw[sel]
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame({'x1': r['x1'], 'y1': r['y1'], 'x2': r['x2'], 'y2': r['y2'], 'x1-map': r['x1_map'],
                              'y1-map': r['y1_map'], 'x2-map': r['x2_map'], 'y2-map': r['y2_map'],
                              'Distance': r['distance'], 'along': r['along'], 'across': r['across'],
                              'model-type': label, 'threshold': tail[1], 'confidence': tail[2], 'inliers': True}
                             )[OUT_COLS].to_csv(os.path.join(out_dir, f"{win}_aff_{label}_{'_'.join(tail[1:])}"),
                                                index=False)
        n += 1
    return n


def made_sizes(job_dir: str, keep: int) -> List[int]:
    """Sub-cell sizes already made for a job with this keep (SUBCELLS_<px>/)."""
    out = []
    for m in glob.glob(os.path.join(job_dir, 'SUBCELLS_*', 'SUBCELLS.json')):
        try:
            with open(m, encoding='utf-8') as f:
                d = json.load(f)
            if int(d.get('keep', -1)) == int(keep) and d.get('v') == VERSION:
                out.append(int(d['px']))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(out)


def job_accuracy(job_dir: str, truth_csv: str, subcell_px=(), keep: int = 10, radius_m: float = 5000.0,
                 recompute: bool = False, quiet: bool = True) -> pd.DataFrame:
    """TRUTH_MATCH_ACCURACY.csv of one finished job: the plain results and
    every sub-cell size asked for (a number or a list), plus the sizes made
    for it before, so asking for 256 after 512 keeps both in the file; the
    variants are told apart by RANSAC label (magsac90-sc256) and the
    subcell_px column. Read back when made with the same settings."""
    import automatch_cross as X
    import automatch_truth as T
    sizes = subcell_px if isinstance(subcell_px, (list, tuple)) else [subcell_px]
    asked = sorted({int(s) for s in sizes if s and int(s) > 0})
    sizes = sorted(set(asked) | set(made_sizes(job_dir, keep)))
    path = os.path.join(job_dir, 'TRUTH_MATCH_ACCURACY.csv')
    want = {'truth_csv': os.path.abspath(truth_csv), 'subcell_px': sizes, 'keep': int(keep),
            'radius_m': float(radius_m), 'v': VERSION}
    if not recompute and X._reusable(job_dir, 'accuracy', want, 'TRUTH_MATCH_ACCURACY.csv',
                                     ['TRUTH_RANKING.csv', 'RUN_MANIFEST.csv']):
        return pd.read_csv(path)
    crs = T.working_crs_of(job_dir)
    if not crs:
        raise ValueError('working CRS unknown (no RUN_MANIFEST.csv / job_used.json)')
    runs = T.find_runs(job_dir)
    if sizes:
        res = T.job_resolution(job_dir)
        if not res:
            raise ValueError('working resolution unknown: sub-cells need it')
        settings = T.consensus_settings(job_dir)
        runs = runs + subcell_runs(job_dir, sizes, int(keep), res, settings, recompute,
                                   label=f'{os.path.basename(os.path.normpath(job_dir))}: ')
    # nothing rebuilt or written in the job's own folders: a run without its chip consensus is still
    # scored match by match
    r = T.compare(job_dir, truth_csv, radius_m, 'consensus', crs, write=False, quiet=quiet, runs=runs,
                  rebuild=False, progress=f'{os.path.basename(os.path.normpath(job_dir))}: ')
    acc = r.get('accuracy')
    acc = acc if isinstance(acc, pd.DataFrame) else pd.DataFrame()
    if acc.empty:
        print(f'[Accuracy] {os.path.basename(os.path.normpath(job_dir))}: no match files that this version reads '
              f'(raw_matches_* / filtered_*); no match accuracy')
        return acc
    acc.to_csv(path, index=False)
    if sizes and not r['ranking'].empty:
        r['ranking'].to_csv(os.path.join(job_dir, 'TRUTH_RANKING_SUBCELLS.csv'), index=False)
    X._remember(job_dir, 'accuracy', want)
    return acc


def summarize(acc: pd.DataFrame) -> pd.DataFrame:
    """Per sensor x window size (x keypoint count) x sub-cell size (0 =
    plain): the medians over its configurations and jobs; then the same per
    detector + matcher. Columns: level, sensor, window_size, num_features, subcell_px,
    configuration, jobs, configs, precision_inliers, good_inliers_per_window,
    windows_90pct_good, windows_mostly_wrong, truth_rmse_m."""
    if acc.empty:
        return pd.DataFrame()
    a = acc.copy()
    a['subcell_px'] = a['ransac'].astype(str).str.extract(r'-sc(\d+)$')[0].fillna(0).astype(int)
    a['configuration'] = a['detector'].astype(str) + ' + ' + a['matcher'].astype(str)
    cols = ['precision_inliers', 'good_inliers_per_window', 'windows_90pct_good', 'windows_mostly_wrong',
            'truth_rmse_m']
    rows = []
    keys = ['sensor', 'window_size', 'num_features', 'subcell_px']
    for level, by in (('sensor', keys), ('configuration', keys + ['configuration'])):
        for k, g in a.groupby(by, dropna=False):
            r = dict(zip(by, k if isinstance(k, tuple) else (k,)))
            r.update({'level': level, 'jobs': int(g['job'].nunique()), 'configs': int(len(g))})
            for c in cols:
                v = pd.to_numeric(g[c], errors='coerce') if c in g.columns else pd.Series(dtype=float)
                r[c] = round(float(v.median()), 4) if v.notna().any() else None
            rows.append(r)
    out = pd.DataFrame(rows)
    order = ['level', 'sensor', 'window_size', 'num_features', 'subcell_px', 'configuration', 'jobs', 'configs'] + cols
    return out[[c for c in order if c in out.columns]]


def format_summary(s: pd.DataFrame) -> str:
    """Per sensor: window size by window size, the median share of good
    RANSAC inliers and good inliers per window, plain and with sub-cells."""
    if s.empty:
        return ''
    t = s[s['level'] == 'sensor']
    lines = ['[Accuracy] median share of good matches among the RANSAC inliers (good inliers per window), '
             'over every configuration of the jobs:']
    for sensor, g in t.groupby('sensor'):
        lines.append(f'  {sensor}:')
        for r in g.sort_values(['window_size', 'num_features', 'subcell_px']).itertuples():
            p = '-' if pd.isna(r.precision_inliers) else f'{100 * r.precision_inliers:5.1f} %'
            lines.append(f"    {str(r.window_size) + ' px':>8}  keypoints {str(r.num_features):<6} "
                         f"{(f'sub-cells {r.subcell_px}' if r.subcell_px else 'plain'):<15}{p} "
                         f"({r.good_inliers_per_window})"
                         f"   windows >= 90 % right: {r.windows_90pct_good}   jobs {r.jobs}")
    return '\n'.join(lines)
