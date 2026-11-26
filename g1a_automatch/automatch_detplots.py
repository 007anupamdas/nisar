"""
Look vectors detector by detector, for the methods that are close to the
manual points.

The manual points have their own distortion, which changes across the scene.
So the methods are chosen by how close they come to the manual points (the
job's truth ranking: a method's best configuration, within `within` times the
best truth RMSE and reaching at least half of the truth points), not by one
smooth field. Among those methods only their agreement counts: in each cell
(the finest agreement level, 64 px) a method's offset is kept where at least
`need` of them agree there within the cross-check tolerance; how the offset
changes from cell to cell does not matter (no rigid / single-offset filter).
Every agreeing point is kept: no thinning to the 24 / 48 final points.

Writes into <job>/DETECTOR_PLOTS/:
    <detector>_vectors.png    that method's agreed points, with the manual points
    ALL_AGREED_vectors.png    all of them together
    DETECTOR_PLOTS.csv        per method: its truth RMSE (all its points), the
                              agreed points, their truth RMSE and mean offset

    python automatch_job.py detplots <job folder> [--truth pts.csv] [--within 1.5] [--max 8] [--need 2]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def family(det: str) -> str:
    d = re.sub(r'[-_](kt0p2|fm1|fsp1).*$', '', str(det))
    return re.sub(r'^(disk|dog|dedode)_.*$', r'\1', d)


def close_to_manual(ranking: pd.DataFrame, within: float = 1.5, max_n: int = 8,
                    min_reach: float = 0.5) -> pd.DataFrame:
    """Each method's best configuration (one per detector family), those within
    `within` x the best truth RMSE and reaching min_reach of the truth points."""
    r = ranking.dropna(subset=['truth_rmse_m']).copy()
    r = r[r['truth_reached'] >= min_reach * r['truth_total']]
    if r.empty:
        return r
    r['family'] = r['detector'].map(family)
    out = []
    for _, rc in r.groupby('channel', sort=False):        # each band on its own (truth per band)
        best = rc.sort_values('rank').groupby('family', sort=False).head(1)
        lim = within * float(best['truth_rmse_m'].min())
        out.append(best[best['truth_rmse_m'] <= lim].sort_values('truth_rmse_m').head(max_n))
    return pd.concat(out).reset_index(drop=True) if out else r.iloc[0:0]


def run(job_dir: str, truth_csv: str = '', within: float = 1.5, max_n: int = 8, need: int = 2,
        cell_px: Optional[int] = None, quiet: bool = False) -> Dict:
    import automatch_cross as X
    import automatch_report as RP
    import automatch_truth as T
    truth_csv = truth_csv or X.job_truth_csv(job_dir)
    rk_path = os.path.join(job_dir, 'TRUTH_RANKING.csv')
    if not truth_csv or not os.path.exists(rk_path):
        raise ValueError('needs the truth ranking (a job with a truth file, or Compare with truth first)')
    crs = T.working_crs_of(job_dir)
    sel = close_to_manual(pd.read_csv(rk_path), within, max_n)
    if len(sel) < 2 or sel.groupby('channel').size().max() < 2:
        raise ValueError(f'{len(sel)} method(s) close to the manual points: need two in a band to compare')
    out = {'job': job_dir, 'methods': []}
    plots = os.path.join(job_dir, 'DETECTOR_PLOTS')
    os.makedirs(plots, exist_ok=True)
    rows = []
    multi = sel['channel'].nunique() > 1          # several bands (truth per band): each its own plots
    for ch, s in sel.groupby('channel'):
        every = []
        pre = f'{ch}_' if multi else ''
        keys = {(str(r.channel), str(r.sweep), str(r.detector), str(r.matcher), str(r.ransac))
                for r in s.itertuples()}
        cfgs = [c for c in X._configs(job_dir, only=keys) if c['channel'] == str(ch)]
        if len(cfgs) < 2:
            continue
        th = pd.Series([c['ransac'] for c in cfgs]).str.extract(r'(\d+(?:\.\d+)?)$')[0].astype(float)
        tol = 1.5 * float(th.median()) if th.notna().any() else 1000.0
        res = float(T.job_resolution(job_dir) or 10.0)
        L = int(cell_px or (X._job_cell_px(job_dir) or [64])[0])
        pts, cells = X.consistent_points(cfgs, L * res, tol, max(2, min(int(need), len(cfgs))), all_points=True)
        tch = T.truth_for_channel(truth_csv, str(ch))         # per band: its own truth file
        truth = T.truth_in_crs(T.load_truth(tch), crs)
        ctx = RP.plot_context(job_dir, tch)
        try:
            ctx['manual'] = RP.manual_stats(tch)
        except Exception:
            pass
        for c in cfgs:
            mine = pts[(pts['cross_detector'] == c['detector']) & (pts['cross_matcher'] == c['matcher'])
                       & (pts['cross_sweep'] == c['sweep'])] if len(pts) else pts
            rk = s[(s['detector'] == c['detector']) & (s['matcher'] == c['matcher'])].iloc[0]
            row = {'channel': ch, 'detector': c['detector'], 'matcher': c['matcher'], 'ransac': c['ransac'],
                   'sweep': c['sweep'], 'truth_rmse_all_m': round(float(rk['truth_rmse_m']), 1),
                   'truth_reached_all': int(rk['truth_reached']), 'points_all': int(len(c['points'])),
                   'points_agreed': int(len(mine)), 'cell_px': L, 'tolerance_m': round(tol, 1),
                   'truth_file': os.path.basename(tch)}
            if len(mine):
                sc, _ = T.score_config({'points': mine.assign(chip=c['detector'])[['E', 'N', 'dE', 'dN', 'chip']]},
                                       truth, 5000.0)
                row.update(truth_rmse_agreed_m=sc.get('truth_rmse_m'), truth_reached_agreed=sc.get('truth_reached'),
                           mean_dE_m=round(float(mine['dE'].mean()), 1), mean_dN_m=round(float(mine['dN'].mean()), 1))
                p = [{'In_X': e, 'In_Y': n, 'DX_Err': de, 'DY_Err': dn} for e, n, de, dn in
                     zip(mine['E'], mine['N'], mine['dE'], mine['dN'])]
                every += p
                name = re.sub(r'[^A-Za-z0-9._+-]+', '_', f"{pre}{c['detector']}_{c['matcher']}")
                png = os.path.join(plots, f'{name}_vectors.png')
                RP.plot_vectors(p, png, st=RP.stats(p), ctx=dict(ctx, title=(ctx.get('title') or '') + (
                    f"   {c['detector']} + {c['matcher']}: {len(mine)} agreed point(s)")))
                row['plot'] = png
            rows.append(row)
        if every:
            RP.plot_vectors(every, os.path.join(plots, f'{pre}ALL_AGREED_vectors.png'), st=RP.stats(every),
                            ctx=dict(ctx, title=(ctx.get('title') or '') + f'   {len(cfgs)} methods close to the '
                                                 f'manual points: {len(every)} agreed point(s)'))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(plots, 'DETECTOR_PLOTS.csv'), index=False)
    out['methods'] = rows
    if not quiet:
        for r in rows:
            print(f"[DetPlots] {r['detector']} + {r['matcher']}: truth {r['truth_rmse_all_m']} m (all its points); "
                  f"{r['points_agreed']} agreed point(s)"
                  + (f", truth {r.get('truth_rmse_agreed_m')} m at {r.get('truth_reached_agreed')}"
                     if r.get('truth_rmse_agreed_m') is not None else ''), flush=True)
        print(f'[DetPlots] {len(rows)} method(s) close to the manual points -> {plots}', flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Look vectors detector by detector, for the methods close to the '
                                             'manual points (their agreed points, no thinning)')
    ap.add_argument('job_dir')
    ap.add_argument('--truth', default='', help="the manual points (default: the job's)")
    ap.add_argument('--within', type=float, default=1.5, help='methods within this x the best truth RMSE (1.5)')
    ap.add_argument('--max', type=int, default=8, help='at most this many methods (8)')
    ap.add_argument('--need', type=int, default=2, help='a cell counts where at least this many agree (2)')
    ap.add_argument('--cell-px', type=int, default=None, help="cell size in px (default the job's finest level)")
    ap.add_argument('--share-local', default='')
    ap.add_argument('--share-server', default='')
    a = ap.parse_args(argv)
    import automatch_cross as X
    X.SHARE = (a.share_local, a.share_server)
    try:
        run(a.job_dir, a.truth, a.within, a.max, a.need, a.cell_px)
    except ValueError as e:
        print(f'[DetPlots] {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
