"""
How far each detector is from the truth, by distortion level, from the
uploaded logs (log/automatch_<job>*.log and log/TRUTH_RANKING_<job>.csv).

Distortion of a scene: the spread of its truth (manual) points' errors around
their mean offset, sqrt(var dE + var dN), from the report line
'[Report] manual points: n, mean dE .. dN .., RMSE E .. N ..'. A scene that is
only shifted has spread 0; the more the error changes across the scene, the
larger it is. Jobs without that line (older code) are left out.

Per job and detector family (disk_depth / disk_epipolar -> disk, -kt0p2 etc.
-> its base) the best configuration's truth RMSE; per distortion level the
median over the jobs in it. One heat map per sensor, rows ordered by the
detector's median over all its jobs. Also DEVIATION_BY_DISTORTION.csv (the
numbers behind it) and DEVIATION_JOBS.csv (each job's distortion).

    python automatch_deviation.py [log folder] [--out folder]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from typing import Dict

SENSOR_RE = re.compile(r'G1A_?(MV|HV|HS|ML)(?=[_0-9A-Z])')
MANUAL_RE = re.compile(r'manual points: (\d+), mean dE (-?[\d.]+) m dN (-?[\d.]+) m, RMSE E ([\d.]+) N ([\d.]+) m')
LEVELS = [(0.0, 300.0, 'low\n< 0.3 km'), (300.0, 1500.0, 'medium\n0.3–1.5 km'),
          (1500.0, 5000.0, 'high\n1.5–5 km'), (5000.0, float('inf'), 'very high\n> 5 km')]
# the sequential blue ramp (light = small error, dark = large)
RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
        '#1c5cab', '#184f95', '#104281', '#0d366b']
ALL = 'all levels'
NO_TRUTH_JOBS = {'268697011', '2686970151', '268697081'}   # workstation tests: no truth of their own
INK, MUTED, SURFACE = '#1f1f1d', '#6b6b66', '#ffffff'


def family(det: str) -> str:
    d = re.sub(r'[-_](kt0p2|fm1|fsp1).*$', '', str(det))
    return re.sub(r'^(disk|dog|dedode)_.*$', r'\1', d)


def job_distortion(log_dir: str, job: str) -> Dict:
    base = job.split('_')[0]
    txt = ''.join(open(p, errors='ignore').read() for p in sorted(glob.glob(os.path.join(log_dir, f'automatch_{base}*.log'))))
    m = SENSOR_RE.search(txt)
    r = {'job': job, 'sensor': m.group(1) if m else ''}
    mm = MANUAL_RE.search(txt)
    if mm:
        n, me, mn, rme, rmn = map(float, mm.groups())
        r.update(n_truth=int(n), offset_m=math.hypot(me, mn),
                 spread_m=math.sqrt(max(rme ** 2 - me ** 2, 0.0) + max(rmn ** 2 - mn ** 2, 0.0)))
    return r


def collect(log_dir: str):
    import pandas as pd
    jobs, rows = [], []
    for f in sorted(glob.glob(os.path.join(log_dir, 'TRUTH_RANKING_*.csv'))):
        job = os.path.basename(f)[len('TRUTH_RANKING_'):-4]
        if job.split('_')[0] in NO_TRUTH_JOBS:
            continue
        info = job_distortion(log_dir, job)
        if 'spread_m' not in info or not info['sensor']:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not {'truth_rmse_m', 'truth_reached', 'truth_total', 'detector'} <= set(d.columns):
            continue
        d = d[d['truth_reached'] > 0].copy()
        if d.empty:
            continue
        tot = float(d['truth_total'].max())
        d['family'] = d['detector'].map(family)
        d['score'] = d['truth_rmse_m'] * (tot / d['truth_reached']) ** 2     # missed truth points count
        best = d.sort_values('score').groupby('family').first().reset_index()
        jobs.append(info)
        for r in best.itertuples():
            rows.append({'job': job, 'sensor': info['sensor'], 'spread_m': info['spread_m'], 'detector': r.family,
                         'variant': r.detector, 'matcher': r.matcher, 'truth_rmse_m': float(r.truth_rmse_m),
                         'reached': int(r.truth_reached), 'total': int(r.truth_total)})
    return pd.DataFrame(jobs), pd.DataFrame(rows)


def stats(table):
    """All jobs together, per sensor, detector and distortion level (and over
    all levels): jobs, median / mean / 90th-percentile / best / worst truth RMSE
    of the detector's best configuration, and the share of truth points reached."""
    import pandas as pd
    both = pd.concat([table, table.assign(level=ALL)])
    g = both.groupby(['sensor', 'detector', 'level'])
    out = g.agg(jobs=('job', 'nunique'), median_rmse_m=('truth_rmse_m', 'median'), mean_rmse_m=('truth_rmse_m', 'mean'),
                p90_rmse_m=('truth_rmse_m', lambda v: v.quantile(0.9)), best_rmse_m=('truth_rmse_m', 'min'),
                worst_rmse_m=('truth_rmse_m', 'max'), reached=('reached', 'sum'), truth_points=('total', 'sum'))
    out['reached_pct'] = 100.0 * out['reached'] / out['truth_points']
    out = out.reset_index()
    rank = {lab.replace('\n', ' '): i for i, (_, _, lab) in enumerate(LEVELS)}
    out['level'] = out['level'].str.replace('\n', ' ')
    out['_o'] = out['level'].map(rank).fillna(len(rank))
    return out.sort_values(['sensor', '_o', 'median_rmse_m']).drop(columns='_o').round(1)


def level_of(spread: float) -> str:
    return next(lab for lo, hi, lab in LEVELS if lo <= spread < hi)


def plot(table, jobs, png: str, min_jobs: int = 2) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import BoundaryNorm, ListedColormap
    sensors = [s for s in ('MV', 'HS', 'HV', 'ML') if s in set(table['sensor'])]
    # a sensor where even its best detector is over 10 km off has no working match yet (ML so far): left out
    sensors = [s for s in sensors
               if table[table['sensor'] == s].groupby('detector')['truth_rmse_m'].median().min() <= 10000.0]
    levels = [lab for _, _, lab in LEVELS]
    panels = []
    for s in sensors:
        t = table[table['sensor'] == s]
        n_jobs = t['job'].nunique()
        keep = t.groupby('detector')['job'].nunique()
        keep = keep[keep >= min(min_jobs, n_jobs)].index
        t = t[t['detector'].isin(keep)]
        order = t.groupby('detector')['truth_rmse_m'].median().sort_values().index.tolist()
        reach = (t.groupby('detector')['reached'].sum() / t.groupby('detector')['total'].sum() * 100).to_dict()
        cols = [lv for lv in levels if lv in set(t['level'])] + [ALL]
        t = pd.concat([t, t.assign(level=ALL)])
        grid = np.full((len(order), len(cols)), np.nan)
        nn = np.zeros_like(grid, dtype=int)
        for i, det in enumerate(order):
            for k, lv in enumerate(cols):
                v = t[(t['detector'] == det) & (t['level'] == lv)]['truth_rmse_m']
                if len(v):
                    grid[i, k], nn[i, k] = float(v.median()), len(v)
        njobs = [jobs[(jobs['sensor'] == s) & ((jobs['level'] == lv) | (lv == ALL))].shape[0] for lv in cols]
        panels.append((s, order, cols, grid, nn, njobs, reach))
    # one colour scale for every panel: log-spaced bins from 30 m to 100 km
    bounds = np.geomspace(30.0, 100000.0, len(RAMP) + 1)
    cmap, norm = ListedColormap(RAMP), BoundaryNorm(bounds, len(RAMP))
    heights = [max(len(p[1]), 3) for p in panels]
    head_in = 1.3
    fig_h = head_in + 0.26 * sum(heights) + 0.9 * len(panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, fig_h), gridspec_kw={'height_ratios': heights},
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (s, order, cols, grid, nn, njobs, reach) in zip(axes[:, 0], panels):
        masked = np.ma.masked_invalid(grid)
        ax.pcolormesh(masked, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
        ax.set_facecolor('#f0efec')                  # not run at that level
        for i in range(grid.shape[0]):
            for k in range(grid.shape[1]):
                v = grid[i, k]
                if np.isfinite(v):
                    dark = norm(v) >= len(RAMP) * 0.55
                    txt = f'{v / 1000:.1f} km' if v >= 1000 else f'{v:.0f} m'
                    ax.text(k + 0.5, i + 0.5, txt + (f'  ({nn[i, k]})' if nn[i, k] > 1 else ''), ha='center',
                            va='center', fontsize=8.5, color=SURFACE if dark else INK)
        ax.set_yticks(np.arange(len(order)) + 0.5)
        ax.set_yticklabels([f'{d}  ({reach[d]:.0f} % reached)' for d in order], fontsize=9, color=INK)
        ax.axvline(len(cols) - 1, color=INK, lw=1.2)          # the all-levels column apart
        ax.set_xticks(np.arange(len(cols)) + 0.5)
        ax.set_xticklabels([f'{c}\n{n} job{"s" if n != 1 else ""}' for c, n in zip(cols, njobs)], fontsize=8.5,
                           color=MUTED)
        ax.xaxis.tick_top()
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{s}', loc='left', fontsize=12, fontweight='bold', color=INK, pad=34)
    fig.text(0.02, 1 - 0.25 / fig_h, 'Deviation from truth per detector, by scene distortion (all jobs)',
             ha='left', va='top', fontsize=14, fontweight='bold', color=INK)
    fig.text(0.02, 1 - 0.6 / fig_h,
             'Cell: median truth RMSE of the detector\'s best configuration over all jobs at that level (n jobs '
             'in brackets); last column all levels together.\nRows: best first; % of the truth points it '
             'reached over all its jobs.\nDistortion: spread of the truth points\' errors around their '
             'mean offset (0 = the scene is only shifted). Grey: not run at that level.',
             fontsize=8.5, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - head_in / fig_h), h_pad=2.0)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('log_dir', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log'))
    ap.add_argument('--out', default='', help='where the plot and tables go (default: the log folder)')
    a = ap.parse_args(argv)
    out = a.out or a.log_dir
    os.makedirs(out, exist_ok=True)
    jobs, table = collect(a.log_dir)
    if table.empty:
        print('no job with truth ranking and manual-point statistics')
        return 1
    jobs['level'] = jobs['spread_m'].map(level_of)
    table['level'] = table['spread_m'].map(level_of)
    jobs.round(1).to_csv(os.path.join(out, 'DEVIATION_JOBS.csv'), index=False)
    agg = stats(table)
    agg.to_csv(os.path.join(out, 'DEVIATION_BY_DISTORTION.csv'), index=False)
    png = plot(table, jobs, os.path.join(out, 'DEVIATION_BY_DISTORTION.png'))
    print(f'{len(jobs)} job(s) with a known distortion: '
          + ', '.join(f'{s} {n}' for s, n in jobs['sensor'].value_counts().items()) + f' -> {png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
