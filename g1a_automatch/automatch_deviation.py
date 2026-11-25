"""
How far each detector is from the truth, by distortion level, from the
uploaded logs (log/automatch_<job>*.log and log/TRUTH_RANKING_<job>.csv).

Distortion of a scene: how much its truth (manual) points' errors spread
around their mean offset. Newer jobs: sqrt(var dE + var dN) from the report
line '[Report] manual points: n, mean dE .. dN .., RMSE E .. N ..' of its log;
older jobs: error_spread_km of BENCH_JOBS.csv (the benchmark's measure, the
largest distance from the mean offset). A scene that is only shifted has
spread 0; the more the error changes across the scene, the larger it is.

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
LEVELS = [(0.0, 500.0, 'low\n< 0.5 km'), (500.0, 2000.0, 'medium\n0.5–2 km'),
          (2000.0, 8000.0, 'high\n2–8 km'), (8000.0, float('inf'), 'very high\n> 8 km')]
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
    ts = re.search(r'(20\d\d)[-/](\d\d)[-/](\d\d)[ T](\d\d):(\d\d)', txt)
    if ts:
        r['started'] = '{}-{}-{} {}:{}'.format(*ts.groups())
    mm = MANUAL_RE.search(txt)
    if mm:
        n, me, mn, rme, rmn = map(float, mm.groups())
        r.update(n_truth=int(n), offset_m=math.hypot(me, mn),
                 spread_m=math.sqrt(max(rme ** 2 - me ** 2, 0.0) + max(rmn ** 2 - mn ** 2, 0.0)))
    return r


def bench_spread(log_dir: str) -> Dict[str, Dict]:
    """Older jobs: the truth error spread the benchmark measured (BENCH_JOBS.csv,
    error_spread_km: the largest distance of a truth point's error from the
    scene's mean offset), by job number."""
    import pandas as pd
    out = {}
    for f in glob.glob(os.path.join(log_dir, 'BENCH_JOBS*.csv')):
        try:
            b = pd.read_csv(f)
        except Exception:
            continue
        for r in b.to_dict('records'):
            if pd.notna(r.get('error_spread_km')):
                out[re.sub(r'^output', '', str(r['job']))] = {
                    'spread_m': float(r['error_spread_km']) * 1000.0,
                    'sensor': str(r['sensor']) if pd.notna(r.get('sensor')) else '',
                    'offset_m': math.hypot(float(r.get('error_dE_km') or 0), float(r.get('error_dN_km') or 0)) * 1000,
                    'n_truth': r.get('truth_points'), 'spread_from': 'BENCH_JOBS (largest)'}
    return out


TRUTH_FILES = {'347': 'gt_347.csv'}       # truth files uploaded beside the logs, by job


def truth_file_spread(path: str) -> Dict:
    """Spread of a truth file's own errors (DX_Err, DY_Err): rms around the mean."""
    import pandas as pd
    t = pd.read_csv(path, encoding='utf-8-sig')
    dx, dy = t['DX_Err'].astype(float), t['DY_Err'].astype(float)
    return {'spread_m': math.sqrt(dx.var(ddof=0) + dy.var(ddof=0)), 'offset_m': math.hypot(dx.mean(), dy.mean()),
            'n_truth': len(t), 'spread_from': f'truth file {os.path.basename(path)} (rms)'}


def collect(log_dir: str):
    import pandas as pd
    jobs, rows = [], []
    bench = bench_spread(log_dir)
    for f in sorted(glob.glob(os.path.join(log_dir, 'TRUTH_RANKING_*.csv'))):
        job = os.path.basename(f)[len('TRUTH_RANKING_'):-4]
        if job.split('_')[0] in NO_TRUTH_JOBS:
            continue
        info = job_distortion(log_dir, job)
        if 'spread_m' in info:
            info['spread_from'] = 'log: manual points (rms)'
        elif job.split('_')[0] in TRUTH_FILES and os.path.exists(os.path.join(log_dir, TRUTH_FILES[job.split('_')[0]])):
            info.update(truth_file_spread(os.path.join(log_dir, TRUTH_FILES[job.split('_')[0]])))
        elif job.split('_')[0] in bench:
            b = bench[job.split('_')[0]]
            info.update({k: v for k, v in b.items() if k != 'sensor'})
            info['sensor'] = info['sensor'] or b['sensor']
        if not info['sensor']:
            continue
        info.setdefault('spread_m', float('nan'))
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
                         'reached': int(r.truth_reached), 'total': int(r.truth_total),
                         's_per_window': float(getattr(r, 'sec_per_window', float('nan')))})
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
    if spread != spread:                       # NaN: not known
        return ''
    return next(lab for lo, hi, lab in LEVELS if lo <= spread < hi)


LEVEL_COLORS = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']   # low .. very high (ordinal blue)
UNKNOWN = '#b4b2ab'


def plot_top(table, jobs, png: str, top: int = 10, min_reach: float = 0.5) -> str:
    """Per sensor, its top detectors over all jobs (by median truth RMSE, among
    those run in at least half of the sensor's jobs that reach at least half of
    the truth points): one dot per job (colour:
    the scene's distortion level, grey when not known), the median as a bar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    sensors = [s for s in ('MV', 'HS', 'HV', 'ML') if s in set(table['sensor'])]
    sensors = [s for s in sensors
               if table[table['sensor'] == s].groupby('detector')['truth_rmse_m'].median().min() <= 10000.0]
    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 3.4 * len(sensors) + 1.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    labels = [lab for _, _, lab in LEVELS]
    for ax, s in zip(axes[:, 0], sensors):
        t = table[table['sensor'] == s]
        n_jobs = t['job'].nunique()
        cnt = t.groupby('detector')['job'].nunique()
        rch = t.groupby('detector')['reached'].sum() / t.groupby('detector')['total'].sum()
        ok = [d for d in cnt.index if cnt[d] >= max(2, n_jobs // 2) and rch[d] >= min_reach]
        med = t[t['detector'].isin(ok)].groupby('detector')['truth_rmse_m'].median().sort_values()
        dets = med.index[:top].tolist()
        for i, d in enumerate(dets):
            u = t[t['detector'] == d]
            x = i + rng.uniform(-0.22, 0.22, len(u))
            cols = [LEVEL_COLORS[labels.index(lv)] if lv in labels else UNKNOWN for lv in u['level']]
            ax.scatter(x, u['truth_rmse_m'] / 1000.0, s=26, c=cols, edgecolors=SURFACE, linewidths=0.8, zorder=3)
            ax.plot([i - 0.32, i + 0.32], [med[d] / 1000.0] * 2, color=INK, lw=2.2, solid_capstyle='round',
                    zorder=4)
            reach = 100.0 * u['reached'].sum() / max(u['total'].sum(), 1)
            ax.text(i, 0.97, f'{med[d]:.0f} m\n{reach:.0f} %', transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=8, color=MUTED)
        ax.set_yscale('log')
        ax.set_ylim(0.02, max(200.0, float(t[t['detector'].isin(dets)]['truth_rmse_m'].max()) / 1000.0 * 12.0))
        ax.set_xticks(range(len(dets)))
        ax.set_xticklabels(dets, rotation=30, ha='right', fontsize=9, color=INK)
        ax.set_ylabel('deviation from truth (km)', fontsize=9, color=MUTED)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f'{v:g}'))
        ax.grid(axis='y', color='#e7e6e2', lw=0.8, zorder=0)
        ax.tick_params(length=0, colors=MUTED)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#cfcdc6')
        ax.set_title(f'{s}: top {len(dets)} of {len(med)} detectors reaching \u2265 {100 * min_reach:.0f} % of the '
                     f'truth points, {n_jobs} jobs', loc='left', fontsize=12,
                     fontweight='bold', color=INK, pad=26)
    handles = [Line2D([], [], marker='o', ls='', color=c, markersize=7, label=lab.replace('\n', ' '))
               for c, lab in zip(LEVEL_COLORS, labels)]
    handles += [Line2D([], [], marker='o', ls='', color=UNKNOWN, markersize=7, label='distortion not known'),
                Line2D([], [], color=INK, lw=2.2, label='median over the jobs')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.985), ncol=6, frameon=False, fontsize=8.5,
               title='scene distortion (spread of the truth errors)', title_fontsize=8.5, alignment='left')
    fig.text(0.01, 1.0, 'Top detectors per sensor: deviation from truth over all jobs', ha='left', va='bottom',
             fontsize=14, fontweight='bold', color=INK)
    fig.text(0.99, 0.003, 'Each dot one job: truth RMSE of the detector\'s best configuration. Above each: median, '
             'share of truth points reached.', ha='right', va='bottom', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), h_pad=2.5)
    fig.savefig(png, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return png


def top_detectors(t, top: int, min_reach: float = 0.5):
    """(the top detectors of one sensor's rows by median truth RMSE, among those
    run in at least half of its jobs that reach at least min_reach of the truth
    points; their medians)."""
    n_jobs = t['job'].nunique()
    cnt = t.groupby('detector')['job'].nunique()
    rch = t.groupby('detector')['reached'].sum() / t.groupby('detector')['total'].sum()
    ok = [d for d in cnt.index if cnt[d] >= max(2, n_jobs // 2) and rch[d] >= min_reach]
    med = t[t['detector'].isin(ok)].groupby('detector')['truth_rmse_m'].median().sort_values()
    return med.index[:top].tolist(), med


def _smooth(x, y, n: int = 24):
    """A smooth curve through (x, y) on a log y axis: shape-preserving cubic
    (PCHIP) in log10(y), so it never overshoots between two methods; NaN gaps
    split it."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    out = []
    ok = np.isfinite(y)
    i = 0
    while i < len(x):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and ok[j + 1]:
            j += 1
        xs, ys = np.asarray(x[i:j + 1], float), np.log10(np.asarray(y[i:j + 1], float))
        if len(xs) >= 2:
            xx = np.linspace(xs[0], xs[-1], n * (len(xs) - 1) + 1)
            out.append((xx, 10 ** PchipInterpolator(xs, ys)(xx)))
        i = j + 1
    return out


def plot_lines(table, png: str, top: int = 5, min_reach: float = 0.5, smooth: bool = False,
               title: str = '', note: str = '', fastest: int = 0) -> str:
    """Per sensor, its top methods on the x axis, the truth error on the y axis,
    one line per job through its errors with those methods (colour: the scene's
    distortion level, grey when not known); the median over the jobs bold."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import numpy as np
    from matplotlib.lines import Line2D
    sensors = [s for s in ('MV', 'HS', 'HV', 'ML') if s in set(table['sensor'])]
    sensors = [s for s in sensors
               if table[table['sensor'] == s].groupby('detector')['truth_rmse_m'].median().min() <= 10000.0]
    labels = [lab for _, _, lab in LEVELS]
    rows = top > 5                                   # many methods: one sensor per row
    if rows:
        fig, axes = plt.subplots(len(sensors), 1, figsize=(13, 4.6 * len(sensors) + 1.6), squeeze=False)
        axes = axes[:, 0]
    else:
        fig, axes = plt.subplots(1, len(sensors), figsize=(5.2 * len(sensors), 6.4), squeeze=False)
        axes = axes[0]
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes, sensors):
        t = table[table['sensor'] == s]
        dets, med = top_detectors(t, top, min_reach)
        piv = t[t['detector'].isin(dets)].pivot_table(index='job', columns='detector', values='truth_rmse_m',
                                                      aggfunc='min')[dets] / 1000.0
        lvl = t.drop_duplicates('job').set_index('job')['level']
        x = np.arange(len(dets))
        tags = []
        for job, row in piv.iterrows():
            lv = lvl.get(job, '')
            col = LEVEL_COLORS[labels.index(lv)] if lv in labels else UNKNOWN
            y = row.to_numpy(float)
            if smooth:
                for xx, yy in _smooth(x, y):
                    ax.plot(xx, yy, color=col, lw=1.3, alpha=0.9, zorder=2)
                ax.plot(x, y, ls='', color=col, marker='o', markersize=4.5, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, zorder=3)
            else:
                ax.plot(x, y, color=col, lw=1.3, alpha=0.9, marker='o', markersize=4.5, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, zorder=2)
            last = np.where(np.isfinite(y))[0]
            if len(last) and np.nanmax(y) >= 3 * float(np.nanmedian(piv.to_numpy())):   # label the far-off jobs
                k = last[-1]
                tags.append([x[k], y[k], str(job) if len(str(job)) <= 10 else '…' + str(job)[-8:]])
        tags.sort(key=lambda r: r[1])                       # spread the labels: at least 22 % apart (log axis)
        for i in range(1, len(tags)):
            if tags[i][0] == tags[i - 1][0] and tags[i][1] < tags[i - 1][1] * 1.22:
                tags[i][1] = tags[i - 1][1] * 1.22
        for tx, ty, lab in tags:
            ax.text(tx + 0.1, ty, lab, fontsize=7, color=MUTED, va='center')
        my = np.array([med[d] / 1000.0 for d in dets])
        for xx, yy in (_smooth(x, my) if smooth else [(x, my)]):
            ax.plot(xx, yy, color=INK, lw=2.6, zorder=4)
        ax.plot(x, my, ls='', color=INK, marker='o', markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=5)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f'{v:g}'))
        ax.set_xticks(x)
        spw = t.groupby('detector')['s_per_window'].median() if 's_per_window' in t else None
        fast = []
        if fastest and spw is not None:          # the quickest of these methods: a shaded band each
            sp = spw.reindex(dets).dropna().sort_values()
            fast = sp.index[:fastest].tolist()
            for k, d in enumerate(fast):
                i = dets.index(d)
                ax.axvspan(i - 0.42, i + 0.42, color='#fff1cc', zorder=0, lw=0)
                ax.text(i, 1.0, f'fastest {k + 1}', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                        fontsize=8.5, fontweight='bold', color='#8a5a00')

        def lab(d):
            sec = f'\n{spw[d]:.2f} s/window' if spw is not None and d in spw and spw[d] == spw[d] else ''
            return f'{d}\nmedian {med[d]:.0f} m' + sec
        ax.set_xticklabels([lab(d) for d in dets], rotation=0 if rows else 25,
                           ha='center' if rows else 'right', fontsize=8.5 if rows else 9, color=INK)
        ax.set_xlim(-0.3, len(dets) - 0.2)
        ax.grid(axis='y', color='#e7e6e2', lw=0.8, zorder=0)
        ax.tick_params(length=0, colors=MUTED)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#cfcdc6')
        ax.set_title(f'{s}: top {len(dets)}, {len(piv)} jobs', loc='left', fontsize=12, fontweight='bold',
                     color=INK, pad=20 if fast else 6)
    for ax in (axes if rows else axes[:1]):
        ax.set_ylabel('error against truth (km, log)', fontsize=9.5, color=MUTED)
    handles = [Line2D([], [], color=c, lw=1.6, marker='o', markersize=5, label=lab.replace('\n', ' '))
               for c, lab in zip(LEVEL_COLORS, labels)]
    handles += [Line2D([], [], color=UNKNOWN, lw=1.6, marker='o', markersize=5, label='distortion not known'),
                Line2D([], [], color=INK, lw=2.6, marker='o', markersize=7, label='median over the jobs')]
    top_y = 1 - 0.75 / fig.get_figheight()
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, top_y), ncol=6, frameon=False,
               fontsize=8.5, title='one line per job, by scene distortion (spread of the truth errors)',
               title_fontsize=8.5, alignment='left')
    fig.text(0.005, 1 - 0.12 / fig.get_figheight(),
             title or f'Top {top} methods per sensor: error against truth, job by job', ha='left', va='top',
             fontsize=14, fontweight='bold', color=INK)
    fig.text(0.995, 0.005, (note + '   ' if note else '') + f'Top {top} by median error among methods reaching at '
             'least half of the truth points. Far-off jobs labelled with their number.', ha='right', va='bottom',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 1.55 / fig.get_figheight()), w_pad=2.0, h_pad=2.5)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


# speed classes (median seconds per window): upper limits, then (label, colour)
SPEED_CLASSES = ([0.5, 2.0, float('inf')],
                 [('fast < 0.5 s', '#1baf7a'), ('medium 0.5–2 s', '#2a78d6'), ('slow > 2 s', '#eb6834')])


def speed_color(spw) -> str:
    if spw != spw:                              # NaN: speed not known
        return UNKNOWN
    return next(c for lim, (_l, c) in zip(*SPEED_CLASSES) if spw < lim)


def plot_closest(table, png: str, title: str = '', note: str = '', min_reach: float = 0.5, show: int = 15) -> str:
    """Per sensor, how often each method was the closest to truth in a job (the
    lowest truth RMSE among the methods reaching at least min_reach of that
    job's truth points), and how often it was among the 3 closest: horizontal
    bars, the most often closest first."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    import numpy as np
    import pandas as pd
    t = table[table['reached'] >= min_reach * table['total']].copy()
    t['rank'] = t.groupby('job')['truth_rmse_m'].rank(method='min')
    sensors = [s for s in ('MV', 'HS', 'HV', 'ML') if s in set(t['sensor'])]
    sensors = [s for s in sensors if t[t['sensor'] == s]['truth_rmse_m'].min() <= 10000.0]
    panels = []
    for s in sensors:
        u = t[t['sensor'] == s]
        n_jobs = u['job'].nunique()
        win = u[u['rank'] == 1].groupby('detector')['job'].nunique()
        top3 = u[u['rank'] <= 3].groupby('detector')['job'].nunique()
        ran = u.groupby('detector')['job'].nunique()
        spw = u.groupby('detector')['s_per_window'].median()
        best_rmse = u[u['rank'] == 1].groupby('detector')['truth_rmse_m'].median()
        d = pd.DataFrame({'win': win, 'top3': top3, 'ran': ran, 'spw': spw, 'rmse': best_rmse}).fillna(
            {'win': 0, 'top3': 0})
        d = d[d['top3'] > 0].sort_values(['win', 'top3'], ascending=False).head(show)
        panels.append((s, n_jobs, d))
    heights = [max(len(p[2]), 3) for p in panels]
    fig_h = 1.6 + 0.32 * sum(heights) + 0.9 * len(panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, fig_h), gridspec_kw={'height_ratios': heights},
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    from matplotlib.patches import Patch
    for ax, (s, n_jobs, d) in zip(axes[:, 0], panels):
        y = np.arange(len(d))
        cols = [speed_color(v) for v in d['spw']]
        ax.barh(y, d['top3'], height=0.62, color=cols, alpha=0.32, zorder=2)        # among the 3 closest
        ax.barh(y, d['win'], height=0.62, color=cols, zorder=3)                     # the closest
        for i, (det, r) in enumerate(d.iterrows()):
            txt = f"closest in {int(r['win'])}, top 3 in {int(r['top3'])} of {int(r['ran'])} jobs"
            if r['win'] > 0 and r['rmse'] == r['rmse']:
                txt += f"  ·  {r['rmse']:.0f} m when closest"
            if r['spw'] == r['spw']:
                txt += f"  ·  {r['spw']:.2f} s/window"
            ax.text(r['top3'] + 0.15, i, txt, va='center', fontsize=8, color=MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index, fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(float(d['top3'].max()), 1.0) * 2.4)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(axis='x', color='#e7e6e2', lw=0.8, zorder=0)
        ax.tick_params(length=0, colors=MUTED)
        for sp in ('top', 'right', 'bottom'):
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color('#cfcdc6')
        ax.set_xlabel('jobs', fontsize=9, color=MUTED)
        ax.set_title(f'{s}: {n_jobs} jobs', loc='left', fontsize=12, fontweight='bold', color=INK)
    h = [Patch(color=c, label=lab) for lab, c in SPEED_CLASSES[1]]
    h += [Patch(color='#8a8a85', label='solid: the closest'), Patch(color='#8a8a85', alpha=0.32,
                                                                   label='light: among the 3 closest')]
    fig.legend(handles=h, loc='upper left', bbox_to_anchor=(0.01, 1 - 0.75 / fig_h), ncol=5, frameon=False,
               fontsize=9, title='colour: speed (median seconds per window)', title_fontsize=8.5,
               alignment='left')
    fig.text(0.01, 1 - 0.15 / fig_h, title or 'Which method is closest to truth, sensor by sensor', ha='left',
             va='top', fontsize=14, fontweight='bold', color=INK)
    fig.text(0.99, 0.004, (note + '   ' if note else '') + f'Per job, among the methods reaching at least '
             f'{100 * min_reach:.0f} % of its truth points; ranked by how often closest, then how often in the 3 '
             'closest.', ha='right', va='bottom', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1 - 1.45 / fig_h), h_pad=2.0)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


def plot_time_accuracy(table, png: str, title: str = '', note: str = '', min_reach: float = 0.5,
                       useful: float = 3.0) -> str:
    """Per sensor, each method once: its median seconds per window (x, log) against
    its median truth RMSE (y, log, m) over the jobs; filled when it reaches at
    least min_reach of the truth points, hollow otherwise. Among the methods
    within `useful` x the best error, those no other one beats on both time and
    accuracy are joined; they and the most accurate ones are numbered and listed
    beside the panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    t = table.dropna(subset=['s_per_window', 'truth_rmse_m'])
    t = t[t['s_per_window'] > 0]
    sensors = [s for s in ('MV', 'HS', 'HV', 'ML') if s in set(t['sensor'])]
    sensors = [s for s in sensors if t[t['sensor'] == s].groupby('detector')['truth_rmse_m'].median().min() <= 10000.0]
    if not sensors:
        return ''
    fig, axes = plt.subplots(len(sensors), 1, figsize=(13, 4.9 * len(sensors) + 1.4), squeeze=False)
    axes = axes.T                                    # axes[0]: one per sensor, one sensor per row
    fig.patch.set_facecolor(SURFACE)
    nice = [10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    for ax, s in zip(axes[0], sensors):
        u = t[t['sensor'] == s]
        n_jobs = u['job'].nunique()
        g = u.groupby('detector').agg(rmse=('truth_rmse_m', 'median'), spw=('s_per_window', 'median'),
                                      jobs=('job', 'nunique'), reached=('reached', 'sum'), total=('total', 'sum'))
        g = g[g['jobs'] >= max(2, n_jobs // 3)]
        g['reach'] = g['reached'] / g['total']
        ok = g[g['reach'] >= min_reach]
        lim = useful * float(ok['rmse'].min()) if len(ok) else 0.0
        cand = ok[ok['rmse'] <= lim].sort_values('spw')
        front, best = [], float('inf')               # faster first: kept while more accurate than all faster ones
        for d, r in cand.iterrows():
            if r['rmse'] < best:
                front.append(d)
                best = r['rmse']
        named = list(dict.fromkeys(front + list(ok.sort_values('rmse').index[:5])))
        # linear time axis: up to a bit past most methods (and every named one); slower ones at the edge
        xcap = max(1.3 * float(g['spw'].quantile(0.9)), 1.15 * float(g.loc[named, 'spw'].max()) if named else 0.0)
        far = g[g['spw'] > xcap]
        g = g.assign(spw_raw=g['spw'], spw=g['spw'].clip(upper=xcap))
        ok = g[g['reach'] >= min_reach]
        low = g[g['reach'] < min_reach]
        ax.scatter(low['spw'], low['rmse'], s=34, facecolors='none', edgecolors=UNKNOWN, linewidths=1.2, zorder=2,
                   label=f'reaches < {100 * min_reach:.0f} % of the truth points')
        rest = ok.drop(index=front)
        ax.scatter(rest['spw'], rest['rmse'], s=40, color='#86b6ef', edgecolors=SURFACE, linewidths=0.8, zorder=3,
                   label='method')
        fr = ok.loc[front]
        ax.plot(fr['spw'], fr['rmse'], color=INK, lw=1.4, zorder=4)
        ax.scatter(fr['spw'], fr['rmse'], s=64, color='#1c5cab', edgecolors=SURFACE, linewidths=1.0, zorder=5,
                   label='no other both faster and more accurate')
        named.sort(key=lambda d: g.loc[d, 'rmse'])
        ly = None
        for d, r in far.sort_values('rmse').iterrows():     # slower than the axis: an arrowhead at the edge, named
            ax.scatter([xcap], [r['rmse']], marker='>', s=46, color='#86b6ef' if r['reach'] >= min_reach else UNKNOWN,
                       zorder=6)
            y = r['rmse'] if ly is None else max(r['rmse'], ly * 1.12)     # labels apart on the log axis
            ly = y
            ax.annotate(f"{d} {r['spw']:.1f} s", (xcap, y), xytext=(-8, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=7, color=MUTED, zorder=6)
        lines = []
        for k, d in enumerate(named, 1):
            r = g.loc[d]
            ax.annotate(str(k), (r['spw'], r['rmse']), xytext=(5, 4), textcoords='offset points', fontsize=8,
                        fontweight='bold', color=INK, zorder=6)
            lines.append(f"{k:>2}  {d:<18.18} {r['rmse']:>6.0f} m {r['spw_raw']:>6.2f} s"
                         + ('  *' if d in front else ''))
        ax.text(1.02, 1.0, 'numbered (* joined):\n' + '\n'.join(lines), transform=ax.transAxes, ha='left',
                va='top', fontsize=7.5, family='monospace', color=INK)
        ax.set_xlim(0, xcap * 1.04)
        ax.set_yscale('log')
        lo, hi = float(g['rmse'].min()), float(g['rmse'].max())
        ax.yaxis.set_major_locator(mt.FixedLocator([v for v in nice if lo / 1.5 <= v <= hi * 1.5] or nice))
        ax.yaxis.set_minor_locator(mt.NullLocator())
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _p: f'{v:g} m' if v < 1000 else f'{v / 1000:g} km'))
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _p: f'{v:g}'))
        ax.set_xlabel('seconds per window (median over the jobs)' + (f'; slower than {xcap:.1f} s: at the edge'
                                                                         if len(far) else ''), fontsize=9, color=MUTED)
        ax.set_ylabel('error against truth (median over the jobs, log)', fontsize=9, color=MUTED)
        ax.grid(color='#e7e6e2', lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_title(f'{s}: {len(g)} methods, {n_jobs} jobs', loc='left', fontsize=12, fontweight='bold', color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    H = fig.get_figheight()
    fig.legend(h, l, loc='upper left', bbox_to_anchor=(0.005, 1 - 0.55 / H), ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.005, 1 - 0.1 / H, title or 'Time against accuracy, method by method', ha='left', va='top',
             fontsize=14, fontweight='bold', color=INK)
    fig.text(0.995, 0.005, (note + '   ' if note else '') + "Each method's best configuration per job; down and to "
             f"the left is better. Joined: among those within {useful:g} x the best error, no other method is both "
             'faster and more accurate.', ha='right', va='bottom', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.3 / H, 0.7, 1 - 1.0 / H), h_pad=2.5)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


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
    head_in = 1.5
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
             'mean offset (0 = only shifted):\nnewer jobs from their logs, older ones from BENCH_JOBS.csv. Grey: not run at that level.',
             fontsize=8.5, color=MUTED, va='top')
    fig.tight_layout(rect=(0, 0, 1, 1 - head_in / fig_h), h_pad=2.0)
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


def is_workstation(job: str) -> bool:
    """A workstation run: not a server job number (a product name, or a long ID)."""
    b = str(job).split('_')[0]
    return not b.isdigit() or int(b) >= 100000


def recent_jobs(jobs, after_job: int):
    """(server jobs above after_job, plus the workstation runs that started at
    or after the first of them; that start time)."""
    server = jobs[[not is_workstation(j) and int(str(j).split('_')[0]) > after_job for j in jobs['job']]]
    if server.empty:
        return [], ''
    since = str(server['started'].dropna().min()) if 'started' in server else ''
    ws = jobs[[is_workstation(j) for j in jobs['job']]]
    if since and 'started' in ws:
        ws = ws[ws['started'].astype(str) >= since]
    else:
        ws = ws.iloc[0:0]
    return server['job'].tolist() + ws['job'].tolist(), since


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('log_dir', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log'))
    ap.add_argument('--out', default='', help='where the plot and tables go (default: the log folder)')
    ap.add_argument('--after-job', type=int, default=400,
                    help='also a plot of only the server jobs above this number and the workstation runs that '
                         'started after the first of them (default 400)')
    a = ap.parse_args(argv)
    out = a.out or a.log_dir
    os.makedirs(out, exist_ok=True)
    jobs, table = collect(a.log_dir)
    if table.empty:
        print('no job with truth ranking and manual-point statistics')
        return 1
    jobs['level'] = jobs['spread_m'].map(level_of)
    table['level'] = table['spread_m'].map(level_of)
    top_png = plot_top(table, jobs, os.path.join(out, 'TOP_DETECTORS_DEVIATION.png'))
    lines_png = plot_lines(table, os.path.join(out, 'TOP5_BY_JOB.png'))
    ta_png = plot_time_accuracy(table, os.path.join(out, 'TIME_VS_ACCURACY.png'),
                                note=f"{jobs['job'].nunique()} job(s) with truth.")
    print(f"{jobs['job'].nunique()} job(s) with truth -> {top_png}, {lines_png}, {ta_png}")
    sel, since = recent_jobs(jobs, a.after_job)
    if sel:
        rt = table[table['job'].isin(sel)]
        n_ws = sum(1 for j in sel if is_workstation(j))
        png = plot_lines(rt, os.path.join(out, f'TOP10_BY_JOB_AFTER_{a.after_job}.png'), top=10, smooth=True,
                         fastest=3,
                         title=f'Top 10 methods per sensor: error against truth, jobs above {a.after_job} '
                               f'(and workstation runs since {since})',
                         note=f'{len(sel) - n_ws} server + {n_ws} workstation job(s). Shaded: the 3 fastest '
                              f'of the 10 (median seconds per window over these jobs).')
        cpng = plot_closest(rt, os.path.join(out, f'CLOSEST_TO_TRUTH_AFTER_{a.after_job}.png'),
                            title=f'Which method is closest to truth, sensor by sensor (jobs above {a.after_job})',
                            note=f'{len(sel) - n_ws} server + {n_ws} workstation job(s).')
        tpng = plot_time_accuracy(rt, os.path.join(out, f'TIME_VS_ACCURACY_AFTER_{a.after_job}.png'),
                                  title=f'Time against accuracy, method by method (jobs above {a.after_job})',
                                  note=f'{len(sel) - n_ws} server + {n_ws} workstation job(s).')
        print(f'{len(sel)} job(s) above {a.after_job} / since {since} -> {png}, {cpng}, {tpng}')
    jobs, table = jobs[jobs['level'] != ''], table[table['level'] != '']
    jobs.round(1).to_csv(os.path.join(out, 'DEVIATION_JOBS.csv'), index=False)
    agg = stats(table)
    agg.to_csv(os.path.join(out, 'DEVIATION_BY_DISTORTION.csv'), index=False)
    png = plot(table, jobs, os.path.join(out, 'DEVIATION_BY_DISTORTION.png'))
    print(f'{len(jobs)} job(s) with a known distortion: '
          + ', '.join(f'{s} {n}' for s, n in jobs['sensor'].value_counts().items()) + f' -> {png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
