"""
Compare finished runs with a truth file, without running them again.

A run made without a truth file (e.g. the run window's auto mode) is scored
afterwards: rerunning it with the truth file would process it again, because
the truth file also sets the job (window size from its largest error, the
search range, the band), so the results would not be the same ones.

For each run folder (a folder with job_used.json; given a parent, e.g. the
run window's output with auto_top3_agree2/, auto_top5_agree3/, ... or
fastest/, fast/, ..., every run folder in it):
  - TRUTH_RANKING.csv etc.: every detector + matcher against the truth
    (automatch_truth.compare; the run's own matches, nothing matched again);
  - each final point set the run made (RIVAL_FINAL_*, its rigid variant,
    RIVAL_BEST_FINAL_*, RIVAL_CONSISTENT_*) scored at the truth points: truth
    RMSE and how many truth points it reaches;
  - REPORT/ with the manual points' row (automatch_report).
TRUTH_CHECK.csv in the folder given: one row per run and point set, and
TRUTH_CHECK.png: the point sets' truth RMSE per run.

    python automatch_job.py truthcheck <run folder | folder of runs> --truth pts.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional


def run_folders(folder: str) -> List[str]:
    """folder itself when it is a run, else the runs in it (one level, then two)."""
    if os.path.exists(os.path.join(folder, 'job_used.json')):
        return [folder]
    for pat in ('*', os.path.join('*', '*')):
        found = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(folder, pat, 'job_used.json')))
        if found:
            return found
    return []


def point_sets(run: str) -> List[str]:
    """The final point sets a run wrote (not the _detail files)."""
    pats = ('RIVAL_FINAL_*.csv', 'RIVAL_BEST_FINAL_*.csv', 'RIVAL_CONSISTENT_*.csv')
    out = []
    for p in pats:
        out += [f for f in sorted(glob.glob(os.path.join(run, p))) if not f.endswith('_detail.csv')]
    return out


def score_file(path: str, truth, radius_m: float) -> Dict:
    import automatch_rival as R
    import automatch_truth as T
    p = R.read_rival_csv(path)
    if p.empty:
        return {'points': 0}
    pts = p.assign(E=p['ix'], N=p['iy'], dE=p['ix'] - p['rx'], dN=p['iy'] - p['ry'], chip=os.path.basename(path))
    s, _ = T.score_config({'points': pts[['E', 'N', 'dE', 'dN', 'chip']]}, truth, radius_m)
    return {'points': len(p), 'truth_rmse_m': s.get('truth_rmse_m'), 'truth_reached': s.get('truth_reached'),
            'truth_total': s.get('truth_total')}


def check_run(run: str, truth_csv: str, radius_m: float = 5000.0) -> List[Dict]:
    import automatch_truth as T
    name = os.path.basename(os.path.normpath(run))
    crs = T.working_crs_of(run)
    if not crs:
        raise ValueError(f'{name}: working CRS unknown (no job_used.json / matches)')
    res = T.compare(run, truth_csv, radius_m, 'consensus', crs, write=True, quiet=True, rebuild=False)
    rows = []
    rk = res.get('ranking')
    if rk is not None and len(rk):
        b = rk.sort_values('rank').iloc[0]
        rows.append({'run': name, 'set': f"best single: {b['detector']} + {b['matcher']}",
                     'truth_rmse_m': float(b['truth_rmse_m']), 'truth_reached': int(b['truth_reached']),
                     'truth_total': int(b['truth_total'])})
    truth = T.truth_in_crs(T.load_truth(truth_csv), crs)
    for f in point_sets(run):
        try:
            rows.append({'run': name, 'set': os.path.basename(f)[:-4], **score_file(f, truth, radius_m)})
        except Exception as e:
            print(f'[TruthCheck] {name}: {os.path.basename(f)} not scored ({type(e).__name__}: {e})')
    try:
        import automatch_report as RP
        RP.report(run, quiet=True, truth_csv=truth_csv)
    except Exception as e:
        print(f'[TruthCheck] {name}: REPORT not written ({type(e).__name__}: {e})')
    for r in rows:
        rm = r.get('truth_rmse_m')
        print(f"[TruthCheck] {name}: {r['set']}: "
              + (f"truth RMSE {rm:.0f} m at {r.get('truth_reached')}/{r.get('truth_total')}"
                 if rm == rm and rm is not None else 'no truth point reached'), flush=True)
    return rows


def plot(df, png: str) -> str:
    """Per run, its point sets' truth RMSE (horizontal bars, log)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = df.dropna(subset=['truth_rmse_m'])
    if d.empty:
        return ''
    runs = list(dict.fromkeys(d['run']))
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.0 + 0.32 * len(d) + 0.7 * len(runs)), squeeze=False,
                             gridspec_kw={'height_ratios': [max(2, (d['run'] == r).sum()) for r in runs]})
    for ax, r in zip(axes[:, 0], runs):
        u = d[d['run'] == r].sort_values('truth_rmse_m')
        cols = ['#eb6834' if s.startswith('best single') else '#2a78d6' for s in u['set']]
        ax.barh(range(len(u)), u['truth_rmse_m'], color=cols, height=0.62)
        for i, (_, x) in enumerate(u.iterrows()):
            ax.text(x['truth_rmse_m'] * 1.05, i, f"{x['truth_rmse_m']:.0f} m  ({int(x['truth_reached'])}/"
                                                f"{int(x['truth_total'])})", va='center', fontsize=8, color='#6b6b66')
        ax.set_yticks(range(len(u)))
        ax.set_yticklabels(u['set'], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlim(right=float(u['truth_rmse_m'].max()) * 4)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_title(r, loc='left', fontsize=11, fontweight='bold')
    axes[-1, 0].set_xlabel('truth RMSE (m, log); orange: the best single detector + matcher')
    fig.suptitle('Finished runs against the truth file', x=0.01, ha='left', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Compare finished runs with a truth file, without running them again.')
    ap.add_argument('folder', help='a run folder, or a folder of runs (auto steps, tiers, jobs)')
    ap.add_argument('--truth', required=True, help='the truth (manual points) CSV')
    ap.add_argument('--radius-km', type=float, default=5.0)
    ap.add_argument('--share-local', default='', help="where the server's share is mounted here, e.g. V:\\")
    ap.add_argument('--share-server', default='', help='the same share on the server, e.g. /maintenance/')
    a = ap.parse_args(argv)
    import pandas as pd
    import automatch_cross as X
    X.SHARE = (a.share_local, a.share_server)
    runs = run_folders(a.folder)
    if not runs:
        print(f'[TruthCheck] no finished run (job_used.json) in {a.folder}')
        return 1
    rows = []
    for r in runs:
        try:
            rows += check_run(r, a.truth, a.radius_km * 1000.0)
        except Exception as e:
            print(f'[TruthCheck] {os.path.basename(r)}: not compared ({type(e).__name__}: {e})', flush=True)
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.folder, 'TRUTH_CHECK.csv'), index=False)
    png = plot(df, os.path.join(a.folder, 'TRUTH_CHECK.png'))
    print(f'[TruthCheck] {len(runs)} run(s) compared -> {os.path.join(a.folder, "TRUTH_CHECK.csv")}'
          + (f', {png}' if png else ''), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
