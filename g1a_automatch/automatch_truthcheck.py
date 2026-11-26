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

    python automatch_job.py truthcheck <run folder | folder of runs> --truth pts.csv [--redo-final]

--redo-final (an updated truth file): the final points are picked again from
the saved matches with the new ranking (redo_final), before everything is
scored; nothing is matched again.
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


def redo_final(run: str, truth_csv: str, res: Dict, crs: str) -> List[str]:
    """The final points again with this truth, from the saved matches (nothing
    matched again): RIVAL_BEST follows the new ranking, the cross-check picks its
    top methods by it and writes the consistent points, the final and rigid
    picks again, then RIVAL_BEST_FINAL. The job's settings (top N, agreeing,
    levels, refined points) are kept; job_used.json then names this truth file
    (the one before is kept as job_used_before_truth.json)."""
    import json
    import shutil
    import automatch_cross as X
    import automatch_job as J
    jp = os.path.join(run, 'job_used.json')
    with open(jp, encoding='utf-8') as f:
        job = json.load(f)
    if os.path.abspath(str(job.get('truth_csv') or '')) != os.path.abspath(truth_csv):
        keep = os.path.join(run, 'job_used_before_truth.json')
        if not os.path.exists(keep):
            shutil.copyfile(jp, keep)
        job['truth_csv'] = truth_csv
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(job, f, indent=2)
    done = J.refresh_after_compare(run, res)
    top = int(job.get('cross_top_n') or 5)
    if top >= 2:
        X.run(run, top, int(job.get('cross_min_agree') or 0), float(job.get('cross_cell_km') or 25.0),
              truth_csv=truth_csv, working_crs=crs, keep=int(job.get('cross_keep_per_cell') or 3),
              max_points=int(500 if job.get('cross_max_points') is None else job['cross_max_points']),
              final_points=int(job.get('final_points') or 0), quiet=True)
        done += X.best_finals(run, crs, int(job.get('final_points') or 0))
    print(f'[TruthCheck] {os.path.basename(os.path.normpath(run))}: final points picked again with '
          f'{os.path.basename(truth_csv)} (top {top} by it; nothing matched again)', flush=True)
    return done


def check_run(run: str, truth_csv: str, radius_m: float = 5000.0, refinal: bool = False) -> List[Dict]:
    import automatch_truth as T
    name = os.path.basename(os.path.normpath(run))
    crs = T.working_crs_of(run)
    if not crs:
        raise ValueError(f'{name}: working CRS unknown (no job_used.json / matches)')
    res = T.compare(run, truth_csv, radius_m, 'consensus', crs, write=True, quiet=True, rebuild=False)
    try:                                         # which truth file this run was last compared with
        with open(os.path.join(run, 'TRUTH_SOURCE.txt'), 'w', encoding='utf-8') as f:
            f.write(os.path.abspath(truth_csv) + '\n')
    except OSError:
        pass
    if refinal:
        redo_final(run, truth_csv, res, crs)
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
    try:                                         # the methods close to the manual points, detector by detector
        import automatch_detplots as DP
        DP.run(run, truth_csv, quiet=True)
    except Exception as e:
        print(f'[TruthCheck] {name}: detector plots not drawn ({type(e).__name__}: {e})')
    try:
        keypoint_plot(run)
    except Exception as e:
        print(f'[TruthCheck] {name}: keypoint test plot not drawn ({type(e).__name__}: {e})')
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


def keypoint_plot(run: str) -> str:
    """A keypoint test (one run with two keypoint counts, win<W>_nfauto and
    win<W>_nf<N>): per detector, its best configuration's truth RMSE, truth
    points reached and seconds per window at each count; KEYPOINT_TEST.png."""
    import re
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rk = os.path.join(run, 'TRUTH_RANKING.csv')
    if not os.path.exists(rk):
        return ''
    t = pd.read_csv(rk)
    sweeps = sorted(t['sweep'].astype(str).unique(), key=lambda v: (0 if v.endswith('nfauto') else 1, v))
    if len(sweeps) < 2:
        return ''
    t = t.sort_values('rank').groupby(['sweep', 'detector']).first().reset_index()
    perf = os.path.join(run, 'PERFORMANCE.csv')
    if os.path.exists(perf):
        p = pd.read_csv(perf).groupby(['sweep', 'detector'])['sec_per_window'].median().rename('spw_perf')
        t = t.merge(p.reset_index(), on=['sweep', 'detector'], how='left')
        t['sec_per_window'] = t['spw_perf'].fillna(t.get('sec_per_window'))
    a, b = sweeps[0], sweeps[1]
    w = t.pivot_table(index='detector', columns='sweep', values=['truth_rmse_m', 'truth_reached', 'sec_per_window'])
    w = w.dropna(subset=[('truth_rmse_m', a), ('truth_rmse_m', b)])
    if w.empty:
        return ''
    w = w.sort_values(('truth_rmse_m', a))
    nb = re.search(r'nf(\d+)', b)
    lab_a, lab_b = 'usual density', (f'{nb.group(1)} keypoints per window' if nb else b)
    ca, cb = '#2a78d6', '#eb6834'
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.6 + 0.36 * len(w)), sharey=True,
                             gridspec_kw={'width_ratios': [1.4, 1, 1]})
    y = range(len(w))
    for ax, col, title, log in ((axes[0], 'truth_rmse_m', 'truth RMSE (m, log)', True),
                                (axes[1], 'truth_reached', 'truth points reached', False),
                                (axes[2], 'sec_per_window', 'seconds per window', False)):
        va, vb = w[(col, a)].to_numpy(float), w[(col, b)].to_numpy(float)
        for i in y:
            ax.plot([va[i], vb[i]], [i, i], color='#cfcdc6', lw=2, zorder=1)
        ax.scatter(va, list(y), s=46, color=ca, zorder=3, label=lab_a, edgecolors='white', linewidths=0.8)
        ax.scatter(vb, list(y), s=46, color=cb, zorder=3, label=lab_b, edgecolors='white', linewidths=0.8)
        if log:
            ax.set_xscale('log')
            for i in y:
                r = vb[i] / va[i] if va[i] else float('nan')
                ax.text(max(va[i], vb[i]) * 1.12, i, f'{r:.2f}x', va='center', fontsize=8, color='#6b6b66')
        ax.set_title(title, fontsize=10, loc='left', color='#1f1f1d')
        ax.grid(axis='x', color='#e7e6e2', lw=0.8, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(w.index, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=8.5, frameon=False)
    med = (w[('truth_rmse_m', b)] / w[('truth_rmse_m', a)]).median()
    spd = (w[('sec_per_window', b)] / w[('sec_per_window', a)]).median()
    fig.suptitle(f'Keypoint test: {lab_b} against the usual density -- median error x{med:.2f}, time per window '
                 f'x{spd:.2f}', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    png = os.path.join(run, 'KEYPOINT_TEST.png')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[KeypointTest] {len(w)} detector(s): truth error x{med:.2f}, time per window x{spd:.2f} -> {png}',
          flush=True)
    return png


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


def _truth_key(name: str) -> str:
    """A truth file's name as the product it is for: 'pts_267403451_band4.csv' -> '267403451'."""
    import re
    k = os.path.splitext(os.path.basename(name))[0].lower()
    k = re.sub(r'^(pts|gt|truth|gcp|manual)[_-]+', '', k)
    k = re.sub(r'[_-]+(pts|gt|truth|gcp|manual)$', '', k)
    return re.sub(r'[_-]+b(and)?\d+$', '', k)


class TruthFolder:
    """The truth files of one folder (and below), found for a job by its input's
    file name or one of its folders (e.g. the request ID); the job's band first."""

    def __init__(self, folder: str):
        self.by_key: Dict[str, List[str]] = {}
        for d, _, files in os.walk(folder or ''):
            for f in files:
                if f.lower().endswith(('.csv', '.txt')):
                    self.by_key.setdefault(_truth_key(f), []).append(os.path.join(d, f))

    def find(self, input_path: str, bands: List[str]) -> str:
        import re
        p = str(input_path or '').replace('\\', '/')
        keys = [os.path.splitext(os.path.basename(p))[0].lower()]
        keys += [x.lower() for x in p.split('/')[-4:-1] if x and x.lower() not in ('input', 'inputs', 'data')][::-1]
        for k in keys:
            got = self.by_key.get(_truth_key(k)) or self.by_key.get(k)
            if got:
                same = [g for g in got for b in bands if re.search(rf'[_-]{b}(\D|$)', os.path.basename(g).lower())]
                return sorted(same or got)[0]
        return ''


def truth_for(run: str, truth: str = '', folder: Optional[TruthFolder] = None) -> str:
    """The truth file of a run: the one given, else its own in the truth folder
    (by its input's name), else the one its job names (if it is here)."""
    import json
    if truth:
        return truth
    try:
        with open(os.path.join(run, 'job_used.json'), encoding='utf-8') as f:
            j = json.load(f)
    except (OSError, ValueError):
        return ''
    if folder is not None:
        got = folder.find(j.get('input_path') or '', [str(c).lower() for c in (j.get('channels') or [])])
        if got:
            return got
    own = str(j.get('truth_csv') or '')
    return own if own and os.path.exists(own) else ''


def changed(run: str, truth_csv: str) -> bool:
    """The run has not been compared with this truth file as it is now: no
    ranking, another truth file named, or the file is newer than the ranking."""
    import json
    rk = os.path.join(run, 'TRUTH_RANKING.csv')
    if not os.path.exists(rk):
        return True
    named = ''
    try:
        with open(os.path.join(run, 'TRUTH_SOURCE.txt'), encoding='utf-8') as f:
            named = f.read().strip()
    except OSError:
        try:
            with open(os.path.join(run, 'job_used.json'), encoding='utf-8') as f:
                named = str(json.load(f).get('truth_csv') or '')
        except (OSError, ValueError):
            pass
    if os.path.basename(named).lower() != os.path.basename(truth_csv).lower():
        return True
    return os.path.getmtime(truth_csv) > os.path.getmtime(rk)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Compare finished runs with truth files, without running them again.')
    ap.add_argument('folders', nargs='+', help='run folders, or folders of runs (server outputs through the share, '
                                               'workstation outputs, auto steps, tiers)')
    ap.add_argument('--truth', default='', help='one truth file for every run (else --truth-folder)')
    ap.add_argument('--truth-folder', default='',
                    help="the truth files, named after each input (its file name or one of its folders, e.g. the "
                         "request ID); a run without one there uses the truth file its job names")
    ap.add_argument('--all', action='store_true',
                    help='every run with a truth file (default: only those whose truth file changed since their '
                         'last comparison, or that were never compared)')
    ap.add_argument('--radius-km', type=float, default=5.0)
    ap.add_argument('--redo-final', action='store_true',
                    help='also pick the final points again with the truth (an updated truth file): the cross-check '
                         'from the saved matches, its top methods by the new ranking; nothing matched again')
    ap.add_argument('--share-local', default='', help="where the server's share is mounted here, e.g. V:\\")
    ap.add_argument('--share-server', default='', help='the same share on the server, e.g. /maintenance/')
    a = ap.parse_args(argv)
    import pandas as pd
    import automatch_cross as X
    X.SHARE = (a.share_local, a.share_server)
    if not a.truth and not a.truth_folder:
        ap.error('give --truth or --truth-folder')
    tf = TruthFolder(a.truth_folder) if a.truth_folder else None
    rows, done, skipped = [], [], []
    for folder in a.folders:
        runs = run_folders(folder)
        if not runs:
            print(f'[TruthCheck] no finished run (job_used.json) in {folder}')
        frows = []
        for r in runs:
            name = os.path.basename(os.path.normpath(r))
            t = truth_for(r, a.truth, tf)
            if not t:
                skipped.append(f'{name} (no truth file)')
                continue
            if not a.all and not a.truth and not changed(r, t):
                skipped.append(f'{name} (truth unchanged)')
                continue
            print(f'[TruthCheck] {name}: {os.path.basename(t)}', flush=True)
            try:
                frows += check_run(r, t, a.radius_km * 1000.0, a.redo_final)
                done.append(name)
            except Exception as e:
                print(f'[TruthCheck] {name}: not compared ({type(e).__name__}: {e})', flush=True)
        if frows:
            df = pd.DataFrame(frows)
            df.to_csv(os.path.join(folder, 'TRUTH_CHECK.csv'), index=False)
            png = plot(df, os.path.join(folder, 'TRUTH_CHECK.png')) if df['run'].nunique() <= 12 else ''
            print(f'[TruthCheck] {folder}: {df["run"].nunique()} run(s) -> TRUTH_CHECK.csv' + (f', {png}' if png else ''),
                  flush=True)
        rows += frows
    if skipped:
        print(f'[TruthCheck] left alone: {len(skipped)} run(s): ' + ', '.join(skipped[:20])
              + (' ...' if len(skipped) > 20 else ''), flush=True)
    print(f'[TruthCheck] {len(done)} run(s) compared' + (' (final points picked again)' if a.redo_final else ''),
          flush=True)
    return 0 if done or not rows else 1

if __name__ == '__main__':
    sys.exit(main())
