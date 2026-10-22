"""
Cross-check between detectors: keep the matches that several of the best
detector + matcher configurations agree on.

A single configuration can be consistently wrong (many matches on the wrong
ground pass its own RANSAC and chip consensus). Where several of the best
configurations measure the same offset at the same place, the offset is far
more likely right. So:

1. the top N configurations of each channel are chosen, one per detector:
   by the ground-truth ranking when there is one (TRUTH_RANKING.csv), else
   by how well each agrees with all the others (no truth needed);
2. the scene is cut into cells (cross_cell_km); in each cell every top
   configuration with matches gives its median offset, and the cell counts
   when at least cross_min_agree of them lie within 1.5 x the RANSAC
   threshold (3 px by default) of their common median;
3. the matches of the agreeing configurations in those cells, each within
   the tolerance of that median, are written as RIVAL_CROSS_<channel>.csv
   (+ _detail.csv naming the detector of every point), and CROSS_SUMMARY.csv
   lists the configurations used, the cells and, with ground truth, how the
   cross-checked points score against it.

    python automatch_job.py cross <output_dir> [--top 5] [--min-agree 3] [--cell-km 25] [--truth pts.csv]
    python automatch_job.py cross <output_dir> --from-truth-points     # a finished job, no match files
    python automatch_job.py cross <folder of job outputs> --batch      # every job, results in CROSS_ALL.csv
"""
from __future__ import annotations

import argparse
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

MAX_POINTS_PER_CELL = 200      # per configuration and cell, in the RIVAL file
KEEP_PER_CELL = 3              # agreeing configurations kept per cell: the least distorted there
OUTLIER_K = 3.0                # points further than this many x the cell's own scatter are dropped
RELIABLE_SHARE = 0.15          # share of cells that must pass for a job to count as reliable
# (CROSS_ALL of 34 jobs: every km-off job kept 0-4 % of its cells, the good ones 17-85 %)


def _spread(p: pd.DataFrame) -> float:
    """Robust scatter (m) of one configuration's offsets in one cell: how
    much its matches there disagree among themselves (local distortion,
    noise, mixed wrong matches). 1.4826 x the median distance to its median."""
    if len(p) < 3:
        return float('inf')
    d = np.hypot(p['dE'] - p['dE'].median(), p['dN'] - p['dN'].median())
    return float(1.4826 * np.median(d))


def _configs(output_dir: str, chips: str = 'consensus') -> List[Dict]:
    """Every configuration of every run under output_dir, with its matches."""
    import automatch_rival as R
    import automatch_truth as T
    out = []
    settings = T.consensus_settings(output_dir)
    for run in T.find_runs(output_dir):
        for key, cfg in T.config_points(run['filtered_dir'], run['final_dir'], chips, settings).items():
            pts = R.collect_points(run['filtered_dir'], cfg['files'])
            if pts.empty:
                continue
            pts = pts.assign(E=pts['x1-map'], N=pts['y1-map'], dE=pts['x1-map'] - pts['x2-map'],
                             dN=pts['y1-map'] - pts['y2-map'])
            out.append({'channel': str(cfg['channel']), 'sweep': run['sweep'], 'detector': cfg['detector'],
                        'matcher': cfg['matcher'], 'ransac': cfg['ransac'], 'points': pts})
    return out


def _cell_medians(pts: pd.DataFrame, cell_m: float) -> Dict[Tuple[int, int], Tuple[float, float, int]]:
    ij = np.floor(pts[['E', 'N']].to_numpy() / cell_m).astype(int)
    g = pts.assign(_i=ij[:, 0], _j=ij[:, 1]).groupby(['_i', '_j'])
    return {k: (float(v['dE'].median()), float(v['dN'].median()), len(v)) for k, v in g}


def _agreement_scores(cfgs: List[Dict], cell_m: float, tol_m: float) -> List[float]:
    """Without ground truth: each configuration's share of cells (weighted by
    how many cells it covers) where its median offset lies within tol_m of the
    median of all configurations there."""
    meds = [_cell_medians(c['points'], cell_m) for c in cfgs]
    allc: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for m in meds:
        for k, (de, dn, _) in m.items():
            allc.setdefault(k, []).append((de, dn))
    centre = {k: (float(np.median([v[0] for v in vs])), float(np.median([v[1] for v in vs])))
              for k, vs in allc.items() if len(vs) >= 2}
    scores = []
    for m in meds:
        hits = sum(1 for k, (de, dn, _) in m.items()
                   if k in centre and math.hypot(de - centre[k][0], dn - centre[k][1]) <= tol_m)
        scores.append(hits * hits / max(1, len(m)))    # agreeing cells x share agreeing
    return scores


def pick_top(cfgs: List[Dict], top_n: int, cell_m: float, tol_m: float,
             ranking: Optional[pd.DataFrame] = None) -> Tuple[List[Dict], str]:
    """(the top_n configurations, one per detector, how they were chosen)."""
    if ranking is not None and not ranking.empty and 'rank' in ranking.columns:
        order = {}
        for r in ranking.sort_values('rank').itertuples():
            order.setdefault((str(r.channel), str(r.sweep), str(r.detector), str(r.matcher), str(r.ransac)),
                             int(r.rank))
        scored = [(order.get((c['channel'], c['sweep'], c['detector'], c['matcher'], c['ransac'])), c)
                  for c in cfgs]
        scored = sorted([s for s in scored if s[0] is not None], key=lambda s: s[0])
        how = 'ground-truth ranking'
        ranked = [c for _, c in scored]
    else:
        sc = _agreement_scores(cfgs, cell_m, tol_m)
        ranked = [c for _, c in sorted(zip(sc, cfgs), key=lambda s: -s[0])]
        how = 'agreement with the other configurations'
    top, seen = [], set()
    for c in ranked:
        if c['detector'] in seen:
            continue
        seen.add(c['detector'])
        top.append(c)
        if len(top) >= top_n:
            break
    return top, how


def cross_check(top: List[Dict], cell_m: float, tol_m: float, min_agree: int,
                keep: int = KEEP_PER_CELL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(points kept, one row per cell). A cell is kept when min_agree of the
    top configurations with matches there lie within tol_m of their median.
    Of those, the keep with the least scatter in the cell (_spread: the least
    local distortion) give the points, within tol_m of their own median; of
    these, points further than OUTLIER_K x the cell's own scatter from the
    cell's centre are dropped (distortion differs from cell to cell, so each
    cell is judged by itself). The cell row reports the kept spreads, how far
    apart their medians are, the cell's scatter and the points dropped."""
    meds = [_cell_medians(c['points'], cell_m) for c in top]
    cells = sorted(set().union(*[set(m) for m in meds])) if meds else []
    kept, rows = [], []
    for k in cells:
        have = [(i, m[k]) for i, m in enumerate(meds) if k in m]
        if len(have) < 2:
            rows.append({'cell_i': k[0], 'cell_j': k[1], 'configs_with_matches': len(have), 'agreeing': 0,
                         'kept': False})
            continue
        cde = float(np.median([v[0] for _, v in have]))
        cdn = float(np.median([v[1] for _, v in have]))
        agree = [i for i, v in have if math.hypot(v[0] - cde, v[1] - cdn) <= tol_m]
        ok = len(agree) >= min_agree
        rows.append({'cell_i': k[0], 'cell_j': k[1], 'cell_E': (k[0] + 0.5) * cell_m,
                     'cell_N': (k[1] + 0.5) * cell_m, 'configs_with_matches': len(have),
                     'agreeing': len(agree), 'median_dE_m': cde, 'median_dN_m': cdn, 'kept': ok,
                     'detectors': ', '.join(top[i]['detector'] for i in agree)})
        if not ok:
            continue
        inside = {}
        for i in agree:
            p = top[i]['points']
            ij = np.floor(p[['E', 'N']].to_numpy() / cell_m).astype(int)
            inside[i] = p[(ij[:, 0] == k[0]) & (ij[:, 1] == k[1])]
        spread = {i: _spread(inside[i]) for i in agree}
        best = sorted(agree, key=lambda i: spread[i])[:max(1, keep)]
        bm = [meds[i][k] for i in best]
        cde = float(np.median([v[0] for v in bm]))
        cdn = float(np.median([v[1] for v in bm]))
        gap = max((math.hypot(a[0] - b[0], a[1] - b[1]) for a in bm for b in bm), default=0.0)
        rows[-1].update({'kept_detectors': ', '.join(top[i]['detector'] for i in best),
                         'kept_spreads_m': ', '.join(f'{spread[i]:.0f}' for i in best),
                         'kept_max_gap_m': round(gap, 1), 'median_dE_m': cde, 'median_dN_m': cdn})
        cell_pts = []
        for i in best:
            sel = inside[i]
            sel = sel[np.hypot(sel['dE'] - cde, sel['dN'] - cdn) <= tol_m]
            cell_pts.append(sel.assign(cross_detector=top[i]['detector'], cross_matcher=top[i]['matcher'],
                                       cross_sweep=top[i]['sweep']))
        pool = pd.concat(cell_pts, ignore_index=True)
        # points that stray from this cell's own centre, judged by this cell's own
        # scatter (distortion differs from cell to cell)
        n0 = len(pool)
        if n0 >= 5:
            mde, mdn = float(pool['dE'].median()), float(pool['dN'].median())
            dist = np.hypot(pool['dE'] - mde, pool['dN'] - mdn)
            sig = max(float(1.4826 * np.median(dist)), 1.0)
            pool = pool[dist <= OUTLIER_K * sig]
            rows[-1].update({'cell_scatter_m': round(sig, 1), 'outliers_dropped': int(n0 - len(pool))})
        for _, sel in pool.groupby('cross_detector', sort=False):
            if len(sel) > MAX_POINTS_PER_CELL:
                sel = sel.sample(MAX_POINTS_PER_CELL, random_state=0)
            kept.append(sel)
    pts = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    return pts, pd.DataFrame(rows)


def run(output_dir: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
        tol_m: Optional[float] = None, truth_csv: str = '', working_crs: Optional[str] = None,
        cfgs: Optional[List[Dict]] = None, quiet: bool = False, keep: int = KEEP_PER_CELL) -> Dict[str, Dict]:
    """Cross-check every channel of an output folder; returns per channel
    {'csv', 'summary', 'configs', 'cells', 'points', 'truth_rmse_m'}."""
    import automatch_rival as R
    import automatch_truth as T
    if top_n < 2:
        return {}
    crs = working_crs or T.working_crs_of(output_dir)
    if not crs:
        raise ValueError('working CRS unknown: pass working_crs')
    try:
        settings = T.consensus_settings(output_dir) or {}
    except Exception:                      # a folder copied off the server: input paths do not resolve here
        settings = {}
    # not the chip-consensus tolerance: jobs set that loosely (job 364: 31.5 km), and
    # then nothing was filtered; 1.5 x the RANSAC threshold (3 px by default) instead
    tol = float(tol_m or 0.0)
    cell_m = float(cell_km) * 1000.0
    cfgs = cfgs if cfgs is not None else _configs(output_dir)
    rank_path = os.path.join(output_dir, 'TRUTH_RANKING.csv')
    ranking = pd.read_csv(rank_path) if os.path.exists(rank_path) else None
    truth_csv = truth_csv or job_truth_csv(output_dir)
    truth = T.truth_in_crs(T.load_truth(truth_csv), crs) if truth_csv else None
    need = min_agree if min_agree > 0 else max(2, top_n // 2 + 1)
    out: Dict[str, Dict] = {}
    summary_rows = []
    for ch in sorted({c['channel'] for c in cfgs}):
        mine = [c for c in cfgs if c['channel'] == ch]
        rk = ranking[ranking['channel'].astype(str) == ch] if ranking is not None else None
        if not tol:                        # 1.5 x the RANSAC threshold (3 px with the default 2 px)
            th = pd.Series([c['ransac'] for c in mine]).str.extract(r'(\d+(?:\.\d+)?)$')[0].astype(float)
            tol = 1.5 * float(th.median()) if th.notna().any() else 1000.0
        top, how = pick_top(mine, top_n, cell_m, tol, rk)
        if len(top) < 2:
            print(f'[Cross] {ch}: fewer than 2 configurations to compare; skipped')
            continue
        req = min(need, len(top))
        pts, cells = cross_check(top, cell_m, tol, req, keep)
        kept = int(cells['kept'].sum()) if not cells.empty else 0
        n_cells = int((cells['configs_with_matches'] >= 2).sum()) if not cells.empty else 0
        share = kept / n_cells if n_cells else 0.0
        res = {'channel': ch, 'chosen_by': how, 'top_n': len(top), 'min_agree': req, 'keep_per_cell': keep,
               'cell_km': cell_km, 'tolerance_m': tol, 'cells_with_matches': n_cells, 'cells_kept': kept,
               'kept_share': round(share, 3), 'reliable': bool(share >= RELIABLE_SHARE and len(pts) > 0),
               'points': int(len(pts)),
               'configs': ' | '.join(f"{c['detector']} {c['matcher']} {c['ransac']} {c['sweep']}" for c in top)}
        if len(pts):
            res['mean_dE_m'] = float(pts['dE'].mean())
            res['mean_dN_m'] = float(pts['dN'].mean())
            path = os.path.join(output_dir, f'RIVAL_CROSS_{ch}.csv')
            R.write_rival_csv(pts, crs, path, path[:-4] + '_detail.csv')
            res['csv'] = path
            if truth is not None:
                s, _ = T.score_config({'points': pts.assign(chip=pts['source_file'])[['E', 'N', 'dE', 'dN', 'chip']]},
                                      truth, 5000.0)
                res.update({'truth_rmse_m': s['truth_rmse_m'], 'truth_reached': s['truth_reached'],
                            'truth_total': s['truth_total']})
        cells.to_csv(os.path.join(output_dir, f'CROSS_CELLS_{ch}.csv'), index=False)
        summary_rows.append(res)
        out[ch] = res
        if not quiet:
            msg = (f"[Cross] {ch}: top {len(top)} by {how}; {kept} of {n_cells} cells ({100 * share:.0f} %) where "
                   f"{req}+ agree within {tol:.0f} m, the {keep} least distorted kept; {len(pts)} points; "
                   + ('reliable' if res['reliable'] else f"NOT reliable (under {100 * RELIABLE_SHARE:.0f} % of cells agree)"))
            if res.get('truth_rmse_m') is not None:
                msg += f"; truth RMSE {res['truth_rmse_m']:.0f} m at {res['truth_reached']}/{res['truth_total']}"
            print(msg + (f" -> {res['csv']}" if res.get('csv') else ''))
            for c in top:
                print(f"         {c['detector']} + {c['matcher']} {c['ransac']} ({c['sweep']})")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, 'CROSS_SUMMARY.csv'), index=False)
    return out


def from_truth_points(points_csv: str, ranking_csv: str, top_n: int = 5, min_agree: int = 0,
                      tol_m: Optional[float] = None) -> pd.DataFrame:
    """The cross-check replayed on a finished job's TRUTH_POINTS.csv and
    TRUTH_RANKING.csv alone (no match files): at each truth point the top_n
    configurations (one per detector, in the ranking's order) give their
    estimate; when min_agree of them lie within tol_m of their median, that
    median is the cross-checked estimate there. One row per channel: the
    cross-checked RMSE and points reached, beside the best single
    configuration's. tol_m defaults to 1.5 x the RANSAC threshold (3 px with
    the default 2 px threshold)."""
    pts = pd.read_csv(points_csv)
    rk = pd.read_csv(ranking_csv).sort_values('rank')
    key = ['channel', 'sweep', 'detector', 'matcher', 'ransac']
    rows = []
    for ch, g in rk.groupby('channel', sort=False):
        top, seen = [], set()
        for r in g.itertuples():
            if r.detector in seen:
                continue
            seen.add(r.detector)
            top.append(r)
            if len(top) >= top_n:
                break
        if len(top) < 2:
            continue
        need = min(min_agree if min_agree > 0 else max(2, top_n // 2 + 1), len(top))
        if tol_m:
            tol = float(tol_m)
        else:
            m = pd.Series([str(t.ransac) for t in top]).str.extract(r'(\d+(?:\.\d+)?)$')[0].astype(float)
            tol = 1.5 * float(m.median()) if m.notna().any() else 1000.0
        sel = pd.concat([pts[(pts[key[0]].astype(str) == str(t.channel)) & (pts['sweep'] == t.sweep)
                             & (pts['detector'] == t.detector) & (pts['matcher'].astype(str) == str(t.matcher))
                             & (pts['ransac'].astype(str) == str(t.ransac))] for t in top])
        sel = sel[sel['method'].isin(['local', 'surface']) & sel['tool_dE_m'].notna()]
        diffs, n_pts = [], int(pts.loc[pts['channel'].astype(str) == str(ch), 'truth_id'].nunique())
        for tid, q in sel.groupby('truth_id'):
            if len(q) < need:
                continue
            mde, mdn = float(q['tool_dE_m'].median()), float(q['tool_dN_m'].median())
            ok = q[np.hypot(q['tool_dE_m'] - mde, q['tool_dN_m'] - mdn) <= tol]
            if len(ok) < need:
                continue
            de, dn = float(ok['tool_dE_m'].median()), float(ok['tool_dN_m'].median())
            diffs.append((de - float(q['truth_dE_m'].iloc[0]), dn - float(q['truth_dN_m'].iloc[0])))
        d = np.asarray(diffs, float).reshape(-1, 2)
        best = g.iloc[0]
        rows.append({'channel': ch, 'top_n': len(top), 'min_agree': need, 'tolerance_m': tol,
                     'cross_rmse_m': float(np.sqrt((d ** 2).sum(1).mean())) if len(d) else None,
                     'cross_reached': len(d), 'truth_total': n_pts,
                     'best_single': f"{best['detector']} + {best['matcher']} {best['sweep']}",
                     'best_single_rmse_m': best.get('truth_rmse_m'), 'best_single_reached': best.get('truth_reached'),
                     'configs': ' | '.join(f'{t.detector} {t.matcher} {t.sweep}' for t in top)})
    return pd.DataFrame(rows)


def parse_jobs(text: str) -> Optional[set]:
    """'335-363, 370' -> {335, ..., 363, 370}; '' -> None (every job)."""
    if not text or not str(text).strip():
        return None
    out = set()
    for part in re.split(r'[,;\s]+', str(text).strip()):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', part)
        if m:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f'jobs: {part!r} is not a number or a range (e.g. 335-363, 370)')
    return out


def job_state(d: str) -> str:
    """'finished' (an automatch run that ended: job_used.json, RUN_MANIFEST.csv
    and '[Job] done' at the end of automatch.log), 'compare' (a compare
    job's TRUTH_RANKING.csv + TRUTH_POINTS.csv), 'running' (an automatch run
    still going), or '' (not an automatch output: another user's or tool's)."""
    has = lambda n: os.path.exists(os.path.join(d, n))
    if has('job_used.json') and (has('automatch.log') or has('RUN_MANIFEST.csv')):
        done = has('RUN_MANIFEST.csv')
        if done and has('automatch.log'):
            try:
                with open(os.path.join(d, 'automatch.log'), 'rb') as f:
                    f.seek(max(0, os.path.getsize(os.path.join(d, 'automatch.log')) - 8192))
                    done = b'[Job] done' in f.read()
            except OSError:
                pass
        return 'finished' if done else 'running'
    if has('TRUTH_RANKING.csv') and has('TRUTH_POINTS.csv'):
        return 'compare'
    return ''


def find_jobs(parent: str, jobs: Optional[set] = None) -> Tuple[List[str], Dict[str, str]]:
    """(finished automatch job folders, {skipped folder: why}) in parent and
    one level below it (output/output359), optionally only those whose
    number (the last digits of the name) is in jobs."""
    cands = [parent] + sorted(os.path.join(parent, n) for n in os.listdir(parent)
                              if os.path.isdir(os.path.join(parent, n)))
    found, skipped = [], {}
    for d in cands:
        name = os.path.basename(os.path.normpath(d))
        if jobs is not None and d != parent:
            m = re.search(r'(\d+)\D*$', name)
            if not m or int(m.group(1)) not in jobs:
                continue
        st = job_state(d)
        if st in ('finished', 'compare'):
            found.append(d)
        elif st == 'running':
            skipped[name] = 'still running'
    return found, skipped


def local_path(server_path: str, job_dir: str, server_job_dir: str) -> str:
    """A path written on the server (/maintenance/...) as this machine sees
    it: the server's output folder for the job and the folder it is opened
    from here (V:\\ICIGDev\\GPUPOC\\output\\output359) share their last parts; what
    differs in front is the mount, swapped in."""
    if not server_path or os.path.exists(server_path):
        return server_path
    a = [x for x in re.split(r'[\\/]+', str(server_job_dir).rstrip('/\\')) if x]
    b = [x for x in re.split(r'[\\/]+', os.path.abspath(job_dir).rstrip('/\\')) if x]
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k].lower() == b[-1 - k].lower():
        k += 1
    if k == 0:
        return server_path
    srv_prefix = '/'.join(a[:len(a) - k])
    loc_prefix = os.path.abspath(job_dir)
    for _ in range(k):
        loc_prefix = os.path.dirname(loc_prefix)
    sp = str(server_path).replace('\\', '/').lstrip('/')
    if not sp.lower().startswith(srv_prefix.lower()):
        return server_path
    rest = [x for x in sp[len(srv_prefix):].split('/') if x]
    return os.path.join(loc_prefix, *rest)


SHARE: Tuple[str, str] = ('', '')     # (local, server) mount, e.g. ('V:\\', '/maintenance/'), as Pack uses


def _by_share(server_path: str) -> str:
    loc, srv = SHARE
    if not loc or not srv:
        return server_path
    sp = str(server_path).replace('\\', '/')
    srv_n = srv.replace('\\', '/').rstrip('/') + '/'
    if not sp.lower().startswith(srv_n.lower()):
        return server_path
    return os.path.join(loc, *[x for x in sp[len(srv_n):].split('/') if x])


def job_truth_csv(job_dir: str) -> str:
    """The ground-truth CSV the job was run with (truth_csv in its
    job_used.json), as this machine sees it; '' when it has none or the file
    cannot be reached from here."""
    import json
    p = os.path.join(job_dir, 'job_used.json')
    if not os.path.exists(p):
        return ''
    try:
        with open(p, encoding='utf-8') as f:
            job = json.load(f)
    except (OSError, ValueError):
        return ''
    t = job.get('truth_csv') or ''
    if not t:
        return ''
    for loc in (local_path(t, job_dir, job.get('output_dir') or ''), _by_share(t)):
        if loc and os.path.exists(loc):
            return loc
    print(f'[Cross] {os.path.basename(job_dir)}: its ground truth {t} is not reachable from here '
          f'(set the share mapping, e.g. V:\\ = /maintenance/)')
    return ''


SETTINGS_FILE = 'CROSS_SETTINGS.json'


def _reusable(d: str, kind: str, want: Dict, result: str, inputs: List[str]) -> bool:
    """True when the folder already holds this kind of cross-check result,
    made with the same settings and newer than what it was computed from
    (a later compare rewrites TRUTH_RANKING.csv, and then it is redone)."""
    import json
    sp, rp = os.path.join(d, SETTINGS_FILE), os.path.join(d, result)
    if not (os.path.exists(sp) and os.path.exists(rp)):
        return False
    try:
        with open(sp, encoding='utf-8') as f:
            got = json.load(f).get(kind)
    except (OSError, ValueError):
        return False
    if got != want:
        return False
    t = os.path.getmtime(rp)
    return all(not os.path.exists(os.path.join(d, n)) or os.path.getmtime(os.path.join(d, n)) <= t for n in inputs)


def _remember(d: str, kind: str, want: Dict) -> None:
    import json
    sp = os.path.join(d, SETTINGS_FILE)
    try:
        with open(sp, encoding='utf-8') as f:
            allk = json.load(f)
    except (OSError, ValueError):
        allk = {}
    allk[kind] = want
    try:
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(allk, f, indent=1)
    except OSError:
        pass


def batch(parent: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
          tol_m: Optional[float] = None, truth_csv: str = '', jobs: str = '', recompute: bool = False,
          keep: int = KEEP_PER_CELL) -> int:
    """Every finished automatch job in parent or one level below it
    (output/output359), or only the numbers in jobs ('335-363, 370'): the
    full cross-check when its match files are there, and the replay on
    TRUTH_POINTS.csv when that is. Running jobs and folders that are not
    automatch outputs (other users', other tools') are left alone. A folder
    already cross-checked with the same settings is read back, not computed
    again (recompute=True redoes it). One row per job and channel in
    parent/CROSS_ALL.csv, with 'reused' marking the rows read back."""
    import glob as _glob
    found, skipped = find_jobs(parent, parse_jobs(jobs))
    for name, why in skipped.items():
        print(f'[Cross] {name}: skipped, {why}')
    print(f'[Cross] {len(found)} finished job(s) to check under {parent}')
    rows = []
    for d in found:
        job = os.path.relpath(d, parent)
        print(f'[Cross] ===== {job}')
        want = {'top_n': top_n, 'min_agree': min_agree, 'keep': keep, 'cell_km': cell_km, 'tolerance_m': tol_m,
                'truth_csv': truth_csv or ''}
        if not recompute and _reusable(d, 'matches', want, 'CROSS_SUMMARY.csv', ['TRUTH_RANKING.csv',
                                                                                  'RUN_MANIFEST.csv']):
            for r in pd.read_csv(os.path.join(d, 'CROSS_SUMMARY.csv')).to_dict('records'):
                rows.append({'job': job, 'mode': 'matches', 'reused': True, **r})
            print(f'[Cross] {job}: cross-check on the matches read back from CROSS_SUMMARY.csv (same settings)')
        elif _glob.glob(os.path.join(d, '**', 'filtered_*'), recursive=True):
            try:
                tcsv = truth_csv or job_truth_csv(d)
                print(f'[Cross] {job}: ground truth ' + (tcsv if tcsv else 'none found in its settings'))
                res = run(d, top_n, min_agree, cell_km, tol_m, tcsv, keep=keep)
                for ch, r in res.items():
                    rows.append({'job': job, 'mode': 'matches', 'reused': False,
                                 **{k: v for k, v in r.items() if k != 'channel'}, 'channel': ch})
                if res:
                    _remember(d, 'matches', want)
            except Exception as e:
                print(f'[Cross] {job}: {type(e).__name__}: {e}')
        tp, tr = os.path.join(d, 'TRUTH_POINTS.csv'), os.path.join(d, 'TRUTH_RANKING.csv')
        want_t = {k: want[k] for k in ('top_n', 'min_agree', 'tolerance_m')}
        if os.path.exists(tp) and os.path.exists(tr):
            try:
                reuse = not recompute and _reusable(d, 'truth points', want_t, 'CROSS_FROM_TRUTH_POINTS.csv',
                                                    ['TRUTH_POINTS.csv', 'TRUTH_RANKING.csv'])
                if reuse:
                    t = pd.read_csv(os.path.join(d, 'CROSS_FROM_TRUTH_POINTS.csv'))
                    print(f'[Cross] {job}: truth-point replay read back from CROSS_FROM_TRUTH_POINTS.csv')
                else:
                    t = from_truth_points(tp, tr, top_n, min_agree, tol_m)
                    t.to_csv(os.path.join(d, 'CROSS_FROM_TRUTH_POINTS.csv'), index=False)
                    _remember(d, 'truth points', want_t)
                for r in t.to_dict('records'):
                    rows.append({'job': job, 'mode': 'truth points', 'reused': reuse, **r})
                    print(f"[Cross] {job} {r['channel']}: cross-checked {r['cross_rmse_m'] or float('nan'):.0f} m at "
                          f"{r['cross_reached']}/{r['truth_total']} points; best single "
                          f"{r['best_single_rmse_m']:.0f} m ({r['best_single']})")
            except Exception as e:
                print(f'[Cross] {job} truth points: {type(e).__name__}: {e}')
    if rows:
        out = os.path.join(parent, 'CROSS_ALL.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'[Cross] {len(rows)} row(s) -> {out}')
    else:
        print(f'[Cross] no job output folder with match files or TRUTH_POINTS.csv under {parent}')
    return 0 if rows else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Keep the matches several of the best configurations agree on')
    ap.add_argument('output_dir')
    ap.add_argument('--top', type=int, default=5, help='configurations compared, one per detector (default 5)')
    ap.add_argument('--min-agree', type=int, default=0, help='how many must agree in a cell (default: more than half)')
    ap.add_argument('--keep', type=int, default=KEEP_PER_CELL,
                    help='agreeing configurations kept per cell, the least distorted there (default 3)')
    ap.add_argument('--cell-km', type=float, default=25.0, help='cell size (default 25 km)')
    ap.add_argument('--tolerance-m', type=float, default=None, help='agreement tolerance (default: 1.5 x the '
                                                                         'RANSAC threshold, 3 px)')
    ap.add_argument('--truth', default='', help="ground-truth CSV, to score the result (default: each job's "
                                                 "own truth_csv, from its job_used.json)")
    ap.add_argument('--batch', action='store_true',
                    help='output_dir holds many job output folders: cross-check each, results in CROSS_ALL.csv')
    ap.add_argument('--share-local', default='', help="where the server's share is mounted here, e.g. V:\\")
    ap.add_argument('--share-server', default='', help='the same share on the server, e.g. /maintenance/')
    ap.add_argument('--recompute', action='store_true',
                    help='with --batch: compute again even where a cross-check with the same settings exists')
    ap.add_argument('--jobs', default='', help="with --batch: only these job numbers, e.g. '335-363, 370'")
    ap.add_argument('--from-truth-points', action='store_true',
                    help='replay on output_dir/TRUTH_POINTS.csv + TRUTH_RANKING.csv only (no match files)')
    a = ap.parse_args(argv)
    global SHARE
    SHARE = (a.share_local, a.share_server)
    if a.batch:
        return batch(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth, a.jobs, a.recompute,
                     a.keep)
    if a.from_truth_points:
        t = from_truth_points(os.path.join(a.output_dir, 'TRUTH_POINTS.csv'),
                              os.path.join(a.output_dir, 'TRUTH_RANKING.csv'), a.top, a.min_agree, a.tolerance_m)
        print(t.to_string(index=False))
        t.to_csv(os.path.join(a.output_dir, 'CROSS_FROM_TRUTH_POINTS.csv'), index=False)
        return 0 if len(t) else 1
    res = run(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth, keep=a.keep)
    return 0 if res else 1


if __name__ == '__main__':
    sys.exit(main())
