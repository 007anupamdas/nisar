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
   when at least cross_min_agree of them lie within the consensus tolerance
   of their common median;
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
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAX_POINTS_PER_CELL = 200      # per configuration and cell, in the RIVAL file


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


def cross_check(top: List[Dict], cell_m: float, tol_m: float, min_agree: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(points kept, one row per cell). A cell is kept when min_agree of the
    top configurations with matches there lie within tol_m of their median."""
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
        for i in agree:
            p = top[i]['points']
            ij = np.floor(p[['E', 'N']].to_numpy() / cell_m).astype(int)
            sel = p[(ij[:, 0] == k[0]) & (ij[:, 1] == k[1])]
            sel = sel[np.hypot(sel['dE'] - cde, sel['dN'] - cdn) <= tol_m]
            if len(sel) > MAX_POINTS_PER_CELL:
                sel = sel.sample(MAX_POINTS_PER_CELL, random_state=0)
            kept.append(sel.assign(cross_detector=top[i]['detector'], cross_matcher=top[i]['matcher'],
                                   cross_sweep=top[i]['sweep']))
    pts = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    return pts, pd.DataFrame(rows)


def run(output_dir: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
        tol_m: Optional[float] = None, truth_csv: str = '', working_crs: Optional[str] = None,
        cfgs: Optional[List[Dict]] = None, quiet: bool = False) -> Dict[str, Dict]:
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
    tol = float(tol_m or settings.get('tolerance_m') or 0.0)
    cell_m = float(cell_km) * 1000.0
    cfgs = cfgs if cfgs is not None else _configs(output_dir)
    rank_path = os.path.join(output_dir, 'TRUTH_RANKING.csv')
    ranking = pd.read_csv(rank_path) if os.path.exists(rank_path) else None
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
        pts, cells = cross_check(top, cell_m, tol, req)
        kept = int(cells['kept'].sum()) if not cells.empty else 0
        res = {'channel': ch, 'chosen_by': how, 'top_n': len(top), 'min_agree': req, 'cell_km': cell_km,
               'tolerance_m': tol, 'cells_with_matches': int((cells['configs_with_matches'] >= 2).sum())
               if not cells.empty else 0, 'cells_kept': kept, 'points': int(len(pts)),
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
            msg = (f"[Cross] {ch}: top {len(top)} by {how}; {kept} of {res['cells_with_matches']} cells where "
                   f"{req}+ agree within {tol:.0f} m; {len(pts)} points")
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


def batch(parent: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
          tol_m: Optional[float] = None, truth_csv: str = '') -> int:
    """Every job output folder under parent (one holding RUN_MANIFEST.csv or
    TRUTH_RANKING.csv, up to two levels down): the full cross-check when its
    match files are there, and the replay on TRUTH_POINTS.csv when that is.
    One row per job and channel in parent/CROSS_ALL.csv."""
    import glob as _glob
    found = sorted({os.path.dirname(p) for pat in ('RUN_MANIFEST.csv', 'TRUTH_RANKING.csv')
                    for p in _glob.glob(os.path.join(parent, pat)) + _glob.glob(os.path.join(parent, '*', pat))
                    + _glob.glob(os.path.join(parent, '*', '*', pat))})
    rows = []
    for d in found:
        job = os.path.relpath(d, parent)
        print(f'[Cross] ===== {job}')
        if _glob.glob(os.path.join(d, '**', 'filtered_*'), recursive=True):
            try:
                for ch, r in run(d, top_n, min_agree, cell_km, tol_m, truth_csv).items():
                    rows.append({'job': job, 'mode': 'matches', **{k: v for k, v in r.items() if k != 'channel'},
                                 'channel': ch})
            except Exception as e:
                print(f'[Cross] {job}: {type(e).__name__}: {e}')
        tp, tr = os.path.join(d, 'TRUTH_POINTS.csv'), os.path.join(d, 'TRUTH_RANKING.csv')
        if os.path.exists(tp) and os.path.exists(tr):
            try:
                t = from_truth_points(tp, tr, top_n, min_agree, tol_m)
                t.to_csv(os.path.join(d, 'CROSS_FROM_TRUTH_POINTS.csv'), index=False)
                for r in t.to_dict('records'):
                    rows.append({'job': job, 'mode': 'truth points', **r})
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
    ap.add_argument('--cell-km', type=float, default=25.0, help='cell size (default 25 km)')
    ap.add_argument('--tolerance-m', type=float, default=None, help='agreement tolerance (default: the job\'s '
                                                                         'consensus tolerance)')
    ap.add_argument('--truth', default='', help='ground-truth CSV, to score the result')
    ap.add_argument('--batch', action='store_true',
                    help='output_dir holds many job output folders: cross-check each, results in CROSS_ALL.csv')
    ap.add_argument('--from-truth-points', action='store_true',
                    help='replay on output_dir/TRUTH_POINTS.csv + TRUTH_RANKING.csv only (no match files)')
    a = ap.parse_args(argv)
    if a.batch:
        return batch(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth)
    if a.from_truth_points:
        t = from_truth_points(os.path.join(a.output_dir, 'TRUTH_POINTS.csv'),
                              os.path.join(a.output_dir, 'TRUTH_RANKING.csv'), a.top, a.min_agree, a.tolerance_m)
        print(t.to_string(index=False))
        t.to_csv(os.path.join(a.output_dir, 'CROSS_FROM_TRUTH_POINTS.csv'), index=False)
        return 0 if len(t) else 1
    res = run(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth)
    return 0 if res else 1


if __name__ == '__main__':
    sys.exit(main())
