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
   cross-checked points score against it;
4. RIVAL_FINAL_<channel>.csv is the file to load in RIVAL: at most
   cross_max_points (500) of those points, spread over the whole area they
   cover (thin): one per cell of a fine grid, the one closest to the offset
   the points of that cell agree on.

    python automatch_job.py cross <output_dir> [--top 5] [--min-agree 3] [--cell-km 25] [--truth pts.csv]
    python automatch_job.py cross <output_dir> --from-truth-points     # a finished job, no match files
    python automatch_job.py cross <folder of job outputs> --batch      # every job, results in CROSS_ALL.csv
"""
from __future__ import annotations

import argparse
import math
import time
import glob
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
MAX_FINAL_POINTS = 500         # RIVAL_DENSE_<channel>.csv: up to this many, spread over what has matches (0: off)
FINAL_POINTS = 24              # RIVAL_FINAL_<channel>.csv: one best point per sub-tile of a grid over the scene
CONSISTENT_CELL_PX = 64        # RIVAL_CONSISTENT_<channel>.csv: cells of this many working pixels (8 x 8 in a
                               # 512 px window); a cell counts when the top detectors agree there, one point kept
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
    # each configuration's points grouped by cell once (not filtered again for every cell:
    # 64 px cells on MV are tens of thousands)
    by_cell = []
    for c in top:
        p = c['points']
        ij = np.floor(p[['E', 'N']].to_numpy() / cell_m).astype(int)
        by_cell.append({k: v for k, v in p.groupby([ij[:, 0], ij[:, 1]], sort=False).indices.items()})
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
        inside = {i: top[i]['points'].iloc[by_cell[i].get(k, [])] for i in agree}
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


def thin(pts: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """At most max_points of the cross-checked points, spread over the whole
    area they cover: a square grid as fine as gives no more occupied cells
    than max_points, and of each cell the one point closest to the median
    offset of the points around it (4 x 4 cells: the offset most of them
    agree on); a point far off that (OUTLIER_K x the usual scatter) is left
    out. Every part of the image that has points keeps one; dense parts are
    thinned."""
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    E, N = pts['E'].to_numpy(float), pts['N'].to_numpy(float)
    e0, n0 = E.min(), N.min()

    def cells(size):
        return np.floor((E - e0) / size).astype(np.int64) * 1000003 + np.floor((N - n0) / size).astype(np.int64)
    lo, hi = 1.0, max(E.max() - e0, N.max() - n0, 1.0) + 1.0      # hi: one cell
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if len(np.unique(cells(mid))) > max_points:
            lo = mid
        else:
            hi = mid
    p = pts.assign(_cell=cells(hi), _near=cells(4.0 * hi))      # the offset agreed on: of the 4 x 4 cells around
    g = p.groupby('_near')
    p['_d'] = np.hypot(p['dE'] - g['dE'].transform('median'), p['dN'] - g['dN'].transform('median'))
    best = p.loc[p.groupby('_cell')['_d'].idxmin().to_numpy()]
    limit = OUTLIER_K * 1.4826 * float(np.median(p['_d']))       # a cell's only point, off its surroundings: dropped
    if limit > 0:
        best = best[best['_d'] <= limit]
    return pts.loc[best.index].reset_index(drop=True)


def scene_extent(output_dir: str, crs: str, fallback: Optional[pd.DataFrame] = None):
    """((xmin, ymin, xmax, ymax) of the input image in the working CRS,
    covered(x0, y0, x1, y1) -> share of that box on valid image or None,
    source) -- the image as this machine reaches it (share mapping), else
    the extent of fallback's points (no coverage test then)."""
    import json
    try:
        with open(os.path.join(output_dir, 'job_used.json'), encoding='utf-8') as f:
            job = json.load(f)
    except (OSError, ValueError):
        job = {}
    inp = job.get('input_path') or ''
    for loc in (inp, local_path(inp, output_dir, job.get('output_dir') or ''), _by_share(inp)):
        if loc and os.path.isfile(loc):
            try:
                import rasterio as rio
                from pyproj import CRS, Transformer
                from rasterio.enums import Resampling
                with rio.open(loc) as src:
                    s = max(1.0, max(src.width, src.height) / 512.0)
                    h, w = max(1, int(src.height / s)), max(1, int(src.width / s))
                    a = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
                    nod = src.nodatavals[0] if src.nodatavals else None
                    tr = src.transform * src.transform.scale(src.width / float(w), src.height / float(h))
                    rcrs = src.crs
                valid = np.isfinite(a) & (a != 0) & ((a != nod) if nod is not None else True)
                to_w = Transformer.from_crs(rcrs, CRS.from_user_input(crs), always_xy=True)
                to_r = Transformer.from_crs(CRS.from_user_input(crs), rcrs, always_xy=True)
                rows, cols = np.nonzero(valid)
                if not len(rows):
                    break
                xs, ys = tr * (cols + 0.5, rows + 0.5)
                X, Y = to_w.transform(np.asarray(xs), np.asarray(ys))
                inv = ~tr

                def covered(x0, y0, x1, y1):
                    gx, gy = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
                    rx, ry = to_r.transform(gx.ravel(), gy.ravel())
                    c, r = inv * (np.asarray(rx), np.asarray(ry))
                    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
                    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
                    hit = np.zeros(len(r), bool)
                    hit[ok] = valid[r[ok], c[ok]]
                    return float(hit.mean())
                return (float(np.min(X)), float(np.min(Y)), float(np.max(X)), float(np.max(Y))), covered, 'image'
            except Exception:
                break
    if fallback is not None and len(fallback):
        return ((float(fallback['E'].min()), float(fallback['N'].min()), float(fallback['E'].max()),
                 float(fallback['N'].max())), None, 'matches (the image is not reachable here)')
    return None, None, ''


def final_grid(pts: pd.DataFrame, extent, n_points: int, covered=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(one point per sub-tile, the tiles): the scene's extent cut into a
    grid of about n_points tiles of its shape; in each tile, of the points
    there (the cross-checked ones: several detectors agree on them, cloud and
    featureless interiors left out when the filter is on), the one closest
    to the offset they agree on (their median) -- the most accurate one the
    matches can tell. A tile without such points fails, with the reason
    (outside the image, or nothing agreed there: water, cloud, featureless)."""
    if extent is None or n_points <= 0:
        return pts.iloc[0:0], pd.DataFrame()
    x0, y0, x1, y1 = extent
    W, H = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    nx = max(1, int(round(math.sqrt(n_points * W / H))))
    ny = max(1, int(math.ceil(n_points / float(nx))))
    tw, th = W / nx, H / ny
    E = pts['E'].to_numpy(float) if len(pts) else np.zeros(0)
    N = pts['N'].to_numpy(float) if len(pts) else np.zeros(0)
    ti = np.clip(np.floor((E - x0) / tw).astype(int), 0, nx - 1) if len(pts) else E.astype(int)
    tj = np.clip(np.floor((N - y0) / th).astype(int), 0, ny - 1) if len(pts) else N.astype(int)
    chosen, rows = [], []
    for j in range(ny - 1, -1, -1):             # north to south, west to east: tile 1 is the top left
        for i in range(nx):
            box = (x0 + i * tw, y0 + j * th, x0 + (i + 1) * tw, y0 + (j + 1) * th)
            row = {'tile': len(rows) + 1, 'col': i + 1, 'row': ny - j, 'E_min': box[0], 'N_min': box[1],
                   'E_max': box[2], 'N_max': box[3]}
            share = covered(*box) if covered is not None else None
            row['image_share'] = None if share is None else round(share, 2)
            sel = pts[(ti == i) & (tj == j)] if len(pts) else pts
            if not len(sel):
                row['status'] = ('outside the image' if share is not None and share < 0.1
                                 else 'no agreed matches (water, cloud or featureless?)')
                rows.append(row)
                continue
            mde, mdn = float(sel['dE'].median()), float(sel['dN'].median())
            d = np.hypot(sel['dE'] - mde, sel['dN'] - mdn)
            if 'agree_spread_m' in sel.columns:   # consistent points: the most detectors, the tightest spread
                order = sel.assign(_d=d).sort_values(['n_agree', 'agree_spread_m', '_d'],
                                                     ascending=[False, True, True])
                pick = order.index[0]
            else:
                pick = d.idxmin()
            best = sel.loc[pick]
            row.update({'status': 'ok', 'candidates': int(len(sel)),
                        'detectors': int(sel['cross_detector'].nunique()) if 'cross_detector' in sel.columns else None,
                        'tile_median_dE_m': round(mde, 2), 'tile_median_dN_m': round(mdn, 2),
                        'tile_scatter_m': round(float(1.4826 * np.median(d)), 2),
                        'point_dE_m': round(float(best['dE']), 2), 'point_dN_m': round(float(best['dN']), 2),
                        'point_off_median_m': round(float(d.loc[pick]), 2)})
            if 'agree_spread_m' in sel.columns:
                row.update({'point_detectors_agreeing': int(best['n_agree']),
                            'point_agree_spread_m': float(best['agree_spread_m'])})
            rows.append(row)
            chosen.append(sel.loc[[pick]].assign(tile=row['tile']))
    fin = pd.concat(chosen, ignore_index=True) if chosen else pts.iloc[0:0]
    return fin, pd.DataFrame(rows)


def _job_setting(d: str, key: str, default):
    try:
        import json
        with open(os.path.join(d, 'job_used.json'), encoding='utf-8') as f:
            v = json.load(f).get(key)
        return default if v is None else type(default)(v)
    except (OSError, ValueError, TypeError):
        return default


def consistent_points(top: List[Dict], cell_m: float, tol_m: float, need: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(one point per consistent cell, the cells): the scene in cells of
    cell_m (64 working pixels: 8 x 8 cells in a 512 px window); in each, the
    top configurations' offsets there (their medians in the cell), and the
    cell counts when at least `need` of them lie within tol_m of the median
    of all of them. Its point: of the agreeing configurations' points within
    tol_m of that, the one closest to it (the cell's best keypoint), with the
    number agreeing (n_agree) and how far apart they are (agree_spread_m).
    Done for all cells at once (tens of thousands on MV)."""
    if not top:
        return pd.DataFrame(), pd.DataFrame()
    frames = []
    for k, c in enumerate(top):
        p = c['points']
        if len(p):
            frames.append(p.assign(_cfg=k, cross_detector=c['detector'], cross_matcher=c['matcher'],
                                   cross_sweep=c['sweep']))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    allp = pd.concat(frames, ignore_index=True)
    allp['cell_i'] = np.floor(allp['E'].to_numpy() / cell_m).astype(np.int64)
    allp['cell_j'] = np.floor(allp['N'].to_numpy() / cell_m).astype(np.int64)
    cm = allp.groupby(['cell_i', 'cell_j', '_cfg'], sort=False)[['dE', 'dN']].median().reset_index()
    cmed = cm.groupby(['cell_i', 'cell_j'])[['dE', 'dN']].median().rename(columns={'dE': 'mE', 'dN': 'mN'})
    cm = cm.join(cmed, on=['cell_i', 'cell_j'])
    cm['agrees'] = np.hypot(cm['dE'] - cm['mE'], cm['dN'] - cm['mN']) <= tol_m
    ag = cm[cm['agrees']]
    stats = ag.groupby(['cell_i', 'cell_j']).agg(agreeing=('_cfg', 'size'), aE0=('dE', 'min'), aE1=('dE', 'max'),
                                                  aN0=('dN', 'min'), aN1=('dN', 'max'),
                                                  median_dE_m=('dE', 'median'), median_dN_m=('dN', 'median'))
    cells = cm.groupby(['cell_i', 'cell_j']).agg(configs_with_matches=('_cfg', 'size')).join(stats)
    cells['agreeing'] = cells['agreeing'].fillna(0).astype(int)
    cells['kept'] = (cells['agreeing'] >= max(2, int(need))) & (cells['configs_with_matches'] >= 2)
    cells['kept_max_gap_m'] = np.hypot(cells['aE1'] - cells['aE0'], cells['aN1'] - cells['aN0']).round(1)
    cells = cells.drop(columns=['aE0', 'aE1', 'aN0', 'aN1']).reset_index()
    kc = cells[cells['kept']]
    ok = ag.merge(kc[['cell_i', 'cell_j', 'median_dE_m', 'median_dN_m', 'agreeing', 'kept_max_gap_m']],
                  on=['cell_i', 'cell_j'])[['cell_i', 'cell_j', '_cfg', 'median_dE_m', 'median_dN_m', 'agreeing',
                                            'kept_max_gap_m']]
    cand = allp.merge(ok, on=['cell_i', 'cell_j', '_cfg'])
    if cand.empty:
        return cand, cells
    d = np.hypot(cand['dE'] - cand['median_dE_m'], cand['dN'] - cand['median_dN_m'])
    cand = cand.assign(_d=d)[d <= tol_m]
    best = cand.sort_values('_d').groupby(['cell_i', 'cell_j'], sort=False).head(1)
    best = best.assign(n_agree=best['agreeing'].astype(int), agree_spread_m=best['kept_max_gap_m'],
                       off_agreed_m=best['_d'].round(1))
    return best.drop(columns=['_d', '_cfg', 'cell_i', 'cell_j', 'median_dE_m', 'median_dN_m', 'agreeing',
                              'kept_max_gap_m']).reset_index(drop=True), cells


def _job_cell_px(d: str) -> int:
    """The job's cross_cell_px (job_used.json), else CONSISTENT_CELL_PX."""
    return _job_setting(d, 'cross_cell_px', CONSISTENT_CELL_PX)


def _job_final_points(d: str) -> int:
    """The job's final_points (job_used.json), else FINAL_POINTS."""
    return _job_setting(d, 'final_points', FINAL_POINTS)


def final_of(rival_csv: str, crs: str, final_points: int = FINAL_POINTS, output_dir: str = '') -> str:
    """RIVAL_BEST_<...>.csv -> RIVAL_BEST_FINAL_<...>.csv beside it: the same
    per-tile pick (final_grid) for the best single configuration's file. ''
    when off, the file is unreadable or it is a FINAL file already."""
    import automatch_rival as R
    name = os.path.basename(rival_csv)
    if final_points <= 0 or '_FINAL_' in name or not name.startswith('RIVAL_BEST_'):
        return ''
    try:
        p = R.read_rival_csv(rival_csv)
    except Exception:
        return ''
    if p.empty:
        return ''
    pts = pd.DataFrame({'x1-map': p['ix'], 'y1-map': p['iy'], 'x2-map': p['rx'], 'y2-map': p['ry'],
                        'E': p['ix'], 'N': p['iy'], 'dE': p['ix'] - p['rx'], 'dN': p['iy'] - p['ry']})
    import automatch_rival as R2
    pts = R2.apply_point_mask(pts)                  # the interior / cloud filter, when on
    ch = re.search(r'_(band\d+|[A-Z]{2})\.csv$', name)
    cpath = os.path.join(os.path.dirname(rival_csv), f'CONSISTENT_CELLS_{ch.group(1)}.csv') if ch else ''
    if cpath and os.path.exists(cpath):           # only where the top detectors agree (64 px cells)
        try:
            cells = pd.read_csv(cpath)
            cells = cells[cells['kept'].astype(str).str.lower() == 'true']
            cell_m = float(cells['cell_m'].iloc[0]) if len(cells) else 0.0
            tol = float(cells['tolerance_m'].iloc[0]) if len(cells) else 0.0
            if cell_m > 0:
                ij = np.floor(pts[['E', 'N']].to_numpy() / cell_m).astype(int)
                p2 = pts.assign(cell_i=ij[:, 0], cell_j=ij[:, 1]).merge(
                    cells[['cell_i', 'cell_j', 'median_dE_m', 'median_dN_m']], on=['cell_i', 'cell_j'])
                p2 = p2[np.hypot(p2['dE'] - p2['median_dE_m'], p2['dN'] - p2['median_dN_m']) <= tol]
                d = np.hypot(p2['dE'] - p2['median_dE_m'], p2['dN'] - p2['median_dN_m'])
                pts = p2.assign(_d=d).sort_values('_d').groupby(['cell_i', 'cell_j']).head(1).drop(
                    columns=['_d', 'cell_i', 'cell_j', 'median_dE_m', 'median_dN_m']).reset_index(drop=True)
        except Exception as e:
            print(f'[Cross] {name}: consistent cells not applied ({type(e).__name__}: {e})')
    extent, covered, _ = scene_extent(output_dir or os.path.dirname(rival_csv), crs, pts)
    fin, tiles = final_grid(pts, extent, final_points, covered)
    out = os.path.join(os.path.dirname(rival_csv), name.replace('RIVAL_BEST_', 'RIVAL_BEST_FINAL_', 1))
    R.write_rival_csv(fin, crs, out)
    ok = int((tiles['status'] == 'ok').sum()) if len(tiles) else 0
    print(f'[Cross] {name}: {len(pts)} points -> {len(fin)} best, one per sub-tile ({ok} of {len(tiles)} tiles) -> {out}')
    return out


def best_finals(output_dir: str, crs: Optional[str] = None, final_points: Optional[int] = None) -> List[str]:
    """final_of for every RIVAL_BEST_*.csv of a job folder."""
    import glob as _glob
    import automatch_truth as T
    crs = crs or T.working_crs_of(output_dir)
    n = _job_final_points(output_dir) if final_points is None else final_points
    if not crs:
        return []
    return [o for o in (final_of(p, crs, n, output_dir)
                        for p in sorted(_glob.glob(os.path.join(output_dir, 'RIVAL_BEST_*.csv')))) if o]


def run(output_dir: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
        tol_m: Optional[float] = None, truth_csv: str = '', working_crs: Optional[str] = None,
        cfgs: Optional[List[Dict]] = None, quiet: bool = False, keep: int = KEEP_PER_CELL,
        out_dir: Optional[str] = None, ranking_csv: Optional[str] = None,
        max_points: int = MAX_FINAL_POINTS, final_points: Optional[int] = None,
        cell_px: Optional[int] = None) -> Dict[str, Dict]:
    """Cross-check every channel of an output folder; returns per channel
    {'csv', 'summary', 'configs', 'cells', 'points', 'truth_rmse_m'}.
    out_dir: where its files go (default output_dir); ranking_csv: the truth
    ranking to pick the top configurations by (default output_dir's)."""
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
    out_dir = out_dir or output_dir
    os.makedirs(out_dir, exist_ok=True)
    rank_path = ranking_csv or os.path.join(output_dir, 'TRUTH_RANKING.csv')
    ranking = pd.read_csv(rank_path) if os.path.exists(rank_path) else None
    truth_csv = truth_csv or job_truth_csv(output_dir)
    truth = T.truth_in_crs(T.load_truth(truth_csv), crs) if truth_csv else None
    need = min_agree if min_agree > 0 else max(2, top_n // 2 + 1)
    final_points = _job_final_points(output_dir) if final_points is None else int(final_points)
    cell_px = _job_cell_px(output_dir) if cell_px is None else int(cell_px)
    res_m = float(T.job_resolution(output_dir) or 10.0)
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
            path = os.path.join(out_dir, f'RIVAL_CROSS_{ch}.csv')
            R.write_rival_csv(pts, crs, path, path[:-4] + '_detail.csv')
            res['csv'] = path
            if max_points > 0:             # many, spread over what has matches
                dense = thin(pts, max_points)
                dpath = os.path.join(out_dir, f'RIVAL_DENSE_{ch}.csv')
                R.write_rival_csv(dense, crs, dpath, dpath[:-4] + '_detail.csv')
                res.update({'dense_csv': dpath, 'dense_points': int(len(dense))})
            pool = pts
            if cell_px > 0:                # the best keypoint of each 64 px cell where the top detectors agree
                cpts, ccells = consistent_points(top, cell_px * res_m, tol, req)
                ccells.assign(cell_m=cell_px * res_m, tolerance_m=tol).to_csv(
                    os.path.join(out_dir, f'CONSISTENT_CELLS_{ch}.csv'), index=False)
                n_c = int(ccells['kept'].sum()) if len(ccells) else 0
                res.update({'consistent_cell_m': round(cell_px * res_m, 1), 'consistent_cells': n_c,
                            'consistent_points': int(len(cpts))})
                if len(cpts):
                    cp = os.path.join(out_dir, f'RIVAL_CONSISTENT_{ch}.csv')
                    R.write_rival_csv(cpts, crs, cp, cp[:-4] + '_detail.csv')
                    res['consistent_csv'] = cp
                    pool = cpts
                    if truth is not None:
                        s, _ = T.score_config({'points': cpts.assign(chip=cpts['source_file'])[
                            ['E', 'N', 'dE', 'dN', 'chip']]}, truth, 5000.0)
                        res.update({'consistent_truth_rmse_m': s['truth_rmse_m'],
                                    'consistent_truth_reached': s['truth_reached']})
            if final_points > 0:           # the file for RIVAL: one best point per sub-tile of the scene
                extent, covered, src = scene_extent(output_dir, crs, pool)
                fin, tiles = final_grid(pool, extent, final_points, covered)
                fpath = os.path.join(out_dir, f'RIVAL_FINAL_{ch}.csv')
                R.write_rival_csv(fin, crs, fpath, fpath[:-4] + '_detail.csv')
                tiles.to_csv(os.path.join(out_dir, f'FINAL_TILES_{ch}.csv'), index=False)
                n_ok = int((tiles['status'] == 'ok').sum()) if len(tiles) else 0
                res.update({'final_csv': fpath, 'final_points': int(len(fin)), 'final_tiles': int(len(tiles)),
                            'final_tiles_ok': n_ok, 'final_extent_from': src})
                if truth is not None and len(fin):
                    s, _ = T.score_config({'points': fin.assign(chip=fin['source_file'])[
                        ['E', 'N', 'dE', 'dN', 'chip']]}, truth, 5000.0)
                    res.update({'final_truth_rmse_m': s['truth_rmse_m'], 'final_truth_reached': s['truth_reached']})
            if truth is not None:
                s, _ = T.score_config({'points': pts.assign(chip=pts['source_file'])[['E', 'N', 'dE', 'dN', 'chip']]},
                                      truth, 5000.0)
                res.update({'truth_rmse_m': s['truth_rmse_m'], 'truth_reached': s['truth_reached'],
                            'truth_total': s['truth_total']})
        cells.to_csv(os.path.join(out_dir, f'CROSS_CELLS_{ch}.csv'), index=False)
        summary_rows.append(res)
        out[ch] = res
        if not quiet:
            msg = (f"[Cross] {ch}: top {len(top)} by {how}; {kept} of {n_cells} cells ({100 * share:.0f} %) where "
                   f"{req}+ agree within {tol:.0f} m, the {keep} least distorted kept; {len(pts)} points; "
                   + ('reliable' if res['reliable'] else f"NOT reliable (under {100 * RELIABLE_SHARE:.0f} % of cells agree)"))
            if res.get('truth_rmse_m') is not None:
                msg += f"; truth RMSE {res['truth_rmse_m']:.0f} m at {res['truth_reached']}/{res['truth_total']}"
            print(msg + (f" -> {res['csv']}" if res.get('csv') else ''))
            if res.get('consistent_points') is not None:
                print(f"[Cross] {ch}: {res['consistent_points']} consistent points: the best keypoint of each "
                      f"{cell_px} px cell ({res['consistent_cell_m']:.0f} m) where {req}+ of the top {len(top)} agree"
                      + (f" (truth RMSE {res['consistent_truth_rmse_m']:.0f} m at {res['consistent_truth_reached']}/"
                         f"{res['truth_total']})" if res.get('consistent_truth_rmse_m') is not None else '')
                      + (f" -> {res['consistent_csv']}" if res.get('consistent_csv') else ''))
            if res.get('final_csv'):
                print(f"[Cross] {ch}: for RIVAL: {res['final_points']} best points, one per sub-tile "
                      f"({res['final_tiles_ok']} of {res['final_tiles']} tiles have one; FINAL_TILES_{ch}.csv says "
                      f"why the others failed)"
                      + (f" (truth RMSE {res['final_truth_rmse_m']:.0f} m at {res['final_truth_reached']}/"
                         f"{res['truth_total']})" if res.get('final_truth_rmse_m') is not None else '')
                      + f" -> {res['final_csv']}")
            for c in top:
                print(f"         {c['detector']} + {c['matcher']} {c['ransac']} ({c['sweep']})")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, 'CROSS_SUMMARY.csv'), index=False)
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


STALE_S = 6 * 3600.0               # a log untouched this long: the job is not running any more


def job_state(d: str) -> str:
    """'finished' (an automatch run that ended: job_used.json, RUN_MANIFEST.csv
    and '[Job] done' at the end of automatch.log), 'compare' (a compare
    job's TRUTH_RANKING.csv + TRUTH_POINTS.csv), 'running' (an automatch run
    still going), or '' (not an automatch output: another user's or tool's)."""
    has = lambda n: os.path.exists(os.path.join(d, n))
    if has('job_used.json') and (has('automatch.log') or has('RUN_MANIFEST.csv')):
        done = has('RUN_MANIFEST.csv')
        tail = b''
        if has('automatch.log'):
            try:
                with open(os.path.join(d, 'automatch.log'), 'rb') as f:
                    f.seek(max(0, os.path.getsize(os.path.join(d, 'automatch.log')) - 8192))
                    tail = f.read()
            except OSError:
                pass
            done = done and b'[Job] done' in tail
        if done:
            return 'finished'
        # ended with an error after matching (e.g. job 1066: the end-of-job truth step):
        # its matches and ranking are there, so it can be benchmarked and cross-checked
        stopped = b'Traceback (most recent call last)' in tail or (
            has('automatch.log') and time.time() - os.path.getmtime(os.path.join(d, 'automatch.log')) > STALE_S)
        if stopped and (has('TRUTH_RANKING.csv') or glob.glob(os.path.join(d, 'RUN_MANIFEST_gpu*.csv'))):
            return 'finished'
        return 'running'
    if has('TRUTH_RANKING.csv') and has('TRUTH_POINTS.csv'):
        return 'compare'
    return ''


def find_jobs(parent: str, jobs: Optional[set] = None) -> Tuple[List[str], Dict[str, str]]:
    """(finished automatch job folders, {skipped folder: why}) in parent and
    one level below it (output/output359), optionally only those whose
    number (the last digits of the name) is in jobs."""
    cands = [parent] + sorted(os.path.join(parent, n) for n in os.listdir(parent)
                              if os.path.isdir(os.path.join(parent, n))
                              and n not in ('CLOUDFREE', 'rival', '_cache', '_queue'))   # a job's own folders
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


def _job_max_points(d: str) -> int:
    """The job's cross_max_points (job_used.json), else the default."""
    try:
        import json
        with open(os.path.join(d, 'job_used.json'), encoding='utf-8') as f:
            v = json.load(f).get('cross_max_points')
        return MAX_FINAL_POINTS if v is None else int(v)
    except (OSError, ValueError, TypeError):
        return MAX_FINAL_POINTS


def cross_job(d: str, job: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
              tol_m: Optional[float] = None, truth_csv: str = '', recompute: bool = False,
              keep: int = KEEP_PER_CELL, max_points: Optional[int] = None,
              final_points: Optional[int] = None) -> List[Dict]:
    """One finished job: the cross-check on its match files and its replay at
    the truth points, each read back when made with the same settings. Rows
    as in CROSS_ALL.csv."""
    import glob as _glob
    rows: List[Dict] = []
    print(f'[Cross] ===== {job}')
    n_final = _job_max_points(d) if max_points is None else int(max_points)
    n_tiles = _job_final_points(d) if final_points is None else int(final_points)
    want = {'top_n': top_n, 'min_agree': min_agree, 'keep': keep, 'cell_km': cell_km, 'tolerance_m': tol_m,
            'truth_csv': truth_csv or '', 'max_points': n_final, 'final_points': n_tiles,
            'cell_px': _job_cell_px(d), 'v': 3}
    if not recompute and _reusable(d, 'matches', want, 'CROSS_SUMMARY.csv', ['TRUTH_RANKING.csv',
                                                                              'RUN_MANIFEST.csv']):
        for r in pd.read_csv(os.path.join(d, 'CROSS_SUMMARY.csv')).to_dict('records'):
            rows.append({'job': job, 'mode': 'matches', 'reused': True, **r})
        print(f'[Cross] {job}: cross-check on the matches read back from CROSS_SUMMARY.csv (same settings)')
    elif _glob.glob(os.path.join(d, '**', 'filtered_*'), recursive=True):
        try:
            tcsv = truth_csv or job_truth_csv(d)
            print(f'[Cross] {job}: ground truth ' + (tcsv if tcsv else 'none found in its settings'))
            res = run(d, top_n, min_agree, cell_km, tol_m, tcsv, keep=keep, max_points=n_final, final_points=n_tiles)
            try:                               # the best single configuration, picked the same way
                best_finals(d, None, n_tiles)
            except Exception as e:
                print(f'[Cross] {job}: RIVAL_BEST_FINAL not written ({type(e).__name__}: {e})')
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
    return rows


def batch(parent: str, top_n: int = 5, min_agree: int = 0, cell_km: float = 25.0,
          tol_m: Optional[float] = None, truth_csv: str = '', jobs: str = '', recompute: bool = False,
          keep: int = KEEP_PER_CELL, max_points: Optional[int] = None, final_points: Optional[int] = None) -> int:
    """Every finished automatch job in parent or one level below it
    (output/output359), or only the numbers in jobs ('335-363, 370'): the
    full cross-check when its match files are there, and the replay on
    TRUTH_POINTS.csv when that is. Running jobs and folders that are not
    automatch outputs (other users', other tools') are left alone. A folder
    already cross-checked with the same settings is read back, not computed
    again (recompute=True redoes it). One row per job and channel in
    parent/CROSS_ALL.csv, with 'reused' marking the rows read back."""
    found, skipped = find_jobs(parent, parse_jobs(jobs))
    for name, why in skipped.items():
        print(f'[Cross] {name}: skipped, {why}')
    print(f'[Cross] {len(found)} finished job(s) to check under {parent}')
    rows = []
    for d in found:
        rows += cross_job(d, os.path.relpath(d, parent), top_n, min_agree, cell_km, tol_m, truth_csv, recompute, keep,
                          max_points, final_points)
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
    ap.add_argument('--cell-px', type=int, default=None,
                    help=f'RIVAL_CONSISTENT: cells of this many working pixels where the top detectors must agree, '
                         f'one best point each (default the job\'s cross_cell_px, else {CONSISTENT_CELL_PX}; 0: off)')
    ap.add_argument('--final-points', type=int, default=None,
                    help=f'RIVAL_FINAL / RIVAL_BEST_FINAL: one best point per sub-tile of a grid of about this many '
                         f'over the scene (default the job\'s final_points, else {FINAL_POINTS})')
    ap.add_argument('--max-points', type=int, default=None,
                    help=f'RIVAL_FINAL_<channel>.csv: at most this many best points over the whole coverage '
                         f'(default the job\'s, else {MAX_FINAL_POINTS}; 0: off)')
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
    ap.add_argument('--device', default='', help='accepted for symmetry with the other commands: the cross-check '
                                                 'only reads the saved matches and needs no GPU (cpu is fine)')
    a = ap.parse_args(argv)
    if str(a.device).strip().lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # '' is read as "not set" by some torch builds
    global SHARE
    SHARE = (a.share_local, a.share_server)
    if a.batch:
        return batch(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth, a.jobs, a.recompute,
                     a.keep, a.max_points, a.final_points)
    if a.from_truth_points:
        t = from_truth_points(os.path.join(a.output_dir, 'TRUTH_POINTS.csv'),
                              os.path.join(a.output_dir, 'TRUTH_RANKING.csv'), a.top, a.min_agree, a.tolerance_m)
        print(t.to_string(index=False))
        t.to_csv(os.path.join(a.output_dir, 'CROSS_FROM_TRUTH_POINTS.csv'), index=False)
        return 0 if len(t) else 1
    try:
        res = run(a.output_dir, a.top, a.min_agree, a.cell_km, a.tolerance_m, a.truth, keep=a.keep,
                  max_points=_job_max_points(a.output_dir) if a.max_points is None else a.max_points,
                  final_points=a.final_points, cell_px=a.cell_px)
        best_finals(a.output_dir, None, a.final_points)
    except ValueError as e:
        print(f'[Cross] {e}')
        return 2
    try:
        import automatch_report as RP
        RP.report(a.output_dir)
    except Exception as e:
        print(f'[Cross] REPORT not written: {type(e).__name__}: {e}')
    return 0 if res else 1


if __name__ == '__main__':
    sys.exit(main())
