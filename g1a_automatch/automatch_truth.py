#!/usr/bin/env python3
"""
automatch_truth -- which detector + matcher agrees best with manually measured
ground truth?

Ground truth is a RIVAL CSV of manually measured points (RIVAL's 'Export CSV':
In_X, In_Y, Ref_X, Ref_Y, DX_Err, DY_Err, Row, In_Lon, In_Lat, Ref_Lon,
Ref_Lat); the lon/lat columns are used, so the file's own X/Y grid does not
matter. Truth errors are recomputed in each run's working CRS, the grid the
tool measures in.

Every configuration a run tried -- detector variant x matcher (x matcher
setting x RANSAC setting) -- is scored, not only the one the chip consensus
picked. At each truth point the configuration's own error is estimated from
its matches:
    local    median error of its points within radius_m of the truth point
    surface  otherwise, its robust polynomial error surface (fitted to its
             chips) evaluated there -- 'extrapolated' when the point lies
             outside its chips' coverage; extrapolations are listed but not
             scored
and compared with the manual error there. Per configuration: truth points
reached, RMSE / bias / max of (tool - truth) in metres, rank.

Files (in the output folder):
    TRUTH_BY_DETECTOR_MATCHER.csv   the best setting of each detector + matcher, ranked
    TRUTH_RANKING.csv               every configuration, ranked
    TRUTH_POINTS.csv                every configuration x truth point

    python automatch_truth.py <output_dir> --truth manual.csv [--radius-km 5] [--chips all]

A job with truth_csv set does this after its run (also: automatch_job.py
compare). Chips: 'consensus' (default) scores the chips that survived the
chip consensus of each configuration, 'all' every chip it matched.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

HERE_ = os.path.dirname(os.path.abspath(__file__))
if HERE_ not in sys.path:
    sys.path.insert(0, HERE_)
import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TRUTH_COLUMNS = ['In_Lon', 'In_Lat', 'Ref_Lon', 'Ref_Lat']
RANK_COLS = ['rank', 'channel', 'sweep', 'detector', 'matcher', 'ransac', 'truth_rmse_m', 'truth_mean_dE_m',
             'truth_mean_dN_m', 'truth_max_m', 'truth_reached', 'truth_total', 'sec_per_window', 'pass_seconds',
             'n_local', 'n_surface',
             'n_extrapolated', 'n_chips', 'n_points', 'chips']


# ─────────────────────────────────────────────────────────────────────────────
# truth
# ─────────────────────────────────────────────────────────────────────────────
def load_truth(path: str) -> pd.DataFrame:
    """Manual points: truth_id, in_lon, in_lat, ref_lon, ref_lat."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{os.path.basename(path)}: ground truth needs the columns {TRUTH_COLUMNS} '
                         f'(missing {missing}); RIVAL "Export CSV" files have them')
    df = df.dropna(subset=TRUTH_COLUMNS)
    if df.empty:
        raise ValueError(f'{os.path.basename(path)}: no complete rows')
    ids = df['Row'].tolist() if 'Row' in df.columns else list(range(1, len(df) + 1))
    return pd.DataFrame({'truth_id': ids, 'in_lon': df['In_Lon'].astype(float).values,
                         'in_lat': df['In_Lat'].astype(float).values,
                         'ref_lon': df['Ref_Lon'].astype(float).values,
                         'ref_lat': df['Ref_Lat'].astype(float).values})


def truth_in_crs(truth: pd.DataFrame, crs: str) -> pd.DataFrame:
    """Truth positions (input image, 'In') and errors In - Ref in metres of crs."""
    from pyproj import CRS, Transformer
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True)
    ei, ni = tr.transform(truth['in_lon'].values, truth['in_lat'].values)
    er, nr = tr.transform(truth['ref_lon'].values, truth['ref_lat'].values)
    out = truth.copy()
    out['E'], out['N'] = np.asarray(ei), np.asarray(ni)
    out['dE'], out['dN'] = np.asarray(ei) - np.asarray(er), np.asarray(ni) - np.asarray(nr)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# the tool's configurations
# ─────────────────────────────────────────────────────────────────────────────
def _variant(filename: str) -> str:
    """Full detector-variant prefix of a match file name (dog_sosnet-aff1, ...)."""
    parts = os.path.splitext(filename)[0].split('_')
    for k in range(2, len(parts) - 3):
        if (parts[k].startswith('to') and parts[k + 1].startswith('pair')
                and parts[k + 2].startswith('scan') and parts[k + 3].startswith('pix')):
            return '_'.join(parts[:k - 1])
    return parts[0]


def _matcher_label(p: Dict) -> str:
    m = str(p.get('match_method'))
    if m == 'smnn' and p.get('match_parameter') is not None:
        return f"smnn{float(p['match_parameter']):g}"
    return m


def config_points(run_dir_filtered: str, final_dir: str, chips: str = 'consensus',
                  settings: Optional[Dict] = None) -> Dict[str, Dict]:
    """{config_key: {'detector','matcher','ransac','points': DataFrame(E, N, dE, dN, chip)}}
    from one run's filtered match files."""
    import automatch_rival as R
    import automatch_engine as E
    name = 'CHIP_STATS_SURVIVORS' if chips == 'consensus' else 'CHIP_STATS_ALL'
    tables = sorted(glob.glob(os.path.join(final_dir, f'{name}*.csv')))
    if not tables and not glob.glob(os.path.join(final_dir, 'CHIP_STATS_ALL*.csv')) and settings is not None:
        # the run stopped before its consensus (e.g. an unusable manual GCP
        # file): redo the consensus from the saved matches, no re-matching
        print(f'[Truth] {os.path.basename(final_dir)}: no consensus tables -- rebuilding them from '
              f'the saved matches')
        try:
            E.ChipConsensusSelector.select_configs(csv_dir=run_dir_filtered, output_dir=final_dir,
                                                   manual_gcp_csv='', **settings)
        except Exception as e:
            print(f'[Truth] rebuild failed: {type(e).__name__}: {e}')
        tables = sorted(glob.glob(os.path.join(final_dir, f'{name}*.csv')))
    if not tables:
        return {}
    stats = pd.read_csv(tables[0])
    if stats.empty or 'config_key' not in stats.columns:
        return {}
    out: Dict[str, Dict] = {}
    for key, g in stats.groupby('config_key'):
        files = g['filename'].tolist()
        pts = R.collect_points(run_dir_filtered, files)
        if pts.empty:
            continue
        p = E.MatchStatistics._parse_filename(files[0]) or {}
        out[key] = {
            'files': files,
            'detector': _variant(files[0]), 'matcher': _matcher_label(p),
            'ransac': f"{p.get('ransac_method')}{float(p.get('ransac_threshold', 0)):g}",
            'channel': p.get('nisar_pol'),
            'points': pd.DataFrame({'E': pts['x1-map'].values, 'N': pts['y1-map'].values,
                                    'dE': (pts['x1-map'] - pts['x2-map']).values,
                                    'dN': (pts['y1-map'] - pts['y2-map']).values,
                                    'chip': pts['source_file'].values}),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# estimating a configuration's error at a truth point
# ─────────────────────────────────────────────────────────────────────────────
def _design(x, y, degree):
    cols = [np.ones_like(x), x, y]
    if degree >= 2:
        cols += [x * y, x * x, y * y]
    return np.column_stack(cols)


def fit_surface(points: pd.DataFrame) -> Optional[Dict]:
    """Robust polynomial error surface over the chips (one median per chip):
    affine, quadratic from 12 chips on; 3-sigma (MAD) clipping."""
    ch = points.groupby('chip')[['E', 'N', 'dE', 'dN']].median()
    if len(ch) < 3:
        return None
    x0, y0 = float(ch['E'].mean()), float(ch['N'].mean())
    s = max(float(np.ptp(ch['E'].values)), float(np.ptp(ch['N'].values)), 1.0)
    x, y = (ch['E'].values - x0) / s, (ch['N'].values - y0) / s
    degree = 2 if len(ch) >= 12 else 1
    keep = np.ones(len(ch), bool)
    ca = cc = None
    for _ in range(5):
        A = _design(x[keep], y[keep], degree)
        if keep.sum() < A.shape[1] + 1:
            break
        ca, *_ = np.linalg.lstsq(A, ch['dE'].values[keep], rcond=None)
        cc, *_ = np.linalg.lstsq(A, ch['dN'].values[keep], rcond=None)
        Af = _design(x, y, degree)
        r = np.hypot(ch['dE'].values - Af @ ca, ch['dN'].values - Af @ cc)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826
        new = r <= max(3.0 * mad + np.median(r[keep]), 1.0)
        if new.sum() == keep.sum() or new.sum() < A.shape[1] + 1:
            break
        keep = new
    if ca is None:
        return None
    try:
        from shapely.geometry import MultiPoint
        hull = MultiPoint(list(zip(ch['E'].values[keep], ch['N'].values[keep]))).convex_hull
    except Exception:
        hull = None
    return {'x0': x0, 'y0': y0, 's': s, 'degree': degree, 'ca': ca, 'cc': cc, 'hull': hull}


def surface_at(surf: Dict, E0: float, N0: float) -> Tuple[float, float]:
    A = _design(np.array([(E0 - surf['x0']) / surf['s']]), np.array([(N0 - surf['y0']) / surf['s']]),
                surf['degree'])
    return float((A @ surf['ca'])[0]), float((A @ surf['cc'])[0])


def local_plane(e: np.ndarray, n: np.ndarray, dE: np.ndarray, dN: np.ndarray, radius_m: float,
                min_points: int = 12) -> Optional[Tuple[float, float]]:
    """(dE, dN) at offset (0, 0) from a robust plane through the neighbours
    (e, n = their offsets from the truth point). A median of the neighbours
    is off by the error gradient times the distance between the truth point
    and their centroid -- 40 m per km east-west on the G1A scenes, so up to
    ~200 m within 5 km. The plane removes that. Along one axis only when the
    neighbours lie on a line; None when they are too few or bunched."""
    if len(dE) < min_points:
        return None
    X = np.column_stack([e, n])
    cov = np.cov(X.T)
    evals, evecs = np.linalg.eigh(cov)
    spread = np.sqrt(np.clip(evals, 0, None))
    if spread[1] < 0.1 * radius_m:
        return None                                  # bunched: nothing to fit a slope to
    if spread[0] < 0.1 * radius_m:                   # on a line: slope along it only
        A = np.column_stack([np.ones(len(e)), X @ evecs[:, 1]])
    else:
        A = np.column_stack([np.ones(len(e)), e, n])
    keep = np.ones(len(e), bool)
    ca = cn = None
    for _ in range(4):
        if keep.sum() < A.shape[1] + 3:
            return None
        ca, *_ = np.linalg.lstsq(A[keep], dE[keep], rcond=None)
        cn, *_ = np.linalg.lstsq(A[keep], dN[keep], rcond=None)
        r = np.hypot(dE - A @ ca, dN - A @ cn)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = r <= max(np.median(r[keep]) + 3.0 * mad, 1.0)
        if new.sum() == keep.sum():
            break
        keep = new
    return float(ca[0]), float(cn[0])


def estimate(points: pd.DataFrame, surf: Optional[Dict], E0: float, N0: float, radius_m: float,
             min_points: int = 5, local: str = 'plane') -> Tuple[Optional[float], Optional[float], str, int]:
    """(dE, dN, method, n used): from the neighbours within radius_m (a local
    plane, or their median when local='median' or no plane fits), else the
    surface, else nothing."""
    d = np.hypot(points['E'].values - E0, points['N'].values - N0)
    near = d <= radius_m
    if near.sum() >= min_points:
        if local == 'plane':
            fit = local_plane(points['E'].values[near] - E0, points['N'].values[near] - N0,
                              points['dE'].values[near], points['dN'].values[near], radius_m)
            if fit is not None:
                return fit[0], fit[1], 'local', int(near.sum())
        return (float(np.median(points['dE'].values[near])), float(np.median(points['dN'].values[near])),
                'local', int(near.sum()))
    if surf is None:
        return None, None, 'none', 0
    dE, dN = surface_at(surf, E0, N0)
    inside = True
    if surf['hull'] is not None:
        from shapely.geometry import Point
        inside = surf['hull'].buffer(radius_m).contains(Point(E0, N0))
    return dE, dN, ('surface' if inside else 'extrapolated'), 0


def score_config(cfg: Dict, truth: pd.DataFrame, radius_m: float, local: str = 'plane') -> Tuple[Dict, List[Dict]]:
    pts = cfg['points']
    surf = fit_surface(pts)
    rows, diffs = [], []
    counts = {'local': 0, 'surface': 0, 'extrapolated': 0, 'none': 0}
    for t in truth.itertuples():
        dE, dN, how, n = estimate(pts, surf, t.E, t.N, radius_m, local=local)
        counts[how] += 1
        row = {'truth_id': t.truth_id, 'truth_E': t.E, 'truth_N': t.N, 'truth_dE_m': t.dE,
               'truth_dN_m': t.dN, 'tool_dE_m': dE, 'tool_dN_m': dN, 'method': how, 'n_points_used': n}
        if dE is not None:
            row['diff_dE_m'], row['diff_dN_m'] = dE - t.dE, dN - t.dN
            row['diff_m'] = float(np.hypot(dE - t.dE, dN - t.dN))
            if how in ('local', 'surface'):
                diffs.append((dE - t.dE, dN - t.dN))
        rows.append(row)
    d = np.asarray(diffs, float).reshape(-1, 2)
    summary = {'truth_reached': len(d), 'truth_total': len(truth), 'n_local': counts['local'],
               'n_surface': counts['surface'], 'n_extrapolated': counts['extrapolated'],
               'n_chips': int(pts['chip'].nunique()), 'n_points': int(len(pts)),
               'truth_rmse_m': float(np.sqrt(np.mean(np.sum(d ** 2, 1)))) if len(d) else None,
               'truth_mean_dE_m': float(d[:, 0].mean()) if len(d) else None,
               'truth_mean_dN_m': float(d[:, 1].mean()) if len(d) else None,
               'truth_max_m': float(np.max(np.hypot(d[:, 0], d[:, 1]))) if len(d) else None}
    return summary, rows


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Configurations that reach at least half of the truth points first; among
    them the lowest RMSE, then the most points reached. (The number reached
    used to come before the RMSE: in job 315 GFTT + NN, 63 km off at all 8
    points, ranked above configurations within 250 m at 6 of 8.)"""
    if df.empty:
        return df
    df = df.copy()
    half = df['truth_total'] / 2.0
    df['_cover'] = (df['truth_reached'] >= half) & df['truth_rmse_m'].notna()
    df['_rmse'] = df['truth_rmse_m'].fillna(np.inf)
    df = df.sort_values(['_cover', '_rmse', 'truth_reached'], ascending=[False, True, False])
    df.insert(0, 'rank', range(1, len(df) + 1))
    return df.drop(columns=['_cover', '_rmse'])


# ─────────────────────────────────────────────────────────────────────────────
# runs
# ─────────────────────────────────────────────────────────────────────────────
def find_runs(output_dir: str) -> List[Dict]:
    """Every (filtered, final) folder pair under output_dir, with its sweep point."""
    runs = []
    for filt in sorted(glob.glob(os.path.join(output_dir, '**', 'filtered_*'), recursive=True)):
        if not os.path.isdir(filt):
            continue
        final = os.path.join(os.path.dirname(filt), 'final_' + os.path.basename(filt)[len('filtered_'):])
        if not os.path.isdir(final):
            continue
        rel = os.path.relpath(filt, output_dir).split(os.sep)
        runs.append({'filtered_dir': filt, 'final_dir': final,
                     'sweep': rel[0] if rel[0].startswith('win') else ''})
    return runs


def consensus_settings(output_dir: str) -> Optional[Dict]:
    """The job's chip-consensus settings (from job_used.json), for rebuilding
    a consensus that did not run. None when the folder has no job file."""
    path = os.path.join(output_dir, 'job_used.json')
    if not os.path.exists(path):
        return None
    import automatch_engine as E
    import automatch_job as J
    with open(path, encoding='utf-8') as f:
        job = J.normalize(json.load(f))
    res = job['target_resolution']
    if not res:
        try:
            res = E.InputScene(job['input_path'], E.PipelineConfig(
                nisar_band=job['nisar_band'], nisar_frequency=job['nisar_frequency'])).native_res
        except Exception:
            res = None
    res = float(res or 10.0)
    return {'tolerance_m': job['consensus_tolerance_m'] or job['consensus_tolerance_px'] * res,
            'mode_bin_m': max(0.1, job['consensus_mode_bin_px'] * res),
            'min_inliers_per_chip': job['min_inliers_per_chip'],
            'min_surviving_chips': job['min_surviving_chips'],
            'model': job['consensus_model'], 'surface_degree': job['consensus_surface']}


def working_crs_of(output_dir: str) -> Optional[str]:
    man = os.path.join(output_dir, 'RUN_MANIFEST.csv')
    if os.path.exists(man):
        m = pd.read_csv(man)
        if 'working_crs' in m.columns and m['working_crs'].notna().any():
            return str(m['working_crs'].dropna().iloc[0])
    job = os.path.join(output_dir, 'job_used.json')
    if os.path.exists(job):
        import automatch_engine as E
        with open(job, encoding='utf-8') as f:
            j = json.load(f)
        sc = E.InputScene(j['input_path'], E.PipelineConfig(nisar_band=j.get('nisar_band', 'auto'),
                                                            nisar_frequency=j.get('nisar_frequency', 'A')))
        return sc.working_crs
    return None


def compare(output_dir: str, truth_csv: str, radius_m: float = 5000.0, chips: str = 'consensus',
            working_crs: Optional[str] = None, write: bool = True, quiet: bool = False,
            runs: Optional[List[Dict]] = None, local: str = 'plane',
            max_points_per_chip: int = 200) -> Dict:
    """Score every configuration of the given runs ({'filtered_dir',
    'final_dir', 'sweep'}; default: every run found under output_dir), write
    the TRUTH_* files and, per channel, the RIVAL CSV of the configuration that
    agrees best with the truth (RIVAL_TRUTH_BEST_<channel>.csv). Returns
    {'by_detector_matcher', 'ranking', 'points', 'truth', 'crs', 'best'}."""
    crs = working_crs or working_crs_of(output_dir)
    if not crs:
        raise ValueError('working CRS unknown: pass working_crs (e.g. EPSG:32640)')
    truth = truth_in_crs(load_truth(truth_csv), crs)
    rows, point_rows = [], []
    chip_files: Dict[Tuple, Tuple[str, List[str]]] = {}
    settings = consensus_settings(output_dir)
    for run in (runs if runs is not None else find_runs(output_dir)):
        timing = {}
        tpath = os.path.join(run['final_dir'], 'PASS_TIMING.csv')
        if os.path.exists(tpath):
            timing = {r['matcher']: r for r in pd.read_csv(tpath).to_dict('records')}
        for key, cfg in config_points(run['filtered_dir'], run['final_dir'], chips, settings).items():
            summary, prow = score_config(cfg, truth, radius_m, local)
            t = timing.get(cfg['matcher'], {})
            summary['pass_seconds'] = t.get('seconds')
            summary['sec_per_window'] = t.get('sec_per_window')
            base = {'channel': cfg['channel'], 'sweep': run['sweep'], 'detector': cfg['detector'],
                    'matcher': cfg['matcher'], 'ransac': cfg['ransac'], 'chips': chips}
            rows.append({**base, **summary, 'filtered_dir': run['filtered_dir']})
            point_rows += [{**base, **r} for r in prow]
            chip_files[(base['channel'], base['sweep'], base['detector'], base['matcher'], base['ransac'])] = \
                (run['filtered_dir'], cfg['files'])
    ranking = rank(pd.DataFrame(rows))
    if not ranking.empty:
        ranking = ranking[[c for c in RANK_COLS if c in ranking.columns]
                          + [c for c in ranking.columns if c not in RANK_COLS]]
        # the best setting (matcher parameter / RANSAC) of each detector + matcher
        fam = ranking['matcher'].str.replace(r'[0-9.]+$', '', regex=True).str.split('_').str[0]
        by = ranking.assign(matcher_family=fam).drop_duplicates(['channel', 'detector', 'matcher_family'])
        by = by.drop(columns=['rank']).rename(columns={'matcher': 'best_matcher_setting'})
        by = rank(by)
        cols = ['rank', 'channel', 'detector', 'matcher_family', 'best_matcher_setting', 'ransac', 'sweep']
        by = by[cols + [c for c in by.columns if c not in cols]]
    else:
        by = ranking
    points = pd.DataFrame(point_rows)
    best: Dict[str, Dict] = {}
    if write:
        by.to_csv(os.path.join(output_dir, 'TRUTH_BY_DETECTOR_MATCHER.csv'), index=False)
        ranking.to_csv(os.path.join(output_dir, 'TRUTH_RANKING.csv'), index=False)
        points.to_csv(os.path.join(output_dir, 'TRUTH_POINTS.csv'), index=False)
        best = write_truth_best(by, chip_files, crs, output_dir, max_points_per_chip)
    if not quiet:
        print(format_ranking(by, len(truth), radius_m, chips, local))
        for ch, b in best.items():
            print(f"[Truth] best for {ch}: {b['detector']} + {b['matcher']} {b['ransac']} -> {b['csv']}")
    return {'by_detector_matcher': by, 'ranking': ranking, 'points': points, 'truth': truth, 'crs': crs,
            'best': best}


def write_truth_best(by: pd.DataFrame, chip_files: Dict, crs: str, output_dir: str,
                     max_points_per_chip: int = 200) -> Dict[str, Dict]:
    """RIVAL_TRUTH_BEST_<channel>.csv (+ _detail.csv): the points of the
    configuration ranked first against the truth, for DPQED_rival.py."""
    import automatch_rival as R
    out: Dict[str, Dict] = {}
    if by.empty:
        return out
    for ch, g in by.groupby('channel', sort=False):
        top = g.sort_values('rank').iloc[0]
        if pd.isna(top.get('truth_rmse_m')):
            continue
        key = (top['channel'], top['sweep'], top['detector'], top['best_matcher_setting'], top['ransac'])
        if key not in chip_files:
            continue
        filtered_dir, files = chip_files[key]
        pts = R.thin_per_chip(R.collect_points(filtered_dir, files), max_points_per_chip)
        if pts.empty:
            continue
        path = os.path.join(output_dir, f'RIVAL_TRUTH_BEST_{ch}.csv')
        R.write_rival_csv(pts, crs, path, path[:-4] + '_detail.csv')
        out[str(ch)] = {'csv': path, 'detector': top['detector'], 'matcher': top['best_matcher_setting'],
                        'ransac': top['ransac'], 'sweep': top['sweep'],
                        'truth_rmse_m': float(top['truth_rmse_m'])}
    return out


def score_detail_csv(detail_csv: str, truth: pd.DataFrame, radius_m: float = 5000.0,
                     local: str = 'plane') -> Dict:
    """Truth metrics of one exported point set (a run's RIVAL _detail.csv:
    In/Ref in the working CRS, one chip per source_file)."""
    d = pd.read_csv(detail_csv, encoding='utf-8-sig')
    if d.empty:
        return {}
    pts = pd.DataFrame({'E': d['In_X'].astype(float), 'N': d['In_Y'].astype(float),
                        'dE': d['In_X'].astype(float) - d['Ref_X'].astype(float),
                        'dN': d['In_Y'].astype(float) - d['Ref_Y'].astype(float),
                        'chip': d['source_file'] if 'source_file' in d.columns else 0})
    summary, _ = score_config({'points': pts}, truth, radius_m, local)
    return summary


def format_ranking(by: pd.DataFrame, n_truth: int, radius_m: float, chips: str, local: str = 'plane',
                   top: int = 20) -> str:
    if by.empty:
        return '[Truth] no configuration to score'
    how = 'a local plane through' if local == 'plane' else 'the median of'
    lines = [f'[Truth] {n_truth} ground-truth point(s); tool error from {how} its matches within '
             f'{radius_m / 1000:g} km (else its error surface); {chips} chips. Best detector + matcher:',
             f"  {'#':>3} {'channel':<8} {'detector':<26} {'matcher':<7} {'setting':<14} {'RMSE m':>8} "
             f"{'bias dE':>8} {'bias dN':>8} {'max m':>8} {'reached':>8} {'s/window':>9}"]
    for r in by.head(top).itertuples():
        f = (lambda v: f'{v:8.1f}' if v is not None and not pd.isna(v) else f"{'-':>8}")
        lines.append(f"  {r.rank:>3} {str(r.channel):<8} {str(r.detector):<26} {str(r.matcher_family):<7} "
                     f"{str(r.best_matcher_setting) + ' ' + str(r.ransac):<14} {f(r.truth_rmse_m)} "
                     f"{f(r.truth_mean_dE_m)} {f(r.truth_mean_dN_m)} {f(r.truth_max_m)} "
                     f"{r.truth_reached:>4}/{r.truth_total:<3} "
                     f"{f(getattr(r, 'sec_per_window', None))}")
    return '\n'.join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Rank the detector + matcher configurations of an automatch '
                                             'output folder against manually measured ground truth.')
    ap.add_argument('output_dir')
    ap.add_argument('--truth', required=True, help='RIVAL CSV of manually measured points')
    ap.add_argument('--radius-km', type=float, default=5.0,
                    help='truth point to tool points: local neighbourhood (default 5 km)')
    ap.add_argument('--chips', choices=['consensus', 'all'], default='consensus')
    ap.add_argument('--crs', default='', help="working CRS if the folder does not record it (EPSG:326xx)")
    ap.add_argument('--local', choices=['plane', 'median'], default='plane',
                    help='tool error at a truth point: plane through its neighbours (default; follows the '
                         'error gradient) or their median (the earlier rule)')
    a = ap.parse_args(argv)
    res = compare(a.output_dir, a.truth, a.radius_km * 1000.0, a.chips, a.crs or None, local=a.local)
    import automatch_job as J
    more = J.refresh_after_compare(a.output_dir, res)
    print(f"\n[Truth] written: {os.path.join(a.output_dir, 'TRUTH_BY_DETECTOR_MATCHER.csv')}, "
          f"TRUTH_RANKING.csv, TRUTH_POINTS.csv" + ''.join(f', {os.path.basename(p)}' for p in more))
    return 0 if not res['ranking'].empty else 1


if __name__ == '__main__':
    sys.exit(main())
