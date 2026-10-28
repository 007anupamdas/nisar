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
    TRUTH_MATCH_ACCURACY.csv        match by match: per configuration and sweep point
                                    (window size, keypoints), the share of each window's
                                    matches that are right, before and after RANSAC

Match by match: the truth points give the scene's error field (affine, or
quadratic when it predicts left-out points better). A match is good when its
displacement lies within its RANSAC threshold (or twice the field's own
leave-one-out error, if larger) of the field at its position. Every window a
configuration matched counts, not only the consensus survivors, so a sweep
over window sizes or num_features shows how the share of good matches
changes with them (printed as a table when the run swept several). Before
RANSAC comes from the raw_matches_* files (the filtered ones hold only the
inliers).

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
MAX_TRUTH_ERROR_KM = 500.0     # a point further off than this is a broken row, not an error


def _coord(df: pd.DataFrame, col: str, lim: float) -> pd.Series:
    """A lon/lat column as numbers; blank, text ('-', 'NA'), exactly 0 (an
    unset RIVAL point) and out-of-range values become NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    v = pd.to_numeric(df[col], errors='coerce')
    return v.where((v != 0) & (v.abs() <= lim))


def _xy(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    v = pd.to_numeric(df[col], errors='coerce')
    return v


def _fill_from_xy(lon: pd.Series, lat: pd.Series, x: pd.Series, y: pd.Series,
                  known: List[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]) -> Tuple[pd.Series, pd.Series, List]:
    """Missing lon/lat of a point from its projected X/Y, through the X/Y ->
    lon/lat mapping fitted on the file's own complete points (input and
    reference points share RIVAL's X/Y grid). Quadratic with 8+ known
    points, else affine with 4+; used only when it fits them within ~20 m."""
    gx = np.concatenate([k[0].values for k in known])
    gy = np.concatenate([k[1].values for k in known])
    glo = np.concatenate([k[2].values for k in known])
    gla = np.concatenate([k[3].values for k in known])
    ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(glo) & np.isfinite(gla)
    want = (lon.isna() | lat.isna()) & x.notna() & y.notna() & ((x != 0) | (y != 0))
    if not want.any() or ok.sum() < 4:
        return lon, lat, []
    gx, gy, glo, gla = gx[ok], gy[ok], glo[ok], gla[ok]
    mx, my = gx.mean(), gy.mean()
    sc = max(np.ptp(gx), np.ptp(gy), 1.0)

    def design(a, b):
        u, v = (a - mx) / sc, (b - my) / sc
        cols = [np.ones_like(u), u, v] + ([u * u, u * v, v * v] if ok.sum() >= 8 else [])
        return np.column_stack(cols)
    A = design(gx, gy)
    clo = np.linalg.lstsq(A, glo, rcond=None)[0]
    cla = np.linalg.lstsq(A, gla, rcond=None)[0]
    resid = max(np.abs(A @ clo - glo).max(), np.abs(A @ cla - gla).max())
    if resid > 2e-4:                      # ~20 m: the X/Y grid is not one smooth map of lon/lat
        return lon, lat, []
    B = design(x[want].values, y[want].values)
    lon, lat = lon.copy(), lat.copy()
    lon[want], lat[want] = B @ clo, B @ cla
    return lon, lat, list(want[want].index)


def load_truth(path: str, max_error_km: float = MAX_TRUTH_ERROR_KM) -> pd.DataFrame:
    """Manual points: truth_id, in_lon, in_lat, ref_lon, ref_lat.

    RIVAL exports sometimes leave a point's Ref_Lon / Ref_Lat empty, 0 or
    text: such a row used to count as an error of thousands of km (a 0) or
    stop the file (text). A missing lon/lat is recovered from the row's
    X/Y when the file's other points give the X/Y -> lon/lat mapping; a row
    still incomplete, or with input and reference more than max_error_km
    apart, is left out, with a note naming it."""
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    name = os.path.basename(path)
    missing = [c for c in TRUTH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{name}: ground truth needs the columns {TRUTH_COLUMNS} '
                         f'(missing {missing}); RIVAL "Export CSV" files have them')
    ids = df['Row'].tolist() if 'Row' in df.columns else list(range(1, len(df) + 1))
    ilo, ila = _coord(df, 'In_Lon', 180), _coord(df, 'In_Lat', 90)
    rlo, rla = _coord(df, 'Ref_Lon', 180), _coord(df, 'Ref_Lat', 90)
    ix, iy, rx, ry = _xy(df, 'In_X'), _xy(df, 'In_Y'), _xy(df, 'Ref_X'), _xy(df, 'Ref_Y')
    known = [(ix, iy, ilo, ila), (rx, ry, rlo, rla)]
    rlo, rla, fixed_r = _fill_from_xy(rlo, rla, rx, ry, known)
    ilo, ila, fixed_i = _fill_from_xy(ilo, ila, ix, iy, known)
    for idx in fixed_r:
        print(f'[Truth] {name}: point {ids[idx]} had no Ref_Lon/Ref_Lat; taken from its Ref_X/Ref_Y')
    for idx in fixed_i:
        print(f'[Truth] {name}: point {ids[idx]} had no In_Lon/In_Lat; taken from its In_X/In_Y')
    keep = ilo.notna() & ila.notna() & rlo.notna() & rla.notna()
    for idx in df.index[~keep]:
        gone = [c for c, v in (('In_Lon', ilo), ('In_Lat', ila), ('Ref_Lon', rlo), ('Ref_Lat', rla))
                if pd.isna(v[idx])]
        print(f'[Truth] WARNING {name}: point {ids[idx]} left out: no usable {", ".join(gone)}')
    # input and reference of one point are a geolocation error apart, never continents
    lat0 = np.radians((ila + rla) / 2.0)
    dist_km = np.hypot((ilo - rlo) * 111.32 * np.cos(lat0), (ila - rla) * 110.57)
    far = keep & (dist_km > max_error_km)
    for idx in df.index[far]:
        print(f'[Truth] WARNING {name}: point {ids[idx]} left out: input and reference are '
              f'{dist_km[idx]:.0f} km apart (more than {max_error_km:g} km: a broken row)')
    keep &= ~far
    if not keep.any():
        raise ValueError(f'{name}: no complete rows')
    return pd.DataFrame({'truth_id': [ids[i] for i in df.index[keep]],
                         'in_lon': ilo[keep].astype(float).values, 'in_lat': ila[keep].astype(float).values,
                         'ref_lon': rlo[keep].astype(float).values, 'ref_lat': rla[keep].astype(float).values})


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


def ransac_label(method, threshold) -> str:
    """magsac40; a sub-cell variant (method magsac-sc512) is magsac40-sc512."""
    m = str(method)
    base, _, sub = m.partition('-sc')
    return f"{base}{float(threshold or 0):g}" + (f'-sc{sub}' if sub else '')


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
            'ransac': ransac_label(p.get('ransac_method'), p.get('ransac_threshold', 0)),
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


# ─────────────────────────────────────────────────────────────────────────────
# match by match: how many of a window's matches are right
# ─────────────────────────────────────────────────────────────────────────────
def truth_field(truth: pd.DataFrame, max_drop: int = 3) -> Optional[Dict]:
    """The scene's true error (dE, dN) everywhere, from the truth points: an
    affine field, or a quadratic one when it predicts left-out points clearly
    better (by 0.8, from 9 points on). Up to max_drop points whose
    leave-one-out residual is over 3 x the median are dropped (a mis-marked
    point), keeping 6. loo_rmse_m says how well the field knows the error
    between the points. None with fewer than 4 points."""
    E, N = truth['E'].to_numpy(float), truth['N'].to_numpy(float)
    dE, dN = truth['dE'].to_numpy(float), truth['dN'].to_numpy(float)
    if len(E) < 4:
        return None
    x0, y0 = float(E.mean()), float(N.mean())
    s = max(float(np.ptp(E)), float(np.ptp(N)), 1.0)
    x, y = (E - x0) / s, (N - y0) / s

    def loo(keep, degree):
        idx = np.flatnonzero(keep)
        need = 6 if degree == 2 else 3
        if len(idx) < need + 2:
            return None
        r = np.full(len(E), np.nan)
        for i in idx:
            m = keep.copy()
            m[i] = False
            A = _design(x[m], y[m], degree)
            ca, *_ = np.linalg.lstsq(A, dE[m], rcond=None)
            cc, *_ = np.linalg.lstsq(A, dN[m], rcond=None)
            a = _design(x[i:i + 1], y[i:i + 1], degree)
            r[i] = float(np.hypot((a @ ca)[0] - dE[i], (a @ cc)[0] - dN[i]))
        return r

    keep = np.ones(len(E), bool)
    dropped = []
    while True:
        r1 = loo(keep, 1)
        if r1 is None:
            return None
        r2 = loo(keep, 2) if keep.sum() >= 9 else None
        rm1 = float(np.sqrt(np.nanmean(r1[keep] ** 2)))
        rm2 = float(np.sqrt(np.nanmean(r2[keep] ** 2))) if r2 is not None else np.inf
        degree, r, rmse = (2, r2, rm2) if rm2 < 0.8 * rm1 and rm1 - rm2 > 1.0 else (1, r1, rm1)
        worst = int(np.nanargmax(np.where(keep, r, -1)))
        if (len(dropped) < max_drop and keep.sum() > 6
                and r[worst] > 3.0 * float(np.nanmedian(r[keep])) and r[worst] > 1.0):
            # a mis-marked point also inflates its neighbours' leave-one-out
            # residuals (an edge point's most): drop the point whose removal
            # leaves the best-fitting set
            best, best_rm = None, np.inf
            for i in np.flatnonzero(keep):
                m = keep.copy()
                m[i] = False
                rr = loo(m, degree)
                if rr is None:
                    continue
                rm = float(np.sqrt(np.nanmean(rr[m] ** 2)))
                if rm < best_rm:
                    best, best_rm = int(i), rm
            if best is None:
                break
            keep[best] = False
            dropped.append(int(truth['truth_id'].iloc[best]) if 'truth_id' in truth.columns else best)
            continue
        break
    A = _design(x[keep], y[keep], degree)
    ca, *_ = np.linalg.lstsq(A, dE[keep], rcond=None)
    cc, *_ = np.linalg.lstsq(A, dN[keep], rcond=None)
    zone = None
    try:
        from shapely.geometry import MultiPoint
        zone = MultiPoint(list(zip(E[keep], N[keep]))).convex_hull
    except Exception:
        pass
    return {'x0': x0, 'y0': y0, 's': s, 'degree': degree, 'ca': ca, 'cc': cc, 'loo_rmse_m': rmse,
            'n_used': int(keep.sum()), 'dropped': dropped, 'hull': zone, 'E': E[keep], 'N': N[keep]}


def field_at(field: Dict, E: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = _design((np.asarray(E, float) - field['x0']) / field['s'], (np.asarray(N, float) - field['y0']) / field['s'],
                field['degree'])
    return A @ field['ca'], A @ field['cc']


def field_covers(field: Dict, E: np.ndarray, N: np.ndarray, radius_m: float) -> np.ndarray:
    """Points where the field is known: within radius_m of the truth points'
    convex hull (further out it would be extrapolated)."""
    E, N = np.asarray(E, float), np.asarray(N, float)
    if field.get('hull') is not None:
        try:
            import shapely
            return np.asarray(shapely.contains_xy(field['hull'].buffer(radius_m), E, N), bool)
        except Exception:
            pass
    out = np.zeros(len(E), bool)
    for te, tn in zip(field['E'], field['N']):
        out |= np.hypot(E - te, N - tn) <= radius_m
    return out


def match_accuracy(filtered_dir: str, field: Dict, radius_m: float = 5000.0,
                   raw_dir: Optional[str] = None, misfits: Optional[Dict] = None) -> List[Dict]:
    """Per configuration (channel, detector, matcher, RANSAC) of one run, how
    many of each window's matches are right: a match is good when its
    displacement is within tolerance of the truth field at its position;
    tolerance = the window's RANSAC threshold, or twice the field's own
    leave-one-out error if that is larger. Over every window the matcher
    produced (not only the consensus survivors), before RANSAC ('all') and
    after it ('inliers'). The filtered files hold only RANSAC's inliers; all
    matches come from the run's raw_matches_* folder beside them, and a
    window RANSAC kept nothing of counts with 0 inliers. Matches outside the
    truth points' area are not judged."""
    import automatch_engine as E

    misfits = {} if misfits is None else misfits      # {file: (matches, misfit of the judged ones)}

    def score(path, cols, thr):
        """(n, judged, good) of a match file, judged at tolerance thr (each
        file read once, whatever the RANSAC settings asking)."""
        if path not in misfits:
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.DataFrame()
            x1, y1, x2, y2 = cols
            if df.empty or not set(cols) <= set(df.columns):
                misfits[path] = (len(df), np.zeros(0))
            else:
                e1, n1 = df[x1].to_numpy(float), df[y1].to_numpy(float)
                tE, tN = field_at(field, e1, n1)
                res = np.hypot(e1 - df[x2].to_numpy(float) - tE, n1 - df[y2].to_numpy(float) - tN)
                misfits[path] = (len(df), res[field_covers(field, e1, n1, radius_m)])
        n, res = misfits[path]
        return n, len(res), int((res <= max(thr, 2.0 * field['loo_rmse_m'])).sum())

    def meta(name):
        p = E.MatchStatistics._parse_filename(name) or {}
        return (p.get('nisar_pol'), _variant(name), _matcher_label(p)), p

    filt: Dict[str, Dict[str, Tuple[str, float]]] = {}   # window -> RANSAC label -> (file, threshold)
    cfg_of: Dict[str, Tuple] = {}
    ransacs: Dict[Tuple, Dict[str, float]] = {}
    for path in sorted(glob.glob(os.path.join(filtered_dir, '*.csv'))):
        name = os.path.basename(path)
        cfg, p = meta(name)
        if not p or '_aff_' not in name:
            continue
        win = name[:name.index('_aff_')]
        ran = ransac_label(p.get('ransac_method'), p.get('ransac_threshold', 0))
        thr = float(p.get('ransac_threshold') or 0)
        filt.setdefault(win, {})[ran] = (path, thr)
        cfg_of[win] = cfg
        ransacs.setdefault(cfg, {})[ran] = thr
    raw_dir = raw_dir or os.path.join(os.path.dirname(filtered_dir),
                                      'raw_matches_' + os.path.basename(filtered_dir)[len('filtered_'):])
    raws = {os.path.basename(p)[:-len('_raw.csv')]: p
            for p in glob.glob(os.path.join(raw_dir, '*_raw.csv'))} if os.path.isdir(raw_dir) else {}
    every_thr = sorted({t for d in ransacs.values() for t in d.values()})
    acc: Dict[Tuple, List] = {}
    for win in sorted(set(filt) | set(raws)):
        cfg = cfg_of.get(win)
        if cfg is None:                                 # RANSAC kept nothing in this window
            cfg, p = meta(f'{win}_aff_none_0_0_.csv')
            if not p:
                continue
        labels = ransacs.get(cfg) or {f'-{t:g}': t for t in every_thr[:1]} or {'-': 0.0}
        for ran, thr in labels.items():
            n_all, j_all, g_all = (score(raws[win], ('x1_map', 'y1_map', 'x2_map', 'y2_map'), thr)
                                   if win in raws else (0, 0, 0))
            got = filt.get(win, {}).get(ran)
            n_in, j_in, g_in = (score(got[0], ('x1-map', 'y1-map', 'x2-map', 'y2-map'), got[1])
                                if got else (0, 0, 0))
            if win not in raws:                         # no raw file: the inliers are all that is known
                n_all, j_all, g_all = n_in, j_in, g_in
            acc.setdefault((*cfg, ran), []).append((n_all, j_all, g_all, n_in, j_in, g_in))
    rows = []
    for (ch, det, mat, ran), v in acc.items():
        a = np.asarray(v, float).reshape(-1, 6)
        seen = a[:, 1] > 0                                   # windows with judged matches
        seen_inl = a[:, 4] > 0
        share = np.divide(a[:, 5], a[:, 4], out=np.zeros(len(a)), where=seen_inl)
        rows.append({'channel': ch, 'detector': det, 'matcher': mat, 'ransac': ran,
                     'windows': int(len(a)), 'windows_judged': int(seen.sum()),
                     'matches_per_window': float(np.median(a[:, 0])) if len(a) else 0.0,
                     'precision_all': round(float(a[:, 2].sum() / a[:, 1].sum()), 4) if a[:, 1].sum() else None,
                     'inliers_per_window': float(np.median(a[:, 3])) if len(a) else 0.0,
                     'precision_inliers': round(float(a[:, 5].sum() / a[:, 4].sum()), 4) if a[:, 4].sum() else None,
                     'good_inliers_per_window': float(np.median(a[seen, 5])) if seen.any() else 0.0,
                     'windows_90pct_good': round(float((share[seen_inl] >= 0.9).mean()), 4) if seen_inl.any() else None,
                     'windows_mostly_wrong': round(float((share[seen_inl] < 0.5).mean()), 4) if seen_inl.any() else None,
                     'windows_no_inliers': round(float((a[seen, 3] < 3).mean()), 4) if seen.any() else None})
    return rows


def sweep_parts(sweep: str) -> Tuple[Optional[int], str]:
    """(window size, keypoints) of a sweep tag such as win512_nf10."""
    import re
    w = re.search(r'win(\d+)', str(sweep or ''))
    nf = re.search(r'nf(\w+)', str(sweep or ''))
    return (int(w.group(1)) if w else None), (nf.group(1) if nf else '')


def format_by_sweep(acc: pd.DataFrame, top: int = 15) -> str:
    """Precision of the RANSAC inliers (and good inliers per window) of each
    configuration at each sweep point -- window size and / or keypoint count
    -- when the run swept several."""
    if acc.empty:
        return ''
    num = (lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v)))
    points = sorted({(w, n) for w, n in zip(acc['window_size'], acc['num_features'])},
                    key=lambda p: (num(p[0]), num(p[1])))
    if len(points) < 2:
        return ''
    wins = {w for w, _ in points}
    nfs = {n for _, n in points}
    name = (lambda w, n: f'nf {n}') if len(wins) == 1 else (
        (lambda w, n: f'{w} px') if len(nfs) == 1 else (lambda w, n: f'{w} px/{n}'))
    what = ('keypoints per window' if len(wins) == 1 else
            'window size' if len(nfs) == 1 else 'window size / keypoints per window')
    a = acc.assign(cfg=acc['channel'].astype(str) + ' ' + acc['detector'].astype(str) + ' + '
                   + acc['matcher'].astype(str) + ' ' + acc['ransac'].astype(str),
                   pt=list(zip(acc['window_size'], acc['num_features'])))
    order = a.groupby('cfg')['precision_inliers'].mean().sort_values(ascending=False).index[:top]
    lines = [f'[Truth] share of good matches among the RANSAC inliers (good inliers per window), by {what}:',
             f"  {'configuration':<52}" + ''.join(f'{name(*p):>16}' for p in points)]
    for cfg in order:
        g = a[a['cfg'] == cfg]
        cells = []
        for p in points:
            r = g[g['pt'] == p]
            if len(r) and pd.notna(r['precision_inliers'].iloc[0]):
                r = r.iloc[0]
                cells.append(f"{100 * r['precision_inliers']:5.0f}% ({r['good_inliers_per_window']:g})")
            else:
                cells.append('-')
        lines.append(f'  {cfg[:52]:<52}' + ''.join(f'{c:>16}' for c in cells))
    return '\n'.join(lines)


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
        if rel[0].startswith('SUBCELLS_'):          # made afterwards from this job's matches (automatch_accuracy)
            continue
        runs.append({'filtered_dir': filt, 'final_dir': final,
                     'sweep': rel[0] if rel[0].startswith('win') else ''})
    return runs


def job_resolution(output_dir: str) -> Optional[float]:
    """The working pixel size of a finished job: its target_resolution, else
    the 'working resolution' its log names (the input may not be reachable
    from here), else the input's own."""
    import re
    job = {}
    try:
        with open(os.path.join(output_dir, 'job_used.json'), encoding='utf-8') as f:
            job = json.load(f)
    except (OSError, ValueError):
        pass
    if job.get('target_resolution'):
        return float(job['target_resolution'])
    try:
        with open(os.path.join(output_dir, 'automatch.log'), encoding='utf-8', errors='replace') as f:
            m = re.search(r'working resolution ([\d.]+) m', f.read(400000))
        if m:
            return float(m.group(1))
    except OSError:
        pass
    if job.get('input_path'):
        try:
            import automatch_engine as E
            return float(E.InputScene(job['input_path'], E.PipelineConfig(
                nisar_band=job.get('nisar_band', 'auto'), nisar_frequency=job.get('nisar_frequency', 'A'))).native_res)
        except Exception:
            pass
    return None


def consensus_settings(output_dir: str) -> Optional[Dict]:
    """The job's chip-consensus settings (from job_used.json), for rebuilding
    a consensus that did not run. None when the folder has no job file."""
    path = os.path.join(output_dir, 'job_used.json')
    if not os.path.exists(path):
        return None
    import automatch_job as J
    with open(path, encoding='utf-8') as f:
        job = J.normalize(json.load(f))
    res = float(job_resolution(output_dir) or 10.0)
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
            max_points_per_chip: int = 200, rebuild: bool = True, progress: str = '') -> Dict:
    """Score every configuration of the given runs ({'filtered_dir',
    'final_dir', 'sweep'}; default: every run found under output_dir), write
    the TRUTH_* files and, per channel, the RIVAL CSV of the configuration that
    agrees best with the truth (RIVAL_TRUTH_BEST_<channel>.csv). Returns
    {'by_detector_matcher', 'ranking', 'points', 'truth', 'crs', 'best'}.
    rebuild=False: runs whose chip consensus is missing are not rebuilt (and
    nothing is written into them); their matches are still scored match by
    match. progress: a line per run, starting with it. Each raw match file
    is read once, however many runs (plain, sub-cell variants) use it."""
    crs = working_crs or working_crs_of(output_dir)
    if not crs:
        raise ValueError('working CRS unknown: pass working_crs (e.g. EPSG:32640)')
    truth = truth_in_crs(load_truth(truth_csv), crs)
    rows, point_rows, acc_rows = [], [], []
    chip_files: Dict[Tuple, Tuple[str, List[str]]] = {}
    settings = consensus_settings(output_dir) if rebuild else None
    field = truth_field(truth)
    misfits: Dict = {}                                 # raw match files scored once, shared by the runs
    run_list = runs if runs is not None else find_runs(output_dir)
    for k, run in enumerate(run_list, 1):
        if progress:
            print(f"[Truth] {progress}scoring run {k}/{len(run_list)}: "
                  f"{os.path.relpath(run['filtered_dir'], output_dir)}", flush=True)
        if field is not None:
            win, nf = sweep_parts(run['sweep'])
            try:
                acc_rows += [{'sweep': run['sweep'], 'window_size': win, 'num_features': nf, **r}
                             for r in match_accuracy(run['filtered_dir'], field, radius_m, run.get('raw_dir'),
                                                     misfits)]
            except Exception as e:
                print(f"[Truth] match accuracy of {run['filtered_dir']}: {type(e).__name__}: {e}")
        timing = {}
        tpath = os.path.join(run['final_dir'], 'PASS_TIMING.csv')
        if os.path.exists(tpath):
            try:
                timing = {r['matcher']: r for r in pd.read_csv(tpath).to_dict('records')}
            except Exception:                         # empty or cut off: no timing, still scored
                timing = {}
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
    acc = pd.DataFrame(acc_rows)
    if not acc.empty:            # 0: the plain result; else the sub-cell size of a <method>-sc<px> variant
        acc.insert(acc.columns.get_loc('ransac') + 1, 'subcell_px',
                   acc['ransac'].astype(str).str.extract(r'-sc(\d+)$')[0].fillna(0).astype(int))
    keys = ['channel', 'sweep', 'detector', 'matcher', 'ransac']
    acc_cols = ['precision_inliers', 'windows_90pct_good', 'windows_mostly_wrong', 'good_inliers_per_window',
                'precision_all']
    table = pd.DataFrame(rows)
    if not acc.empty and not table.empty:
        k = table[keys].astype(str).agg('|'.join, axis=1)
        a = acc.set_index(acc[keys].astype(str).agg('|'.join, axis=1))[acc_cols]
        a = a[~a.index.duplicated()]
        for c in acc_cols:
            table[c] = k.map(a[c]).values
    ranking = rank(table)
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
    if not acc.empty and not ranking.empty:
        k = acc[keys].astype(str).agg('|'.join, axis=1)
        r = ranking.set_index(ranking[keys].astype(str).agg('|'.join, axis=1))
        r = r[~r.index.duplicated()]
        acc['truth_rmse_m'] = k.map(r['truth_rmse_m']).values
        acc['truth_reached'] = k.map(r['truth_reached']).values
    if not acc.empty:
        acc = acc.sort_values(['channel', 'detector', 'matcher', 'ransac', 'window_size', 'num_features'],
                              key=lambda s: s.map(lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v))))
    best: Dict[str, Dict] = {}
    if write:
        # with nothing to rank, a header only: an empty file stops whatever reads it later (job 313)
        (by if len(by.columns) else pd.DataFrame(columns=RANK_COLS)).to_csv(
            os.path.join(output_dir, 'TRUTH_BY_DETECTOR_MATCHER.csv'), index=False)
        (ranking if len(ranking.columns) else pd.DataFrame(columns=RANK_COLS)).to_csv(
            os.path.join(output_dir, 'TRUTH_RANKING.csv'), index=False)
        (points if len(points.columns) else pd.DataFrame(columns=['channel', 'sweep', 'detector', 'matcher',
                                                                  'ransac', 'truth_id'])).to_csv(
            os.path.join(output_dir, 'TRUTH_POINTS.csv'), index=False)
        if not acc.empty:
            acc.to_csv(os.path.join(output_dir, 'TRUTH_MATCH_ACCURACY.csv'), index=False)
        best = write_truth_best(by, chip_files, crs, output_dir, max_points_per_chip)
    if not quiet:
        print(format_ranking(by, len(truth), radius_m, chips, local))
        if field is not None:
            print(f"[Truth] match by match: the truth points give "
                  f"{'a quadratic' if field['degree'] == 2 else 'an affine'} error field ({field['n_used']} points, leave-one-out {field['loo_rmse_m']:.0f} m"
                  + (f"; dropped point(s) {field['dropped']}" if field['dropped'] else '')
                  + f"); a match is good within max(RANSAC threshold, {2 * field['loo_rmse_m']:.0f} m) of it "
                    f"(TRUTH_MATCH_ACCURACY.csv)")
            nf_table = format_by_sweep(acc) if not acc.empty else ''
            if nf_table:
                print(nf_table)
        for ch, b in best.items():
            print(f"[Truth] best for {ch}: {b['detector']} + {b['matcher']} {b['ransac']} -> {b['csv']}")
    return {'by_detector_matcher': by, 'ranking': ranking, 'points': points, 'truth': truth, 'crs': crs,
            'best': best, 'truth_csv': truth_csv, 'accuracy': acc, 'field': field}


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
    lines += coverage_warnings(by)
    return '\n'.join(lines)


def coverage_warnings(by: pd.DataFrame, min_points: int = 2, factor: float = 3.0,
                      floor_m: float = 1000.0) -> List[str]:
    """A warning per channel whose first-ranked configuration is more than
    floor_m off while one reaching fewer points (at least min_points) is
    factor times closer: the matches then cover only part of the scene, and
    the configurations reaching half of the points do it through wrong chips
    (job 343: 1.1 km at 5 of 9 points, against 179-480 m at 2-4 of 9)."""
    out = []
    if by.empty or 'channel' not in by.columns:
        return out
    for ch, g in by.groupby('channel', sort=False):
        top = g.iloc[0]
        if pd.isna(top['truth_rmse_m']) or top['truth_rmse_m'] <= floor_m:
            continue
        few = g[(g['truth_reached'] >= min_points) & g['truth_rmse_m'].notna()]
        if few.empty:
            continue
        alt = few.loc[few['truth_rmse_m'].idxmin()]
        if alt['truth_rmse_m'] * factor > top['truth_rmse_m']:
            continue
        out.append(f"[Truth] WARNING {ch}: rank 1 ({top['detector']} + {top['matcher_family']}) is "
                   f"{top['truth_rmse_m']:.0f} m at {top['truth_reached']}/{top['truth_total']} points, "
                   f"while {alt['detector']} + {alt['matcher_family']} is {alt['truth_rmse_m']:.0f} m at "
                   f"{alt['truth_reached']}/{alt['truth_total']}. The matches cover only part of the scene "
                   f"(clouds, water or a narrow overlap); no configuration is reliable across it, and "
                   f"RIVAL_BEST follows rank 1.")
    return out


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
