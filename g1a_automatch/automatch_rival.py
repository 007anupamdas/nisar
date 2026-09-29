#!/usr/bin/env python3
"""
automatch_rival -- write automatch results as CSV files DPQED_rival.py loads.

The file is byte-for-byte the layout of RIVAL's own 'Export CSV':

    In_X, In_Y, Ref_X, Ref_Y, DX_Err, DY_Err, Row, In_Lon, In_Lat, Ref_Lon, Ref_Lat

In_* is the image being assessed (G1A / NISAR), Ref_* the reference, both in
the working CRS (the input's projected CRS, or the UTM zone of its centre when
it is in lon/lat -- the same rule RIVAL applies). DX/DY = In - Ref in metres.
RIVAL's 'Load CSV' reads In_X/In_Y/Ref_X/Ref_Y by name and recomputes the
errors, RMSE and CE90 itself; the lon/lat columns make the file meaningful
without knowing the grid. Rows are written with RIVAL's csv_record and the
summary uses RIVAL's accuracy_stats (both ported verbatim in automatch_refs).

A second '<name>_detail.csv' carries the same eleven columns followed by
provenance (detector, matcher, RANSAC setting, pair, chip) for analysis.
"""

import csv
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import automatch_refs as refs

RIVAL_HEADER = ["In_X", "In_Y", "Ref_X", "Ref_Y", "DX_Err", "DY_Err",
                "Row", "In_Lon", "In_Lat", "Ref_Lon", "Ref_Lat"]
DETAIL_EXTRA = ["channel", "reference", "detector", "matcher", "match_parameter",
                "ransac_method", "ransac_threshold", "ransac_confidence",
                "pair_id", "scan", "pix", "source_file"]


def _parse(name: str) -> Optional[Dict]:
    import automatch_engine as E
    return E.MatchStatistics._parse_filename(name)


def collect_points(filtered_dir: str, filenames: List[str]) -> pd.DataFrame:
    """Inlier correspondences of the given filtered CSVs, with provenance."""
    frames = []
    for name in filenames:
        path = os.path.join(filtered_dir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'inliers' in df.columns:
            df = df[df['inliers'].astype(str).str.lower().str.contains('true')]
        if df.empty:
            continue
        p = _parse(name) or {}
        df = df.assign(source_file=name, detector=p.get('detector'),
                       matcher=p.get('match_method'), match_parameter=p.get('match_parameter'),
                       ransac_method=p.get('ransac_method'), ransac_threshold=p.get('ransac_threshold'),
                       ransac_confidence=p.get('ransac_confidence'), pair_id=p.get('pair_id'),
                       scan=p.get('scan'), pix=p.get('pix'), channel=p.get('nisar_pol'),
                       reference=p.get('s1_ref_tag'))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # The same correspondence can survive in several windows' RANSAC sets.
    return out.drop_duplicates(subset=['x1-map', 'y1-map', 'x2-map', 'y2-map'])


def thin_per_chip(df: pd.DataFrame, max_per_chip: int, seed: int = 0) -> pd.DataFrame:
    """At most max_per_chip points per chip, spread over the chip: points are
    binned on a grid and taken one per cell in turn. 0 keeps everything."""
    if max_per_chip <= 0 or df.empty:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby('source_file', sort=False):
        if len(g) <= max_per_chip:
            keep.append(g)
            continue
        k = max(1, int(math.ceil(math.sqrt(max_per_chip))))
        x, y = g['x1-map'].to_numpy(), g['y1-map'].to_numpy()
        cx = np.minimum(((x - x.min()) / max(np.ptp(x), 1e-9) * k).astype(int), k - 1)
        cy = np.minimum(((y - y.min()) / max(np.ptp(y), 1e-9) * k).astype(int), k - 1)
        cell = cy * k + cx
        order = rng.permutation(len(g))
        rank = np.empty(len(g), int)
        seen: Dict[int, int] = {}
        for i in order:
            rank[i] = seen.get(cell[i], 0)
            seen[cell[i]] = rank[i] + 1
        pick = np.lexsort((order, rank))[:max_per_chip]
        keep.append(g.iloc[np.sort(pick)])
    return pd.concat(keep, ignore_index=True)


def _lonlat_fn(working_crs: str):
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_user_input(working_crs)
        if crs.is_geographic:
            return lambda x, y: (x, y)
        tr = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        return lambda x, y: tr.transform(x, y)
    except Exception as e:
        print(f'[RIVAL] lon/lat columns left blank: {e}')
        return lambda x, y: None


def write_rival_csv(points: pd.DataFrame, working_crs: str, path: str,
                    detail_path: Optional[str] = None) -> Dict:
    """Write RIVAL's CSV (and optionally the detail CSV). Returns the stats."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    to_ll = _lonlat_fn(working_crs)
    dx, dy = [], []
    rows, detail_rows = [], []
    cols = {c: points[c].to_numpy() for c in ('x1-map', 'y1-map', 'x2-map', 'y2-map')}
    for i in range(len(points)):
        ix, iy = float(cols['x1-map'][i]), float(cols['y1-map'][i])
        rx, ry = float(cols['x2-map'][i]), float(cols['y2-map'][i])
        cells = [f'{ix:.3f}', f'{iy:.3f}', f'{rx:.3f}', f'{ry:.3f}',
                 f'{ix - rx:.3f}', f'{iy - ry:.3f}']
        rec = refs.csv_record(i + 1, cells, to_ll(ix, iy), to_ll(rx, ry))
        rows.append(rec)
        dx.append(ix - rx)
        dy.append(iy - ry)
        if detail_path:
            src = points.iloc[i]
            detail_rows.append(rec + [src.get(c, '') for c in DETAIL_EXTRA])

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(RIVAL_HEADER)
        w.writerows(rows)
    if detail_path:
        with open(detail_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(RIVAL_HEADER + DETAIL_EXTRA)
            w.writerows(detail_rows)
    return summarize(dx, dy)


def summarize(dx: List[float], dy: List[float]) -> Dict:
    st = refs.accuracy_stats(list(dx), list(dy))
    out = {'n': st['n'], 'rmse_x': st['rmse_x'], 'rmse_y': st['rmse_y'],
           'ce90': st['ce90'], 'ce90_circular': st['circular']}
    if dx:
        a, b = np.asarray(dx), np.asarray(dy)
        out.update({'mean_dx': float(a.mean()), 'mean_dy': float(b.mean()),
                    'median_dx': float(np.median(a)), 'median_dy': float(np.median(b)),
                    'std_dx': float(a.std()), 'std_dy': float(b.std())})
    return out


def _fallback_files(filtered_dir: str, stats_dir: str) -> List[str]:
    """No consensus: the configuration with the most inliers, all its chips."""
    summ = os.path.join(stats_dir, 'SUMMARY_ALL.csv')
    if not os.path.exists(summ):
        return []
    df = pd.read_csv(summ)
    if df.empty:
        return []
    keys = ['detector', 'disk_mode', 'det_wt', 'desc_wt', 'match_method', 'match_parameter',
            'ransac_method', 'ransac_threshold', 'ransac_confidence']
    keys = [k for k in keys if k in df.columns]
    df['_key'] = df[keys].astype(str).agg('|'.join, axis=1)
    best = df.groupby('_key')['num_inliers'].sum().idxmax()
    return df.loc[df['_key'] == best, 'filename'].tolist()


def export_run(record: Dict, working_crs: str, out_dir: str, scene_name: str,
               max_per_chip: int = 200, seed: int = 0) -> Optional[Dict]:
    """RIVAL CSV for one (channel, detector) run record from the engine.

    Uses the chips of the consensus-best configuration; when consensus found
    none, falls back to the configuration with the most inliers and marks the
    result 'unvalidated'."""
    if record.get('status') not in ('ok', 'no-consensus'):
        return None
    filtered = record.get('filtered_dir')
    if not filtered or not os.path.isdir(filtered):
        return None
    validated = True
    files: List[str] = []
    manifest = record.get('best_chip_manifest_csv')
    if manifest and os.path.exists(manifest):
        files = pd.read_csv(manifest)['filename'].tolist()
    if not files:
        files = _fallback_files(filtered, record.get('statistics_dir', ''))
        validated = False
    if not files:
        return None

    pts = collect_points(filtered, files)
    if pts.empty:
        return None
    n_all = len(pts)
    stats_all = summarize((pts['x1-map'] - pts['x2-map']).tolist(),
                          (pts['y1-map'] - pts['y2-map']).tolist())
    pts = thin_per_chip(pts, max_per_chip, seed)

    first = pts.iloc[0]
    cfg = f"{first['matcher']}" + (f"{first['match_parameter']}" if pd.notna(first['match_parameter']) else '')
    base = (f"RIVAL_{scene_name}_{record['nisar_pol']}_to{record['s1_ref_tag']}_"
            f"{record['detector_tag']}_{cfg}_r{first['ransac_threshold']:g}")
    path = os.path.join(out_dir, base + '.csv')
    detail = os.path.join(out_dir, base + '_detail.csv')
    stats = write_rival_csv(pts, working_crs, path, detail)
    return {'rival_csv': path, 'detail_csv': detail, 'validated': validated,
            'n_inliers_all': n_all, 'stats_all': stats_all, **stats,
            'detector': record['detector_tag'], 'channel': record['nisar_pol'],
            'reference': record['s1_ref_tag'], 'matcher_config': cfg,
            'ransac_threshold': float(first['ransac_threshold'])}


def read_rival_csv(path: str) -> pd.DataFrame:
    """Read the file the way DPQED_rival.load_csv_smart does (by name)."""
    mapping = {'ix': ['In X', 'In_X', 'x1-map'], 'iy': ['In Y', 'In_Y', 'y1-map'],
               'rx': ['Ref X', 'Ref_X', 'x2-map'], 'ry': ['Ref Y', 'Ref_Y', 'y2-map']}
    out = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            out.append({k: float(next((row[c] for c in mapping[k] if c in row), '0.000'))
                        for k in mapping})
    return pd.DataFrame(out)
