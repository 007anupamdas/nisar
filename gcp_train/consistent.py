"""
Training labels from g1a_automatch's consistent points.

A G1A scene's manual points do not fit one smooth field closely enough to
label crops (leave-one-out 7-85 px on MV; see SCOPE_CONSISTENT_POINTS.md).
automatch's consistent points are thousands per scene: the best keypoint of
each agreement cell where several unrelated detectors measure the same
offset. pairs.py --labels consistent (or a manifest row with
labels=consistent) labels each crop from the consistent points in and around
it: one homography per crop (fit.LocalPoints), so distortion that changes
across the scene does not matter.

This file reads automatch job output folders (nothing is imported from
g1a_automatch; only its files are read) and writes the manifest:

    python consistent.py <job folder | folder of job folders> ... --out scenes_consistent.csv
        [--level 64] [--min-agree 3] [--include-unreliable]
        [--map /maintenance/=V:/ ...]       # server paths as this machine sees them

Per job and band, the finest agreement level's RIVAL_CONSISTENT_<band>_<L>px_detail.csv
(older jobs: RIVAL_CONSISTENT_<band>_detail.csv), with the input, the band
and the references from job_used.json. A job whose cross-check was not
reliable for that band is left out (--include-unreliable keeps it). Every
job with a truth file goes to the test split: the truth judges, it never
trains (its pairs are still written, to check the labels).

Older jobs without consistent points get them without matching again:
    python automatch_cross.py <folder of jobs> --batch --jobs 1061-1104 --recompute
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

SENSOR_RE = re.compile(r'G1A_?(MV|HV|HS|ML)(?=[_0-9A-Z])')


def load_consistent(path: str, min_agree: int = 3) -> pd.DataFrame:
    """A RIVAL_CONSISTENT_*_detail.csv as tie points (point_id, in_lon, in_lat,
    ref_lon, ref_lat) with their detector, n_agree and agree_spread_m; only
    the points at least min_agree detectors agreed on."""
    d = pd.read_csv(path, encoding='utf-8-sig')
    need = ['In_Lon', 'In_Lat', 'Ref_Lon', 'Ref_Lat']
    if not set(need) <= set(d.columns):
        raise ValueError(f'{os.path.basename(path)}: not a RIVAL point file (no {", ".join(need)})')
    for k in need:
        d[k] = pd.to_numeric(d[k], errors='coerce')
    d = d.dropna(subset=need)
    if 'n_agree' in d.columns:
        d = d[pd.to_numeric(d['n_agree'], errors='coerce').fillna(0) >= min_agree]
    out = pd.DataFrame({'point_id': range(1, len(d) + 1), 'in_lon': d['In_Lon'].to_numpy(),
                        'in_lat': d['In_Lat'].to_numpy(), 'ref_lon': d['Ref_Lon'].to_numpy(),
                        'ref_lat': d['Ref_Lat'].to_numpy()})
    for k in ('detector', 'n_agree', 'agree_spread_m'):
        if k in d.columns:
            out[k] = d[k].to_numpy()
    if out.empty:
        raise ValueError(f'{os.path.basename(path)}: no point agreed by {min_agree}+ detectors')
    return out


def job_folders(paths: List[str]) -> List[str]:
    """Job output folders (with job_used.json) given or under the paths (two levels)."""
    out = []
    for p in paths:
        if os.path.exists(os.path.join(p, 'job_used.json')):
            out.append(p)
            continue
        for pat in ('*', os.path.join('*', '*')):
            out += sorted(os.path.dirname(f) for f in glob.glob(os.path.join(p, pat, 'job_used.json')))
    return list(dict.fromkeys(os.path.normpath(p) for p in out))


def consistent_file(job: str, band: str, level: Optional[int] = None) -> str:
    """The finest agreement level's (or the given level's) consistent-point detail file of a band, or ''."""
    files = glob.glob(os.path.join(job, f'RIVAL_CONSISTENT_{band}_*px_detail.csv'))
    lv = []
    for f in files:
        m = re.search(r'_(\d+)px_detail\.csv$', f)
        if m:
            lv.append((int(m.group(1)), f))
    if level:
        lv = [x for x in lv if x[0] == int(level)]
    if lv:
        return sorted(lv)[0][1]
    old = os.path.join(job, f'RIVAL_CONSISTENT_{band}_detail.csv')
    return old if os.path.exists(old) and not level else ''


def _reliable(job: str, band: str) -> Optional[bool]:
    try:
        c = pd.read_csv(os.path.join(job, 'CROSS_SUMMARY.csv'))
        r = c[c['channel'].astype(str) == band]
        return bool(str(r['reliable'].iloc[0]).lower() == 'true') if len(r) else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


def apply_map(path: str, maps: List[Tuple[str, str]]) -> str:
    """A server path as this machine sees it ('/maintenance/' -> 'V:/')."""
    p = str(path or '')
    for a, b in maps:
        if p.replace('\\', '/').startswith(a.replace('\\', '/')):
            return b + p.replace('\\', '/')[len(a.replace('\\', '/')):]
    return p


def manifest_rows(job: str, level: Optional[int] = None, include_unreliable: bool = False,
                  maps: Optional[List[Tuple[str, str]]] = None) -> Tuple[List[Dict], List[str]]:
    """(manifest rows of one job, notes on what was left out)."""
    maps = maps or []
    name = os.path.basename(os.path.normpath(job))
    with open(os.path.join(job, 'job_used.json'), encoding='utf-8') as f:
        j = json.load(f)
    inp = apply_map(j.get('input_path') or '', maps)
    m = SENSOR_RE.search(os.path.basename(inp))
    sensor = f'G1A_{m.group(1)}' if m else ''
    bands = [c for c in (j.get('channels') or []) if str(c).startswith('band')]
    if not bands:
        bands = sorted({re.search(r'RIVAL_CONSISTENT_(band\d+)', f).group(1)
                        for f in glob.glob(os.path.join(job, 'RIVAL_CONSISTENT_band*_detail.csv'))})
    cloud = sorted(glob.glob(os.path.join(job, 'CLOUD_MASK*.tif')))
    rows, notes = [], []
    for b in bands:
        f = consistent_file(job, b, level)
        if not f:
            notes.append(f'{name} {b}: no consistent points (run automatch_cross.py ... --recompute)')
            continue
        rel = _reliable(job, b)
        if rel is False and not include_unreliable:
            notes.append(f'{name} {b}: cross-check not reliable (too few cells agree): left out')
            continue
        stem = os.path.splitext(os.path.basename(inp))[0]
        rows.append({'scene_id': f'{stem}_{b}', 'sensor': sensor, 'date': '', 'input_path': inp,
                     'input_band': int(re.search(r'(\d+)', b).group(1)), 'labels': 'consistent', 'gt_path': f,
                     'reference': apply_map(j.get('reference_dir') or '', maps),
                     'reference_band': int(j.get('reference_band') or 1),
                     'cloud_mask': cloud[0] if cloud else '',
                     'split': 'test' if str(j.get('truth_csv') or '').strip() else '',
                     'truth_csv': apply_map(j.get('truth_csv') or '', maps), 'job': name})
    return rows, notes


def _n_points(path: str) -> int:
    try:
        with open(path, encoding='utf-8-sig') as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scene (product and band): several job folders of one
    product (the run window's auto steps, its tiers, reruns) would give the
    same ground twice, and could split it between train and test. The job with
    the most consistent points is kept; the scene only tests when any of its
    jobs had a truth file."""
    if df.empty:
        return df
    df = df.assign(points=[_n_points(p) for p in df['gt_path']])
    test = df.groupby('scene_id')['split'].apply(lambda v: (v == 'test').any())
    truth = df[df['truth_csv'].astype(str).str.strip() != ''].groupby('scene_id')['truth_csv'].first()
    out = df.sort_values('points', ascending=False).drop_duplicates('scene_id').copy()
    out['split'] = ['test' if test[s] else sp for s, sp in zip(out['scene_id'], out['split'])]
    out['truth_csv'] = [truth.get(s, t) for s, t in zip(out['scene_id'], out['truth_csv'])]
    dropped = len(df) - len(out)
    if dropped:
        print(f'[Consistent] {dropped} job folder(s) of a product already listed left out (the one with the most '
              f'consistent points kept)')
    return out.sort_values('scene_id').reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Manifest of training scenes from automatch consistent points')
    ap.add_argument('jobs', nargs='+', help='job output folders, or folders of them')
    ap.add_argument('--out', default='scenes_consistent.csv')
    ap.add_argument('--level', type=int, default=None, help='agreement level in px (default: the finest)')
    ap.add_argument('--include-unreliable', action='store_true')
    ap.add_argument('--map', action='append', default=[], metavar='SERVER=LOCAL',
                    help="path prefix as this machine sees it, e.g. /maintenance/=V:/ (repeatable)")
    a = ap.parse_args(argv)
    maps = [tuple(m.split('=', 1)) for m in a.map if '=' in m]
    rows, notes = [], []
    folders = job_folders(a.jobs)
    for job in folders:
        try:
            r, n = manifest_rows(job, a.level, a.include_unreliable, maps)
        except (OSError, ValueError) as e:
            r, n = [], [f'{os.path.basename(job)}: {type(e).__name__}: {e}']
        rows += r
        notes += n
    for n in notes:
        print(f'[Consistent] {n}')
    if not rows:
        print(f'[Consistent] no job with consistent points among {len(folders)} folder(s)')
        return 1
    df = dedupe(pd.DataFrame(rows))
    df.to_csv(a.out, index=False)
    n_test = int((df['split'] == 'test').sum())
    print(f'[Consistent] {len(df)} scene(s) from {df["job"].nunique()} job(s) -> {a.out} '
          f'({n_test} with a truth file: test only)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
