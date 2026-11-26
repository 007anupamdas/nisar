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


_IMG_EXT = ('.tif', '.tiff', '.img', '.vrt', '.h5', '.jp2')


class InputIndex:
    """Input images under several folders, by file name: a job's input_path
    from another machine (server / workstation) found where it is here."""

    def __init__(self, roots: List[str]):
        self.by_name: Dict[str, str] = {}
        for root in roots or []:
            for d, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(_IMG_EXT):
                        self.by_name.setdefault(f.lower(), os.path.join(d, f))

    def find(self, path: str) -> str:
        p = str(path or '')
        if p and os.path.exists(p):
            return p
        return self.by_name.get(os.path.basename(p.replace('\\', '/')).lower(), p)


def _truth_key(name: str) -> str:
    """A truth file's name as the product it is for: 'pts_267403451_band4.csv' -> '267403451'."""
    s = os.path.splitext(os.path.basename(name))[0].lower()
    s = re.sub(r'^(pts|gt|truth|gcp|manual)[_-]+', '', s)
    s = re.sub(r'[_-]+(pts|gt|truth|gcp|manual)$', '', s)
    s = re.sub(r'[_-]+b(and)?\d+$', '', s)
    return s


class TruthIndex:
    """Truth files of one folder (and below), found for an input by name: the
    input's file name, or one of its folders (e.g. the request ID folder)."""

    def __init__(self, folder: str):
        self.by_key: Dict[str, List[str]] = {}
        root = os.path.normpath(folder or '')
        for d, _, files in os.walk(folder or ''):
            # a file also counts under the names of its folders below the truth folder
            # (truth/<product or request ID>/pts_band4.csv)
            subs = [x for x in os.path.relpath(d, root).replace('\\', '/').split('/') if x not in ('', '.')]
            for f in files:
                if f.lower().endswith(('.csv', '.txt')):
                    p = os.path.join(d, f)
                    for k in dict.fromkeys([_truth_key(f)] + [_truth_key(x) for x in subs] + [x.lower() for x in subs]):
                        self.by_key.setdefault(k, []).append(p)

    def find(self, input_path: str, band: str = '') -> str:
        """The truth file of an input: by its file name, else one of its folders
        (e.g. the request ID); a name that matches files in more than one folder
        (pts_band4.csv in every product folder, a folder 'MV') is ambiguous and
        not used. The input's band first (band4, B4, band_4)."""
        import re
        p = str(input_path or '').replace('\\', '/')
        keys = [os.path.splitext(os.path.basename(p))[0].lower()]
        # the input's folders only when specific (a request ID, a product: digits, 6+ characters), not 'MV', 'ADRIN'
        keys += [x.lower() for x in p.split('/')[-4:-1]
                 if len(x) >= 6 and re.search(r'\d', x) and x.lower() not in ('input', 'inputs', 'data')][::-1]
        nums = [m.group(1) for b in ([band] if band else []) for m in [re.search(r'(\d+)', str(b))] if m]
        for k in keys:
            got = self.by_key.get(_truth_key(k)) or self.by_key.get(k)
            if not got or len({os.path.dirname(g) for g in got}) > 1:
                continue
            same = [g for g in got for n in nums
                    if re.search(rf'[_\-. ](?:band_?|b)0*{n}(?:\D|$)', os.path.splitext(os.path.basename(g))[0].lower())]
            return sorted(same or got)[0]
        return ''


def manifest_rows(job: str, level: Optional[int] = None, include_unreliable: bool = False,
                  maps: Optional[List[Tuple[str, str]]] = None,
                  band_modality: Optional[Dict[str, str]] = None, inputs: Optional[InputIndex] = None,
                  truths: Optional[TruthIndex] = None) -> Tuple[List[Dict], List[str]]:
    """(manifest rows of one job, notes on what was left out)."""
    maps = maps or []
    name = os.path.basename(os.path.normpath(job))
    with open(os.path.join(job, 'job_used.json'), encoding='utf-8') as f:
        j = json.load(f)
    inp = apply_map(j.get('input_path') or '', maps)
    if inputs is not None:
        inp = inputs.find(inp)
    if not os.path.exists(inp):
        return [], [f'{name}: input {inp} not found here (give the input folders with --inputs)']
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
        tcsv = apply_map(j.get('truth_csv') or '', maps)
        if truths is not None and not (tcsv and os.path.exists(tcsv)):
            tcsv = truths.find(inp, b) or tcsv      # an auto run without truth: its truth from the truth folder
        rows.append({'scene_id': f'{stem}_{b}', 'sensor': sensor, 'date': '', 'input_path': inp,
                     'input_band': int(re.search(r'(\d+)', b).group(1)), 'labels': 'consistent', 'gt_path': f,
                     'reference': apply_map(j.get('reference_dir') or '', maps),
                     'reference_band': int(j.get('reference_band') or 1),
                     'modality': (band_modality or {}).get(f'{sensor}:{b}', (band_modality or {}).get(b, '')),
                     'cloud_mask': cloud[0] if cloud else '',
                     'split': 'test' if str(tcsv or '').strip() else '',
                     'truth_csv': tcsv, 'job': name})
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
    ap.add_argument('jobs', nargs='+', help='job output folders, or folders of them (several places allowed)')
    ap.add_argument('--inputs', action='append', default=[],
                    help='a folder of input images (repeatable): where a job\'s input is when its saved path is '
                         'from another machine')
    ap.add_argument('--truth-folder', default='',
                    help='the folder of truth files, named after each input (its file name or one of its folders, '
                         'e.g. the request ID): those scenes only test')
    ap.add_argument('--out', default='scenes_consistent.csv')
    ap.add_argument('--level', type=int, default=None, help='agreement level in px (default: the finest)')
    ap.add_argument('--include-unreliable', action='store_true')
    ap.add_argument('--band-modality', default='',
                    help="what each band is, e.g. 'band4=nir,band3=visible' or per sensor 'G1A_MV:band4=nir' "
                         "(visible, nir, swir, mwir, lwir, sar); empty: visible")
    ap.add_argument('--map', action='append', default=[], metavar='SERVER=LOCAL',
                    help="path prefix as this machine sees it, e.g. /maintenance/=V:/ (repeatable)")
    a = ap.parse_args(argv)
    maps = [tuple(m.split('=', 1)) for m in a.map if '=' in m]
    bm = dict(x.strip().split('=', 1) for x in a.band_modality.split(',') if '=' in x)
    rows, notes = [], []
    folders = job_folders(a.jobs)
    inputs = InputIndex([apply_map(p, maps) for p in a.inputs]) if a.inputs else None
    truths = TruthIndex(apply_map(a.truth_folder, maps)) if a.truth_folder else None
    if inputs is not None:
        print(f'[Consistent] {len(inputs.by_name)} input image(s) under {len(a.inputs)} folder(s)')
    if truths is not None:
        print(f'[Consistent] {sum(len(v) for v in truths.by_key.values())} truth file(s) in {a.truth_folder}')
    for job in folders:
        try:
            r, n = manifest_rows(job, a.level, a.include_unreliable, maps, bm, inputs, truths)
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
