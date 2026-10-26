"""
How well a matcher does on the held-out pairs, before and after fine-tuning.

Per pair (its true homography H_0to1 is known):
    matches       matches returned
    correct       of those, within thr px of where H puts them
    precision     correct / matches
    h_ok          a homography estimated from the matches (RANSAC, thr px)
                  moves the crop's corners less than thr px from H's
Summaries go to EVAL_<label>.csv: overall, per sensor, per modality pair
(input -> reference, e.g. lwir -> visible) and per pixel-size band. Run it on
the test split (scenes never used for training), once with the official
weights and once with the fine-tuned checkpoint, through Glue Factory's own
model code:

    python evaluate.py --conf configs/aliked_lightglue_gcp.yaml --index pairs/pairs.csv \\
        --checkpoint outputs/training/<experiment>/checkpoint_best.tar --label finetuned
    python evaluate.py --conf configs/aliked_lightglue_gcp.yaml --index pairs/pairs.csv --label official
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def pair_metrics(k0: np.ndarray, k1: np.ndarray, m0: np.ndarray, H: np.ndarray, size: tuple,
                 thr: float = 3.0) -> Dict:
    """k0 (N, 2), k1 (M, 2) keypoints; m0 (N,) index into k1 or -1; H 3 x 3;
    size (W, H) of image 0."""
    import cv2
    valid = m0 >= 0
    n = int(valid.sum())
    out = {'matches': n, 'correct': 0, 'precision': 0.0, 'h_ok': False, 'corner_err_px': None}
    if n == 0:
        return out
    p0 = k0[valid].astype(np.float64)
    p1 = k1[m0[valid]].astype(np.float64)
    q = np.column_stack([p0, np.ones(n)]) @ H.T
    q = q[:, :2] / q[:, 2:3]
    ok = np.hypot(*(q - p1).T) <= thr
    out['correct'] = int(ok.sum())
    out['precision'] = round(float(ok.mean()), 4)
    if n >= 4:
        He, _ = cv2.findHomography(p0, p1, cv2.RANSAC, thr)
        if He is not None:
            w, h = size
            c = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], float)
            a = c @ H.T
            b = c @ He.T
            err = float(np.mean(np.hypot(*((a[:, :2] / a[:, 2:3]) - (b[:, :2] / b[:, 2:3])).T)))
            out['corner_err_px'] = round(err, 3)
            out['h_ok'] = err <= thr
    return out


RES_BANDS = [(0, 10, '<10 m'), (10, 30, '10-30 m'), (30, 100, '30-100 m'), (100, 300, '100-300 m'),
             (300, float('inf'), '>=300 m')]


def res_band(px: float) -> str:
    for lo, hi, name in RES_BANDS:
        if lo <= px < hi:
            return name
    return 'unknown'


def summarize(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    groups = [('all', 'all', df)] + [('sensor', k, d) for k, d in df.groupby('sensor')]
    if 'modality0' in df:
        mp = df['modality0'].astype(str) + ' -> ' + df['modality1'].astype(str)
        groups += [('modality', k, d) for k, d in df.groupby(mp)]
    if 'px_m' in df:
        groups += [('resolution', name, df[df['px_m'].map(res_band) == name]) for _, _, name in RES_BANDS
                   if (df['px_m'].map(res_band) == name).any()]
    g = []
    for by, key, d in groups:
        g.append({'by': by, 'group': key, 'pairs': len(d), 'mean_matches': round(float(d['matches'].mean()), 1),
                  'mean_correct': round(float(d['correct'].mean()), 1),
                  'precision': round(float(d['correct'].sum() / max(d['matches'].sum(), 1)), 4),
                  'h_success': round(float(d['h_ok'].mean()), 4)})
    return pd.DataFrame(g)


def run_gluefactory(conf_path: str, index: str, checkpoint: Optional[str], split: str = 'test',
                    thr: float = 3.0, device: Optional[str] = None, max_pairs: Optional[int] = None,
                    overrides: Optional[List[str]] = None) -> List[Dict]:
    import torch
    from omegaconf import OmegaConf
    from gluefactory.models import get_model
    from dataset import PairDataset
    from modality import MODALITIES
    conf = OmegaConf.load(conf_path)
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    mconf = conf.model
    if checkpoint:
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        mconf = OmegaConf.merge(mconf, OmegaConf.create((cp.get('conf') or {}).get('model') or {}))
    model = get_model(mconf.name)(mconf)
    if checkpoint:
        model.load_state_dict(cp['model'], strict=False)
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(dev).eval()
    ds = PairDataset(index, split, channels=int(conf.data.get('channels', 3)), augment=False, swap=False,
                     max_pairs=max_pairs)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            batch = {'view0': {k: v[None].to(dev) for k, v in s['view0'].items()},
                     'view1': {k: v[None].to(dev) for k, v in s['view1'].items()},
                     'H_0to1': s['H_0to1'][None].to(dev),
                     'modality0': torch.tensor([s['modality0']], device=dev),
                     'modality1': torch.tensor([s['modality1']], device=dev),
                     'px_m': torch.tensor([s['px_m']], device=dev)}
            pred = model(batch)
            k0 = pred['keypoints0'][0].cpu().numpy()
            k1 = pred['keypoints1'][0].cpu().numpy()
            m0 = pred['matches0'][0].cpu().numpy()
            size = tuple(s['view0']['image_size'].numpy().astype(int))
            rows.append({'name': s['name'], 'scene': s['scene'], 'sensor': s['sensor'],
                         'modality0': MODALITIES[s['modality0']], 'modality1': MODALITIES[s['modality1']],
                         'px_m': s['px_m'],
                         **pair_metrics(k0, k1, m0, s['H_0to1'].numpy().astype(np.float64), size, thr)})
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Matcher accuracy on the held-out tie-point pairs')
    ap.add_argument('--conf', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--checkpoint', default='', help='fine-tuned checkpoint; empty: the config\'s (official) weights')
    ap.add_argument('--split', default='test')
    ap.add_argument('--thr', type=float, default=3.0)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--max-pairs', type=int, default=0)
    ap.add_argument('--set', action='append', default=[], help='config override, e.g. model.matcher.condition=false')
    a = ap.parse_args(argv)
    rows = run_gluefactory(a.conf, a.index, a.checkpoint or None, a.split, a.thr, max_pairs=a.max_pairs or None,
                           overrides=a.set)
    label = a.label or ('finetuned' if a.checkpoint else 'official')
    out = a.out or os.path.dirname(os.path.abspath(a.index))
    pd.DataFrame(rows).to_csv(os.path.join(out, f'EVAL_{label}_pairs.csv'), index=False)
    s = summarize(rows)
    s.to_csv(os.path.join(out, f'EVAL_{label}.csv'), index=False)
    print(s.to_string(index=False))
    return 0 if rows else 1


if __name__ == '__main__':
    sys.exit(main())
