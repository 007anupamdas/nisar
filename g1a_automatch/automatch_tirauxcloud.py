"""
TIRAuxCloud (Orion-AI-Lab, arXiv 2602.21905, 2026; code Apache-2.0, data and
weights CC BY 4.0) as an EXPERIMENTAL cloud method for ML (cloud_method
'tirauxcloud'). Only its architectures are rebuilt here (segmentation-models-
pytorch); none of its code is needed or run.

What its released thermal checkpoints take (read from its code, 95e615d):
  one thermal channel, already min-max normalised to about 0-1 when its
  training patches were built; [B, 1, 256, 256] float32; output logits
  [B, classes, H, W] (argmax; 2 classes: clear / cloud; 3: clear / thin /
  cloud); checkpoints are state_dicts named model_hash_<hex>.pth, listed in
  its results/models_table*.csv.
  - Landsat models (100 m): the mean of B10 and B11 TOA radiance. The
    normalisation constants are not published: tiraux_norm [lo, hi]
    (W/m2/sr/um) is a guess to tune (default [0, 72], which puts warm
    ground near the ~0.13 seen in its sample patches).
  - "VIIRS" models (200 m): by their files, DMSP-OLS night-time thermal
    (10.5-12.6 um, 8-bit counts, T = 0.4706 DN + 190 K) scaled by 1/251,
    so x = (BT - 190) / 118.1, BT here the mean of BT10.8 and BT12.0. This
    is inferred from the data, not stated by the authors.
  The split window, ice, CO2 and water-vapour bands of ML are not used.
  Its authors' own zero-shot transfer between sensors reached mIoU 0.22-
  0.24, so judge it against reference outlines (the cloud sweep) before
  trusting it.

Each ML tile of about 256 model pixels is resampled to the model's pixel
size (tiraux_px_m: 200 m OLS, 100 m Landsat), predicted, and the cloud
probability averaged back onto the ML grid; cloud where it is at least
tiraux_threshold (0.5).

Install on the workstation (in the job's environment):
    pip install segmentation-models-pytorch==0.5.0
and from https://huggingface.co/datasets/tirauxcloud/TIRAuxCloud take
model_files.tar.gz (10.6 GB), unpack it, and point cloud_tiraux_checkpoint
at one .pth, e.g. the OLS SegFormer model_hash_78713d1f6de22d2.pth or the
OLS Unet model_hash_8ce85fd78026576.pth (Landsat: Unet 41be05a2c8237ec,
SegFormer 50d4e5a4775675d).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

ARCHS = ('SegFormer', 'Unet')
MODES = {'ols': 200.0, 'landsat': 100.0}           # input kind -> its pixel size (m)
DEFAULTS = {'checkpoint': '', 'arch': 'SegFormer', 'mode': 'ols', 'classes': 2, 'px_m': 0.0,
            'norm': [0.0, 72.0], 'threshold': 0.5, 'tile': 256}
OLS_T0, OLS_SPAN = 190.0, 0.4706 * 251.0            # x = (BT - T0) / span
_MODELS: Dict = {}


def options(job: Dict) -> Dict:
    """The TIRAuxCloud settings of a job (cloud_tiraux_*)."""
    o = dict(DEFAULTS)
    for k in DEFAULTS:
        v = job.get(f'cloud_tiraux_{k}')
        if v not in (None, ''):
            o[k] = v
    o['arch'] = {a.lower(): a for a in ARCHS}.get(str(o['arch']).lower(), o['arch'])
    o['mode'] = str(o['mode']).lower()
    o['classes'] = int(o['classes'])
    o['px_m'] = float(o['px_m'] or 0) or MODES.get(o['mode'], 200.0)
    o['norm'] = [float(x) for x in (o['norm'] or DEFAULTS['norm'])]
    o['threshold'] = float(o['threshold'])
    o['tile'] = int(o['tile'])
    return o


def check(opts: Dict) -> list:
    """What stops TIRAuxCloud from running here (empty: nothing)."""
    out = []
    try:
        import torch  # noqa: F401
        import segmentation_models_pytorch  # noqa: F401
    except Exception as e:
        out.append(f'TIRAuxCloud needs torch and segmentation-models-pytorch (pip install '
                   f'segmentation-models-pytorch==0.5.0): {type(e).__name__}: {e}')
    ck = opts.get('checkpoint') or ''
    if not ck:
        out.append('TIRAuxCloud: set cloud_tiraux_checkpoint to a model_hash_<...>.pth from its model_files.tar.gz')
    elif not os.path.isfile(ck):
        out.append(f'TIRAuxCloud checkpoint not found: {ck}')
    if opts.get('arch') not in ARCHS:
        out.append(f"TIRAuxCloud arch {opts.get('arch')!r}: one of {', '.join(ARCHS)}")
    if opts.get('mode') not in MODES:
        out.append(f"TIRAuxCloud mode {opts.get('mode')!r}: one of {', '.join(MODES)}")
    return out


def build(arch: str, classes: int):
    """The architecture, untrained, offline (no encoder download)."""
    import segmentation_models_pytorch as smp
    if arch == 'Unet':
        return smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=1, classes=classes)
    if arch == 'SegFormer':
        return smp.create_model(arch='segformer', encoder_name='mit_b2', encoder_weights=None, in_channels=1,
                                classes=classes)
    raise ValueError(f'TIRAuxCloud arch {arch!r}: one of {", ".join(ARCHS)}')


def load(opts: Dict, device: Optional[str] = None):
    """The model with its checkpoint, cached per (checkpoint, arch, classes, device)."""
    import torch
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    key = (os.path.abspath(opts['checkpoint']), opts['arch'], opts['classes'], dev)
    if key not in _MODELS:
        m = build(opts['arch'], opts['classes'])
        sd = torch.load(opts['checkpoint'], map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith('encoder') for k in sd):
            sd = sd['state_dict']
        m.load_state_dict(sd)
        _MODELS.clear()
        _MODELS[key] = m.to(dev).eval()
    return _MODELS[key], dev


def model_input(bt: Dict[float, np.ndarray], rad_si: Dict[float, np.ndarray], valid: np.ndarray,
                opts: Dict) -> np.ndarray:
    """The one channel the model takes, normalised (module docstring); NaN
    where not valid."""
    k108 = min(bt, key=lambda u: abs(u - 10.8))
    k12 = min(bt, key=lambda u: abs(u - 12.0))
    if opts['mode'] == 'ols':
        x = ((bt[k108] + bt[k12]) / 2.0 - OLS_T0) / OLS_SPAN
    else:
        lo, hi = opts['norm']
        x = ((rad_si[k108] + rad_si[k12]) / 2.0 - lo) / max(hi - lo, 1e-9)
    return np.where(valid, x, np.nan).astype(np.float32)


def probabilities(x: np.ndarray, valid: np.ndarray, px_m: float, opts: Dict, device: Optional[str] = None,
                  model=None) -> np.ndarray:
    """Class probabilities [classes, H, W] on the image's grid: tiles of
    ~opts['tile'] model pixels, each resampled to the model's pixel size,
    predicted, and the probabilities averaged back onto the image pixels."""
    import cv2
    import torch
    net, dev = (model, device or 'cpu') if model is not None else load(opts, device)
    f = max(px_m / float(opts['px_m']), 1e-6)            # model pixels per image pixel
    tile = int(opts['tile'])
    n = max(8, int(round(tile / f)))                     # image pixels per tile side
    fill = float(np.nanmedian(x[valid])) if valid.any() else 0.0
    xf = np.where(valid, x, fill).astype(np.float32)
    H, W = x.shape
    prob = None
    batch, where = [], []

    def flush():
        nonlocal prob
        if not batch:
            return
        with torch.no_grad():
            t = torch.from_numpy(np.stack(batch)[:, None]).float().to(dev)
            out = net(t)
            out = out[0] if isinstance(out, (tuple, list)) else out
            p = torch.softmax(out.float(), dim=1).cpu().numpy()
        if prob is None:
            prob = np.zeros((p.shape[1], H, W), np.float32)
        for (y0, x0, h, w), pi in zip(where, p):
            for c in range(pi.shape[0]):           # the tile covers n x n image pixels, padding included
                full = cv2.resize(pi[c].astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
                prob[c, y0:y0 + h, x0:x0 + w] = full[:h, :w]
        batch.clear()
        where.clear()
    for y0 in range(0, H, n):
        for x0 in range(0, W, n):
            h, w = min(n, H - y0), min(n, W - x0)
            sub = np.pad(xf[y0:y0 + h, x0:x0 + w], ((0, n - h), (0, n - w)), mode='edge')
            batch.append(cv2.resize(sub, (tile, tile), interpolation=cv2.INTER_LINEAR))
            where.append((y0, x0, h, w))
            if len(batch) >= 8:
                flush()
    flush()
    return prob


def mask(bt: Dict[float, np.ndarray], rad_si: Dict[float, np.ndarray], valid: np.ndarray, px_m: float,
         opts: Dict, device: Optional[str] = None, model=None, prob: Optional[np.ndarray] = None
         ) -> Tuple[np.ndarray, Dict]:
    """(classes 0 clear / 1 cloud / 2 thin cloud (3-class models: where thin
    is likelier than cloud) / 255 no data, info). Cloud where the
    probability of any cloud is at least opts['threshold']."""
    if prob is None:
        prob = probabilities(model_input(bt, rad_si, valid, opts), valid, px_m, opts, device, model)
    pc = prob[1:].sum(0) if prob.shape[0] > 1 else prob[0]
    cls = np.zeros(valid.shape, np.uint8)
    cloud = valid & (pc >= opts['threshold'])
    cls[cloud] = 1
    if prob.shape[0] >= 3:
        cls[cloud & (prob[1] > prob[2])] = 2
    cls[~valid] = 255
    nv = max(int(valid.sum()), 1)
    return cls, {'tiraux_cloud_prob_median': round(float(np.median(pc[valid])), 3) if valid.any() else None,
                 'test_model_cloud_pct': round(100.0 * float(cloud.sum()) / nv, 2)}
