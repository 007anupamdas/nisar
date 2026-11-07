"""
Learned cloud masks on TOA reflectance (automatch_reflect), as options:

ukis-csmask (DLR, Apache-2.0; cloud_method 'csmask', MV / HV)
  Its 4-band U-Net (blue, green, red, NIR; trained on Sentinel-2 / Landsat
  at ~30 m, SPARCS + CloudSEN12+) through its own package:
      pip install ukis-csmask[gpu]     (or [cpu]; weights come with it)
  Classes: cloud -> thick (1), cloud shadow -> shadow (3). Large images go
  in blocks of cloud_csmask_block px (4096) with a 256 px margin.

SpecTf (NASA JPL, Apache-2.0; cloud_method 'spectf', HV / HS)
  A per-pixel spectral transformer (the EMIT cloud mask: F1 0.95 on EMIT):
  each band is a token (TOA reflectance, band wavelength), so any band set
  works. Its small model (below, as in its spectf/model.py: SpecTfEncoder,
  1 layer, 64 dims, 8 heads, max pooling) is rebuilt here and its weights
  loaded -- none of its package is needed. Weights (109 KB):
      https://raw.githubusercontent.com/emit-sds/SpecTf/main/spectf_cloud/weights/current.pt
  -> cloud_spectf_weights. Bands at 405-1275 and 1321-2455 nm are used (as
  EMIT). HS alone covers only 900-2500 nm; cloud_spectf_vnir names the HV
  image of the same pass, whose 405-900 nm bands are added on the HS grid,
  giving the full spectrum the model was trained on. Cloud where its
  probability is at least cloud_spectf_threshold (0.51, its own).
  Spectra over cloud_spectf_max_bands (300) bands are thinned evenly.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

SPECTF_DROP = [(0.0, 405.0), (1275.0, 1321.0), (2455.0, 1e9)]
SPECTF_DEFAULTS = {'weights': '', 'threshold': 0.51, 'max_bands': 300, 'vnir': '', 'batch': 4096}
CSMASK_DEFAULTS = {'product_level': 'l1c', 'block': 4096}
_SPECTF: Dict = {}


# ---------------------------------------------------------------- SpecTf model
def spectf_model(banddef_nm, state_dict=None):
    """SpecTfEncoder (as in SpecTf's spectf/model.py, Copyright 2025 California
    Institute of Technology, Apache License 2.0): the same modules and names,
    so its state_dict loads."""
    import torch
    from torch import nn

    class BandConcat(nn.Module):
        def __init__(self, banddef, mean=1440, std=600):
            super().__init__()
            self.banddef = (banddef.unsqueeze(-1).unsqueeze(0) - mean) / std

        def forward(self, spectra):
            return torch.cat((spectra, self.banddef.to(spectra.device, spectra.dtype).expand_as(spectra)), dim=-1)

    class SpectralEmbed(nn.Module):
        def __init__(self, n_filters=64):
            super().__init__()
            self.linear = nn.Linear(2, n_filters)
            self.activation = nn.Tanh()

        def forward(self, x):
            return self.activation(self.linear(x))

    class FeedForwardBlock(nn.Module):
        def __init__(self, dim_model, dim_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(dim_model, dim_ff, bias=True)
            self.linear2 = nn.Linear(dim_ff, dim_model, bias=True)
            self.dropout = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.gelu = nn.GELU()

        def forward(self, x):
            return self.dropout2(self.linear2(self.dropout(self.gelu(self.linear1(x)))))

    class AttentionBlock(nn.Module):
        def __init__(self, dim_model, num_heads, dropout=0.1):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, bias=True, batch_first=True)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q, k, v):
            return self.dropout(self.attention(q, k, v)[0])

    class EncoderLayer(nn.Module):
        def __init__(self, dim_model, num_heads, dim_ff, dropout=0.1):
            super().__init__()
            self.attention = AttentionBlock(dim_model, num_heads, dropout)
            self.ff = FeedForwardBlock(dim_model, dim_ff, dropout)
            self.norm1 = nn.LayerNorm(dim_model)
            self.norm2 = nn.LayerNorm(dim_model)

        def forward(self, x):
            n = self.norm1(x)
            x = self.attention(n, n, n)
            return self.ff(self.norm2(x))

    class SpecTfEncoder(nn.Module):
        def __init__(self, banddef, dim_output=2, num_heads=8, dim_proj=64, dim_ff=64, dropout=0.1):
            super().__init__()
            self.band_concat = BandConcat(banddef)
            self.spectral_embed = SpectralEmbed(n_filters=dim_proj)
            self.layers = nn.ModuleList([EncoderLayer(dim_proj, num_heads, dim_ff, dropout)])
            self.head = nn.Linear(dim_proj, dim_output)

        def forward(self, x):
            x = self.spectral_embed(self.band_concat(x))
            for layer in self.layers:
                x = layer(x)
            return self.head(torch.max(x, dim=1)[0])

    m = SpecTfEncoder(torch.tensor(np.asarray(banddef_nm, np.float32)))
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return m.eval()


def spectf_options(job: Dict) -> Dict:
    o = dict(SPECTF_DEFAULTS)
    for k in SPECTF_DEFAULTS:
        v = job.get(f'cloud_spectf_{k}')
        if v not in (None, ''):
            o[k] = v
    o['threshold'] = float(o['threshold'])
    o['max_bands'] = int(o['max_bands'])
    o['batch'] = int(o['batch'])
    return o


def spectf_check(opts: Dict) -> List[str]:
    out = []
    try:
        import torch  # noqa: F401
    except Exception as e:
        out.append(f'SpecTf needs torch: {type(e).__name__}: {e}')
    w = opts.get('weights') or ''
    if not w:
        out.append('SpecTf: set cloud_spectf_weights to its current.pt (109 KB, '
                   'raw.githubusercontent.com/emit-sds/SpecTf/main/spectf_cloud/weights/current.pt)')
    elif not os.path.isfile(w):
        out.append(f'SpecTf weights not found: {w}')
    if opts.get('vnir') and not os.path.isfile(opts['vnir']):
        out.append(f"cloud_spectf_vnir not found: {opts['vnir']}")
    return out


def spectf_bands(wl: List[Optional[float]], max_bands: int = 300) -> List[Tuple[int, float]]:
    """[(1-based band, nm)] SpecTf uses: inside 405-1275 and 1321-2455 nm,
    thinned evenly to at most max_bands."""
    keep = [(i + 1, float(w)) for i, w in enumerate(wl or [])
            if w is not None and not any(lo <= w < hi for lo, hi in SPECTF_DROP)]
    if max_bands and len(keep) > max_bands:
        idx = np.unique(np.round(np.linspace(0, len(keep) - 1, max_bands)).astype(int))
        keep = [keep[i] for i in idx]
    return keep


def spectf_probability(refl: np.ndarray, banddef_nm, valid: np.ndarray, opts: Dict, device: Optional[str] = None,
                       model=None) -> np.ndarray:
    """Cloud probability per pixel from reflectance [bands, H, W]."""
    import torch
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        key = (os.path.abspath(opts['weights']), tuple(np.round(banddef_nm, 2)), dev)
        if key not in _SPECTF:
            sd = torch.load(opts['weights'], map_location='cpu', weights_only=True)
            _SPECTF.clear()
            _SPECTF[key] = spectf_model(banddef_nm, sd).to(dev)
        model = _SPECTF[key]
    H, W = valid.shape
    prob = np.zeros(H * W, np.float32)
    idx = np.flatnonzero(valid.ravel())
    flat = refl.reshape(refl.shape[0], -1)
    with torch.inference_mode():
        for s in range(0, idx.size, int(opts.get('batch') or 4096)):
            sel = idx[s:s + int(opts.get('batch') or 4096)]
            x = torch.from_numpy(np.nan_to_num(flat[:, sel].T.astype(np.float32))[:, :, None]).to(dev)
            p = torch.softmax(model(x).float(), dim=1)[:, 1]
            prob[sel] = p.cpu().numpy()
    return prob.reshape(H, W)


# ----------------------------------------------------------------- ukis-csmask
def csmask_options(job: Dict) -> Dict:
    o = dict(CSMASK_DEFAULTS)
    if job.get('cloud_csmask_block'):
        o['block'] = int(job['cloud_csmask_block'])
    if job.get('cloud_csmask_level'):
        o['product_level'] = str(job['cloud_csmask_level']).lower()
    return o


def csmask_check(opts: Dict) -> List[str]:
    try:
        import ukis_csmask  # noqa: F401
        return []
    except Exception as e:
        return [f'ukis-csmask is not installed here (pip install ukis-csmask[gpu], or [cpu]): '
                f'{type(e).__name__}: {e}']


def csmask_classes(refl: np.ndarray, valid: np.ndarray, opts: Dict, device: Optional[str] = None) -> np.ndarray:
    """Classes (0 clear, 1 cloud, 3 shadow, 255 no data) from reflectance
    [blue, green, red, nir, H, W] by ukis-csmask, block by block."""
    from ukis_csmask.mask import CSmask
    providers = None
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device and str(device).startswith('cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    H, W = valid.shape
    img = np.moveaxis(np.nan_to_num(refl, nan=0.0), 0, -1).astype(np.float32)
    img[~valid] = 0.0
    out = np.zeros((H, W), np.uint8)
    blk, mg = int(opts.get('block') or 4096), 256
    for y0 in range(0, H, blk):
        for x0 in range(0, W, blk):
            ya, xa = max(0, y0 - mg), max(0, x0 - mg)
            yb, xb = min(H, y0 + blk + mg), min(W, x0 + blk + mg)
            cs = CSmask(img[ya:yb, xa:xb], band_order=['blue', 'green', 'red', 'nir'],
                        product_level=opts.get('product_level', 'l1c'), nodata_value=0, invalid_buffer=0,
                        providers=providers)
            c = np.asarray(cs.csm)[..., 0] if np.asarray(cs.csm).ndim == 3 else np.asarray(cs.csm)
            sub = c[y0 - ya:y0 - ya + min(blk, H - y0), x0 - xa:x0 - xa + min(blk, W - x0)]
            out[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = np.where(sub == 1, 1, np.where(sub == 2, 3, 0))
    out[~valid] = 255
    return out
