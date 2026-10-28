"""
Training pairs from scenes with manual tie points.

Each scene of the manifest (scenes.csv) is fitted (fit.py). For an accepted
scene, crops of crop_px x crop_px pixels are cut inside the area its tie
points cover (grown by 15 %), each at its own pixel size, and for each crop
the reference is resampled onto a north-up grid of that pixel size, around
the crop's true ground position shifted at random by up to shift_frac of the
crop (the misalignment a matcher meets after a coarse offset). The crop pair
comes with H_0to1, the 3 x 3 homography taking input-crop pixels to
reference-crop pixels (pixel centres at integers, as OpenCV and Glue Factory
use), fitted to the scene model on an 11 x 11 grid. Crops whose model is not
(nearly) projective over the crop (grid residual > 0.5 px), too little valid
data, or, when a cloud mask is given, too much cloud are skipped.

Pixel size: nothing to set. A pair can be no finer than the coarser of its
two images (a 30 m reference holds no 6 m detail), and no coarser than lets
a crop fit inside the points' area. Between the two, up to max_scale times
the finer limit, every crop draws its own pixel size (evenly in log scale),
so each sensor is seen at many resolutions; the pixel size goes with the
pair, and the conditioned matcher is told it. res_m (option or manifest
column) narrows the range: a number fixes it, lo-hi limits it.

    python pairs.py --manifest scenes.csv --out pairs/ [--crop 512 --per-scene 40]

manifest columns:
    scene_id, sensor, date, input_path, input_band, gt_path, reference
    (a raster or a folder of rasters, e.g. L8_ref), reference_band,
    [modality] or [wavelength_nm] of the input band (modality.py: visible,
    nir, swir, mwir, lwir, sar; neither: SAR sensors by name, else visible),
    [reference_modality] or [reference_wavelength_nm] (default visible),
    [cloud_mask] (a raster on the input grid: 0 clear, 1-3 cloud / shadow,
    e.g. CLOUD_MASK.tif from automatch), [split] (train / val / test, else
    assigned by scene), [res_m] (this scene's pixel size or lo-hi range)
    Several rows may share a scene_id (other bands of the same image): they
    share its split.

writes:
    pairs/<scene_id>/b<band>_<k>_0.png, _1.png  (8-bit, stretched as the matching tool does)
    pairs/<scene_id>/b<band>_<k>_m0.png, _m1.png (valid masks)
    pairs/pairs.csv    one row per pair: scene, sensor, split, files, H_0to1,
                       px_m, modality0 (input), modality1 (reference)
    pairs/scenes_fit.csv   per scene: fit type, leave-one-out RMSE (px), pixel
                       sizes and range, used or why not
    pairs/splits.csv   scene -> split: keep the test scenes out of any training
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fit as F  # noqa: E402
import geo  # noqa: E402
import gt as G  # noqa: E402
import modality as M  # noqa: E402


def assign_split(scene_id: str, val: float = 0.1, test: float = 0.2) -> str:
    """Deterministic split by scene (never by crop: crops of one scene look alike)."""
    h = int(hashlib.sha1(str(scene_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if h < test else ('val' if h < test + val else 'train')


def _int(v, default: int = 1) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def res_spec(v) -> Optional[Tuple[float, float]]:
    """(lo, hi) pixel-size limits in metres from a number (fixed), a [lo, hi]
    list or 'lo-hi' text; None (no limits) when empty, 0 or 'auto'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v]
    else:
        vals = [float(x) for x in re.findall(r'\d+(?:\.\d*)?', str(v))]
    vals = [x for x in vals if x > 0]
    return (min(vals), max(vals)) if vals else None


def crop_homography(model: F.SceneModel, src, col0: float, row0: float, size: int, ref_x0: float, ref_y1: float,
                    px: float, factor: float = 1.0) -> (np.ndarray, float):
    """H_0to1 (input-crop pixel -> reference-crop pixel) and its largest grid
    residual (px). A crop pixel is factor input pixels wide; its centre u lies
    at input pixel col0 + (u + 0.5) * factor - 0.5."""
    import cv2
    g = np.linspace(0, size - 1, 11)
    uu, vv = np.meshgrid(g, g)
    E, N = model(col0 + (uu + 0.5) * factor - 0.5, row0 + (vv + 0.5) * factor - 0.5)
    xr = (E - ref_x0) / px - 0.5
    yr = (ref_y1 - N) / px - 0.5
    srcp = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
    dstp = np.column_stack([xr.ravel(), yr.ravel()]).astype(np.float64)
    H, _ = cv2.findHomography(srcp, dstp, 0)
    p = np.column_stack([srcp, np.ones(len(srcp))]) @ H.T
    p = p[:, :2] / p[:, 2:3]
    return H, float(np.hypot(*(p - dstp).T).max())


def build_scene(rec: Dict, out_dir: str, crop: int, per_scene: int, shift_frac: float, min_valid: float,
                max_cloud: float, max_loo_px: float, ref_cache: Dict, rng: np.random.Generator,
                res_m=None, max_scale: float = 16.0, prefix: Optional[str] = None) -> (Dict, List[Dict]):
    import cv2
    import rasterio as rio
    sid = str(rec['scene_id'])
    info = {'scene_id': sid, 'sensor': rec.get('sensor', ''), 'date': rec.get('date', ''), 'used': False}
    try:
        m_in = M.resolve(rec.get('modality'), rec.get('wavelength_nm'), rec.get('sensor', ''))
        m_ref = M.resolve(rec.get('reference_modality'), rec.get('reference_wavelength_nm'))
        info.update(modality=m_in, reference_modality=m_ref)
        pts = G.load_points(rec['gt_path'], quiet=True)
        res = F.fit_scene(pts, rec['input_path'], max_loo_px=max_loo_px)
    except Exception as e:
        info['why'] = f'{type(e).__name__}: {e}'
        return info, []
    info.update({k: res.get(k) for k in ('type', 'loo_rmse_px', 'n_used', 'n_points', 'why')})
    info['dropped'] = ' '.join(str(p) for p in res.get('dropped', []))
    if res.get('mean_error_m'):
        info['mean_error_dE_m'], info['mean_error_dN_m'] = res['mean_error_m']
    if not res['ok']:
        return info, []
    model = res['model']
    ref = str(rec['reference'])
    lons = np.r_[pts['ref_lon'], pts['in_lon']]
    lats = np.r_[pts['ref_lat'], pts['in_lat']]
    grow = 0.5 + 0.5 * max(np.ptp(lons), np.ptp(lats))
    bbox = (lons.min() - grow, lats.min() - grow, lons.max() + grow, lats.max() + grow)
    refs = [ref] if os.path.isfile(ref) else geo.references_for(ref, bbox, ref_cache)
    if not refs:
        info['why'] = 'no reference raster covers the tie points'
        return info, []
    rband = _int(rec.get('reference_band'))
    ib = _int(rec.get('input_band'))
    prefix = prefix or f'b{ib}'
    c0, r0, c1, r1 = res['bounds_px']
    gw, gh = (c1 - c0) * 0.15, (r1 - r0) * 0.15
    cmask = None
    if rec.get('cloud_mask') and isinstance(rec['cloud_mask'], str) and os.path.exists(rec['cloud_mask']):
        cmask = rio.open(rec['cloud_mask'])
    rows = []
    with rio.open(rec['input_path']) as src:
        ac0, ac1 = max(0.0, c0 - gw), min(float(src.width), c1 + gw)     # where crops may lie
        ar0, ar1 = max(0.0, r0 - gh), min(float(src.height), r1 + gh)
        in_px = model.pixel_m((ac0 + ac1) / 2.0, (ar0 + ar1) / 2.0)      # true ground pixel, from the fit
        ref_px = max(geo.pixel_m(p) for p in refs)
        floor = max(in_px, ref_px)                                       # finer than this, one side is blur
        cap = in_px * min(ac1 - ac0, ar1 - ar0) / crop                   # coarser, a crop leaves the area
        spec = res_spec(rec.get('res_m')) or res_spec(rec.get('work_res_m')) or res_spec(res_m)
        lo, hi = (floor, floor * max_scale) if spec is None else (max(spec[0], floor), max(spec[1], floor))
        hi = min(hi, cap)
        info.update(input_px_m=round(in_px, 3), reference_px_m=round(ref_px, 3))
        if hi < lo * 0.999:
            info['why'] = (f'the tie points cover less than one {crop} px crop at {lo:.4g} m (the image is '
                           f'{in_px:.4g} m, the reference {ref_px:.4g} m): a smaller --crop would fit')
            return info, []
        fixed = hi / lo < 1.001
        info['res_m'] = f'{lo:.4g}' if fixed else f'{lo:.4g}-{hi:.4g}'
        os.makedirs(os.path.join(out_dir, sid), exist_ok=True)
        tries = 0
        while len(rows) < per_scene and tries < per_scene * 8:
            tries += 1
            px = lo if fixed else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            factor = px / in_px                                          # input pixels per crop pixel (>= 1)
            span = crop * factor
            lo_c, hi_c = int(np.ceil(ac0)), int(ac1 - span)
            lo_r, hi_r = int(np.ceil(ar0)), int(ar1 - span)
            if hi_c < lo_c or hi_r < lo_r:
                continue
            col0, row0 = int(rng.integers(lo_c, hi_c + 1)), int(rng.integers(lo_r, hi_r + 1))
            a0, v0 = geo.read_scaled(src, ib, col0, row0, crop, factor)
            if v0.mean() < min_valid:
                continue
            cloud = 0.0
            if cmask is not None:
                m, mv = geo.read_window(cmask, 1, col0, row0, int(round(span)))
                cloud = float(np.isin(m, (1, 2, 3)).mean())
                if cloud > max_cloud:
                    continue
            Ec, Nc = model(col0 + span / 2.0 - 0.5, row0 + span / 2.0 - 0.5)
            sx, sy = rng.uniform(-shift_frac, shift_frac, 2) * crop * px
            x0 = float(Ec) + sx - crop * px / 2.0
            y1 = float(Nc) + sy + crop * px / 2.0
            a1, v1 = geo.reproject_to_grid([(p, rband) for p in refs], model.crs, x0, y1, px, crop)
            if v1.mean() < min_valid:
                continue
            H, err = crop_homography(model, src, col0, row0, crop, x0, y1, px, factor)
            if err > 0.5:
                continue
            k = len(rows)
            base = os.path.join(out_dir, sid, f'{prefix}_{k:04d}')
            cv2.imwrite(base + '_0.png', geo.stretch_u8(a0, v0))
            cv2.imwrite(base + '_1.png', geo.stretch_u8(a1, v1))
            cv2.imwrite(base + '_m0.png', (v0 * 255).astype(np.uint8))
            cv2.imwrite(base + '_m1.png', (v1 * 255).astype(np.uint8))
            rows.append({'scene_id': sid, 'sensor': rec.get('sensor', ''), 'date': rec.get('date', ''), 'band': ib,
                         'img0': os.path.relpath(base + '_0.png', out_dir), 'img1': os.path.relpath(base + '_1.png', out_dir),
                         'mask0': os.path.relpath(base + '_m0.png', out_dir), 'mask1': os.path.relpath(base + '_m1.png', out_dir),
                         'H': ' '.join(f'{v:.10g}' for v in H.ravel()), 'px_m': round(px, 3),
                         'modality0': m_in, 'modality1': m_ref, 'col0': col0, 'row0': row0,
                         'shift_x_px': round(sx / px, 1), 'shift_y_px': round(-sy / px, 1), 'cloud': round(cloud, 3),
                         'h_residual_px': round(err, 3)})
    if cmask is not None:
        cmask.close()
    info['used'] = bool(rows)
    info['pairs'] = len(rows)
    if not rows:
        info['why'] = 'no crop with enough valid data in input and reference'
    return info, rows


def build(manifest: str, out_dir: str, crop: int = 512, per_scene: int = 40, shift_frac: float = 0.25,
          min_valid: float = 0.6, max_cloud: float = 0.5, max_loo_px: float = 1.5, val: float = 0.1,
          test: float = 0.2, seed: int = 0, res_m=None, max_scale: float = 16.0) -> Dict:
    man = pd.read_csv(manifest)
    os.makedirs(out_dir, exist_ok=True)
    cache_p = os.path.join(out_dir, 'reference_footprints.json')
    try:
        ref_cache = json.load(open(cache_p))
    except (OSError, ValueError):
        ref_cache = {}
    rng = np.random.default_rng(seed)
    infos, rows = [], []
    scene_split: Dict[str, str] = {}
    seen: Dict[str, int] = {}
    for rec in man.to_dict('records'):
        sid = str(rec['scene_id'])
        split = str(rec.get('split') or '').strip().lower()
        if split not in ('train', 'val', 'test'):
            split = assign_split(sid, val, test)
        if scene_split.setdefault(sid, split) != split:
            print(f'[Pairs] {sid}: the rows of one scene share a split; {scene_split[sid]} kept')
            split = scene_split[sid]
        tag = f"b{_int(rec.get('input_band'))}"
        seen[sid + tag] = seen.get(sid + tag, 0) + 1
        prefix = tag + (f'_{seen[sid + tag]}' if seen[sid + tag] > 1 else '')
        info, r = build_scene(rec, out_dir, crop, per_scene, shift_frac, min_valid, max_cloud, max_loo_px,
                              ref_cache, rng, res_m, max_scale, prefix)
        info['split'] = split
        for x in r:
            x['split'] = split
        infos.append(info)
        rows += r
        print(f"[Pairs] {sid} ({info.get('sensor', '')}, {split}): "
              + (f"{info.get('pairs', 0)} pair(s) at {info.get('res_m')} m, {info.get('modality')} -> "
                 f"{info.get('reference_modality')}; {info.get('type')} fit, leave-one-out "
                 f"{info.get('loo_rmse_px')} px on {info.get('n_used')} point(s)" if info['used']
                 else f"not used: {info.get('why')}"), flush=True)
    json.dump(ref_cache, open(cache_p, 'w'))
    pd.DataFrame(infos).to_csv(os.path.join(out_dir, 'scenes_fit.csv'), index=False)
    pd.DataFrame([{'scene_id': i['scene_id'], 'sensor': i.get('sensor', ''), 'split': i['split']} for i in infos]
                 ).to_csv(os.path.join(out_dir, 'splits.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'pairs.csv'), index=False)
    n = pd.DataFrame(rows)['split'].value_counts().to_dict() if rows else {}
    print(f"[Pairs] {sum(1 for i in infos if i['used'])} of {len(infos)} scene(s) used; pairs per split {n}")
    return {'scenes': infos, 'pairs': len(rows), 'per_split': n}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Training pairs from scenes with manual tie points')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--per-scene', type=int, default=40)
    ap.add_argument('--shift', type=float, default=0.25, help='random misalignment, fraction of the crop')
    ap.add_argument('--min-valid', type=float, default=0.6)
    ap.add_argument('--max-cloud', type=float, default=0.5)
    ap.add_argument('--max-loo-px', type=float, default=1.5)
    ap.add_argument('--val', type=float, default=0.1)
    ap.add_argument('--test', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', default='',
                    help='pixel size limits in m: 45 (fixed) or 45-315 (a range); empty: from each scene\'s images')
    ap.add_argument('--max-scale', type=float, default=16.0,
                    help='without --res: pixel sizes up to this many times the coarser image\'s')
    a = ap.parse_args(argv)
    r = build(a.manifest, a.out, a.crop, a.per_scene, a.shift, a.min_valid, a.max_cloud, a.max_loo_px, a.val, a.test,
              a.seed, a.res or None, a.max_scale)
    return 0 if r['pairs'] else 1


if __name__ == '__main__':
    sys.exit(main())
