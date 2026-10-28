"""
Cloud mask benchmark: every combination of the cloud settings on one or more
images, without a matching run.

    python automatch_job.py cloudsweep <image> [<image> ...] --out <folder>
        [--conf 0.6-0.99:0.1] [--crisp 6] [--no-spec]
        [--bright 60-90:5] [--blob-km2 0.25 1 4] [--shadow-km 5 10 20]
        [--reference <cloud mask raster | polygons .geojson / .shp> ...]
        [--model-dir <weights>/omnicloudmask] [--device cpu] [--max-side 2048]
        [--images grid|all|none] [--sheets]

Ranges are lo-hi:step (the end is kept when the steps miss it: 0.6-0.99:0.1
is 0.6, 0.7, 0.8, 0.9, 0.99) or a list of values.

Band sets: the red / green / NIR bands by the band specs ('spec'), and every
3 of the --crisp N crispest bands (automatch_engine.band_quality), short to
long wavelength as the job's 'quality' choice feeds them; for MV, 6 of 6
gives all 20 combinations. OmniCloudMask runs once per band set; every
threshold x bright-blob setting is then applied to its probabilities (on an
overview of at most --max-side pixels, so a full grid takes minutes), both
the model alone ('check: none') and with the bright-blob check
(automatch_cloud.refine: brighter than the bright percentile of the clear
ground in both visible bands, blobs of min_blob_km2 or more, shadows within
shadow_near_km of a blob).

Per combination (CLOUD_SWEEP.csv): cloud / thick / thin / shadow %, the
masked share (--classes, default thick cloud), cloud blobs, dark_cloud_pct
(the share of cloud darker than the median clear ground: high means ground
taken for cloud) and brightness_gap (how far the clouds' median brightness
lies above the clear ground's 95th percentile, in robust standard deviations
of the clear ground: real clouds stand well above). With a reference cloud
mask -- a raster on any grid (non-zero = cloud) or cloud outlines drawn as
polygons (GeoJSON or shapefile) -- also IoU, precision, recall and F1 of the
masked classes against it; that is the benchmark proper. Without a reference
the numbers describe the masks but cannot say which is right: look at the
quicklooks.

Writes into <folder>/<YYYYmmdd_HHMMSS>_sweep/:
    CLOUD_SWEEP.csv             one row per image x band set x combination
    CLOUD_SWEEP_SUMMARY.csv     per combination over the images (medians;
                                ranked by F1 when there are references)
    <image>/<band set>/conf0.7_model.png, conf0.7_bright75_blob1_shadow10.png ...
                                one quicklook per combination, its settings
                                in a bar above (--images grid: the model alone
                                and each brightness at the middle blob and
                                shadow values, per threshold; all: every
                                combination); the png column of
                                CLOUD_SWEEP.csv names each row's
    SHEET_<image>_<band set>.png  with --sheets: the same as one stacked
                                image (threshold rows x brightness columns)
    CLOUD_SWEEP_PARAMS_<time>.txt  the sweep's settings
    top_<k>_<...>/              with references: the --write-top best
                                combinations at full resolution (mask,
                                quicklook, CLOUD_PARAMS)
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import automatch_native  # noqa: F401,E402  (rasterio's C++ runtime before pandas')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import automatch_cloud as C  # noqa: E402

DEFAULTS = {'conf': '0.6-0.99:0.1', 'bright': '60-90:5', 'blob_km2': [0.25, 1.0, 4.0], 'shadow_km': [5.0, 10.0, 20.0],
            'crisp': 6, 'spec': True, 'classes': [1], 'max_side': 2048, 'write_top': 3}


def parse_range(v) -> List[float]:
    """lo-hi:step (or lo:hi:step), the end kept when the steps miss it, or a
    list of values (text or numbers)."""
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, (list, tuple)):
        vals = [float(x) for x in v]
    else:
        t = str(v).strip()
        m = re.fullmatch(r'([\d.]+)\s*(?:-|:|to)\s*([\d.]+)\s*(?::|/|step)\s*([\d.]+)', t)
        if m:
            a, b, st = (float(x) for x in m.groups())
            if st <= 0 or b < a:
                raise ValueError(f'range {t!r}: expected lo-hi:step with lo <= hi and step > 0')
            n = int(np.floor((b - a) / st + 1e-9))
            vals = [a + i * st for i in range(n + 1)]
            if vals[-1] < b - 1e-9:
                vals.append(b)
        else:
            try:
                vals = [float(x) for x in re.split(r'[,;\s]+', t) if x]
            except ValueError:
                raise ValueError(f'{t!r}: expected lo-hi:step (e.g. 0.6-0.99:0.1) or values (e.g. 0.25, 1, 4)')
    return sorted({round(x, 6) for x in vals})


def band_sets(path: str, sensor: str, spec: bool = True, crisp: int = 6) -> List[Tuple[str, List[int], str, List]]:
    """[(name, bands, how chosen, wavelengths)]: the spec bands and every 3 of
    the crisp crispest bands (short to long wavelength)."""
    import automatch_engine as E
    sets = []
    wl_all = C.band_wavelengths(path, sensor)
    if spec:
        try:
            b, how, wl = C.choose_bands(path, sensor, None, 'spec')
            sets.append(('spec', list(b), how, wl))
        except ValueError as e:
            print(f'[Sweep] {os.path.basename(path)}: no spec bands ({e})')
    if crisp and int(crisp) >= 3:
        top = [int(c[4:]) for c in E.pick_bands(E.band_quality(path), int(crisp))]
        top.sort(key=lambda b: (wl_all[b - 1] if wl_all else b))
        if len(top) < 3:
            print(f'[Sweep] {os.path.basename(path)}: fewer than 3 bands with detail above their noise')
        for combo in itertools.combinations(top, 3):
            sets.append((f"b{'-'.join(str(b) for b in combo)}", list(combo), f'3 of the {len(top)} crispest',
                         [wl_all[b - 1] if wl_all else None for b in combo]))
    seen, out = set(), []
    for s in sets:
        if tuple(s[1]) not in seen:
            seen.add(tuple(s[1]))
            out.append(s)
    return out


def load_reference(ref: str, image: str) -> Tuple[np.ndarray, np.ndarray]:
    """(cloud, known) on the image's grid from a reference: a raster on any
    grid (non-zero = cloud, its no-data unknown) or polygons (GeoJSON,
    shapefile) outlining the clouds (everything else clear)."""
    import rasterio as rio
    from rasterio.warp import reproject, Resampling
    with rio.open(image) as s:
        shape, transform, crs = (s.height, s.width), s.transform, s.crs
    ext = os.path.splitext(ref)[1].lower()
    if ext in ('.geojson', '.json', '.shp'):
        from rasterio import features
        from shapely.geometry import Polygon, shape as to_shape
        from shapely.ops import transform as reshape
        from pyproj import Transformer
        geoms, src_crs = [], 'EPSG:4326'
        if ext == '.shp':
            import automatch_refs as R
            _, feats, _ = R._read_index_builtin(ref)            # outer rings, in lon/lat
            geoms = [Polygon(ring) for _, ring in feats if ring and len(ring) >= 3]
        else:
            with open(ref, encoding='utf-8') as f:
                gj = json.load(f)
            name = (((gj.get('crs') or {}).get('properties') or {}).get('name') or '')
            m = re.search(r'EPSG:+(\d+)', name)
            if m and m.group(1) != '4326':
                src_crs = f'EPSG:{m.group(1)}'
            feats = gj.get('features') or ([gj] if gj.get('type') == 'Feature' else [])
            geoms = [to_shape(ft['geometry']) for ft in feats if ft.get('geometry')]
            if not feats and gj.get('type') in ('Polygon', 'MultiPolygon'):
                geoms = [to_shape(gj)]
        tr = Transformer.from_crs(src_crs, crs, always_xy=True)
        geoms = [reshape(tr.transform, g) for g in geoms if not g.is_empty]
        cloud = features.rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0,
                                   dtype='uint8') > 0 if geoms else np.zeros(shape, bool)
        return cloud, np.ones(shape, bool)
    with rio.open(ref) as r:
        nd = r.nodata
        out = np.full(shape, np.nan, np.float32)
        reproject(rio.band(r, 1), out, src_nodata=nd, dst_nodata=np.nan, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    known = np.isfinite(out)
    return known & (out != 0), known


def _metrics(masked: np.ndarray, ref: Optional[np.ndarray], known: Optional[np.ndarray]) -> Dict:
    if ref is None:
        return {}
    m, r = masked & known, ref & known
    tp, fp, fn = int((m & r).sum()), int((m & ~r).sum()), int((~m & r).sum())
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rc / (p + rc) if p and rc else (0.0 if (tp + fp + fn) else None)
    return {'ref_iou': round(tp / (tp + fp + fn), 4) if tp + fp + fn else None,
            'ref_precision': None if p is None else round(p, 4), 'ref_recall': None if rc is None else round(rc, 4),
            'ref_f1': None if f1 is None else round(f1, 4), 'ref_cloud_pct': round(100.0 * r.sum() / max(known.sum(), 1), 2)}


def _render(a: np.ndarray, cls: np.ndarray, v: np.ndarray, max_side: int, label: str) -> np.ndarray:
    """One combination's quicklook (false colour NIR, red, green; thick cloud
    purple, thin yellow, shadow blue) at most max_side pixels, its settings in
    a bar above."""
    import cv2
    step = max(1, int(np.ceil(max(cls.shape) / float(max_side))))
    aa, cc, vv = a[:, ::step, ::step], cls[::step, ::step], v[::step, ::step]
    chans = []
    for b in (2, 0, 1):
        x = aa[b][vv]
        lo, hi = (np.percentile(x, [2, 98]) if x.size else (0.0, 1.0))
        chans.append(np.clip((aa[b] - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8))
    rgb = np.dstack(chans[::-1]).copy()
    for k, colour in ((1, (200, 0, 200)), (2, (0, 255, 255)), (3, (255, 0, 0))):
        m = cc == k
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array(colour)).astype(np.uint8)
    rgb[~vv] = (40, 40, 40)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    lines, cur = [], ''
    for part in re.split(r'(?<=,) |  ', label):           # wrap at ', ' and double spaces to the image's width
        nxt = f'{cur} {part}'.strip() if cur else part
        if cur and cv2.getTextSize(nxt, font, scale, 1)[0][0] > rgb.shape[1] - 8:
            lines.append(cur)
            nxt = part
        cur = nxt
    lines.append(cur)
    bar = np.full((22 * len(lines) + 6, rgb.shape[1], 3), 255, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (4, 18 + 22 * i), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, rgb])


def _thumb(a: np.ndarray, cls: np.ndarray, v: np.ndarray, side: int, label: str) -> np.ndarray:
    import cv2
    step = max(1, int(np.ceil(max(cls.shape) / float(side))))
    aa, cc, vv = a[:, ::step, ::step], cls[::step, ::step], v[::step, ::step]
    chans = []
    for b in (2, 0, 1):
        x = aa[b][vv]
        lo, hi = (np.percentile(x, [2, 98]) if x.size else (0.0, 1.0))
        chans.append(np.clip((aa[b] - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8))
    rgb = np.dstack(chans[::-1]).copy()
    for k, colour in ((1, (200, 0, 200)), (2, (0, 255, 255)), (3, (255, 0, 0))):
        m = cc == k
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array(colour)).astype(np.uint8)
    rgb[~vv] = (40, 40, 40)
    tile = np.full((side + 22, side, 3), 255, np.uint8)
    h, w = rgb.shape[:2]
    tile[22:22 + min(h, side), :min(w, side)] = rgb[:side, :side]
    cv2.putText(tile, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def sweep_image(path: str, out_dir: str, sensor: str = '', sets: Optional[List] = None,
                confs: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 0.99), brights: Sequence[float] = (60, 70, 80, 90),
                blobs: Sequence[float] = (1.0,), shadows: Sequence[float] = (10.0,), classes: Sequence[int] = (1,),
                model_dir: str = '', device: Optional[str] = None, reference: str = '', predictor=None,
                max_side: int = 2048, sheet_side: int = 260, images: str = 'grid', image_side: int = 1600,
                sheets: bool = False) -> List[Dict]:
    """Every combination on one image; rows of CLOUD_SWEEP.csv. One PNG per
    combination into out_dir/<image>/<band set>/ (images 'grid': the model
    alone and each brightness at the middle blob and shadow values, per
    threshold; 'all': every combination; 'none'); with sheets also the
    stacked SHEET_<image>_<band set>.png."""
    import cv2
    import rasterio as rio
    import automatch_bench as B
    stem = os.path.splitext(os.path.basename(path))[0]
    with rio.open(path) as src:
        nd = src.nodata if src.nodata is not None else 0
        H, W = src.height, src.width
        px_full = C.pixel_m(src.transform, src.crs, src.height)
        if not sensor:
            sensor = B.sensor_of(stem, abs(src.transform.a) if src.crs is not None and src.crs.is_projected else None)
    sets = sets if sets is not None else band_sets(path, sensor)
    step = max(1, int(np.ceil(max(H, W) / float(max_side))))
    px = px_full * step
    ref = known = None
    if reference:
        ref, known = load_reference(reference, path)
        ref, known = ref[::step, ::step], known[::step, ::step]
    mid_blob = sorted(blobs)[len(blobs) // 2]
    mid_shadow = sorted(shadows)[len(shadows) // 2]
    classes = [int(c) for c in classes]
    rows = []
    os.makedirs(out_dir, exist_ok=True)
    for name, bands, how, wl in sets:
        with rio.open(path) as src:
            arr = np.stack([src.read(b).astype(np.float32) for b in bands])
        valid = np.all(np.isfinite(arr) & (arr != nd), axis=0)
        t0 = time.time()
        prob = np.asarray((predictor or C.predict)(np.where(valid, arr, nd), float(nd), model_dir, device))
        secs = round(time.time() - t0, 1)
        a, v = arr[:, ::step, ::step], valid[::step, ::step]
        p = prob[:, ::step, ::step] if prob.ndim == 3 else prob[::step, ::step]
        del arr, prob
        order = list(np.argsort([w if w is not None else i for i, w in enumerate(wl)])) if wl else [0, 1, 2]
        vis = order[:2]
        nv = max(int(v.sum()), 1)
        base = {'image': path, 'sensor': sensor, 'band_set': name, 'bands': ' '.join(str(b) for b in bands),
                'wavelengths_nm': ' '.join(f'{w:.0f}' if w else '?' for w in (wl or [])), 'band_choice': how,
                'model_seconds': secs, 'pixel_m': round(px, 1)}
        print(f"[Sweep] {stem} {name} (bands {bands}): model {secs} s", flush=True)
        tiles: Dict[Tuple[float, str], np.ndarray] = {}

        def row(cls, conf, check, q=None, bk=None, s=None, nblobs=None):
            masked = np.isin(cls, classes) & v
            cloud = np.isin(cls, (1, 2)) & v
            clear = (cls == 0) & v
            pct = {k: round(100.0 * int((cls == k).sum()) / nv, 3) for k in (1, 2, 3)}
            dark = gap = None
            if cloud.any() and clear.any():
                br = (a[vis[0]] + a[vis[1]]) / 2.0
                dark = round(100.0 * float((br[cloud] < np.median(br[clear])).mean()), 1)
                cl = br[clear]
                mad = 1.4826 * float(np.median(np.abs(cl - np.median(cl)))) or 1e-6
                gap = round((float(np.median(br[cloud])) - float(np.percentile(cl, 95))) / mad, 2)
            return {**base, 'confidence': conf, 'check': check, 'bright_pct': q, 'min_blob_km2': bk,
                    'shadow_near_km': s, 'thick_pct': pct[1], 'thin_pct': pct[2], 'shadow_pct': pct[3],
                    'cloud_pct': round(pct[1] + pct[2], 3), 'masked_pct': round(100.0 * masked.sum() / nv, 3),
                    'blobs': nblobs, 'dark_cloud_pct': dark, 'brightness_gap': gap, **_metrics(masked, ref, known)}
        img_dir = os.path.join(out_dir, stem, name)

        def png(cls, label, fname):
            os.makedirs(img_dir, exist_ok=True)
            cv2.imwrite(os.path.join(img_dir, fname), _render(a, cls, v, image_side, f'{stem}  {name}  {label}'))
            return os.path.relpath(os.path.join(img_dir, fname), out_dir)
        for conf in confs:
            c0 = C.classify(p, conf)
            c0 = np.where(v, c0, 255).astype(np.uint8)
            rows.append(row(c0, conf, 'none'))
            if images in ('grid', 'all'):
                rows[-1]['png'] = png(c0, f'confidence {conf:g}, the model alone', f'conf{conf:g}_model.png')
            if sheets:
                tiles[(conf, 'model')] = _thumb(a, c0, v, sheet_side, f'{name} {conf:g} model')
            model_cloud = np.isin(c0, (1, 2)) & v
            shadow0 = (c0 == 3) & v
            clear = v & (c0 == 0)
            refpx = clear if clear.sum() >= 1000 else v
            thr = {i: np.percentile(a[i][refpx], list(brights)) for i in vis}
            for qi, q in enumerate(brights):
                bright = v & (a[vis[0]] > thr[vis[0]][qi]) & (a[vis[1]] > thr[vis[1]][qi])
                cand = model_cloud & bright
                n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
                areas = st[:, cv2.CC_STAT_AREA]
                for bk in blobs:
                    keep = areas >= max(1, int(round(bk * 1e6 / px ** 2)))
                    keep[0] = False
                    cloud = keep[lab] & cand
                    dist = (cv2.distanceTransform((~cloud).astype(np.uint8), cv2.DIST_L2, 5) * px
                            if cloud.any() else None)
                    for s in shadows:
                        near = (np.ones_like(cloud) if s <= 0 else
                                (dist <= s * 1000.0 if dist is not None else np.zeros_like(cloud)))
                        cls = c0.copy()
                        cls[model_cloud & ~cloud] = 0
                        cls[shadow0 & ~near] = 0
                        rows.append(row(cls, conf, 'bright blobs', q, bk, s, int(keep.sum())))
                        middle = bk == mid_blob and s == mid_shadow
                        if images == 'all' or (images == 'grid' and middle):
                            rows[-1]['png'] = png(cls, f'confidence {conf:g}, brighter than {q:g}th pct, blob >= '
                                                       f'{bk:g} km2, shadow within {s:g} km',
                                                  f'conf{conf:g}_bright{q:g}_blob{bk:g}_shadow{s:g}.png')
                        if sheets and middle:
                            tiles[(conf, f'{q:g}')] = _thumb(a, cls, v, sheet_side,
                                                             f'{conf:g} bright {q:g} blob {bk:g} sh {s:g}')
        if sheets:
            cols = ['model'] + [f'{q:g}' for q in brights]
            sheet = np.vstack([np.hstack([tiles.get((c, k), np.full((sheet_side + 22, sheet_side, 3), 255, np.uint8))
                                          for k in cols]) for c in confs])
            cv2.imwrite(os.path.join(out_dir, f'SHEET_{stem}_{name}.png'), sheet)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per combination over the images: medians, how many images; ranked by
    the median F1 when references were given."""
    if df.empty:
        return df
    keys = ['sensor', 'band_set', 'confidence', 'check', 'bright_pct', 'min_blob_km2', 'shadow_near_km']
    metrics = [c for c in ('ref_f1', 'ref_iou', 'ref_precision', 'ref_recall', 'cloud_pct', 'masked_pct',
                           'shadow_pct', 'dark_cloud_pct', 'brightness_gap', 'blobs') if c in df.columns]
    g = df.assign(**{k: df[k].astype(str) for k in keys}).groupby(keys, dropna=False)
    out = g[metrics].median(numeric_only=True).round(4).reset_index()
    out.insert(len(keys), 'images', g['image'].nunique().values)
    if 'ref_f1' in out.columns and out['ref_f1'].notna().any():
        out = out.sort_values(['ref_f1', 'ref_iou'], ascending=False)
    return out


def run(images: List[str], out_dir: str, references: Optional[List[str]] = None, sensor: str = '',
        conf='0.6-0.99:0.1', bright='60-90:5', blob_km2=(0.25, 1.0, 4.0), shadow_km=(5.0, 10.0, 20.0),
        crisp: int = 6, spec: bool = True, classes=(1,), model_dir: str = '', device: Optional[str] = None,
        max_side: int = 2048, write_top: int = 3, predictor=None, pngs: str = 'grid', sheets: bool = False) -> Dict:
    """The sweep over the images into <out_dir>/<time>_sweep/; returns
    {'folder', 'rows', 'summary'}."""
    stamp = C.run_stamp()
    folder = os.path.join(out_dir, f'{stamp}_sweep')
    k = 1
    while os.path.exists(folder):
        k += 1
        folder = os.path.join(out_dir, f'{stamp}_sweep_{k}')
    os.makedirs(folder)
    confs, brights = parse_range(conf), parse_range(bright)
    blobs, shadows = parse_range(blob_km2), parse_range(shadow_km)
    if any(not 0 < c < 1 for c in confs):
        raise ValueError(f'confidence thresholds must lie between 0 and 1, got {confs}')
    refs = list(references or [])
    if refs and len(refs) != len(images):
        raise ValueError(f'{len(refs)} reference(s) for {len(images)} image(s): give one per image, in order')
    C.write_params(folder, {'kind': 'cloud sweep', 'images': '; '.join(os.path.abspath(i) for i in images),
                            'references': '; '.join(refs), 'confidence': ', '.join(f'{c:g}' for c in confs),
                            'band_sets': ('spec + ' if spec else '') + f'every 3 of the {crisp} crispest bands',
                            'bright_pct': ', '.join(f'{b:g}' for b in brights),
                            'min_blob_km2': ', '.join(f'{b:g}' for b in blobs),
                            'shadow_near_km': ', '.join(f'{s:g}' for s in shadows),
                            'masked_classes': ', '.join(str(c) for c in classes), 'max_side_px': max_side,
                            'model_weights': model_dir or "OmniCloudMask's own folder",
                            'device': device or 'GPU when there is one'}, stamp=stamp)
    os.rename(os.path.join(folder, f'CLOUD_PARAMS_{stamp}.txt'), os.path.join(folder, f'CLOUD_SWEEP_PARAMS_{stamp}.txt'))
    rows = []
    for i, img in enumerate(images):
        sets = band_sets(img, sensor or _sensor(img), spec, crisp)
        n = len(confs) * (1 + len(brights) * len(blobs) * len(shadows))
        print(f'[Sweep] {os.path.basename(img)}: {len(sets)} band set(s) x {n} combination(s)', flush=True)
        rows += sweep_image(img, folder, sensor, sets, confs, brights, blobs, shadows, classes, model_dir, device,
                            refs[i] if refs else '', predictor, max_side, images=pngs, sheets=sheets)
        pd.DataFrame(rows).to_csv(os.path.join(folder, 'CLOUD_SWEEP.csv'), index=False)    # as it goes
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(folder, 'CLOUD_SWEEP.csv'), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(folder, 'CLOUD_SWEEP_SUMMARY.csv'), index=False)
    if refs and write_top and 'ref_f1' in df.columns and df['ref_f1'].notna().any():
        top = df.sort_values(['ref_f1', 'ref_iou'], ascending=False).head(int(write_top))
        for k, r in enumerate(top.to_dict('records'), 1):
            sub = os.path.join(folder, f"top_{k}_{os.path.splitext(os.path.basename(r['image']))[0]}_{r['band_set']}"
                                       f"_{r['confidence']:g}")
            ropts = None if r['check'] == 'none' else {'bright_pct': r['bright_pct'], 'min_blob_km2': r['min_blob_km2'],
                                                       'shadow_near_km': r['shadow_near_km']}
            wl = [None if w == '?' else float(w) for w in str(r.get('wavelengths_nm') or '').split()] or None
            C.compute(r['image'], [int(b) for b in str(r['bands']).split()], sub, model_dir, device, predictor,
                      r['confidence'], ropts, wl, stamp)
            C.write_params(sub, {'rank': k, 'ref_f1': r.get('ref_f1'), 'ref_iou': r.get('ref_iou')}, merge=True,
                           stamp=stamp)
    print(format_summary(s))
    print(f'[Sweep] {len(df)} combination(s) in {folder} (CLOUD_SWEEP.csv, CLOUD_SWEEP_SUMMARY.csv; one PNG per '
          f'combination shown in <image>/<band set>/, its name in the png column)',
          flush=True)
    return {'folder': folder, 'rows': df, 'summary': s}


def _sensor(img: str) -> str:
    import rasterio as rio
    import automatch_bench as B
    with rio.open(img) as s:
        px = abs(s.transform.a) if s.crs is not None and s.crs.is_projected else None
    return B.sensor_of(os.path.splitext(os.path.basename(img))[0], px)


def format_summary(s: pd.DataFrame, top: int = 15) -> str:
    if s.empty:
        return '[Sweep] nothing to summarise'
    if 'ref_f1' in s.columns and s['ref_f1'].notna().any():
        lines = ['[Sweep] best combinations against the reference masks (median over the images):',
                 f"  {'band set':<10} {'conf':>5} {'check':<13} {'bright':>6} {'blob':>5} {'shadow':>6} "
                 f"{'F1':>6} {'IoU':>6} {'prec':>6} {'recall':>6} {'cloud %':>8}"]
        for r in s.head(top).itertuples():
            lines.append(f"  {r.band_set:<10} {r.confidence:>5} {r.check:<13} {r.bright_pct:>6} {r.min_blob_km2:>5} "
                         f"{r.shadow_near_km:>6} {r.ref_f1:>6} {r.ref_iou:>6} {r.ref_precision:>6} {r.ref_recall:>6} "
                         f"{r.cloud_pct:>8}")
        return '\n'.join(lines)
    return ('[Sweep] no reference masks: CLOUD_SWEEP_SUMMARY.csv describes every combination (cloud %, '
            'dark_cloud_pct, brightness_gap) but cannot say which is right -- look at the PNGs in <image>/<band '
            'set>/, or give --reference cloud outlines for a few images')


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Cloud mask benchmark: every combination of the cloud settings')
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', default='', help='output folder (default: beside the first image)')
    ap.add_argument('--reference', nargs='*', default=[],
                    help='reference cloud masks, one per image in order: raster (non-zero = cloud) or polygons '
                         '(.geojson / .shp)')
    ap.add_argument('--sensor', default='')
    ap.add_argument('--conf', default=DEFAULTS['conf'], help='thresholds: lo-hi:step or values')
    ap.add_argument('--bright', default=DEFAULTS['bright'], help='bright percentiles: lo-hi:step or values')
    ap.add_argument('--blob-km2', nargs='+', default=[str(v) for v in DEFAULTS['blob_km2']])
    ap.add_argument('--shadow-km', nargs='+', default=[str(v) for v in DEFAULTS['shadow_km']])
    ap.add_argument('--crisp', type=int, default=DEFAULTS['crisp'], help='every 3 of the N crispest bands (0: none)')
    ap.add_argument('--no-spec', action='store_true', help='leave out the spec bands')
    ap.add_argument('--classes', type=int, nargs='+', default=DEFAULTS['classes'],
                    help='classes compared with the reference: 1 thick, 2 thin, 3 shadow')
    ap.add_argument('--model-dir', default='')
    ap.add_argument('--device', default=None)
    ap.add_argument('--max-side', type=int, default=DEFAULTS['max_side'])
    ap.add_argument('--write-top', type=int, default=DEFAULTS['write_top'])
    ap.add_argument('--images', dest='pngs', choices=['grid', 'all', 'none'], default='grid',
                    help="one PNG per combination: grid = the model alone and each brightness at the middle blob "
                         "and shadow values (default); all = every combination (thousands); none")
    ap.add_argument('--sheets', action='store_true', help='also the stacked contact sheets SHEET_*.png')
    a = ap.parse_args(argv)
    problems = C.check({'cloud_model_dir': a.model_dir})
    if problems:
        print('\n'.join(problems))
        return 2
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.images[0])), 'cloud_sweep')
    run(a.images, out, a.reference, a.sensor, a.conf, a.bright, ' '.join(a.blob_km2), ' '.join(a.shadow_km),
        a.crisp, not a.no_spec, a.classes, a.model_dir, a.device, a.max_side, a.write_top, pngs=a.pngs,
        sheets=a.sheets)
    return 0


if __name__ == '__main__':
    sys.exit(main())
