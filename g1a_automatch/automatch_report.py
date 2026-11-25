"""
What RIVAL would make of a job's final points, made here: for every
RIVAL_FINAL_<band>.csv and RIVAL_BEST_FINAL_*.csv of a job folder (or any
RIVAL CSV named)

    <job>/REPORT/<name>.shp / .shx / .dbf / .prj   the points (WGS 84) with
                                                   their errors as attributes
    <job>/REPORT/<name>_vectors.png                look vectors: an arrow per
                                                   point from input to reference
                                                   position, coloured by length
    <job>/REPORT/REPORT_STATS.csv                  RIVAL's statistics per file:
                                                   n, mean / median / std of the
                                                   errors, RMSE (about zero),
                                                   CE90, largest error

    python automatch_job.py report <job folder | RIVAL csv> [--all]
        --all: every RIVAL_*.csv of the folder, not only the FINAL ones

Made at the end of every job and by 'finish' and 'cross'; this command makes
them for a finished job. The shapefile is written here (no GDAL needed).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import struct
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
FIELDS = [('ROW', 'N', 9, 0), ('IN_X', 'N', 18, 3), ('IN_Y', 'N', 18, 3), ('REF_X', 'N', 18, 3),
          ('REF_Y', 'N', 18, 3), ('DX_ERR', 'N', 14, 3), ('DY_ERR', 'N', 14, 3), ('DIST', 'N', 14, 3),
          ('REF_LON', 'N', 14, 7), ('REF_LAT', 'N', 14, 7)]


def read_points(csv_path: str) -> List[Dict]:
    """RIVAL's CSV rows as dicts of floats (In_X .. Ref_Lat)."""
    import csv
    out = []
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                rec = {k: float(r[k]) for k in ('In_X', 'In_Y', 'Ref_X', 'Ref_Y', 'In_Lon', 'In_Lat', 'Ref_Lon',
                                                'Ref_Lat')}
            except (KeyError, ValueError):
                continue
            rec['Row'] = int(float(r.get('Row') or len(out) + 1))
            rec['DX_Err'] = rec['In_X'] - rec['Ref_X']
            rec['DY_Err'] = rec['In_Y'] - rec['Ref_Y']
            out.append(rec)
    return out


def stats(points: List[Dict]) -> Dict:
    """RIVAL's statistics of a point set (automatch_refs.accuracy_stats: RMSE
    about zero, CE90 from the mean of the two axes) plus mean, median, std
    and the largest error."""
    import numpy as np
    import automatch_refs as refs
    dx = [p['DX_Err'] for p in points]
    dy = [p['DY_Err'] for p in points]
    st = refs.accuracy_stats(dx, dy)
    out = {'n': st['n'], 'rmse_x_m': st['rmse_x'], 'rmse_y_m': st['rmse_y'], 'ce90_m': st['ce90'],
           'ce90_circular': st['circular']}
    if dx:
        a, b = np.asarray(dx), np.asarray(dy)
        d = np.hypot(a, b)
        out.update({'mean_dx_m': float(a.mean()), 'mean_dy_m': float(b.mean()), 'median_dx_m': float(np.median(a)),
                    'median_dy_m': float(np.median(b)), 'std_dx_m': float(a.std()), 'std_dy_m': float(b.std()),
                    'mean_dist_m': float(d.mean()), 'max_dist_m': float(d.max()),
                    'rmse_m': float(math.sqrt((a * a + b * b).mean()))})
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def write_shapefile(points: List[Dict], stem: str) -> str:
    """A point shapefile (WGS 84 lon / lat of the input position) with the
    errors as attributes: stem.shp / .shx / .dbf / .prj. Returns the .shp."""
    recs = [(p['In_Lon'], p['In_Lat']) for p in points]
    if recs:
        xs, ys = zip(*recs)
        box = (min(xs), min(ys), max(xs), max(ys))
    else:
        box = (0.0, 0.0, 0.0, 0.0)
    rec_len = (4 + 16) // 2                        # shape type + x, y, in 16-bit words
    content = 100 + len(recs) * (8 + rec_len * 2)
    with open(stem + '.shp', 'wb') as shp, open(stem + '.shx', 'wb') as shx:
        for f, length in ((shp, content), (shx, 100 + 8 * len(recs))):
            f.write(struct.pack('>iiiiiii', 9994, 0, 0, 0, 0, 0, length // 2))
            f.write(struct.pack('<ii', 1000, 1))
            f.write(struct.pack('<dddddddd', *box, 0.0, 0.0, 0.0, 0.0))
        offset = 50
        for i, (x, y) in enumerate(recs, 1):
            shp.write(struct.pack('>ii', i, rec_len))
            shp.write(struct.pack('<idd', 1, x, y))
            shx.write(struct.pack('>ii', offset, rec_len))
            offset += 4 + rec_len
    with open(stem + '.dbf', 'wb') as dbf:
        n_rec = len(recs)
        rec_size = 1 + sum(w for _, _, w, _ in FIELDS)
        head = 32 + 32 * len(FIELDS) + 1
        dbf.write(struct.pack('<BBBBIHH20x', 3, 26, 1, 1, n_rec, head, rec_size))
        for name, typ, width, dec in FIELDS:
            dbf.write(struct.pack('<11sc4xBB14x', name.encode('ascii'), typ.encode('ascii'), width, dec))
        dbf.write(b'\r')
        for p in points:
            row = b' '
            vals = (p['Row'], p['In_X'], p['In_Y'], p['Ref_X'], p['Ref_Y'], p['DX_Err'], p['DY_Err'],
                    math.hypot(p['DX_Err'], p['DY_Err']), p['Ref_Lon'], p['Ref_Lat'])
            for (name, typ, width, dec), v in zip(FIELDS, vals):
                text = f'{v:.{dec}f}' if dec else f'{int(round(v))}'
                row += text.rjust(width)[-width:].encode('ascii')
            dbf.write(row)
        dbf.write(b'\x1a')
    with open(stem + '.prj', 'w', encoding='ascii') as f:
        f.write(WGS84_PRJ)
    return stem + '.shp'


def plot_context(job_dir: str, truth_csv: str = '') -> Dict:
    """What a job's look-vector plots share: the working CRS, the scene's
    extent and outline, the product's quicklook PNG (from the product folder,
    else an overview of the input) as background, the truth points and the
    title Satellite - Sensor - Session - Strip from the product metadata.
    Anything that cannot be read here is left out."""
    ctx: Dict = {'title': '', 'extent': None, 'background': None, 'truth': None, 'crs': ''}
    try:
        import json
        import automatch_cross as X
        import automatch_presets as P
        import automatch_truth as T
        ctx['crs'] = T.working_crs_of(job_dir) or ''
        with open(os.path.join(job_dir, 'job_used.json'), encoding='utf-8') as f:
            job = json.load(f)
        inp = job.get('input_path') or ''
        loc = next((p for p in (inp, X.local_path(inp, job_dir, job.get('output_dir') or ''), X._by_share(inp))
                    if p and os.path.isfile(p)), '')
        meta = P.product_meta(loc) if loc else {}
        name = os.path.basename(inp)
        sat = meta.get('SatID') or meta.get('SatelliteID') or meta.get('Satellite') or meta.get('Mission') or (
            name.split('_')[0] if name.startswith(('G1A', 'C3', 'E0')) else '')
        sensor = meta.get('Sensor') or (name.split('_')[1] if name.startswith('G1A_') else '')
        session = (meta.get('SessionNumber') or meta.get('SessionNo') or meta.get('Session')
                   or meta.get('ImagingOrbitNo') or '')
        strip = meta.get('StripNumber') or meta.get('SegmentNumber') or meta.get('StripNo') or ''
        dop = meta.get('DateOfPass') or meta.get('DateofPass') or meta.get('DOP') or ''
        try:
            from datetime import datetime as _dt
            dop = _dt.strptime(str(dop).strip(), '%d-%b-%Y').strftime('%d-%b-%Y')     # 17-Nov-2026
        except ValueError:
            pass
        ss = '/'.join(str(v) for v in (session, strip) if str(v).strip())
        ctx['title'] = (f'{sat}: ' if sat else '') + ' '.join(str(v) for v in (sensor, dop, ss) if str(v).strip())
        if loc and ctx['crs']:
            ctx.update(_scene_background(loc, ctx['crs'], prod_extent(meta, ctx['crs'])))
        tcsv = truth_csv or X.job_truth_csv(job_dir)
        if tcsv and ctx['crs']:
            t = T.truth_in_crs(T.load_truth(tcsv), ctx['crs'])
            ctx['truth'] = t[['E', 'N', 'dE', 'dN']].copy()
    except Exception as e:
        print(f'[Report] plot context partly unavailable ({type(e).__name__}: {e})')
    return ctx


def prod_extent(meta: Dict, crs: str):
    """(E0, E1, N0, N1) in crs of the product's corners (ProdULLat / ProdULLon
    ... ProdLLLon of BAND_META), or None when the metadata does not give them."""
    try:
        lat = [float(meta[f'Prod{c}Lat']) for c in ('UL', 'UR', 'LR', 'LL')]
        lon = [float(meta[f'Prod{c}Lon']) for c in ('UL', 'UR', 'LR', 'LL')]
    except (KeyError, ValueError, TypeError):
        return None
    from pyproj import CRS, Transformer
    xs, ys = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True).transform(lon, lat)
    return (min(xs), max(xs), min(ys), max(ys))


def _scene_background(raster: str, crs: str, prod_ext=None) -> Dict:
    """{'extent': (E0, E1, N0, N1) m, 'background': (rgb array, extent)}: the
    extent from the product corners (prod_ext, BAND_META Prod*) when given,
    else the raster's bounds; the product's quicklook PNG (a .png / .jpg in the product folder, the
    largest; 'QL' / 'browse' names first), laid over the raster's extent;
    else an overview of the raster's first band."""
    import glob as _glob
    import numpy as np
    import rasterio as rio
    from pyproj import CRS, Transformer
    from rasterio.enums import Resampling
    with rio.open(raster) as src:
        b, rcrs = src.bounds, src.crs
        s = max(1.0, max(src.width, src.height) / 2000.0)
        ov = src.read(1, out_shape=(max(1, int(src.height / s)), max(1, int(src.width / s))),
                      resampling=Resampling.average).astype('float32')
        nod = src.nodata
    same = rcrs is not None and CRS.from_user_input(rcrs).equals(CRS.from_user_input(crs))
    if same:
        ext = (b.left, b.right, b.bottom, b.top)
    else:
        tr = Transformer.from_crs(rcrs, CRS.from_user_input(crs), always_xy=True)
        xs, ys = tr.transform([b.left, b.right, b.right, b.left], [b.top, b.top, b.bottom, b.bottom])
        ext = (min(xs), max(xs), min(ys), max(ys))
    if prod_ext is not None:                          # the product's own corners (BAND_META), as the plots use
        ext = prod_ext
    folder = os.path.dirname(os.path.abspath(raster))
    pngs = [p for p in _glob.glob(os.path.join(folder, '*')) if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pngs.sort(key=lambda p: (not any(k in os.path.basename(p).upper() for k in ('QL', 'BROWSE', 'QUICK')),
                             -os.path.getsize(p)))
    img = None
    if pngs:
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(pngs[0])
        except Exception:
            img = None
    if img is None:                                   # the raster itself, stretched
        valid = np.isfinite(ov) & (ov != 0) & ((ov != nod) if nod is not None else True)
        if valid.any():
            lo, hi = np.percentile(ov[valid], [2, 98])
            g = np.clip((ov - lo) / max(hi - lo, 1e-9), 0, 1)
            img = np.dstack([g, g, g, valid.astype('float32')])
    return {'extent': ext, 'background': (img, ext) if img is not None else None}


# The look-vector plot's settings, as DPQED_rival.py (plot_look_vectors) on the
# l-band-accuracy-cog branch: bright colours only (cyan -> green -> yellow ->
# orange -> red) so every arrow stands out on the scene, white point markers,
# 'Error (km)' over a short colour bar, bold 16 pt axis labels and bold ticks,
# the RMSE under the title and Mean / Std / CE90 below the plot.
BRIGHT = ((0, 200, 255), (0, 220, 120), (255, 230, 0), (255, 140, 0), (255, 40, 40))


def error_unit(errors_m) -> str:
    """'m' when the errors are metres (90 % of them under 1 km), else 'km'."""
    import numpy as np
    e = np.asarray([v for v in errors_m if v == v], float)
    return 'm' if len(e) and float(np.percentile(np.abs(e), 90)) < 1000.0 else 'km'


def _len(v, unit: str = 'km') -> str:
    v = float(v or 0)
    return f'{v:.0f} m' if unit == 'm' else f'{v / 1000.0:.3f} km'


def _ne(st: Dict, ky: str, kx: str, unit: str = 'km') -> str:
    return f"{_len(st.get(ky), unit)} N / {_len(st.get(kx), unit)} E"


def vectors_title(title: str, st: Optional[Dict], unit: str = 'km') -> str:
    """'<title>' over 'RMSE n km N / e km E' (DPQED_rival.vectors_title); in metres when unit is 'm'."""
    if not (st and st.get('n')):
        return title
    rm = f"RMSE {_ne(st, 'rmse_y_m', 'rmse_x_m', unit)}"
    return f'{title}\n{rm}' if title else rm


def vectors_footer(st: Optional[Dict], unit: str = 'km') -> str:
    """'Mean ...   |   Std ...   |   CE90 c km' (DPQED_rival.vectors_footer); in metres when unit is 'm'."""
    if not (st and st.get('n')):
        return ''
    return (f"n {st['n']}   |   Mean {_ne(st, 'mean_dy_m', 'mean_dx_m', unit)}   |   "
            f"Std {_ne(st, 'std_dy_m', 'std_dx_m', unit)}   |   CE90 {_len(st.get('ce90_m'), unit)}"
            + ('' if st.get('ce90_circular', True) else ' (axes unequal)'))


def plot_vectors(points: List[Dict], png: str, title: str = '', st: Optional[Dict] = None,
                 ctx: Optional[Dict] = None) -> str:
    """Look vectors: an arrow per point from its input position towards its
    reference position (the geolocation error), coloured by length, on the
    product's quicklook; the truth points (manual) marked with their own
    arrows. The figure takes the scene's shape (a long HS strip: a long
    figure), the colour scale lies along its longer edge, the statistics
    below it, the title Satellite - Sensor - Session - Strip."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    ctx = ctx or {}
    x = np.asarray([p['In_X'] for p in points], float) / 1000.0
    y = np.asarray([p['In_Y'] for p in points], float) / 1000.0
    dx = np.asarray([p['DX_Err'] for p in points], float)
    dy = np.asarray([p['DY_Err'] for p in points], float)
    d = np.hypot(dx, dy)
    tr = ctx.get('truth')
    if ctx.get('extent'):
        e0, e1, n0, n1 = [v / 1000.0 for v in ctx['extent']]
    elif len(x):
        pad = max(np.ptp(x), np.ptp(y), 1.0) * 0.05
        e0, e1, n0, n1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    else:
        e0, e1, n0, n1 = 0.0, 1.0, 0.0, 1.0
    W, H = max(e1 - e0, 1e-3), max(n1 - n0, 1e-3)
    wide = W >= H
    # the figure in inches from the scene's own shape: the map 15 in along the
    # long side, the colour scale along that side, the statistics beyond it
    L = 15.0
    if wide:
        mw, mh = L, max(2.0, L * H / W)
        left, right, top, bottom = 1.2, 1.2, 0.75, 1.75     # page room for an edge arrow
        fw, fh = left + mw + right, top + mh + bottom
        ax = fig_ax = (left / fw, bottom / fh, mw / fw, mh / fh)
        cax_r = ((left + 0.35 * mw) / fw, 0.85 / fh, 0.3 * mw / fw, 0.12 / fh)
        key_xy = ((left + 0.72 * mw) / fw, 0.9 / fh)
        stat_xy, stat_ha, stat_va = (0.5, 0.15 / fh), 'center', 'bottom'
    else:
        mh, mw = L, max(2.0, L * W / H)
        left, top, bottom = 0.9, 1.1, 0.8
        fw, fh = left + mw + 0.3 + 0.15 + 1.1 + 4.6, top + mh + bottom
        ax = fig_ax = (left / fw, bottom / fh, mw / fw, mh / fh)
        cax_r = ((left + mw + 0.3) / fw, (bottom + 0.35 * mh) / fh, 0.15 / fw, 0.3 * mh / fh)
        key_xy = ((left + mw + 0.3) / fw, (bottom + 0.08 * mh) / fh)
        stat_xy, stat_ha, stat_va = ((left + mw + 1.7) / fw, 0.5), 'left', 'center'
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes(fig_ax)
    cax = fig.add_axes(cax_r)
    bg = ctx.get('background')
    if bg is not None:
        img, ext = bg
        ax.imshow(img, extent=[v / 1000.0 for v in ext], origin='upper', zorder=0, interpolation='bilinear')
    span = max(W, H)
    dmax = max(float(d.max()) if len(d) else 0.0,
               float(np.hypot(tr['dE'], tr['dN']).max()) if tr is not None and len(tr) else 0.0, 1.0)
    scale = (dmax / 1000.0) / (0.06 * span)           # the longest arrow: 6 % of the scene's long side
    unit = error_unit(list(d) or (list(np.hypot(tr['dE'], tr['dN'])) if tr is not None else []))
    uf = 1.0 if unit == 'm' else 1000.0                # errors shown in metres or km (positions stay km)
    if len(x):
        from matplotlib.colors import LinearSegmentedColormap
        bright = LinearSegmentedColormap.from_list('rival_bright', [tuple(c / 255.0 for c in v) for v in BRIGHT])
        q = ax.quiver(x, y, -dx / 1000.0, -dy / 1000.0, d / uf, angles='xy', scale_units='xy', scale=scale,
                      units='inches', width=0.035, cmap=bright, zorder=3)
        q.set_clip_on(False)                 # an arrow at the scene's edge may run past the frame
        ax.scatter(x, y, s=14, c='white', edgecolors='0.35', linewidths=0.6, zorder=4)
        cb = fig.colorbar(q, cax=cax, orientation='horizontal' if wide else 'vertical')
        cb.ax.tick_params(labelsize=8)
        cb.ax.set_title(f'Error ({unit})', fontsize=9, loc='left' if wide else 'center')
        ref = float(10 ** np.floor(np.log10(dmax)))
        ax.quiverkey(q, key_xy[0], key_xy[1], ref / 1000.0, f'Arrow: {ref / uf:g} {unit}', labelpos='E',
                     coordinates='figure', color='k')
    else:
        cax.set_visible(False)
    if tr is not None and len(tr):
        tx, ty = tr['E'].to_numpy() / 1000.0, tr['N'].to_numpy() / 1000.0
        ax.quiver(tx, ty, -tr['dE'].to_numpy() / 1000.0, -tr['dN'].to_numpy() / 1000.0, angles='xy',
                  scale_units='xy', scale=scale, units='inches', width=0.05, color='red', zorder=5)
        ax.scatter(tx, ty, s=90, marker='*', c='red', edgecolors='white', linewidths=0.8, zorder=6,
                   label=f'manual points ({len(tr)})')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ax.set_xlim(e0, e1)
    ax.set_ylim(n0, n1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('E (km)', fontweight='bold', fontsize=16)
    ax.set_ylabel('N (km)', fontweight='bold', fontsize=16)
    if ctx.get('crs'):                     # lon / lat on the axes (along the scene's middle row / column)
        try:
            from matplotlib.ticker import FuncFormatter
            from pyproj import CRS, Transformer
            to_ll = Transformer.from_crs(CRS.from_user_input(ctx['crs']), CRS.from_epsg(4326), always_xy=True)
            xc, yc = (e0 + e1) / 2.0 * 1000.0, (n0 + n1) / 2.0 * 1000.0

            def lon_of(v, _pos=None):
                lo = to_ll.transform(v * 1000.0, yc)[0]
                return f'{abs(lo):.2f}°{"E" if lo >= 0 else "W"}'

            def lat_of(v, _pos=None):
                la = to_ll.transform(xc, v * 1000.0)[1]
                return f'{abs(la):.2f}°{"N" if la >= 0 else "S"}'
            from matplotlib.ticker import FixedLocator
            to_xy = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(ctx['crs']), always_xy=True)
            lon0, lon1 = sorted([to_ll.transform(e0 * 1000.0, yc)[0], to_ll.transform(e1 * 1000.0, yc)[0]])
            lat0, lat1 = sorted([to_ll.transform(xc, n0 * 1000.0)[1], to_ll.transform(xc, n1 * 1000.0)[1]])
            lonc, latc = to_ll.transform(xc, yc)

            def nice(lo, hi, want):
                for st in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
                    if (hi - lo) / st <= want:
                        return [round(v, 6) for v in np.arange(np.ceil(lo / st) * st, hi + 1e-9, st)]
                return [lo, hi]
            nx = 10 if W >= H else 4
            ny = 4 if W >= H else 10
            ax.xaxis.set_major_locator(FixedLocator([to_xy.transform(v, latc)[0] / 1000.0 for v in nice(lon0, lon1, nx)]))
            ax.yaxis.set_major_locator(FixedLocator([to_xy.transform(lonc, v)[1] / 1000.0 for v in nice(lat0, lat1, ny)]))
            ax.xaxis.set_major_formatter(FuncFormatter(lon_of))
            ax.yaxis.set_major_formatter(FuncFormatter(lat_of))
            ax.set_xlabel('Longitude', fontweight='bold', fontsize=16)
            ax.set_ylabel('Latitude', fontweight='bold', fontsize=16)
        except Exception:
            pass
    ax.grid(True, alpha=0.25)
    name = os.path.splitext(os.path.basename(png))[0].replace('_vectors', '')
    ax.set_title(vectors_title(ctx.get('title') or title or name, st, unit), fontsize=14, fontweight='bold', pad=4)
    if st:
        txt = vectors_footer(st, unit)
        man = ctx.get('manual')
        if man:
            txt += f"\nmanual points: {vectors_footer(man, unit)}   |   RMSE {_ne(man, 'rmse_y_m', 'rmse_x_m', unit)}"
        if not wide:
            txt = txt.replace('   |   ', '\n')
        fig.text(*stat_xy, txt, ha=stat_ha, va=stat_va, fontsize=11)
    fig.canvas.draw()                        # tick labels exist once drawn: bold them
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('bold')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def manual_stats(truth_csv: str) -> Dict:
    """RIVAL's statistics of the manual points (the ground-truth file):
    their DX_Err / DY_Err (In - Ref, m), else In - Ref of their coordinates."""
    import pandas as pd
    t = pd.read_csv(truth_csv, encoding='utf-8-sig')
    cols = {c.lower().replace(' ', '_'): c for c in t.columns}
    if 'dx_err' in cols and 'dy_err' in cols:
        dx, dy = t[cols['dx_err']].astype(float), t[cols['dy_err']].astype(float)
    else:
        dx = t[cols['in_x']].astype(float) - t[cols['ref_x']].astype(float)
        dy = t[cols['in_y']].astype(float) - t[cols['ref_y']].astype(float)
    pts = [{'DX_Err': float(a), 'DY_Err': float(b)} for a, b in zip(dx, dy)]
    return stats(pts)


COMPARE = ('mean_dx_m', 'mean_dy_m', 'rmse_x_m', 'rmse_y_m', 'rmse_m', 'ce90_m')


def report_file(csv_path: str, out_dir: str, ctx: Optional[Dict] = None) -> Dict:
    """Shapefile, look-vector plot and statistics of one RIVAL CSV."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    pts = read_points(csv_path)
    st = stats(pts)
    shp = write_shapefile(pts, os.path.join(out_dir, name))
    png = plot_vectors(pts, os.path.join(out_dir, name + '_vectors.png'), name, st, ctx)
    return {'file': os.path.basename(csv_path), **st, 'shapefile': shp, 'vectors_png': png}


def report(job_dir: str, all_files: bool = False, quiet: bool = False, truth_csv: str = '') -> List[Dict]:
    """REPORT/ of a job folder: every RIVAL_FINAL_* and RIVAL_BEST_FINAL_*
    (all_files: every RIVAL_*.csv but the _detail ones); REPORT_STATS.csv,
    with the manual points' statistics (the job's ground truth, or
    truth_csv) as a row of their own and each file's difference to them
    (<stat>_vs_manual: file minus manual; the goal is near 0)."""
    import pandas as pd
    if os.path.isfile(job_dir):
        files, out = [job_dir], os.path.join(os.path.dirname(job_dir), 'REPORT')
    else:
        allr = [p for p in sorted(glob.glob(os.path.join(job_dir, 'RIVAL_*.csv'))) if not p.endswith('_detail.csv')]
        if all_files:
            files = allr
        else:                              # every set worth looking at: FINAL (each level), BEST_FINAL, BEST,
            keep = ('RIVAL_FINAL_', 'RIVAL_BEST_', 'RIVAL_DENSE_', 'RIVAL_CONSISTENT_')   # DENSE, CONSISTENT
            files = [p for p in allr if os.path.basename(p).startswith(keep)]
        out = os.path.join(job_dir, 'REPORT')
    if not truth_csv and os.path.isdir(job_dir):
        try:
            import automatch_cross as X
            truth_csv = X.job_truth_csv(job_dir) or X.job_truth_csv(os.path.dirname(os.path.normpath(job_dir)))
        except Exception:
            truth_csv = ''
    manual = None
    if truth_csv and os.path.exists(truth_csv):
        try:
            manual = manual_stats(truth_csv)
        except Exception as e:
            print(f'[Report] manual points {os.path.basename(truth_csv)} unreadable: {type(e).__name__}: {e}')
    jd = job_dir if os.path.isdir(job_dir) else os.path.dirname(job_dir)
    if not os.path.exists(os.path.join(jd, 'job_used.json')):          # FILTERED/ or CLOUDFREE/ of a job
        jd = os.path.dirname(os.path.normpath(jd))
    ctx = plot_context(jd, truth_csv) if files else {}
    ctx['manual'] = manual
    rows = []
    for p in files:
        try:
            rows.append(report_file(p, out, ctx))
        except Exception as e:
            print(f'[Report] {os.path.basename(p)}: {type(e).__name__}: {e}')
            rows.append({'file': os.path.basename(p), 'error': f'{type(e).__name__}: {e}'})
    if manual and rows:
        for r in rows:
            for k in COMPARE:
                if r.get(k) is not None and manual.get(k) is not None:
                    r[f'{k.replace("_m", "")}_vs_manual_m'] = round(float(r[k]) - float(manual[k]), 3)
        rows.append({'file': f'MANUAL ({os.path.basename(truth_csv)})', **manual})
    if rows:
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, 'REPORT_STATS.csv'), index=False)
        if not quiet:
            for r in rows:
                if 'n' in r and not str(r['file']).startswith('MANUAL'):
                    print(f"[Report] {r['file']}: {r['n']} points, mean dE {r.get('mean_dx_m')} m dN "
                          f"{r.get('mean_dy_m')} m, RMSE E {r['rmse_x_m']} N {r['rmse_y_m']} m, CE90 {r['ce90_m']} m"
                          f" -> {os.path.basename(r['shapefile'])}, {os.path.basename(r['vectors_png'])}")
            if manual:
                print(f"[Report] manual points: {manual['n']}, mean dE {manual.get('mean_dx_m')} m dN "
                      f"{manual.get('mean_dy_m')} m, RMSE E {manual['rmse_x_m']} N {manual['rmse_y_m']} m, CE90 "
                      f"{manual['ce90_m']} m (each file's difference to these: *_vs_manual_m)")
            print(f'[Report] {len(rows)} row(s): shapefiles, look-vector plots and REPORT_STATS.csv in {out}')
    elif not quiet:
        print(f'[Report] no RIVAL CSV in {job_dir}')
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Shapefile, look-vector plot and RIVAL statistics of the final points')
    ap.add_argument('job_dir', help='a job output folder, or one RIVAL CSV')
    ap.add_argument('--all', action='store_true', help='every RIVAL_*.csv of the folder, not only the FINAL ones')
    a = ap.parse_args(argv)
    return 0 if report(a.job_dir, a.all) else 1


if __name__ == '__main__':
    sys.exit(main())
