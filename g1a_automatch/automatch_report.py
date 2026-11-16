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


def plot_vectors(points: List[Dict], png: str, title: str = '', st: Optional[Dict] = None) -> str:
    """Look vectors: an arrow per point from its input position towards its
    reference position (the geolocation error), in the working map
    coordinates (km), coloured by length; the statistics in the corner."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.asarray([p['In_X'] for p in points]) / 1000.0
    y = np.asarray([p['In_Y'] for p in points]) / 1000.0
    dx = np.asarray([p['DX_Err'] for p in points])
    dy = np.asarray([p['DY_Err'] for p in points])
    d = np.hypot(dx, dy)
    fig, ax = plt.subplots(figsize=(9, 8))
    if len(x):
        span = max(np.ptp(x), np.ptp(y), 1.0)
        scale = (d.max() / 1000.0 or 1.0) / (0.08 * span)      # the longest arrow: 8 % of the extent
        q = ax.quiver(x, y, -dx / 1000.0, -dy / 1000.0, d, angles='xy', scale_units='xy', scale=scale,
                      width=0.003, cmap='viridis')
        fig.colorbar(q, ax=ax, label='error (m)')
        ax.scatter(x, y, s=6, c='k', zorder=3)
    ax.set_aspect('equal')
    ax.set_xlabel('E (km)')
    ax.set_ylabel('N (km)')
    ax.set_title(title or os.path.basename(png))
    if st:
        txt = (f"n {st['n']}   mean dE {st.get('mean_dx_m', 0):.1f} m, dN {st.get('mean_dy_m', 0):.1f} m\n"
               f"RMSE E {st['rmse_x_m']:.1f} m, N {st['rmse_y_m']:.1f} m   CE90 {st['ce90_m']:.1f} m"
               + ('' if st.get('ce90_circular', True) else ' (axes unequal)') + f"\nlargest {st.get('max_dist_m', 0):.1f} m")
        ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def report_file(csv_path: str, out_dir: str) -> Dict:
    """Shapefile, look-vector plot and statistics of one RIVAL CSV."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    pts = read_points(csv_path)
    st = stats(pts)
    shp = write_shapefile(pts, os.path.join(out_dir, name))
    png = plot_vectors(pts, os.path.join(out_dir, name + '_vectors.png'), name, st)
    return {'file': os.path.basename(csv_path), **st, 'shapefile': shp, 'vectors_png': png}


def report(job_dir: str, all_files: bool = False, quiet: bool = False) -> List[Dict]:
    """REPORT/ of a job folder: every RIVAL_FINAL_* and RIVAL_BEST_FINAL_*
    (all_files: every RIVAL_*.csv but the _detail ones); REPORT_STATS.csv."""
    import pandas as pd
    if os.path.isfile(job_dir):
        files, out = [job_dir], os.path.join(os.path.dirname(job_dir), 'REPORT')
    else:
        pat = 'RIVAL_*.csv' if all_files else 'RIVAL_*FINAL_*.csv'
        files = [p for p in sorted(glob.glob(os.path.join(job_dir, pat))) if not p.endswith('_detail.csv')]
        out = os.path.join(job_dir, 'REPORT')
    rows = []
    for p in files:
        try:
            rows.append(report_file(p, out))
        except Exception as e:
            print(f'[Report] {os.path.basename(p)}: {type(e).__name__}: {e}')
            rows.append({'file': os.path.basename(p), 'error': f'{type(e).__name__}: {e}'})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, 'REPORT_STATS.csv'), index=False)
        if not quiet:
            for r in rows:
                if 'n' in r:
                    print(f"[Report] {r['file']}: {r['n']} points, mean dE {r.get('mean_dx_m')} m dN "
                          f"{r.get('mean_dy_m')} m, RMSE E {r['rmse_x_m']} N {r['rmse_y_m']} m, CE90 {r['ce90_m']} m"
                          f" -> {os.path.basename(r['shapefile'])}, {os.path.basename(r['vectors_png'])}")
            print(f'[Report] {len(rows)} file(s): shapefiles, look-vector plots and REPORT_STATS.csv in {out}')
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
