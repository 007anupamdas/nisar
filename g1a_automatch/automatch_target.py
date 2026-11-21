"""
Targeting accuracy: how far the acquired scene lies from the intended one.
A port of DPQED_targeting.py (the g1a repository) as a module, without GDAL
or geopandas, driven by a product folder or a request ID.

For a product (its BAND_META.txt: scene corners, strip start / centre / end
times, orbit, strip, scene, sensor, imaging mode) and the day's range file
(the intended ground track: one line per time with the centre, start-of-
swath, end-of-swath and target lat / lon), the intended position is
interpolated at the scene's start, centre and end times and compared with
the acquired centre line (the mid-points of the top and bottom edges and
the scene centre), in metres of the local UTM (polar stereographic beyond
60 deg); the difference is turned along / across track by the scene's
heading. Outputs, in the output folder:

  TARGET_<suffix>.csv     the row DPQED_targeting wrote (DOP, orbit, strip,
                          scene, mode, sensor, orbit / attitude source,
                          along / across, northing / easting, roll / pitch /
                          yaw, times, the acquired and intended positions,
                          along / across at start / centre / end)
  TARGET_<suffix>.png     the scene outline, the acquired and intended lines
  TARGET_<suffix>_prod_line.shp / _prod_point.shp / _range_line_{s,t,e}.shp /
                          _range_point.shp   as before (WGS 84)

    python automatch_job.py target <product folder | BAND_META.txt | request ID>
        [--root X:/DPS/G1A] [--rv T:/PRODG1A/RV] [--out D:/g1a/tgt] [--all-scenes]

A request ID is looked up as <root>/<ID>/BAND_META.txt; --all-scenes also
takes <ID>21, <ID>31, ... (the further scenes of a request, as before).
The range file is the one in --rv whose name holds the date (ddMONyyyy),
the sensor's long name (MV -> MX-VNIR) and '101600'.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import struct
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SENSOR_NAMES = {'MV': 'MX-VNIR', 'HV': 'HyS-VNIR', 'HS': 'HyS-SWIR', 'ML': 'MX-LWIR', 'MW': 'MW', 'LW': 'LW'}
ATTITUDE = {'7': 'Gyro', '1': 'StarSensor-1', '2': 'StarSensor-2'}
RANGE_TAG = '101600'


# ── metadata ────────────────────────────────────────────────────────────────
def read_band_meta(path: str) -> Dict[str, str]:
    """key=value lines of BAND_META.txt (keys as written; values stripped)."""
    out: Dict[str, str] = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if '=' in line:
                k, _, v = line.partition('=')
                out[k.strip()] = v.strip()
    return out


def _get(meta: Dict[str, str], *names, default=None):
    for n in names:
        if n in meta and meta[n] != '':
            return meta[n]
    return default


def _ms(text: str) -> float:
    """'06:27:39.123456' or '06:27:39:123' (after a 'T' when a date leads) -> milliseconds of the day."""
    t = text.split('T')[-1].split(' ')[-1]
    parts = re.split(r'[:.]', t)
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    frac = parts[3] if len(parts) > 3 else '0'
    return 3600000.0 * h + 60000.0 * m + 1000.0 * s + float('0.' + frac) * 1000.0


def scene_of(meta: Dict[str, str]) -> Dict:
    """What the targeting needs from a product's metadata."""
    need = ['ImageULLat', 'ImageULLon', 'ImageURLat', 'ImageURLon', 'ImageLLLat', 'ImageLLLon', 'ImageLRLat',
            'ImageLRLon']
    missing = [k for k in need if _get(meta, k) is None]
    if missing:
        raise ValueError(f'BAND_META without {", ".join(missing)}')
    c = {k: float(meta[k]) for k in need}
    s = {'corners': c, 'sensor': _get(meta, 'Sensor', 'SensorName', default=''),
         'satellite': _get(meta, 'SatID', 'SatelliteID', 'Satellite', 'Mission', default=''),
         'mode': _get(meta, 'ImagingMode', default='DSC'), 'code': _get(meta, 'ProductCode', default=''),
         'orbit': _get(meta, 'ImagingOrbitNo', default=''), 'strip': _get(meta, 'SegmentNumber', 'StripNumber', default=''),
         'scene': _get(meta, 'SceneNumber', default=''),
         'orbit_source': _get(meta, 'SourceOfOrbit', default=''),
         'attitude_source': _get(meta, 'SourceOfAttitude', default=''),
         'roll': float(_get(meta, 'SceneCenterRoll', 'SceneCentreRoll', default='nan')),
         'pitch': float(_get(meta, 'SceneCenterPitch', 'SceneCentrePitch', default='nan')),
         'yaw': float(_get(meta, 'SceneCenterYaw', 'SceneCentreYaw', default='nan')),
         'date_of_pass': _get(meta, 'DateOfPass', default=''),
         'centre_lon': float(_get(meta, 'StripCenterLon', 'StripCentreLon', 'SceneCenterLon', default='nan')),
         'centre_lat': float(_get(meta, 'StripCenterLat', 'StripCentreLat', 'SceneCenterLat', default='nan'))}
    for key, names in (('t_start', ('StripStartTime', 'SceneStartTime')),
                       ('t_centre', ('SceneCenterTime', 'SceneCentreTime', 'StripCenterTime')),
                       ('t_end', ('StripEndTime', 'SceneEndTime'))):
        v = _get(meta, *names)
        if v is None:
            raise ValueError(f'BAND_META without {names[0]}')
        s[key] = _ms(v)
        s[key + '_text'] = v.split('T')[-1].split(' ')[-1]
    # the acquired centre line: mid-points of the top and bottom edges, the centre
    s['uc'] = ((c['ImageULLon'] + c['ImageLLLon']) / 2.0, (c['ImageULLat'] + c['ImageLLLat']) / 2.0)
    s['lc'] = ((c['ImageURLon'] + c['ImageLRLon']) / 2.0, (c['ImageURLat'] + c['ImageLRLat']) / 2.0)
    s['sc'] = ((s['uc'][0] + s['lc'][0]) / 2.0, (s['uc'][1] + s['lc'][1]) / 2.0)
    s['um'] = ((c['ImageULLon'] + c['ImageURLon']) / 2.0, (c['ImageULLat'] + c['ImageURLat']) / 2.0)
    s['lm'] = ((c['ImageLLLon'] + c['ImageLRLon']) / 2.0, (c['ImageLLLat'] + c['ImageLRLat']) / 2.0)
    return s


def date_tag(date_of_pass: str) -> str:
    """'12-NOV-2026' (or '12-Nov-2026') -> '12NOV2026', as the range files are named."""
    d = datetime.strptime(date_of_pass.strip(), '%d-%b-%Y')
    return d.strftime('%d%b%Y').upper()


def find_range_file(rv_dir: str, date_of_pass: str, sensor: str) -> str:
    """The range file of the day and sensor in rv_dir ('' when none)."""
    tag, long = date_tag(date_of_pass), SENSOR_NAMES.get(sensor, sensor)
    cands = [p for p in sorted(glob.glob(os.path.join(rv_dir, '*')))
             if tag in os.path.basename(p) and long in os.path.basename(p) and RANGE_TAG in os.path.basename(p)]
    return cands[-1] if cands else ''


def read_range_file(path: str) -> List[Dict]:
    """The range file's lines: year month day hour min sec msec lat_c lon_c
    lat_s lon_s lat_e lon_e lat_t lon_t ... -> dicts with 't' (ms of day)."""
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            p = line.split()
            if len(p) <= 10:
                continue
            try:
                t = 3600000.0 * int(p[3]) + 60000.0 * int(p[4]) + 1000.0 * int(p[5]) + float(p[6])
                rows.append({'t': t, 'lat_c': float(p[7]), 'lon_c': float(p[8]), 'lat_s': float(p[9]),
                             'lon_s': float(p[10]), 'lat_e': float(p[11]), 'lon_e': float(p[12]),
                             'lat_t': float(p[13]), 'lon_t': float(p[14])})
            except (ValueError, IndexError):
                continue
    return rows


# ── geometry ────────────────────────────────────────────────────────────────
def local_crs(lon: float, lat: float) -> str:
    """UTM of the point, polar stereographic beyond 60 deg (as DPQED_targeting)."""
    if lat > 60:
        return 'EPSG:3413'
    if lat < -60:
        return 'EPSG:3031'
    zone = int(math.floor(((180.0 + lon) / 6.0) % 60) + 1)
    return f'+proj=utm +zone={zone} +{"south" if lat < 0 else "north"} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'


def _to_map(crs: str):
    from pyproj import CRS, Transformer
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True)
    return lambda lon, lat: tr.transform(lon, lat)


def _interp(rows: List[Dict], key: str, t: float) -> float:
    ts = [r['t'] for r in rows]
    if t <= ts[0]:
        return rows[0][key]
    if t >= ts[-1]:
        return rows[-1][key]
    for i in range(1, len(ts)):
        if ts[i] >= t:
            w = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9)
            return rows[i - 1][key] + w * (rows[i][key] - rows[i - 1][key])
    return rows[-1][key]


def target_accuracy(scene: Dict, rows: List[Dict]) -> Dict:
    """Along / across-track targeting error (m) of a scene against the
    intended track: at the scene start, centre and end (where the range
    file covers them) and their mean; positive along = acquired ahead."""
    if not rows:
        raise ValueError('empty range file')
    t0, t1 = rows[0]['t'], rows[-1]['t']
    have_start, have_end = t0 < scene['t_start'], t1 > scene['t_end']
    if not (t0 < scene['t_centre'] < t1):
        raise ValueError('the range file does not cover the scene centre time')
    to_map = _to_map(local_crs(*scene['sc']))
    intended = {}
    for name, t in (('start', scene['t_start']), ('centre', scene['t_centre']), ('end', scene['t_end'])):
        intended[name] = {k: _interp(rows, k, t) for k in ('lat_c', 'lon_c', 'lat_s', 'lon_s', 'lat_e', 'lon_e',
                                                             'lat_t', 'lon_t')}
    imap = {n: to_map(intended[n]['lon_c'], intended[n]['lat_c']) for n in ('start', 'centre', 'end')}
    # The acquired centre line runs between the mid-points of the two edges
    # the track crosses. DPQED_targeting took the left / right edges (UL+LL,
    # UR+LR) and turned them by the imaging mode; the products' corner naming
    # is not the same everywhere, so here the pair (left / right or top /
    # bottom) that runs along the intended track is taken, its start the end
    # nearer the intended start.
    c = scene['corners']
    UL, UR, LL, LR = (to_map(c['ImageULLon'], c['ImageULLat']), to_map(c['ImageURLon'], c['ImageURLat']),
                      to_map(c['ImageLLLon'], c['ImageLLLat']), to_map(c['ImageLRLon'], c['ImageLRLat']))
    mid = lambda p, q: ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)   # edge mid-points in metres, not in lon / lat
    sc = mid(mid(UL, LR), mid(UR, LL))
    track = (imap['end'][0] - imap['start'][0], imap['end'][1] - imap['start'][1])
    best, best_cos = None, -1.0
    for a, b in ((mid(UL, LL), mid(UR, LR)), (mid(UL, UR), mid(LL, LR))):
        v = (b[0] - a[0], b[1] - a[1])
        cos = abs(v[0] * track[0] + v[1] * track[1]) / max(math.hypot(*v) * math.hypot(*track), 1e-9)
        if cos > best_cos:
            best, best_cos = (a, b), cos
    a, b = best
    if math.hypot(a[0] - imap['start'][0], a[1] - imap['start'][1]) > math.hypot(b[0] - imap['start'][0],
                                                                                 b[1] - imap['start'][1]):
        a, b = b, a
    acquired = {'start': a, 'centre': sc, 'end': b}
    errs = {}
    for name in ('start', 'centre', 'end'):
        ix, iy = imap[name]
        ax, ay = acquired[name]
        errs[name] = (ax - ix, ay - iy)                       # easting, northing error (acquired - intended)
    use = ['centre'] + (['start'] if have_start else []) + (['end'] if have_end else [])
    ex = sum(errs[n][0] for n in use) / len(use)
    ey = sum(errs[n][1] for n in use) / len(use)
    dx, dy = b[0] - a[0], b[1] - a[1]                        # the flight direction: start -> end
    theta = math.atan2(dy, dx)

    def along_across(e):
        return (e[0] * math.cos(theta) + e[1] * math.sin(theta), e[1] * math.cos(theta) - e[0] * math.sin(theta))
    al, ac = along_across((ex, ey))
    from pyproj import CRS, Transformer
    back = Transformer.from_crs(CRS.from_user_input(local_crs(*scene['sc'])), CRS.from_epsg(4326), always_xy=True)
    out = {'along_m': round(al, 3), 'across_m': round(ac, 3), 'easting_err_m': round(ex, 3),
           'northing_err_m': round(ey, 3), 'heading_deg': round(math.degrees(theta), 3),
           'range_covers_start': have_start, 'range_covers_end': have_end, 'used': use,
           'intended': intended, 'acquired_lonlat': {n: back.transform(*acquired[n]) for n in acquired}}
    for n in ('start', 'centre', 'end'):
        a, c = along_across(errs[n])
        out[f'along_{n}_m'], out[f'across_{n}_m'] = round(a, 3), round(c, 3)
    return out


# ── outputs ─────────────────────────────────────────────────────────────────
def _ne(v: float, pos: str, neg: str) -> str:
    r = round(abs(v), 1) if abs(v) <= 1000 else round(abs(v))
    return f'{r} {pos if v >= 0 else neg}'


def write_outputs(scene: Dict, res: Dict, out_dir: str, prod_id: str) -> Dict[str, str]:
    """CSV row, PNG and shapefiles of one scene's targeting result."""
    import automatch_report as RP
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"{scene['date_of_pass']}_{scene['sensor']}_{scene['orbit']}_{scene['strip']}_{scene['scene']}"
    paths = {}
    row = {'DOP': scene['date_of_pass'], 'Orbit': scene['orbit'], 'Strip-ID': scene['strip'], 'Scene': scene['scene'],
           'Mode': scene['mode'], 'Sensor': scene['sensor'], 'OrbitSource': scene['orbit_source'],
           'AttitudeSource': scene['attitude_source'], 'AlongTrack': res['along_m'], 'AcrossTrack': res['across_m'],
           'Northing': _ne(res['northing_err_m'], 'N', 'S'), 'Easting': _ne(res['easting_err_m'], 'E', 'W'),
           'Roll': scene['roll'], 'Pitch': scene['pitch'], 'Yaw': scene['yaw'], 'SceneStart': scene['t_start_text'],
           'SceneCenter': scene['t_centre_text'], 'SceneEnd': scene['t_end_text'],
           'UCLat': res['acquired_lonlat']['start'][1], 'UCLon': res['acquired_lonlat']['start'][0],
           'SCLat': scene['sc'][1], 'SCLon': scene['sc'][0],
           'LCLat': res['acquired_lonlat']['end'][1], 'LCLon': res['acquired_lonlat']['end'][0],
           'RangeLat-SceneStart': res['intended']['start']['lat_c'], 'RangeLon-SceneStart': res['intended']['start']['lon_c'],
           'RangeLat-SceneCentre': res['intended']['centre']['lat_c'], 'RangeLon-SceneCentre': res['intended']['centre']['lon_c'],
           'RangeLat-SceneEnd': res['intended']['end']['lat_c'], 'RangeLon-SceneEnd': res['intended']['end']['lon_c'],
           'AlongStart': res['along_start_m'], 'AcrossStart': res['across_start_m'], 'AlongCentre': res['along_centre_m'],
           'AcrossCentre': res['across_centre_m'], 'AlongEnd': res['along_end_m'], 'AcrossEnd': res['across_end_m'],
           'ID': prod_id, 'Heading_deg': res['heading_deg'], 'AttitudeSourceName': ATTITUDE.get(scene['attitude_source'], 'Other')}
    paths['csv'] = os.path.join(out_dir, f'TARGET_{suffix}.csv')
    with open(paths['csv'], 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    # the plot: outline, acquired and intended centre lines
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = scene['corners']
    xs = [c['ImageULLon'], c['ImageLLLon'], c['ImageLRLon'], c['ImageURLon'], c['ImageULLon']]
    ys = [c['ImageULLat'], c['ImageLLLat'], c['ImageLRLat'], c['ImageURLat'], c['ImageULLat']]
    i = res['intended']
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xs, ys, 'k-', lw=1, label='scene')
    acq = res['acquired_lonlat']
    ax.plot([acq['start'][0], acq['centre'][0], acq['end'][0]], [acq['start'][1], acq['centre'][1], acq['end'][1]],
            'r-o', ms=4, label='acquired')
    ax.plot([i['start']['lon_c'], i['centre']['lon_c'], i['end']['lon_c']],
            [i['start']['lat_c'], i['centre']['lat_c'], i['end']['lat_c']], '-o', color='olive', ms=4, label='intended')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f"{scene['satellite'] or 'G1A'} {scene['date_of_pass']} {scene['sensor']} {scene['orbit']}/"
                 f"{scene['strip']}/{scene['scene']}\nTarget accuracy (along / across): {round(res['along_m'])} / "
                 f"{round(res['across_m'])} m")
    fig.tight_layout()
    paths['png'] = os.path.join(out_dir, f'TARGET_{suffix}.png')
    fig.savefig(paths['png'], dpi=120)
    plt.close(fig)
    # shapefiles (WGS 84): the acquired line and points, the intended lines and points
    base = os.path.join(out_dir, f'TARGET_{suffix}')
    acq_pts = [acq['start'], acq['centre'], acq['end']]
    paths['prod_line'] = write_polyline([acq_pts], base + '_prod_line')
    paths['prod_point'] = write_points(acq_pts, base + '_prod_point')
    for key in ('s', 't', 'e'):
        line = [(i[n][f'lon_{key}'], i[n][f'lat_{key}']) for n in ('start', 'end')]
        paths[f'range_line_{key}'] = write_polyline([line], base + f'_range_line_{key}')
    paths['range_point'] = write_points([(i['start']['lon_c'], i['start']['lat_c']),
                                         (i['end']['lon_c'], i['end']['lat_c'])], base + '_range_point')
    return paths


def _shp_header(f, shape_type: int, box, length_words: int):
    f.write(struct.pack('>iiiiiii', 9994, 0, 0, 0, 0, 0, length_words))
    f.write(struct.pack('<ii', 1000, shape_type))
    f.write(struct.pack('<dddddddd', *box, 0.0, 0.0, 0.0, 0.0))


def _dbf_ids(stem: str, n: int) -> None:
    with open(stem + '.dbf', 'wb') as dbf:
        dbf.write(struct.pack('<BBBBIHH20x', 3, 26, 1, 1, n, 32 + 32 + 1, 1 + 9))
        dbf.write(struct.pack('<11sc4xBB14x', b'ID', b'N', 9, 0))
        dbf.write(b'\r')
        for i in range(1, n + 1):
            dbf.write(b' ' + str(i).rjust(9).encode('ascii'))
        dbf.write(b'\x1a')


def write_points(points: List[Tuple[float, float]], stem: str) -> str:
    """A point shapefile (WGS 84) with an ID field."""
    import automatch_report as RP
    xs, ys = ([p[0] for p in points] or [0.0]), ([p[1] for p in points] or [0.0])
    box = (min(xs), min(ys), max(xs), max(ys))
    rec_len = 10
    with open(stem + '.shp', 'wb') as shp, open(stem + '.shx', 'wb') as shx:
        _shp_header(shp, 1, box, (100 + len(points) * (8 + rec_len * 2)) // 2)
        _shp_header(shx, 1, box, (100 + 8 * len(points)) // 2)
        offset = 50
        for i, (x, y) in enumerate(points, 1):
            shp.write(struct.pack('>ii', i, rec_len))
            shp.write(struct.pack('<idd', 1, x, y))
            shx.write(struct.pack('>ii', offset, rec_len))
            offset += 4 + rec_len
    _dbf_ids(stem, len(points))
    with open(stem + '.prj', 'w', encoding='ascii') as f:
        f.write(RP.WGS84_PRJ)
    return stem + '.shp'


def write_polyline(lines: List[List[Tuple[float, float]]], stem: str) -> str:
    """A polyline shapefile (WGS 84), one record per line, with an ID field."""
    import automatch_report as RP
    allp = [p for line in lines for p in line] or [(0.0, 0.0)]
    box = (min(p[0] for p in allp), min(p[1] for p in allp), max(p[0] for p in allp), max(p[1] for p in allp))
    recs = []
    for line in lines:
        n = len(line)
        body = struct.pack('<i', 3) + struct.pack('<dddd', min(p[0] for p in line), min(p[1] for p in line),
                                                  max(p[0] for p in line), max(p[1] for p in line))
        body += struct.pack('<ii', 1, n) + struct.pack('<i', 0)
        for x, y in line:
            body += struct.pack('<dd', x, y)
        recs.append(body)
    total = 100 + sum(8 + len(b) for b in recs)
    with open(stem + '.shp', 'wb') as shp, open(stem + '.shx', 'wb') as shx:
        _shp_header(shp, 3, box, total // 2)
        _shp_header(shx, 3, box, (100 + 8 * len(recs)) // 2)
        offset = 50
        for i, b in enumerate(recs, 1):
            shp.write(struct.pack('>ii', i, len(b) // 2))
            shp.write(b)
            shx.write(struct.pack('>ii', offset, len(b) // 2))
            offset += 4 + len(b) // 2
    _dbf_ids(stem, len(recs))
    with open(stem + '.prj', 'w', encoding='ascii') as f:
        f.write(RP.WGS84_PRJ)
    return stem + '.shp'


# ── driving it ──────────────────────────────────────────────────────────────
def meta_paths(what: str, root: str = '', all_scenes: bool = False) -> List[Tuple[str, str]]:
    """[(product id, BAND_META.txt)] for a BAND_META path, a product folder,
    or a request ID under root (with all_scenes: ID, ID21, ID31, ... while
    they exist, as DPQED_targeting walked them)."""
    if os.path.isfile(what):
        return [(os.path.basename(os.path.dirname(os.path.abspath(what))), what)]
    if os.path.isdir(what):
        p = os.path.join(what, 'BAND_META.txt')
        return [(os.path.basename(os.path.normpath(what)), p)] if os.path.exists(p) else []
    out = []
    ids = [what] + ([f'{what}{z}1' for z in range(2, 10)] if all_scenes else [])
    for pid in ids:
        p = os.path.join(root, pid, 'BAND_META.txt')
        if os.path.exists(p):
            out.append((pid, p))
        elif pid != what:
            break
    return out


def run(what: str, root: str = '', rv_dir: str = '', out_dir: str = '', all_scenes: bool = False,
        range_file: str = '', quiet: bool = False) -> List[Dict]:
    """Targeting accuracy of each scene of a product / request: the rows written
    (TARGET_ALL.csv in out_dir gathers them)."""
    found = meta_paths(what, root, all_scenes)
    if not found:
        raise FileNotFoundError(f'no BAND_META.txt for {what!r}' + (f' under {root}' if root else ''))
    out_dir = out_dir or os.path.dirname(found[0][1])
    rows = []
    for pid, mp in found:
        scene = scene_of(read_band_meta(mp))
        rf = range_file or find_range_file(rv_dir, scene['date_of_pass'], scene['sensor'])
        if not rf:
            print(f"[Target] {pid}: no range file for {scene['date_of_pass']} {SENSOR_NAMES.get(scene['sensor'], scene['sensor'])} "
                  f"in {rv_dir or '(no RV folder given)'}")
            rows.append({'ID': pid, 'error': 'no range file'})
            continue
        try:
            res = target_accuracy(scene, read_range_file(rf))
        except ValueError as e:
            print(f'[Target] {pid}: {e}')
            rows.append({'ID': pid, 'error': str(e)})
            continue
        paths = write_outputs(scene, res, out_dir, pid)
        if not quiet:
            print(f"[Target] {pid}: {scene['date_of_pass']} {scene['sensor']} orbit {scene['orbit']} strip "
                  f"{scene['strip']} scene {scene['scene']} ({scene['mode']}): along {res['along_m']:.0f} m, across "
                  f"{res['across_m']:.0f} m (start {res['along_start_m']:.0f}/{res['across_start_m']:.0f}, centre "
                  f"{res['along_centre_m']:.0f}/{res['across_centre_m']:.0f}, end {res['along_end_m']:.0f}/"
                  f"{res['across_end_m']:.0f}; mean of {', '.join(res['used'])}) -> {paths['csv']}")
        with open(paths['csv'], encoding='utf-8') as f:
            rows += list(csv.DictReader(f))
        rows[-1]['range_file'] = rf
    if rows:
        import pandas as pd
        allp = os.path.join(out_dir, 'TARGET_ALL.csv')
        old = pd.read_csv(allp) if os.path.exists(allp) else pd.DataFrame()
        new = pd.DataFrame(rows)
        if not old.empty and 'ID' in old.columns:
            old = old[~old['ID'].astype(str).isin(new['ID'].astype(str))]
        pd.concat([old, new], ignore_index=True).to_csv(allp, index=False)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Targeting accuracy of a product against its intended track')
    ap.add_argument('what', help='a product folder, its BAND_META.txt, or a request ID (with --root)')
    ap.add_argument('--root', default='X:/DPS/G1A', help='where <request ID>/BAND_META.txt lives')
    ap.add_argument('--rv', default='T:/PRODG1A/RV', help='the range files (intended tracks)')
    ap.add_argument('--range-file', default='', help='this range file instead of searching --rv')
    ap.add_argument('--out', default='', help='output folder (default: beside BAND_META.txt)')
    ap.add_argument('--all-scenes', action='store_true', help='a request ID: also its further scenes (ID21, ID31 ...)')
    a = ap.parse_args(argv)
    rows = run(a.what, a.root, a.rv, a.out, a.all_scenes, a.range_file)
    return 0 if rows and all('error' not in r for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
