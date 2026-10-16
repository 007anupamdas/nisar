#!/usr/bin/env python3
"""
Synthetic test data with a KNOWN geolocation error, for tests_automatch.py.

One textured "world" is sampled three ways:
  input      3-band UTM GeoTIFF whose georeferencing is wrong by (dE, dN):
             a feature at true ground P is written at map position P + (dE, dN)
  C1         degree tiles in lon/lat (N16E78.tif ...), correct georeferencing
  L8_ref     UTM scenes + Meta/index.shp naming them, correct georeferencing

So automatch should report In - Ref = (dE, dN) everywhere.

    python synthetic_data.py <out_dir> [dE] [dN]
"""

import os
import struct
import sys

import cv2
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer

UTM = 'EPSG:32644'          # zone 44N covers 78-84 E
TILE = (78.0, 16.0)         # C1 tile N16E78: 78-79 E, 16-17 N
WORLD_RES = 20.0


def _world(seed=7):
    """Multi-scale smoothed noise + sharp blobs over the whole C1 tile, on a
    20 m UTM grid. Returns (array, affine)."""
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    xs, ys = to_utm.transform([TILE[0], TILE[0] + 1, TILE[0], TILE[0] + 1],
                              [TILE[1], TILE[1], TILE[1] + 1, TILE[1] + 1])
    x0, x1 = min(xs) - 5000, max(xs) + 5000
    y0, y1 = min(ys) - 5000, max(ys) + 5000
    W = int((x1 - x0) / WORLD_RES)
    H = int((y1 - y0) / WORLD_RES)
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), np.float32)
    for scale, amp in ((64, 1.0), (16, 0.6), (4, 0.35)):
        small = rng.random((H // scale + 2, W // scale + 2)).astype(np.float32)
        img += amp * cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    n_blobs = (H * W) // 4000
    # sub-pixel centres (cv2 'shift' = 4 fractional bits): features must not
    # sit on the world grid, or grid phase leaks into the measured offset
    cx = (rng.random(n_blobs) * W * 16).astype(np.int64)
    cy = (rng.random(n_blobs) * H * 16).astype(np.int64)
    r = rng.integers(2, 9, n_blobs)
    v = rng.random(n_blobs).astype(np.float32) * 2.0
    for x, y, rr, vv in zip(cx, cy, r, v):
        cv2.circle(img, (int(x), int(y)), int(rr) * 16, float(vv), -1,
                   lineType=cv2.LINE_AA, shift=4)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img -= img.min()
    img = 50.0 + 1000.0 * img / img.max()
    return img, Affine(WORLD_RES, 0, x0, 0, -WORLD_RES, y1)


def _sample(world, wtf, E, N):
    """world value at UTM coords (E, N) arrays, bilinear."""
    inv = ~wtf
    col = (E - wtf.c) / wtf.a - 0.5
    row = (N - wtf.f) / wtf.e - 0.5
    return cv2.remap(world, col.astype(np.float32), row.astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0)


def make_input(out_dir, world, wtf, dE, dN, res=20.0, size_km=30.0):
    """3-band UTM GeoTIFF over ~16.4 N 78.5 E, georeferencing off by (dE, dN)."""
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    cE, cN = to_utm.transform(78.5, 16.45)
    n = int(size_km * 1000 / res)
    x0, y1 = cE - n * res / 2, cN + n * res / 2
    tf = Affine(res, 0, x0, 0, -res, y1)
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    E = x0 + cols * res
    N = y1 - rows * res
    # written at P + d  <=>  pixel at map Q shows true ground Q - d
    base = _sample(world, wtf, E - dE, N - dN)
    bands = [base, 0.8 * base + 30, np.sqrt(base) * 25]
    # a fill wedge, like a swath edge
    wedge = cols > (n - rows * 0.3)
    path = os.path.join(out_dir, 'G1A_SYNTH_L1.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=n, width=n, count=3,
                       dtype='uint16', crs=UTM, transform=tf, nodata=0) as dst:
        for i, b in enumerate(bands, start=1):
            b = b.copy()
            b[wedge] = 0
            dst.write(np.clip(b, 1, 65535).astype('uint16'), i)
    return path


def make_c1(out_dir, world, wtf, res_deg=0.0004):
    """C1-style degree tile N16E78.tif in lon/lat, plus a neighbour."""
    folder = os.path.join(out_dir, 'C1')
    os.makedirs(folder, exist_ok=True)
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    paths = []
    for lon0, lat0 in (TILE, (TILE[0], TILE[1] + 1)):
        n = int(round(1.0 / res_deg))
        lon = lon0 + (np.arange(n) + 0.5) * res_deg
        lat = lat0 + 1 - (np.arange(n) + 0.5) * res_deg
        LON, LAT = np.meshgrid(lon, lat)
        E, N = to_utm.transform(LON, LAT)
        img = _sample(world, wtf, np.asarray(E), np.asarray(N))
        tf = Affine(res_deg, 0, lon0, 0, -res_deg, lat0 + 1)
        name = f'N{int(lat0)}E{int(lon0)}.tif'
        path = os.path.join(folder, name)
        with rasterio.open(path, 'w', driver='GTiff', height=n, width=n, count=1,
                           dtype='float32', crs='EPSG:4326', transform=tf, nodata=0) as dst:
            dst.write(img.astype('float32'), 1)
        paths.append(path)
    return folder


def _write_index_shp(shp_path, records):
    """Minimal Polygon shapefile writer: records = [(name, ring[(x, y)...])]."""
    def poly_content(ring):
        ring = list(ring) + [ring[0]]
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        c = struct.pack('<i4d', 5, min(xs), min(ys), max(xs), max(ys))
        c += struct.pack('<ii', 1, len(ring)) + struct.pack('<i', 0)
        for x, y in ring:
            c += struct.pack('<2d', x, y)
        return c

    contents = [poly_content(r) for _, r in records]
    allx = [p[0] for _, r in records for p in r]
    ally = [p[1] for _, r in records for p in r]
    bbox = (min(allx), min(ally), max(allx), max(ally))

    def header(length_words):
        return (struct.pack('>i', 9994) + b'\0' * 20 + struct.pack('>i', length_words)
                + struct.pack('<ii', 1000, 5) + struct.pack('<4d', *bbox) + b'\0' * 32)

    shp_len = 50 + sum(4 + len(c) // 2 for c in contents)
    with open(shp_path, 'wb') as f:
        f.write(header(shp_len))
        for i, c in enumerate(contents, start=1):
            f.write(struct.pack('>ii', i, len(c) // 2) + c)
    shx = os.path.splitext(shp_path)[0] + '.shx'
    with open(shx, 'wb') as f:
        f.write(header(50 + 4 * len(contents)))
        off = 50
        for c in contents:
            f.write(struct.pack('>ii', off, len(c) // 2))
            off += 4 + len(c) // 2
    # DBF with one C(80) field "FILENAME"
    dbf = os.path.splitext(shp_path)[0] + '.dbf'
    flen = 80
    with open(dbf, 'wb') as f:
        n = len(records)
        hdr_len = 32 + 32 + 1
        rec_len = 1 + flen
        f.write(struct.pack('<B3BIHH20x', 3, 124, 1, 1, n, hdr_len, rec_len))
        f.write(b'FILENAME'.ljust(11, b'\0') + b'C' + b'\0' * 4 + bytes([flen, 0]) + b'\0' * 14)
        f.write(b'\x0d')
        for name, _ in records:
            f.write(b' ' + name.encode().ljust(flen, b' '))
        f.write(b'\x1a')
    with open(os.path.splitext(shp_path)[0] + '.prj', 'w') as f:
        f.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')


def make_l8(out_dir, world, wtf, res=30.0):
    """L8_ref-style: two UTM scenes, rotated-swath fill, Meta/index.shp."""
    folder = os.path.join(out_dir, 'L8_ref')
    os.makedirs(os.path.join(folder, 'Meta'), exist_ok=True)
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, 'EPSG:4326', always_xy=True)
    records = []
    for k, (lon, lat) in enumerate(((78.45, 16.45), (78.95, 16.9))):
        cE, cN = to_utm.transform(lon, lat)
        n = int(60000 / res)
        x0, y1 = cE - n * res / 2, cN + n * res / 2
        cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
        img = _sample(world, wtf, x0 + cols * res, y1 - rows * res)
        img[(cols + rows * 0.25) < n * 0.1] = 0
        name = f'LC08_SYNTH_{k + 1:02d}.tif'
        with rasterio.open(os.path.join(folder, name), 'w', driver='GTiff', height=n, width=n,
                           count=1, dtype='uint16', crs=UTM,
                           transform=Affine(res, 0, x0, 0, -res, y1), nodata=0) as dst:
            dst.write(np.clip(img, 0, 65535).astype('uint16'), 1)
        corners = [(x0, y1), (x0 + n * res, y1), (x0 + n * res, y1 - n * res), (x0, y1 - n * res)]
        ring = [to_ll.transform(x, y) for x, y in corners]
        records.append((name, ring))
    _write_index_shp(os.path.join(folder, 'Meta', 'index.shp'), records)
    return folder


def make_all(out_dir, dE=4000.0, dN=-2500.0):
    os.makedirs(out_dir, exist_ok=True)
    world, wtf = _world()
    inp = make_input(out_dir, world, wtf, dE, dN)
    c1 = make_c1(out_dir, world, wtf)
    l8 = make_l8(out_dir, world, wtf)
    return {'input': inp, 'C1': c1, 'L8_ref': l8, 'dE': dE, 'dN': dN}


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'synthetic'
    dE = float(sys.argv[2]) if len(sys.argv) > 2 else 4000.0
    dN = float(sys.argv[3]) if len(sys.argv) > 3 else -2500.0
    print(make_all(out, dE, dN))


def make_nisar_h5(out_dir, world, wtf, dE=4013.0, dN=-2487.0, res=20.0, size_km=30.0):
    """Minimal NISAR L-band GSLC: compound {r, i} complex HH, NaN fill,
    pixel-centre x/yCoordinates, projection EPSG, identification/boundingPolygon.
    Georeferencing off by (dE, dN) like make_input."""
    import h5py
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, 'EPSG:4326', always_xy=True)
    cE, cN = to_utm.transform(78.5, 16.45)
    n = int(size_km * 1000 / res)
    x0, y1 = cE - n * res / 2, cN + n * res / 2
    xc = x0 + (np.arange(n) + 0.5) * res
    yc = y1 - (np.arange(n) + 0.5) * res
    E, N = np.meshgrid(xc, yc)
    amp = _sample(world, wtf, E - dE, N - dN)
    phase = np.random.default_rng(1).random(amp.shape).astype(np.float32) * 2 * np.pi
    z = np.zeros(amp.shape, dtype=[('r', '<f4'), ('i', '<f4')])
    z['r'], z['i'] = amp * np.cos(phase), amp * np.sin(phase)
    fill = (np.arange(n)[None, :] > (n - np.arange(n)[:, None] * 0.3))
    z['r'][fill], z['i'][fill] = np.nan, np.nan
    name = 'NISAR_L2_PR_GSLC_SYNTH_001'
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + '.h5')
    ring = [to_ll.transform(x, y) for x, y in ((x0, y1), (x0 + n * res, y1),
                                               (x0 + n * res, y1 - n * res), (x0, y1 - n * res))]
    wkt = 'POLYGON ((' + ', '.join(f'{lo} {la}' for lo, la in ring + ring[:1]) + '))'
    with h5py.File(path, 'w') as f:
        g = f.create_group('science/LSAR/GSLC/grids/frequencyA')
        g.create_dataset('HH', data=z, chunks=(256, 256))
        g.create_dataset('xCoordinates', data=xc)
        g.create_dataset('yCoordinates', data=yc)
        g.create_dataset('projection', data=np.uint32(32644))
        f.create_dataset('science/LSAR/identification/boundingPolygon', data=wkt.encode())
    return path


def distortion_field(E, N, Ec, Nc, half=30000.0):
    """Known error field (input minus true ground), shaped like the manual RIVAL
    fields seen on real scenes: an east-west scale error, curvature that grows
    eastwards, and a mild north-south term. Metres."""
    u, v = (E - Ec) / half, (N - Nc) / half
    dE = 4000.0 + 0.06 * (E - Ec) + 800.0 * v ** 2 * (u + 1.0)
    dN = -2500.0 + 0.01 * (N - Nc) + 300.0 * v ** 2
    return dE, dN


def make_input_distorted(out_dir, world, wtf, res=20.0, size_km=60.0):
    """3-band UTM GeoTIFF whose error varies across the scene by kilometres
    (see distortion_field). Returns (path, (Ec, Nc))."""
    to_utm = Transformer.from_crs('EPSG:4326', UTM, always_xy=True)
    Ec, Nc = to_utm.transform(78.5, 16.45)
    n = int(size_km * 1000 / res)
    x0, y1 = Ec - n * res / 2, Nc + n * res / 2
    tf = Affine(res, 0, x0, 0, -res, y1)
    cols, rows = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    E = x0 + cols * res
    N = y1 - rows * res
    dE, dN = distortion_field(E, N, Ec, Nc)
    base = _sample(world, wtf, E - dE, N - dN)
    path = os.path.join(out_dir, 'G1A_DIST_L1.tif')
    with rasterio.open(path, 'w', driver='GTiff', height=n, width=n, count=3,
                       dtype='uint16', crs=UTM, transform=tf, nodata=0) as dst:
        for i, b in enumerate([base, 0.8 * base + 30, np.sqrt(base) * 25], start=1):
            dst.write(np.clip(b, 1, 65535).astype('uint16'), i)
    return path, (Ec, Nc)
