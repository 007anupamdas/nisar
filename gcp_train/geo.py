"""Raster access (.img with pyramids, GeoTIFF, VRT, NISAR amplitude) and coordinates."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

RASTER_EXTS = ('.img', '.tif', '.tiff', '.vrt', '.jp2')


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def transformer(src_crs, dst_crs):
    from pyproj import CRS, Transformer
    return Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)


def lonlat_to_pixel(src, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(col, row) with pixel centres at integers, of lon/lat in an open raster."""
    tr = transformer('EPSG:4326', src.crs)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    inv = ~src.transform
    c, r = inv * (np.asarray(x), np.asarray(y))
    return np.asarray(c) - 0.5, np.asarray(r) - 0.5


def pixel_to_crs(src, col: np.ndarray, row: np.ndarray, crs) -> Tuple[np.ndarray, np.ndarray]:
    """Map coordinates in crs of pixel centres (col, row) of an open raster."""
    x, y = src.transform * (np.asarray(col, float) + 0.5, np.asarray(row, float) + 0.5)
    return transformer(src.crs, crs).transform(np.asarray(x), np.asarray(y))


def ground_pixel_m(src) -> float:
    """The raster's pixel size on the ground, in metres."""
    if src.crs is not None and src.crs.is_projected:
        return float(abs(src.transform.a))
    cx, cy = src.transform * (src.width / 2.0, src.height / 2.0)
    return float(abs(src.transform.a) * 111320.0 * math.cos(math.radians(cy)))


def read_window(src, band: int, col0: int, row0: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """(data, valid) of a size x size window at native resolution; outside the
    raster and no-data are invalid."""
    import rasterio.windows as W
    a = src.read(band, window=W.Window(col0, row0, size, size), boundless=True,
                 fill_value=src.nodata if src.nodata is not None else 0).astype(np.float32)
    nd = src.nodata if src.nodata is not None else 0
    valid = np.isfinite(a) & (a != nd)
    return a, valid


def read_scaled(src, band: int, col0: float, row0: float, size: int, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """(data, valid) of a size x size crop whose pixels are factor input pixels
    wide (factor > 1: coarser), starting at input pixel (col0, row0). Averaged
    (GDAL uses the .img / .ovr pyramids for this), no-data excluded."""
    if factor <= 1.0 + 1e-9:
        return read_window(src, band, int(round(col0)), int(round(row0)), size)
    import rasterio.windows as W
    from rasterio.enums import Resampling
    nd = src.nodata if src.nodata is not None else 0
    win = W.Window(col0, row0, size * factor, size * factor)
    a = src.read(band, window=win, out_shape=(size, size), resampling=Resampling.average, boundless=True,
                 fill_value=nd).astype(np.float32)
    valid = np.isfinite(a) & (a != nd)
    return a, valid


def reproject_to_grid(paths_bands: List[Tuple[str, int]], crs, x0: float, y1: float, px: float, size: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """The reference rasters resampled (bilinear) onto a north-up size x size
    grid of px metres whose top-left corner is (x0, y1) in crs; the first
    raster with data wins where several cover a pixel. (data, valid)."""
    import rasterio as rio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    dst_t = from_origin(x0, y1, px, px)
    out = np.zeros((size, size), np.float32)
    have = np.zeros((size, size), bool)
    for path, band in paths_bands:
        with rio.open(path) as s:
            nd = s.nodata if s.nodata is not None else 0
            buf = np.full((size, size), np.nan, np.float32)
            reproject(rio.band(s, band), buf, src_nodata=nd, dst_nodata=np.nan,
                      dst_transform=dst_t, dst_crs=crs, resampling=Resampling.bilinear)
        new = ~have & np.isfinite(buf) & (buf != nd)
        out[new] = buf[new]
        have |= new
        if have.all():
            break
    return out, have


def footprint_lonlat(path: str) -> Tuple[float, float, float, float]:
    """(west, south, east, north) of a raster in lon/lat."""
    import rasterio as rio
    from rasterio.warp import transform_bounds
    with rio.open(path) as s:
        return tuple(transform_bounds(s.crs, 'EPSG:4326', *s.bounds, densify_pts=21))


def find_rasters(folder: str) -> List[str]:
    out = []
    for dp, _, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(RASTER_EXTS) and not f.lower().endswith('.aux.xml'):
                out.append(os.path.join(dp, f))
    return sorted(out)


def references_for(ref_dir: str, bbox: Tuple[float, float, float, float], cache: Optional[Dict] = None
                   ) -> List[str]:
    """Reference rasters under ref_dir whose footprint meets the lon/lat bbox;
    footprints are kept in cache (a dict, e.g. loaded from JSON) between calls."""
    cache = {} if cache is None else cache
    w, s, e, n = bbox
    hits = []
    for p in find_rasters(ref_dir):
        fp = cache.get(p)
        if fp is None:
            try:
                fp = footprint_lonlat(p)
            except Exception:
                fp = (0.0, 0.0, 0.0, 0.0)
            cache[p] = list(fp)
        if fp[0] < e and fp[2] > w and fp[1] < n and fp[3] > s:
            hits.append(p)
    return hits


def stretch_u8(a: np.ndarray, valid: np.ndarray, plo: float = 1.0, phi: float = 99.0) -> np.ndarray:
    """log1p, then the plo..phi percentiles of the valid pixels to 0..255 (as
    the matching tool stretches windows); invalid pixels 0."""
    x = np.where(valid, a, np.nan).astype(np.float64)
    pos = valid & (x > 0)
    x[pos] = np.log1p(x[pos])
    v = x[valid]
    out = np.zeros(a.shape, np.uint8)
    if v.size < 10:
        return out
    lo, hi = np.percentile(v, [plo, phi])
    if hi <= lo:
        return out
    y = np.clip((x - lo) / (hi - lo), 0, 1) * 255.0
    out[valid] = y[valid].astype(np.uint8)
    return out
