#!/usr/bin/env python3
"""
cog_locate -- manual geolocation (absolute location error) assessment for
NISAR L-band products, driven straight off Cloud-Optimized GeoTIFFs.

Why this exists
---------------
The imw / dqe_integrated pipeline measures NISAR-vs-Sentinel-1 offsets
automatically. Sometimes you want the opposite: to *look* at a feature, put
the cursor on it yourself, and read its coordinates. That is the only way to
sanity-check what the automatic matcher is telling you, and it is the standard
way absolute location error (ALE) is measured against corner reflectors.

Nothing here downloads a granule. A COG is read with HTTP range requests
(GDAL /vsicurl), so pulling a 2 km x 2 km chip out of a 30 GB scene costs a
few hundred kilobytes -- overviews and internal tiling do the work.

What you get
------------
  1. `info`  -- georeferencing, CRS, overview pyramid, pixel spacing in metres,
                and the bytes actually read off the wire.
  2. `chip`  -- a window (centred on lat/lon, map X/Y, or a bbox) written as a
                PNG plus a sidecar JSON carrying the affine transform.
  3. `view`  -- a single self-contained HTML file: pan/zoom the chip, and the
                cursor reads out lat/lon, map X/Y and source pixel live. Click
                to drop points, snap them to the local backscatter peak, and
                export CSV.

                `view` covers the two ways location accuracy actually gets
                measured:
                  (a) one image against surveyed ground truth
                      (`--overlay corner_reflectors.csv`) -- gives ALE directly;
                  (b) one image against a reference image (`--b <uri>`) -- two
                      linked panels, click the same feature in both, get the
                      offset. Same across/along/distance convention the
                      dqe_imw CSVs use, and the differencing stays valid when
                      the two products are in different projections.

Dependencies: numpy, and rasterio for anything that reads a real raster
(h5py only for the NISAR HDF5 path). PNG writing is done here with zlib, so
Pillow/matplotlib are not needed -- one less thing to prefetch onto an
air-gapped box.

Examples
--------
  # what am I looking at?
  python cog_locate.py info https://host/nisar_gcov_HH.tif

  # 3 km chip around a corner reflector, then open the viewer
  python cog_locate.py view https://host/nisar_gcov_HH.tif \\
      --center 34.8021,-118.0765 --size 3000m --out cr_check.html \\
      --overlay reflectors.csv

  # NISAR against a Sentinel-1 reference, side by side
  python cog_locate.py view nisar_HH.tif --b s1_vv.tif \\
      --center 34.80,-118.07 --size 4000m --out pair.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Chip", "open_raster", "read_chip", "stretch", "encode_png",
    "lonlat_grid", "load_points_csv",
]


# =============================================================================
# GDAL / rasterio environment for cheap remote reads
# =============================================================================
# These are the settings that decide whether a remote COG read costs 200 kB or
# 200 MB. GDAL_DISABLE_READDIR_ON_OPEN stops GDAL probing for sidecar files it
# will never find on an HTTP endpoint; the CURL cache keeps re-read tiles local.
GDAL_REMOTE_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "CPL_VSIL_CURL_CACHE_SIZE": str(256 * 1024 * 1024),
    "GDAL_NUM_THREADS": "ALL_CPUS",
}


def _require_rasterio():
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the host env
        raise SystemExit(
            "rasterio is required to read rasters.\n"
            "  pip install rasterio        (bundles its own GDAL wheels)\n"
            f"import error was: {exc}"
        )
    import rasterio
    return rasterio


def normalize_uri(uri: str) -> str:
    """Turn a user-supplied path/URL into something GDAL will stream.

    Local paths and GDAL-native connection strings (HDF5:"..." , NETCDF: ,
    /vsi*) are passed through untouched.
    """
    if uri.startswith(("/vsi", "HDF5:", "NETCDF:", "GTIFF_DIR:")):
        return uri
    if uri.startswith(("http://", "https://")):
        return "/vsicurl/" + uri
    if uri.startswith("s3://"):
        return "/vsis3/" + uri[len("s3://"):]
    if uri.startswith("gs://"):
        return "/vsigs/" + uri[len("gs://"):]
    return uri


def open_raster(uri: str, extra_env: Optional[Dict[str, str]] = None):
    """Open a raster for streaming. Returns (dataset, rasterio_env).

    The caller is responsible for closing both; `read_chip` handles the common
    case. The env has to stay alive for the lifetime of the dataset, which is
    why it is handed back rather than used as a context manager here.
    """
    rasterio = _require_rasterio()
    opts = dict(GDAL_REMOTE_OPTS)
    if extra_env:
        opts.update(extra_env)
    env = rasterio.Env(**opts)
    env.__enter__()
    try:
        ds = rasterio.open(normalize_uri(uri))
    except Exception:
        env.__exit__(None, None, None)
        raise
    return ds, env


# =============================================================================
# Coordinate helpers
# =============================================================================
def to_lonlat(crs, xs: Sequence[float], ys: Sequence[float]
              ) -> Tuple[List[float], List[float]]:
    """Project map coordinates to WGS84 lon/lat.

    Tries rasterio.warp first (always present when we have a raster), then
    pyproj, then assumes the CRS already is lon/lat.
    """
    xs = list(map(float, xs))
    ys = list(map(float, ys))
    if crs is None:
        return xs, ys
    try:
        from rasterio.warp import transform as _rio_transform
        lon, lat = _rio_transform(crs, "EPSG:4326", xs, ys)
        return list(lon), list(lat)
    except Exception:
        pass
    try:
        from pyproj import Transformer
        tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(xs, ys)
        return list(lon), list(lat)
    except Exception:
        return xs, ys


def from_lonlat(crs, lons: Sequence[float], lats: Sequence[float]
                ) -> Tuple[List[float], List[float]]:
    """Inverse of `to_lonlat`."""
    lons = list(map(float, lons))
    lats = list(map(float, lats))
    if crs is None:
        return lons, lats
    try:
        from rasterio.warp import transform as _rio_transform
        xs, ys = _rio_transform("EPSG:4326", crs, lons, lats)
        return list(xs), list(ys)
    except Exception:
        pass
    try:
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tf.transform(lons, lats)
        return list(xs), list(ys)
    except Exception:
        return lons, lats


def lonlat_grid(crs, transform, width: int, height: int, n: int = 17) -> Dict:
    """Sample lon/lat on a coarse grid over a chip.

    The viewer bilinearly interpolates this instead of carrying a projection
    library into the browser. Over a chip a few km across the interpolation
    error is far below a pixel for UTM, polar stereographic and geographic
    CRSs alike; `residual_m` in the output reports the check we ran.
    """
    n = max(2, int(n))
    us = np.linspace(0.0, float(width), n)
    vs = np.linspace(0.0, float(height), n)
    uu, vv = np.meshgrid(us, vs)
    a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                        transform.d, transform.e, transform.f)
    xs = c + a * uu + b * vv
    ys = f + d * uu + e * vv
    lon, lat = to_lonlat(crs, xs.ravel().tolist(), ys.ravel().tolist())
    lon = np.asarray(lon, dtype=float).reshape(uu.shape)
    lat = np.asarray(lat, dtype=float).reshape(uu.shape)

    residual_m = _grid_residual(crs, transform, width, height, us, vs, lon, lat)
    return {
        "n": n,
        "lon": lon.ravel().tolist(),
        "lat": lat.ravel().tolist(),
        "residual_m": residual_m,
    }


def _grid_residual(crs, transform, width, height, us, vs, lon, lat) -> float:
    """Worst-case error of the bilinear lon/lat grid, in metres.

    Checks the grid against the true projection at cell centres -- the point
    where bilinear interpolation is least accurate.
    """
    try:
        mu = 0.5 * (us[:-1] + us[1:])
        mv = 0.5 * (vs[:-1] + vs[1:])
        uu, vv = np.meshgrid(mu, mv)
        a, b, c, d, e, f = (transform.a, transform.b, transform.c,
                            transform.d, transform.e, transform.f)
        xs = c + a * uu + b * vv
        ys = f + d * uu + e * vv
        tlon, tlat = to_lonlat(crs, xs.ravel().tolist(), ys.ravel().tolist())
        tlon = np.asarray(tlon).reshape(uu.shape)
        tlat = np.asarray(tlat).reshape(uu.shape)

        # bilinear estimate at the same points = mean of the 4 surrounding nodes
        ilon = 0.25 * (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:])
        ilat = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:])

        mlat = np.deg2rad(np.clip(tlat, -89.9, 89.9))
        dx = (ilon - tlon) * 111320.0 * np.cos(mlat)
        dy = (ilat - tlat) * 110540.0
        return float(np.nanmax(np.hypot(dx, dy)))
    except Exception:
        return float("nan")


def pixel_size_m(crs, transform, lat_hint: float = 0.0) -> Tuple[float, float]:
    """Pixel spacing in metres, whatever the CRS is."""
    sx = abs(transform.a)
    sy = abs(transform.e)
    try:
        is_geographic = bool(crs and crs.is_geographic)
    except Exception:
        is_geographic = False
    if is_geographic:
        sx = sx * 111320.0 * math.cos(math.radians(lat_hint))
        sy = sy * 110540.0
    return sx, sy


# =============================================================================
# Chip extraction
# =============================================================================
@dataclass
class Chip:
    """A decimated window of a raster plus everything needed to georeference it."""
    data: np.ndarray                 # float32, NaN where nodata
    transform: object                # affine of THIS chip (chip px -> map)
    crs: object
    src_transform: object            # affine of the full-res source
    dec: int                         # decimation factor applied
    window: Tuple[int, int, int, int]  # col_off, row_off, width, height (full res)
    uri: str
    band: int
    nodata: object = None
    units: str = ""
    stats: Dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def chip_to_map(self, u: float, v: float) -> Tuple[float, float]:
        t = self.transform
        return (t.c + t.a * u + t.b * v, t.f + t.d * u + t.e * v)

    def bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = [], []
        for u, v in ((0, 0), (self.width, 0), (0, self.height),
                     (self.width, self.height)):
            x, y = self.chip_to_map(u, v)
            xs.append(x)
            ys.append(y)
        return min(xs), min(ys), max(xs), max(ys)


def _parse_size(size: str, px_x: float, px_y: float) -> Tuple[int, int]:
    """`--size` accepts 3000m, 3km, 2048px, or WxH in either unit."""
    s = str(size).strip().lower().replace(" ", "")
    if "x" in s and not s.endswith("px"):
        a, _, b = s.partition("x")
        w, _ = _parse_size(a, px_x, px_y)
        h, _ = _parse_size(b, px_y, px_y)
        return w, h
    if s.endswith("px"):
        n = int(round(float(s[:-2])))
        return n, n
    if s.endswith("km"):
        m = float(s[:-2]) * 1000.0
    elif s.endswith("m"):
        m = float(s[:-1])
    else:
        m = float(s)  # bare number = metres
    return max(1, int(round(m / max(px_x, 1e-9)))), max(1, int(round(m / max(px_y, 1e-9))))


def read_chip(uri: str,
              band: int = 1,
              center_lonlat: Optional[Tuple[float, float]] = None,
              center_xy: Optional[Tuple[float, float]] = None,
              bbox: Optional[Tuple[float, float, float, float]] = None,
              size: str = "2000m",
              max_px: int = 1600,
              resample: str = "average",
              full: bool = False,
              zero_is_nodata: bool = True) -> Chip:
    """Pull one window out of a (possibly remote) raster.

    Only the tiles the window touches are fetched. When the requested window is
    larger than `max_px`, the read is decimated -- which GDAL serves out of the
    overview pyramid, so a whole-scene preview stays cheap.
    """
    rasterio = _require_rasterio()
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from affine import Affine

    ds, env = open_raster(uri)
    try:
        if band < 1 or band > ds.count:
            raise SystemExit(f"band {band} out of range: {uri} has {ds.count} band(s)")

        clat = 0.0
        try:
            if ds.crs and ds.crs.is_geographic:
                clat = float((ds.bounds.bottom + ds.bounds.top) / 2.0)
        except Exception:
            pass
        px_x, px_y = pixel_size_m(ds.crs, ds.transform, clat)

        if full:
            col_off, row_off = 0, 0
            win_w, win_h = ds.width, ds.height
        elif bbox is not None:
            minx, miny, maxx, maxy = bbox
            inv = ~ds.transform
            cs, rs = [], []
            for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
                c, r = inv * (x, y)
                cs.append(c)
                rs.append(r)
            col_off, row_off = int(math.floor(min(cs))), int(math.floor(min(rs)))
            win_w = max(1, int(math.ceil(max(cs) - min(cs))))
            win_h = max(1, int(math.ceil(max(rs) - min(rs))))
        else:
            if center_xy is not None:
                cx, cy = center_xy
            elif center_lonlat is not None:
                lon, lat = center_lonlat
                (cx,), (cy,) = from_lonlat(ds.crs, [lon], [lat])
            else:
                cx = (ds.bounds.left + ds.bounds.right) / 2.0
                cy = (ds.bounds.bottom + ds.bounds.top) / 2.0
            win_w, win_h = _parse_size(size, px_x, px_y)
            ccol, crow = ~ds.transform * (cx, cy)
            col_off = int(round(ccol - win_w / 2.0))
            row_off = int(round(crow - win_h / 2.0))

        # A window fully outside the raster is a user error worth naming, but a
        # partial overlap is normal at scene edges -- boundless reads handle it.
        if (col_off + win_w <= 0 or row_off + win_h <= 0
                or col_off >= ds.width or row_off >= ds.height):
            raise SystemExit(
                f"requested window is entirely outside {uri}\n"
                f"  window (full-res px): col {col_off}..{col_off + win_w}, "
                f"row {row_off}..{row_off + win_h}\n"
                f"  raster is {ds.width} x {ds.height} px, bounds {tuple(ds.bounds)}\n"
                f"  CRS {ds.crs}"
            )

        dec = max(1, int(math.ceil(max(win_w, win_h) / float(max_px))))
        out_h = max(1, int(round(win_h / dec)))
        out_w = max(1, int(round(win_w / dec)))

        rs_enum = {
            "nearest": Resampling.nearest,
            "average": Resampling.average,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }.get(resample, Resampling.average)
        if dec == 1:
            rs_enum = Resampling.nearest  # no resampling happening anyway

        window = Window(col_off, row_off, win_w, win_h)
        data = ds.read(band, window=window, out_shape=(out_h, out_w),
                       resampling=rs_enum, boundless=True,
                       fill_value=(ds.nodata if ds.nodata is not None else 0))
        data = np.asarray(data, dtype=np.float32)

        if ds.nodata is not None and not math.isnan(ds.nodata):
            data[data == np.float32(ds.nodata)] = np.nan
        elif zero_is_nodata:
            # No declared nodata: exact zeros are the fill value essentially
            # every SAR product uses, and a log stretch cannot render them
            # anyway. --keep-zeros turns this off for data where 0 is real.
            data[data == 0] = np.nan
        data[~np.isfinite(data)] = np.nan

        chip_tf = ds.window_transform(window) * Affine.scale(
            win_w / float(out_w), win_h / float(out_h))

        return Chip(
            data=data,
            transform=chip_tf,
            crs=ds.crs,
            src_transform=ds.transform,
            dec=dec,
            window=(col_off, row_off, win_w, win_h),
            uri=uri,
            band=band,
            nodata=ds.nodata,
            units=(ds.units[band - 1] if ds.units and len(ds.units) >= band else "") or "",
            stats={"src_width": ds.width, "src_height": ds.height,
                   "src_dtype": str(ds.dtypes[band - 1]),
                   "px_x_m": px_x, "px_y_m": px_y,
                   "overviews": list(ds.overviews(band))},
        )
    finally:
        ds.close()
        env.__exit__(None, None, None)


# =============================================================================
# Radiometric stretch
# =============================================================================
# Backscatter spans four or five orders of magnitude, so a linear stretch shows
# a black image with a few white dots. dB plus a percentile clip is what makes
# a SAR scene readable -- and it is also what makes a corner reflector's peak
# visible as a peak rather than a saturated blob.
STRETCH_CHOICES = ("db", "amp-db", "linear", "log", "none")


def stretch(data: np.ndarray, mode: str = "db",
            pct: Tuple[float, float] = (2.0, 98.0),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Map raw values to display values.

    Returns (display_float, scaled_uint8, meta). `display_float` keeps the
    physical units (dB, usually) so the viewer can report the value under the
    cursor; `scaled_uint8` is what gets encoded into the PNG.
    """
    d = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(d)

    if mode == "db":
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = 10.0 * np.log10(np.where(d > 0, d, np.nan))
        units = "dB"
    elif mode == "amp-db":
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = 20.0 * np.log10(np.where(d > 0, d, np.nan))
        units = "dB"
    elif mode == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.log10(np.where(d > 0, d, np.nan))
        units = "log10"
    else:
        disp = d.copy()
        units = ""

    disp = disp.astype(np.float32)
    valid = np.isfinite(disp)
    if not valid.any():
        raise SystemExit(
            "chip contains no valid pixels -- it is entirely nodata.\n"
            "Check --center / --bbox against the raster bounds shown by `info`."
        )

    vals = disp[valid]
    lo = float(vmin) if vmin is not None else float(np.percentile(vals, pct[0]))
    hi = float(vmax) if vmax is not None else float(np.percentile(vals, pct[1]))
    if not (hi > lo):
        hi = lo + 1e-6

    if mode == "none":
        lo, hi = (float(vmin) if vmin is not None else float(vals.min()),
                  float(vmax) if vmax is not None else float(vals.max()))
        if not (hi > lo):
            hi = lo + 1e-6

    scaled = np.zeros(disp.shape, dtype=np.uint8)
    norm = (disp - lo) / (hi - lo)
    np.clip(norm, 0.0, 1.0, out=norm)
    scaled[valid] = (norm[valid] * 255.0 + 0.5).astype(np.uint8)

    meta = {
        "mode": mode, "units": units, "vmin": lo, "vmax": hi,
        "pct": list(pct),
        "valid_fraction": float(valid.mean()),
        "p50": float(np.percentile(vals, 50)),
        "min": float(vals.min()), "max": float(vals.max()),
        "n_valid": int(valid.sum()), "n_total": int(finite.size),
    }
    return disp, scaled, meta


# 16-stop LUTs, linearly interpolated to 256 entries at build time. Keeping the
# colour tables here means no matplotlib dependency on the processing box.
_CMAPS = {
    "gray": [(0, 0, 0), (255, 255, 255)],
    "gray-inv": [(255, 255, 255), (0, 0, 0)],
    "viridis": [
        (68, 1, 84), (72, 33, 115), (67, 62, 133), (56, 88, 140),
        (45, 112, 142), (37, 133, 142), (30, 155, 138), (42, 176, 127),
        (82, 197, 105), (134, 213, 73), (194, 223, 35), (253, 231, 37),
    ],
    "inferno": [
        (0, 0, 4), (22, 11, 57), (66, 10, 104), (106, 23, 110),
        (147, 38, 103), (188, 55, 84), (221, 81, 58), (243, 120, 25),
        (252, 165, 10), (246, 215, 70), (252, 255, 164),
    ],
    "magma": [
        (0, 0, 4), (24, 15, 62), (68, 15, 118), (114, 31, 129),
        (159, 47, 127), (206, 68, 111), (241, 105, 92), (252, 158, 108),
        (254, 209, 165), (252, 253, 191),
    ],
}


def build_lut(name: str) -> np.ndarray:
    """256x3 uint8 colour table."""
    stops = _CMAPS.get(name)
    if stops is None:
        raise SystemExit(f"unknown colormap '{name}'. choices: {', '.join(sorted(_CMAPS))}")
    stops_arr = np.asarray(stops, dtype=np.float64)
    xp = np.linspace(0.0, 255.0, len(stops_arr))
    x = np.arange(256, dtype=np.float64)
    lut = np.stack([np.interp(x, xp, stops_arr[:, i]) for i in range(3)], axis=1)
    return np.clip(lut + 0.5, 0, 255).astype(np.uint8)


# =============================================================================
# PNG encoding (zlib only -- no Pillow)
# =============================================================================
def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def encode_png(scaled: np.ndarray, valid: np.ndarray,
               cmap: str = "gray", level: int = 6) -> bytes:
    """Encode an 8-bit image with an alpha channel for nodata.

    Transparent nodata matters more than it sounds: at a scene edge or over
    water you want to see that there is no data, not a black patch you might
    mistake for low backscatter.
    """
    h, w = scaled.shape
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    if cmap in ("gray", None, ""):
        # colour type 4: grayscale + alpha, 2 bytes/px
        raw = np.dstack([scaled, alpha])
        color_type, channels = 4, 2
    else:
        lut = build_lut(cmap)
        rgb = lut[scaled]
        raw = np.dstack([rgb, alpha])
        color_type, channels = 6, 4

    # Filter type 0 (None) per scanline. Sub/Paeth would compress a bit better
    # but this is already dominated by zlib on speckle-heavy SAR imagery.
    stride = w * channels
    body = np.zeros((h, stride + 1), dtype=np.uint8)
    body[:, 1:] = raw.reshape(h, stride)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(body.tobytes(), level))
    png += _png_chunk(b"IEND", b"")
    return png


def quantize_values(disp: np.ndarray) -> Dict:
    """Pack the displayed physical values into uint16 for the viewer.

    The viewer needs the real values for two things: reporting dB under the
    cursor, and snapping a click to the local backscatter peak. uint16 over the
    chip's own range gives ~0.002 dB resolution on a 100 dB span, which is far
    finer than anything that affects a peak location.
    """
    valid = np.isfinite(disp)
    if not valid.any():
        return {"lo": 0.0, "hi": 1.0, "b64": "", "encoding": "none"}
    vals = disp[valid]
    lo = float(vals.min())
    hi = float(vals.max())
    if not (hi > lo):
        hi = lo + 1e-6
    q = np.zeros(disp.shape, dtype=np.uint16)
    norm = (disp - lo) / (hi - lo)
    # 0 is reserved for "no data"; real values occupy 1..65535.
    q[valid] = (1 + norm[valid] * 65534.0).astype(np.uint16)
    import base64
    # Deliberately NOT compressed: plain base64 decodes with atob() in every
    # browser, whereas inflating in the page needs DecompressionStream, which
    # a locked-down corporate browser may not have. --no-values is the escape
    # hatch if the page size matters more.
    raw = q.astype("<u2").tobytes()
    return {
        "lo": lo, "hi": hi,
        "b64": base64.b64encode(raw).decode("ascii"),
        "encoding": "u16le",
        "bytes": len(raw),
    }


# =============================================================================
# Point / ground-truth CSV
# =============================================================================
def load_points_csv(path: str, crs=None) -> List[Dict]:
    """Read surveyed points (corner reflectors, GCPs, anything).

    Accepts lat/lon or projected x/y, with flexible column names, so a
    reflector list exported from a survey sheet usually just works:
        name,lat,lon           |  id,latitude,longitude,height
        name,x,y               |  name,easting,northing
    """
    import csv as _csv
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sniff = fh.read(4096)
        fh.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sniff, delimiters=",;\t ")
        except Exception:
            dialect = _csv.excel
        reader = _csv.DictReader(fh, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit(f"{path}: no header row found")
        keymap = {(k or "").strip().lower().lstrip("#").strip(): k
                  for k in reader.fieldnames}

        def pick(*names):
            for n in names:
                if n in keymap:
                    return keymap[n]
            return None

        k_lat = pick("lat", "latitude", "y_lat", "lat_deg")
        k_lon = pick("lon", "long", "longitude", "x_lon", "lon_deg")
        k_x = pick("x", "easting", "east", "utm_x", "x_m")
        k_y = pick("y", "northing", "north", "utm_y", "y_m")
        k_name = pick("name", "id", "site", "label", "station", "cr", "reflector")

        if not ((k_lat and k_lon) or (k_x and k_y)):
            raise SystemExit(
                f"{path}: need lat/lon or x/y columns; found {reader.fieldnames}")

        for i, row in enumerate(reader):
            try:
                rec = {"name": (row.get(k_name) or f"P{i + 1}").strip()}
                if k_lat and k_lon:
                    rec["lat"] = float(row[k_lat])
                    rec["lon"] = float(row[k_lon])
                else:
                    rec["x"] = float(row[k_x])
                    rec["y"] = float(row[k_y])
                rows.append(rec)
            except (TypeError, ValueError):
                continue  # blank or comment line

    if not rows:
        raise SystemExit(f"{path}: no usable rows")

    # Fill in whichever representation is missing so the viewer has both.
    if "lat" in rows[0]:
        xs, ys = from_lonlat(crs, [r["lon"] for r in rows], [r["lat"] for r in rows])
        for r, x, y in zip(rows, xs, ys):
            r["x"], r["y"] = x, y
    else:
        lons, lats = to_lonlat(crs, [r["x"] for r in rows], [r["y"] for r in rows])
        for r, lon, lat in zip(rows, lons, lats):
            r["lon"], r["lat"] = lon, lat
    return rows


# =============================================================================
# Panel payload (what the viewer actually consumes)
# =============================================================================
def build_panel(chip: Chip, label: str, stretch_mode: str,
                pct: Tuple[float, float], vmin, vmax, cmap: str,
                embed_values: bool = True,
                grid_n: int = 17) -> Dict:
    """Turn a Chip into the JSON-able blob the HTML viewer renders."""
    import base64

    disp, scaled, smeta = stretch(chip.data, stretch_mode, pct, vmin, vmax)
    valid = np.isfinite(disp)
    png = encode_png(scaled, valid, cmap=cmap)

    t = chip.transform
    st = chip.src_transform
    grid = lonlat_grid(chip.crs, t, chip.width, chip.height, grid_n)
    clat = float(np.nanmean(np.asarray(grid["lat"], dtype=float))) if grid["lat"] else 0.0
    px_x, px_y = pixel_size_m(chip.crs, t, clat)

    values = quantize_values(disp) if embed_values else {"encoding": "none", "b64": ""}

    try:
        crs_name = chip.crs.to_string() if chip.crs else "(none)"
        epsg = chip.crs.to_epsg() if chip.crs else None
    except Exception:
        crs_name, epsg = str(chip.crs), None

    return {
        "label": label,
        "uri": chip.uri,
        "band": chip.band,
        "width": chip.width,
        "height": chip.height,
        "png": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        # chip pixel -> map coords
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        # map coords -> full-res source pixel, so a picked point can be quoted
        # in the same pixel space the dqe pipeline works in
        "src_inv": list((~st)[:6]),
        "src_transform": [st.a, st.b, st.c, st.d, st.e, st.f],
        "dec": chip.dec,
        "window": list(chip.window),
        "crs": crs_name,
        "epsg": epsg,
        "px_x_m": px_x,
        "px_y_m": px_y,
        "grid": grid,
        "stretch": smeta,
        "values": values,
        "cmap": cmap,
        "src": chip.stats,
    }


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


# =============================================================================
# Commands
# =============================================================================
def cmd_info(args) -> int:
    rasterio = _require_rasterio()
    ds, env = open_raster(args.uri)
    try:
        b = ds.bounds
        clat = 0.0
        try:
            if ds.crs and ds.crs.is_geographic:
                clat = (b.bottom + b.top) / 2.0
        except Exception:
            pass
        px_x, px_y = pixel_size_m(ds.crs, ds.transform, clat)
        corners_x = [b.left, b.right, b.right, b.left]
        corners_y = [b.bottom, b.bottom, b.top, b.top]
        lons, lats = to_lonlat(ds.crs, corners_x, corners_y)

        print(f"URI          : {args.uri}")
        print(f"driver       : {ds.driver}")
        print(f"size         : {ds.width} x {ds.height} px, {ds.count} band(s)")
        print(f"dtype        : {', '.join(ds.dtypes)}")
        print(f"nodata       : {ds.nodata}")
        print(f"CRS          : {ds.crs}")
        print(f"pixel size   : {px_x:.4g} x {px_y:.4g} m")
        print(f"transform    : {tuple(round(v, 6) for v in ds.transform[:6])}")
        print(f"bounds (map) : left={b.left:.3f} bottom={b.bottom:.3f} "
              f"right={b.right:.3f} top={b.top:.3f}")
        print("bounds (ll)  : "
              f"lon {min(lons):.6f}..{max(lons):.6f}  "
              f"lat {min(lats):.6f}..{max(lats):.6f}")
        print(f"center (ll)  : {sum(lats) / 4:.6f}, {sum(lons) / 4:.6f}")
        bs = ds.block_shapes[0] if ds.block_shapes else None
        print(f"block shape  : {bs}   (tiled={bool(bs and bs[0] != ds.height)})")
        no_ov = "(none -- not a COG pyramid; decimated reads will be slow)"
        for i in range(1, ds.count + 1):
            ov = ds.overviews(i)
            print(f"overviews b{i} : {ov if ov else no_ov}")
        if ds.descriptions and any(ds.descriptions):
            print(f"descriptions : {ds.descriptions}")
        if ds.subdatasets:
            print("subdatasets  :")
            for sd in ds.subdatasets:
                print(f"    {sd}")
        tags = ds.tags()
        if tags and args.tags:
            print("tags         :")
            for k, v in sorted(tags.items()):
                print(f"    {k} = {v}")
    finally:
        ds.close()
        env.__exit__(None, None, None)
    return 0


def _numbers(raw: str, n: int, flag: str, shape: str) -> List[float]:
    try:
        parts = [float(v) for v in str(raw).replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"{flag}: could not read numbers from '{raw}' "
                         f"-- expected {shape}")
    if len(parts) != n:
        raise SystemExit(f"{flag}: expected {n} numbers ({shape}), "
                         f"got {len(parts)} from '{raw}'")
    return parts


def _center_from_args(args):
    center_lonlat = center_xy = bbox = None
    if getattr(args, "bbox", None):
        bbox = tuple(_numbers(args.bbox, 4, "--bbox",
                              "minx,miny,maxx,maxy in the raster's CRS"))
    if getattr(args, "center", None):
        lat, lon = _numbers(args.center, 2, "--center", "lat,lon in degrees")
        if not (-90 <= lat <= 90 and -180 <= lon <= 360):
            raise SystemExit(f"--center: {lat},{lon} is not a plausible lat,lon "
                             "-- note the order is latitude first")
        center_lonlat = (lon, lat)
    if getattr(args, "center_xy", None):
        x, y = _numbers(args.center_xy, 2, "--center-xy", "x,y in the raster's CRS")
        center_xy = (x, y)
    return center_lonlat, center_xy, bbox


def _read_from_args(args, uri, band):
    center_lonlat, center_xy, bbox = _center_from_args(args)
    return read_chip(uri, band=band,
                     center_lonlat=center_lonlat, center_xy=center_xy, bbox=bbox,
                     size=args.size, max_px=args.max_px,
                     resample=args.resample, full=args.full,
                     zero_is_nodata=not getattr(args, "keep_zeros", False))


def cmd_chip(args) -> int:
    chip = _read_from_args(args, args.uri, args.band)
    disp, scaled, smeta = stretch(chip.data, args.stretch,
                                  tuple(args.pct), args.vmin, args.vmax)
    png = encode_png(scaled, np.isfinite(disp), cmap=args.cmap)

    out_png = args.out or "chip.png"
    if not out_png.lower().endswith(".png"):
        out_png += ".png"
    with open(out_png, "wb") as fh:
        fh.write(png)

    t = chip.transform
    grid = lonlat_grid(chip.crs, t, chip.width, chip.height, 3)
    sidecar = {
        "uri": chip.uri, "band": chip.band,
        "width": chip.width, "height": chip.height,
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "crs": str(chip.crs),
        "dec": chip.dec, "window": list(chip.window),
        "stretch": smeta,
        "corners_lonlat": {"lon": grid["lon"], "lat": grid["lat"]},
        "src": chip.stats,
    }
    out_json = out_png[:-4] + ".json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2)

    print(f"wrote {out_png}  ({chip.width} x {chip.height} px, {_fmt_bytes(len(png))})")
    print(f"wrote {out_json}")
    print(f"  decimation   : {chip.dec}x  (window {chip.window[2]} x {chip.window[3]} "
          f"full-res px)")
    print(f"  display range: {smeta['vmin']:.2f} .. {smeta['vmax']:.2f} {smeta['units']}"
          f"  ({smeta['mode']}, {smeta['valid_fraction'] * 100:.1f}% valid)")
    return 0


def cmd_view(args) -> int:
    from cog_viewer import build_viewer_html

    panels = []
    chip_a = _read_from_args(args, args.uri, args.band)
    panels.append(build_panel(chip_a, args.label_a or os.path.basename(args.uri),
                              args.stretch, tuple(args.pct), args.vmin, args.vmax,
                              args.cmap, embed_values=not args.no_values,
                              grid_n=args.grid_nodes))
    print(f"[A] {args.uri}\n    {chip_a.width} x {chip_a.height} px, dec {chip_a.dec}x, "
          f"{panels[0]['px_x_m']:.3g} m/px, {panels[0]['stretch']['valid_fraction'] * 100:.1f}% valid")

    if args.b:
        # The reference is read over the SAME map footprint, not the same pixel
        # window: the two products rarely share a grid, and it is the ground
        # footprint that has to match for the comparison to mean anything.
        bnds = chip_a.bounds()
        chip_b = read_chip(args.b, band=args.band_b,
                           bbox=_bbox_in_crs(bnds, chip_a.crs, args.b),
                           max_px=args.max_px, resample=args.resample,
                           zero_is_nodata=not args.keep_zeros)
        panels.append(build_panel(chip_b, args.label_b or os.path.basename(args.b),
                                  args.stretch_b or args.stretch, tuple(args.pct),
                                  None, None, args.cmap,
                                  embed_values=not args.no_values,
                                  grid_n=args.grid_nodes))
        print(f"[B] {args.b}\n    {chip_b.width} x {chip_b.height} px, dec {chip_b.dec}x, "
              f"{panels[1]['px_x_m']:.3g} m/px, "
              f"{panels[1]['stretch']['valid_fraction'] * 100:.1f}% valid")

    overlay = load_points_csv(args.overlay, chip_a.crs) if args.overlay else []
    if overlay:
        # Per-panel projected coordinates, indexed by panel position.
        lons = [p["lon"] for p in overlay]
        lats = [p["lat"] for p in overlay]
        per_panel = []
        for ch in ([chip_a] + ([chip_b] if args.b else [])):
            xs, ys = from_lonlat(ch.crs, lons, lats)
            per_panel.append(list(zip(xs, ys)))
        for i, p in enumerate(overlay):
            p["xy"] = [[per_panel[k][i][0], per_panel[k][i][1]]
                       for k in range(len(per_panel))]

        minx, miny, maxx, maxy = chip_a.bounds()
        inside = [p for p in overlay
                  if minx <= p["x"] <= maxx and miny <= p["y"] <= maxy]
        print(f"[overlay] {len(overlay)} point(s) loaded from {args.overlay}, "
              f"{len(inside)} inside the chip")
        if not inside:
            print("          none fall inside the chip -- check the CSV coordinates "
                  "or widen --size")

    html = build_viewer_html(
        panels=panels,
        overlay=overlay,
        title=args.title or "cog_locate -- L-band geolocation check",
        assoc_radius=args.assoc_radius,
        snap_radius=args.snap_radius,
    )
    out = args.out or "cog_locate_view.html"
    if not out.lower().endswith((".html", ".htm")):
        out += ".html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    size = os.path.getsize(out)
    print(f"\nwrote {out}  ({_fmt_bytes(size)})")
    if size > 40 * 1024 * 1024:
        print("  NOTE: that is a big page. Re-run with --no-values or a smaller "
              "--max-px if the browser struggles.")
    print(f"  open it with: xdg-open {out}    (or just double-click it)")
    return 0


def _bbox_in_crs(bounds, src_crs, other_uri):
    """Re-express a bbox from one raster's CRS in another raster's CRS."""
    minx, miny, maxx, maxy = bounds
    ds, env = open_raster(other_uri)
    try:
        dst_crs = ds.crs
    finally:
        ds.close()
        env.__exit__(None, None, None)
    try:
        same = (src_crs and dst_crs and src_crs == dst_crs)
    except Exception:
        same = False
    if same or dst_crs is None or src_crs is None:
        return (minx, miny, maxx, maxy)
    lons, lats = to_lonlat(src_crs, [minx, minx, maxx, maxx], [miny, maxy, miny, maxy])
    xs, ys = from_lonlat(dst_crs, lons, lats)
    return (min(xs), min(ys), max(xs), max(ys))


def cmd_selftest(args) -> int:
    """Build a synthetic scene and view it -- proves the toolchain end to end.

    Uses no network and no rasterio: handy for checking the viewer on a machine
    that has not got the geo stack installed yet.
    """
    from affine import Affine
    from cog_viewer import build_viewer_html

    h = w = 512
    # A geographic chip near the NISAR cal site, ~0.3 m/px equivalent.
    res = 3.0e-6
    lon0, lat0 = -118.10, 34.82
    tf = Affine(res, 0.0, lon0, 0.0, -res, lat0)

    rng = np.random.default_rng(0)
    # Speckle: exponential intensity, the right first-order model for SAR power.
    data = rng.exponential(0.05, size=(h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    data += 0.02 * (1 + np.sin(xx / 40.0) * np.cos(yy / 55.0))
    # Three bright point targets standing in for corner reflectors.
    truth = [(128.5, 160.5), (300.25, 210.75), (390.0, 400.0)]
    for (pr, pc) in truth:
        r2 = (yy - pr) ** 2 + (xx - pc) ** 2
        data += 40.0 * np.exp(-r2 / 2.0)
    data[0:12, :] = np.nan  # a nodata edge to exercise the alpha channel

    # Use a real CRS when rasterio is around so pixel sizes come out in metres;
    # without it the chip still renders, just with degree-sized "metres".
    try:
        from rasterio.crs import CRS
        crs = CRS.from_epsg(4326)
    except Exception:
        crs = None

    chip = Chip(data=data, transform=tf, crs=crs, src_transform=tf, dec=1,
                window=(0, 0, w, h), uri="selftest://synthetic", band=1)
    panel = build_panel(chip, "synthetic", "db", (2.0, 98.0), None, None,
                        args.cmap, embed_values=True, grid_n=5)

    overlay = []
    for i, (pr, pc) in enumerate(truth):
        x, y = chip.chip_to_map(pc + 0.5, pr + 0.5)
        # Offset the "surveyed" position by a known 1.5 px so the ALE readout
        # has something non-zero to show.
        overlay.append({"name": f"CR{i + 1}", "x": x + res * 1.5, "y": y - res * 1.5,
                        "lon": x + res * 1.5, "lat": y - res * 1.5})

    html = build_viewer_html([panel], overlay, "cog_locate selftest",
                             assoc_radius=50.0, snap_radius=args.snap_radius)
    out = args.out or "cog_locate_selftest.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out} ({_fmt_bytes(os.path.getsize(out))})")
    print("Three synthetic point targets are offset 1.5 px NE of their listed "
          "'surveyed' positions.")
    print("Click each bright dot with snap-to-peak on: the ALE column should "
          "read about 1.5 px worth of metres, and the mean bias should be "
          "consistent across all three.")
    return 0


# =============================================================================
# CLI
# =============================================================================
def _add_window_args(p):
    g = p.add_argument_group("window")
    g.add_argument("--center", help="chip centre as lat,lon (WGS84)")
    g.add_argument("--center-xy", help="chip centre as x,y in the raster's own CRS")
    g.add_argument("--bbox", help="minx,miny,maxx,maxy in the raster's own CRS")
    g.add_argument("--full", action="store_true",
                   help="whole scene (decimated to --max-px; cheap on a real COG)")
    g.add_argument("--size", default="2000m",
                   help="window size: 3000m, 3km, 2048px, or WxH (default 2000m)")
    g.add_argument("--max-px", type=int, default=1600,
                   help="max chip dimension; larger windows are decimated "
                        "via overviews (default 1600)")
    g.add_argument("--keep-zeros", action="store_true",
                   help="treat exact zeros as real data (default: zeros are "
                        "nodata when the raster declares none)")
    g.add_argument("--resample", default="average",
                   choices=("average", "nearest", "bilinear", "cubic"),
                   help="resampling for decimated reads (default average). "
                        "Use nearest when locating point targets.")


def _add_render_args(p):
    g = p.add_argument_group("rendering")
    g.add_argument("--stretch", default="db", choices=STRETCH_CHOICES,
                   help="db=10log10 (power, the NISAR GCOV case), "
                        "amp-db=20log10 (amplitude), default db")
    g.add_argument("--pct", type=float, nargs=2, default=[2.0, 98.0],
                   metavar=("LO", "HI"), help="percentile clip (default 2 98)")
    g.add_argument("--vmin", type=float, help="explicit display minimum (post-stretch units)")
    g.add_argument("--vmax", type=float, help="explicit display maximum")
    g.add_argument("--cmap", default="gray",
                   choices=tuple(sorted(_CMAPS)),
                   help="colour table (default gray)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cog_locate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="georeferencing + overview pyramid of a raster")
    pi.add_argument("uri")
    pi.add_argument("--tags", action="store_true", help="also dump GDAL metadata tags")
    pi.set_defaults(func=cmd_info)

    pc = sub.add_parser("chip", help="write a PNG + georeferencing sidecar")
    pc.add_argument("uri")
    pc.add_argument("--band", type=int, default=1)
    pc.add_argument("--out", help="output PNG path (default chip.png)")
    _add_window_args(pc)
    _add_render_args(pc)
    pc.set_defaults(func=cmd_chip)

    pv = sub.add_parser("view", help="build the interactive HTML viewer")
    pv.add_argument("uri", help="the L-band product (COG URL, s3://, or local path)")
    pv.add_argument("--band", type=int, default=1)
    pv.add_argument("--b", help="second raster to compare against (e.g. an S1 reference)")
    pv.add_argument("--band-b", type=int, default=1)
    pv.add_argument("--label-a", help="panel A caption")
    pv.add_argument("--label-b", help="panel B caption")
    pv.add_argument("--stretch-b", choices=STRETCH_CHOICES,
                    help="separate stretch for panel B (optical references "
                         "usually want 'linear')")
    pv.add_argument("--overlay", help="CSV of surveyed points (corner reflectors / GCPs)")
    pv.add_argument("--assoc-radius", type=float, default=100.0,
                    help="metres: how close a picked point must be to an overlay "
                         "point to be paired with it (default 100)")
    pv.add_argument("--snap-radius", type=int, default=6,
                    help="pixels: search radius for snap-to-peak (default 6)")
    pv.add_argument("--grid-nodes", type=int, default=17,
                    help="lon/lat interpolation grid density (default 17)")
    pv.add_argument("--no-values", action="store_true",
                    help="do not embed raw values (smaller page; disables the "
                         "dB readout and snap-to-peak)")
    pv.add_argument("--title", help="page title")
    pv.add_argument("--out", help="output HTML path (default cog_locate_view.html)")
    _add_window_args(pv)
    _add_render_args(pv)
    pv.set_defaults(func=cmd_view)

    ps = sub.add_parser("selftest",
                        help="build a synthetic viewer (no network, no rasterio)")
    ps.add_argument("--out")
    ps.add_argument("--cmap", default="gray", choices=tuple(sorted(_CMAPS)))
    ps.add_argument("--snap-radius", type=int, default=6)
    ps.set_defaults(func=cmd_selftest)

    return p


def main(argv=None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
