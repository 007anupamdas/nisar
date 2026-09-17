#!/usr/bin/env python3
"""DPQED_gcov2tif - a NISAR GCOV HDF5 as a GeoTIFF that RADIAL can measure.

Why this exists
---------------
RADIAL reads a GCOV '.h5' directly, but only where h5py is installed -- and a
QGIS built without it says so and sends you here. The obvious move is then
DPQED_h52tif.py, which predates GCOV: it is written for a GSLC, takes the
magnitude of complex channels, and writes the bands unnamed. Three things go
missing on the way, and none of them announces itself:

  the band names      A GeoTIFF band carries a description, and RADIAL reads
                      the polarization out of it -- HHHH becomes the HH_
                      column. Unnamed, every column exports as b1_, b2_, and
                      two products' exports no longer line up by name.

  the RTC factor      GCOV stores gamma0. Converting to sigma0 needs
                      rtcGammaToSigmaFactor, which sits beside the covariance
                      terms in the HDF5 and is simply not carried across. A
                      TIF without it cannot offer sigma0 at all.

  the half pixel      A grid states its pixel CENTRES; a geotransform is
                      anchored on the outer EDGE. Half a 30 m pixel is 15 m --
                      the size of the errors RIVAL exists to measure.

So this writes what RADIAL actually wants: the diagonal covariance terms as
named bands, the RTC factor beside them as one more, the grid's own
geotransform and projection, and NaN as nodata.

The result is a Cloud Optimized GeoTIFF -- tiled, with an overview pyramid,
and with the headers and overviews laid out ahead of the full-resolution data
so a reader can find its way around the file from the first few kilobytes. That
is what makes panning a 20000 px scene in QGIS bearable, and it is what
cog_locate.py in this repo expects of a raster it is pointed at; over HTTP it
is the difference between fetching the tiles you asked for and fetching the
file. The layout is GDAL's COG driver's, not this file's approximation of it.
Pass --no-cog for a plain tiled GeoTIFF.

'Store it as an attribute or a header'
--------------------------------------
The factor cannot be a header. It is a value PER PIXEL -- it depends on the
local slope, which is the whole reason it exists -- so a single number in the
TIFF tags would be a different measurement, right only where the ground is
flat. It goes in as a band.

What does go in the header is which band that is: RTC_GAMMA_TO_SIGMA_BAND, a
plain GDAL metadata item. RADIAL finds the factor by band name first and falls
back to that tag, so a TIF written by some other tool can declare its factor
band without having to rename anything.

Nothing here converts to sigma0. The TIF holds gamma0, as the product does,
plus the means to convert it; RADIAL's 'As:' selector does the conversion and
records which convention each figure is in. Baking sigma0 into the pixels would
produce a file that looks exactly like a gamma0 one.

Usage
-----
    python3 DPQED_gcov2tif.py NISAR_..._GCOV.h5
    python3 DPQED_gcov2tif.py in.h5 -o out.tif --frequency A --band LSAR
    python3 DPQED_gcov2tif.py in.h5 --list          # what is in the file
    python3 DPQED_gcov2tif.py in.h5 --no-factor     # terms only
    python3 DPQED_gcov2tif.py in.h5 --no-cog        # plain GeoTIFF instead

Needs h5py and numpy, and either GDAL or rasterio to write -- whichever the
environment that ran DPQED_h52tif.py already has.
"""

import argparse
import os
import re
import sys

import numpy as np

# Rows read and written at a time. A frequency-A GCOV term can be 20000 px
# square, which is 1.6 GB per band in float32: reading one whole is how a
# conversion turns into a swap storm on a laptop.
BLOCK_ROWS = 1024

# GeoTIFF creation. DEFLATE and tiling because the result is opened in a GIS
# and read in windows; predictor 3 is the floating-point one, which is what
# these bands are; BIGTIFF only when it is actually needed.
TIFF_OPTIONS = ("TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256",
                "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER",
                "NUM_THREADS=ALL_CPUS")
OVERVIEW_LEVELS = (2, 4, 8, 16, 32)

# Cloud Optimized GeoTIFF, written by GDAL's own COG driver rather than
# assembled here: the layout is the whole point of the format and a hand-rolled
# approximation of it is just a tiled GeoTIFF with a misleading name. The driver
# builds the overview pyramid itself, averaging -- which is the right
# reduction for power, where the mean of a block IS the block's backscatter.
COG_OPTIONS = ("COMPRESS=DEFLATE", "PREDICTOR=FLOATING_POINT", "BLOCKSIZE=256",
               "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS",
               "OVERVIEWS=AUTO", "RESAMPLING=AVERAGE")

# The header item naming the factor's band, for a reader that cannot rely on
# band descriptions surviving whatever wrote the file.
RTC_FACTOR_BAND_KEY = "RTC_GAMMA_TO_SIGMA_BAND"


# ── BEGIN SHARED GCOV HELPERS ─────────────────────────────────────────────────
# Character-identical to the same block in DPQED_radial.py. Two files rather
# than an import because RADIAL is exec'd as a single file in the QGIS console,
# where a sibling import is not reliably on the path; tests_radial_stats.py
# asserts the two copies have not drifted.

GCOV_POL_TERMS = ("HHHH", "HVHV", "VHVH", "VVVV", "RHRH", "RVRV")
GCOV_GRID_RE = re.compile(
    r"science/(?P<band>[LS]SAR)/GCOV/grids/frequency(?P<freq>[A-Z])/"
    r"(?P<term>[A-Z]{4})$")
RTC_FACTOR_RE = re.compile(r"(?i)gamma.?to.?sigma")


def gcov_grids(subdataset_names):
    """GDAL's subdataset list as {(band, frequency): {term: name}}.

    GDAL reports every dataset in the HDF5 tree, which for a GCOV is the
    covariance terms plus coordinate vectors, masks and metadata. Only the
    grids under a frequency are of interest, and only their four-letter terms.
    """
    grids = {}
    for name in subdataset_names:
        match = GCOV_GRID_RE.search(str(name).strip())
        if match:
            key = (match.group("band"), match.group("freq"))
            grids.setdefault(key, {})[match.group("term")] = str(name).strip()
    return grids


def gcov_diagonal_terms(terms):
    """The diagonal covariance terms, in a fixed order, from what a grid holds.

    Only the diagonal is offered. An off-diagonal term such as HHHV is a
    complex covariance between two channels: its magnitude is a correlation,
    not a backscatter, and averaging it into a gamma0 column would produce a
    number that looks like a measurement and is not one.
    """
    return [term for term in GCOV_POL_TERMS if term in set(terms)]


def geotransform_from_coords(x_coords, y_coords, tol=1e-6):
    """A north-up geotransform from a GCOV grid's coordinate vectors.

    The vectors give pixel CENTRES; a geotransform is anchored on the outer
    EDGE of the first pixel, so each origin steps back half a pixel. Getting
    that wrong offsets every ROI by half a pixel -- 15 m on a 30 m GCOV grid,
    which is the size of the errors RIVAL exists to measure.
    """
    x = np.asarray(x_coords, dtype=float).ravel()
    y = np.asarray(y_coords, dtype=float).ravel()
    if x.size < 2 or y.size < 2:
        raise ValueError("coordinate vectors need at least two samples")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    for name, values, step in (("x", x, dx), ("y", y, dy)):
        spacing = np.diff(values)
        if not np.all(np.abs(spacing - step) <= abs(step) * tol):
            raise ValueError(f"{name} coordinates are not evenly spaced; "
                             "a VRT cannot describe that grid")
    return (float(x[0] - dx / 2.0), dx, 0.0, float(y[0] - dy / 2.0), 0.0, dy)


# ── END SHARED GCOV HELPERS ───────────────────────────────────────────────────


def choose_grid(grids, band=None, frequency=None):
    """Which grid to convert, and why.

    Frequency A is the wideband channel and is what a GCOV carries when it
    carries one, so it wins by default; B is converted when it is asked for or
    when it is all there is.
    """
    if not grids:
        raise ValueError("no GCOV grids in this file -- a GSLC or an RSLC is "
                         "not a GCOV, and has no gamma0 to measure")
    keys = sorted(grids, key=lambda k: (k[1] != "A", k))
    if band:
        keys = [k for k in keys if k[0].upper() == band.upper()]
    if frequency:
        keys = [k for k in keys if k[1].upper() == frequency.upper()]
    if not keys:
        raise ValueError(f"no grid matches band={band} frequency={frequency}; "
                         f"this file has "
                         f"{', '.join(f'{b} frequency{f}' for b, f in sorted(grids))}")
    return keys[0]


def plan_bands(members, with_factor=True):
    """(band names to write, the factor's 1-based band number or None).

    The factor goes last, so the polarization bands keep the numbers they would
    have had without it and a script written against a factor-less TIF still
    reads the right band.
    """
    terms = gcov_diagonal_terms(members)
    if not terms:
        raise ValueError("this grid holds no diagonal covariance term "
                         "(HHHH, HVHV, VHVH, VVVV); there is no backscatter "
                         "in it to write")
    names = list(terms)
    factor_band = None
    if with_factor:
        extras = sorted(name for name in members if RTC_FACTOR_RE.search(name))
        if extras:
            names.append(extras[0])
            factor_band = len(names)
    return names, factor_band


def describe(h5_path, handle=None):
    """What grids and terms a file holds, as printable lines."""
    import h5py
    lines = []
    opened = handle is None
    handle = handle or h5py.File(h5_path, "r")
    try:
        paths = []
        handle.visit(paths.append)
        grids = gcov_grids(paths)
        if not grids:
            return ["no GCOV grids in this file"]
        for key in sorted(grids):
            group = f"science/{key[0]}/GCOV/grids/frequency{key[1]}"
            members = sorted(handle[group])
            terms = gcov_diagonal_terms(members)
            factor = [m for m in members if RTC_FACTOR_RE.search(m)]
            shape = handle[f"{group}/{terms[0]}"].shape if terms else "?"
            lines.append(f"{key[0]} frequency{key[1]}  {shape}")
            lines.append(f"    diagonal terms : {', '.join(terms) or '(none)'}")
            lines.append(f"    RTC factor     : "
                         f"{', '.join(factor) or '(absent -- no sigma0)'}")
            others = [m for m in members
                      if m not in terms and m not in factor
                      and not m.endswith("Coordinates")]
            if others:
                lines.append(f"    also present   : {', '.join(others)}")
    finally:
        if opened:
            handle.close()
    return lines


def _pick_writer():
    """GDAL if the environment has it, else rasterio."""
    try:
        from osgeo import gdal  # noqa: F401
        return _gdal_writer
    except ImportError:
        pass
    try:
        import rasterio  # noqa: F401
        return _rasterio_writer
    except ImportError:
        return None


def convert(h5_path, out_path=None, band=None, frequency=None,
            with_factor=True, cog=True, verbose=True):
    """Write one GCOV grid to a GeoTIFF. Returns the path written."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        paths = []
        handle.visit(paths.append)
        key = choose_grid(gcov_grids(paths), band, frequency)
        group = f"science/{key[0]}/GCOV/grids/frequency{key[1]}"
        members = list(handle[group])
        names, factor_band = plan_bands(members, with_factor)

        first = handle[f"{group}/{names[0]}"]
        height, width = int(first.shape[0]), int(first.shape[1])
        geotransform = geotransform_from_coords(
            handle[f"{group}/xCoordinates"][()],
            handle[f"{group}/yCoordinates"][()])
        epsg = int(np.asarray(handle[f"{group}/projection"][()]).ravel()[0])

        for name in names:
            shape = handle[f"{group}/{name}"].shape
            if tuple(shape[:2]) != (height, width):
                raise ValueError(
                    f"{name} is {shape}, not {(height, width)}: the bands of "
                    "one GeoTIFF have to be on one grid")

        if out_path is None:
            out_path = os.path.splitext(h5_path)[0] + "_gcov.tif"
        metadata = {
            "NISAR_PRODUCT": os.path.basename(h5_path),
            "NISAR_BAND": key[0],
            "NISAR_FREQUENCY": key[1],
            "BACKSCATTER": "gamma0",
            "GCOV2TIF_BANDS": ",".join(names),
        }
        if factor_band:
            metadata[RTC_FACTOR_BAND_KEY] = str(factor_band)

        if verbose:
            print(f"[GCOV] {key[0]} frequency{key[1]}: {width} x {height}, "
                  f"EPSG:{epsg}")
            for number, name in enumerate(names, start=1):
                note = "  <- RTC gamma-to-sigma factor" if number == factor_band else ""
                print(f"[GCOV]   band {number}: {name}{note}")
            if factor_band is None:
                print("[GCOV]   no RTC factor written: sigma0 will not be "
                      "selectable from this TIF")

        writer = _pick_writer()
        if writer is None:
            raise RuntimeError("neither GDAL nor rasterio is available to "
                               "write a GeoTIFF")
        # The COG driver only copies -- it has no Create() -- so the bands are
        # written to a sibling first. A sibling rather than the system temp
        # directory, so the rename at the end stays on one filesystem and a
        # 20 GB product is not copied across devices to finish.
        staged = out_path + ".building.tif" if cog else out_path
        try:
            writer(staged, handle, group, names, width, height, geotransform,
                   epsg, metadata, False, verbose)
            if cog:
                _to_cog(staged, out_path, verbose)
        finally:
            if cog and os.path.exists(staged):
                try:
                    os.remove(staged)
                except OSError as e:
                    print(f"[GCOV] could not remove {staged}: {e}")
    if verbose:
        print(f"[GCOV] -> {out_path}{' (COG)' if cog else ''}")
    return out_path


def _to_cog(staged, out_path, verbose=True):
    """Re-lay a written GeoTIFF as a Cloud Optimized one, in place of it.

    Band descriptions, nodata and the dataset metadata -- including which band
    is the RTC factor -- have to survive this, since everything downstream
    reads the file by them. GDAL's copy carries all three; the tests read the
    finished COG back and check.
    """
    if verbose:
        print("[GCOV] laying out as a COG ...")
    try:
        from osgeo import gdal
        source = gdal.Open(staged, gdal.GA_ReadOnly)
        if source is None:
            raise RuntimeError(f"could not reopen {staged}")
        driver = gdal.GetDriverByName("COG")
        if driver is None:
            raise RuntimeError("this GDAL has no COG driver (it needs 3.1+)")
        copy = driver.CreateCopy(out_path, source, options=list(COG_OPTIONS))
        if copy is None:
            raise RuntimeError(f"COG copy of {staged} failed")
        copy = source = None
        return
    except ImportError:
        pass
    import rasterio.shutil
    rasterio.shutil.copy(staged, out_path, driver="COG",
                         **dict(option.split("=", 1) for option in COG_OPTIONS))


def _blocks(height, rows=BLOCK_ROWS):
    for row0 in range(0, height, rows):
        yield row0, min(rows, height - row0)


def _gdal_writer(out_path, handle, group, names, width, height, geotransform,
                 epsg, metadata, overviews=False, verbose=True):
    from osgeo import gdal, osr
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, width, height, len(names), gdal.GDT_Float32,
                       options=list(TIFF_OPTIONS))
    if ds is None:
        raise RuntimeError(f"GDAL could not create {out_path}")
    try:
        ds.SetGeoTransform(list(geotransform))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        ds.SetProjection(srs.ExportToWkt())
        ds.SetMetadata({k: str(v) for k, v in metadata.items()})
        for number, name in enumerate(names, start=1):
            band = ds.GetRasterBand(number)
            band.SetDescription(name)       # what RADIAL reads the column from
            band.SetNoDataValue(float("nan"))
            source = handle[f"{group}/{name}"]
            for row0, rows in _blocks(height):
                band.WriteArray(
                    np.asarray(source[row0:row0 + rows, :], dtype=np.float32),
                    0, row0)
            band.FlushCache()
        if overviews:
            if verbose:
                print("[GCOV] building overviews ...")
            ds.BuildOverviews("AVERAGE", list(OVERVIEW_LEVELS))
    finally:
        ds = None


def _rasterio_writer(out_path, handle, group, names, width, height,
                     geotransform, epsg, metadata, overviews=False,
                     verbose=True):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.windows import Window
    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "count": len(names), "dtype": "float32",
        "crs": CRS.from_epsg(int(epsg)),
        "transform": Affine.from_gdal(*geotransform),
        "nodata": float("nan"), "tiled": True, "blockxsize": 256,
        "blockysize": 256, "compress": "DEFLATE", "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for number, name in enumerate(names, start=1):
            source = handle[f"{group}/{name}"]
            for row0, rows in _blocks(height):
                dst.write(
                    np.asarray(source[row0:row0 + rows, :], dtype=np.float32),
                    number, window=Window(0, row0, width, rows))
        dst.descriptions = tuple(names)
        dst.update_tags(**{k: str(v) for k, v in metadata.items()})
        if overviews:
            if verbose:
                print("[GCOV] building overviews ...")
            dst.build_overviews(list(OVERVIEW_LEVELS), Resampling.average)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="DPQED_gcov2tif",
        description=__doc__.split("Usage")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="NISAR GCOV '.h5'")
    parser.add_argument("-o", "--output", help="GeoTIFF to write "
                        "(default: <input>_gcov.tif)")
    parser.add_argument("--band", help="LSAR or SSAR (default: whichever is "
                        "there)")
    parser.add_argument("--frequency", help="A or B (default: A)")
    parser.add_argument("--no-factor", action="store_true",
                        help="leave out rtcGammaToSigmaFactor -- sigma0 is "
                             "then unavailable from the TIF")
    parser.add_argument("--no-cog", action="store_true",
                        help="write a plain tiled GeoTIFF instead of a Cloud "
                             "Optimized one -- no overview pyramid, and no "
                             "layout guarantee")
    parser.add_argument("--list", action="store_true",
                        help="print what the file holds and stop")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        import h5py  # noqa: F401
    except ImportError:
        print("DPQED_gcov2tif needs h5py to read the grid's corner and "
              "projection,\nwhich the HDF5 subdatasets do not carry.\n"
              "  pip install h5py", file=sys.stderr)
        return 2
    if _pick_writer() is None:
        print("DPQED_gcov2tif needs GDAL or rasterio to write a GeoTIFF.",
              file=sys.stderr)
        return 2

    try:
        if args.list:
            for line in describe(args.input):
                print(line)
            return 0
        convert(args.input, args.output, args.band, args.frequency,
                not args.no_factor, not args.no_cog, not args.quiet)
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f"DPQED_gcov2tif: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
