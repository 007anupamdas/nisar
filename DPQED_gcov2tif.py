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
Set INPUT (and OUTPUT, if you want it somewhere particular) at the top of this
file and run it, the way DPQED_h52tif.py is run. There are no options, because
there is nothing to choose that the product does not already say:

    band        from the granule name -- 'NISAR_L2_...' is L-band, 'NISAR_S2_'
                is S -- and from the file itself when the name does not say, or
                says something the file does not hold.
    frequency   whichever the product carries; A when it carries both, since
                that is the wideband channel.
    terms       the diagonal ones that are there.
    extras      the RTC factor and the incidence cube, when the product has
                them.

Asking a person to repeat any of that is asking them to get it wrong.

Needs h5py and numpy, and either GDAL or rasterio to write -- whichever the
environment that ran DPQED_h52tif.py already has.
"""

import os
import re
import sys

import numpy as np

# ── WHAT TO CONVERT ───────────────────────────────────────────────────────────
# Set these two and run the file. Everything below is read out of the product.
INPUT = r"V:\ICIGDev\GPUPOC\input\dqe\inp\NISAR_L2_PR_GCOV.h5"
OUTPUT = None       # None: '<input>_gcov.tif', written beside the product


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
INCIDENCE_BAND_KEY = "INCIDENCE_ANGLE_BAND"
INCIDENCE_HEIGHT_KEY = "INCIDENCE_ANGLE_HEIGHT_M"


# ── BEGIN SHARED GCOV HELPERS ─────────────────────────────────────────────────
# Character-identical to the same block in DPQED_radial.py. Two files rather
# than an import because RADIAL is exec'd as a single file in the QGIS console,
# where a sibling import is not reliably on the path; tests_radial_stats.py
# asserts the two copies have not drifted.

GCOV_POL_TERMS = ("HHHH", "HVHV", "VHVH", "VVVV", "RHRH", "RVRV")
GCOV_GRID_RE = re.compile(
    r"science/(?P<band>[LS]SAR)/GCOV/(?:grids/)?frequency(?P<freq>[A-Z])/"
    r"(?P<term>[A-Z]{4})$")
RTC_FACTOR_RE = re.compile(r"(?i)gamma.?to.?sigma")

# The geometry cubes, which sit under metadata rather than with the grids and
# are sampled on their own coarse grid at several heights above the ellipsoid.
RADAR_GRID_RE = re.compile(
    r"science/(?P<band>[LS]SAR)/GCOV/metadata/radarGrid/(?P<name>[A-Za-z0-9_]+)$")
INCIDENCE_NAME = "incidenceAngle"

# Which height layer of a cube to take. Without a DEM there is nothing to
# choose one with, so the layer nearest the ellipsoid is used and the height
# actually taken is written into the TIFF's header rather than left implicit.
INCIDENCE_HEIGHT_M = 0.0


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


def gcov_group_of(dataset_name):
    """The grid group one term's path sits in, whatever layout the product uses.

    Taken from the match rather than rebuilt from the band and frequency,
    because the 'grids/' segment is not there in every product and a rebuilt
    path is a guess that fails as a missing dataset three steps later.
    """
    text = str(dataset_name).strip()
    match = GCOV_GRID_RE.search(text)
    if not match:
        return None
    return text[:match.start("term")].rstrip("/")


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


# The band a granule name declares, two ways. Some names carry the tag whole --
# 'NISAR_LSAR_...' -- and the real ones do not: they put it in the second field
# as a letter beside the processing level, 'NISAR_L2_PR_GCOV_...' for L-band
# and 'NISAR_S2_...' for S. RIVAL's band_from_name only knows the first
# spelling, which is why it falls back to the sidecar's Sensor field and the
# centre frequency; here the file itself is the fallback, and a better one.
# No word boundary: the tag is nearly always spelled between underscores, and
# \b does not match between '_' and 'L'. A bare substring is safe here because
# 'NISAR' contains neither LSAR nor SSAR, and only the basename is searched.
BAND_TAG_RE = re.compile(r"(?i)([LS])SAR")
BAND_FIELD_RE = re.compile(r"(?i)\bNISAR[_-]([LS])\d")


def band_from_filename(path):
    """LSAR or SSAR as the file's name declares it, or None.

    A preference, not a verdict: a name is metadata someone typed, and what the
    product actually holds wins when the two disagree.
    """
    name = os.path.basename(str(path or ""))
    for pattern in (BAND_TAG_RE, BAND_FIELD_RE):
        match = pattern.search(name)
        if match:
            return match.group(1).upper() + "SAR"
    return None


def _axis_weights(axis, wanted):
    """Bracketing index and fraction along one ascending axis, per wanted value.

    Values off either end come back flagged rather than clamped: an incidence
    angle taken from beyond the cube is an extrapolation, and a plausible
    looking one, which is worse than a gap that says so.
    """
    index = np.clip(np.searchsorted(axis, wanted, side="right") - 1,
                    0, axis.size - 2)
    lower, upper = axis[index], axis[index + 1]
    span = np.where(upper > lower, upper - lower, 1.0)
    frac = np.clip((wanted - lower) / span, 0.0, 1.0)
    inside = (wanted >= axis[0]) & (wanted <= axis[-1])
    return index, frac, inside


def bilinear_grid(x_src, y_src, values, x_dst, y_dst):
    """`values` resampled onto the destination coordinate vectors.

    Bilinear, and vectorized over the whole destination block: a geometry cube
    is a few hundred samples across and the image grid is tens of thousands, so
    a per-pixel loop in Python would take longer than every other part of a
    conversion put together.

    Either set of coordinates may descend -- a north-up grid's y does -- so
    both are put the same way round first, taking the values with them.
    """
    x = np.asarray(x_src, dtype=float).ravel()
    y = np.asarray(y_src, dtype=float).ravel()
    grid = np.asarray(values, dtype=float)
    if x.size < 2 or y.size < 2 or grid.shape != (y.size, x.size):
        raise ValueError(f"cube is {grid.shape}, not {(y.size, x.size)}")
    if x[-1] < x[0]:
        x, grid = x[::-1], grid[:, ::-1]
    if y[-1] < y[0]:
        y, grid = y[::-1], grid[::-1, :]
    xi, fx, x_in = _axis_weights(x, np.asarray(x_dst, dtype=float).ravel())
    yi, fy, y_in = _axis_weights(y, np.asarray(y_dst, dtype=float).ravel())
    g00 = grid[np.ix_(yi, xi)]
    g01 = grid[np.ix_(yi, xi + 1)]
    g10 = grid[np.ix_(yi + 1, xi)]
    g11 = grid[np.ix_(yi + 1, xi + 1)]
    fx_, fy_ = fx[None, :], fy[:, None]
    out = (g00 * (1 - fx_) * (1 - fy_) + g01 * fx_ * (1 - fy_)
           + g10 * (1 - fx_) * fy_ + g11 * fx_ * fy_)
    return np.where(y_in[:, None] & x_in[None, :], out, np.nan)


def radar_grid_datasets(paths, band=None):
    """{name: path} for a band's radarGrid metadata cubes."""
    found = {}
    for path in paths:
        match = RADAR_GRID_RE.search(str(path).strip())
        if match and (band is None or match.group("band") == band):
            found[match.group("name")] = str(path).strip()
    return found


def height_layer(cube, heights=None, at_height=INCIDENCE_HEIGHT_M):
    """One (y, x) layer of a cube, and the height it was taken at.

    A radarGrid cube is sampled at several heights above the ellipsoid, because
    where a point images from depends on how high it is. Choosing between them
    properly needs a DEM; this has none, so it takes the layer nearest
    `at_height` and returns which one it was, for the header to record. A cube
    that is already two-dimensional is returned as it is.
    """
    cube = np.asarray(cube)
    if cube.ndim == 2:
        return cube, None
    if cube.ndim != 3:
        raise ValueError(f"a radarGrid cube is 2- or 3-D, not {cube.ndim}-D")
    if heights is None:
        return cube[0], None
    heights = np.asarray(heights, dtype=float).ravel()
    index = int(np.argmin(np.abs(heights - float(at_height))))
    return cube[index], float(heights[index])


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
            group = gcov_group_of(next(iter(grids[key].values())))
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
            with_factor=True, with_incidence=True, cog=True, verbose=True):
    """Write one GCOV grid to a GeoTIFF. Returns the path written."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        paths = []
        handle.visit(paths.append)
        all_grids = gcov_grids(paths)
        wanted = band or band_from_filename(h5_path)
        present = sorted({key[0] for key in all_grids})
        if wanted and wanted not in present:
            # Said rather than silently overridden: a name and its contents
            # disagreeing is worth knowing about, and the contents are right.
            if verbose:
                print(f"[GCOV] the name says {wanted}, the file holds "
                      f"{', '.join(present)}; taking what is in the file")
            wanted = None
        elif wanted and verbose:
            print(f"[GCOV] {wanted}, from the file name")
        key = choose_grid(all_grids, wanted, frequency)
        group = gcov_group_of(next(iter(all_grids[key].values())))
        members = list(handle[group])
        names, factor_band = plan_bands(members, with_factor)

        first = handle[f"{group}/{names[0]}"]
        height, width = int(first.shape[0]), int(first.shape[1])
        x_grid = np.asarray(handle[f"{group}/xCoordinates"][()], dtype=float)
        y_grid = np.asarray(handle[f"{group}/yCoordinates"][()], dtype=float)
        geotransform = geotransform_from_coords(x_grid, y_grid)
        epsg = int(np.asarray(handle[f"{group}/projection"][()]).ravel()[0])

        for name in names:
            shape = handle[f"{group}/{name}"].shape
            if tuple(shape[:2]) != (height, width):
                raise ValueError(
                    f"{name} is {shape}, not {(height, width)}: the bands of "
                    "one GeoTIFF have to be on one grid")

        # Each band is a name and a way to produce its rows, so a band that is
        # computed rather than copied -- the incidence angle, resampled from a
        # cube on another grid entirely -- is written by the same loop.
        layers = [(name, _dataset_reader(handle[f"{group}/{name}"]))
                  for name in names]
        incidence_band = incidence_height = None
        if with_incidence:
            cubes = radar_grid_datasets(paths, key[0])
            if INCIDENCE_NAME in cubes:
                layer, incidence_height = height_layer(
                    handle[cubes[INCIDENCE_NAME]][()],
                    handle[cubes["heightAboveEllipsoid"]][()]
                    if "heightAboveEllipsoid" in cubes else None)
                layers.append((INCIDENCE_NAME, _cube_reader(
                    handle[cubes["xCoordinates"]][()],
                    handle[cubes["yCoordinates"]][()], layer, x_grid, y_grid)))
                incidence_band = len(layers)
            elif verbose:
                print("[GCOV] no incidenceAngle cube under metadata/radarGrid: "
                      "no incidence column from this TIF")

        if out_path is None:
            out_path = os.path.splitext(h5_path)[0] + "_gcov.tif"
        metadata = {
            "NISAR_PRODUCT": os.path.basename(h5_path),
            "NISAR_BAND": key[0],
            "NISAR_FREQUENCY": key[1],
            "BACKSCATTER": "gamma0",
            "GCOV2TIF_BANDS": ",".join(name for name, _ in layers),
        }
        if factor_band:
            metadata[RTC_FACTOR_BAND_KEY] = str(factor_band)
        if incidence_band:
            metadata[INCIDENCE_BAND_KEY] = str(incidence_band)
            if incidence_height is not None:
                metadata[INCIDENCE_HEIGHT_KEY] = f"{incidence_height:.1f}"

        if verbose:
            print(f"[GCOV] {key[0]} frequency{key[1]}: {width} x {height}, "
                  f"EPSG:{epsg}")
            notes = {factor_band: "  <- RTC gamma-to-sigma factor",
                     incidence_band: "  <- incidence angle, degrees"}
            for number, (name, _) in enumerate(layers, start=1):
                print(f"[GCOV]   band {number}: {name}{notes.get(number, '')}")
            if incidence_height is not None:
                print(f"[GCOV]   incidence taken at "
                      f"{incidence_height:.1f} m above the ellipsoid")
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
            writer(staged, layers, width, height, geotransform,
                   epsg, metadata, verbose)
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


def _dataset_reader(dataset):
    """Rows of an HDF5 dataset, as the writer asks for them."""
    return lambda row0, rows: np.asarray(dataset[row0:row0 + rows, :],
                                         dtype=np.float32)


def _cube_reader(x_src, y_src, layer, x_grid, y_grid):
    """Rows of a geometry cube resampled onto the image grid.

    Resampled a block at a time rather than whole: the destination is the full
    image grid, and materialising an incidence angle for twenty thousand rows
    at once costs as much memory as a covariance term.
    """
    def read(row0, rows):
        return bilinear_grid(x_src, y_src, layer, x_grid,
                             y_grid[row0:row0 + rows]).astype(np.float32)
    return read


def _blocks(height, rows=BLOCK_ROWS):
    for row0 in range(0, height, rows):
        yield row0, min(rows, height - row0)


def _gdal_writer(out_path, layers, width, height, geotransform,
                 epsg, metadata, verbose=True):
    from osgeo import gdal, osr
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, width, height, len(layers), gdal.GDT_Float32,
                       options=list(TIFF_OPTIONS))
    if ds is None:
        raise RuntimeError(f"GDAL could not create {out_path}")
    try:
        ds.SetGeoTransform(list(geotransform))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        ds.SetProjection(srs.ExportToWkt())
        ds.SetMetadata({k: str(v) for k, v in metadata.items()})
        for number, (name, read) in enumerate(layers, start=1):
            band = ds.GetRasterBand(number)
            band.SetDescription(name)       # what RADIAL reads the column from
            band.SetNoDataValue(float("nan"))
            for row0, rows in _blocks(height):
                band.WriteArray(read(row0, rows), 0, row0)
            band.FlushCache()
    finally:
        ds = None


def _rasterio_writer(out_path, layers, width, height,
                     geotransform, epsg, metadata, verbose=True):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.windows import Window
    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "count": len(layers), "dtype": "float32",
        "crs": CRS.from_epsg(int(epsg)),
        "transform": Affine.from_gdal(*geotransform),
        "nodata": float("nan"), "tiled": True, "blockxsize": 256,
        "blockysize": 256, "compress": "DEFLATE", "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for number, (name, read) in enumerate(layers, start=1):
            for row0, rows in _blocks(height):
                dst.write(read(row0, rows), number,
                          window=Window(0, row0, width, rows))
        dst.descriptions = tuple(name for name, _ in layers)
        dst.update_tags(**{k: str(v) for k, v in metadata.items()})


def main(input_path=None, output_path=None):
    """Convert INPUT to OUTPUT. Set them above and run the file.

    There is nothing else to pass. Which band, which frequency, which terms,
    whether the product carries an RTC factor or an incidence cube -- all of it
    is in the file or in its name, and asking a person to repeat it is asking
    them to get it wrong.
    """
    h5_path = input_path or INPUT
    out_path = output_path or OUTPUT
    if not h5_path or not os.path.exists(h5_path):
        print(f"DPQED_gcov2tif: set INPUT at the top of this file to a NISAR "
              f"GCOV '.h5'.\n  INPUT is currently {h5_path!r}", file=sys.stderr)
        return 2
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
        for line in describe(h5_path):
            print(f"[GCOV] {line}")
        convert(h5_path, out_path)
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f"DPQED_gcov2tif: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
