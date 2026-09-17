"""Round-trip a synthetic NISAR GCOV through DPQED_gcov2tif and read it back.

Real h5py in, real GeoTIFF out, read with rasterio: the geotransform, the CRS,
the band names and the pixels are checked against what went in, rather than
against a mock that would agree with whatever the code did.

    python3 tests_gcov2tif.py

The GDAL writer is the one path this cannot run for real -- the bindings
available here are built for another Python -- so it is driven against a
recording stub, which still catches a wrong call, a wrong argument order or a
block written to the wrong row. The rasterio path is exercised end to end.
"""
import os
import sys
import tempfile
import types

import numpy as np
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import h5py
import rasterio

import DPQED_gcov2tif as C

failures = []


def check(label, got, want):
    ok_ = got == want
    print(f"{'PASS' if ok_ else 'FAIL'}  {label}: {got!r}")
    if not ok_:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}"
          + (f": {detail}" if detail else ""))
    if not condition:
        failures.append(f"{label}: {detail}")


# ── a synthetic GCOV ──────────────────────────────────────────────────────────
# Taller than one write block, so the block loop really loops: a converter that
# wrote only the first block would otherwise pass every check below.
HEIGHT, WIDTH = C.BLOCK_ROWS + 176, 64
EPSG, STEP = 32644, 30.0
X0, Y0 = 500015.0, 4000985.0       # pixel CENTRES, as a grid states them

rng = np.random.default_rng(11)
HHHH = rng.exponential(0.05, (HEIGHT, WIDTH)).astype(np.float32)
HVHV = rng.exponential(0.01, (HEIGHT, WIDTH)).astype(np.float32)
FACTOR = rng.uniform(0.6, 1.8, (HEIGHT, WIDTH)).astype(np.float32)


# The geometry cube: its own coarse grid, three heights, and a plane in x and
# y so an interpolation error shows up as a number rather than as noise.
CUBE_NX, CUBE_NY = 9, 7
CUBE_X = X0 - STEP + np.linspace(0, (WIDTH + 1) * STEP, CUBE_NX)
CUBE_Y = Y0 + STEP - np.linspace(0, (HEIGHT + 1) * STEP, CUBE_NY)
CUBE_HEIGHTS = np.array([-500.0, 0.0, 3000.0])


def incidence_at(x, y):
    """The plane the fixture's cube samples, to check interpolation against."""
    return 34.0 + 3e-5 * (np.asarray(x) - X0) + 1e-5 * (Y0 - np.asarray(y))


def write_h5(path, grids=(("LSAR", "A"),), with_factor=True,
             off_diagonal=True, product="GCOV", shapes=None,
             with_cube=True, grids_segment=True):
    with h5py.File(path, "w") as f:
        if with_cube:
            # radarGrid is per band, not per frequency: a file with both A and
            # B has one cube group, and writing it twice is an h5py error.
            for band in dict.fromkeys(band for band, _ in grids):
                meta = f.require_group(
                    f"science/{band}/{product}/metadata/radarGrid")
                plane = incidence_at(CUBE_X[None, :], CUBE_Y[:, None])
                # Each height layer offset a little, so taking the wrong one
                # is visible rather than harmless.
                meta["incidenceAngle"] = np.stack(
                    [plane + offset for offset in (-0.4, 0.0, 0.9)])
                meta["heightAboveEllipsoid"] = CUBE_HEIGHTS
                meta["xCoordinates"] = CUBE_X
                meta["yCoordinates"] = CUBE_Y
        for band, freq in grids:
            segment = "grids/" if grids_segment else ""
            group = f.require_group(
                f"science/{band}/{product}/{segment}frequency{freq}")
            shape = (shapes or {}).get((band, freq))
            group["HHHH"] = HHHH if shape is None else np.zeros(shape, np.float32)
            group["HVHV"] = HVHV
            if off_diagonal:
                group["HHHV"] = np.zeros((HEIGHT, WIDTH), np.complex64)
            if with_factor:
                group["rtcGammaToSigmaFactor"] = FACTOR
            group["xCoordinates"] = X0 + STEP * np.arange(WIDTH)
            group["yCoordinates"] = Y0 - STEP * np.arange(HEIGHT)
            group["projection"] = np.array(EPSG)
    return path


tmp = tempfile.mkdtemp()
h5_path = write_h5(os.path.join(tmp, "NISAR_L2_GCOV.h5"))

# ── 1. what the file holds ────────────────────────────────────────────────────
print("\n── --list ──")
lines = "\n".join(C.describe(h5_path))
ok("the grid is listed", "LSAR frequencyA" in lines, lines)
ok("with its diagonal terms", "HHHH, HVHV" in lines, lines)
ok("and its RTC factor", "rtcGammaToSigmaFactor" in lines, lines)

# ── 2. which bands get written ────────────────────────────────────────────────
print("\n── planning the bands ──")
check("diagonal terms, then the factor last",
      C.plan_bands(["HHHH", "HVHV", "HHHV", "rtcGammaToSigmaFactor"]),
      (["HHHH", "HVHV", "rtcGammaToSigmaFactor"], 3))
check("the factor can be left out",
      C.plan_bands(["HHHH", "rtcGammaToSigmaFactor"], with_factor=False),
      (["HHHH"], None))
check("a product without one", C.plan_bands(["HHHH", "VVVV"]),
      (["HHHH", "VVVV"], None))
# The polarization bands keep their numbers whether or not the factor is there,
# so a script written against a factor-less TIF still reads the right band.
check("the factor never displaces a polarization",
      C.plan_bands(["HHHH", "HVHV", "rtcGammaToSigmaFactor"])[0][:2],
      C.plan_bands(["HHHH", "HVHV"], with_factor=False)[0][:2])
try:
    C.plan_bands(["HHHV", "rtcGammaToSigmaFactor"])
    check("a grid with no backscatter is refused", "accepted", "rejected")
except ValueError as e:
    ok("a grid with no backscatter is refused", "diagonal" in str(e), str(e))

# ── 3. the real conversion ────────────────────────────────────────────────────
print("\n── converting, for real ──")
tif_path = C.convert(h5_path, verbose=False)
check("named after the product", os.path.basename(tif_path),
      "NISAR_L2_GCOV_gcov.tif")

with rasterio.open(tif_path) as ds:
    check("a band per diagonal term, plus the factor and the incidence angle",
          ds.count, 4)
    check("and the complex term is not among them",
          list(ds.descriptions),
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"])
    check("size", (ds.width, ds.height), (WIDTH, HEIGHT))
    check("projection", ds.crs.to_epsg(), EPSG)
    # The geotransform is anchored half a pixel back from the first centre.
    check("georeferenced from the coordinate vectors",
          [ds.transform.c, ds.transform.a, ds.transform.f, ds.transform.e],
          [X0 - STEP / 2.0, STEP, Y0 + STEP / 2.0, -STEP])
    ok("nodata is NaN", ds.nodata is None or np.isnan(ds.nodata),
       repr(ds.nodata))
    tags = ds.tags()
    check("the header declares the factor's band",
          tags.get(C.RTC_FACTOR_BAND_KEY), "3")
    check("and that the pixels are gamma0", tags.get("BACKSCATTER"), "gamma0")
    check("provenance is recorded",
          (tags.get("NISAR_BAND"), tags.get("NISAR_FREQUENCY")), ("LSAR", "A"))

    # Every block, not just the first: the last row has to be the last row.
    written = ds.read(1)
    ok("the pixels survive the round trip",
       np.allclose(written, HHHH, rtol=1e-6), f"max diff "
       f"{np.abs(written - HHHH).max():.3g}")
    ok("including past the first write block",
       np.allclose(ds.read(1)[C.BLOCK_ROWS:], HHHH[C.BLOCK_ROWS:], rtol=1e-6))
    ok("and the factor is the factor",
       np.allclose(ds.read(3), FACTOR, rtol=1e-6))
    ok("the bands are tiled, so QGIS reads windows", ds.profile.get("tiled"))

# ── 4. what RADIAL makes of it ────────────────────────────────────────────────
# The point of naming the bands: RADIAL reads the polarization out of the name
# and finds the factor by it. This is the same function RADIAL uses.
print("\n── read back the way RADIAL reads it ──")
import ast


def radial_helpers():
    """DPQED_radial's pure slice, without importing Qt."""
    import math
    import re
    text = open(os.path.join(HERE, "DPQED_radial.py"), encoding="utf-8").read()
    safe = {"tuple": tuple, "sorted": sorted, "re": re, "math": math}
    consts = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        consts[target.id] = eval(compile(
                            ast.Expression(node.value), "radial", "eval"),
                            dict(safe), dict(consts))
                    except Exception:
                        pass
    body = text[text.index("# ── BEGIN PURE HELPERS"):
                text.index("# ── END PURE HELPERS")]
    ns = dict(consts, np=np, math=math, re=re, os=os, print=print)
    exec(compile(body, "radial", "exec"), ns)
    return ns


R = radial_helpers()
with rasterio.open(tif_path) as ds:
    # QGIS labels a described band '<n>: <name>'.
    labels = [f"{i}: {name}" for i, name in enumerate(ds.descriptions, start=1)]
check("RADIAL finds the factor by name", R["factor_band_index"](labels), 3)
check("and names the columns for the polarizations",
      [R["band_prefix"](label, i) for i, label in enumerate(labels[:2], start=1)],
      ["HH", "HV"])
# A TIF written by DPQED_h52tif.py names nothing, which is the case the header
# item exists for.
check("an unnamed TIF yields no factor by name",
      R["factor_band_index"](["Band 1", "Band 2", "Band 3"]), None)
check("and its columns fall back to band numbers",
      [R["band_prefix"](l, i) for i, l in enumerate(["Band 1", "Band 2"], 1)],
      ["b1", "b2"])

# ── 4b. it is a COG, and the naming survived being laid out as one ───────────
# The COG driver copies the file: band descriptions, nodata and the metadata
# that says which band is the factor all have to come through it, because
# everything downstream reads the file by exactly those.
print("\n── Cloud Optimized ──")
with rasterio.open(tif_path) as ds:
    check("GDAL says the layout is COG",
          ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"), "COG")
    ok("tiled", ds.profile.get("tiled"), str(ds.profile.get("blockxsize")))
    ok("with an overview pyramid", len(ds.overviews(1)) > 0,
       str(ds.overviews(1)))
    check("every band has one", [len(ds.overviews(b)) for b in ds.indexes],
          [len(ds.overviews(1))] * ds.count)
    check("band names survived the layout copy", list(ds.descriptions),
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"])
    check("and so did the factor header",
          ds.tags().get(C.RTC_FACTOR_BAND_KEY), "3")
    ok("and the nodata", ds.nodata is None or np.isnan(ds.nodata))
    # An overview of power is an average of power, which is the backscatter of
    # the block -- so the top of the pyramid should sit near the scene mean.
    top = ds.read(1, out_shape=(1, ds.height // 16, ds.width // 16))
    ok("overviews average rather than subsample",
       abs(float(np.nanmean(top)) - float(HHHH.mean())) < 0.01,
       f"{float(np.nanmean(top)):.4f} vs {float(HHHH.mean()):.4f}")
ok("no staging file was left behind",
   not os.path.exists(tif_path + ".building.tif"))

plain_tif = C.convert(h5_path, os.path.join(tmp, "plain.tif"), cog=False,
                      verbose=False)
with rasterio.open(plain_tif) as ds:
    ok("--no-cog writes a plain GeoTIFF",
       ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG",
       str(ds.tags(ns="IMAGE_STRUCTURE")))
    check("still tiled and still named", (bool(ds.profile.get("tiled")),
                                          list(ds.descriptions)),
          (True, ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"]))

# ── 4c. the incidence angle band ─────────────────────────────────────────────
# The cube is on its own coarse grid at three heights; the band has to come out
# on the image grid, from the layer nearest the ellipsoid, and match the plane
# the fixture sampled.
print("\n── incidence angle ──")
with rasterio.open(tif_path) as ds:
    check("it is a band, after the terms and the factor",
          list(ds.descriptions),
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"])
    tags = ds.tags()
    check("the header says which band", tags.get(C.INCIDENCE_BAND_KEY), "4")
    check("and at what height it was taken",
          tags.get(C.INCIDENCE_HEIGHT_KEY), "0.0")
    angle = ds.read(4)
    # The image grid's own pixel centres, to evaluate the plane at.
    cols = np.arange(ds.width)
    rows = np.arange(ds.height)
    xs = ds.transform.c + ds.transform.a * (cols + 0.5)
    ys = ds.transform.f + ds.transform.e * (rows + 0.5)
    want = incidence_at(xs[None, :], ys[:, None])
    ok("resampled onto the image grid, to the plane it sampled",
       np.nanmax(np.abs(angle - want)) < 1e-3,
       f"max err {float(np.nanmax(np.abs(angle - want))):.2e} deg")
    ok("the layer nearest the ellipsoid, not the first one",
       abs(float(np.nanmean(angle - want))) < 1e-3,
       f"offset {float(np.nanmean(angle - want)):+.3f} deg "
       f"(-0.4 would be the first layer)")
    ok("and it is not constant -- a cube read as one value would pass "
       "everything else", float(np.nanmax(angle) - np.nanmin(angle)) > 0.05,
       f"spans {float(np.nanmax(angle) - np.nanmin(angle)):.3f} deg")

no_inc = C.convert(h5_path, os.path.join(tmp, "noinc.tif"),
                   with_incidence=False, verbose=False)
with rasterio.open(no_inc) as ds:
    check("--no-incidence leaves it out", list(ds.descriptions),
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor"])
    ok("and declares nothing", C.INCIDENCE_BAND_KEY not in ds.tags())

bare = write_h5(os.path.join(tmp, "nocube.h5"), with_cube=False)
bare_tif = C.convert(bare, os.path.join(tmp, "nocube.tif"), verbose=False)
with rasterio.open(bare_tif) as ds:
    check("a product with no cube simply has no incidence band",
          list(ds.descriptions), ["HHHH", "HVHV", "rtcGammaToSigmaFactor"])

# ── 4d. the layout without 'grids/' ──────────────────────────────────────────
# Products differ over whether the frequency groups sit under 'grids/'. The
# group is taken from where a term was found, so both work; this is the one
# that was assumed and got wrong.
print("\n── the other group layout ──")
flat = write_h5(os.path.join(tmp, "flat.h5"), grids_segment=False)
flat_tif = C.convert(flat, os.path.join(tmp, "flat.tif"), verbose=False)
with rasterio.open(flat_tif) as ds:
    check("GCOV/frequencyA, with no grids/ segment", list(ds.descriptions),
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"])
    check("georeferenced the same way", ds.transform.c, X0 - STEP / 2.0)
check("and the group is derived, not rebuilt",
      C.gcov_group_of("science/LSAR/GCOV/frequencyA/HHHH"),
      "science/LSAR/GCOV/frequencyA")
check("either layout parses",
      sorted(C.gcov_grids(["science/LSAR/GCOV/grids/frequencyA/HHHH",
                           "science/SSAR/GCOV/frequencyB/VVVV"])),
      [("LSAR", "A"), ("SSAR", "B")])

# ── 5. --no-factor, and grid selection ────────────────────────────────────────
print("\n── options ──")
plain = C.convert(h5_path, os.path.join(tmp, "nofactor.tif"),
                  with_factor=False, verbose=False)
with rasterio.open(plain) as ds:
    check("terms only, the incidence angle being a separate question",
          list(ds.descriptions), ["HHHH", "HVHV", "incidenceAngle"])
    ok("and no factor is declared", C.RTC_FACTOR_BAND_KEY not in ds.tags())

two = write_h5(os.path.join(tmp, "two.h5"),
               grids=(("LSAR", "A"), ("LSAR", "B")))
check("frequency A wins by default",
      C.choose_grid(C.gcov_grids([
          "science/LSAR/GCOV/grids/frequencyB/HHHH",
          "science/LSAR/GCOV/grids/frequencyA/HHHH"])), ("LSAR", "A"))
out_b = C.convert(two, os.path.join(tmp, "b.tif"), frequency="B", verbose=False)
with rasterio.open(out_b) as ds:
    check("and B is converted when asked for", ds.count, 4)
try:
    C.choose_grid(C.gcov_grids(["science/LSAR/GCOV/grids/frequencyA/HHHH"]),
                  frequency="B")
    check("a frequency that is not there is refused", "accepted", "rejected")
except ValueError as e:
    ok("a frequency that is not there is refused, and says what is",
       "frequencyA" in str(e), str(e))

# ── 6. what is not a GCOV ─────────────────────────────────────────────────────
print("\n── refusals ──")
gslc = write_h5(os.path.join(tmp, "gslc.h5"), product="GSLC")
try:
    C.convert(gslc, os.path.join(tmp, "gslc.tif"), verbose=False)
    check("a GSLC is refused", "accepted", "rejected")
except ValueError as e:
    ok("a GSLC is refused", "GCOV" in str(e), str(e))

ragged = write_h5(os.path.join(tmp, "ragged.h5"),
                  shapes={("LSAR", "A"): (HEIGHT - 3, WIDTH)})
try:
    C.convert(ragged, os.path.join(tmp, "ragged.tif"), verbose=False)
    check("bands on different grids are refused", "accepted", "rejected")
except ValueError as e:
    ok("bands on different grids are refused", "one grid" in str(e), str(e))

# ── 7. the GDAL writer, against a recording stub ──────────────────────────────
print("\n── the GDAL path ──")
calls = {"bands": {}, "blocks": [], "meta": None, "gt": None,
         "create": None, "cog": None}


def install_gdal_stub():
    def make_band(number):
        band = MagicMock()
        band.SetDescription = lambda d: calls["bands"].setdefault(
            number, {}).__setitem__("description", d)
        band.SetNoDataValue = lambda v: calls["bands"].setdefault(
            number, {}).__setitem__("nodata", v)
        band.WriteArray = lambda array, x, y: calls["blocks"].append(
            (number, x, y, array.shape, str(array.dtype)))
        return band

    made = {}
    ds = MagicMock()
    ds.SetGeoTransform = lambda gt: calls.__setitem__("gt", list(gt))
    ds.SetMetadata = lambda m: calls.__setitem__("meta", dict(m))
    ds.GetRasterBand = lambda n: made.setdefault(n, make_band(n))
    driver = MagicMock()

    def create(path, w, h, count, dtype, options=None):
        calls["create"] = (path, w, h, count, dtype, list(options or []))
        return ds
    driver.Create = create

    def create_copy(path, src, options=None):
        calls["cog"] = (path, list(options or []))
        return MagicMock()
    cog_driver = MagicMock()
    cog_driver.CreateCopy = create_copy
    gdal = types.ModuleType("osgeo.gdal")
    gdal.GDT_Float32 = "Float32"
    gdal.GA_ReadOnly = 0
    gdal.Open = lambda path, mode=0: MagicMock(_path=path)
    gdal.GetDriverByName = lambda name: cog_driver if name == "COG" else driver
    osr = types.ModuleType("osgeo.osr")
    osr.SpatialReference = lambda: MagicMock(
        ExportToWkt=lambda: "WKT", ImportFromEPSG=lambda e: 0)
    osgeo = types.ModuleType("osgeo")
    osgeo.gdal, osgeo.osr = gdal, osr
    sys.modules.update({"osgeo": osgeo, "osgeo.gdal": gdal, "osgeo.osr": osr})


install_gdal_stub()
gdal_out = os.path.join(tmp, "viagdal.tif")
C.convert(h5_path, gdal_out, verbose=False)
check("GDAL is preferred when it is there",
      calls["create"][0].startswith(gdal_out), True)
check("created at the grid's size, one band per layer",
      calls["create"][1:5], (WIDTH, HEIGHT, 4, "Float32"))
ok("tiled and compressed", "COMPRESS=DEFLATE" in calls["create"][5],
   str(calls["create"][5]))
check("georeferenced", calls["gt"],
      [X0 - STEP / 2.0, STEP, 0.0, Y0 + STEP / 2.0, 0.0, -STEP])
check("every band is described",
      [calls["bands"][n]["description"] for n in (1, 2, 3, 4)],
      ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"])
ok("every band gets NaN nodata",
   all(np.isnan(calls["bands"][n]["nodata"]) for n in (1, 2, 3, 4)))
check("the header declares the factor band",
      calls["meta"].get(C.RTC_FACTOR_BAND_KEY), "3")
# Blocks are written at an offset, and the offsets have to tile the raster
# exactly -- WriteArray takes (array, xoff, yoff), and swapping those writes
# the whole scene into a column.
band1 = [b for b in calls["blocks"] if b[0] == 1]
check("written in blocks", len(band1), 2)
check("at x=0, stepping down in y", [(b[1], b[2]) for b in band1],
      [(0, 0), (0, C.BLOCK_ROWS)])
check("covering every row exactly once",
      sum(b[3][0] for b in band1), HEIGHT)
check("full width, as float32",
      ({b[3][1] for b in band1}, {b[4] for b in band1}),
      ({WIDTH}, {"float32"}))
# The bands are written to a sibling and only then laid out as a COG, because
# the COG driver has no Create() -- and the staging file must not be what the
# caller is left holding.
check("the COG copy lands on the requested path", calls["cog"][0], gdal_out)
ok("with the COG options", "OVERVIEWS=AUTO" in calls["cog"][1]
   and "RESAMPLING=AVERAGE" in calls["cog"][1], str(calls["cog"][1]))
ok("the bands were staged beside it, not written there directly",
   calls["create"][0] == gdal_out + ".building.tif", calls["create"][0])

# ── 8. the helpers this file copies from DPQED_radial ─────────────────────────
# Three functions live in both, because RADIAL is exec'd as one file in the
# QGIS console and cannot import a sibling. Copies drift; this is what stops
# them.
print("\n── shared helpers have not drifted ──")


def function_source(path, name):
    text = open(path, encoding="utf-8").read()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


radial = os.path.join(HERE, "DPQED_radial.py")
for name in ("gcov_grids", "gcov_group_of", "gcov_diagonal_terms",
             "geotransform_from_coords"):
    check(f"{name} is the same in both files",
          function_source(__file__.replace("tests_gcov2tif.py",
                                           "DPQED_gcov2tif.py"), name)
          == function_source(radial, name), True)
check("and so are the constants they read",
      (C.GCOV_POL_TERMS, C.GCOV_GRID_RE.pattern, C.RTC_FACTOR_RE.pattern),
      (R["GCOV_POL_TERMS"], R["GCOV_GRID_RE"].pattern,
       R["RTC_FACTOR_RE"].pattern))
check("the header key matches the one RADIAL looks for",
      C.RTC_FACTOR_BAND_KEY, R["RTC_FACTOR_BAND_KEY"])

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
