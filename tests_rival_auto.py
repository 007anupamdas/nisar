"""Exercise RIVAL/AUTO's footprint geometry without QGIS.

Two halves. The first execs the pure-helper slice of DPQED_rival_auto.py -- so
what runs is the shipped code, not a copy -- and checks the geometry on its own.
The second opens real GeoTIFFs written here with rasterio, including one whose
valid region is the actual SSAR swath from the '.met' committed in this repo, and
measures what the mask hull recovers against what the bounding box would claim.
"""
import ast
import atexit
import math
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import rasterio
from affine import Affine
from rasterio import features
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "DPQED_rival_auto.py")
RIVAL = os.path.join(HERE, "DPQED_rival.py")

BEGIN = "# ── BEGIN PURE HELPERS"
END   = "# ── END PURE HELPERS"


def load_slice(path, extra=None):
    """The file's pure-helper slice, with its module-level constants."""
    text = open(path, encoding="utf-8").read()
    consts = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            try:
                consts[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    body = text[text.index(BEGIN):text.index(END)]
    ns = dict(consts, math=math, os=os, re=re, print=print)
    ns.update(extra or {})
    exec(compile(body, path, "exec"), ns)
    return ns


H = load_slice(SRC)
import json                                     # noqa: E402  (for RIVAL's slice)
import xml.etree.ElementTree as ET              # noqa: E402
R = load_slice(RIVAL, {"json": json, "ET": ET})

failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def close(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r} (want {want}±{tol})")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want}±{tol}")


# ── 1. affine and the pixel grid ──────────────────────────────────────────────
# rasterio's Affine iterates (a, b, c, d, e, f): x = a*col + b*row + c. North-up
# rasters carry a negative e, so row 0 is the TOP -- getting this backwards
# flips every footprint about its own centre without erroring.
T = (5.0, 0.0, 100000.0, 0.0, -5.0, 2000000.0)
check("origin corner", H["apply_affine"](T, 0, 0), (100000.0, 2000000.0))
check("one pixel right", H["apply_affine"](T, 1, 0), (100005.0, 2000000.0))
check("one pixel down is south", H["apply_affine"](T, 0, 1), (100000.0, 1999995.0))

ring = H["grid_corner_ring"](4, 3, T)
check("grid ring has four corners", len(ring), 4)
check("grid ring spans the whole grid",
      (ring[0], ring[2]), ((100000.0, 2000000.0), (100020.0, 1999985.0)))
check("grid ring area is width*height*pixel area",
      H["ring_area"](ring), 4 * 3 * 25.0)

# ── 2. convex hull ────────────────────────────────────────────────────────────
sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
check("hull of a square is the square", sorted(H["convex_hull"](sq)), sorted(sq))
check("interior points are dropped",
      sorted(H["convex_hull"](sq + [(0.5, 0.5), (0.3, 0.7)])), sorted(sq))
check("collinear points are dropped",
      sorted(H["convex_hull"]([(0, 0), (1, 0), (2, 0), (2, 2)])),
      sorted([(0, 0), (2, 0), (2, 2)]))
check("duplicates collapse", H["convex_hull"]([(0, 0), (0, 0)]), [(0, 0)])
check("two points come back as given",
      sorted(H["convex_hull"]([(1, 1), (0, 0)])), [(0, 0), (1, 1)])
check("empty stays empty", H["convex_hull"]([]), [])

# a hull is a cover: every input point is inside it
tri = H["convex_hull"]([(0, 0), (4, 0), (0, 4), (1, 1), (2, 1), (1, 2)])
check("hull area covers the inputs", H["ring_area"](tri), 8.0)

# ── 3. row spans and their corners ────────────────────────────────────────────
mask = [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0]]
check("spans skip empty rows", H["row_spans"](mask), [(1, 1, 2), (2, 0, 3)])
check("a hole inside a row does not split it",
      H["row_spans"]([[1, 0, 1]]), [(0, 0, 2)])
check("no valid pixels gives no spans", H["row_spans"]([[0, 0], [0, 0]]), [])

pts = H["span_corner_points"]([(0, 1, 2)], 1, 1)
check("a span contributes its four outer corners",
      sorted(pts), sorted([(1, 0), (3, 0), (3, 1), (1, 1)]))
pts4 = H["span_corner_points"]([(0, 1, 2)], 4, 4)
check("decimation scales the corners outward",
      sorted(pts4), sorted([(4, 0), (12, 0), (12, 4), (4, 4)]))

# ── 4. the hull is smaller than the box for a diagonal swath ──────────────────
# The whole point of the tool: a slanted swath in a north-up grid.
N = 40
diag = [[1 if abs((c - r)) <= 5 else 0 for c in range(N)] for r in range(N)]
unit = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
hull = H["mask_hull_ring"](diag, unit, 1, 1)
box  = H["grid_corner_ring"](N, N, unit)
check("a diagonal band gives a hull with at least four vertices",
      len(hull) >= 4, True)
over = H["over_coverage"](box, hull)
ok = 1.5 < over < 4.0
print(f"{'PASS' if ok else 'FAIL'}  diagonal swath: box claims {over:.2f}x the hull")
if not ok:
    failures.append(f"diagonal over-coverage {over}")
check("the hull is strictly inside the box",
      H["ring_area"](hull) < H["ring_area"](box), True)

check("an empty mask has no hull", H["mask_hull_ring"]([[0, 0], [0, 0]], unit, 1, 1),
      None)
full = [[1, 1], [1, 1]]
check("a full mask hulls to the whole grid",
      H["ring_area"](H["mask_hull_ring"](full, unit, 1, 1)), 4.0)

# ── 5. densify ────────────────────────────────────────────────────────────────
d = H["densify_ring"]([(0, 0), (10, 0), (10, 10), (0, 10)], per_edge=5)
check("densify gives per_edge points per edge", len(d), 20)
check("densify keeps the original vertices",
      all(v in d for v in [(0, 0), (10, 0), (10, 10), (0, 10)]), True)
check("densify closes the loop (last edge returns toward the start)",
      d[-1], (0.0, 2.0))
check("densify preserves area", H["ring_area"](d), 100.0)
check("per_edge below 2 is a no-op",
      H["densify_ring"]([(0, 0), (1, 1)], per_edge=1), [(0, 0), (1, 1)])

# ── 6. probe decimation ───────────────────────────────────────────────────────
check("a small raster is read whole", H["probe_decimation"](100, 80, 512), (100, 80))
check("a large raster is capped on its long side",
      H["probe_decimation"](20000, 10000, 512), (512, 256))
check("the long side can be the height",
      H["probe_decimation"](10000, 20000, 512), (256, 512))
check("never smaller than one pixel",
      H["probe_decimation"](10000, 3, 512), (512, 1))
check("a zero-sized raster gives zero", H["probe_decimation"](0, 10, 512), (0, 0))

# ── 7. real rasters ───────────────────────────────────────────────────────────
def load_upto(path, marker):
    """Everything above a banner, exec'd. Stops before the GUI half needs QGIS."""
    text = open(path, encoding="utf-8").read()
    assert marker in text, f"{path} has no '{marker}' banner"
    ns = {}
    exec(compile(text[:text.index(marker)], path, "exec"), ns)
    return ns


read_footprint = load_upto(SRC, "# ── THE DASHBOARD")["read_raster_footprint"]

TMP = tempfile.mkdtemp(prefix="rival_auto_")
atexit.register(shutil.rmtree, TMP, True)


def write_tif(name, data, nodata=None, crs="EPSG:32643",
              transform=Affine(5.0, 0, 100000.0, 0, -5.0, 2000000.0)):
    path = os.path.join(TMP, name)
    profile = {"driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
               "count": 1, "dtype": data.dtype.name, "transform": transform}
    if crs:
        profile["crs"] = CRS.from_string(crs)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


# every pixel valid -> the grid, and it says so
p = write_tif("full.tif", np.ones((50, 60), dtype="float32"))
rec = read_footprint(p)
check("a fully valid raster uses the pixel grid", rec["derived"], "extent")
check("  and reports why", rec["note"],
      "every pixel valid (no nodata declared)")
check("  area is width*height*pixel area",
      H["ring_area"](rec["ring_map"]), 50 * 60 * 25.0)

# a slanted swath with declared nodata -> the hull, materially smaller
sw = np.full((400, 400), np.nan, dtype="float32")
for r in range(400):
    lo, hi = max(0, r - 60), min(400, r + 60)
    sw[r, lo:hi] = 1.0
p = write_tif("swath.tif", sw, nodata=float("nan"))
rec = read_footprint(p)
check("a slanted swath uses the mask", rec["derived"], "mask")
ok = 1.5 < rec["over"] < 4.0
print(f"{'PASS' if ok else 'FAIL'}  swath: box claims {rec['over']:.2f}x the hull")
if not ok:
    failures.append(f"swath over-coverage {rec['over']}")
check("  the hull is inside the box",
      H["ring_area"](rec["ring_map"]) < H["ring_area"](rec["box_ring"]), True)

# all nodata -> falls back, and says so rather than inventing a footprint
p = write_tif("empty.tif", np.full((50, 50), np.nan, dtype="float32"),
              nodata=float("nan"))
rec = read_footprint(p)
check("an all-nodata raster falls back to the grid", rec["derived"], "extent")
check("  and says so", "no valid pixels" in (rec["note"] or ""), True)
check("  valid fraction is zero", rec["valid_fraction"], 0.0)

# no CRS -> refused with a reason, not placed at the equator
p = write_tif("nocrs.tif", np.ones((10, 10), dtype="float32"), crs=None)
try:
    read_footprint(p)
    check("a raster with no CRS is refused", "no exception", "ValueError")
except ValueError as e:
    check("a raster with no CRS is refused", "no CRS" in str(e), True)

# A degenerate transform is refused -- though not by the guard that names it.
# GDAL will not write a zero-scale geotransform, and drops the georeferencing
# altogether rather than writing a broken one, so the file comes back with no
# CRS at all and is refused one step earlier. Worth pinning as the real
# behaviour: either way it is a refusal with a reason, not a footprint placed
# at the equator. The zero-pixel guard still covers the case a driver keeps the
# CRS (a hand-written VRT will).
p = write_tif("degenerate.tif", np.ones((10, 10), dtype="float32"),
              transform=Affine(0.0, 0, 0.0, 0, -5.0, 0.0))
try:
    read_footprint(p)
    check("a zero pixel size is refused", "no exception", "ValueError")
except ValueError as e:
    check("a zero pixel size is refused, GDAL having dropped the CRS",
          "no CRS" in str(e), True)

# zeros with nodata unset: indistinguishable from data unless opted in
z = np.zeros((100, 100), dtype="float32")
z[40:60, :] = 1.0
p = write_tif("zerofill.tif", z)
rec = read_footprint(p)
check("undeclared zero fill is not guessed at", rec["derived"], "extent")
rec = read_footprint(p, zero_is_nodata=True)
check("TREAT_ZERO_AS_NODATA opts in", rec["derived"], "mask")
close("  and recovers the 20% band", rec["valid_fraction"], 0.20, 0.02)

# mask reading off entirely
rec = read_footprint(os.path.join(TMP, "swath.tif"), from_mask=False)
check("mask reading can be switched off", rec["derived"], "extent")
check("  and says so", rec["note"], "mask reading disabled")

# ── 8. the real SSAR swath from the committed '.met' ──────────────────────────
met = os.path.join(HERE, "NISAR_S2_PR_GSLC_028_084_A_010_3700_DHNA_A_"
                         "20260819T001733_20260819T001810_P00500_M_F_I_001.met")
meta = R["parse_meta_text"](open(met, encoding="utf-8", errors="replace").read(),
                            os.path.basename(met))
truth_ll = meta["ring"]
print(f"\n[REAL] SSAR swath from the .met: {len(truth_ll)} vertices, "
      f"{R['format_bounds'](R['rings_bounds']([truth_ll]))}")

# put it on the ground in its own UTM zone, then build a raster of exactly that
utm = CRS.from_epsg(32643)
xs, ys = warp_transform(CRS.from_epsg(4326), utm,
                        [p[0] for p in truth_ll], [p[1] for p in truth_ll])
truth_utm = list(zip(xs, ys))
minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)
PIX = 60.0                      # coarse: this is a geometry test, not a data one
w = int((maxx - minx) / PIX) + 2
h = int((maxy - miny) / PIX) + 2
tr = Affine(PIX, 0, minx - PIX, 0, -PIX, maxy + PIX)
poly = {"type": "Polygon", "coordinates": [truth_utm + [truth_utm[0]]]}
burn = features.rasterize([(poly, 1)], out_shape=(h, w), transform=tr,
                          fill=0, dtype="uint8")
data = np.where(burn == 1, 1.0, np.nan).astype("float32")
p = write_tif("ssar_swath.tif", data, nodata=float("nan"),
              crs="EPSG:32643", transform=tr)

rec = read_footprint(p)
check("the real swath is read from the mask", rec["derived"], "mask")

truth_area = H["ring_area"](truth_utm)
hull_area  = H["ring_area"](rec["ring_map"])
box_area   = H["ring_area"](rec["box_ring"])
print(f"[REAL] swath {truth_area / 1e6:.0f} km², "
      f"mask hull {hull_area / 1e6:.0f} km², "
      f"pixel grid {box_area / 1e6:.0f} km²")
print(f"[REAL] the hull recovers {hull_area / truth_area:.3f}x the true swath; "
      f"the grid would claim {box_area / truth_area:.2f}x")

ok = 0.98 <= hull_area / truth_area <= 1.06
print(f"{'PASS' if ok else 'FAIL'}  mask hull recovers the real swath "
      f"({hull_area / truth_area:.3f}x)")
if not ok:
    failures.append(f"hull/truth = {hull_area / truth_area}")

ok = box_area / truth_area > 1.25
print(f"{'PASS' if ok else 'FAIL'}  the pixel grid would over-claim materially "
      f"({box_area / truth_area:.2f}x)")
if not ok:
    failures.append(f"box/truth = {box_area / truth_area}")

# ── 9. the two backends must agree ────────────────────────────────────────────
# QGIS has GDAL and no rasterio; this container has rasterio and no GDAL. The
# GDAL path would therefore ship completely unexercised, and its one real trap
# is silent: GDAL and rasterio order the geotransform differently, so getting
# affine_from_gdal wrong misplaces every footprint without raising anything.
#
# So: stub osgeo.gdal over a *real* rasterio read of the same file. The stub
# takes its geotransform from rasterio's own to_gdal(), so what is being tested
# is this file's conversion back, against real data, rather than my arithmetic
# checked against my arithmetic.
check("GDAL order -> Affine order",
      H["affine_from_gdal"]((100000.0, 5.0, 0.0, 2000000.0, 0.0, -5.0)),
      (5.0, 0.0, 100000.0, 0.0, -5.0, 2000000.0))
check("a rotated geotransform keeps both rotation terms",
      H["affine_from_gdal"]((10.0, 1.0, 0.5, 20.0, 0.25, -1.0)),
      (1.0, 0.5, 10.0, 0.25, -1.0, 20.0))


class _StubBand:
    """A GDAL band whose reads are served by rasterio, over the same file."""

    def __init__(self, src):
        self.src = src

    def GetMaskFlags(self):
        from rasterio.enums import MaskFlags
        return 0x01 if list(self.src.mask_flag_enums[0]) == [MaskFlags.all_valid] \
            else 0x02

    def GetMaskBand(self):
        return _StubMaskBand(self.src)

    def ReadAsArray(self, x=0, y=0, w=None, h=None, buf_xsize=None,
                    buf_ysize=None):
        from rasterio.enums import Resampling
        return self.src.read(1, out_shape=(buf_ysize, buf_xsize),
                             resampling=Resampling.nearest)


class _StubMaskBand(_StubBand):
    def ReadAsArray(self, x=0, y=0, w=None, h=None, buf_xsize=None,
                    buf_ysize=None):
        from rasterio.enums import Resampling
        return self.src.read_masks(1, out_shape=(buf_ysize, buf_xsize),
                                   resampling=Resampling.nearest)


class _StubDataset:
    def __init__(self, path):
        self.src = rasterio.open(path)
        self.RasterXSize = self.src.width
        self.RasterYSize = self.src.height

    def GetProjection(self):
        return self.src.crs.to_wkt() if self.src.crs else ""

    def GetGeoTransform(self):
        return self.src.transform.to_gdal()     # rasterio's own conversion

    def GetRasterBand(self, i):
        return _StubBand(self.src)


class _StubGdal:
    GA_ReadOnly = 0
    GMF_ALL_VALID = 0x01

    @staticmethod
    def UseExceptions():
        pass

    @staticmethod
    def Open(path, mode=0):
        return _StubDataset(path)


import types                                            # noqa: E402
osgeo = types.ModuleType("osgeo")
osgeo.gdal = _StubGdal
sys.modules["osgeo"] = osgeo
sys.modules["osgeo.gdal"] = _StubGdal

for fixture in ("swath.tif", "full.tif", "ssar_swath.tif"):
    fp = os.path.join(TMP, fixture)
    a = read_footprint(fp, backend="gdal")
    b = read_footprint(fp, backend="rasterio")
    check(f"{fixture}: backends agree on the backend used",
          (a["backend"], b["backend"]), ("gdal", "rasterio"))
    check(f"{fixture}: backends agree on how the footprint was derived",
          a["derived"], b["derived"])
    check(f"{fixture}: backends agree on the ring", a["ring_map"], b["ring_map"])
    check(f"{fixture}: backends agree on the pixel grid",
          a["box_ring"], b["box_ring"])
    close(f"{fixture}: backends agree on over-coverage",
          a["over"], b["over"], 1e-12)

# zero-as-nodata goes through the band read rather than the mask, so it is a
# second GDAL entry point and needs its own agreement check
zp = os.path.join(TMP, "zerofill.tif")
za = read_footprint(zp, zero_is_nodata=True, backend="gdal")
zb = read_footprint(zp, zero_is_nodata=True, backend="rasterio")
check("zero-as-nodata: backends agree on the ring", za["ring_map"], zb["ring_map"])
check("zero-as-nodata: backends agree it came from the mask",
      (za["derived"], zb["derived"]), ("mask", "mask"))
close("zero-as-nodata: backends agree on the valid fraction",
      za["valid_fraction"], zb["valid_fraction"], 1e-12)

# auto falls through to rasterio when osgeo will not import
del sys.modules["osgeo"]
del sys.modules["osgeo.gdal"]
rec = read_footprint(os.path.join(TMP, "swath.tif"), backend="auto")
check("auto falls back to rasterio when osgeo is absent", rec["backend"],
      "rasterio")

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
