"""Exercise DPQED_radial's statistics, masking and naming without QGIS.

The pure helpers in DPQED_radial.py sit between explicit markers; this execs
that exact slice, so what is tested is the code that ships rather than a copy
of it. Qt, QGIS, GDAL and h5py are all absent here, which is the point: the
arithmetic that produces a gamma0 figure should not need a map canvas to be
checked.

    python3 tests_radial_stats.py          # exits non-zero on any assertion

The interesting cases are the ones where a plausible implementation is quietly
wrong: averaging dB, dropping the negative pixels a noise-subtracted GCOV
carries, and truncating two field names into one DBF column.
"""
import ast
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "DPQED_radial.py")
SRC_QT6 = os.path.join(HERE, "DPQED_radial_qt6.py")

BEGIN = "# ── BEGIN PURE HELPERS"
END = "# ── END PURE HELPERS"


def load_helpers(path=SRC):
    """The module's constants and its pure slice, without importing Qt.

    Constants are evaluated rather than literal_eval'd: DOMAIN_DEFAULT is
    another constant's name, STAT_KEYS is a comprehension over STAT_FIELDS and
    GCOV_GRID_RE is a compiled pattern, and a loader that only understood
    literals would silently skip all three and leave the slice half-defined.
    Nothing but the module's own top-level assignments is evaluated.
    """
    text = open(path, encoding="utf-8").read()
    safe = {"tuple": tuple, "sorted": sorted, "re": re, "math": math}
    consts = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            try:
                consts[target.id] = eval(
                    compile(ast.Expression(node.value), path, "eval"),
                    dict(safe), dict(consts))
            except Exception:
                pass
    for required in ("DOMAIN_POWER", "DOMAIN_AMPLITUDE", "DOMAIN_DB",
                     "DOMAIN_DEFAULT", "STAT_FIELDS", "STAT_KEYS",
                     "ROI_FIELDS", "DBF_NAME_LIMIT", "GCOV_POL_TERMS",
                     "GCOV_GRID_RE", "ZERO_IS_NODATA"):
        assert required in consts, f"constant {required} not found"
    body = text[text.index(BEGIN):text.index(END)]
    namespace = dict(consts, np=np, math=math, re=re, os=os, print=print)
    exec(compile(body, path, "exec"), namespace)
    return namespace


H = load_helpers()
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def close(label, got, want, tol):
    ok = got is not None and math.isfinite(got) and abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r} "
          f"(want {want} +/- {tol})")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want} +/- {tol}")


def is_nan(label, got):
    ok = got is not None and isinstance(got, float) and math.isnan(got)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r} (want NaN)")
    if not ok:
        failures.append(f"{label}: got {got!r}, want NaN")


rng = np.random.default_rng(20260916)

# ── 1. THE THREE DOMAINS DESCRIBE ONE SCENE ───────────────────────────────────
# The same pixels handed over as power, as amplitude and as dB must produce the
# same answer. If they do not, the domain selector is not a statement about the
# raster, it is a knob that changes the result.
print("\n── domains ──")
power = rng.exponential(0.08, 200_000)          # 1-look intensity, gamma0 ~ -11 dB
stats_power = H["roi_statistics"](power, "power")
stats_amp = H["roi_statistics"](np.sqrt(power), "amplitude")
stats_db = H["roi_statistics"](10.0 * np.log10(power), "db")
close("power vs amplitude mean_db", stats_amp["mean_db"],
      stats_power["mean_db"], 1e-9)
close("power vs dB mean_db", stats_db["mean_db"], stats_power["mean_db"], 1e-9)
close("power vs amplitude ENL", stats_amp["enl"], stats_power["enl"], 1e-9)
close("power vs dB ENL", stats_db["enl"], stats_power["enl"], 1e-9)

# ── 2. ENL IS A NUMBER OF LOOKS ───────────────────────────────────────────────
# Single-look intensity is exponential: its standard deviation equals its mean,
# so (mean/std)^2 is 1. Averaging L independent looks divides the variance by L
# and the estimate has to follow.
print("\n── ENL ──")
close("1-look ENL", H["roi_statistics"](power, "power")["enl"], 1.0, 0.02)
for looks in (4, 16):
    averaged = rng.exponential(0.08, (looks, 100_000)).mean(axis=0)
    close(f"{looks}-look ENL", H["roi_statistics"](averaged, "power")["enl"],
          looks, looks * 0.05)
constant = H["roi_statistics"](np.full(500, 0.25), "power")
close("uniform patch cv", constant["cv"], 0.0, 1e-9)
close("uniform patch mean_db", constant["mean_db"], -6.0206, 1e-3)

# ── 3. THE dB OF THE MEAN IS NOT THE MEAN OF THE dB ───────────────────────────
# The whole reason everything is computed on linear power. Over single-look
# speckle the log average sits about 2.5 dB low -- a bias the size of the
# radiometric differences this tool exists to find.
print("\n── log bias ──")
log_average = float(np.mean(10.0 * np.log10(power)))
close("mean of dB pixels is biased low", stats_power["mean_db"] - log_average,
      2.507, 0.05)
close("mean_db is 10log10(mean)", stats_power["mean_db"],
      10.0 * math.log10(float(power.mean())), 1e-9)

# ── 4. NOISE-SUBTRACTED PIXELS ────────────────────────────────────────────────
# A GCOV pixel can come out slightly negative once the noise floor is removed.
# Dropping those biases the mean up by what the subtraction removed, so they
# stay in the linear moments and are reported, not hidden.
print("\n── negative pixels ──")
values = np.array([-0.02, -0.01, 0.10, 0.30, 0.50, 0.70])
stats = H["roi_statistics"](values, "power")
check("negatives counted", stats["n"], 6)
check("negatives reported", stats["nonpos"], 2)
close("negatives kept in the mean", stats["mean"], float(values.mean()), 1e-12)
is_nan("min_db of a negative pixel", stats["min_db"])
close("max_db from the positive tail", stats["max_db"],
      10.0 * math.log10(0.70), 1e-9)
empty = H["roi_statistics"](np.array([]), "power")
check("no pixels: n", empty["n"], 0)
is_nan("no pixels: mean_db", empty["mean_db"])
all_negative = H["roi_statistics"](np.array([-1.0, -2.0]), "power")
is_nan("every pixel negative: mean_db", all_negative["mean_db"])
is_nan("every pixel negative: ENL", all_negative["enl"])

# ── 4b. GAMMA0 TO SIGMA0 ──────────────────────────────────────────────────────
# GCOV stores gamma0. Sigma0 is the same measurement against a flat reference
# area, and the product ships the conversion as a per-pixel factor. It is a
# ratio of areas, so it multiplies POWER -- and the test that matters is that
# it is not applied anywhere else, because applying it to amplitudes would be
# out by a square and to dB pixels it would be an addition.
print("\n── sigma0 ──")
gamma = rng.exponential(0.05, 50_000)
flat = H["roi_statistics"](gamma, "power")
doubled = H["roi_statistics"](gamma, "power", scale=np.full(gamma.size, 2.0))
close("a factor of 2 is +3.01 dB", doubled["mean_db"] - flat["mean_db"],
      10.0 * math.log10(2.0), 1e-9)
close("and the linear mean doubles", doubled["mean"], flat["mean"] * 2.0, 1e-12)
# A constant factor rescales every pixel alike, so the speckle statistics --
# which are ratios -- cannot move. A real factor varies with slope and does
# move them, which is a property of the terrain, not an error.
close("a constant factor leaves cv alone", doubled["cv"], flat["cv"], 1e-12)
close("and leaves ENL alone", doubled["enl"], flat["enl"], 1e-9)

# The same conversion reached through each domain must give one answer: the
# factor is applied after the pixels are power, never to what the raster held.
for domain, held in (("power", gamma), ("amplitude", np.sqrt(gamma)),
                     ("db", 10.0 * np.log10(gamma))):
    scaled = H["roi_statistics"](held, domain, scale=np.full(gamma.size, 3.0))
    close(f"via {domain}", scaled["mean_db"],
          flat["mean_db"] + 10.0 * math.log10(3.0), 1e-9)

# A varying factor is the real case: every pixel gets its own.
factor = rng.uniform(0.5, 2.0, gamma.size)
varied = H["roi_statistics"](gamma, "power", scale=factor)
close("a per-pixel factor is a per-pixel product", varied["mean"],
      float((gamma * factor).mean()), 1e-12)
# A factor that is missing or non-positive takes its pixel out rather than
# inventing a sigma0 for it.
holed = factor.copy()
holed[:100] = np.nan
check("pixels with no factor leave the measurement",
      H["roi_statistics"](gamma, "power", scale=holed)["n"], gamma.size - 100)
check("no scale at all is the gamma0 case",
      H["roi_statistics"](gamma, "power", scale=None), flat)

print("\n── finding the factor band ──")
check("NISAR's own name", H["factor_band_index"](
    ["HHHH", "HVHV", "rtcGammaToSigmaFactor"]), 3)
check("however QGIS labelled it", H["factor_band_index"](
    ["1: HHHH", "2: rtcGammaToSigmaFactor"]), 2)
check("case and separators do not matter", H["factor_band_index"](
    ["gamma_to_sigma", "HHHH"]), 1)
check("a product without one", H["factor_band_index"](["HHHH", "HVHV"]), None)
check("and a band that merely mentions sigma is not it",
      H["factor_band_index"](["sigma0_HH", "gamma0_HV"]), None)
check("no bands at all", H["factor_band_index"]([]), None)

# ── 4c. SAMPLING A GEOMETRY CUBE ──────────────────────────────────────────────
# The incidence angle comes off a coarse cube at the ROI's centre. A bilinear
# interpolator has one defining property -- it reproduces a linear function
# exactly -- and one decision worth pinning: what it does outside its grid.
print("\n── sampling a cube ──")
cx = np.linspace(500000.0, 500800.0, 9)
cy = np.linspace(4001000.0, 4000200.0, 7)          # descending, as north-up is
plane = 34.0 + 3e-5 * (cx[None, :] - cx[0]) + 1e-5 * (cy[0] - cy[:, None])


def plane_at(x, y):
    return 34.0 + 3e-5 * (x - cx[0]) + 1e-5 * (cy[0] - y)


for x, y in ((500000.0, 4001000.0), (500413.0, 4000561.0),
             (500800.0, 4000200.0), (500123.4, 4000987.6)):
    close(f"a plane is reproduced at ({x:.0f}, {y:.0f})",
          H["bilinear_at"](cx, cy, plane, x, y), plane_at(x, y), 1e-9)
# Ascending y as well: the vectors are flipped internally, values with them.
close("and with the y vector the other way up",
      H["bilinear_at"](cx, cy[::-1], plane[::-1, :], 500413.0, 4000561.0),
      plane_at(500413.0, 4000561.0), 1e-9)
for x, y in ((499999.0, 4000600.0), (500801.0, 4000600.0),
             (500400.0, 4001001.0), (500400.0, 4000199.0)):
    is_nan(f"outside the cube at ({x:.0f}, {y:.0f})",
           H["bilinear_at"](cx, cy, plane, x, y))
is_nan("a cube whose shape does not match its vectors",
       H["bilinear_at"](cx, cy, plane[:, :3], 500400.0, 4000600.0))
is_nan("a one-sample axis", H["bilinear_at"]([1.0], cy, plane, 1.0, 4000600.0))

print("\n── finding the incidence band ──")
check("NISAR's own name", H["incidence_band_index"](
    ["HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"]), 4)
check("however QGIS labelled it",
      H["incidence_band_index"](["1: HHHH", "2: incidence_angle"]), 2)
check("a product without one",
      H["incidence_band_index"](["HHHH", "HVHV"]), None)
check("the radarGrid cubes are found by band",
      sorted(H["radar_grid_datasets"]([
          "science/LSAR/GCOV/metadata/radarGrid/incidenceAngle",
          "science/LSAR/GCOV/metadata/radarGrid/xCoordinates",
          "science/SSAR/GCOV/metadata/radarGrid/incidenceAngle",
      ], "LSAR")), ["incidenceAngle", "xCoordinates"])

print("\n── either GCOV group layout ──")
# The 'grids/' segment is not in every product, and the group is taken from
# where a term was found rather than rebuilt from band and frequency.
for layout in ("science/LSAR/GCOV/grids/frequencyA/HHHH",
               "science/LSAR/GCOV/frequencyA/HHHH"):
    check(f"{layout.split('GCOV/')[1]} parses",
          sorted(H["gcov_grids"]([layout])), [("LSAR", "A")])
check("and the group comes from the path",
      H["gcov_group_of"]("science/LSAR/GCOV/frequencyA/HHHH"),
      "science/LSAR/GCOV/frequencyA")
check("including a GDAL subdataset spelling",
      H["gcov_group_of"](
          'HDF5:"/d/x.h5"://science/SSAR/GCOV/grids/frequencyB/VVVV'),
      'HDF5:"/d/x.h5"://science/SSAR/GCOV/grids/frequencyB')
check("something that is not a term", H["gcov_group_of"]("science/x"), None)

# ── 5. WHAT COUNTS AS A PIXEL ─────────────────────────────────────────────────
# Zero is fill in power and in amplitude, and a perfectly ordinary bright pixel
# in dB. NaN has to be excluded explicitly: it equals nothing, itself included,
# so a nodata comparison alone lets every one of them through.
print("\n── validity ──")
data = np.array([[1.0, 0.0, np.nan], [-999.0, 2.0, 3.0]])
mask = H["valid_mask"](data, -999.0, True, "power")
check("zeros, nodata and NaN out", int(mask.sum()), 3)
check("zero kept in dB", int(H["valid_mask"](data, -999.0, True, "db").sum()), 4)
check("zero kept when asked", int(H["valid_mask"](data, -999.0, False,
                                                  "power").sum()), 4)
check("no nodata declared", int(H["valid_mask"](data, None, False).sum()), 5)
check("NaN nodata is not a filter",
      int(H["valid_mask"](np.array([1.0, np.nan]), float("nan"), False).sum()), 1)

# ── 6. WHICH PIXELS ARE IN THE ROI ────────────────────────────────────────────
print("\n── polygon mask ──")
square = H["rect_ring"](0, 0, 4, 4)
square_px = H["ring_to_pixels"](square, 0.0, 4.0, 1.0, 1.0)
check("a 4x4 rectangle is 16 pixels",
      int(H["polygon_mask"](square_px, 4, 4).sum()), 16)
# Concave: an L covering seven of the sixteen cells. A scanline fill that
# toggled instead of counting crossings would fill the notch.
ell = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
check("concave L", int(H["polygon_mask"](
    H["ring_to_pixels"](ell, 0.0, 4.0, 1.0, 1.0), 4, 4).sum()), 7)
# A diagonal edge staircases: every cell the hypotenuse clips keeps its centre
# on the inside, so a small triangle overcounts by its perimeter. What has to
# hold is that the error is a perimeter and not a bias -- over a 400-pixel
# triangle it is 4%, over a 160000-pixel one it is a fifth of a percent.
for side, tolerance in ((20, 0.06), (400, 0.005)):
    triangle = [(0.0, 0.0), (float(side), 0.0), (0.0, float(side))]
    count = int(H["polygon_mask"](
        H["ring_to_pixels"](triangle, 0.0, float(side), 1.0, 1.0),
        side, side).sum())
    area = side * side / 2.0
    close(f"triangle of {side} px counts its area", count / area, 1.0, tolerance)
check("a ring outside the window", int(H["polygon_mask"](
    H["ring_to_pixels"]([(20, 20), (24, 20), (24, 24)], 0.0, 4.0, 1.0, 1.0),
    4, 4).sum()), 0)
check("two corners are not a polygon",
      int(H["polygon_mask"]([(0, 0), (1, 1)], 4, 4).sum()), 0)

# A ring that is not on the pixel grid takes the pixels whose CENTRES it holds,
# which is the rule QGIS's zonal statistics and gdal_rasterize apply.
# 2.8 pixels wide, with both edges inside a cell: it holds the centres at 1.5
# and 2.5 in each direction and no others, so four pixels are measured. An
# any-touch rule would return sixteen and quietly average in the neighbours.
offset = H["rect_ring"](0.6, 0.6, 3.4, 3.4)
check("centres inside, not cells touched", int(H["polygon_mask"](
    H["ring_to_pixels"](offset, 0.0, 4.0, 1.0, 1.0), 4, 4).sum()), 4)

# ── 7. THE WINDOW THE PIXELS ARE READ FROM ────────────────────────────────────
print("\n── pixel window ──")
check("window at the origin",
      H["pixel_window"]((0, 0, 4, 4), 0.0, 100.0, 1.0, 1.0, 200, 200),
      (0, 96, 4, 4))
check("window offset into the raster",
      H["pixel_window"]((10, 0, 14, 4), 0.0, 100.0, 1.0, 1.0, 200, 200),
      (10, 96, 4, 4))
check("clipped at the raster edge",
      H["pixel_window"]((-5, 95, 5, 105), 0.0, 100.0, 1.0, 1.0, 200, 200),
      (0, 0, 5, 5))
check("no overlap at all",
      H["pixel_window"]((500, 500, 600, 600), 0.0, 100.0, 1.0, 1.0, 200, 200),
      None)
check("30 m pixels",
      H["pixel_window"]((300, 0, 420, 120), 0.0, 3000.0, 30.0, 30.0, 500, 500),
      (10, 96, 4, 4))
check("empty bounds", H["pixel_window"](None, 0.0, 1.0, 1.0, 1.0, 10, 10), None)

# End to end: a ring in map units, on a 30 m grid, covering a known patch.
print("\n── window and mask together ──")
gt_x, gt_y, step = 500000.0, 4000000.0, 30.0
ring = H["rect_ring"](500300.0, 3999700.0, 500480.0, 3999880.0)
window = H["pixel_window"](H["ring_bounds"](ring), gt_x, gt_y, step, step,
                           2000, 2000)
check("6x6 pixels of a 30 m grid", window, (10, 4, 6, 6))
col0, row0, ncol, nrow = window
mask = H["polygon_mask"](
    H["ring_to_pixels"](ring, gt_x, gt_y, step, step, col0, row0), ncol, nrow)
check("every pixel of the window is in the ROI", int(mask.sum()), 36)

# The measurement itself: a synthetic scene with one patch at a known gamma0,
# read through the ring that was drawn over it.
scene = rng.exponential(0.01, (2000, 2000))              # -20 dB background
scene[4:10, 10:16] = rng.exponential(0.1, (6, 6))        # -10 dB patch
patch = scene[row0:row0 + nrow, col0:col0 + ncol][mask]
close("the ROI reads the patch, not the background",
      H["roi_statistics"](patch, "power")["mean_db"], -10.0, 1.5)

# ── 8. RING GEOMETRY ──────────────────────────────────────────────────────────
print("\n── geometry ──")
check("rectangle area", H["ring_area"](H["rect_ring"](0, 0, 30, 20)), 600.0)
check("area ignores winding", H["ring_area"]([(0, 0), (0, 2), (2, 2), (2, 0)]),
      4.0)
check("L area", H["ring_area"](ell), 7.0)
check("square centroid", H["ring_centroid"](H["rect_ring"](0, 0, 4, 4)),
      (2.0, 2.0))
# A run of vertices along one edge would drag a vertex mean toward it; the
# area-weighted centroid does not move.
check("centroid is not the vertex mean",
      H["ring_centroid"]([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 4),
                          (0, 4)]), (2.0, 2.0))
check("degenerate ring falls back to the vertex mean",
      H["ring_centroid"]([(0, 0), (2, 0), (4, 0)]), (2.0, 0.0))
check("bounds", H["ring_bounds"](ell), (0, 0, 4, 4))
check("no ring, no bounds", H["ring_bounds"]([]), None)

print("\n── double-click duplicates ──")
check("the repeated corner goes",
      H["dedupe_ring"]([(0, 0), (4, 0), (4, 4), (4, 4)], 0.5),
      [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)])
check("and so does one that closes the ring",
      H["dedupe_ring"]([(0, 0), (4, 0), (4, 4), (0.1, 0.1)], 0.5),
      [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)])
check("a real corner stays",
      len(H["dedupe_ring"]([(0, 0), (4, 0), (4, 4), (0, 4)], 0.5)), 4)
check("zero tolerance keeps everything",
      len(H["dedupe_ring"]([(0, 0), (4, 0), (4, 4), (4, 4.0001)], 0.0)), 4)

# ── 9. COLUMN NAMES ───────────────────────────────────────────────────────────
print("\n── band prefixes ──")
check("GCOV diagonal term", H["band_prefix"]("HHHH", 1), "HH")
check("GCOV cross term", H["band_prefix"]("HVHV", 2), "HV")
check("a named GeoTIFF band", H["band_prefix"]("gamma0_HH", 1), "HH")
check("RIVAL's numbered label", H["band_prefix"]("2: HV", 2), "HV")
check("a raster that names nothing", H["band_prefix"]("Band 1", 1), "b1")
check("an empty name", H["band_prefix"]("", 3), "b3")
check("a word that merely contains a pair",
      H["band_prefix"]("archive", 4), "ARCHIV")
check("something else entirely", H["band_prefix"]("sigma0", 2), "SIGMA0")

print("\n── DBF field names ──")
reserved = [name for name, _ in H["ROI_FIELDS"]]
names = H["dbf_field_names"](["HH", "HV", "VH", "VV"], reserved=reserved)
check("nothing over the DBF cap",
      max(len(name) for name in names.values()) <= H["DBF_NAME_LIMIT"], True)
check("every column distinct",
      len({name.lower() for name in names.values()}), len(names))
check("no collision with the ROI fields",
      {n.lower() for n in names.values()} & {r.lower() for r in reserved},
      set())
check("the obvious name where it fits", names[("HH", "mean_db")], "HH_mean_db")
# Two long prefixes truncate to the same ten characters; the second has to be
# told apart, or the DBF quietly holds one column where there should be two.
longnames = H["dbf_field_names"](["LONGPREFIX1", "LONGPREFIX2"], ("mean_db",))
check("truncation is disambiguated",
      len(set(longnames.values())), 2)
check("and still fits", max(len(n) for n in longnames.values()) <= 10, True)

# ── 10. THE EXPORTED TABLE ────────────────────────────────────────────────────
print("\n── long-format rows ──")
rois = [
    {"roi": 1, "name": "forest", "kind": "rect", "npix": 36, "area_m2": 32400.0,
     "cx": 1.0, "cy": 2.0, "lon": 77.0, "lat": 13.0, "domain": "power",
     "src": "a.vrt",
     "stats": {"HHHH": H["roi_statistics"](power[:1000], "power"),
               "HVHV": H["roi_statistics"](power[1000:2000], "power")}},
    {"roi": 2, "name": "water", "kind": "polygon", "npix": 12, "area_m2": 10800.0,
     "cx": 3.0, "cy": 4.0, "lon": 77.1, "lat": 13.1, "domain": "power",
     "src": "a.vrt", "stats": {}},
]
rows = H["stat_rows"](rois, ["HHHH", "HVHV"])
check("a row per ROI and band, plus the header", len(rows), 5)
check("header starts with the ROI fields",
      rows[0][:len(H["ROI_FIELDS"])], [n for n, _ in H["ROI_FIELDS"]])
check("then the band and the statistics",
      rows[0][len(H["ROI_FIELDS"]):],
      ["band"] + list(H["STAT_KEYS"]))
check("every row is the width of the header",
      {len(row) for row in rows}, {len(rows[0])})
check("the band is named in the row", rows[1][len(H["ROI_FIELDS"])], "HHHH")
check("a band with no statistics still gets a row",
      rows[3][len(H["ROI_FIELDS"]) + 1:], [None] * len(H["STAT_KEYS"]))

# ── 11. THE FOOTER ────────────────────────────────────────────────────────────
print("\n── summary across ROIs ──")
summary = H["summarise"]([
    {"mean_db": -10.0, "enl": 4.0},
    {"mean_db": -12.0, "enl": 6.0},
    {"mean_db": -14.0, "enl": 100.0},
])
check("ROIs counted", summary["count"], 3)
close("mean of the ROI means", summary["mean_db"], -12.0, 1e-9)
close("brightest minus darkest", summary["spread_db"], 4.0, 1e-9)
close("median ENL, so one odd ROI cannot set it", summary["enl"], 6.0, 1e-9)
undefined = H["summarise"]([{"mean_db": float("nan"), "enl": float("nan")}, {}])
check("nothing measurable", undefined["count"], 0)
is_nan("no mean to report", undefined["mean_db"])
is_nan("spread needs two ROIs", H["summarise"]([{"mean_db": -10.0}])["spread_db"])

# ── 11b. STATISTICS PER CLASS ─────────────────────────────────────────────────
# Ten vegetation ROIs and eight water ones share a scene, not a population. A
# spread taken across both is the difference between two land covers, not the
# product's uniformity, so the figures are grouped before they are summarised.
print("\n── by class ──")


def roi(label, mean_db, enl=4.0):
    return {"class": label, "stats": {"HH": {"mean_db": mean_db, "enl": enl}}}


mixed = ([roi("vegetation", -8.0 + 0.1 * i) for i in range(10)]
         + [roi("water", -22.0 - 0.1 * i) for i in range(8)])
by_class = dict(H["class_summary"](mixed, "HH"))
check("a row per class, and one for all of them", list(by_class),
      ["vegetation", "water", "all"])
check("each counting its own", by_class["vegetation"]["count"], 10)
check("and the other's", by_class["water"]["count"], 8)
close("vegetation reads its own brightness",
      by_class["vegetation"]["mean_db"], -7.55, 0.01)
close("water reads its own", by_class["water"]["mean_db"], -22.35, 0.01)
# The point of the whole exercise: a spread within a class is uniformity; the
# spread across both is 14 dB of land cover.
close("spread within vegetation", by_class["vegetation"]["spread_db"], 0.9, 1e-9)
close("spread within water", by_class["water"]["spread_db"], 0.7, 1e-9)
# -7.1 (the brightest vegetation) down to -22.7 (the darkest water).
close("and across everything, which is not a uniformity figure",
      by_class["all"]["spread_db"], 15.6, 1e-9)

check("one class needs no combined row",
      [label for label, _ in H["class_summary"](mixed[:10], "HH")],
      ["vegetation"])
check("no ROIs at all", H["class_summary"]([], "HH"), [])

print("\n── grouping ──")
check("classes keep the order they first appear",
      [label for label, _ in H["group_by_class"](
          [roi("water", -20), roi("snow", -5), roi("water", -21)])],
      ["water", "snow"])
check("and their members", [len(m) for _, m in H["group_by_class"](
    [roi("water", -20), roi("snow", -5), roi("water", -21)])], [2, 1])
# Case-folded to group, shown as first typed: 'Water' and 'water' are one
# class. A misspelling stays its own, which is how it gets noticed.
check("case does not split a class",
      [(label, len(m)) for label, m in H["group_by_class"](
          [roi("Water", -20), roi("water", -21), roi("wate", -22)])],
      [("Water", 2), ("wate", 1)])
check("an unset class has a name of its own",
      H["group_by_class"]([{"stats": {}}])[0][0], H["ROI_CLASS_UNSET"])
check("and so does an empty one",
      H["class_key"]("   "), H["ROI_CLASS_UNSET"])
check("whitespace does not make a new class",
      H["class_key"]("  Water  "), "water")

print("\n── the by-class export ──")
rows = H["class_summary_rows"](mixed, ["HH", "HV"])
check("a header and a row per class and band", len(rows), 1 + 3 * 2)
check("under names that say what they are", rows[0],
      ["class", "band", "rois", "mean_db", "spread_db", "enl"])
check("the band is named in the row", [r[1] for r in rows[1:]],
      ["HH"] * 3 + ["HV"] * 3)
check("and the class", [r[0] for r in rows[1:4]],
      ["vegetation", "water", "all"])

print("\n── formatting ──")
check("a number", H["format_stat"](3.14159, "{:.2f}"), "3.14")
check("NaN", H["format_stat"](float("nan")), "--")
check("missing", H["format_stat"](None), "--")
check("infinite", H["format_stat"](float("inf")), "--")

# ── 12. NISAR GCOV ────────────────────────────────────────────────────────────
print("\n── GCOV grids ──")
paths = [
    "science/LSAR/GCOV/grids/frequencyA/HHHH",
    "science/LSAR/GCOV/grids/frequencyA/HVHV",
    "science/LSAR/GCOV/grids/frequencyA/HHHV",       # off-diagonal, complex
    "science/LSAR/GCOV/grids/frequencyA/xCoordinates",
    "science/LSAR/GCOV/grids/frequencyB/VVVV",
    "science/LSAR/GCOV/metadata/processingInformation",
]
grids = H["gcov_grids"](paths)
check("one entry per frequency", sorted(grids), [("LSAR", "A"), ("LSAR", "B")])
check("terms found", sorted(grids[("LSAR", "A")]),
      ["HHHH", "HHHV", "HVHV"])
check("the diagonal only, in a fixed order",
      H["gcov_diagonal_terms"](grids[("LSAR", "A")]), ["HHHH", "HVHV"])
check("quad-pol order", H["gcov_diagonal_terms"](
    ["VVVV", "HHHH", "VHVH", "HVHV"]), ["HHHH", "HVHV", "VHVH", "VVVV"])
check("GDAL's own subdataset spelling", sorted(H["gcov_grids"]([
    'HDF5:"/data/NISAR_GCOV.h5"://science/SSAR/GCOV/grids/frequencyA/VVVV',
])), [("SSAR", "A")])
check("a GSLC is not a GCOV",
      H["gcov_grids"](["science/LSAR/GSLC/grids/frequencyA/HH"]), {})

print("\n── geotransform ──")
# Coordinate vectors give pixel CENTRES; the geotransform is anchored on the
# outer edge, half a pixel before the first one.
gt = H["geotransform_from_coords"]([500015.0, 500045.0, 500075.0],
                                   [4000985.0, 4000955.0, 4000925.0])
check("origin steps back half a pixel", (gt[0], gt[3]), (500000.0, 4001000.0))
check("pixel size and sign", (gt[1], gt[5]), (30.0, -30.0))
check("no rotation", (gt[2], gt[4]), (0.0, 0.0))
try:
    H["geotransform_from_coords"]([0.0, 10.0, 25.0], [0.0, -10.0, -20.0])
    check("an uneven grid is refused", "accepted", "rejected")
except ValueError:
    check("an uneven grid is refused", "rejected", "rejected")
try:
    H["geotransform_from_coords"]([0.0], [0.0])
    check("a one-sample grid is refused", "accepted", "rejected")
except ValueError:
    check("a one-sample grid is refused", "rejected", "rejected")

print("\n── VRT ──")
import xml.etree.ElementTree as ET
xml = H["gcov_vrt_xml"](
    [("HHHH", 'HDF5:"/data/a&b.h5"://science/LSAR/GCOV/grids/frequencyA/HHHH'),
     ("VVVV", 'HDF5:"/data/a&b.h5"://science/LSAR/GCOV/grids/frequencyA/VVVV')],
    2048, 1024, gt, "EPSG:32644")
root = ET.fromstring(xml)                      # unparseable XML fails here
check("raster size", (root.get("rasterXSize"), root.get("rasterYSize")),
      ("2048", "1024"))
check("projection", root.findtext("SRS"), "EPSG:32644")
check("one band per term", len(root.findall("VRTRasterBand")), 2)
check("bands are named for their terms",
      [b.findtext("Description") for b in root.findall("VRTRasterBand")],
      ["HHHH", "VVVV"])
check("the HDF5 path survives escaping",
      root.find("VRTRasterBand/SimpleSource/SourceFilename").text,
      'HDF5:"/data/a&b.h5"://science/LSAR/GCOV/grids/frequencyA/HHHH')
check("georeferenced",
      [float(v) for v in root.findtext("GeoTransform").split(",")], list(gt))
check("fill is NaN, not a number in the data's range",
      root.findtext("VRTRasterBand/NoDataValue"), "nan")

# ── 13. THE TWO BUILDS SHARE ONE ARITHMETIC ───────────────────────────────────
# DPQED_radial_qt6.py is the Qt6 twin. The two differ only in how they name Qt
# and QGIS things, and nothing between the PURE HELPERS markers names either --
# so the slices have to be identical, character for character. They are two
# files because a QGIS build is one Qt or the other; they are not two
# implementations, and this is what stops them becoming two.
print("\n── the Qt5 and Qt6 builds ──")


def pure_slice(path):
    text = open(path, encoding="utf-8").read()
    return text[text.index(BEGIN):text.index(END)]


if os.path.exists(SRC_QT6):
    check("the pure slice is the same in both builds",
          pure_slice(SRC_QT6) == pure_slice(SRC), True)
    QT6 = load_helpers(SRC_QT6)
    check("and the constants it reads are too",
          [QT6[name] for name in ("STAT_KEYS", "ROI_FIELDS", "DOMAIN_DEFAULT",
                                  "DBF_NAME_LIMIT", "GCOV_POL_TERMS")],
          [H[name] for name in ("STAT_KEYS", "ROI_FIELDS", "DOMAIN_DEFAULT",
                                "DBF_NAME_LIMIT", "GCOV_POL_TERMS")])
    # Spot-checked through the Qt6 slice's own functions, not just compared as
    # text: an identical slice that would not exec is still a broken file.
    close("the Qt6 build computes the same gamma0",
          QT6["roi_statistics"](power, "power")["mean_db"],
          stats_power["mean_db"], 1e-12)
    check("and rasterizes the same pixels", int(QT6["polygon_mask"](
        QT6["ring_to_pixels"](ell, 0.0, 4.0, 1.0, 1.0), 4, 4).sum()), 7)
    check("and names the same columns", QT6["band_prefix"]("HHHH", 1), "HH")
else:
    check("the Qt6 build is present", "missing", "present")

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
