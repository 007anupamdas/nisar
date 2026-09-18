"""RADIAL - RADiometric Image Assessment and Logger (QGIS).

One canvas, a set of ROIs drawn on it, and the radiometry of each one.

Where RIVAL measures WHERE a product puts the ground, this measures WHAT it
reports there: mark small rectangles or polygons over targets whose backscatter
you know something about -- a calibration site, a stretch of rainforest, a
reservoir, a bare field -- and read gamma0, its spread, and the number of looks
behind it. The ROIs and every statistic they produced export as one shapefile.

Two canvases were RIVAL's whole point: a measurement there is a correspondence
between an image and a reference, so both had to be on screen. Radiometry is a
measurement of one product against a known ground, so there is one canvas here,
and the second product enters by being loaded in turn over the same ROIs.

WORKFLOW
    Load GCOV        a GeoTIFF or VRT, or a NISAR GCOV '.h5' (see below)
    Rect / Polygon   draw ROIs; the table fills as each one closes
    Pan / Zoom       the view tools; Ctrl+1..5 selects any of the five
    Normalize        stretch the view, so the ROI is drawn on a legible scene
    Export SHP       the ROIs as polygons, every statistic in the table

    Loading another product does not clear the ROIs -- they are re-measured
    against it. That is the cross-product comparison: draw once, load each
    product in turn, export each one, and the difference between two exports is
    the radiometric difference between the products over the same ground.

    'Load ROIs' reads a polygon shapefile back in and re-measures it, so an ROI
    set outlives the session that drew it and can be shared between analysts.

STATISTICS, per ROI and per band
    Everything is computed on LINEAR POWER. GCOV carries gamma0 as power, and
    that is the only domain in which these quantities mean what they are called:
    the mean of dB pixels is not the dB of the mean (single-look speckle puts
    about 2.5 dB between them), and looks estimated from dB values are not
    looks. 'Domain' states what the raster holds -- power (the GCOV case),
    amplitude, or dB -- and the pixels are converted from it once, up front.

    n           valid pixels: finite, not nodata, and not zero unless
                'Zeros are data' is ticked. A SAR product means fill by zero.
    mean, std   linear power, and cv = std/mean
    enl         equivalent number of looks, (mean/std)^2 over the ROI. This is
                the ROI's own homogeneity, not the product's: over anything but
                a uniform target it reads low, because the scene's variation is
                counted as speckle. Read it on the rainforest patch, not on the
                one straddling a field boundary.
    mean_db     10log10(mean) -- the calibrated figure to quote and compare
    sdev_db     spread of the dB pixels, which is what the stretch shows
    min/p5/median/p95/max, in dB, from percentiles of the power
    nonpos      pixels at or below zero. Noise subtraction can leave a GCOV
                pixel slightly negative; those are kept in the linear mean,
                where they belong, and excluded from the dB spread, where they
                are undefined. A large count here means the dB columns describe
                only part of the ROI.

    The footer summarises the selected band across ROIs: the mean of the ROI
    means, the spread between the brightest and darkest, and the median ENL.
    Over patches of one cover type that spread is the product's uniformity.

EXPORT
    'Export SHP' writes one polygon per ROI in the working CRS, carrying every
    statistic for every band: <pol>_mean_db, <pol>_enl and the rest, with the
    polarization taken from the band's own name (a GCOV 'HHHH' term is the HH
    power, so it prefixes HH_). DBF caps a field name at 10 characters, so the
    same numbers are written again beside the shapefile as '<stem>_stats.csv',
    one row per ROI and band, under names that are not truncated. 'Export CSV'
    writes that table alone.

    The export also records the domain the pixels were read as and the raster
    they came from, because a gamma0 figure without them is not reproducible.

NISAR GCOV '.h5'
    A GCOV product ships as HDF5, not as a COG. Handed one, this builds a VRT
    beside it stacking the frequency's covariance terms as bands -- HHHH, HVHV,
    VVVV in that order -- with the grid's own geotransform and projection, and
    loads that. Nothing is copied: the VRT reads the HDF5 in place. It needs
    GDAL's HDF5 driver; without it, convert with DPQED_h52tif.py and load the
    TIF instead.

    Only the diagonal terms are offered. The off-diagonal terms are complex
    covariances, whose magnitude is not a backscatter and does not belong in a
    gamma0 column.

ADOPTED FROM RIVAL
    Normalize is the same translucent per-canvas tool with the same clip: it
    measures WHAT IS IN VIEW, applies the SAR sqrt-gamma stretch, and pins the
    result, so panning and zooming cannot re-stretch the scene under an ROI
    being drawn. The R/G/B band picker, the exclusive Pan/Zoom row on Ctrl+1..5,
    the sampled statistics that keep a big COG from freezing the window, and the
    working CRS adopted from the raster all behave as they do there.

    The stretch is a rendering. It does not touch the statistics, which are
    always read from the source pixels at full resolution.

Run it from the QGIS Python console:

    exec(open(r"path/to/DPQED_radial.py").read())
"""

import csv
import math
import os
import re
import sys
import threading
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QFileDialog, QHeaderView, QCheckBox, QComboBox,
                             QMessageBox, QApplication, QShortcut, QLabel,
                             QFrame, QButtonGroup, QPlainTextEdit, QAbstractItemView)
from PyQt5.QtCore import Qt, QObject, QEvent, QVariant, QRectF
from PyQt5.QtGui import (QKeySequence, QFont, QColor, QFontMetricsF,
                         QPainter)
from qgis.gui import (QgsMapCanvas, QgsMapCanvasItem, QgsMapTool,
                      QgsMapToolPan, QgsMapToolZoom, QgsRubberBand)
from qgis.core import (QgsProject, QgsPointXY, QgsRasterLayer, QgsVectorLayer,
                       QgsGeometry, QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform, QgsSingleBandGrayRenderer,
                       QgsMultiBandColorRenderer, QgsContrastEnhancement,
                       QgsRasterBandStats, QgsRectangle, QgsFields, QgsField,
                       QgsFeature, QgsVectorFileWriter, QgsWkbTypes)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# NORMALIZE. Identical in behaviour to RIVAL's, and the reasoning is the same:
# a SAR scene stretched on a plain min/max is set by a handful of bright
# scatterers and renders black, and re-measuring on every pan changes the
# picture under a measurement in progress.
NORM_GAMMA = 0.5                 # sqrt stretch
NORM_CLIP_CHOICES = (0.0, 0.5, 1.0, 2.0, 5.0)
NORM_CLIP_DEFAULT = 2.0
NORM_USE_DATA_RANGE = True
NORM_MIN = 0                     # fallback range, only when nothing measurable
NORM_MAX = 1500

# Working (projected) CRS the ROI geometry and the exported shapefile live in.
# Adopted from the raster when it carries a projected CRS; this is the fallback.
# Areas are quoted in its units squared, so a geographic CRS is refused.
WORKING_CRS_DEFAULT = "EPSG:32644"      # UTM 44N

RASTER_EXTS = (".tif", ".tiff", ".vrt")
H5_EXTS = (".h5", ".hdf5", ".he5")

# Map tools, in button order. The two ROI tools draw; the rest move the view.
TOOL_SELECT = "select"
TOOL_RECT = "rect"
TOOL_POLY = "polygon"
TOOL_PAN = "pan"
TOOL_ZOOM_IN = "zoom in"
TOOL_ZOOM_OUT = "zoom out"
MAP_TOOLS = (TOOL_SELECT, TOOL_RECT, TOOL_POLY, TOOL_PAN, TOOL_ZOOM_IN,
             TOOL_ZOOM_OUT)

# ROI outlines. Cyan reads over the red/magenta a two-band SAR composite tends
# toward, and over grey; the selected ROI goes yellow so the table and the
# canvas always agree about which one is being read.
ROI_COLOR = (0, 255, 255)
ROI_COLOR_SELECTED = (255, 255, 0)
ROI_COLOR_DRAWING = (255, 0, 255)
ROI_WIDTH = 2
ROI_WIDTH_SELECTED = 3
ROI_FILL_ALPHA = 40             # a hint of fill, so a small ROI is findable

# The ROI's number, drawn at its centre. Colour alone tells you which ROI the
# table is on; it does not tell you which of a dozen ROIs is number 7, and the
# table's first column is the only place that number otherwise exists.
ROI_LABEL_POINT = 9
ROI_LABEL_PAD = 3.0
ROI_LABEL_BACKDROP = (0, 0, 0, 160)     # so a number over bright scene reads

# A polygon closed by double-click receives the same vertex twice -- the press
# that precedes the double-click has already added it. Consecutive vertices
# within this many screen pixels of each other are one vertex.
VERTEX_MERGE_PX = 3

# An ROI smaller than this is a misclick, not a measurement.
MIN_ROI_PIXELS = 4

# This tool is for small ROIs over uniform targets: the pixels are read at full
# resolution, and they are read per band. A window past this is refused rather
# than left to swap the machine to a halt -- zoom in and draw a smaller one.
ROI_MAX_PIXELS = 20_000_000

# What the pixels hold. Everything is converted to linear power before anything
# is computed: the mean of dB pixels is not the dB of the mean, and looks
# estimated from dB values are not looks. NISAR GCOV is power.
DOMAIN_POWER = "power"
DOMAIN_AMPLITUDE = "amplitude"
DOMAIN_DB = "db"
DOMAIN_CHOICES = (
    (DOMAIN_POWER, "Power (GCOV gamma0)"),
    (DOMAIN_AMPLITUDE, "Amplitude (|GSLC|)"),
    (DOMAIN_DB, "dB"),
)
DOMAIN_DEFAULT = DOMAIN_POWER

# GCOV carries RTC-corrected gamma0: backscatter referred to the terrain's own
# sloped area. Sigma0 refers the same measurement to a flat ground area
# instead, and the product ships the conversion beside the data as a per-pixel
# layer -- sigma0 = gamma0 x rtcGammaToSigmaFactor. The two differ by several
# dB on any slope and not at all on flat ground, so which one a figure is
# cannot be read off the number, and every export records it.
BACKSCATTER_GAMMA0 = "gamma0"
BACKSCATTER_SIGMA0 = "sigma0"
BACKSCATTER_CHOICES = (
    (BACKSCATTER_GAMMA0, "gamma0 (as stored)"),
    (BACKSCATTER_SIGMA0, "sigma0 (RTC factor)"),
)
BACKSCATTER_DEFAULT = BACKSCATTER_GAMMA0

# The band holding that factor. NISAR names it 'rtcGammaToSigmaFactor'; the
# match is loose so a GeoTIFF carrying the same layer under a tidied-up name is
# still recognised, and it is a search rather than an equality so a band
# described as '3: rtcGammaToSigmaFactor' matches too.
RTC_FACTOR_RE = re.compile(r"(?i)gamma.?to.?sigma")

# A GeoTIFF whose bands carry no description -- which is most of them, since a
# writer has to go out of its way to set one -- can still declare which band
# holds the factor, as a plain GDAL metadata item. DPQED_gcov2tif.py writes
# both; the name is tried first and this is the fallback.
RTC_FACTOR_BAND_KEY = "RTC_GAMMA_TO_SIGMA_BAND"

# The incidence angle, in degrees. Like the RTC factor it is a band of the
# raster and not a channel, and like it, it is reported per ROI rather than
# measured as backscatter. DPQED_gcov2tif.py resamples it from the product's
# metadata/radarGrid cube; a GCOV loaded straight from its '.h5' has the cube
# itself, and is read from that instead.
INCIDENCE_RE = re.compile(r"(?i)incidence.?angle")
INCIDENCE_BAND_KEY = "INCIDENCE_ANGLE_BAND"
RADAR_GRID_RE = re.compile(
    r"science/(?P<band>[LS]SAR)/GCOV/metadata/radarGrid/(?P<name>[A-Za-z0-9_]+)$")
INCIDENCE_NAME = "incidenceAngle"

# ── ROI CLASSES ───────────────────────────────────────────────────────────────
# What an ROI is over. Statistics are reported per class, because the figures
# that matter -- the spread between ROIs, the median ENL -- only mean anything
# within one land cover. A spread taken across water and vegetation together is
# not a measure of the product's uniformity; it is the difference between two
# land covers, which is something you knew before you drew them.
#
# The picker is editable, so this list is the common cases rather than the
# permitted ones: anything typed becomes a class, here or in the table.
ROI_CLASSES = ("vegetation", "water", "snow")
ROI_CLASS_DEFAULT = ROI_CLASSES[0]
ROI_CLASS_UNSET = "unclassified"
ROI_CLASS_ALL = "all"
# The class the noise-floor estimate is taken over. Water, because calm water
# is the nearest thing to radiometrically empty ground a scene reliably has --
# not because it returns nothing, which it does not: wind roughening puts a
# real signal in it, and that is one reason the estimate is a bound.
NESZ_CLASS = "water"

# Zero is how a SAR product says 'no data' when it declares no nodata value --
# outside the swath, beyond the frame, masked in processing. Counted as data it
# drags every mean down and puts an ROI's looks estimate on the floor. Untick
# only for a raster where zero is a real measurement.
ZERO_IS_NODATA = True

# Statistics computed per ROI per band: key, column header, DBF type, format.
# The keys double as the DBF field suffixes, so they are already at the length
# a 2-character polarization prefix leaves -- do not lengthen them.
STAT_FIELDS = (
    ("n",       "N",         "int",    "{:.0f}"),
    ("mean",    "Mean",      "double", "{:.6g}"),
    ("std",     "Std",       "double", "{:.6g}"),
    ("cv",      "CV",        "double", "{:.3f}"),
    ("enl",     "ENL",       "double", "{:.2f}"),
    ("mean_db", "Mean dB",   "double", "{:.2f}"),
    ("sdev_db", "Std dB",    "double", "{:.2f}"),
    ("min_db",  "Min dB",    "double", "{:.2f}"),
    ("p5_db",   "P5 dB",     "double", "{:.2f}"),
    ("med_db",  "Median dB", "double", "{:.2f}"),
    ("p95_db",  "P95 dB",    "double", "{:.2f}"),
    ("max_db",  "Max dB",    "double", "{:.2f}"),
    ("nonpos",  "Non-pos",   "int",    "{:.0f}"),
)
STAT_KEYS = tuple(key for key, _, _, _ in STAT_FIELDS)

# Which of those the table shows for the selected band. The rest are in the
# detail panel, which shows every band at once, and in both exports.
TABLE_STATS = ("n", "mean_db", "sdev_db", "cv", "enl", "med_db", "p5_db", "p95_db")

# Per-ROI fields, written before the per-band statistics. 'domain' and 'src'
# are not decoration: a gamma0 figure is not reproducible without knowing what
# the pixels were read as and which raster they came from.
ROI_FIELDS = (
    ("roi",     "int"),
    ("name",    "string"),
    ("kind",    "string"),
    ("npix",    "int"),
    ("area_m2", "double"),
    ("cx",      "double"),
    ("cy",      "double"),
    ("lon",     "double"),
    ("lat",     "double"),
    ("domain",  "string"),
    ("class",   "string"),
    ("backscat", "string"),
    ("inc_deg", "double"),
    ("src",     "string"),
)
ROI_TABLE_COLUMNS = ("roi", "name", "class", "kind", "npix", "area_m2",
                     "inc_deg")

# The by-class summary beside the ROI table: one row per class, for the band
# the table is showing.
CLASS_TABLE_COLUMNS = ("Class", "ROIs", "Mean dB", "Spread dB", "ENL")

# DBF caps a field name at 10 characters and silently truncates past it, which
# turns HH_mean_db and HH_med_db into one column. Names are built to fit and
# checked for collisions instead; the companion CSV carries the full names.
DBF_NAME_LIMIT = 10

# Percentile clip for the initial composite, before Normalize is pressed.
RGB_CLIP_LOW = 0.02
RGB_CLIP_HIGH = 0.98

# Pixels sampled when working out a stretch. Unbounded, QGIS reads the WHOLE
# raster at full resolution to build the histogram, per band, on the GUI thread.
# This is the figure QGIS itself uses for its estimated min/max. It bounds the
# RENDERING only -- ROI statistics are never sampled.
RASTER_SAMPLE_SIZE = 250000

RGB_DEFAULT_BANDS = (1, 1, 1)    # all three on band 1 renders grey

# A GCOV grid in the HDF5 tree. Products differ over whether the frequency
# groups sit under 'grids/' or directly under GCOV, so that segment is optional
# and the group is taken from where a term was actually found rather than
# rebuilt from a guess -- a layout this file assumed once and got wrong.
# The diagonal covariance terms are the backscatter; the off-diagonal ones are
# complex and are not offered.
GCOV_POL_TERMS = ("HHHH", "HVHV", "VHVH", "VVVV", "RHRH", "RVRV")
GCOV_GRID_RE = re.compile(
    r"science/(?P<band>[LS]SAR)/GCOV/(?:grids/)?frequency(?P<freq>[A-Z])/"
    r"(?P<term>[A-Z]{4})$")


# ── BEGIN PURE HELPERS ────────────────────────────────────────────────────────
# Everything between these markers is plain Python and numpy -- no Qt, no QGIS,
# no GDAL -- so the statistics, the polygon rasterization and the field naming
# can be exercised without a QGIS session. tests_radial_stats.py execs exactly
# this slice, so what is tested is the code that ships rather than a copy.

def to_power(values, domain=DOMAIN_DEFAULT):
    """Pixel values as linear power, whatever domain the raster carries.

    Every statistic below is computed here and nowhere else. Averaging dB is
    averaging logarithms: over single-look speckle the mean of the dB pixels
    sits about 2.5 dB below the dB of the mean, and an ENL taken from dB values
    is not a number of looks at all. Converting once, up front, is what makes
    the rest of this file allowed to be simple.
    """
    values = np.asarray(values, dtype=float)
    if domain == DOMAIN_POWER:
        return values
    if domain == DOMAIN_AMPLITUDE:
        return values * values
    if domain == DOMAIN_DB:
        return np.power(10.0, values / 10.0)
    raise ValueError(f"unknown domain {domain!r}")


def to_db(value):
    """10log10, with anything at or below zero returned as NaN.

    Noise subtraction can leave a GCOV pixel slightly negative. That is a real
    measurement and belongs in the linear mean; its logarithm does not exist,
    and inventing one -- clamping to a floor, dropping it silently -- would put
    a number in a dB column that no pixel supports.
    """
    value = np.asarray(value, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.where(value > 0, value, np.nan))
    return out if out.ndim else float(out)


def valid_mask(data, nodata=None, zero_is_nodata=ZERO_IS_NODATA,
               domain=DOMAIN_DEFAULT):
    """Which pixels are measurements: finite, not nodata, and not fill.

    NaN must be excluded explicitly rather than left to the nodata test: NaN
    never equals anything, itself included, so a `data != nodata` test passes
    every NaN through, and one NaN makes every percentile NaN.

    Zero is how a SAR product says 'nothing here' when it declares no nodata
    value. That is true of power and of amplitude, and false in dB, where 0 dB
    is a power of 1 -- a perfectly ordinary bright pixel. The domain decides.
    """
    data = np.asarray(data, dtype=float)
    mask = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        mask &= (data != nodata)
    if zero_is_nodata and domain in (DOMAIN_POWER, DOMAIN_AMPLITUDE):
        mask &= (data != 0.0)
    return mask


def empty_statistics():
    """The statistics of no pixels: counts zero, everything else undefined."""
    stats = {key: float("nan") for key in STAT_KEYS}
    stats["n"] = 0
    stats["nonpos"] = 0
    return stats


def roi_statistics(values, domain=DOMAIN_DEFAULT, scale=None):
    """Radiometry of one ROI in one band, from its already-valid pixels.

    `values` is what the caller decided was data -- validity is a property of
    the raster, and is settled before this is called. Everything here is the
    arithmetic.

    `scale` multiplies the pixels once they are power, which is the only place
    the RTC gamma-to-sigma factor can be applied: it is a ratio of reference
    areas, so it scales a backscatter. Applied to amplitudes it would be out by
    a square, and to dB pixels it would be an addition rather than a product.

    The linear moments carry every pixel, negatives included, because dropping
    the low tail of a noise-subtracted product biases the mean upward by
    exactly the amount the noise subtraction was trying to remove. The dB
    columns carry only the positive pixels, since the others have no logarithm,
    and `nonpos` says how many were left out so the reader can judge whether
    the dB figures describe the ROI or a part of it.
    """
    power = to_power(np.asarray(values, dtype=float).ravel(), domain)
    if scale is not None:
        power = power * np.asarray(scale, dtype=float).ravel()
    power = power[np.isfinite(power)]
    if power.size == 0:
        return empty_statistics()

    mean = float(power.mean())
    std = float(power.std())
    positive = power[power > 0]
    nan = float("nan")

    stats = {
        "n": int(power.size),
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 0 else nan,
        # ENL as the ROI reports it: (mean/std)^2 of the intensity. Over a
        # uniform target this is the product's looks; over anything else the
        # scene's own variation is counted as speckle and it reads low.
        "enl": (mean / std) ** 2 if (mean > 0 and std > 0) else nan,
        "mean_db": float(to_db(mean)) if mean > 0 else nan,
        "nonpos": int(power.size - positive.size),
    }
    lo, p5, med, p95, hi = (float(v) for v in np.percentile(
        power, [0.0, 5.0, 50.0, 95.0, 100.0]))
    for key, value in (("min_db", lo), ("p5_db", p5), ("med_db", med),
                       ("p95_db", p95), ("max_db", hi)):
        stats[key] = float(to_db(value)) if value > 0 else nan
    # The spread the eye sees: dB pixels, not the dB of the linear spread.
    stats["sdev_db"] = (float(np.std(to_db(positive)))
                        if positive.size > 1 else nan)
    return stats


def factor_band_index(labels):
    """1-based band number of the RTC gamma-to-sigma factor, or None.

    The factor is not a channel and is taken out of the bands that get
    measured: its mean is a ratio of areas, and a column of those sitting
    beside the gamma0 columns, under the same headings, would be read as a
    backscatter by anyone who did not write this.

    It stays in the raster and in the R/G/B picker, because looking at it is
    how you see where the terrain correction is doing the most work -- and
    therefore where gamma0 and sigma0 have least to do with each other.
    """
    for index, label in enumerate(labels, start=1):
        if RTC_FACTOR_RE.search(str(label or "")):
            return index
    return None


def incidence_band_index(labels):
    """1-based band number of the incidence angle, or None.

    Taken out of the measured bands for the same reason as the RTC factor: its
    mean is an angle, and an angle in a column headed mean_db beside the
    gamma0 columns would be read as a backscatter.
    """
    for index, label in enumerate(labels, start=1):
        if INCIDENCE_RE.search(str(label or "")):
            return index
    return None


def radar_grid_datasets(paths, band=None):
    """{name: path} for a band's radarGrid metadata cubes."""
    found = {}
    for path in paths:
        match = RADAR_GRID_RE.search(str(path).strip())
        if match and (band is None or match.group("band") == band):
            found[match.group("name")] = str(path).strip()
    return found


def bilinear_at(x_coords, y_coords, values, at_x, at_y):
    """`values` sampled at one point, bilinearly, or NaN outside its grid.

    For reading a geometry cube at an ROI's centre, where the cube is a few
    hundred samples across and the point is one. Either coordinate vector may
    descend -- a north-up grid's y does -- so both are put the same way round
    first, taking the values with them.

    Outside the grid it returns NaN rather than the nearest edge value: an
    incidence angle from beyond the cube is an extrapolation, and a plausible
    looking one, which is worse than a gap that says so.
    """
    x = np.asarray(x_coords, dtype=float).ravel()
    y = np.asarray(y_coords, dtype=float).ravel()
    grid = np.asarray(values, dtype=float)
    nan = float("nan")
    if x.size < 2 or y.size < 2 or grid.shape != (y.size, x.size):
        return nan
    if x[-1] < x[0]:
        x, grid = x[::-1], grid[:, ::-1]
    if y[-1] < y[0]:
        y, grid = y[::-1], grid[::-1, :]
    if not (x[0] <= at_x <= x[-1] and y[0] <= at_y <= y[-1]):
        return nan
    i = min(max(int(np.searchsorted(x, at_x, side="right")) - 1, 0), x.size - 2)
    j = min(max(int(np.searchsorted(y, at_y, side="right")) - 1, 0), y.size - 2)
    fx = (at_x - x[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else 0.0
    fy = (at_y - y[j]) / (y[j + 1] - y[j]) if y[j + 1] != y[j] else 0.0
    return float(grid[j, i] * (1 - fx) * (1 - fy)
                 + grid[j, i + 1] * fx * (1 - fy)
                 + grid[j + 1, i] * (1 - fx) * fy
                 + grid[j + 1, i + 1] * fx * fy)


def roi_conventions(roi):
    """The backscatter conventions an ROI was measured in, gamma0 first.

    A GCOV carrying its RTC factor gets both out of one read of the pixels, so
    an export can state both and a reader need not be told which one to ask
    for. A product without the factor has only the gamma0 it holds.
    """
    stats = roi.get("stats") or {}
    return [name for name in (BACKSCATTER_GAMMA0, BACKSCATTER_SIGMA0)
            if stats.get(name)]


def roi_stats(roi, band, backscatter=None):
    """One band's statistics in one convention, or None.

    `roi["stats"]` is keyed by convention and then by band. Defaulting to the
    ROI's own `backscat` means every caller that does not care which
    convention it is looking at gets the one the window is showing.
    """
    stats = roi.get("stats") or {}
    if backscatter is None:
        backscatter = roi.get("backscat") or BACKSCATTER_GAMMA0
    return (stats.get(backscatter) or {}).get(band)


def class_key(value):
    """The label an ROI groups under: trimmed and case-folded."""
    text = str(value or "").strip()
    return text.lower() if text else ROI_CLASS_UNSET


def group_by_class(rois):
    """[(label, [roi])], in the order the classes first appear.

    Case-folded to group and shown as first typed, so 'Water' and 'water' are
    one class rather than two. A misspelling is still its own class, which is
    the right answer: it shows up as a group of one rather than being quietly
    folded into the class it was meant to be.
    """
    groups = {}
    for roi in rois:
        key = class_key(roi.get("class"))
        if key not in groups:
            label = str(roi.get("class") or "").strip() or ROI_CLASS_UNSET
            groups[key] = (label, [])
        groups[key][1].append(roi)
    return list(groups.values())


def visible_rois(rois, class_filter):
    """The ROIs a class filter leaves on view.

    ROI_CLASS_ALL means every one. Any other value keeps the ROIs of that class
    alone, matched the way the by-class summary groups them, so what the filter
    shows and what the summary counts can never disagree.

    Only the display is filtered. Every export writes `rois` whole: a filter is
    a way of reading a measured set, not a decision about what was measured,
    and a shapefile that quietly held a third of the ROIs because a dropdown
    was left on 'water' would be the worst kind of wrong -- plausible.
    """
    key = class_key(class_filter)
    if key == class_key(ROI_CLASS_ALL):
        return list(rois or ())
    return [roi for roi in rois or () if class_key(roi.get("class")) == key]


def class_summary(rois, band, backscatter=None):
    """[(label, summary)] per class for one band, then all of them together.

    The combined row comes last and is labelled: it is the scene-wide
    brightness, which is worth a glance, and it is not a uniformity figure once
    more than one land cover is in it. It is left out entirely when there is
    only one class, where it would just repeat that class's row.
    """
    groups = group_by_class(rois)
    summaries = [(label, summarise([roi_stats(roi, band, backscatter)
                                    for roi in members]))
                 for label, members in groups]
    if len(groups) > 1:
        summaries.append((ROI_CLASS_ALL, summarise(
            [roi_stats(roi, band, backscatter) for roi in rois])))
    return summaries


def class_slug(label, fallback="class"):
    """A class name as a filename fragment: lowercase, no surprises.

    Everything outside a-z, 0-9 and '-' becomes '_', because a class is typed
    freely and 'open water / lake' is a legal class and an illegal filename on
    at least one of the systems this runs on. Trimmed to something a person can
    still read at a glance in a directory listing.
    """
    text = str(label or "").strip().lower()
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_"
                      for char in text).strip("_")
    return (cleaned[:40].strip("_") or fallback)


def class_exports(rois, fallback="class"):
    """[(label, slug, [roi])] per class, with the slugs made unique.

    Two classes can slug to one name -- 'open water' and 'open-water' both
    become 'open_water' -- and the second would overwrite the first's files
    with no error at all, which is how a class silently vanishes from an export
    directory. So a collision gets a numbered suffix rather than a coin toss.
    """
    exports, taken = [], set()
    for label, members in group_by_class(rois):
        slug = base = class_slug(label, fallback)
        index = 2
        while slug in taken:
            slug = f"{base}_{index}"
            index += 1
        taken.add(slug)
        exports.append((label, slug, members))
    return exports


def nesz_estimate(rois, band, nesz_class=NESZ_CLASS, backscatter=None):
    """The noise floor this scene shows, from the ROIs of one class.

    NESZ is a property of the instrument and the geometry, not of the pixels,
    and nothing in an ROI separates scene from noise: what a dark ROI measures
    is scene PLUS noise. So this is not a measurement of NESZ. It is the
    tightest **upper bound** the imagery can give -- mean sigma0 over ground
    that returns as close to nothing as this scene offers -- and it is reported
    under that name, with the count of ROIs it rests on, so it is never quoted
    as though the instrument had been characterised.

    In sigma0 always, whatever the window is showing: NESZ is defined against
    sigma0, and a gamma0 figure carrying the name would be wrong by the RTC
    factor -- which varies across the scene, so the error would not even be a
    constant. A product with no RTC factor gets no estimate at all rather than
    a gamma0 one relabelled.

    Pooled by pixel count, not by averaging the ROI means: a 4000-pixel lake
    and a 40-pixel pond are not equal evidence about the floor. The pooling is
    in linear power, because that is what averages.

    Three numbers come back rather than one:

      nesz_db    the pooled mean. The bound.
      floor_db   the darkest single ROI's mean. Tighter, and noisier -- the
                 same bound taken from the least-returning water found.
      nonpos     pixels at or below zero across those ROIs. A noise-subtracted
                 product has already had its floor removed, so these are the
                 sign that the bound is measuring the subtraction and not the
                 instrument. Many of them and the number means little.
    """
    if backscatter is None:
        backscatter = BACKSCATTER_SIGMA0
    key = class_key(nesz_class)
    weights, powers, means_db, pixels, nonpos, used = [], [], [], 0, 0, 0
    for roi in rois or ():
        if class_key(roi.get("class")) != key:
            continue
        stats = roi_stats(roi, band, backscatter)
        if not stats:
            continue
        used += 1
        count = stats.get("n") or 0
        mean = stats.get("mean")
        nonpos += int(stats.get("nonpos") or 0)
        pixels += int(count)
        if count and mean is not None and np.isfinite(mean) and mean > 0:
            weights.append(float(count))
            powers.append(float(mean))
        value = stats.get("mean_db")
        if value is not None and np.isfinite(value):
            means_db.append(float(value))
    nan = float("nan")
    pooled = (float(np.average(powers, weights=weights))
              if powers else nan)
    return {
        "class": nesz_class,
        "backscat": backscatter,
        "rois": used,
        "n": pixels,
        "nonpos": nonpos,
        "nesz_db": to_db(pooled) if powers else nan,
        "floor_db": min(means_db) if means_db else nan,
    }


def nesz_margin_db(roi, band, nesz_db, backscatter=None):
    """How far one ROI stands above the noise floor, in dB, or NaN.

    The figure that says whether an ROI was measured or merely sampled the
    floor. A few dB of margin and the backscatter reported for it is mostly
    noise, whatever the mean says.
    """
    if nesz_db is None or not np.isfinite(nesz_db):
        return float("nan")
    stats = roi_stats(roi, band, backscatter or BACKSCATTER_SIGMA0)
    value = (stats or {}).get("mean_db")
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value) - float(nesz_db)


def nesz_rows(rois, bands, nesz_class=NESZ_CLASS):
    """The noise-floor estimate in long form: a header and a row per band."""
    header = ["band", "class", "rois", "n", "nonpos", "nesz_db", "floor_db",
              "note"]
    note = ("upper bound: mean sigma0 over %s, scene+noise, "
            "not a measured NESZ" % nesz_class)
    rows = [header]
    for band in bands or ():
        estimate = nesz_estimate(rois, band, nesz_class)
        rows.append([band, estimate["class"], estimate["rois"], estimate["n"],
                     estimate["nonpos"], estimate["nesz_db"],
                     estimate["floor_db"], note])
    return rows


def class_summary_rows(rois, bands):
    """The by-class summary in long form: a row per class, band and convention.

    Both conventions where both were measured, for the same reason the per-ROI
    table carries both: exporting twice to compare gamma0 with sigma0 is two
    files that have to be lined up by hand, and the second one is the one that
    gets forgotten.
    """
    rows = [["class", "band", "backscat", "rois", "mean_db", "spread_db",
             "enl"]]
    conventions = []
    for roi in rois:
        for name in roi_conventions(roi):
            if name not in conventions:
                conventions.append(name)
    for band in bands:
        for convention in conventions or [BACKSCATTER_GAMMA0]:
            for label, summary in class_summary(rois, band, convention):
                rows.append([label, band, convention, summary["count"],
                             summary["mean_db"], summary["spread_db"],
                             summary["enl"]])
    return rows


def point_in_ring(ring, x, y):
    """Is (x, y) inside this ring? Even-odd, the same rule as the ROI mask.

    The same test polygon_mask applies to a grid, for one point: an ROI that
    holds a pixel's centre holds a click at that centre, so what you select is
    what you measured.
    """
    inside = False
    count = len(ring or ())
    if count < 3:
        return False
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def roi_at(rois, x, y):
    """The ROI under a point, or None.

    The smallest of the ones holding it, so a small ROI drawn inside a large
    one can still be picked -- otherwise the big one would swallow every click
    within it and the small one could only ever be reached from the table.
    """
    hits = [roi for roi in rois or ()
            if point_in_ring(roi.get("ring") or [], x, y)]
    if not hits:
        return None
    return min(hits, key=lambda roi: ring_area(roi.get("ring") or []))


def summarise(stats_list):
    """Across ROIs, for one band: brightness, uniformity, looks.

    The spread between the brightest and darkest ROI mean is the useful figure
    when the ROIs are patches of one cover type -- it is the product's
    radiometric uniformity over that ground, in dB. The median ENL is taken
    rather than the mean because one ROI landing on a field boundary drags an
    average down and cannot drag a median far.
    """
    means = [s["mean_db"] for s in stats_list
             if s and np.isfinite(s.get("mean_db", float("nan")))]
    enls = [s["enl"] for s in stats_list
            if s and np.isfinite(s.get("enl", float("nan")))]
    nan = float("nan")
    return {
        "count": len(means),
        "mean_db": float(np.mean(means)) if means else nan,
        "spread_db": float(max(means) - min(means)) if len(means) > 1 else nan,
        "enl": float(np.median(enls)) if enls else nan,
    }


# ── ROI GEOMETRY ──────────────────────────────────────────────────────────────
# A ring is a list of (x, y) in the working CRS, open: the closing edge runs
# from the last vertex back to the first and is never stored, so a ring has
# exactly as many vertices as were clicked.

def rect_ring(x0, y0, x1, y1):
    """The rectangle spanned by two corners, counter-clockwise from its origin."""
    xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
    ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def dedupe_ring(ring, tol):
    """Drop vertices within `tol` of the one before, and of the first.

    Closing a polygon by double-click hands the tool the same vertex twice --
    the press that precedes the double-click has already added it -- and a
    repeated vertex is a zero-length edge, which a scanline fill counts as a
    crossing and a shoelace area does not.
    """
    out = []
    for x, y in ring:
        if out and math.hypot(x - out[-1][0], y - out[-1][1]) <= tol:
            continue
        out.append((float(x), float(y)))
    while len(out) > 1 and math.hypot(out[-1][0] - out[0][0],
                                      out[-1][1] - out[0][1]) <= tol:
        out.pop()
    return out


def ring_bounds(ring):
    """(xmin, ymin, xmax, ymax), or None for a ring with no vertices."""
    if not ring:
        return None
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def ring_area(ring):
    """Absolute shoelace area, in the ring's own units squared."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def ring_centroid(ring):
    """Area-weighted centroid, falling back to the vertex mean when degenerate.

    The vertex mean is not the centroid of a polygon -- a run of closely spaced
    vertices along one edge pulls it there -- so it is only used when the
    shoelace area is zero and the real centroid does not exist.
    """
    if not ring:
        return None
    if len(ring) >= 3:
        cross_sum = cx = cy = 0.0
        for i, (x1, y1) in enumerate(ring):
            x2, y2 = ring[(i + 1) % len(ring)]
            cross = x1 * y2 - x2 * y1
            cross_sum += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if cross_sum != 0.0:
            return (cx / (3.0 * cross_sum), cy / (3.0 * cross_sum))
    return (sum(p[0] for p in ring) / len(ring),
            sum(p[1] for p in ring) / len(ring))


def pixel_window(bounds, origin_x, origin_y, px, py, width, height):
    """`bounds` as a whole-pixel window clipped to the raster, or None.

    (col0, row0, ncol, nrow), north-up: `origin_y` is the TOP edge and rows run
    downward. The window is grown outward to pixel boundaries so no pixel whose
    centre may fall inside the ROI is cut off before the mask is applied.
    """
    if bounds is None or px <= 0 or py <= 0:
        return None
    xmin, ymin, xmax, ymax = bounds
    col0 = int(math.floor((xmin - origin_x) / px))
    col1 = int(math.ceil((xmax - origin_x) / px))
    row0 = int(math.floor((origin_y - ymax) / py))
    row1 = int(math.ceil((origin_y - ymin) / py))
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, int(width)), min(row1, int(height))
    if col1 <= col0 or row1 <= row0:
        return None             # the ROI does not overlap the raster
    return (col0, row0, col1 - col0, row1 - row0)


def ring_to_pixels(ring, origin_x, origin_y, px, py, col0=0, row0=0):
    """A ring in map units as one in pixels, relative to a window's corner."""
    return [((x - origin_x) / px - col0, (origin_y - y) / py - row0)
            for x, y in ring]


def polygon_mask(ring_px, ncol, nrow):
    """Which pixels of a window fall inside a ring, by even-odd at the centre.

    A pixel is in the ROI when its CENTRE is, which is the same rule QGIS's
    zonal statistics and gdal_rasterize apply by default: it is unbiased over a
    large ROI and, unlike an any-touch rule, it cannot pull a neighbouring
    field's pixels into a small one.

    Vectorized per edge rather than per pixel: each edge toggles every pixel to
    the right of where it crosses that row, and an odd number of toggles leaves
    a pixel inside. A 40 x 40 ROI is 1600 pixels and this is not the expensive
    part of a measurement -- reading them off disk is.
    """
    ncol, nrow = int(ncol), int(nrow)
    mask = np.zeros((nrow, ncol), dtype=bool)
    if ncol <= 0 or nrow <= 0 or len(ring_px) < 3:
        return mask
    ys = np.arange(nrow, dtype=float) + 0.5
    xs = np.arange(ncol, dtype=float) + 0.5
    for i, (x1, y1) in enumerate(ring_px):
        x2, y2 = ring_px[(i + 1) % len(ring_px)]
        if y1 == y2:
            continue            # horizontal edges cross no row
        # Half-open in y, so a vertex exactly on a row's centre is counted once
        crosses = (y1 <= ys) != (y2 <= ys)
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        mask ^= crosses[:, None] & (xs[None, :] > at[:, None])
    return mask


# ── FIELD NAMING ──────────────────────────────────────────────────────────────

# A polarization as a band name declares it. Bounded by a non-letter or the
# ends of the name, so a word that merely contains the pair -- and a term that
# doubles it, as a GCOV diagonal does -- are told apart.
POL_IN_NAME_RE = re.compile(
    r"(?i)(?:^|[^A-Za-z])(HH|HV|VH|VV|RH|RV|LH|LV)"
    r"(?:HH|HV|VH|VV|RH|RV|LH|LV)?(?:$|[^A-Za-z])")


def band_prefix(label, index):
    """A short, safe column prefix for one band, from whatever it is called.

    The polarization is what an analyst reads the column by, so it is looked
    for first and kept whole: a GCOV diagonal term names the product of one
    polarization with itself -- HHHH is the HH power -- and a GeoTIFF written
    by cog_locate or DPQED_h52tif calls the same band 'gamma0_HH'. Both give
    HH, where truncating to the first six characters would give 'HHHH' and
    'gamma0' and lose which channel the second one was.

    A raster that names nothing falls back to its band number: 'b2_mean_db' at
    least says which band it came from.
    """
    text = str(label or "").strip()
    text = re.sub(r"^\s*\d+\s*:\s*", "", text)          # RIVAL's '2: HV'
    if not text or re.fullmatch(r"(?i)band\s*0*\d*", text):
        return f"b{int(index)}"
    pol = POL_IN_NAME_RE.search(text)
    if pol:
        return pol.group(1).upper()
    text = re.sub(r"[^0-9A-Za-z]+", "", text)
    if not text:
        return f"b{int(index)}"
    return text[:6].upper()


def dbf_field_names(prefixes, keys=STAT_KEYS, reserved=(),
                    limit=DBF_NAME_LIMIT):
    """{(prefix, key): column name}, each inside DBF's 10-character cap.

    DBF does not reject a long name, it truncates it -- and two truncations
    that collide become one column holding whichever was written last, with no
    error anywhere. Names are built to fit and de-duplicated here instead, and
    the companion CSV carries the untruncated ones for anyone who needs them.
    """
    taken = {str(name).lower() for name in reserved}
    out = {}
    for prefix in prefixes:
        for key in keys:
            base = f"{prefix}_{key}"[:limit]
            name, n = base, 1
            while name.lower() in taken:
                tail = str(n)
                name = base[:max(limit - len(tail), 1)] + tail
                n += 1
            taken.add(name.lower())
            out[(prefix, key)] = name
    return out


def polygon_wkt(ring):
    """A ring as WKT, closed, or None if it is not a polygon.

    Text rather than a geometry object because the writers that take it differ
    between QGIS versions and OGR builds, and every one of them parses WKT.
    """
    points = [(float(x), float(y)) for x, y in ring or []]
    if len(points) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    inner = ", ".join(f"{x:.10g} {y:.10g}" for x, y in points)
    return f"POLYGON(({inner}))"


def shapefile_schema(plan):
    """[(column name, kind)] for the attribute table, from the export plan."""
    stat_kinds = {key: kind for key, _, kind, _ in STAT_FIELDS}
    roi_kinds = dict(ROI_FIELDS)
    return [(name, roi_kinds[roi_key] if roi_key is not None
             else stat_kinds[stat_key])
            for name, roi_key, _, stat_key in plan]


def dbf_safe(value, kind):
    """A value a DBF column can actually hold: plain int, float, str or None.

    A numpy scalar, a NaN and an infinity all reach a writer that will either
    refuse the feature or store something that is not a number -- and none of
    them says so. A shapefile that quietly lost its attributes looks exactly
    like one that was never written, which is the failure this guards.
    """
    if value is None:
        return None
    if kind == "string":
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if kind == "int" else number


def roi_attributes(roi, plan):
    """One ROI's attributes, in the order the plan declared the fields.

    The order is the plan's and not a second traversal that happens to match:
    a shapefile's attributes are positional, so a field list and an attribute
    list built separately drift into each other's columns and the writer
    reports nothing wrong.
    """
    schema = shapefile_schema(plan)
    attributes = []
    for (_, kind), (_, roi_key, label, stat_key) in zip(schema, plan):
        if roi_key is not None:
            attributes.append(dbf_safe(roi.get(roi_key), kind))
        else:
            stats = roi_stats(roi, label) or {}
            attributes.append(dbf_safe(stats.get(stat_key), kind))
    return attributes


def shapefile_records(rois, plan):
    """[(wkt, attributes)] ready for any writer. ROIs with no ring are left out."""
    records = []
    for roi in rois or ():
        wkt = polygon_wkt(roi.get("ring"))
        if wkt is not None:
            records.append((wkt, roi_attributes(roi, plan)))
    return records


def stat_rows(rois, bands):
    """The ROI table in long form: one row per ROI and band, full names.

    Long rather than wide because it is what anything downstream wants -- a
    pivot table, a groupby, a plot of mean_db against band -- and because it
    does not have to be redesigned when a product carries four polarizations
    instead of two.
    """
    names = [name for name, _ in ROI_FIELDS]
    rows = [names + ["band"] + list(STAT_KEYS)]
    convention_at = names.index("backscat")
    for roi in rois:
        base = [roi.get(name) for name in names]
        for convention in roi_conventions(roi) or [roi.get("backscat")]:
            # The convention varies down the rows rather than across the
            # columns: one more row per ROI costs nothing, where one more set
            # of columns would need every statistic renamed to fit beside it.
            row_base = list(base)
            row_base[convention_at] = convention
            for band in bands:
                stats = roi_stats(roi, band, convention) or {}
                rows.append(row_base + [band]
                            + [stats.get(key) for key in STAT_KEYS])
    return rows


def format_stat(value, fmt="{:.3f}"):
    """A statistic as the table shows it; an undefined one as a dash."""
    try:
        if value is None:
            return "--"
        number = float(value)
        if not math.isfinite(number):
            return "--"
        return fmt.format(number)
    except (TypeError, ValueError):
        return "--"


# ── NISAR GCOV HDF5 ───────────────────────────────────────────────────────────

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


def _xml_escape(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def gcov_vrt_xml(sources, width, height, geotransform, srs, dtype="Float32",
                 nodata="nan"):
    """A VRT stacking GCOV covariance terms as bands, georeferenced.

    `sources` is [(band description, GDAL dataset name)] in band order. Written
    out rather than built with gdal.BuildVRT because the HDF5 subdatasets carry
    no geotransform of their own -- the grid states its corner in coordinate
    vectors beside the data -- and a VRT built from them would stack the bands
    correctly and place them nowhere.

    Nothing is copied: the VRT reads the HDF5 in place, so it costs a few
    hundred bytes and stays valid as long as the product sits where it is.
    """
    lines = [f'<VRTDataset rasterXSize="{int(width)}" '
             f'rasterYSize="{int(height)}">',
             f'  <SRS>{_xml_escape(srs)}</SRS>',
             '  <GeoTransform>' +
             ', '.join(f"{v:.16g}" for v in geotransform) + '</GeoTransform>']
    for index, (description, source) in enumerate(sources, start=1):
        lines += [
            f'  <VRTRasterBand dataType="{dtype}" band="{index}">',
            f'    <Description>{_xml_escape(description)}</Description>',
            f'    <NoDataValue>{nodata}</NoDataValue>',
            '    <SimpleSource>',
            '      <SourceFilename relativeToVRT="0">'
            f'{_xml_escape(source)}</SourceFilename>',
            '      <SourceBand>1</SourceBand>',
            f'      <SrcRect xOff="0" yOff="0" xSize="{int(width)}" '
            f'ySize="{int(height)}"/>',
            f'      <DstRect xOff="0" yOff="0" xSize="{int(width)}" '
            f'ySize="{int(height)}"/>',
            '    </SimpleSource>',
            '  </VRTRasterBand>',
        ]
    lines.append('</VRTDataset>')
    return "\n".join(lines) + "\n"


# ── END PURE HELPERS ──────────────────────────────────────────────────────────


# ── ROI NUMBER ON THE CANVAS ──────────────────────────────────────────────────
class RoiLabelItem(QgsMapCanvasItem):
    """The ROI's number, drawn at its centre and kept there.

    A canvas item rather than an annotation: the canvas repositions one of
    these itself on every pan and zoom, which is the whole difficulty with
    putting text on a map. An annotation would need that done by hand, and its
    API has moved more between QGIS versions than this one has.

    The number is drawn on a translucent backdrop because a thin glyph over a
    bright scatterer is not readable, and the whole point of the label is to be
    read at a glance.
    """

    def __init__(self, canvas, point, text, colour):
        super().__init__(canvas)
        self.canvas = canvas
        self.point = point
        self.text = str(text)
        self.colour = colour
        self.font = QFont()
        self.font.setBold(True)
        self.font.setPointSize(ROI_LABEL_POINT)
        self.updatePosition()

    def set_state(self, point, text, colour):
        """Move and re-letter the label without rebuilding it."""
        self.prepareGeometryChange()
        self.point, self.text, self.colour = point, str(text), colour
        self.updatePosition()
        try:
            self.show()     # it may have been hidden by the class filter
        except Exception:
            pass            # a build whose canvas item cannot hide never did
        self.update()

    def boundingRect(self):
        try:
            metrics = QFontMetricsF(self.font).boundingRect(self.text)
            width, height = metrics.width(), metrics.height()
        except Exception:
            # Font metrics need a running application. Estimate rather than
            # fail: a label the wrong size still says which ROI this is.
            width, height = 7.0 * max(len(self.text), 1), 14.0
        width += 2 * ROI_LABEL_PAD
        height += 2 * ROI_LABEL_PAD
        return QRectF(-width / 2.0, -height / 2.0, width, height)

    def paint(self, painter, option=None, widget=None):
        try:
            rect = self.boundingRect()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(*ROI_LABEL_BACKDROP))
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setPen(QColor(*self.colour))
            painter.setFont(self.font)
            painter.drawText(rect, Qt.AlignCenter, self.text)
        except Exception as e:
            print(f"[ROI] label: {e}")

    def updatePosition(self):
        try:
            self.setPos(self.toCanvasCoordinates(self.point))
        except Exception:
            pass


# ── ROI SELECTION TOOL ────────────────────────────────────────────────────────
class RoiSelectTool(QgsMapTool):
    """Click an ROI to select it, on the canvas and in the table at once.

    Drawing tools make ROIs and the table edits them, which left the canvas
    unable to answer 'that one' -- the thing you are looking at when you decide
    an ROI is wrong. Selecting here selects the row, so Delete and the editable
    cells apply to what was clicked.
    """

    def __init__(self, canvas, dashboard):
        super().__init__(canvas)
        self.canvas = canvas
        self.dashboard = dashboard
        self.setCursor(Qt.ArrowCursor)

    def canvasReleaseEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        point = self.toMapCoordinates(e.pos())
        self.dashboard.select_roi_at(point.x(), point.y())


# ── ROI DRAWING TOOL ──────────────────────────────────────────────────────────
class RoiMapTool(QgsMapTool):
    """Draw one ROI: a dragged rectangle, or a polygon clicked corner by corner.

    One class for both because they differ only in how the ring is collected --
    the preview, the cancel, and the handoff to the table are the same, and two
    classes would be two places to fix the next thing found wrong with either.

    A rectangle finishes on the mouse release. A polygon finishes on a
    right-click or a double-click, needs three corners, and takes Backspace to
    undo the last one; Escape abandons whatever is in progress.
    """

    def __init__(self, canvas, dashboard, mode):
        super().__init__(canvas)
        self.canvas = canvas
        self.dashboard = dashboard
        self.mode = mode
        self.anchor = None          # rectangle: where the drag started
        self.vertices = []          # polygon: the corners clicked so far
        self.band = None
        self.setCursor(Qt.CrossCursor)

    # ── preview ──
    def _rubber(self):
        if self.band is None:
            self.band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            colour = QColor(*ROI_COLOR_DRAWING)
            fill = QColor(*ROI_COLOR_DRAWING)
            fill.setAlpha(ROI_FILL_ALPHA)
            self.band.setColor(colour)
            self.band.setFillColor(fill)
            self.band.setWidth(ROI_WIDTH)
        return self.band

    def _preview(self, ring):
        band = self._rubber()
        band.reset(QgsWkbTypes.PolygonGeometry)
        for index, (x, y) in enumerate(ring):
            band.addPoint(QgsPointXY(x, y), index == len(ring) - 1)
        band.show()

    def _merge_tolerance(self):
        """Screen pixels as map units, for merging a double-click's two clicks."""
        try:
            return float(self.canvas.mapUnitsPerPixel()) * VERTEX_MERGE_PX
        except Exception:
            return 0.0

    # ── mouse ──
    def canvasPressEvent(self, e):
        point = self.toMapCoordinates(e.pos())
        if self.mode == TOOL_RECT:
            if e.button() == Qt.LeftButton:
                self.anchor = (point.x(), point.y())
            return
        if e.button() == Qt.RightButton:
            self.finish()
            return
        self.vertices.append((point.x(), point.y()))
        self._preview(self.vertices)

    def canvasMoveEvent(self, e):
        point = self.toMapCoordinates(e.pos())
        if self.mode == TOOL_RECT:
            if self.anchor is not None:
                self._preview(rect_ring(self.anchor[0], self.anchor[1],
                                        point.x(), point.y()))
        elif self.vertices:
            self._preview(self.vertices + [(point.x(), point.y())])

    def canvasReleaseEvent(self, e):
        if self.mode != TOOL_RECT or self.anchor is None:
            return
        if e.button() != Qt.LeftButton:
            return
        point = self.toMapCoordinates(e.pos())
        ring = rect_ring(self.anchor[0], self.anchor[1], point.x(), point.y())
        self.cancel()
        self.dashboard.add_roi(ring, "rect")

    def canvasDoubleClickEvent(self, e):
        if self.mode == TOOL_POLY:
            self.finish()

    def keyPressEvent(self, e):
        try:
            key = e.key()
        except Exception:
            return
        if key == Qt.Key_Escape:
            self.cancel()
        elif key == Qt.Key_Backspace and self.mode == TOOL_POLY and self.vertices:
            self.vertices.pop()
            self._preview(self.vertices)

    # ── lifecycle ──
    def finish(self):
        """Close the polygon being drawn and hand it to the table."""
        ring = dedupe_ring(self.vertices, self._merge_tolerance())
        self.cancel()
        if len(ring) >= 3:
            self.dashboard.add_roi(ring, "polygon")
        elif ring:
            print(f"[ROI] {len(ring)} corner(s) is not a polygon; discarded")

    def cancel(self):
        self.anchor = None
        self.vertices = []
        if self.band is not None:
            self.band.reset(QgsWkbTypes.PolygonGeometry)

    def deactivate(self):
        # A half-drawn polygon left behind would reappear, with corners from
        # one view, the moment this tool was picked up again.
        self.cancel()
        try:
            super().deactivate()
        except Exception:
            pass


class OverlayResizeFilter(QObject):
    """Keep a translucent overlay spanning the canvas it sits on."""

    def __init__(self, parent, container, height=35):
        super().__init__(parent)
        self.container = container
        self.parent_widget = parent
        self.height = height

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self.container.setGeometry(10, 10, self.parent_widget.width() - 20,
                                       self.height)
        return super().eventFilter(obj, event)


# ── MAIN WINDOW ───────────────────────────────────────────────────────────────
class RadiometricDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RADIAL - Radiometric Image Assessment and Logger")
        self.resize(1500, 950)

        self.raster_layer = None
        self.raster_path = ""
        self.band_labels = []           # what each band is called, in band order
        # The bands that are backscatter, as (GDAL band number, label): the RTC
        # factor is a band of the raster and not one of these.
        self.measure_bands = []
        self.band_prefixes = []         # the column prefix each one exports under
        self.factor_band = None         # GDAL band number of the RTC factor
        self.incidence_band = None      # GDAL band number of the incidence angle
        # (x, y, degrees) read from a '.h5' product's radarGrid, for when the
        # raster carries no incidence band of its own.
        self.incidence_cube = None
        self.rois = []                  # plain data; see add_roi for the shape
        self.roi_bands = {}             # id -> QgsRubberBand drawn on the canvas
        self.roi_labels = {}            # id -> its number, drawn at its centre
        self._next_roi_id = 1
        self._stretch_cache = {}
        # Bounds pinned by Normalize, per band. Set only when the button is
        # pressed and kept until it is pressed again or another raster loads:
        # a scene that re-stretches while an ROI is being drawn over it is a
        # scene whose two halves were judged against different pictures.
        self.norm_bounds = {}
        self._filling_table = False

        self.wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        self.proj_crs = QgsCoordinateReferenceSystem(WORKING_CRS_DEFAULT)
        self._rebuild_transforms()
        self._configure_gdal_cache()

        self.canvas = QgsMapCanvas()
        self.canvas.enableAntiAliasing(False)
        self.canvas.setCachingEnabled(True)
        self.canvas.setParallelRenderingEnabled(True)

        # Normalize and the band picker ride on the canvas, not in the button
        # row: they act on what is in view, so they belong with the thing they
        # measure. The row below is for actions on the ROIs.
        self.overlay = QWidget(self.canvas)
        self.overlay.setGeometry(10, 10, 460, 35)
        self.overlay.setStyleSheet("background-color: rgba(255,255,255,153);")
        overlay_layout = QHBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(5, 5, 5, 5)
        self.btn_norm = self._normalize_button(
            "Stretch the view to what is currently in it  (Ctrl+R).\n"
            "The result is kept until you press it again -- panning and\n"
            "zooming do not re-stretch, so the scene an ROI was drawn on is\n"
            "the same scene when you come back to it.\n\n"
            "Rendering only: ROI statistics are read from the source pixels.")
        self.clip_combo = self._clip_combo()
        overlay_layout.addWidget(self.btn_norm)
        overlay_layout.addWidget(self.clip_combo)
        overlay_layout.addWidget(QLabel("R G B"))
        self.band_combos = []
        for channel in ("red", "green", "blue"):
            combo = QComboBox()
            combo.setToolTip(
                f"Band shown as {channel}.\nSet all three for a composite; "
                f"the same band in all three renders grey.")
            overlay_layout.addWidget(combo, 1)
            self.band_combos.append(combo)
        self.overlay.hide()
        self.overlay_filter = OverlayResizeFilter(self.canvas, self.overlay)
        self.canvas.installEventFilter(self.overlay_filter)

        # ── the ROI table, and one ROI in full beside it ──
        self.table = QTableWidget(0, len(ROI_TABLE_COLUMNS) + len(TABLE_STATS))
        self.table.setHorizontalHeaderLabels(self._table_headers())
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.setToolTip(
            "One row per ROI, for the band selected below.\n"
            "Double-click a name to label it; the label is exported.")

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMinimumWidth(430)
        self.detail.setMaximumWidth(560)
        detail_font = QFont("Monospace")
        detail_font.setStyleHint(QFont.TypeWriter)
        detail_font.setPointSize(9)
        self.detail.setFont(detail_font)
        self.detail.setPlaceholderText(
            "Every band of the selected ROI, in full, ready to paste into a "
            "report.")

        # ── buttons ──
        self.btn_load = QPushButton("Load GCOV")
        self.btn_load.setToolTip(
            "Load a GCOV raster: a GeoTIFF or VRT, or a NISAR '.h5', which is\n"
            "wrapped in a VRT beside it  (Ctrl+L).\n\n"
            "ROIs are kept and re-measured against the new raster -- that is\n"
            "how two products are compared over the same ground.")
        self.btn_load_rois = QPushButton("Load ROIs")
        self.btn_load_rois.setToolTip(
            "Read ROIs back from a polygon shapefile and measure them here "
            "(Ctrl+O).")
        self.btn_shp = QPushButton("Export SHP")
        self.btn_shp.setToolTip(
            "Write the ROIs as polygons with every statistic  (Ctrl+Shift+S).\n"
            "The same numbers are written beside it as '<stem>_stats.csv',\n"
            "under names DBF's 10-character cap cannot hold.")
        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.setToolTip(
            "Write the statistics alone, one row per ROI and band  (Ctrl+S).")
        self.btn_del = QPushButton("Delete ROI")
        self.btn_del.setToolTip("Delete the selected ROI  (Ctrl+Delete)")
        self.btn_clear = QPushButton("Clear ROIs")
        self.btn_clear.setToolTip("Delete every ROI")

        self.tool_buttons = {}
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        tips = {
            TOOL_SELECT: "Click an ROI to select it, here and in the table "
                         "below  (Ctrl+1).\nThe smallest ROI under the click "
                         "wins, so one drawn inside\nanother is still "
                         "reachable. Click empty ground to deselect.",
            TOOL_RECT: "Drag a rectangle over the target  (Ctrl+2)",
            TOOL_POLY: "Click the corners; right-click or double-click to "
                       "close, Backspace undoes one  (Ctrl+3)",
            TOOL_PAN: "Drag to move the view  (Ctrl+4)",
            TOOL_ZOOM_IN: "Drag a box, or click, to zoom in  (Ctrl+5)",
            TOOL_ZOOM_OUT: "Drag a box, or click, to zoom out  (Ctrl+6)",
        }
        labels = {TOOL_SELECT: "Select", TOOL_RECT: "Rect",
                  TOOL_POLY: "Polygon", TOOL_PAN: "Pan",
                  TOOL_ZOOM_IN: "Zoom In", TOOL_ZOOM_OUT: "Zoom Out"}
        for index, mode in enumerate(MAP_TOOLS):
            button = QPushButton(labels[mode])
            button.setCheckable(True)
            button.setToolTip(tips[mode])
            self.tool_group.addButton(button, index)
            self.tool_buttons[mode] = button
        self.tool_buttons[TOOL_RECT].setChecked(True)

        self.stats_band_combo = QComboBox()
        self.stats_band_combo.setToolTip(
            "Which band the table and the summary below report.\n"
            "Both exports carry every band whatever this says.")
        self.domain_combo = QComboBox()
        for value, label in DOMAIN_CHOICES:
            self.domain_combo.addItem(label, value)
        self.domain_combo.setCurrentIndex(
            [value for value, _ in DOMAIN_CHOICES].index(DOMAIN_DEFAULT))
        self.domain_combo.setToolTip(
            "What the pixels hold. Everything is computed on linear power, so\n"
            "this is what they are converted FROM -- get it wrong and the dB\n"
            "figures and the looks estimate are both wrong, silently.\n\n"
            "NISAR GCOV carries gamma0 as power. A GSLC magnitude is\n"
            "amplitude. Only pick dB for a raster already in dB.")
        self.backscatter_combo = QComboBox()
        for value, label in BACKSCATTER_CHOICES:
            self.backscatter_combo.addItem(label, value)
        self.backscatter_combo.setCurrentIndex(0)
        self.backscatter_combo.setEnabled(False)
        self.backscatter_combo.setToolTip(
            "Which backscatter convention the statistics are in.\n\n"
            "GCOV stores gamma0, referred to the terrain's own sloped area.\n"
            "sigma0 refers it to flat ground instead, using the product's own\n"
            "per-pixel rtcGammaToSigmaFactor -- several dB apart on a slope,\n"
            "identical on the flat.\n\n"
            "This is NOT the Domain selector beside it. Domain says what the\n"
            "pixels hold -- power, amplitude, dB -- and never changes the\n"
            "convention; this does, and only this.\n\n"
            "Selectable only when the raster carries the factor: a GCOV loaded\n"
            "from its '.h5' does, and a GeoTIFF does when it was written by\n"
            "DPQED_gcov2tif.py. Every export records which one it is.")

        # One dropdown doing two jobs, which is deliberate: it says which
        # class you are working on, and working on a class means both drawing
        # into it and looking at it alone. Two controls for that would let them
        # disagree -- drawing water while reading vegetation -- and nothing on
        # screen would say which one the numbers below belonged to.
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        self.class_combo.addItems([ROI_CLASS_ALL] + list(ROI_CLASSES))
        self.class_combo.setCurrentIndex(
            1 + list(ROI_CLASSES).index(ROI_CLASS_DEFAULT))
        self.class_combo.setToolTip(
            "The class you are working on: the canvas and the table show its\n"
            "ROIs alone, and a new ROI is drawn into it.\n\n"
            f"'{ROI_CLASS_ALL}' shows every ROI. Nothing can be drawn there,\n"
            "because there would be no saying what class it landed in.\n\n"
            "Filtering changes what you see, never what you have: every\n"
            "export writes all the ROIs, whatever the filter is set to.\n\n"
            "Statistics are reported per class, because a spread taken across\n"
            "water and vegetation together is not the product's uniformity --\n"
            "it is the difference between two land covers.\n\n"
            "Editable: type anything and it becomes a class. The class of an\n"
            "ROI already drawn is editable in its table row.")
        self.class_combo.currentTextChanged.connect(self.on_class_filter)

        self.cb_zero_data = QCheckBox("Zeros are data")
        self.cb_zero_data.setChecked(not ZERO_IS_NODATA)
        self.cb_zero_data.setToolTip(
            "Off (the default): zero is fill, as a SAR product means it, and\n"
            "is left out of the statistics. On: zero is a measurement.\n"
            "Ignored for a raster in dB, where 0 dB is a power of 1.")

        # One set of figures for a mixed population of ROIs answered the wrong
        # question, so the footer's three labels became a row per class. What
        # is left in the footer is context: which band and convention the
        # numbers above are in.
        summary_font = QFont()
        summary_font.setBold(True)
        summary_font.setPointSize(10)
        self.lbl_context = QLabel("--")
        self.lbl_context.setFont(summary_font)
        self.lbl_context.setAlignment(Qt.AlignCenter)
        self.lbl_context.setFrameShape(QFrame.StyledPanel)
        self.lbl_context.setStyleSheet(
            "QLabel {"
            "  background-color: #1e1e2e;"
            "  color: #cdd6f4;"
            "  border: 1px solid #45475a;"
            "  border-radius: 4px;"
            "  padding: 4px 10px;"
            "}")

        self.class_table = QTableWidget(0, len(CLASS_TABLE_COLUMNS))
        self.class_table.setHorizontalHeaderLabels(list(CLASS_TABLE_COLUMNS))
        self.class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.class_table.setMaximumHeight(170)
        self.class_table.setToolTip(
            "The selected band, summarised within each class.\n\n"
            "Spread is the brightest ROI mean minus the darkest: over patches\n"
            "of one cover type that is the product's radiometric uniformity.\n"
            "ENL is the median, so one ROI on a boundary cannot set it.\n\n"
            "The 'all' row is the scene-wide brightness, and is not a\n"
            "uniformity figure once more than one land cover is in it.")

        tool_row = QHBoxLayout()
        for mode in MAP_TOOLS:
            tool_row.addWidget(self.tool_buttons[mode])
        tool_row.addSpacing(20)
        tool_row.addWidget(QLabel("Stats band:"))
        tool_row.addWidget(self.stats_band_combo)
        tool_row.addSpacing(12)
        tool_row.addWidget(QLabel("Domain:"))
        tool_row.addWidget(self.domain_combo)
        tool_row.addSpacing(12)
        tool_row.addWidget(QLabel("Backscatter:"))
        tool_row.addWidget(self.backscatter_combo)
        tool_row.addWidget(self.cb_zero_data)
        tool_row.addSpacing(12)
        tool_row.addWidget(QLabel("Drawing:"))
        tool_row.addWidget(self.class_combo)
        tool_row.addStretch()

        action_row = QHBoxLayout()
        for widget in (self.btn_load, self.btn_load_rois, self.btn_shp,
                       self.btn_csv, self.btn_del, self.btn_clear):
            action_row.addWidget(widget)
        action_row.addStretch()
        action_row.addWidget(self.lbl_context)

        right = QVBoxLayout()
        right.addWidget(self.class_table, 1)
        right.addWidget(self.detail, 2)
        bottom = QHBoxLayout()
        bottom.addWidget(self.table, 3)
        bottom.addLayout(right, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas, 4)
        main_layout.addLayout(tool_row)
        main_layout.addLayout(action_row)
        main_layout.addLayout(bottom, 2)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.btn_load.clicked.connect(self.load_raster)
        self.btn_load_rois.clicked.connect(self.load_rois)
        self.btn_shp.clicked.connect(self.export_shapefile)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_del.clicked.connect(self.delete_roi)
        self.btn_clear.clicked.connect(self.clear_rois)
        self.btn_norm.clicked.connect(self.normalize_to_view)
        self.tool_group.buttonClicked.connect(lambda _: self.apply_map_tool())
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.itemChanged.connect(self.on_item_changed)
        self.stats_band_combo.currentIndexChanged.connect(
            lambda _: self.refresh_table())
        self.domain_combo.currentIndexChanged.connect(
            lambda _: self.recompute_all("domain changed"))
        self.backscatter_combo.currentIndexChanged.connect(
            lambda _: self.apply_backscatter())
        self.cb_zero_data.stateChanged.connect(
            lambda _: self.recompute_all("zero handling changed"))
        for combo in self.band_combos:
            combo.currentIndexChanged.connect(lambda _: self.apply_bands())

        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self.load_raster)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.load_rois)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.export_csv)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(
            self.export_shapefile)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(
            self.normalize_to_view)
        QShortcut(QKeySequence("Ctrl+Delete"), self).activated.connect(
            self.delete_roi)
        QShortcut(QKeySequence("F5"), self).activated.connect(self.zoom_to_roi)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self.cancel_drawing)
        for index, mode in enumerate(MAP_TOOLS, start=1):
            QShortcut(QKeySequence(f"Ctrl+{index}"), self).activated.connect(
                lambda m=mode: self.set_map_tool(m))

        self.adopt_existing_layer()
        self.init_map_tools()
        self.update_summary()

    # ── small builders ──────────────────────────────────────────────────────
    @staticmethod
    def _table_headers():
        headers = {"roi": "ROI", "name": "Name", "class": "Class",
                   "kind": "Kind", "npix": "Pixels", "area_m2": "Area m²",
                   "inc_deg": "Inc \u00b0"}
        titles = {key: title for key, title, _, _ in STAT_FIELDS}
        return ([headers[key] for key in ROI_TABLE_COLUMNS]
                + [titles[key] for key in TABLE_STATS])

    @staticmethod
    def _clip_combo():
        """The percentage Normalize clips off each end."""
        combo = QComboBox()
        combo.setFixedWidth(64)
        for pct in NORM_CLIP_CHOICES:
            combo.addItem(f"{pct:g}%", pct)
        combo.setCurrentIndex(NORM_CLIP_CHOICES.index(NORM_CLIP_DEFAULT))
        combo.setToolTip(
            "Percentage clipped off each end when Normalize measures.\n"
            "2% ignores the brightest and darkest 2%, so a handful of bright\n"
            "scatterers cannot set the whole scene. 0% is a true min/max.")
        combo.setStyleSheet(
            "QComboBox { background-color: rgba(255,255,255,170); }")
        return combo

    @staticmethod
    def _normalize_button(tooltip):
        button = QPushButton("Normalize")
        button.setToolTip(tooltip)
        button.setFixedWidth(90)
        button.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,170);"
            " border: 1px solid rgba(0,0,0,90); border-radius: 3px;"
            " padding: 2px 6px; }"
            "QPushButton:hover { background-color: rgba(255,255,255,215); }")
        return button

    def clip_percent(self):
        try:
            value = self.clip_combo.currentData()
            if value is None:
                value = float(self.clip_combo.currentText().rstrip("%"))
            return float(value)
        except Exception:
            return NORM_CLIP_DEFAULT

    def domain(self):
        """What the pixels hold, as one of the DOMAIN_* values."""
        try:
            value = self.domain_combo.currentData()
            if value in (DOMAIN_POWER, DOMAIN_AMPLITUDE, DOMAIN_DB):
                return value
        except Exception:
            pass
        return DOMAIN_DEFAULT

    def class_filter(self):
        """The class on show, which is ROI_CLASS_ALL when that is all of them."""
        try:
            text = str(self.class_combo.currentText()).strip()
            if text:
                return text
        except Exception:
            pass
        return ROI_CLASS_ALL

    def roi_class(self):
        """The class the next ROI drawn will carry."""
        text = self.class_filter()
        if class_key(text) == class_key(ROI_CLASS_ALL):
            return ROI_CLASS_DEFAULT        # the drawing tools are off anyway
        return text

    def shown_rois(self):
        """The ROIs the filter leaves on view, in the table's row order."""
        return visible_rois(self.rois, self.class_filter())

    def on_class_filter(self, _text=None):
        """The filter changed: redraw the table, the canvas and the tools."""
        self.clear_selection()
        self.refresh_table()
        self.apply_class_tools()

    def apply_class_tools(self):
        """With every class on show there is no class to draw into.

        Disabling the two drawing tools is the honest version of this. The
        alternative -- letting an ROI be drawn and picking a class for it --
        puts an ROI in a class the user never chose, and the table would show
        it as though they had.
        """
        drawing = class_key(self.class_filter()) != class_key(ROI_CLASS_ALL)
        buttons = getattr(self, "tool_buttons", None) or {}
        for mode in (TOOL_RECT, TOOL_POLY):
            button = buttons.get(mode)
            if button is None:
                continue
            button.setEnabled(drawing)
            if not drawing and button.isChecked():
                select = buttons.get(TOOL_SELECT)
                if select is not None:
                    select.setChecked(True)
        self.apply_map_tool()

    def backscatter(self):
        """gamma0 as stored, or sigma0 through the RTC factor."""
        try:
            value = self.backscatter_combo.currentData()
            if value in (BACKSCATTER_GAMMA0, BACKSCATTER_SIGMA0):
                return value
        except Exception:
            pass
        return BACKSCATTER_DEFAULT

    def zero_is_nodata(self):
        try:
            return not bool(self.cb_zero_data.isChecked())
        except Exception:
            return ZERO_IS_NODATA

    def _configure_gdal_cache(self):
        try:
            from osgeo import gdal
            gdal.SetCacheMax(512 * 1024 * 1024)
            gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
            gdal.SetConfigOption("VSI_CACHE", "TRUE")
            gdal.SetConfigOption("VSI_CACHE_SIZE", "20000000")
            gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
        except Exception as e:
            print(f"GDAL cache config skipped: {e}")

    def ensure_overviews(self, path):
        """Build overviews in the background, so panning a big GCOV is usable."""
        def _build():
            ds = None
            try:
                from osgeo import gdal
                ds = gdal.Open(path, gdal.GA_ReadOnly)
                if (ds and ds.GetRasterBand(1) and
                        ds.GetRasterBand(1).GetOverviewCount() == 0):
                    print(f"[OVR] Building: {os.path.basename(path)} ...")
                    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
                    print("[OVR] Done.")
            except Exception as e:
                print(f"[OVR] Skipped for {os.path.basename(path)}: {e}")
            finally:
                ds = None
        threading.Thread(target=_build, daemon=True).start()

    # ── CRS ──────────────────────────────────────────────────────────────────
    def _rebuild_transforms(self):
        self.transform_proj_to_wgs = QgsCoordinateTransform(
            self.proj_crs, self.wgs84_crs, QgsProject.instance())
        self.transform_wgs_to_proj = QgsCoordinateTransform(
            self.wgs84_crs, self.proj_crs, QgsProject.instance())
        self._apply_canvas_crs()

    def _apply_canvas_crs(self):
        """Pin the canvas to the working CRS.

        A bare QgsMapCanvas inherits the project's CRS, so an ROI could be
        drawn in whatever CRS the project happened to carry while its area was
        quoted in the raster's -- wrong numbers, and no error anywhere.
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        try:
            canvas.setDestinationCrs(self.proj_crs)
            print(f"[CRS] canvas "
                  f"{self.proj_crs.authid() or self.proj_crs.description()}")
        except Exception as e:
            print(f"[CRS] could not pin canvas CRS: {e}")

    def adopt_working_crs(self, crs):
        """Take the working CRS from the raster, bringing the ROIs with it.

        ROI geometry, the exported shapefile and the area column all live in
        this CRS, so a hard-coded zone is wrong the moment a scene sits in
        another one. Existing ROIs are reprojected rather than left behind: the
        whole point of keeping them across a load is that they describe the
        same ground in the next product.

        Geographic CRSs are refused. An area in square degrees is not an area.
        """
        if crs is None or not crs.isValid() or crs.isGeographic():
            return False
        if crs.authid() and crs.authid() == self.proj_crs.authid():
            return False
        old_crs, old_name = self.proj_crs, (self.proj_crs.authid()
                                            or self.proj_crs.description())
        self.proj_crs = crs
        self._rebuild_transforms()
        self._reproject_rois(old_crs, crs)
        print(f"[CRS] Working CRS {old_name} -> "
              f"{crs.authid() or crs.description()} (from the raster)")
        return True

    def _reproject_rois(self, old_crs, new_crs):
        if not self.rois:
            return
        try:
            transform = QgsCoordinateTransform(old_crs, new_crs,
                                               QgsProject.instance())
        except Exception as e:
            print(f"[CRS] ROIs could not be reprojected: {e}")
            return
        for roi in self.rois:
            moved = []
            for x, y in roi["ring"]:
                point = transform.transform(QgsPointXY(x, y))
                moved.append((point.x(), point.y()))
            roi["ring"] = moved
        print(f"[CRS] {len(self.rois)} ROI(s) reprojected")

    def _to_lonlat(self, x, y):
        try:
            point = self.transform_proj_to_wgs.transform(QgsPointXY(x, y))
            return (point.x(), point.y())
        except Exception:
            return (None, None)

    def _ring_in_layer_crs(self, ring):
        """An ROI ring in the raster's own CRS, for reading its pixels."""
        layer = self.raster_layer
        if layer is None:
            return ring
        try:
            src, dst = self.proj_crs, layer.crs()
            if src.authid() and dst.authid() and src.authid() == dst.authid():
                return ring
            transform = QgsCoordinateTransform(src, dst, QgsProject.instance())
            return [(lambda p: (p.x(), p.y()))(transform.transform(
                QgsPointXY(x, y))) for x, y in ring]
        except Exception as e:
            print(f"[ROI] ring transform: {e}")
            return ring

    # ── RASTER LOADING ───────────────────────────────────────────────────────
    @staticmethod
    def _open_filter():
        """The file dialog's filter, from the extensions the loader handles."""
        rasters = " ".join(f"*{ext}" for ext in RASTER_EXTS)
        hdf5 = " ".join(f"*{ext}" for ext in H5_EXTS)
        return (f"GCOV raster or HDF5 ({rasters} {hdf5});;"
                f"Raster ({rasters});;NISAR HDF5 ({hdf5});;All files (*)")

    def load_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a GCOV raster or NISAR HDF5", "", self._open_filter())
        if path:
            self.load_path(path)

    def load_path(self, path):
        """Load a raster, keeping the ROIs and re-measuring them against it."""
        source = path
        # A cube belongs to the product it came from, so it goes when another
        # raster arrives -- otherwise the last GCOV's geometry would be
        # reported against whatever is loaded next.
        self.incidence_cube = None
        if path.lower().endswith(H5_EXTS):
            source = self.gcov_vrt_for(path)
            if not source:
                return False
        layer = QgsRasterLayer(source, os.path.basename(source))
        if not layer.isValid():
            QMessageBox.critical(
                self, "Load", f"QGIS could not open:\n{source}")
            return False
        if self.raster_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.raster_layer.id())
            except Exception:
                pass
        QgsProject.instance().addMapLayer(layer, False)
        self.raster_layer = layer
        self.raster_path = source
        self.ensure_overviews(source)
        self.adopt_working_crs(layer.crs())
        self.canvas.setLayers([layer])
        self.canvas.setExtent(layer.extent())
        self.populate_band_picker(layer)
        self.canvas.refresh()
        print(f"[LOAD] {source}")
        if self.rois:
            self.recompute_all(f"loaded {os.path.basename(source)}")
        else:
            self.refresh_table()
        return True

    def adopt_existing_layer(self):
        """Take a raster already in the project, so the tool opens on it."""
        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                  if isinstance(lyr, QgsRasterLayer) and lyr.isValid()]
        if not layers:
            return
        layer = layers[0]
        self.raster_layer = layer
        self.raster_path = layer.source()
        self.ensure_overviews(layer.source())
        self.adopt_working_crs(layer.crs())
        self.canvas.setLayers([layer])
        self.canvas.setExtent(layer.extent())
        self.populate_band_picker(layer)
        self.canvas.refresh()
        print(f"[LOAD] adopted {layer.name()} from the project")

    # ── NISAR GCOV HDF5 -> VRT ───────────────────────────────────────────────
    def gcov_vrt_for(self, h5_path):
        """Wrap a GCOV's covariance terms in a VRT beside it, and return it.

        The HDF5 subdatasets carry no georeferencing of their own -- the grid
        states its corner in coordinate vectors sitting beside the data -- so
        the VRT is written out rather than built by gdal.BuildVRT, which would
        stack the bands correctly and place them nowhere.

        Nothing is copied. The VRT is a few hundred bytes that read the HDF5 in
        place, so it costs nothing to leave beside the product and re-use.
        """
        try:
            import h5py
        except Exception as e:
            QMessageBox.critical(
                self, "GCOV",
                "Reading a NISAR '.h5' needs h5py, which is not available "
                f"here:\n{e}\n\nConvert the product with DPQED_h52tif.py and "
                "load the GeoTIFF instead.")
            return None
        try:
            with h5py.File(h5_path, "r") as handle:
                paths = []
                handle.visit(lambda name: paths.append(name))
                grids = gcov_grids(paths)
                if not grids:
                    QMessageBox.critical(
                        self, "GCOV",
                        "No GCOV grids in this file. RADIAL reads GCOV "
                        "products;\na GSLC or RSLC has to be converted with "
                        "DPQED_h52tif.py first.")
                    return None
                # Frequency A is the wideband channel and is what a GCOV
                # carries when it carries one; B is stated when it is chosen.
                key = sorted(grids, key=lambda k: (k[1] != "A", k))[0]
                terms = gcov_diagonal_terms(grids[key])
                if not terms:
                    QMessageBox.critical(
                        self, "GCOV",
                        f"{key[0]} frequency{key[1]} holds no diagonal "
                        "covariance term.\nOnly HHHH, HVHV, VHVH and VVVV are "
                        "backscatter; the off-diagonal\nterms are complex "
                        "correlations and are not measured here.")
                    return None
                group = gcov_group_of(next(iter(grids[key].values())))
                # The RTC factor rides along when the product has it, so
                # sigma0 is available without a second file to keep aligned
                # with this one. It is a band of the raster and not one of the
                # measured ones -- factor_band_index takes it back out.
                extras = [name for name in handle[group]
                          if RTC_FACTOR_RE.search(name)]
                first = handle[f"{group}/{terms[0]}"]
                height, width = int(first.shape[0]), int(first.shape[1])
                dtype = {"float32": "Float32", "float64": "Float64"}.get(
                    str(first.dtype), "Float32")
                geotransform = geotransform_from_coords(
                    handle[f"{group}/xCoordinates"][()],
                    handle[f"{group}/yCoordinates"][()])
                epsg = int(np.asarray(handle[f"{group}/projection"][()]).ravel()[0])
                self.incidence_cube = self._read_incidence_cube(handle, paths,
                                                                key[0])
        except KeyError as e:
            QMessageBox.critical(
                self, "GCOV",
                f"The grid is missing {e}, which the georeferencing needs.")
            return None
        except Exception as e:
            QMessageBox.critical(self, "GCOV", f"Could not read the HDF5:\n{e}")
            return None

        sources = [(term, f'HDF5:"{h5_path}"://{group}/{term}')
                   for term in terms + extras]
        xml = gcov_vrt_xml(sources, width, height, geotransform, f"EPSG:{epsg}",
                           dtype)
        vrt_path = os.path.splitext(h5_path)[0] + "_gcov.vrt"
        try:
            with open(vrt_path, "w", encoding="utf-8") as fh:
                fh.write(xml)
        except OSError as e:
            import tempfile
            vrt_path = os.path.join(
                tempfile.gettempdir(),
                os.path.splitext(os.path.basename(h5_path))[0] + "_gcov.vrt")
            print(f"[GCOV] {e}; writing the VRT to {vrt_path} instead")
            try:
                with open(vrt_path, "w", encoding="utf-8") as fh:
                    fh.write(xml)
            except OSError as inner:
                QMessageBox.critical(self, "GCOV",
                                     f"Could not write a VRT:\n{inner}")
                return None
        print(f"[GCOV] {key[0]} frequency{key[1]}: {', '.join(terms)}"
              + (f" (+ {', '.join(extras)})" if extras else "")
              + f" ({width} x {height}, EPSG:{epsg}) -> {vrt_path}")
        return vrt_path

    @staticmethod
    def _read_incidence_cube(handle, paths, band):
        """(x, y, degrees) from a product's radarGrid, or None.

        The layer nearest the ellipsoid is taken. Choosing between a cube's
        heights properly needs a DEM, which this does not have; across the
        heights a cube spans the angle moves far less than the difference
        between two ROIs, so the nearest layer is used and the height is
        printed rather than left implicit.
        """
        cubes = radar_grid_datasets(paths, band)
        if INCIDENCE_NAME not in cubes:
            return None
        try:
            values = np.asarray(handle[cubes[INCIDENCE_NAME]][()], dtype=float)
            if values.ndim == 3:
                index = 0
                if "heightAboveEllipsoid" in cubes:
                    heights = np.asarray(
                        handle[cubes["heightAboveEllipsoid"]][()],
                        dtype=float).ravel()
                    index = int(np.argmin(np.abs(heights)))
                    print(f"[GCOV] incidence angle at {heights[index]:.0f} m "
                          f"above the ellipsoid")
                values = values[index]
            cube = (np.asarray(handle[cubes["xCoordinates"]][()], dtype=float),
                    np.asarray(handle[cubes["yCoordinates"]][()], dtype=float),
                    values)
            print(f"[GCOV] incidence angle cube {values.shape}: ROI centres "
                  f"will be interpolated from it")
            return cube
        except Exception as e:
            print(f"[GCOV] incidence cube unreadable: {e}")
            return None

    # ── NORMALIZE: SAR SQRT-GAMMA STRETCH ────────────────────────────────────
    def view_extent(self):
        """The canvas's view as a rectangle in the raster's own CRS."""
        layer = self.raster_layer
        if layer is None:
            return None
        try:
            extent = self.canvas.extent()
        except Exception:
            return None
        try:
            src, dst = self.proj_crs, layer.crs()
            if src.authid() and dst.authid() and src.authid() == dst.authid():
                return extent
            return QgsCoordinateTransform(
                src, dst, QgsProject.instance()).transformBoundingBox(extent)
        except Exception as e:
            print(f"[NORM] view extent: {e}")
            return None

    def view_pixel_window(self, extent):
        """`extent`, in the layer's CRS, as a raster pixel window, or None."""
        layer = self.raster_layer
        if extent is None or layer is None:
            return None
        try:
            full = layer.extent()
            px = abs(float(layer.rasterUnitsPerPixelX()))
            py = abs(float(layer.rasterUnitsPerPixelY()))
            if px <= 0 or py <= 0:
                return None
            x0 = int((extent.xMinimum() - full.xMinimum()) / px)
            y0 = int((full.yMaximum() - extent.yMaximum()) / py)
            w = int(max(extent.xMaximum() - extent.xMinimum(), px) / px)
            h = int(max(extent.yMaximum() - extent.yMinimum(), py) / py)
            return (x0, y0, max(w, 1), max(h, 1))
        except Exception as e:
            print(f"[NORM] pixel window: {e}")
            return None

    def normalize_range(self, provider, band, extent=None):
        """The range to apply the gamma over, read from the raster itself.

        Full min/max of a bounded sample rather than a percentile clip: the
        percentiles are taken later, on the stretched values, and clipping
        twice would compound.
        """
        if NORM_USE_DATA_RANGE:
            lo, hi = self.sampled_cut(provider, band, 0.0, 1.0, extent)
            if lo is None or hi is None or hi <= lo:
                stats = self.sampled_stats(provider, band, extent)
                if stats is not None:
                    lo, hi = stats.minimumValue, stats.maximumValue
            if lo is not None and hi is not None and hi > lo:
                return (float(lo), float(hi))
            print(f"[NORM] band {band}: could not measure a range, "
                  f"falling back to {NORM_MIN}..{NORM_MAX}")
        return (float(NORM_MIN), float(NORM_MAX))

    def _gamma_bounds(self, source, band_no, dn_min, dn_max, window, clip_pct):
        """(min, max) for the SAR sqrt-gamma stretch, or (None, None).

        Reads a downsampled tile, applies the power stretch, and inverse-maps
        the output percentiles back to input values, so the result drives a
        plain QgsContrastEnhancement and QGIS does the rest.
        """
        try:
            from osgeo import gdal
            ds = gdal.Open(source, gdal.GA_ReadOnly)
            if not ds:
                return (None, None)
            band = ds.GetRasterBand(band_no)
            if band is None:
                return (None, None)
            x0, y0, w, h = window or (0, 0, band.XSize, band.YSize)
            x0 = max(0, min(int(x0), band.XSize - 1))
            y0 = max(0, min(int(y0), band.YSize - 1))
            w = max(1, min(int(w), band.XSize - x0))
            h = max(1, min(int(h), band.YSize - y0))
            data = band.ReadAsArray(x0, y0, w, h,
                                    min(w, 1000), min(h, 1000)).astype(float)
            nodata = band.GetNoDataValue()
            ds = None

            # Zeros stay in: this measures the PICTURE, and fill that
            # renders black is part of what is in view. The statistics have
            # their own, stricter, view of what counts as a pixel.
            mask = valid_mask(data, nodata, zero_is_nodata=False)
            dn_range = float(dn_max - dn_min)
            if not np.isfinite(dn_range) or dn_range <= 0:
                return (None, None)
            norm = (np.clip(data, dn_min, dn_max) - dn_min) / dn_range
            stretched = np.power(norm, NORM_GAMMA) * 255.0
            valid = stretched[mask]
            if valid.size == 0:
                return (None, None)

            clip = min(max(float(clip_pct), 0.0), 49.0)
            lo_out = float(np.percentile(valid, clip))
            hi_out = float(np.percentile(valid, 100.0 - clip))
            if not (np.isfinite(lo_out) and np.isfinite(hi_out)):
                return (None, None)
            inv = 1.0 / NORM_GAMMA
            min_dn = dn_min + dn_range * ((lo_out / 255.0) ** inv)
            max_dn = dn_min + dn_range * ((hi_out / 255.0) ** inv)
            if not (np.isfinite(min_dn) and np.isfinite(max_dn)) or max_dn <= min_dn:
                return (None, None)
            return (min_dn, max_dn)
        except Exception as e:
            print(f"[NORM] {e}")
            return (None, None)

    def normalize_to_view(self):
        """Stretch the canvas to what is currently in view, and keep it."""
        layer = self.raster_layer
        if layer is None or not layer.isValid():
            QMessageBox.warning(self, "Normalize", "No raster loaded.")
            return
        provider = layer.dataProvider()
        extent = self.view_extent()
        window = self.view_pixel_window(extent)
        clip = self.clip_percent()
        pinned = {}
        for band in sorted(set(self._selected_bands())) or [1]:
            base_lo, base_hi = self.normalize_range(provider, band, extent)
            lo, hi = self._gamma_bounds(layer.source(), band, base_lo, base_hi,
                                        window, clip)
            if lo is None or hi is None or hi <= lo:
                print(f"[NORM] band {band}: view has no usable range, "
                      f"leaving it as it was")
                continue
            pinned[band] = (lo, hi)
            print(f"[NORM] band {band}: view {base_lo:.6g}..{base_hi:.6g} "
                  f"clip {clip:g}% -> stretch {lo:.6g}..{hi:.6g}")
        if not pinned:
            QMessageBox.warning(
                self, "Normalize",
                "Nothing measurable in the current view -- it may be all "
                "nodata. Move to where there is data and press again.")
            return
        self.norm_bounds.update(pinned)
        self.apply_bands()

    # ── BAND PICKER ──────────────────────────────────────────────────────────
    def _band_labels(self, layer):
        """Human labels for a raster's bands, using the names it carries."""
        provider = layer.dataProvider()
        labels = []
        for band in range(1, provider.bandCount() + 1):
            name = ""
            try:
                name = (layer.bandName(band) or "").strip()
            except Exception:
                pass
            if not name or re.fullmatch(r"Band\s*0*\d+", name):
                name = f"Band {band}"
            elif not name.lower().startswith("band"):
                name = f"{band}: {name}"
            labels.append(name)
        return labels

    def clear_stretch_cache(self, source=None):
        if source is None:
            self._stretch_cache = {}
        else:
            for key in [k for k in self._stretch_cache if k[0] == source]:
                del self._stretch_cache[key]

    def populate_band_picker(self, layer):
        """Fill the R/G/B combos and the stats-band list from the raster."""
        if layer is None or not layer.isValid():
            self.overlay.hide()
            self.band_labels, self.band_prefixes = [], []
            self.measure_bands, self.factor_band = [], None
            self.incidence_band = None
            return
        self.clear_stretch_cache(layer.source())
        self.norm_bounds = {}
        labels = self._band_labels(layer)
        self.band_labels = labels
        self.factor_band = (factor_band_index(labels)
                            or self._band_from_metadata(
                                layer, RTC_FACTOR_BAND_KEY, len(labels)))
        self.incidence_band = (incidence_band_index(labels)
                               or self._band_from_metadata(
                                   layer, INCIDENCE_BAND_KEY, len(labels)))
        skip = {self.factor_band, self.incidence_band}
        self.measure_bands = [(index, label)
                              for index, label in enumerate(labels, start=1)
                              if index not in skip]
        self.band_prefixes = [band_prefix(label, index)
                              for index, label in self.measure_bands]
        count = len(labels)
        if count < 1:
            self.overlay.hide()
            return
        try:
            for combo, default in zip(self.band_combos, RGB_DEFAULT_BANDS):
                combo.blockSignals(True)
                combo.clear()
                for label in labels:
                    combo.addItem(label)
                combo.setCurrentIndex(min(default - 1, count - 1))
        finally:
            for combo in self.band_combos:
                combo.blockSignals(False)
        for combo in self.band_combos:
            combo.setVisible(count >= 2)
        try:
            self.stats_band_combo.blockSignals(True)
            self.stats_band_combo.clear()
            for _, label in self.measure_bands:
                self.stats_band_combo.addItem(label)
            self.stats_band_combo.setCurrentIndex(0)
        finally:
            self.stats_band_combo.blockSignals(False)
        # sigma0 is only offerable when the conversion is in the raster.
        try:
            self.backscatter_combo.setEnabled(self.factor_band is not None)
            if self.factor_band is None:
                self.backscatter_combo.setCurrentIndex(0)
                self.backscatter_combo.setToolTip(
                    "sigma0 needs this product's rtcGammaToSigmaFactor, and\n"
                    "this raster does not carry it -- so every figure here is\n"
                    "gamma0, as stored.\n\n"
                    "A GCOV loaded from its '.h5' carries it. A GeoTIFF does\n"
                    "when DPQED_gcov2tif.py wrote it; DPQED_h52tif.py does not\n"
                    "write it, and names no bands either.")
        except Exception:
            pass
        self.overlay.show()
        self.overlay.raise_()
        print(f"[BANDS] {count}: {', '.join(labels)} "
              f"-> {', '.join(self.band_prefixes)}")
        if self.factor_band is not None:
            print(f"[BANDS] band {self.factor_band} is the RTC "
                  f"gamma-to-sigma factor: not measured, sigma0 available")
        if self.incidence_band is not None:
            print(f"[BANDS] band {self.incidence_band} is the incidence "
                  f"angle: not measured, reported per ROI")
        elif self.incidence_cube is None:
            print("[BANDS] no incidence angle in this raster: the inc_deg "
                  "column will be empty")
        self.apply_bands()

    def _band_from_metadata(self, layer, key, band_count):
        """A band number the raster's header declares, or None.

        A GeoTIFF band carries a description, but a writer has to go out of its
        way to set one and most do not -- so a TIF can hold the RTC factor, or
        the incidence angle, in a band nothing names. These headers say which
        band that is. They are only consulted when no band name matches, so a
        file that says both and disagrees with itself is read the way a person
        would read it.
        """
        try:
            from osgeo import gdal
            ds = gdal.Open(layer.source(), gdal.GA_ReadOnly)
            if ds is None:
                return None
            raw = ds.GetMetadataItem(key)
            ds = None
            if raw is None:
                return None
            number = int(str(raw).strip())
        except Exception:
            return None
        if not 1 <= number <= band_count:
            print(f"[BANDS] {key}={raw} is not one of this raster's "
                  f"{band_count} band(s); ignored")
            return None
        print(f"[BANDS] band {number} declared by the raster's {key} header")
        return number

    def _selected_bands(self):
        return [max(combo.currentIndex(), 0) + 1 for combo in self.band_combos]

    def stats_band(self):
        """The band the table and the summary report."""
        try:
            index = self.stats_band_combo.currentIndex()
            if 0 <= index < len(self.measure_bands):
                return self.measure_bands[index][1]
        except Exception:
            pass
        return self.measure_bands[0][1] if self.measure_bands else ""

    @staticmethod
    def sampled_cut(provider, band, low, high, extent=None):
        """cumulativeCut over a bounded sample, or (None, None).

        The sampled overload is tried first: unsampled means a full-resolution
        pass over the whole raster, per band, on the GUI thread.
        """
        try:
            return provider.cumulativeCut(
                band, low, high,
                extent if extent is not None else QgsRectangle(),
                RASTER_SAMPLE_SIZE)
        except TypeError:
            pass
        except Exception:
            return (None, None)
        try:
            return provider.cumulativeCut(band, low, high)
        except Exception:
            return (None, None)

    @staticmethod
    def sampled_stats(provider, band, extent=None):
        """bandStatistics over a bounded sample, or None. Same reasoning."""
        try:
            return provider.bandStatistics(
                band, QgsRasterBandStats.Min | QgsRasterBandStats.Max,
                extent if extent is not None else QgsRectangle(),
                RASTER_SAMPLE_SIZE)
        except TypeError:
            pass
        except Exception:
            return None
        try:
            return provider.bandStatistics(band)
        except Exception:
            return None

    def _stretch_for(self, layer, band):
        """Contrast enhancement for one band, pinned bounds winning outright."""
        provider = layer.dataProvider()
        pinned = self.norm_bounds.get(band)
        if pinned is None:
            key = (layer.source(), band)
            cached = self._stretch_cache.get(key)
            if cached is not None:
                pinned = cached
            else:
                lo, hi = self.sampled_cut(provider, band, RGB_CLIP_LOW,
                                          RGB_CLIP_HIGH)
                if lo is None or hi is None or hi <= lo:
                    stats = self.sampled_stats(provider, band)
                    if stats is not None:
                        lo, hi = stats.minimumValue, stats.maximumValue
                if lo is None or hi is None or hi <= lo:
                    lo, hi = 0.0, 1.0
                pinned = (lo, hi)
                self._stretch_cache[key] = pinned
        ce = QgsContrastEnhancement(provider.dataType(band))
        ce.setContrastEnhancementAlgorithm(
            QgsContrastEnhancement.StretchToMinimumMaximum)
        ce.setMinimumValue(pinned[0])
        ce.setMaximumValue(pinned[1])
        return ce

    def apply_bands(self):
        """Render the canvas from the picked bands. Rendering only."""
        layer = self.raster_layer
        if layer is None or not layer.isValid():
            return
        provider = layer.dataProvider()
        try:
            count = max(provider.bandCount(), 1)
            red, green, blue = [min(b, count) for b in self._selected_bands()]
            if count < 2:
                renderer = QgsSingleBandGrayRenderer(provider, 1)
                renderer.setContrastEnhancement(self._stretch_for(layer, 1))
                shown = "grey band 1"
            else:
                renderer = QgsMultiBandColorRenderer(provider, red, green, blue)
                renderer.setRedContrastEnhancement(self._stretch_for(layer, red))
                renderer.setGreenContrastEnhancement(self._stretch_for(layer, green))
                renderer.setBlueContrastEnhancement(self._stretch_for(layer, blue))
                shown = f"R={red} G={green} B={blue}"
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            self.canvas.refresh()
            stretch = ("pinned" if self.norm_bounds
                       else f"{RGB_CLIP_LOW:.0%}-{RGB_CLIP_HIGH:.0%}")
            print(f"[BANDS] rendered as {shown} ({stretch} stretch)")
        except Exception as e:
            print(f"[BANDS] {e}")

    # ── ROIs ─────────────────────────────────────────────────────────────────
    def add_roi(self, ring, kind, name=None, select=True):
        """Record a drawn ROI, measure it, and put it in the table.

        An ROI is plain data -- ring, name, and one statistics dict per band --
        so everything that reads one (the table, the detail panel, both
        exports) reads the same thing, and none of them needs a canvas.
        """
        ring = [(float(x), float(y)) for x, y in ring]
        if len(ring) < 3 or ring_area(ring) <= 0.0:
            print("[ROI] no area; discarded")
            return None
        roi_id = self._next_roi_id
        roi = {
            "roi": roi_id,
            "name": name or f"ROI {roi_id}",
            "kind": kind,
            "class": self.roi_class(),
            "ring": ring,
            "npix": 0,
            "area_m2": 0.0,
            "cx": None, "cy": None, "lon": None, "lat": None,
            "domain": self.domain(),
            "src": os.path.basename(self.raster_path),
            "stats": {},
        }
        self._geometry_fields(roi)
        self.measure_roi(roi)
        if self.raster_layer is not None and roi["npix"] < MIN_ROI_PIXELS:
            # A stray click during a drag, or an ROI drawn off the edge of the
            # scene. Either way there is nothing in it to measure.
            print(f"[ROI] {roi['npix']} pixel(s) inside -- too small to "
                  f"measure, discarded")
            return None
        self._next_roi_id += 1
        self.rois.append(roi)
        self.draw_roi(roi)
        self.refresh_table()
        if select:
            self.select_roi(roi_id)
        print(f"[ROI] {roi['name']}: {kind}, {roi['npix']} px, "
              f"{roi['area_m2']:.0f} m²")
        return roi

    def _geometry_fields(self, roi):
        """Area, centroid and lon/lat, all from the ring in the working CRS."""
        roi["area_m2"] = ring_area(roi["ring"])
        centroid = ring_centroid(roi["ring"])
        if centroid is not None:
            roi["cx"], roi["cy"] = centroid
            roi["lon"], roi["lat"] = self._to_lonlat(*centroid)

    def draw_roi(self, roi, selected=False):
        """Outline one ROI on the canvas, in its selected or ordinary colour."""
        band = self.roi_bands.get(roi["roi"])
        if band is None:
            band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            self.roi_bands[roi["roi"]] = band
        try:
            band.reset(QgsWkbTypes.PolygonGeometry)
            for index, (x, y) in enumerate(roi["ring"]):
                band.addPoint(QgsPointXY(x, y),
                              index == len(roi["ring"]) - 1)
            rgb = ROI_COLOR_SELECTED if selected else ROI_COLOR
            fill = QColor(*rgb)
            fill.setAlpha(ROI_FILL_ALPHA)
            band.setColor(QColor(*rgb))
            band.setFillColor(fill)
            band.setWidth(ROI_WIDTH_SELECTED if selected else ROI_WIDTH)
            band.show()
        except Exception as e:
            print(f"[ROI] draw: {e}")
        self._draw_roi_label(roi, rgb)

    def _draw_roi_label(self, roi, rgb):
        """The ROI's number at its centre, in the outline's colour.

        Guarded rather than assumed: a canvas item is the idiomatic way to put
        text on a QGIS canvas, but if a build will not have one, an ROI with no
        number is a smaller loss than a tool that will not start.
        """
        centre = ring_centroid(roi.get("ring") or [])
        if centre is None:
            return
        try:
            point = QgsPointXY(centre[0], centre[1])
            label = self.roi_labels.get(roi["roi"])
            if label is None:
                self.roi_labels[roi["roi"]] = RoiLabelItem(
                    self.canvas, point, roi["roi"], rgb)
            else:
                label.set_state(point, roi["roi"], rgb)
        except Exception as e:
            print(f"[ROI] label: {e}")
            self.roi_labels.pop(roi["roi"], None)

    def redraw_rois(self):
        """Re-outline the ROIs on show, the selected one in yellow.

        The ones the class filter excludes are hidden rather than dropped:
        their rubber bands and numbers stay built, so switching the filter back
        costs nothing and the ROI is exactly the one that was there before.
        """
        selected = self.selected_roi()
        selected_id = selected["roi"] if selected else None
        shown = {id(roi) for roi in self.shown_rois()}
        for roi in self.rois:
            if id(roi) in shown:
                self.draw_roi(roi, selected=(roi["roi"] == selected_id))
            else:
                self._hide_roi(roi)

    def _hide_roi(self, roi):
        """Take one ROI off the canvas without forgetting it."""
        for store in (self.roi_bands, self.roi_labels):
            item = store.get(roi["roi"])
            if item is None:
                continue
            try:
                item.hide()
            except Exception as e:
                print(f"[ROI] hide: {e}")

    def _drop_roi_band(self, roi_id):
        band = self.roi_bands.pop(roi_id, None)
        label = self.roi_labels.pop(roi_id, None)
        scene = None
        try:
            scene = self.canvas.scene()
        except Exception:
            pass
        if band is not None:
            try:
                band.reset(QgsWkbTypes.PolygonGeometry)
                if scene:
                    scene.removeItem(band)
            except Exception as e:
                print(f"[ROI] remove: {e}")
        if label is not None and scene:
            # A number left behind outlives the ROI it names, and then names
            # whichever ROI is renumbered into its place.
            try:
                scene.removeItem(label)
            except Exception as e:
                print(f"[ROI] remove label: {e}")

    def selected_roi(self):
        try:
            row = self.table.currentRow()
        except Exception:
            return None
        shown = getattr(self, "_table_rois", None) or []
        if row is None or row < 0 or row >= len(shown):
            return None
        return shown[row]

    def select_roi(self, roi_id):
        for row, roi in enumerate(getattr(self, "_table_rois", None) or []):
            if roi["roi"] == roi_id:
                self.table.setCurrentCell(row, 0)
                return

    def clear_selection(self):
        """Leave no ROI selected.

        clearSelection() alone is not enough: it drops the highlight and leaves
        the CURRENT cell where it was, so selected_roi() -- which reads the
        current row -- would still name the ROI that was just deselected, and
        Delete would take it. The current cell has to go too.
        """
        try:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
        except Exception as e:
            print(f"[ROI] deselect: {e}")

    def select_roi_at(self, x, y):
        """Select whatever ROI is under a map point, or clear the selection.

        Selecting the table row rather than tracking a separate canvas
        selection: there is one selected ROI, the table owns which, and Delete
        and the editable cells already work off it.
        """
        roi = roi_at(self.rois, x, y)
        if roi is None:
            self.clear_selection()
            self.redraw_rois()
            self.canvas.refresh()
            return None
        self.select_roi(roi["roi"])
        print(f"[ROI] selected {roi['roi']}: {roi['name']}")
        return roi

    def delete_roi(self):
        roi = self.selected_roi()
        if roi is None:
            return
        self._drop_roi_band(roi["roi"])
        self.rois.remove(roi)
        self.refresh_table()
        self.canvas.refresh()
        print(f"[ROI] deleted {roi['name']}")

    def clear_rois(self):
        if not self.rois:
            return
        answer = QMessageBox.question(
            self, "Clear ROIs",
            f"Delete all {len(self.rois)} ROI(s)?\n"
            "Export them first if they are not saved.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for roi in list(self.rois):
            self._drop_roi_band(roi["roi"])
        self.rois = []
        self._next_roi_id = 1
        self.refresh_table()
        self.canvas.refresh()
        print("[ROI] all cleared")

    def cancel_drawing(self):
        """Escape: abandon the polygon in progress, or drop the selection."""
        tool = getattr(self, "map_tools", {}).get(self.current_map_tool())
        if isinstance(tool, RoiMapTool):
            tool.cancel()
        self.clear_selection()
        self.redraw_rois()

    def zoom_to_roi(self):
        """Bring the selected ROI up, with room around it to see its setting."""
        roi = self.selected_roi()
        if roi is None:
            return
        bounds = ring_bounds(roi["ring"])
        if bounds is None:
            return
        xmin, ymin, xmax, ymax = bounds
        pad = max(xmax - xmin, ymax - ymin, 1.0) * 0.5
        try:
            self.canvas.setExtent(QgsRectangle(xmin - pad, ymin - pad,
                                               xmax + pad, ymax + pad))
            self.canvas.refresh()
        except Exception as e:
            print(f"[ROI] zoom: {e}")

    # ── MEASUREMENT ──────────────────────────────────────────────────────────
    def measure_roi(self, roi):
        """Read the ROI's pixels from the source raster and compute its stats.

        Read from the source at full resolution through GDAL, not from what the
        canvas is showing: the renderer's business is a stretch over a
        downsampled overview, and a measurement taken off that would be a
        measurement of the picture rather than of the product.
        """
        roi["stats"] = {}
        roi["npix"] = 0
        roi["domain"] = self.domain()
        roi["backscat"] = BACKSCATTER_GAMMA0
        roi["inc_deg"] = self._incidence_from_cube(roi)
        roi["src"] = os.path.basename(self.raster_path)
        layer = self.raster_layer
        if layer is None or not layer.isValid() or not self.measure_bands:
            return False
        try:
            from osgeo import gdal
            ds = gdal.Open(layer.source(), gdal.GA_ReadOnly)
        except Exception as e:
            print(f"[STATS] GDAL could not open the raster: {e}")
            return False
        if ds is None:
            print(f"[STATS] GDAL could not open {layer.source()}")
            return False
        try:
            gt = ds.GetGeoTransform()
            if gt is None or gt[2] or gt[4]:
                print("[STATS] the raster's geotransform is rotated; "
                      "ROI statistics need a north-up grid")
                return False
            if gt[5] > 0:
                print("[STATS] the raster runs south-up, which this does not "
                      "read")
                return False
            px, py = abs(gt[1]), abs(gt[5])
            ring = self._ring_in_layer_crs(roi["ring"])
            window = pixel_window(ring_bounds(ring), gt[0], gt[3], px, py,
                                  ds.RasterXSize, ds.RasterYSize)
            if window is None:
                print(f"[STATS] {roi['name']} does not overlap the raster")
                return False
            col0, row0, ncol, nrow = window
            if ncol * nrow > ROI_MAX_PIXELS:
                QMessageBox.warning(
                    self, "ROI too large",
                    f"That ROI spans {ncol * nrow:,} pixels of the raster. "
                    f"RADIAL reads every\none of them, for every band, at "
                    f"full resolution -- draw a smaller\none over the target "
                    f"you mean to measure.\n\nA long diagonal polygon spans "
                    f"far more than it encloses; a\nrectangle over the same "
                    f"target may fit where it does not.")
                return False
            mask = polygon_mask(
                ring_to_pixels(ring, gt[0], gt[3], px, py, col0, row0),
                ncol, nrow)
            roi["npix"] = int(mask.sum())
            if roi["npix"] == 0:
                return False
            domain = self.domain()
            zero_nodata = self.zero_is_nodata()
            # Read once, measure twice. The factor costs one window read, and
            # with it in hand gamma0 and sigma0 are two cheap passes over
            # pixels already in memory -- so both are stored, and switching
            # between them later reads nothing from disk.
            factor, factor_ok = self._rtc_factor(ds, window)
            roi["stats"] = {BACKSCATTER_GAMMA0: {}}
            if factor is not None:
                roi["stats"][BACKSCATTER_SIGMA0] = {}
            wanted = self.backscatter()
            roi["backscat"] = (wanted if wanted in roi["stats"]
                               else BACKSCATTER_GAMMA0)
            angle = self._incidence_from_band(ds, window, mask, ring, gt, px, py)
            if angle is not None:
                roi["inc_deg"] = angle
            for index, label in self.measure_bands:
                band = ds.GetRasterBand(index)
                if band is None:
                    continue
                data = band.ReadAsArray(col0, row0, ncol, nrow)
                if data is None:
                    continue
                if np.iscomplexobj(data):
                    # A complex band is an off-diagonal covariance term or a
                    # GSLC channel. Its magnitude is a correlation or an
                    # amplitude, not a backscatter, and quietly taking one
                    # would put a number in a gamma0 column that means
                    # something else.
                    print(f"[STATS] band {index} ({label}) is complex; "
                          f"not a backscatter, so it is left unmeasured")
                    continue
                data = data.astype(float)
                good = mask & valid_mask(data, band.GetNoDataValue(),
                                         zero_nodata, domain)
                # gamma0 keeps every valid pixel: a factor missing somewhere
                # is a gap in the conversion, not in what the product
                # recorded, and dropping those pixels here would make the two
                # conventions disagree about which ROI they describe.
                roi["stats"][BACKSCATTER_GAMMA0][label] = roi_statistics(
                    data[good], domain)
                if factor is not None:
                    converted = good & factor_ok
                    roi["stats"][BACKSCATTER_SIGMA0][label] = roi_statistics(
                        data[converted], domain, factor[converted])
            return True
        except Exception as e:
            print(f"[STATS] {roi['name']}: {e}")
            return False
        finally:
            ds = None

    def _rtc_factor(self, ds, window):
        """The gamma-to-sigma factor over a window, and where it is usable.

        Read whenever the raster carries it, not only when sigma0 is the
        selection: both conventions are measured from one read, so the
        selection decides what is DISPLAYED and never what was measured. A
        raster without the factor returns (None, None) and the ROI has the
        gamma0 it holds -- never relabelled on a conversion that never
        happened.
        """
        if self.factor_band is None:
            return (None, None)
        try:
            band = ds.GetRasterBand(self.factor_band)
            if band is None:
                return (None, None)
            col0, row0, ncol, nrow = window
            data = band.ReadAsArray(col0, row0, ncol, nrow)
            if data is None:
                return (None, None)
            data = data.astype(float)
            # A factor at or below zero is not a ratio of areas. Those pixels
            # have no sigma0, so they leave the measurement rather than
            # becoming one.
            return (data, valid_mask(data, band.GetNoDataValue(), False)
                    & (data > 0))
        except Exception as e:
            print(f"[STATS] RTC factor: {e}")
            return (None, None)

    def _incidence_from_band(self, ds, window, mask, ring, gt, px, py):
        """The ROI's incidence angle, in degrees, from the raster's own band.

        At the ROI's centre, which is the figure an incidence angle is quoted
        as -- and where that pixel has none, the mean over the ROI, so a centre
        landing on a gap does not lose the column. Incidence varies by well
        under a degree across an ROI of the size this tool is for, so the two
        agree to the second decimal; the fallback is about robustness, not
        about a different measurement.
        """
        if self.incidence_band is None:
            return None
        try:
            band = ds.GetRasterBand(self.incidence_band)
            if band is None:
                return None
            col0, row0, ncol, nrow = window
            data = band.ReadAsArray(col0, row0, ncol, nrow)
            if data is None:
                return None
            data = data.astype(float)
            good = mask & valid_mask(data, band.GetNoDataValue(), False)
            if not good.any():
                return None
            centre = ring_centroid(ring)
            if centre is not None:
                col = int((centre[0] - gt[0]) / px) - col0
                row = int((gt[3] - centre[1]) / py) - row0
                if 0 <= row < nrow and 0 <= col < ncol and good[row, col]:
                    return float(data[row, col])
            return float(data[good].mean())
        except Exception as e:
            print(f"[STATS] incidence: {e}")
            return None

    def _incidence_from_cube(self, roi):
        """The ROI centre's incidence angle from a '.h5' product's own cube.

        Used when the raster carries no incidence band -- which is every VRT
        built from an HDF5, since a VRT stacks datasets on one grid and the
        geometry cubes are on another, coarser one at several heights. The cube
        is read once when the product loads and sampled per ROI here, which is
        cheaper than resampling it onto the image grid to read one value off it.
        """
        cube = self.incidence_cube
        if not cube:
            return None
        centre = ring_centroid(roi.get("ring") or [])
        if centre is None:
            return None
        value = bilinear_at(cube[0], cube[1], cube[2], centre[0], centre[1])
        return None if not np.isfinite(value) else float(value)

    def apply_backscatter(self):
        """Switch which convention the window and the shapefile report.

        Nothing is re-read: both were measured when the ROI was, so this picks
        which of two stored answers is being looked at. An ROI whose raster
        had no factor stays on the gamma0 it holds rather than following the
        selection into a convention it was never measured in.
        """
        wanted = self.backscatter()
        for roi in self.rois:
            available = roi_conventions(roi)
            roi["backscat"] = (wanted if wanted in available
                               else (available[0] if available
                                     else BACKSCATTER_GAMMA0))
        self.refresh_table()

    def recompute_all(self, reason=""):
        """Re-measure every ROI, after a change to what is being measured."""
        if not self.rois:
            self.refresh_table()
            return
        for roi in self.rois:
            self.measure_roi(roi)
        note = f" ({reason})" if reason else ""
        print(f"[STATS] {len(self.rois)} ROI(s) re-measured{note}")
        self.refresh_table()

    # ── TABLE, SUMMARY, DETAIL ───────────────────────────────────────────────
    def refresh_table(self):
        """Rebuild the table for the selected band, keeping the selection."""
        selected = self.selected_roi()
        selected_id = selected["roi"] if selected else None
        band = self.stats_band()
        formats = {key: fmt for key, _, _, fmt in STAT_FIELDS}
        editable = {ROI_TABLE_COLUMNS.index("name"),
                    ROI_TABLE_COLUMNS.index("class")}
        # The table's rows are the filtered ROIs, and every lookup from a row
        # goes through this list rather than indexing self.rois: with a filter
        # on, row 0 is not ROI 0, and an index that assumed it would rename and
        # delete the wrong ROI without a word.
        self._table_rois = self.shown_rois()
        self._filling_table = True
        try:
            self.table.setRowCount(0)
            self.table.setRowCount(len(self._table_rois))
            for row, roi in enumerate(self._table_rois):
                stats = roi_stats(roi, band) or {}
                values = [str(roi["roi"]), roi["name"],
                          roi.get("class") or ROI_CLASS_UNSET, roi["kind"],
                          f"{roi['npix']}", f"{roi['area_m2']:.1f}",
                          format_stat(roi.get("inc_deg"), "{:.2f}")]
                values += [format_stat(stats.get(key), formats[key])
                           for key in TABLE_STATS]
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    if column not in editable:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, column, item)
        finally:
            self._filling_table = False
        if selected_id is not None:
            self.select_roi(selected_id)
        self.update_summary()
        self.update_detail()
        self.redraw_rois()

    def on_selection_changed(self):
        if self._filling_table:
            return
        self.update_detail()
        self.redraw_rois()

    def on_item_changed(self, item):
        """A typed label belongs to the ROI, and goes out with the export."""
        if self._filling_table or item is None:
            return
        try:
            row, column = item.row(), item.column()
        except Exception:
            return
        shown = getattr(self, "_table_rois", None) or []
        if not 0 <= row < len(shown):
            return
        roi = shown[row]
        if column == ROI_TABLE_COLUMNS.index("name"):
            roi["name"] = item.text().strip() or roi["name"]
            self.update_detail()
        elif column == ROI_TABLE_COLUMNS.index("class"):
            # Re-classing an ROI moves it between groups, so the summary has
            # to be redrawn -- the figures it was in are no longer its. With a
            # filter on it can also move the ROI out of view, which is why the
            # whole table is rebuilt rather than the one row repainted.
            roi["class"] = item.text().strip() or ROI_CLASS_UNSET
            self.refresh_table()

    def update_summary(self):
        """One row per class: how bright, how alike, how many looks.

        Per class rather than over everything drawn, because the spread and
        the median ENL only mean anything within one land cover -- ten
        vegetation ROIs and eight water ones share a scene, not a population.
        """
        band = self.stats_band()
        rows = class_summary(self.rois, band)
        try:
            self.class_table.setRowCount(0)
            self.class_table.setRowCount(len(rows))
            for row, (label, summary) in enumerate(rows):
                values = [label, str(summary["count"]),
                          format_stat(summary["mean_db"], "{:.2f}"),
                          format_stat(summary["spread_db"], "{:.2f}"),
                          format_stat(summary["enl"], "{:.2f}")]
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(row, column, item)
        except Exception as e:
            print(f"[SUMMARY] {e}")
        estimate = nesz_estimate(self.rois, band)
        self.lbl_context.setText(
            f"{band or '(no band)'}   \u00b7   {self.backscatter()}   \u00b7   "
            f"{len(self.rois)} ROI in {len(group_by_class(self.rois))} class(es)"
            + (f"   \u00b7   noise floor \u2264 "
               f"{format_stat(estimate['nesz_db'], '{:.2f}')} dB "
               f"({estimate['rois']} {NESZ_CLASS})"
               if estimate["rois"] else ""))

    def update_detail(self):
        """The selected ROI, every band, in a block that pastes into a report."""
        roi = self.selected_roi()
        if roi is None:
            self.detail.setPlainText("")
            return
        lon = format_stat(roi.get("lon"), "{:.6f}")
        lat = format_stat(roi.get("lat"), "{:.6f}")
        lines = [
            f"ROI {roi['roi']}  {roi['name']}  "
            f"[{roi.get('class') or ROI_CLASS_UNSET}, {roi['kind']}]",
            f"  {roi['npix']} px   {roi['area_m2']:.0f} m²   "
            f"lon/lat {lon}, {lat}",
            f"  {roi.get('backscat')}"
            + (", both conventions in the CSV"
               if len(roi_conventions(roi)) > 1 else "")
            + f", read as {roi.get('domain')}, "
            + f"from {roi.get('src') or '(no raster)'}",
            f"  incidence {format_stat(roi.get('inc_deg'), '{:.2f}')} deg",
            "",
            f"  {'band':<10}{'n':>8}{'mean dB':>10}{'std dB':>9}{'cv':>8}"
            f"{'ENL':>8}{'p5 dB':>9}{'p95 dB':>9}{'nonpos':>8}",
        ]
        for _, label in self.measure_bands:
            stats = roi_stats(roi, label) or {}
            lines.append(
                f"  {label[:10]:<10}"
                f"{format_stat(stats.get('n'), '{:.0f}'):>8}"
                f"{format_stat(stats.get('mean_db'), '{:.2f}'):>10}"
                f"{format_stat(stats.get('sdev_db'), '{:.2f}'):>9}"
                f"{format_stat(stats.get('cv'), '{:.3f}'):>8}"
                f"{format_stat(stats.get('enl'), '{:.2f}'):>8}"
                f"{format_stat(stats.get('p5_db'), '{:.2f}'):>9}"
                f"{format_stat(stats.get('p95_db'), '{:.2f}'):>9}"
                f"{format_stat(stats.get('nonpos'), '{:.0f}'):>8}")
        lines += ["", "  linear power (mean, std):"]
        for _, label in self.measure_bands:
            stats = roi_stats(roi, label) or {}
            lines.append(f"  {label[:10]:<10}"
                         f"{format_stat(stats.get('mean'), '{:.6g}'):>14}"
                         f"{format_stat(stats.get('std'), '{:.6g}'):>14}")

        # How far this ROI stands above the floor the water ROIs show. A few dB
        # and its backscatter is mostly noise, whatever its mean says -- which
        # is not visible in any other number on the panel.
        margins = []
        for _, label in self.measure_bands:
            estimate = nesz_estimate(self.rois, label)
            if not estimate["rois"]:
                continue
            margin = nesz_margin_db(roi, label, estimate["nesz_db"])
            margins.append(
                f"  {label[:10]:<10}"
                f"{format_stat(estimate['nesz_db'], '{:.2f}'):>14}"
                f"{format_stat(margin, '{:.2f}'):>14}")
        if margins:
            lines += ["", f"  noise floor and margin, sigma0 dB "
                          f"(bound from {NESZ_CLASS} ROIs):",
                      f"  {'band':<10}{'<= NESZ':>14}{'margin':>14}"] + margins
        self.detail.setPlainText("\n".join(lines))

    # ── EXPORT ───────────────────────────────────────────────────────────────
    def export_fields(self):
        """(QgsFields, [(field name, roi key, band label, stat key)]).

        Built from the raster actually loaded, so a two-polarization product
        exports two sets of columns and a quad-pol one exports four, rather
        than every product exporting a fixed grid mostly full of nulls.
        """
        fields = QgsFields()
        types = {"int": QVariant.Int, "double": QVariant.Double,
                 "string": QVariant.String}
        plan = []

        def add(name, kind):
            """Append one field, and refuse to carry on if it did not take.

            QgsFields.append() returns False on a name it will not accept --
            a duplicate, or one the driver rejects -- and the caller is free
            to ignore it. Ignoring it is how a shapefile ends up one column
            short of its attribute lists, which then land in the wrong
            columns or are dropped whole with nothing reported.
            """
            field = (QgsField(name, types[kind], "", 64)
                     if kind == "string" else QgsField(name, types[kind]))
            if fields.append(field) is False:
                raise RuntimeError(f"the writer would not take a field "
                                   f"named {name!r}")

        for name, kind in ROI_FIELDS:
            add(name, kind)
            plan.append((name, name, None, None))
        names = dbf_field_names(self.band_prefixes, STAT_KEYS,
                                reserved=[name for name, _ in ROI_FIELDS])
        kinds = {key: kind for key, _, kind, _ in STAT_FIELDS}
        for prefix, (_, label) in zip(self.band_prefixes, self.measure_bands):
            for key in STAT_KEYS:
                name = names[(prefix, key)]
                add(name, kinds[key])
                plan.append((name, None, label, key))
        return fields, plan

    def export_shapefile(self):
        """Write the ROIs as polygons carrying every statistic.

        Polygons, not the points RIVAL exports: the measurement IS the area,
        and a point in the middle of it would throw away the only record of
        which pixels produced the numbers. The same file reloads through 'Load
        ROIs', so an ROI set can be drawn once and run over every product.

        Every ROI goes out, whatever the class filter is showing: the filter is
        a way of reading the table, not a decision about what was measured.
        """
        if not self.rois:
            QMessageBox.warning(self, "Export SHP", "No ROIs to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI shapefile", "", "Shapefile (*.shp)")
        if not path:
            return
        if not path.lower().endswith(".shp"):
            path += ".shp"

        fields, plan = self.export_fields()
        try:
            count, how = self._write_shapefile(path, fields, plan)
        except Exception as e:
            QMessageBox.critical(self, "Export SHP", f"Could not write:\n{e}")
            return
        if not count:
            QMessageBox.critical(
                self, "Export SHP",
                f"Nothing was written to\n{path}\n\n"
                "The ROIs carry no usable rings.")
            return

        # DBF caps a field name at 10 characters, so the same numbers go out
        # again under names that say what they are -- and beside them the
        # by-class summary, which is the thing a report quotes and which no
        # per-ROI table states outright.
        stem = os.path.splitext(path)[0]
        csv_path = stem + "_stats.csv"
        class_path = stem + "_by_class.csv"
        nesz_path = stem + "_nesz.csv"
        bands = [label for _, label in self.measure_bands]
        written = self._write_csv(csv_path)
        written_classes = self._write_rows(
            class_path, class_summary_rows(self.rois, bands))
        written_nesz = self._write_rows(nesz_path,
                                        nesz_rows(self.rois, bands))
        per_class = self._export_by_class(stem, fields, plan)
        crs = self.proj_crs.authid() or self.proj_crs.description()
        print(f"[EXPORT] {count} ROI(s) -> {path} [{crs}] via {how}")
        QMessageBox.information(
            self, "Export SHP",
            f"{count} ROI(s) written to\n{path}\n\n"
            f"CRS: {crs}\n"
            f"Bands: {', '.join(l for _, l in self.measure_bands) or '(none)'}\n"
            f"As: {self.backscatter()}\n"
            f"Writer: {how}\n"
            + (f"\nFull-length statistics beside it:\n{csv_path}"
               if written else "")
            + (f"\nBy class:\n{class_path}" if written_classes else "")
            + (f"\nNoise floor:\n{nesz_path}" if written_nesz else "")
            + (("\n\nPer class, beside them:\n"
                + "\n".join(f"  {label}: {os.path.basename(stem)}_{slug}"
                             f".shp ({n} ROI)"
                             for label, slug, n in per_class))
               if per_class else ""))

    def _export_by_class(self, stem, fields, plan):
        """A shapefile and a statistics CSV per class, beside the whole set.

        The whole-set files stay exactly as they were -- this adds to them
        rather than replacing them, because the two are read by different
        people: one report covers the product, and one analyst covers water.

        Written whatever the class filter is showing, and written for every
        class rather than the selected one, for the same reason the whole-set
        export ignores the filter: what was measured is not a function of what
        happens to be on screen.

        A single-class set gets none of these. The whole-set file already is
        that class, and a second copy of it under a longer name is not a
        by-class export -- it is a duplicate that someone eventually diffs.
        """
        exports = class_exports(self.rois)
        if len(exports) < 2:
            return []
        written = []
        for label, slug, members in exports:
            shp_path = f"{stem}_{slug}.shp"
            try:
                count, _ = self._write_shapefile(shp_path, fields, plan,
                                                 members)
            except Exception as e:
                print(f"[EXPORT] {label}: {e}")
                continue
            if not count:
                print(f"[EXPORT] {label}: nothing written to {shp_path}")
                continue
            # No per-class summary CSV: the whole-set '_by_class.csv' already
            # holds this class's row, and a one-row file beside it is another
            # thing to keep in step for no new information.
            self._write_csv(f"{stem}_{slug}_stats.csv", members)
            written.append((label, slug, count))
        return written

    def _write_shapefile(self, path, fields, plan, rois=None):
        """Write the ROIs, by whichever route actually produces features.

        QGIS's writer is tried first, because it is the one that knows the
        project's transform context. It is not trusted, though: a
        QgsVectorFileWriter that cannot take a field or a feature says so in a
        return value nobody is obliged to read, and the result is a .shp with
        no rows and no error at all -- exactly the blank file this replaced. So
        the features are counted, and if the count is zero the same records go
        out again through OGR directly, which fails loudly or not at all.
        """
        if rois is None:
            rois = self.rois
        errors = []
        for name, write in (("QGIS", self._write_shapefile_qgis),
                            ("OGR", self._write_shapefile_ogr)):
            try:
                count = write(path, fields, plan, rois)
            except Exception as e:                        # try the next route
                errors.append(f"{name}: {e}")
                continue
            if count:
                return count, name
            errors.append(f"{name}: wrote no features")
        print("[EXPORT] no writer produced features -- " + "; ".join(errors))
        return 0, "; ".join(errors)

    def _write_shapefile_qgis(self, path, fields, plan, rois):
        """Write through QgsVectorFileWriter, checking everything it returns."""
        writer = self._make_writer(path, fields)
        if writer is None:
            raise RuntimeError("no usable QgsVectorFileWriter")
        error = getattr(writer, "hasError", lambda: 0)()
        if error:
            message = getattr(writer, "errorMessage", lambda: "")()
            del writer
            raise RuntimeError(f"{message or error}")
        count = 0
        try:
            for roi in rois:
                ring = [QgsPointXY(x, y) for x, y in roi.get("ring") or ()]
                if len(ring) < 3:
                    continue
                ring.append(ring[0])         # a shapefile ring is closed
                feature = QgsFeature(fields)
                feature.setGeometry(QgsGeometry.fromPolygonXY([ring]))
                feature.setAttributes(self.feature_attributes(roi, plan))
                if writer.addFeature(feature) is False:
                    message = getattr(writer, "errorMessage", lambda: "")()
                    raise RuntimeError(
                        f"ROI {roi.get('roi')} refused: {message or 'no reason given'}")
                count += 1
        finally:
            del writer      # flushes and closes the .shp/.dbf/.shx/.prj
        return count

    def _write_shapefile_ogr(self, path, fields, plan, rois):
        """Write the same records through OGR, with no Qt layer in between.

        The fallback exists because the QGIS writer's failures are silent and
        this one's are not: every call here returns a code that is checked, so
        a file that comes out of this function either holds the features or
        never existed.
        """
        from osgeo import ogr, osr

        records = shapefile_records(rois, plan)
        if not records:
            return 0
        driver = ogr.GetDriverByName("ESRI Shapefile")
        if driver is None:
            raise RuntimeError("GDAL has no ESRI Shapefile driver")
        if os.path.exists(path):
            driver.DeleteDataSource(path)
        source = driver.CreateDataSource(path)
        if source is None:
            raise RuntimeError(f"could not create {path}")
        reference = osr.SpatialReference()
        wkt = self.proj_crs.toWkt() if self.proj_crs is not None else ""
        if wkt:
            reference.ImportFromWkt(wkt)
        else:
            reference = None
        layer = source.CreateLayer(os.path.splitext(os.path.basename(path))[0],
                                   reference, ogr.wkbPolygon)
        if layer is None:
            raise RuntimeError("could not create the shapefile layer")
        ogr_types = {"int": ogr.OFTInteger, "double": ogr.OFTReal,
                     "string": ogr.OFTString}
        for name, kind in shapefile_schema(plan):
            definition = ogr.FieldDefn(name, ogr_types[kind])
            if kind == "string":
                definition.SetWidth(64)
            if layer.CreateField(definition) != 0:
                raise RuntimeError(f"field {name} refused")
        count = 0
        for wkt_geometry, attributes in records:
            feature = ogr.Feature(layer.GetLayerDefn())
            geometry = ogr.CreateGeometryFromWkt(wkt_geometry)
            if geometry is None:
                raise RuntimeError(f"unreadable ring: {wkt_geometry[:60]}")
            feature.SetGeometry(geometry)
            for index, value in enumerate(attributes):
                if value is None:
                    feature.SetFieldNull(index)
                else:
                    feature.SetField(index, value)
            if layer.CreateFeature(feature) != 0:
                raise RuntimeError("OGR refused a feature")
            count += 1
        layer = None
        source = None       # closes the file set
        return count

    def feature_attributes(self, roi, plan):
        """One ROI's attributes, in the order `plan` declared the fields.

        The order is the plan's, not a second traversal that happens to match:
        a shapefile's attributes are positional, so a field list and an
        attribute list built separately drift into each other's columns and the
        writer reports nothing wrong.
        """
        return roi_attributes(roi, plan)

    def _make_writer(self, path, fields):
        """QgsVectorFileWriter across the versions that changed its API."""
        try:
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "ESRI Shapefile"
            options.fileEncoding = "UTF-8"
        except Exception:
            options = None
        if options is not None:
            for factory in ("create", "createWriter"):
                make = getattr(QgsVectorFileWriter, factory, None)
                if make is None:
                    continue
                try:
                    return make(path, fields, QgsWkbTypes.Polygon, self.proj_crs,
                                QgsProject.instance().transformContext(),
                                options)
                except Exception:
                    continue
        return QgsVectorFileWriter(path, "UTF-8", fields,
                                   QgsWkbTypes.Polygon, self.proj_crs, "ESRI Shapefile")

    def export_csv(self):
        if not self.rois:
            QMessageBox.warning(self, "Export CSV", "No ROIs to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI statistics", "", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        if self._write_csv(path):
            QMessageBox.information(
                self, "Export CSV",
                f"{len(self.rois)} ROI(s) x {len(self.measure_bands)} band(s) "
                f"written to\n{path}")

    def _write_csv(self, path, rois=None):
        """The statistics in long form: one row per ROI and band."""
        return self._write_rows(
            path, stat_rows(self.rois if rois is None else rois,
                            [label for _, label in self.measure_bands]))

    def _write_rows(self, path, rows):
        """Write a table of rows, or say why it could not be written."""
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                csv.writer(handle).writerows(rows)
            print(f"[EXPORT] {len(rows) - 1} row(s) -> {path}")
            return True
        except OSError as e:
            QMessageBox.critical(self, "Export CSV", f"Could not write:\n{e}")
            return False

    # ── ROI IMPORT ───────────────────────────────────────────────────────────
    def load_rois(self):
        """Read ROIs back from a polygon layer and measure them here.

        An ROI set outlives the session that drew it: the same rectangles over
        the same calibration sites can be run over next month's product, or
        over a second product of the same ground, and the two exports differ
        only by what the products say.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", "",
            "Polygon layers (*.shp *.gpkg *.geojson *.json);;All files (*)")
        if not path:
            return
        layer = QgsVectorLayer(path, "rois", "ogr")
        if not layer.isValid():
            QMessageBox.critical(self, "Load ROIs",
                                 f"QGIS could not open:\n{path}")
            return
        transform = None
        try:
            if layer.crs().authid() != self.proj_crs.authid():
                transform = QgsCoordinateTransform(layer.crs(), self.proj_crs,
                                                   QgsProject.instance())
        except Exception as e:
            print(f"[ROI] import transform: {e}")
        name_index = -1
        try:
            lowered = [f.name().lower() for f in layer.fields()]
            for candidate in ("name", "roi_name", "label", "site"):
                if candidate in lowered:
                    name_index = lowered.index(candidate)
                    break
        except Exception:
            pass

        added = skipped = 0
        for feature in layer.getFeatures():
            label = None
            if name_index >= 0:
                try:
                    value = feature.attributes()[name_index]
                    label = str(value) if value not in (None, "") else None
                except Exception:
                    label = None
            for ring in self._rings_from_geometry(feature.geometry()):
                if transform is not None:
                    ring = [(lambda p: (p.x(), p.y()))(
                        transform.transform(QgsPointXY(x, y)))
                        for x, y in ring]
                if self.add_roi(ring, "polygon", name=label, select=False):
                    added += 1
                else:
                    skipped += 1
        self.refresh_table()
        self.canvas.refresh()
        print(f"[ROI] imported {added} from {os.path.basename(path)}"
              + (f", {skipped} skipped" if skipped else ""))
        if not added:
            QMessageBox.warning(
                self, "Load ROIs",
                "Nothing was imported. The layer needs polygons, and they "
                "have to\nfall on the raster that is loaded.")

    @staticmethod
    def _rings_from_geometry(geometry):
        """Exterior rings of a (multi)polygon, open, as (x, y) tuples.

        Holes are dropped rather than honoured. A hole would have to be carried
        through the mask, the area and the shapefile alike, and an ROI over a
        uniform target is not the place for one -- if a corner has to come out,
        draw two ROIs.
        """
        if geometry is None:
            return []
        try:
            if geometry.isEmpty():
                return []
            parts = (geometry.asMultiPolygon() if geometry.isMultipart()
                     else [geometry.asPolygon()])
        except Exception as e:
            print(f"[ROI] geometry: {e}")
            return []
        rings = []
        for part in parts or []:
            if not part:
                continue
            points = [(p.x(), p.y()) for p in part[0]]
            if len(points) > 1 and points[0] == points[-1]:
                points = points[:-1]        # a stored ring is closed; ours are not
            if len(points) >= 3:
                rings.append(points)
        return rings

    # ── MAP TOOLS ────────────────────────────────────────────────────────────
    def init_map_tools(self):
        self._apply_canvas_crs()
        self.map_tools = {
            TOOL_SELECT: RoiSelectTool(self.canvas, self),
            TOOL_RECT: RoiMapTool(self.canvas, self, TOOL_RECT),
            TOOL_POLY: RoiMapTool(self.canvas, self, TOOL_POLY),
            TOOL_PAN: QgsMapToolPan(self.canvas),
            TOOL_ZOOM_IN: QgsMapToolZoom(self.canvas, False),
            TOOL_ZOOM_OUT: QgsMapToolZoom(self.canvas, True),
        }
        self.apply_map_tool()

    def current_map_tool(self):
        for mode, button in self.tool_buttons.items():
            if button.isChecked():
                return mode
        return TOOL_RECT

    def set_map_tool(self, mode):
        button = self.tool_buttons.get(mode)
        if button is None:
            return
        button.setChecked(True)
        self.apply_map_tool()

    def apply_map_tool(self, _checked=None):
        """Put the selected tool on the canvas.

        Read from the button row rather than from a signal argument, which
        arrives differently depending on how the change was made.
        """
        tools = getattr(self, "map_tools", None)
        if not tools:
            return              # called before init_map_tools
        mode = self.current_map_tool()
        self.canvas.setMapTool(tools[mode])
        cursor = {TOOL_PAN: Qt.OpenHandCursor,
                  TOOL_SELECT: Qt.ArrowCursor}.get(mode, Qt.CrossCursor)
        try:
            self.canvas.setCursor(cursor)
        except Exception:
            pass
        print(f"[TOOL] {mode}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
app = QApplication.instance() or QApplication(sys.argv)
win = RadiometricDashboard()
win.show()
