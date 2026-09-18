"""Drive DPQED_radial's Qt-side class with PyQt5/QGIS/GDAL stubbed out.

Covers what tests_radial_stats.py cannot: drawing an ROI, reading its pixels
off a raster, carrying an ROI set from one product to the next, and laying the
export's columns out in the order its fields were declared. Geometry and CRS
calls are stubs, so this proves the code paths execute and route correctly --
it says nothing about whether QGIS renders anything.

    python3 tests_radial_gui_stub.py       # exits non-zero on any assertion

The stubbing traps worth knowing, inherited from tests_rival_gui_stub.py: a
MagicMock caches its return_value, so one QgsRubberBand() would be handed back
for every ROI and a per-ROI difference could not be seen; and an attribute
fetched from a bare mock is a fresh mock each time, so `widget.hide.called` is
always False. Both get real stand-ins below.
"""
import os
import sys
import tempfile
import types

import numpy as np
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))

for name in ["PyQt5", "PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui",
             "qgis", "qgis.gui", "qgis.core"]:
    sys.modules[name] = MagicMock()

qw = sys.modules["PyQt5.QtWidgets"]
qg = sys.modules["qgis.gui"]
qc = sys.modules["PyQt5.QtCore"]


class _Base:
    """Stand-in for the Qt classes the module subclasses.

    Attributes are memoised: a fresh MagicMock per access would put every call
    on a different mock, and nothing could be asserted about any of them.
    """
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, n):
        m = MagicMock()
        object.__setattr__(self, n, m)
        return m


for module, names in ((qw, ["QMainWindow", "QWidget"]),
                      (qg, ["QgsMapTool", "QgsMapCanvasItem"]),
                      (qc, ["QObject"])):
    for name in names:
        setattr(module, name, type(name, (_Base,), {}))

qg.QgsMapCanvas = MagicMock(side_effect=lambda *a, **k: MagicMock())


def _tool_factory(kind):
    def make(*a, **k):
        m = MagicMock()
        m._tool, m._args = kind, a
        return m
    return MagicMock(side_effect=make)


qg.QgsMapToolPan = _tool_factory("pan")
qg.QgsMapToolZoom = _tool_factory("zoom")
# One rubber band per ROI, each remembering the points it was given: a cached
# return_value would make every ROI draw over the same item.
qg.QgsRubberBand = MagicMock(side_effect=lambda *a, **k: MagicMock(_points=[]))

sys.path.insert(0, HERE)
import DPQED_radial as R

print("imported OK;", type(R.win).__name__, "built")
win = R.win
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def ok(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}"
          + (f": {detail}" if detail else ""))
    if not condition:
        failures.append(f"{label}: {detail}")


# ── stand-ins the real classes need to be real ────────────────────────────────
class _PointXY:
    def __init__(self, x, y=None):
        if y is None:
            x, y = x.x(), x.y()
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def __repr__(self):
        return f"({self._x:.3f}, {self._y:.3f})"


class _Rect:
    def __init__(self, *b):
        self._b = tuple(float(v) for v in b)

    def xMinimum(self):
        return self._b[0]

    def yMinimum(self):
        return self._b[1]

    def xMaximum(self):
        return self._b[2]

    def yMaximum(self):
        return self._b[3]

    def __repr__(self):
        return (f"[{self._b[0]:.1f},{self._b[1]:.1f} .. "
                f"{self._b[2]:.1f},{self._b[3]:.1f}]")


class _CRS:
    def __init__(self, authid):
        self._a = authid

    def isValid(self):
        return True

    def isGeographic(self):
        return False

    def authid(self):
        return self._a

    def description(self):
        return self._a


class _Item:
    """QTableWidgetItem stand-in that remembers its text and its editability."""
    EDITABLE = 2

    def __init__(self, text=""):
        self._text, self._flags = str(text), self.EDITABLE

    def text(self):
        return self._text

    def flags(self):
        return self._flags

    def setFlags(self, flags):
        self._flags = flags

    def row(self):
        return self._row

    def column(self):
        return self._column


class _Table:
    """Enough QTableWidget to hold what refresh_table writes."""
    def __init__(self):
        self.cells, self.rows, self._current = {}, 0, -1

    def setRowCount(self, n):
        self.rows = n
        if n == 0:
            self.cells = {}

    def rowCount(self):
        return self.rows

    def setItem(self, row, column, item):
        item._row, item._column = row, column
        self.cells[(row, column)] = item

    def item(self, row, column):
        return self.cells.get((row, column))

    def text(self, row, column):
        item = self.cells.get((row, column))
        return item.text() if item else None

    def row_texts(self, row):
        return [self.text(row, c) for c in range(
            len(R.ROI_TABLE_COLUMNS) + len(R.TABLE_STATS))]

    def currentRow(self):
        return self._current

    def setCurrentCell(self, row, _column):
        self._current = row

    def clearSelection(self):
        # Deliberately does NOT move the current cell -- a real QTableWidget
        # does not either, and a stub that did hid exactly that bug once.
        pass

    def setHorizontalHeaderLabels(self, _):
        pass

    def blockSignals(self, _):
        pass


class _Combo:
    def __init__(self, items=(), index=0):
        self._items, self._idx = list(items), index

    def blockSignals(self, _):
        pass

    def clear(self):
        self._items, self._idx = [], -1

    def addItem(self, text, data=None):
        self._items.append((text, data))

    def count(self):
        return len(self._items)

    def setCurrentIndex(self, i):
        self._idx = i

    def setCurrentText(self, text):
        for index, (label, _) in enumerate(self._items):
            if label == text:
                self._idx = index
                return
        self._items.append((text, None))    # editable, as the real one is
        self._idx = len(self._items) - 1

    def setVisible(self, _):
        pass

    def currentIndex(self):
        return self._idx

    def currentText(self):
        return self._items[self._idx][0] if 0 <= self._idx < len(self._items) else ""

    def currentData(self):
        return self._items[self._idx][1] if 0 <= self._idx < len(self._items) else None


class _Check:
    def __init__(self, on=False):
        self._on = on

    def isChecked(self):
        return self._on

    def setChecked(self, v):
        self._on = bool(v)


class _Spin:
    """A spin box that holds a value and remembers whether it is on screen."""
    def __init__(self, value=0.0):
        self._value, self._visible = float(value), False

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = float(v)

    def setVisible(self, v):
        self._visible = bool(v)

    def isVisible(self):
        return self._visible


class _Button:
    """A real checkable button: the mock reports every button as checked."""
    def __init__(self, group):
        self._on, self._group, self._enabled = False, group, True

    def setCheckable(self, _):
        pass

    def setEnabled(self, v):
        self._enabled = bool(v)

    def isEnabled(self):
        return self._enabled

    def setToolTip(self, _):
        pass

    def setChecked(self, v):
        if v:
            for other in self._group.values():
                other._on = False
        self._on = bool(v)

    def isChecked(self):
        return self._on


R.QgsPointXY = _PointXY
R.QgsRectangle = _Rect
R.QTableWidgetItem = _Item

warned = []
qw.QMessageBox.warning = lambda *a, **k: warned.append(a[-1])
qw.QMessageBox.critical = lambda *a, **k: warned.append("CRITICAL: " + a[-1])
qw.QMessageBox.information = lambda *a, **k: None
qw.QMessageBox.question = lambda *a, **k: qw.QMessageBox.Yes

win.table = _Table()
win.domain_combo = _Combo([(label, value) for value, label in R.DOMAIN_CHOICES],
                          index=0)
win.cb_zero_data = _Check(False)
win.class_combo = _Combo(
    [(name, None) for name in (R.ROI_CLASS_ALL,) + R.ROI_CLASSES], index=1)
win.class_table = _Table()
win.lbl_context = MagicMock()
win.stats_band_combo = _Combo()
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.proj_crs = _CRS("EPSG:32644")
win._rebuild_transforms()


# ── a raster made of real numpy, read through a stubbed GDAL ──────────────────
def install_gdal(bands, geotransform, nodata=None):
    """Serve `bands` (a list of 2-D arrays) as a GDAL dataset.

    ReadAsArray really slices, so a window computed wrongly reads the wrong
    ground and the statistics say so -- which is the only way to tell a correct
    window from one that merely has the right shape.
    """
    reads = []

    def make_band(array):
        band = MagicMock()
        band.XSize, band.YSize = array.shape[1], array.shape[0]
        band.GetNoDataValue = MagicMock(return_value=nodata)

        def read(col0=0, row0=0, ncol=None, nrow=None, *a, **k):
            ncol = array.shape[1] if ncol is None else ncol
            nrow = array.shape[0] if nrow is None else nrow
            reads.append((col0, row0, ncol, nrow))
            window = array[row0:row0 + nrow, col0:col0 + ncol]
            if a:                       # the decimated read Normalize asks for
                out_x, out_y = a[0], a[1]
                window = window[:out_y, :out_x]
            return window
        band.ReadAsArray = read
        return band

    ds = MagicMock()
    ds.RasterXSize, ds.RasterYSize = bands[0].shape[1], bands[0].shape[0]
    ds.GetGeoTransform = MagicMock(return_value=geotransform)
    ds.GetRasterBand = MagicMock(
        side_effect=lambda i: make_band(bands[i - 1]) if 1 <= i <= len(bands)
        else None)
    osgeo = types.ModuleType("osgeo")
    gdal = types.ModuleType("osgeo.gdal")
    gdal.GA_ReadOnly = 0
    gdal.Open = MagicMock(return_value=ds)
    osgeo.gdal = gdal
    sys.modules["osgeo"], sys.modules["osgeo.gdal"] = osgeo, gdal
    return reads


def fake_layer(source="/data/gcov.vrt", crs="EPSG:32644", bands=2):
    layer = MagicMock()
    layer.isValid.return_value = True
    layer.source.return_value = source
    layer.crs.return_value = _CRS(crs)
    layer.extent.return_value = _Rect(0, 0, 1, 1)
    # Real numbers again: a MagicMock posting floats to nothing, and the ENL
    # side suggestion would come back None with the failure printed, not raised.
    layer.rasterUnitsPerPixelX.return_value = 30.0
    layer.rasterUnitsPerPixelY.return_value = 30.0
    provider = MagicMock()
    provider.bandCount.return_value = bands
    # Real numbers, not a mock: `lo, hi = provider.cumulativeCut(...)` unpacks
    # its result, and a mock unpacks to nothing -- which apply_bands catches
    # and prints, so the stretch path would look exercised and never run.
    provider.cumulativeCut = MagicMock(return_value=(0.0, 1.0))
    provider.bandStatistics = MagicMock(
        return_value=MagicMock(minimumValue=0.0, maximumValue=1.0))
    layer.dataProvider.return_value = provider
    return layer


# A 200 x 200 scene on a 30 m UTM grid, with one 6 x 6 patch 10 dB above the
# background. Both bands are exponential, i.e. single-look intensity.
rng = np.random.default_rng(7)
GT = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
hh = rng.exponential(0.01, (200, 200))
hv = rng.exponential(0.002, (200, 200))
hh[4:10, 10:16] = rng.exponential(0.1, (6, 6))       # -10 dB
hv[4:10, 10:16] = rng.exponential(0.02, (6, 6))      # -17 dB
reads = install_gdal([hh, hv], GT)

win.raster_layer = fake_layer()
win.raster_path = "/data/gcov.vrt"
win.band_labels = ["HHHH", "HVHV"]
win.measure_bands = list(enumerate(win.band_labels, start=1))
win.factor_band = None
win.band_prefixes = [R.band_prefix(label, i)
                     for i, label in enumerate(win.band_labels, start=1)]
win.backscatter_combo = _Combo(
    [(label, value) for value, label in R.BACKSCATTER_CHOICES], index=0)
win.backscatter_combo.setEnabled = lambda v: setattr(
    win.backscatter_combo, "_enabled", bool(v))

# ── 1. an ROI reads the ground it was drawn over ──────────────────────────────
print("\n── measuring an ROI ──")
patch = R.rect_ring(500300.0, 3999700.0, 500480.0, 3999880.0)
roi = win.add_roi(patch, "rect")
ok("the ROI was accepted", roi is not None)
check("every pixel of the window is in it", roi["npix"], 36)
check("the window read is the patch", reads[-1], (10, 4, 6, 6))
check("statistics are keyed by convention first", sorted(roi["stats"]),
      [R.BACKSCATTER_GAMMA0])
check("with a block per measured band",
      sorted(roi["stats"][R.BACKSCATTER_GAMMA0]), ["HHHH", "HVHV"])
ok("HH reads the patch's own gamma0",
   abs(R.roi_stats(roi, "HHHH")["mean_db"] + 10.0) < 1.5,
   f"{R.roi_stats(roi, 'HHHH')['mean_db']:.2f} dB")
ok("HV reads its own, 7 dB below",
   abs(R.roi_stats(roi, "HVHV")["mean_db"] + 17.0) < 1.5,
   f"{R.roi_stats(roi, 'HVHV')['mean_db']:.2f} dB")
ok("single-look speckle reads about one look",
   abs(R.roi_stats(roi, "HHHH")["enl"] - 1.0) < 0.5,
   f"ENL {R.roi_stats(roi, 'HHHH')['enl']:.2f}")
check("area from the ring, not the pixels", round(roi["area_m2"]), 32400)
check("the raster is recorded with the numbers", roi["src"], "gcov.vrt")
check("and so is the domain", roi["domain"], R.DOMAIN_POWER)

# The background, to prove the ROI is not simply averaging the whole scene.
background = win.add_roi(R.rect_ring(503000.0, 3994000.0, 503180.0, 3994180.0),
                         "rect")
ok("a second ROI elsewhere reads the background",
   abs(R.roi_stats(background, "HHHH")["mean_db"] + 20.0) < 1.5,
   f"{R.roi_stats(background, 'HHHH')['mean_db']:.2f} dB")
check("two ROIs, numbered in order",
      [r["roi"] for r in win.rois], [1, 2])

# ── 2. ROIs that are not measurements ────────────────────────────────────────
print("\n── ROIs that are refused ──")
before = len(win.rois)
check("a click with no drag", win.add_roi(
    R.rect_ring(500300.0, 3999700.0, 500300.0, 3999700.0), "rect"), None)
check("a sliver under one pixel", win.add_roi(
    R.rect_ring(500300.0, 3999700.0, 500301.0, 3999701.0), "rect"), None)
check("an ROI off the edge of the scene", win.add_roi(
    R.rect_ring(900000.0, 3000000.0, 900500.0, 3000500.0), "rect"), None)
check("none of them reached the table", len(win.rois), before)
# The read cap, lowered so a test scene can exceed it: a window past it is
# refused rather than read, because every pixel of it would be read for every
# band at full resolution.
del warned[:]
cap = R.ROI_MAX_PIXELS
R.ROI_MAX_PIXELS = 100
huge = win.add_roi(R.rect_ring(500000.0, 3999000.0, 501000.0, 4000000.0),
                   "rect")
R.ROI_MAX_PIXELS = cap
check("an ROI larger than the read cap is refused", huge, None)
ok("and says so rather than reading it",
   any("pixels" in str(w) for w in warned), str(warned))

# ── 3. the drawing tools ─────────────────────────────────────────────────────
print("\n── drawing ──")
drawn = []
win.add_roi = lambda ring, kind, **k: drawn.append((kind, ring)) or {"roi": 0}


class _Event:
    def __init__(self, x, y, button=None):
        self._pos, self.button_ = (x, y), button

    def pos(self):
        return self._pos

    def button(self):
        return self.button_


def drive(tool, points):
    """Feed a tool map coordinates, the way the canvas would."""
    tool.toMapCoordinates = lambda pos: _PointXY(*pos)


rect_tool = R.RoiMapTool(win.canvas, win, R.TOOL_RECT)
drive(rect_tool, None)
R.Qt.LeftButton, R.Qt.RightButton = "L", "R"
rect_tool.canvasPressEvent(_Event(500000.0, 4000000.0, "L"))
rect_tool.canvasMoveEvent(_Event(500100.0, 3999900.0))
rect_tool.canvasReleaseEvent(_Event(500120.0, 3999880.0, "L"))
check("a drag makes a rectangle", drawn[-1][0], "rect")
check("spanned by its two corners", drawn[-1][1],
      R.rect_ring(500000.0, 4000000.0, 500120.0, 3999880.0))
ok("and the preview is cleared", rect_tool.anchor is None)

del drawn[:]
poly_tool = R.RoiMapTool(win.canvas, win, R.TOOL_POLY)
drive(poly_tool, None)
win.canvas.mapUnitsPerPixel = MagicMock(return_value=10.0)
for x, y in [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)]:
    poly_tool.canvasPressEvent(_Event(x, y, "L"))
# The double-click's own press lands on the last corner again, within a few
# screen pixels: one corner, not two, or the ring carries a zero-length edge.
poly_tool.canvasPressEvent(_Event(4.0, 302.0, "L"))
poly_tool.canvasDoubleClickEvent(_Event(4.0, 302.0, "L"))
check("clicked corners make a polygon", drawn[-1][0], "polygon")
check("the double-click's repeat is merged away", drawn[-1][1],
      [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)])

del drawn[:]
for x, y in [(0.0, 0.0), (300.0, 0.0)]:
    poly_tool.canvasPressEvent(_Event(x, y, "L"))
poly_tool.canvasPressEvent(_Event(0.0, 0.0, "R"))
check("two corners are not a polygon", drawn, [])

del drawn[:]
for x, y in [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0)]:
    poly_tool.canvasPressEvent(_Event(x, y, "L"))
poly_tool.cancel()
poly_tool.canvasPressEvent(_Event(0.0, 0.0, "R"))
check("Escape abandons what was being drawn", drawn, [])
win.add_roi = R.RadiometricDashboard.add_roi.__get__(win, R.RadiometricDashboard)

# ── 4. one ROI set, two products ─────────────────────────────────────────────
print("\n── carrying ROIs to the next product ──")
first = [dict(R.roi_stats(r, "HHHH")) for r in win.rois]
brighter = [hh * 2.0, hv * 2.0]          # the same scene, 3 dB up
install_gdal(brighter, GT)
R.QgsRasterLayer = MagicMock(
    side_effect=lambda path, name: fake_layer(source=path))
win.populate_band_picker = MagicMock()   # the picker is exercised in section 8
win.load_path("/data/second_gcov.vrt")
check("the ROIs survived the load", len(win.rois), 2)
check("and were re-measured against it",
      [r["src"] for r in win.rois], ["second_gcov.vrt"] * 2)
gaps = [R.roi_stats(r, "HHHH")["mean_db"] - f["mean_db"]
        for r, f in zip(win.rois, first)]
ok("a product 3 dB brighter reads 3 dB brighter",
   all(abs(gap - 3.0103) < 1e-6 for gap in gaps),
   ", ".join(f"{gap:+.4f} dB" for gap in gaps))

# ── 5. the working CRS follows the raster, and the ROIs follow it ────────────
print("\n── CRS ──")
win.canvas.setDestinationCrs.reset_mock()
moved = _CRS("EPSG:32643")
R.QgsCoordinateTransform = MagicMock(side_effect=lambda *a, **k: MagicMock(
    transform=lambda point: _PointXY(point.x() + 1000.0, point.y() + 2000.0)))
before_rings = [list(r["ring"]) for r in win.rois]
check("a projected CRS is adopted", win.adopt_working_crs(moved), True)
check("the canvas is re-pinned to it",
      win.canvas.setDestinationCrs.call_args[0][0], moved)
check("the ROIs came with it",
      [r["ring"][0] for r in win.rois],
      [(x + 1000.0, y + 2000.0) for (x, y) in
       [ring[0] for ring in before_rings]])


class _Geographic(_CRS):
    def isGeographic(self):
        return True


check("a geographic CRS is refused -- an area in square degrees is not an area",
      win.adopt_working_crs(_Geographic("EPSG:4326")), False)
win.proj_crs = _CRS("EPSG:32644")

# ── 6. the table ─────────────────────────────────────────────────────────────
print("\n── table ──")
win.stats_band_combo = _Combo([(label, None) for label in win.band_labels],
                              index=0)
win.refresh_table()
check("a row per ROI", win.table.rowCount(), 2)
check("headers name every column",
      len(R.RadiometricDashboard._table_headers()),
      len(R.ROI_TABLE_COLUMNS) + len(R.TABLE_STATS))
check("the ROI's own columns come first",
      win.table.row_texts(0)[:4], ["1", "ROI 1", "vegetation", "rect"])
name_column = R.ROI_TABLE_COLUMNS.index("name")
class_column = R.ROI_TABLE_COLUMNS.index("class")
ok("the name and the class are the editable cells, and nothing else",
   win.table.item(0, name_column).flags() == _Item.EDITABLE
   and win.table.item(0, class_column).flags() == _Item.EDITABLE
   and win.table.item(0, 0).flags() != _Item.EDITABLE)
win.table.item(0, name_column)._text = "calibration site"
win.on_item_changed(win.table.item(0, name_column))
check("a typed label sticks to the ROI", win.rois[0]["name"], "calibration site")
win.stats_band_combo.setCurrentIndex(1)
win.refresh_table()
hh_row = None
check("switching band re-reads the table, not the raster",
      win.table.row_texts(0)[:4],
      ["1", "calibration site", "vegetation", "rect"])
mean_column = len(R.ROI_TABLE_COLUMNS) + R.TABLE_STATS.index("mean_db")
check("and shows that band's figure",
      win.table.text(0, mean_column),
      R.format_stat(R.roi_stats(win.rois[0], "HVHV")["mean_db"], "{:.2f}"))
win.stats_band_combo.setCurrentIndex(0)
win.refresh_table()

# ── 7. export columns ────────────────────────────────────────────────────────
print("\n── export ──")
R.QgsFields = list
R.QgsField = lambda name, *a: name
fields, plan = win.export_fields()
check("a column per ROI field, and per band and statistic",
      len(fields), len(R.ROI_FIELDS) + 2 * len(R.STAT_KEYS))
check("the plan matches the fields one for one", len(plan), len(fields))
check("fields and plan agree on every name",
      [entry[0] for entry in plan], list(fields))
ok("nothing over DBF's cap", max(len(name) for name in fields) <= 10,
   max(fields, key=len))
check("no two columns collide", len({n.lower() for n in fields}), len(fields))
ok("the polarizations name their columns",
   "HH_mean_db" in fields and "HV_mean_db" in fields,
   ", ".join(n for n in fields if n.endswith("mean_db")))

attributes = win.feature_attributes(win.rois[0], plan)
check("an attribute per field", len(attributes), len(fields))
check("the ROI's own fields lead, in ROI_FIELDS order",
      attributes[:3], [1, "calibration site", "rect"])
check("class among them, where ROI_FIELDS puts it",
      attributes[[n for n, _ in R.ROI_FIELDS].index("class")], "vegetation")
check("then the statistics, band by band",
      attributes[fields.index("HH_mean_db")],
      R.roi_stats(win.rois[0], "HHHH")["mean_db"])
check("an undefined statistic is NULL, not the string 'nan'",
      R.dbf_safe(float("nan"), "double"), None)

empty_roi = {"roi": 9, "name": "unmeasured", "kind": "rect", "stats": {}}
check("an unmeasured ROI still fills its row",
      len(win.feature_attributes(empty_roi, plan)), len(fields))

with tempfile.TemporaryDirectory() as tmp:
    csv_path = os.path.join(tmp, "stats.csv")
    ok("the statistics CSV writes", win._write_csv(csv_path))
    import csv as _csv
    with open(csv_path, encoding="utf-8-sig") as handle:
        rows = list(_csv.reader(handle))
    check("a row per ROI and band, plus a header", len(rows), 1 + 2 * 2)
    check("under names DBF could not hold",
          rows[0][-len(R.STAT_KEYS):], list(R.STAT_KEYS))
    check("the band is named in each row",
          [row[len(R.ROI_FIELDS)] for row in rows[1:]],
          ["HHHH", "HVHV", "HHHH", "HVHV"])

# ── 8. the band picker and the stretch ───────────────────────────────────────
print("\n── bands ──")
win.populate_band_picker = R.RadiometricDashboard.populate_band_picker.__get__(
    win, R.RadiometricDashboard)
layer = fake_layer(bands=2)
layer.bandName.side_effect = lambda b: ["HHHH", "HVHV"][b - 1]
captured = []
R.QgsMultiBandColorRenderer = lambda p, r, g, b: (
    captured.append(("rgb", r, g, b)) or MagicMock())
R.QgsSingleBandGrayRenderer = lambda p, b: (
    captured.append(("grey", b)) or MagicMock())
win.raster_layer = layer
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(layer)
check("every slot offers every band",
      [[text for text, _ in combo._items] for combo in win.band_combos],
      [["1: HHHH", "2: HVHV"]] * 3)
check("the default renders grey", captured[-1], ("rgb", 1, 1, 1))
check("the stats band list follows the raster",
      [text for text, _ in win.stats_band_combo._items], ["1: HHHH", "2: HVHV"])
check("columns are named for the polarizations", win.band_prefixes, ["HH", "HV"])

print("\n── normalize ──")
install_gdal([hh, hv], GT)
win.canvas.extent = MagicMock(return_value=_Rect(500000, 3994000, 506000, 4000000))
layer.rasterUnitsPerPixelX.return_value = 30.0
layer.rasterUnitsPerPixelY.return_value = 30.0
layer.extent.return_value = _Rect(500000, 3994000, 506000, 4000000)
provider = layer.dataProvider()
provider.cumulativeCut = MagicMock(return_value=(0.0, 0.5))
provider.bandStatistics = MagicMock(
    return_value=MagicMock(minimumValue=0.0, maximumValue=0.5))
win.norm_bounds = {}
win.normalize_to_view()
ok("the stretch is pinned, per band", set(win.norm_bounds) == {1},
   str(sorted(win.norm_bounds)))
pinned = dict(win.norm_bounds)
win.canvas.extent = MagicMock(return_value=_Rect(500000, 3999000, 501000, 4000000))
win.apply_bands()
check("panning does not re-measure it", win.norm_bounds, pinned)
lo, hi = pinned[1]
ok("and it lands inside the data's range", 0.0 <= lo < hi <= 0.5,
   f"{lo:.6g}..{hi:.6g}")

# ── 9. the map tools ─────────────────────────────────────────────────────────
print("\n── map tools ──")
group = {}
win.tool_buttons = {mode: _Button(group) for mode in R.MAP_TOOLS}
group.update(win.tool_buttons)
win.tool_buttons[R.TOOL_RECT].setChecked(True)
win.init_map_tools()
for mode in R.MAP_TOOLS:
    win.set_map_tool(mode)
    applied = win.canvas.setMapTool.call_args[0][0]
    ok(f"{mode} is applied to the canvas", applied is win.map_tools[mode])
    checked = [m for m, b in win.tool_buttons.items() if b.isChecked()]
    ok(f"{mode} is the only tool selected", checked == [mode], str(checked))
check("a distinct tool object per button",
      len({id(win.map_tools[m]) for m in R.MAP_TOOLS}), len(R.MAP_TOOLS))
check("pan is the pan tool", win.map_tools[R.TOOL_PAN]._tool, "pan")
check("zoom out is the out variant",
      (win.map_tools[R.TOOL_ZOOM_IN]._args[1],
       win.map_tools[R.TOOL_ZOOM_OUT]._args[1]), (False, True))
ok("every ROI tool draws, in its own mode",
   all(isinstance(win.map_tools[m], R.RoiMapTool)
       and win.map_tools[m].mode == m for m in R.DRAWING_TOOLS),
   str(R.DRAWING_TOOLS))

# ── the point buffer ─────────────────────────────────────────────────────────
# One click, one square of a stated size on the ground. The size is read at the
# click, so the box is a setting for the next ROI and never for one drawn.
print("\n── point buffer ──")
check("a square is centred on the click, not cornered at it",
      R.square_ring(100.0, 200.0, 20.0),
      [(90.0, 190.0), (110.0, 190.0), (110.0, 210.0), (90.0, 210.0)])
check("its area is the side squared",
      R.ring_area(R.square_ring(0.0, 0.0, 30.0)), 900.0)
for bad in (0.0, -10.0, None, float("nan"), "wide"):
    check(f"{bad!r} is not a side length", R.square_ring(0.0, 0.0, bad), None)

win.point_side_spin = _Spin(R.POINT_SIDE_DEFAULT_M)
check("the box starts on the default", win.point_side(),
      R.POINT_SIDE_DEFAULT_M)
win.set_map_tool(R.TOOL_POINT)
ok("picking the tool shows the side-length box",
   win.point_side_spin.isVisible())
win.set_map_tool(R.TOOL_RECT)
ok("and leaving it takes the box away",
   not win.point_side_spin.isVisible())

win.set_map_tool(R.TOOL_POINT)
win.point_side_spin.setValue(120.0)
before = len(win.rois)
win.map_tools[R.TOOL_POINT].place_point(500400.0, 3999800.0)
check("a click makes one ROI", len(win.rois) - before, 1)
placed = win.rois[-1]
check("of the kind that says how it was made", placed["kind"], R.TOOL_POINT)
check("with the side the box was showing at the click",
      round(R.ring_area(placed["ring"])), 120 * 120)
centre = R.ring_centroid(placed["ring"])
ok("centred on the click",
   abs(centre[0] - 500400.0) < 1e-6 and abs(centre[1] - 3999800.0) < 1e-6,
   str(centre))
check("and it carries the picker's class, like any other ROI",
      placed["class"], win.roi_class())

# Suggest sizes the square from the raster's own posting.
win.btn_point_suggest = _Spin(0.0)
win.raster_layer = fake_layer()
del warned[:]
win.suggest_point_side()
pixel = win.raster_pixel_m()
ok("the posting comes from the raster", pixel and pixel > 0, str(pixel))
check("and Suggest sets the side that posting needs",
      win.point_side(), R.enl_side_for_precision(pixel))
ok("which is a real ENL sample size, not a round number",
   R.enl_precision((win.point_side() / pixel) ** 2) <= R.ENL_TARGET_ERROR,
   f"+/-{R.enl_precision((win.point_side() / pixel) ** 2):.1%}")
saved_layer, win.raster_layer = win.raster_layer, None
del warned[:]
win.suggest_point_side()
ok("with no raster it says so rather than guessing",
   any("Load a raster" in str(w) for w in warned), str(warned))
win.raster_layer = saved_layer

win.point_side_spin.setValue(240.0)
check("changing the box does not touch the ROI already drawn",
      round(R.ring_area(placed["ring"])), 120 * 120)
win.map_tools[R.TOOL_POINT].place_point(500400.0, 3999800.0)
check("only the next one", round(R.ring_area(win.rois[-1]["ring"])), 240 * 240)

# ── 9b. selecting an ROI on the canvas, and its number beside it ─────────────
print("\n── select, and the numbers on the canvas ──")
install_gdal([hh, hv], GT)
sel_layer = fake_layer(bands=2)
sel_layer.bandName.side_effect = lambda b: ["HHHH", "HVHV"][b - 1]
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(sel_layer)
win.raster_layer = sel_layer
win.rois = []
win._next_roi_id = 1
win.roi_labels = {}

outer = win.add_roi(R.rect_ring(500000.0, 3994000.0, 502000.0, 3996000.0), "rect")
inner = win.add_roi(R.rect_ring(500800.0, 3994800.0, 501000.0, 3995000.0), "rect")
check("two ROIs, the second inside the first",
      [r["roi"] for r in win.rois], [1, 2])

# Every ROI gets a number drawn at its centre.
check("a label per ROI", sorted(win.roi_labels), [1, 2])
label = win.roi_labels[1]
check("saying which ROI it is", label.text, "1")
check("placed at the ROI's centre",
      (round(label.point.x()), round(label.point.y())), (501000, 3995000))

# Selecting: the smallest ROI under the click wins, so the inner one is
# reachable even though the outer one also holds the point.
picked = win.select_roi_at(500900.0, 3994900.0)
check("the click picks the smaller of the two", picked["roi"], 2)
check("and the table row follows it", win.table.currentRow(), 1)
check("a click only the outer one holds picks the outer one",
      win.select_roi_at(500100.0, 3994100.0)["roi"], 1)
check("and moves the row again", win.table.currentRow(), 0)
check("a click on open ground selects nothing",
      win.select_roi_at(400000.0, 3000000.0), None)
check("and clears the row", win.table.currentRow(), -1)

# The selected ROI's number takes the selected colour, as its outline does.
win.select_roi_at(500900.0, 3994900.0)
win.redraw_rois()
check("the selected ROI's number is the selected colour",
      win.roi_labels[2].colour, R.ROI_COLOR_SELECTED)
check("and the others are not", win.roi_labels[1].colour, R.ROI_COLOR)

# Deleting takes the number with it: a label left behind would name whichever
# ROI was renumbered into its place.
win.table.setCurrentCell(1, 0)
win.delete_roi()
check("the ROI is gone", [r["roi"] for r in win.rois], [1])
check("and so is its number", sorted(win.roi_labels), [1])
win.clear_rois()
check("clearing takes every number", win.roi_labels, {})

# ── 10. reading an ROI set back in ───────────────────────────────────────────
print("\n── importing ROIs ──")


def fake_geometry(rings, multi=False):
    geometry = MagicMock()
    geometry.isEmpty.return_value = False
    geometry.asPolygon.return_value = [[_PointXY(x, y) for x, y in rings[0]]]
    geometry.asMultiPolygon.return_value = [
        [[_PointXY(x, y) for x, y in ring]] for ring in rings]
    geometry.isMultipart.return_value = multi
    return geometry


closed = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
check("a stored ring's closing vertex is dropped",
      R.RadiometricDashboard._rings_from_geometry(fake_geometry([closed])),
      [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]])
check("every part of a multipolygon becomes an ROI",
      len(R.RadiometricDashboard._rings_from_geometry(
          fake_geometry([closed, closed], multi=True))), 2)
check("a line is not an ROI",
      R.RadiometricDashboard._rings_from_geometry(
          fake_geometry([[(0.0, 0.0), (1.0, 1.0)]])), [])
check("no geometry at all",
      R.RadiometricDashboard._rings_from_geometry(None), [])

# ── 10b. sigma0, through the product's own RTC factor ────────────────────────
# A GCOV loaded from its '.h5' carries rtcGammaToSigmaFactor as a band. It is
# not a channel: it must stay out of the measured bands and out of the export's
# columns, and selecting sigma0 must move every figure by exactly the factor.
print("\n── sigma0 ──")
factor = np.full((200, 200), 2.0)           # +3.0103 dB everywhere
install_gdal([hh, hv, factor], GT)
sigma_layer = fake_layer(bands=3)
sigma_layer.bandName.side_effect = lambda b: [
    "HHHH", "HVHV", "rtcGammaToSigmaFactor"][b - 1]
win.raster_layer = sigma_layer
win.raster_path = "/data/gcov.vrt"
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(sigma_layer)

check("the factor is found", win.factor_band, 3)
check("and is not one of the measured bands",
      [label for _, label in win.measure_bands], ["1: HHHH", "2: HVHV"])
check("the R/G/B picker still offers it, to look at",
      [text for text, _ in win.band_combos[0]._items],
      ["1: HHHH", "2: HVHV", "3: rtcGammaToSigmaFactor"])
check("the stats band list leaves it out",
      [text for text, _ in win.stats_band_combo._items], ["1: HHHH", "2: HVHV"])
check("no factor column is exported", win.band_prefixes, ["HH", "HV"])
ok("sigma0 became selectable", win.backscatter_combo._enabled)

win.rois = []
win._next_roi_id = 1
patch_ring = R.rect_ring(500300.0, 3999700.0, 500480.0, 3999880.0)
gamma_roi = win.add_roi(patch_ring, "rect")
check("gamma0 by default", gamma_roi["backscat"], R.BACKSCATTER_GAMMA0)
# recompute_all measures IN PLACE, so the gamma0 figures have to be copied out
# before they are overwritten -- comparing the dict with itself afterwards
# would show no movement whatever the conversion did.
before = {band: dict(stats) for band, stats
          in gamma_roi["stats"][R.BACKSCATTER_GAMMA0].items()}

# Switching convention must not touch the disk: both were measured when the
# ROI was, so this is a change of which stored answer is being shown.
reads_before = len(reads)
win.backscatter_combo.setCurrentIndex(1)
win.apply_backscatter()
check("switching to sigma0 re-reads nothing", len(reads), reads_before)
sigma_roi = win.rois[0]
check("now recorded as sigma0", sigma_roi["backscat"], R.BACKSCATTER_SIGMA0)
check("and both conventions are held at once",
      R.roi_conventions(sigma_roi),
      [R.BACKSCATTER_GAMMA0, R.BACKSCATTER_SIGMA0])
for band in ("1: HHHH", "2: HVHV"):
    moved = abs(R.roi_stats(sigma_roi, band)["mean_db"]
                - before[band]["mean_db"] - 3.0103) < 1e-3
    ok(f"{band} moved by the factor, exactly",
       moved, f"{before[band]['mean_db']:.4f} dB gamma0 -> "
              f"{R.roi_stats(sigma_roi, band)['mean_db']:.4f} dB sigma0")
check("the same pixels were measured",
      R.roi_stats(sigma_roi, "1: HHHH")["n"], before["1: HHHH"]["n"])
ok("speckle statistics did not move, the factor being flat here",
   abs(R.roi_stats(sigma_roi, "1: HHHH")["enl"]
       - before["1: HHHH"]["enl"]) < 1e-9)

# The convention rides with the numbers, in both exports.
R.QgsFields = list
R.QgsField = lambda name, *a: name
fields, plan = win.export_fields()
ok("no factor columns in the shapefile",
   not any(f.upper().startswith(("RTC", "GAMMA")) for f in fields),
   ", ".join(fields[len(R.ROI_FIELDS):][:4]))
check("the convention is a column", "backscat" in fields, True)
check("and carries the right value",
      win.feature_attributes(sigma_roi, plan)[fields.index("backscat")],
      R.BACKSCATTER_SIGMA0)

# A plain GeoTIFF has no factor, so sigma0 cannot be offered at all.
install_gdal([hh, hv], GT)
plain = fake_layer(bands=2)
plain.bandName.side_effect = lambda b: ["HHHH", "HVHV"][b - 1]
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(plain)
check("no factor band", win.factor_band, None)
ok("so sigma0 is not selectable", not win.backscatter_combo._enabled)
win.raster_layer = plain
win.backscatter_combo.setCurrentIndex(1)     # as if it had been left there
win.rois = []
win._next_roi_id = 1
no_factor = win.add_roi(patch_ring, "rect")
check("and a figure is never labelled sigma0 without the conversion",
      no_factor["backscat"], R.BACKSCATTER_GAMMA0)
win.apply_backscatter()
check("nor does switching the selector drag it into one",
      win.rois[0]["backscat"], R.BACKSCATTER_GAMMA0)
check("it simply has the one convention", R.roi_conventions(win.rois[0]),
      [R.BACKSCATTER_GAMMA0])
win.backscatter_combo.setCurrentIndex(0)

# ── 10c. the incidence angle per ROI ─────────────────────────────────────────
# A band named incidenceAngle is reported per ROI and never measured as
# backscatter. The fixture's band is a plane, so the value at the ROI centre is
# something the test can compute independently.
print("\n── incidence angle ──")
gt_x, gt_y, step = GT[0], GT[3], GT[1]
cols = np.arange(200)
rows = np.arange(200)
xs = gt_x + step * (cols + 0.5)
ys = gt_y - step * (rows + 0.5)
incidence = 34.0 + 3e-5 * (xs[None, :] - gt_x) + 1e-5 * (gt_y - ys[:, None])
install_gdal([hh, hv, factor, incidence], GT)
inc_layer = fake_layer(bands=4)
inc_layer.bandName.side_effect = lambda b: [
    "HHHH", "HVHV", "rtcGammaToSigmaFactor", "incidenceAngle"][b - 1]
win.raster_layer = inc_layer
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(inc_layer)

check("the incidence band is found", win.incidence_band, 4)
check("and is not measured as backscatter",
      [label for _, label in win.measure_bands], ["1: HHHH", "2: HVHV"])
check("nor exported as a column", win.band_prefixes, ["HH", "HV"])

win.rois = []
win._next_roi_id = 1
inc_roi = win.add_roi(patch_ring, "rect")
centre = R.ring_centroid(patch_ring)
expected = 34.0 + 3e-5 * (centre[0] - gt_x) + 1e-5 * (gt_y - centre[1])
ok("the ROI reports its centre's incidence",
   abs(inc_roi["inc_deg"] - expected) < 0.01,
   f"{inc_roi['inc_deg']:.4f} vs {expected:.4f} deg")
check("it is a column of the table",
      win.table.text(0, R.ROI_TABLE_COLUMNS.index("inc_deg")),
      R.format_stat(inc_roi["inc_deg"], "{:.2f}"))
check("and of the export", "inc_deg" in [n for n, _ in R.ROI_FIELDS], True)

# A raster with no incidence at all leaves the column empty rather than
# inventing a number.
install_gdal([hh, hv], GT)
bare = fake_layer(bands=2)
bare.bandName.side_effect = lambda b: ["HHHH", "HVHV"][b - 1]
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(bare)
win.raster_layer = bare
win.rois = []
win._next_roi_id = 1
check("no incidence band", win.incidence_band, None)
check("so no angle is reported", win.add_roi(patch_ring, "rect")["inc_deg"], None)

# The other source: a '.h5' product's own cube, sampled at the ROI centre.
print("\n── incidence from a cube ──")
cube_x = np.linspace(gt_x - step, gt_x + 201 * step, 9)
cube_y = np.linspace(gt_y + step, gt_y - 201 * step, 7)
cube_v = 34.0 + 3e-5 * (cube_x[None, :] - gt_x) + 1e-5 * (gt_y - cube_y[:, None])
win.incidence_cube = (cube_x, cube_y, cube_v)
win.rois = []
win._next_roi_id = 1
from_cube = win.add_roi(patch_ring, "rect")
ok("the cube is sampled at the ROI centre",
   abs(from_cube["inc_deg"] - expected) < 0.01,
   f"{from_cube['inc_deg']:.4f} vs {expected:.4f} deg")
far = win.add_roi(R.rect_ring(9e6, 9e6, 9e6 + 200, 9e6 + 200), "rect")
check("an ROI off the cube gets no angle rather than an extrapolated one",
      far["inc_deg"] if far else None, None)
win.incidence_cube = None

# ── 10d. statistics per class ────────────────────────────────────────────────
# Ten vegetation ROIs and eight water ones: the whole point is that they are
# summarised apart.
print("\n── by class ──")
install_gdal([hh, hv], GT)
plain2 = fake_layer(bands=2)
plain2.bandName.side_effect = lambda b: ["HHHH", "HVHV"][b - 1]
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.stats_band_combo = _Combo()
win.populate_band_picker(plain2)
win.raster_layer = plain2
win.rois = []
win._next_roi_id = 1

# Two patches of the scene, drawn as two classes: the -10 dB patch as
# vegetation, the -20 dB background as water.
bright = R.rect_ring(500300.0, 3999700.0, 500480.0, 3999880.0)
dark = R.rect_ring(503000.0, 3994000.0, 503180.0, 3994180.0)
win.class_combo.setCurrentIndex(1)                 # vegetation
check("the picker sets the class of what is drawn next",
      win.roi_class(), "vegetation")
veg = win.add_roi(bright, "rect")
check("and the ROI carries it", veg["class"], "vegetation")
win.class_combo.setCurrentIndex(2)                 # water
wet = win.add_roi(dark, "rect")
check("switching the picker switches what the next ROI is",
      wet["class"], "water")

band = win.stats_band()
summary = dict(R.class_summary(win.rois, band))
check("a row per class, plus all", list(summary),
      ["vegetation", "water", "all"])
ok("vegetation reads the bright patch",
   abs(summary["vegetation"]["mean_db"] + 10.0) < 1.5,
   f"{summary['vegetation']['mean_db']:.2f} dB")
ok("water reads the dark one",
   abs(summary["water"]["mean_db"] + 20.0) < 1.5,
   f"{summary['water']['mean_db']:.2f} dB")
ok("the spread within a class is not the gap between them",
   summary["all"]["spread_db"] > 8.0
   and not np.isfinite(summary["vegetation"]["spread_db"]),
   f"all {summary['all']['spread_db']:.2f} dB, "
   f"one ROI per class so no within-class spread")

# The filter is the same dropdown, so what is on screen right now is the water
# ROI alone. Everything measured is still there; only the view is narrowed.
win.on_class_filter()
check("the filter shows its class alone",
      [roi["class"] for roi in win.shown_rois()], ["water"])
check("and the table holds that one row", win.table.rowCount(), 1)
check("naming the ROI that is in it", win.table.text(0, 0), "2")
check("the other ROI is not gone, just not shown", len(win.rois), 2)
check("so a shapefile still carries both",
      len(R.shapefile_records(win.rois, win.export_fields()[1])), 2)
check("and so does the CSV: a header and a row per ROI and band",
      len(R.stat_rows(win.rois,
                      [label for _, label in win.measure_bands])), 5)
win.class_combo.setCurrentIndex(0)                 # (all)
win.on_class_filter()
check("switching to 'all' brings them back", win.table.rowCount(), 2)
check("and nothing may be drawn into 'all'",
      win.roi_class(), R.ROI_CLASS_DEFAULT)

# The GUI table shows the same thing.
win.refresh_table()
check("the by-class table has a row per class and the total",
      win.class_table.rowCount(), 3)
check("naming the classes", [win.class_table.text(r, 0) for r in range(3)],
      ["vegetation", "water", "all"])
check("with their ROI counts",
      [win.class_table.text(r, 1) for r in range(3)], ["1", "1", "2"])

# Re-classing an ROI in its table row moves it between groups.
win.table.item(1, class_column)._text = "vegetation"
win.on_item_changed(win.table.item(1, class_column))
check("a re-typed class sticks to the ROI", win.rois[1]["class"], "vegetation")
check("and the groups follow it",
      [label for label, _ in R.group_by_class(win.rois)], ["vegetation"])
check("one class, so no combined row", win.class_table.rowCount(), 1)
check("and it now holds both", win.class_table.text(0, 1), "2")

# An emptied class cell is not an empty class.
win.table.item(1, class_column)._text = "   "
win.on_item_changed(win.table.item(1, class_column))
check("blanking a class does not make one with no name",
      win.rois[1]["class"], R.ROI_CLASS_UNSET)

# Both exports carry it.
R.QgsFields = list
R.QgsField = lambda name, *a: name
fields, plan = win.export_fields()
check("class is a shapefile column", "class" in fields, True)
check("and carries the ROI's own",
      win.feature_attributes(win.rois[0], plan)[fields.index("class")],
      "vegetation")
with tempfile.TemporaryDirectory() as tmp2:
    path = os.path.join(tmp2, "by_class.csv")
    ok("the by-class summary writes",
       win._write_rows(path, R.class_summary_rows(
           win.rois, [label for _, label in win.measure_bands])))
    import csv as _csv2
    with open(path, encoding="utf-8-sig") as handle:
        crows = list(_csv2.reader(handle))
    # Two classes by now -- vegetation, and the one just blanked -- so three
    # rows per band: each class, then all.
    check("a header and a row per class, band and convention",
          len(crows), 1 + 3 * 2)
    check("under names that say what they are", crows[0],
          ["class", "band", "backscat", "rois", "mean_db", "spread_db",
           "enl"])

# The noise floor rides out beside them, from the window's own measured ROIs.
print("\n── the noise floor, end to end ──")
bands = [label for _, label in win.measure_bands]
water = [roi for roi in win.rois if R.class_key(roi.get("class")) == "water"]
check("the fixture has no water left after the re-classing", water, [])
estimate = R.nesz_estimate(win.rois, bands[0])
check("so there is no floor to report", estimate["rois"], 0)

# Classing one water is not enough on this fixture, and that is the point: its
# raster carries no RTC factor, so nothing here was measured in sigma0 and
# there is no sigma0 to take a floor from. A gamma0 number under the name NESZ
# would be wrong by the RTC factor, so none is offered.
win.rois[1]["class"] = "water"
estimate = R.nesz_estimate(win.rois, bands[0])
check("a water ROI on a gamma0-only product still yields nothing",
      estimate["rois"], 0)
ok("and no number to quote", not np.isfinite(estimate["nesz_db"]), "NaN")
check("the ROI really is water, so it is the sigma0 that is missing",
      R.roi_stats(win.rois[1], bands[0], R.BACKSCATTER_SIGMA0), None)

# Give the same ROI a sigma0 and the bound appears, with the margin beside it.
win.rois[1]["stats"][R.BACKSCATTER_SIGMA0] = {
    bands[0]: {"n": 36, "mean": 1.0e-3, "mean_db": R.to_db(1.0e-3),
               "nonpos": 0}}
estimate = R.nesz_estimate(win.rois, bands[0])
check("with a sigma0 to read, the water ROI is the evidence",
      estimate["rois"], 1)
ok("and the bound is that ROI's own mean",
   abs(estimate["nesz_db"] - R.to_db(1.0e-3)) < 1e-9,
   f"{estimate['nesz_db']:.2f} dB")
win.rois[0]["stats"][R.BACKSCATTER_SIGMA0] = {
    bands[0]: {"n": 36, "mean": 0.1, "mean_db": R.to_db(0.1), "nonpos": 0}}
margin = R.nesz_margin_db(win.rois[0], bands[0], estimate["nesz_db"])
ok("and the vegetation ROI stands 20 dB above it",
   abs(margin - 20.0) < 1e-9, f"{margin:.2f} dB")
with tempfile.TemporaryDirectory() as tmp3:
    path = os.path.join(tmp3, "nesz.csv")
    ok("the noise-floor CSV writes",
       win._write_rows(path, R.nesz_rows(win.rois, bands)))
    import csv as _csv3
    with open(path, encoding="utf-8-sig") as handle:
        nrows = list(_csv3.reader(handle))
    check("a header and a row per band", len(nrows), 1 + len(bands))
    check("under names that say what they are", nrows[0],
          ["band", "class", "rois", "n", "nonpos", "nesz_db", "floor_db",
           "note"])
    ok("and every row says it is a bound, not a measurement",
       all("upper bound" in row[-1] for row in nrows[1:]), nrows[1][-1])
win.rois[1]["class"] = "vegetation"
for roi in win.rois[:2]:
    roi["stats"].pop(R.BACKSCATTER_SIGMA0, None)

# ── 11. a NISAR GCOV '.h5' becomes a georeferenced VRT ───────────────────────
print("\n── GCOV HDF5 ──")


class _Dataset:
    def __init__(self, array):
        self._a = np.asarray(array)

    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    def __getitem__(self, _key):
        return self._a


class _H5File:
    def __init__(self, contents):
        self._c = contents

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def visit(self, fn):
        for name in self._c:
            fn(name)

    def __getitem__(self, name):
        if name in self._c:
            return self._c[name]
        # h5py hands back a Group for a path that is not a dataset, and
        # iterating one yields its children's names -- which is how the RTC
        # factor is looked for beside the covariance terms.
        prefix = name.rstrip("/") + "/"
        children = [key[len(prefix):] for key in self._c
                    if key.startswith(prefix) and "/" not in key[len(prefix):]]
        if not children:
            raise KeyError(name)
        return children


def install_h5py(contents):
    module = types.ModuleType("h5py")
    module.File = lambda path, mode="r": _H5File(contents)
    sys.modules["h5py"] = module


GRID = "science/LSAR/GCOV/grids/frequencyA"
# Coordinate vectors give pixel centres, 30 m apart, north-up.
install_h5py({
    f"{GRID}/HHHH": _Dataset(np.zeros((120, 100), dtype=np.float32)),
    f"{GRID}/HVHV": _Dataset(np.zeros((120, 100), dtype=np.float32)),
    f"{GRID}/HHHV": _Dataset(np.zeros((120, 100), dtype=np.complex64)),
    f"{GRID}/rtcGammaToSigmaFactor": _Dataset(
        np.ones((120, 100), dtype=np.float32)),
    f"{GRID}/xCoordinates": _Dataset(500015.0 + 30.0 * np.arange(100)),
    f"{GRID}/yCoordinates": _Dataset(4000985.0 - 30.0 * np.arange(120)),
    f"{GRID}/projection": _Dataset(np.array(32644)),
})

with tempfile.TemporaryDirectory() as tmp:
    h5_path = os.path.join(tmp, "NISAR_L2_GCOV.h5")
    open(h5_path, "wb").close()
    vrt_path = win.gcov_vrt_for(h5_path)
    ok("a VRT is written beside the product",
       vrt_path == os.path.join(tmp, "NISAR_L2_GCOV_gcov.vrt"), str(vrt_path))
    import xml.etree.ElementTree as ET
    root = ET.parse(vrt_path).getroot()
    check("sized from the covariance term",
          (root.get("rasterXSize"), root.get("rasterYSize")), ("100", "120"))
    check("the grid's own projection", root.findtext("SRS"), "EPSG:32644")
    check("georeferenced from the coordinate vectors, half a pixel back",
          [float(v) for v in root.findtext("GeoTransform").split(",")],
          [500000.0, 30.0, 0.0, 4001000.0, 0.0, -30.0])
    check("the diagonal terms become bands -- not the complex one, and the "
          "RTC factor rides along",
          [b.findtext("Description") for b in root.findall("VRTRasterBand")],
          ["HHHH", "HVHV", "rtcGammaToSigmaFactor"])
    check("so sigma0 is available from the '.h5' alone",
          R.factor_band_index(
              [b.findtext("Description")
               for b in root.findall("VRTRasterBand")]), 3)
    check("each band reads its own subdataset in place",
          root.find("VRTRasterBand/SimpleSource/SourceFilename").text,
          f'HDF5:"{h5_path}"://{GRID}/HHHH')

    # A GSLC has no GCOV grids: it is refused, and says where to convert it.
    del warned[:]
    install_h5py({"science/LSAR/GSLC/grids/frequencyA/HH":
                  _Dataset(np.zeros((4, 4), dtype=np.complex64))})
    check("a GSLC is refused", win.gcov_vrt_for(h5_path), None)
    ok("and is pointed at the converter",
       any("DPQED_h52tif" in str(w) for w in warned), str(warned))

    # Without h5py there is no way to read the grid's corner at all.
    del warned[:]
    saved = sys.modules.pop("h5py")
    sys.modules["h5py"] = None          # an import that raises, not one that works
    check("no h5py, no guess at the georeferencing",
          win.gcov_vrt_for(h5_path), None)
    ok("and it says so", any("h5py" in str(w) for w in warned), str(warned))
    sys.modules["h5py"] = saved


# ── A REAL SHAPEFILE, NOT A STUB ──────────────────────────────────────────────
# The blank-shapefile bug was invisible to every stub: the writer was asked for
# features, said nothing, and produced a file with no rows. So this section
# takes the records the tool would hand a writer and puts them through a real
# OGR round trip -- field names, types, values and rings -- and reads back what
# landed. If OGR will not take these records, it fails here rather than in a
# .shp the user opens a week later.
try:
    import numpy as _np
    from pyogrio.raw import write as _ogr_write, read as _ogr_read
    from shapely import wkt as _wkt
except ImportError:                     # the suite still runs without them
    print("\n-- real shapefile round trip: skipped (no pyogrio/shapely)")
else:
    print("\n-- real shapefile round trip")
    shp_fields, shp_plan = win.export_fields()
    records = R.shapefile_records(win.rois, shp_plan)
    schema = R.shapefile_schema(shp_plan)
    check("a record per ROI that has a ring", len(records), len(win.rois))
    check("the schema and the field list agree",
          [name for name, _ in schema], list(shp_fields))

    fillers = {"int": 0, "double": float("nan"), "string": ""}
    dtypes = {"int": "int64", "double": "float64", "string": "object"}
    columns = []
    for index, (_, kind) in enumerate(schema):
        values = [record[1][index] for record in records]
        values = [fillers[kind] if value is None else value
                  for value in values]
        columns.append(_np.array(values, dtype=dtypes[kind]))
    geometries = _np.array(
        [_wkt.loads(record[0]).wkb for record in records], dtype=object)

    with tempfile.TemporaryDirectory() as tmp:
        shp_path = os.path.join(tmp, "rois.shp")
        _ogr_write(shp_path, geometries, columns,
                   _np.array([name for name, _ in schema], dtype=object),
                   driver="ESRI Shapefile", geometry_type="Polygon",
                   crs="EPSG:32645")
        meta, _, back_geoms, back_fields = _ogr_read(shp_path)

    check("every ROI came back", len(back_geoms), len(records))
    ok("and every one carries a ring",
       all(geometry is not None and len(geometry) for geometry in back_geoms),
       str([geometry is None for geometry in back_geoms]))
    areas = [_wkt.loads(record[0]).area for record in records]
    ok("with an area, so the ring is a polygon and not a line",
       all(area > 0 for area in areas), str(areas))
    check("a column per field survived the DBF",
          len(meta["fields"]), len(schema))
    names_back = [str(name) for name in meta["fields"]]
    check("under the names the schema declared, truncated as DBF truncates",
          names_back, [name[:10] for name, _ in schema])
    check("the ROI serial numbers came back",
          [int(value) for value in back_fields[0]],
          [roi["roi"] for roi in win.rois])
    class_at = [n for n, _ in R.ROI_FIELDS].index("class")
    check("and the classes, as text",
          [str(value) for value in back_fields[class_at]],
          [roi.get("class") for roi in win.rois])
    label = win.measure_bands[0][1]
    mean_at = [index for index, (_, _, band, key) in enumerate(shp_plan)
               if band == label and key == "mean_db"][0]
    wanted = (R.roi_stats(win.rois[0], label) or {}).get("mean_db")
    check("and a statistic, as a number in its own column",
          round(float(back_fields[mean_at][0]), 6), round(float(wanted), 6))


# ── A FILE SET PER CLASS ──────────────────────────────────────────────────────
# Exporting writes the whole set, and beside it one shapefile and one
# statistics CSV per class. What is tested here is the grouping, the file
# naming and which ROIs land in which file; the writer itself is covered by the
# round trip above, and is stood in for by pyogrio so that this runs without
# QGIS. That substitution is the point of _write_shapefile taking its ROIs as
# an argument rather than reaching for self.rois.
try:
    import numpy as _np2
    from pyogrio.raw import write as _ogr_write2, read as _ogr_read2
    from shapely import wkt as _wkt2
except ImportError:
    print("\n-- a file set per class: skipped (no pyogrio/shapely)")
else:
    print("\n-- a file set per class")

    def _write_via_ogr(path, fields, plan, rois=None):
        rois = win.rois if rois is None else rois
        records = R.shapefile_records(rois, plan)
        if not records:
            return 0, "none"
        schema = R.shapefile_schema(plan)
        fillers = {"int": 0, "double": float("nan"), "string": ""}
        dtypes = {"int": "int64", "double": "float64", "string": "object"}
        columns = []
        for index, (_, kind) in enumerate(schema):
            values = [fillers[kind] if record[1][index] is None
                      else record[1][index] for record in records]
            columns.append(_np2.array(values, dtype=dtypes[kind]))
        _ogr_write2(path,
                    _np2.array([_wkt2.loads(r[0]).wkb for r in records],
                               dtype=object),
                    columns,
                    _np2.array([name for name, _ in schema], dtype=object),
                    driver="ESRI Shapefile", geometry_type="Polygon",
                    crs="EPSG:32645")
        return len(records), "OGR"

    win.rois[0]["class"] = "vegetation"
    win.rois[1]["class"] = "water"
    fields, plan = win.export_fields()
    real_writer = win._write_shapefile
    win._write_shapefile = _write_via_ogr
    try:
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "rois")
            written = win._export_by_class(stem, fields, plan)
            check("a file set per class",
                  [(label, count) for label, _, count in written],
                  [("vegetation", 1), ("water", 1)])
            for label, slug, _ in written:
                ok(f"{label}: a shapefile that is not empty",
                   os.path.getsize(f"{stem}_{slug}.shp") > 100,
                   f"{os.path.getsize(f'{stem}_{slug}.shp')} bytes")
                ok(f"{label}: the statistics CSV beside it",
                   os.path.exists(f"{stem}_{slug}_stats.csv"))

            # The class file holds that class, and says so when read back.
            _, _, geoms, back = _ogr_read2(f"{stem}_water.shp")
            class_at = [n for n, _ in R.ROI_FIELDS].index("class")
            check("the water shapefile holds the water ROI alone",
                  [str(v) for v in back[class_at]], ["water"])
            check("with its geometry", len(geoms), 1)

            import csv as _csv4
            with open(f"{stem}_water_stats.csv",
                      encoding="utf-8-sig") as handle:
                rows = list(_csv4.reader(handle))
            check("and its CSV is a header and a row per band",
                  len(rows), 1 + len(win.measure_bands))
            check("holding that ROI and no other",
                  {row[0] for row in rows[1:]}, {str(win.rois[1]["roi"])})
            ok("no per-class summary CSV: the whole-set one has the row",
               not os.path.exists(f"{stem}_water_by_class.csv"))

            # Written from every ROI, not from what the filter is showing.
            win.class_combo.setCurrentText("vegetation")
            win.on_class_filter()
            check("the filter is showing one class", len(win.shown_rois()), 1)
            written = win._export_by_class(os.path.join(tmp, "filtered"),
                                           fields, plan)
            check("and the export still writes both",
                  [label for label, _, _ in written], ["vegetation", "water"])
            win.class_combo.setCurrentText(R.ROI_CLASS_ALL)
            win.on_class_filter()

            # One class is not a by-class export; it is the whole set under a
            # longer name, and writing it makes a duplicate someone diffs.
            win.rois[1]["class"] = "vegetation"
            check("a single-class set gets no per-class files",
                  win._export_by_class(os.path.join(tmp, "single"),
                                       fields, plan), [])
            ok("and none were written",
               not any(name.startswith("single") for name in os.listdir(tmp)),
               str(sorted(n for n in os.listdir(tmp)
                          if n.startswith("single"))))
    finally:
        win._write_shapefile = real_writer

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
