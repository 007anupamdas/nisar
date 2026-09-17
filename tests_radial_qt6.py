"""Drive DPQED_radial_qt6 with a REAL PyQt6, and only QGIS stubbed.

The Qt5 suite mocks PyQt5 away, which is right for testing wiring and wrong for
testing a port: a mock answers to any spelling, so `Qt.CrossCursor` and
`Qt.CursorShape.CrossCursor` would both pass and the one that matters would not
be told from the one that crashes on a Qt6 build.

So this imports PyQt6 for real and runs Qt offscreen. Every widget is built,
every signal connected, every enum resolved by Qt itself -- a wrong scoped name
is an AttributeError here rather than on the user's machine. Only QGIS is
stubbed, since it cannot be pip-installed, and its canvas is a real QWidget so
the overlay really parents into it.

    QT_QPA_PLATFORM=offscreen python3 tests_radial_qt6.py

The QGIS stub deliberately offers the QGIS 4 spellings ONLY -- there is no
QgsWkbTypes in it at all -- so the import resolves the modern names or fails.
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from unittest.mock import MagicMock

from PyQt6.QtCore import Qt, QMetaType, QPoint, PYQT_VERSION_STR, QT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QWidget, QAbstractItemView

HERE = os.path.dirname(os.path.abspath(__file__))
app = QApplication.instance() or QApplication(sys.argv[:1])

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


# ── stand-ins for the QGIS types the module actually uses ────────────────────
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


class _CRS:
    def __init__(self, authid="EPSG:32644"):
        self._a = authid

    def isValid(self):
        return True

    def isGeographic(self):
        return False

    def authid(self):
        return self._a

    def description(self):
        return self._a


class _Canvas(QWidget):
    """QgsMapCanvas as a REAL QWidget, so the overlay parents into something.

    A MagicMock canvas would make `QWidget(self.canvas)` a TypeError, and the
    translucent Normalize bar is the one piece of this window that is a child
    of the map rather than of the layout.
    """
    def __init__(self, *a, **k):
        super().__init__()
        self.resize(900, 600)
        self._extent = _Rect(0, 0, 1, 1)
        self.crs_set, self.tools_set, self.layers_set = [], [], []
        self.refreshed = 0

    def enableAntiAliasing(self, _):
        pass

    def setCachingEnabled(self, _):
        pass

    def setParallelRenderingEnabled(self, _):
        pass

    def setDestinationCrs(self, crs):
        self.crs_set.append(crs)

    def setMapTool(self, tool):
        self.tools_set.append(tool)

    def setLayers(self, layers):
        self.layers_set.append(layers)

    def setExtent(self, extent):
        self._extent = extent

    def extent(self):
        return self._extent

    def refresh(self):
        self.refreshed += 1

    def mapUnitsPerPixel(self):
        return 10.0


class _MapTool:
    """QgsMapTool: a real class, because RoiMapTool subclasses it."""
    def __init__(self, canvas=None):
        self._canvas = canvas

    def setCursor(self, cursor):
        self._cursor = cursor

    def toMapCoordinates(self, pos):
        return _PointXY(float(pos.x()), float(pos.y()))

    def deactivate(self):
        pass


def _tool_factory(kind):
    def make(*a, **k):
        m = MagicMock()
        m._tool, m._args = kind, a
        return m
    return make


class _Qgis:
    """Only the QGIS 4 spellings -- the resolver gets no Qt5-era fallback."""
    class GeometryType:
        Polygon = "Qgis.GeometryType.Polygon"

    class WkbType:
        Polygon = "Qgis.WkbType.Polygon"

    class RasterBandStatistic:
        Min = 1
        Max = 2


qgis = types.ModuleType("qgis")
qgis_gui = types.ModuleType("qgis.gui")
qgis_core = types.ModuleType("qgis.core")
qgis.gui, qgis.core = qgis_gui, qgis_core
sys.modules.update({"qgis": qgis, "qgis.gui": qgis_gui, "qgis.core": qgis_core})

qgis_gui.QgsMapCanvas = _Canvas
qgis_gui.QgsMapTool = _MapTool
qgis_gui.QgsMapToolPan = _tool_factory("pan")
qgis_gui.QgsMapToolZoom = _tool_factory("zoom")
qgis_gui.QgsRubberBand = lambda *a, **k: MagicMock(_args=a)

for name in ("QgsProject", "QgsRasterLayer", "QgsVectorLayer", "QgsGeometry",
             "QgsCoordinateReferenceSystem", "QgsCoordinateTransform",
             "QgsSingleBandGrayRenderer", "QgsMultiBandColorRenderer",
             "QgsContrastEnhancement", "QgsFields", "QgsField", "QgsFeature",
             "QgsVectorFileWriter"):
    setattr(qgis_core, name, MagicMock())
qgis_core.Qgis = _Qgis
qgis_core.QgsPointXY = _PointXY
qgis_core.QgsRectangle = _Rect
# No QgsWkbTypes and no QgsRasterBandStats: this is a QGIS 4 build.

sys.path.insert(0, HERE)
import DPQED_radial_qt6 as R

print(f"imported OK under PyQt6 {PYQT_VERSION_STR} / Qt {QT_VERSION_STR}; "
      f"{type(R.win).__name__} built")
win = R.win

# ── 1. the window really exists ──────────────────────────────────────────────
print("\n── the window, built for real ──")
ok("it is a QMainWindow", win.isWidgetType())
ok("the canvas is the real widget the overlay parents into",
   win.overlay.parent() is win.canvas)
check("five tool buttons", len(win.tool_buttons), 5)
check("the table has a column per header",
   win.table.columnCount(),
   len(R.ROI_TABLE_COLUMNS) + len(R.TABLE_STATS))
check("headers are set",
      win.table.horizontalHeaderItem(0).text(), "ROI")
check("row selection, one at a time",
      (win.table.selectionBehavior(), win.table.selectionMode()),
      (QAbstractItemView.SelectionBehavior.SelectRows,
       QAbstractItemView.SelectionMode.SingleSelection))
check("the domain combo offers every domain", win.domain_combo.count(),
      len(R.DOMAIN_CHOICES))
check("and starts on the GCOV one", win.domain(), R.DOMAIN_POWER)
check("zeros are fill by default", win.zero_is_nodata(), True)
check("the clip combo starts at the default", win.clip_percent(),
      R.NORM_CLIP_DEFAULT)

# ── 2. the QGIS 4 names resolved ─────────────────────────────────────────────
print("\n── QGIS 4 names ──")
check("polygon geometry type", R.GEOMETRY_POLYGON, "Qgis.GeometryType.Polygon")
check("polygon WKB type", R.WKB_POLYGON, "Qgis.WkbType.Polygon")
check("raster min|max flags", R.RASTER_STATS_MINMAX, 3)
check("field types come from QMetaType", R.FIELD_TYPES,
      {"int": QMetaType.Type.Int, "double": QMetaType.Type.Double,
       "string": QMetaType.Type.QString})
ok("QgsWkbTypes really was absent", R.QgsWkbTypes is None)
# The other direction: a QGIS 3 build, where only the old spellings exist.
check("it falls back to the QGIS 3 spelling",
      R._resolve("x", lambda: None.missing, lambda: "old-spelling"),
      "old-spelling")
try:
    R._resolve("something nothing has", lambda: None.missing)
    check("a build with neither fails loudly", "silent", "ImportError")
except ImportError as e:
    ok("a build with neither fails loudly, naming the symbol",
       "something nothing has" in str(e), str(e))

# ── 3. real Qt enums, resolved by Qt ─────────────────────────────────────────
print("\n── Qt6 enums, as Qt itself resolves them ──")
win.canvas.setCursor(Qt.CursorShape.CrossCursor)
check("the cursor the tool row sets", win.canvas.cursor().shape(),
      Qt.CursorShape.CrossCursor)
check("the summary labels are centred", win.lbl_mean.alignment(),
      Qt.AlignmentFlag.AlignCenter)
check("the detail panel is read-only", win.detail.isReadOnly(), True)


# ── 4. an ROI measured off a real raster, through a stubbed GDAL ─────────────
def install_gdal(bands, geotransform, nodata=None):
    reads = []

    def make_band(array):
        band = MagicMock()
        band.XSize, band.YSize = array.shape[1], array.shape[0]
        band.GetNoDataValue = MagicMock(return_value=nodata)

        def read(col0=0, row0=0, ncol=None, nrow=None, *a, **k):
            ncol = array.shape[1] if ncol is None else ncol
            nrow = array.shape[0] if nrow is None else nrow
            reads.append((col0, row0, ncol, nrow))
            return array[row0:row0 + nrow, col0:col0 + ncol]
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


print("\n── measuring, with the real table ──")
rng = np.random.default_rng(7)
GT = (500000.0, 30.0, 0.0, 4000000.0, 0.0, -30.0)
hh = rng.exponential(0.01, (200, 200))
hv = rng.exponential(0.002, (200, 200))
hh[4:10, 10:16] = rng.exponential(0.1, (6, 6))      # -10 dB patch
hv[4:10, 10:16] = rng.exponential(0.02, (6, 6))
reads = install_gdal([hh, hv], GT)

layer = MagicMock()
layer.isValid.return_value = True
layer.source.return_value = "/data/gcov.vrt"
layer.crs.return_value = _CRS()
win.raster_layer = layer
win.raster_path = "/data/gcov.vrt"
win.proj_crs = _CRS()
win.band_labels = ["HHHH", "HVHV"]
win.measure_bands = list(enumerate(win.band_labels, start=1))
win.factor_band = None
win.band_prefixes = [R.band_prefix(b, i)
                     for i, b in enumerate(win.band_labels, start=1)]
win.stats_band_combo.clear()
for label in win.band_labels:
    win.stats_band_combo.addItem(label)
win.stats_band_combo.setCurrentIndex(0)

roi = win.add_roi(R.rect_ring(500300.0, 3999700.0, 500480.0, 3999880.0), "rect")
ok("the ROI was accepted", roi is not None)
check("36 pixels inside", roi["npix"], 36)
check("read from the patch's own window", reads[-1], (10, 4, 6, 6))
ok("and reads the patch's gamma0",
   abs(roi["stats"]["HHHH"]["mean_db"] + 10.0) < 1.5,
   f"{roi['stats']['HHHH']['mean_db']:.2f} dB")

# The table is a real QTableWidget, so its signals really fire -- which is the
# only way to find out whether the guard against them is doing its job.
check("a real row appeared", win.table.rowCount(), 1)
check("with the ROI's own columns",
      [win.table.item(0, c).text() for c in range(3)], ["1", "ROI 1", "rect"])
name_column = R.ROI_TABLE_COLUMNS.index("name")
ok("only the name cell is editable",
   bool(win.table.item(0, name_column).flags() & Qt.ItemFlag.ItemIsEditable)
   and not bool(win.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable))
check("the footer reports the band",
      win.lbl_mean.text().startswith("Mean:"), True)

# itemChanged fires for real here: renaming must reach the ROI, and refilling
# the table must not be mistaken for a rename.
win.table.item(0, name_column).setText("calibration site")
check("a typed label reaches the ROI", win.rois[0]["name"], "calibration site")
win.refresh_table()
check("refilling the table does not rename anything",
      win.rois[0]["name"], "calibration site")

# ── 4b. sigma0, through a real QComboBox ─────────────────────────────────────
print("\n── sigma0 ──")
factor = np.full((200, 200), 2.0)               # +3.0103 dB everywhere
install_gdal([hh, hv, factor], GT)
sigma_layer = MagicMock()
sigma_layer.isValid.return_value = True
sigma_layer.source.return_value = "/data/gcov.vrt"
sigma_layer.crs.return_value = _CRS()
sigma_layer.bandName.side_effect = lambda b: [
    "HHHH", "HVHV", "rtcGammaToSigmaFactor"][b - 1]
provider = MagicMock()
provider.bandCount.return_value = 3
provider.cumulativeCut = MagicMock(return_value=(0.0, 1.0))
provider.bandStatistics = MagicMock(
    return_value=MagicMock(minimumValue=0.0, maximumValue=1.0))
sigma_layer.dataProvider.return_value = provider
win.raster_layer = sigma_layer
win.populate_band_picker(sigma_layer)

check("the factor band is found", win.factor_band, 3)
check("and left out of the measured bands",
      [label for _, label in win.measure_bands], ["1: HHHH", "2: HVHV"])
ok("the real combo became selectable", win.backscatter_combo.isEnabled())
check("it offers both conventions", win.backscatter_combo.count(),
      len(R.BACKSCATTER_CHOICES))
check("starting on gamma0", win.backscatter(), R.BACKSCATTER_GAMMA0)

win.rois = []
win._next_roi_id = 1
ring = R.rect_ring(500300.0, 3999700.0, 500480.0, 3999880.0)
roi = win.add_roi(ring, "rect")
check("recorded as gamma0", roi["backscat"], R.BACKSCATTER_GAMMA0)
before = {band: dict(stats) for band, stats in roi["stats"].items()}

win.backscatter_combo.setCurrentIndex(1)        # the real signal fires here
check("selecting sigma0 re-measures on its own", win.rois[0]["backscat"],
      R.BACKSCATTER_SIGMA0)
for band in ("1: HHHH", "2: HVHV"):
    ok(f"{band} moved by exactly the factor",
       abs(win.rois[0]["stats"][band]["mean_db"]
           - before[band]["mean_db"] - 3.0103) < 1e-3,
       f"{before[band]['mean_db']:.4f} -> "
       f"{win.rois[0]['stats'][band]['mean_db']:.4f} dB")
check("the detail panel names the convention",
      R.BACKSCATTER_SIGMA0 in win.detail.toPlainText(), True)
win.backscatter_combo.setCurrentIndex(0)

# ── 5. the drawing tools, with real Qt mouse buttons ─────────────────────────
print("\n── drawing, with real Qt buttons ──")
drawn = []
real_add_roi = win.add_roi
win.add_roi = lambda ring, kind, **k: drawn.append((kind, ring)) or {"roi": 0}


class _Event:
    def __init__(self, x, y, button=None):
        self._p, self._b = QPoint(int(x), int(y)), button

    def pos(self):
        return self._p

    def button(self):
        return self._b


rect_tool = R.RoiMapTool(win.canvas, win, R.TOOL_RECT)
rect_tool.canvasPressEvent(_Event(100, 400, Qt.MouseButton.LeftButton))
rect_tool.canvasMoveEvent(_Event(150, 350))
rect_tool.canvasReleaseEvent(_Event(160, 340, Qt.MouseButton.LeftButton))
check("a drag makes a rectangle", drawn[-1][0], "rect")
check("spanned by its corners", drawn[-1][1],
      R.rect_ring(100.0, 400.0, 160.0, 340.0))

del drawn[:]
poly = R.RoiMapTool(win.canvas, win, R.TOOL_POLY)
for x, y in [(0, 0), (300, 0), (300, 300), (0, 300)]:
    poly.canvasPressEvent(_Event(x, y, Qt.MouseButton.LeftButton))
poly.canvasPressEvent(_Event(4, 302, Qt.MouseButton.LeftButton))
poly.canvasDoubleClickEvent(_Event(4, 302, Qt.MouseButton.LeftButton))
check("clicked corners make a polygon", drawn[-1][0], "polygon")
check("the double-click's repeated corner is merged away", drawn[-1][1],
      [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)])

del drawn[:]
for x, y in [(0, 0), (300, 0), (300, 300)]:
    poly.canvasPressEvent(_Event(x, y, Qt.MouseButton.LeftButton))
poly.canvasPressEvent(_Event(0, 0, Qt.MouseButton.RightButton))
check("a right-click closes it", len(drawn), 1)

del drawn[:]
for x, y in [(0, 0), (300, 0), (300, 300)]:
    poly.canvasPressEvent(_Event(x, y, Qt.MouseButton.LeftButton))
poly.cancel()
poly.canvasPressEvent(_Event(0, 0, Qt.MouseButton.RightButton))
check("cancelling abandons it", drawn, [])
win.add_roi = real_add_roi

# _event_pos: both Qt6 spellings, since a build may have dropped the old one.
print("\n── mouse position, either spelling ──")
check("pos()", R._event_pos(_Event(7, 9)), QPoint(7, 9))


class _PositionOnly:
    def position(self):
        from PyQt6.QtCore import QPointF
        return QPointF(7.4, 9.6)


check("position() when pos() is gone", R._event_pos(_PositionOnly()),
      QPoint(7, 10))

# ── 6. the map tools ─────────────────────────────────────────────────────────
print("\n── map tools ──")
for mode in R.MAP_TOOLS:
    win.set_map_tool(mode)
    ok(f"{mode} applied", win.canvas.tools_set[-1] is win.map_tools[mode])
    checked = [m for m, b in win.tool_buttons.items() if b.isChecked()]
    ok(f"{mode} is the only one checked", checked == [mode], str(checked))
check("five distinct tools",
      len({id(win.map_tools[m]) for m in R.MAP_TOOLS}), 5)
check("zoom out is the out variant",
      (win.map_tools[R.TOOL_ZOOM_IN]._args[1],
       win.map_tools[R.TOOL_ZOOM_OUT]._args[1]), (False, True))
win.set_map_tool(R.TOOL_PAN)
check("pan gets the open hand", win.canvas.cursor().shape(),
      Qt.CursorShape.OpenHandCursor)
win.set_map_tool(R.TOOL_RECT)
check("drawing gets the cross", win.canvas.cursor().shape(),
      Qt.CursorShape.CrossCursor)

# ── 7. export ────────────────────────────────────────────────────────────────
print("\n── export ──")
R.QgsFields = list
R.QgsField = lambda name, *a: name
win.rois[0]["name"] = "calibration site"    # a typed label has to be exported
fields, plan = win.export_fields()
check("a column per ROI field, band and statistic",
      len(fields), len(R.ROI_FIELDS) + 2 * len(R.STAT_KEYS))
check("fields and plan agree", [e[0] for e in plan], list(fields))
ok("nothing over DBF's cap", max(len(n) for n in fields) <= 10)
ok("the polarizations name their columns", "HH_mean_db" in fields)
attributes = win.feature_attributes(win.rois[0], plan)
check("an attribute per field", len(attributes), len(fields))
check("the ROI's own fields lead", attributes[:3],
      [1, "calibration site", "rect"])

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "stats.csv")
    ok("the statistics CSV writes", win._write_csv(path))
    import csv as _csv
    with open(path, encoding="utf-8-sig") as handle:
        rows = list(_csv.reader(handle))
    check("a row per ROI and band, plus a header", len(rows), 1 + 1 * 2)
    check("under untruncated names", rows[0][-len(R.STAT_KEYS):],
          list(R.STAT_KEYS))

# ── 8. the shortcuts really registered ───────────────────────────────────────
print("\n── shortcuts ──")
from PyQt6.QtGui import QShortcut
shortcuts = {s.key().toString() for s in win.findChildren(QShortcut)}
for expected in ("Ctrl+L", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S", "Ctrl+R",
                 "F5", "Ctrl+1", "Ctrl+5"):
    ok(f"{expected} is bound", expected in shortcuts,
       f"have {sorted(shortcuts)}")

# ── 9. the OTHER build: QGIS 3 spellings only ────────────────────────────────
# The fallback candidates are the half of the resolver the QGIS 4 stub above
# never reaches, so they are the half that can rot unnoticed -- and did: a
# blanket rename once turned every one of them into a reference to the
# constant it was being used to define. Re-import the module against a build
# that has only the old spellings, which is the only thing that runs them.
print("\n── resolving against a QGIS 3 build ──")
import importlib


class _OldQgis:
    """No GeometryType, no WkbType, no RasterBandStatistic."""


class _OldWkbTypes:
    PolygonGeometry = "QgsWkbTypes.PolygonGeometry"
    Polygon = "QgsWkbTypes.Polygon"


class _OldBandStats:
    Min = 4
    Max = 8


qgis_core.Qgis = _OldQgis
qgis_core.QgsWkbTypes = _OldWkbTypes
qgis_core.QgsRasterBandStats = _OldBandStats
try:
    OLD = importlib.reload(R)
    check("falls back to the old geometry type", OLD.GEOMETRY_POLYGON,
          "QgsWkbTypes.PolygonGeometry")
    check("and the old WKB type", OLD.WKB_POLYGON, "QgsWkbTypes.Polygon")
    check("and the old statistic flags", OLD.RASTER_STATS_MINMAX, 12)
    ok("a fallback never resolves to the name it is defining",
       OLD.GEOMETRY_POLYGON != OLD.WKB_POLYGON)
finally:
    qgis_core.Qgis = _Qgis
    for name in ("QgsWkbTypes", "QgsRasterBandStats"):
        if hasattr(qgis_core, name):
            delattr(qgis_core, name)

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
