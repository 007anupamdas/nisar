"""Drive DPQED_rival's Qt-side class with PyQt5/QGIS stubbed out.

Covers what tests_rival_meta.py cannot: the folder scan, the CRS wiring and the
canvas behaviour, none of which live in the pure-helper slice. Geometry and CRS
calls are stubs, so this proves the code paths execute and route correctly -- it
says nothing about whether QGIS renders anything.

    python3 tests_rival_gui_stub.py        # exits non-zero on any assertion

Two stubbing traps worth knowing, both of which hid real differences until
fixed: a MagicMock caches its return_value, so QgsMapCanvas() handed back ONE
object for both canvases, and QgsPointXY(...) handed back one point for every
coordinate. Both now get real stand-ins.
"""
import os, sys, shutil, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
from unittest.mock import MagicMock

for name in ["PyQt5", "PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.QtGui",
             "qgis", "qgis.gui", "qgis.core"]:
    sys.modules[name] = MagicMock()

# Qt base classes must be real classes for subclassing to work
qw = sys.modules["PyQt5.QtWidgets"]
qg = sys.modules["qgis.gui"]
qc = sys.modules["PyQt5.QtCore"]
class _Base:
    """Stand-in for the Qt classes the module subclasses.

    Attributes are memoised: returning a fresh MagicMock per access is the
    mirror of the cached-return_value trap -- every call would land on a
    different mock, so `widget.hide.called` was always False.
    """
    def __init__(self, *a, **k): pass

    def __getattr__(self, n):
        m = MagicMock()
        object.__setattr__(self, n, m)
        return m
for mod, names in ((qw, ["QMainWindow", "QWidget"]),
                   (qg, ["QgsMapTool"]),
                   (qc, ["QObject"])):
    for n in names:
        setattr(mod, n, type(n, (_Base,), {}))

# A MagicMock caches its return_value, so QgsMapCanvas() would hand back ONE
# object for both canvases and hide any per-canvas difference. Give each call a
# fresh mock.
qg.QgsMapCanvas = MagicMock(side_effect=lambda *a, **k: MagicMock())

# Same trap for the map tools: QgsMapToolZoom(canvas, False) and (canvas, True)
# differ only by that flag, so one cached return_value would make zoom-in and
# zoom-out indistinguishable. Record the arguments on each instance instead.
def _tool_factory(name):
    def make(*a, **k):
        m = MagicMock()
        m._tool, m._args = name, a
        return m
    return MagicMock(side_effect=make)

qg.QgsMapToolPan  = _tool_factory("pan")
qg.QgsMapToolZoom = _tool_factory("zoom")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import DPQED_rival as R
print("imported OK; QCDashboard built:", type(R.win).__name__)


class _PointXY:
    """QgsPointXY stand-in. The mock caches its return_value, so every
    QgsPointXY(...) would otherwise be the same object and identity tests
    between distinct points would silently pass."""
    def __init__(self, x, y=None):
        if y is None:
            x, y = x.x(), x.y()
        self._x, self._y = float(x), float(y)
    def x(self): return self._x
    def y(self): return self._y
    def __repr__(self): return f"({self._x:.3f}, {self._y:.3f})"

R.QgsPointXY = _PointXY


class _Rect:
    """QgsRectangle stand-in, for the same reason as _PointXY."""
    def __init__(self, *b): self._b = tuple(float(v) for v in b)
    def xMinimum(self): return self._b[0]
    def yMinimum(self): return self._b[1]
    def xMaximum(self): return self._b[2]
    def yMaximum(self): return self._b[3]
    def __repr__(self):
        return (f"[{self._b[0]:.1f},{self._b[1]:.1f} .. "
                f"{self._b[2]:.1f},{self._b[3]:.1f}]")

R.QgsRectangle = _Rect


qw.QFileDialog.getExistingDirectory = MagicMock()
warned = []
qw.QMessageBox.warning  = lambda *a, **k: warned.append(a[-1])
qw.QMessageBox.critical = lambda *a, **k: warned.append("CRITICAL: " + a[-1])
win = R.win
win.input_tif_layer = None


def scan(folder):
    """Run the real select_reference_folder over a prepared folder."""
    del warned[:]
    qw.QFileDialog.getExistingDirectory = MagicMock(return_value=folder)
    win.dropdown_band.currentText = MagicMock(return_value="All bands")
    win.ref_footprints = {}
    win.select_reference_folder()
    return win.ref_mode, dict(win.ref_footprints), list(warned)


def report(title, mode, prints, warns):
    print(f"\n{title}: mode={mode}, {len(prints)} footprint(s)")
    for path, rec in sorted(prints.items()):
        print(f"  {rec['band']:4s} {rec['source']:24s} {len(rec['ring']):3d} vtx"
              f"  -> {os.path.basename(path)}  {win._label_for(path)}")
    if warns:
        print("  warnings:", warns)


# ── 0. the input scene's own footprint comes from its sidecar ────────────────
d0 = tempfile.mkdtemp()
met0 = ("NISAR_S2_PR_GSLC_028_084_A_010_3700_DHNA_A_"
        "20260819T001733_20260819T001810_P00500_M_F_I_001.met")
shutil.copy(os.path.join(HERE, met0), d0)
input_tif = os.path.join(d0, met0[:-len(".met")] + ".tif")
open(input_tif, "w").close()
ring = win._input_footprint_ring(input_tif)
assert ring == [(76.534748, 17.614687), (78.827076, 18.171194),
                (79.385351, 15.993327), (77.112174, 15.446038)], ring
# the swath, not the product grid: strictly inside the raster's own bbox
assert R.rings_bounds([ring]) == (76.534748, 15.446038, 79.385351, 18.171194)
assert win._input_footprint_ring(os.path.join(d0, "no_sidecar.tif")) is None
print("input footprint read from the scene's own .met")
shutil.rmtree(d0, ignore_errors=True)

# ── 1. NISAR sidecars, split across a Meta subfolder ─────────────────────────
d1 = tempfile.mkdtemp()
xml = ("NISAR_L2_PR_GSLC_028_084_A_010_4005_DHDH_A_"
       "20260819T001734_20260819T001808_P05023_N_F_J_001.h5.iso.xml")
met = ("NISAR_S2_PR_GSLC_028_084_A_010_3700_DHNA_A_"
       "20260819T001733_20260819T001810_P00500_M_F_I_001.met")
os.makedirs(os.path.join(d1, "Meta"))
for sidecar in (xml, met):
    shutil.copy(os.path.join(HERE, sidecar), os.path.join(d1, "Meta"))
open(os.path.join(d1, xml[:-len(".h5.iso.xml")] + ".tif"), "w").close()
open(os.path.join(d1, met[:-len(".met")] + ".tif"), "w").close()
mode, prints, warns = scan(d1)
report("NISAR sidecars in Meta/, rasters above", mode, prints, warns)
assert mode == "sidecar", mode
assert sorted(r["band"] for r in prints.values()) == ["LSAR", "SSAR"]
assert not warns, warns
assert all(win._label_for(p).startswith("[") for p in prints)

# band filter still narrows, and offers only the tags present
win._refresh_band_choices()
assert [win.dropdown_band.addItem.call_args_list[i][0][0]
        for i in range(len(win.dropdown_band.addItem.call_args_list))][-2:] \
    == ["LSAR", "SSAR"], "band combo should list both tags"

# ── 2. C1: the name is the footprint ─────────────────────────────────────────
d2 = tempfile.mkdtemp()
# the reported folder: unpadded degree counts and an '_ortho' suffix, with some
# tiles filed two levels down the way a real collection is organised
for name in ("N16E73.tif", "N16E74.tif"):
    open(os.path.join(d2, name), "w").close()
nested = os.path.join(d2, "Kerala", "2023")
os.makedirs(nested)
for name in ("N17E73.tif", "N8E76_ortho.tif", "N9E76_ortho.tif"):
    open(os.path.join(nested, name), "w").close()
mode, prints, warns = scan(d2)
report("C1 degree tiles", mode, prints, warns)
assert mode == "degree-tile", mode
assert len(prints) == 5, prints
n16e73 = [r for p, r in prints.items() if p.endswith("N16E73.tif")][0]
assert n16e73["ring"] == [(73.0, 17.0), (74.0, 17.0), (74.0, 16.0), (73.0, 16.0)]
# no NISAR band here: labels stay unprefixed and the filter collapses
assert all(r["band"] == "UNK" for r in prints.values())
n8 = [r for p, r in prints.items() if p.endswith("N8E76_ortho.tif")][0]
assert n8["ring"] == [(76.0, 9.0), (77.0, 9.0), (77.0, 8.0), (76.0, 8.0)], n8
assert all(not win._label_for(p).startswith("[") for p in prints)
assert not warns, warns

# ── 3. rasters with nothing to place them ────────────────────────────────────
d3 = tempfile.mkdtemp()
open(os.path.join(d3, "some_scene.tif"), "w").close()
open(os.path.join(d3, "readme.txt"), "w").close()
mode, prints, warns = scan(d3)
report("unplaceable rasters", mode, prints, warns)
assert mode is None and not prints, (mode, prints)
assert warns and "index.shp" in warns[0] and "N16E73.tif" in warns[0], warns

# ── 4. an empty folder is refused outright ───────────────────────────────────
d4 = tempfile.mkdtemp()
mode, prints, warns = scan(d4)
report("no rasters at all", mode, prints, warns)
assert warns and warns[0].startswith("CRITICAL:"), warns

# ── 5. both canvases are pinned to the CRS their picks are read as ───────────
def crs_calls(canvas):
    return [c[0][0] for c in canvas.setDestinationCrs.call_args_list]

win._apply_canvas_crs()
# NISAR is UTM, C1/L8 are WGS84: both canvases render in the working CRS so a
# reference pick is already in the table's units.
assert crs_calls(win.canvas_left)[-1] is win.proj_crs, "left canvas not pinned"
assert crs_calls(win.canvas_right)[-1] is win.proj_crs, "right canvas not pinned"

# adopting a working CRS from the input raster must re-pin the left canvas
class _CRS:
    def __init__(self, authid): self._a = authid
    def isValid(self): return True
    def isGeographic(self): return False
    def authid(self): return self._a
    def description(self): return self._a

new_crs = _CRS("EPSG:32643")
assert win.adopt_working_crs(new_crs) is True
assert win.proj_crs is new_crs
assert crs_calls(win.canvas_left)[-1] is new_crs, "left canvas not re-pinned"
assert crs_calls(win.canvas_right)[-1] is new_crs, "right canvas not re-pinned"
print("\nboth canvases pinned to", win.proj_crs.authid(), "and re-pinned on adopt")

# a reference pick is taken verbatim -- no conversion left to get wrong
tool = R.DragMapTool(win.canvas_right, win, False)
tool.toMapCoordinates = MagicMock(return_value=_PointXY(325000.0, 1900000.0))
win.table.currentRow = MagicMock(return_value=0)
recorded = []
win.table.setItem = lambda row, col, item: recorded.append((col, item))
R.QTableWidgetItem = lambda text: text     # bound at import, patch it there
win.table.item = MagicMock(return_value=None)
tool.update_data(object())
picked = [(c, v) for c, v in recorded if c in (2, 3)]
print("reference pick ->", picked)
assert picked == [(2, "325000.000"), (3, "1900000.000")], picked

# a geographic CRS is refused: the error columns are metres
assert win.adopt_working_crs(_CRS("EPSG:4326")) is True  # not geographic per stub
class _Geo(_CRS):
    def isGeographic(self): return True
assert win.adopt_working_crs(_Geo("EPSG:4326")) is False, "geographic CRS accepted"
print("geographic CRS refused as a working CRS")

# ── 6. a mark on the input canvas drives the reference canvas ────────────────
R.REF_CANVAS_CRS = "working"
win.proj_crs = _CRS("EPSG:32644")
win._rebuild_transforms()

# one reference tile, its projected ring a 2 km box around (325000, 1900000)
tile = "/refs/N17E78.tif"
win.ref_footprints = {tile: {
    "ring": [(78.0, 18.0), (79.0, 18.0), (79.0, 17.0), (78.0, 17.0)],
    "ring_proj": [(324000.0, 1901000.0), (326000.0, 1901000.0),
                  (326000.0, 1899000.0), (324000.0, 1899000.0)],
    "band": "UNK", "crs": None, "granule": None, "source": "tile-name (1 deg)"}}
win.ref_tif_list = [tile]

inside  = R.QgsPointXY(325000.0, 1900000.0)
outside = R.QgsPointXY(500000.0, 1900000.0)
# QgsGeometry is a mock, so drive containment off the real ring instead
def _fake_geom(wkt=None):
    g = MagicMock()
    g.contains = lambda pt: getattr(pt, "_inside", False)
    g.area = lambda: 4.0e6
    return g
R.QgsGeometry.fromWkt = _fake_geom
R.QgsGeometry.fromPointXY = lambda pt: type("G", (), {"_inside": pt is inside})()

assert win.reference_for_point(inside) == tile
assert win.reference_for_point(outside) is None
print("\nreference_for_point: inside ->", os.path.basename(tile), ", outside -> None")

win.current_ref_layer = None
win._load_ref_layer = MagicMock(return_value=MagicMock())
win.canvas_right.setCenter.reset_mock()
win.canvas_right.zoomScale.reset_mock()
win.draw_marker = MagicMock()

win.canvas_right.setExtent.reset_mock()
win.canvas_right.size.return_value = MagicMock(width=lambda: 800,
                                               height=lambda: 400)
win.follow_input_point(inside)
assert win._load_ref_layer.called, "tile covering the point was not loaded"
# The view must be set from an explicit ground rectangle. setCenter + zoomScale
# is what left the canvas at the origin: a scale silently does nothing on a
# canvas that has not been laid out, and a click then read as (-456, -244).
assert win.canvas_right.setExtent.called, "reference view never got an extent"
rect = win.canvas_right.setExtent.call_args[0][0]
cx = (rect.xMinimum() + rect.xMaximum()) / 2.0
cy = (rect.yMinimum() + rect.yMaximum()) / 2.0
assert (cx, cy) == (325000.0, 1900000.0), (cx, cy)
assert rect.xMaximum() - rect.xMinimum() == R.REF_VIEW_WIDTH_M, rect
# 2:1 canvas -> half the ground height, so the aspect is honoured
assert rect.yMaximum() - rect.yMinimum() == R.REF_VIEW_WIDTH_M / 2.0, rect
assert win.draw_marker.called, "reference canvas was not marked"
print("follow_input_point: loaded tile, extent", rect, "centred on", (cx, cy),
      ", marked")

# and the marker sits at the pick, in the canvas's CRS
marked = win.draw_marker.call_args[0][0]
assert (marked.x(), marked.y()) == (325000.0, 1900000.0), marked

# ── 7. the extent handed to the canvas is in the canvas's CRS ────────────────
wgs_layer = MagicMock()
wgs_layer.crs.return_value = _CRS("EPSG:4326")
wgs_layer.extent.return_value = _Rect(78.0, 17.0, 79.0, 18.0)
transformed = _Rect(300000.0, 1880000.0, 400000.0, 1990000.0)
R.QgsCoordinateTransform = MagicMock(
    return_value=MagicMock(transformBoundingBox=MagicMock(return_value=transformed)))
got = win._extent_in_ref_canvas(wgs_layer)
assert got is transformed, "WGS84 extent handed to a UTM canvas unconverted"
print("extent of a WGS84 layer converted for the UTM canvas")

same = MagicMock()
same.crs.return_value = win.proj_crs
same.extent.return_value = _Rect(1, 2, 3, 4)
assert win._extent_in_ref_canvas(same) is same.extent.return_value, \
    "same-CRS extent should not be transformed"
print("same-CRS extent passed through untouched")

# ── 8. the wgs84 fallback converts picks instead of pixels ───────────────────
R.REF_CANVAS_CRS = "wgs84"
win.transform_proj_to_wgs = MagicMock(
    transform=MagicMock(return_value=R.QgsPointXY(78.5, 17.2)))
win.transform_wgs_to_proj = MagicMock(
    transform=MagicMock(return_value=R.QgsPointXY(325000.0, 1900000.0)))
assert win._ref_canvas_crs() is win.wgs84_crs
out = win._to_ref_canvas(inside)
assert win.transform_proj_to_wgs.transform.called, "no conversion under wgs84 mode"
back = win._from_ref_canvas(out)
assert win.transform_wgs_to_proj.transform.called
# the view rectangle is built in metres and then converted, so it is the same
# patch of ground either way
win.transform_proj_to_wgs.transformBoundingBox = MagicMock(
    return_value=_Rect(78.49, 17.19, 78.51, 17.21))
wgs_rect = win._ref_view_rect(inside)
assert win.transform_proj_to_wgs.transformBoundingBox.called, \
    "metric view rect handed to a WGS84 canvas unconverted"
metric = win.transform_proj_to_wgs.transformBoundingBox.call_args[0][0]
assert metric.xMaximum() - metric.xMinimum() == R.REF_VIEW_WIDTH_M, metric
print("REF_CANVAS_CRS='wgs84': canvas is WGS84, picks convert both ways, "
      "view rect converted", metric, "->", wgs_rect)
R.REF_CANVAS_CRS = "working"

# ── 9. every map tool applies to BOTH canvases ───────────────────────────────
# Real checkable buttons: the mock would report every button as checked at once,
# so an exclusive row could not be told apart from a broken one.
class _Button:
    def __init__(self): self._on = False
    def setCheckable(self, _): pass
    def setToolTip(self, _): pass
    def setChecked(self, v):
        if v:
            for b in win.tool_buttons.values():
                b._on = False
        self._on = bool(v)
    def isChecked(self): return self._on

win.tool_buttons = {m: _Button() for m in R.MAP_TOOLS}
win.tool_buttons[R.TOOL_MARK].setChecked(True)
win.init_map_tools()

def tools():
    return (win.canvas_left.setMapTool.call_args[0][0],
            win.canvas_right.setMapTool.call_args[0][0])

for mode in R.MAP_TOOLS:
    win.set_map_tool(mode)
    assert tools() == win.map_tools[mode], f"{mode} not applied to both canvases"
    checked = [m for m, b in win.tool_buttons.items() if b.isChecked()]
    assert checked == [mode], f"tool row not exclusive: {checked}"
print("\nall four tools apply to both canvases, one selected at a time:",
      ", ".join(R.MAP_TOOLS))

# four distinct tool objects per canvas, and zoom-out really is the out variant
per_canvas = [win.map_tools[m][0] for m in R.MAP_TOOLS]
assert len(set(map(id, per_canvas))) == 4, "map tools are not distinct"
assert win.map_tools[R.TOOL_PAN][0]._tool == "pan"
zoom_in  = win.map_tools[R.TOOL_ZOOM_IN][0]
zoom_out = win.map_tools[R.TOOL_ZOOM_OUT][0]
assert zoom_in._tool == zoom_out._tool == "zoom"
assert zoom_in._args[1] is False, zoom_in._args
assert zoom_out._args[1] is True, zoom_out._args
print("zoom in/out built as the in and out variants, one per canvas")

# marking still routes to the DragMapTool the measuring paths reach by name
win.set_map_tool(R.TOOL_MARK)
assert tools() == (win.tool_left, win.tool_right), "marking not restored"

# ── 9b. with Sync Maps on, the reference follows the input's scale ────────────
win.cb_sync.isChecked = MagicMock(return_value=True)
win.canvas_left.extent = MagicMock(return_value=_Rect(320000, 1898000,
                                                      330000, 1902000))
rect = win._ref_view_rect(_PointXY(325000.0, 1900000.0))
assert rect.xMaximum() - rect.xMinimum() == 10000.0, rect
print("zoomed input (10 km wide) -> reference view matches:", rect)

# unsynced, it falls back to the fixed reference width
win.cb_sync.isChecked = MagicMock(return_value=False)
rect = win._ref_view_rect(_PointXY(325000.0, 1900000.0))
assert rect.xMaximum() - rect.xMinimum() == R.REF_VIEW_WIDTH_M, rect
print("sync off -> reference view back to REF_VIEW_WIDTH_M:", rect)

# a degenerate extent must not produce a zero-width view
win.cb_sync.isChecked = MagicMock(return_value=True)
win.canvas_left.extent = MagicMock(return_value=_Rect(0, 0, 0, 0))
rect = win._ref_view_rect(_PointXY(325000.0, 1900000.0))
assert rect.xMaximum() - rect.xMinimum() == R.REF_VIEW_WIDTH_M, rect
print("degenerate input extent -> falls back, not a zero-width view")
win.cb_sync.isChecked = MagicMock(return_value=False)

# ── 10. the input R/G/B picker ───────────────────────────────────────────────
class _Provider:
    def __init__(self, n): self._n = n
    def bandCount(self): return self._n
    def dataType(self, band): return 6
    def cumulativeCut(self, band, lo, hi): return (0.1 * band, 10.0 * band)
    def bandStatistics(self, band):
        return MagicMock(minimumValue=0.0, maximumValue=1.0)


def fake_layer(n, names=None):
    lyr = MagicMock()
    lyr.isValid.return_value = True
    lyr.dataProvider.return_value = _Provider(n)
    lyr.bandName.side_effect = (lambda b: names[b - 1]) if names else \
        (lambda b: f"Band {b:03d}")
    return lyr

# names the raster carries are used; QGIS's synthetic 'Band 001' is not
pol = fake_layer(3, ["HH", "HV", "HH/HV"])
assert win._band_labels(pol) == ["1: HH", "2: HV", "3: HH/HV"], win._band_labels(pol)
assert win._band_labels(fake_layer(2)) == ["Band 1", "Band 2"]
print("band labels from the raster:", win._band_labels(pol))

captured = []
R.QgsMultiBandColorRenderer = lambda p, r, g, b: (
    captured.append(("rgb", r, g, b)) or MagicMock())
R.QgsSingleBandGrayRenderer = lambda p, b: (
    captured.append(("grey", b)) or MagicMock())

# a real combo stand-in: the mock would report one shared current index
class _Combo:
    def __init__(self): self._items, self._idx, self._visible = [], -1, True
    def blockSignals(self, _): pass
    def clear(self): self._items, self._idx = [], -1
    def addItem(self, t): self._items.append(t)
    def count(self): return len(self._items)
    def setCurrentIndex(self, i): self._idx = i
    def setVisible(self, v): self._visible = bool(v)
    def currentIndex(self): return self._idx
    def currentText(self): return self._items[self._idx] if self._idx >= 0 else ""

win.norm_bounds = {"input": {}, "ref": {}}

# NISAR carries two bands; every slot offers both and the default is 1,1,1
dual = fake_layer(2, ["HH", "HV"])
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.input_tif_layer = dual
win.band_container.show.reset_mock()
win.populate_band_picker(dual)
assert [c._items for c in win.band_combos] == [["1: HH", "2: HV"]] * 3, \
    "each slot should offer every band"
assert [c.currentIndex() for c in win.band_combos] == [0, 0, 0], "default is not 1,1,1"
assert captured[-1] == ("rgb", 1, 1, 1), captured
assert win.band_container.show.called, "picker hidden for a 2-band raster"
print("2-band input -> every slot lists both, default", captured[-1])

# any band in any slot, repeats allowed
win.band_combos[1].setCurrentIndex(1)
win.apply_input_bands()
assert captured[-1] == ("rgb", 1, 2, 1), captured
print("HH/HV/HH selected ->", captured[-1])

# a 3-band chip works the same way
win.input_tif_layer = pol
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.populate_band_picker(pol)
assert [c._items for c in win.band_combos] == [["1: HH", "2: HV", "3: HH/HV"]] * 3
assert captured[-1] == ("rgb", 1, 1, 1), captured
for combo, idx in zip(win.band_combos, (0, 1, 2)):
    combo.setCurrentIndex(idx)
win.apply_input_bands()
assert captured[-1] == ("rgb", 1, 2, 3), captured
print("3-band input -> composite", captured[-1])

# single band: nothing to choose between, so no picker, still rendered
single = fake_layer(1, ["HH"])
win.input_tif_layer = single
win.band_container.show.reset_mock()
win.populate_band_picker(single)
# the combos go, but the overlay stays: Normalize acts on any raster and has to
# stay reachable even when there is nothing to compose
assert not any(c._visible for c in win.band_combos), "combos shown for one band"
assert win.band_container.show.called, "overlay hidden, taking Normalize with it"
assert captured[-1] == ("grey", 1), captured
print("1-band input -> combos hidden, overlay kept, rendered", captured[-1])

# ── 11. Normalize pins a stretch, and panning does not disturb it ────────────
win.input_tif_layer = dual
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.norm_bounds = {"input": {}, "ref": {}}
win.populate_band_picker(dual)

# with nothing pinned, the band goes through the percentile clip
before = len(captured)
win.apply_input_bands()
assert captured[-1][0] == "rgb", captured[-1]

# pinning band 1 makes _stretch_for return those bounds verbatim
win.norm_bounds["input"][1] = (0.167, 0.5715)
ce = win._stretch_for(dual, 1)
assert ce.setMinimumValue.call_args[0][0] == 0.167, ce.setMinimumValue.call_args
assert ce.setMaximumValue.call_args[0][0] == 0.5715, ce.setMaximumValue.call_args
print("\nNormalize pins band 1 at", win.norm_bounds["input"][1],
      "and _stretch_for returns it verbatim")

# an unpinned band still falls through to the percentile clip
measured = []
_orig_cut = win.sampled_cut
win.sampled_cut = lambda p, b, lo, hi, ext=None: (
    measured.append(b) or (0.1, 0.9))
win._stretch_cache = {}
win._stretch_for(dual, 2)
assert measured == [2], measured
win._stretch_for(dual, 1)          # pinned: must not measure again
assert measured == [2], measured
win.sampled_cut = _orig_cut
print("unpinned bands measure; a pinned band never re-measures")

# ── 11b. stretch statistics are sampled and cached ───────────────────────────
# An unsampled cumulativeCut reads the WHOLE raster at full resolution, per
# band, on the GUI thread -- the regression that froze QGIS for minutes on load.
calls = {"cut": [], "stats": []}

class _BigProvider:
    def bandCount(self): return 2
    def dataType(self, band): return 6
    def cumulativeCut(self, band, lo, hi, extent=None, sample=None):
        calls["cut"].append((band, sample))
        if sample is None:                     # the unsampled overload
            raise AssertionError("unsampled cumulativeCut would scan everything")
        return (0.1, 10.0)
    def bandStatistics(self, band, stats=None, extent=None, sample=None):
        calls["stats"].append((band, sample))
        return MagicMock(minimumValue=0.0, maximumValue=1.0)

big = MagicMock()
big.isValid.return_value = True
big.source.return_value = "/big/scene.tif"
big.dataProvider.return_value = _BigProvider()
big.bandName.side_effect = lambda b: ["HH", "HV"][b - 1]

win.cb_normalize_input.isChecked = MagicMock(return_value=False)
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.input_tif_layer = big
win._stretch_cache = {}
win.populate_band_picker(big)

assert calls["cut"], "no stretch computed"
assert all(sample == R.RASTER_SAMPLE_SIZE for _, sample in calls["cut"]), calls
print("\nstretch sampled at", R.RASTER_SAMPLE_SIZE, "px, not the whole raster")

# default is 1,1,1 -- three channels on one band must not cost three passes
assert len(calls["cut"]) == 1, calls["cut"]
print("R=G=B=1 -> one statistics pass, not three:", calls["cut"])

# re-picking the channel order reuses the cache
before = len(calls["cut"])
win.band_combos[1].setCurrentIndex(1)
win.apply_input_bands()
assert [b for b, _ in calls["cut"]] == [1, 2], calls["cut"]
win.band_combos[1].setCurrentIndex(0)
win.apply_input_bands()
assert len(calls["cut"]) == before + 1, "re-picking recomputed a cached band"
print("re-picking a band reuses the cache:", calls["cut"])

# Normalize now measures the CURRENT VIEW and pins the result, so it is an
# action rather than a mode: nothing about it is cached by _stretch_for.
win.canvas_left.extent = MagicMock(return_value=_Rect(324000, 1899000,
                                                      326000, 1901000))
win.clip_combos = {"input": MagicMock(currentData=lambda: 2.0),
                   "ref": MagicMock(currentData=lambda: 2.0)}
win.view_extent_for = MagicMock(return_value=_Rect(324000, 1899000,
                                                   326000, 1901000))
win.pixel_window = MagicMock(return_value=(10, 20, 400, 400))
win._gamma_bounds = MagicMock(return_value=(0.2, 0.5))
win.norm_bounds = {"input": {}, "ref": {}}
win.input_tif_layer = big
win.band_combos = [_Combo(), _Combo(), _Combo()]
win.populate_band_picker(big)
win.normalize_to_view("input")

assert win._gamma_bounds.called, "Normalize did not reach the gamma bounds"
args = win._gamma_bounds.call_args[0]
assert args[4] == (10, 20, 400, 400), args      # the view's pixel window
assert args[5] == 2.0, args                     # the chosen clip percentage
assert win.norm_bounds["input"][1] == (0.2, 0.5), win.norm_bounds
print("Normalize measures the view window", args[4], "at clip", args[5],
      "and pins", win.norm_bounds["input"][1])

# panning must not disturb it: the pinned bounds survive a re-render
win.canvas_left.extent = MagicMock(return_value=_Rect(400000, 1899000,
                                                      402000, 1901000))
win._gamma_bounds.reset_mock()
win.apply_input_bands()
assert not win._gamma_bounds.called, "re-render re-measured after a pan"
assert win.norm_bounds["input"][1] == (0.2, 0.5), win.norm_bounds
print("panning does not re-stretch; the pinned bounds survive")

# the clip percentage reaches the measurement
win.clip_combos["input"] = MagicMock(currentData=lambda: 5.0)
assert win.clip_percent("input") == 5.0
win.clip_combos["input"] = MagicMock(currentData=lambda: None,
                                     currentText=lambda: "1%")
assert win.clip_percent("input") == 1.0      # falls back to the label
print("clip percentage read from the tool, label as fallback")

# loading a raster drops only that raster's cached bounds
win.clear_stretch_cache("/big/scene.tif")
assert not any(k[0] == "/big/scene.tif" for k in win._stretch_cache)
print("loading a raster clears its own cached bounds")

# ── 11c. the mark is a solid cross in a contrasting colour ───────────────────
made = []
R.QgsVertexMarker = MagicMock(side_effect=lambda canvas: (
    made.append(MagicMock()) or made[-1]))
R.QColor = lambda *rgb: ("color", rgb)

# section 6 replaced draw_marker with a mock to check follow_input_point; put
# the real method back so this exercises the shipped drawing code
win.draw_marker = R.QCDashboard.draw_marker.__get__(win, R.QCDashboard)
win.markers = {"left": [], "right": []}
win.canvas_left.scene.return_value = MagicMock()
win.draw_marker(_PointXY(325000.0, 1900000.0), win.canvas_left,
                R.MARKER_COLOR_INPUT)

items = win.markers["left"]
assert len(items) == 1, items          # one solid cross, no halo behind it
assert items[0].setColor.call_args[0][0] == ("color", R.MARKER_COLOR_INPUT)

# The input is shown as a red/cyan composite and the reference as greyscale, so
# a red, green or grey mark disappears into one of them.
for name, rgb in (("input", R.MARKER_COLOR_INPUT), ("ref", R.MARKER_COLOR_REF)):
    r, g, b = rgb
    assert not (r == g == b), f"{name} mark is a grey"
    assert rgb not in ((255, 0, 0), (0, 255, 0)), f"{name} mark is red or green"
assert R.MARKER_COLOR_INPUT != R.MARKER_COLOR_REF, "both marks the same colour"
print("\nsolid single cross, input %s / ref %s, neither red, green nor grey"
      % (R.MARKER_COLOR_INPUT, R.MARKER_COLOR_REF))

# re-marking takes the old item off the scene
scene = win.canvas_left.scene.return_value
scene.removeItem.reset_mock()
win.draw_marker(_PointXY(325010.0, 1900000.0), win.canvas_left,
                R.MARKER_COLOR_INPUT)
removed = [c[0][0] for c in scene.removeItem.call_args_list]
assert removed == [items[0]], removed
print("re-marking removes the previous cross, leaving no orphan")

# clear_markers empties both canvases
win.canvas_right.scene.return_value = MagicMock()
win.draw_marker(_PointXY(325000.0, 1900000.0), win.canvas_right,
                R.MARKER_COLOR_REF)
win.clear_markers()
assert win.markers["left"] == [] and win.markers["right"] == [], win.markers
print("clear_markers empties both canvases")

# ── 12. arrow keys nudge the mark, not the view ──────────────────────────────
class _Table:
    """Enough QTableWidget for the nudge path, with real cell values."""
    def __init__(self, row, cells):
        self._row, self._cells = row, dict(cells)
    def currentRow(self): return self._row
    def item(self, row, col):
        v = self._cells.get((row, col))
        return None if v is None else MagicMock(text=lambda v=v: v)
    def setItem(self, row, col, item): self._cells[(row, col)] = str(item)
    def blockSignals(self, _): pass
    def rowCount(self): return 1

def utm_layer(px, py):
    lyr = MagicMock()
    lyr.isValid.return_value = True
    lyr.crs.return_value = win.proj_crs
    lyr.rasterUnitsPerPixelX.return_value = px
    lyr.rasterUnitsPerPixelY.return_value = py
    return lyr

win.proj_crs = _CRS("EPSG:32644")
win._rebuild_transforms()
win.input_tif_layer = utm_layer(5.0, 5.0)      # NISAR posts at 5 m
win.current_ref_layer = None
win.draw_marker = MagicMock()
win.calculate_error = MagicMock()
win.canvas_left.setExtent.reset_mock()
win.canvas_left.setCenter.reset_mock()

# In X/Y marked at (325000, 1900000); one press of Right
win.table = _Table(0, {(0, 0): "325000.000", (0, 1): "1900000.000",
                       (0, 2): "0.000", (0, 3): "0.000"})
assert win.nudge_point(True, 1, 0) is True
assert win.table._cells[(0, 0)] == "325005.000", win.table._cells
assert win.table._cells[(0, 1)] == "1900000.000", win.table._cells
print("\nRight arrow -> In X moves one 5 m pixel east:",
      win.table._cells[(0, 0)])

# Up is north (+Y), and Shift multiplies the step
assert win.nudge_point(True, 0, 1) is True
assert win.table._cells[(0, 1)] == "1900005.000", win.table._cells
assert win.nudge_point(True, 0, -R.NUDGE_SHIFT_FACTOR) is True
assert win.table._cells[(0, 1)] == "1899955.000", win.table._cells
print("Up = north, Shift steps", R.NUDGE_SHIFT_FACTOR, "pixels:",
      win.table._cells[(0, 1)])

# the error is recomputed and the marker redrawn -- but the view is untouched
assert win.calculate_error.called and win.draw_marker.called
assert not win.canvas_left.setExtent.called, "nudging moved the view"
assert not win.canvas_left.setCenter.called, "nudging moved the view"
print("marker and error updated, view untouched")

# nothing marked on that side yet -> not handled, so the canvas still pans
assert win.nudge_point(False, 1, 0) is False, "nudged an unmarked Ref"
win.table = _Table(0, {(0, 0): "0.000", (0, 1): "0.000"})
assert win.nudge_point(True, 1, 0) is False, "nudged an unmarked In"
win.table = _Table(-1, {})
assert win.nudge_point(True, 1, 0) is False, "nudged with no row selected"
print("unmarked side / no row -> not handled, canvas keeps the key")

# a WGS84 reference tile: its pixel is degrees, so the step is measured through
# the transform rather than added to a UTM coordinate as if it were metres
wgs = MagicMock()
wgs.isValid.return_value = True
wgs.crs.return_value = _CRS("EPSG:4326")
wgs.rasterUnitsPerPixelX.return_value = 2.5e-5     # ~2.8 m at this latitude
wgs.rasterUnitsPerPixelY.return_value = 2.5e-5
win.current_ref_layer = wgs
R.QgsCoordinateTransform = MagicMock(side_effect=lambda src, dst, prj: MagicMock(
    transform=lambda p: _PointXY(p.x() + 1.0, p.y() + 2.0)))
step = win._pixel_step(False, _PointXY(325000.0, 1900000.0))
assert step is not None and step[0] > 0 and step[1] > 0, step
print("WGS84 reference pixel measured through the transform:",
      tuple(round(v, 3) for v in step))

# ── 12a. the normalize range is read from the raster, per band ───────────────
# Reported from a real L-band chip: band 1 reads 0.158..0.580 and band 2
# 0.07..0.27 -- linear amplitude, a third scale after dB and DN. No hard-coded
# range fits all three, so it has to come from the raster.
class _AmpProvider:
    """A two-band amplitude raster with the reported ranges."""
    RANGES = {1: (0.158, 0.580), 2: (0.07, 0.27)}
    def __init__(self): self.cut_calls = []
    def bandCount(self): return 2
    def dataType(self, band): return 6
    def cumulativeCut(self, band, lo, hi, extent=None, sample=None):
        self.cut_calls.append((band, lo, hi, sample))
        if sample is None:
            raise AssertionError("unsampled cumulativeCut")
        return self.RANGES[band] if (lo, hi) == (0.0, 1.0) else (0.2, 0.5)
    def bandStatistics(self, band, stats=None, extent=None, sample=None):
        lo, hi = self.RANGES[band]
        return MagicMock(minimumValue=lo, maximumValue=hi)

amp = _AmpProvider()
assert win.normalize_range(amp, 1) == (0.158, 0.580), win.normalize_range(amp, 1)
assert win.normalize_range(amp, 2) == (0.07, 0.27), win.normalize_range(amp, 2)
print("\nnormalize range read from the raster per band:",
      win.normalize_range(amp, 1), win.normalize_range(amp, 2))

# full min/max, not a percentile clip -- the percentiles are taken later on the
# stretched values and clipping twice would compound
assert all(args[1:3] == (0.0, 1.0) for args in amp.cut_calls), amp.cut_calls
assert all(args[3] == R.RASTER_SAMPLE_SIZE for args in amp.cut_calls), amp.cut_calls

# a provider that cannot be measured falls back rather than failing
class _Unmeasurable:
    def cumulativeCut(self, *a, **k): raise RuntimeError("no histogram")
    def bandStatistics(self, *a, **k): raise RuntimeError("no stats")
assert win.normalize_range(_Unmeasurable(), 1) == (float(R.NORM_MIN),
                                                   float(R.NORM_MAX))
print("unmeasurable raster falls back to NORM_MIN..NORM_MAX:",
      win.normalize_range(_Unmeasurable(), 1))

# and the flag turns the whole thing off without touching anything else
R.NORM_USE_DATA_RANGE = False
assert win.normalize_range(amp, 1) == (float(R.NORM_MIN), float(R.NORM_MAX))
R.NORM_USE_DATA_RANGE = True
print("NORM_USE_DATA_RANGE=False restores the fixed range")

# ── 12b. the gamma stretch on dB data, and on data carrying NaN ──────────────
# Reported: 'Normalize NISAR' renders the input black while the reference
# normalises fine. Two independent causes, both exercised here against real
# numpy arrays through a stubbed GDAL.
import numpy as np

def fake_gdal(array, nodata=None):
    band = MagicMock()
    band.XSize, band.YSize = array.shape[1], array.shape[0]
    band.ReadAsArray = MagicMock(return_value=array)
    band.GetNoDataValue = MagicMock(return_value=nodata)
    ds = MagicMock()
    ds.GetRasterBand = MagicMock(return_value=band)
    osgeo = types.ModuleType("osgeo")
    gdal = types.ModuleType("osgeo.gdal")
    gdal.GA_ReadOnly = 0
    gdal.Open = MagicMock(return_value=ds)
    osgeo.gdal = gdal
    sys.modules["osgeo"], sys.modules["osgeo.gdal"] = osgeo, gdal

# section 11 replaced _gamma_bounds with a mock to check the Normalize routing;
# put the real method back so this exercises the shipped arithmetic
win._gamma_bounds = R.QCDashboard._gamma_bounds.__get__(win, R.QCDashboard)

rng = np.random.default_rng(0)

# (1) an S-band-like DN scene over the reference's own 0-1500 range: unchanged
dn = rng.uniform(20.0, 900.0, size=(64, 64))
fake_gdal(dn)
lo, hi = win._gamma_bounds("/ref.tif", 1)
assert lo is not None and hi > lo, (lo, hi)
assert R.NORM_MIN <= lo < hi <= R.NORM_MAX, (lo, hi)
print("\nreference DN scene over NORM_MIN..NORM_MAX: bounds",
      (round(lo, 1), round(hi, 1)), "-- behaviour unchanged")

# (2) a NISAR chip in float32 dB. Forcing the reference's 0-1500 range clips
# every pixel to the very bottom of the curve, which is what rendered black.
db = rng.uniform(-28.0, 2.0, size=(64, 64))
span = float(db.max() - db.min())
fake_gdal(db)

# It does not fail loudly -- it returns a technically valid range that covers
# almost none of the data, so nearly every pixel clamps to black. That is the
# reported symptom, pinned here so the fix cannot silently regress.
forced = win._gamma_bounds("/in.tif", 1)
forced_span = forced[1] - forced[0]
assert forced_span / span < 0.10, (forced, span)
print("dB scene forced through 0-1500 -> stretch covers only "
      f"{100 * forced_span / span:.1f}% of the data: that is the black render")

# over its own range it produces a usable stretch inside the data
own = win._gamma_bounds("/in.tif", 1, float(db.min()), float(db.max()))
assert own[0] is not None and own[1] > own[0], own
assert db.min() <= own[0] < own[1] <= db.max(), own
own_span = own[1] - own[0]
assert own_span / span > 0.5, (own, span)
print("dB scene over its own range -> covers "
      f"{100 * own_span / span:.1f}% of the data:",
      tuple(round(v, 2) for v in own))

# (3) NaN nodata, which --gtiff writes. NaN != NaN, so a nodata test alone lets
# every NaN through and one NaN makes every percentile NaN.
holed = db.copy()
holed[:8, :8] = np.nan
fake_gdal(holed, nodata=float("nan"))
with_nan = win._gamma_bounds("/in.tif", 1, float(db.min()), float(db.max()))
assert with_nan[0] is not None and np.isfinite(with_nan[0]), with_nan
assert abs(with_nan[0] - own[0]) < 1.0 and abs(with_nan[1] - own[1]) < 1.0, \
    (with_nan, own)
print("NaN nodata excluded -> bounds still finite and close to the clean scene:",
      tuple(round(v, 2) for v in with_nan))

# all-NaN is refused rather than returning nonsense
fake_gdal(np.full((32, 32), np.nan))
assert win._gamma_bounds("/in.tif", 1, -30.0, 5.0) == (None, None)
# a degenerate range is refused too
fake_gdal(db)
assert win._gamma_bounds("/in.tif", 1, 5.0, 5.0) == (None, None)
print("all-NaN and degenerate ranges refused, so the caller falls back")

# ── 13. shapefile export collects only fully marked rows ─────────────────────
class _Rows:
    def __init__(self, rows): self._rows = rows
    def rowCount(self): return len(self._rows)
    def item(self, row, col):
        v = self._rows[row][col]
        return None if v is None else MagicMock(text=lambda v=v: v)

win.transform_proj_to_wgs = MagicMock(
    transform=MagicMock(return_value=_PointXY(78.5, 17.2)))
win.table = _Rows([
    ["325010.000", "1900007.000", "325000.000", "1900000.000"],   # complete
    ["0.000", "0.000", "325000.000", "1900000.000"],              # no input
    ["325010.000", "1900007.000", "0.000", "0.000"],              # no reference
    ["325020.000", "1900000.000", "325000.000", "1900000.000"],   # complete
    [None, None, None, None],                                     # empty row
])
rows = win.export_rows()
assert [r["row"] for r in rows] == [1, 4], [r["row"] for r in rows]
print("\nexport skips half-marked and empty rows, keeping table numbering:",
      [r["row"] for r in rows])
assert rows[0]["dx"] == 10.0 and rows[0]["dy"] == 7.0, rows[0]
assert rows[1]["dx"] == 20.0 and rows[1]["dy"] == 0.0, rows[1]
assert rows[1]["bearing"] == 90.0, rows[1]        # due east
print("errors and bearings carried through:",
      [(r["dx"], r["dy"], r["bearing"]) for r in rows])

# a row is skipped rather than exported as an offset from the origin
assert all(r["ref_x"] != 0.0 for r in rows), rows

# lon/lat come from the working-CRS transform, not from the map units
assert rows[0]["in_lon"] == 78.5 and rows[0]["in_lat"] == 17.2, rows[0]
print("lon/lat filled from the transform, for quiver.py's columns")

# nothing marked at all -> the writer is never reached
win.table = _Rows([["0.000", "0.000", "0.000", "0.000"]])
assert win.export_rows() == []
qw.QFileDialog.getSaveFileName = MagicMock(return_value=("/tmp/should_not.shp", ""))
del warned[:]
win.save_shapefile()
assert not qw.QFileDialog.getSaveFileName.called, "asked for a path with no rows"
assert warned and "marked" in warned[0], warned
print("no marked rows -> warned, no file dialog, nothing written")

for d in (d1, d2, d3, d4):
    shutil.rmtree(d, ignore_errors=True)
print("\nstubbed integration OK")
