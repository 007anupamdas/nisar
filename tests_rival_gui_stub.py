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
import os, sys, shutil, tempfile

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
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return MagicMock()
for mod, names in ((qw, ["QMainWindow", "QWidget"]),
                   (qg, ["QgsMapTool"]),
                   (qc, ["QObject"])):
    for n in names:
        setattr(mod, n, type(n, (_Base,), {}))

# A MagicMock caches its return_value, so QgsMapCanvas() would hand back ONE
# object for both canvases and hide any per-canvas difference. Give each call a
# fresh mock.
qg.QgsMapCanvas = MagicMock(side_effect=lambda *a, **k: MagicMock())

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
class _Rect:
    def __init__(self, *b): self._b = b
    def xMinimum(self): return self._b[0]
    def yMinimum(self): return self._b[1]
    def xMaximum(self): return self._b[2]
    def yMaximum(self): return self._b[3]

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

win.follow_input_point(inside)
assert win._load_ref_layer.called, "tile covering the point was not loaded"
centred = win.canvas_right.setCenter.call_args[0][0]
assert (centred.x(), centred.y()) == (325000.0, 1900000.0), centred
assert win.canvas_right.zoomScale.called, "reference canvas did not zoom"
assert win.draw_marker.called, "reference canvas was not marked"
print("follow_input_point: loaded tile, centred", (centred.x(), centred.y()),
      ", zoomed, marked")

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
print("REF_CANVAS_CRS='wgs84': canvas is WGS84 and picks convert both ways")
R.REF_CANVAS_CRS = "working"

for d in (d1, d2, d3, d4):
    shutil.rmtree(d, ignore_errors=True)
print("\nstubbed integration OK")
