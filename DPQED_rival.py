import csv
import os
import sys
import re
import threading
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QFileDialog, QHeaderView, QCheckBox, QComboBox,
                             QMessageBox, QApplication, QShortcut, QLabel,
                             QFrame)
from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtGui import QKeySequence, QFont
from qgis.gui import QgsMapCanvas, QgsMapTool, QgsVertexMarker
from qgis.core import (QgsProject, QgsPointXY, QgsRasterLayer, QgsGeometry,
                       QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsSingleBandGrayRenderer, QgsContrastEnhancement, QgsRectangle)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
ZOOM_INPUT_PLACEHOLDER = 700000000   # fixed zoom for left/input canvas
ZOOM_REF_PLACEHOLDER   = 70000   # fixed zoom for right/ref canvas
left_view_width = 5000

SCALE_LEFT  = 700000000      # left canvas fixed zoom scale (UTM metres) — no longer used
SCALE_RIGHT = 70000          # right canvas fixed zoom scale (WGS84 degrees) — no longer used
NORM_MIN    = 0           # SAR normalization min DN
NORM_MAX    = 1500        # SAR normalization max DN
NORM_GAMMA  = 0.5         # gamma exponent for sqrt stretch (0.5 = square root)


# ── MAP TOOL ──────────────────────────────────────────────────────────────────
class DragMapTool(QgsMapTool):
    def __init__(self, canvas, parent, is_left_map):
        super().__init__(canvas)
        self.canvas      = canvas
        self.parent      = parent
        self.is_left_map = is_left_map
        self.dragging    = False
        self.setCursor(Qt.CrossCursor)
        if not self.is_left_map:
            self.transform_to_utm = QgsCoordinateTransform(
                parent.wgs84_crs, parent.utm44n_crs, QgsProject.instance()
            )

    def canvasPressEvent(self, e):
        self.dragging = True
        self.update_data(e.pos())

    def canvasMoveEvent(self, e):
        if self.dragging:
            self.update_data(e.pos())

    def canvasReleaseEvent(self, e):
        self.dragging = False
        self.update_data(e.pos())

    def update_data(self, pos):
        row = self.parent.table.currentRow()
        if row < 0:
            return
        point = self.toMapCoordinates(pos)
        col   = 0 if self.is_left_map else 2
        if not self.is_left_map:
            p = self.transform_to_utm.transform(point)
            sx, sy = p.x(), p.y()
        else:
            sx, sy = point.x(), point.y()
        try:
            self.parent.table.blockSignals(True)
            self.parent.table.setItem(row, col,     QTableWidgetItem(f"{sx:.3f}"))
            self.parent.table.setItem(row, col + 1, QTableWidgetItem(f"{sy:.3f}"))
        finally:
            self.parent.table.blockSignals(False)
        self.parent.draw_marker(point, self.canvas,
                                Qt.red if self.is_left_map else Qt.green)
        self.parent.calculate_error(row)


# ── DROPDOWN RESIZE FILTER ────────────────────────────────────────────────────
class DropdownResizeFilter(QObject):
    def __init__(self, parent, container):
        super().__init__(parent)
        self.container     = container
        self.parent_widget = parent

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Resize:
            self.container.setGeometry(
                10, 10, self.parent_widget.width() - 20, 35
            )
        return super().eventFilter(obj, event)


# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
class QCDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RIVAL - Reference Image Validation and Accuracy Logger")
        self.resize(1500, 900)

        self.markers           = {"left": None, "right": None}
        self.ref_folder_path   = None
        self.ref_meta_data     = {}
        self.ref_meta_data_utm = {}
        self.ref_tif_list      = []
        self.input_tif_layer   = None
        self.current_ref_layer = None
        self._syncing          = False

        self.wgs84_crs  = QgsCoordinateReferenceSystem("EPSG:4326")
        self.utm44n_crs = QgsCoordinateReferenceSystem("EPSG:32644")
        self.transform_utm_to_wgs = QgsCoordinateTransform(
            self.utm44n_crs, self.wgs84_crs, QgsProject.instance()
        )
        self.transform_wgs_to_utm = QgsCoordinateTransform(
            self.wgs84_crs, self.utm44n_crs, QgsProject.instance()
        )

        self._configure_gdal_cache()

        self.canvas_left  = QgsMapCanvas()
        self.canvas_right = QgsMapCanvas()
        self.canvas_left.enableAntiAliasing(False)
        self.canvas_right.enableAntiAliasing(False)
        self.canvas_left.setCachingEnabled(True)
        self.canvas_right.setCachingEnabled(True)
        self.canvas_left.setParallelRenderingEnabled(True)
        self.canvas_right.setParallelRenderingEnabled(True)

        self.dropdown_container = QWidget(self.canvas_right)
        self.dropdown_container.setGeometry(10, 10, 600, 35)
        self.dropdown_container.setStyleSheet("background-color: rgba(255,255,255,153);")
        _dl = QVBoxLayout(self.dropdown_container)
        _dl.setContentsMargins(5, 5, 5, 5)
        self.dropdown_ref = QComboBox()
        _dl.addWidget(self.dropdown_ref)
        self.dropdown_container.hide()
        self.resize_filter = DropdownResizeFilter(self.canvas_right, self.dropdown_container)
        self.canvas_right.installEventFilter(self.resize_filter)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["In X", "In Y", "Ref X", "Ref Y", "Error in X", "Error in Y"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        self.btn_input_tif        = QPushButton("Load Input TIF")
        self.btn_reference_folder = QPushButton("Select Reference Folder")
        self.btn_load  = QPushButton("Load CSV")
        self.btn_load.setToolTip("Load CSV  (Ctrl+O)")
        self.btn_add   = QPushButton("Add Row")
        self.btn_add.setToolTip("Add Row  (Ctrl+N)")
        self.btn_save  = QPushButton("Export CSV")
        self.btn_save.setToolTip("Export CSV  (Ctrl+S)")
        self.btn_del   = QPushButton("Delete Row")
        self.btn_del.setToolTip("Delete Row  (Ctrl+Delete)")
        self.cb_sync      = QCheckBox("Sync Maps")
        self.cb_sync.setChecked(True)
        self.cb_normalize = QCheckBox("Normalize Ref")
        self.cb_normalize.setChecked(False)
        self.cb_normalize.setToolTip(
            "OFF = QGIS default auto-stretch (natural look)\n"
            "ON  = SAR sqrt-gamma stretch over DN 0-1500"
        )

        _stat_font = QFont()
        _stat_font.setBold(True)
        _stat_font.setPointSize(10)
        self.lbl_rmse_x = QLabel("RMSE X:  0.000 m")
        self.lbl_rmse_y = QLabel("RMSE Y:  0.000 m")
        self.lbl_ce90   = QLabel("CE90:    0.000 m")
        for lbl in [self.lbl_rmse_x, self.lbl_rmse_y, self.lbl_ce90]:
            lbl.setFont(_stat_font)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFrameShape(QFrame.StyledPanel)
            lbl.setStyleSheet(
                "QLabel {"
                "  background-color: #1e1e2e;"
                "  color: #cdd6f4;"
                "  border: 1px solid #45475a;"
                "  border-radius: 4px;"
                "  padding: 4px 10px;"
                "}"
            )

        stats_layout = QHBoxLayout()
        stats_layout.addStretch()
        stats_layout.addWidget(self.lbl_rmse_x)
        stats_layout.addWidget(self.lbl_rmse_y)
        stats_layout.addWidget(self.lbl_ce90)

        map_layout = QHBoxLayout()
        map_layout.addWidget(self.canvas_left)
        map_layout.addWidget(self.canvas_right)

        btn_layout = QHBoxLayout()
        for w in [self.btn_input_tif, self.btn_reference_folder,
                  self.btn_load, self.btn_add, self.btn_save,
                  self.btn_del, self.cb_sync, self.cb_normalize]:
            btn_layout.addWidget(w)

        main_layout = QVBoxLayout()
        main_layout.addLayout(map_layout, 4)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.table, 2)
        main_layout.addLayout(stats_layout)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.btn_input_tif.clicked.connect(lambda: self.manual_load_tif(True))
        self.btn_reference_folder.clicked.connect(self.select_reference_folder)
        self.btn_load.clicked.connect(self.load_csv_smart)
        self.btn_add.clicked.connect(self.add_manual_row)
        self.btn_save.clicked.connect(self.save_csv)
        self.btn_del.clicked.connect(self.delete_row)
        self.table.itemChanged.connect(self.handle_manual_typing)
        self.table.itemSelectionChanged.connect(self.sync_view_to_row)
        self.canvas_left.extentsChanged.connect(self.sync_canvas_extents)
        self.dropdown_ref.currentIndexChanged.connect(self.load_reference_tif_from_dropdown)
        self.cb_normalize.stateChanged.connect(self.toggle_normalization)

        QShortcut(QKeySequence("Ctrl+N"),      self).activated.connect(self.add_manual_row)
        QShortcut(QKeySequence("Ctrl+S"),      self).activated.connect(self.save_csv)
        QShortcut(QKeySequence("Ctrl+O"),      self).activated.connect(self.load_csv_smart)
        QShortcut(QKeySequence("Ctrl+Delete"), self).activated.connect(self.delete_row)
        QShortcut(QKeySequence("F5"),          self).activated.connect(self.sync_view_to_row)
        QShortcut(QKeySequence("Escape"),      self).activated.connect(self.clear_markers)

        self.auto_connect_layers()
        self.init_map_tools()

    # ── GDAL CACHE ────────────────────────────────────────────────────────────
    def _configure_gdal_cache(self):
        try:
            from osgeo import gdal
            gdal.SetCacheMax(512 * 1024 * 1024)
            gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
            gdal.SetConfigOption("VSI_CACHE",        "TRUE")
            gdal.SetConfigOption("VSI_CACHE_SIZE",   "20000000")
            gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
        except Exception as e:
            print(f"GDAL cache config skipped: {e}")

    # ── OVERVIEW PYRAMIDS ─────────────────────────────────────────────────────
    def ensure_overviews(self, tif_path):
        def _build():
            ds = None
            try:
                from osgeo import gdal
                ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
                if (ds and ds.GetRasterBand(1) and
                        ds.GetRasterBand(1).GetOverviewCount() == 0):
                    print(f"[OVR] Building: {os.path.basename(tif_path)} ...")
                    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
                    print("[OVR] Done.")
            except Exception as e:
                print(f"[OVR] Skipped for {os.path.basename(tif_path)}: {e}")
            finally:
                ds = None
        threading.Thread(target=_build, daemon=True).start()

    # ── NORMALIZATION: SAR SQRT-GAMMA STRETCH ─────────────────────────────────
    def normalize_layer(self, layer):
        """SAR sqrt-gamma stretch: gamma=NORM_GAMMA over DN range NORM_MIN..NORM_MAX.
        Reads a downsampled tile, applies power stretch, inverse-maps 2%-98%
        percentile output back to input DN for QgsContrastEnhancement."""
        if not layer or not layer.isValid():
            return
        try:
            from osgeo import gdal
            ds = gdal.Open(layer.source(), gdal.GA_ReadOnly)
            if not ds:
                return
            band   = ds.GetRasterBand(1)
            xsize  = min(band.XSize, 1000)
            ysize  = min(band.YSize, 1000)
            data   = band.ReadAsArray(0, 0, band.XSize, band.YSize,
                                      xsize, ysize).astype(float)
            nodata = band.GetNoDataValue()
            ds     = None

            # Build valid-pixel mask
            mask = np.ones(data.shape, dtype=bool)
            if nodata is not None:
                mask &= (data != nodata)

            # Gamma stretch: clip to [NORM_MIN, NORM_MAX], normalise, apply power
            dn_range     = max(float(NORM_MAX - NORM_MIN), 1.0)
            data_clipped = np.clip(data, NORM_MIN, NORM_MAX)
            norm         = (data_clipped - NORM_MIN) / dn_range
            stretched    = np.power(norm, NORM_GAMMA) * 255.0

            valid = stretched[mask]
            if valid.size == 0:
                return

            p2_out  = float(np.percentile(valid, 2))
            p98_out = float(np.percentile(valid, 98))

            # Inverse-map stretched percentiles back to input DN values
            inv_gamma = 1.0 / NORM_GAMMA
            min_dn = NORM_MIN + dn_range * ((p2_out  / 255.0) ** inv_gamma)
            max_dn = NORM_MIN + dn_range * ((p98_out / 255.0) ** inv_gamma)

            if max_dn <= min_dn:
                min_dn, max_dn = float(NORM_MIN), float(NORM_MAX)

            provider = layer.dataProvider()
            ce = QgsContrastEnhancement(provider.dataType(1))
            ce.setMinimumValue(min_dn)
            ce.setMaximumValue(max_dn)
            ce.setContrastEnhancementAlgorithm(
                QgsContrastEnhancement.StretchToMinimumMaximum
            )
            renderer = QgsSingleBandGrayRenderer(provider, 1)
            renderer.setContrastEnhancement(ce)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:
            print(f"[NORM] Error: {e}")

    # ── RESET: QGIS DEFAULT AUTO-STRETCH ─────────────────────────────────────
    def reset_normalization(self, layer):
        """Restore QGIS default auto-stretch using full band statistics.
        This gives the natural, visually accurate look preferred by the user."""
        if not layer or not layer.isValid():
            return
        try:
            provider  = layer.dataProvider()
            stats     = provider.bandStatistics(1)
            ce        = QgsContrastEnhancement(provider.dataType(1))
            ce.setContrastEnhancementAlgorithm(
                QgsContrastEnhancement.StretchToMinimumMaximum
            )
            ce.setMinimumValue(stats.minimumValue)
            ce.setMaximumValue(stats.maximumValue)
            renderer = QgsSingleBandGrayRenderer(provider, 1)
            renderer.setContrastEnhancement(ce)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:
            print(f"[RESET NORM] Error: {e}")

    def toggle_normalization(self, state):
        """Toggle SAR gamma stretch ON / restore QGIS default stretch OFF."""
        if not self.current_ref_layer or not self.current_ref_layer.isValid():
            return
        if state == Qt.Checked:
            self.normalize_layer(self.current_ref_layer)
        else:
            self.reset_normalization(self.current_ref_layer)
        self.canvas_right.refresh()

    # ── REFERENCE TIF LOADER ─────────────────────────────────────────────────
    def _load_ref_layer(self, tif_path, set_extent=True):
        """Single entry point for all reference TIF loads."""
        self.ensure_overviews(tif_path)
        self.canvas_right.setRenderFlag(False)
        try:
            lyr = QgsRasterLayer(tif_path, "Reference_TIF")
            if not lyr.isValid():
                print(f"[LOAD] Failed: {tif_path}")
                self.canvas_right.setLayers([])
                return None
            QgsProject.instance().addMapLayer(lyr, False)
            self.current_ref_layer = lyr
            self.canvas_right.setLayers([lyr])

            # only touch extent if caller really wants it
            if set_extent:
                self.canvas_right.setExtent(lyr.extent())

            if self.cb_normalize.isChecked():
                self.normalize_layer(lyr)
            else:
                self.reset_normalization(lyr)
            return lyr
        except Exception as e:
            print(f"[LOAD] Error: {e}")
            self.canvas_right.setLayers([])
            return None
        finally:
            self.canvas_right.setRenderFlag(True)
            self.canvas_right.refresh()

    # ── REFERENCE FOLDER ─────────────────────────────────────────────────────
    def select_reference_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Reference Folder")
        if not folder_path:
            return
        self.ref_folder_path   = folder_path
        self.ref_meta_data     = {}
        self.ref_meta_data_utm = {}
        errors = []
        try:
            files = os.listdir(folder_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read folder: {e}")
            return
        transform_wgs_to_utm = QgsCoordinateTransform(
            self.wgs84_crs, self.utm44n_crs, QgsProject.instance()
        )
        for meta_file in [f for f in files if f.endswith("_meta.txt")]:
            try:
                meta_path = os.path.join(folder_path, meta_file)
                corners   = self.parse_meta_file(meta_path)
                if not corners:
                    errors.append(f"{meta_file}: Could not parse corners")
                    continue
                base       = meta_file.replace("_meta.txt", "")
                candidates = [base + ".tif", base + ".TIF"]
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = re.search(r"Files\s+(\S+\.tif)", f.read(), re.IGNORECASE)
                    if m:
                        candidates.append(m.group(1))
                tif_path = next(
                    (os.path.join(folder_path, c) for c in candidates
                     if os.path.exists(os.path.join(folder_path, c))), None
                )
                if not tif_path:
                    errors.append(f"{meta_file}: TIF not found")
                    continue
                self.ref_meta_data[tif_path] = corners
                utm_corners = {}
                for k, v in corners.items():
                    p = transform_wgs_to_utm.transform(QgsPointXY(*v))
                    utm_corners[k] = (p.x(), p.y())
                self.ref_meta_data_utm[tif_path] = utm_corners
            except Exception as e:
                errors.append(f"{meta_file}: {e}")

        if errors:
            txt = "\n".join(errors[:10])
            if len(errors) > 10:
                txt += f"\n\n...and {len(errors) - 10} more"
            QMessageBox.warning(self, "Parsing Warnings", txt)
        self.filter_reference_tifs()

    # ── PARSE META FILE ───────────────────────────────────────────────────────
    def parse_meta_file(self, meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                content = f.read()
            corners   = {}
            label_map = {
                "Upper Left":  "UL", "Upper Right": "UR",
                "Lower Left":  "LL", "Lower Right": "LR"
            }
            for raw_line in content.splitlines():
                line = raw_line.strip()
                for label, key in label_map.items():
                    if label in line and "(" in line and "," in line:
                        try:
                            lon = float(line.split("(")[1].split(",")[0].strip())
                            lat = float(line.split("(")[1].split(",")[1]
                                        .strip().split(")")[0])
                            corners[key] = (lon, lat)
                        except (IndexError, ValueError) as e:
                            print(f"[META] {label} in {meta_path}: {e}")
                        break
            if len(corners) == 4:
                return corners
            missing = {"UL", "UR", "LL", "LR"} - set(corners.keys())
            print(f"[META] {os.path.basename(meta_path)}: missing {missing}")
            return None
        except Exception as e:
            print(f"[META] Error {meta_path}: {e}")
            return None

    # ── STAGE 1 FILTER ────────────────────────────────────────────────────────
    def filter_reference_tifs(self):
        self.ref_tif_list = []
        self.dropdown_ref.blockSignals(True)
        self.dropdown_ref.clear()
        self.dropdown_ref.blockSignals(False)
        if not self.ref_meta_data:
            self.dropdown_container.hide()
            return
        if not self.input_tif_layer or not self.input_tif_layer.isValid():
            self.ref_tif_list = list(self.ref_meta_data.keys())
        else:
            try:
                tf = QgsCoordinateTransform(
                    self.input_tif_layer.crs(), self.wgs84_crs, QgsProject.instance()
                )
                input_geom = QgsGeometry.fromRect(
                    tf.transformBoundingBox(self.input_tif_layer.extent())
                )
                for tif_path, corners in self.ref_meta_data.items():
                    try:
                        ul, ur, lr, ll = (corners["UL"], corners["UR"],
                                          corners["LR"], corners["LL"])
                        wkt = (f"POLYGON(({ul[0]} {ul[1]},{ur[0]} {ur[1]},"
                               f"{lr[0]} {lr[1]},{ll[0]} {ll[1]},"
                               f"{ul[0]} {ul[1]}))")
                        if QgsGeometry.fromWkt(wkt).intersects(input_geom):
                            self.ref_tif_list.append(tif_path)
                    except Exception as e:
                        print(f"[FILTER] {tif_path}: {e}")
            except Exception as e:
                print(f"[FILTER] Error: {e}")
                self.ref_tif_list = list(self.ref_meta_data.keys())

        if self.ref_tif_list:
            self.dropdown_ref.blockSignals(True)
            for p in self.ref_tif_list:
                self.dropdown_ref.addItem(os.path.basename(p))
            self.dropdown_ref.blockSignals(False)
            self.dropdown_container.show()
            self.dropdown_container.raise_()
            self.dropdown_ref.setCurrentIndex(0)
        else:
            self.dropdown_ref.blockSignals(True)
            self.dropdown_ref.addItem("No matches found")
            self.dropdown_ref.blockSignals(False)
            self.dropdown_container.show()
            self.dropdown_container.raise_()
            self.cleanup_reference_layer()
            self.canvas_right.setLayers([])
            self.canvas_right.refresh()

    # ── DROPDOWN LOAD ─────────────────────────────────────────────────────────
    def load_reference_tif_from_dropdown(self, index):
        if index < 0 or index >= len(self.ref_tif_list):
            return
        tif_path = self.ref_tif_list[index]
        if self.markers["right"]:
            sc = self.canvas_right.scene()
            if sc:
                sc.removeItem(self.markers["right"])
            self.markers["right"] = None
        self.cleanup_reference_layer()
        self._load_ref_layer(tif_path)

    def cleanup_reference_layer(self):
        if self.current_ref_layer and self.current_ref_layer.isValid():
            QgsProject.instance().removeMapLayer(self.current_ref_layer.id())
        self.current_ref_layer = None

    # ── STAGE 2 AUTO-SWITCH ───────────────────────────────────────────────────
    def auto_switch_reference_tif(self, row):
        if not self.ref_tif_list:
            return
        try:
            ri = self.table.item(row, 2)
            rj = self.table.item(row, 3)
            if not ri or not rj:
                return
            rx, ry = float(ri.text()), float(rj.text())
            if rx == 0.0 and ry == 0.0:
                self.cleanup_reference_layer()
                self.canvas_right.setLayers([])
                self.canvas_right.refresh()
                return
            pt           = QgsGeometry.fromPointXY(QgsPointXY(rx, ry))
            best, best_a = None, float("inf")
            for tif_path in self.ref_tif_list:
                if tif_path not in self.ref_meta_data_utm:
                    continue
                c  = self.ref_meta_data_utm[tif_path]
                ul, ur, lr, ll = c["UL"], c["UR"], c["LR"], c["LL"]
                wkt = (f"POLYGON(({ul[0]} {ul[1]},{ur[0]} {ur[1]},"
                       f"{lr[0]} {lr[1]},{ll[0]} {ll[1]},{ul[0]} {ul[1]}))")
                geom = QgsGeometry.fromWkt(wkt)
                if geom.contains(pt):
                    a = geom.area()
                    if a < best_a:
                        best_a, best = a, tif_path
            if best:
                if (self.current_ref_layer and
                        self.current_ref_layer.isValid() and
                        os.path.normpath(self.current_ref_layer.source())
                        == os.path.normpath(best)):
                    return
                idx = self.ref_tif_list.index(best)
                try:
                    self.dropdown_ref.blockSignals(True)
                    self.dropdown_ref.setCurrentIndex(idx)
                finally:
                    self.dropdown_ref.blockSignals(False)
                self.cleanup_reference_layer()
                self._load_ref_layer(best, set_extent=False)
            else:
                self.cleanup_reference_layer()
                self.canvas_right.setLayers([])
                self.canvas_right.refresh()
        except Exception as e:
            print(f"[AUTO-SWITCH] {e}")

    # ── MARKERS & SYNC ────────────────────────────────────────────────────────
    def handle_manual_typing(self, item):
        if item.column() < 4:
            self.calculate_error(item.row())
            text = item.text().strip()
            try:
                float(text)
                if text and not text.endswith(".") and not text.endswith("-"):
                    self.sync_view_to_row()
            except ValueError:
                pass

    def draw_marker(self, point, canvas, color):
        key = "left" if canvas == self.canvas_left else "right"
        if self.markers[key]:
            sc = canvas.scene()
            if sc:
                sc.removeItem(self.markers[key])
        m = QgsVertexMarker(canvas)
        m.setCenter(point)
        m.setIconType(QgsVertexMarker.ICON_CROSS)
        m.setColor(color)
        m.setPenWidth(1)
        m.setIconSize(20)
        self.markers[key] = m
        canvas.refresh()

    def sync_view_to_row(self):
        """Snap both canvases to selected row — left and right at fixed per-canvas zoom."""
        row = self.table.currentRow()
        if row < 0:
            return
        self.clear_markers()

        # enforce fixed zoom on left/input
        

        sync_was = self.cb_sync.isChecked()
        if sync_was:
            self.cb_sync.setChecked(False)
        try:
            self.auto_switch_reference_tif(row)

            def gc(r, c):
                it = self.table.item(r, c)
                try:
                    return float(it.text()) if it and it.text() else 0.0
                except (ValueError, AttributeError):
                    return 0.0

            ix, iy = gc(row, 0), gc(row, 1)
            rx, ry = gc(row, 2), gc(row, 3)

            if (ix != 0.0) or (iy != 0.0):
                p = QgsPointXY(ix, iy)
                self.draw_marker(p, self.canvas_left, Qt.red)
                canvas_size = self.canvas_left.size()
                aspect = canvas_size.width() / max(canvas_size.height(), 1)
                
                width_m = left_view_width
                height_m = width_m / aspect
                rect = QgsRectangle(
                ix - width_m / 2,
                iy - height_m / 2,
                ix + width_m / 2,
                iy + height_m / 2)
                
                
                self.canvas_left.setExtent(rect)
                self.canvas_left.refresh()
                
                
                
                
                
                

            if rx != 0.0:
                p_wgs = self.transform_utm_to_wgs.transform(QgsPointXY(rx, ry))
                self.canvas_right.setCenter(p_wgs)
                self.canvas_right.zoomScale(ZOOM_REF_PLACEHOLDER)
                self.draw_marker(p_wgs, self.canvas_right, Qt.green)

        except Exception as e:
            print(f"[SYNC ROW] {e}")
        finally:
            if sync_was:
                self.cb_sync.setChecked(True)

    def sync_canvas_extents(self):
        """Pan left (UTM) -> transform centre to WGS84 -> pan right at fixed ref zoom."""
        if not self.cb_sync.isChecked() or self._syncing:
            return
        self._syncing = True
        try:
            left_center_utm  = self.canvas_left.center()
            right_center_wgs = self.transform_utm_to_wgs.transform(left_center_utm)
            self.canvas_right.setCenter(right_center_wgs)
            self.canvas_right.zoomScale(ZOOM_REF_PLACEHOLDER)
            self.canvas_right.refresh()
        except Exception as e:
            print(f"[SYNC EXTENTS] {e}")
        finally:
            self._syncing = False

    def clear_markers(self):
        for key, canvas in [("left", self.canvas_left), ("right", self.canvas_right)]:
            if self.markers[key]:
                sc = canvas.scene()
                if sc:
                    sc.removeItem(self.markers[key])
                self.markers[key] = None
        self.canvas_left.refresh()
        self.canvas_right.refresh()

    # ── ERROR & STATS ─────────────────────────────────────────────────────────
    def calculate_error(self, row):
        """DX = In_X - Ref_X,  DY = In_Y - Ref_Y  (UTM metres)."""
        try:
            self.table.blockSignals(True)
            ix = float(self.table.item(row, 0).text())
            iy = float(self.table.item(row, 1).text())
            rx = float(self.table.item(row, 2).text())
            ry = float(self.table.item(row, 3).text())
            self.table.setItem(row, 4, QTableWidgetItem(f"{ix - rx:.3f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{iy - ry:.3f}"))
        except Exception:
            pass
        finally:
            self.table.blockSignals(False)
        self.update_stats()

    def update_stats(self):
        err_x_list, err_y_list = [], []
        for r in range(self.table.rowCount()):
            try:
                item_x = self.table.item(r, 4)
                item_y = self.table.item(r, 5)
                if item_x and item_y:
                    err_x_list.append(float(item_x.text()))
                    err_y_list.append(float(item_y.text()))
            except (ValueError, AttributeError):
                continue
        if not err_x_list:
            self.lbl_rmse_x.setText("RMSE X:  0.000 m")
            self.lbl_rmse_y.setText("RMSE Y:  0.000 m")
            self.lbl_ce90.setText("CE90:    0.000 m")
            return
        ex_arr = np.array(err_x_list)
        ey_arr = np.array(err_y_list)
        rmse_x = np.sqrt(np.mean(ex_arr ** 2))
        rmse_y = np.sqrt(np.mean(ey_arr ** 2))
        ce90   = np.sqrt(np.sum(ex_arr ** 2) + np.sum(ey_arr ** 2))
        self.lbl_rmse_x.setText(f"RMSE X:  {rmse_x:.3f} m")
        self.lbl_rmse_y.setText(f"RMSE Y:  {rmse_y:.3f} m")
        self.lbl_ce90.setText(f"CE90:    {ce90:.3f} m")

    # ── CSV ───────────────────────────────────────────────────────────────────
    def load_csv_smart(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open QC CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.table.setRowCount(0)
        mapping = {
            "ix": ["In X", "In_X", "x1-map"],
            "iy": ["In Y", "In_Y", "y1-map"],
            "rx": ["Ref X", "Ref_X", "x2-map"],
            "ry": ["Ref Y", "Ref_Y", "y2-map"],
        }
        with open(path, "r", encoding="utf-8-sig") as f:
            for row_data in csv.DictReader(f):
                r = self.table.rowCount()
                try:
                    self.table.blockSignals(True)
                    self.table.insertRow(r)
                    for i, key in enumerate(["ix", "iy", "rx", "ry"]):
                        val = next(
                            (row_data[k] for k in mapping[key] if k in row_data), "0.000"
                        )
                        self.table.setItem(r, i, QTableWidgetItem(val))
                finally:
                    self.table.blockSignals(False)
                self.calculate_error(r)
        self.update_stats()

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["In_X", "In_Y", "Ref_X", "Ref_Y", "DX_Err", "DY_Err"])
            for r in range(self.table.rowCount()):
                w.writerow([
                    self.table.item(r, c).text() if self.table.item(r, c) else "0.000"
                    for c in range(6)
                ])

    def add_manual_row(self):
        r = self.table.rowCount()
        try:
            self.table.blockSignals(True)
            self.table.insertRow(r)
            for i in range(6):
                self.table.setItem(r, i, QTableWidgetItem("0.000"))
        finally:
            self.table.blockSignals(False)
        self.table.setCurrentCell(r, 0)
        self.update_stats()

    def delete_row(self):
        for row in sorted(
            {i.row() for i in self.table.selectedIndexes()}, reverse=True
        ):
            self.table.removeRow(row)
        self.update_stats()

    # ── TIF LOADING & INIT ────────────────────────────────────────────────────
    def manual_load_tif(self, is_input):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TIF", "", "TIF (*.tif *.tiff *.TIF *.TIFF *.vrt)"
        )
        if not path:
            return
        lyr = QgsRasterLayer(path, "Input_TIF" if is_input else "Reference_TIF")
        if not lyr.isValid():
            return
        if is_input:
            if self.input_tif_layer and self.input_tif_layer.isValid():
                QgsProject.instance().removeMapLayer(self.input_tif_layer.id())
            self.input_tif_layer = None
            QgsProject.instance().addMapLayer(lyr, False)
            self.input_tif_layer = lyr
            self.canvas_left.setLayers([lyr])
            self.canvas_left.setExtent(lyr.extent())
            self.canvas_left.refresh()
            if self.ref_folder_path:
                self.filter_reference_tifs()
        else:
            self.cleanup_reference_layer()
            self._load_ref_layer(path)

    def auto_connect_layers(self):
        layers = QgsProject.instance().mapLayers().values()
        in_l  = [l for l in layers if "input" in l.name().lower()]
        ref_l = [l for l in layers if "ref"   in l.name().lower()]
        if in_l:
            self.input_tif_layer = in_l[0]
            self.canvas_left.setLayers([in_l[0]])
            self.canvas_left.setExtent(in_l[0].extent())
            self.canvas_left.refresh()
        if ref_l:
            self.ensure_overviews(ref_l[0].source())
            self.current_ref_layer = ref_l[0]
            self.canvas_right.setLayers([ref_l[0]])
            self.canvas_right.setExtent(ref_l[0].extent())
            self.reset_normalization(ref_l[0])
            self.canvas_right.refresh()

    def init_map_tools(self):
        self.tool_left  = DragMapTool(self.canvas_left,  self, True)
        self.tool_right = DragMapTool(self.canvas_right, self, False)
        self.canvas_left.setMapTool(self.tool_left)
        self.canvas_right.setMapTool(self.tool_right)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
app = QApplication.instance() or QApplication(sys.argv)
win = QCDashboard()
win.show()
