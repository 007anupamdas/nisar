"""RIVAL/AUTO - the RIVAL dashboard with footprints read from the rasters.

RIVAL learns where each reference tile sits from metadata beside it: a sidecar
per raster, a shapefile indexing the set, or a degree-tile name. That covers the
collections it was built against, and fails on the ordinary case of a folder of
georeferenced TIFs with nothing beside them.

This variant drops all three modes. A georeferenced raster already states where
it is -- CRS, affine transform, pixel dimensions -- so the footprint is read
from the file. Nothing else is needed and nothing else is looked for.

It is the same dashboard otherwise: the class is RIVAL's, with the folder scan
and the input footprint replaced. Fixing anything in DPQED_rival.py fixes it
here too; there is no second copy of the GUI.

## The bounding box would be wrong, so it is not used

RIVAL reads sidecars for a reason worth restating, because this tool has to
answer it. A NISAR frame is a slanted swath stored in a north-up grid, so its
extent claims a great deal of ground the scene has no data over -- measured on
the two granules in this branch, 47% and 76% more than the swath itself. Offer
tiles against that box and you get offered tiles whose nodata wedge sits under
the point you wanted to measure.

The raster answers this itself, without metadata: the nodata mask *is* the data
footprint. The mask is read at a decimated resolution, reduced to the first and
last valid pixel of each row (the only ones a convex hull can use), and hulled.
For a swath -- convex by construction -- that recovers the real quadrilateral.

So each footprint is derived one of two ways, and which one is always printed:

  mask     the valid-data hull. Used whenever the raster declares nodata or
           carries a mask band, and some but not all of it is valid.
  extent   the full pixel grid. Used when everything is valid (then the two
           agree anyway) or when no mask is declared, which is the honest
           answer: an undeclared fill value is not discoverable.

The over-coverage of the box against the hull is printed per raster, so a
collection whose fill is undeclared announces itself as a row of 1.00x while
the swaths around it read 1.5x-2x.

Fill that is written as plain zeros with nodata left unset cannot be told from
valid data -- 0 is a real amplitude, and a real dB value. TREAT_ZERO_AS_NODATA
opts into treating it as fill for a collection known to be written that way. It
is off by default because getting it wrong silently eats real data.

## Edges are densified before reprojection

A straight edge in UTM is a curve in lon/lat, so a four-corner ring transformed
vertex by vertex cuts the corner -- hundreds of metres on a full frame. Every
ring is densified to EDGE_DENSIFY points per edge before transforming.

## Which library opens the rasters

QGIS ships GDAL's Python bindings and does **not** ship rasterio, so importing
rasterio in the QGIS console fails outright. Both are wrapped to the same few
questions: GDAL is tried first, rasterio second, and the answers are identical
either way. The one real trap is that the two order their geotransform
differently -- see affine_from_gdal -- which misplaces a footprint silently
rather than raising, so it is pinned by a test against the same fixture read
both ways.

The GUI half needs QGIS, as RIVAL does. tests_rival_auto.py covers the geometry
and both backends without QGIS.
"""

import os
import sys

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# Long side of the decimated mask read. The hull of a swath is set by its
# corners, so this only has to resolve the corner well: at 512 the worst-case
# placement error is one probe pixel, ~40 m on a 20 000 px frame, against the
# kilometres of over-coverage it removes. Raising it costs a bigger read.
MASK_PROBE_PX = 512

# Points per edge when a ring is densified before reprojection. A UTM edge is a
# curve in lon/lat; 8 leaves sub-metre sagitta on a 200 km edge.
EDGE_DENSIFY = 8

# Derive the footprint from the valid-data mask, not just the pixel grid.
FOOTPRINT_FROM_MASK = True

# Treat exact zeros as fill as well as the declared mask. Off: 0 is a valid
# amplitude and a valid dB value, so this eats real data when it is wrong.
TREAT_ZERO_AS_NODATA = False

# A hull that covers this little of its own bounding box is suspicious enough to
# print loudly -- usually a raster whose mask is mostly empty.
HULL_COVERAGE_WARN = 0.05

# Budget for probing a mask at full resolution, in megapixels.
#
# This is the difference between this tool's folder scan and RIVAL's. RIVAL
# reads a few kB of sidecar per tile; reading the mask instead means touching
# the imagery, and a nodata mask is computed FROM the pixels, so there is no
# cheap corner to read. A folder of large scenes on a network share is then
# minutes of I/O on the GUI thread, which presents as QGIS not responding.
#
# So the mask is only probed when it is cheap: through an overview pyramid at
# any size, or at full resolution below this budget. Above it, with no
# overviews, the pixel grid is used and the reason is printed -- a reference
# ortho is north-up and full anyway, so the two agree; it is slanted swaths
# that need the mask, and those are worth building overviews for.
MASK_PROBE_MAX_MPIX = 64.0

# Which library opens the rasters. QGIS ships GDAL's Python bindings and does
# NOT ship rasterio, so "auto" tries osgeo first and only falls back to rasterio
# for a plain Python environment (which is where the tests run). Force one with
# "gdal" or "rasterio" to find out which is in play.
RASTER_BACKEND = "auto"

RIVAL_FILE = "DPQED_rival.py"
RIVAL_ENTRY_MARKER = "# ── ENTRY POINT"


# ── BEGIN PURE HELPERS ────────────────────────────────────────────────────────
def apply_affine(t, col, row):
    """Map coordinate of a pixel *corner*, for rasterio's affine tuple order.

    t is (a, b, c, d, e, f) as rasterio's Affine iterates: x = a*col + b*row + c.
    Corners, not centres: (0, 0) is the outer corner of the first pixel, which
    is what a footprint ring wants.
    """
    a, b, c, d, e, f = t
    return (a * col + b * row + c, d * col + e * row + f)


def affine_from_gdal(gt):
    """GDAL's GetGeoTransform tuple in rasterio's Affine order.

    The two libraries agree on the arithmetic and disagree on the order, which
    is the sort of difference that produces a footprint in the wrong place with
    no error anywhere. GDAL gives (originX, pixelW, rowRot, originY, colRot,
    pixelH) and computes x = gt[0] + col*gt[1] + row*gt[2]; Affine iterates
    (a, b, c, d, e, f) and computes x = a*col + b*row + c. So the origins move
    from the front of each triple to the back.
    """
    ox, px, rx, oy, ry, py = gt
    return (px, rx, ox, ry, py, oy)


def grid_corner_ring(width, height, t):
    """The four outer corners of the pixel grid, clockwise from the origin."""
    return [apply_affine(t, c, r) for c, r in
            ((0, 0), (width, 0), (width, height), (0, height))]


def convex_hull(points):
    """Monotone-chain hull. Fewer than three distinct points come back as given."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def row_spans(mask):
    """(row, first_valid_col, last_valid_col) for each row holding valid data.

    Only the ends of a row matter: every other valid pixel in that row lies
    between them, so it cannot be a hull vertex. This is what keeps the hull
    over a 512x512 probe a few thousand points rather than a quarter of a
    million, and it is exact rather than an approximation.
    """
    spans = []
    for r, row in enumerate(mask):
        first = last = None
        for c, v in enumerate(row):
            if v:
                if first is None:
                    first = c
                last = c
        if first is not None:
            spans.append((r, first, last))
    return spans


def span_corner_points(spans, sx, sy):
    """Pixel-corner points, in source pixels, bounding each row span.

    A probe pixel covers sx by sy source pixels, so its outer corners are at
    col*sx and (col+1)*sx. Taking the outer corners keeps the hull a cover of
    the valid area rather than a crop of it -- the safe direction to err when
    the answer decides which tiles get offered.
    """
    pts = []
    for r, c0, c1 in spans:
        y0, y1 = r * sy, (r + 1) * sy
        x0, x1 = c0 * sx, (c1 + 1) * sx
        pts.extend([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return pts


def ring_area(ring):
    """Absolute shoelace area, in whatever units the ring is in."""
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def densify_ring(ring, per_edge=EDGE_DENSIFY):
    """Insert points along each edge, so reprojection cannot cut the corner."""
    if len(ring) < 2 or per_edge < 2:
        return list(ring)
    out = []
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        for k in range(per_edge):
            f = k / float(per_edge)
            out.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
    return out


def mask_hull_ring(mask, t, sx, sy):
    """Map-coordinate hull of the valid pixels in a decimated mask.

    Returns None when nothing is valid -- a caller cannot build a footprint from
    that and should say so rather than fall back silently.
    """
    spans = row_spans(mask)
    if not spans:
        return None
    px = span_corner_points(spans, sx, sy)
    hull_px = convex_hull(px)
    if len(hull_px) < 3:
        return None
    return [apply_affine(t, c, r) for c, r in hull_px]


def probe_decimation(width, height, probe_px=MASK_PROBE_PX):
    """(out_width, out_height) for a mask read whose long side is <= probe_px."""
    if width <= 0 or height <= 0:
        return 0, 0
    longest = max(width, height)
    if longest <= probe_px:
        return width, height
    scale = probe_px / float(longest)
    return max(1, int(width * scale)), max(1, int(height * scale))


def over_coverage(box_ring, data_ring):
    """How much more ground the bounding box claims than the data hull does."""
    data = ring_area(data_ring)
    if data <= 0:
        return None
    return ring_area(box_ring) / data
# ── END PURE HELPERS ──────────────────────────────────────────────────────────


# ── OPENING A RASTER: GDAL IN QGIS, RASTERIO OUTSIDE IT ───────────────────────
class RasterSource:
    """The little a footprint needs from a raster, from either backend.

    QGIS ships GDAL's Python bindings and does not ship rasterio; a plain
    Python environment usually has it the other way round. Rather than pick one
    and be unusable in half the places this runs, both are wrapped to the same
    handful of questions and the answers are identical either way.
    """

    def __init__(self, backend, width, height, transform, crs_wkt,
                 all_valid, has_overviews, read_mask, read_band):
        self.backend = backend
        self.width = width
        self.height = height
        self.transform = transform
        self.crs_wkt = crs_wkt
        self.all_valid = all_valid        # no nodata declared: every pixel data
        self.has_overviews = has_overviews  # a pyramid to read the mask through
        self.read_mask = read_mask        # (out_w, out_h) -> 2-D truthy
        self.read_band = read_band        # (out_w, out_h) -> 2-D values


def open_gdal(path):
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError("GDAL could not open it")
    wkt = ds.GetProjection()
    band = ds.GetRasterBand(1)
    # GMF_ALL_VALID is GDAL saying there is no nodata and no mask band, so the
    # mask would come back solid -- worth knowing before reading it.
    all_valid = bool(band.GetMaskFlags() & gdal.GMF_ALL_VALID)

    def read_mask(out_w, out_h):
        return band.GetMaskBand().ReadAsArray(
            0, 0, ds.RasterXSize, ds.RasterYSize,
            buf_xsize=out_w, buf_ysize=out_h)

    def read_band(out_w, out_h):
        return band.ReadAsArray(0, 0, ds.RasterXSize, ds.RasterYSize,
                                buf_xsize=out_w, buf_ysize=out_h)

    # GDAL serves a decimated read out of the overview pyramid when there is
    # one, which is what makes probing a large scene affordable at all.
    has_ov = band.GetOverviewCount() > 0

    return RasterSource("gdal", ds.RasterXSize, ds.RasterYSize,
                        affine_from_gdal(ds.GetGeoTransform()),
                        wkt or None, all_valid, has_ov, read_mask, read_band)


def open_rasterio(path):
    import rasterio
    from rasterio.enums import Resampling, MaskFlags

    src = rasterio.open(path)
    t = src.transform
    all_valid = list(src.mask_flag_enums[0]) == [MaskFlags.all_valid]

    def read_mask(out_w, out_h):
        return src.read_masks(1, out_shape=(out_h, out_w),
                              resampling=Resampling.nearest)

    def read_band(out_w, out_h):
        return src.read(1, out_shape=(out_h, out_w),
                        resampling=Resampling.nearest)

    return RasterSource("rasterio", src.width, src.height,
                        (t.a, t.b, t.c, t.d, t.e, t.f),
                        src.crs.to_wkt() if src.crs else None,
                        all_valid, bool(src.overviews(1)), read_mask, read_band)


def open_raster(path, backend=None):
    """Open with GDAL if it is there, rasterio if it is not."""
    backend = backend or RASTER_BACKEND
    if backend == "gdal":
        return open_gdal(path)
    if backend == "rasterio":
        return open_rasterio(path)
    try:
        return open_gdal(path)
    except ImportError:
        pass
    try:
        return open_rasterio(path)
    except ImportError:
        raise ValueError(
            "neither osgeo.gdal nor rasterio is importable. Inside QGIS the "
            "GDAL bindings ship with it, so this usually means the script is "
            "being run by a different Python than QGIS's own.")


# ── READING A RASTER'S FOOTPRINT ──────────────────────────────────────────────
def read_raster_footprint(path, probe_px=MASK_PROBE_PX,
                          from_mask=FOOTPRINT_FROM_MASK,
                          zero_is_nodata=TREAT_ZERO_AS_NODATA,
                          max_mpix=MASK_PROBE_MAX_MPIX,
                          backend=None):
    """The raster's own footprint, in its own CRS.

    Returns a dict, or raises ValueError with a reason a user can act on. The
    ring is the valid-data hull where a mask says something useful, and the
    pixel grid otherwise; 'derived' says which, and is not guesswork -- it is
    what was actually used.
    """
    src = open_raster(path, backend)
    if not src.crs_wkt:
        raise ValueError("no CRS: the raster does not say where it is")
    if not src.width or not src.height:
        raise ValueError("zero-sized raster")
    if src.transform[0] == 0 or src.transform[4] == 0:
        raise ValueError("degenerate transform (zero pixel size)")

    box = grid_corner_ring(src.width, src.height, src.transform)
    rec = {"ring_map": box, "box_ring": box, "derived": "extent",
           "crs_wkt": src.crs_wkt, "width": src.width, "height": src.height,
           "over": 1.0, "valid_fraction": 1.0, "note": None,
           "backend": src.backend}

    if not from_mask:
        rec["note"] = "mask reading disabled"
        return rec
    if src.all_valid and not zero_is_nodata:
        # No nodata declared, so the mask is solid and reading it would only
        # confirm the box. Saying 'extent' here is the truth, not a fallback.
        rec["note"] = "every pixel valid (no nodata declared)"
        return rec

    mpix = src.width * src.height / 1e6
    if not src.has_overviews and mpix > max_mpix:
        rec["note"] = (f"{mpix:.0f} Mpix and no overviews: too costly to probe "
                       f"the mask, so the pixel grid is used. Build overviews "
                       f"to get the data footprint")
        return rec

    out_w, out_h = probe_decimation(src.width, src.height, probe_px)
    try:
        mask = src.read_mask(out_w, out_h)
        valid = mask != 0
        if zero_is_nodata:
            valid = valid & (src.read_band(out_w, out_h) != 0)
    except Exception as e:                       # driver without mask support
        rec["note"] = f"mask unreadable ({e})"
        return rec

    n_valid = int(valid.sum())
    rec["valid_fraction"] = n_valid / float(out_w * out_h)
    if n_valid == 0:
        rec["note"] = "no valid pixels in the mask; using the pixel grid"
        return rec
    if n_valid == out_w * out_h:
        rec["note"] = "every pixel valid"
        return rec

    sx = src.width / float(out_w)
    sy = src.height / float(out_h)
    hull = mask_hull_ring(valid.tolist(), src.transform, sx, sy)
    if hull is None:
        rec["note"] = "mask gave no usable hull; using the pixel grid"
        return rec

    rec["ring_map"] = hull
    rec["derived"] = "mask"
    rec["over"] = over_coverage(box, hull) or 1.0
    return rec


def describe_footprint(name, rec):
    """One line per raster, saying where the footprint came from."""
    if rec["derived"] == "mask":
        return (f"[AUTO] {name}: mask hull, {len(rec['ring_map'])} vertices, "
                f"box claims {rec['over']:.2f}x the ground")
    reason = rec.get("note") or "no mask"
    return f"[AUTO] {name}: pixel grid ({reason})"


# ── THE DASHBOARD ─────────────────────────────────────────────────────────────
def load_rival():
    """RIVAL's module body, minus its entry point, in a fresh namespace.

    DPQED_rival.py builds and shows its window at import, so it cannot simply be
    imported -- that would open a second dashboard. Its own test suite already
    slices the file at explicit markers; this cuts at the entry-point banner for
    the same reason, and fails loudly rather than quietly running the wrong half
    if that banner ever moves.
    """
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), RIVAL_FILE)
    try:
        text = open(src, encoding="utf-8").read()
    except OSError as e:
        raise SystemExit(f"{RIVAL_FILE} must sit beside this script: {e}")
    if RIVAL_ENTRY_MARKER not in text:
        raise SystemExit(
            f"{RIVAL_FILE} has no '{RIVAL_ENTRY_MARKER}' banner, so the point "
            f"to stop at cannot be found. Refusing to guess.")
    body = text[:text.index(RIVAL_ENTRY_MARKER)]
    ns = {"__name__": "dpqed_rival", "__file__": src}
    exec(compile(body, src, "exec"), ns)
    return ns


_RIVAL = load_rival()

QCDashboard   = _RIVAL["QCDashboard"]
QgsPointXY    = _RIVAL["QgsPointXY"]
QMessageBox   = _RIVAL["QMessageBox"]
QFileDialog   = _RIVAL["QFileDialog"]
QApplication  = _RIVAL["QApplication"]
QgsCoordinateReferenceSystem = _RIVAL["QgsCoordinateReferenceSystem"]
QgsCoordinateTransform       = _RIVAL["QgsCoordinateTransform"]
QgsProject    = _RIVAL["QgsProject"]
band_from_name = _RIVAL["band_from_name"]
rings_bounds   = _RIVAL["rings_bounds"]
format_bounds  = _RIVAL["format_bounds"]
BAND_UNKNOWN   = _RIVAL["BAND_UNKNOWN"]
RASTER_EXTS    = _RIVAL["RASTER_EXTS"]

REF_MODE_RASTER = "raster"


class AutoFootprintDashboard(QCDashboard):
    """RIVAL, with every footprint read from the raster instead of metadata."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "RIVAL/AUTO - footprints read from the rasters themselves")

    # ── REFERENCE FOLDER ─────────────────────────────────────────────────────
    def select_reference_folder(self):
        """Every raster under the folder, placed by its own georeferencing.

        No mode detection: there is one way to read a footprint here, and a
        raster either carries the georeferencing or it does not.
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Reference Folder")
        if not folder_path:
            return
        self.ref_folder_path = folder_path
        self.ref_footprints  = {}
        self.ref_mode        = None
        self._refresh_band_choices()

        rasters, _metas, _others = self._folder_entries(folder_path)
        if not rasters:
            QMessageBox.critical(
                self, "Error",
                f"No rasters ({', '.join(RASTER_EXTS)}) in {folder_path} "
                f"or its subfolders.")
            self.filter_reference_tifs()
            return

        print(f"[AUTO] {os.path.basename(folder_path)}: {len(rasters)} raster(s), "
              f"footprints from the rasters themselves")
        errors = self._scan_rasters(rasters)

        self.ref_mode = REF_MODE_RASTER
        self._reproject_footprints()
        self._refresh_band_choices()
        self._report_scan(errors)
        self.filter_reference_tifs()

    def _scan_rasters(self, rasters):
        """Read each raster's footprint and file it in RIVAL's own record shape."""
        errors, from_mask, boxed = [], 0, 0
        total = len(rasters)
        # Printed as it goes: this scan touches imagery rather than sidecars, so
        # on a network share it is the slowest thing the tool does and a silent
        # window looks like a hang.
        print(f"[AUTO] reading {total} footprint(s)...")
        for i, (name, path) in enumerate(sorted(rasters.items()), start=1):
            try:
                rec = read_raster_footprint(path)
            except ValueError as e:
                errors.append(f"{name}: {e}")
                continue
            except Exception as e:
                errors.append(f"{name}: unreadable ({e})")
                continue

            print(f"[AUTO] {i}/{total} " + describe_footprint(name, rec)[7:])
            if rec["derived"] == "mask":
                from_mask += 1
            else:
                boxed += 1
            if rec["valid_fraction"] and rec["valid_fraction"] < HULL_COVERAGE_WARN:
                print(f"[AUTO] {name}: only {rec['valid_fraction']:.1%} of the "
                      f"grid is valid -- check the nodata value is right")

            ring = self._ring_to_wgs84(rec["ring_map"], rec["crs_wkt"])
            if not ring:
                errors.append(f"{name}: footprint would not transform to WGS84")
                continue

            self.ref_footprints[path] = {
                "ring": ring,
                # The band tag comes from the granule name, which is part of the
                # file rather than metadata beside it. Unknown is honest for a
                # reference tile that is not a NISAR product at all.
                "band": band_from_name(name) or BAND_UNKNOWN,
                "crs": rec["crs_wkt"],
                "granule": None,
                "source": f"raster ({rec['derived']})",
                "meta": None,
            }

        print(f"[AUTO] {from_mask} from the valid-data mask, "
              f"{boxed} from the pixel grid")
        return errors

    def _ring_to_wgs84(self, ring_map, crs_wkt):
        """Densify, then transform to lon/lat, so the edges keep their shape."""
        try:
            src = QgsCoordinateReferenceSystem.fromWkt(crs_wkt)
            if not src.isValid():
                return None
            tf = QgsCoordinateTransform(src, self.wgs84_crs, QgsProject.instance())
            out = []
            for x, y in densify_ring(ring_map):
                q = tf.transform(QgsPointXY(x, y))
                out.append((q.x(), q.y()))
            return out if len(out) >= 3 else None
        except Exception as e:
            print(f"[AUTO] transform: {e}")
            return None

    def _report_scan(self, errors):
        counts = {}
        for rec in self.ref_footprints.values():
            counts[rec["band"]] = counts.get(rec["band"], 0) + 1
        if counts:
            print("[AUTO] Tagged: "
                  + ", ".join(f"{n} {b}" for b, n in sorted(counts.items())))
        if self.ref_footprints:
            bounds = rings_bounds([r["ring"] for r in self.ref_footprints.values()])
            print(f"[AUTO] {len(self.ref_footprints)} footprint(s) covering "
                  f"{format_bounds(bounds)}")
        if errors:
            txt = "\n".join(errors[:10])
            if len(errors) > 10:
                txt += f"\n\n...and {len(errors) - 10} more"
            QMessageBox.warning(self, "Footprint Warnings", txt)
        elif not self.ref_footprints:
            QMessageBox.warning(
                self, "Reference Folder",
                "No raster in that folder carries usable georeferencing.\n\n"
                "This tool places tiles by their own CRS and transform; a "
                "plain image with no projection cannot be placed.")

    # ── INPUT SCENE ──────────────────────────────────────────────────────────
    def _input_footprint_ring(self, raster_path):
        """The input scene's footprint, from its own mask rather than a sidecar.

        Same reasoning as the reference side: for a slanted swath the mask hull
        is the swath, where the pixel grid is the swath plus its nodata wedges.
        """
        try:
            rec = read_raster_footprint(raster_path)
        except Exception as e:
            print(f"[INPUT] {os.path.basename(raster_path)}: {e}")
            return None
        ring = self._ring_to_wgs84(rec["ring_map"], rec["crs_wkt"])
        if not ring:
            return None
        print(f"[INPUT] footprint from the raster's {rec['derived']}: "
              f"{len(ring)} vertices, {format_bounds(rings_bounds([ring]))}"
              + (f", box claims {rec['over']:.2f}x"
                 if rec["derived"] == "mask" else ""))
        return ring


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
app = QApplication.instance() or QApplication(sys.argv)
win = AutoFootprintDashboard()
win.show()
