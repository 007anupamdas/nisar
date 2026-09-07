"""RIVAL - Reference Image Validation and Accuracy Logger (QGIS).

Two linked canvases: the input raster on the left, a reference tile on the right.
Drag on either to fill the selected row; the table reports DX/DY in the working
CRS's metres, with RMSE and CE90 underneath.

Reference tiles are discovered by whichever means the collection uses to declare
its footprints; the folder is inspected and the mode chosen automatically.

  index-shp    a shapefile indexing the set, one attribute naming each raster
               (L8: Meta/index.shp)
  sidecar      one metadata file per raster (NISAR and S1):
                 <product>.met         JSON        -- Image* swath corners
                 <product>.h5.iso.xml  ISO 19115-2 -- gml:posList footprint
                 <stem>_meta.txt       gdalinfo    -- legacy corner lines
  degree-tile  the name is the footprint (C1: N16E73.tif is 16-17 N, 73-74 E)

An index or sidecar states the real footprint while a tile name only implies a
nominal cell, so the name is the last resort rather than the first guess. Set
REF_MODE_OVERRIDE to force one. Imagery at the top of the folder and metadata in
a 'Meta' subfolder are matched across that split.

NISAR footprints are tagged LSAR or SSAR -- from the '.met' Sensor field, else
the tag in the granule name, else the centre frequency (L ~1.24 GHz, S ~3.2 GHz).
Nothing is guessed: an untaggable footprint is UNK and shows unprefixed, and the
band filter offers only tags the folder actually holds.

Marking on the input canvas drives the reference canvas: the tile covering that
ground is brought up, centred and marked, so the reference is always showing the
place being measured. Clicking the same feature on the right then fills Ref X/Y
and the row's error.

Mark / Pan / Zoom In / Zoom Out (Ctrl+1..4) is one exclusive row applied to both
canvases at once -- leaving one marking while the other is being zoomed only
produces stray picks. Mark is the only tool that fills the table; the rest move
the view. With 'Sync Maps' on the reference follows the input's centre and
scale, so zooming either side keeps both at the same ground width.

The mark is a coloured cross -- red for the input, green for the reference --
over a wider translucent yellow one. The halo is what makes it findable over
bright SAR speckle or a pale ortho, where a thin cross disappears; the
translucency keeps the pixel being measured visible through it.

The arrow keys over a canvas move that side's mark by one source pixel (Shift
for ten), rather than panning the view: while measuring, the thing being
refined is the point. The input's own pixel size is used on the left and the
reference tile's on the right, converted from its CRS, so a step is one pixel of
whatever is under the cursor. Arrow keys in the table still move between cells.

A multi-band input is composed from an R/G/B picker over the left canvas. Every
slot offers every band and a band may be repeated, so a two-band NISAR chip can
be shown any way round; the default puts band 1 in all three, which renders
grey. Bands are listed by the names the raster carries -- cog_locate's --gtiff
writes the polarization into each description, so a chip offers HH and HV rather
than 'Band 1'.

'Normalize NISAR' and 'Normalize Ref' select the same SAR sqrt-gamma stretch for
their canvas, computed by one shared routine so the word means the same thing on
both. Unticked, the input clips each channel at its 2%-98% percentiles. Either
way a plain min/max is avoided: on SAR it is set by a handful of bright
scatterers and renders the scene black.

NISAR rasters are UTM and the C1/L8 references are WGS84. By default both
canvases are pinned to the working CRS and QGIS reprojects the reference as it
draws, so In and Ref are directly comparable and a pick needs no conversion; set
REF_CANVAS_CRS = "wgs84" to draw the reference in its own CRS instead, converting
picks rather than pixels. The working CRS is adopted from the input raster when
that carries a projected one, falling back to WORKING_CRS_DEFAULT -- a fixed zone
is wrong as soon as a scene sits in another, and an LSAR frame is wide enough to
straddle two.

Which references are offered is decided against the input scene's own lon/lat
footprint, read from its sidecar when one sits beside it, rather than its
north-up extent.
"""

import csv
import json
import os
import sys
import re
import threading
import xml.etree.ElementTree as ET
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QFileDialog, QHeaderView, QCheckBox, QComboBox,
                             QMessageBox, QApplication, QShortcut, QLabel,
                             QFrame, QButtonGroup)
from PyQt5.QtCore import Qt, QObject, QEvent
from PyQt5.QtGui import QKeySequence, QFont, QColor
from qgis.gui import (QgsMapCanvas, QgsMapTool, QgsMapToolPan, QgsMapToolZoom,
                      QgsVertexMarker)
from qgis.core import (QgsProject, QgsPointXY, QgsRasterLayer, QgsVectorLayer,
                       QgsGeometry, QgsCoordinateReferenceSystem,
                       QgsCoordinateTransform, QgsSingleBandGrayRenderer,
                       QgsMultiBandColorRenderer, QgsContrastEnhancement,
                       QgsRasterBandStats, QgsRectangle)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# Both views are now set from an explicit ground rectangle -- see
# REF_VIEW_WIDTH_M below for why. The old fixed zoom scales are gone with the
# last zoomScale() call that used them.
left_view_width = 5000       # ground width of the input view, metres
NORM_MIN    = 0           # SAR normalization min DN
NORM_MAX    = 1500        # SAR normalization max DN
NORM_GAMMA  = 0.5         # gamma exponent for sqrt stretch (0.5 = square root)

# Working (projected) CRS the table's In/Ref columns and the error metres live in.
# Adopted from the input TIF when that carries a projected CRS; this is the fallback.
WORKING_CRS_DEFAULT = "EPSG:32644"     # UTM 44N

# Sidecar metadata sitting beside '<product>.h5': SSAR ships '<product>.met',
# a plain text file; LSAR ships '<product>.h5.iso.xml', ISO 19115-2. Both are
# scanned, and each footprint is tagged with its band.
# '_meta.txt' is kept for folders written before the '.met' convention.
META_SUFFIXES_TEXT = (".met", "_meta.txt")
META_SUFFIX_XML    = ".iso.xml"

# NISAR L-band is centred near 1.24 GHz, S-band near 3.2 GHz. Anything below this
# split is LSAR, anything above is SSAR.
BAND_SPLIT_HZ = 2.0e9
BAND_UNKNOWN  = "UNK"

# A reference collection declares its tile footprints in one of three ways. The
# folder is inspected and the mode chosen automatically; set REF_MODE_OVERRIDE to
# one of the REF_MODE_* values below to force it when the guess is wrong.
REF_MODE_SIDECAR = "sidecar"       # one metadata file per raster (NISAR, S1)
REF_MODE_INDEX   = "index-shp"     # one shapefile indexing the whole set (L8)
REF_MODE_TILE    = "degree-tile"   # the footprint is in the name (C1: N16E73)
REF_MODE_OVERRIDE = None

# 'N16E73.tif' is the cell whose SOUTH-WEST corner is 16 N, 73 E, i.e. 16-17 N
# by 73-74 E. Change this if a collection is tiled at another step.
DEGREE_TILE_SIZE = 1.0

RASTER_EXTS = (".tif", ".tiff", ".vrt")

# Subfolder names that hold the metadata rather than the imagery.
META_DIR_NAMES = ("meta", "metadata")

# A reference collection is often filed into subfolders -- by region, by year, or
# imagery above with a Meta/ beside it. The whole tree is walked to this depth,
# stopping at the file cap so pointing the picker at a huge drive cannot hang.
REF_SCAN_DEPTH    = 4
REF_SCAN_MAX_FILES = 50000

# Which CRS the reference canvas draws in. "working" matches the input canvas,
# so In and Ref are directly comparable and a pick needs no conversion, at the
# cost of QGIS reprojecting the WGS84 reference as it draws. Switch to "wgs84"
# if that reprojection is slow on a given machine -- picks are then converted
# instead, and every other behaviour is identical.
REF_CANVAS_CRS = "working"      # "working" | "wgs84"

# Ground width of the reference view, in metres. The extent is set from an
# explicit rectangle rather than setCenter + zoomScale: a scale depends on the
# widget's size and DPI and quietly does nothing on a canvas that has not been
# laid out yet, which leaves the view at the origin with the raster off-screen.
REF_VIEW_WIDTH_M = 2000.0

# Map tools offered on both canvases, in button order. "mark" is the measuring
# tool -- it is what fills the table; the rest only move the view.
TOOL_MARK     = "mark"
TOOL_PAN      = "pan"
TOOL_ZOOM_IN  = "zoom in"
TOOL_ZOOM_OUT = "zoom out"
MAP_TOOLS = (TOOL_MARK, TOOL_PAN, TOOL_ZOOM_IN, TOOL_ZOOM_OUT)

# Arrow keys nudge the marked point by this many source pixels; Shift multiplies
# it, for closing a gap of tens of pixels without holding the key down.
NUDGE_PIXELS       = 1
NUDGE_SHIFT_FACTOR = 10

# The mark is a coloured cross over a wider translucent yellow one. The halo is
# what makes it findable over bright SAR speckle or a pale ortho, where a thin
# red or green cross disappears; the translucency keeps the pixel being measured
# visible through it.
MARKER_SIZE           = 20
MARKER_PEN_WIDTH      = 1
MARKER_OUTLINE_RGBA   = (255, 255, 0, 110)     # yellow, ~43% opaque
MARKER_OUTLINE_WIDTH  = 4
MARKER_OUTLINE_EXTRA  = 2                      # px wider than the cross itself

# Percentile clip for the input composite. A min/max stretch on SAR is dominated
# by a handful of bright scatterers and leaves the scene black.
RGB_CLIP_LOW  = 0.02
RGB_CLIP_HIGH = 0.98

# Pixels sampled when working out a stretch. Unbounded, QGIS reads the WHOLE
# raster at full resolution to build the histogram -- per band, on the GUI
# thread -- which on a large COG is minutes of a frozen window. This is the
# same figure QGIS itself uses for its estimated min/max.
RASTER_SAMPLE_SIZE = 250000

# Which band each channel starts on. All three on band 1 renders grey, so the
# scene is legible before any choice is made; NISAR carries HH and HV, and which
# pairing is useful depends on the scene.
RGB_DEFAULT_BANDS = (1, 1, 1)


# ── BEGIN PURE HELPERS ────────────────────────────────────────────────────────
# Everything between these markers is plain Python -- no Qt, no QGIS -- so the
# metadata parsing can be exercised without a QGIS session. tests_rival_meta.py
# execs exactly this slice.

_ISO_NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gml": "http://www.opengis.net/gml/3.2",
    "gmx": "http://www.isotc211.org/2005/gmx",
    "eos": "http://earthdata.nasa.gov/schema/eos",
    "gmi": "http://www.isotc211.org/2005/gmi",
}


def band_from_name(name):
    """LSAR / SSAR spelled out in a file or granule name, if it is there."""
    upper = (name or "").upper()
    for tag in ("LSAR", "SSAR"):
        if tag in upper:
            return tag
    return None


def band_from_frequency(hz):
    """NISAR centre frequency -> band tag. L-band ~1.24 GHz, S-band ~3.2 GHz."""
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return None
    if hz <= 0:
        return None
    return "LSAR" if hz < BAND_SPLIT_HZ else "SSAR"


def close_ring(ring):
    """Return the ring with its first vertex repeated at the end."""
    if len(ring) >= 3 and ring[0] != ring[-1]:
        return list(ring) + [ring[0]]
    return list(ring)


def ring_wkt(ring):
    """POLYGON WKT for an ordered vertex list. Returns None if under 3 vertices."""
    if not ring or len(ring) < 3:
        return None
    closed = close_ring(ring)
    pts = ",".join(f"{x} {y}" for x, y in closed)
    return f"POLYGON(({pts}))"


def rings_bounds(rings):
    """(min_lon, min_lat, max_lon, max_lat) over several rings, or None."""
    xs = [x for ring in rings for x, _ in ring]
    ys = [y for ring in rings for _, y in ring]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def format_bounds(b):
    """A lon/lat box as something readable in a dialog."""
    if not b:
        return "empty"
    return (f"{b[1]:.3f}..{b[3]:.3f} lat, {b[0]:.3f}..{b[2]:.3f} lon")


def parse_pos_list(text):
    """gml:posList -> [(lon, lat), ...].

    NISAR writes comma-separated 'lon lat height' triples; the GML default is one
    whitespace-separated run. Both are accepted, and a 2-value (lon lat) run too.
    """
    if not text:
        return []
    chunks = [c for c in text.replace("\n", " ").split(",") if c.strip()]
    if len(chunks) > 1:
        ring = []
        for chunk in chunks:
            parts = chunk.split()
            if len(parts) >= 2:
                ring.append((float(parts[0]), float(parts[1])))
        return ring
    parts = text.split()
    for step in (3, 2):
        if len(parts) >= 3 * step and len(parts) % step == 0:
            return [(float(parts[i]), float(parts[i + 1]))
                    for i in range(0, len(parts), step)]
    return []


def parse_meta_json(obj, source="<met>"):
    """SSAR '.met' JSON -> footprint record, or None.

    Two corner sets are given. Image* is the real slanted swath; Prod* is the
    north-up product grid the raster spans, which includes the nodata wedges
    either side. Image* is preferred for the same reason the ISO ring is
    preferred over a bounding box -- it does not claim ground the scene has no
    data over. Prod* is the fallback.
    """
    def ring_for(prefix):
        out = []
        for corner in ("UL", "UR", "LR", "LL"):
            lat = obj.get(f"{prefix}{corner}Lat")
            lon = obj.get(f"{prefix}{corner}Lon")
            if lat is None or lon is None:
                return None
            try:
                out.append((float(lon), float(lat)))
            except (TypeError, ValueError):
                return None
        return out

    ring = ring_for("Image")
    extent = "image"
    if ring is None:
        ring, extent = ring_for("Prod"), "product-grid"
    if ring is None:
        print(f"[META] {source}: no Image*/Prod* corner set")
        return None

    band = band_from_name(str(obj.get("Sensor", ""))) or band_from_name(source)
    if band is None:
        band = band_from_name(str(obj.get("OTSProductID", "")))

    crs = None
    epsg = obj.get("EPSG")
    if epsg not in (None, ""):
        try:
            crs = f"EPSG:{int(epsg)}"
        except (TypeError, ValueError):
            crs = None

    # OTSProductID is the bare product id, no extension. Leave it that way: what
    # sits in the reference folder is '<product>.tif', not the '.h5', and
    # match_raster only ever looks for rasters.
    return {
        "ring": ring,
        "band": band or BAND_UNKNOWN,
        "crs": crs,
        "granule": obj.get("OTSProductID") or None,
        "source": f"met-json ({extent})",
    }


def parse_meta_text(content, source="<text>"):
    """Text sidecar -> footprint record, or None.

    SSAR '.met' is JSON; the older '_meta.txt' is gdalinfo output, whose corner
    lines look like  Upper Left  ( 78.0312500,  17.1234500).
    """
    stripped = content.lstrip()
    if stripped.startswith("{"):
        try:
            return parse_meta_json(json.loads(stripped), source)
        except (ValueError, AttributeError) as e:
            print(f"[META] {source}: JSON parse error: {e}")
            return None

    corners = {}
    label_map = {
        "Upper Left":  "UL", "Upper Right": "UR",
        "Lower Left":  "LL", "Lower Right": "LR",
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
                    print(f"[META] {label} in {source}: {e}")
                break
    if len(corners) != 4:
        missing = {"UL", "UR", "LL", "LR"} - set(corners.keys())
        print(f"[META] {source}: missing {sorted(missing)}")
        return None

    band = band_from_name(content)
    if band is None:
        m = re.search(r"(?:centre|center)\s*frequency\s*[:=]?\s*"
                      r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)",
                      content, re.IGNORECASE)
        if m:
            band = band_from_frequency(m.group(1))
    if band is None:
        m = re.search(r"\bband\s*[:=]\s*([LS])\b", content, re.IGNORECASE)
        if m:
            band = m.group(1).upper() + "SAR"

    m = re.search(r"Files\s*:?\s+(\S+\.(?:tif|tiff|vrt))", content, re.IGNORECASE)
    return {
        "ring": [corners["UL"], corners["UR"], corners["LR"], corners["LL"]],
        "band": band or BAND_UNKNOWN,
        "crs": None,
        "granule": m.group(1) if m else None,
        "source": "text",
    }


def _iso_text(root, path):
    node = root.find(path, _ISO_NS)
    return node.text.strip() if node is not None and node.text else None


def parse_meta_iso_xml(content, source="<xml>"):
    """NISAR ISO 19115-2 '.iso.xml' -> footprint record, or None.

    Takes the full bounding polygon rather than a corner box: a NISAR frame is a
    slanted swath, so its four extreme corners over-cover it badly.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"[META] {source}: XML parse error: {e}")
        return None

    ring = []
    for node in root.iter("{%s}posList" % _ISO_NS["gml"]):
        ring = parse_pos_list(node.text)
        if len(ring) >= 3:
            break
    if len(ring) < 3:
        print(f"[META] {source}: no usable gml:posList")
        return None

    crs = None
    for node in root.iter("{%s}referenceSystemInfo" % _ISO_NS["gmd"]):
        code = _iso_text(node, ".//gmd:code/gco:CharacterString")
        if code:
            m = re.search(r"EPSG:*(\d{4,6})", code)
            crs = f"EPSG:{m.group(1)}" if m else code
            break

    granule = None
    for tag in ("{%s}FileName" % _ISO_NS["gmx"],
                "{%s}CharacterString" % _ISO_NS["gco"]):
        for node in root.iter(tag):
            if node.text and node.text.strip().endswith(".h5"):
                granule = node.text.strip()
                break
        if granule:
            break

    band = band_from_name(granule or "") or band_from_name(source)
    if band is None:
        for node in root.iter("{%s}EOS_AdditionalAttribute" % _ISO_NS["eos"]):
            name = _iso_text(node, ".//eos:name/eos:CharacterString") or ""
            if name.strip().lower() != "center frequency":
                continue
            band = band_from_frequency(
                _iso_text(node, "./eos:value/eos:CharacterString"))
            if band:
                break

    return {
        "ring": ring,
        "band": band or BAND_UNKNOWN,
        "crs": crs,
        "granule": granule,
        "source": "iso-xml",
    }


def meta_base_stem(meta_name):
    """Strip the sidecar suffix, leaving the stem its raster shares.

    '<product>.h5.iso.xml' and '<product>.met' both reduce to '<product>'.
    """
    for suffix in (META_SUFFIX_XML,) + META_SUFFIXES_TEXT:
        if meta_name.lower().endswith(suffix.lower()):
            stem = meta_name[: -len(suffix)]
            break
    else:
        stem = os.path.splitext(meta_name)[0]
    if stem.lower().endswith(".h5"):
        stem = stem[:-3]
    return stem


def match_raster(files, stem, granule=None):
    """Pick the raster a sidecar describes, out of a folder listing.

    Exact stem match first, then the granule name the metadata itself gives, then
    any raster whose name starts with the stem -- DPQED_h52tif writes
    '<product>1.tif', so a plain stem+'.tif' does not always exist.

    Only rasters are ever considered. The metadata names the '.h5' it was written
    from, but what sits in the reference folder is the '.tif', so any '.h5' the
    caller passes in is reduced to its stem rather than looked for.
    """
    exts = (".tif", ".tiff", ".vrt")
    lower = {f.lower(): f for f in files}

    stems = [stem]
    if granule:
        g = os.path.basename(granule)
        if g.lower().endswith(".h5"):
            g = g[:-3]
        stems.append(os.path.splitext(g)[0] if "." in g else g)

    for cand in stems:
        for ext in exts:
            hit = lower.get((cand + ext).lower())
            if hit:
                return hit

    prefixed = sorted(
        f for f in files
        if f.lower().startswith(stem.lower()) and f.lower().endswith(exts)
    )
    return prefixed[0] if prefixed else None


# Degree counts are not reliably zero-padded: 'N8E76_ortho.tif' is as real as
# 'N16E073.tif'. Latitude takes 1-2 digits, longitude 1-3, and the trailing
# lookahead stops a longer run being read as a tile.
_TILE_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})(?![0-9])", re.IGNORECASE)


def parse_degree_tile(name, size=DEGREE_TILE_SIZE):
    """'N16E73.tif' -> the degree cell it names, or None.

    The token gives the cell's south-west corner, so N16E73 spans 16-17 N by
    73-74 E. Padding is not assumed: 'N8E76', 'N16E73' and 'N16E073' all parse.
    NISAR granule names cannot collide: they start 'NISAR', and a digit must
    follow the hemisphere letter.
    """
    m = _TILE_RE.match(os.path.basename(name))
    if not m:
        return None
    ns, lat_s, ew, lon_s = m.groups()
    lat = float(lat_s) * (-1 if ns.upper() == "S" else 1)
    lon = float(lon_s) * (-1 if ew.upper() == "W" else 1)
    if not (-90.0 <= lat <= 90.0 - size) or not (-180.0 <= lon <= 180.0 - size):
        print(f"[META] {name}: tile {lat},{lon} out of range")
        return None
    return {
        "ring": [(lon, lat + size), (lon + size, lat + size),
                 (lon + size, lat), (lon, lat)],
        "band": BAND_UNKNOWN,
        "crs": None,
        "granule": None,
        "source": f"tile-name ({size:g} deg)",
    }


# Attribute names a shapefile index plausibly stores its raster names under,
# best first. Matched exactly before being matched as a substring.
NAME_FIELD_HINTS = ("filename", "file_name", "fname", "file", "name", "tile",
                    "tilename", "tile_name", "image", "scene", "location",
                    "path", "label", "id")


def rank_name_fields(field_names):
    """Order a shapefile's attributes by how likely they name the raster."""
    def score(field):
        low = field.lower()
        for i, hint in enumerate(NAME_FIELD_HINTS):
            if low == hint:
                return (0, i)
        for i, hint in enumerate(NAME_FIELD_HINTS):
            if hint in low:
                return (1, i)
        return (2, 0)
    return sorted(field_names, key=lambda f: (score(f), f.lower()))


def build_name_lookup(raster_names):
    """Case-insensitive name -> actual name, built once for a whole index pass."""
    return {n.lower(): n for n in raster_names}


def resolve_index_name(value, raster_names):
    """An index attribute's value -> the raster it names, or None.

    Values seen in the wild are a bare stem, a filename, or a full path from
    whatever machine wrote the index -- so only the basename is trusted, and a
    missing extension is filled in.

    `raster_names` is a sequence of names, or a mapping from build_name_lookup;
    pass the mapping when walking an index so the lookup is not rebuilt per
    feature.
    """
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    base = os.path.basename(text)
    if not base:
        return None
    lower = (raster_names if isinstance(raster_names, dict)
             else build_name_lookup(raster_names))
    if base.lower() in lower:
        return lower[base.lower()]
    stem = os.path.splitext(base)[0]
    for ext in RASTER_EXTS:
        hit = lower.get((stem + ext).lower())
        if hit:
            return hit
    return None


def pick_index_shapefile(names):
    """Choose the indexing shapefile from a listing, preferring 'index.shp'."""
    shps = sorted(n for n in names if n.lower().endswith(".shp"))
    if not shps:
        return None
    for shp in shps:
        if os.path.splitext(os.path.basename(shp))[0].lower() == "index":
            return shp
    return shps[0]


def detect_reference_mode(names):
    """Work out how a reference folder declares its footprints, or None.

    Order matters: an explicit index or per-raster sidecar states the real
    footprint, while a tile name only implies a nominal cell, so the name is the
    last resort rather than the first guess.
    """
    if pick_index_shapefile(names):
        return REF_MODE_INDEX
    if any(is_meta_file(n) for n in names):
        return REF_MODE_SIDECAR
    if any(parse_degree_tile(n) for n in names
           if n.lower().endswith(RASTER_EXTS)):
        return REF_MODE_TILE
    return None


def is_meta_file(name):
    """Which sidecar parser a filename belongs to, or None."""
    low = name.lower()
    if low.endswith(META_SUFFIX_XML.lower()):
        return "iso-xml"
    if any(low.endswith(suffix.lower()) for suffix in META_SUFFIXES_TEXT):
        return "text"
    return None


# ── END PURE HELPERS ──────────────────────────────────────────────────────────


# ── MAP TOOL ──────────────────────────────────────────────────────────────────
class DragMapTool(QgsMapTool):
    def __init__(self, canvas, parent, is_left_map):
        super().__init__(canvas)
        self.canvas      = canvas
        self.parent      = parent
        self.is_left_map = is_left_map
        self.dragging    = False
        self.setCursor(Qt.CrossCursor)

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
        col = 0 if self.is_left_map else 2
        # Both canvases render in the working CRS -- QGIS reprojects the WGS84
        # reference on the fly -- so a pick is already in the units the table
        # and the error columns use. No transform, and none to get wrong.
        if self.is_left_map:
            sx, sy = point.x(), point.y()
        else:
            p = self.parent._from_ref_canvas(point)
            sx, sy = p.x(), p.y()
        try:
            self.parent.table.blockSignals(True)
            self.parent.table.setItem(row, col,     QTableWidgetItem(f"{sx:.3f}"))
            self.parent.table.setItem(row, col + 1, QTableWidgetItem(f"{sy:.3f}"))
        finally:
            self.parent.table.blockSignals(False)
        self.parent.draw_marker(point, self.canvas,
                                Qt.red if self.is_left_map else Qt.green)
        self.parent.calculate_error(row)
        if self.is_left_map:
            self.parent.follow_input_point(QgsPointXY(sx, sy))


# ── ARROW-KEY NUDGE ───────────────────────────────────────────────────────────
class ArrowNudgeFilter(QObject):
    """Turn the arrow keys over a canvas into a one-pixel move of the mark.

    QgsMapCanvas takes the arrow keys to pan, which is the wrong thing while
    measuring: the position being refined is the point, not the view. The event
    is swallowed only when a nudge actually happened, so panning still works
    when there is no point to move -- and the table keeps its own arrow-key cell
    navigation, since this filter only sees keys while a canvas has focus.
    """

    _DELTAS = None      # built lazily; Qt.Key_* are not available at import in
                        # a stubbed environment

    def __init__(self, parent, is_left_map):
        super().__init__(parent)
        self.dashboard   = parent
        self.is_left_map = is_left_map

    def _deltas(self):
        if ArrowNudgeFilter._DELTAS is None:
            ArrowNudgeFilter._DELTAS = {
                Qt.Key_Left:  (-1, 0),
                Qt.Key_Right: (1, 0),
                Qt.Key_Up:    (0, 1),      # north is +Y in a projected CRS
                Qt.Key_Down:  (0, -1),
            }
        return ArrowNudgeFilter._DELTAS

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.KeyPress:
                delta = self._deltas().get(event.key())
                if delta is not None:
                    step = NUDGE_PIXELS
                    if event.modifiers() & Qt.ShiftModifier:
                        step *= NUDGE_SHIFT_FACTOR
                    if self.dashboard.nudge_point(
                            self.is_left_map, delta[0] * step, delta[1] * step):
                        return True         # handled: do not let the canvas pan
        except Exception as e:
            print(f"[NUDGE] {e}")
        return super().eventFilter(obj, event)


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

        self.markers           = {"left": [], "right": []}
        self.ref_folder_path   = None
        self.ref_footprints    = {}
        self.ref_mode          = None
        self.input_ring        = None
        self._stretch_cache    = {}
        self.ref_tif_list      = []
        self.input_tif_layer   = None
        self.current_ref_layer = None
        self._syncing          = False

        self.wgs84_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        self.proj_crs  = QgsCoordinateReferenceSystem(WORKING_CRS_DEFAULT)
        self._rebuild_transforms()

        self._configure_gdal_cache()

        self.canvas_left  = QgsMapCanvas()
        self.canvas_right = QgsMapCanvas()
        self.canvas_left.enableAntiAliasing(False)
        self.canvas_right.enableAntiAliasing(False)
        self.canvas_left.setCachingEnabled(True)
        self.canvas_right.setCachingEnabled(True)
        self.canvas_left.setParallelRenderingEnabled(True)
        self.canvas_right.setParallelRenderingEnabled(True)

        self.band_container = QWidget(self.canvas_left)
        self.band_container.setGeometry(10, 10, 420, 35)
        self.band_container.setStyleSheet("background-color: rgba(255,255,255,153);")
        _bl = QHBoxLayout(self.band_container)
        _bl.setContentsMargins(5, 5, 5, 5)
        _bl.addWidget(QLabel("R G B"))
        self.band_combos = []
        for channel in ("red", "green", "blue"):
            combo = QComboBox()
            combo.setToolTip(
                f"Band shown as {channel} on the input canvas.\n"
                f"Set all three for a composite; set red alone for greyscale.")
            _bl.addWidget(combo, 1)
            self.band_combos.append(combo)
        self.band_container.hide()
        self.band_resize_filter = DropdownResizeFilter(self.canvas_left,
                                                       self.band_container)
        self.canvas_left.installEventFilter(self.band_resize_filter)

        self.dropdown_container = QWidget(self.canvas_right)
        self.dropdown_container.setGeometry(10, 10, 600, 35)
        self.dropdown_container.setStyleSheet("background-color: rgba(255,255,255,153);")
        _dl = QHBoxLayout(self.dropdown_container)
        _dl.setContentsMargins(5, 5, 5, 5)
        self.dropdown_band = QComboBox()
        self.dropdown_band.addItem("All bands")
        self.dropdown_band.setToolTip(
            "Restrict the reference list to one NISAR band.\n"
            "The tag comes from the sidecar metadata: LSAR/SSAR in the granule\n"
            "name, else the centre frequency (L ~1.24 GHz, S ~3.2 GHz)."
        )
        self.dropdown_ref = QComboBox()
        _dl.addWidget(self.dropdown_band)
        _dl.addWidget(self.dropdown_ref, 1)
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
        # One exclusive row of map tools, applied to both canvases at once.
        self.tool_buttons = {}
        self.tool_group   = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        tips = {
            TOOL_MARK: "Click or drag to mark the point being measured (Ctrl+1)",
            TOOL_PAN: "Drag to move the view (Ctrl+2)",
            TOOL_ZOOM_IN: "Drag a box, or click, to zoom in (Ctrl+3)",
            TOOL_ZOOM_OUT: "Drag a box, or click, to zoom out (Ctrl+4)",
        }
        for i, mode in enumerate(MAP_TOOLS):
            button = QPushButton(mode.title())
            button.setCheckable(True)
            button.setToolTip(tips[mode] + "\n\nApplies to both canvases; the "
                              "mouse wheel zooms whichever tool is active.")
            self.tool_group.addButton(button, i)
            self.tool_buttons[mode] = button
        self.tool_buttons[TOOL_MARK].setChecked(True)
        self.cb_normalize_input = QCheckBox("Normalize NISAR")
        self.cb_normalize_input.setChecked(False)
        self.cb_normalize_input.setToolTip(
            "OFF = each channel clipped at its 2%-98% percentiles\n"
            "ON  = SAR sqrt-gamma stretch over DN 0-1500, per channel"
        )
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
        for w in ([self.btn_input_tif, self.btn_reference_folder,
                   self.btn_load, self.btn_add, self.btn_save, self.btn_del]
                  + [self.tool_buttons[m] for m in MAP_TOOLS]
                  + [self.cb_sync, self.cb_normalize_input, self.cb_normalize]):
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
        self.dropdown_band.currentIndexChanged.connect(lambda _: self.filter_reference_tifs())
        self.cb_normalize.stateChanged.connect(self.toggle_normalization)
        self.tool_group.buttonClicked.connect(lambda _: self.apply_map_tool())
        self.cb_normalize_input.stateChanged.connect(
            lambda _: self.apply_input_bands())
        for combo in self.band_combos:
            combo.currentIndexChanged.connect(lambda _: self.apply_input_bands())

        QShortcut(QKeySequence("Ctrl+N"),      self).activated.connect(self.add_manual_row)
        QShortcut(QKeySequence("Ctrl+S"),      self).activated.connect(self.save_csv)
        QShortcut(QKeySequence("Ctrl+O"),      self).activated.connect(self.load_csv_smart)
        QShortcut(QKeySequence("Ctrl+Delete"), self).activated.connect(self.delete_row)
        QShortcut(QKeySequence("F5"),          self).activated.connect(self.sync_view_to_row)
        QShortcut(QKeySequence("Escape"),      self).activated.connect(self.clear_markers)
        for i, mode in enumerate(MAP_TOOLS, start=1):
            QShortcut(QKeySequence(f"Ctrl+{i}"), self).activated.connect(
                lambda m=mode: self.set_map_tool(m))

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
    def _gamma_bounds(self, source, band_no=1):
        """(min_dn, max_dn) for the SAR sqrt-gamma stretch, or (None, None).

        Reads a downsampled tile, applies the power stretch, and inverse-maps
        the 2%-98% output percentiles back to input DN so the result can drive a
        plain QgsContrastEnhancement. Shared by the reference and input canvases
        so 'Normalize' means the same thing on both.
        """
        try:
            from osgeo import gdal
            ds = gdal.Open(source, gdal.GA_ReadOnly)
            if not ds:
                return (None, None)
            band   = ds.GetRasterBand(band_no)
            if band is None:
                return (None, None)
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
                return (None, None)

            p2_out  = float(np.percentile(valid, 2))
            p98_out = float(np.percentile(valid, 98))

            # Inverse-map stretched percentiles back to input DN values
            inv_gamma = 1.0 / NORM_GAMMA
            min_dn = NORM_MIN + dn_range * ((p2_out  / 255.0) ** inv_gamma)
            max_dn = NORM_MIN + dn_range * ((p98_out / 255.0) ** inv_gamma)

            if max_dn <= min_dn:
                min_dn, max_dn = float(NORM_MIN), float(NORM_MAX)
            return (min_dn, max_dn)
        except Exception as e:
            print(f"[NORM] Error: {e}")
            return (None, None)

    def normalize_layer(self, layer):
        """Apply the SAR sqrt-gamma stretch to the reference layer."""
        if not layer or not layer.isValid():
            return
        min_dn, max_dn = self._gamma_bounds(layer.source(), 1)
        if min_dn is None:
            return
        try:
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
            stats     = self.sampled_stats(provider, 1)
            if stats is None:
                return
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

            # only touch extent if caller really wants it -- and in the CRS the
            # canvas is drawing in, not the layer's
            ext = self._extent_in_ref_canvas(lyr)
            try:
                print(f"[LOAD] {os.path.basename(tif_path)} "
                      f"[{lyr.crs().authid()}] spans "
                      f"{ext.xMinimum():.1f},{ext.yMinimum():.1f} .. "
                      f"{ext.xMaximum():.1f},{ext.yMaximum():.1f} "
                      f"in the reference canvas CRS")
            except Exception:
                pass
            if set_extent:
                self.canvas_right.setExtent(ext)

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

    # ── WORKING CRS ──────────────────────────────────────────────────────────
    def _rebuild_transforms(self):
        """Rebuild the WGS84 <-> working-CRS transforms after a CRS change."""
        self.transform_proj_to_wgs = QgsCoordinateTransform(
            self.proj_crs, self.wgs84_crs, QgsProject.instance()
        )
        self.transform_wgs_to_proj = QgsCoordinateTransform(
            self.wgs84_crs, self.proj_crs, QgsProject.instance()
        )
        self._apply_canvas_crs()

    def _ref_canvas_crs(self):
        return self.wgs84_crs if REF_CANVAS_CRS == "wgs84" else self.proj_crs

    def _to_ref_canvas(self, pt):
        """A point in the working CRS -> the reference canvas's CRS."""
        if REF_CANVAS_CRS == "wgs84":
            return self.transform_proj_to_wgs.transform(pt)
        return pt

    def _from_ref_canvas(self, pt):
        """A point picked on the reference canvas -> the working CRS."""
        if REF_CANVAS_CRS == "wgs84":
            return self.transform_wgs_to_proj.transform(pt)
        return pt

    def _extent_in_ref_canvas(self, lyr):
        """A layer's extent expressed in the reference canvas's CRS.

        The extent comes back in the LAYER's CRS. Handing a WGS84 rectangle to a
        canvas drawing in UTM puts the view at (73, 16) metres -- off the coast
        of Africa, with the raster nowhere in sight and no error raised.
        """
        dest = self._ref_canvas_crs()
        src  = lyr.crs()
        try:
            if src.authid() and dest.authid() and src.authid() == dest.authid():
                return lyr.extent()
            return QgsCoordinateTransform(
                src, dest, QgsProject.instance()
            ).transformBoundingBox(lyr.extent())
        except Exception as e:
            print(f"[LOAD] extent transform failed: {e}")
            return lyr.extent()

    def _left_view_width(self):
        """The input canvas's current ground width in metres, or None."""
        try:
            extent = self.canvas_left.extent()
            width = float(extent.xMaximum()) - float(extent.xMinimum())
        except Exception:
            return None
        return width if width > 0 else None

    def _ref_view_rect(self, pt, width_m=None):
        """A view box of ground width around a working-CRS point.

        Built in metres and then converted, so it is the same patch of ground
        whichever CRS the reference canvas is drawing in.

        With 'Sync Maps' on, the width follows the input canvas, so zooming
        either side keeps the two showing the same scale -- which is the point
        of having a zoom tool at all. Otherwise it is REF_VIEW_WIDTH_M.
        """
        if width_m is None:
            width_m = None
            try:
                if self.cb_sync.isChecked():
                    width_m = self._left_view_width()
            except Exception:
                width_m = None
            if width_m is None:
                width_m = REF_VIEW_WIDTH_M
        aspect = 1.0
        try:
            size = self.canvas_right.size()
            aspect = float(size.width()) / max(float(size.height()), 1.0)
        except Exception:
            pass
        if not aspect or aspect <= 0:
            aspect = 1.0
        half_w = width_m / 2.0
        half_h = (width_m / aspect) / 2.0
        rect = QgsRectangle(pt.x() - half_w, pt.y() - half_h,
                            pt.x() + half_w, pt.y() + half_h)
        if REF_CANVAS_CRS == "wgs84":
            try:
                rect = self.transform_proj_to_wgs.transformBoundingBox(rect)
            except Exception as e:
                print(f"[VIEW] rect transform failed: {e}")
        return rect

    def show_ref_at(self, pt, colour):
        """Put the reference canvas over a working-CRS point and mark it."""
        rect = self._ref_view_rect(pt)
        self.canvas_right.setExtent(rect)
        self.canvas_right.refresh()
        self.draw_marker(self._to_ref_canvas(pt), self.canvas_right, colour)

    def _apply_canvas_crs(self):
        """Pin each canvas to the CRS it draws in.

        The input canvas is always the working CRS. The reference canvas follows
        REF_CANVAS_CRS: the working CRS by default, so QGIS reprojects the WGS84
        reference as it draws and a pick needs no conversion; or WGS84, which
        skips that reprojection and converts picks instead.

        Either way it must be set explicitly. A bare QgsMapCanvas inherits the
        project's CRS, so a pick could otherwise be read in whatever CRS the
        reference happened to carry -- silently wrong numbers, not an error.
        """
        left = getattr(self, "canvas_left", None)
        right = getattr(self, "canvas_right", None)
        if left is None or right is None:
            return          # called from __init__ before the canvases exist
        try:
            left.setDestinationCrs(self.proj_crs)
            ref = self._ref_canvas_crs()
            right.setDestinationCrs(ref)
            print(f"[CRS] input canvas "
                  f"{self.proj_crs.authid() or self.proj_crs.description()}, "
                  f"reference canvas {ref.authid() or ref.description()}")
        except Exception as e:
            print(f"[CRS] could not pin canvas CRS: {e}")

    def adopt_working_crs(self, crs):
        """Take the working CRS from the input raster.

        The table's In/Ref columns and the error metres are expressed in it, so a
        hard-coded zone is wrong the moment a scene sits in another one -- and an
        LSAR frame is wide enough to straddle two. Geographic CRSs are refused:
        the errors have to come out in metres.
        """
        if crs is None or not crs.isValid() or crs.isGeographic():
            return False
        if crs.authid() and crs.authid() == self.proj_crs.authid():
            return False
        old = self.proj_crs.authid() or self.proj_crs.description()
        self.proj_crs = crs
        self._rebuild_transforms()
        self._reproject_footprints()
        print(f"[CRS] Working CRS {old} -> "
              f"{crs.authid() or crs.description()} (from input raster)")
        return True

    def _reproject_footprints(self):
        """Re-derive every footprint's projected ring in the current working CRS."""
        for rec in self.ref_footprints.values():
            rec["ring_proj"] = [
                (lambda q: (q.x(), q.y()))(
                    self.transform_wgs_to_proj.transform(QgsPointXY(lon, lat)))
                for lon, lat in rec["ring"]
            ]

    # ── REFERENCE FOLDER ─────────────────────────────────────────────────────
    def _folder_entries(self, folder_path):
        """Rasters, sidecars and everything else under the chosen folder.

        Collections get filed into subfolders -- by region, by year, or imagery
        above with a Meta/ beside it -- so the tree is walked rather than just
        listed, and the two halves are matched by name across that split. Files
        are keyed by basename, so a name appearing twice keeps the first and is
        reported rather than silently shadowing.
        """
        rasters, metas, others = {}, {}, {}
        dupes, count, truncated = [], 0, False
        root_depth = folder_path.rstrip(os.sep).count(os.sep)

        for dirpath, dirnames, filenames in os.walk(folder_path):
            if dirpath.rstrip(os.sep).count(os.sep) - root_depth >= REF_SCAN_DEPTH:
                dirnames[:] = []
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                count += 1
                if count > REF_SCAN_MAX_FILES:
                    truncated = True
                    break
                full = os.path.join(dirpath, name)
                if name.lower().endswith(RASTER_EXTS):
                    bucket = rasters
                elif is_meta_file(name):
                    bucket = metas
                else:
                    bucket = others
                if name in bucket:
                    dupes.append(name)
                else:
                    bucket[name] = full
            if truncated:
                break

        if truncated:
            print(f"[META] stopped after {REF_SCAN_MAX_FILES} files -- point the "
                  f"picker at a narrower folder if tiles are missing")
        if dupes:
            print(f"[META] {len(dupes)} duplicate filename(s) across subfolders, "
                  f"first kept (e.g. {', '.join(sorted(set(dupes))[:3])})")
        return rasters, metas, others

    def select_reference_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Reference Folder")
        if not folder_path:
            return
        self.ref_folder_path = folder_path
        self.ref_footprints  = {}
        self.ref_mode        = None
        self._refresh_band_choices()
        errors = []

        rasters, metas, others = self._folder_entries(folder_path)
        if not rasters:
            QMessageBox.critical(
                self, "Error",
                f"No rasters ({', '.join(RASTER_EXTS)}) in {folder_path} "
                f"or its subfolders.")
            self.filter_reference_tifs()
            return

        listing = list(rasters) + list(metas) + list(others)
        mode = REF_MODE_OVERRIDE or detect_reference_mode(listing)
        if mode is None:
            QMessageBox.warning(
                self, "Reference Folder",
                f"{len(rasters)} raster(s) found, but no footprints could be "
                f"read from them.\n\nExpected one of:\n"
                f"  - a shapefile index (e.g. Meta/index.shp)\n"
                f"  - a sidecar per raster ('.met', '.h5.iso.xml', '_meta.txt')\n"
                f"  - degree-tile names (e.g. N16E73.tif)\n\n"
                f"Set REF_MODE_OVERRIDE in the script to force one.")
            self.filter_reference_tifs()
            return

        print(f"[META] {os.path.basename(folder_path)}: {len(rasters)} raster(s), "
              f"footprints from {mode}")
        if mode == REF_MODE_INDEX:
            errors = self._scan_index(rasters, {**metas, **others})
        elif mode == REF_MODE_SIDECAR:
            errors = self._scan_sidecars(rasters, metas)
        else:
            errors = self._scan_tile_names(rasters)

        self.ref_mode = mode
        self._reproject_footprints()
        self._refresh_band_choices()

        counts = {}
        for rec in self.ref_footprints.values():
            counts[rec["band"]] = counts.get(rec["band"], 0) + 1
        if counts:
            print("[META] Tagged: "
                  + ", ".join(f"{n} {b}" for b, n in sorted(counts.items())))
        if self.ref_footprints:
            bounds = rings_bounds([r["ring"] for r in self.ref_footprints.values()])
            print(f"[META] {len(self.ref_footprints)} footprint(s) covering "
                  f"{format_bounds(bounds)}")
        if errors:
            txt = "\n".join(errors[:10])
            if len(errors) > 10:
                txt += f"\n\n...and {len(errors) - 10} more"
            QMessageBox.warning(self, "Parsing Warnings", txt)
        elif not self.ref_footprints:
            QMessageBox.warning(
                self, "Reference Folder",
                f"Read footprints from {mode}, but none matched a raster.")
        self.filter_reference_tifs()

    def _input_footprint_ring(self, raster_path):
        """The input NISAR scene's true lon/lat footprint from its own sidecar.

        The raster's extent is the north-up product grid, which includes the
        nodata wedges either side of a slanted swath. The '.met' Image* corners
        and the ISO gml:posList give the swath itself, so selecting reference
        tiles against it offers only tiles the scene actually has data over.
        Falls back to the extent when no sidecar sits beside the raster.
        """
        folder = os.path.dirname(raster_path)
        base   = os.path.basename(raster_path)
        dirs   = [folder]
        try:
            for entry in sorted(os.listdir(folder)):
                if (os.path.isdir(os.path.join(folder, entry))
                        and entry.lower() in META_DIR_NAMES):
                    dirs.append(os.path.join(folder, entry))
        except OSError as e:
            print(f"[INPUT] {folder}: {e}")
            return None

        for d in dirs:
            try:
                names = sorted(os.listdir(d))
            except OSError:
                continue
            for name in names:
                kind = is_meta_file(name)
                if not kind:
                    continue
                if match_raster([base], meta_base_stem(name)) != base:
                    continue
                try:
                    with open(os.path.join(d, name), "r",
                              encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError as e:
                    print(f"[INPUT] {name}: {e}")
                    continue
                rec = (parse_meta_iso_xml(content, name) if kind == "iso-xml"
                       else parse_meta_text(content, name))
                if rec and rec.get("ring"):
                    print(f"[INPUT] footprint from {name}: "
                          f"{len(rec['ring'])} vertices, "
                          f"{format_bounds(rings_bounds([rec['ring']]))}")
                    return rec["ring"]
        return None

    def _scan_sidecars(self, rasters, metas):
        """One metadata file per raster: NISAR '.met' / '.h5.iso.xml', or gdalinfo."""
        errors = []
        for meta_file, meta_path in sorted(metas.items()):
            kind = is_meta_file(meta_file)
            try:
                with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                rec = (parse_meta_iso_xml(content, meta_file) if kind == "iso-xml"
                       else parse_meta_text(content, meta_file))
                if not rec:
                    errors.append(f"{meta_file}: could not parse footprint")
                    continue
                stem = meta_base_stem(meta_file)
                tif  = match_raster(list(rasters), stem, rec.get("granule"))
                if not tif:
                    errors.append(f"{meta_file}: no raster found for '{stem}'")
                    continue
                rec["meta"] = meta_path
                self.ref_footprints[rasters[tif]] = rec
            except Exception as e:
                errors.append(f"{meta_file}: {e}")
        return errors

    def _scan_tile_names(self, rasters):
        """The footprint is the name: 'N16E73.tif' is a degree cell."""
        errors = []
        for name, path in sorted(rasters.items()):
            rec = parse_degree_tile(name)
            if rec is None:
                errors.append(f"{name}: name is not a degree tile "
                              f"(expected e.g. N16E73, N8E76, S34E018)")
                continue
            rec["meta"] = None
            self.ref_footprints[path] = rec
        return errors

    def _scan_index(self, rasters, candidates):
        """One shapefile indexes the set, its attributes naming each raster."""
        errors = []
        shp_name = pick_index_shapefile(list(candidates))
        shp_path = candidates[shp_name]
        lyr = QgsVectorLayer(shp_path, "ref_index", "ogr")
        if not lyr.isValid():
            return [f"{shp_name}: not a readable vector layer"]

        tf = QgsCoordinateTransform(lyr.crs(), self.wgs84_crs, QgsProject.instance())
        fields = rank_name_fields([f.name() for f in lyr.fields()])
        if not fields:
            return [f"{shp_name}: no attributes to match raster names against"]

        # Lock onto the first attribute that actually resolves to a raster, so a
        # later feature with a blank cell cannot silently switch fields.
        lookup = build_name_lookup(rasters)
        name_field, unmatched = None, 0
        for feat in lyr.getFeatures():
            if name_field is None:
                for field in fields:
                    if resolve_index_name(feat[field], lookup):
                        name_field = field
                        print(f"[META] {shp_name}: raster names from "
                              f"attribute '{field}'")
                        break
                if name_field is None:
                    unmatched += 1
                    continue

            tif = resolve_index_name(feat[name_field], lookup)
            if not tif:
                unmatched += 1
                continue
            ring = self._geometry_ring(feat.geometry(), tf)
            if not ring:
                errors.append(f"{shp_name}: '{tif}' has no usable polygon")
                continue
            self.ref_footprints[rasters[tif]] = {
                "ring": ring, "band": BAND_UNKNOWN, "crs": lyr.crs().authid() or None,
                "granule": None, "source": f"index ({shp_name}:{name_field})",
                "meta": shp_path,
            }

        if name_field is None:
            errors.append(
                f"{shp_name}: no attribute matched any raster name. Fields are: "
                + ", ".join(fields[:12]))
        elif unmatched:
            # An index routinely spans more than the folder holds. Worth stating,
            # not worth a dialog on every load.
            print(f"[META] {shp_name}: {unmatched} index entr(ies) name a raster "
                  f"not in this folder; {len(self.ref_footprints)} matched")
        return errors

    @staticmethod
    def _geometry_ring(geom, transform):
        """A feature's exterior ring in WGS84, largest part if multipart."""
        if geom is None or geom.isEmpty():
            return None
        try:
            if geom.isMultipart():
                parts = [p[0] for p in geom.asMultiPolygon() if p]
                if not parts:
                    return None
                pts = max(parts, key=len)
            else:
                poly = geom.asPolygon()
                if not poly:
                    return None
                pts = poly[0]
            ring = []
            for pt in pts:
                q = transform.transform(QgsPointXY(pt))
                ring.append((q.x(), q.y()))
            return ring if len(ring) >= 3 else None
        except Exception as e:
            print(f"[META] geometry: {e}")
            return None

    # ── STAGE 1 FILTER ────────────────────────────────────────────────────────
    def _band_choice(self):
        txt = self.dropdown_band.currentText()
        return None if txt.startswith("All") else txt

    def _refresh_band_choices(self):
        """Offer only the band tags this folder actually holds.

        A NISAR reference splits LSAR/SSAR; a C1 or L8 reference has no band at
        all, so the filter collapses to a single disabled entry rather than
        listing choices that would empty the list.
        """
        tags = sorted({rec.get("band", BAND_UNKNOWN)
                       for rec in self.ref_footprints.values()} - {BAND_UNKNOWN})
        try:
            self.dropdown_band.blockSignals(True)
            self.dropdown_band.clear()
            self.dropdown_band.addItem("All bands")
            for tag in tags:
                self.dropdown_band.addItem(tag)
        finally:
            self.dropdown_band.blockSignals(False)
        self.dropdown_band.setEnabled(bool(tags))

    def _label_for(self, tif_path):
        rec  = self.ref_footprints.get(tif_path, {})
        band = rec.get("band", BAND_UNKNOWN)
        name = os.path.basename(tif_path)
        return f"[{band}] {name}" if band != BAND_UNKNOWN else name

    def filter_reference_tifs(self):
        self.ref_tif_list = []
        self.dropdown_ref.blockSignals(True)
        self.dropdown_ref.clear()
        self.dropdown_ref.blockSignals(False)
        if not self.ref_footprints:
            self.dropdown_container.hide()
            return

        band = self._band_choice()
        candidates = [t for t, rec in self.ref_footprints.items()
                      if band is None or rec.get("band") == band]

        if not self.input_tif_layer or not self.input_tif_layer.isValid():
            self.ref_tif_list = candidates
        else:
            try:
                if self.input_ring:
                    input_geom = QgsGeometry.fromWkt(ring_wkt(self.input_ring))
                else:
                    tf = QgsCoordinateTransform(
                        self.input_tif_layer.crs(), self.wgs84_crs,
                        QgsProject.instance()
                    )
                    input_geom = QgsGeometry.fromRect(
                        tf.transformBoundingBox(self.input_tif_layer.extent())
                    )
                for tif_path in candidates:
                    try:
                        wkt = ring_wkt(self.ref_footprints[tif_path]["ring"])
                        if wkt and QgsGeometry.fromWkt(wkt).intersects(input_geom):
                            self.ref_tif_list.append(tif_path)
                    except Exception as e:
                        print(f"[FILTER] {tif_path}: {e}")
                if candidates and not self.ref_tif_list:
                    # Footprints read fine, they just do not cover this scene.
                    # Say so with both extents rather than an empty dropdown.
                    box = input_geom.boundingBox()
                    ref_b = rings_bounds(
                        [self.ref_footprints[t]["ring"] for t in candidates])
                    print(f"[FILTER] none of {len(candidates)} reference "
                          f"footprint(s) overlap the input.")
                    print(f"[FILTER]   input     : "
                          f"{format_bounds((box.xMinimum(), box.yMinimum(), box.xMaximum(), box.yMaximum()))}")
                    print(f"[FILTER]   reference : {format_bounds(ref_b)}")
            except Exception as e:
                print(f"[FILTER] Error: {e}")
                self.ref_tif_list = candidates

        if self.ref_tif_list:
            self.dropdown_ref.blockSignals(True)
            for p in self.ref_tif_list:
                self.dropdown_ref.addItem(self._label_for(p))
            self.dropdown_ref.blockSignals(False)
            self.dropdown_container.show()
            self.dropdown_container.raise_()
            self.dropdown_ref.setCurrentIndex(0)
        else:
            label = ("No matches found" if not candidates else
                     f"None of {len(candidates)} tile(s) overlap the input")
            self.dropdown_ref.blockSignals(True)
            self.dropdown_ref.addItem(label)
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
        self._remove_markers(self.canvas_right, "right")
        self.cleanup_reference_layer()
        self._load_ref_layer(tif_path)

    def cleanup_reference_layer(self):
        if self.current_ref_layer and self.current_ref_layer.isValid():
            QgsProject.instance().removeMapLayer(self.current_ref_layer.id())
        self.current_ref_layer = None

    # ── STAGE 2 AUTO-SWITCH ───────────────────────────────────────────────────
    def reference_for_point(self, pt):
        """The smallest loaded reference footprint containing a working-CRS point."""
        target      = QgsGeometry.fromPointXY(pt)
        best, best_a = None, float("inf")
        for tif_path in self.ref_tif_list:
            rec = self.ref_footprints.get(tif_path)
            if not rec or not rec.get("ring_proj"):
                continue
            wkt = ring_wkt(rec["ring_proj"])
            if not wkt:
                continue
            geom = QgsGeometry.fromWkt(wkt)
            if geom.contains(target):
                area = geom.area()
                if area < best_a:
                    best_a, best = area, tif_path
        return best

    def show_reference_for(self, pt):
        """Bring up the reference tile covering a point. True if one is showing."""
        best = self.reference_for_point(pt)
        if not best:
            self.cleanup_reference_layer()
            self.canvas_right.setLayers([])
            self.canvas_right.refresh()
            return False
        if (self.current_ref_layer and self.current_ref_layer.isValid()
                and os.path.normpath(self.current_ref_layer.source())
                == os.path.normpath(best)):
            return True          # already showing; do not reload on every drag
        try:
            self.dropdown_ref.blockSignals(True)
            self.dropdown_ref.setCurrentIndex(self.ref_tif_list.index(best))
        finally:
            self.dropdown_ref.blockSignals(False)
        self.cleanup_reference_layer()
        self._load_ref_layer(best, set_extent=False)
        return True

    def follow_input_point(self, pt):
        """Point the reference canvas at a position marked on the input canvas.

        The reference is only useful at the place being measured, so marking on
        the left brings up the tile covering that ground, centres the right
        canvas on the same position and marks it. The marker is red -- it mirrors
        the input, and is not a reference pick until clicked on the right, which
        redraws it green.
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            if not self.show_reference_for(pt):
                return
            self.show_ref_at(pt, Qt.red)
        except Exception as e:
            print(f"[FOLLOW] {e}")
        finally:
            self._syncing = False

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
                # no reference pick yet -- follow the input point instead of
                # blanking the canvas, so the row is still usable
                ii, ij = self.table.item(row, 0), self.table.item(row, 1)
                try:
                    ix, iy = float(ii.text()), float(ij.text())
                except (AttributeError, ValueError):
                    ix = iy = 0.0
                if ix or iy:
                    self.show_reference_for(QgsPointXY(ix, iy))
                else:
                    self.cleanup_reference_layer()
                    self.canvas_right.setLayers([])
                    self.canvas_right.refresh()
                return
            self.show_reference_for(QgsPointXY(rx, ry))
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

    def _remove_markers(self, canvas, key):
        """Take this canvas's marker items off its scene."""
        existing = self.markers.get(key) or []
        if not isinstance(existing, (list, tuple)):
            existing = [existing]
        scene = canvas.scene()
        for item in existing:
            if item is None:
                continue
            try:
                if scene:
                    scene.removeItem(item)
            except Exception as e:
                print(f"[MARKER] {e}")
        self.markers[key] = []

    def draw_marker(self, point, canvas, color):
        """A cross in `color`, haloed by a wider translucent yellow one."""
        key = "left" if canvas == self.canvas_left else "right"
        self._remove_markers(canvas, key)

        halo = QgsVertexMarker(canvas)
        halo.setCenter(point)
        halo.setIconType(QgsVertexMarker.ICON_CROSS)
        halo.setColor(QColor(*MARKER_OUTLINE_RGBA))
        halo.setPenWidth(MARKER_OUTLINE_WIDTH)
        halo.setIconSize(MARKER_SIZE + MARKER_OUTLINE_EXTRA)

        cross = QgsVertexMarker(canvas)
        cross.setCenter(point)
        cross.setIconType(QgsVertexMarker.ICON_CROSS)
        cross.setColor(color)
        cross.setPenWidth(MARKER_PEN_WIDTH)
        cross.setIconSize(MARKER_SIZE)

        # halo first so the coloured cross sits on top of it
        self.markers[key] = [halo, cross]
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
                
                
                
                
                
                

            if rx != 0.0 or ry != 0.0:
                self.show_ref_at(QgsPointXY(rx, ry), Qt.green)
            elif ix != 0.0 or iy != 0.0:
                # no reference pick on this row yet: sit on the input position
                self.show_ref_at(QgsPointXY(ix, iy), Qt.red)

        except Exception as e:
            print(f"[SYNC ROW] {e}")
        finally:
            if sync_was:
                self.cb_sync.setChecked(True)

    def sync_canvas_extents(self):
        """Follow the input canvas: same centre and, now, the same scale.

        Fired by extentsChanged, so it covers panning and zooming alike -- the
        reference tracks whatever the input view does.
        """
        if not self.cb_sync.isChecked() or self._syncing:
            return
        self._syncing = True
        try:
            self.canvas_right.setExtent(
                self._ref_view_rect(self.canvas_left.center()))
            self.canvas_right.refresh()
        except Exception as e:
            print(f"[SYNC EXTENTS] {e}")
        finally:
            self._syncing = False

    def clear_markers(self):
        for key, canvas in [("left", self.canvas_left), ("right", self.canvas_right)]:
            self._remove_markers(canvas, key)
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
            self.ensure_overviews(path)
            self.input_ring = self._input_footprint_ring(path)
            if self.input_ring is None:
                print("[INPUT] no sidecar beside the raster; "
                      "selecting references against its full extent")
            self.adopt_working_crs(lyr.crs())
            self.canvas_left.setLayers([lyr])
            self.canvas_left.setExtent(lyr.extent())
            self.populate_band_picker(lyr)
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
            self.ensure_overviews(in_l[0].source())
            self.adopt_working_crs(in_l[0].crs())
            self.canvas_left.setLayers([in_l[0]])
            self.canvas_left.setExtent(in_l[0].extent())
            self.populate_band_picker(in_l[0])
            self.canvas_left.refresh()
        if ref_l:
            self.ensure_overviews(ref_l[0].source())
            self.current_ref_layer = ref_l[0]
            self.canvas_right.setLayers([ref_l[0]])
            self.canvas_right.setExtent(ref_l[0].extent())
            self.reset_normalization(ref_l[0])
            self.canvas_right.refresh()

    def init_map_tools(self):
        """Build one set of map tools per canvas, then activate the current one."""
        self._apply_canvas_crs()
        self.map_tools = {
            TOOL_MARK: (DragMapTool(self.canvas_left, self, True),
                        DragMapTool(self.canvas_right, self, False)),
            TOOL_PAN: (QgsMapToolPan(self.canvas_left),
                       QgsMapToolPan(self.canvas_right)),
            TOOL_ZOOM_IN: (QgsMapToolZoom(self.canvas_left, False),
                           QgsMapToolZoom(self.canvas_right, False)),
            TOOL_ZOOM_OUT: (QgsMapToolZoom(self.canvas_left, True),
                            QgsMapToolZoom(self.canvas_right, True)),
        }
        # kept for the marking paths, which reach for these by name
        self.tool_left, self.tool_right = self.map_tools[TOOL_MARK]
        self.nudge_filters = (ArrowNudgeFilter(self, True),
                              ArrowNudgeFilter(self, False))
        self.canvas_left.installEventFilter(self.nudge_filters[0])
        self.canvas_right.installEventFilter(self.nudge_filters[1])
        self.apply_map_tool()

    # ── ARROW-KEY NUDGE ───────────────────────────────────────────────────────
    def _pixel_step(self, is_left_map, pt):
        """One source pixel, in working-CRS units, at a given point.

        The input raster is already in the working CRS, so its pixel size is
        used directly. A reference tile is usually WGS84, where a pixel is a
        fraction of a degree, so it is measured by stepping one pixel in the
        layer's own CRS and transforming both ends back.
        """
        layer = self.input_tif_layer if is_left_map else self.current_ref_layer
        if layer is None or not layer.isValid():
            return None
        try:
            px = abs(float(layer.rasterUnitsPerPixelX()))
            py = abs(float(layer.rasterUnitsPerPixelY()))
        except Exception:
            return None
        if px <= 0 or py <= 0:
            return None
        try:
            src = layer.crs()
            if src.authid() and src.authid() == self.proj_crs.authid():
                return (px, py)
            project = QgsProject.instance()
            to_layer = QgsCoordinateTransform(self.proj_crs, src, project)
            to_proj  = QgsCoordinateTransform(src, self.proj_crs, project)
            here = to_layer.transform(pt)
            over = to_proj.transform(QgsPointXY(here.x() + px, here.y() + py))
            step_x, step_y = abs(over.x() - pt.x()), abs(over.y() - pt.y())
            if step_x <= 0 or step_y <= 0:
                return None
            return (step_x, step_y)
        except Exception as e:
            print(f"[NUDGE] pixel size: {e}")
            return None

    def nudge_point(self, is_left_map, dx_px, dy_px):
        """Move the selected row's mark by whole pixels. True if it moved.

        Only the mark moves -- the view is left where it is, which is the whole
        point of taking the arrow keys off the canvas.
        """
        row = self.table.currentRow()
        if row < 0:
            return False
        col = 0 if is_left_map else 2
        try:
            x = float(self.table.item(row, col).text())
            y = float(self.table.item(row, col + 1).text())
        except (AttributeError, ValueError):
            return False
        if x == 0.0 and y == 0.0:
            return False        # nothing marked on this side yet

        point = QgsPointXY(x, y)
        step = self._pixel_step(is_left_map, point)
        if step is None:
            return False
        nx, ny = x + dx_px * step[0], y + dy_px * step[1]

        try:
            self.table.blockSignals(True)
            self.table.setItem(row, col,     QTableWidgetItem(f"{nx:.3f}"))
            self.table.setItem(row, col + 1, QTableWidgetItem(f"{ny:.3f}"))
        finally:
            self.table.blockSignals(False)

        moved = QgsPointXY(nx, ny)
        if is_left_map:
            self.draw_marker(moved, self.canvas_left, Qt.red)
        else:
            self.draw_marker(self._to_ref_canvas(moved), self.canvas_right,
                             Qt.green)
        self.calculate_error(row)
        return True

    def current_map_tool(self):
        for mode, button in self.tool_buttons.items():
            if button.isChecked():
                return mode
        return TOOL_MARK

    def set_map_tool(self, mode):
        """Select a tool by name, keeping the button row in step."""
        button = self.tool_buttons.get(mode)
        if button is None:
            return
        button.setChecked(True)
        self.apply_map_tool()

    def apply_map_tool(self, _checked=None):
        """Put the selected tool on BOTH canvases.

        One selection for both: the canvases are linked, so leaving one in
        marking mode while zooming the other only produces stray picks. The mode
        is read from the button row rather than a signal argument, which arrives
        differently depending on how the change was made.
        """
        mode = self.current_map_tool()
        tools = getattr(self, "map_tools", None)
        if not tools:
            return          # called before init_map_tools
        left, right = tools[mode]
        self.canvas_left.setMapTool(left)
        self.canvas_right.setMapTool(right)
        cursor = {TOOL_MARK: Qt.CrossCursor,
                  TOOL_PAN: Qt.OpenHandCursor}.get(mode, Qt.CrossCursor)
        for canvas in (self.canvas_left, self.canvas_right):
            try:
                canvas.setCursor(cursor)
            except Exception:
                pass
        print(f"[TOOL] {mode}")

    # ── INPUT BAND COMPOSITE ──────────────────────────────────────────────────
    def _band_labels(self, layer):
        """Human labels for a raster's bands, using the names it carries.

        cog_locate's --gtiff writes the polarization into each band description,
        so a NISAR chip lists 'HH' and 'HV' rather than 'Band 1'.
        """
        provider = layer.dataProvider()
        labels = []
        for band in range(1, provider.bandCount() + 1):
            name = ""
            try:
                name = (layer.bandName(band) or "").strip()
            except Exception:
                pass
            # QGIS synthesises 'Band 001' when a raster names nothing
            if not name or re.fullmatch(r"Band\s*0*\d+", name):
                name = f"Band {band}"
            elif not name.lower().startswith("band"):
                name = f"{band}: {name}"
            labels.append(name)
        return labels

    def clear_stretch_cache(self, source=None):
        """Drop cached stretch bounds, for one raster or all of them."""
        if source is None:
            self._stretch_cache = {}
        else:
            for key in [k for k in self._stretch_cache if k[0] == source]:
                del self._stretch_cache[key]

    def populate_band_picker(self, layer):
        """Fill the R/G/B combos from the input raster, then apply the default.

        Every slot offers every band and any band may be repeated -- NISAR
        carries two, and which pairing reads best is a judgement about the
        scene. The default puts band 1 in all three, which renders grey.
        """
        if layer is None or not layer.isValid():
            self.band_container.hide()
            return
        self.clear_stretch_cache(layer.source())
        labels = self._band_labels(layer)
        count  = len(labels)
        if count < 1:
            self.band_container.hide()
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
        # nothing to choose between with one band
        if count < 2:
            self.band_container.hide()
        else:
            self.band_container.show()
            self.band_container.raise_()
        print(f"[BANDS] input has {count}: {', '.join(labels)}")
        self.apply_input_bands()

    def _selected_bands(self):
        """The chosen 1-based band per channel."""
        return [max(combo.currentIndex(), 0) + 1 for combo in self.band_combos]

    @staticmethod
    def sampled_cut(provider, band, low, high):
        """cumulativeCut over a bounded sample, or (None, None).

        The sampled overload is tried first and the unsampled signature only as
        a fallback for older QGIS, because unsampled means a full-resolution
        pass over the whole raster.
        """
        try:
            return provider.cumulativeCut(band, low, high, QgsRectangle(),
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
    def sampled_stats(provider, band):
        """bandStatistics over a bounded sample, or None. Same reasoning."""
        try:
            return provider.bandStatistics(
                band, QgsRasterBandStats.Min | QgsRasterBandStats.Max,
                QgsRectangle(), RASTER_SAMPLE_SIZE)
        except TypeError:
            pass
        except Exception:
            return None
        try:
            return provider.bandStatistics(band)
        except Exception:
            return None

    def _stretch_for(self, layer, band):
        """Contrast enhancement for one input band.

        'Normalize NISAR' selects the same SAR sqrt-gamma stretch the reference
        uses; otherwise the band is clipped at the RGB percentiles. Either way a
        plain min/max is avoided -- on SAR it is set by a handful of bright
        scatterers and renders the scene black.

        Bounds are cached per raster, band and stretch: re-picking the channel
        order is a common action and must not recompute statistics each time.
        """
        provider = layer.dataProvider()
        normalize = bool(self.cb_normalize_input.isChecked())
        key = (layer.source(), band, normalize)
        cached = self._stretch_cache.get(key)
        if cached is not None:
            lo, hi = cached
        else:
            lo = hi = None
            if normalize:
                lo, hi = self._gamma_bounds(layer.source(), band)
            if lo is None or hi is None or hi <= lo:
                lo, hi = self.sampled_cut(provider, band,
                                          RGB_CLIP_LOW, RGB_CLIP_HIGH)
            if lo is None or hi is None or hi <= lo:
                stats = self.sampled_stats(provider, band)
                if stats is not None:
                    lo, hi = stats.minimumValue, stats.maximumValue
            if lo is None or hi is None or hi <= lo:
                lo, hi = 0.0, 1.0
            self._stretch_cache[key] = (lo, hi)
        ce = QgsContrastEnhancement(provider.dataType(band))
        ce.setContrastEnhancementAlgorithm(
            QgsContrastEnhancement.StretchToMinimumMaximum)
        ce.setMinimumValue(lo)
        ce.setMaximumValue(hi)
        return ce

    def apply_input_bands(self):
        """Render the input canvas from the picked bands."""
        layer = self.input_tif_layer
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
            self.canvas_left.refresh()
            stretch = ("gamma" if self.cb_normalize_input.isChecked()
                       else f"{RGB_CLIP_LOW:.0%}-{RGB_CLIP_HIGH:.0%}")
            print(f"[BANDS] input rendered as {shown} ({stretch} stretch)")
        except Exception as e:
            print(f"[BANDS] {e}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
app = QApplication.instance() or QApplication(sys.argv)
win = QCDashboard()
win.show()
