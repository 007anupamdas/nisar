#!/usr/bin/env python3
"""
automatch_refs -- reference-collection discovery for the NISAR auto-matcher.

Finds which reference rasters exist in a folder and where each one lies, the
same way DPQED_rival.py does, but without QGIS so it runs inside the batch
matcher and its GUI:

  index-shp    a shapefile indexing the set, one attribute naming each raster
               (L8_ref: Meta/index.shp)
  sidecar      one metadata file per raster: '.met' (JSON or text),
               '.h5.iso.xml' (ISO 19115-2 gml:posList), '_meta.txt' (gdalinfo)
               -- S1 reference folders are this mode
  degree-tile  the name is the footprint (C1: N16E73.tif is 16-17 N, 73-74 E)

The block between the PORTED FROM DPQED_rival.py markers is copied verbatim
from RIVAL's pure-helper section (tests_automatch.py checks it still matches
when DPQED_rival.py is available), so a folder RIVAL understands is understood
here identically. Only the orchestration below it is new: RIVAL reads the
index through QgsVectorLayer; here it is GDAL/OGR, fiona, or a small built-in
.shp/.dbf reader, whichever is available.
"""

import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

REF_MODE_OVERRIDE = None


# ── PORTED FROM DPQED_rival.py (BEGIN) ────────────────────────────────────────
# Sidecar metadata sitting beside '<product>.h5': SSAR ships '<product>.met',
# a plain text file; LSAR ships '<product>.h5.iso.xml', ISO 19115-2. Both are
# scanned, and each footprint is tagged with its band.
# '_meta.txt' is kept for folders written before the '.met' convention.
META_SUFFIXES_TEXT = (".met", "_meta.txt")

META_SUFFIX_XML    = ".iso.xml"

# A sidecar is also recognised by carrying 'meta' anywhere in its NAME, which is
# how some processors label one -- '<scene>_META.txt', 'METADATA.xml' -- rather
# than by a fixed suffix. The EXTENSION still decides which parser runs, and
# only these are considered: a raster called '..._metadata.tif' and an index
# called '..._meta.shp' both have 'meta' in the name and are not sidecars.
META_NAME_TOKEN     = "meta"

META_NAME_EXTS_XML  = (".xml",)

META_NAME_EXTS_TEXT = (".txt", ".met", ".json", ".hdr", ".ini", ".dat")

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

# A full Landsat or Sentinel archive is hundreds of thousands of files once the
# sidecars are counted, and the cap is on files walked rather than rasters kept,
# so a low one silently truncates a real collection. High enough to hold one,
# still low enough that a drive root reports instead of hanging.
REF_SCAN_MAX_FILES = 250000

# What counts as "no data here" on top of the raster's own declared nodata and
# NaN. 0 is the usual fill; 3 is what an earlier ADRIN pipeline wrote, and is
# carried over from the corner_coord definition used there.
REF_FILL_VALUES = (0, 3)

# Percentile clip for the input composite. A min/max stretch on SAR is dominated
# by a handful of bright scatterers and leaves the scene black.
# CE90: the radius holding 90% of a circular normal error. The radial error is
# then Rayleigh and its 90th percentile is 2.146 sigma, with sigma the RMS of
# the two axes -- equivalently 1.5175 x the radial RMSE. The circle is only a
# fair model while the two axes are comparable; NSSDA draws that line at a
# min/max ratio of 0.6, below which the figure is reported with a caveat rather
# than presented as if the error really were circular.
CE90_SIGMA = 2.146

CE90_MIN_AXIS_RATIO = 0.6

_ISO_NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gml": "http://www.opengis.net/gml/3.2",
    "gmx": "http://www.isotc211.org/2005/gmx",
    "eos": "http://earthdata.nasa.gov/schema/eos",
    "gmi": "http://www.isotc211.org/2005/gmi",
}

# Degree counts are not reliably zero-padded: 'N8E76_ortho.tif' is as real as
# 'N16E073.tif'. Latitude takes 1-2 digits, longitude 1-3, and the trailing
# lookahead stops a longer run being read as a tile.
_TILE_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})(?![0-9])", re.IGNORECASE)

# Attribute names a shapefile index plausibly stores its raster names under,
# best first. Matched exactly before being matched as a substring.
NAME_FIELD_HINTS = ("filename", "file_name", "fname", "file", "name", "tile",
                    "tilename", "tile_name", "image", "scene", "location",
                    "path", "label", "id")

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

def is_data_value(value, nodata, fill_values):
    """Does a pixel carry data, or is it fill?

    NaN is fill however it was declared: NaN != NaN, so comparing it against a
    nodata value would keep it, which is the same trap that once rendered a
    whole scene black here. Fill values are compared as floats so an integer
    band and a float one behave alike.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v:
        return False
    try:
        if nodata is not None and v == float(nodata):
            return False
    except (TypeError, ValueError):
        pass
    return not any(v == float(f) for f in fill_values)

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

    A name-token sidecar loses its extension and then the 'meta' label, with
    the separator beside it: '<scene>_META.txt' and 'META_<scene>.txt' both
    reduce to '<scene>'. The label is only taken as a label when a separator
    marks it off, so '<scene>_METADATA.xml' still reduces to '<scene>' while a
    file named only for being metadata -- 'METADATA.xml' -- reduces to '',
    naming no raster at all. The caller decides what an empty stem is worth:
    beside a single input raster it is that raster's, and in a reference folder
    of many it names none of them.
    """
    for suffix in (META_SUFFIX_XML,) + META_SUFFIXES_TEXT:
        if meta_name.lower().endswith(suffix.lower()):
            stem = meta_name[: -len(suffix)]
            break
    else:
        stem = os.path.splitext(meta_name)[0]
        idx = stem.lower().rfind(META_NAME_TOKEN)
        if idx != -1:
            before = stem[:idx].rstrip("_-. ")
            rest   = stem[idx + len(META_NAME_TOKEN):]
            after  = rest.lstrip("_-. ") if rest[:1] in ("_", "-", ".", " ") else ""
            stem = before or after
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

    stems = [stem] if stem else []
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

    if not stem:
        # '' is a prefix of every name, so the fallback below would hand back
        # whichever raster sorted first -- a silent misattribution.
        return None
    prefixed = sorted(
        f for f in files
        if f.lower().startswith(stem.lower()) and f.lower().endswith(exts)
    )
    return prefixed[0] if prefixed else None

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
    """Which sidecar parser a filename belongs to, or None.

    Two ways in. A known suffix -- '.met', '_meta.txt', '.h5.iso.xml' -- or
    'meta' anywhere in the name together with an extension a sidecar is
    actually written with. The extension, never the name, decides which parser
    runs; a name-token match with an extension outside the two lists is refused
    rather than guessed at, which is what keeps '..._metadata.tif' a raster and
    '..._meta.shp' an index.

    An extension is required on the token path, so a directory called 'meta'
    beside the imagery is not mistaken for a file to parse -- META_DIR_NAMES
    handles those.
    """
    low = name.lower()
    if low.endswith(META_SUFFIX_XML.lower()):
        return "iso-xml"
    if any(low.endswith(suffix.lower()) for suffix in META_SUFFIXES_TEXT):
        return "text"
    if META_NAME_TOKEN not in low:
        return None
    ext = os.path.splitext(low)[1]
    if ext in META_NAME_EXTS_XML:
        return "iso-xml"
    if ext in META_NAME_EXTS_TEXT:
        return "text"
    return None

def accuracy_stats(err_x, err_y):
    """Per-axis RMSE and CE90 for a set of picked errors, in map metres.

    RMSE is taken about zero, not about the mean, so a systematic shift counts
    against the accuracy instead of being quietly subtracted out of it -- an
    absolute location error is the whole point of the measurement.

    'circular' reports whether the two axes are close enough for CE90 to mean
    what it says; see CE90_MIN_AXIS_RATIO. The number is returned either way,
    so the caller can caveat it rather than withhold it. No picks is all zeros,
    not an error: the panel shows this before anything has been marked.
    """
    n = len(err_x)
    if n == 0 or n != len(err_y):
        return {"n": 0, "rmse_x": 0.0, "rmse_y": 0.0, "ce90": 0.0,
                "circular": True}
    rmse_x = math.sqrt(sum(e * e for e in err_x) / n)
    rmse_y = math.sqrt(sum(e * e for e in err_y) / n)
    sigma = math.sqrt((rmse_x ** 2 + rmse_y ** 2) / 2.0)
    hi = max(rmse_x, rmse_y)
    ratio = min(rmse_x, rmse_y) / hi if hi else 1.0
    return {"n": n, "rmse_x": rmse_x, "rmse_y": rmse_y,
            "ce90": CE90_SIGMA * sigma,
            "circular": ratio >= CE90_MIN_AXIS_RATIO}

def csv_record(index, cells, in_lonlat, ref_lonlat):
    """One CSV row: the table's own six columns, then the row number and lon/lat.

    The first six keep their order and meaning, so a file written before this
    still reads and anything parsing by position is unaffected.

    lon/lat is left BLANK rather than zero for an end that has not been marked.
    A (0, 0) map coordinate transforms to a real place on Earth -- the Gulf of
    Guinea for a geographic CRS, some point offshore for a projected one -- and
    writing that would read as a measurement rather than an empty cell.
    """
    out = [("" if c is None else str(c)) for c in cells[:6]]
    out += [""] * (6 - len(out))

    def pair(ll):
        try:
            return [f"{float(ll[0]):.8f}", f"{float(ll[1]):.8f}"]
        except (TypeError, ValueError, IndexError):
            return ["", ""]

    return out + [str(index)] + pair(in_lonlat) + pair(ref_lonlat)

# ── PORTED FROM DPQED_rival.py (END) ──────────────────────────────────────────


# =============================================================================
# FOLDER WALK (mirrors QCDashboard._folder_entries)
# =============================================================================
def folder_entries(folder_path: str):
    """Rasters, sidecars and everything else under the folder, by basename.

    The tree is walked to REF_SCAN_DEPTH so imagery above a Meta/ subfolder is
    matched with its metadata. A name seen twice keeps the first and is
    reported, as RIVAL does."""
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
        print(f"[REFS] STOPPED after {REF_SCAN_MAX_FILES} files with "
              f"{len(rasters)} raster(s) kept -- point at a narrower folder.")
    if dupes:
        print(f"[REFS] {len(dupes)} duplicate filename(s) across subfolders, "
              f"first kept (e.g. {', '.join(sorted(set(dupes))[:3])})")
    return rasters, metas, others


# =============================================================================
# SHAPEFILE INDEX READING
# =============================================================================
def _ring_area(ring):
    a = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


def _read_prj(shp_path: str) -> Optional[str]:
    prj = os.path.splitext(shp_path)[0] + ".prj"
    if os.path.exists(prj):
        with open(prj, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip() or None
    return None


def _read_dbf(dbf_path: str, encoding: str = "utf-8") -> Tuple[List[str], List[Dict]]:
    with open(dbf_path, "rb") as f:
        data = f.read()
    n_rec = struct.unpack("<I", data[4:8])[0]
    hdr_len, rec_len = struct.unpack("<HH", data[8:12])
    fields, pos = [], 32
    while pos + 32 <= hdr_len and data[pos] != 0x0D:
        raw = data[pos:pos + 32]
        name = raw[:11].split(b"\x00", 1)[0].decode("ascii", "replace").strip()
        fields.append((name, chr(raw[11]), raw[16]))
        pos += 32
    rows = []
    for i in range(n_rec):
        off = hdr_len + i * rec_len
        rec = data[off:off + rec_len]
        if not rec or rec[:1] == b"*":
            rows.append(None)          # deleted: keep index alignment with .shp
            continue
        vals, p = {}, 1
        for name, ftype, flen in fields:
            raw = rec[p:p + flen]
            p += flen
            try:
                txt = raw.decode(encoding)
            except UnicodeDecodeError:
                txt = raw.decode("latin-1")
            txt = txt.strip()
            if ftype in "NF" and txt:
                try:
                    vals[name] = float(txt) if ("." in txt or "e" in txt.lower()) else int(txt)
                except ValueError:
                    vals[name] = txt
            else:
                vals[name] = txt
        rows.append(vals)
    return [f[0] for f in fields], rows


def _read_shp_polygons(shp_path: str) -> List[Optional[List[Tuple[float, float]]]]:
    """Polygon / PolygonZ / PolygonM records -> largest ring per record."""
    with open(shp_path, "rb") as f:
        data = f.read()
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type not in (5, 15, 25):
        raise ValueError(f"{os.path.basename(shp_path)}: shape type {shape_type} "
                         f"is not a polygon layer")
    out, pos = [], 100
    while pos + 8 <= len(data):
        _, content_words = struct.unpack(">ii", data[pos:pos + 8])
        content = data[pos + 8:pos + 8 + content_words * 2]
        pos += 8 + content_words * 2
        stype = struct.unpack("<i", content[:4])[0]
        if stype == 0:
            out.append(None)
            continue
        n_parts, n_pts = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts]))
        base = 44 + 4 * n_parts
        pts = [struct.unpack("<2d", content[base + 16 * k: base + 16 * k + 16])
               for k in range(n_pts)]
        rings = [pts[a:b] for a, b in zip(parts, parts[1:] + [n_pts])]
        rings = [r for r in rings if len(r) >= 3]
        out.append(max(rings, key=_ring_area) if rings else None)
    return out


def _to_lonlat_fn(crs_text: Optional[str]):
    """Coordinate transform to lon/lat for the layer's CRS (identity if it is
    already geographic or unknown-but-plausibly-lon/lat)."""
    if not crs_text:
        return None
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_user_input(crs_text)
        if crs.is_geographic:
            return None
        tr = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        return lambda x, y: tr.transform(x, y)
    except Exception as e:
        print(f"[REFS] CRS '{str(crs_text)[:60]}...' unusable ({e}); "
              f"assuming lon/lat")
        return None


def _read_index_builtin(shp_path: str):
    """Built-in .shp/.dbf reader (no GDAL needed)."""
    stem = os.path.splitext(shp_path)[0]
    dbf = next((stem + e for e in (".dbf", ".DBF") if os.path.exists(stem + e)), None)
    if dbf is None:
        raise FileNotFoundError(f"{shp_path}: no .dbf beside it")
    cpg = stem + ".cpg"
    enc = "utf-8"
    if os.path.exists(cpg):
        with open(cpg) as f:
            enc = (f.read().strip() or "utf-8").lower()
    names, rows = _read_dbf(dbf, enc)
    rings = _read_shp_polygons(shp_path)
    fn = _to_lonlat_fn(_read_prj(shp_path))
    feats = []
    for attrs, ring in zip(rows, rings):
        if attrs is None:
            continue
        if ring and fn:
            ring = [fn(x, y) for x, y in ring]
        feats.append((attrs, ring))
    return names, feats, "builtin"


def _read_index_fiona(shp_path: str):
    """fiona (rasterio ecosystem) reader."""
    import fiona
    with fiona.open(shp_path) as src:
        names = list(src.schema["properties"].keys())
        fn = _to_lonlat_fn(src.crs_wkt or _read_prj(shp_path))
        feats = []
        for feat in src:
            geom = feat["geometry"]
            ring = None
            if geom:
                coords = geom["coordinates"]
                polys = coords if geom["type"] == "MultiPolygon" else [coords]
                rings = [[(c[0], c[1]) for c in p[0]] for p in polys if p]
                ring = max(rings, key=_ring_area) if rings else None
            if ring and fn:
                ring = [fn(x, y) for x, y in ring]
            feats.append((dict(feat["properties"]), ring))
        return names, feats, "fiona"


def _read_index_ogr(shp_path: str):
    """GDAL/OGR reader, last resort."""
    from osgeo import ogr
    ds = ogr.Open(shp_path)
    if ds is None:
        raise ValueError("OGR could not open the layer")
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    srs = lyr.GetSpatialRef()
    fn = _to_lonlat_fn(srs.ExportToWkt() if srs else _read_prj(shp_path))
    feats = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        ring = None
        if geom is not None:
            polys = [geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] \
                if geom.GetGeometryName().upper().startswith("MULTI") else [geom]
            rings = []
            for p in polys:
                if p is not None and p.GetGeometryCount():
                    r = p.GetGeometryRef(0)
                    rings.append([(r.GetX(k), r.GetY(k)) for k in range(r.GetPointCount())])
            ring = max(rings, key=_ring_area) if rings else None
        if ring and fn:
            ring = [fn(x, y) for x, y in ring]
        feats.append(({n: feat.GetField(n) for n in names}, ring))
    return names, feats, "ogr"


def read_index_features(shp_path: str) -> Tuple[List[str], List[Tuple[Dict, list]], str]:
    """(field names, [(attributes, lon/lat ring)], reader used).

    The built-in reader goes first: an index is small, it needs no GDAL
    bindings, and it covers Polygon/PolygonZ/PolygonM. fiona and OGR are only
    tried if it cannot read the layer."""
    errors = []
    for reader in (_read_index_builtin, _read_index_fiona, _read_index_ogr):
        try:
            return reader(shp_path)
        except ImportError as e:
            errors.append(f"{reader.__name__}: not installed ({e})")
        except Exception as e:
            errors.append(f"{reader.__name__}: {type(e).__name__}: {e}")
    raise ValueError("; ".join(errors))


# =============================================================================
# SCANS (mirror QCDashboard._scan_sidecars / _scan_tile_names / _scan_index)
# =============================================================================
def _scan_sidecars(rasters, metas):
    footprints, errors = {}, []
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
            tif = match_raster(list(rasters), stem, rec.get("granule"))
            if not tif:
                errors.append(f"{meta_file}: no raster found for '{stem}'")
                continue
            rec["meta"] = meta_path
            footprints[rasters[tif]] = rec
        except Exception as e:
            errors.append(f"{meta_file}: {e}")
    return footprints, errors


def _scan_tile_names(rasters):
    footprints, errors = {}, []
    for name, path in sorted(rasters.items()):
        rec = parse_degree_tile(name)
        if rec is None:
            errors.append(f"{name}: name is not a degree tile "
                          f"(expected e.g. N16E73, N8E76, S34E018)")
            continue
        rec["meta"] = None
        footprints[path] = rec
    return footprints, errors


def _scan_index(rasters, candidates):
    footprints, errors = {}, []
    shp_name = pick_index_shapefile(list(candidates))
    shp_path = candidates[shp_name]
    try:
        field_names, feats, reader = read_index_features(shp_path)
    except Exception as e:
        return {}, [f"{shp_name}: not a readable vector layer ({e})"]
    fields = rank_name_fields(field_names)
    if not fields:
        return {}, [f"{shp_name}: no attributes to match raster names against"]

    lookup = build_name_lookup(rasters)
    name_field, unmatched = None, 0
    for attrs, ring in feats:
        if name_field is None:
            for field in fields:
                if resolve_index_name(attrs.get(field), lookup):
                    name_field = field
                    print(f"[REFS] {shp_name}: raster names from attribute "
                          f"'{field}' (reader: {reader})")
                    break
            if name_field is None:
                unmatched += 1
                continue
        tif = resolve_index_name(attrs.get(name_field), lookup)
        if not tif:
            unmatched += 1
            continue
        if not ring or len(ring) < 3:
            errors.append(f"{shp_name}: '{tif}' has no usable polygon")
            continue
        footprints[rasters[tif]] = {
            "ring": [tuple(p) for p in ring], "band": BAND_UNKNOWN, "crs": None,
            "granule": None, "source": f"index ({shp_name}:{name_field})",
            "meta": shp_path,
        }
    if name_field is None:
        errors.append(f"{shp_name}: no attribute matched any raster name. "
                      f"Fields are: " + ", ".join(fields[:12]))
    elif unmatched:
        print(f"[REFS] {shp_name}: {unmatched} index entr(ies) name a raster not "
              f"in this folder; {len(footprints)} matched")
    return footprints, errors


def scan_reference_folder(folder_path: str, mode: Optional[str] = None) -> Dict:
    """Discover every reference raster and its lon/lat footprint ring.

    Returns {'mode', 'footprints': {raster_path: record}, 'errors': [...],
    'n_rasters'}; each record carries 'ring' [(lon, lat), ...], 'source', 'band'.
    Raises when the folder is missing, holds no rasters, or no mode applies."""
    if not folder_path or not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Reference folder not found: {folder_path!r}")
    rasters, metas, others = folder_entries(folder_path)
    if not rasters:
        raise FileNotFoundError(f"No rasters ({', '.join(RASTER_EXTS)}) in "
                                f"{folder_path} or its subfolders")
    listing = list(rasters) + list(metas) + list(others)
    mode = mode or REF_MODE_OVERRIDE or detect_reference_mode(listing)
    if mode is None:
        raise ValueError(
            f"{len(rasters)} raster(s) in {folder_path}, but no footprints could "
            f"be read. Expected a shapefile index (e.g. Meta/index.shp), a "
            f"sidecar per raster ('.met', '.h5.iso.xml', '_meta.txt'), or "
            f"degree-tile names (e.g. N16E73.tif).")
    if mode == REF_MODE_INDEX:
        footprints, errors = _scan_index(rasters, {**metas, **others})
    elif mode == REF_MODE_SIDECAR:
        footprints, errors = _scan_sidecars(rasters, metas)
    elif mode == REF_MODE_TILE:
        footprints, errors = _scan_tile_names(rasters)
    else:
        raise ValueError(f"Unknown reference mode {mode!r}")
    print(f"[REFS] {os.path.basename(os.path.normpath(folder_path))}: "
          f"{len(rasters)} raster(s), {len(footprints)} footprint(s) from {mode}")
    for e in errors[:10]:
        print(f"[REFS]   {e}")
    if len(errors) > 10:
        print(f"[REFS]   ...and {len(errors) - 10} more")
    return {"mode": mode, "footprints": footprints, "errors": errors,
            "n_rasters": len(rasters)}


def reference_label(folder_path: str) -> str:
    """Short alphanumeric label for file names: 'L8_ref' -> 'L8ref'."""
    base = os.path.basename(os.path.normpath(folder_path or "")) or "REF"
    label = re.sub(r"[^A-Za-z0-9]", "", base)
    return label[:24] or "REF"
