"""Consolidate RIVAL results across scenes into one table.

RIVAL wrote its picks in several layouts over its versions -- In X / In_X /
x1-map headers, with or without lon/lat, with or without a statistics file --
so the statistics are recomputed here from the picks themselves, the same way
for every file, rather than trusted from whichever version wrote them.

For each results CSV named after a product id
    26<request id: 5 digits><product no.>1[ any suffix ].csv
(e.g. 266697511.csv, 266697511_rival.csv, 266697511_requestid.csv; never a
*_stats.csv), the product's BAND_META.txt is read from
    <input product folder>/<product id>/BAND_META.txt
for date, session id, strip id and scene centre, and one row is written per
product: ancillary info, then the targeting error -- mean, spread, RMSE, CE90
-- and where the picks sit (centre of the GCPs in lon/lat).

Special products hold several tifs in one folder, each with an XML beside it
named after the tif: <tif stem>_META.xml (AISD layout). A results CSV named
after such a tif (or its folder) -- <tif stem>_rival_band6.csv -- is matched
to that XML anywhere under the products folder, and sensor (SENSORNAME), date
(DATEOFPASS), session (SESSION_NUMBER), strip (STRIPID) and centre (mean of
the IMD corners) come from it. Both kinds land in one table; product_kind says
which.

Runs without QGIS: plain PyQt5 for the window, or headless:
    python rival_consolidate.py --results R --products P --out table.csv
"""
import csv
import math
import os
import re
import sys

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# 26 + 5-digit request id + product number + trailing 1, as a whole leading
# run of digits; whatever follows the digits is a suffix (_rival, _requestid).
PRODUCT_ID_RE = re.compile(r"^(26(\d{5})(\d+)1)(?!\d)")
SKIP_TOKEN = "_stats"
BAND_META_NAME = "BAND_META.txt"
META_XML_SUFFIX = "_META.xml"
# AISD XML tags read for special products: field -> tags tried in order
XML_TAGS = {
    "sensor":     ("SENSORNAME",),
    "date":       ("DATEOFPASS", "DATEOFIMAGE"),
    "session_id": ("SESSION_NUMBER",),
    "strip_id":   ("STRIPID",),
}
XML_CORNERS = {"lat": ("ULLAT", "URLAT", "LRLAT", "LLLAT"),
               "lon": ("ULLON", "URLON", "LRLON", "LLLON")}

# BAND_META keys are matched case-insensitively, ignoring _ - and spaces, on
# the first key that CONTAINS one of these words. Edit if a field is missed --
# the log names the key each value came from.
META_KEYS = {
    "date":       ("dateofpass", "acquisitiondate", "dateofacquisition",
                   "imagingdate", "passdate", "date"),
    # SessionNumber / StripNumber are the keys BAND_META actually uses; they
    # are tried exactly first, so no other key mentioning "session" or
    # "strip" can be taken for them.
    "sensor":     ("sensor",),
    "session_id": ("sessionnumber", "sessionid", "sessionno", "session"),
    "strip_id":   ("stripnumber", "stripid", "stripno", "strip"),
    "prod_code":  ("prodcode", "productcode"),
    "centre_lat": ("scenecenterlat", "scenecentrelat", "centerlat", "centrelat"),
    "centre_lon": ("scenecenterlon", "scenecentrelon", "centerlon", "centrelon"),
}
CORNER_KEYS = {  # fallback for the centre: mean of the product corners
    "lat": ("produllat", "produrlat", "prodlllat", "prodlrlat"),
    "lon": ("produllon", "produrlon", "prodlllon", "prodlrlon"),
}

# CE90 the way RIVAL computes it: 2.146 x sigma, sigma the RMS of the axes.
CE90_SIGMA = 2.146
CE90_MIN_AXIS_RATIO = 0.6

# Column names RIVAL has used for the picks, oldest to newest.
PICK_COLUMNS = {
    "ix": ("In X", "In_X", "x1-map"), "iy": ("In Y", "In_Y", "y1-map"),
    "rx": ("Ref X", "Ref_X", "x2-map"), "ry": ("Ref Y", "Ref_Y", "y2-map"),
    "in_lon": ("In_Lon",), "in_lat": ("In_Lat",),
    "ref_lon": ("Ref_Lon",), "ref_lat": ("Ref_Lat",),
}

OUT_COLUMNS = [
    "product_id", "request_id", "product_no", "band", "sensor", "date", "session_id",
    "strip_id", "prod_code", "product_kind", "source", "result_set", "date_source",
    "centre_lat", "centre_lon", "n_points",
    "mean_dx", "mean_dy", "std_dx", "std_dy", "rmse_dx", "rmse_dy",
    "mean_radial", "rmse_radial", "max_radial", "ce90", "axis_ratio",
    "ce90_circular", "error_frame", "gcp_mean_lat", "gcp_mean_lon", "results_csv", "band_meta",
]


# ── PURE HELPERS ──────────────────────────────────────────────────────────────
def product_id_of(filename):
    """(product_id, request_id, product_no) for a results CSV name, or None.

    None for anything that is not a .csv, a *_stats.csv, or a name that does
    not start with a 26<5 digits><n>1 product id.
    """
    stem, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() != ".csv" or SKIP_TOKEN in stem.lower():
        return None
    m = PRODUCT_ID_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


BAND_RE = re.compile(r"_band(\d+(?:-\d+)*)(?=[._]|$)", re.IGNORECASE)


def band_of(filename):
    """'6' for X_rival_band6.csv (RIVAL tags exports with the band marked on),
    '6-4-3' for a composite, '' for an untagged file."""
    m = BAND_RE.search(os.path.basename(filename))
    return m.group(1) if m else ""


def _norm(key):
    return re.sub(r"[\s_\-]", "", key).lower()


def parse_band_meta(text):
    """BAND_META text -> {normalised key: (original key, value)}.

    Takes 'Key = Value' and 'Key: Value' lines; the first occurrence wins.
    """
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*([^=:]+?)\s*[=:]\s*(.*?)\s*$", line)
        if not m or not m.group(1):
            continue
        k = _norm(m.group(1))
        if k and k not in out:
            out[k] = (m.group(1).strip(), m.group(2).strip())
    return out


def pick_meta(meta, words):
    """(value, original key) for the first word some key contains, or (None, None).

    Exact matches are tried before 'contains', so 'date' does not grab
    'processingdate' when 'dateofpass' is there.
    """
    for w in words:
        if w in meta:
            return meta[w][1], meta[w][0]
    for w in words:
        for k, (orig, v) in meta.items():
            if w in k:
                return v, orig
    return None, None


def _num(v):
    try:
        return float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def meta_ancillary(meta):
    """{date, session_id, strip_id, centre_lat, centre_lon} plus 'sources'."""
    out, sources = {}, {}
    for field, words in META_KEYS.items():
        v, k = pick_meta(meta, words)
        out[field] = v
        sources[field] = k
    for f in ("centre_lat", "centre_lon"):
        out[f] = _num(out[f])
    if out["centre_lat"] is None or out["centre_lon"] is None:
        lats = [_num(meta[k][1]) for k in CORNER_KEYS["lat"] if k in meta]
        lons = [_num(meta[k][1]) for k in CORNER_KEYS["lon"] if k in meta]
        lats = [v for v in lats if v is not None]
        lons = [v for v in lons if v is not None]
        if len(lats) == 4 and len(lons) == 4:
            out["centre_lat"], out["centre_lon"] = sum(lats) / 4, sum(lons) / 4
            sources["centre_lat"] = sources["centre_lon"] = "mean of Prod corners"
    out["sources"] = sources
    return out


def _col(row, names):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return row[n]
    return None


# Vincenty on WGS84, the same function as DPQED_rival.ground_offset (the tests
# check they agree): errors as true east/north metres, in no map grid.
_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563


def ground_offset(in_lonlat, ref_lonlat):
    """(east, north) metres from the reference to the input pick, on WGS84.

    Vincenty's inverse formula: the geodesic distance s and its azimuth a at
    the reference, so east = s sin a, north = s cos a -- the error as a
    surveyor on the ground at the reference would measure it, in no map grid.
    Agrees with PROJ's geodesic (pyproj.Geod) to well under a millimetre at
    these distances. None if the iteration does not converge (only for
    near-antipodal points, never two picks of one feature).
    """
    lon1, lat1 = map(math.radians, ref_lonlat)
    lon2, lat2 = map(math.radians, in_lonlat)
    a, f = _WGS84_A, _WGS84_F
    b = a * (1 - f)
    L = lon2 - lon1
    U1 = math.atan((1 - f) * math.tan(lat1))
    U2 = math.atan((1 - f) * math.tan(lat2))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        if ss == 0:
            return (0.0, 0.0)
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sigma = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a if c2a else 0.0
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        prev = lam
        lam = L + (1 - C) * f * sa * (sigma + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm * c2sm)))
        if abs(lam - prev) < 1e-12:
            break
    else:
        return None
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm * c2sm)
                                   - B / 6 * c2sm * (-3 + 4 * ss * ss) * (-3 + 4 * c2sm * c2sm)))
    dist = b * A * (sigma - ds)
    az = math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam))
    return (dist * math.sin(az), dist * math.cos(az))



def pick_errors(rows):
    """(errors, input lon/lats, frame) for the fully marked picks.

    A row with both ends in lon/lat (RIVAL wrote In_Lon..Ref_Lat from the CSV
    lon/lat version on) is measured on the ground, as RIVAL now measures it:
    true east/north metres, the same in every UTM zone a wide swath crosses.
    A row without them -- older CSVs -- can only be differenced on its grid.
    frame is "ground", "grid" or "mixed" for the file.
    """
    errs, lonlats, kinds = [], [], set()
    for row in rows:
        try:
            ix, iy = float(_col(row, PICK_COLUMNS["ix"])), float(_col(row, PICK_COLUMNS["iy"]))
            rx, ry = float(_col(row, PICK_COLUMNS["rx"])), float(_col(row, PICK_COLUMNS["ry"]))
        except (TypeError, ValueError):
            continue
        if (ix == 0 and iy == 0) or (rx == 0 and ry == 0):
            continue
        lon, lat = _num(_col(row, PICK_COLUMNS["in_lon"])), _num(_col(row, PICK_COLUMNS["in_lat"]))
        rlon, rlat = _num(_col(row, PICK_COLUMNS["ref_lon"])), _num(_col(row, PICK_COLUMNS["ref_lat"]))
        g = None
        if None not in (lon, lat, rlon, rlat):
            g = ground_offset((lon, lat), (rlon, rlat))
        if g is not None:
            errs.append(g); kinds.add("ground")
        else:
            errs.append((ix - rx, iy - ry)); kinds.add("grid")
        if lon is not None and lat is not None:
            lonlats.append((lon, lat))
    frame = "mixed" if len(kinds) > 1 else (kinds.pop() if kinds else "")
    return errs, lonlats, frame


def read_picks(rows):
    """Fully marked picks from CSV dict-rows: (errors [(dx,dy)], lonlats [(lon,lat)]).

    dx = In - Ref, recomputed from the coordinates; a row with either end
    unset (blank or 0,0) is not a measurement and is skipped.
    """
    errs, lonlats = [], []
    for row in rows:
        try:
            ix, iy = float(_col(row, PICK_COLUMNS["ix"])), float(_col(row, PICK_COLUMNS["iy"]))
            rx, ry = float(_col(row, PICK_COLUMNS["rx"])), float(_col(row, PICK_COLUMNS["ry"]))
        except (TypeError, ValueError):
            continue
        if (ix == 0 and iy == 0) or (rx == 0 and ry == 0):
            continue
        errs.append((ix - rx, iy - ry))
        lon, lat = _num(_col(row, PICK_COLUMNS["in_lon"])), _num(_col(row, PICK_COLUMNS["in_lat"]))
        if lon is not None and lat is not None:
            lonlats.append((lon, lat))
    return errs, lonlats


def error_summary(errors):
    """Mean, sample std, RMSE about zero, radial figures and CE90 -- as RIVAL."""
    xs = [e[0] for e in errors]
    ys = [e[1] for e in errors]
    n = len(xs)
    if n == 0:
        return {"n": 0}
    mean = lambda v: sum(v) / len(v)
    std = lambda v: (math.sqrt(sum((a - mean(v)) ** 2 for a in v) / (len(v) - 1))
                     if len(v) > 1 else 0.0)
    rmse_x = math.sqrt(sum(e * e for e in xs) / n)
    rmse_y = math.sqrt(sum(e * e for e in ys) / n)
    rad = [math.hypot(a, b) for a, b in zip(xs, ys)]
    hi = max(rmse_x, rmse_y)
    ratio = min(rmse_x, rmse_y) / hi if hi else 1.0
    return {
        "n": n, "mean_x": mean(xs), "mean_y": mean(ys),
        "std_x": std(xs), "std_y": std(ys), "rmse_x": rmse_x, "rmse_y": rmse_y,
        "mean_r": mean(rad), "rmse_r": math.sqrt(mean([r * r for r in rad])),
        "max_r": max(rad),
        "ce90": CE90_SIGMA * math.sqrt((rmse_x ** 2 + rmse_y ** 2) / 2.0),
        "axis_ratio": ratio, "circular": ratio >= CE90_MIN_AXIS_RATIO,
    }


def results_stem(filename):
    """The scene name a RIVAL export is named after: X_rival_band6.csv -> X."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.split(r"_rival", stem, maxsplit=1, flags=re.IGNORECASE)[0]


def index_meta_xml(products_folder):
    """{lower-case name: path} of every *_META.xml under products_folder.

    Keyed by the tif stem the XML is named after, and also by the folder it
    sits in -- RIVAL names an export after the folder (Load Input Folder) or
    the tif (Load Input TIF). A folder holding several XMLs is ambiguous and
    is not keyed.
    """
    idx, by_dir = {}, {}
    if not products_folder or not os.path.isdir(products_folder):
        return idx
    suf = META_XML_SUFFIX.lower()
    for root, _dirs, files in os.walk(products_folder):
        for f in files:
            if f.lower().endswith(suf):
                path = os.path.join(root, f)
                idx[f[:-len(suf)].lower()] = path
                by_dir.setdefault(os.path.basename(root).lower(), []).append(path)
    for d, paths in by_dir.items():
        if len(paths) == 1:
            idx.setdefault(d, paths[0])
    return idx


def parse_meta_xml(path):
    """meta_ancillary-shaped dict from a special product's AISD XML."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()

    def first(tag):
        for el in root.iter(tag):
            if el.text and el.text.strip():
                return el.text.strip()
        return None
    out, sources = {}, {}
    for field, tags in XML_TAGS.items():
        out[field], sources[field] = None, None
        for t in tags:
            v = first(t)
            if v is not None and v not in ("-9999", "-9999."):
                out[field], sources[field] = v, t
                break
    out["centre_lat"] = out["centre_lon"] = None
    imd = root.find(".//IMD")
    if imd is not None:
        lats = [_num(imd.findtext(t)) for t in XML_CORNERS["lat"]]
        lons = [_num(imd.findtext(t)) for t in XML_CORNERS["lon"]]
        if None not in lats and None not in lons:
            out["centre_lat"], out["centre_lon"] = sum(lats) / 4, sum(lons) / 4
            sources["centre_lat"] = sources["centre_lon"] = "mean of IMD corners"
    out["sources"] = sources
    return out


def _ids_for(name, meta_xml):
    """(ids, kind) for a file or folder name, or (None, None)."""
    ids = product_id_of(name if name.lower().endswith(".csv") else name + ".csv")
    if ids:
        return ids, "regular"
    if meta_xml:
        stem = results_stem(name)
        if stem.lower() in meta_xml:
            return (stem, "", ""), "special"
    return None, None


def find_results(folder, meta_xml=None, source="manual"):
    """Every results CSV under folder (recursively), keyed by
    (source, product id, band, result set).

    Values are ((product_id, request_id, product_no), path, kind). kind is
    'regular' for a 26<reqid><prodno>1 name, 'special' for a name whose scene
    has a *_META.xml in meta_xml (index_meta_xml).

    The product is read from the CSV's name (RIVAL: X_rival_band6.csv) or,
    failing that, from the nearest folder above it named after a product --
    the automatic matcher writes RIVAL_FINAL_band6.csv into a job folder
    named after the scene. Its several sets (FINAL, BEST_FINAL, DENSE ...)
    are kept apart as the result set. When one key has several CSVs the
    newest wins and the others are reported.
    """
    found, dupes = {}, []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            low = f.lower()
            if not low.endswith(".csv") or SKIP_TOKEN in low or low.endswith("_detail.csv"):
                continue
            ids, kind = _ids_for(f, meta_xml)
            if not ids:
                rel = os.path.relpath(root, folder)
                parts = [] if rel == "." else rel.split(os.sep)
                for part in reversed(parts):
                    ids, kind = _ids_for(part, meta_xml)
                    if ids:
                        break
            if not ids:
                continue
            path = os.path.join(root, f)
            stem = os.path.splitext(f)[0]
            rset = ""
            if not product_id_of(f) and not results_stem(f).lower() in (meta_xml or {}):
                rset = re.sub(r"_band[\d-]+", "", stem, flags=re.IGNORECASE)
            key = (source, ids[0], band_of(f), rset)
            if key in found:
                keep, drop = sorted([found[key][1], path], key=os.path.getmtime,
                                    reverse=True)
                dupes.append((key, drop))
                found[key] = (ids, keep, kind)
            else:
                found[key] = (ids, path, kind)
    return found, dupes


def _fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def product_folders(products):
    """A product folder, or several (a list, or ';'-separated), as a list of
    the existing ones."""
    if isinstance(products, str):
        products = [p for p in products.split(";")]
    return [p.strip() for p in (products or []) if p and p.strip()
            and os.path.isdir(p.strip())]


DATE_FORMATS = ("%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d%m%Y",
                "%Y%m%d", "%d-%B-%Y", "%d %b %Y")


def parse_date(text):
    """A date of pass as a datetime.date, from the forms metadata uses
    (12-MAR-2025, 11-11-2026, 2026-11-11 ...), or None."""
    from datetime import datetime
    t = " ".join(str(text or "").split())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(t.title() if "%b" in fmt or "%B" in fmt else t,
                                     fmt).date()
        except ValueError:
            continue
    return None


NAME_DATE_RE = re.compile(r"(?:^|_)(?:S\d{6}_E\d{6}_)?(\d{2})(\d{2})(20\d{2})(?=[A-Za-z_]|$)")


def date_from_path(path):
    """A date of pass read from the names along a path, when the metadata has
    none: a scene name such as ..._S062321_E063509_11112026DIP... (ddmmyyyy),
    in the CSV's name or any folder above it. 'dd-mm-yyyy' or None."""
    for part in reversed(re.split(r"[\\/]", path or "")):
        for m in NAME_DATE_RE.finditer(part):
            d = parse_date("".join(m.groups()))
            if d:
                return d.strftime("%d-%m-%Y")
    return None


def date_from_product_folder(folders, pid):
    """A date from the names of the files in the product's own folder
    (<product folder>/<id>/ ... *_11112026DIP*.tif), or None -- for a truth
    whose only link is its product folder id."""
    for f in folders or []:
        d0 = os.path.join(f, pid)
        if not os.path.isdir(d0):
            continue
        for root, _dirs, files in os.walk(d0):
            for n in sorted(files):
                d = date_from_path(n)
                if d:
                    return d
    return None


BY_DATE_COLUMNS = ["date", "source", "n_products", "n_points", "mean_dx", "mean_dy",
                   "std_dx", "std_dy", "rmse_dx", "rmse_dy", "rmse_radial",
                   "max_radial", "ce90", "products"]


def summarise_by_date(rows):
    """One row per date of pass (and source): every point of that day pooled,
    the same statistics as per product, and which products went into it."""
    groups = {}
    for r in rows:
        d = parse_date(r.get("date"))
        key = (d.isoformat() if d else "unknown", r.get("source") or "")
        g = groups.setdefault(key, {"errs": [], "products": []})
        g["errs"] += r.get("_errs") or []
        g["products"].append(r["product_id"] + (f" b{r['band']}" if r.get("band") else ""))
    out = []
    for (date, source), g in sorted(groups.items()):
        st = error_summary(g["errs"])
        row = {"date": date, "source": source, "n_products": len(g["products"]),
               "n_points": st["n"], "products": " ".join(g["products"])}
        if st["n"]:
            row.update({k: _fmt(st[v]) for k, v in (
                ("mean_dx", "mean_x"), ("mean_dy", "mean_y"), ("std_dx", "std_x"),
                ("std_dy", "std_y"), ("rmse_dx", "rmse_x"), ("rmse_dy", "rmse_y"),
                ("rmse_radial", "rmse_r"), ("max_radial", "max_r"), ("ce90", "ce90"))})
        out.append(row)
    return out


def consolidate(results_folder, products_folder, out_csv, log=print,
                date_from=None, date_to=None, by_date=False):
    """products_folder: one input product folder or several (up to the GUI's
    three) -- each product's metadata is looked for in all of them.

    date_from / date_to (dates or text, either may be None): keep only the
    products whose date of pass lies in that range. by_date: also write
    <out>_by_date.csv, one row per date of pass with all its points pooled."""
    folders = product_folders(products_folder)
    xml_index = {}
    for f in reversed(folders):                 # the first folder wins a clash
        xml_index.update(index_meta_xml(f))
    sources = (results_folder if isinstance(results_folder, dict)
               else {"manual": results_folder})
    return _consolidate(sources, folders, xml_index, out_csv, log,
                        date_from, date_to, by_date)


def _band_meta_path(folders, pid):
    for f in folders:
        p = os.path.join(f, pid, BAND_META_NAME)
        if os.path.isfile(p):
            return p
    return os.path.join(folders[0], pid, BAND_META_NAME) if folders else ""


def _consolidate(sources, folders, xml_index, out_csv, log=print,
                 date_from=None, date_to=None, by_date=False):
    """Build the table. Returns the number of products written."""
    found, dupes = {}, []
    for src_name, folder in sources.items():
        if folder and os.path.isdir(folder):
            f_, d_ = find_results(folder, xml_index, src_name)
            found.update(f_)
            dupes += d_
    results_folder = ", ".join(f"{k}: {v}" for k, v in sources.items() if v)
    for pid, drop in dupes:
        log(f"[DUP] {pid[0]} {pid[1]} band {pid[2] or '-'}: also {drop} -- the newer file is used")
    if not found:
        log(f"[WARN] no 26<reqid><prodno>1*.csv, nor a CSV named after a "
            f"*{META_XML_SUFFIX} product, under {results_folder}")
    rows = []
    for key in sorted(found):
        (pid, req, prodno), path, kind = found[key]
        source, band, rset = key[0], key[2], key[3]
        with open(path, newline="", encoding="utf-8-sig") as f:
            errs, lonlats, frame = pick_errors(csv.DictReader(f))
        st = error_summary(errs)
        meta_path = (xml_index.get(pid.lower(), "") if kind == "special"
                     else _band_meta_path(folders, pid))
        anc = {"sensor": None, "date": None, "session_id": None, "strip_id": None,
               "centre_lat": None, "centre_lon": None, "sources": {}}
        if kind == "special" and meta_path:
            try:
                anc = parse_meta_xml(meta_path)
                used = ", ".join(f"{k}<-{v}" for k, v in anc["sources"].items() if v)
                log(f"[META] {pid} (special): {used}")
            except Exception as e:
                log(f"[META] {pid}: cannot read {meta_path}: {e}")
        elif os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8", errors="replace") as f:
                anc = meta_ancillary(parse_band_meta(f.read()))
            missing = [k for k in ("date", "session_id", "strip_id",
                                   "centre_lat") if anc[k] is None]
            used = ", ".join(f"{k}<-{v}" for k, v in anc["sources"].items() if v)
            log(f"[META] {pid}: {used}"
                + (f"; MISSING {', '.join(missing)}" if missing else ""))
        else:
            log(f"[META] {pid}: no {meta_path}")
            meta_path = ""
        row = {
            "product_id": pid, "request_id": req, "product_no": prodno,
            "band": band, "product_kind": kind,
            "source": source, "result_set": rset,
            "date": (anc["date"] or date_from_path(path) or date_from_path(meta_path)
                     or date_from_product_folder(folders, pid)),
            "date_source": ("meta" if anc["date"] else
                            "name" if (date_from_path(path) or date_from_path(meta_path))
                            else "product folder files"
                            if date_from_product_folder(folders, pid) else ""),
            "session_id": anc["session_id"],
            "strip_id": anc["strip_id"], "sensor": anc["sensor"],
            "prod_code": anc.get("prod_code"),
            "centre_lat": _fmt(anc["centre_lat"], 6),
            "centre_lon": _fmt(anc["centre_lon"], 6),
            "n_points": st["n"], "results_csv": path, "band_meta": meta_path,
        }
        if st["n"]:
            row.update({
                "mean_dx": _fmt(st["mean_x"]), "mean_dy": _fmt(st["mean_y"]),
                "std_dx": _fmt(st["std_x"]), "std_dy": _fmt(st["std_y"]),
                "rmse_dx": _fmt(st["rmse_x"]), "rmse_dy": _fmt(st["rmse_y"]),
                "mean_radial": _fmt(st["mean_r"]), "rmse_radial": _fmt(st["rmse_r"]),
                "max_radial": _fmt(st["max_r"]), "ce90": _fmt(st["ce90"]),
                "axis_ratio": _fmt(st["axis_ratio"]), "error_frame": frame,
                "ce90_circular": "yes" if st["circular"] else "no",
            })
        if lonlats:
            row["gcp_mean_lon"] = _fmt(sum(p[0] for p in lonlats) / len(lonlats), 6)
            row["gcp_mean_lat"] = _fmt(sum(p[1] for p in lonlats) / len(lonlats), 6)
        row["_errs"] = errs
        rows.append(row)
        log(f"[OK] {pid}: n={st['n']}"
            + (f", CE90 {st['ce90']:.3f} m" if st["n"] else " (no fully marked rows)"))
    lo = parse_date(date_from) if isinstance(date_from, str) else date_from
    hi = parse_date(date_to) if isinstance(date_to, str) else date_to
    if lo or hi:
        kept = []
        for r in rows:
            d = parse_date(r.get("date"))
            if d and (not lo or d >= lo) and (not hi or d <= hi):
                kept.append(r)
        log(f"[DATE] {len(kept)} of {len(rows)} product(s) dated "
            f"{lo or '...'} .. {hi or '...'} (undated ones left out)")
        rows = kept
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    if by_date:
        bd = os.path.splitext(out_csv)[0] + "_by_date.csv"
        summ = summarise_by_date(rows)
        with open(bd, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=BY_DATE_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in summ:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in BY_DATE_COLUMNS})
        log(f"[DATE] {len(summ)} date(s) -> {bd}")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in OUT_COLUMNS})
    log(f"[DONE] {len(rows)} product(s) -> {out_csv}")
    return len(rows)


# ── GUI ───────────────────────────────────────────────────────────────────────
def run_gui():
    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLabel,
                                 QLineEdit, QPushButton, QFileDialog,
                                 QPlainTextEdit, QMessageBox)

    own_app = QApplication.instance() is None      # not inside QGIS
    app = QApplication.instance() or QApplication(sys.argv)
    settings = QSettings("NRSC", "RIVAL-consolidate")
    win = QWidget()
    win.setWindowTitle("RIVAL -- consolidate results")
    grid = QGridLayout(win)
    fields = {}

    def row(r, label, key, pick_file=False):
        grid.addWidget(QLabel(label), r, 0)
        edit = QLineEdit(str(settings.value(key, "") or ""))
        grid.addWidget(edit, r, 1)
        btn = QPushButton("Browse")

        def browse():
            if pick_file:
                p, _ = QFileDialog.getSaveFileName(win, label, edit.text(),
                                                   "CSV (*.csv)")
            else:
                p = QFileDialog.getExistingDirectory(win, label, edit.text())
            if p:
                edit.setText(p)
                settings.setValue(key, p)
        btn.clicked.connect(browse)
        grid.addWidget(btn, r, 2)
        fields[key] = edit

    row(0, "Manual results (RIVAL CSVs):", "results")
    row(1, "Auto results (matcher job folders, optional):", "auto")
    row(2, "Product folder 1 (<id>/BAND_META.txt, *_META.xml):", "products")
    row(3, "Product folder 2 (optional):", "products2")
    row(4, "Product folder 3 (optional):", "products3")
    row(5, "Consolidated CSV:", "out", pick_file=True)
    from PyQt5.QtWidgets import QCheckBox, QHBoxLayout
    drow = QHBoxLayout()
    drow.addWidget(QLabel("Date of pass from:"))
    d_from = QLineEdit(str(settings.value("date_from", "") or ""))
    d_from.setPlaceholderText("e.g. 01-11-2026 (blank: no limit)")
    drow.addWidget(d_from)
    drow.addWidget(QLabel("to:"))
    d_to = QLineEdit(str(settings.value("date_to", "") or ""))
    d_to.setPlaceholderText("e.g. 30-11-2026")
    drow.addWidget(d_to)
    by_date = QCheckBox("Also summarise date-wise (<out>_by_date.csv)")
    by_date.setChecked(str(settings.value("by_date", "false")).lower() == "true")
    drow.addWidget(by_date)
    grid.addLayout(drow, 8, 0, 1, 3)
    log_box = QPlainTextEdit()
    log_box.setReadOnly(True)
    grid.addWidget(log_box, 7, 0, 1, 3)

    def log(msg):
        log_box.appendPlainText(msg)
        QApplication.processEvents()

    def go():
        keys = ("results", "products", "products2", "products3", "out", "auto")
        res, p1, p2, p3, out, auto = (fields[k].text().strip() for k in keys)
        for k in keys:
            settings.setValue(k, fields[k].text().strip())
        prods = product_folders([p1, p2, p3])
        if not ((os.path.isdir(res) or os.path.isdir(auto)) and prods and out):
            QMessageBox.warning(win, "Consolidate",
                                "Choose a manual and/or an auto results folder, at "
                                "least one input product folder and the output CSV.")
            return
        df, dt = d_from.text().strip(), d_to.text().strip()
        for txt in (df, dt):
            if txt and not parse_date(txt):
                QMessageBox.warning(win, "Consolidate", f"Cannot read the date '{txt}'.")
                return
        settings.setValue("date_from", df)
        settings.setValue("date_to", dt)
        settings.setValue("by_date", "true" if by_date.isChecked() else "false")
        log_box.clear()
        log(f"[INFO] product folders: {', '.join(prods)}")
        try:
            n = consolidate({"manual": res, "auto": auto}, prods, out, log,
                            df or None, dt or None, by_date.isChecked())
        except Exception as e:
            QMessageBox.critical(win, "Consolidate", f"Failed:\n{e}")
            return
        QMessageBox.information(win, "Consolidate", f"{n} product(s) written to\n{out}")

    run = QPushButton("Consolidate")
    run.clicked.connect(go)
    grid.addWidget(run, 6, 0, 1, 3)
    win.resize(820, 480)
    win.show()
    if own_app:
        app.exec_()
    return win


if __name__ == "__main__":
    if "--results" in sys.argv:
        import argparse
        ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        ap.add_argument("--results", required=True, help="manual RIVAL CSVs")
        ap.add_argument("--auto", default="", help="automatic matcher results")
        ap.add_argument("--from", dest="date_from", default=None, help="date of pass from")
        ap.add_argument("--to", dest="date_to", default=None, help="date of pass to")
        ap.add_argument("--by-date", action="store_true",
                        help="also write <out>_by_date.csv")
        ap.add_argument("--products", required=True, nargs="+",
                        help="one or more input product folders")
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        consolidate({"manual": a.results, "auto": a.auto}, a.products, a.out,
                    date_from=a.date_from, date_to=a.date_to, by_date=a.by_date)
    else:
        run_gui()
