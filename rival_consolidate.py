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

# BAND_META keys are matched case-insensitively, ignoring _ - and spaces, on
# the first key that CONTAINS one of these words. Edit if a field is missed --
# the log names the key each value came from.
META_KEYS = {
    "date":       ("dateofpass", "acquisitiondate", "dateofacquisition",
                   "imagingdate", "passdate", "date"),
    "session_id": ("sessionid", "session"),
    "strip_id":   ("stripid", "strip"),
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
}

OUT_COLUMNS = [
    "product_id", "request_id", "product_no", "date", "session_id", "strip_id",
    "centre_lat", "centre_lon", "n_points",
    "mean_dx", "mean_dy", "std_dx", "std_dy", "rmse_dx", "rmse_dy",
    "mean_radial", "rmse_radial", "max_radial", "ce90", "axis_ratio",
    "ce90_circular", "gcp_mean_lat", "gcp_mean_lon", "results_csv", "band_meta",
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


def find_results(folder):
    """Every results CSV under folder (recursively), keyed by product id.

    When one product has several CSVs -- written by different RIVAL versions --
    the most recently modified wins, and the others are reported.
    """
    found, dupes = {}, []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            ids = product_id_of(f)
            if not ids:
                continue
            path = os.path.join(root, f)
            pid = ids[0]
            if pid in found:
                keep, drop = sorted([found[pid][1], path], key=os.path.getmtime,
                                    reverse=True)
                dupes.append((pid, drop))
                found[pid] = (ids, keep)
            else:
                found[pid] = (ids, path)
    return found, dupes


def _fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def consolidate(results_folder, products_folder, out_csv, log=print):
    """Build the table. Returns the number of products written."""
    found, dupes = find_results(results_folder)
    for pid, drop in dupes:
        log(f"[DUP] {pid}: also {drop} -- the newer file is used")
    if not found:
        log(f"[WARN] no 26<reqid><prodno>1*.csv under {results_folder}")
    rows = []
    for pid in sorted(found):
        (pid, req, prodno), path = found[pid]
        with open(path, newline="", encoding="utf-8-sig") as f:
            errs, lonlats = read_picks(csv.DictReader(f))
        st = error_summary(errs)
        meta_path = os.path.join(products_folder, pid, BAND_META_NAME)
        anc = {"date": None, "session_id": None, "strip_id": None,
               "centre_lat": None, "centre_lon": None, "sources": {}}
        if os.path.isfile(meta_path):
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
            "date": anc["date"], "session_id": anc["session_id"],
            "strip_id": anc["strip_id"],
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
                "axis_ratio": _fmt(st["axis_ratio"]),
                "ce90_circular": "yes" if st["circular"] else "no",
            })
        if lonlats:
            row["gcp_mean_lon"] = _fmt(sum(p[0] for p in lonlats) / len(lonlats), 6)
            row["gcp_mean_lat"] = _fmt(sum(p[1] for p in lonlats) / len(lonlats), 6)
        rows.append(row)
        log(f"[OK] {pid}: n={st['n']}"
            + (f", CE90 {st['ce90']:.3f} m" if st["n"] else " (no fully marked rows)"))
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
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

    row(0, "Results folder (RIVAL CSVs):", "results")
    row(1, "Input product folder (<id>/BAND_META.txt):", "products")
    row(2, "Consolidated CSV:", "out", pick_file=True)
    log_box = QPlainTextEdit()
    log_box.setReadOnly(True)
    grid.addWidget(log_box, 4, 0, 1, 3)

    def log(msg):
        log_box.appendPlainText(msg)
        QApplication.processEvents()

    def go():
        res, prod, out = (fields[k].text().strip() for k in ("results", "products", "out"))
        for k in ("results", "products", "out"):
            settings.setValue(k, fields[k].text().strip())
        if not (os.path.isdir(res) and os.path.isdir(prod) and out):
            QMessageBox.warning(win, "Consolidate",
                                "Choose the results folder, the input product "
                                "folder and the output CSV.")
            return
        log_box.clear()
        try:
            n = consolidate(res, prod, out, log)
        except Exception as e:
            QMessageBox.critical(win, "Consolidate", f"Failed:\n{e}")
            return
        QMessageBox.information(win, "Consolidate", f"{n} product(s) written to\n{out}")

    run = QPushButton("Consolidate")
    run.clicked.connect(go)
    grid.addWidget(run, 3, 0, 1, 3)
    win.resize(820, 480)
    win.show()
    if own_app:
        app.exec_()
    return win


if __name__ == "__main__":
    if "--results" in sys.argv:
        import argparse
        ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        ap.add_argument("--results", required=True)
        ap.add_argument("--products", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        consolidate(a.results, a.products, a.out)
    else:
        run_gui()
