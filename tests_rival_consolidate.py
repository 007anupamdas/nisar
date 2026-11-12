"""Checks for rival_consolidate.py -- no Qt, no QGIS needed."""
import csv
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rival_consolidate as C
from tests_rival_meta import load_helpers   # RIVAL's own pure helpers

failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


print("-- which CSVs are results --")
f = C.product_id_of
check("bare id", f("266697511.csv"), ("266697511", "66975", "1"))
check("_rival suffix", f("266697511_rival.csv"), ("266697511", "66975", "1"))
check("_requestid suffix", f("266697521_requestid.csv"), ("266697521", "66975", "2"))
check("a two-digit product number", f("2666975121.csv"), ("2666975121", "66975", "12"))
check("_stats is never a result", f("266697511_rival_stats.csv"), None)
check("not ending in 1", f("266697512.csv"), None)
check("not starting with 26", f("366697511.csv"), None)
check("not a csv", f("266697511.shp"), None)
check("an unrelated csv", f("picks.csv"), None)

print("\n-- BAND_META --")
meta = C.parse_band_meta(
    "DateOfPass = 12-MAR-2025\nSessionID: 4471\nStrip_Id = 3\n"
    "ProcessingDate = 20-MAR-2025\nProdULLat=17.0\nProdULLon=78.0\n"
    "ProdURLat=17.0\nProdURLon=79.0\nProdLLLat=16.0\nProdLLLon=78.0\n"
    "ProdLRLat=16.0\nProdLRLon=79.0\n")
a = C.meta_ancillary(meta)
check("date is the pass date, not the processing date", a["date"], "12-MAR-2025")
check("session id", a["session_id"], "4471")
check("strip id", a["strip_id"], "3")
check("centre falls back to the corner mean", (a["centre_lat"], a["centre_lon"]),
      (16.5, 78.5))
real = C.meta_ancillary(C.parse_band_meta(
    "SessionType = DAY\nSessionNumber = 4471\nStripCount = 9\n"
    "StripNumber = 3\n"))
check("SessionNumber, not SessionType", real["session_id"], "4471")
check("StripNumber, not StripCount", real["strip_id"], "3")
check("and the log names the real keys",
      (real["sources"]["session_id"], real["sources"]["strip_id"]),
      ("SessionNumber", "StripNumber"))
a2 = C.meta_ancillary(C.parse_band_meta("SceneCenterLat = 16.25\nSceneCenterLon = 78.75"))
check("a stated scene centre wins", (a2["centre_lat"], a2["centre_lon"]), (16.25, 78.75))
check("absent fields stay empty", a2["date"], None)

print("\n-- picks from every RIVAL layout --")
old = [{"x1-map": "1003", "y1-map": "1004", "x2-map": "1000", "y2-map": "1000"},
       {"x1-map": "2005", "y1-map": "2000", "x2-map": "2000", "y2-map": "2000"},
       {"x1-map": "3000", "y1-map": "3000", "x2-map": "0", "y2-map": "0"}]
new = [{"In_X": "1003", "In_Y": "1004", "Ref_X": "1000", "Ref_Y": "1000",
        "In_Lon": "78.1", "In_Lat": "16.1"},
       {"In_X": "2005", "In_Y": "2000", "Ref_X": "2000", "Ref_Y": "2000",
        "In_Lon": "78.3", "In_Lat": "16.3"},
       {"In_X": "", "In_Y": "", "Ref_X": "5", "Ref_Y": "5"}]
check("oldest headers, half-marked row skipped", C.read_picks(old)[0],
      [(3.0, 4.0), (5.0, 0.0)])
check("current headers, lon/lat kept", C.read_picks(new),
      ([(3.0, 4.0), (5.0, 0.0)], [(78.1, 16.1), (78.3, 16.3)]))

H = load_helpers()
mine = C.error_summary([(3.0, 4.0), (5.0, 0.0)])
theirs = H["error_summary"]([(3.0, 4.0), (5.0, 0.0)])
for k in ("mean_x", "mean_y", "std_x", "std_y", "rmse_x", "rmse_y", "ce90",
          "mean_r", "rmse_r", "max_r", "axis_ratio", "circular"):
    check(f"{k} matches RIVAL's own statistics", abs(mine[k] - theirs[k]) < 1e-12
          if isinstance(mine[k], float) else mine[k] == theirs[k], True)

print("\n-- errors on the ground when both ends have lon/lat --")
both = [{"In_X": "1", "In_Y": "1", "Ref_X": "2", "Ref_Y": "2",
         "In_Lon": "86.50285883468196", "In_Lat": "16.367465862502733",
         "Ref_Lon": "86.5", "Ref_Lat": "17.0"}]
e, _, frame = C.pick_errors(both)
check("east/north from lon/lat, not the grid columns",
      (round(e[0][0], 2), round(e[0][1], 2), frame), (305.43, -69999.93, "ground"))
check("same function as RIVAL's", C.ground_offset((86.5029, 16.3675), (86.5, 17.0)),
      H["ground_offset"]((86.5029, 16.3675), (86.5, 17.0)))
check("an old CSV without lon/lat is differenced on its grid",
      C.pick_errors(old)[2], "grid")
check("a file with both kinds says so", C.pick_errors(old + both)[2], "mixed")

print("\n-- end to end --")
res, prod = tempfile.mkdtemp(), tempfile.mkdtemp()
os.makedirs(os.path.join(res, "csv"))
def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
write(os.path.join(res, "266697511.csv"), old)
write(os.path.join(res, "csv", "266697521_rival.csv"), new)
write(os.path.join(res, "csv", "266697521_rival_stats.csv"),
      [{"Statistic": "ce90", "Value": "999"}])
write(os.path.join(res, "picks.csv"), old)
os.makedirs(os.path.join(prod, "266697511"))
open(os.path.join(prod, "266697511", "BAND_META.txt"), "w").write(
    "Sensor = LISS4\nSensorMode = MX\n"
    "DateOfPass = 12-MAR-2025\nSessionID = 4471\nStripId = 3\n"
    "SceneCenterLat = 16.5\nSceneCenterLon = 78.5\n")
out = os.path.join(res, "out", "all.csv")
logs = []
n = C.consolidate(res, prod, out, logs.append)
rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
check("two products, stats file and stray csv ignored", (n, [r["product_id"] for r in rows]),
      (2, ["266697511", "266697521"]))
r1, r2 = rows
check("ancillary from BAND_META", (r1["date"], r1["session_id"], r1["strip_id"],
                                   r1["centre_lat"]),
      ("12-MAR-2025", "4471", "3", "16.500000"))
check("Sensor from its own key, not SensorMode", r1["sensor"], "LISS4")
check("sensor sits beside the other ancillary columns",
      C.OUT_COLUMNS[:6], ["product_id", "request_id", "product_no", "band",
                          "sensor", "date"])
check("targeting mean and CE90 from the picks", (r1["n_points"], r1["mean_dx"],
                                                 r1["mean_dy"], r1["ce90"]),
      ("2", "4.000", "2.000", f"{H['error_summary']([(3, 4), (5, 0)])['ce90']:.3f}"))
check("a product without BAND_META is still listed, ancillary blank",
      (r2["date"], r2["band_meta"], r2["n_points"]), ("", "", "2"))
check("GCP centre in lon/lat when the CSV has it",
      (r2["gcp_mean_lon"], r2["gcp_mean_lat"]), ("78.200000", "16.200000"))
check("the missing BAND_META is reported", any("no " in l and "266697521" in l
                                               for l in logs), True)
shutil.rmtree(res, ignore_errors=True); shutil.rmtree(prod, ignore_errors=True)

print()
# ── per-band results stay separate ─────────────────────────────────────────
check("band from a RIVAL name", (C.band_of("2666975121_rival_band6.csv"),
      C.band_of("P_rival_band6-4-3.csv"), C.band_of("2666975121_rival.csv")),
      ("6", "6-4-3", ""))
res2 = tempfile.mkdtemp()
write(os.path.join(res2, "266697511_rival_band1.csv"), old)
write(os.path.join(res2, "266697511_rival_band6.csv"), new)
out2 = os.path.join(res2, "out.csv")
n2 = C.consolidate(res2, prod, out2, lambda *a: None)
rows2 = list(csv.DictReader(open(out2, encoding="utf-8-sig")))
check("two bands of one product are two rows, not a duplicate",
      (n2, [(r["product_id"], r["band"]) for r in rows2]),
      (2, [("266697511", "1"), ("266697511", "6")]))

# ── special products: several tifs per folder, <tif stem>_META.xml beside each
SP = "G1AHS026111104EX000006_S062321_E063509_11112026DIP_DQ_ort"
XML = """<?xml version="1.0" encoding="utf-8"?>
<AISD><IMD><ULLON>71.844418</ULLON><ULLAT>28.351090</ULLAT>
<URLON>88.926815</URLON><URLAT>28.351090</URLAT><LRLON>88.926815</LRLON>
<LRLAT>25.532447</LRLAT><LLLON>71.844418</LLLON><LLLAT>25.532447</LLLAT></IMD>
<IMAGE><SATID>G1A</SATID><SENSORNAME>HS</SENSORNAME><DATEOFPASS>11-11-2026</DATEOFPASS>
<STRIPID>000006</STRIPID><SESSION_NUMBER>026111104</SESSION_NUMBER>
<IMAGING_ORBIT>-9999</IMAGING_ORBIT></IMAGE></AISD>"""
res3, prod3 = tempfile.mkdtemp(), tempfile.mkdtemp()
d = os.path.join(prod3, "batch_11nov", "HSfolder")
os.makedirs(d)
open(os.path.join(d, SP + "_META.xml"), "w").write(XML)
open(os.path.join(d, "OTHER_scene_META.xml"), "w").write(XML.replace("000006", "000007"))
os.makedirs(os.path.join(prod3, "266697511"))
open(os.path.join(prod3, "266697511", "BAND_META.txt"), "w").write(
    "Sensor = LISS4\nDateOfPass = 12-MAR-2025\nSessionID = 4471\nStripId = 3\n")
os.makedirs(os.path.join(res3, "csv"))
write(os.path.join(res3, "csv", SP + "_rival_band6.csv"), new)
write(os.path.join(res3, "csv", "266697511_rival_band2.csv"), old)
write(os.path.join(res3, "csv", "unrelated_rival.csv"), old)
write(os.path.join(res3, "csv", SP + "_rival_band6_stats.csv"),
      [{"Statistic": "ce90", "Value": "1"}])
idx = C.index_meta_xml(prod3)
check("XML indexed by tif stem; a folder with two XMLs is not a key",
      (SP.lower() in idx, "hsfolder" in idx), (True, False))
a = C.parse_meta_xml(idx[SP.lower()])
check("special ancillary from the XML",
      (a["sensor"], a["date"], a["session_id"], a["strip_id"],
       round(a["centre_lat"], 6), round(a["centre_lon"], 6)),
      ("HS", "11-11-2026", "026111104", "000006", 26.941769, 80.385616))
out3 = os.path.join(res3, "all.csv")
n3 = C.consolidate(res3, prod3, out3, lambda *a: None)
rows3 = {r["product_id"]: r for r in csv.DictReader(open(out3, encoding="utf-8-sig"))}
check("regular and special products in one table; stats and unknown CSVs skipped",
      (n3, sorted((k, r["product_kind"], r["band"]) for k, r in rows3.items())),
      (2, sorted([("266697511", "regular", "2"), (SP, "special", "6")])))
r = rows3[SP]
check("special row carries the XML fields and path",
      (r["sensor"], r["date"], r["session_id"], r["strip_id"], r["n_points"],
       r["band_meta"].endswith(SP + "_META.xml")),
      ("HS", "11-11-2026", "026111104", "000006", "2", True))
check("regular row unchanged", (rows3["266697511"]["sensor"],
      rows3["266697511"]["product_kind"]), ("LISS4", "regular"))
# a lone XML in its folder also keys the folder: export named after the folder
prod4 = tempfile.mkdtemp(); os.makedirs(os.path.join(prod4, "HS_ONE"))
open(os.path.join(prod4, "HS_ONE", SP + "_META.xml"), "w").write(XML)
check("folder-named export matches a folder holding one XML",
      "hs_one" in C.index_meta_xml(prod4), True)

if failures:
    print(f"{len(failures)} FAILURE(S)")
    for x in failures:
        print("  -", x)
    sys.exit(1)
print("all checks passed")
