"""Exercise DPQED_rival's sidecar parsing without QGIS.

The pure helpers in DPQED_rival.py sit between explicit markers; this execs that
exact slice, so what is tested is the code that ships, not a copy of it.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "DPQED_rival.py")

BEGIN = "# ── BEGIN PURE HELPERS"
END   = "# ── END PURE HELPERS"


def load_helpers():
    text = open(SRC, encoding="utf-8").read()
    consts = {}
    for name in ("BAND_SPLIT_HZ", "BAND_UNKNOWN",
                 "META_SUFFIXES_TEXT", "META_SUFFIX_XML"):
        m = re.search(rf"^{name}\s*=\s*(.+?)\s*(?:#.*)?$", text, re.M)
        assert m, f"constant {name} not found"
        consts[name] = eval(m.group(1))
    body = text[text.index(BEGIN):text.index(END)]
    ns = dict(consts, os=os, re=re, json=json, ET=ET, print=print)
    exec(compile(body, SRC, "exec"), ns)
    return ns


H = load_helpers()
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


# ── 1. LSAR: the real ASF ISO XML in this repo ────────────────────────────────
xml_name = ("NISAR_L2_PR_GSLC_028_084_A_010_4005_DHDH_A_"
            "20260819T001734_20260819T001808_P05023_N_F_J_001.h5.iso.xml")
xml_path = os.path.join(HERE, xml_name)
rec = H["parse_meta_iso_xml"](open(xml_path, encoding="utf-8").read(), xml_name)

check("iso band (centre freq 1.239 GHz)", rec["band"], "LSAR")
check("iso crs", rec["crs"], "EPSG:32644")
check("iso source", rec["source"], "iso-xml")
check("iso granule", rec["granule"], xml_name[:-len(".iso.xml")])
check("iso ring vertices (40 distinct + closing repeat)", len(rec["ring"]), 41)
check("iso distinct vertices", len(set(rec["ring"])), 40)
check("iso first vertex", rec["ring"][0], (79.4687993811434, 16.0411147145733))
check("iso ring is closed", rec["ring"][0] == rec["ring"][-1], True)
lons = [v[0] for v in rec["ring"]]
lats = [v[1] for v in rec["ring"]]
check("iso lon range plausible", 76.0 < min(lons) and max(lons) < 80.0, True)
check("iso lat range plausible", 15.0 < min(lats) and max(lats) < 19.0, True)

# The frame is a slanted swath: its convex bbox is much larger than the polygon.
# That is exactly why the ring is kept instead of four corners.
ring = H["close_ring"](rec["ring"])
area = abs(sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
               for i in range(len(ring) - 1))) / 2.0
bbox = (max(lons) - min(lons)) * (max(lats) - min(lats))
print(f"      ring area {area:.4f} deg^2 vs bbox {bbox:.4f} deg^2 "
      f"({100 * area / bbox:.1f}% of bbox)")
check("ring is tighter than its bbox", area < 0.8 * bbox, True)
check("iso wkt starts as polygon",
      H["ring_wkt"](rec["ring"]).startswith("POLYGON(("), True)

# ── 2. SSAR: the real '.met' JSON sidecar in this repo ────────────────────────
met_name = ("NISAR_S2_PR_GSLC_028_084_A_010_3700_DHNA_A_"
            "20260819T001733_20260819T001810_P00500_M_F_I_001.met")
met = H["parse_meta_text"](
    open(os.path.join(HERE, met_name), encoding="utf-8").read(), met_name)

check("met band (Sensor field)", met["band"], "SSAR")
check("met crs (EPSG field)", met["crs"], "EPSG:32644")
check("met source prefers the swath", met["source"], "met-json (image)")
check("met granule", met["granule"], met_name[:-len(".met")] + ".h5")
check("met ring is the 4 Image corners", met["ring"],
      [(76.534748, 17.614687), (78.827076, 18.171194),
       (79.385351, 15.993327), (77.112174, 15.446038)])

# Image* is the slanted swath; Prod* is the north-up grid the raster spans.
# Preferring Image* is what stops the tool offering a tile whose data does not
# reach the picked point.
def shoelace(r):
    r = H["close_ring"](r)
    return abs(sum(r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1]
                   for i in range(len(r) - 1))) / 2.0

met_obj = json.load(open(os.path.join(HERE, met_name), encoding="utf-8"))
prod = H["parse_meta_json"](
    {k: v for k, v in met_obj.items() if not k.startswith("Image")}, met_name)
check("Prod* used when Image* absent", prod["source"], "met-json (product-grid)")
print(f"      Image swath {shoelace(met['ring']):.4f} deg^2 vs "
      f"Prod grid {shoelace(prod['ring']):.4f} deg^2")
check("swath is tighter than the product grid",
      shoelace(met["ring"]) < 0.8 * shoelace(prod["ring"]), True)

check("met missing all corners rejected",
      H["parse_meta_json"]({"Sensor": "SSAR"}, "empty.met"), None)
check("malformed JSON rejected",
      H["parse_meta_text"]('{"Sensor": ', "bad.met"), None)

# ── 3. legacy gdalinfo-style text sidecar (unchanged path) ────────────────────
TEXT = """Driver: GTiff/GeoTIFF
Files: NISAR_SSAR_GSLC_demo1.tif
Size is 10980, 10980
Corner Coordinates:
Upper Left  (  77.0000000,  17.5000000)
Lower Left  (  77.0000000,  17.0000000)
Upper Right (  77.5000000,  17.5000000)
Lower Right (  77.5000000,  17.0000000)
"""
t = H["parse_meta_text"](TEXT, "NISAR_SSAR_GSLC_demo.met")
check("text corners in UL,UR,LR,LL order", t["ring"],
      [(77.0, 17.5), (77.5, 17.5), (77.5, 17.0), (77.0, 17.0)])
check("text band from SSAR in name", t["band"], "SSAR")
check("text granule", t["granule"], "NISAR_SSAR_GSLC_demo1.tif")
check("text source", t["source"], "text")

# no band tag anywhere -> UNK, never a guess
plain = H["parse_meta_text"](TEXT.replace("SSAR", "XXXX"), "plain.met")
check("untagged text -> UNK", plain["band"], "UNK")

# centre frequency in a text sidecar is honoured too
freq = H["parse_meta_text"](
    TEXT.replace("SSAR", "XXXX") + "Center frequency: 3200000000.0\n", "f.met")
check("text band from 3.2 GHz", freq["band"], "SSAR")
freq_l = H["parse_meta_text"](
    TEXT.replace("SSAR", "XXXX") + "Center frequency: 1.239e9\n", "fl.met")
check("text band from 1.239 GHz", freq_l["band"], "LSAR")

# a sidecar missing a corner is rejected, not half-parsed
broken = H["parse_meta_text"](
    "\n".join(l for l in TEXT.splitlines() if "Lower Right" not in l), "bad.met")
check("incomplete corners rejected", broken, None)
check("malformed xml rejected", H["parse_meta_iso_xml"]("<not-xml", "x.iso.xml"), None)
check("xml without posList rejected",
      H["parse_meta_iso_xml"]("<a xmlns='urn:x'><b/></a>", "y.iso.xml"), None)

# ── 4. band helpers ──────────────────────────────────────────────────────────
check("band_from_frequency 1.239e9", H["band_from_frequency"](1.239e9), "LSAR")
check("band_from_frequency 1.2935e9", H["band_from_frequency"](1.2935e9), "LSAR")
check("band_from_frequency 3.2e9", H["band_from_frequency"](3.2e9), "SSAR")
check("band_from_frequency junk", H["band_from_frequency"]("n/a"), None)
check("band_from_frequency zero", H["band_from_frequency"](0), None)
check("band_from_name lsar lowercase", H["band_from_name"]("nisar_lsar_gslc.h5"), "LSAR")
check("band_from_name absent", H["band_from_name"]("cartosat_ortho.tif"), None)

# ── 5. posList shapes ─────────────────────────────────────────────────────────
check("posList comma triples",
      H["parse_pos_list"]("1 2 3,4 5 6,7 8 9"), [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)])
check("posList whitespace triples",
      H["parse_pos_list"]("1 2 3 4 5 6 7 8 9"), [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)])
check("posList whitespace pairs",
      H["parse_pos_list"]("1 2 3 4 5 6 7 8"),
      [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)])
check("posList empty", H["parse_pos_list"](""), [])

# ── 6. sidecar routing and raster matching ────────────────────────────────────
check("routes .iso.xml", H["is_meta_file"](xml_name), "iso-xml")
check("routes .met", H["is_meta_file"]("SCENE_A.met"), "text")
check("routes legacy _meta.txt", H["is_meta_file"]("scene_meta.txt"), "text")
check("ignores the .h5 itself", H["is_meta_file"]("SCENE_A.h5"), None)
check("ignores others", H["is_meta_file"]("scene.tif"), None)

# '<product>.h5' and '<product>.met' share a stem; the XML sidecar hangs off the
# .h5 name, so both must reduce to the same thing.
check("stem drops .h5.iso.xml", H["meta_base_stem"](xml_name),
      xml_name[:-len(".h5.iso.xml")])
check("stem drops .met", H["meta_base_stem"]("SCENE_A.met"), "SCENE_A")
check(".met and .h5.iso.xml agree on the stem",
      H["meta_base_stem"]("SCENE_A.met") == H["meta_base_stem"]("SCENE_A.h5.iso.xml"),
      True)
check("stem drops legacy _meta.txt", H["meta_base_stem"]("SCENE_A_meta.txt"), "SCENE_A")

files = ["SCENE_A.tif", "SCENE_B1.tif", "SCENE_B.met", "SCENE_B.h5", "notes.txt"]
check("exact stem match", H["match_raster"](files, "SCENE_A"), "SCENE_A.tif")
check("h52tif '<product>1.tif' prefix match",
      H["match_raster"](files, "SCENE_B"), "SCENE_B1.tif")
check("granule name fallback",
      H["match_raster"](["OTHER.tif"], "NOPE", "OTHER.h5"), "OTHER.tif")
check("no raster -> None", H["match_raster"](files, "SCENE_Z"), None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
