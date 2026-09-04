"""Exercise DPQED_rival's sidecar parsing without QGIS.

The pure helpers in DPQED_rival.py sit between explicit markers; this execs that
exact slice, so what is tested is the code that ships, not a copy of it.
"""
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
                 "META_SUFFIX_TEXT", "META_SUFFIX_XML"):
        m = re.search(rf"^{name}\s*=\s*(.+?)\s*(?:#.*)?$", text, re.M)
        assert m, f"constant {name} not found"
        consts[name] = eval(m.group(1))
    body = text[text.index(BEGIN):text.index(END)]
    ns = dict(consts, os=os, re=re, ET=ET, print=print)
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

# ── 2. SSAR: gdalinfo-style text sidecar (unchanged path) ─────────────────────
TEXT = """Driver: GTiff/GeoTIFF
Files: NISAR_SSAR_GSLC_demo1.tif
Size is 10980, 10980
Corner Coordinates:
Upper Left  (  77.0000000,  17.5000000)
Lower Left  (  77.0000000,  17.0000000)
Upper Right (  77.5000000,  17.5000000)
Lower Right (  77.5000000,  17.0000000)
"""
t = H["parse_meta_text"](TEXT, "demo_meta.txt")
check("text corners in UL,UR,LR,LL order", t["ring"],
      [(77.0, 17.5), (77.5, 17.5), (77.5, 17.0), (77.0, 17.0)])
check("text band from SSAR in name", t["band"], "SSAR")
check("text granule", t["granule"], "NISAR_SSAR_GSLC_demo1.tif")
check("text source", t["source"], "text")

# no band tag anywhere -> UNK, never a guess
plain = H["parse_meta_text"](TEXT.replace("SSAR", "XXXX"), "plain_meta.txt")
check("untagged text -> UNK", plain["band"], "UNK")

# centre frequency in a text sidecar is honoured too
freq = H["parse_meta_text"](
    TEXT.replace("SSAR", "XXXX") + "Center frequency: 3200000000.0\n", "f_meta.txt")
check("text band from 3.2 GHz", freq["band"], "SSAR")
freq_l = H["parse_meta_text"](
    TEXT.replace("SSAR", "XXXX") + "Center frequency: 1.239e9\n", "fl_meta.txt")
check("text band from 1.239 GHz", freq_l["band"], "LSAR")

# a sidecar missing a corner is rejected, not half-parsed
broken = H["parse_meta_text"](
    "\n".join(l for l in TEXT.splitlines() if "Lower Right" not in l), "bad_meta.txt")
check("incomplete corners rejected", broken, None)
check("malformed xml rejected", H["parse_meta_iso_xml"]("<not-xml", "x.iso.xml"), None)
check("xml without posList rejected",
      H["parse_meta_iso_xml"]("<a xmlns='urn:x'><b/></a>", "y.iso.xml"), None)

# ── 3. band helpers ───────────────────────────────────────────────────────────
check("band_from_frequency 1.239e9", H["band_from_frequency"](1.239e9), "LSAR")
check("band_from_frequency 1.2935e9", H["band_from_frequency"](1.2935e9), "LSAR")
check("band_from_frequency 3.2e9", H["band_from_frequency"](3.2e9), "SSAR")
check("band_from_frequency junk", H["band_from_frequency"]("n/a"), None)
check("band_from_frequency zero", H["band_from_frequency"](0), None)
check("band_from_name lsar lowercase", H["band_from_name"]("nisar_lsar_gslc.h5"), "LSAR")
check("band_from_name absent", H["band_from_name"]("cartosat_ortho.tif"), None)

# ── 4. posList shapes ─────────────────────────────────────────────────────────
check("posList comma triples",
      H["parse_pos_list"]("1 2 3,4 5 6,7 8 9"), [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)])
check("posList whitespace triples",
      H["parse_pos_list"]("1 2 3 4 5 6 7 8 9"), [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)])
check("posList whitespace pairs",
      H["parse_pos_list"]("1 2 3 4 5 6 7 8"),
      [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)])
check("posList empty", H["parse_pos_list"](""), [])

# ── 5. sidecar routing and raster matching ────────────────────────────────────
check("routes .iso.xml", H["is_meta_file"](xml_name), "iso-xml")
check("routes _meta.txt", H["is_meta_file"]("scene_meta.txt"), "text")
check("ignores others", H["is_meta_file"]("scene.tif"), None)

check("stem drops .h5.iso.xml", H["meta_base_stem"](xml_name),
      xml_name[:-len(".h5.iso.xml")])
check("stem drops _meta.txt", H["meta_base_stem"]("SCENE_A_meta.txt"), "SCENE_A")

files = ["SCENE_A.tif", "SCENE_B1.tif", "SCENE_B_meta.txt", "notes.txt"]
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
