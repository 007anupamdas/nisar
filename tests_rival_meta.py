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
    # Every module-level UPPER_CASE literal the helpers may reference. Picked up
    # generically so a new constant does not need adding here by hand.
    consts = {}
    for name, expr in re.findall(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$",
                                 text, re.M):
        try:
            consts[name] = eval(expr, {"__builtins__": {}}, dict(consts))
        except Exception:
            pass
    for required in ("BAND_SPLIT_HZ", "BAND_UNKNOWN", "META_SUFFIXES_TEXT",
                     "META_SUFFIX_XML", "DEGREE_TILE_SIZE", "RASTER_EXTS"):
        assert required in consts, f"constant {required} not found"
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
check("met granule is the bare product id", met["granule"],
      met_name[:-len(".met")])
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
check("an .h5 alone is not a raster",
      H["match_raster"](["SCENE_B.h5"], "SCENE_B", "SCENE_B.h5"), None)

# What sits in the reference folder is the '.tif'; the '.h5' the metadata names
# need not exist. Pin every naming shape either band's raster turns up as.
for label, meta_name, rec_for in (
        ("LSAR", xml_name, rec),
        ("SSAR", met_name, met)):
    stem = H["meta_base_stem"](meta_name)
    for shape in (".tif", ".h5.tif", "1.tif", ".TIF"):
        folder_tif = stem + shape
        check(f"{label} raster named '<product>{shape}'",
              H["match_raster"]([folder_tif, "unrelated.tif"], stem,
                                rec_for["granule"]),
              folder_tif)

# ── 7. C1: the footprint is the tile name ────────────────────────────────────
# 'N16E73.tif' is the cell whose SOUTH-WEST corner is 16 N, 73 E.
tile = H["parse_degree_tile"]("N16E73.tif")
check("N16E73 ring (UL,UR,LR,LL)", tile["ring"],
      [(73.0, 17.0), (74.0, 17.0), (74.0, 16.0), (73.0, 16.0)])
check("tile source", tile["source"], "tile-name (1 deg)")
check("tile has no band", tile["band"], "UNK")
check("three-digit lon", H["parse_degree_tile"]("N16E073.tif")["ring"][0], (73.0, 17.0))
check("southern hemisphere", H["parse_degree_tile"]("S34E018.tif")["ring"][3],
      (18.0, -34.0))
check("western hemisphere", H["parse_degree_tile"]("N40W105.tif")["ring"][3],
      (-105.0, 40.0))
check("suffix after the token is ignored",
      H["parse_degree_tile"]("N16E73_ORTHO_v2.tif")["ring"][3], (73.0, 16.0))

# Reported from a real C1 folder: degree counts are not zero-padded.
check("unpadded latitude 'N8E76_ortho.tif'",
      H["parse_degree_tile"]("N8E76_ortho.tif")["ring"],
      [(76.0, 9.0), (77.0, 9.0), (77.0, 8.0), (76.0, 8.0)])
check("unpadded latitude, padded longitude",
      H["parse_degree_tile"]("N8E076.tif")["ring"][3], (76.0, 8.0))
check("unpadded longitude", H["parse_degree_tile"]("N16E7.tif")["ring"][3],
      (7.0, 16.0))
check("lowercase token", H["parse_degree_tile"]("n8e76_ortho.tif")["ring"][3],
      (76.0, 8.0))
check("a three-digit latitude is not a tile",
      H["parse_degree_tile"]("N123E45.tif"), None)
check("a four-digit longitude is not a tile",
      H["parse_degree_tile"]("N16E7300.tif"), None)
# An N8 tile cannot meet a 16-18 N scene; the bounds are what makes that legible
# rather than an empty dropdown.
n8   = H["parse_degree_tile"]("N8E76_ortho.tif")["ring"]
n9   = H["parse_degree_tile"]("N9E76_ortho.tif")["ring"]
check("bounds over several tiles", H["rings_bounds"]([n8, n9]),
      (76.0, 8.0, 77.0, 10.0))
check("bounds of nothing", H["rings_bounds"]([]), None)
check("bounds read as lat then lon", H["format_bounds"]((76.0, 8.0, 77.0, 10.0)),
      "8.000..10.000 lat, 76.000..77.000 lon")
check("bounds of nothing reads as empty", H["format_bounds"](None), "empty")

check("unpadded names still detect as degree-tile",
      H["detect_reference_mode"](["N8E76_ortho.tif", "N9E76_ortho.tif"]),
      "degree-tile")
check("not a tile", H["parse_degree_tile"]("cartosat_ortho.tif"), None)
check("a NISAR granule is not a tile", H["parse_degree_tile"](xml_name), None)
check("out-of-range lat rejected", H["parse_degree_tile"]("N95E073.tif"), None)

# ── 8. L8: a shapefile index names the rasters ───────────────────────────────
# The real L8 index.shp stores tif names under 'FileName'.
check("'FileName' wins over other plausible fields",
      H["rank_name_fields"](["OBJECTID", "Shape_Area", "FileName",
                             "path", "acq_date"])[0], "FileName")
check("matching is case-insensitive on the field name",
      H["rank_name_fields"](["FILENAME", "name"])[0], "FILENAME")

fields = ["OBJECTID", "geom_area", "FILENAME", "path", "acq_date"]
check("name-ish attributes rank first", H["rank_name_fields"](fields)[0], "FILENAME")
ranked = H["rank_name_fields"](fields)
check("hinted attributes all rank above unhinted ones",
      ranked.index("path") < ranked.index("acq_date")
      and ranked.index("OBJECTID") < ranked.index("acq_date")
      and ranked.index("OBJECTID") < ranked.index("geom_area"), True)

scenes = ["LC08_144048_20240102.tif", "LC08_144049_20240102.TIF"]
check("index value: bare stem",
      H["resolve_index_name"]("LC08_144048_20240102", scenes), scenes[0])
check("index value: filename",
      H["resolve_index_name"]("LC08_144048_20240102.tif", scenes), scenes[0])
check("index value: a path from another machine",
      H["resolve_index_name"](r"D:\\refs\\L8\\LC08_144048_20240102.tif", scenes),
      scenes[0])
check("index value: case-insensitive extension",
      H["resolve_index_name"]("LC08_144049_20240102.tif", scenes), scenes[1])
check("index value naming a missing raster",
      H["resolve_index_name"]("LC08_999999_20240102.tif", scenes), None)
check("index value blank", H["resolve_index_name"]("   ", scenes), None)
check("index value null", H["resolve_index_name"](None, scenes), None)

# L8 tif names carry no structure, so the index attribute is the only link --
# and with many features the lookup is built once and reused.
lookup = H["build_name_lookup"](scenes)
check("prebuilt lookup resolves the same",
      H["resolve_index_name"]("LC08_144048_20240102.tif", lookup), scenes[0])
check("prebuilt lookup, case-insensitive",
      H["resolve_index_name"]("lc08_144049_20240102.TIF", lookup), scenes[1])
check("prebuilt lookup, absent raster",
      H["resolve_index_name"]("LC08_000000_20240102.tif", lookup), None)

check("prefers index.shp", H["pick_index_shapefile"](["tiles.shp", "index.shp"]),
      "index.shp")
check("falls back to the only shapefile",
      H["pick_index_shapefile"](["tiles.shp"]), "tiles.shp")
check("no shapefile", H["pick_index_shapefile"](["a.tif", "b.met"]), None)

# ── 9. picking the mode from a folder listing ────────────────────────────────
check("L8 folder -> index",
      H["detect_reference_mode"](["LC08_a.tif", "LC08_b.tif", "index.shp",
                                  "index.dbf", "index.shx"]), "index-shp")
check("NISAR folder -> sidecar",
      H["detect_reference_mode"]([met_name, xml_name, "a.tif"]), "sidecar")
check("C1 folder -> degree-tile",
      H["detect_reference_mode"](["N16E73.tif", "N16E74.tif", "N17E73.tif"]),
      "degree-tile")
# An index or sidecar states the real footprint; a tile name only implies a
# nominal cell, so the name must never win over either.
check("index beats a sidecar",
      H["detect_reference_mode"](["a.tif", "a.met", "index.shp"]), "index-shp")
check("sidecar beats a tile name",
      H["detect_reference_mode"](["N16E73.tif", "N16E73.met"]), "sidecar")
check("rasters with nothing to place them",
      H["detect_reference_mode"](["scene_a.tif", "readme.txt"]), None)
check("empty folder", H["detect_reference_mode"]([]), None)

print()
if failures:
    print(f"{len(failures)} FAILURE(S)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
