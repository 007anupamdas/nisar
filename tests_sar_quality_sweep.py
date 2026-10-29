"""Exercise sar_quality_sweep's verdicts and arithmetic without rasterio.

The pure helpers in sar_quality_sweep.py sit between explicit markers; this
execs that exact slice, so what is tested is the code that ships rather than a
copy of it. rasterio and numpy are both absent here, which is the point: the
reasoning that turns seven window sizes into "plateau" or "upper bound" should
not need a raster to be checked.

    python3 tests_sar_quality_sweep.py     # exits non-zero on any assertion

The interesting cases are the ones a plausible implementation gets quietly
wrong: a saturated res_eff averaged in as though it were a measurement, a
sequence still falling at the smallest window reported as a plateau, and
pixels-per-sample taken from one axis when the correlation cell has two.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sar_quality_sweep.py")

BEGIN = "# ── BEGIN PURE HELPERS"
END = "# ── END PURE HELPERS"


def load_helpers(path=SRC):
    """The module's constants and its pure slice, without importing rasterio."""
    text = open(path, encoding="utf-8").read()
    start, end = text.index(BEGIN), text.index(END)
    namespace = {"math": math, "os": os, "re": re, "json": None}
    # Constants first: the slice's defaults are those names, and a loader that
    # skipped them would leave every signature half-defined.
    for line in text[:start].splitlines():
        if re.match(r"^[A-Z][A-Z0-9_]*\s*=", line):
            exec(line, namespace)
    exec(compile(text[start:end], path, "exec"), namespace)
    return namespace


H = load_helpers()
failures = []


def check(label, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def close(label, got, want, tol):
    ok = got is not None and math.isfinite(got) and abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: {got!r} (want {want} +/- {tol})")
    if not ok:
        failures.append(f"{label}: got {got!r}, want {want} +/- {tol}")


def section(title):
    print("\n" + "-" * 70)
    print(title)


# ── the editable block ──────────────────────────────────────────────────────
section("WHAT TO SWEEP: the constants a person edits instead of passing flags")

check("every default size is at or above the floor",
      [s for s in H["SIZES"] if s < H["SIZE_MIN"]], [])
check("and the sweep spans at least three octaves, or there is no plateau "
      "to find", max(H["SIZES"]) // min(H["SIZES"]) >= 8, True)
check("the smallest default size is the floor itself, so the plateau is "
      "reached if it exists", min(H["SIZES"]), H["SIZE_MIN"])
check("nothing is swept twice", len(set(H["SIZES"])), len(H["SIZES"]))
check("RASTERS is empty so the command line is the default path",
      tuple(H["RASTERS"]), ())
check("CENTERS likewise", tuple(H["CENTERS"]), ())
check("and BAND_NAME is unset rather than a guess", H["BAND_NAME"], None)


# ── lags ─────────────────────────────────────────────────────────────────────
section("lag_for_size: a quarter of the side, capped, never under two")

check("a big window takes the cap", H["lag_for_size"](512), 16)
check("256 is still at the cap", H["lag_for_size"](256), 16)
check("64 takes a quarter", H["lag_for_size"](64), 16)
check("48 is a quarter of itself", H["lag_for_size"](48), 12)
check("96 does not exceed the cap", H["lag_for_size"](96), 16)
check("a tiny window keeps two lags to interpolate between",
      H["lag_for_size"](4), 2)
check("and one pixel cannot go below that", H["lag_for_size"](1), 2)


# ── saturation ───────────────────────────────────────────────────────────────
section("saturated: the 1/e crossing that was never found")

check("res_eff at exactly max_lag * posting is the sentinel",
      H["saturated"](160.0, 10.0, 16), True)
check("and beyond it, likewise", H["saturated"](200.0, 10.0, 16), True)
check("a crossing inside the lags is a measurement",
      H["saturated"](14.0, 10.0, 16), False)
check("floating point does not turn a real ceiling into a measurement",
      H["saturated"](159.99999999, 10.0, 16), True)
check("a 37 m product saturates at its own ceiling",
      H["saturated"](8 * 36.4, 36.4, 8), True)
check("just inside it does not", H["saturated"](8 * 36.4 - 1.0, 36.4, 8), False)
check("a nan res_eff cannot be trusted either way",
      H["saturated"](float("nan"), 10.0, 16), True)
check("nor a missing posting", H["saturated"](14.0, 0.0, 16), True)


# ── pixels per sample ────────────────────────────────────────────────────────
section("pixels_per_sample: the correlation cell has two axes")

close("critical sampling is one pixel per sample",
      H["pixels_per_sample"](1.0, 1.0), 1.0, 1e-9)
close("1.4 x 1.4 is where the assumed 2.0 is right",
      H["pixels_per_sample"](1.4, 1.4), 1.96, 1e-9)
close("3x oversampled on both axes is nine, not three",
      H["pixels_per_sample"](3.0, 3.0), 9.0, 1e-9)
close("an anisotropic cell multiplies rather than picking a side",
      H["pixels_per_sample"](1.2, 2.4), 2.88, 1e-9)
close("a cell smaller than a pixel is floored, not believed",
      H["pixels_per_sample"](0.7, 0.8), 1.0, 1e-9)
check("a nan axis gives no figure",
      math.isnan(H["pixels_per_sample"](float("nan"), 1.0)), True)


# ── precision ────────────────────────────────────────────────────────────────
section("precision_factor: how far a quoted ENL precision is out")

close("the assumed value is the fixed point",
      H["precision_factor"](2.0), 1.0, 1e-9)
close("one pixel per sample makes the quoted precision too wide",
      H["precision_factor"](1.0), 0.7071, 1e-4)
close("nine makes it too narrow by about 2.1x",
      H["precision_factor"](9.0), 2.1213, 1e-4)
close("four doubles the honest error bar",
      H["precision_factor"](8.0), 2.0, 1e-9)
check("zero pixels per sample is not a ratio",
      math.isnan(H["precision_factor"](0.0)), True)

# The whole point of the exercise, checked end to end: +/-7.3% is quoted for a
# 900-pixel Antarctic ROI at two pixels per sample, and a product measured at
# 3 x 3 would owe a wider figure than the report claims.
pps = H["pixels_per_sample"](3.0, 3.0)
quoted = 1.1 * math.sqrt(2.0 / (900 / 2.0))
honest = 1.1 * math.sqrt(2.0 / (900 / pps))
close("the quoted Antarctic precision is 7.3%", quoted * 100, 7.33, 0.05)
close("at 3 x 3 the honest figure is nearer 15.6%", honest * 100, 15.56, 0.05)
close("which is the precision_factor, arrived at the long way",
      honest / quoted, H["precision_factor"](pps), 1e-9)


# ── verdicts ─────────────────────────────────────────────────────────────────
section("classify: four outcomes, smallest window first")

check("scene entering the bigger windows leaves a plateau at the bottom",
      H["classify"]([1.42, 1.44, 1.45, 1.80, 2.40]),
      ("plateau", 1.42, 3))
check("a sweep that never moved was never scene-dominated",
      H["classify"]([1.40, 1.42, 1.41, 1.43]),
      ("flat throughout", 1.40, 4))
check("still climbing at the smallest window is a bound, not a plateau",
      H["classify"]([1.50, 1.90, 2.40, 3.10]),
      ("no plateau", 1.50, 1))
check("bouncing without settling is noise",
      H["classify"]([1.50, 2.90, 1.60, 2.80]),
      ("unstable", 1.50, 1))
check("one window is not a sweep",
      H["classify"]([1.42]), ("single run", 1.42, 1))
check("nothing usable says so rather than returning a number",
      H["classify"]([])[0], "no data")
check("and nans do not count as data", H["classify"]([float("nan")])[0],
      "no data")
check("a sequence that falls as the window grows is not the expected shape",
      H["classify"]([2.40, 1.90, 1.50])[0], "unstable")
check("two agreeing runs are enough for a plateau",
      H["classify"]([1.42, 1.45, 2.90]), ("plateau", 1.42, 2))
check("the tolerance is on the plateau's own base, not its neighbour",
      H["classify"]([1.40, 1.47, 1.54, 1.61])[2], 2)
check("one noisy step does not disqualify a rising tail",
      H["classify"]([1.50, 2.00, 1.98, 2.60]), ("no plateau", 1.50, 1))

check("every verdict carries a note", sorted(
      k for k in ("plateau", "flat throughout", "no plateau", "unstable",
                  "single run", "no data")
      if H["verdict_note"](k) != k),
      ["flat throughout", "no data", "no plateau", "plateau", "single run",
       "unstable"])
check("and an upper bound says so in capitals, where it cannot be skimmed past",
      "UPPER BOUND" in H["verdict_note"]("no plateau"), True)


# ── names ────────────────────────────────────────────────────────────────────
section("size_from_name: the fallback for files sar_quality wrote")

check("the loop's own naming", H["size_from_name"]("sweep_256.json"), 256)
check("a path, not just a name",
      H["size_from_name"]("/tmp/run/sweep_128.json"), 128)
check("digits after the size are the size",
      H["size_from_name"]("w3_size256.json"), 256)
check("a year is not a window size", H["size_from_name"]("L_gcov_2026.json"),
      None)
check("nor is a size below the floor",
      H["size_from_name"]("sweep_32.json"), None)
check("no digits at all", H["size_from_name"]("L_gcov.json"), None)
check("a marker is required, so a dated directory cannot win",
      H["size_from_name"]("/2026-10-29/sweep_192.json"), 192)
check("a track and frame in the name are not sizes",
      H["size_from_name"]("NISAR_L2_PR_GCOV_028_170.json"), None)
check("the marker may be bare", H["size_from_name"]("w1_s512.json"), 512)
check("case does not matter", H["size_from_name"]("Sweep-384.json"), 384)
check("and the last marked size wins",
      H["size_from_name"]("sweep_256_size128.json"), 128)


# ── reading sar_quality's output ─────────────────────────────────────────────
section("band_rows, window_flags and collate: what comes out of a report")

report = {"path": "/data/L_gcov.tif", "bands": [
    {"band": 1, "name": "HHHH", "oversmp_x": 1.42, "oversmp_y": 1.44,
     "res_eff_x_m": 14.2, "res_eff_y_m": 14.4, "px_m": 10.0, "valid": 1.0,
     "cv_scene": 0.12, "bright_ratio": 2.1},
    {"band": 2, "name": "HVHV", "valid": 0.31,
     "note": "under half the window is data"},
    {"band": 3, "name": "rtcGammaToSigma", "oversmp_x": 9.0, "oversmp_y": 9.0,
     "res_eff_x_m": 160.0, "res_eff_y_m": 160.0, "px_m": 10.0, "valid": 1.0,
     "cv_scene": 0.02, "bright_ratio": 1.4},
]}

check("a skipped band holds no statistics and is not read",
      [r["name"] for r in H["band_rows"](report)],
      ["HHHH", "rtcGammaToSigma"])
check("and a name filter is how you avoid measuring the RTC factor",
      [r["name"] for r in H["band_rows"](report, "HHHH")], ["HHHH"])
check("the guards travel with the number, not separately",
      sorted(k for k in H["band_rows"](report)[0]
             if k in ("cv_scene", "bright_ratio")),
      ["bright_ratio", "cv_scene"])

section("window_flags: the module's own guards, read back")

clean = {"res_eff_x_m": 14.2, "res_eff_y_m": 14.4, "px_m": 10.0, "valid": 1.0,
         "cv_scene": 0.12, "bright_ratio": 2.1}
check("a good window carries nothing", H["window_flags"](clean, 16), ())
check("the city is flagged by its bright ratio, at the module's own 33",
      H["window_flags"](dict(clean, bright_ratio=33.0), 16), ("bright 33",))
check("a field-filled window is flagged as scene",
      H["window_flags"](dict(clean, cv_scene=0.42), 16), ("scene",))
check("0.30 exactly is not over the threshold",
      H["window_flags"](dict(clean, cv_scene=0.30), 16), ())
check("15.0 exactly likewise",
      H["window_flags"](dict(clean, bright_ratio=15.0), 16), ())
check("a window that slid off the raster says how much it kept",
      H["window_flags"](dict(clean, valid=0.62), 16), ("valid 62%",))
check("and several faults are all reported",
      H["window_flags"](dict(clean, res_eff_x_m=160.0, cv_scene=0.5,
                             bright_ratio=40.0, valid=0.5), 16),
      ("saturated", "scene", "bright 40", "valid 50%"))

section("the copied thresholds have not drifted from sar_quality.py")

src = open(os.path.join(HERE, "sar_quality.py"), encoding="utf-8").read()
for name in ("SCENE_DOMINATES_CV", "BRIGHT_TARGET_RATIO"):
    theirs = float(re.search(rf"^{name}\s*=\s*([0-9.]+)", src,
                             re.MULTILINE).group(1))
    check(f"{name} matches the module it was copied from", H[name], theirs)

section("collate: a series per centre, raster and band")

runs = [{"size": 256, "max_lag": 16, "center": [78.3, 17.33],
         "reports": [report]},
        {"size": 128, "max_lag": 16, "center": [78.3, 17.33],
         "reports": [report]},
        {"size": 128, "max_lag": 16, "center": [78.5, 17.4],
         "reports": [report]}]
series = H["collate"](runs)
check("two centres do not pool into one series",
      sorted({k[0] for k in series}), ["78.3000,17.3300", "78.5000,17.4000"])
check("ordered by ascending size, which is the order the verdict reads",
      [p[0] for p in series[("78.3000,17.3300", "L_gcov.tif", "HHHH")]],
      [128, 256])
check("the RTC factor's saturated res_eff is flagged, not averaged in",
      series[("78.3000,17.3300", "L_gcov.tif", "rtcGammaToSigma")][0][4],
      ("saturated",))
check("no centre is labelled rather than guessed at",
      sorted(H["collate"]([{"size": 128, "max_lag": 16, "center": None,
                            "reports": [report]}]))[0][0],
      "mid-scene")
check("fmt_center rounds to four places, which is ~10 m",
      H["fmt_center"]((78.299999, 17.33)), "78.3000,17.3300")

# ── conclusions ──────────────────────────────────────────────────────────────
section("conclude: a series into the figure that goes in the report")

points = [(64, 1.42, 1.44, 2.04, ()), (96, 1.43, 1.45, 2.07, ()),
          (128, 1.44, 1.46, 2.10, ()), (256, 1.90, 1.92, 3.65, ()),
          (512, 2.60, 2.64, 6.86, ())]
c = H["conclude"](points)
check("the plateau is found", c["verdict"], "plateau")
close("and its pixels-per-sample read off it", c["pixels_per_sample"], 2.04,
      0.005)
close("which vindicates the assumed 2.0", c["precision_factor"], 1.01, 0.01)
check("over the sizes that agreed", c["plateau_sizes"], [64, 96, 128])

sat = [(64, 1.42, 1.44, 2.04, ()), (96, 1.43, 1.45, 2.07, ()),
       (256, 16.0, 16.0, 256.0, ("saturated",))]
c = H["conclude"](sat)
check("a saturated run is excluded from the verdict", c["verdict"],
      "flat throughout")
close("so it cannot drag pixels-per-sample", c["pixels_per_sample"], 2.04,
      0.005)
check("but it is counted and reported", (c["saturated"], c["runs"]), (1, 3))

check("every run saturating leaves nothing to conclude",
      H["conclude"]([(256, 16.0, 16.0, 256.0, ("saturated",))])["verdict"],
      "no data")


# ── unusable centres and agreement ──────────────────────────────────────────
section("unusable: a patch of ground that cannot answer at any size")

allbright = [(64, 2.0, 2.0, 4.0, ("bright 33",)),
             (128, 2.4, 2.4, 5.8, ("bright 31", "scene"))]
check("every window bright-dominated means the centre is unusable",
      H["unusable"](allbright), True)
check("one clean window is enough to keep it",
      H["unusable"](allbright + [(96, 1.4, 1.4, 1.96, ())]), False)
check("a scene-dominated big window alone does not condemn a centre",
      H["unusable"]([(64, 1.4, 1.4, 1.96, ()),
                     (256, 2.6, 2.6, 6.8, ("scene",))]), False)
check("saturation is its own verdict, not unusability",
      H["unusable"]([(256, 16.0, 16.0, 256.0, ("saturated",))]), False)

section("agree: do several centres say the same thing")

check("two centres within the tolerance agree",
      H["agree"]([2.04, 2.10])[0], "centres agree")
check("and the median is reported, not the mean",
      H["agree"]([2.0, 2.1, 6.0])[1], 2.1)
check("a wide spread is reported as disagreement",
      H["agree"]([2.0, 4.0])[0], "centres disagree")
close("the spread is the half-range over the median",
      H["agree"]([1.9, 2.1])[2], 0.05, 1e-9)
check("one centre is not agreement", H["agree"]([2.04])[0], "one centre")
check("none at all says so", H["agree"]([])[0], "no centres")
check("four agreeing centres is the result worth reporting",
      H["agree"]([2.02, 2.04, 2.07, 2.10])[0], "centres agree")

section("across_centres: only centres that reached a plateau are pooled")

many = {
    ("78.3,17.3", "L.tif", "HHHH"): [(64, 1.42, 1.44, 2.04, ()),
                                     (96, 1.43, 1.45, 2.07, ())],
    ("78.5,17.4", "L.tif", "HHHH"): [(64, 1.44, 1.46, 2.10, ()),
                                     (96, 1.45, 1.46, 2.12, ())],
    # still falling: an upper bound, so it must not join the median
    ("78.7,17.5", "L.tif", "HHHH"): [(64, 2.10, 2.12, 4.45, ()),
                                     (96, 2.80, 2.82, 7.90, ()),
                                     (128, 3.60, 3.62, 13.03, ())],
    # unusable: bright at every size
    ("78.9,17.6", "L.tif", "HHHH"): [(64, 1.9, 1.9, 3.61, ("bright 33",))],
}
summary = H["across_centres"](many)
check("one entry per raster and band", sorted(summary), [("L.tif", "HHHH")])
verdict, mid, spread, centres = summary[("L.tif", "HHHH")]
check("only the two that plateaued are pooled", sorted(centres),
      ["78.3,17.3", "78.5,17.4"])
check("and they agree", verdict, "centres agree")
close("at a median near two pixels per sample", mid, 2.07, 0.005)


# ── the table ────────────────────────────────────────────────────────────────
section("format_table: the block that goes in the report")

text = H["format_table"]({("78.3,17.3", "L_gcov.tif", "HHHH"): points})
check("the centre, raster and band head the block",
      "78.3,17.3  L_gcov.tif  HHHH" in text, True)
check("every swept size appears",
      all(f"  {s:6d}" in text for s in (64, 96, 128, 256, 512)), True)
check("the verdict is stated, not left to be inferred",
      "-> plateau" in text, True)
check("and it names the constant it bears on",
      "DPQED_radial.py" in text, True)

bound = H["format_table"]({("78.3,17.3", "S_gcov.tif", "HHHH"): [
    (64, 2.10, 2.12, 4.45, ()), (128, 2.80, 2.82, 7.90, ()),
    (256, 3.60, 3.62, 13.03, ())]})
check("an upper bound is unmissable in the table too",
      "UPPER BOUND" in bound, True)

multi = H["format_table"](many)
check("several centres bring an across-centres block",
      "ACROSS CENTRES" in multi, True)
check("which states the agreement", "centres agree" in multi, True)
check("and an unusable centre says so where it was tried",
      "cannot measure resolution" in multi, True)

section("parse_centers: the forms a person actually pastes")

check("semicolon separated",
      H["parse_centers"]("78.30,17.33; 78.50,17.40"),
      [(78.30, 17.33), (78.50, 17.40)])
check("newline separated with comments",
      H["parse_centers"]("78.30,17.33  # lake\n78.50,17.40 # farmland"),
      [(78.30, 17.33), (78.50, 17.40)])
check("whitespace instead of a comma",
      H["parse_centers"]("78.30 17.33"), [(78.30, 17.33)])
check("negative longitudes, for the Amazon",
      H["parse_centers"]("-60.25,-3.10"), [(-60.25, -3.10)])
check("a blank list is empty rather than an error",
      H["parse_centers"]("  \n # nothing\n"), [])
check("nothing at all", H["parse_centers"](None), [])
try:
    H["parse_centers"]("78.30,17.33,99")
    check("three numbers are refused rather than truncated", False, True)
except ValueError as e:
    check("three numbers are refused rather than truncated",
          "not a lon,lat pair" in str(e), True)
try:
    H["parse_centers"]("lake")
    check("a word is refused", False, True)
except ValueError:
    check("a word is refused", True, True)

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
