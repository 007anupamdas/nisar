"""Find a product's real oversampling by shrinking the window until it stops mattering.

`sar_quality.py` measures `res_eff` over the WHOLE window, so any scene
structure inside it widens the autocorrelation and the figure reads coarse.
Over a scene with no large homogeneous patch -- which is most scenes -- a single
window therefore cannot say whether 3.0 means a soft product or a window full
of fields. Written because that is the position the Hyderabad GCOV put us in:
the bright speckly zones are urban, and no 2.5 km patch of uniform ground was
obviously there to be found.

The way out is not a better window, it is a sequence of them. Shrink the window
at a fixed centre and `oversmp` falls as scene leaves it, then flattens once
only the impulse response and speckle remain. That plateau is the product's
oversampling; the rise above it is the scene. No window has to be perfect.

    python3 sar_quality_sweep.py L_gcov.tif S_gcov.tif --center 78.30,17.33

Four outcomes, and the table says which one happened rather than leaving it to
be judged by eye:

  plateau            the usual case. The flat value is the answer.
  flat throughout    the window was never scene-dominated. Strongest result.
  no plateau         still falling at the smallest size: even that much ground
                     is not homogeneous. The smallest value is an UPPER BOUND
                     and is reported as one.
  unstable           moving around at small sizes without settling, which is
                     sampling noise rather than a measurement.

WHAT THIS IS FOR

`DPQED_radial.py` turns an ROI's pixel count into an ENL precision through

    ENL_PIXELS_PER_SAMPLE = 2.0

-- how many posted pixels make one independent speckle sample. Its own comment
calls it "the one figure here that is a rule of thumb rather than arithmetic",
and every ENL precision in a report scales with its square root. A measured
`oversmp` settles it, because pixels per sample is the correlation cell's area:

    pixels per sample ~ oversmp_x * oversmp_y

So a plateau at 1.0 x 1.0 means the constant is conservative by two and the
quoted precisions are a factor sqrt(2) wider than they need to be; a plateau at
3 x 3 means nine, and every quoted precision is about 2.1x too narrow. Either
way it is an answer, which 2.0 by assertion is not.

WHAT IS NOT REPORTED

`sar_quality.py` is calibrated for single-look AMPLITUDE: `SPECKLE_CV_1LOOK` is
the Rayleigh sigma/mean, and `ENL`, `cv_floor`, `tex@` and `cnr@` all descend
from it. A GCOV holds multilooked POWER, where single-look cv is 1.0, so those
four come out wrong by a constant factor and this script does not print them.
`oversmp` and `aniso` are normalised autocorrelation widths and a ratio of two
of them, so neither depends on that calibration.
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import sys

# ── WHAT TO SWEEP ────────────────────────────────────────────────────────────
# Set these and run the file, as DPQED_gcov2tif.py is set and run. Anything
# passed on the command line overrides them, so one script serves both habits:
# edit-and-run while a scene is being explored, and an explicit command once a
# figure is going in a report, where the command line IS the record of what was
# measured and a constant edited three runs ago is not.
RASTERS = ()                 # ('L_gcov.tif', 'S_gcov.tif')
CENTERS = ()                 # ((78.30, 17.33), (78.50, 17.40))
BAND_NAME = None             # 'HHHH' -- a description, not an index

# Window sides in pixels, swept at every centre. Geometric rather than linear:
# the question is which power of two the plateau sits above, and a linear sweep
# spends its runs in the large sizes where the answer is already known to be
# wrong.
SIZES = (512, 384, 256, 192, 128, 96, 64)

# Smallest window worth fitting an autocorrelation to. At lag L the sum runs
# over N-L shifted rows, so a small window spends most of its samples on the
# longest lags and the 1/e crossing starts moving between runs over the same
# ground. 64 is where that is still tolerable; below it the estimate is noise.
#
# Note this is BELOW sar_quality's own 160 px threshold (32 * (2*pad+1)), under
# which `speckle_region` silently returns the whole window instead of the
# quietest patch. That degrades `speckle_corr_m`, `cv_speckle` and `cv_scene`
# -- none of which this script reports -- and leaves res_eff, oversmp and aniso
# untouched, which is why going under 160 is safe here and would not be there.
SIZE_MIN = 64

# Correlation lags. sar_quality defaults to 16, which at 10 m posting puts the
# measurable ceiling at 160 m -- far beyond any real resolution. The cap only
# binds on the small windows, where a quarter of the side is less than 16; the
# module applies the same quarter rule to its own quiet region.
MAX_LAG_CAP = 16
LAG_FRACTION = 4

# Two pixels-per-sample figures count as the same plateau within this FRACTION
# of the smaller. Relative rather than absolute because the quantity compared
# is a product of two oversmp values: an absolute 0.08 is 5% at 1.5 and 1% at 9,
# so it would call a tight plateau a slope wherever the cell is small. The
# figure itself is set by the 1/e crossing being interpolated between whole
# lags, which moves either axis by a few percent on its own.
PLATEAU_TOL = 0.08

# A plateau wants more than one point agreeing with itself.
PLATEAU_MIN_RUNS = 2

# What DPQED_radial.py assumes, and what this script measures against.
PIXELS_PER_SAMPLE_ASSUMED = 2.0

# sar_quality's own two guards, copied rather than imported: that module opens
# rasterio at import, and the slice below is meant to stay readable on a
# machine that has none. tests_sar_quality_sweep.py reads the figures back out
# of sar_quality.py as text and fails if these drift from them, so a copy
# cannot quietly go stale.
SCENE_DOMINATES_CV = 0.30
BRIGHT_TARGET_RATIO = 15.0

# Plateau values from two centres count as agreeing within this fraction of
# their median. Looser than PLATEAU_TOL because two patches of ground are not
# two window sizes over one patch: different surfaces, different brightness,
# different distance from the floor.
AGREE_TOL = 0.15

# ── BEGIN PURE HELPERS ───────────────────────────────────────────────────────
# Everything below is arithmetic on numbers sar_quality already produced: no
# rasterio, no numpy, no file reading. tests_sar_quality_sweep.py execs this
# slice, so the tested code is the code that ships.


def lag_for_size(size, cap=MAX_LAG_CAP, fraction=LAG_FRACTION):
    """Correlation lags a window of `size` pixels can support.

    A quarter of the side, capped. Two is the floor: `one_over_e_width` needs
    at least one lag to interpolate between and a profile of one point can only
    ever return its own length.
    """
    return max(2, min(int(cap), int(size) // int(fraction)))


def saturated(res_eff_m, px_m, max_lag):
    """Did the 1/e crossing fall outside the lags that were computed?

    `one_over_e_width` returns `len(profile)` when the correlation never drops
    below 1/e, so res_eff comes back as exactly max_lag * posting. That is a
    sentinel, not a measurement, and a sweep that averaged it in would report
    the ceiling it hit instead of the width it failed to find.

    The comparison is relative rather than to a fixed epsilon: the sentinel
    arrives as a float multiplication whose last bits depend on the posting, so
    an absolute tolerance that is generous at 10 m is meaningless at 36.4 m.
    """
    if not (px_m and math.isfinite(res_eff_m) and math.isfinite(px_m)):
        return True
    return res_eff_m >= px_m * max_lag * (1.0 - 1e-6)


def pixels_per_sample(oversmp_x, oversmp_y):
    """Posted pixels in one independent speckle sample.

    The correlation cell is two-dimensional, so the areal figure is the product
    of the two axes' widths and not either one of them. Floored at 1.0: a cell
    smaller than a pixel would mean neighbouring pixels are more independent
    than the sampling allows, which is a measurement artefact rather than a
    product property.
    """
    if not (math.isfinite(oversmp_x) and math.isfinite(oversmp_y)):
        return float("nan")
    return max(1.0, oversmp_x * oversmp_y)


def precision_factor(pps, assumed=PIXELS_PER_SAMPLE_ASSUMED):
    """How far an ENL precision quoted at `assumed` is out, as a multiplier.

    Precision goes as sqrt(2/N) and N is the pixel count divided by pixels per
    sample, so the quoted figure scales with the square root of the ratio.
    Above 1.0 the quoted precision is too narrow -- the honest error bar is
    wider than the report claims, which is the direction that matters.
    """
    if not (math.isfinite(pps) and pps > 0 and assumed > 0):
        return float("nan")
    return math.sqrt(pps / assumed)


def classify(values, tol=PLATEAU_TOL, min_runs=PLATEAU_MIN_RUNS):
    """Read a size sweep, smallest window first, into a verdict and a value.

    `values` is the usable oversmp figures ordered by ascending window size.
    The plateau is however many of the smallest windows agree with the
    smallest one; what sits above it is scene entering the window.

    Returns (verdict, value, plateau_length). The verdict names which of the
    four outcomes happened, because the distinction that matters -- a plateau
    against a sequence still falling at the smallest size -- is the difference
    between a measurement and an upper bound, and nothing downstream can
    recover it from the number alone.
    """
    usable = [v for v in values if v is not None and math.isfinite(v)]
    if not usable:
        return "no data", float("nan"), 0
    if len(usable) == 1:
        return "single run", usable[0], 1
    base = usable[0]
    run = 1
    while run < len(usable) and abs(usable[run] - base) <= tol * abs(base):
        run += 1
    if run == len(usable):
        return "flat throughout", base, run
    if run >= min_runs:
        return "plateau", base, run
    # One point, then a jump. Rising means scene came in with the bigger
    # window, which is the expected shape and leaves the smallest value as a
    # bound rather than a plateau. Falling or bouncing is not a trend at all.
    rest = usable[1:]
    if (all(v > base * (1.0 + tol) for v in rest)
            and _monotone_up(rest, tol * abs(base))):
        return "no plateau", base, 1
    return "unstable", base, 1


def _monotone_up(values, tol):
    """Non-decreasing to within `tol`, so one noisy step does not disqualify."""
    return all(b >= a - tol for a, b in zip(values, values[1:]))


def verdict_note(verdict):
    """What the verdict means for the number beside it, in one line."""
    return {
        "plateau": "the flat value is the product's oversampling",
        "flat throughout": "never scene-dominated; the strongest form of this result",
        "no plateau": "UPPER BOUND only -- still falling at the smallest window",
        "unstable": "sampling noise, not a measurement; try another centre",
        "single run": "one window is not a sweep; vary --sizes",
        "no data": "every run saturated or was skipped",
    }.get(verdict, verdict)


def size_from_name(name):
    """The window size a `sar_quality.py --json` file's name carries, or None.

    A fallback for files written by hand before this script existed: the JSON
    that `sar_quality.py` writes records px_m and every statistic but NOT the
    window it used, so a sweep read back from those files can only recover the
    size from how they were named.

    A marker is required rather than taking the digits on their own, because
    every other number a person puts in a filename is also digits: a granule
    year, a track, a frame, a date. `L_gcov_2026.json` read as a 2026-pixel
    window is a plausible size and would sort into the sweep as the largest
    one, quietly bending the verdict. Refusing to guess costs a renamed file
    and cannot corrupt a result, so `sweep_256`, `size256` and `_s256` are
    read and anything else is skipped with a message.
    """
    found = re.findall(r"(?:size|sweep|_s)[-_]?(\d+)",
                       os.path.basename(name), re.IGNORECASE)
    if not found:
        return None
    size = int(found[-1])
    return size if SIZE_MIN <= size <= 100000 else None


def band_rows(report, band_name=None):
    """The fields this script reads, per band of one sar_quality report.

    Dicts rather than a tuple because what a window is worth takes more than
    its oversmp: cv_scene and bright_ratio are how the module itself decides
    whether a res_eff is a resolution at all, and a sweep over several centres
    is exactly the case where that has to travel with the number.

    Bands carrying a `note` are the ones sar_quality skipped for having under
    half a window of data, and they hold no statistics to read.
    """
    nan = float("nan")
    out = []
    for b in report.get("bands", []):
        if "note" in b:
            continue
        if band_name is not None and b.get("name") != band_name:
            continue
        out.append({"name": b.get("name", "?"),
                    "oversmp_x": b.get("oversmp_x", nan),
                    "oversmp_y": b.get("oversmp_y", nan),
                    "res_eff_x_m": b.get("res_eff_x_m", nan),
                    "res_eff_y_m": b.get("res_eff_y_m", nan),
                    "px_m": b.get("px_m", nan),
                    "valid": b.get("valid", nan),
                    "cv_scene": b.get("cv_scene", nan),
                    "bright_ratio": b.get("bright_ratio", nan)})
    return out


def window_flags(row, max_lag):
    """What is wrong with this window, in the module's own terms.

    Two flags disqualify the number and are named in DISQUALIFYING:

    `saturated` -- the 1/e crossing was never found, so res_eff is the ceiling
    it hit rather than a width.

    `bright` -- strong scatterers' sidelobes dominate the autocorrelation, so
    res_eff measures the brightest targets' impulse response and not the
    product's. The module's own remedy is to move, not to shrink: a bright
    target inside a 64-pixel window is still inside it. Measured from max/p99,
    which is a ratio of percentiles and so carries no calibration constant and
    is valid on power.

    `cv_scene` is recorded and NEVER disqualifies, because on a power product
    it does not mean what its name says. `scene_contrast` subtracts a speckle
    cv capped at SPECKLE_CV_1LOOK = 0.523, the Rayleigh figure for single-look
    AMPLITUDE. Multilooked power has speckle cv 1/sqrt(ENL), which exceeds that
    cap for any product below ENL 3.66 -- so the excess speckle of a
    fewer-looks product is misread as scene. Measured on the Hyderabad pair
    over one patch of water: L (ENL 4.9) reports cv_scene 0.000 and S (ENL 2.7)
    reports 0.377 over the SAME ground, the whole of S's figure being its own
    speckle. A flag that fires on the product with fewer looks regardless of
    what is under it cannot be allowed to rule a window out.
    """
    flags = []
    if (saturated(row["res_eff_x_m"], row["px_m"], max_lag)
            or saturated(row["res_eff_y_m"], row["px_m"], max_lag)):
        flags.append("saturated")
    bright = row["bright_ratio"]
    if math.isfinite(bright) and bright > BRIGHT_TARGET_RATIO:
        flags.append(f"bright {bright:.0f}")
    cv_scene = row["cv_scene"]
    if math.isfinite(cv_scene) and cv_scene > SCENE_DOMINATES_CV:
        flags.append(f"cv_scene {cv_scene:.2f}")
    valid = row["valid"]
    if math.isfinite(valid) and valid < 0.999:
        flags.append(f"valid {valid:.0%}")
    return tuple(flags)


def disqualified(flags):
    """Do these flags mean the window cannot measure a resolution at all?"""
    return any(f == "saturated" or f.startswith("bright") for f in flags)


def collate(runs, band_name=None):
    """Sweep runs into {(centre, raster, band): [(size, ox, oy, areal, flags)]}.

    Ordered by ascending size, which is the order `classify` reads. Saturated
    runs are kept with a flag rather than dropped: a sweep where the big
    windows all saturated looks identical to one where they were never run,
    and the table should be able to say which.

    The centre is part of the key, so coordinates are never pooled. Two
    patches of ground measured at one size are not a sweep, and silently
    averaging them would turn "these four centres agree" -- the finding -- into
    a single number that cannot be checked.
    """
    series = {}
    for run in sorted(runs, key=lambda r: (str(r.get("center")), r["size"])):
        size, max_lag = run["size"], run["max_lag"]
        centre = fmt_center(run.get("center"))
        for report in run.get("reports", []):
            raster = os.path.basename(report.get("path", "?"))
            for row in band_rows(report, band_name):
                areal = pixels_per_sample(row["oversmp_x"], row["oversmp_y"])
                series.setdefault((centre, raster, row["name"]), []).append(
                    (size, row["oversmp_x"], row["oversmp_y"], areal,
                     window_flags(row, max_lag)))
    return series


def fmt_center(center):
    """A centre as a short label, or 'mid-scene' when none was given."""
    if not center:
        return "mid-scene"
    return f"{float(center[0]):.4f},{float(center[1]):.4f}"


def agree(values, tol=AGREE_TOL):
    """Do several centres' plateaus say the same thing?

    Returns (verdict, median, spread) where spread is the half-range as a
    fraction of the median. Agreement across patches of different ground is
    what makes the figure a property of the PRODUCT rather than of one patch,
    so it is reported separately from any single centre's plateau and never
    folded into it.
    """
    usable = sorted(v for v in values if v is not None and math.isfinite(v))
    if not usable:
        return "no centres", float("nan"), float("nan")
    mid = (usable[len(usable) // 2] if len(usable) % 2
           else (usable[len(usable) // 2 - 1] + usable[len(usable) // 2]) / 2.0)
    if len(usable) == 1:
        return "one centre", mid, 0.0
    spread = (usable[-1] - usable[0]) / 2.0 / mid if mid else float("inf")
    return ("centres agree" if spread <= tol else "centres disagree"), mid, spread


def conclude(points):
    """One series of (size, ox, oy, areal, flags) into its verdict.

    Saturated points are excluded from the verdict and stay in the table: a
    saturated run says the window was too structured to measure, which is
    evidence about the window and none about the product.
    """
    clean = [p for p in points if not disqualified(p[4])]
    verdict, areal, run = classify([p[3] for p in clean])
    sizes = [p[0] for p in clean[:run]] if run else []
    return {"verdict": verdict, "pixels_per_sample": areal,
            "precision_factor": precision_factor(areal),
            "plateau_runs": run, "plateau_sizes": sizes,
            "excluded": sum(1 for p in points if disqualified(p[4])),
            "runs": len(points)}


def unusable(points):
    """Was every window at this centre disqualified?

    A centre like that is a patch of ground that cannot answer the question:
    the module's own guards say its res_eff is not a resolution at any size
    swept, so its plateau -- if it has one -- is a plateau in something else.
    """
    return bool(points) and all(disqualified(p[4]) for p in points)


def across_centres(series):
    """{(raster, band): (verdict, median, spread, centres)} over the centres.

    Centres the guards rule out are excluded here but stay in the table, so a
    coordinate that could not answer is visible as having been tried rather
    than missing.
    """
    by_band = {}
    for (centre, raster, band), points in series.items():
        if unusable(points):
            continue
        c = conclude(points)
        if c["verdict"] in ("plateau", "flat throughout"):
            by_band.setdefault((raster, band), []).append(
                (centre, c["pixels_per_sample"]))
    out = {}
    for key, found in by_band.items():
        verdict, mid, spread = agree([v for _, v in found])
        out[key] = (verdict, mid, spread, [c for c, _ in found])
    return out


def format_table(series):
    """The sweep as text: a block per centre, raster and band, then verdicts."""
    lines = []
    for (centre, raster, band), points in sorted(series.items()):
        lines.append(f"\n{centre}  {raster}  {band}")
        lines.append(f"  {'size':>6s} {'oversmp x':>10s} {'oversmp y':>10s} "
                     f"{'px/sample':>10s}  flags")
        for size, ox, oy, areal, flags in points:
            lines.append(f"  {size:6d} {ox:10.2f} {oy:10.2f} {areal:10.2f}  "
                         f"{', '.join(flags)}")
        c = conclude(points)
        lines.append(f"  -> {c['verdict']}: {verdict_note(c['verdict'])}")
        if math.isfinite(c["pixels_per_sample"]):
            lines.append(
                f"     pixels per sample {c['pixels_per_sample']:.2f} against "
                f"{PIXELS_PER_SAMPLE_ASSUMED:g} assumed in DPQED_radial.py; "
                f"ENL precisions scale by {c['precision_factor']:.2f}x")
            if c["plateau_sizes"]:
                lines.append(f"     over sizes {c['plateau_sizes']}")
        if c["excluded"]:
            lines.append(f"     {c['excluded']} of {c['runs']} runs saturated or "
                         f"bright-dominated, excluded from the verdict")
        if unusable(points):
            lines.append("     EVERY window here was disqualified: this centre "
                         "cannot measure resolution")
    summary = across_centres(series)
    if summary:
        lines.append("\nACROSS CENTRES")
        for (raster, band), (verdict, mid, spread, centres) in sorted(summary.items()):
            lines.append(f"  {raster}  {band}: {verdict} -- median "
                         f"{mid:.2f} px/sample, spread {spread:+.0%}, "
                         f"{len(centres)} centre(s)")
            lines.append(f"     ENL precisions scale by "
                         f"{precision_factor(mid):.2f}x against "
                         f"{PIXELS_PER_SAMPLE_ASSUMED:g} assumed")
    return "\n".join(lines)
def parse_centers(text, path=None):
    """lon,lat pairs from a --centers string or a file, one pair per entry.

    Accepts the forms a person actually pastes: pairs separated by semicolons
    or newlines, with or without spaces, and '#' starting a comment so a list
    of coordinates can say what each one is over.
    """
    if path:
        text = open(path, encoding="utf-8").read()
    out = []
    for chunk in re.split(r"[;\n]", text or ""):
        chunk = chunk.split("#", 1)[0].strip()
        if not chunk:
            continue
        parts = [p for p in re.split(r"[,\s]+", chunk) if p]
        if len(parts) != 2:
            raise ValueError(f"not a lon,lat pair: {chunk!r}")
        out.append((float(parts[0]), float(parts[1])))
    return out


def load_runs(patterns):
    """Sweep runs from this script's own JSON, or from sar_quality's."""
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            if isinstance(doc, dict) and "runs" in doc:
                runs.extend(doc["runs"])         # written by this script
                continue
            size = size_from_name(path)          # written by sar_quality.py
            if size is None:
                print(f"[sweep] {path}: no window size in the name and none in "
                      f"the file; skipped", file=sys.stderr)
                continue
            reports = doc if isinstance(doc, list) else [doc]
            runs.append({"size": size, "max_lag": lag_for_size(size),
                         "reports": reports})
    return runs


# ── END PURE HELPERS ─────────────────────────────────────────────────────────


def run_sweep(paths, centers, sizes, band, hist_bins=1):
    """Call sar_quality.report once per centre and size, keeping what it returns.

    Imported here rather than at the top so the pure slice above stays usable
    -- and testable -- on a machine with no rasterio.
    """
    try:
        import sar_quality
    except ImportError as e:
        # The same courtesy DPQED_gcov2tif.py extends for h5py: a missing
        # dependency is a sentence, not a traceback through someone else's
        # module. sar_quality opens rasterio and numpy at import, so this
        # catches either, and --from-json still works without both.
        raise SystemExit(
            f"sar_quality_sweep needs sar_quality.py and what it imports "
            f"({e.name}) to open rasters.\n"
            f"  pip install rasterio numpy\n"
            f"  -- or work from JSON already written: "
            f"sar_quality_sweep.py --from-json 'sweep_*.json'")

    runs = []
    for center in centers:
        for size in sorted(sizes):
            max_lag = lag_for_size(size)
            print(f"[sweep] {fmt_center(center)}  size {size} px, "
                  f"max_lag {max_lag}", file=sys.stderr)
            reports = []
            for path in paths:
                # report() prints a page per raster; the sweep's own table is
                # the output that matters, so its stdout is captured, dropped.
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        reports.append(sar_quality.report(
                            path, size, center, max_lag, band,
                            hist_bins=hist_bins, show_hist=False))
                except Exception as e:          # one bad size is not the sweep
                    print(f"[sweep] {os.path.basename(path)} at {size}: {e}",
                          file=sys.stderr)
            runs.append({"size": size, "max_lag": max_lag,
                         "center": list(center) if center else None,
                         "reports": reports})
    return runs


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rasters", nargs="*",
                   help="GeoTIFFs to sweep; convert a GCOV '.h5' with "
                        "DPQED_gcov2tif.py first")
    p.add_argument("--center", help="lon,lat to centre every window on. "
                                    "Without it the windows sit mid-scene, "
                                    "which for a slanted swath may be nodata")
    p.add_argument("--centers", help="several centres, 'lon,lat; lon,lat; ...' "
                                     "-- the whole size sweep is run at each "
                                     "and the table reports whether they agree")
    p.add_argument("--centers-file", help="the same list, one lon,lat per line; "
                                          "'#' starts a comment")
    p.add_argument("--sizes", help="comma-separated window sides in pixels "
                                   f"(default {','.join(str(s) for s in SIZES)})")
    p.add_argument("--band", type=int, action="append", dest="band",
                   help="band to analyse; repeatable, default all. An index is "
                        "not the same channel in two products -- check the "
                        "band name in the table")
    p.add_argument("--band-name", help="keep only bands with this description, "
                                       "which an index cannot guarantee")
    p.add_argument("--json", help="write the collected runs here, so the table "
                                  "can be redrawn without re-reading rasters")
    p.add_argument("--from-json", nargs="+", metavar="PATH",
                   help="read runs already written instead of opening rasters")
    args = p.parse_args(argv)

    rasters = args.rasters or list(RASTERS)
    if not rasters and not args.from_json:
        p.error("give at least one raster, set RASTERS at the top of this "
                "file, or pass --from-json")

    if args.from_json:
        runs = load_runs(args.from_json)
    else:
        sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
                 else list(SIZES))
        small = [s for s in sizes if s < SIZE_MIN]
        if small:
            print(f"[sweep] {small} below {SIZE_MIN} px: too few samples for a "
                  f"stable autocorrelation, dropped", file=sys.stderr)
            sizes = [s for s in sizes if s >= SIZE_MIN]
        if not sizes:
            p.error(f"no window size at or above {SIZE_MIN} px")
        try:
            centers = parse_centers(args.centers or args.center,
                                    args.centers_file)
        except (ValueError, OSError) as e:
            p.error(str(e))
        if not centers:
            centers = [tuple(c) for c in CENTERS] or [None]
        runs = run_sweep(rasters, centers, sizes, args.band)

    series = collate(runs, args.band_name or BAND_NAME)
    if not series:
        print("nothing to report: no band carried statistics", file=sys.stderr)
        return 1
    print("OVERSAMPLING SWEEP -- oversmp and aniso only; ENL, cv_floor, tex@ "
          "and cnr@\nare calibrated for single-look amplitude and are wrong on "
          "GCOV power.")
    print(format_table(series))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"runs": runs,
                       "conclusions": {f"{c}|{r}|{b}": conclude(pts)
                                       for (c, r, b), pts in series.items()},
                       "across_centres": {f"{r}|{b}": v for (r, b), v
                                          in across_centres(series).items()}},
                      fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
