"""Measure what a SAR raster actually delivers: distribution, blur, noise, texture.

Written because two NISAR GSLC products posted at the same 5 m looked nothing
alike -- one crisp, one soft and grainy -- and "it looks blurry" is not a
finding. Every number here is something you can put in a report next to the
product it came from.

Four groups, per band, over a window you choose:

DISTRIBUTION
    valid, min, p1, p2, median, mean, p98, p99, max, std, cv, and the dynamic
    range in dB. The percentiles matter more than min/max: one bright scatterer
    or one un-flagged fill pixel sets min/max and tells you nothing.

    A histogram is printed with them, binned over p1..p99 rather than min..max
    for the same reason -- a single outlier otherwise puts every real pixel in
    the first bin -- with the counts that fall outside reported at the ends.
    The shape says things the summary cannot: single-look SAR amplitude over
    homogeneous ground is Rayleigh, leaning left with a long tail, so a
    symmetric bell means something has averaged the image, a spike at zero
    means fill is being counted as data, and a second mode means the window
    straddles two surfaces and the looks estimate below will read low. Exact
    bin edges and counts go to --json for plotting elsewhere.

BLUR -- how much ground one pixel really represents
    rho1        lag-1 autocorrelation of amplitude, along each map axis. A
                critically sampled image has near-independent neighbours
                (~0.0-0.15). Correlated neighbours mean the scene was resolved
                more coarsely than it was posted.
    res_eff     effective resolution: the lag where autocorrelation falls to
                1/e, times the pixel size.
    oversmp     res_eff / posting. 1.0 is critically sampled; 3.0 means the
                product carries a third of the detail its grid implies.
    aniso       res_eff across / along. SAR resolves range and azimuth
                independently, so a product coarser in range than azimuth
                blurs in one direction -- which is what a slant-range figure
                quoted as if it were ground range looks like. Geocoding rotates
                range/azimuth onto the map axes by the track heading, so read
                this as "is there a preferred direction", not an exact ratio.

NOISE -- how much of the variation is speckle rather than scene
    enl         equivalent number of looks, estimated from the most homogeneous
                sub-block in the window rather than the whole window, so a
                field boundary crossing the sample does not corrupt it. Single
                look amplitude sits at 1.0. Above ~1.5 something has smoothed
                the image; well below 1.0 means even the quietest block still
                holds real scene variation.
    cv_floor    the coefficient of variation of that block. 0.52 is fully
                developed single-look speckle.

TEXTURE -- whether features survive at a given scale, which is what "I can see
    it in one image and not the other" actually means
    tex@10m     ratio of the variation left after averaging to the variation
    tex@20m     speckle alone would leave, at 10/20/40 m blocks. 1.0 means the
    tex@40m     scene is indistinguishable from speckle at that scale: there is
                nothing there to see. 2.0 means real structure dominates.
                Correlation from oversampling is divided out using res_eff, so
                a soft image is not credited with texture it does not have.

CALIBRATION
    Measured against synthetic single-look speckle with known properties, at
    5 m posting, so the columns can be read rather than guessed at:

      fixture                       oversmp   ENL   tex@10m  @20m  @40m
      critically sampled, no scene    1.00    1.08    1.04    1.04  1.04
      same, 3x oversampled            2.50    1.15    0.89    0.93  1.06
      structured scene                1.07    1.08    1.11    1.40  2.22
      same scene, 3x oversampled      2.63    1.15    0.94    0.99  1.33

    So: oversmp reads 1.00 when resolution matches posting and 2.50 at a true
    3x; ENL runs ~8% high on a genuine single-look image, so treat 1.0-1.2 as
    one look; texture sits near 1.0 when there is nothing to see and climbs
    with scale when there is. The last two rows are the same ground truth --
    blurring it 3x cuts the 40 m texture from 2.22 to 1.33, which is precisely
    the "visible in one product, not the other" complaint, quantified.

Usage:
    python sar_quality.py lsar.tif ssar.tif --center 78.03,16.80 --size 1024
    python sar_quality.py scene.tif --band 1 --size 2048 --json out.json

Needs numpy and rasterio. Reads windows, so scene size does not matter.
"""
import argparse
import json
import math
import sys

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

# Coefficient of variation of fully developed single-look SAR amplitude. The
# Rayleigh distribution's sigma/mean = sqrt(4/pi - 1); every looks estimate
# below is calibrated against it.
SPECKLE_CV_1LOOK = math.sqrt(4.0 / math.pi - 1.0)

# Block side used to hunt for the most homogeneous patch when estimating looks,
# and the percentile of block cv taken as the speckle floor. Small enough to
# land inside one field, large enough for a stable variance. The strict minimum
# over a thousand blocks is an order statistic and reads ~7% low, which inflates
# the looks estimate by 15%; the 5th percentile is just as robust to a field
# edge crossing the sample and is very nearly unbiased.
ENL_BLOCK = 32
ENL_PERCENTILE = 5.0

# The 1/e width of a perfectly uncorrelated field is not zero: with rho(1) = 0
# the crossing interpolates to 1 - 1/e of a pixel. Effective resolution is
# divided by this to give a correlation length that reads 1 pixel when the
# image is critically sampled, so `oversmp` is 1.00 there rather than 0.63.
RES_EFF_FLOOR = 1.0 - 1.0 / math.e

# Ground scales, in metres, at which texture is reported.
TEXTURE_SCALES_M = (10.0, 20.0, 40.0)

# max/p99 above this means a few very bright scatterers dominate the window,
# which widens the autocorrelation and makes res_eff read coarse. Measured: a
# pure-speckle fixture sits near 1.8, a window over Hyderabad city at 33.
BRIGHT_TARGET_RATIO = 15.0

# Histogram: bins, and the width in characters of the printed bar.
HIST_BINS = 24
HIST_WIDTH = 46


def pick_window(ds, size, center):
    """A `size`-square window at `center` lon/lat when given, else mid-scene."""
    if center is None:
        col = max(0, ds.width // 2 - size // 2)
        row = max(0, ds.height // 2 - size // 2)
    else:
        lon, lat = center
        xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        row_f, col_f = ds.index(xs[0], ys[0])
        col = max(0, int(col_f) - size // 2)
        row = max(0, int(row_f) - size // 2)
    return Window(col, row,
                  min(size, ds.width - col), min(size, ds.height - row))


def distribution(values):
    """Percentile-led summary; min/max are reported but never relied on."""
    p1, p2, p50, p98, p99 = np.percentile(values, [1, 2, 50, 98, 99])
    mean = float(values.mean())
    std = float(values.std())
    lo = float(p1) if p1 > 0 else float(values[values > 0].min()) if (values > 0).any() else 0.0
    return {
        "min": float(values.min()), "p1": float(p1), "p2": float(p2),
        "median": float(p50), "mean": mean, "p98": float(p98),
        "p99": float(p99), "max": float(values.max()), "std": std,
        "cv": std / mean if mean else float("nan"),
        "dyn_range_db": 20.0 * math.log10(float(p99) / lo) if lo > 0 else float("nan"),
    }


def histogram(values, bins=HIST_BINS):
    """Counts over p1..p99, plus what fell outside at each end."""
    lo, hi = np.percentile(values, [1, 99])
    if not (hi > lo):
        lo, hi = float(values.min()), float(values.max())
    if not (hi > lo):
        hi = lo + 1e-9
    inside = values[(values >= lo) & (values <= hi)]
    counts, edges = np.histogram(inside, bins=bins, range=(float(lo), float(hi)))
    return {
        "lo": float(lo), "hi": float(hi), "bins": int(bins),
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "below": int((values < lo).sum()), "above": int((values > hi).sum()),
    }


def print_histogram(h, mean, median, width=HIST_WIDTH):
    peak = max(h["counts"]) or 1
    edges = h["edges"]
    print(f"    hist   {h['bins']} bins over p1..p99 "
          f"{h['lo']:.4g}..{h['hi']:.4g}   "
          f"{h['below']} below, {h['above']} above")
    for i, count in enumerate(h["counts"]):
        # One marker column, so mean and median are placed rather than described.
        mark = " "
        if edges[i] <= median < edges[i + 1]:
            mark = "M"
        if edges[i] <= mean < edges[i + 1]:
            mark = "X" if mark == "M" else "m"
        bar = "#" * int(round(width * count / peak))
        print(f"      {edges[i]:>10.4g} {mark}|{bar:<{width}} {count}")
    print("             (M median, m mean, X both)")


def autocorr_profile(a, axis, max_lag):
    """Correlation of the mean-removed field with itself shifted along `axis`."""
    a = a - a.mean()
    denom = float((a * a).sum())
    out = []
    for lag in range(1, max_lag + 1):
        if axis == 0:
            num = float((a[lag:, :] * a[:-lag, :]).sum())
        else:
            num = float((a[:, lag:] * a[:, :-lag]).sum())
        out.append(num / denom if denom else float("nan"))
    return out


def one_over_e_width(profile):
    """Lag where correlation first drops below 1/e, linearly interpolated."""
    target = 1.0 / math.e
    prev = 1.0
    for i, c in enumerate(profile, start=1):
        if c < target:
            span = prev - c
            return (i - 1) + (prev - target) / span if span else float(i)
        prev = c
    return float(len(profile))


def speckle_floor(a, block=ENL_BLOCK, pct=ENL_PERCENTILE):
    """cv of the quietest blocks, and the looks that implies.

    Estimating looks over a whole window measures the scene, not the speckle:
    any field edge inflates the variance and the answer comes out below one
    look, which is meaningless. The quietest blocks are the closest thing to a
    homogeneous target the image offers without being told where one is.
    """
    h = (a.shape[0] // block) * block
    w = (a.shape[1] // block) * block
    if h < block or w < block:
        return float("nan"), float("nan")
    blocks = a[:h, :w].reshape(h // block, block, w // block, block)
    means = blocks.mean(axis=(1, 3))
    stds = blocks.std(axis=(1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = np.where(means > 0, stds / means, np.nan)
    if not np.isfinite(cvs).any():
        return float("nan"), float("nan")
    cv_floor = float(np.nanpercentile(cvs[np.isfinite(cvs)], pct))
    looks = (SPECKLE_CV_1LOOK / cv_floor) ** 2 if cv_floor > 0 else float("inf")
    return cv_floor, looks


def correlation_length(res_eff_m, px_m):
    """Spacing between independent samples, in metres, floored at one pixel."""
    return max(px_m, res_eff_m / RES_EFF_FLOOR)


def texture(a, cv_floor, px_m, res_eff_m, scales=TEXTURE_SCALES_M):
    """How much variation survives averaging, against what speckle predicts.

    Averaging k x k pixels divides speckle by the square root of the number of
    INDEPENDENT samples in the block -- which is not k^2 when the image is
    oversampled, since neighbouring pixels repeat each other. The correlation
    length gives the real sample spacing, so a soft image is not credited with
    texture it does not have. What is left above 1.0 is scene.
    """
    corr_m = correlation_length(res_eff_m, px_m)
    out = {}
    for scale in scales:
        k = max(1, int(round(scale / px_m)))
        h = (a.shape[0] // k) * k
        w = (a.shape[1] // k) * k
        if h < k * 4 or w < k * 4 or not (cv_floor > 0):
            out[f"tex@{scale:g}m"] = float("nan")
            continue
        means = a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        m = float(means.mean())
        cv_obs = float(means.std()) / m if m else float("nan")
        indep = max(1.0, (k * px_m / corr_m) ** 2)
        cv_expected = cv_floor / math.sqrt(indep)
        out[f"tex@{scale:g}m"] = cv_obs / cv_expected if cv_expected else float("nan")
    return out


def analyse_band(ds, band, win, max_lag, hist_bins=HIST_BINS):
    data = ds.read(band, window=win).astype("float64")
    finite = np.isfinite(data)
    if ds.nodata is not None and not math.isnan(ds.nodata):
        finite &= data != ds.nodata
    valid = float(finite.mean())
    if valid < 0.5:
        return {"band": band, "valid": valid, "note": "under half the window is data"}

    values = data[finite]
    # A single hole poisons a correlation sum; the mean contributes nothing to
    # a mean-removed product, so it is the neutral fill.
    filled = np.where(finite, data, values.mean())

    px = abs(ds.transform.a)
    py = abs(ds.transform.e)
    rows = autocorr_profile(filled, 0, max_lag)
    cols = autocorr_profile(filled, 1, max_lag)
    res_y = one_over_e_width(rows) * py
    res_x = one_over_e_width(cols) * px
    cv_floor, looks = speckle_floor(filled)

    out = {
        "band": band,
        "name": ds.descriptions[band - 1] or f"band {band}",
        "valid": valid,
        "px_m": px, "py_m": py,
    }
    out.update(distribution(values))
    out["hist"] = histogram(values, bins=hist_bins)
    out.update({
        "rho1_y": rows[0], "rho1_x": cols[0],
        "res_eff_y_m": res_y, "res_eff_x_m": res_x,
        "oversmp_y": correlation_length(res_y, py) / py if py else float("nan"),
        "oversmp_x": correlation_length(res_x, px) / px if px else float("nan"),
        "aniso": res_x / res_y if res_y else float("nan"),
        "cv_floor": cv_floor, "enl": looks,
    })
    out.update(texture(filled, cv_floor, (px + py) / 2.0,
                       (res_x + res_y) / 2.0))
    ratio = out["max"] / out["p99"] if out["p99"] > 0 else float("inf")
    out["bright_ratio"] = ratio
    if ratio > BRIGHT_TARGET_RATIO:
        out["warning"] = (f"bright targets dominate (max/p99 {ratio:.0f}): "
                          f"res_eff reads coarse here, measure resolution over "
                          f"homogeneous terrain")
    return out


def report(path, size, center, max_lag, bands, hist_bins=HIST_BINS,
           show_hist=True):
    with rasterio.open(path) as ds:
        win = pick_window(ds, size, center)
        print(f"\n{path}")
        print(f"  {ds.width} x {ds.height} px, {abs(ds.transform.a):g} x "
              f"{abs(ds.transform.e):g} m, {ds.crs}, {ds.count} band(s), "
              f"overviews {ds.overviews(1)}, nodata {ds.nodata}")
        print(f"  window col {int(win.col_off)} row {int(win.row_off)} "
              f"{int(win.width)} x {int(win.height)}")
        results = []
        for band in (bands or range(1, ds.count + 1)):
            r = analyse_band(ds, band, win, max_lag, hist_bins)
            results.append(r)
            if "note" in r:
                print(f"  {r['band']}: {r['note']} ({r['valid']:.0%} valid)")
                continue
            print(f"  {r['name']}")
            print(f"    dist   min {r['min']:.4g}  p2 {r['p2']:.4g}  "
                  f"med {r['median']:.4g}  mean {r['mean']:.4g}  "
                  f"p98 {r['p98']:.4g}  max {r['max']:.4g}")
            print(f"           std {r['std']:.4g}  cv {r['cv']:.3f}  "
                  f"dynamic range {r['dyn_range_db']:.1f} dB  "
                  f"valid {r['valid']:.1%}")
            print(f"    blur   rho1 y {r['rho1_y']:+.2f} x {r['rho1_x']:+.2f}   "
                  f"res_eff y {r['res_eff_y_m']:.1f} m x {r['res_eff_x_m']:.1f} m   "
                  f"oversmp y {r['oversmp_y']:.2f} x {r['oversmp_x']:.2f}   "
                  f"aniso {r['aniso']:.2f}")
            print(f"    noise  cv_floor {r['cv_floor']:.3f}  ENL {r['enl']:.2f}")
            if "warning" in r:
                print(f"    !!     {r['warning']}")
            if show_hist:
                print_histogram(r["hist"], r["mean"], r["median"])
            print("    tex    " + "  ".join(
                f"{k} {r[k]:.2f}" for k in r if k.startswith("tex@")))
        return {"path": path, "bands": results}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rasters", nargs="+")
    p.add_argument("--size", type=int, default=1024,
                   help="side of the sample window in pixels (default 1024)")
    p.add_argument("--center", help="lon,lat to centre the window on; without "
                                    "it the window sits mid-scene, which for a "
                                    "slanted swath may be nodata")
    p.add_argument("--band", type=int, action="append", dest="bands",
                   help="band to analyse; repeatable, default all")
    p.add_argument("--max-lag", type=int, default=16)
    p.add_argument("--bins", type=int, default=HIST_BINS,
                   help=f"histogram bins (default {HIST_BINS})")
    p.add_argument("--no-hist", action="store_true",
                   help="skip the printed histogram; --json still carries it")
    p.add_argument("--json", help="also write the numbers to this file")
    args = p.parse_args(argv)

    center = None
    if args.center:
        lon, lat = (float(v) for v in args.center.split(","))
        center = (lon, lat)

    everything = [report(path, args.size, center, args.max_lag, args.bands,
                         args.bins, not args.no_hist)
                  for path in args.rasters]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(everything, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
