"""Exercise DPQED_radial's quality helpers without QGIS.

The pure helpers in DPQED_radial.py sit between explicit markers; this execs
that exact slice, so what is tested is the code that ships rather than a copy
of it. Qt, QGIS and GDAL are absent here, which is the point: whether a
product is oversampled, and by how much, is arithmetic over an array and
should not need a map canvas to be checked.

    python3 tests_radial_quality.py        # exits non-zero on any assertion

The interesting cases are the ones a plausible implementation gets quietly
wrong: the speckle constant for power rather than amplitude, which moves every
ENL by 3.66; a sub-crop taken from a corner rather than concentrically, which
measures four different places and calls it a window-size sweep; a bright
target's sidelobes averaged into a plateau; and the difference between a
plateau and a sequence still falling at the smallest crop, which is the
difference between a measurement and an upper bound.
"""
import ast
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "DPQED_radial.py")

BEGIN = "# ── BEGIN PURE HELPERS"
END = "# ── END PURE HELPERS"


def load_helpers(path=SRC):
    """The module's constants and its pure slice, without importing Qt."""
    text = open(path, encoding="utf-8").read()
    safe = {"tuple": tuple, "sorted": sorted, "re": re, "math": math}
    consts = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            try:
                consts[target.id] = eval(
                    compile(ast.Expression(node.value), path, "eval"),
                    dict(safe), dict(consts))
            except Exception:
                pass
    for required in ("QUALITY_SIZES", "QUALITY_SIZE_MIN", "QUALITY_BLOCK",
                     "QUALITY_RES_EFF_FLOOR", "QUALITY_BRIGHT_RATIO",
                     "SPECKLE_CV_1LOOK_POWER"):
        assert required in consts, f"constant {required} not found"
    body = text[text.index(BEGIN):text.index(END)]
    namespace = dict(consts, np=np, math=math, re=re, os=os, print=print)
    exec(compile(body, path, "exec"), namespace)
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


def speckle(looks, n, seed=0):
    """L-look power speckle: mean 1, cv = 1/sqrt(L), the real distribution."""
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=looks, scale=1.0 / looks, size=(n, n))


def boxcar(a, k):
    """a smoothed by a k-by-k box, which gives a triangular autocorrelation."""
    out = np.copy(a)
    for axis in (0, 1):
        c = np.cumsum(np.concatenate(
            [np.zeros((1,) + out.shape[1:]) if axis == 0
             else np.zeros(out.shape[:1] + (1,)), out], axis=axis), axis=axis)
        if axis == 0:
            out = (c[k:, :] - c[:-k, :]) / k
        else:
            out = (c[:, k:] - c[:, :-k]) / k
    return out


# ── the speckle constant ─────────────────────────────────────────────────────
section("SPECKLE_CV_1LOOK_POWER: the constant the whole calibration turns on")

check("single-look POWER has cv of exactly one", H["SPECKLE_CV_1LOOK_POWER"],
      1.0)
close("the amplitude figure it is NOT is 0.523",
      math.sqrt(4.0 / math.pi - 1.0), 0.5227, 1e-4)
close("using that one instead would divide every ENL by 3.66",
      1.0 / (4.0 / math.pi - 1.0), 3.660, 0.001)


# ── looks from speckle ───────────────────────────────────────────────────────
section("quality_speckle_floor: ENL recovered from synthetic power speckle")

for looks in (1, 4, 12):
    cv_floor, enl = H["quality_speckle_floor"](speckle(looks, 256, seed=looks))
    close(f"{looks}-look speckle has cv near {1/math.sqrt(looks):.3f}",
          cv_floor, 1.0 / math.sqrt(looks), 0.06)
    close(f"and its ENL comes back near {looks}", enl, looks,
          max(0.35, looks * 0.18))

cv_floor, enl = H["quality_speckle_floor"](speckle(4, 256, seed=4))
close("the same crop read on the amplitude constant would read 3.66x low",
      enl * (4.0 / math.pi - 1.0), 4.0 * (4.0 / math.pi - 1.0), 0.3)
check("an array smaller than one block has no floor to report",
      [math.isnan(v) for v in H["quality_speckle_floor"](np.ones((8, 8)))],
      [True, True])


# ── oversampling ─────────────────────────────────────────────────────────────
section("quality_of_crop: oversampling recovered from a known correlation")

flat = H["quality_of_crop"](speckle(4, 256, seed=1), 10.0, 10.0)
close("uncorrelated speckle is critically sampled on x", flat["oversmp_x"],
      1.0, 0.12)
close("and on y", flat["oversmp_y"], 1.0, 0.12)
close("so one pixel is one independent sample",
      flat["pixels_per_sample"], 1.0, 0.25)
close("and it is isotropic", flat["aniso"], 1.0, 0.1)
check("with no bright target in it",
      flat["bright"], False)
check("and the crossing was found inside the lags", flat["saturated"], False)

# A k-box filter gives rho(lag) = 1 - lag/k, whose 1/e crossing is at 0.632k.
# Dividing by QUALITY_RES_EFF_FLOOR, which is that same 0.632, returns k --
# so this is an exact test of the floor constant, not an approximate one.
for k in (2, 3, 4):
    blurred = H["quality_of_crop"](boxcar(speckle(4, 320, seed=k), k),
                                   10.0, 10.0)
    close(f"a {k}-pixel box reads oversmp {k}", blurred["oversmp_x"], k,
          0.35)
    close(f"and {k*k} pixels per independent sample",
          blurred["pixels_per_sample"], k * k, k * k * 0.35)

close("res_eff is that width in metres, not in pixels",
      H["quality_of_crop"](boxcar(speckle(4, 320, seed=3), 3), 10.0, 10.0
                           )["res_eff_x_m"] / 10.0,
      3.0 * H["QUALITY_RES_EFF_FLOOR"], 1.0)

anis = H["quality_of_crop"](
    boxcar(speckle(4, 320, seed=7), 3)[:, ::1], 10.0, 20.0)
check("aniso uses the two postings it is given, not one",
      anis["oversmp_y"] < anis["oversmp_x"], True)


# ── texture and scene ────────────────────────────────────────────────────────
section("quality_texture and quality_scene_cv on power")

tex = H["quality_texture"](speckle(4, 256, seed=2), 0.5, 10.0, 10.0)
close("pure speckle leaves nothing above its own floor at 40 m",
      tex["tex@40m"], 1.0, 0.45)

close("a product at its own speckle floor shows no scene",
      H["quality_scene_cv"](0.5, 0.5), 0.0, 1e-12)
close("variation above the floor is scene",
      H["quality_scene_cv"](0.8, 0.5), math.sqrt((0.64 - 0.25) / 1.25), 1e-9)
check("and below it, nothing rather than a negative root",
      H["quality_scene_cv"](0.4, 0.5), 0.0)

# The Hyderabad measurement that forced this: two products over one patch of
# water, L at ENL 4.9 and S at ENL 2.7. On the amplitude cap S's own speckle
# is reported as scene contrast of 0.377; on the power cap both read zero.
AMP = math.sqrt(4.0 / math.pi - 1.0)
close("on the amplitude cap the fewer-looks product invents scene",
      math.sqrt((0.674 ** 2 - AMP ** 2) / (1.0 + AMP ** 2)), 0.377, 0.001)
check("while the better-looked one reports exactly none, an infinite ratio "
      "between two products over one patch of water",
      (0.477 ** 2 - AMP ** 2) <= 0, True)
# On the power cap neither is capped at all, and what is left is the residual
# of an imperfect speckle reference rather than a band difference.
close("on the power cap the fewer-looks product reports far less",
      H["quality_scene_cv"](0.674, 0.612), 0.241, 0.001)
close("and the other reports a figure of the same order, not zero",
      H["quality_scene_cv"](0.477, 0.450), 0.144, 0.001)
check("which turns an infinite ratio into one under two",
      H["quality_scene_cv"](0.674, 0.612)
      / H["quality_scene_cv"](0.477, 0.450) < 2.0, True)

# That residual is itself the reason the reference may not be cv_floor.
section("quality_quiet_region: where the speckle reference comes from")

uniform = speckle(4, 256, seed=11)
quiet = H["quality_quiet_region"](uniform)
check("a large crop has a quiet region", quiet is not None, True)
check("and it is five blocks a side", quiet.shape,
      (5 * H["QUALITY_BLOCK"], 5 * H["QUALITY_BLOCK"]))
check("a crop under 160 pixels has none", 
      H["quality_quiet_region"](speckle(4, 128, seed=11)), None)

cv_floor_ref, _ = H["quality_speckle_floor"](uniform)
q_ref = float(quiet.std()) / float(quiet.mean())
check("the 5th-percentile floor sits below the quiet region's own cv",
      cv_floor_ref < q_ref, True)
check("so using it as the reference would manufacture more scene than using "
      "the region does",
      H["quality_scene_cv"](float(uniform.std()) / float(uniform.mean()),
                            cv_floor_ref)
      > H["quality_scene_cv"](float(uniform.std()) / float(uniform.mean()),
                              q_ref), True)

big_crop = H["quality_of_crop"](uniform, 10.0, 10.0)
check("a crop with a quiet region says its scene figure is usable",
      big_crop["scene_ok"], True)
close("and reports almost no scene over pure speckle",
      big_crop["cv_scene"], 0.0, 0.08)
small_crop = H["quality_of_crop"](speckle(4, 128, seed=11), 10.0, 10.0)
check("a crop without one says so rather than quoting a figure as if it were "
      "sound", small_crop["scene_ok"], False)


# ── crops ────────────────────────────────────────────────────────────────────
section("quality_centred and quality_fill")

grid = np.arange(100.0).reshape(10, 10)
crop = H["quality_centred"](grid, 4)
check("a crop is taken from the centre, not a corner",
      (crop.shape, float(crop[0, 0])), ((4, 4), 33.0))
check("an oversized crop does not fit", H["quality_centred"](grid, 12), None)
check("a crop the size of the array is the array",
      H["quality_centred"](grid, 10).shape, (10, 10))

holed = np.ones((10, 10))
mask = np.ones((10, 10), dtype=bool)
mask[0, 0] = False
holed[0, 0] = 999.0
filled = H["quality_fill"](holed, mask)
close("a hole is filled with the valid mean", float(filled[0, 0]), 1.0, 1e-12)
check("too many holes and the crop is refused rather than correlated",
      H["quality_fill"](holed, np.zeros((10, 10), dtype=bool)), None)
half = np.zeros((10, 10), dtype=bool)
half[:5] = True
check("half a window of fill is past the limit",
      H["quality_fill"](holed, half), None)


# ── guards ───────────────────────────────────────────────────────────────────
section("quality_bright_ratio and quality_saturated")

clean = speckle(4, 128, seed=5)
close("pure speckle sits near the fixture value", H["quality_bright_ratio"](clean),
      1.8, 0.6)
spiked = np.copy(clean)
spiked[64, 64] = clean.max() * 200.0
check("one strong scatterer is caught by the guard",
      H["quality_bright_ratio"](spiked) > H["QUALITY_BRIGHT_RATIO"], True)

check("res_eff at exactly max_lag times the posting is the ceiling",
      H["quality_saturated"](160.0, 10.0, 16), True)
check("a crossing inside the lags is a measurement",
      H["quality_saturated"](14.0, 10.0, 16), False)
check("a missing posting cannot be trusted either way",
      H["quality_saturated"](14.0, 0.0, 16), True)


# ── the sweep ────────────────────────────────────────────────────────────────
section("quality_sweep_roi: concentric crops of one ROI")

big = speckle(4, 200, seed=9)
rows = H["quality_sweep_roi"](big, None, 10.0, 10.0)
check("only the crops that fit are measured",
      [r["size"] for r in rows], [64, 96, 128, 192])
check("smallest first, which is the order the verdict reads",
      [r["size"] for r in rows] == sorted(r["size"] for r in rows), True)
check("an ROI below the floor gets no figures at all",
      H["quality_sweep_roi"](speckle(4, 48, seed=9), None, 10.0, 10.0), [])
check("nor does a one-dimensional array",
      H["quality_sweep_roi"](np.ones(400), None, 10.0, 10.0), [])


# ── verdicts ─────────────────────────────────────────────────────────────────
section("quality_plateau: four outcomes, smallest crop first")

check("scene entering the bigger crops leaves a plateau at the bottom",
      H["quality_plateau"]([1.42, 1.44, 1.45, 1.80, 2.40]),
      ("plateau", 1.42, 3))
check("a sweep that never moved was never scene-dominated",
      H["quality_plateau"]([1.40, 1.42, 1.41, 1.43]),
      ("flat throughout", 1.40, 4))
check("still climbing at the smallest crop is a bound, not a plateau",
      H["quality_plateau"]([1.50, 1.90, 2.40, 3.10]),
      ("no plateau", 1.50, 1))
check("bouncing without settling is noise",
      H["quality_plateau"]([1.50, 2.90, 1.60, 2.80]), ("unstable", 1.50, 1))
check("one crop is not a sweep", H["quality_plateau"]([1.42]),
      ("single crop", 1.42, 1))
check("nothing usable says so rather than returning a number",
      H["quality_plateau"]([])[0], "no data")
check("the tolerance is relative, so a small cell is not called a slope",
      H["quality_plateau"]([1.52, 1.61, 1.64]),
      ("flat throughout", 1.52, 3))
check("and a large cell gets no free pass from the same figure",
      H["quality_plateau"]([9.00, 9.80, 11.00])[0], "no plateau")


# ── one ROI's conclusion ─────────────────────────────────────────────────────
section("quality_conclude_roi: which crops may decide the verdict")

def row(size, pps, **flags):
    r = {"size": size, "pixels_per_sample": pps,
         "saturated": False, "bright": False}
    r.update(flags)
    return r

good = [row(64, 1.52), row(96, 1.61), row(128, 1.64), row(192, 3.19)]
c = H["quality_conclude_roi"](good)
check("the plateau is found", c["verdict"], "plateau")
close("and read off the smallest crops", c["pixels_per_sample"], 1.52, 1e-9)
check("over the crops that agreed", c["sizes"], [64, 96, 128])

mixed = good[:3] + [row(192, 3.19, bright=True), row(256, 99.0, saturated=True)]
c = H["quality_conclude_roi"](mixed)
check("a bright crop cannot decide the verdict", c["verdict"],
      "flat throughout")
check("and both disqualified crops are counted", (c["excluded"], c["crops"]),
      (2, 5))
check("a crop carrying scene is NOT disqualified, which is the premise",
      H["quality_conclude_roi"](good)["crops"], 4)
check("every crop disqualified leaves nothing to conclude",
      H["quality_conclude_roi"]([row(64, 1.5, bright=True)])["verdict"],
      "no data")


# ── pooling by class ─────────────────────────────────────────────────────────
section("quality_by_class: every ROI, grouped by what it is over")

per_roi = [
    ("water", {"verdict": "plateau", "pixels_per_sample": 1.52}),
    ("water", {"verdict": "flat throughout", "pixels_per_sample": 1.60}),
    ("water", {"verdict": "no plateau", "pixels_per_sample": 4.40}),
    ("vegetation", {"verdict": "plateau", "pixels_per_sample": 2.10}),
    ("vegetation", {"verdict": "unstable", "pixels_per_sample": 9.90}),
]
by = H["quality_by_class"](per_roi)
check("one entry per class", sorted(by), ["vegetation", "water"])
close("the median is over the ROIs that settled", by["water"]["pixels_per_sample"],
      1.56, 1e-9)
check("an upper bound does not join the median",
      (by["water"]["rois"], by["water"]["settled"]), (3, 2))
close("and the spread is the half-range over the median",
      by["water"]["spread"], (1.60 - 1.52) / 2.0 / 1.56, 1e-9)
check("a class where only one ROI settled still reports it",
      (by["vegetation"]["rois"], by["vegetation"]["settled"]), (2, 1))
check("and a class where none did says so",
      math.isnan(H["quality_by_class"](
          [("sand", {"verdict": "unstable", "pixels_per_sample": 3.0})]
      )["sand"]["pixels_per_sample"]), True)


# ── sizing an ROI for this ───────────────────────────────────────────────────
section("quality_side_for_sweep: how large an ROI has to be")

check("at 10 m posting the sweep needs a 5.12 km side",
      H["quality_side_for_sweep"](10.0), 5200.0)
check("at 30 m it needs three times that",
      H["quality_side_for_sweep"](30.0), 15400.0)
check("a smaller sweep needs a smaller ROI, rounded to something typable",
      H["quality_side_for_sweep"](10.0, 128), 1300.0)
check("a posting that is not a number gets no suggestion",
      H["quality_side_for_sweep"]("wide"), None)
check("nor does one at zero", H["quality_side_for_sweep"](0.0), None)

side = H["quality_side_for_sweep"](10.0)
check("which is far larger than the point buffer's backscatter default",
      side > 10 * H["POINT_SIDE_DEFAULT_M"], True)


print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print("  -", failure)
    sys.exit(1)
print("all checks passed")
