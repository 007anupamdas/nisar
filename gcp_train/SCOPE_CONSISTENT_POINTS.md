# Scope: training pairs from automatch's consistent points

Status: proposal, not built yet.

## Why

`pairs.py` labels crops from one smooth field fitted through a scene's manual
points. It accepts a scene only when that field fits within 1.5 px
(leave-one-out). G1A scenes never pass:

| Sensor | Leave-one-out fit of the truth | At 10 m pixels |
|---|---|---|
| MV, best | 69–180 m (jobs 1099, 1092, 1090, 1087, 1082) | 7–18 px |
| MV, typical | 250–840 m | 25–85 px |
| HS / HV | 0.3–3 km | |

Either the manual points are a few pixels off, or the distortion varies more
locally than a smooth field can follow. Either way the manual points cannot
label G1A crops.

automatch already finds thousands of **consistent points** per scene. These
are matches where several unrelated detectors agree within the cross-check
tolerance (about 3 px). They are dense enough to label each crop on its own,
from the points inside it, so local distortion no longer matters.

## The labels: one homography per crop

For each crop of an accepted scene:

1. Take the consistent points inside the crop, plus a 10 % margin.
2. Fit a homography from input pixel to reference ground with RANSAC.
3. Accept the crop only when **all** of these hold:
   - **Points:** at least 20 points.
   - **Agreement:** agreed by at least 3 detectors (`n_agree`), from at least
     2 families, e.g. one dense (RoMa / MINIMA) and one sparse (DISK / DoG).
   - **Fit:** residual RMS at most 1.0 px, and at most 10 % RANSAC outliers.
   - **Coverage:** the points span at least half the crop (convex hull), so
     the homography is not extrapolated.
   - **Masks:** cloud and featureless area below the limits already used by
     `pairs.py`.
4. Everything after this is unchanged: reference resampling, random shift,
   the 11 × 11 grid check, PNGs and `pairs.csv`. `crop_homography` takes the
   per-crop model in place of the scene model.

Expected yield per MV scene: 8,000–21,000 consistent points (jobs 1079, 1080).
With 512 px crops (about 5 km), that is about 40–100 accepted crops. Twenty MV
scenes give about 1,000–2,000 pairs, which is enough to fine-tune LightGlue
from its official weights.

## The danger: learning only what already works

The labels come from where the current detectors already agree. A model
trained on them could simply learn those places again. Three guards:

- **Whole crops are learnt, not only the points.** The homography supervises
  every pixel of the crop, including ground between the points where no
  detector matched.
- **Agreement must cross families.** A dense and a sparse method rarely share
  a mistake.
- **The truth files judge, and never train.** Every scene with a truth file
  is held out of training. The model is scored there, end to end in
  automatch, against the manual points (see Evaluation below).

## What gets built

| Piece | Where | Size |
|---|---|---|
| `consistent.py`: read automatch job folders. Uses `job_used.json` (input, band, reference), `RIVAL_CONSISTENT_<band>_64px_detail.csv` (`n_agree`, `agree_spread_m`, the detectors) and `CROSS_SUMMARY.csv` (reliable or not). Writes a scenes manifest with `labels=consistent` and the point file. | gcp_train | ~150 lines |
| `LocalModel`: the per-crop homography, with the same interface as `SceneModel` (`__call__`, `pixel_m`); `build_scene` picks it when `labels=consistent`. | gcp_train `pairs.py`, `fit.py` | ~100 lines |
| Split rules: every scene with a truth file goes to test; the rest are split by scene. | gcp_train `pairs.py` | small |
| Tests: a synthetic scene with known local distortion, consistent points with noise and 10 % outliers. Checks: crop homographies recovered within 0.5 px, sparse crops refused, truth scenes never in train. | gcp_train `tests.py` | ~120 lines |
| A setting that loads fine-tuned LightGlue weights (and the conditioning) for `disk` / `aliked` + `lgm`. It only reads a weights file and imports nothing from gcp_train. | g1a_automatch | ~60 lines |

That is about 1.5–2 days of work, then GPU time: the smoke run takes minutes,
and 20 epochs on about 2,000 pairs take a few hours on the A100.

## Running it

1. **Make sure each job has consistent points.** Jobs from before the
   agreement levels get them without matching again:
   `python automatch_cross.py <output> --batch --jobs 1061-1104 --recompute`.
2. **Build the pairs:** `consistent.py` → `scenes.csv`, then
   `pairs.py --labels consistent`.
3. **Train:** DISK + LightGlue on MV first. It is the steadiest MV method and
   gcp_train already has `configs/disk_lightglue_gcp.yaml`. Run
   `"smoke": true` first, then 20 epochs. Then the same with
   `"train_extractor": true`, because the problem is too few correct DISK
   keypoints, not wrong matches.
4. **Evaluate:**
   - `evaluate.py` on the test pairs (precision, homography success). This is
     only a sanity check, since those labels come from the same source.
   - **The real test, in automatch:** run the held-out truth scenes with the
     default DISK + its best matcher and with the fine-tuned DISK +
     LightGlue, as two detectors in one job, then compare TRUTH_RANKING and
     `TRUTH_MATCH_ACCURACY.csv`.

## When it counts as better

On the held-out MV truth scenes, compared with default DISK in the same jobs:

- **More correct matches:** correct matches per window clearly up. Default
  DISK has about 10 (median, jobs 1061+).
- **No loss in accuracy:** truth RMSE equal or better (median over the
  scenes), and truth points reached equal or more.
- **No new failures:** no scene that default DISK gets right goes wrong.

If it doesn't meet these, the default weights stay. The next things to try
would be ALIKED + LightGlue, or a dense matcher trained on the same pairs.

## Open questions

- **Which job output folders the server still holds.** Training needs
  `job_used.json`, the consistent-point files, and the input and reference
  rasters. The match files themselves are not needed.
- **How accurate the manual points are.** If they are good to 1–2 px, the
  large fit residuals are real local distortion. That makes per-crop labels
  the only workable option, and it also caps how close any method can get to
  a smooth truth field.
- **Sensors:** MV only at first. HS and HV have fewer scenes and larger
  distortion, so they come after MV shows a gain.
