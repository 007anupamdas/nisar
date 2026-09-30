# g1a_automatch — automatic geolocation assessment for G1A (and NISAR)

Measures the geolocation error of an image (G1A multi-band raster, or a NISAR
GSLC/GCOV `.h5`) against a reference collection (**L8_ref**, **C1**, or any
folder `DPQED_rival.py` understands) by automatic feature matching. It writes
**CSV files that `DPQED_rival.py` loads directly** (*Load CSV*), so every
automatic match can be inspected and edited in RIVAL.

This folder is self-contained (no imports from the rest of the repo). The
NISAR↔S1 production scripts in the repo root are separate and untouched.

## Files

| File | Role |
|------|------|
| `DPQED_automatch.py` | GUI (PyQt5 / PyQt6 / PySide6). Every setting is a control; runs jobs as a subprocess. |
| `automatch_job.py` | Job file + CLI: `template`, `preflight`, `run`, `detectors`, `inspect`. The GUI uses exactly this. |
| `automatch_engine.py` | Engine: input reader (raster bands / NISAR pols), reference search, coarse-to-fine matching, RANSAC, statistics, chip consensus. Derived from `dqeagdq_integrated_v2.py`. |
| `automatch_refs.py` | Reference discovery. The rules are **copied verbatim from `DPQED_rival.py`** (index-shp / sidecar / degree-tile); the tests check they still match. No QGIS needed. |
| `automatch_rival.py` | RIVAL CSV writer (RIVAL's own header, `csv_record` and `accuracy_stats`). |
| `automatch_imcui.py` | image-matching-webui models, **only for algorithms kornia lacks**. |
| `imw_configs.py` | imcui model catalog (resolved from the installed imcui's registry). |
| `prefetch_weights.py` | Download all kornia + imcui weights into one folder for an offline workstation. |
| `tests_automatch.py`, `synthetic_data.py` | Tests, with synthetic data whose geolocation error is known. |

## Install (workstation, 16 GB GPU)

```bash
conda create -n automatch python=3.11
conda activate automatch
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
# optional, for SuperPoint/SuperGlue/RoMa/DKM/eLoFTR/ASpanFormer:
pip install -e /path/to/image-matching-webui
```

## Use

**GUI:** `python DPQED_automatch.py`
1. *Data*: input image → **Inspect input** (lists band1…bandN or HH/HV…; tick
   the ones to run, none = all). Reference folder (mode is detected like RIVAL).
   Output folder.
2. *Detectors & matchers*: tick detectors and their matchers.
3. *Windows & offsets*: window sizes, keypoints, **max expected error**
   (default 50 km), coarse method, optional known offset from RIVAL.
4. **Preflight** (checks paths, bands, reference coverage, detectors, GPU),
   then **Run**. Results appear in the table; double-click a row to open its folder.

**Command line (same job file):**
```bash
python automatch_job.py template > job.json      # edit paths
python automatch_job.py preflight job.json
python automatch_job.py run job.json --set window_sizes=[1024,2048]
```

## Output → RIVAL

```
<output>/RIVAL_BEST_<scene>_<channel>.csv   best run per channel (highest consensus score)
<output>/rival/RIVAL_<scene>_win<W>_nf<N>_<channel>_to<ref>_<detector>_<matcher>_r<thr>.csv
<output>/rival/..._detail.csv               same rows + detector/RANSAC/chip provenance
<output>/RUN_MANIFEST.csv                   every run: status, coarse offsets, mean dE/dN, RMSE, CE90
<output>/automatch.log
```
Columns are RIVAL's export layout: `In_X, In_Y, Ref_X, Ref_Y, DX_Err, DY_Err,
Row, In_Lon, In_Lat, Ref_Lon, Ref_Lat`. In = the assessed image, Ref = the
reference, in the working CRS (the input's projected CRS, or the UTM zone of
its centre when the input is in lon/lat, the same rule RIVAL uses). DX/DY =
In − Ref. In RIVAL: load the input raster and the reference folder, then
*Load CSV*; RIVAL recomputes RMSE and CE90 itself.

## Errors of kilometres

Windows compared at the same map position cannot match when the error is
2–30 km (a 1024 px window at 10–20 m is only 10–20 km wide). So:

1. References are searched, and cropped, **`max_expected_error_m` beyond** the
   declared overlap.
2. Per image/reference pair, a **coarse offset** is estimated at ~60 m:
   the selected matcher first (robust 2-D mode of all match offsets), then
   phase correlation of gradient magnitude, then `initial_offset_m` if given.
   Every estimate is written to `raw_matches_*/COARSE_OFFSETS.csv`.
3. Fine windows read the reference **shifted by that offset and padded by
   `search_margin_m`**. The final errors are measured from the matched
   coordinates themselves, so they contain the full offset.

The reference is reprojected onto a grid **snapped to the input's pixel grid**
(same origin modulo one pixel). With an arbitrary sub-pixel phase between the
two grids, integer-precision keypoints split the measured offset between two
values either side of the truth.

RANSAC thresholds and consensus tolerance are given **in pixels** of the
working resolution (the input's native resolution unless `target_resolution`
is set), so the defaults suit any pixel size.

## Errors that vary across the scene (internal distortion)

Scenes with a large scale or warp error do not have "an" error: in the manual
RIVAL points seen so far the east-west error ran from 17 to 40 km across one
scene, and from -8.6 to +3.9 km across another. The tool treats the error as
a field:

- **Coarse offsets per cell.** The coarse stage estimates the offset in a grid
  of cells about one window wide (`coarse_cell_km`, 0 = one window) and each
  window reads the reference at its own interpolated offset. The field is
  written to `raw_matches_*/COARSE_FIELD_pair###.csv`.
- **Search margin per window** grows by how much the offset changes between
  the window's centre and its corners.
- **Chip consistency against a surface** (`consensus_model: surface`, the
  default). Chips are compared with a robust smooth surface through all chips'
  errors (`consensus_surface`: auto picks affine / bilinear / quadratic /
  biquadratic from the chip count), with a tolerance that adapts to how well
  the surface fits, and against their neighbours when it cannot fit closely.
  `constant` (the NISAR rule: agree with the most common error) keeps only the
  chips near that value when the error varies -- on a synthetic 6 % scale
  error it kept a handful of 25 windows. `none` keeps every chip that passed
  RANSAC.
- **Distortion summary** in `RUN_MANIFEST.csv` and the GUI results: error range
  (`dE_min_m`..`dE_max_m`, `dN_min_m`..`dN_max_m`), an affine fit of the error
  over the scene (`affine_dE_m`, `affine_dN_m` at the points' centroid,
  `affine_rot_deg`, `affine_scale_E_ppm`, `affine_scale_N_ppm`,
  `affine_shear_ppm`) and `affine_resid_rmse_m`, the distortion left after the
  best affine correction.

How to read the results:

1. The per-point errors in the RIVAL CSV *are* the result. Load it in
   `DPQED_rival.py` (or `quiver.py`) to see the field; a mean, RMSE or CE90
   over such a scene mixes different errors and describes no single place.
2. `affine_*` says how much is a simple rotation / scale / shear (a sensor or
   projection model error that a first-order correction removes);
   `affine_resid_rmse_m` says how much is left for a higher-order model.
3. Keep windows small enough that the error is close to linear inside each one
   (per-window RANSAC fits an affine); where the error curves strongly, try
   smaller windows in the window-size sweep.
4. With a manual GCP CSV the consensus score uses each chip's distance from
   the GCP-fitted error surface, which is the most direct check of the result.

## Detectors

kornia first: `sift`, `disk`, `dedode`, `aliked`, `xfeat`, `xfeatstar`
(XFeat* semi-dense), `keynet`, `loftr`. Matchers: `smnn`; `ada` (AdaLAM) for
every sparse detector (SIFT, DISK, DeDoDe, ALIKED, XFeat, KeyNet); `lgm`
(LightGlue) for DISK, ALIKED and DeDoDe (`dedodeb`/`dedodeg` weights chosen
from the descriptor). kornia's SIFT LightGlue found no matches on upright
RootSIFT and XFeat has no kornia LightGlue weights, so neither is offered.
AdaLAM uses kornia's own defaults (search expansion 4, 128 iterations, min
confidence 200); the NISAR-S1 pipeline's 1 / 2048 / 1000 can be set in
Configure…. imcui adds only what kornia lacks:
SuperPoint+LightGlue, SuperPoint+SuperGlue, eLoFTR, ASpanFormer, RoMa, DKM.
imcui's DISK/ALIKED/SIFT/XFeat/LoFTR variants are refused in favour of
kornia's. `python automatch_job.py detectors` lists what is available.

### Detector and matcher parameters

In the GUI, **Configure…** on a detector's row opens its parameters. Tick
several choices, or type a comma-separated list, to try each value:

| Detector | Parameters |
|---|---|
| `sift` | RootSIFT on/off, upright on/off, response threshold |
| `disk` | weights: depth / epipolar |
| `dedode` | detector weights × descriptor weights, as listed by the installed kornia (0.8: L-upright / L-C4 / L-SO2 / L-C4-v2 × B-/G- upright / C4 / SO2; G-* load a 1.2 GB DINOv2-L) |
| `aliked` | model t16 / n16 / n16rot / n32, detection threshold, NMS radius |
| `xfeat` | detection threshold |
| `keynet` | upright on/off, response threshold |
| `loftr` | weights: outdoor / indoor / indoor_new |
| LightGlue (`lgm`) | filter threshold, depth confidence, width confidence |
| AdaLAM (`ada`) | search expansion, RANSAC iterations, min confidence, min inliers, refit, mutual-NN seeds |
| imcui models | detection / match threshold, plus each model's own settings read from the installed imcui |

- Every combination of **detector** values runs as its own variant, named
  after the values that differ from the default (`sift-rs0` = RootSIFT off,
  `dedode_L-C4-v2_B-upright`, `aliked_aliked-n16-dt0p3`). Variants compete
  with each other and with the other detectors for `RIVAL_BEST_*`.
- **Matcher** values (LightGlue, AdaLAM) add matching passes inside one
  variant (`lgm`, `lgm_lf0p2`, …) and compete in that variant's consensus,
  like the SMNN thresholds.
- Preflight shows how many matching passes the job will make. Only one model
  is loaded at a time, whatever the number of variants.

The same thing in a job file:
```json
"detector_params": {
  "sift":   {"rootsift": [true, false]},
  "dedode": {"detector_weights": ["L-C4-v2"], "descriptor_weights": ["B-upright", "G-upright"]},
  "disk":   {"checkpoint": ["depth", "epipolar"], "lgm.filter_threshold": [0.1, 0.2]}
}
```

**What "best" means.** Without a manual GCP CSV the consensus score is the
number of matches that agree with each other, which favours dense matchers
and high keypoint counts. With `manual_gcp_csv` the score uses each chip's
distance from the GCP-fitted error surface, which is the better basis for
choosing between variants.

## 16 GB GPU

Defaults: 1024 px windows, one model resident at a time (unloaded after its
run), windows skipped rather than crashing when less than `min_gpu_free_gb`
is free. Dense imcui models (RoMa, DKM) are the heaviest; keep them at 1024 px.

## Offline weights

Weights are looked for, before any download, in kornia's default folder
(`~/.cache/torch/hub/checkpoints`, on Windows `%USERPROFILE%\.cache\torch\hub\checkpoints`),
in `TORCH_HOME` if set, and in the GUI's **Weights folder** (`weights_cache_dir`).
Weights you downloaded earlier with kornia are therefore used as they are.

For anything missing, on a connected machine run
`python prefetch_weights.py /path/cache` (every selectable weight: both DISK
checkpoints, all DeDoDe, ALIKED and LoFTR weights, and the imcui models;
`--defaults-only` for just the defaults), copy the folder and set it as the
Weights folder. A model whose weights cannot be found stops that detector at
once with a message naming the folders searched; the other detectors go on.

## Tests

```bash
python tests_automatch.py                                  # fast
python tests_automatch.py --e2e --rival /path/DPQED_rival.py
```
The e2e test builds a 3-band raster whose georeferencing is wrong by
(4013 m, −2487 m), a C1 degree-tile set and an L8 index collection, runs the
CLI with SIFT and requires the RIVAL CSV to recover the offset within half a
pixel (achieved: C1 4013.3 / −2487.2 m, L8 4013.5 / −2484.4 m).

## Moving to its own repository

```bash
git subtree split --prefix=g1a_automatch -b g1a-automatch
git push <new-repo-url> g1a-automatch:main
```
(or simply copy this folder).

## Known limits

- **G1A format:** read through rasterio (GeoTIFF/VRT/JP2/…, any georeferenced
  multi-band raster). A product that needs per-pixel lat/lon arrays (no
  affine georeferencing) needs a small reader added to `InputScene`.
- Not verified here: DeDoDe (weights could not be fetched in the test
  sandbox; the class is unchanged from the original pipeline) and the imcui
  models (imcui was not installed in the sandbox).
- kornia's LightGlue has no XFeat weights, and its KeyNet weights returned no
  matches in testing, so `lgm` is not offered for `xfeat` or `keynet`.
