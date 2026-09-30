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
| `automatch_job.py` | Job file + CLI: `template`, `preflight`, `run`, `detectors`, `weights`, `compare`, `inspect`. The GUI uses exactly this. |
| `automatch_engine.py` | Engine: input reader (raster bands / NISAR pols), reference search, coarse-to-fine matching, RANSAC, statistics, chip consensus. Derived from `dqeagdq_integrated_v2.py`. |
| `automatch_refs.py` | Reference discovery. The rules are **copied verbatim from `DPQED_rival.py`** (index-shp / sidecar / degree-tile); the tests check they still match. No QGIS needed. |
| `automatch_rival.py` | RIVAL CSV writer (RIVAL's own header, `csv_record` and `accuracy_stats`). |
| `automatch_imcui.py` | image-matching-webui models, **only for algorithms kornia lacks**. |
| `imw_configs.py` | imcui catalogue: the full matrix (22 sparse + 16 dense rows), resolved from the installed imcui's registry. |
| `automatch_weights.py` | Which weight files each detector needs and whether they are on this machine (nothing is downloaded). |
| `automatch_truth.py` | Ranks every detector + matcher of a run against manually measured ground truth. |
| `prefetch_weights.py` | Collect all kornia + imcui weights into one folder for an offline workstation. |
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
2. *Detectors & matchers*: tick detectors and their matchers. **Check weights**
   shows which weight files they need and whether they are on this machine.
3. *Windows & offsets*: window sizes, keypoints, **max expected error**
   (default 50 km), coarse method, optional known offset from RIVAL.
4. **Preflight** (checks paths, bands, reference coverage, detectors, weight
   files, GPU), then **Run**. Results appear in the table; double-click a row
   to open its folder.

**Command line (same job file):**
```bash
python automatch_job.py template > job.json      # edit paths
python automatch_job.py preflight job.json
python automatch_job.py run job.json --set window_sizes=[1024,2048]
python automatch_job.py weights job.json          # weight files that job needs
python automatch_job.py compare <output> --truth manual_pts_rival.csv
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

`python automatch_job.py detectors` (or the GUI's list) shows the whole
catalogue: what can be selected, and every imcui row that is not offered with
the reason.

**kornia first.** Detectors: `sift`, `disk`, `dedode`, `aliked`, `xfeat`,
`xfeatstar` (XFeat* semi-dense), `keynet` (with or without AffNet), `dog` (DoG
keypoints with a learned descriptor: HardNet, HardNet8, SOSNet, HyNet or
TFeat, AffNet on/off), `gftt` and `hessian` (GFTT- / Hessian-AffNet-HardNet),
`loftr`.

Matchers run by default: `smnn`; `ada` (AdaLAM) for every sparse detector;
`lgm` (LightGlue) for SIFT, DISK, DeDoDe (`dedodeb`/`dedodeg` weights from the
descriptor), ALIKED, KeyNet and DoG-HardNet. kornia's LightGlue cannot run
XFeat, so XFeat has no `lgm`. Offered but run only when ticked: `mnn` (mutual
nearest neighbour), `snn` (ratio test), `nn` and `fginn`. On a synthetic pair
shifted by (7, −4) px every detector recovered the shift with every matcher.
AdaLAM uses kornia's own defaults (search expansion 4, 128 iterations, min
confidence 200); the NISAR-S1 pipeline's 1 / 2048 / 1000 can be set in
Configure….

**imcui adds only what kornia lacks.** The catalogue is the full matrix
(`imw_configs.py`): SuperPoint + LightGlue / SuperGlue / mutual-NN, R2D2,
RoRD, D2-Net, ALIKE, SFD2, RDD, LiftFeat, RIPE, DarkFeat, LANet (each with
mutual-NN), MINIMA-LoFTR / -RoMa, XoFTR, OmniGlue, GIM-RoMa / -DKM, eLoFTR,
ASpanFormer, TopicFM, RoMa, DKM, DaD-RoMa, RDD-dense and XFeat+LightGlue. Rows
kornia covers are refused, each with the kornia replacement named (DISK/ALIKED
/SIFT LightGlue → `disk`/`aliked`/`sift` with `lgm`; `hardnet-nn`,
`sosnet-nn` → `dog` with that descriptor and `mnn`; `rootsift-nn` → `sift`;
`dedode-nn` → `dedode`; AdaLAM rows → `ada`; `loftr` → `loftr`;
`xfeat-dense` → `xfeatstar`). MINIMA and GIM rows are always offered: kornia
does not ship those weights. Rows missing from the installed imcui are listed
as such.

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
| `keynet` | AffNet on/off, upright on/off, response threshold |
| `dog` | descriptor HardNet / HardNet8 / SOSNet / HyNet / TFeat, AffNet on/off, upright on/off, response threshold |
| `gftt`, `hessian` | upright on/off |
| `loftr` | weights: outdoor / indoor / indoor_new, coarse match threshold (kornia default 0.2) |
| LightGlue (`lgm`) | filter threshold, depth confidence, width confidence |
| AdaLAM (`ada`) | search expansion, RANSAC iterations, min confidence, min inliers, refit, mutual-NN seeds |
| SNN (`snn`), FGINN (`fginn`) | ratio threshold; FGINN also spatial threshold and mutual check |
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

## Which detector + matcher is most accurate? (ground truth)

Give the manually measured points (a RIVAL CSV such as `manual_pts_rival.csv`:
`In_X … In_Lon, In_Lat, Ref_Lon, Ref_Lat`) as **Ground truth CSV** (`truth_csv`).
After the run, **every configuration each detector ran** -- detector variant ×
matcher × matcher setting × RANSAC setting, not only the one the consensus
picked -- is scored against it:

- at each truth point the configuration's own error is estimated: the median
  of its matches within `truth_radius_m` (5 km), else its robust error
  surface; points outside its coverage are listed as extrapolated and not
  scored. Truth errors are recomputed from the lon/lat columns in the working
  CRS, so the truth file's own X/Y grid does not matter;
- `TRUTH_BY_DETECTOR_MATCHER.csv`: each detector + matcher at its best setting,
  ranked by truth points reached, then RMSE of (tool − truth), with the bias
  (mean dE, dN) and the largest disagreement. `TRUTH_RANKING.csv` has every
  configuration, `TRUTH_POINTS.csv` every configuration × truth point;
- the GUI shows the ranking when the run ends and adds `truth_rmse_m` to each
  result row; `RUN_MANIFEST.csv` gets the same columns.

Existing results can be scored again, e.g. with more truth points, without
matching again:
```bash
python automatch_job.py compare <output> --truth manual_pts_rival.csv [--radius-km 5] [--chips all]
```
To choose a detector + matcher first and fine-tune later: tick all detectors
and all matchers (including the optional ones), keep one window size and one
RANSAC setting, and read `TRUTH_BY_DETECTOR_MATCHER.csv`. The ground truth is
used for evaluation only; it does not steer matching or the consensus.

## 16 GB GPU

Defaults: 1024 px windows, one model resident at a time (unloaded after its
run), windows skipped rather than crashing when less than `min_gpu_free_gb`
is free. Dense imcui models (RoMa, DKM) are the heaviest; keep them at 1024 px.

## Offline weights

Weights are looked for, before any download, in kornia's default folder
(`~/.cache/torch/hub/checkpoints`, on Windows `%USERPROFILE%\.cache\torch\hub\checkpoints`),
in `TORCH_HOME` if set, and in the GUI's **Weights folder** (`weights_cache_dir`);
imcui's HuggingFace files likewise in the active HuggingFace cache, the Weights
folder's and `~/.cache/huggingface/hub`. Weights downloaded earlier with kornia
or imcui are therefore used as they are.

**Checking** (GUI: *Check weights*; CLI below; preflight does it for the
selected detectors): every model is built on the CPU while its weight requests
are intercepted and looked up; nothing is downloaded and no weight file is
read, so the whole catalogue takes about half a minute. For each detector it
lists every file with its model, status (ok / MISSING / TRUNCATED -- an
interrupted download / EMPTY), the folder it was found in, and for missing
files the URL and the exact name to save it under.
```bash
python automatch_weights.py                          # every detector, every weight choice
python automatch_weights.py --detectors dedode,dog   # or: automatch_job.py weights job.json
python automatch_weights.py --load                   # also load each model from the files (slower)
```
Some imcui models load their weights in ways the quick check cannot follow;
they are marked `partial` -- use `--load` for those.

For anything missing, on a connected machine run
`python prefetch_weights.py /path/cache` (every weight choice of every
detector and the imcui models; `--defaults-only` for just the defaults;
`--only dog,sift`): files already on that machine are copied into the folder,
the rest downloaded under the names the loaders expect. Copy the folder and
set it as the Weights folder. A model whose weights cannot be found stops that
detector at once with a message naming the folders searched; the other
detectors go on.

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
- Not verified here: the imcui models themselves (imcui is not installed in
  the test sandbox; the bridge, the catalogue and the HuggingFace lookup are
  tested against a mock with imcui's registry layout).
- LightGlue for SIFT and KeyNet used to find no matches: their descriptors
  came as a (1, N, D) batch, which kornia's LightGlue matcher reads as a
  single descriptor. Fixed; with LightGlue they now match (95–100 % correct
  on the synthetic test).
- LoFTR used a configuration (Sinkhorn coarse matching with an untrained
  dustbin score, threshold 1) that finds no matches with the released weights,
  and a module kornia does not ship. It now uses kornia's own configuration for
  those weights (dual softmax, threshold 0.2, settable).
