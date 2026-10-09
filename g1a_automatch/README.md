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
| `automatch_gpuaas.py` | Entry point for the GPU service (`submit_job` `code_path`). |
| `automatch_server.py` | HTTP front end for a GPU server: submit and follow jobs with curl. |
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
python automatch_job.py bands G1A_HV.tif --top 3  # band quality of a cube
```

## Hyperspectral cubes: which band (auto_bands)

The good bands of a G1A HS/HV cube change from image to image. In one, the
first bands are poor and those around 50 are crisp. In another, everything
above 40 is junk. `auto_bands: N` (GUI: *Auto bands*) matches only the N
bands with the best signal-to-noise. No reference is needed. It applies to
rasters when `channels` is `[]`; ticked channels win.

How each band is scored:
- **noise**: the pixel noise on six full-resolution 256 px blocks, measured
  with a kernel that cancels flat areas, ramps and most edges (Immerkær).
  Striping (column and row gains) is added to it.
- **signal**: the spread of the band's own values (p98 − p2), with the noise
  taken out.
- **SNR dB** is 20·log10(signal / noise); 0 dB means as much noise as scene.
- A band with no signal above its noise, or less than half of the best
  band's data, is unusable (−99).
- The picks are at least 1/30 of the band count apart (6 bands in a
  180-band cube), so a top-3 is not three neighbours of one peak.

The scores are written to `BAND_QUALITY.csv` in the output folder and the
log. Preflight prints them too, so a server preflight shows the pick before
a long run. With more than 20 bands, `channels` `[]` and `auto_bands` 0,
preflight warns that every band will be matched.

The score measures image quality, not the spectral match to the reference.
Among bands of similar SNR, the one closest to the reference's band (e.g.
Landsat red / NIR) may match better.

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
2. Per image/reference pair, a **coarse offset** is estimated at ~60 m (at
   the image's own pixel size when that is coarser, and coarser still for
   very large pairs): the selected matcher first (robust 2-D mode of all
   match offsets), then phase correlation of gradient magnitude, then
   `initial_offset_m` if given. All pairs share one image, so the estimates
   are then checked against each other. First the matcher's own: offsets
   are grouped with their neighbours (within 5 km, plus 50 m per km between
   the pairs, which the scene's own distortion needs), each group weighing
   its matches. When the heaviest group outweighs every other at least
   twice and rests on a well-supported matcher offset, matcher offsets
   outside it are replaced by the nearest group member's (`from-pair-N
   (matcher outlier)`; job 325: XoFTR had five pairs 19–51 km off on 8–13
   matches). Then a phase-correlation offset more than 5 km (or 3 x
   `search_margin_m`) from every offset the matcher measured is replaced by
   the nearest matcher pair's offset, as a failed pair is
   (`from-pair-N (phasecorr disagreed)` in the log).
   When the pairs still differ by more than half of `max_expected_error_m`,
   the log warns. Every estimate is written to `raw_matches_*/COARSE_OFFSETS.csv`.
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

A detector whose class the installed kornia lacks is not offered on that
machine (kornia 0.8.1 has no ALIKED or XFeat); imcui's version of it is
offered there instead, and `detectors` / the env check list it as
"not available here".

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

- at each truth point the configuration's own error is estimated from its
  matches within `truth_radius_m` (5 km): a robust plane through them,
  evaluated at the truth point, so the error gradient of a distorted scene
  does not bias it (the median of the neighbours did: on a synthetic scene
  with a 60 m/km east-west gradient it scored points that are accurate to
  17 m at 82 m RMSE, the plane at 8 m; `compare --local median` gives the
  earlier rule). Without neighbours, its robust error surface; points outside
  its coverage are listed as extrapolated and not scored. Truth errors are
  recomputed from the lon/lat columns in the working CRS, so the truth file's
  own X/Y grid does not matter;
- `TRUTH_BY_DETECTOR_MATCHER.csv`: each detector + matcher at its best setting,
  with the bias (mean dE, dN) and the largest disagreement. Ranked: the
  configurations that reach at least half of the truth points first, among
  them the lowest RMSE of (tool − truth), then the most points reached. (The
  number reached used to come before the RMSE, and on job 315 a GFTT + NN
  set 63 km off at all 8 points ranked above sets within 250 m at 6 of 8.)
  A warning follows the ranking when rank 1 is more than 1 km off while a
  configuration reaching fewer points (at least 2) is three times closer:
  the matches then cover only part of the scene (job 343: 1.1 km at 5 of 9
  points against 179 m at 2 of 9), and no configuration is reliable across it;
  `TRUTH_RANKING.csv` has every configuration, `TRUTH_POINTS.csv` every
  configuration × truth point; `PERFORMANCE.csv` follows the same order;
- the GUI shows the ranking when the run ends and adds `truth_rmse_m` to each
  result row; `RUN_MANIFEST.csv` gets the same columns;
- `RIVAL_TRUTH_BEST_<channel>.csv`: the points of the configuration ranked
  first, for RIVAL. With ground truth, `RIVAL_BEST_*` is that file; the pick
  of the chip consensus is kept as `RIVAL_CONSENSUS_BEST_*` (on the first G1A
  scene the consensus picked a DeDoDe set 13 km off the truth: many
  consistent but wrong matches outvote the right ones).

Existing results can be scored again, e.g. with more truth points, without
matching again:
```bash
python automatch_job.py compare <output> --truth manual_pts_rival.csv [--radius-km 5] [--chips all]
```
To choose a detector + matcher first and fine-tune later: tick all detectors
and all matchers (including the optional ones), keep one window size and one
RANSAC setting, and read `TRUTH_BY_DETECTOR_MATCHER.csv`. The ground truth is
used for evaluation only; it does not steer matching or the consensus.

**Manual GCP CSV is a different input**: NISAR-style GCPs (`scan, pix, Map_X,
Map_Y, Map_X_ref, Map_Y_ref`) that steer the chip consensus. A RIVAL file
there is refused by preflight; leave it empty when choosing a detector +
matcher (the same file in both fields would score the consensus with the
points that picked it). If a run's consensus did not complete, `compare`
rebuilds it from the saved matches, so the run can still be ranked without
matching again.

## GPU service (gpuaas submit_job): step by step

The GPU server already has the packages and weights the NISAR runs use, but
not the G1A image or the reference collection. Everything below is done from
the workstation; the server is only reached through the GPU service.

**1. Copy the code once.** Copy this folder to the server's exe folder, e.g.
`V:\ICIGDev\GPUPOC\exe\g1a_automatch` (= `/maintenance/ICIGDev/GPUPOC/exe/g1a_automatch`).
Copy it again after updating the code, **into the folder the curl's
`code_path` names** (pack's default is `/maintenance/ICIGDev/GPUPOC/exe/anup/nisar/g1a_automatch/`).
Every log starts with the code it ran, `automatch 2026.10.05, code 1a2b3c4d (<folder>)`,
and `PACK_REPORT.txt` shows the workstation's: the two fingerprints match
only when the server runs the same code. A settings file holds the paths and
only the settings that differ from the defaults, and a setting the server's
code does not know is ignored with a warning (`PREFLIGHT.json` lists it) --
code older than 2026.10.05 refuses it instead: `Unknown job key(s)` (jobs 304–307).

**2. Make a job in the GUI on the workstation**, as for a local run (input,
reference folder, Ground truth CSV, Manual GCP CSV empty, detectors,
matchers, window size, ...), and *Save job…*. This only collects the
settings and paths; nothing needs to run locally.

**3. Pack the scene for the server.** In the GUI: *Pack for server…* (bottom
row). It remembers the folders that stay the same (server weights folder,
code path, the `V:\` = `/maintenance/` share mapping, GPU memory to request),
fills the server folder from the share folder, and shows the copying in the
Log and the curl requests at the end. Window sizes and the keypoint cap come
from the main window only (the dialog shows them). Or from the conda prompt:
```bat
python automatch_job.py pack D:\jobs\set1.json ^
   --to V:\ICIGDev\GPUPOC\input\g1a\set1 ^
   --server-dir /maintenance/ICIGDev/GPUPOC/input/g1a/set1 ^
   --server-weights /maintenance/ICIGDev/GPUPOC/input/dqe/imw_runtime/imw_cache ^
   --set window_sizes=[3072] --set detectors=all --set matchers=all
```
It prints its progress (copying references can take a while: one scene
with 24 Landsat-8 pan tiles was 12 GB) and copies the image (with its sidecars), **only the reference rasters within
`max_expected_error_m` of the image** together with the collection's index
shapefile or sidecars, and the truth CSV, and writes
`automatch_settings.json` with the server paths, the three `mode_*.json`
files, `submit_gpuaas.sh` and `PACK_REPORT.txt` (the four curl requests,
ready to paste). It also checks, through the share, that `code_path` and
`--server-weights` exist on the server. An interrupted pack can be started
again: files already copied are kept. `--dry-run`
shows the list and size first. `--server-weights` is the folder `imw.py` uses
(imcui weights; kornia's default folder on the server is searched as well).
`--set` changes settings for the server only: 3072 px windows suit a 40 GB
A100; `"all"` takes every detector and every matcher the server offers
(kornia + imcui).

**4. Submit, in this order** (from `PACK_REPORT.txt`, or `./submit_gpuaas.sh <mode>`).
The service checks that every item of `input_path` is an existing file, so
the check to run is chosen with a file: the settings, plus `mode_env.json`,
`mode_weights.json` or `mode_preflight.json` (written by pack, each just
`{"mode": "env"}` etc.); the run is the settings file alone.
```bash
curl -X POST "http://gpuaas.private.nrsc.gov.in:8000/submit_job" -H "Content-Type: application/json" -H "X-User-Name: $(whoami)" -d '{
 "code_path": "/maintenance/ICIGDev/GPUPOC/exe/g1a_automatch/automatch_gpuaas.py",
 "conda_env": "mpad",
 "max_gpu_mem_required": 40000,
 "input_path": "/maintenance/ICIGDev/GPUPOC/input/g1a/set1/automatch_settings.json,/maintenance/ICIGDev/GPUPOC/input/g1a/set1/mode_env.json"
 }'
```
then the same with `mode_weights.json`, `mode_preflight.json`, and finally
`"input_path": ".../automatch_settings.json"` alone for the run:

| input_path | what it does (seconds unless run) | read in the job's output folder |
|---|---|---|
| settings + `mode_env.json` | packages and versions, GPU and free memory, internet, weight folders, detector list on the node | `ENVIRONMENT.txt` |
| settings + `mode_weights.json` | every weight file the selection needs, found or missing (nothing downloaded) | `WEIGHTS_REPORT.txt` |
| settings + `mode_preflight.json` | image, references, truth, settings; number of matching passes; warns about windows too large on the ground for the pixel size or larger than the image, and settings the code does not know | `PREFLIGHT.json` |
| settings alone | the run | `PERFORMANCE.csv`, `TRUTH_BY_DETECTOR_MATCHER.csv`, `RUN_MANIFEST.csv`, `rival/`, `automatch.log` |

If `weights` lists missing files (a detector the NISAR runs never used),
either drop that detector, or run `python prefetch_weights.py D:\w --only <detectors>`
on a connected machine and copy `D:\w\torch\hub\checkpoints\*` into
`<weights folder>/torch/hub/checkpoints/`.

**5. Read the result.** `PERFORMANCE.csv`: one row per detector + matcher,
best first -- truth RMSE, seconds per window, chips and inliers kept, peak
GPU memory, and `gpu_window_px`: the largest piece of a window the detector
could match on that GPU (smaller than the window size when it had to work
in tiles, see *GPU windows*). `TRUTH_BY_DETECTOR_MATCHER.csv`: the accuracy
ranking alone.

Notes:
- All paths in the settings are server paths (`/maintenance/...`), never `V:\`.
- To rank a finished run again on the server (newer scoring rule, more truth
  points), make a mode file `{"mode": "compare", "compare_dir": "/maintenance/.../output297"}`
  and submit settings + that file (`max_gpu_mem_required` can be small: no
  GPU is used). The truth comes from the settings' `truth_csv`; the TRUTH files
  and `RIVAL_TRUTH_BEST_*.csv` go into that run's folder and the job's own, and
  `PERFORMANCE.csv` and `RIVAL_BEST_*` are rewritten in the new order. A run
  made by an older version (e.g. under the earlier ranking rule) needs only
  this, not a new run. On the workstation: `python automatch_job.py compare
  <output folder> --truth <truth.csv>` does the same.
- To change a setting, edit `automatch_settings.json` on the server share (or
  re-run pack with other `--set` values: files already copied are kept).
- `"env": {"NISAR_IMW_RESIZE_MAX": "3072"}` in the settings sets environment
  variables for the job (here: imcui models see the full 3072 px window
  instead of 2048).
- Without pack: put the image, references and truth on the server yourself
  and use `examples/gpuaas_g1a_select.json` with its paths filled in.

## A100 server: run it with curl

`automatch_server.py` puts an HTTP front end on the same job runner (standard
library only). Jobs are the same JSON files; paths in them are paths on the
server. Jobs run one at a time on the GPU.

```bash
# on the server
python automatch_server.py --port 8765 --jobs-dir /data/automatch_jobs        # localhost only
#   or reachable from other machines:  --host 0.0.0.0 --token SECRET
# from the workstation: ssh -L 8765:localhost:8765 user@server   (then use localhost:8765)

H="Authorization: Bearer SECRET"                    # only if --token was given
curl -s -H "$H" -X POST localhost:8765/weights   --data-binary @examples/a100_select_job.json
curl -s -H "$H" -X POST localhost:8765/preflight --data-binary @examples/a100_select_job.json
curl -s -H "$H" -X POST localhost:8765/jobs      --data-binary @examples/a100_select_job.json   # -> {"id": ...}
curl -s -H "$H" localhost:8765/jobs/<id>                        # state, progress, results, truth ranking
curl -s -H "$H" "localhost:8765/jobs/<id>/log?tail=50"
curl -s -H "$H" localhost:8765/jobs/<id>/files                  # output files
curl -s -H "$H" localhost:8765/jobs/<id>/files/TRUTH_BY_DETECTOR_MATCHER.csv -o ranking.csv
curl -s -H "$H" -X POST localhost:8765/jobs/<id>/stop
curl -s -H "$H" -X POST localhost:8765/compare -d '{"output_dir": "/data/out", "truth_csv": "/data/manual.csv"}'
```
Without HTTP: `nohup python automatch_job.py run job.json > run.log 2>&1 &` does the same.

`examples/a100_select_job.json` is a detector + matcher selection run for a
40 GB A100: every kornia detector with every matcher, one RANSAC setting,
3072 px windows, the ground truth. Window size: at 2048 px the keypoint budget
already reaches its 32 000 cap, so larger windows mostly grow the image-sized
memory, about with window area -- 16 GB full at 2048 px scales to about 3072 px
on 40 GB (2.25x the area); 4096 px (4x) is likely too much for DeDoDe-G / DISK.
Add imcui detectors (`imw-...`) to `detectors` as listed by `/detectors`.

**Speed next to accuracy.** Each run records the wall time of every matching
pass (`PASS_TIMING.csv`: seconds and seconds per window, detection included,
since each pass detects again) and each detector's total time and peak GPU
memory (`detector_seconds`, `gpu_peak_gb` in `RUN_MANIFEST.csv`); the truth
ranking carries `sec_per_window` beside the RMSE.

## Keypoints per window

With `num_features` "auto", a window gets `keypoint_density` keypoints per
megapixel (9000), up to `max_num_features` (32000, GUI: *Max keypoints per
window*; the Windows tab shows what each window size gets). 2048 px windows
already reach 32000, so larger windows get fewer keypoints per km² unless the
cap is raised: 90000 keeps the density at 3072 px. Matching then builds no
keypoints × keypoints distance table (29 GB at 85000 a side): SMNN, MNN, SNN,
NN, FGINN and AdaLAM find nearest neighbours a slice at a time above
1.5e8 pairs, with the same matches as kornia's functions. LightGlue and the
detectors' time still grow with the count. FGINN follows the method row by
row (kornia's version compares every row with the first row's candidates).
The log prints each detector's time and peak GPU memory when it finishes.

## GPU memory (16 GB workstation, GPU windows)

Defaults: 1024 px windows, one model resident at a time (unloaded after its
run), windows skipped rather than crashing when less than `min_gpu_free_gb`
is free. Dense imcui models (RoMa, DKM, the LoFTR family) are the heaviest;
they now find their own tile size (below).

**GPU windows.** A window that does not fit in the GPU is matched in tiles.
Each detector starts with whole windows; when it runs out of GPU memory it
cuts every window into 2 x 2 tiles, then 3 x 3, 4 x 4, 6 x 6, 8 x 8 (never
below 256 px) until its tiles fit, and keeps that size for the rest of its
run. So each detector finds the window size that fits this GPU, whatever
`window_sizes` says. The tiles share the window's keypoint budget by area,
and their matches are pooled into the window: every detector still gives one
result per window, comparable with the others. The log says when a detector
steps down (`out of GPU memory in 2048 px (...): 2048 px windows now matched
in 1024 px tiles on NVIDIA A100-SXM4-40GB (40 GB), 8000 keypoints per tile`),
and `RUN_MANIFEST.csv` and `PERFORMANCE.csv` give each detector's
`gpu_window_px`. Setting `gpu_window_px` forces a tile size (0 = find it).
Before this, runs 297 (A4500, 20 GB, 2048 px) and 299 (A100, 40 GB, 3072 px)
lost 217 and 190 windows to out-of-memory, 178 and 169 of them in RDD, XoFTR,
ELoFTR, ASpanFormer, MINIMA-LoFTR and TopicFM, whose match matrix grows with
the fourth power of the window side.

On Windows the NVIDIA driver can hand out more memory than the card has,
spilling into system RAM ("CUDA - Sysmem Fallback Policy"): nothing runs out,
everything slows down. On the 16 GB RTX 5000 DeDoDe reached 25–27 GB and took
14–30 min per variant (0.5–2 min on the A100). A detector whose memory spills
is treated as out of memory: smaller tiles from the next window on, and the
log names the setting. NVIDIA Control Panel > Manage 3D settings > CUDA -
Sysmem Fallback Policy > *Prefer No Sysmem Fallback* (for python.exe) makes
it a plain out-of-memory, handled at the first window.

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

- **imcui XFeat + LighterGlue and kornia LightGlue in one process:**
  LighterGlue replaces kornia's LightGlue defaults (a class attribute) with
  its own 96-dimension settings, and every kornia LightGlue built after it
  failed to load its weights (job 318, second sweep: SIFT, DISK, KeyNet, DoG,
  DeDoDe). automatch keeps kornia's defaults and puts them back before each
  kornia LightGlue it builds.

- **Coarse offsets of dense matchers:** RoMa, DKM, GIM, XoFTR and similar
  sometimes return a wrong coarse offset for a few pairs (run 297: 3 of 22
  pairs for MINIMA-RoMa; job 325: 5 of 15 for XoFTR). Offsets outside the
  clearly heaviest group are replaced (see *Errors of kilometres*); when no
  group clearly outweighs the others they are kept, and the log warns when
  the pairs disagree by more than half of `max_expected_error_m`.
- **Window size is a ground size.** 2048 px is 92 km at 45 m (G1A MX-VNIR)
  but 369 km at 180 m (G1A HS) and 645 km at 315 m. Run 300 (315 m, 2048 px)
  got 4–6 chips and results tens of km off; on jobs 312–317 (180 m and 315 m)
  most reference pairs gave no window or one (windows under 500 px a side were
  skipped; now windows go down to 128 px, for strips like a 39 km wide HS
  scene, 216 px at 180 m), and on job 313 (3072 px) no detector reached a
  consensus. Choose windows of about 90 km: 2048 px at 45 m, 512 px at 180 m,
  about 300 px at 315 m. Preflight warns when a window is more than twice
  that, or larger than the image.

- **Mixed conda / pip environments (Linux):** if a pip wheel (pandas, torch)
  loads the system C++ runtime before conda-forge GDAL, rasterio fails with
  "GLIBCXX_3.4.30 not found". `automatch_native.py` imports rasterio first in
  every entry module to avoid it.

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
