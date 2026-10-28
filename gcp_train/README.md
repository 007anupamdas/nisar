# gcp_train — fine-tune matchers on manually measured tie points

A separate tool (it does not import, and is not imported by, `g1a_automatch`).
It turns the manual ground-truth points you already have for many satellites
(C2E, C3, RISAT, NISAR, RS2, MicroSat, hyperspectral, …) into supervised
training pairs, and fine-tunes LightGlue on them with
[Glue Factory](https://github.com/cvg/glue-factory) (Apache-2.0, the library
LightGlue was trained with). Everything runs on your machines; no image leaves
them.

## Why it works with 10–15 points per scene

A matcher cannot learn from 15 points. The points' value is that they fix the
scene's geometry. Per scene the points are fitted to a mapping from image
pixel to true ground (`fit.py`):
- **Model:** affine, or quadratic where the error changes across the scene,
  as in G1A job 347 (+49 km in the west to +77 km in the east).
- **Choice:** by leave-one-out error, dropping up to three stray points.
- **Acceptance:** only when the leave-one-out RMSE is at most `max_loo_px`
  (1.5 px). Worse supervision would teach the matcher wrong positions.

With the mapping known everywhere, every crop of the scene becomes a pair
with an exact homography (`pairs.py`). Tens of scenes per sensor then give
thousands of supervised pairs.

## What the model is shown, and what it is told

- **Every resolution, without a setting.** A pair can be no finer than the
  coarser of its two images: a 30 m reference holds no 6 m detail, so a pair
  at 6 m would teach matching against blur. Above that floor, every crop
  draws its own pixel size, up to where a crop still fits inside the
  points' area (at most 16 × the floor). Each sensor is therefore seen at
  many resolutions, read from the `.img` pyramids.
- **The kind of image.** Each manifest row says what its band is: `visible`,
  `nir`, `swir`, `mwir`, `lwir` (thermal) or `sar`, by name or by
  wavelength (`modality.py`). The vocabulary is physics, not satellite: a
  G1A MX-LWIR band gets the conditioning learnt for MicroSat's thermal band,
  though G1A itself is not in the training data.
- **The matcher is told both.** The matcher, `gf_cond_lightglue`, receives
  each view's modality and the pair's pixel size (`condition.py`). It uses
  them as a learnt scale and shift of the keypoint descriptors (FiLM)
  before LightGlue matches. The scale and shift start at zero, so training
  starts from exactly the official matcher.
- **What to look at, not only how to match.** `"train_extractor": true` also
  fine-tunes the keypoint network (ALIKED / DISK) through the matching loss,
  so its descriptors adapt to thermal or SAR texture. If that is not
  enough, the next step is a detector-free matcher, which decides where to
  match from both images together: XoFTR for thermal; for SAR, a dense
  matcher (LoFTR or RoMa) trained on these pairs.

## The ground truth

RIVAL's ground-truth layout: the CSV's columns, in its order, separated by
spaces or tabs, without a header (a header and commas also work):

    In_X In_Y Ref_X Ref_Y DX_Err DY_Err Row In_Lon In_Lat Ref_Lon Ref_Lat

Unset points (a lon/lat of 0, `-`, or a row that ends early because a space
file cannot hold an empty cell) are recovered from their X/Y when the file's
other points give that mapping, else left out with a note. Rows more than
500 km apart are dropped as broken (`gt.py`).

## 1. Scenes: `scenes.csv`

One row per scene:

| column | |
|---|---|
| `scene_id` | unique, e.g. `MS2_20260412` |
| `sensor` | e.g. `MicroSat`, `C2E`, `RS2_LISS4`, `G1A_MV` — used for splits and per-sensor training / evaluation |
| `date` | |
| `input_path` | the image: `.img` (with its pyramids), GeoTIFF, VRT; NISAR: an amplitude GeoTIFF made first |
| `input_band` | the band matched, e.g. a MicroSat thermal band |
| `modality` or `wavelength_nm` | what that band is: `visible`, `nir`, `swir`, `mwir`, `lwir`, `sar` (aliases: optical, pan, thermal, tir, …), or its centre wavelength (nm or µm). Neither: SAR sensors are known by name (RISAT, NISAR, …), anything else counts as visible. **Set it for thermal bands.** |
| `gt_path` | its tie-point file |
| `reference` | the reference raster, or a folder of them (e.g. L8_ref): the ones covering the points are found by footprint |
| `reference_band` | |
| `reference_modality` or `reference_wavelength_nm` (optional) | default visible |
| `cloud_mask` (optional) | a mask on the input grid (0 clear, 1–3 cloud / shadow, e.g. automatch's `CLOUD_MASK.tif`): crops with more than `max_cloud` are skipped |
| `split` (optional) | `train` / `val` / `test`; else assigned per scene (20 % test, 10 % val) |
| `res_m` (optional) | limits this scene's pixel sizes: a number fixes it, `45-315` sets a range |

Several rows may share a `scene_id`, for example other bands of one cube.
They share its split.

## 2. Pairs

    python pairs.py --manifest scenes.csv --out pairs/ --crop 512 --per-scene 40

- **Crops:** each accepted scene gives `per-scene` crops inside the area its
  points cover (grown by 15 %), each at its own pixel size (above).
- **Reference crop:** resampled onto a north-up grid of that pixel size,
  averaged where it is coarser than the reference, and shifted at random by
  up to a quarter crop (the misalignment a matcher meets after the coarse
  step).
- **Homography:** `H_0to1` maps input-crop pixels to reference-crop pixels.
  It is fitted on an 11 × 11 grid; crops where the scene model is not
  near-projective (more than 0.5 px off) are skipped.
- **Pixel size:** nothing to set. `--res 45-315` (or `res_m` per scene) only
  narrows the range, for example to weight training towards G1A's scales;
  `--max-scale` changes the 16 × limit.
- **Outputs:**
  - `pairs/<scene>/b<band>_<k>_0.png` and `_1.png`, stretched as the
    matching tool stretches windows, with masks `_m0.png` and `_m1.png`;
  - `pairs.csv`, with each pair's `px_m`, `modality0` (input) and
    `modality1` (reference);
  - `scenes_fit.csv`: per scene, the fit type, leave-one-out RMSE, points
    used, mean error, both images' pixel sizes, the pixel-size range, and
    why a scene was not used;
  - `splits.csv`.

**Test scenes never train.** The split is by scene, never by crop, because
crops of one scene look alike. To measure generalisation, also hold out
whole sensors or months by setting `split` in the manifest.

## 3. Fine-tune (Glue Factory)

On the server, once:

    git clone https://github.com/cvg/glue-factory && pip install -e glue-factory

Offline, Glue Factory fetches the ALIKED / DISK and official LightGlue weights
through torch.hub. Copy these into the torch hub cache
(`$TORCH_HOME/hub/checkpoints`):
- `aliked-n16.pth`
- `aliked_lightglue_v0-1_arxiv.pth`
- `disk_lightglue_v0-1_arxiv.pth`

Alternatively, give a local file as `model.matcher.weights`.

Through GPUaaS (`python gcp_train/gpuaas.py "." <output> <settings.json>`):

```json
{"mode": "pairs", "manifest": "/maintenance/.../scenes.csv", "crop": 512, "per_scene": 40}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked", "smoke": true}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "experiment": "aliked_lg_all", "epochs": 20, "batch_size": 16}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "experiment": "aliked_lg_plain", "epochs": 20, "condition": false}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "experiment": "aliked_lg_thermal_ext", "epochs": 20, "train_extractor": true}
{"mode": "eval", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked", "label": "official"}
{"mode": "eval", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "checkpoint": "/maintenance/.../checkpoint_best.tar", "label": "aliked_lg_all"}
```

- **Run `"smoke": true` first.** It is one epoch on 64 pairs, minutes on the
  A100, and proves Glue Factory, the weights and the data before a long run.
- **Configs:** `configs/aliked_lightglue_gcp.yaml` and
  `configs/disk_lightglue_gcp.yaml`. Both fine-tune the conditioned LightGlue
  from its official weights, at a learning rate of 1e-5 (1e-3 for the
  conditioning, which starts at zero). The keypoint extractor stays frozen
  unless `"train_extractor": true`.
- **Comparisons to run:** one model on every sensor (all modalities and
  resolutions), the same without the conditioning (`"condition": false`),
  and with the extractor trained. The evaluation then shows what each
  brings, per modality and resolution.
- **Filters:** `"sensors"` and `"modalities"` (input modalities, e.g.
  `["lwir"]`) restrict training to part of the pairs.
- **Outputs:** training writes these into the output folder (`export.py`):
  - `checkpoint_best.tar`;
  - `<experiment>_lightglue.pth`: the matcher's weights in the official
    LightGlue layout, which kornia's LightGlue loads;
  - `<experiment>_lightglue_condition.pth`: the conditioning, applied to the
    descriptors before LightGlue (the formula is in the `.json` beside it);
  - `<experiment>_extractor.pth`, when the extractor was trained.

## 4. Evaluate

`evaluate.py`, or the `eval` mode, scores a matcher on the test split. It
reports:
- matches per pair;
- precision: matches within 3 px of where the true H puts them;
- homography success: corners within 3 px.

Results are given overall, per sensor, per modality pair (input → reference,
e.g. `lwir -> visible`) and per pixel-size band (`<10 m` … `>=300 m`), in
`EVAL_<label>.csv`. Compare the official and the fine-tuned weights on the
same test scenes.

## Priorities

- **Optical first, MicroSat in particular.** Its thermal bands against an
  optical reference are the same cross-modal problem as G1A's MX-LWIR (ML).
- **For thermal, compare the frozen and the trained extractor.** ALIKED's
  keypoints were learnt on visible imagery, so `"train_extractor": true`
  matters most for `lwir` and `sar` inputs. If thermal precision stays low
  even then, the next step is XoFTR, which was built for visible–thermal
  matching.

## Checks

`python tests.py` builds synthetic scenes in which the truth is known:
- a reference holds the true ground;
- an input image shows it through a known affine or quadratic mapping while
  its own georeference is wrong;
- tie points in your text format, with unset rows and an outlier.

The checks:
- the fit recovers the mapping to within 1 m and drops the outlier;
- badly marked scenes are refused;
- every pair's H aligns its crops (correlation above 0.9), also at a fixed
  coarser pixel size;
- the automatic pixel size starts at the coarser image (raised to a 120 m
  reference's when that is coarser) and varies per crop up to where a crop
  fits;
- a coarser reference grid is averaged (it equals the block mean), not
  sampled;
- modalities resolve from names, wavelengths and SAR sensor names, and
  unknown names are refused;
- rows of one scene get their own file names and share its split;
- the dataset gives Glue Factory's sample layout, with both modalities and
  the pixel size, exchanged when the views are swapped;
- the conditioning starts as the identity and learns per modality; the
  conditioned matcher transforms only the views it should;
- the metric separates good from shuffled matches, and summaries come per
  sensor, modality pair and pixel-size band;
- export separates LightGlue, the conditioning and a trained extractor;
- the train mode builds Glue Factory's command line (smoke, filters,
  extractor training, plain LightGlue) and collects the checkpoint and its
  export.

Not checked here: Glue Factory itself (not installed in this sandbox; the
conditioned matcher was checked on a stand-in for its LightGlue). The `smoke`
run on the server is that check.
