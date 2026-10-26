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
| `gt_path` | its tie-point file |
| `reference` | the reference raster, or a folder of them (e.g. L8_ref): the ones covering the points are found by footprint |
| `reference_band` | |
| `cloud_mask` (optional) | a mask on the input grid (0 clear, 1–3 cloud / shadow, e.g. automatch's `CLOUD_MASK.tif`): crops with more than `max_cloud` are skipped |
| `split` (optional) | `train` / `val` / `test`; else assigned per scene (20 % test, 10 % val) |
| `work_res_m` (optional) | this scene's working pixel size |

## 2. Pairs

    python pairs.py --manifest scenes.csv --out pairs/ --crop 512 --per-scene 40 --work-res 45

- **Crops:** each accepted scene gives `per-scene` crops inside the area its
  points cover (grown by 15 %).
- **Reference crop:** resampled onto a north-up grid of the working pixel
  size, shifted at random by up to a quarter crop (the misalignment a
  matcher meets after the coarse step).
- **Homography:** `H_0to1` maps input-crop pixels to reference-crop pixels.
  It is fitted on an 11 × 11 grid; crops where the scene model is not
  near-projective (more than 0.5 px off) are skipped.
- **Working resolution:** `--work-res` matters for metre-scale sensors
  (Cartosat-3, C2E, LISS-IV) against a 30 m reference. At their native pixel
  a sharp image would face a blurred reference. Crops are read coarser from
  the pyramids. For G1A, train at its scales: 45 m (MV), 180 m (HS),
  315 m (HV).
- **Outputs:**
  - `pairs/<scene>/<k>_0.png` and `_1.png`, stretched as the matching tool
    stretches windows, with masks `_m0.png` and `_m1.png`;
  - `pairs.csv`;
  - `scenes_fit.csv`: per scene, the fit type, leave-one-out RMSE, points
    used, mean error, and why it was not used;
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
{"mode": "pairs", "manifest": "/maintenance/.../scenes.csv", "crop": 512, "per_scene": 40, "work_res_m": 45}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked", "smoke": true}
{"mode": "train", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "experiment": "aliked_lg_microsat", "epochs": 20, "batch_size": 16, "sensors": ["MicroSat"]}
{"mode": "eval", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked", "label": "official"}
{"mode": "eval", "index": "/maintenance/.../pairs/pairs.csv", "config": "aliked",
 "checkpoint": "/maintenance/.../checkpoint_best.tar", "label": "finetuned"}
```

- **Run `"smoke": true` first.** It is one epoch on 64 pairs, minutes on the
  A100, and proves Glue Factory, the weights and the data before a long run.
- **Configs:** `configs/aliked_lightglue_gcp.yaml` and
  `configs/disk_lightglue_gcp.yaml`. Both keep the keypoint extractor frozen
  and fine-tune LightGlue from its official weights at a learning rate of
  1e-5.
- **Outputs:** training writes `checkpoint_best.tar` and
  `<experiment>_lightglue.pth` into the output folder. The `.pth` holds the
  matcher's weights in the official LightGlue layout, which kornia's
  LightGlue loads (`export.py`).

## 4. Evaluate

`evaluate.py`, or the `eval` mode, scores a matcher on the test split. It
reports:
- matches per pair;
- precision: matches within 3 px of where the true H puts them;
- homography success: corners within 3 px.

Results are given per sensor and overall, in `EVAL_<label>.csv`. Compare the
official and the fine-tuned weights on the same test scenes.

## Priorities

- **Optical first, MicroSat in particular.** Its thermal bands against an
  optical reference are the same cross-modal problem as G1A's MX-LWIR (ML).
- **For thermal, the frozen-extractor LightGlue fine-tune is a start.**
  ALIKED's keypoints were learnt on visible imagery. If thermal precision
  stays low, the next steps are fine-tuning the extractor too
  (`model.extractor.trainable: True`, a smaller learning rate), or XoFTR,
  which was built for visible–thermal matching.

## Checks

`python tests.py` builds synthetic scenes in which the truth is known:
- a reference holds the true ground;
- an input image shows it through a known affine or quadratic mapping while
  its own georeference is wrong;
- tie points in your text format, with unset rows and an outlier.

The checks:
- the fit recovers the mapping to within 1 m and drops the outlier;
- badly marked scenes are refused;
- every pair's H aligns its crops (correlation above 0.9), also at a coarser
  working resolution;
- the dataset gives Glue Factory's sample layout;
- the metric separates good from shuffled matches;
- export keeps only the matcher;
- the train mode builds Glue Factory's command line (smoke: one epoch on 64
  pairs) and collects the checkpoint and its export.

Not checked here: Glue Factory itself (not installed in this sandbox). The
`smoke` run on the server is that check.
