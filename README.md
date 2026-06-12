# NISAR–S1 geolocation-error matching: image-matching-webui bridge

Measures NISAR GSLC location error against Sentinel-1 reference GeoTIFFs by
feature matching, using [image-matching-webui / imcui](https://github.com/Vincentqyw/image-matching-webui)
models (SuperPoint+LightGlue, ALIKED+LightGlue, DISK+LightGlue, xfeat,
SuperGlue, LoFTR, eLoFTR, ASpanFormer, RoMa, DKM, …) as drop-in
detector/matcher replacements for the original kornia pipeline.

## Files

| File | Role |
|------|------|
| `dqe_integrated.py` | The original kornia production pipeline (`DPQED_agdqe_all.py`), made import-safe. Contains the H5/MET reader, S1 reference fetcher, pair preprocessor, windowing, RANSAC grid, statistics and chip-consensus stages. Still runnable standalone exactly as before. |
| `dqe_imw.py` | The imcui bridge. Wraps each imcui model as a `DiskBasedMatcher`, so preprocessing → windowing → RANSAC grid → statistics → chip consensus are reused unchanged and results are directly comparable with the kornia runs. |
| `imw_configs.py` | Model catalog. Resolves confs from imcui's own registry so LightGlue weight/feature pairings, eLoFTR checkpoints, RGB-vs-gray preprocessing etc. are always correct for the installed imcui version. Edit `CATALOG` to add/remove models. |
| `prefetch_imw_weights.py` | Downloads every checkpoint in the catalog on an internet-connected box, for hand-carry to the air-gapped machine. |

## Running

```bash
# same CLI shape as the kornia pipeline
python dqe_imw.py x <output_dir> <scene_dir>,<vh_ref_dir>,<vv_ref_dir>
```

- `scene_dir` must contain `<name>.h5` + `<name>.met` (NISAR GSLC).
- `vh_ref_dir` / `vv_ref_dir` hold S1 reference `*.tif` + `*_meta.txt` files.
- Outputs land in `<output_dir>/imw_win<N>/<POL>_toS1<TAG>/{raw_matches,filtered,statistics,final}_same-res_imw-<model>/`
  plus the usual `SUMMARY_ALL.csv`, `CONSENSUS_SCORES.csv`, `BEST_FINAL_SUMMARY.csv`.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NISAR_TEMP_DIR` | `<output_dir>/temp_cache` | Where coregistered `pair###_*.tif` chips are cached. Point it at an existing kornia-run cache to skip re-coregistration. |
| `NISAR_FORCE_REFETCH` | `0` | `1` = ignore cached pairs, re-fetch S1 references. |
| `NISAR_IMW_WIN_SIZES` | `1024` | Comma list of window sizes to sweep, e.g. `1024,2048`. |
| `NISAR_IMW_ONLY` | (all) | Comma list of catalog tags to run, e.g. `sp-lg,roma`. |
| `NISAR_IMW_RESIZE_MAX` | `2048` | Long-side cap for imcui preprocessing. Keep ≥ window size so chips aren't resized (preserves 10 m/px geometry). |
| `NISAR_IMW_DENSE_MAX_KP` | `4096` | Max matches kept by dense matchers (RoMa/DKM/LoFTR…). |

## Offline / air-gapped use

On an internet-connected machine with the same imcui version:

```bash
export HF_HOME=/tmp/imw_cache/huggingface
export TORCH_HOME=/tmp/imw_cache/torch
export XDG_CACHE_HOME=/tmp/imw_cache/xdg
python prefetch_imw_weights.py        # PREFETCH_INSECURE_SSL=1 if behind TLS-inspecting proxy
tar czf imw_cache.tgz -C /tmp/imw_cache .
```

On the air-gapped machine, extract and set the same three variables plus
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before running `dqe_imw.py`.

## Adding a model

Add a row to `CATALOG` in `imw_configs.py`:

```python
# sparse:  (tag-without-underscores, extractor conf name, matcher conf name)
('r2d2-nn',  'r2d2', 'NN-ratio'),
# dense:   (tag, None, dense matcher conf name)
('minima',   None,   'minima_loftr'),
```

Conf names come from your installed imcui registry; unknown names are skipped
at startup with a message listing the available ones.
