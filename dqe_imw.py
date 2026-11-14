#!/usr/bin/env python3
"""
NISAR-S1 Pipeline -- image-matching-webui (imcui) bridge

Plugs Vincentqyw's image-matching-webui (`imcui`) into the existing
DiskBasedMatcher framework so you can A/B-test additional detector/matcher
combinations (SuperPoint+LightGlue, ALIKED+LightGlue, RoMa, DKM, ASpanFormer,
xfeat, LoFTR variants, ...) without touching the preprocessing / RANSAC /
chip-consensus stages.

Each imcui configuration (one row in `IMW_CONFIGS` below) is wrapped as a
single matcher class instance. imcui handles detection AND matching in one
call, so we override `_process_single_window` (same pattern as LoFTRMatcher).

The output CSVs use this file_id template, picked to slot into the existing
filename parser with minimal disruption:

    imw-<configname>_<pol>_toS1<tag>_pair<id>_scan<r>_pix<c>_lgm_aff_<m>_<t>_<c>_.csv
                                                         ^^^^
                              "lgm" is a stand-in matcher token so the existing
                              MatchStatistics._parse_filename branch for SIFT-
                              like detectors handles it. Underscores INSIDE the
                              imcui config name are replaced with `-` so the
                              token count stays predictable.

Usage:
    python dqe_imw.py <unused> <output_dir> <scene_dir,vh_ref_dir,vv_ref_dir>

Same CLI shape as dqe_integrated.py (the import-safe copy of
DPQED_agdqe_all.py that lives next to this file).
"""

import os
import sys
import gc
import time
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch as th

# Import the existing pipeline (dqe_integrated.py = import-safe copy of
# DPQED_agdqe_all.py, shipped in this repo).
from dqe_integrated import (
    PipelineConfig,
    DiskBasedMatcher,
    ProductionPipelinePolwise,
    MatchStatistics,
    ChipConsensusSelector,
    NISARH5Reader,
    ReferenceFetcher,
    DiskBasedPreprocessor,
    RANSACFilter,
    safe_cuda_empty_cache,
    setup_logging,
    device,
)

# imcui imports are deferred until first use so the rest of the pipeline still
# runs even if imcui isn't on PYTHONPATH (e.g. in a env without it installed).
_IMW_API = None


def _load_imw():
    global _IMW_API
    if _IMW_API is None:
        from imcui.api import ImageMatchingAPI  # noqa: WPS433
        _IMW_API = ImageMatchingAPI
    return _IMW_API


# =============================================================================
# IMW CONFIG CATALOG
# =============================================================================
# The catalog lives in imw_configs.py and is resolved against the conf
# registry of the INSTALLED imcui (imcui.hloc.configs), so model names,
# weights and preprocessing always line up with your install. Edit the
# catalog THERE; select a subset at runtime with NISAR_IMW_ONLY=tag1,tag2.
from imw_configs import IMW_CONFIGS  # noqa: E402


# =============================================================================
# IMW MATCHER WRAPPER
# =============================================================================
class IMWMatcher(DiskBasedMatcher):
    """
    Wraps a single imcui ImageMatchingAPI instance and presents it as a
    DiskBasedMatcher so the rest of the NISAR pipeline (preprocessor,
    RANSAC, statistics, chip consensus) reuses unchanged.
    """

    def __init__(self, config: PipelineConfig, imw_tag: str,
                 imw_conf: Dict, dense: bool):
        super().__init__(config)
        if '_' in imw_tag:
            raise ValueError(f"imw_tag must not contain '_' (got '{imw_tag}')")
        self.imw_tag = imw_tag
        self.imw_conf = imw_conf
        self.dense = dense
        self.api = None
        self._load_failed = False  # set True after a failed load to stop retrying

    # ---- model loading (once per matcher) ------------------------------------
    def _ensure_api(self) -> bool:
        """Load the imcui model once. Returns True if usable, False if the
        load has failed (so the caller can short-circuit every window instead
        of re-attempting a doomed HuggingFace download thousands of times)."""
        if self.api is not None:
            return True
        if self._load_failed:
            return False
        try:
            ImageMatchingAPI = _load_imw()
            print(f'[{self.get_detector_name()}] Loading model...')
            self.api = ImageMatchingAPI(
                conf=self.imw_conf,
                device=str(self.device) if not isinstance(self.device, str) else self.device,
                detect_threshold=0.015,
                max_keypoints=self.config.num_features,
                match_threshold=0.2,
            )
            return True
        except Exception as e:
            self._load_failed = True
            print(f'[{self.get_detector_name()}] MODEL LOAD FAILED: '
                  f'{type(e).__name__}: {e}')
            print(f'[{self.get_detector_name()}] Aborting this matcher '
                  f'(no per-window retry). If this is a network/HuggingFace '
                  f'error, pre-download checkpoints and set HF_HUB_OFFLINE=1.')
            return False

    # ---- DiskBasedMatcher hooks ----------------------------------------------
    def detect_and_describe(self, img1, img2):
        # Not used -- imcui does detect+match in one call. See override below.
        raise NotImplementedError

    def get_detector_name(self) -> str:
        return f'imw-{self.imw_tag}'

    def get_filename_prefix(self) -> str:
        return f'imw-{self.imw_tag}'

    def get_available_matchers(self) -> List[str]:
        # We expose a single "lgm" matcher token in the filename so the existing
        # MatchStatistics parser (which expects a recognised matcher tag) keeps
        # working. The real matcher identity is encoded in imw_tag.
        return ['lgm']

    def _detector_needs_inpaint(self) -> bool:
        # Most learned models behave like DISK/DeDoDe at hard nodata edges.
        return True

    @staticmethod
    def _has_gpu_headroom(min_free_gb: float = 5.0) -> bool:
        if not th.cuda.is_available():
            return True
        try:
            free, _total = th.cuda.mem_get_info()
            return free / 1024**3 >= min_free_gb
        except Exception:
            return True

    # ---- LoFTR-style override ------------------------------------------------
    def _process_single_window(self, nisar_data, s1_data,
                               nisar_x, nisar_y, s1_x, s1_y,
                               metadata, matcher_name, matcher_param,
                               nisar_nodata=None, s1_nodata=None):
        # Short-circuit immediately if the model could not be loaded, so a
        # network/checkpoint failure does not retry on every single window.
        if self._load_failed:
            return None
        try:
            img1, mask1 = self._norm_img(nisar_data, nodata=nisar_nodata, return_mask=True)
            img2, mask2 = self._norm_img(s1_data,    nodata=s1_nodata,    return_mask=True)

            if img1.max() == 0.0 or img2.max() == 0.0:
                return None
            if np.isnan(img1).any() or np.isnan(img2).any():
                return None

            valid_frac1 = mask1.mean()
            valid_frac2 = mask2.mean()
            if (valid_frac1 < self.config.min_valid_fraction
                    or valid_frac2 < self.config.min_valid_fraction):
                return None

            if self._detector_needs_inpaint():
                img1 = self._inpaint_nodata(img1, mask1)
                img2 = self._inpaint_nodata(img2, mask2)

            # imcui expects HxWx3 RGB uint8
            u1 = (img1 * 255.0).clip(0, 255).astype(np.uint8)
            u2 = (img2 * 255.0).clip(0, 255).astype(np.uint8)
            rgb1 = cv2.cvtColor(u1, cv2.COLOR_GRAY2RGB)
            rgb2 = cv2.cvtColor(u2, cv2.COLOR_GRAY2RGB)

            if not self._ensure_api():
                return None

            if not self._has_gpu_headroom(min_free_gb=5.0):
                safe_cuda_empty_cache()
                if not self._has_gpu_headroom(min_free_gb=3.0):
                    if self.config.debug_mode:
                        print(f'[{self.get_detector_name()}] Skipping ({nisar_x},{nisar_y}):'
                              f' low GPU headroom')
                    return None

            with th.inference_mode():
                pred = self.api(rgb1, rgb2)

            # Prefer post-RANSAC inliers when available, fall back to raw matches.
            mkp1 = pred.get('mmkeypoints0_orig')
            mkp2 = pred.get('mmkeypoints1_orig')
            if mkp1 is None or mkp2 is None or len(mkp1) == 0:
                mkp1 = pred.get('mkeypoints0_orig')
                mkp2 = pred.get('mkeypoints1_orig')
            if mkp1 is None or len(mkp1) == 0:
                return None

            mkp1 = np.asarray(mkp1, dtype=np.float32)
            mkp2 = np.asarray(mkp2, dtype=np.float32)

            h1, w1 = img1.shape
            h2, w2 = img2.shape

            in_bounds = (
                (mkp1[:, 0] >= 0) & (mkp1[:, 0] < w1)
                & (mkp1[:, 1] >= 0) & (mkp1[:, 1] < h1)
                & (mkp2[:, 0] >= 0) & (mkp2[:, 0] < w2)
                & (mkp2[:, 1] >= 0) & (mkp2[:, 1] < h2)
            )
            mkp1 = mkp1[in_bounds]
            mkp2 = mkp2[in_bounds]
            if len(mkp1) == 0:
                return None

            # Reject correspondences touching nodata in either image
            ix1 = mkp1[:, 0].astype(int).clip(0, w1 - 1)
            iy1 = mkp1[:, 1].astype(int).clip(0, h1 - 1)
            ix2 = mkp2[:, 0].astype(int).clip(0, w2 - 1)
            iy2 = mkp2[:, 1].astype(int).clip(0, h2 - 1)
            ok = mask1[iy1, ix1] & mask2[iy2, ix2]
            mkp1 = mkp1[ok]
            mkp2 = mkp2[ok]
            if len(mkp1) < 4:
                return None

            return self._build_match_record(
                mkp1, mkp2,
                nisar_x, nisar_y, s1_x, s1_y,
                metadata,
            )

        except Exception as e:
            print(f'[{self.get_detector_name()}] Exception ({nisar_x},{nisar_y}):'
                  f' {type(e).__name__}: {e}')
            if self.config.debug_mode:
                traceback.print_exc()
            return None

    # ---- Record builder (mirrors _extract_matches in DiskBasedMatcher) -------
    def _build_match_record(self, mkpts1, mkpts2,
                            nisar_wx, nisar_wy, s1_wx, s1_wy,
                            metadata) -> Dict:
        x1 = mkpts1[:, 0].tolist()
        y1 = mkpts1[:, 1].tolist()
        x2 = mkpts2[:, 0].tolist()
        y2 = mkpts2[:, 1].tolist()

        X1, Y1 = zip(*[
            self._xy_to_map(x1[i], y1[i],
                            metadata['x01'], metadata['y01'],
                            metadata['xres1'], metadata['yres1'],
                            nisar_wy, nisar_wx)
            for i in range(len(x1))
        ])
        X2, Y2 = zip(*[
            self._xy_to_map(x2[i], y2[i],
                            metadata['x02'], metadata['y02'],
                            metadata['xres2'], metadata['yres2'],
                            s1_wy, s1_wx)
            for i in range(len(x2))
        ])

        dist   = [float(np.hypot(X1[i] - X2[i], Y1[i] - Y2[i])) for i in range(len(x1))]
        along  = [Y1[i] - Y2[i] for i in range(len(x1))]
        across = [X1[i] - X2[i] for i in range(len(x1))]

        global_row = metadata['nisar_crop_col_offset'] + nisar_wx
        global_col = metadata['nisar_crop_row_offset'] + nisar_wy

        # Keep matcher token = 'lgm' so the existing parser (DISK branch w/o
        # smnn) handles it. The real model identity lives in the prefix.
        file_id = self._build_file_id(metadata, global_row, global_col, 'lgm', '')

        return {
            'file_id': file_id,
            'pair_id': metadata['pair_id'],
            'global_scan': global_row,
            'global_pix': global_col,
            'local_init_x': nisar_wx,
            'local_init_y': nisar_wy,
            'matcher': 'lgm',
            'match_param': '',
            'x1': [v + nisar_wy for v in x1],
            'y1': [v + nisar_wx for v in y1],
            'x2': [v + s1_wy   for v in x2],
            'y2': [v + s1_wx   for v in y2],
            'X1': list(X1), 'Y1': list(Y1),
            'X2': list(X2), 'Y2': list(Y2),
            'distance': dist, 'along': along, 'across': across,
        }

    def unload_model(self):
        self.api = None
        safe_cuda_empty_cache()
        gc.collect()
        print(f'[{self.get_detector_name()}] Model unloaded.')

    def save_matches_to_csv(self, all_matches, output_dir):
        # The parent pipeline calls this exactly once per matcher, right after
        # the last pair has been matched. Unloading here keeps only ONE imcui
        # model resident at a time across the catalog sweep.
        super().save_matches_to_csv(all_matches, output_dir)
        self.unload_model()


# =============================================================================
# FILENAME PARSER EXTENSION
# =============================================================================
# MatchStatistics._parse_filename and ChipConsensusSelector both inspect the
# CSV filename. We register a new branch for 'imw-...' prefixes by monkey-
# patching _parse_filename. This keeps the existing detectors working
# untouched.

_original_parse_filename = MatchStatistics._parse_filename


def _parse_filename_with_imw(filename: str):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if parts and parts[0].startswith('imw-'):
        try:
            detector_full = parts[0]  # e.g. 'imw-sp-lg'
            imw_tag = detector_full[len('imw-'):]
            nisar_pol  = parts[1]
            s1_ref_tag = parts[2].replace('toS1', '')
            pair_id = int(parts[3].replace('pair', ''))
            scan    = int(parts[4].replace('scan', ''))
            pix     = int(parts[5].replace('pix', ''))
            match_method = parts[6]  # 'lgm'
            # parts[7] == 'aff'
            if parts[7] != 'aff':
                return None
            ransac_method     = parts[8]
            ransac_threshold  = float(parts[9])
            ransac_confidence = float(parts[10])
            return {
                'disk_mode': None,
                'det_wt': None,
                'desc_wt': imw_tag,        # stash imw tag here for grouping
                'detector': detector_full,  # full 'imw-xxx' so configs separate
                'nisar_pol': nisar_pol,
                's1_ref_tag': s1_ref_tag,
                'pair_id': pair_id,
                'scan': scan,
                'pix': pix,
                'match_method': match_method,
                'match_parameter': None,
                'ransac_method': ransac_method,
                'ransac_threshold': ransac_threshold,
                'ransac_confidence': ransac_confidence,
            }
        except (ValueError, IndexError):
            return None
    return _original_parse_filename(filename)


MatchStatistics._parse_filename = staticmethod(_parse_filename_with_imw)
# ChipConsensusSelector reads through MatchStatistics, so it inherits the patch.


# =============================================================================
# PIPELINE SUBCLASS
# =============================================================================
class ProductionPipelineIMW(ProductionPipelinePolwise):
    """
    Same as the parent pipeline but `_build_matchers` materialises
    IMWMatcher instances from the IMW_CONFIGS catalog.
    """

    def __init__(self, config: PipelineConfig, mode: str = 'same-res',
                 imw_configs: Optional[List[Tuple[str, Dict, bool]]] = None):
        super().__init__(config, mode=mode)
        self.imw_configs = imw_configs if imw_configs is not None else IMW_CONFIGS

    def _build_matchers(self, detector_types):
        # detector_types is ignored -- we build one matcher per IMW_CONFIGS row.
        # If you want to mix imcui models with the existing SIFT/DISK/etc., run
        # the original pipeline first and this one second; outputs are written
        # to separate per-detector directories.
        matchers = []
        for tag, conf, dense in self.imw_configs:
            matchers.append(IMWMatcher(self.config, tag, conf, dense))
        return matchers


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    input_      = sys.argv[3].split(',')
    scene_dir   = input_[0]
    base_output = sys.argv[2]

    os.makedirs(base_output, exist_ok=True)
    setup_logging(base_output)

    if not IMW_CONFIGS:
        print('[IMW] No imcui configs resolved (is image-matching-webui '
              'installed in this environment?). Nothing to do.')
        sys.exit(1)

    if len(input_) > 3:
        print(f'[IMW] NOTE: ignoring extra CLI tokens: {input_[3:]} '
              f'(this pipeline takes scene_dir,vh_ref_dir,vv_ref_dir)')

    # Where to store / read intermediate pair{NNN}_*.tif chips.
    #   NISAR_TEMP_DIR=/abs/path  → reuse an existing kornia-pipeline cache.
    #   unset                     → write fresh chips inside <base_output>/temp_cache.
    # Either way, references are auto-fetched on the first run; subsequent
    # runs against the same temp_dir reuse the cached chips.
    shared_temp = os.environ.get(
        'NISAR_TEMP_DIR',
        os.path.join(base_output, 'temp_cache'),
    )
    os.makedirs(shared_temp, exist_ok=True)

    # If NISAR_FORCE_REFETCH=1 is set, ignore any cached pairs and re-fetch
    # references from scratch. Useful when the cache is from a different scene
    # or a different reference catalog.
    force_refetch = os.environ.get('NISAR_FORCE_REFETCH', '0') == '1'

    print(f'[IMW] scene_dir   = {scene_dir}')
    print(f'[IMW] base_output = {base_output}')
    print(f'[IMW] temp_dir    = {shared_temp}')
    print(f'[IMW] force_refetch = {force_refetch}')

    WINDOW_SIZES = [int(x) for x in os.environ.get(
        'NISAR_IMW_WIN_SIZES', '1024'
    ).split(',')]
    print(f'[IMW] Window sizes: {WINDOW_SIZES}')

    for win in WINDOW_SIZES:
        win_out   = os.path.join(base_output, f'imw_win{win}')
        done_flag = os.path.join(win_out, 'POL_RUN_SUMMARY.csv')
        if os.path.exists(done_flag):
            print(f'[IMW] Skipping win={win} -- POL_RUN_SUMMARY.csv exists.')
            continue

        print('=' * 80)
        print(f'[IMW] >>> window_size = {win}')
        t0 = time.time()

        config = PipelineConfig(
            window_size          = win,
            window_size_small    = max(250, win // 2),
            loftr_max_window     = win,
            target_resolution    = 10,
            use_disk_cache       = True,
            cleanup_after_pair   = False,
            check_existing_pairs = (not force_refetch),
            use_amp              = True,
            debug_mode           = (win > 2000),

            smnn_thresholds      = [0.95],   # unused for imcui matchers
            ransac_methods       = [4],
            ransac_thresholds    = [1, 2, 3],
            ransac_confidences   = [0.91, 0.95, 0.99],

            reference_dir_vv     = input_[2],
            reference_dir_vh     = input_[1],

            output_base_dir      = win_out,
            temp_dir             = shared_temp,

            s1_decimation        = 10,
            min_area             = 200.0,

            consensus_tolerance_m = 5.0,
            consensus_mode_bin_m  = 0.5,
            min_inliers_per_chip  = 3,
            min_surviving_chips   = 1 if win >= 2000 else 2,
        )
        config.num_features = config.compute_num_features(win)
        print(f'win={win} -> num_features={config.num_features}')

        pipeline = ProductionPipelineIMW(config, mode='same-res')
        try:
            # detector_types arg is ignored by ProductionPipelineIMW
            pipeline.run(scene_dir, detector_types=['imw'])
        except Exception as e:
            print(f'[IMW] ERROR at win={win}: {type(e).__name__}: {e}')
            traceback.print_exc()

        del pipeline
        safe_cuda_empty_cache()
        gc.collect()

        elapsed = time.time() - t0
        print(f'[IMW] <<< Completed win={win} in {elapsed / 60:.1f} min')

    print('=' * 80)
    print('[IMW] ALL WINDOW SIZES COMPLETE')
    print('=' * 80)
