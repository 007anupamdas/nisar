#!/usr/bin/env python3
"""
automatch_imcui -- image-matching-webui (imcui) models as automatch detectors.

kornia comes first. An imcui configuration is registered only when kornia has
no implementation of the same algorithm: DISK, ALIKED, SIFT, XFeat (sparse and
XFeat*), DeDoDe and LoFTR always come from kornia, so imcui's disk-lightglue,
aliked-lightglue, sift-lightglue, xfeat-lightglue, xfeat_dense and loftr are
refused. What remains from the default catalog is SuperPoint+LightGlue,
SuperPoint+SuperGlue, eLoFTR, ASpanFormer, RoMa and DKM.

Model weights: set AUTOMATCH_WEIGHTS_CACHE (or PipelineConfig.weights_cache_dir
via the job file) to a folder laid out like prefetch_imw_weights.py writes it
(huggingface/, torch/). HF_HOME / TORCH_HOME are only set when not already
set, and HF_HUB_OFFLINE is set when that folder exists, so an offline machine
never tries the network.
"""

import gc
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch as th

import automatch_engine as E

_API = None


def configure_weights_cache(path: Optional[str] = None) -> Optional[str]:
    path = path or os.environ.get('AUTOMATCH_WEIGHTS_CACHE')
    if not path:
        return None
    if not os.path.isdir(path):
        print(f'[imcui] weights cache {path} not found -- using default caches')
        return None
    os.environ.setdefault('HF_HOME', os.path.join(path, 'huggingface'))
    os.environ.setdefault('TORCH_HOME', os.path.join(path, 'torch'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    print(f'[imcui] weights cache: {path} (offline)')
    return path


def imcui_available() -> Tuple[bool, str]:
    try:
        import imcui  # noqa: F401
        from imcui.api import ImageMatchingAPI  # noqa: F401
        return True, ''
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def _api_class():
    global _API
    if _API is None:
        import logging
        from imcui.api import ImageMatchingAPI
        logging.getLogger('hloc').setLevel(logging.ERROR)
        _API = ImageMatchingAPI
    return _API


def algorithm_family(conf: Dict, dense: bool) -> List[str]:
    """Algorithm names a catalog row uses, for the kornia-priority check."""
    fams = []
    if dense:
        fams.append(str(conf.get('matcher', {}).get('model', {}).get('name', '')))
    else:
        fams.append(str(conf.get('feature', {}).get('model', {}).get('name', '')))
        feats = conf.get('matcher', {}).get('model', {}).get('features')
        if feats:
            fams.append(str(feats))
    return [f.lower() for f in fams if f]


class IMWMatcher(E.DenseWindowMatcher):
    """One imcui ImageMatchingAPI configuration as a DenseWindowMatcher."""

    matcher_token = 'internal'

    def __init__(self, config: E.PipelineConfig, imw_tag: str, imw_conf: Dict, dense: bool):
        super().__init__(config)
        if '_' in imw_tag:
            raise ValueError(f"imw_tag must not contain '_' (got '{imw_tag}')")
        self.imw_tag = imw_tag
        self.imw_conf = imw_conf
        self.dense = dense
        self.api = None
        self._load_failed = False
        self._n_saved = 0

    def get_detector_name(self) -> str:
        return f'imw-{self.imw_tag}'

    def get_filename_prefix(self) -> str:
        return f'imw-{self.imw_tag}'

    def _detector_needs_inpaint(self) -> bool:
        return True

    def _ensure_api(self) -> bool:
        if self.api is not None:
            return True
        if self._load_failed:
            return False
        try:
            print(f'[{self.get_detector_name()}] Loading model...')
            self.api = _api_class()(
                conf=self.imw_conf,
                device=str(self.device),
                detect_threshold=getattr(self.config, 'imw_detect_threshold', 0.015),
                max_keypoints=self.config.num_features,
                match_threshold=getattr(self.config, 'imw_match_threshold', 0.2),
            )
            return True
        except Exception as e:
            self._load_failed = True
            print(f'[{self.get_detector_name()}] MODEL LOAD FAILED: {type(e).__name__}: {e}')
            print(f'[{self.get_detector_name()}] skipping this detector. For an offline '
                  f'machine, prefetch weights and set AUTOMATCH_WEIGHTS_CACHE.')
            return False

    @staticmethod
    def _gpu_free_gb() -> Optional[float]:
        if not th.cuda.is_available():
            return None
        try:
            free, _ = th.cuda.mem_get_info()
            return free / 1024 ** 3
        except Exception:
            return None

    def match_images(self, img1: np.ndarray, img2: np.ndarray):
        if not self._ensure_api():
            raise RuntimeError(f'{self.get_detector_name()} model unavailable')
        need = getattr(self.config, 'min_gpu_free_gb', 1.5)
        free = self._gpu_free_gb()
        if free is not None and free < need:
            E.safe_cuda_empty_cache()
            free = self._gpu_free_gb()
            if free is not None and free < need:
                raise RuntimeError(f'only {free:.1f} GB GPU memory free (< {need} GB)')
        u1 = (img1 * 255.0).clip(0, 255).astype(np.uint8)
        u2 = (img2 * 255.0).clip(0, 255).astype(np.uint8)
        with th.inference_mode():
            pred = self.api(cv2.cvtColor(u1, cv2.COLOR_GRAY2RGB), cv2.cvtColor(u2, cv2.COLOR_GRAY2RGB))
        # imcui's own geometric check first, raw matches if it rejected all
        m1, m2 = pred.get('mmkeypoints0_orig'), pred.get('mmkeypoints1_orig')
        if m1 is None or m2 is None or len(m1) == 0:
            m1, m2 = pred.get('mkeypoints0_orig'), pred.get('mkeypoints1_orig')
        if m1 is None or len(m1) == 0:
            return None, None
        return np.asarray(m1, np.float64), np.asarray(m2, np.float64)

    def on_window_matched(self, img1, img2, mkp1, mkp2, nisar_x, nisar_y, metadata):
        if not getattr(self.config, 'save_match_images', False):
            return
        if self._n_saved >= getattr(self.config, 'max_match_images', 50):
            return
        save_match_image(self.config.output_base_dir, self.imw_tag, img1, img2, mkp1, mkp2,
                         metadata.get('pair_id', 0), nisar_x, nisar_y)
        self._n_saved += 1

    def unload_model(self):
        if self.api is not None:
            self.api = None
            E.safe_cuda_empty_cache()
            gc.collect()
            print(f'[{self.get_detector_name()}] Model unloaded.')


def save_match_image(out_root, tag, img1, img2, mkp1, mkp2, pair_id, wx, wy, max_lines=300):
    """Side-by-side PNG of a window's matches (cv2 only, no matplotlib)."""
    try:
        a = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
        h = max(a.shape[0], b.shape[0])
        canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
        canvas[:a.shape[0], :a.shape[1]] = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        canvas[:b.shape[0], a.shape[1]:] = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        step = max(1, len(mkp1) // max_lines)
        for (x1, y1), (x2, y2) in zip(mkp1[::step], mkp2[::step]):
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + a.shape[1], int(y2)),
                     (60, 255, 60), 1, cv2.LINE_AA)
        out = os.path.join(out_root, 'match_imgs', tag)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, f'pair{pair_id:03d}_w{wx}_{wy}.png'), canvas)
    except Exception as e:
        print(f'[{tag}] match image not saved: {e}')


def register_imcui_detectors(weights_cache: Optional[str] = None) -> Dict[str, List]:
    """Register every catalog row kornia does not cover.

    Returns {'registered': [names], 'skipped': [(tag, reason)], 'error': str}."""
    configure_weights_cache(weights_cache)
    ok, err = imcui_available()
    if not ok:
        return {'registered': [], 'skipped': [], 'error': f'imcui not importable: {err}'}
    import imw_configs
    catalog = imw_configs.build_catalog()
    registered, skipped = [], []
    for tag, conf, dense in catalog:
        fams = algorithm_family(conf, dense)
        name = f'imw-{tag}'
        covered = sorted(set(fams) & E.KORNIA_FAMILIES)
        if covered:
            skipped.append((tag, f'{", ".join(covered)} provided by kornia'))
            continue
        if E.register_detector(name, (lambda c, t=tag, cf=conf, d=dense: IMWMatcher(c, t, cf, d)),
                               families=fams, source='imcui', matchers=['internal']):
            registered.append(name)
    for tag, why in skipped:
        print(f'[imcui] {tag}: not offered ({why})')
    return {'registered': registered, 'skipped': skipped, 'error': ''}
