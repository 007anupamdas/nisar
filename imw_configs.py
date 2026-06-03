#!/usr/bin/env python3
"""
imcui (image-matching-webui) configuration catalog.

Kept deliberately free of heavy imports (no torch / kornia / rasterio) so it
can be imported on a download-only / internet-connected machine that only has
`imcui` installed, for weight prefetching (see prefetch_imw_weights.py), as
well as inside the full NISAR pipeline (dqe_imw.py).

Each catalog entry is: (short_tag, conf_dict, dense_flag)
  short_tag : lower-case, alphanumeric + '-' only.  NO underscores
              (underscores break the CSV file_id parser in dqe_imw.py).
  conf_dict : imcui ImageMatchingAPI conf (see imcui/hloc/configs/*).
  dense     : True for end-to-end dense matchers, False for detector+matcher.
"""

from typing import Dict, List, Optional, Tuple


def sparse_conf(feat_name: str, matcher_name: str,
                max_kp: int = 4096, resize_max: int = 1600,
                keypoint_threshold: float = 0.005) -> Dict:
    return {
        'feature': {
            'output': f'feats-{feat_name}-n{max_kp}-rmax{resize_max}',
            'model': {
                'name': feat_name,
                'max_keypoints': max_kp,
                'keypoint_threshold': keypoint_threshold,
            },
            'preprocessing': {
                'grayscale': True,
                'force_resize': False,
                'resize_max': resize_max,
                'dfactor': 8,
            },
        },
        'matcher': {
            'output': f'matches-{matcher_name}',
            'model': {
                'name': matcher_name,
                'match_threshold': 0.2,
            },
        },
        'dense': False,
    }


def dense_conf(matcher_name: str, weights: Optional[str] = None,
               max_kp: int = 4000, resize_max: int = 1024) -> Dict:
    model_cfg = {
        'name': matcher_name,
        'max_keypoints': max_kp,
        'match_threshold': 0.2,
    }
    if weights is not None:
        model_cfg['weights'] = weights
    return {
        'matcher': {
            'output': f'matches-{matcher_name}',
            'model': model_cfg,
            'preprocessing': {
                'grayscale': True,
                'force_resize': False,
                'resize_max': resize_max,
                'dfactor': 8,
            },
            'max_error': 1,
            'cell_size': 1,
        },
        'dense': True,
    }


# Curated catalog. Comment out rows you don't want to evaluate to save time.
IMW_CONFIGS: List[Tuple[str, Dict, bool]] = [
    # ── Sparse: detector + matcher ──────────────────────────────────────────
    ('sp-lg',         sparse_conf('superpoint', 'lightglue', max_kp=8192), False),
    ('aliked-lg',     sparse_conf('aliked',     'lightglue', max_kp=8192), False),
    ('disk-lg',       sparse_conf('disk',       'lightglue', max_kp=8192), False),
    ('xfeat-lg',      sparse_conf('xfeat',      'lightglue', max_kp=8192), False),
    ('sp-sg',         sparse_conf('superpoint', 'superglue', max_kp=4096), False),
    # ── Dense / end-to-end matchers ─────────────────────────────────────────
    ('eloftr',        dense_conf('eloftr',      weights='outdoor'),        True),
    ('aspanformer',   dense_conf('aspanformer', weights='outdoor'),        True),
    ('roma',          dense_conf('roma',        weights='outdoor',
                                 max_kp=2000, resize_max=864),             True),
    ('dkm',           dense_conf('dkm',         weights='outdoor',
                                 max_kp=2000, resize_max=864),             True),
    ('xfeat-dense',   dense_conf('xfeat_dense'),                           True),
]
