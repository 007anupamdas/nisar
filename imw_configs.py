#!/usr/bin/env python3
"""
imcui (image-matching-webui) configuration catalog.

Configs are resolved from imcui's OWN registry (imcui/hloc/configs/{extractors,
matchers}.py, exposed as extract_features.confs / match_features.confs /
match_dense.confs) instead of being hand-written here. That guarantees each
model gets the exact conf the webui itself uses -- critically:

  * the LightGlue confs carry `features` + `model_name` (e.g. disk-lightglue
    -> disk_lightglue.pth). A bare {'name': 'lightglue'} silently loads the
    SuperPoint-LightGlue weights, which cannot match DISK/ALIKED/xfeat
    descriptors.
  * eloftr wants `model_name: eloftr_outdoor.ckpt` (not weights='outdoor').
  * roma / dkm / xfeat expect RGB input (grayscale: False), loftr/superpoint
    expect grayscale.

This module therefore needs `imcui` importable (both the prefetch box and the
pipeline box have it). If imcui is missing, IMW_CONFIGS resolves to [] with a
loud warning instead of crashing the import.

Each resolved entry is: (short_tag, conf_dict, dense_flag)
  short_tag : lower-case, alphanumeric + '-' only.  NO underscores
              (underscores break the CSV file_id parser in dqe_imw.py).
  conf_dict : imcui ImageMatchingAPI conf
              sparse: {'feature': ..., 'matcher': ..., 'dense': False}
              dense:  {'matcher': ..., 'dense': True}
  dense     : True for end-to-end dense matchers, False for detector+matcher.

Tuning via environment variables:
  NISAR_IMW_RESIZE_MAX    long-side cap fed to imcui preprocessing
                          (default 2048; window chips of 1024 px pass through
                          unresized, which preserves the 10 m/px geometry)
  NISAR_IMW_DENSE_MAX_KP  max matches kept by dense matchers (default 4096)
  NISAR_IMW_ONLY          comma list of tags to keep, e.g. "sp-lg,roma"
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

RESIZE_MAX = int(os.environ.get('NISAR_IMW_RESIZE_MAX', '2048'))
DENSE_MAX_KP = int(os.environ.get('NISAR_IMW_DENSE_MAX_KP', '4096'))

# ─────────────────────────────────────────────────────────────────────────────
# Catalog: (tag, extractor conf name | None, matcher conf name)
#   extractor None  -> dense / end-to-end matcher (match_dense.confs)
#   extractor given -> sparse pair (extract_features.confs + match_features.confs)
# Conf names must exist in YOUR installed imcui version; unknown names are
# skipped with a warning that lists what IS available, so version drift shows
# up at startup instead of after hours of matching.
# ─────────────────────────────────────────────────────────────────────────────
CATALOG: List[Tuple[str, Optional[str], str]] = [
    # ── Sparse: detector + matcher ───────────────────────────────────────────
    ('sp-lg',       'superpoint_max', 'superpoint-lightglue'),
    ('aliked-lg',   'aliked-n16',     'aliked-lightglue'),
    ('disk-lg',     'disk',           'disk-lightglue'),
    ('xfeat-lg',    'xfeat',          'xfeat_lightglue'),
    ('sp-sg',       'superpoint_max', 'superglue'),
    ('sift-lg',     'sift',           'sift-lightglue'),
    # ── Dense / end-to-end matchers ──────────────────────────────────────────
    ('loftr',       None,             'loftr'),
    ('eloftr',      None,             'eloftr'),
    ('aspanformer', None,             'aspanformer'),
    ('roma',        None,             'roma'),
    ('dkm',         None,             'dkm'),
    ('xfeat-dense', None,             'xfeat_dense'),
]


def _override_preprocessing(conf_section: Dict) -> None:
    """Keep native chip resolution: no forced WxH resize, generous long-side
    cap. grayscale/dfactor stay whatever the model's registry conf says."""
    pp = conf_section.setdefault('preprocessing', {})
    pp['force_resize'] = False
    pp['resize_max'] = RESIZE_MAX


def build_conf(feature_name: Optional[str], matcher_name: str) -> Tuple[Dict, bool]:
    """Resolve one catalog row into an ImageMatchingAPI conf dict."""
    from imcui.hloc import extract_features, match_dense, match_features

    dense = feature_name is None
    if dense:
        if matcher_name not in match_dense.confs:
            raise KeyError(
                f"dense matcher conf '{matcher_name}' not in this imcui. "
                f"Available: {sorted(match_dense.confs.keys())}"
            )
        mconf = copy.deepcopy(match_dense.confs[matcher_name])
        _override_preprocessing(mconf)
        mconf.setdefault('model', {})['max_keypoints'] = DENSE_MAX_KP
        return {'matcher': mconf, 'dense': True}, True

    if feature_name not in extract_features.confs:
        raise KeyError(
            f"extractor conf '{feature_name}' not in this imcui. "
            f"Available: {sorted(extract_features.confs.keys())}"
        )
    if matcher_name not in match_features.confs:
        raise KeyError(
            f"matcher conf '{matcher_name}' not in this imcui. "
            f"Available: {sorted(match_features.confs.keys())}"
        )
    fconf = copy.deepcopy(extract_features.confs[feature_name])
    mconf = copy.deepcopy(match_features.confs[matcher_name])
    _override_preprocessing(fconf)
    _override_preprocessing(mconf)
    # max_keypoints / keypoint_threshold are injected per-run by
    # ImageMatchingAPI(_update_config) from the dqe_imw.py side.
    return {'feature': fconf, 'matcher': mconf, 'dense': False}, False


def build_catalog(catalog: Optional[List[Tuple[str, Optional[str], str]]] = None
                  ) -> List[Tuple[str, Dict, bool]]:
    only = os.environ.get('NISAR_IMW_ONLY', '').strip()
    keep = {t.strip() for t in only.split(',') if t.strip()} if only else None

    entries: List[Tuple[str, Dict, bool]] = []
    failures: List[Tuple[str, str]] = []
    for tag, feat, match in (catalog if catalog is not None else CATALOG):
        if '_' in tag:
            failures.append((tag, "tag contains '_' (breaks file_id parser)"))
            continue
        if keep is not None and tag not in keep:
            continue
        try:
            conf, dense = build_conf(feat, match)
            entries.append((tag, conf, dense))
        except Exception as e:
            failures.append((tag, f'{type(e).__name__}: {e}'))

    if failures:
        print('[imw_configs] WARNING: skipped configs (not available in this '
              'imcui install):')
        for tag, err in failures:
            print(f'[imw_configs]   - {tag}: {err}')
    print(f'[imw_configs] {len(entries)} configs resolved: '
          f'{", ".join(t for t, _, _ in entries) or "(none)"}')
    return entries


try:
    IMW_CONFIGS: List[Tuple[str, Dict, bool]] = build_catalog()
except ImportError as _e:
    print(f'[imw_configs] ERROR: imcui not importable ({_e}). '
          f'IMW_CONFIGS is empty -- install image-matching-webui '
          f'(pip install -e <repo>) first.')
    IMW_CONFIGS = []
