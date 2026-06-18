#!/usr/bin/env python3
"""
imcui (image-matching-webui) configuration catalog -- FULL MATRIX edition.

Instead of hand-building conf dicts (which silently loaded the wrong
feature-specific weights, e.g. superpoint_lightglue.pth for an ALIKED run),
this composes each ImageMatchingAPI conf directly from imcui's own validated
conf tables:

    imcui.hloc.extract_features.confs   # sparse detectors
    imcui.hloc.match_features.confs     # sparse matchers (need a detector)
    imcui.hloc.match_dense.confs        # end-to-end / dense matchers

That guarantees the correct weights + feature pairing for every entry.

Each catalog row is (short_tag, conf_dict, dense_flag):
  short_tag : lower-case, alphanumeric + '-' only. NO underscores/()/spaces
              (the CSV file_id parser in dqe_imw.py splits on '_').
  dense     : True  -> end-to-end matcher (no separate detector)
              False -> detector + matcher

Edit SPARSE_SPECS / DENSE_SPECS below to choose what to run. Comment a row
out to skip it (saves prefetch + GPU time).
"""

import copy
from typing import Dict, List, Tuple

from imcui.hloc import extract_features, match_features, match_dense


# =============================================================================
# SPARSE: (tag, extractor_conf_name, sparse_matcher_conf_name)
# =============================================================================
# LightGlue is feature-specific -- pair each detector with ITS lightglue conf.
# Generic matchers (NN-mutual, adalam, Dual-Softmax) work with any descriptor,
# so they let you sweep detectors that have no learned matcher of their own.
SPARSE_SPECS: List[Tuple[str, str, str]] = [
    # ---- learned, feature-matched (correct weight pairing) -----------------
    ('sp-lg',        'superpoint_max', 'superpoint-lightglue'),
    ('disk-lg',      'disk',           'disk-lightglue'),
    ('aliked-lg',    'aliked-n16',     'aliked-lightglue'),
    ('sift-lg',      'sift',           'sift-lightglue'),
    ('sp-sg',        'superpoint_max', 'superglue'),

    # ---- generic NN sweep across SAR-relevant detectors --------------------
    # rord = rotation-robust D2Net; darkfeat = low-SNR robust; both good for SAR.
    ('sp-nn',        'superpoint_max', 'NN-mutual'),
    ('dedode-nn',    'dedode',         'NN-mutual'),
    ('r2d2-nn',      'r2d2',           'NN-mutual'),
    ('rord-nn',      'rord',           'NN-mutual'),
    ('d2net-nn',     'd2net-ss',       'NN-mutual'),
    ('alike-nn',     'alike',          'NN-mutual'),
    ('sfd2-nn',      'sfd2',           'NN-mutual'),
    ('rdd-nn',       'rdd',            'NN-mutual'),
    ('liftfeat-nn',  'liftfeat',       'NN-mutual'),
    ('ripe-nn',      'ripe',           'NN-mutual'),
    ('darkfeat-nn',  'darkfeat',       'NN-mutual'),
    ('lanet-nn',     'lanet',          'NN-mutual'),
    ('rootsift-nn',  'rootsift',       'NN-mutual'),
    ('sosnet-nn',    'sosnet',         'NN-mutual'),
    ('hardnet-nn',   'hardnet',        'NN-mutual'),

    # ---- geometry-aware (AdaLAM) -------------------------------------------
    ('aliked-adalam', 'aliked-n16',    'adalam'),
    ('disk-adalam',   'disk',          'adalam'),
]


# =============================================================================
# DENSE: (tag, dense_matcher_conf_name)
# =============================================================================
DENSE_SPECS: List[Tuple[str, str]] = [
    # ---- CROSS-MODAL champions (PRIORITY for SAR<->optical) ----------------
    ('minima-loftr', 'minima_loftr'),   # trained on multimodal synthetic pairs
    ('minima-roma',  'minima_roma'),
    ('xoftr',        'xoftr'),           # thermal<->visible; transfers to SAR<->opt
    ('omniglue',     'omniglue'),        # cross-domain generalization
    ('gim-roma',     'gim_roma'),        # trained on diverse internet video
    ('gim-dkm',      'gim(dkm)'),

    # ---- general dense (strong for SAR<->SAR) ------------------------------
    ('loftr',        'loftr'),
    ('eloftr',       'eloftr'),
    ('aspanformer',  'aspanformer'),
    ('topicfm',      'topicfm'),
    ('roma',         'roma'),
    ('dkm',          'dkm'),
    ('dad-roma',     'dad_roma'),
    ('rdd-dense',    'rdd_dense'),
    ('xfeat-lg',     'xfeat_lightglue'), # self-contained xfeat + lightglue
    ('xfeat-dense',  'xfeat_dense'),
    ('jamma',        'jamma'),

    # ---- heavy / experimental (enable deliberately) ------------------------
    # ('mast3r',     'mast3r'),     # 3D recon backbone, very heavy
    # ('duster',     'duster'),     # DUSt3R, very heavy
    # ('cotr',       'cotr'),       # extremely slow
    # ('sold2',      'sold2'),      # LINE matcher -- useless on SAR speckle
    # ('gluestick',  'gluestick'),  # line+point -- speckle unfriendly
]

# NOTE: global-retrieval descriptors (dir, netvlad, openibl, cosplace,
# eigenplaces) and the dummy 'example' extractor are intentionally excluded --
# they are image-retrieval / placeholders, not local feature matchers, and
# cannot drive the NISAR window-matching pipeline.


def _build() -> List[Tuple[str, Dict, bool]]:
    cfgs: List[Tuple[str, Dict, bool]] = []
    seen = set()

    def _check_tag(tag):
        if '_' in tag or '(' in tag or ' ' in tag:
            raise ValueError(f"bad tag '{tag}' (no _ () or space allowed)")
        if tag in seen:
            raise ValueError(f"duplicate tag '{tag}'")
        seen.add(tag)

    for tag, feat_name, matcher_name in SPARSE_SPECS:
        _check_tag(tag)
        if feat_name not in extract_features.confs:
            print(f"[imw_configs] SKIP {tag}: extractor '{feat_name}' not found")
            continue
        if matcher_name not in match_features.confs:
            print(f"[imw_configs] SKIP {tag}: matcher '{matcher_name}' not found")
            continue
        conf = {
            'feature': copy.deepcopy(extract_features.confs[feat_name]),
            'matcher': copy.deepcopy(match_features.confs[matcher_name]),
            'dense': False,
        }
        cfgs.append((tag, conf, False))

    for tag, matcher_name in DENSE_SPECS:
        _check_tag(tag)
        if matcher_name not in match_dense.confs:
            print(f"[imw_configs] SKIP {tag}: dense matcher '{matcher_name}' not found")
            continue
        conf = {
            'matcher': copy.deepcopy(match_dense.confs[matcher_name]),
            'dense': True,
        }
        cfgs.append((tag, conf, True))

    return cfgs


IMW_CONFIGS: List[Tuple[str, Dict, bool]] = _build()

if __name__ == '__main__':
    print(f"{len(IMW_CONFIGS)} configs built:")
    for tag, conf, dense in IMW_CONFIGS:
        kind = 'dense' if dense else 'sparse'
        mname = conf['matcher']['model'].get('name')
        print(f"  {tag:16s} [{kind}] matcher.model.name={mname}")
