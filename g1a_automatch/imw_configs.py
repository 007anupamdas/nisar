#!/usr/bin/env python3
"""
imcui (image-matching-webui) configuration catalogue -- the full matrix (the
same rows as the NISAR tool's imw_configs.py).

Every ImageMatchingAPI conf is composed from imcui's OWN registry
(imcui.hloc.extract_features.confs / match_features.confs / match_dense.confs)
instead of being hand-written here, so each model gets exactly the conf the
webui uses -- in particular the right weights for feature-specific matchers
(disk-lightglue -> disk_lightglue.pth; a bare {'name': 'lightglue'} would
silently load the SuperPoint weights).

kornia first: automatch_imcui registers a row only when kornia does not
provide the same algorithm. Rows whose detector or matcher kornia has (DISK,
ALIKED, SIFT/RootSIFT, DeDoDe, DoG-HardNet/SOSNet, XFeat, LoFTR) are reported
as "not offered" together with the kornia detector to use instead. Rows in
KORNIA_EXEMPT are never refused: they use weights kornia does not ship.

build_catalog() returns (short_tag, conf_dict, dense_flag) per row:
  short_tag : lower-case alphanumerics and '-' only -- no '_', '(' or spaces
              (underscores break the file_id parser)
  conf_dict : sparse {'feature': ..., 'matcher': ..., 'dense': False}
              dense  {'matcher': ..., 'dense': True}
Rows the installed imcui does not have are skipped and reported, so version
drift shows up in the detector list instead of hours into a run.

Edit SPARSE_SPECS / DENSE_SPECS to choose what is offered (comment a row out
to hide it). Environment:
  NISAR_IMW_ONLY          comma list of tags to keep, e.g. "sp-lg,minima-roma"
  NISAR_IMW_RESIZE_MAX    long-side cap for imcui preprocessing (default 2048).
                          force_resize is switched off everywhere: several
                          registry confs resize to 640x480 / 320x240, which
                          would resample a window and lose its geometry.
  NISAR_IMW_DENSE_MAX_KP  matches kept by dense matchers (default 4096)
"""

import copy
import os
from typing import Dict, List, Optional, Tuple

RESIZE_MAX = int(os.environ.get('NISAR_IMW_RESIZE_MAX', '2048'))
DENSE_MAX_KP = int(os.environ.get('NISAR_IMW_DENSE_MAX_KP', '4096'))

# =============================================================================
# SPARSE: (tag, extractor conf name, sparse matcher conf name)
# =============================================================================
# LightGlue is feature-specific -- each detector is paired with ITS lightglue
# conf. Generic matchers (NN-mutual, adalam) work with any descriptor.
SPARSE_SPECS: List[Tuple[str, str, str]] = [
    # ---- learned, feature-matched ------------------------------------------
    ('sp-lg',        'superpoint_max', 'superpoint-lightglue'),
    ('disk-lg',      'disk',           'disk-lightglue'),
    ('aliked-lg',    'aliked-n16',     'aliked-lightglue'),
    ('sift-lg',      'sift',           'sift-lightglue'),
    ('sp-sg',        'superpoint_max', 'superglue'),

    # ---- generic mutual-NN sweep across detectors ---------------------------
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

    # ---- geometry-aware (AdaLAM) --------------------------------------------
    # ('aliked-adalam', 'aliked-n16', 'adalam'),  # imcui's AdaLAM needs keypoint scales,
    #                                             # which ALIKED does not give ("Missing key scales0")
    ('disk-adalam',   'disk',          'adalam'),
]

# =============================================================================
# DENSE: (tag, dense matcher conf name)
# =============================================================================
DENSE_SPECS: List[Tuple[str, str]] = [
    # ---- cross-modal ---------------------------------------------------------
    ('minima-loftr', 'minima_loftr'),   # trained on multimodal synthetic pairs
    ('minima-roma',  'minima_roma'),
    ('xoftr',        'xoftr'),          # thermal <-> visible
    ('omniglue',     'omniglue'),       # cross-domain generalisation
    ('gim-roma',     'gim_roma'),       # trained on diverse internet video
    ('gim-dkm',      'gim(dkm)'),

    # ---- general dense -------------------------------------------------------
    ('loftr',        'loftr'),
    ('eloftr',       'eloftr'),
    ('aspanformer',  'aspanformer'),
    ('topicfm',      'topicfm'),
    ('roma',         'roma'),
    ('dkm',          'dkm'),
    ('dad-roma',     'dad_roma'),
    ('rdd-dense',    'rdd_dense'),
    ('xfeat-lg',     'xfeat_lightglue'),  # self-contained xfeat + lightglue
    ('xfeat-dense',  'xfeat_dense'),
    # ('jamma',      'jamma'),  # imcui ships the conf but not the matcher module

    # ---- heavy / experimental (enable deliberately) --------------------------
    # ('mast3r',     'mast3r'),     # 3D reconstruction backbone, very heavy
    # ('duster',     'duster'),     # DUSt3R, very heavy
    # ('cotr',       'cotr'),       # extremely slow
    # ('sold2',      'sold2'),      # line matcher
    # ('gluestick',  'gluestick'),  # line + point
]
# Global-retrieval descriptors (dir, netvlad, openibl, cosplace, eigenplaces)
# and the dummy 'example' extractor are not local-feature matchers and are
# left out.

# Rows offered even though kornia implements the same architecture: they use
# weights kornia does not ship (MINIMA multimodal training, GIM).
KORNIA_EXEMPT = {'minima-loftr', 'minima-roma', 'gim-roma', 'gim-dkm'}


def catalog_rows() -> List[Tuple[str, Optional[str], str]]:
    """Every row: (tag, extractor conf name or None for dense, matcher conf)."""
    return ([(t, f, m) for t, f, m in SPARSE_SPECS]
            + [(t, None, m) for t, m in DENSE_SPECS])


def _keep_native_resolution(conf_section: Dict) -> None:
    pp = conf_section.setdefault('preprocessing', {})
    pp['force_resize'] = False
    pp['resize_max'] = RESIZE_MAX


def _registry():
    from imcui.hloc import extract_features, match_dense, match_features
    return extract_features.confs, match_features.confs, match_dense.confs


def _names(confs: Dict, limit: int = 40) -> str:
    names = sorted(confs)
    return ', '.join(names[:limit]) + (' ...' if len(names) > limit else '')


def build_conf(feature_name: Optional[str], matcher_name: str) -> Tuple[Dict, bool]:
    """Resolve one row into an ImageMatchingAPI conf. KeyError (naming what
    the installed imcui has) when a conf name is not in its registry."""
    feats, matchers, dense_confs = _registry()
    if feature_name is None:
        if matcher_name not in dense_confs:
            raise KeyError(f"dense matcher '{matcher_name}' is not in this imcui "
                           f"(it has: {_names(dense_confs)})")
        mconf = copy.deepcopy(dense_confs[matcher_name])
        _keep_native_resolution(mconf)
        mconf.setdefault('model', {})['max_keypoints'] = DENSE_MAX_KP
        return {'matcher': mconf, 'dense': True}, True
    if feature_name not in feats:
        raise KeyError(f"extractor '{feature_name}' is not in this imcui (it has: {_names(feats)})")
    if matcher_name not in matchers:
        raise KeyError(f"matcher '{matcher_name}' is not in this imcui (it has: {_names(matchers)})")
    fconf = copy.deepcopy(feats[feature_name])
    mconf = copy.deepcopy(matchers[matcher_name])
    _keep_native_resolution(fconf)
    _keep_native_resolution(mconf)
    # max_keypoints / keypoint_threshold are set per run by ImageMatchingAPI
    return {'feature': fconf, 'matcher': mconf, 'dense': False}, False


def _check_tag(tag: str, seen: set) -> Optional[str]:
    if not tag or any(c in tag for c in '_( ') or tag != tag.lower():
        return "bad tag (lower-case, no '_', '(' or spaces)"
    if tag in seen:
        return 'duplicate tag'
    seen.add(tag)
    return None


def build_catalog(report: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, Dict, bool]]:
    """Resolve every row the installed imcui provides. Rows that cannot be
    resolved are skipped; (tag, reason) is appended to `report` if given.
    ImportError when imcui itself is not importable."""
    _registry()  # fail fast (ImportError) when imcui is missing
    only = {t.strip() for t in os.environ.get('NISAR_IMW_ONLY', '').split(',') if t.strip()}
    entries: List[Tuple[str, Dict, bool]] = []
    skipped: List[Tuple[str, str]] = []
    seen: set = set()
    for tag, feat, match in catalog_rows():
        bad = _check_tag(tag, seen)
        if bad:
            skipped.append((tag, bad))
            continue
        if only and tag not in only:
            continue
        try:
            conf, dense = build_conf(feat, match)
        except Exception as e:
            skipped.append((tag, str(e.args[0]) if isinstance(e, KeyError) and e.args else
                            f'{type(e).__name__}: {e}'))
            continue
        entries.append((tag, conf, dense))
    for tag, why in skipped:
        print(f'[imw_configs] {tag}: skipped ({why})')
    print(f'[imw_configs] {len(entries)} of {len(catalog_rows())} imcui configurations resolved')
    if report is not None:
        report.extend(skipped)
    return entries


def __getattr__(name):
    # IMW_CONFIGS (as in the NISAR tool) is built on first use, not on import
    if name == 'IMW_CONFIGS':
        try:
            return build_catalog()
        except ImportError as e:
            print(f'[imw_configs] imcui not importable ({e}); IMW_CONFIGS is empty')
            return []
    raise AttributeError(name)


if __name__ == '__main__':
    for tag, conf, dense in build_catalog():
        print(f"  {tag:16s} [{'dense' if dense else 'sparse'}] "
              f"matcher.model.name={conf['matcher'].get('model', {}).get('name')}")
