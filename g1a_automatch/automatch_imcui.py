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
(huggingface/, torch/). HF_HOME is set (if not already) and HF_HUB_OFFLINE
turned on; TORCH_HOME is left alone so kornia keeps its own weight folder --
the cache's torch/hub is searched for weights and used as the torch-hub folder
only while imcui models load and run.
"""

import contextlib
import copy
import gc
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch as th

import automatch_engine as E

_API = None
# torch-hub folder of the configured weights cache: switched in only while an
# imcui model loads or runs, so kornia keeps its own (default) folder.
_IMCUI_HUB: Optional[str] = None
_WEIGHTS_ROOT: Optional[str] = None
_ORIG_HF = None


def configure_weights_cache(path: Optional[str] = None) -> Optional[str]:
    path = path or os.environ.get('AUTOMATCH_WEIGHTS_CACHE')
    if not path:
        return None
    if not os.path.isdir(path):
        print(f'[imcui] weights cache {path} not found -- using default caches')
        return None
    global _IMCUI_HUB, _WEIGHTS_ROOT
    _WEIGHTS_ROOT = os.path.abspath(path)
    os.environ.setdefault('HF_HOME', os.path.join(path, 'huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    # TORCH_HOME is deliberately NOT changed: that would also move kornia's
    # weight folder away from where its weights were downloaded before. The
    # cache is searched for kornia weights too (add_weights_dir), and becomes
    # the torch-hub folder only while imcui models load and run.
    E.add_weights_dir(path)
    hub = os.path.join(path, 'torch', 'hub')
    _IMCUI_HUB = hub if os.path.isdir(hub) else None
    print(f'[imcui] weights cache: {path} (offline)')
    return path


# ── HuggingFace weights ──────────────────────────────────────────────────────
# imcui fetches most checkpoints with huggingface_hub.hf_hub_download. Like the
# torch-hub lookup in automatch_engine, the file is first looked for in every
# HuggingFace cache on this machine -- the active one, the weights folder's
# and the default ~/.cache/huggingface/hub -- so weights imcui downloaded
# earlier are used even when HF_HOME points elsewhere.
def hf_cache_dirs() -> List[str]:
    dirs = []
    try:
        from huggingface_hub import constants
        dirs.append(constants.HF_HUB_CACHE)
    except Exception:
        pass
    if os.getenv('HF_HUB_CACHE'):
        dirs.append(os.environ['HF_HUB_CACHE'])
    if os.getenv('HF_HOME'):
        dirs.append(os.path.join(os.environ['HF_HOME'], 'hub'))
    if _WEIGHTS_ROOT:
        dirs.append(os.path.join(_WEIGHTS_ROOT, 'huggingface', 'hub'))
    xdg = os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache'))
    dirs.append(os.path.join(xdg, 'huggingface', 'hub'))
    out: List[str] = []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in out:
            out.append(d)
    return out


def find_hf_file(repo_id: str, filename: str, subfolder: Optional[str] = None,
                 repo_type: Optional[str] = None, revision: Optional[str] = None
                 ) -> Tuple[Optional[str], List[str]]:
    """(path in a HuggingFace cache or None, caches searched)."""
    dirs = hf_cache_dirs()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None, dirs
    name = f'{subfolder}/{filename}' if subfolder else filename
    for d in dirs:
        try:
            p = try_to_load_from_cache(repo_id, name, cache_dir=d, revision=revision, repo_type=repo_type)
        except Exception:
            p = None
        if isinstance(p, str) and os.path.isfile(p):
            return p, dirs
    return None, dirs


def _hf_download_any_cache(repo_id, filename, *args, **kwargs):
    path, searched = find_hf_file(repo_id, filename, kwargs.get('subfolder'),
                                  kwargs.get('repo_type'), kwargs.get('revision'))
    if E.WEIGHTS_TRACE_HOOK is not None:
        sub = kwargs.get('subfolder')
        name = f'{sub}/{filename}' if sub else filename
        answer = E.WEIGHTS_TRACE_HOOK('hf', url=f'hf:{repo_id}/{name}', file=os.path.basename(filename),
                                      path=path, searched=searched, repo=repo_id, repo_file=name,
                                      repo_type=kwargs.get('repo_type'), revision=kwargs.get('revision'))
        if answer is not None:
            return answer
    if path:
        return path
    return _ORIG_HF(repo_id, filename, *args, **kwargs)


def install_hf_lookup() -> bool:
    """Route hf_hub_download (the package's and any imcui module's copy)
    through the cache lookup. Safe to call repeatedly; False without
    huggingface_hub."""
    global _ORIG_HF
    try:
        import huggingface_hub
        import huggingface_hub.file_download as fd
    except ImportError:
        return False
    if _ORIG_HF is None:
        _ORIG_HF = getattr(fd, 'hf_hub_download', None) or huggingface_hub.hf_hub_download
    for mod in [huggingface_hub, fd] + [m for n, m in list(sys.modules.items())
                                        if m is not None and n.split('.')[0] == 'imcui']:
        if getattr(mod, 'hf_hub_download', None) is _ORIG_HF:
            setattr(mod, 'hf_hub_download', _hf_download_any_cache)
    return True


@contextlib.contextmanager
def _imcui_hub():
    """Use the weights cache as the torch-hub folder for the duration."""
    if not _IMCUI_HUB:
        yield
        return
    old = th.hub.get_dir()
    th.hub.set_dir(_IMCUI_HUB)
    try:
        yield
    finally:
        th.hub.set_dir(old)


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
        fmodel = conf.get('feature', {}).get('model', {})
        fams.append(str(fmodel.get('name', '')))
        if fmodel.get('descriptor'):      # imcui 'hardnet' = DoG + HardNet, etc.
            fams.append(str(fmodel['descriptor']))
        feats = conf.get('matcher', {}).get('model', {}).get('features')
        if feats:
            fams.append(str(feats))
    return [f.lower() for f in fams if f]


# Keys ImageMatchingAPI overwrites from its own arguments (imcui api/core.py
# _updata_config): they are offered as api.detect_threshold /
# api.match_threshold instead, and the keypoint budget follows num_features.
# 'name' / 'model_name' / 'features' select the weights and their pairing.
_HIDDEN_KEYS = {
    'feature': {'name', 'model_name', 'max_keypoints', 'keypoint_threshold',
                'max_num_keypoints', 'detection_threshold'},
    'matcher': {'name', 'model_name', 'features', 'match_threshold'},
}


def _token_for(section: str, key: str, used: set) -> str:
    base = (section[0] + ''.join(p[0] for p in key.split('_') if p))[:4].lower()
    tok, i = base, 2
    while tok in used or tok in ('kt', 'mt'):
        tok, i = f'{base}{i}', i + 1
    used.add(tok)
    return tok


def imcui_param_specs(conf: Dict, dense: bool) -> List[E.ParamSpec]:
    """Editable parameters of one catalog configuration, read from the conf the
    installed imcui itself provides (so they always match its version)."""
    specs: List[E.ParamSpec] = []
    if not dense:
        specs.append(E.ParamSpec('api.detect_threshold', 'float', 0.015, token='kt',
                                 label='Detection threshold',
                                 help="imcui's keypoint_threshold for the detector"))
    specs.append(E.ParamSpec('api.match_threshold', 'float', 0.2, token='mt',
                             label='Match threshold', help="imcui's match_threshold"))
    used: set = set()
    for section in (('matcher',) if dense else ('feature', 'matcher')):
        model = (conf.get(section) or {}).get('model') or {}
        for key, val in model.items():
            if key in _HIDDEN_KEYS[section] or isinstance(val, (dict, list, tuple)) or val is None:
                continue
            kind = ('bool' if isinstance(val, bool) else 'int' if isinstance(val, int)
                    else 'float' if isinstance(val, float) else 'str')
            specs.append(E.ParamSpec(f'{section}.{key}', kind, val, token=_token_for(section, key, used),
                                     label=f'{section} · {key}'))
    return specs


class IMWMatcher(E.DenseWindowMatcher):
    """One imcui ImageMatchingAPI configuration as a DenseWindowMatcher."""

    matcher_token = 'internal'

    def __init__(self, config: E.PipelineConfig, imw_tag: str, imw_conf: Dict, dense: bool,
                 overrides: Optional[Dict] = None):
        super().__init__(config)
        if '_' in imw_tag:
            raise ValueError(f"imw_tag must not contain '_' (got '{imw_tag}')")
        overrides = dict(overrides or {})
        conf = copy.deepcopy(imw_conf)
        # 'dense' is what imcui 0.0.x reads; newer imcui reads 'standalone'.
        conf['dense'] = conf['standalone'] = bool(dense)
        self.detect_threshold = float(overrides.pop('api.detect_threshold', config.imw_detect_threshold))
        self.match_threshold = float(overrides.pop('api.match_threshold', config.imw_match_threshold))
        for key, val in overrides.items():
            section, k = key.split('.', 1)
            conf.setdefault(section, {}).setdefault('model', {})[k] = val
        # Keep the conf in step with what the API forces, whichever version.
        if not dense and 'feature' in conf:
            conf['feature'].setdefault('model', {})['keypoint_threshold'] = self.detect_threshold
        if 'matcher' in conf:
            conf['matcher'].setdefault('model', {})['match_threshold'] = self.match_threshold
        self.imw_tag = imw_tag
        self.imw_conf = conf
        self.dense = dense
        self.api = None
        self._n_saved = 0

    def get_detector_name(self) -> str:
        return f'imw-{self.imw_tag}'

    def _prefix_base(self) -> str:
        return f'imw-{self.imw_tag}'

    def _detector_needs_inpaint(self) -> bool:
        return True

    def _ensure_api(self) -> bool:
        """Load the imcui model once; raises E.ModelLoadError if it cannot be
        (remembered for the whole job, so it is not retried per window)."""
        if self.api is not None:
            return True

        def build():
            print(f'[{self.get_detector_name()}] Loading model...')
            install_hf_lookup()   # model modules imported since registration
            with _imcui_hub():
                return _api_class()(
                    conf=self.imw_conf,
                    device=str(self.device),
                    detect_threshold=self.detect_threshold,
                    max_keypoints=self.config.num_features,
                    match_threshold=self.match_threshold,
                )
        self.api = self._load_model(f'imcui {self.imw_tag}', build)
        return True

    def _build_models(self) -> None:
        self._ensure_api()

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
        self._ensure_api()
        need = getattr(self.config, 'min_gpu_free_gb', 1.5)
        free = self._gpu_free_gb()
        if free is not None and free < need:
            E.safe_cuda_empty_cache()
            free = self._gpu_free_gb()
            if free is not None and free < need:
                raise RuntimeError(f'only {free:.1f} GB GPU memory free (< {need} GB)')
        u1 = (img1 * 255.0).clip(0, 255).astype(np.uint8)
        u2 = (img2 * 255.0).clip(0, 255).astype(np.uint8)
        with th.inference_mode(), _imcui_hub():
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


# imcui matcher model -> the kornia matcher that does the same job
_MATCHER_EQUIVALENT = {'lightglue': 'lgm', 'nearest_neighbor': 'mnn', 'adalam': 'ada'}


def kornia_replacement(conf: Dict, dense: bool, fams: List[str]) -> str:
    """How to get a kornia-covered row from kornia, e.g. "'dog' (descriptor
    hardnet) with mnn". '' when kornia does not cover the row."""
    covered = [f for f in fams if f in E.KORNIA_FAMILIES]
    if not covered:
        return ''
    fam = covered[-1]            # most specific: descriptor / matcher features
    det = E.KORNIA_EQUIVALENT.get(fam, fam)
    text = f"'{det}'"
    if det == 'dog' and fam != 'dog':
        text += f' (descriptor {fam})'
    elif det == 'sift' and fam == 'rootsift':
        text += ' (RootSIFT on)'
    if not dense:
        mname = str(conf.get('matcher', {}).get('model', {}).get('name', '')).lower()
        m = _MATCHER_EQUIVALENT.get(mname)
        if m and m in E.KORNIA_MATCHERS.get(det, []) + E.OPTIONAL_MATCHERS.get(det, []):
            text += f' with {m}'
    return text


def register_imcui_detectors(weights_cache: Optional[str] = None) -> Dict[str, List]:
    """Register every catalogue row kornia does not cover.

    Returns {'registered': [names], 'skipped': [(tag, reason)],
    'not_offered': [{'tag', 'kind', 'reason', 'use'}], 'error': str};
    kind is 'kornia' (use the kornia detector named in 'use') or
    'unavailable' (not in the installed imcui)."""
    configure_weights_cache(weights_cache)
    ok, err = imcui_available()
    if not ok:
        return {'registered': [], 'skipped': [], 'not_offered': [], 'error': f'imcui not importable: {err}'}
    install_hf_lookup()
    import imw_configs
    unavailable: List[Tuple[str, str]] = []
    try:
        catalog = imw_configs.build_catalog(unavailable)
    except ImportError as e:
        return {'registered': [], 'skipped': [], 'not_offered': [], 'error': f'imcui not importable: {e}'}
    exempt = set(getattr(imw_configs, 'KORNIA_EXEMPT', ()))
    registered, not_offered = [], []
    for tag, conf, dense in catalog:
        fams = algorithm_family(conf, dense)
        name = f'imw-{tag}'
        use = '' if tag in exempt else kornia_replacement(conf, dense, fams)
        if use:
            covered = sorted(set(fams) & E.KORNIA_FAMILIES)
            not_offered.append({'tag': tag, 'kind': 'kornia', 'use': use,
                                'reason': f'{", ".join(covered)} is provided by kornia: use {use}'})
            continue
        if E.register_detector(name, (lambda c, kw, t=tag, cf=conf, d=dense: IMWMatcher(c, t, cf, d, kw)),
                               families=() if tag in exempt else fams, source='imcui',
                               matchers=['internal'], params=imcui_param_specs(conf, dense)):
            registered.append(name)
    for tag, why in unavailable:
        not_offered.append({'tag': tag, 'kind': 'unavailable', 'use': '',
                            'reason': f'not in the installed imcui: {why}'})
    for row in not_offered:
        print(f"[imcui] {row['tag']}: not offered ({row['reason']})")
    return {'registered': registered, 'skipped': [(r['tag'], r['reason']) for r in not_offered],
            'not_offered': not_offered, 'error': ''}
