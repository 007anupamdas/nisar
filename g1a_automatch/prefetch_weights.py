#!/usr/bin/env python3
"""
Download every model weight automatch can use, into one folder you can copy to
an offline workstation.

kornia fetches its weights (DISK, ALIKED, XFeat, KeyNet/AffNet/HardNet, DeDoDe,
LoFTR, LightGlue) on first use into TORCH_HOME; imcui fetches from HuggingFace
into HF_HOME. Pointing both at one root and instantiating every model fills it.

    python prefetch_weights.py <cache_root> [--skip-imcui] [--only sift,disk] [--defaults-only]

Every weight a detector can be configured with is fetched (both DISK
checkpoints, all DeDoDe detector and descriptor weights, all ALIKED models,
all LoFTR weights), so any choice made in the GUI works offline;
--defaults-only fetches just the default of each.

Then copy <cache_root> to the workstation and set the GUI's 'Weights folder'
(or AUTOMATCH_WEIGHTS_CACHE) to it. Weights are looked up there and in kornia's
default folder (~/.cache/torch/hub) before anything is downloaded; TORCH_HOME
does not need to be set.

Behind a TLS-inspecting proxy, PREFETCH_INSECURE_SSL=1 disables certificate
checks for this download only -- use it only on a trusted network.
"""

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_root')
    ap.add_argument('--skip-imcui', action='store_true')
    ap.add_argument('--only', default='', help='comma list of detector names')
    ap.add_argument('--defaults-only', action='store_true',
                    help="fetch only each detector's default weights")
    a = ap.parse_args()

    root = os.path.abspath(a.cache_root)
    os.makedirs(root, exist_ok=True)
    os.environ['TORCH_HOME'] = os.path.join(root, 'torch')
    os.environ['HF_HOME'] = os.path.join(root, 'huggingface')
    if os.environ.get('PREFETCH_INSECURE_SSL') == '1':
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('[prefetch] WARNING: TLS verification disabled for this run')

    import numpy as np
    import torch as th
    import automatch_engine as E

    only = {x.strip() for x in a.only.split(',') if x.strip()}
    img = np.random.default_rng(0).random((256, 256)).astype('float32')
    t = th.from_numpy(img)[None, None]
    ok, failed = [], []

    cfg = E.PipelineConfig(num_features=256, use_amp=False)
    names = list(E.KORNIA_DETECTORS)
    if not a.skip_imcui:
        import automatch_imcui as IM
        res = IM.register_imcui_detectors(root)
        if res['error']:
            print(f'[prefetch] imcui skipped: {res["error"]}')
        names += res['registered']
    jobs = []
    for name in names:
        if only and name not in only:
            continue
        weights = [sp for sp in E.detector_param_specs(name)
                   if sp.kind == 'choice' and sp.scope == 'detector']
        if weights and not a.defaults_only:
            # one run per weight file: each choice with the other settings at default
            jobs += [(name, {sp.name: [c]}) for sp in weights for c in sp.choices]
        else:
            jobs.append((name, {}))
    for name, params in jobs:
        label = name + (' ' + ', '.join(f'{k}={v[0]}' for k, v in params.items()) if params else '')
        print(f'[prefetch] {label} ...')
        try:
            m = E.build_detector(name, cfg, params)
            if isinstance(m, E.DenseWindowMatcher):
                m.match_images(img, img)
            elif isinstance(m, E.LoFTRMatcher):
                m._process_single_window(img * 1000, img * 1000, 0, 0, 0, 0,
                                         {'x01': 0, 'y01': 0, 'xres1': 1, 'yres1': -1, 'x02': 0, 'y02': 0,
                                          'xres2': 1, 'yres2': -1, 'nisar_crop_row_offset': 0,
                                          'nisar_crop_col_offset': 0, 'pair_id': 0, 'nisar_pol': 'x',
                                          's1_ref_tag': 'x'}, 'loftr_internal', None)
            else:
                l1, d1, l2, d2 = m.detect_and_describe(t, t)
                if 'lgm' in m.get_available_matchers():
                    hw = th.tensor(t.shape[2:])
                    m.match_lgm(d1, d2, l1, l2, hw, hw, feature_name=m._lightglue_feature_name())
            ok.append(label)
        except Exception as e:
            failed.append((label, f'{type(e).__name__}: {e}'))
            traceback.print_exc()
        finally:
            try:
                m.unload_model()
            except Exception:
                pass
    print(f'\n[prefetch] ok: {", ".join(ok) or "-"}')
    for n, err in failed:
        print(f'[prefetch] FAILED {n}: {err}')
    print(f'[prefetch] cache root: {root}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
