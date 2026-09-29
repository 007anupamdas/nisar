#!/usr/bin/env python3
"""
Download every model weight automatch can use, into one folder you can copy to
an offline workstation.

kornia fetches its weights (DISK, ALIKED, XFeat, KeyNet/AffNet/HardNet, DeDoDe,
LoFTR, LightGlue) on first use into TORCH_HOME; imcui fetches from HuggingFace
into HF_HOME. Pointing both at one root and instantiating every model fills it.

    python prefetch_weights.py <cache_root> [--skip-imcui] [--only sift,disk_depth]

Then copy <cache_root> to the workstation and set AUTOMATCH_WEIGHTS_CACHE (or
'imcui weights cache' in the GUI) to it; also set TORCH_HOME=<cache_root>/torch
so kornia finds its files offline.

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
    for name in names:
        if only and name not in only:
            continue
        print(f'[prefetch] {name} ...')
        try:
            m = E.build_detector(name, cfg)
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
            ok.append(name)
        except Exception as e:
            failed.append((name, f'{type(e).__name__}: {e}'))
            traceback.print_exc()
    print(f'\n[prefetch] ok: {", ".join(ok) or "-"}')
    for n, err in failed:
        print(f'[prefetch] FAILED {n}: {err}')
    print(f'[prefetch] cache root: {root}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
