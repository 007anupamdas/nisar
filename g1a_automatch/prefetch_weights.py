#!/usr/bin/env python3
"""
Collect every model weight automatch can use into one folder you can copy to
an offline workstation.

    python prefetch_weights.py <cache_root> [--skip-imcui] [--only sift,disk] [--defaults-only]

1. The weights check (automatch_weights) lists, without loading anything, every
   file each detector variant needs: every weight choice of every detector
   (both DISK checkpoints, all DeDoDe detector x descriptor weights, all
   ALIKED and LoFTR models, DoG descriptors, AffNet/OriNet, the LightGlue
   weights, and the imcui models). --defaults-only: default settings only.
2. Files already on this machine (e.g. in ~/.cache/torch/hub/checkpoints) are
   copied into <cache_root>; the rest are downloaded there, under the exact
   names the loaders look for.
3. Models whose weights cannot be listed that way (some imcui models) are
   built once for real with <cache_root> as their download folder.

Then copy <cache_root> to the workstation and set the GUI's 'Weights folder'
(or AUTOMATCH_WEIGHTS_CACHE) to it. Weights are looked up there and in the
default folders before anything is downloaded; TORCH_HOME does not need to be
set. `python automatch_weights.py --weights-cache <cache_root>` shows what is
there.

Behind a TLS-inspecting proxy, PREFETCH_INSECURE_SSL=1 disables certificate
checks for this download only -- use it only on a trusted network.
"""

import argparse
import os
import shutil
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
                    help="only each detector's default settings")
    a = ap.parse_args()

    root = os.path.abspath(a.cache_root)
    ckpt_dir = os.path.join(root, 'torch', 'hub', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = os.path.join(root, 'torch')
    os.environ['HF_HOME'] = os.path.join(root, 'huggingface')
    if os.environ.get('PREFETCH_INSECURE_SSL') == '1':
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        print('[prefetch] WARNING: TLS verification disabled for this run')

    import torch as th
    import automatch_engine as E
    import automatch_imcui as IM
    import automatch_weights as W

    th.hub.set_dir(os.path.join(root, 'torch', 'hub'))
    E.add_weights_dir(root)
    if not a.skip_imcui:
        res = IM.register_imcui_detectors(root)
        if res['error']:
            print(f'[prefetch] imcui skipped: {res["error"]}')
    only = [x.strip() for x in a.only.split(',') if x.strip()] or None
    if a.defaults_only:
        orig_axes = W._weight_axes
        W._weight_axes = lambda name: {}
    print('[prefetch] listing the weight files ...')
    rep = W.check(detectors=only, imcui=not a.skip_imcui, quiet=True)
    if a.defaults_only:
        W._weight_axes = orig_axes

    ok, failed = [], []
    rebuild = []            # (detector, variant) to build for real
    seen = set()
    for det in rep['detectors']:
        for f in det['files']:
            key = (f['kind'], f['source'], f['file'])
            if key in seen:
                continue
            seen.add(key)
            try:
                if f['kind'] == 'torch-hub':
                    target = os.path.join(ckpt_dir, f['file'])
                    if os.path.isfile(target) and W.file_status(target) == 'ok':
                        ok.append(f"{f['file']} (already there)")
                    elif f['status'] == 'ok':
                        shutil.copy2(f['path'], target)
                        ok.append(f"{f['file']} (copied from {f['folder']})")
                    else:
                        print(f"[prefetch] downloading {f['file']} <- {f['source']}")
                        th.hub.download_url_to_file(f['source'], target + '.part', progress=True)
                        os.replace(target + '.part', target)
                        ok.append(f"{f['file']} (downloaded)")
                elif f['kind'] == 'hf' and f.get('repo'):
                    import huggingface_hub
                    download = IM._ORIG_HF or huggingface_hub.hf_hub_download
                    download(f['repo'], f['repo_file'], repo_type=f.get('repo_type'),
                             revision=f.get('revision'), cache_dir=os.path.join(root, 'huggingface', 'hub'))
                    ok.append(f"{f['repo_file']} (HuggingFace {f['repo']})")
                else:
                    rebuild += [(det['detector'], v) for v in f['needed_by'][:1]]
            except Exception as e:
                failed.append((f['file'], f'{type(e).__name__}: {e}'))
        for v in det['variants']:
            if v['status'] in ('partial', 'error'):
                rebuild.append((det['detector'], v['variant']))

    # models whose weights could not be listed: build them for real, once,
    # with the cache as their download folder
    done = set()
    cfg = E.PipelineConfig(num_features=256, use_amp=False)
    for det, variant in rebuild:
        if variant in done:
            continue
        done.add(variant)
        axes = {} if a.defaults_only else W._weight_axes(E.resolve_detector_name(det)[0])
        for m in E.iter_variants(det, cfg, axes if E.resolve_detector_name(det)[0] in E.KORNIA_DETECTORS else {}):
            if m.get_filename_prefix() != variant:
                continue
            print(f'[prefetch] building {variant} to fetch its weights ...')
            try:
                m.load_models()
                ok.append(f'{variant} (built)')
            except Exception as e:
                failed.append((variant, f'{type(e).__name__}: {e}'))
                traceback.print_exc()
            finally:
                m.unload_model()

    print(f'\n[prefetch] {len(ok)} item(s) ready')
    for n in ok:
        print(f'  ok  {n}')
    for n, err in failed:
        print(f'[prefetch] FAILED {n}: {err}')
    print(f'[prefetch] cache root: {root}\n'
          f'[prefetch] check it with: python automatch_weights.py --weights-cache {root}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
