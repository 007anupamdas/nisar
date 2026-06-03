#!/usr/bin/env python3
"""
Prefetch ALL imcui model weights for the configs in imw_configs.IMW_CONFIGS.

Run this on an INTERNET-CONNECTED machine that has `imcui` installed (it does
NOT need the NISAR geospatial stack -- only imcui + torch). It instantiates
every config and runs one dummy match so that every checkpoint (including the
extra backbone weights that dense matchers like RoMa/DKM pull lazily) is
downloaded into the cache directories. Then tar up the cache and hand-carry it
to the air-gapped machine.

USAGE
-----
    # 1. Choose a clean, self-contained cache root so it's easy to copy:
    export HF_HOME=/tmp/imw_cache/huggingface
    export TORCH_HOME=/tmp/imw_cache/torch
    export XDG_CACHE_HOME=/tmp/imw_cache/xdg        # catches misc caches

    python prefetch_imw_weights.py

    # 2. Package everything:
    tar czf imw_cache.tgz -C /tmp/imw_cache .

    # 3. Hand-carry imw_cache.tgz to the intranet machine and extract, then
    #    on that machine set (BEFORE running dqe_imw.py):
    #       export HF_HOME=/opt/imw_cache/huggingface
    #       export TORCH_HOME=/opt/imw_cache/torch
    #       export XDG_CACHE_HOME=/opt/imw_cache/xdg
    #       export HF_HUB_OFFLINE=1
    #       export TRANSFORMERS_OFFLINE=1

NOTES
-----
* Keep the imcui VERSION identical on both machines, or repo IDs / filenames
  may differ and the offline run will miss a file.
* Runs on CPU by default (device="cpu") -- you do NOT need a GPU just to
  download weights. Override with PREFETCH_DEVICE=cuda if you want to also
  validate a forward pass on GPU.
* If a single config fails to download, the script keeps going and reports a
  summary at the end so you can see exactly which model is the problem.
"""

import os
import sys
import traceback

import numpy as np

from imw_configs import IMW_CONFIGS


def _dummy_pair(size: int = 512):
    """Two small structured RGB uint8 images with enough texture that the
    matchers actually run their full forward pass (and thus trigger any
    lazily-downloaded second-stage weights)."""
    rng = np.random.default_rng(0)
    base = (rng.random((size, size)) * 255).astype(np.uint8)
    # Add some gradient structure so detectors find keypoints.
    yy, xx = np.mgrid[0:size, 0:size]
    grad = ((xx + yy) % 256).astype(np.uint8)
    img0 = (0.5 * base + 0.5 * grad).astype(np.uint8)
    # Shift the second image slightly so there is a real disparity to match.
    img1 = np.roll(img0, shift=8, axis=1)
    rgb0 = np.stack([img0] * 3, axis=-1)
    rgb1 = np.stack([img1] * 3, axis=-1)
    return rgb0, rgb1


def main() -> int:
    device = os.environ.get('PREFETCH_DEVICE', 'cpu')
    print(f'[prefetch] device={device}')
    print(f'[prefetch] HF_HOME={os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")}')
    print(f'[prefetch] TORCH_HOME={os.environ.get("TORCH_HOME", "(default ~/.cache/torch)")}')
    print(f'[prefetch] {len(IMW_CONFIGS)} configs to fetch\n')

    from imcui.api import ImageMatchingAPI

    img0, img1 = _dummy_pair()
    ok, failed = [], []

    for tag, conf, dense in IMW_CONFIGS:
        print('=' * 70)
        print(f'[prefetch] >>> {tag} (dense={dense})')
        try:
            api = ImageMatchingAPI(conf=conf, device=device,
                                   detect_threshold=0.015,
                                   max_keypoints=2048,
                                   match_threshold=0.2)
            # Force a forward pass so any second-stage weights download too.
            try:
                _ = api(img0, img1)
            except Exception as fe:  # forward may fail on CPU for some models
                print(f'[prefetch]   (forward pass note: {type(fe).__name__}: {fe})')
                print('[prefetch]   weights likely still downloaded during init.')
            del api
            print(f'[prefetch] <<< {tag} OK')
            ok.append(tag)
        except Exception as e:
            print(f'[prefetch] !!! {tag} FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            failed.append((tag, f'{type(e).__name__}: {e}'))

    print('\n' + '=' * 70)
    print(f'[prefetch] DONE. ok={len(ok)} failed={len(failed)}')
    if ok:
        print(f'[prefetch]   downloaded: {", ".join(ok)}')
    if failed:
        print('[prefetch]   FAILED configs (resolve before going offline):')
        for tag, err in failed:
            print(f'[prefetch]     - {tag}: {err}')
        return 1
    print('[prefetch] All configs fetched. Now tar up your cache dirs and '
          'transfer them to the air-gapped machine.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
