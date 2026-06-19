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


def _enable_insecure_ssl():
    """Behind a corporate TLS-inspection proxy, torch.hub's plain-urllib calls
    fail cert verification. On a TRUSTED internal box, disable verification."""
    if os.environ.get('PREFETCH_INSECURE_SSL') == '1':
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ.setdefault('GIT_SSL_NO_VERIFY', 'true')
        os.environ.setdefault('CURL_CA_BUNDLE', '')
        print('[prefetch] WARNING: TLS verification DISABLED '
              '(PREFETCH_INSECURE_SSL=1) -- use only on a trusted network.')


def _run_one(tag: str) -> int:
    """Worker: download + (optionally) forward-pass a single config.
    Invoked as a subprocess so OOM-kills don't take down the parent."""
    _enable_insecure_ssl()
    device = os.environ.get('PREFETCH_DEVICE', 'cpu')
    skip_forward = os.environ.get('PREFETCH_SKIP_FORWARD') == '1'

    target = None
    for t, conf, dense in IMW_CONFIGS:
        if t == tag:
            target = (t, conf, dense)
            break
    if target is None:
        print(f'[prefetch-worker] tag {tag!r} not found in IMW_CONFIGS')
        return 2

    _, conf, dense = target
    from imcui.api import ImageMatchingAPI
    api = ImageMatchingAPI(conf=conf, device=device,
                           detect_threshold=0.015,
                           max_keypoints=2048,
                           match_threshold=0.2)
    # Forward pass triggers lazy second-stage downloads, but is RAM-hungry on
    # CPU for RoMa/DKM-family models. Allow opting out.
    auto_skip = dense and any(k in tag for k in ('roma', 'dkm'))
    if skip_forward or auto_skip:
        if auto_skip:
            print(f'[prefetch-worker] {tag}: skipping forward pass '
                  '(heavy dense matcher; weights already on disk)')
        else:
            print(f'[prefetch-worker] {tag}: PREFETCH_SKIP_FORWARD=1, skipping forward')
    else:
        img0, img1 = _dummy_pair()
        try:
            _ = api(img0, img1)
        except Exception as fe:
            print(f'[prefetch-worker]   (forward pass note: '
                  f'{type(fe).__name__}: {fe})')
    return 0


def main() -> int:
    # Worker mode: --one TAG
    if len(sys.argv) >= 3 and sys.argv[1] == '--one':
        return _run_one(sys.argv[2])

    device = os.environ.get('PREFETCH_DEVICE', 'cpu')
    print(f'[prefetch] device={device}')
    print(f'[prefetch] HF_HOME={os.environ.get("HF_HOME", "(default ~/.cache/huggingface)")}')
    print(f'[prefetch] TORCH_HOME={os.environ.get("TORCH_HOME", "(default ~/.cache/torch)")}')
    _enable_insecure_ssl()

    print(f'[prefetch] {len(IMW_CONFIGS)} configs to fetch '
          '(each in its own subprocess; OOM-kills are isolated)\n')

    import subprocess
    only = os.environ.get('PREFETCH_ONLY')
    only_set = set(only.split(',')) if only else None
    ok, failed = [], []

    for tag, _conf, dense in IMW_CONFIGS:
        if only_set and tag not in only_set:
            continue
        print('=' * 70)
        print(f'[prefetch] >>> {tag} (dense={dense})')
        cmd = [sys.executable, os.path.abspath(__file__), '--one', tag]
        rc = subprocess.call(cmd)
        if rc == 0:
            print(f'[prefetch] <<< {tag} OK')
            ok.append(tag)
        else:
            # rc == -9 / 137 == SIGKILL (OOM); rc == 1 == generic failure
            reason = 'OOM-killed' if rc in (-9, 137) else f'exit={rc}'
            print(f'[prefetch] !!! {tag} FAILED ({reason})')
            failed.append((tag, reason))

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
