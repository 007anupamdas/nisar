"""
A fine-tuned Glue Factory checkpoint as LightGlue weights other tools load.

Glue Factory saves the whole two-view model ('model': extractor.* and
matcher.* weights). The matcher's weights, with the 'matcher.' prefix
removed, are LightGlue weights in the layout the official LightGlue, kornia's
LightGlue and Glue Factory itself read (transformers.<i>.self_attn ...).

    python export.py outputs/training/<experiment>/checkpoint_best.tar --out <name>_lightglue.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def matcher_state(checkpoint: Dict) -> Dict:
    state = checkpoint.get('model', checkpoint)
    out = {k[len('matcher.'):]: v for k, v in state.items() if k.startswith('matcher.')}
    if not out:
        raise ValueError('no matcher.* weights in this checkpoint (is it a Glue Factory two-view checkpoint?)')
    return out


def export(cp_path: str, out_path: str) -> Dict:
    import torch
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)     # your own training output
    sd = matcher_state(cp)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(sd, out_path)
    meta = {'from': os.path.abspath(cp_path), 'epoch': cp.get('epoch'), 'eval': cp.get('eval'),
            'matcher_conf': ((cp.get('conf') or {}).get('model') or {}).get('matcher'), 'tensors': len(sd)}
    with open(os.path.splitext(out_path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, default=str)
    print(f'[Export] {len(sd)} tensors -> {out_path}')
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Glue Factory checkpoint -> LightGlue weights')
    ap.add_argument('checkpoint')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    export(a.checkpoint, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
