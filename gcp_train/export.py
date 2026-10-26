"""
A fine-tuned Glue Factory checkpoint as LightGlue weights other tools load.

Glue Factory saves the whole two-view model ('model': extractor.* and
matcher.* weights). The matcher's weights, with the 'matcher.' prefix
removed, are LightGlue weights in the layout the official LightGlue, kornia's
LightGlue and Glue Factory itself read (transformers.<i>.self_attn ...).
The conditioned matcher's own part (condition.*, see condition.py) goes to
<name>_lightglue_condition.pth beside them: a tool applies it to the
descriptors before LightGlue. A trained extractor (extractor.*) goes to
<name>_extractor.pth.

    python export.py outputs/training/<experiment>/checkpoint_best.tar --out <name>_lightglue.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def matcher_state(checkpoint: Dict) -> Dict:
    state = checkpoint.get('model', checkpoint)
    out = {k[len('matcher.'):]: v for k, v in state.items() if k.startswith('matcher.')}
    if not out:
        raise ValueError('no matcher.* weights in this checkpoint (is it a Glue Factory two-view checkpoint?)')
    return out


def export(cp_path: str, out_path: str) -> Dict:
    import torch
    from modality import MODALITIES
    cp = torch.load(cp_path, map_location='cpu', weights_only=False)     # your own training output
    sd = matcher_state(cp)
    cond = {k[len('condition.'):]: v for k, v in sd.items() if k.startswith('condition.')}
    sd = {k: v for k, v in sd.items() if not k.startswith('condition.')}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    torch.save(sd, out_path)
    mconf = ((cp.get('conf') or {}).get('model') or {})
    meta = {'from': os.path.abspath(cp_path), 'epoch': cp.get('epoch'), 'eval': cp.get('eval'),
            'matcher_conf': mconf.get('matcher'), 'tensors': len(sd)}
    if cond:
        torch.save(cond, stem + '_condition.pth')
        meta['condition'] = {'file': os.path.basename(stem + '_condition.pth'), 'modalities': list(MODALITIES),
                             'apply': "before LightGlue, per view: d' = d * (1 + g) + b, (g, b) = "
                                      "modality.weight[m] + resolution(log2(px_m / 30) / 4) (condition.py)"}
    state = cp.get('model', cp)
    ext = {k[len('extractor.'):]: v for k, v in state.items() if k.startswith('extractor.')}
    if ext and ((mconf.get('extractor') or {}).get('trainable')):
        name = os.path.basename(stem)
        name = name[:-len('_lightglue')] if name.endswith('_lightglue') else name
        p = os.path.join(os.path.dirname(stem), name + '_extractor.pth')
        torch.save(ext, p)
        meta['extractor'] = {'file': os.path.basename(p), 'conf': mconf.get('extractor')}
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, default=str)
    print(f'[Export] {len(sd)} tensors -> {out_path}' + (f' (+ condition, {len(cond)} tensors)' if cond else '')
          + (f" (+ extractor -> {meta['extractor']['file']})" if 'extractor' in meta else ''))
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
