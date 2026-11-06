"""
GPU-service entry for gcp_train (the same calling convention as automatch's):

    python gcp_train/gpuaas.py "." <output folder> <settings.json>[,<more.json>]

The JSON files merge in order; "mode" picks the step:

  pairs   {"mode": "pairs", "manifest": "/maintenance/.../scenes.csv",
           "crop": 512, "per_scene": 40, "max_loo_px": 1.5}
          pixel sizes come from each scene's images (pairs.py); "res_m": 45
          fixes them, "res_m": [45, 315] limits them
          -> <output>/pairs/ (pairs.csv, scenes_fit.csv, splits.csv, images)
  train   {"mode": "train", "index": "/maintenance/.../pairs/pairs.csv",
           "config": "aliked" | "disk" | "/path/to/config.yaml",
           "experiment": "aliked_lg_all", "epochs": 20, "batch_size": 16,
           "sensors": ["MicroSat"], "modalities": ["lwir"], "set": ["train.lr=5e-6"]}
          "smoke": true runs one short epoch on 64 pairs first (minutes): it
          proves Glue Factory, the weights and the data before a long run.
          The matcher is told each view's modality and the pixel size
          (gf_cond_lightglue); "condition": false trains plain LightGlue.
          "train_extractor": true also fine-tunes the keypoint network
          (thermal, SAR: what it looks at, not only how it matches).
          -> Glue Factory training (gluefactory must be installed in the job's
             environment); checkpoint_best.tar and <experiment>_lightglue.pth
             (+ _condition.pth, + _extractor.pth) copied to <output>
  export  {"mode": "export", "checkpoint": ".../checkpoint_best.tar", "name": "aliked_lg_all"}
  eval    {"mode": "eval", "index": ".../pairs.csv", "config": "aliked",
           "checkpoint": ".../checkpoint_best.tar" (empty: official weights)}
          -> <output>/EVAL_<label>.csv per sensor, modality pair and pixel-size band

Offline servers: Glue Factory downloads the ALIKED / DISK and official
LightGlue weights through torch.hub; put them in the torch hub cache
($TORCH_HOME/hub/checkpoints: aliked-n16.pth, aliked_lightglue_v0-1_arxiv.pth,
disk_lightglue_v0-1_arxiv.pth, ...) or give a local file as model.matcher.weights.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CONFIGS = {'aliked': os.path.join(HERE, 'configs', 'aliked_lightglue_gcp.yaml'),
           'disk': os.path.join(HERE, 'configs', 'disk_lightglue_gcp.yaml')}


def parse(argv: List[str]):
    if len(argv) >= 4:
        out_dir, raw = argv[2], argv[3]
    elif len(argv) == 3:
        out_dir, raw = argv[1], argv[2]
    else:
        raise SystemExit(__doc__)
    job: Dict = {}
    for t in [x.strip() for x in raw.split(',') if x.strip()]:
        with open(t, encoding='utf-8-sig') as f:
            job.update(json.load(f))
    return out_dir, job


def run_train(job: Dict, out_dir: str) -> int:
    try:
        import gluefactory  # noqa: F401
        from gluefactory import settings
    except Exception as e:
        print(f'[gcp_train] Glue Factory is not installed in this environment ({type(e).__name__}: {e}). '
              f'Install it once: git clone https://github.com/cvg/glue-factory && pip install -e glue-factory')
        return 2
    conf = CONFIGS.get(str(job.get('config', 'aliked')), job.get('config'))
    exp = str(job.get('experiment') or 'gcp_finetune')
    args = [sys.executable, '-m', 'gluefactory.train', exp, '--conf', conf, f"data.index={job['index']}"]
    if job.get('smoke'):          # one short epoch on a few pairs: proves the set-up before a long run
        args += ['train.epochs=1', 'data.max_pairs=64', 'train.eval_every_iter=20', 'train.log_every_iter=5',
                 'data.train_batch_size=4', 'data.val_batch_size=4', 'data.num_workers=2']
    elif job.get('epochs'):
        args.append(f"train.epochs={int(job['epochs'])}")
    if job.get('batch_size'):
        b = int(job['batch_size'])
        args += [f'data.train_batch_size={b}', f'data.val_batch_size={b}']
    if job.get('sensors'):
        args.append('data.sensors=' + json.dumps([str(s) for s in job['sensors']]))   # YAML reads JSON lists
    if job.get('modalities'):
        args.append('data.modalities=' + json.dumps([str(s) for s in job['modalities']]))
    if job.get('condition') is False:
        args.append('model.matcher.name=matchers.lightglue')
    if job.get('train_extractor'):
        args += ['model.extractor.trainable=True', 'model.extractor.freeze_batch_normalization=True']
    args += [str(s) for s in (job.get('set') or [])]
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get('PYTHONPATH', ''))
    print('[gcp_train] ' + ' '.join(args), flush=True)
    rc = subprocess.call(args, env=env)
    exp_dir = os.path.join(str(settings.TRAINING_PATH), exp)
    best = os.path.join(exp_dir, 'checkpoint_best.tar')
    if os.path.exists(best):
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(best, os.path.join(out_dir, 'checkpoint_best.tar'))
        import export as X
        X.export(best, os.path.join(out_dir, f'{exp}_lightglue.pth'))
    for n in ('config.yaml', 'log.txt'):
        if os.path.exists(os.path.join(exp_dir, n)):
            shutil.copyfile(os.path.join(exp_dir, n), os.path.join(out_dir, n))
    return rc


def main(argv: List[str] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    out_dir, job = parse(argv)
    os.makedirs(out_dir, exist_ok=True)
    mode = str(job.get('mode', 'pairs')).lower()
    print(f'[gcp_train] mode {mode} -> {out_dir}', flush=True)
    if mode == 'pairs':
        import pairs as P
        r = P.build(job['manifest'], os.path.join(out_dir, 'pairs'), crop=int(job.get('crop', 512)),
                    per_scene=int(job.get('per_scene', 40)), shift_frac=float(job.get('shift', 0.25)),
                    min_valid=float(job.get('min_valid', 0.6)), max_cloud=float(job.get('max_cloud', 0.5)),
                    max_loo_px=float(job.get('max_loo_px', 1.5)), val=float(job.get('val', 0.1)),
                    test=float(job.get('test', 0.2)), seed=int(job.get('seed', 0)),
                    res_m=job.get('res_m', job.get('work_res_m')), max_scale=float(job.get('max_scale', 16)))
        return 0 if r['pairs'] else 1
    if mode == 'train':
        return run_train(job, out_dir)
    if mode == 'export':
        import export as X
        X.export(job['checkpoint'], os.path.join(out_dir, f"{job.get('name', 'finetuned')}_lightglue.pth"))
        return 0
    if mode == 'eval':
        import evaluate as V
        conf = CONFIGS.get(str(job.get('config', 'aliked')), job.get('config'))
        a = ['--conf', conf, '--index', job['index'], '--out', out_dir, '--split', job.get('split', 'test')]
        if job.get('checkpoint'):
            a += ['--checkpoint', job['checkpoint']]
        if job.get('label'):
            a += ['--label', job['label']]
        if job.get('condition') is False:
            a += ['--set', 'model.matcher.name=matchers.lightglue']
        return V.main(a)
    print(f'[gcp_train] unknown mode {mode!r}: pairs, train, export or eval')
    return 2


if __name__ == '__main__':
    sys.exit(main())
