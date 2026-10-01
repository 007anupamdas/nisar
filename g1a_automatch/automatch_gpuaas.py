#!/usr/bin/env python3
"""
automatch_gpuaas -- entry point for the GPU service (gpuaas .../submit_job).

The service runs the script given as "code_path" with the same arguments as
the NISAR scripts:  sys.argv[2] = the output folder it assigns,
sys.argv[3] = the request's "input_path" (comma-separated). Everything after
that is the normal automatch job (automatch_job.run_job): the same logic and
the same outputs as the GUI and the command line.

"input_path" (paths on the server, comma-separated, no spaces needed):

    <input image>,<reference folder>[,<truth.csv>][,<settings.json>]
        the G1A raster (or NISAR scene), the L8_ref / C1 folder, optionally
        the manual RIVAL points, optionally a job file with the settings
        (detectors, matchers, window sizes, RANSAC, weights folder, ...)
    <job.json>
        one job file that also holds input_path / reference_dir / truth_csv

A job file may hold "mode": "run" (default), "weights" (check every model
weight file on the GPU node, nothing downloaded) or "preflight" (validate,
no matching); a token run / weights / preflight in input_path does the same.
The output folder of the request replaces the job file's output_dir.

Outputs (in the output folder): RUN_MANIFEST.csv, PERFORMANCE.csv (speed and
accuracy of every detector + matcher), TRUTH_*.csv (with a truth file),
rival/*.csv, automatch.log; WEIGHTS_REPORT.txt / PREFLIGHT.json in those modes.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODES = ('run', 'preflight', 'weights')


def _value(text: str):
    """Value of a key=value token: a|b|c is a list, JSON where it parses."""
    def one(v):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return [one(v) for v in text.split('|')] if '|' in text else one(text)


def build_job(argv: List[str]) -> Tuple[str, Dict]:
    """(mode, normalized job) from the service's arguments."""
    import automatch_job as J
    if len(argv) >= 4:
        out_dir, raw = argv[2], argv[3]
    elif len(argv) == 3:                       # <script> <output> <input_path>
        out_dir, raw = argv[1], argv[2]
    elif len(argv) == 2:                       # <script> <input_path>
        out_dir, raw = '', argv[1]
    else:
        raise SystemExit(__doc__)
    tokens = [t.strip() for t in raw.split(',') if t.strip()]
    job: Dict = {}
    mode = 'run'
    paths, overrides = [], {}
    for t in tokens:
        if t.lower() in MODES:
            mode = t.lower()
        elif t.lower().endswith('.json'):
            with open(t, encoding='utf-8-sig') as f:
                d = json.load(f)
            mode = str(d.pop('mode', mode)).lower()
            job.update(d)
        elif '=' in t and not os.path.exists(t):
            k, v = t.split('=', 1)
            overrides[k.strip()] = _value(v.strip())
        else:
            paths.append(t)
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
    csvs = [p for p in paths if p.lower().endswith('.csv')]
    others = [p for p in paths if not p.lower().endswith('.csv')]
    if len(others) > 2 or len(csvs) > 1:
        raise ValueError(f'input_path: expected <input>,<reference folder>[,<truth.csv>][,<settings.json>], '
                         f'got {tokens}')
    if others:
        job['input_path'] = others[0]
    if len(others) > 1:
        job['reference_dir'] = others[1]
    if csvs:
        job['truth_csv'] = csvs[0]
    job.update(overrides)
    if out_dir:
        job['output_dir'] = out_dir
        if not job.get('temp_dir'):
            job['temp_dir'] = os.path.join(out_dir, '_cache')
    return mode, J.normalize(job)


def main(argv: List[str] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    print(f'[GPUaaS] arguments: {argv[1:]}', flush=True)
    os.environ.setdefault('AUTOMATCH_EVENTS', '0')    # no GUI progress lines in the service log
    mode, job = build_job(argv)
    if not job['output_dir']:
        print('[GPUaaS] no output folder: the service passes it as the 2nd argument, '
              'or set output_dir in the job file')
        return 2
    os.makedirs(job['output_dir'], exist_ok=True)
    with open(os.path.join(job['output_dir'], 'job_gpuaas.json'), 'w', encoding='utf-8') as f:
        json.dump({'mode': mode, **job}, f, indent=2)
    print(f"[GPUaaS] mode {mode}: input {job['input_path']}, reference {job['reference_dir']}, "
          f"truth {job['truth_csv'] or '-'}, output {job['output_dir']}", flush=True)

    import automatch_job as J
    if mode == 'weights':
        import automatch_weights as W
        rep = W.check(job=job, quiet=True)
        text = W.format_report(rep)
        print(text)
        with open(os.path.join(job['output_dir'], 'WEIGHTS_REPORT.txt'), 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        with open(os.path.join(job['output_dir'], 'WEIGHTS_REPORT.json'), 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, default=str)
        return 1 if rep['summary']['missing'] else 0
    if mode == 'preflight':
        res = J.preflight(job)
        print(json.dumps(res, indent=2, default=str))
        with open(os.path.join(job['output_dir'], 'PREFLIGHT.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, default=str)
        return 1 if res['errors'] else 0
    try:
        J.run_job(job)
    except SystemExit as e:
        return int(e.code or 1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
