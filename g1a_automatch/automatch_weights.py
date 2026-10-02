#!/usr/bin/env python3
"""
automatch_weights -- which model weight files each detector needs, and whether
they are on this machine.

    python automatch_weights.py                         # every detector, every weight choice
    python automatch_weights.py --detectors dedode,dog  # just these
    python automatch_weights.py --job job.json          # exactly what that job will load
    python automatch_weights.py --load                  # also load each model from the files
    python automatch_weights.py --json report.json      # machine-readable report

The same check runs as `python automatch_job.py weights ...`, behind the GUI's
"Check weights" button, and in preflight for the selected detectors.

How it works: every detector variant builds its networks on the CPU while each
weight request is intercepted -- torch-hub checkpoints (kornia, some imcui
models) and HuggingFace files (imcui). A request is looked up exactly where a
run would look (kornia's ~/.cache/torch/hub/checkpoints, TORCH_HOME/hub, the
Weights folder, the HuggingFace caches) and answered with an empty stand-in:
nothing is downloaded and no weight file is read, so the whole catalogue takes
seconds. With --load the files found are really loaded, which proves they are
complete and belong to the model, and a missing one fails at once instead of
being downloaded.
"""

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

STUB_PREFIX = '<not-on-disk>'
_INIT_FUNCS = ('uniform_', 'normal_', 'trunc_normal_', 'constant_', 'ones_', 'zeros_', 'eye_',
               'dirac_', 'xavier_uniform_', 'xavier_normal_', 'kaiming_uniform_', 'kaiming_normal_',
               'orthogonal_', 'sparse_')


class _Stub(dict):
    """Empty stand-in for a checkpoint; any key gives another stand-in."""

    def __missing__(self, key):
        return _Stub()


class TraceStop(RuntimeError):
    """A model fetched weights in a way the quick check cannot follow."""


def file_status(path: Optional[str]) -> str:
    """ok | MISSING | EMPTY | TRUNCATED (a zip-format checkpoint that does not
    open as a zip, i.e. an interrupted download) | UNREADABLE."""
    if not path or not os.path.isfile(path):
        return 'MISSING'
    try:
        if os.path.getsize(path) == 0:
            return 'EMPTY'
        with open(path, 'rb') as f:
            head = f.read(4)
        if head == b'PK\x03\x04' and not zipfile.is_zipfile(path):
            return 'TRUNCATED'
    except OSError:
        return 'UNREADABLE'
    return 'ok'


def _hub_repo_dir(repo: str) -> Optional[str]:
    """Local copy of a torch.hub GitHub repo ('owner/name[:ref]'), if any."""
    import automatch_engine as E
    owner, _, rest = repo.partition('/')
    name, _, ref = rest.partition(':')
    for hub in E.weight_search_dirs():
        for r in ([ref] if ref else ['main', 'master']):
            d = os.path.join(hub, f'{owner}_{name}_{r}')
            if os.path.isdir(d):
                return d
    return None


class Tracer:
    """Records every weight request while active(); see the module doc."""

    def __init__(self, load: bool = False):
        self.load = load
        self.requests: List[Dict] = []
        self._served: set = set()

    def hook(self, kind: str, url: str = '', file: str = '', path: Optional[str] = None,
             searched: Optional[List[str]] = None, **extra):
        """Called by automatch_engine (torch hub) and automatch_imcui
        (HuggingFace, with repo / repo_file / repo_type / revision in extra)
        for each request. Returns the answer, or None to let the normal
        loader read the file."""
        import automatch_engine as E
        rec = {'kind': kind, 'model': E.MODEL_LABELS[-1] if E.MODEL_LABELS else '',
               'file': file, 'source': url, 'path': path, 'searched': list(searched or []),
               'status': file_status(path), **extra}
        self.requests.append(rec)
        if self.load:
            if rec['status'] != 'ok':
                raise FileNotFoundError(f"{file}: {rec['status'].lower()} "
                                        f"(looked in {', '.join(searched or []) or 'nowhere'})")
            return None
        if kind == 'hf':
            p = path if rec['status'] == 'ok' else f'{STUB_PREFIX}/{file}'
            self._served.add(p)
            return p
        return _Stub()

    @contextlib.contextmanager
    def active(self):
        import torch as th
        import automatch_engine as E
        saved_hook, saved_failed = E.WEIGHTS_TRACE_HOOK, dict(E.FAILED_MODELS)
        patches: List[Tuple[object, str, object]] = []

        def patch(obj, name, value):
            if hasattr(obj, name):
                patches.append((obj, name, getattr(obj, name)))
                setattr(obj, name, value)

        orig_hub_load = th.hub.load

        def hub_load(repo_or_dir, model, *args, **kwargs):
            local = kwargs.get('source') == 'local'
            path = repo_or_dir if local and os.path.isdir(repo_or_dir) else (
                None if local else _hub_repo_dir(repo_or_dir))
            self.requests.append({'kind': 'torch-hub-repo',
                                  'model': E.MODEL_LABELS[-1] if E.MODEL_LABELS else '',
                                  'file': f'{repo_or_dir}:{model}', 'source': f'torch.hub {repo_or_dir}',
                                  'path': path, 'searched': E.weight_search_dirs(),
                                  'status': 'ok' if path else 'MISSING'})
            if self.load:
                return orig_hub_load(repo_or_dir, model, *args, **kwargs)
            raise TraceStop(f'loads code and weights with torch.hub.load({repo_or_dir!r}, {model!r})')

        E.WEIGHTS_TRACE_HOOK = self.hook
        patch(th.hub, 'load', hub_load)
        if not self.load:
            from torch.nn.modules.module import _IncompatibleKeys
            orig_torch_load = th.load

            def torch_load(f, *args, **kwargs):
                if isinstance(f, (str, os.PathLike)):
                    p = os.fspath(f)
                    if p in self._served or p.startswith(STUB_PREFIX):
                        return _Stub()
                return orig_torch_load(f, *args, **kwargs)

            patch(th, 'load', torch_load)
            patch(th.nn.Module, 'load_state_dict',
                  lambda module, state_dict, *a, **k: _IncompatibleKeys([], []))
            for name in _INIT_FUNCS:     # random init is wasted work here
                patch(th.nn.init, name, lambda tensor, *a, **k: tensor)
        try:
            yield self
        finally:
            for obj, name, value in reversed(patches):
                setattr(obj, name, value)
            E.WEIGHTS_TRACE_HOOK = saved_hook
            E.FAILED_MODELS.clear()
            E.FAILED_MODELS.update(saved_failed)


def check_variant(m, tracer: Tracer) -> Dict:
    """Build one variant's networks under the tracer."""
    import automatch_engine as E
    n0 = len(tracer.requests)
    status, error = 'ok', ''
    m._noted_missing = True       # no "lgm not available for this variant" notes
    try:
        with tracer.active():
            m.load_models()
    except Exception as e:
        cause = e.__cause__ if isinstance(e, E.ModelLoadError) and e.__cause__ is not None else e
        if isinstance(cause, TraceStop):
            status, error = 'partial', str(cause)
        else:
            status, error = 'error', f'{type(cause).__name__}: {cause}'
    finally:
        try:
            m.unload_model()
        except Exception:
            pass
        gc.collect()
    reqs = tracer.requests[n0:]
    if any(r['status'] != 'ok' for r in reqs) and status in ('ok', 'error'):
        # a missing file is the finding; any error after it came from the stand-in
        status, error = 'missing', ('' if not tracer.load else error)
    elif status == 'error' and reqs and not tracer.load:
        # every file was found; the model then read the stand-in itself (e.g.
        # imcui's r2d2 takes its network definition from the checkpoint)
        status, error = 'partial', (f'its weight files are found, but the quick check cannot build it '
                                    f'from stand-ins ({error}); the run loads them for real, or use --load')
    elif status == 'ok' and not reqs:
        status = 'no weights'
    return {'variant': m.get_filename_prefix(), 'status': status, 'error': error, 'files': reqs}


def _weight_axes(name: str) -> Dict[str, List]:
    """Every value of each choice / yes-no detector parameter: weights can
    depend on either (DoG descriptor, AffNet on/off, KeyNet upright -> OriNet)."""
    import automatch_engine as E
    axes: Dict[str, List] = {}
    for sp in E.detector_param_specs(name):
        if sp.scope != 'detector':
            continue
        if sp.kind == 'choice':
            axes[sp.name] = list(sp.choices or [sp.default])
        elif sp.kind == 'bool':
            axes[sp.name] = [sp.default, not sp.default]
    return axes


def plan_units(job: Optional[Dict] = None, detectors: Optional[List[str]] = None
               ) -> List[Tuple[str, Dict, Optional[Dict]]]:
    """[(detector, parameter values, detector_matchers)] to check: what the
    job runs, or every weight choice of the named (default: all) detectors."""
    import automatch_engine as E
    if job is not None:
        return [(d, (job.get('detector_params') or {}).get(d) or {}, job.get('matchers') or {})
                for d in job['detectors']]
    catalog = {d['name']: d for d in E.available_detectors()}
    names = detectors or list(catalog)
    units = []
    for d in names:
        base, _ = E.resolve_detector_name(d)
        if base not in catalog:
            raise ValueError(f'unknown detector {d!r}; available: {sorted(catalog)}')
        axes = _weight_axes(base) if catalog[base]['source'] == 'kornia' else {}
        units.append((d, axes, {base: list(catalog[base]['matchers'])}))
    return units


def _put_in() -> str:
    import torch as th
    import automatch_engine as E
    return os.path.join(E.WEIGHT_DIRS[0] if E.WEIGHT_DIRS else th.hub.get_dir(), 'checkpoints')


def summarize(detectors: List[Dict], load: bool) -> Dict:
    import automatch_engine as E
    missing: Dict[Tuple[str, str], Dict] = {}
    n_files = n_ok = 0
    for entry in detectors:
        files: Dict[Tuple[str, str, str], Dict] = {}
        for v in entry['variants']:
            for r in v['files']:
                key = (r['kind'], r['source'], r['file'])
                f = files.setdefault(key, {k: r[k] for k in ('kind', 'source', 'file', 'path', 'status', 'model')
                                           + tuple(x for x in ('repo', 'repo_file', 'repo_type', 'revision')
                                                   if x in r)})
                f.setdefault('needed_by', [])
                if v['variant'] not in f['needed_by']:
                    f['needed_by'].append(v['variant'])
        for f in files.values():
            f['size_mb'] = round(os.path.getsize(f['path']) / 1e6, 1) if f['status'] == 'ok' else None
            f['folder'] = os.path.dirname(f['path']) if f['status'] == 'ok' else ''
            f['all_variants'] = len(f['needed_by']) == len(entry['variants'])
            n_files += 1
            n_ok += f['status'] == 'ok'
            if f['status'] != 'ok':
                m = missing.setdefault((f['source'], f['file']), {
                    'file': f['file'], 'source': f['source'], 'kind': f['kind'], 'status': f['status'],
                    'detectors': []})
                if entry['detector'] not in m['detectors']:
                    m['detectors'].append(entry['detector'])
        entry['files'] = sorted(files.values(), key=lambda f: (f['status'] == 'ok', f['model'], f['file']))
        st = {v['status'] for v in entry['variants']}
        entry['status'] = ('error' if entry.get('error') or 'error' in st else 'missing' if 'missing' in st
                           else 'partial' if 'partial' in st else 'ok' if 'ok' in st else 'no weights')
    try:
        import automatch_imcui as IM
        hf = IM.hf_cache_dirs()
    except Exception:
        hf = []
    counts: Dict[str, int] = {}
    for e in detectors:
        counts[e['status']] = counts.get(e['status'], 0) + 1
    return {
        'mode': 'load' if load else 'check',
        'torch_folders': [os.path.join(d, 'checkpoints') for d in E.weight_search_dirs()],
        'hf_caches': hf,
        'put_in': _put_in(),
        'detectors': detectors,
        'missing': list(missing.values()),
        'summary': {'detectors': len(detectors), 'files': n_files, 'found': n_ok,
                    'missing': n_files - n_ok, 'by_status': counts},
    }


def check(job: Optional[Dict] = None, detectors: Optional[List[str]] = None, load: bool = False,
          weights_cache: str = '', imcui: bool = True, quiet: bool = False,
          progress: Optional[Callable[[int, int, str], None]] = None) -> Dict:
    """Run the check; returns the report (see summarize)."""
    def hushed():
        return contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()

    with hushed():
        import automatch_engine as E
        import automatch_imcui as IM
        if job is not None:
            weights_cache = job.get('weights_cache_dir') or weights_cache
        if imcui:
            IM.register_imcui_detectors(weights_cache or None)
        else:
            IM.configure_weights_cache(weights_cache or None)
        units = plan_units(job, detectors)
    prev = E.device
    E.device = E.th.device('cpu')    # built on the CPU; nothing runs
    tracer = Tracer(load)
    out: List[Dict] = []
    try:
        for i, (det, params, matchers) in enumerate(units):
            cfg = E.PipelineConfig(num_features=256, use_amp=False, smnn_thresholds=[0.9],
                                   detector_matchers=matchers or {})
            entry = {'detector': det, 'variants': [], 'error': ''}
            with hushed():
                try:
                    variants = list(E.iter_variants(det, cfg, params))
                except Exception as e:
                    entry['error'] = f'{type(e).__name__}: {e}'
                    variants = []
                for m in variants:
                    entry['variants'].append(check_variant(m, tracer))
            out.append(entry)
            if progress:
                progress(i + 1, len(units), det)
    finally:
        E.device = prev
    return summarize(out, load)


def format_report(rep: Dict) -> str:
    lines = ['Weights folders searched (torch hub):']
    lines += [f'  {d}' + ('' if os.path.isdir(d) else '   (does not exist)') for d in rep['torch_folders']]
    if rep.get('hf_caches'):
        lines.append('HuggingFace caches searched (imcui):')
        lines += [f'  {d}' + ('' if os.path.isdir(d) else '   (does not exist)') for d in rep['hf_caches']]
    for e in rep['detectors']:
        nv = len(e['variants'])
        lines.append(f"\n{e['detector']}  [{e['status']}]  {nv} variant(s) checked")
        if e.get('error'):
            lines.append(f"  ERROR {e['error']}")
        for f in e['files']:
            where = (f"{f['size_mb']:>7.1f} MB  {f['folder']}" if f['status'] == 'ok'
                     else f"<- {f['source']}")
            need = '' if f['all_variants'] else (
                '   [' + ', '.join(f['needed_by'][:3]) + (f' +{len(f["needed_by"]) - 3}' if len(f['needed_by']) > 3
                                                        else '') + ']')
            lines.append(f"  {f['status']:<9} {f['file']:<38} {where}{need}")
        for v in e['variants']:
            if v['status'] in ('error', 'partial') or (v['error'] and rep['mode'] == 'load'):
                lines.append(f"  {v['variant']}: {v['status']} -- {v['error']}")
        if not e['files'] and not e.get('error') and all(v['status'] == 'no weights' for v in e['variants']):
            lines.append('  no weight files (nothing to download)')
    s = rep['summary']
    lines.append(f"\n{s['files']} weight file(s) for {s['detectors']} detector(s): {s['found']} found, "
                 f"{s['missing']} missing or damaged ({rep['mode']} mode)")
    if any(v['status'] == 'partial' for e in rep['detectors'] for v in e['variants']):
        lines.append("'partial': the quick check cannot follow how that model loads its weights; "
                     "run with --load to try a real offline load.")
    if rep['missing']:
        lines.append(f"\nMissing -- download each file and save it UNDER THE NAME SHOWN in\n  {rep['put_in']}")
        for m in rep['missing']:
            if m['kind'] == 'torch-hub':
                lines.append(f"  {m['file']:<38} <- {m['source']}   ({', '.join(m['detectors'])})")
            else:
                lines.append(f"  {m['file']:<38} <- {m['source']}   ({', '.join(m['detectors'])}; "
                             f"HuggingFace/torch.hub file: use prefetch_weights.py)")
        lines.append('or, on a connected machine: python prefetch_weights.py <folder> '
                     '[--only <detectors>], copy <folder> here and set it as the Weights folder.')
    return '\n'.join(lines)


def compact(rep: Dict) -> Dict:
    """The report without per-request search lists (for progress events)."""
    out = dict(rep)
    out['detectors'] = [{**e, 'variants': [{k: v for k, v in var.items() if k != 'files'}
                                           for var in e['variants']]} for e in rep['detectors']]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Check which model weights each detector needs and '
                                             'whether they are on this machine.')
    ap.add_argument('--job', default='', help='check what this job file will load')
    ap.add_argument('--detectors', default='', help='comma list (default: every detector)')
    ap.add_argument('--load', action='store_true',
                    help='also load each model from the files found (slower; proves they load)')
    ap.add_argument('--weights-cache', default='', help='extra weights folder (the GUI Weights folder)')
    ap.add_argument('--no-imcui', action='store_true', help='kornia detectors only')
    ap.add_argument('--json', default='', help='also write the report as JSON to this file')
    ap.add_argument('--verbose', action='store_true', help="show the models' own loading messages")
    a = ap.parse_args(argv)

    import automatch_job as J
    job = J.load_job(a.job) if a.job else None
    dets = [d.strip() for d in a.detectors.split(',') if d.strip()] or None

    def progress(done, total, det):
        J.emit({'event': 'weights_progress', 'done': done, 'total': total, 'detector': det})

    rep = check(job=job, detectors=dets, load=a.load, weights_cache=a.weights_cache,
                imcui=not a.no_imcui, quiet=not a.verbose, progress=progress)
    print(format_report(rep))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, default=str)
        print(f'\nreport written to {a.json}')
    J.emit({'event': 'weights', **compact(rep)})
    bad = rep['summary']['missing'] or any(e['status'] in ('error',) for e in rep['detectors'])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
