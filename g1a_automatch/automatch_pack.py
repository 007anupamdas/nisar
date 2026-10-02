#!/usr/bin/env python3
"""
automatch_pack -- prepare one scene for a machine that cannot see your data
(the GPU server has no access to the reference collection).

On the workstation, from a job file (the GUI saves one with 'Save job…', and
every run leaves <output>/job_gui.json):

    python automatch_job.py pack <job.json> --to <folder the server can read>
        --server-dir <the same folder as the server sees it>
        [--server-weights DIR] [--server-cache DIR] [--set key=value ...] [--dry-run]

e.g.  --to V:\\ICIGDev\\GPUPOC\\input\\g1a\\set1
      --server-dir /maintenance/ICIGDev/GPUPOC/input/g1a/set1

It copies into --to:
    input/                  the image, with its sidecar / world / aux files
    reference/<name>/       only the reference rasters that can lie within
                            max_expected_error_m of the image, plus what the
                            collection's footprints come from (its index
                            shapefile, or each raster's sidecar), keeping
                            the folder layout -- not the whole collection
    truth/                  the ground-truth CSV (and manual GCP CSV), if set
    automatch_settings.json the job, every path rewritten for the server
    mode_env.json, mode_weights.json, mode_preflight.json
                            one-line files that pick the check to run: the GPU
                            service accepts only existing paths in input_path,
                            so the mode is given as a file, not a word
    submit_gpuaas.sh        curl requests: ./submit_gpuaas.sh env|weights|preflight|run
    PACK_REPORT.txt         what was copied, and the curl commands to paste
then reads the copied reference folder back the way a run will, to check that
every selected reference is found. It prints what it is doing while it copies;
files already there with the same size are not copied again, so an
interrupted pack can simply be started again.
"""

import argparse
import json
import os
import posixpath
import shutil
import sys
import time
from pathlib import PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GPUAAS_URL = 'http://gpuaas.private.nrsc.gov.in:8000/submit_job'
GPUAAS_CODE = '/maintenance/ICIGDev/GPUPOC/exe/g1a_automatch/automatch_gpuaas.py'
MODES = ('env', 'weights', 'preflight')


def log(msg: str, end: str = '\n'):
    print(f'[pack] {msg}', end=end, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# files that belong to a raster
# ─────────────────────────────────────────────────────────────────────────────
_FOLDERS: Dict[str, Dict] = {}


def _folder_index(folder: str) -> Dict:
    """Names in a folder, indexed once: a reference folder can hold a hundred
    thousand files on a network drive, so it is listed one time, not once per
    selected raster."""
    idx = _FOLDERS.get(folder)
    if idx is not None:
        return idx
    import automatch_refs as refs
    t0 = time.time()
    head: Dict[str, List[str]] = {}
    meta: Dict[str, List[str]] = {}
    n = 0
    with os.scandir(folder or '.') as it:
        for e in it:
            try:
                if not e.is_file():
                    continue
            except OSError:
                continue
            n += 1
            low = e.name.lower()
            head.setdefault(low.split('.', 1)[0], []).append(e.name)
            if refs.is_meta_file(e.name):
                meta.setdefault((refs.meta_base_stem(e.name) or '').lower(), []).append(e.name)
    idx = _FOLDERS[folder] = {'head': head, 'meta': meta}
    if time.time() - t0 > 3:
        log(f'listed {n} files in {folder} ({time.time() - t0:.0f} s, once)')
    return idx


def companions(path: str) -> List[str]:
    """Files beside a raster that belong to it: X.tif.aux.xml, X.tif.ovr,
    X.tfw, X.prj, X.aux.xml, and RIVAL sidecars (X.met, X_meta.txt, ...)."""
    import automatch_refs as refs
    folder, base = os.path.split(path)
    lb = base.lower()
    stem = os.path.splitext(lb)[0]
    idx = _folder_index(folder)
    cands = set(idx['head'].get(lb.split('.', 1)[0], [])) | set(idx['meta'].get(stem, []))
    out = []
    for name in sorted(cands):
        low = name.lower()
        if low == lb:
            continue
        if low.startswith(lb + '.'):
            out.append(os.path.join(folder, name))       # X.tif.aux.xml, X.tif.ovr
        elif low.startswith(stem + '.') and not low.endswith(refs.RASTER_EXTS):
            out.append(os.path.join(folder, name))       # X.tfw, X.prj, X.met, ...
        elif refs.is_meta_file(name) and (refs.meta_base_stem(name) or '').lower() == stem:
            out.append(os.path.join(folder, name))       # X_meta.txt, X.h5.iso.xml
    return out


# ─────────────────────────────────────────────────────────────────────────────
# which references
# ─────────────────────────────────────────────────────────────────────────────
def input_footprint(job: Dict):
    """(lon/lat footprint of the image, its source, working CRS) -- as preflight
    and the run get it."""
    import automatch_engine as E
    cfg = E.PipelineConfig(nisar_band=job['nisar_band'], nisar_frequency=job['nisar_frequency'])
    scene = E.InputScene(job['input_path'], cfg)
    ch = (job['channels'] or scene.channels)[0]
    if scene.kind == 'raster':
        path, band = scene.channel_raster(ch, '')
        foot, src = scene.footprint_lonlat(path, band)
    else:
        foot, src = E.NISARH5Reader.footprint_lonlat(scene.info)
    return foot, src, scene.working_crs


def select_references(job: Dict) -> Tuple[Dict, List[str], str]:
    """(reference catalogue, rasters whose declared footprint touches the image
    footprint grown by max_expected_error_m + search_margin_m, footprint source)."""
    import automatch_engine as E
    import automatch_refs as refs
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon
    from shapely.ops import transform
    mode = None if job['reference_mode'] in ('', 'auto', None) else job['reference_mode']
    log(f"reading the reference collection {job['reference_dir']} ...")
    cat = refs.scan_reference_folder(job['reference_dir'], mode)
    log(f"reading the image footprint of {os.path.basename(job['input_path'])} ...")
    foot, src, crs = input_footprint(job)
    grow = float(job['max_expected_error_m']) + float(job['search_margin_m'] or 0)
    to_m = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(crs), always_xy=True)
    to_ll = Transformer.from_crs(CRS.from_user_input(crs), CRS.from_epsg(4326), always_xy=True)
    grown = transform(to_m.transform, E._densify_ring(E._as_2d_polygon(foot))).buffer(grow)
    grown_ll = E._as_2d_polygon(transform(to_ll.transform, E._densify_ring(E._as_2d_polygon(grown))))
    hits = []
    for tif, rec in sorted(cat['footprints'].items()):
        ring = rec.get('ring') or []
        if len(ring) >= 3 and grown_ll.intersects(E._as_2d_polygon(Polygon(ring))):
            hits.append(tif)
    log(f"{len(hits)} of {len(cat['footprints'])} references lie within {grow / 1000:g} km of the image "
        f"(footprint from {src})")
    return cat, hits, src


# ─────────────────────────────────────────────────────────────────────────────
# copying
# ─────────────────────────────────────────────────────────────────────────────
def _size(path: str) -> int:
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path)


def _mb(n: float) -> str:
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.1f} MB'


def _run_plan(plan: List[Tuple[str, str, int]], dry: bool) -> Tuple[int, int]:
    """Copy each (src, dst, size), reporting as it goes. A file already at the
    destination with the same size is kept; copies go through '<dst>.part' so
    an interrupted copy never looks complete."""
    copied = skipped = 0
    total = sum(s for _, _, s in plan)
    done = 0
    t_all = time.time()
    for i, (src, dst, size) in enumerate(plan, 1):
        tag = f'({i}/{len(plan)}) {os.path.basename(src.rstrip(os.sep))} {_mb(size)}'
        if dry:
            log(f'{tag} -> {dst}')
            continue
        if os.path.isdir(src):
            if os.path.isdir(dst):
                skipped += 1
                log(f'{tag} already there')
            else:
                log(f'{tag} copying folder ...', end='')
                t0 = time.time()
                shutil.copytree(src, dst)
                copied += 1
                print(f' done ({time.time() - t0:.0f} s)', flush=True)
            done += size
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            skipped += 1
            done += size
            log(f'{tag} already there')
            continue
        log(f'{tag} copying ...', end='')
        t0 = time.time()
        shutil.copyfile(src, dst + '.part')
        try:
            shutil.copystat(src, dst + '.part')
        except OSError:
            pass
        os.replace(dst + '.part', dst)
        copied += 1
        done += size
        dt = max(time.time() - t0, 1e-3)
        print(f' done ({dt:.0f} s, {size / 1e6 / dt:.0f} MB/s; {_mb(done)} of {_mb(total)})', flush=True)
    if not dry:
        log(f'copying finished in {(time.time() - t_all) / 60:.1f} min: {copied} copied, {skipped} already there')
    return copied, skipped


# ─────────────────────────────────────────────────────────────────────────────
# server paths
# ─────────────────────────────────────────────────────────────────────────────
def local_for(server_path: str, dest: str, server_dir: str) -> Optional[str]:
    """The local path of a server path, from how --to and --server-dir name the
    same folder (V:\\X\\Y <-> /maintenance/X/Y gives V:\\ <-> /maintenance/).
    None when the server path lies outside that mapping."""
    if not server_path:
        return None
    lp = list((PureWindowsPath if os.name == 'nt' else PurePosixPath)(os.path.abspath(dest)).parts)
    sp = list(PurePosixPath(server_dir).parts)
    k = 0
    while k < min(len(lp), len(sp)) - 1 and lp[-1 - k].lower() == sp[-1 - k].lower():
        k += 1
    local_root, server_root = lp[:len(lp) - k], sp[:len(sp) - k]
    tp = list(PurePosixPath(server_path).parts)
    if [p.lower() for p in tp[:len(server_root)]] != [p.lower() for p in server_root]:
        return None
    rest = tp[len(server_root):]
    return os.path.join(*local_root, *rest) if local_root else server_path


def curl_command(input_path: str, code_path: str, url: str, conda_env: str, gpu_mb: int) -> str:
    body = json.dumps({'code_path': code_path, 'conda_env': conda_env, 'max_gpu_mem_required': gpu_mb,
                       'input_path': input_path}, indent=1)
    return (f'curl -X POST "{url}" -H "Content-Type: application/json" '
            f'-H "X-User-Name: $(whoami)" -d \'{body}\'')


def mode_inputs(server_dir: str) -> Dict[str, str]:
    """input_path for each mode: the settings file, plus a mode file for the
    checks (every item must be an existing path for the GPU service)."""
    settings = posixpath.join(server_dir, 'automatch_settings.json')
    out = {m: f"{settings},{posixpath.join(server_dir, f'mode_{m}.json')}" for m in MODES}
    out['run'] = settings
    return out


# ─────────────────────────────────────────────────────────────────────────────
def pack(job: Dict, dest: str, server_dir: str, server_weights: str = '', server_cache: str = '',
         dry_run: bool = False, code_path: str = GPUAAS_CODE, url: str = GPUAAS_URL,
         conda_env: str = 'mpad', gpu_mb: int = 40000) -> Dict:
    import automatch_refs as refs
    if not job['input_path'] or not os.path.exists(job['input_path']):
        raise FileNotFoundError(f"input_path not found here: {job['input_path']!r}")
    if not job['reference_dir'] or not os.path.isdir(job['reference_dir']):
        raise FileNotFoundError(f"reference_dir not found here: {job['reference_dir']!r}")
    if not server_dir.startswith('/'):
        raise ValueError(f'--server-dir must be the folder as the server sees it (/maintenance/...), '
                         f'got {server_dir!r}')
    dest = os.path.abspath(dest)
    plan: List[Tuple[str, str]] = []
    settings = dict(job)

    # input image (a NISAR scene folder is copied whole)
    inp = os.path.abspath(job['input_path'])
    name = os.path.basename(inp.rstrip(os.sep))
    plan.append((inp, os.path.join(dest, 'input', name)))
    if os.path.isfile(inp):
        plan += [(c, os.path.join(dest, 'input', os.path.basename(c))) for c in companions(inp)]
    settings['input_path'] = posixpath.join(server_dir, 'input', name)

    # references: only those the run can use, with their footprint source
    cat, hits, foot_src = select_references(job)
    ref_root = os.path.abspath(job['reference_dir'])
    ref_name = os.path.basename(ref_root.rstrip(os.sep)) or 'reference'
    ref_dest = os.path.join(dest, 'reference', ref_name)
    if hits:
        log('finding the files that go with each selected reference ...')
    metas = set()
    for tif in hits:
        plan.append((tif, os.path.join(ref_dest, os.path.relpath(tif, ref_root))))
        plan += [(c, os.path.join(ref_dest, os.path.relpath(c, ref_root))) for c in companions(tif)]
        meta = cat['footprints'][tif].get('meta')
        if meta:
            metas.add(meta)
    for meta in sorted(metas):
        plan.append((meta, os.path.join(ref_dest, os.path.relpath(meta, ref_root))))
        if cat['mode'] == refs.REF_MODE_INDEX:          # .shx / .dbf / .prj / .cpg of the index
            plan += [(c, os.path.join(ref_dest, os.path.relpath(c, ref_root))) for c in companions(meta)]
    settings['reference_dir'] = posixpath.join(server_dir, 'reference', ref_name)

    # ground truth / manual GCPs
    used = set()
    for key, prefix in (('truth_csv', ''), ('manual_gcp_csv', 'gcp_')):
        if job.get(key):
            if not os.path.exists(job[key]):
                raise FileNotFoundError(f'{key} not found here: {job[key]!r}')
            base = os.path.basename(job[key])
            if base in used:
                base = prefix + base
            used.add(base)
            plan.append((os.path.abspath(job[key]), os.path.join(dest, 'truth', base)))
            settings[key] = posixpath.join(server_dir, 'truth', base)

    settings['output_dir'] = posixpath.join(server_dir, 'output')   # the GPU service replaces it
    settings['temp_dir'] = server_cache or ''
    settings['weights_cache_dir'] = server_weights or ''
    inputs = mode_inputs(server_dir)

    # de-duplicate (a sidecar can be both a companion and the footprint source)
    seen, sized = set(), []
    for s, d in plan:
        if d not in seen:
            seen.add(d)
            sized.append((s, d, _size(s)))
    total = sum(x[2] for x in sized)
    ref_bytes = sum(x[2] for x in sized if x[1].startswith(ref_dest))
    log(f"{len(sized)} file(s), {_mb(total)} ({_mb(ref_bytes)} of references) "
        + ('-- dry run, nothing is copied:' if dry_run else f'-> {dest}'))

    # server paths that should already exist there, checked through the share
    warnings = []
    for label, sp in (('code_path', code_path), ('--server-weights', server_weights),
                      ('--server-cache', server_cache)):
        lp = local_for(sp, dest, server_dir)
        if sp and lp and not os.path.exists(lp):
            warnings.append(f'{label} {sp} is not there (looked at {lp} from here)'
                            + (' -- copy the g1a_automatch folder there' if label == 'code_path' else
                               ' -- check the path' if label == '--server-weights' else
                               ' -- it is created by the run if the server may write there'))
    for w in warnings:
        log(f'WARNING: {w}')

    copied, skipped = _run_plan(sized, dry_run)

    cmds = {m: curl_command(inputs[m], code_path, url, conda_env, gpu_mb) for m in MODES + ('run',)}
    report = [f"Packed for the server: {name}",
              f"  image footprint from: {foot_src}",
              f"  reference collection: {ref_root} ({cat['mode']}, {len(cat['footprints'])} footprints)",
              f"  references within {float(job['max_expected_error_m']) / 1000:g} km "
              f"(+{float(job['search_margin_m'] or 0) / 1000:g} km margin) of the image: {len(hits)}",
              *[f'    {os.path.relpath(t, ref_root)}' for t in hits],
              f"  {len(sized)} file(s), {_mb(total)}"
              + (' (dry run: nothing copied)' if dry_run else f' -- {copied} copied, {skipped} already there'),
              f"  local folder : {dest}",
              f"  server folder: {server_dir}",
              *[f'  WARNING: {w}' for w in warnings],
              '',
              'Submit in this order. input_path holds only existing files (the service checks each one):',
              'the settings file, plus a mode file for the three checks. Each job writes its report into',
              'the output folder the service gives it.']
    for m, why in (('env', 'what the GPU node has: packages, GPU, weight folders, detector list -> ENVIRONMENT.txt'),
                   ('weights', 'every weight file the job needs, found or missing -> WEIGHTS_REPORT.txt'),
                   ('preflight', 'inputs, references, settings, number of matching passes -> PREFLIGHT.json'),
                   ('run', 'the run -> PERFORMANCE.csv, TRUTH_BY_DETECTOR_MATCHER.csv, rival/, automatch.log')):
        report += ['', f'# {m}: {why}', cmds[m]]
    text = '\n'.join(report)

    check = None
    if not dry_run:
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, 'automatch_settings.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(settings, f, indent=2)
        for m in MODES:
            with open(os.path.join(dest, f'mode_{m}.json'), 'w', encoding='utf-8', newline='\n') as f:
                json.dump({'mode': m}, f)
        script = ['#!/bin/bash',
                  f'# automatch on the GPU service for {name}.  Usage: ./submit_gpuaas.sh env|weights|preflight|run',
                  f'URL="{url}"', f'CODE="{code_path}"', f'DIR="{server_dir}"',
                  'case "${1:-}" in',
                  '  env|weights|preflight) INPUT="$DIR/automatch_settings.json,$DIR/mode_$1.json" ;;',
                  '  run) INPUT="$DIR/automatch_settings.json" ;;',
                  '  *) echo "usage: $0 env|weights|preflight|run"; exit 1 ;;',
                  'esac',
                  'curl -X POST "$URL" -H "Content-Type: application/json" -H "X-User-Name: $(whoami)" \\',
                  f'  -d "{{\\"code_path\\": \\"$CODE\\", \\"conda_env\\": \\"{conda_env}\\", '
                  f'\\"max_gpu_mem_required\\": {gpu_mb}, \\"input_path\\": \\"$INPUT\\"}}"',
                  'echo']
        with open(os.path.join(dest, 'submit_gpuaas.sh'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(script) + '\n')
        # read the copy back the way a run will
        log('checking the copied reference folder ...')
        try:
            back = refs.scan_reference_folder(ref_dest, cat['mode'])
            found = {os.path.basename(p).lower() for p in back['footprints']}
            lost = [os.path.basename(t) for t in hits if os.path.basename(t).lower() not in found]
            check = {'mode': back['mode'], 'found': len(hits) - len(lost), 'lost': lost}
            text += (f"\n\nCheck: the copied reference folder reads as {back['mode']}; "
                     f"{check['found']} of {len(hits)} selected references found"
                     + (f"; NOT found: {', '.join(lost)}" if lost else ' -- OK'))
        except Exception as e:
            check = {'error': f'{type(e).__name__}: {e}'}
            text += f'\n\nCheck FAILED: the copied reference folder could not be read ({check["error"]})'
        with open(os.path.join(dest, 'PACK_REPORT.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(text + '\n')
    return {'text': text, 'references': hits, 'files': len(sized), 'bytes': total, 'copied': copied,
            'skipped': skipped, 'settings': settings, 'check': check, 'commands': cmds, 'inputs': inputs,
            'warnings': warnings}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Copy one scene, the references it needs and a settings file '
                                             'with server paths into a folder the GPU server can read.')
    ap.add_argument('job', help="job file (GUI 'Save job…', or <output>/job_gui.json)")
    ap.add_argument('--to', required=True, help='local folder the server can read, e.g. V:\\...\\g1a\\set1')
    ap.add_argument('--server-dir', required=True,
                    help='the same folder as the server sees it, e.g. /maintenance/.../g1a/set1')
    ap.add_argument('--server-weights', default='',
                    help='weights folder on the server (the one imw.py uses: .../imw_runtime/imw_cache)')
    ap.add_argument('--server-cache', default='',
                    help='writable folder on the server for the preprocessing cache, reused between '
                         'jobs (default: inside each job\'s output folder)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                    help='change a setting for the server, e.g. --set window_sizes=[3072]')
    ap.add_argument('--dry-run', action='store_true', help='list what would be copied, copy nothing')
    ap.add_argument('--code-path', default=GPUAAS_CODE, help='automatch_gpuaas.py as the server sees it')
    ap.add_argument('--url', default=GPUAAS_URL)
    ap.add_argument('--conda-env', default='mpad')
    ap.add_argument('--gpu-mb', type=int, default=40000, help='max_gpu_mem_required')
    a = ap.parse_args(argv)
    import automatch_job as J
    job = J.load_job(a.job, a.set)
    res = pack(job, a.to, a.server_dir.rstrip('/') or '/', a.server_weights, a.server_cache, a.dry_run,
               a.code_path, a.url, a.conda_env, a.gpu_mb)
    print()
    print(res['text'])
    bad = res['check'] and (res['check'].get('lost') or res['check'].get('error'))
    return 1 if bad or not res['references'] else 0


if __name__ == '__main__':
    sys.exit(main())
