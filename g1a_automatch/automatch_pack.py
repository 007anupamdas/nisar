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
    submit_gpuaas.sh        curl requests: ./submit_gpuaas.sh env|weights|preflight|run
    PACK_REPORT.txt         what was copied, and the curl commands to paste
and then reads the copied reference folder back the way a run will, to check
that every selected reference is still found. Files already there with the
same size are not copied again.
"""

import argparse
import json
import math
import os
import posixpath
import shutil
import sys
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GPUAAS_URL = 'http://gpuaas.private.nrsc.gov.in:8000/submit_job'
GPUAAS_CODE = '/maintenance/ICIGDev/GPUPOC/exe/g1a_automatch/automatch_gpuaas.py'


def companions(path: str) -> List[str]:
    """Files beside a raster that belong to it: X.tif.aux.xml, X.tif.ovr,
    X.tfw, X.prj, X.aux.xml, and RIVAL sidecars (X.met, X_meta.txt, ...)."""
    import automatch_refs as refs
    folder, base = os.path.split(path)
    stem = os.path.splitext(base)[0].lower()
    out = []
    for name in sorted(os.listdir(folder or '.')):
        low = name.lower()
        if low == base.lower() or os.path.isdir(os.path.join(folder, name)):
            continue
        if low.startswith(base.lower() + '.'):
            out.append(os.path.join(folder, name))
        elif low.startswith(stem + '.') and not low.endswith(refs.RASTER_EXTS):
            out.append(os.path.join(folder, name))       # X.tfw, X.prj, X.met, ...
        elif refs.is_meta_file(name) and refs.meta_base_stem(name).lower() == stem:
            out.append(os.path.join(folder, name))       # X_meta.txt, X.h5.iso.xml
    return out


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
    cat = refs.scan_reference_folder(job['reference_dir'], mode)
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
    return cat, hits, src


def _copy(src: str, dst: str, plan: List[Tuple[str, str]]):
    plan.append((src, dst))


def _size(path: str) -> int:
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path)


def _run_plan(plan: List[Tuple[str, str]], dry: bool) -> Tuple[int, int]:
    copied = skipped = 0
    for src, dst in plan:
        if dry:
            continue
        if os.path.isdir(src):
            if os.path.isdir(dst):
                skipped += 1
                continue
            shutil.copytree(src, dst)
            copied += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def curl_command(mode: str, settings_server: str, code_path: str, url: str, conda_env: str,
                 gpu_mb: int) -> str:
    body = json.dumps({'code_path': code_path, 'conda_env': conda_env, 'max_gpu_mem_required': gpu_mb,
                       'input_path': f'{mode},{settings_server}'}, indent=1)
    return (f'curl -X POST "{url}" -H "Content-Type: application/json" '
            f'-H "X-User-Name: $(whoami)" -d \'{body}\'')


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
    _copy(inp, os.path.join(dest, 'input', name), plan)
    if os.path.isfile(inp):
        for c in companions(inp):
            _copy(c, os.path.join(dest, 'input', os.path.basename(c)), plan)
    settings['input_path'] = posixpath.join(server_dir, 'input', name)

    # references: only those the run can use, with their footprint source
    cat, hits, foot_src = select_references(job)
    ref_root = os.path.abspath(job['reference_dir'])
    ref_name = os.path.basename(ref_root.rstrip(os.sep)) or 'reference'
    ref_dest = os.path.join(dest, 'reference', ref_name)
    metas = set()
    for tif in hits:
        rel = os.path.relpath(tif, ref_root)
        _copy(tif, os.path.join(ref_dest, rel), plan)
        for c in companions(tif):
            _copy(c, os.path.join(ref_dest, os.path.relpath(c, ref_root)), plan)
        meta = cat['footprints'][tif].get('meta')
        if meta:
            metas.add(meta)
    for meta in sorted(metas):
        _copy(meta, os.path.join(ref_dest, os.path.relpath(meta, ref_root)), plan)
        if cat['mode'] == refs.REF_MODE_INDEX:          # .shx / .dbf / .prj / .cpg of the index
            for c in companions(meta):
                _copy(c, os.path.join(ref_dest, os.path.relpath(c, ref_root)), plan)
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
            _copy(os.path.abspath(job[key]), os.path.join(dest, 'truth', base), plan)
            settings[key] = posixpath.join(server_dir, 'truth', base)

    settings['output_dir'] = posixpath.join(server_dir, 'output')   # the GPU service replaces it
    settings['temp_dir'] = server_cache or ''
    settings['weights_cache_dir'] = server_weights or ''
    settings_server = posixpath.join(server_dir, 'automatch_settings.json')

    # de-duplicate (a sidecar can be both a companion and the footprint source)
    seen, uniq = set(), []
    for s, d in plan:
        if d not in seen:
            seen.add(d)
            uniq.append((s, d))
    plan = uniq
    total = sum(_size(s) for s, _ in plan)
    copied, skipped = _run_plan(plan, dry_run)

    cmds = {m: curl_command(m, settings_server, code_path, url, conda_env, gpu_mb)
            for m in ('env', 'weights', 'preflight', 'run')}
    report = [f"Packed for the server: {name}",
              f"  image footprint from: {foot_src}",
              f"  reference collection: {ref_root} ({cat['mode']}, {len(cat['footprints'])} footprints)",
              f"  references within {float(job['max_expected_error_m']) / 1000:g} km "
              f"(+{float(job['search_margin_m'] or 0) / 1000:g} km margin) of the image: {len(hits)}",
              *[f'    {os.path.relpath(t, ref_root)}' for t in hits],
              f"  {len(plan)} file(s), {total / 1e6:.1f} MB"
              + (' (dry run: nothing copied)' if dry_run else f' -- {copied} copied, {skipped} already there'),
              f"  local folder : {dest}",
              f"  server folder: {server_dir}",
              f"  settings     : {settings_server}",
              '',
              'Submit (in this order; each job writes its report into the output folder the service gives it):']
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
        script = ['#!/bin/bash',
                  f'# automatch on the GPU service for {name}.  Usage: ./submit_gpuaas.sh env|weights|preflight|run',
                  f'URL="{url}"', f'CODE="{code_path}"', f'SETTINGS="{settings_server}"',
                  'case "${1:-}" in env|weights|preflight|run) ;; *) echo "usage: $0 env|weights|preflight|run"; exit 1;; esac',
                  'curl -X POST "$URL" -H "Content-Type: application/json" -H "X-User-Name: $(whoami)" \\',
                  f'  -d "{{\\"code_path\\": \\"$CODE\\", \\"conda_env\\": \\"{conda_env}\\", '
                  f'\\"max_gpu_mem_required\\": {gpu_mb}, \\"input_path\\": \\"$1,$SETTINGS\\"}}"',
                  'echo']
        with open(os.path.join(dest, 'submit_gpuaas.sh'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(script) + '\n')
        # read the copy back the way a run will
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
    return {'text': text, 'references': hits, 'files': len(plan), 'bytes': total, 'copied': copied,
            'skipped': skipped, 'settings': settings, 'check': check, 'commands': cmds}


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
    print(res['text'])
    bad = res['check'] and (res['check'].get('lost') or res['check'].get('error'))
    return 1 if bad or not res['references'] else 0


if __name__ == '__main__':
    sys.exit(main())
