"""
A local copy of the reference tiles inside a region: G1A sees India and its
neighbours only, so a job never needs the rest of a global reference set.

    python automatch_job.py refsubset <reference folder> <local folder>
        [--region 40 -10 110 45]     # lon_min lat_min lon_max lat_max (G1A default)
        [--per-tile N]               # Landsat: the newest N dates per path/row (0: all)
        [--dry-run]                  # count and size only, nothing copied

Every reference whose footprint (from its index / sidecar / degree-tile
name, as the jobs read it) meets the region is copied, and beside each copy a
'<name>.met' sidecar states its footprint, so the local folder needs no index
and is scanned in seconds (sidecar mode). Copies already there with the same
size are skipped, so a second run only adds what is new. REFSUBSET.csv lists
what was taken. Point a job's reference folder at the local folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

G1A_REGION = (40.0, -10.0, 110.0, 45.0)        # lon_min, lat_min, lon_max, lat_max


def ring_meets(ring: Sequence[Tuple[float, float]], region: Sequence[float]) -> bool:
    """Whether a lon/lat ring meets the region box."""
    from shapely.geometry import Polygon, box
    if not ring or len(ring) < 3:
        return False
    try:
        return Polygon(ring).buffer(0).intersects(box(*region))
    except Exception:
        lons, lats = [p[0] for p in ring], [p[1] for p in ring]
        return not (max(lons) < region[0] or min(lons) > region[2] or max(lats) < region[1] or min(lats) > region[3])


def landsat_tile(name: str) -> Optional[Tuple[str, str]]:
    """(path/row, date key) of a Landsat scene name (LC81500402015103LGN00 or
    LC08_L1TP_150040_20150413_...), else None."""
    m = re.match(r'^L[COTEM]\d(\d{3})(\d{3})(\d{4})(\d{3})', name)
    if m:
        return f'{m.group(1)}{m.group(2)}', f'{m.group(3)}{m.group(4)}'
    m = re.match(r'^L[COTEM]\d{2}_\w{4}_(\d{6})_(\d{8})', name)
    if m:
        return m.group(1), m.group(2)
    return None


def pick(footprints: Dict[str, Dict], region: Sequence[float], per_tile: int = 0) -> List[str]:
    """The reference paths inside the region; with per_tile, only the newest
    per_tile dates of each Landsat path/row (other names all kept)."""
    inside = [p for p, rec in sorted(footprints.items()) if ring_meets(rec.get('ring') or [], region)]
    if not per_tile:
        return inside
    groups: Dict[str, List[Tuple[str, str]]] = {}
    keep = []
    for p in inside:
        t = landsat_tile(os.path.basename(p))
        if t is None:
            keep.append(p)
        else:
            groups.setdefault(t[0], []).append((t[1], p))
    for _, items in sorted(groups.items()):
        keep += [p for _, p in sorted(items, reverse=True)[:per_tile]]
    return sorted(keep)


def met_for(ring: Sequence[Tuple[float, float]], name: str, rec: Dict) -> Dict:
    """A '.met' JSON footprint the reference scanner reads (Image* corners,
    Prod* the bounding box)."""
    pts = [(float(x), float(y)) for x, y in ring]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    out = {'ProdULLat': max(lats), 'ProdULLon': min(lons), 'ProdURLat': max(lats), 'ProdURLon': max(lons),
           'ProdLRLat': min(lats), 'ProdLRLon': max(lons), 'ProdLLLat': min(lats), 'ProdLLLon': min(lons),
           'Sensor': name, 'automatch_footprint_source': str(rec.get('source') or '')}
    if len(pts) == 4:          # the scene's own corners: UL, UR, LR, LL by position
        by_lat = sorted(pts, key=lambda p: -p[1])
        top, bottom = sorted(by_lat[:2]), sorted(by_lat[2:])
        for k, (lon, lat) in (('UL', top[0]), ('UR', top[1]), ('LR', bottom[1]), ('LL', bottom[0])):
            out[f'Image{k}Lat'], out[f'Image{k}Lon'] = lat, lon
    if rec.get('crs'):
        m = re.search(r'(\d+)$', str(rec['crs']))
        if m:
            out['EPSG'] = int(m.group(1))
    return out


def subset(source: str, dest: str, region: Sequence[float] = G1A_REGION, per_tile: int = 0,
           dry_run: bool = False, mode: Optional[str] = None) -> Dict:
    """Copy the references inside region from source to dest (module
    docstring); {'n', 'bytes', 'copied', 'skipped', 'csv'}."""
    import automatch_refs as refs
    cat = refs.scan_reference_folder(source, mode)
    chosen = pick(cat['footprints'], region, per_tile)
    total = sum(os.path.getsize(p) for p in chosen if os.path.exists(p))
    print(f'[RefSubset] {len(chosen)} of {len(cat["footprints"])} reference(s) meet the region '
          f'{", ".join(f"{v:g}" for v in region)} (lon/lat){f", newest {per_tile} per Landsat tile" if per_tile else ""}: '
          f'{total / 1e9:.1f} GB', flush=True)
    if dry_run:
        return {'n': len(chosen), 'bytes': total, 'copied': 0, 'skipped': 0, 'csv': ''}
    os.makedirs(dest, exist_ok=True)
    rows, copied, skipped = [], 0, 0
    for k, p in enumerate(chosen, 1):
        name = os.path.basename(p)
        out = os.path.join(dest, name)
        rec = cat['footprints'][p]
        if os.path.exists(out) and os.path.getsize(out) == os.path.getsize(p):
            skipped += 1
        else:
            shutil.copyfile(p, out)
            copied += 1
        stem = os.path.splitext(name)[0]
        with open(os.path.join(dest, stem + '.met'), 'w', encoding='utf-8') as f:
            json.dump(met_for(rec['ring'], name, rec), f)
        rows.append({'reference': name, 'source': p, 'bytes': os.path.getsize(out)})
        if k % 50 == 0 or k == len(chosen):
            print(f'[RefSubset] {k}/{len(chosen)} ({copied} copied, {skipped} already there)', flush=True)
    import pandas as pd
    csv = os.path.join(dest, 'REFSUBSET.csv')
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f'[RefSubset] done: {copied} copied, {skipped} already there; point the jobs\' reference folder at {dest}',
          flush=True)
    return {'n': len(chosen), 'bytes': total, 'copied': copied, 'skipped': skipped, 'csv': csv}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Copy the reference tiles inside a region to a local folder')
    ap.add_argument('source', help='the reference folder (e.g. the L8_ref share)')
    ap.add_argument('dest', help='the local folder to fill')
    ap.add_argument('--region', type=float, nargs=4, default=list(G1A_REGION),
                    metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
                    help='default: G1A coverage, 40 -10 110 45')
    ap.add_argument('--per-tile', type=int, default=0, help='Landsat: the newest N dates per path/row (0: all)')
    ap.add_argument('--dry-run', action='store_true', help='count and size only')
    a = ap.parse_args(argv)
    subset(a.source, a.dest, a.region, a.per_tile, a.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
