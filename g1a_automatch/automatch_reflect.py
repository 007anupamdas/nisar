"""
Top-of-atmosphere reflectance of G1A radiance, for the learned cloud masks
that need it (ukis-csmask on MV / HV, SpecTf on HV / HS):

    rho = pi * L * d^2 / (E0(lambda) * cos(sun zenith))

  L        pixel value x radiance_scale in radiance_unit (G1A: x 0.001,
           W/cm2/sr/um), as W/m2/sr/um.
  E0       extraterrestrial solar irradiance at the band centre (about
           Thuillier 2003 / the Sentinel-2 band values, smoothed; a few %).
  d        Earth-Sun distance (AU) on the day of acquisition (1 without a
           date).
  zenith   per pixel from the acquisition time (UTC) and the pixel's
           latitude / longitude (NOAA's solar position formulas), else the
           scene's sun elevation / zenith from the metadata.

Where the sun angle and time come from (the first found wins, each logged):
  1. the job: cloud_sun_zenith_deg (> 0; then the same zenith for the whole
     scene, whatever else is found) or cloud_acq_time_utc
     ('2026-03-01T05:30:00');
  2. the raster's own tags (any namespace);
  3. metadata files beside the image (its name + .xml / .json / .txt /
     .met / .meta / .hdr / .yaml, and *META*.txt / *.xml in the folder):
     keys like SunElevation / Sun_Elevation_Angle / SolarZenith and
     SceneCenterTime / AcquisitionTime / DateOfPass + time;
  4. a time stamp in the file name (YYYYMMDDThhmmss), as a last resort.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# extraterrestrial solar irradiance (W m-2 um-1) by wavelength (nm): Sentinel-2 band values where they exist
E0_NM = [350, 400, 443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1000, 1100, 1250, 1375, 1500, 1610,
         1800, 2000, 2190, 2300, 2500]
E0_W = [1050, 1650, 1884, 1959, 1823, 1512, 1425, 1288, 1163, 1041, 955, 813, 740, 600, 450, 367, 300, 246,
        170, 115, 85, 70, 53]

ELEV_KEY = re.compile(r'(sun|solar).*(elev|altitude)', re.I)
ZEN_KEY = re.compile(r'(sun|solar).*zen', re.I)
TIME_KEY = re.compile(r'(time|date)', re.I)
TIME_GOOD = re.compile(r'(scene|acq|image|imaging|start|center|centre|sensing|observ|pass|capture)', re.I)
TIME_BAD = re.compile(r'(process|generat|creat|product|update|modif|ingest|archiv|deliver|epoch|zone|offset|format|'
                      r'exposure|integration|dwell|line)', re.I)
SIDE_EXT = ('.xml', '.json', '.txt', '.met', '.meta', '.hdr', '.yaml', '.yml', '.ini')


def e0(nm: float) -> float:
    """Extraterrestrial solar irradiance (W m-2 um-1) at nm (350-2500 nm)."""
    return float(np.interp(nm, E0_NM, E0_W))


def earth_sun_au(t: Optional[dt.datetime]) -> float:
    """Earth-Sun distance (AU) on t's day (1.0 without t)."""
    if t is None:
        return 1.0
    doy = t.timetuple().tm_yday
    return 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))


def sun_zenith(lat: np.ndarray, lon: np.ndarray, t: dt.datetime) -> np.ndarray:
    """Solar zenith angle (deg) at lat / lon (deg, east positive) at UTC
    time t (NOAA general solar position formulas, ~0.1 deg)."""
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60.0 + t.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60.0 + eqt + 4.0 * np.asarray(lon, np.float64)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(np.asarray(lat, np.float64))
    cz = np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))


def parse_time(text: str) -> Optional[dt.datetime]:
    """A UTC date-time from text in the usual metadata forms, or None."""
    s = str(text or '').strip().strip('"\'')
    if not s:
        return None
    s2 = re.sub(r'(\.\d+)?\s*(Z|UTC|GMT)$', '', s, flags=re.I).replace('T', ' ').strip()
    for f in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M:%S', '%d-%b-%Y %H:%M:%S', '%d-%b-%Y %H:%M',
              '%d/%m/%Y %H:%M:%S', '%Y%m%d %H%M%S', '%Y%m%d%H%M%S', '%d-%m-%Y %H:%M:%S', '%Y-%j %H:%M:%S'):
        try:
            return dt.datetime.strptime(s2, f)
        except ValueError:
            pass
    m = re.search(r'(20\d{2})-?(\d{2})-?(\d{2})[ T_]?(\d{2}):?(\d{2}):?(\d{2})?', s)
    if m:
        y, mo, d, h, mi, se = (int(x) if x else 0 for x in m.groups())
        try:
            return dt.datetime(y, mo, d, h, mi, se)
        except ValueError:
            return None
    return None


def _number(v) -> Optional[float]:
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(v))
    return float(m.group(0)) if m else None


def _pairs_from_file(p: str) -> List[Tuple[str, str]]:
    """(key, value) pairs of a metadata file: XML elements and attributes,
    JSON keys (flattened), or key = value / key: value lines."""
    try:
        if os.path.getsize(p) > 5_000_000:
            return []
        text = open(p, encoding='utf-8', errors='replace').read()
    except OSError:
        return []
    out: List[Tuple[str, str]] = []
    ext = os.path.splitext(p)[1].lower()
    if ext == '.xml' or text.lstrip().startswith('<'):
        try:
            import xml.etree.ElementTree as ET
            for el in ET.fromstring(text).iter():
                tag = el.tag.split('}')[-1]
                if el.text and el.text.strip():
                    out.append((tag, el.text.strip()))
                for k, v in el.attrib.items():
                    out.append((f'{tag}.{k}', v))
                    if k.lower() in ('name', 'key', 'id') and el.text:
                        out.append((v, el.text.strip()))
            return out
        except Exception:
            pass
    if ext == '.json' or text.lstrip().startswith('{'):
        try:
            def walk(o, pre=''):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, f'{pre}.{k}' if pre else str(k))
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        walk(v, pre)
                else:
                    out.append((pre.split('.')[-1], str(o)))
            walk(json.loads(text))
            return out
        except Exception:
            pass
    for line in text.splitlines():
        m = re.match(r'\s*([A-Za-z_][\w .\-/()]*?)\s*[=:]\s*(.+?)\s*;?\s*$', line)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def metadata_pairs(path: str) -> List[Tuple[str, str, str]]:
    """(source, key, value) of the raster's tags and the metadata files beside it."""
    import rasterio as rio
    out = []
    try:
        with rio.open(path) as src:
            for ns in [None] + list(src.tag_namespaces() or []):
                try:
                    tags = src.tags(ns=ns) if ns else src.tags()
                except Exception:
                    continue
                for k, v in tags.items():
                    out.append((f'tags{":" + ns if ns else ""}', k, v))
    except Exception:
        pass
    folder, base = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    stem = os.path.splitext(base)[0]
    cands = set()
    for p in glob.glob(os.path.join(folder, glob.escape(stem) + '*')):
        if p != os.path.abspath(path) and os.path.splitext(p)[1].lower() in SIDE_EXT:
            cands.add(p)
    for pat in ('*META*.txt', '*meta*.txt', '*.xml', '*.json', '*.met'):
        cands.update(p for p in glob.glob(os.path.join(folder, pat)) if not p.lower().endswith('.aux.xml'))
    for p in sorted(cands)[:20]:
        for k, v in _pairs_from_file(p):
            out.append((os.path.basename(p), k, v))
    return out


def sun_info(path: str, job: Optional[Dict] = None) -> Dict:
    """{'zenith_deg': scene value or None, 'time': UTC datetime or None,
    'zenith_from', 'time_from'} (module docstring), and 'seen': the
    sun / time keys found, to name in a message when nothing fits."""
    job = job or {}
    info: Dict = {'zenith_deg': None, 'time': None, 'zenith_from': '', 'time_from': '', 'seen': [], 'prefer': 'time'}
    z = job.get('cloud_sun_zenith_deg')
    if z not in (None, '') and float(z) > 0:            # given in the job: it wins over any time
        info.update(zenith_deg=float(z), zenith_from='job: cloud_sun_zenith_deg', prefer='zenith')
    t = parse_time(job.get('cloud_acq_time_utc') or '')
    if t is not None:
        info.update(time=t, time_from='job: cloud_acq_time_utc')
    pairs = metadata_pairs(path) if (info['zenith_deg'] is None or info['time'] is None) else []
    date_only = time_only = None
    # time keys: acquisition-like names first; processing / creation dates never
    tkeys = [(0 if TIME_GOOD.search(k) else 1, i) for i, (_, k, _) in enumerate(pairs)
             if TIME_KEY.search(k) and not TIME_BAD.search(k)]
    order = [i for _, i in sorted(tkeys)]
    for src, k, v in pairs:
        kk = re.sub(r'[\s\-]', '_', k)
        if ELEV_KEY.search(kk) or ZEN_KEY.search(kk) or (TIME_KEY.search(kk) and not TIME_BAD.search(kk)):
            info['seen'].append(f'{src}: {k} = {str(v)[:40]}')
        if info['zenith_deg'] is None:
            n = _number(v)
            if n is not None and ZEN_KEY.search(kk) and 0 <= n <= 90:
                info.update(zenith_deg=n, zenith_from=f'{src}: {k}')
            elif n is not None and ELEV_KEY.search(kk) and -5 <= n <= 90:
                info.update(zenith_deg=90.0 - n, zenith_from=f'{src}: {k} (elevation)')
    for i in order:
        src, k, v = pairs[i]
        kk = re.sub(r'[\s\-]', '_', k)
        if info['time'] is None:
            pt = parse_time(v)
            if pt is not None and (pt.hour or pt.minute or pt.second or re.search(r'\d:\d', str(v))):
                info.update(time=pt, time_from=f'{src}: {k}')
            elif pt is None and re.fullmatch(r'\s*\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|UTC)?\s*', str(v), re.I):
                time_only = (src, k, str(v))
            elif re.search(r'date', kk, re.I):
                d = parse_time(f'{v} 00:00:00') or parse_time(v)
                if d is not None:
                    date_only = (src, k, d)
    if info['time'] is None and date_only and time_only:
        hh = [int(x) for x in re.findall(r'\d+', time_only[2])[:3]] + [0, 0]
        info.update(time=date_only[2].replace(hour=hh[0], minute=hh[1], second=hh[2]),
                    time_from=f'{date_only[0]}: {date_only[1]} + {time_only[1]}')
    if info['time'] is None:
        m = re.search(r'(20\d{2})(\d{2})(\d{2})[T_]?(\d{2})(\d{2})(\d{2})', os.path.basename(path))
        if m:
            try:
                info.update(time=dt.datetime(*[int(x) for x in m.groups()]), time_from='the file name')
            except ValueError:
                pass
    return info


def zenith_grid(path: str, shape: Tuple[int, int], info: Dict, step: int = 64) -> Tuple[np.ndarray, str]:
    """(solar zenith (deg) on the image grid, how): per pixel from the time
    and each pixel's latitude / longitude when the time is known (computed
    on a coarse grid and interpolated), else the scene's value."""
    import cv2
    import rasterio as rio
    H, W = shape
    if info.get('time') is not None and not (info.get('prefer') == 'zenith' and info.get('zenith_deg') is not None):
        from pyproj import Transformer
        with rio.open(path) as src:
            tr, crs = src.transform, src.crs
        if crs is not None:
            ys = np.linspace(0, H - 1, max(2, H // step + 2))
            xs = np.linspace(0, W - 1, max(2, W // step + 2))
            gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
            px, py = tr * (gx, gy)
            lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(px, py)
            z = sun_zenith(np.asarray(lat), np.asarray(lon), info['time']).astype(np.float32)
            how = f"per pixel from the time {info['time'].isoformat()} UTC ({info['time_from']})"
            return cv2.resize(z, (W, H), interpolation=cv2.INTER_LINEAR), how
    if info.get('zenith_deg') is not None:
        return (np.full((H, W), float(info['zenith_deg']), np.float32),
                f"{info['zenith_deg']:.2f} deg for the whole scene ({info['zenith_from']})")
    seen = '; '.join(info.get('seen') or []) or 'none'
    raise ValueError('no sun angle or acquisition time found for reflectance (set cloud_sun_zenith_deg or '
                     f'cloud_acq_time_utc). Sun / time keys seen: {seen}')


def toa_reflectance(L_si: np.ndarray, nm: float, cos_z: np.ndarray, d_au: float = 1.0) -> np.ndarray:
    """TOA reflectance of radiance L_si (W m-2 sr-1 um-1) at nm."""
    return (math.pi * L_si * d_au ** 2 / (e0(nm) * np.maximum(cos_z, 1e-3))).astype(np.float32)


def read_reflectance(path: str, bands: List[Tuple[int, float]], scale: float = 0.001, unit: str = 'W/cm2/sr/um',
                     job: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(reflectance [len(bands), H, W] float32 with NaN where not valid,
    valid, info {'zenith_how', 'd_au', 'sun_zenith_median_deg', 'night'})
    of bands [(1-based band, centre nm)]."""
    import rasterio as rio
    import automatch_cloudphys as P
    with rio.open(path) as src:
        nd = src.nodata if src.nodata is not None else 0
        raw = np.stack([src.read(b).astype(np.float64) for b, _ in bands])
        shape = (src.height, src.width)
    valid = np.all(np.isfinite(raw) & (raw != nd), axis=0)
    raw = np.stack([P.destripe(r, valid) for r in raw])          # banding is not cloud
    si = sun_info(path, job)
    z, how = zenith_grid(path, shape, si)
    cz = np.cos(np.radians(z))
    d = earth_sun_au(si.get('time'))
    u = unit if P.unit_name(unit) != 'auto' else 'W/m2/sr/um'
    refl = np.stack([toa_reflectance(P.to_si(raw[i] * scale, nm / 1000.0, u), nm, cz, d)
                     for i, (_, nm) in enumerate(bands)])
    night = valid & (z > 85.0)
    refl[:, ~valid] = np.nan
    info = {'zenith_how': how, 'd_au': round(d, 4), 'night_pct': round(100.0 * night.sum() / max(valid.sum(), 1), 1),
            'sun_zenith_median_deg': round(float(np.median(z[valid])), 2) if valid.any() else None,
            'acq_time_utc': si['time'].isoformat() if si.get('time') else ''}
    return refl, valid & ~night, info
