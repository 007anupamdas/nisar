"""
Ground-truth tie points (manual, RIVAL layout).

The columns are RIVAL's ground-truth CSV in this order, separated by commas,
spaces or tabs, with or without a header line:

    In_X  In_Y  Ref_X  Ref_Y  DX_Err  DY_Err  Row  In_Lon  In_Lat  Ref_Lon  Ref_Lat

In_*  : where the point is according to the image being assessed (its own,
        erroneous geolocation); Ref_* : where it really is (the reference).
Only In_Lon / In_Lat / Ref_Lon / Ref_Lat are needed; X / Y are RIVAL's local
grid and serve to recover a missing lon/lat.

Unset points are common in these files: a lon/lat of 0, text such as '-', or
(in space-separated files, where an empty cell cannot be written) a row that
simply ends after Row or after In_Lat. Such points are recovered from their
X / Y through the X/Y -> lon/lat mapping of the file's complete points when
that mapping is smooth (quadratic, from 8 points), else left out with a note.
Rows whose input and reference are more than max_error_km apart are broken
rows, not errors, and are left out too.
"""
from __future__ import annotations

import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

COLUMNS = ['In_X', 'In_Y', 'Ref_X', 'Ref_Y', 'DX_Err', 'DY_Err', 'Row', 'In_Lon', 'In_Lat', 'Ref_Lon', 'Ref_Lat']
MAX_ERROR_KM = 500.0


def _num(tok: str) -> float:
    try:
        v = float(tok)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


def read_table(path: str) -> pd.DataFrame:
    """The file's rows as the RIVAL columns (NaN where a value is absent)."""
    rows = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = [t for t in re.split(r'[,\s;]+', line) if t != '']
            if not toks:
                continue
            if not re.match(r'^[-+]?(\d|\.\d)', toks[0]):          # a header line
                if any(t.strip().lower() == 'in_lon' for t in toks):
                    continue
                continue
            vals = [_num(t) for t in toks[:len(COLUMNS)]]
            vals += [float('nan')] * (len(COLUMNS) - len(vals))
            rows.append(vals)
    return pd.DataFrame(rows, columns=COLUMNS)


def _coord(v: pd.Series, lim: float) -> pd.Series:
    return v.where((v != 0) & (v.abs() <= lim))


def _fill_from_xy(lon, lat, x, y, known):
    gx = np.concatenate([k[0].values for k in known])
    gy = np.concatenate([k[1].values for k in known])
    glo = np.concatenate([k[2].values for k in known])
    gla = np.concatenate([k[3].values for k in known])
    ok = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(glo) & np.isfinite(gla)
    want = (lon.isna() | lat.isna()) & x.notna() & y.notna() & ((x != 0) | (y != 0))
    if not want.any() or ok.sum() < 4:
        return lon, lat, []
    gx, gy, glo, gla = gx[ok], gy[ok], glo[ok], gla[ok]
    mx, my = gx.mean(), gy.mean()
    sc = max(np.ptp(gx), np.ptp(gy), 1.0)

    def design(a, b):
        u, v = (a - mx) / sc, (b - my) / sc
        return np.column_stack([np.ones_like(u), u, v] + ([u * u, u * v, v * v] if ok.sum() >= 8 else []))
    A = design(gx, gy)
    clo = np.linalg.lstsq(A, glo, rcond=None)[0]
    cla = np.linalg.lstsq(A, gla, rcond=None)[0]
    if max(np.abs(A @ clo - glo).max(), np.abs(A @ cla - gla).max()) > 2e-4:     # ~20 m
        return lon, lat, []
    B = design(x[want].values, y[want].values)
    lon, lat = lon.copy(), lat.copy()
    lon[want], lat[want] = B @ clo, B @ cla
    return lon, lat, list(want[want].index)


def load_points(path: str, max_error_km: float = MAX_ERROR_KM, quiet: bool = False) -> pd.DataFrame:
    """Tie points: point_id, in_lon, in_lat, ref_lon, ref_lat (complete rows only)."""
    df = read_table(path)
    name = os.path.basename(path)
    say = (lambda *a: None) if quiet else print
    if df.empty:
        raise ValueError(f'{name}: no tie points')
    ids = [int(v) if math.isfinite(v) else i + 1 for i, v in enumerate(df['Row'])]
    ilo, ila = _coord(df['In_Lon'], 180), _coord(df['In_Lat'], 90)
    rlo, rla = _coord(df['Ref_Lon'], 180), _coord(df['Ref_Lat'], 90)
    known = [(df['In_X'], df['In_Y'], ilo, ila), (df['Ref_X'], df['Ref_Y'], rlo, rla)]
    rlo, rla, fr = _fill_from_xy(rlo, rla, df['Ref_X'], df['Ref_Y'], known)
    ilo, ila, fi = _fill_from_xy(ilo, ila, df['In_X'], df['In_Y'], known)
    for i in fr:
        say(f'[GT] {name}: point {ids[i]} had no Ref_Lon/Ref_Lat; taken from its Ref_X/Ref_Y')
    for i in fi:
        say(f'[GT] {name}: point {ids[i]} had no In_Lon/In_Lat; taken from its In_X/In_Y')
    keep = ilo.notna() & ila.notna() & rlo.notna() & rla.notna()
    for i in df.index[~keep]:
        say(f'[GT] {name}: point {ids[i]} left out: incomplete')
    lat0 = np.radians((ila + rla) / 2.0)
    dist = np.hypot((ilo - rlo) * 111.32 * np.cos(lat0), (ila - rla) * 110.57)
    far = keep & (dist > max_error_km)
    for i in df.index[far]:
        say(f'[GT] {name}: point {ids[i]} left out: input and reference {dist[i]:.0f} km apart (broken row)')
    keep &= ~far
    if not keep.any():
        raise ValueError(f'{name}: no complete tie points')
    return pd.DataFrame({'point_id': [ids[i] for i in df.index[keep]],
                         'in_lon': ilo[keep].values, 'in_lat': ila[keep].values,
                         'ref_lon': rlo[keep].values, 'ref_lat': rla[keep].values}).reset_index(drop=True)
