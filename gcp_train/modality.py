"""
What kind of image a view is, as the matcher is told it.

The vocabulary is by physics, not by satellite, so what is learnt on
MicroSat's thermal band carries over to G1A's MX-LWIR:

    visible  400-700 nm (and panchromatic)
    nir      700-1000 nm
    swir     1000-2500 nm
    mwir     3-5 um (reflected and emitted)
    lwir     7-14 um (emitted: thermal)
    sar      microwave (RISAT, NISAR, ...)

A manifest row gives it as `modality` (a name above or an alias: optical,
pan, thermal, tir, ...) or as `wavelength_nm` (a band's centre). Neither:
SAR sensors are known by name, anything else counts as visible.
"""
from __future__ import annotations

import math
import re
from typing import Optional

MODALITIES = ('visible', 'nir', 'swir', 'mwir', 'lwir', 'sar')

ALIASES = {'optical': 'visible', 'vis': 'visible', 'pan': 'visible', 'panchromatic': 'visible', 'vnir': 'visible',
           'rgb': 'visible', 'red': 'visible', 'green': 'visible', 'blue': 'visible',
           'near-infrared': 'nir', 'mir': 'mwir', 'thermal': 'lwir', 'tir': 'lwir', 'lwir': 'lwir',
           'microwave': 'sar', 'radar': 'sar'}

SAR_NAMES = ('risat', 'nisar', 'radarsat', 'rcm', 'sentinel-1', 'sentinel1', 'eos-04', 'eos04')


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or not str(v).strip()


def from_wavelength(nm: float) -> str:
    """The modality of a band centred at nm nanometres (micrometres are
    recognised by size: a value under 30 is taken as um)."""
    nm = float(nm)
    if nm < 30:
        nm *= 1000.0
    if nm < 700:
        return 'visible'
    if nm < 1000:
        return 'nir'
    if nm < 2600:
        return 'swir'
    if nm < 6000:
        return 'mwir'
    if nm < 20000:
        return 'lwir'
    return 'sar'


def resolve(modality=None, wavelength_nm=None, sensor: str = '', default: str = 'visible') -> str:
    if not _blank(modality):
        m = str(modality).strip().lower()
        m = ALIASES.get(m, m)
        if m not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}: one of {', '.join(MODALITIES)} "
                             f"(or {', '.join(sorted(ALIASES))})")
        return m
    if not _blank(wavelength_nm):
        return from_wavelength(float(wavelength_nm))
    s = str(sensor or '').lower()
    if any(k in s for k in SAR_NAMES) or 'sar' in re.split(r'[^a-z0-9]+', s):
        return 'sar'
    return default


def index(name) -> int:
    """The modality's number (what the matcher's embedding is indexed by);
    unknown or missing counts as visible."""
    try:
        return MODALITIES.index(str(name).strip().lower())
    except ValueError:
        return 0
