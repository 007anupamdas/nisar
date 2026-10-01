#!/usr/bin/env python3
"""
automatch_engine -- NISAR H5 vs reference-collection automatic matching.

Derived from dqeagdq_integrated_v2.py (the NISAR-S1 production pipeline),
which is left untouched; this module is the engine behind automatch_job.py
(CLI) and DPQED_automatch.py (GUI). References are any collection that
DPQED_rival.py understands -- L8_ref (index shapefile), C1 (degree tiles),
S1 or NISAR sidecar folders -- discovered through automatch_refs.

Pol-wise + H5 scene reader + multi-detector + chip consensus

Merged design:
- H5/MET scene discovery and polarization handling from frozen pol-wise pipeline
- Two-step S1 overlap logic from v6
- Multi-detector / multi-matcher flow from v6 (not frozen to a single detector/matcher)
- Chip consensus selection from frozen pol-wise pipeline
- Manual-GCP error surface for consensus scoring (ManualGCPLoader / ErrorSurfaceModel)

Robustness notes (this revision)
--------------------------------
* Import-safe: nothing reads sys.argv at import. Call setup_logging(dir) from
  your entry point. dqe_job.py / dqe_gui.py drive this module programmatically.
* NISAR H5: L-band (LSAR) and S-band (SSAR), frequencyA/B, GSLC and GCOV;
  complex stored natively OR as a compound {r, i} pair; NaN fill -> nodata 0.
  The amplitude raster is written block-wise to a cached GeoTIFF instead of
  being held in RAM (a full GSLC is tens of GB).
* The .met sidecar is optional. The swath footprint comes from, in order:
  .met Image* corners -> .h5.iso.xml gml:posList -> H5 identification/
  boundingPolygon -> valid-data hull of the amplitude raster.
* References: discovered with RIVAL's rules (index-shp / sidecar /
  degree-tile, see automatch_refs). Each candidate is confirmed against its
  valid-pixel hull (fill = nodata, NaN, and RIVAL's REF_FILL_VALUES), computed
  in the raster's own CRS and reprojected, never assumed lon/lat.
* Pair cache is keyed by scene + grid + pol + reference dir + resolution, so a
  cache directory shared between scenes can never feed one scene's chips into
  another scene's run.
* The S1 window for each NISAR window is taken from the window's map bounds,
  not from a pixel scale factor, so a one-pixel difference in crop origins no
  longer shifts which ground is compared.
* Map coordinates use pixel centres (index i -> origin + (i + 0.5) * res).
  Along/across errors are unchanged at same resolution; absolute coordinates
  (and RIVAL overlays) move half a pixel onto the true centre.
* scan/pix in file names and chip statistics are the window's top-left ROW and
  COLUMN in the ORIGINAL H5 grid (previously row/col offsets were swapped and
  mixed resampled with original pixel units). This is what manual_gcp_csv's
  scan/pix columns are matched against.
"""

import os
import re
import copy
import csv
import gc
import glob
import hashlib
import itertools
import json
import math
import shutil
import time
import warnings
import sys
import traceback
import xml.etree.ElementTree as ET
import h5py as hp

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Callable

import cv2
import kornia as K
import kornia.color  # noqa: F401  (K.color is lazy on some releases)
import kornia.feature as KF
import numpy as np
import pandas as pd
import rasterio as rt
import rasterio.windows
import rioxarray as rxr
import shapely.ops
import shapely.wkt
import torch as th
import xarray as xr

from pyproj import CRS as PJCRS, Transformer
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine, from_origin, xy as rt_xy
from rasterio.warp import reproject as warp_reproject, Resampling, transform_bounds
from shapely.geometry import Polygon, MultiPoint, box as shp_box, mapping
from shapely.ops import transform

import automatch_refs as refs

print(f"kornia version: {K.__version__}")


# =============================================================================
# LOGGING
# =============================================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (ValueError, OSError):
                pass  # a closed log file must not take the run down

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except (ValueError, OSError):
                pass


_ORIG_STDOUT, _ORIG_STDERR = sys.stdout, sys.stderr
_LOG_FILE = None


def setup_logging(output_dir: str, filename: str = 'log') -> str:
    """Mirror stdout/stderr into <output_dir>/<filename>. Safe to call again:
    the previous log file is closed and the streams re-pointed."""
    global _LOG_FILE
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.close()
        except Exception:
            pass
    _LOG_FILE = open(log_path, 'a', buffering=1, encoding='utf-8')
    sys.stdout = Tee(_ORIG_STDOUT, _LOG_FILE)
    sys.stderr = Tee(_ORIG_STDERR, _LOG_FILE)
    print(f'Logging to {log_path}')
    return log_path


warnings.filterwarnings('ignore')


# =============================================================================
# DEVICE
# =============================================================================
def _auto_device() -> th.device:
    # kornia.utils is not reachable as an attribute on every kornia release,
    # so pick the device with torch directly.
    if th.cuda.is_available():
        return th.device('cuda')
    mps = getattr(th.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return th.device('mps')
    return th.device('cpu')


device = _auto_device()


def set_device(name: str = 'auto'):
    """'auto' | 'cuda' | 'cuda:N' | 'mps' | 'cpu'. Matchers read the module
    global at construction, so call this before building a pipeline."""
    global device
    name = (name or 'auto').lower()
    if name == 'auto':
        device = _auto_device()
    elif name.startswith('cuda') and not th.cuda.is_available():
        print(f'[Device] {name} requested but CUDA is not available -- using CPU')
        device = th.device('cpu')
    else:
        device = th.device(name)
    describe_device()
    return device


def describe_device():
    print(f'Using device: {device}')
    if device.type == 'cuda' and th.cuda.is_available():
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
        th.backends.cudnn.benchmark = True
        idx = device.index or 0
        print(f'GPU: {th.cuda.get_device_name(idx)}')
        print(f'CUDA: {th.version.cuda}')
        mem_total = th.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f'GPU Memory: {mem_total:.1f} GB')


describe_device()


def safe_cuda_empty_cache():
    try:
        th.cuda.empty_cache()
    except RuntimeError:
        pass


# =============================================================================
# MODEL WEIGHTS
# =============================================================================
# Extra torch-hub folders searched for weights (the configured weights cache);
# see add_weights_dir(). kornia's own default folder is always searched.
WEIGHT_DIRS: List[str] = []


def add_weights_dir(path: str) -> None:
    """Also look for model weights in <path>/checkpoints (a torch-hub folder)
    or <path>/torch/hub/checkpoints (a cache root written by prefetch_weights)."""
    for cand in (path, os.path.join(path, 'torch', 'hub')):
        cand = os.path.abspath(os.path.expanduser(cand))
        if os.path.isdir(os.path.join(cand, 'checkpoints')) and cand not in WEIGHT_DIRS:
            WEIGHT_DIRS.append(cand)


def _default_hub_dir() -> str:
    xdg = os.getenv('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache'))
    return os.path.join(xdg, 'torch', 'hub')


def weight_search_dirs() -> List[str]:
    dirs = [th.hub.get_dir(), _default_hub_dir()]
    if os.getenv('TORCH_HOME'):
        dirs.append(os.path.join(os.environ['TORCH_HOME'], 'hub'))
    out: List[str] = []
    for d in dirs + WEIGHT_DIRS:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in out:
            out.append(d)
    return out


_ORIG_LOAD_STATE_DICT = th.hub.load_state_dict_from_url


def find_weight_file(url: str, file_name: Optional[str] = None,
                     model_dir: Optional[str] = None) -> Tuple[str, Optional[str], List[str]]:
    """(file name, path if present in any weights folder else None, folders
    searched) for a torch-hub weight URL -- the lookup the loader below does."""
    from urllib.parse import urlparse
    fname = file_name or os.path.basename(urlparse(url).path)
    folders = ([model_dir] if model_dir else []) + [os.path.join(h, 'checkpoints')
                                                    for h in weight_search_dirs()]
    for folder in folders:
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            return fname, path, folders
    return fname, None, folders


# While automatch_weights traces a model, it installs a hook here that sees
# every weight request (and may answer it); None in normal runs.
WEIGHTS_TRACE_HOOK: Optional[Callable] = None


def _load_state_dict_any_cache(url, model_dir=None, map_location=None, progress=True,
                               check_hash=False, file_name=None, **kwargs):
    """torch.hub.load_state_dict_from_url that first looks for the file in
    every known weights folder, so weights downloaded earlier (e.g. into
    kornia's default ~/.cache/torch/hub) are used on an offline machine even
    when TORCH_HOME points elsewhere. Downloads only if found nowhere."""
    fname, path, folders = find_weight_file(url, file_name, model_dir)
    if WEIGHTS_TRACE_HOOK is not None:
        answer = WEIGHTS_TRACE_HOOK('torch-hub', url=url, file=fname, path=path, searched=folders)
        if answer is not None:
            return answer
    if path:
        if os.path.dirname(path) != os.path.join(th.hub.get_dir(), 'checkpoints'):
            print(f'[Weights] {fname} from {os.path.dirname(path)}')
        try:
            return th.load(path, map_location=map_location,
                           weights_only=kwargs.get('weights_only', False))
        except TypeError:  # torch without weights_only
            return th.load(path, map_location=map_location)
    return _ORIG_LOAD_STATE_DICT(url, model_dir=model_dir, map_location=map_location,
                                 progress=progress, check_hash=check_hash, file_name=file_name, **kwargs)


th.hub.load_state_dict_from_url = _load_state_dict_any_cache


class ModelLoadError(RuntimeError):
    """A detector's network could not be built (typically: its weights are not
    on this machine and cannot be downloaded). Stops that detector."""


# label -> message, for the whole job: a failed load is not retried per window
FAILED_MODELS: Dict[str, str] = {}
# labels of the models being built right now (innermost last), for the
# weights checker to say which model asked for which file
MODEL_LABELS: List[str] = []


# =============================================================================
# PROGRESS
# =============================================================================
# dqe_job.py installs a callable here; the engine calls report() at coarse
# milestones and once per window. Keep the payload JSON-serialisable.
PROGRESS_HOOK: Optional[Callable[[Dict], None]] = None


def report(event: str, **fields):
    if PROGRESS_HOOK is None:
        return
    try:
        PROGRESS_HOOK({'event': event, **fields})
    except Exception:
        pass


# =============================================================================
# CONFIG
# =============================================================================
# Map coordinate of pixel index i is origin + (i + PIXEL_CENTER_OFFSET) * res.
# Keypoints from kornia / imcui put pixel centres on integer coordinates.
PIXEL_CENTER_OFFSET = 0.5

# RANSAC estimator tokens. The legacy numeric token 4 has always run MAGSAC
# (the code ignored the value), so it keeps meaning MAGSAC here even though
# cv2.LMEDS happens to be 4 as well.
RANSAC_ESTIMATORS = {
    '4': cv2.USAC_MAGSAC,
    'magsac': cv2.USAC_MAGSAC,
    'ransac': cv2.RANSAC,
    'lmeds': cv2.LMEDS,
    'accurate': getattr(cv2, 'USAC_ACCURATE', cv2.USAC_MAGSAC),
    'prosac': getattr(cv2, 'USAC_PROSAC', cv2.RANSAC),
}


def ransac_flag(method) -> int:
    key = str(method).strip().lower()
    if key not in RANSAC_ESTIMATORS:
        raise ValueError(f'Unknown RANSAC method {method!r}. '
                         f'Use one of {sorted(RANSAC_ESTIMATORS)}')
    return RANSAC_ESTIMATORS[key]


@dataclass
class PipelineConfig:
    window_size: int = 1024   # 16 GB workstation default (4096 suits an A100)
    window_size_small: int = 512  # fallback if valid fraction too low
    min_valid_fraction: float = 0.30  # skip if < 30% valid
    inpaint_radius: int = 5
    keypoint_density: float = 9000
    num_features: int = 8000
    min_num_features: int = 1000
    max_num_features: int = 32000
    target_resolution: Optional[int] = 10
    loftr_max_window: int = 1024

    use_disk_cache: bool = True
    cleanup_after_pair: bool = False
    check_existing_pairs: bool = True
    use_amp: bool = True
    debug_mode: bool = False

    # kornia's own AdaLAM defaults (what kornia.feature.match_adalam uses when
    # given no settings). The NISAR-S1 pipeline used 1 / 2048 / 1000, which is
    # strict for small keypoint budgets; both are reachable from Configure…
    adalam_force_seed_mnn: bool = True
    adalam_search_expansion: int = 4
    adalam_ransac_iters: int = 128
    adalam_min_confidence: int = 200
    adalam_refit: bool = True

    smnn_thresholds: List[float] = None
    ransac_methods: List = None
    ransac_thresholds: List[float] = None
    ransac_confidences: List[float] = None

    # Reference collection used for every polarisation (L8_ref, C1, ...).
    reference_dir: str = ''
    reference_label: str = ''          # file-name tag; default from folder name
    reference_mode: Optional[str] = None  # force 'index-shp'|'sidecar'|'degree-tile'
    reference_band: int = 1            # band of a multi-band reference raster
    # Per-input-channel reference band, e.g. {'band1': 4, 'band2': 3}.
    reference_band_map: Optional[Dict[str, int]] = None
    reference_fill_values: Tuple = (0, 3)  # RIVAL's REF_FILL_VALUES
    # Optional S1-style per-pol folders: HH/VV -> _vv, HV/VH -> _vh. When set
    # they take precedence over reference_dir for those pols.
    reference_dir_vv: str = ''
    reference_dir_vh: str = ''

    output_base_dir: str = './output'
    temp_dir: str = './temp_coregistered'

    s1_decimation: int = 10
    min_area: float = 200.0

    consensus_tolerance_m: float = 5.0
    consensus_mode_bin_m: float = 0.5
    min_inliers_per_chip: int = 6
    min_surviving_chips: int = 3

    # Path to manually-observed GCP CSV produced by analyst
    # Columns: scan, pix, Map_X, Map_Y, Map_X_ref, Map_Y_ref
    # If empty, select_configs falls back to inlier-count-only score
    manual_gcp_csv: str = ''

    # ── NISAR H5 selection ──────────────────────────────────────────────────
    nisar_band: str = 'auto'        # 'auto' | 'LSAR' | 'SSAR'
    nisar_frequency: str = 'A'      # 'A' | 'B'
    # Input channels to process: NISAR pols ('HH', ...) or raster bands
    # ('band1', ...). None = all channels the input has.
    pols: Optional[List[str]] = None

    # ── Detector / matcher selection ────────────────────────────────────────
    # {detector: [matcher, ...]} restricts which matchers run, e.g.
    # {'disk': ['lgm']}. Missing detectors run all supported matchers.
    detector_matchers: Optional[Dict[str, List[str]]] = None
    # {detector: {param: [values...]}}: detector parameters (each combination
    # runs as a separately named variant) and matcher parameters named
    # 'lgm.<key>' / 'ada.<key>' (each combination is a matching pass). See
    # DETECTOR_PARAMS / detector_param_specs().
    detector_params: Optional[Dict[str, Dict[str, List]]] = None

    # Reference DN scale applied before normalisation (1.0 = off). S1 GRD
    # folders used 0.003162 in the S1 pipeline; optical/C1 references use 1.0.
    s1_calibration_factor: float = 1.0

    # Consensus output file suffix (from agdqe_all_v2): '' keeps plain names.
    consensus_filename_suffix: str = ''
    # How chips are judged consistent:
    #   'surface'  within tolerance of a robust smooth surface fitted to all
    #              chips' mean errors (errors that vary across the scene)
    #   'constant' within tolerance of the most common error (the original rule)
    #   'none'     every chip with enough RANSAC inliers is kept
    consensus_model: str = 'surface'
    consensus_surface_degree: str = 'auto'   # 'auto' | 'affine' | 'bilinear' | 'quadratic'

    # ── Large-offset handling (coarse-to-fine) ──────────────────────────────
    # Expected worst-case geolocation error. Reference candidates, and the
    # reference crop, are searched this far beyond the declared overlap.
    max_expected_error_m: float = 50000.0
    # 'auto'   matcher at coarse resolution, phase correlation if that is weak
    # 'matcher' / 'phasecorr' force one; 'manual' uses initial_offset_m;
    # 'none'   windows compared at the same map location (legacy behaviour)
    coarse_method: str = 'auto'
    coarse_resolution_m: float = 60.0
    coarse_max_px: int = 2048
    coarse_min_support: int = 8
    coarse_min_peak: float = 0.03       # phase-correlation response floor
    # ── imcui bridge / GPU ──────────────────────────────────────────────────
    imw_detect_threshold: float = 0.015
    imw_match_threshold: float = 0.2
    min_gpu_free_gb: float = 1.5        # skip a window rather than OOM
    save_match_images: bool = False
    max_match_images: int = 50
    weights_cache_dir: str = ''

    # Manual (dE, dN) in metres, NISAR minus reference, e.g. from RIVAL picks.
    initial_offset_m: Optional[Tuple[float, float]] = None
    # Reference windows are padded by this much around the shifted NISAR window.
    search_margin_m: float = 1500.0
    # Internal distortion: the coarse offset is estimated per grid cell (about
    # one window wide unless coarse_cell_m is set) and interpolated at each
    # window, instead of one translation per image/reference pair.
    coarse_local: bool = True
    coarse_cell_m: Optional[float] = None

    def __post_init__(self):
        if self.smnn_thresholds is None:
            self.smnn_thresholds = [0.90, 0.95, 0.99]
        if self.ransac_methods is None:
            self.ransac_methods = [4]
        if self.ransac_thresholds is None:
            self.ransac_thresholds = [1, 2, 3]
        if self.ransac_confidences is None:
            self.ransac_confidences = [0.91, 0.95, 0.99]
        for m in self.ransac_methods:
            ransac_flag(m)  # fail at config time, not after hours of matching
            if '_' in str(m):
                raise ValueError(f"RANSAC method token {m!r} must not contain '_'")

    def compute_num_features(self, window_size: int) -> int:
        mpx = (window_size ** 2) / 1000000
        n = int(self.keypoint_density * mpx)
        return max(self.min_num_features, min(n, self.max_num_features))


# =============================================================================
# FOOTPRINT HELPERS
# =============================================================================
def parse_pos_list(text: str) -> List[Tuple[float, float]]:
    """gml:posList -> [(lon, lat), ...]. Same rules as DPQED_rival.py:
    NISAR writes comma-separated 'lon lat h' triples; plain GML is one
    whitespace run of triples or pairs."""
    if not text:
        return []
    chunks = [c for c in text.replace('\n', ' ').split(',') if c.strip()]
    if len(chunks) > 1:
        ring = []
        for chunk in chunks:
            parts = chunk.split()
            if len(parts) >= 2:
                ring.append((float(parts[0]), float(parts[1])))
        return ring
    parts = text.split()
    for step in (3, 2):
        if len(parts) >= 3 * step and len(parts) % step == 0:
            return [(float(parts[i]), float(parts[i + 1]))
                    for i in range(0, len(parts), step)]
    return []


def _as_2d_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != 'Polygon':
        return None
    poly = Polygon([(c[0], c[1]) for c in geom.exterior.coords])
    return poly if poly.is_valid else poly.buffer(0)


def _densify_ring(poly: Polygon, per_edge: int = 8) -> Polygon:
    pts = list(poly.exterior.coords)
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        for k in range(per_edge):
            t = k / per_edge
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    out.append(out[0])
    return Polygon(out)


def valid_hull_from_raster(path: str, probe_px: int = 1024,
                           extra_nodata: Tuple = (0,)) -> Optional[Polygon]:
    """Convex hull of the valid pixels of band 1, in the raster's own CRS.

    Read decimated; each row contributes only its first and last valid pixel
    (the only points a convex hull can use). Pixel centres are used and the
    hull is grown by one probe pixel so a decimated read does not trim the
    swath edge."""
    with rt.open(path) as src:
        H, W = src.height, src.width
        scale = max(1.0, max(H, W) / float(probe_px))
        oh, ow = max(2, int(round(H / scale))), max(2, int(round(W / scale)))
        data = src.read(1, out_shape=(oh, ow), resampling=Resampling.nearest)
        valid = np.isfinite(data)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= data != src.nodata
        for v in extra_nodata:
            valid &= data != v
        if not valid.any():
            return None
        rows = np.where(valid.any(axis=1))[0]
        first = valid[rows].argmax(axis=1)
        last = ow - 1 - valid[rows][:, ::-1].argmax(axis=1)
        tfm = src.transform * Affine.scale(W / ow, H / oh)
        pts = []
        for r, c0, c1 in zip(rows, first, last):
            pts.append(tfm * (c0 + 0.5, r + 0.5))
            pts.append(tfm * (c1 + 0.5, r + 0.5))
        hull = MultiPoint(pts).convex_hull
        grow = abs(tfm.a) + abs(tfm.e)
        hull = hull.buffer(grow, join_style=2)
        return _as_2d_polygon(hull)


def polygon_to_lonlat(poly: Polygon, src_crs) -> Polygon:
    crs = PJCRS.from_user_input(src_crs)
    if crs.is_geographic:
        return poly
    tr = Transformer.from_crs(crs, PJCRS.from_epsg(4326), always_xy=True)
    return _as_2d_polygon(transform(tr.transform, _densify_ring(poly)))


# =============================================================================
# NISAR H5 / MET READER
# =============================================================================
class NISARH5Reader:
    # Kept for backward compatibility; discover_scene() searches all of these.
    GROUP_CANDIDATES = [
        'science/SSAR/GSLC/grids/frequencyA',
    ]
    BANDS = ('LSAR', 'SSAR')
    PRODUCTS = ('GSLC', 'GCOV')
    POL_ORDER = ('HH', 'HV', 'VH', 'VV', 'RH', 'RV', 'LH', 'LV')
    # GCOV stores covariance terms; the diagonal ones are backscatter power.
    GCOV_TERMS = {'HH': 'HHHH', 'HV': 'HVHV', 'VH': 'VHVH', 'VV': 'VVVV',
                  'RH': 'RHRH', 'RV': 'RVRV'}

    # ── discovery ────────────────────────────────────────────────────────────
    @staticmethod
    def discover_scene(path: str, band: str = 'auto', frequency: str = 'A') -> Dict:
        """Accepts a scene directory (<name>/<name>.h5, or a directory holding
        exactly one .h5) or a path to the .h5 itself. Sidecars are optional."""
        path = os.path.abspath(path)
        if os.path.isfile(path):
            if not path.lower().endswith(('.h5', '.hdf5')):
                raise ValueError(f'Not an HDF5 file: {path}')
            h5_path = path
        elif os.path.isdir(path):
            base = os.path.basename(os.path.normpath(path))
            h5_path = os.path.join(path, base + '.h5')
            if not os.path.exists(h5_path):
                h5s = sorted(glob.glob(os.path.join(path, '*.h5')) +
                             glob.glob(os.path.join(path, '*.hdf5')))
                if not h5s:
                    raise FileNotFoundError(f'No .h5 file in {path}')
                if len(h5s) > 1:
                    raise ValueError(
                        f'{len(h5s)} .h5 files in {path}; pass the .h5 path '
                        f'itself: {[os.path.basename(p) for p in h5s]}')
                h5_path = h5s[0]
        else:
            raise FileNotFoundError(f'Scene not found: {path}')

        scene_dir = os.path.dirname(h5_path)
        stem = os.path.splitext(os.path.basename(h5_path))[0]

        def first_existing(cands):
            for c in cands:
                if c and os.path.exists(c):
                    return c
            return None

        dir_base = os.path.basename(os.path.normpath(scene_dir))
        met_path = first_existing([
            os.path.join(scene_dir, stem + '.met'),
            os.path.join(scene_dir, dir_base + '.met'),
        ])
        if met_path is None:
            mets = glob.glob(os.path.join(scene_dir, '*.met'))
            met_path = mets[0] if len(mets) == 1 else None
        iso_path = first_existing([
            h5_path + '.iso.xml',
            os.path.join(scene_dir, stem + '.iso.xml'),
        ])

        with hp.File(h5_path, 'r') as f:
            grid = NISARH5Reader._locate_grid(f, band, frequency)
            grp = f[grid['grid_path']]
            epsg = NISARH5Reader._read_epsg(grp)
            pols = NISARH5Reader._pols_in_group(grp, grid['product'])
            shape = None
            if pols:
                shape = tuple(grp[NISARH5Reader._dataset_name(pols[0], grid['product'])].shape)

        if epsg is None and met_path:
            try:
                epsg = int(NISARH5Reader._load_meta_dict(met_path).get('EPSG'))
            except (TypeError, ValueError):
                epsg = None
        if epsg is None:
            raise ValueError(f'No EPSG code in {grid["grid_path"]}/projection '
                             f'and no .met EPSG fallback for {h5_path}')

        return {
            'scene_dir': scene_dir,
            'scene_name': stem,
            'h5_path': h5_path,
            'met_path': met_path,
            'iso_path': iso_path,
            'band': grid['band'],
            'product': grid['product'],
            'frequency': grid['frequency'],
            'grid_path': grid['grid_path'],
            'epsg': int(epsg),
            'pols': pols,
            'shape': shape,
        }

    @staticmethod
    def _locate_grid(h5f, band: str = 'auto', frequency: str = 'A') -> Dict:
        band = (band or 'auto').upper()
        freqs = [frequency.upper()] if frequency else ['A', 'B']
        bands = NISARH5Reader.BANDS if band == 'AUTO' else (band,)
        tried = []
        for b in bands:
            for p in NISARH5Reader.PRODUCTS:
                for fq in freqs:
                    gp = f'science/{b}/{p}/grids/frequency{fq}'
                    tried.append(gp)
                    if gp in h5f and 'xCoordinates' in h5f[gp] and 'yCoordinates' in h5f[gp]:
                        return {'band': b, 'product': p, 'frequency': fq, 'grid_path': gp}
        raise KeyError(f'No geocoded NISAR grid found in {h5f.filename}. Tried: {tried}')

    @staticmethod
    def _find_group(h5f):
        """Backward-compatible: first geocoded grid in the file."""
        return h5f[NISARH5Reader._locate_grid(h5f)['grid_path']]

    @staticmethod
    def _read_epsg(grp) -> Optional[int]:
        if 'projection' not in grp:
            return None
        ds = grp['projection']
        try:
            v = np.asarray(ds[()]).ravel()
            if v.size and np.issubdtype(v.dtype, np.number) and int(v[0]) > 0:
                return int(v[0])
        except Exception:
            pass
        for key in ('epsg_code', 'EPSG', 'epsg'):
            if key in ds.attrs:
                try:
                    return int(np.asarray(ds.attrs[key]).ravel()[0])
                except Exception:
                    pass
        return None

    @staticmethod
    def _dataset_name(pol: str, product: str) -> str:
        return NISARH5Reader.GCOV_TERMS.get(pol, pol) if product == 'GCOV' else pol

    @staticmethod
    def _pols_in_group(grp, product: str) -> List[str]:
        keys = set(grp.keys())
        return [p for p in NISARH5Reader.POL_ORDER
                if NISARH5Reader._dataset_name(p, product) in keys
                and getattr(grp[NISARH5Reader._dataset_name(p, product)], 'ndim', 0) == 2]

    @staticmethod
    def _load_meta_text(met_path: str) -> str:
        with open(met_path, 'r') as f:
            return f.read()

    @staticmethod
    def _load_meta_dict(met_path: str) -> Dict:
        try:
            with open(met_path, 'r') as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except (ValueError, OSError):
            pass
        meta = {}
        with open(met_path, 'r') as f:
            for line in f:
                line = line.strip().strip(',').strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    meta[key.strip().strip('"')] = value.strip().strip('"')
        return meta

    @staticmethod
    def available_pols(scene_info: Dict) -> List[str]:
        """Polarisations actually present in the H5 grid. The .met text is no
        longer consulted: a pol named there but absent from the file used to
        crash later in open_memfile."""
        if scene_info.get('pols') is not None:
            return list(scene_info['pols'])
        with hp.File(scene_info['h5_path'], 'r') as f:
            gp = scene_info.get('grid_path') or NISARH5Reader._locate_grid(f)['grid_path']
            product = scene_info.get('product', 'GSLC')
            return NISARH5Reader._pols_in_group(f[gp], product)

    @staticmethod
    def choose_s1_ref_tag(pol: str) -> str:
        return 'VV' if pol in ('HH', 'VV') else 'VH'

    # ── amplitude ────────────────────────────────────────────────────────────
    @staticmethod
    def _amplitude(block: np.ndarray, product: str) -> np.ndarray:
        dt = block.dtype
        if dt.names:  # compound {r, i}
            names = list(dt.names)
            re_ = block[names[0]].astype(np.float32)
            im_ = block[names[1]].astype(np.float32)
            amp = np.sqrt(re_ * re_ + im_ * im_)
        elif np.iscomplexobj(block):
            amp = np.abs(block).astype(np.float32)
        elif product == 'GCOV':
            amp = np.sqrt(np.clip(block.astype(np.float32), 0, None))
        else:
            amp = np.abs(block.astype(np.float32))
        amp[~np.isfinite(amp)] = 0.0
        return amp

    @staticmethod
    def grid_transform(scene_info: Dict) -> Tuple[Affine, CRS]:
        with hp.File(scene_info['h5_path'], 'r') as f:
            grp = f[scene_info['grid_path']]
            x = np.asarray(grp['xCoordinates'][()], dtype=np.float64)
            y = np.asarray(grp['yCoordinates'][()], dtype=np.float64)
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        tf = Affine.translation(float(x[0]) - dx / 2.0, float(y[0]) - dy / 2.0) * Affine.scale(dx, dy)
        return tf, CRS.from_epsg(int(scene_info['epsg']))

    @staticmethod
    def amplitude_geotiff(scene_info: Dict, pol: str, out_path: str,
                          block_bytes: int = 256 * 1024 * 1024) -> str:
        """Write |z| (GSLC) or sqrt(power) (GCOV) for one pol to a tiled
        GeoTIFF, block-wise. Reused if already written for this H5."""
        stamp_path = out_path + '.json'
        h5_stat = os.stat(scene_info['h5_path'])
        stamp = {'h5': os.path.abspath(scene_info['h5_path']), 'size': h5_stat.st_size,
                 'mtime': int(h5_stat.st_mtime), 'grid': scene_info['grid_path'], 'pol': pol}
        if os.path.exists(out_path) and os.path.exists(stamp_path):
            try:
                with open(stamp_path) as fh:
                    if json.load(fh) == stamp:
                        print(f'[H5] Reusing amplitude raster {out_path}')
                        return out_path
            except (ValueError, OSError):
                pass

        tf, crs = NISARH5Reader.grid_transform(scene_info)
        tmp = out_path + '.partial.tif'
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        t0 = time.time()
        with hp.File(scene_info['h5_path'], 'r') as f:
            grp = f[scene_info['grid_path']]
            name = NISARH5Reader._dataset_name(pol, scene_info.get('product', 'GSLC'))
            if name not in grp:
                raise KeyError(f'Polarization {pol} not present in {scene_info["h5_path"]}. '
                               f'Available: {NISARH5Reader._pols_in_group(grp, scene_info.get("product", "GSLC"))}')
            ds = grp[name]
            H, W = ds.shape
            chunk_rows = ds.chunks[0] if ds.chunks else 256
            rows = max(1, int(block_bytes // max(1, W * ds.dtype.itemsize)))
            rows = max(chunk_rows, (rows // chunk_rows) * chunk_rows)
            profile = dict(driver='GTiff', height=H, width=W, count=1, dtype='float32',
                           crs=crs, transform=tf, nodata=0.0, BIGTIFF='IF_SAFER',
                           TILED='YES', BLOCKXSIZE=512, BLOCKYSIZE=512)
            with rt.open(tmp, 'w', **profile) as dst:
                for r0 in range(0, H, rows):
                    r1 = min(H, r0 + rows)
                    amp = NISARH5Reader._amplitude(ds[r0:r1, :], scene_info.get('product', 'GSLC'))
                    dst.write(amp, 1, window=rasterio.windows.Window(0, r0, W, r1 - r0))
                    report('h5_read', pol=pol, done=r1, total=H)
        os.replace(tmp, out_path)
        with open(stamp_path, 'w') as fh:
            json.dump(stamp, fh)
        print(f'[H5] {pol} amplitude {W}x{H} -> {out_path} ({time.time() - t0:.1f}s)')
        return out_path

    @staticmethod
    def open_memfile(h5_path: str, pol: str):
        """Backward-compatible in-memory variant (whole scene in RAM)."""
        info = NISARH5Reader.discover_scene(h5_path)
        tf, crs = NISARH5Reader.grid_transform(info)
        with hp.File(h5_path, 'r') as f:
            ds = f[info['grid_path']][NISARH5Reader._dataset_name(pol, info['product'])]
            arr = NISARH5Reader._amplitude(ds[()], info['product'])
        memfile = MemoryFile()
        with memfile.open(driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1,
                          dtype='float32', crs=crs, transform=tf, nodata=0.0) as ds_out:
            ds_out.write(arr, 1)
        return memfile

    # ── footprint ────────────────────────────────────────────────────────────
    @staticmethod
    def valid_polygon_from_met(met_path: str):
        meta = NISARH5Reader._load_meta_dict(met_path)
        poly = Polygon([
            (float(meta['ImageULLon']), float(meta['ImageULLat'])),
            (float(meta['ImageURLon']), float(meta['ImageURLat'])),
            (float(meta['ImageLRLon']), float(meta['ImageLRLat'])),
            (float(meta['ImageLLLon']), float(meta['ImageLLLat'])),
            (float(meta['ImageULLon']), float(meta['ImageULLat'])),
        ])
        return meta, poly

    @staticmethod
    def _polygon_from_iso(iso_path: str) -> Optional[Polygon]:
        root = ET.parse(iso_path).getroot()
        for node in root.iter():
            if node.tag.endswith('posList'):
                ring = parse_pos_list(node.text or '')
                if len(ring) >= 3:
                    return _as_2d_polygon(Polygon(ring))
        return None

    @staticmethod
    def _polygon_from_h5(scene_info: Dict) -> Optional[Polygon]:
        with hp.File(scene_info['h5_path'], 'r') as f:
            gp = f'science/{scene_info["band"]}/identification/boundingPolygon'
            if gp not in f:
                return None
            raw = f[gp][()]
        if isinstance(raw, np.ndarray):
            raw = raw.ravel()[0]
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', 'ignore')
        return _as_2d_polygon(shapely.wkt.loads(str(raw).strip()))

    @staticmethod
    def footprint_lonlat(scene_info: Dict, amplitude_tif: Optional[str] = None
                         ) -> Tuple[Polygon, str]:
        """Swath footprint in lon/lat and the source it came from."""
        attempts = []
        if scene_info.get('met_path'):
            try:
                _, poly = NISARH5Reader.valid_polygon_from_met(scene_info['met_path'])
                poly = _as_2d_polygon(poly)
                if poly is not None and poly.area > 0:
                    return poly, 'met'
            except Exception as e:
                attempts.append(f'met: {e}')
        if scene_info.get('iso_path'):
            try:
                poly = NISARH5Reader._polygon_from_iso(scene_info['iso_path'])
                if poly is not None and poly.area > 0:
                    return poly, 'iso.xml'
            except Exception as e:
                attempts.append(f'iso.xml: {e}')
        try:
            poly = NISARH5Reader._polygon_from_h5(scene_info)
            if poly is not None and poly.area > 0:
                return poly, 'h5-boundingPolygon'
        except Exception as e:
            attempts.append(f'boundingPolygon: {e}')
        if amplitude_tif and os.path.exists(amplitude_tif):
            hull = valid_hull_from_raster(amplitude_tif)
            if hull is not None:
                return polygon_to_lonlat(hull, f'EPSG:{scene_info["epsg"]}'), 'valid-data-hull'
        raise RuntimeError('Could not determine the NISAR footprint. ' + '; '.join(attempts))


# =============================================================================
# REFERENCE FETCHER
# =============================================================================
class ReferenceFetcher:
    """Reference rasters overlapping the NISAR swath.

    Candidates come from automatch_refs.scan_reference_folder (RIVAL's rules:
    index-shp / sidecar / degree-tile), are pre-filtered on their declared
    footprint, then confirmed on their valid-pixel hull, since a declared
    footprint (an index box, a degree cell) can include fill."""

    def __init__(self, reference_dir: str, decimation: int = 10, min_area: float = 200.0,
                 mode: Optional[str] = None, fill_values: Tuple = (0, 3)):
        self.reference_dir = reference_dir
        self.decimation = decimation
        self.min_area = min_area
        self.mode = mode
        self.fill_values = tuple(fill_values or ())
        self.catalog: Optional[Dict] = None

    def scan(self) -> Dict:
        if self.catalog is None:
            self.catalog = refs.scan_reference_folder(self.reference_dir, self.mode)
        return self.catalog

    def _valid_footprint(self, tif_path: str) -> Optional[Polygon]:
        """Valid-data hull of a reference raster, in lon/lat."""
        try:
            with rt.open(tif_path) as src:
                crs = src.crs
                probe = max(50, max(src.height, src.width) // max(1, self.decimation))
            if crs is None:
                print(f'[RefFetch] {os.path.basename(tif_path)} has no CRS -- skipped')
                return None
            hull = valid_hull_from_raster(tif_path, probe_px=min(probe, 2048),
                                          extra_nodata=self.fill_values)
            return None if hull is None else polygon_to_lonlat(hull, crs)
        except Exception as e:
            print(f'[RefFetch] ERROR reading valid footprint of {tif_path}: {e}')
            return None

    # Backward-compatible name.
    _get_s1_valid_corners = _valid_footprint

    def fetch_overlapping_references(self, footprint: Polygon, working_crs,
                                     max_error_m: float = 0.0) -> List[Dict]:
        """References that can hold the ground under the NISAR swath.

        With a geolocation error of up to max_error_m, a reference just beside
        the declared footprint may hold the true ground, so candidates are
        tested against the footprint grown by that much. Two regions are kept
        per reference, both in the NISAR UTM CRS:
          intersection_utm  NISAR ground that may have a match in this ref
                            (footprint  AND  ref-valid grown by max_error_m)
          ref_clip_utm      reference ground to keep for matching
                            (ref-valid  AND  footprint grown by max_error_m)"""
        print(f'[RefFetch] Searching in: {self.reference_dir} '
              f'(search buffer {max_error_m / 1000.0:.1f} km)')
        catalog = self.scan()
        nis_crs = f'EPSG:{int(working_crs)}' if str(working_crs).isdigit() else str(working_crs)
        to_utm = Transformer.from_crs(PJCRS('epsg:4326'), PJCRS.from_user_input(nis_crs), always_xy=True)
        to_ll = Transformer.from_crs(PJCRS.from_user_input(nis_crs), PJCRS('epsg:4326'), always_xy=True)

        def utm(poly_ll):
            return _as_2d_polygon(transform(to_utm.transform, _densify_ring(poly_ll)))

        foot_utm = utm(footprint)
        foot_grown_utm = foot_utm.buffer(max_error_m) if max_error_m > 0 else foot_utm
        foot_grown_ll = _as_2d_polygon(transform(to_ll.transform, _densify_ring(
            _as_2d_polygon(foot_grown_utm)))) if max_error_m > 0 else footprint

        overlapping_refs = []
        for tif_path, rec in sorted(catalog['footprints'].items()):
            tif_name = os.path.basename(tif_path)
            declared = _as_2d_polygon(Polygon(rec['ring'])) if len(rec.get('ring') or []) >= 3 else None
            if declared is None or not foot_grown_ll.intersects(declared):
                continue

            print(f'[RefFetch] Candidate: {tif_name} ({rec.get("source")}) - checking valid pixels...')
            valid_ll = self._valid_footprint(tif_path)
            if valid_ll is None or not foot_grown_ll.intersects(valid_ll):
                print(f'[RefFetch] {tif_name}: no valid pixels near the swath -- skipped')
                continue
            valid_utm = utm(valid_ll)
            if valid_utm is None:
                continue
            valid_grown = valid_utm.buffer(max_error_m) if max_error_m > 0 else valid_utm
            inter_utm = _as_2d_polygon(foot_utm.intersection(valid_grown))
            ref_clip = _as_2d_polygon(valid_utm.intersection(foot_grown_utm))
            if inter_utm is None or ref_clip is None:
                continue
            area_km2 = inter_utm.area / 1e6
            ref_km2 = ref_clip.area / 1e6
            if min(area_km2, ref_km2) < self.min_area:
                print(f'[RefFetch] {tif_name}: overlap {min(area_km2, ref_km2):.1f} km2 '
                      f'< min_area -- skipped')
                continue

            overlapping_refs.append({
                'tif_path': tif_path,
                'utm_crs': nis_crs,
                'intersection_utm': inter_utm,
                'ref_clip_utm': ref_clip,
                'area_km2': area_km2,
                'footprint_source': rec.get('source'),
            })
            print(f'[RefFetch] OK {tif_name} | NISAR area {area_km2:.1f} km2, '
                  f'reference area {ref_km2:.1f} km2')

        print(f'[RefFetch] Total confirmed: {len(overlapping_refs)} references '
              f'(of {len(catalog["footprints"])} in the {catalog["mode"]} catalog)')
        return overlapping_refs


# =============================================================================
# INPUT SCENE (NISAR H5 or any georeferenced multi-band raster, e.g. G1A)
# =============================================================================
def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def _crs_string(crs) -> str:
    epsg = crs.to_epsg() if crs is not None else None
    return f'EPSG:{epsg}' if epsg else crs.to_wkt()


class InputScene:
    """The image being assessed, as a set of single-band channels.

      nisar   NISAR GSLC/GCOV H5; channels are polarisations (HH, HV, ...)
      raster  anything rasterio opens (GeoTIFF, VRT, JP2, ...); channels are
              band1..bandN. A raster in lon/lat is warped on the fly to the
              UTM zone of its centre so errors come out in metres.

    Every channel is exposed as (path, band_index) of a raster in the working
    CRS, which is all the preprocessor needs."""

    def __init__(self, path: str, config: 'PipelineConfig'):
        self.path = os.path.abspath(path)
        self.config = config
        self.kind = self._detect_kind(self.path)
        if self.kind == 'nisar':
            self.info = NISARH5Reader.discover_scene(self.path, config.nisar_band,
                                                     config.nisar_frequency)
            self.name = self.info['scene_name']
            self.channels = list(self.info['pols'])
            self.working_crs = f'EPSG:{self.info["epsg"]}'
            self.cache_subdir = f'{self.info["band"]}{self.info["frequency"]}'
            self.raster_path = self.info['h5_path']
            self.native_res = None
        else:
            self.raster_path = self.path
            self.name = os.path.splitext(os.path.basename(self.path))[0]
            with rt.open(self.path) as src:
                if src.crs is None:
                    raise ValueError(f'{self.path} carries no CRS; cannot georeference it')
                n = src.count
                descs = list(src.descriptions or [])
                if src.crs.is_projected:
                    self.working_crs = _crs_string(src.crs)
                else:
                    b = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                    self.working_crs = f'EPSG:{utm_epsg_for((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)}'
                self.native_res = abs(src.transform.a) if src.crs.is_projected else None
                self.source_crs = _crs_string(src.crs)
            self.channels = [f'band{i}' for i in range(1, n + 1)]
            self.band_descriptions = {f'band{i}': (descs[i - 1] if i - 1 < len(descs) else None)
                                      for i in range(1, n + 1)}
            self.cache_subdir = 'raster'
            self.info = {'scene_name': self.name, 'h5_path': self.path, 'met_path': None}

    @staticmethod
    def _detect_kind(path: str) -> str:
        p = path
        if os.path.isdir(p):
            return 'nisar'  # scene directories are the NISAR convention
        if p.lower().endswith(('.h5', '.hdf5', '.he5')):
            try:
                with hp.File(p, 'r') as f:
                    if any(f'science/{b}' in f for b in NISARH5Reader.BANDS):
                        return 'nisar'
            except OSError:
                pass
        return 'raster'

    def describe(self) -> Dict:
        d = {'kind': self.kind, 'name': self.name, 'path': self.path,
             'channels': self.channels, 'working_crs': self.working_crs}
        if self.kind == 'nisar':
            d.update({k: self.info.get(k) for k in ('band', 'product', 'frequency',
                                                    'grid_path', 'shape', 'met_path', 'iso_path')})
        else:
            d.update({'source_crs': self.source_crs, 'native_res': self.native_res,
                      'band_descriptions': self.band_descriptions})
        return d

    def cache_dir(self, temp_root: str) -> str:
        return os.path.join(temp_root, self.name, self.cache_subdir)

    def identity(self) -> Dict:
        st = os.stat(self.raster_path)
        return {'path': self.raster_path, 'size': st.st_size, 'mtime': int(st.st_mtime),
                'kind': self.kind, 'grid': self.info.get('grid_path')}

    def channel_raster(self, channel: str, cache_dir: str) -> Tuple[str, int]:
        if channel not in self.channels:
            raise KeyError(f'{channel} not in {self.name}; available: {self.channels}')
        if self.kind == 'nisar':
            tif = os.path.join(cache_dir, f'{self.name}_{channel}_amp.tif')
            return NISARH5Reader.amplitude_geotiff(self.info, channel, tif), 1
        return self.raster_path, int(channel[len('band'):])

    def footprint_lonlat(self, channel_path: str, band_index: int) -> Tuple[Polygon, str]:
        if self.kind == 'nisar':
            return NISARH5Reader.footprint_lonlat(self.info, channel_path)
        # A sidecar beside the raster (RIVAL's rules) states the swath; else
        # the valid-data hull of the band being matched.
        folder = os.path.dirname(self.path)
        base = os.path.basename(self.path)
        try:
            for name in sorted(os.listdir(folder)):
                kind = refs.is_meta_file(name)
                if not kind:
                    continue
                stem = refs.meta_base_stem(name)
                if stem and refs.match_raster([base], stem) != base:
                    continue
                with open(os.path.join(folder, name), 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                rec = (refs.parse_meta_iso_xml(content, name) if kind == 'iso-xml'
                       else refs.parse_meta_text(content, name))
                if rec and len(rec.get('ring') or []) >= 3:
                    return _as_2d_polygon(Polygon(rec['ring'])), f'sidecar ({name})'
        except OSError:
            pass
        with rt.open(self.path) as src:
            crs = src.crs
        hull = self._band_hull(band_index)
        if hull is None:
            raise RuntimeError(f'{self.name} band{band_index} has no valid pixels')
        return polygon_to_lonlat(hull, crs), 'valid-data-hull'

    def _band_hull(self, band_index: int) -> Optional[Polygon]:
        with rt.open(self.path) as src:
            H, W = src.height, src.width
            scale = max(1.0, max(H, W) / 1024.0)
            oh, ow = max(2, int(H / scale)), max(2, int(W / scale))
            data = src.read(band_index, out_shape=(oh, ow), resampling=Resampling.nearest)
            valid = np.isfinite(data.astype(np.float64))
            if src.nodata is not None and np.isfinite(src.nodata):
                valid &= data != src.nodata
            valid &= data != 0
            if not valid.any():
                return None
            rows = np.where(valid.any(axis=1))[0]
            first = valid[rows].argmax(axis=1)
            last = ow - 1 - valid[rows][:, ::-1].argmax(axis=1)
            tfm = src.transform * Affine.scale(W / ow, H / oh)
            pts = []
            for r, c0, c1 in zip(rows, first, last):
                pts += [tfm * (c0 + 0.5, r + 0.5), tfm * (c1 + 0.5, r + 0.5)]
            return _as_2d_polygon(MultiPoint(pts).convex_hull.buffer(abs(tfm.a) + abs(tfm.e), join_style=2))

    def open_working(self, channel_path: str):
        """rasterio dataset of the channel raster in the working CRS."""
        src = rt.open(channel_path)
        if src.crs is not None and CRS.from_user_input(self.working_crs) == src.crs:
            return src
        from rasterio.vrt import WarpedVRT
        print(f'[Input] warping {os.path.basename(channel_path)} {src.crs} -> {self.working_crs}')
        return WarpedVRT(src, crs=self.working_crs, resampling=Resampling.bilinear)


# =============================================================================
# PREPROCESSOR
# =============================================================================
def pair_cache_key(scene: 'InputScene', channel: str, reference_dir: str,
                   config: 'PipelineConfig') -> str:
    blob = json.dumps({
        'scene': scene.identity(), 'channel': channel,
        'refs': os.path.abspath(reference_dir) if reference_dir else '',
        'ref_band': config.reference_band, 'fill': list(config.reference_fill_values or ()),
        'res': config.target_resolution, 'min_area': config.min_area,
        'max_err': config.max_expected_error_m, 'v': 3,
    }, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class DiskBasedPreprocessor:
    def __init__(self, temp_dir: str, target_resolution: Optional[int] = None,
                 reference_band: int = 1, fill_values: Tuple = (0, 3)):
        self.temp_dir = temp_dir
        self.target_resolution = target_resolution
        self.reference_band = int(reference_band or 1)
        self.fill_values = tuple(fill_values or ())
        os.makedirs(temp_dir, exist_ok=True)

    def _clip_single_pair(
        self,
        nisar_src,
        nisar_nodata_val,
        s1_ref: dict,
        pair_id: int,
        nisar_pol: str,
        s1_ref_tag: str,
        band_index: int = 1,
    ) -> Optional[Dict]:
        try:
            intersection_utm = s1_ref['intersection_utm']
            utm_crs = s1_ref['utm_crs']

            if str(nisar_src.crs) != str(utm_crs) and CRS.from_user_input(utm_crs) != nisar_src.crs:
                print(f'[WARN] CRS mismatch: nisar_src={nisar_src.crs}, utm={utm_crs}')
                to_nisar_crs = Transformer.from_crs(utm_crs, str(nisar_src.crs), always_xy=True)
                intersection_utm = shapely.ops.transform(to_nisar_crs.transform, intersection_utm)
                utm_crs = str(nisar_src.crs)

            nisar_arr, nisar_transform = rio_mask(
                nisar_src,
                [mapping(intersection_utm)],
                crop=True,
                nodata=nisar_nodata_val,
                all_touched=True,
                indexes=[band_index],
            )
            crop_transform = nisar_transform

            if self.target_resolution:
                src_res = abs(nisar_src.transform.a)
                scale = src_res / self.target_resolution
                src_res_y = abs(nisar_src.transform.e)
                scale_y = src_res_y / self.target_resolution
                new_h = max(1, int(nisar_arr.shape[1] * scale_y))
                new_w = max(1, int(nisar_arr.shape[2] * scale))
                new_transform = from_origin(
                    nisar_transform.c,
                    nisar_transform.f,
                    self.target_resolution,
                    self.target_resolution,
                )

                resampled = np.zeros((nisar_arr.shape[0], new_h, new_w), dtype=nisar_arr.dtype)
                warp_reproject(
                    source=nisar_arr,
                    destination=resampled,
                    src_transform=nisar_transform,
                    src_crs=nisar_src.crs,
                    dst_transform=new_transform,
                    dst_crs=nisar_src.crs,
                    resampling=Resampling.lanczos,
                    src_nodata=nisar_nodata_val,
                    dst_nodata=nisar_nodata_val,
                )
                nisar_arr = resampled
                nisar_transform = new_transform

            s1 = rxr.open_rasterio(s1_ref['tif_path'], masked=True)
            n_bands = int(s1.sizes.get('band', 1))
            if self.reference_band > n_bands:
                raise ValueError(f'reference_band={self.reference_band} but '
                                 f'{os.path.basename(s1_ref["tif_path"])} has {n_bands} band(s)')
            s1 = s1.isel(band=[self.reference_band - 1]).astype('float32')
            # Declared nodata is already NaN (masked=True); RIVAL's fill values
            # join it so reprojection cannot blend fill into valid edge pixels.
            if self.fill_values:
                s1 = s1.where(~s1.isin(list(self.fill_values)))
            s1.rio.write_nodata(np.nan, encoded=False, inplace=True)
            s1_nodata_val = np.nan

            # Crop in the reference's own CRS before reprojecting: reprojecting
            # a whole S1 scene only to clip most of it away was the slowest
            # step of preprocessing.
            ref_clip_utm = s1_ref.get('ref_clip_utm') or intersection_utm
            try:
                minx, miny, maxx, maxy = ref_clip_utm.buffer(2000.0).bounds
                s1 = s1.rio.clip_box(minx, miny, maxx, maxy, crs=utm_crs)
            except Exception as e:
                print(f'[Preprocessor] pre-crop skipped for pair {pair_id}: {e}')

            # Reproject onto a grid SNAPPED to the input pixel grid (same
            # origin modulo one pixel). With an arbitrary sub-pixel phase
            # between the two grids, integer-precision keypoints quantise the
            # measured offset to one of two values straddling the truth.
            res = self.target_resolution or abs(nisar_transform.a)
            gx0, gy0 = nisar_transform.c, nisar_transform.f
            minx, miny, maxx, maxy = ref_clip_utm.bounds
            ax0 = gx0 + math.floor((minx - gx0) / res) * res
            ay1 = gy0 + math.ceil((maxy - gy0) / res) * res
            out_w = max(1, int(math.ceil((maxx - ax0) / res)))
            out_h = max(1, int(math.ceil((ay1 - miny) / res)))
            reproj_kwargs = {
                'dst_crs': utm_crs,
                'nodata': s1_nodata_val,
                'resampling': Resampling.bilinear,
                'shape': (out_h, out_w),
                'transform': Affine(res, 0.0, ax0, 0.0, -res, ay1),
            }

            s1_utm = s1.rio.reproject(**reproj_kwargs)
            s1_cropped = s1_utm.rio.clip(
                [mapping(ref_clip_utm)],
                crs=utm_crs,
                all_touched=True,
                drop=True,
            ).compute()

            nisar_out = os.path.join(self.temp_dir, f'pair{pair_id:03d}_nisar.tif')
            nisar_meta = nisar_src.meta.copy()
            nisar_meta.update({
                'height': nisar_arr.shape[1],
                'width': nisar_arr.shape[2],
                'transform': nisar_transform,
                'driver': 'GTiff',
                'BIGTIFF': 'IF_SAFER',
                'TILED': 'YES',
                'BLOCKXSIZE': 512,
                'BLOCKYSIZE': 512,
                'nodata': nisar_nodata_val,
                'count': nisar_arr.shape[0],
            })
            with rt.open(nisar_out, 'w', **nisar_meta) as dst:
                dst.write(nisar_arr)

            s1_out = os.path.join(self.temp_dir, f'pair{pair_id:03d}_s1.tif')
            s1_cropped.rio.write_nodata(s1_nodata_val, inplace=True)
            s1_cropped.rio.to_raster(
                s1_out,
                driver='GTiff',
                BIGTIFF='IF_SAFER',
                TILED='YES',
                BLOCKXSIZE=512,
                BLOCKYSIZE=512,
            )

            nisar_res = abs(nisar_transform.a)
            s1_res = abs(float(s1_cropped.rio.resolution()[0]))
            scale_factor = nisar_res / s1_res

            # Crop origin in ORIGINAL H5 pixels (row = scan, col = pix).
            row_off = int(round((crop_transform.f - nisar_src.transform.f) / nisar_src.transform.e))
            col_off = int(round((crop_transform.c - nisar_src.transform.c) / nisar_src.transform.a))

            meta_out = os.path.join(self.temp_dir, f'pair{pair_id:03d}_meta.json')
            pair_meta = {
                'nisar_path': nisar_out,
                's1_path': s1_out,
                's1_source': s1_ref['tif_path'],
                'reference_path': s1_ref['tif_path'],
                'reference_band': self.reference_band,
                'footprint_source': s1_ref.get('footprint_source'),
                'pair_id': pair_id,
                'utm_crs': str(utm_crs),
                'bounds': list(intersection_utm.bounds),
                'area_km2': float(s1_ref.get('area_km2', intersection_utm.area / 1e6)),
                'scale_factor': scale_factor,
                'scale_per_pix': scale_factor,
                'nisar_nodata': float(nisar_nodata_val),
                's1_nodata': float(s1_nodata_val),
                'x01': float(nisar_transform.c),
                'y01': float(nisar_transform.f),
                'xres1': float(nisar_transform.a),
                'yres1': float(nisar_transform.e),
                'nisar_src_xres': float(nisar_src.transform.a),
                'nisar_src_yres': float(nisar_src.transform.e),
                'nisar_band_index': band_index,
                'nisar_crop_row_offset': row_off,
                'nisar_crop_col_offset': col_off,
                'nisar_pol': nisar_pol,
                's1_ref_tag': s1_ref_tag,
            }

            with rt.open(s1_out) as s1src:
                pair_meta.update({
                    'x02': float(s1src.transform.c),
                    'y02': float(s1src.transform.f),
                    'xres2': float(s1src.transform.a),
                    'yres2': float(s1src.transform.e),
                    'nisar_shape': [int(nisar_arr.shape[0]), int(nisar_arr.shape[1]), int(nisar_arr.shape[2])],
                    's1_shape': [int(s1src.count), int(s1src.height), int(s1src.width)],
                })

            with open(meta_out, 'w') as f:
                json.dump(pair_meta, f, indent=2)

            del s1, s1_utm, s1_cropped, nisar_arr, nisar_transform, nisar_meta
            xr.backends.file_manager.FILE_CACHE.clear()
            gc.collect()

            print(f'[Preprocessor] Pair {pair_id}: metadata saved -> {meta_out}')
            return pair_meta

        except Exception as e:
            print(f'[Preprocessor] ERROR in pair {pair_id}: {e}')
            traceback.print_exc()
            return None

    def create_all_pairs(
        self,
        scene: 'InputScene',
        s1_references: list,
        nisar_pol: str,
        s1_ref_tag: str,
        channel_path: str,
        band_index: int = 1,
        cache_key: str = '',
    ) -> list:
        # Stale pairs from an earlier run must not be mixed with this one.
        for old in glob.glob(os.path.join(self.temp_dir, 'pair*_*')):
            try:
                os.remove(old)
            except OSError:
                pass

        disk_pairs = []
        nisar_src = scene.open_working(channel_path)
        try:
            nodata = nisar_src.nodata
            nisar_nodata_val = nodata if nodata is not None and np.isfinite(nodata) else 0.0
            print(
                f'[Preprocessor] {scene.name} {nisar_pol}: {nisar_src.width}x{nisar_src.height}, '
                f'res={abs(nisar_src.transform.a):.2f}, nodata={nisar_nodata_val}, ref={s1_ref_tag}'
            )
            for idx, s1_ref in enumerate(s1_references, start=1):
                report('preprocess', pol=nisar_pol, done=idx - 1, total=len(s1_references))
                pair = self._clip_single_pair(
                    nisar_src, nisar_nodata_val, s1_ref, idx, nisar_pol, s1_ref_tag, band_index
                )
                if pair:
                    pair['nisar_input'] = scene.raster_path
                    pair['nisar_met'] = scene.info.get('met_path')
                    pair['scene_name'] = scene.name
                    pair['cache_key'] = cache_key
                    with open(os.path.join(self.temp_dir, f'pair{idx:03d}_meta.json'), 'w') as f:
                        json.dump(pair, f, indent=2)
                    disk_pairs.append(pair)

                gc.collect()
                xr.backends.file_manager.FILE_CACHE.clear()
                print(f'[Memory] Pair {idx} done, cache cleared')
        finally:
            nisar_src.close()
        report('preprocess', pol=nisar_pol, done=len(s1_references), total=len(s1_references))
        return disk_pairs

    def load_existing_pairs(self, cache_key: Optional[str] = None) -> Optional[list]:
        metas = sorted(glob.glob(os.path.join(self.temp_dir, 'pair*_meta.json')))
        if not metas:
            return None

        pairs = []
        for meta_file in metas:
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            except (ValueError, OSError):
                print(f'[Cache] {os.path.basename(meta_file)} unreadable - re-running coregistration.')
                return None
            if cache_key is not None and meta.get('cache_key') != cache_key:
                print(f'[Cache] {self.temp_dir} belongs to a different scene/config - rebuilding.')
                return None
            if not all(os.path.exists(meta.get(k, '')) for k in ('nisar_path', 's1_path')):
                print(f'[Cache] Pair {meta.get("pair_id")} incomplete - re-running coregistration.')
                return None
            pairs.append(meta)

        print(f'[Cache] All {len(pairs)} pairs loaded from {self.temp_dir}')
        return pairs


# =============================================================================
# COARSE ALIGNMENT (km-scale offsets)
# =============================================================================
def field_offset(field: Dict, x: float, y: float) -> Tuple[float, float]:
    """(dx, dy) of a coarse offset field at map point (x, y): bilinear between
    cell centres, held constant beyond the outermost centres."""
    nx, ny = int(field['nx']), int(field['ny'])
    gdx, gdy = np.asarray(field['dx'], dtype=float), np.asarray(field['dy'], dtype=float)
    fx = (x - field['x0']) / field['cw'] - 0.5
    fy = (field['y1'] - y) / field['ch'] - 0.5
    fx = min(max(fx, 0.0), nx - 1.0)
    fy = min(max(fy, 0.0), ny - 1.0)
    i0, j0 = int(math.floor(fx)), int(math.floor(fy))
    i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
    tx, ty = fx - i0, fy - j0

    def interp(g):
        top = g[j0, i0] * (1 - tx) + g[j0, i1] * tx
        bot = g[j1, i0] * (1 - tx) + g[j1, i1] * tx
        return float(top * (1 - ty) + bot * ty)
    return interp(gdx), interp(gdy)


class CoarseAligner:
    """Estimate the (dE, dN) translation, NISAR/input minus reference, of one
    pair at coarse resolution, so fine windows can be read from the right
    reference ground even when the error is tens of kilometres.

    Returns {'dx', 'dy', 'method', 'support', 'n', 'peak'}; dx/dy in metres."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ── reading ──────────────────────────────────────────────────────────────
    def _coarse_read(self, path: str, res: float):
        with rt.open(path) as src:
            f = max(1.0, res / abs(src.transform.a))
            oh, ow = max(8, int(src.height / f)), max(8, int(src.width / f))
            data = src.read(1, out_shape=(oh, ow), resampling=Resampling.average,
                            masked=False).astype(np.float32)
            tfm = src.transform * Affine.scale(src.width / ow, src.height / oh)
            nodata = src.nodata
        bad = ~np.isfinite(data)
        if nodata is not None and np.isfinite(nodata):
            bad |= data == nodata
        data[bad] = np.nan
        return data, tfm

    def _resolution(self, pair: Dict) -> float:
        base = max(abs(pair['xres1']), abs(pair['xres2']))
        res = max(self.config.coarse_resolution_m, 2.0 * base)
        # keep the larger image under coarse_max_px on its long side
        longest = max(abs(pair['bounds'][2] - pair['bounds'][0]),
                      abs(pair['bounds'][3] - pair['bounds'][1])) + 2 * self.config.max_expected_error_m
        return max(res, longest / float(self.config.coarse_max_px))

    # ── estimators ───────────────────────────────────────────────────────────
    def _by_matcher(self, pair: Dict, matcher, res: float) -> Optional[Dict]:
        a, ta = self._coarse_read(pair['nisar_path'], res)
        b, tb = self._coarse_read(pair['s1_path'], res)
        meta = dict(pair)
        meta.update({'x01': ta.c, 'y01': ta.f, 'xres1': ta.a, 'yres1': ta.e,
                     'x02': tb.c, 'y02': tb.f, 'xres2': tb.a, 'yres2': tb.e,
                     'nisar_src_xres': ta.a, 'nisar_src_yres': ta.e,
                     'nisar_crop_row_offset': 0, 'nisar_crop_col_offset': 0})
        name, param = matcher.matcher_runs()[0]
        if name == 'smnn':
            param = max(self.config.smnn_thresholds)
        with th.inference_mode():
            rec = matcher._process_single_window(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0),
                                                 0, 0, 0, 0, meta, name, param,
                                                 nisar_nodata=0.0, s1_nodata=0.0)
        if not rec or len(rec['X1']) < 3:
            return None
        x = np.asarray(rec['X1'], dtype=np.float64)
        y = np.asarray(rec['Y1'], dtype=np.float64)
        dx = x - np.asarray(rec['X2'], dtype=np.float64)
        dy = y - np.asarray(rec['Y2'], dtype=np.float64)
        est = self._mode_translation(dx, dy, bin_m=3.0 * res)
        if est is not None:
            # every coarse correspondence, for the local offset field
            est['samples'] = np.column_stack([x, y, dx, dy])
        return est

    def _mode_translation(self, dx, dy, bin_m: float) -> Optional[Dict]:
        """Densest (dx, dy) cluster: 2-D histogram peak, then the median of the
        matches within two bins of it. Random matches spread out; the true
        offset piles up."""
        lim = self.config.max_expected_error_m + bin_m
        keep = (np.abs(dx) <= lim) & (np.abs(dy) <= lim)
        dx, dy = dx[keep], dy[keep]
        if len(dx) < 3:
            return None
        edges = np.arange(-lim, lim + bin_m, bin_m)
        hist, ex, ey = np.histogram2d(dx, dy, bins=[edges, edges])
        i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
        cx, cy = (ex[i] + ex[i + 1]) / 2.0, (ey[j] + ey[j + 1]) / 2.0
        near = (np.abs(dx - cx) <= 2 * bin_m) & (np.abs(dy - cy) <= 2 * bin_m)
        return {'dx': float(np.median(dx[near])), 'dy': float(np.median(dy[near])),
                'support': int(near.sum()), 'n': int(len(dx)), 'peak': None}

    @staticmethod
    def _edges(img: np.ndarray) -> np.ndarray:
        valid = np.isfinite(img)
        x = np.where(valid, img, np.nan)
        x = np.log1p(np.clip(x, 0, None)) if np.nanmin(x) >= 0 else x
        med = np.nanmedian(x) if valid.any() else 0.0
        x = np.where(valid, x, med).astype(np.float32)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        # zero the gradient along the fill boundary: it is not ground
        edge = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
        g[edge] = 0.0
        g -= g[~edge].mean() if (~edge).any() else 0.0
        g[edge] = 0.0
        return g

    def _by_phase_correlation(self, pair: Dict, res: float) -> Optional[Dict]:
        """Both images placed on one north-up grid over their union, gradient
        magnitude, then cv2.phaseCorrelate. Modality-agnostic enough for SAR or
        optical against an optical reference at coarse scale."""
        a, ta = self._coarse_read(pair['nisar_path'], res)
        b, tb = self._coarse_read(pair['s1_path'], res)
        x0 = min(ta.c, tb.c)
        y0 = max(ta.f, tb.f)
        x1 = max(ta.c + a.shape[1] * ta.a, tb.c + b.shape[1] * tb.a)
        y1 = min(ta.f + a.shape[0] * ta.e, tb.f + b.shape[0] * tb.e)
        W = int(math.ceil((x1 - x0) / res))
        H = int(math.ceil((y0 - y1) / res))
        if W < 16 or H < 16:
            return None
        grid = from_origin(x0, y0, res, res)

        def place(img, tfm):
            out = np.full((H, W), np.nan, np.float32)
            warp_reproject(source=img, destination=out, src_transform=tfm,
                           src_crs=pair['utm_crs'], dst_transform=grid, dst_crs=pair['utm_crs'],
                           resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
            return out

        ga, gb = self._edges(place(a, ta)), self._edges(place(b, tb))
        win = cv2.createHanningWindow((W, H), cv2.CV_32F)
        (sx, sy), peak = cv2.phaseCorrelate(gb, ga, win)
        # ga (input) is gb (reference) moved by (sx, sy) pixels: the input
        # shows a feature (sx, sy) px right/down of where the reference does.
        return {'dx': float(sx * res), 'dy': float(-sy * res),
                'support': None, 'n': None, 'peak': float(peak)}

    # ── local offset field ───────────────────────────────────────────────────
    def _cell_m(self, pair: Dict) -> float:
        if self.config.coarse_cell_m:
            return float(self.config.coarse_cell_m)
        return float(self.config.window_size) * abs(pair['xres1'])

    def _phasecorr_samples(self, pair: Dict, res: float, dx0: float, dy0: float) -> np.ndarray:
        """Local offsets by tiled phase correlation, after moving the reference
        by the global offset onto the input's coarse grid: each tile only has to
        find a small residual. Rows (x, y, dx, dy) at tile centres."""
        a, ta = self._coarse_read(pair['nisar_path'], res)
        b, tb = self._coarse_read(pair['s1_path'], res)
        H, W = a.shape
        bb = np.full((H, W), np.nan, np.float32)
        warp_reproject(source=b, destination=bb, src_transform=Affine.translation(dx0, dy0) * tb,
                       src_crs=pair['utm_crs'], dst_transform=ta, dst_crs=pair['utm_crs'],
                       resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        ga, gb = self._edges(a), self._edges(bb)
        va, vb = np.isfinite(a), np.isfinite(bb)
        T = int(max(64, min(256, round(self._cell_m(pair) / res))))
        T = min(T, H, W)
        if T < 32:
            return np.zeros((0, 4))
        step = max(16, T // 2)
        win = cv2.createHanningWindow((T, T), cv2.CV_32F)
        out = []
        for r0 in range(0, H - T + 1, step):
            for c0 in range(0, W - T + 1, step):
                if va[r0:r0 + T, c0:c0 + T].mean() < 0.5 or vb[r0:r0 + T, c0:c0 + T].mean() < 0.5:
                    continue
                (sx, sy), peak = cv2.phaseCorrelate(gb[r0:r0 + T, c0:c0 + T].copy(),
                                                    ga[r0:r0 + T, c0:c0 + T].copy(), win)
                if peak < self.config.coarse_min_peak or max(abs(sx), abs(sy)) > T / 4.0:
                    continue
                xc, yc = ta * (c0 + T / 2.0, r0 + T / 2.0)
                out.append((xc, yc, dx0 + sx * res, dy0 - sy * res))
        return np.asarray(out, dtype=np.float64).reshape(-1, 4)

    def _offset_field(self, pair: Dict, samples: Optional[np.ndarray], res: float,
                      min_support: int) -> Optional[Dict]:
        """Grid of local (dx, dy) over the input crop: each cell takes the
        robust mode of the samples in and around it (cells overlap by half a
        cell each side); cells without enough support copy the nearest cell
        that has it. JSON-serialisable; read with field_offset()."""
        if samples is None or len(samples) == 0:
            return None
        H, W = int(pair['nisar_shape'][1]), int(pair['nisar_shape'][2])
        x0, y1 = float(pair['x01']), float(pair['y01'])
        x1 = x0 + W * float(pair['xres1'])
        y0 = y1 + H * float(pair['yres1'])
        cell = self._cell_m(pair)
        nx = max(1, int(round((x1 - x0) / cell)))
        ny = max(1, int(round((y1 - y0) / cell)))
        cw, ch = (x1 - x0) / nx, (y1 - y0) / ny
        gdx = np.full((ny, nx), np.nan)
        gdy = np.full((ny, nx), np.nan)
        sup = np.zeros((ny, nx), dtype=int)
        xs, ys = samples[:, 0], samples[:, 1]
        for j in range(ny):
            for i in range(nx):
                cx, cy = x0 + (i + 0.5) * cw, y1 - (j + 0.5) * ch
                sel = (np.abs(xs - cx) <= cw) & (np.abs(ys - cy) <= ch)
                n = int(sel.sum())
                if n < max(1, min_support):
                    continue
                if n >= 3:
                    est = self._mode_translation(samples[sel, 2], samples[sel, 3], bin_m=3.0 * res)
                else:
                    est = {'dx': float(np.median(samples[sel, 2])), 'dy': float(np.median(samples[sel, 3])),
                           'support': n, 'n': n}
                if est and est['support'] >= max(1, min_support) and est['support'] >= 0.2 * est['n']:
                    gdx[j, i], gdy[j, i], sup[j, i] = est['dx'], est['dy'], est['support']
        valid = np.isfinite(gdx)
        if not valid.any():
            return None
        filled = ~valid
        if filled.any():
            vj, vi = np.nonzero(valid)
            for j, i in zip(*np.nonzero(filled)):
                k = int(np.argmin((vj - j) ** 2 + (vi - i) ** 2))
                gdx[j, i], gdy[j, i] = gdx[vj[k], vi[k]], gdy[vj[k], vi[k]]
        return {'x0': x0, 'y1': y1, 'cw': cw, 'ch': ch, 'nx': nx, 'ny': ny,
                'dx': np.round(gdx, 3).tolist(), 'dy': np.round(gdy, 3).tolist(),
                'support': sup.tolist(), 'filled': filled.tolist()}

    @staticmethod
    def _field_summary(field: Optional[Dict]) -> str:
        if not field:
            return ''
        dx, dy = np.asarray(field['dx']), np.asarray(field['dy'])
        return (f'field {field["ny"]}x{field["nx"]} cells, dE {dx.min():.0f}..{dx.max():.0f} m, '
                f'dN {dy.min():.0f}..{dy.max():.0f} m, {int(np.sum(field["filled"]))} filled')

    # ── public ───────────────────────────────────────────────────────────────
    def estimate(self, pair: Dict, matcher=None) -> Dict:
        cfg = self.config
        method = (cfg.coarse_method or 'auto').lower()
        zero = {'dx': 0.0, 'dy': 0.0, 'support': None, 'n': None, 'peak': None}
        if method == 'none':
            return {**zero, 'method': 'none'}
        if method == 'manual':
            if cfg.initial_offset_m is None:
                raise ValueError("coarse_method='manual' needs initial_offset_m=(dE, dN)")
            return {**zero, 'dx': float(cfg.initial_offset_m[0]),
                    'dy': float(cfg.initial_offset_m[1]), 'method': 'manual'}

        res = self._resolution(pair)
        tried = []
        if method in ('auto', 'matcher') and matcher is not None:
            try:
                est = self._by_matcher(pair, matcher, res)
                samples = est.pop('samples', None) if est else None
                ok = est is not None and est['support'] >= cfg.coarse_min_support \
                    and est['support'] >= 0.2 * est['n']
                # With internal distortion the global mode can be weak while each
                # region is consistent, so a good local field also counts.
                field = (self._offset_field(pair, samples, res, cfg.coarse_min_support)
                         if cfg.coarse_local and est is not None else None)
                tried.append(f"matcher: {est if est else 'no matches'}")
                if ok or field is not None:
                    out = {**est, 'method': f'matcher@{res:.0f}m', 'field': field}
                    if not ok:  # report the field's centre value as the pair offset
                        out['dx'] = float(np.median(np.asarray(field['dx'])))
                        out['dy'] = float(np.median(np.asarray(field['dy'])))
                    return out
            except ModelLoadError:
                raise
            except Exception as e:
                tried.append(f'matcher: {type(e).__name__}: {e}')
        if method in ('auto', 'phasecorr'):
            try:
                est = self._by_phase_correlation(pair, res)
                tried.append(f"phasecorr: {est}")
                if est is not None and est['peak'] >= cfg.coarse_min_peak:
                    field = None
                    if cfg.coarse_local:
                        samples = self._phasecorr_samples(pair, res, est['dx'], est['dy'])
                        field = self._offset_field(pair, samples, res, 1)
                    return {**est, 'method': f'phasecorr@{res:.0f}m', 'field': field}
            except Exception as e:
                tried.append(f'phasecorr: {type(e).__name__}: {e}')
        if cfg.initial_offset_m is not None:
            print(f'[Coarse] pair {pair["pair_id"]}: automatic estimate failed; '
                  f'using initial_offset_m={cfg.initial_offset_m}')
            return {**zero, 'dx': float(cfg.initial_offset_m[0]),
                    'dy': float(cfg.initial_offset_m[1]), 'method': 'manual-fallback'}
        print(f'[Coarse] pair {pair["pair_id"]}: no reliable coarse offset '
              f'({"; ".join(tried)}); windows compared in place')
        return {**zero, 'method': 'failed'}


# =============================================================================
# BASE MATCHER
# =============================================================================
class BaseMatcher(ABC):

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = device
        self.use_amp = config.use_amp and th.cuda.is_available()
        self._lgm_models: Dict = {}
        self.adalam_config = self._setup_adalam_config()
        # Set by build_variants(): the registry name the user selected, the
        # variant's name suffix, and per-matcher parameter grids
        # ({'lgm': {'filter_threshold': [0.1, 0.2]}, 'ada': {...}}).
        self.registry_name: Optional[str] = None
        self.variant_suffix: str = ''
        self.matcher_grid: Dict[str, Dict[str, List]] = {}

    def _setup_adalam_config(self) -> Dict:
        cfg = KF.adalam.get_adalam_default_config()
        cfg['force_seed_mnn'] = self.config.adalam_force_seed_mnn
        cfg['search_expansion'] = self.config.adalam_search_expansion
        cfg['ransac_iters'] = self.config.adalam_ransac_iters
        cfg['min_confidence'] = self.config.adalam_min_confidence
        cfg['refit'] = self.config.adalam_refit
        cfg['device'] = self.device
        return cfg

    @staticmethod
    def _calibrate_s1_dn(arr: np.ndarray, nodata=None, factor: float = 0.003162) -> np.ndarray:
        """ Convert S1 GRD DN to signma-n0 amp"""
        out = arr.astype(np.float32)
        if factor == 1.0:
            return out
        valid = (out != nodata) if nodata is not None else (out>0)
        out[valid] = out[valid] * factor
        return out

    @staticmethod
    def _norm_img(img: np.ndarray, nodata=None, plo: float = 1, phi: float = 99,
                  return_mask: bool = False):
        x = img.astype(np.float32)
        if nodata is not None:
            x[x == nodata] = np.nan
        x[x > 1e10] = np.nan
        x[x < -1e10] = np.nan

        # Build valid mask BEFORE any modification
        valid_mask = ~np.isnan(x)  # [H, W] bool

        valid = x[valid_mask]
        if valid.size == 0 or valid.max() == valid.min():
            out = np.zeros_like(x)
            return (out, valid_mask) if return_mask else out

        log_mask = valid_mask & (x > 0)
        x[log_mask] = np.log1p(x[log_mask])

        valid = x[valid_mask]
        lo = np.percentile(valid, plo)
        hi = np.percentile(valid, phi)
        if hi == lo:
            out = np.zeros_like(x)
            return (out, valid_mask) if return_mask else out

        y = np.clip(x, lo, hi)
        y = (y - lo) / (hi - lo + 1e-6)
        y[~valid_mask] = 0.0  # zero out nodata

        return (y, valid_mask) if return_mask else y

    # @staticmethod
    # def _norm_img(img: np.ndarray, nodata=None, plo: float = 1, phi: float = 99) -> np.ndarray:
    #     x = img.astype(np.float32)
    #     if nodata is not None:
    #         x[x == nodata] = np.nan
    #     x[x > 1e10] = np.nan
    #     x[x < -1e10] = np.nan
    #
    #     valid = x[~np.isnan(x)]
    #     if valid.size == 0 or valid.max() == valid.min():
    #         return np.zeros_like(x)
    #
    #     log_mask = (~np.isnan(x) & (x > 0))
    #     x[log_mask] = np.log1p(x[log_mask])
    #
    #     valid = x[~np.isnan(x)]
    #     lo = np.percentile(valid, plo)
    #     hi = np.percentile(valid, phi)
    #     if hi == lo:
    #         return np.zeros_like(x)
    #
    #     y = np.clip(x, lo, hi)
    #     y = (y - lo) / (hi - lo + 1e-6)
    #     y[np.isnan(x)] = 0.0
    #     return y

    @staticmethod
    def _inpaint_nodata(img_norm: np.ndarray, valid_mask: np.ndarray,
                        radius: int = 5) -> np.ndarray:
        """
        Fill nodata regions using Telea inpainting so detectors
        don't see hard 0-boundary edges.
        """
        nodata_region = (~valid_mask).astype(np.uint8)  # 1 = nodata, 0 = valid
        if nodata_region.sum() == 0:
            return img_norm  # nothing to do

        img_u8 = (img_norm * 255).clip(0, 255).astype(np.uint8)
        inpainted_u8 = cv2.inpaint(img_u8, nodata_region, radius, cv2.INPAINT_TELEA)
        inpainted = inpainted_u8.astype(np.float32) / 255.0

        # Preserve valid pixels exactly — only fill nodata
        result = img_norm.copy()
        result[~valid_mask] = inpainted[~valid_mask]
        return result

    @staticmethod
    def _filter_lafs_by_mask(lafs: th.Tensor, descs: th.Tensor,
                             valid_mask: np.ndarray,
                             check_window: bool = True,
                             window_margin: int = 8) -> Tuple[th.Tensor, th.Tensor]:
        """
        Remove keypoints whose center (or support window) falls in nodata.
        lafs:  [1, N, 2, 3]
        descs: [N, D]  or  [1, N, D]
        valid_mask: [H, W] numpy bool
        """
        H, W = valid_mask.shape
        mask_t = th.from_numpy(valid_mask).to(lafs.device)  # [H, W]

        centers = KF.get_laf_center(lafs)  # [1, N, 2]  (x, y)
        cx = centers[0, :, 0].long().clamp(0, W - 1)  # [N]
        cy = centers[0, :, 1].long().clamp(0, H - 1)  # [N]

        # Center check
        keep = mask_t[cy, cx]  # [N] bool

        if check_window:
            # Also reject keypoints whose descriptor window touches nodata
            # window_margin approximates the SIFT/DISK 8px gradient support
            x0 = (cx - window_margin).clamp(0, W - 1)
            x1 = (cx + window_margin).clamp(0, W - 1)
            y0 = (cy - window_margin).clamp(0, H - 1)
            y1 = (cy + window_margin).clamp(0, H - 1)

            # Check all four corners of the support window
            keep = (keep
                    & mask_t[y0, x0]
                    & mask_t[y0, x1]
                    & mask_t[y1, x0]
                    & mask_t[y1, x1])

        if keep.sum() == 0:
            return None, None

        lafs_f = lafs[:, keep, :, :]  # [1, M, 2, 3]
        if descs.dim() == 2:
            descs_f = descs[keep]  # [M, D]
        else:
            descs_f = descs[:, keep, :]  # [1, M, D]

        return lafs_f, descs_f

    @staticmethod
    def _xy_to_map(x, y, x0, y0, xres, yres, offset_x, offset_y):
        return (x0 + (offset_x + x + PIXEL_CENTER_OFFSET) * xres,
                y0 + (offset_y + y + PIXEL_CENTER_OFFSET) * yres)

    @staticmethod
    def _global_scan_pix(metadata: Dict, win_row: int, win_col: int) -> Tuple[int, int]:
        """Window top-left as (scan=row, pix=col) in the ORIGINAL H5 grid.

        win_row/win_col are pixel offsets in the (possibly resampled) chip;
        the crop offsets are already in original H5 pixels."""
        ry = abs(metadata['yres1'] / metadata.get('nisar_src_yres', metadata['yres1']))
        rx = abs(metadata['xres1'] / metadata.get('nisar_src_xres', metadata['xres1']))
        scan = int(metadata['nisar_crop_row_offset'] + round(win_row * ry))
        pix = int(metadata['nisar_crop_col_offset'] + round(win_col * rx))
        return scan, pix

    @staticmethod
    def _get_matching_keypoints(kp1, kp2, idxs):
        kp1_sq = kp1.squeeze()
        kp2_sq = kp2.squeeze()

        valid = (idxs[:, 0] < kp1_sq.shape[0]) & (idxs[:, 1] < kp2_sq.shape[0])
        idxs = idxs[valid]
        if idxs.shape[0] == 0:
            return None, None

        mkpts1 = KF.get_laf_center(kp1_sq[idxs[:, 0]].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        mkpts2 = KF.get_laf_center(kp2_sq[idxs[:, 1]].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        return mkpts1, mkpts2

    @abstractmethod
    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        pass

    @abstractmethod
    def get_available_matchers(self) -> List[str]:
        pass

    def _prefix_base(self) -> str:
        return self.get_detector_name()

    def _load_model(self, label: str, factory: Callable):
        """Build a network once. A failure is remembered for the rest of the
        job and raised as ModelLoadError, so the detector stops with a clear
        message instead of failing -- and re-trying a download -- silently in
        every window."""
        if label in FAILED_MODELS:
            raise ModelLoadError(FAILED_MODELS[label])
        MODEL_LABELS.append(label)
        try:
            return factory()
        except ModelLoadError:
            raise
        except Exception as e:
            looked = '; '.join(os.path.join(d, 'checkpoints') for d in weight_search_dirs())
            msg = (f'{label} could not be loaded ({type(e).__name__}: {e}). Looked for its weights in: '
                   f'{looked}. On an offline machine, run prefetch_weights.py on a connected one '
                   f'and set the weights folder to its output (or copy the files into one of these).')
            FAILED_MODELS[label] = msg
            print(f'[{self.get_filename_prefix()}] MODEL LOAD FAILED: {msg}')
            raise ModelLoadError(msg) from e
        finally:
            MODEL_LABELS.pop()

    def get_filename_prefix(self) -> str:
        """Detector token in every output name: the class's base prefix plus
        the variant suffix (non-default parameters, e.g. 'sift-rs0')."""
        return self._prefix_base() + (self.variant_suffix or '')

    def match_smnn(self, descs1, descs2, threshold: float):
        return KF.match_smnn(descs1.squeeze(0), descs2.squeeze(0), th.tensor(threshold))

    def match_adalam(self, descs1, descs2, lafs1, lafs2, hw1, hw2, overrides: Optional[Dict] = None):
        cfg = dict(self.adalam_config)
        cfg.update(overrides or {})
        return KF.match_adalam(
            descs1.squeeze(0),
            descs2.squeeze(0),
            lafs1,
            lafs2,
            config=cfg,
            hw1=hw1,
            hw2=hw2
        )

    def match_generic(self, name: str, descs1, descs2, lafs1, lafs2, param: Optional[Dict] = None):
        """kornia's plain matchers: nn, mnn (mutual NN), snn (ratio test) and
        fginn (ratio test against the first geometrically distinct neighbour)."""
        d1, d2 = descs1.squeeze(0), descs2.squeeze(0)
        p = dict(param or {})
        if name == 'nn':
            return KF.match_nn(d1, d2)
        if name == 'mnn':
            return KF.match_mnn(d1, d2)
        if name == 'snn':
            return KF.match_snn(d1, d2, th=float(p.get('th', 0.8)))
        if name == 'fginn':
            return KF.match_fginn(d1, d2, lafs1, lafs2, th=float(p.get('th', 0.8)),
                                  spatial_th=float(p.get('spatial_th', 10.0)),
                                  mutual=bool(p.get('mutual', False)))
        raise ValueError(f'Unknown matcher: {name}')

    def _lightglue(self, feature_name: str, params: Optional[Dict] = None):
        """The LightGlue matcher for these features and parameters (built once)."""
        key = (feature_name, tuple(sorted((params or {}).items())))
        model = self._lgm_models.get(key)
        if model is None:
            print(f'{self.get_detector_name()}: Initializing LightGlue {dict(params or {}) or "(defaults)"}...')
            model = self._load_model(
                f'LightGlue ({feature_name})',
                lambda: KF.LightGlueMatcher(feature_name=feature_name,
                                            params=dict(params or {})).eval().to(self.device))
            self._lgm_models[key] = model
        return model

    def match_lgm(self, descs1, descs2, lafs1, lafs2, hw1, hw2, feature_name='disk',
                  params: Optional[Dict] = None):
        model = self._lightglue(feature_name, params)
        # LightGlueMatcher wants (N, D): a (1, N, D) batch (SIFT, KeyNet, DoG)
        # reads as a single descriptor and silently returns no matches
        d1 = descs1.squeeze(0) if descs1.dim() == 3 else descs1
        d2 = descs2.squeeze(0) if descs2.dim() == 3 else descs2
        with th.no_grad():
            return model(
                d1,
                d2,
                lafs1,
                lafs2,
                hw1=hw1,
                hw2=hw2
            )

    def unload_model(self):
        """Drop every loaded network so the next detector starts with free GPU
        memory (instances are built one at a time, see _iter_matchers)."""
        released = False
        for k, v in list(vars(self).items()):
            if isinstance(v, th.nn.Module):
                setattr(self, k, None)
                released = True
        if self._lgm_models:
            self._lgm_models.clear()
            released = True
        if released:
            safe_cuda_empty_cache()
            gc.collect()

    def _detector_needs_inpaint(self) -> bool:
        return False  # default: no inpainting


# =============================================================================
# DISK-BASED MATCHER WRAPPER
# =============================================================================
class DiskBasedMatcher(BaseMatcher):

    def process_disk_cached_pair(self, pair: Dict) -> List[Dict]:
        print(f"{self.get_filename_prefix()}: Processing pair {pair['pair_id']} from disk...")

        with rt.open(pair['nisar_path']) as nisar_src, rt.open(pair['s1_path']) as s1_src:
            nisar_h, nisar_w = nisar_src.height, nisar_src.width
            s1_h, s1_w = s1_src.height, s1_src.width

            strategy = self._determine_window_strategy(nisar_src, s1_src)
            all_matches = []
            start = time.time()

            for matcher_name, matcher_param in self.matcher_runs():
                t_pass = time.time()
                matches = self._process_windows_from_disk(
                    nisar_src, s1_src, pair, strategy, matcher_name, matcher_param
                )
                if th.cuda.is_available():
                    th.cuda.synchronize()
                self._time_pass(matcher_name, matcher_param, time.time() - t_pass, len(matches))
                all_matches.extend(matches)

            elapsed = time.time() - start
            print(f"{self.get_filename_prefix()}: {len(all_matches)} match-sets in {elapsed:.2f}s")
            return all_matches

    def _build_models(self) -> None:
        """Build the detector network(s) of this variant if not built yet
        (each subclass; nothing for model-free detectors)."""

    def load_models(self) -> None:
        """Build every network this variant uses -- the detector, and one
        LightGlue per parameter set of the selected matchers -- without
        running any of them. Raises ModelLoadError."""
        self._build_models()
        for name, param in self.matcher_runs():
            if name == 'lgm':
                self._lightglue(self._lightglue_feature_name(), param)

    def warmup(self) -> None:
        """Load every network this variant will use (its detector, and one
        LightGlue per parameter set) on a tiny synthetic pair before any real
        work, so missing weights stop the detector in seconds, not hours.
        Raises ModelLoadError."""
        rng = np.random.default_rng(0)
        img = (rng.random((256, 256)) * 1000.0 + 1.0).astype(np.float32)
        meta = {'x01': 0.0, 'y01': 0.0, 'xres1': 1.0, 'yres1': -1.0, 'x02': 0.0, 'y02': 0.0,
                'xres2': 1.0, 'yres2': -1.0, 'nisar_crop_row_offset': 0, 'nisar_crop_col_offset': 0,
                'pair_id': 0, 'nisar_pol': 'warmup', 's1_ref_tag': 'warmup'}
        done = set()
        for name, param in self.matcher_runs():
            key = (name, repr(param)) if name == 'lgm' else (name,)
            if key in done:
                continue
            done.add(key)
            self._process_single_window(img, img.copy(), 0, 0, 0, 0, meta, name, param,
                                        nisar_nodata=0.0, s1_nodata=0.0)
        self._window_errors = 0

    def _pass_label(self, matcher_name: str, param) -> str:
        """The matcher label the file names (and the truth ranking) use."""
        if matcher_name == 'smnn':
            return f'smnn{float(param):g}'
        p = self._matcher_param_str(matcher_name, param)
        return f'{matcher_name}_{p}' if p else matcher_name

    def _time_pass(self, matcher_name: str, param, seconds: float, n_windows: int) -> None:
        """Wall time of each matching pass (detection + matching of every
        window; each pass detects again), summed over pairs."""
        t = getattr(self, 'pass_timing', None)
        if t is None:
            t = self.pass_timing = {}
        rec = t.setdefault(self._pass_label(matcher_name, param), {'seconds': 0.0, 'windows': 0})
        rec['seconds'] += seconds
        rec['windows'] += n_windows

    def matcher_runs(self) -> List[Tuple[str, object]]:
        """(matcher, parameter) for every matching pass of this detector
        variant: one per SMNN threshold, one per LightGlue / AdaLAM parameter
        combination (from matcher_grid), one for anything else."""
        runs: List[Tuple[str, object]] = []
        for m in self.selected_matchers():
            if m == 'smnn':
                runs += [('smnn', t) for t in self.config.smnn_thresholds]
            elif self.matcher_grid.get(m):
                grid = self.matcher_grid[m]
                keys = list(grid)
                for combo in itertools.product(*[grid[k] for k in keys]):
                    runs.append((m, dict(zip(keys, combo))))
            else:
                runs.append((m, None))
        return runs

    def _matcher_param_str(self, matcher_name: str, param) -> str:
        """File-name token for a matcher parameter ('' = defaults)."""
        if matcher_name == 'smnn':
            return f'{param}'
        if isinstance(param, dict) and param:
            specs = {sp.name.split('.', 1)[1]: sp
                     for sp in detector_param_specs(self.registry_name or self.get_detector_name())
                     if sp.scope == matcher_name}
            toks = [f'{specs[k].token}{param_token_value(v)}' for k, v in param.items()
                    if k in specs and v != specs[k].default]
            return '-'.join(toks)
        return ''

    def get_optional_matchers(self) -> List[str]:
        """kornia matchers offered on request: they run only when selected."""
        return list(GENERIC_MATCHERS) if 'smnn' in self.get_available_matchers() else []

    def selected_matchers(self) -> List[str]:
        """The default matchers (get_available_matchers), or the selection in
        config.detector_matchers (keyed by the registry name the user
        selected), which may add the optional ones."""
        supported = self.get_available_matchers()
        dm = self.config.detector_matchers or {}
        wanted = None
        for key in (self.registry_name, self.get_filename_prefix(), self.get_detector_name()):
            if key and key in dm:
                wanted = dm[key]
                break
        if not wanted:
            return supported
        offered = supported + [m for m in self.get_optional_matchers() if m not in supported]
        # matchers the detector family offers but this variant cannot run
        # (LightGlue has DoG weights for HardNet only): skipped, with a note
        family = (KORNIA_MATCHERS.get(self.registry_name or '', [])
                  + OPTIONAL_MATCHERS.get(self.registry_name or '', []))
        bad = [m for m in wanted if m not in offered and m not in family]
        if bad:
            raise ValueError(f'{self.get_filename_prefix()}: unsupported matcher(s) {bad}; '
                             f'supported: {offered}')
        missing = [m for m in wanted if m not in offered]
        if missing and not getattr(self, '_noted_missing', False):
            self._noted_missing = True
            print(f'[{self.get_filename_prefix()}] {", ".join(missing)} not available for this '
                  f'variant -- skipped (available: {", ".join(offered)})')
        return [m for m in offered if m in wanted]

    def save_matches_to_csv(self, all_matches: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for match in all_matches:
            csv_path = os.path.join(output_dir, f"{match['file_id']}_raw.csv")
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'x1', 'y1', 'x2', 'y2',
                    'x1_map', 'y1_map', 'x2_map', 'y2_map',
                    'distance', 'along', 'across'
                ])
                for i in range(len(match['x1'])):
                    writer.writerow([
                        match['x1'][i], match['y1'][i],
                        match['x2'][i], match['y2'][i],
                        match['X1'][i], match['Y1'][i],
                        match['X2'][i], match['Y2'][i],
                        match['distance'][i],
                        match['along'][i],
                        match['across'][i],
                    ])

    def _determine_window_strategy(self, nisar_src, s1_src) -> Dict:
        """
            Scan the full image at low resolution to identify valid-data windows.
            Only tile windows that have enough valid data.
            """
        base = self.config.window_size
        fallback = self.config.window_size_small

        # Low-res validity scan (1/16 resolution, negligible cost)
        overview_h = max(10, nisar_src.height // 16)
        overview_w = max(10, nisar_src.width // 16)
        overview = nisar_src.read(1, out_shape=(overview_h, overview_w))
        nodata_val = nisar_src.nodata or 0
        valid_map = (overview != nodata_val).astype(float)  # [overview_h, overview_w]

        windows = []
        for x in range(0, nisar_src.height, base):
            for y in range(0, nisar_src.width, base):
                sx = min(base, nisar_src.height - x)
                sy = min(base, nisar_src.width - y)
                if sx < 500 or sy < 500:
                    continue

                # Map window coords to overview space
                ox0 = int(x / nisar_src.height * overview_h)
                oy0 = int(y / nisar_src.width * overview_w)
                ox1 = int((x + sx) / nisar_src.height * overview_h)
                oy1 = int((y + sy) / nisar_src.width * overview_w)
                ox1, oy1 = min(ox1, overview_h), min(oy1, overview_w)

                region = valid_map[ox0:ox1, oy0:oy1]
                if region.size == 0:
                    continue

                valid_frac = region.mean()
                if valid_frac >= self.config.min_valid_fraction:
                    windows.append((x, y, sx, sy))
                # else: skip entirely — don't even attempt

        return {
            'type': 'VALID_ADAPTIVE',
            'description': f'{len(windows)} valid windows from {base}px grid',
            'windows': windows,
        }

        # min_w = 500
        #
        # if nisar_h <= base and nisar_w <= base:
        #     return {
        #         'type': 'FULL_IMAGE',
        #         'description': f'Single window {nisar_h}x{nisar_w}',
        #         'windows': [(0, 0, nisar_h, nisar_w)]
        #     }
        #
        # x_pos = []
        # x = 0
        # while x < nisar_h:
        #     x_end = min(x + base, nisar_h)
        #     if (x_end - x) >= min_w:
        #         x_pos.append((x, x_end - x))
        #     if x_end == nisar_h:
        #         break
        #     x += base
        #
        # y_pos = []
        # y = 0
        # while y < nisar_w:
        #     y_end = min(y + base, nisar_w)
        #     if (y_end - y) >= min_w:
        #         y_pos.append((y, y_end - y))
        #     if y_end == nisar_w:
        #         break
        #     y += base
        #
        # windows = [(xs, ys, sx, sy) for xs, sx in x_pos for ys, sy in y_pos]
        # t = 'ADAPTIVE_GRID' if len(windows) > 1 else 'SINGLE_WINDOW'
        # return {
        #     'type': t,
        #     'description': f'{len(windows)} windows adaptive edges',
        #     'windows': windows
        # }

    def _build_file_id(self, metadata, global_row, global_col, matcher_name, match_param_str):
        base = (
            f"{self.get_filename_prefix()}_{metadata['nisar_pol']}_to{metadata['s1_ref_tag']}"
            f"_pair{metadata['pair_id']:03d}_scan{global_row}_pix{global_col}_{matcher_name}"
        )
        if match_param_str != '':
            base += f"_{match_param_str}"
        return base

    def _process_windows_from_disk(
        self,
        nisar_src,
        s1_src,
        metadata,
        strategy,
        matcher_name,
        matcher_param
    ) -> List[Dict]:
        matches = []
        nisar_nodata = nisar_src.nodata if nisar_src.nodata is not None else metadata.get('nisar_nodata', 0)
        s1_nodata = s1_src.nodata if s1_src.nodata is not None else metadata.get('s1_nodata', 0)
        windows = strategy['windows']
        n_err = 0
        pstr = self._matcher_param_str(matcher_name, matcher_param) if matcher_param is not None else ''
        label = f"{self.get_filename_prefix()}/{matcher_name}" + (f"@{pstr}" if pstr else '')

        for idx, (wx, wy, sx, sy) in enumerate(windows):
            try:
                nisar_win = rt.windows.Window(wy, wx, sy, sx)
                nisar_data = nisar_src.read(1, window=nisar_win)

                field = metadata.get('coarse_field')
                margin = metadata.get('search_margin_m', 0.0)
                if field:
                    xc, yc = nisar_src.transform * (wy + sy / 2.0, wx + sx / 2.0)
                    wdx, wdy = field_offset(field, xc, yc)
                    # the offset varies inside the window too (scale / warp):
                    # widen the search by how far the corners' offsets depart
                    # from the centre's
                    spread = 0.0
                    for cc, rr in ((wy, wx), (wy + sy, wx), (wy, wx + sx), (wy + sy, wx + sx)):
                        ox, oy = field_offset(field, *(nisar_src.transform * (cc, rr)))
                        spread = max(spread, abs(ox - wdx), abs(oy - wdy))
                    margin += spread
                else:
                    wdx, wdy = metadata.get('coarse_dx', 0.0), metadata.get('coarse_dy', 0.0)
                s1_win = self._s1_window_for(
                    nisar_src, s1_src, wx, wy, sx, sy, wdx, wdy, margin)
                if s1_win is None:
                    continue
                s1wy, s1wx = int(s1_win.col_off), int(s1_win.row_off)
                s1_data = s1_src.read(1, window=s1_win)

                s1_data = self._calibrate_s1_dn(s1_data, nodata=s1_nodata,
                                                factor=self.config.s1_calibration_factor)

                if nisar_data.size == 0 or s1_data.size == 0:
                    continue
                if nisar_data.shape[0] < 100 or nisar_data.shape[1] < 100:
                    continue
                if s1_data.shape[0] < 50 or s1_data.shape[1] < 50:
                    continue

                match_data = self._process_single_window(
                    nisar_data,
                    s1_data,
                    wx,
                    wy,
                    s1wx,
                    s1wy,
                    metadata,
                    matcher_name,
                    matcher_param,
                    nisar_nodata=nisar_nodata,
                    s1_nodata=s1_nodata
                )
                if match_data:
                    matches.append(match_data)

            except ModelLoadError:
                raise
            except Exception as e:
                n_err += 1
                if self.config.debug_mode or n_err <= 3:
                    print(f'[{label}] window ({wx},{wy}) error: {type(e).__name__}: {e}')
                if 'device-side assert' in str(e) or 'out of memory' in str(e).lower():
                    safe_cuda_empty_cache()
                    try:
                        th.cuda.synchronize()
                    except Exception:
                        pass
            finally:
                if (idx + 1) % 20 == 0:
                    safe_cuda_empty_cache()
                report('window', detector=self.get_filename_prefix(), matcher=label,
                       pair=metadata.get('pair_id'), done=idx + 1, total=len(windows))

        if n_err > 3 and not self.config.debug_mode:
            print(f'[{label}] {n_err} windows failed in pair {metadata.get("pair_id")} '
                  f'(first 3 shown; set debug_mode for all)')
        return matches

    @staticmethod
    def _s1_window_for(nisar_src, s1_src, wx, wy, sx, sy,
                       dx: float = 0.0, dy: float = 0.0, margin: float = 0.0):
        """Reference read window for NISAR window (wx=row, wy=col, sx=rows,
        sy=cols): the same ground moved by the coarse offset (a feature at P in
        the input is at P - (dx, dy) in the reference), grown by margin metres
        and clipped to the reference raster. None if empty."""
        x0, y0 = nisar_src.transform * (wy, wx)
        x1, y1 = nisar_src.transform * (wy + sy, wx + sx)
        x0, x1 = min(x0, x1) - dx - margin, max(x0, x1) - dx + margin
        y0, y1 = min(y0, y1) - dy - margin, max(y0, y1) - dy + margin
        inv = ~s1_src.transform
        c0, r0 = inv * (x0, y0)
        c1, r1 = inv * (x1, y1)
        col0, col1 = int(round(min(c0, c1))), int(round(max(c0, c1)))
        row0, row1 = int(round(min(r0, r1))), int(round(max(r0, r1)))
        col0, row0 = max(0, col0), max(0, row0)
        col1, row1 = min(s1_src.width, col1), min(s1_src.height, row1)
        if col1 <= col0 or row1 <= row0:
            return None
        return rt.windows.Window(col0, row0, col1 - col0, row1 - row0)

    def _process_single_window(self, nisar_data, s1_data,
                               nisar_x, nisar_y, s1_x, s1_y,
                               metadata, matcher_name, matcher_param,
                               nisar_nodata=None, s1_nodata=None):
        try:
            # ── Level 0: normalise + extract masks ──────────────────────────
            img1, mask1 = self._norm_img(nisar_data, nodata=nisar_nodata, return_mask=True)
            img2, mask2 = self._norm_img(s1_data, nodata=s1_nodata, return_mask=True)

            if img1.max() == 0.0 or img2.max() == 0.0:
                return None

            valid_frac1 = mask1.mean()
            valid_frac2 = mask2.mean()
            if valid_frac1 < self.config.min_valid_fraction or \
                    valid_frac2 < self.config.min_valid_fraction:
                return None  # too much nodata in this window

            # ── Level 1: inpaint before dense detectors ──────────────────────
            # DISK and LoFTR see inpainted image (no hard edges)
            # SIFT is robust enough without inpainting (gradient hist is local)
            needs_inpaint = self._detector_needs_inpaint()
            if needs_inpaint:
                img1 = self._inpaint_nodata(img1, mask1)
                img2 = self._inpaint_nodata(img2, mask2)

            t1 = th.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t2 = th.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).to(self.device)
            hw1 = th.tensor(t1.shape[2:], device=self.device)
            hw2 = th.tensor(t2.shape[2:], device=self.device)

            # ── Detect + describe ────────────────────────────────────────────
            with th.inference_mode():
                if self.use_amp and th.cuda.is_available():
                    with th.cuda.amp.autocast():
                        lafs1, descs1, lafs2, descs2 = self.detect_and_describe(t1, t2)
                else:
                    lafs1, descs1, lafs2, descs2 = self.detect_and_describe(t1, t2)

            if lafs1 is None or lafs2 is None:
                return None

            # ── Level 2: filter keypoints by valid mask ──────────────────────
            lafs1, descs1 = self._filter_lafs_by_mask(lafs1, descs1, mask1)
            lafs2, descs2 = self._filter_lafs_by_mask(lafs2, descs2, mask2)

            if lafs1 is None or lafs2 is None:
                return None
            if lafs1.shape[1] < 8 or lafs2.shape[1] < 8:
                return None  # too few keypoints after masking

            # ── Match ────────────────────────────────────────────────────────
            with th.no_grad():
                if matcher_name == 'ada':
                    _, idxs = self.match_adalam(descs1, descs2, lafs1, lafs2, hw1, hw2,
                                                overrides=matcher_param)
                    match_param_str = self._matcher_param_str('ada', matcher_param)
                elif matcher_name == 'smnn':
                    _, idxs = self.match_smnn(descs1, descs2, matcher_param)
                    match_param_str = f'{matcher_param}'
                elif matcher_name == 'lgm':
                    _, idxs = self.match_lgm(descs1, descs2, lafs1, lafs2, hw1, hw2,
                                             feature_name=self._lightglue_feature_name(),
                                             params=matcher_param)
                    match_param_str = self._matcher_param_str('lgm', matcher_param)
                elif matcher_name in GENERIC_MATCHERS:
                    _, idxs = self.match_generic(matcher_name, descs1, descs2, lafs1, lafs2,
                                                 matcher_param)
                    match_param_str = self._matcher_param_str(matcher_name, matcher_param)
                else:
                    raise ValueError(f'Unknown matcher: {matcher_name}')

            if idxs.shape[0] == 0:
                return None

            return self._extract_matches(lafs1, lafs2, idxs,
                                         nisar_x, nisar_y, s1_x, s1_y,
                                         metadata, matcher_name, match_param_str)

        except ModelLoadError:
            raise
        except Exception as e:
            self._window_errors = getattr(self, '_window_errors', 0) + 1
            if self.config.debug_mode or self._window_errors <= 3:
                print(f'[{self.get_filename_prefix()}] window ({nisar_x},{nisar_y}) error: '
                      f'{type(e).__name__}: {e}')
            return None

    # def _process_single_window(
    #     self,
    #     nisar_data,
    #     s1_data,
    #     nisar_x,
    #     nisar_y,
    #     s1_x,
    #     s1_y,
    #     metadata,
    #     matcher_name,
    #     matcher_param,
    #     nisar_nodata=None,
    #     s1_nodata=None
    # ) -> Optional[Dict]:
    #     try:
    #         img1 = self._norm_img(nisar_data, nodata=nisar_nodata)
    #         img2 = self._norm_img(s1_data, nodata=s1_nodata)
    #
    #         if img1.max() == 0.0 or img2.max() == 0.0:
    #             return None
    #         if np.isnan(img1).any() or np.isnan(img2).any():
    #             return None
    #
    #         img1 = th.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).to(self.device)
    #         img2 = th.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).to(self.device)
    #
    #         hw1 = th.tensor(img1.shape[2:], device=self.device)
    #         hw2 = th.tensor(img2.shape[2:], device=self.device)
    #
    #         with th.inference_mode():
    #             if self.use_amp and th.cuda.is_available():
    #                 with th.cuda.amp.autocast():
    #                     lafs1, descs1, lafs2, descs2 = self.detect_and_describe(img1, img2)
    #             else:
    #                 lafs1, descs1, lafs2, descs2 = self.detect_and_describe(img1, img2)
    #
    #         if lafs1 is None or lafs2 is None:
    #             return None
    #
    #         with th.no_grad():
    #             if matcher_name == 'ada':
    #                 _, idxs = self.match_adalam(descs1, descs2, lafs1, lafs2, hw1, hw2)
    #                 match_param_str = ''
    #             elif matcher_name == 'smnn':
    #                 _, idxs = self.match_smnn(descs1, descs2, matcher_param)
    #                 match_param_str = f'{matcher_param}'
    #             elif matcher_name == 'lgm':
    #                 _, idxs = self.match_lgm(descs1, descs2, lafs1, lafs2, hw1, hw2, feature_name=self._lightglue_feature_name())
    #                 match_param_str = ''
    #             else:
    #                 raise ValueError(f'Unknown matcher: {matcher_name}')
    #
    #         if idxs.shape[0] == 0:
    #             return None
    #
    #         return self._extract_matches(
    #             lafs1, lafs2, idxs,
    #             nisar_x, nisar_y, s1_x, s1_y,
    #             metadata, matcher_name, match_param_str
    #         )
    #
    #     except Exception as e:
    #         if self.config.debug_mode:
    #             print(f'DEBUG Exception in window {nisar_x},{nisar_y}: {type(e).__name__}: {e}')
    #         return None

    def _extract_matches(
        self,
        lafs1,
        lafs2,
        idxs,
        nisar_wx,
        nisar_wy,
        s1_wx,
        s1_wy,
        metadata,
        matcher_name,
        match_param_str
    ) -> Dict:
        mkpts1, mkpts2 = self._get_matching_keypoints(lafs1, lafs2, idxs)
        if mkpts1 is None or len(mkpts1) == 0:
            return None

        x1, y1 = zip(*mkpts1)
        x2, y2 = zip(*mkpts2)

        X1, Y1 = zip(*[
            self._xy_to_map(
                x1[i], y1[i],
                metadata['x01'], metadata['y01'],
                metadata['xres1'], metadata['yres1'],
                nisar_wy, nisar_wx
            )
            for i in range(len(x1))
        ])

        X2, Y2 = zip(*[
            self._xy_to_map(
                x2[i], y2[i],
                metadata['x02'], metadata['y02'],
                metadata['xres2'], metadata['yres2'],
                s1_wy, s1_wx
            )
            for i in range(len(x2))
        ])

        x1_global = [v + nisar_wy for v in x1]
        x2_global = [v + s1_wy for v in x2]
        y1_global = [v + nisar_wx for v in y1]
        y2_global = [v + s1_wx for v in y2]

        global_row, global_col = self._global_scan_pix(metadata, nisar_wx, nisar_wy)

        dist = [np.sqrt((X1[i] - X2[i])**2 + (Y1[i] - Y2[i])**2) for i in range(len(x1))]
        along = [Y1[i] - Y2[i] for i in range(len(x1))]
        across = [X1[i] - X2[i] for i in range(len(x1))]

        file_id = self._build_file_id(metadata, global_row, global_col, matcher_name, match_param_str)

        return {
            'file_id': file_id,
            'pair_id': metadata['pair_id'],
            'global_scan': global_row,
            'global_pix': global_col,
            'local_init_x': nisar_wx,
            'local_init_y': nisar_wy,
            'matcher': matcher_name,
            'match_param': match_param_str,
            'x1': x1_global,
            'y1': y1_global,
            'x2': x2_global,
            'y2': y2_global,
            'X1': list(X1),
            'Y1': list(Y1),
            'X2': list(X2),
            'Y2': list(Y2),
            'distance': dist,
            'along': along,
            'across': across,
        }

    def _lightglue_feature_name(self) -> str:
        return 'disk'

    def _detector_needs_inpaint(self) -> bool:
        return True  # DISK U-Net is sensitive to hard boundary edges


# =============================================================================
# DETECTOR IMPLEMENTATIONS
# =============================================================================
class SIFTMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig, rootsift: bool = True, upright: bool = True,
                 score_threshold: float = 0.0):
        super().__init__(config)
        self.rootsift = rootsift
        self.upright = upright
        self.score_threshold = score_threshold
        self.sift = None

    def _build_models(self) -> None:
        if self.sift is None:
            print(f'SIFT: Initializing (rootsift={self.rootsift}, upright={self.upright})...')
            extra = {'score_threshold': self.score_threshold} if self.score_threshold else {}
            self.sift = self._load_model('SIFT', lambda: KF.SIFTFeature(
                self.config.num_features,
                upright=self.upright,
                rootsift=self.rootsift,
                device=self.device,
                **extra
            ))

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        lafs1, _, descs1 = self.sift(img1)
        lafs2, _, descs2 = self.sift(img2)
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'sift'

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'lgm', 'ada']

    def _lightglue_feature_name(self) -> str:
        return 'sift'

    def _detector_needs_inpaint(self) -> bool:
        return False


class DISKMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig, checkpoint: str = 'depth'):
        super().__init__(config)
        self.checkpoint = checkpoint
        self.disk = None

    def _build_models(self) -> None:
        if self.disk is None:
            print(f'DISK: Loading {self.checkpoint} model...')
            self.disk = self._load_model(
                f'DISK ({self.checkpoint} weights)',
                lambda: KF.DISK.from_pretrained(device=self.device, checkpoint=self.checkpoint).eval())

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        img1_rgb = K.color.grayscale_to_rgb(img1)
        img2_rgb = K.color.grayscale_to_rgb(img2)

        if img1.shape[2:] == img2.shape[2:]:
            inp = th.cat([img1_rgb, img2_rgb], dim=0)
            features = self.disk(inp, n=self.config.num_features, pad_if_not_divisible=True)
            f1, f2 = features[0], features[1]
        else:
            f1 = self.disk(img1_rgb, n=self.config.num_features, pad_if_not_divisible=True)[0]
            f2 = self.disk(img2_rgb, n=self.config.num_features, pad_if_not_divisible=True)[0]

        kps1, descs1 = f1.keypoints, f1.descriptors
        kps2, descs2 = f2.keypoints, f2.descriptors

        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], 96 * th.ones(1, len(kps1), 1, 1, device=self.device)
        )
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], 96 * th.ones(1, len(kps2), 1, 1, device=self.device)
        )
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'disk'

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'lgm', 'ada']

    def _prefix_base(self) -> str:
        return f'disk_{self.checkpoint}'

    def _lightglue_feature_name(self) -> str:
        return 'disk'

    def _detector_needs_inpaint(self) -> bool:
        return True  # DISK U-Net is sensitive to hard boundary edges


class DeDoDeMatcher(DiskBasedMatcher):

    def __init__(self, config: PipelineConfig, detector_weights: str = 'L-C4', descriptor_weights: str = 'G-C4'):
        super().__init__(config)
        self.detector_weights = detector_weights
        self.descriptor_weights = descriptor_weights
        self.dedode = None

    def _build_models(self) -> None:
        if self.dedode is None:
            print(f'DeDoDe: Loading {self.detector_weights}/{self.descriptor_weights}...')
            # DeDoDe runs its DINOv2 (G-*) branch in float16 by default; that
            # only works under CUDA autocast, so use float32 on CPU / MPS.
            kwargs = {} if self.device.type == 'cuda' else {'amp_dtype': th.float32}

            def build():
                try:
                    model = KF.DeDoDe.from_pretrained(
                        detector_weights=self.detector_weights,
                        descriptor_weights=self.descriptor_weights, **kwargs)
                except TypeError:  # kornia without the amp_dtype argument
                    model = KF.DeDoDe.from_pretrained(
                        detector_weights=self.detector_weights,
                        descriptor_weights=self.descriptor_weights)
                return model.eval().to(self.device)
            self.dedode = self._load_model(
                f'DeDoDe ({self.detector_weights} / {self.descriptor_weights})', build)

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        img1_rgb = K.color.grayscale_to_rgb(img1)
        img2_rgb = K.color.grayscale_to_rgb(img2)

        if img1.shape[2:] == img2.shape[2:]:
            inp = th.cat([img1_rgb, img2_rgb], dim=0)
            keypoints, scores, descriptors = self.dedode(inp, n=self.config.num_features)
            kps1, descs1 = keypoints[0], descriptors[0]
            kps2, descs2 = keypoints[1], descriptors[1]
        else:
            kps1, _, descs1 = self.dedode(img1_rgb, n=self.config.num_features)
            kps2, _, descs2 = self.dedode(img2_rgb, n=self.config.num_features)
            kps1 = kps1[0]
            descs1 = descs1[0]
            kps2 = kps2[0]
            descs2 = descs2[0]

        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], 96 * th.ones(1, len(kps1), 1, 1, device=self.device)
        )
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], 96 * th.ones(1, len(kps2), 1, 1, device=self.device)
        )
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'dedode'

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'lgm', 'ada']

    def _lightglue_feature_name(self) -> str:
        # kornia ships LightGlue weights for both DeDoDe descriptor families
        return 'dedodeg' if str(self.descriptor_weights).upper().startswith('G') else 'dedodeb'

    def _prefix_base(self) -> str:
        return f'dedode_{self.detector_weights}_{self.descriptor_weights}'

    def _detector_needs_inpaint(self) -> bool:
        return True  # VGG-19 + DINOv2 both pooling-sensitive to boundaries

class LoFTRMatcher(DiskBasedMatcher):
    """kornia LoFTR with kornia's own configuration for the pretrained weights
    (dual-softmax coarse matching). The coarse threshold is the confidence a
    coarse match needs (kornia default 0.2).

    The configuration this class inherited switched the coarse matching to
    Sinkhorn with an untrained dustbin score and a threshold of 1; with the
    released weights that finds no matches at all, so it is not used."""

    def __init__(self, config: PipelineConfig, pretrained: str = 'outdoor', coarse_threshold: float = 0.2):
        super().__init__(config)
        self.pretrained = pretrained
        self.coarse_threshold = coarse_threshold
        self.loftr = None
        from kornia.feature.loftr.loftr import default_cfg
        # a copy: kornia's LoFTR writes into the config it is given
        self.konfig = copy.deepcopy(default_cfg)
        self.konfig['match_coarse']['thr'] = float(coarse_threshold)

    def _determine_window_strategy(self, nisar_src, s1_src) -> Dict:
        original = self.config.window_size
        self.config.window_size = self.config.loftr_max_window
        strategy = super()._determine_window_strategy(nisar_src, s1_src)
        self.config.window_size = original
        return strategy

    def _build_models(self) -> None:
        if self.loftr is None:
            print(f'LoFTR: Initializing model ({self.pretrained} weights, '
                  f'coarse threshold {self.coarse_threshold})...')
            self.loftr = self._load_model(
                f'LoFTR ({self.pretrained} weights)',
                lambda: KF.LoFTR(pretrained=self.pretrained, config=self.konfig).eval().to(self.device))

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        # LoFTR doesn't use this, but we must implement the abstract method
        pass

    def get_detector_name(self) -> str:
        return 'loftr'

    def get_available_matchers(self) -> List[str]:
        return ['loftr_internal']

    def _detector_needs_inpaint(self) -> bool:
        return True  # ResNetFPN propagates hard edges across receptive field

    def _process_single_window(self, nisar_data, s1_data, nisar_x, nisar_y, s1_x, s1_y,
                               metadata, matcher_name, matcher_param,
                               nisar_nodata=None, s1_nodata=None):
        try:
            img1, mask1 = self._norm_img(nisar_data, nodata=nisar_nodata, return_mask=True)  # was: self.norm_img
            img2, mask2 = self._norm_img(s1_data, nodata=s1_nodata, return_mask=True)  # was: self.norm_img

            if img1.max() == 0.0 or img2.max() == 0.0:
                return None
            if np.isnan(img1).any() or np.isnan(img2).any():
                return None

            t1 = th.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).to(self.device)
            t2 = th.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).to(self.device)

            h1, w1 = t1.shape[2:]
            h2, w2 = t2.shape[2:]
            t1_padded = th.nn.functional.pad(t1, (0, (8 - w1 % 8) % 8, 0, (8 - h1 % 8) % 8))
            t2_padded = th.nn.functional.pad(t2, (0, (8 - w2 % 8) % 8, 0, (8 - h2 % 8) % 8))

            self._build_models()
            with th.no_grad():
                if self.use_amp and th.cuda.is_available():
                    with th.cuda.amp.autocast():
                        correspondences = self.loftr({'image0': t1_padded, 'image1': t2_padded})
                else:
                    correspondences = self.loftr({'image0': t1_padded, 'image1': t2_padded})

            mkpts1 = correspondences['keypoints0'].cpu().numpy()
            mkpts2 = correspondences['keypoints1'].cpu().numpy()
            if len(mkpts1) == 0:
                return None

            valid_mask = ((mkpts1[:, 0] < w1) & (mkpts1[:, 1] < h1) &
                          (mkpts2[:, 0] < w2) & (mkpts2[:, 1] < h2))
            mkpts1 = mkpts1[valid_mask]
            mkpts2 = mkpts2[valid_mask]

            kp1_x = mkpts1[:, 0].astype(int).clip(0, w1 - 1)
            kp1_y = mkpts1[:, 1].astype(int).clip(0, h1 - 1)
            kp2_x = mkpts2[:, 0].astype(int).clip(0, w2 - 1)
            kp2_y = mkpts2[:, 1].astype(int).clip(0, h2 - 1)

            valid_corr = (mask1[kp1_y, kp1_x] & mask2[kp2_y, kp2_x])
            mkpts1 = mkpts1[valid_corr]
            mkpts2 = mkpts2[valid_corr]

            if len(mkpts1) == 0:
                return None

            x1, y1 = zip(*mkpts1)
            x2, y2 = zip(*mkpts2)

            X1, Y1 = zip(*[
                self._xy_to_map(x1[i], y1[i], metadata['x01'], metadata['y01'],  # was: self.xy_to_map
                                metadata['xres1'], metadata['yres1'], nisar_y, nisar_x)
                for i in range(len(x1))
            ])
            X2, Y2 = zip(*[
                self._xy_to_map(x2[i], y2[i], metadata['x02'], metadata['y02'],  # was: self.xy_to_map
                                metadata['xres2'], metadata['yres2'], s1_y, s1_x)
                for i in range(len(x2))
            ])

            dist = [np.sqrt((X1[i] - X2[i]) ** 2 + (Y1[i] - Y2[i]) ** 2) for i in range(len(x1))]
            along = [Y1[i] - Y2[i] for i in range(len(x1))]  # northing diff  (was X1-X2 — WRONG)
            across = [X1[i] - X2[i] for i in range(len(x1))]  # easting diff   (was Y1-Y2 — WRONG)

            # Match _extract_matches convention: col_offset → global_row, row_offset → global_col
            global_row, global_col = self._global_scan_pix(metadata, nisar_x, nisar_y)
            file_id = self._build_file_id(metadata, global_row, global_col, 'loftr_internal', '')

            return {
                'file_id': file_id,
                'pair_id': metadata['pair_id'],
                'global_scan': global_row,
                'global_pix': global_col,
                'local_init_x': nisar_x,
                'local_init_y': nisar_y,
                'matcher': 'loftr_internal',
                'match_param': '',
                'x1': [v + nisar_y for v in x1], 'y1': [v + nisar_x for v in y1],
                'x2': [v + s1_y for v in x2], 'y2': [v + s1_x for v in y2],
                'X1': list(X1), 'Y1': list(Y1),
                'X2': list(X2), 'Y2': list(Y2),
                'distance': dist, 'along': along, 'across': across,
            }

        except ModelLoadError:
            raise
        except Exception as e:
            # Always print LoFTR errors — not gated by debug_mode
            print(f'[LoFTR] Exception at window ({nisar_x},{nisar_y}): {type(e).__name__}: {e}')
            return None

class ALIKEDMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig,
                 model_name: str = 'aliked-n16', detection_threshold: float = 0.2,
                 nms_radius: int = 2):
        super().__init__(config)
        self.model_name = model_name
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        self.aliked = None

    def _build_models(self) -> None:
        if self.aliked is None:
            print(f'ALIKED: Loading {self.model_name}...')
            # kornia >= 0.8: from_pretrained(model_name, max_num_keypoints,
            # detection_threshold, nms_radius, device); forward -> list of
            # ALIKEDFeatures, one per image.
            self.aliked = self._load_model(f'ALIKED ({self.model_name})', lambda: KF.ALIKED.from_pretrained(
                model_name=self.model_name,
                max_num_keypoints=self.config.num_features,
                detection_threshold=self.detection_threshold,
                nms_radius=self.nms_radius,
                device=self.device,
            ).eval().to(self.device))

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        # ALIKED expects 3-channel input
        img1_rgb = K.color.grayscale_to_rgb(img1)
        img2_rgb = K.color.grayscale_to_rgb(img2)

        out1 = self.aliked(img1_rgb)[0]
        out2 = self.aliked(img2_rgb)[0]

        kps1, descs1 = out1.keypoints, out1.descriptors
        kps2, descs2 = out2.keypoints, out2.descriptors

        # Wrap into LAF format for compatibility with rest of pipeline
        lafs1 = KF.laf_from_center_scale_ori(
            kps1[None], 32 * th.ones(1, len(kps1), 1, 1, device=self.device)
        )
        lafs2 = KF.laf_from_center_scale_ori(
            kps2[None], 32 * th.ones(1, len(kps2), 1, 1, device=self.device)
        )

        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'aliked'

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'lgm', 'ada']   # LightGlue natively supports aliked

    def _prefix_base(self) -> str:
        return f'aliked_{self.model_name}'

    def _lightglue_feature_name(self) -> str:
        return 'aliked'          # Kornia LightGlueMatcher knows this name

    def _detector_needs_inpaint(self) -> bool:
        return True              # Deformable conv is boundary-sensitive like DISK

class XFeatMatcher(DiskBasedMatcher):
    """kornia XFeat, sparse: detectAndCompute -> smnn."""

    def __init__(self, config: PipelineConfig, detection_threshold: float = 0.05):
        super().__init__(config)
        self.detection_threshold = detection_threshold
        self.xfeat = None

    def _build_models(self) -> None:
        self._model()

    def _model(self):
        if self.xfeat is None:
            print('XFeat: Loading kornia XFeat...')
            self.xfeat = self._load_model('XFeat', lambda: KF.XFeat.from_pretrained(
                top_k=self.config.num_features,
                detection_threshold=self.detection_threshold).eval().to(self.device))
        return self.xfeat

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        xf = self._model()
        out = []
        for img in (img1, img2):
            f = xf.detectAndCompute(img.float(), top_k=self.config.num_features)[0]
            kps, descs = f['keypoints'], f['descriptors']
            lafs = KF.laf_from_center_scale_ori(
                kps[None], 32 * th.ones(1, len(kps), 1, 1, device=kps.device))
            out += [lafs, descs]
        return out[0], out[1], out[2], out[3]

    def get_detector_name(self) -> str:
        return 'xfeat'

    def get_available_matchers(self) -> List[str]:
        # kornia's LightGlueMatcher has no 'xfeat' weights (0.8.x): no 'lgm'.
        return ['smnn', 'ada']

    def _detector_needs_inpaint(self) -> bool:
        return True


class KeyNetMatcher(DiskBasedMatcher):
    """kornia KeyNet + (AffNet) + HardNet; LightGlue with kornia's KeyNet-AffNet-HardNet weights."""

    def __init__(self, config: PipelineConfig, upright: bool = True, score_threshold: float = 0.0,
                 affnet: bool = True):
        super().__init__(config)
        self.upright = upright
        self.score_threshold = score_threshold
        self.affnet = affnet
        self.feat = None

    def _build_models(self) -> None:
        if self.feat is None:
            cls = KF.KeyNetAffNetHardNet if self.affnet else KF.KeyNetHardNet
            label = 'KeyNet-AffNet-HardNet' if self.affnet else 'KeyNet-HardNet'
            print(f'KeyNet: Loading {label} (upright={self.upright})...')
            extra = {'score_threshold': self.score_threshold} if self.score_threshold else {}
            self.feat = self._load_model(label, lambda: cls(
                num_features=self.config.num_features, upright=self.upright,
                device=self.device, **extra).eval())

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        lafs1, _, descs1 = self.feat(img1.float())
        lafs2, _, descs2 = self.feat(img2.float())
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'keynet'

    def get_available_matchers(self) -> List[str]:
        # kornia's keynet_affnet_hardnet LightGlue returned no matches on
        # upright KeyNet features in testing, so it is not offered.
        return ['smnn', 'lgm', 'ada']

    def _lightglue_feature_name(self) -> str:
        return 'keynet_affnet_hardnet'

    def _detector_needs_inpaint(self) -> bool:
        return False


class DoGDescriptorMatcher(DiskBasedMatcher):
    """kornia DoG keypoints (SIFT's scale space) with a learned patch
    descriptor: HardNet, HardNet8, SOSNet, HyNet or TFeat; optionally AffNet
    affine shapes. LightGlue has weights for DoG-HardNet (with or without
    AffNet), so 'lgm' is offered for the HardNet descriptor only."""

    DESCRIPTORS = {'hardnet': 'HardNet', 'hardnet8': 'HardNet8', 'sosnet': 'SOSNet',
                   'hynet': 'HyNet', 'tfeat': 'TFeat'}

    def __init__(self, config: PipelineConfig, descriptor: str = 'hardnet', affnet: bool = False,
                 upright: bool = True, score_threshold: float = 0.0):
        super().__init__(config)
        if descriptor not in self.DESCRIPTORS:
            raise ValueError(f'descriptor must be one of {sorted(self.DESCRIPTORS)}')
        self.descriptor = descriptor
        self.affnet = affnet
        self.upright = upright
        self.score_threshold = score_threshold
        self.feat = None

    def _build_models(self) -> None:
        if self.feat is None:
            def build():
                try:
                    from kornia.feature.scale_space_detector import get_default_detector_config
                    cfg = get_default_detector_config()
                except ImportError:
                    cfg = None
                extra = {'score_threshold': self.score_threshold} if self.score_threshold else {}
                det = KF.MultiResolutionDetector(
                    KF.BlobDoGSingle(1.0, 1.6), self.config.num_features, cfg,
                    ori_module=KF.PassLAF() if self.upright else KF.LAFOrienter(19),
                    aff_module=KF.LAFAffNetShapeEstimator(True) if self.affnet else KF.PassLAF(),
                    **extra)
                desc = KF.LAFDescriptor(getattr(KF, self.DESCRIPTORS[self.descriptor])(True),
                                        patch_size=32, grayscale_descriptor=True)
                return KF.LocalFeature(det, desc).to(self.device).eval()
            print(f'DoG: Loading {self.DESCRIPTORS[self.descriptor]}'
                  f'{" + AffNet" if self.affnet else ""} (upright={self.upright})...')
            self.feat = self._load_model(
                f'DoG-{"AffNet-" if self.affnet else ""}{self.DESCRIPTORS[self.descriptor]}', build)

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        lafs1, _, descs1 = self.feat(img1.float())
        lafs2, _, descs2 = self.feat(img2.float())
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'dog'

    def _prefix_base(self) -> str:
        return f'dog_{self.descriptor}'

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'lgm', 'ada'] if self.descriptor == 'hardnet' else ['smnn', 'ada']

    def _lightglue_feature_name(self) -> str:
        return 'dog_affnet_hardnet' if self.affnet else 'doghardnet'

    def _detector_needs_inpaint(self) -> bool:
        return False


class AffNetHardNetMatcher(DiskBasedMatcher):
    """kornia's ready-made GFTT-AffNet-HardNet and Hessian-AffNet-HardNet."""

    KINDS = {'gftt': ('GFTTAffNetHardNet', 'GFTT-AffNet-HardNet'),
             'hessian': ('HesAffNetHardNet', 'Hessian-AffNet-HardNet')}

    def __init__(self, config: PipelineConfig, kind: str = 'gftt', upright: bool = True):
        super().__init__(config)
        self.kind = kind
        self.upright = upright
        self.feat = None

    def _build_models(self) -> None:
        if self.feat is None:
            cls_name, label = self.KINDS[self.kind]
            print(f'{label}: Loading (upright={self.upright})...')
            self.feat = self._load_model(label, lambda: getattr(KF, cls_name)(
                num_features=self.config.num_features, upright=self.upright,
                device=self.device).eval())

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        self._build_models()
        lafs1, _, descs1 = self.feat(img1.float())
        lafs2, _, descs2 = self.feat(img2.float())
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return self.kind

    def get_available_matchers(self) -> List[str]:
        return ['smnn', 'ada']

    def _detector_needs_inpaint(self) -> bool:
        return False


class DenseWindowMatcher(DiskBasedMatcher):
    """Base for methods that detect AND match in one call (XFeat*, imcui).

    Subclasses implement match_images(img1, img2) on normalised float images
    in [0, 1] (H x W) and return (mkpts1, mkpts2) as N x 2 (x, y) arrays in
    each image's own pixel coordinates. Masking, bounds checks and the match
    record are shared."""

    matcher_token = 'internal'

    def detect_and_describe(self, img1, img2):
        raise NotImplementedError('dense matchers use match_images()')

    def get_available_matchers(self) -> List[str]:
        return [self.matcher_token]

    def match_images(self, img1: np.ndarray, img2: np.ndarray):
        raise NotImplementedError

    def _process_single_window(self, nisar_data, s1_data,
                               nisar_x, nisar_y, s1_x, s1_y,
                               metadata, matcher_name, matcher_param,
                               nisar_nodata=None, s1_nodata=None):
        img1, mask1 = self._norm_img(nisar_data, nodata=nisar_nodata, return_mask=True)
        img2, mask2 = self._norm_img(s1_data, nodata=s1_nodata, return_mask=True)
        if img1.max() == 0.0 or img2.max() == 0.0:
            return None
        if (mask1.mean() < self.config.min_valid_fraction
                or mask2.mean() < self.config.min_valid_fraction):
            return None
        if self._detector_needs_inpaint():
            img1 = self._inpaint_nodata(img1, mask1)
            img2 = self._inpaint_nodata(img2, mask2)

        mkp1, mkp2 = self.match_images(img1, img2)
        if mkp1 is None or len(mkp1) == 0:
            return None
        mkp1 = np.asarray(mkp1, dtype=np.float64).reshape(-1, 2)
        mkp2 = np.asarray(mkp2, dtype=np.float64).reshape(-1, 2)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        ok = ((mkp1[:, 0] >= 0) & (mkp1[:, 0] < w1) & (mkp1[:, 1] >= 0) & (mkp1[:, 1] < h1)
              & (mkp2[:, 0] >= 0) & (mkp2[:, 0] < w2) & (mkp2[:, 1] >= 0) & (mkp2[:, 1] < h2))
        mkp1, mkp2 = mkp1[ok], mkp2[ok]
        if len(mkp1) == 0:
            return None
        ix1 = mkp1[:, 0].astype(int).clip(0, w1 - 1)
        iy1 = mkp1[:, 1].astype(int).clip(0, h1 - 1)
        ix2 = mkp2[:, 0].astype(int).clip(0, w2 - 1)
        iy2 = mkp2[:, 1].astype(int).clip(0, h2 - 1)
        ok = mask1[iy1, ix1] & mask2[iy2, ix2]
        mkp1, mkp2 = mkp1[ok], mkp2[ok]
        if len(mkp1) < 4:
            return None
        self.on_window_matched(img1, img2, mkp1, mkp2, nisar_x, nisar_y, metadata)
        return self._record_from_points(mkp1, mkp2, nisar_x, nisar_y, s1_x, s1_y,
                                        metadata, self.matcher_token)

    def on_window_matched(self, img1, img2, mkp1, mkp2, nisar_x, nisar_y, metadata):
        """Hook (e.g. saving match previews)."""

    def _record_from_points(self, mkpts1, mkpts2, nisar_wx, nisar_wy, s1_wx, s1_wy,
                            metadata, matcher_name) -> Dict:
        x1, y1 = mkpts1[:, 0].tolist(), mkpts1[:, 1].tolist()
        x2, y2 = mkpts2[:, 0].tolist(), mkpts2[:, 1].tolist()
        X1, Y1 = zip(*[self._xy_to_map(x1[i], y1[i], metadata['x01'], metadata['y01'],
                                       metadata['xres1'], metadata['yres1'], nisar_wy, nisar_wx)
                       for i in range(len(x1))])
        X2, Y2 = zip(*[self._xy_to_map(x2[i], y2[i], metadata['x02'], metadata['y02'],
                                       metadata['xres2'], metadata['yres2'], s1_wy, s1_wx)
                       for i in range(len(x2))])
        dist = [float(np.hypot(X1[i] - X2[i], Y1[i] - Y2[i])) for i in range(len(x1))]
        along = [Y1[i] - Y2[i] for i in range(len(x1))]
        across = [X1[i] - X2[i] for i in range(len(x1))]
        global_row, global_col = self._global_scan_pix(metadata, nisar_wx, nisar_wy)
        file_id = self._build_file_id(metadata, global_row, global_col, matcher_name, '')
        return {
            'file_id': file_id, 'pair_id': metadata['pair_id'],
            'global_scan': global_row, 'global_pix': global_col,
            'local_init_x': nisar_wx, 'local_init_y': nisar_wy,
            'matcher': matcher_name, 'match_param': '',
            'x1': [v + nisar_wy for v in x1], 'y1': [v + nisar_wx for v in y1],
            'x2': [v + s1_wy for v in x2], 'y2': [v + s1_wx for v in y2],
            'X1': list(X1), 'Y1': list(Y1), 'X2': list(X2), 'Y2': list(Y2),
            'distance': dist, 'along': along, 'across': across,
        }


class XFeatStarMatcher(DenseWindowMatcher):
    """kornia XFeat* (semi-dense, coarse match + refinement)."""

    matcher_token = 'internal'

    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.xfeat = None

    def _build_models(self) -> None:
        if self.xfeat is None:
            print('XFeat*: Loading kornia XFeat...')
            self.xfeat = self._load_model(
                'XFeat*', lambda: KF.XFeat.from_pretrained(top_k=self.config.num_features).eval().to(self.device))

    def match_images(self, img1, img2):
        self._build_models()
        t1 = th.from_numpy(img1).float()[None, None].to(self.device)
        t2 = th.from_numpy(img2).float()[None, None].to(self.device)
        with th.inference_mode():
            m1, m2 = self.xfeat.match_xfeat_star(t1, t2, top_k=self.config.num_features)
        return m1.detach().cpu().numpy(), m2.detach().cpu().numpy()

    def get_detector_name(self) -> str:
        return 'xfeatstar'

    def _detector_needs_inpaint(self) -> bool:
        return True


# =============================================================================
# RANSAC
# =============================================================================
class RANSACFilter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def filter_matches_in_memory(self, match_data_list: List[Dict], output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        filtered_files = []
        for match_data in match_data_list:
            filtered_files.extend(self._filter_single_match_set(match_data, output_dir))
        return filtered_files

    def _filter_single_match_set(self, match_data: Dict, output_dir: str) -> List[str]:
        X1 = match_data['X1']
        Y1 = match_data['Y1']
        X2 = match_data['X2']
        Y2 = match_data['Y2']

        if len(X1) == 0:
            return []

        mkpts1 = np.array(list(zip(X1, Y1)))
        mkpts2 = np.array(list(zip(X2, Y2)))
        generated = []
        basename = match_data['file_id']

        for method in self.config.ransac_methods:
            for thr in self.config.ransac_thresholds:
                for conf in self.config.ransac_confidences:
                    out_name = f'{basename}_aff_{method}_{thr}_{conf}_.csv'
                    out_path = os.path.join(output_dir, out_name)

                    try:
                        _, inliers = cv2.estimateAffine2D(
                            mkpts1,
                            mkpts2,
                            method=ransac_flag(method),
                            ransacReprojThreshold=thr,
                            confidence=conf
                        )
                        if inliers is None:
                            continue

                        inliers = inliers.ravel() > 0
                        if inliers.sum() == 0:
                            continue

                        with open(out_path, mode='w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                'x1', 'y1', 'x2', 'y2',
                                'x1-map', 'y1-map', 'x2-map', 'y2-map',
                                'Distance', 'along', 'across',
                                'model-type', 'threshold', 'confidence', 'inliers'
                            ])
                            for i in range(len(X1)):
                                if inliers[i]:
                                    writer.writerow([
                                        match_data['x1'][i], match_data['y1'][i],
                                        match_data['x2'][i], match_data['y2'][i],
                                        X1[i], Y1[i], X2[i], Y2[i],
                                        match_data['distance'][i],
                                        match_data['along'][i],
                                        match_data['across'][i],
                                        method, thr, conf, True
                                    ])

                        generated.append(out_path)
                    except Exception:
                        continue

        return generated


# =============================================================================
# MATCH STATISTICS
# =============================================================================
def _ref_tag_of(token: str) -> str:
    """'toL8ref' -> 'L8ref'; legacy 'toS1VV' -> 'S1VV'."""
    return token[2:] if token.startswith('to') else token


class MatchStatistics:
    @staticmethod
    def _parse_filename(filename: str) -> Optional[Dict]:
        """Parse a filtered-CSV name into its run parameters.

        <det>[_<sub>...]_<pol>_to<tag>_pair<N>_scan<R>_pix<C>_<matcher>[_<p>]_aff_<m>_<thr>_<conf>_.csv

        Anchored on the pair/scan/pix run, so every detector prefix parses the
        same way (sift, disk_<ckpt>, dedode_<det>_<desc>, aliked_<model>,
        xfeat, xfeatstar, keynet, loftr, imw-<tag>). Field meanings match the
        per-detector branches this replaces."""
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) < 10:
            return None
        try:
            k = next(i for i in range(2, len(parts) - 3)
                     if parts[i].startswith('to')
                     and parts[i + 1].startswith('pair')
                     and parts[i + 2].startswith('scan')
                     and parts[i + 3].startswith('pix'))
            aff = parts.index('aff', k + 4)
        except (StopIteration, ValueError):
            return None
        try:
            detector = parts[0]
            extras = parts[1:k - 1]
            rec = {'disk_mode': None, 'det_wt': None, 'desc_wt': None}
            if detector == 'dedode' and len(extras) >= 2:
                rec['det_wt'], rec['desc_wt'] = extras[0], extras[1]
            elif detector.startswith('imw-'):
                rec['desc_wt'] = detector[len('imw-'):]
            elif extras:
                rec['disk_mode'] = extras[0]
            tail = parts[k + 4:aff]
            if not tail:
                return None
            if tail[0] == 'smnn' and len(tail) >= 2:
                match_method, match_parameter = 'smnn', float(tail[1])
            else:
                match_method, match_parameter = '_'.join(tail), None
            rec.update({
                'detector': detector,
                'nisar_pol': parts[k - 1],
                's1_ref_tag': _ref_tag_of(parts[k]),
                'pair_id': int(parts[k + 1][len('pair'):]),
                'scan': int(parts[k + 2][len('scan'):]),
                'pix': int(parts[k + 3][len('pix'):]),
                'match_method': match_method,
                'match_parameter': match_parameter,
                'ransac_method': parts[aff + 1],
                'ransac_threshold': float(parts[aff + 2]),
                'ransac_confidence': float(parts[aff + 3]),
            })
            return rec
        except (IndexError, ValueError):
            return None

    @staticmethod
    def compute_statistics(csv_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for filename in sorted(os.listdir(csv_dir)):
            if not filename.endswith('.csv'):
                continue

            params = MatchStatistics._parse_filename(filename)
            if params is None:
                continue

            filepath = os.path.join(csv_dir, filename)
            along_errors = []
            across_errors = []
            num_inliers = 0

            try:
                with open(filepath, 'r') as f:
                    for row in csv.DictReader(f):
                        if 'true' in row.get('inliers', '').lower():
                            num_inliers += 1
                            along_errors.append(float(row['along']))
                            across_errors.append(float(row['across']))
            except Exception:
                continue

            if num_inliers == 0:
                continue

            along_arr = np.array(along_errors)
            across_arr = np.array(across_errors)
            radial = np.sqrt(along_arr**2 + across_arr**2)

            rec = {
                'filename': filename,
                'num_inliers': num_inliers,
                'along_mean': float(along_arr.mean()),
                'across_mean': float(across_arr.mean()),
                'along_std': float(along_arr.std()),
                'across_std': float(across_arr.std()),
                'overall_rmse': float(np.sqrt(np.mean(along_arr**2 + across_arr**2))),
                'ce90': float(np.percentile(radial, 90)),
            }
            rec.update(params)
            results.append(rec)

        if results:
            pd.DataFrame(results).to_csv(os.path.join(output_dir, 'SUMMARY_ALL.csv'), index=False)



# =============================================================================
# MANUAL GCP LOADER
# =============================================================================
class ManualGCPLoader:
    """
    Loads agency-provided GCP CSV and computes along/across errors
    in the same UTM-metre convention used by the pipeline:
        across_error = Map_X - Map_X_ref   (Easting diff)
        along_error  = Map_Y - Map_Y_ref   (Northing diff)
    CSV required columns: scan, pix, Map_X, Map_Y, Map_X_ref, Map_Y_ref
    """
    REQUIRED = {'scan', 'pix', 'Map_X', 'Map_Y', 'Map_X_ref', 'Map_Y_ref'}

    @staticmethod
    def load(gcp_csv: str) -> pd.DataFrame:
        df = pd.read_csv(gcp_csv)
        missing = ManualGCPLoader.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f'GCP file missing columns: {missing}')
        df['across_error'] = df['Map_X'] - df['Map_X_ref']
        df['along_error']  = df['Map_Y'] - df['Map_Y_ref']
        df['pix_col']      = df['pix'].astype(float)
        df['scan_row']     = df['scan'].astype(float)
        print(f'ManualGCP: loaded {len(df)} GCPs')
        print(f'  along_error  mean={df["along_error"].mean():.2f} m  '
              f'std={df["along_error"].std():.2f} m')
        print(f'  across_error mean={df["across_error"].mean():.2f} m  '
              f'std={df["across_error"].std():.2f} m')
        return df


# =============================================================================
# ERROR SURFACE MODEL  (bilinear / biquadratic + zone-adaptive RANSAC)
# =============================================================================
class ErrorSurfaceModel:
    """
    Fits a 2-D polynomial error surface to the manually observed GCPs using
    RANSAC with per-zone adaptive thresholds.

    Spatial coords : pix_col (range direction) x scan_row (azimuth direction)
    Mode           : 'bilinear'    — 4 coefficients  (1, x, y, xy)
                     'biquadratic' — 9 coefficients  (adds x², y², x²y, xy², x²y²)

    Workflow
    --------
    1. Zone-detrend along PIX axis (n_pix_zones slabs) to flatten spatial
       gradient *for outlier detection only*.
    2. RANSAC with adaptive per-zone threshold = k × zone-MAD (floor 1 m,
       cap 3 × ransac_thresh).
    3. Final LSQ fit on original (non-detrended) errors, inlier set only.
    4. predict(pix_col, scan_row) → (pred_along, pred_across) for any chip.
    """

    def __init__(self, mode: str = 'bilinear',
                 ransac_thresh_m: float = 5.0,
                 ransac_iters: int = 2000,
                 min_inlier_ratio: float = 0.5,
                 n_pix_zones: int = 4):
        self.mode             = mode
        self.ransac_thresh    = ransac_thresh_m
        self.ransac_iters     = ransac_iters
        self.min_inlier_ratio = min_inlier_ratio
        self.n_pix_zones      = n_pix_zones
        self.ca = self.cc = None
        self.inlier_mask: Optional[np.ndarray] = None

    def _A(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.mode == 'bilinear':
            return np.column_stack([np.ones_like(x), x, y, x * y])
        return np.column_stack([np.ones_like(x), x, y, x * y,
                                x**2, y**2, x**2 * y, x * y**2, x**2 * y**2])

    def _min_pts(self) -> int:
        return 4 if self.mode == 'bilinear' else 9

    def _fit_coeffs(self, x, y, za, zc):
        A = self._A(x, y)
        ca, *_ = np.linalg.lstsq(A, za, rcond=None)
        cc, *_ = np.linalg.lstsq(A, zc, rcond=None)
        return ca, cc

    def _zone_detrend(self, x, za, zc):
        """Subtract per-pix-zone median — used ONLY during RANSAC."""
        zone_edges = np.linspace(x.min(), x.max(), self.n_pix_zones + 1)
        za_d, zc_d = za.copy(), zc.copy()
        for i in range(self.n_pix_zones):
            lo, hi = zone_edges[i], zone_edges[i + 1]
            mask = (x >= lo) & (x <= hi)
            if mask.sum() < 2:
                continue
            za_d[mask] -= np.median(za[mask])
            zc_d[mask] -= np.median(zc[mask])
        return za_d, zc_d

    def _adaptive_thresh(self, x, za_d, zc_d, k: float = 2.5) -> np.ndarray:
        """Per-point threshold = k × zone-MAD on detrended errors."""
        thresholds = np.full(len(x), self.ransac_thresh)
        zone_edges = np.linspace(x.min(), x.max(), self.n_pix_zones + 1)
        for i in range(self.n_pix_zones):
            lo, hi = zone_edges[i], zone_edges[i + 1]
            mask = (x >= lo) & (x <= hi)
            if mask.sum() < 3:
                continue
            mad_a = np.median(np.abs(za_d[mask] - np.median(za_d[mask])))
            mad_c = np.median(np.abs(zc_d[mask] - np.median(zc_d[mask])))
            zone_thresh = k * max(mad_a, mad_c, 1.0)
            thresholds[mask] = min(zone_thresh, self.ransac_thresh * 3)
        return thresholds

    def fit(self, gcp_df: pd.DataFrame):
        x  = gcp_df['pix_col'].values.astype(float)
        y  = gcp_df['scan_row'].values.astype(float)
        za = gcp_df['along_error'].values.astype(float)
        zc = gcp_df['across_error'].values.astype(float)
        n  = len(x)
        k  = self._min_pts()

        za_d, zc_d = self._zone_detrend(x, za, zc)
        thresholds = self._adaptive_thresh(x, za_d, zc_d)

        best_inliers = np.zeros(n, dtype=bool)
        rng = np.random.default_rng(42)

        for _ in range(self.ransac_iters):
            idx = rng.choice(n, k, replace=False)
            try:
                ca_t, cc_t = self._fit_coeffs(x[idx], y[idx],
                                              za_d[idx], zc_d[idx])
            except np.linalg.LinAlgError:
                continue
            A  = self._A(x, y)
            ra = za_d - A @ ca_t
            rc = zc_d - A @ cc_t
            res = np.sqrt(ra**2 + rc**2)
            inliers = res < thresholds
            if inliers.sum() > best_inliers.sum():
                best_inliers = inliers

        if best_inliers.sum() / n < self.min_inlier_ratio:
            print(f'WARN ErrorSurface: only {best_inliers.sum()}/{n} inliers '
                  f'— falling back to full LSQ')
            best_inliers = np.ones(n, dtype=bool)

        self.inlier_mask = best_inliers
        self.ca, self.cc = self._fit_coeffs(
            x[best_inliers], y[best_inliers],
            za[best_inliers], zc[best_inliers])

        A_in  = self._A(x[best_inliers], y[best_inliers])
        rmse_a = np.sqrt(np.mean((za[best_inliers] - A_in @ self.ca)**2))
        rmse_c = np.sqrt(np.mean((zc[best_inliers] - A_in @ self.cc)**2))
        outlier_ids = gcp_df.index[~best_inliers].tolist()
        print(f'ErrorSurface ({self.mode}+RANSAC+adaptive): '
              f'{best_inliers.sum()} inliers, {(~best_inliers).sum()} rejected')
        print(f'  Final fit RMSE: along={rmse_a:.2f}m  across={rmse_c:.2f}m')
        if outlier_ids:
            print(f'  Rejected GCP indices (human-error candidates): {outlier_ids}')

    def predict(self, pix_col, scan_row):
        if self.ca is None:
            raise RuntimeError('ErrorSurfaceModel.fit() must be called before predict()')
        A = self._A(pix_col, scan_row)
        return A @ self.ca, A @ self.cc


# =============================================================================
# CHIP CONSENSUS
# =============================================================================
class ChipConsensusSelector:
    @staticmethod
    def _parse_filename(filename: str) -> Optional[Dict]:
        return MatchStatistics._parse_filename(filename)

    @staticmethod
    def _config_key(params: Dict):
        return (
            params.get('detector'),
            params.get('disk_mode'),
            params.get('det_wt'),
            params.get('desc_wt'),
            params.get('nisar_pol'),
            params.get('s1_ref_tag'),
            params.get('match_method'),
            params.get('match_parameter'),
            params.get('ransac_method'),
            params.get('ransac_threshold'),
            params.get('ransac_confidence'),
        )

    @staticmethod
    def _chip_key(params: Dict):
        return (
            params.get('pair_id'),
            params.get('scan'),
            params.get('pix'),
        )

    @staticmethod
    def _mode_center(values, bin_size=0.5):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan
        if len(vals) == 1:
            return float(vals[0])

        lo = math.floor(vals.min() / bin_size) * bin_size
        hi = math.ceil(vals.max() / bin_size) * bin_size + bin_size
        bins = np.arange(lo, hi + bin_size, bin_size)
        hist, edges = np.histogram(vals, bins=bins)
        idx = int(np.argmax(hist))
        left = edges[idx]
        right = edges[idx + 1]
        members = vals[(vals >= left) & (vals < right)]
        if len(members) == 0:
            return float((left + right) / 2.0)
        return float(np.median(members))

    SURFACE_TERMS = {'affine': 3, 'bilinear': 4, 'quadratic': 6, 'biquadratic': 9}

    @staticmethod
    def _surface_design(u, v, degree):
        cols = [np.ones(len(u)), u, v]
        if degree in ('bilinear', 'quadratic', 'biquadratic'):
            cols.append(u * v)
        if degree in ('quadratic', 'biquadratic'):
            cols += [u * u, v * v]
        if degree == 'biquadratic':
            cols += [u * u * v, u * v * v, u * u * v * v]
        return np.column_stack(cols)

    @staticmethod
    def _surface_keep(x, y, a, c, tol: float, degree: str = 'auto', k_sigma: float = 3.0,
                      iters: int = 6):
        """Robust smooth surface through chip mean errors (a = along, c = across)
        over chip position (x, y).

        Fitted by iterative trimming: fit, measure the residual scale robustly
        (1.4826 x median absolute residual of the kept chips), keep chips within
        max(tol, k_sigma x scale), refit. The tolerance therefore adapts to how
        well the surface can follow the real error field, while a chip whose
        error disagrees with its surroundings by far more is still rejected.
        Returns (keep, fit_a, fit_c, degree, threshold) or None if too few chips."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        a, c = np.asarray(a, float), np.asarray(c, float)
        n = len(x)
        if degree == 'auto':
            degree = ('biquadratic' if n >= 18 else 'quadratic' if n >= 12 else 'bilinear' if n >= 8
                      else 'affine' if n >= 5 else None)
        if degree is None or n < ChipConsensusSelector.SURFACE_TERMS[degree] + 2:
            return None
        u = (x - x.mean()) / max(np.ptp(x) / 2.0, 1e-9)
        v = (y - y.mean()) / max(np.ptp(y) / 2.0, 1e-9)
        A = ChipConsensusSelector._surface_design(u, v, degree)
        k = A.shape[1]
        keep = np.ones(n, dtype=bool)
        thr = tol
        fa = fc = None
        for _ in range(iters):
            if keep.sum() < k:
                return None
            ca = np.linalg.lstsq(A[keep], a[keep], rcond=None)[0]
            cc = np.linalg.lstsq(A[keep], c[keep], rcond=None)[0]
            fa, fc = A @ ca, A @ cc
            r = np.maximum(np.abs(a - fa), np.abs(c - fc))
            thr = max(tol, k_sigma * 1.4826 * float(np.median(r[keep])))
            new = r <= thr
            if np.array_equal(new, keep):
                break
            keep = new
        if thr > 3.0 * tol:
            # The surface cannot follow the field closely (its tolerance had to
            # grow), so a wrong chip could hide inside it: judge the chips
            # against their neighbours as well.
            keep = keep & ChipConsensusSelector._local_keep(u, v, a, c, keep, tol, k_sigma)
        return keep, fa, fc, degree, thr

    @staticmethod
    def _local_keep(u, v, a, c, keep, tol: float, k_sigma: float = 3.0, k_nn: int = 8):
        """Second, model-free test: each kept chip against a plane through its
        nearest kept neighbours (itself excluded). A smooth error field is
        predicted well locally even where one global surface cannot follow it,
        so a chip that disagrees with its surroundings stands out."""
        idx = np.nonzero(keep)[0]
        if len(idx) < 6:
            return np.ones(len(u), dtype=bool)
        P = np.column_stack([u[idx], v[idx]])
        r = np.zeros(len(idx))
        for m, i in enumerate(idx):
            d = np.hypot(*(P - P[m]).T)
            d[m] = np.inf
            nb = np.argsort(d)[:min(k_nn, len(idx) - 1)]
            A = np.column_stack([np.ones(len(nb)), P[nb, 0] - P[m, 0], P[nb, 1] - P[m, 1]])
            pa = np.linalg.lstsq(A, a[idx[nb]], rcond=None)[0][0]
            pc = np.linalg.lstsq(A, c[idx[nb]], rcond=None)[0][0]
            r[m] = max(abs(a[i] - pa), abs(c[i] - pc))
        thr = max(tol, k_sigma * 1.4826 * float(np.median(r)))
        out = np.ones(len(u), dtype=bool)
        out[idx[r > thr]] = False
        return out

    @staticmethod
    def build_chip_stats(csv_dir: str, min_inliers_per_chip: int = 6) -> pd.DataFrame:
        rows = []
        for filename in sorted(os.listdir(csv_dir)):
            if not filename.endswith('.csv'):
                continue

            params = ChipConsensusSelector._parse_filename(filename)
            if params is None:
                continue

            fpath = os.path.join(csv_dir, filename)
            true_along = []
            true_across = []

            try:
                with open(fpath, 'r', newline='') as f:
                    for row in csv.DictReader(f):
                        if 'true' in str(row.get('inliers', '')).lower():
                            true_along.append(float(row['along']))
                            true_across.append(float(row['across']))
            except Exception:
                continue

            n = len(true_along)
            if n < min_inliers_per_chip:
                continue

            ta = np.asarray(true_along, dtype=float)
            tc = np.asarray(true_across, dtype=float)

            rec = {
                'filename': filename,
                'pair_id': params['pair_id'],
                'scan': params['scan'],
                'pix': params['pix'],
                'chip_key': str(ChipConsensusSelector._chip_key(params)),
                'config_key': str(ChipConsensusSelector._config_key(params)),
                'num_inliers': int(n),
                'along_mean': float(ta.mean()),
                'across_mean': float(tc.mean()),
                'along_std': float(ta.std()),
                'across_std': float(tc.std()),
                'rmse_total': float(np.sqrt(np.mean(ta**2 + tc**2))),
            }
            rec.update(params)
            rows.append(rec)

        return pd.DataFrame(rows)

    @staticmethod
    def select_configs(
        csv_dir: str,
        output_dir: str,
        tolerance_m: float = 5.0,
        mode_bin_m: float = 0.5,
        min_inliers_per_chip: int = 6,
        min_surviving_chips: int = 3,
        manual_gcp_csv: str = '',
        filename_suffix: str = '',
        model: str = 'constant',
        surface_degree: str = 'auto',
    ):
        """
        manual_gcp_csv: path to analyst-observed GCP CSV
            Columns: scan, pix, Map_X, Map_Y, Map_X_ref, Map_Y_ref
            ManualGCPLoader converts these to (along_error, across_error).
            ErrorSurfaceModel fits a bilinear surface over (pix, scan) space
            using zone-adaptive RANSAC to reject GCP blunders.
            Each chip's residual to the surface (resid_rmse) replaces the
            meaningless chip-to-chip spread_penalty in the score.
            score = agg_inliers / (mean_resid_rmse + 1e-6)
        """
        os.makedirs(output_dir, exist_ok=True)
        df = ChipConsensusSelector.build_chip_stats(csv_dir, min_inliers_per_chip)
        if df.empty:
            print('No valid chip CSV files found.')
            return None

        # ── Fit error surface from manual GCPs ───────────────────────────────
        surf_model: Optional[ErrorSurfaceModel] = None
        if manual_gcp_csv and os.path.exists(manual_gcp_csv):
            try:
                gcp_df = ManualGCPLoader.load(manual_gcp_csv)
                surf_model = ErrorSurfaceModel(mode='bilinear')
                surf_model.fit(gcp_df)
            except Exception as e:
                # a file that cannot be used must not throw away the matching
                print(f'WARNING: manual GCP file not used ({type(e).__name__}: {e}); '
                      f'consensus scored without it')
                surf_model = None
        if surf_model is not None:
            pred_a, pred_c          = surf_model.predict(df['pix'].values,
                                                          df['scan'].values)
            df['along_resid']  = df['along_mean']  - pred_a
            df['across_resid'] = df['across_mean'] - pred_c
            df['resid_rmse']   = np.sqrt(df['along_resid']**2 +
                                          df['across_resid']**2)
        else:
            df['along_resid']  = np.nan
            df['across_resid'] = np.nan
            df['resid_rmse']   = np.nan

        df.to_csv(os.path.join(output_dir, f'CHIP_STATS_ALL{filename_suffix}.csv'), index=False)

        survivors = []
        config_scores = []

        for config_key, g in df.groupby('config_key'):
            if len(g) == 0:
                continue

            am_center = ChipConsensusSelector._mode_center(g['along_mean'].values, mode_bin_m)
            cm_center = ChipConsensusSelector._mode_center(g['across_mean'].values, mode_bin_m)
            as_center = ChipConsensusSelector._mode_center(g['along_std'].values, mode_bin_m)
            cs_center = ChipConsensusSelector._mode_center(g['across_std'].values, mode_bin_m)

            gg = g.copy()
            gg['d_along_mean'] = (gg['along_mean'] - am_center).abs()
            gg['d_across_mean'] = (gg['across_mean'] - cm_center).abs()
            gg['d_along_std'] = (gg['along_std'] - as_center).abs()
            gg['d_across_std'] = (gg['across_std'] - cs_center).abs()

            used = 'constant'
            surface_rmse = float('nan')
            fit = None
            if model == 'surface':
                fit = ChipConsensusSelector._surface_keep(
                    gg['pix'].values, gg['scan'].values, gg['along_mean'].values,
                    gg['across_mean'].values, tolerance_m, surface_degree)
            if model == 'none':
                used = 'none'
                gg['keep'] = True
            elif fit is not None:
                keep_s, fa, fc, deg, thr = fit
                used = f'surface:{deg}:tol{thr:.0f}m'
                gg['along_fit'], gg['across_fit'] = fa, fc
                # Per-chip spread follows the local error gradient when the error
                # varies across the scene, so it is not compared between chips.
                gg['keep'] = keep_s
                if keep_s.any():
                    surface_rmse = float(np.sqrt(np.mean((gg['along_mean'].values[keep_s] - fa[keep_s]) ** 2
                                                         + (gg['across_mean'].values[keep_s] - fc[keep_s]) ** 2)))
            else:
                gg['keep'] = (
                    (gg['d_along_mean'] <= tolerance_m) &
                    (gg['d_across_mean'] <= tolerance_m) &
                    (gg['d_along_std'] <= tolerance_m) &
                    (gg['d_across_std'] <= tolerance_m)
                )

            kept = gg[gg['keep']].copy()
            rejected = gg[~gg['keep']].copy()

            if len(kept) < min_surviving_chips:
                continue

            agg_inliers = int(kept['num_inliers'].sum())
            n_chips     = int(len(kept))
            n_pairs     = int(kept['pair_id'].nunique())

            along_mean  = float(kept['along_mean'].mean())
            across_mean = float(kept['across_mean'].mean())
            # chip-to-chip std kept for diagnostics only — NOT used in score
            along_std_chips  = float(kept['along_mean'].std()) if len(kept) > 1 else 0.0
            across_std_chips = float(kept['across_mean'].std()) if len(kept) > 1 else 0.0

            # ── Reference-surface residual score ─────────────────────────────
            # If GCPs provided: score = inliers / mean(resid_rmse per chip)
            # resid_rmse measures how well each chip matches the error surface
            # fitted to the analyst's manually observed points.
            # chip-to-chip spread_penalty is intentionally removed — it is
            # meaningless when offsets are spatially variant (which they are).
            if surf_model is not None and 'resid_rmse' in kept.columns:
                mean_resid = float(kept['resid_rmse'].mean())
                score = agg_inliers / (mean_resid + 1e-6)
            else:
                # Fallback: inlier-count only (no spread penalty)
                score = float(agg_inliers)
            mean_resid = float(kept['resid_rmse'].mean()) if 'resid_rmse' in kept.columns and surf_model is not None else float('nan')

            first = kept.iloc[0].to_dict()
            config_scores.append({
                'config_key': config_key,
                'detector': first.get('detector'),
                'disk_mode': first.get('disk_mode'),
                'det_wt': first.get('det_wt'),
                'desc_wt': first.get('desc_wt'),
                'nisar_pol': first.get('nisar_pol'),
                's1_ref_tag': first.get('s1_ref_tag'),
                'match_method': first.get('match_method'),
                'match_parameter': first.get('match_parameter'),
                'ransac_method': first.get('ransac_method'),
                'ransac_threshold': first.get('ransac_threshold'),
                'ransac_confidence': first.get('ransac_confidence'),
                'mode_along': am_center,
                'mode_across': cm_center,
                'mode_along_std': as_center,
                'mode_across_std': cs_center,
                'surviving_chips': n_chips,
                'surviving_pairs': n_pairs,
                'rejected_chips': int(len(rejected)),
                'agg_inliers': agg_inliers,
                'along_mean': along_mean,
                'across_mean': across_mean,
                'along_std_chips': along_std_chips,    # diagnostic only
                'across_std_chips': across_std_chips,  # diagnostic only
                'mean_resid_rmse': mean_resid,          # surface residual (primary accuracy metric)
                'consensus_model': used,
                'surface_rmse_m': surface_rmse,         # kept chips about the fitted surface
                'score': score,
            })
            survivors.append(kept)

        if not config_scores:
            print('No configuration survived the chip consensus filter.')
            return None

        chip_keep_df = pd.concat(survivors, ignore_index=True)
        scores_df = pd.DataFrame(config_scores).sort_values('score', ascending=False)

        chip_keep_df.to_csv(os.path.join(output_dir, f'CHIP_STATS_SURVIVORS{filename_suffix}.csv'), index=False)
        scores_df.to_csv(os.path.join(output_dir, f'CONSENSUS_SCORES{filename_suffix}.csv'), index=False)

        best = scores_df.iloc[0]
        best_key = best['config_key']
        best_chip_df = chip_keep_df[chip_keep_df['config_key'] == best_key].copy()
        best_chip_df.to_csv(os.path.join(output_dir, f'BEST_CHIP_SET{filename_suffix}.csv'), index=False)

        manifest_fields = [
            'filename', 'pair_id', 'scan', 'pix', 'num_inliers',
            'along_mean', 'across_mean', 'along_std', 'across_std',
            'detector', 'disk_mode', 'det_wt', 'desc_wt',
            'nisar_pol', 's1_ref_tag', 'match_method', 'match_parameter',
            'ransac_method', 'ransac_threshold', 'ransac_confidence'
        ]
        manifest_fields = [f for f in manifest_fields if f in best_chip_df.columns]

        best_chip_df[manifest_fields].to_csv(
            os.path.join(output_dir, f'BEST_CHIP_MANIFEST{filename_suffix}.csv'),
            index=False
        )

        best_summary = pd.DataFrame([best.to_dict()])
        best_summary.to_csv(os.path.join(output_dir, f'BEST_FINAL_SUMMARY{filename_suffix}.csv'), index=False)

        print('Saved:')
        print(' CHIP_STATS_ALL.csv')
        print(' CHIP_STATS_SURVIVORS.csv')
        print(' CONSENSUS_SCORES.csv')
        print(' BEST_CHIP_SET.csv')
        print(' BEST_CHIP_MANIFEST.csv')
        print(' BEST_FINAL_SUMMARY.csv')

        return {
            'best_chip_manifest_csv': os.path.join(output_dir, f'BEST_CHIP_MANIFEST{filename_suffix}.csv'),
            'best_final_summary_csv': os.path.join(output_dir, f'BEST_FINAL_SUMMARY{filename_suffix}.csv'),
            'best_config': best.to_dict(),
        }


# =============================================================================
# DETECTOR REGISTRY
# =============================================================================
@dataclass
class ParamSpec:
    """One user-settable parameter of a detector or of a matcher it uses.

    scope 'detector' parameters change the model: each combination of values
    runs as its own named variant. scope 'lgm' / 'ada' parameters change a
    matcher: each combination is an extra matching pass of the same variant.
    token is the short tag written into names when the value is not the
    default ('' when the class's own prefix already encodes the value)."""
    name: str
    kind: str                         # 'choice' | 'bool' | 'float' | 'int' | 'str'
    default: object
    choices: Optional[List] = None
    token: str = ''
    scope: str = 'detector'           # 'detector' | 'lgm' | 'ada'
    label: str = ''
    help: str = ''

    def as_dict(self) -> Dict:
        return asdict(self)


_PC = PipelineConfig  # dataclass defaults for the AdaLAM settings below
LGM_PARAMS = [
    ParamSpec('lgm.filter_threshold', 'float', 0.1, token='lf', scope='lgm',
              label='LightGlue filter threshold', help='minimum match confidence (kornia default 0.1)'),
    ParamSpec('lgm.depth_confidence', 'float', 0.95, token='ld', scope='lgm',
              label='LightGlue depth confidence', help='early-stop confidence; -1 disables early stopping'),
    ParamSpec('lgm.width_confidence', 'float', 0.99, token='lw', scope='lgm',
              label='LightGlue width confidence', help='point-pruning confidence; -1 disables pruning'),
]
ADA_PARAMS = [
    ParamSpec('ada.search_expansion', 'int', _PC.adalam_search_expansion, token='as', scope='ada',
              label='AdaLAM search expansion', help='neighbourhood radius multiplier'),
    ParamSpec('ada.ransac_iters', 'int', _PC.adalam_ransac_iters, token='ai', scope='ada',
              label='AdaLAM RANSAC iterations'),
    ParamSpec('ada.min_confidence', 'int', _PC.adalam_min_confidence, token='ac', scope='ada',
              label='AdaLAM min confidence', help='minimum inlier confidence per neighbourhood'),
    ParamSpec('ada.min_inliers', 'int', 6, token='am', scope='ada', label='AdaLAM min inliers'),
    ParamSpec('ada.refit', 'bool', _PC.adalam_refit, token='ar', scope='ada', label='AdaLAM refit'),
    ParamSpec('ada.force_seed_mnn', 'bool', _PC.adalam_force_seed_mnn, token='afs', scope='ada',
              label='AdaLAM mutual-NN seeds only', help='force_seed_mnn: seeds must be mutual nearest neighbours'),
]

# kornia's plain descriptor matchers, offered for every descriptor-based
# detector but run only when selected (the defaults stay smnn / lgm / ada).
GENERIC_MATCHERS = ['mnn', 'snn', 'nn', 'fginn']
SNN_PARAMS = [
    ParamSpec('snn.th', 'float', 0.8, token='sn', scope='snn', label='SNN ratio threshold',
              help="Lowe's ratio test: best / second-best descriptor distance must be below this"),
]
FGINN_PARAMS = [
    ParamSpec('fginn.th', 'float', 0.8, token='ft', scope='fginn', label='FGINN ratio threshold'),
    ParamSpec('fginn.spatial_th', 'float', 10.0, token='fs', scope='fginn', label='FGINN spatial threshold (px)',
              help='the second neighbour is the nearest one at least this far from the first'),
    ParamSpec('fginn.mutual', 'bool', False, token='fm', scope='fginn', label='FGINN mutual check'),
]
GENERIC_PARAMS = SNN_PARAMS + FGINN_PARAMS


def _dedode_weight_names(kind: str, fallback: List[str]) -> List[str]:
    """DeDoDe weight names the INSTALLED kornia knows ('detector' or
    'descriptor'), so the dialog never offers a weight that cannot load."""
    try:
        from kornia.feature.dedode import dedode as _dd
        names = list((getattr(_dd, 'urls', {}) or {}).get(kind, {}).keys())
        return names or fallback
    except Exception:
        return fallback


_DEDODE_DET = _dedode_weight_names('detector', ['L-upright', 'L-C4', 'L-SO2', 'L-C4-v2'])
_DEDODE_DESC = _dedode_weight_names('descriptor', ['B-upright', 'B-C4', 'B-SO2', 'G-upright', 'G-C4'])

DETECTOR_PARAMS: Dict[str, List[ParamSpec]] = {
    'sift': [
        ParamSpec('rootsift', 'bool', True, token='rs', label='RootSIFT descriptors'),
        ParamSpec('upright', 'bool', True, token='up', label='Upright (no orientation)'),
        ParamSpec('score_threshold', 'float', 0.0, token='st', label='Response threshold'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'disk': [
        ParamSpec('checkpoint', 'choice', 'depth', ['depth', 'epipolar'], label='Weights'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'dedode': [
        ParamSpec('detector_weights', 'choice',
                  'L-C4' if 'L-C4' in _DEDODE_DET else _DEDODE_DET[0], _DEDODE_DET,
                  label='Detector weights', help='names read from the installed kornia'),
        ParamSpec('descriptor_weights', 'choice',
                  'G-C4' if 'G-C4' in _DEDODE_DESC else _DEDODE_DESC[0], _DEDODE_DESC,
                  label='Descriptor weights',
                  help='names read from the installed kornia; G-* load a 1.2 GB DINOv2-L backbone'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'aliked': [
        ParamSpec('model_name', 'choice', 'aliked-n16',
                  ['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'], label='Model'),
        ParamSpec('detection_threshold', 'float', 0.2, token='dt', label='Detection threshold'),
        ParamSpec('nms_radius', 'int', 2, token='nms', label='NMS radius'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'xfeat': [
        ParamSpec('detection_threshold', 'float', 0.05, token='dt', label='Detection threshold'),
    ] + ADA_PARAMS + GENERIC_PARAMS,
    'xfeatstar': [],
    'keynet': [
        ParamSpec('affnet', 'bool', True, token='aff', label='AffNet affine shapes'),
        ParamSpec('upright', 'bool', True, token='up', label='Upright (no orientation)'),
        ParamSpec('score_threshold', 'float', 0.0, token='st', label='Response threshold'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'dog': [
        ParamSpec('descriptor', 'choice', 'hardnet', ['hardnet', 'hardnet8', 'sosnet', 'hynet', 'tfeat'],
                  label='Descriptor', help="LightGlue ('lgm') runs for HardNet only"),
        ParamSpec('affnet', 'bool', False, token='aff', label='AffNet affine shapes'),
        ParamSpec('upright', 'bool', True, token='up', label='Upright (no orientation)'),
        ParamSpec('score_threshold', 'float', 0.0, token='st', label='Response threshold'),
    ] + LGM_PARAMS + ADA_PARAMS + GENERIC_PARAMS,
    'gftt': [ParamSpec('upright', 'bool', True, token='up', label='Upright (no orientation)')]
    + ADA_PARAMS + GENERIC_PARAMS,
    'hessian': [ParamSpec('upright', 'bool', True, token='up', label='Upright (no orientation)')]
    + ADA_PARAMS + GENERIC_PARAMS,
    'loftr': [
        ParamSpec('pretrained', 'choice', 'outdoor', ['outdoor', 'indoor', 'indoor_new'],
                  token='w', label='Weights'),
        ParamSpec('coarse_threshold', 'float', 0.2, token='ct', label='Coarse match threshold',
                  help='confidence a coarse match needs (kornia default 0.2)'),
    ],
}

# kornia implementations, built with the detector-scope parameters as kwargs.
# They take priority: an external bridge (imcui) may only register algorithms
# that are NOT listed here -- see register_detector.
KORNIA_DETECTORS: Dict[str, Callable[['PipelineConfig', Dict], DiskBasedMatcher]] = {
    'sift':      lambda c, kw: SIFTMatcher(c, **kw),
    'disk':      lambda c, kw: DISKMatcher(c, **kw),
    'dedode':    lambda c, kw: DeDoDeMatcher(c, **kw),
    'aliked':    lambda c, kw: ALIKEDMatcher(c, **kw),
    'xfeat':     lambda c, kw: XFeatMatcher(c, **kw),
    'xfeatstar': lambda c, kw: XFeatStarMatcher(c),
    'keynet':    lambda c, kw: KeyNetMatcher(c, **kw),
    'dog':       lambda c, kw: DoGDescriptorMatcher(c, **kw),
    'gftt':      lambda c, kw: AffNetHardNetMatcher(c, kind='gftt', **kw),
    'hessian':   lambda c, kw: AffNetHardNetMatcher(c, kind='hessian', **kw),
    'loftr':     lambda c, kw: LoFTRMatcher(c, **kw),
}
# Matchers each kornia detector runs by default (for GUIs; the classes are the
# truth), and the ones it offers on request.
KORNIA_MATCHERS: Dict[str, List[str]] = {
    'sift': ['smnn', 'lgm', 'ada'], 'disk': ['smnn', 'lgm', 'ada'], 'dedode': ['smnn', 'lgm', 'ada'],
    'aliked': ['smnn', 'lgm', 'ada'], 'xfeat': ['smnn', 'ada'], 'xfeatstar': ['internal'],
    'keynet': ['smnn', 'lgm', 'ada'], 'loftr': ['loftr_internal'],
    'dog': ['smnn', 'lgm', 'ada'], 'gftt': ['smnn', 'ada'], 'hessian': ['smnn', 'ada'],
}
OPTIONAL_MATCHERS: Dict[str, List[str]] = {
    d: list(GENERIC_MATCHERS) for d, ms in KORNIA_MATCHERS.items() if 'smnn' in ms}
# Algorithm families kornia covers; an external detector naming one of these
# is refused (e.g. imcui 'disk-lightglue' -> use kornia 'disk' + 'lgm').
KORNIA_FAMILIES = {'sift', 'rootsift', 'disk', 'dedode', 'aliked', 'xfeat',
                   'xfeat_dense', 'xfeatstar', 'keynet', 'loftr',
                   'dog', 'hardnet', 'hardnet8', 'sosnet', 'hynet', 'tfeat', 'gftt', 'hessian'}
# The kornia detector that replaces an external row of a covered family
# (shown when an imcui row is not offered).
KORNIA_EQUIVALENT: Dict[str, str] = {
    'sift': 'sift', 'rootsift': 'sift', 'disk': 'disk', 'dedode': 'dedode', 'aliked': 'aliked',
    'xfeat': 'xfeat', 'xfeat_dense': 'xfeatstar', 'xfeatstar': 'xfeatstar', 'keynet': 'keynet',
    'loftr': 'loftr', 'dog': 'dog', 'hardnet': 'dog', 'hardnet8': 'dog', 'sosnet': 'dog',
    'hynet': 'dog', 'tfeat': 'dog', 'gftt': 'gftt', 'hessian': 'hessian',
}
# Older names -> (registry name, fixed parameters).
DETECTOR_ALIASES: Dict[str, Tuple[str, Dict[str, List]]] = {
    'disk_depth': ('disk', {'checkpoint': ['depth']}),
    'disk_epipolar': ('disk', {'checkpoint': ['epipolar']}),
}

EXTERNAL_DETECTORS: Dict[str, Dict] = {}


def register_detector(name: str, factory: Callable, families=(), source: str = 'external',
                      matchers: Optional[List[str]] = None,
                      params: Optional[List[ParamSpec]] = None) -> bool:
    """Register a non-kornia detector. factory(config, params_dict) builds one
    variant. Refused (returns False) when kornia already provides the
    algorithm (by name or by any of `families`)."""
    fam = {f.lower() for f in families}
    if name in KORNIA_DETECTORS or fam & KORNIA_FAMILIES:
        covered = sorted(fam & KORNIA_FAMILIES) or [name]
        print(f'[Registry] {name} ({source}) not registered: {", ".join(covered)} '
              f'is provided by kornia')
        return False
    EXTERNAL_DETECTORS[name] = {'factory': factory, 'source': source,
                                'matchers': matchers or ['internal'], 'params': list(params or [])}
    return True


def resolve_detector_name(name: str) -> Tuple[str, Dict[str, List]]:
    """(registry name, fixed parameters) for a selectable or legacy name."""
    if name in DETECTOR_ALIASES:
        return DETECTOR_ALIASES[name]
    if name in EXTERNAL_DETECTORS:
        return name, {}
    return name.lower(), {}


def detector_param_specs(name: str) -> List[ParamSpec]:
    base, _ = resolve_detector_name(name) if name else (name, {})
    if base in DETECTOR_PARAMS:
        return DETECTOR_PARAMS[base]
    return list(EXTERNAL_DETECTORS.get(base, {}).get('params', []))


def param_token_value(v) -> str:
    """Value as a file-name-safe token: True->1, 0.25->0p25, -1->m1."""
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return f'{v:g}'.replace('-', 'm').replace('.', 'p').replace('+', '')
    if isinstance(v, int):
        return str(v).replace('-', 'm')
    return re.sub(r'[^A-Za-z0-9]', '', str(v))[:16] or 'x'


def _coerce(sp: ParamSpec, v):
    if sp.kind == 'bool':
        if isinstance(v, str):
            low = v.strip().lower()
            if low not in ('1', '0', 'true', 'false', 'yes', 'no', 'on', 'off'):
                raise ValueError(f'{sp.name}: {v!r} is not a yes/no value')
            return low in ('1', 'true', 'yes', 'on')
        return bool(v)
    if sp.kind == 'int':
        f = float(v)
        if f != int(f):
            raise ValueError(f'{sp.name}: {v!r} is not a whole number')
        return int(f)
    if sp.kind == 'float':
        return float(v)
    if sp.kind == 'choice':
        if v not in (sp.choices or []):
            raise ValueError(f'{sp.name}: {v!r} is not one of {sp.choices}')
        return v
    return str(v)


def normalize_param_values(name: str, params: Optional[Dict]) -> Dict[str, List]:
    """Validate and type user values: {param: [values]} (scalars allowed).
    Raises ValueError naming the detector and parameter."""
    specs = {sp.name: sp for sp in detector_param_specs(name)}
    out: Dict[str, List] = {}
    for key, vals in (params or {}).items():
        if key not in specs:
            raise ValueError(f'{name}: unknown parameter {key!r}; known: {sorted(specs) or "none"}')
        vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]
        if not vals:
            continue
        try:
            typed = [_coerce(specs[key], v) for v in vals]
        except (TypeError, ValueError) as e:
            raise ValueError(f'{name}: {e}') from None
        out[key] = list(dict.fromkeys(typed))
    return out


def expand_variants(name: str, params: Optional[Dict] = None) -> List[Tuple[Dict, Dict, str]]:
    """[(detector kwargs, matcher grid, name suffix)] -- one per combination of
    detector-scope values. Matcher-scope values stay as lists in the grid."""
    base, fixed = resolve_detector_name(name)
    vals = normalize_param_values(base, {**(params or {}), **fixed})
    specs = detector_param_specs(base)
    det = [sp for sp in specs if sp.scope == 'detector']
    axes = [vals.get(sp.name, [sp.default]) for sp in det]
    grid: Dict[str, Dict[str, List]] = {}
    for sp in specs:
        if sp.scope != 'detector':
            grid.setdefault(sp.scope, {})[sp.name.split('.', 1)[1]] = vals.get(sp.name, [sp.default])
    out = []
    for combo in (itertools.product(*axes) if det else [()]):
        kwargs = {sp.name: v for sp, v in zip(det, combo)}
        toks = [f'{sp.token}{param_token_value(v)}' for sp, v in zip(det, combo)
                if sp.token and v != sp.default]
        out.append((kwargs, grid, ('-' + '-'.join(toks)) if toks else ''))
    return out


def count_runs(name: str, params: Optional[Dict], matchers: List[str], n_smnn: int) -> Tuple[int, int]:
    """(detector variants, matching passes over all variants) for preflight.
    Counted on the variants themselves (building one loads no model), so a
    matcher a variant cannot run (lgm with DoG-SOSNet) is not counted.
    ValueError for an unsupported matcher or parameter."""
    base, _ = resolve_detector_name(name)
    cfg = PipelineConfig(smnn_thresholds=[0.5 + i for i in range(n_smnn)],
                         detector_matchers={base: list(matchers)} if matchers else {})
    variants = list(iter_variants(name, cfg, params or {}))
    passes = sum(len(v.matcher_runs()) for v in variants)
    return len(variants), passes


def available_detectors() -> List[Dict]:
    """Selectable detectors: 'matchers' lists every matcher offered,
    'default_matchers' the ones that run when the job names none."""
    out = [{'name': n, 'source': 'kornia',
            'matchers': KORNIA_MATCHERS.get(n, []) + OPTIONAL_MATCHERS.get(n, []),
            'default_matchers': KORNIA_MATCHERS.get(n, []),
            'params': [sp.as_dict() for sp in DETECTOR_PARAMS.get(n, [])]}
           for n in KORNIA_DETECTORS]
    out += [{'name': n, 'source': d['source'], 'matchers': d['matchers'],
             'default_matchers': d['matchers'],
             'params': [sp.as_dict() for sp in d.get('params', [])]}
            for n, d in EXTERNAL_DETECTORS.items()]
    return out


def build_variants(name: str, config: 'PipelineConfig',
                   params: Optional[Dict] = None) -> List[DiskBasedMatcher]:
    """One matcher instance per detector-parameter combination. Models load
    lazily, so building is cheap; run and release them one at a time."""
    return list(iter_variants(name, config, params))


def iter_variants(name: str, config: 'PipelineConfig', params: Optional[Dict] = None):
    base, _ = resolve_detector_name(name)
    if params is None:
        params = (config.detector_params or {}).get(name)
        if params is None and base != name:
            params = (config.detector_params or {}).get(base)
    if base in KORNIA_DETECTORS:
        factory = KORNIA_DETECTORS[base]
    elif base in EXTERNAL_DETECTORS:
        factory = EXTERNAL_DETECTORS[base]['factory']
    else:
        raise ValueError(f'Unknown detector {name!r}. Available: '
                         f'{[d["name"] for d in available_detectors()]}')
    for kwargs, grid, suffix in expand_variants(name, params):
        m = factory(config, dict(kwargs))
        m.registry_name = base
        m.variant_suffix = suffix
        m.matcher_grid = {k: dict(v) for k, v in grid.items()}
        yield m


def build_detector(name: str, config: 'PipelineConfig', params: Optional[Dict] = None) -> DiskBasedMatcher:
    """The first variant of a detector (defaults unless params say otherwise)."""
    return next(iter_variants(name, config, params))


# =============================================================================
# PIPELINE
# =============================================================================
class AutoMatchPipeline:
    """Input scene (NISAR H5 or multi-band raster) x reference collection.

    For each selected channel: find overlapping references, build per-reference
    pairs, then for each detector estimate a coarse offset per pair, match
    windows, RANSAC-filter, compute statistics and run chip consensus."""

    def __init__(self, config: PipelineConfig, mode: str = 'same-res'):
        self.config = config
        self.mode = mode
        self.ransac_filter = RANSACFilter(config)
        self._manual_gcp_csv = config.manual_gcp_csv
        self.scene: Optional[InputScene] = None

    # ── references ───────────────────────────────────────────────────────────
    def _reference_for(self, channel: str) -> Tuple[str, str, int]:
        cfg = self.config
        ref_band = (cfg.reference_band_map or {}).get(channel, cfg.reference_band)
        if channel in ('HH', 'VV') and cfg.reference_dir_vv:
            return cfg.reference_dir_vv, 'S1VV', ref_band
        if channel in ('HV', 'VH') and cfg.reference_dir_vh:
            return cfg.reference_dir_vh, 'S1VH', ref_band
        if not cfg.reference_dir:
            raise ValueError(f'No reference folder configured for {channel}')
        label = cfg.reference_label or refs.reference_label(cfg.reference_dir)
        label = re.sub(r'[^A-Za-z0-9]', '', label) or 'REF'
        return cfg.reference_dir, label, ref_band

    # Backward-compatible name used by the S1 pipeline.
    def _reference_dir_for_pol(self, pol: str) -> Tuple[str, str]:
        d, tag, _ = self._reference_for(pol)
        return d, tag

    def _iter_matchers(self, detector_types: List[str]):
        """Every detector variant, built one at a time so only one set of
        model weights is resident: the caller unloads each before the next
        is constructed. Duplicate names (e.g. 'disk' and 'disk_depth') run once."""
        seen = set()
        for d in detector_types:
            for m in iter_variants(d, self.config):
                key = m.get_filename_prefix()
                if key in seen:
                    continue
                seen.add(key)
                yield m

    def _build_matchers(self, detector_types: List[str]) -> List[DiskBasedMatcher]:
        return list(self._iter_matchers(detector_types))

    # ── one channel ──────────────────────────────────────────────────────────
    def _prepare_pairs(self, channel: str) -> Tuple[List[Dict], str, str]:
        cfg, scene = self.config, self.scene
        ref_dir, tag, ref_band = self._reference_for(channel)
        base_cache = scene.cache_dir(cfg.temp_dir)
        pair_dir = os.path.join(base_cache, f'{channel}_to{tag}')
        os.makedirs(pair_dir, exist_ok=True)
        target_res = cfg.target_resolution if self.mode == 'same-res' else None

        cfg_for_key = PipelineConfig(**{**asdict(cfg), 'reference_band': ref_band})
        key = pair_cache_key(scene, channel, ref_dir, cfg_for_key)
        pre = DiskBasedPreprocessor(pair_dir, target_res, ref_band, cfg.reference_fill_values)

        pairs = pre.load_existing_pairs(key) if (cfg.use_disk_cache and cfg.check_existing_pairs) else None
        if pairs is None:
            report('stage', stage='footprint', channel=channel)
            channel_path, band_idx = scene.channel_raster(channel, base_cache)
            footprint, src = scene.footprint_lonlat(channel_path, band_idx)
            print(f'[{channel}] footprint from {src}: '
                  f'{", ".join(f"{v:.3f}" for v in footprint.bounds)} (lon/lat)')
            report('stage', stage='references', channel=channel)
            fetcher = ReferenceFetcher(ref_dir, decimation=cfg.s1_decimation, min_area=cfg.min_area,
                                       mode=cfg.reference_mode, fill_values=cfg.reference_fill_values)
            found = fetcher.fetch_overlapping_references(footprint, scene.working_crs,
                                                         cfg.max_expected_error_m)
            if not found:
                raise RuntimeError(f'No reference in {ref_dir} overlaps {scene.name} {channel} '
                                   f'(search buffer {cfg.max_expected_error_m / 1000:.0f} km)')
            report('stage', stage='preprocess', channel=channel)
            pairs = pre.create_all_pairs(scene, found, channel, tag, channel_path, band_idx, key)
            if not pairs:
                raise RuntimeError(f'All {len(found)} reference pairs failed to preprocess')
        return pairs, ref_dir, tag

    def _run_single_pol(self, scene_info, pol: str, detector_types: List[str]) -> List[Dict]:
        cfg = self.config
        disk_pairs, reference_dir, s1_ref_tag = self._prepare_pairs(pol)
        pol_out_dir = os.path.join(cfg.output_base_dir, f'{pol}_to{s1_ref_tag}')
        os.makedirs(pol_out_dir, exist_ok=True)
        suffix = 'same-res' if self.mode == 'same-res' else 'multi-res'
        aligner = CoarseAligner(cfg)
        shared_coarse: Dict[int, Dict] = {}
        matcher_free = (cfg.coarse_method or 'auto').lower() in ('none', 'manual', 'phasecorr')

        run_records = []
        for matcher in self._iter_matchers(detector_types):
            tag = matcher.get_filename_prefix()
            print('-' * 80)
            print(f'[{pol}] Running {tag}')
            report('detector', channel=pol, detector=tag)
            rec = {'nisar_pol': pol, 's1_ref_tag': s1_ref_tag, 'reference_dir': reference_dir,
                   'detector_tag': tag, 'output_dir': pol_out_dir, 'status': 'ok', 'error': ''}
            t_det = time.time()
            if th.cuda.is_available():
                try:
                    th.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            try:
                matcher.selected_matchers()  # validate the matcher selection early
                report('stage', stage='warmup', channel=pol, detector=tag)
                matcher.warmup()             # missing weights stop here, in seconds
                offsets, all_match_data = [], []
                ests: Dict[int, Dict] = {}
                for pair in disk_pairs:
                    pid = pair['pair_id']
                    if matcher_free and pid in shared_coarse:
                        ests[pid] = shared_coarse[pid]
                    else:
                        report('stage', stage='coarse', channel=pol, detector=tag, pair=pid)
                        ests[pid] = aligner.estimate(pair, None if matcher_free else matcher)
                        if matcher_free:
                            shared_coarse[pid] = ests[pid]
                self._borrow_failed_offsets(disk_pairs, ests)
                for pair in disk_pairs:
                    pid = pair['pair_id']
                    est = ests[pid]
                    print(f'[Coarse] {tag} pair {pid}: dE={est["dx"]:.1f} m dN={est["dy"]:.1f} m '
                          f'({est["method"]}, support={est.get("support")}, peak={est.get("peak")}) '
                          f'{CoarseAligner._field_summary(est.get("field"))}')
                    offsets.append({'pair_id': pid, 'reference': os.path.basename(pair.get('reference_path', '')),
                                    **est})
                    margin = 0.0 if est['method'] == 'none' else cfg.search_margin_m * est.get('margin_scale', 1.0)
                    work = dict(pair, coarse_dx=est['dx'], coarse_dy=est['dy'],
                                coarse_method=est['method'], coarse_field=est.get('field'),
                                search_margin_m=margin)
                    all_match_data.extend(matcher.process_disk_cached_pair(work))
                safe_cuda_empty_cache()

                raw_dir = os.path.join(pol_out_dir, f'raw_matches_{suffix}_{tag}')
                # A rerun replaces this detector's outputs: files left by an
                # earlier run (other RANSAC / matcher settings) would otherwise
                # be read back into this run's statistics and consensus.
                for stage in ('raw_matches', 'filtered', 'statistics', 'final'):
                    shutil.rmtree(os.path.join(pol_out_dir, f'{stage}_{suffix}_{tag}'), ignore_errors=True)
                matcher.save_matches_to_csv(all_match_data, raw_dir)
                pd.DataFrame([{k: v for k, v in o.items() if k != 'field'} for o in offsets]).to_csv(
                    os.path.join(raw_dir, 'COARSE_OFFSETS.csv'), index=False)
                for o in offsets:  # the local offset field of each pair, cell by cell
                    f = o.get('field')
                    if not f:
                        continue
                    rows = [{'row': j, 'col': i,
                             'x_center': f['x0'] + (i + 0.5) * f['cw'],
                             'y_center': f['y1'] - (j + 0.5) * f['ch'],
                             'dE_m': f['dx'][j][i], 'dN_m': f['dy'][j][i],
                             'support': f['support'][j][i], 'filled_from_neighbour': f['filled'][j][i]}
                            for j in range(f['ny']) for i in range(f['nx'])]
                    pd.DataFrame(rows).to_csv(
                        os.path.join(raw_dir, f'COARSE_FIELD_pair{int(o["pair_id"]):03d}.csv'), index=False)

                filter_dir = os.path.join(pol_out_dir, f'filtered_{suffix}_{tag}')
                self.ransac_filter.filter_matches_in_memory(all_match_data, filter_dir)
                stats_dir = os.path.join(pol_out_dir, f'statistics_{suffix}_{tag}')
                MatchStatistics.compute_statistics(filter_dir, stats_dir)
                final_dir = os.path.join(pol_out_dir, f'final_{suffix}_{tag}')
                summary = ChipConsensusSelector.select_configs(
                    csv_dir=filter_dir, output_dir=final_dir,
                    tolerance_m=cfg.consensus_tolerance_m, mode_bin_m=cfg.consensus_mode_bin_m,
                    min_inliers_per_chip=cfg.min_inliers_per_chip,
                    min_surviving_chips=cfg.min_surviving_chips,
                    manual_gcp_csv=self._manual_gcp_csv,
                    filename_suffix=cfg.consensus_filename_suffix,
                    model=cfg.consensus_model,
                    surface_degree=cfg.consensus_surface_degree,
                )
                rec.update({
                    'raw_dir': raw_dir, 'filtered_dir': filter_dir, 'statistics_dir': stats_dir,
                    'final_dir': final_dir, 'n_match_sets': len(all_match_data),
                    'coarse_offsets': json.dumps([{k: o[k] for k in ('pair_id', 'dx', 'dy', 'method')}
                                                  for o in offsets]),
                    'best_chip_manifest_csv': None if summary is None else summary.get('best_chip_manifest_csv'),
                    'best_final_summary_csv': None if summary is None else summary.get('best_final_summary_csv'),
                })
                if summary is None:
                    rec['status'] = 'no-consensus'
                timing = getattr(matcher, 'pass_timing', None) or {}
                if timing:
                    os.makedirs(final_dir, exist_ok=True)
                    pd.DataFrame([{'matcher': k, 'seconds': round(v['seconds'], 2), 'windows': v['windows'],
                                   'sec_per_window': round(v['seconds'] / max(1, v['windows']), 3)}
                                  for k, v in timing.items()]).to_csv(
                        os.path.join(final_dir, 'PASS_TIMING.csv'), index=False)
                del all_match_data
            except ModelLoadError as e:
                rec.update({'status': 'failed', 'error': f'model not available: {e}'})
                print(f'[{pol}] {tag} SKIPPED -- model not available (see message above)')
            except Exception as e:
                rec.update({'status': 'failed', 'error': f'{type(e).__name__}: {e}'})
                print(f'[{pol}] {tag} FAILED: {type(e).__name__}: {e}')
                traceback.print_exc()
            finally:
                rec['seconds'] = round(time.time() - t_det, 1)
                if th.cuda.is_available():
                    try:
                        rec['gpu_peak_gb'] = round(th.cuda.max_memory_allocated() / 1024 ** 3, 2)
                    except Exception:
                        pass
                unload = getattr(matcher, 'unload_model', None)
                if callable(unload):
                    unload()
                del matcher
                safe_cuda_empty_cache()
                gc.collect()
            run_records.append(rec)
            report('detector_done', channel=pol, detector=tag, status=rec['status'])

        if cfg.cleanup_after_pair and disk_pairs:
            shutil.rmtree(os.path.dirname(disk_pairs[0]['nisar_path']), ignore_errors=True)
        return run_records

    @staticmethod
    def _borrow_failed_offsets(pairs: List[Dict], ests: Dict[int, Dict]) -> None:
        """A pair whose coarse estimate failed takes the offset of the nearest
        pair of the same scene that succeeded (its field, read at this pair's
        centre), with a doubled search margin -- all pairs share one input
        image, so a neighbour's offset is a far better guess than zero."""
        good = [p for p in pairs if ests[p['pair_id']]['method'] not in ('failed',)]
        if not good:
            return

        def centre(p):
            b = p['bounds']
            return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0

        for p in pairs:
            e = ests[p['pair_id']]
            if e['method'] != 'failed':
                continue
            cx, cy = centre(p)
            src = min(good, key=lambda q: math.hypot(centre(q)[0] - cx, centre(q)[1] - cy))
            g = ests[src['pair_id']]
            dx, dy = field_offset(g['field'], cx, cy) if g.get('field') else (g['dx'], g['dy'])
            ests[p['pair_id']] = {'dx': float(dx), 'dy': float(dy), 'support': None, 'n': None,
                                  'peak': None, 'field': None, 'margin_scale': 2.0,
                                  'method': f'from-pair-{src["pair_id"]}'}

    # ── whole scene ──────────────────────────────────────────────────────────
    def run(self, scene_dir: str, detector_types: List[str] = None):
        if detector_types is None:
            detector_types = ['disk', 'sift']
        for d in detector_types:  # fail before any preprocessing on a bad parameter
            expand_variants(d, (self.config.detector_params or {}).get(d))
        self.scene = InputScene(scene_dir, self.config)
        available = self.scene.channels
        wanted = self.config.pols or available
        missing = [c for c in wanted if c not in available]
        if missing:
            raise ValueError(f'Channel(s) {missing} not in {self.scene.name}; available: {available}')
        if not wanted:
            raise RuntimeError(f'No channels found in {self.scene.name}')

        print('=' * 80)
        print('AUTOMATCH PIPELINE')
        print('=' * 80)
        for k, v in self.scene.describe().items():
            print(f'{k:>18}: {v}')
        print(f'{"channels run":>18}: {wanted}')
        print(f'{"detectors":>18}: {detector_types}')

        all_runs = []
        for i, ch in enumerate(wanted):
            print('-' * 80)
            print(f'Running channel: {ch}')
            report('channel', channel=ch, done=i, total=len(wanted))
            try:
                all_runs.extend(self._run_single_pol(self.scene.info, ch, detector_types))
            except Exception as e:
                print(f'[{ch}] FAILED before matching: {type(e).__name__}: {e}')
                traceback.print_exc()
                all_runs.append({'nisar_pol': ch, 'status': 'failed',
                                 'error': f'{type(e).__name__}: {e}'})

        os.makedirs(self.config.output_base_dir, exist_ok=True)
        pd.DataFrame(all_runs).to_csv(
            os.path.join(self.config.output_base_dir, 'POL_RUN_SUMMARY.csv'), index=False)
        print('=' * 80)
        print('ALL CHANNELS COMPLETE')
        print('=' * 80)
        return all_runs


# Backward-compatible name.
ProductionPipelinePolwise = AutoMatchPipeline


if __name__ == '__main__':
    # The command-line entry point lives in automatch_job.py:
    #     python automatch_job.py --help
    import automatch_job
    sys.exit(automatch_job.main())
