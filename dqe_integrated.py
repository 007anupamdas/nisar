#!/usr/bin/env python3
"""
NISAR-S1 Production Matching Pipeline
Pol-wise + H5 scene reader + multi-detector + chip consensus

Merged design:
- H5/MET scene discovery and polarization handling from frozen pol-wise pipeline
- Two-step S1 overlap logic from v6
- Multi-detector / multi-matcher flow from v6 (not frozen to a single detector/matcher)
- Chip consensus selection from frozen pol-wise pipeline

NOTE: This is the import-safe version of DPQED_agdqe_all.py. The only
intentional differences from that script are:
  1. Logging is wrapped in setup_logging(output_dir) instead of running at
     import time (so `import dqe_integrated` works without sys.argv set up).
  2. PipelineConfig gains compute_num_features(window_size), used by the
     image-matching-webui bridge (dqe_imw.py) when sweeping window sizes.
  3. The hardcoded temp_dir in __main__ can be overridden with NISAR_TEMP_DIR.
  4. Large commented-out blocks (old _norm_img / _process_single_window /
     _determine_window_strategy variants) were removed for readability.
Everything else (classes, math, file naming) is identical, so its outputs are
directly comparable with the original kornia runs.
"""

import os
import re
import csv
import gc
import json
import math
import shutil
import time
import warnings
import sys
import traceback
import h5py as hp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import pandas as pd
import rasterio as rt
import rioxarray as rxr
import shapely.ops
import torch as th
import xarray as xr

from pyproj import CRS as PJCRS, Transformer
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine, from_origin, xy as rt_xy
from rasterio.warp import reproject as warp_reproject, Resampling
from shapely.geometry import Polygon, mapping
from shapely.ops import transform

print(f"kornia version: {K.__version__}")


# =============================================================================
# LOGGING
# =============================================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def setup_logging(output_dir: str) -> str:
    """Mirror stdout/stderr into <output_dir>/log.

    Call this once from your entry point (it used to run at import time,
    which broke `import dqe_integrated` from other scripts).
    """
    log_path = os.path.join(output_dir, 'log')
    os.makedirs(output_dir, exist_ok=True)
    log_file = open(log_path, 'w', buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f'Logging to {log_path}')
    return log_path


warnings.filterwarnings('ignore')

device = K.utils.get_cuda_or_mps_device_if_available()
print(f'Using device: {device}')
if th.cuda.is_available():
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.benchmark = True
    print(f'GPU: {th.cuda.get_device_name(0)}')
    print(f'CUDA: {th.version.cuda}')
    mem_total = th.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {mem_total:.1f} GB')


def safe_cuda_empty_cache():
    try:
        th.cuda.empty_cache()
    except RuntimeError:
        pass


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class PipelineConfig:
    window_size: int = 4096   # primary window for A100
    window_size_small: int = 2048  # fallback if valid fraction too low
    min_valid_fraction: float = 0.30  # skip if < 30% valid
    inpaint_radius: int = 5
    keypoint_density: float = 9000
    max_num_features: int = 16000
    num_features: int = 16000
    target_resolution: Optional[int] = 10
    loftr_max_window: int = 1024

    use_disk_cache: bool = True
    cleanup_after_pair: bool = False
    check_existing_pairs: bool = True
    use_amp: bool = True
    debug_mode: bool = False

    adalam_force_seed_mnn: bool = True
    adalam_search_expansion: int = 1
    adalam_ransac_iters: int = 2048
    adalam_min_confidence: int = 1000
    adalam_refit: bool = True

    smnn_thresholds: List[float] = None
    ransac_methods: List[int] = None
    ransac_thresholds: List[float] = None
    ransac_confidences: List[float] = None

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

    def __post_init__(self):
        if self.smnn_thresholds is None:
            self.smnn_thresholds = [0.90, 0.95, 0.99]
        if self.ransac_methods is None:
            self.ransac_methods = [4]
        if self.ransac_thresholds is None:
            self.ransac_thresholds = [1, 2, 3]
        if self.ransac_confidences is None:
            self.ransac_confidences = [0.91, 0.95, 0.99]

    def compute_num_features(self, window_size: int) -> int:
        """Keypoint budget for a given window size.

        keypoint_density is interpreted as keypoints per 1024x1024 window,
        scaled by window area and capped at max_num_features. Used by the
        image-matching-webui bridge when sweeping window sizes.
        """
        n = int(self.keypoint_density * (window_size / 1024.0) ** 2)
        return int(min(max(n, 1024), self.max_num_features))


# =============================================================================
# NISAR H5 / MET READER
# =============================================================================
class NISARH5Reader:
    GROUP_CANDIDATES = [
        'science/SSAR/GSLC/grids/frequencyA',
    ]
    POL_ORDER = ('HH', 'HV', 'VH', 'VV', 'RH', 'RV', 'LH', 'LV')

    @staticmethod
    def discover_scene(scene_dir: str) -> Dict:
        scene_dir = os.path.abspath(scene_dir)
        base = os.path.basename(os.path.normpath(scene_dir))
        h5_path = os.path.join(scene_dir, base + '.h5')
        met_path = os.path.join(scene_dir, base + '.met')

        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(f'Input must be a directory: {scene_dir}')
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f'Missing H5 file: {h5_path}')
        if not os.path.exists(met_path):
            raise FileNotFoundError(f'Missing MET file: {met_path}')

        return {
            'scene_dir': scene_dir,
            'scene_name': base,
            'h5_path': h5_path,
            'met_path': met_path,
        }

    @staticmethod
    def _find_group(h5f):
        for g in NISARH5Reader.GROUP_CANDIDATES:
            if g in h5f:
                return h5f[g]
        raise KeyError(f'Could not find GSLC frequencyA group in {h5f.filename}')

    @staticmethod
    def _load_meta_text(met_path: str) -> str:
        with open(met_path, 'r') as f:
            return f.read()

    @staticmethod
    def _load_meta_dict(met_path: str) -> Dict:
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
        met_text = NISARH5Reader._load_meta_text(scene_info['met_path']).upper()
        meta_pols = []
        for pol in NISARH5Reader.POL_ORDER:
            if re.search(rf'\b{pol}\b', met_text):
                meta_pols.append(pol)

        with hp.File(scene_info['h5_path'], 'r') as f:
            grp = NISARH5Reader._find_group(f)
            h5_pols = [pol for pol in NISARH5Reader.POL_ORDER if pol in grp.keys()]

        ordered = []
        for pol in NISARH5Reader.POL_ORDER:
            if pol in meta_pols or pol in h5_pols:
                ordered.append(pol)
        return ordered

    @staticmethod
    def choose_s1_ref_tag(pol: str) -> str:
        return 'VV' if pol in ('HH', 'VV') else 'VH'

    @staticmethod
    def open_memfile(h5_path: str, pol: str):
        with hp.File(h5_path, 'r') as f:
            grp = NISARH5Reader._find_group(f)

            if pol not in grp:
                raise KeyError(f'Polarization {pol} not present in {h5_path}. Available: {list(grp.keys())}')

            arr = np.abs(grp[pol][()]).astype('float32')
            x_coord = grp['xCoordinates'][()]
            y_coord = grp['yCoordinates'][()]
            epsg = int(grp['projection'][()])
            dx = float(x_coord[1] - x_coord[0])
            dy = float(y_coord[1] - y_coord[0])

            x_ori = float(x_coord[0] - dx / 2.0)
            y_ori = float(y_coord[0] - dy / 2.0)
            transform_aff = Affine.translation(x_ori, y_ori) * Affine.scale(dx, dy)
            crs = CRS.from_epsg(epsg)

            memfile = MemoryFile()
            with memfile.open(
                driver='GTiff',
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform_aff,
                nodata=0.0,
                BIGTIFF='YES',
                TILED='YES',
                BLOCKXSIZE=512,
                BLOCKYSIZE=512,
            ) as ds:
                ds.write(arr, 1)
        return memfile

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


# =============================================================================
# REFERENCE FETCHER
# =============================================================================
class ReferenceFetcher:
    def __init__(self, reference_dir: str, decimation: int = 10, min_area: float = 200.0):
        self.reference_dir = reference_dir
        self.decimation = decimation
        self.min_area = min_area

    @staticmethod
    def _parse_coords(line: str) -> Tuple[float, float]:
        lon = float(line.split('(')[1].split(',')[0].strip())
        lat = float(line.split('(')[1].split(',')[1].strip().split(')')[0])
        return lon, lat

    def _get_s1_valid_corners(self, tif_path: str) -> Optional[Polygon]:
        try:
            with rt.open(tif_path) as src:
                H, W = src.height, src.width
                out_h = max(50, H // self.decimation)
                out_w = max(50, W // self.decimation)
                data = src.read(1, out_shape=(out_h, out_w))
                tfm = src.transform * Affine.scale(W / out_w, H / out_h)

                URr, URc = 0, 0
                found = False
                for r in range(out_h):
                    for c in range(out_w):
                        if data[r, c] != 0:
                            URr, URc = r, c
                            found = True
                            break
                    if found:
                        break

                ULr, ULc = 0, 0
                found = False
                for c in range(out_w):
                    for r in range(out_h):
                        if data[r, c] != 0:
                            ULr, ULc = r, c
                            found = True
                            break
                    if found:
                        break

                LRr, LRc = out_h - 1, out_w - 1
                found = False
                for c in range(out_w - 1, -1, -1):
                    for r in range(out_h - 1, -1, -1):
                        if data[r, c] != 0:
                            LRr, LRc = r, c
                            found = True
                            break
                    if found:
                        break

                LLr, LLc = out_h - 1, 0
                found = False
                for r in range(out_h - 1, -1, -1):
                    for c in range(out_w - 1, -1, -1):
                        if data[r, c] != 0:
                            LLr, LLc = r, c
                            found = True
                            break
                    if found:
                        break

                URlon, URlat = rt_xy(tfm, URr, URc)
                ULlon, ULlat = rt_xy(tfm, ULr, ULc)
                LRlon, LRlat = rt_xy(tfm, LRr, LRc)
                LLlon, LLlat = rt_xy(tfm, LLr, LLc)

                poly = Polygon([
                    (ULlon, ULlat), (URlon, URlat),
                    (LRlon, LRlat), (LLlon, LLlat),
                    (ULlon, ULlat)
                ])
                return poly if poly.is_valid else poly.buffer(0)
        except Exception as e:
            print(f'[RefFetch] ERROR computing valid corners for {tif_path}: {e}')
            return None

    def fetch_overlapping_references(self, met_path: str) -> List[Dict]:
        print(f'[RefFetch] Searching in: {self.reference_dir}')

        meta, poly_nis = NISARH5Reader.valid_polygon_from_met(met_path)
        nis_crs = f"EPSG:{meta['EPSG']}"
        wgs_crs = PJCRS('epsg:4326')
        to_utm = Transformer.from_crs(wgs_crs, nis_crs, always_xy=True)

        overlapping_refs = []

        for filename in sorted(os.listdir(self.reference_dir)):
            if not filename.endswith('_meta.txt'):
                continue

            tif_name = filename.replace('_meta.txt', '.tif')
            tif_path = os.path.join(self.reference_dir, tif_name)
            if not os.path.exists(tif_path):
                continue

            with open(os.path.join(self.reference_dir, filename)) as f:
                content = f.readlines()

            corners = {}
            for item in content:
                if 'Upper Left' in item:
                    corners['UL'] = self._parse_coords(item)
                elif 'Lower Left' in item:
                    corners['LL'] = self._parse_coords(item)
                elif 'Upper Right' in item:
                    corners['UR'] = self._parse_coords(item)
                elif 'Lower Right' in item:
                    corners['LR'] = self._parse_coords(item)

            if len(corners) != 4:
                continue

            poly_s1_meta = Polygon([
                corners['UL'], corners['UR'], corners['LR'], corners['LL'], corners['UL']
            ])
            if not poly_nis.intersects(poly_s1_meta):
                continue

            print(f'[RefFetch] Candidate: {tif_name} - computing exact valid corners...')
            poly_s1_valid = self._get_s1_valid_corners(tif_path)
            if poly_s1_valid is None:
                continue
            if not poly_nis.intersects(poly_s1_valid):
                continue

            inter_wgs = poly_nis.intersection(poly_s1_valid)
            inter_utm = transform(to_utm.transform, inter_wgs)
            area_km2 = inter_utm.area / 1e6

            if area_km2 < self.min_area:
                continue

            overlapping_refs.append({
                'tif_path': tif_path,
                'utm_crs': nis_crs,
                'intersection_utm': inter_utm,
            })
            print(f'[RefFetch] OK {tif_name} | Intersection area: {area_km2:.2f} km2')

        print(f'[RefFetch] Total confirmed: {len(overlapping_refs)} references')
        return overlapping_refs


# =============================================================================
# PREPROCESSOR
# =============================================================================
class DiskBasedPreprocessor:
    def __init__(self, temp_dir: str, target_resolution: Optional[int] = None):
        self.temp_dir = temp_dir
        self.target_resolution = target_resolution
        os.makedirs(temp_dir, exist_ok=True)

    def _clip_single_pair(
        self,
        nisar_src,
        nisar_nodata_val,
        s1_ref: dict,
        pair_id: int,
        nisar_pol: str,
        s1_ref_tag: str
    ) -> Optional[Dict]:
        try:
            intersection_utm = s1_ref['intersection_utm']
            utm_crs = s1_ref['utm_crs']

            if str(nisar_src.crs) != str(utm_crs):
                print(f'[WARN] CRS mismatch: nisar_src={nisar_src.crs}, utm={utm_crs}')
                to_nisar_crs = Transformer.from_crs(utm_crs, str(nisar_src.crs), always_xy=True)
                intersection_utm = shapely.ops.transform(to_nisar_crs.transform, intersection_utm)
                utm_crs = str(nisar_src.crs)

            nisar_arr, nisar_transform = rio_mask(
                nisar_src,
                [mapping(intersection_utm)],
                crop=True,
                nodata=nisar_nodata_val,
                all_touched=True
            )

            if self.target_resolution:
                src_res = abs(nisar_src.transform.a)
                scale = src_res / self.target_resolution
                new_h = max(1, int(nisar_arr.shape[1] * scale))
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
            s1_nodata_val = s1.rio.nodata if s1.rio.nodata is not None else 0

            reproj_kwargs = {
                'dst_crs': utm_crs,
                'nodata': s1_nodata_val,
                'resampling': Resampling.bilinear,
            }
            if self.target_resolution is not None:
                reproj_kwargs['resolution'] = self.target_resolution

            s1_utm = s1.rio.reproject(**reproj_kwargs)
            s1_cropped = s1_utm.rio.clip(
                [mapping(intersection_utm)],
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
                'BIGTIFF': 'YES',
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
                BIGTIFF='YES',
                TILED='YES',
                BLOCKXSIZE=512,
                BLOCKYSIZE=512,
            )

            nisar_res = abs(nisar_transform.a)
            s1_res = abs(float(s1_cropped.rio.resolution()[0]))
            scale_factor = nisar_res / s1_res

            row_off = int((nisar_transform.f - nisar_src.transform.f) / nisar_src.transform.e)
            col_off = int((nisar_transform.c - nisar_src.transform.c) / nisar_src.transform.a)

            meta_out = os.path.join(self.temp_dir, f'pair{pair_id:03d}_meta.json')
            pair_meta = {
                'nisar_path': nisar_out,
                's1_path': s1_out,
                'pair_id': pair_id,
                'utm_crs': str(utm_crs),
                'bounds': list(intersection_utm.bounds),
                'scale_factor': scale_factor,
                'scale_per_pix': scale_factor,
                'nisar_nodata': float(nisar_nodata_val),
                's1_nodata': float(s1_nodata_val),
                'x01': float(nisar_transform.c),
                'y01': float(nisar_transform.f),
                'xres1': float(nisar_transform.a),
                'yres1': float(nisar_transform.e),
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
        h5_path: str,
        met_path: str,
        s1_references: list,
        nisar_pol: str,
        s1_ref_tag: str
    ) -> list:
        disk_pairs = []
        memfile = NISARH5Reader.open_memfile(h5_path, nisar_pol)

        try:
            with memfile.open() as nisar_src:
                nisar_nodata_val = nisar_src.nodata if nisar_src.nodata is not None else 0.0
                nisar_res = abs(nisar_src.transform.a)
                print(
                    f'[Preprocessor] NISAR H5 opened once: {nisar_src.width}x{nisar_src.height}, '
                    f'res={nisar_res}m, nodata={nisar_nodata_val}, pol={nisar_pol}, s1ref={s1_ref_tag}'
                )

                for idx, s1_ref in enumerate(s1_references, start=1):
                    pair = self._clip_single_pair(
                        nisar_src, nisar_nodata_val, s1_ref, idx, nisar_pol, s1_ref_tag
                    )
                    if pair:
                        pair['nisar_input'] = h5_path
                        pair['nisar_met'] = met_path
                        disk_pairs.append(pair)

                    gc.collect()
                    xr.backends.file_manager.FILE_CACHE.clear()
                    print(f'[Memory] Pair {idx} done, cache cleared')
        finally:
            memfile.close()

        return disk_pairs

    def load_existing_pairs(self) -> Optional[list]:
        existing = sorted([f for f in os.listdir(self.temp_dir) if f.endswith('_meta.json')])
        if not existing:
            return None

        n_pairs = len(existing)
        pairs = []
        for pair_id in range(1, n_pairs + 1):
            nisar_chip = os.path.join(self.temp_dir, f'pair{pair_id:03d}_nisar.tif')
            s1_chip = os.path.join(self.temp_dir, f'pair{pair_id:03d}_s1.tif')
            meta_file = os.path.join(self.temp_dir, f'pair{pair_id:03d}_meta.json')
            if not all(os.path.exists(p) for p in [nisar_chip, s1_chip, meta_file]):
                print(f'[Cache] Pair {pair_id} incomplete - re-running coregistration.')
                return None
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            pairs.append(meta)

        print(f'[Cache] All {n_pairs} pairs loaded from {self.temp_dir}')
        return pairs


# =============================================================================
# BASE MATCHER
# =============================================================================
class BaseMatcher(ABC):

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = device
        self.use_amp = config.use_amp and th.cuda.is_available()
        self.lgm_model = None
        self.adalam_config = self._setup_adalam_config()

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
    def _calibrate_s1_dn(arr: np.ndarray, nodata=None) -> np.ndarray:
        """ Convert S1 GRD DN to signma-n0 amp"""
        out = arr.astype(np.float32)
        valid = (out != nodata) if nodata is not None else (out > 0)
        out[valid] = out[valid] * 0.003162
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
        return x0 + (offset_x + x) * xres, y0 + (offset_y + y) * yres

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

    def get_filename_prefix(self) -> str:
        return self.get_detector_name()

    def match_smnn(self, descs1, descs2, threshold: float):
        return KF.match_smnn(descs1.squeeze(0), descs2.squeeze(0), th.tensor(threshold))

    def match_adalam(self, descs1, descs2, lafs1, lafs2, hw1, hw2):
        return KF.match_adalam(
            descs1.squeeze(0),
            descs2.squeeze(0),
            lafs1,
            lafs2,
            config=self.adalam_config,
            hw1=hw1,
            hw2=hw2
        )

    def match_lgm(self, descs1, descs2, lafs1, lafs2, hw1, hw2, feature_name='disk'):
        if self.lgm_model is None:
            print(f'{self.get_detector_name()}: Initializing LightGlue...')
            self.lgm_model = KF.LightGlueMatcher(feature_name=feature_name).eval().to(self.device)
        d1 = descs1.squeeze(0) if descs1.dim() == 2 else descs1
        d2 = descs2.squeeze(0) if descs2.dim() == 2 else descs2
        with th.no_grad():
            return self.lgm_model(
                d1,
                d2,
                lafs1,
                lafs2,
                hw1=hw1,
                hw2=hw2
            )

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

            for matcher_name in self.get_available_matchers():
                if matcher_name == 'smnn':
                    for thr in self.config.smnn_thresholds:
                        matches = self._process_windows_from_disk(
                            nisar_src, s1_src, pair, strategy, matcher_name, thr
                        )
                        all_matches.extend(matches)
                else:
                    matches = self._process_windows_from_disk(
                        nisar_src, s1_src, pair, strategy, matcher_name, None
                    )
                    all_matches.extend(matches)

            elapsed = time.time() - start
            print(f"{self.get_filename_prefix()}: {len(all_matches)} match-sets in {elapsed:.2f}s")
            return all_matches

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

    def _build_file_id(self, metadata, global_row, global_col, matcher_name, match_param_str):
        base = (
            f"{self.get_filename_prefix()}_{metadata['nisar_pol']}_toS1{metadata['s1_ref_tag']}"
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
        scale_factor = metadata.get('scale_factor', 1.0)
        nisar_nodata = nisar_src.nodata if nisar_src.nodata is not None else metadata.get('nisar_nodata', 0)
        s1_nodata = s1_src.nodata if s1_src.nodata is not None else metadata.get('s1_nodata', 0)

        for idx, (wx, wy, sx, sy) in enumerate(strategy['windows']):
            try:
                nisar_win = rt.windows.Window(wy, wx, sy, sx)
                nisar_data = nisar_src.read(1, window=nisar_win)

                s1wx = int(wx * scale_factor)
                s1wy = int(wy * scale_factor)
                s1sx = int(sx * scale_factor)
                s1sy = int(sy * scale_factor)

                s1sx = min(s1sx, s1_src.height - s1wx)
                s1sy = min(s1sy, s1_src.width - s1wy)

                if s1sx <= 0 or s1sy <= 0:
                    continue

                s1_win = rt.windows.Window(s1wy, s1wx, s1sy, s1sx)
                s1_data = s1_src.read(1, window=s1_win)

                s1_data = self._calibrate_s1_dn(s1_data, nodata=s1_nodata)

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

            except Exception as e:
                if self.config.debug_mode:
                    print(f'Window error at {wx},{wy}: {e}')
                if 'device-side assert' in str(e):
                    safe_cuda_empty_cache()
                    try:
                        th.cuda.synchronize()
                    except Exception:
                        pass
            finally:
                if (idx + 1) % 20 == 0:
                    safe_cuda_empty_cache()

        return matches

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
                    _, idxs = self.match_adalam(descs1, descs2, lafs1, lafs2, hw1, hw2)
                    match_param_str = ''
                elif matcher_name == 'smnn':
                    _, idxs = self.match_smnn(descs1, descs2, matcher_param)
                    match_param_str = f'{matcher_param}'
                elif matcher_name == 'lgm':
                    _, idxs = self.match_lgm(descs1, descs2, lafs1, lafs2, hw1, hw2,
                                             feature_name=self._lightglue_feature_name())
                    match_param_str = ''
                else:
                    raise ValueError(f'Unknown matcher: {matcher_name}')

            if idxs.shape[0] == 0:
                return None

            return self._extract_matches(lafs1, lafs2, idxs,
                                         nisar_x, nisar_y, s1_x, s1_y,
                                         metadata, matcher_name, match_param_str)

        except Exception as e:
            if self.config.debug_mode:
                print(f'[{self.get_detector_name()}] Exception ({nisar_x},{nisar_y}): '
                      f'{type(e).__name__}: {e}')
            return None

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

        global_row = metadata['nisar_crop_col_offset'] + nisar_wx
        global_col = metadata['nisar_crop_row_offset'] + nisar_wy

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
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.sift = None

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        if self.sift is None:
            print('SIFT: Initializing...')
            self.sift = KF.SIFTFeature(
                self.config.num_features,
                upright=True,
                rootsift=True,
                device=self.device
            )
        lafs1, _, descs1 = self.sift(img1)
        lafs2, _, descs2 = self.sift(img2)
        return lafs1, descs1, lafs2, descs2

    def get_detector_name(self) -> str:
        return 'sift'

    def get_available_matchers(self) -> List[str]:
        return ['smnn']

    def _detector_needs_inpaint(self) -> bool:
        return False


class DISKMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig, checkpoint: str = 'depth'):
        super().__init__(config)
        self.checkpoint = checkpoint
        self.disk = None

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        if self.disk is None:
            print(f'DISK: Loading {self.checkpoint} model...')
            self.disk = KF.DISK.from_pretrained(device=self.device, checkpoint=self.checkpoint).eval()

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

    def get_filename_prefix(self) -> str:
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

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        if self.dedode is None:
            print(f'DeDoDe: Loading {self.detector_weights}/{self.descriptor_weights}...')
            self.dedode = KF.DeDoDe.from_pretrained(
                detector_weights=self.detector_weights,
                descriptor_weights=self.descriptor_weights
            ).eval().to(self.device)

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
        return ['smnn']

    def get_filename_prefix(self) -> str:
        return f'dedode_{self.detector_weights}_{self.descriptor_weights}'

    def _detector_needs_inpaint(self) -> bool:
        return True  # VGG-19 + DINOv2 both pooling-sensitive to boundaries


class LoFTRMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.loftr = None

        # Use your custom config from the test script
        self.konfig = {
            "backbone_type": "ResNetFPN",
            "resolution": (8, 2),
            "fine_window_size": 5,
            "fine_concat_coarse_feat": True,
            "resnetfpn": {"initial_dim": 128, "block_dims": [128, 196, 256]},
            "coarse": {
                "d_model": 256, "d_ffn": 256, "nhead": 8,
                "layer_names": ["self", "cross"] * 4,
                "attention": "linear", "temp_bug_fix": False,
            },
            "match_coarse": {
                "thr": 1, "border_rm": 2, "match_type": "dual_softmax",
                "dsmax_temperature": 0.12, "skh_iters": 10,
                "skh_init_bin_score": 0.1, "skh_prefilter": True,
                "train_coarse_percent": 0.4, "train_pad_num_gt_min": 200,
                "sparse_spvs": False
            },
            "fine": {"d_model": 128, "d_ffn": 128, "nhead": 8, "layer_names": ["self", "cross"], "attention": "linear"},
        }

    def _determine_window_strategy(self, nisar_src, s1_src) -> Dict:
        original = self.config.window_size
        self.config.window_size = self.config.loftr_max_window
        strategy = super()._determine_window_strategy(nisar_src, s1_src)
        self.config.window_size = original
        return strategy

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
            img1, mask1 = self._norm_img(nisar_data, nodata=nisar_nodata, return_mask=True)
            img2, mask2 = self._norm_img(s1_data, nodata=s1_nodata, return_mask=True)

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

            if self.loftr is None:
                print('LoFTR: Initializing model...')
                self.loftr = KF.LoFTR(pretrained='outdoor', config=self.konfig).eval().to(self.device)
                from kornia.feature.loftr.utils.superglue import log_optimal_transport
                self.loftr.coarse_matching.match_type = 'sinkhorn'
                self.loftr.coarse_matching.bin_score = th.nn.Parameter(
                    th.tensor(self.konfig['match_coarse']['skh_init_bin_score'], requires_grad=True)
                ).to(self.device)
                self.loftr.coarse_matching.log_optimal_transport = log_optimal_transport
                self.loftr.coarse_matching.skh_iters = 2
                self.loftr.coarse_matching.skh_prefilter = True

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
                self._xy_to_map(x1[i], y1[i], metadata['x01'], metadata['y01'],
                                metadata['xres1'], metadata['yres1'], nisar_y, nisar_x)
                for i in range(len(x1))
            ])
            X2, Y2 = zip(*[
                self._xy_to_map(x2[i], y2[i], metadata['x02'], metadata['y02'],
                                metadata['xres2'], metadata['yres2'], s1_y, s1_x)
                for i in range(len(x2))
            ])

            dist = [np.sqrt((X1[i] - X2[i]) ** 2 + (Y1[i] - Y2[i]) ** 2) for i in range(len(x1))]
            along = [Y1[i] - Y2[i] for i in range(len(x1))]  # northing diff
            across = [X1[i] - X2[i] for i in range(len(x1))]  # easting diff

            # Match _extract_matches convention: col_offset → global_row, row_offset → global_col
            global_row = metadata['nisar_crop_col_offset'] + nisar_x
            global_col = metadata['nisar_crop_row_offset'] + nisar_y
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

        except Exception as e:
            # Always print LoFTR errors — not gated by debug_mode
            print(f'[LoFTR] Exception at window ({nisar_x},{nisar_y}): {type(e).__name__}: {e}')
            return None


class ALIKEDMatcher(DiskBasedMatcher):
    def __init__(self, config: PipelineConfig,
                 model_name: str = 'aliked-n16'):
        super().__init__(config)
        self.model_name = model_name
        self.aliked = None

    def detect_and_describe(self, img1: th.Tensor, img2: th.Tensor) -> Tuple:
        if self.aliked is None:
            print(f'ALIKED: Loading {self.model_name}...')
            self.aliked = KF.ALIKED(
                model_name=self.model_name,
                device=self.device,
                top_k=-1,
                scores_th=0.2,
                n_limit=self.config.num_features
            ).eval().to(self.device)

        # ALIKED expects 3-channel input
        img1_rgb = K.color.grayscale_to_rgb(img1)
        img2_rgb = K.color.grayscale_to_rgb(img2)

        out1 = self.aliked(img1_rgb)
        out2 = self.aliked(img2_rgb)

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
        return ['smnn', 'lgm']   # LightGlue natively supports aliked

    def get_filename_prefix(self) -> str:
        return f'aliked_{self.model_name}'

    def _lightglue_feature_name(self) -> str:
        return 'aliked'          # Kornia LightGlueMatcher knows this name

    def _detector_needs_inpaint(self) -> bool:
        return True              # Deformable conv is boundary-sensitive like DISK


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
                            method=cv2.USAC_MAGSAC,
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
class MatchStatistics:
    @staticmethod
    def _parse_filename(filename: str) -> Optional[Dict]:
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) < 10:
            return None

        def _to_int(tok, prefix):
            return int(tok.replace(prefix, ''))

        default_extra = {
            'disk_mode': None,
            'det_wt': None,
            'desc_wt': None
        }

        try:
            if parts[0] == 'sift':
                detector = 'sift'
                nisar_pol = parts[1]
                s1_ref_tag = parts[2].replace('toS1', '')
                i = 3
                pair_id = _to_int(parts[i], 'pair'); i += 1
                scan = _to_int(parts[i], 'scan'); i += 1
                pix = _to_int(parts[i], 'pix'); i += 1
                match_method = parts[i]; i += 1
                match_parameter = float(parts[i]) if match_method == 'smnn' else None
                if match_method == 'smnn':
                    i += 1
                if parts[i] != 'aff':
                    return None
                ransac_method = parts[i + 1]
                ransac_threshold = float(parts[i + 2])
                ransac_confidence = float(parts[i + 3])
                return {**default_extra, **{
                    'detector': detector,
                    'nisar_pol': nisar_pol,
                    's1_ref_tag': s1_ref_tag,
                    'pair_id': pair_id,
                    'scan': scan,
                    'pix': pix,
                    'match_method': match_method,
                    'match_parameter': match_parameter,
                    'ransac_method': ransac_method,
                    'ransac_threshold': ransac_threshold,
                    'ransac_confidence': ransac_confidence,
                }}

            if parts[0] == 'disk':
                detector = 'disk'
                disk_mode = parts[1]
                nisar_pol = parts[2]
                s1_ref_tag = parts[3].replace('toS1', '')
                i = 4
                pair_id = _to_int(parts[i], 'pair'); i += 1
                scan = _to_int(parts[i], 'scan'); i += 1
                pix = _to_int(parts[i], 'pix'); i += 1
                match_method = parts[i]; i += 1
                if match_method == 'smnn':
                    match_parameter = float(parts[i])
                    i += 1
                else:
                    match_parameter = None
                if parts[i] != 'aff':
                    return None
                ransac_method = parts[i + 1]
                ransac_threshold = float(parts[i + 2])
                ransac_confidence = float(parts[i + 3])
                return {**default_extra, **{
                    'detector': detector,
                    'disk_mode': disk_mode,
                    'nisar_pol': nisar_pol,
                    's1_ref_tag': s1_ref_tag,
                    'pair_id': pair_id,
                    'scan': scan,
                    'pix': pix,
                    'match_method': match_method,
                    'match_parameter': match_parameter,
                    'ransac_method': ransac_method,
                    'ransac_threshold': ransac_threshold,
                    'ransac_confidence': ransac_confidence,
                }}

            if parts[0] == 'dedode':
                detector = 'dedode'
                det_wt = parts[1]
                desc_wt = parts[2]
                nisar_pol = parts[3]
                s1_ref_tag = parts[4].replace('toS1', '')
                i = 5
                pair_id = _to_int(parts[i], 'pair'); i += 1
                scan = _to_int(parts[i], 'scan'); i += 1
                pix = _to_int(parts[i], 'pix'); i += 1
                match_method = parts[i]; i += 1
                match_parameter = float(parts[i]) if match_method == 'smnn' else None
                if match_method == 'smnn':
                    i += 1
                if parts[i] != 'aff':
                    return None
                ransac_method = parts[i + 1]
                ransac_threshold = float(parts[i + 2])
                ransac_confidence = float(parts[i + 3])
                return {**default_extra, **{
                    'detector': detector,
                    'det_wt': det_wt,
                    'desc_wt': desc_wt,
                    'nisar_pol': nisar_pol,
                    's1_ref_tag': s1_ref_tag,
                    'pair_id': pair_id,
                    'scan': scan,
                    'pix': pix,
                    'match_method': match_method,
                    'match_parameter': match_parameter,
                    'ransac_method': ransac_method,
                    'ransac_threshold': ransac_threshold,
                    'ransac_confidence': ransac_confidence,
                }}

            if parts[0] == 'loftr':
                nisar_pol = parts[1]
                s1_ref_tag = parts[2].replace('toS1', '')
                pair_id = _to_int(parts[3], 'pair')
                scan = _to_int(parts[4], 'scan')
                pix = _to_int(parts[5], 'pix')
                # parts[6]='loftr', parts[7]='internal', parts[8]='aff'
                if len(parts) < 12 or parts[8] != 'aff':
                    return None
                ransac_method = parts[9]
                ransac_threshold = float(parts[10])
                ransac_confidence = float(parts[11])
                return {**default_extra, **{
                    'detector': 'loftr',
                    'nisar_pol': nisar_pol,
                    's1_ref_tag': s1_ref_tag,
                    'pair_id': pair_id, 'scan': scan, 'pix': pix,
                    'match_method': 'loftr_internal', 'match_parameter': None,
                    'ransac_method': ransac_method,
                    'ransac_threshold': ransac_threshold,
                    'ransac_confidence': ransac_confidence,
                }}

        except Exception:
            return None

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
    ):
        os.makedirs(output_dir, exist_ok=True)
        df = ChipConsensusSelector.build_chip_stats(csv_dir, min_inliers_per_chip)
        if df.empty:
            print('No valid chip CSV files found.')
            return None

        df.to_csv(os.path.join(output_dir, 'CHIP_STATS_ALL.csv'), index=False)

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
            n_chips = int(len(kept))
            n_pairs = int(kept['pair_id'].nunique())

            along_mean = float(kept['along_mean'].mean())
            across_mean = float(kept['across_mean'].mean())
            along_std_chips = float(kept['along_mean'].std()) if len(kept) > 1 else 0.0
            across_std_chips = float(kept['across_mean'].std()) if len(kept) > 1 else 0.0

            mean_penalty = abs(along_mean) + abs(across_mean)
            spread_penalty = along_std_chips + across_std_chips + 1e-6
            score = agg_inliers / (mean_penalty + spread_penalty + 1e-6)

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
                'along_std_chips': along_std_chips,
                'across_std_chips': across_std_chips,
                'mean_penalty': mean_penalty,
                'spread_penalty': spread_penalty,
                'score': score,
            })
            survivors.append(kept)

        if not config_scores:
            print('No configuration survived the chip consensus filter.')
            return None

        chip_keep_df = pd.concat(survivors, ignore_index=True)
        scores_df = pd.DataFrame(config_scores).sort_values('score', ascending=False)

        chip_keep_df.to_csv(os.path.join(output_dir, 'CHIP_STATS_SURVIVORS.csv'), index=False)
        scores_df.to_csv(os.path.join(output_dir, 'CONSENSUS_SCORES.csv'), index=False)

        best = scores_df.iloc[0]
        best_key = best['config_key']
        best_chip_df = chip_keep_df[chip_keep_df['config_key'] == best_key].copy()
        best_chip_df.to_csv(os.path.join(output_dir, 'BEST_CHIP_SET.csv'), index=False)

        manifest_fields = [
            'filename', 'pair_id', 'scan', 'pix', 'num_inliers',
            'along_mean', 'across_mean', 'along_std', 'across_std',
            'detector', 'disk_mode', 'det_wt', 'desc_wt',
            'nisar_pol', 's1_ref_tag', 'match_method', 'match_parameter',
            'ransac_method', 'ransac_threshold', 'ransac_confidence'
        ]
        manifest_fields = [f for f in manifest_fields if f in best_chip_df.columns]

        best_chip_df[manifest_fields].to_csv(
            os.path.join(output_dir, 'BEST_CHIP_MANIFEST.csv'),
            index=False
        )

        best_summary = pd.DataFrame([best.to_dict()])
        best_summary.to_csv(os.path.join(output_dir, 'BEST_FINAL_SUMMARY.csv'), index=False)

        print('Saved:')
        print(' CHIP_STATS_ALL.csv')
        print(' CHIP_STATS_SURVIVORS.csv')
        print(' CONSENSUS_SCORES.csv')
        print(' BEST_CHIP_SET.csv')
        print(' BEST_CHIP_MANIFEST.csv')
        print(' BEST_FINAL_SUMMARY.csv')

        return {
            'best_chip_manifest_csv': os.path.join(output_dir, 'BEST_CHIP_MANIFEST.csv'),
            'best_final_summary_csv': os.path.join(output_dir, 'BEST_FINAL_SUMMARY.csv'),
            'best_config': best.to_dict(),
        }


# =============================================================================
# PIPELINE
# =============================================================================
class ProductionPipelinePolwise:
    def __init__(self, config: PipelineConfig, mode: str = 'same-res'):
        self.config = config
        self.mode = mode
        self.ransac_filter = RANSACFilter(config)

    def _reference_dir_for_pol(self, pol: str) -> Tuple[str, str]:
        s1_ref_tag = NISARH5Reader.choose_s1_ref_tag(pol)
        if s1_ref_tag == 'VV':
            return self.config.reference_dir_vv, s1_ref_tag
        return self.config.reference_dir_vh, s1_ref_tag

    def _build_matchers(self, detector_types: List[str]) -> List[DiskBasedMatcher]:
        detectors = []
        for d in detector_types:
            dl = d.lower()
            if dl == 'sift':
                detectors.append(SIFTMatcher(self.config))
            elif dl == 'disk':
                detectors.append(DISKMatcher(self.config, checkpoint='depth'))
                detectors.append(DISKMatcher(self.config, checkpoint='epipolar'))
            elif dl == 'dedode':
                detectors.append(DeDoDeMatcher(self.config, detector_weights='L-C4', descriptor_weights='G-C4'))
            elif dl == "loftr":
                detectors.append(LoFTRMatcher(self.config))
            else:
                raise ValueError(f'Unknown detector type: {d}')
        return detectors

    def _run_single_pol(self, scene_info: Dict, pol: str, detector_types: List[str]) -> List[Dict]:
        reference_dir, s1_ref_tag = self._reference_dir_for_pol(pol)
        if not reference_dir:
            raise ValueError(f'Reference directory missing for S1 {s1_ref_tag}')

        pol_temp_dir = os.path.join(self.config.temp_dir, f'{pol}_toS1{s1_ref_tag}')
        pol_out_dir = os.path.join(self.config.output_base_dir, f'{pol}_toS1{s1_ref_tag}')
        os.makedirs(pol_temp_dir, exist_ok=True)
        os.makedirs(pol_out_dir, exist_ok=True)

        reffetcher = ReferenceFetcher(
            reference_dir,
            decimation=self.config.s1_decimation,
            min_area=self.config.min_area
        )
        preprocessor = DiskBasedPreprocessor(
            pol_temp_dir,
            self.config.target_resolution if self.mode == 'same-res' else None
        )

        if self.config.use_disk_cache and self.config.check_existing_pairs:
            disk_pairs = preprocessor.load_existing_pairs()
        else:
            disk_pairs = None

        if disk_pairs is None:
            refs = reffetcher.fetch_overlapping_references(scene_info['met_path'])
            if not refs:
                raise RuntimeError(f'No overlapping S1 references found for pol {pol}')
            disk_pairs = preprocessor.create_all_pairs(
                scene_info['h5_path'],
                scene_info['met_path'],
                refs,
                pol,
                s1_ref_tag
            )

        run_records = []
        matchers_to_run = self._build_matchers(detector_types)

        for matcher in matchers_to_run:
            tag = matcher.get_filename_prefix()
            suffix = 'same-res' if self.mode == 'same-res' else 'multi-res'

            print('-' * 80)
            print(f'[{pol}] Running {tag}')

            all_match_data = []
            for pair in disk_pairs:
                all_match_data.extend(matcher.process_disk_cached_pair(pair))
            safe_cuda_empty_cache()

            raw_dir = os.path.join(pol_out_dir, f'raw_matches_{suffix}_{tag}')
            matcher.save_matches_to_csv(all_match_data, raw_dir)

            filter_dir = os.path.join(pol_out_dir, f'filtered_{suffix}_{tag}')
            self.ransac_filter.filter_matches_in_memory(all_match_data, filter_dir)

            stats_dir = os.path.join(pol_out_dir, f'statistics_{suffix}_{tag}')
            MatchStatistics.compute_statistics(filter_dir, stats_dir)

            final_dir = os.path.join(pol_out_dir, f'final_{suffix}_{tag}')
            summary = ChipConsensusSelector.select_configs(
                csv_dir=filter_dir,
                output_dir=final_dir,
                tolerance_m=self.config.consensus_tolerance_m,
                mode_bin_m=self.config.consensus_mode_bin_m,
                min_inliers_per_chip=self.config.min_inliers_per_chip,
                min_surviving_chips=self.config.min_surviving_chips,
            )

            run_records.append({
                'nisar_pol': pol,
                's1_ref_tag': s1_ref_tag,
                'reference_dir': reference_dir,
                'detector_tag': tag,
                'output_dir': pol_out_dir,
                'raw_dir': raw_dir,
                'filtered_dir': filter_dir,
                'statistics_dir': stats_dir,
                'final_dir': final_dir,
                'best_chip_manifest_csv': None if summary is None else summary.get('best_chip_manifest_csv'),
                'best_final_summary_csv': None if summary is None else summary.get('best_final_summary_csv'),
            })

            del all_match_data
            safe_cuda_empty_cache()
            gc.collect()

        if self.config.cleanup_after_pair:
            shutil.rmtree(pol_temp_dir, ignore_errors=True)

        return run_records

    def run(self, scene_dir: str, detector_types: List[str] = None):
        if detector_types is None:
            detector_types = ['disk', 'dedode', 'sift']

        scene_info = NISARH5Reader.discover_scene(scene_dir)
        available_pols = NISARH5Reader.available_pols(scene_info)

        if not available_pols:
            raise RuntimeError('No supported polarizations found from scene metadata/H5.')

        print('=' * 80)
        print('POL-WISE NISAR-S1 PIPELINE')
        print('=' * 80)
        print(f"Scene dir : {scene_info['scene_dir']}")
        print(f"Scene h5  : {scene_info['h5_path']}")
        print(f"Scene met : {scene_info['met_path']}")
        print(f'Pols      : {available_pols}')
        print(f'Detectors : {detector_types}')

        all_runs = []
        for pol in available_pols:
            print('-' * 80)
            print(f'Running polarization: {pol}')
            all_runs.extend(self._run_single_pol(scene_info, pol, detector_types))

        pd.DataFrame(all_runs).to_csv(
            os.path.join(self.config.output_base_dir, 'POL_RUN_SUMMARY.csv'),
            index=False
        )

        print('=' * 80)
        print('ALL POLARIZATIONS COMPLETE')
        print('=' * 80)
        print(f"Summary CSV: {os.path.join(self.config.output_base_dir, 'POL_RUN_SUMMARY.csv')}")
        return all_runs


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    setup_logging(sys.argv[2])

    input_ = sys.argv[3].split(',')
    config = PipelineConfig(
        target_resolution=10,
        use_disk_cache=True,
        cleanup_after_pair=False,
        check_existing_pairs=True,
        use_amp=True,
        debug_mode=False,

        smnn_thresholds=[0.90, 0.925, 0.95, 0.975, 0.99],
        ransac_methods=[4],
        ransac_thresholds=[1, 2, 3],
        ransac_confidences=[0.91, 0.95, 0.99],

        reference_dir_vv=input_[2],
        reference_dir_vh=input_[1],

        output_base_dir=sys.argv[2],
        temp_dir=os.environ.get('NISAR_TEMP_DIR', '/maintenance/ICIGDev/GPUPOC/inter/dqe/set2/'),

        s1_decimation=10,
        min_area=200.0,

        consensus_tolerance_m=5.0,
        consensus_mode_bin_m=0.5,
        min_inliers_per_chip=3,
        min_surviving_chips=2,
    )

    scene_dir = input_[0]

    pipeline = ProductionPipelinePolwise(config, mode='same-res')
    pipeline.run(scene_dir, detector_types=['disk', 'dedode', 'sift', 'loftr'])
