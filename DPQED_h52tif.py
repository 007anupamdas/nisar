import os.path
import re
import h5py as hp
import numpy as np
import matplotlib
import rioxarray as rxr
import xarray as xr
import rasterio as rt
matplotlib.use("TkAgg")
from affine import Affine
from matplotlib import pyplot as plt
from rasterio.crs import CRS

root = r'V:\ICIGDev\GPUPOC\input\dqe\inp\4may26\NISAR_S2_PR_GSLC_019_105_A_018_3700_DHNA_A_20260504T111941_20260504T112018_D00409_M_F_I_001'
prod = root.split('\\')[-1]


h5_path = os.path.join(root, prod + '.h5')
tif_path = os.path.join(root, prod + '1.tif')

# hf = rt.open(r'D:\eos6\New folder\O2_10SEP2021_004_013_GAN_L1B_ST_S.hdf', 'r')
# o3 = rxr.open_rasterio(r'X:\APA\SAN\NIS\SS\2026\JAN\NISAR_S2_PR_GSLC_010_170_A_013_3700_DHNA_A_20260120T232921_20260120T232958_D00407_M_F_I_001\NISAR_S2_PR_GSLC_010_170_A_013_3700_DHNA_A_20260120T232921_20260120T232958_D00407_M_F_I_001.h5')
# ds = rxr.open_rasterio(os.path.join(root, prod) + '.h5')

# The group path is found in the file rather than written into the script. It
# encodes four independent things -- band (LSAR/SSAR), product (GSLC/GCOV/
# RSLC), grids vs swaths, and frequency (A/B) -- so a literal only ever works
# for the one granule it was written against and fails on the next with a bare
# KeyError. Set GROUP_PATH_OVERRIDE to force a particular group.
#
# cog_locate's nisar_h5.describe() walks the same layout for the streaming
# reader; this is kept self-contained so the script can be copied on its own.
GROUP_PATH_OVERRIDE = None
FREQUENCY_PREFERENCE = ("A", "B")   # A is the wider band, so the finer posting

POL_NAMES = ("HH", "HV", "VH", "VV", "RH", "RV",
             "HHHH", "HVHV", "VHVH", "VVVV", "HHHV", "HHVV", "HVVV")

_GRID_RE = re.compile(
    r"^/science/(?P<band>[LS]SAR)/(?P<product>[A-Z]+)/"
    r"(?P<space>grids|swaths)/frequency(?P<freq>[AB])$")


def find_grid_groups(f):
    """Every frequency group in the file, and what each one carries."""
    found = {}

    def visit(name, obj):
        if not isinstance(obj, hp.Group):
            return
        path = "/" + name
        m = _GRID_RE.match(path)
        if not m:
            return
        found[path] = {
            "band": m.group("band"), "product": m.group("product"),
            "space": m.group("space"), "freq": m.group("freq"),
            "pols": [p for p in POL_NAMES
                     if isinstance(obj.get(p), hp.Dataset) and obj[p].ndim == 2],
            # A north-up GeoTIFF needs a map grid. RSLC swaths are in slant
            # range and carry no projection, so there is no transform to write.
            "geocoded": "xCoordinates" in obj and "yCoordinates" in obj,
        }

    f.visititems(visit)
    return found


def pick_grid_group(f):
    """(path, info) for the group to convert. Prints everything it found."""
    groups = find_grid_groups(f)
    if not groups:
        raise SystemExit(
            "no /science/<band>/<product>/(grids|swaths)/frequency* group in "
            + h5_path)

    for path in sorted(groups):
        info = groups[path]
        print(f"[H5] {path}  pols={','.join(info['pols']) or '(none)'}"
              f"  {'geocoded' if info['geocoded'] else 'no map grid'}")

    if GROUP_PATH_OVERRIDE:
        path = "/" + GROUP_PATH_OVERRIDE.strip("/")
        if path not in groups:
            raise SystemExit(f"GROUP_PATH_OVERRIDE {path} is not in this file")
        return path, groups[path]

    usable = {p: i for p, i in groups.items() if i["geocoded"] and i["pols"]}
    if not usable:
        raise SystemExit(
            "no geocoded frequency group holding imagery. This writes a "
            "north-up GeoTIFF, so it needs xCoordinates/yCoordinates -- which "
            "an RSLC, being in slant range, does not carry.")

    for freq in FREQUENCY_PREFERENCE:
        for path in sorted(usable):
            if usable[path]["freq"] == freq:
                return path, usable[path]
    path = sorted(usable)[0]
    return path, usable[path]


with hp.File(h5_path, 'r') as f:
    group_path, info = pick_grid_group(f)
    pols = info["pols"]
    print(f"[H5] using {group_path}  ({info['band']} {info['product']} "
          f"frequency{info['freq']}, {len(pols)} band(s): {', '.join(pols)})")
    grp = f[group_path]

    # One band per polarization the group actually holds, rather than assuming
    # HH and HV: a single-pol product carries only one of them, a quad-pol
    # four, and a GCOV names them HHHH/HVHV instead.
    amps = [np.abs(grp[pol][()]).astype('float32') for pol in pols]

    x_coord = grp['xCoordinates'][()]
    y_coord = grp['yCoordinates'][()]

    proj_wkt = grp['projection'][()]
    print(proj_wkt)
    # epsg1 = grp['projection'][()]
    # if isinstance(proj_wkt, bytes):
    #     proj_wkt = proj_wkt.decode('utf-8')
    # crs = CRS.from_wkt(proj_wkt)

    dx = x_coord[1] - x_coord[0]
    dy = y_coord[1] - y_coord[0]

    x_ori = x_coord[0] - (dx/2.0)
    y_ori = y_coord[0] - (dy / 2.0)

    transform = Affine.translation(x_ori, y_ori) * Affine.scale(dx, dy)
    epsg1 = grp['projection'][()]
    crs1 = CRS.from_epsg(epsg1)

    # print(f"CRS: {crs.to_string()}")
    print(f"x_origin from HDF5: {x_coord[0]:.1f}")

    with rt.open(
        tif_path,
        'w',
        driver="GTiff",
        height=amps[0].shape[0],
        width=amps[0].shape[1],
        count=len(amps),
        dtype=amps[0].dtype,
        crs=crs1,
        transform=transform,
        compress="DEFLATE",
        tiled=True,
        windowed=True,
        BIGTIFF="YES",
        blockxsize=256,
        blockysize=256
    ) as dst:
        for i, (pol, amp) in enumerate(zip(pols, amps), start=1):
            dst.write(amp, i)
            # With the band count no longer fixed at HH+HV, the names are the
            # only way to tell which band is which -- and QGIS lists them.
            dst.set_band_description(i, pol)

# hh = ds[8].science_LSAR_RSLC_swaths_frequencyA_HH.squeeze('band', drop=True)
# hv = ds[8].science_LSAR_RSLC_swaths_frequencyA_HV.squeeze('band', drop=True)
# vh = ds[8].science_LSAR_RSLC_swaths_frequencyA_VH.squeeze('band', drop=True)
# vv = ds[8].science_LSAR_RSLC_swaths_frequencyA_VV.squeeze('band', drop=True)

# hh = ds[0].science_SSAR_GSLC_grids_frequencyA_HH.squeeze('band', drop=True)
# hv = ds[0].science_SSAR_GSLC_grids_frequencyA_HV.squeeze('band', drop=True)
# vh = ds[0].science_SSAR_GSLC_grids_frequencyA_VH.squeeze('band', drop=True)
# vv = ds[0].science_SSAR_GSLC_grids_frequencyA_VV.squeeze('band', drop=True)

# hh_out = np.abs(hh).astype('float32')
# hv_out = np.abs(hv).astype('float32')
# vh_out = np.abs(vh).astype('float32')
# vv_out = np.abs(vv).astype('float32')

# ds_out = xr.Dataset(
#     {
#         "HH": hh_out,
#         # "HV": hv_out,
#         # "VH": vh_out,
#         # "VV": vv_out,
#     }
# )

# ds_out = ds_out.assign_coords(x=ds[0]["x"], y=ds[0]["y"])
# ds_out.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
#
# if ds[0].rio.crs is not None:
#     ds_out.rio.write_crs(ds.rio.crs, inplace=True)
#
# ds_out.rio.to_raster(
#     os.path.join(root, prod)+ '.tif',
#     driver="GTiff",
#     dtype="float32",
#     compress="DEFLATE",
#     tiled=True,
#     windowed=True,
#     BIGTIFF="YES"
# )

# data_bands = []
# data_bands.append(bands['10'].values[0])
# for i in range(2,9):
#     band_name = f'Band{i}'
#     data = bands[f'{11+i}'].values[0]
#     data_bands.append(data)
#
# ds = xr.Dataset()
#
# for i in range(1,9):
#     ds[f'band{i}'] = xr.DataArray(
#         data_bands[i-1],
#         dims=("y", "x"),
#         coords={
#             "longitude": (('y', 'x'), lon),
#             "latitude": (('y', 'x'), lat)
#         },
#     attrs= bands.attrs)
#
# # ds.rio.write_crs("EPSG:4326", inplace=True)
# # ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
# # ds.rio.to_raster(r'D:\eos6\New folder\test.tif')
#
# multi_band = xr.concat([ds[f'band{i}'] for i in range(1,9)], dim='band')
# multi_band.coords['band'] = np.arange(1,9)
# multi_band.coords['latitude'] = ds.coords['latitude']
# multi_band.coords['longitude'] = ds.coords['longitude']
#
# multi_band.rio.write_crs("EPSG:4326", inplace=True)
# multi_band.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
# multi_band.rio.write_transform(Affine(dx, 0 , ulx, 0, dy, uly),inplace=True)
# multi_band.rio.to_raster(r'D:\eos6\New folder\test.tif')