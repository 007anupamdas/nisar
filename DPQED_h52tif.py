import os.path
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

group_path = "science/SSAR/GSLC/grids/frequencyA"

with hp.File(h5_path, 'r') as f:
    grp = f[group_path]
    grp.keys()
    print(grp.keys())
    hh_cmp = grp['HH'][()]
    hv_cmp = grp['HV'][()]
    hh_amp = np.abs(hh_cmp).astype('float32')
    hv_amp = np.abs(hv_cmp).astype('float32')

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
        height=hh_amp.shape[0],
        width=hh_amp.shape[1],
        count=2,
        dtype=hh_amp.dtype,
        crs=crs1,
        transform=transform,
        compress="DEFLATE",
        tiled=True,
        windowed=True,
        BIGTIFF="YES",
        blockxsize=256,
        blockysize=256
    ) as dst:
        dst.write(hh_amp, 1)
        dst.write(hv_amp, 2)

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