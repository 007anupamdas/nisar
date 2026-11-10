from osgeo import gdal

loc_1 = r'D:\nisar\lsar\s\NISAR_S2_PR_GSLC_028_084_A_010_3700_DHNA_A_20260819T001733_20260819T001810_P00500_M_F_I_001.tif'
# loc_2 = r'D:\nisar\test\NISAR_S2_PR_GSLC_009_084_A_010_3700_DHNA_A_20260103T001736_20260103T001812_D00407_M_F_I_003\ref\RI1MRS_17N_076E_HH.tif'
out_ = r'D:\nisar\lsar\s\NISAR_S2_PR_GSLC_017_040_A_018_3700_DHNA_A_20260405T230713_20260405T230724_P00500_P_P_I_001_cog.tif'

gdal.Translate(
    out_,
    loc_1,
    format="COG",
    creationOptions=[
        "COMPRESS=DEFLATE",
        "TILED=YES",
        "BIGTIFF=YES",
        "NUM_THREADS=ALL_CPUS",
        "BLOCKXSIZE=512",
        "BLOCKXSIZE=512",
        "PREDICTOR=2",
        "OVERVIEWS=IGNORE_EXISTING"
    ]
)