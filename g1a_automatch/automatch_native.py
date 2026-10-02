"""
automatch_native -- imported first by the modules that load compiled
packages: it loads rasterio's native libraries before any other.

In a conda environment that mixes conda-forge GDAL (rasterio) with pip wheels
(torch, pandas, opencv), the first extension that needs the C++ runtime
decides which libstdc++ the whole process uses. A pip wheel takes the system
copy (/lib64/libstdc++.so.6), which can be too old for conda-forge GDAL:
"version `GLIBCXX_3.4.30' not found (required by ... libgdal.so)". rasterio
takes the environment's newer copy, which serves every later import too.
"""
try:
    import rasterio  # noqa: F401
except Exception:  # missing or broken: the import that needs it reports why
    pass
