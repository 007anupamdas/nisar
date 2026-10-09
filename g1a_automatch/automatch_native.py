"""
automatch_native -- imported first by the modules that load compiled
packages: it loads rasterio's native libraries before any other.

In a conda environment that mixes conda-forge GDAL (rasterio) with pip wheels
(torch, pandas, opencv), the first extension that needs the C++ runtime
decides which libstdc++ the whole process uses. A pip wheel takes the system
copy (/lib64/libstdc++.so.6), which can be too old for conda-forge GDAL:
"version `GLIBCXX_3.4.30' not found (required by ... libgdal.so)". rasterio
takes the environment's newer copy, which serves every later import too.

It also shows GDAL's "geographic CRS EPSG:4326 got from GeoTIFF keys is not
the same as the one from the EPSG registry" notice once instead of at every
reference read (4,486 of the 20,247 lines of the log of run 299).
"""
import logging

try:
    import rasterio  # noqa: F401
except Exception:  # missing or broken: the import that needs it reports why
    pass


class _OnceGeoKeysNotice(logging.Filter):
    seen = False

    def filter(self, record):
        if 'is not the same as the one from the EPSG registry' not in record.getMessage():
            return True
        if _OnceGeoKeysNotice.seen:
            return False
        _OnceGeoKeysNotice.seen = True
        record.msg = f'{record.msg} (shown once; repeats hidden)'
        return True


if not any(isinstance(f, _OnceGeoKeysNotice) for f in logging.getLogger('rasterio._env').filters):
    logging.getLogger('rasterio._env').addFilter(_OnceGeoKeysNotice())
