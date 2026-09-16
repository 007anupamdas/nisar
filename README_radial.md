# RADIAL — radiometric assessment of NISAR GCOV products

A QGIS window with one canvas. Draw rectangles or polygons over targets whose
backscatter you know something about, read gamma0 and its statistics for each
one, and export the whole set as a shapefile carrying every number.

Where [`DPQED_rival.py`](DPQED_rival.py) measures **where** a product puts the
ground, this measures **what** it reports there.

```
exec(open(r"path/to/DPQED_radial.py").read())      # QGIS Python console
```

Pan, zoom, Normalize and the R/G/B band picker behave exactly as they do in
RIVAL — same stretch, same clip, same pinning. The differences are one canvas
instead of two, ROIs instead of point picks, and statistics instead of offsets.

## Workflow

| Step | |
|---|---|
| **Load GCOV** | a GeoTIFF or VRT, or a NISAR `.h5` (wrapped in a VRT, below) |
| **Normalize** | stretch the view so the target is legible before you draw on it |
| **Rect** / **Polygon** | draw ROIs; each one fills a row as it closes |
| **Export SHP** | polygons plus every statistic, and a full-named CSV beside it |

`Ctrl+1..5` selects the tool, `F5` zooms to the selected ROI, `Ctrl+Delete`
removes it, `Escape` abandons a polygon in progress. A polygon closes on a
right-click or a double-click; `Backspace` takes back a corner.

Double-click a **Name** cell to label an ROI. The label is exported.

### Comparing two products over the same ground

Loading a raster does **not** clear the ROIs — they are re-measured against it.

```
draw the ROIs once  ->  Export SHP        (product A)
Load GCOV, product B                      (same ROIs, re-measured)
                    ->  Export SHP        (product B)
```

The two exports differ only by what the products say. `Load ROIs` reads a
shapefile back in, so a set drawn today can be run over next month's product,
or handed to someone else.

## The statistics

Everything is computed on **linear power**, per ROI and per band.

| Column | |
|---|---|
| `n` | valid pixels: finite, not nodata, not zero unless *Zeros are data* |
| `mean`, `std` | linear power |
| `cv` | `std / mean` |
| `enl` | equivalent number of looks, `(mean/std)²` |
| `mean_db` | `10·log10(mean)` — **the calibrated figure to quote** |
| `sdev_db` | spread of the dB pixels, which is what the stretch shows |
| `min_db` `p5_db` `med_db` `p95_db` `max_db` | percentiles of the power, in dB |
| `nonpos` | pixels at or below zero, excluded from the dB columns |

The footer summarises the selected band across ROIs: the mean of the ROI means,
the spread between brightest and darkest, and the median ENL.

### How to read ENL

`enl` is **the ROI's own homogeneity**, not the product's looks. Over a uniform
distributed target it is the number of looks; over anything else the scene's own
variation is counted as speckle and it reads low. Draw it on the rainforest
patch, not on the one straddling a field boundary. A single-look GCOV term over
uniform ground reads ~1; a 4-look product reads ~4.

Compare ENL **between products over the same ROI**, not between ROIs.

### Three traps worth knowing

**Domain.** The selector states what the pixels hold — it is not a display
option. Everything is converted from it to power once, up front, because the
mean of dB pixels is not the dB of the mean: over single-look speckle the log
average sits about **2.5 dB low**, which is the size of the differences this
tool exists to find. NISAR GCOV is power. A GSLC magnitude is amplitude.

**Zeros.** A SAR product means "no data" by zero when it declares no nodata
value. Counted as data they drag every mean down and put ENL on the floor, so
they are excluded by default. The rule is suspended for a raster in dB, where
0 dB is a power of 1 — an ordinary bright pixel.

**Negative pixels.** Noise subtraction can leave a GCOV pixel slightly
negative. Those stay in the linear mean, where they belong — dropping them
adds back exactly what the noise subtraction removed — and are left out of the
dB columns, where no logarithm exists. `nonpos` counts them. A large count
means the dB columns describe only part of the ROI.

## What gets exported

`Export SHP` writes one polygon per ROI in the working CRS:

```
roi  name  kind  npix  area_m2  cx  cy  lon  lat  domain  src
HH_n  HH_mean  HH_std  HH_cv  HH_enl  HH_mean_db  HH_sdev_db  HH_min_db …
HV_n  HV_mean  …
```

The prefix comes from the band's own name: a GCOV `HHHH` term is the HH power,
and a GeoTIFF band called `gamma0_HH` is the same channel, so both export under
`HH_`. A raster that names nothing falls back to `b1_`, `b2_`.

`domain` and `src` are not decoration — a gamma0 figure is not reproducible
without knowing what the pixels were read as and which raster they came from.

DBF caps a field name at 10 characters and **truncates silently** past it, so
the same numbers are written again beside the shapefile as `<stem>_stats.csv`,
one row per ROI and band, under untruncated names. `Export CSV` writes that
table alone.

`npix` is the ROI's geometry — pixels whose centre falls inside the ring. A
band's `n` may be lower where that band has nodata.

## NISAR GCOV `.h5`

A GCOV product ships as HDF5. Handed one, RADIAL writes a VRT beside it
(`<product>_gcov.vrt`) stacking the frequency's **diagonal** covariance terms
as bands — HHHH, HVHV, VHVH, VVVV — with the grid's own geotransform and
projection. Nothing is copied; the VRT reads the HDF5 in place.

Off-diagonal terms are complex covariances. Their magnitude is a correlation,
not a backscatter, so they are not offered and a complex band is never measured.

This needs `h5py` (for the grid's corner, which the subdatasets do not carry)
and GDAL's HDF5 driver. Without them, convert with
[`DPQED_h52tif.py`](DPQED_h52tif.py) and load the GeoTIFF.

## Tests

No QGIS needed for either:

```bash
python3 tests_radial_stats.py      # the statistics, masking and field naming
python3 tests_radial_gui_stub.py   # the window, with PyQt5/QGIS/GDAL stubbed
```

`tests_radial_stats.py` execs the pure-helper slice of the module itself, so it
tests the code that ships. It checks the things a plausible implementation gets
quietly wrong: that the three domains describe one scene, that ENL recovers a
known number of looks, that the 2.5 dB log bias is not in `mean_db`, and that
two long field names cannot truncate into one DBF column.
