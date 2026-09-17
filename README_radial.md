# RADIAL — radiometric assessment of NISAR GCOV products

A QGIS window with one canvas. Draw rectangles or polygons over targets whose
backscatter you know something about, read gamma0 and its statistics for each
one, and export the whole set as a shapefile carrying every number.

Where [`DPQED_rival.py`](DPQED_rival.py) measures **where** a product puts the
ground, this measures **what** it reports there.

```
exec(open(r"path/to/DPQED_radial.py").read())       # QGIS 3.x  (Qt5)
exec(open(r"path/to/DPQED_radial_qt6.py").read())   # QGIS 4.x  (Qt6)
```

**Which file.** Run the Qt6 one if QGIS greets you with `PyQt5 classes cannot be
imported in a QGIS build based on Qt6` — that is QGIS 4.x, and anything else
built against Qt6. The two are the same tool: they differ only in how they name
Qt and QGIS things, and everything between the `PURE HELPERS` markers is
byte-identical, which `tests_radial_stats.py` checks so they cannot drift.

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

## gamma0 or sigma0

GCOV stores **gamma0** — backscatter referred to the terrain's own sloped area.
**sigma0** refers the same measurement to a flat ground area instead. On flat
ground they are identical; on a slope they are several dB apart, and which one
a figure is cannot be read off the number.

The product ships the conversion with the data, as a per-pixel layer:

```
sigma0 = gamma0 × rtcGammaToSigmaFactor
```

Load a GCOV from its `.h5` and that layer rides along in the VRT, so the **As:**
selector offers sigma0. Pick it and every ROI is re-measured; the factor is
applied to **linear power**, which is the only place a ratio of areas belongs —
on amplitudes it would be out by a square, on dB pixels it would be an addition.

The factor is a band of the raster but not a channel. It stays out of the
measured bands and out of the export's columns (its mean is a ratio of areas,
and a column of those under the same headings as the gamma0 columns would be
read as a backscatter), but it stays in the R/G/B picker — looking at it is how
you see where the terrain correction is doing the most work, and therefore where
the two conventions have least to do with each other.

Every ROI records which convention produced it, in the `backscat` column. A
raster with no factor layer cannot offer sigma0 at all, and nothing is ever
labelled sigma0 on the strength of a conversion that did not happen.

## ROI classes

Each ROI carries a **class** — what it is over. The **Drawing:** picker sets the
class of the next ROI you draw, so you mark ten vegetation patches, switch, and
mark eight water ones; the **Class** cell in each row is editable afterwards.
The picker is editable too, so `vegetation`, `water` and `snow` are the common
cases rather than the permitted ones — type anything and it becomes a class.

Statistics are then summarised **per class**, in the table beside the ROI list:

| Class | ROIs | Mean dB | Spread dB | ENL |
|---|---|---|---|---|
| vegetation | 10 | −7.55 | 0.90 | 4.1 |
| water | 8 | −22.35 | 0.70 | 3.9 |
| all | 18 | −13.12 | 15.60 | 4.0 |

This is the point of the feature. **Spread** is the brightest ROI mean minus the
darkest: within one land cover that is the product's radiometric uniformity,
and across two it is just the gap between vegetation and water — 15.6 dB of
land cover, which you knew before you drew anything. The `all` row is kept, and
labelled, because scene-wide brightness is worth a glance; it is not a
uniformity figure. It is left out when there is only one class.

Classes are case-folded to group and shown as first typed, so `Water` and
`water` are one class. A misspelling stays its own class — which is how you
notice it, rather than having it quietly folded into the one you meant.

The class is a column in the shapefile and in the per-ROI CSV, and **Export SHP**
writes the summary itself beside them as `<stem>_by_class.csv`, one row per
class and band. That is the table a report quotes, and no per-ROI listing states
it outright.

## Incidence angle

Each ROI reports the incidence angle at its **centre**, in degrees, in the
`inc_deg` column — in the table and in both exports. Backscatter depends on it,
so two ROIs that disagree are not comparable until you know whether they were
looked at from the same angle.

It comes from the product's `metadata/radarGrid/incidenceAngle` cube, by
whichever of two routes the raster allows:

- a **band** named `incidenceAngle`, which `DPQED_gcov2tif.py` resamples onto
  the image grid — the value at the ROI's centre pixel, or the ROI mean where
  that pixel has no data;
- the **cube itself**, when a GCOV was loaded straight from its `.h5`, sampled
  bilinearly at the ROI centre. A VRT cannot carry the cube — it stacks
  datasets on one grid, and the geometry cubes are on another, coarser one — so
  it is read once at load and sampled per ROI.

Either way it is **not measured as backscatter**: its mean is an angle, and an
angle in a column headed `mean_db` beside the gamma0 columns would be read as
one. A raster with neither route leaves `inc_deg` empty rather than guessing,
and an ROI outside the cube gets nothing rather than an extrapolation — the
failure that would otherwise look like a measurement.

The cubes are sampled at several heights above the ellipsoid. Choosing between
them properly needs a DEM, which this has none of, so the layer nearest the
ellipsoid is used — and the height actually taken is written into the TIFF
header as `INCIDENCE_ANGLE_HEIGHT_M` rather than left implicit.

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

A constant factor rescales every pixel alike, so switching to sigma0 moves
`mean_db` and leaves `cv` and `enl` where they were. A real factor varies with
slope and moves them too — that is the terrain, not an error.

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

**Domain is not the sigma0 control.** Two selectors sit side by side and they do
different jobs. **Domain** says what the pixels *hold* — power, amplitude or dB
— and never changes the convention. **Backscatter** converts gamma0 to sigma0,
and is the only thing that does. Picking `Power` does not give you sigma0; if
the Backscatter selector is greyed out, the raster carries no RTC factor and
every figure is gamma0.

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
roi  name  kind  npix  area_m2  cx  cy  lon  lat  domain  class  backscat  inc_deg  src
HH_n  HH_mean  HH_std  HH_cv  HH_enl  HH_mean_db  HH_sdev_db  HH_min_db …
HV_n  HV_mean  …
```

The prefix comes from the band's own name: a GCOV `HHHH` term is the HH power,
and a GeoTIFF band called `gamma0_HH` is the same channel, so both export under
`HH_`. A raster that names nothing falls back to `b1_`, `b2_`.

`domain`, `backscat` and `src` are not decoration — a backscatter figure is not
reproducible without knowing what the pixels were read as, which convention
they are in, and which raster they came from. The statistic columns keep the
same names under either convention, so a gamma0 export and a sigma0 export of
one ROI set compare column by column.

DBF caps a field name at 10 characters and **truncates silently** past it, so
the same numbers are written again beside the shapefile as `<stem>_stats.csv`,
one row per ROI and band, under untruncated names. `Export CSV` writes that
table alone.

`npix` is the ROI's geometry — pixels whose centre falls inside the ring. A
band's `n` may be lower where that band has nodata.

## Getting a GCOV in: `.h5`, or a TIF

RADIAL reads a GCOV `.h5` directly **where QGIS has h5py**. Many builds do not,
and then it says so and sends you to a converter.

Set two lines at the top of `DPQED_gcov2tif.py` and run it, the way
`DPQED_h52tif.py` is run:

```python
INPUT = r"V:\...\NISAR_L2_PR_GCOV_..._001.h5"
OUTPUT = None       # None: '<input>_gcov.tif', beside the product
```

There are no options, because there is nothing to choose that the product does
not already say:

| | |
|---|---|
| **band** | from the granule name — `NISAR_L2_…` is L-band, `NISAR_S2_…` is S — and from the file itself when the name says nothing, or says something the file does not hold. |
| **frequency** | whichever the product carries; A when it carries both, that being the wideband channel. |
| **terms** | the diagonal ones that are there. |
| **RTC factor, incidence cube** | found beside the terms, in whatever group the product keeps them in. |

Asking a person to repeat any of that is asking them to get it wrong. (Note
that RIVAL's `band_from_name` only recognises a whole `LSAR`/`SSAR` tag, which
no real granule name carries — it is why RIVAL falls back to the sidecar's
Sensor field. Here the file itself is the fallback, and a better one.)

Use `DPQED_gcov2tif.py`, not `DPQED_h52tif.py`. The older script predates GCOV
— it is written for a GSLC, takes the magnitude of complex channels, and leaves
three things behind, none of which announces itself:

| | |
|---|---|
| **band names** | RADIAL reads the polarization from a band's description. Unnamed, every column exports as `b1_`, `b2_`, and two products' exports stop lining up by name. |
| **the RTC factor** | not carried across at all, so the TIF cannot offer sigma0. |
| **the half pixel** | a grid states pixel *centres*; a geotransform is anchored on the *edge*. 15 m on a 30 m grid. |

It also carries the **incidence angle**, resampled from the product's
`metadata/radarGrid` cube onto the image grid as one more named band, so
`inc_deg` is available from the TIF alone.

A note on layout: RADIAL once assumed the frequency groups live under
`GCOV/grids/`. Some products put them directly under `GCOV/`. Both are read
now — the group is taken from where a covariance term was actually found
rather than rebuilt from the band and frequency.

The output is a **Cloud Optimized GeoTIFF** — tiled, with an averaged overview
pyramid, laid out by GDAL's own COG driver so headers and overviews precede the
full-resolution data. That is what makes panning a 20000 px scene bearable, and
it is what `cog_locate.py` expects of a raster.

### "Can I store the factor as an attribute or a header?"

No — and this is worth being clear about. `rtcGammaToSigmaFactor` is a value
**per pixel**: it depends on the local slope, which is the entire reason it
exists. A single number in the TIFF tags would be a different measurement,
correct only where the ground happens to be flat. It goes in as a **band**.

What does go in the header is *which band that is*:

```
RTC_GAMMA_TO_SIGMA_BAND = 3
```

a plain GDAL metadata item. RADIAL looks for a band whose name matches first,
and falls back to that tag — so a TIF written by some other tool can declare
its factor band without renaming anything. `INCIDENCE_ANGLE_BAND` does the
same for the incidence angle. Set either with
`gdal_edit.py -mo RTC_GAMMA_TO_SIGMA_BAND=3 your.tif`, provided the band is
actually in the file; no tag can conjure a layer that was never written.

The converter writes gamma0, as the product holds it, plus the factor. It does
not bake in sigma0 — that would produce a file indistinguishable from a gamma0
one. RADIAL does the conversion, and records which convention each figure is in.

## Reading a GCOV `.h5` directly

A GCOV product ships as HDF5. Handed one, RADIAL writes a VRT beside it
(`<product>_gcov.vrt`) stacking the frequency's **diagonal** covariance terms
as bands — HHHH, HVHV, VHVH, VVVV — plus `rtcGammaToSigmaFactor` when the
product carries it, with the grid's own geotransform and projection. Nothing is
copied; the VRT reads the HDF5 in place. That is what makes sigma0 available
from the `.h5` alone, with no second file to keep aligned with the first.

Off-diagonal terms are complex covariances. Their magnitude is a correlation,
not a backscatter, so they are not offered and a complex band is never measured.

This needs `h5py` (for the grid's corner, which the subdatasets do not carry)
and GDAL's HDF5 driver. Without them, convert with
[`DPQED_h52tif.py`](DPQED_h52tif.py) and load the GeoTIFF.

## Tests

No QGIS needed for any of them:

```bash
python3 tests_radial_stats.py                          # statistics, masking, field naming
python3 tests_radial_gui_stub.py                       # the Qt5 window, QGIS stubbed
python3 tests_gcov2tif.py                              # a real .h5 -> a real COG
QT_QPA_PLATFORM=offscreen python3 tests_radial_qt6.py  # the Qt6 window, for real
```

The Qt6 suite is the odd one out, deliberately. Mocking PyQt away is right for
testing wiring and wrong for testing a port: a mock answers to any spelling, so
`Qt.CrossCursor` and `Qt.CursorShape.CrossCursor` would both pass and only one
of them works on a Qt6 build. So it installs real PyQt6, runs Qt offscreen and
stubs only QGIS — every widget built, every signal connected, every enum
resolved by Qt itself. Its QGIS stub offers the QGIS 4 spellings only, then
reloads the module against a QGIS 3 stub, so both halves of the name resolution
are exercised rather than only the half the author's own build has.

`tests_radial_stats.py` execs the pure-helper slice of the module itself, so it
tests the code that ships. It checks the things a plausible implementation gets
quietly wrong: that the three domains describe one scene, that ENL recovers a
known number of looks, that the 2.5 dB log bias is not in `mean_db`, and that
two long field names cannot truncate into one DBF column.

## Porting notes, if you touch the Qt6 file

The two builds differ in exactly these places, each verified against real
PyQt5 5.15.11 and PyQt6 6.11:

| | Qt5 | Qt6 |
|---|---|---|
| enums | `Qt.CrossCursor` | `Qt.CursorShape.CrossCursor`, and so on throughout |
| `QShortcut` | `QtWidgets` | `QtGui` |
| field types | `QVariant.Int` | `QMetaType.Type.Int` — `QVariant` still imports on Qt6 but carries no type members |
| geometry / WKB types | `QgsWkbTypes.*` | `Qgis.GeometryType.*`, `Qgis.WkbType.*` |
| raster stat flags | `QgsRasterBandStats.*` | `Qgis.RasterBandStatistic.*` |

The QGIS rows are a QGIS-version difference rather than a Qt one, so the Qt6
file resolves them at import with a fallback to the older spelling. A build
with neither fails there, naming the symbol, rather than three layers down in
an export at the end of a session's work.

Every scoped enum spelling above also works under PyQt5 5.15, so one file
using `qgis.PyQt.*` imports could serve both builds. That is a different change
from a port, and is not what is here.
