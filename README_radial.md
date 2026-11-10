# RADIAL — radiometric assessment of NISAR GCOV products

A QGIS window with one canvas. Draw rectangles or polygons over targets whose
backscatter you know something about, read gamma0 and its statistics for each
one, and export the whole set as a shapefile carrying every number.

Where [`DPQED_rival.py`](DPQED_rival.py) measures **where** a product puts the
ground, this measures **what** it reports there.

```
exec(open(r"path/to/DPQED_radial.py").read())       # QGIS 3.x  (Qt5)
```

**Which QGIS.** The 3.x series, which is Qt5. That is the only build this
targets.

`DPQED_radial_qt6.py` is a Qt6/QGIS-4 port that is **no longer maintained**. It
is left in the tree for reference, but nothing checks it any more and it has
already fallen behind `DPQED_radial.py`. Do not run it; take it from git
history if the port is ever wanted back.

Pan, zoom, Normalize and the R/G/B band picker behave exactly as they do in
RIVAL — same stretch, same clip, same pinning. The differences are one canvas
instead of two, ROIs instead of point picks, and statistics instead of offsets.

## Workflow

| Step | |
|---|---|
| **Load GCOV** | a GeoTIFF or VRT, or a NISAR `.h5` (wrapped in a VRT, below) |
| **Normalize** | stretch the view so the target is legible before you draw on it |
| **Select** | click an ROI to select it; drag a selected one to move it |
| **Rect** / **Polygon** | draw ROIs; each one fills a row as it closes |
| **Point Buffer** | click a target; the ROI is a square of a chosen side |
| **Export SHP** | polygons plus every statistic, and a full-named CSV beside it |

Every ROI is **numbered on the canvas** at its centre, in its outline's colour
— cyan, or yellow when selected — so the table's first column is not the only
place that number exists.

**Select** answers the question the canvas could not: *that one*. Click an ROI
and its table row is selected, so `Ctrl+Delete` and the editable Name and Class
cells apply to what you clicked. The **smallest** ROI under the click wins, so
one drawn inside another is still reachable; clicking open ground deselects.

**The view follows the selection.** Select a row and the canvas pans onto that
ROI, so the row and the thing it names are never in different places. Two
restraints: it does not **zoom** — the scale you are reading the scene at is
your decision, and clicking down the table to compare ROIs should not keep
changing it (`F5` fits an ROI when fitting is what you want) — and it does not
move at all when the ROI is **already comfortably on screen**, so clicking an
ROI on the canvas never shifts the ground under the cursor. An ROI larger than
the window is fitted, since panning cannot bring it into view.

**Moving an ROI.** Drag one that is *already selected* and it slides to new
ground, keeping its number, name and class, and re-measuring where it lands.
An ROI drawn by eye lands slightly off as often as not, and the only remedy
before this was to delete it and draw it again — which changes its number and
loses its label.

Only a *selected* ROI can be dragged: picking one up on the same click that
selects it would turn every unsteady click into a move, shifting the ROI before
its row had even appeared. A press only counts as a drag once the mouse has
travelled a few **screen** pixels, not map units — the hand that slips is the
same size at every zoom. `Escape` abandons a drag in progress; nothing has been
touched until the mouse is released.

The move is a **translation only** — the shape never changes. A drag that could
also reshape would change what an ROI measures without changing what it is
called, and figures already exported for it would quietly stop describing it.
If a move would land the ROI off the scene or on nodata, it is put back where
it was and says so, rather than sitting there reading as an empty measurement.

`Ctrl+1..7` selects the tool, `F5` zooms to the selected ROI, `Ctrl+Delete`
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

**Both conventions are measured from one read of the pixels**, so the CSV
exports gives both and you never have to export twice and line the files up by
hand. The convention varies *down the rows* — a `backscat` column, beside
`band` — rather than across the columns, where every statistic would have to be
renamed to fit a second set beside it.

The **Backscatter** selector therefore changes only what is *displayed*: the
table, the by-class summary and the shapefile follow it, and switching it
re-reads nothing. The shapefile carries one convention because DBF has a column
budget; the CSV has no such limit and carries both.

Every ROI records which convention is on display, in the `backscat` column. A
raster with no factor layer has gamma0 alone — it cannot offer sigma0, and
switching the selector will not drag it into a convention it was never measured
in.

One detail worth knowing: gamma0 keeps **every** valid pixel, while sigma0 keeps
only those where the factor is also valid. A factor missing over part of an ROI
is a gap in the conversion, not in what the product recorded, so the two
conventions can report slightly different pixel counts for the same ROI.

## Loading an ROI set back

`Load ROIs` reads a polygon layer and re-measures every ROI against whatever
raster is loaded. It reads what each ROI already knows about itself:

| Column | Also accepted as |
|---|---|
| `name` | `roi_name`, `label`, `site`, `id` |
| `class` | `roi_class`, `cover`, `landcover`, `type`, `category` |
| `kind` | `shape`, `geom_kind` |

matched case-insensitively, so a set exported here reloads as exactly what it
was — the classes included. If the layer has **no** class column, every ROI
takes the Drawing picker's class and the tool says so, because that is
otherwise a silent re-labelling of the whole set.

## Point buffer

Some targets you can point at but cannot outline: a corner reflector, a buoy, a
small clearing. **Point Buffer** takes one click and makes a square ROI of a
stated size, centred on it. A **Side (m)** box appears beside the tool buttons
while the tool is selected — and only while it is selected, since it means
nothing under the others — and the square follows the cursor at that size
before you click, so you can see what it covers on this scene.

The side is in **metres of the working CRS**: a real size on the ground, not a
number of pixels and not a size on screen, so the same setting gives the same
ROI over a 30 m product and a 10 m one. It is read at the moment of the click,
so changing the box changes the next ROI and never one already drawn.

Square rather than round, deliberately. An ROI is rasterized by whether a pixel
*centre* falls inside it, and a circle's edge is a staircase whose step count
depends on where the centre landed within its pixel — two clicks a metre apart
would give different pixel counts for the same radius. An axis-aligned square
does not do that, and over a target small enough to click on, a square and a
circle are not a radiometric difference.

The point of the tool is that every ROI is then **the same size**, which is
what makes a set of them comparable.

### How big, for ENL

**Suggest** sets the side to what an ENL estimate needs on the raster you have
loaded. Backscatter is a mean and converges fast; ENL is a ratio of moments and
does not, so the two want very different amounts of ground:

| Target on ENL | Independent samples | Pixels | 10 m | 20 m | 30 m |
|---|---|---|---|---|---|
| ±20% | 60 | 121 | 110 m | 220 m | 330 m |
| **±10%** (default) | 242 | 484 | **220 m** | **440 m** | **660 m** |
| ±5% | 968 | 1936 | 440 m | 880 m | 1400 m |

The chain behind those numbers, each link arguable and none of it hidden:

1. **ENL's relative standard error is ≈ √(2/N)** in the number of *independent*
   samples. Simulated here over gamma-distributed intensities at 1, 4 and 12
   looks, the realised spread runs 5–10% above that asymptote past N ≈ 200 and
   further below it, so N is inflated by 1.1 rather than taken from the limit.
2. **The estimator is biased high at small N** — 11% at N = 25, 2% at N = 100.
   That is the direction that matters: a small ROI reports *more* looks than
   the product has, so an under-sized ROI flatters the product rather than
   obviously breaking.
3. **Pixels are not independent samples.** Multilooking and the impulse
   response correlate neighbours, so the pixel count is divided by 2 before it
   becomes a sample count. Two is conservative for a product posted at about
   its resolution and **optimistic for a heavily oversampled one** — it is the
   one number here that is a rule of thumb rather than arithmetic, and it is a
   constant (`ENL_PIXELS_PER_SAMPLE`) so it can be argued with.

The default stays at 300 m because most ROIs are drawn to read backscatter,
where it is ample, and 660 m is a lot of uniform ground to demand. Press
Suggest when ENL is the point.

The detail panel reports what each ROI's ENL is actually worth — `±10%`,
`±22%`, or *too few pixels* — from its own pixel count, so an ROI that was
drawn too small says so instead of quietly reporting an optimistic number.

Point ROIs carry `point` in the `kind` column, so an export says which ROIs
were placed this way and which were outlined.

## Sorting the table

Click a column heading to sort by it — **descending first**, because the
question a radiometric table gets opened with is which ROI is the brightest,
the noisiest or the largest, not which is the least of them. Click the same
heading again to reverse it, and a third time to return to the order the ROIs
were drawn in, so there is always a way back to the sequence the ROI numbers
mean.

Numbers sort as numbers, so 10 comes after 9 rather than before it, and text
sorts case-folded, so `Old ice` and `old ice` order together. An ROI with
**nothing** in that column — an unmeasured statistic, a blank class — stays at
the bottom in *both* directions: missing is the absence of a figure, not a low
one, and reversing the order should not turn it into a high one. Ties keep ROI
order, so the rows do not reshuffle between one redraw and the next.

Sorting only reorders the view. It does not renumber the ROIs, and the exports
are unaffected.

## ROI classes

Each ROI carries a **class** — what it is over. The **Drawing:** picker says
which class you are working on, and it does both halves of that:

* a new ROI is drawn into it, so you mark ten vegetation patches, switch, and
  mark eight water ones;
* the canvas and the ROI table show **that class alone** — pick `water` and the
  vegetation ROIs come off the screen and out of the table.

The **Class** cell in each row is editable afterwards, and re-classing an ROI
while a filter is on moves it out of view, because the filter is what it says.
`(all)` at the top of the picker shows every ROI; nothing can be drawn there,
so the three drawing tools switch off while it is selected — an ROI drawn
under `all` would land in a class nobody chose. The picker is editable too, so
`vegetation`, `water`, `snow`, `old ice`, `new ice` and `sand` are the common
cases rather than the permitted ones — type anything and it becomes a class.

**Filtering changes what you see, never what you have.** Every export —
`Export SHP`, `Export CSV` and the two CSVs written beside the shapefile —
writes all the ROIs, whatever the picker is set to.

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

## Noise floor (NESZ), from water

Draw ROIs over calm water, class them `water`, and the tool reports the noise
floor the scene shows — in the context line, in the detail panel beside each
band, and as `<stem>_nesz.csv` next to the shapefile:

```
band   class   rois    n   nonpos   nesz_db   floor_db   note
HH     water      3  9412        0    -24.31     -25.02   upper bound: …
HV     water      3  9412      140    -26.88     -27.40   upper bound: …
```

**It is a bound, not a measurement, and the CSV says so in every row.** NESZ is
a property of the instrument and the geometry; nothing in an ROI separates
scene from noise, so what a water ROI measures is scene *plus* noise. That
makes the figure an upper bound on NESZ — the tightest one imagery can give,
and the one worth quoting when a noise budget has to come from the product
itself rather than from a calibration report.

Three things the number is sensitive to, all of them visible in the row:

* **Calm water is not zero-backscatter.** Wind roughening puts a real signal in
  it, so a windy lake raises the bound. `floor_db` — the darkest single ROI —
  is the tighter, noisier version of the same estimate; a large gap between it
  and `nesz_db` means the water ROIs disagree, which usually means wind.
* **`nonpos`.** A noise-subtracted product has already had its floor removed,
  so pixels come out at or below zero. Many of them and the bound is measuring
  the subtraction rather than the instrument.
* **`rois` and `n`.** Pooled by pixel count, in linear power, so a 4000-pixel
  lake and a 40-pixel pond are not equal evidence. Three ROIs is a figure; one
  is an anecdote.

Always in **sigma0**, whatever the Backscatter dropdown is showing. NESZ is
defined against sigma0, and a gamma0 figure carrying the name would be wrong
by the RTC factor — which varies across the scene, so the error would not even
be a constant. A product with no RTC factor therefore gets no estimate at all,
rather than a gamma0 one relabelled.

The detail panel puts the selected ROI's **margin** beside it — its sigma0 mean
minus the floor. That is the number that says whether an ROI was measured or
merely sampled the floor: a few dB of margin and its backscatter is mostly
noise, whatever its mean says.

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

`Export SHP` writes one polygon per ROI in the working CRS — every ROI, not
just the class on view:

```
roi  name  kind  npix  area_m2  cx  cy  lon  lat  domain  class  backscat  inc_deg  src
                                                    ^ varies per row in the CSV
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

### The file set

One `Export SHP` writes the whole set, and beside it one file set per class:

```
rois.shp                    every ROI
rois_stats.csv              every ROI x band, under untruncated names
rois_by_class.csv           the summary a report quotes
rois_nesz.csv               the noise floor, from the water ROIs

rois_vegetation.shp         that class alone
rois_vegetation_stats.csv
rois_water.shp
rois_water_stats.csv
```

The per-class files are written from the same records as the whole-set file, so
they cannot disagree with it about an ROI. They are written for **every** class
and regardless of the class filter, for the same reason the whole-set export
ignores it: what was measured is not a function of what happens to be on screen.

A class name becomes a filename fragment — lowercased, with anything outside
`a-z0-9-_` replaced, so `open water / lake` is a legal class and a legal
filename. Two classes that would land on one name get a numbered suffix rather
than one silently overwriting the other.

A **single-class** ROI set gets none of these. The whole-set file already is
that class, and a second copy under a longer name is a duplicate, not a
by-class export. There is no per-class `_by_class.csv` either — the whole-set
one already holds every class's row.

`npix` is the ROI's geometry — pixels whose centre falls inside the ring. A
band's `n` may be lower where that band has nodata.

The shapefile's attribute columns are the CSV's columns, built from one plan so
they cannot drift apart, and the write is checked at every step: a field the
driver will not take, a feature it refuses, an error on the writer. If QGIS's
writer produces no features anyway, the same records go out again through OGR
directly, and the message box names which writer wrote the file. A `.shp` with
no rows and no complaint — which is what this replaced — is no longer one of
the things that can happen.

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
python3 tests_radial_stats.py       # statistics, masking, field naming
python3 tests_radial_gui_stub.py    # the window, QGIS and Qt stubbed
python3 tests_gcov2tif.py           # a real .h5 -> a real COG
```

`tests_radial_gui_stub.py` also puts the shapefile records through a real OGR
round trip where `pyogrio` and `shapely` are installed — written to a real
`.shp`, read back, compared field by field. That part is skipped, not failed,
where they are not.

`tests_radial_stats.py` execs the pure-helper slice of the module itself, so it
tests the code that ships. It checks the things a plausible implementation gets
quietly wrong: that the three domains describe one scene, that ENL recovers a
known number of looks, that the 2.5 dB log bias is not in `mean_db`, and that
two long field names cannot truncate into one DBF column.
