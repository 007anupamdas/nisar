# cog_locate — manual geolocation checking for L-band products

Measure where an L-band feature *actually* sits, by looking at it and putting
the cursor on it. Reads Cloud-Optimized GeoTIFFs over HTTP range requests, so
no granule ever hits the disk.

This is the manual counterpart to the automatic `dqe_imw` / `dqe_integrated`
matcher. Use it to sanity-check what the matcher reports, to work a scene the
matcher fails on, or to measure absolute location error against surveyed
corner reflectors — which is the reference method, not a fallback.

```
cog_locate.py     CLI + raster/geo core
cog_viewer.py     builds the self-contained HTML viewer
nisar_h5.py       streams NISAR HDF5 products over HTTP range requests
```

Requires `numpy` and `rasterio`, plus `h5py` for NISAR HDF5 products. PNG encoding is done with `zlib` alone, so
Pillow and matplotlib are not needed — one less thing to prefetch onto an
air-gapped box. `python cog_locate.py selftest` builds a synthetic viewer with
no network and no rasterio at all, which is the quickest way to check the page
works in your browser.

## The three commands

```bash
# 1. What am I looking at? CRS, pixel size, overview pyramid, bounds.
python cog_locate.py info https://host/path/nisar_gcov_HH.tif

# 2. A PNG plus a JSON sidecar carrying the affine transform.
python cog_locate.py chip https://host/path/nisar_gcov_HH.tif \
    --center 34.8021,-118.0765 --size 3km --out chip.png

# 3. The interactive viewer.
python cog_locate.py view https://host/path/nisar_gcov_HH.tif \
    --center 34.8021,-118.0765 --size 3km --out check.html
```

`--center` is `lat,lon`. Use `--center-xy x,y` to give the raster's own map
coordinates, `--bbox minx,miny,maxx,maxy` for an explicit footprint, or
`--full` for the whole scene (cheap: it comes out of the overviews).

Accepted inputs: an `https://` URL, `s3://`, `gs://`, a local path, a NISAR
`.h5` (local or remote — see below), or a GDAL connection string.

## NISAR products from ASF

**There is no COG.** ASF publishes NISAR L2 as a single HDF5 and nothing else —
CMR lists exactly one data file per granule, plus browse PNGs. A GSLC granule
runs to ~22 GB. GDAL cannot help here either: its HDF5 driver will not open a
`/vsicurl/` path.

So `nisar_h5.py` streams the HDF5 itself. h5py can read from any seekable
file-like object, so it is given one backed by `Range:` requests with a block
cache; HDF5's own chunked layout means only the chunks intersecting your window
come over the wire. A window read off a 22 GB granule costs a few MB.

```bash
# what frequencies and polarizations does this granule have?
python cog_locate.py info https://nisar.asf.earthdatacloud.nasa.gov/NISAR/…/GRANULE.h5

URI          : …/GRANULE.h5
product      : LSAR GSLC
file size    : 22.17 GB  (streamed, not downloaded)
frequencyA   : pols HH, HV   EPSG:32611
    HH    …, complex64, chunks (128, 128), complex
suggested    : --rgb HH,HV,HH/HV
```

Every read reports what it cost, so the saving is visible rather than claimed.

**Credentials.** ASF gates the data GET behind Earthdata Login (the HEAD is
open, the GET is not — the redirect to `urs.earthdata.nasa.gov` returns 401).
Credentials are read from where they already live and are **never** taken on a
command line, where they would land in your shell history and the process table:

1. `$EARTHDATA_TOKEN` — a bearer token from
   <https://urs.earthdata.nasa.gov/profile> → Generate Token. Preferred: scoped
   and revocable, and not your password.
2. `~/.netrc` — the standard NASA/ASF mechanism:
   ```
   machine urs.earthdata.nasa.gov
     login YOUR_USERNAME
     password YOUR_PASSWORD
   ```
   then `chmod 600 ~/.netrc`.

Nothing here logs, echoes or persists a credential. You must also have accepted
the NISAR EULA once by downloading any granule through the Earthdata web UI.

**GSLC is complex.** Geocoded SLC stores complex amplitude; the tool converts to
intensity (`|z|²`) before the dB stretch, which is the right quantity both for
display and for locating a point target's peak.

`--freq A|B` selects the frequency sub-band, `--pol HH` a single polarization,
`--h5-block KB` tunes the range-request size.

## Multispectral composites

```bash
python cog_locate.py view …/GRANULE.h5 --rgb auto \
    --center 34.8021,-118.0765 --size 3km --values all --out rgb.html
```

`--rgb` takes three channels — polarizations, or ratios of two — mapped to red,
green and blue. Each is stretched on its own percentiles, so a weak cross-pol
channel is not crushed by a strong co-pol one.

`--rgb auto` picks the best composite the product actually contains:

| product | composite | reading it |
|---|---|---|
| quad-pol | `HH,HV,VV` | the conventional Pauli-like assignment |
| dual-pol (DH: HH+HV) | `HH,HV,HH/HV` | red = surface scattering, green = volume scattering from vegetation |

**A dual-pol granule has no VV.** A `DHDH` acquisition transmits H only and
receives H and V, so it carries HH and HV and nothing else — asking for VV gets
you a clear error listing what is actually there. The co/cross ratio stands in
for the third channel, and it is not filler: it separates surface from volume
scattering, which is most of what a three-colour SAR composite is read for.

For a plain COG, `--rgb` takes band numbers instead: `--rgb 3,2,1`.

`--values all` embeds every channel so the cursor reports each one's dB
(`HH -15.31 dB / HV -15.21 dB / HH/HV -0.10 dB`) and every picked point carries
all three into the CSV. It roughly triples the page size; the default `first`
embeds only the red channel, which is all snap-to-peak needs.

## Google Earth

Two ways out, for checking a position against imagery you trust.

**Picked points → KML.** The `Download KML` button in the viewer writes your
picks as placemarks, each carrying its map coordinate, source pixel and channel
values in its description. Open it straight in Google Earth.

**The chip itself → KMZ.** Build the page with `--kml out.kmz` and you also get
the NISAR chip as a GroundOverlay, so you can drape it over Google Earth's
basemap and see directly whether a feature lands where it should:

```bash
python cog_locate.py view …/GRANULE.h5 --rgb auto \
    --center 34.8021,-118.0765 --size 3km --out check.html --kml check.kmz
```

The overlay is placed with `<gx:LatLonQuad>`, which takes the four true corners.
That is exact for any projection — a `<LatLonBox>` can only model a north-up
rectangle plus one rotation angle, which is wrong for a UTM chip. A path ending
`.kml` instead writes just the footprint and overlay points, with no imagery.

One caveat worth knowing: Google Earth's own basemap has its own geolocation
error, typically a few metres and occasionally much worse in relief. It is a
good sanity check and a poor reference — for a number you intend to quote, use
surveyed points via `--overlay`.

## What the viewer does

Open the HTML file. It is one self-contained page — imagery, georeferencing
and pixel values are all embedded, nothing is fetched. That matters when the
machine with internet access is not the machine you want to look at imagery on.

Move the cursor and the box at the bottom left reads out, live:

```
lat 34.8308179   34°49'50.945"N
lon -118.2465880   118°14'47.717"W
X   386009.990   Y 3854989.998
src px  col 600.999  row 501.000
value   13.70 dB
```

Latitude/longitude, the projected map coordinate, the pixel in the
**full-resolution source raster** (the same pixel space `dqe_integrated`
works in — not the chip's), and the pixel value in the units of the stretch.

Click to drop a marker. With **Snap** on (default), the marker jumps to the
brightest pixel nearby and is then refined to sub-pixel precision by fitting a
parabola through the peak and its neighbours — which is how a point target's
position should be read off an image, not by eye. Drag a marker to move it,
`Alt`-click to delete, rename it in the table, export everything as CSV.

Keys: `f` fit · `1` 1:1 · `g` graticule · `s` snap · `l` link panels ·
`t` truth overlay · `i` invert · `+`/`-` zoom · arrows pan · `Del` delete.
Wheel zooms about the cursor; drag pans. Brightness/contrast/gamma sliders
retone the chip in the page, so a dark scene does not need re-extracting.

## Measuring location accuracy

### Against surveyed points (absolute location error)

The direct measurement. Give it a CSV of surveyed positions:

```csv
name,lat,lon
CR1,34.830753462,-118.246718188
CR2,34.784058855,-118.191369116
```

```bash
python cog_locate.py view nisar_gcov_HH.tif \
    --center 34.8308,-118.2467 --size 1500m \
    --overlay reflectors.csv --out ale.html
```

Surveyed points draw as red circled crosses. Click each imaged peak; the
**Accuracy** tab pairs every pick with the nearest surveyed point (within
`--assoc-radius`, default 100 m) and reports

`ΔX = image − survey`, so positive means the feature images east / north of
where it really is.

Column names are flexible — `lat`/`latitude`, `lon`/`longitude`,
`x`/`easting`, `y`/`northing`, `name`/`id`/`site` all work, and projected
coordinates are accepted instead of lat/lon.

### Against a reference image

```bash
python cog_locate.py view nisar_gcov_HH.tif --b s1_reference.tif \
    --center 34.80,-118.07 --size 4km --out pair.html
```

Two linked panels. The reference is read over the **same ground footprint**,
not the same pixel window — the two products rarely share a grid, and it is
the footprint that has to match for the comparison to mean anything. Panning
or zooming either panel moves the other to match.

Click a feature in A, then the same feature in B. Points are auto-named in
order (`P1`, `P2`, …) per panel, so the n-th click in A pairs with the n-th in
B. The Accuracy tab reports `ΔX = A − B` as `across` and `ΔY` as `along` —
the same sign convention as the `dqe_imw` CSVs, so the numbers sit directly
alongside the matcher's output.

An optical reference usually wants `--stretch-b linear`.

**On grid orientation.** If the two products are in different projections —
a UTM NISAR grid against a geographic reference, say — their grids are
genuinely rotated relative to each other by the meridian convergence, the
angle between grid north and true north (`Δλ · sin φ`, which reaches 0.68° at
1.2° from a UTM central meridian). Linking matches the panels exactly at the
centre of the view, so features drift apart the further you look from it. The
Scene info tab states each panel's grid bearing and warns when they differ,
with the drift per kilometre. **Measurements are unaffected**: every pick
carries its own coordinates, and cross-CRS offsets are differenced in lon/lat
scaled to metres at that latitude rather than by subtracting incompatible map
coordinates.

### What the summary means

| | |
|---|---|
| **bias ΔX / ΔY** | the systematic shift — the interesting number |
| **σ X / σ Y** | scatter of the individual picks |
| **RMSE (2D)** | `sqrt(mean(ΔX² + ΔY²))`, bias and scatter together |
| **RMSE debiased** | scatter alone, bias removed |
| **CE90 empirical** | 90th percentile of the radial errors |
| **CE90 ≈ 2.146σ** | parametric form, assumes a circular normal error |

A large bias with a small σ is a systematic geolocation shift, and that is the
result worth chasing: it points at timing, geometry or DEM height rather than
at the imagery. The empirical CE90 needs a fair number of points before it
means much; the parametric form is the more stable estimate on a handful of
reflectors.

## Getting the resolution right

**This matters more than anything else here.** The viewer warns you on the
Scene info tab when a panel is decimated, because a picked position is only
ever as good as the pixels you picked it from.

`--size` sets the ground extent; `--max-px` (default 1600) caps the chip
dimension. Ask for more pixels than the cap and the read is decimated, served
straight out of the COG's overview pyramid — fast, and fine for finding your
way around, useless for measuring. For an actual measurement, keep the window
under the cap so it reads at full resolution:

```bash
# 1600 px cap, 10 m pixels -> anything under 16 km reads at full resolution
python cog_locate.py view nisar_gcov_HH.tif \
    --center 34.8308,-118.2467 --size 2km \
    --resample nearest --overlay reflectors.csv --out ale.html
```

`--resample nearest` stops any averaging from moving a peak. The Scene info
tab always states the decimation actually applied, so check it before quoting
a number.

## Rendering

`--stretch db` (default) is `10·log10`, right for a GCOV power product.
`--stretch amp-db` is `20·log10` for amplitude; `linear` for optical.
`--pct 2 98` sets the clip percentiles, or pin the range with `--vmin/--vmax`
in post-stretch units (dB). `--cmap gray|gray-inv|viridis|inferno|magma`.

Zeros are treated as nodata when the raster declares no nodata value, since
that is what essentially every SAR product means by them; `--keep-zeros`
turns that off for data where zero is real.

## Cost

A `view` of a 3 km chip out of a 194 MB remote COG, measured end to end:

```
$ time python cog_locate.py view https://…/B04.tif --center 16.6874,33.5147 --size 3000m
[A] 300 x 300 px, dec 1x, 10 m/px, 100.0% valid
real    0m1.382s
```

Only the tiles the window touches are fetched. The GDAL settings that make
that true (`GDAL_DISABLE_READDIR_ON_OPEN`, HTTP/2 multiplexing, the CURL
block cache) are set in `GDAL_REMOTE_OPTS` in `cog_locate.py`.

If the source is not a real COG — no internal tiling, no overview pyramid —
`info` says so, and windowed reads will be slow because GDAL has to walk
whole strips. That is a property of the file, not of this tool.

## Page size

Raw pixel values are embedded (uint16-quantized) to drive the dB readout and
snap-to-peak. A 1600×1600 chip runs to roughly 8 MB of page. `--no-values`
drops them and shrinks the file by about 4×, at the cost of both features.

The quantization is deliberately **not** compressed: plain base64 decodes with
`atob()` in any browser, whereas inflating in-page needs `DecompressionStream`,
which a locked-down corporate browser may not have.

## Accuracy of the tool itself

Lon/lat comes from a bilinear mesh (`--grid-nodes`, default 17×17) sampled
through the real projection at build time, so the browser needs no projection
library. The mesh's worst-case interpolation error is measured at build time
and reported on the Scene info tab — for a few-km chip it lands around
10⁻⁴ m, five orders of magnitude below the pixel.

End-to-end, against a synthetic scene with a known offset injected: a click
deliberately placed 3.2 px off a point target, with snap on, recovers an
injected 12.000 m / 7.000 m offset as **11.990 m / 6.998 m** — about 1 cm, or
0.001 px. The dominant error in practice is you deciding which pixel is the
feature, not the arithmetic.

## Verifying a change

```bash
python cog_locate.py selftest --out selftest.html   # no network, no rasterio
```

Three synthetic point targets sit 1.5 px from their listed "surveyed"
positions. Click each with snap on: the ALE column should read 1.5 px worth of
metres, and the bias should be consistent across all three.
