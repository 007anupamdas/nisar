#!/usr/bin/env python3
"""
Self-contained HTML viewer generator for cog_locate.

The output is one file with no external requests -- imagery, georeferencing and
raw values are all embedded. That is deliberate: the machine that reads the COG
(internet-facing) is often not the machine you want to inspect the imagery on
(intranet), and a single HTML file crosses that gap without a server.

Everything the page needs to turn a mouse position into a coordinate:
  * `transform` -- chip pixel -> map coordinate (affine, from the source COG)
  * `grid`      -- lon/lat sampled on a coarse mesh, bilinearly interpolated
                   in-page, so no projection library is needed in the browser
  * `src_inv`   -- map coordinate -> full-resolution source pixel, so a point
                   can be quoted in the same pixel space dqe_integrated uses
  * `values`    -- uint16-quantized displayed values, for the dB readout and
                   for snapping a click to the local backscatter peak
"""

from __future__ import annotations

import json
from typing import Dict, List


def build_viewer_html(panels: List[Dict],
                      overlay: List[Dict],
                      title: str = "cog_locate",
                      assoc_radius: float = 100.0,
                      snap_radius: int = 6) -> str:
    payload = {
        "panels": panels,
        "overlay": overlay,
        "assocRadius": float(assoc_radius),
        "snapRadius": int(snap_radius),
        "title": title,
    }
    blob = json.dumps(_finite(payload), allow_nan=False, separators=(",", ":"))
    return (_TEMPLATE
            .replace("__TITLE__", _escape(title))
            .replace("__PAYLOAD__", blob.replace("</", "<\\/")))


def _finite(obj):
    """Replace non-finite floats with null.

    JSON has no NaN, and the residual of a lon/lat mesh that failed to build is
    exactly the sort of value that would otherwise take the whole page down at
    serialisation time. The viewer already guards every such field.
    """
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #14161a; --panel: #1c1f26; --line: #2e333d; --fg: #e6e9ef;
    --dim: #98a1b0; --accent: #58a6ff; --good: #3fb950; --warn: #d29922;
    --bad: #f85149; --pick: #ffd33d; --truth: #ff7b72;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
           padding: 8px 12px; background: var(--panel);
           border-bottom: 1px solid var(--line); }
  header h1 { font-size: 14px; margin: 0 12px 0 0; font-weight: 600; }
  button, select { background: #262b34; color: var(--fg); border: 1px solid var(--line);
           border-radius: 5px; padding: 4px 9px; font: inherit; cursor: pointer; }
  button:hover { background: #313847; }
  button.on { background: var(--accent); border-color: var(--accent); color: #05070a;
              font-weight: 600; }
  label.sl { display: flex; align-items: center; gap: 5px; color: var(--dim);
             font-size: 12px; }
  input[type=range] { width: 92px; }
  #stage { display: flex; gap: 6px; padding: 6px; height: 60vh; min-height: 320px; }
  .pane { position: relative; flex: 1 1 0; background: #000;
          border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .pane.active { border-color: var(--accent); }
  .pane canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
  .cap { position: absolute; top: 0; left: 0; right: 0; padding: 4px 8px;
         background: linear-gradient(#000c, #0000); font-size: 12px;
         pointer-events: none; display: flex; justify-content: space-between; gap: 8px; }
  .cap b { font-weight: 600; }
  .cap .meta { color: var(--dim); font-variant-numeric: tabular-nums; }
  .hud:empty { display: none; }
  .hud { position: absolute; left: 8px; bottom: 8px; padding: 6px 9px;
         background: #000000cc; border: 1px solid var(--line); border-radius: 5px;
         font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
         white-space: pre; pointer-events: none; }
  #tabs { display: flex; gap: 4px; padding: 0 12px; border-bottom: 1px solid var(--line); }
  #tabs button { border-radius: 5px 5px 0 0; border-bottom: none; }
  section.tab { display: none; padding: 10px 12px 24px; }
  section.tab.on { display: block; }
  table { border-collapse: collapse; width: 100%; font: 12px/1.5 ui-monospace,
          SFMono-Regular, Consolas, monospace; }
  th, td { border-bottom: 1px solid var(--line); padding: 3px 7px; text-align: right;
           white-space: nowrap; }
  th { color: var(--dim); font-weight: 600; text-align: right; position: sticky;
       top: 0; background: var(--bg); }
  td:first-child, th:first-child, td.l, th.l { text-align: left; }
  tr.sel td { background: #1d3350; }
  .wrap { max-height: 42vh; overflow: auto; border: 1px solid var(--line);
          border-radius: 6px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
         margin-bottom: 8px; }
  .cards { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
          padding: 8px 12px; min-width: 132px; }
  .card .k { color: var(--dim); font-size: 11px; text-transform: uppercase;
             letter-spacing: .04em; }
  .card .v { font: 600 17px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace; }
  .note { color: var(--dim); font-size: 12px; max-width: 78ch; }
  .note code { background: #262b34; padding: 1px 4px; border-radius: 3px; }
  kbd { background: #262b34; border: 1px solid var(--line); border-bottom-width: 2px;
        border-radius: 4px; padding: 0 5px; font: 11px ui-monospace, monospace; }
  .warnbox { border-left: 3px solid var(--warn); padding: 6px 10px; margin: 8px 0;
             background: #d2992218; color: #f0d69a; font-size: 12px; }
  h3 { font-size: 13px; margin: 16px 0 6px; }
  input[type=text] { background: #12151a; color: var(--fg); border: 1px solid var(--line);
                     border-radius: 4px; padding: 1px 5px; font: inherit; width: 88px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <button id="bFit" title="Fit chip to window (f)">Fit</button>
  <button id="bOne" title="One screen pixel per chip pixel (1)">1:1</button>
  <button id="bLink" title="Lock both panels to the same ground footprint (l)">Link</button>
  <button id="bGrid" title="Map coordinate graticule (g)">Grid</button>
  <button id="bSnap" title="Snap clicks to the local backscatter peak (s)">Snap</button>
  <button id="bTruth" title="Show surveyed overlay points (t)">Truth</button>
  <label class="sl">bright <input type="range" id="rBright" min="-100" max="100" value="0"></label>
  <label class="sl">contr <input type="range" id="rContrast" min="-100" max="100" value="0"></label>
  <label class="sl">gamma <input type="range" id="rGamma" min="20" max="300" value="100"></label>
  <button id="bInv" title="Invert (i)">Invert</button>
  <button id="bReset">Reset tone</button>
</header>

<div id="stage"></div>

<div id="tabs">
  <button data-tab="points" class="on">Points</button>
  <button data-tab="ale">Accuracy</button>
  <button data-tab="meta">Scene info</button>
  <button data-tab="help">Help</button>
</div>

<section class="tab on" id="tab-points">
  <div class="row">
    <button id="bClear">Clear all points</button>
    <button id="bDelSel">Delete selected</button>
    <button id="bCopy">Copy CSV</button>
    <button id="bDown">Download CSV</button>
    <span class="note" id="pointsHint"></span>
  </div>
  <div class="wrap"><table id="tPoints"></table></div>
</section>

<section class="tab" id="tab-ale">
  <div id="aleHead"></div>
  <div class="cards" id="aleCards"></div>
  <div class="wrap"><table id="tAle"></table></div>
  <div class="row" style="margin-top:8px">
    <button id="bDownAle">Download accuracy CSV</button>
  </div>
  <p class="note" id="aleNote"></p>
</section>

<section class="tab" id="tab-meta"><div id="metaBody"></div></section>

<section class="tab" id="tab-help">
  <p class="note">
    <b>Reading a coordinate.</b> Move the cursor: the box at the bottom left of each
    panel shows latitude/longitude, the projected map coordinate, the pixel in the
    full-resolution source raster, and the pixel value in the units of the stretch
    (dB for a power product). Nothing is interpolated except lon/lat, which comes
    from a bilinear mesh accurate to well under a pixel over a chip this size.
  </p>
  <p class="note">
    <b>Picking a point.</b> Click to drop a marker. With <kbd>Snap</kbd> on, the
    marker jumps to the brightest pixel nearby and is then refined to sub-pixel
    precision by fitting a parabola through the peak and its neighbours -- which is
    how a corner reflector's position should be read off an image. Drag a marker to
    move it, <kbd>Alt</kbd>+click it to delete, and edit its name in the table.
  </p>
  <p class="note">
    <b>Measuring accuracy.</b> Two ways, both on the Accuracy tab.
    With a surveyed overlay loaded, each picked point is matched to the nearest
    surveyed point and the difference is the absolute location error.
    With two panels, points that share a name are paired -- click a feature in the
    left panel and the same feature in the right, and the offset between them is
    reported. Sign convention matches the dqe_imw CSVs:
    <code>across = &Delta;X (easting)</code>, <code>along = &Delta;Y (northing)</code>,
    both computed as <em>this image minus the reference</em>.
  </p>
  <p class="note">
    <b>Keys.</b>
    <kbd>f</kbd> fit &middot; <kbd>1</kbd> 1:1 &middot; <kbd>g</kbd> grid &middot;
    <kbd>s</kbd> snap &middot; <kbd>l</kbd> link panels &middot; <kbd>t</kbd> truth
    overlay &middot; <kbd>i</kbd> invert &middot; <kbd>+</kbd>/<kbd>-</kbd> zoom
    &middot; arrows pan &middot; <kbd>Del</kbd> delete selected &middot;
    <kbd>Esc</kbd> deselect. Wheel zooms about the cursor; drag pans.
  </p>
</section>

<script>
"use strict";
const DATA = __PAYLOAD__;

/* ---------------------------------------------------------------- helpers */
const $ = s => document.querySelector(s);
const clamp = (v, a, b) => v < a ? a : (v > b ? b : v);
const fmt = (v, n) => (v === null || v === undefined || !isFinite(v)) ? "--" : v.toFixed(n);

function dms(deg, isLat) {
  if (!isFinite(deg)) return "--";
  const hemi = isLat ? (deg < 0 ? "S" : "N") : (deg < 0 ? "W" : "E");
  let a = Math.abs(deg);
  const d = Math.floor(a); a = (a - d) * 60;
  const m = Math.floor(a); const s = (a - m) * 60;
  return `${d}°${String(m).padStart(2, "0")}'${s.toFixed(3).padStart(6, "0")}"${hemi}`;
}

function b64ToU16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
}

/* ------------------------------------------------------------------ Panel */
class Panel {
  constructor(spec, index) {
    this.spec = spec;
    this.index = index;
    this.name = String.fromCharCode(65 + index);       // "A", "B"
    const t = spec.transform;
    this.t = { a: t[0], b: t[1], c: t[2], d: t[3], e: t[4], f: t[5] };
    const det = this.t.a * this.t.e - this.t.b * this.t.d;
    this.tinv = det === 0 ? null : {
      a: this.t.e / det, b: -this.t.b / det,
      d: -this.t.d / det, e: this.t.a / det,
    };
    this.si = spec.src_inv;
    this.vals = (spec.values && spec.values.b64) ? b64ToU16(spec.values.b64) : null;
    this.vlo = spec.values ? spec.values.lo : 0;
    this.vhi = spec.values ? spec.values.hi : 1;
    // Ground aspect: how many metres a vertical pixel covers per horizontal one.
    // 1 for a projected CRS with square pixels; ~1.2 for a geographic grid.
    this.ar = (spec.px_y_m && spec.px_x_m) ? (spec.px_y_m / spec.px_x_m) : 1;
    if (!isFinite(this.ar) || this.ar <= 0) this.ar = 1;
    this.view = { scale: 1, tx: 0, ty: 0, fitted: false };
    this.build();
  }

  /* --- geometry ------------------------------------------------------- */
  toMap(u, v) {
    return { x: this.t.c + this.t.a * u + this.t.b * v,
             y: this.t.f + this.t.d * u + this.t.e * v };
  }
  fromMap(x, y) {
    if (!this.tinv) return { u: 0, v: 0 };
    const dx = x - this.t.c, dy = y - this.t.f;
    return { u: this.tinv.a * dx + this.tinv.b * dy,
             v: this.tinv.d * dx + this.tinv.e * dy };
  }
  toSrcPixel(x, y) {
    const s = this.si;
    return { col: s[0] * x + s[1] * y + s[2], row: s[3] * x + s[4] * y + s[5] };
  }
  /* Bilinear lon/lat from the embedded mesh. */
  toLonLat(u, v) {
    const g = this.spec.grid, n = g.n;
    const gu = u / this.spec.width * (n - 1);
    const gv = v / this.spec.height * (n - 1);
    const i = clamp(Math.floor(gu), 0, n - 2), j = clamp(Math.floor(gv), 0, n - 2);
    const fu = gu - i, fv = gv - j;
    const k = (jj, ii) => jj * n + ii;
    const bl = (arr) =>
      arr[k(j, i)] * (1 - fu) * (1 - fv) + arr[k(j, i + 1)] * fu * (1 - fv) +
      arr[k(j + 1, i)] * (1 - fu) * fv + arr[k(j + 1, i + 1)] * fu * fv;
    return { lon: bl(g.lon), lat: bl(g.lat) };
  }
  fromLonLat(lon, lat) {
    let u = this.spec.width / 2, v = this.spec.height / 2;
    for (let it = 0; it < 12; it++) {
      const c0 = this.toLonLat(u, v);
      const dlon = lon - c0.lon, dlat = lat - c0.lat;
      const h = Math.max(1, this.spec.width / 64);
      const cu = this.toLonLat(u + h, v), cv = this.toLonLat(u, v + h);
      const a = (cu.lon - c0.lon) / h, b = (cv.lon - c0.lon) / h;
      const c = (cu.lat - c0.lat) / h, d = (cv.lat - c0.lat) / h;
      const det = a * d - b * c;
      if (!det) break;
      const du = (d * dlon - b * dlat) / det;
      const dv = (-c * dlon + a * dlat) / det;
      u += du; v += dv;
      if (Math.abs(du) + Math.abs(dv) < 1e-7) break;
    }
    return { u, v };
  }
  valueAt(u, v) {
    if (!this.vals) return NaN;
    const i = Math.floor(u), j = Math.floor(v);
    if (i < 0 || j < 0 || i >= this.spec.width || j >= this.spec.height) return NaN;
    const q = this.vals[j * this.spec.width + i];
    if (q === 0) return NaN;
    return this.vlo + (q - 1) / 65534 * (this.vhi - this.vlo);
  }

  /* --- snap to the local peak, refined to sub-pixel --------------------- */
  snap(u, v, radius) {
    if (!this.vals) return { u, v, snapped: false };
    const W = this.spec.width, H = this.spec.height;
    let bi = Math.round(u - 0.5), bj = Math.round(v - 0.5), best = -Infinity;
    const ci = clamp(Math.round(u - 0.5), 0, W - 1), cj = clamp(Math.round(v - 0.5), 0, H - 1);
    for (let j = cj - radius; j <= cj + radius; j++) {
      if (j < 0 || j >= H) continue;
      for (let i = ci - radius; i <= ci + radius; i++) {
        if (i < 0 || i >= W) continue;
        const q = this.vals[j * W + i];
        if (q === 0) continue;
        if (q > best) { best = q; bi = i; bj = j; }
      }
    }
    if (!isFinite(best)) return { u, v, snapped: false };
    // Parabolic (3-point) refinement in each axis about the peak sample.
    const at = (i, j) => {
      if (i < 0 || j < 0 || i >= W || j >= H) return null;
      const q = this.vals[j * W + i];
      return q === 0 ? null : q;
    };
    const refine = (m, l, r) => {
      if (l === null || r === null) return 0;
      const den = (l - 2 * m + r);
      if (den === 0) return 0;
      return clamp(0.5 * (l - r) / den, -0.5, 0.5);
    };
    const m = at(bi, bj);
    const du = refine(m, at(bi - 1, bj), at(bi + 1, bj));
    const dv = refine(m, at(bi, bj - 1), at(bi, bj + 1));
    return { u: bi + 0.5 + du, v: bj + 0.5 + dv, snapped: true };
  }

  /* --- DOM / rendering -------------------------------------------------- */
  build() {
    const s = this.spec;
    this.el = document.createElement("div");
    this.el.className = "pane";
    this.el.innerHTML =
      `<canvas></canvas>
       <div class="cap"><b>${this.name} &middot; ${escapeHtml(s.label)}</b>
       <span class="meta">${s.width}×${s.height} px &middot; ${s.px_x_m.toFixed(2)} m/px`
       + (s.dec > 1 ? ` &middot; dec ${s.dec}×` : "")
       + ` &middot; ${s.epsg ? "EPSG:" + s.epsg : escapeHtml(s.crs)}</span></div>
       <div class="hud"></div>`;
    this.canvas = this.el.querySelector("canvas");
    this.hud = this.el.querySelector(".hud");
    this.ctx = this.canvas.getContext("2d");

    this.base = document.createElement("canvas");
    this.base.width = s.width; this.base.height = s.height;
    this.toned = document.createElement("canvas");
    this.toned.width = s.width; this.toned.height = s.height;

    this.img = new Image();
    this.img.onload = () => {
      this.base.getContext("2d").drawImage(this.img, 0, 0);
      this.baseData = this.base.getContext("2d")
        .getImageData(0, 0, s.width, s.height);
      this.applyTone();
      this.fit();
      draw();
    };
    this.img.src = s.png;
  }

  applyTone() {
    if (!this.baseData) return;
    const src = this.baseData.data;
    const out = new ImageData(this.spec.width, this.spec.height);
    const dst = out.data;
    const bright = +$("#rBright").value / 100;          // -1 .. 1
    const contrast = +$("#rContrast").value / 100;      // -1 .. 1
    const gamma = +$("#rGamma").value / 100;            // 0.2 .. 3
    const inv = $("#bInv").classList.contains("on");
    const k = Math.tan((contrast + 1) * Math.PI / 4);   // 0..inf, 1 at centre
    const lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) {
      let x = i / 255;
      x = Math.pow(clamp(x, 0, 1), 1 / gamma);
      x = (x - 0.5) * k + 0.5 + bright;
      if (inv) x = 1 - x;
      lut[i] = clamp(x, 0, 1) * 255;
    }
    for (let i = 0; i < src.length; i += 4) {
      dst[i] = lut[src[i]]; dst[i + 1] = lut[src[i + 1]];
      dst[i + 2] = lut[src[i + 2]]; dst[i + 3] = src[i + 3];
    }
    this.toned.getContext("2d").putImageData(out, 0, 0);
  }

  fit() {
    const r = this.canvas.getBoundingClientRect();
    const s = Math.min(r.width / this.spec.width,
                       r.height / (this.spec.height * this.ar)) * 0.97;
    this.view.scale = s || 1;
    this.view.tx = (r.width - this.spec.width * this.view.scale) / 2;
    this.view.ty = (r.height - this.spec.height * this.view.scale * this.ar) / 2;
    this.view.fitted = true;
  }

  get scaleY() { return this.view.scale * this.ar; }

  screenToChip(sx, sy) {
    return { u: (sx - this.view.tx) / this.view.scale,
             v: (sy - this.view.ty) / this.scaleY };
  }
  chipToScreen(u, v) {
    return { x: u * this.view.scale + this.view.tx,
             y: v * this.scaleY + this.view.ty };
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ------------------------------------------------------------------ state */
const PANELS = DATA.panels.map((p, i) => new Panel(p, i));
const stage = $("#stage");
PANELS.forEach(p => stage.appendChild(p.el));
let active = 0;
let points = [];
let nextId = 1;
let selected = null;
const flags = { grid: false, snap: true, link: PANELS.length > 1, truth: DATA.overlay.length > 0 };

$("#bSnap").classList.toggle("on", flags.snap);
$("#bLink").classList.toggle("on", flags.link);
$("#bTruth").classList.toggle("on", flags.truth);
if (PANELS.length < 2) $("#bLink").disabled = true;
if (!DATA.overlay.length) $("#bTruth").disabled = true;
PANELS[0].el.classList.add("active");

/* -------------------------------------------------------------- rendering */
function resize() {
  const dpr = window.devicePixelRatio || 1;
  for (const p of PANELS) {
    const r = p.canvas.getBoundingClientRect();
    p.canvas.width = Math.max(1, Math.round(r.width * dpr));
    p.canvas.height = Math.max(1, Math.round(r.height * dpr));
    p.dpr = dpr;
    if (!p.view.fitted) p.fit();
  }
  draw();
}

function niceStep(targetM) {
  const e = Math.pow(10, Math.floor(Math.log10(targetM)));
  const n = targetM / e;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * e;
}

function draw() {
  for (const p of PANELS) drawPanel(p);
}

function drawPanel(p) {
  const ctx = p.ctx, dpr = p.dpr || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = p.canvas.width / dpr, H = p.canvas.height / dpr;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  if (!p.baseData) return;

  ctx.imageSmoothingEnabled = p.view.scale < 1;
  ctx.drawImage(p.toned, p.view.tx, p.view.ty,
                p.spec.width * p.view.scale, p.spec.height * p.scaleY);

  if (flags.grid) drawGrid(p, ctx, W, H);
  if (flags.truth) drawTruth(p, ctx);
  drawPoints(p, ctx);
  drawScaleBar(p, ctx, W, H);
}

function drawGrid(p, ctx, W, H) {
  // Graticule on round map-coordinate values, stepped in the CRS's own units
  // so this works for a projected CRS (metres) and a geographic one (degrees)
  // alike. Roughly five lines cross the view at any zoom.
  const unitsPerScreenPx = Math.abs(p.t.a) / p.view.scale;
  const step = niceStep(unitsPerScreenPx * Math.min(W, H) / 5);
  const dec = Math.max(0, Math.ceil(-Math.log10(step)) + 1);
  const tl = p.screenToChip(0, 0), br = p.screenToChip(W, H);
  const m0 = p.toMap(tl.u, tl.v), m1 = p.toMap(br.u, br.v);

  ctx.save();
  ctx.strokeStyle = "#58a6ff55";
  ctx.fillStyle = "#8ec7ff";
  ctx.lineWidth = 1;
  ctx.font = "11px ui-monospace, monospace";

  const x0 = Math.ceil(Math.min(m0.x, m1.x) / step) * step;
  for (let x = x0; x <= Math.max(m0.x, m1.x); x += step) {
    const c = p.fromMap(x, m0.y), s = p.chipToScreen(c.u, c.v);
    ctx.beginPath(); ctx.moveTo(s.x, 0); ctx.lineTo(s.x, H); ctx.stroke();
    ctx.fillText(x.toFixed(dec), s.x + 3, 34);   // below the caption bar
  }
  const y0 = Math.ceil(Math.min(m0.y, m1.y) / step) * step;
  for (let y = y0; y <= Math.max(m0.y, m1.y); y += step) {
    const c = p.fromMap(m0.x, y), s = p.chipToScreen(c.u, c.v);
    ctx.beginPath(); ctx.moveTo(0, s.y); ctx.lineTo(W, s.y); ctx.stroke();
    ctx.fillText(y.toFixed(dec), 3, s.y - 3);
  }
  ctx.restore();
}

function drawScaleBar(p, ctx, W, H) {
  const mPerPx = p.spec.px_x_m / p.view.scale;
  let len = niceStep(mPerPx * 110);
  const px = len / mPerPx;
  if (!isFinite(px) || px < 8) return;
  const x = W - px - 14, y = H - 16;
  ctx.save();
  ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y - 5); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 5);
  ctx.stroke();
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
  const lbl = len >= 1000 ? (len / 1000) + " km" : len + " m";
  ctx.strokeText(lbl, x + px / 2, y - 8);
  ctx.fillText(lbl, x + px / 2, y - 8);
  ctx.restore();
}

function drawTruth(p, ctx) {
  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  for (const o of DATA.overlay) {
    const xy = o.xy ? o.xy[p.index] : [o.x, o.y];
    if (!xy) continue;
    const c = p.fromMap(xy[0], xy[1]);
    if (c.u < -50 || c.v < -50 || c.u > p.spec.width + 50 || c.v > p.spec.height + 50) continue;
    const s = p.chipToScreen(c.u, c.v);
    ctx.strokeStyle = "#ff7b72"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(s.x - 9, s.y); ctx.lineTo(s.x - 3, s.y);
    ctx.moveTo(s.x + 3, s.y); ctx.lineTo(s.x + 9, s.y);
    ctx.moveTo(s.x, s.y - 9); ctx.lineTo(s.x, s.y - 3);
    ctx.moveTo(s.x, s.y + 3); ctx.lineTo(s.x, s.y + 9);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(s.x, s.y, 11, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#000"; ctx.fillRect(s.x + 12, s.y - 15, ctx.measureText(o.name).width + 6, 14);
    ctx.fillStyle = "#ff9d97"; ctx.fillText(o.name, s.x + 15, s.y - 4);
  }
  ctx.restore();
}

function drawPoints(p, ctx) {
  ctx.save();
  ctx.font = "11px ui-monospace, monospace";
  for (const pt of points) {
    if (pt.panel !== p.index) continue;
    const s = p.chipToScreen(pt.u, pt.v);
    const on = selected === pt.id;
    ctx.strokeStyle = on ? "#ffffff" : "#ffd33d";
    ctx.lineWidth = on ? 2 : 1.5;
    ctx.beginPath();
    ctx.moveTo(s.x - 8, s.y); ctx.lineTo(s.x + 8, s.y);
    ctx.moveTo(s.x, s.y - 8); ctx.lineTo(s.x, s.y + 8);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(s.x, s.y, 4, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.fillRect(s.x + 9, s.y + 2, ctx.measureText(pt.name).width + 6, 14);
    ctx.fillStyle = "#ffd33d";
    ctx.fillText(pt.name, s.x + 12, s.y + 13);
  }
  ctx.restore();
}

/* ----------------------------------------------------------- interaction */
function panelAt(target) {
  return PANELS.find(p => p.el.contains(target));
}

function localPos(p, ev) {
  const r = p.canvas.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

function hitPoint(p, sx, sy) {
  for (const pt of points) {
    if (pt.panel !== p.index) continue;
    const s = p.chipToScreen(pt.u, pt.v);
    if (Math.hypot(s.x - sx, s.y - sy) <= 9) return pt;
  }
  return null;
}

let drag = null;

for (const p of PANELS) {
  p.canvas.addEventListener("mousedown", ev => {
    setActive(p.index);
    const q = localPos(p, ev);
    const hit = hitPoint(p, q.x, q.y);
    if (hit && ev.altKey) {
      points = points.filter(x => x.id !== hit.id);
      refresh(); ev.preventDefault(); return;
    }
    if (hit) {
      selected = hit.id;
      drag = { kind: "point", panel: p, pt: hit, moved: false };
      refresh(); ev.preventDefault(); return;
    }
    drag = { kind: "pan", panel: p, x: q.x, y: q.y,
             tx: p.view.tx, ty: p.view.ty, moved: false,
             linkStart: PANELS.map(o => ({ tx: o.view.tx, ty: o.view.ty })) };
    ev.preventDefault();
  });

  p.canvas.addEventListener("mousemove", ev => {
    const q = localPos(p, ev);
    if (drag && drag.panel === p) {
      if (drag.kind === "pan") {
        const dx = q.x - drag.x, dy = q.y - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
        p.view.tx = drag.tx + dx; p.view.ty = drag.ty + dy;
        if (flags.link) syncFrom(p);
        draw();
      } else if (drag.kind === "point") {
        drag.moved = true;
        const c = p.screenToChip(q.x, q.y);
        setPointAt(drag.pt, p, c.u, c.v, false);
        draw(); renderTables();
      }
      return;
    }
    updateHud(p, q.x, q.y);
  });

  p.canvas.addEventListener("mouseleave", () => { p.hud.textContent = ""; });

  p.canvas.addEventListener("mouseup", ev => {
    const wasDrag = drag;
    drag = null;
    if (!wasDrag || wasDrag.panel !== p) return;
    if (wasDrag.moved) { refresh(); return; }
    if (wasDrag.kind === "point") return;      // plain click on an existing point
    const q = localPos(p, ev);
    addPoint(p, q.x, q.y);
  });

  p.canvas.addEventListener("wheel", ev => {
    ev.preventDefault();
    const q = localPos(p, ev);
    zoomAt(p, q.x, q.y, Math.pow(1.0015, -ev.deltaY));
  }, { passive: false });
}

function zoomAt(p, sx, sy, factor) {
  const before = p.screenToChip(sx, sy);
  p.view.scale = clamp(p.view.scale * factor, 0.02, 400);
  p.view.tx = sx - before.u * p.view.scale;
  p.view.ty = sy - before.v * p.scaleY;
  if (flags.link) syncFrom(p);
  draw();
}

/* Keep the other panel showing the same ground footprint, not the same pixels:
   the two products have different grids and often different pixel spacing. */
function syncFrom(src) {
  const rs = src.canvas.getBoundingClientRect();
  const cCenter = src.screenToChip(rs.width / 2, rs.height / 2);
  const ll = src.toLonLat(cCenter.u, cCenter.v);
  const mPerScreenPx = src.spec.px_x_m / src.view.scale;
  for (const p of PANELS) {
    if (p === src) continue;
    const r = p.canvas.getBoundingClientRect();
    p.view.scale = clamp(p.spec.px_x_m / mPerScreenPx, 0.02, 400);
    const c = p.fromLonLat(ll.lon, ll.lat);
    p.view.tx = r.width / 2 - c.u * p.view.scale;
    p.view.ty = r.height / 2 - c.v * p.scaleY;
  }
}

function setActive(i) {
  active = i;
  PANELS.forEach((p, k) => p.el.classList.toggle("active", k === i));
}

function describe(p, u, v) {
  const m = p.toMap(u, v);
  const ll = p.toLonLat(u, v);
  const sp = p.toSrcPixel(m.x, m.y);
  return { u, v, x: m.x, y: m.y, lon: ll.lon, lat: ll.lat,
           srcCol: sp.col, srcRow: sp.row, val: p.valueAt(u, v) };
}

function updateHud(p, sx, sy) {
  const c = p.screenToChip(sx, sy);
  if (c.u < 0 || c.v < 0 || c.u >= p.spec.width || c.v >= p.spec.height) {
    p.hud.textContent = "(outside chip)";
    return;
  }
  const d = describe(p, c.u, c.v);
  const u = p.spec.stretch.units || "";
  p.hud.textContent =
    `lat ${d.lat.toFixed(7)}   ${dms(d.lat, true)}\n` +
    `lon ${d.lon.toFixed(7)}   ${dms(d.lon, false)}\n` +
    `X   ${d.x.toFixed(3)}   Y ${d.y.toFixed(3)}\n` +
    `src px  col ${d.srcCol.toFixed(1)}  row ${d.srcRow.toFixed(1)}\n` +
    `value   ${isFinite(d.val) ? (d.val.toFixed(2) + (u ? " " + u : "")) : "nodata"}`;
}

function setPointAt(pt, p, u, v, useSnap) {
  if (useSnap && flags.snap) {
    const s = p.snap(u, v, DATA.snapRadius);
    u = s.u; v = s.v; pt.snapped = s.snapped;
  } else if (useSnap) {
    pt.snapped = false;
  }
  Object.assign(pt, describe(p, u, v));
}

function addPoint(p, sx, sy) {
  const c = p.screenToChip(sx, sy);
  if (c.u < 0 || c.v < 0 || c.u >= p.spec.width || c.v >= p.spec.height) return;
  // Auto-name so that the n-th point in panel A pairs with the n-th in panel B.
  let maxN = 0;
  for (const x of points) {
    if (x.panel !== p.index) continue;
    const m = /^P(\d+)$/.exec(x.name);
    if (m) maxN = Math.max(maxN, +m[1]);
  }
  const pt = { id: nextId++, panel: p.index, name: "P" + (maxN + 1), snapped: false };
  setPointAt(pt, p, c.u, c.v, true);
  points.push(pt);
  selected = pt.id;
  refresh();
}

/* ------------------------------------------------------------------ stats */
function stats(vals) {
  const n = vals.length;
  if (!n) return null;
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  const varr = n > 1 ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : 0;
  return { n, mean, sd: Math.sqrt(varr) };
}

function accuracySummary(rows) {
  if (!rows.length) return null;
  const de = rows.map(r => r.dx), dn = rows.map(r => r.dy);
  const se = stats(de), sn = stats(dn);
  const rad = rows.map(r => Math.hypot(r.dx, r.dy)).sort((a, b) => a - b);
  const rmse = Math.sqrt(rows.reduce((a, r) => a + r.dx * r.dx + r.dy * r.dy, 0) / rows.length);
  const idx = 0.9 * (rad.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  const ce90emp = rad[lo] + (rad[hi] - rad[lo]) * (idx - lo);
  const sigma = Math.sqrt((se.sd ** 2 + sn.sd ** 2) / 2);
  return { n: rows.length, biasE: se.mean, biasN: sn.mean, sdE: se.sd, sdN: sn.sd,
           rmse, ce90emp, ce90par: 2.146 * sigma,
           rmseDebiased: Math.sqrt(rows.reduce((a, r) =>
             a + (r.dx - se.mean) ** 2 + (r.dy - sn.mean) ** 2, 0) / rows.length) };
}

/* Panel A picks vs. surveyed overlay points -> absolute location error. */
function aleRows() {
  const out = [];
  if (!DATA.overlay.length) return out;
  for (const pt of points) {
    let best = null, bestD = Infinity, bestXy = null;
    for (const o of DATA.overlay) {
      const xy = o.xy ? o.xy[pt.panel] : [o.x, o.y];
      if (!xy) continue;
      const d = Math.hypot(pt.x - xy[0], pt.y - xy[1]);
      if (d < bestD) { bestD = d; best = o; bestXy = xy; }
    }
    if (!best || bestD > DATA.assocRadius) continue;
    out.push({ kind: "ale", label: pt.name, ref: best.name, panel: PANELS[pt.panel].name,
               px: PANELS[pt.panel].spec.px_x_m,
               dx: pt.x - bestXy[0], dy: pt.y - bestXy[1], pt, ref_o: best });
  }
  return out;
}

/* Same-named points in panel A and panel B -> image-to-image offset. */
function pairRows() {
  const out = [];
  if (PANELS.length < 2) return out;
  const byName = new Map();
  for (const pt of points) {
    if (!byName.has(pt.name)) byName.set(pt.name, {});
    byName.get(pt.name)[pt.panel] = pt;
  }
  const sameCrs = PANELS[0].spec.crs === PANELS[1].spec.crs;
  for (const [name, g] of byName) {
    if (!g[0] || !g[1]) continue;
    let dx, dy;
    if (sameCrs) {
      // The usual case, and exact: subtract in the shared projection.
      dx = g[0].x - g[1].x;
      dy = g[0].y - g[1].y;
    } else {
      // Different projections -- differencing the raw map coordinates would be
      // meaningless, so difference in lon/lat and scale to metres at this
      // latitude. Good to centimetres over the few-metre offsets at issue.
      const phi = (g[0].lat + g[1].lat) / 2 * Math.PI / 180;
      const mPerDegLat = 111132.92 - 559.82 * Math.cos(2 * phi)
                       + 1.175 * Math.cos(4 * phi) - 0.0023 * Math.cos(6 * phi);
      const mPerDegLon = 111412.84 * Math.cos(phi) - 93.5 * Math.cos(3 * phi)
                       + 0.118 * Math.cos(5 * phi);
      dx = (g[0].lon - g[1].lon) * mPerDegLon;
      dy = (g[0].lat - g[1].lat) * mPerDegLat;
    }
    out.push({ kind: "pair", label: name, ref: PANELS[1].name,
               px: PANELS[0].spec.px_x_m, dx, dy, a: g[0], b: g[1] });
  }
  return out;
}

function activeRows() {
  const ale = aleRows();
  return ale.length ? ale : pairRows();
}

/* ------------------------------------------------------------------ tables */
function renderTables() {
  renderPoints();
  renderAccuracy();
}

function renderPoints() {
  const t = $("#tPoints");
  const u = PANELS[0].spec.stretch.units || "";
  let h = `<thead><tr><th class="l">pane</th><th class="l">name</th>
    <th>latitude</th><th>longitude</th><th>X</th><th>Y</th>
    <th>src col</th><th>src row</th><th>value${u ? " (" + escapeHtml(u) + ")" : ""}</th>
    <th class="l">snap</th></tr></thead><tbody>`;
  for (const pt of points) {
    h += `<tr data-id="${pt.id}" class="${selected === pt.id ? "sel" : ""}">
      <td class="l">${PANELS[pt.panel].name}</td>
      <td class="l"><input type="text" data-name="${pt.id}" value="${escapeHtml(pt.name)}"></td>
      <td>${pt.lat.toFixed(7)}</td><td>${pt.lon.toFixed(7)}</td>
      <td>${pt.x.toFixed(3)}</td><td>${pt.y.toFixed(3)}</td>
      <td>${pt.srcCol.toFixed(2)}</td><td>${pt.srcRow.toFixed(2)}</td>
      <td>${isFinite(pt.val) ? pt.val.toFixed(2) : "--"}</td>
      <td class="l">${pt.snapped ? "peak" : "manual"}</td></tr>`;
  }
  t.innerHTML = h + "</tbody>";
  t.querySelectorAll("tr[data-id]").forEach(tr => {
    tr.addEventListener("click", ev => {
      if (ev.target.tagName === "INPUT") return;
      selected = +tr.dataset.id; refresh();
    });
  });
  t.querySelectorAll("input[data-name]").forEach(inp => {
    inp.addEventListener("change", () => {
      const pt = points.find(x => x.id === +inp.dataset.name);
      if (pt) { pt.name = inp.value.trim() || pt.name; refresh(); }
    });
  });

  const nA = points.filter(p => p.panel === 0).length;
  const nB = points.filter(p => p.panel === 1).length;
  $("#pointsHint").textContent = PANELS.length > 1
    ? `${nA} in A, ${nB} in B — points sharing a name are paired on the Accuracy tab.`
    : `${nA} point${nA === 1 ? "" : "s"}.`;
}

function renderAccuracy() {
  const rows = activeRows();
  const isAle = rows.length && rows[0].kind === "ale";
  const head = $("#aleHead"), cards = $("#aleCards"), t = $("#tAle");

  head.innerHTML = isAle
    ? `<p class="note"><b>Absolute location error</b> — each picked point against the
       nearest surveyed point within ${DATA.assocRadius} m.
       <code>&Delta;X = image − survey</code>; positive means the feature images
       east / north of where it really is.</p>`
    : (PANELS.length > 1
        ? `<p class="note"><b>Image-to-image offset</b> — points sharing a name in
           panels A and B. <code>&Delta;X = A − B</code>, i.e.
           <code>across</code>, and <code>&Delta;Y</code> is <code>along</code>,
           matching the dqe_imw CSV convention.</p>`
        : `<p class="note">Load surveyed points with <code>--overlay points.csv</code>,
           or add a reference raster with <code>--b &lt;uri&gt;</code>, to measure
           accuracy here. Without either, the Points tab still gives you the
           coordinates of everything you click.</p>`);

  if (!rows.length) {
    cards.innerHTML = ""; t.innerHTML = "";
    $("#aleNote").textContent = "";
    return;
  }

  const px = PANELS[0].spec.px_x_m;
  const s = accuracySummary(rows);
  const card = (k, v) => `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  cards.innerHTML =
    card("points", s.n) +
    card("bias ΔX", fmt(s.biasE, 2) + " m") +
    card("bias ΔY", fmt(s.biasN, 2) + " m") +
    card("σ X", fmt(s.sdE, 2) + " m") +
    card("σ Y", fmt(s.sdN, 2) + " m") +
    card("RMSE (2D)", fmt(s.rmse, 2) + " m") +
    card("RMSE debiased", fmt(s.rmseDebiased, 2) + " m") +
    card("CE90 empirical", fmt(s.ce90emp, 2) + " m") +
    card("CE90 ≈ 2.146σ", fmt(s.ce90par, 2) + " m") +
    card("bias in pixels", fmt(Math.hypot(s.biasE, s.biasN) / px, 2) + " px");

  let h = `<thead><tr><th class="l">point</th><th class="l">${isAle ? "survey" : "pane"}</th>
    <th>&Delta;X across (m)</th><th>&Delta;Y along (m)</th><th>distance (m)</th>
    <th>distance (px)</th></tr></thead><tbody>`;
  for (const r of rows) {
    const d = Math.hypot(r.dx, r.dy);
    h += `<tr><td class="l">${escapeHtml(r.label)}</td>
      <td class="l">${escapeHtml(r.ref)}</td>
      <td>${r.dx.toFixed(2)}</td><td>${r.dy.toFixed(2)}</td>
      <td>${d.toFixed(2)}</td><td>${(d / (r.px || px)).toFixed(2)}</td></tr>`;
  }
  t.innerHTML = h + "</tbody>";

  $("#aleNote").textContent =
    "CE90 empirical is the 90th percentile of the radial errors, which needs a fair " +
    "number of points to mean much; the 2.146σ form assumes a circular normal " +
    "error and is the more stable estimate for small samples. A large bias with a " +
    "small σ is a systematic geolocation shift, which is the interesting case: " +
    "it points at timing, geometry or DEM height rather than at the imagery.";
}

/* ------------------------------------------------------------------- CSV */
function pointsCsv() {
  const u = PANELS[0].spec.stretch.units || "";
  let s = `panel,name,latitude,longitude,map_x,map_y,crs,src_col,src_row,` +
          `chip_u,chip_v,value_${u || "raw"},snapped,source\n`;
  for (const pt of points) {
    const p = PANELS[pt.panel];
    s += [p.name, pt.name, pt.lat.toFixed(9), pt.lon.toFixed(9),
          pt.x.toFixed(4), pt.y.toFixed(4), p.spec.epsg ? "EPSG:" + p.spec.epsg : p.spec.crs,
          pt.srcCol.toFixed(3), pt.srcRow.toFixed(3),
          pt.u.toFixed(3), pt.v.toFixed(3),
          isFinite(pt.val) ? pt.val.toFixed(4) : "",
          pt.snapped ? "peak" : "manual", p.spec.uri].join(",") + "\n";
  }
  return s;
}

function accuracyCsv() {
  const rows = activeRows();
  const px = PANELS[0].spec.px_x_m;
  let s = "kind,point,reference,across_dx_m,along_dy_m,distance_m,distance_px\n";
  for (const r of rows) {
    const d = Math.hypot(r.dx, r.dy);
    s += [r.kind, r.label, r.ref, r.dx.toFixed(4), r.dy.toFixed(4),
          d.toFixed(4), (d / (r.px || px)).toFixed(4)].join(",") + "\n";
  }
  const su = accuracySummary(rows);
  if (su) {
    s += "\n# summary\n";
    s += `# n,${su.n}\n# bias_across_m,${su.biasE.toFixed(4)}\n`;
    s += `# bias_along_m,${su.biasN.toFixed(4)}\n# sd_across_m,${su.sdE.toFixed(4)}\n`;
    s += `# sd_along_m,${su.sdN.toFixed(4)}\n# rmse_2d_m,${su.rmse.toFixed(4)}\n`;
    s += `# rmse_debiased_m,${su.rmseDebiased.toFixed(4)}\n`;
    s += `# ce90_empirical_m,${su.ce90emp.toFixed(4)}\n`;
    s += `# ce90_parametric_m,${su.ce90par.toFixed(4)}\n`;
  }
  return s;
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* --------------------------------------------------------------- controls */
function refresh() { draw(); renderTables(); }

$("#bFit").onclick = () => { PANELS.forEach(p => p.fit()); draw(); };
$("#bOne").onclick = () => {
  const p = PANELS[active];
  const r = p.canvas.getBoundingClientRect();
  const c = p.screenToChip(r.width / 2, r.height / 2);
  p.view.scale = 1;
  p.view.tx = r.width / 2 - c.u; p.view.ty = r.height / 2 - c.v * p.ar;
  if (flags.link) syncFrom(p);
  draw();
};
$("#bGrid").onclick = () => { flags.grid = !flags.grid; $("#bGrid").classList.toggle("on", flags.grid); draw(); };
$("#bSnap").onclick = () => { flags.snap = !flags.snap; $("#bSnap").classList.toggle("on", flags.snap); };
$("#bTruth").onclick = () => { flags.truth = !flags.truth; $("#bTruth").classList.toggle("on", flags.truth); draw(); };
$("#bLink").onclick = () => {
  flags.link = !flags.link;
  $("#bLink").classList.toggle("on", flags.link);
  if (flags.link) { syncFrom(PANELS[active]); draw(); }
};
$("#bInv").onclick = () => {
  $("#bInv").classList.toggle("on");
  PANELS.forEach(p => p.applyTone()); draw();
};
$("#bReset").onclick = () => {
  $("#rBright").value = 0; $("#rContrast").value = 0; $("#rGamma").value = 100;
  $("#bInv").classList.remove("on");
  PANELS.forEach(p => p.applyTone()); draw();
};
let toneTimer = null;
for (const id of ["#rBright", "#rContrast", "#rGamma"]) {
  $(id).addEventListener("input", () => {
    if (toneTimer) cancelAnimationFrame(toneTimer);
    toneTimer = requestAnimationFrame(() => {
      PANELS.forEach(p => p.applyTone()); draw();
    });
  });
}

$("#bClear").onclick = () => { points = []; selected = null; refresh(); };
$("#bDelSel").onclick = () => { points = points.filter(p => p.id !== selected); selected = null; refresh(); };
$("#bCopy").onclick = async () => {
  const txt = pointsCsv();
  try {
    await navigator.clipboard.writeText(txt);
    $("#bCopy").textContent = "Copied";
  } catch (e) {
    // Clipboard access is often blocked for file:// pages; fall back to a
    // window the user can select from.
    const w = window.open("", "_blank");
    w.document.write("<pre>" + escapeHtml(txt) + "</pre>");
    $("#bCopy").textContent = "Opened in tab";
  }
  setTimeout(() => { $("#bCopy").textContent = "Copy CSV"; }, 1600);
};
$("#bDown").onclick = () => download("cog_locate_points.csv", pointsCsv());
$("#bDownAle").onclick = () => download("cog_locate_accuracy.csv", accuracyCsv());

document.querySelectorAll("#tabs button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(x => x.classList.remove("on"));
    document.querySelectorAll("section.tab").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    $("#tab-" + b.dataset.tab).classList.add("on");
  };
});

window.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const p = PANELS[active];
  const step = ev.shiftKey ? 120 : 30;
  switch (ev.key) {
    case "f": $("#bFit").click(); break;
    case "1": $("#bOne").click(); break;
    case "g": $("#bGrid").click(); break;
    case "s": $("#bSnap").click(); break;
    case "l": if (!$("#bLink").disabled) $("#bLink").click(); break;
    case "t": if (!$("#bTruth").disabled) $("#bTruth").click(); break;
    case "i": $("#bInv").click(); break;
    case "+": case "=": { const r = p.canvas.getBoundingClientRect();
      zoomAt(p, r.width / 2, r.height / 2, 1.25); break; }
    case "-": { const r = p.canvas.getBoundingClientRect();
      zoomAt(p, r.width / 2, r.height / 2, 0.8); break; }
    case "ArrowLeft": p.view.tx += step; if (flags.link) syncFrom(p); draw(); break;
    case "ArrowRight": p.view.tx -= step; if (flags.link) syncFrom(p); draw(); break;
    case "ArrowUp": p.view.ty += step; if (flags.link) syncFrom(p); draw(); break;
    case "ArrowDown": p.view.ty -= step; if (flags.link) syncFrom(p); draw(); break;
    case "Delete": case "Backspace": $("#bDelSel").click(); break;
    case "Escape": selected = null; refresh(); break;
    default: return;
  }
  ev.preventDefault();
});

/* ------------------------------------------------------------- scene info */
/* Bearing of the chip's +u axis, in degrees clockwise from true north. */
function gridBearing(p) {
  const a = p.toLonLat(0, p.spec.height / 2);
  const b = p.toLonLat(p.spec.width, p.spec.height / 2);
  const phi = (a.lat + b.lat) / 2 * Math.PI / 180;
  const de = (b.lon - a.lon) * Math.cos(phi);
  const dn = (b.lat - a.lat);
  return (Math.atan2(de, dn) * 180 / Math.PI + 360) % 360;
}

function renderMeta() {
  let h = "";
  for (const p of PANELS) {
    const s = p.spec, st = s.stretch, w = s.window;
    h += `<h3>Panel ${p.name} — ${escapeHtml(s.label)}</h3>
      <table><tbody>
      <tr><td class="l">source</td><td class="l">${escapeHtml(s.uri)} (band ${s.band})</td></tr>
      <tr><td class="l">source raster</td><td class="l">${s.src.src_width || "?"} × ${s.src.src_height || "?"} px, ${escapeHtml(String(s.src.src_dtype || "?"))}</td></tr>
      <tr><td class="l">overviews</td><td class="l">${(s.src.overviews && s.src.overviews.length) ? s.src.overviews.join(", ") : "(none)"}</td></tr>
      <tr><td class="l">window read (full-res px)</td><td class="l">col ${w[0]}, row ${w[1]}, ${w[2]} × ${w[3]}</td></tr>
      <tr><td class="l">decimation</td><td class="l">${s.dec}×</td></tr>
      <tr><td class="l">chip pixel size</td><td class="l">${s.px_x_m.toFixed(4)} × ${s.px_y_m.toFixed(4)} m</td></tr>
      <tr><td class="l">CRS</td><td class="l">${escapeHtml(s.crs)}</td></tr>
      <tr><td class="l">stretch</td><td class="l">${escapeHtml(st.mode)}, ${st.vmin.toFixed(2)} .. ${st.vmax.toFixed(2)} ${escapeHtml(st.units)} (${st.pct[0]}–${st.pct[1]} pct)</td></tr>
      <tr><td class="l">valid pixels</td><td class="l">${(st.valid_fraction * 100).toFixed(1)}%</td></tr>
      <tr><td class="l">lon/lat mesh error</td><td class="l">${isFinite(s.grid.residual_m) ? s.grid.residual_m.toExponential(2) + " m" : "n/a"}</td></tr>
      <tr><td class="l">values embedded</td><td class="l">${p.vals ? "yes (" + s.values.encoding + ")" : "no — dB readout and snap-to-peak disabled"}</td></tr>
      <tr><td class="l">grid orientation</td><td class="l">+X axis bears ${gridBearing(p).toFixed(3)}° from true north</td></tr>
      </tbody></table>`;
    if (s.dec > 1) {
      h += `<div class="warnbox">This panel is decimated ${s.dec}×, so a picked
        position is only good to about ${(s.px_x_m).toFixed(1)} m. For measuring
        location accuracy, re-run with a smaller <code>--size</code> (or a larger
        <code>--max-px</code>) so the chip is read at full resolution, and use
        <code>--resample nearest</code>.</div>`;
    }
  }
  if (PANELS.length > 1) {
    // 90 deg = +X points due east, the axis-aligned case.
    const d = (gridBearing(PANELS[1]) - gridBearing(PANELS[0]) + 540) % 360 - 180;
    if (Math.abs(d) > 0.05) {
      const drift = Math.abs(d) * Math.PI / 180 * 1000;
      h += `<div class="warnbox">The two panels sit on grids rotated
        ${d.toFixed(3)}° apart — normal when comparing a UTM product against a
        geographic one, where meridian convergence is the difference between
        grid north and true north. Linking matches the panels at the centre of
        the view, so features drift apart by roughly
        ${drift.toFixed(1)} m per km away from it. Your measurements are not
        affected: every picked point carries its own coordinates, and offsets
        between panels are computed in a common frame.</div>`;
    }
  }
  $("#metaBody").innerHTML = h;
}

/* -------------------------------------------------------------------- go */
window.addEventListener("resize", resize);
resize();
renderMeta();
renderTables();
</script>
</body>
</html>
"""
