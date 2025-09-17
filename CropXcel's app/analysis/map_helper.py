# PLACE INTO: CropXcel/analysis/map_helpers_optional.py

import os, json, base64
import numpy as np
import folium
from folium.plugins import Fullscreen, MeasureControl, MousePosition
from folium.raster_layers import ImageOverlay
import geopandas as gpd

# --- 0) Preconditions (asserts to catch silent bugs) ---
assert 'top_lat' in globals() and 'bot_lat' in globals() and 'left_lon' in globals() and 'right_lon' in globals(), \
    "Bounds variables (top_lat/bot_lat/left_lon/right_lon) are missing."
assert 'risk_web' in globals(), "risk_web (2D float array in [0,1]) is required."
assert 'OUT_PNG_NC' in globals() and os.path.exists(OUT_PNG_NC), "Overlay PNG path OUT_PNG_NC not found."

bounds = [[bot_lat, left_lon], [top_lat, right_lon]]  # south-west, north-east
center_lat = (top_lat + bot_lat) / 2.0
center_lon = (left_lon + right_lon) / 2.0

# --- 1) Map container ---
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    tiles=None,
    control_scale=True
)

# --- 2) Minimal CSS for hotspot popups (kept compact) ---
m.get_root().html.add_child(folium.Element("""
<style>
.hotspot-card{font:13px/1.35 system-ui,Segoe UI,Roboto,Arial,sans-serif;color:#0f172a}
.hotspot-hd{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.badge{padding:2px 8px;border-radius:999px;font-weight:700;font-size:11px;color:#fff;display:inline-block}
.badge.high{background:#e31a1c}.badge.caution{background:#ff7f00}.badge.low{background:#1f78b4}
.kv{margin:6px 0}.kv b{color:#0b3b3b}
.riskbar{height:8px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin:6px 0 2px}
.riskbar > span{display:block;height:100%}
.riskbar.high{background:#fde2e1}.riskbar.caution{background:#fff1da}.riskbar.low{background:#dbeafe}
.riskbar > span.high{background:#e31a1c}.riskbar > span.caution{background:#ff7f00}.riskbar > span.low{background:#1f78b4}
.tip{margin:6px 0 2px;color:#334155}
.action{margin-top:6px;padding:6px 8px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px}
.subttl{font-weight:700;margin:8px 0 4px}
.chartwrap{border:1px solid #e5e7eb;border-radius:8px;padding:6px;margin-top:4px}
.small{font-size:11px;color:#475569}
                                           /* brand wordmark */
.brand{ display:flex; align-items:center; gap:10px; }
.brand-mark{ filter: drop-shadow(0 2px 6px rgba(0,0,0,.18)); border-radius:12px; }
.brand-text{ display:flex; flex-direction:column; line-height:1; }
.brand-name{
  font-weight:800; font-size:18px; letter-spacing:.3px;
  background: linear-gradient(90deg,#34d399,#0ea5e9);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  text-shadow: 0 1px 1px rgba(0,0,0,.15);
}
.brand-sub{
  margin-top:2px; font-size:11px; color:#e2f3f3;
  padding:2px 6px; border-radius:999px;
  background: rgba(255,255,255,.12);
  backdrop-filter: blur(2px);
}
</style>
"""))

# --- 3) Basemaps ---
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite (Esri)",
    show=True,
    control=True
).add_to(m)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="OpenStreetMap (OSM)",
    show=False,
    control=True
).add_to(m)

# --- 4) Field boundary (optional) ---
fp = None
if 'FIELD_GEOJSON' in globals() and FIELD_GEOJSON and os.path.exists(FIELD_GEOJSON):
    fp = FIELD_GEOJSON
elif 'FIELD_SHP' in globals() and FIELD_SHP and os.path.exists(FIELD_SHP):
    fp = FIELD_SHP

if fp:
    try:
        field_gdf = gpd.read_file(fp).to_crs(epsg=4326)
        folium.GeoJson(
            field_gdf.__geo_interface__, name="Field boundary",
            style_function=lambda x: {"color":"#ffffff","weight":2.5,"opacity":0.9,"fillColor":"#000000","fillOpacity":0.0}
        ).add_to(m)
    except Exception as e:
        print(f"[WARN] Could not add field boundary: {e}")

# --- 5) Overlay (ImageOverlay with data URI) ---
def _png_to_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

image_src = _png_to_data_uri(OUT_PNG_NC)
# --- 5b) Overlay (ImageOverlay with data URI) ---
overlay = ImageOverlay(
    name="Waterlogging Risk (0–1)",
    image=image_src,      # default to native
    bounds=bounds,
    opacity=0.70,
    interactive=False,
    cross_origin=False,
    zindex=3
).add_to(m)

# --- 5a) Optional: alternate overlay resolutions (client-switchable) ---
from PIL import Image
import io

def _pil_to_data_uri(im: Image.Image) -> str:
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("utf-8")

# Open the native PNG once
_pil = Image.open(OUT_PNG_NC)
W, H = _pil.size

# Downsample visually (NEAREST keeps blocks crisp for “pixel” look)
_pil_med = _pil.resize((max(1, W//2), max(1, H//2)), resample=Image.NEAREST)
_pil_low = _pil.resize((max(1, W//4), max(1, H//4)), resample=Image.NEAREST)

# Data-URIs for each option
overlay_sources = {
    "Fine (native)": image_src,                 # already built above
    "Medium (1/2 px count)": _pil_to_data_uri(_pil_med),
    "Coarse (1/4 px count)": _pil_to_data_uri(_pil_low),
}

def _pil_to_data_uri(im: Image.Image) -> str:
    bio = io.BytesIO()
    im.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("utf-8")

# Open the native PNG once
_pil = Image.open(OUT_PNG_NC)
W, H = _pil.size

# Downsample visually (NEAREST keeps blocks crisp for “pixel” look)
_pil_med = _pil.resize((max(1, W//2), max(1, H//2)), resample=Image.NEAREST)
_pil_low = _pil.resize((max(1, W//4), max(1, H//4)), resample=Image.NEAREST)

# Data-URIs for each option
overlay_sources = {
    "Fine (native)": image_src,                 # already built above
    "Medium (1/2 px count)": _pil_to_data_uri(_pil_med),
    "Coarse (1/4 px count)": _pil_to_data_uri(_pil_low),
}

# --- 6) Hotspots (markers + styled popup) ---
LEVEL_COLOR = {
    "High":    "#FF00FF",   # magenta
    "Caution": "#00FFFF",   # cyan
    "Low":     "#FFFFFF"    # white (with black border)
}
LEVEL_KEY   = {"High":"high", "Caution":"caution", "Low":"low"}

TIP_MODE = "by_reason"   # or "universal"
UNIVERSAL_TIP = "Check for standing water or blocked drains after rainfall."
REASON_TIP = {
    "Radar drop (water)": "Likely standing water or saturated soil after rain / poor drainage.",
    "VH/VV ratio change": "Structure/moisture change (possible sub-canopy water, lodging, or irrigation).",
    "Signal variability": "Patchy signal; inspect low spots, wheel tracks, or uneven leveling."
}
def make_tip(reason): return UNIVERSAL_TIP if TIP_MODE == "universal" else REASON_TIP.get(reason, UNIVERSAL_TIP)
ACTION_BY_LEVEL = {
    "High":    "Act now: drain standing water (open outlets/pump), inspect in-field low spots.",
    "Caution": "Monitor in the next 1–2 days. Walk field edges; prepare drainage.",
    "Low":     "Routine checks only. No immediate action required."
}
def make_action(level): return ACTION_BY_LEVEL.get(level, ACTION_BY_LEVEL["Caution"])

def build_hotspot_popup(h):
    lvl_key  = LEVEL_KEY.get(h["level"], "caution")
    risk_pct = h.get("risk_pct", int(round(h["risk"]*100)))
    tip_txt  = make_tip(h.get("reason"))
    act_txt  = make_action(h["level"])
    return f"""
    <div class="hotspot-card">
      <div class="hotspot-hd">
        <span class="badge {lvl_key}">{h['level']}</span>
        <div><b>Waterlogging risk:</b> {h['risk']:.3f} <span class="small">({risk_pct}%)</span></div>
      </div>
      <div class="riskbar {lvl_key}"><span class="{lvl_key}" style="width:{risk_pct}%;"></span></div>
      <div class="kv"><b>Area:</b> {h['area_ha']:.3f} ha <span class="small">({h['pixels']} px)</span></div>
      <div class="tip"><b>Tip:</b> {tip_txt}</div>
      <div class="chartwrap">
        <div class="subttl">Why risky?</div>
        <img src="data:image/png;base64,{h['chart_b64']}" style="width:100%;display:block;border-radius:6px"/>
      </div>
      <div class="action"><b>Action:</b> {act_txt}</div>
    </div>
    """

hotspots = (result.get("hotspots", []) if 'result' in globals() and isinstance(result, dict) else [])
def _in_bounds(lat, lon, bds):
    (south, west), (north, east) = bds
    return (south <= lat <= north) and (west <= lon <= east)

skipped = 0
for h in hotspots:
    lat, lon = float(h["lat"]), float(h["lon"])
    if not _in_bounds(lat, lon, bounds):
        skipped += 1
        continue
    html = build_hotspot_popup(h)
    folium.CircleMarker(
        [lat, lon],
        radius=8,
        weight=2,
        color="#000000",  # black outline
        fill=True,
        fill_color=LEVEL_COLOR.get(h["level"], "#FF00FF"),
        fill_opacity=0.95
    ).add_child(folium.Popup(html, max_width=320)).add_to(m)

if skipped:
    print(f"[WARN] {skipped} hotspot(s) fell outside map bounds and were skipped.")

# --- 7) Controls ---
folium.LayerControl(collapsed=False).add_to(m)
Fullscreen().add_to(m)
MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
MousePosition(position="bottomright", separator=" | ", num_digits=5).add_to(m)

# --- 8) Hover-to-sample (single lightweight probe bound to THIS map) ---
from folium import Element
arr = np.clip(risk_web, 0, 1)  # ensure in [0,1]
rows, cols = arr.shape
data_u16 = (arr * 1000).round().astype(np.uint16).ravel().tolist()
mask_list = None
if 'aoi_mask_web' in globals() and aoi_mask_web is not None and aoi_mask_web.shape == arr.shape:
    mask_list = aoi_mask_web.astype(np.uint8).ravel().tolist()

(south, west), (north, east) = bounds
map_var = m.get_name()  # the actual JS map variable

hover_js = f"""
<script>
window.addEventListener('load', function(){{
  // obtain Folium's map variable from window safely
  const map = window[{json.dumps(map_var)}];
  if (!map) return;

  const rows = {rows}, cols = {cols};
  const south = {south}, west = {west}, north = {north}, east = {east};
  const data = new Uint16Array({json.dumps(data_u16)});
  const mask = { 'new Uint8Array(' + json.dumps(mask_list) + ')' if mask_list is not None else 'null' };

  function idx(lat, lng){{
    if (lat < south || lat > north || lng < west || lng > east) return -1;
    const r = Math.floor((north - lat) / (north - south) * rows);
    const c = Math.floor((lng  - west)  / (east - west)  * cols);
    if (r < 0 || r >= rows || c < 0 || c >= cols) return -1;
    return r*cols + c;
  }}

  const tip = L.tooltip({{permanent:false, direction:'top', opacity:0.96, offset:[0,-10]}});
  let last = -2, raf = null;

  map.on('mousemove', (e) => {{
    if (raf) return;
    raf = requestAnimationFrame(() => {{
      raf = null;
      const k = idx(e.latlng.lat, e.latlng.lng);
      if (k < 0 || (mask && mask[k] === 0)) {{
        if (map.hasLayer(tip)) map.removeLayer(tip);
        last = -2; return;
      }}
      if (k === last) {{ tip.setLatLng(e.latlng); if(!map.hasLayer(tip)) tip.addTo(map); return; }}
      last = k;
      const v = data[k]/1000.0, pct=(v*100).toFixed(1);
      tip.setLatLng(e.latlng).setContent(
        `<div style="font:13px/1.35 system-ui;min-width:210px">
           <div style="font-weight:700;margin-bottom:2px">Waterlogging risk</div>
           <div style="font-size:12px;color:#0f172a">Value: <b>${{pct}}%</b>
             <span style="opacity:.6">(${{v.toFixed(3)}})</span></div>
         </div>`
      ).addTo(map);
    }});
  }});
  map.on('mouseout', () => {{ if (map.hasLayer(tip)) map.removeLayer(tip); last = -2; }});
}});
</script>
"""
m.get_root().html.add_child(Element(hover_js))

# --- 9) Client control: switch overlay resolution on the fly ---
from folium import Element
res_switch_js = f"""
<script>
window.addEventListener('load', function(){{
  const map = window[{json.dumps(m.get_name())}];
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!map || !overlay) return;

  const SOURCES = {json.dumps(overlay_sources)};   // label -> data URI

  // Simple dropdown control
  const ResControl = L.Control.extend({{
    options: {{ position: 'topleft' }},
    onAdd: function() {{
      const div = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
      div.style.background = '#fff';
      div.style.padding = '4px 6px';
      div.style.boxShadow = '0 1px 3px rgba(0,0,0,.25)';
      div.title = 'Overlay resolution';

      const sel = L.DomUtil.create('select', '', div);
      sel.style.border = 'none';
      sel.style.outline = 'none';
      sel.style.font = '12px/1.2 system-ui,Segoe UI,Roboto,Arial,sans-serif';
      for (const label of Object.keys(SOURCES)) {{
        const opt = document.createElement('option');
        opt.value = label;
        opt.textContent = 'Resolution: ' + label;
        sel.appendChild(opt);
      }}
      sel.value = Object.keys(SOURCES)[0]; // Fine (native)

      // Prevent map drag when interacting with the select
      L.DomEvent.disableClickPropagation(div);
      sel.addEventListener('change', () => {{
        overlay.setUrl(SOURCES[sel.value]);
      }});

      return div;
    }}
  }});

  map.addControl(new ResControl());
}});
</script>
"""
m.get_root().html.add_child(Element(res_switch_js))

from folium import Element

blend_css_js = f"""
<style>
/* keep the overlay blocky & readable when zooming */
.leaflet-image-layer {{image-rendering: pixelated;}}
</style>
<script>
window.addEventListener('load', () => {{
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!overlay) return;
  // Wait until the <img> exists, then set blend mode
  const el = overlay.getElement ? overlay.getElement() : null;
  if (el) {{
    el.style.mixBlendMode = 'multiply';   // lets basemap texture show through
  }} else {{
    overlay.once('load', () => {{
      const el2 = overlay.getElement();
      if (el2) el2.style.mixBlendMode = 'multiply';
    }});
  }}
}});
</script>
"""
m.get_root().html.add_child(Element(blend_css_js))

opacity_control = f"""
<script>
window.addEventListener('load', () => {{
  const map = window[{json.dumps(m.get_name())}];
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!map || !overlay) return;

  // control UI
  const C = L.Control.extend({{
    options: {{position:'topleft'}},
    onAdd: function() {{
      const div = L.DomUtil.create('div','leaflet-bar');
      div.style.background = '#fff';
      div.style.padding = '6px';
      div.style.boxShadow = '0 1px 3px rgba(0,0,0,.3)';
      div.title = 'Overlay opacity';
      const label = document.createElement('div');
      label.style.font = '12px system-ui,Segoe UI,Roboto,Arial';
      label.style.marginBottom = '4px';
      label.textContent = 'Overlay opacity';
      const range = document.createElement('input');
      range.type = 'range'; range.min = 0; range.max = 100; range.value = 70;
      range.style.width = '160px';
      // stop map drag while using slider
      L.DomEvent.disableClickPropagation(div);
      range.addEventListener('input', () => {{
        overlay.setOpacity(range.value/100);
      }});
      div.appendChild(label); div.appendChild(range);
      return div;
    }}
  }});
  map.addControl(new C());
}});
</script>
"""
m.get_root().html.add_child(Element(opacity_control))

peek_js = f"""
<script>
window.addEventListener('load', () => {{
  const map = window[{json.dumps(m.get_name())}];
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!map || !overlay) return;

  let last = 0.70;  // remember last opacity (match your default)
  overlay.setOpacity(last);

  function hide(){{
    last = overlay.options.opacity ?? last;
    overlay.setOpacity(0.0);
  }}
  function show(){{
    overlay.setOpacity(last);
  }}

  // Hold Shift to peek
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Shift') hide(); }});
  document.addEventListener('keyup',   (e) => {{ if (e.key === 'Shift') show(); }});

  // Right mouse hold to peek
  map.on('mousedown', (e) => {{
    if (e.originalEvent && e.originalEvent.button === 2) hide();
  }});
  map.on('mouseup', () => show());
}});
</script>
"""
m.get_root().html.add_child(Element(peek_js))

import numpy as np
from skimage.measure import find_contours

# 1) choose a threshold (adjust to your scale)
thr = 0.70  # ≥70% risk
arr = np.clip(risk_web, 0, 1)
rows, cols = arr.shape
(south, west), (north, east) = bounds

# 2) find image-space contours
contours = find_contours(arr, thr)

# 3) convert (row, col) -> (lat, lon)
def rc_to_latlon(r, c):
    lat = north - (r/rows) * (north - south)
    lon = west  + (c/cols) * (east  - west)
    return float(lat), float(lon)

features = []
for seg in contours:
    # seg: N x 2 array of (row, col) floats
    coords = [rc_to_latlon(r, c)[::-1] for r, c in seg]  # GeoJSON: [lon, lat]
    if len(coords) >= 3:
        features.append({
            "type":"Feature",
            "geometry":{"type":"LineString","coordinates":coords},
            "properties":{"level": f">= {int(thr*100)}%"}
        })

if features:
    geojson = {"type":"FeatureCollection","features":features}
    folium.GeoJson(
        geojson,
        name=f"Risk contours (≥{int(thr*100)}%)",
        style_function=lambda f: {"color":"#ff4d4f","weight":2,"opacity":0.9,"dashArray":"4 2"}
    ).add_to(m)

contours

# --- Contour outlines (multi-threshold, always on top, with diagnostics) ---
import numpy as np
from folium import GeoJson, map as folium_map, Element
try:
    from skimage.measure import find_contours
except Exception as e:
    raise RuntimeError("scikit-image is required for contours: pip install scikit-image") from e

arr = np.clip(risk_web, 0, 1).astype(float)
rows, cols = arr.shape
(south, west), (north, east) = bounds

# 1) create a dedicated pane above overlays so lines are visible
contour_pane = folium_map.CustomPane("contours", z_index=650)  # overlayPane=400, tooltipPane=650
contour_pane.add_to(m)

# 2) helper: (row,col) -> (lon,lat)
def rc_to_lonlat(r, c):
    lat = north - (r/rows) * (north - south)
    lon = west  + (c/cols) * (east  - west)
    return float(lon), float(lat)

# 3) choose multiple levels so something always shows
levels = [0.50, 0.70, 0.90]
colors = {0.50: "#22c55e", 0.70: "#f59e0b", 0.90: "#ef4444"}  # green / amber / red
widths = {0.50: 1.5, 0.70: 2.0, 0.90: 2.5}

total_segments = 0
for lvl in levels:
    segs = find_contours(arr, lvl)
    if not segs:
        print(f"[Contours] Level {lvl:.2f}: 0 segments")
        continue

    feats = []
    for seg in segs:
        if seg.shape[0] < 4:   # skip tiny blips
            continue
        coords = [rc_to_lonlat(r, c) for r, c in seg]
        feats.append({
            "type": "Feature",
            "geometry": { "type": "LineString", "coordinates": coords },
            "properties": { "level": lvl }
        })

    if feats:
        total_segments += len(feats)
        GeoJson(
            {"type": "FeatureCollection", "features": feats},
            name=f"Contours ≥{int(lvl*100)}%",
            pane="contours",
            style_function=lambda f, L=lvl: {
                "color": colors[L], "weight": widths[L], "opacity": 1.0, "dashArray": "6 3"
            },
            tooltip=folium.features.GeoJsonTooltip(fields=[], aliases=[], labels=False)
        ).add_to(m)

print(f"[Contours] Drawn segments (all levels): {total_segments}")

# Optional: bring pane to front in Leaflet (extra safety)
m.get_root().html.add_child(Element("""
<script>
window.addEventListener('load', () => {
  const mapEl = document.querySelector('.leaflet-pane.contours');
  if (mapEl) mapEl.style.zIndex = 650;
});
</script>
"""))


import numpy as np
from skimage.measure import label, regionprops, find_contours

arr = np.clip(risk_web, 0, 1)
thr = 0.70
mask = arr >= thr

# label connected patches and keep the largest N by area
lab = label(mask, connectivity=2)
props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
keep_labels = {p.label for p in props[:25]}     # keep top 25 patches (tune N)
mask_clean = np.isin(lab, list(keep_labels))

# draw contours only around kept patches (same rc→lon/lat conversion you already use)
segs = find_contours(mask_clean.astype(float), 0.5)
# (convert segs to GeoJSON lines as before; style color '#ef4444', weight 2.5)


from skimage.morphology import remove_small_objects, binary_opening, disk

mask = arr >= 0.70
mask = remove_small_objects(mask, min_size=30)   # drop specks < 30 px
mask = binary_opening(mask, disk(1))             # smooth jaggies a bit
# then contour mask as above at 0.5