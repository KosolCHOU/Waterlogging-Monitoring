# Notebook → Module Mapping

01. **utils.py**  ←  _Imports + EE init_
   - Preview: `import os, json
from datetime import date as _py_date, timedelta

import ee, geemap
import ipywidgets as widgets

import numpy as np
import pandas as pd
import rasterio

# --- Earth Engine auth/init -`

02. **overlays.py**  ←  _Map + draw → ee.Geometry (polygon-first)_
   - Preview: `# --- Map ---
m = geemap.Map(center=[11.45, 105.42],
               zoom=15,
               basemap='HYBRID',
               min_zoom=5,
               max_zoom=22,         # let user zoom further
   `

03. **overlays.py**  ←  _Map + draw → ee.Geometry (polygon-first)_
   - Preview: `field_area = field.area().getInfo()
print("Area (ha):", field_area / 1e4)`

04. **utils.py**  ←  _AOI fallback (load from saved file if not clicked)_
   - Preview: `# Try to populate field if user hasn't clicked yet
try:
    field
except NameError:
    if os.path.exists("field_drawn.geojson"):
        print("[AOI] Loading field from field_drawn.geojson")
      `

05. **temporal.py**  ←  _Time windows + helpers_
   - Preview: `if field is None:
    raise RuntimeError("Define 'field' first (draw and click the button).")

END = (_py_date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

EVENT_DAYS = 15   # event window lengt`

06. **utils.py**  ←  _S1 loaders + safe math_
   - Preview: `def load_s1_ic(start, end, geom):
    ic = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW`

07. **utils.py**  ←  _Find latest acquisition safely (for info)_
   - Preview: `# --- Add at top with your other params ---
TIMEZONE = 'Asia/Phnom_Penh'

def inclusive_end(d):
    # Earth Engine filterDate end is EXCLUSIVE; advance by 1 day to include the calendar day.
    return`

08. **utils.py**  ←  _Build event/baseline stacks and change metrics_
   - Preview: `s1_evt_raw  = load_s1_ic(evt_s,  evt_e,  field).map(db_to_linear_keep)
s1_base_raw = load_s1_ic(base_s, base_e, field).map(db_to_linear_keep)

evt_size = s1_evt_raw.size()

evt_img = ee.Image(ee.Algor`

09. **utils.py**  ←  _Optional: export size sanity (warn if very large)_
   - Preview: `# Rough pixel count estimate at 10 m
try:
    aoi_area_m2 = field.area().getInfo()
    px = aoi_area_m2 / (SCALE_METERS * SCALE_METERS)
    if px > 40_000_000:  # ~200 MP as a soft warning
        pri`

10. **utils.py**  ←  _-------------------- FINAL STACK (11 bands) --------------------_
   - Preview: `stack = ee.Image.cat([
    evt_core,                # 3 bands with exact event date
    base_core,               # 3 bands with baseline window dates
    vv_logratio_db,          # 1
    vh_logratio_d`

11. **utils.py**  ←  _Export to GeoTIFF_
   - Preview: `print(f"[EXPORT] Saving stack: {OUT_STACK}")
geemap.ee_export_image(
    stack,
    OUT_STACK,
    scale=SCALE_METERS,
    region=field,
    file_per_band=False,
    crs=CRS
)
print("[OK] Exported:", `

12. **utils.py**  ←  _Configure expected logical bands (11 total)_
   - Preview: `# ======== CONFIG (logical expectations) ========
TIF_PATH = OUT_STACK  # or a literal path
# These are the "base" logical band names your pipeline exports (11 total)
EXPECTED_BASES = [
    'S1_VV_CUR`

13. **utils.py**  ←  _Helpers (read names, compute valid %, normalize_
   - Preview: `import os, re
import numpy as np
import pandas as pd
import rasterio

# Matches your naming scheme and extracts the logical "base" key.
# Examples:
#  S1_VV_CURR_D20250830              -> S1_VV_CURR
#`

14. **utils.py**  ←  _Main checker (date-aware)_
   - Preview: `def main_check_tif(tif_path=TIF_PATH):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        print(f"[IN`

15. **utils.py**  ←  _Config + helpers_
   - Preview: `# ==== CONFIG ====
tif_path = OUT_STACK     # change if needed
save_pngs = False       # True to save each band as PNG into ./plots
cols = 3                # grid columns for subplot layout

# Logical`

16. **utils.py**  ←  _Open file, resolve indices, choose plotting list_
   - Preview: `# --- Cell 2 (robust to generic Band1..Band11 names) ---

from collections import OrderedDict

if not os.path.exists(tif_path):
    raise FileNotFoundError(f"GeoTIFF not found: {tif_path}")

EXPECTED_`

17. **utils.py**  ←  _Plot_
   - Preview: `# Choose colormaps per feature
CMAP = {
    # backscatter/ratio snapshots (unit depends on your export; current/base are linear here)
    'S1_VV_CURR': 'viridis',
    'S1_VH_CURR': 'viridis',
    'S1_`

18. **utils.py**  ←  _Helpers_
   - Preview: `from datetime import datetime, timedelta, date as date_class
from dateutil.relativedelta import relativedelta
import ee, pandas as pd, numpy as np, os

# --- Params ---
END   = (date_class.today() - t`

19. **utils.py**  ←  _Build features entirely in EE_
   - Preview: `steps = date_steps(START, END, STEP_DAYS)

def feature_for_step(step_end):
    step_end = ee.Date(step_end)
    evt_start  = step_end.advance(-EVENT_DAYS, 'day')
    base_end   = evt_start.advance(-GA`

20. **utils.py**  ←  _Fetch once, post-process, save_
   - Preview: `# Bring to client ONCE
records = fc.getInfo()['features']

# Normalize to rows
rows = []
for f in records:
    p = f['properties']
    rows.append({
        "date_local": p.get('date_local'),
        `

21. **utils.py**  ←  _Time-Series Visualization_
   - Preview: `import matplotlib.pyplot as plt

# Reload df
df = pd.read_csv(OUTPUT_CSV, parse_dates=["date"], index_col="date")

groups = {
    "Current (linear)": ["S1_VV_CURR", "S1_VH_CURR", "S1_VH_VV_CURR"],
   `

22. **temporal.py**  ←  _GEE: CHIRPS rolling totals (mm)_
   - Preview: `import ee, datetime as dt
ee.Initialize()

def chirps_totals_mm(aoi: ee.Geometry, end_utc=None):
    end = ee.Date(end_utc or dt.datetime.utcnow())
    start7  = end.advance(-7, 'day')
    start30 = e`

23. **utils.py**  ←  _GEE: Sentinel-1 quick water mask (VH + Otsu)_
   - Preview: `def chirps_totals_for_fields(fc: ee.FeatureCollection, end_utc=None):
    end = ee.Date(end_utc or dt.datetime.utcnow())
    start7  = end.advance(-7, 'day')
    start30 = end.advance(-30, 'day')

   `

24. **utils.py**  ←  _Forecast: Open-Meteo 72h hourly precipitation (mm)_
   - Preview: `s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(field)
      .filterDate(end.advance(-20,'day'), end)
      .filter(ee.Filter.eq('instrumentMode','IW'))
      .filter(ee.Filter.eq('o`

25. **utils.py**  ←  _Rule engine (clear, editable)_
   - Preview: `import requests, pandas as pd, datetime as dt

# Extract centroid coordinates from the field geometry
field_bounds = field.bounds().getInfo()
coords = field_bounds['coordinates'][0]
lats = [coord[1] f`

26. **utils.py**  ←  _Rule engine (clear, editable)_
   - Preview: `def advise(s1_water_ha, imerg_24h, om_48h, sowing_stage: bool):
    # thresholds: tune per province/soil later
    if s1_water_ha > 0.1:  # 0.1 ha flooded inside AOI
        return "RED", "Water detec`

27. **utils.py**  ←  _Import all necessaries libraires_
   - Preview: `import os, io, base64, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import rasterio
from rasterio.transform import xy, array_bounds, from_bound`

28. **utils.py**  ←  _---------------- CONFIG ----------------_
   - Preview: `# ------- Inputs -------
MULTIBAND_PATH = tif_path                 # 11-band S1 stack you exported earlier
TIME_SERIES_CSV = OUTPUT_CSV              # AOI mean time series csv

# Band indices (1-based`

29. **temporal.py**  ←  _---------- Temporal Engine controls ----------_
   - Preview: `# Temporal/alert products (if your temporal engine writes them)
ALERTS_CSV      = f"alerts_{_base}.csv"
ALERTS_PLOT_PNG = f"S1_alerts_plot_{_base}.png"
RECS_CSV        = f"recommendations_{_base}.csv"`

30. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def pct_stretch(a, pmin=2, pmax=98):
    """Percentile stretch to 0..1 on finite values; returns float32."""
    if a is None:
        return None
    a = a.astype("float32")
    # Use nanpercentile s`

31. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def finite_percentiles(a, lo=2, hi=98):
    if a is None:
        return 0.0, 1.0
    lo_v, hi_v = np.nanpercentile(a, [lo, hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
`

32. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def _is_db(x):
    # heuristic: SAR dB typically ~[-30, +5]
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return False
    return (np.nanmin(finite) < -10.0) or (np.nanmedian(finite)`

33. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `with rasterio.open(MULTIBAND_PATH) as src:
    profile = src.profile
    W, H = src.width, src.height
    src_crs = src.crs
    src_transform = src.transform
    src_bounds = src.bounds

    arrays = `

34. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `# Derive current and/or baseline VH/VV ratios if missing
def _safe_ratio(numer, denom):
    out = np.full_like(numer, np.nan, dtype=np.float32)
    mask = np.isfinite(numer) & np.isfinite(denom) & (de`

35. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def save_geotiff(path, data, profile, nodata_val=-9999.0):
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=nodata_val)
    with rasterio.open(path, "w", **p) as dst:
        dst.w`

36. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def top_reason_label(key: str):
    return {
        "sar_water_drop":  "Sharp drop in radar backscatter (possible inundation).",
        "sar_ratio_change":"Change in VH/VV ratio (flooded vegetation `

37. **utils.py**  ←  _--------------- HELPERS ----------------_
   - Preview: `def risk_cmap_redgreen():
    # green = low risk, red = high risk
    return ListedColormap([
        "#006400",  # dark green
        "#228B22",  # medium green
        "#ADFF2F",  # yellow-green
   `

38. **utils.py**  ←  _--------------- READ STACK ---------------_
   - Preview: `# Read the raster data first (masked→NaN)
with rasterio.open(MULTIBAND_PATH) as src:
    profile = src.profile
    W, H = src.width, src.height
    src_crs = src.crs
    src_transform = src.transform
`

39. **utils.py**  ←  _--------------- CONTRIBUTORS ---------------_
   - Preview: `# ----- SAR-first contributors (robust to missing bands) -----

def stretch01(x, invert=False):
    s = pct_stretch(x)
    return None if s is None else (1 - s if invert else s)

contributors = []

# `

40. **utils.py**  ←  _--------------- REPROJECT TO EPSG:4326 ---------------_
   - Preview: `dst_crs = rasterio.crs.CRS.from_epsg(4326)

# Build target grid (4326) from source bounds
dst_transform, dst_w, dst_h = calculate_default_transform(
    src_crs, dst_crs, W, H, *src_bounds  # src_boun`

41. **overlays.py**  ←  _Risk colormap_
   - Preview: `from matplotlib.colors import ListedColormap

def risk_cmap_redgreen():
    # green (low) → yellow → orange → red (high)
    return ListedColormap([
        "#006400", "#228B22", "#7FFF00", "#FFFF00",`

42. **overlays.py**  ←  _PNG overlay helper_
   - Preview: `import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import imageio

def save_overlay_png(arr, out_png, *, aoi_mask=None, fixed01=True,
    `

43. **utils.py**  ←  _Build AOI mask on the web grid_
   - Preview: `from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon

def build_aoi_mask(field_gdf, transform, out_shape, target_crs):
    """
    Rasterize AOI polygon`

44. **overlays.py**  ←  _Compute bounds_
   - Preview: `from rasterio.transform import xy

def latlon_from_transform(t, row, col):
    x, y = xy(t, row, col)  # x=lon, y=lat in EPSG:4326
    return y, x

top_lat, left_lon  = latlon_from_transform(web_trans`

45. **overlays.py**  ←  _Save overlays_
   - Preview: `# With colorbar
save_overlay_png(
    risk_web, OUT_PNG_CB,
    aoi_mask=aoi_mask_web, fixed01=True,
    with_colorbar=True, title="Waterlogging Risk (0–1)",
    outside_mode="transparent", feather_px`

46. **utils.py**  ←  _Imports & constants_
   - Preview: `# =========================
# Cell 1: imports & constants
# =========================
import io, base64, json, warnings
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from `

47. **utils.py**  ←  _Helpers_
   - Preview: `# =========================
# Cell 2: helpers
# =========================
def top_reason_label(idx_or_name):
    if isinstance(idx_or_name, int):
        return FRIENDLY_LABELS[idx_or_name] if 0 <= id`

48. **hotspots.py**  ←  _AOI reprojection & mask_
   - Preview: `# =========================
# Cell 3: AOI reprojection & mask
# =========================
def _to_src_crs_geom(field_polygon, src_crs):
    """Return a shapely geometry in src_crs (GeoDataFrame -> to_`

49. **utils.py**  ←  _Main extraction_
   - Preview: `# =========================
# Cell 4: main extraction
# =========================
def extract_hotspots(
    risk,                 # 2D numpy array, values in [0,1] (or any continuous risk proxy)
    s`

50. **utils.py**  ←  _Main extraction_
   - Preview: `# =========================
# Cell 5: example call (edit to your variables)
# =========================
# You already have these from your pipeline:
# risk: 2D np.ndarray in [0,1]
# src_transform: ras`

51. **overlays.py**  ←  _MAP_
   - Preview: `import os, json, base64
import numpy as np
import folium
from folium.plugins import Fullscreen, MeasureControl, MousePosition
from folium.raster_layers import ImageOverlay
import geopandas as gpd

# -`

52. **overlays.py**  ←  _MAP_
   - Preview: `from folium import Element

blend_css_js = f"""
<style>
/* keep the overlay blocky & readable when zooming */
.leaflet-image-layer {{image-rendering: pixelated;}}
</style>
<script>
window.addEventList`

53. **overlays.py**  ←  _MAP_
   - Preview: `opacity_control = f"""
<script>
window.addEventListener('load', () => {{
  const map = window[{json.dumps(m.get_name())}];
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!map || !ov`

54. **overlays.py**  ←  _MAP_
   - Preview: `peek_js = f"""
<script>
window.addEventListener('load', () => {{
  const map = window[{json.dumps(m.get_name())}];
  const overlay = window[{json.dumps(overlay.get_name())}];
  if (!map || !overlay) r`

55. **overlays.py**  ←  _MAP_
   - Preview: `import numpy as np
from skimage.measure import find_contours

# 1) choose a threshold (adjust to your scale)
thr = 0.70  # ≥70% risk
arr = np.clip(risk_web, 0, 1)
rows, cols = arr.shape
(south, west),`

56. **overlays.py**  ←  _MAP_
   - Preview: `contours`

57. **overlays.py**  ←  _MAP_
   - Preview: `# --- Contour outlines (multi-threshold, always on top, with diagnostics) ---
import numpy as np
from folium import GeoJson, map as folium_map, Element
try:
    from skimage.measure import find_contou`

58. **overlays.py**  ←  _MAP_
   - Preview: `import numpy as np
from skimage.measure import label, regionprops, find_contours

arr = np.clip(risk_web, 0, 1)
thr = 0.70
mask = arr >= thr

# label connected patches and keep the largest N by area
l`

59. **overlays.py**  ←  _MAP_
   - Preview: `from skimage.morphology import remove_small_objects, binary_opening, disk

mask = arr >= 0.70
mask = remove_small_objects(mask, min_size=30)   # drop specks < 30 px
mask = binary_opening(mask, disk(1)`

60. **temporal.py**  ←  _TEMPORAL ENGINE_
   - Preview: `def compute_temporal_engine_s1(csv_path):
    """
    Temporal engine for Sentinel-1 waterlogging.
    Returns: (alerts_df, insights_df, alerts_plot_png_path, insights_csv_path)
    Side-effect: write`

61. **temporal.py**  ←  _TEMPORAL ENGINE_
   - Preview: `# --------- REQUIRED GLOBALS (safe defaults) ----------
# Rolling window for robust z-score
ROLL_WINDOW_DAYS    = 60

# Z-score threshold for alert bucket (more negative = stronger anomaly)
Z_THRESHOL`

62. **temporal.py**  ←  _TEMPORAL ENGINE_
   - Preview: `def compute_temporal_engine_s1(csv_path):
    """
    Temporal engine for Sentinel-1 waterlogging.
    Returns: (alerts_df, recs_df, ALERTS_PLOT_PNG)

    Side-effect: also writes an 'insights' CSV (o`

63. **temporal.py**  ←  _TEMPORAL ENGINE_
   - Preview: `# --------- RUN + DISPLAY ----------
csv_path = f"timeseries_{os.path.splitext(OUT_STACK)[0]}.csv"
alerts_df, recs_df, plot_path = compute_temporal_engine_s1(csv_path)

print("alerts_df shape:", getat`

64. **temporal.py**  ←  _TEMPORAL ENGINE_
   - Preview: `# ------- 0) Setup -------
edges   = np.array([0.00, 0.30, 0.50, 0.70, 1.00], dtype=float)   # 4 bins: [0,.3), [.3,.5), [.5,.7), [.7,1]
labels  = [0, 1, 2, 3]  # 0=Healthy, 1=Watch, 2=Concern, 3=Alert`

65. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `# ========================== CROPXCEL DASH (FINAL DROP-IN) ==========================
# - Robust to missing upstream cells (safe fallbacks)
# - Donut rings w/ smooth rotation (fun, but accessible & pa`

66. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `# ------------------------------- SCALE / DONUT DATA --------------------------------
def _pct(val, tot): 
    return (100.0 * val / tot) if (tot and tot > 0) else 0.0

scale_data = []
for k in [0, 1,`

67. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `# ------------------------------ INSIGHTS / TABLES ----------------------------------
INSIGHTS_CSV = (ALERTS_CSV.replace("alerts", "insights")
                if "alerts" in os.path.basename(ALERTS_CS`

68. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `farmer_display`

69. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `technical_display`

70. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `# ------------------------------ ANALYSIS SCALE HTML ---------------------------------
scale_section = f"""
<div class="card scaleCard">

  <div class="scaleBody">
    <!-- legend -->
    <div class="`

71. **utils.py**  ←  _-------- Display tables / PANEL HTML --------_
   - Preview: `# ------------------------------- BOUNDS FOR MAP FIT --------------------------------
BOUNDS_JSON = "null"
try:
    if 'field_gdf' in globals() and field_gdf is not None and len(field_gdf) > 0:
      `

72. **overlays.py**  ←  _Template — Part A (Map bootstrap + donut tagging)_
   - Preview: `from string import Template

partA = r"""
<script>
/* ================= MAP BOOTSTRAP ================= */
(function(){
  const BOUNDS = $BOUNDS;  // [[S,W],[N,E]]

  function getMap(){
    const el =`

73. **utils.py**  ←  _Template — Part B (CSS only)_
   - Preview: `partB = r"""
<style>
  html, body {
    height:100%; overflow:hidden; margin:0; background:#f1f5f9;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  }

  /* ===== LA`

74. **utils.py**  ←  _Template — Part C (HTML only)_
   - Preview: `partC = r"""
<!-- Brand -->
<div class="brand" aria-label="CropXcel" style="padding:6px 10px 0 10px;">
  <span style="font-weight:900;font-size:20px;background:#fff;border-radius:8px;padding:4px 8px;b`

75. **utils.py**  ←  _Template — Part D (UI logic, DOM-ready)_
   - Preview: `partD = r"""
<script>
(function(){
  const classes = [
    { k:0, pct:52.8, ha:24.68, color:"#2ecc71" }, // Healthy
    { k:1, pct:29.8, ha:13.94, color:"#f1c40f" }, // Watch
    { k:2, pct:12.0, ha:5`

76. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `partE = r"""
<script>
(function(){
  function getMap(){
    const el=document.querySelector('div[id^="map_"]');
    return el ? (window[el.id.replace(/-/g,'_')] || null) : null;
  }
  function refresh`

77. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `from string import Template

tpl = Template(
    partA +
    partB +
    partC+
    partD +
    partE
)`

78. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `# === Advice (dual: farmer-friendly + technical) ===
TH_WATER_HA          = max(0.1, 0.05 * field_area)
TH_FORECAST_RED      = 30.0
TH_FORECAST_YELLOW   = 15.0
TH_PAST24H_YELLOW    = 25.0

def advise_`

79. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `print("Advice level:", ADVICE_LEVEL)
print("Advice message:", ADVICE_MSG)
print("S1 water ha:", S1_WATER_HA)
print("IMERG 24h mm:", IMERG_24H_MM)
print("Forecast 48h mm:", FORECAST_48H_MM)`

80. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `from html import escape

ADVICE_MSG_HTML      = escape(ADVICE_MSG)
ADVICE_SUBTEXT_HTML  = escape(ADVICE_SUBTEXT)

ADVICE_FARMER_JSON = json.dumps(ADVICE_FARMER)  # keep JSON for dataset
ADVICE_TECH_JS`

81. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `# --- (before templating) make JS-safe strings ---
ADVICE_FARMER_JSON  = json.dumps(ADVICE_FARMER)
ADVICE_TECH_JSON    = json.dumps(ADVICE_TECH)
ADVICE_SUBTEXT_JSON = json.dumps(ADVICE_SUBTEXT)

# ---`

82. **utils.py**  ←  _Template — Part E (final nudge; DOM-ready)_
   - Preview: `m.get_root().html.add_child(Element(dashboard_html))
m.save(OUT_HTML)`
