# PLACE INTO: CropXcel/analysis/engine.py

import os, json
from datetime import date as _py_date, timedelta

import ee, geemap
import ipywidgets as widgets

import numpy as np
import pandas as pd
import rasterio

# --- Earth Engine auth/init ---
try:
    ee.Initialize(project='angelic-edition-466705-r8')
    print("[EE] Initialized with existing credentials.")
except Exception:
    print("[EE] Authenticating...")
    ee.Authenticate()
    ee.Initialize(project='angelic-edition-466705-r8')
    print("[EE] Authenticated and initialized.")

# --- Map ---
m = geemap.Map(center=[11.45, 105.42],
               zoom=15,
               basemap='HYBRID',
               min_zoom=5,
               max_zoom=22,         # let user zoom further
               max_native_zoom=18)  # last zoom level with real data

# --- State / UI ---
LAST = {"gj": None}
# --- Draw control (rectangle + polygon optional) ---
dc = m.draw_control
dc.rectangle = {"shapeOptions": {"color": "#22c55e", "weight": 2, "opacity": 1}}
dc.polygon   = {"shapeOptions": {"color": "#ff7800", "weight": 2, "opacity": 1}}
dc.polyline = {}
dc.circle = {}
dc.circlemarker = {}
dc.marker = {}

# ---------- helpers ----------
def _pick_latest_feature(geo_json: dict):
    """Normalize the draw/edit payload to a single Feature."""
    if not geo_json:
        return None
    if geo_json.get("type") == "Feature":
        return geo_json
    if geo_json.get("type") == "FeatureCollection":
        feats = geo_json.get("features", [])
        return feats[-1] if feats else None
    # geometry-only dict → wrap as Feature
    return {"type": "Feature", "geometry": geo_json, "properties": {}}

def _is_axis_aligned_rect(coords):
    """Detect Leaflet rectangle ring with axis-aligned edges."""
    try:
        ring = coords[0]
        if len(ring) != 5:
            return False
        xs = [p[0] for p in ring[:4]]
        ys = [p[1] for p in ring[:4]]
        return (len({min(xs), max(xs)}) == 2) and (len({min(ys), max(ys)}) == 2)
    except Exception:
        return False

def _bbox_from_any_polygon_coords(coords):
    """BBox for any polygon ring list."""
    flat = [pt for ring in coords for pt in ring[:-1]]
    xs = [p[0] for p in flat]
    ys = [p[1] for p in flat]
    return min(xs), min(ys), max(xs), max(ys)

def _ee_geom_from_drawn(feature_or_geom: dict) -> ee.Geometry:
    """Convert drawn Feature/Geometry → ee.Geometry WITHOUT forcing rectangles."""
    if feature_or_geom is None:
        raise ValueError("Nothing drawn yet.")
    geom = feature_or_geom["geometry"] if feature_or_geom.get("type") == "Feature" else feature_or_geom
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not gtype or coords is None:
        raise ValueError("Invalid GeoJSON.")

    # cast to float (defensive)
    def _as_float(x):
        return [_as_float(xx) for xx in x] if isinstance(x, (list, tuple)) else float(x)
    coords = _as_float(coords)

    if gtype == "Polygon":
        # Keep polygon exactly as drawn (even if it looks like a rectangle)
        return ee.Geometry.Polygon(coords)
    if gtype == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords)
    if gtype == "LineString":
        # Keep as a line; if you want an area later, buffer explicitly at that moment.
        return ee.Geometry.LineString(coords)
    if gtype == "Point":
        # Keep as a point; buffer later only when needed.
        return ee.Geometry.Point(coords)
    # Fallback
    return ee.Geometry(geom)

# ---------- draw/edit callback with rectangle lock ----------
def _on_draw(target, action, geo_json):
    feat = _pick_latest_feature(geo_json)

    if action == "deleted":
        LAST["gj"] = None
        print("[deleted] cleared last feature.")
        return

    if feat is None:
        print(f"[{action}] no usable feature found.")
        return

    # Keep polygon/rectangle as drawn
    LAST["gj"] = feat
    print(f"[{action}] captured: {feat['geometry']['type']}")

dc.on_draw(_on_draw)

# ---------- "Use drawn polygon" button (save + visualize + EE object) ----------
def handle_use_click(_):
    if LAST["gj"] is None:
        print("⚠️ Draw or edit a shape first.")
        return

    global field
    ee_geom = None   # <--- initialize here so it always exists

    # Convert to EE geometry
    try:
        ee_geom = _ee_geom_from_drawn(LAST["gj"])
    except Exception as e:
        print("⚠️ Could not convert to EE geometry:", e)
        try:
            print(json.dumps(LAST["gj"], indent=2))
        except Exception:
            pass
        return

    # Make globally available
    field = ee_geom

    # Visualize
    try:
        m.addLayer(field, {"color": "red"}, "Field")
        m.centerObject(field, 15)
    except Exception as viz_err:
        print("Visualization note:", viz_err)

    # Print paste-ready snippet
    if ee_geom is not None:   # <--- safety check
        try:
            t = ee_geom.type().getInfo()
            print("Paste this:")
            if t == "Polygon":
                print("field = ee.Geometry.Polygon(")
                print(ee_geom.coordinates().getInfo())
                print(")")
            elif t == "MultiPolygon":
                print("field = ee.Geometry.MultiPolygon(")
                print(ee_geom.coordinates().getInfo())
                print(")")
            elif t == "LineString":
                print("field = ee.Geometry.LineString(")
                print(ee_geom.coordinates().getInfo())
                print(")")
            elif t == "Point":
                print("field = ee.Geometry.Point(")
                print(ee_geom.coordinates().getInfo())
                print(")")
            else:
                print(f"# {t}")
                print("field = ee.Geometry(")
                print(ee_geom.toGeoJSON().getInfo())
                print(")")
        except Exception:
            pass

    # Save EXACT drawn feature as GeoJSON (robust)
    try:
        with open("field_drawn.geojson", "w", encoding="utf-8") as f:
            json.dump(LAST["gj"], f, indent=2)
        print("✅ Saved to field_drawn.geojson")
    except Exception as e:
        print("Save skipped:", e)

btn = widgets.Button(description="Use drawn polygon")
btn.on_click(handle_use_click)
m.add_widget(btn, position="bottomright")

# show map
m

field_area = field.area().getInfo()
print("Area (ha):", field_area / 1e4)

# Try to populate `field` if user hasn't clicked yet
try:
    field
except NameError:
    if os.path.exists("field_drawn.geojson"):
        print("[AOI] Loading field from field_drawn.geojson")
        with open("field_drawn.geojson", "r", encoding="utf-8") as f:
            gj = json.load(f)
        field = ee.Geometry(gj)
    else:
        field = None
        print("⚠️ No AOI yet. Draw on the map and click 'Use drawn polygon'.")

if field is None:
    raise RuntimeError("Define 'field' first (draw and click the button).")

END = (_py_date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

EVENT_DAYS = 15   # event window length
BASE_DAYS  = 45   # baseline window length
GAP_DAYS   = 5    # gap between baseline and event

ORBIT_PASS   = None    # 'ASCENDING' or 'DESCENDING' or None
SCALE_METERS = 10
CRS          = None    # e.g., 'EPSG:32648'
OUT_STACK = input("Enter output filename (e.g., 'my_field.tif'): ").strip() or 'L1.tif'
if not OUT_STACK.endswith('.tif'):
    OUT_STACK += '.tif'

end   = ee.Date(END)
evt_s = end.advance(-EVENT_DAYS, 'day')
evt_e = end
base_e = evt_s.advance(-GAP_DAYS, 'day')
base_s = base_e.advance(-BASE_DAYS, 'day')

def fmt(d):          return ee.Date(d).format('YYYY-MM-dd')
def fmt_compact(d):  return ee.Date(d).format('YYYYMMdd')

print(f"[INFO] Event window : {fmt(evt_s).getInfo()} → {fmt(evt_e).getInfo()}")
print(f"[INFO] Baseline win : {fmt(base_s).getInfo()} → {fmt(base_e).getInfo()}")

def load_s1_ic(start, end, geom):
    ic = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    if ORBIT_PASS:
        ic = ic.filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS))
    return ic

def db_to_linear_keep(img):
    vv_lin = ee.Image(10).pow(img.select('VV').divide(10)).rename('VV_lin')
    vh_lin = ee.Image(10).pow(img.select('VH').divide(10)).rename('VH_lin')
    return (ee.Image(img)
            .addBands(vv_lin)
            .addBands(vh_lin)
            .copyProperties(img, img.propertyNames()))

def lin_to_db_safe(img_lin):
    # Mask non-positive before log10 to avoid -inf
    masked = img_lin.updateMask(img_lin.gt(0))
    return masked.log10().multiply(10)

def median_core(ic_lin, geom):
    def safe_median(col, names):
        return ee.Image(ee.Algorithms.If(
            col.size().gt(0),
            col.median().clip(geom),
            ee.Image.constant([0] * len(names)).updateMask(ee.Image.constant(0)).rename(names).clip(geom)
        ))
    vv_med = ee.Image(safe_median(ic_lin.select('VV_lin'), ['VV_lin']))
    vh_med = ee.Image(safe_median(ic_lin.select('VH_lin'), ['VH_lin']))
    ratio  = vh_med.divide(vv_med).updateMask(vv_med.gt(0)).rename('VH_VV')
    return vv_med.addBands(vh_med).addBands(ratio)

def std_band(ic_lin, band, geom):
    return ee.Image(ee.Algorithms.If(
        ic_lin.size().gt(1),
        ic_lin.select(band).reduce(ee.Reducer.stdDev()).rename(f"{band}_STD").clip(geom),
        ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename(f"{band}_STD")
    ))

# --- Add at top with your other params ---
TIMEZONE = 'Asia/Phnom_Penh'

def inclusive_end(d):
    # Earth Engine filterDate end is EXCLUSIVE; advance by 1 day to include the calendar day.
    return ee.Date(d).advance(1, 'day')

# --- Replace your "latest S1 acquisition" snippet with this ---
info_end   = inclusive_end(END)                 # <— include END day
info_start = ee.Date(END).advance(-30, 'day')

s1_info = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(field)
           .filterDate(info_start, info_end)    # end inclusive via +1 day
           .sort("system:time_start", False))

if s1_info.size().getInfo() > 0:
    latest_img = ee.Image(s1_info.first())
    # Show local (Cambodia) time so it matches what you expect on charts/labels.
    acq_time_local = latest_img.date().format("YYYY-MM-dd HH:mm", TIMEZONE).getInfo()
    acq_time_utc   = latest_img.date().format("YYYY-MM-dd HH:mm").getInfo()
    print("Latest acquisition (local):", acq_time_local)
    print("Latest acquisition (UTC)  :", acq_time_utc)
else:
    print("Latest acquisition: none in the last 30 days.")


s1_evt_raw  = load_s1_ic(evt_s,  evt_e,  field).map(db_to_linear_keep)
s1_base_raw = load_s1_ic(base_s, base_e, field).map(db_to_linear_keep)

evt_size = s1_evt_raw.size()

evt_img = ee.Image(ee.Algorithms.If(
    evt_size.gt(0),
    s1_evt_raw.sort('system:time_start', False).first(),
    ee.Image.constant([0,0]).updateMask(ee.Image.constant(0)).rename(['VV_lin','VH_lin'])
)).clip(field)

evt_date_str = ee.String(ee.Algorithms.If(
    evt_size.gt(0),
    ee.Date(ee.Image(s1_evt_raw.sort('system:time_start', False).first()).get('system:time_start')).format('YYYY-MM-dd'),
    'NA'
))
evt_date_tag = ee.String(ee.Algorithms.If(
    evt_size.gt(0),
    ee.Date(ee.Image(s1_evt_raw.sort('system:time_start', False).first()).get('system:time_start')).format('YYYYMMdd'),
    'NA'
))

evt_vv   = evt_img.select('VV_lin')
evt_vh   = evt_img.select('VH_lin')
evt_ratio= evt_vh.divide(evt_vv).updateMask(evt_vv.gt(0))

evt_core = (evt_vv.rename(ee.String('S1_VV_CURR_D').cat(evt_date_tag)))
evt_core = evt_core.addBands(evt_vh.rename(ee.String('S1_VH_CURR_D').cat(evt_date_tag)))
evt_core = evt_core.addBands(evt_ratio.rename(ee.String('S1_VH_VV_CURR_D').cat(evt_date_tag)))

base_core_lin = median_core(s1_base_raw, field)
base_suffix   = ee.String('_S').cat(fmt_compact(base_s)).cat('_E').cat(fmt_compact(base_e))

base_core  = base_core_lin.select('VV_lin').rename(ee.String('S1_VV_BASE').cat(base_suffix))
base_core  = base_core.addBands(base_core_lin.select('VH_lin').rename(ee.String('S1_VH_BASE').cat(base_suffix)))
base_core  = base_core.addBands(base_core_lin.select('VH_VV').rename(ee.String('S1_VH_VV_BASE').cat(base_suffix)))

vv_logratio_db = lin_to_db_safe(evt_vv.divide(base_core_lin.select('VV_lin'))).rename(
    ee.String('S1_VV_LOGRATIO_DB_D').cat(evt_date_tag).cat(base_suffix)
)
vh_logratio_db = lin_to_db_safe(evt_vh.divide(base_core_lin.select('VH_lin'))).rename(
    ee.String('S1_VH_LOGRATIO_DB_D').cat(evt_date_tag).cat(base_suffix)
)
vh_vv_diff = evt_ratio.subtract(base_core_lin.select('VH_VV')).rename(
    ee.String('S1_VH_VV_DIFF_D').cat(evt_date_tag).cat(base_suffix)
)

vv_std = std_band(s1_evt_raw, 'VV_lin', field).rename(
    ee.String('S1_VV_STD_W').cat(fmt_compact(evt_s)).cat('_').cat(fmt_compact(evt_e))
)
vh_std = std_band(s1_evt_raw, 'VH_lin', field).rename(
    ee.String('S1_VH_STD_W').cat(fmt_compact(evt_s)).cat('_').cat(fmt_compact(evt_e))
)

stack = ee.Image.cat([
    evt_core, base_core,
    vv_logratio_db, vh_logratio_db, vh_vv_diff,
    vv_std, vh_std
]).toFloat()

print("Bands in export:", stack.bandNames().getInfo())

# Rough pixel count estimate at 10 m
try:
    aoi_area_m2 = field.area().getInfo()
    px = aoi_area_m2 / (SCALE_METERS * SCALE_METERS)
    if px > 40_000_000:  # ~200 MP as a soft warning
        print(f"⚠️ Large export (~{px/1e6:.1f} MP). Consider a coarser scale or smaller AOI.")
except Exception:
    pass

stack = ee.Image.cat([
    evt_core,                # 3 bands with exact event date
    base_core,               # 3 bands with baseline window dates
    vv_logratio_db,          # 1
    vh_logratio_db,          # 1
    vh_vv_diff,              # 1
    vv_std, vh_std           # 2
]).toFloat()

print("Bands in export:", stack.bandNames().getInfo())

print(f"[EXPORT] Saving stack: {OUT_STACK}")
geemap.ee_export_image(
    stack,
    OUT_STACK,
    scale=SCALE_METERS,
    region=field,
    file_per_band=False,
    crs=CRS
)
print("[OK] Exported:", OUT_STACK)