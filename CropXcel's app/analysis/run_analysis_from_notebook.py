# analysis/run_analysis_from_notebook.py
import json, uuid, pathlib
from datetime import date as date_class, timedelta, datetime as dt
from dateutil.relativedelta import relativedelta
import ee

EE_PROJECT = "angelic-edition-466705-r8"

def _ensure_ee():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Initialize()

# ---------- helpers (definitions only) ----------
def date_steps(start_str: str, end_str: str, step_days: int = 10):
    d0 = dt.fromisoformat(start_str).date()
    d1 = dt.fromisoformat(end_str).date()
    out = []
    while d0 <= d1:
        out.append(ee.Date(d0.strftime("%Y-%m-%d")))
        d0 = d0 + timedelta(days=step_days)
    return ee.List(out)

def load_s1_ic(start, end, geom):
    ic = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return ic

def db_to_linear(img_db):
    return ee.Image(10).pow(img_db.divide(10))

def add_linear(img_db):
    vv_lin = db_to_linear(img_db.select("VV")).rename("VV_lin")
    vh_lin = db_to_linear(img_db.select("VH")).rename("VH_lin")
    return img_db.addBands([vv_lin, vh_lin])

def ic_median_lin(ic, geom):
    ic_lin = ic.map(add_linear)
    return ee.Image(ee.Algorithms.If(
        ic_lin.size().gt(0),
        ic_lin.select(["VV_lin", "VH_lin"]).median().clip(geom),
        ee.Image.constant([0, 0]).updateMask(ee.Image.constant(0)).rename(["VV_lin", "VH_lin"]).clip(geom)
    ))

def safe_divide(numer, denom):
    return numer.divide(denom).updateMask(denom.gt(0))

# ---------- main API ----------
def run_analysis_from_notebook(
    aoi_geojson: dict,
    start: str | None = None,
    end: str | None = None,
    event_days: int | None = None,
    base_days: int | None = None,
    gap_days: int | None = None,
    step_days: int = 10
) -> dict:
    _ensure_ee()

    EVENT_DAYS = event_days or 15
    BASE_DAYS  = base_days  or 45
    GAP_DAYS   = gap_days   or 5
    END   = end   or (date_class.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    START = start or (date_class.today() - relativedelta(months=4)).strftime("%Y-%m-%d")

    geom = ee.Geometry(aoi_geojson)
    # Leaflet bounds in lat/lon order
    leaflet_bounds = _leaflet_bounds_from_geojson(aoi_geojson)

    steps = date_steps(START, END, step_days)
    step_end = ee.Date(steps.get(steps.length().subtract(1)))

    evt_start  = step_end.advance(-EVENT_DAYS, "day")
    base_end   = evt_start.advance(-GAP_DAYS, "day")
    base_start = base_end.advance(-BASE_DAYS, "day")

    ic_evt  = load_s1_ic(evt_start, step_end.advance(1, "day"), geom)
    ic_base = load_s1_ic(base_start, base_end.advance(1, "day"), geom)

    evt_med  = ic_median_lin(ic_evt,  geom)
    base_med = ic_median_lin(ic_base, geom)

    vv_curr = evt_med.select("VV_lin")
    vh_curr = evt_med.select("VH_lin")
    vv_base = base_med.select("VV_lin")
    vh_base = base_med.select("VH_lin")

    ratio_curr = safe_divide(vh_curr, vv_curr)
    ratio_base = safe_divide(vh_base, vv_base)
    risk = ratio_curr.subtract(ratio_base).unitScale(-0.5, 0.5).clamp(0, 1).rename("risk")

    vis = {"min": 0, "max": 1, "palette": ["#00ff00", "#ffff00", "#ff0000"]}
    md = ee.Image(risk.clip(geom)).getMapId(vis)
    tile_url = f"https://earthengine.googleapis.com/map/{md['mapid']}/{{z}}/{{x}}/{{y}}?token={md['token']}"
    # Optional non-tiled PNG overlay (can be used with L.imageOverlay)
    vis_img = ee.Image(risk.clip(geom)).visualize(**vis)
    overlay_png_url = vis_img.getThumbURL({
        "region": geom,     # can also pass dict(aoi_geojson) but ee.Geometry is fine
        "scale": 10,        # ~10 m; increase for faster/smaller image
        "format": "png"
    })

    hs = risk.gt(0.7).selfMask()
    vectors = hs.reduceToVectors(geometry=geom, scale=30, geometryType="centroid", maxPixels=1e9)
    pts = vectors.map(lambda f: f.set({"risk": risk.reduceRegion(ee.Reducer.mean(), f.geometry(), 30).get("risk")}))
    gj = ee.FeatureCollection(pts).getInfo()

    media_dir = pathlib.Path("media/hotspots"); media_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.geojson"
    (media_dir / fname).write_text(json.dumps(gj))
    hotspots_url = f"/media/hotspots/{fname}"

    px_area = ee.Image.pixelArea().rename("area").clip(geom)
    total_ha = px_area.reduceRegion(ee.Reducer.sum(), geom, 30).getNumber("area").getInfo() / 10000.0
    high_ha  = hs.multiply(ee.Image.pixelArea()).rename("area") \
                 .reduceRegion(ee.Reducer.sum(), geom, 30).getNumber("area").getInfo() / 10000.0

    return {
        "tile_url": tile_url,                   # your tiled overlay
        "overlay_png_url": overlay_png_url,     # NEW: single PNG overlay URL
        "hotspots_url": hotspots_url,
        "bounds": leaflet_bounds,               # NEW: Leaflet [[S,W],[N,E]]
        "stats": {
            "area_total_ha": round(total_ha, 3),
            "area_highrisk_ha": round(high_ha, 3),
            "pct_high": round((high_ha / total_ha * 100) if total_ha else 0.0, 2),
        },
        "last_step_end": step_end.format("YYYY-MM-dd").getInfo(),
    }

def _leaflet_bounds_from_geojson(aoi_geojson: dict):
    """Return [[south, west], [north, east]] from a Polygon/MultiPolygon GeoJSON."""
    t = aoi_geojson.get("type")
    coords = aoi_geojson.get("coordinates", [])
    pts = []
    if t == "Feature":
        return _leaflet_bounds_from_geojson(aoi_geojson["geometry"])
    if t == "Polygon" and coords:
        pts = coords[0]
    elif t == "MultiPolygon" and coords and coords[0] and coords[0][0]:
        pts = coords[0][0]
    else:
        return [[0,0],[0,0]]
    lats = [p[1] for p in pts]; lons = [p[0] for p in pts]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]
