# CropXcel/analysis/run_analysis_from_notebook.py
# Purpose: minimal "analysis" like your notebook (no GEE calls here).
# Input: AOI geojson + path to a pre-exported Sentinel-1 stack GeoTIFF.
# Output: PNG overlay + hotspots GeoJSON + bounds (Leaflet order).

from __future__ import annotations
import json, uuid, re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds, from_bounds
import imageio
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import shape
# Reuse your existing hotspot code (do NOT duplicate)
from .hotspots import extract_hotspots

MEDIA_ROOT = Path("media")
OVER_DIR   = MEDIA_ROOT / "overlays"
HOT_DIR    = MEDIA_ROOT / "hotspots"
for d in (OVER_DIR, HOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- tiny helpers (local only) -----------------------------------------------
def _bounds_from_geom(aoi: dict):
    t = aoi.get("type")
    coords = aoi.get("coordinates", [])
    if t == "Feature":
        return _bounds_from_geom(aoi["geometry"])
    if t == "Polygon" and coords:
        ring = coords[0]
    elif t == "MultiPolygon" and coords and coords[0] and coords[0][0]:
        ring = coords[0][0]
    else:
        return [[0,0],[0,0]]
    lats = [p[1] for p in ring]; lons = [p[0] for p in ring]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

def _pct_stretch(a, pmin=2, pmax=98):
    finite = np.isfinite(a)
    if not finite.any(): return np.zeros_like(a, dtype="float32")
    lo, hi = np.nanpercentile(a[finite], [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype="float32")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype("float32")

def _risk_palette():
    # green → yellow → orange → red
    return mcolors.ListedColormap(
        ["#006400","#2a8a2a","#7bdc46","#ffff66","#ffc04d","#ff8a4d","#ff5233","#b51212"]
    )

def _save_png01(arr01, out_png):
    rgba = cm.ScalarMappable(norm=mcolors.Normalize(0,1), cmap=_risk_palette()).to_rgba(arr01, bytes=True)
    rgba[~np.isfinite(arr01), 3] = 0
    imageio.imwrite(out_png, rgba)

# ---- main API (called by your tasks/view) ------------------------------------
def run_analysis_from_notebook(
    aoi_geojson: dict,
    *,
    stack_tif_path: str | Path,     # ← path to the exported S1 stack (your notebook export)
    max_web_width: int = 2000,      # downsample for web
) -> dict:
    """
    Returns:
      {
        "overlay_png_url": "/media/overlays/risk_xxx.png",
        "hotspots_url": "/media/hotspots/hs_xxx.geojson",
        "bounds": [[S,W],[N,E]]
      }
    """
    tif_path = Path(stack_tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"Stack not found: {tif_path}")

    # --- 1) Read stack bands (no duplication of heavy logic)
    # Expected band order from your export (keep aligned with your notebook):
    # 1 S1_VV_CURR, 2 S1_VH_CURR, 3 S1_VH_VV_CURR,
    # 4 S1_VV_BASE, 5 S1_VH_BASE, 6 S1_VH_VV_BASE,
    # 7 S1_VV_LOGRATIO_DB, 8 S1_VH_LOGRATIO_DB, 9 S1_VH_VV_DIFF,
    # 10 S1_VV_STD, 11 S1_VH_STD
    with rasterio.open(str(tif_path)) as src:
        src_crs, src_tr = src.crs, src.transform
        W, H = src.width, src.height
        arr = {i: src.read(i, masked=True).filled(np.nan).astype("float32") for i in range(1, min(12, src.count+1))}

    # --- 2) Compose a risk proxy (exactly the notebook spirit, but minimal)
    # components in 0..1
    # (a) Water drop → large negative logratio (use VH/VV dB, pick max)
    drop = []
    if 7 in arr: drop.append(-arr[7])
    if 8 in arr: drop.append(-arr[8])
    drop01 = _pct_stretch(np.nanmax(np.dstack(drop), axis=2)) if drop else np.zeros((H, W), "float32")
    # (b) Ratio change → |VH/VV diff|
    ratio01 = _pct_stretch(np.abs(arr.get(9, np.zeros((H, W), "float32"))))
    # (c) Variability → max(stdVV, stdVH)
    var_stack = np.dstack([a for k,a in arr.items() if k in (10,11)]) if (10 in arr or 11 in arr) else None
    var01 = _pct_stretch(np.nanmax(var_stack, axis=2)) if var_stack is not None else np.zeros((H, W), "float32")

    # weights (keep simple, you can tune later like in notebook)
    w_drop, w_ratio, w_var = 0.5, 0.3, 0.2
    wsum = w_drop + w_ratio + w_var
    risk = np.clip((w_drop*drop01 + w_ratio*ratio01 + w_var*var01) / wsum, 0, 1).astype("float32")

    # --- 3) Reproject to EPSG:4326 + downsample for web
    dst_crs = rasterio.crs.CRS.from_epsg(4326)
    if src_crs == dst_crs:
        risk4326 = risk.copy()
        dst_tr, dst_w, dst_h = src_tr, W, H
    else:
        dst_tr, dst_w, dst_h = calculate_default_transform(src_crs, dst_crs, W, H, *array_bounds(H, W, src_tr))
        risk4326 = np.full((dst_h, dst_w), np.nan, "float32")
        reproject(
            risk, risk4326,
            src_transform=src_tr, src_crs=src_crs,
            dst_transform=dst_tr, dst_crs=dst_crs,
            dst_nodata=np.nan, resampling=Resampling.bilinear
        )

    if dst_w > max_web_width:
        scale = max_web_width / float(dst_w)
        web_w, web_h = max_web_width, max(1, int(round(dst_h*scale)))
    else:
        web_w, web_h = dst_w, dst_h

    web_tr = from_bounds(*array_bounds(dst_h, dst_w, dst_tr), width=web_w, height=web_h)
    risk_web = np.full((web_h, web_w), np.nan, "float32")
    reproject(
        risk4326, risk_web,
        src_transform=dst_tr, src_crs=dst_crs,
        dst_transform=web_tr, dst_crs=dst_crs,
        dst_nodata=np.nan, resampling=Resampling.max
    )

    # Build a proper profile for the web grid (EPSG:4326, float32, nodata)
    profile_web = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": web_tr,
        "width": int(risk_web.shape[1]),
        "height": int(risk_web.shape[0]),
        "tiled": True,
        "compress": "deflate",
        "predictor": 2,
        "blockxsize": 256,
        "blockysize": 256,
    }

    # Optional: embed field id into filename if you have it (else just tag)
    field_id_for_name = None
    try:
        # If your timeseries filename is in the job result and looks like timeseries_field_<id>_*.csv
        pass  # (leave as None if not easily available here)
    except Exception:
        pass

    # Pull field id from stack filename if present: stack_field_<id>_*.tif
    m = re.search(r"field_(\d+)", str(stack_tif_path))
    field_id_for_name = m.group(1) if m else None

    risk_tag = uuid.uuid4().hex[:8]
    risk_name = (f"risk_field_{field_id_for_name}_{risk_tag}.tif"
                if field_id_for_name else f"risk_{risk_tag}.tif")

    ov_dir = Path("media") / "overlays"
    ov_dir.mkdir(parents=True, exist_ok=True)
    risk_tif_abs = ov_dir / risk_name

    with rasterio.open(str(risk_tif_abs), "w", **profile_web) as dst:
        dst.write(np.nan_to_num(risk_web, nan=-9999.0).astype("float32"), 1)

    risk_tif_url = f"/media/overlays/{risk_name}"

    # --- 4) Save one overlay PNG (no legend; clean for Leaflet ImageOverlay)
    tag = uuid.uuid4().hex[:8]
    out_png = OVER_DIR / f"risk_{tag}.png"
    _save_png01(risk_web, str(out_png))

    # --- 4b) Save a compact client-side probe (risk_web -> Uint16 + meta)
    PROBE_DIR = MEDIA_ROOT / "probes"
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe_tag = uuid.uuid4().hex[:8]
    probe_bin = PROBE_DIR / f"probe_{probe_tag}.bin"
    probe_json = PROBE_DIR / f"probe_{probe_tag}.json"

    # pack risk_web (float32 0..1) into uint16 [0..1000]
    risk_clip = np.clip(risk_web, 0, 1).astype("float32")
    risk_u16 = np.round(risk_clip * 1000).astype("uint16")
    # WRITE (overwrite) data bytes
    with probe_bin.open("wb") as f:
        f.write(risk_u16.tobytes(order="C"))

    # optional mask: inside AOI = 1, else 0 (keeps tooltip quiet outside)
    aoi_mask_web = None
    try:
        from shapely.geometry import shape, Point
        poly = shape(aoi_geojson["geometry"]) if aoi_geojson.get("type")=="Feature" else shape(aoi_geojson)
        aoi_mask_web = np.zeros_like(risk_u16, dtype="uint8")
        (south, west), (north, east) = array_bounds(web_h, web_w, web_tr)[1::-1], array_bounds(web_h, web_w, web_tr)[3:1:-1]
        ys = np.linspace(north - (north-south)/(2*web_h), south + (north-south)/(2*web_h), web_h)
        xs = np.linspace(west + (east-west)/(2*web_w),  east - (east-west)/(2*web_w),  web_w)
        minx, miny, maxx, maxy = poly.bounds
        for r, lat in enumerate(ys):
            if lat < miny or lat > maxy: continue
            for c, lon in enumerate(xs):
                if lon < minx or lon > maxx: continue
                if poly.contains(Point(lon, lat)):
                    aoi_mask_web[r, c] = 1
    except Exception:
        pass

    meta = {
        "rows": int(risk_u16.shape[0]),
        "cols": int(risk_u16.shape[1]),
        "web_bounds": [[float(array_bounds(web_h, web_w, web_tr)[1]),
                        float(array_bounds(web_h, web_w, web_tr)[0])],
                    [float(array_bounds(web_h, web_w, web_tr)[3]),
                        float(array_bounds(web_h, web_w, web_tr)[2])]],
        "scale": 1000,
        "has_mask": bool(aoi_mask_web is not None),
        "layout": {"data_bytes": int(risk_u16.size * 2)}
    }

    if aoi_mask_web is not None:
        # APPEND mask bytes safely
        with probe_bin.open("ab") as f:
            f.write(aoi_mask_web.ravel(order="C").tobytes(order="C"))
        meta["layout"]["mask_bytes"] = int(aoi_mask_web.size)

    probe_json.write_text(json.dumps(meta), encoding="utf-8")

    # --- 5) Hotspots (reuse your module; accurate areas computed on source grid)
    field_geom = None
    try:
        if aoi_geojson.get("type") == "Feature":
            field_geom = shape(aoi_geojson["geometry"])
        else:
            field_geom = shape(aoi_geojson)
    except Exception:
        field_geom = None

    hs = {}
    try:
        hs = extract_hotspots(
            risk=risk,
            src_transform=src_tr, src_crs=src_crs,
            field_polygon=field_geom,
            comps=[drop01, ratio01, var01],
            weights=[0.5, 0.3, 0.2],
            HOTSPOT_PERCENTILE=80,
            MIN_HOTSPOT_AREA_PIX=10,
            MAX_HOTSPOTS=25,
            OUT_GEOJSON=str(HOT_DIR / f"hotspots_{tag}.geojson")
        )
    except Exception as e:
        print("⚠️ Hotspot extraction failed:", e)
        hs = {}

    # Safely build hotspots_url
    from pathlib import Path as _P
    hs_path = _P(hs.get("geojson_path") or hs.get("filename", ""))
    hotspots_url = f"/media/hotspots/{hs_path.name}" if hs_path.name else ""

    return {
        "overlay_png_url": f"/media/overlays/{out_png.name}",
        "hotspots_url": hotspots_url,
        "bounds": _bounds_from_geom(aoi_geojson),
        "probe_bin_url": f"/media/probes/{probe_bin.name}",
        "probe_meta_url": f"/media/probes/{probe_json.name}",
        "risk_tif_path": str(risk_tif_abs),
        "risk_tif_url": risk_tif_url,
    }

