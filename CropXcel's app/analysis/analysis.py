# CropXcel/analysis/run_analysis_from_notebook.py
# Purpose: minimal "analysis" like your notebook (no GEE calls here).
# Input: AOI geojson + path to a pre-exported Sentinel-1 stack GeoTIFF.
# Output: PNG overlay + hotspots GeoJSON + bounds (Leaflet order).

from __future__ import annotations
import json, uuid
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds, from_bounds
import imageio
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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

    # --- 4) Save one overlay PNG (no legend; clean for Leaflet ImageOverlay)
    tag = uuid.uuid4().hex[:8]
    out_png = OVER_DIR / f"risk_{tag}.png"
    _save_png01(risk_web, str(out_png))

    # --- 5) Hotspots (reuse your module; accurate areas computed on source grid)
    hs = extract_hotspots(
        risk=risk,
        src_transform=src_tr,
        src_crs=src_crs,
        field_polygon=None,                # AOI optional; you can pass it later if needed
        HOTSPOT_PERCENTILE=80,
        MIN_HOTSPOT_AREA_PIX=10,
        MAX_HOTSPOTS=25,
        OUT_GEOJSON=str(HOT_DIR / f"hotspots_{tag}.geojson")
    )
    hotspots_url = f"/media/hotspots/{Path(hs.get('geojson_path','')).name}" if hs.get("geojson_path") else ""

    # --- 6) Map bounds (Leaflet order) — from AOI, not the raster grid
    bounds = _bounds_from_geom(aoi_geojson)

    return {
        "overlay_png_url": f"/media/overlays/{out_png.name}",
        "hotspots_url": hotspots_url,
        "bounds": bounds
    }