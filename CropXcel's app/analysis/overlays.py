# PLACE INTO: CropXcel/analysis/overlays.py

import os, io, base64, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import rasterio
from rasterio.transform import xy, array_bounds, from_bounds
from rasterio.warp import (calculate_default_transform, reproject,
                           Resampling, transform as crs_transform)

from skimage.measure import label, regionprops
import folium
from folium.raster_layers import ImageOverlay
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

# ------- Inputs -------
MULTIBAND_PATH = tif_path                 # 11-band S1 stack you exported earlier
TIME_SERIES_CSV = OUTPUT_CSV              # AOI mean time series csv

# Band indices (1-based) matching your exported order
BANDS = {
    "S1_VV_CURR":        1,   # linear power
    "S1_VH_CURR":        2,   # linear power
    "S1_VH_VV_CURR":     3,   # unitless ratio (VH/VV)
    "S1_VV_BASE":        4,   # linear power
    "S1_VH_BASE":        5,   # linear power
    "S1_VH_VV_BASE":     6,   # unitless ratio
    "S1_VV_LOGRATIO_DB": 7,   # dB
    "S1_VH_LOGRATIO_DB": 8,   # dB
    "S1_VH_VV_DIFF":     9,   # unitless (ratio diff)
    "S1_VV_STD":         10,  # linear σ
    "S1_VH_STD":         11,  # linear σ
}
BAND_COUNT_EXPECTED = 11

# Optional field boundary overlay (one path is enough)
FIELD_GEOJSON = r""
FIELD_SHP     = r""

# Contributor weights (set to 0.0 if layer not present)
WEIGHTS = {
    "sar_water_drop":  0.40,  # dominated by LOGRATIO_DB (VH mostly)
    "sar_ratio_change":0.25,  # from S1_VH_VV_DIFF (and/or ratio deltas)
    "sar_variability": 0.15,  # from S1_*_STD
    "water_extent":    0.15,  # requires MNDWI/water mask (set to 0.0 if unavailable)
    "veg_signal":      0.05,  # requires NDVI/VARI (set to 0.0 if unavailable)
}

# Hotspot extraction
HOTSPOT_PERCENTILE   = 80      # higher → fewer, stronger hotspots
MIN_HOTSPOT_AREA_PIX = 30
MAX_HOTSPOTS         = 30

# Web layout
MAX_WEB_WIDTH = 1400

# Outputs (consistent naming)
_base = os.path.splitext(os.path.basename(MULTIBAND_PATH))[0]
OUT_TIF     = f"waterlogging_monitoring_{_base}.tif"
OUT_PNG_NC  = f"waterlogging_monitoring_nolegend_{_base}.png"
OUT_PNG_CB  = f"waterlogging_monitoring_colorbar_{_base}.png"
OUT_HTML    = f"waterlogging_monitoring_{_base}.html"
OUT_GEOJSON = f"hotspots_{_base}.geojson"


# Temporal/alert products (if your temporal engine writes them)
ALERTS_CSV      = f"alerts_{_base}.csv"
ALERTS_PLOT_PNG = f"S1_alerts_plot_{_base}.png"
RECS_CSV        = f"recommendations_{_base}.csv"

# Optional temporal knobs (used by your separate time engine)
ROLL_WINDOW_DAYS       = 60
BASELINE_WINDOWS_DAYS  = [12, 24, 36]

# Thresholds for change logic (sign conventions)
MAX_DROP_DB_VH   = -1.5   # flag if VV/VH logratio (dB) <= this
MAX_DROP_DB_VV   = -1.0
MIN_PCT_DROP_LIN = 0.15   # 15% linear drop (if you also test linear deltas)
Z_THRESHOLD      = -1.5
MIN_CONSECUTIVE  = 2

INTERPOLATE_TIME = True
INTERP_LIMIT     = 1

# Panel side
PANEL_ANCHOR = "right"

# --- Sanity checks ---
if not os.path.exists(MULTIBAND_PATH):
    raise FileNotFoundError(f"Raster not found: {MULTIBAND_PATH}")
if not os.path.exists(TIME_SERIES_CSV):
    raise FileNotFoundError(f"Time-series CSV not found: {TIME_SERIES_CSV}")

# Optional: band count check
with rasterio.open(MULTIBAND_PATH) as _src_chk:
    if _src_chk.count != BAND_COUNT_EXPECTED:
        print(f"[WARN] Found {_src_chk.count} bands (expected {BAND_COUNT_EXPECTED}). "
              "Make sure BANDS indices match actual order.")

def pct_stretch(a, pmin=2, pmax=98):
    """Percentile stretch to 0..1 on finite values; returns float32."""
    if a is None:
        return None
    a = a.astype("float32")
    # Use nanpercentile so NaNs are ignored automatically
    lo, hi = np.nanpercentile(a, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(a*0, 0, 1).astype("float32")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype("float32")

# Keep these thin wrappers if you like the names:
def flood_anomaly_index(a, pmin=5, pmax=95):
    return pct_stretch(a, pmin, pmax)

def z_to_01(a, invert=False):
    s = pct_stretch(a)
    return None if s is None else (1 - s if invert else s)

def dual_tail_risk(a):
    s = pct_stretch(a)
    return None if s is None else (np.abs(s - 0.5) * 2).astype("float32")

def finite_percentiles(a, lo=2, hi=98):
    if a is None:
        return 0.0, 1.0
    lo_v, hi_v = np.nanpercentile(a, [lo, hi])
    if not np.isfinite(lo_v) or not np.isfinite(hi_v) or hi_v <= lo_v:
        # fall back to global finite range
        finite = np.isfinite(a)
        if not np.any(finite):
            return 0.0, 1.0
        return float(np.nanmin(a[finite])), float(np.nanmax(a[finite]))
    return float(lo_v), float(hi_v)

def _is_db(x):
    # heuristic: SAR dB typically ~[-30, +5]
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return False
    return (np.nanmin(finite) < -10.0) or (np.nanmedian(finite) < 0.0)

def _to_linear(x):
    # 10^(x/10)
    return np.power(10.0, x / 10.0).astype(np.float32)

with rasterio.open(MULTIBAND_PATH) as src:
    profile = src.profile
    W, H = src.width, src.height
    src_crs = src.crs
    src_transform = src.transform
    src_bounds = src.bounds

    arrays = {}
    for name, idx in BANDS.items():
        # read masked, then fill mask with NaN so downstream math is safe
        band = src.read(idx, masked=True)
        arrays[name] = band.filled(np.nan).astype("float32")

# Derive current and/or baseline VH/VV ratios if missing
def _safe_ratio(numer, denom):
    out = np.full_like(numer, np.nan, dtype=np.float32)
    mask = np.isfinite(numer) & np.isfinite(denom) & (denom > 0)
    out[mask] = (numer[mask] / denom[mask]).astype(np.float32)
    return out

# Current ratio
if "S1_VH_VV_CURR" not in arrays and {"S1_VH_CURR","S1_VV_CURR"}.issubset(arrays):
    vh, vv = arrays["S1_VH_CURR"], arrays["S1_VV_CURR"]
    if _is_db(vh) or _is_db(vv):
        vh, vv = _to_linear(vh), _to_linear(vv)
    arrays["S1_VH_VV_CURR"] = _safe_ratio(vh, vv)

# Baseline ratio
if "S1_VH_VV_BASE" not in arrays and {"S1_VH_BASE","S1_VV_BASE"}.issubset(arrays):
    vhb, vvb = arrays["S1_VH_BASE"], arrays["S1_VV_BASE"]
    if _is_db(vhb) or _is_db(vvb):
        vhb, vvb = _to_linear(vhb), _to_linear(vvb)
    arrays["S1_VH_VV_BASE"] = _safe_ratio(vhb, vvb)

def save_geotiff(path, data, profile, nodata_val=-9999.0):
    p = profile.copy()
    p.update(count=1, dtype="float32", nodata=nodata_val)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.nan_to_num(data, nan=nodata_val).astype("float32"), 1)

def top_reason_label(key: str):
    return {
        "sar_water_drop":  "Sharp drop in radar backscatter (possible inundation).",
        "sar_ratio_change":"Change in VH/VV ratio (flooded vegetation vs. dry).",
        "sar_variability":"High temporal variability (speckle/patchy water).",
        "water_extent":    "Surface water extent (from MNDWI when available).",
        "veg_signal":      "Low greenness (NDVI/VARI)—possible stress."
    }.check(key, "Inspect this zone.")


def risk_cmap_redgreen():
    # green = low risk, red = high risk
    return ListedColormap([
        "#006400",  # dark green
        "#228B22",  # medium green
        "#ADFF2F",  # yellow-green
        "#FFFF00",  # yellow
        "#FFA500",  # orange
        "#FF4500",  # orange-red
        "#B22222",  # firebrick
        "#8B0000"   # dark red
    ])

def tiny_bar_png(values, labels, title="Contributors"):
    fig, ax = plt.subplots(figsize=(2.8, 1.8), dpi=150)
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def embed_image_base64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def lin_to_db_vis(arr, vmin=-25, vmax=-5):
    """
    Convert linear power → dB safely and clip for visualization.
    Args:
        arr  : numpy array (linear values, can contain NaN/0)
        vmin : lower bound in dB (default -25)
        vmax : upper bound in dB (default -5)
    Returns:
        float32 array in dB, clipped to [vmin, vmax]
    """
    db = 10.0 * np.log10(np.maximum(arr, 1e-9)).astype("float32")  # safe dB
    return np.clip(db, vmin, vmax)

# Read the raster data first (masked→NaN)
with rasterio.open(MULTIBAND_PATH) as src:
    profile = src.profile
    W, H = src.width, src.height
    src_crs = src.crs
    src_transform = src.transform
    src_bounds = src.bounds
    
    arrays = {}
    for name, idx in BANDS.items():
        band = src.read(idx, masked=True)              # <- respect mask
        arrays[name] = band.filled(np.nan).astype("float32")

def _is_db(x: np.ndarray) -> bool:
    # crude but effective: S1 backscatter dB usually ~[-30, +5]
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return False
    return (np.nanmin(finite) < -10.0) or (np.nanmedian(finite) < 0.0)

def _to_linear(x: np.ndarray) -> np.ndarray:
    # 10^(x/10) — cast after; np.power has no dtype kwarg
    return np.power(10.0, x / 10.0).astype(np.float32)

def _safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full_like(numer, np.nan, dtype=np.float32)
    m = np.isfinite(numer) & np.isfinite(denom) & (denom > 0)
    out[m] = (numer[m] / denom[m]).astype(np.float32)
    return out

# --- derive VH/VV ratios for CURRENT/BASE if missing ---
# Current
if ("S1_VH_VV_CURR" not in arrays) and {"S1_VH_CURR","S1_VV_CURR"}.issubset(arrays):
    vh, vv = arrays["S1_VH_CURR"], arrays["S1_VV_CURR"]
    if _is_db(vh) or _is_db(vv):
        vh, vv = _to_linear(vh), _to_linear(vv)
    arrays["S1_VH_VV_CURR"] = _safe_ratio(vh, vv)

# Baseline
if ("S1_VH_VV_BASE" not in arrays) and {"S1_VH_BASE","S1_VV_BASE"}.issubset(arrays):
    vhb, vvb = arrays["S1_VH_BASE"], arrays["S1_VV_BASE"]
    if _is_db(vhb) or _is_db(vvb):
        vhb, vvb = _to_linear(vhb), _to_linear(vvb)
    arrays["S1_VH_VV_BASE"] = _safe_ratio(vhb, vvb)

def pct_stretch(a, pmin=2, pmax=98):
    """Percentile stretch to 0..1 on finite values; returns float32."""
    if a is None:
        return None
    a = a.astype("float32")
    lo, hi = np.nanpercentile(a, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(a*0, 0, 1).astype("float32")
    return np.clip((a - lo) / (hi - lo), 0, 1).astype("float32")

# ----- SAR-first contributors (robust to missing bands) -----

def stretch01(x, invert=False):
    s = pct_stretch(x)
    return None if s is None else (1 - s if invert else s)

contributors = []

# 1) SAR water drop (prefer VH; use VV too). More negative logratio dB = riskier
drop_parts = []
if "S1_VH_LOGRATIO_DB" in arrays:
    drop_parts.append((-arrays["S1_VH_LOGRATIO_DB"]).astype("float32"))  # negate: drop -> high
if "S1_VV_LOGRATIO_DB" in arrays:
    drop_parts.append((-arrays["S1_VV_LOGRATIO_DB"]).astype("float32"))

if drop_parts:
    # combine VH/VV then stretch once
    sar_water_drop = pct_stretch(np.nanmax(np.dstack(drop_parts), axis=2))
    if sar_water_drop is not None and np.any(np.isfinite(sar_water_drop)):
        contributors.append(("sar_water_drop", WEIGHTS.get("sar_water_drop", 0.40), sar_water_drop))

# 2) SAR ratio change (flooded veg alters VH/VV). Prefer diff vs baseline; else anomaly of current ratio
sar_ratio_change = None
if "S1_VH_VV_DIFF" in arrays:
    sar_ratio_change = pct_stretch(np.abs(arrays["S1_VH_VV_DIFF"]))
else:
    # Fallback: anomaly from median of current ratio if present
    for rk in ("S1_VH_VV_CURR", "S1_VH_VV_BASE"):
        if rk in arrays:
            rr = arrays[rk].astype("float32")
            sar_ratio_change = pct_stretch(np.abs(rr - np.nanmedian(rr)))
            break
if sar_ratio_change is not None and np.any(np.isfinite(sar_ratio_change)):
    contributors.append(("sar_ratio_change", WEIGHTS.get("sar_ratio_change", 0.25), sar_ratio_change))

# 3) SAR variability (patchy/unstable wetness)
var_parts = []
if "S1_VH_STD" in arrays: var_parts.append(arrays["S1_VH_STD"].astype("float32"))
if "S1_VV_STD" in arrays: var_parts.append(arrays["S1_VV_STD"].astype("float32"))
if var_parts:
    sar_variability = pct_stretch(np.nanmax(np.dstack(var_parts), axis=2))
    if sar_variability is not None and np.any(np.isfinite(sar_variability)):
        contributors.append(("sar_variability", WEIGHTS.get("sar_variability", 0.15), sar_variability))

# 4) Optional: water extent (optical fallback when available)
water_extent = None
for opt_key in ("MNDWI", "NDWI"):
    if opt_key in arrays:
        water_extent = stretch01(arrays[opt_key])  # high = more water
        break
if water_extent is not None and np.any(np.isfinite(water_extent)):
    contributors.append(("water_extent", WEIGHTS.get("water_extent", 0.15), water_extent))

# 5) Optional: veg signal (only if NDVI available)
if "NDVI" in arrays:
    ndvi_low = stretch01(arrays["NDVI"], invert=True)  # low NDVI -> higher risk
    if ndvi_low is not None and np.any(np.isfinite(ndvi_low)):
        contributors.append(("veg_signal", WEIGHTS.get("veg_signal", 0.05), ndvi_low))

# Safety check
if not contributors:
    raise RuntimeError("No valid contributors found for SAR waterlogging. Check band list & weights.")

# ----- Weighted blend (normalize weights safely) -----
labels, weights_raw, comps = zip(*contributors)
weights = np.asarray(weights_raw, dtype="float32")
w_sum = float(np.nansum(weights))
if w_sum <= 0:
    # fallback: equal weights
    weights = np.full(len(comps), 1.0/len(comps), dtype="float32")
else:
    weights /= w_sum

stack = np.dstack(comps)  # H x W x K
mask = np.isfinite(stack)
w_b  = weights.reshape((1,1,-1)) * mask
num  = np.nansum(stack * w_b, axis=2)
den  = np.sum(w_b, axis=2)
risk = np.divide(num, den, out=np.full_like(num, np.nan, dtype="float32"), where=den>0)
risk = np.clip(risk, 0, 1).astype("float32")

save_geotiff(OUT_TIF, risk, profile)
print(f"[OK] {OUT_TIF}")

dst_crs = rasterio.crs.CRS.from_epsg(4326)

# Build target grid (4326) from source bounds
dst_transform, dst_w, dst_h = calculate_default_transform(
    src_crs, dst_crs, W, H, *src_bounds  # src_bounds=(left,bottom,right,top)
)

def _as_masked(a: np.ndarray) -> np.ma.MaskedArray:
    # mask non-finite values so resamplers ignore them
    return np.ma.array(a, mask=~np.isfinite(a))

def reproj(src_arr, resampling=Resampling.bilinear):
    out = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(
        source=_as_masked(src_arr),            # <-- masked source
        destination=out,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        # let mask drive nodata handling; don't pass src_nodata=np.nan
        dst_nodata=np.nan,
        resampling=resampling
    )
    return out

# Skip reproj if already 4326
if src_crs == dst_crs:
    risk4326 = risk.astype("float32", copy=True)
    dst_transform, dst_w, dst_h = src_transform, W, H
else:
    risk4326 = reproj(risk, resampling=Resampling.bilinear)

# Downsample for web
if dst_w > MAX_WEB_WIDTH:
    scale = MAX_WEB_WIDTH / float(dst_w)
    web_w = MAX_WEB_WIDTH
    web_h = max(1, int(round(dst_h * scale)))  # guard >=1
    # choose downsample method for risk
    DOWNSAMPLE = Resampling.max   # or Resampling.average if you prefer smoothing
else:
    web_w, web_h = dst_w, dst_h
    DOWNSAMPLE = Resampling.bilinear

bounds4326 = array_bounds(dst_h, dst_w, dst_transform)  # (left,bottom,right,top)
web_transform = from_bounds(*bounds4326, width=web_w, height=web_h)

risk_web = np.full((web_h, web_w), np.nan, dtype="float32")
reproject(
    source=_as_masked(risk4326),               # <-- masked source again
    destination=risk_web,
    src_transform=dst_transform, src_crs=dst_crs,
    dst_transform=web_transform, dst_crs=dst_crs,
    dst_nodata=np.nan,
    resampling=DOWNSAMPLE
)

from matplotlib.colors import ListedColormap

def risk_cmap_redgreen():
    # green (low) → yellow → orange → red (high)
    return ListedColormap([
        "#006400", "#228B22", "#7FFF00", "#FFFF00",
        "#FFA500", "#FF7F50", "#FF4500", "#8B0000"
    ])

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import imageio

def save_overlay_png(arr, out_png, *, aoi_mask=None, fixed01=True,
                     with_colorbar=False, title=None, outside_mode="transparent",
                     red_rgba=(220, 73, 47, 255), feather_px=0, cmap=None):
    """
    Save array as PNG overlay. If aoi_mask is provided (bool array), pixels
    outside are transparent or tinted red.
    """
    if cmap is None:
        cmap = risk_cmap_redgreen()

    # --- normalize ---
    finite = np.isfinite(arr)
    if fixed01:
        vmin, vmax = 0.0, 1.0
    else:
        if finite.any():
            lo, hi = np.percentile(arr[finite], [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
        else:
            lo, hi = 0.0, 1.0
        vmin, vmax = float(lo), float(hi)

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(arr, bytes=True)

    # mask NaNs transparent
    rgba[~finite, 3] = 0

    if aoi_mask is not None:
        if aoi_mask.shape != arr.shape:
            raise ValueError("AOI mask shape does not match array.")
        outside = ~aoi_mask
        if outside_mode == "transparent":
            rgba[outside, 3] = 0
        elif outside_mode == "red":
            rr, gg, bb, aa = red_rgba
            rgba[outside, 0] = rr; rgba[outside, 1] = gg
            rgba[outside, 2] = bb; rgba[outside, 3] = aa

        # optional feather only if we have a mask
        if feather_px and feather_px > 0:
            try:
                from scipy.ndimage import distance_transform_edt
                # distance to outside (edge softening)
                dist_in = distance_transform_edt(aoi_mask)
                edge_band = np.clip(dist_in / float(feather_px), 0.0, 1.0)
                rgba[..., 3] = (rgba[..., 3].astype(float) * edge_band).astype(np.uint8)
            except Exception:
                pass

    if with_colorbar:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        ax.imshow(rgba)
        ax.axis("off")
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          orientation="horizontal", fraction=0.046, pad=0.04)
        if title: cb.set_label(title)
        fig.savefig(out_png, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        imageio.imwrite(out_png, rgba)

from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon

def build_aoi_mask(field_gdf, transform, out_shape, target_crs):
    """
    Rasterize AOI polygon onto the same grid as the array you will overlay.
    """
    gdf = field_gdf.to_crs(target_crs)   # reproject to raster CRS
    geom = gdf.unary_union
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return rasterize(
        [(g.__geo_interface__, 1) for g in geoms],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8"
    ).astype(bool)

# Build field_gdf (robust to MultiPolygon)
field_info = ee.Geometry(field).getInfo()
if field_info["type"] == "Polygon":
    rings = field_info["coordinates"]
    field_polys = [Polygon(rings[0])]
elif field_info["type"] == "MultiPolygon":
    field_polys = [Polygon(r[0]) for r in field_info["coordinates"]]
else:
    # Fallback: take bounds
    coords = ee.Geometry(field).bounds().getInfo()["coordinates"][0]
    field_polys = [Polygon(coords)]

field_gdf = gpd.GeoDataFrame(geometry=field_polys, crs="EPSG:4326")

# IMPORTANT: mask must match risk_web grid (web_transform, dst_crs, risk_web.shape)
aoi_mask_web = build_aoi_mask(field_gdf, web_transform, risk_web.shape, dst_crs)

from rasterio.transform import xy

def latlon_from_transform(t, row, col):
    x, y = xy(t, row, col)  # x=lon, y=lat in EPSG:4326
    return y, x

top_lat, left_lon  = latlon_from_transform(web_transform, 0, 0)
bot_lat, right_lon = latlon_from_transform(web_transform, risk_web.shape[0]-1, risk_web.shape[1]-1)
bounds = [[bot_lat, left_lon], [top_lat, right_lon]]

# With colorbar
save_overlay_png(
    risk_web, OUT_PNG_CB,
    aoi_mask=aoi_mask_web, fixed01=True,
    with_colorbar=True, title="Waterlogging Risk (0–1)",
    outside_mode="transparent", feather_px=2,
    cmap=risk_cmap_redgreen()
)

# No colorbar
save_overlay_png(
    risk_web, OUT_PNG_NC,
    aoi_mask=aoi_mask_web, fixed01=True,
    with_colorbar=False,
    outside_mode="transparent", feather_px=2,
    cmap=risk_cmap_redgreen()
)