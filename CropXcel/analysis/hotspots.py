# PLACE INTO: CropXcel/analysis/hotspots.py

# =========================
# Cell 1: imports & constants
# =========================
import io, base64, json, warnings
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from skimage.measure import label, regionprops
from rasterio.transform import xy as rio_xy
from rasterio.features import rasterize
from pyproj import Transformer, CRS

# Optional denoise (salt-and-pepper). It'll be skipped if SciPy is missing.
try:
    from scipy.ndimage import binary_opening, generate_binary_structure
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    warnings.warn("[Hotspots] SciPy not found; morphology denoise will be skipped.")

# Farmer-facing labels (3 typical S1 reasons)
FRIENDLY_LABELS = ["Radar drop (water)", "VH/VV ratio change", "Signal variability"]

# Default knobs (you may override in the function call)
DEFAULTS = dict(
    HOTSPOT_PERCENTILE    = 90,     # top 10%
    MIN_HOTSPOT_AREA_PIX  = 5,      # minimum region area (pixels)
    MAX_HOTSPOTS          = 10,     # cap
    OUT_GEOJSON           = "hotspots.geojson",
    USE_POLYGONS_GEOJSON  = False   # export footprints instead of centroids (advanced)
)

# =========================
# Cell 2: helpers
# =========================
def top_reason_label(idx_or_name):
    if isinstance(idx_or_name, int):
        return FRIENDLY_LABELS[idx_or_name] if 0 <= idx_or_name < len(FRIENDLY_LABELS) else str(idx_or_name)
    m = {
        "SAR_water_drop": "Radar drop (water)",
        "SAR_ratio_change": "VH/VV ratio change",
        "SAR_variability": "Signal variability",
    }
    return m.get(str(idx_or_name), str(idx_or_name))

def risk_level(r):  # r in [0,1]
    if r >= 0.70: return "High"
    if r >= 0.40: return "Caution"
    return "Low"

def action_for(level):
    return {
        "High":   "Check field now. Drain standing water if possible (open outlets or pump).",
        "Caution":"Monitor in next 1–2 days. Walk field edges; prepare drainage.",
        "Low":    "All good. Keep routine checks."
    }[level]

def area_from_pixels(area_px, transform):
    """
    Pixel area (m²) robust to rotation/shear.
    Per-pixel area = |a*e - b*d| from the 2x2 upper-left of the affine.
    """
    px_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    area_m2 = float(area_px) * px_area_m2
    area_ha = area_m2 / 10_000.0
    return area_m2, area_ha

def tiny_bar_png_friendly(values, labels=None, title="Why risky?", width=420, height=220, dpi=120):
    import matplotlib
    matplotlib.rcParams["figure.max_open_warning"] = 0
    import numpy as np
    import matplotlib.pyplot as plt

    vals = np.asarray(values, dtype=float)
    vals = np.clip(np.nan_to_num(vals, nan=0.0), 0.0, 1.0)

    denom = vals.sum()
    pct = np.zeros_like(vals) if denom <= 1e-9 else (vals / denom) * 100.0

    if labels is None:
        labels = FRIENDLY_LABELS[:len(vals)]
    else:
        labels = list(labels)

    palette = ["#d73027", "#fdae61", "#4575b4"]  # water, ratio, variability
    colors = [palette[i % len(palette)] for i in range(len(pct))]

    fig_w, fig_h = width / dpi, height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    y = np.arange(len(pct))
    display_pct = np.maximum(pct, 2.0)  # enforce minimum visible bar
    bars = ax.barh(y, display_pct, color=colors, height=0.45, edgecolor="#333", linewidth=0.6)

    for i, b in enumerate(bars):
        ax.text(b.get_width() + 1.0, b.get_y() + b.get_height()/2,
                f"{pct[i]:.0f}%", va="center", fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Contribution (%)", fontsize=9)
    ax.set_title(title, fontsize=11, weight="600", pad=6)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# =========================
# Cell 3: AOI reprojection & mask
# =========================
def _to_src_crs_geom(field_polygon, src_crs):
    """Return a shapely geometry in src_crs (GeoDataFrame -> to_crs)."""
    if field_polygon is None:
        return None

    if isinstance(field_polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
        gdf = field_polygon.copy()
        if gdf.crs is None:
            # If unknown, assume WGS84. Change this if you know the true CRS.
            gdf.set_crs("EPSG:4326", inplace=True)
    else:
        # bare shapely geometry -> assume WGS84 unless you know otherwise
        gdf = gpd.GeoDataFrame(geometry=[field_polygon], crs="EPSG:4326")

    # robust CRS comparison/convert
    src_crs_norm = CRS.from_user_input(src_crs)
    if CRS.from_user_input(gdf.crs) != src_crs_norm:
        gdf = gdf.to_crs(src_crs_norm)

    return gdf.unary_union  # shapely geometry

def build_aoi_mask(field_polygon, src_crs, src_transform, out_shape):
    """Rasterize field polygon in src_crs onto the risk grid (out_shape)."""
    geom_src = _to_src_crs_geom(field_polygon, src_crs)
    if geom_src is None:
        return None
    mask = rasterize(
        [(geom_src, 1)],
        out_shape=out_shape,
        transform=src_transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    ).astype(bool)
    return mask

# =========================
# Cell 4: main extraction
# =========================
def extract_hotspots(
    risk,                 # 2D numpy array, values in [0,1] (or any continuous risk proxy)
    src_transform,        # rasterio Affine for 'risk'
    src_crs,              # CRS string or rasterio CRS (e.g., "EPSG:32648")
    field_polygon=None,   # shapely/GeoSeries/GeoDataFrame (AOI); optional
    comps=None,           # list/array of 2D arrays (same HxW as risk) for contributions; optional
    weights=None,         # 1D list/array of weights for comps; optional
    HOTSPOT_PERCENTILE=DEFAULTS["HOTSPOT_PERCENTILE"],
    MIN_HOTSPOT_AREA_PIX=DEFAULTS["MIN_HOTSPOT_AREA_PIX"],
    MAX_HOTSPOTS=DEFAULTS["MAX_HOTSPOTS"],
    OUT_GEOJSON=DEFAULTS["OUT_GEOJSON"],
    USE_POLYGONS_GEOJSON=DEFAULTS["USE_POLYGONS_GEOJSON"],
):
    """
    Returns: dict with keys {hotspots: list[dict], geojson_path: str|None}
    """
    # ---- sanity on risk ----
    risk = np.asarray(risk)
    assert risk.ndim == 2, "risk must be a 2D array"
    H, W = risk.shape

    finite = risk[np.isfinite(risk)]
    if finite.size == 0:
        print("[INFO] risk has no finite values; skipping hotspots.")
        return {"hotspots": [], "geojson_path": None}

    rmin, rmax = float(finite.min()), float(finite.max())
    print(f"[DEBUG] risk_range=({rmin:.5f}, {rmax:.5f})")

    if np.allclose(rmin, rmax, atol=1e-6):
        print("[INFO] risk is (nearly) constant; nothing stands out.")
        return {"hotspots": [], "geojson_path": None}

    # ---- AOI mask (may be None) ----
    aoi_mask = build_aoi_mask(field_polygon, src_crs, src_transform, out_shape=risk.shape)
    if aoi_mask is not None:
        print(f"[DEBUG] AOI_coverage_pixels={int(aoi_mask.sum())} ({aoi_mask.sum() / (H*W):.2%})")

    # ---- primary threshold (AOI-aware) ----
    finite_all = risk[np.isfinite(risk)]
    if aoi_mask is not None:
        finite_in = risk[aoi_mask & np.isfinite(risk)]
        finite_use = finite_in if finite_in.size > 0 else finite_all
    else:
        finite_use = finite_all

    thr = np.nanpercentile(finite_use, HOTSPOT_PERCENTILE)
    cand_mask = np.where(np.isnan(risk), False, risk >= thr)
    mask_h = cand_mask if aoi_mask is None else (cand_mask & aoi_mask)
    cand_in_aoi = int(mask_h.sum())
    print(f"[DEBUG] thr@p{HOTSPOT_PERCENTILE}={thr:.5f}, candidates_in_AOI={cand_in_aoi}")

    # ---- fallback if AOI+thr removes everything (also AOI-aware) ----
    if cand_in_aoi == 0:
        thr2 = np.nanpercentile(finite_use, 75.0)
        mask_h = np.where(np.isnan(risk), False, risk >= thr2)
        if aoi_mask is not None:
            mask_h = mask_h & aoi_mask
        print(f"[DEBUG] relaxed thr@p75={thr2:.5f}, candidates_in_AOI={int(mask_h.sum())}")

    # ---- optional morphology opening (denoise) ----
    if _HAS_SCIPY and mask_h.any():
        st = generate_binary_structure(2, 2)  # 8-connectivity
        mask_h = binary_opening(mask_h, structure=st, iterations=1)

    print(f"[DEBUG] final_candidates_in_mask={int(mask_h.sum())}")
    if not mask_h.any():
        print("[INFO] No hotspots (after fallback/denoise).")
        return {"hotspots": [], "geojson_path": None}

    # ---- connected components & filter by area ----
    lbl = label(mask_h, connectivity=2)
    regs = [r for r in regionprops(lbl) if r.area >= max(3, MIN_HOTSPOT_AREA_PIX)]
    print(f"[DEBUG] regions_after_area_filter={len(regs)}")

    if not regs:
        print("[INFO] No regions >= MIN_HOTSPOT_AREA_PIX.")
        return {"hotspots": [], "geojson_path": None}

    # ---- components / weights (optional explanations) ----
    comps_stack = None
    weights_arr = None
    if comps is not None and weights is not None:
        comps_stack = np.dstack(comps).transpose(2, 0, 1).astype("float32")  # (C,H,W)
        weights_arr = np.array(weights, dtype="float32")
        assert comps_stack.shape[1:] == risk.shape, "components must match risk shape"
        assert weights_arr.ndim == 1 and weights_arr.shape[0] == comps_stack.shape[0], \
            "weights length must equal #components"

    # ---- centroid -> lon/lat transformer (if needed) ----
    need_reproj = CRS.from_user_input(src_crs) != CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True) if need_reproj else None

    # ---- score & explain each region ----
    scored = []
    for r in regs:
        region_mask = (lbl == r.label)
        mean_risk = float(np.nanmean(risk[region_mask]))

        if comps_stack is not None:
            comp_means = np.array([float(np.nanmean(c[region_mask])) for c in comps_stack], dtype="float32")
            comp_means = np.clip(np.nan_to_num(comp_means, nan=0.0), 0.0, 1.0)
            wsum = weights_arr.sum() if weights_arr is not None else 0.0
            w_safe = weights_arr / (wsum if wsum > 1e-9 else 1.0)
            weighted = comp_means * w_safe
            top_idx = int(np.argmax(weighted))
        else:
            comp_means = np.array([mean_risk], dtype="float32")
            top_idx = 0

        scored.append((mean_risk, r.area, r, comp_means, top_idx))

    # sort by mean risk, then area
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if MAX_HOTSPOTS is not None:
        scored = scored[:MAX_HOTSPOTS]

    # ---- build output records ----
    hotspots = []
    outside = 0
    for mean_risk, area_px, r, comp_means, top_idx in scored:
        rr, cc = r.centroid  # (row, col) in array coords (float)
        x_src, y_src = rio_xy(src_transform, rr, cc, offset="center")  # center of that pixel

        if need_reproj:
            lon, lat = transformer.transform(x_src, y_src)
        else:
            lon, lat = float(x_src), float(y_src)

        # sanity: skip impossible coords
        if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
            outside += 1
            continue

        area_m2, area_ha = area_from_pixels(area_px, src_transform)
        level = risk_level(mean_risk)
        reason = top_reason_label(top_idx)

        chart_b64 = tiny_bar_png_friendly(
            values=(comp_means).tolist(),
            labels=(FRIENDLY_LABELS[:len(comp_means)] if len(comp_means) > 1 else ["Overall risk"]),
            title="Why risky?"
        )

        hotspots.append({
            "lat": float(lat), "lon": float(lon),
            "risk": float(mean_risk),
            "risk_pct": int(round(mean_risk * 100)),
            "level": level,
            "pixels": int(area_px),
            "area_m2": round(area_m2, 1),
            "area_ha": round(area_ha, 4),
            "reason": reason,
            "action": action_for(level),
            "chart_b64": chart_b64
        })

    if outside:
        print(f"[WARN] skipped {outside} hotspot(s) with invalid/out-of-range coordinates")

    # ---- GeoJSON export ----
    geojson_path = None
    if hotspots:
        if USE_POLYGONS_GEOJSON:
            # Export centroids as points first (simple & robust).
            # For polygon footprints, you can extend here if needed later.
            print("[INFO] USE_POLYGONS_GEOJSON=True requested, but polygon export is not enabled in this minimal build.")
        gdf = gpd.GeoDataFrame(
            [{
                "geometry": Point(h["lon"], h["lat"]),
                "risk_pct": h["risk_pct"],
                "level": h["level"],
                "area_m2": h["area_m2"],
                "area_ha": h["area_ha"],
                "reason": h["reason"],
                "action": h["action"]
            } for h in hotspots],
            crs="EPSG:4326"
        )
        geojson_path = OUT_GEOJSON
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"[OK] {geojson_path} ({len(gdf)} hotspots)")
    else:
        print("[INFO] No hotspots found (after region scoring).")

    return {"hotspots": hotspots, "geojson_path": geojson_path}