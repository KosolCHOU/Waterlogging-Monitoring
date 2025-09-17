# PLACE INTO: CropXcel/analysis/temporal.py

import ee, datetime as dt
ee.Initialize()

def chirps_totals_mm(aoi: ee.Geometry, end_utc=None):
    end = ee.Date(end_utc or dt.datetime.utcnow())
    start7  = end.advance(-7, 'day')
    start30 = end.advance(-30, 'day')

    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')

    # Build images
    img7  = chirps.filterDate(start7,  end).sum().rename('tot7')
    img30 = chirps.filterDate(start30, end).sum().rename('tot30')

    # Avoid nulls on sparse periods
    img = ee.Image.cat([img7, img30]).unmask(0)

    # One server-side dict; pull it client-side ONCE
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=500,
        maxPixels=1e9,
        tileScale=4
    ).getInfo()  # <-- top-level; OK

    tot7_mm  = float(stats.get('tot7', 0.0)  or 0.0)
    tot30_mm = float(stats.get('tot30', 0.0) or 0.0)
    return tot7_mm, tot30_mm

def chirps_totals_for_fields(fc: ee.FeatureCollection, end_utc=None):
    end = ee.Date(end_utc or dt.datetime.utcnow())
    start7  = end.advance(-7, 'day')
    start30 = end.advance(-30, 'day')

    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select('precipitation')
    img = (chirps.filterDate(start7, end).sum().rename('tot7')
                     .addBands(chirps.filterDate(start30, end).sum().rename('tot30'))
                     .unmask(0))

    out_fc = img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=500,
        tileScale=4
    )
    # Safe: pull once (if the collection is reasonably small). For large sets → Export.table.toDrive
    feats = out_fc.getInfo()['features']
    # Convert to simple list of dicts: {id, tot7, tot30}
    rows = []
    for f in feats:
        props = f.get('properties', {})
        gid = props.get('id') or f.get('id')
        rows.append({
            "id": gid,
            "tot7_mm": float(props.get('tot7') or 0.0),
            "tot30_mm": float(props.get('tot30') or 0.0)
        })
    return rows

s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(field)
      .filterDate(end.advance(-20,'day'), end)
      .filter(ee.Filter.eq('instrumentMode','IW'))
      .filter(ee.Filter.eq('orbitProperties_pass','DESCENDING'))
      .select('VH') )

img = ee.Image(s1.sort('system:time_start', False).first()).focal_median(30)
vh = img.clip(field)

# Otsu threshold on VH backscatter (in dB-like scale)
hist = vh.reduceRegion(ee.Reducer.histogram(255), field, 20).get('VH')
def otsu(hist_dict):
    import numpy as np
    h = np.array(hist_dict['histogram']); b = np.array(hist_dict['bucketMeans'])
    w0 = np.cumsum(h) / h.sum(); w1 = 1 - w0
    m0 = np.cumsum(h*b) / h.sum(); m1 = (m0[-1] - m0) / w1
    sb = w0*w1*(m0-m1)**2; t = b[sb.argmax()]; return float(t)
thr = otsu(ee.Dictionary(hist).getInfo())

water = vh.lte(thr)  # boolean water mask
water_area_ha = water.updateMask(water).multiply(ee.Image.pixelArea()).reduceRegion(
    ee.Reducer.sum(), field, 10).getInfo()['VH'] / 10000.0
print("water_ha:", round(water_area_ha,2))


import requests, pandas as pd, datetime as dt

# Extract centroid coordinates from the field geometry
field_bounds = field.bounds().getInfo()
coords = field_bounds['coordinates'][0]
lats = [coord[1] for coord in coords]
lons = [coord[0] for coord in coords]
lat = sum(lats) / len(lats)  # centroid latitude
lon = sum(lons) / len(lons)  # centroid longitude

url = "https://api.open-meteo.com/v1/forecast"
params = dict(latitude=lat, longitude=lon, hourly="precipitation", forecast_days=3, timezone="Asia/Bangkok")
r = requests.get(url, params=params, timeout=20); r.raise_for_status()
j = r.json()
df = pd.DataFrame({"time": j["hourly"]["time"], "precip_mm": j["hourly"]["precipitation"]})
df["time"] = pd.to_datetime(df["time"])
next24 = df[df["time"] <= df["time"].min() + pd.Timedelta(hours=24)]["precip_mm"].sum()
next48 = df[df["time"] <= df["time"].min() + pd.Timedelta(hours=48)]["precip_mm"].sum()
print("forecast_48h_mm:", round(float(next48),1))
print("forecast_24h_mm:", round(float(next24),1))

def advise(s1_water_ha, imerg_24h, om_48h, sowing_stage: bool):
    # thresholds: tune per province/soil later
    if s1_water_ha > 0.1:  # 0.1 ha flooded inside AOI
        return "RED", "Water detected in field. Delay sowing/pumping; improve drainage."
    if sowing_stage and om_48h >= 30:
        return "RED", "Heavy rain likely (<48h). Protect seed; avoid pumping."
    if imerg_24h >= 25 or om_48h >= 15:
        return "YELLOW", "Rainy window. Monitor; pump only if urgently needed."
    return "GREEN", "Good window for sowing/irrigation."

def compute_temporal_engine_s1(csv_path):
    """
    Temporal engine for Sentinel-1 waterlogging.
    Returns: (alerts_df, insights_df, alerts_plot_png_path, insights_csv_path)
    Side-effect: writes alerts.csv (if any) and insights.csv + a PNG plot.
    """
    # -------------------- imports --------------------
    import os, shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # -------------------- SAFE DEFAULTS --------------------
    ROLL_WINDOW_DAYS    = globals().get("ROLL_WINDOW_DAYS", 30)
    Z_THRESHOLD         = globals().get("Z_THRESHOLD", -1.5)      # more negative = wetter anomaly
    MIN_ABS_DROP_DB_VH  = globals().get("MIN_ABS_DROP_DB_VH", -1.5)
    MIN_ABS_DROP_DB_VV  = globals().get("MIN_ABS_DROP_DB_VV", -1.0)
    MIN_PCT_DROP_LINEAR = globals().get("MIN_PCT_DROP_LINEAR", 0.10)  # ≥10% VH/VV drop
    MIN_CONSECUTIVE     = globals().get("MIN_CONSECUTIVE", 1)
    ALERTS_CSV          = globals().get("ALERTS_CSV", "alerts.csv")
    ALERTS_PLOT_PNG     = globals().get("ALERTS_PLOT_PNG", "alerts_plot.png")

    # -------------------- load & prep --------------------
    if not os.path.exists(csv_path):
        return pd.DataFrame(), pd.DataFrame(), None, None

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), None, None

    # force numeric to avoid stray strings
    for s in ["S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
              "S1_VH_VV_CURR","S1_VH_VV_DIFF","S1_VH_VV_BASE","S1_VH_STD"]:
        if s in df.columns:
            df[s] = pd.to_numeric(df[s], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # choose available columns
    vh      = df.get("S1_VH_CURR")
    vv      = df.get("S1_VV_CURR")
    vh_lrdb = df.get("S1_VH_LOGRATIO_DB")     # curr - base (dB)
    vv_lrdb = df.get("S1_VV_LOGRATIO_DB")
    ratio   = df.get("S1_VH_VV_CURR")         # linear VH/VV
    ratio_d = df.get("S1_VH_VV_DIFF")
    ratio_b = df.get("S1_VH_VV_BASE") if "S1_VH_VV_BASE" in df.columns else None
    vh_std  = df.get("S1_VH_STD")

    # primary series (prefer logratio; else VH dB)
    primary = vh_lrdb if vh_lrdb is not None else vh
    if primary is None or primary.dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    # -------------------- robust z-score --------------------
    def robust_z(series, win_days):
        r = series.rolling(f"{win_days}D", closed="left")
        med = r.median()
        mad = r.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=False)
        z = 0.6745 * (series - med) / mad.replace(0, np.nan)
        return z

    z = robust_z(primary, ROLL_WINDOW_DAYS)

    def ok_z(v):
        # allow NaN early; otherwise require v <= threshold (more negative = wetter)
        return True if pd.isna(v) else bool(v <= Z_THRESHOLD)

    z_ok = z.apply(ok_z)

    # -------------------- rules --------------------
    rule_vh_vs_base = (vh_lrdb <= MIN_ABS_DROP_DB_VH) if vh_lrdb is not None else pd.Series(False, index=df.index)
    rule_vv_vs_base = (vv_lrdb <= MIN_ABS_DROP_DB_VV) if vv_lrdb is not None else pd.Series(False, index=df.index)

    if ratio is not None and ratio_b is not None:
        rule_ratio_pct = ((ratio / ratio_b) - 1.0 <= -MIN_PCT_DROP_LINEAR)
    else:
        rule_ratio_pct = pd.Series(False, index=df.index)

    rule_vh_low_abs = (vh <= -18.0) if vh is not None else pd.Series(False, index=df.index)

    if vh_std is not None:
        s_med = vh_std.rolling(f"{ROLL_WINDOW_DAYS}D", closed="left").median()
        rule_smooth = (vh_std <= s_med * 0.9)
    else:
        rule_smooth = pd.Series(False, index=df.index)

    candidates = rule_vh_vs_base | rule_ratio_pct | rule_vh_low_abs | (rule_vv_vs_base & rule_smooth)
    raw_alerts = candidates & z_ok

    # persistence
    def enforce_persistence(series: pd.Series, k: int) -> pd.Series:
        if k <= 1:
            return series.fillna(False)
        out = series.copy().fillna(False)
        c = 0
        vals = out.values
        for i in range(len(vals)):
            c = (c + 1) if vals[i] else 0
            vals[i] = c >= k
        out[:] = vals
        return out

    alerts_mask = enforce_persistence(raw_alerts, MIN_CONSECUTIVE)

    # -------------------- severity & confidence --------------------
    WATCH_Z = -0.8
    ALERT_Z = Z_THRESHOLD
    WATCH_ANY_DROP = -0.5

    ACTIONS = {
        "Alert":  "Check field now. Drain standing water if possible (open outlets/pump).",
        "Watch":  "Monitor next pass (6–12 days). Walk field edges; avoid over-irrigation.",
        "Healthy":"All good. Keep routine checks."
    }

    def _clip01(x): 
        return float(np.nanmax([0.0, np.nanmin([1.0, x])]))

    def _nan0(x):
        return 0.0 if (x is None or pd.isna(x)) else float(x)

    def _bool_at(s, t):
        try:
            return (t in s.index) and (not pd.isna(s.loc[t])) and bool(s.loc[t])
        except Exception:
            return False

    def _rules_at(t):
        r = {
            "vh_db_drop":     _bool_at(rule_vh_vs_base, t),
            "vv_db_drop":     _bool_at(rule_vv_vs_base, t),
            "ratio_pct_drop": _bool_at(rule_ratio_pct, t),
            "vh_low_abs":     _bool_at(rule_vh_low_abs, t),
            "smooth":         _bool_at(rule_smooth, t),
            "z_watch":        (t in z.index and (not pd.isna(z.loc[t])) and z.loc[t] <= WATCH_Z),
            "z_alert":        (t in z.index and (not pd.isna(z.loc[t])) and z.loc[t] <= ALERT_Z),
            "small_drop":     (vh_lrdb is not None and t in vh_lrdb.index and (not pd.isna(vh_lrdb.loc[t])) and vh_lrdb.loc[t] <= WATCH_ANY_DROP)
        }
        return r

    def reasons_at(t):
        rs, r = [], _rules_at(t)
        if r["vh_db_drop"] and vh_lrdb is not None: rs.append(f"VH logΔ ≤ {MIN_ABS_DROP_DB_VH:.1f} dB")
        elif r["small_drop"]:                        rs.append("VH slightly lower vs base")
        if r["vv_db_drop"] and vv_lrdb is not None: rs.append(f"VV logΔ ≤ {MIN_ABS_DROP_DB_VV:.1f} dB")
        if r["ratio_pct_drop"] and ratio_b is not None: rs.append(f"VH/VV drop ≥ {int(MIN_PCT_DROP_LINEAR*100)}% vs base")
        if r["vh_low_abs"]:                         rs.append("VH ≤ -18 dB")
        if (t in z.index) and (not pd.isna(z.loc[t])): rs.append(f"z = {z.loc[t]:.1f}")
        if r["vv_db_drop"] and r["smooth"]:         rs.append("VV logΔ low & smooth")
        return ", ".join(rs) if rs else "Signals normal vs baseline"

    def severity_confidence_at(t):
        z_t      = _nan0(z.loc[t])       if (t in z.index) else np.nan
        vh_lr_t  = _nan0(vh_lrdb.loc[t]) if (vh_lrdb is not None and t in vh_lrdb.index) else np.nan
        vv_lr_t  = _nan0(vv_lrdb.loc[t]) if (vv_lrdb is not None and t in vv_lrdb.index) else np.nan

        r = _rules_at(t)
        rule_count = sum([r["vh_db_drop"], r["vv_db_drop"], r["ratio_pct_drop"], r["vh_low_abs"], r["z_watch"]])
        rule_count_norm = _clip01(rule_count / 5.0)

        # z severity (negative = wetter)
        if not pd.isna(z_t) and z_t <= WATCH_Z:
            z_sev = _clip01((WATCH_Z - z_t) / (WATCH_Z - ALERT_Z + 1e-6))
        else:
            z_sev = 0.0

        vh_sev = _clip01((0 - vh_lr_t) / abs(MIN_ABS_DROP_DB_VH)) if (not pd.isna(vh_lr_t) and vh_lr_t <= 0) else 0.0
        vv_sev = _clip01((0 - vv_lr_t) / abs(MIN_ABS_DROP_DB_VV)) if (not pd.isna(vv_lr_t) and vv_lr_t <= 0) else 0.0
        ratio_sev = 0.7 if r["ratio_pct_drop"] else 0.0
        persist_boost = 0.15 if _bool_at(alerts_mask, t) else 0.0

        sev01 = _clip01(0.35*z_sev + 0.30*vh_sev + 0.15*vv_sev + 0.20*ratio_sev + persist_boost)
        severity_0_100 = int(round(sev01 * 100))

        conf = 0.40
        conf += 0.25 * rule_count_norm
        conf += 0.15 * (1.0 if _bool_at(alerts_mask, t) else 0.0)
        if vh_std is not None:
            med_std = vh_std.rolling(f"{ROLL_WINDOW_DAYS}D", closed="left").median()
            if (t in med_std.index) and (t in vh_std.index) and (not pd.isna(vh_std.loc[t])) and (not pd.isna(med_std.loc[t])):
                conf += 0.10 * (1.0 if float(vh_std.loc[t]) <= 0.9 * float(med_std.loc[t]) else 0.0)
        key_ok = int(t in z.index and not pd.isna(z_t)) + int(vh_lrdb is not None and t in vh_lrdb.index and not pd.isna(vh_lr_t)) + int(vv_lrdb is not None and t in vv_lrdb.index and not pd.isna(vv_lr_t))
        conf += 0.10 * _clip01(key_ok / 3.0)
        confidence_0_1 = _clip01(conf)

        return severity_0_100, confidence_0_1

    def classify_level_with_severity(t):
        r = _rules_at(t)
        level_orig = (
            "Alert" if (_bool_at(alerts_mask, t) or r["z_alert"])
            else ("Watch" if (r["vh_db_drop"] or r["vv_db_drop"] or r["ratio_pct_drop"] or r["z_watch"])
                  else "Healthy")
        )
        sev, _ = severity_confidence_at(t)
        if sev >= 65: return "Alert"
        if sev >= 35: return "Watch"
        return level_orig

    # -------------------- build insights table --------------------
    rows = []
    for t in df.index:
        sev, conf = severity_confidence_at(t)
        level = classify_level_with_severity(t)
        rows.append({
            "date": t,
            "S1_VH_CURR": float(vh.loc[t])      if vh is not None and t in vh.index else np.nan,
            "S1_VV_CURR": float(vv.loc[t])      if vv is not None and t in vv.index else np.nan,
            "S1_VH_LOGRATIO_DB": float(vh_lrdb.loc[t]) if vh_lrdb is not None and t in vh_lrdb.index else np.nan,
            "S1_VV_LOGRATIO_DB": float(vv_lrdb.loc[t]) if vv_lrdb is not None and t in vv_lrdb.index else np.nan,
            "S1_VH_VV_CURR": float(ratio.loc[t]) if ratio is not None and t in ratio.index else np.nan,
            "S1_VH_VV_DIFF": float(ratio_d.loc[t]) if ratio_d is not None and t in ratio_d.index else np.nan,
            "zscore": float(z.loc[t]) if (t in z.index and not pd.isna(z.loc[t])) else np.nan,
            "status": level,
            "severity_0_100": int(sev),
            "confidence_0_1": float(conf),
            "reasons": reasons_at(t),
            "actions": ACTIONS[level]
        })
    insights_df = pd.DataFrame(rows).sort_values("date")

    # save insights.csv beside ALERTS_CSV (or in cwd)
    insights_csv = (ALERTS_CSV.replace("alerts", "insights")
                    if "alerts" in os.path.basename(ALERTS_CSV).lower()
                    else os.path.join(os.path.dirname(ALERTS_CSV) or ".", "insights.csv"))
    try:
        os.makedirs(os.path.dirname(insights_csv) or ".", exist_ok=True)
        insights_df.to_csv(insights_csv, index=False)
    except Exception:
        pass

    # -------------------- alerts.csv (subset) --------------------
    alerts = insights_df[insights_df["status"] == "Alert"].copy()
    if not alerts.empty:
        keep = ["date","S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
                "S1_VH_VV_CURR","S1_VH_VV_DIFF","zscore","severity_0_100","confidence_0_1",
                "status","reasons","actions"]
        alerts = alerts[[c for c in keep if c in alerts.columns]]
        try:
            os.makedirs(os.path.dirname(ALERTS_CSV) or ".", exist_ok=True)
            alerts.to_csv(ALERTS_CSV, index=False)
        except Exception:
            pass

    # -------------------- PNG PLOT (last ~4 months) --------------------
    end_date = primary.index.max()
    start_date = end_date - pd.DateOffset(months=4)
    primary_4m = primary.loc[start_date:end_date]
    insights_4m = insights_df[(insights_df["date"] >= start_date) & (insights_df["date"] <= end_date)]

    # high-contrast colors for judges
    COL_ALERT, COL_WATCH, COL_HEALTHY, EDGE = "#FF00FF", "#00FFFF", "#FFFFFF", "black"

    def add_risk_bands(ax, series):
        name = (getattr(series, "name", "") or "").upper()
        y = series.values.astype(float)
        if np.all(~np.isfinite(y)):
            return
        ymin, ymax = np.nanpercentile(y, [5, 95])
        pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        if "LOGRATIO" in name:
            ax.axhspan(ymin - pad, Z_THRESHOLD,  facecolor="red",    alpha=0.08, label="High risk")
            ax.axhspan(Z_THRESHOLD, 0,          facecolor="yellow", alpha=0.12, label="Caution")
            ax.axhspan(0,  ymax + pad,          facecolor="green",  alpha=0.06, label="Safe")
        else:
            ax.axhspan(ymin - pad, -18.0,       facecolor="red",    alpha=0.08, label="High risk")
            ax.axhspan(-18.0, -14.0,            facecolor="yellow", alpha=0.12, label="Caution")
            ax.axhspan(-14.0, ymax + pad,       facecolor="green",  alpha=0.06, label="Safe")
        ax.set_ylim(ymin - pad, ymax + pad)

    def pick_y(df_points):
        if ("S1_VH_LOGRATIO_DB" in df_points.columns) and df_points["S1_VH_LOGRATIO_DB"].notna().any():
            return "S1_VH_LOGRATIO_DB"
        if ("S1_VH_CURR" in df_points.columns) and df_points["S1_VH_CURR"].notna().any():
            return "S1_VH_CURR"
        return None  # will fallback to primary

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(primary_4m.index, primary_4m.values, marker="o", markersize=6, linewidth=2.5,
            color="#2ca096", label="Soil moisture / water level (proxy)")

    grp_alert   = insights_4m[insights_4m["status"] == "Alert"]
    grp_watch   = insights_4m[insights_4m["status"] == "Watch"]
    grp_healthy = insights_4m[insights_4m["status"] == "Healthy"]

    def scatter_group(df_points, label, color, marker, size=110, edge="black"):
        if df_points.empty:
            return
        ycol = pick_y(df_points)
        yvals = df_points[ycol] if ycol is not None else df_points["date"].map(lambda d: primary.loc[d] if d in primary.index else np.nan)
        ax.scatter(df_points["date"], yvals, s=size, c=color, marker=marker,
                   edgecolors=edge, linewidths=1, label=label, zorder=3)

    scatter_group(grp_alert,   "Waterlogging Alert",  COL_ALERT,   "o", size=130, edge=EDGE)
    scatter_group(grp_watch,   "Watch",               COL_WATCH,   "^", size=110, edge=EDGE)
    scatter_group(grp_healthy, "Healthy",             COL_HEALTHY, "s", size=90,  edge=EDGE)

    add_risk_bands(ax, primary_4m)

    ax.set_title("Sentinel-1 Waterlogging Monitor (last 4 months)", fontsize=16, weight="bold")
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Moisture / Water Level (proxy)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    latest_val = float(primary_4m.dropna().iloc[-1])
    ax.axhline(latest_val, color="brown", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Reference ({latest_val:.2f})")

    handles, labels = ax.get_legend_handles_labels()
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    ax.legend(*zip(*uniq), frameon=True, fontsize=11,
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(ALERTS_PLOT_PNG) or ".", exist_ok=True)
    plt.savefig(ALERTS_PLOT_PNG, dpi=180)
    plt.close(fig)

    # optional: sync to same folder as your web HTML (no-op if same)
    dst_png = ALERTS_PLOT_PNG

    return alerts, insights_df, dst_png, insights_csv

# --------- REQUIRED GLOBALS (safe defaults) ----------
# Rolling window for robust z-score
ROLL_WINDOW_DAYS    = 60

# Z-score threshold for alert bucket (more negative = stronger anomaly)
Z_THRESHOLD         = -1.5

# Min absolute log-ratio (curr - base) drop in dB to flag VH/VV
# (These are negative numbers; e.g., ≤ -1.5 dB means "at least 1.5 dB lower than base")
MIN_ABS_DROP_DB_VH  = -1.5
MIN_ABS_DROP_DB_VV  = -1.0

# Min % drop in linear VH/VV ratio vs base (e.g., 8% drop)
MIN_PCT_DROP_LINEAR = 0.08

# Require alerts to persist k passes (set 1 to disable persistence)
MIN_CONSECUTIVE     = 1

# I/O (used by the function when saving)
ALERTS_CSV          = "alerts.csv"        # where alert rows would be written
ALERTS_PLOT_PNG     = "s1_plot.png"       # where the PNG will be saved

def compute_temporal_engine_s1(csv_path):
    """
    Temporal engine for Sentinel-1 waterlogging.
    Returns: (alerts_df, recs_df, ALERTS_PLOT_PNG)

    Side-effect: also writes an 'insights' CSV (one row per S1 pass) next to ALERTS_CSV.
    """
    import os
    import pandas as pd, numpy as np
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    # -------------------- load & index --------------------
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # --- pick from your columns (all optional except at least one S1 metric) ---
    vh      = df.get("S1_VH_CURR")            # dB
    vv      = df.get("S1_VV_CURR")            # dB
    vh_lrdb = df.get("S1_VH_LOGRATIO_DB")     # (curr-base) in dB
    vv_lrdb = df.get("S1_VV_LOGRATIO_DB")     # (curr-base) in dB
    ratio   = df.get("S1_VH_VV_CURR")         # linear VH/VV
    ratio_d = df.get("S1_VH_VV_DIFF")         # linear diff: curr - base
    ratio_b = df.get("S1_VH_VV_BASE") if "S1_VH_VV_BASE" in df else None
    vh_std  = df.get("S1_VH_STD")

    # --- primary series for anomaly scoring (prefer logratio, else VH dB) ---
    primary = vh_lrdb if vh_lrdb is not None else vh
    if primary is None:
        return pd.DataFrame(), pd.DataFrame(), None

    # -------------------- robust z-score --------------------
    def robust_z(series, win_days):
        r = series.rolling(f"{win_days}D", closed="left")
        med = r.median()
        mad = r.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=False)
        return 0.6745 * (series - med) / mad.replace(0, np.nan)

    z = robust_z(primary, ROLL_WINDOW_DAYS)

    def ok_z(v):  # don't block on NaN (early window)
        return True if pd.isna(v) else (v <= Z_THRESHOLD)
    z_ok = z.apply(ok_z)

    # -------------------- rules (your thresholds) --------------------
    rule_vh_vs_base = (vh_lrdb <= MIN_ABS_DROP_DB_VH) if vh_lrdb is not None else pd.Series(False, index=df.index)
    rule_vv_vs_base = (vv_lrdb <= MIN_ABS_DROP_DB_VV) if vv_lrdb is not None else pd.Series(False, index=df.index)

    # ratio percent drop vs base if base exists
    if ratio is not None and ratio_b is not None:
        # percent change = (curr/base - 1), drop if ≤ -MIN_PCT_DROP_LINEAR
        rule_ratio_pct = ((ratio / ratio_b) - 1.0 <= -MIN_PCT_DROP_LINEAR)
    else:
        rule_ratio_pct = pd.Series(False, index=df.index)

    # absolute low VH (classic flood/water signature)
    rule_vh_low_abs = (vh <= -18.0) if vh is not None else pd.Series(False, index=df.index)

    # smoother field when inundated (optional helper)
    if vh_std is not None:
        s_med = vh_std.rolling(f"{ROLL_WINDOW_DAYS}D", closed="left").median()
        rule_smooth = (vh_std <= s_med * 0.9)
    else:
        rule_smooth = pd.Series(False, index=df.index)

    candidates = rule_vh_vs_base | rule_ratio_pct | rule_vh_low_abs | (rule_vv_vs_base & rule_smooth)
    raw_alerts = candidates & z_ok

    # persistence (local helper so we don’t touch your globals)
    def enforce_persistence(series: pd.Series, k: int) -> pd.Series:
        if k <= 1:
            return series.fillna(False)
        out = series.copy().fillna(False)
        c = 0
        for i, v in enumerate(out):
            c = (c + 1) if v else 0
            out.iloc[i] = c >= k
        return out

    alerts_mask = enforce_persistence(raw_alerts, MIN_CONSECUTIVE)

    # ==========================================================
    # ALWAYS-ON CLASSIFICATION + SEVERITY & CONFIDENCE
    # ==========================================================
    WATCH_Z = -0.8             # softer early warning
    ALERT_Z = Z_THRESHOLD      # e.g., -1.5
    WATCH_ANY_DROP = -0.5      # small base-drop still useful to mention

    ACTIONS = {
        "Alert":  "Check field now. Drain standing water if possible (open outlets/pump).",
        "Watch":  "Monitor next pass (6–12 days). Walk field edges; avoid over-irrigation.",
        "Healthy":"All good. Keep routine checks."
    }

    # ---- helpers for scoring ----
    def _clip01(x): 
        return float(np.nanmax([0.0, np.nanmin([1.0, x])]))

    def _nan0(x):
        return 0.0 if (x is None or pd.isna(x)) else float(x)

    def _rules_at(t):
        r = {
            "vh_db_drop":     bool(rule_vh_vs_base.loc[t]) if not pd.isna(rule_vh_vs_base.loc[t]) else False,
            "vv_db_drop":     bool(rule_vv_vs_base.loc[t]) if not pd.isna(rule_vv_vs_base.loc[t]) else False,
            "ratio_pct_drop": bool(rule_ratio_pct.loc[t])  if not pd.isna(rule_ratio_pct.loc[t])  else False,
            "vh_low_abs":     bool(rule_vh_low_abs.loc[t]) if not pd.isna(rule_vh_low_abs.loc[t]) else False,
            "smooth":         bool(rule_smooth.loc[t])     if not pd.isna(rule_smooth.loc[t])     else False,
            "z_watch":        (not pd.isna(z.loc[t]) and z.loc[t] <= WATCH_Z),
            "z_alert":        (not pd.isna(z.loc[t]) and z.loc[t] <= ALERT_Z),
            "small_drop":     (vh_lrdb is not None and (not pd.isna(vh_lrdb.loc[t])) and vh_lrdb.loc[t] <= WATCH_ANY_DROP)
        }
        return r

    def reasons_at(t):
        rs = []
        r = _rules_at(t)
        if r["vh_db_drop"] and vh_lrdb is not None: rs.append(f"VH logΔ ≤ {MIN_ABS_DROP_DB_VH:.1f} dB")
        elif r["small_drop"]:                         rs.append("VH slightly lower vs base")
        if r["vv_db_drop"] and vv_lrdb is not None: rs.append(f"VV logΔ ≤ {MIN_ABS_DROP_DB_VV:.1f} dB")
        if r["ratio_pct_drop"] and ratio_b is not None: rs.append(f"VH/VV drop ≥ {int(MIN_PCT_DROP_LINEAR*100)}% vs base")
        if r["vh_low_abs"]:                         rs.append("VH ≤ -18 dB")
        if not pd.isna(z.loc[t]):                   rs.append(f"z = {z.loc[t]:.1f}")
        if r["vv_db_drop"] and r["smooth"]:         rs.append("VV logΔ low & smooth")
        if not rs:
            rs = ["Signals normal vs baseline"]
        return ", ".join(rs)

    def severity_confidence_at(t):
        """
        severity_0_100 ~ strength of the waterlogging signal
        confidence_0_1 ~ trust in this classification (agreement + data quality + persistence)
        Both are intentionally simple & explainable for farmers/judges.
        """
        # pull features at t
        z_t      = _nan0(z.loc[t]) if not pd.isna(z.loc[t]) else np.nan
        vh_lr_t  = _nan0(vh_lrdb.loc[t]) if (vh_lrdb is not None and t in vh_lrdb.index) else np.nan
        vv_lr_t  = _nan0(vv_lrdb.loc[t]) if (vv_lrdb is not None and t in vv_lrdb.index) else np.nan
        ratio_t  = _nan0(ratio.loc[t])   if (ratio is not None   and t in ratio.index)   else np.nan
        ratio_b_t= _nan0(ratio_b.loc[t]) if (ratio_b is not None and t in ratio_b.index) else np.nan
        std_t    = _nan0(vh_std.loc[t])  if (vh_std is not None  and t in vh_std.index)  else np.nan

        r = _rules_at(t)
        rule_count = sum([r["vh_db_drop"], r["vv_db_drop"], r["ratio_pct_drop"], r["vh_low_abs"], r["z_watch"]])
        rule_count_norm = _clip01(rule_count / 5.0)

        # --- severity components (all 0..1), then map to 0..100
        # 1) z-score component: more negative → stronger
        if not pd.isna(z_t) and z_t <= WATCH_Z:
            z_sev = _clip01((WATCH_Z - z_t) / (WATCH_Z - ALERT_Z + 1e-6))  # 0 at watch boundary, 1 near/under ALERT_Z
        else:
            z_sev = 0.0

        # 2) VH dB drop component (logratio ≤ threshold)
        if not pd.isna(vh_lr_t) and vh_lr_t <= 0:
            # scale: 0 at 0 dB, 1 when at/under MIN_ABS_DROP_DB_VH
            vh_sev = _clip01((0 - vh_lr_t) / abs(MIN_ABS_DROP_DB_VH))
        else:
            vh_sev = 0.0

        # 3) VV dB drop (optional)
        if not pd.isna(vv_lr_t) and vv_lr_t <= 0:
            vv_sev = _clip01((0 - vv_lr_t) / abs(MIN_ABS_DROP_DB_VV))
        else:
            vv_sev = 0.0

        # 4) Ratio % drop (binary-ish signal → mid bump if present)
        ratio_sev = 0.0
        if ratio is not None and ratio_b is not None and r["ratio_pct_drop"]:
            ratio_sev = 0.7

        # 5) Persistence boost (if we are inside the alerts_mask run)
        persist_boost = 0.15 if (not pd.isna(alerts_mask.loc[t]) and bool(alerts_mask.loc[t])) else 0.0

        # Weighted blend → severity_0_1
        # weights sum ≈ 1.0 (plus small persistence boost)
        sev01 = (
            0.35 * z_sev +
            0.30 * vh_sev +
            0.15 * vv_sev +
            0.20 * ratio_sev
        )
        sev01 = _clip01(sev01 + persist_boost)
        severity_0_100 = int(round(sev01 * 100))

        # --- confidence (agreement + quality + persistence), 0..1
        # start base confidence a bit below mid (so agreement matters)
        conf = 0.40
        conf += 0.25 * rule_count_norm                                # more independent signals
        conf += 0.15 * (1.0 if (not pd.isna(alerts_mask.loc[t]) and bool(alerts_mask.loc[t])) else 0.0)  # persistence
        # data quality: not-too-noisy & not-too-missing
        # lower vh_std than recent median ⇒ smoother ⇒ add trust
        if vh_std is not None:
            med_std = vh_std.rolling(f"{ROLL_WINDOW_DAYS}D", closed="left").median()
            if t in med_std.index and not pd.isna(std_t) and not pd.isna(med_std.loc[t]):
                conf += 0.10 * (1.0 if std_t <= 0.9 * float(med_std.loc[t]) else 0.0)
        # key inputs available?
        key_ok = int(not pd.isna(z_t)) + int(not pd.isna(vh_lr_t)) + int(not pd.isna(vv_lr_t))
        conf += 0.10 * _clip01(key_ok / 3.0)

        confidence_0_1 = _clip01(conf)
        return severity_0_100, confidence_0_1

    def classify_level_with_severity(t):
        # keep your original logic but let severity override unclear cases
        level_orig = (
            "Alert" if (alerts_mask.loc[t] or (not pd.isna(z.loc[t]) and z.loc[t] <= ALERT_Z))
            else ("Watch" if (
                    (vh_lrdb is not None and (not pd.isna(vh_lrdb.loc[t])) and vh_lrdb.loc[t] <= WATCH_ANY_DROP)
                    or rule_vh_vs_base.loc[t]
                    or rule_vv_vs_base.loc[t]
                    or rule_ratio_pct.loc[t]
                    or (not pd.isna(z.loc[t]) and z.loc[t] <= WATCH_Z)
                ) else "Healthy")
        )
        sev, _ = severity_confidence_at(t)
        # severity thresholds → simple, explainable buckets
        if sev >= 65:
            return "Alert"
        if sev >= 35:
            return "Watch"
        return level_orig

    status_rows = []
    for t in df.index:
        sev, conf = severity_confidence_at(t)
        level = classify_level_with_severity(t)
        status_rows.append({
            "date": t,
            "S1_VH_CURR": float(vh.loc[t])      if vh is not None and t in vh.index else np.nan,
            "S1_VV_CURR": float(vv.loc[t])      if vv is not None and t in vv.index else np.nan,
            "S1_VH_LOGRATIO_DB": float(vh_lrdb.loc[t]) if vh_lrdb is not None and t in vh_lrdb.index else np.nan,
            "S1_VV_LOGRATIO_DB": float(vv_lrdb.loc[t]) if vv_lrdb is not None and t in vv_lrdb.index else np.nan,
            "S1_VH_VV_CURR": float(ratio.loc[t]) if ratio is not None and t in ratio.index else np.nan,
            "S1_VH_VV_DIFF": float(ratio_d.loc[t]) if ratio_d is not None and t in ratio_d.index else np.nan,
            "zscore": float(z.loc[t]) if not pd.isna(z.loc[t]) else np.nan,
            "status": level,
            "severity_0_100": int(sev),
            "confidence_0_1": float(conf),
            "reasons": reasons_at(t),
            "actions": ACTIONS[level]
        })
    status_df = pd.DataFrame(status_rows).sort_values("date")

    # save an insights CSV next to your alerts CSV
    insights_csv = (ALERTS_CSV.replace("alerts", "insights")
                    if "alerts" in os.path.basename(ALERTS_CSV).lower()
                    else os.path.join(os.path.dirname(ALERTS_CSV), "insights.csv"))
    try:
        status_df.to_csv(insights_csv, index=False)
    except Exception:
        pass  # don't fail the run if the path isn't writable

    # -------------------- alerts table (derived from status) --------------------
    alerts = status_df[status_df["status"] == "Alert"].copy()
    if not alerts.empty:
        # keep only the latest fields you care about
        keep_cols = ["date","S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
                     "S1_VH_VV_CURR","S1_VH_VV_DIFF","zscore","severity_0_100","confidence_0_1",
                     "status","reasons","actions"]
        alerts = alerts[[c for c in keep_cols if c in alerts.columns]]
        alerts.to_csv(ALERTS_CSV, index=False)

    # -------------------- plot (last 4 months) --------------------
    # -------------------- PNG-ONLY PLOT (last 4 months) --------------------
    import os, shutil
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # window to show in PNG
    end_date = primary.index.max()
    start_date = end_date - pd.DateOffset(months=4)
    primary_4m = primary.loc[start_date:end_date]
    alerts_4m  = alerts[(alerts["date"] >= start_date) & (alerts["date"] <= end_date)] if not alerts.empty else pd.DataFrame()

    # ---- build PNG ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(primary_4m.index, primary_4m.values,
            marker="o", markersize=6, linewidth=2.5, color="#2ca096",
            label="Soil moisture / water level (proxy)")

    # ---- choose a Y column for points (prefer logratio; else VH dB; else fallback to primary) ----
    def pick_y(df_points):
        if ("S1_VH_LOGRATIO_DB" in df_points.columns) and df_points["S1_VH_LOGRATIO_DB"].notna().any():
            return "S1_VH_LOGRATIO_DB"
        if ("S1_VH_CURR" in df_points.columns) and df_points["S1_VH_CURR"].notna().any():
            return "S1_VH_CURR"
        # final fallback: map dates to primary series
        return None

    # slice status for last 4 months
    view_df = status_df[(status_df["date"] >= start_date) & (status_df["date"] <= end_date)].copy()

    # split by status
    grp_alert  = view_df[view_df["status"] == "Alert"]
    grp_watch  = view_df[view_df["status"] == "Watch"]
    grp_healthy= view_df[view_df["status"] == "Healthy"]

    # plot each group
    def scatter_group(df_points, label, color, marker, size=110, edge="black"):
        if df_points.empty:
            return
        ycol = pick_y(df_points)
        if ycol is not None:
            yvals = df_points[ycol]
        else:
            # fallback: pull from primary by date
            yvals = df_points["date"].map(lambda d: primary.loc[d] if d in primary.index else np.nan)
        ax.scatter(df_points["date"], yvals,
                   s=size, c=color, marker=marker, edgecolors=edge, linewidths=1,
                   label=label, zorder=3)

    scatter_group(grp_alert,   "Waterlogging Alert",  "#d62728", "^", size=130)  # red circles
    scatter_group(grp_watch,   "Watch",               "#f3ff0e", "o", size=110)  # orange triangles
    scatter_group(grp_healthy, "Healthy",             "#3eac0b", "s", size=90)   # blue squares

    # risk bands
    ax.axhspan(-6, -3, facecolor="red",    alpha=0.08, label="High risk")
    ax.axhspan(-3,  0, facecolor="yellow", alpha=0.12, label="Caution")
    ax.axhspan( 0,  5, facecolor="green",  alpha=0.08, label="Safe")

    ax.set_title("Sentinel-1 Waterlogging Monitor (last 4 months)", fontsize=16, weight="bold")
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Moisture / Water Level (proxy)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()


    # --- add horizontal reference line (BEFORE legend) ---
    latest_val = float(primary_4m.iloc[-1])
    ax.axhline(latest_val, color="brown", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Reference line ({latest_val:.2f})")

    # --- legend (outside; bottom example) ---
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]

    ax.legend(*zip(*uniq),
              frameon=True, fontsize=11,
              loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3)

    fig.tight_layout()  # if clipping, use: plt.savefig(..., bbox_inches="tight")
    plt.savefig(ALERTS_PLOT_PNG, dpi=180)
    plt.close(fig)

    # -------------------- SYNC PNG to WEB FOLDER (optional & safe) --------------------
    # If your web HTML and PNG are already in the same directory, this will simply skip copying.
    WATERLOGGING_DASHBOARD_HTML = "NDWI_alerts_plotL1.html"   # change if needed
    OUT_WEB_HTML = WATERLOGGING_DASHBOARD_HTML

    _web_dir = os.path.dirname(OUT_WEB_HTML) or "."
    os.makedirs(_web_dir, exist_ok=True)

    _png_name = os.path.basename(ALERTS_PLOT_PNG)
    src_png = os.path.abspath(ALERTS_PLOT_PNG)
    dst_png = os.path.abspath(os.path.join(_web_dir, _png_name))

    if src_png != dst_png:
        shutil.copyfile(src_png, dst_png)

    # return PNG path (no HTML)
    plot_rel = os.path.relpath(dst_png if os.path.exists(dst_png) else src_png, start=_web_dir)
    return alerts, status_df, (dst_png if os.path.exists(dst_png) else src_png)

# --------- RUN + DISPLAY ----------
csv_path = f"timeseries_{os.path.splitext(OUT_STACK)[0]}.csv"
alerts_df, recs_df, plot_path = compute_temporal_engine_s1(csv_path)

print("alerts_df shape:", getattr(alerts_df, "shape", None))
print("recs_df shape:",   getattr(recs_df, "shape", None))
print("plot_path:", plot_path)

# show the PNG in the notebook
from IPython.display import Image, display
display(Image(filename=plot_path))

# ------- 0) Setup -------
edges   = np.array([0.00, 0.30, 0.50, 0.70, 1.00], dtype=float)   # 4 bins: [0,.3), [.3,.5), [.5,.7), [.7,1]
labels  = [0, 1, 2, 3]  # 0=Healthy, 1=Watch, 2=Concern, 3=Alert
names   = {0: "Healthy", 1: "Watch", 2: "Concern", 3: "Alert"}
palette = {0:"#2ecc71", 1:"#f1c40f", 2:"#e67e22", 3:"#e74c3c"}  # green→red (overlay)

# Optional AOI mask: True inside field; if you have it, pass it. Otherwise treat all as valid.
valid_mask = np.isfinite(risk)
if 'aoi_mask' in globals() and aoi_mask is not None and aoi_mask.shape == risk.shape:
    valid_mask &= aoi_mask.astype(bool)

# ------- 1) Classify risk into 4 classes -------
# Keep logic exact to your thresholds but robust to small numeric drift
risk_clip = np.clip(risk, 0.0, 1.0)
# np.digitize returns 1..len(bins). We want 0..3 for 4 classes.
# Use the same half-open intervals: [0,.3), [.3,.5), [.5,.7), [.7,1.0]
classes = np.digitize(risk_clip, bins=edges[1:-1], right=False)  # bins=(0.30,0.50,0.70)
# classes ∈ {0,1,2,3}; set invalid pixels to -1 so they don’t count
classes = np.where(valid_mask, classes, -1).astype(np.int32)

# ------- 2) Pixel area (m² and ha) robust to rotation/shear -------
# per-pixel area = |a*e - b*d|
px_area_m2 = abs(src_transform.a * src_transform.e - src_transform.b * src_transform.d)
px_area_ha = px_area_m2 / 10_000.0

# ------- 3) Area per class using bincount (fast) -------
# Map invalid (-1) to 0 temporarily by masking counts
flat = classes.ravel()
counts = np.bincount(flat[flat >= 0], minlength=4)  # counts for class 0..3 only
area_by_class = {k: float(counts[k]) * px_area_ha for k in range(4)}
total_ha = sum(area_by_class.values())

# ------- 4) Summary (farmer-friendly) -------
print(f"Total classified area: {total_ha:.2f} ha")
for k in labels:
    ha  = area_by_class.get(k, 0.0)
    pct = (100.0 * ha / total_ha) if total_ha > 1e-12 else 0.0
    print(f"{names[k]:<8}: {ha:,.2f} ha ({pct:,.1f}%)")