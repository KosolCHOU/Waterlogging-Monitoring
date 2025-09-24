# analysis/insights.py
# Sentinel-1 insights engine + HTML builders for CropXcel's dashboard.
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import os, base64, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from django.conf import settings
import matplotlib as mpl

def _default_media_root() -> str:
    try:
        from django.conf import settings
        if getattr(settings, "MEDIA_ROOT", None):
            return str(settings.MEDIA_ROOT)
    except Exception:
        pass
    return os.path.join(os.getcwd(), "media")

__all__ = [
    # config
    "ROLL_WINDOW_DAYS","Z_THRESHOLD","MIN_ABS_DROP_DB_VH","MIN_ABS_DROP_DB_VV",
    "MIN_PCT_DROP_LINEAR","MIN_CONSECUTIVE","ALERTS_CSV","ALERTS_PLOT_PNG",
    # engines
    "compute_temporal_engine_s1",
    # builders
    "build_scale_data","render_legend_rows","format_total_badge",
    "prepare_farmer_view","prepare_technical_view","df_to_html_table",
    "render_plot_section","build_insights_html",
]

# --------- SAFE DEFAULTS (override from settings or caller if needed) ----------
ROLL_WINDOW_DAYS    = int(os.getenv("S1_ROLL_WINDOW_DAYS", 60))
Z_THRESHOLD         = float(os.getenv("S1_Z_THRESHOLD", -1.5))
MIN_ABS_DROP_DB_VH  = float(os.getenv("S1_MIN_ABS_DROP_DB_VH", -1.5))
MIN_ABS_DROP_DB_VV  = float(os.getenv("S1_MIN_ABS_DROP_DB_VV", -1.0))
MIN_PCT_DROP_LINEAR = float(os.getenv("S1_MIN_PCT_DROP_LINEAR", 0.08))
MIN_CONSECUTIVE     = int(os.getenv("S1_MIN_CONSECUTIVE", 1))
ALERTS_CSV          = os.getenv("S1_ALERTS_CSV", "alerts.csv")
ALERTS_PLOT_PNG     = os.getenv("S1_PLOT_PNG", "s1_plot.png")

# ---------- low-level helpers ----------
def _clip01(x: float) -> float:
    return float(np.nanmax([0.0, np.nanmin([1.0, x])]))

def embed_image_base64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def df_safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _pct(val: float, tot: float) -> float:
    return (100.0 * float(val) / float(tot)) if (tot and tot > 0) else 0.0

# ---------- temporal engine ----------
def compute_temporal_engine_s1(
    csv_path: str,
    *,
    media_root: str | None = None,
    subdir: str | None = None,
    plot_name: str | None = None,
    insights_name: str = "insights.csv",
    alerts_name: str = "alerts.csv",
    add_date_to_plot: bool = True,
):
    """
    Temporal engine for Sentinel-1 waterlogging.
    Returns: alerts_df, insights_df, plot_png_path, insights_csv_path
    Side effects: writes insights.csv (+ alerts.csv if any) and a PNG plot.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # ---------- load ----------
    if not os.path.exists(csv_path):
        return pd.DataFrame(), pd.DataFrame(), None, None

    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), None, None

    for s in ["S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
              "S1_VH_VV_CURR","S1_VH_VV_DIFF","S1_VH_VV_BASE","S1_VH_STD"]:
        if s in df.columns:
            df[s] = pd.to_numeric(df[s], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # ---------- pick series ----------
    vh      = df.get("S1_VH_CURR")
    vv      = df.get("S1_VV_CURR")
    vh_lrdb = df.get("S1_VH_LOGRATIO_DB")
    vv_lrdb = df.get("S1_VV_LOGRATIO_DB")
    ratio   = df.get("S1_VH_VV_CURR")
    ratio_d = df.get("S1_VH_VV_DIFF")
    ratio_b = df.get("S1_VH_VV_BASE") if "S1_VH_VV_BASE" in df.columns else None
    vh_std  = df.get("S1_VH_STD")

    primary = vh_lrdb if (vh_lrdb is not None and vh_lrdb.notna().any()) else vh
    if primary is None or primary.dropna().empty:
        return pd.DataFrame(), pd.DataFrame(), None, None

    # ---------- robust z ----------
    def robust_z(series, win_days: int):
        r = series.rolling(f"{win_days}D", closed="left")
        med = r.median()
        mad = r.apply(lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=False)
        return 0.6745 * (series - med) / mad.replace(0, np.nan)

    z = robust_z(primary, ROLL_WINDOW_DAYS)
    WATCH_Z, ALERT_Z = -0.8, Z_THRESHOLD
    WATCH_ANY_DROP = -0.5
    z_ok = z.apply(lambda v: True if pd.isna(v) else bool(v <= Z_THRESHOLD))

    # ---------- rules ----------
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

    def enforce_persistence(series: pd.Series, k: int) -> pd.Series:
        if k <= 1:
            return series.fillna(False)
        out = series.copy().fillna(False)
        vals = out.values; c = 0
        for i in range(len(vals)):
            c = (c + 1) if vals[i] else 0
            vals[i] = (c >= k)
        out[:] = vals
        return out

    alerts_mask = enforce_persistence(raw_alerts, MIN_CONSECUTIVE)

    ACTIONS = {
        "Alert":  "Check field now. Drain standing water if possible (open outlets/pump).",
        "Watch":  "Monitor next pass (6–12 days). Walk field edges; avoid over-irrigation.",
        "Healthy":"All good. Keep routine checks."
    }

    def _clip01(x): return float(np.nanmax([0.0, np.nanmin([1.0, x])]))
    def _nan0(x):   return 0.0 if (x is None or pd.isna(x)) else float(x)
    def _bool_at(s, t):
        try:    return (t in s.index) and (not pd.isna(s.loc[t])) and bool(s.loc[t])
        except: return False

    def _rules_at(t):
        return {
            "vh_db_drop":     _bool_at(rule_vh_vs_base, t),
            "vv_db_drop":     _bool_at(rule_vv_vs_base, t),
            "ratio_pct_drop": _bool_at(rule_ratio_pct, t),
            "vh_low_abs":     _bool_at(rule_vh_low_abs, t),
            "smooth":         _bool_at(rule_smooth, t),
            "z_watch":        (t in z.index and (not pd.isna(z.loc[t])) and z.loc[t] <= WATCH_Z),
            "z_alert":        (t in z.index and (not pd.isna(z.loc[t])) and z.loc[t] <= ALERT_Z),
            "small_drop":     (vh_lrdb is not None and t in vh_lrdb.index and (not pd.isna(vh_lrdb.loc[t])) and vh_lrdb.loc[t] <= WATCH_ANY_DROP),
        }

    def reasons_at(t):
        r = _rules_at(t); rs = []
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
        z_sev = _clip01((WATCH_Z - z_t) / (WATCH_Z - ALERT_Z + 1e-6)) if (not pd.isna(z_t) and z_t <= WATCH_Z) else 0.0
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
        return severity_0_100, _clip01(conf)

    def classify_level_with_severity(t):
        r = _rules_at(t)
        level = ("Alert" if (_bool_at(alerts_mask, t) or r["z_alert"])
                 else ("Watch" if (r["vh_db_drop"] or r["vv_db_drop"] or r["ratio_pct_drop"] or r["z_watch"])
                       else "Healthy"))
        sev, _ = severity_confidence_at(t)
        if sev >= 65: return "Alert"
        if sev >= 35: return "Watch"
        return level

    rows = []
    for t in df.index:
        sev, conf = severity_confidence_at(t)
        rows.append({
            "date": t,
            "S1_VH_CURR": float(vh.loc[t])      if vh is not None and t in vh.index else np.nan,
            "S1_VV_CURR": float(vv.loc[t])      if vv is not None and t in vv.index else np.nan,
            "S1_VH_LOGRATIO_DB": float(vh_lrdb.loc[t]) if vh_lrdb is not None and t in vh_lrdb.index else np.nan,
            "S1_VV_LOGRATIO_DB": float(vv_lrdb.loc[t]) if vv_lrdb is not None and t in vv_lrdb.index else np.nan,
            "S1_VH_VV_CURR": float(ratio.loc[t]) if ratio is not None and t in ratio.index else np.nan,
            "S1_VH_VV_DIFF": float(ratio_d.loc[t]) if ratio_d is not None and t in ratio_d.index else np.nan,
            "zscore": float(z.loc[t]) if (t in z.index and not pd.isna(z.loc[t])) else np.nan,
            "status": classify_level_with_severity(t),
            "severity_0_100": int(sev),
            "confidence_0_1": float(conf),
            "reasons": reasons_at(t),
            "actions": ACTIONS[classify_level_with_severity(t)],
        })
    insights_df = pd.DataFrame(rows).sort_values("date")

    # ---------- output locations under MEDIA ----------
    mr = media_root or _default_media_root()
    if subdir:
        # keep option, but you said you won't use it — pass subdir=None to avoid nesting
        sd = subdir.strip("/").replace("..","_")
        insights_dir = os.path.join(mr, "insights", sd)
        plots_dir    = os.path.join(mr, "plots", sd)
    else:
        insights_dir = os.path.join(mr, "insights")
        plots_dir    = os.path.join(mr, "plots")
    os.makedirs(insights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Names
    base  = os.path.splitext(os.path.basename(csv_path))[0]  # e.g., timeseries_field_170_20250919_091422
    stamp = ""
    if add_date_to_plot and not insights_df.empty:
        stamp = "_" + pd.to_datetime(insights_df["date"].max(), errors="coerce").strftime("%Y%m%d")

    insights_csv  = os.path.join(insights_dir, f"insights_{base}.csv")
    alerts_csv    = os.path.join(insights_dir, f"alerts_{base}.csv")
    plot_png_path = os.path.join(plots_dir,  f"{(plot_name or 's1_plot')}_{base}{stamp}.png")

    # ---------- write CSVs ----------
    try:
        insights_df.to_csv(insights_csv, index=False)
    except Exception:
        insights_csv = None  # keep going

    alerts = insights_df[insights_df["status"] == "Alert"].copy()
    if not alerts.empty:
        keep = ["date","S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
                "S1_VH_VV_CURR","S1_VH_VV_DIFF","zscore","severity_0_100","confidence_0_1",
                "status","reasons","actions"]
        alerts = alerts[[c for c in keep if c in alerts.columns]]
        try:
            alerts.to_csv(alerts_csv, index=False)
        except Exception:
            pass

    # ---------- plot (last 4 months) ----------
    end_date = primary.index.max()
    start_date = end_date - pd.DateOffset(months=4)
    primary_4m = primary.loc[start_date:end_date]
    view_df = insights_df[(insights_df["date"] >= start_date) & (insights_df["date"] <= end_date)].copy()

    # 👉 Make plot 3-class only (Concern → Watch)
    if "status" in view_df.columns:
        view_df["status"] = view_df["status"].replace({"Concern": "Watch"})

    # --- Palette (match dashboard vibes) ---
    COL_ALERT   = "#e74c3c"  # red
    COL_WATCH   = "#f1c40f"  # yellow
    COL_HEALTHY = "#2ecc71"  # green
    LINE_COLOR  = "#2ca089"  # soft teal

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(primary_4m.index, primary_4m.values, marker="o", markersize=5,
            linewidth=2.5, color=LINE_COLOR,
            label="Soil moisture / water level (proxy)")

    def add_gradient_background(ax, series, alpha=0.16):
        """
        Draw a vertical gradient background from red (bottom) → yellow → green (top),
        matching the dashboard vibe.
        """
        y = series.values.astype(float)
        if np.all(~np.isfinite(y)):
            return
        ymin, ymax = np.nanpercentile(y, [5, 95])
        pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        y0, y1 = ymin - pad, ymax + pad
        ax.set_ylim(y0, y1)

        # red → yellow → green, bottom→top (origin='lower')
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "risk", ["#e74c3c", "#f1c40f", "#2ecc71"]
        )
        # vertical gradient
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax.imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            extent=[ax.get_xlim()[0], ax.get_xlim()[1], y0, y1],
            origin="lower",
            alpha=alpha,
            zorder=0,
        )

        # optional legend keys for background zones (small, subtle)
        from matplotlib.patches import Patch
        ax._risk_patches = [
            Patch(facecolor="#2ecc71", alpha=alpha, label="Safe"),
            Patch(facecolor="#f1c40f", alpha=alpha, label="Caution"),
            Patch(facecolor="#e74c3c", alpha=alpha, label="High risk"),
        ]

    def pick_y(df_points):
        if ("S1_VH_LOGRATIO_DB" in df_points.columns) and df_points["S1_VH_LOGRATIO_DB"].notna().any():
            return "S1_VH_LOGRATIO_DB"
        if ("S1_VH_CURR" in df_points.columns) and df_points["S1_VH_CURR"].notna().any():
            return "S1_VH_CURR"
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(primary_4m.index, primary_4m.values, marker="o", markersize=5,
            linewidth=2.5, color=LINE_COLOR,
            label="Soil moisture / water level (proxy)")

    # groups (now only 3)
    grp_alert   = view_df[view_df["status"] == "Alert"]
    grp_watch   = view_df[view_df["status"] == "Watch"]
    grp_healthy = view_df[view_df["status"] == "Healthy"]

    def scatter_group(df_points, label, color, marker, size=130, edge="black"):
        if df_points.empty: return
        ycol = pick_y(df_points)
        yvals = df_points[ycol] if ycol is not None else df_points["date"].map(
            lambda d: primary.loc[d] if d in primary.index else np.nan
        )
        ax.scatter(df_points["date"], yvals, s=size, c=color, marker=marker,
                   edgecolors=edge, linewidths=1.2, label=label, zorder=3)

    # shapes: o, ^, X (clearly distinct)
    scatter_group(grp_alert,   "Waterlogging Alert", COL_ALERT,   "X")
    scatter_group(grp_watch,   "Watch",              COL_WATCH,   "^")
    scatter_group(grp_healthy, "Healthy",            COL_HEALTHY, "o")

    # Gradient background (soft, heatmap vibe)
    add_gradient_background(ax, primary_4m)
    ax.set_title("Sentinel-1 Waterlogging Monitor (last 4 months)",
                 fontsize=15, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Moisture / Water Level (proxy)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    latest_val = float(primary_4m.dropna().iloc[-1])
    ax.axhline(latest_val, color="brown", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"Reference ({latest_val:.2f})")

    # legend cleanup (include background keys if present)
    handles, labels = ax.get_legend_handles_labels()
    if hasattr(ax, "_risk_patches"):
        handles = handles + ax._risk_patches
        labels  = labels  + [p.get_label() for p in ax._risk_patches]

    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=True, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    fig.tight_layout()
    try:
        plt.savefig(plot_png_path, dpi=180, facecolor="white")
    finally:
        plt.close(fig)

    return alerts, insights_df, plot_png_path, insights_csv

# --- replace build_scale_data(...) with this version ---
def build_scale_data(
    area_by_class: Dict[int, float] | Dict[str, float],
    total_ha: Optional[float] = None,
    names: Optional[Dict[int, str]] = None,
    palette: Optional[Dict[int, str]] = None,
    classes: Iterable[int] = (0, 1, 3),   # ⬅️ drop 2 (Concern)
) -> List[Dict]:
    names = names or {0: "Healthy", 1: "Watch", 3: "Alert"}   # ⬅️ drop Concern
    palette = palette or {0: "#2ecc71", 1: "#f1c40f", 3: "#e74c3c"}  # ⬅️ drop orange

    # normalize so both int and str keys work
    def _get(d, k, default=0.0):
        if d is None: return default
        if k in d: return d[k]
        sk = str(k)
        return d.get(sk, default)

    total = (float(total_ha) if total_ha is not None
             else float(sum(_get(area_by_class, k, 0.0) for k in classes)) if area_by_class else 0.0)

    rows = []
    for k in classes:
        ha = float(_get(area_by_class, k, 0.0)) if area_by_class else 0.0
        pct = _pct(ha, total)
        rows.append({
            "k": k, "label": names.get(k, str(k)), "ha": ha, "pct": pct,
            "color": palette.get(k, "#999"),
        })
    return rows

def render_legend_rows(scale_data: List[Dict]) -> str:
    return "\n".join([
        f"""<div class="legrow" data-k="{d['k']}" data-ha="{d['ha']:.2f}"
                 data-pct="{d['pct']:.1f}" style="--c:{d['color']}; --pct:{d['pct']:.1f};">
              <span class="bubble" aria-hidden="true"></span>
              <span class="lg-name">{d['label']}</span>
              <span class="lg-pill">{d['ha']:.2f} ha</span>
              <div class="lg-bar" role="progressbar" aria-label="{d['label']} share"
                   aria-valuemin="0" aria-valuemax="100" aria-valuenow="{d['pct']:.1f}">
                <span class="fill"></span>
              </div>
              <span class="lg-val">{d['pct']:.1f}%</span>
            </div>"""
        for d in scale_data
    ])

def format_total_badge(total_ha: Optional[float]) -> str:
    return f"{float(total_ha):,.2f} Ha" if (total_ha and total_ha > 0) else "—"

# ---------- Insights tables ----------
def prepare_farmer_view(insights_df: Optional[pd.DataFrame],
                        recs_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    if insights_df is None or insights_df.empty:
        return None
    d = insights_df.copy()
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce")
    if recs_df is not None and not recs_df.empty:
        r = recs_df.copy()
        r["date"] = pd.to_datetime(r.get("date"), errors="coerce")
        merge_fields = [c for c in ["severity_0_100","confidence_0_1"] if c in r.columns]
        if merge_fields:
            d = d.merge(r[["date"]+merge_fields], on="date", how="left", suffixes=("", "_r"))
            for f in merge_fields:
                if f + "_r" in d.columns:
                    d[f] = pd.to_numeric(d.get(f), errors="coerce").combine_first(
                           pd.to_numeric(d[f + "_r"], errors="coerce"))
                    d.drop(columns=[f + "_r"], inplace=True)
    if "status" in d.columns:
        def badge_html(s):
            mapping = {
                "Healthy": '<span class="healthy">🟢 Healthy</span>',
                "Watch":   '<span class="watch">🟡 Watch</span>',
                "Alert":   '<span class="alert">🔴 Alert</span>'
            }
            return mapping.get(s.replace("🟢 ","").replace("🟡 ","").replace("🔴 ",""), s)
        d["status"] = d["status"].apply(badge_html)

    if "actions" in d.columns:
        d["actions"] = d["actions"].apply(lambda a: f"<div>{a}</div>" if a else "")

    d = d.sort_values("date", ascending=False)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")

    for col in ["severity_0_100","confidence_0_1"]:
        if col not in d.columns: d[col] = pd.NA

    d["severity_0_100"] = (pd.to_numeric(d["severity_0_100"], errors="coerce")
                             .round().astype("Int64").astype(str).replace("<NA>",""))
    d["confidence_0_1"] = pd.to_numeric(d["confidence_0_1"], errors="coerce").map(
        lambda x: (f"{x:.2f}" if pd.notna(x) else "")
    )
    cols = [c for c in ["date","status","actions"] if c in d.columns]
    return d[cols]

def prepare_technical_view(insights_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if insights_df is None or insights_df.empty:
        return None
    d = insights_df.copy()
    d = d.drop(columns=[c for c in ["status","actions"] if c in d.columns], errors="ignore")
    ordered = ["date","zscore","severity_0_100","confidence_0_1","S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
            "S1_VH_VV_CURR","S1_VH_VV_DIFF"]
    ordered = [c for c in ordered if c in d.columns]
    d = d[ordered]
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    num_cols = [c for c in d.columns if c != "date"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda x: (f"{x:.4f}" if pd.notna(x) else ""))
    d = d.rename(columns={
        "zscore":"z","severity_0_100":"Severity","confidence_0_1":"Confidence","S1_VH_CURR":"VH","S1_VV_CURR":"VV",
        "S1_VH_LOGRATIO_DB":"VH_log","S1_VV_LOGRATIO_DB":"VV_log",
        "S1_VH_VV_CURR":"VH/VV","S1_VH_VV_DIFF":"Δ(VH/VV)"
        
    })
    return d

def df_to_html_table(df: Optional[pd.DataFrame], classes: str="minitable", max_rows: int=14) -> str:
    if df is None or (hasattr(df,"empty") and df.empty):
        return "<div class='empty'>No records.</div>"
    return df.head(max_rows).to_html(index=False, classes=classes, escape=False, border=0, na_rep="—")

# ---------- Plot block ----------
def render_plot_section(plot_path: Optional[str]) -> str:
    if not plot_path:
        return "<div class='muted'>No plot available.</div>"
    b64 = embed_image_base64(plot_path)
    return f"<img src='{b64}' alt='S1 plot' style='width:100%;height:auto;'/>" if b64 \
           else "<div class='muted'>No plot available.</div>"

# ---------- High-level one-call builder ----------
def build_insights_html(
    insights_csv: Optional[str] = None,
    recs_csv: Optional[str] = None,
    area_by_class: Optional[Dict[int, float]] = None,
    total_ha: Optional[float] = None,
    names: Optional[Dict[int, str]] = None,
    palette: Optional[Dict[int, str]] = None,
    plot_path: Optional[str] = None,
    farmer_rows: int = 14,
    technical_rows: int = 20,
) -> Dict[str, str]:
    """
    Loads CSVs (if provided), prepares farmer/technical tables, legend rows,
    total badge, and the plot <img> block.
    """
    insights_df = df_safe_read_csv(insights_csv) if insights_csv else pd.DataFrame()
    recs_df     = df_safe_read_csv(recs_csv)     if recs_csv     else pd.DataFrame()

    farmer_display    = prepare_farmer_view(insights_df, recs_df)
    technical_display = prepare_technical_view(insights_df)

    scale = build_scale_data(area_by_class or {}, total_ha, names, palette)
    legend_rows_html = render_legend_rows(scale)
    total_badge = format_total_badge(total_ha if total_ha is not None else sum((area_by_class or {}).values()))
    plot_section = render_plot_section(plot_path)

    return {
        "farmer_table_html":    df_to_html_table(farmer_display,    max_rows=farmer_rows),
        "technical_table_html": df_to_html_table(technical_display, max_rows=technical_rows),
        "legend_rows_html":     legend_rows_html,
        "total_badge":          total_badge,
        "plot_section":         plot_section,
    }

def rel_to_media(abs_path: str) -> str:
    if not abs_path: return ""
    mr = str(settings.MEDIA_ROOT)
    ap = os.path.abspath(abs_path)
    if ap.startswith(os.path.abspath(mr) + os.sep):
        return os.path.relpath(ap, mr).replace("\\","/")
    return ""

# analysis/insights.py
import numpy as np, rasterio

def classify_and_area(
    risk_tif_path: str,
    thresholds=(0.25, 0.45, 0.65),
    scale_from: str | None = None,
    default_pixel_area_m2: float | None = None,
):
    """
    Returns (area_by_class_dict, total_ha).
      Classes: 0=Healthy, 1=Watch, 2=Concern, 3=Alert (>= last threshold)
    """
    with rasterio.open(risk_tif_path) as ds:
        a = ds.read(1).astype("float32")
        a[np.isclose(a, ds.nodata)] = np.nan if ds.nodata is not None else a
        transform = ds.transform
        # pixel area (m²) from geotransform (approx; fine for small AOIs)
        if default_pixel_area_m2 is not None:
            px_m2 = default_pixel_area_m2
        else:
            # area of one pixel in projected CRS; if EPSG:4326, fallback to ~meter scale later
            px_m2 = abs(transform.a * transform.e)  # width * height
            if px_m2 == 0 or not np.isfinite(px_m2):
                px_m2 = 1.0  # fallback; better: pass default_pixel_area_m2

        finite = np.isfinite(a)
        vals = a[finite]

        t0, t1, t2 = thresholds
        cls0 = (vals <  t0)
        cls1 = (vals >= t0) & (vals <  t2)   # merge old cls1 + cls2
        cls3 = (vals >= t2)

        count0 = int(cls0.sum())
        count1 = int(cls1.sum())
        count3 = int(cls3.sum())

        areas_m2 = [count0*px_m2, count1*px_m2, count3*px_m2]
        areas_ha = [round(x/10000.0, 6) for x in areas_m2]
        total_ha = round(sum(areas_ha), 6)

        # return only 3 keys
        area_by_class = {"0": areas_ha[0], "1": areas_ha[1], "3": areas_ha[2]}
        return area_by_class, total_ha

