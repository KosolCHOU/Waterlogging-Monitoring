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
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings

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
# Require persistence by default to reduce false positives
MIN_CONSECUTIVE     = int(os.getenv("S1_MIN_CONSECUTIVE", 2))
# Absolute low VH cutoff in dB (converted to linear for comparisons)
VH_ABS_DB_CUTOFF    = float(os.getenv("S1_VH_ABS_DB_CUTOFF", -18.0))
# Robust z safeguards
MIN_Z_SAMPLES       = int(os.getenv("S1_MIN_Z_SAMPLES", 6))      # require at least N prior samples
MAD_EPS             = float(os.getenv("S1_MAD_EPS", 1e-6))        # floor to avoid huge z from tiny MAD
WATCH_Z_DEFAULT     = float(os.getenv("S1_WATCH_Z", -0.8))       # env-tunable watch band
RATIO_EPS           = float(os.getenv("S1_RATIO_EPS", 1e-9))      # avoid div-by-zero in ratio drop
WINSOR_PCT          = float(os.getenv("S1_WINSOR_PCT", 0.0))      # 0 disables; else e.g., 1.0 clips [p,100-p]
# Enhanced robustness parameters
ENABLE_SEASONAL_DECOMP = bool(os.getenv("S1_ENABLE_SEASONAL", "True").lower() in ("true", "1", "yes"))
SEASONAL_PERIOD      = int(os.getenv("S1_SEASONAL_PERIOD", 365))   # days in agricultural cycle
MIN_SEASONAL_SAMPLES = int(os.getenv("S1_MIN_SEASONAL_SAMPLES", 24)) # minimum samples for seasonal decomp
ADAPTIVE_WINDOW      = bool(os.getenv("S1_ADAPTIVE_WINDOW", "True").lower() in ("true", "1", "yes"))
MIN_WINDOW_DAYS      = int(os.getenv("S1_MIN_WINDOW_DAYS", 30))     # minimum adaptive window size
MAX_WINDOW_DAYS      = int(os.getenv("S1_MAX_WINDOW_DAYS", 120))    # maximum adaptive window size
STATIONARITY_TEST    = bool(os.getenv("S1_STATIONARITY_TEST", "True").lower() in ("true", "1", "yes"))
ADF_PVALUE_THRESH    = float(os.getenv("S1_ADF_PVALUE", 0.05))      # p-value threshold for stationarity
REGIME_DETECTION     = bool(os.getenv("S1_REGIME_DETECTION", "True").lower() in ("true", "1", "yes"))
REGIME_WINDOW_RATIO  = float(os.getenv("S1_REGIME_WINDOW_RATIO", 0.3)) # fraction of data for regime test
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

# ---------- Enhanced statistical functions ----------
def seasonal_detrend(series: pd.Series, period: int = None) -> Tuple[pd.Series, bool]:
    """
    Apply seasonal decomposition to remove seasonal patterns.
    Returns: (detrended_series, success_flag)
    """
    if not ENABLE_SEASONAL_DECOMP or len(series.dropna()) < MIN_SEASONAL_SAMPLES:
        return series, False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Use automatic period detection if not specified
            if period is None:
                period = min(SEASONAL_PERIOD, len(series.dropna()) // 3)
            
            # Ensure we have enough data for decomposition
            if len(series.dropna()) < 2 * period:
                return series, False
            
            # Forward fill NaNs for decomposition, then restore them
            series_filled = series.fillna(method='ffill').fillna(method='bfill')
            
            decomposition = seasonal_decompose(
                series_filled, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Remove seasonal component but keep trend + residual
            detrended = decomposition.trend + decomposition.resid
            
            # Restore original NaN positions
            detrended[series.isna()] = np.nan
            
            return detrended, True
            
    except Exception as e:
        # Fallback to original series if decomposition fails
        return series, False

def test_stationarity(series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int, bool]:
    """
    Test for stationarity using ADF test and apply differencing if needed.
    Returns: (stationary_series, num_differences, is_stationary)
    """
    if not STATIONARITY_TEST or len(series.dropna()) < MIN_Z_SAMPLES:
        return series, 0, True  # Assume stationary if insufficient data
    
    try:
        current_series = series.dropna()
        differences = 0
        
        for diff_order in range(max_diff + 1):
            if len(current_series) < MIN_Z_SAMPLES:
                break
                
            # Perform ADF test
            adf_result = adfuller(current_series, autolag='AIC')
            p_value = adf_result[1]
            
            if p_value <= ADF_PVALUE_THRESH:
                # Series is stationary
                if diff_order == 0:
                    return series, 0, True
                else:
                    # Return differenced series aligned with original index
                    diff_series = series.copy()
                    for i in range(differences):
                        diff_series = diff_series.diff()
                    return diff_series, differences, True
            
            # Apply differencing for next iteration
            if diff_order < max_diff:
                current_series = current_series.diff().dropna()
                differences += 1
        
        # If still not stationary after max differences, return last attempt
        diff_series = series.copy()
        for i in range(differences):
            diff_series = diff_series.diff()
        return diff_series, differences, False
        
    except Exception:
        return series, 0, True  # Fallback to original series

def detect_regime_change(series: pd.Series, window_ratio: float = 0.3) -> bool:
    """
    Detect potential regime changes in the time series.
    Returns True if a significant regime change is detected.
    """
    if not REGIME_DETECTION or len(series.dropna()) < MIN_Z_SAMPLES * 2:
        return False
    
    try:
        clean_series = series.dropna()
        n = len(clean_series)
        window_size = max(MIN_Z_SAMPLES, int(n * window_ratio))
        
        if n < 2 * window_size:
            return False
        
        # Compare recent window with historical window
        recent = clean_series.iloc[-window_size:]
        historical = clean_series.iloc[:window_size]
        
        # Use Mann-Whitney U test for distribution change
        statistic, p_value = stats.mannwhitneyu(
            historical, recent, alternative='two-sided'
        )
        
        return p_value < ADF_PVALUE_THRESH
        
    except Exception:
        return False

def adaptive_window_size(series: pd.Series, base_window: int) -> int:
    """
    Determine optimal window size based on data characteristics.
    """
    if not ADAPTIVE_WINDOW:
        return base_window
    
    try:
        clean_series = series.dropna()
        n = len(clean_series)
        
        if n < MIN_WINDOW_DAYS:
            return min(base_window, n)
        
        # Calculate data density (average time between observations)
        if len(clean_series) > 1:
            time_diff = clean_series.index.to_series().diff().dt.days.median()
            if pd.notna(time_diff) and time_diff > 0:
                # Adjust window based on data frequency
                density_factor = min(2.0, max(0.5, 6.0 / time_diff))
                adjusted_window = int(base_window * density_factor)
                return max(MIN_WINDOW_DAYS, min(MAX_WINDOW_DAYS, adjusted_window))
        
        return base_window
        
    except Exception:
        return base_window

def enhanced_mad(series: pd.Series, c: float = 1.4826) -> float:
    """
    Enhanced MAD calculation with better outlier handling.
    c = 1.4826 for normal distribution consistency, 0.6745 for robust z-score
    """
    try:
        clean_data = series.dropna()
        if len(clean_data) == 0:
            return np.nan
        
        median = np.median(clean_data)
        
        # Use iterative MAD calculation to handle extreme outliers
        deviations = np.abs(clean_data - median)
        mad = np.median(deviations)
        
        # Apply additional outlier filtering if MAD is very small
        if mad < MAD_EPS:
            # Use interquartile range as backup
            q75, q25 = np.percentile(clean_data, [75, 25])
            iqr = q75 - q25
            mad = max(mad, iqr / 2.0)
        
        return mad * c
        
    except Exception:
        return np.nan

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

    # Optional global winsorization to damp outliers before z (low-risk)
    if WINSOR_PCT and WINSOR_PCT > 0:
        try:
            lo, hi = np.nanpercentile(primary.values, [WINSOR_PCT, 100.0 - WINSOR_PCT])
            primary = primary.clip(lower=lo, upper=hi)
        except Exception:
            pass

    # ---------- robust z ----------
    def robust_z(series, win_days: int, min_periods: int = 1, mad_eps: float = 1e-9):
        """Enhanced time-based rolling robust z: 0.6745*(x - median)/MAD.
        - closed='left' excludes the current sample -> only uses historical data
        - min_periods ensures we have enough data to compute a stable median/MAD
        - MAD below mad_eps is treated as NaN to avoid inflated z
        - Integrates seasonal decomposition, stationarity testing, and regime detection
        """
        # Apply seasonal decomposition if enabled
        working_series = series.copy()
        seasonal_success = False
        if ENABLE_SEASONAL_DECOMP:
            working_series, seasonal_success = seasonal_detrend(working_series)
        
        # Test for stationarity and apply differencing if needed
        stationarity_info = {'differences': 0, 'is_stationary': True}
        if STATIONARITY_TEST:
            working_series, stationarity_info['differences'], stationarity_info['is_stationary'] = test_stationarity(working_series)
        
        # Detect regime changes
        regime_change = detect_regime_change(working_series) if REGIME_DETECTION else False
        
        # Adapt window size based on data characteristics and regime detection
        adaptive_window = adaptive_window_size(working_series, win_days)
        if regime_change:
            # Use shorter window if regime change detected
            adaptive_window = max(MIN_WINDOW_DAYS, adaptive_window // 2)
        
        # Calculate rolling statistics with enhanced methods
        r = working_series.rolling(f"{adaptive_window}D", closed="left", min_periods=min_periods)
        med = r.median()
        
        # Use enhanced MAD calculation
        mad = r.apply(lambda x: enhanced_mad(pd.Series(x), c=0.6745) if len(x) > 0 else np.nan, raw=False)
        
        # Guard tiny/zero MAD with improved threshold
        effective_mad_eps = max(mad_eps, MAD_EPS)
        mad = mad.where(mad > effective_mad_eps, np.nan)
        
        z_scores = (working_series - med) / mad
        
        # Add metadata for diagnostics
        z_scores.attrs = {
            'seasonal_decomposed': seasonal_success,
            'stationarity_info': stationarity_info,
            'regime_change_detected': regime_change,
            'adaptive_window_used': adaptive_window,
            'original_window': win_days
        }
        
        return z_scores

    # Calculate adaptive window size for this dataset
    effective_window = adaptive_window_size(primary, ROLL_WINDOW_DAYS)
    z = robust_z(primary, effective_window, min_periods=max(2, MIN_Z_SAMPLES//2), mad_eps=MAD_EPS)
    z_display = z.copy()

    def simple_robust_z(series: pd.Series) -> pd.Series | None:
        # Lightweight fallback when the main robust z yields no values (e.g., limited history)
        if series is None:
            return None
        s = series.astype(float)
        if s.notna().sum() < 3:
            return None
        med = s.median()
        mad = (s - med).abs().median()
        if pd.notna(mad) and mad > MAD_EPS:
            return 0.6745 * (s - med) / mad
        std = s.std(ddof=0)
        if pd.notna(std) and std > 0:
            return (s - s.mean()) / std
        return pd.Series(0.0, index=s.index)

    fallback_z = simple_robust_z(primary)
    if fallback_z is not None:
        if z_display.dropna().empty:
            z_display = fallback_z.reindex(z_display.index)
        else:
            z_display = z_display.combine_first(fallback_z.reindex(z_display.index))

    WATCH_Z, ALERT_Z = WATCH_Z_DEFAULT, Z_THRESHOLD
    WATCH_ANY_DROP = -0.5
    # Require a sufficient history and valid z for gating - use adaptive window info
    adaptive_window_used = getattr(z, 'attrs', {}).get('adaptive_window_used', effective_window)
    z_count = primary.rolling(f"{adaptive_window_used}D", closed="left").count()
    valid_z = (z_count >= MIN_Z_SAMPLES) & z.notna()
    z_ok = (z <= Z_THRESHOLD) & valid_z

    # Rolling median/IQR provide a local baseline so slow drifts still register
    try:
        primary_med = primary.rolling(f"{adaptive_window_used}D", closed="left").median()
    except Exception:
        primary_med = None
    try:
        primary_iqr = primary.rolling(f"{adaptive_window_used}D", closed="left").quantile(0.75) - \
                      primary.rolling(f"{adaptive_window_used}D", closed="left").quantile(0.25)
    except Exception:
        primary_iqr = None

    # ---------- rules ----------
    rule_vh_vs_base = (vh_lrdb <= MIN_ABS_DROP_DB_VH) if vh_lrdb is not None else pd.Series(False, index=df.index)
    rule_vv_vs_base = (vv_lrdb <= MIN_ABS_DROP_DB_VV) if vv_lrdb is not None else pd.Series(False, index=df.index)
    if ratio is not None and ratio_b is not None:
        safe = (ratio_b.astype(float).abs() > RATIO_EPS) & ratio.notna() & ratio_b.notna()
        rule_ratio_pct = pd.Series(False, index=df.index)
        rule_ratio_pct[safe] = ((ratio[safe] / ratio_b[safe]) - 1.0 <= -MIN_PCT_DROP_LINEAR)
    else:
        rule_ratio_pct = pd.Series(False, index=df.index)
    # Absolute low VH threshold: S1_VH_CURR is linear power, but the configured cutoff is in dB.
    # Convert dB -> linear and compare to avoid log-of-zero issues.
    if vh is not None:
        try:
            _VH_LIN_CUTOFF = float(10 ** (VH_ABS_DB_CUTOFF / 10.0))
            # vh is already coerced to numeric above; comparison keeps NaNs -> False
            rule_vh_low_abs = (vh <= _VH_LIN_CUTOFF)
        except Exception:
            rule_vh_low_abs = pd.Series(False, index=df.index)
    else:
        rule_vh_low_abs = pd.Series(False, index=df.index)

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
    def _nan0(x):
        if x is None:
            return 0.0
        # Handle Series (if multiple values returned due to duplicate index)
        if isinstance(x, pd.Series):
            x = x.iloc[0] if len(x) > 0 else np.nan
        return 0.0 if pd.isna(x) else float(x)
    
    def _safe_loc(series, t):
        """Safely get a single value from a series, handling duplicate indices."""
        if series is None or t not in series.index:
            return np.nan
        val = series.loc[t]
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else np.nan
        return val
    
    def _bool_at(s, t):
        try:
            val = _safe_loc(s, t)
            return (not pd.isna(val)) and bool(val)
        except:
            return False

    def _rules_at(t):
        z_val = _safe_loc(z, t)
        vh_lrdb_val = _safe_loc(vh_lrdb, t)
        return {
            "vh_db_drop":     _bool_at(rule_vh_vs_base, t),
            "vv_db_drop":     _bool_at(rule_vv_vs_base, t),
            "ratio_pct_drop": _bool_at(rule_ratio_pct, t),
            "vh_low_abs":     _bool_at(rule_vh_low_abs, t),
            "smooth":         _bool_at(rule_smooth, t),
            "z_watch":        (not pd.isna(z_val) and z_val <= WATCH_Z),
            "z_alert":        (not pd.isna(z_val) and z_val <= ALERT_Z),
            "small_drop":     (not pd.isna(vh_lrdb_val) and vh_lrdb_val <= WATCH_ANY_DROP),
        }

    def reasons_at(t):
        r = _rules_at(t)
        rs = []
        if r["vh_db_drop"] and vh_lrdb is not None:
            rs.append(f"VH logΔ ≤ {MIN_ABS_DROP_DB_VH:.1f} dB")
        elif r["small_drop"]:
            rs.append("VH slightly lower vs base")
        if r["vv_db_drop"] and vv_lrdb is not None:
            rs.append(f"VV logΔ ≤ {MIN_ABS_DROP_DB_VV:.1f} dB")
        if r["ratio_pct_drop"] and ratio_b is not None:
            rs.append(f"VH/VV drop ≥ {int(MIN_PCT_DROP_LINEAR*100)}% vs base")
        if r["vh_low_abs"]:
            rs.append(f"VH ≤ {VH_ABS_DB_CUTOFF:.1f} dB")
        z_val = _safe_loc(z, t)
        if not pd.isna(z_val):
            rs.append(f"z = {z_val:.1f}")
            
            # Add enhanced diagnostic info
            z_attrs = getattr(z, 'attrs', {})
            if z_attrs.get('seasonal_decomposed', False):
                rs.append("(seasonal-adjusted)")
            if z_attrs.get('regime_change_detected', False):
                rs.append("(regime-change)")
            if z_attrs.get('adaptive_window_used', 0) != z_attrs.get('original_window', 0):
                rs.append(f"(adaptive-window: {z_attrs.get('adaptive_window_used', 0)}d)")
                
        if r["vv_db_drop"] and r["smooth"]:
            rs.append("VV logΔ low & smooth")
        return ", ".join(rs) if rs else "Signals normal vs baseline"

    # Precompute rolling median of VH std for confidence once
    med_std_series = None
    if vh_std is not None:
        try:
            med_std_series = vh_std.rolling(f"{ROLL_WINDOW_DAYS}D", closed="left").median()
        except Exception:
            med_std_series = None

    def severity_confidence_at(t):
        z_t      = _nan0(_safe_loc(z, t))
        vh_lr_t  = _nan0(_safe_loc(vh_lrdb, t))
        vv_lr_t  = _nan0(_safe_loc(vv_lrdb, t))
        r = _rules_at(t)
        rule_count = sum([r["vh_db_drop"], r["vv_db_drop"], r["ratio_pct_drop"], r["vh_low_abs"], r["z_watch"]])
        rule_count_norm = _clip01(rule_count / 5.0)
        
        # Make severity calculation more robust
        z_sev = 0.0
        if not pd.isna(z_t) and z_t <= WATCH_Z:
            z_sev = _clip01((WATCH_Z - z_t) / (WATCH_Z - ALERT_Z + 1e-6))
        
        vh_sev = 0.0
        if not pd.isna(vh_lr_t) and vh_lr_t <= 0:
            vh_sev = _clip01((0 - vh_lr_t) / abs(MIN_ABS_DROP_DB_VH))
        
        vv_sev = 0.0
        if not pd.isna(vv_lr_t) and vv_lr_t <= 0:
            vv_sev = _clip01((0 - vv_lr_t) / abs(MIN_ABS_DROP_DB_VV))
        
        ratio_sev = 0.7 if r["ratio_pct_drop"] else 0.0
        persist_boost = 0.15 if _bool_at(alerts_mask, t) else 0.0
        baseline_boost = 0.0  # Detect gradual moisture rise against recent norm
        primary_med_val = _safe_loc(primary_med, t)
        primary_val = _safe_loc(primary, t)
        if not pd.isna(primary_med_val) and not pd.isna(primary_val):
            baseline_val = float(primary_med_val)
            current_val = float(primary_val)
            drop = baseline_val - current_val
            if drop > 0:
                spread = None
                primary_iqr_val = _safe_loc(primary_iqr, t)
                vh_std_val = _safe_loc(vh_std, t)
                if not pd.isna(primary_iqr_val):
                    spread = float(primary_iqr_val)
                elif not pd.isna(vh_std_val):
                    spread = float(vh_std_val)
                if spread is not None:
                    spread = max(spread, 1e-3)
                    baseline_boost = _clip01(drop / (spread * 1.5))
        sev01 = _clip01(0.35*z_sev + 0.30*vh_sev + 0.15*vv_sev + 0.20*ratio_sev + persist_boost + 0.20*baseline_boost)
        severity_0_100 = int(round(sev01 * 100))

        conf = 0.40
        conf += 0.25 * rule_count_norm
        conf += 0.15 * (1.0 if _bool_at(alerts_mask, t) else 0.0)
        
        # Simplified confidence calculation without rolling statistics that can vary
        vh_std_val = _safe_loc(vh_std, t)
        med_std_val = _safe_loc(med_std_series, t)
        if not pd.isna(vh_std_val) and not pd.isna(med_std_val):
            conf += 0.10 * (1.0 if float(vh_std_val) <= 0.9 * float(med_std_val) else 0.0)
                
        key_ok = int(t in z.index and not pd.isna(z_t)) + int(vh_lrdb is not None and t in vh_lrdb.index and not pd.isna(vh_lr_t)) + int(vv_lrdb is not None and t in vv_lrdb.index and not pd.isna(vv_lr_t))
        conf += 0.10 * _clip01(key_ok / 3.0)
        return severity_0_100, _clip01(conf)

    def classify_level_with_severity(t):
        r = _rules_at(t)
        # First determine base level using rules (this should be deterministic)
        base_level = ("Alert" if (_bool_at(alerts_mask, t) or r["z_alert"])
                     else ("Watch" if (r["vh_db_drop"] or r["vv_db_drop"] or r["ratio_pct_drop"] or r["z_watch"])
                           else "Healthy"))
        
        # Get severity - if calculation fails, fall back to base level
        try:
            sev, _ = severity_confidence_at(t)
            # Ensure severity is a valid number
            if not isinstance(sev, (int, float)) or pd.isna(sev):
                return base_level
        except:
            return base_level
        
        # Simplified stability logic with more conservative thresholds
        if base_level == "Alert":
            # If rules say Alert, keep it unless severity is very low
            return "Alert" if sev >= 10 else ("Watch" if sev >= 5 else "Healthy")
        elif base_level == "Watch":
            # If rules say Watch, apply severity-based upgrades/downgrades
            if sev >= 75:  # Very high severity -> Alert
                return "Alert"
            elif sev < 15:  # Very low severity -> Healthy
                return "Healthy"
            else:
                return "Watch"
        else:  # base_level == "Healthy"
            # If rules say Healthy, only upgrade with very high severity
            z_val = _safe_loc(z, t)
            if sev >= 70 or (not pd.isna(z_val) and z_val <= WATCH_Z):
                return "Alert"
            elif sev >= 40:
                return "Watch"
            else:
                return "Healthy"

    rows = []
    for t in df.index:
        sev, conf = severity_confidence_at(t)
        # Calculate status once to avoid inconsistency from multiple calls
        status = classify_level_with_severity(t)
        z_val = _safe_loc(z, t)
        rows.append({
            "date": t,
            "S1_VH_CURR": float(_safe_loc(vh, t)) if not pd.isna(_safe_loc(vh, t)) else np.nan,
            "S1_VV_CURR": float(_safe_loc(vv, t)) if not pd.isna(_safe_loc(vv, t)) else np.nan,
            "S1_VH_LOGRATIO_DB": float(_safe_loc(vh_lrdb, t)) if not pd.isna(_safe_loc(vh_lrdb, t)) else np.nan,
            "S1_VV_LOGRATIO_DB": float(_safe_loc(vv_lrdb, t)) if not pd.isna(_safe_loc(vv_lrdb, t)) else np.nan,
            "S1_VH_VV_CURR": float(_safe_loc(ratio, t)) if not pd.isna(_safe_loc(ratio, t)) else np.nan,
            "S1_VH_VV_DIFF": float(_safe_loc(ratio_d, t)) if not pd.isna(_safe_loc(ratio_d, t)) else np.nan,
            "zscore": float(_safe_loc(z_display, t)) if not pd.isna(_safe_loc(z_display, t)) else np.nan,
            "status": status,
            "severity_0_100": int(sev),
            "confidence_0_1": float(conf),
            "reasons": reasons_at(t),
            "actions": ACTIONS[status],  # Use the cached status instead of calling function again
        })
    insights_df = pd.DataFrame(rows).sort_values("date")

    # --- Post-process to harden the insights table ---
    def _postprocess_insights(d: pd.DataFrame) -> pd.DataFrame:
        if d is None or d.empty:
            return d
        d = d.copy()
        # Ensure datetime index/column correctness
        d["date"] = pd.to_datetime(d.get("date"), errors="coerce")

        # Clip numeric ranges and fix types
        if "severity_0_100" in d.columns:
            d["severity_0_100"] = pd.to_numeric(d["severity_0_100"], errors="coerce").fillna(0)
            d["severity_0_100"] = d["severity_0_100"].clip(lower=0, upper=100).round().astype(int)
        else:
            d["severity_0_100"] = 0

        if "confidence_0_1" in d.columns:
            d["confidence_0_1"] = pd.to_numeric(d["confidence_0_1"], errors="coerce").fillna(0.0)
            d["confidence_0_1"] = d["confidence_0_1"].clip(lower=0.0, upper=1.0)
        else:
            d["confidence_0_1"] = 0.0

        # Normalize status to known set
        valid = {"Alert": "Alert", "Watch": "Watch", "Healthy": "Healthy"}
        d["status"] = d.get("status").map(lambda s: valid.get(str(s), "Healthy"))

        # Deduplicate by date: prefer higher status, then higher severity, then higher confidence
        pri = {"Healthy": 0, "Watch": 1, "Alert": 2}
        d["_pri"] = d["status"].map(pri).fillna(0)
        d = d.sort_values(["date", "_pri", "severity_0_100", "confidence_0_1"], ascending=[True, False, False, False])
        d = d.drop_duplicates(subset=["date"], keep="first")
        d = d.drop(columns=["_pri"], errors="ignore")
        d = d.sort_values("date")
        return d

    insights_df = _postprocess_insights(insights_df)

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
              <div class="lg-bar" role="progressbar" aria-label="{d['label']} share"
                   aria-valuemin="0" aria-valuemax="100" aria-valuenow="{d['pct']:.1f}">
                <span class="fill"></span>
                <span class="bar-text">{d['ha']:.2f} ha</span>
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
    if insights_df is None or (hasattr(insights_df, "empty") and insights_df.empty):
        return None
    d = insights_df.copy()
    if "date" in d.columns:
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
    if "date" in d.columns:
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")

    for col in ["severity_0_100","confidence_0_1"]:
        if col not in d.columns: d[col] = pd.NA

    if "severity_0_100" in d.columns:
        d["severity_0_100"] = (pd.to_numeric(d["severity_0_100"], errors="coerce")
                                 .round().astype("Int64").astype(str).replace("<NA>",""))
    if "confidence_0_1" in d.columns:
        d["confidence_0_1"] = pd.to_numeric(d["confidence_0_1"], errors="coerce").map(
            lambda x: (f"{x:.2f}" if pd.notna(x) else "")
        )
    cols = [c for c in ["date","status","actions"] if c in d.columns]
    return d[cols]

def prepare_technical_view(insights_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if insights_df is None or (hasattr(insights_df, "empty") and insights_df.empty):
        return None
    d = insights_df.copy()
    d = d.drop(columns=[c for c in ["status","actions"] if c in d.columns], errors="ignore")
    ordered = ["date","zscore","severity_0_100","confidence_0_1","S1_VH_CURR","S1_VV_CURR","S1_VH_LOGRATIO_DB","S1_VV_LOGRATIO_DB",
            "S1_VH_VV_CURR","S1_VH_VV_DIFF"]
    ordered = [c for c in ordered if c in d.columns]
    d = d[ordered]
    if "zscore" in d.columns:
        z_vals = pd.to_numeric(d["zscore"], errors="coerce")
        if z_vals.isna().any():
            # Fill display-only z values from a simple robust z using the best available signal
            fallback_source = None
            for candidate in ["S1_VH_LOGRATIO_DB","S1_VH_CURR","S1_VH_VV_DIFF","S1_VH_VV_CURR"]:
                if candidate in d.columns:
                    series = pd.to_numeric(d[candidate], errors="coerce")
                    if series.notna().sum() >= 3:
                        fallback_source = series
                        break
            if fallback_source is not None:
                med = fallback_source.median()
                mad = (fallback_source - med).abs().median()
                if pd.notna(mad) and mad > MAD_EPS:
                    fallback = 0.6745 * (fallback_source - med) / mad
                else:
                    std = fallback_source.std(ddof=0)
                    if pd.notna(std) and std > 0:
                        fallback = (fallback_source - fallback_source.mean()) / std
                    else:
                        fallback = pd.Series(0.0, index=fallback_source.index)
                z_vals = z_vals.combine_first(fallback.reindex(z_vals.index))
        d["zscore"] = z_vals
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values("date", ascending=False)
        d["date"] = d["date"].dt.strftime("%Y-%m-%d")
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
    thresholds=(0.25, 0.40, 0.5),
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
