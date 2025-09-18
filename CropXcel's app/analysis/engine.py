# analysis/engine.py
"""
Engine functions to export Sentinel-1 stacks from Google Earth Engine.
This file is refactored for easier reuse in Django views/tasks.
"""

import os, json
from datetime import date as _py_date, timedelta
import ee, geemap
from datetime import date as _py_date, datetime, timedelta
try:
    from dateutil.relativedelta import relativedelta  # optional but handy
except Exception:
    relativedelta = None  # will fallback if missing

# --- Earth Engine auth/init ---
try:
    ee.Initialize(project='angelic-edition-466705-r8')
    print("[EE] Initialized with existing credentials.")
except Exception:
    print("[EE] Authenticating...")
    ee.Authenticate()
    ee.Initialize(project='angelic-edition-466705-r8')
    print("[EE] Authenticated and initialized.")

# ---------------------------
# Helpers
# ---------------------------

def load_s1_ic(start, end, geom, orbit_pass=None):
    """Load Sentinel-1 GRD collection with VV+VH bands."""
    ic = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    if orbit_pass:
        ic = ic.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
    return ic


def db_to_linear_keep(img):
    vv_lin = ee.Image(10).pow(img.select('VV').divide(10)).rename('VV_lin')
    vh_lin = ee.Image(10).pow(img.select('VH').divide(10)).rename('VH_lin')
    return (ee.Image(img)
            .addBands(vv_lin)
            .addBands(vh_lin)
            .copyProperties(img, img.propertyNames()))


def lin_to_db_safe(img_lin):
    masked = img_lin.updateMask(img_lin.gt(0))
    return masked.log10().multiply(10)


def median_core(ic_lin, geom):
    """Compute median VV/VH/ratio stack for baseline window."""
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


# ---------------------------
# Main Export Function
# ---------------------------

def export_stack_from_geom(geom_geojson: dict,
                           out_tif: str,
                           scale_m=10,
                           orbit_pass=None,
                           crs=None) -> str:
    """
    Export Sentinel-1 stack for given AOI geometry.
    Args:
        geom_geojson : dict - GeoJSON geometry
        out_tif      : str  - output filename (local path)
        scale_m      : int  - export scale (default 10 m)
        orbit_pass   : str  - 'ASCENDING', 'DESCENDING', or None
        crs          : str  - optional CRS (e.g., 'EPSG:32648')
    Returns:
        str: path to saved .tif
    """
    geom = ee.Geometry(geom_geojson)

    # Define time windows
    END        = (_py_date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    EVENT_DAYS = 15
    BASE_DAYS  = 45
    GAP_DAYS   = 5

    end   = ee.Date(END)
    evt_s = end.advance(-EVENT_DAYS, 'day')
    evt_e = end
    base_e = evt_s.advance(-GAP_DAYS, 'day')
    base_s = base_e.advance(-BASE_DAYS, 'day')

    # Load image collections
    s1_evt_raw  = load_s1_ic(evt_s,  evt_e,  geom, orbit_pass).map(db_to_linear_keep)
    s1_base_raw = load_s1_ic(base_s, base_e, geom, orbit_pass).map(db_to_linear_keep)

    # Event image
    evt_img = ee.Image(ee.Algorithms.If(
        s1_evt_raw.size().gt(0),
        s1_evt_raw.sort('system:time_start', False).first(),
        ee.Image.constant([0,0]).updateMask(ee.Image.constant(0)).rename(['VV_lin','VH_lin'])
    )).clip(geom)

    evt_vv   = evt_img.select('VV_lin')
    evt_vh   = evt_img.select('VH_lin')
    evt_ratio= evt_vh.divide(evt_vv).updateMask(evt_vv.gt(0))

    evt_core = evt_vv.rename('S1_VV_CURR')
    evt_core = evt_core.addBands(evt_vh.rename('S1_VH_CURR'))
    evt_core = evt_core.addBands(evt_ratio.rename('S1_VH_VV_CURR'))

    # Baseline
    base_core_lin = median_core(s1_base_raw, geom)
    base_core  = base_core_lin.select('VV_lin').rename('S1_VV_BASE')
    base_core  = base_core.addBands(base_core_lin.select('VH_lin').rename('S1_VH_BASE'))
    base_core  = base_core.addBands(base_core_lin.select('VH_VV').rename('S1_VH_VV_BASE'))

    # Ratios & STD
    vv_logratio_db = lin_to_db_safe(evt_vv.divide(base_core_lin.select('VV_lin'))).rename('S1_VV_LOGRATIO_DB')
    vh_logratio_db = lin_to_db_safe(evt_vh.divide(base_core_lin.select('VH_lin'))).rename('S1_VH_LOGRATIO_DB')
    vh_vv_diff     = evt_ratio.subtract(base_core_lin.select('VH_VV')).rename('S1_VH_VV_DIFF')

    vv_std = std_band(s1_evt_raw, 'VV_lin', geom)
    vh_std = std_band(s1_evt_raw, 'VH_lin', geom)

    # Final stack
    stack = ee.Image.cat([
        evt_core, base_core,
        vv_logratio_db, vh_logratio_db, vh_vv_diff,
        vv_std, vh_std
    ]).toFloat()

    # Export
    print(f"[EXPORT] Saving stack: {out_tif}")
    geemap.ee_export_image(
        stack,
        out_tif,
        scale=scale_m,
        region=geom,
        file_per_band=False,
        crs=crs
    )
    print("[OK] Exported:", out_tif)

    return out_tif

def export_s1_timeseries(geom_geojson: dict,
                         out_csv: str,
                         start: str | None = None,
                         end: str | None = None,
                         step_days: int = 10,
                         event_days: int = 15,
                         base_days: int = 45,
                         gap_days: int = 5,
                         orbit_pass: str | None = None,
                         tz: str = "Asia/Phnom_Penh") -> str:
    """
    Export a Sentinel-1 time series (AOI means per step) to CSV.

    Reuses existing helpers in engine.py:
      - load_s1_ic
      - db_to_linear_keep
      - lin_to_db_safe
      - median_core
      - std_band

    Args:
        geom_geojson: GeoJSON geometry dict for the AOI.
        out_csv: Output CSV filepath.
        start: ISO date 'YYYY-MM-DD' (default = 4 months before 'end').
        end: ISO date 'YYYY-MM-DD' (default = yesterday).
        step_days: Step size (days) for sliding end-date windows.
        event_days: Event window length.
        base_days: Baseline window length.
        gap_days: Gap between baseline end and event start.
        orbit_pass: 'ASCENDING' | 'DESCENDING' | None.
        tz: Timezone used for human-readable date strings.

    Returns:
        Path to the saved CSV.
    """
    import pandas as _pd  # keep dependency local

    # ---- Defaults for start/end
    _end = (end or (_py_date.today() - timedelta(days=1)).strftime("%Y-%m-%d"))
    if start is None:
        if relativedelta:
            _start = (_py_date.today() - relativedelta(months=4)).strftime("%Y-%m-%d")
        else:
            _start = (_py_date.today() - timedelta(days=120)).strftime("%Y-%m-%d")
    else:
        _start = start

    # ---- Build date steps (inclusive of the last step)
    def _date_steps(s_str: str, e_str: str, step: int) -> ee.List:
        d0 = datetime.fromisoformat(s_str)
        d1 = datetime.fromisoformat(e_str)
        steps = []
        while d0 < d1:
            d2 = min(d0 + timedelta(days=step), d1)
            steps.append(ee.Date(d2.strftime("%Y-%m-%d")))
            if d2 == d1:
                break
            d0 = d2
        return ee.List(steps)

    steps = _date_steps(_start, _end, step_days)
    geom = ee.Geometry(geom_geojson)

    # ---- Min/max day helper for diagnostics
    MS_PER_DAY = 86400000.0
    def _minmax_day(ic):
        times = ee.List(ic.aggregate_array('system:time_start'))
        day_min = ee.Number(ee.Algorithms.If(times.size().gt(0),
                                             ee.Number(times.reduce(ee.Reducer.min())).divide(MS_PER_DAY),
                                             ee.Number(0)))
        day_max = ee.Number(ee.Algorithms.If(times.size().gt(0),
                                             ee.Number(times.reduce(ee.Reducer.max())).divide(MS_PER_DAY),
                                             ee.Number(0)))
        return day_min.toFloat(), day_max.toFloat()

    # ---- Build one Feature per step_end
    def _feature_for_step(step_end):
        step_end = ee.Date(step_end)
        evt_start  = step_end.advance(-event_days, 'day')
        base_end   = evt_start.advance(-gap_days, 'day')
        base_start = base_end.advance(-base_days, 'day')

        # Load collections and convert to linear (reuse helpers)
        ic_evt_raw  = load_s1_ic(evt_start, step_end.advance(1, 'day'), geom, orbit_pass).map(db_to_linear_keep)
        ic_base_raw = load_s1_ic(base_start, base_end.advance(1, 'day'), geom, orbit_pass).map(db_to_linear_keep)

        # Baseline median (VV_lin, VH_lin, VH_VV via median_core)
        base_core_lin = median_core(ic_base_raw, geom)

        # Event "current" = median over event window (to be robust)
        evt_core_lin = median_core(ic_evt_raw, geom)

        vv_curr   = evt_core_lin.select('VV_lin').rename('S1_VV_CURR')
        vh_curr   = evt_core_lin.select('VH_lin').rename('S1_VH_CURR')
        ratio_cur = evt_core_lin.select('VH_VV').rename('S1_VH_VV_CURR')

        vv_base   = base_core_lin.select('VV_lin').rename('S1_VV_BASE')
        vh_base   = base_core_lin.select('VH_lin').rename('S1_VH_BASE')
        ratio_bas = base_core_lin.select('VH_VV').rename('S1_VH_VV_BASE')

        # Log-ratios (reuse lin_to_db_safe)
        vv_logratio_db = lin_to_db_safe(vv_curr.divide(vv_base)).rename('S1_VV_LOGRATIO_DB')
        vh_logratio_db = lin_to_db_safe(vh_curr.divide(vh_base)).rename('S1_VH_LOGRATIO_DB')
        vh_vv_diff     = ratio_cur.subtract(ratio_bas).rename('S1_VH_VV_DIFF')

        # STD within event window (reuse std_band; expects linear bands)
        vv_std = std_band(ic_evt_raw, 'VV_lin', geom).rename('S1_VV_STD')
        vh_std = std_band(ic_evt_raw, 'VH_lin', geom).rename('S1_VH_STD')

        stack = ee.Image.cat([
            vv_curr, vh_curr, ratio_cur,
            vv_base, vh_base, ratio_bas,
            vv_logratio_db, vh_logratio_db, vh_vv_diff,
            vv_std, vh_std
        ]).toFloat()

        # Region mean stats (robust settings for larger AOIs)
        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4
        )

        # Diagnostics
        evt_ct  = ic_evt_raw.size()
        base_ct = ic_base_raw.size()
        evt_min_d, evt_max_d   = _minmax_day(ic_evt_raw)
        base_min_d, base_max_d = _minmax_day(ic_base_raw)

        props = stats.combine(ee.Dictionary({
            # Millis timestamps (authoritative)
            'date_ms': step_end.millis(),
            'evt_start_ms':  evt_start.millis(),
            'evt_end_ms':    step_end.millis(),
            'base_start_ms': base_start.millis(),
            'base_end_ms':   base_end.millis(),

            # Human-friendly labels (do not use for indexing)
            'date_local': step_end.format('YYYY-MM-dd', tz),
            'S1_EVENT_START_LOCAL': evt_start.format('YYYY-MM-dd', tz),
            'S1_EVENT_END_LOCAL':   step_end.format('YYYY-MM-dd', tz),
            'S1_BASE_START_LOCAL':  base_start.format('YYYY-MM-dd', tz),
            'S1_BASE_END_LOCAL':    base_end.format('YYYY-MM-dd', tz),

            # Counts & coverage diagnostics
            'S1_EVENT_COUNT': evt_ct,
            'S1_BASE_COUNT':  base_ct,
            'S1_EVENT_DAY_MIN': evt_min_d,
            'S1_EVENT_DAY_MAX': evt_max_d,
            'S1_BASE_DAY_MIN':  base_min_d,
            'S1_BASE_DAY_MAX':  base_max_d
        }), overwrite=True)

        return ee.Feature(None, props)

    fc = ee.FeatureCollection(steps.map(_feature_for_step))

    # ---- Bring once to client & write CSV locally
    recs = fc.getInfo().get('features', [])
    rows = []
    for f in recs:
        p = f.get('properties', {})
        rows.append({
            "date_local": p.get('date_local'),

            "S1_VV_CURR": p.get('S1_VV_CURR'),
            "S1_VH_CURR": p.get('S1_VH_CURR'),
            "S1_VH_VV_CURR": p.get('S1_VH_VV_CURR'),

            "S1_VV_BASE": p.get('S1_VV_BASE'),
            "S1_VH_BASE": p.get('S1_VH_BASE'),
            "S1_VH_VV_BASE": p.get('S1_VH_VV_BASE'),

            "S1_VV_LOGRATIO_DB": p.get('S1_VV_LOGRATIO_DB'),
            "S1_VH_LOGRATIO_DB": p.get('S1_VH_LOGRATIO_DB'),
            "S1_VH_VV_DIFF": p.get('S1_VH_VV_DIFF'),

            "S1_VV_STD": p.get('S1_VV_STD'),
            "S1_VH_STD": p.get('S1_VH_STD'),

            "S1_EVENT_COUNT": p.get('S1_EVENT_COUNT'),
            "S1_BASE_COUNT":  p.get('S1_BASE_COUNT'),
            "S1_EVENT_START_LOCAL": p.get('S1_EVENT_START_LOCAL'),
            "S1_EVENT_END_LOCAL":   p.get('S1_EVENT_END_LOCAL'),
            "S1_BASE_START_LOCAL":  p.get('S1_BASE_START_LOCAL'),
            "S1_BASE_END_LOCAL":    p.get('S1_BASE_END_LOCAL'),

            "date_ms": p.get('date_ms'),
            "evt_start_ms": p.get('evt_start_ms'),
            "evt_end_ms": p.get('evt_end_ms'),
            "base_start_ms": p.get('base_start_ms'),
            "base_end_ms": p.get('base_end_ms'),
        })

    if not rows:
        # Write an empty CSV with headers to keep downstream code simple
        _pd.DataFrame(columns=[
            "date_local","S1_VV_CURR","S1_VH_CURR","S1_VH_VV_CURR",
            "S1_VV_BASE","S1_VH_BASE","S1_VH_VV_BASE",
            "S1_VV_LOGRATIO_DB","S1_VH_LOGRATIO_DB","S1_VH_VV_DIFF",
            "S1_VV_STD","S1_VH_STD",
            "S1_EVENT_COUNT","S1_BASE_COUNT",
            "S1_EVENT_START_LOCAL","S1_EVENT_END_LOCAL","S1_BASE_START_LOCAL","S1_BASE_END_LOCAL",
            "date_ms","evt_start_ms","evt_end_ms","base_start_ms","base_end_ms"
        ]).to_csv(out_csv, index=False)
        return out_csv

    df = _pd.DataFrame(rows)

    # Convert ms → Cambodia-local naive datetime for indexing
    ts = _pd.to_datetime(df['date_ms'], unit='ms', utc=True).dt.tz_convert(tz)
    df['date'] = ts.dt.tz_localize(None)
    df = df.sort_values('date').set_index('date')

    metric_cols = [
        "S1_VV_CURR","S1_VH_CURR","S1_VH_VV_CURR",
        "S1_VV_BASE","S1_VH_BASE","S1_VH_VV_BASE",
        "S1_VV_LOGRATIO_DB","S1_VH_LOGRATIO_DB","S1_VH_VV_DIFF",
        "S1_VV_STD","S1_VH_STD"
    ]
    df = df.dropna(how='all', subset=metric_cols)

    df.to_csv(out_csv, float_format="%.6f")
    print(f"[OK] Saved time series → {out_csv} ({len(df)} rows)")
    return out_csv
