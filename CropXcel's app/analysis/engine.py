# analysis/engine.py
"""
Engine functions to export Sentinel-1 stacks from Google Earth Engine.
This file is refactored for easier reuse in Django views/tasks.
"""

import os, json
from datetime import date as _py_date, timedelta
import ee, geemap

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