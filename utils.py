# PLACE INTO: CropXcel/analysis/utils.py

# ======== CONFIG (logical expectations) ========
TIF_PATH = OUT_STACK  # or a literal path
# These are the "base" logical band names your pipeline exports (11 total)
EXPECTED_BASES = [
    'S1_VV_CURR', 'S1_VH_CURR', 'S1_VH_VV_CURR',
    'S1_VV_BASE', 'S1_VH_BASE', 'S1_VH_VV_BASE',
    'S1_VV_LOGRATIO_DB', 'S1_VH_LOGRATIO_DB', 'S1_VH_VV_DIFF',
    'S1_VV_STD', 'S1_VH_STD'
]
MIN_VALID_PCT_WARN = 1.0  # warn if a band has <1% valid pixels

import os, re
import numpy as np
import pandas as pd
import rasterio

# Matches your naming scheme and extracts the logical "base" key.
# Examples:
#  S1_VV_CURR_D20250830              -> S1_VV_CURR
#  S1_VV_BASE_S20250801_E20250820    -> S1_VV_BASE
#  S1_VV_STD_W20250816_20250831      -> S1_VV_STD
#  S1_VH_VV_DIFF_D20250830_S...      -> S1_VH_VV_DIFF
BASE_RE = re.compile(
    r'^(?P<base>S1_(?:VV|VH|VH_VV)_(?:CURR|BASE|LOGRATIO_DB|DIFF|STD))'
    r'(?:_D\d{8}|_S\d{8}_E\d{8}|_W\d{8}_\d{8})?$'
)

def normalize_band_name(raw: str) -> str:
    raw = (raw or "").strip()
    m = BASE_RE.match(raw)
    return m.group('base') if m else raw  # fall back to raw if it doesn't match

def _get_band_name(src, idx1):
    """Try multiple places for a band name; fallback to Band{idx}."""
    if src.descriptions and len(src.descriptions) >= idx1 and src.descriptions[idx1-1]:
        return src.descriptions[idx1-1]
    tags = src.tags(idx1) or {}
    for key in ('name', 'BAND_NAME', 'long_name', 'DESCRIPTION', 'band_name'):
        if key in tags and str(tags[key]).strip():
            return str(tags[key]).strip()
    return f"Band{idx1}"

def _valid_percent(src, idx1):
    """% valid using masks, nodata, and NaN."""
    arr = src.read(idx1, masked=True)
    if isinstance(arr, np.ma.MaskedArray):
        total = arr.size
        valid = total - np.count_nonzero(arr.mask)
        if np.issubdtype(arr.dtype, np.floating):
            # NaNs are already masked in many cases, but this is safe
            valid -= np.count_nonzero(np.isnan(arr.filled(np.nan)))
        return 100.0 * valid / total if total else 0.0
    else:
        data = arr
        total = data.size
        valid = total
        nd = src.nodata
        if nd is not None:
            valid -= np.count_nonzero(data == nd)
        if np.issubdtype(data.dtype, np.floating):
            valid -= np.count_nonzero(np.isnan(data))
        return 100.0 * valid / total if total else 0.0

def main_check_tif(tif_path=TIF_PATH):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        print(f"[INFO] File: {tif_path}")
        print(f"[INFO] Size: {src.width} x {src.height}, Bands: {src.count}, CRS: {src.crs}")
        if src.transform:
            print(f"[INFO] Pixel size ~ {abs(src.transform.a):.2f} x {abs(src.transform.e):.2f} map units")

        band_raw = [_get_band_name(src, i) for i in range(1, src.count+1)]
        rows = []
        for i in range(1, src.count+1):
            raw = band_raw[i-1]
            base = normalize_band_name(raw)
            vp = _valid_percent(src, i)
            rows.append({
                "index_1based": i,
                "band_name_raw": raw,
                "band_base": base,
                "valid_percent": round(vp, 3),
                "dtype": str(src.dtypes[i-1]),
            })

    df = pd.DataFrame(rows)

    # --- Reporting ---
    print("\n=== BAND SUMMARY (raw → base) ===")
    print(df.to_string(index=False))

    # Base-name coverage checks (logical)
    found_bases = set(df["band_base"].tolist())
    expected_bases = set(EXPECTED_BASES)

    missing = [b for b in EXPECTED_BASES if b not in found_bases]
    extras  = [b for b in sorted(found_bases - expected_bases)]

    print("\n=== CHECKS (logical bands) ===")
    print(f"[INFO] Found {len(found_bases)} unique logical bases.")
    if not missing:
        print("[OK] All expected logical bands present.")
    else:
        print(f"[WARN] Missing logical bands: {missing}")

    if extras:
        print(f"[NOTE] Extra logical bands detected (unexpected): {extras}")
    else:
        print("[OK] No unexpected logical bands.")

    # Valid coverage
    low_valid = df[df["valid_percent"] < MIN_VALID_PCT_WARN]
    print("\n=== COVERAGE ===")
    if not low_valid.empty:
        print(f"[WARN] Bands with very low valid coverage (<{MIN_VALID_PCT_WARN}%):")
        print(low_valid[["index_1based","band_name_raw","valid_percent"]].to_string(index=False))
        print("       Tip: near-zero coverage on *_BASE or *_CURR can occur if the window has 0 scenes.")
    else:
        print("[OK] All bands have reasonable valid coverage.")

# Run
main_check_tif()