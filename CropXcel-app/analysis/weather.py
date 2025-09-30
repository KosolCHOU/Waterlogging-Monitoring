# analysis/services/weather.py
import datetime as dt
from typing import Dict, Any, List, Tuple
import requests
from shapely.geometry import shape as shp_shape

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def _latlon_from_geom(geom) -> Tuple[float, float]:
    if isinstance(geom, dict):
        g = shp_shape(geom)
        c = g.centroid
        return float(c.y), float(c.x)
    c = geom.centroid
    return float(c.y), float(c.x)

def get_forecast_for_field(field) -> Dict[str, Any]:
    lat, lon = _latlon_from_geom(field.geom)
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Bangkok",
        "hourly": ",".join([
            "precipitation","precipitation_probability","rain","showers",
            "temperature_2m","wind_speed_10m"
        ]),
        "daily": ",".join([
            "precipitation_sum","precipitation_probability_max","rain_sum",
            "showers_sum","temperature_2m_max","temperature_2m_min","wind_speed_10m_max"
        ]),
        "forecast_days": 7,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=12)
    r.raise_for_status()
    return _summarize(r.json())

def _summarize(data: Dict[str, Any]) -> Dict[str, Any]:
    daily = data.get("daily", {}) or {}
    hourly = data.get("hourly", {}) or {}

    days = len(daily.get("time", []) or [])
    daily_rows: List[Dict[str, Any]] = []
    for i in range(days):
        daily_rows.append({
            "date": daily["time"][i],
            "rain_mm": float(daily.get("rain_sum", [0]*days)[i] or 0.0),
            "precip_mm": float(daily.get("precipitation_sum", [0]*days)[i] or 0.0),
            "prob_max": int(daily.get("precipitation_probability_max", [0]*days)[i] or 0),
            "tmin": float(daily.get("temperature_2m_min", [0]*days)[i] or 0.0),
            "tmax": float(daily.get("temperature_2m_max", [0]*days)[i] or 0.0),
            "wind_max": float(daily.get("wind_speed_10m_max", [0]*days)[i] or 0.0),
        })

    # Build compact 72h series
    htime = hourly.get("time", []) or []
    prob  = hourly.get("precipitation_probability", []) or []
    rain  = hourly.get("rain", []) or []
    now_idx = 0
    end = min(now_idx + 72, len(htime))
    hourly72 = []
    peak_prob = 0
    peak_idx = now_idx
    total_rain_72 = 0.0
    for j in range(now_idx, end):
        hh = (htime[j] or "")[-5:]  # "HH:MM"
        p  = int(prob[j] or 0)
        r  = float(rain[j] or 0.0)
        hourly72.append({"h": hh, "prob": p, "rain": round(r, 2)})
        total_rain_72 += r
        if p > peak_prob:
            peak_prob, peak_idx = p, j

    # Quick “today” facts from first daily row
    today = daily_rows[0] if daily_rows else {
        "tmin": 0, "tmax": 0, "prob_max": 0, "rain_mm": 0, "wind_max": 0
    }

    # Rules of thumb (advisories)
    advisories: List[str] = []
    flags = []
    if total_rain_72 >= 30 or peak_prob >= 70:
        advisories.append("Heavy rain likely within 72 hours. Prepare drainage and clear outlets.")
        flags.append("heavy_rain_soon")
    wet_days = sum(1 for d in daily_rows[:5] if d["precip_mm"] >= 10 or d["prob_max"] >= 60)
    if wet_days >= 3:
        advisories.append("Prolonged wet spell expected (≥3 days). Avoid fertilizer application and reduce irrigation.")
        flags.append("wet_spell")
    if any(d["wind_max"] >= 35 for d in daily_rows[:5]):
        advisories.append("Strong winds possible. Secure pumps, tarps and young seedlings.")
        flags.append("wind_risk")
    if not advisories:
        advisories.append("No severe weather signals. Maintain current schedule and monitor field moisture.")

    peak_time = htime[peak_idx][-5:] if htime else None

    return {
        "source": "open-meteo.com",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "daily": daily_rows,
        "today": {
            "tmin": today["tmin"], "tmax": today["tmax"],
            "prob_max": today["prob_max"], "rain_mm": today["rain_mm"],
            "wind_max": today["wind_max"]
        },
        "hourly72": hourly72,
        "rain_mm_72h": round(total_rain_72, 1),
        "precip_prob_peak_72h": int(peak_prob),
        "precip_prob_peak_time": peak_time,
        "advisories": advisories,
        "flags": flags,
    }