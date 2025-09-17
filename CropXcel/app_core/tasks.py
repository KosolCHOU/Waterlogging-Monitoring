# app_core/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from pathlib import Path
import json, folium

from .models import AnalysisJob
from analysis.run_analysis_from_notebook import run_analysis_from_notebook


def _media_rel_from_url(url: str) -> str:
    """
    Turn '/media/hotspots/x.geojson', 'media/hotspots/x.geojson',
    or even '/media/media/hotspots/x.geojson' into 'hotspots/x.geojson'.
    """
    rel = (url or "")
    # Remove one MEDIA_URL prefix if present
    if settings.MEDIA_URL and rel.startswith(settings.MEDIA_URL):
        rel = rel[len(settings.MEDIA_URL):]
    rel = rel.lstrip("/")
    # Remove any remaining leading 'media/' fragments (defensive)
    while rel.startswith("media/"):
        rel = rel[6:]
    return rel


@shared_task
def run_waterlogging_analysis(job_id: int):
    job = AnalysisJob.objects.get(id=job_id)
    job.status = "running"
    job.message = "Starting…"
    job.save(update_fields=["status", "message"])

    try:
        field = job.field
        result = run_analysis_from_notebook(aoi_geojson=field.geom)

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        ov_dir = media_root / "overlays"
        ov_dir.mkdir(parents=True, exist_ok=True)

        # --- bounds
        gj = field.geom
        coords = gj["coordinates"][0] if gj["type"] == "Polygon" else gj["coordinates"][0][0]
        lats = [pt[1] for pt in coords]; lons = [pt[0] for pt in coords]
        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
        result["bounds"] = bounds
        job.result = result
        center = [(bounds[0][0]+bounds[1][0])/2, (bounds[0][1]+bounds[1][1])/2]

        # --- base & overlay
        m = folium.Map(location=center, zoom_start=16, tiles=None, control_scale=True)
        folium.TileLayer("Esri.WorldImagery", name="Basemap").add_to(m)
        folium.raster_layers.TileLayer(
            tiles=result["tile_url"], name="Waterlogging risk", attr="Earth Engine",
            opacity=0.9, overlay=True
        ).add_to(m)

        # --- hotspots (robust resolution; won't crash if missing)
        hot_rel = _media_rel_from_url(result.get("hotspots_url", ""))
        hotspots_path = (media_root / hot_rel).resolve()
        try:
            gj_hot = json.loads(hotspots_path.read_text(encoding="utf-8"))
            for f in gj_hot.get("features", []):
                lon, lat = f["geometry"]["coordinates"]
                r = f.get("properties", {}).get("risk", 0)
                folium.CircleMarker(
                    location=[lat, lon], radius=6, weight=2, color="#ff00ff",
                    fill=True, fill_opacity=1
                ).add_child(
                    folium.Popup(f"<b>Hotspot</b><br>Risk: {(r*100):.1f}%")
                ).add_to(m)
        except FileNotFoundError:
            job.message = f"Hotspots not found: {hotspots_path.name}"
            job.save(update_fields=["message"])
        except json.JSONDecodeError:
            job.message = f"Hotspots JSON invalid: {hotspots_path.name}"
            job.save(update_fields=["message"])

        folium.LayerControl(collapsed=False).add_to(m)
        m.fit_bounds(bounds)

        # --- write HTML overlay
        html_name = f"field_{field.id}_{result['last_step_end']}.html"
        html_path = ov_dir / html_name
        m.save(str(html_path))

        # --- expose to frontend
        result["overlay_html_url"] = f"{settings.MEDIA_URL}overlays/{html_name}"
        job.overlay_html = f"overlays/{html_name}"
        result.update({
        "bounds": bounds,                 # [[s,w],[n,e]]
        "palette": ["#006837","#1a9850","#fee08b","#f46d43","#d73027"],  # if you have it
        "scale_m": 10,                    # native scale used
        })
        job.result = result
        job.overlay_html = f"overlays/{html_name}"   # not "media/overlays/..."
        job.save()
        job.status = "ready"
        job.message = "Finished"
        job.finished_at = timezone.now()
        job.save(update_fields=["result", "overlay_html", "status", "message", "finished_at"])

    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "message", "finished_at"])
        raise