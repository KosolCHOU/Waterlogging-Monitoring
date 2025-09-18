# app_core/tasks.py
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from pathlib import Path
import json

from .models import AnalysisJob
from analysis.analysis import run_analysis_from_notebook

from pathlib import Path
from datetime import datetime


def _media_urlify(path_or_url: str) -> str:
    """
    Normalize any path like 'hotspots/x.geojson' or '/media/hotspots/x.geojson'
    to a browser-ready URL under MEDIA_URL.
    """
    if not path_or_url:
        return ""
    media_url = getattr(settings, "MEDIA_URL", "/media/")
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    rel = path_or_url
    if rel.startswith(media_url):
        rel = rel[len(media_url):]
    rel = rel.lstrip("/")
    while rel.startswith("media/"):
        rel = rel[6:]
    return f"{media_url}{rel}"

# app_core/tasks.py  (replace the function body)
@shared_task
def run_waterlogging_analysis(job_id: int):
    job = AnalysisJob.objects.get(id=job_id)
    job.status = "running"
    job.message = "Starting…"
    job.save(update_fields=["status", "message"])

    try:
        from pathlib import Path
        from django.conf import settings
        from analysis.analysis import run_analysis_from_notebook  # ← your renamed module

        field = job.field

        # 1) Resolve stack path (prefer model attribute; fallback to newest in /media/stacks)
        tif_path = Path(getattr(field, "stack_path", "") or "")
        if not tif_path.is_file():
            stacks_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "stacks"
            cand = sorted(stacks_dir.glob("*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                tif_path = cand[0]
            else:
                raise FileNotFoundError(f"No stack found for field {field.id} (set FieldAOI.stack_path or put a .tif in {stacks_dir})")

        # 2) Run the notebook-aligned analysis (returns overlay/hotspots/bounds)
        aoi_geojson = field.geom  # must be a GeoJSON-like dict
        result = run_analysis_from_notebook(aoi_geojson, stack_tif_path=str(tif_path))

        # 3) Normalize URLs to MEDIA_URL (works for both '/media/..' and 'hotspots/..')
        def _urlify(rel_or_url: str) -> str:
            if not rel_or_url:
                return ""
            media_url = getattr(settings, "MEDIA_URL", "/media/")
            if rel_or_url.startswith(("http://", "https://")):
                return rel_or_url
            rel = rel_or_url
            if rel.startswith(media_url):
                rel = rel[len(media_url):]
            rel = rel.lstrip("/")
            while rel.startswith("media/"):
                rel = rel[6:]
            return f"{media_url}{rel}"

        job.result = {
            "bounds": result.get("bounds"),
            "overlay_png_url": _urlify(result.get("overlay_png_url", "")),
            "hotspots_url": _urlify(result.get("hotspots_url", "")),
            "probe_bin_url": _urlify(result.get("probe_bin_url", "")),
            "probe_meta_url": _urlify(result.get("probe_meta_url", "")),
        }
        job.status = "done"
        job.message = "Completed"
        job.save(update_fields=["result", "status", "message"])

    except Exception as e:
        job.status = "failed"
        job.message = f"{type(e).__name__}: {e}"
        job.save(update_fields=["status", "message"])
        raise