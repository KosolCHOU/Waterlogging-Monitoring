# app_core/tasks.py
from celery import shared_task
from django.conf import settings
from pathlib import Path
import json
from datetime import datetime

from .models import AnalysisJob
from analysis.analysis import run_analysis_from_notebook


# ---------- Helpers ----------
def _media_rel_from_url(path_or_url: str) -> str:
    """
    Turn '/media/hotspots/x.geojson', 'media/hotspots/x.geojson',
    'hotspots/x.geojson', or even '/media/media/hotspots/x.geojson'
    into 'hotspots/x.geojson'.
    """
    if not path_or_url:
        return ""
    rel = str(path_or_url)

    media_url = getattr(settings, "MEDIA_URL", "/media/")
    if rel.startswith(media_url):
        rel = rel[len(media_url):]

    rel = rel.lstrip("/")
    while rel.startswith("media/"):
        rel = rel[6:]
    return rel


def _fs_from_media(path_or_url: str) -> Path:
    """
    Convert any media URL/relative path into an absolute filesystem path
    under MEDIA_ROOT.
    """
    rel = _media_rel_from_url(path_or_url)
    return Path(getattr(settings, "MEDIA_ROOT", "media")) / rel


def _media_urlify(path_or_url: str) -> str:
    """
    Convert a relative media path or filesystem path into a browser URL
    under MEDIA_URL (idempotent if a full http(s) URL).
    """
    if not path_or_url:
        return ""
    s = str(path_or_url)

    # pass-through for http(s)
    if s.startswith(("http://", "https://")):
        return s

    # if it's a filesystem path inside MEDIA_ROOT, make it relative
    try:
        p = Path(s)
        media_root = Path(getattr(settings, "MEDIA_ROOT", "media")).resolve()
        if p.is_absolute():
            try:
                s = str(p.resolve().relative_to(media_root))
            except Exception:
                # not under MEDIA_ROOT -> leave as-is; caller should only hand us media files
                pass
    except Exception:
        pass

    rel = _media_rel_from_url(s)
    return f"{getattr(settings, 'MEDIA_URL', '/media/')}{rel}"


# ---------- Task ----------
@shared_task
def run_waterlogging_analysis(job_id: int):
    job = AnalysisJob.objects.get(id=job_id)
    job.status = "running"
    job.message = "Starting…"
    job.save(update_fields=["status", "message"])

    try:
        field = job.field

        # 1) Resolve stack path (prefer model attribute; else newest in /media/stacks)
        tif_path = Path(getattr(field, "stack_path", "") or "")
        if not tif_path.is_file():
            stacks_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "stacks"
            stacks_dir.mkdir(parents=True, exist_ok=True)
            cand = sorted(stacks_dir.glob("*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                tif_path = cand[0]
            else:
                raise FileNotFoundError(
                    f"No stack found for field {field.id}. "
                    f"Set FieldAOI.stack_path or put a .tif in {stacks_dir}"
                )

        job.message = f"Using stack: {tif_path}"
        job.save(update_fields=["message"])

        # 2) Run the notebook-aligned analysis
        #    Expected to return URLs *or* filesystem paths. We normalize below.
        aoi_geojson = field.geom  # GeoJSON-like dict
        result = run_analysis_from_notebook(aoi_geojson, stack_tif_path=str(tif_path))

        # 3) Normalize everything to MEDIA_URL for the frontend
        overlay_url   = _media_urlify(result.get("overlay_png_url", ""))
        hotspots_url  = _media_urlify(result.get("hotspots_url", ""))
        probe_bin_url = _media_urlify(result.get("probe_bin_url", ""))
        probe_meta_url= _media_urlify(result.get("probe_meta_url", ""))

        # 4) (Optional) sanity check: the files actually exist on disk if they’re local
        for label, url in [
            ("overlay", overlay_url),
            ("hotspots", hotspots_url),
            ("probe_bin", probe_bin_url),
            ("probe_meta", probe_meta_url),
        ]:
            if url and not url.startswith(("http://", "https://")):
                if not _fs_from_media(url).exists():
                    raise FileNotFoundError(f"{label} missing on disk -> {_fs_from_media(url)}")

        # --- keep anything saved earlier (e.g., timeseries_file/timeseries_path) ---
        merged = dict(job.result or {})
        merged.update({
            "bounds": result.get("bounds"),
            "overlay_png_url": overlay_url,
            "hotspots_url": hotspots_url,
            "probe_bin_url": probe_bin_url,
            "probe_meta_url": probe_meta_url,
        })

        # (optional) If timeseries wasn't set yet, auto-attach the newest CSV for this field
        if not merged.get("timeseries_path"):
            ts_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "timeseries"
            if ts_dir.exists():
                cand = sorted(ts_dir.glob(f"timeseries_field_{field.id}_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
                if cand:
                    latest = cand[0]
                    rel = latest.relative_to(Path(getattr(settings, "MEDIA_ROOT", "media")))
                    merged["timeseries_path"] = str(latest)
                    merged["timeseries_file"] = f"{getattr(settings, 'MEDIA_URL', '/media/')}{rel.as_posix()}"

        job.result = merged
        job.status = "done"
        job.message = "Completed"
        job.save(update_fields=["result", "status", "message"])

    except Exception as e:
        job.status = "failed"
        job.message = f"{type(e).__name__}: {e}"
        job.save(update_fields=["status", "message"])
        raise