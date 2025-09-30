# app_core/tasks.py
from celery import shared_task
from django.conf import settings
from pathlib import Path
from datetime import datetime
import json

from .models import AnalysisJob
from analysis.analysis import run_analysis_from_notebook

# --- NEW: scale/area helpers ---
import rasterio
from pyproj import Geod
from analysis.insights import classify_and_area


# ----------------------- small path/url helpers -----------------------
def _media_rel_from_url(path_or_url: str) -> str:
    """Return a media-relative path (no leading /media/)."""
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
    """Convert media-relative (or URL) to filesystem path under MEDIA_ROOT."""
    rel = _media_rel_from_url(path_or_url)
    return Path(getattr(settings, "MEDIA_ROOT", "media")) / rel


def _media_urlify(path_or_url: str) -> str:
    """Return a proper /media/... URL from any path or URL-ish input."""
    if not path_or_url:
        return ""
    s = str(path_or_url)

    if s.startswith(("http://", "https://")):
        return s

    try:
        p = Path(s)
        media_root = Path(getattr(settings, "MEDIA_ROOT", "media")).resolve()
        if p.is_absolute():
            try:
                s = str(p.resolve().relative_to(media_root))
            except Exception:
                pass
    except Exception:
        pass

    rel = _media_rel_from_url(s)
    return f"{getattr(settings, 'MEDIA_URL', '/media/')}{rel}"


# ----------------------- main task -----------------------
@shared_task
def run_waterlogging_analysis(job_id: int):
    """End-to-end local analysis (no GEE here). Produces overlay, probe, hotspots,
    risk_tif + area_by_class + total_ha, and links the latest timeseries CSV for the field.
    """
    job = AnalysisJob.objects.get(id=job_id)
    job.status = "running"
    job.message = "Starting…"
    job.save(update_fields=["status", "message"])

    try:
        field = job.field

        # ---------- Resolve stack ----------
        stacks_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)

        # Prefer path saved on Field (if you use that), else search by field id
        tif_path = Path(getattr(field, "stack_path", "") or "")
        if not tif_path.is_file():
            specific = sorted(
                list(stacks_dir.glob(f"*field_{field.id}_*.tif")) +
                list(stacks_dir.glob(f"*_{field.id}_*.tif")) +
                list(stacks_dir.glob(f"*{field.id}*.tif")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if specific:
                tif_path = specific[0]
            else:
                # fall back to newest stack
                cand = sorted(stacks_dir.glob("*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
                if cand:
                    tif_path = cand[0]
                else:
                    raise FileNotFoundError(
                        f"No stack found for field {field.id}. "
                        f"Put a TIF in {stacks_dir} containing 'field_{field.id}' in its name."
                    )

        job.message = f"Using stack: {tif_path.name}"
        job.save(update_fields=["message"])

        # ---------- Run notebook-like local analysis ----------
        aoi_geojson = field.geom
        result = run_analysis_from_notebook(
            aoi_geojson,
            stack_tif_path=str(tif_path),
        )

        # ---------- Normalize outputs ----------
        overlay_url    = _media_urlify(result.get("overlay_png_url", ""))
        hotspots_url   = _media_urlify(result.get("hotspots_url", ""))
        probe_bin_url  = _media_urlify(result.get("probe_bin_url", ""))
        probe_meta_url = _media_urlify(result.get("probe_meta_url", ""))

        risk_tif_path = result.get("risk_tif_path", "") or ""
        risk_tif_url  = _media_urlify(result.get("risk_tif_url", "")) if risk_tif_path else ""

        # Fallback: discover a risk_*.tif for this field/job
        ov_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "overlays"
        if (not risk_tif_path) or (not Path(risk_tif_path).exists()):
            cand = sorted(ov_dir.glob(f"risk_field_{field.id}_*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                cand = sorted(ov_dir.glob(f"risk_job_{job.id}_*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                cand = sorted(ov_dir.glob("risk_from_probe_*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                cand = sorted(ov_dir.glob("risk_*.tif"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                risk_tif_path = str(cand[0])
                risk_tif_url  = _media_urlify(risk_tif_path)

        # Sanity: expected files should exist (only check local media files)
        for label, url in [
            ("overlay", overlay_url),
            ("hotspots", hotspots_url),
            ("probe_bin", probe_bin_url),
            ("probe_meta", probe_meta_url),
        ]:
            if url and not url.startswith(("http://", "https://")):
                target = _fs_from_media(url)
                if not target.exists():
                    raise FileNotFoundError(f"{label} missing on disk -> {target}")

        # ---------- NEW: compute area_by_class & total_ha ----------
        area_by_class = None
        total_ha = None
        if risk_tif_path and Path(risk_tif_path).exists():
            with rasterio.open(risk_tif_path) as ds:
                rows, cols = ds.height, ds.width
                left, bottom, right, top = ds.bounds
            # geodesic bbox area (EPSG:4326 expected) → pixel area
            g = Geod(ellps="WGS84")
            area_m2, _ = g.polygon_area_perimeter(
                [left, right, right, left, left],
                [bottom, bottom, top, top, bottom]
            )
            px_area_m2 = abs(area_m2) / float(rows * cols)

            # thresholds align with UI (Healthy/Watch/Concern/Alert)
            area_by_class, total_ha = classify_and_area(
                risk_tif_path,
                thresholds=(0.20, 0.40, 0.60),
                scale_from=None,  # risk tif is already 0..1
                default_pixel_area_m2=px_area_m2
            )

        # ---------- Attach latest timeseries CSV if missing ----------
        merged = dict(job.result or {})
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

        # ---------- Save job result ----------
        merged.update({
            "bounds": result.get("bounds"),
            "overlay_png_url": overlay_url,
            "hotspots_url": hotspots_url,
            "probe_bin_url": probe_bin_url,
            "probe_meta_url": probe_meta_url,
            "risk_tif_path": risk_tif_path,
            "risk_tif_url": risk_tif_url,
        })
        if area_by_class is not None:
            merged["area_by_class"] = area_by_class
        if total_ha is not None:
            merged["total_ha"] = total_ha

        job.result = merged
        job.status = "done"
        job.message = "Completed"
        job.save(update_fields=["result", "status", "message"])

    except Exception as e:
        job.status = "failed"
        job.message = f"{type(e).__name__}: {e}"
        job.save(update_fields=["status", "message"])
        raise
