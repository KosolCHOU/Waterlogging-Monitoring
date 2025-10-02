# app_core/tasks.py
from celery import shared_task
from django.conf import settings
from pathlib import Path
from datetime import datetime
import json

from .models import AnalysisJob

# Make analysis imports conditional to prevent blocking during startup
try:
    from analysis.analysis import run_analysis_from_notebook
    from analysis.insights import classify_and_area, compute_temporal_engine_s1

    ANALYSIS_AVAILABLE = True
except Exception as e:
    print(f"[TASKS] Analysis functions not available: {e}")
    ANALYSIS_AVAILABLE = False

    def run_analysis_from_notebook(*args, **kwargs):
        raise Exception("Analysis functions not configured")

    def classify_and_area(*args, **kwargs):
        raise Exception("Analysis functions not configured")

    def compute_temporal_engine_s1(*args, **kwargs):
        raise Exception("Analysis functions not configured")


# --- NEW: scale/area helpers ---
try:
    from pyproj import Geod

    GEOSPATIAL_AVAILABLE = True
except Exception as e:
    print(f"[TASKS] Geospatial libraries not available: {e}")
    GEOSPATIAL_AVAILABLE = False


# ----------------------- small path/url helpers -----------------------
def _media_rel_from_url(path_or_url: str) -> str:
    """Return a media-relative path (no leading /media/)."""
    if not path_or_url:
        return ""
    rel = str(path_or_url)

    media_url = getattr(settings, "MEDIA_URL", "/media/")
    if rel.startswith(media_url):
        rel = rel[len(media_url) :]
    elif rel.startswith("/probe-bin/"):
        # Handle probe-bin URLs - map them to probes directory
        filename = rel[len("/probe-bin/") :]
        return f"probes/{filename}"

    rel = rel.lstrip("/")

    # Legacy alias: older jobs stored /media/probe-bin/... which should resolve to probes/
    legacy_prefix = "probe-bin/"
    if rel.startswith(legacy_prefix):
        rel = f"probes/{rel[len(legacy_prefix):]}"

    while rel.startswith("media/"):
        rel = rel[6:]
        if rel.startswith(legacy_prefix):
            rel = f"probes/{rel[len(legacy_prefix):]}"
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
    # Import rasterio at function level to ensure it's available throughout
    if GEOSPATIAL_AVAILABLE:
        import rasterio

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
            # Look for field-specific stacks with exact patterns only
            specific = sorted(
                list(stacks_dir.glob(f"*field_{field.id}_*.tif"))
                + list(stacks_dir.glob(f"*_{field.id}_*.tif")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Validate that found files are really for this field (not just containing the digit)
            valid_specific = []
            for candidate in specific:
                if (
                    f"field_{field.id}_" in candidate.name
                    or f"_{field.id}_" in candidate.name
                ):
                    # Also check if it has enough bands
                    if GEOSPATIAL_AVAILABLE:
                        try:
                            with rasterio.open(str(candidate)) as src:
                                if src.count >= 11:
                                    valid_specific.append(candidate)
                        except Exception:
                            continue
                    else:
                        valid_specific.append(candidate)

            if valid_specific:
                tif_path = valid_specific[0]
            else:
                # No field-specific stack found, fallback to newest general stack
                # But first check if it has the right number of bands (11)
                cand = sorted(
                    stacks_dir.glob("*.tif"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                suitable_stack = None
                if GEOSPATIAL_AVAILABLE:
                    for stack_candidate in cand:
                        try:
                            with rasterio.open(str(stack_candidate)) as src:
                                if src.count >= 11:  # Need at least 11 bands
                                    suitable_stack = stack_candidate
                                    break
                        except Exception:
                            continue

                if suitable_stack:
                    tif_path = suitable_stack
                else:
                    # No suitable stack found, generate a new one
                    from datetime import datetime as dt

                    ts = dt.now().strftime("%Y%m%d_%H%M%S")
                    new_stack_path = stacks_dir / f"stack_field_{field.id}_{ts}.tif"

                    try:
                        # Import the export function
                        from analysis.engine import export_stack_from_geom

                        export_stack_from_geom(field.geom, str(new_stack_path))
                        if new_stack_path.exists():
                            tif_path = new_stack_path
                        else:
                            raise FileNotFoundError(
                                f"Failed to generate stack for field {field.id}"
                            )
                    except Exception as e:
                        raise FileNotFoundError(
                            f"No suitable stack found for field {field.id} and failed to generate new one: {e}"
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
        overlay_url = _media_urlify(result.get("overlay_png_url", ""))
        hotspots_url = _media_urlify(result.get("hotspots_url", ""))
        probe_bin_url = _media_urlify(result.get("probe_bin_url", ""))
        probe_meta_url = _media_urlify(result.get("probe_meta_url", ""))

        risk_tif_path = result.get("risk_tif_path", "") or ""
        risk_tif_url = (
            _media_urlify(result.get("risk_tif_url", "")) if risk_tif_path else ""
        )

        # Fallback: discover a risk_*.tif for this field/job
        ov_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "overlays"
        if (not risk_tif_path) or (not Path(risk_tif_path).exists()):
            cand = sorted(
                ov_dir.glob(f"risk_field_{field.id}_*.tif"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not cand:
                cand = sorted(
                    ov_dir.glob(f"risk_job_{job.id}_*.tif"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if not cand:
                cand = sorted(
                    ov_dir.glob("risk_from_probe_*.tif"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if not cand:
                cand = sorted(
                    ov_dir.glob("risk_*.tif"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if cand:
                risk_tif_path = str(cand[0])
                risk_tif_url = _media_urlify(risk_tif_path)

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
                [left, right, right, left, left], [bottom, bottom, top, top, bottom]
            )
            px_area_m2 = abs(area_m2) / float(rows * cols)

            # thresholds align with UI (Healthy/Watch/Concern/Alert)
            area_by_class, total_ha = classify_and_area(
                risk_tif_path,
                thresholds=(0.20, 0.40, 0.60),
                scale_from=None,  # risk tif is already 0..1
                default_pixel_area_m2=px_area_m2,
            )

        # ---------- Attach latest timeseries CSV if missing ----------
        merged = dict(job.result or {})
        if not merged.get("timeseries_path"):
            ts_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "timeseries"
            if ts_dir.exists():
                cand = sorted(
                    ts_dir.glob(f"timeseries_field_{field.id}_*.csv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if cand:
                    latest = cand[0]
                    rel = latest.relative_to(
                        Path(getattr(settings, "MEDIA_ROOT", "media"))
                    )
                    merged["timeseries_path"] = str(latest)
                    merged["timeseries_file"] = (
                        f"{getattr(settings, 'MEDIA_URL', '/media/')}{rel.as_posix()}"
                    )

        # ---------- Generate plot and insights ----------
        plot_url = None
        insights_csv_url = None

        if merged.get("timeseries_path"):
            ts_path = merged["timeseries_path"]
            try:
                alerts_df, insights_df, plot_png, insights_csv = (
                    compute_temporal_engine_s1(
                        ts_path,
                        media_root=getattr(settings, "MEDIA_ROOT", "media"),
                    )
                )

                if plot_png:
                    plot_url = _media_urlify(plot_png)
                    merged["plot_path"] = plot_png

                if insights_csv:
                    insights_csv_url = _media_urlify(insights_csv)
                    merged["insights_csv_path"] = insights_csv

            except Exception as plot_error:
                print(f"[TASKS] Plot generation failed: {plot_error}")
                import traceback

                traceback.print_exc()

        # ---------- Save job result ----------
        merged.update(
            {
                "bounds": result.get("bounds"),
                "overlay_png_url": overlay_url,
                "hotspots_url": hotspots_url,
                "probe_bin_url": probe_bin_url,
                "probe_meta_url": probe_meta_url,
                "risk_tif_path": risk_tif_path,
                "risk_tif_url": risk_tif_url,
                "plot_url": plot_url,
                "insights_csv_url": insights_csv_url,
            }
        )
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
