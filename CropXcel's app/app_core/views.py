# app_core/views.py
import json
from pathlib import Path
from datetime import datetime

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .serializers import FieldSerializer, JobSerializer
from .tasks import run_waterlogging_analysis

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, Http404, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_http_methods
from django.db.models import Prefetch

from .models import FieldAOI, AnalysisJob
from analysis.engine import export_stack_from_geom
from analysis.hotspots import extract_hotspots

# New: local geodesic area (no GEE)
from shapely.geometry import shape
from pyproj import Geod

@require_http_methods(["POST"])
def aoi_upload(request):
    """
    Save a drawn AOI to /media/aoi, export its stack, create Field + Job,
    and return info with a link to the risk map.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
        feature = payload.get("feature")
        if not feature:
            return HttpResponseBadRequest("Missing 'feature'")

        geom_geojson = feature["geometry"] if feature.get("type") == "Feature" else feature
        g = shape(geom_geojson)

        if g.is_empty:
            return HttpResponseBadRequest("Empty geometry")

        # Geodesic area (m² → ha)
        geod = Geod(ellps="WGS84")
        area_m2, _perim = geod.geometry_area_perimeter(g)
        area_ha = abs(area_m2) / 10_000.0

        # Save AOI file
        media_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "aoi"
        media_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        aoi_path = media_dir / f"field_{ts}.geojson"
        aoi_path.write_text(json.dumps(
            {"type": "Feature", "geometry": geom_geojson, "properties": {}},
            indent=2
        ), encoding="utf-8")

        # Export stack to /media/stacks/
        stack_dir = Path(getattr(settings, "MEDIA_ROOT", "media")) / "stacks"
        stack_dir.mkdir(parents=True, exist_ok=True)
        tif_path = stack_dir / f"stack_{ts}.tif"
        try:
            export_stack_from_geom(geom_geojson, str(tif_path))
            tif_exported = tif_path.name
        except Exception as gee_err:
            print("⚠️ GEE export failed:", gee_err)
            tif_exported = None

        # --- NEW: create FieldAOI + AnalysisJob ---
        field = FieldAOI.objects.create(
            geom=geom_geojson,
            # add other required fields if your model needs them (e.g. name="Field 58")
        )
        job = AnalysisJob.objects.create(field=field, status="queued", message="Created from AOI upload")

        if settings.DEBUG:
            # run synchronously in dev → avoids 404 when opening /fields/{id}/risk/
            run_waterlogging_analysis(job.id)
        else:
            # enqueue for Celery in production
            run_waterlogging_analysis.delay(job.id)

        return JsonResponse({
            "ok": True,
            "area_ha": round(float(area_ha), 4),
            "aoi_file": aoi_path.name,
            "tif_file": tif_exported,
            "field_id": field.id,
            "job_id": job.id,
            "next_url": f"/fields/{field.id}/risk/",
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Invalid payload: {e}")

# ---------- Page: risk map ----------
def risk_map(request, field_id: int):
    field = get_object_or_404(FieldAOI, id=field_id)

    # 1) Find latest job for this field (any status)
    job = (AnalysisJob.objects
           .filter(field=field)
           .order_by("-id")
           .first())

    # Optional: allow forcing a re-run with ?rerun=1
    force_rerun = request.GET.get("rerun") == "1"

    # 2) If no job yet, or force rerun → create one and kick it off
    if (job is None) or force_rerun:
        job = AnalysisJob.objects.create(field=field, status="queued", message="Created from risk_map")
        if settings.DEBUG:
            # run synchronously during dev so the page can show quickly
            run_waterlogging_analysis(job.id)
        else:
            run_waterlogging_analysis.delay(job.id)

    # 3) While job isn’t done (or has empty result), show a lightweight "processing" page
    if job.status != "done" or not job.result:
        html = f"""
        <!doctype html>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="3">  <!-- auto-refresh every 3s -->
        <title>Analyzing…</title>
        <style>
          body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 2rem; }}
          .pill {{ display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#eef7f3; }}
        </style>
        <h1>Field #{field_id}: analysis in progress</h1>
        <p>Status: <span class="pill">{job.status}</span></p>
        <p>Message: {job.message or "Working…"}</p>
        <p>This page will refresh automatically. You can also <a href="?rerun=1">re-run</a>.</p>
        """
        return HttpResponse(html)

    # 4) When done → render your map as before
    bounds = job.result.get("bounds") or [[field.geom["coordinates"][0][0][1],
                                           field.geom["coordinates"][0][0][0]],
                                          [field.geom["coordinates"][0][2][1],
                                           field.geom["coordinates"][0][2][0]]]
    ctx = {
        "job_id": job.id,
        "bounds": json.dumps(bounds),
        "tile_url": "",
        "overlay_png": job.result.get("overlay_png_url") or "",
        "hotspots_url": job.result.get("hotspots_url") or "",
    }
    return render(request, "risk_map.html", ctx)

# ---------- API: probe (hover/click sampling) ----------
@require_http_methods(["GET"])
def probe(request, job_id: int):
    """
    Return nearest hotspot info at (lat, lon) using pre-computed hotspots.geojson
    """
    try:
        lat = float(request.GET.get("lat"))
        lon = float(request.GET.get("lon"))
    except (TypeError, ValueError):
        return HttpResponseBadRequest("lat & lon are required")

    job = get_object_or_404(AnalysisJob, id=job_id)
    hotspots_url = job.result.get("hotspots_url")
    if not hotspots_url:
        return JsonResponse({"value": None, "level": None, "source": "none"})

    # Resolve local file path
    media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
    rel = hotspots_url.lstrip("/")
    if rel.startswith("media/"):
        rel = rel[6:]
    fpath = media_root / rel

    if not fpath.exists():
        return JsonResponse({"value": None, "level": None, "source": "missing"})

    try:
        gj = json.loads(fpath.read_text(encoding="utf-8"))
        feats = gj.get("features", [])
        if not feats:
            return JsonResponse({"value": None, "level": None, "source": "empty"})

        # haversine helper
        def haversine(y1, x1, y2, x2):
            R = 6371000.0
            dy = math.radians(y2 - y1)
            dx = math.radians(x2 - x1)
            a = math.sin(dy / 2) ** 2 + math.cos(math.radians(y1)) * math.cos(math.radians(y2)) * math.sin(dx / 2) ** 2
            return 2 * R * math.asin(math.sqrt(a))

        # find nearest hotspot
        best = None
        for f in feats:
            geom = f.get("geometry", {})
            if geom.get("type") == "Point":
                x, y = geom.get("coordinates", [None, None])
                if x is None or y is None:
                    continue
                d = haversine(lat, lon, y, x)
                if (best is None) or (d < best[0]):
                    best = (d, f)

        if not best:
            return JsonResponse({"value": None, "level": None, "source": "nohotspot"})

        props = best[1].get("properties", {})
        return JsonResponse({
            "value": round(float(props.get("risk_pct", 0)) / 100.0, 3),  # convert % back to 0–1
            "level": props.get("level"),
            "reason": props.get("reason"),
            "action": props.get("action"),
            "area_ha": props.get("area_ha"),
            "source": "hotspots"
        })

    except Exception as e:
        return JsonResponse({"value": None, "level": None, "source": f"error: {e}"})
    
# ---------- DRF API ----------
class FieldViewSet(viewsets.ModelViewSet):
    """
    Minimal ViewSet so router.register('fields', FieldViewSet, ...) works.
    Provides:
      - GET /api/fields/                 (list)
      - POST /api/fields/                (create)
      - GET /api/fields/{id}/            (retrieve)
      - GET /api/fields/{id}/latest_job/ (latest analysis job result)
      - POST /api/fields/{id}/analyze/   (enqueue local analysis via Celery)
    """
    queryset = FieldAOI.objects.all().order_by("-id")
    serializer_class = FieldSerializer

    @action(detail=True, methods=["get"])
    def latest_job(self, request, pk=None):
        field = self.get_object()
        job = (AnalysisJob.objects
               .filter(field=field, status__in=["done", "running", "queued", "failed"])
               .order_by("-id")
               .first())
        if not job:
            return Response({"detail": "No jobs yet."}, status=status.HTTP_404_NOT_FOUND)
        return Response(JobSerializer(job).data, status=200)

    @action(detail=True, methods=["post"])
    def analyze(self, request, pk=None):
        """
        Enqueue local analysis (no GEE). Expects the FieldAOI to have stack path info.
        """
        field = self.get_object()
        # create a job record
        job = AnalysisJob.objects.create(field=field, status="queued", message="Queued by API")
        # fire Celery task
        run_waterlogging_analysis.delay(job.id)
        return Response({"ok": True, "job_id": job.id}, status=202)

def home(request):
    # If you have a fields list page, redirect there instead.
    return render(request, "home.html")
