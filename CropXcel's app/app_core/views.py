# app_core/views.py
import os
import json
import math
from pathlib import Path
from datetime import datetime

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .serializers import FieldSerializer, JobSerializer
from .tasks import run_waterlogging_analysis

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, Http404, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_http_methods
from django.db.models import Prefetch
from django.urls import reverse

from .models import FieldAOI, AnalysisJob
from analysis.engine import export_stack_from_geom, export_s1_timeseries
from analysis.insights import compute_temporal_engine_s1, build_insights_html, classify_and_area

# New: local geodesic area (no GEE)
from shapely.geometry import shape, mapping
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
        area_m2, _ = geod.geometry_area_perimeter(g)
        area_ha = abs(area_m2) / 10_000.0

        # Save AOI file
        media_dir = Path(settings.MEDIA_ROOT) / "aoi"
        media_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        aoi_path = media_dir / f"field_{ts}.geojson"
        aoi_path.write_text(json.dumps(
            {"type": "Feature", "geometry": geom_geojson, "properties": {}},
            indent=2
        ), encoding="utf-8")

        # Export stack → /media/stacks/
        stack_dir = Path(settings.MEDIA_ROOT) / "stacks"
        stack_dir.mkdir(parents=True, exist_ok=True)
        tif_path = stack_dir / f"stack_{ts}.tif"
        try:
            export_stack_from_geom(geom_geojson, str(tif_path))
            tif_exported = tif_path.name if tif_path.exists() else None
        except Exception as gee_err:
            print("⚠️ GEE export failed:", gee_err)
            tif_exported = None

        # Create Field + Job
        field = FieldAOI.objects.create(geom=geom_geojson)
        job = AnalysisJob.objects.create(field=field, status="queued", message="Created from AOI upload")

        # Always record stack_path if file exists
        job.result = {**(job.result or {}), "stack_path": str(tif_path) if tif_exported else None}
        job.save(update_fields=["result"])

        # Run analysis sync/async
        if settings.DEBUG:
            run_waterlogging_analysis(job.id)
        else:
            run_waterlogging_analysis.delay(job.id)

        # Export time-series → /media/timeseries/
        ts_dir = Path(settings.MEDIA_ROOT) / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ts_csv = ts_dir / f"timeseries_field_{field.id}_{stamp}.csv"

        media_url = settings.MEDIA_URL.rstrip("/")
        csv_rel = f"timeseries/{ts_csv.name}"
        csv_url = f"{media_url}/{csv_rel}"

        timeseries_file = None
        timeseries_path = None
        try:
            export_s1_timeseries(
                geom_geojson=field.geom,
                out_csv=str(ts_csv),
                tz="Asia/Phnom_Penh",
            )
            timeseries_file = csv_url
            timeseries_path = str(ts_csv)
            job.result = {
                **(job.result or {}),
                "timeseries_file": timeseries_file,
                "timeseries_path": timeseries_path,
            }
            job.save(update_fields=["result"])
        except Exception as e:
            print("⚠️ Time-series export failed:", e)

        # Optionally re-run analysis after TS write
        if settings.DEBUG:
            run_waterlogging_analysis(job.id)
        else:
            run_waterlogging_analysis.delay(job.id)

        return JsonResponse({
            "ok": True,
            "area_ha": round(float(area_ha), 4),
            "aoi_file": aoi_path.name,
            "tif_file": tif_exported,
            "field_id": field.id,
            "job_id": job.id,
            "timeseries_file": timeseries_file,
            "timeseries_path": timeseries_path,
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
    if (job is None) and force_rerun:
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
        # NEW:
        "probe_bin": job.result.get("probe_bin_url") or "",
        "probe_meta": job.result.get("probe_meta_url") or "",
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
    
    @action(detail=True, methods=["post"])
    def export_timeseries(self, request, pk=None):
        field = self.get_object()
        geom_geojson = field.geom

        start = request.data.get("start")
        end = request.data.get("end")
        step_days = int(request.data.get("step_days", 10))
        orbit = request.data.get("orbit")

        media_root = Path(getattr(settings, "MEDIA_ROOT", "media"))
        media_url  = getattr(settings, "MEDIA_URL", "/media/")
        ts_dir = media_root / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"timeseries_field_{field.id}_{stamp}.csv"
        csv_path = ts_dir / fname

        try:
            export_s1_timeseries(
                geom_geojson=geom_geojson,
                out_csv=str(csv_path),
                start=start,
                end=end,
                step_days=step_days,
                orbit_pass=orbit or None,
                tz="Asia/Phnom_Penh",
            )
        except Exception as e:
            return Response({"ok": False, "error": f"GEE export failed: {e}"}, status=400)

        csv_rel = f"timeseries/{fname}"
        csv_url = (media_url.rstrip("/") + "/" + csv_rel).replace("//", "/")
        return Response({"ok": True, "csv_file": fname, "csv_url": csv_url}, status=200)

def lands(request):
    # If you have a fields list page, redirect there instead.
    return render(request, "lands.html")


def dashboard_index(request):
    # 1) Always try the last viewed field (cookie) first
    last = request.COOKIES.get("last_field")
    if last:
        return redirect(f"/dashboard/{last}/")

    # 2) Fallbacks if no cookie yet
    latest_field = FieldAOI.objects.order_by("-id").first()
    if latest_field:
        return redirect(f"/dashboard/{latest_field.id}/")

    return redirect(reverse("lands"))

def dashboard(request, field_id: int):
    field = get_object_or_404(FieldAOI, id=field_id)
    job = (AnalysisJob.objects.filter(field=field).order_by("-id").first())

    # --- always remember last viewed field ---
    def remember(resp):
        resp.set_cookie("last_field", str(field_id), max_age=60*60*24*30, path="/")
        return resp

    if not job or job.status != "done" or not job.result:
        status_txt = (job and job.status) or "queued"
        msg = (job and (job.message or "")) or ""
        html = f"""
        <!doctype html><meta charset="utf-8">
        <title>Analyzing…</title>
        <meta http-equiv="refresh" content="3">
        <style>
        body {{ font-family: system-ui, Arial; margin: 24px; }}
        .pill {{ display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#eef7f3; }}
        .err {{ background:#fee2e2; color:#991b1b; padding:10px; border-radius:8px; }}
        a.btn {{ display:inline-block; margin-top:10px; padding:8px 12px; background:#0ea5e9; color:#fff; border-radius:8px; text-decoration:none; }}
        </style>
        <h1>Analysis pending…</h1>
        <p>Status: <span class="pill">{status_txt}</span></p>
        {"<div class='err'><b>Reason:</b> " + msg + "</div>" if status_txt == "failed" and msg else ""}
        {"<a class='btn' href='?rerun=1'>↻ Re-run analysis</a>" if job else ""}
        <p>This page will refresh automatically.</p>
        """
        return remember(HttpResponse(html))

    bounds = job.result.get("bounds") or [[...],[...]]
    # build insights HTML parts
    # Resolve insights CSV from the saved timeseries path (preferred) or fallbacks
    insights_csv = job.result.get("timeseries_path") \
        or job.result.get("insights_csv_path") \
        or job.result.get("insights_csv_url")   # last resort if you ever store a URL

    parts = build_insights_html(
        insights_csv=job.result.get("insights_csv_path"),  # ← use processed insights
        recs_csv=job.result.get("recs_csv_url"),
        area_by_class=job.result.get("area_by_class") or {},
        total_ha=job.result.get("total_ha"),
        plot_path=job.result.get("plot_path"),
    )

    ctx = {
    "job_id": job.id,
    "bounds": json.dumps(bounds),
    "overlay_png": job.result.get("overlay_png_url") or "",
    "hotspots_url": job.result.get("hotspots_url") or "",
    "probe_bin": job.result.get("probe_bin_url") or "",
    "probe_meta": job.result.get("probe_meta_url") or "",
    "field": field,  # <-- important for template
    }
    resp = render(request, "dashboard.html", ctx)
    return remember(resp)

# app_core/views.py → field_insights_api()
def field_insights_api(request, field_id: int):
    job = (AnalysisJob.objects
           .filter(field_id=field_id, status="done")
           .order_by("-id").first())
    if not job or not job.result:
        raise Http404("No completed analysis for this field yet.")

    # 1) Preferred: the stored full path
    ts_path = job.result.get("timeseries_path")

    # 2) Fallback: try MEDIA_ROOT + relative
    if not ts_path or not os.path.exists(ts_path):
        rel = (job.result.get("timeseries_file") or "").lstrip("/")
        if rel.startswith("media/"):
            rel = rel[6:]
        candidate = os.path.join(settings.MEDIA_ROOT, rel)
        if rel and os.path.exists(candidate):
            ts_path = candidate

    # 3) Fallback: pick the newest CSV that matches this field
    if (not ts_path) or (not os.path.exists(ts_path)):
        ts_dir = os.path.join(settings.MEDIA_ROOT, "timeseries")
        if os.path.isdir(ts_dir):
            import glob
            pattern = os.path.join(ts_dir, f"timeseries_field_{field_id}_*.csv")
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                ts_path = matches[0]

    if not ts_path or not os.path.exists(ts_path):
        # return empty UI instead of 404, so the page stays usable
        return JsonResponse({
            "plot_png": None,
            "insights_csv": None,
            "alerts_count": 0,
            "area_by_class": job.result.get("area_by_class") or {},
            "total_ha": job.result.get("total_ha") or 0.0,
            **build_insights_html(
                insights_csv=None,
                area_by_class=job.result.get("area_by_class") or {},
                total_ha=job.result.get("total_ha"),
                plot_path=None,
            )
        })

    # --- compute/update scale if missing ---
    risk_tif = job.result.get("stack_path")  # make sure your pipeline saved it
    if risk_tif and os.path.exists(risk_tif):
        abc, tot = classify_and_area(risk_tif)
        job.result = {**(job.result or {}), "area_by_class": abc, "total_ha": tot}
        job.save(update_fields=["result"])
    else:
        abc = job.result.get("area_by_class") or {}
        tot = job.result.get("total_ha") or 0.0

    # --- proceed with computation (alerts/insights) ---
    alerts_df, insights_df, plot_png, insights_csv = compute_temporal_engine_s1(
        ts_path,
        media_root=settings.MEDIA_ROOT,
    )

    job.result = {
        **(job.result or {}),
        "plot_path": plot_png,
        "plot_url": (settings.MEDIA_URL.rstrip("/") + "/plots/" + os.path.basename(plot_png)).replace("//","/") if plot_png else None,
        "insights_csv_path": insights_csv,
        "insights_csv_url": (settings.MEDIA_URL.rstrip("/") + "/insights/" + os.path.basename(insights_csv)).replace("//","/") if insights_csv else None,
        "area_by_class": abc,
        "total_ha": tot,
    }
    job.save(update_fields=["result"])

    html = build_insights_html(
        insights_csv=insights_csv,
        area_by_class=abc,
        total_ha=tot,
        plot_path=plot_png,
    )

    return JsonResponse({
        "plot_png": plot_png,
        "insights_csv": insights_csv,
        "alerts_count": int(getattr(alerts_df, "shape", [0, 0])[0]),
        "area_by_class": abc,
        "total_ha": tot,
        **html
    })