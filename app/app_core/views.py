# app_core/views.py
import os
import json
import math
from pathlib import Path
from datetime import datetime, date

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .serializers import FieldSerializer, JobSerializer
from .tasks import run_waterlogging_analysis

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest, Http404, HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import CropRecommendForm, ProfileImageForm, ProfileForm
from django.contrib.auth import logout, login
from django.utils import timezone
from django.views import View
from .forms import CropRecForm
from .models import FieldAOI, AnalysisJob, Profile
from app_core.ml.recommender import predict_crop

from .models import FieldAOI, AnalysisJob, Profile
# from analysis.engine import export_stack_from_geom, export_s1_timeseries
# from analysis.insights import compute_temporal_engine_s1, build_insights_html, classify_and_area
# from analysis.weather import get_forecast_for_field
from .forms import SignupForm

# New: local geodesic area (no GEE)
from shapely.geometry import shape as shp_shape  

# app_core/views.py
@require_http_methods(["POST"])
def aoi_upload(request):
    """
    Save a drawn AOI, create Field + Job, export stack/time-series,
    then kick off the analysis.

    Change: When user leaves `name` empty, assign a per-user sequence:
            "Field #<count for this user>" instead of global id.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
        feature = payload.get("feature")
        user_name = (payload.get("name") or "").strip()
        if not feature:
            return HttpResponseBadRequest("Missing 'feature'")

        geom_geojson = feature["geometry"] if feature.get("type") == "Feature" else feature
        
        # Lazy import
        from analysis.engine import export_stack_from_geom, export_s1_timeseries

        # --- area calc ---
        from shapely.geometry import shape
        from pyproj import Geod
        g = shape(geom_geojson)
        if g.is_empty:
            return HttpResponseBadRequest("Empty geometry")
        geod = Geod(ellps="WGS84")
        area_m2, _ = geod.geometry_area_perimeter(g)
        area_ha = abs(area_m2) / 10_000.0

        # --- Save AOI file ---
        media_dir = Path(settings.MEDIA_ROOT) / "aoi"
        media_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        aoi_path = media_dir / f"field_{ts}.geojson"
        aoi_path.write_text(json.dumps(
            {"type": "Feature", "geometry": geom_geojson, "properties": {}},
            indent=2
        ), encoding="utf-8")

        # --- Owner (must be logged in for per-user numbering) ---
        owner = request.user if request.user.is_authenticated else None

        # --- Decide display name ---
        # If user typed a name, use it. Otherwise generate a per-user sequence.
        auto_name = None
        if not user_name:
            if owner:
                # Count this user's existing fields to get the next sequence
                # e.g., if they already have 2, this becomes "Field #3"
                next_seq = FieldAOI.objects.filter(owner=owner).count() + 1
                base = f"Field #{next_seq}"

                # Ensure uniqueness in case of race/rename: Field #3, Field #3 (2), ...
                candidate = base
                bump = 2
                while FieldAOI.objects.filter(owner=owner, name=candidate).exists():
                    candidate = f"{base} ({bump})"
                    bump += 1
                auto_name = candidate
            else:
                # Anonymous: fallback later to global id (old behavior)
                auto_name = None

        # --- Create field ---
        field = FieldAOI.objects.create(
            owner=owner,
            name=(user_name or auto_name or ""),  # may be blank for anonymous case
            geom=geom_geojson,
            area_ha=area_ha
        )

        # Final fallback for anonymous users with blank name → use global id
        if not (user_name or auto_name):
            field.name = f"Field #{field.id}"
            field.save(update_fields=["name"])

        # --- Export stack (same as before) ---
        stacks_dir = Path(settings.MEDIA_ROOT) / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        tif_path = stacks_dir / f"stack_field_{field.id}_{ts}.tif"
        tif_exported = None
        try:
            export_stack_from_geom(geom_geojson, str(tif_path))
            if tif_path.exists():
                tif_exported = tif_path.name
        except Exception as gee_err:
            print("⚠️ GEE export failed:", gee_err)

        # --- Create Job and store stack_path ---
        job = AnalysisJob.objects.create(field=field, status="queued", message="Created from AOI upload")
        job.result = {**(job.result or {}), "stack_path": (str(tif_path) if tif_exported else None)}
        job.save(update_fields=["result"])

        # --- Export time-series (same as before) ---
        ts_dir = Path(settings.MEDIA_ROOT) / "timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ts_csv = ts_dir / f"timeseries_field_{field.id}_{stamp}.csv"
        export_s1_timeseries(geom_geojson=field.geom, out_csv=str(ts_csv), tz="Asia/Phnom_Penh")

        timeseries_file = (settings.MEDIA_URL.rstrip("/") + f"/timeseries/{ts_csv.name}")
        timeseries_path = str(ts_csv)
        job.result = {
            **(job.result or {}),
            "timeseries_file": timeseries_file,
            "timeseries_path": timeseries_path,
        }
        job.save(update_fields=["result"])

        # --- Kick off analysis (unchanged) ---
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
            "name": field.name,  # return the resolved name
            "job_id": job.id,
            "timeseries_file": timeseries_file,
            "timeseries_path": timeseries_path,
            "next_url": f"/fields/{field.id}/risk/",
        })

    except Exception as e:
        return HttpResponseBadRequest(f"Invalid payload: {e}")

# ---------- Page: risk map ----------
@login_required
def risk_map(request, field_id: int):
    field = get_object_or_404(FieldAOI, id=field_id, owner=request.user)

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
    User-scoped Field API.
    GET /api/fields/         → only my fields (newest first)
    POST /api/fields/        → owner auto-set to request.user
    """
    serializer_class = FieldSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        # Default: always show only my fields
        qs = FieldAOI.objects.filter(owner=user).order_by("-id")

        # Staff can explicitly request all fields for debugging/ops
        if user.is_staff and self.request.query_params.get("all") == "1":
            qs = FieldAOI.objects.all().order_by("-id")

        return qs

    def perform_create(self, serializer):
        # Ensure owner is always the current user
        serializer.save(owner=self.request.user)

    @action(detail=True, methods=["get"])
    def latest_job(self, request, pk=None):
        field = self.get_object()  # scoped by get_queryset(), so already safe
        job = (AnalysisJob.objects
               .filter(field=field, status__in=["done", "running", "queued", "failed"])
               .order_by("-id")
               .first())
        if not job:
            return Response({"detail": "No jobs yet."}, status=status.HTTP_404_NOT_FOUND)
        return Response(JobSerializer(job).data, status=200)

    @action(detail=True, methods=["post"])
    def analyze(self, request, pk=None):
        field = self.get_object()
        job = AnalysisJob.objects.create(field=field, status="queued", message="Queued by API")
        run_waterlogging_analysis.delay(job.id)
        return Response({"ok": True, "job_id": job.id}, status=202)

    @action(detail=True, methods=["post"])
    def export_timeseries(self, request, pk=None):
        field = self.get_object()
        # ... keep your existing body unchanged ...
        # (no change needed below this line)
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
        
        # Lazy import
        from analysis.engine import export_s1_timeseries

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

@login_required
def lands(request):
    # If you have a fields list page, redirect there instead.
    return render(request, "lands.html")

from django.contrib.auth.decorators import login_required

@login_required
def dashboard_index(request):
    """
    Redirect to a sensible dashboard start:
      1) If cookie 'last_field' refers to *my* field → go there
      2) Else if I have any fields → go to my newest one
      3) Else → go to Lands (empty state)
    """
    # 1) cookie (validate ownership)
    last = request.COOKIES.get("last_field")
    if last and last.isdigit():
        try:
            FieldAOI.objects.get(id=int(last), owner=request.user)
            return redirect(f"/dashboard/{int(last)}/")
        except FieldAOI.DoesNotExist:
            pass  # ignore stale/foreign cookie

    # 2) my newest field
    mine_latest = FieldAOI.objects.filter(owner=request.user).order_by("-id").first()
    if mine_latest:
        return redirect(f"/dashboard/{mine_latest.id}/")

    # 3) no fields yet
    return redirect("lands")


@login_required
def dashboard(request, field_id: int):
    field = get_object_or_404(FieldAOI, id=field_id, owner=request.user)  # ✅ ownership check
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
    
    # Lazy import
    from analysis.insights import build_insights_html
    
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

    # Build context for stats bar
    latest_risk = "Unknown"
    problem_areas_count = 0
    last_updated = "Never"
    
    # Calculate latest risk from insights data
    insights_csv = job.result.get("timeseries_path") or job.result.get("insights_csv_path")
    if insights_csv and os.path.exists(insights_csv):
        try:
            import pandas as pd
            df = pd.read_csv(insights_csv)
            if not df.empty and 'level' in df.columns:
                latest_risk = df.iloc[-1]['level'] if 'level' in df.columns else latest_risk
        except Exception:
            pass
    
    # Get problem areas count from job result (if available) or count from GeoJSON file
    problem_areas_count = 0
    
    # First try to get from job result
    if job.result and "hotspots_count" in job.result:
        problem_areas_count = job.result.get("hotspots_count", 0)
    
    # If not available, try to read from GeoJSON file
    if problem_areas_count == 0:
        hotspots_url = job.result.get("hotspots_url") if job.result else None
        if hotspots_url:
            try:
                # Handle both /media/hotspots/file.geojson and media/hotspots/file.geojson
                hotspots_path = hotspots_url.lstrip("/")
                full_hotspots_path = os.path.join(settings.MEDIA_ROOT, hotspots_path)
                
                if os.path.exists(full_hotspots_path):
                    with open(full_hotspots_path, 'r') as f:
                        hotspots_data = json.load(f)
                        if 'features' in hotspots_data:
                            problem_areas_count = len(hotspots_data['features'])
                else:
                    # Try alternative path construction
                    alt_path = os.path.join(settings.MEDIA_ROOT, hotspots_url.lstrip("/media/"))
                    if os.path.exists(alt_path):
                        with open(alt_path, 'r') as f:
                            hotspots_data = json.load(f)
                            if 'features' in hotspots_data:
                                problem_areas_count = len(hotspots_data['features'])
            except Exception as e:
                # Try to find the most recent hotspots file as fallback
                try:
                    hotspots_dir = os.path.join(settings.MEDIA_ROOT, "hotspots")
                    if os.path.exists(hotspots_dir):
                        hotspot_files = [f for f in os.listdir(hotspots_dir) if f.startswith("hotspots_") and f.endswith(".geojson")]
                        if hotspot_files:
                            # Get the most recent file
                            latest_file = max(hotspot_files, key=lambda f: os.path.getmtime(os.path.join(hotspots_dir, f)))
                            with open(os.path.join(hotspots_dir, latest_file), 'r') as f:
                                hotspots_data = json.load(f)
                                if 'features' in hotspots_data:
                                    problem_areas_count = len(hotspots_data['features'])
                except Exception:
                    pass
    
    # Get last updated time
    if job.finished_at:
        from django.utils.timesince import timesince
        last_updated = timesince(job.finished_at) + " ago"

    # Get satellite capture date from metadata
    satellite_capture_date = None
    satellite_event_count = None
    if job.result and job.result.get("stack_path"):
        stack_path = job.result.get("stack_path")
        meta_path = str(stack_path) + ".meta.json"
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                    # Prefer locally-adjusted acquisition date, fallback to UTC fields
                    satellite_capture_date = (
                        meta_data.get("acq_local")
                        or meta_data.get("event_end")
                        or meta_data.get("acq_utc")
                        or meta_data.get("event_end_utc")
                    )
                    satellite_event_count = meta_data.get("event_count")

                    # Format the date nicely if available
                    if satellite_capture_date:
                        from datetime import datetime
                        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
                            try:
                                capture_dt = datetime.strptime(satellite_capture_date, fmt)
                                satellite_capture_date = capture_dt.strftime("%b %d, %Y")
                                break
                            except ValueError:
                                continue
        except Exception as e:
            print(f"Could not read satellite metadata: {e}")

    ctx = {
    "job_id": job.id,
    "bounds": json.dumps(bounds),
    "overlay_png": job.result.get("overlay_png_url") or "",
    "hotspots_url": job.result.get("hotspots_url") or "",
    "probe_bin": job.result.get("probe_bin_url") or "",
    "probe_meta": job.result.get("probe_meta_url") or "",
    "field": field,  # <-- important for template
    # Stats bar data
    "latest_risk": latest_risk,
    "alert_zones_count": problem_areas_count,
    "last_updated": last_updated,
    "satellite_capture_date": satellite_capture_date,
    "satellite_event_count": satellite_event_count,
    }
    resp = render(request, "dashboard.html", ctx)
    return remember(resp)

# app_core/views.py → field_insights_api()
@require_http_methods(["GET"])
@login_required
def field_insights_api(request, field_id: int):
    # enforce that the insights are for *my* field
    get_object_or_404(FieldAOI, id=field_id, owner=request.user)
    job = (AnalysisJob.objects
           .filter(field_id=field_id, status="done")
           .order_by("-id").first())
    if not job or not job.result:
        raise Http404("No completed analysis for this field yet.")

    # 1) Preferred: the stored full path
    
    # Lazy imports
    from analysis.insights import compute_temporal_engine_s1, build_insights_html, classify_and_area

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
    # --- inside field_insights_api, before classify_and_area() ---
    risk_tif = job.result.get("risk_tif_path")

    # Fallback 1: derive a local path from URL if present
    if (not risk_tif or not os.path.exists(risk_tif)):
        url = (job.result.get("risk_tif_url") or "").lstrip("/")
        if url.startswith("media/"):  # convert /media/... → MEDIA_ROOT/...
            candidate = os.path.join(settings.MEDIA_ROOT, url.split("media/",1)[-1])
            if os.path.exists(candidate):
                risk_tif = candidate

    # Fallback 2: pick newest risk_*.tif under MEDIA_ROOT/overlays
    if (not risk_tif) or (not os.path.exists(risk_tif)):
        import glob
        ov_dir = os.path.join(settings.MEDIA_ROOT, "overlays")
        matches = sorted(glob.glob(os.path.join(ov_dir, f"risk_field_{field_id}_*.tif")), reverse=True)
        if matches:
            risk_tif = matches[0]

    # Proceed only if we now have a valid path
    if risk_tif and os.path.exists(risk_tif):
        # compute geodesic px area from the GeoTIFF footprint (EPSG:4326 expected)
        import rasterio
        from pyproj import Geod
        with rasterio.open(risk_tif) as ds:
            rows, cols = ds.height, ds.width
            left, bottom, right, top = ds.bounds
        g = Geod(ellps="WGS84")
        area_m2, _ = g.polygon_area_perimeter([left,right,right,left,left],
                                            [bottom,bottom,top,top,bottom])[:2]
        px_area_m2 = abs(area_m2) / float(rows * cols)

        abc, tot = classify_and_area(
            risk_tif,
            thresholds=(0.25, 0.40, 0.5),   # Better align with 8-color visual mapping
            scale_from=None,                 # risk tif is already 0–1
            default_pixel_area_m2=px_area_m2
        )
        job.result = {**(job.result or {}), "area_by_class": abc, "total_ha": tot,
                    "risk_tif_path": risk_tif}
        job.save(update_fields=["result"])
    else:
        abc = job.result.get("area_by_class") or {}
        tot = job.result.get("total_ha") or 0.0

    # --- proceed with computation (alerts/insights) ---
    alerts_df, insights_df, plot_png, insights_csv = compute_temporal_engine_s1(
        ts_path,
        media_root=settings.MEDIA_ROOT,
    )

    risk_tif_path_existing = job.result.get("risk_tif_path")

    job.result = {
        **(job.result or {}),
        "plot_path": plot_png,
        "plot_url": (settings.MEDIA_URL.rstrip("/") + "/plots/" + os.path.basename(plot_png)).replace("//","/") if plot_png else None,
        "insights_csv_path": insights_csv,
        "insights_csv_url": (settings.MEDIA_URL.rstrip("/") + "/insights/" + os.path.basename(insights_csv)).replace("//","/") if insights_csv else None,
        "area_by_class": abc,
        "total_ha": tot,

        # WRITE BACK THE RESOLVED risk_tif (if valid), not the old one
        "risk_tif_path": (risk_tif if risk_tif and os.path.exists(risk_tif) else job.result.get("risk_tif_path")),
        "risk_tif_url":  ((settings.MEDIA_URL.rstrip("/") + "/overlays/" + os.path.basename(risk_tif)).replace("//","/")
                        if risk_tif and os.path.exists(risk_tif) else job.result.get("risk_tif_url")),
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

def about(request):
    return render(request, "about.html")

@login_required
def analytics(request, field_id: int | None = None):
    """
    Analytics also must be per-user.
    """
    if field_id is None:
        # Try cookie but validate ownership
        last = request.COOKIES.get("last_field")
        if last and last.isdigit():
            try:
                FieldAOI.objects.get(pk=int(last), owner=request.user)
                return redirect("analytics", field_id=int(last))
            except FieldAOI.DoesNotExist:
                pass
        # Fallback: my latest field (not global)
        latest_field = FieldAOI.objects.filter(owner=request.user).order_by("-id").first()
        if latest_field:
            return redirect("analytics", field_id=latest_field.id)
        return redirect("lands")   # no fields


    # 2) Normal analytics rendering (unchanged logic you already have)
    field = get_object_or_404(FieldAOI, id=field_id, owner=request.user)

    # robust bounds (your existing version)
    try:
        if isinstance(field.geom, dict):
            g = shp_shape(field.geom)
            minx, miny, maxx, maxy = g.bounds
        else:
            minx, miny, maxx, maxy = field.geom.extent
        bounds = [[miny, minx], [maxy, maxx]]
    except Exception:
        if isinstance(field.geom, dict):
            g = shp_shape(field.geom); c = g.centroid
            lat, lon = float(c.y), float(c.x)
        else:
            c = field.geom.centroid; lat, lon = float(c.y), float(c.x)
        eps = 1e-3
        bounds = [[lat - eps, lon - eps], [lat + eps, lon + eps]]

    field_label = (field.name or "").strip() or f"Field #{field.id}"
    ctx = {
        "field": field,
        "field_label": field_label,
        "bounds": bounds,
        "job_id": request.GET.get("job_id") or "null",
    }

    resp = render(request, "analytics.html", ctx)

    # 3) ✅ Remember this field (same cookie name used by Dashboard)
    #    Your dashboard() already sets: resp.set_cookie("last_field", ...)
    resp.set_cookie("last_field", str(field_id), max_age=60*60*24*30, path="/", samesite="Lax")

    return resp

@require_http_methods(["GET"])
def forecast_json(request, field_id: int):
    """
    Return 7-day + 72h summary forecast for a field (Open-Meteo).
    """
    field = get_object_or_404(FieldAOI, id=field_id)
    field = get_object_or_404(FieldAOI, id=field_id)
    try:
        from analysis.weather import get_forecast_for_field
        data = get_forecast_for_field(field)
        # Shape response the way your frontend likes:
        return JsonResponse({"ok": True, **data}, status=200)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

def _age_from_dob(dob):
    if not dob:
        return None
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return max(age, 0)


@login_required
def profile(request):
    user_obj = request.user
    # always have a profile
    profile_obj, _ = Profile.objects.get_or_create(user=user_obj)
    age_years = _age_from_dob(profile_obj.date_of_birth)

    # fields summary
    my_fields = FieldAOI.objects.filter(owner=user_obj)
    fields_count = my_fields.count()
    fields_area = round(sum((f.area_ha or 0.0) for f in my_fields), 2)

    # initials fallback for avatar
    display_name = (user_obj.get_full_name() or user_obj.get_username() or "").strip()
    initials = "".join([p[0] for p in display_name.split() if p][:2]).upper() \
               or (user_obj.username[:2].upper() if user_obj.username else "U")

    if request.method == "POST":
        # remove avatar (if you later add a button named="remove_avatar")
        if "remove_avatar" in request.POST:
            if profile_obj.avatar:
                profile_obj.avatar.delete(save=False)
                profile_obj.avatar = None
                profile_obj.save(update_fields=["avatar"])
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse({"ok": True, "avatar_url": None})
            messages.success(request, "Profile picture removed.")
            return redirect("profile")

        # upload/change avatar
        form = ProfileImageForm(request.POST, request.FILES, instance=profile_obj)
        if form.is_valid():
            saved = form.save()
            new_url = saved.avatar.url + f"?v={int(timezone.now().timestamp())}"
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse({"ok": True, "avatar_url": new_url})
            messages.success(request, "Profile picture updated.")
            return redirect("profile")
        else:
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                # flatten Django error dict into a single message
                err_list = []
                for _, errs in form.errors.items():
                    err_list.extend(errs)
                return JsonResponse({"ok": False, "error": "; ".join(err_list)}, status=400)
            messages.error(request, "Upload failed. Please fix the errors below.")
            # fall-through to re-render page with errors below
    else:
        form = ProfileImageForm(instance=profile_obj)

    ctx = {
        "user_obj": user_obj,
        "profile_obj": profile_obj,
        "age_years": age_years,
        "initials": initials,
        "member_since": user_obj.date_joined,
        "last_login": user_obj.last_login or user_obj.date_joined,
        "now": timezone.now(),
        "form": form,
        "fields_count": fields_count,
        "fields_area": f"{fields_area:.2f}",
    }
    return render(request, "profile.html", ctx)

class LogoutViewAllowGet(View):
    template_name = "registration/logout.html"  # you said logout.html is inside registration/

    # Handle both GET and POST (and HEAD just in case)
    def get(self, request, *args, **kwargs):
        return self._do_logout(request)

    def post(self, request, *args, **kwargs):
        return self._do_logout(request)

    def _do_logout(self, request):
        logout(request)
        # If you ever want to support ?next=... later, you could redirect here instead.
        return render(request, self.template_name)

def signup(request):
    """
    Signup with optional profile fields. Only username/password are required.
    """
    next_url = request.GET.get("next") or request.POST.get("next") or "profile"
    # Allow query params such as ?start=1 or ?step=form to bypass the intro screen.
    requested_direct_form = request.GET.get("start") in {"1", "true", "yes"} or request.GET.get("step") == "form"
    show_intro = not requested_direct_form

    if request.method == "POST":
        form = SignupForm(request.POST)
        # Once the user posts the form we always show the form panel.
        show_intro = False
        if form.is_valid():
            user = form.save(commit=True)  # creates user AND writes profile via form.save()

            # --- DEFENSIVE: also persist optional fields here ---
            cd = form.cleaned_data
            prof = getattr(user, "profile", None) or Profile.objects.create(user=user)
            prof.full_name     = (cd.get("full_name") or "").strip()
            prof.phone         = (cd.get("phone") or "").strip()
            prof.date_of_birth = cd.get("date_of_birth") or None
            if cd.get("main_crop"):
                prof.main_crop = cd["main_crop"]
            if cd.get("province"):
                prof.province = cd["province"]
            prof.save()
            # ----------------------------------------------------

            login(request, user)
            messages.success(request, "🌱 Welcome! Your account is ready.")
            return redirect(next_url)
        else:
            # collect all field + non-field errors
            error_list = []
            for field, errs in form.errors.items():
                for err in errs:
                    if field == "__all__":
                        error_list.append(err)  # e.g. password mismatch
                    else:
                        error_list.append(f"{field.capitalize()}: {err}")

            if error_list:
                # show them all in one friendly message
                messages.error(request, "⚠️ " + " | ".join(error_list))
    else:
        form = SignupForm()

    if form.errors:
        show_intro = False

    ctx = {
        "form": form,
        "show_intro": show_intro,
        "next_url": next_url,
    }
    return render(request, "registration/signup.html", ctx)

@login_required
def edit_profile(request):
    prof, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ProfileForm(request.POST, instance=prof)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated.")
            return redirect("profile")
    else:
        form = ProfileForm(instance=prof)
    return render(request, "edit_profile.html", {"form": form})

def support(request):
    # super simple support page; replace with your real flow later
    return render(request, "support.html", {})

@login_required
def crop_recommend_simple(request):
    """
    Field-independent crop recommendation.
    Inputs: N, P, K, temperature, humidity, pH, rainfall.
    Output: recommended crop (+ optional probabilities).
    """
    result_label, prob_map = None, {}
    if request.method == "POST":
        form = CropRecForm(request.POST)
        if form.is_valid():
            feats = form.cleaned_features()
            result_label, prob_map = predict_crop(feats)
            messages.success(request, f"✅ Recommended crop: {result_label}")
            # Show results below the form
        else:
            messages.error(request, "Please fix the highlighted fields.")
    else:
        form = CropRecForm()

    return render(request, "crop_recommend_simple.html", {
        "form": form,
        "result_label": result_label,
        "probs": prob_map,
    })

from django.contrib import messages
from django.contrib.auth.decorators import login_required

@login_required
def crop_recommend_simple(request):
    """
    Field-independent crop recommendation.
    Inputs: N, P, K, temperature, humidity, pH, rainfall.
    Output: best result + table of top 3.
    """
    result_label = None
    top3 = []   # list of dicts: [{"crop": str, "prob": float, "pct": int}, ...]
    if request.method == "POST":
        form = CropRecForm(request.POST)
        if form.is_valid():
            feats = form.cleaned_features()
            # Expect predict_crop to return: (best_label, probs_dict)
            # probs_dict example: {"SenPidor-1":0.61, "IR66":0.23, "PhkaRumduol":0.16}
            result_label, probs = predict_crop(feats)

            # ---- compute top 3 (with defensive checks) ----
            if isinstance(probs, dict) and probs:
                # sort by prob desc
                ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                # normalize if not already (avoid sum=0)
                s = sum(v for _, v in ranked) or 1.0
                top3 = [
                    {
                        "crop": k,
                        "prob": float(v),
                        "pct": int(round((v / s) * 100)) if s != 0 else 0,
                    }
                    for k, v in ranked
                ]
                # Recompute best from ranked to be safe
                result_label = top3[0]["crop"] if top3 else result_label

            messages.success(request, f"✅ Top recommendation: {result_label}")
        else:
            messages.error(request, "Please fix the highlighted fields.")
    else:
        form = CropRecForm()

    return render(request, "crop_recommend_simple.html", {
        "form": form,
        "result_label": result_label,
        "top3": top3,
    })
