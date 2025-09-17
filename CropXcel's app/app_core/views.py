import json
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_http_methods

from analysis.run_analysis_from_notebook import run_analysis_from_notebook
from .models import FieldAOI, AnalysisJob

# Earth Engine
import ee
EE_PROJECT = "angelic-edition-466705-r8"

# DRF
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import FieldAOI, AnalysisJob
from .serializers import FieldSerializer, JobSerializer
from .tasks import run_waterlogging_analysis  # <-- needed by FieldViewSet

# Use your notebook logic
from analysis.run_analysis_from_notebook import run_analysis_from_notebook  # <-- IMPORTANT

# Initialize EE once
_initialized = False
def _ensure_ee():
    try:
        ee.data.getAssetRoots()
    except Exception:
        ee.Initialize(project=EE_PROJECT)

@action(detail=True, methods=["get"])
def probe(self, request, pk=None):
    """Return risk value at a point, and simple contribution breakdown.
       You can reuse the notebook’s logic to compute both.
    """
    try:
      lat = float(request.GET.get("lat"))
      lon = float(request.GET.get("lon"))
    except (TypeError, ValueError):
      return Response({"error":"lat/lon required"}, status=status.HTTP_400_BAD_REQUEST)

    field = self.get_object()
    _ensure_ee()

    # --- SAMPLE IMPLEMENTATION ---
    # Rebuild the same risk image you used to get tile_url in your notebook.
    # If your notebook exposes a helper, import and call it here.
    from analysis.run_analysis_from_notebook import build_risk_image_and_contrib

    try:
        img, contrib = build_risk_image_and_contrib(field.geom)  # returns ee.Image('risk'), {'weights':...}
        pt = ee.Geometry.Point([lon, lat])
        # sample at native scale used in the tile (e.g., 10m or 20m)
        scale = 10
        val = img.sample(pt, scale=scale).first().get('risk').getInfo()
        # contrib can be a dict: {'signal_var': 0.62, 'vh_vv': 0.25, 'radar_drop': 0.13}
        return Response({
            "value": float(val or 0),
            "contrib": contrib or {}
        })
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def home(request):
    return render(request, "home.html")

@require_http_methods(["POST"])
def aoi_upload(request):
    _ensure_ee()
    try:
        payload = json.loads(request.body.decode("utf-8"))
        feature = payload.get("feature")
        if not feature:
            return HttpResponseBadRequest("Missing 'feature'")

        geom_geojson = feature["geometry"] if feature.get("type") == "Feature" else feature
        ee_geom = ee.Geometry(geom_geojson)

        # compute area on server (m² → ha)
        area_m2 = ee_geom.area(maxError=1).getInfo()
        area_ha = area_m2 / 10000.0

        # write file
        media_dir = Path(settings.MEDIA_ROOT) / "aoi"
        media_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = media_dir / f"field_{ts}.geojson"
        out_path.write_text(json.dumps(feature, indent=2), encoding="utf-8")

        # (optional) kick off analysis here or leave it to /api/fields/
        return JsonResponse({
            "ok": True,
            "area_ha": area_ha,
            "aoi_file": out_path.name,
        })
    except Exception as e:
        return HttpResponseBadRequest(f"Invalid payload: {e}")

# ---------- DRF API (async flow, optional) ----------
class FieldViewSet(viewsets.ModelViewSet):
    queryset = FieldAOI.objects.all().order_by("-id")
    serializer_class = FieldSerializer

    @action(detail=True, methods=["get"])
    def latest_job(self, request, pk=None):
        job = AnalysisJob.objects.filter(field_id=pk).order_by("-created_at").first()
        from .serializers import JobSerializer
        return Response(JobSerializer(job).data if job else {}, status=200)

    def create(self, request, *args, **kwargs):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)
        field = ser.save()

        # DEV: run sync so we get a result immediately
        from .models import AnalysisJob
        from analysis.run_analysis_from_notebook import run_analysis_from_notebook

        job = AnalysisJob.objects.create(field=field, status="running", message="Sync analysis…")
        try:
            result = run_analysis_from_notebook(aoi_geojson=field.geom)
            # compute bounds like your task does
            gj = field.geom
            coords = gj["coordinates"][0] if gj["type"] == "Polygon" else gj["coordinates"][0][0]
            lats = [pt[1] for pt in coords]; lons = [pt[0] for pt in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
            result["bounds"] = bounds

            job.result = result
            job.status = "ready"
            job.message = "Finished (sync)"
            job.save()
        except Exception as e:
            job.status = "failed"
            job.message = str(e)
            job.save()

        return Response({"field_id": field.id, "job_id": job.id}, status=status.HTTP_201_CREATED)
    
def _bounds_from_geom(geom):
    # Works for Polygon or MultiPolygon GeoJSON
    t = geom.get("type")
    coords = geom.get("coordinates", [])
    pts = []

    if t == "Polygon":
        pts = coords[0]
    elif t == "MultiPolygon":
        # flatten first polygon’s outer ring (simple + safe)
        if coords and coords[0] and coords[0][0]:
            pts = coords[0][0]
    else:
        return [[0,0],[0,0]]

    lats = [p[1] for p in pts]
    lons = [p[0] for p in pts]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]

def risk_map(request, pk):
    field = get_object_or_404(FieldAOI, pk=pk)
    job = AnalysisJob.objects.filter(field=field).order_by('-created_at').first()

    # If no job or still running → wait page
    if not job or job.status in ("queued", "running"):
        return render(request, "risk_wait.html", {"field": field, "job": job})

    # Be robust: result might be a JSON string or None
    res = job.result or {}
    if isinstance(res, str):
        try:
            import json
            res = json.loads(res)
        except Exception:
            res = {}

    # Bounds: from result if present, else from AOI
    bounds = res.get("bounds") or _bounds_from_geom(field.geom)

    # Sources for overlay/hotspots
    tile_url = res.get("tile_url", "")
    overlay_png_url = res.get("overlay_png_url", "")
    hotspots_url = ""
    if job.hotspots_geojson:
        hotspots_url = job.hotspots_geojson.url
    else:
        hotspots_url = res.get("hotspots_url", "")

    # If we truly have nothing to render (no tile, no png, no hotspots), keep waiting
    if not tile_url and not overlay_png_url and not hotspots_url:
        return render(request, "risk_wait.html", {"field": field, "job": job})

    return render(request, "risk_map.html", {
        "field": field,
        "bounds": bounds,
        "tile_url": tile_url,
        "overlay_png": overlay_png_url,
        "hotspots_url": hotspots_url,
    })
