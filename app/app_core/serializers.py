# app_core/serializers.py
from rest_framework import serializers
from .models import FieldAOI, AnalysisJob

from math import fabs
try:
    from shapely.geometry import shape
except Exception:
    shape = None
try:
    from pyproj import Geod
    _GEOD = Geod(ellps="WGS84")
except Exception:
    _GEOD = None


class FieldSerializer(serializers.ModelSerializer):
    area_ha = serializers.SerializerMethodField()

    class Meta:
        model = FieldAOI
        fields = ("id", "name", "geom", "area_ha", "created_at")

    def validate(self, attrs):
        geom = attrs.get("geom") or getattr(self.instance, "geom", None)
        if not geom:
            raise serializers.ValidationError({"geom": "Geometry is required."})
        gtype = (geom.get("type") or "").lower()
        if "polygon" not in gtype:
            raise serializers.ValidationError({"geom": "Geometry must be Polygon or MultiPolygon."})
        return attrs

    def create(self, validated_data):
        # Auto-name if missing
        name = (validated_data.get("name") or "").strip()
        if not name:
            # use count+1 so it’s stable even if some rows are deleted later
            n = FieldAOI.objects.count() + 1
            validated_data["name"] = f"Field {n}"
        return super().create(validated_data)

    def get_area_ha(self, obj):
        # If your model already stores area_ha, prefer it
        val = getattr(obj, "area_ha", None)
        if isinstance(val, (int, float)) and val > 0:
            return round(float(val), 2)

        # Otherwise compute from geom (WGS84) → geodesic area
        try:
            gj = obj.geom  # GeoJSON dict
            if not gj:
                return None

            # Approach A: pyproj.Geod (no shapely needed)
            if _GEOD:
                # handle Polygon & MultiPolygon as rings
                def ring_area(coords):
                    lons, lats = zip(*coords)
                    area, _ = _GEOD.polygon_area_perimeter(lons, lats)
                    return fabs(area)  # may be negative by orientation

                gtype = gj.get("type")
                area_m2 = 0.0
                if gtype == "Polygon":
                    # first ring is exterior, holes subtract automatically by sign
                    for ring in gj["coordinates"]:
                        area_m2 += ring_area(ring)
                elif gtype == "MultiPolygon":
                    for poly in gj["coordinates"]:
                        for ring in poly:
                            area_m2 += ring_area(ring)
                else:
                    return None
                return round(area_m2 / 10000.0, 2)

            # Approach B: shapely fallback (if available)
            if shape:
                geom = shape(gj)
                # rough geodesic: use geod on exterior + interiors
                if geom.geom_type == "Polygon":
                    area_m2 = fabs(_GEOD.geometry_area_perimeter(geom)[0]) if _GEOD else geom.area
                else:
                    area_m2 = fabs(_GEOD.geometry_area_perimeter(geom)[0]) if _GEOD else geom.area
                return round(area_m2 / 10000.0, 2)
        except Exception:
            return None

        return None


class JobSerializer(serializers.ModelSerializer):
    overlay_png_url = serializers.SerializerMethodField()
    hotspots_url = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisJob
        fields = (
            "id", "status", "message", "result", "created_at", "finished_at",
            "overlay_png_url", "hotspots_url"
        )

    def get_overlay_png_url(self, obj):
        return obj.overlay_png.url if getattr(obj, "overlay_png", None) else None

    def get_hotspots_url(self, obj):
        return obj.hotspots_geojson.url if getattr(obj, "hotspots_geojson", None) else None