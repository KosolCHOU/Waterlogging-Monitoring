from rest_framework import serializers
from .models import FieldAOI, AnalysisJob

class FieldSerializer(serializers.ModelSerializer):
    class Meta:
        model = FieldAOI
        fields = ("id", "name", "geom", "area_ha", "created_at")

class JobSerializer(serializers.ModelSerializer):
    overlay_png_url = serializers.SerializerMethodField()
    hotspots_url = serializers.SerializerMethodField()
    class Meta:
        model = AnalysisJob
        fields = ("id","status","message","result","created_at","finished_at",
                  "overlay_png_url","hotspots_url")

    def get_overlay_png_url(self, obj):
        return obj.overlay_png.url if obj.overlay_png else None
    def get_hotspots_url(self, obj):
        return obj.hotspots_geojson.url if obj.hotspots_geojson else None