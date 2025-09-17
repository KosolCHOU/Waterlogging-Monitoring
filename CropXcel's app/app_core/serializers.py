from rest_framework import serializers
from .models import FieldAOI, AnalysisJob

class FieldSerializer(serializers.ModelSerializer):
    class Meta:
        model = FieldAOI
        fields = ("id", "name", "geom", "area_ha", "created_at")

class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ("id", "status", "message", "overlay_html", "result", "created_at", "finished_at")