from django.db import models

# Create your models here.
from django.db import models
class AOI(models.Model):
    name = models.CharField(max_length=120)
    geojson = models.FileField(upload_to="aoi/")
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return self.name

class FieldAOI(models.Model):
    name = models.CharField(max_length=255, blank=True)
    geom = models.JSONField()   # stores the AOI geometry as GeoJSON
    area_ha = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or f"Field {self.id}"


class AnalysisJob(models.Model):
    field = models.ForeignKey(FieldAOI, on_delete=models.CASCADE, related_name="jobs")
    status = models.CharField(max_length=20, default="queued")
    message = models.TextField(blank=True)
    result = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    overlay_html = models.FileField(upload_to="overlays/", blank=True, null=True)
    overlay_png  = models.ImageField(upload_to="overlays/", blank=True, null=True)
    hotspots_geojson = models.FileField(upload_to="hotspots/", blank=True, null=True)  # NEW

    def __str__(self):
        return f"Job {self.id} for Field {self.field_id} ({self.status})"