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
    
# app_core/models.py
import os
from uuid import uuid4
from django.utils import timezone
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

def avatar_upload_to(instance, filename):
    base, ext = os.path.splitext(filename or "")
    ext = ext.lower() if ext else ".jpg"
    return f"avatars/user_{instance.user_id}/{timezone.now():%Y/%m}/{uuid4().hex}{ext}"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    avatar = models.ImageField(upload_to=avatar_upload_to, blank=True, null=True)

    def __str__(self):
        return f"Profile({self.user.username})"

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        if hasattr(instance, "profile"):
            instance.profile.save()
