# app_core/models.py

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from uuid import uuid4
import os

def avatar_upload_to(instance, filename):
    base, ext = os.path.splitext(filename or "")
    ext = ext.lower() if ext else ".jpg"
    return f"avatars/user_{instance.user_id}/{timezone.now():%Y/%m}/{uuid4().hex}{ext}"

UNSPECIFIED = "unspecified"
PROVINCE_CHOICES = [
    ("Banteay Meanchey", "Banteay Meanchey"),
    ("Battambang", "Battambang"),
    ("Kampong Cham", "Kampong Cham"),
    ("Kampong Chhnang", "Kampong Chhnang"),
    ("Kampong Speu", "Kampong Speu"),
    ("Kampong Thom", "Kampong Thom"),
    ("Kampot", "Kampot"),
    ("Kandal", "Kandal"),
    ("Koh Kong", "Koh Kong"),
    ("Kratie", "Kratie"),
    ("Mondulkiri", "Mondulkiri"),
    ("Phnom Penh", "Phnom Penh"),
    ("Preah Vihear", "Preah Vihear"),
    ("Prey Veng", "Prey Veng"),
    ("Pursat", "Pursat"),
    ("Ratanakiri", "Ratanakiri"),
    ("Siem Reap", "Siem Reap"),
    ("Preah Sihanouk", "Preah Sihanouk"),
    ("Stung Treng", "Stung Treng"),
    ("Svay Rieng", "Svay Rieng"),
    ("Takeo", "Takeo"),
    ("Oddar Meanchey", "Oddar Meanchey"),
    ("Kep", "Kep"),
    ("Pailin", "Pailin"),
    ("Tbong Khmum", "Tbong Khmum"),
]

MAIN_CROP_CHOICES = [
    ("Sen Kra’ob", "Sen Kra’ob"),
    ("Phka Rumduol", "Phka Rumduol"),
    ("Other", "Other"),
]

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    full_name = models.CharField(max_length=120, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    main_crop = models.CharField(max_length=64, choices=MAIN_CROP_CHOICES,
                                 default=UNSPECIFIED, blank=True)
    province = models.CharField(max_length=64, choices=PROVINCE_CHOICES,
                                default=UNSPECIFIED, blank=True)
    # ✅ NEW: primary contact number
    phone = models.CharField(max_length=20, blank=True, help_text="Primary contact number")
    avatar = models.ImageField(upload_to="avatars/", blank=True, null=True)

    def __str__(self):
        return self.full_name or self.user.get_username()

class FieldAOI(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="fields", null=True, blank=True)
    name = models.CharField(max_length=255, blank=True)
    geom = models.JSONField()
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
    hotspots_geojson = models.FileField(upload_to="hotspots/", blank=True, null=True)

    def __str__(self):
        return f"Job {self.id} for Field {self.field_id} ({self.status})"

from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    else:
        if hasattr(instance, "profile"):
            instance.profile.save()
