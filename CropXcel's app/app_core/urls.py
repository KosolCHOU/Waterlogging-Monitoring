# app_core/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'fields', views.FieldViewSet, basename='fields')

urlpatterns = [
    path("", views.home, name="home"),
    path("aoi_upload/", views.aoi_upload, name="aoi_upload"),  # ← add this
    path("fields/<int:field_id>/risk/", views.risk_map, name="risk_map"),
    path("probe/<int:job_id>/", views.probe, name="probe"),
    path("api/", include(router.urls)),
]
