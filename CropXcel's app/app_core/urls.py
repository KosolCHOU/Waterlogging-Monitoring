# app_core/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'fields', views.FieldViewSet, basename='fields')

urlpatterns = [
    path("", views.lands, name="home"),         # NEW root path
    path("lands/", views.lands, name="lands"),
    path("aoi_upload/", views.aoi_upload, name="aoi_upload"),
    path("dashboard/<int:field_id>/", views.dashboard, name="dashboard"),
    path("probe/<int:job_id>/", views.probe, name="probe"),
    path("api/", include(router.urls)),
]
