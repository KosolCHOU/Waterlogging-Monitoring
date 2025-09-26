# app_core/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'fields', views.FieldViewSet, basename='fields')

urlpatterns = [
    path("", views.lands, name="home"),
    path("lands/", views.lands, name="lands"),
    path("aoi_upload/", views.aoi_upload, name="aoi_upload"),
    path("dashboard/", views.dashboard_index, name="dashboard_index"),
    path("dashboard/<int:field_id>/", views.dashboard, name="dashboard"),
    path("probe/<int:job_id>/", views.probe, name="probe"),
    # path("fields/<int:field_id>/timeseries/", views.export_s1_timeseries, name="export_timeseries"),
    path("fields/<int:field_id>/insights/", views.field_insights_api, name="field_insights_api"),
    path("api/", include(router.urls)),
    path("about/", views.about, name="about"),
    path('fields/<int:field_id>/analytics/', views.analytics, name='analytics'),
    path('analytics/', views.analytics, name='analytics_no_field'),
    path('fields/<int:field_id>/forecast.json', views.forecast_json, name='forecast_json'),
]
