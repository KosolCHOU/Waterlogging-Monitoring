from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'fields', views.FieldViewSet, basename='fields')

urlpatterns = [
    path("", views.home, name="home"),
    path("aoi/upload/", views.aoi_upload, name="aoi_upload"),
    path("fields/<int:pk>/risk/", views.risk_map, name="risk_map"),
    path("api/", include(router.urls)),
]