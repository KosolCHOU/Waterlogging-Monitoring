# app_core/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from app_core.views import LogoutViewAllowGet
from app_core.decorators import login_required_with_message

router = DefaultRouter()
router.register(r'fields', views.FieldViewSet, basename='fields')

urlpatterns = [
    path("", login_required_with_message(views.lands), name="home"),
    path("lands/", login_required_with_message(views.lands), name="lands"),
    path("aoi_upload/", login_required_with_message(views.aoi_upload), name="aoi_upload"),
    path("dashboard/", login_required_with_message(views.dashboard_index), name="dashboard_index"),
    path("dashboard/<int:field_id>/", login_required_with_message(views.dashboard), name="dashboard"),
    path("probe/<int:job_id>/", login_required_with_message(views.probe), name="probe"),
    path("fields/<int:field_id>/insights/", login_required_with_message(views.field_insights_api), name="field_insights_api"),
    path("api/", include(router.urls)),
    path("about/", login_required_with_message(views.about), name="about"),
    path('fields/<int:field_id>/analytics/', login_required_with_message(views.analytics), name='analytics'),
    path('analytics/', login_required_with_message(views.analytics), name='analytics_no_field'),
    path('fields/<int:field_id>/forecast.json', login_required_with_message(views.forecast_json), name='forecast_json'),
    path("profile/", login_required_with_message(views.profile), name="profile"),
    path("accounts/logout/", LogoutViewAllowGet.as_view(next_page="login"), name="logout"),
]
