from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from app_core import views as core_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('accounts/signup/', core_views.signup, name='signup'),
    path("", include("app_core.urls")),
    path("accounts/", include("django.contrib.auth.urls"))
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)