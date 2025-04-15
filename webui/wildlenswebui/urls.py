"""
URL configuration for wildlenswebui project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from wildlenswebui import views
from django.contrib import admin

urlpatterns = [
    path('old', views.home, name='home'),
    path('', views.index, name='index'),
    path('scan/', views.scan_track, name='scan_track'),
    path('analize/', views.analize, name='analize'),
    path('animal/<int:animal_id>/', views.animal_detail, name='animal_detail'),
    path('admin/', admin.site.urls),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)