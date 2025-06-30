from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnimalViewSet, AnalysisViewSet
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework.documentation import include_docs_urls
from rest_framework.schemas import get_schema_view
from api.views import AnimalViewSet, AnalysisViewSet

router = DefaultRouter()
router.register(r'animals', AnimalViewSet)
router.register(r'analyses', AnalysisViewSet)

schema_view = get_schema_view(title='WildLens API', description='API pour l\'application d\'identification d\'animaux')

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls')),
    path('docs/', include_docs_urls(title='WildLens API Documentation')),
    path('schema/', schema_view, name='api-schema'),
]
router = DefaultRouter()
router.register(r'animals', AnimalViewSet)
router.register(r'analyses', AnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
]