# Ce fichier est maintenu pour la compatibilité
# Les vues sont désormais dans le dossier /api/views/
from api.views.animal_views import AnimalViewSet
from api.views.analysis_views import AnalysisViewSet

__all__ = ['AnimalViewSet', 'AnalysisViewSet']