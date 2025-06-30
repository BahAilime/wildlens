# Ce fichier est maintenu pour la compatibilité
# Les sérialiseurs sont désormais dans le dossier /api/serializers/
from api.serializers.animal_serializer import AnimalSerializer
from api.serializers.analysis_serializer import AnalysisSerializer

__all__ = ['AnimalSerializer', 'AnalysisSerializer']