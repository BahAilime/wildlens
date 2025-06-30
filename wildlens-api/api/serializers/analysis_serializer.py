from rest_framework import serializers
from api.models import Analysis, Animal


class AnalysisSerializer(serializers.ModelSerializer):
    animal_details = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Analysis
        fields = [
            'id',
            'date_creation',
            'latitude',
            'longitude',
            'animal',
            'animal_details',
            'confidence',
            'image'
        ]
        read_only_fields = ['date_creation']

    def get_animal_details(self, obj):
        """Retourne les détails de l'animal associé à l'analyse"""
        return {
            'id': obj.animal.id,
            'espece': obj.animal.espece,
            'nom_latin': obj.animal.nom_latin,
            'famille': obj.animal.famille,
            'description': obj.animal.description,
            'image': obj.animal.image.url if obj.animal.image else None
        }
