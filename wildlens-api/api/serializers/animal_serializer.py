from rest_framework import serializers
from api.models import Animal


class AnimalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Animal
        fields = [
            'id',
            'espece',
            'description',
            'nom_latin',
            'famille',
            'taille',
            'region',
            'habitat',
            'fun_fact',
            'image'
        ]
