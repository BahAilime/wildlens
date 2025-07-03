import base64
import io
from PIL import Image
from django.test import TestCase
from api.models import Animal


class WildLensAnalysisTestCase(TestCase):
    """Classe de base pour les tests d'analyse WildLens"""

    def setUp(self):
        """Configuration des données de test"""
        # Créer des animaux de test correspondant aux indices du modèle
        self.chat = Animal.objects.create(
            id=1,
            espece="Chat sauvage",
            nom_latin="Felis silvestris",
            famille="Felidae",
            description="Petit félin sauvage",
            habitat="Forêts et prairies",
            region="Europe",
            taille="40-65 cm",
            fun_fact="Ancêtre du chat domestique"
        )

        self.loup = Animal.objects.create(
            id=2,
            espece="Loup",
            nom_latin="Canis lupus",
            famille="Canidae",
            description="Grand canidé sauvage",
            habitat="Forêts et toundra",
            region="Hémisphère Nord",
            taille="100-160 cm",
            fun_fact="Ancêtre du chien domestique"
        )

    def create_test_paw_image_base64(self, color='brown'):
        """Crée une image de patte en base64 pour les tests"""
        # Créer une image de test 224x224 (format attendu par le modèle)
        image = Image.new('RGB', (224, 224), color=color)

        # Convertir en base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"

    def get_test_coordinates(self):
        """Retourne des coordonnées de test"""
        return {
            'latitude': 48.8566,
            'longitude': 2.3522
        }

    def get_test_date(self):
        """Retourne une date de test au format ISO"""
        return "2024-01-15T10:30:00Z"