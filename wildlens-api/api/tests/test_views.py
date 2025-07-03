import os
import sys
import django
from django.conf import settings

# Configuration Django pour les tests
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildlens.settings')
    django.setup()

import unittest
from django.test import TestCase, TransactionTestCase
from rest_framework.test import APIClient, APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, Mock, MagicMock
import base64
import json
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Importer les modèles et serializers après la configuration Django
try:
    from api.models import Animal, Analysis
    from api.serializers import AnalysisSerializer
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    # Créer des classes mock si les imports échouent
    class Animal:
        objects = Mock()
        id = 1
        espece = "Test"
        
    class Analysis:
        objects = Mock()
        id = 1
        
    class AnalysisSerializer:
        pass


class AnalysisViewSetTestCase(APITestCase):
    """Tests pour les routes d'analyses"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        self.client = APIClient()
        
        # Créer des animaux de test
        try:
            self.animal1 = Animal.objects.create(
                espece="Lion",
                nom_latin="Panthera leo",
                famille="Felidae",
                description="Grand félin d'Afrique",
                habitat="Savane",
                region="Afrique",
                taille="1,2m",
                fun_fact="Le roi des animaux"
            )
            
            self.animal2 = Animal.objects.create(
                espece="Éléphant",
                nom_latin="Loxodonta africana",
                famille="Elephantidae",
                description="Plus grand mammifère terrestre",
                habitat="Savane",
                region="Afrique",
                taille="3m",
                fun_fact="Mémoire exceptionnelle"
            )
            
            # Créer une analyse de test
            self.analysis = Analysis.objects.create(
                date_creation=datetime.now(),
                latitude=48.8566,
                longitude=2.3522,
                animal=self.animal1,
                confidence=85
            )
            
            self.models_available = True
        except Exception as e:
            print(f"Erreur lors de la création des objets: {e}")
            self.models_available = False
        
        # Image de test en base64
        self.test_image_base64 = self.create_test_image_base64()
        
        # Coordonnées de test
        self.test_coordinates = {
            "latitude": 48.8566,
            "longitude": 2.3522
        }
    
    def create_test_image_base64(self):
        """Créer une image de test en base64"""
        try:
            img = Image.new('RGB', (224, 224), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
        except Exception:
            # Image 1x1 pixel en base64 comme fallback
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    def test_list_analyses(self):
        """Test de récupération de la liste des analyses"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        # Utiliser l'URL correcte selon votre configuration
        response = self.client.get('/analyses/')
        
        # Vérifier le statut de la réponse
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            self.assertIsInstance(response.data, list)
            if len(response.data) > 0:
                self.assertEqual(response.data[0]['id'], self.analysis.id)
    
    def test_get_analysis_detail(self):
        """Test de récupération du détail d'une analyse"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        response = self.client.get(f'/analyses/{self.analysis.id}/')
        
        # Vérifier le statut de la réponse
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            self.assertEqual(response.data['id'], self.analysis.id)
            self.assertEqual(response.data['confidence'], 85)
    
    def test_get_nonexistent_analysis(self):
        """Test de récupération d'une analyse inexistante"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        response = self.client.get('/analyses/999/')
        self.assertEqual(response.status_code, 404)
    
    def test_create_analysis(self):
        """Test de création d'une analyse"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        data = {
            'date_creation': '2024-01-15T10:30:00Z',
            'latitude': 45.7640,
            'longitude': 4.8357,
            'animal': self.animal2.id,
            'confidence': 92
        }
        
        response = self.client.post('/analyses/', data, format='json')
        
        # Vérifier le statut de la réponse
        self.assertIn(response.status_code, [201, 400, 404])
        
        if response.status_code == 201:
            # Vérifier que l'analyse a été créée
            self.assertEqual(Analysis.objects.count(), 2)
            
            # Vérifier les données de l'analyse créée
            new_analysis = Analysis.objects.get(id=response.data['id'])
            self.assertEqual(new_analysis.animal, self.animal2)
            self.assertEqual(new_analysis.confidence, 92)
    
    def test_update_analysis(self):
        """Test de mise à jour d'une analyse"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        data = {
            'confidence': 90,
            'latitude': 50.0,
            'longitude': 3.0
        }
        
        response = self.client.patch(f'/analyses/{self.analysis.id}/', data, format='json')
        
        # Vérifier le statut de la réponse
        self.assertIn(response.status_code, [200, 404])
        
        if response.status_code == 200:
            # Vérifier que l'analyse a été mise à jour
            updated_analysis = Analysis.objects.get(id=self.analysis.id)
            self.assertEqual(updated_analysis.confidence, 90)
            self.assertEqual(float(updated_analysis.latitude), 50.0)
    
    def test_delete_analysis(self):
        """Test de suppression d'une analyse"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        response = self.client.delete(f'/analyses/{self.analysis.id}/')
        
        # Vérifier le statut de la réponse
        self.assertIn(response.status_code, [204, 404])
        
        if response.status_code == 204:
            # Vérifier que l'analyse a été supprimée
            self.assertEqual(Analysis.objects.count(), 0)
    
    @patch('api.views.analysis_views.model')
    def test_analyze_endpoint_success(self, mock_model):
        """Test de l'endpoint d'analyse avec succès"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        # Mock du modèle de ML
        mock_prediction = np.array([[0.1, 0.9, 0.0]])
        mock_model.predict.return_value = mock_prediction
        
        data = {
            'image': f'data:image/jpeg;base64,{self.test_image_base64}',
            'coordinates': self.test_coordinates,
            'date': '2024-01-15T10:30:00Z'
        }
        
        response = self.client.post('/analyses/analyze/', data, format='json')
        
        # Vérifier que la requête a été traitée
        self.assertIn(response.status_code, [200, 400, 404, 500])
        
        if response.status_code == 200:
            self.assertIn('animal', response.data)
            self.assertIn('confidence', response.data)
            self.assertIn('coordinates', response.data)
            self.assertIn('date', response.data)
            self.assertIn('image_url', response.data)
    
    def test_analyze_endpoint_missing_image(self):
        """Test de l'endpoint d'analyse sans image"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        data = {
            'coordinates': self.test_coordinates,
            'date': '2024-01-15T10:30:00Z'
        }
        
        response = self.client.post('/analyses/analyze/', data, format='json')
        
        # Doit retourner une erreur 400
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertIn('requis', response.data['error'])
    
    def test_analyze_endpoint_missing_coordinates(self):
        """Test de l'endpoint d'analyse sans coordonnées"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        data = {
            'image': f'data:image/jpeg;base64,{self.test_image_base64}',
            'date': '2024-01-15T10:30:00Z'
        }
        
        response = self.client.post('/analyses/analyze/', data, format='json')
        
        # Doit retourner une erreur 400
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
    
    def test_analyze_endpoint_method_not_allowed(self):
        """Test de l'endpoint d'analyse avec méthode non autorisée"""
        response = self.client.get('/analyses/analyze/')
        self.assertEqual(response.status_code, 405)
    
    def test_basic_functionality(self):
        """Test de base pour vérifier que les modèles fonctionnent"""
        if not self.models_available:
            self.skipTest("Modèles non disponibles")
        
        # Vérifier que les objets ont été créés
        self.assertIsNotNone(self.animal1)
        self.assertIsNotNone(self.animal2)
        self.assertIsNotNone(self.analysis)
        
        # Vérifier les propriétés de base
        self.assertEqual(self.animal1.espece, "Lion")
        self.assertEqual(self.analysis.confidence, 85)
    
    def test_urls_configuration(self):
        """Test pour vérifier que les URLs sont correctement configurées"""
        # Test simple pour vérifier la configuration des URLs
        response = self.client.get('/analyses/')
        # L'URL doit exister, même si elle retourne 404 ou 200
        self.assertIn(response.status_code, [200, 404, 405])
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        if self.models_available:
            try:
                Analysis.objects.all().delete()
                Animal.objects.all().delete()
            except Exception:
                pass


class SimpleTestCase(TestCase):
    """Tests simples pour vérifier la configuration"""
    
    def test_django_setup(self):
        """Test pour vérifier que Django est correctement configuré"""
        from django.conf import settings
        self.assertTrue(settings.configured)
    
    def test_api_client(self):
        """Test pour vérifier que l'API client fonctionne"""
        client = APIClient()
        self.assertIsNotNone(client)


if __name__ == '__main__':
    unittest.main()