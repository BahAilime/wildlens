import json
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from api.models import Animal, Analysis
from ..utils.test_helpers import WildLensAnalysisTestCase
import numpy as np


class AnalysisIntegrationTest(WildLensAnalysisTestCase):
    """Tests d'intégration pour l'analyse des pattes"""

    def setUp(self):
        super().setUp()
        self.client = APIClient()
        self.analyze_url = reverse('analysis-analyze')

    @patch('api.ml_model.model.predict')
    def test_complete_analysis_workflow(self, mock_predict):
        """Test complet du workflow d'analyse"""
        # Mock de la prédiction du modèle
        mock_predict.return_value = np.array([[0.05, 0.95]])  # Index 1 = Loup (95%)

        # Données de test
        test_data = {
            'image': self.create_test_paw_image_base64(),
            'coordinates': self.get_test_coordinates(),
            'date': self.get_test_date()
        }

        # Appel de l'API
        response = self.client.post(self.analyze_url, test_data, format='json')

        # Vérifications
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        response_data = response.json()

        # Vérifier la structure de la réponse
        self.assertIn('id', response_data)
        self.assertIn('animal', response_data)
        self.assertIn('confidence', response_data)
        self.assertIn('coordinates', response_data)
        self.assertIn('date', response_data)
        self.assertIn('image_url', response_data)

        # Vérifier les données de l'animal
        animal_data = response_data['animal']
        self.assertEqual(animal_data['espece'], 'Loup')
        self.assertEqual(animal_data['nom_latin'], 'Canis lupus')
        self.assertEqual(animal_data['famille'], 'Canidae')

        # Vérifier la confiance
        self.assertEqual(response_data['confidence'], 95)

        # Vérifier que l'analyse a été sauvegardée
        self.assertEqual(Analysis.objects.count(), 1)
        analysis = Analysis.objects.first()
        self.assertEqual(analysis.animal, self.loup)
        self.assertEqual(analysis.confidence, 95)
        self.assertEqual(analysis.latitude, 48.8566)
        self.assertEqual(analysis.longitude, 2.3522)

        # Vérifier que le modèle a été appelé
        mock_predict.assert_called_once()

    @patch('api.ml_model.model.predict')
    def test_analysis_with_different_predictions(self, mock_predict):
        """Test avec différentes prédictions"""
        test_cases = [
            {
                'prediction': np.array([[0.95, 0.05]]),  # Index 0 = Loup
                'expected_animal_id': 1,
                'expected_confidence': 95
            },
            {
                'prediction': np.array([[0.20, 0.80]]),  # Index 1 = Chien
                'expected_animal_id': 2,
                'expected_confidence': 80
            }
        ]

        for case in test_cases:
            with self.subTest(case=case):
                mock_predict.return_value = case['prediction']

                test_data = {
                    'image': self.create_test_paw_image_base64(),
                    'coordinates': self.get_test_coordinates(),
                    'date': self.get_test_date()
                }

                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                response_data = response.json()

                self.assertEqual(response_data['animal']['id'], case['expected_animal_id'])
                self.assertEqual(response_data['confidence'], case['expected_confidence'])

    def test_analysis_with_missing_data(self):
        """Test avec données manquantes"""
        test_cases = [
            {'image': None, 'coordinates': self.get_test_coordinates(), 'date': self.get_test_date()},
            {'image': self.create_test_paw_image_base64(), 'coordinates': None, 'date': self.get_test_date()},
            {'image': self.create_test_paw_image_base64(), 'coordinates': self.get_test_coordinates(), 'date': None}
        ]

        for test_data in test_cases:
            with self.subTest(test_data=test_data):
                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
                response_data = response.json()
                self.assertIn('error', response_data)
                self.assertIn('requis', response_data['error'])

    def test_analysis_with_invalid_base64_image(self):
        """Test avec image base64 invalide"""
        test_data = {
            'image': 'invalid_base64_string',
            'coordinates': self.get_test_coordinates(),
            'date': self.get_test_date()
        }

        response = self.client.post(self.analyze_url, test_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        response_data = response.json()
        self.assertIn('error', response_data)

    @patch('api.ml_model.model.predict')
    def test_analysis_image_processing_pipeline(self, mock_predict):
        """Test du pipeline de traitement d'image"""
        mock_predict.return_value = np.array([[0.90, 0.10]])

        # Test avec différents formats d'image base64
        test_formats = [
            self.create_test_paw_image_base64('red'),
            self.create_test_paw_image_base64('blue'),
            self.create_test_paw_image_base64('green')
        ]

        for img_base64 in test_formats:
            with self.subTest(image=img_base64[:50]):
                test_data = {
                    'image': img_base64,
                    'coordinates': self.get_test_coordinates(),
                    'date': self.get_test_date()
                }

                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_200_OK)

                # Vérifier que l'image a été traitée (224x224, normalisée)
                call_args = mock_predict.call_args[0][0]
                self.assertEqual(call_args.shape, (1, 224, 224, 3))
                self.assertTrue(np.all(call_args >= -1.0) and np.all(call_args <= 1.0))

    @patch('api.ml_model.model.predict')
    def test_analysis_coordinates_validation(self, mock_predict):
        """Test de validation des coordonnées"""
        mock_predict.return_value = np.array([[0.85, 0.15]])

        test_coordinates = [
            {'latitude': 48.8566, 'longitude': 2.3522},  # Paris - valide
            {'latitude': -90.0, 'longitude': 180.0},  # Limites - valide
            {'latitude': 90.0, 'longitude': -180.0},  # Limites - valide
        ]

        for coords in test_coordinates:
            with self.subTest(coords=coords):
                test_data = {
                    'image': self.create_test_paw_image_base64(),
                    'coordinates': coords,
                    'date': self.get_test_date()
                }

                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                response_data = response.json()

                self.assertEqual(response_data['coordinates']['latitude'], coords['latitude'])
                self.assertEqual(response_data['coordinates']['longitude'], coords['longitude'])

    @patch('api.ml_model.model.predict')
    def test_analysis_date_parsing(self, mock_predict):
        """Test du parsing des dates"""
        mock_predict.return_value = np.array([[0.88, 0.12]])

        test_dates = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+01:00",
            "2024-01-15T10:30:00.123Z"
        ]

        for date_str in test_dates:
            with self.subTest(date=date_str):
                test_data = {
                    'image': self.create_test_paw_image_base64(),
                    'coordinates': self.get_test_coordinates(),
                    'date': date_str
                }

                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                response_data = response.json()
                self.assertIn('date', response_data)

    @patch('api.ml_model.model.predict')
    def test_analysis_image_storage(self, mock_predict):
        """Test du stockage des images"""
        mock_predict.return_value = np.array([[0.92, 0.08]])

        test_data = {
            'image': self.create_test_paw_image_base64(),
            'coordinates': self.get_test_coordinates(),
            'date': self.get_test_date()
        }

        response = self.client.post(self.analyze_url, test_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        response_data = response.json()

        # Vérifier que l'image a été stockée
        self.assertIn('image_url', response_data)
        self.assertIsNotNone(response_data['image_url'])

        # Vérifier que l'analyse a une image associée
        analysis = Analysis.objects.first()
        self.assertTrue(analysis.image)
        self.assertTrue(analysis.image.url)

    @patch('api.ml_model.model.predict')
    def test_analysis_error_handling(self, mock_predict):
        """Test de gestion d'erreurs lors de l'analyse"""
        # Simuler une erreur du modèle
        mock_predict.side_effect = Exception("Erreur du modèle TensorFlow")

        test_data = {
            'image': self.create_test_paw_image_base64(),
            'coordinates': self.get_test_coordinates(),
            'date': self.get_test_date()
        }

        response = self.client.post(self.analyze_url, test_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        response_data = response.json()
        self.assertIn('error', response_data)
        self.assertIn('Erreur du modèle TensorFlow', response_data['error'])

    @patch('api.ml_model.model.predict')
    def test_analysis_confidence_calculation(self, mock_predict):
        """Test du calcul de confiance"""
        test_cases = [
            {'prediction': np.array([[0.95, 0.05]]), 'expected_confidence': 95},
            {'prediction': np.array([[0.67, 0.33]]), 'expected_confidence': 67},
            {'prediction': np.array([[0.51, 0.49]]), 'expected_confidence': 51},
            {'prediction': np.array([[0.12, 0.88]]), 'expected_confidence': 88}
        ]

        for case in test_cases:
            with self.subTest(case=case):
                mock_predict.return_value = case['prediction']

                test_data = {
                    'image': self.create_test_paw_image_base64(),
                    'coordinates': self.get_test_coordinates(),
                    'date': self.get_test_date()
                }

                response = self.client.post(self.analyze_url, test_data, format='json')

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                response_data = response.json()

                self.assertEqual(response_data['confidence'], case['expected_confidence'])

    @patch('api.ml_model.model.predict')
    def test_analysis_database_consistency(self, mock_predict):
        """Test de cohérence avec la base de données"""
        mock_predict.return_value = np.array([[0.93, 0.07]])

        initial_count = Analysis.objects.count()

        test_data = {
            'image': self.create_test_paw_image_base64(),
            'coordinates': self.get_test_coordinates(),
            'date': self.get_test_date()
        }

        response = self.client.post(self.analyze_url, test_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Vérifier qu'une nouvelle analyse a été créée
        self.assertEqual(Analysis.objects.count(), initial_count + 1)

        # Vérifier la cohérence des données
        analysis = Analysis.objects.latest('id')
        response_data = response.json()

        self.assertEqual(analysis.id, response_data['id'])
        self.assertEqual(analysis.confidence, response_data['confidence'])
        self.assertEqual(analysis.animal.id, response_data['animal']['id'])