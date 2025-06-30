from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from api.models import Animal, Analysis
from api.serializers import AnalysisSerializer
import numpy as np
from PIL import Image, ImageOps
import cv2
import base64
import io
from django.core.files.base import ContentFile
from django.utils import timezone
import os
from datetime import datetime
import dateutil.parser
from api.ml_model import model, class_names

class AnalysisViewSet(viewsets.ModelViewSet):
    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer

    def save_base64_image(self, base64_data, upload_path='analyses-images/images/'):
        try:
            try:
                format, imgstr = base64_data.split(';base64,')
                ext = format.split('/')[-1].lower()
            except ValueError:
                imgstr = base64_data
                ext = 'jpg'
            except:
                return None

            decoded_data = base64.b64decode(imgstr)

            if ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                ext = 'jpg'

            image_data = io.BytesIO(decoded_data)

            try:
                img = Image.open(image_data)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                image_data_processed = io.BytesIO()
                img.save(image_data_processed, format='JPEG', quality=80)
                image_data_processed.seek(0)

            except Exception as e:
                print(f"Erreur de traitement d'image: {e}")
                return None

            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"image_{timestamp}.jpg"

            full_upload_path = os.path.join(os.path.dirname(__file__), upload_path)

            if not os.path.exists(full_upload_path):
                os.makedirs(full_upload_path)

            django_file = ContentFile(image_data_processed.read(), name=file_name)

            return django_file, os.path.join(upload_path, file_name), file_name
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'image base64: {e}")
            return None

    def base64_to_cv2_image(self, base64_string):
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]

        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @action(detail=False, methods=['post'])
    def analyze(self, request):
        try:
            image_data = request.data.get('image')
            coordinates = request.data.get('coordinates')
            date_str = request.data.get('date')

            if not all([image_data, coordinates, date_str]):
                return Response(
                    {"error": "Image, coordinates et date sont requis"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            date = dateutil.parser.isoparse(date_str)

            # Préparation de l'image pour le modèle
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            cv2_img = self.base64_to_cv2_image(image_data)
            image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            image_analyse = Image.fromarray(image).convert("RGB")

            # Redimensionnement
            size = (224, 224)
            image = ImageOps.fit(image_analyse, size, Image.Resampling.LANCZOS)

            # Normalisation
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array

            # Prédiction
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence_score = float(prediction[0][index])

            # Récupération de l'animal prédit
            predicted_animal = Animal.objects.get(id__iexact=index+1)

            # Sauvegarde de l'image et création de l'analyse
            image, image_path, image_file_name = self.save_base64_image(image_data)

            analysis = Analysis.objects.create(
                date_creation=date,
                latitude=coordinates.get('latitude'),
                longitude=coordinates.get('longitude'),
                animal=predicted_animal,
                confidence=round(confidence_score * 100)
            )
            analysis.image.save(image_file_name, image, save=True)

            return Response({
                "id": analysis.id,
                "animal": {
                    "id": predicted_animal.id,
                    "espece": predicted_animal.espece,
                    "nom_latin": predicted_animal.nom_latin,
                    "famille": predicted_animal.famille,
                    "description": predicted_animal.description,
                    "habitat": predicted_animal.habitat,
                    "region": predicted_animal.region,
                    "taille": predicted_animal.taille,
                    "fun_fact": predicted_animal.fun_fact,
                    "image": predicted_animal.image.url if predicted_animal.image else None
                },
                "confidence": round(confidence_score * 100),
                "coordinates": coordinates,
                "date": date.isoformat(),
                "image_url": analysis.image.url
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
