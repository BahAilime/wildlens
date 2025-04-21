from PIL import ImageOps, Image
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from .forms import TrackUploadForm
import cv2
import numpy as np
import base64
import json
from .models import Analysis, Animal
from datetime import datetime
import dateutil.parser
from .ml_model import model, class_names
import io
import base64
import io
from django.core.files.base import ContentFile
from PIL import Image
from django.utils import timezone
import os
from django.conf import settings  # Import settings to access MEDIA_ROOT


def save_base64_image(base64_data, upload_path='analyses-images/images/'):
    try:
        # 1. Split and Decode Base64 Data:
        try:
            format, imgstr = base64_data.split(';base64,')
            ext = format.split('/')[-1].lower()  # Extract extension and lowercase it
        except ValueError: #If no format is specified.
            imgstr = base64_data
            ext = 'jpg' #Assume the extension to be jpg
        except:
            return None #If the formatting of the string is really weird.

        decoded_data = base64.b64decode(imgstr)

        # Validate extension
        if ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            ext = 'jpg'  # Set to a default valid extension.

        # 2. Create Image Object from Decoded Data:
        image_data = io.BytesIO(decoded_data)

        # 3. Process and Optimize (with Pillow):
        try:
            img = Image.open(image_data)

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create a new BytesIO object for the processed image
            image_data_processed = io.BytesIO()

            # Save as JPEG with optimization. Change 'quality' as needed
            img.save(image_data_processed, format='JPEG', quality=80)
            image_data_processed.seek(0)  # Reset the stream position

        except Exception as e:
            print(f"Image processing error: {e}")
            return None

        # 4. Generate Unique File Name:
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"image_{timestamp}.jpg" # Always save as JPG after processing.

        # 5. Create Full File Path:
        full_upload_path = os.path.join(os.path.dirname(__file__), upload_path)

        # Ensure upload directory exists
        if not os.path.exists(full_upload_path):
            os.makedirs(full_upload_path)

        django_file = ContentFile(image_data_processed.read(), name=file_name)

        return django_file, os.path.join(upload_path, file_name), file_name
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return None


def base64_to_cv2_image(base64_string):
    # retire les en-tetes comme "data:image/jpeg;base64,"
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)

    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def home(request):
    return render(request, 'wildlenswebui/home.html')

def scan_track(request):
    if request.method == 'POST':
        form = TrackUploadForm(request.POST, request.FILES)
        if form.is_valid():

            # TODO: Utiliser le modèle pour identifier l'animal
            identified_animal = Animal.objects.first()

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'id': identified_animal.id,
                    'espece': identified_animal.espece,
                    'nom_latin': identified_animal.nom_latin,
                    'famille': identified_animal.famille,
                    'taille': identified_animal.taille,
                    'habitat': identified_animal.habitat,
                    'description': identified_animal.description,
                    'region': identified_animal.region,
                    'fun_fact': identified_animal.fun_fact,
                    'image_url': identified_animal.image.url if identified_animal.image else '',
                })
            else:
                return redirect('animal_detail', animal_id=identified_animal.id)
    else:
        form = TrackUploadForm()

    return render(request, 'wildlenswebui/scan.html', {'form': form})

def animal_detail(request, animal_id):
    animal = Animal.objects.get(pk=animal_id)
    return render(request, 'wildlenswebui/animal_detail.html', {'animal': animal})

def index(request):
    return render(request, 'wildlenswebui/index.html',)

def analize(request):
    image_data = request.POST.get('image')
    coo = request.POST.get('coordinates')
    coo = json.loads(coo)
    date = dateutil.parser.isoparse(request.POST.get('date'))

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    cv2_img = base64_to_cv2_image(image_data)

    # Convertir l'image OpenCV en image PIL
    image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    image_analyse = Image.fromarray(image).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image_analyse, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    print(index)

    # for animal in Animal.objects.all():
    #     print(animal.espece)

    predicted_animal = Animal.objects.get(id__iexact=index+1)

    # Créer une nouvelle analyse
    image, image_path, image_file_name = save_base64_image(image_data)
    newAnalysis = Analysis.objects.create(
        date_creation=date,
        latitude=coo["latitude"],
        longitude=coo["longitude"],
        animal=predicted_animal,
        confidence=round(confidence_score * 100),
    )
    newAnalysis.image.save(image_file_name, image, save=True)

    # PIL (RGB) → numpy (BGR)
    image_opencv = np.array(image_analyse)
    image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)

    # Encodage en JPEG + base64 pour l'envoyer
    _, buffer = cv2.imencode('.jpg', image_opencv)
    image_base64 = base64.b64encode(buffer).decode()

    try:
        specie_image = predicted_animal.image.url
    except:
        specie_image = ""

    return JsonResponse({
        "name": predicted_animal.espece,
        "latin": predicted_animal.nom_latin,
        "image": specie_image,
        "description": predicted_animal.description,
        "habitat": predicted_animal.habitat,
        "track_info": f"Trouvé à: {coo['latitude']}, {coo['longitude']}",
        "confidence": round(confidence_score * 100),
        "region": predicted_animal.region,
        "famille": predicted_animal.famille,
        "taille": predicted_animal.taille,
        "fun_fact": predicted_animal.fun_fact,
        "img": image_base64
    })