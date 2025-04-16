from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Animal
from .forms import TrackUploadForm
import cv2
import numpy as np
import base64
import json
from .models import Analysis, Animal
from datetime import datetime
import dateutil.parser


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
    
    image = base64_to_cv2_image(image_data)
    resized_img = cv2.resize(image, (224, 224))

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Application d'un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Forcer le type en uint8 après le flou
    blurred = np.uint8(blurred)

    # Seuillage adaptatif pour isoler l'empreinte
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Conversion BGR à RGB (TensorFlow utilise RGB)
    rgb_img = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)


    # a = model.predict(rgb_img)
    a = "YES"

    newAnalysis = Analysis.objects.create(
        date_creation=date,
        latitude=coo["latitude"],
        longitude=coo["longitude"],
        animal=Animal.objects.first(),
        confidence=75,
        image=image_data
    )

    return JsonResponse({
        "name": str(a),
        "latin": "Testus testicus",
        "image": "test.jpg",
        "description": "A test species for testing purposes.",
        "habitat": "Test environment",
        "track_info": "Test track information",
        "confidence": 75,
        "img": base64.b64encode(cv2.imencode('.jpg', rgb_img)[1]).decode()
        })
