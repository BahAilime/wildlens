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
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    cv2_img = base64_to_cv2_image(image_data)

    # Convertir l'image OpenCV en image PIL
    image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

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
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    newAnalysis = Analysis.objects.create(
        date_creation=date,
        latitude=coo["latitude"],
        longitude=coo["longitude"],
        animal=Animal.objects.first(),
        confidence=75,
        image=image_data
    )

    # Capturer la summary dans une string
    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    summary = stream.getvalue()

    return JsonResponse({
        "name": str(class_name),
        "latin": "Testus testicus",
        "image": "test.jpg",
        "description": "A test species for testing purposes.",
        "habitat": "Test environment",
        "track_info": "Test track information",
        "confidence": 75,
        "img": base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()
        })
