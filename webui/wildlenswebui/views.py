# wildlenswebui/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import Animal
from .forms import TrackUploadForm

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