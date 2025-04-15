from django.contrib import admin
from .models import Animal, Analysis

@admin.register(Animal)
class AnimalAdmin(admin.ModelAdmin):
    list_display = ('espece', 'nom_latin')  # Colonnes affichées dans la liste
    list_filter = ('famille', 'region')  # Filtres sur le côté
    search_fields = ('espece', 'nom_latin')  # Champs de recherche
    fieldsets = (  # Organisation des champs dans le formulaire d'édition
        ('Informations générales', {
            'fields': ('espece', 'nom_latin', 'famille', 'image')
        }),
        ('Description', {
            'fields': ('description', 'taille', 'fun_fact')
        }),
        ('Localisation', {
            'fields': ('region', 'habitat')
        })
    )

@admin.register(Analysis)
class AnalyseAdmin(admin.ModelAdmin):
    list_display = ('date_creation', 'animal', 'confidence', 'latitude', 'longitude')
    list_filter = ('animal', 'date_creation')
    date_hierarchy = 'date_creation'  # Navigation par date
    readonly_fields = ('date_creation',)