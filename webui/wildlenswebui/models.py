from django.db import models
from django.utils import timezone

class Animal(models.Model):
    espece = models.CharField(max_length=100)
    description = models.TextField()
    nom_latin = models.CharField(max_length=100)
    famille = models.CharField(max_length=100)
    taille = models.TextField()
    region = models.TextField()
    habitat = models.TextField()
    fun_fact = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to='animal_images/', blank=True, null=True)

    def __str__(self):
        return self.espece

class Analysis(models.Model):
    date_creation = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    animal = models.ForeignKey('Animal', on_delete=models.CASCADE, related_name='analyses')
    confidence = models.FloatField()
    image = models.ImageField(upload_to='analyses-images/images/')
    
    def __str__(self):
        return f"Analyse {self.id} - {self.animal.espece} ({self.date_creation.strftime('%d/%m/%Y %H:%M')})"
    
    class Meta:
        verbose_name = "Analyse"
        verbose_name_plural = "Analyses"