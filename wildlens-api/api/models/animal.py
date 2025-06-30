from django.db import models


class Animal(models.Model):
    espece = models.CharField(max_length=100)
    description = models.TextField(default="Aucune description disponible.")
    nom_latin = models.CharField(max_length=100, null=True)
    famille = models.CharField(max_length=100, default="Mammifère")
    taille = models.TextField(null=True)
    region = models.TextField(null=True)
    habitat = models.TextField(null=True)
    fun_fact = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to='animal_images/', blank=True, null=True)

    def __str__(self):
        return self.espece
