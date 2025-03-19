from django.db import models

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