from django.db import models
from django.utils import timezone
from api.models.animal import Animal


class Analysis(models.Model):
    date_creation = models.DateTimeField(default=timezone.now)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    animal = models.ForeignKey(Animal, on_delete=models.CASCADE, related_name='analyses')
    confidence = models.FloatField()
    image = models.ImageField(upload_to='analyses-images/images/')

    def __str__(self):
        return f"Analyse {self.id} - {self.animal.espece} ({self.date_creation.strftime('%d/%m/%Y %H:%M')})"

    class Meta:
        verbose_name = "Analyse"
        verbose_name_plural = "Analyses"
