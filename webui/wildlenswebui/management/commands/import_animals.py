import csv
from django.core.management.base import BaseCommand
from wildlenswebui.models import Animal

class Command(BaseCommand):
    help = 'Importe les animaux depuis un fichier CSV'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Chemin vers le fichier CSV')

    def handle(self, *args, **kwargs):
        csv_file_path = kwargs['csv_file']
        
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=';')
            next(csv_reader)  # Ignore la ligne d'en-tête
            
            for row in csv_reader:
                if len(row) >= 8:  # Assurez-vous que la ligne a tous les champs nécessaires
                    Animal.objects.create(
                        espece=row[0],
                        description=row[1],
                        nom_latin=row[2],
                        famille=row[3],
                        taille=row[4],
                        region=row[5],
                        habitat=row[6],
                        fun_fact=row[7] if len(row) > 7 else None
                    )
                    self.stdout.write(self.style.SUCCESS(f'Animal "{row[0]}" importé avec succès'))
                else:
                    self.stdout.write(self.style.WARNING(f'Ligne ignorée car incomplète: {row}'))