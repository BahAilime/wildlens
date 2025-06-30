from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from api.models import Animal
from api.serializers import AnimalSerializer
import csv

class AnimalViewSet(viewsets.ModelViewSet):
    queryset = Animal.objects.all()
    serializer_class = AnimalSerializer

    @action(detail=False, methods=['post'])
    def import_csv(self, request):
        try:
            if 'file' not in request.FILES:
                return Response(
                    {"error": "Aucun fichier CSV n'a été fourni"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            csv_file = request.FILES['file']
            if not csv_file.name.endswith('.csv'):
                return Response(
                    {"error": "Le fichier doit être au format CSV"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            decoded_file = csv_file.read().decode('utf-8').splitlines()
            csv_reader = csv.reader(decoded_file, delimiter=';')
            next(csv_reader)  # Ignorer l'en-tête

            imported_count = 0
            skipped_count = 0
            results = []

            for row in csv_reader:
                if len(row) >= 8:
                    animal = Animal.objects.create(
                        espece=row[0],
                        description=row[1],
                        nom_latin=row[2],
                        famille=row[3],
                        taille=row[4],
                        region=row[5],
                        habitat=row[6],
                        fun_fact=row[7] if len(row) > 7 else None
                    )
                    imported_count += 1
                    results.append({
                        "status": "success",
                        "espece": row[0],
                        "id": animal.id
                    })
                else:
                    skipped_count += 1
                    results.append({
                        "status": "error",
                        "message": f"Ligne ignorée car incomplète",
                        "data": row
                    })

            return Response({
                "message": f"{imported_count} animaux importés, {skipped_count} lignes ignorées",
                "results": results
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
