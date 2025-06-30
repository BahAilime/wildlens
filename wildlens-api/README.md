# WildLens API

API REST Django pour l'application WildLens permettant l'identification d'animaux par IA.

## Description

Cette API constitue le backend de l'application WildLens. Elle fournit les fonctionnalités suivantes :

- Gestion des espèces animales (CRUD)
- Analyse d'images pour identification d'animaux
- Stockage des analyses avec géolocalisation
- Authentification sécurisée

## Configuration requise

- Python 3.9+
- Django 4.2.20
- Autres dépendances listées dans requirements.txt

## Installation

### Installation classique

```bash
# Cloner le dépôt
git clone https://github.com/BahAilime/wildlens.git
cd wildlens

# Créer un environnement virtuel
python -m virtualenv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Appliquer les migrations
python manage.py migrate

# Créer un superutilisateur
python manage.py createsuperuser

# Démarrer le serveur de développement
python manage.py runserver
```

### Installation avec Docker

```bash
# Construire et démarrer les conteneurs
docker-compose up -d --build

# Créer un superutilisateur
docker-compose exec backend python manage.py createsuperuser
```

## Utilisation de l'API

L'API est accessible à l'adresse : http://localhost:8000/api/

Documentation Swagger UI : http://localhost:8000/api/docs/

### Principaux endpoints

- `/api/animals/` - Gestion des espèces animales
- `/api/analyses/` - Gestion des analyses
- `/api/analyses/analyze/` - Point d'entrée pour l'analyse d'images

## Architecture du projet

```
.
├── api/                  # Application principale
│   ├── models/           # Modèles de données séparés
│   ├── serializers/      # Sérialiseurs pour l'API REST
│   ├── views/            # Vues et ViewSets
│   ├── tests/            # Tests unitaires
│   └── ml_model.py       # Modèle d'IA pour l'identification
├── core/                 # Configuration du projet Django
├── Dockerfile            # Configuration Docker
└── docker-compose.yml    # Configuration des services
```

## Licence

Distribué sous licence MIT. Voir `LICENSE` pour plus d'informations.
