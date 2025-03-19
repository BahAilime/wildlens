# Wildlens - Classification des Pattes de Mammifères 🐾

![HEY](/webui/wildlenswebui/static/wildaware-high-resolution-color-logo-crop.png)

**Wildlens** est un projet innovant développé dans le cadre du MSPR de l'EPSI en 3ème année (B3 DevIA spécialité datasciences). Ce projet combine l'intelligence artificielle et le développement web pour créer un modèle de classification des pattes de mammifères et une interface utilisateur intuitive.

## 🌟 Fonctionnalités

- **Classification des Pattes de Mammifères** : Utilisation de techniques avancées de machine learning pour classifier les pattes de différents mammifères.
- **Interface Utilisateur Web** : Une interface web développée avec Django pour interagir avec le modèle de classification.
- **Gestion des Données** : Intégration facile des données des animaux dans la base de données via un script Python.
- **Gestion des Dépendances** : Utilisation de Poetry pour une gestion simplifiée des dépendances.

## 🛠️ Installation

### Prérequis

- Python 3.9 pour la création du modèle
- Poetry (pour la gestion des dépendances)

### Étapes d'Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/BahAilime/wildlens.git
   cd wildlens
   ```

2. **Installer les dépendances avec Poetry** :
   ```bash
   poetry install
   ```

3. **Importer les données des animaux dans la base de données** :
   ```bash
   python manage.py import_animals ../infos_especes.csv
   ```

## 🚀 Utilisation

### Lancer le Serveur

Pour démarrer le serveur de développement, exécutez la commande suivante :

```bash
poetry run manage.py runserver
```

Accédez à l'interface web en ouvrant votre navigateur et en allant à l'adresse `http://127.0.0.1:8000/`.

## 📂 Structure du Projet

- **Partie IA** : Utilisation de Pandas, NumPy, et Scikit-learn pour le traitement des données et la création du modèle de classification.
- **Partie Web** : Développement de l'interface utilisateur avec Django.
