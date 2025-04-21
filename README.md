# Wildlens - Classification des Pattes de Mammifères 🐾

![HEY](/webui/wildlenswebui/static/wildaware-high-resolution-color-logo-crop.png)

**Wildlens** est un projet innovant développé dans le cadre du MSPR de l'EPSI en 3ème année (B3 DevIA spécialité datasciences). Ce projet combine l'intelligence artificielle et le développement web pour créer un modèle de classification des pattes de mammifères et une interface utilisateur intuitive.

## 🌟 Fonctionnalités

- **Classification des Pattes de Mammifères** : Utilisation de techniques avancées de machine learning pour classifier les pattes de différents mammifères.
- **Interface Utilisateur Web** : Une interface web développée avec Django pour interagir avec le modèle de classification.
- **Gestion des Données** : Intégration facile des données des animaux dans la base de données via un script Python.

## 🛠️ Installation

### Prérequis

- Python 3.9 (requis pour la compatibilité avec TensorFlow)
- Assurez-vous d'avoir Python 3.9 installé en plus de toute autre version de Python.

### Étapes d'Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/BahAilime/wildlens.git
   cd wildlens
   ```

2. **Accéder au répertoire webui** :
   ```bash
   cd webui
   ```

3. **Créer un environnement virtuel Python 3.9** :
   ```bash
   py -3.9 -m venv venv
   ```

4. **Activer l'environnement virtuel** :

   *   **Sous Linux/macOS :**
        ```bash
        source venv/bin/activate
        ```
   *   **Sous Windows (PowerShell) :**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
   *   **Sous Windows (CMD) :**
        ```batch
        .\venv\Scripts\activate.bat
        ```

5. **Installer les dépendances** :
   ```bash
   py -3.9 -m pip install -r requirements.txt
   ```

6. **Revenir au répertoire racine du projet**
   ```bash
   cd ..
   ```

7. **Importer les données des animaux dans la base de données** :
   ```bash
   python manage.py import_animals webui/infos_especes.csv
   ```

## 🚀 Utilisation

### Lancer le Serveur

1. **Accéder au répertoire webui** :
   ```bash
   cd webui
   ```
2. **Activer l'environnement virtuel** :

   *   **Sous Linux/macOS :**
        ```bash
        source venv/bin/activate
        ```
   *   **Sous Windows (PowerShell) :**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
   *   **Sous Windows (CMD) :**
        ```batch
        .\venv\Scripts\activate.bat
        ```

3. **Démarrer le serveur de développement :**
   ```bash
   py -3.9 manage.py runserver
   ```

Accédez à l'interface web en ouvrant votre navigateur et en allant à l'adresse `http://127.0.0.1:8000/`.

## 📂 Structure du Projet

- **Partie IA** : Utilisation de Pandas, NumPy, et TensorFlow pour le traitement des données et la création du modèle de classification.
- **Partie Web** : Développement de l'interface utilisateur avec Django.