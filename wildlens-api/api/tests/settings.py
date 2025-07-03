from django.conf import settings
import os

# Utiliser une base de données de test
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Répertoire temporaire pour les fichiers de test
MEDIA_ROOT = '/tmp/wildlens_test_media'

# S'assurer que le répertoire existe
os.makedirs(MEDIA_ROOT, exist_ok=True)