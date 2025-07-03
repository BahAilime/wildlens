import os
import sys
import django
from django.conf import settings

def pytest_configure():
    """Configuration de pytest pour Django"""
    # Définir la variable d'environnement pour les settings Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wildlens.settings')
    
    # Si les settings ne sont pas configurés, créer une configuration minimale
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY='test-secret-key-for-testing-only',
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': ':memory:',
                }
            },
            INSTALLED_APPS=[
                'django.contrib.contenttypes',
                'django.contrib.auth',
                'django.contrib.sessions',
                'django.contrib.messages',
                'rest_framework',
                'api',
            ],
            MIDDLEWARE=[
                'django.middleware.security.SecurityMiddleware',
                'django.contrib.sessions.middleware.SessionMiddleware',
                'django.middleware.common.CommonMiddleware',
                'django.middleware.csrf.CsrfViewMiddleware',
                'django.contrib.auth.middleware.AuthenticationMiddleware',
                'django.contrib.messages.middleware.MessageMiddleware',
            ],
            ROOT_URLCONF='api.urls',
            TEMPLATES=[
                {
                    'BACKEND': 'django.template.backends.django.DjangoTemplates',
                    'DIRS': [],
                    'APP_DIRS': True,
                    'OPTIONS': {
                        'context_processors': [
                            'django.template.context_processors.debug',
                            'django.template.context_processors.request',
                            'django.contrib.auth.context_processors.auth',
                            'django.contrib.messages.context_processors.messages',
                        ],
                    },
                },
            ],
            USE_TZ=True,
            REST_FRAMEWORK={
                'DEFAULT_AUTHENTICATION_CLASSES': [],
                'DEFAULT_PERMISSION_CLASSES': [],
                'TEST_REQUEST_DEFAULT_FORMAT': 'json',
            },
            MEDIA_ROOT='/tmp/test_media',
            MEDIA_URL='/media/',
            STATIC_URL='/static/',
        )
    
    # Configurer Django
    django.setup()