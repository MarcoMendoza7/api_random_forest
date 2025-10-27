from pathlib import Path
import os

# settings.py

# Clave secreta para Django (modo desarrollo)
SECRET_KEY = 'django-insecure-<a&j=o*@a9sssl0vu89a^&9--u^pw3sykyi+tt(8xrw-5(gv#j)>'

BASE_DIR = Path(__file__).resolve().parent.parent

# Carpeta del frontend
FRONTEND_DIR = BASE_DIR / 'frontend'

# Debug y hosts
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

# Static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [FRONTEND_DIR]

# Templates
TEMPLATES_DIR = FRONTEND_DIR
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'api', 'templates')],  # <-- aquí buscamos index.html
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
]

INSTALLED_APPS = [
    'django.contrib.admin',       # necesario para /admin/
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api',                        # tu app de la API
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',      # <--- obligatorio
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',   # <--- obligatorio
    'django.contrib.messages.middleware.MessageMiddleware',      # <--- obligatorio
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# Rutas del proyecto
ROOT_URLCONF = 'project.urls'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
