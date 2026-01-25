"""
Django settings for ProjectAegis project.

Project Aegis: AI-Powered Clinical Decision Support System
for Drug-Drug Interaction (DDI) Prediction using GNN and GraphRAG.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/
"""

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-k5a8aa$4z4zprc2yhp86y&6sjs4pyz&%u4uo#p@@m8*nm^(_q0')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = ['localhost', '127.0.0.1', '.run.app', '*']


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # Third-party apps
    'rest_framework',
    'corsheaders',
    # Project apps
    'ddi_api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Add Whitenoise for static files
    'corsheaders.middleware.CorsMiddleware',  # CORS must be before CommonMiddleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS Configuration - Allow React frontend
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://aegis-frontend-667446742007.us-central1.run.app",
]
CORS_ALLOW_ALL_ORIGINS = True # Temporary for MVP to ensure it works
CORS_ALLOW_CREDENTIALS = True

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
}

# Redis Cache Configuration - Switched to LocMem for MVP (No Redis needed)
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "aegis-cache",
    }
}

# Neo4j Configuration (Knowledge Graph)
# In Docker, hostname is 'neo4j', locally it's 'localhost'
NEO4J_CONFIG = {
    'uri': os.environ.get('NEO4J_URI', 'bolt://neo4j:7687'),
    'user': os.environ.get('NEO4J_USER', 'neo4j'),
    'password': os.environ.get('NEO4J_PASSWORD', 'password123'),
}

# AI Model Configuration
AI_MODEL_CONFIG = {
    'model_path': BASE_DIR / 'models' / 'aegis_ddi_model.pt',
    'onnx_path': BASE_DIR / 'models' / 'aegis_model_optimized.onnx',
    'device': 'cpu',  # Use 'cuda' if GPU is available
}

# =============================================================================
# DDI RETRIEVAL CONFIGURATION (RAG System)
# =============================================================================
# This controls how context sentences are retrieved for PubMedBERT predictions.
#
# Options:
#   'rag'    - [DEFAULT] Live PubMed API - Fetches real medical literature in real-time.
#              Best accuracy, requires internet. ~1-2 second latency per query.
#              Uses NCBI E-utilities API (free, rate-limited to 3 req/sec).
#
#   'hybrid' - [NOT IMPLEMENTED] Checks local corpus first, falls back to PubMed API
#              if no matching sentences found. Balance of speed and coverage.
#
#   'local'  - [NOT IMPLEMENTED] Offline mode using pre-downloaded DDI corpus.
#              Fast (~10ms) but limited to downloaded data. Requires data ingestion.
#              Would use Neo4j or SQLite for sentence storage.
#
DDI_RETRIEVAL_CONFIG = {
    'mode': os.environ.get('DDI_RETRIEVAL_MODE', 'rag'),  # 'rag', 'hybrid', 'local'
    
    # PubMed API Settings (for 'rag' and 'hybrid' modes)
    'pubmed': {
        'base_url': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
        'max_results': 5,           # Number of abstracts to fetch
        'timeout_seconds': 10,      # API request timeout
        'cache_ttl_hours': 24,      # Cache results in Redis for this long
    },
    
    # Local Corpus Settings (for 'local' and 'hybrid' modes) - NOT IMPLEMENTED
    'local': {
        'corpus_path': BASE_DIR / 'data' / 'ddi_sentences.json',  # Would store pre-downloaded sentences
        'use_vector_search': False,  # If True, use embeddings for semantic search
    }
}

ROOT_URLCONF = 'ProjectAegis.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ProjectAegis.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
