"""
Django settings for ProjectAegis project.

Project Aegis: AI-Powered Clinical Decision Support System
for Drug-Drug Interaction (DDI) Prediction using GNN and GraphRAG.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/
"""

from pathlib import Path
import os
import sys

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

if load_dotenv is not None and not os.environ.get('K_SERVICE'):
    # Allow local scripts (outside docker compose) to reuse web/.env values.
    # On Cloud Run, rely on service environment variables and ignore bundled .env.
    load_dotenv(BASE_DIR / '.env', override=False)


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = ""):
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(',') if item.strip()]


RUNNING_TESTS = 'test' in sys.argv or 'PYTEST_CURRENT_TEST' in os.environ


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = _env_bool('DEBUG', False)

# SECURITY WARNING: keep the secret key used in production secret!
if RUNNING_TESTS:
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'aegis-test-secret-key')
elif DEBUG:
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'aegis-dev-secret-key')
else:
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError('DJANGO_SECRET_KEY environment variable is required when DEBUG=False')

ALLOWED_HOSTS = _env_csv(
    'ALLOWED_HOSTS',
    'localhost,127.0.0.1,.run.app,aegishealth.dev,www.aegishealth.dev'
)


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
    "https://aegis-frontend-ivk6owqmqa-uc.a.run.app",
    "https://aegishealth.dev",
] + _env_csv('EXTRA_CORS_ALLOWED_ORIGINS', '')
CORS_ALLOW_ALL_ORIGINS = _env_bool('CORS_ALLOW_ALL_ORIGINS', False)
CORS_ALLOW_CREDENTIALS = _env_bool('CORS_ALLOW_CREDENTIALS', True)

# REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_THROTTLE_CLASSES': [] if RUNNING_TESTS else [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': os.environ.get('DRF_THROTTLE_ANON', '120/hour'),
        'user': os.environ.get('DRF_THROTTLE_USER', '1000/hour'),
    },
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
# LLM RESEARCH ASSISTANT (Gemini)
# =============================================================================
GEMINI_CONFIG = {
    'api_key': os.environ.get('GEMINI_API_KEY', ''),
    'model': os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
    'max_output_tokens': 4096,
    'temperature': 0.3,       # Low for clinical accuracy
    'top_p': 0.9,
}

ASSISTANT_CONFIG = {
    'enabled': _env_bool('AEGIS_ASSISTANT_ENABLED', False),
    'access_password': os.environ.get('AEGIS_ASSISTANT_PASSWORD', ''),
    'max_context_tokens': 4000,
    'max_pubmed_results': 3,
}

# NCBI API key for higher PubMed rate limits (10 req/sec vs 3 req/sec)
# Register at: https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_API_KEY = os.environ.get('NCBI_API_KEY', '')

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
