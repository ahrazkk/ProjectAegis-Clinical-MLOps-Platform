"""
URL configuration for Project Aegis

API Endpoints:
- /api/v1/ - DDI API (prediction, polypharmacy, chat, drugs)
- /admin/ - Django Admin interface
"""
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse


def api_root(request):
    """API root endpoint with documentation."""
    return JsonResponse({
        'name': 'Project Aegis API',
        'version': '1.0.0',
        'description': 'AI-Powered Clinical Decision Support System for Drug-Drug Interaction Prediction',
        'endpoints': {
            'predict': '/api/v1/predict/',
            'polypharmacy': '/api/v1/polypharmacy/',
            'chat': '/api/v1/chat/',
            'drugs': '/api/v1/drugs/',
            'search': '/api/v1/search/',
            'health': '/api/v1/health/',
        },
        'documentation': '/api/v1/',
    })


urlpatterns = [
    path('', api_root, name='api-root'),
    path('admin/', admin.site.urls),
    path('api/v1/', include('ddi_api.urls')),
]
