"""
URL Configuration for DDI API

Endpoints:
- /api/v1/predict/ - DDI prediction for 2 drugs
- /api/v1/polypharmacy/ - N-way drug interaction analysis
- /api/v1/chat/ - GraphRAG research assistant
- /api/v1/drugs/ - Drug search and CRUD
- /api/v1/health/ - System health check
- /api/v1/stats/ - Database statistics
- /api/v1/alternatives/ - Therapeutic alternatives
- /api/v1/compare/ - Drug comparison
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    DDIPredictionView,
    PolypharmacyView,
    ChatView,
    DrugSearchView,
    HealthCheckView,
    DrugViewSet,
    PredictionLogViewSet,
    EnhancedDrugInfoView,
    EnhancedInteractionInfoView,
    RealWorldEvidenceView,
    DatabaseStatsView,
    TherapeuticAlternativesView,
    DrugComparisonView,
)

# Create router for ViewSets
router = DefaultRouter()
router.register(r'drugs', DrugViewSet, basename='drug')
router.register(r'history', PredictionLogViewSet, basename='prediction-history')

urlpatterns = [
    # Core prediction endpoints
    path('predict/', DDIPredictionView.as_view(), name='ddi-predict'),
    path('polypharmacy/', PolypharmacyView.as_view(), name='polypharmacy'),
    path('chat/', ChatView.as_view(), name='chat'),
    
    # Search
    path('search/', DrugSearchView.as_view(), name='drug-search'),
    
    # Enhanced information endpoints
    path('drug-info/', EnhancedDrugInfoView.as_view(), name='drug-info'),
    path('interaction-info/', EnhancedInteractionInfoView.as_view(), name='interaction-info'),
    path('real-world-evidence/', RealWorldEvidenceView.as_view(), name='real-world-evidence'),
    
    # Dashboard & Analytics endpoints
    path('stats/', DatabaseStatsView.as_view(), name='database-stats'),
    path('alternatives/', TherapeuticAlternativesView.as_view(), name='therapeutic-alternatives'),
    path('compare/', DrugComparisonView.as_view(), name='drug-compare'),
    
    # Health check
    path('health/', HealthCheckView.as_view(), name='health-check'),
    
    # ViewSet routes
    path('', include(router.urls)),
]
