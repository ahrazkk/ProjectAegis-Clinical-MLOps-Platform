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
    PolypharmacyDigitalTwinView,
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
    GraphNodesView,
    GraphNeighborhoodView,
    GraphEdgesView,
    DrugBiologyView,
    MechanismMapView,
)

# Scanner endpoints
from .views_scanner import (
    lookup_by_ndc,
    pill_search,
    enhanced_drug_search,
    validate_barcode,
    analyze_pill_image,
    identify_pill_multimodal,
)

# Create router for ViewSets
router = DefaultRouter()
router.register(r'drugs', DrugViewSet, basename='drug')
router.register(r'history', PredictionLogViewSet, basename='prediction-history')

urlpatterns = [
    # Core prediction endpoints
    path('predict/', DDIPredictionView.as_view(), name='ddi-predict'),
    path('polypharmacy/', PolypharmacyView.as_view(), name='polypharmacy'),
    path('polypharmacy-digital-twin/', PolypharmacyDigitalTwinView.as_view(), name='polypharmacy-digital-twin'),
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
    
    # Drug Scanner endpoints
    path('drugs/ndc/<str:ndc_code>/', lookup_by_ndc, name='ndc-lookup'),
    path('drugs/pill-search/', pill_search, name='pill-search'),
    path('drugs/enhanced-search/', enhanced_drug_search, name='enhanced-search'),
    path('scanner/validate-barcode/', validate_barcode, name='validate-barcode'),
    path('scanner/analyze-pill/', analyze_pill_image, name='analyze-pill'),
    path('scanner/identify-pill/', identify_pill_multimodal, name='identify-pill'),
    
    # Graph visualization endpoints (Galaxy Viewer)
    path('graph/nodes/', GraphNodesView.as_view(), name='graph-nodes'),
    path('graph/neighborhood/', GraphNeighborhoodView.as_view(), name='graph-neighborhood'),
    path('graph/edges/', GraphEdgesView.as_view(), name='graph-edges'),

    # Knowledge Graph V2: Biology endpoints
    path('graph/drug-biology/', DrugBiologyView.as_view(), name='drug-biology'),
    path('graph/mechanism-map/', MechanismMapView.as_view(), name='mechanism-map'),

    # ViewSet routes
    path('', include(router.urls)),
]
