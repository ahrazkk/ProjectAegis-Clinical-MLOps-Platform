"""
API Views for Project Aegis DDI API

These views provide the REST endpoints for:
1. DDI Prediction (2-drug interaction)
2. Polypharmacy Analysis (N-way interactions)
3. GraphRAG Chatbot (Research Assistant)
4. Drug Search and CRUD operations
"""

import time
import uuid
import logging
from typing import Dict, List

from django.conf import settings
from rest_framework import status, viewsets
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Drug, DrugTarget, DrugDrugInteraction, SideEffect, PredictionLog
from .serializers import (
    DDIPredictionRequestSerializer,
    DDIPredictionResponseSerializer,
    PolypharmacyRequestSerializer,
    PolypharmacyResponseSerializer,
    ChatRequestSerializer,
    ChatResponseSerializer,
    DrugSerializer,
    DrugDrugInteractionSerializer,
    PredictionLogSerializer,
)
from .services.ddi_predictor import get_ddi_service, DDIPrediction

logger = logging.getLogger(__name__)


# ============== Sample Drug Data (for demo) ==============
# In production, this would come from the database
SAMPLE_DRUGS = {
    'warfarin': {
        'name': 'Warfarin',
        'smiles': 'CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O',
        'drugbank_id': 'DB00682'
    },
    'aspirin': {
        'name': 'Aspirin',
        'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'drugbank_id': 'DB00945'
    },
    'ibuprofen': {
        'name': 'Ibuprofen',
        'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'drugbank_id': 'DB01050'
    },
    'metformin': {
        'name': 'Metformin',
        'smiles': 'CN(C)C(=N)NC(=N)N',
        'drugbank_id': 'DB00331'
    },
    'lisinopril': {
        'name': 'Lisinopril',
        'smiles': 'NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O',
        'drugbank_id': 'DB00722'
    },
    'atorvastatin': {
        'name': 'Atorvastatin',
        'smiles': 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)c(-c2ccccc2)n1CC[C@H](O)C[C@H](O)CC(=O)O',
        'drugbank_id': 'DB01076'
    },
    'omeprazole': {
        'name': 'Omeprazole',
        'smiles': 'COC1=CC2=NC(CS(=O)C3=NC4=C(N3)C=CC=C4C)=NC2=CC1OC',
        'drugbank_id': 'DB00338'
    },
    'simvastatin': {
        'name': 'Simvastatin',
        'smiles': 'CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12',
        'drugbank_id': 'DB00641'
    }
}


def get_risk_level(risk_score: float) -> str:
    """Convert numeric risk score to categorical level."""
    if risk_score < 0.2:
        return 'low'
    elif risk_score < 0.5:
        return 'medium'
    elif risk_score < 0.8:
        return 'high'
    return 'critical'


def lookup_drug(drug_input: Dict) -> Dict:
    """Look up drug SMILES from name or ID."""
    name = drug_input.get('name', '').lower()
    smiles = drug_input.get('smiles', '')
    
    # If SMILES provided, use it
    if smiles:
        return {
            'name': drug_input.get('name', 'Unknown'),
            'smiles': smiles
        }
    
    # Try to find in sample data
    if name in SAMPLE_DRUGS:
        return SAMPLE_DRUGS[name]
    
    # Try partial match
    for key, drug in SAMPLE_DRUGS.items():
        if name in key or key in name:
            return drug
    
    # Return with dummy SMILES for demo
    return {
        'name': drug_input.get('name', 'Unknown'),
        'smiles': 'C'  # Methane as fallback
    }


# ============== API Views ==============

class DDIPredictionView(APIView):
    """
    POST /api/v1/predict/
    
    Predict drug-drug interaction between two drugs.
    Returns risk score, severity, affected systems, and mechanism hypothesis.
    """
    
    def post(self, request):
        serializer = DDIPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        start_time = time.time()
        
        # Look up drugs
        drug_a = lookup_drug(data['drug_a'])
        drug_b = lookup_drug(data['drug_b'])
        
        # Get prediction service
        service = get_ddi_service()
        
        # Make prediction
        prediction = service.predict(
            drug_a['smiles'],
            drug_b['smiles'],
            drug_a['name'],
            drug_b['name']
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Build response
        response_data = {
            'drug_a': prediction.drug_a,
            'drug_b': prediction.drug_b,
            'risk_score': prediction.risk_score,
            'risk_level': get_risk_level(prediction.risk_score),
            'severity': prediction.severity,
            'confidence': prediction.confidence,
            'mechanism_hypothesis': prediction.mechanism_hypothesis,
            'affected_systems': [
                {'system': sys, 'severity': prediction.risk_score, 'symptoms': []}
                for sys in prediction.affected_systems
            ],
            'inference_time_ms': inference_time
        }
        
        # Include explanation if requested
        if data.get('include_explanation', True):
            response_data['explanation'] = {
                'raw_probability': prediction.raw_probability,
                'calibrated_probability': prediction.calibrated_probability,
                'model_version': 'aegis-v1.0',
                # XAI visualization data would go here
            }
        
        # Log prediction
        try:
            PredictionLog.objects.create(
                drug_list=[drug_a['name'], drug_b['name']],
                risk_score=prediction.risk_score,
                calibrated_score=prediction.calibrated_probability,
                severity_prediction=prediction.severity,
                inference_time_ms=inference_time
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")
        
        return Response(response_data)


class PolypharmacyView(APIView):
    """
    POST /api/v1/polypharmacy/
    
    Analyze interactions between multiple drugs (N-way).
    Returns network graph data for visualization.
    """
    
    def post(self, request):
        serializer = PolypharmacyRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        start_time = time.time()
        
        # Look up all drugs
        drugs = [lookup_drug(d) for d in data['drugs']]
        
        # Get prediction service
        service = get_ddi_service()
        
        # Analyze polypharmacy
        result = service.predict_polypharmacy(drugs)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Build body map (aggregate affected systems)
        body_map = {}
        for interaction in result['interactions']:
            for system in interaction['affected_systems']:
                if system not in body_map:
                    body_map[system] = 0
                body_map[system] = max(body_map[system], interaction['risk_score'])
        
        response_data = {
            'drugs': result['drugs'],
            'interactions': result['interactions'],
            'total_interactions': result['total_interactions'],
            'max_risk_score': result['max_risk_score'],
            'overall_risk_level': get_risk_level(result['max_risk_score']),
            'hub_drug': result['hub_drug'],
            'hub_interaction_count': result['hub_interaction_count'],
            'body_map': body_map,
            'inference_time_ms': inference_time
        }
        
        # Log prediction
        try:
            PredictionLog.objects.create(
                drug_list=result['drugs'],
                risk_score=result['max_risk_score'],
                severity_prediction=get_risk_level(result['max_risk_score']),
                inference_time_ms=inference_time
            )
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")
        
        return Response(response_data)


class ChatView(APIView):
    """
    POST /api/v1/chat/
    
    GraphRAG-powered research assistant.
    Answers clinical questions using knowledge graph and literature.
    """
    
    def post(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        message = data['message']
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        # TODO: Implement GraphRAG with LangChain + Neo4j
        # For now, return a placeholder response
        
        # Simple keyword-based response for demo
        response_text = self._generate_demo_response(message)
        
        response_data = {
            'response': response_text,
            'sources': [
                {
                    'title': 'DrugBank Database',
                    'url': 'https://go.drugbank.com/',
                    'type': 'database'
                }
            ],
            'related_drugs': data.get('context_drugs', []),
            'session_id': session_id
        }
        
        return Response(response_data)
    
    def _generate_demo_response(self, message: str) -> str:
        """Generate a demo response based on keywords."""
        message_lower = message.lower()
        
        if 'warfarin' in message_lower and 'aspirin' in message_lower:
            return (
                "Warfarin and Aspirin have a **major interaction**. Both drugs affect blood clotting "
                "through different mechanisms:\n\n"
                "- **Warfarin** inhibits vitamin K-dependent clotting factors\n"
                "- **Aspirin** inhibits platelet aggregation via COX-1 inhibition\n\n"
                "Combined use significantly increases bleeding risk. If co-administration is necessary, "
                "close INR monitoring and dose adjustment are required."
            )
        
        if 'mechanism' in message_lower:
            return (
                "Drug-drug interactions occur through several mechanisms:\n\n"
                "1. **Pharmacokinetic**: Absorption, distribution, metabolism (CYP450), excretion\n"
                "2. **Pharmacodynamic**: Synergistic, additive, or antagonistic effects at receptors\n"
                "3. **Transporter-mediated**: P-glycoprotein and other drug transporters\n\n"
                "Our model analyzes molecular structure to predict interaction likelihood."
            )
        
        if 'cyp' in message_lower or 'enzyme' in message_lower:
            return (
                "CYP450 enzymes are responsible for metabolizing ~75% of drugs:\n\n"
                "- **CYP3A4**: Metabolizes the most drugs\n"
                "- **CYP2D6**: Important for many psychiatric and cardiovascular drugs\n"
                "- **CYP2C9**: Metabolizes warfarin and NSAIDs\n\n"
                "Inhibitors can increase drug levels; inducers can decrease effectiveness."
            )
        
        return (
            "I'm the Aegis Research Assistant. I can help you understand drug-drug interactions, "
            "mechanisms of action, and clinical recommendations.\n\n"
            "Try asking about specific drugs or interaction mechanisms!"
        )


class DrugSearchView(APIView):
    """
    GET /api/v1/drugs/search/?q=<query>
    
    Search for drugs by name for autocomplete.
    """
    
    def get(self, request):
        query = request.query_params.get('q', '').lower()
        
        if len(query) < 2:
            return Response({'results': []})
        
        # Search in sample data
        results = []
        for key, drug in SAMPLE_DRUGS.items():
            if query in drug['name'].lower():
                results.append({
                    'name': drug['name'],
                    'drugbank_id': drug['drugbank_id'],
                    'has_smiles': bool(drug.get('smiles'))
                })
        
        # Also search database
        try:
            db_drugs = Drug.objects.filter(name__icontains=query)[:10]
            for drug in db_drugs:
                results.append({
                    'name': drug.name,
                    'drugbank_id': drug.drugbank_id,
                    'has_smiles': bool(drug.smiles)
                })
        except Exception as e:
            logger.warning(f"Database search failed: {e}")
        
        return Response({'results': results[:20]})


class HealthCheckView(APIView):
    """
    GET /api/v1/health/
    
    System health check endpoint.
    """
    
    def get(self, request):
        health = {
            'status': 'healthy',
            'services': {
                'database': 'ok',
                'ai_model': 'ok',
                'neo4j': 'not_configured'  # Will be updated when Neo4j is set up
            },
            'version': 'aegis-v1.0.0'
        }
        
        # Check database
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except Exception as e:
            health['services']['database'] = f'error: {str(e)}'
            health['status'] = 'degraded'
        
        # Check AI model
        try:
            service = get_ddi_service()
            if service.model is None:
                health['services']['ai_model'] = 'not_loaded'
        except Exception as e:
            health['services']['ai_model'] = f'error: {str(e)}'
            health['status'] = 'degraded'
        
        return Response(health)


# ============== ViewSets for CRUD ==============

class DrugViewSet(viewsets.ModelViewSet):
    """
    CRUD operations for Drug model.
    """
    queryset = Drug.objects.all()
    serializer_class = DrugSerializer
    
    @action(detail=False, methods=['get'])
    def by_drugbank_id(self, request):
        """Get drug by DrugBank ID."""
        drugbank_id = request.query_params.get('id')
        if not drugbank_id:
            return Response({'error': 'id parameter required'}, status=400)
        
        try:
            drug = Drug.objects.get(drugbank_id=drugbank_id)
            serializer = self.get_serializer(drug)
            return Response(serializer.data)
        except Drug.DoesNotExist:
            return Response({'error': 'Drug not found'}, status=404)


class PredictionLogViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only access to prediction history.
    """
    queryset = PredictionLog.objects.all()
    serializer_class = PredictionLogSerializer
