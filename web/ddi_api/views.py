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
from .services.knowledge_graph import KnowledgeGraphService
from .services.pubmedbert_predictor import get_pubmedbert_predictor

logger = logging.getLogger(__name__)


def get_drug_from_knowledge_graph(name: str) -> Dict:
    """Look up drug from Neo4j Knowledge Graph."""
    try:
        kg = KnowledgeGraphService
        if kg.is_connected():
            results = kg.search_drugs(name, limit=1)
            if results:
                drug = results[0]
                return {
                    'name': drug.get('name', name),
                    'smiles': drug.get('smiles', ''),
                    'drugbank_id': drug.get('id', '')
                }
    except Exception as e:
        logger.warning(f"Knowledge graph lookup failed: {e}")
    return None


def get_known_interaction(drug1_id: str, drug2_id: str) -> Dict:
    """Check for known interaction in Knowledge Graph."""
    try:
        kg = KnowledgeGraphService
        if kg.is_connected():
            result = kg.check_interaction(drug1_id, drug2_id)
            if result:
                return result
    except Exception as e:
        logger.warning(f"Interaction lookup failed: {e}")
    return None


from .services.drug_service import get_drug_service

def lookup_drug(drug_input: Dict) -> Dict:
    """Look up drug SMILES from name or ID - first from Knowledge Graph, then DrugService."""
    name = drug_input.get('name', '').lower().strip()
    smiles = drug_input.get('smiles', '')
    drugbank_id = drug_input.get('drugbank_id', '')
    
    # 1. Try Knowledge Graph (Neo4j)
    if name and not drugbank_id:
        kg_result = get_drug_from_knowledge_graph(name)
        if kg_result:
            return {
                'name': kg_result.get('name', name),
                'smiles': kg_result.get('smiles', '') or smiles,
                'drugbank_id': kg_result.get('drugbank_id', '')
            }
    
    # 2. Try Local DrugService (JSON DB)
    drug_service = get_drug_service()
    
    # Try ID lookup first
    if drugbank_id:
        found = drug_service.get_drug(drugbank_id)
        if found:
            return {
                'name': found['name'],
                'smiles': found['smiles'] or smiles, # Use found SMILES if available
                'drugbank_id': found['drugbank_id']
            }
            
    # Try Name lookup
    if name:
        found = drug_service.get_drug(name)
        if found:
            return {
                'name': found['name'],
                'smiles': found['smiles'] or smiles,
                'drugbank_id': found['drugbank_id']
            }
            
    # 3. Fallback: Return what we have (even if just name)
    return {
        'name': drug_input.get('name', 'Unknown'),
        'smiles': smiles,
        'drugbank_id': drugbank_id
    }


# ============== API Views ==============

def get_risk_level(risk_score: float) -> str:
    """Map numerical risk score (0-1) to risk level string."""
    if risk_score >= 0.8: return 'critical'
    if risk_score >= 0.6: return 'high'
    if risk_score >= 0.3: return 'medium'
    return 'low'


class DDIPredictionView(APIView):
    """
    POST /api/v1/predict/
    
    Predict drug-drug interaction between two drugs.
    Returns risk score, severity, affected systems, and mechanism hypothesis.
    Uses Knowledge Graph for known interactions, AI model for novel pairs.
    """
    
    def post(self, request):
        try:
            return self._internal_post_logic(request)
        except Exception as e:
            import traceback
            logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': 'Internal server error during prediction', 'details': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _internal_post_logic(self, request):
        serializer = DDIPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        start_time = time.time()
        
        # Look up drugs
        drug_a = lookup_drug(data['drug_a'])
        drug_b = lookup_drug(data['drug_b'])
        
        # Check for known interaction in Knowledge Graph first
        known_interaction = None
        if drug_a.get('drugbank_id') and drug_b.get('drugbank_id'):
            known_interaction = get_known_interaction(
                drug_a['drugbank_id'], 
                drug_b['drugbank_id']
            )
        
        if known_interaction:
            # Use known interaction from Knowledge Graph
            severity = known_interaction.get('severity', 'moderate')
            mechanism = known_interaction.get('mechanism', 'Known interaction from clinical database.')
            
            severity_scores = {'minor': 0.3, 'moderate': 0.6, 'severe': 0.85}
            risk_score = severity_scores.get(severity, 0.5)
            
            response_data = {
                'drug_a': drug_a['name'],
                'drug_b': drug_b['name'],
                'risk_score': risk_score,
                'risk_level': get_risk_level(risk_score),
                'severity': severity,
                'confidence': 0.95,  # High confidence for known interactions
                'mechanism_hypothesis': mechanism,
                'affected_systems': [
                    {'system': 'See mechanism', 'severity': risk_score, 'symptoms': []}
                ],
                'inference_time_ms': (time.time() - start_time) * 1000,
                'source': 'knowledge_graph'
            }
        else:
            # Use PubMedBERT model for text-based DDI prediction
            # This model was trained on ~19,000 DDI Corpus sentences
            pubmedbert = get_pubmedbert_predictor()
            
            if pubmedbert.is_loaded:
                # Use PubMedBERT for prediction (primary method)
                prediction = pubmedbert.predict(drug_a['name'], drug_b['name'])
                
                # Map interaction type to affected systems
                affected_systems_map = {
                    'mechanism': ['liver', 'metabolic'],
                    'effect': ['cardiovascular', 'hematologic'],
                    'advise': ['general'],
                    'int': ['general'],
                    'no_interaction': []
                }
                affected = affected_systems_map.get(prediction.interaction_type, [])
                
                response_data = {
                    'drug_a': prediction.drug_a,
                    'drug_b': prediction.drug_b,
                    'risk_score': prediction.risk_score,
                    'risk_level': get_risk_level(prediction.risk_score),
                    'severity': prediction.severity,
                    'confidence': prediction.confidence,
                    'mechanism_hypothesis': pubmedbert.get_mechanism_description(
                        prediction.interaction_type, 
                        prediction.drug_a, 
                        prediction.drug_b,
                        prediction.confidence
                    ),
                    'affected_systems': [
                        {'system': sys, 'severity': prediction.risk_score, 'symptoms': []}
                        for sys in affected
                    ],
                    'inference_time_ms': (time.time() - start_time) * 1000,
                    'source': 'pubmedbert',
                    'interaction_type': prediction.interaction_type,
                    'all_probabilities': prediction.all_probabilities
                }
            else:
                # Fallback to molecular structure-based model
                logger.warning("PubMedBERT not loaded, falling back to GNN model")
                service = get_ddi_service()
                prediction = service.predict(
                    drug_a.get('smiles', ''),
                    drug_b.get('smiles', ''),
                    drug_a['name'],
                    drug_b['name']
                )
                
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
                    'inference_time_ms': (time.time() - start_time) * 1000,
                    'source': 'ai_model'
                }
        
        inference_time = response_data['inference_time_ms']
        
        # Include explanation if requested
        if data.get('include_explanation', True):
            response_data['explanation'] = {
                'model_version': 'aegis-v1.0',
                'data_source': response_data.get('source', 'ai_model'),
            }
        
        # Log prediction
        try:
            PredictionLog.objects.create(
                drug_list=[drug_a['name'], drug_b['name']],
                risk_score=response_data['risk_score'],
                calibrated_score=response_data['risk_score'],
                severity_prediction=response_data['severity'],
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
        from .services.graphrag_chatbot import get_chatbot
        
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        message = data['message']
        context_drugs = data.get('context_drugs', [])
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        try:
            # Use GraphRAG chatbot for intelligent responses
            chatbot = get_chatbot()
            result = chatbot.process_message(
                message=message,
                context_drugs=context_drugs,
                session_id=session_id
            )
            
            response_data = {
                'response': result.response,
                'sources': result.sources,
                'related_drugs': result.related_drugs,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"GraphRAG chatbot error: {e}")
            # Fallback to simple response
            response_data = {
                'response': self._fallback_response(message),
                'sources': [{'title': 'DrugBank Database', 'url': 'https://go.drugbank.com/', 'type': 'database'}],
                'related_drugs': context_drugs,
                'session_id': session_id
            }
        
        return Response(response_data)
    
    def _fallback_response(self, message: str) -> str:
        """Generate fallback response if GraphRAG fails."""
        return (
            "I'm the Aegis Research Assistant. I can help you understand drug-drug interactions, "
            "mechanisms of action, and clinical recommendations.\n\n"
            "Try asking about specific drugs like Warfarin, Aspirin, or Metformin!"
        )


class DrugSearchView(APIView):
    """
    GET /api/v1/drugs/search/?q=<query>
    
    Search for drugs by name for autocomplete.
    Uses Neo4j Knowledge Graph for comprehensive drug database.
    """
    
    def get(self, request):
        query = request.query_params.get('q', '').lower().strip()
        
        if len(query) < 2:
            return Response({'results': []})
        
        results = []
        
        # Search Knowledge Graph first
        try:
            kg = KnowledgeGraphService
            if kg.is_connected():
                kg_results = kg.search_drugs(query, limit=15)
                for drug in kg_results:
                    results.append({
                        'name': drug.get('name', ''),
                        'drugbank_id': drug.get('id', ''),
                        'smiles': drug.get('smiles', ''),  # Include SMILES for 3D visualization
                        'category': drug.get('category', ''),
                    })
        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")
        
        # If no KG results, search in local DrugService (JSON DB)
        if not results:
            try:
                drug_service = get_drug_service()
                service_results = drug_service.search_drugs(query, limit=10)
                
                for drug in service_results:
                    results.append({
                        'name': drug['name'],
                        'drugbank_id': drug.get('drugbank_id', ''),
                        'smiles': drug.get('smiles', ''),
                        'category': drug.get('category', ''),
                    })
            except Exception as e:
                logger.warning(f"DrugService search failed: {e}")
        
        # Also search database
        try:
            db_drugs = Drug.objects.filter(name__icontains=query)[:10]
            for drug in db_drugs:
                # Avoid duplicates
                if not any(r['drugbank_id'] == drug.drugbank_id for r in results):
                    results.append({
                        'name': drug.name,
                        'drugbank_id': drug.drugbank_id,
                        'smiles': drug.smiles or '',  # Include SMILES for 3D visualization
                        'category': drug.drug_class or '',
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
        # Check Neo4j
        neo4j_status = 'error'
        try:
            if KnowledgeGraphService.is_connected():
                neo4j_status = 'ok'
        except:
            pass
        
        health = {
            'status': 'healthy',
            'services': {
                'database': 'ok',
                'ai_model': 'ok',
                'neo4j': neo4j_status
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
