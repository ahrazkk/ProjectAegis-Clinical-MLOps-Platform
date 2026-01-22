"""
API Views for Project Aegis DDI API

These views provide the REST endpoints for:
1. DDI Prediction (2-drug interaction)
2. Polypharmacy Analysis (N-way interactions)
3. GraphRAG Chatbot (Research Assistant)
4. Drug Search and CRUD operations
"""

import re
import time
import uuid
import logging
from typing import Dict, List, Optional, Tuple

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


# ============== Drug Name Normalization ==============

def normalize_drug_name(name: str) -> Tuple[str, str]:
    """
    Normalize drug names to find matches for variants like:
    - Stereoisomers: (R)-Warfarin, (S)-Warfarin, (+)-Amphetamine, L-Dopa
    - Salts: Warfarin Sodium, Aspirin Calcium, Metoprolol Tartrate
    - Brand vs Generic: Tylenol -> Acetaminophen
    
    Returns: (normalized_name, original_base_name)
    """
    if not name:
        return '', ''
    
    original = name.strip()
    normalized = original.lower()
    
    # 1. Remove stereochemistry prefixes
    # Matches: (R)-, (S)-, (RS)-, (±)-, (+)-, (-)-, D-, L-, d-, l-, R-, S-
    stereochemistry_pattern = r'^[\(\[]?[RSrs±\+\-DdLl]+[\)\]]?-\s*'
    normalized = re.sub(stereochemistry_pattern, '', normalized)
    
    # 2. Remove salt/ester suffixes
    salt_suffixes = [
        ' sodium', ' potassium', ' calcium', ' magnesium', ' zinc',
        ' hydrochloride', ' hcl', ' hydrobromide', ' sulfate', ' sulphate',
        ' acetate', ' citrate', ' maleate', ' fumarate', ' tartrate',
        ' mesylate', ' besylate', ' phosphate', ' nitrate', ' chloride',
        ' bromide', ' iodide', ' succinate', ' gluconate', ' lactate',
        ' propionate', ' valerate', ' dipropionate', ' dihydrate',
        ' monohydrate', ' anhydrous', ' extended release', ' er', ' sr',
        ' xl', ' xr', ' cr', ' la', ' sustained release'
    ]
    for suffix in salt_suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
    
    # 3. Brand name to generic mappings (common ones)
    brand_to_generic = {
        'tylenol': 'acetaminophen', 'advil': 'ibuprofen', 'motrin': 'ibuprofen',
        'aleve': 'naproxen', 'lipitor': 'atorvastatin', 'zocor': 'simvastatin',
        'crestor': 'rosuvastatin', 'prilosec': 'omeprazole', 'nexium': 'esomeprazole',
        'zantac': 'ranitidine', 'pepcid': 'famotidine', 'coumadin': 'warfarin',
        'plavix': 'clopidogrel', 'xarelto': 'rivaroxaban', 'eliquis': 'apixaban',
        'pradaxa': 'dabigatran', 'lasix': 'furosemide', 'lopressor': 'metoprolol',
        'toprol': 'metoprolol', 'norvasc': 'amlodipine', 'prozac': 'fluoxetine',
        'zoloft': 'sertraline', 'lexapro': 'escitalopram', 'paxil': 'paroxetine',
        'wellbutrin': 'bupropion', 'effexor': 'venlafaxine', 'cymbalta': 'duloxetine',
        'xanax': 'alprazolam', 'ativan': 'lorazepam', 'valium': 'diazepam',
        'klonopin': 'clonazepam', 'ambien': 'zolpidem', 'lunesta': 'eszopiclone',
        'vicodin': 'hydrocodone', 'percocet': 'oxycodone', 'oxycontin': 'oxycodone',
        'ultram': 'tramadol', 'synthroid': 'levothyroxine', 'glucophage': 'metformin',
        'januvia': 'sitagliptin', 'lantus': 'insulin glargine', 'humira': 'adalimumab',
        'enbrel': 'etanercept', 'remicade': 'infliximab', 'prednisone': 'prednisone',
        'medrol': 'methylprednisolone', 'decadron': 'dexamethasone',
        'augmentin': 'amoxicillin', 'zithromax': 'azithromycin', 'cipro': 'ciprofloxacin',
        'levaquin': 'levofloxacin', 'diflucan': 'fluconazole', 'viagra': 'sildenafil',
        'cialis': 'tadalafil', 'singulair': 'montelukast', 'proventil': 'albuterol',
        'ventolin': 'albuterol', 'flovent': 'fluticasone', 'advair': 'fluticasone',
        'symbicort': 'budesonide', 'spiriva': 'tiotropium',
    }
    if normalized in brand_to_generic:
        normalized = brand_to_generic[normalized]
    
    # 4. Clean up remaining punctuation
    normalized = re.sub(r'[^\w\s-]', '', normalized).strip()
    
    return normalized, original


def search_drug_with_normalization(name: str) -> Optional[Dict]:
    """
    Search for a drug, trying normalized name if exact match fails.
    Returns the drug dict with 'matched_as' field showing what matched.
    """
    kg = KnowledgeGraphService
    if not kg.is_connected():
        return None
    
    normalized_name, original_name = normalize_drug_name(name)
    
    # Try exact match first
    results = kg.search_drugs(name, limit=1)
    if results:
        drug = results[0]
        drug['matched_as'] = 'exact'
        drug['original_query'] = name
        return drug
    
    # Try normalized name
    if normalized_name and normalized_name != name.lower():
        results = kg.search_drugs(normalized_name, limit=1)
        if results:
            drug = results[0]
            drug['matched_as'] = 'normalized'
            drug['original_query'] = name
            drug['normalized_to'] = normalized_name
            return drug
    
    return None


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


def get_known_interaction(drug1_id: str, drug2_id: str, drug1_name: str = None, drug2_name: str = None) -> Dict:
    """
    Check for known interaction in Knowledge Graph.
    Tries both exact match and normalized drug lookups.
    
    Args:
        drug1_id: DrugBank ID of first drug
        drug2_id: DrugBank ID of second drug
        drug1_name: Original drug name (for normalization fallback)
        drug2_name: Original drug name (for normalization fallback)
    """
    try:
        kg = KnowledgeGraphService
        if not kg.is_connected():
            return None
            
        # Try exact match first with provided IDs
        result = kg.check_interaction(drug1_id, drug2_id)
        if result:
            return result
        
        # If names provided, try to find interaction via normalized drug names
        # This handles cases like (R)-Warfarin -> Warfarin
        name1 = drug1_name or drug1_id
        name2 = drug2_name or drug2_id
        
        norm1, _ = normalize_drug_name(name1)
        norm2, _ = normalize_drug_name(name2)
        
        # Check if normalization changed anything
        needs_norm1 = norm1 != name1.lower()
        needs_norm2 = norm2 != name2.lower()
        
        if needs_norm1 or needs_norm2:
            # Look up the normalized drug's actual DrugBank ID
            norm_id1 = drug1_id
            norm_id2 = drug2_id
            
            if needs_norm1:
                # Search for the base drug (e.g., "warfarin" instead of "(R)-Warfarin")
                results = kg.search_drugs(norm1, limit=1)
                if results:
                    norm_id1 = results[0].get('id', drug1_id)
                    logger.info(f"Normalized {name1} -> {norm1} (ID: {norm_id1})")
            
            if needs_norm2:
                results = kg.search_drugs(norm2, limit=1)
                if results:
                    norm_id2 = results[0].get('id', drug2_id)
                    logger.info(f"Normalized {name2} -> {norm2} (ID: {norm_id2})")
            
            # Try interaction with normalized IDs
            if norm_id1 != drug1_id or norm_id2 != drug2_id:
                result = kg.check_interaction(norm_id1, norm_id2)
                if result:
                    result['matched_via_normalization'] = True
                    result['original_drugs'] = [name1, name2]
                    result['normalized_to'] = [norm1, norm2]
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
    
    # 1. Try Knowledge Graph (Neo4j) with normalization
    if name and not drugbank_id:
        # First try exact match, then normalized
        kg_result = search_drug_with_normalization(name)
        if kg_result:
            matched_name = kg_result.get('name', name)
            logger.info(f"Drug lookup: '{name}' -> matched as '{matched_name}' ({kg_result.get('matched_as', 'exact')})")
            return {
                'name': matched_name,
                'original_name': name,  # Keep original for reference
                'smiles': kg_result.get('smiles', '') or smiles,
                'drugbank_id': kg_result.get('id', ''),
                'therapeutic_class': kg_result.get('therapeutic_class', ''),
                'matched_via': kg_result.get('matched_as', 'exact')
            }
    
    # 2. Try Local DrugService (JSON DB)
    drug_service = get_drug_service()
    
    # Try ID lookup first
    if drugbank_id:
        found = drug_service.get_drug(drugbank_id)
        if found:
            return {
                'name': found['name'],
                'smiles': found['smiles'] or smiles,
                'drugbank_id': found['drugbank_id']
            }
            
    # Try Name lookup (with normalization)
    if name:
        # Try exact first
        found = drug_service.get_drug(name)
        if not found:
            # Try normalized
            normalized_name, _ = normalize_drug_name(name)
            if normalized_name != name.lower():
                found = drug_service.get_drug(normalized_name)
        
        if found:
            return {
                'name': found['name'],
                'smiles': found['smiles'] or smiles,
                'drugbank_id': found['drugbank_id']
            }
            
    # 3. Fallback: Return what we have (even if just name)
    # Also try to get the normalized base name
    normalized_name, _ = normalize_drug_name(drug_input.get('name', 'Unknown'))
    return {
        'name': drug_input.get('name', 'Unknown'),
        'normalized_name': normalized_name,
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
        
        # Get original input names for normalization fallback
        original_name_a = data['drug_a'].get('name', drug_a.get('name', ''))
        original_name_b = data['drug_b'].get('name', drug_b.get('name', ''))
        
        # Check for known interaction in Knowledge Graph first
        # Pass both IDs and original names so we can try normalized lookups
        known_interaction = None
        if drug_a.get('drugbank_id') and drug_b.get('drugbank_id'):
            known_interaction = get_known_interaction(
                drug_a['drugbank_id'], 
                drug_b['drugbank_id'],
                original_name_a,
                original_name_b
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
                    'all_probabilities': prediction.all_probabilities,
                    # Context sentence information for transparency
                    'context_sentence': prediction.context_sentence,
                    'context_source': prediction.context_source,
                    'template_category': prediction.template_category
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
    
    Results are sorted: drugs WITH SMILES appear first, then drugs without.
    Each result includes has_smiles flag for UI color-coding.
    """
    
    def get(self, request):
        query = request.query_params.get('q', '').lower().strip()
        
        if len(query) < 2:
            return Response({'results': []})
        
        results = []
        seen_names = set()  # Track unique drug names (lowercase)
        
        # Search Knowledge Graph first
        try:
            kg = KnowledgeGraphService
            if kg.is_connected():
                kg_results = kg.search_drugs(query, limit=30)  # Get more to filter
                for drug in kg_results:
                    name_lower = drug.get('name', '').lower()
                    if name_lower and name_lower not in seen_names:
                        smiles = drug.get('smiles', '') or ''
                        results.append({
                            'name': drug.get('name', ''),
                            'drugbank_id': drug.get('id', ''),
                            'smiles': smiles,
                            'category': drug.get('category', ''),
                            'has_smiles': bool(smiles and len(smiles) > 5),  # Valid SMILES
                        })
                        seen_names.add(name_lower)
        except Exception as e:
            logger.warning(f"Knowledge graph search failed: {e}")
        
        # If no KG results, search in local DrugService (JSON DB)
        if not results:
            try:
                drug_service = get_drug_service()
                service_results = drug_service.search_drugs(query, limit=20)
                
                for drug in service_results:
                    name_lower = drug['name'].lower()
                    if name_lower not in seen_names:
                        smiles = drug.get('smiles', '') or ''
                        results.append({
                            'name': drug['name'],
                            'drugbank_id': drug.get('drugbank_id', ''),
                            'smiles': smiles,
                            'category': drug.get('category', ''),
                            'has_smiles': bool(smiles and len(smiles) > 5),
                        })
                        seen_names.add(name_lower)
            except Exception as e:
                logger.warning(f"DrugService search failed: {e}")
        
        # Also search database
        try:
            db_drugs = Drug.objects.filter(name__icontains=query)[:20]
            for drug in db_drugs:
                name_lower = drug.name.lower()
                # Avoid duplicates
                if name_lower not in seen_names:
                    smiles = drug.smiles or ''
                    results.append({
                        'name': drug.name,
                        'drugbank_id': drug.drugbank_id,
                        'smiles': smiles,
                        'category': drug.drug_class or '',
                        'has_smiles': bool(smiles and len(smiles) > 5),
                    })
                    seen_names.add(name_lower)
        except Exception as e:
            logger.warning(f"Database search failed: {e}")
        
        # SORT: Drugs with SMILES first, then alphabetically
        results.sort(key=lambda x: (
            0 if x['has_smiles'] else 1,  # SMILES first
            x['name'].lower()  # Then alphabetically
        ))
        
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


# ============== Enhanced Drug Info Endpoints ==============

class EnhancedDrugInfoView(APIView):
    """
    GET /api/v1/drug-info/?name=<drug_name>
    
    Returns comprehensive drug information including:
    - Side effects from SIDER database
    - Real-world adverse event statistics from OpenFDA FAERS
    - Interaction count from Knowledge Graph
    """
    
    def get(self, request):
        drug_name = request.query_params.get('name', '').strip()
        if not drug_name:
            return Response(
                {'error': 'name parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        include_faers = request.query_params.get('faers', 'true').lower() == 'true'
        
        try:
            from .services.enhanced_drug_service import get_enhanced_drug_service
            service = get_enhanced_drug_service()
            info = service.get_drug_info(drug_name, include_faers=include_faers)
            return Response(info.to_dict())
        except Exception as e:
            logger.error(f"Enhanced drug info failed: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class EnhancedInteractionInfoView(APIView):
    """
    GET /api/v1/interaction-info/?drug1=<name>&drug2=<name>
    
    Returns comprehensive interaction information including:
    - Polypharmacy side effects from TWOSIDES
    - Real-world co-reported adverse events from OpenFDA FAERS
    - Evidence sources (DDI Corpus, Knowledge Graph, etc.)
    """
    
    def get(self, request):
        drug1 = request.query_params.get('drug1', '').strip()
        drug2 = request.query_params.get('drug2', '').strip()
        
        if not drug1 or not drug2:
            return Response(
                {'error': 'drug1 and drug2 parameters required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        include_faers = request.query_params.get('faers', 'true').lower() == 'true'
        
        try:
            from .services.enhanced_drug_service import get_enhanced_drug_service
            service = get_enhanced_drug_service()
            info = service.get_interaction_info(drug1, drug2, include_faers=include_faers)
            return Response(info.to_dict())
        except Exception as e:
            logger.error(f"Enhanced interaction info failed: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RealWorldEvidenceView(APIView):
    """
    GET /api/v1/real-world-evidence/?drug1=<name>&drug2=<name>
    
    Returns real-world evidence from FDA Adverse Event Reporting System.
    Shows how many adverse event reports mention these drugs together.
    """
    
    def get(self, request):
        drug1 = request.query_params.get('drug1', '').strip()
        drug2 = request.query_params.get('drug2', '')
        
        if not drug1:
            return Response(
                {'error': 'drug1 parameter required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Dynamic import to avoid startup issues
            from .management.commands.import_external_data import get_openfda_importer
            openfda = get_openfda_importer()
            
            if drug2:
                # Pair query
                result = openfda.get_pair_reports(drug1, drug2)
            else:
                # Single drug query
                result = openfda.get_adverse_events(drug1, limit=15)
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Real-world evidence query failed: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
