"""
Ensemble DDI Prediction Service

Combines multiple prediction sources for more robust DDI predictions:
1. PubMedBERT (NLP-based)
2. GNN/ChemicalX (Structure-based)
3. CYP450 Database (Pharmacokinetic)
4. OpenFDA FAERS (Real-world evidence)

This ensemble approach provides:
- Higher accuracy through model consensus
- Better explainability with multiple perspectives
- Fallback options when one model fails
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PredictionSource(str, Enum):
    """Available prediction sources."""
    PUBMEDBERT = "pubmedbert"
    GNN = "gnn"
    CYP450 = "cyp450"
    OPENFDA = "openfda"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    ENSEMBLE = "ensemble"


@dataclass
class SourcePrediction:
    """Prediction from a single source."""
    source: PredictionSource
    interaction_type: str
    severity: str
    confidence: float
    risk_score: float
    mechanism: str
    available: bool = True


@dataclass
class EnsemblePrediction:
    """Combined prediction from multiple sources."""
    drug1: str
    drug2: str
    final_interaction_type: str
    final_severity: str
    final_confidence: float
    final_risk_score: float
    consensus_level: str  # high, medium, low
    source_predictions: List[SourcePrediction]
    combined_mechanism: str
    recommendations: List[str]
    evidence_summary: Dict[str, Any]


class EnsembleDDIPredictor:
    """
    Ensemble DDI Predictor combining multiple models and data sources.
    
    Architecture:
    1. Query each available source independently
    2. Weight predictions based on source reliability and availability
    3. Combine using voting/averaging with confidence weighting
    4. Generate comprehensive explanation from all sources
    """
    
    # Source weights (based on typical accuracy/reliability)
    SOURCE_WEIGHTS = {
        PredictionSource.CYP450: 0.30,        # High reliability for PK interactions
        PredictionSource.PUBMEDBERT: 0.25,    # Good for literature-based evidence
        PredictionSource.KNOWLEDGE_GRAPH: 0.20,  # Curated database
        PredictionSource.GNN: 0.15,           # Structure-based predictions
        PredictionSource.OPENFDA: 0.10,       # Real-world adverse events
    }
    
    # Severity ordering for consensus
    SEVERITY_ORDER = {
        'none': 0,
        'minor': 1,
        'moderate': 2,
        'major': 3,
        'severe': 4
    }
    
    def __init__(self):
        """Initialize ensemble predictor with lazy-loaded sources."""
        self._pubmedbert = None
        self._gnn = None
        self._cyp450 = None
        self._openfda = None
        self._kg = None
    
    def _get_pubmedbert(self):
        """Lazy load PubMedBERT predictor."""
        if self._pubmedbert is None:
            try:
                from .pubmedbert_predictor import get_pubmedbert_predictor
                self._pubmedbert = get_pubmedbert_predictor()
            except ImportError:
                logger.warning("PubMedBERT predictor not available")
        return self._pubmedbert
    
    def _get_gnn(self):
        """Lazy load GNN predictor."""
        if self._gnn is None:
            try:
                from .gnn_predictor import get_gnn_predictor
                self._gnn = get_gnn_predictor()
            except ImportError:
                logger.warning("GNN predictor not available")
        return self._gnn
    
    def _get_cyp450(self):
        """Lazy load CYP450 database."""
        if self._cyp450 is None:
            try:
                from .cyp450_database import get_cyp450_database
                self._cyp450 = get_cyp450_database()
            except ImportError:
                logger.warning("CYP450 database not available")
        return self._cyp450
    
    def _get_openfda(self):
        """Lazy load OpenFDA service."""
        if self._openfda is None:
            try:
                from .openfda_service import get_openfda_service
                self._openfda = get_openfda_service()
            except ImportError:
                logger.warning("OpenFDA service not available")
        return self._openfda
    
    def _get_kg(self):
        """Lazy load knowledge graph service."""
        if self._kg is None:
            try:
                from .knowledge_graph import KnowledgeGraphService
                if KnowledgeGraphService.is_connected():
                    self._kg = KnowledgeGraphService
            except ImportError:
                logger.warning("Knowledge graph not available")
        return self._kg
    
    def _get_pubmedbert_prediction(
        self, 
        drug1: str, 
        drug2: str
    ) -> Optional[SourcePrediction]:
        """Get prediction from PubMedBERT."""
        predictor = self._get_pubmedbert()
        
        if predictor is None or not predictor.is_loaded:
            return SourcePrediction(
                source=PredictionSource.PUBMEDBERT,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism='PubMedBERT model not available',
                available=False
            )
        
        try:
            pred = predictor.predict(drug1, drug2)
            return SourcePrediction(
                source=PredictionSource.PUBMEDBERT,
                interaction_type=pred.interaction_type,
                severity=pred.severity,
                confidence=pred.confidence,
                risk_score=pred.risk_score,
                mechanism=predictor.get_mechanism_description(
                    pred.interaction_type, drug1, drug2, pred.confidence
                ),
                available=True
            )
        except Exception as e:
            logger.error(f"PubMedBERT prediction failed: {e}")
            return SourcePrediction(
                source=PredictionSource.PUBMEDBERT,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism=f'Prediction error: {str(e)}',
                available=False
            )
    
    def _get_gnn_prediction(
        self, 
        drug1: str, 
        drug2: str,
        smiles1: Optional[str] = None,
        smiles2: Optional[str] = None
    ) -> Optional[SourcePrediction]:
        """Get prediction from GNN model."""
        predictor = self._get_gnn()
        
        if predictor is None:
            return SourcePrediction(
                source=PredictionSource.GNN,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism='GNN predictor not available',
                available=False
            )
        
        try:
            pred = predictor.predict(drug1, drug2, smiles1, smiles2)
            return SourcePrediction(
                source=PredictionSource.GNN,
                interaction_type=pred.interaction_type,
                severity=pred.severity,
                confidence=pred.confidence,
                risk_score=pred.interaction_probability,
                mechanism=pred.mechanism_hypothesis,
                available=True
            )
        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return SourcePrediction(
                source=PredictionSource.GNN,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism=f'Prediction error: {str(e)}',
                available=False
            )
    
    def _get_cyp450_prediction(
        self, 
        drug1: str, 
        drug2: str
    ) -> Optional[SourcePrediction]:
        """Get prediction from CYP450 database."""
        db = self._get_cyp450()
        
        if db is None:
            return SourcePrediction(
                source=PredictionSource.CYP450,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism='CYP450 database not available',
                available=False
            )
        
        try:
            interactions = db.check_cyp_interaction(drug1, drug2)
            
            if not interactions:
                return SourcePrediction(
                    source=PredictionSource.CYP450,
                    interaction_type='no_interaction',
                    severity='none',
                    confidence=0.7,
                    risk_score=0.0,
                    mechanism=f'No CYP450-mediated interaction found between {drug1} and {drug2}',
                    available=True
                )
            
            # Take most severe interaction
            most_severe = max(interactions, key=lambda x: self.SEVERITY_ORDER.get(x.severity, 0))
            
            return SourcePrediction(
                source=PredictionSource.CYP450,
                interaction_type='mechanism',
                severity=most_severe.severity,
                confidence=0.9,  # High confidence for database-based
                risk_score=self.SEVERITY_ORDER.get(most_severe.severity, 2) / 4,
                mechanism=most_severe.mechanism,
                available=True
            )
        except Exception as e:
            logger.error(f"CYP450 lookup failed: {e}")
            return SourcePrediction(
                source=PredictionSource.CYP450,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism=f'Lookup error: {str(e)}',
                available=False
            )
    
    def _get_openfda_prediction(
        self, 
        drug1: str, 
        drug2: str
    ) -> Optional[SourcePrediction]:
        """Get prediction from OpenFDA FAERS data."""
        service = self._get_openfda()
        
        if service is None:
            return SourcePrediction(
                source=PredictionSource.OPENFDA,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism='OpenFDA service not available',
                available=False
            )
        
        try:
            result = service.get_interaction_severity_from_faers(drug1, drug2)
            
            if result is None or result.get('total_reports', 0) == 0:
                return SourcePrediction(
                    source=PredictionSource.OPENFDA,
                    interaction_type='no_interaction',
                    severity='none',
                    confidence=0.3,  # Low confidence for absence of evidence
                    risk_score=0.0,
                    mechanism=f'No adverse event reports found for {drug1} + {drug2} combination',
                    available=True
                )
            
            severity = result.get('estimated_severity', 'moderate')
            top_reactions = result.get('top_reactions', [])
            
            mechanism = f"FDA FAERS data: {result['total_reports']} reports, {result['serious_reports']} serious. "
            if top_reactions:
                top_3 = [r[0] for r in top_reactions[:3]]
                mechanism += f"Common reactions: {', '.join(top_3)}"
            
            return SourcePrediction(
                source=PredictionSource.OPENFDA,
                interaction_type='effect',
                severity=severity,
                confidence=min(0.8, 0.3 + (result['total_reports'] / 100)),  # More reports = higher confidence
                risk_score=result.get('serious_ratio', 0.5),
                mechanism=mechanism,
                available=True
            )
        except Exception as e:
            logger.error(f"OpenFDA lookup failed: {e}")
            return SourcePrediction(
                source=PredictionSource.OPENFDA,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism=f'Lookup error: {str(e)}',
                available=False
            )
    
    def _get_kg_prediction(
        self, 
        drug1: str, 
        drug2: str
    ) -> Optional[SourcePrediction]:
        """Get prediction from knowledge graph."""
        kg = self._get_kg()
        
        if kg is None:
            return SourcePrediction(
                source=PredictionSource.KNOWLEDGE_GRAPH,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism='Knowledge graph not available',
                available=False
            )
        
        try:
            # Need to look up drug IDs first
            drug1_results = kg.get_drug(name=drug1)
            drug2_results = kg.get_drug(name=drug2)
            
            if not drug1_results or not drug2_results:
                return SourcePrediction(
                    source=PredictionSource.KNOWLEDGE_GRAPH,
                    interaction_type='unknown',
                    severity='unknown',
                    confidence=0.0,
                    risk_score=0.0,
                    mechanism=f'Drug not found in knowledge graph',
                    available=False
                )
            
            drug1_id = drug1_results[0].get('d', {}).get('drugbank_id')
            drug2_id = drug2_results[0].get('d', {}).get('drugbank_id')
            
            if not drug1_id or not drug2_id:
                return SourcePrediction(
                    source=PredictionSource.KNOWLEDGE_GRAPH,
                    interaction_type='unknown',
                    severity='unknown',
                    confidence=0.0,
                    risk_score=0.0,
                    mechanism='Drug IDs not available',
                    available=False
                )
            
            interaction = kg.check_interaction(drug1_id, drug2_id)
            
            if interaction is None:
                return SourcePrediction(
                    source=PredictionSource.KNOWLEDGE_GRAPH,
                    interaction_type='no_interaction',
                    severity='none',
                    confidence=0.5,
                    risk_score=0.0,
                    mechanism=f'No interaction found in knowledge graph',
                    available=True
                )
            
            return SourcePrediction(
                source=PredictionSource.KNOWLEDGE_GRAPH,
                interaction_type='mechanism',
                severity=interaction.get('severity', 'moderate'),
                confidence=0.95,  # Very high confidence for curated data
                risk_score=self.SEVERITY_ORDER.get(interaction.get('severity', 'moderate'), 2) / 4,
                mechanism=interaction.get('mechanism') or interaction.get('description', 'Known interaction'),
                available=True
            )
        except Exception as e:
            logger.error(f"Knowledge graph lookup failed: {e}")
            return SourcePrediction(
                source=PredictionSource.KNOWLEDGE_GRAPH,
                interaction_type='unknown',
                severity='unknown',
                confidence=0.0,
                risk_score=0.0,
                mechanism=f'Lookup error: {str(e)}',
                available=False
            )
    
    def _combine_predictions(
        self,
        predictions: List[SourcePrediction],
        drug1: str,
        drug2: str
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple sources using weighted voting.
        """
        # Filter to available predictions
        available = [p for p in predictions if p.available and p.interaction_type != 'unknown']
        
        if not available:
            return EnsemblePrediction(
                drug1=drug1,
                drug2=drug2,
                final_interaction_type='unknown',
                final_severity='unknown',
                final_confidence=0.0,
                final_risk_score=0.0,
                consensus_level='none',
                source_predictions=predictions,
                combined_mechanism='No prediction sources available',
                recommendations=['Unable to assess interaction - verify with clinical pharmacist'],
                evidence_summary={'available_sources': 0}
            )
        
        # Calculate weighted severity
        severity_scores = []
        total_weight = 0
        
        for pred in available:
            weight = self.SOURCE_WEIGHTS.get(pred.source, 0.1) * pred.confidence
            severity_num = self.SEVERITY_ORDER.get(pred.severity, 2)
            severity_scores.append((severity_num, weight))
            total_weight += weight
        
        if total_weight > 0:
            weighted_severity = sum(s * w for s, w in severity_scores) / total_weight
        else:
            weighted_severity = 2  # Default moderate
        
        # Map back to severity string
        severity_map = {0: 'none', 1: 'minor', 2: 'moderate', 3: 'major', 4: 'severe'}
        final_severity = severity_map.get(round(weighted_severity), 'moderate')
        
        # Determine interaction type (most common among high-confidence predictions)
        type_counts: Dict[str, float] = {}
        for pred in available:
            if pred.interaction_type != 'no_interaction':
                type_counts[pred.interaction_type] = type_counts.get(pred.interaction_type, 0) + pred.confidence
        
        if type_counts:
            final_type = max(type_counts, key=lambda x: type_counts[x])
        else:
            final_type = 'no_interaction'
        
        # Calculate consensus level
        no_interaction_count = sum(1 for p in available if p.interaction_type == 'no_interaction')
        interaction_count = len(available) - no_interaction_count
        
        if len(available) >= 3:
            if interaction_count >= len(available) - 1 or no_interaction_count >= len(available) - 1:
                consensus_level = 'high'
            elif interaction_count >= len(available) / 2 or no_interaction_count >= len(available) / 2:
                consensus_level = 'medium'
            else:
                consensus_level = 'low'
        else:
            consensus_level = 'low'
        
        # Calculate final confidence and risk
        final_confidence = sum(p.confidence for p in available) / len(available)
        final_risk = sum(p.risk_score * self.SOURCE_WEIGHTS.get(p.source, 0.1) for p in available) / sum(
            self.SOURCE_WEIGHTS.get(p.source, 0.1) for p in available
        )
        
        # Combine mechanisms
        mechanisms = [p.mechanism for p in available if p.mechanism and p.available]
        combined_mechanism = ' | '.join(mechanisms[:3])  # Top 3 mechanisms
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_type, final_severity, final_confidence, consensus_level, available
        )
        
        # Evidence summary
        evidence_summary = {
            'available_sources': len(available),
            'total_sources_checked': len(predictions),
            'consensus_level': consensus_level,
            'sources_agree_interaction': interaction_count,
            'sources_say_no_interaction': no_interaction_count,
            'weighted_severity_score': weighted_severity
        }
        
        return EnsemblePrediction(
            drug1=drug1,
            drug2=drug2,
            final_interaction_type=final_type,
            final_severity=final_severity,
            final_confidence=final_confidence,
            final_risk_score=final_risk,
            consensus_level=consensus_level,
            source_predictions=predictions,
            combined_mechanism=combined_mechanism,
            recommendations=recommendations,
            evidence_summary=evidence_summary
        )
    
    def _generate_recommendations(
        self,
        interaction_type: str,
        severity: str,
        confidence: float,
        consensus: str,
        predictions: List[SourcePrediction]
    ) -> List[str]:
        """Generate clinical recommendations based on ensemble prediction."""
        recommendations = []
        
        if interaction_type == 'no_interaction':
            if confidence > 0.7 and consensus == 'high':
                recommendations.append("✅ Low interaction risk - combination appears safe based on available evidence")
            else:
                recommendations.append("⚠️ Limited evidence - monitor for unexpected effects")
        elif severity == 'severe':
            recommendations.append("🚫 AVOID combination if possible - severe interaction risk")
            recommendations.append("📋 If combination necessary, implement intensive monitoring")
        elif severity == 'major':
            recommendations.append("⚠️ Major interaction - consider alternative therapy")
            recommendations.append("📊 If used together, monitor closely for adverse effects")
        elif severity == 'moderate':
            recommendations.append("ℹ️ Moderate interaction - use with caution")
            recommendations.append("📝 Document clinical rationale if using combination")
        else:
            recommendations.append("ℹ️ Minor/minimal interaction risk")
        
        if consensus == 'low':
            recommendations.append("⚠️ Low consensus between prediction sources - verify with clinical pharmacist")
        
        return recommendations
    
    def predict(
        self,
        drug1: str,
        drug2: str,
        smiles1: Optional[str] = None,
        smiles2: Optional[str] = None,
        use_all_sources: bool = True
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction for a drug pair.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            smiles1: Optional SMILES for first drug
            smiles2: Optional SMILES for second drug
            use_all_sources: Whether to query all available sources
            
        Returns:
            EnsemblePrediction with combined results
        """
        predictions = []
        
        # Always try core predictors
        predictions.append(self._get_pubmedbert_prediction(drug1, drug2))
        predictions.append(self._get_cyp450_prediction(drug1, drug2))
        
        if use_all_sources:
            predictions.append(self._get_gnn_prediction(drug1, drug2, smiles1, smiles2))
            predictions.append(self._get_kg_prediction(drug1, drug2))
            # Note: OpenFDA API can be slow (rate-limited) so it's disabled by default.
            # Uncomment the line below to enable real-time FAERS data (may add latency):
            # predictions.append(self._get_openfda_prediction(drug1, drug2))
        
        return self._combine_predictions(predictions, drug1, drug2)
    
    def get_available_sources(self) -> Dict[str, bool]:
        """Check which prediction sources are available."""
        return {
            'pubmedbert': self._get_pubmedbert() is not None and self._get_pubmedbert().is_loaded,
            'gnn': self._get_gnn() is not None,
            'cyp450': self._get_cyp450() is not None,
            'openfda': self._get_openfda() is not None,
            'knowledge_graph': self._get_kg() is not None
        }


# Singleton instance
_ensemble_predictor: Optional[EnsembleDDIPredictor] = None


def get_ensemble_predictor() -> EnsembleDDIPredictor:
    """Get or create the ensemble predictor singleton."""
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsembleDDIPredictor()
    return _ensemble_predictor
