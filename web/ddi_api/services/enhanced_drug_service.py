"""
Enhanced Drug Information Service

Provides enriched drug data from multiple sources:
- Side effects from SIDER
- Polypharmacy effects from TWOSIDES  
- Real-world evidence from OpenFDA FAERS
- Drug properties from Knowledge Graph
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


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
    
    # 3. Brand name to generic mappings
    brand_to_generic = {
        'tylenol': 'acetaminophen', 'advil': 'ibuprofen', 'motrin': 'ibuprofen',
        'aleve': 'naproxen', 'lipitor': 'atorvastatin', 'zocor': 'simvastatin',
        'crestor': 'rosuvastatin', 'prilosec': 'omeprazole', 'nexium': 'esomeprazole',
        'coumadin': 'warfarin', 'plavix': 'clopidogrel', 'xarelto': 'rivaroxaban',
    }
    if normalized in brand_to_generic:
        normalized = brand_to_generic[normalized]
    
    # 4. Clean up remaining punctuation
    normalized = re.sub(r'[^\w\s-]', '', normalized).strip()
    
    return normalized, original


@dataclass
class SideEffect:
    """A drug side effect."""
    name: str
    frequency: Optional[str] = None  # common, uncommon, rare
    severity: Optional[str] = None   # mild, moderate, severe
    source: str = 'sider'


@dataclass
class AdverseEventStats:
    """Real-world adverse event statistics."""
    total_reports: int = 0
    top_reactions: List[Dict] = field(default_factory=list)
    source: str = 'openfda'


@dataclass
class PolypharmacyEffect:
    """A polypharmacy (multi-drug) side effect."""
    effect: str
    drug1: str
    drug2: str
    score: float
    source: str = 'twosides'


@dataclass
class EnhancedDrugInfo:
    """Comprehensive drug information."""
    name: str
    drugbank_id: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    
    # Side effects from SIDER
    side_effects: List[SideEffect] = field(default_factory=list)
    
    # Real-world evidence from OpenFDA
    adverse_events: Optional[AdverseEventStats] = None
    
    # Known interactions
    interaction_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'drugbank_id': self.drugbank_id,
            'description': self.description,
            'category': self.category,
            'side_effects': [asdict(se) for se in self.side_effects],
            'adverse_events': asdict(self.adverse_events) if self.adverse_events else None,
            'interaction_count': self.interaction_count
        }


@dataclass  
class EnhancedInteractionInfo:
    """Comprehensive interaction information."""
    drug1: str
    drug2: str
    
    # Prediction results
    risk_score: float = 0.0
    severity: str = 'unknown'
    mechanism: str = ''
    
    # Polypharmacy side effects from TWOSIDES
    polypharmacy_effects: List[PolypharmacyEffect] = field(default_factory=list)
    
    # Real-world co-reported events from OpenFDA
    faers_reports: int = 0
    faers_reactions: List[Dict] = field(default_factory=list)
    
    # Clinical evidence
    evidence_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'drug1': self.drug1,
            'drug2': self.drug2,
            'risk_score': self.risk_score,
            'severity': self.severity,
            'mechanism': self.mechanism,
            'polypharmacy_effects': [asdict(pe) for pe in self.polypharmacy_effects],
            'faers_reports': self.faers_reports,
            'faers_reactions': self.faers_reactions,
            'evidence_sources': self.evidence_sources
        }


class EnhancedDrugService:
    """Service for retrieving enhanced drug information."""
    
    def __init__(self):
        self._sider = None
        self._twosides = None
        self._openfda = None
        self._initialized = False
    
    def _lazy_init(self):
        """Lazy initialization of importers."""
        if self._initialized:
            return
        
        try:
            from ddi_api.management.commands.import_external_data import (
                get_sider_importer, 
                get_twosides_importer,
                get_openfda_importer
            )
            self._sider = get_sider_importer()
            self._twosides = get_twosides_importer()
            self._openfda = get_openfda_importer()
            logger.info(f"Initialized external data: SIDER={len(self._sider.side_effects) if self._sider else 0} drugs")
        except Exception as e:
            logger.warning(f"Failed to initialize external importers: {e}")
            import traceback
            traceback.print_exc()
        
        self._initialized = True
    
    def get_drug_info(self, drug_name: str, include_faers: bool = True) -> EnhancedDrugInfo:
        """
        Get comprehensive drug information.
        
        Args:
            drug_name: Name of the drug
            include_faers: Whether to query OpenFDA (slower but real-time)
            
        Returns:
            EnhancedDrugInfo with all available data
        """
        self._lazy_init()
        
        info = EnhancedDrugInfo(name=drug_name)
        
        # Get side effects from SIDER
        if self._sider:
            try:
                effects = self._sider.get_side_effects(drug_name, limit=15)
                info.side_effects = [
                    SideEffect(name=effect, source='sider')
                    for effect in effects
                ]
            except Exception as e:
                logger.debug(f"SIDER lookup failed for {drug_name}: {e}")
        
        # Get real-world adverse events from OpenFDA
        if include_faers and self._openfda:
            try:
                faers_data = self._openfda.get_adverse_events(drug_name, limit=10)
                if 'error' not in faers_data:
                    info.adverse_events = AdverseEventStats(
                        total_reports=faers_data.get('total_reports', 0),
                        top_reactions=faers_data.get('top_reactions', []),
                        source='openfda'
                    )
            except Exception as e:
                logger.debug(f"OpenFDA lookup failed for {drug_name}: {e}")
        
        # Get interaction count from Knowledge Graph
        try:
            from .knowledge_graph import KnowledgeGraphService as KG
            if KG.is_connected():
                results = KG.run_query('''
                    MATCH (d:Drug)-[r:INTERACTS_WITH]-()
                    WHERE toLower(d.name) = toLower($name)
                    RETURN count(r) as count
                ''', {'name': drug_name})
                if results:
                    info.interaction_count = results[0].get('count', 0)
        except Exception as e:
            logger.debug(f"KG lookup failed for {drug_name}: {e}")
        
        return info
    
    def get_interaction_info(self, drug1: str, drug2: str, 
                              include_faers: bool = True) -> EnhancedInteractionInfo:
        """
        Get comprehensive interaction information.
        Uses drug name normalization to match variants like (R)-Warfarin -> Warfarin.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            include_faers: Whether to query OpenFDA
            
        Returns:
            EnhancedInteractionInfo with all available data
        """
        self._lazy_init()
        
        # Normalize drug names for better matching
        norm1, orig1 = normalize_drug_name(drug1)
        norm2, orig2 = normalize_drug_name(drug2)
        
        info = EnhancedInteractionInfo(drug1=drug1, drug2=drug2)
        evidence = []
        
        # Track if we matched via normalization
        matched_normalized = False
        
        # Get polypharmacy effects from TWOSIDES - try both normalized and original
        if self._twosides:
            try:
                # Try original names first
                effects = self._twosides.get_pair_effects(drug1, drug2, limit=10)
                # If no results, try normalized
                if not effects and (norm1 != drug1.lower() or norm2 != drug2.lower()):
                    effects = self._twosides.get_pair_effects(norm1, norm2, limit=10)
                    if effects:
                        matched_normalized = True
                        
                info.polypharmacy_effects = [
                    PolypharmacyEffect(
                        effect=e['side_effect'],
                        drug1=drug1,
                        drug2=drug2,
                        score=e['score'],
                        source='twosides'
                    )
                    for e in effects
                ]
                if effects:
                    evidence.append('twosides')
            except Exception as e:
                logger.debug(f"TWOSIDES lookup failed: {e}")
        
        # Get real-world co-reported events from OpenFDA
        if include_faers and self._openfda:
            try:
                # Try normalized names for FAERS (often better matches)
                faers_data = self._openfda.get_pair_reports(norm1, norm2)
                if 'error' not in faers_data:
                    info.faers_reports = faers_data.get('total_reports', 0)
                    info.faers_reactions = faers_data.get('top_reactions', [])
                    if info.faers_reports > 0:
                        evidence.append('openfda_faers')
            except Exception as e:
                logger.debug(f"OpenFDA pair lookup failed: {e}")
        
        # Check DDI Corpus - try both original and normalized
        try:
            from .ddi_sentence_db import get_ddi_sentence_db
            db = get_ddi_sentence_db()
            sentence = db.find_sentence(drug1, drug2)
            if not sentence and (norm1 != drug1.lower() or norm2 != drug2.lower()):
                sentence = db.find_sentence(norm1, norm2)
                if sentence:
                    matched_normalized = True
            if sentence:
                evidence.append('ddi_corpus')
        except Exception as e:
            logger.debug(f"DDI Corpus lookup failed: {e}")
        
        # Check Knowledge Graph - use normalized names for better matching
        try:
            from .knowledge_graph import KnowledgeGraphService as KG
            if KG.is_connected():
                # First try exact match
                results = KG.run_query('''
                    MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
                    WHERE toLower(d1.name) = toLower($drug1)
                      AND toLower(d2.name) = toLower($drug2)
                    RETURN r.severity as severity, r.mechanism as mechanism, r.description as description
                    LIMIT 1
                ''', {'drug1': drug1, 'drug2': drug2})
                
                # If no match, try with normalized names
                if not results and (norm1 != drug1.lower() or norm2 != drug2.lower()):
                    results = KG.run_query('''
                        MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
                        WHERE toLower(d1.name) = toLower($drug1)
                          AND toLower(d2.name) = toLower($drug2)
                        RETURN r.severity as severity, r.mechanism as mechanism, r.description as description
                        LIMIT 1
                    ''', {'drug1': norm1, 'drug2': norm2})
                    if results:
                        matched_normalized = True
                
                # If still no match, try CONTAINS for partial matches
                if not results:
                    results = KG.run_query('''
                        MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
                        WHERE toLower(d1.name) CONTAINS toLower($drug1)
                          AND toLower(d2.name) CONTAINS toLower($drug2)
                        RETURN r.severity as severity, r.mechanism as mechanism, r.description as description
                        LIMIT 1
                    ''', {'drug1': norm1, 'drug2': norm2})
                
                if results:
                    info.severity = results[0].get('severity', 'unknown')
                    info.mechanism = results[0].get('mechanism', '') or results[0].get('description', '')
                    evidence.append('knowledge_graph')
                    if matched_normalized:
                        evidence.append(f'matched_via_normalization:{norm1}+{norm2}')
        except Exception as e:
            logger.debug(f"KG interaction lookup failed: {e}")
        
        info.evidence_sources = evidence
        return info


# Global instance
_enhanced_service = None


def get_enhanced_drug_service() -> EnhancedDrugService:
    """Get or create the enhanced drug service."""
    global _enhanced_service
    if _enhanced_service is None:
        _enhanced_service = EnhancedDrugService()
    return _enhanced_service
