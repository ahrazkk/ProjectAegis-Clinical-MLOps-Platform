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
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone

from django.conf import settings

logger = logging.getLogger(__name__)


OPENFDA_FAERS_CAVEATS = [
    "FAERS reports are voluntary and do not establish causality.",
    "A single report may contain multiple drugs and reactions without one-to-one attribution.",
    "Report volume can reflect usage and reporting behavior, not true incidence rates.",
]

EVIDENCE_SOURCE_WEIGHTS = {
    'knowledge_graph': 1.0,
    'ddi_corpus': 0.9,
    'twosides': 0.78,
    'openfda_faers': 0.62,
    'normalization_layer': 0.35,
    'evidence_aggregator': 0.2,
}

PRIMARY_SOURCE_IDS = {'knowledge_graph', 'ddi_corpus', 'twosides', 'openfda_faers'}


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
        'coumadin': 'warfarin', 'warafin': 'warfarin',
        'asa': 'aspirin', 'acetylsalicylic acid': 'aspirin', 'acetylsalicylate': 'aspirin',
        'plavix': 'clopidogrel', 'xarelto': 'rivaroxaban',
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
    faers_caveats: List[str] = field(default_factory=list)
    faers_freshness: Dict[str, str] = field(default_factory=dict)
    
    # Clinical evidence
    evidence_sources: List[str] = field(default_factory=list)
    evidence_chain: List[Dict] = field(default_factory=list)
    evidence_summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        faers_data = {
            'total_reports': self.faers_reports,
            'top_reactions': self.faers_reactions,
            'serious_outcomes': {},
            'source': 'openfda_faers',
            'freshness': self.faers_freshness,
            'caveats': self.faers_caveats,
        }

        return {
            'drug1': self.drug1,
            'drug2': self.drug2,
            'risk_score': self.risk_score,
            'severity': self.severity,
            'mechanism': self.mechanism,
            'polypharmacy_effects': [asdict(pe) for pe in self.polypharmacy_effects],
            'faers_reports': self.faers_reports,
            'faers_reactions': self.faers_reactions,
            'faers_data': faers_data,
            'evidence_sources': self.evidence_sources,
            'evidence_chain': self.evidence_chain,
            'evidence_summary': self.evidence_summary,
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

    @staticmethod
    def _source_weight(source_id: str) -> float:
        """Return normalized trust weight for a given evidence source."""
        return EVIDENCE_SOURCE_WEIGHTS.get(source_id, 0.5)

    @classmethod
    def _build_evidence_summary(cls, evidence_chain: List[Dict]) -> Dict:
        """Aggregate evidence-chain items into weighted support/conflict metrics."""
        if not evidence_chain:
            return {
                'total_items': 0,
                'supporting_items': 0,
                'uncertain_items': 0,
                'unique_sources': [],
                'primary_source_coverage': {
                    'covered': 0,
                    'total': len(PRIMARY_SOURCE_IDS),
                    'ratio': 0.0,
                    'missing': sorted(PRIMARY_SOURCE_IDS),
                },
                'weighted_support_score': 0.0,
                'weighted_uncertainty_score': 1.0,
                'confidence_band': 'low',
                'disagreement': {
                    'has_conflict': False,
                    'level': 'none',
                    'conflicting_sources': [],
                    'narrative': 'No evidence items available.',
                },
                'source_strengths': [],
                'uncertainty_reasons': [
                    {
                        'type': 'insufficient_evidence',
                        'source_id': 'evidence_aggregator',
                        'source_label': 'Evidence Aggregator',
                        'reason': 'No evidence-chain entries were generated for this interaction pair.',
                    }
                ],
            }

        source_rows = {}
        weighted_support_numerator = 0.0
        weighted_uncertainty_numerator = 0.0
        weighted_total = 0.0

        supporting_items = 0
        uncertain_items = 0
        uncertainty_reasons = []

        for item in evidence_chain:
            source = item.get('source') or {}
            source_id = source.get('id', 'unknown')
            source_label = source.get('label', source_id)
            supports = bool(item.get('supports_interaction'))
            strength_score = float(item.get('strength_score', 0.0) or 0.0)
            source_weight = float(item.get('source_weight', cls._source_weight(source_id)) or 0.0)

            weighted_total += source_weight
            if supports:
                supporting_items += 1
                weighted_support_numerator += strength_score * source_weight
            else:
                uncertain_items += 1
                weighted_uncertainty_numerator += strength_score * source_weight

            row = source_rows.setdefault(source_id, {
                'source_id': source_id,
                'source_label': source_label,
                'weight': source_weight,
                'evidence_count': 0,
                'support_count': 0,
                'uncertain_count': 0,
                'strength_sum': 0.0,
                'weighted_strength_sum': 0.0,
            })
            row['evidence_count'] += 1
            row['support_count'] += 1 if supports else 0
            row['uncertain_count'] += 0 if supports else 1
            row['strength_sum'] += strength_score
            row['weighted_strength_sum'] += strength_score * source_weight

            if not supports:
                reason = {
                    'type': item.get('claim_type', 'uncertainty_signal'),
                    'source_id': source_id,
                    'source_label': source_label,
                    'reason': item.get('claim', 'Uncertain evidence signal detected.'),
                }
                caveats = item.get('caveats') or []
                if caveats:
                    reason['caveat'] = caveats[0]
                uncertainty_reasons.append(reason)

        weighted_support_score = (weighted_support_numerator / weighted_total) if weighted_total > 0 else 0.0
        weighted_uncertainty_score = 1.0 - weighted_support_score

        source_strengths = []
        supporting_sources = []
        uncertain_sources = []

        for row in source_rows.values():
            avg_strength = row['strength_sum'] / row['evidence_count'] if row['evidence_count'] else 0.0
            support_ratio = row['support_count'] / row['evidence_count'] if row['evidence_count'] else 0.0
            weighted_strength = row['weighted_strength_sum'] / row['evidence_count'] if row['evidence_count'] else 0.0

            source_strengths.append({
                'source_id': row['source_id'],
                'source_label': row['source_label'],
                'weight': round(row['weight'], 3),
                'evidence_count': row['evidence_count'],
                'support_ratio': round(support_ratio, 3),
                'avg_strength': round(avg_strength, 3),
                'weighted_strength': round(weighted_strength, 3),
            })

            if support_ratio >= 0.6:
                supporting_sources.append(row['source_id'])
            elif support_ratio <= 0.4:
                uncertain_sources.append(row['source_id'])

        source_strengths.sort(key=lambda x: x['weighted_strength'], reverse=True)

        has_conflict = bool(supporting_sources and uncertain_sources)
        conflicting_sources = sorted(set(supporting_sources + uncertain_sources)) if has_conflict else []

        if has_conflict and abs(weighted_support_score - weighted_uncertainty_score) <= 0.1:
            disagreement_level = 'high'
        elif has_conflict:
            disagreement_level = 'moderate'
        else:
            disagreement_level = 'none'

        if weighted_support_score >= 0.72 and not has_conflict:
            confidence_band = 'high'
        elif weighted_support_score >= 0.45:
            confidence_band = 'moderate'
        else:
            confidence_band = 'low'

        covered_primary_sources = sorted(set(source_rows.keys()).intersection(PRIMARY_SOURCE_IDS))
        missing_primary_sources = sorted(PRIMARY_SOURCE_IDS.difference(set(source_rows.keys())))
        primary_coverage_ratio = len(covered_primary_sources) / len(PRIMARY_SOURCE_IDS)

        if has_conflict:
            disagreement_narrative = (
                'Sources disagree on interaction strength; review the uncertainty reasons and source-specific caveats.'
            )
        elif uncertain_items > 0:
            disagreement_narrative = 'Some evidence is weak or non-supportive; treat conclusion as conditional.'
        else:
            disagreement_narrative = 'Available sources are directionally consistent for this interaction pair.'

        if not uncertainty_reasons and confidence_band != 'high':
            uncertainty_reasons.append({
                'type': 'limited_strength',
                'source_id': 'evidence_aggregator',
                'source_label': 'Evidence Aggregator',
                'reason': 'Evidence support remains moderate/low despite no explicit source conflict.',
            })

        return {
            'total_items': len(evidence_chain),
            'supporting_items': supporting_items,
            'uncertain_items': uncertain_items,
            'unique_sources': sorted(source_rows.keys()),
            'primary_source_coverage': {
                'covered': len(covered_primary_sources),
                'total': len(PRIMARY_SOURCE_IDS),
                'ratio': round(primary_coverage_ratio, 3),
                'missing': missing_primary_sources,
            },
            'weighted_support_score': round(weighted_support_score, 3),
            'weighted_uncertainty_score': round(weighted_uncertainty_score, 3),
            'confidence_band': confidence_band,
            'disagreement': {
                'has_conflict': has_conflict,
                'level': disagreement_level,
                'conflicting_sources': conflicting_sources,
                'narrative': disagreement_narrative,
            },
            'source_strengths': source_strengths,
            'uncertainty_reasons': uncertainty_reasons,
        }
    
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
        evidence_chain = []

        def add_evidence_item(
            claim_type: str,
            claim: str,
            source_id: str,
            source_label: str,
            source_category: str,
            strength_score: float,
            supports_interaction: bool,
            details: Optional[Dict] = None,
            caveats: Optional[List[str]] = None,
            freshness: Optional[Dict] = None,
        ):
            source_weight = self._source_weight(source_id)
            bounded_strength = round(max(0.0, min(1.0, float(strength_score))), 3)
            evidence_chain.append({
                'step': len(evidence_chain) + 1,
                'claim_type': claim_type,
                'claim': claim,
                'supports_interaction': supports_interaction,
                'strength_score': bounded_strength,
                'source_weight': round(source_weight, 3),
                'weighted_strength_score': round(bounded_strength * source_weight, 3),
                'source': {
                    'id': source_id,
                    'label': source_label,
                    'category': source_category,
                },
                'details': details or {},
                'freshness': freshness or {},
                'caveats': caveats or [],
            })

        ddi_sentence = None
        
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

                    numeric_scores = []
                    for effect in effects:
                        try:
                            numeric_scores.append(float(effect.get('score', 0.0)))
                        except (TypeError, ValueError):
                            continue

                    mean_signal = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.35
                    top_effects = [
                        {
                            'effect': effect.get('side_effect', ''),
                            'score': effect.get('score', 0.0),
                        }
                        for effect in effects[:3]
                    ]

                    add_evidence_item(
                        claim_type='polypharmacy_effect_signal',
                        claim=(
                            f"TWOSIDES reports {len(effects)} co-administration side-effect signal(s) "
                            f"for {drug1} + {drug2}."
                        ),
                        source_id='twosides',
                        source_label='TWOSIDES',
                        source_category='polypharmacy_dataset',
                        strength_score=mean_signal,
                        supports_interaction=True,
                        details={'top_effects': top_effects},
                        caveats=['Signal-level association data; not standalone causal proof.'],
                        freshness={'type': 'snapshot_dataset'},
                    )
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
                    info.faers_caveats = list(OPENFDA_FAERS_CAVEATS)
                    info.faers_freshness = {
                        'update_frequency': 'quarterly',
                        'expected_lag': '3 months or more',
                        'queried_at': datetime.now(timezone.utc).isoformat(),
                        'source': 'https://open.fda.gov/apis/drug/event/',
                    }

                    faers_strength = min(0.95, max(0.1, math.log10(info.faers_reports + 1) / 4))
                    faers_supports = info.faers_reports > 0

                    add_evidence_item(
                        claim_type='real_world_safety_signal',
                        claim=(
                            f"FAERS includes {info.faers_reports:,} co-reported adverse-event record(s) "
                            f"for {drug1} + {drug2}."
                        ) if faers_supports else (
                            f"No FAERS co-report signal was found for {drug1} + {drug2} in this query window."
                        ),
                        source_id='openfda_faers',
                        source_label='OpenFDA FAERS',
                        source_category='post_market_surveillance',
                        strength_score=faers_strength,
                        supports_interaction=faers_supports,
                        details={
                            'total_reports': info.faers_reports,
                            'top_reactions': info.faers_reactions[:5],
                        },
                        caveats=info.faers_caveats,
                        freshness=info.faers_freshness,
                    )

                    if info.faers_reports > 0:
                        evidence.append('openfda_faers')
            except Exception as e:
                logger.debug(f"OpenFDA pair lookup failed: {e}")
        
        # Check DDI Corpus - try both original and normalized
        try:
            from .ddi_sentence_db import get_ddi_sentence_db
            db = get_ddi_sentence_db()
            ddi_sentence = db.find_sentence(drug1, drug2)
            if not ddi_sentence and (norm1 != drug1.lower() or norm2 != drug2.lower()):
                ddi_sentence = db.find_sentence(norm1, norm2)
                if ddi_sentence:
                    matched_normalized = True

            if ddi_sentence:
                evidence.append('ddi_corpus')

                add_evidence_item(
                    claim_type='literature_sentence_evidence',
                    claim='A curated DDI corpus sentence supports this pair-level interaction context.',
                    source_id='ddi_corpus',
                    source_label='DDI Corpus',
                    source_category='literature_corpus',
                    strength_score=getattr(ddi_sentence, 'confidence', 0.75),
                    supports_interaction=True,
                    details={
                        'sentence': getattr(ddi_sentence, 'sentence', ''),
                        'interaction_type': getattr(ddi_sentence, 'interaction_type', ''),
                        'source': getattr(ddi_sentence, 'source', ''),
                    },
                    freshness={'type': 'snapshot_dataset'},
                )
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

                # If still no match, try CONTAINS for partial matches.
                # Guard against short acronyms (e.g., "asa") causing spurious matches.
                if not results and len(norm1) >= 4 and len(norm2) >= 4:
                    results = KG.run_query('''
                        MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
                        WHERE toLower(d1.name) CONTAINS toLower($drug1)
                          AND toLower(d2.name) CONTAINS toLower($drug2)
                        RETURN r.severity as severity, r.mechanism as mechanism, r.description as description
                        LIMIT 1
                    ''', {'drug1': norm1, 'drug2': norm2})
                
                if results:
                    raw_severity = results[0].get('severity', 'unknown')
                    raw_mechanism = results[0].get('mechanism', '')
                    raw_description = results[0].get('description', '')
                    info.mechanism = raw_mechanism or raw_description
                    evidence.append('knowledge_graph')

                    if raw_severity is None:
                        description_text = str(raw_description or '').strip().lower()
                        mechanism_text = str(raw_mechanism or '').strip().lower()
                        looks_like_placeholder_only = (
                            not mechanism_text and (
                                not description_text
                                or description_text.startswith('e.g.')
                                or 'no clinically significant interaction' in description_text
                            )
                        )
                        severity_text = 'no_interaction' if looks_like_placeholder_only else 'unknown'
                    else:
                        severity_text = str(raw_severity or 'unknown').lower().strip().replace(' ', '_')

                    if severity_text in {'none', 'no-interaction'}:
                        severity_text = 'no_interaction'

                    info.severity = 'no_interaction' if severity_text == 'no_interaction' else (raw_severity or 'unknown')
                    severity_strength = {
                        'severe': 0.95,
                        'major': 0.9,
                        'moderate': 0.75,
                        'minor': 0.55,
                        'no_interaction': 0.2,
                        'unknown': 0.5,
                    }.get(severity_text, 0.5)
                    info.risk_score = {
                        'severe': 0.9,
                        'major': 0.82,
                        'moderate': 0.65,
                        'minor': 0.4,
                        'no_interaction': 0.08,
                    }.get(severity_text, info.risk_score)

                    add_evidence_item(
                        claim_type='curated_knowledge_graph_relation',
                        claim=(
                            f"Knowledge Graph indicates no clinically significant interaction for {drug1} + {drug2}."
                            if severity_text == 'no_interaction'
                            else (
                                f"Knowledge Graph contains a curated relation for {drug1} + {drug2} "
                                f"with severity '{info.severity}'."
                            )
                        ),
                        source_id='knowledge_graph',
                        source_label='Neo4j Knowledge Graph',
                        source_category='curated_ddi_relation',
                        strength_score=severity_strength,
                        supports_interaction=severity_text != 'no_interaction',
                        details={
                            'severity': info.severity,
                            'mechanism': info.mechanism,
                        },
                        freshness={'type': 'internal_graph_snapshot'},
                    )

                    if matched_normalized:
                        evidence.append(f'matched_via_normalization:{norm1}+{norm2}')
        except Exception as e:
            logger.debug(f"KG interaction lookup failed: {e}")

        if matched_normalized:
            add_evidence_item(
                claim_type='name_normalization',
                claim=f"Name normalization was applied: '{drug1}'/'{drug2}' -> '{norm1}'/'{norm2}'.",
                source_id='normalization_layer',
                source_label='Normalization Layer',
                source_category='query_preprocessing',
                strength_score=0.5,
                supports_interaction=True,
                details={'normalized_drug1': norm1, 'normalized_drug2': norm2},
                freshness={'type': 'runtime'},
            )

        if not evidence_chain:
            add_evidence_item(
                claim_type='insufficient_evidence',
                claim=(
                    f"No direct cross-source evidence chain was found for {drug1} + {drug2}; "
                    "treat this as uncertain and verify with clinical references."
                ),
                source_id='evidence_aggregator',
                source_label='Evidence Aggregator',
                source_category='meta',
                strength_score=0.2,
                supports_interaction=False,
                caveats=['Absence of evidence is not evidence of no interaction.'],
                freshness={'type': 'runtime'},
            )
        
        info.evidence_sources = evidence
        info.evidence_chain = evidence_chain
        info.evidence_summary = self._build_evidence_summary(evidence_chain)
        return info


# Global instance
_enhanced_service = None


def get_enhanced_drug_service() -> EnhancedDrugService:
    """Get or create the enhanced drug service."""
    global _enhanced_service
    if _enhanced_service is None:
        _enhanced_service = EnhancedDrugService()
    return _enhanced_service
