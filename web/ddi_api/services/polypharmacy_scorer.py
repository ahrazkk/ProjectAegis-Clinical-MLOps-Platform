"""
Polypharmacy Risk Scoring Service

Provides comprehensive risk assessment for patients taking multiple medications.
Polypharmacy (≥5 medications) significantly increases DDI risk.

Reference:
- Masnoon et al., "What is polypharmacy? A systematic review of definitions" BMC Geriatrics 2017
- Clinical guidelines on medication safety
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import itertools

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Polypharmacy risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class DrugInfo:
    """Information about a drug for risk assessment."""
    name: str
    drugbank_id: Optional[str] = None
    therapeutic_class: Optional[str] = None
    smiles: Optional[str] = None
    is_high_risk: bool = False
    is_narrow_therapeutic_index: bool = False


@dataclass
class PatientProfile:
    """Patient information for personalized risk assessment."""
    age: Optional[int] = None
    weight: Optional[float] = None  # kg
    creatinine_clearance: Optional[float] = None  # mL/min
    hepatic_function: Optional[str] = None  # normal, mild, moderate, severe
    pregnancy_status: Optional[str] = None  # none, pregnant, lactating
    genetic_factors: Optional[Dict[str, str]] = None  # e.g., {"CYP2D6": "poor_metabolizer"}
    
    def get_age_risk_factor(self) -> float:
        """Get risk multiplier based on age."""
        if self.age is None:
            return 1.0
        
        if self.age >= 80:
            return 1.8  # Very high risk
        elif self.age >= 65:
            return 1.5  # High risk (geriatric)
        elif self.age <= 12:
            return 1.4  # Pediatric considerations
        elif self.age <= 2:
            return 1.6  # Neonatal/infant
        else:
            return 1.0  # Adult baseline
    
    def get_renal_risk_factor(self) -> float:
        """Get risk multiplier based on renal function."""
        if self.creatinine_clearance is None:
            return 1.0
        
        if self.creatinine_clearance < 15:
            return 2.0  # Severe renal impairment
        elif self.creatinine_clearance < 30:
            return 1.7  # Moderate-severe
        elif self.creatinine_clearance < 60:
            return 1.4  # Moderate
        elif self.creatinine_clearance < 90:
            return 1.2  # Mild
        else:
            return 1.0  # Normal
    
    def get_hepatic_risk_factor(self) -> float:
        """Get risk multiplier based on hepatic function."""
        if self.hepatic_function is None:
            return 1.0
        
        hepatic_factors = {
            'normal': 1.0,
            'mild': 1.3,
            'moderate': 1.6,
            'severe': 2.0
        }
        return hepatic_factors.get(self.hepatic_function.lower(), 1.0)


@dataclass
class InteractionResult:
    """Result of checking a drug pair interaction."""
    drug1: str
    drug2: str
    severity: str
    risk_score: float
    mechanism: Optional[str] = None
    source: str = "unknown"


@dataclass 
class PolypharmacyRiskReport:
    """Comprehensive polypharmacy risk report."""
    total_medications: int
    polypharmacy_level: str
    overall_risk_level: RiskLevel
    overall_risk_score: float
    interaction_count: int
    severe_interactions: int
    major_interactions: int
    moderate_interactions: int
    high_risk_drug_count: int
    nti_drug_count: int
    duplicate_therapy_count: int
    drug_interactions: List[InteractionResult]
    duplicate_therapies: List[Dict[str, Any]]
    recommendations: List[str]
    patient_adjusted: bool
    patient_risk_factors: Dict[str, float] = field(default_factory=dict)


# High-risk medications that require extra monitoring
HIGH_RISK_MEDICATIONS = {
    'warfarin', 'heparin', 'enoxaparin',  # Anticoagulants
    'insulin', 'metformin', 'glipizide', 'glyburide',  # Diabetes
    'digoxin', 'amiodarone',  # Cardiac
    'methotrexate', 'cyclophosphamide',  # Chemotherapy
    'lithium',  # Psychiatric
    'phenytoin', 'carbamazepine', 'valproic acid',  # Anticonvulsants
    'opioids', 'morphine', 'fentanyl', 'oxycodone',  # Opioids
}

# Narrow Therapeutic Index drugs
NTI_MEDICATIONS = {
    'warfarin', 'digoxin', 'lithium', 'phenytoin',
    'theophylline', 'cyclosporine', 'tacrolimus', 'sirolimus',
    'levothyroxine', 'carbamazepine', 'valproic acid',
    'aminoglycosides', 'gentamicin', 'vancomycin',
}

# Therapeutic classes for duplicate therapy detection
THERAPEUTIC_CLASSES = {
    'nsaids': ['ibuprofen', 'naproxen', 'aspirin', 'celecoxib', 'meloxicam', 'diclofenac', 'indomethacin'],
    'ace_inhibitors': ['lisinopril', 'enalapril', 'ramipril', 'benazepril', 'captopril'],
    'arbs': ['losartan', 'valsartan', 'irbesartan', 'candesartan', 'olmesartan'],
    'beta_blockers': ['metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol'],
    'statins': ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin'],
    'ppis': ['omeprazole', 'esomeprazole', 'pantoprazole', 'lansoprazole', 'rabeprazole'],
    'ssris': ['fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram'],
    'benzodiazepines': ['diazepam', 'lorazepam', 'alprazolam', 'clonazepam', 'temazepam'],
    'opioids': ['morphine', 'oxycodone', 'hydrocodone', 'fentanyl', 'codeine', 'tramadol'],
    'diuretics': ['furosemide', 'hydrochlorothiazide', 'chlorthalidone', 'spironolactone'],
    'anticoagulants': ['warfarin', 'rivaroxaban', 'apixaban', 'dabigatran', 'edoxaban'],
    'antiplatelets': ['aspirin', 'clopidogrel', 'ticagrelor', 'prasugrel'],
}


class PolypharmacyRiskScorer:
    """
    Comprehensive polypharmacy risk assessment service.
    
    Features:
    - Multi-drug interaction analysis
    - Duplicate therapy detection
    - High-risk medication identification
    - Patient-specific risk adjustment
    - Evidence-based recommendations
    """
    
    def __init__(self):
        """Initialize the risk scorer."""
        self._cyp450_db = None
        self._predictor = None
        self._gnn_predictor = None
    
    def _get_cyp450_db(self):
        """Lazy load CYP450 database."""
        if self._cyp450_db is None:
            try:
                from .cyp450_database import get_cyp450_database
                self._cyp450_db = get_cyp450_database()
            except ImportError:
                logger.warning("CYP450 database not available")
        return self._cyp450_db
    
    def _get_predictor(self):
        """Lazy load PubMedBERT predictor."""
        if self._predictor is None:
            try:
                from .pubmedbert_predictor import get_pubmedbert_predictor
                self._predictor = get_pubmedbert_predictor()
            except ImportError:
                logger.warning("PubMedBERT predictor not available")
        return self._predictor
    
    def _get_gnn_predictor(self):
        """Lazy load GNN predictor."""
        if self._gnn_predictor is None:
            try:
                from .gnn_predictor import get_gnn_predictor
                self._gnn_predictor = get_gnn_predictor()
            except ImportError:
                logger.warning("GNN predictor not available")
        return self._gnn_predictor
    
    def classify_polypharmacy(self, medication_count: int) -> str:
        """
        Classify polypharmacy level based on medication count.
        
        Standard definitions:
        - Minor: 0-4 medications
        - Polypharmacy: 5-9 medications
        - Excessive polypharmacy: ≥10 medications
        """
        if medication_count < 5:
            return "minor"
        elif medication_count < 10:
            return "polypharmacy"
        else:
            return "excessive_polypharmacy"
    
    def identify_high_risk_medications(
        self, 
        medications: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Identify high-risk and narrow therapeutic index medications.
        
        Returns:
            Tuple of (high_risk_drugs, nti_drugs)
        """
        meds_lower = [m.lower().strip() for m in medications]
        
        high_risk = [m for m in meds_lower if m in HIGH_RISK_MEDICATIONS]
        nti = [m for m in meds_lower if m in NTI_MEDICATIONS]
        
        return high_risk, nti
    
    def detect_duplicate_therapies(
        self, 
        medications: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect duplicate therapy (multiple drugs in same class).
        
        Returns:
            List of duplicate therapy findings
        """
        meds_lower = {m.lower().strip() for m in medications}
        duplicates = []
        
        for class_name, class_drugs in THERAPEUTIC_CLASSES.items():
            matches = [d for d in class_drugs if d in meds_lower]
            
            if len(matches) > 1:
                duplicates.append({
                    'therapeutic_class': class_name,
                    'drugs': matches,
                    'count': len(matches),
                    'recommendation': f"Consider reviewing need for multiple {class_name.replace('_', ' ')}"
                })
        
        return duplicates
    
    def check_all_interactions(
        self,
        medications: List[str],
        use_gnn: bool = False
    ) -> List[InteractionResult]:
        """
        Check all pairwise drug interactions.
        
        Args:
            medications: List of drug names
            use_gnn: Whether to use GNN predictor (vs PubMedBERT)
            
        Returns:
            List of interaction results
        """
        interactions = []
        
        # Get predictor
        if use_gnn:
            predictor = self._get_gnn_predictor()
        else:
            predictor = self._get_predictor()
        
        cyp_db = self._get_cyp450_db()
        
        # Check all pairs
        for drug1, drug2 in itertools.combinations(medications, 2):
            # Check CYP450 interactions first
            cyp_interactions = []
            if cyp_db:
                cyp_interactions = cyp_db.check_cyp_interaction(drug1, drug2)
            
            # If CYP interaction found, use that
            if cyp_interactions:
                for cyp_int in cyp_interactions:
                    interactions.append(InteractionResult(
                        drug1=drug1,
                        drug2=drug2,
                        severity=cyp_int.severity,
                        risk_score=self._severity_to_score(cyp_int.severity),
                        mechanism=cyp_int.mechanism,
                        source=f"CYP450 ({cyp_int.enzyme.value})"
                    ))
            # Otherwise use ML predictor
            elif predictor and predictor.is_loaded:
                try:
                    if use_gnn:
                        pred = predictor.predict(drug1, drug2)
                        if pred.interaction_type != 'no_interaction':
                            interactions.append(InteractionResult(
                                drug1=drug1,
                                drug2=drug2,
                                severity=pred.severity,
                                risk_score=pred.interaction_probability,
                                mechanism=pred.mechanism_hypothesis,
                                source="GNN"
                            ))
                    else:
                        pred = predictor.predict(drug1, drug2)
                        if pred.interaction_type != 'no_interaction':
                            interactions.append(InteractionResult(
                                drug1=drug1,
                                drug2=drug2,
                                severity=pred.severity,
                                risk_score=pred.risk_score,
                                mechanism=predictor.get_mechanism_description(
                                    pred.interaction_type, drug1, drug2
                                ),
                                source="PubMedBERT"
                            ))
                except Exception as e:
                    logger.warning(f"Prediction failed for {drug1}-{drug2}: {e}")
        
        # Sort by risk score (highest first)
        interactions.sort(key=lambda x: x.risk_score, reverse=True)
        
        return interactions
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity string to numeric score."""
        severity_scores = {
            'severe': 0.9,
            'major': 0.75,
            'moderate': 0.5,
            'minor': 0.25,
            'none': 0.0
        }
        return severity_scores.get(severity.lower(), 0.5)
    
    def calculate_overall_risk(
        self,
        medications: List[str],
        interactions: List[InteractionResult],
        high_risk_count: int,
        nti_count: int,
        duplicate_count: int,
        patient: Optional[PatientProfile] = None
    ) -> Tuple[RiskLevel, float, Dict[str, float]]:
        """
        Calculate overall polypharmacy risk score.
        
        Returns:
            Tuple of (risk_level, risk_score, risk_factors)
        """
        # Base risk from medication count
        med_count = len(medications)
        if med_count >= 10:
            count_risk = 0.6
        elif med_count >= 7:
            count_risk = 0.4
        elif med_count >= 5:
            count_risk = 0.3
        else:
            count_risk = 0.1
        
        # Risk from interactions
        severe_count = sum(1 for i in interactions if i.severity == 'severe')
        major_count = sum(1 for i in interactions if i.severity == 'major')
        moderate_count = sum(1 for i in interactions if i.severity == 'moderate')
        
        interaction_risk = min(1.0, (
            severe_count * 0.3 +
            major_count * 0.15 +
            moderate_count * 0.05
        ))
        
        # Risk from high-risk medications
        high_risk_risk = min(0.4, high_risk_count * 0.1)
        
        # Risk from NTI medications
        nti_risk = min(0.3, nti_count * 0.1)
        
        # Risk from duplicate therapies
        duplicate_risk = min(0.2, duplicate_count * 0.05)
        
        # Combine risks
        base_risk = (
            count_risk * 0.2 +
            interaction_risk * 0.4 +
            high_risk_risk * 0.15 +
            nti_risk * 0.15 +
            duplicate_risk * 0.1
        )
        
        risk_factors = {
            'medication_count': count_risk,
            'drug_interactions': interaction_risk,
            'high_risk_medications': high_risk_risk,
            'narrow_therapeutic_index': nti_risk,
            'duplicate_therapies': duplicate_risk
        }
        
        # Apply patient-specific adjustments
        final_risk = base_risk
        if patient:
            age_factor = patient.get_age_risk_factor()
            renal_factor = patient.get_renal_risk_factor()
            hepatic_factor = patient.get_hepatic_risk_factor()
            
            patient_multiplier = (age_factor + renal_factor + hepatic_factor) / 3
            final_risk = min(1.0, base_risk * patient_multiplier)
            
            risk_factors['age_adjustment'] = age_factor
            risk_factors['renal_adjustment'] = renal_factor
            risk_factors['hepatic_adjustment'] = hepatic_factor
        
        # Determine risk level
        if final_risk >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif final_risk >= 0.6:
            risk_level = RiskLevel.VERY_HIGH
        elif final_risk >= 0.4:
            risk_level = RiskLevel.HIGH
        elif final_risk >= 0.2:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return risk_level, final_risk, risk_factors
    
    def generate_recommendations(
        self,
        interactions: List[InteractionResult],
        duplicates: List[Dict],
        high_risk: List[str],
        nti: List[str],
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate clinical recommendations based on risk assessment."""
        recommendations = []
        
        # Severe interactions
        severe_interactions = [i for i in interactions if i.severity == 'severe']
        if severe_interactions:
            for interaction in severe_interactions[:3]:  # Top 3
                recommendations.append(
                    f"⚠️ SEVERE: Review {interaction.drug1}-{interaction.drug2} combination. "
                    f"Consider alternative therapy or increased monitoring."
                )
        
        # Major interactions
        major_interactions = [i for i in interactions if i.severity == 'major']
        if major_interactions:
            recommendations.append(
                f"⚡ {len(major_interactions)} major interactions detected. "
                "Review therapy and consider dose adjustments."
            )
        
        # Duplicate therapies
        for dup in duplicates:
            recommendations.append(
                f"🔄 Duplicate therapy: {', '.join(dup['drugs'])} "
                f"({dup['therapeutic_class'].replace('_', ' ')}). "
                f"Consider discontinuing one agent."
            )
        
        # High-risk medications
        if high_risk:
            recommendations.append(
                f"⚠️ {len(high_risk)} high-risk medication(s): {', '.join(high_risk)}. "
                "Ensure appropriate monitoring is in place."
            )
        
        # NTI medications
        if nti:
            recommendations.append(
                f"📊 {len(nti)} narrow therapeutic index drug(s): {', '.join(nti)}. "
                "Monitor drug levels and adjust doses as needed."
            )
        
        # General recommendations based on risk level
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.VERY_HIGH]:
            recommendations.append(
                "🏥 Consider comprehensive medication review by clinical pharmacist."
            )
            recommendations.append(
                "📋 Recommend regular reassessment of medication necessity."
            )
        
        if not recommendations:
            recommendations.append("✅ No significant polypharmacy concerns identified.")
        
        return recommendations
    
    def assess_polypharmacy_risk(
        self,
        medications: List[str],
        patient: Optional[PatientProfile] = None,
        use_gnn: bool = False
    ) -> PolypharmacyRiskReport:
        """
        Comprehensive polypharmacy risk assessment.
        
        Args:
            medications: List of medication names
            patient: Optional patient profile for personalization
            use_gnn: Whether to use GNN predictor
            
        Returns:
            Complete PolypharmacyRiskReport
        """
        # Clean medication list
        meds = [m.strip() for m in medications if m.strip()]
        
        if not meds:
            return PolypharmacyRiskReport(
                total_medications=0,
                polypharmacy_level="none",
                overall_risk_level=RiskLevel.LOW,
                overall_risk_score=0.0,
                interaction_count=0,
                severe_interactions=0,
                major_interactions=0,
                moderate_interactions=0,
                high_risk_drug_count=0,
                nti_drug_count=0,
                duplicate_therapy_count=0,
                drug_interactions=[],
                duplicate_therapies=[],
                recommendations=["No medications to analyze."],
                patient_adjusted=False
            )
        
        # Identify special medications
        high_risk, nti = self.identify_high_risk_medications(meds)
        
        # Detect duplicate therapies
        duplicates = self.detect_duplicate_therapies(meds)
        
        # Check all interactions
        interactions = self.check_all_interactions(meds, use_gnn=use_gnn)
        
        # Count by severity
        severe_count = sum(1 for i in interactions if i.severity == 'severe')
        major_count = sum(1 for i in interactions if i.severity == 'major')
        moderate_count = sum(1 for i in interactions if i.severity == 'moderate')
        
        # Calculate overall risk
        risk_level, risk_score, risk_factors = self.calculate_overall_risk(
            meds, interactions, len(high_risk), len(nti), len(duplicates), patient
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            interactions, duplicates, high_risk, nti, risk_level
        )
        
        return PolypharmacyRiskReport(
            total_medications=len(meds),
            polypharmacy_level=self.classify_polypharmacy(len(meds)),
            overall_risk_level=risk_level,
            overall_risk_score=risk_score,
            interaction_count=len(interactions),
            severe_interactions=severe_count,
            major_interactions=major_count,
            moderate_interactions=moderate_count,
            high_risk_drug_count=len(high_risk),
            nti_drug_count=len(nti),
            duplicate_therapy_count=len(duplicates),
            drug_interactions=interactions,
            duplicate_therapies=duplicates,
            recommendations=recommendations,
            patient_adjusted=patient is not None,
            patient_risk_factors=risk_factors
        )


# Singleton instance
_polypharmacy_scorer: Optional[PolypharmacyRiskScorer] = None


def get_polypharmacy_scorer() -> PolypharmacyRiskScorer:
    """Get or create the polypharmacy risk scorer singleton."""
    global _polypharmacy_scorer
    if _polypharmacy_scorer is None:
        _polypharmacy_scorer = PolypharmacyRiskScorer()
    return _polypharmacy_scorer
