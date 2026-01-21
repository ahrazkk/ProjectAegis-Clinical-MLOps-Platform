"""
Risk Scoring Logic for Clinical Decision Support
Implements Section 3 of the AI Model Specification
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from enum import Enum


class InteractionType(Enum):
    """DDI Interaction Types from DDIExtraction 2013 Corpus"""
    NONE = 0  # No interaction
    MECHANISM = 1  # Pharmacokinetic mechanism
    EFFECT = 2  # Clinical effect
    ADVISE = 3  # Advisory/Recommendation
    INT = 4  # General interaction


class RiskLevel(Enum):
    """Clinical Risk Categorization"""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class RiskScorer:
    """
    Clinical Risk Scoring Service
    
    Implements the risk scoring pipeline from Section 3 of the specification:
    1. Probability Calibration (Temperature Scaling)
    2. Weighted Risk Score Calculation
    3. Risk Categorization
    
    Severity Weights (Section 3.2):
    - Minor (Advise): 0.2
    - Moderate (Effect): 0.6
    - Major (Mechanism): 0.9
    
    Risk Thresholds (Section 3.3):
    - Low Risk: R < 0.3
    - Moderate Risk: 0.3 <= R < 0.7
    - High Risk: R >= 0.7
    """
    
    # Severity weights from specification (Section 3.2)
    SEVERITY_WEIGHTS = {
        InteractionType.NONE: 0.0,
        InteractionType.ADVISE: 0.2,  # Minor
        InteractionType.EFFECT: 0.6,  # Moderate
        InteractionType.MECHANISM: 0.9,  # Major (Pharmacokinetic)
        InteractionType.INT: 0.5,  # General interaction (medium severity)
    }
    
    # Risk thresholds from specification (Section 3.3)
    RISK_THRESHOLDS = {
        RiskLevel.LOW: (0.0, 0.3),
        RiskLevel.MODERATE: (0.3, 0.7),
        RiskLevel.HIGH: (0.7, 1.0),
    }
    
    def __init__(self, use_calibrated_probs: bool = True):
        """
        Args:
            use_calibrated_probs: Whether to use temperature-scaled probabilities
        """
        self.use_calibrated_probs = use_calibrated_probs
        
    def calculate_risk_score(
        self,
        relation_probs: torch.Tensor,
        interaction_types: Optional[List[InteractionType]] = None,
    ) -> torch.Tensor:
        """
        Calculate scalar risk score using weighted sum formula (Section 3.2).
        
        R = Σ(P_calibrated_i × W_i)
        
        Args:
            relation_probs: [batch_size, num_classes] - Calibrated probabilities
            interaction_types: List of InteractionType enum values matching class order
                              If None, assumes [NONE, MECHANISM, EFFECT, ADVISE, INT]
        
        Returns:
            risk_scores: [batch_size] - Scalar risk scores in [0, 1]
        """
        if interaction_types is None:
            interaction_types = [
                InteractionType.NONE,
                InteractionType.MECHANISM,
                InteractionType.EFFECT,
                InteractionType.ADVISE,
                InteractionType.INT,
            ]
        
        # Get severity weights for each class
        weights = torch.tensor(
            [self.SEVERITY_WEIGHTS[itype] for itype in interaction_types],
            dtype=relation_probs.dtype,
            device=relation_probs.device,
        )
        
        # Weighted sum: R = Σ(P_i × W_i)
        risk_scores = torch.sum(relation_probs * weights.unsqueeze(0), dim=-1)
        
        return risk_scores
    
    def categorize_risk(self, risk_score: float) -> RiskLevel:
        """
        Map scalar risk score to categorical risk level (Section 3.3).
        
        Args:
            risk_score: Scalar risk score in [0, 1]
            
        Returns:
            RiskLevel enum value
        """
        if risk_score < self.RISK_THRESHOLDS[RiskLevel.LOW][1]:
            return RiskLevel.LOW
        elif risk_score < self.RISK_THRESHOLDS[RiskLevel.MODERATE][1]:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH
    
    def compute_full_risk_profile(
        self,
        relation_probs: torch.Tensor,
        interaction_types: Optional[List[InteractionType]] = None,
    ) -> List[Dict[str, any]]:
        """
        Compute complete risk profile for batch of predictions.
        
        Args:
            relation_probs: [batch_size, num_classes] - Calibrated probabilities
            interaction_types: List of InteractionType enum values
            
        Returns:
            List of dictionaries containing:
                - risk_score: Scalar risk score
                - risk_level: Categorical risk level (Low/Moderate/High)
                - predicted_interaction: Most likely interaction type
                - interaction_probability: Probability of predicted interaction
                - all_probabilities: Dict of all class probabilities
        """
        risk_scores = self.calculate_risk_score(relation_probs, interaction_types)
        
        if interaction_types is None:
            interaction_types = [
                InteractionType.NONE,
                InteractionType.MECHANISM,
                InteractionType.EFFECT,
                InteractionType.ADVISE,
                InteractionType.INT,
            ]
        
        results = []
        
        for i in range(relation_probs.size(0)):
            probs = relation_probs[i]
            risk_score = risk_scores[i].item()
            
            # Predicted interaction type
            pred_idx = torch.argmax(probs).item()
            predicted_interaction = interaction_types[pred_idx]
            interaction_probability = probs[pred_idx].item()
            
            # Risk categorization
            risk_level = self.categorize_risk(risk_score)
            
            # All class probabilities
            all_probs = {
                itype.name: probs[j].item()
                for j, itype in enumerate(interaction_types)
            }
            
            results.append({
                "risk_score": risk_score,
                "risk_level": risk_level.value,
                "predicted_interaction": predicted_interaction.name,
                "interaction_probability": interaction_probability,
                "all_probabilities": all_probs,
            })
        
        return results
    
    @staticmethod
    def get_clinical_recommendation(risk_level: RiskLevel, predicted_interaction: str) -> str:
        """
        Generate clinical recommendation based on risk level.
        
        Args:
            risk_level: RiskLevel enum value
            predicted_interaction: Name of predicted interaction type
            
        Returns:
            Clinical recommendation string
        """
        recommendations = {
            RiskLevel.HIGH: (
                "⚠️ HIGH RISK: Avoid co-administration if possible. "
                "Consider alternative medications. If unavoidable, close monitoring required."
            ),
            RiskLevel.MODERATE: (
                "⚡ MODERATE RISK: Monitor patient closely. "
                "Adjust dosing if necessary. Be alert for adverse effects."
            ),
            RiskLevel.LOW: (
                "ℹ️ LOW RISK: Generally safe to co-administer. "
                "Standard monitoring procedures apply."
            ),
        }
        
        return recommendations.get(risk_level, "No specific recommendation available.")


class MultiDrugRiskScorer:
    """
    Risk Scorer for Polypharmacy (N-way Drug Interactions)
    
    Extends the binary risk scoring to handle multiple concurrent medications.
    """
    
    def __init__(self, base_scorer: RiskScorer):
        """
        Args:
            base_scorer: Base RiskScorer for pairwise interactions
        """
        self.base_scorer = base_scorer
    
    def compute_pairwise_risks(
        self,
        drug_list: List[str],
        pairwise_probs: Dict[Tuple[str, str], torch.Tensor],
    ) -> Dict[Tuple[str, str], Dict[str, any]]:
        """
        Compute risk scores for all pairwise interactions in a drug list.
        
        Args:
            drug_list: List of drug identifiers
            pairwise_probs: Dict mapping (drug1, drug2) -> calibrated probabilities
            
        Returns:
            Dict mapping (drug1, drug2) -> risk profile
        """
        pairwise_risks = {}
        
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                key = (drug1, drug2)
                if key in pairwise_probs:
                    probs = pairwise_probs[key].unsqueeze(0)  # Add batch dimension
                    risk_profile = self.base_scorer.compute_full_risk_profile(probs)[0]
                    pairwise_risks[key] = risk_profile
        
        return pairwise_risks
    
    def compute_aggregate_risk(
        self,
        pairwise_risks: Dict[Tuple[str, str], Dict[str, any]],
    ) -> Dict[str, any]:
        """
        Compute aggregate risk score for polypharmacy regimen.
        
        Uses maximum risk score among all pairs (conservative approach).
        
        Args:
            pairwise_risks: Dict from compute_pairwise_risks()
            
        Returns:
            Aggregate risk profile
        """
        if not pairwise_risks:
            return {
                "aggregate_risk_score": 0.0,
                "aggregate_risk_level": RiskLevel.LOW.value,
                "highest_risk_pair": None,
                "num_interactions": 0,
            }
        
        # Find maximum risk score (conservative approach)
        max_risk = 0.0
        max_risk_pair = None
        num_moderate_high = 0
        
        for pair, risk_profile in pairwise_risks.items():
            risk_score = risk_profile["risk_score"]
            if risk_score > max_risk:
                max_risk = risk_score
                max_risk_pair = pair
            
            if risk_profile["risk_level"] in [RiskLevel.MODERATE.value, RiskLevel.HIGH.value]:
                num_moderate_high += 1
        
        aggregate_level = self.base_scorer.categorize_risk(max_risk)
        
        return {
            "aggregate_risk_score": max_risk,
            "aggregate_risk_level": aggregate_level.value,
            "highest_risk_pair": max_risk_pair,
            "num_interactions": len(pairwise_risks),
            "num_moderate_high_risk": num_moderate_high,
        }
