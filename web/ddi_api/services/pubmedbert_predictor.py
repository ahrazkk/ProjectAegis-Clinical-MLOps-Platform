"""
PubMedBERT-based DDI Prediction Service

This module provides DDI predictions using a fine-tuned PubMedBERT model
trained on the DDI Corpus (~19,000 annotated drug interaction sentences).

The model classifies drug pairs into interaction types:
- no_interaction: No known interaction
- mechanism: Explains HOW drugs interact (CYP450, protein binding, etc.)
- effect: Describes WHAT happens (bleeding, toxicity, etc.)
- advise: Clinical guidance (monitor, adjust dose, avoid)
- int: Generic interaction mention

This approach uses NLP/text understanding rather than molecular structure,
making it more interpretable and aligned with clinical literature.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. PubMedBERT predictions disabled.")


@dataclass
class DDITextPrediction:
    """Result of a PubMedBERT DDI prediction."""
    drug_a: str
    drug_b: str
    interaction_type: str  # mechanism, effect, advise, int, no_interaction
    confidence: float
    risk_score: float
    severity: str
    formatted_input: str
    all_probabilities: Dict[str, float]


class PubMedBERTPredictor:
    """
    PubMedBERT-based DDI Prediction Service.
    
    Uses a fine-tuned BertForSequenceClassification model trained on DDI Corpus
    to predict drug-drug interaction types from text.
    """
    
    # Mapping from DDI Corpus labels to severity/risk
    LABEL_TO_SEVERITY = {
        'no_interaction': ('none', 0.0),
        'int': ('minor', 0.3),
        'advise': ('moderate', 0.5),
        'effect': ('moderate', 0.7),
        'mechanism': ('severe', 0.9),
    }
    
    # Templates for generating context sentences
    CONTEXT_TEMPLATES = [
        "When {drug1} and {drug2} are taken together, drug interactions may occur.",
        "The combination of {drug1} with {drug2} requires clinical consideration.",
        "Patients taking {drug1} should be monitored when also prescribed {drug2}.",
        "{drug1} may interact with {drug2} through various pharmacological mechanisms.",
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize the PubMedBERT predictor.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model = None
        self.tokenizer = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            return
        
        # Find model path
        if model_path is None:
            # Look for DDI_Model_Final at the repo root level
            # Path: web/ddi_api/services/pubmedbert_predictor.py -> web -> molecular-ai -> DDI_PROJECTV2-FRONTEND
            base_dir = Path(__file__).parent.parent.parent.parent.parent  # this_file -> services -> ddi_api -> web -> molecular-ai -> DDI_PROJECTV2-FRONTEND
            model_path = base_dir / 'DDI_Model_Final'
            
            # Fallback: try relative to web directory
            if not model_path.exists():
                alt_path = Path(__file__).parent.parent.parent.parent / 'DDI_Model_Final'  # web/../DDI_Model_Final
                if alt_path.exists():
                    model_path = alt_path
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: Path):
        """Load the fine-tuned model and tokenizer."""
        try:
            logger.info(f"Loading PubMedBERT model from {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            
            # Get label mappings from config
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            
            self.is_loaded = True
            logger.info(f"PubMedBERT model loaded successfully on {self.device}")
            logger.info(f"Labels: {list(self.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load PubMedBERT model: {e}")
            self.is_loaded = False
    
    def _format_input(self, drug1: str, drug2: str, context: Optional[str] = None) -> str:
        """
        Format input text with entity markers for the model.
        
        Args:
            drug1: First drug name
            drug2: Second drug name  
            context: Optional context sentence (will use template if not provided)
            
        Returns:
            Formatted text with <e1>, </e1>, <e2>, </e2> markers
        """
        if context:
            # Use provided context and mark the drugs
            text = context
            # Try to find and mark the drugs in the text
            text = text.replace(drug1, f"<e1>{drug1}</e1>", 1)
            text = text.replace(drug2, f"<e2>{drug2}</e2>", 1)
        else:
            # Generate from template
            template = self.CONTEXT_TEMPLATES[0]
            text = template.format(drug1=f"<e1>{drug1}</e1>", drug2=f"<e2>{drug2}</e2>")
        
        return text
    
    def predict(self, drug1: str, drug2: str, context: Optional[str] = None) -> DDITextPrediction:
        """
        Predict DDI between two drugs using PubMedBERT.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            context: Optional context sentence mentioning both drugs
            
        Returns:
            DDITextPrediction with interaction type, confidence, and risk
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, returning default prediction")
            return DDITextPrediction(
                drug_a=drug1,
                drug_b=drug2,
                interaction_type='unknown',
                confidence=0.0,
                risk_score=0.0,
                severity='unknown',
                formatted_input='',
                all_probabilities={}
            )
        
        # Format input text
        formatted_text = self._format_input(drug1, drug2, context)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)[0]
        probs_dict = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        
        # Get predicted class
        predicted_id = torch.argmax(probs).item()
        predicted_label = self.id2label[predicted_id]
        confidence = float(probs[predicted_id])
        
        # Map to severity and risk score
        severity, base_risk = self.LABEL_TO_SEVERITY.get(predicted_label, ('unknown', 0.5))
        risk_score = base_risk * confidence  # Scale by confidence
        
        return DDITextPrediction(
            drug_a=drug1,
            drug_b=drug2,
            interaction_type=predicted_label,
            confidence=confidence,
            risk_score=risk_score,
            severity=severity,
            formatted_input=formatted_text,
            all_probabilities=probs_dict
        )
    
    def predict_with_multiple_contexts(self, drug1: str, drug2: str, 
                                        contexts: Optional[List[str]] = None) -> DDITextPrediction:
        """
        Predict using multiple context templates and aggregate results.
        
        This can improve robustness by averaging predictions across different
        phrasings of the same drug pair.
        """
        if contexts is None:
            contexts = [t.format(drug1=drug1, drug2=drug2) for t in self.CONTEXT_TEMPLATES]
        
        all_predictions = []
        for ctx in contexts:
            pred = self.predict(drug1, drug2, ctx)
            all_predictions.append(pred)
        
        # Aggregate: average probabilities
        avg_probs = {}
        for label in self.id2label.values():
            avg_probs[label] = sum(p.all_probabilities.get(label, 0) for p in all_predictions) / len(all_predictions)
        
        # Find best label from averaged probs
        best_label = max(avg_probs, key=avg_probs.get)
        avg_confidence = avg_probs[best_label]
        severity, base_risk = self.LABEL_TO_SEVERITY.get(best_label, ('unknown', 0.5))
        
        return DDITextPrediction(
            drug_a=drug1,
            drug_b=drug2,
            interaction_type=best_label,
            confidence=avg_confidence,
            risk_score=base_risk * avg_confidence,
            severity=severity,
            formatted_input=f"[Aggregated from {len(contexts)} contexts]",
            all_probabilities=avg_probs
        )
    
    def get_mechanism_description(self, interaction_type: str, drug1: str, drug2: str, confidence: float = 0.0) -> str:
        """Generate a human-readable mechanism description based on prediction."""
        conf_pct = int(confidence * 100)
        prefix = f"AI Analysis ({conf_pct}% Confidence):"
        
        descriptions = {
            'no_interaction': f"{prefix} No significant pharmacological interaction is predicted between {drug1} and {drug2}.",
            'mechanism': f"{prefix} {drug1} and {drug2} interact through shared metabolic pathways (e.g., CYP450 enzymes) or protein binding competition.",
            'effect': f"{prefix} Taking {drug1} with {drug2} may result in altered therapeutic effects or increased adverse reactions.",
            'advise': f"{prefix} Clinical monitoring is recommended when combining {drug1} with {drug2}. Dose adjustment may be necessary.",
            'int': f"{prefix} A pharmacological interaction is predicted between {drug1} and {drug2}. Consult clinical guidelines.",
        }
        return descriptions.get(interaction_type, f"{prefix} Potential interaction detected between {drug1} and {drug2}.")


# Singleton instance
_pubmedbert_predictor: Optional[PubMedBERTPredictor] = None


def get_pubmedbert_predictor() -> PubMedBERTPredictor:
    """Get or create the PubMedBERT predictor singleton."""
    global _pubmedbert_predictor
    if _pubmedbert_predictor is None:
        _pubmedbert_predictor = PubMedBERTPredictor()
    return _pubmedbert_predictor
