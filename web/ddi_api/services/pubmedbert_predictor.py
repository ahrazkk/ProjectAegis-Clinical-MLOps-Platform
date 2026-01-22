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

# Import retriever for RAG mode
try:
    from django.conf import settings as django_settings
    from .pubmed_retriever import get_retriever, RetrievedContext
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("PubMed retriever not available. Using template-only mode.")

# Import DDI Sentence Database for real sentences
try:
    from .ddi_sentence_db import get_ddi_sentence_db, DDISentence
    DDI_SENTENCE_DB_AVAILABLE = True
except ImportError:
    DDI_SENTENCE_DB_AVAILABLE = False
    logger.warning("DDI Sentence DB not available. Using template-only mode.")


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
    # New fields for context transparency
    context_sentence: str = ''  # The actual sentence used for prediction
    context_source: str = 'template'  # 'template', 'rag', 'user_provided'
    template_category: str = ''  # 'effect', 'mechanism', 'advise', 'neutral', etc.


class PubMedBERTPredictor:
    """
    PubMedBERT-based DDI Prediction Service.
    
    Uses a fine-tuned BertForSequenceClassification model trained on DDI Corpus
    to predict drug-drug interaction types from text.
    """
    
    # Mapping from DDI Corpus labels to severity/risk
    # Updated based on clinical significance analysis
    LABEL_TO_SEVERITY = {
        'no_interaction': ('none', 0.0),
        'int': ('moderate', 0.4),      # Generic interaction - could be moderate
        'advise': ('moderate', 0.6),   # Clinical guidance needed - moderate concern
        'effect': ('major', 0.75),     # Describes adverse effects - major concern
        'mechanism': ('severe', 0.85), # CYP450/pathway interaction - severe
    }
    
    # DDI Corpus-style context templates
    # These use language patterns similar to the training data for better predictions
    # The model was trained to recognize these patterns and classify accordingly
    CONTEXT_TEMPLATES = {
        # Effect-focused templates (trained on sentences describing clinical effects)
        'effect': [
            "The concomitant use of {drug1} with {drug2} may result in enhanced pharmacological effects and increased clinical risk.",
            "Concurrent administration of {drug1} and {drug2} significantly increases the risk of serious adverse effects.",
            "{drug1} may potentiate the effects of {drug2}, leading to increased toxicity.",
        ],
        # Mechanism-focused templates (trained on CYP450, metabolism sentences)
        'mechanism': [
            "{drug1} inhibits the metabolism of {drug2} through CYP450 enzyme interaction, increasing plasma concentrations.",
            "The pharmacokinetic interaction between {drug1} and {drug2} involves hepatic enzyme inhibition.",
            "{drug1} affects the clearance of {drug2} by modulating drug-metabolizing enzymes.",
        ],
        # Advise-focused templates (trained on clinical guidance sentences)
        'advise': [
            "Patients taking {drug1} should be carefully monitored when concomitantly prescribed {drug2}, with dose adjustment if necessary.",
            "Clinical caution is advised when combining {drug1} with {drug2}; therapeutic drug monitoring may be required.",
            "Healthcare providers should consider alternative therapy when {drug1} is used with {drug2}.",
        ],
        # Neutral/generic templates (minimal guidance to model)
        'neutral': [
            "When {drug1} and {drug2} are taken together, drug interactions may occur.",
            "The combination of {drug1} with {drug2} requires clinical consideration.",
        ],
    }
    
    # Flattened list for backward compatibility
    CONTEXT_TEMPLATES_FLAT = [
        "The concomitant use of {drug1} with {drug2} may result in enhanced pharmacological effects and increased clinical risk.",
        "{drug1} inhibits the metabolism of {drug2} through CYP450 enzyme interaction.",
        "Patients taking {drug1} should be carefully monitored when concomitantly prescribed {drug2}.",
        "When {drug1} and {drug2} are taken together, drug interactions may occur.",
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
            # Docker: Model is mounted at /app/DDI_Model_Final
            docker_path = Path('/app/DDI_Model_Final')
            if docker_path.exists():
                model_path = docker_path
            else:
                # Local dev: Look for DDI_Model_Final at the repo root level
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
            # Check if context already has entity markers
            has_e1 = '<e1>' in context and '</e1>' in context
            has_e2 = '<e2>' in context and '</e2>' in context
            
            if has_e1 and has_e2:
                # Already has markers, use as-is
                return context
            
            text = context
            
            # Case-insensitive replacement for drug names
            import re
            
            if not has_e1:
                # Mark drug1 - try exact match first, then case-insensitive
                if drug1 in text:
                    text = text.replace(drug1, f"<e1>{drug1}</e1>", 1)
                else:
                    pattern = re.compile(re.escape(drug1), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        text = text[:match.start()] + f"<e1>{match.group()}</e1>" + text[match.end():]
                    else:
                        # Drug not found in context - prepend it
                        text = f"<e1>{drug1}</e1>: " + text
            
            if not has_e2:
                # Mark drug2
                if drug2 in text:
                    text = text.replace(drug2, f"<e2>{drug2}</e2>", 1)
                else:
                    pattern = re.compile(re.escape(drug2), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        text = text[:match.start()] + f"<e2>{match.group()}</e2>" + text[match.end():]
                    else:
                        # Drug not found in context - append it
                        text = text + f" with <e2>{drug2}</e2>."
            
            return text
        else:
            # Generate from DDI Corpus-style template (effect-focused for better detection)
            template = self.CONTEXT_TEMPLATES['effect'][0]
            text = template.format(drug1=f"<e1>{drug1}</e1>", drug2=f"<e2>{drug2}</e2>")
        
        return text
    
    def _predict_single(self, formatted_text: str, drug1: str, drug2: str, 
                         context_sentence: str = '', context_source: str = 'template',
                         template_category: str = '') -> DDITextPrediction:
        """
        Run a single prediction on pre-formatted text.
        
        This is the core inference method - takes formatted text with entity markers
        and returns a prediction.
        """
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
            all_probabilities=probs_dict,
            context_sentence=context_sentence or formatted_text,
            context_source=context_source,
            template_category=template_category
        )
    
    def predict(self, drug1: str, drug2: str, context: Optional[str] = None, 
                use_ensemble: bool = True) -> DDITextPrediction:
        """
        Predict DDI between two drugs using PubMedBERT.
        
        Strategy (in order of preference):
        1. If explicit context provided → use it directly
        2. Check DDI Sentence Database for real sentences → highest accuracy
        3. Use ensemble prediction with templates → reliable fallback
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            context: Optional context sentence mentioning both drugs
            use_ensemble: If True, use multiple templates when no context (default: True)
            
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
        
        # =====================================================================
        # If explicit context provided, use single prediction
        # =====================================================================
        if context is not None:
            formatted_text = self._format_input(drug1, drug2, context)
            # Remove entity markers for display
            import re
            display_context = re.sub(r'</?e[12]>', '', context)
            return self._predict_single(
                formatted_text, drug1, drug2,
                context_sentence=display_context,
                context_source='user_provided',
                template_category='custom'
            )
        
        # =====================================================================
        # Check DDI Sentence Database for real sentences (highest accuracy)
        # =====================================================================
        if DDI_SENTENCE_DB_AVAILABLE:
            try:
                db = get_ddi_sentence_db()
                ddi_sentence = db.find_sentence(drug1, drug2)
                
                if ddi_sentence:
                    logger.info(f"Found DDI sentence for {drug1}-{drug2} from {ddi_sentence.source}")
                    formatted_text = self._format_input(drug1, drug2, ddi_sentence.sentence)
                    prediction = self._predict_single(
                        formatted_text, drug1, drug2,
                        context_sentence=ddi_sentence.sentence,
                        context_source=f'ddi_corpus_{ddi_sentence.source}',
                        template_category=ddi_sentence.interaction_type
                    )
                    # Boost confidence for real sentences
                    boosted_confidence = min(0.99, prediction.confidence * ddi_sentence.confidence)
                    return DDITextPrediction(
                        drug_a=prediction.drug_a,
                        drug_b=prediction.drug_b,
                        interaction_type=prediction.interaction_type,
                        confidence=boosted_confidence,
                        risk_score=prediction.risk_score * (boosted_confidence / prediction.confidence),
                        severity=prediction.severity,
                        formatted_input=prediction.formatted_input,
                        all_probabilities=prediction.all_probabilities,
                        context_sentence=ddi_sentence.sentence,
                        context_source=f'ddi_corpus ({ddi_sentence.source})',
                        template_category=ddi_sentence.interaction_type
                    )
            except Exception as e:
                logger.warning(f"DDI Sentence DB lookup failed: {e}")
        
        # =====================================================================
        # Use ensemble prediction with templates (reliable fallback)
        # =====================================================================
        if use_ensemble:
            return self.predict_ensemble(drug1, drug2)
        else:
            # Fallback to single template (effect-focused)
            template = self.CONTEXT_TEMPLATES['effect'][0]
            display_context = template.format(drug1=drug1, drug2=drug2)
            formatted_text = self._format_input(drug1, drug2, None)
            return self._predict_single(
                formatted_text, drug1, drug2,
                context_sentence=display_context,
                context_source='template',
                template_category='effect'
            )
    
    def predict_ensemble(self, drug1: str, drug2: str) -> DDITextPrediction:
        """
        Predict using ensemble of DDI Corpus-style templates.
        
        Strategy: Use the EFFECT template as primary (most clinically relevant),
        and use other templates to validate the prediction.
        
        The effect template asks "do these drugs cause adverse effects together?"
        which is the most important clinical question for DDI.
        """
        # Primary prediction: Use effect-focused template
        primary_template = self.CONTEXT_TEMPLATES['effect'][0]
        formatted = primary_template.format(
            drug1=f"<e1>{drug1}</e1>", 
            drug2=f"<e2>{drug2}</e2>"
        )
        primary_pred = self._predict_single(formatted, drug1, drug2)
        
        # Secondary validations with other templates
        secondary_templates = [
            self.CONTEXT_TEMPLATES['mechanism'][0],
            self.CONTEXT_TEMPLATES['advise'][0],
        ]
        
        secondary_preds = []
        for template in secondary_templates:
            formatted = template.format(
                drug1=f"<e1>{drug1}</e1>", 
                drug2=f"<e2>{drug2}</e2>"
            )
            pred = self._predict_single(formatted, drug1, drug2)
            secondary_preds.append(pred)
        
        # Count how many predictions show interaction (not no_interaction)
        all_preds = [primary_pred] + secondary_preds
        interaction_count = sum(
            1 for p in all_preds 
            if p.interaction_type != 'no_interaction'
        )
        consensus = interaction_count / len(all_preds)
        
        # Use primary prediction result with consensus-adjusted confidence
        if consensus >= 0.67:  # 2/3 or more agree there's an interaction
            confidence_boost = 1.1
        elif consensus >= 0.33:  # At least 1/3 agree
            confidence_boost = 1.0
        else:
            # Majority say no interaction - trust them
            confidence_boost = 0.7
        
        final_confidence = min(0.99, primary_pred.confidence * confidence_boost)
        
        severity, base_risk = self.LABEL_TO_SEVERITY.get(
            primary_pred.interaction_type, ('unknown', 0.5)
        )
        
        # Create human-readable context sentence (without entity markers for display)
        display_context = primary_template.format(drug1=drug1, drug2=drug2)
        
        return DDITextPrediction(
            drug_a=drug1,
            drug_b=drug2,
            interaction_type=primary_pred.interaction_type,
            confidence=final_confidence,
            risk_score=base_risk * final_confidence,
            severity=severity,
            formatted_input=f"[Ensemble: 3 templates, consensus: {consensus:.0%}, primary: {primary_pred.interaction_type}]",
            all_probabilities=primary_pred.all_probabilities,
            context_sentence=display_context,
            context_source='template',
            template_category='effect (ensemble)'
        )
    
    def predict_with_multiple_contexts(self, drug1: str, drug2: str, 
                                        contexts: Optional[List[str]] = None) -> DDITextPrediction:
        """
        Predict using multiple context templates and aggregate results.
        
        This can improve robustness by averaging predictions across different
        phrasings of the same drug pair.
        
        DEPRECATED: Use predict_ensemble() instead for better results.
        """
        if contexts is None:
            # Use flat list of templates
            contexts = [t.format(
                drug1=f"<e1>{drug1}</e1>", 
                drug2=f"<e2>{drug2}</e2>"
            ) for t in self.CONTEXT_TEMPLATES_FLAT]
        
        all_predictions = []
        for ctx in contexts:
            pred = self.predict(drug1, drug2, ctx, use_ensemble=False)
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
