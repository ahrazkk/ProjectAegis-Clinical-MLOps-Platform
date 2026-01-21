"""
DDI Model Inference Module
Provides high-level API for making predictions with trained models
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .ddi_model import DDIModel
from .tokenization import DDITokenizer
from .risk_scorer import RiskScorer, TemperatureScaling


logger = logging.getLogger(__name__)


@dataclass
class DDIPrediction:
    """
    Result of a DDI prediction
    """
    drug1: str
    drug2: str
    has_interaction: bool
    interaction_type: Optional[str]
    raw_probability: float
    calibrated_probability: float
    risk_score: float
    risk_category: str
    confidence: float


class DDIPredictor:
    """
    High-level interface for DDI prediction

    Provides easy-to-use methods for:
    - Loading trained models
    - Making predictions on drug pairs
    - Batch predictions
    - Risk scoring and categorization
    """

    # Interaction type labels
    INTERACTION_TYPES = {
        0: 'none',
        1: 'advice',     # Minor
        2: 'effect',     # Moderate
        3: 'mechanism',  # Major (Pharmacokinetic)
        4: 'int'         # Major (Pharmacodynamic)
    }

    # Class to severity mapping for risk scoring
    CLASS_TO_SEVERITY = {
        0: 'none',
        1: 'minor',
        2: 'moderate',
        3: 'major',
        4: 'major'
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_binary: bool = True
    ):
        """
        Initialize DDI Predictor

        Args:
            model_path: Path to saved model checkpoint
            device: Computation device (cuda/cpu)
            use_binary: Whether model uses binary classification
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_binary = use_binary

        # Initialize tokenizer
        self.tokenizer = DDITokenizer()

        # Initialize model
        self.model = None
        self.temperature_scaling = TemperatureScaling()

        # Initialize risk scorer
        self.risk_scorer = RiskScorer(
            class_to_severity=self.CLASS_TO_SEVERITY
        )

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load model from checkpoint

        Args:
            model_path: Path to model checkpoint file
        """
        logger.info(f"Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})

        # Initialize model with saved config
        self.model = DDIModel(
            encoder_name=config.get('encoder_name',
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
            num_relation_classes=config.get('num_relation_classes', 1),
            num_ner_classes=config.get('num_ner_classes', 3),
            head_dropout_rate=config.get('head_dropout_rate', 0.1),
            use_binary=config.get('use_binary', True)
        )

        # Resize embeddings for special tokens
        self.tokenizer.resize_model_embeddings(self.model)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load temperature scaling
        if 'temperature' in checkpoint:
            self.temperature_scaling.temperature.data = torch.tensor([checkpoint['temperature']])

        self.use_binary = config.get('use_binary', True)

        logger.info("Model loaded successfully")

    def predict(
        self,
        text: str,
        drug1_span: Tuple[int, int],
        drug2_span: Tuple[int, int],
        drug1_name: Optional[str] = None,
        drug2_name: Optional[str] = None
    ) -> DDIPrediction:
        """
        Predict interaction for a single drug pair

        Args:
            text: Clinical text containing drug mentions
            drug1_span: Character span (start, end) for first drug
            drug2_span: Character span (start, end) for second drug
            drug1_name: Optional name of first drug
            drug2_name: Optional name of second drug

        Returns:
            DDIPrediction object with prediction results
        
        Raises:
            ValueError: If drug spans are invalid (out of bounds or overlapping)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate drug spans
        text_len = len(text)
        if drug1_span[0] < 0 or drug1_span[1] > text_len or drug1_span[0] >= drug1_span[1]:
            raise ValueError(
                f"Invalid drug1_span {drug1_span}: must be within text bounds [0, {text_len}] "
                f"and start < end"
            )
        if drug2_span[0] < 0 or drug2_span[1] > text_len or drug2_span[0] >= drug2_span[1]:
            raise ValueError(
                f"Invalid drug2_span {drug2_span}: must be within text bounds [0, {text_len}] "
                f"and start < end"
            )
        
        # Check for overlapping spans
        if not (drug1_span[1] <= drug2_span[0] or drug2_span[1] <= drug1_span[0]):
            raise ValueError(
                f"Drug spans overlap: drug1_span={drug1_span}, drug2_span={drug2_span}. "
                f"Spans must not overlap."
            )

        # Extract drug names from spans if not provided
        if drug1_name is None:
            drug1_name = text[drug1_span[0]:drug1_span[1]]
        if drug2_name is None:
            drug2_name = text[drug2_span[0]:drug2_span[1]]

        # Mark drugs in text
        marked_text = self.tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        # Tokenize
        inputs = self.tokenizer.tokenize(marked_text)

        # Move to device
        input_ids = inputs['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(self.device)
        drug1_mask = inputs['drug1_mask'].unsqueeze(0).to(self.device)
        drug2_mask = inputs['drug2_mask'].unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            relation_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask
            )

        # Calculate probabilities
        if self.use_binary:
            # For binary classification, apply temperature scaling to logits then sigmoid
            scaled_logits = relation_logits / self.temperature_scaling.temperature
            calibrated_prob = torch.sigmoid(scaled_logits).item()
            raw_prob = torch.sigmoid(relation_logits).item()

            has_interaction = calibrated_prob > 0.5
            interaction_type = None if not has_interaction else 'interaction'

            # For binary, risk score is simply the calibrated probability
            risk_score = calibrated_prob
        else:
            raw_probs = torch.softmax(relation_logits, dim=-1)
            calibrated_probs = self.temperature_scaling(relation_logits)

            predicted_class = torch.argmax(raw_probs, dim=-1).item()
            raw_prob = raw_probs[0, int(predicted_class)].item()
            calibrated_prob = calibrated_probs[0, predicted_class].item()

            has_interaction = predicted_class > 0
            interaction_type = self.INTERACTION_TYPES.get(int(predicted_class), 'unknown')

            # Calculate risk score using severity weights
            risk_score = self.risk_scorer.calculate_risk_score(calibrated_probs.cpu().numpy())[0]

        # Categorize risk
        risk_category_result = self.risk_scorer.categorize_risk(risk_score)
        risk_category = risk_category_result if isinstance(risk_category_result, str) else risk_category_result[0]

        # Confidence based on distance from decision boundary
        if self.use_binary:
            confidence = abs(raw_prob - 0.5) * 2
        else:
            confidence = raw_prob  # Max probability as confidence

        return DDIPrediction(
            drug1=drug1_name,
            drug2=drug2_name,
            has_interaction=has_interaction,
            interaction_type=interaction_type,
            raw_probability=float(raw_prob),
            calibrated_probability=float(calibrated_prob),
            risk_score=float(risk_score),
            risk_category=risk_category,
            confidence=float(confidence)
        )

    def predict_batch(
        self,
        samples: List[Dict]
    ) -> List[DDIPrediction]:
        """
        Batch prediction for multiple drug pairs

        Args:
            samples: List of sample dictionaries with keys:
                - text: Clinical text
                - drug1_span: (start, end) tuple
                - drug2_span: (start, end) tuple
                - drug1_name: Optional drug name
                - drug2_name: Optional drug name

        Returns:
            List of DDIPrediction objects
        """
        predictions = []

        for sample in samples:
            pred = self.predict(
                text=sample['text'],
                drug1_span=sample['drug1_span'],
                drug2_span=sample['drug2_span'],
                drug1_name=sample.get('drug1_name'),
                drug2_name=sample.get('drug2_name')
            )
            predictions.append(pred)

        return predictions

    def predict_from_names(
        self,
        drug1_name: str,
        drug2_name: str,
        context_template: Optional[str] = None
    ) -> DDIPrediction:
        """
        Predict interaction from drug names only

        Uses a template to create synthetic context.
        
        Warning: Predictions from drug names alone using this generic template may be less
        reliable than predictions from actual clinical text, as the artificial context may
        not match the model's training data distribution. For best results, provide real
        clinical text context using the predict() method.

        Args:
            drug1_name: Name of first drug
            drug2_name: Name of second drug
            context_template: Optional template string with {drug1} and {drug2} placeholders

        Returns:
            DDIPrediction object
        """
        if context_template is None:
            context_template = (
                "The patient is taking {drug1} and {drug2}. "
                "Evaluate potential drug-drug interaction."
            )
            logger.warning(
                "Using default generic context template for prediction. "
                "Predictions may be less reliable than with actual clinical text. "
                "Consider using the predict() method with real clinical context."
            )

        # Create text with drug positions
        text = context_template.format(drug1=drug1_name, drug2=drug2_name)

        # Find drug spans
        drug1_start = text.find(drug1_name)
        drug1_end = drug1_start + len(drug1_name)
        drug2_start = text.find(drug2_name)
        drug2_end = drug2_start + len(drug2_name)

        return self.predict(
            text=text,
            drug1_span=(drug1_start, drug1_end),
            drug2_span=(drug2_start, drug2_end),
            drug1_name=drug1_name,
            drug2_name=drug2_name
        )

    def get_risk_explanation(self, prediction: DDIPrediction) -> str:
        """
        Generate human-readable explanation of risk assessment

        Args:
            prediction: DDIPrediction object

        Returns:
            Explanation string
        """
        if not prediction.has_interaction:
            return (
                f"No significant drug-drug interaction detected between "
                f"{prediction.drug1} and {prediction.drug2}. "
                f"Confidence: {prediction.confidence:.1%}"
            )

        severity_explanations = {
            'low': "minimal clinical concern",
            'moderate': "monitor patient for potential adverse effects",
            'high': "significant clinical concern - consider alternative therapy"
        }

        explanation = (
            f"Potential drug-drug interaction detected between "
            f"{prediction.drug1} and {prediction.drug2}.\n"
            f"- Risk Level: {prediction.risk_category.upper()}\n"
            f"- Risk Score: {prediction.risk_score:.2f}\n"
            f"- Confidence: {prediction.confidence:.1%}\n"
            f"- Recommendation: {severity_explanations.get(prediction.risk_category, 'Review with clinical pharmacist')}"
        )

        return explanation
