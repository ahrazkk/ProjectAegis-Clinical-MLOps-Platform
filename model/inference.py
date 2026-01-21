"""
Inference Utility for DDI Relation Extraction Model
Simple interface for making predictions on new text
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .ddi_model import DDIRelationModel, ModelWithTemperature
from .data_preprocessor import DDIDataPreprocessor
from .risk_scorer import RiskScorer, InteractionType, RiskLevel

logger = logging.getLogger(__name__)


class DDIPredictor:
    """
    High-level interface for DDI prediction with risk scoring.
    
    Example usage:
        predictor = DDIPredictor.from_pretrained("path/to/checkpoint.pt")
        result = predictor.predict(
            text="The interaction between aspirin and warfarin may increase bleeding risk.",
            drug1_span=(26, 33),
            drug2_span=(38, 46)
        )
        print(result)
    """
    
    def __init__(
        self,
        model: DDIRelationModel,
        preprocessor: DDIDataPreprocessor,
        risk_scorer: RiskScorer,
        use_calibration: bool = True,
        temperature: float = 1.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Trained DDIRelationModel
            preprocessor: DDIDataPreprocessor instance
            risk_scorer: RiskScorer instance
            use_calibration: Whether to use temperature scaling
            temperature: Temperature value for calibration
            device: Device to run inference on
            class_names: List of interaction class names
        """
        self.device = device
        
        # Wrap model with temperature scaling if needed
        if use_calibration:
            self.model = ModelWithTemperature(model, temperature=temperature).to(device)
        else:
            self.model = model.to(device)
        
        self.model.eval()
        
        self.preprocessor = preprocessor
        self.risk_scorer = risk_scorer
        self.use_calibration = use_calibration
        
        # Default class names for DDIExtraction 2013
        self.class_names = class_names or [
            "None",
            "Mechanism",
            "Effect",
            "Advise",
            "Int",
        ]
        
        # Map to InteractionType enum
        self.interaction_types = [
            InteractionType.NONE,
            InteractionType.MECHANISM,
            InteractionType.EFFECT,
            InteractionType.ADVISE,
            InteractionType.INT,
        ]
    
    def predict(
        self,
        text: str,
        drug1_span: Tuple[int, int],
        drug2_span: Tuple[int, int],
        return_details: bool = True,
    ) -> Dict:
        """
        Predict DDI for a drug pair and compute risk score.
        
        Args:
            text: Input text containing drug mentions
            drug1_span: (start, end) character offsets for first drug
            drug2_span: (start, end) character offsets for second drug
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary with prediction results and risk assessment
        """
        # Preprocess input
        encoding = self.preprocessor.encode_single(text, drug1_span, drug2_span)
        
        # Move to device and add batch dimension
        input_ids = encoding['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].unsqueeze(0).to(self.device)
        drug1_mask = encoding['drug1_mask'].unsqueeze(0).to(self.device)
        drug2_mask = encoding['drug2_mask'].unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if self.use_calibration:
                pred_outputs = self.model.predict_calibrated(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drug1_mask=drug1_mask,
                    drug2_mask=drug2_mask,
                )
                relation_probs = pred_outputs['relation_probs_calibrated']
            else:
                pred_outputs = self.model.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drug1_mask=drug1_mask,
                    drug2_mask=drug2_mask,
                )
                relation_probs = pred_outputs['relation_probs']
        
        # Compute risk profile
        risk_profiles = self.risk_scorer.compute_full_risk_profile(
            relation_probs,
            self.interaction_types,
        )
        risk_profile = risk_profiles[0]
        
        # Extract drug text
        drug1_text = text[drug1_span[0]:drug1_span[1]]
        drug2_text = text[drug2_span[0]:drug2_span[1]]
        
        # Build result
        result = {
            "drug_pair": (drug1_text, drug2_text),
            "predicted_interaction": risk_profile['predicted_interaction'],
            "interaction_probability": risk_profile['interaction_probability'],
            "risk_score": risk_profile['risk_score'],
            "risk_level": risk_profile['risk_level'],
        }
        
        # Add clinical recommendation
        risk_level = RiskLevel[risk_profile['risk_level'].upper()]
        result["recommendation"] = RiskScorer.get_clinical_recommendation(
            risk_level,
            risk_profile['predicted_interaction']
        )
        
        # Add detailed information if requested
        if return_details:
            result["all_probabilities"] = risk_profile['all_probabilities']
            result["input_text"] = text
            result["drug1_span"] = drug1_span
            result["drug2_span"] = drug2_span
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        drug1_spans: List[Tuple[int, int]],
        drug2_spans: List[Tuple[int, int]],
    ) -> List[Dict]:
        """
        Predict DDI for multiple drug pairs.
        
        Args:
            texts: List of input texts
            drug1_spans: List of (start, end) tuples for first drugs
            drug2_spans: List of (start, end) tuples for second drugs
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for text, d1_span, d2_span in zip(texts, drug1_spans, drug2_spans):
            result = self.predict(text, d1_span, d2_span)
            results.append(result)
        return results
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_calibration: bool = True,
        allow_untrained: bool = False,
    ):
        """
        Load predictor from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to load model on
            use_calibration: Whether to use temperature scaling
            allow_untrained: If True, allows loading without trained weights
                           (for testing only - will issue warning)
            
        Returns:
            DDIPredictor instance
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_file = Path(checkpoint_path)
        
        # Check if checkpoint exists
        if not checkpoint_file.exists():
            if allow_untrained:
                logger.warning(
                    "⚠️  WARNING: Checkpoint file not found. "
                    "Creating predictor with UNTRAINED model. "
                    "Predictions will be unreliable! "
                    "This should only be used for testing purposes."
                )
                # Create untrained model
                model = DDIRelationModel(num_relation_classes=5, num_ner_classes=5)
                preprocessor = DDIDataPreprocessor()
                model.encoder.resize_token_embeddings(len(preprocessor.tokenizer))
                risk_scorer = RiskScorer(use_calibrated_probs=use_calibration)
                
                return cls(
                    model=model,
                    preprocessor=preprocessor,
                    risk_scorer=risk_scorer,
                    use_calibration=use_calibration,
                    temperature=1.5,
                    device=device,
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}. "
                    f"If you want to test with an untrained model, set allow_untrained=True"
                )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract hyperparameters
        hyperparams = checkpoint.get('hyperparameters', {})
        
        # Initialize model
        model = DDIRelationModel(
            num_relation_classes=5,
            num_ner_classes=5,
            head_dropout_rate=hyperparams.get('head_dropout_rate', 0.1),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize preprocessor
        preprocessor = DDIDataPreprocessor()
        
        # Resize model embeddings if needed (for marker tokens)
        model.encoder.resize_token_embeddings(len(preprocessor.tokenizer))
        
        # Initialize risk scorer
        risk_scorer = RiskScorer(use_calibrated_probs=use_calibration)
        
        # Get temperature from checkpoint if available
        temperature = 1.5
        if 'temperature' in checkpoint:
            temperature = checkpoint['temperature']
        
        return cls(
            model=model,
            preprocessor=preprocessor,
            risk_scorer=risk_scorer,
            use_calibration=use_calibration,
            temperature=temperature,
            device=device,
        )
    
    def save_prediction_report(
        self,
        predictions: List[Dict],
        output_path: str,
        format: str = "json",
    ):
        """
        Save predictions to file.
        
        Args:
            predictions: List of prediction results
            output_path: Path to save file
            format: Output format ("json" or "csv")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
        
        elif format == "csv":
            import csv
            
            # Flatten predictions for CSV
            rows = []
            for pred in predictions:
                row = {
                    "drug1": pred['drug_pair'][0],
                    "drug2": pred['drug_pair'][1],
                    "predicted_interaction": pred['predicted_interaction'],
                    "interaction_probability": pred['interaction_probability'],
                    "risk_score": pred['risk_score'],
                    "risk_level": pred['risk_level'],
                    "recommendation": pred['recommendation'],
                }
                rows.append(row)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(predictions)} predictions to {output_path}")


def predict_from_text(
    text: str,
    drug1: str,
    drug2: str,
    model_path: Optional[str] = None,
    allow_untrained: bool = False,
) -> Dict:
    """
    Convenience function to predict DDI from text and drug names.
    
    Automatically finds drug spans in text.
    
    Args:
        text: Input text containing drug mentions
        drug1: Name of first drug
        drug2: Name of second drug
        model_path: Path to model checkpoint (if None, uses untrained model with warning)
        allow_untrained: Must be True to use untrained model (for testing only)
        
    Returns:
        Prediction result dictionary
        
    Raises:
        ValueError: If drugs not found in text or if untrained model used without permission
    """
    # Find drug spans
    drug1_start = text.lower().find(drug1.lower())
    drug2_start = text.lower().find(drug2.lower())
    
    if drug1_start == -1 or drug2_start == -1:
        raise ValueError(f"Could not find drugs '{drug1}' or '{drug2}' in text")
    
    drug1_span = (drug1_start, drug1_start + len(drug1))
    drug2_span = (drug2_start, drug2_start + len(drug2))
    
    # Create predictor
    if model_path is not None:
        predictor = DDIPredictor.from_pretrained(model_path)
    else:
        if not allow_untrained:
            raise ValueError(
                "No model_path provided. To use an untrained model for testing, "
                "set allow_untrained=True. Note: Predictions will be unreliable."
            )
        
        logger.warning(
            "⚠️  WARNING: Using UNTRAINED model. "
            "Predictions will be unreliable! "
            "For production use, provide a model_path to a trained checkpoint."
        )
        
        # Use untrained model (for testing)
        from .ddi_model import DDIRelationModel
        model = DDIRelationModel()
        preprocessor = DDIDataPreprocessor()
        model.encoder.resize_token_embeddings(len(preprocessor.tokenizer))
        risk_scorer = RiskScorer()
        predictor = DDIPredictor(model, preprocessor, risk_scorer)
    
    # Make prediction
    return predictor.predict(text, drug1_span, drug2_span)
