"""
Project Aegis: DDI Prediction Model
PubMedBERT-based Drug-Drug Interaction Relation Extraction Model
"""

from .ddi_model import DDIRelationModel, ModelWithTemperature
from .risk_scorer import RiskScorer, InteractionType, RiskLevel
from .data_preprocessor import DDIDataPreprocessor, DDIDataset
from .trainer import DDITrainer
from .evaluator import DDIEvaluator
from .inference import DDIPredictor, predict_from_text
from .config import get_config, MODEL_CONFIG, TRAINING_CONFIG

__version__ = "1.0.0"

__all__ = [
    "DDIRelationModel",
    "ModelWithTemperature",
    "RiskScorer",
    "InteractionType",
    "RiskLevel",
    "DDIDataPreprocessor",
    "DDIDataset",
    "DDITrainer",
    "DDIEvaluator",
    "DDIPredictor",
    "predict_from_text",
    "get_config",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
]
