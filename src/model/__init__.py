"""
Drug-Drug Interaction (DDI) Clinical Decision Support System
Model Package Initialization

Architecture based on MCR III Model Specification:
- Encoder: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- Relation Head: DDI classification (binary or multi-class)
- Auxiliary Head: NER for entity boundary learning
- Risk Scoring: Temperature-calibrated severity-weighted scores
"""

# Core model components
from .ddi_model import DDIModel
from .relation_head import RelationHead
from .auxiliary_head import AuxiliaryHead

# Risk scoring and calibration
from .risk_scorer import RiskScorer, TemperatureScaling

# Tokenization
from .tokenization import DDITokenizer

# Dataset handling
from .dataset import DDIDataset, create_data_loaders

# Training
from .trainer import DDITrainer, TrainingConfig

# Evaluation
from .evaluation import (
    calculate_metrics,
    calculate_pr_auc,
    evaluate_model,
    StratifiedKFoldValidator,
    ErrorAnalyzer,
    ErrorType
)

# Hyperparameter tuning
from .hyperparameter_config import (
    VizierStudyConfig,
    ParameterSpec,
    DDI_VIZIER_STUDY,
    get_default_search_space,
    vizier_trial_to_training_config
)

# Inference
from .inference import DDIPredictor, DDIPrediction

__all__ = [
    # Core model
    'DDIModel',
    'RelationHead',
    'AuxiliaryHead',

    # Risk scoring
    'RiskScorer',
    'TemperatureScaling',

    # Tokenization
    'DDITokenizer',

    # Dataset
    'DDIDataset',
    'create_data_loaders',

    # Training
    'DDITrainer',
    'TrainingConfig',

    # Evaluation
    'calculate_metrics',
    'calculate_pr_auc',
    'evaluate_model',
    'StratifiedKFoldValidator',
    'ErrorAnalyzer',
    'ErrorType',

    # Hyperparameter tuning
    'VizierStudyConfig',
    'ParameterSpec',
    'DDI_VIZIER_STUDY',
    'get_default_search_space',
    'vizier_trial_to_training_config',

    # Inference
    'DDIPredictor',
    'DDIPrediction'
]

__version__ = '1.0.0'

