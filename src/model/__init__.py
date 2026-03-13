"""
Drug-Drug Interaction (DDI) Clinical Decision Support System
Model Package Initialization

Architecture based on MCR III Model Specification:
- Encoder: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- Relation Head: DDI classification (binary or multi-class)
- Auxiliary Head: NER for entity boundary learning
- Risk Scoring: Temperature-calibrated severity-weighted scores
"""

# Core model components (requires transformers — optional for GNN-only usage)
try:
    from .ddi_model import DDIModel
    from .relation_head import RelationHead
    from .auxiliary_head import AuxiliaryHead
    from .risk_scorer import RiskScorer, TemperatureScaling
    from .tokenization import DDITokenizer
    from .dataset import DDIDataset, create_data_loaders
    from .trainer import DDITrainer, TrainingConfig
    from .evaluation import (
        calculate_metrics,
        calculate_pr_auc,
        evaluate_model,
        StratifiedKFoldValidator,
        ErrorAnalyzer,
        ErrorType
    )
    from .hyperparameter_config import (
        VizierStudyConfig,
        ParameterSpec,
        DDI_VIZIER_STUDY,
        get_default_search_space,
        parse_vizier_trial
    )
    from .inference import DDIPredictor, DDIPrediction
except (ImportError, AttributeError):
    # transformers/torch version mismatch — GNN modules still work
    pass

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
    'parse_vizier_trial',

    # Inference
    'DDIPredictor',
    'DDIPrediction'
]

__version__ = '1.0.0'

