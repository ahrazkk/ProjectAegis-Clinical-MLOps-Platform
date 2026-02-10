"""
Drug-Drug Interaction (DDI) Clinical Decision Support System
Model Package Initialization

Architecture based on MCR III Model Specification:
- PubMedBERT Encoder: Text-based DDI prediction
- GNN Encoder: Structure-based DDI prediction from molecular graphs (SMILES)
- Relation Head: DDI classification (binary or multi-class)
- Auxiliary Head: NER for entity boundary learning
- Risk Scoring: Temperature-calibrated severity-weighted scores
"""

# Core model components (PubMedBERT)
from .ddi_model import DDIModel
from .relation_head import RelationHead
from .auxiliary_head import AuxiliaryHead

# GNN model components (Molecular Graph)
from .gnn_model import DDIGraphModel, MolecularGNNEncoder, DDIInteractionHead
from .gnn_featurizer import MolecularGraphFeaturizer
from .gnn_dataset import DDIGraphDataset, create_graph_data_loaders
from .gnn_trainer import GNNTrainer, GNNTrainingConfig
from .gnn_inference import GNNPredictor

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
    parse_vizier_trial
)

# Inference
from .inference import DDIPredictor, DDIPrediction

__all__ = [
    # Core model (PubMedBERT)
    'DDIModel',
    'RelationHead',
    'AuxiliaryHead',

    # GNN model (Molecular Graph)
    'DDIGraphModel',
    'MolecularGNNEncoder',
    'DDIInteractionHead',
    'MolecularGraphFeaturizer',
    'DDIGraphDataset',
    'create_graph_data_loaders',
    'GNNTrainer',
    'GNNTrainingConfig',
    'GNNPredictor',

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
    'DDIPrediction',
    'GNNPredictor',
]

__version__ = '1.1.0'
