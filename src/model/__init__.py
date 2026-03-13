"""
Drug-Drug Interaction (DDI) Clinical Decision Support System
Model Package Initialization

Architecture based on MCR III Model Specification:
- GNN Encoder: Structure-based DDI prediction from molecular graphs (SMILES)
- Edge-Conditioned GIN layers for molecular message passing
- Risk Scoring: Temperature-calibrated severity-weighted scores
"""

# GNN model components
from .gnn_model import DDIGraphModel, MolecularGNNEncoder, DDIInteractionHead
from .gnn_featurizer import MolecularGraphFeaturizer
from .gnn_dataset import DDIGraphDataset, create_graph_data_loaders
from .gnn_trainer import GNNTrainer, GNNTrainingConfig
from .gnn_inference import GNNPredictor, DDIPrediction

# Risk scoring and calibration
from .risk_scorer import RiskScorer, TemperatureScaling

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

__all__ = [
    # GNN model
    'DDIGraphModel',
    'MolecularGNNEncoder',
    'DDIInteractionHead',
    'MolecularGraphFeaturizer',
    'DDIGraphDataset',
    'create_graph_data_loaders',
    'GNNTrainer',
    'GNNTrainingConfig',
    'GNNPredictor',
    'DDIPrediction',

    # Risk scoring
    'RiskScorer',
    'TemperatureScaling',

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
]

__version__ = '2.0.0'
