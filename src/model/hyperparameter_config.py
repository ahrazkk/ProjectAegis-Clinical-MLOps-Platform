"""
Hyperparameter Configuration for Vertex AI Vizier
Defines the search space and optimization settings for hyperparameter tuning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json


class ParameterType(Enum):
    """Parameter types for Vizier"""
    DOUBLE = "DOUBLE"
    INTEGER = "INTEGER"
    CATEGORICAL = "CATEGORICAL"


class ScaleType(Enum):
    """Scale types for numerical parameters"""
    LINEAR_SCALE = "LINEAR_SCALE"
    LOG_SCALE = "LOG_SCALE"
    NONE = "NONE"


@dataclass
class ParameterSpec:
    """
    Specification for a single hyperparameter

    Reference: MCR III Section 2 - Hyperparameter Tuning Strategy
    """
    name: str
    param_type: ParameterType
    scale_type: ScaleType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    feasible_values: Optional[List[Any]] = None
    rationale: str = ""

    def to_vizier_spec(self) -> Dict:
        """Convert to Vertex AI Vizier parameter specification format"""
        spec: Dict[str, Any] = {
            "parameterId": self.name,
            "displayName": self.name
        }

        if self.param_type == ParameterType.DOUBLE:
            spec["doubleValueSpec"] = {
                "minValue": float(self.min_value) if self.min_value is not None else 0.0,
                "maxValue": float(self.max_value) if self.max_value is not None else 1.0
            }
            if self.scale_type == ScaleType.LOG_SCALE:
                spec["scaleType"] = "UNIT_LOG_SCALE"
            else:
                spec["scaleType"] = "UNIT_LINEAR_SCALE"

        elif self.param_type == ParameterType.INTEGER:
            spec["integerValueSpec"] = {
                "minValue": int(self.min_value) if self.min_value is not None else 0,
                "maxValue": int(self.max_value) if self.max_value is not None else 100
            }
            spec["scaleType"] = "UNIT_LINEAR_SCALE"

        elif self.param_type == ParameterType.CATEGORICAL:
            if self.feasible_values is not None:
                spec["categoricalValueSpec"] = {
                    "values": [str(v) for v in self.feasible_values]
                }
            else:
                raise ValueError(f"Categorical parameter '{self.name}' requires feasible_values")
            spec["scaleType"] = "UNIT_LINEAR_SCALE"

        return spec


@dataclass
class VizierStudyConfig:
    """
    Complete configuration for a Vertex AI Vizier hyperparameter study

    Reference: MCR III Section 2
    """
    study_display_name: str = "ddi_model_hpo"
    optimization_goal: str = "MAXIMIZE"  # Maximize PR-AUC
    optimization_metric: str = "pr_auc"
    algorithm: str = "ALGORITHM_UNSPECIFIED"  # Bayesian Optimization (default)
    max_trial_count: int = 50
    parallel_trial_count: int = 3

    # Search space defined as per MCR III specification
    parameters: List[ParameterSpec] = field(default_factory=lambda: [
        ParameterSpec(
            name="learning_rate",
            param_type=ParameterType.DOUBLE,
            scale_type=ScaleType.LOG_SCALE,
            min_value=1e-6,
            max_value=5e-5,
            rationale="Transformer stability - small learning rates required"
        ),
        ParameterSpec(
            name="batch_size",
            param_type=ParameterType.CATEGORICAL,
            scale_type=ScaleType.NONE,
            feasible_values=[8, 16, 32],
            rationale="VRAM constraints vs. Gradient stability trade-off"
        ),
        ParameterSpec(
            name="weight_decay",
            param_type=ParameterType.DOUBLE,
            scale_type=ScaleType.LINEAR_SCALE,
            min_value=0.0,
            max_value=0.1,
            rationale="AdamW L2 regularization parameter"
        ),
        ParameterSpec(
            name="num_warmup_steps",
            param_type=ParameterType.INTEGER,
            scale_type=ScaleType.LINEAR_SCALE,
            min_value=100,
            max_value=1000,
            rationale="Prevent early-stage training instability"
        ),
        ParameterSpec(
            name="head_dropout_rate",
            param_type=ParameterType.DOUBLE,
            scale_type=ScaleType.LINEAR_SCALE,
            min_value=0.1,
            max_value=0.3,
            rationale="Regularization for custom relation head"
        ),
        ParameterSpec(
            name="aux_loss_weight",
            param_type=ParameterType.DOUBLE,
            scale_type=ScaleType.LINEAR_SCALE,
            min_value=0.2,
            max_value=0.8,
            rationale="Balance between Primary (DDI) and Auxiliary (NER) tasks"
        )
    ])

    def to_vizier_study_spec(self) -> Dict:
        """
        Generate complete Vertex AI Vizier study specification

        Returns:
            Dictionary compatible with Vertex AI Vizier API
        """
        return {
            "displayName": self.study_display_name,
            "studySpec": {
                "algorithm": self.algorithm,
                "metrics": [{
                    "metricId": self.optimization_metric,
                    "goal": self.optimization_goal
                }],
                "parameters": [p.to_vizier_spec() for p in self.parameters]
            },
            "maxTrialCount": self.max_trial_count,
            "parallelTrialCount": self.parallel_trial_count
        }

    def to_json(self, filepath: str):
        """Save study configuration to JSON file"""
        config = self.to_vizier_study_spec()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'VizierStudyConfig':
        """Load study configuration from JSON file"""
        with open(filepath, 'r') as f:
            json.load(f)
        # Parse and return config (simplified for now)
        return cls()


def vizier_trial_to_training_config(trial_params: Dict) -> Dict:
    """
    Convert Vizier trial parameters to training config format

    Args:
        trial_params: Parameters from Vizier trial

    Returns:
        Dictionary of training hyperparameters
    """
    return {
        'learning_rate': float(trial_params.get('learning_rate', 2e-5)),
        'batch_size': int(trial_params.get('batch_size', 16)),
        'weight_decay': float(trial_params.get('weight_decay', 0.01)),
        'num_warmup_steps': int(trial_params.get('num_warmup_steps', 500)),
        'head_dropout_rate': float(trial_params.get('head_dropout_rate', 0.1)),
        'aux_loss_weight': float(trial_params.get('aux_loss_weight', 0.5))
    }


def get_default_search_space() -> Dict[str, Any]:
    """
    Get the default hyperparameter search space

    Returns:
        Dictionary describing the search space
    """
    return {
        'learning_rate': {
            'type': 'double',
            'scale': 'log',
            'range': [1e-6, 5e-5],
            'default': 2e-5
        },
        'batch_size': {
            'type': 'categorical',
            'values': [8, 16, 32],
            'default': 16
        },
        'weight_decay': {
            'type': 'double',
            'scale': 'linear',
            'range': [0.0, 0.1],
            'default': 0.01
        },
        'num_warmup_steps': {
            'type': 'integer',
            'scale': 'linear',
            'range': [100, 1000],
            'default': 500
        },
        'head_dropout_rate': {
            'type': 'double',
            'scale': 'linear',
            'range': [0.1, 0.3],
            'default': 0.1
        },
        'aux_loss_weight': {
            'type': 'double',
            'scale': 'linear',
            'range': [0.2, 0.8],
            'default': 0.5
        }
    }


# Pre-configured study for DDI model
DDI_VIZIER_STUDY = VizierStudyConfig(
    study_display_name="ddi_pubmedbert_hpo",
    optimization_goal="MAXIMIZE",
    optimization_metric="pr_auc",
    max_trial_count=50,
    parallel_trial_count=3
)
