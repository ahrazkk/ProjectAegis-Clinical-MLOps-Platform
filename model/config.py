"""
Hyperparameter Configuration for DDI Model Training
Based on Section 2 of the AI Model Specification
"""

# Model Architecture Configuration
MODEL_CONFIG = {
    # Base model from HuggingFace
    "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    
    # Task-specific parameters
    "num_relation_classes": 5,  # None, Mechanism, Effect, Advise, Int
    "num_ner_classes": 5,  # O, B-DRUG, I-DRUG, B-BRAND, I-BRAND
    
    # Architecture parameters
    "relation_hidden_dim": 768,  # BERT hidden size
    "freeze_encoder": False,  # Whether to freeze BERT weights
    
    # Input parameters
    "max_length": 512,  # Maximum sequence length
}


# Hyperparameter Search Space (Section 2)
# For Bayesian Optimization (e.g., Vertex AI Vizier)
HYPERPARAMETER_SEARCH_SPACE = {
    "learning_rate": {
        "type": "DOUBLE",
        "scaling": "LOG_SCALE",
        "min_value": 1e-6,
        "max_value": 5e-5,
        "default": 2e-5,
    },
    "batch_size": {
        "type": "CATEGORICAL",
        "values": [8, 16, 32],
        "default": 16,
    },
    "weight_decay": {
        "type": "DOUBLE",
        "scaling": "LINEAR_SCALE",
        "min_value": 0.0,
        "max_value": 0.1,
        "default": 0.01,
    },
    "num_warmup_steps": {
        "type": "INTEGER",
        "scaling": "LINEAR_SCALE",
        "min_value": 100,
        "max_value": 1000,
        "default": 500,
    },
    "head_dropout_rate": {
        "type": "DOUBLE",
        "scaling": "LINEAR_SCALE",
        "min_value": 0.1,
        "max_value": 0.3,
        "default": 0.1,
    },
    "aux_loss_weight": {
        "type": "DOUBLE",
        "scaling": "LINEAR_SCALE",
        "min_value": 0.2,
        "max_value": 0.8,
        "default": 0.5,
    },
}


# Default Training Configuration
TRAINING_CONFIG = {
    # Optimizer parameters
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    
    # Learning rate schedule
    "num_warmup_steps": 500,
    "lr_schedule": "linear_with_warmup",
    
    # Training parameters
    "num_epochs": 10,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    
    # Task weighting
    "aux_loss_weight": 0.5,  # Weight for auxiliary NER loss
    
    # Dropout
    "head_dropout_rate": 0.1,
    
    # Early stopping
    "early_stopping_patience": 3,
    
    # Evaluation
    "eval_steps": None,  # Evaluate every N steps (None = per epoch)
    
    # Mixed precision training (optional)
    "fp16": False,
}


# Risk Scoring Configuration (Section 3)
RISK_SCORING_CONFIG = {
    # Temperature scaling
    "use_temperature_scaling": True,
    "initial_temperature": 1.5,
    
    # Severity weights (Section 3.2)
    "severity_weights": {
        "NONE": 0.0,
        "ADVISE": 0.2,  # Minor
        "EFFECT": 0.6,  # Moderate
        "MECHANISM": 0.9,  # Major (Pharmacokinetic)
        "INT": 0.5,  # General interaction
    },
    
    # Risk thresholds (Section 3.3)
    "risk_thresholds": {
        "LOW": (0.0, 0.3),
        "MODERATE": (0.3, 0.7),
        "HIGH": (0.7, 1.0),
    },
}


# Evaluation Configuration (Section 4)
EVALUATION_CONFIG = {
    # Primary metric
    "primary_metric": "pr_auc",  # PR-AUC (Section 4)
    
    # Cross-validation
    "cv_folds": 10,  # Stratified 10-fold (Section 4)
    "cv_random_state": 42,
    
    # Metrics to compute
    "metrics": [
        "pr_auc",  # Primary
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "accuracy",  # Reported but not primary
        "per_class_pr_auc",
    ],
    
    # Error analysis
    "perform_error_analysis": True,
    "error_types": [
        "negation_misinterpretation",
        "implicit_interaction",
        "entity_boundary_error",
        "complex_sentence_structure",
        "other",
    ],
}


# Data Configuration
DATA_CONFIG = {
    # Dataset
    "corpus": "DDIExtraction2013",
    "corpus_path": None,  # Path to corpus directory
    
    # Splits
    "train_split": "train",
    "dev_split": "dev",
    "test_split": "test",
    
    # Class names (DDIExtraction 2013)
    "relation_classes": [
        "None",
        "Mechanism",
        "Effect",
        "Advise",
        "Int",
    ],
    
    "ner_classes": [
        "O",  # Outside
        "B-DRUG",  # Begin drug mention
        "I-DRUG",  # Inside drug mention
        "B-BRAND",  # Begin brand name
        "I-BRAND",  # Inside brand name
    ],
}


# Model Checkpoint Configuration
CHECKPOINT_CONFIG = {
    "save_dir": "checkpoints/",
    "save_best_only": True,
    "save_total_limit": 3,
    "checkpoint_metric": "val_pr_auc",
    "checkpoint_mode": "max",
}


# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_dir": "logs/",
    "tensorboard": True,
    "wandb": False,  # Set to True if using Weights & Biases
    "wandb_project": "project-aegis-ddi",
}


# Device Configuration
DEVICE_CONFIG = {
    "device": "cuda",  # "cuda" or "cpu"
    "num_workers": 4,  # DataLoader workers
    "pin_memory": True,
}


def get_config():
    """Get complete configuration dictionary"""
    return {
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "risk_scoring": RISK_SCORING_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "data": DATA_CONFIG,
        "checkpoint": CHECKPOINT_CONFIG,
        "logging": LOGGING_CONFIG,
        "device": DEVICE_CONFIG,
        "hyperparameter_search": HYPERPARAMETER_SEARCH_SPACE,
    }


def print_config():
    """Print configuration in readable format"""
    import json
    config = get_config()
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    print_config()
