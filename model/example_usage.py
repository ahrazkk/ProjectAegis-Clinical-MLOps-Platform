"""
Example Usage Script for DDI Relation Extraction Model
Demonstrates how to use the implemented model components
"""

import torch
from pathlib import Path
import logging

from model import (
    DDIRelationModel,
    ModelWithTemperature,
    RiskScorer,
    DDIDataPreprocessor,
    DDITrainer,
    DDIEvaluator,
)
from model.data_preprocessor import DDIDataset
from model.risk_scorer import InteractionType, RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_preprocessing():
    """Example: Data preprocessing with entity markers"""
    logger.info("=== Example 1: Data Preprocessing ===")
    
    # Initialize preprocessor
    preprocessor = DDIDataPreprocessor()
    
    # Example text with drug entities
    text = "The interaction between aspirin and warfarin may increase bleeding risk."
    drug1_span = (26, 33)  # "aspirin"
    drug2_span = (38, 46)  # "warfarin"
    
    # Encode single example
    encoding = preprocessor.encode_single(text, drug1_span, drug2_span)
    
    logger.info(f"Input IDs shape: {encoding['input_ids'].shape}")
    logger.info(f"Drug1 mask positions: {encoding['drug1_mask'].nonzero()}")
    logger.info(f"Drug2 mask positions: {encoding['drug2_mask'].nonzero()}")
    
    # Decode to see marked text
    tokens = preprocessor.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    logger.info(f"Tokenized text: {' '.join(tokens[:30])}...")
    
    return preprocessor, encoding


def example_model_inference():
    """Example: Model inference"""
    logger.info("\n=== Example 2: Model Inference ===")
    
    # Initialize model
    model = DDIRelationModel(
        num_relation_classes=5,  # None, Mechanism, Effect, Advise, Int
        num_ner_classes=5,  # O, B-DRUG, I-DRUG, B-BRAND, I-BRAND
    )
    
    # Get preprocessor and example encoding
    preprocessor, encoding = example_preprocessing()
    
    # Prepare batch (add batch dimension)
    input_ids = encoding['input_ids'].unsqueeze(0)
    attention_mask = encoding['attention_mask'].unsqueeze(0)
    drug1_mask = encoding['drug1_mask'].unsqueeze(0)
    drug2_mask = encoding['drug2_mask'].unsqueeze(0)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drug1_mask=drug1_mask,
            drug2_mask=drug2_mask,
        )
    
    logger.info(f"Relation logits shape: {outputs['relation_logits'].shape}")
    logger.info(f"NER logits shape: {outputs['ner_logits'].shape}")
    logger.info(f"Drug1 vector shape: {outputs['drug1_vec'].shape}")
    logger.info(f"Drug2 vector shape: {outputs['drug2_vec'].shape}")
    
    # Get predictions
    pred_outputs = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        drug1_mask=drug1_mask,
        drug2_mask=drug2_mask,
    )
    
    logger.info(f"Predicted class: {pred_outputs['relation_pred'].item()}")
    logger.info(f"Relation probabilities: {pred_outputs['relation_probs'][0]}")
    
    return model, pred_outputs


def example_temperature_scaling():
    """Example: Temperature scaling for calibration"""
    logger.info("\n=== Example 3: Temperature Scaling ===")
    
    # Create base model
    base_model = DDIRelationModel(num_relation_classes=5)
    
    # Wrap with temperature scaling
    calibrated_model = ModelWithTemperature(base_model, temperature=1.5)
    
    # Get example encoding
    preprocessor, encoding = example_preprocessing()
    
    input_ids = encoding['input_ids'].unsqueeze(0)
    attention_mask = encoding['attention_mask'].unsqueeze(0)
    drug1_mask = encoding['drug1_mask'].unsqueeze(0)
    drug2_mask = encoding['drug2_mask'].unsqueeze(0)
    
    # Get calibrated predictions
    calibrated_model.eval()
    pred_outputs = calibrated_model.predict_calibrated(
        input_ids=input_ids,
        attention_mask=attention_mask,
        drug1_mask=drug1_mask,
        drug2_mask=drug2_mask,
    )
    
    logger.info(f"Temperature: {pred_outputs['temperature']}")
    logger.info(f"Calibrated probabilities: {pred_outputs['relation_probs_calibrated'][0]}")
    
    return calibrated_model


def example_risk_scoring():
    """Example: Clinical risk scoring"""
    logger.info("\n=== Example 4: Risk Scoring ===")
    
    # Initialize risk scorer
    risk_scorer = RiskScorer(use_calibrated_probs=True)
    
    # Example calibrated probabilities (from model)
    # [None, Mechanism, Effect, Advise, Int]
    relation_probs = torch.tensor([
        [0.1, 0.6, 0.2, 0.05, 0.05],  # High mechanism probability
        [0.8, 0.05, 0.05, 0.05, 0.05],  # No interaction
        [0.2, 0.1, 0.5, 0.1, 0.1],  # Moderate effect
    ])
    
    # Compute risk profiles
    risk_profiles = risk_scorer.compute_full_risk_profile(relation_probs)
    
    for i, profile in enumerate(risk_profiles):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  Risk Score: {profile['risk_score']:.3f}")
        logger.info(f"  Risk Level: {profile['risk_level']}")
        logger.info(f"  Predicted Interaction: {profile['predicted_interaction']}")
        logger.info(f"  Confidence: {profile['interaction_probability']:.3f}")
        
        # Get clinical recommendation
        risk_level = RiskLevel[profile['risk_level'].upper()]
        recommendation = RiskScorer.get_clinical_recommendation(
            risk_level,
            profile['predicted_interaction']
        )
        logger.info(f"  Recommendation: {recommendation}")
    
    return risk_scorer, risk_profiles


def example_training_setup():
    """Example: Setting up model training"""
    logger.info("\n=== Example 5: Training Setup ===")
    
    # Create model
    model = DDIRelationModel(
        num_relation_classes=5,
        num_ner_classes=5,
        head_dropout_rate=0.1,
    )
    
    # Create trainer with hyperparameters from Section 2 of specification
    trainer = DDITrainer(
        model=model,
        learning_rate=2e-5,  # Section 2: range [1e-6, 5e-5]
        weight_decay=0.01,  # Section 2: range [0.0, 0.1]
        num_warmup_steps=500,  # Section 2: range [100, 1000]
        aux_loss_weight=0.5,  # Section 2: range [0.2, 0.8]
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Learning rate: {trainer.learning_rate}")
    logger.info(f"Weight decay: {trainer.weight_decay}")
    logger.info(f"Auxiliary loss weight: {trainer.aux_loss_weight}")
    
    return trainer


def example_evaluation_setup():
    """Example: Setting up model evaluation"""
    logger.info("\n=== Example 6: Evaluation Setup ===")
    
    # Create model
    model = DDIRelationModel(num_relation_classes=5)
    
    # Create evaluator
    evaluator = DDIEvaluator(model=model)
    
    # Example class names for DDIExtraction 2013 corpus
    class_names = ["None", "Mechanism", "Effect", "Advise", "Int"]
    
    logger.info(f"Evaluator initialized with {len(class_names)} classes")
    logger.info(f"Primary metric: PR-AUC (Section 4 requirement)")
    logger.info(f"Validation method: Stratified 10-fold CV (Section 4 requirement)")
    
    return evaluator, class_names


def example_dataset_creation():
    """Example: Creating a dataset"""
    logger.info("\n=== Example 7: Dataset Creation ===")
    
    # Example data
    examples = [
        {
            "text": "The interaction between aspirin and warfarin may increase bleeding risk.",
            "drug1_span": (26, 33),
            "drug2_span": (38, 46),
            "relation_label": 2,  # Effect
            "ner_labels": [0] * 50,  # Placeholder
        },
        {
            "text": "Ibuprofen should not be taken with aspirin due to drug interaction.",
            "drug1_span": (0, 9),
            "drug2_span": (35, 42),
            "relation_label": 3,  # Advise
            "ner_labels": [0] * 50,  # Placeholder
        },
    ]
    
    # Create preprocessor
    preprocessor = DDIDataPreprocessor()
    
    # Create dataset
    dataset = DDIDataset(
        examples=examples,
        preprocessor=preprocessor,
        include_ner=True,
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"Relation label: {sample['relation_label']}")
    
    return dataset


def main():
    """Run all examples"""
    logger.info("=" * 60)
    logger.info("DDI Relation Extraction Model - Example Usage")
    logger.info("=" * 60)
    
    try:
        # Run examples
        example_preprocessing()
        example_model_inference()
        example_temperature_scaling()
        example_risk_scoring()
        example_training_setup()
        example_evaluation_setup()
        example_dataset_creation()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
