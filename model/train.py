"""
Training Script for DDI Relation Extraction Model
Quick start script for model training
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from model import (
    DDIRelationModel,
    DDIDataPreprocessor,
    DDIDataset,
    DDITrainer,
    DDIEvaluator,
    get_config,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path, preprocessor: DDIDataPreprocessor):
    """
    Load training data from DDIExtraction 2013 corpus.
    
    Args:
        data_path: Path to data directory or file
        preprocessor: DDIDataPreprocessor instance
        
    Returns:
        train_dataset, val_dataset, test_dataset
        
    Note:
        This is a placeholder implementation. To use with DDIExtraction 2013 corpus:
        
        1. Download corpus from: https://github.com/isegura/DDICorpus
        2. Expected structure:
           data_path/
           ├── Train/
           │   └── DrugBank/*.xml
           ├── Test/
           │   └── DrugBank/*.xml
        
        3. Implement XML parsing using data_preprocessor.load_ddi_extraction_2013_corpus()
        4. Parse <sentence>, <entity>, and <pair> tags
        5. Extract text, entity spans, and relation labels
        
        For production use, replace this placeholder with actual corpus loading.
    """
    logger.warning("Using dummy data - implement load_data() for actual DDIExtraction 2013 corpus")
    logger.warning("See function docstring for integration steps")
    
    # Create dummy examples
    examples = [
        {
            "text": f"Sample text {i} with drug1 and drug2 interaction.",
            "drug1_span": (20, 25),
            "drug2_span": (30, 35),
            "relation_label": i % 5,
            "ner_labels": [0] * 50,
        }
        for i in range(100)
    ]
    
    # Split data
    train_size = int(0.7 * len(examples))
    val_size = int(0.15 * len(examples))
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:train_size + val_size]
    test_examples = examples[train_size + val_size:]
    
    # Create datasets
    train_dataset = DDIDataset(train_examples, preprocessor, include_ner=True)
    val_dataset = DDIDataset(val_examples, preprocessor, include_ner=True)
    test_dataset = DDIDataset(test_examples, preprocessor, include_ner=True)
    
    logger.info(f"Loaded data: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_dataset, val_dataset, test_dataset


def main(args):
    """Main training function"""
    
    # Load configuration
    config = get_config()
    
    # Override with command-line arguments
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    logger.info("=" * 60)
    logger.info("DDI Relation Extraction Model Training")
    logger.info("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize preprocessor
    logger.info("Initializing preprocessor...")
    preprocessor = DDIDataPreprocessor(
        model_name=config['model']['model_name'],
        max_length=config['model']['max_length'],
    )
    
    # Load data
    logger.info("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        Path(args.data_path) if args.data_path else None,
        preprocessor,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory'] and device == "cuda",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory'] and device == "cuda",
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = DDIRelationModel(
        model_name=config['model']['model_name'],
        num_relation_classes=config['model']['num_relation_classes'],
        num_ner_classes=config['model']['num_ner_classes'],
        head_dropout_rate=config['training']['head_dropout_rate'],
        relation_hidden_dim=config['model']['relation_hidden_dim'],
        freeze_encoder=config['model']['freeze_encoder'],
    )
    
    # Resize embeddings for marker tokens
    model.encoder.resize_token_embeddings(len(preprocessor.tokenizer))
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = DDITrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_warmup_steps=config['training']['num_warmup_steps'],
        aux_loss_weight=config['training']['aux_loss_weight'],
        max_grad_norm=config['training']['max_grad_norm'],
    )
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics function
    def compute_metrics(predictions, labels, probabilities):
        evaluator = DDIEvaluator(model, device)
        return evaluator.compute_metrics(
            predictions,
            labels,
            probabilities,
            class_names=config['data']['relation_classes'],
        )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=save_dir,
        early_stopping_patience=config['training']['early_stopping_patience'],
        compute_metrics_fn=compute_metrics,
    )
    
    # Evaluate on test set
    if not args.skip_test:
        logger.info("Evaluating on test set...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
        )
        
        evaluator = DDIEvaluator(model, device)
        test_metrics = evaluator.evaluate_model(
            test_loader,
            class_names=config['data']['relation_classes'],
        )
        
        logger.info("=" * 60)
        logger.info("Test Set Results")
        logger.info("=" * 60)
        logger.info(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
        logger.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
        logger.info(f"Macro Precision: {test_metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        
        if 'classification_report' in test_metrics:
            logger.info("\nClassification Report:")
            logger.info(test_metrics['classification_report'])
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {save_dir / 'best_model.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDI Relation Extraction Model")
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to DDIExtraction 2013 corpus directory"
    )
    
    # Training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default from config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default from config)"
    )
    
    # Device arguments
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test set evaluation"
    )
    
    args = parser.parse_args()
    
    main(args)
