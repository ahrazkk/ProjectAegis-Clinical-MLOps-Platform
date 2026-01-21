#!/usr/bin/env python
"""
DDI Model Training Script
Main entry point for training the Drug-Drug Interaction prediction model

Usage:
    python train.py --data_path /path/to/data.json --output_dir ./checkpoints

Reference: MCR III Model Specification
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

import torch

from ddi_model import DDIModel
from tokenization import DDITokenizer
from dataset import DDIDataset, create_data_loaders
from trainer import DDITrainer, TrainingConfig
from evaluation import StratifiedKFoldValidator, evaluate_model, ErrorAnalyzer
from hyperparameter_config import get_default_search_space


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DDI Prediction Model based on MCR III Specification'
    )

    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to training data JSON file'
    )
    parser.add_argument(
        '--val_data_path',
        type=str,
        default=None,
        help='Path to validation data (if not using k-fold CV)'
    )

    # Model arguments
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        help='Pretrained encoder model name'
    )
    parser.add_argument(
        '--use_binary',
        action='store_true',
        default=True,
        help='Use binary classification (default: True)'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=5,
        help='Number of interaction classes for multi-class'
    )

    # Training arguments
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay for AdamW (default: 0.01)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of warmup steps (default: 500)'
    )
    parser.add_argument(
        '--head_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for classification heads (default: 0.1)'
    )
    parser.add_argument(
        '--aux_loss_weight',
        type=float,
        default=0.5,
        help='Weight for auxiliary NER loss (default: 0.5)'
    )

    # Validation arguments
    parser.add_argument(
        '--use_kfold',
        action='store_true',
        help='Use stratified k-fold cross-validation'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=10,
        help='Number of folds for cross-validation (default: 10)'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save_best_only',
        action='store_true',
        default=True,
        help='Save only the best model (default: True)'
    )

    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )

    return parser.parse_args()


def load_data(data_path: str):
    """Load training data from JSON file"""
    logger.info(f"Loading data from {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples")
    return data


def main():
    """Main training function"""
    args = parse_args()

    # Print configuration
    logger.info("=" * 60)
    logger.info("DDI Model Training - MCR III Specification")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = DDITokenizer(encoder_name=args.encoder_name)

    # Load data
    train_data = load_data(args.data_path)

    # Create training config
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.warmup_steps,
        head_dropout_rate=args.head_dropout,
        aux_loss_weight=args.aux_loss_weight,
        num_epochs=args.num_epochs,
        use_binary=args.use_binary,
        num_relation_classes=1 if args.use_binary else args.num_classes,
        encoder_name=args.encoder_name,
        save_best_only=args.save_best_only
    )

    if args.use_kfold:
        # Stratified K-Fold Cross-Validation
        logger.info(f"Using {args.n_folds}-fold cross-validation")

        # Create full dataset
        dataset = DDIDataset(train_data, tokenizer, use_binary_labels=args.use_binary)

        # Get labels for stratification
        import numpy as np
        labels = np.array([
            sample.get('has_interaction', 0) if args.use_binary
            else sample.get('interaction_type', 0)
            for sample in train_data
        ])

        # Run cross-validation
        validator = StratifiedKFoldValidator(n_splits=args.n_folds)
        cv_results = validator.cross_validate(
            dataset=dataset,
            model_class=DDIModel,
            trainer_class=DDITrainer,
            config=config,
            tokenizer=tokenizer,
            labels=labels
        )

        logger.info("\nCross-Validation Results:")
        logger.info(f"  Mean PR-AUC: {cv_results['mean_pr_auc']:.4f} ± {cv_results['std_pr_auc']:.4f}")

        # Save CV results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'cv_results.json', 'w') as f:
            json.dump({
                'mean_pr_auc': cv_results['mean_pr_auc'],
                'std_pr_auc': cv_results['std_pr_auc'],
                'n_folds': cv_results['n_splits'],
                'config': config.to_dict()
            }, f, indent=2)

    else:
        # Standard train/val split
        if args.val_data_path:
            val_data = load_data(args.val_data_path)
        else:
            # Use last 20% as validation
            split_idx = int(len(train_data) * 0.8)
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
            logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val")

        # Create datasets
        train_dataset = DDIDataset(train_data, tokenizer, use_binary_labels=args.use_binary)
        val_dataset = DDIDataset(val_data, tokenizer, use_binary_labels=args.use_binary)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset,
            batch_size=args.batch_size
        )

        # Initialize model
        logger.info("Initializing model...")
        model = DDIModel(
            encoder_name=args.encoder_name,
            num_relation_classes=config.num_relation_classes,
            num_ner_classes=config.num_ner_classes,
            head_dropout_rate=config.head_dropout_rate,
            use_binary=args.use_binary
        )

        # Initialize trainer
        trainer = DDITrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            output_dir=args.output_dir
        )

        # Train model
        logger.info("Starting training...")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            metric_for_best='pr_auc'
        )

        logger.info("\nTraining Complete!")
        logger.info(f"Best PR-AUC: {results['best_metric']:.4f}")

        # Run error analysis on validation set
        logger.info("\nRunning error analysis...")
        error_analyzer = ErrorAnalyzer()

        final_metrics = evaluate_model(
            model=model,
            test_loader=val_loader,
            device=device,
            use_binary=args.use_binary,
            error_analyzer=error_analyzer
        )

        error_analyzer.print_summary()
        error_analyzer.save_report(Path(args.output_dir) / 'error_analysis.json')

        # Save training summary
        summary = {
            'best_pr_auc': results['best_metric'],
            'final_metrics': {k: v for k, v in final_metrics.items() if k != 'confusion_matrix'},
            'config': config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        with open(Path(args.output_dir) / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
