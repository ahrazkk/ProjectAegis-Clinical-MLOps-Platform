"""
Train Enhanced GNN DDI Model
=============================
Uses the expanded dataset (Neo4j + DDI Corpus + TWOSIDES)
with improved architecture (focal loss, deeper head, 4 GNN layers).

Prerequisites:
    1. Run prepare_enhanced_data.py to generate enhanced train/val/test JSON
    2. Requires PyTorch with CUDA and RDKit

Usage:
    python train_enhanced.py                    # Default enhanced settings
    python train_enhanced.py --epochs 100       # More epochs
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add model package to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from model.gnn_model import DDIGraphModel
from model.gnn_featurizer import MolecularGraphFeaturizer, ATOM_FEATURE_DIM, EDGE_FEATURE_DIM
from model.gnn_dataset import DDIGraphDataset, create_graph_data_loaders
from model.gnn_trainer import GNNTrainer, GNNTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # molecular-ai/
DATA_DIR = PROJECT_ROOT / 'web' / 'data' / 'gnn_training_enhanced'
CHECKPOINT_DIR = PROJECT_ROOT / 'web' / 'models' / 'gnn'


def load_training_data(data_dir: Path, use_binary: bool = True, max_atoms: int = 128):
    """Load and featurize training/validation data."""
    train_path = data_dir / 'train.json'
    val_path = data_dir / 'val.json'

    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Enhanced training data not found at {data_dir}/\n"
            f"Run prepare_enhanced_data.py first."
        )
        sys.exit(1)

    featurizer = MolecularGraphFeaturizer(max_atoms=max_atoms)

    logger.info("Loading training data...")
    train_dataset = DDIGraphDataset.from_json(
        str(train_path), featurizer, use_binary_labels=use_binary
    )

    logger.info("Loading validation data...")
    val_dataset = DDIGraphDataset.from_json(
        str(val_path), featurizer, use_binary_labels=use_binary
    )

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced GNN DDI Model")

    # Model architecture — improved defaults
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of GNN layers (default: 4, was 3)')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout rate (default: 0.15)')
    parser.add_argument('--max-atoms', type=int, default=128,
                        help='Max atoms per molecule (default: 128)')

    # Training — improved defaults
    parser.add_argument('--epochs', type=int, default=80,
                        help='Max training epochs (default: 80)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')

    # Loss
    parser.add_argument('--no-focal-loss', action='store_true',
                        help='Disable focal loss (use BCE instead)')
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.05)

    # Output
    parser.add_argument('--output-dir', type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR))

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (training will be slower)")

    use_binary = True
    num_classes = 1

    # Load data
    data_dir = Path(args.data_dir)
    train_dataset, val_dataset = load_training_data(
        data_dir, use_binary=use_binary, max_atoms=args.max_atoms
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error("No valid training samples.")
        sys.exit(1)

    # Data loaders
    train_loader, val_loader = create_graph_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Model — enhanced architecture
    model = DDIGraphModel(
        atom_feature_dim=ATOM_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        num_relation_classes=num_classes,
        dropout_rate=args.dropout,
        use_binary=use_binary,
        use_jumping_knowledge=True,
    )

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} total, {trainable:,} trainable")

    # Training config — enhanced
    config = GNNTrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_layers,
        use_jumping_knowledge=True,
        max_atoms=args.max_atoms,
        num_epochs=args.epochs,
        max_grad_norm=1.0,
        use_binary=use_binary,
        num_relation_classes=num_classes,
        early_stopping_patience=args.patience,
        log_interval=10,
        save_best_only=True,
        use_focal_loss=not args.no_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    # Create trainer
    trainer = GNNTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )

    # Train
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting ENHANCED GNN Training")
    logger.info(f"{'='*60}")
    logger.info(f"Improvements over v1:")
    logger.info(f"  - Dataset: {len(train_dataset)} samples (was 2,002)")
    logger.info(f"  - GNN layers: {args.num_layers} (was 3)")
    logger.info(f"  - Loss: {'Focal' if not args.no_focal_loss else 'BCE'} (was BCE)")
    logger.info(f"  - Interaction head: product+diff+sum (was concatenation)")
    logger.info(f"  - Label smoothing: {args.label_smoothing}")
    logger.info(f"Config: {json.dumps(config.to_dict(), indent=2)}")

    start_time = datetime.now()
    results = trainer.train(train_loader, val_loader)
    elapsed = datetime.now() - start_time

    # Results
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Time: {elapsed}")
    logger.info(f"Best PR-AUC: {results['best_metric']:.4f}")

    # Save results
    output_path = Path(args.output_dir)
    results_file = output_path / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        serializable = {
            'best_metric': float(results['best_metric']),
            'config': results['config'],
            'elapsed_seconds': elapsed.total_seconds(),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'device': str(device),
            'data_sources': 'neo4j+ddi_corpus+twosides',
            'version': 'enhanced_v2',
        }
        json.dump(serializable, f, indent=2)

    logger.info(f"\nModel saved to: {output_path}")
    logger.info(f"  Best model: gnn_best_model.pt")
    logger.info(f"  Predictions: evaluation_predictions.json (for real charts)")
    logger.info(f"  History: training_history.json (for loss curves)")


if __name__ == '__main__':
    main()
