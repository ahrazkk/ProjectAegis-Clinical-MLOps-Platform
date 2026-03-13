"""
Train GNN DDI Model

Trains the Edge-Conditioned GIN model for drug-drug interaction prediction
using molecular graph representations of drug pairs.

Prerequisites:
    1. Run prepare_gnn_data.py first to generate train/val JSON files
    2. Requires PyTorch with CUDA and RDKit

Usage:
    python train_gnn.py                          # Default settings
    python train_gnn.py --epochs 100 --lr 5e-4   # Custom hyperparams
    python train_gnn.py --binary                  # Binary classification
    python train_gnn.py --multiclass              # Severity classification
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
DATA_DIR = PROJECT_ROOT / 'web' / 'data' / 'gnn_training'
CHECKPOINT_DIR = PROJECT_ROOT / 'web' / 'models' / 'gnn'


def load_training_data(
    data_dir: Path,
    use_binary: bool = True,
    max_atoms: int = 128,
) -> tuple:
    """Load and featurize training/validation data."""
    train_path = data_dir / 'train.json'
    val_path = data_dir / 'val.json'

    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Training data not found at {data_dir}/\n"
            f"Run prepare_gnn_data.py first to generate the data."
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
    parser = argparse.ArgumentParser(description="Train GNN DDI Model")

    # Task type
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        '--binary', action='store_true', default=True,
        help='Binary classification: interacts or not (default)'
    )
    task_group.add_argument(
        '--multiclass', action='store_true',
        help='Multi-class: none/minor/moderate/severe'
    )

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--max-atoms', type=int, default=128,
                        help='Max atoms per molecule (default: 128)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')

    # Output
    parser.add_argument('--output-dir', type=str, default=str(CHECKPOINT_DIR),
                        help='Checkpoint output directory')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                        help='Training data directory')

    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (training will be slower)")

    # Task configuration
    use_binary = not args.multiclass
    num_classes = 1 if use_binary else 4  # none/minor/moderate/severe

    logger.info(f"Task: {'Binary' if use_binary else 'Multi-class'} classification")
    logger.info(f"Classes: {num_classes}")

    # Load data
    data_dir = Path(args.data_dir)
    train_dataset, val_dataset = load_training_data(
        data_dir, use_binary=use_binary, max_atoms=args.max_atoms
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error("No valid training samples. Check SMILES data.")
        sys.exit(1)

    # Create data loaders
    train_loader, val_loader = create_graph_data_loaders(
        train_dataset, val_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Windows compatibility
    )

    # Create model
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

    # Training config
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
    logger.info(f"Starting GNN Training")
    logger.info(f"{'='*60}")
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

    # Save training results
    output_path = Path(args.output_dir)
    results_file = output_path / 'training_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert numpy values that may be in history
        serializable = {
            'best_metric': float(results['best_metric']),
            'config': results['config'],
            'elapsed_seconds': elapsed.total_seconds(),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'device': str(device),
        }
        json.dump(serializable, f, indent=2)

    logger.info(f"\nModel saved to: {output_path}")
    logger.info(f"  Best model: gnn_best_model.pt")
    logger.info(f"  Final model: gnn_final_model.pt")
    logger.info(f"  Results: training_results.json")


if __name__ == '__main__':
    main()
