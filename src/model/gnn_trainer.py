"""
GNN Model Training Module
Training loop for GNN-based DDI prediction with the same patterns as DDITrainer

Reference: MCR III - Training with AdamW, warmup scheduling, early stopping

Enhancements (v2):
- Focal Loss for class imbalance (replaces BCEWithLogitsLoss)
- Label smoothing for regularization
- Save evaluation predictions for real visualization generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

from .gnn_model import DDIGraphModel
from .risk_scorer import TemperatureScaling

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Addresses class imbalance by down-weighting easy examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


@dataclass
class GNNTrainingConfig:
    """
    Training configuration for GNN-based DDI model.

    Mirrors DDITrainer's TrainingConfig with GNN-specific parameters.
    """
    # Core hyperparameters
    learning_rate: float = 5e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    num_warmup_steps: int = 200
    dropout_rate: float = 0.15

    # GNN architecture
    hidden_dim: int = 256
    num_gnn_layers: int = 4
    use_jumping_knowledge: bool = True
    max_atoms: int = 128

    # Training parameters
    num_epochs: int = 80
    max_grad_norm: float = 1.0
    use_binary: bool = True
    num_relation_classes: int = 1

    # Loss function
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    # Logging and checkpointing
    log_interval: int = 20
    save_best_only: bool = True
    early_stopping_patience: int = 15

    def to_dict(self) -> Dict:
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'num_warmup_steps': self.num_warmup_steps,
            'dropout_rate': self.dropout_rate,
            'hidden_dim': self.hidden_dim,
            'num_gnn_layers': self.num_gnn_layers,
            'use_jumping_knowledge': self.use_jumping_knowledge,
            'max_atoms': self.max_atoms,
            'num_epochs': self.num_epochs,
            'max_grad_norm': self.max_grad_norm,
            'use_binary': self.use_binary,
            'num_relation_classes': self.num_relation_classes,
            'use_focal_loss': self.use_focal_loss,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'label_smoothing': self.label_smoothing,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GNNTrainingConfig':
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class GNNTrainer:
    """
    Trainer for GNN-based DDI Model.

    Implements:
    - AdamW optimizer with cosine annealing + warmup
    - Gradient clipping
    - Early stopping on PR-AUC
    - Model checkpointing
    - Temperature scaling calibration

    Reference: MCR III Sections 2, 3.1
    """

    def __init__(
        self,
        model: DDIGraphModel,
        config: GNNTrainingConfig,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None
    ):
        """
        Args:
            model: DDIGraphModel instance
            config: GNN training configuration
            device: Training device (cuda/cpu)
            output_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.output_dir = Path(output_dir) if output_dir else Path('./checkpoints/gnn')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Loss function — Focal Loss for better class imbalance handling
        if config.use_binary:
            if config.use_focal_loss:
                self.loss_fn = FocalLoss(
                    alpha=config.focal_alpha,
                    gamma=config.focal_gamma
                )
                logger.info(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
                logger.info("Using BCEWithLogitsLoss")
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.label_smoothing = config.label_smoothing

        # Temperature scaling
        self.temperature_scaling = TemperatureScaling()

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.best_metric = 0.0
        self.patience_counter = 0
        self.training_history = []

    def _setup_optimizer(self, num_training_steps: int):
        """Setup AdamW optimizer with cosine annealing schedule."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Cosine annealing with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config.learning_rate * 0.01
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prediction loss with optional label smoothing.

        Args:
            logits: Model output logits
            labels: Ground truth labels

        Returns:
            Loss tensor
        """
        if self.config.use_binary:
            # Apply label smoothing: 0 -> eps, 1 -> 1-eps
            if self.label_smoothing > 0:
                smoothed = labels * (1 - self.label_smoothing) + (1 - labels) * self.label_smoothing
            else:
                smoothed = labels
            return self.loss_fn(logits.squeeze(-1), smoothed)
        else:
            return self.loss_fn(logits, labels)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            drug1_nf = batch['drug1_node_features'].to(self.device)
            drug1_adj = batch['drug1_adjacency'].to(self.device)
            drug1_ef = batch['drug1_edge_features'].to(self.device)
            drug1_mask = batch['drug1_node_mask'].to(self.device)
            drug2_nf = batch['drug2_node_features'].to(self.device)
            drug2_adj = batch['drug2_adjacency'].to(self.device)
            drug2_ef = batch['drug2_edge_features'].to(self.device)
            drug2_mask = batch['drug2_node_mask'].to(self.device)
            labels = batch['relation_label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                drug1_nf, drug1_adj, drug1_ef, drug1_mask,
                drug2_nf, drug2_adj, drug2_ef, drug2_mask
            )

            # Compute loss
            loss = self._compute_loss(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({'loss': total_loss / num_batches})

        return {'train_loss': total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []
        num_batches = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            drug1_nf = batch['drug1_node_features'].to(self.device)
            drug1_adj = batch['drug1_adjacency'].to(self.device)
            drug1_ef = batch['drug1_edge_features'].to(self.device)
            drug1_mask = batch['drug1_node_mask'].to(self.device)
            drug2_nf = batch['drug2_node_features'].to(self.device)
            drug2_adj = batch['drug2_adjacency'].to(self.device)
            drug2_ef = batch['drug2_edge_features'].to(self.device)
            drug2_mask = batch['drug2_node_mask'].to(self.device)
            labels = batch['relation_label'].to(self.device)

            logits = self.model(
                drug1_nf, drug1_adj, drug1_ef, drug1_mask,
                drug2_nf, drug2_adj, drug2_ef, drug2_mask
            )

            loss = self._compute_loss(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            if self.config.use_binary:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

        from .evaluation import calculate_metrics
        metrics = calculate_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_logits),
            use_binary=self.config.use_binary
        )
        metrics['val_loss'] = total_loss / num_batches

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        metric_for_best: str = 'pr_auc'
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            metric_for_best: Metric for model selection (default: PR-AUC)

        Returns:
            Training results including best metrics
        """
        num_training_steps = len(train_loader) * self.config.num_epochs
        self._setup_optimizer(num_training_steps)

        logger.info(f"Starting GNN training for {self.config.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val PR-AUC: {val_metrics.get('pr_auc', 0):.4f}")

            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            self.training_history.append(epoch_metrics)

            # Check for best model
            current_metric = val_metrics.get(metric_for_best, 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                if self.config.save_best_only:
                    self._save_checkpoint('gnn_best_model.pt', epoch_metrics)
                    logger.info(f"  Saved best GNN model (PR-AUC: {current_metric:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self._save_checkpoint('gnn_final_model.pt', self.training_history[-1])

        # Calibrate temperature scaling
        self._calibrate_temperature(val_loader)

        # Save predictions for real visualization generation
        self._save_evaluation_predictions(val_loader)

        return {
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }

    @torch.no_grad()
    def _save_evaluation_predictions(self, val_loader: DataLoader):
        """Save real y_true, y_pred, y_scores for chart generation."""
        self.model.eval()

        # Load best model for final evaluation
        best_path = self.output_dir / 'gnn_best_model.pt'
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        all_labels = []
        all_scores = []

        for batch in val_loader:
            drug1_nf = batch['drug1_node_features'].to(self.device)
            drug1_adj = batch['drug1_adjacency'].to(self.device)
            drug1_ef = batch['drug1_edge_features'].to(self.device)
            drug1_mask = batch['drug1_node_mask'].to(self.device)
            drug2_nf = batch['drug2_node_features'].to(self.device)
            drug2_adj = batch['drug2_adjacency'].to(self.device)
            drug2_ef = batch['drug2_edge_features'].to(self.device)
            drug2_mask = batch['drug2_node_mask'].to(self.device)
            labels = batch['relation_label']

            logits = self.model(
                drug1_nf, drug1_adj, drug1_ef, drug1_mask,
                drug2_nf, drug2_adj, drug2_ef, drug2_mask
            )

            scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(scores.tolist())

        # Save as JSON for chart generation
        eval_data = {
            'y_true': all_labels,
            'y_scores': all_scores,
            'n_samples': len(all_labels),
            'timestamp': datetime.now().isoformat(),
        }
        eval_path = self.output_dir / 'evaluation_predictions.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f)

        # Also save training history for loss curves
        history_path = self.output_dir / 'training_history.json'
        serializable_history = []
        for h in self.training_history:
            entry = {}
            for k, v in h.items():
                if isinstance(v, (float, int, str)):
                    entry[k] = v
                elif isinstance(v, np.floating):
                    entry[k] = float(v)
                else:
                    entry[k] = str(v)
            serializable_history.append(entry)
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        logger.info(f"Saved evaluation predictions ({len(all_labels)} samples) to {eval_path}")
        logger.info(f"Saved training history ({len(self.training_history)} epochs) to {history_path}")

    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'temperature': self.temperature_scaling.temperature.item(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'temperature' in checkpoint:
            self.temperature_scaling.temperature.data = torch.tensor(
                [checkpoint['temperature']]
            )

        logger.info(f"Loaded GNN checkpoint from {filepath}")

    @torch.no_grad()
    def _calibrate_temperature(self, val_loader: DataLoader):
        """
        Calibrate temperature scaling on validation set.

        Reference: MCR III Section 3.1 - Temperature Scaling
        """
        self.model.eval()

        all_logits = []
        all_labels = []

        for batch in val_loader:
            drug1_nf = batch['drug1_node_features'].to(self.device)
            drug1_adj = batch['drug1_adjacency'].to(self.device)
            drug1_ef = batch['drug1_edge_features'].to(self.device)
            drug1_mask = batch['drug1_node_mask'].to(self.device)
            drug2_nf = batch['drug2_node_features'].to(self.device)
            drug2_adj = batch['drug2_adjacency'].to(self.device)
            drug2_ef = batch['drug2_edge_features'].to(self.device)
            drug2_mask = batch['drug2_node_mask'].to(self.device)
            labels = batch['relation_label']

            logits = self.model(
                drug1_nf, drug1_adj, drug1_ef, drug1_mask,
                drug2_nf, drug2_adj, drug2_ef, drug2_mask
            )

            all_logits.append(logits.cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if not self.config.use_binary:
            self.temperature_scaling.calibrate(logits, labels.long())
            logger.info(
                f"Calibrated temperature: "
                f"{self.temperature_scaling.temperature.item():.3f}"
            )
