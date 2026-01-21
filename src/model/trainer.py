"""
DDI Model Training Module
Implements multi-task training with relation classification and NER auxiliary task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

from .ddi_model import DDIModel
from .tokenization import DDITokenizer
from .risk_scorer import TemperatureScaling


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training configuration following MCR III Hyperparameter Specification

    Reference: MCR III Section 2 - Hyperparameter Tuning Strategy
    """
    # Core hyperparameters (tunable via Vertex AI Vizier)
    learning_rate: float = 2e-5           # Range: [1e-6, 5e-5], LOG_SCALE
    batch_size: int = 16                   # Options: [8, 16, 32]
    weight_decay: float = 0.01             # Range: [0.0, 0.1], LINEAR_SCALE
    num_warmup_steps: int = 500            # Range: [100, 1000], LINEAR_SCALE
    head_dropout_rate: float = 0.1         # Range: [0.1, 0.3], LINEAR_SCALE
    aux_loss_weight: float = 0.5           # Range: [0.2, 0.8], LINEAR_SCALE

    # Fixed training parameters
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    use_binary: bool = True
    num_relation_classes: int = 1          # 1 for binary, k for multi-class
    num_ner_classes: int = 3               # O, B-DRUG, I-DRUG

    # Model configuration
    encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    freeze_encoder_epochs: int = 0          # Number of epochs to freeze encoder

    # Logging and checkpointing
    log_interval: int = 50
    save_best_only: bool = True
    early_stopping_patience: int = 3

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'num_warmup_steps': self.num_warmup_steps,
            'head_dropout_rate': self.head_dropout_rate,
            'aux_loss_weight': self.aux_loss_weight,
            'num_epochs': self.num_epochs,
            'max_grad_norm': self.max_grad_norm,
            'use_binary': self.use_binary,
            'num_relation_classes': self.num_relation_classes,
            'num_ner_classes': self.num_ner_classes,
            'encoder_name': self.encoder_name,
            'freeze_encoder_epochs': self.freeze_encoder_epochs
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class DDITrainer:
    """
    Trainer for DDI Model with multi-task learning

    Implements:
    - Combined loss: relation classification + auxiliary NER
    - AdamW optimizer with warmup scheduling
    - Gradient clipping
    - Early stopping
    - Model checkpointing

    Reference: MCR III Sections 1.2, 1.3, 2
    """

    def __init__(
        self,
        model: DDIModel,
        tokenizer: DDITokenizer,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize DDI Trainer

        Args:
            model: DDIModel instance
            tokenizer: DDITokenizer instance
            config: Training configuration
            device: Training device (cuda/cpu)
            output_dir: Directory for saving checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir) if output_dir else Path('./checkpoints')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resize model embeddings for special tokens
        tokenizer.resize_model_embeddings(model)

        # Move model to device
        self.model.to(self.device)

        # Loss functions
        if config.use_binary:
            self.relation_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.relation_loss_fn = nn.CrossEntropyLoss()

        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # Temperature scaling for calibration
        self.temperature_scaling = TemperatureScaling()

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.best_metric = 0.0
        self.patience_counter = 0
        self.training_history = []

    def _setup_optimizer(self, num_training_steps: int):
        """
        Setup AdamW optimizer and learning rate scheduler

        Reference: MCR III Section 2 - AdamW L2 regularization
        """
        # Separate encoder and head parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.encoder.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate
            },
            {
                'params': [p for n, p in self.model.encoder.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate
            },
            {
                'params': list(self.model.relation_head.parameters()) +
                         list(self.model.auxiliary_head.parameters()),
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate * 10  # Higher LR for heads
            }
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters)

        # Linear warmup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _compute_loss(
        self,
        relation_logits: torch.Tensor,
        ner_logits: torch.Tensor,
        relation_labels: torch.Tensor,
        ner_labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined multi-task loss

        Total Loss = L_relation + aux_loss_weight * L_ner

        Args:
            relation_logits: Relation classification logits
            ner_logits: NER classification logits
            relation_labels: Ground truth relation labels
            ner_labels: Ground truth NER labels
            attention_mask: Attention mask for valid tokens

        Returns:
            total_loss, relation_loss, ner_loss
        """
        # Relation loss
        if self.config.use_binary:
            relation_loss = self.relation_loss_fn(
                relation_logits.squeeze(-1),
                relation_labels
            )
        else:
            relation_loss = self.relation_loss_fn(relation_logits, relation_labels)

        # NER loss (masked to valid tokens only)
        # Reshape for cross entropy: [batch_size * seq_len, num_classes]
        active_loss = attention_mask.view(-1) == 1
        active_logits = ner_logits.view(-1, ner_logits.size(-1))[active_loss]
        active_labels = ner_labels.view(-1)[active_loss]

        ner_loss = self.ner_loss_fn(active_logits, active_labels)

        # Combined loss with auxiliary weight
        total_loss = relation_loss + self.config.aux_loss_weight * ner_loss

        return total_loss, relation_loss, ner_loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training DataLoader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Optionally freeze encoder for initial epochs
        if epoch < self.config.freeze_encoder_epochs:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        total_loss = 0.0
        relation_loss_sum = 0.0
        ner_loss_sum = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            drug1_mask = batch['drug1_mask'].to(self.device)
            drug2_mask = batch['drug2_mask'].to(self.device)
            ner_labels = batch['ner_labels'].to(self.device)
            relation_labels = batch['relation_label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            relation_logits, ner_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask
            )

            # Compute loss
            loss, rel_loss, ner_loss = self._compute_loss(
                relation_logits, ner_logits,
                relation_labels, ner_labels,
                attention_mask
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            relation_loss_sum += rel_loss.item()
            ner_loss_sum += ner_loss.item()
            num_batches += 1

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': total_loss / num_batches,
                    'rel_loss': relation_loss_sum / num_batches,
                    'ner_loss': ner_loss_sum / num_batches
                })

        return {
            'train_loss': total_loss / num_batches,
            'train_relation_loss': relation_loss_sum / num_batches,
            'train_ner_loss': ner_loss_sum / num_batches
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set

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
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            drug1_mask = batch['drug1_mask'].to(self.device)
            drug2_mask = batch['drug2_mask'].to(self.device)
            ner_labels = batch['ner_labels'].to(self.device)
            relation_labels = batch['relation_label'].to(self.device)

            # Forward pass
            relation_logits, ner_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask
            )

            # Compute loss
            loss, _, _ = self._compute_loss(
                relation_logits, ner_logits,
                relation_labels, ner_labels,
                attention_mask
            )

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            if self.config.use_binary:
                probs = torch.sigmoid(relation_logits.squeeze(-1))
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(relation_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(relation_labels.cpu().numpy())
            all_logits.extend(relation_logits.cpu().numpy())

        # Calculate metrics
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
    ) -> Dict[str, any]:
        """
        Full training loop with early stopping

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            metric_for_best: Metric to use for model selection

        Returns:
            Training results including best metrics
        """
        num_training_steps = len(train_loader) * self.config.num_epochs
        self._setup_optimizer(num_training_steps)

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val PR-AUC: {val_metrics.get('pr_auc', 0):.4f}")

            # Track history
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            self.training_history.append(epoch_metrics)

            # Check for best model
            current_metric = val_metrics.get(metric_for_best, 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0

                if self.config.save_best_only:
                    self._save_checkpoint('best_model.pt', epoch_metrics)
                    logger.info(f"  Saved best model (PR-AUC: {current_metric:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Save final model
        self._save_checkpoint('final_model.pt', self.training_history[-1])

        # Calibrate temperature scaling
        self._calibrate_temperature(val_loader)

        return {
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }

    def _save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'temperature': self.temperature_scaling.temperature.item(),
            'config': self.config.to_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'temperature' in checkpoint:
            self.temperature_scaling.temperature.data = torch.tensor([checkpoint['temperature']])

        logger.info(f"Loaded checkpoint from {filepath}")

    @torch.no_grad()
    def _calibrate_temperature(self, val_loader: DataLoader):
        """
        Calibrate temperature scaling on validation set

        Reference: MCR III Section 3.1 - Temperature Scaling
        """
        self.model.eval()

        all_logits = []
        all_labels = []

        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            drug1_mask = batch['drug1_mask'].to(self.device)
            drug2_mask = batch['drug2_mask'].to(self.device)
            relation_labels = batch['relation_label']

            relation_logits, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask
            )

            all_logits.append(relation_logits.cpu())
            all_labels.append(relation_labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Calibrate temperature
        if not self.config.use_binary:
            self.temperature_scaling.calibrate(logits, labels.long())
            logger.info(f"Calibrated temperature: {self.temperature_scaling.temperature.item():.3f}")
