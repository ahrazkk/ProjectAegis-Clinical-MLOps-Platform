"""
Training Pipeline for DDI Relation Extraction Model
Implements hyperparameter configuration from Section 2 of specification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path

from .ddi_model import DDIRelationModel, ModelWithTemperature

logger = logging.getLogger(__name__)


class DDITrainer:
    """
    Training Pipeline for DDI Relation Extraction
    
    Implements training configuration from Section 2 (Hyperparameter Tuning Strategy):
    - Learning rate with warmup
    - AdamW optimizer with weight decay
    - Auxiliary loss weighting
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: DDIRelationModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 500,
        aux_loss_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: DDIRelationModel instance
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Peak learning rate (Section 2: range [1e-6, 5e-5])
            weight_decay: L2 regularization for AdamW (Section 2: range [0.0, 0.1])
            num_warmup_steps: Linear warmup steps (Section 2: range [100, 1000])
            aux_loss_weight: Weight for auxiliary NER loss (Section 2: range [0.2, 0.8])
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.device = device
        
        # Hyperparameters from Section 2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.aux_loss_weight = aux_loss_weight
        self.max_grad_norm = max_grad_norm
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_relation_loss": [],
            "train_ner_loss": [],
            "val_loss": [],
            "val_metrics": [],
        }
        
        # Optimizer (AdamW as per Section 2)
        self.optimizer = None
        self.scheduler = None
    
    def setup_optimizer(self, num_training_steps: int):
        """
        Setup AdamW optimizer with linear warmup scheduler.
        
        Args:
            num_training_steps: Total number of training steps
        """
        # AdamW with weight decay (Section 2)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        
        # Linear warmup + linear decay scheduler (Section 2)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Optimizer setup: lr={self.learning_rate}, weight_decay={self.weight_decay}")
        logger.info(f"Scheduler: {self.num_warmup_steps} warmup steps, {num_training_steps} total steps")
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        relation_labels: torch.Tensor,
        ner_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with auxiliary task weighting.
        
        Loss = L_relation + α × L_ner
        where α = aux_loss_weight (Section 2: range [0.2, 0.8])
        
        Args:
            outputs: Model outputs dictionary
            relation_labels: [batch_size] - Relation class labels
            ner_labels: [batch_size, seq_len] - NER labels (optional)
            
        Returns:
            Dictionary with total_loss, relation_loss, ner_loss
        """
        # Relation classification loss (Primary task)
        relation_loss = F.cross_entropy(
            outputs["relation_logits"],
            relation_labels,
        )
        
        losses = {
            "relation_loss": relation_loss,
            "total_loss": relation_loss,
        }
        
        # Auxiliary NER loss (Section 1.3)
        if ner_labels is not None:
            # Flatten for token-level classification
            ner_logits_flat = outputs["ner_logits"].view(-1, outputs["ner_logits"].size(-1))
            ner_labels_flat = ner_labels.view(-1)
            
            # Ignore padding tokens (label = -100)
            ner_loss = F.cross_entropy(
                ner_logits_flat,
                ner_labels_flat,
                ignore_index=-100,
            )
            
            losses["ner_loss"] = ner_loss
            
            # Combined loss with auxiliary weight
            losses["total_loss"] = relation_loss + self.aux_loss_weight * ner_loss
        
        return losses
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_relation_loss = 0.0
        total_ner_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            drug1_mask = batch["drug1_mask"].to(self.device)
            drug2_mask = batch["drug2_mask"].to(self.device)
            relation_labels = batch["relation_label"].to(self.device)
            
            ner_labels = None
            if "ner_labels" in batch:
                ner_labels = batch["ner_labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask,
            )
            
            # Compute loss
            losses = self.compute_loss(outputs, relation_labels, ner_labels)
            loss = losses["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_relation_loss += losses["relation_loss"].item()
            if "ner_loss" in losses:
                total_ner_loss += losses["ner_loss"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        return {
            "train_loss": total_loss / num_batches,
            "train_relation_loss": total_relation_loss / num_batches,
            "train_ner_loss": total_ner_loss / num_batches if num_batches > 0 else 0.0,
        }
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            eval_loader: DataLoader for evaluation data
            compute_metrics_fn: Optional function to compute additional metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                drug1_mask = batch["drug1_mask"].to(self.device)
                drug2_mask = batch["drug2_mask"].to(self.device)
                relation_labels = batch["relation_label"].to(self.device)
                
                ner_labels = None
                if "ner_labels" in batch:
                    ner_labels = batch["ner_labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drug1_mask=drug1_mask,
                    drug2_mask=drug2_mask,
                )
                
                # Compute loss
                losses = self.compute_loss(outputs, relation_labels, ner_labels)
                total_loss += losses["total_loss"].item()
                
                # Get predictions
                probs = F.softmax(outputs["relation_logits"], dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(relation_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = {
            "val_loss": total_loss / len(eval_loader),
        }
        
        # Compute additional metrics if provided
        if compute_metrics_fn is not None:
            additional_metrics = compute_metrics_fn(
                predictions=np.array(all_preds),
                labels=np.array(all_labels),
                probabilities=np.array(all_probs),
            )
            metrics.update(additional_metrics)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        eval_steps: Optional[int] = None,
        save_dir: Optional[Path] = None,
        early_stopping_patience: int = 3,
        compute_metrics_fn: Optional[Callable] = None,
    ) -> Dict[str, List]:
        """
        Full training loop with validation and early stopping.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            eval_steps: Evaluate every N steps (if None, evaluate per epoch)
            save_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs without improvement before stopping
            compute_metrics_fn: Function to compute evaluation metrics
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer and scheduler
        num_training_steps = len(train_loader) * num_epochs
        self.setup_optimizer(num_training_steps)
        
        # Create save directory
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training batches per epoch: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, compute_metrics_fn)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            if compute_metrics_fn is not None:
                logger.info(f"Val Metrics: {val_metrics}")
            
            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_relation_loss"].append(train_metrics["train_relation_loss"])
            self.history["train_ner_loss"].append(train_metrics["train_ner_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_metrics"].append(val_metrics)
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                
                if save_dir is not None:
                    self.save_checkpoint(save_dir / "best_model.pt", epoch, val_metrics)
                    logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training complete!")
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_warmup_steps": self.num_warmup_steps,
                "aux_loss_weight": self.aux_loss_weight,
            },
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
