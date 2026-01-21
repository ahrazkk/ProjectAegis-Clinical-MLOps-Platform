"""
Evaluation Protocol for DDI Relation Extraction
Implements Section 4 of the AI Model Specification
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from .ddi_model import DDIRelationModel, ModelWithTemperature
from .trainer import DDITrainer

logger = logging.getLogger(__name__)


class DDIEvaluator:
    """
    Evaluation Protocol for DDI Relation Extraction
    
    Implements Section 4 requirements:
    - Primary Metric: PR-AUC (Precision-Recall Area Under Curve)
    - Validation Method: Stratified 10-fold Cross-Validation
    - Error Analysis: Manual review by error type
    """
    
    def __init__(
        self,
        model: DDIRelationModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: Trained DDIRelationModel instance
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
    
    def compute_pr_auc(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute PR-AUC (Primary Metric from Section 4).
        
        For multi-class, computes macro-averaged and per-class PR-AUC.
        
        Args:
            labels: [num_samples] - True class labels
            probabilities: [num_samples, num_classes] - Predicted probabilities
            class_names: List of class names for reporting
            
        Returns:
            Dictionary with PR-AUC scores
        """
        num_classes = probabilities.shape[1]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        # Macro-averaged PR-AUC (Section 4: Primary Metric)
        macro_pr_auc = average_precision_score(
            labels,
            probabilities,
            average="macro",
        )
        
        # Per-class PR-AUC
        per_class_pr_auc = {}
        for i, class_name in enumerate(class_names):
            # Binary labels for this class
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]
            
            # Skip if no positive samples for this class
            if binary_labels.sum() == 0:
                per_class_pr_auc[class_name] = 0.0
                continue
            
            pr_auc = average_precision_score(binary_labels, class_probs)
            per_class_pr_auc[class_name] = pr_auc
        
        return {
            "macro_pr_auc": macro_pr_auc,
            "per_class_pr_auc": per_class_pr_auc,
        }
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: [num_samples] - Predicted class indices
            labels: [num_samples] - True class labels
            probabilities: [num_samples, num_classes] - Predicted probabilities
            class_names: List of class names
            
        Returns:
            Dictionary of metrics
        """
        # PR-AUC (Primary Metric - Section 4)
        pr_auc_metrics = self.compute_pr_auc(labels, probabilities, class_names)
        
        # Additional metrics
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
        macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
        
        # Accuracy (explicitly rejected in Section 4 due to class imbalance)
        accuracy = (predictions == labels).mean()
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            "pr_auc": pr_auc_metrics["macro_pr_auc"],  # Primary metric
            "per_class_pr_auc": pr_auc_metrics["per_class_pr_auc"],
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "accuracy": accuracy,  # Reported but not primary metric
            "confusion_matrix": cm,
        }
        
        return metrics
    
    def evaluate_model(
        self,
        eval_loader: DataLoader,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            eval_loader: DataLoader for evaluation data
            class_names: List of class names
            
        Returns:
            Evaluation metrics dictionary
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                drug1_mask = batch["drug1_mask"].to(self.device)
                drug2_mask = batch["drug2_mask"].to(self.device)
                relation_labels = batch["relation_label"]
                
                # Get predictions
                pred_outputs = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drug1_mask=drug1_mask,
                    drug2_mask=drug2_mask,
                )
                
                all_preds.extend(pred_outputs["relation_pred"].cpu().numpy())
                all_labels.extend(relation_labels.numpy())
                all_probs.extend(pred_outputs["relation_probs"].cpu().numpy())
        
        # Compute metrics
        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        probabilities = np.array(all_probs)
        
        metrics = self.compute_metrics(predictions, labels, probabilities, class_names)
        
        # Add classification report
        if class_names is not None:
            report = classification_report(
                labels,
                predictions,
                target_names=class_names,
                zero_division=0,
            )
            metrics["classification_report"] = report
        
        return metrics
    
    def stratified_k_fold_cv(
        self,
        dataset,
        k: int = 10,
        trainer_factory: callable = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Stratified K-Fold Cross-Validation (Section 4 requirement).
        
        Section 4: "Stratified 10-fold Cross-Validation. Stratification is 
        mandatory to preserve rare interaction classes in every fold."
        
        Args:
            dataset: Full dataset
            k: Number of folds (default 10 as per Section 4)
            trainer_factory: Function that creates a new trainer instance
            class_names: List of class names
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting Stratified {k}-Fold Cross-Validation")
        
        # Extract labels for stratification
        labels = np.array([example["relation_label"] for example in dataset.examples])
        
        # Stratified K-Fold splitter
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_metrics = []
        fold_pr_aucs = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
            logger.info(f"Fold {fold_idx}/{k}")
            
            # Create train/val subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
            
            # Create and train model for this fold
            if trainer_factory is not None:
                trainer = trainer_factory()
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=5,  # Fewer epochs for CV
                    compute_metrics_fn=lambda p, l, pr: self.compute_metrics(p, l, pr, class_names),
                )
                self.model = trainer.model
            
            # Evaluate on validation fold
            fold_result = self.evaluate_model(val_loader, class_names)
            fold_metrics.append(fold_result)
            fold_pr_aucs.append(fold_result["pr_auc"])
            
            logger.info(f"Fold {fold_idx} PR-AUC: {fold_result['pr_auc']:.4f}")
        
        # Aggregate results across folds
        mean_pr_auc = np.mean(fold_pr_aucs)
        std_pr_auc = np.std(fold_pr_aucs)
        
        logger.info(f"Cross-Validation Complete")
        logger.info(f"Mean PR-AUC: {mean_pr_auc:.4f} ± {std_pr_auc:.4f}")
        
        return {
            "mean_pr_auc": mean_pr_auc,
            "std_pr_auc": std_pr_auc,
            "fold_pr_aucs": fold_pr_aucs,
            "fold_metrics": fold_metrics,
        }
    
    def error_analysis(
        self,
        eval_loader: DataLoader,
        error_type_classifier: Optional[callable] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Error Analysis (Section 4 requirement).
        
        Section 4: "Manual review of failures classified by specific error types 
        (e.g., 'Negation Misinterpretation', 'Implicit Interaction')"
        
        Args:
            eval_loader: DataLoader for evaluation data
            error_type_classifier: Function to classify error types
            class_names: List of class names
            
        Returns:
            Dictionary with error analysis results
        """
        self.model.eval()
        
        errors = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                drug1_mask = batch["drug1_mask"].to(self.device)
                drug2_mask = batch["drug2_mask"].to(self.device)
                relation_labels = batch["relation_label"]
                
                # Get predictions
                pred_outputs = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    drug1_mask=drug1_mask,
                    drug2_mask=drug2_mask,
                )
                
                predictions = pred_outputs["relation_pred"].cpu().numpy()
                labels = relation_labels.numpy()
                probs = pred_outputs["relation_probs"].cpu().numpy()
                
                # Find errors
                for i in range(len(predictions)):
                    if predictions[i] != labels[i]:
                        error_info = {
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "predicted": int(predictions[i]),
                            "true_label": int(labels[i]),
                            "confidence": float(probs[i][predictions[i]]),
                            "probabilities": probs[i].tolist(),
                        }
                        
                        if class_names is not None:
                            error_info["predicted_class"] = class_names[predictions[i]]
                            error_info["true_class"] = class_names[labels[i]]
                        
                        # Classify error type if classifier provided
                        if error_type_classifier is not None:
                            error_type = error_type_classifier(batch, i, error_info)
                            error_info["error_type"] = error_type
                        
                        errors.append(error_info)
        
        # Aggregate error statistics
        error_stats = {
            "total_errors": len(errors),
            "errors": errors,
        }
        
        if error_type_classifier is not None:
            # Count errors by type
            error_types = {}
            for error in errors:
                etype = error.get("error_type", "Unknown")
                error_types[etype] = error_types.get(etype, 0) + 1
            error_stats["error_types"] = error_types
        
        logger.info(f"Error Analysis: {len(errors)} errors found")
        
        return error_stats
    
    def plot_precision_recall_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Plot Precision-Recall curves for each class.
        
        Args:
            labels: True class labels
            probabilities: Predicted probabilities
            class_names: List of class names
            save_path: Path to save figure
        """
        num_classes = probabilities.shape[1]
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]
            
            if binary_labels.sum() == 0:
                continue
            
            precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f"{class_name} (AUC={pr_auc:.3f})")
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid(True)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curve to {save_path}")
        
        plt.close()
