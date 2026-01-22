"""
DDI Model Evaluation Module
Implements evaluation metrics, cross-validation, and error analysis
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import Counter
import json
from pathlib import Path

from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """
    Error categories for DDI prediction failures

    Reference: MCR III Section 4 - Error Analysis Protocol
    """
    NEGATION_MISINTERPRETATION = "negation_misinterpretation"
    IMPLICIT_INTERACTION = "implicit_interaction"
    CONTEXT_WINDOW_LIMITATION = "context_window_limitation"
    ENTITY_BOUNDARY_ERROR = "entity_boundary_error"
    RARE_INTERACTION_TYPE = "rare_interaction_type"
    AMBIGUOUS_LANGUAGE = "ambiguous_language"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    UNKNOWN = "unknown"


@dataclass
class PredictionError:
    """Represents a single prediction error for analysis"""
    sample_id: int
    text: str
    drug1: str
    drug2: str
    true_label: int
    predicted_label: int
    confidence: float
    error_type: ErrorType
    notes: str = ""


def calculate_pr_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Calculate Precision-Recall AUC

    Primary metric for model selection due to class imbalance.

    Reference: MCR III Section 4 - PR-AUC as primary metric

    Args:
        y_true: Ground truth binary labels
        y_scores: Predicted probability scores

    Returns:
        PR-AUC score
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. PR-AUC is undefined.")
        return 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return float(pr_auc)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_logits: np.ndarray,
    use_binary: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_logits: Raw model logits
        use_binary: Whether using binary classification

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if use_binary:
        # Convert logits to probabilities
        y_scores = 1 / (1 + np.exp(-y_logits.flatten()))  # sigmoid

        # PR-AUC (Primary Metric)
        metrics['pr_auc'] = calculate_pr_auc(y_true, y_scores)

        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0

        # Precision, Recall, F1
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Accuracy (reported but not used for selection)
        metrics['accuracy'] = np.mean(y_true == y_pred)

    else:
        # Multi-class metrics
        y_probs = np.exp(y_logits) / np.exp(y_logits).sum(axis=1, keepdims=True)

        # Macro-averaged PR-AUC
        pr_aucs = []
        for i in range(y_logits.shape[1]):
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() > 0:
                pr_aucs.append(calculate_pr_auc(binary_true, y_probs[:, i]))
        metrics['pr_auc'] = np.mean(pr_aucs) if pr_aucs else 0.0

        # Macro-averaged ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except ValueError:
            metrics['roc_auc'] = 0.0

        # Classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['accuracy'] = np.mean(y_true == y_pred)

    return metrics


class StratifiedKFoldValidator:
    """
    Stratified K-Fold Cross-Validation for DDI Model

    Ensures each fold contains representative samples of rare interaction classes.

    Reference: MCR III Section 4 - Stratified 10-fold Cross-Validation
    """

    def __init__(
        self,
        n_splits: int = 10,
        random_state: int = 42
    ):
        """
        Initialize validator

        Args:
            n_splits: Number of folds (default: 10)
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        self.fold_results = []

    def cross_validate(
        self,
        dataset,
        model_class,
        trainer_class,
        config,
        tokenizer,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation

        Args:
            dataset: Full dataset
            model_class: DDIModel class
            trainer_class: DDITrainer class
            config: TrainingConfig
            tokenizer: DDITokenizer
            labels: Array of labels for stratification

        Returns:
            Cross-validation results with mean and std of metrics
        """
        self.fold_results = []
        indices = np.arange(len(dataset))

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices, labels)):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            logger.info(f"{'='*50}")

            # Create fold datasets
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())

            # Create data loaders
            from .dataset import DDIDataset
            train_loader = DataLoader(
                train_subset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=DDIDataset.collate_fn
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=DDIDataset.collate_fn
            )

            # Initialize fresh model for each fold
            model = model_class(
                encoder_name=config.encoder_name,
                num_relation_classes=config.num_relation_classes,
                num_ner_classes=config.num_ner_classes,
                head_dropout_rate=config.head_dropout_rate,
                use_binary=config.use_binary
            )

            # Train
            trainer = trainer_class(
                model=model,
                tokenizer=tokenizer,
                config=config,
                output_dir=f'./checkpoints/fold_{fold + 1}'
            )

            results = trainer.train(train_loader, val_loader)

            # Store fold results
            fold_result = {
                'fold': fold + 1,
                'best_pr_auc': results['best_metric'],
                'history': results['training_history']
            }
            self.fold_results.append(fold_result)

            logger.info(f"Fold {fold + 1} Best PR-AUC: {results['best_metric']:.4f}")

        # Aggregate results
        pr_aucs = [r['best_pr_auc'] for r in self.fold_results]

        return {
            'mean_pr_auc': np.mean(pr_aucs),
            'std_pr_auc': np.std(pr_aucs),
            'fold_results': self.fold_results,
            'n_splits': self.n_splits
        }


class ErrorAnalyzer:
    """
    Error Analysis Protocol for DDI Predictions

    Categorizes and analyzes model failures for improvement insights.

    Reference: MCR III Section 4 - Error Analysis
    """

    def __init__(self):
        self.errors: List[PredictionError] = []
        self.error_counts: Counter = Counter()

    def add_error(
        self,
        sample_id: int,
        text: str,
        drug1: str,
        drug2: str,
        true_label: int,
        predicted_label: int,
        confidence: float,
        error_type: Optional[ErrorType] = None
    ):
        """Add a prediction error for analysis"""
        # Auto-classify error type if not provided
        if error_type is None:
            error_type = self._classify_error(
                text, true_label, predicted_label
            )

        error = PredictionError(
            sample_id=sample_id,
            text=text,
            drug1=drug1,
            drug2=drug2,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
            error_type=error_type
        )

        self.errors.append(error)
        self.error_counts[error_type] += 1

    def _classify_error(
        self,
        text: str,
        true_label: int,
        predicted_label: int
    ) -> ErrorType:
        """
        Automatically classify error type based on heuristics

        Args:
            text: Original text
            true_label: Ground truth label
            predicted_label: Model prediction

        Returns:
            Classified error type
        """
        text_lower = text.lower()

        # Check for negation patterns
        negation_patterns = [
            'no interaction', 'not interact', 'does not', 'do not',
            'without', 'unlikely', 'no evidence', 'not associated'
        ]
        if any(pattern in text_lower for pattern in negation_patterns):
            return ErrorType.NEGATION_MISINTERPRETATION

        # Check for implicit interaction language
        implicit_patterns = [
            'may', 'might', 'could', 'possibly', 'potential',
            'caution', 'monitor', 'consider'
        ]
        if any(pattern in text_lower for pattern in implicit_patterns):
            return ErrorType.IMPLICIT_INTERACTION

        # False positive vs false negative
        if true_label == 0 and predicted_label > 0:
            return ErrorType.FALSE_POSITIVE
        elif true_label > 0 and predicted_label == 0:
            return ErrorType.FALSE_NEGATIVE

        return ErrorType.UNKNOWN

    def get_summary(self) -> Dict[str, Any]:
        """Get error analysis summary"""
        total_errors = len(self.errors)

        summary = {
            'total_errors': total_errors,
            'error_distribution': {
                error_type.value: count
                for error_type, count in self.error_counts.most_common()
            },
            'error_percentages': {
                error_type.value: count / total_errors * 100
                for error_type, count in self.error_counts.items()
            } if total_errors > 0 else {}
        }

        return summary

    def get_examples_by_type(
        self,
        error_type: ErrorType,
        n_examples: int = 5
    ) -> List[PredictionError]:
        """Get example errors of a specific type"""
        examples = [e for e in self.errors if e.error_type == error_type]
        return examples[:n_examples]

    def save_report(self, filepath: str):
        """Save error analysis report to JSON"""
        report = {
            'summary': self.get_summary(),
            'errors': [
                {
                    'sample_id': e.sample_id,
                    'text': e.text[:200] + '...' if len(e.text) > 200 else e.text,
                    'drug1': e.drug1,
                    'drug2': e.drug2,
                    'true_label': e.true_label,
                    'predicted_label': e.predicted_label,
                    'confidence': e.confidence,
                    'error_type': e.error_type.value
                }
                for e in self.errors
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved error analysis report to {filepath}")

    def print_summary(self):
        """Print formatted error analysis summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nTotal Errors: {summary['total_errors']}")
        print("\nError Distribution:")
        print("-" * 40)

        for error_type, count in summary['error_distribution'].items():
            percentage = summary['error_percentages'].get(error_type, 0)
            bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
            print(f"  {error_type:30s} {bar} {count:4d} ({percentage:5.1f}%)")

        print("=" * 60)


def evaluate_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    use_binary: bool = True,
    error_analyzer: Optional[ErrorAnalyzer] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation

    Args:
        model: Trained DDIModel
        test_loader: Test DataLoader
        device: Computation device
        use_binary: Binary or multi-class classification
        error_analyzer: Optional error analyzer for failure analysis

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            drug1_mask = batch['drug1_mask'].to(device)
            drug2_mask = batch['drug2_mask'].to(device)
            relation_labels = batch['relation_label']

            relation_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask
            )

            if use_binary:
                probs = torch.sigmoid(relation_logits.squeeze(-1))
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(relation_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(relation_labels.numpy())
            all_logits.extend(relation_logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_logits),
        use_binary=use_binary
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()

    logger.info("\nEvaluation Results:")
    logger.info(f"  PR-AUC (Primary): {metrics['pr_auc']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")

    return metrics
