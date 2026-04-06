"""
GNN Retraining Pipeline

Workflow:
1. Export approved corrections from Neo4j
2. Merge with existing training data
3. Create train/val/test splits
4. Retrain the GraphSAGE model
5. Evaluate on held-out test set
6. Save model checkpoint with metrics
7. Log retraining event to audit trail

This can be triggered manually via management command or API endpoint.
"""

import hashlib
import json
import logging
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Severity label mapping (matches ConfidenceCalibrator)
SEVERITY_TO_LABEL = {
    'no_interaction': 0, 'none': 0,
    'minor': 1,
    'moderate': 2,
    'severe': 3, 'major': 3,
    'critical': 4, 'contraindicated': 4,
}
LABEL_TO_SEVERITY = {0: 'none', 1: 'minor', 2: 'moderate', 3: 'severe', 4: 'critical'}

# Default training configuration
DEFAULT_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 64,
    'patience': 10,
    'hidden_channels': 256,
    'out_channels': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'focal_alpha': 0.8,
    'focal_gamma': 2.0,
    'weight_decay': 1e-4,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

# Module-level status tracking for the API
_pipeline_status = {
    'state': 'idle',          # idle | running | completed | failed
    'last_run_at': None,
    'metrics': None,
    'checkpoint_path': None,
    'error': None,
    'correction_count': 0,
}
_status_lock = threading.Lock()


def get_pipeline_status() -> Dict[str, Any]:
    """Return a copy of the current pipeline status."""
    with _status_lock:
        return dict(_pipeline_status)


def _update_status(**kwargs):
    with _status_lock:
        _pipeline_status.update(kwargs)


class RetrainingPipeline:
    """
    End-to-end retraining pipeline for the Macroscopic GraphSAGE DDI model.

    The pipeline exports approved human corrections from Neo4j, merges them
    with the existing graph dataset, retrains the model, evaluates it, and
    saves a versioned checkpoint.  It does NOT automatically replace the
    live model -- the checkpoint must be promoted manually.
    """

    def __init__(self):
        self.web_dir = Path(__file__).resolve().parent.parent.parent  # web/
        self.data_dir = self.web_dir / 'data'
        self.checkpoint_dir = self.web_dir / 'models' / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Export corrections
    # ------------------------------------------------------------------
    def export_corrections(self) -> List[Dict]:
        """
        Query Neo4j for all approved corrections.

        Returns a list of dicts with keys:
        drug_a, drug_b, corrected_severity, approved_at, plus original
        prediction metadata.
        """
        try:
            from .correction_memory import get_correction_memory
            mem = get_correction_memory()
            corrections = mem.export_training_data()
            logger.info("Exported %d approved corrections for retraining", len(corrections))
            return corrections
        except Exception as e:
            logger.error("Failed to export corrections: %s", e)
            return []

    # ------------------------------------------------------------------
    # 2. Prepare training data
    # ------------------------------------------------------------------
    def prepare_training_data(
        self,
        corrections: List[Dict],
        existing_data_path: Optional[str] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Load the existing PyG graph dataset, inject correction-based edge
        labels, and produce train / val / test splits.

        Parameters
        ----------
        corrections : list of dict
            Output of ``export_corrections()``.
        existing_data_path : str, optional
            Path to the ``neo4j_gnn_dataset.pt`` file.  Defaults to
            ``web/data/neo4j_gnn_dataset.pt``.

        Returns
        -------
        (train_data, val_data, test_data) -- PyG Data objects produced by
        ``RandomLinkSplit``.
        """
        import torch
        import torch_geometric.transforms as T
        import pandas as pd

        dataset_path = Path(existing_data_path) if existing_data_path else self.data_dir / 'neo4j_gnn_dataset.pt'
        if not dataset_path.exists():
            raise FileNotFoundError(f"Graph dataset not found at {dataset_path}")

        data = torch.load(dataset_path, weights_only=False)
        logger.info("Loaded graph dataset: %d nodes, %d edges",
                     data.num_nodes, data.edge_index.size(1))

        # Build node name -> index mapping
        mapping_path = self.data_dir / 'node_mapping.csv'
        name_to_idx: Dict[str, int] = {}
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            for _, row in mapping_df.iterrows():
                if pd.notna(row.get('name')):
                    name_to_idx[str(row['name']).strip().lower()] = int(row['pyg_id'])

        # Apply corrections as additional positive edges (or override labels)
        if corrections and name_to_idx:
            new_src, new_dst = [], []
            for c in corrections:
                drug_a = c.get('drug_a', '').strip().lower()
                drug_b = c.get('drug_b', '').strip().lower()
                severity = c.get('corrected_severity', '').lower()
                label = SEVERITY_TO_LABEL.get(severity)
                if label is None:
                    continue
                idx_a = name_to_idx.get(drug_a)
                idx_b = name_to_idx.get(drug_b)
                if idx_a is None or idx_b is None:
                    logger.debug("Skipping correction %s<->%s: drug not in graph", drug_a, drug_b)
                    continue

                # For severity > 0 (i.e. there IS an interaction), add edge
                if label > 0:
                    new_src.extend([idx_a, idx_b])
                    new_dst.extend([idx_b, idx_a])

            if new_src:
                new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
                data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
                # Deduplicate
                data.edge_index = torch.unique(data.edge_index, dim=1)
                logger.info("Injected %d correction edges into the graph", len(new_src) // 2)

        # Create train/val/test splits via RandomLinkSplit
        config = DEFAULT_CONFIG
        transform = T.RandomLinkSplit(
            num_val=config['val_ratio'],
            num_test=config['test_ratio'],
            is_undirected=True,
            add_negative_train_samples=True,
        )
        train_data, val_data, test_data = transform(data)
        logger.info("Split sizes  train=%d  val=%d  test=%d",
                     train_data.edge_label.size(0),
                     val_data.edge_label.size(0),
                     test_data.edge_label.size(0))
        return train_data, val_data, test_data

    # ------------------------------------------------------------------
    # 3. Train model
    # ------------------------------------------------------------------
    def train_model(
        self,
        train_data,
        val_data,
        config: Optional[Dict] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Initialize a fresh MacroscopicDDIGNN and train it with early
        stopping on validation loss.

        Returns
        -------
        (model, metrics_dict)
        """
        import torch
        import sys

        cfg = {**DEFAULT_CONFIG, **(config or {})}

        # Import the model class
        model_src = self.web_dir.parent / 'src' / 'model'
        for p in [str(model_src), str(self.web_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from macroscopic_ddi_gnn import MacroscopicDDIGNN, FocalLoss

        model = MacroscopicDDIGNN(
            in_channels=train_data.num_features,
            hidden_channels=cfg['hidden_channels'],
            out_channels=cfg['out_channels'],
            num_layers=cfg['num_layers'],
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
        )
        criterion = FocalLoss(alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        metrics: Dict[str, Any] = {'epochs_run': 0}

        for epoch in range(1, cfg['epochs'] + 1):
            # --- train ---
            model.train()
            optimizer.zero_grad()
            out = model(train_data, train_data.edge_label_index)
            loss = criterion(out, train_data.edge_label.float())
            loss.backward()
            optimizer.step()

            # --- validate ---
            model.eval()
            with torch.no_grad():
                val_out = model(val_data, val_data.edge_label_index)
                val_loss = criterion(val_out, val_data.edge_label.float())

            val_loss_val = val_loss.item()

            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %03d | train_loss=%.4f | val_loss=%.4f",
                            epoch, loss.item(), val_loss_val)

            # Early stopping
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg['patience']:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, cfg['patience'])
                    break

            metrics['epochs_run'] = epoch

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        metrics.update({
            'final_train_loss': loss.item(),
            'best_val_loss': best_val_loss,
        })
        return model, metrics

    # ------------------------------------------------------------------
    # 4. Evaluate model
    # ------------------------------------------------------------------
    def evaluate_model(self, model, test_data) -> Dict[str, float]:
        """
        Evaluate the trained model on the held-out test split.

        Returns a dict with: accuracy, precision, recall, f1, auc_roc.
        """
        import torch
        import numpy as np

        model.eval()
        with torch.no_grad():
            out = model(test_data, test_data.edge_label_index)
            probs = torch.sigmoid(out).cpu().numpy()
            labels = test_data.edge_label.cpu().numpy()

        preds = (probs >= 0.5).astype(int)

        # Compute metrics safely (handle edge cases with try/except)
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        metrics: Dict[str, float] = {}
        metrics['accuracy'] = float(accuracy_score(labels, preds))
        metrics['precision'] = float(precision_score(labels, preds, zero_division=0))
        metrics['recall'] = float(recall_score(labels, preds, zero_division=0))
        metrics['f1'] = float(f1_score(labels, preds, zero_division=0))

        try:
            metrics['auc_roc'] = float(roc_auc_score(labels, probs))
        except ValueError:
            # Only one class present in labels
            metrics['auc_roc'] = 0.0

        logger.info("Test metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # 5. Save checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        model,
        metrics: Dict[str, Any],
        version_tag: Optional[str] = None,
        correction_count: int = 0,
    ) -> str:
        """
        Save model weights and metadata to a versioned checkpoint file.

        Returns the absolute path to the saved checkpoint.
        """
        import torch

        now = datetime.now(timezone.utc)
        date_str = now.strftime('%Y%m%d_%H%M%S')

        if version_tag is None:
            # Auto-increment: count existing checkpoints
            existing = list(self.checkpoint_dir.glob('gnn_v*_*.pt'))
            version_tag = str(len(existing) + 1)

        filename = f"gnn_v{version_tag}_{date_str}.pt"
        filepath = self.checkpoint_dir / filename

        # Compute a hash of the model weights for integrity verification
        state_dict = model.state_dict()
        weight_bytes = b''
        for v in state_dict.values():
            weight_bytes += v.cpu().numpy().tobytes()
        data_hash = hashlib.sha256(weight_bytes).hexdigest()[:16]

        checkpoint = {
            'model_state_dict': state_dict,
            'metadata': {
                'version_tag': version_tag,
                'training_date': now.isoformat(),
                'correction_count': correction_count,
                'data_hash': data_hash,
                'metrics': metrics,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info("Checkpoint saved: %s", filepath)
        return str(filepath)

    # ------------------------------------------------------------------
    # 6. Full pipeline orchestration
    # ------------------------------------------------------------------
    def run_full_pipeline(self, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Orchestrate the complete retraining workflow.

        Parameters
        ----------
        config : dict, optional
            Override default training hyperparameters.

        Returns
        -------
        dict with keys: status, correction_count, train_metrics,
        test_metrics, checkpoint_path, duration_seconds.
        """
        _update_status(state='running', error=None)
        start = time.time()
        result: Dict[str, Any] = {'status': 'failed'}

        try:
            # Step 1: Export corrections
            corrections = self.export_corrections()
            result['correction_count'] = len(corrections)
            logger.info("Step 1/6 complete: %d corrections exported", len(corrections))

            # Step 2: Prepare training data
            train_data, val_data, test_data = self.prepare_training_data(corrections)
            logger.info("Step 2/6 complete: data prepared")

            # Step 3: Train model
            model, train_metrics = self.train_model(train_data, val_data, config)
            result['train_metrics'] = train_metrics
            logger.info("Step 3/6 complete: training done (%d epochs)", train_metrics.get('epochs_run', 0))

            # Step 4: Evaluate
            test_metrics = self.evaluate_model(model, test_data)
            result['test_metrics'] = test_metrics
            logger.info("Step 4/6 complete: evaluation done")

            # Step 5: Save checkpoint
            checkpoint_path = self.save_checkpoint(
                model, {**train_metrics, **test_metrics},
                correction_count=len(corrections),
            )
            result['checkpoint_path'] = checkpoint_path
            logger.info("Step 5/6 complete: checkpoint saved")

            # Step 6: Audit log
            try:
                from .audit_service import AuditService
                AuditService.log_event(
                    event_type='retraining',
                    payload={
                        'correction_count': len(corrections),
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'checkpoint_path': checkpoint_path,
                    },
                    actor='retraining_pipeline',
                )
            except Exception as e:
                logger.warning("Audit logging for retraining failed: %s", e)

            duration = time.time() - start
            result['status'] = 'completed'
            result['duration_seconds'] = round(duration, 2)

            _update_status(
                state='completed',
                last_run_at=datetime.now(timezone.utc).isoformat(),
                metrics={**train_metrics, **test_metrics},
                checkpoint_path=checkpoint_path,
                correction_count=len(corrections),
                error=None,
            )
            logger.info("Retraining pipeline completed in %.1fs", duration)

        except Exception as e:
            duration = time.time() - start
            result['status'] = 'failed'
            result['error'] = str(e)
            result['duration_seconds'] = round(duration, 2)
            _update_status(
                state='failed',
                last_run_at=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )
            logger.exception("Retraining pipeline failed: %s", e)

        return result


# Singleton access
_pipeline_instance = None


def get_pipeline() -> RetrainingPipeline:
    """Get singleton RetrainingPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RetrainingPipeline()
    return _pipeline_instance
