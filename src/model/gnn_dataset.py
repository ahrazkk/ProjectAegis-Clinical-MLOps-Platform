"""
GNN Dataset for Drug-Drug Interaction Prediction
Handles SMILES-based data loading and molecular graph preprocessing

Reference: MCR III - GNN Model data pipeline
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
from pathlib import Path
import logging

from .gnn_featurizer import MolecularGraphFeaturizer

logger = logging.getLogger(__name__)


class DDIGraphDataset(Dataset):
    """
    Dataset for GNN-based Drug-Drug Interaction prediction.

    Expected data format (each sample):
    {
        "drug1_smiles": "CC(=O)Oc1ccccc1C(=O)O",      # Aspirin
        "drug2_smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",  # Warfarin
        "drug1_name": "aspirin",                          # Optional
        "drug2_name": "warfarin",                         # Optional
        "interaction_type": 3,                             # Class index
        "has_interaction": 1                               # Binary label
    }
    """

    INTERACTION_TYPES = {
        0: 'none',
        1: 'advice',     # Minor
        2: 'effect',     # Moderate
        3: 'mechanism',  # Major (Pharmacokinetic)
        4: 'int'         # Major (Pharmacodynamic)
    }

    SEVERITY_MAPPING = {
        0: 'none',
        1: 'minor',
        2: 'moderate',
        3: 'major',
        4: 'major'
    }

    def __init__(
        self,
        data: List[Dict[str, Any]],
        featurizer: MolecularGraphFeaturizer,
        use_binary_labels: bool = True
    ):
        """
        Args:
            data: List of sample dictionaries with SMILES and labels
            featurizer: MolecularGraphFeaturizer instance
            use_binary_labels: If True, use binary labels; if False, multi-class
        """
        self.data = data
        self.featurizer = featurizer
        self.use_binary_labels = use_binary_labels

        self.processed_samples = self._preprocess_all()
        logger.info(
            f"DDIGraphDataset: {len(self.processed_samples)}/{len(data)} "
            f"samples successfully featurized"
        )

    def _preprocess_all(self) -> List[Dict[str, torch.Tensor]]:
        """Pre-process all samples by converting SMILES to graph tensors."""
        processed = []
        failed = 0

        for sample in self.data:
            result = self._preprocess_sample(sample)
            if result is not None:
                processed.append(result)
            else:
                failed += 1

        if failed > 0:
            logger.warning(f"Failed to featurize {failed} samples (invalid SMILES)")

        return processed

    def _preprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Preprocess a single sample: convert SMILES pair to graph tensors.

        Args:
            sample: Raw sample dictionary with SMILES and labels

        Returns:
            Processed sample with graph tensors and labels, or None if invalid
        """
        try:
            smiles1 = sample['drug1_smiles']
            smiles2 = sample['drug2_smiles']

            graphs = self.featurizer.smiles_pair_to_graphs(smiles1, smiles2)
            if graphs is None:
                return None

            # Get label
            if self.use_binary_labels:
                label = torch.tensor(
                    sample.get('has_interaction', 0), dtype=torch.float
                )
            else:
                label = torch.tensor(
                    sample.get('interaction_type', 0), dtype=torch.long
                )

            result = {
                'drug1_node_features': graphs['drug1_node_features'],
                'drug1_adjacency': graphs['drug1_adjacency'],
                'drug1_edge_features': graphs['drug1_edge_features'],
                'drug1_node_mask': graphs['drug1_node_mask'],
                'drug2_node_features': graphs['drug2_node_features'],
                'drug2_adjacency': graphs['drug2_adjacency'],
                'drug2_edge_features': graphs['drug2_edge_features'],
                'drug2_node_mask': graphs['drug2_node_mask'],
                'relation_label': label,
            }
            return result

        except Exception as e:
            logger.debug(f"Error processing sample: {e}")
            return None

    def __len__(self) -> int:
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_samples[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader.

        Args:
            batch: List of processed samples

        Returns:
            Batched tensors
        """
        return {
            'drug1_node_features': torch.stack([s['drug1_node_features'] for s in batch]),
            'drug1_adjacency': torch.stack([s['drug1_adjacency'] for s in batch]),
            'drug1_edge_features': torch.stack([s['drug1_edge_features'] for s in batch]),
            'drug1_node_mask': torch.stack([s['drug1_node_mask'] for s in batch]),
            'drug2_node_features': torch.stack([s['drug2_node_features'] for s in batch]),
            'drug2_adjacency': torch.stack([s['drug2_adjacency'] for s in batch]),
            'drug2_edge_features': torch.stack([s['drug2_edge_features'] for s in batch]),
            'drug2_node_mask': torch.stack([s['drug2_node_mask'] for s in batch]),
            'relation_label': torch.stack([s['relation_label'] for s in batch]),
        }

    @classmethod
    def from_json(
        cls,
        filepath: str,
        featurizer: MolecularGraphFeaturizer,
        use_binary_labels: bool = True
    ) -> 'DDIGraphDataset':
        """
        Load dataset from JSON file.

        Args:
            filepath: Path to JSON file
            featurizer: MolecularGraphFeaturizer instance
            use_binary_labels: Use binary or multi-class labels

        Returns:
            DDIGraphDataset instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data, featurizer, use_binary_labels)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        featurizer: MolecularGraphFeaturizer,
        smiles1_col: str = 'drug1_smiles',
        smiles2_col: str = 'drug2_smiles',
        name1_col: str = 'drug1_name',
        name2_col: str = 'drug2_name',
        label_col: str = 'interaction_type',
        use_binary_labels: bool = True
    ) -> 'DDIGraphDataset':
        """
        Load dataset from pandas DataFrame.

        Args:
            df: Input DataFrame with SMILES columns
            featurizer: MolecularGraphFeaturizer instance
            smiles1_col: Column name for drug1 SMILES
            smiles2_col: Column name for drug2 SMILES
            name1_col: Column name for drug1 name
            name2_col: Column name for drug2 name
            label_col: Column name for interaction label
            use_binary_labels: Use binary or multi-class labels

        Returns:
            DDIGraphDataset instance
        """
        data = []
        for _, row in df.iterrows():
            sample = {
                'drug1_smiles': row[smiles1_col],
                'drug2_smiles': row[smiles2_col],
                'interaction_type': int(row[label_col]),
                'has_interaction': 1 if int(row[label_col]) > 0 else 0
            }
            if name1_col in row:
                sample['drug1_name'] = row[name1_col]
            if name2_col in row:
                sample['drug2_name'] = row[name2_col]
            data.append(sample)

        return cls(data, featurizer, use_binary_labels)


def create_graph_data_loaders(
    train_dataset: DDIGraphDataset,
    val_dataset: DDIGraphDataset,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders for graph data.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DDIGraphDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DDIGraphDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
