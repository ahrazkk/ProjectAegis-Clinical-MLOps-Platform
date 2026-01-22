"""
DDI Dataset for Drug-Drug Interaction Prediction
Handles data loading and preprocessing for training and evaluation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
import logging

from .tokenization import DDITokenizer


logger = logging.getLogger(__name__)


class DDIDataset(Dataset):
    """
    Dataset class for Drug-Drug Interaction prediction

    Expected data format (each sample):
    {
        "text": "Concurrent use of aspirin and warfarin may increase bleeding risk.",
        "drug1": {"name": "aspirin", "start": 19, "end": 26},
        "drug2": {"name": "warfarin", "start": 31, "end": 39},
        "interaction_type": 2,  # Class index (0=none, 1=minor, 2=moderate, 3=major)
        "has_interaction": 1    # Binary label for interaction presence
    }
    """

    # Interaction type mappings
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
        tokenizer: DDITokenizer,
        use_binary_labels: bool = True,
        lazy_loading: bool = False
    ):
        """
        Initialize DDI Dataset

        Args:
            data: List of sample dictionaries
            tokenizer: DDITokenizer instance
            use_binary_labels: If True, use binary interaction labels;
                             if False, use multi-class interaction types
            lazy_loading: If True, process samples on-demand during iteration (slower per-batch
                         but faster initialization and lower memory); if False, pre-process all 
                         samples during initialization (default, faster training)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.use_binary_labels = use_binary_labels
        self.lazy_loading = lazy_loading

        if lazy_loading:
            # Lazy loading: keep track of valid indices only
            logger.info("Using lazy loading mode - samples will be processed on-demand")
            self.processed_samples = None
            self.valid_indices = list(range(len(data)))
        else:
            # Eager loading: pre-process all samples
            self.processed_samples = self._preprocess_all()
            self.valid_indices = None

    def _preprocess_all(self) -> List[Dict[str, torch.Tensor]]:
        """Pre-process all samples for faster training"""
        processed = []
        failed_count = 0

        for sample in self.data:
            processed_sample = self._preprocess_sample(sample)
            if processed_sample is not None:
                processed.append(processed_sample)
            else:
                failed_count += 1

        if failed_count > 0:
            logger.warning(
                f"Dropped {failed_count} samples during preprocessing "
                f"({failed_count}/{len(self.data)} = {100*failed_count/len(self.data):.1f}%). "
                f"Successfully processed {len(processed)} samples."
            )

        return processed

    def _preprocess_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Preprocess a single sample

        Args:
            sample: Raw sample dictionary

        Returns:
            Processed sample with tokenized inputs and labels
        """
        try:
            text = sample['text']
            drug1 = sample['drug1']
            drug2 = sample['drug2']

            # Insert drug markers into text
            marked_text = self.tokenizer.mark_drugs_in_text(
                text,
                drug1_span=(drug1['start'], drug1['end']),
                drug2_span=(drug2['start'], drug2['end'])
            )

            # Tokenize
            tokenized = self.tokenizer.tokenize(marked_text)

            # Get labels
            if self.use_binary_labels:
                relation_label = torch.tensor(sample.get('has_interaction', 0), dtype=torch.float)
            else:
                relation_label = torch.tensor(sample.get('interaction_type', 0), dtype=torch.long)

            # Store severity as metadata (not in tensor dict)
            result = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'drug1_mask': tokenized['drug1_mask'],
                'drug2_mask': tokenized['drug2_mask'],
                'ner_labels': tokenized['ner_labels'],
                'relation_label': relation_label,
            }
            return result
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None

    def __len__(self) -> int:
        if self.lazy_loading:
            return len(self.valid_indices)
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.lazy_loading:
            # Process sample on-demand with robustness: skip invalid samples
            if not self.valid_indices:
                raise RuntimeError("No valid samples available in lazy-loading dataset.")

            # Ensure idx is within current bounds (may change if we drop invalid indices)
            idx = idx % len(self.valid_indices)

            attempts = 0
            max_attempts = len(self.valid_indices)

            while attempts < max_attempts and self.valid_indices:
                data_idx = self.valid_indices[idx]
                sample = self.data[data_idx]
                processed = self._preprocess_sample(sample)

                if processed is not None:
                    return processed

                # If processing failed, drop this index and try another one
                logger.warning(
                    "Removing invalid sample at data index %s from valid_indices during lazy loading.",
                    data_idx,
                )
                self.valid_indices.pop(idx)

                if not self.valid_indices:
                    break

                # Adjust idx to stay within new bounds and increment attempts
                idx = idx % len(self.valid_indices)
                attempts += 1

            raise RuntimeError("No valid samples available after removing invalid entries during lazy loading.")
        return self.processed_samples[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for DataLoader

        Args:
            batch: List of processed samples

        Returns:
            Batched tensors
        """
        return {
            'input_ids': torch.stack([s['input_ids'] for s in batch]),
            'attention_mask': torch.stack([s['attention_mask'] for s in batch]),
            'drug1_mask': torch.stack([s['drug1_mask'] for s in batch]),
            'drug2_mask': torch.stack([s['drug2_mask'] for s in batch]),
            'ner_labels': torch.stack([s['ner_labels'] for s in batch]),
            'relation_label': torch.stack([s['relation_label'] for s in batch])
        }

    @classmethod
    def from_json(
        cls,
        filepath: str,
        tokenizer: DDITokenizer,
        use_binary_labels: bool = True
    ) -> 'DDIDataset':
        """
        Load dataset from JSON file

        Args:
            filepath: Path to JSON file
            tokenizer: DDITokenizer instance
            use_binary_labels: Use binary or multi-class labels

        Returns:
            DDIDataset instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(data, tokenizer, use_binary_labels)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tokenizer: DDITokenizer,
        text_col: str = 'text',
        drug1_col: str = 'drug1',
        drug2_col: str = 'drug2',
        label_col: str = 'interaction_type',
        use_binary_labels: bool = True
    ) -> 'DDIDataset':
        """
        Load dataset from pandas DataFrame

        Args:
            df: Input DataFrame
            tokenizer: DDITokenizer instance
            text_col: Column name for text
            drug1_col: Column name for drug1 info (dict or JSON string)
            drug2_col: Column name for drug2 info
            label_col: Column name for interaction label
            use_binary_labels: Use binary or multi-class labels

        Returns:
            DDIDataset instance
        """
        data = []
        for _, row in df.iterrows():
            sample = {
                'text': row[text_col],
                'drug1': row[drug1_col] if isinstance(row[drug1_col], dict) else json.loads(row[drug1_col]),
                'drug2': row[drug2_col] if isinstance(row[drug2_col], dict) else json.loads(row[drug2_col]),
                'interaction_type': int(row[label_col]),
                'has_interaction': 1 if int(row[label_col]) > 0 else 0
            }
            data.append(sample)

        return cls(data, tokenizer, use_binary_labels)


def create_data_loaders(
    train_dataset: DDIDataset,
    val_dataset: DDIDataset,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders

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
        collate_fn=DDIDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DDIDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
