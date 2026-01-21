"""
Data Preprocessing for DDI Relation Extraction
Handles tokenization with entity marker injection as per Section 1.1 of specification
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import re


class DDIDataPreprocessor:
    """
    Preprocessor for DDI Relation Extraction
    
    Handles:
    1. Entity marker token injection ([DRUG1], [/DRUG1], [DRUG2], [/DRUG2])
    2. Tokenization with PubMedBERT tokenizer
    3. Creation of entity marker masks for pooling
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add entity marker tokens to tokenizer
        self.marker_tokens = ["[DRUG1]", "[/DRUG1]", "[DRUG2]", "[/DRUG2]"]
        num_added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.marker_tokens
        })
        
        print(f"Added {num_added} special marker tokens to tokenizer")
        
        # Store token IDs for marker detection
        self.drug1_start_id = self.tokenizer.convert_tokens_to_ids("[DRUG1]")
        self.drug1_end_id = self.tokenizer.convert_tokens_to_ids("[/DRUG1]")
        self.drug2_start_id = self.tokenizer.convert_tokens_to_ids("[DRUG2]")
        self.drug2_end_id = self.tokenizer.convert_tokens_to_ids("[/DRUG2]")
    
    def inject_entity_markers(
        self,
        text: str,
        drug1_span: Tuple[int, int],
        drug2_span: Tuple[int, int],
    ) -> str:
        """
        Inject entity marker tokens around drug mentions.
        
        Args:
            text: Original text
            drug1_span: (start, end) character offsets for first drug entity
            drug2_span: (start, end) character offsets for second drug entity
            
        Returns:
            Text with markers: "... [DRUG1] aspirin [/DRUG1] ... [DRUG2] warfarin [/DRUG2] ..."
        """
        # Sort spans by position (left to right)
        spans = [(drug1_span, 1), (drug2_span, 2)]
        spans.sort(key=lambda x: x[0][0])
        
        # Build marked text from right to left to preserve offsets
        marked_text = text
        for (start, end), drug_id in reversed(spans):
            if drug_id == 1:
                marked_text = (
                    marked_text[:start] +
                    f"[DRUG1] {marked_text[start:end]} [/DRUG1]" +
                    marked_text[end:]
                )
            else:
                marked_text = (
                    marked_text[:start] +
                    f"[DRUG2] {marked_text[start:end]} [/DRUG2]" +
                    marked_text[end:]
                )
        
        return marked_text
    
    def create_marker_masks(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create binary masks for DRUG1 and DRUG2 marker token positions.
        
        Args:
            input_ids: [seq_len] or [batch_size, seq_len] - Token IDs
            
        Returns:
            drug1_mask: Binary mask for [DRUG1] ... [/DRUG1] tokens
            drug2_mask: Binary mask for [DRUG2] ... [/DRUG2] tokens
        """
        is_batched = input_ids.dim() == 2
        if not is_batched:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        
        drug1_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        drug2_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for i in range(batch_size):
            # Find DRUG1 markers
            drug1_starts = (input_ids[i] == self.drug1_start_id).nonzero(as_tuple=True)[0]
            drug1_ends = (input_ids[i] == self.drug1_end_id).nonzero(as_tuple=True)[0]
            
            if len(drug1_starts) > 0 and len(drug1_ends) > 0:
                start_idx = drug1_starts[0].item()
                end_idx = drug1_ends[0].item()
                drug1_mask[i, start_idx:end_idx+1] = True
            
            # Find DRUG2 markers
            drug2_starts = (input_ids[i] == self.drug2_start_id).nonzero(as_tuple=True)[0]
            drug2_ends = (input_ids[i] == self.drug2_end_id).nonzero(as_tuple=True)[0]
            
            if len(drug2_starts) > 0 and len(drug2_ends) > 0:
                start_idx = drug2_starts[0].item()
                end_idx = drug2_ends[0].item()
                drug2_mask[i, start_idx:end_idx+1] = True
        
        if not is_batched:
            drug1_mask = drug1_mask.squeeze(0)
            drug2_mask = drug2_mask.squeeze(0)
        
        return drug1_mask, drug2_mask
    
    def encode_single(
        self,
        text: str,
        drug1_span: Tuple[int, int],
        drug2_span: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single example with entity markers.
        
        Args:
            text: Original text
            drug1_span: (start, end) for first drug entity
            drug2_span: (start, end) for second drug entity
            
        Returns:
            Dictionary with input_ids, attention_mask, drug1_mask, drug2_mask
        """
        # Inject markers
        marked_text = self.inject_entity_markers(text, drug1_span, drug2_span)
        
        # Tokenize
        encoding = self.tokenizer(
            marked_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create marker masks
        input_ids = encoding["input_ids"].squeeze(0)
        drug1_mask, drug2_mask = self.create_marker_masks(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "drug1_mask": drug1_mask,
            "drug2_mask": drug2_mask,
        }
    
    def encode_batch(
        self,
        texts: List[str],
        drug1_spans: List[Tuple[int, int]],
        drug2_spans: List[Tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of examples with entity markers.
        
        Args:
            texts: List of text strings
            drug1_spans: List of (start, end) tuples for first drug
            drug2_spans: List of (start, end) tuples for second drug
            
        Returns:
            Dictionary with batched tensors
        """
        # Inject markers for all examples
        marked_texts = [
            self.inject_entity_markers(text, d1_span, d2_span)
            for text, d1_span, d2_span in zip(texts, drug1_spans, drug2_spans)
        ]
        
        # Tokenize batch
        encoding = self.tokenizer(
            marked_texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Create marker masks
        drug1_masks, drug2_masks = self.create_marker_masks(encoding["input_ids"])
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "drug1_mask": drug1_masks,
            "drug2_mask": drug2_masks,
        }


class DDIDataset(Dataset):
    """
    PyTorch Dataset for DDI Relation Extraction
    
    Expects data in format:
    {
        "text": str,
        "drug1_span": (start, end),
        "drug2_span": (start, end),
        "relation_label": int,  # Relation class index
        "ner_labels": List[int],  # Optional: token-level NER labels
    }
    """
    
    def __init__(
        self,
        examples: List[Dict],
        preprocessor: DDIDataPreprocessor,
        include_ner: bool = True,
    ):
        """
        Args:
            examples: List of data examples
            preprocessor: DDIDataPreprocessor instance
            include_ner: Whether to include NER labels
        """
        self.examples = examples
        self.preprocessor = preprocessor
        self.include_ner = include_ner
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Encode text with entity markers
        encoding = self.preprocessor.encode_single(
            text=example["text"],
            drug1_span=example["drug1_span"],
            drug2_span=example["drug2_span"],
        )
        
        # Add relation label
        encoding["relation_label"] = torch.tensor(example["relation_label"], dtype=torch.long)
        
        # Add NER labels if available
        if self.include_ner and "ner_labels" in example:
            ner_labels = example["ner_labels"]
            
            # Pad or truncate to match tokenized length
            max_len = encoding["input_ids"].size(0)
            if len(ner_labels) < max_len:
                ner_labels = ner_labels + [-100] * (max_len - len(ner_labels))
            else:
                ner_labels = ner_labels[:max_len]
            
            encoding["ner_labels"] = torch.tensor(ner_labels, dtype=torch.long)
        
        return encoding


def load_ddi_extraction_2013_corpus(
    corpus_path: str,
    split: str = "train",
) -> List[Dict]:
    """
    Load DDIExtraction 2013 corpus data.
    
    Args:
        corpus_path: Path to corpus directory
        split: "train", "dev", or "test"
        
    Returns:
        List of examples in standard format
    """
    # This is a placeholder - actual implementation would parse XML files
    # from the DDIExtraction 2013 corpus
    
    examples = []
    
    # TODO: Implement XML parsing for DDIExtraction 2013 corpus
    # Expected XML structure:
    # <sentence>
    #   <entity id="e1" type="drug" charOffset="10-17">aspirin</entity>
    #   <entity id="e2" type="drug" charOffset="25-32">warfarin</entity>
    #   <pair id="p1" e1="e1" e2="e2" ddi="true" type="effect"/>
    # </sentence>
    
    print(f"Warning: load_ddi_extraction_2013_corpus not fully implemented")
    print(f"Please implement XML parsing for {corpus_path}/{split}")
    
    return examples
