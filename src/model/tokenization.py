"""
DDI Tokenizer with Special Drug Marker Tokens
Handles text tokenization with [DRUG1], [/DRUG1], [DRUG2], [/DRUG2] markers
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import re


class DDITokenizer:
    """
    Tokenizer for Drug-Drug Interaction text with entity markers.

    Special Tokens:
    - [DRUG1], [/DRUG1]: Markers for the first drug entity
    - [DRUG2], [/DRUG2]: Markers for the second drug entity

    Reference: MCR III Section 1.1 - Input Processing
    """

    SPECIAL_TOKENS = {
        'drug1_start': '[DRUG1]',
        'drug1_end': '[/DRUG1]',
        'drug2_start': '[DRUG2]',
        'drug2_end': '[/DRUG2]'
    }

    # NER labels for auxiliary task
    NER_LABELS = {
        'O': 0,       # Outside
        'B-DRUG': 1,  # Beginning of drug entity
        'I-DRUG': 2   # Inside drug entity
    }

    def __init__(
        self,
        encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length: int = 512
    ):
        """
        Initialize DDI Tokenizer

        Args:
            encoder_name: Pretrained tokenizer model name
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.max_length = max_length

        # Add special drug marker tokens
        special_tokens = list(self.SPECIAL_TOKENS.values())
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        # Store token IDs for marker identification
        self.drug1_start_id = self.tokenizer.convert_tokens_to_ids('[DRUG1]')
        self.drug1_end_id = self.tokenizer.convert_tokens_to_ids('[/DRUG1]')
        self.drug2_start_id = self.tokenizer.convert_tokens_to_ids('[DRUG2]')
        self.drug2_end_id = self.tokenizer.convert_tokens_to_ids('[/DRUG2]')

    def get_vocab_size(self) -> int:
        """Get vocabulary size including special tokens"""
        return len(self.tokenizer)

    def mark_drugs_in_text(
        self,
        text: str,
        drug1_span: Tuple[int, int],
        drug2_span: Tuple[int, int]
    ) -> str:
        """
        Insert drug markers around drug mentions in text

        Args:
            text: Original text
            drug1_span: Character span (start, end) for first drug
            drug2_span: Character span (start, end) for second drug

        Returns:
            Marked text with drug entity markers
        """
        # Handle overlapping spans - sort by position
        spans = [(drug1_span, '1'), (drug2_span, '2')]
        spans.sort(key=lambda x: x[0][0], reverse=True)  # Process from end to avoid offset issues

        marked_text = text
        for span, drug_num in spans:
            start, end = span
            if drug_num == '1':
                marked_text = (
                    marked_text[:start] +
                    self.SPECIAL_TOKENS['drug1_start'] + ' ' +
                    marked_text[start:end] + ' ' +
                    self.SPECIAL_TOKENS['drug1_end'] +
                    marked_text[end:]
                )
            else:
                marked_text = (
                    marked_text[:start] +
                    self.SPECIAL_TOKENS['drug2_start'] + ' ' +
                    marked_text[start:end] + ' ' +
                    self.SPECIAL_TOKENS['drug2_end'] +
                    marked_text[end:]
                )

        return marked_text

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and create drug entity masks

        Args:
            text: Text with drug markers already inserted
            return_tensors: Output format ("pt" for PyTorch)

        Returns:
            Dictionary containing:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - drug1_mask: Binary mask for DRUG1 tokens
            - drug2_mask: Binary mask for DRUG2 tokens
            - ner_labels: NER labels for auxiliary task
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create drug masks
        drug1_mask = self._create_drug_mask(input_ids, self.drug1_start_id, self.drug1_end_id)
        drug2_mask = self._create_drug_mask(input_ids, self.drug2_start_id, self.drug2_end_id)

        # Create NER labels
        ner_labels = self._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'drug1_mask': drug1_mask,
            'drug2_mask': drug2_mask,
            'ner_labels': ner_labels
        }

    def batch_tokenize(
        self,
        texts: List[str],
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts

        Args:
            texts: List of texts with drug markers
            return_tensors: Output format

        Returns:
            Batched tokenization outputs
        """
        batch_outputs = [self.tokenize(text, return_tensors) for text in texts]

        return {
            'input_ids': torch.stack([o['input_ids'] for o in batch_outputs]),
            'attention_mask': torch.stack([o['attention_mask'] for o in batch_outputs]),
            'drug1_mask': torch.stack([o['drug1_mask'] for o in batch_outputs]),
            'drug2_mask': torch.stack([o['drug2_mask'] for o in batch_outputs]),
            'ner_labels': torch.stack([o['ner_labels'] for o in batch_outputs])
        }

    def _create_drug_mask(
        self,
        input_ids: torch.Tensor,
        start_token_id: int,
        end_token_id: int
    ) -> torch.Tensor:
        """
        Create binary mask for tokens between drug markers (inclusive)

        Args:
            input_ids: Token IDs
            start_token_id: ID of start marker token
            end_token_id: ID of end marker token

        Returns:
            Binary mask tensor
        """
        mask = torch.zeros_like(input_ids, dtype=torch.float)

        # Find marker positions
        start_positions = (input_ids == start_token_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == end_token_id).nonzero(as_tuple=True)[0]

        # Set mask to 1 between start and end markers (inclusive)
        if len(start_positions) > 0 and len(end_positions) > 0:
            start_idx = start_positions[0].item()
            end_idx = end_positions[0].item()
            mask[start_idx:end_idx + 1] = 1.0

        return mask

    def _create_ner_labels(
        self,
        input_ids: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create NER labels for auxiliary task

        Labels:
        - O (0): Outside any drug entity
        - B-DRUG (1): Beginning of drug entity
        - I-DRUG (2): Inside drug entity

        Args:
            input_ids: Token IDs
            drug1_mask: Mask for first drug
            drug2_mask: Mask for second drug

        Returns:
            NER label tensor
        """
        # Initialize all as O (outside)
        ner_labels = torch.zeros_like(input_ids, dtype=torch.long)

        # Combined drug mask (excluding marker tokens themselves)
        marker_ids = {
            self.drug1_start_id, self.drug1_end_id,
            self.drug2_start_id, self.drug2_end_id
        }

        # Process drug1 region
        drug1_positions = (drug1_mask > 0).nonzero(as_tuple=True)[0]
        for i, pos in enumerate(drug1_positions):
            if input_ids[pos].item() not in marker_ids:
                if i == 0 or input_ids[drug1_positions[i-1]].item() in marker_ids:
                    ner_labels[pos] = self.NER_LABELS['B-DRUG']
                else:
                    ner_labels[pos] = self.NER_LABELS['I-DRUG']

        # Process drug2 region
        drug2_positions = (drug2_mask > 0).nonzero(as_tuple=True)[0]
        for i, pos in enumerate(drug2_positions):
            if input_ids[pos].item() not in marker_ids:
                if i == 0 or input_ids[drug2_positions[i-1]].item() in marker_ids:
                    ner_labels[pos] = self.NER_LABELS['B-DRUG']
                else:
                    ner_labels[pos] = self.NER_LABELS['I-DRUG']

        return ner_labels

    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(input_ids, skip_special_tokens=False)

    def resize_model_embeddings(self, model) -> None:
        """
        Resize model embeddings to account for added special tokens

        Args:
            model: The DDIModel with encoder to resize
        """
        model.encoder.resize_token_embeddings(len(self.tokenizer))
