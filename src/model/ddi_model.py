"""
DDI Prediction Model Architecture
Implements the complete model with PubMedBERT encoder, Relation Head, and Auxiliary Head
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional, Tuple

from .relation_head import RelationHead
from .auxiliary_head import AuxiliaryHead


class DDIModel(nn.Module):
    """
    Drug-Drug Interaction Prediction Model
    
    Architecture:
    - Encoder: PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
    - Relation Head: Binary or Multi-class interaction classification
    - Auxiliary Head: Named Entity Recognition (NER) task for regularization
    
    Reference: MCR III Model Specification
    """
    
    def __init__(
        self,
        encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_relation_classes: int = 1,
        num_ner_classes: int = 3,
        head_dropout_rate: float = 0.1,
        use_binary: bool = True,
        freeze_encoder: bool = False
    ):
        """
        Initialize DDI Model
        
        Args:
            encoder_name: Pretrained BERT model identifier
            num_relation_classes: Number of interaction types (1 for binary, k for multi-class)
            num_ner_classes: Number of NER classes (e.g., O, B-DRUG, I-DRUG)
            head_dropout_rate: Dropout rate for classification heads
            use_binary: If True, use sigmoid activation; if False, use softmax
            freeze_encoder: If True, freeze encoder weights during training
        """
        super(DDIModel, self).__init__()
        
        # Load PubMedBERT encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Relation Head for DDI classification
        self.relation_head = RelationHead(
            input_dim=self.hidden_size * 2,  # Concatenated drug vectors
            hidden_dim=self.hidden_size,
            num_classes=num_relation_classes,
            dropout_rate=head_dropout_rate,
            use_binary=use_binary
        )
        
        # Auxiliary Head for NER task
        self.auxiliary_head = AuxiliaryHead(
            input_dim=self.hidden_size,
            num_classes=num_ner_classes,
            dropout_rate=head_dropout_rate
        )
        
        self.use_binary = use_binary
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            drug1_mask: Binary mask for DRUG1 marker tokens [batch_size, seq_len]
            drug2_mask: Binary mask for DRUG2 marker tokens [batch_size, seq_len]
            token_type_ids: Token type IDs (optional) [batch_size, seq_len]
        
        Returns:
            relation_logits: Interaction classification logits [batch_size, num_relation_classes]
            ner_logits: Token classification logits [batch_size, seq_len, num_ner_classes]
        """
        # Encode input through PubMedBERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get last hidden state [batch_size, seq_len, hidden_size]
        sequence_output = encoder_outputs.last_hidden_state
        
        # Extract drug vectors using mean pooling over marker tokens
        drug1_vec = self._mean_pooling(sequence_output, drug1_mask)  # [batch_size, hidden_size]
        drug2_vec = self._mean_pooling(sequence_output, drug2_mask)  # [batch_size, hidden_size]
        
        # Concatenate drug vectors for relation classification
        combined_vec = torch.cat([drug1_vec, drug2_vec], dim=-1)  # [batch_size, hidden_size * 2]
        
        # Relation Head: DDI classification
        relation_logits = self.relation_head(combined_vec)
        
        # Auxiliary Head: NER classification
        ner_logits = self.auxiliary_head(sequence_output)
        
        return relation_logits, ner_logits
    
    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over masked tokens
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            mask: Binary mask [batch_size, seq_len]
        
        Returns:
            pooled_output: Mean pooled vectors [batch_size, hidden_size]
        """
        # Expand mask to match hidden_states dimensions
        mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum hidden states where mask is 1
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        
        # Count non-zero mask positions
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def get_relation_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get calibrated probabilities for relation classification
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            drug1_mask: DRUG1 marker mask
            drug2_mask: DRUG2 marker mask
            token_type_ids: Token type IDs (optional)
        
        Returns:
            probabilities: Relation classification probabilities
        """
        relation_logits, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drug1_mask=drug1_mask,
            drug2_mask=drug2_mask,
            token_type_ids=token_type_ids
        )
        
        if self.use_binary:
            probs = torch.sigmoid(relation_logits)
        else:
            probs = torch.softmax(relation_logits, dim=-1)
        
        return probs
