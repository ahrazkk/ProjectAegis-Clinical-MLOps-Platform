"""
Auxiliary Head for Named Entity Recognition (NER)
Implements token classification to regularize the model by learning entity boundaries
"""

import torch
import torch.nn as nn


class AuxiliaryHead(nn.Module):
    """
    Token Classification Head for NER Task
    
    Architecture:
    1. Dropout Layer: Regularization
    2. Output Layer: Dense with softmax activation
    
    Purpose: Regularizes the model by forcing it to learn entity boundaries
    Reference: MCR III Section 1.3
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT hidden size
        num_classes: int = 3,   # e.g., O, B-DRUG, I-DRUG
        dropout_rate: float = 0.1
    ):
        """
        Initialize Auxiliary Head
        
        Args:
            input_dim: Input dimension (BERT hidden size)
            num_classes: Number of NER classes
            dropout_rate: Dropout probability
        """
        super(AuxiliaryHead, self).__init__()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer for token classification
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            sequence_output: Hidden states from encoder [batch_size, seq_len, input_dim]
        
        Returns:
            logits: Token classification logits [batch_size, seq_len, num_classes]
        """
        # Apply dropout
        x = self.dropout(sequence_output)
        
        # Token classification (no activation - raw logits)
        logits = self.classifier(x)
        
        return logits
