"""
Relation Head for DDI Classification
Implements the primary task head for determining drug-drug interactions
"""

import torch
import torch.nn as nn


class RelationHead(nn.Module):
    """
    Relation Classification Head
    
    Architecture (Sequential):
    1. Dense Layer: 768 units, GELU activation
    2. Dropout Layer: Regularization
    3. LayerNorm: Stabilization (epsilon=1e-12)
    4. Output Layer: Binary (sigmoid) or Multi-class (softmax)
    
    Reference: MCR III Section 1.2
    """
    
    def __init__(
        self,
        input_dim: int = 1536,  # 768 + 768 (concatenated drug vectors)
        hidden_dim: int = 768,
        num_classes: int = 1,
        dropout_rate: float = 0.1,
        use_binary: bool = True
    ):
        """
        Initialize Relation Head
        
        Args:
            input_dim: Input dimension (concatenated drug1_vec and drug2_vec)
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (1 for binary, k for multi-class)
            dropout_rate: Dropout probability
            use_binary: If True, use sigmoid; if False, use softmax
        """
        super(RelationHead, self).__init__()
        
        self.use_binary = use_binary
        
        # Dense layer with GELU activation (matches BERT internal activation)
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization for stability (epsilon=1e-12 as per spec)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Dense layer with GELU activation
        x = self.dense(x)
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output layer (no activation - raw logits)
        logits = self.classifier(x)
        
        return logits
