"""
DDI Relation Extraction Model
Implements the PubMedBERT-based architecture from the AI Model Specification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class DDIRelationModel(nn.Module):
    """
    Drug-Drug Interaction Relation Extraction Model
    
    Based on microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    with entity marker pooling and dual-head architecture.
    
    Architecture:
    1. Encoder: PubMedBERT with entity marker tokens ([DRUG1], [/DRUG1], [DRUG2], [/DRUG2])
    2. Relation Head: Binary/Multi-class DDI classification
    3. Auxiliary Head: NER task for entity boundary detection
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_relation_classes: int = 5,  # None, Mechanism, Effect, Advise, Int
        num_ner_classes: int = 5,  # O, B-DRUG, I-DRUG, B-BRAND, I-BRAND
        head_dropout_rate: float = 0.1,
        relation_hidden_dim: int = 768,
        freeze_encoder: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for PubMedBERT
            num_relation_classes: Number of DDI relation types
            num_ner_classes: Number of NER token classes
            head_dropout_rate: Dropout rate for task heads
            relation_hidden_dim: Hidden dimension for relation head
            freeze_encoder: Whether to freeze BERT encoder weights
        """
        super().__init__()
        
        # 1.1 Encoder Configuration
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Entity marker tokens will be added to tokenizer externally
        self.marker_tokens = ["[DRUG1]", "[/DRUG1]", "[DRUG2]", "[/DRUG2]"]
        
        # 1.2 Relation Head (Primary Task)
        # Input: Concatenated entity vectors [drug1_vec; drug2_vec] = 1536 dim
        self.relation_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, relation_hidden_dim),  # 1536 -> 768
            nn.GELU(),  # BERT internal activation
            nn.Dropout(head_dropout_rate),
            nn.LayerNorm(relation_hidden_dim, eps=1e-12),
            nn.Linear(relation_hidden_dim, num_relation_classes)
        )
        
        # 1.3 Auxiliary Head (NER Task)
        # Input: Sequence output (per-token hidden states)
        self.ner_head = nn.Sequential(
            nn.Dropout(head_dropout_rate),
            nn.Linear(self.hidden_size, num_ner_classes)
        )
        
        self.num_relation_classes = num_relation_classes
        self.num_ner_classes = num_ner_classes
        
    def mean_pool_markers(
        self,
        sequence_output: torch.Tensor,
        marker_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling of hidden states corresponding to marker tokens.
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            marker_mask: [batch_size, seq_len] - 1 for marker positions, 0 elsewhere
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Expand mask to match hidden dimension
        marker_mask_expanded = marker_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        
        # Sum hidden states at marker positions
        sum_embeddings = torch.sum(sequence_output * marker_mask_expanded, dim=1)
        
        # Count number of marker tokens
        sum_mask = marker_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        
        # Mean pooling
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            attention_mask: [batch_size, seq_len] - Attention mask
            drug1_mask: [batch_size, seq_len] - Mask for DRUG1 marker tokens
            drug2_mask: [batch_size, seq_len] - Mask for DRUG2 marker tokens
            token_type_ids: [batch_size, seq_len] - Token type IDs (optional)
            
        Returns:
            Dictionary containing:
                - relation_logits: [batch_size, num_relation_classes]
                - ner_logits: [batch_size, seq_len, num_ner_classes]
                - drug1_vec: [batch_size, hidden_size]
                - drug2_vec: [batch_size, hidden_size]
                - sequence_output: [batch_size, seq_len, hidden_size]
        """
        # 1.1 Encoder: Get sequence output from PubMedBERT
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, 768]
        
        # 1.1 Pooling Strategy: Mean pool entity marker tokens
        drug1_vec = self.mean_pool_markers(sequence_output, drug1_mask)  # [batch_size, 768]
        drug2_vec = self.mean_pool_markers(sequence_output, drug2_mask)  # [batch_size, 768]
        
        # 1.2 Relation Head: Concatenate entity vectors
        relation_input = torch.cat([drug1_vec, drug2_vec], dim=-1)  # [batch_size, 1536]
        relation_logits = self.relation_head(relation_input)  # [batch_size, num_classes]
        
        # 1.3 Auxiliary Head: Per-token NER classification
        ner_logits = self.ner_head(sequence_output)  # [batch_size, seq_len, num_ner_classes]
        
        return {
            "relation_logits": relation_logits,
            "ner_logits": ner_logits,
            "drug1_vec": drug1_vec,
            "drug2_vec": drug2_vec,
            "sequence_output": sequence_output,
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with softmax probabilities.
        
        Returns:
            Dictionary containing:
                - relation_probs: [batch_size, num_relation_classes]
                - relation_pred: [batch_size] - Predicted class indices
                - ner_probs: [batch_size, seq_len, num_ner_classes]
                - ner_pred: [batch_size, seq_len] - Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask,
                token_type_ids=token_type_ids,
            )
            
            relation_probs = F.softmax(outputs["relation_logits"], dim=-1)
            relation_pred = torch.argmax(relation_probs, dim=-1)
            
            ner_probs = F.softmax(outputs["ner_logits"], dim=-1)
            ner_pred = torch.argmax(ner_probs, dim=-1)
            
            return {
                "relation_probs": relation_probs,
                "relation_pred": relation_pred,
                "ner_probs": ner_probs,
                "ner_pred": ner_pred,
            }


class ModelWithTemperature(nn.Module):
    """
    Temperature Scaling Wrapper for Probability Calibration
    
    Implements the post-processing calibration required by Section 3.1 of the specification.
    Temperature Scaling divides logits by a learned temperature parameter T before softmax,
    producing better-calibrated probabilities for clinical use.
    
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self, model: DDIRelationModel, temperature: float = 1.5):
        """
        Args:
            model: The base DDI model to wrap
            temperature: Initial temperature value (default 1.5)
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temperature scaling applied to relation logits.
        
        Returns outputs with temperature-scaled logits.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            drug1_mask=drug1_mask,
            drug2_mask=drug2_mask,
            token_type_ids=token_type_ids,
        )
        
        # Apply temperature scaling to relation logits
        outputs["relation_logits_scaled"] = outputs["relation_logits"] / self.temperature
        
        return outputs
    
    def predict_calibrated(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        drug1_mask: torch.Tensor,
        drug2_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with temperature-scaled (calibrated) probabilities.
        
        Returns:
            Dictionary containing calibrated relation probabilities.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                drug1_mask=drug1_mask,
                drug2_mask=drug2_mask,
                token_type_ids=token_type_ids,
            )
            
            # Use temperature-scaled logits for calibrated probabilities
            relation_probs_calibrated = F.softmax(outputs["relation_logits_scaled"], dim=-1)
            relation_pred = torch.argmax(relation_probs_calibrated, dim=-1)
            
            return {
                "relation_probs_calibrated": relation_probs_calibrated,
                "relation_pred": relation_pred,
                "temperature": self.temperature.item(),
            }
    
    def set_temperature(self, valid_loader, criterion=nn.CrossEntropyLoss()):
        """
        Optimize temperature parameter using a validation set.
        
        This should be called after training the base model to find the optimal
        temperature that minimizes negative log-likelihood on the validation set.
        
        Args:
            valid_loader: DataLoader for validation data
            criterion: Loss function (typically NLLLoss or CrossEntropyLoss)
        """
        self.model.eval()
        
        # Collect logits and labels from validation set
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in valid_loader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    drug1_mask=batch["drug1_mask"],
                    drug2_mask=batch["drug2_mask"],
                )
                logits_list.append(outputs["relation_logits"])
                labels_list.append(batch["relation_label"])
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()
