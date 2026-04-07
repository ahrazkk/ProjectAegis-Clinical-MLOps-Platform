"""
Graph Neural Network for Drug-Drug Interaction Prediction
Pure PyTorch implementation - no torch_geometric dependency required

Architecture:
- GIN (Graph Isomorphism Network) encoder for molecular graphs
- Edge-conditioned message passing
- Multi-scale readout with jumping knowledge
- DDI prediction head matching RelationHead pattern

Reference: MCR III - GNN Model for structure-aware DDI prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .gnn_featurizer import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM


class EdgeConditionedGINConv(nn.Module):
    """
    Edge-Conditioned Graph Isomorphism Network Convolution Layer

    Extends standard GIN with bond (edge) feature conditioning:
        h_i' = MLP((1 + eps) * h_i + Σ_j (h_j * edge_transform(e_ij)))

    where e_ij are edge features transformed through a linear layer.

    Reference: Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        eps: float = 0.0,
        train_eps: bool = True
    ):
        """
        Args:
            in_dim: Input node feature dimension
            out_dim: Output node feature dimension
            edge_dim: Edge feature dimension
            eps: Initial epsilon value for self-loop weighting
            train_eps: Whether epsilon is a learnable parameter
        """
        super().__init__()

        # MLP for node update: 2-layer with BatchNorm and ReLU
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        # Edge feature transform: project edge features to node feature space
        self.edge_transform = nn.Linear(edge_dim, in_dim)

        # Learnable epsilon for self-loop importance
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for one GIN convolution layer.

        Args:
            x: Node features [batch, max_atoms, in_dim]
            adj: Adjacency matrix [batch, max_atoms, max_atoms]
                 (without self-loops for message passing; self-loops handled by eps)
            edge_features: Edge features [batch, max_atoms, max_atoms, edge_dim]
            node_mask: Node mask [batch, max_atoms]

        Returns:
            Updated node features [batch, max_atoms, out_dim]
        """
        B, N, D = x.shape

        # Remove self-loops from adjacency for message computation
        # (self-connection is handled separately via eps)
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        adj_no_self = adj * (1 - eye)

        # Transform edge features to gate neighbor messages
        # edge_features: [B, N, N, edge_dim] → [B, N, N, in_dim]
        edge_gates = torch.sigmoid(self.edge_transform(edge_features))

        # Compute neighbor messages: h_j * edge_gate(e_ij)
        # x_expanded: [B, 1, N, D] for broadcasting with adj [B, N, N]
        x_expanded = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        gated_messages = x_expanded * edge_gates  # [B, N, N, D]

        # Aggregate: weighted sum using adjacency
        # adj_no_self: [B, N, N] → [B, N, N, 1]
        adj_expanded = adj_no_self.unsqueeze(-1)
        neighbor_sum = (gated_messages * adj_expanded).sum(dim=2)  # [B, N, D]

        # GIN update: (1 + eps) * h_i + aggregate
        out = (1 + self.eps) * x + neighbor_sum  # [B, N, D]

        # Apply MLP (needs reshaping for BatchNorm)
        mask_flat = node_mask.view(-1).bool()
        out_flat = out.view(-1, D)  # [B*N, D]

        # Only apply MLP to real atoms to avoid BatchNorm issues with padding
        if mask_flat.any():
            out_masked = out_flat[mask_flat]  # [num_real_atoms, D]
            out_masked = self.mlp(out_masked)  # [num_real_atoms, out_dim]
            result = torch.zeros(B * N, out_masked.size(-1), device=x.device)
            result[mask_flat] = out_masked
        else:
            # Fallback: all padding (shouldn't happen in practice)
            result = self.mlp(out_flat)

        out_dim = result.size(-1)
        result = result.view(B, N, out_dim)

        # Mask padding nodes
        result = result * node_mask.unsqueeze(-1)

        return result


class MolecularGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular graphs.

    Architecture:
    1. Atom embedding projection (linear layer)
    2. K layers of EdgeConditionedGINConv with residual connections
    3. Jumping Knowledge: concatenation of all layer outputs
    4. Global readout: masked mean + max pooling

    Produces a fixed-size molecular fingerprint vector from variable-size graphs.
    """

    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_jumping_knowledge: bool = True
    ):
        """
        Args:
            atom_feature_dim: Dimension of input atom features
            edge_feature_dim: Dimension of input edge features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GIN convolution layers
            dropout: Dropout rate between layers
            use_jumping_knowledge: Concatenate outputs from all layers
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_jumping_knowledge = use_jumping_knowledge

        # Initial atom feature projection
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.atom_bn = nn.BatchNorm1d(hidden_dim)

        # GIN convolution layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                EdgeConditionedGINConv(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=edge_feature_dim
                )
            )

        # Dropout between layers
        self.dropout = nn.Dropout(dropout)

        # Output dimension depends on jumping knowledge
        if use_jumping_knowledge:
            self.output_dim = hidden_dim * (num_layers + 1)  # All layers + initial
            self.jk_linear = nn.Linear(self.output_dim, hidden_dim)
            self.output_dim = hidden_dim  # After JK projection
        else:
            self.output_dim = hidden_dim

        # Readout produces 2x hidden_dim (mean + max concatenated)
        self.readout_dim = self.output_dim * 2

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a batch of molecular graphs into fixed-size vectors.

        Args:
            node_features: [batch, max_atoms, atom_feature_dim]
            adjacency: [batch, max_atoms, max_atoms]
            edge_features: [batch, max_atoms, max_atoms, edge_feature_dim]
            node_mask: [batch, max_atoms]

        Returns:
            Graph-level embeddings [batch, readout_dim]
        """
        B, N, _ = node_features.shape

        # Initial atom embedding
        h = node_features.view(-1, node_features.size(-1))  # [B*N, feat_dim]
        mask_flat = node_mask.view(-1).bool()

        if mask_flat.any():
            h_masked = h[mask_flat]
            h_masked = self.atom_embedding(h_masked)
            h_masked = self.atom_bn(h_masked)
            h_masked = F.relu(h_masked)
            h_out = torch.zeros(B * N, self.hidden_dim, device=h.device)
            h_out[mask_flat] = h_masked
        else:
            h_out = F.relu(self.atom_bn(self.atom_embedding(h)))

        h = h_out.view(B, N, self.hidden_dim)
        h = h * node_mask.unsqueeze(-1)

        # Collect layer outputs for jumping knowledge
        layer_outputs = [h]

        # Message passing layers
        for conv in self.conv_layers:
            h = conv(h, adjacency, edge_features, node_mask)
            h = self.dropout(h)
            layer_outputs.append(h)

        # Jumping Knowledge: concatenate all layer representations
        if self.use_jumping_knowledge:
            h_jk = torch.cat(layer_outputs, dim=-1)  # [B, N, hidden*(layers+1)]
            # Project back to hidden_dim
            h_flat = h_jk.view(-1, h_jk.size(-1))
            if mask_flat.any():
                h_proj = torch.zeros(B * N, self.output_dim, device=h.device)
                h_proj[mask_flat] = self.jk_linear(h_flat[mask_flat])
            else:
                h_proj = self.jk_linear(h_flat)
            h = h_proj.view(B, N, self.output_dim)
            h = h * node_mask.unsqueeze(-1)

        # Global readout: masked mean + max pooling
        graph_embedding = self._global_readout(h, node_mask)

        return graph_embedding

    def _global_readout(
        self,
        h: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Masked global pooling combining mean and max.

        Args:
            h: Node embeddings [batch, max_atoms, dim]
            node_mask: [batch, max_atoms]

        Returns:
            Graph embeddings [batch, dim * 2]
        """
        mask = node_mask.unsqueeze(-1)  # [B, N, 1]

        # Masked mean pooling
        h_sum = (h * mask).sum(dim=1)  # [B, dim]
        mask_count = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        h_mean = h_sum / mask_count  # [B, dim]

        # Masked max pooling (set padding to -inf before max)
        h_for_max = h.clone()
        h_for_max[mask.squeeze(-1) == 0] = float('-inf')
        h_max, _ = h_for_max.max(dim=1)  # [B, dim]
        # Handle all-padding case
        h_max = torch.where(
            torch.isinf(h_max),
            torch.zeros_like(h_max),
            h_max
        )

        # Concatenate mean and max
        return torch.cat([h_mean, h_max], dim=-1)  # [B, dim*2]


class DDIInteractionHead(nn.Module):
    """
    DDI Interaction Prediction Head (Enhanced v2)

    Takes drug pair embeddings and predicts interaction using
    concatenation + element-wise product + absolute difference.

    This captures:
    - Concatenation: individual drug properties
    - Element-wise product: interaction-specific signal
    - Absolute difference: structural dissimilarity

    Architecture:
    1. Feature combination layer
    2. Dense Layer with GELU
    3. Dropout + LayerNorm
    4. Hidden Layer with GELU (deeper head for more capacity)
    5. Output Layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 1,
        dropout_rate: float = 0.1,
        use_binary: bool = True
    ):
        """
        Args:
            input_dim: Input dimension (concatenated drug embeddings = readout_dim * 2)
            hidden_dim: Hidden layer dimension
            num_classes: 1 for binary, k for multi-class
            dropout_rate: Dropout probability
            use_binary: If True, use sigmoid; if False, use softmax
        """
        super().__init__()

        self.use_binary = use_binary
        # input_dim = readout_dim * 2 (from concatenation)
        # Each drug embedding is readout_dim = input_dim // 2
        single_dim = input_dim // 2

        # Combined features: concat + product + abs_diff = 3 * single_dim
        combined_dim = single_dim * 3

        self.dense1 = nn.Linear(combined_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-12)

        # Deeper head for more capacity
        self.dense2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(hidden_dim // 2, eps=1e-12)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Concatenated drug pair embedding [batch, input_dim]

        Returns:
            Logits [batch, num_classes]
        """
        # Split into individual drug embeddings
        half = x.size(-1) // 2
        drug1_emb = x[:, :half]
        drug2_emb = x[:, half:]

        # Multi-perspective combination
        product = drug1_emb * drug2_emb          # interaction signal
        abs_diff = torch.abs(drug1_emb - drug2_emb)  # dissimilarity
        combined = torch.cat([x, product, abs_diff], dim=-1)  # [B, 3 * half]
        # Note: x already contains [drug1; drug2] concatenation but combined_dim
        # expects 3 * single_dim. x has 2*single_dim, so we use [product, abs_diff, sum]:
        combined = torch.cat([product, abs_diff, drug1_emb + drug2_emb], dim=-1)

        x = self.dense1(combined)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.ln1(x)

        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.ln2(x)

        logits = self.classifier(x)
        return logits


class DDIGraphModel(nn.Module):
    """
    Complete GNN-based Drug-Drug Interaction Prediction Model

    Architecture:
    1. Shared MolecularGNNEncoder processes each drug's molecular graph
    2. Drug embeddings are concatenated: [drug1_emb; drug2_emb]
    3. DDIInteractionHead predicts interaction type/severity

    This model is structure-aware and can predict interactions for novel
    drugs unseen during training, as long as their SMILES are available.

    Reference: MCR III - GNN Model (structure-based prediction)
    """

    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_relation_classes: int = 1,
        dropout_rate: float = 0.1,
        use_binary: bool = True,
        use_jumping_knowledge: bool = True
    ):
        """
        Args:
            atom_feature_dim: Dimension of atom features from featurizer
            edge_feature_dim: Dimension of edge features from featurizer
            hidden_dim: Hidden dimension for GNN and prediction head
            num_gnn_layers: Number of GIN message-passing layers
            num_relation_classes: 1 for binary, k for multi-class
            dropout_rate: Dropout rate for prediction head
            use_binary: Binary or multi-class classification
            use_jumping_knowledge: Use JK connections in GNN encoder
        """
        super().__init__()

        self.use_binary = use_binary
        self.hidden_dim = hidden_dim

        # Shared molecular encoder for both drugs
        self.encoder = MolecularGNNEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout_rate,
            use_jumping_knowledge=use_jumping_knowledge
        )

        # DDI prediction head
        # Input: readout_dim * 2 (passed as input_dim, then internally split)
        self.interaction_head = DDIInteractionHead(
            input_dim=self.encoder.readout_dim * 2,
            hidden_dim=hidden_dim,
            num_classes=num_relation_classes,
            dropout_rate=dropout_rate,
            use_binary=use_binary
        )

    def forward(
        self,
        drug1_node_features: torch.Tensor,
        drug1_adjacency: torch.Tensor,
        drug1_edge_features: torch.Tensor,
        drug1_node_mask: torch.Tensor,
        drug2_node_features: torch.Tensor,
        drug2_adjacency: torch.Tensor,
        drug2_edge_features: torch.Tensor,
        drug2_node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for DDI prediction from molecular graphs.

        Args:
            drug1_node_features: [batch, max_atoms, atom_feat_dim]
            drug1_adjacency: [batch, max_atoms, max_atoms]
            drug1_edge_features: [batch, max_atoms, max_atoms, edge_feat_dim]
            drug1_node_mask: [batch, max_atoms]
            drug2_node_features: [batch, max_atoms, atom_feat_dim]
            drug2_adjacency: [batch, max_atoms, max_atoms]
            drug2_edge_features: [batch, max_atoms, max_atoms, edge_feat_dim]
            drug2_node_mask: [batch, max_atoms]

        Returns:
            relation_logits: [batch, num_relation_classes]
        """
        # Encode both molecular graphs using shared encoder
        drug1_emb = self.encoder(
            drug1_node_features, drug1_adjacency,
            drug1_edge_features, drug1_node_mask
        )  # [B, readout_dim]

        drug2_emb = self.encoder(
            drug2_node_features, drug2_adjacency,
            drug2_edge_features, drug2_node_mask
        )  # [B, readout_dim]

        # Concatenate drug embeddings
        combined = torch.cat([drug1_emb, drug2_emb], dim=-1)  # [B, readout_dim*2]

        # Predict interaction
        logits = self.interaction_head(combined)

        return logits

    def get_drug_embedding(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the learned embedding for a single drug molecule.

        Useful for downstream tasks like drug similarity or clustering.

        Args:
            node_features: [batch, max_atoms, atom_feat_dim]
            adjacency: [batch, max_atoms, max_atoms]
            edge_features: [batch, max_atoms, max_atoms, edge_feat_dim]
            node_mask: [batch, max_atoms]

        Returns:
            Drug embedding [batch, readout_dim]
        """
        return self.encoder(node_features, adjacency, edge_features, node_mask)

    def get_relation_probabilities(
        self,
        drug1_node_features: torch.Tensor,
        drug1_adjacency: torch.Tensor,
        drug1_edge_features: torch.Tensor,
        drug1_node_mask: torch.Tensor,
        drug2_node_features: torch.Tensor,
        drug2_adjacency: torch.Tensor,
        drug2_edge_features: torch.Tensor,
        drug2_node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probabilities for DDI prediction.

        Args:
            drug1_*, drug2_*: Molecular graph tensors for both drugs

        Returns:
            Probabilities [batch, num_classes]
        """
        logits = self.forward(
            drug1_node_features, drug1_adjacency,
            drug1_edge_features, drug1_node_mask,
            drug2_node_features, drug2_adjacency,
            drug2_edge_features, drug2_node_mask
        )

        if self.use_binary:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)
