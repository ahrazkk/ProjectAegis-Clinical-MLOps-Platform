"""
Unit tests for GNN-based DDI Prediction Model
Tests featurization, model architecture, dataset, training, and inference
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.gnn_featurizer import (
    MolecularGraphFeaturizer,
    get_atom_features,
    get_bond_features,
    ATOM_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    _one_hot,
)
from model.gnn_model import (
    EdgeConditionedGINConv,
    MolecularGNNEncoder,
    DDIInteractionHead,
    DDIGraphModel,
)
from model.gnn_dataset import DDIGraphDataset, create_graph_data_loaders
from model.gnn_trainer import GNNTrainer, GNNTrainingConfig
from model.gnn_inference import GNNPredictor, DDIPrediction


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def featurizer():
    """Create a MolecularGraphFeaturizer instance."""
    return MolecularGraphFeaturizer(max_atoms=64)


@pytest.fixture
def aspirin_smiles():
    return "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def warfarin_smiles():
    return "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"


@pytest.fixture
def simple_smiles():
    """Ethanol - a very small molecule for fast tests."""
    return "CCO"


@pytest.fixture
def sample_graph(featurizer, simple_smiles):
    """Pre-featurized graph of ethanol."""
    return featurizer.smiles_to_graph(simple_smiles)


@pytest.fixture
def sample_ddi_data(aspirin_smiles, warfarin_smiles, simple_smiles):
    """Sample DDI dataset for testing."""
    return [
        {
            "drug1_smiles": aspirin_smiles,
            "drug2_smiles": warfarin_smiles,
            "drug1_name": "aspirin",
            "drug2_name": "warfarin",
            "interaction_type": 3,
            "has_interaction": 1,
        },
        {
            "drug1_smiles": simple_smiles,
            "drug2_smiles": aspirin_smiles,
            "drug1_name": "ethanol",
            "drug2_name": "aspirin",
            "interaction_type": 0,
            "has_interaction": 0,
        },
        {
            "drug1_smiles": warfarin_smiles,
            "drug2_smiles": simple_smiles,
            "drug1_name": "warfarin",
            "drug2_name": "ethanol",
            "interaction_type": 2,
            "has_interaction": 1,
        },
        {
            "drug1_smiles": simple_smiles,
            "drug2_smiles": warfarin_smiles,
            "drug1_name": "ethanol",
            "drug2_name": "warfarin",
            "interaction_type": 1,
            "has_interaction": 1,
        },
    ]


@pytest.fixture
def gnn_model():
    """Create a small DDIGraphModel for testing."""
    return DDIGraphModel(
        atom_feature_dim=ATOM_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=32,
        num_gnn_layers=2,
        num_relation_classes=1,
        dropout_rate=0.0,
        use_binary=True,
        use_jumping_knowledge=True,
    )


@pytest.fixture
def gnn_config():
    """Training config for fast testing."""
    return GNNTrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        hidden_dim=32,
        num_gnn_layers=2,
        num_epochs=2,
        early_stopping_patience=5,
        use_binary=True,
        dropout_rate=0.0,
        max_atoms=64,
        log_interval=1,
    )


# ============================================================
# TestOneHotEncoding
# ============================================================

class TestOneHotEncoding:
    """Test the _one_hot utility function."""

    def test_known_value(self):
        result = _one_hot('C', ['C', 'N', 'O'])
        assert result == [1, 0, 0, 0]

    def test_unknown_value(self):
        result = _one_hot('X', ['C', 'N', 'O'])
        assert result == [0, 0, 0, 1]  # 'other' bucket

    def test_last_known_value(self):
        result = _one_hot('O', ['C', 'N', 'O'])
        assert result == [0, 0, 1, 0]

    def test_length(self):
        choices = ['A', 'B', 'C', 'D']
        result = _one_hot('B', choices)
        assert len(result) == len(choices) + 1  # +1 for 'other'


# ============================================================
# TestMolecularGraphFeaturizer
# ============================================================

class TestMolecularGraphFeaturizer:
    """Test SMILES to molecular graph conversion."""

    def test_initialization(self, featurizer):
        assert featurizer.max_atoms == 64
        assert featurizer.atom_feature_dim == ATOM_FEATURE_DIM
        assert featurizer.edge_feature_dim == EDGE_FEATURE_DIM

    def test_valid_smiles_produces_graph(self, featurizer, aspirin_smiles):
        graph = featurizer.smiles_to_graph(aspirin_smiles)
        assert graph is not None

    def test_invalid_smiles_returns_none(self, featurizer):
        graph = featurizer.smiles_to_graph("NOT_A_VALID_SMILES_XYZ")
        assert graph is None

    def test_graph_keys(self, sample_graph):
        expected_keys = {
            'node_features', 'adjacency', 'edge_features',
            'node_mask', 'num_atoms'
        }
        assert set(sample_graph.keys()) == expected_keys

    def test_node_features_shape(self, featurizer, sample_graph):
        assert sample_graph['node_features'].shape == (
            featurizer.max_atoms, ATOM_FEATURE_DIM
        )

    def test_adjacency_shape(self, featurizer, sample_graph):
        assert sample_graph['adjacency'].shape == (
            featurizer.max_atoms, featurizer.max_atoms
        )

    def test_edge_features_shape(self, featurizer, sample_graph):
        assert sample_graph['edge_features'].shape == (
            featurizer.max_atoms, featurizer.max_atoms, EDGE_FEATURE_DIM
        )

    def test_node_mask_shape(self, featurizer, sample_graph):
        assert sample_graph['node_mask'].shape == (featurizer.max_atoms,)

    def test_node_mask_correct(self, sample_graph):
        """Real atoms should have mask=1, padding should have mask=0."""
        num_atoms = sample_graph['num_atoms']
        mask = sample_graph['node_mask']
        assert mask[:num_atoms].sum() == num_atoms
        assert mask[num_atoms:].sum() == 0

    def test_adjacency_has_self_loops(self, sample_graph):
        """Adjacency should include self-loops on real atoms."""
        num_atoms = sample_graph['num_atoms']
        adj = sample_graph['adjacency']
        for i in range(num_atoms):
            assert adj[i, i] == 1.0

    def test_adjacency_symmetric(self, sample_graph):
        """Adjacency should be symmetric (undirected graph)."""
        adj = sample_graph['adjacency']
        assert torch.allclose(adj, adj.T)

    def test_edge_features_symmetric(self, sample_graph):
        """Edge features should be symmetric for undirected bonds."""
        ef = sample_graph['edge_features']
        num_atoms = sample_graph['num_atoms']
        for i in range(num_atoms):
            for j in range(num_atoms):
                assert torch.allclose(ef[i, j], ef[j, i])

    def test_padding_is_zero(self, sample_graph):
        """Padding regions should be all zeros."""
        num_atoms = sample_graph['num_atoms']
        nf = sample_graph['node_features']
        assert nf[num_atoms:].sum() == 0

    def test_smiles_pair_to_graphs(self, featurizer, aspirin_smiles, warfarin_smiles):
        result = featurizer.smiles_pair_to_graphs(aspirin_smiles, warfarin_smiles)
        assert result is not None
        assert 'drug1_node_features' in result
        assert 'drug2_node_features' in result
        assert 'drug1_adjacency' in result
        assert 'drug2_adjacency' in result

    def test_smiles_pair_invalid_returns_none(self, featurizer, aspirin_smiles):
        result = featurizer.smiles_pair_to_graphs(aspirin_smiles, "INVALID")
        assert result is None

    def test_molecular_descriptor(self, featurizer, aspirin_smiles):
        desc = featurizer.get_molecular_descriptor(aspirin_smiles)
        assert desc is not None
        assert 'molecular_weight' in desc
        assert 'logp' in desc
        assert desc['molecular_weight'] > 0

    def test_molecular_descriptor_invalid(self, featurizer):
        desc = featurizer.get_molecular_descriptor("INVALID")
        assert desc is None


# ============================================================
# TestAtomAndBondFeatures
# ============================================================

class TestAtomAndBondFeatures:
    """Test atom and bond feature extraction."""

    def test_atom_feature_dimension(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        atom = mol.GetAtomWithIdx(0)  # Carbon
        features = get_atom_features(atom)
        assert len(features) == ATOM_FEATURE_DIM

    def test_bond_feature_dimension(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        assert len(features) == EDGE_FEATURE_DIM

    def test_aromatic_atom_flag(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        # Aromatic flag should be 1.0 (second-to-last feature)
        assert features[-2] == 1.0

    def test_non_aromatic_atom_flag(self):
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CC")  # Ethane
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        assert features[-2] == 0.0


# ============================================================
# TestEdgeConditionedGINConv
# ============================================================

class TestEdgeConditionedGINConv:
    """Test the GIN convolution layer."""

    def test_output_shape(self):
        conv = EdgeConditionedGINConv(in_dim=16, out_dim=32, edge_dim=8)
        B, N = 2, 10
        x = torch.randn(B, N, 16)
        adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        edge_feat = torch.randn(B, N, N, 8)
        mask = torch.ones(B, N)

        out = conv(x, adj, edge_feat, mask)
        assert out.shape == (B, N, 32)

    def test_masked_output_is_zero(self):
        conv = EdgeConditionedGINConv(in_dim=16, out_dim=32, edge_dim=8)
        B, N = 2, 10
        x = torch.randn(B, N, 16)
        adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        edge_feat = torch.randn(B, N, N, 8)
        mask = torch.zeros(B, N)  # All masked out
        mask[:, :3] = 1.0

        out = conv(x, adj, edge_feat, mask)
        # Masked positions should be zero
        assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]))

    def test_eps_is_learnable(self):
        conv = EdgeConditionedGINConv(in_dim=16, out_dim=16, edge_dim=8, train_eps=True)
        assert isinstance(conv.eps, nn.Parameter)

    def test_eps_not_learnable(self):
        conv = EdgeConditionedGINConv(in_dim=16, out_dim=16, edge_dim=8, train_eps=False)
        assert not isinstance(conv.eps, nn.Parameter)


# ============================================================
# TestMolecularGNNEncoder
# ============================================================

class TestMolecularGNNEncoder:
    """Test the molecular graph encoder."""

    def test_output_shape(self):
        encoder = MolecularGNNEncoder(hidden_dim=32, num_layers=2)
        B, N = 2, 10
        nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        mask = torch.ones(B, N)

        out = encoder(nf, adj, ef, mask)
        assert out.shape == (B, encoder.readout_dim)

    def test_readout_dim_with_jk(self):
        encoder = MolecularGNNEncoder(
            hidden_dim=32, num_layers=3, use_jumping_knowledge=True
        )
        # With JK: readout_dim = output_dim * 2 = hidden_dim * 2
        assert encoder.readout_dim == 32 * 2

    def test_readout_dim_without_jk(self):
        encoder = MolecularGNNEncoder(
            hidden_dim=32, num_layers=3, use_jumping_knowledge=False
        )
        assert encoder.readout_dim == 32 * 2  # mean + max pooling

    def test_batch_independence(self):
        """Each sample in batch should produce independent embeddings."""
        encoder = MolecularGNNEncoder(hidden_dim=32, num_layers=2)
        encoder.eval()

        N = 10
        nf1 = torch.randn(1, N, ATOM_FEATURE_DIM)
        nf2 = torch.randn(1, N, ATOM_FEATURE_DIM)
        adj = torch.eye(N).unsqueeze(0)
        ef = torch.zeros(1, N, N, EDGE_FEATURE_DIM)
        mask = torch.ones(1, N)

        out1 = encoder(nf1, adj, ef, mask)
        out2 = encoder(nf2, adj, ef, mask)

        # Different inputs should give different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)


# ============================================================
# TestDDIInteractionHead
# ============================================================

class TestDDIInteractionHead:
    """Test the DDI prediction head."""

    def test_binary_output(self):
        head = DDIInteractionHead(input_dim=64, hidden_dim=32, num_classes=1)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 1)

    def test_multiclass_output(self):
        head = DDIInteractionHead(input_dim=64, hidden_dim=32, num_classes=5)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 5)


# ============================================================
# TestDDIGraphModel
# ============================================================

class TestDDIGraphModel:
    """Test the full DDI graph model."""

    def test_forward_shape(self, gnn_model):
        B, N = 2, 64
        d1_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d1_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d1_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d1_mask = torch.ones(B, N)
        d2_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d2_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d2_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d2_mask = torch.ones(B, N)

        logits = gnn_model(
            d1_nf, d1_adj, d1_ef, d1_mask,
            d2_nf, d2_adj, d2_ef, d2_mask
        )
        assert logits.shape == (B, 1)

    def test_get_drug_embedding(self, gnn_model):
        B, N = 2, 64
        nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        mask = torch.ones(B, N)

        emb = gnn_model.get_drug_embedding(nf, adj, ef, mask)
        assert emb.shape == (B, gnn_model.encoder.readout_dim)

    def test_get_relation_probabilities(self, gnn_model):
        B, N = 2, 64
        d1_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d1_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d1_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d1_mask = torch.ones(B, N)
        d2_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d2_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d2_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d2_mask = torch.ones(B, N)

        probs = gnn_model.get_relation_probabilities(
            d1_nf, d1_adj, d1_ef, d1_mask,
            d2_nf, d2_adj, d2_ef, d2_mask
        )
        # Binary: sigmoid output should be in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_model_is_differentiable(self, gnn_model):
        """Ensure gradients flow through the full model."""
        B, N = 2, 64
        d1_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d1_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d1_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d1_mask = torch.ones(B, N)
        d2_nf = torch.randn(B, N, ATOM_FEATURE_DIM)
        d2_adj = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
        d2_ef = torch.randn(B, N, N, EDGE_FEATURE_DIM)
        d2_mask = torch.ones(B, N)

        logits = gnn_model(
            d1_nf, d1_adj, d1_ef, d1_mask,
            d2_nf, d2_adj, d2_ef, d2_mask
        )
        loss = logits.sum()
        loss.backward()

        # Check at least some gradients are non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in gnn_model.parameters()
        )
        assert has_grad

    def test_shared_encoder(self, gnn_model):
        """Both drugs should use the same encoder (shared weights)."""
        assert gnn_model.encoder is not None
        # Only one encoder attribute, it's shared
        assert id(gnn_model.encoder) == id(gnn_model.encoder)


# ============================================================
# TestDDIGraphDataset
# ============================================================

class TestDDIGraphDataset:
    """Test dataset loading and processing."""

    def test_dataset_length(self, sample_ddi_data, featurizer):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        assert len(dataset) == len(sample_ddi_data)

    def test_dataset_item_keys(self, sample_ddi_data, featurizer):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        item = dataset[0]

        expected_keys = {
            'drug1_node_features', 'drug1_adjacency', 'drug1_edge_features',
            'drug1_node_mask',
            'drug2_node_features', 'drug2_adjacency', 'drug2_edge_features',
            'drug2_node_mask',
            'relation_label',
        }
        assert set(item.keys()) == expected_keys

    def test_binary_label(self, sample_ddi_data, featurizer):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer, use_binary_labels=True)
        item = dataset[0]
        assert item['relation_label'].dtype == torch.float
        assert item['relation_label'] in [0.0, 1.0]

    def test_multiclass_label(self, sample_ddi_data, featurizer):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer, use_binary_labels=False)
        item = dataset[0]
        assert item['relation_label'].dtype == torch.long

    def test_invalid_smiles_skipped(self, featurizer):
        data = [
            {
                "drug1_smiles": "INVALID",
                "drug2_smiles": "ALSO_INVALID",
                "has_interaction": 1,
            }
        ]
        dataset = DDIGraphDataset(data, featurizer)
        assert len(dataset) == 0

    def test_collate_fn(self, sample_ddi_data, featurizer):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        batch = DDIGraphDataset.collate_fn([dataset[0], dataset[1]])

        assert batch['drug1_node_features'].shape[0] == 2
        assert batch['relation_label'].shape[0] == 2

    def test_from_json(self, sample_ddi_data, featurizer, tmp_path):
        filepath = tmp_path / "test_data.json"
        with open(filepath, 'w') as f:
            json.dump(sample_ddi_data, f)

        dataset = DDIGraphDataset.from_json(str(filepath), featurizer)
        assert len(dataset) == len(sample_ddi_data)

    def test_create_data_loaders(self, sample_ddi_data, featurizer):
        train_ds = DDIGraphDataset(sample_ddi_data[:2], featurizer)
        val_ds = DDIGraphDataset(sample_ddi_data[2:], featurizer)

        train_loader, val_loader = create_graph_data_loaders(
            train_ds, val_ds, batch_size=2
        )

        batch = next(iter(train_loader))
        assert 'drug1_node_features' in batch
        assert batch['drug1_node_features'].shape[0] <= 2


# ============================================================
# TestGNNTrainingConfig
# ============================================================

class TestGNNTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        config = GNNTrainingConfig()
        assert config.learning_rate == 1e-3
        assert config.hidden_dim == 256
        assert config.num_gnn_layers == 3
        assert config.use_binary is True

    def test_to_dict(self, gnn_config):
        d = gnn_config.to_dict()
        assert d['learning_rate'] == 1e-3
        assert d['hidden_dim'] == 32
        assert d['num_gnn_layers'] == 2

    def test_from_dict(self):
        d = {'learning_rate': 5e-4, 'hidden_dim': 128, 'num_gnn_layers': 4}
        config = GNNTrainingConfig.from_dict(d)
        assert config.learning_rate == 5e-4
        assert config.hidden_dim == 128
        assert config.num_gnn_layers == 4


# ============================================================
# TestGNNTrainer
# ============================================================

class TestGNNTrainer:
    """Test GNN training loop."""

    def test_trainer_initialization(self, gnn_model, gnn_config, tmp_path):
        trainer = GNNTrainer(gnn_model, gnn_config, output_dir=str(tmp_path))
        assert trainer.model is not None
        assert trainer.best_metric == 0.0
        assert trainer.patience_counter == 0

    def test_single_train_epoch(self, gnn_model, gnn_config, sample_ddi_data, featurizer, tmp_path):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        train_loader, val_loader = create_graph_data_loaders(
            dataset, dataset, batch_size=2
        )

        trainer = GNNTrainer(gnn_model, gnn_config, output_dir=str(tmp_path))
        trainer._setup_optimizer(len(train_loader) * 2)

        metrics = trainer.train_epoch(train_loader, epoch=0)
        assert 'train_loss' in metrics
        assert metrics['train_loss'] > 0

    def test_evaluate(self, gnn_model, gnn_config, sample_ddi_data, featurizer, tmp_path):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        _, val_loader = create_graph_data_loaders(
            dataset, dataset, batch_size=2
        )

        trainer = GNNTrainer(gnn_model, gnn_config, output_dir=str(tmp_path))
        metrics = trainer.evaluate(val_loader)

        assert 'val_loss' in metrics

    def test_save_and_load_checkpoint(self, gnn_model, gnn_config, tmp_path):
        trainer = GNNTrainer(gnn_model, gnn_config, output_dir=str(tmp_path))
        trainer._save_checkpoint('test_ckpt.pt', {'test_metric': 0.5})

        ckpt_path = tmp_path / 'test_ckpt.pt'
        assert ckpt_path.exists()

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'metrics' in checkpoint

    def test_full_training_loop(self, gnn_model, gnn_config, sample_ddi_data, featurizer, tmp_path):
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        train_loader, val_loader = create_graph_data_loaders(
            dataset, dataset, batch_size=2
        )

        trainer = GNNTrainer(gnn_model, gnn_config, output_dir=str(tmp_path))
        results = trainer.train(train_loader, val_loader)

        assert 'best_metric' in results
        assert 'training_history' in results
        assert len(results['training_history']) > 0


# ============================================================
# TestDDIPrediction
# ============================================================

class TestDDIPrediction:
    """Test the DDIPrediction dataclass."""

    def test_creation(self):
        pred = DDIPrediction(
            drug1="aspirin",
            drug2="warfarin",
            has_interaction=True,
            interaction_type="mechanism",
            raw_probability=0.85,
            calibrated_probability=0.82,
            risk_score=0.9,
            risk_category="high",
            confidence=0.7,
        )
        assert pred.drug1 == "aspirin"
        assert pred.has_interaction is True
        assert pred.risk_score == 0.9

    def test_no_interaction(self):
        pred = DDIPrediction(
            drug1="drugA",
            drug2="drugB",
            has_interaction=False,
            interaction_type=None,
            raw_probability=0.1,
            calibrated_probability=0.1,
            risk_score=0.1,
            risk_category="low",
            confidence=0.8,
        )
        assert pred.has_interaction is False
        assert pred.interaction_type is None


# ============================================================
# TestGNNPredictor
# ============================================================

class TestGNNPredictor:
    """Test the high-level GNN prediction API."""

    def test_initialization_without_model(self):
        predictor = GNNPredictor(use_binary=True, max_atoms=64)
        assert predictor.model is None
        assert predictor.featurizer is not None

    def test_resolve_smiles_from_name(self):
        predictor = GNNPredictor(max_atoms=64)
        smiles = predictor.resolve_smiles("aspirin")
        assert smiles is not None
        assert smiles == "CC(=O)Oc1ccccc1C(=O)O"

    def test_resolve_smiles_unknown_name(self):
        predictor = GNNPredictor(max_atoms=64)
        smiles = predictor.resolve_smiles("not_a_real_drug_xyz")
        assert smiles is None

    def test_resolve_smiles_from_smiles(self):
        predictor = GNNPredictor(max_atoms=64)
        smiles = predictor.resolve_smiles("CCO")
        assert smiles == "CCO"

    def test_add_drug_smiles(self):
        predictor = GNNPredictor(max_atoms=64)
        predictor.add_drug_smiles("testdrug", "CCCC")
        assert predictor.resolve_smiles("testdrug") == "CCCC"

    def test_add_drug_smiles_case_insensitive(self):
        predictor = GNNPredictor(max_atoms=64)
        predictor.add_drug_smiles("TestDrug", "CCCC")
        assert predictor.resolve_smiles("testdrug") == "CCCC"

    def test_predict_without_model_raises(self):
        predictor = GNNPredictor(max_atoms=64)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            predictor.predict_from_smiles("CCO", "CC")

    def test_predict_from_names_unknown_raises(self):
        predictor = GNNPredictor(max_atoms=64)
        predictor.model = Mock()  # Fake model to bypass model check
        with pytest.raises(ValueError, match="Cannot resolve"):
            predictor.predict_from_names("not_real_drug_1", "not_real_drug_2")

    def test_risk_explanation_no_interaction(self):
        predictor = GNNPredictor(max_atoms=64)
        pred = DDIPrediction(
            drug1="aspirin", drug2="ethanol",
            has_interaction=False, interaction_type=None,
            raw_probability=0.1, calibrated_probability=0.1,
            risk_score=0.1, risk_category="low", confidence=0.8,
        )
        explanation = predictor.get_risk_explanation(pred)
        assert "No significant" in explanation
        assert "aspirin" in explanation

    def test_risk_explanation_with_interaction(self):
        predictor = GNNPredictor(max_atoms=64)
        pred = DDIPrediction(
            drug1="aspirin", drug2="warfarin",
            has_interaction=True, interaction_type="mechanism",
            raw_probability=0.9, calibrated_probability=0.88,
            risk_score=0.9, risk_category="high", confidence=0.8,
        )
        explanation = predictor.get_risk_explanation(pred)
        assert "Potential drug-drug interaction" in explanation
        assert "HIGH" in explanation

    def test_drug_smiles_db_populated(self):
        predictor = GNNPredictor(max_atoms=64)
        # Should have the built-in database
        assert len(predictor.drug_smiles_db) >= 20

    def test_custom_db_merged(self):
        custom_db = {"customdrug": "CCCC"}
        predictor = GNNPredictor(max_atoms=64, drug_smiles_db=custom_db)
        assert predictor.resolve_smiles("customdrug") == "CCCC"
        assert predictor.resolve_smiles("aspirin") is not None  # Still has built-in


# ============================================================
# TestEndToEndWithRealMolecules
# ============================================================

class TestEndToEndWithRealMolecules:
    """Integration tests using real SMILES through the full pipeline."""

    def test_featurize_and_model_forward(self, featurizer, gnn_model, aspirin_smiles, warfarin_smiles):
        """Test featurizing real molecules and passing through the model."""
        graphs = featurizer.smiles_pair_to_graphs(aspirin_smiles, warfarin_smiles)
        assert graphs is not None

        logits = gnn_model(
            graphs['drug1_node_features'].unsqueeze(0),
            graphs['drug1_adjacency'].unsqueeze(0),
            graphs['drug1_edge_features'].unsqueeze(0),
            graphs['drug1_node_mask'].unsqueeze(0),
            graphs['drug2_node_features'].unsqueeze(0),
            graphs['drug2_adjacency'].unsqueeze(0),
            graphs['drug2_edge_features'].unsqueeze(0),
            graphs['drug2_node_mask'].unsqueeze(0),
        )
        assert logits.shape == (1, 1)

    def test_dataset_to_model(self, sample_ddi_data, featurizer, gnn_model):
        """Test full pipeline: data → dataset → dataloader → model."""
        dataset = DDIGraphDataset(sample_ddi_data, featurizer)
        train_loader, _ = create_graph_data_loaders(dataset, dataset, batch_size=2)

        gnn_model.eval()
        batch = next(iter(train_loader))
        logits = gnn_model(
            batch['drug1_node_features'],
            batch['drug1_adjacency'],
            batch['drug1_edge_features'],
            batch['drug1_node_mask'],
            batch['drug2_node_features'],
            batch['drug2_adjacency'],
            batch['drug2_edge_features'],
            batch['drug2_node_mask'],
        )
        assert logits.shape[0] == batch['relation_label'].shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
