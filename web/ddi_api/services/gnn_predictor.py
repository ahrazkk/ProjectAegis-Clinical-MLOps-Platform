"""
ChemicalX GNN-Based DDI Prediction Service

Provides Graph Neural Network-based drug-drug interaction predictions
using molecular graph representations. This is an alternative/complement
to the PubMedBERT NLP-based approach.

Reference:
- ChemicalX: A Deep Learning Library for Drug Pair Scoring (https://arxiv.org/abs/2202.05240)
- DeepDDI: Predicting Drug-Drug Interactions via Deep Learning
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from difflib import get_close_matches

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GNN predictions will use fallback mode.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Molecular fingerprints will use simplified features.")


class ModelType(str, Enum):
    """Available GNN model architectures."""
    MACROSCOPIC_GNN = "macroscopic_gnn" # V2 Model: Dense GraphSAGE
    TRAINED_GNN = "trained_gnn"  # V1 Model: Trained Edge-Conditioned GIN
    DEEPDDI = "deepddi"
    MHCADDI = "mhcaddi"
    SSIDDI = "ssiddi"
    CASTER = "caster"
    SIMPLE_MLP = "simple_mlp"


@dataclass
class GNNPrediction:
    """Result of a GNN-based DDI prediction."""
    drug1: str
    drug2: str
    interaction_probability: float
    interaction_type: str
    confidence: float
    severity: str
    model_used: str
    smiles1: Optional[str]
    smiles2: Optional[str]
    fingerprint_similarity: Optional[float]
    mechanism_hypothesis: str
    raw_interaction_probability: Optional[float] = None
    calibration_method: str = "none"
    calibration_version: str = "none"
    fallback_reason: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


class MolecularFeatureExtractor:
    """
    Extracts molecular features from drug structures for GNN input.
    
    Supports:
    - Morgan Fingerprints (ECFP)
    - MACCS Keys
    - RDKit Descriptors
    - Simplified features when RDKit unavailable
    """
    
    # Default fingerprint parameters
    FP_RADIUS = 2
    FP_BITS = 2048
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.use_rdkit = RDKIT_AVAILABLE
    
    def smiles_to_fingerprint(self, smiles: str) -> Optional[List[int]]:
        """
        Convert SMILES to molecular fingerprint.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            Binary fingerprint as list of ints, or None if conversion fails
        """
        if not smiles:
            return None
        
        if self.use_rdkit:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    return None
                
                # Generate Morgan fingerprint (ECFP4)
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 
                    radius=self.FP_RADIUS, 
                    nBits=self.FP_BITS
                )
                return list(fp)
                
            except Exception as e:
                logger.warning(f"Error generating fingerprint: {e}")
                return None
        else:
            # Fallback: Hash-based pseudo-fingerprint
            return self._hash_fingerprint(smiles)
    
    def _hash_fingerprint(self, smiles: str) -> List[int]:
        """
        Generate a pseudo-fingerprint using hashing when RDKit unavailable.
        
        This is NOT chemically meaningful but provides consistent features.
        """
        # Create deterministic hash-based features
        h = hashlib.sha256(smiles.encode()).digest()
        
        # Convert to binary fingerprint
        fp = []
        for byte in h:
            for i in range(8):
                fp.append((byte >> i) & 1)
        
        # Pad to standard length
        while len(fp) < self.FP_BITS:
            fp.extend(fp[:min(len(h) * 8, self.FP_BITS - len(fp))])
        
        return fp[:self.FP_BITS]
    
    def get_molecular_descriptors(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Calculate molecular descriptors for a compound.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of descriptor name -> value
        """
        if not self.use_rdkit or not smiles:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            }
        except Exception as e:
            logger.warning(f"Error calculating descriptors: {e}")
            return None
    
    def calculate_tanimoto_similarity(
        self, 
        smiles1: str, 
        smiles2: str
    ) -> Optional[float]:
        """
        Calculate Tanimoto similarity between two molecules.
        
        Args:
            smiles1: First molecule SMILES
            smiles2: Second molecule SMILES
            
        Returns:
            Tanimoto coefficient (0-1) or None
        """
        fp1 = self.smiles_to_fingerprint(smiles1)
        fp2 = self.smiles_to_fingerprint(smiles2)
        
        if fp1 is None or fp2 is None:
            return None
        
        # Calculate Tanimoto coefficient
        intersection = sum(a & b for a, b in zip(fp1, fp2))
        union = sum(a | b for a, b in zip(fp1, fp2))
        
        if union == 0:
            return 0.0
        
        return intersection / union


# Only define PyTorch model class when torch is available
if TORCH_AVAILABLE:
    class SimpleDDIPredictor(nn.Module):
        """
        Simple MLP-based DDI predictor using molecular fingerprints.
        
        Architecture:
        - Input: Concatenated fingerprints of drug pair
        - Hidden layers with ReLU and dropout
        - Output: Interaction probability
        
        This is a lightweight alternative when full GNN models are unavailable.
        """
        
        def __init__(
            self,
            input_dim: int = 4096,  # 2 x 2048 fingerprint bits
            hidden_dims: List[int] = [1024, 512, 256],
            num_classes: int = 5,
            dropout: float = 0.3
        ):
            """
            Initialize the DDI predictor.
            
            Args:
                input_dim: Size of concatenated fingerprint input
                hidden_dims: List of hidden layer sizes
                num_classes: Number of interaction types to predict
                dropout: Dropout rate
            """
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            self.feature_extractor = nn.Sequential(*layers)
            self.classifier = nn.Linear(prev_dim, num_classes)
        
        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """Forward pass."""
            features = self.feature_extractor(x)
            logits = self.classifier(features)
            return logits
else:
    # Placeholder when PyTorch not available
    SimpleDDIPredictor = None


class GNNDDIPredictor:
    """
    GNN-based Drug-Drug Interaction Predictor.
    
    Supports multiple model architectures and provides predictions
    based on molecular structure rather than text.
    
    This complements the PubMedBERT approach by:
    1. Working with any drug with a known structure (SMILES)
    2. Not requiring text context
    3. Capturing structural similarities between drugs
    """
    
    # Interaction type labels (aligned with PubMedBERT)
    INTERACTION_TYPES = [
        'no_interaction',
        'mechanism',
        'effect',
        'advise',
        'int'
    ]
    
    # Severity mapping
    SEVERITY_MAP = {
        'no_interaction': ('none', 0.0),
        'int': ('moderate', 0.4),
        'advise': ('moderate', 0.6),
        'effect': ('major', 0.75),
        'mechanism': ('severe', 0.85),
    }
    
    def __init__(self, model_type: ModelType = ModelType.MACROSCOPIC_GNN):
        """
        Initialize the GNN DDI predictor.
        
        Args:
            model_type: Type of model architecture to use
        """
        self.model_type = model_type
        self.feature_extractor = MolecularFeatureExtractor()
        self.model = None
        self.graph_featurizer = None
        
        # Specific for MACROSCOPIC_GNN
        self.macro_graph_data = None
        self.name_to_idx = {}
        
        self.is_loaded = False
        
        if TORCH_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the neural network model."""
        try:
            # V2: Macroscopic Model (The new state-of-the-art framework)
            if self.model_type == ModelType.MACROSCOPIC_GNN:
                self._load_macroscopic_gnn()
                if self.is_loaded:
                    return

            # First try loading the trained GNN model (V1)
            if self.model_type == ModelType.TRAINED_GNN:
                self._load_trained_gnn()
                if self.is_loaded:
                    return

            if self.model_type == ModelType.SIMPLE_MLP or not self.is_loaded:
                if SimpleDDIPredictor is None:
                    logger.warning("SimpleDDIPredictor not available (PyTorch not installed)")
                    self.is_loaded = False
                    return
                    
                self.model = SimpleDDIPredictor(
                    input_dim=4096,
                    hidden_dims=[1024, 512, 256],
                    num_classes=len(self.INTERACTION_TYPES)
                )
                self.model.eval()
                self.model_type = ModelType.SIMPLE_MLP
                self.is_loaded = True
                logger.info(f"Initialized {self.model_type.value} model (fallback)")
            else:
                logger.warning(f"Model type {self.model_type.value} not yet implemented")
                self.is_loaded = False
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.is_loaded = False

    def _load_macroscopic_gnn(self):
        """Load the V2 Macroscopic GraphSAGE Model and Tensor Space."""
        import sys
        import pandas as pd
        model_src = Path(__file__).parent.parent.parent.parent / 'src' / 'model'
        if str(model_src) not in sys.path:
            sys.path.insert(0, str(model_src))
            
        web_dir = Path(__file__).parent.parent.parent
        if str(web_dir) not in sys.path:
            sys.path.insert(0, str(web_dir))
            
        data_dir = Path(__file__).parent.parent.parent / 'data'
        
        try:
            from macroscopic_ddi_gnn import MacroscopicDDIGNN
            
            dataset_path = data_dir / "neo4j_gnn_dataset.pt"
            mapping_path = data_dir / "node_mapping.csv"
            weights_path = data_dir / "macroscopic_gnn_weights.pth"

            if not dataset_path.exists() or not mapping_path.exists() or not weights_path.exists():
                logger.warning("Macroscopic GNN assets missing. Ensure extract_graph_dataset.py and train_macroscopic_model.py have been executed.")
                return

            self.macro_graph_data = torch.load(dataset_path, weights_only=False)
            mapping_df = pd.read_csv(mapping_path)

            for _, row in mapping_df.iterrows():
                if pd.notna(row['name']):
                    self.name_to_idx[str(row['name']).strip().lower()] = row['pyg_id']
                    
            self.model = MacroscopicDDIGNN(
                in_channels=self.macro_graph_data.num_features,
                hidden_channels=256,
                out_channels=128,
                num_layers=3
            )
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
            self.model.eval()

            self.model_type = ModelType.MACROSCOPIC_GNN
            self.is_loaded = True
            logger.info("Successfully loaded V2 Macroscopic GNN Predictor (~98.6% AUC).")

        except Exception as e:
            logger.error(f"Failed to load Macroscopic GNN Predictor: {e}")
            self.model = None

    def _load_trained_gnn(self):
        """Load the trained Edge-Conditioned GIN model from checkpoint."""
        import sys
        model_src = Path(__file__).parent.parent.parent.parent / 'src' / 'model'
        if str(model_src) not in sys.path:
            sys.path.insert(0, str(model_src.parent))

        try:
            from model.gnn_model import DDIGraphModel
            from model.gnn_featurizer import (
                MolecularGraphFeaturizer, ATOM_FEATURE_DIM, EDGE_FEATURE_DIM
            )

            checkpoint_path = (
                Path(__file__).parent.parent.parent / 'models' / 'gnn' / 'gnn_best_model.pt'
            )
            if not checkpoint_path.exists():
                logger.warning(f"Trained GNN checkpoint not found at {checkpoint_path}")
                return

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})

            self.model = DDIGraphModel(
                atom_feature_dim=ATOM_FEATURE_DIM,
                edge_feature_dim=EDGE_FEATURE_DIM,
                hidden_dim=config.get('hidden_dim', 256),
                num_gnn_layers=config.get('num_gnn_layers', 3),
                num_relation_classes=config.get('num_relation_classes', 1),
                dropout_rate=config.get('dropout_rate', 0.1),
                use_binary=config.get('use_binary', True),
                use_jumping_knowledge=config.get('use_jumping_knowledge', True),
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.graph_featurizer = MolecularGraphFeaturizer(
                max_atoms=config.get('max_atoms', 128)
            )

            # Store calibration parameters
            self._temperature = checkpoint.get('temperature', 1.0)
            self._platt_a = checkpoint.get('platt_a', 1.0)
            self._platt_b = checkpoint.get('platt_b', 0.0)

            metrics = checkpoint.get('metrics', {})
            logger.info(
                f"Loaded trained GNN model (PR-AUC: {metrics.get('pr_auc', 'N/A')}, "
                f"params: {sum(p.numel() for p in self.model.parameters()):,})"
            )
            self.model_type = ModelType.TRAINED_GNN
            self.is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load trained GNN: {e}")
            self.model = None
            self.graph_featurizer = None

    def _resolve_smiles(
        self, drug1: str, drug2: str,
        smiles1: Optional[str], smiles2: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Look up SMILES from drug_db.json for drugs missing SMILES."""
        if smiles1 and smiles2:
            return smiles1, smiles2

        if not hasattr(self, '_drug_db'):
            self._drug_db = {}
            db_path = Path(__file__).parent.parent.parent / 'data' / 'drug_db.json'
            if db_path.exists():
                import json
                try:
                    with open(db_path) as f:
                        for drug in json.load(f):
                            name = drug.get('name', '').lower()
                            if name and drug.get('smiles'):
                                self._drug_db[name] = drug['smiles']
                except Exception as e:
                    logger.warning(f"Failed to load drug_db.json: {e}")

        if not smiles1:
            smiles1 = self._drug_db.get(drug1.lower())
        if not smiles2:
            smiles2 = self._drug_db.get(drug2.lower())
        return smiles1, smiles2

    def predict(
        self,
        drug1: str,
        drug2: str,
        smiles1: Optional[str] = None,
        smiles2: Optional[str] = None
    ) -> GNNPrediction:
        """
        Predict DDI between two drugs using molecular features.
        
        Args:
            drug1: Name of first drug
            drug2: Name of second drug
            smiles1: SMILES of first drug (optional, will lookup if not provided)
            smiles2: SMILES of second drug
            
        Returns:
            GNNPrediction with interaction details
        """
        # Resolve SMILES from drug_db.json if not provided
        if not smiles1 or not smiles2:
            smiles1, smiles2 = self._resolve_smiles(drug1, drug2, smiles1, smiles2)

        # Calculate similarity if both SMILES available
        similarity = None
        if smiles1 and smiles2:
            similarity = self.feature_extractor.calculate_tanimoto_similarity(smiles1, smiles2)

        # Use Macroscopic GraphSAGE Engine (V2)
        if self.is_loaded and self.model_type == ModelType.MACROSCOPIC_GNN:
            result = self._predict_with_macroscopic_model(drug1, drug2, smiles1, smiles2, similarity)
            return self._apply_correction_calibration(result, drug1, drug2)

        # Use legacy trained GNN model if available
        if (self.is_loaded and self.model_type == ModelType.TRAINED_GNN
                and self.graph_featurizer is not None
                and smiles1 and smiles2):
            result = self._predict_with_trained_gnn(
                drug1, drug2, smiles1, smiles2, similarity
            )
            return self._apply_correction_calibration(result, drug1, drug2)

        # Get fingerprints for fallback MLP
        fp1 = self.feature_extractor.smiles_to_fingerprint(smiles1) if smiles1 else None
        fp2 = self.feature_extractor.smiles_to_fingerprint(smiles2) if smiles2 else None
        
        # If no SMILES or model not loaded, return heuristic prediction
        if not self.is_loaded or fp1 is None or fp2 is None:
            result = self._heuristic_prediction(
                drug1,
                drug2,
                smiles1,
                smiles2,
                similarity,
                fallback_reason="model_not_loaded_or_missing_smiles",
            )
            return self._apply_correction_calibration(result, drug1, drug2)

        # Run MLP fallback inference
        try:
            import torch
            
            # Concatenate fingerprints
            combined_fp = torch.tensor(fp1 + fp2, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.model(combined_fp)
                probs = F.softmax(logits, dim=-1)[0]
            
            # Get prediction
            pred_idx = torch.argmax(probs).item()
            interaction_type = self.INTERACTION_TYPES[pred_idx]
            confidence = float(probs[pred_idx])
            interaction_probability = float(1 - probs[0])  # 1 - P(no_interaction)
            
            severity, risk_score = self.SEVERITY_MAP.get(interaction_type, ('unknown', 0.5))
            
            # Generate mechanism hypothesis based on structural features
            mechanism = self._generate_mechanism_hypothesis(
                drug1, drug2, smiles1, smiles2, similarity, interaction_type
            )
            
            result = GNNPrediction(
                drug1=drug1,
                drug2=drug2,
                interaction_probability=interaction_probability,
                interaction_type=interaction_type,
                confidence=confidence,
                severity=severity,
                model_used=self.model_type.value,
                smiles1=smiles1,
                smiles2=smiles2,
                fingerprint_similarity=similarity,
                mechanism_hypothesis=mechanism,
                raw_interaction_probability=interaction_probability,
                calibration_method="none",
                calibration_version="none",
                provenance={
                    "prediction_path": "simple_mlp",
                    "model_backend": "simple_mlp",
                },
            )
            return self._apply_correction_calibration(result, drug1, drug2)

        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            result = self._heuristic_prediction(
                drug1,
                drug2,
                smiles1,
                smiles2,
                similarity,
                fallback_reason="mlp_inference_error",
            )
            return self._apply_correction_calibration(result, drug1, drug2)

    def _apply_correction_calibration(
        self, result: GNNPrediction, drug1: str, drug2: str
    ) -> GNNPrediction:
        """Apply correction-based calibration to a GNN prediction result."""
        try:
            from .confidence_calibrator import get_confidence_calibrator
            calibrator = get_confidence_calibrator()
            adj_severity, adj_confidence, cal_info = calibrator.adjust(
                drug1, drug2, result.severity, result.confidence
            )
            if cal_info.get('correction_adjusted'):
                result.severity = adj_severity
                result.confidence = adj_confidence
                result.calibration_method = f"{result.calibration_method}+correction"
                logger.info("Calibrator adjusted %s+%s: %s", drug1, drug2, cal_info)
        except Exception as e:
            logger.debug("Calibration adjustment skipped: %s", e)
        return result

    def _predict_with_macroscopic_model(
        self,
        drug1: str,
        drug2: str,
        smiles1: Optional[str],
        smiles2: Optional[str],
        similarity: Optional[float]
    ) -> GNNPrediction:
        """Run prediction using the V2 Macroscopic GraphSAGE Model."""
        try:
            d1_clean = drug1.strip().lower()
            d2_clean = drug2.strip().lower()
            
            # Helper to find close matches if exact isn't in graph
            def _get_idx(name):
                if name in self.name_to_idx:
                    return name, self.name_to_idx[name]
                matches = get_close_matches(name, self.name_to_idx.keys(), n=1, cutoff=0.7)
                if matches:
                    return matches[0], self.name_to_idx[matches[0]]
                return None, None
                
            actual_d1, idx1 = _get_idx(d1_clean)
            actual_d2, idx2 = _get_idx(d2_clean)

            # If either drug is entirely foreign to the graph, fallback to heuristics
            if idx1 is None or idx2 is None:
                logger.info(f"Macroscopic model fallback: {drug1} or {drug2} not mapped in graph space.")
                return self._heuristic_prediction(
                    drug1,
                    drug2,
                    smiles1,
                    smiles2,
                    similarity,
                    fallback_reason="drug_not_in_graph_embedding",
                )

            # Create testing tensor array
            edge_label_index = torch.tensor([[idx1], [idx2]], dtype=torch.long)
            
            with torch.no_grad():
                out = self.model(self.macro_graph_data, edge_label_index)
                prob = torch.sigmoid(out).item()

            # Map Macroscopic Probability to Severity
            if prob < 0.3:
                interaction_type = 'no_interaction'
                severity = 'none'
            elif prob < 0.6:
                interaction_type = 'advise'
                severity = 'minor'
            elif prob < 0.85:
                interaction_type = 'effect'
                severity = 'moderate'
            else:
                interaction_type = 'mechanism'
                severity = 'severe'

            # Calculate confidence curve (confident when very high or very low)
            confidence = abs(prob - 0.5) * 2  

            mechanism = self._generate_mechanism_hypothesis(
                actual_d1.title(), actual_d2.title(), smiles1, smiles2, similarity, interaction_type
            )
            mechanism = f"[Macroscopic Tensor DB Match] {mechanism}"

            return GNNPrediction(
                drug1=drug1,
                drug2=drug2,
                interaction_probability=float(prob),
                interaction_type=interaction_type,
                confidence=float(confidence),
                severity=severity,
                model_used='macroscopic_gnn',
                smiles1=smiles1,
                smiles2=smiles2,
                fingerprint_similarity=similarity,
                mechanism_hypothesis=mechanism,
                raw_interaction_probability=float(prob),
                calibration_method="none",
                calibration_version="none",
                provenance={
                    "prediction_path": "macroscopic_gnn",
                    "model_backend": "macroscopic_gnn",
                    "graph_match": {
                        "drug1": actual_d1,
                        "drug2": actual_d2,
                    },
                    "node_indices": [int(idx1), int(idx2)],
                },
            )

        except Exception as e:
            logger.error(f"Macroscopic GNN inference failed: {e}")
            return self._heuristic_prediction(
                drug1,
                drug2,
                smiles1,
                smiles2,
                similarity,
                fallback_reason="macroscopic_inference_error",
            )

    def _predict_with_trained_gnn(
        self,
        drug1: str,
        drug2: str,
        smiles1: str,
        smiles2: str,
        similarity: Optional[float]
    ) -> GNNPrediction:
        """Run prediction using the trained Edge-Conditioned GIN model."""
        try:
            graphs = self.graph_featurizer.smiles_pair_to_graphs(smiles1, smiles2)
            if graphs is None:
                return self._heuristic_prediction(
                    drug1,
                    drug2,
                    smiles1,
                    smiles2,
                    similarity,
                    fallback_reason="graph_featurization_failed",
                )

            with torch.no_grad():
                # Add batch dimension
                logits = self.model(
                    graphs['drug1_node_features'].unsqueeze(0),
                    graphs['drug1_adjacency'].unsqueeze(0),
                    graphs['drug1_edge_features'].unsqueeze(0),
                    graphs['drug1_node_mask'].unsqueeze(0),
                    graphs['drug2_node_features'].unsqueeze(0),
                    graphs['drug2_adjacency'].unsqueeze(0),
                    graphs['drug2_edge_features'].unsqueeze(0),
                    graphs['drug2_node_mask'].unsqueeze(0),
                )

                # Apply Platt scaling calibration: sigmoid(a * logit + b)
                raw_logit = logits.squeeze()
                raw_prob = torch.sigmoid(raw_logit).item()
                calibrated = self._platt_a * raw_logit + self._platt_b
                prob = torch.sigmoid(calibrated).item()

            # Map binary probability to severity
            if prob < 0.3:
                interaction_type = 'no_interaction'
                severity = 'none'
            elif prob < 0.5:
                interaction_type = 'advise'
                severity = 'minor'
            elif prob < 0.7:
                interaction_type = 'effect'
                severity = 'moderate'
            else:
                interaction_type = 'mechanism'
                severity = 'severe'

            confidence = abs(prob - 0.5) * 2  # 0 at 0.5, 1 at 0/1

            mechanism = self._generate_mechanism_hypothesis(
                drug1, drug2, smiles1, smiles2, similarity, interaction_type
            )

            return GNNPrediction(
                drug1=drug1,
                drug2=drug2,
                interaction_probability=float(prob),
                interaction_type=interaction_type,
                confidence=float(confidence),
                severity=severity,
                model_used='trained_gnn',
                smiles1=smiles1,
                smiles2=smiles2,
                fingerprint_similarity=similarity,
                mechanism_hypothesis=mechanism,
                raw_interaction_probability=float(raw_prob),
                calibration_method="platt_scaling",
                calibration_version=f"platt_a={self._platt_a:.6f};platt_b={self._platt_b:.6f}",
                provenance={
                    "prediction_path": "trained_gnn",
                    "model_backend": "edge_conditioned_gin",
                    "temperature": self._temperature,
                    "platt_a": self._platt_a,
                    "platt_b": self._platt_b,
                },
            )

        except Exception as e:
            logger.error(f"Trained GNN inference failed: {e}")
            return self._heuristic_prediction(
                drug1,
                drug2,
                smiles1,
                smiles2,
                similarity,
                fallback_reason="trained_gnn_inference_error",
            )
    
    def _heuristic_prediction(
        self,
        drug1: str,
        drug2: str,
        smiles1: Optional[str],
        smiles2: Optional[str],
        similarity: Optional[float],
        fallback_reason: Optional[str] = None,
    ) -> GNNPrediction:
        """
        Generate heuristic prediction when model unavailable.
        
        Uses structural similarity and known drug classes to estimate risk.
        """
        # Default prediction
        interaction_type = 'no_interaction'
        confidence = 0.3
        interaction_prob = 0.1
        
        # Adjust based on similarity
        if similarity is not None:
            if similarity > 0.7:
                # Very similar structures - likely same target/pathway
                interaction_type = 'mechanism'
                interaction_prob = 0.6
                confidence = 0.5
            elif similarity > 0.4:
                # Moderate similarity - possible interaction
                interaction_type = 'int'
                interaction_prob = 0.3
                confidence = 0.4
        
        severity, _ = self.SEVERITY_MAP.get(interaction_type, ('unknown', 0.5))
        
        mechanism = f"Structural analysis indicates {similarity * 100:.1f}% molecular similarity. " if similarity is not None else ""
        mechanism += "Prediction based on heuristic analysis - verify with clinical data."
        
        return GNNPrediction(
            drug1=drug1,
            drug2=drug2,
            interaction_probability=interaction_prob,
            interaction_type=interaction_type,
            confidence=confidence,
            severity=severity,
            model_used='heuristic',
            smiles1=smiles1,
            smiles2=smiles2,
            fingerprint_similarity=similarity,
            mechanism_hypothesis=mechanism,
            raw_interaction_probability=interaction_prob,
            calibration_method="none",
            calibration_version="none",
            fallback_reason=fallback_reason,
            provenance={
                "prediction_path": "heuristic",
                "fallback_reason": fallback_reason,
                "has_model": self.is_loaded,
                "has_smiles": bool(smiles1 and smiles2),
            },
        )
    
    def _generate_mechanism_hypothesis(
        self,
        drug1: str,
        drug2: str,
        smiles1: Optional[str],
        smiles2: Optional[str],
        similarity: Optional[float],
        interaction_type: str
    ) -> str:
        """
        Generate a hypothesis about the interaction mechanism.
        
        Based on molecular features and predicted interaction type.
        """
        hypotheses = {
            'mechanism': f"GNN analysis suggests {drug1} and {drug2} may share metabolic pathways or targets based on structural features.",
            'effect': f"Structural analysis indicates {drug1} and {drug2} may have overlapping pharmacodynamic effects.",
            'advise': f"Based on molecular similarity, clinical monitoring is recommended when combining {drug1} and {drug2}.",
            'int': f"GNN model predicts potential interaction between {drug1} and {drug2} based on molecular features.",
            'no_interaction': f"GNN analysis suggests low interaction probability between {drug1} and {drug2}."
        }
        
        base_hypothesis = hypotheses.get(interaction_type, "Interaction mechanism requires further investigation.")
        
        if similarity is not None:
            base_hypothesis += f" Molecular similarity: {similarity:.1%}."
        
        return base_hypothesis
    
    def predict_polypharmacy(self, drugs: List[Dict[str, str]]) -> Dict:
        """
        Predict interactions for multiple drugs by evaluating all unique pairs.

        Returns a response contract that matches the existing API/view/frontend
        expectations for polypharmacy network rendering.
        """
        n = len(drugs)
        interactions: List[Dict[str, Any]] = []
        max_risk = 0.0
        degree_count: Dict[str, int] = {}
        drug_names = [d.get("name", "Unknown") for d in drugs]

        def risk_level_from_score(score: float) -> str:
            if score >= 0.8:
                return "critical"
            if score >= 0.6:
                return "high"
            if score >= 0.3:
                return "medium"
            return "low"

        # We expose system buckets for BodyMap aggregation even when
        # pairwise inference does not provide detailed organ labels.
        interaction_system_map = {
            "mechanism": ["Metabolic/CYP450"],
            "effect": ["Pharmacodynamic"],
            "advise": ["Systemic"],
            "int": ["Systemic"],
        }

        for i in range(n):
            for j in range(i + 1, n):
                pred = self.predict(
                    drugs[i].get("name", "Unknown"),
                    drugs[j].get("name", "Unknown"),
                    drugs[i].get("smiles", ""),
                    drugs[j].get("smiles", "")
                )

                risk_score = float(pred.interaction_probability)

                # Keep response size focused on clinically meaningful edges.
                if risk_score <= 0.3:
                    continue

                source = pred.drug1
                target = pred.drug2
                affected_systems = interaction_system_map.get(pred.interaction_type, ["Systemic"])

                interactions.append({
                    "source": source,
                    "target": target,
                    "risk_score": risk_score,
                    "risk_level": risk_level_from_score(risk_score),
                    "severity": pred.severity,
                    "mechanism": pred.mechanism_hypothesis,
                    "affected_systems": affected_systems,
                })

                max_risk = max(max_risk, risk_score)
                degree_count[source] = degree_count.get(source, 0) + 1
                degree_count[target] = degree_count.get(target, 0) + 1

        interactions.sort(key=lambda x: x["risk_score"], reverse=True)
        hub_drug = max(degree_count, key=degree_count.get) if degree_count else None

        return {
            "drugs": drug_names,
            "interactions": interactions,
            "total_interactions": len(interactions),
            "max_risk_score": max_risk,
            "overall_risk_level": risk_level_from_score(max_risk),
            "hub_drug": hub_drug,
            "hub_interaction_count": degree_count.get(hub_drug, 0) if hub_drug else 0,
            "drugs_analyzed": n,
        }


    def batch_predict(
        self,
        drug_pairs: List[Tuple[str, str, Optional[str], Optional[str]]]
    ) -> List[GNNPrediction]:
        """
        Predict interactions for multiple drug pairs.
        
        Args:
            drug_pairs: List of (drug1, drug2, smiles1, smiles2) tuples
            
        Returns:
            List of GNNPrediction objects
        """
        return [
            self.predict(d1, d2, s1, s2) 
            for d1, d2, s1, s2 in drug_pairs
        ]


# Singleton instance
_gnn_predictor: Optional[GNNDDIPredictor] = None


def get_gnn_predictor(model_type: ModelType = ModelType.MACROSCOPIC_GNN) -> GNNDDIPredictor:
    """Get or create the GNN DDI predictor singleton."""
    global _gnn_predictor
    if _gnn_predictor is None or _gnn_predictor.model_type != model_type:
        _gnn_predictor = GNNDDIPredictor(model_type)
    return _gnn_predictor
