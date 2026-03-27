"""
GNN Inference Module for DDI Prediction
High-level API for making predictions with trained GNN models

Reference: MCR III - GNN Model inference pipeline
"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

from .gnn_model import DDIGraphModel
from .gnn_featurizer import MolecularGraphFeaturizer, ATOM_FEATURE_DIM, EDGE_FEATURE_DIM
from .risk_scorer import RiskScorer, TemperatureScaling

logger = logging.getLogger(__name__)


@dataclass
class DDIPrediction:
    """
    Result of a DDI prediction (shared by all model types)
    """
    drug1: str
    drug2: str
    has_interaction: bool
    interaction_type: Optional[str]
    raw_probability: float
    calibrated_probability: float
    risk_score: float
    risk_category: str
    confidence: float


# Common drug name → SMILES lookup for convenience
DRUG_SMILES_DB = {
    'aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'warfarin': 'CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O',
    'ibuprofen': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
    'metformin': 'CN(C)C(=N)NC(=N)N',
    'amoxicillin': 'CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@@H]1C(=O)O',
    'lisinopril': 'NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O',
    'atorvastatin': 'CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(c2ccc(F)cc2)c(c3ccccc3)c1C(=O)Nc4ccccc4',
    'metoprolol': 'COCCc1ccc(OCC(O)CNC(C)C)cc1',
    'omeprazole': 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1',
    'simvastatin': 'CCC(C)(C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)CC(=O)O)[C@@H]21',
    'clopidogrel': 'COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1',
    'acetaminophen': 'CC(=O)Nc1ccc(O)cc1',
    'ciprofloxacin': 'O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O',
    'fluoxetine': 'CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1',
    'sertraline': 'CN[C@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc21',
    'losartan': 'CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1',
    'amlodipine': 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl',
    'hydrochlorothiazide': 'NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O',
    'prednisone': 'C[C@]12C=CC(=O)C=C1CC[C@@H]1[C@@H]3CC[C@](O)(C(=O)CO)[C@@]3(C)C[C@H](O)[C@@H]12',
    'gabapentin': 'NCC1(CC(=O)O)CCCCC1',
}


class GNNPredictor:
    """
    High-level interface for GNN-based DDI prediction.

    Provides:
    - SMILES-based drug pair prediction
    - Drug name lookup (with built-in SMILES database)
    - Batch predictions
    - Risk scoring and categorization

    This predictor uses molecular structure (SMILES) instead of text,
    enabling predictions for novel drugs without published literature.
    """

    INTERACTION_TYPES = {
        0: 'none',
        1: 'advice',     # Minor
        2: 'effect',     # Moderate
        3: 'mechanism',  # Major (Pharmacokinetic)
        4: 'int'         # Major (Pharmacodynamic)
    }

    CLASS_TO_SEVERITY = {
        0: 'none',
        1: 'minor',
        2: 'moderate',
        3: 'major',
        4: 'major'
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_binary: bool = True,
        max_atoms: int = 128,
        drug_smiles_db: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            model_path: Path to saved GNN model checkpoint
            device: Computation device (cuda/cpu)
            use_binary: Whether model uses binary classification
            max_atoms: Maximum atoms per molecule for featurization
            drug_smiles_db: Custom drug name → SMILES mapping
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.use_binary = use_binary

        # Initialize featurizer
        self.featurizer = MolecularGraphFeaturizer(max_atoms=max_atoms)

        # Initialize model
        self.model = None
        self.temperature_scaling = TemperatureScaling()

        # Risk scorer
        self.risk_scorer = RiskScorer(class_to_severity=self.CLASS_TO_SEVERITY)

        # Drug name → SMILES database
        self.drug_smiles_db = {
            **(drug_smiles_db or {}),
            **DRUG_SMILES_DB
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load GNN model from checkpoint.

        Args:
            model_path: Path to model checkpoint file
        """
        logger.info(f"Loading GNN model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})

        self.model = DDIGraphModel(
            atom_feature_dim=ATOM_FEATURE_DIM,
            edge_feature_dim=EDGE_FEATURE_DIM,
            hidden_dim=config.get('hidden_dim', 256),
            num_gnn_layers=config.get('num_gnn_layers', 3),
            num_relation_classes=config.get('num_relation_classes', 1),
            dropout_rate=config.get('dropout_rate', 0.1),
            use_binary=config.get('use_binary', True),
            use_jumping_knowledge=config.get('use_jumping_knowledge', True)
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        if 'temperature' in checkpoint:
            self.temperature_scaling.temperature.data = torch.tensor(
                [checkpoint['temperature']]
            )

        self.use_binary = config.get('use_binary', True)
        logger.info("GNN model loaded successfully")

    def resolve_smiles(self, drug: str) -> Optional[str]:
        """
        Resolve a drug identifier to its SMILES string.

        Accepts either a SMILES string directly or a drug name
        that exists in the lookup database.

        Args:
            drug: SMILES string or drug name

        Returns:
            SMILES string, or None if unresolvable
        """
        # Check if it's already a valid SMILES
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(drug)
            if mol is not None:
                return drug
        except ImportError:
            pass

        # Look up by name (case-insensitive)
        name_lower = drug.lower().strip()
        if name_lower in self.drug_smiles_db:
            return self.drug_smiles_db[name_lower]

        return None

    def predict_from_smiles(
        self,
        smiles1: str,
        smiles2: str,
        drug1_name: Optional[str] = None,
        drug2_name: Optional[str] = None
    ) -> DDIPrediction:
        """
        Predict DDI from two SMILES strings.

        Args:
            smiles1: SMILES for first drug
            smiles2: SMILES for second drug
            drug1_name: Optional display name for first drug
            drug2_name: Optional display name for second drug

        Returns:
            DDIPrediction with risk assessment
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Featurize molecules
        graphs = self.featurizer.smiles_pair_to_graphs(smiles1, smiles2)
        if graphs is None:
            raise ValueError(
                f"Failed to parse SMILES. "
                f"Drug1: '{smiles1}', Drug2: '{smiles2}'"
            )

        # Prepare tensors (add batch dimension)
        d1_nf = graphs['drug1_node_features'].unsqueeze(0).to(self.device)
        d1_adj = graphs['drug1_adjacency'].unsqueeze(0).to(self.device)
        d1_ef = graphs['drug1_edge_features'].unsqueeze(0).to(self.device)
        d1_mask = graphs['drug1_node_mask'].unsqueeze(0).to(self.device)
        d2_nf = graphs['drug2_node_features'].unsqueeze(0).to(self.device)
        d2_adj = graphs['drug2_adjacency'].unsqueeze(0).to(self.device)
        d2_ef = graphs['drug2_edge_features'].unsqueeze(0).to(self.device)
        d2_mask = graphs['drug2_node_mask'].unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(
                d1_nf, d1_adj, d1_ef, d1_mask,
                d2_nf, d2_adj, d2_ef, d2_mask
            )

        # Calculate probabilities and risk
        if self.use_binary:
            raw_prob = torch.sigmoid(logits).item()
            calibrated_probs = self.temperature_scaling(logits)
            calibrated_prob = calibrated_probs.squeeze().item()

            # Updated to 0.6 threshold to eliminate "Alert Fatigue" (False Positives)
            has_interaction = raw_prob >= 0.6
            interaction_type = None if not has_interaction else 'interaction'
            risk_score = calibrated_prob
        else:
            raw_probs = torch.softmax(logits, dim=-1)
            calibrated_probs = self.temperature_scaling(logits)

            predicted_class = torch.argmax(raw_probs, dim=-1).item()
            raw_prob = raw_probs[0, predicted_class].item()
            calibrated_prob = calibrated_probs[0, predicted_class].item()

            has_interaction = predicted_class > 0
            interaction_type = self.INTERACTION_TYPES.get(
                predicted_class, 'unknown'
            )
            risk_score = self.risk_scorer.calculate_risk_score(
                calibrated_probs.cpu().numpy()
            )[0]

        risk_category_result = self.risk_scorer.categorize_risk(risk_score)
        risk_category = (
            risk_category_result
            if isinstance(risk_category_result, str)
            else risk_category_result[0]
        )

        # Confidence
        if self.use_binary:
            confidence = abs(raw_prob - 0.5) * 2
        else:
            confidence = raw_prob

        return DDIPrediction(
            drug1=drug1_name or smiles1,
            drug2=drug2_name or smiles2,
            has_interaction=has_interaction,
            interaction_type=interaction_type,
            raw_probability=float(raw_prob),
            calibrated_probability=float(calibrated_prob),
            risk_score=float(risk_score),
            risk_category=risk_category,
            confidence=float(confidence)
        )

    def predict_from_names(
        self,
        drug1_name: str,
        drug2_name: str
    ) -> DDIPrediction:
        """
        Predict DDI from drug names using the SMILES lookup database.

        Args:
            drug1_name: Name of first drug
            drug2_name: Name of second drug

        Returns:
            DDIPrediction with risk assessment

        Raises:
            ValueError: If drug name cannot be resolved to SMILES
        """
        smiles1 = self.resolve_smiles(drug1_name)
        smiles2 = self.resolve_smiles(drug2_name)

        if smiles1 is None:
            raise ValueError(
                f"Cannot resolve '{drug1_name}' to SMILES. "
                f"Provide SMILES directly or add to drug database."
            )
        if smiles2 is None:
            raise ValueError(
                f"Cannot resolve '{drug2_name}' to SMILES. "
                f"Provide SMILES directly or add to drug database."
            )

        return self.predict_from_smiles(
            smiles1, smiles2,
            drug1_name=drug1_name,
            drug2_name=drug2_name
        )

    def predict_batch(
        self,
        drug_pairs: List[Dict[str, str]]
    ) -> List[DDIPrediction]:
        """
        Batch prediction for multiple drug pairs.

        Args:
            drug_pairs: List of dicts with keys:
                - drug1_smiles or drug1_name
                - drug2_smiles or drug2_name

        Returns:
            List of DDIPrediction objects
        """
        predictions = []

        for pair in drug_pairs:
            smiles1 = pair.get('drug1_smiles') or self.resolve_smiles(
                pair.get('drug1_name', '')
            )
            smiles2 = pair.get('drug2_smiles') or self.resolve_smiles(
                pair.get('drug2_name', '')
            )

            if smiles1 is None or smiles2 is None:
                logger.warning(f"Skipping pair: could not resolve SMILES")
                continue

            pred = self.predict_from_smiles(
                smiles1, smiles2,
                drug1_name=pair.get('drug1_name'),
                drug2_name=pair.get('drug2_name')
            )
            predictions.append(pred)

        return predictions

    def add_drug_smiles(self, name: str, smiles: str):
        """
        Add a drug name → SMILES mapping to the lookup database.

        Args:
            name: Drug name (case-insensitive)
            smiles: SMILES string
        """
        self.drug_smiles_db[name.lower().strip()] = smiles

    def get_risk_explanation(self, prediction: DDIPrediction) -> str:
        """
        Generate human-readable explanation of GNN risk assessment.

        Args:
            prediction: DDIPrediction object

        Returns:
            Explanation string
        """
        if not prediction.has_interaction:
            return (
                f"No significant drug-drug interaction detected between "
                f"{prediction.drug1} and {prediction.drug2} based on "
                f"molecular structure analysis. "
                f"Confidence: {prediction.confidence:.1%}"
            )

        severity_explanations = {
            'low': "minimal clinical concern based on molecular structure",
            'moderate': "monitor patient - structural analysis suggests potential interaction",
            'high': "significant structural interaction risk - consider alternative therapy"
        }

        return (
            f"Potential drug-drug interaction detected between "
            f"{prediction.drug1} and {prediction.drug2} "
            f"(structure-based GNN analysis).\n"
            f"- Risk Level: {prediction.risk_category.upper()}\n"
            f"- Risk Score: {prediction.risk_score:.2f}\n"
            f"- Confidence: {prediction.confidence:.1%}\n"
            f"- Recommendation: {severity_explanations.get(prediction.risk_category, 'Review with clinical pharmacist')}"
        )
