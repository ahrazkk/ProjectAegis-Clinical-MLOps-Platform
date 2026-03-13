"""
Molecular Graph Featurizer for GNN-based DDI Prediction
Converts SMILES strings to molecular graph tensors using RDKit

Reference: MCR III - GNN Model (Structure-based prediction)
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Atom feature dimensions
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se']
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
] if RDKIT_AVAILABLE else []

# Bond feature dimensions
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
] if RDKIT_AVAILABLE else []

# Total atom feature dimension:
# symbol(13) + degree(7) + charge(6) + hs(6) + hybridization(6) + aromatic(1) + ring(1) = 40
ATOM_FEATURE_DIM = len(ATOM_SYMBOLS) + 1 + len(DEGREES) + 1 + len(FORMAL_CHARGES) + 1 + \
                   len(NUM_HS) + 1 + len(HYBRIDIZATIONS) + 1 + 2  # 40

# Total edge feature dimension:
# bond_type(5) + conjugated(1) + ring(1) + stereo(1) = 8
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 3  # 8


def _one_hot(value, choices: list) -> List[int]:
    """Create one-hot encoding with an 'other' category."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # 'other' category
    return encoding


def get_atom_features(atom) -> List[float]:
    """
    Extract atom-level features for GNN node representation.

    Features (40-dim):
    - Atom symbol: one-hot (C, N, O, S, F, Cl, Br, I, P, Si, B, Se, other)
    - Degree: one-hot (0-5, other)
    - Formal charge: one-hot (-2 to +2, other)
    - Num hydrogens: one-hot (0-4, other)
    - Hybridization: one-hot (SP, SP2, SP3, SP3D, SP3D2, other)
    - Is aromatic: binary
    - Is in ring: binary

    Args:
        atom: RDKit atom object

    Returns:
        Feature vector as list of floats
    """
    features = []
    features.extend(_one_hot(atom.GetSymbol(), ATOM_SYMBOLS))
    features.extend(_one_hot(atom.GetDegree(), DEGREES))
    features.extend(_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES))
    features.extend(_one_hot(atom.GetNumExplicitHs(), NUM_HS))
    features.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    return features


def get_bond_features(bond) -> List[float]:
    """
    Extract bond-level features for GNN edge representation.

    Features (8-dim):
    - Bond type: one-hot (single, double, triple, aromatic, other)
    - Is conjugated: binary
    - Is in ring: binary
    - Stereo: binary (has stereo or not)

    Args:
        bond: RDKit bond object

    Returns:
        Feature vector as list of floats
    """
    features = []
    features.extend(_one_hot(bond.GetBondType(), BOND_TYPES))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    features.append(1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0)
    return features


class MolecularGraphFeaturizer:
    """
    Converts SMILES strings to padded molecular graph tensors.

    Produces fixed-size tensors for batching without torch_geometric:
    - Node features: [max_atoms, atom_feature_dim]
    - Adjacency matrix: [max_atoms, max_atoms]
    - Edge feature matrix: [max_atoms, max_atoms, edge_feature_dim]
    - Node mask: [max_atoms] (1 for real atoms, 0 for padding)

    Reference: MCR III - GNN Model for structure-aware DDI prediction
    """

    def __init__(self, max_atoms: int = 128):
        """
        Initialize featurizer.

        Args:
            max_atoms: Maximum number of atoms per molecule (molecules with
                       more atoms are truncated, smaller ones are padded)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for molecular featurization. "
                "Install with: pip install rdkit-pypi"
            )
        self.max_atoms = max_atoms
        self.atom_feature_dim = ATOM_FEATURE_DIM
        self.edge_feature_dim = EDGE_FEATURE_DIM

    def smiles_to_graph(self, smiles: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a SMILES string to padded molecular graph tensors.

        Args:
            smiles: SMILES string representing the molecule

        Returns:
            Dictionary with:
            - node_features: [max_atoms, atom_feature_dim] float tensor
            - adjacency: [max_atoms, max_atoms] float tensor
            - edge_features: [max_atoms, max_atoms, edge_feature_dim] float tensor
            - node_mask: [max_atoms] float tensor
            - num_atoms: int (actual number of atoms before padding)
            Returns None if SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

<<<<<<< HEAD
        # Add hydrogens for accurate counts, then remove
        # (Skip 3D embedding — GNN only uses 2D topology features)
        mol = Chem.AddHs(mol)
=======
        # Add hydrogens for better feature extraction, then remove
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
>>>>>>> 945095214748d4fd654c25672d2fb69c01a3f8d9
        mol = Chem.RemoveHs(mol)

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None

        effective_atoms = min(num_atoms, self.max_atoms)

        # Initialize tensors
        node_features = torch.zeros(self.max_atoms, self.atom_feature_dim)
        adjacency = torch.zeros(self.max_atoms, self.max_atoms)
        edge_features = torch.zeros(self.max_atoms, self.max_atoms, self.edge_feature_dim)
        node_mask = torch.zeros(self.max_atoms)

        # Fill atom (node) features
        for i in range(effective_atoms):
            atom = mol.GetAtomWithIdx(i)
            features = get_atom_features(atom)
            node_features[i] = torch.tensor(features, dtype=torch.float)
            node_mask[i] = 1.0

        # Fill bond (edge) features and adjacency matrix
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i >= self.max_atoms or j >= self.max_atoms:
                continue

            # Adjacency (undirected)
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

            # Edge features (symmetric)
            bond_feat = get_bond_features(bond)
            edge_features[i, j] = torch.tensor(bond_feat, dtype=torch.float)
            edge_features[j, i] = torch.tensor(bond_feat, dtype=torch.float)

        # Add self-loops to adjacency
        for i in range(effective_atoms):
            adjacency[i, i] = 1.0

        return {
            'node_features': node_features,
            'adjacency': adjacency,
            'edge_features': edge_features,
            'node_mask': node_mask,
            'num_atoms': effective_atoms
        }

    def smiles_pair_to_graphs(
        self,
        smiles1: str,
        smiles2: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Convert a pair of SMILES strings to graph tensors for DDI prediction.

        Args:
            smiles1: SMILES for first drug
            smiles2: SMILES for second drug

        Returns:
            Dictionary with drug1_* and drug2_* prefixed tensors,
            or None if either SMILES is invalid.
        """
        graph1 = self.smiles_to_graph(smiles1)
        graph2 = self.smiles_to_graph(smiles2)

        if graph1 is None or graph2 is None:
            return None

        return {
            'drug1_node_features': graph1['node_features'],
            'drug1_adjacency': graph1['adjacency'],
            'drug1_edge_features': graph1['edge_features'],
            'drug1_node_mask': graph1['node_mask'],
            'drug2_node_features': graph2['node_features'],
            'drug2_adjacency': graph2['adjacency'],
            'drug2_edge_features': graph2['edge_features'],
            'drug2_node_mask': graph2['node_mask'],
        }

    def get_molecular_descriptor(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        Get global molecular descriptors as auxiliary features.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of molecular descriptors, or None if invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        }
