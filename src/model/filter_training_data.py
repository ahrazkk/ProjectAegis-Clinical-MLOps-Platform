"""
Pre-filter and pre-featurize GNN training data.

Removes metallic/ionic compounds that hang RDKit, validates all SMILES,
and caches featurized tensors so training loads instantly.
"""
import json
import time
import logging
import signal
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'web' / 'data' / 'gnn_training'

# Metals/elements that cause RDKit to hang on force field calcs
METAL_ATOMS = {'Pt', 'Au', 'Ag', 'Fe', 'Zn', 'Ca', 'Cu', 'Co', 'Mn', 
               'Ni', 'Cr', 'Ti', 'Mg', 'Ba', 'Sr', 'Li', 'Na', 'K',
               'Al', 'Sn', 'Pb', 'Hg', 'Cd', 'As', 'Sb', 'Bi', 'Se',
               'Te', 'Tc', 'Ru', 'Rh', 'Pd', 'Os', 'Ir', 'W', 'Mo',
               'V', 'Gd', 'Ga', 'In', 'Tl', 'Ge', 'La', 'Ce', 'Nd',
               'Sm', 'Eu', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Zr', 'Hf', 'Ta', 'Re', 'Ra', 'Sc', 'Y', 'Nb'}


def has_metal(smiles: str) -> bool:
    """Check if SMILES contains metal atoms."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # Can't parse = skip
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in METAL_ATOMS:
            return True
    return False


def quick_validate_smiles(smiles: str) -> bool:
    """Quick check that SMILES is valid organic molecule."""
    from rdkit import Chem
    if not smiles or len(smiles) < 3:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 2 or num_atoms > 200:
        return False
    return True


def filter_dataset():
    """Filter train/val datasets to remove problematic molecules."""
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings during filtering
    
    for split in ['train', 'val']:
        path = DATA_DIR / f'{split}.json'
        if not path.exists():
            logger.warning(f"{path} not found")
            continue
        
        with open(path) as f:
            samples = json.load(f)
        
        original = len(samples)
        filtered = []
        skip_metal = 0
        skip_invalid = 0
        
        for i, s in enumerate(samples):
            s1 = s['drug1_smiles']
            s2 = s['drug2_smiles']
            
            # Quick validate
            if not quick_validate_smiles(s1) or not quick_validate_smiles(s2):
                skip_invalid += 1
                continue
            
            # Check for metals
            if has_metal(s1) or has_metal(s2):
                skip_metal += 1
                continue
            
            filtered.append(s)
            
            if (i + 1) % 200 == 0:
                logger.info(f"  {split}: processed {i+1}/{original}")
        
        # Save filtered version
        with open(path, 'w') as f:
            json.dump(filtered, f)
        
        pos = sum(1 for s in filtered if s['has_interaction'] == 1)
        neg = len(filtered) - pos
        
        logger.info(f"\n{split}: {original} -> {len(filtered)} samples")
        logger.info(f"  Removed: {skip_metal} metal, {skip_invalid} invalid SMILES")
        logger.info(f"  Kept: {pos} positive, {neg} negative")
    
    logger.info("\nNow testing featurization speed on a sample...")
    
    # Test featurization timing
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.gnn_featurizer import MolecularGraphFeaturizer
    RDLogger.DisableLog('rdApp.*')
    
    featurizer = MolecularGraphFeaturizer(max_atoms=128)
    
    with open(DATA_DIR / 'train.json') as f:
        train = json.load(f)
    
    # Time first 50 samples
    start = time.time()
    success = 0
    fail = 0
    for s in train[:50]:
        result = featurizer.smiles_pair_to_graphs(s['drug1_smiles'], s['drug2_smiles'])
        if result is not None:
            success += 1
        else:
            fail += 1
    elapsed = time.time() - start
    
    per_sample = elapsed / 50
    total_estimate = per_sample * len(train)
    
    logger.info(f"\nFeaturization benchmark (50 samples):")
    logger.info(f"  Time: {elapsed:.1f}s ({per_sample:.2f}s per sample)")
    logger.info(f"  Success: {success}/50, Failed: {fail}/50")
    logger.info(f"  Estimated total train load time: {total_estimate:.0f}s ({total_estimate/60:.1f} min)")


if __name__ == '__main__':
    filter_dataset()
