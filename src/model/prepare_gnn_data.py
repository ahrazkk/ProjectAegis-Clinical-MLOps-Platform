"""
Prepare GNN Training Data

Builds a dataset of drug pairs with molecular SMILES and interaction labels
for GNN-based DDI prediction training.

Sources:
1. drug_db.json - 47 drugs with SMILES (curated DrugBank data)
2. Curated KNOWN_INTERACTIONS - 45 labeled drug pairs with severity
3. ddi_sentences.db - 3,445 mined drug pairs (optional expansion)
4. PubChem API - SMILES resolution for drugs without structures (optional)

Output: JSON files for train/val splits in web/data/gnn_training/

Usage:
    python prepare_gnn_data.py                  # Base dataset (curated only)
    python prepare_gnn_data.py --expand         # + PubChem expansion
    python prepare_gnn_data.py --expand --max-pubchem 200  # Limit API calls
"""

import json
import sqlite3
import random
import logging
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # molecular-ai/
WEB_DIR = PROJECT_ROOT / 'web'
DATA_DIR = WEB_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'gnn_training'
PUBCHEM_CACHE = DATA_DIR / 'pubchem_smiles_cache.json'

RANDOM_SEED = 42

# ============================================================================
# Curated Known Interactions (drug ID pairs from download_real_data.py)
# Extracted as static data to avoid Django import dependency
# ============================================================================

KNOWN_INTERACTION_PAIRS = [
    # (drug1_id, drug2_id, severity)
    # Warfarin interactions
    ("DB00682", "DB00945", "severe"),
    ("DB00682", "DB01050", "severe"),
    ("DB00682", "DB00788", "severe"),
    ("DB00682", "DB00586", "severe"),
    ("DB00682", "DB00564", "moderate"),
    ("DB00682", "DB00252", "moderate"),
    ("DB00682", "DB00338", "minor"),
    # Statin interactions
    ("DB00641", "DB01118", "severe"),
    ("DB00641", "DB00176", "severe"),
    ("DB00641", "DB00661", "moderate"),
    ("DB00641", "DB01211", "severe"),
    ("DB01076", "DB01211", "moderate"),
    # Digoxin interactions
    ("DB00390", "DB01118", "severe"),
    ("DB00390", "DB00661", "moderate"),
    ("DB00390", "DB00537", "minor"),
    # Immunosuppressant interactions
    ("DB00864", "DB00176", "severe"),
    ("DB00864", "DB01211", "severe"),
    ("DB00864", "DB00564", "moderate"),
    ("DB00091", "DB00641", "moderate"),
    # Antidepressant interactions
    ("DB00472", "DB00334", "moderate"),
    ("DB00176", "DB00338", "minor"),
    # Cardiovascular combinations
    ("DB00571", "DB00661", "moderate"),
    ("DB01136", "DB00390", "moderate"),
    ("DB00264", "DB00661", "moderate"),
    # Anticonvulsant interactions
    ("DB00564", "DB00472", "moderate"),
    ("DB00252", "DB00313", "moderate"),
    ("DB00564", "DB00555", "moderate"),
    # Antiplatelet + others
    ("DB00758", "DB00945", "moderate"),
    ("DB00758", "DB00338", "moderate"),
    # Metformin
    ("DB00331", "DB00537", "minor"),
    # QT prolongation
    ("DB01118", "DB00218", "severe"),
    ("DB00334", "DB00218", "moderate"),
    # Additional interactions
    ("DB00472", "DB01118", "severe"),
    ("DB00316", "DB00682", "minor"),
    ("DB00175", "DB00091", "moderate"),
    ("DB00715", "DB00264", "moderate"),
    ("DB01104", "DB00390", "minor"),
    ("DB00877", "DB01211", "severe"),
    ("DB00465", "DB00682", "severe"),
    ("DB00331", "DB01016", "moderate"),
    ("DB00390", "DB00999", "moderate"),
    ("DB01427", "DB00945", "moderate"),
    ("DB01098", "DB00091", "moderate"),
    ("DB00227", "DB01211", "severe"),
    ("DB01174", "DB00682", "moderate"),
]

SEVERITY_TO_CLASS = {
    "minor": 1,
    "moderate": 2,
    "severe": 3,
}


def load_drug_db() -> Dict[str, Dict]:
    """Load drugs from drug_db.json. Returns dict keyed by drugbank_id."""
    db_path = DATA_DIR / 'drug_db.json'
    if not db_path.exists():
        logger.error(f"drug_db.json not found at {db_path}")
        return {}

    with open(db_path, 'r', encoding='utf-8') as f:
        drugs = json.load(f)

    drug_map = {}
    for drug in drugs:
        did = drug.get('drugbank_id', '')
        smiles = drug.get('smiles', '')
        if did and smiles:
            drug_map[did] = {
                'name': drug['name'],
                'smiles': smiles,
                'category': drug.get('category', ''),
            }
    logger.info(f"Loaded {len(drug_map)} drugs with SMILES from drug_db.json")
    return drug_map


def build_curated_dataset(
    drug_map: Dict[str, Dict],
) -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """
    Build positive samples from curated KNOWN_INTERACTION_PAIRS.

    Returns:
        positives: List of sample dicts
        positive_pairs: Set of (id1, id2) tuples (both orderings)
    """
    positives = []
    positive_pairs = set()

    for id1, id2, severity in KNOWN_INTERACTION_PAIRS:
        if id1 not in drug_map or id2 not in drug_map:
            logger.debug(f"Skipping pair {id1}-{id2}: missing SMILES")
            continue

        d1 = drug_map[id1]
        d2 = drug_map[id2]
        cls = SEVERITY_TO_CLASS.get(severity, 2)

        # Add both orderings for symmetry
        for s1, s2, n1, n2 in [
            (d1['smiles'], d2['smiles'], d1['name'], d2['name']),
            (d2['smiles'], d1['smiles'], d2['name'], d1['name']),
        ]:
            positives.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': n1,
                'drug2_name': n2,
                'interaction_type': cls,
                'has_interaction': 1,
                'severity': severity,
                'source': 'curated',
            })

        positive_pairs.add((id1, id2))
        positive_pairs.add((id2, id1))

    logger.info(
        f"Built {len(positives)} positive samples from "
        f"{len(positives) // 2} curated interactions"
    )
    return positives, positive_pairs


def generate_negative_samples(
    drug_map: Dict[str, Dict],
    positive_pairs: Set[Tuple[str, str]],
    ratio: float = 2.0,
) -> List[Dict]:
    """
    Generate negative (non-interacting) drug pairs.

    Samples from all possible drug pair combinations that aren't in the
    positive set. Uses a controlled ratio to manage class imbalance.

    Args:
        drug_map: Drug ID -> drug info dict
        positive_pairs: Set of known interacting (id1, id2) pairs
        ratio: Negative:Positive ratio (e.g., 2.0 means 2x negatives)
    """
    random.seed(RANDOM_SEED)
    drug_ids = list(drug_map.keys())
    num_positives = len(positive_pairs) // 2  # Each pair counted twice
    target_negatives = int(num_positives * ratio)

    # Generate all possible non-interacting pairs
    candidate_negatives = []
    for id1, id2 in combinations(drug_ids, 2):
        if (id1, id2) not in positive_pairs:
            candidate_negatives.append((id1, id2))

    # Sample negatives
    if len(candidate_negatives) > target_negatives:
        sampled = random.sample(candidate_negatives, target_negatives)
    else:
        sampled = candidate_negatives
        logger.warning(
            f"Only {len(sampled)} negative pairs available "
            f"(target was {target_negatives})"
        )

    negatives = []
    for id1, id2 in sampled:
        d1 = drug_map[id1]
        d2 = drug_map[id2]
        # Add both orderings
        for s1, s2, n1, n2 in [
            (d1['smiles'], d2['smiles'], d1['name'], d2['name']),
            (d2['smiles'], d1['smiles'], d2['name'], d1['name']),
        ]:
            negatives.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': n1,
                'drug2_name': n2,
                'interaction_type': 0,
                'has_interaction': 0,
                'severity': 'none',
                'source': 'negative_sample',
            })

    logger.info(
        f"Generated {len(negatives)} negative samples from "
        f"{len(negatives) // 2} non-interacting pairs"
    )
    return negatives


def fetch_pubchem_smiles(drug_name: str) -> Optional[str]:
    """Fetch canonical SMILES from PubChem by drug name."""
    try:
        encoded = urllib.parse.quote(drug_name)
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/property/CanonicalSMILES/JSON"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'DDI-Research/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                return props[0].get('CanonicalSMILES')
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
            TimeoutError, KeyError):
        pass
    return None


def load_pubchem_cache() -> Dict[str, Optional[str]]:
    """Load cached PubChem SMILES lookups."""
    if PUBCHEM_CACHE.exists():
        with open(PUBCHEM_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_pubchem_cache(cache: Dict[str, Optional[str]]):
    """Save PubChem SMILES cache."""
    with open(PUBCHEM_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def expand_from_ddi_sentences(
    drug_map: Dict[str, Dict],
    positive_pairs: Set[Tuple[str, str]],
    max_pubchem_lookups: int = 200,
) -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """
    Expand dataset using drug pairs from ddi_sentences.db.

    For drugs not in drug_map, attempts to resolve SMILES via PubChem.

    Args:
        drug_map: Existing drug ID -> info map (will be mutated with new drugs)
        positive_pairs: Existing positive pairs (will be mutated)
        max_pubchem_lookups: Maximum PubChem API calls to make
    """
    db_path = DATA_DIR / 'ddi_sentences.db'
    if not db_path.exists():
        logger.warning(f"ddi_sentences.db not found at {db_path}, skipping expansion")
        return [], set()

    # Build name -> SMILES mapping from existing drugs
    name_to_smiles = {}
    for did, info in drug_map.items():
        name_to_smiles[info['name'].lower()] = info['smiles']

    # Load PubChem cache
    pubchem_cache = load_pubchem_cache()
    pubchem_lookups = 0

    # Read unique drug pairs from sentences DB
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT drug1_normalized, drug2_normalized, interaction_type
        FROM ddi_sentences
        WHERE drug1_normalized != drug2_normalized
    """)
    rows = cursor.fetchall()
    conn.close()

    logger.info(f"Found {len(rows)} unique drug pairs in ddi_sentences.db")

    # Map interaction_type to severity class
    type_to_class = {
        'effect': 2,      # moderate
        'mechanism': 3,    # severe (pharmacokinetic)
        'advise': 1,       # minor
        'int': 2,          # moderate
    }

    new_positives = []
    new_pairs = set()

    for drug1_name, drug2_name, int_type in rows:
        d1_lower = drug1_name.lower().strip()
        d2_lower = drug2_name.lower().strip()

        # Skip self-interactions
        if d1_lower == d2_lower:
            continue

        # Resolve SMILES for drug1
        smiles1 = name_to_smiles.get(d1_lower)
        if smiles1 is None and d1_lower in pubchem_cache:
            smiles1 = pubchem_cache[d1_lower]
        if smiles1 is None and pubchem_lookups < max_pubchem_lookups:
            smiles1 = fetch_pubchem_smiles(drug1_name)
            pubchem_cache[d1_lower] = smiles1
            pubchem_lookups += 1
            if pubchem_lookups % 20 == 0:
                logger.info(f"PubChem lookups: {pubchem_lookups}/{max_pubchem_lookups}")
                save_pubchem_cache(pubchem_cache)
                time.sleep(0.5)  # Rate limiting

        # Resolve SMILES for drug2
        smiles2 = name_to_smiles.get(d2_lower)
        if smiles2 is None and d2_lower in pubchem_cache:
            smiles2 = pubchem_cache[d2_lower]
        if smiles2 is None and pubchem_lookups < max_pubchem_lookups:
            smiles2 = fetch_pubchem_smiles(drug2_name)
            pubchem_cache[d2_lower] = smiles2
            pubchem_lookups += 1
            if pubchem_lookups % 20 == 0:
                logger.info(f"PubChem lookups: {pubchem_lookups}/{max_pubchem_lookups}")
                save_pubchem_cache(pubchem_cache)
                time.sleep(0.5)

        # Skip if we can't get SMILES for both drugs
        if not smiles1 or not smiles2:
            continue

        # Update name->smiles map for future lookups
        if d1_lower not in name_to_smiles:
            name_to_smiles[d1_lower] = smiles1
        if d2_lower not in name_to_smiles:
            name_to_smiles[d2_lower] = smiles2

        # Skip if already in positive set (use SMILES as key since we 
        # may not have DrugBank IDs for expanded drugs)
        pair_key = tuple(sorted([smiles1, smiles2]))
        if pair_key in new_pairs:
            continue

        cls = type_to_class.get(int_type, 2)

        new_positives.append({
            'drug1_smiles': smiles1,
            'drug2_smiles': smiles2,
            'drug1_name': drug1_name,
            'drug2_name': drug2_name,
            'interaction_type': cls,
            'has_interaction': 1,
            'severity': {1: 'minor', 2: 'moderate', 3: 'severe'}.get(cls, 'moderate'),
            'source': 'ddi_sentences',
        })
        # Also add reverse
        new_positives.append({
            'drug1_smiles': smiles2,
            'drug2_smiles': smiles1,
            'drug1_name': drug2_name,
            'drug2_name': drug1_name,
            'interaction_type': cls,
            'has_interaction': 1,
            'severity': {1: 'minor', 2: 'moderate', 3: 'severe'}.get(cls, 'moderate'),
            'source': 'ddi_sentences',
        })
        new_pairs.add(pair_key)

    # Save final PubChem cache
    save_pubchem_cache(pubchem_cache)

    logger.info(
        f"Expanded dataset: {len(new_positives)} new positive samples "
        f"from {len(new_pairs)} unique pairs "
        f"({pubchem_lookups} PubChem lookups made)"
    )
    return new_positives, new_pairs


def split_dataset(
    samples: List[Dict],
    val_ratio: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train/val ensuring no SMILES pair leakage.

    Pairs are split at the unique-pair level (before symmetry doubling)
    so (A,B) never appears in both train and val.
    """
    random.seed(RANDOM_SEED)

    # Group by unique SMILES pair
    pair_groups = defaultdict(list)
    for s in samples:
        key = tuple(sorted([s['drug1_smiles'], s['drug2_smiles']]))
        pair_groups[key].append(s)

    # Split at pair level
    pair_keys = list(pair_groups.keys())
    random.shuffle(pair_keys)
    val_count = max(1, int(len(pair_keys) * val_ratio))
    val_keys = set(pair_keys[:val_count])

    train_samples = []
    val_samples = []
    for key, group in pair_groups.items():
        if key in val_keys:
            val_samples.extend(group)
        else:
            train_samples.extend(group)

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def summarize_dataset(samples: List[Dict], name: str):
    """Print summary statistics for a dataset split."""
    pos = sum(1 for s in samples if s['has_interaction'] == 1)
    neg = len(samples) - pos
    severities = defaultdict(int)
    for s in samples:
        severities[s['severity']] += 1

    logger.info(f"\n{name} Dataset Summary:")
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Positive (interacting): {pos}")
    logger.info(f"  Negative (non-interacting): {neg}")
    logger.info(f"  Ratio (neg:pos): {neg/max(pos,1):.1f}:1")
    logger.info(f"  Severity distribution: {dict(severities)}")

    unique_smiles = set()
    for s in samples:
        unique_smiles.add(s['drug1_smiles'])
        unique_smiles.add(s['drug2_smiles'])
    logger.info(f"  Unique drugs (by SMILES): {len(unique_smiles)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GNN training data")
    parser.add_argument(
        '--expand', action='store_true',
        help='Expand dataset using ddi_sentences.db + PubChem API'
    )
    parser.add_argument(
        '--max-pubchem', type=int, default=200,
        help='Maximum PubChem API lookups (default: 200)'
    )
    parser.add_argument(
        '--neg-ratio', type=float, default=2.0,
        help='Negative:Positive sample ratio (default: 2.0)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Validation split ratio (default: 0.2)'
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load curated drugs
    drug_map = load_drug_db()
    if not drug_map:
        logger.error("No drugs loaded. Ensure drug_db.json exists.")
        return

    # Step 2: Build positive samples from curated interactions
    positives, positive_pairs = build_curated_dataset(drug_map)

    # Step 3: Optional expansion from ddi_sentences.db + PubChem
    if args.expand:
        logger.info("\n--- Expanding dataset from ddi_sentences.db ---")
        expanded_positives, expanded_pairs = expand_from_ddi_sentences(
            drug_map, positive_pairs,
            max_pubchem_lookups=args.max_pubchem,
        )
        positives.extend(expanded_positives)

    # Step 4: Generate negative samples
    negatives = generate_negative_samples(
        drug_map, positive_pairs, ratio=args.neg_ratio
    )

    # Step 5: Combine and split
    all_samples = positives + negatives
    train_samples, val_samples = split_dataset(all_samples, args.val_ratio)

    # Step 6: Report statistics
    summarize_dataset(train_samples, "Train")
    summarize_dataset(val_samples, "Validation")

    # Step 7: Save
    train_path = OUTPUT_DIR / 'train.json'
    val_path = OUTPUT_DIR / 'val.json'

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2)

    logger.info(f"\nSaved training data to {OUTPUT_DIR}/")
    logger.info(f"  train.json: {len(train_samples)} samples")
    logger.info(f"  val.json: {len(val_samples)} samples")

    # Also save a metadata file
    meta = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'expanded': args.expand,
        'neg_ratio': args.neg_ratio,
        'val_ratio': args.val_ratio,
        'num_curated_interactions': len(KNOWN_INTERACTION_PAIRS),
        'num_drugs_with_smiles': len(drug_map),
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
