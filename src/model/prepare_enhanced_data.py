"""
Enhanced GNN Training Data Preparation
=======================================
Combines ALL available data sources into a single large training set:
  1. Existing Neo4j pairs (from train.json + val.json)
  2. DDI Corpus (3,445 pairs from PubMed literature)
  3. TWOSIDES (154K+ significant pairs from FDA adverse event reports)

Also generates HARD negatives using Tanimoto similarity instead of random.

Output: web/data/gnn_training_enhanced/{train,val,test}.json
"""

import json
import sqlite3
import random
import sys
from pathlib import Path
from collections import defaultdict

# Optional: RDKit for hard negatives
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. Using random negatives instead of hard negatives.")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'web' / 'data'
OUTPUT_DIR = DATA_DIR / 'gnn_training_enhanced'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_DB_PATH = DATA_DIR / 'drug_db.json'
EXISTING_TRAIN = DATA_DIR / 'gnn_training' / 'train.json'
EXISTING_VAL = DATA_DIR / 'gnn_training' / 'val.json'
DDI_CORPUS_DB = DATA_DIR / 'ddi_sentences.db'
TWOSIDES_PATH = DATA_DIR / 'twosides_significant_pairs.json'


def load_drug_db():
    """Load drug database and build name->SMILES lookup."""
    with open(DRUG_DB_PATH) as f:
        drugs = json.load(f)

    name_to_smiles = {}
    for drug in drugs:
        name = drug['name'].strip().lower()
        smiles = drug.get('smiles', '')
        if smiles:
            name_to_smiles[name] = smiles
            # Also add without common suffixes
            for suffix in [' hydrochloride', ' sodium', ' potassium', ' sulfate',
                           ' citrate', ' acetate', ' maleate', ' tartrate',
                           ' mesylate', ' besylate', ' fumarate', ' succinate']:
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    name_to_smiles[base] = smiles

    print(f"Drug DB: {len(drugs)} drugs, {len(name_to_smiles)} name mappings")
    return name_to_smiles, drugs


def load_existing_pairs():
    """Load existing Neo4j training pairs."""
    pairs = []
    for path in [EXISTING_TRAIN, EXISTING_VAL]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                for item in data:
                    if item.get('has_interaction'):
                        pairs.append({
                            'drug1_smiles': item['drug1_smiles'],
                            'drug2_smiles': item['drug2_smiles'],
                            'drug1_name': item['drug1_name'],
                            'drug2_name': item['drug2_name'],
                            'severity': item.get('severity', 'moderate'),
                            'source': 'neo4j',
                        })
    print(f"Existing Neo4j positive pairs: {len(pairs)}")
    return pairs


def load_ddi_corpus(name_to_smiles):
    """Load DDI Corpus pairs and match to SMILES."""
    if not DDI_CORPUS_DB.exists():
        print("DDI Corpus DB not found, skipping.")
        return []

    conn = sqlite3.connect(str(DDI_CORPUS_DB))
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT drug1_normalized, drug2_normalized, interaction_type
        FROM ddi_sentences
        WHERE confidence >= 0.8
    """)
    rows = cur.fetchall()
    conn.close()

    # Map interaction_type to severity
    type_to_severity = {
        'effect': 'moderate',
        'mechanism': 'moderate',
        'advise': 'moderate',
        'int': 'minor',
    }

    pairs = []
    matched = 0
    unmatched_drugs = set()

    for drug1_norm, drug2_norm, int_type in rows:
        d1 = drug1_norm.strip().lower()
        d2 = drug2_norm.strip().lower()
        s1 = name_to_smiles.get(d1)
        s2 = name_to_smiles.get(d2)

        if s1 and s2 and s1 != s2:
            matched += 1
            pairs.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': drug1_norm,
                'drug2_name': drug2_norm,
                'severity': type_to_severity.get(int_type, 'moderate'),
                'source': 'ddi_corpus',
            })
        else:
            if not s1: unmatched_drugs.add(d1)
            if not s2: unmatched_drugs.add(d2)

    print(f"DDI Corpus: {len(rows)} unique pairs, {matched} matched to SMILES")
    print(f"  Unmatched drugs: {len(unmatched_drugs)} (e.g., {list(unmatched_drugs)[:5]})")
    return pairs


def load_twosides(name_to_smiles, max_pairs=20000):
    """Load TWOSIDES pairs and match to SMILES.

    We cap at max_pairs to avoid overwhelming the dataset —
    TWOSIDES has 154K pairs but many are weak signals.
    We take the top pairs by number of significant conditions.
    """
    if not TWOSIDES_PATH.exists():
        print("TWOSIDES pairs not found, skipping.")
        return []

    with open(TWOSIDES_PATH) as f:
        tw_pairs = json.load(f)

    # Already sorted by sig_conditions descending
    pairs = []
    matched = 0
    unmatched_drugs = set()

    for item in tw_pairs:
        if matched >= max_pairs:
            break
        d1 = item['drug1'].strip().lower()
        d2 = item['drug2'].strip().lower()
        s1 = name_to_smiles.get(d1)
        s2 = name_to_smiles.get(d2)

        if s1 and s2 and s1 != s2:
            matched += 1
            # Severity based on number of significant adverse conditions
            sig = item['sig_conditions']
            if sig >= 500:
                severity = 'severe'
            elif sig >= 50:
                severity = 'moderate'
            else:
                severity = 'minor'

            pairs.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': item['drug1'],
                'drug2_name': item['drug2'],
                'severity': severity,
                'source': 'twosides',
            })
        else:
            if not s1: unmatched_drugs.add(d1)
            if not s2: unmatched_drugs.add(d2)

    print(f"TWOSIDES: {len(tw_pairs)} total pairs, {matched} matched to SMILES (cap={max_pairs})")
    print(f"  Unmatched drugs: {len(unmatched_drugs)}")
    return pairs


def deduplicate_pairs(all_pairs):
    """Remove duplicate drug pairs (keeping first occurrence)."""
    seen = set()
    unique = []
    for pair in all_pairs:
        key = tuple(sorted([pair['drug1_smiles'], pair['drug2_smiles']]))
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    print(f"Deduplicated: {len(all_pairs)} -> {len(unique)} unique pairs")
    return unique


def generate_negatives(positive_pairs, all_smiles_list, ratio=1.0):
    """Generate negative (non-interacting) pairs.

    If RDKit available: hard negatives using Tanimoto similarity.
    Otherwise: random negatives.
    """
    positive_keys = set()
    for p in positive_pairs:
        key = tuple(sorted([p['drug1_smiles'], p['drug2_smiles']]))
        positive_keys.add(key)

    n_negatives = int(len(positive_pairs) * ratio)
    negatives = []

    if HAS_RDKIT and len(all_smiles_list) > 10:
        print(f"Generating {n_negatives} hard negatives using Tanimoto similarity...")
        # Compute Morgan fingerprints for all drugs
        fps = {}
        valid_smiles = []
        for smi in all_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                valid_smiles.append(smi)

        # For each positive pair, find a hard negative:
        # drug with similar structure but no known interaction
        attempts = 0
        max_attempts = n_negatives * 20

        while len(negatives) < n_negatives and attempts < max_attempts:
            attempts += 1
            # Pick a random drug
            s1 = random.choice(valid_smiles)
            if s1 not in fps:
                continue

            # Find a similar drug (Tanimoto > 0.3) that doesn't interact with it
            candidates = []
            sample_pool = random.sample(valid_smiles, min(100, len(valid_smiles)))
            for s2 in sample_pool:
                if s2 == s1 or s2 not in fps:
                    continue
                key = tuple(sorted([s1, s2]))
                if key in positive_keys:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[s1], fps[s2])
                if sim > 0.2:  # Moderately similar = harder negative
                    candidates.append((s2, sim))

            if candidates:
                # Pick the most similar non-interacting drug
                candidates.sort(key=lambda x: -x[1])
                s2, sim = candidates[0]
                key = tuple(sorted([s1, s2]))
                if key not in positive_keys:
                    positive_keys.add(key)  # prevent duplicates
                    # Find names
                    negatives.append({
                        'drug1_smiles': s1,
                        'drug2_smiles': s2,
                        'drug1_name': '',
                        'drug2_name': '',
                        'severity': 'none',
                        'source': 'hard_negative',
                        'has_interaction': 0,
                        'interaction_type': 0,
                    })

        if len(negatives) < n_negatives:
            print(f"  Hard negatives generated: {len(negatives)}, filling rest with random...")
    else:
        print(f"Generating {n_negatives} random negatives...")

    # Fill remaining with random negatives
    remaining = n_negatives - len(negatives)
    attempts = 0
    while len(negatives) < n_negatives and attempts < remaining * 10:
        attempts += 1
        s1 = random.choice(all_smiles_list)
        s2 = random.choice(all_smiles_list)
        if s1 == s2:
            continue
        key = tuple(sorted([s1, s2]))
        if key not in positive_keys:
            positive_keys.add(key)
            negatives.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': '',
                'drug2_name': '',
                'severity': 'none',
                'source': 'random_negative',
                'has_interaction': 0,
                'interaction_type': 0,
            })

    print(f"Total negatives generated: {len(negatives)}")
    return negatives


def split_data(all_samples, val_ratio=0.1, test_ratio=0.1):
    """Split at pair level to prevent leakage."""
    random.shuffle(all_samples)
    n = len(all_samples)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = all_samples[:n_test]
    val = all_samples[n_test:n_test + n_val]
    train = all_samples[n_test + n_val:]

    return train, val, test


def main():
    print("=" * 60)
    print("ENHANCED GNN DATA PREPARATION")
    print("=" * 60)

    # 1. Load drug database
    name_to_smiles, drugs = load_drug_db()

    # 2. Collect positive pairs from all sources
    print("\n--- Loading positive pairs ---")
    neo4j_pairs = load_existing_pairs()
    corpus_pairs = load_ddi_corpus(name_to_smiles)
    twosides_pairs = load_twosides(name_to_smiles, max_pairs=15000)

    all_positive = neo4j_pairs + corpus_pairs + twosides_pairs
    print(f"\nTotal positive pairs before dedup: {len(all_positive)}")
    print(f"  Neo4j: {len(neo4j_pairs)}")
    print(f"  DDI Corpus: {len(corpus_pairs)}")
    print(f"  TWOSIDES: {len(twosides_pairs)}")

    all_positive = deduplicate_pairs(all_positive)

    # Add required fields to positives
    severity_to_type = {'none': 0, 'minor': 1, 'moderate': 2, 'severe': 3}
    for p in all_positive:
        p['has_interaction'] = 1
        p['interaction_type'] = severity_to_type.get(p.get('severity', 'moderate'), 2)

    # 3. Collect all unique SMILES for negative generation
    all_smiles = set()
    for p in all_positive:
        all_smiles.add(p['drug1_smiles'])
        all_smiles.add(p['drug2_smiles'])
    for d in drugs:
        if d.get('smiles'):
            all_smiles.add(d['smiles'])
    all_smiles_list = list(all_smiles)
    print(f"\nUnique drugs (SMILES): {len(all_smiles_list)}")

    # 4. Generate negatives (1:1 ratio)
    print("\n--- Generating negatives ---")
    negatives = generate_negatives(all_positive, all_smiles_list, ratio=1.0)

    # 5. Combine and split
    all_samples = all_positive + negatives
    random.shuffle(all_samples)

    print(f"\n--- Final dataset ---")
    print(f"Total samples: {len(all_samples)}")
    print(f"  Positive: {len(all_positive)}")
    print(f"  Negative: {len(negatives)}")

    train, val, test = split_data(all_samples, val_ratio=0.1, test_ratio=0.1)
    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # Source breakdown
    from collections import Counter
    for name, split in [('train', train), ('val', val), ('test', test)]:
        sources = Counter(s['source'] for s in split)
        pos = sum(1 for s in split if s['has_interaction'])
        neg = sum(1 for s in split if not s['has_interaction'])
        print(f"  {name}: {len(split)} samples (pos={pos}, neg={neg}), sources={dict(sources)}")

    # 6. Save
    for name, data in [('train', train), ('val', val), ('test', test)]:
        path = OUTPUT_DIR / f'{name}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=0)
        print(f"Saved {path} ({len(data)} samples)")

    # Save metadata
    sev_dist = Counter(s['severity'] for s in all_samples if s['has_interaction'])
    metadata = {
        'total_samples': len(all_samples),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'positive_pairs': len(all_positive),
        'negative_pairs': len(negatives),
        'unique_drugs': len(all_smiles_list),
        'severity_distribution': dict(sev_dist),
        'sources': {
            'neo4j': len(neo4j_pairs),
            'ddi_corpus': len(corpus_pairs),
            'twosides': len(twosides_pairs),
        },
        'negative_strategy': 'hard_negative' if HAS_RDKIT else 'random',
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE. Enhanced dataset saved to {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
