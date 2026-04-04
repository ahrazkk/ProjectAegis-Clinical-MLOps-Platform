"""
Export DDI data from Neo4j Aura for GNN training.

Pulls all drug-drug interactions where both drugs have SMILES,
generates balanced negative samples, and saves train/val splits.
"""
import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'web' / 'data' / 'gnn_training'
RANDOM_SEED = 42

# Aura connection
NEO4J_URI = os.getenv('NEO4J_URI', '')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

SEVERITY_MAP = {
    'minor': 'minor',
    'moderate': 'moderate',
    'major': 'severe',
    'severe': 'severe',
}
SEVERITY_CLASS = {'minor': 1, 'moderate': 2, 'severe': 3}


def export_from_aura():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    if not NEO4J_URI:
        raise ValueError("Missing Neo4j URI. Set NEO4J_URI environment variable.")
    if not NEO4J_PASSWORD:
        raise ValueError("Missing Neo4j password. Set NEO4J_PASSWORD environment variable.")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # ================================================================
    # Step 1: Pull all drugs with SMILES
    # ================================================================
    logger.info("Step 1: Pulling drugs from Aura...")
    with driver.session() as session:
        result = session.run('''
            MATCH (d:Drug)
            WHERE d.smiles IS NOT NULL AND d.smiles <> ""
            RETURN d.name AS name, d.smiles AS smiles, d.drugbank_id AS dbid,
                   d.category AS category, d.therapeutic_class AS tclass
        ''')
        drugs = {}
        for record in result:
            name = record['name']
            smiles = record['smiles']
            if name and smiles and len(smiles) > 3:
                drugs[name.lower()] = {
                    'name': name,
                    'smiles': smiles,
                    'drugbank_id': record['dbid'] or '',
                    'category': record['category'] or '',
                    'therapeutic_class': record['tclass'] or '',
                }

    logger.info(f"Loaded {len(drugs)} drugs with valid SMILES from Aura")

    # ================================================================
    # Step 2: Pull all interactions where both drugs have SMILES
    # ================================================================
    logger.info("Step 2: Pulling interactions...")
    with driver.session() as session:
        result = session.run('''
            MATCH (a:Drug)-[i:INTERACTS_WITH]->(b:Drug)
            WHERE a.smiles IS NOT NULL AND a.smiles <> ""
              AND b.smiles IS NOT NULL AND b.smiles <> ""
            RETURN a.name AS drug1, b.name AS drug2,
                   a.smiles AS smiles1, b.smiles AS smiles2,
                   i.severity AS severity, i.description AS description
        ''')

        positives = []
        positive_pairs = set()
        severity_counts = defaultdict(int)

        for record in result:
            d1 = record['drug1']
            d2 = record['drug2']
            s1 = record['smiles1']
            s2 = record['smiles2']
            raw_sev = (record['severity'] or 'moderate').lower()
            severity = SEVERITY_MAP.get(raw_sev, 'moderate')
            sev_class = SEVERITY_CLASS.get(severity, 2)

            # Deduplicate by sorted SMILES pair
            pair_key = tuple(sorted([s1, s2]))
            if pair_key in positive_pairs:
                continue
            positive_pairs.add(pair_key)
            severity_counts[severity] += 1

            # Both orderings for symmetric learning
            for sa, sb, na, nb in [(s1, s2, d1, d2), (s2, s1, d2, d1)]:
                positives.append({
                    'drug1_smiles': sa,
                    'drug2_smiles': sb,
                    'drug1_name': na,
                    'drug2_name': nb,
                    'interaction_type': sev_class,
                    'has_interaction': 1,
                    'severity': severity,
                    'source': 'aura_neo4j',
                })

    driver.close()

    logger.info(f"Extracted {len(positive_pairs)} unique interaction pairs "
                f"({len(positives)} samples with symmetry)")
    logger.info(f"Severity: {dict(severity_counts)}")

    # ================================================================
    # Step 3: Generate negative samples
    # ================================================================
    logger.info("Step 3: Generating negative samples...")

    # Collect all SMILES involved in positives
    drugs_in_positives = set()
    smiles_to_name = {}
    for s in positives:
        drugs_in_positives.add(s['drug1_smiles'])
        drugs_in_positives.add(s['drug2_smiles'])
        smiles_to_name[s['drug1_smiles']] = s['drug1_name']
        smiles_to_name[s['drug2_smiles']] = s['drug2_name']

    drug_smiles_list = sorted(drugs_in_positives)
    logger.info(f"Drug pool: {len(drug_smiles_list)} unique molecules")

    # Generate negative pairs (not in positive set)
    target_negatives = int(len(positive_pairs) * 1.0)  # 1:1 ratio
    candidate_negatives = []
    for i, s1 in enumerate(drug_smiles_list):
        for s2 in drug_smiles_list[i + 1:]:
            pk = tuple(sorted([s1, s2]))
            if pk not in positive_pairs:
                candidate_negatives.append((s1, s2))

    logger.info(f"Candidate negative pairs: {len(candidate_negatives)}")

    if len(candidate_negatives) > target_negatives:
        sampled_negs = random.sample(candidate_negatives, target_negatives)
    else:
        sampled_negs = candidate_negatives

    negatives = []
    for s1, s2 in sampled_negs:
        n1 = smiles_to_name.get(s1, 'unknown')
        n2 = smiles_to_name.get(s2, 'unknown')
        for sa, sb, na, nb in [(s1, s2, n1, n2), (s2, s1, n2, n1)]:
            negatives.append({
                'drug1_smiles': sa,
                'drug2_smiles': sb,
                'drug1_name': na,
                'drug2_name': nb,
                'interaction_type': 0,
                'has_interaction': 0,
                'severity': 'none',
                'source': 'negative_sample',
            })

    logger.info(f"Generated {len(negatives)} negative samples ({len(sampled_negs)} pairs)")

    # ================================================================
    # Step 4: Split at pair level (no leakage)
    # ================================================================
    logger.info("Step 4: Splitting train/val...")

    all_samples = positives + negatives

    pair_groups = defaultdict(list)
    for s in all_samples:
        key = tuple(sorted([s['drug1_smiles'], s['drug2_smiles']]))
        pair_groups[key].append(s)

    pair_keys = list(pair_groups.keys())
    random.shuffle(pair_keys)
    val_count = max(1, int(len(pair_keys) * 0.15))
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

    # ================================================================
    # Summary & Save
    # ================================================================
    for name, samples in [("Train", train_samples), ("Validation", val_samples)]:
        pos = sum(1 for s in samples if s['has_interaction'] == 1)
        neg = len(samples) - pos
        unique_drugs = set()
        for s in samples:
            unique_drugs.add(s['drug1_smiles'])
            unique_drugs.add(s['drug2_smiles'])

        sevs = defaultdict(int)
        for s in samples:
            sevs[s['severity']] += 1

        logger.info(f"\n{name}: {len(samples)} total ({pos} pos, {neg} neg, "
                    f"{len(unique_drugs)} unique drugs)")
        logger.info(f"  Severity: {dict(sevs)}")

    with open(OUTPUT_DIR / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_samples, f)
    with open(OUTPUT_DIR / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_samples, f)

    meta = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'positive_pairs': len(positive_pairs),
        'negative_pairs': len(sampled_negs),
        'unique_drugs': len(drugs_in_positives),
        'severity_distribution': dict(severity_counts),
        'source': 'neo4j_aura',
        'aura_uri': NEO4J_URI,
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # Also export the full drug list for the predictor's SMILES resolution
    drug_export = []
    for info in drugs.values():
        drug_export.append({
            'name': info['name'],
            'drugbank_id': info['drugbank_id'],
            'smiles': info['smiles'],
            'category': info['category'],
            'therapeutic_class': info['therapeutic_class'],
        })
    with open(PROJECT_ROOT / 'web' / 'data' / 'drug_db.json', 'w', encoding='utf-8') as f:
        json.dump(drug_export, f, indent=2)
    logger.info(f"\nUpdated drug_db.json: {len(drug_export)} drugs (was 73)")

    logger.info(f"\nSaved to {OUTPUT_DIR}/")
    logger.info(f"  train.json: {len(train_samples)} samples")
    logger.info(f"  val.json: {len(val_samples)} samples")
    prev = 490
    logger.info(f"\nDataset is {len(all_samples) / prev:.0f}x larger than previous ({prev} samples)")


if __name__ == '__main__':
    export_from_aura()
