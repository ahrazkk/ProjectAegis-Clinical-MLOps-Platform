"""
Expanded GNN Training Data Builder

Massively expands the DDI training dataset by:
1. Loading all 3,445 DDI pairs from ddi_sentences.db (all are confirmed interactions)
2. Resolving SMILES for as many drugs as possible via PubChem API
3. Also incorporating curated DrugBank interactions + drug_db.json
4. Generating balanced negative samples from the expanded drug pool
5. Filtering drug class names (e.g., "nsaids", "ace inhibitors") that aren't molecules

This replaces the old prepare_gnn_data.py which only used 45 curated pairs.
"""

import json
import sqlite3
import random
import logging
import time
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
PUBCHEM_CACHE = DATA_DIR / 'pubchem_smiles_cache_v2.json'

RANDOM_SEED = 42

# Drug class names that are NOT individual molecules — skip these
DRUG_CLASS_NAMES = {
    'nsaids', 'ace inhibitors', 'aceinhibitors', 'antihistamines',
    'barbiturates', 'benzodiazepines', 'beta blockers', 'beta-blockers',
    'calcium channel blockers', 'cardiac glycosides', 'cephalosporins',
    'corticosteroids', 'diuretics', 'fluoroquinolones', 'macrolides',
    'maois', 'opioids', 'opiates', 'penicillins', 'phenothiazines',
    'quinolones', 'ssris', 'statins', 'sulfonamides', 'tetracyclines',
    'thiazide diuretics', 'thiazolidinediones', 'tricyclic antidepressants',
    'tricyclics', 'anticoagulants', 'antipsychotics', 'antidepressants',
    'antiepileptics', 'anticonvulsants', 'aminoglycosides',
    'proton pump inhibitors', 'h2 blockers', 'ace inhibitor',
    'beta blocker', 'calcium channel blocker', 'loop diuretics',
    'potassium-sparing diuretics', 'coumarin anticoagulants',
    'oral contraceptives', 'oral hypoglycemics', 'sulfonylureas',
    'fibrates', 'protease inhibitors', 'nrtis', 'nnrtis',
    '5ht1 agonists', '5ht1b1d agonists', '5ht3 antagonist class',
    'alkalinizing agents', 'acidifying agents', 'antacids',
    'neuromuscular blocking agents', 'skeletal muscle relaxants',
    'sympathomimetics', 'parasympathomimetics', 'anticholinergics',
    'acellular vaccines', 'vaccines', 'immunosuppressants',
    'antifungal agents', 'antifungals', 'azole antifungals',
}


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
                smiles = props[0].get('CanonicalSMILES')
                # Basic validation: reject very short or very long SMILES
                if smiles and 3 < len(smiles) < 500:
                    return smiles
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError,
            TimeoutError, KeyError, Exception):
        pass
    return None


def load_cache() -> Dict[str, Optional[str]]:
    if PUBCHEM_CACHE.exists():
        with open(PUBCHEM_CACHE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Optional[str]]):
    with open(PUBCHEM_CACHE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def is_drug_class(name: str) -> bool:
    """Check if a name refers to a drug class rather than a specific drug."""
    lower = name.lower().strip()
    if lower in DRUG_CLASS_NAMES:
        return True
    # Names with numbers at start (e.g., "125oh2d3") are often metabolites/codes
    if lower[0].isdigit() and not any(c.isalpha() for c in lower[1:4]):
        return True
    # Very short names are usually abbreviations for classes
    if len(lower) <= 2:
        return True
    return False


def resolve_all_smiles(drug_names: Set[str], existing_smiles: Dict[str, str]) -> Dict[str, str]:
    """
    Resolve SMILES for all given drug names using:
    1. Existing drug_db.json mappings
    2. PubChem API (with fresh cache)
    
    Returns: dict of lowercase drug name -> SMILES
    """
    name_to_smiles = dict(existing_smiles)  # Start with known mappings
    
    # Determine which drugs need PubChem lookup
    to_lookup = []
    for name in drug_names:
        lower = name.lower().strip()
        if lower in name_to_smiles:
            continue
        if is_drug_class(lower):
            continue
        to_lookup.append(lower)
    
    logger.info(f"Need PubChem lookup for {len(to_lookup)} drugs "
                f"({len(name_to_smiles)} already resolved, "
                f"{len(drug_names) - len(to_lookup) - len(name_to_smiles)} skipped as classes)")
    
    # Load cache (fresh v2 cache)
    cache = load_cache()
    lookups_needed = [n for n in to_lookup if n not in cache]
    already_cached = [n for n in to_lookup if n in cache]
    
    # Apply cached results
    for name in already_cached:
        if cache[name]:
            name_to_smiles[name] = cache[name]
    
    logger.info(f"Cache hit: {len(already_cached)}, API lookups needed: {len(lookups_needed)}")
    
    # Batch PubChem lookups with rate limiting
    resolved_count = 0
    failed_count = 0
    for i, name in enumerate(lookups_needed):
        smiles = fetch_pubchem_smiles(name)
        cache[name] = smiles
        
        if smiles:
            name_to_smiles[name] = smiles
            resolved_count += 1
        else:
            failed_count += 1
        
        # Rate limiting: PubChem allows 5 req/sec, we'll do 3/sec to be safe
        if (i + 1) % 3 == 0:
            time.sleep(1.0)
        
        # Progress update and cache save every 50
        if (i + 1) % 50 == 0:
            save_cache(cache)
            logger.info(f"  Progress: {i+1}/{len(lookups_needed)} "
                       f"(resolved: {resolved_count}, failed: {failed_count})")
    
    save_cache(cache)
    logger.info(f"PubChem resolution complete: {resolved_count} resolved, {failed_count} failed")
    logger.info(f"Total drugs with SMILES: {len(name_to_smiles)}")
    
    return name_to_smiles


def build_dataset():
    """Build the expanded training dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Phase 1: Collect all known drug SMILES
    # ================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting drug SMILES")
    logger.info("=" * 60)
    
    # Load existing drug_db.json
    db_path = DATA_DIR / 'drug_db.json'
    existing_smiles = {}  # lowercase name -> SMILES
    if db_path.exists():
        with open(db_path) as f:
            for drug in json.load(f):
                name = drug.get('name', '').lower().strip()
                smiles = drug.get('smiles', '')
                if name and smiles:
                    existing_smiles[name] = smiles
    logger.info(f"Loaded {len(existing_smiles)} drugs from drug_db.json")
    
    # Get all unique drug names from ddi_sentences.db
    db_path = DATA_DIR / 'ddi_sentences.db'
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT drug1_normalized FROM ddi_sentences
        UNION
        SELECT DISTINCT drug2_normalized FROM ddi_sentences
    """)
    all_drug_names = set(r[0] for r in cursor.fetchall() if r[0])
    logger.info(f"Found {len(all_drug_names)} unique drug names in ddi_sentences.db")
    
    # Resolve SMILES for all drugs
    name_to_smiles = resolve_all_smiles(all_drug_names, existing_smiles)
    
    # ================================================================
    # Phase 2: Build positive samples from ddi_sentences.db
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Building positive samples from DDI sentences")
    logger.info("=" * 60)
    
    type_to_severity = {
        'mechanism': 'severe',
        'effect': 'moderate',
        'advise': 'minor',
        'int': 'moderate',
    }
    
    cursor.execute("""
        SELECT drug1_normalized, drug2_normalized, interaction_type, COUNT(*) as evidence_count
        FROM ddi_sentences
        WHERE drug1_normalized != drug2_normalized
        GROUP BY drug1_normalized, drug2_normalized, interaction_type
        ORDER BY evidence_count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    positives = []
    positive_smiles_pairs = set()  # Track by SMILES pair to avoid duplicates
    skipped_no_smiles = 0
    skipped_class = 0
    
    for drug1_name, drug2_name, int_type, evidence_count in rows:
        d1 = drug1_name.lower().strip()
        d2 = drug2_name.lower().strip()
        
        if d1 == d2:
            continue
        if is_drug_class(d1) or is_drug_class(d2):
            skipped_class += 1
            continue
        
        smiles1 = name_to_smiles.get(d1)
        smiles2 = name_to_smiles.get(d2)
        
        if not smiles1 or not smiles2:
            skipped_no_smiles += 1
            continue
        
        # Deduplicate by sorted SMILES pair
        pair_key = tuple(sorted([smiles1, smiles2]))
        if pair_key in positive_smiles_pairs:
            continue
        positive_smiles_pairs.add(pair_key)
        
        severity = type_to_severity.get(int_type, 'moderate')
        severity_class = {'minor': 1, 'moderate': 2, 'severe': 3}.get(severity, 2)
        
        # Add both orderings for symmetric learning
        for s1, s2, n1, n2 in [
            (smiles1, smiles2, drug1_name, drug2_name),
            (smiles2, smiles1, drug2_name, drug1_name),
        ]:
            positives.append({
                'drug1_smiles': s1,
                'drug2_smiles': s2,
                'drug1_name': n1,
                'drug2_name': n2,
                'interaction_type': severity_class,
                'has_interaction': 1,
                'severity': severity,
                'source': 'ddi_corpus',
                'evidence_count': evidence_count,
            })
    
    logger.info(f"Built {len(positives)} positive samples from "
                f"{len(positive_smiles_pairs)} unique DDI pairs")
    logger.info(f"Skipped: {skipped_no_smiles} (no SMILES), {skipped_class} (drug classes)")
    
    # ================================================================
    # Phase 3: Generate negative samples
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Generating negative samples")
    logger.info("=" * 60)
    
    # Use all drugs that appeared in positive samples
    drugs_in_positives = set()
    for s in positives:
        drugs_in_positives.add(s['drug1_smiles'])
        drugs_in_positives.add(s['drug2_smiles'])
    
    drug_smiles_list = sorted(drugs_in_positives)
    logger.info(f"Drug pool for negatives: {len(drug_smiles_list)} unique SMILES")
    
    # Build reverse mapping for names
    smiles_to_name = {}
    for name, smiles in name_to_smiles.items():
        if smiles in drugs_in_positives:
            smiles_to_name[smiles] = name
    
    # Generate negatives: pairs NOT in positive set
    random.seed(RANDOM_SEED)
    num_positive_pairs = len(positive_smiles_pairs)
    target_negatives = int(num_positive_pairs * 1.5)  # 1.5:1 ratio
    
    candidate_negatives = []
    for i, s1 in enumerate(drug_smiles_list):
        for s2 in drug_smiles_list[i+1:]:
            pair_key = tuple(sorted([s1, s2]))
            if pair_key not in positive_smiles_pairs:
                candidate_negatives.append((s1, s2))
    
    logger.info(f"Candidate negative pairs: {len(candidate_negatives)}")
    
    if len(candidate_negatives) > target_negatives:
        sampled_negatives = random.sample(candidate_negatives, target_negatives)
    else:
        sampled_negatives = candidate_negatives
    
    negatives = []
    for s1, s2 in sampled_negatives:
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
                'evidence_count': 0,
            })
    
    logger.info(f"Generated {len(negatives)} negative samples from "
                f"{len(sampled_negatives)} pairs")
    
    # ================================================================
    # Phase 4: Split and save
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Splitting and saving dataset")
    logger.info("=" * 60)
    
    all_samples = positives + negatives
    random.seed(RANDOM_SEED)
    
    # Split at pair level to prevent leakage
    pair_groups = defaultdict(list)
    for s in all_samples:
        key = tuple(sorted([s['drug1_smiles'], s['drug2_smiles']]))
        pair_groups[key].append(s)
    
    pair_keys = list(pair_groups.keys())
    random.shuffle(pair_keys)
    val_count = max(1, int(len(pair_keys) * 0.15))  # 15% validation
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
    
    # Summary
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
        
        logger.info(f"\n{name}: {len(samples)} total ({pos} pos, {neg} neg)")
        logger.info(f"  Unique drugs: {len(unique_drugs)}")
        logger.info(f"  Severity: {dict(sevs)}")
    
    # Save
    with open(OUTPUT_DIR / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_samples, f)
    with open(OUTPUT_DIR / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_samples, f)
    
    meta = {
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'positive_pairs': len(positive_smiles_pairs),
        'negative_pairs': len(sampled_negatives),
        'unique_drugs': len(drugs_in_positives),
        'drugs_resolved_from_pubchem': len(name_to_smiles) - len(existing_smiles),
        'skipped_no_smiles': skipped_no_smiles,
        'skipped_drug_classes': skipped_class,
    }
    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"\nSaved to {OUTPUT_DIR}/")
    logger.info(f"  train.json: {len(train_samples)} samples")
    logger.info(f"  val.json: {len(val_samples)} samples")
    logger.info(f"\nDataset is {len(all_samples)/490:.1f}x larger than previous (490 samples)")


if __name__ == '__main__':
    build_dataset()
