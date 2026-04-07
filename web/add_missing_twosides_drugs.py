"""
Step 1: Fetch SMILES for TWOSIDES drugs missing from Neo4j via PubChem API.
Step 2: Create :Drug nodes in Neo4j for found drugs.
Step 3: Re-run TWOSIDES interaction import to pick up newly available pairs.

PubChem PUG REST API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
Free, no API key needed, but rate-limited to ~5 requests/second.
"""
import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pathlib import Path

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRS_FILE = SCRIPT_DIR / 'data' / 'twosides_significant_pairs.json'
CACHE_FILE = SCRIPT_DIR / 'data' / 'pubchem_smiles_cache.json'


def pubchem_lookup(drug_name, retries=2):
    """Look up SMILES from PubChem by drug name. Returns (smiles, cid) or (None, None)."""
    encoded = urllib.parse.quote(drug_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ProjectAegis/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    smiles = props[0].get('CanonicalSMILES') or props[0].get('ConnectivitySMILES')
                    return smiles, props[0].get('CID')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None  # Drug not found
            if e.code == 503 and attempt < retries:
                time.sleep(2)  # Rate limited, wait
                continue
            return None, None
        except Exception:
            if attempt < retries:
                time.sleep(1)
                continue
            return None, None

    return None, None


def categorize_drug(name):
    """Simple keyword-based category assignment."""
    nl = name.lower()
    if any(k in nl for k in ['cillin', 'mycin', 'cycline', 'floxacin', 'sulfa']):
        return 'Antibiotic'
    if any(k in nl for k in ['vir', 'abavir', 'ovir']):
        return 'Antiviral'
    if any(k in nl for k in ['azole', 'fungin']):
        return 'Antifungal'
    if any(k in nl for k in ['olol', 'alol']):
        return 'Beta Blocker'
    if any(k in nl for k in ['pril', 'sartan']):
        return 'Cardiovascular'
    if any(k in nl for k in ['statin', 'vastatin']):
        return 'Statin'
    if any(k in nl for k in ['prazole', 'tidine']):
        return 'GI Agent'
    if any(k in nl for k in ['mab', 'umab', 'zumab']):
        return 'Monoclonal Antibody'
    if any(k in nl for k in ['tinib', 'nib']):
        return 'Kinase Inhibitor'
    if any(k in nl for k in ['sone', 'olone', 'nide']):
        return 'Corticosteroid'
    if any(k in nl for k in ['pine', 'pam', 'lam', 'zepam']):
        return 'CNS Agent'
    if any(k in nl for k in ['profen', 'oxicam', 'fenac']):
        return 'NSAID'
    if any(k in nl for k in ['platin', 'rubicin', 'taxel']):
        return 'Antineoplastic'
    return 'Unknown'


def main():
    print("=" * 60)
    print("Add Missing TWOSIDES Drugs to Neo4j")
    print("=" * 60)

    # Connect to Neo4j
    print(f"\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Get existing drugs
    with driver.session() as session:
        result = session.run("MATCH (d:Drug) RETURN d.name AS name")
        existing = set()
        for r in result:
            if r['name']:
                existing.add(r['name'].lower().strip())
    print(f"  {len(existing)} drugs in Neo4j")

    # Find missing TWOSIDES drugs
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)

    twosides_drugs = set()
    for p in pairs:
        twosides_drugs.add(p['drug1'].strip())
        twosides_drugs.add(p['drug2'].strip())

    missing = []
    for d in twosides_drugs:
        if d.lower().strip() not in existing:
            missing.append(d)
    missing.sort()
    print(f"  {len(missing)} TWOSIDES drugs missing from Neo4j")

    # Load cache if exists
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached PubChem lookups")

    # Fetch SMILES from PubChem
    print(f"\nLooking up SMILES from PubChem for {len(missing)} drugs...")
    print("  (Rate limited to ~5/sec, this will take ~4 minutes)")

    found = []
    not_found = []
    errors = 0
    request_count = 0

    for i, drug_name in enumerate(missing):
        # Check cache first
        cache_key = drug_name.lower().strip()
        if cache_key in cache:
            entry = cache[cache_key]
            if entry and isinstance(entry, dict) and entry.get('smiles'):
                found.append({'name': drug_name, 'smiles': entry['smiles'],
                              'cid': entry.get('cid'), 'category': categorize_drug(drug_name)})
            else:
                not_found.append(drug_name)
            continue

        # PubChem lookup
        smiles, cid = pubchem_lookup(drug_name)
        request_count += 1

        if smiles:
            found.append({'name': drug_name, 'smiles': smiles, 'cid': cid,
                          'category': categorize_drug(drug_name)})
            cache[cache_key] = {'smiles': smiles, 'cid': cid}
        else:
            not_found.append(drug_name)
            cache[cache_key] = {'smiles': None, 'cid': None}

        # Rate limit: ~5 requests/sec
        if request_count % 5 == 0:
            time.sleep(1.1)

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(missing)} — found: {len(found)}, not found: {len(not_found)}")

        # Save cache periodically
        if (i + 1) % 200 == 0:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f)

    # Final cache save
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

    print(f"\n  PubChem Results:")
    print(f"    Found SMILES: {len(found)}")
    print(f"    Not found: {len(not_found)}")

    if not found:
        print("No new drugs to add!")
        driver.close()
        return

    # Insert new Drug nodes into Neo4j
    print(f"\nCreating {len(found)} new :Drug nodes in Neo4j...")

    query = """
    UNWIND $batch AS drug
    MERGE (d:Drug {name: drug.name})
    ON CREATE SET
        d.smiles = drug.smiles,
        d.category = drug.category,
        d.source = 'twosides_pubchem',
        d.pubchem_cid = drug.cid,
        d.updated_at = datetime()
    """

    BATCH_SIZE = 200
    total_created = 0
    with driver.session() as session:
        for i in range(0, len(found), BATCH_SIZE):
            batch = found[i:i + BATCH_SIZE]
            result = session.run(query, batch=batch)
            summary = result.consume()
            created = summary.counters.nodes_created
            total_created += created
            print(f"  Batch {i // BATCH_SIZE + 1}: {created} nodes created")

    print(f"  Total new Drug nodes: {total_created}")

    # Now re-run TWOSIDES interaction import
    print(f"\nRe-importing TWOSIDES interactions with new drugs...")

    # Refresh existing drugs
    with driver.session() as session:
        result = session.run("MATCH (d:Drug) RETURN d.name AS name")
        existing_drugs = {}
        for r in result:
            if r['name']:
                existing_drugs[r['name'].lower().strip()] = r['name']

    print(f"  Neo4j now has {len(existing_drugs)} drugs")

    # Check current relationships
    with driver.session() as session:
        result = session.run("MATCH ()-[r:INTERACTS_WITH]-() RETURN count(r) AS cnt")
        rels_before = result.single()['cnt']

    # Build interaction records
    records = []
    seen = set()
    for p in pairs:
        d1 = p['drug1'].lower().strip()
        d2 = p['drug2'].lower().strip()
        if d1 not in existing_drugs or d2 not in existing_drugs:
            continue
        if d1 == d2:
            continue
        pair_key = tuple(sorted([d1, d2]))
        if pair_key in seen:
            continue
        seen.add(pair_key)

        sc = p['sig_conditions']
        mp = p.get('max_prr', 0)
        if sc >= 100 and mp >= 50:
            sev = 'severe'
        elif sc >= 20 or mp >= 20:
            sev = 'moderate'
        else:
            sev = 'minor'

        records.append({
            'source': existing_drugs[d1],
            'target': existing_drugs[d2],
            'severity': sev,
            'sig_conditions': sc,
            'max_prr': round(mp, 2),
            'desc': f"TWOSIDES: {sc} conditions, PRR {mp:.1f}",
        })

    print(f"  Matchable pairs now: {len(records)}")

    # Insert
    int_query = """
    UNWIND $batch AS interaction
    MATCH (source:Drug {name: interaction.source})
    MATCH (target:Drug {name: interaction.target})
    MERGE (source)-[r:INTERACTS_WITH]-(target)
    ON CREATE SET
        r.severity = interaction.severity,
        r.description = interaction.desc,
        r.source = 'twosides',
        r.evidence_level = 'twosides',
        r.sig_conditions = interaction.sig_conditions,
        r.max_prr = interaction.max_prr,
        r.added_at = datetime()
    """

    total_new_rels = 0
    with driver.session() as session:
        for i in range(0, len(records), 500):
            batch = records[i:i + 500]
            result = session.run(int_query, batch=batch)
            summary = result.consume()
            created = summary.counters.relationships_created
            total_new_rels += created
            if (i // 500 + 1) % 20 == 0:
                print(f"  Batch {i // 500 + 1}: {total_new_rels} new relationships so far...")

    # Final count
    with driver.session() as session:
        result = session.run("MATCH ()-[r:INTERACTS_WITH]-() RETURN count(r) AS cnt")
        rels_after = result.single()['cnt']
        result = session.run("MATCH (d:Drug) RETURN count(d) AS cnt")
        drugs_after = result.single()['cnt']

    driver.close()

    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"  New Drug nodes created: {total_created}")
    print(f"  Total drugs now: {drugs_after}")
    print(f"  New relationships created: {total_new_rels}")
    print(f"  Relationships before: {rels_before:,}")
    print(f"  Relationships after: {rels_after:,}")
    print(f"  Drugs not found in PubChem: {len(not_found)}")

    # Save not-found list for reference
    nf_path = SCRIPT_DIR / 'data' / 'twosides_drugs_not_in_pubchem.json'
    with open(nf_path, 'w') as f:
        json.dump(sorted(not_found), f, indent=2)
    print(f"  Not-found list saved: {nf_path}")


if __name__ == '__main__':
    main()
