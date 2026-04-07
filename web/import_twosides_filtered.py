"""
Import TWOSIDES significant interaction pairs into Neo4j Aura.
Only adds interactions where BOTH drugs already exist in the database.

Source: twosides_significant_pairs.json (pre-filtered: PRR > 2, A >= 3, min 3 conditions)
"""
import os
import json
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

PAIRS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'twosides_significant_pairs.json')

def severity_from_conditions(sig_conditions, max_prr):
    """Map signal strength to severity level."""
    if sig_conditions >= 100 and max_prr >= 50:
        return 'severe'
    elif sig_conditions >= 20 or max_prr >= 20:
        return 'moderate'
    else:
        return 'minor'


def main():
    print("=" * 60)
    print("TWOSIDES -> Neo4j Aura Import (Filtered)")
    print("=" * 60)

    # Connect to Neo4j
    print(f"\nConnecting to Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Get existing drugs from Neo4j
    print("Fetching existing drugs from Neo4j...")
    with driver.session() as session:
        result = session.run("MATCH (d:Drug) RETURN d.name AS name")
        existing_drugs = {}
        for record in result:
            name = record['name']
            if name:
                existing_drugs[name.lower().strip()] = name  # lowercase -> original case
    print(f"  Found {len(existing_drugs)} drugs in Neo4j")

    # Check current relationship count
    with driver.session() as session:
        result = session.run("MATCH ()-[r:INTERACTS_WITH]-() RETURN count(r) AS cnt")
        current_rels = result.single()['cnt']
    print(f"  Current relationships: {current_rels}")

    # Load pre-filtered TWOSIDES pairs
    print(f"\nLoading {PAIRS_FILE}...")
    with open(PAIRS_FILE) as f:
        all_pairs = json.load(f)
    print(f"  {len(all_pairs)} significant pairs loaded")

    # Filter to pairs where BOTH drugs exist in Neo4j
    records = []
    skipped_missing = 0
    skipped_self = 0
    seen_pairs = set()

    for pair in all_pairs:
        d1_lower = pair['drug1'].lower().strip()
        d2_lower = pair['drug2'].lower().strip()

        # Skip if either drug not in Neo4j
        if d1_lower not in existing_drugs or d2_lower not in existing_drugs:
            skipped_missing += 1
            continue

        # Skip self-loops
        if d1_lower == d2_lower:
            skipped_self += 1
            continue

        # Deduplicate (A,B) == (B,A)
        pair_key = tuple(sorted([d1_lower, d2_lower]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        severity = severity_from_conditions(pair['sig_conditions'], pair.get('max_prr', 0))

        records.append({
            'source': existing_drugs[d1_lower],  # Use Neo4j's original casing
            'target': existing_drugs[d2_lower],
            'severity': severity,
            'sig_conditions': pair['sig_conditions'],
            'max_prr': round(pair.get('max_prr', 0), 2),
            'desc': f"TWOSIDES signal: {pair['sig_conditions']} significant conditions, PRR up to {pair.get('max_prr', 0):.1f}",
        })

    print(f"\n  Matchable pairs (both drugs in Neo4j): {len(records)}")
    print(f"  Skipped (drug not in DB): {skipped_missing}")
    print(f"  Skipped (self-loops): {skipped_self}")

    if not records:
        print("No pairs to import!")
        driver.close()
        return

    # Severity breakdown
    sev_counts = {}
    for r in records:
        sev_counts[r['severity']] = sev_counts.get(r['severity'], 0) + 1
    print(f"  Severity breakdown: {sev_counts}")

    # Safety check: Aura free tier has 400K relationship limit
    new_total = current_rels + len(records) * 2  # *2 because undirected
    print(f"  Estimated new total relationships: ~{new_total:,}")
    if new_total > 380000:
        print(f"  WARNING: Approaching Aura free tier limit (400K). Trimming to top pairs by sig_conditions...")
        records.sort(key=lambda r: r['sig_conditions'], reverse=True)
        max_to_add = max(0, (380000 - current_rels) // 2)
        records = records[:max_to_add]
        print(f"  Trimmed to {len(records)} pairs")

    # Batch insert with MERGE (won't duplicate existing relationships)
    query = """
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

    BATCH_SIZE = 500
    total_inserted = 0
    start = time.time()

    print(f"\nInserting {len(records)} pairs in batches of {BATCH_SIZE}...")
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            result = session.run(query, batch=batch)
            summary = result.consume()
            created = summary.counters.relationships_created
            total_inserted += created
            elapsed = time.time() - start
            print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} pairs sent, "
                  f"{created} new relationships created ({elapsed:.1f}s elapsed)")

    # Final count
    with driver.session() as session:
        result = session.run("MATCH ()-[r:INTERACTS_WITH]-() RETURN count(r) AS cnt")
        final_rels = result.single()['cnt']

    driver.close()
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  New relationships created: {total_inserted}")
    print(f"  Relationships before: {current_rels:,}")
    print(f"  Relationships after: {final_rels:,}")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
