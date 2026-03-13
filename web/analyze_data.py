"""Analyze all available DDI data sources."""
import sqlite3
import json
import os

print("=" * 60)
print("DDI SENTENCES DATABASE")
print("=" * 60)

conn = sqlite3.connect('data/ddi_sentences.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print(f"Tables: {tables}")

for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
    count = cursor.fetchone()[0]
    print(f"  {table}: {count} rows")
    cursor.execute(f"PRAGMA table_info([{table}])")
    cols = cursor.fetchall()
    print(f"  Columns: {[c[1] for c in cols]}")

cursor.execute("SELECT COUNT(DISTINCT drug1_normalized || '-' || drug2_normalized) FROM ddi_sentences")
pairs = cursor.fetchone()[0]
print(f"\nUnique directed drug pairs: {pairs}")

# Get all unique drug names
cursor.execute("SELECT DISTINCT drug1_normalized FROM ddi_sentences UNION SELECT DISTINCT drug2_normalized FROM ddi_sentences")
all_drugs = set(r[0] for r in cursor.fetchall() if r[0])
print(f"Unique drug names: {len(all_drugs)}")

# Interaction type distribution
cursor.execute("SELECT interaction_type, COUNT(*) FROM ddi_sentences GROUP BY interaction_type ORDER BY COUNT(*) DESC")
print("\nInteraction types:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Source distribution
cursor.execute("SELECT source, COUNT(*) FROM ddi_sentences GROUP BY source ORDER BY COUNT(*) DESC")
print("\nSources:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Get unique pairs with interaction types (not just sentences)
cursor.execute("""
    SELECT drug1_normalized, drug2_normalized, interaction_type, COUNT(*) as cnt
    FROM ddi_sentences 
    WHERE interaction_type != 'false' AND interaction_type IS NOT NULL AND interaction_type != ''
    GROUP BY drug1_normalized, drug2_normalized
    ORDER BY cnt DESC
    LIMIT 20
""")
print("\nTop 20 interacting pairs:")
for row in cursor.fetchall():
    print(f"  {row[0]} + {row[1]}: type={row[2]}, sentences={row[3]}")

# Count pairs with confirmed interactions (not 'false')
cursor.execute("""
    SELECT COUNT(DISTINCT drug1_normalized || '-' || drug2_normalized)
    FROM ddi_sentences 
    WHERE interaction_type != 'false' AND interaction_type IS NOT NULL AND interaction_type != ''
""")
positive_pairs = cursor.fetchone()[0]
print(f"\nPositive interaction pairs: {positive_pairs}")

cursor.execute("""
    SELECT COUNT(DISTINCT drug1_normalized || '-' || drug2_normalized)
    FROM ddi_sentences 
    WHERE interaction_type = 'false' OR interaction_type IS NULL OR interaction_type = ''
""")
negative_pairs = cursor.fetchone()[0]
print(f"Negative/unknown pairs: {negative_pairs}")

conn.close()

print("\n" + "=" * 60)
print("DRUG DATABASE")
print("=" * 60)

with open('data/drug_db.json') as f:
    drugs = json.load(f)

print(f"Total drugs: {len(drugs)}")
with_smiles = sum(1 for d in drugs if d.get('smiles'))
print(f"With SMILES: {with_smiles}")

drug_names_db = set(d['name'].lower() for d in drugs)
overlap = all_drugs.intersection(drug_names_db)
print(f"\nOverlap with ddi_sentences drugs: {len(overlap)} / {len(all_drugs)}")

# Drugs in sentences that are NOT in drug_db
missing = all_drugs - drug_names_db
print(f"Drugs in sentences but NOT in drug_db: {len(missing)}")
print(f"  Sample: {sorted(list(missing))[:30]}")

print("\n" + "=" * 60)
print("PUBCHEM SMILES CACHE")
print("=" * 60)
cache_path = 'data/pubchem_smiles_cache.json'
if os.path.exists(cache_path):
    with open(cache_path) as f:
        cache = json.load(f)
    resolved = sum(1 for v in cache.values() if v)
    print(f"Total lookups: {len(cache)}")
    print(f"Resolved (have SMILES): {resolved}")
    print(f"Failed: {len(cache) - resolved}")
else:
    print("No cache file found")

print("\n" + "=" * 60)
print("DJANGO SQLITE DATABASE")
print("=" * 60)
conn2 = sqlite3.connect('db.sqlite3')
cursor2 = conn2.cursor()
cursor2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ddi_api%'")
tables = [t[0] for t in cursor2.fetchall()]
for table in tables:
    cursor2.execute(f"SELECT COUNT(*) FROM [{table}]")
    count = cursor2.fetchone()[0]
    print(f"  {table}: {count} rows")
conn2.close()
