"""Quick script to show all drugs and interactions in the database"""
import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

import django
django.setup()

from ddi_api.services.knowledge_graph import KnowledgeGraphService

kg = KnowledgeGraphService

# Get all drugs
drugs = kg.run_query('MATCH (d:Drug) RETURN d.name as name, d.category as category ORDER BY d.name')
print('=' * 60)
print('DRUGS IN DATABASE')
print('=' * 60)
for d in drugs:
    cat = d.get('category', 'Unknown')
    print(f"  - {d['name']} ({cat})")

print(f"\nTotal: {len(drugs)} drugs")

# Get all interactions
interactions = kg.run_query('''
    MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
    RETURN d1.name as drug1, d2.name as drug2, r.severity as severity
    ORDER BY r.severity DESC, d1.name
''')

print('\n' + '=' * 60)
print('KNOWN INTERACTIONS (these will show as "Known" in the app)')
print('=' * 60)

severe = [i for i in interactions if i['severity'] == 'severe']
moderate = [i for i in interactions if i['severity'] == 'moderate']
minor = [i for i in interactions if i['severity'] == 'minor']

print('\n🔴 SEVERE INTERACTIONS:')
for i in severe:
    print(f"   {i['drug1']} + {i['drug2']}")

print('\n🟡 MODERATE INTERACTIONS:')
for i in moderate:
    print(f"   {i['drug1']} + {i['drug2']}")

print('\n🟢 MINOR INTERACTIONS:')
for i in minor:
    print(f"   {i['drug1']} + {i['drug2']}")

print(f"\nTotal: {len(interactions)} interactions")
print('\n' + '=' * 60)
print('HOW TO TEST:')
print('=' * 60)
print('''
1. Go to http://localhost:5173
2. Click "Enter Dashboard"
3. Search for drugs in the search box (left side)
4. Add 2 drugs, then click "Run Analysis"

RECOMMENDED TEST PAIRS:
  - Warfarin + Aspirin (SEVERE - bleeding risk)
  - Simvastatin + Amiodarone (SEVERE - rhabdomyolysis)
  - Warfarin + Ibuprofen (SEVERE - bleeding)
  - Digoxin + Amiodarone (SEVERE - digoxin toxicity)
  - Metoprolol + Verapamil (MODERATE - bradycardia)

CHAT TEST:
  Type in the chat: "What is the interaction between Warfarin and Aspirin?"
''')
