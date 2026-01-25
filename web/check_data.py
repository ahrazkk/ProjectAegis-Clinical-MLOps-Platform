"""Quick Neo4j data dump script"""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ.get('NEO4J_URI'),
    auth=(os.environ.get('NEO4J_USER'), os.environ.get('NEO4J_PASSWORD'))
)

print('='*60)
print('NEO4J AURA DATA SUMMARY')
print('='*60)

with driver.session() as session:
    # Count drugs
    result = session.run('MATCH (d:Drug) RETURN count(d) as count')
    drug_count = result.single()['count']
    print(f'\nTotal Drugs: {drug_count}')
    
    # Count interactions  
    result = session.run('MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count')
    int_count = result.single()['count']
    print(f'Total Interactions: {int_count}')
    
    # By evidence level
    result = session.run('''
        MATCH ()-[i:INTERACTS_WITH]->() 
        RETURN i.evidence_level as source, count(*) as count
        ORDER BY count DESC
    ''')
    print('\nBy source:')
    for r in result:
        source = r['source'] or 'unknown'
        count = r['count']
        print(f'  {source}: {count}')
    
    # By DDI type
    result = session.run('''
        MATCH ()-[i:INTERACTS_WITH]->() 
        WHERE i.ddi_type IS NOT NULL
        RETURN i.ddi_type as type, count(*) as count
        ORDER BY count DESC
    ''')
    print('\nBy DDI type:')
    for r in result:
        ddi_type = r['type']
        count = r['count']
        print(f'  {ddi_type}: {count}')
    
    # Sample drugs
    result = session.run('MATCH (d:Drug) RETURN d.name as name LIMIT 10')
    print('\nSample drugs:')
    for r in result:
        print(f'  - {r["name"]}')

driver.close()
print('\n' + '='*60)
print('Done!')
