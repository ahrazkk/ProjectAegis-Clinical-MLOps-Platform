"""Query Neo4j Aura to check available drug/interaction data."""
from neo4j import GraphDatabase

uri = 'neo4j+s://39312fd4.databases.neo4j.io'
user = 'neo4j'
pwd = '1vlj-fnbN3x_56x4Cty4xBF9v9deOMj-NpFu2dg9Oro'

driver = GraphDatabase.driver(uri, auth=(user, pwd))

with driver.session() as session:
    # Count drugs
    result = session.run('MATCH (d:Drug) RETURN count(d) as cnt')
    drug_count = result.single()['cnt']
    print(f"Total Drug nodes: {drug_count}")
    
    # Count drugs with SMILES
    result = session.run('MATCH (d:Drug) WHERE d.smiles IS NOT NULL AND d.smiles <> "" RETURN count(d) as cnt')
    smiles_count = result.single()['cnt']
    print(f"Drugs with SMILES: {smiles_count}")
    
    # Count interactions
    result = session.run('MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as cnt')
    int_count = result.single()['cnt']
    print(f"INTERACTS_WITH relationships: {int_count}")
    
    # Severity distribution
    result = session.run('''
        MATCH ()-[i:INTERACTS_WITH]->()
        RETURN i.severity as severity, count(*) as cnt
        ORDER BY cnt DESC
    ''')
    print("\nSeverity distribution:")
    for record in result:
        print(f"  {record['severity']}: {record['cnt']}")
    
    # Sample drug nodes (see what properties exist)
    result = session.run('MATCH (d:Drug) RETURN d LIMIT 5')
    print("\nSample Drug nodes:")
    for record in result:
        node = dict(record['d'])
        print(f"  {node.get('name', '?')}: keys={list(node.keys())}")
        if node.get('smiles'):
            print(f"    SMILES: {node['smiles'][:60]}...")
    
    # Sample interactions
    result = session.run('''
        MATCH (a:Drug)-[i:INTERACTS_WITH]->(b:Drug)
        RETURN a.name as drug1, b.name as drug2, i.severity as sev, i.description as desc
        LIMIT 5
    ''')
    print("\nSample interactions:")
    for record in result:
        desc = record['desc'] or ''
        print(f"  {record['drug1']} + {record['drug2']}: severity={record['sev']}")
        if desc:
            print(f"    {desc[:100]}")
    
    # Check other relationship types
    result = session.run('''
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as cnt
        ORDER BY cnt DESC
    ''')
    print("\nAll relationship types:")
    for record in result:
        print(f"  {record['rel_type']}: {record['cnt']}")
    
    # Check node labels
    result = session.run('''
        MATCH (n)
        RETURN labels(n) as labels, count(*) as cnt
        ORDER BY cnt DESC
    ''')
    print("\nAll node types:")
    for record in result:
        print(f"  {record['labels']}: {record['cnt']}")

driver.close()
print("\nDone!")
