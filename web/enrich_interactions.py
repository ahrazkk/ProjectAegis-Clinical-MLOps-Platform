import os
import requests
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def get_rxcui(drug_name):
    """Get the standard RxNorm identifier for a drug name from NIH."""
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                return data['idGroup']['rxnormId'][0]
    except Exception as e:
        pass
    return None

def get_interactions(rxcui):
    """Get drug interactions for a specific RxCUI."""
    url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
    interactions = []
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'interactionTypeGroup' in data:
                for group in data['interactionTypeGroup']:
                    for itype in group.get('interactionType', []):
                        for pair in itype.get('interactionPair', []):
                            desc = pair.get('description', 'Unknown interaction')
                            severity = pair.get('severity', 'N/A')
                            
                            # pair['interactionConcept'][1] is the target drug we are interacting with
                            target_drug = pair['interactionConcept'][1]['minConceptItem']['name']
                            interactions.append({
                                "target": target_drug, 
                                "description": desc,
                                "severity": severity
                            })
    except Exception as e:
        pass
    return interactions

def main():
    print("Connecting to Neo4j Aura...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Fetch all valid drugs in the database
    with driver.session() as session:
        res = session.run("MATCH (d:Drug) WHERE d.smiles IS NOT NULL RETURN d.name as name")
        my_drugs = [record['name'] for record in res]
        
    my_drugs_lower = {d.lower(): d for d in my_drugs}
    
    print(f"Loaded {len(my_drugs)} active molecular drugs from your Neo4j database.")
    print("Querying the NIH RxNav Database for interactions between your specific drugs...")
    
    edges_added = 0
    drugs_processed = 0
    
    with driver.session() as session:
        for i, drug in enumerate(my_drugs):
            # We'll limit to a subset during this run if we want to test, but we can do all 1350
            # Warning: 1350 takes ~10 minutes, so we will print progress actively.
            
            print(f"[{i+1}/{len(my_drugs)}] {drug}")
            
            rxcui = get_rxcui(drug)
            time.sleep(0.1) # Be nice to NIH servers
            
            if not rxcui:
                continue
                
            interactions = get_interactions(rxcui)
            time.sleep(0.1)
            
            for interaction in interactions:
                target_name_lower = interaction['target'].lower()
                
                # Check if the interaction target is ALSO a drug in your Neo4j database
                if target_name_lower in my_drugs_lower:
                    target_db_name = my_drugs_lower[target_name_lower]
                    
                    if target_db_name != drug:
                        # MERGE prevents creating duplicate edges. It acts like an UPSERT.
                        query = """
                        MATCH (d1:Drug {name: $drug1})
                        MATCH (d2:Drug {name: $drug2})
                        MERGE (d1)-[r:INTERACTS_WITH]-(d2)
                        ON CREATE SET r.description = $desc, r.severity = $severity, r.source = 'NIH_RxNav', r.added_at = datetime()
                        """
                        session.run(query, drug1=drug, drug2=target_db_name, desc=interaction['description'], severity=interaction['severity'])
                        edges_added += 1
                        print(f"  [+] New interaction found & linked: {drug} <-> {target_db_name}")

            drugs_processed += 1
            
            # Print a neat summary every 50 drugs
            if drugs_processed % 50 == 0:
                print(f"--- Checkpoint: Added {edges_added} interaction edges so far ---")
                
    print(f"\n[COMPLETE] Successfully added {edges_added} total new interactions to Neo4j!")
    driver.close()

if __name__ == '__main__':
    main()
