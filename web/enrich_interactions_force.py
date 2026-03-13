import os
import requests
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def fetch_rx_interactions(rxcui):
    """Fetch all interactions for a given RxCUI regardless of if it's currently in our DB"""
    url = f"https://rxnav.nlm.nih.gov/REST/interaction/interaction.json?rxcui={rxcui}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

def extract_drug_data(name):
    """If a new drug interacts with ours, fetch its SMILES to add it to the DB."""
    try:
        import pubchempy as pcp
        res = pcp.get_compounds(name, 'name')
        if res:
            return res[0].isomeric_smiles
    except:
        pass
    return None

def main():
    print("Connecting to Neo4j to pull ALL possible RxNav interactions...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Get ~10 drugs from the database that we know have high interactivity
        target_drugs = ['warfarin', 'aspirin', 'ibuprofen', 'heparin', 'lisinopril']
        
        for root_drug in target_drugs:
            print(f"\n--- Expanding universe from root drug: {root_drug.upper()} ---")
            
            # Step 1: Find its RxNorm ID
            rx_url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={root_drug}"
            rxcui = None
            try:
                resp = requests.get(rx_url).json()
                rxcui = resp['idGroup']['rxnormId'][0]
            except:
                continue
                
            if not rxcui: continue
            
            # Step 2: Get ALL interactions
            int_data = fetch_rx_interactions(rxcui)
            if not int_data or 'interactionTypeGroup' not in int_data:
                continue
                
            edges_added = 0
            new_nodes = 0
            
            for group in int_data['interactionTypeGroup']:
                for itype in group.get('interactionType', []):
                    for pair in itype.get('interactionPair', []):
                        
                        target_name = pair['interactionConcept'][1]['minConceptItem']['name']
                        desc = pair.get('description', '')
                        severity = pair.get('severity', 'moderate')
                        
                        # Use Neo4j's MERGE to gracefully handle it
                        # If the target drug doesn't exist, we CREATE it.
                        # If it does exist, we just add the EDGE.
                        query = """
                        // 1. Get or Create Source
                        MERGE (d1:Drug {name: $root_name})
                        
                        // 2. Get or Create Target
                        MERGE (d2:Drug {name: $target_name})
                        ON CREATE SET d2.requires_smiles = true
                        
                        // 3. Create Edge
                        MERGE (d1)-[r:INTERACTS_WITH]-(d2)
                        ON CREATE SET r.description = $desc, r.severity = $severity
                        """
                        try:
                            session.run(query, 
                                        root_name=root_drug.capitalize(),
                                        target_name=target_name.capitalize(),
                                        desc=desc,
                                        severity=severity)
                            edges_added += 1
                        except:
                            pass
                            
            print(f"Added {edges_added} interactions for {root_drug}.")

    driver.close()
    
if __name__ == '__main__':
    main()
