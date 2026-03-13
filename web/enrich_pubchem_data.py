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

# PubChem API Base URL
PUBCHEM_URL_TEMPLATE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES,MolecularWeight,MolecularFormula,IUPACName/JSON"

def get_pubchem_data(drug_name):
    """Fetch chemical properties from PubChem using the drug name."""
    try:
        response = requests.get(PUBCHEM_URL_TEMPLATE.format(drug_name), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties'][0]
                return {
                    'pubchem_cid': props.get('CID'),
                    'smiles': props.get('CanonicalSMILES'),
                    'molecular_weight': str(props.get('MolecularWeight', '')),
                    'molecular_formula': props.get('MolecularFormula'),
                    'iupac_name': props.get('IUPACName')
                }
        elif response.status_code == 404:
            print(f"  [!] Not found in PubChem: {drug_name}")
        else:
            print(f"  [X] API Error {response.status_code} for {drug_name}")
    except Exception as e:
        print(f"  [X] Request failed for {drug_name}: {e}")
    
    return None

def update_drug_in_neo4j(session, drug_id, properties):
    """Update existing drug node in Neo4j with new properties, avoiding duplicate columns."""
    query = """
    MATCH (d:Drug) WHERE id(d) = $node_id
    SET d.smiles = $smiles,
        d.pubchem_cid = $pubchem_cid,
        d.molecular_weight = $molecular_weight,
        d.molecular_formula = $molecular_formula,
        d.iupac_name = $iupac_name,
        d.updated_at = datetime()
    RETURN d.name
    """
    session.run(query, 
                node_id=drug_id, 
                smiles=properties['smiles'],
                pubchem_cid=properties['pubchem_cid'],
                molecular_weight=properties['molecular_weight'],
                molecular_formula=properties['molecular_formula'],
                iupac_name=properties['iupac_name'])

def main():
    print("Connecting to Neo4j Database...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Find drugs that DO NOT have a smiles string
        print("Querying for drugs missing SMILES data...")
        result = session.run("MATCH (d:Drug) WHERE d.smiles IS NULL OR d.smiles = '' RETURN id(d) as id, d.name as name")
        
        drugs_to_update = [{"id": record["id"], "name": record["name"]} for record in result]
        total = len(drugs_to_update)
        print(f"Found {total} drugs requiring enrichment.")
        
        if total == 0:
            print("No action required. All drugs have SMILES.")
            return

        success_count = 0
        
        for i, drug in enumerate(drugs_to_update):
            name = drug['name']
            print(f"[{i+1}/{total}] Processing: {name}")
            
            # Rate limiting for PubChem (avoid getting banned)
            time.sleep(0.3) 
            
            properties = get_pubchem_data(name)
            if properties and properties.get('smiles'):
                update_drug_in_neo4j(session, drug['id'], properties)
                success_count += 1
                print(f"  [+] Successfully updated {name}.")
            
        print(f"\n--- Enrichment Complete ---")
        print(f"Successfully updated {success_count} out of {total} targeted drugs.")

    driver.close()

if __name__ == '__main__':
    main()
