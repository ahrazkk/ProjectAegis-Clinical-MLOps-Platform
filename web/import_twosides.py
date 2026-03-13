import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER', os.getenv('NEO4J_USERNAME'))
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

TWOSIDES_FILE = r"C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\TWOSIDES (1).csv.gz"

def import_twosides_edges():
    print("Connecting to Neo4j to get our valid root drugs...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        res = session.run("MATCH (d:Drug) WHERE d.smiles IS NOT NULL RETURN d.name as name")
        existing_drugs = set([record["name"].upper() for record in res])
        
    print(f"Loaded {len(existing_drugs)} valid drugs from Neo4j.")
    
    print("\nReading TWOSIDES dataset... (This might take a minute as it's a massive 4-million-row compressed file)")
    # We only need the explicit drug names for edges right now
    df = pd.read_csv(TWOSIDES_FILE, usecols=['drug_1_concept_name', 'drug_2_concept_name', 'condition_concept_name'])
    
    # Capitalize everything to match easily
    df['drug_1_concept_name'] = df['drug_1_concept_name'].str.upper()
    df['drug_2_concept_name'] = df['drug_2_concept_name'].str.upper()
    
    # FILTERING STRATEGY:
    # Neo4j free tier limits us to 400k relationships.
    # To keep the graph dense without blowing up the limit, 
    # we ONLY keep interactions where BOTH drugs are already in our database!
    print("Filtering down to interactions involving our known structured dataset...")
    
    mask = df['drug_1_concept_name'].isin(existing_drugs) & df['drug_2_concept_name'].isin(existing_drugs)
    dense_df = df[mask]
    
    # Group by the drug pairs to get the number of side effects and a list of the top ones
    print("Grouping by exact drug pairs...")
    grouped = dense_df.groupby(['drug_1_concept_name', 'drug_2_concept_name']).agg(
        num_side_effects=('condition_concept_name', 'count'),
        sample_effect=('condition_concept_name', 'first')
    ).reset_index()
    
    print(f"Discovered {len(grouped)} dense, high-quality interaction edges between our existing components!")
    
    if len(grouped) == 0:
        print("No intersecting drugs found. Did the extraction names format differently?")
        return

    # Prepare batches for Neo4j Unwind
    records = []
    for _, row in grouped.iterrows():
        # Title case to match our DB again
        d1 = str(row['drug_1_concept_name']).title()
        d2 = str(row['drug_2_concept_name']).title()
        
        # Avoid self-loops
        if d1 == d2: continue
            
        records.append({
            'source': d1,
            'target': d2,
            'severity': f"{row['num_side_effects']} known side effects",
            'desc': f"e.g. {row['sample_effect']}"
        })

    print("Uploading edges to Neo4j Aura securely via batch Unwind...")
    
    # We use UNWIND for massive bulk performance.
    query = """
    UNWIND $batch AS interaction
    MATCH (source:Drug {name: interaction.source})
    MATCH (target:Drug {name: interaction.target})
    MERGE (source)-[r:INTERACTS_WITH]-(target)
    ON CREATE SET 
        r.num_side_effects = interaction.severity, 
        r.description = interaction.desc, 
        r.source = 'TWOSIDES',
        r.added_at = datetime()
    """

    BATCH_SIZE = 1000
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i+BATCH_SIZE]
            session.run(query, batch=batch)
            print(f" -> Inserted batch of {len(batch)} edges...")

    driver.close()
    print("TWOSIDES DATABASE SUCCESSFULLY MAPPED AND UPLOADED!")

if __name__ == '__main__':
    import_twosides_edges()
