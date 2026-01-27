#!/usr/bin/env python
"""
Neo4j Aura Diagnostic and Data Population Script
================================================

This script helps you:
1. Verify connection to Neo4j Aura
2. Check what data exists
3. Populate data from drug_db.json and curated interactions

Usage:
    Set environment variables in .env file, then run:
    python diagnose_neo4j.py

Environment Variables Needed (set in .env file):
    NEO4J_URI=neo4j+s://your-database-id.databases.neo4j.io
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password-here
"""

import os
import sys
import json
from pathlib import Path

# Add the Django project to the path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

# Setup Django
import django
django.setup()

from neo4j import GraphDatabase
from django.conf import settings


def get_neo4j_connection():
    """Get Neo4j connection using environment variables or settings."""
    uri = os.environ.get('NEO4J_URI', settings.NEO4J_CONFIG.get('uri'))
    user = os.environ.get('NEO4J_USER', settings.NEO4J_CONFIG.get('user'))
    password = os.environ.get('NEO4J_PASSWORD', settings.NEO4J_CONFIG.get('password'))
    
    print(f"\n{'='*60}")
    print("NEO4J CONNECTION INFO")
    print(f"{'='*60}")
    print(f"URI: {uri}")
    print(f"User: {user}")
    print(f"Password: {'*' * len(password[:5]) + '...' if password else 'NOT SET'}")
    
    return uri, user, password


def run_query(driver, query, parameters=None):
    """Run a Cypher query."""
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def diagnose():
    """Run diagnostics on Neo4j connection and data."""
    uri, user, password = get_neo4j_connection()
    
    # Check if Aura connection
    is_aura = 'databases.neo4j.io' in uri or uri.startswith('neo4j+s://')
    print(f"\nConnection Type: {'Neo4j Aura (Cloud)' if is_aura else 'Local Neo4j'}")
    
    if not is_aura:
        print("\n⚠️  WARNING: You're connecting to LOCAL Neo4j, not Aura!")
        print("   Set these environment variables to connect to Aura:")
        print("   $env:NEO4J_URI=\"neo4j+s://ca47aebc.databases.neo4j.io\"")
        print("   $env:NEO4J_USER=\"neo4j\"")
        print("   $env:NEO4J_PASSWORD=\"YOUR_PASSWORD\"")
    
    try:
        print(f"\n{'='*60}")
        print("CONNECTING TO NEO4J...")
        print(f"{'='*60}")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ Connection successful!")
        
        # Check data
        print(f"\n{'='*60}")
        print("CHECKING DATA...")
        print(f"{'='*60}")
        
        # Count drugs
        drug_count = run_query(driver, "MATCH (d:Drug) RETURN count(d) as count")
        print(f"📦 Drugs: {drug_count[0]['count'] if drug_count else 0}")
        
        # Count interactions
        interaction_count = run_query(driver, "MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count")
        print(f"🔗 Interactions: {interaction_count[0]['count'] if interaction_count else 0}")
        
        # Show sample drugs
        sample_drugs = run_query(driver, "MATCH (d:Drug) RETURN d.name as name, d.drugbank_id as id LIMIT 5")
        if sample_drugs:
            print(f"\n📋 Sample drugs:")
            for drug in sample_drugs:
                print(f"   - {drug['name']} ({drug['id']})")
        else:
            print("\n❌ No drugs found in database!")
        
        # Check interaction breakdown
        severity_breakdown = run_query(driver, """
            MATCH ()-[i:INTERACTS_WITH]->() 
            RETURN i.severity as severity, count(*) as count
            ORDER BY count DESC
        """)
        if severity_breakdown:
            print(f"\n📊 Interactions by severity:")
            for row in severity_breakdown:
                print(f"   - {row['severity']}: {row['count']}")
        
        # Check evidence levels
        evidence_breakdown = run_query(driver, """
            MATCH ()-[i:INTERACTS_WITH]->() 
            RETURN i.evidence_level as source, count(*) as count
            ORDER BY count DESC
        """)
        if evidence_breakdown:
            print(f"\n📚 Interactions by source:")
            for row in evidence_breakdown:
                print(f"   - {row['source']}: {row['count']}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False


def populate_data(clear=False):
    """Populate Neo4j Aura with drug data and interactions."""
    uri, user, password = get_neo4j_connection()
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("\n✅ Connected to Neo4j")
        
        if clear:
            print("\n⚠️  Clearing all existing data...")
            run_query(driver, "MATCH (n) DETACH DELETE n")
            print("   Done!")
        
        # Create schema
        print("\n📐 Creating schema...")
        schema_queries = [
            "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE",
            "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX drug_category IF NOT EXISTS FOR (d:Drug) ON (d.category)",
        ]
        for query in schema_queries:
            try:
                run_query(driver, query)
            except Exception as e:
                print(f"   Schema note: {e}")
        print("   Done!")
        
        # Load drugs from drug_db.json
        drug_db_path = Path(__file__).parent / 'data' / 'drug_db.json'
        print(f"\n💊 Loading drugs from {drug_db_path}...")
        
        with open(drug_db_path, 'r') as f:
            drugs = json.load(f)
        
        for drug in drugs:
            run_query(driver, """
                MERGE (d:Drug {drugbank_id: $drugbank_id})
                SET d.name = $name,
                    d.smiles = $smiles,
                    d.category = $category,
                    d.updated_at = datetime()
            """, {
                'drugbank_id': drug.get('drugbank_id', ''),
                'name': drug.get('name', ''),
                'smiles': drug.get('smiles'),
                'category': drug.get('category')
            })
        
        print(f"   Loaded {len(drugs)} drugs")
        
        # Load curated interactions (copy from setup_knowledge_graph.py)
        print("\n🔗 Loading curated interactions...")
        
        CURATED_INTERACTIONS = [
            # ANTICOAGULANT INTERACTIONS
            ("DB00682", "Warfarin", "DB00945", "Aspirin", "severe",
             "Additive antiplatelet/anticoagulant effects",
             "Significantly increased risk of bleeding, especially GI hemorrhage"),
            
            ("DB00682", "Warfarin", "DB01050", "Ibuprofen", "severe",
             "NSAID inhibition of platelet function + warfarin anticoagulation",
             "Increased bleeding risk; NSAIDs also displace warfarin from protein binding"),
            
            ("DB00682", "Warfarin", "DB00563", "Metronidazole", "moderate",
             "CYP2C9 inhibition by metronidazole",
             "Increased warfarin effect; monitor INR closely"),
            
            ("DB00682", "Warfarin", "DB01045", "Rifampin", "severe",
             "CYP induction by rifampin",
             "Dramatically reduced warfarin efficacy; requires major dose adjustment"),
            
            # CARDIAC DRUG INTERACTIONS
            ("DB00390", "Digoxin", "DB01118", "Amiodarone", "severe",
             "Reduced digoxin clearance by amiodarone",
             "Digoxin levels increase 70-100%; reduce digoxin dose by 50%"),
            
            ("DB00390", "Digoxin", "DB00661", "Verapamil", "moderate",
             "P-glycoprotein inhibition by verapamil",
             "Increased digoxin levels; monitor and reduce dose if needed"),
            
            ("DB00390", "Digoxin", "DB00908", "Quinidine", "severe",
             "Reduced renal and non-renal clearance of digoxin",
             "Digoxin levels may double; significant toxicity risk"),
            
            # BETA BLOCKER + CCB INTERACTIONS
            ("DB00264", "Metoprolol", "DB00661", "Verapamil", "severe",
             "Additive negative chronotropic and inotropic effects",
             "Risk of severe bradycardia, heart block, and hypotension"),
            
            ("DB00264", "Metoprolol", "DB00343", "Diltiazem", "moderate",
             "Additive effects on cardiac conduction",
             "May cause significant bradycardia; use with caution"),
            
            ("DB00335", "Atenolol", "DB00661", "Verapamil", "moderate",
             "Additive cardiodepressant effects",
             "Risk of bradycardia and AV block"),
            
            # STATIN INTERACTIONS
            ("DB00641", "Simvastatin", "DB01118", "Amiodarone", "severe",
             "CYP3A4 inhibition by amiodarone",
             "Increased simvastatin levels; max dose 20mg with amiodarone"),
            
            ("DB00641", "Simvastatin", "DB01211", "Clarithromycin", "severe",
             "Strong CYP3A4 inhibition by clarithromycin",
             "Avoid combination; high risk of rhabdomyolysis"),
            
            ("DB00641", "Simvastatin", "DB00199", "Erythromycin", "severe",
             "CYP3A4 inhibition by erythromycin",
             "Increased myopathy risk; avoid combination"),
            
            ("DB01076", "Atorvastatin", "DB01211", "Clarithromycin", "moderate",
             "CYP3A4 inhibition increases atorvastatin exposure",
             "Limit atorvastatin dose; monitor for muscle symptoms"),
            
            # OPIOID INTERACTIONS
            ("DB00813", "Fentanyl", "DB00503", "Ritonavir", "severe",
             "CYP3A4 inhibition by ritonavir",
             "Fatal respiratory depression possible; avoid or use extreme caution"),
            
            ("DB00497", "Oxycodone", "DB00196", "Fluconazole", "moderate",
             "CYP3A4 inhibition by fluconazole",
             "Increased oxycodone levels and CNS depression"),
            
            # ANTIDEPRESSANT INTERACTIONS
            ("DB00472", "Fluoxetine", "DB00193", "Tramadol", "moderate",
             "CYP2D6 inhibition + serotonergic effects",
             "Reduced tramadol efficacy; increased serotonin syndrome risk"),
            
            ("DB01104", "Sertraline", "DB00193", "Tramadol", "moderate",
             "Serotonergic interaction",
             "Increased risk of serotonin syndrome and seizures"),
            
            ("DB00715", "Paroxetine", "DB00264", "Metoprolol", "moderate",
             "CYP2D6 inhibition by paroxetine",
             "Significantly increased metoprolol levels; risk of bradycardia"),
            
            # PROTON PUMP INHIBITOR INTERACTIONS
            ("DB00338", "Omeprazole", "DB00758", "Clopidogrel", "moderate",
             "CYP2C19 inhibition reduces clopidogrel activation",
             "Potentially reduced antiplatelet effect; use pantoprazole instead"),
            
            # IMMUNOSUPPRESSANT INTERACTIONS
            ("DB00091", "Cyclosporine", "DB01026", "Ketoconazole", "severe",
             "Strong CYP3A4 inhibition by ketoconazole",
             "Marked increase in cyclosporine levels; nephrotoxicity risk"),
            
            ("DB00864", "Tacrolimus", "DB01211", "Clarithromycin", "severe",
             "CYP3A4 inhibition by clarithromycin",
             "Elevated tacrolimus levels; nephrotoxicity risk"),
            
            # ANTIEPILEPTIC INTERACTIONS
            ("DB00252", "Phenytoin", "DB00196", "Fluconazole", "moderate",
             "CYP2C9 inhibition by fluconazole",
             "Increased phenytoin levels; monitor for toxicity"),
            
            ("DB00564", "Carbamazepine", "DB00199", "Erythromycin", "moderate",
             "CYP3A4 inhibition by erythromycin",
             "Increased carbamazepine levels; toxicity risk"),
            
            # LITHIUM INTERACTIONS
            ("DB01356", "Lithium", "DB01050", "Ibuprofen", "moderate",
             "Reduced renal lithium clearance by NSAIDs",
             "Increased lithium levels; toxicity risk"),
            
            ("DB01356", "Lithium", "DB00722", "Lisinopril", "moderate",
             "ACE inhibitor reduces lithium excretion",
             "Increased lithium levels; monitor closely"),
            
            # BENZODIAZEPINE INTERACTIONS
            ("DB00404", "Alprazolam", "DB01026", "Ketoconazole", "moderate",
             "CYP3A4 inhibition by ketoconazole",
             "Significantly increased alprazolam levels"),
            
            # NO SIGNIFICANT INTERACTIONS
            ("DB00316", "Acetaminophen", "DB00338", "Omeprazole", "minor",
             "No significant interaction",
             "These drugs can be safely used together at standard doses"),
            
            ("DB00722", "Lisinopril", "DB00331", "Metformin", "minor",
             "No significant pharmacokinetic interaction",
             "Safe to use together; may have renoprotective synergy"),
        ]
        
        interaction_count = 0
        for interaction in CURATED_INTERACTIONS:
            drug1_id, drug1_name, drug2_id, drug2_name, severity, mechanism, description = interaction
            
            # Ensure drugs exist
            run_query(driver, """
                MERGE (d:Drug {drugbank_id: $id})
                ON CREATE SET d.name = $name
            """, {'id': drug1_id, 'name': drug1_name})
            
            run_query(driver, """
                MERGE (d:Drug {drugbank_id: $id})
                ON CREATE SET d.name = $name
            """, {'id': drug2_id, 'name': drug2_name})
            
            # Add interaction
            run_query(driver, """
                MATCH (d1:Drug {drugbank_id: $drug1_id})
                MATCH (d2:Drug {drugbank_id: $drug2_id})
                MERGE (d1)-[i:INTERACTS_WITH]->(d2)
                SET i.severity = $severity,
                    i.description = $description,
                    i.mechanism = $mechanism,
                    i.evidence_level = 'curated',
                    i.updated_at = datetime()
            """, {
                'drug1_id': drug1_id,
                'drug2_id': drug2_id,
                'severity': severity,
                'description': description,
                'mechanism': mechanism
            })
            interaction_count += 1
        
        print(f"   Loaded {interaction_count} interactions")
        
        # Final stats
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        
        drug_count = run_query(driver, "MATCH (d:Drug) RETURN count(d) as count")
        print(f"📦 Total Drugs: {drug_count[0]['count']}")
        
        interaction_count = run_query(driver, "MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count")
        print(f"🔗 Total Interactions: {interaction_count[0]['count']}")
        
        driver.close()
        print("\n✅ Data population complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           NEO4J AURA DIAGNOSTIC TOOL                         ║
║           Project Aegis - DDI System                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Options:")
    print("  1. Diagnose - Check connection and data")
    print("  2. Populate - Load drugs and interactions (keeps existing)")
    print("  3. Reset & Populate - Clear all data and reload")
    print("  4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        diagnose()
    elif choice == '2':
        if diagnose():
            print("\n" + "="*60)
            confirm = input("\nProceed with data population? (y/n): ").strip().lower()
            if confirm == 'y':
                populate_data(clear=False)
    elif choice == '3':
        print("\n⚠️  WARNING: This will DELETE all existing data!")
        confirm = input("Are you sure? (type 'yes' to confirm): ").strip().lower()
        if confirm == 'yes':
            populate_data(clear=True)
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
