"""
DrugBank Data Importer for Neo4j Knowledge Graph

This script populates the Neo4j Knowledge Graph with drug data and interactions
from DrugBank, the drug_db.json file, and curated interactions.

Run: python manage.py setup_knowledge_graph
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)


# ============================================================================
# CURATED DRUG INTERACTIONS - High confidence, clinically validated
# ============================================================================
# Format: (drug1_id, drug1_name, drug2_id, drug2_name, severity, mechanism, description)

CURATED_INTERACTIONS = [
    # ===== ANTICOAGULANT INTERACTIONS =====
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
    
    # ===== CARDIAC DRUG INTERACTIONS =====
    ("DB00390", "Digoxin", "DB01118", "Amiodarone", "severe",
     "Reduced digoxin clearance by amiodarone",
     "Digoxin levels increase 70-100%; reduce digoxin dose by 50%"),
    
    ("DB00390", "Digoxin", "DB00661", "Verapamil", "moderate",
     "P-glycoprotein inhibition by verapamil",
     "Increased digoxin levels; monitor and reduce dose if needed"),
    
    ("DB00390", "Digoxin", "DB00908", "Quinidine", "severe",
     "Reduced renal and non-renal clearance of digoxin",
     "Digoxin levels may double; significant toxicity risk"),
    
    # ===== BETA BLOCKER + CCB INTERACTIONS =====
    ("DB00264", "Metoprolol", "DB00661", "Verapamil", "severe",
     "Additive negative chronotropic and inotropic effects",
     "Risk of severe bradycardia, heart block, and hypotension"),
    
    ("DB00264", "Metoprolol", "DB00343", "Diltiazem", "moderate",
     "Additive effects on cardiac conduction",
     "May cause significant bradycardia; use with caution"),
    
    ("DB00335", "Atenolol", "DB00661", "Verapamil", "moderate",
     "Additive cardiodepressant effects",
     "Risk of bradycardia and AV block"),
    
    # ===== STATIN INTERACTIONS =====
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
    
    # ===== OPIOID INTERACTIONS =====
    ("DB00813", "Fentanyl", "DB00503", "Ritonavir", "severe",
     "CYP3A4 inhibition by ritonavir",
     "Fatal respiratory depression possible; avoid or use extreme caution"),
    
    ("DB00497", "Oxycodone", "DB00196", "Fluconazole", "moderate",
     "CYP3A4 inhibition by fluconazole",
     "Increased oxycodone levels and CNS depression"),
    
    # ===== ANTIDEPRESSANT INTERACTIONS =====
    ("DB00472", "Fluoxetine", "DB00193", "Tramadol", "moderate",
     "CYP2D6 inhibition + serotonergic effects",
     "Reduced tramadol efficacy; increased serotonin syndrome risk"),
    
    ("DB01104", "Sertraline", "DB00193", "Tramadol", "moderate",
     "Serotonergic interaction",
     "Increased risk of serotonin syndrome and seizures"),
    
    ("DB00715", "Paroxetine", "DB00264", "Metoprolol", "moderate",
     "CYP2D6 inhibition by paroxetine",
     "Significantly increased metoprolol levels; risk of bradycardia"),
    
    # ===== PROTON PUMP INHIBITOR INTERACTIONS =====
    ("DB00338", "Omeprazole", "DB00758", "Clopidogrel", "moderate",
     "CYP2C19 inhibition reduces clopidogrel activation",
     "Potentially reduced antiplatelet effect; use pantoprazole instead"),
    
    # ===== IMMUNOSUPPRESSANT INTERACTIONS =====
    ("DB00091", "Cyclosporine", "DB01026", "Ketoconazole", "severe",
     "Strong CYP3A4 inhibition by ketoconazole",
     "Marked increase in cyclosporine levels; nephrotoxicity risk"),
    
    ("DB00864", "Tacrolimus", "DB01211", "Clarithromycin", "severe",
     "CYP3A4 inhibition by clarithromycin",
     "Elevated tacrolimus levels; nephrotoxicity risk"),
    
    # ===== ANTIEPILEPTIC INTERACTIONS =====
    ("DB00252", "Phenytoin", "DB00196", "Fluconazole", "moderate",
     "CYP2C9 inhibition by fluconazole",
     "Increased phenytoin levels; monitor for toxicity"),
    
    ("DB00564", "Carbamazepine", "DB00199", "Erythromycin", "moderate",
     "CYP3A4 inhibition by erythromycin",
     "Increased carbamazepine levels; toxicity risk"),
    
    # ===== LITHIUM INTERACTIONS =====
    ("DB01356", "Lithium", "DB01050", "Ibuprofen", "moderate",
     "Reduced renal lithium clearance by NSAIDs",
     "Increased lithium levels; toxicity risk"),
    
    ("DB01356", "Lithium", "DB00722", "Lisinopril", "moderate",
     "ACE inhibitor reduces lithium excretion",
     "Increased lithium levels; monitor closely"),
    
    # ===== BENZODIAZEPINE INTERACTIONS =====
    ("DB00829", "Diazepam", "OPIOIDS", "Opioids", "severe",
     "Additive CNS depression",
     "Risk of profound sedation, respiratory depression, and death"),
    
    ("DB00404", "Alprazolam", "DB01026", "Ketoconazole", "moderate",
     "CYP3A4 inhibition by ketoconazole",
     "Significantly increased alprazolam levels"),
    
    # ===== NO SIGNIFICANT INTERACTIONS =====
    ("DB00316", "Acetaminophen", "DB00338", "Omeprazole", "minor",
     "No significant interaction",
     "These drugs can be safely used together at standard doses"),
    
    ("DB00722", "Lisinopril", "DB00331", "Metformin", "minor",
     "No significant pharmacokinetic interaction",
     "Safe to use together; may have renoprotective synergy"),
]


class Command(BaseCommand):
    help = 'Setup Neo4j Knowledge Graph with drug data and interactions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before loading'
        )
        parser.add_argument(
            '--drugs-only',
            action='store_true',
            help='Only load drugs, not interactions'
        )

    def handle(self, *args, **options):
        from ddi_api.services.knowledge_graph import KnowledgeGraphService as KG
        
        # Check connection
        if not KG.is_connected():
            self.stdout.write(self.style.ERROR('Cannot connect to Neo4j'))
            self.stdout.write('Make sure Neo4j is running and check NEO4J_CONFIG in settings.py')
            return
        
        self.stdout.write(self.style.SUCCESS('Connected to Neo4j'))
        
        if options['clear']:
            self.stdout.write('Clearing existing data...')
            KG.run_query('MATCH (n) DETACH DELETE n')
        
        # Create schema
        self.stdout.write('Creating schema...')
        KG.create_schema()
        
        # Load drugs from drug_db.json
        self.stdout.write('Loading drugs from drug_db.json...')
        drugs_loaded = self.load_drugs_from_json(KG)
        self.stdout.write(self.style.SUCCESS(f'  Loaded {drugs_loaded} drugs'))
        
        if not options['drugs_only']:
            # Load curated interactions
            self.stdout.write('Loading curated interactions...')
            interactions_loaded = self.load_curated_interactions(KG)
            self.stdout.write(self.style.SUCCESS(f'  Loaded {interactions_loaded} interactions'))
        
        # Print stats
        self.print_stats(KG)

    def load_drugs_from_json(self, KG) -> int:
        """Load drugs from drug_db.json."""
        drug_db_path = Path(settings.BASE_DIR) / 'data' / 'drug_db.json'
        
        if not drug_db_path.exists():
            self.stdout.write(self.style.WARNING(f'  drug_db.json not found at {drug_db_path}'))
            return 0
        
        with open(drug_db_path, 'r') as f:
            drugs = json.load(f)
        
        count = 0
        for drug in drugs:
            KG.add_drug(
                drugbank_id=drug.get('drugbank_id', f"UNKNOWN_{count}"),
                name=drug.get('name', ''),
                smiles=drug.get('smiles'),
                category=drug.get('category'),
                description=drug.get('description')
            )
            count += 1
        
        return count

    def load_curated_interactions(self, KG) -> int:
        """Load curated drug interactions."""
        count = 0
        
        for interaction in CURATED_INTERACTIONS:
            drug1_id, drug1_name, drug2_id, drug2_name, severity, mechanism, description = interaction
            
            # Ensure both drugs exist
            # First check if they exist, if not create them
            existing1 = KG.get_drug(drugbank_id=drug1_id)
            if not existing1:
                KG.add_drug(drugbank_id=drug1_id, name=drug1_name)
            
            existing2 = KG.get_drug(drugbank_id=drug2_id)
            if not existing2 and drug2_id != "OPIOIDS":  # Skip generic category
                KG.add_drug(drugbank_id=drug2_id, name=drug2_name)
            
            # Skip if second drug is a generic category
            if drug2_id == "OPIOIDS":
                continue
            
            # Add interaction
            KG.add_interaction(
                drug1_id=drug1_id,
                drug2_id=drug2_id,
                severity=severity,
                mechanism=mechanism,
                description=description,
                evidence_level='curated'
            )
            count += 1
        
        return count

    def print_stats(self, KG):
        """Print Knowledge Graph statistics."""
        drug_count = KG.run_query('MATCH (d:Drug) RETURN count(d) as count')
        interaction_count = KG.run_query('MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count')
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=== Knowledge Graph Stats ==='))
        self.stdout.write(f"  Drugs: {drug_count[0]['count'] if drug_count else 0}")
        self.stdout.write(f"  Interactions: {interaction_count[0]['count'] if interaction_count else 0}")
        
        # Show severity breakdown
        severity_counts = KG.run_query('''
            MATCH ()-[i:INTERACTS_WITH]->() 
            RETURN i.severity as severity, count(*) as count
        ''')
        if severity_counts:
            self.stdout.write('  By severity:')
            for row in severity_counts:
                self.stdout.write(f"    {row['severity']}: {row['count']}")
