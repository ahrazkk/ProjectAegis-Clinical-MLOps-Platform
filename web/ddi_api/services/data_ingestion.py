"""
Data Ingestion Pipeline for DDI Knowledge Graph
Downloads and loads drug interaction data from public sources
"""

import os
import json
import csv
import requests
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Django setup for standalone execution
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
django.setup()

from ddi_api.services.knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DRUGBANK_DIR = DATA_DIR / 'drugbank'
SIDER_DIR = DATA_DIR / 'sider'


class DrugDataIngestion:
    """Handles downloading and ingesting drug data into Neo4j"""
    
    # Public drug interaction datasets (no license required for academic use)
    DATA_SOURCES = {
        'drugbank_open': {
            'url': 'https://go.drugbank.com/releases/5-1-10/downloads/all-drug-links',
            'description': 'DrugBank Open Data (drug names and identifiers)',
            'format': 'csv'
        },
        'twosides': {
            'url': 'http://tatonettilab.org/resources/nsides/TWOSIDES.csv.gz',
            'description': 'TWOSIDES drug-drug interaction database',
            'format': 'csv.gz'
        },
        'sider': {
            'url': 'http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz',
            'description': 'SIDER side effects database',
            'format': 'tsv.gz'
        }
    }
    
    # Sample drug data for demo/testing (when full datasets unavailable)
    SAMPLE_DRUGS = [
        {"id": "DB00945", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "category": "NSAID"},
        {"id": "DB00682", "name": "Warfarin", "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "category": "Anticoagulant"},
        {"id": "DB00563", "name": "Methotrexate", "smiles": "CN(CC1=CN=C2N=C(N)N=C(N)C2=N1)C3=CC=C(C(=O)NC(CCC(=O)O)C(=O)O)C=C3", "category": "Antimetabolite"},
        {"id": "DB01050", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "category": "NSAID"},
        {"id": "DB00316", "name": "Acetaminophen", "smiles": "CC(=O)NC1=CC=C(C=C1)O", "category": "Analgesic"},
        {"id": "DB00390", "name": "Digoxin", "smiles": "CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC7(C)C(C8=CC(=O)OC8)CCC67)C4)OC3C)OC2C)C(O)C(O)C1O", "category": "Cardiac Glycoside"},
        {"id": "DB00641", "name": "Simvastatin", "smiles": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12", "category": "Statin"},
        {"id": "DB00571", "name": "Propranolol", "smiles": "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12", "category": "Beta Blocker"},
        {"id": "DB01118", "name": "Amiodarone", "smiles": "CCCCC1=C(C2=CC=C(OCCN(CC)CC)C=C2)C3=CC(I)=C(OCCC)C(I)=C3O1", "category": "Antiarrhythmic"},
        {"id": "DB00959", "name": "Methylprednisolone", "smiles": "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO", "category": "Corticosteroid"},
        {"id": "DB00564", "name": "Carbamazepine", "smiles": "NC(=O)N1C2=CC=CC=C2C=CC3=CC=CC=C13", "category": "Anticonvulsant"},
        {"id": "DB00175", "name": "Pravastatin", "smiles": "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C12", "category": "Statin"},
        {"id": "DB00864", "name": "Tacrolimus", "smiles": "COC1CC(CCC1OC)CC(C)C2CC(=O)C(C(C(CC(C(C(C(=O)C(C(=O)OC(C(C(CC(=O)C(C=C2C)C)O)OC)C(C)CC=CC)C)O)OC)C)C)O)C", "category": "Immunosuppressant"},
        {"id": "DB00661", "name": "Verapamil", "smiles": "COC1=CC=C(CCN(C)CCCC(C#N)(C(C)C)C2=CC(OC)=C(OC)C=C2)C=C1OC", "category": "Calcium Channel Blocker"},
        {"id": "DB00999", "name": "Hydrochlorothiazide", "smiles": "NS(=O)(=O)C1=CC2=C(NCNS2(=O)=O)C=C1Cl", "category": "Diuretic"},
        {"id": "DB01236", "name": "Sevoflurane", "smiles": "CC(OC(F)(F)F)C(F)(F)F", "category": "Anesthetic"},
        {"id": "DB00176", "name": "Fluvoxamine", "smiles": "COCCCC/C(=N\\OCCN)C1=CC=C(C=C1)C(F)(F)F", "category": "SSRI"},
        {"id": "DB01174", "name": "Phenobarbital", "smiles": "CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2", "category": "Barbiturate"},
        {"id": "DB00252", "name": "Phenytoin", "smiles": "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3", "category": "Anticonvulsant"},
        {"id": "DB01reas1032", "name": "Clopidogrel", "smiles": "COC(=O)C(C1=CC=CS1)C2=C(Cl)C=CC=C2N3CCC4=CC=CC=C4C3", "category": "Antiplatelet"},
    ]
    
    # Sample known interactions (from literature)
    SAMPLE_INTERACTIONS = [
        {"drug1": "DB00945", "drug2": "DB00682", "severity": "severe", 
         "mechanism": "Aspirin inhibits platelet aggregation and displaces warfarin from plasma proteins, significantly increasing bleeding risk",
         "affected_systems": ["cardiovascular", "hematologic"]},
        {"drug1": "DB00682", "drug2": "DB01050", "severity": "severe",
         "mechanism": "NSAIDs inhibit platelet function and may displace warfarin from protein binding sites",
         "affected_systems": ["cardiovascular", "hematologic", "gastrointestinal"]},
        {"drug1": "DB00390", "drug2": "DB01118", "severity": "severe",
         "mechanism": "Amiodarone inhibits P-glycoprotein, reducing digoxin clearance and increasing serum levels",
         "affected_systems": ["cardiovascular"]},
        {"drug1": "DB00390", "drug2": "DB00661", "severity": "moderate",
         "mechanism": "Verapamil reduces renal and nonrenal clearance of digoxin",
         "affected_systems": ["cardiovascular"]},
        {"id": "DB00641", "drug2": "DB00176", "severity": "severe",
         "mechanism": "Fluvoxamine inhibits CYP3A4, dramatically increasing statin levels and risk of rhabdomyolysis",
         "affected_systems": ["musculoskeletal"]},
        {"drug1": "DB00682", "drug2": "DB00564", "severity": "moderate",
         "mechanism": "Carbamazepine induces CYP2C9, increasing warfarin metabolism and reducing anticoagulation",
         "affected_systems": ["hematologic"]},
        {"drug1": "DB00571", "drug2": "DB00661", "severity": "moderate",
         "mechanism": "Both drugs slow cardiac conduction; combination may cause severe bradycardia or AV block",
         "affected_systems": ["cardiovascular"]},
        {"drug1": "DB00864", "drug2": "DB00176", "severity": "severe",
         "mechanism": "Fluvoxamine inhibits CYP3A4, significantly increasing tacrolimus levels and nephrotoxicity risk",
         "affected_systems": ["renal", "immune"]},
        {"drug1": "DB00252", "drug2": "DB00682", "severity": "moderate",
         "mechanism": "Phenytoin induces warfarin metabolism via CYP2C9 and CYP3A4 induction",
         "affected_systems": ["hematologic"]},
        {"drug1": "DB01118", "drug2": "DB00641", "severity": "severe",
         "mechanism": "Amiodarone inhibits CYP3A4, dramatically increasing simvastatin levels",
         "affected_systems": ["musculoskeletal", "hepatic"]},
    ]
    
    # Sample targets (proteins/enzymes)
    SAMPLE_TARGETS = [
        {"id": "P23219", "name": "Prostaglandin G/H synthase 1", "gene": "PTGS1"},  # COX-1
        {"id": "P35354", "name": "Prostaglandin G/H synthase 2", "gene": "PTGS2"},  # COX-2
        {"id": "P11712", "name": "Cytochrome P450 2C9", "gene": "CYP2C9"},
        {"id": "P08684", "name": "Cytochrome P450 3A4", "gene": "CYP3A4"},
        {"id": "P08183", "name": "Multidrug resistance protein 1", "gene": "ABCB1"},  # P-gp
        {"id": "P04798", "name": "Cytochrome P450 1A1", "gene": "CYP1A1"},
        {"id": "Q9UNQ0", "name": "ATP-binding cassette sub-family G member 2", "gene": "ABCG2"},
        {"id": "P00734", "name": "Prothrombin", "gene": "F2"},
        {"id": "P00748", "name": "Coagulation factor XII", "gene": "F12"},
        {"id": "P06276", "name": "Cholinesterase", "gene": "BCHE"},
    ]
    
    # Drug-target relationships
    SAMPLE_DRUG_TARGETS = [
        {"drug": "DB00945", "target": "P23219", "action": "inhibitor"},
        {"drug": "DB00945", "target": "P35354", "action": "inhibitor"},
        {"drug": "DB01050", "target": "P23219", "action": "inhibitor"},
        {"drug": "DB01050", "target": "P35354", "action": "inhibitor"},
        {"drug": "DB00682", "target": "P11712", "action": "substrate"},
        {"drug": "DB00641", "target": "P08684", "action": "substrate"},
        {"drug": "DB00390", "target": "P08183", "action": "substrate"},
        {"drug": "DB01118", "target": "P08684", "action": "inhibitor"},
        {"drug": "DB00176", "target": "P08684", "action": "inhibitor"},
        {"drug": "DB00864", "target": "P08684", "action": "substrate"},
    ]
    
    def __init__(self):
        self.kg = KnowledgeGraphService
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def check_neo4j_connection(self) -> bool:
        """Verify Neo4j is accessible"""
        if self.kg.is_connected():
            logger.info("✓ Neo4j connection successful")
            return True
        else:
            logger.error("✗ Cannot connect to Neo4j. Please ensure Neo4j is running.")
            return False
    
    def initialize_schema(self):
        """Create graph schema (constraints and indexes)"""
        logger.info("Creating Neo4j schema...")
        self.kg.create_schema()
        logger.info("✓ Schema created")
    
    def load_sample_data(self):
        """Load sample drug data for demo/testing"""
        logger.info("Loading sample drug data...")
        
        # Load drugs
        for drug in self.SAMPLE_DRUGS:
            self.kg.add_drug(
                drugbank_id=drug['id'],
                name=drug['name'],
                smiles=drug.get('smiles'),
                category=drug.get('category')
            )
        logger.info(f"✓ Loaded {len(self.SAMPLE_DRUGS)} drugs")
        
        # Load targets
        for target in self.SAMPLE_TARGETS:
            self.kg.add_target(
                uniprot_id=target['id'],
                name=target['name'],
                gene_name=target.get('gene')
            )
        logger.info(f"✓ Loaded {len(self.SAMPLE_TARGETS)} targets")
        
        # Load drug-target relationships
        for rel in self.SAMPLE_DRUG_TARGETS:
            self.kg.link_drug_target(
                drug_id=rel['drug'],
                target_id=rel['target'],
                action=rel.get('action')
            )
        logger.info(f"✓ Loaded {len(self.SAMPLE_DRUG_TARGETS)} drug-target links")
        
        # Load interactions
        for interaction in self.SAMPLE_INTERACTIONS:
            drug1 = interaction.get('drug1')
            drug2 = interaction.get('drug2')
            if drug1 and drug2:
                self.kg.add_interaction(
                    drug1_id=drug1,
                    drug2_id=drug2,
                    severity=interaction['severity'],
                    mechanism=interaction.get('mechanism')
                )
        logger.info(f"✓ Loaded {len(self.SAMPLE_INTERACTIONS)} interactions")
        
        # Print stats
        stats = self.kg.get_stats()
        logger.info(f"Knowledge Graph Stats: {stats}")
    
    def download_twosides(self) -> Optional[Path]:
        """Download TWOSIDES drug interaction dataset"""
        url = self.DATA_SOURCES['twosides']['url']
        output_path = DATA_DIR / 'twosides.csv.gz'
        
        if output_path.exists():
            logger.info(f"TWOSIDES already downloaded: {output_path}")
            return output_path
        
        logger.info("Downloading TWOSIDES dataset (this may take a while)...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✓ Downloaded TWOSIDES to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download TWOSIDES: {e}")
            return None
    
    def load_twosides(self, limit: int = None):
        """Load TWOSIDES data into Neo4j"""
        import gzip
        
        filepath = DATA_DIR / 'twosides.csv.gz'
        if not filepath.exists():
            filepath = self.download_twosides()
            if not filepath:
                logger.warning("TWOSIDES not available, using sample data")
                return
        
        logger.info("Loading TWOSIDES into Neo4j...")
        count = 0
        
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit and count >= limit:
                    break
                
                # TWOSIDES format: drug1, drug2, event, ...
                drug1 = row.get('drug1', row.get('stitch_id1', ''))
                drug2 = row.get('drug2', row.get('stitch_id2', ''))
                
                if drug1 and drug2:
                    # Ensure drugs exist
                    self.kg.add_drug(drugbank_id=drug1, name=drug1)
                    self.kg.add_drug(drugbank_id=drug2, name=drug2)
                    
                    # Add interaction (infer severity from event type)
                    self.kg.add_interaction(
                        drug1_id=drug1,
                        drug2_id=drug2,
                        severity='moderate',  # TWOSIDES doesn't have severity
                        description=row.get('event', '')
                    )
                    count += 1
                    
                    if count % 1000 == 0:
                        logger.info(f"Loaded {count} interactions...")
        
        logger.info(f"✓ Loaded {count} interactions from TWOSIDES")
    
    def run_full_ingestion(self, use_sample: bool = True, twosides_limit: int = 5000):
        """Run the full data ingestion pipeline"""
        logger.info("=" * 50)
        logger.info("Starting DDI Knowledge Graph Ingestion")
        logger.info("=" * 50)
        
        # Check connection
        if not self.check_neo4j_connection():
            logger.error("Aborting: Neo4j not available")
            return False
        
        # Initialize schema
        self.initialize_schema()
        
        # Load data
        if use_sample:
            self.load_sample_data()
        else:
            self.load_twosides(limit=twosides_limit)
        
        # Final stats
        stats = self.kg.get_stats()
        logger.info("=" * 50)
        logger.info("Ingestion Complete!")
        logger.info(f"  Drugs: {stats.get('drug_count', 0)}")
        logger.info(f"  Targets: {stats.get('target_count', 0)}")
        logger.info(f"  Side Effects: {stats.get('side_effect_count', 0)}")
        logger.info(f"  Interactions: {stats.get('interaction_count', 0)}")
        logger.info("=" * 50)
        
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DDI Knowledge Graph Ingestion')
    parser.add_argument('--sample', action='store_true', default=True,
                       help='Use sample data (default: True)')
    parser.add_argument('--full', action='store_true',
                       help='Download and load full TWOSIDES dataset')
    parser.add_argument('--limit', type=int, default=5000,
                       help='Limit number of interactions to load')
    
    args = parser.parse_args()
    
    ingestion = DrugDataIngestion()
    ingestion.run_full_ingestion(
        use_sample=not args.full,
        twosides_limit=args.limit
    )
