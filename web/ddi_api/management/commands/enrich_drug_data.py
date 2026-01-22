"""
Enrich Drug Data Management Command

This command enriches the Neo4j knowledge graph with:
1. SMILES structures from PubChem API
2. Additional drug interactions from DrugBank/public sources
3. Drug descriptions and therapeutic classes
4. Molecular properties (weight, formula, etc.)

Usage:
    python manage.py enrich_drug_data --smiles          # Fetch SMILES only
    python manage.py enrich_drug_data --interactions    # Add interactions only
    python manage.py enrich_drug_data --descriptions    # Fetch drug descriptions
    python manage.py enrich_drug_data --all             # Do everything
"""

import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


# Therapeutic drug classes mapping (ATC-based)
THERAPEUTIC_CLASSES = {
    # Cardiovascular
    "warfarin": "Anticoagulant", "heparin": "Anticoagulant", "enoxaparin": "Anticoagulant",
    "rivaroxaban": "Anticoagulant", "apixaban": "Anticoagulant", "dabigatran": "Anticoagulant",
    "aspirin": "Antiplatelet/NSAID", "clopidogrel": "Antiplatelet", "ticagrelor": "Antiplatelet",
    "atorvastatin": "Statin", "simvastatin": "Statin", "rosuvastatin": "Statin", "pravastatin": "Statin",
    "lovastatin": "Statin", "fluvastatin": "Statin", "pitavastatin": "Statin",
    "lisinopril": "ACE Inhibitor", "enalapril": "ACE Inhibitor", "ramipril": "ACE Inhibitor",
    "losartan": "ARB", "valsartan": "ARB", "irbesartan": "ARB", "olmesartan": "ARB",
    "amlodipine": "Calcium Channel Blocker", "diltiazem": "Calcium Channel Blocker", "verapamil": "Calcium Channel Blocker",
    "metoprolol": "Beta Blocker", "atenolol": "Beta Blocker", "propranolol": "Beta Blocker", "carvedilol": "Beta Blocker",
    "digoxin": "Cardiac Glycoside", "amiodarone": "Antiarrhythmic", "sotalol": "Antiarrhythmic",
    "furosemide": "Diuretic", "hydrochlorothiazide": "Diuretic", "spironolactone": "Diuretic",
    
    # CNS
    "fluoxetine": "SSRI Antidepressant", "sertraline": "SSRI Antidepressant", "paroxetine": "SSRI Antidepressant",
    "citalopram": "SSRI Antidepressant", "escitalopram": "SSRI Antidepressant",
    "venlafaxine": "SNRI Antidepressant", "duloxetine": "SNRI Antidepressant",
    "amitriptyline": "Tricyclic Antidepressant", "nortriptyline": "Tricyclic Antidepressant",
    "bupropion": "Atypical Antidepressant", "mirtazapine": "Atypical Antidepressant",
    "alprazolam": "Benzodiazepine", "lorazepam": "Benzodiazepine", "diazepam": "Benzodiazepine",
    "clonazepam": "Benzodiazepine", "midazolam": "Benzodiazepine",
    "zolpidem": "Sedative-Hypnotic", "eszopiclone": "Sedative-Hypnotic",
    "quetiapine": "Atypical Antipsychotic", "risperidone": "Atypical Antipsychotic", "olanzapine": "Atypical Antipsychotic",
    "aripiprazole": "Atypical Antipsychotic", "haloperidol": "Typical Antipsychotic",
    "gabapentin": "Anticonvulsant", "pregabalin": "Anticonvulsant", "levetiracetam": "Anticonvulsant",
    "phenytoin": "Anticonvulsant", "carbamazepine": "Anticonvulsant", "valproic acid": "Anticonvulsant",
    "lithium": "Mood Stabilizer",
    
    # Pain
    "ibuprofen": "NSAID", "naproxen": "NSAID", "meloxicam": "NSAID", "celecoxib": "COX-2 Inhibitor",
    "acetaminophen": "Analgesic", "tramadol": "Opioid Analgesic",
    "morphine": "Opioid Analgesic", "oxycodone": "Opioid Analgesic", "hydrocodone": "Opioid Analgesic",
    "fentanyl": "Opioid Analgesic", "codeine": "Opioid Analgesic", "methadone": "Opioid Analgesic",
    
    # Antibiotics
    "amoxicillin": "Penicillin Antibiotic", "ampicillin": "Penicillin Antibiotic", "penicillin": "Penicillin Antibiotic",
    "azithromycin": "Macrolide Antibiotic", "clarithromycin": "Macrolide Antibiotic", "erythromycin": "Macrolide Antibiotic",
    "ciprofloxacin": "Fluoroquinolone Antibiotic", "levofloxacin": "Fluoroquinolone Antibiotic",
    "doxycycline": "Tetracycline Antibiotic", "tetracycline": "Tetracycline Antibiotic",
    "metronidazole": "Nitroimidazole Antibiotic", "vancomycin": "Glycopeptide Antibiotic",
    "sulfamethoxazole": "Sulfonamide Antibiotic", "trimethoprim": "Antibiotic",
    "linezolid": "Oxazolidinone Antibiotic", "clindamycin": "Lincosamide Antibiotic",
    
    # Antifungals
    "fluconazole": "Azole Antifungal", "itraconazole": "Azole Antifungal", "ketoconazole": "Azole Antifungal",
    "voriconazole": "Azole Antifungal", "amphotericin": "Antifungal",
    
    # Antivirals
    "acyclovir": "Antiviral", "valacyclovir": "Antiviral", "oseltamivir": "Antiviral",
    "ritonavir": "Protease Inhibitor", "lopinavir": "Protease Inhibitor",
    
    # Diabetes
    "metformin": "Biguanide Antidiabetic", "glipizide": "Sulfonylurea", "glyburide": "Sulfonylurea",
    "glimepiride": "Sulfonylurea", "sitagliptin": "DPP-4 Inhibitor", "linagliptin": "DPP-4 Inhibitor",
    "empagliflozin": "SGLT2 Inhibitor", "dapagliflozin": "SGLT2 Inhibitor",
    "insulin": "Insulin", "liraglutide": "GLP-1 Agonist", "semaglutide": "GLP-1 Agonist",
    
    # GI
    "omeprazole": "Proton Pump Inhibitor", "pantoprazole": "Proton Pump Inhibitor", "esomeprazole": "Proton Pump Inhibitor",
    "lansoprazole": "Proton Pump Inhibitor", "ranitidine": "H2 Blocker", "famotidine": "H2 Blocker",
    "ondansetron": "Antiemetic", "metoclopramide": "Prokinetic",
    
    # Respiratory
    "albuterol": "Beta-2 Agonist", "salmeterol": "Beta-2 Agonist", "formoterol": "Beta-2 Agonist",
    "fluticasone": "Inhaled Corticosteroid", "budesonide": "Inhaled Corticosteroid",
    "montelukast": "Leukotriene Inhibitor", "tiotropium": "Anticholinergic Bronchodilator",
    "prednisone": "Corticosteroid", "prednisolone": "Corticosteroid", "dexamethasone": "Corticosteroid",
    
    # Immunosuppressants
    "tacrolimus": "Calcineurin Inhibitor", "cyclosporine": "Calcineurin Inhibitor",
    "mycophenolate": "Immunosuppressant", "azathioprine": "Immunosuppressant",
    
    # Thyroid
    "levothyroxine": "Thyroid Hormone", "methimazole": "Antithyroid",
    
    # Oncology
    "methotrexate": "Antimetabolite", "cyclophosphamide": "Alkylating Agent",
    "tamoxifen": "Selective Estrogen Receptor Modulator",
}


class PubChemAPI:
    """
    PubChem PUG REST API client for fetching SMILES.
    
    PubChem is a free, public database with millions of compounds.
    Rate limit: ~5 requests/second recommended.
    """
    
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def __init__(self, delay: float = 0.25):
        self.delay = delay  # Delay between requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ProjectAegis-DDI/1.0 (Drug Interaction Research)'
        })
    
    def get_smiles(self, drug_name: str) -> Optional[str]:
        """
        Fetch canonical SMILES for a drug from PubChem.
        
        Args:
            drug_name: Drug name to search
            
        Returns:
            SMILES string or None if not found
        """
        try:
            # Search by name to get compound ID - request multiple SMILES formats
            url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(drug_name)}/property/CanonicalSMILES,IsomericSMILES/JSON"
            
            response = self.session.get(url, timeout=10)
            time.sleep(self.delay)  # Rate limiting
            
            if response.status_code == 200:
                data = response.json()
                properties = data.get('PropertyTable', {}).get('Properties', [])
                if properties:
                    prop = properties[0]
                    # PubChem returns different keys depending on the request
                    smiles = (prop.get('CanonicalSMILES') or 
                              prop.get('IsomericSMILES') or 
                              prop.get('SMILES') or 
                              prop.get('ConnectivitySMILES') or '')
                    if smiles and len(smiles) > 3:
                        return smiles
            elif response.status_code == 404:
                # Try alternative names
                return self._try_alternative_names(drug_name)
                
        except Exception as e:
            logger.debug(f"PubChem lookup failed for {drug_name}: {e}")
        
        return None
    
    def _try_alternative_names(self, drug_name: str) -> Optional[str]:
        """Try alternative drug name formats."""
        alternatives = []
        
        # Remove common suffixes
        for suffix in [' hydrochloride', ' sodium', ' acetate', ' sulfate', ' maleate', 
                       ' citrate', ' fumarate', ' tartrate', ' mesylate', ' hcl']:
            if drug_name.lower().endswith(suffix):
                alternatives.append(drug_name[:-len(suffix)].strip())
        
        # Try without spaces/hyphens
        if ' ' in drug_name or '-' in drug_name:
            alternatives.append(drug_name.replace(' ', '').replace('-', ''))
        
        for alt in alternatives:
            try:
                url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(alt)}/property/CanonicalSMILES,IsomericSMILES/JSON"
                response = self.session.get(url, timeout=10)
                time.sleep(self.delay)
                
                if response.status_code == 200:
                    data = response.json()
                    properties = data.get('PropertyTable', {}).get('Properties', [])
                    if properties:
                        prop = properties[0]
                        smiles = (prop.get('CanonicalSMILES') or 
                                  prop.get('IsomericSMILES') or 
                                  prop.get('SMILES') or 
                                  prop.get('ConnectivitySMILES') or '')
                        if smiles and len(smiles) > 3:
                            return smiles
            except:
                pass
        
        return None
    
    def batch_get_smiles(self, drug_names: List[str], callback=None) -> Dict[str, str]:
        """
        Fetch SMILES for multiple drugs.
        
        Args:
            drug_names: List of drug names
            callback: Optional callback(current, total, drug_name, success)
            
        Returns:
            Dict mapping drug name to SMILES
        """
        results = {}
        total = len(drug_names)
        
        for i, name in enumerate(drug_names):
            smiles = self.get_smiles(name)
            if smiles:
                results[name] = smiles
            
            if callback:
                callback(i + 1, total, name, smiles is not None)
        
        return results
    
    def get_drug_properties(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive drug properties from PubChem.
        
        Returns dict with: molecular_weight, formula, description, synonyms
        """
        try:
            # First get the CID
            search_url = f"{self.BASE_URL}/compound/name/{requests.utils.quote(drug_name)}/cids/JSON"
            response = self.session.get(search_url, timeout=10)
            time.sleep(self.delay)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if not cids:
                return None
            
            cid = cids[0]
            
            # Fetch properties
            props_url = f"{self.BASE_URL}/compound/cid/{cid}/property/MolecularWeight,MolecularFormula,IUPACName/JSON"
            props_response = self.session.get(props_url, timeout=10)
            time.sleep(self.delay)
            
            result = {'cid': cid}
            
            if props_response.status_code == 200:
                props_data = props_response.json()
                properties = props_data.get('PropertyTable', {}).get('Properties', [])
                if properties:
                    prop = properties[0]
                    result['molecular_weight'] = prop.get('MolecularWeight')
                    result['molecular_formula'] = prop.get('MolecularFormula')
                    result['iupac_name'] = prop.get('IUPACName')
            
            # Fetch description
            desc_url = f"{self.BASE_URL}/compound/cid/{cid}/description/JSON"
            desc_response = self.session.get(desc_url, timeout=10)
            time.sleep(self.delay)
            
            if desc_response.status_code == 200:
                desc_data = desc_response.json()
                informations = desc_data.get('InformationList', {}).get('Information', [])
                for info in informations:
                    desc = info.get('Description')
                    if desc and len(desc) > 20:
                        result['description'] = desc[:500]  # Limit length
                        result['description_source'] = info.get('DescriptionSourceName', 'PubChem')
                        break
            
            return result if len(result) > 1 else None
            
        except Exception as e:
            logger.debug(f"PubChem properties lookup failed for {drug_name}: {e}")
            return None
    
    def batch_get_properties(self, drug_names: List[str], callback=None) -> Dict[str, Dict]:
        """
        Fetch properties for multiple drugs.
        """
        results = {}
        total = len(drug_names)
        
        for i, name in enumerate(drug_names):
            props = self.get_drug_properties(name)
            if props:
                results[name] = props
            
            if callback:
                callback(i + 1, total, name, props is not None)
        
        return results


class DrugInteractionEnricher:
    """
    Fetches and imports additional drug interactions from public sources.
    """
    
    # Common high-risk drug interactions (clinically validated)
    # Source: Clinical pharmacology textbooks and FDA guidance
    KNOWN_INTERACTIONS = [
        # Anticoagulants
        ("Warfarin", "Aspirin", "severe", "Increased risk of bleeding due to additive anticoagulant effects"),
        ("Warfarin", "Ibuprofen", "severe", "NSAIDs increase bleeding risk and may affect warfarin metabolism"),
        ("Warfarin", "Vitamin K", "major", "Vitamin K antagonizes warfarin anticoagulant effect"),
        ("Warfarin", "Amiodarone", "severe", "Amiodarone inhibits CYP2C9, increasing warfarin levels"),
        ("Warfarin", "Fluconazole", "severe", "Fluconazole inhibits CYP2C9, increasing warfarin effect"),
        ("Warfarin", "Metronidazole", "major", "Metronidazole inhibits warfarin metabolism"),
        ("Warfarin", "Ciprofloxacin", "major", "Ciprofloxacin may increase anticoagulant effect"),
        
        # Statins
        ("Simvastatin", "Amiodarone", "severe", "Increased risk of myopathy/rhabdomyolysis"),
        ("Simvastatin", "Clarithromycin", "severe", "CYP3A4 inhibition increases statin toxicity"),
        ("Simvastatin", "Erythromycin", "severe", "CYP3A4 inhibition increases statin toxicity"),
        ("Simvastatin", "Itraconazole", "severe", "CYP3A4 inhibition increases statin toxicity"),
        ("Simvastatin", "Grapefruit juice", "major", "CYP3A4 inhibition increases statin levels"),
        ("Atorvastatin", "Clarithromycin", "major", "CYP3A4 inhibition increases statin levels"),
        ("Lovastatin", "Itraconazole", "severe", "Contraindicated - rhabdomyolysis risk"),
        
        # Antidepressants (Serotonin Syndrome risk)
        ("Fluoxetine", "Tramadol", "severe", "Risk of serotonin syndrome and seizures"),
        ("Sertraline", "Tramadol", "severe", "Risk of serotonin syndrome"),
        ("Paroxetine", "Tramadol", "severe", "Risk of serotonin syndrome"),
        ("Fluoxetine", "MAOIs", "severe", "Contraindicated - serotonin syndrome risk"),
        ("Sertraline", "MAOIs", "severe", "Contraindicated - serotonin syndrome risk"),
        ("Fluoxetine", "Linezolid", "severe", "Linezolid has MAOI activity - serotonin syndrome"),
        
        # Cardiovascular
        ("Digoxin", "Amiodarone", "severe", "Amiodarone increases digoxin levels"),
        ("Digoxin", "Verapamil", "major", "Verapamil increases digoxin levels"),
        ("Digoxin", "Quinidine", "severe", "Quinidine increases digoxin levels significantly"),
        ("Metoprolol", "Verapamil", "major", "Enhanced bradycardia and hypotension"),
        ("Diltiazem", "Metoprolol", "major", "Enhanced bradycardia and AV block risk"),
        ("Amlodipine", "Simvastatin", "major", "Amlodipine increases simvastatin exposure"),
        
        # QT Prolongation
        ("Amiodarone", "Haloperidol", "severe", "Additive QT prolongation - torsades risk"),
        ("Amiodarone", "Methadone", "severe", "Additive QT prolongation"),
        ("Fluconazole", "Haloperidol", "major", "Additive QT prolongation"),
        ("Clarithromycin", "Haloperidol", "major", "Additive QT prolongation"),
        ("Erythromycin", "Cisapride", "severe", "Contraindicated - fatal arrhythmias"),
        
        # Diabetes
        ("Metformin", "Contrast media", "major", "Risk of lactic acidosis"),
        ("Glipizide", "Fluconazole", "major", "Increased hypoglycemia risk"),
        ("Glyburide", "Fluconazole", "major", "Increased hypoglycemia risk"),
        
        # CNS Depressants
        ("Oxycodone", "Benzodiazepines", "severe", "Respiratory depression risk - FDA black box"),
        ("Morphine", "Benzodiazepines", "severe", "Respiratory depression risk"),
        ("Fentanyl", "Benzodiazepines", "severe", "Respiratory depression risk"),
        ("Hydrocodone", "Alprazolam", "severe", "Respiratory depression risk"),
        ("Codeine", "Diazepam", "major", "Enhanced CNS depression"),
        
        # Antibiotics
        ("Ciprofloxacin", "Theophylline", "major", "Ciprofloxacin inhibits theophylline metabolism"),
        ("Ciprofloxacin", "Tizanidine", "severe", "Contraindicated - increased tizanidine toxicity"),
        ("Metronidazole", "Alcohol", "severe", "Disulfiram-like reaction"),
        ("Tetracycline", "Antacids", "major", "Reduced tetracycline absorption"),
        ("Ciprofloxacin", "Antacids", "major", "Reduced fluoroquinolone absorption"),
        
        # Potassium
        ("Spironolactone", "Potassium supplements", "severe", "Hyperkalemia risk"),
        ("Lisinopril", "Potassium supplements", "major", "Hyperkalemia risk"),
        ("Enalapril", "Spironolactone", "major", "Hyperkalemia risk"),
        
        # Immunosuppressants
        ("Cyclosporine", "Ketoconazole", "major", "Increased cyclosporine levels"),
        ("Tacrolimus", "Fluconazole", "major", "Increased tacrolimus levels"),
        ("Mycophenolate", "Acyclovir", "moderate", "Competition for renal tubular secretion"),
        
        # Seizure medications
        ("Phenytoin", "Fluconazole", "major", "Increased phenytoin levels"),
        ("Carbamazepine", "Erythromycin", "major", "Increased carbamazepine toxicity"),
        ("Valproic acid", "Carbamazepine", "major", "Complex interaction - levels altered"),
        
        # Thyroid
        ("Levothyroxine", "Calcium carbonate", "major", "Reduced levothyroxine absorption"),
        ("Levothyroxine", "Iron supplements", "major", "Reduced levothyroxine absorption"),
        ("Levothyroxine", "Omeprazole", "moderate", "May reduce levothyroxine absorption"),
        
        # Proton Pump Inhibitors
        ("Omeprazole", "Clopidogrel", "major", "Reduced clopidogrel antiplatelet effect"),
        ("Esomeprazole", "Clopidogrel", "major", "Reduced clopidogrel antiplatelet effect"),
        
        # Lithium
        ("Lithium", "Ibuprofen", "major", "NSAIDs increase lithium levels"),
        ("Lithium", "Lisinopril", "major", "ACE inhibitors increase lithium levels"),
        ("Lithium", "Hydrochlorothiazide", "major", "Thiazides increase lithium levels"),
        
        # Methotrexate
        ("Methotrexate", "NSAIDs", "severe", "Reduced methotrexate clearance - toxicity"),
        ("Methotrexate", "Trimethoprim", "severe", "Bone marrow suppression"),
        ("Methotrexate", "Probenecid", "major", "Reduced methotrexate excretion"),
    ]
    
    def get_interactions(self) -> List[Tuple[str, str, str, str]]:
        """Return list of (drug1, drug2, severity, mechanism) tuples."""
        return self.KNOWN_INTERACTIONS


class Command(BaseCommand):
    help = 'Enrich drug database with SMILES, descriptions, and interactions'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--smiles',
            action='store_true',
            help='Fetch SMILES from PubChem for drugs without structures'
        )
        parser.add_argument(
            '--interactions',
            action='store_true', 
            help='Add known clinical drug interactions'
        )
        parser.add_argument(
            '--descriptions',
            action='store_true',
            help='Fetch drug descriptions from PubChem'
        )
        parser.add_argument(
            '--classes',
            action='store_true',
            help='Add therapeutic drug classes'
        )
        parser.add_argument(
            '--properties',
            action='store_true',
            help='Fetch molecular properties (weight, formula)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Do all enrichment types'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=500,
            help='Maximum number of drugs to process (default: 500)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes'
        )
    
    def handle(self, *args, **options):
        from ddi_api.services.knowledge_graph import KnowledgeGraphService
        
        kg = KnowledgeGraphService
        if not kg.is_connected():
            self.stderr.write(self.style.ERROR('Cannot connect to Neo4j'))
            return
        
        do_smiles = options['smiles'] or options['all']
        do_interactions = options['interactions'] or options['all']
        do_descriptions = options['descriptions'] or options['all']
        do_classes = options['classes'] or options['all']
        do_properties = options['properties'] or options['all']
        dry_run = options['dry_run']
        limit = options['limit']
        
        if not any([do_smiles, do_interactions, do_descriptions, do_classes, do_properties]):
            self.stdout.write(self.style.WARNING('No action specified. Use --smiles, --interactions, --descriptions, --classes, --properties, or --all'))
            return
        
        if do_smiles:
            self.enrich_smiles(kg, limit, dry_run)
        
        if do_classes:
            self.enrich_therapeutic_classes(kg, dry_run)
        
        if do_descriptions:
            self.enrich_descriptions(kg, limit, dry_run)
        
        if do_properties:
            self.enrich_properties(kg, limit, dry_run)
        
        if do_interactions:
            self.enrich_interactions(kg, dry_run)
        
        self.stdout.write(self.style.SUCCESS('\n✅ Enrichment complete!'))
    
    def enrich_smiles(self, kg, limit: int, dry_run: bool):
        """Fetch and add SMILES from PubChem."""
        self.stdout.write(self.style.MIGRATE_HEADING('\n📦 Enriching SMILES from PubChem...'))
        
        # Get drugs without SMILES
        with kg._driver.session() as session:
            result = session.run('''
                MATCH (d:Drug)
                WHERE d.smiles IS NULL OR d.smiles = ''
                RETURN d.name as name, d.id as id
                LIMIT $limit
            ''', limit=limit)
            drugs_without_smiles = [(r['name'], r['id']) for r in result]
        
        self.stdout.write(f'Found {len(drugs_without_smiles)} drugs without SMILES (limit: {limit})')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no changes will be made'))
            for name, _ in drugs_without_smiles[:10]:
                self.stdout.write(f'  Would fetch: {name}')
            if len(drugs_without_smiles) > 10:
                self.stdout.write(f'  ... and {len(drugs_without_smiles) - 10} more')
            return
        
        # Fetch SMILES from PubChem
        pubchem = PubChemAPI(delay=0.3)  # 3 requests/second
        
        success_count = 0
        fail_count = 0
        
        for i, (name, drug_id) in enumerate(drugs_without_smiles):
            # Progress indicator
            if (i + 1) % 10 == 0:
                self.stdout.write(f'  Progress: {i + 1}/{len(drugs_without_smiles)} ({success_count} found)')
            
            smiles = pubchem.get_smiles(name)
            
            if smiles:
                # Update Neo4j
                with kg._driver.session() as session:
                    session.run('''
                        MATCH (d:Drug {name: $name})
                        SET d.smiles = $smiles
                    ''', name=name, smiles=smiles)
                success_count += 1
                self.stdout.write(self.style.SUCCESS(f'  ✓ {name}: {smiles[:50]}...'))
            else:
                fail_count += 1
        
        self.stdout.write(f'\n📊 SMILES enrichment results:')
        self.stdout.write(self.style.SUCCESS(f'  Found: {success_count}'))
        self.stdout.write(f'  Not found: {fail_count}')
    
    def enrich_interactions(self, kg, dry_run: bool):
        """Add known clinical drug interactions."""
        self.stdout.write(self.style.MIGRATE_HEADING('\n💊 Adding known drug interactions...'))
        
        enricher = DrugInteractionEnricher()
        interactions = enricher.get_interactions()
        
        self.stdout.write(f'Have {len(interactions)} clinically validated interactions')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no changes will be made'))
            for drug1, drug2, severity, _ in interactions[:5]:
                self.stdout.write(f'  Would add: {drug1} + {drug2} ({severity})')
            return
        
        added = 0
        updated = 0
        skipped = 0
        
        with kg._driver.session() as session:
            for drug1, drug2, severity, mechanism in interactions:
                # Check if both drugs exist
                result = session.run('''
                    MATCH (d1:Drug), (d2:Drug)
                    WHERE toLower(d1.name) = toLower($drug1) 
                      AND toLower(d2.name) = toLower($drug2)
                    RETURN d1.name as n1, d2.name as n2
                ''', drug1=drug1, drug2=drug2)
                
                record = result.single()
                if not record:
                    # Create drugs if they don't exist
                    session.run('''
                        MERGE (d1:Drug {name: $drug1})
                        ON CREATE SET d1.id = 'ENRICHED_' + replace(toUpper($drug1), ' ', '_')
                        MERGE (d2:Drug {name: $drug2})
                        ON CREATE SET d2.id = 'ENRICHED_' + replace(toUpper($drug2), ' ', '_')
                    ''', drug1=drug1, drug2=drug2)
                
                # Check if interaction exists
                existing = session.run('''
                    MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
                    WHERE toLower(d1.name) = toLower($drug1) 
                      AND toLower(d2.name) = toLower($drug2)
                    RETURN i
                ''', drug1=drug1, drug2=drug2)
                
                if existing.single():
                    # Update existing
                    session.run('''
                        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
                        WHERE toLower(d1.name) = toLower($drug1) 
                          AND toLower(d2.name) = toLower($drug2)
                        SET i.severity = $severity, i.mechanism = $mechanism,
                            i.source = 'clinical_enrichment'
                    ''', drug1=drug1, drug2=drug2, severity=severity, mechanism=mechanism)
                    updated += 1
                else:
                    # Create new interaction
                    session.run('''
                        MATCH (d1:Drug), (d2:Drug)
                        WHERE toLower(d1.name) = toLower($drug1) 
                          AND toLower(d2.name) = toLower($drug2)
                        MERGE (d1)-[i:INTERACTS_WITH]->(d2)
                        SET i.severity = $severity, i.mechanism = $mechanism,
                            i.source = 'clinical_enrichment', i.verified = true
                    ''', drug1=drug1, drug2=drug2, severity=severity, mechanism=mechanism)
                    added += 1
                
                self.stdout.write(f'  ✓ {drug1} ↔ {drug2} ({severity})')
        
        self.stdout.write(f'\n📊 Interaction enrichment results:')
        self.stdout.write(self.style.SUCCESS(f'  Added: {added}'))
        self.stdout.write(f'  Updated: {updated}')
    
    def enrich_therapeutic_classes(self, kg, dry_run: bool):
        """Add therapeutic class labels to drugs."""
        self.stdout.write(self.style.MIGRATE_HEADING('\n🏷️ Adding therapeutic drug classes...'))
        
        self.stdout.write(f'Have {len(THERAPEUTIC_CLASSES)} drug class mappings')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no changes will be made'))
            for drug, drug_class in list(THERAPEUTIC_CLASSES.items())[:5]:
                self.stdout.write(f'  Would add: {drug} → {drug_class}')
            return
        
        updated = 0
        not_found = 0
        
        with kg._driver.session() as session:
            for drug_name, therapeutic_class in THERAPEUTIC_CLASSES.items():
                # Try to find the drug
                result = session.run('''
                    MATCH (d:Drug)
                    WHERE toLower(d.name) = toLower($name)
                    SET d.therapeutic_class = $class
                    RETURN d.name as name
                ''', name=drug_name, **{'class': therapeutic_class})
                
                if result.single():
                    updated += 1
                    self.stdout.write(f'  ✓ {drug_name} → {therapeutic_class}')
                else:
                    not_found += 1
        
        self.stdout.write(f'\n📊 Therapeutic class enrichment results:')
        self.stdout.write(self.style.SUCCESS(f'  Updated: {updated}'))
        self.stdout.write(f'  Not found in DB: {not_found}')
    
    def enrich_descriptions(self, kg, limit: int, dry_run: bool):
        """Fetch and add drug descriptions from PubChem."""
        self.stdout.write(self.style.MIGRATE_HEADING('\n📝 Fetching drug descriptions from PubChem...'))
        
        # Get drugs without descriptions
        with kg._driver.session() as session:
            result = session.run('''
                MATCH (d:Drug)
                WHERE (d.description IS NULL OR d.description = '')
                  AND d.smiles IS NOT NULL
                RETURN d.name as name
                LIMIT $limit
            ''', limit=limit)
            drugs_without_desc = [r['name'] for r in result]
        
        self.stdout.write(f'Found {len(drugs_without_desc)} drugs without descriptions (limit: {limit})')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no changes will be made'))
            for name in drugs_without_desc[:10]:
                self.stdout.write(f'  Would fetch: {name}')
            return
        
        pubchem = PubChemAPI(delay=0.5)  # Slower for description lookups
        
        success_count = 0
        fail_count = 0
        
        for i, name in enumerate(drugs_without_desc):
            if (i + 1) % 10 == 0:
                self.stdout.write(f'  Progress: {i + 1}/{len(drugs_without_desc)} ({success_count} found)')
            
            props = pubchem.get_drug_properties(name)
            
            if props and props.get('description'):
                with kg._driver.session() as session:
                    session.run('''
                        MATCH (d:Drug {name: $name})
                        SET d.description = $description,
                            d.description_source = $source
                    ''', name=name, 
                        description=props['description'],
                        source=props.get('description_source', 'PubChem'))
                success_count += 1
                self.stdout.write(self.style.SUCCESS(f'  ✓ {name}: {props["description"][:60]}...'))
            else:
                fail_count += 1
        
        self.stdout.write(f'\n📊 Description enrichment results:')
        self.stdout.write(self.style.SUCCESS(f'  Found: {success_count}'))
        self.stdout.write(f'  Not found: {fail_count}')
    
    def enrich_properties(self, kg, limit: int, dry_run: bool):
        """Fetch and add molecular properties from PubChem."""
        self.stdout.write(self.style.MIGRATE_HEADING('\n⚗️ Fetching molecular properties from PubChem...'))
        
        # Get drugs with SMILES but no molecular weight
        with kg._driver.session() as session:
            result = session.run('''
                MATCH (d:Drug)
                WHERE d.smiles IS NOT NULL 
                  AND (d.molecular_weight IS NULL OR d.molecular_formula IS NULL)
                RETURN d.name as name
                LIMIT $limit
            ''', limit=limit)
            drugs_without_props = [r['name'] for r in result]
        
        self.stdout.write(f'Found {len(drugs_without_props)} drugs without molecular properties (limit: {limit})')
        
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - no changes will be made'))
            for name in drugs_without_props[:10]:
                self.stdout.write(f'  Would fetch: {name}')
            return
        
        pubchem = PubChemAPI(delay=0.4)
        
        success_count = 0
        fail_count = 0
        
        for i, name in enumerate(drugs_without_props):
            if (i + 1) % 10 == 0:
                self.stdout.write(f'  Progress: {i + 1}/{len(drugs_without_props)} ({success_count} found)')
            
            props = pubchem.get_drug_properties(name)
            
            if props and (props.get('molecular_weight') or props.get('molecular_formula')):
                with kg._driver.session() as session:
                    session.run('''
                        MATCH (d:Drug {name: $name})
                        SET d.molecular_weight = $weight,
                            d.molecular_formula = $formula,
                            d.iupac_name = $iupac,
                            d.pubchem_cid = $cid
                    ''', name=name,
                        weight=props.get('molecular_weight'),
                        formula=props.get('molecular_formula'),
                        iupac=props.get('iupac_name'),
                        cid=props.get('cid'))
                success_count += 1
                self.stdout.write(self.style.SUCCESS(f'  ✓ {name}: MW={props.get("molecular_weight")} {props.get("molecular_formula", "")}'))
            else:
                fail_count += 1
        
        self.stdout.write(f'\n📊 Molecular properties enrichment results:')
        self.stdout.write(self.style.SUCCESS(f'  Found: {success_count}'))
        self.stdout.write(f'  Not found: {fail_count}')
