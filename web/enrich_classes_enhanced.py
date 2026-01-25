"""
Enhanced therapeutic class enrichment using multiple sources:
1. Pattern matching (suffix-based)
2. Known drug mappings (curated list)
3. RxNorm API (free, no key needed)
4. PubChem API (free, no key needed)
"""
from neo4j import GraphDatabase
import requests
import time
import re

# Neo4j Aura connection
URI = 'neo4j+s://ca47aebc.databases.neo4j.io'
USER = 'neo4j'
PASSWORD = 'BYKmHWoR2DeEiiiwO6qBAET273OIaaGv1ZatYpU_vtM'

# RxNorm API base
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# PubChem API base
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class RxNormAPI:
    """Query RxNorm for drug classification."""
    
    def __init__(self, delay=0.2):
        self.delay = delay
        self.session = requests.Session()
        self.cache = {}
    
    def get_rxcui(self, drug_name: str) -> str:
        """Get RxCUI (RxNorm Concept Unique Identifier) for a drug."""
        if drug_name in self.cache:
            return self.cache[drug_name]
        
        url = f"{RXNORM_BASE}/rxcui.json"
        params = {"name": drug_name, "search": 2}  # search=2 for approximate match
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            rxcui = None
            if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                rxcui = data['idGroup']['rxnormId'][0]
            
            self.cache[drug_name] = rxcui
            return rxcui
            
        except KeyboardInterrupt:
            raise
        except Exception:
            return None
    
    def get_drug_class(self, rxcui: str) -> dict:
        """Get drug class information from RxNorm."""
        if not rxcui:
            return None
        
        url = f"{RXNORM_BASE}/rxclass/class/byRxcui.json"
        params = {"rxcui": rxcui}
        
        try:
            time.sleep(self.delay)
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'rxclassDrugInfoList' not in data:
                return None
            
            drug_info = data['rxclassDrugInfoList'].get('rxclassDrugInfo', [])
            
            # Extract therapeutic classes
            result = {
                'therapeutic_class': None,
                'mechanism': None,
                'category': None,
                'atc_class': None
            }
            
            for info in drug_info:
                class_info = info.get('rxclassMinConceptItem', {})
                class_name = class_info.get('className', '')
                class_type = class_info.get('classType', '')
                
                # ATC classification (Anatomical Therapeutic Chemical)
                if class_type == 'ATC1-4':
                    if not result['atc_class']:
                        result['atc_class'] = class_name
                
                # Mechanism of action
                if class_type == 'MOA':
                    if not result['mechanism']:
                        result['mechanism'] = class_name
                
                # Pharmacologic class
                if class_type == 'PE' or class_type == 'EPC':
                    if not result['therapeutic_class']:
                        result['therapeutic_class'] = class_name
                
                # VA class
                if class_type == 'VA':
                    if not result['category']:
                        result['category'] = class_name
            
            return result if any(result.values()) else None
            
        except Exception:
            return None


class PubChemClassifier:
    """Get drug classification from PubChem."""
    
    def __init__(self, delay=0.25):
        self.delay = delay
        self.session = requests.Session()
    
    def get_classification(self, drug_name: str) -> dict:
        """Get drug classification from PubChem."""
        try:
            # First get the CID
            url = f"{PUBCHEM_BASE}/compound/name/{drug_name}/cids/JSON"
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            cid = data.get('IdentifierList', {}).get('CID', [None])[0]
            
            if not cid:
                return None
            
            # Get classification
            url = f"{PUBCHEM_BASE}/compound/cid/{cid}/classification/JSON"
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            hierarchies = data.get('Hierarchies', {}).get('Hierarchy', [])
            
            result = {
                'therapeutic_class': None,
                'category': None
            }
            
            for hierarchy in hierarchies:
                source = hierarchy.get('SourceName', '')
                
                # Look for ChEBI or MeSH classifications
                if source in ['ChEBI', 'MeSH']:
                    nodes = hierarchy.get('Node', [])
                    for node in nodes:
                        info = node.get('Information', {})
                        name = info.get('Name', '')
                        
                        # Look for pharmacological terms
                        if any(term in name.lower() for term in [
                            'agent', 'drug', 'inhibitor', 'blocker', 'agonist',
                            'antagonist', 'anti', 'therapeutic'
                        ]):
                            if not result['therapeutic_class']:
                                result['therapeutic_class'] = name
                            break
            
            return result if result['therapeutic_class'] else None
            
        except Exception:
            return None


# Extended pattern matching
DRUG_CLASS_PATTERNS = {
    # Cardiovascular
    'statin': ('Statin', 'Cardiovascular'),
    'pril': ('ACE Inhibitor', 'Cardiovascular'),
    'sartan': ('ARB', 'Cardiovascular'),
    'olol': ('Beta Blocker', 'Cardiovascular'),
    'dipine': ('Calcium Channel Blocker', 'Cardiovascular'),
    'thiazide': ('Diuretic', 'Cardiovascular'),
    'nitrate': ('Nitrate', 'Cardiovascular'),
    'fibrate': ('Fibrate', 'Cardiovascular'),
    
    # Anti-infectives
    'mycin': ('Antibiotic', 'Anti-infective'),
    'cillin': ('Penicillin', 'Anti-infective'),
    'floxacin': ('Fluoroquinolone', 'Anti-infective'),
    'oxacin': ('Quinolone', 'Anti-infective'),
    'cycline': ('Tetracycline', 'Anti-infective'),
    'azole': ('Antifungal', 'Anti-infective'),
    'conazole': ('Azole Antifungal', 'Anti-infective'),
    'fungin': ('Echinocandin', 'Anti-infective'),
    'vir': ('Antiviral', 'Anti-infective'),
    'navir': ('Protease Inhibitor', 'Antiviral'),
    'ciclovir': ('Nucleoside Analog', 'Antiviral'),
    'tidine': ('H2 Blocker', 'Gastrointestinal'),
    
    # GI
    'prazole': ('Proton Pump Inhibitor', 'Gastrointestinal'),
    'setron': ('5-HT3 Antagonist', 'Gastrointestinal'),
    
    # CNS/Psychiatry
    'pam': ('Benzodiazepine', 'Psychiatry'),
    'lam': ('Benzodiazepine', 'Psychiatry'),
    'zolam': ('Benzodiazepine', 'Psychiatry'),
    'barbital': ('Barbiturate', 'CNS'),
    'triptan': ('Triptan', 'Neurology'),
    'tine': ('Antidepressant', 'Psychiatry'),
    'pine': ('Atypical Antipsychotic', 'Psychiatry'),
    'done': ('Antipsychotic', 'Psychiatry'),
    'ipramine': ('TCA', 'Psychiatry'),
    
    # Pain
    'codone': ('Opioid', 'Pain'),
    'morphone': ('Opioid', 'Pain'),
    'profen': ('NSAID', 'Pain'),
    'coxib': ('COX-2 Inhibitor', 'Pain'),
    'gesic': ('Analgesic', 'Pain'),
    
    # Oncology
    'mab': ('Monoclonal Antibody', 'Biologic'),
    'ximab': ('Chimeric Antibody', 'Biologic'),
    'zumab': ('Humanized Antibody', 'Biologic'),
    'mumab': ('Human Antibody', 'Biologic'),
    'nib': ('Kinase Inhibitor', 'Oncology'),
    'tinib': ('Tyrosine Kinase Inhibitor', 'Oncology'),
    'zomib': ('Proteasome Inhibitor', 'Oncology'),
    'ciclib': ('CDK Inhibitor', 'Oncology'),
    'platin': ('Platinum Agent', 'Oncology'),
    
    # Diabetes
    'gliptin': ('DPP-4 Inhibitor', 'Diabetes'),
    'gliflozin': ('SGLT2 Inhibitor', 'Diabetes'),
    'glitazone': ('Thiazolidinedione', 'Diabetes'),
    'glinide': ('Meglitinide', 'Diabetes'),
    'glutide': ('GLP-1 Agonist', 'Diabetes'),
    
    # Immunology
    'sone': ('Corticosteroid', 'Immunology'),
    'olone': ('Corticosteroid', 'Immunology'),
    'asone': ('Corticosteroid', 'Immunology'),
    'mune': ('Immunosuppressant', 'Immunology'),
    'limus': ('mTOR/Calcineurin Inhibitor', 'Immunology'),
    
    # Hematology
    'parin': ('Anticoagulant', 'Hematology'),
    'gatran': ('Direct Thrombin Inhibitor', 'Hematology'),
    'xaban': ('Factor Xa Inhibitor', 'Hematology'),
    'grel': ('Antiplatelet', 'Hematology'),
    
    # Respiratory
    'lukast': ('Leukotriene Antagonist', 'Respiratory'),
    'terol': ('Beta Agonist', 'Respiratory'),
    'phylline': ('Methylxanthine', 'Respiratory'),
    'tropium': ('Anticholinergic', 'Respiratory'),
    
    # Urology
    'afil': ('PDE5 Inhibitor', 'Urology'),
    'osin': ('Alpha Blocker', 'Urology'),
    'steride': ('5-Alpha Reductase Inhibitor', 'Urology'),
    
    # Neurology
    'racetam': ('Nootropic', 'Neurology'),
    'pezil': ('Cholinesterase Inhibitor', 'Neurology'),
    
    # Endocrine
    'relix': ('GnRH Antagonist', 'Endocrine'),
    'relin': ('GnRH Agonist', 'Endocrine'),
    'rozole': ('Aromatase Inhibitor', 'Oncology'),
    
    # Muscle relaxants
    'curium': ('Neuromuscular Blocker', 'Anesthesia'),
    'onium': ('Neuromuscular Blocker', 'Anesthesia'),
}


def classify_by_pattern(name: str) -> tuple:
    """Classify drug using suffix patterns."""
    name_lower = name.lower().strip()
    
    for pattern, (therapeutic_class, category) in DRUG_CLASS_PATTERNS.items():
        if pattern in name_lower:
            return (therapeutic_class, category)
    
    return None


def main():
    print('=' * 70)
    print('ENHANCED THERAPEUTIC CLASS ENRICHMENT')
    print('Sources: Pattern Matching + RxNorm API + PubChem API')
    print('=' * 70)
    
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    rxnorm = RxNormAPI(delay=0.15)
    pubchem = PubChemClassifier(delay=0.2)
    
    with driver.session() as session:
        # Get drugs without therapeutic class
        result = session.run('''
            MATCH (d:Drug) 
            WHERE d.therapeutic_class IS NULL OR d.therapeutic_class = ''
            RETURN d.name as name
        ''')
        drugs = [r['name'] for r in result if r['name']]
        print(f'\nFound {len(drugs)} drugs without classification')
        
        classified_pattern = 0
        classified_rxnorm = 0
        classified_pubchem = 0
        failed = 0
        
        for i, name in enumerate(drugs):
            if (i + 1) % 25 == 0:
                print(f'  Progress: {i + 1}/{len(drugs)} (Pattern: {classified_pattern}, RxNorm: {classified_rxnorm}, PubChem: {classified_pubchem})')
            
            therapeutic_class = None
            category = None
            mechanism = None
            source = None
            
            # Method 1: Pattern matching (fastest)
            pattern_result = classify_by_pattern(name)
            if pattern_result:
                therapeutic_class, category = pattern_result
                source = 'pattern'
                classified_pattern += 1
            
            # Method 2: RxNorm API (if pattern didn't work)
            if not therapeutic_class:
                rxcui = rxnorm.get_rxcui(name)
                if rxcui:
                    rx_class = rxnorm.get_drug_class(rxcui)
                    if rx_class:
                        therapeutic_class = rx_class.get('therapeutic_class') or rx_class.get('atc_class')
                        category = rx_class.get('category')
                        mechanism = rx_class.get('mechanism')
                        if therapeutic_class:
                            source = 'rxnorm'
                            classified_rxnorm += 1
            
            # Method 3: PubChem (fallback)
            if not therapeutic_class and i < 200:  # Limit PubChem calls
                pc_class = pubchem.get_classification(name)
                if pc_class and pc_class.get('therapeutic_class'):
                    therapeutic_class = pc_class['therapeutic_class']
                    category = pc_class.get('category')
                    source = 'pubchem'
                    classified_pubchem += 1
            
            # Update Neo4j if classified
            if therapeutic_class:
                update_params = {
                    'name': name,
                    'tc': therapeutic_class,
                    'source': source
                }
                
                query = '''
                    MATCH (d:Drug {name: $name})
                    SET d.therapeutic_class = $tc,
                        d.classification_source = $source
                '''
                
                if category:
                    query += ', d.category = $cat'
                    update_params['cat'] = category
                
                if mechanism:
                    query += ', d.mechanism = $mech'
                    update_params['mech'] = mechanism
                
                session.run(query, update_params)
            else:
                failed += 1
        
        # Summary
        total_classified = classified_pattern + classified_rxnorm + classified_pubchem
        print(f'\n' + '=' * 50)
        print(f'CLASSIFICATION RESULTS')
        print(f'=' * 50)
        print(f'  Pattern matching: {classified_pattern}')
        print(f'  RxNorm API:       {classified_rxnorm}')
        print(f'  PubChem API:      {classified_pubchem}')
        print(f'  ---------------------------')
        print(f'  Total classified: {total_classified}')
        print(f'  Unclassified:     {failed}')
        
        # Get final stats
        result = session.run('''
            MATCH (d:Drug)
            RETURN 
                count(*) as total,
                sum(CASE WHEN d.therapeutic_class IS NOT NULL THEN 1 ELSE 0 END) as classified
        ''')
        stats = result.single()
        print(f'\n  Total drugs in DB:    {stats["total"]}')
        print(f'  Total with class:     {stats["classified"]}')
        print(f'  Coverage:             {100*stats["classified"]/stats["total"]:.1f}%')
        
        # Show category distribution
        result = session.run('''
            MATCH (d:Drug)
            WHERE d.therapeutic_class IS NOT NULL
            RETURN d.therapeutic_class as tc, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 15
        ''')
        print(f'\nTop therapeutic classes:')
        for r in result:
            print(f'  {r["tc"]}: {r["cnt"]}')
    
    driver.close()
    print('\n✅ Enhanced enrichment complete!')


if __name__ == '__main__':
    main()
