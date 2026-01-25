"""Add therapeutic classes to drugs in Neo4j Aura."""
from neo4j import GraphDatabase

# Neo4j Aura connection
URI = 'neo4j+s://ca47aebc.databases.neo4j.io'
USER = 'neo4j'
PASSWORD = 'BYKmHWoR2DeEiiiwO6qBAET273OIaaGv1ZatYpU_vtM'

# Drug class patterns based on suffixes
DRUG_CLASS_PATTERNS = {
    'statin': ('Statin', 'Cardiovascular'),
    'pril': ('ACE Inhibitor', 'Cardiovascular'),
    'sartan': ('ARB', 'Cardiovascular'),
    'olol': ('Beta Blocker', 'Cardiovascular'),
    'dipine': ('Calcium Channel Blocker', 'Cardiovascular'),
    'thiazide': ('Diuretic', 'Cardiovascular'),
    'mycin': ('Antibiotic', 'Anti-infective'),
    'cillin': ('Penicillin Antibiotic', 'Anti-infective'),
    'floxacin': ('Fluoroquinolone', 'Anti-infective'),
    'cycline': ('Tetracycline', 'Anti-infective'),
    'azole': ('Antifungal', 'Anti-infective'),
    'prazole': ('Proton Pump Inhibitor', 'Gastrointestinal'),
    'tidine': ('H2 Blocker', 'Gastrointestinal'),
    'pam': ('Benzodiazepine', 'CNS'),
    'lam': ('Benzodiazepine', 'CNS'),
    'barbital': ('Barbiturate', 'CNS'),
    'triptan': ('Triptan', 'CNS'),
    'codone': ('Opioid Analgesic', 'Pain'),
    'morphine': ('Opioid Analgesic', 'Pain'),
    'profen': ('NSAID', 'Pain'),
    'mab': ('Monoclonal Antibody', 'Biologic'),
    'nib': ('Kinase Inhibitor', 'Oncology'),
    'metformin': ('Biguanide', 'Diabetes'),
    'gliptin': ('DPP-4 Inhibitor', 'Diabetes'),
    'gliflozin': ('SGLT2 Inhibitor', 'Diabetes'),
    'glitazone': ('Thiazolidinedione', 'Diabetes'),
    'sone': ('Corticosteroid', 'Immunology'),
    'olone': ('Corticosteroid', 'Immunology'),
    'parin': ('Anticoagulant', 'Hematology'),
    'vir': ('Antiviral', 'Anti-infective'),
    'navir': ('Protease Inhibitor', 'Antiviral'),
    'coxib': ('COX-2 Inhibitor', 'Pain'),
    'oxacin': ('Quinolone', 'Anti-infective'),
    'lukast': ('Leukotriene Antagonist', 'Respiratory'),
    'terol': ('Beta Agonist', 'Respiratory'),
    'asone': ('Corticosteroid', 'Immunology'),
}

# Known drug mappings (common drugs)
KNOWN_DRUGS = {
    'aspirin': ('NSAID/Antiplatelet', 'Pain/Cardiovascular'),
    'acetaminophen': ('Analgesic', 'Pain'),
    'paracetamol': ('Analgesic', 'Pain'),
    'ibuprofen': ('NSAID', 'Pain'),
    'naproxen': ('NSAID', 'Pain'),
    'methotrexate': ('Antimetabolite', 'Immunology/Oncology'),
    'warfarin': ('Anticoagulant', 'Hematology'),
    'heparin': ('Anticoagulant', 'Hematology'),
    'insulin': ('Insulin', 'Diabetes'),
    'digoxin': ('Cardiac Glycoside', 'Cardiovascular'),
    'lithium': ('Mood Stabilizer', 'Psychiatry'),
    'phenytoin': ('Anticonvulsant', 'Neurology'),
    'carbamazepine': ('Anticonvulsant', 'Neurology'),
    'valproic acid': ('Anticonvulsant', 'Neurology'),
    'theophylline': ('Bronchodilator', 'Respiratory'),
    'amiodarone': ('Antiarrhythmic', 'Cardiovascular'),
    'cyclosporine': ('Immunosuppressant', 'Immunology'),
    'tacrolimus': ('Immunosuppressant', 'Immunology'),
    'fluoxetine': ('SSRI', 'Psychiatry'),
    'sertraline': ('SSRI', 'Psychiatry'),
    'paroxetine': ('SSRI', 'Psychiatry'),
    'citalopram': ('SSRI', 'Psychiatry'),
    'escitalopram': ('SSRI', 'Psychiatry'),
    'venlafaxine': ('SNRI', 'Psychiatry'),
    'duloxetine': ('SNRI', 'Psychiatry'),
    'metoprolol': ('Beta Blocker', 'Cardiovascular'),
    'atenolol': ('Beta Blocker', 'Cardiovascular'),
    'propranolol': ('Beta Blocker', 'Cardiovascular'),
    'amlodipine': ('Calcium Channel Blocker', 'Cardiovascular'),
    'lisinopril': ('ACE Inhibitor', 'Cardiovascular'),
    'enalapril': ('ACE Inhibitor', 'Cardiovascular'),
    'losartan': ('ARB', 'Cardiovascular'),
    'valsartan': ('ARB', 'Cardiovascular'),
    'simvastatin': ('Statin', 'Cardiovascular'),
    'atorvastatin': ('Statin', 'Cardiovascular'),
    'rosuvastatin': ('Statin', 'Cardiovascular'),
    'lovastatin': ('Statin', 'Cardiovascular'),
    'pravastatin': ('Statin', 'Cardiovascular'),
    'omeprazole': ('PPI', 'Gastrointestinal'),
    'pantoprazole': ('PPI', 'Gastrointestinal'),
    'lansoprazole': ('PPI', 'Gastrointestinal'),
    'esomeprazole': ('PPI', 'Gastrointestinal'),
    'rabeprazole': ('PPI', 'Gastrointestinal'),
    'metformin': ('Biguanide', 'Diabetes'),
    'glipizide': ('Sulfonylurea', 'Diabetes'),
    'glyburide': ('Sulfonylurea', 'Diabetes'),
    'glimepiride': ('Sulfonylurea', 'Diabetes'),
    'ciprofloxacin': ('Fluoroquinolone', 'Anti-infective'),
    'levofloxacin': ('Fluoroquinolone', 'Anti-infective'),
    'moxifloxacin': ('Fluoroquinolone', 'Anti-infective'),
    'azithromycin': ('Macrolide', 'Anti-infective'),
    'clarithromycin': ('Macrolide', 'Anti-infective'),
    'erythromycin': ('Macrolide', 'Anti-infective'),
    'amoxicillin': ('Penicillin', 'Anti-infective'),
    'ampicillin': ('Penicillin', 'Anti-infective'),
    'penicillin': ('Penicillin', 'Anti-infective'),
    'cephalexin': ('Cephalosporin', 'Anti-infective'),
    'ceftriaxone': ('Cephalosporin', 'Anti-infective'),
    'cefuroxime': ('Cephalosporin', 'Anti-infective'),
    'fluconazole': ('Antifungal', 'Anti-infective'),
    'itraconazole': ('Antifungal', 'Anti-infective'),
    'ketoconazole': ('Antifungal', 'Anti-infective'),
    'voriconazole': ('Antifungal', 'Anti-infective'),
    'prednisone': ('Corticosteroid', 'Immunology'),
    'prednisolone': ('Corticosteroid', 'Immunology'),
    'dexamethasone': ('Corticosteroid', 'Immunology'),
    'hydrocortisone': ('Corticosteroid', 'Immunology'),
    'methylprednisolone': ('Corticosteroid', 'Immunology'),
    'tramadol': ('Opioid Analgesic', 'Pain'),
    'oxycodone': ('Opioid Analgesic', 'Pain'),
    'hydrocodone': ('Opioid Analgesic', 'Pain'),
    'fentanyl': ('Opioid Analgesic', 'Pain'),
    'morphine': ('Opioid Analgesic', 'Pain'),
    'codeine': ('Opioid Analgesic', 'Pain'),
    'methadone': ('Opioid Analgesic', 'Pain'),
    'buprenorphine': ('Opioid Analgesic', 'Pain'),
    'alprazolam': ('Benzodiazepine', 'Psychiatry'),
    'lorazepam': ('Benzodiazepine', 'Psychiatry'),
    'diazepam': ('Benzodiazepine', 'Psychiatry'),
    'clonazepam': ('Benzodiazepine', 'Psychiatry'),
    'temazepam': ('Benzodiazepine', 'Psychiatry'),
    'midazolam': ('Benzodiazepine', 'Psychiatry'),
    'gabapentin': ('Anticonvulsant', 'Neurology'),
    'pregabalin': ('Anticonvulsant', 'Neurology'),
    'lamotrigine': ('Anticonvulsant', 'Neurology'),
    'levetiracetam': ('Anticonvulsant', 'Neurology'),
    'topiramate': ('Anticonvulsant', 'Neurology'),
    'levothyroxine': ('Thyroid Hormone', 'Endocrine'),
    'spironolactone': ('Potassium-Sparing Diuretic', 'Cardiovascular'),
    'furosemide': ('Loop Diuretic', 'Cardiovascular'),
    'hydrochlorothiazide': ('Thiazide Diuretic', 'Cardiovascular'),
    'bumetanide': ('Loop Diuretic', 'Cardiovascular'),
    'torsemide': ('Loop Diuretic', 'Cardiovascular'),
    'clopidogrel': ('Antiplatelet', 'Hematology'),
    'rivaroxaban': ('Anticoagulant', 'Hematology'),
    'apixaban': ('Anticoagulant', 'Hematology'),
    'dabigatran': ('Anticoagulant', 'Hematology'),
    'edoxaban': ('Anticoagulant', 'Hematology'),
    'quinidine': ('Antiarrhythmic', 'Cardiovascular'),
    'procainamide': ('Antiarrhythmic', 'Cardiovascular'),
    'lidocaine': ('Local Anesthetic/Antiarrhythmic', 'Pain/Cardiovascular'),
    'sotalol': ('Antiarrhythmic', 'Cardiovascular'),
    'flecainide': ('Antiarrhythmic', 'Cardiovascular'),
    'dofetilide': ('Antiarrhythmic', 'Cardiovascular'),
    'rifampin': ('Antibiotic', 'Anti-infective'),
    'rifampicin': ('Antibiotic', 'Anti-infective'),
    'isoniazid': ('Antitubercular', 'Anti-infective'),
    'metronidazole': ('Antibiotic', 'Anti-infective'),
    'vancomycin': ('Antibiotic', 'Anti-infective'),
    'gentamicin': ('Aminoglycoside', 'Anti-infective'),
    'tobramycin': ('Aminoglycoside', 'Anti-infective'),
    'amikacin': ('Aminoglycoside', 'Anti-infective'),
    'clindamycin': ('Antibiotic', 'Anti-infective'),
    'doxycycline': ('Tetracycline', 'Anti-infective'),
    'tetracycline': ('Tetracycline', 'Anti-infective'),
    'minocycline': ('Tetracycline', 'Anti-infective'),
    'sulfamethoxazole': ('Sulfonamide', 'Anti-infective'),
    'trimethoprim': ('Antibiotic', 'Anti-infective'),
    'nitrofurantoin': ('Antibiotic', 'Anti-infective'),
    'acyclovir': ('Antiviral', 'Anti-infective'),
    'valacyclovir': ('Antiviral', 'Anti-infective'),
    'oseltamivir': ('Antiviral', 'Anti-infective'),
    'ritonavir': ('Protease Inhibitor', 'Antiviral'),
    'lopinavir': ('Protease Inhibitor', 'Antiviral'),
    'efavirenz': ('NNRTI', 'Antiviral'),
    'tenofovir': ('NRTI', 'Antiviral'),
    'emtricitabine': ('NRTI', 'Antiviral'),
    'hydroxychloroquine': ('Antimalarial/DMARD', 'Immunology'),
    'chloroquine': ('Antimalarial', 'Anti-infective'),
    'colchicine': ('Antigout', 'Immunology'),
    'allopurinol': ('Xanthine Oxidase Inhibitor', 'Immunology'),
    'febuxostat': ('Xanthine Oxidase Inhibitor', 'Immunology'),
    'probenecid': ('Uricosuric', 'Immunology'),
    'sildenafil': ('PDE5 Inhibitor', 'Cardiovascular/Urology'),
    'tadalafil': ('PDE5 Inhibitor', 'Cardiovascular/Urology'),
    'vardenafil': ('PDE5 Inhibitor', 'Urology'),
    'tamsulosin': ('Alpha Blocker', 'Urology'),
    'finasteride': ('5-Alpha Reductase Inhibitor', 'Urology'),
    'dutasteride': ('5-Alpha Reductase Inhibitor', 'Urology'),
    'ranitidine': ('H2 Blocker', 'Gastrointestinal'),
    'famotidine': ('H2 Blocker', 'Gastrointestinal'),
    'cimetidine': ('H2 Blocker', 'Gastrointestinal'),
    'ondansetron': ('Antiemetic', 'Gastrointestinal'),
    'metoclopramide': ('Antiemetic', 'Gastrointestinal'),
    'promethazine': ('Antihistamine/Antiemetic', 'Allergy'),
    'diphenhydramine': ('Antihistamine', 'Allergy'),
    'cetirizine': ('Antihistamine', 'Allergy'),
    'loratadine': ('Antihistamine', 'Allergy'),
    'fexofenadine': ('Antihistamine', 'Allergy'),
    'montelukast': ('Leukotriene Antagonist', 'Respiratory'),
    'albuterol': ('Beta Agonist', 'Respiratory'),
    'salbutamol': ('Beta Agonist', 'Respiratory'),
    'salmeterol': ('Long-Acting Beta Agonist', 'Respiratory'),
    'formoterol': ('Long-Acting Beta Agonist', 'Respiratory'),
    'tiotropium': ('Anticholinergic', 'Respiratory'),
    'ipratropium': ('Anticholinergic', 'Respiratory'),
    'budesonide': ('Inhaled Corticosteroid', 'Respiratory'),
    'fluticasone': ('Inhaled Corticosteroid', 'Respiratory'),
    'beclomethasone': ('Inhaled Corticosteroid', 'Respiratory'),
    'quetiapine': ('Atypical Antipsychotic', 'Psychiatry'),
    'risperidone': ('Atypical Antipsychotic', 'Psychiatry'),
    'olanzapine': ('Atypical Antipsychotic', 'Psychiatry'),
    'aripiprazole': ('Atypical Antipsychotic', 'Psychiatry'),
    'ziprasidone': ('Atypical Antipsychotic', 'Psychiatry'),
    'clozapine': ('Atypical Antipsychotic', 'Psychiatry'),
    'haloperidol': ('Typical Antipsychotic', 'Psychiatry'),
    'chlorpromazine': ('Typical Antipsychotic', 'Psychiatry'),
    'amitriptyline': ('TCA', 'Psychiatry'),
    'nortriptyline': ('TCA', 'Psychiatry'),
    'imipramine': ('TCA', 'Psychiatry'),
    'desipramine': ('TCA', 'Psychiatry'),
    'trazodone': ('Atypical Antidepressant', 'Psychiatry'),
    'mirtazapine': ('Atypical Antidepressant', 'Psychiatry'),
    'bupropion': ('Atypical Antidepressant', 'Psychiatry'),
    'buspirone': ('Anxiolytic', 'Psychiatry'),
    'zolpidem': ('Sedative-Hypnotic', 'Psychiatry'),
    'eszopiclone': ('Sedative-Hypnotic', 'Psychiatry'),
    'methylphenidate': ('Stimulant', 'Psychiatry'),
    'amphetamine': ('Stimulant', 'Psychiatry'),
    'dextroamphetamine': ('Stimulant', 'Psychiatry'),
    'atomoxetine': ('SNRI (ADHD)', 'Psychiatry'),
    'donepezil': ('Cholinesterase Inhibitor', 'Neurology'),
    'rivastigmine': ('Cholinesterase Inhibitor', 'Neurology'),
    'galantamine': ('Cholinesterase Inhibitor', 'Neurology'),
    'memantine': ('NMDA Antagonist', 'Neurology'),
    'levodopa': ('Dopamine Precursor', 'Neurology'),
    'carbidopa': ('DDC Inhibitor', 'Neurology'),
    'pramipexole': ('Dopamine Agonist', 'Neurology'),
    'ropinirole': ('Dopamine Agonist', 'Neurology'),
    'sumatriptan': ('Triptan', 'Neurology'),
    'rizatriptan': ('Triptan', 'Neurology'),
    'zolmitriptan': ('Triptan', 'Neurology'),
}


def classify_drug(name):
    """Classify a drug by name."""
    name_lower = name.lower().strip()
    
    # Check known drugs first
    if name_lower in KNOWN_DRUGS:
        return KNOWN_DRUGS[name_lower]
    
    # Try pattern matching
    for pattern, (therapeutic_class, category) in DRUG_CLASS_PATTERNS.items():
        if pattern in name_lower:
            return (therapeutic_class, category)
    
    return None


def main():
    print('=' * 60)
    print('THERAPEUTIC CLASS ENRICHMENT FOR NEO4J AURA')
    print('=' * 60)
    print(f'\nConnecting to Neo4j Aura...')
    
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    with driver.session() as session:
        # Get all drugs
        result = session.run('MATCH (d:Drug) RETURN d.name as name')
        drugs = [r['name'] for r in result if r['name']]
        print(f'Found {len(drugs)} drugs in Neo4j')
        
        classified = 0
        for name in drugs:
            classification = classify_drug(name)
            if classification:
                therapeutic_class, category = classification
                session.run('''
                    MATCH (d:Drug {name: $name})
                    SET d.therapeutic_class = $tc, d.category = $cat
                ''', name=name, tc=therapeutic_class, cat=category)
                classified += 1
                if classified % 50 == 0:
                    print(f'  Classified {classified} drugs...')
        
        print(f'\n✅ Done! Classified {classified} out of {len(drugs)} drugs')
        
        # Show sample of classified drugs
        result = session.run('''
            MATCH (d:Drug) 
            WHERE d.therapeutic_class IS NOT NULL 
            RETURN d.name as name, d.therapeutic_class as tc, d.category as cat 
            LIMIT 15
        ''')
        print('\nSample classified drugs:')
        for r in result:
            print(f'  {r["name"]}: {r["tc"]} ({r["cat"]})')
        
        # Show category distribution
        result = session.run('''
            MATCH (d:Drug)
            WHERE d.category IS NOT NULL
            RETURN d.category as category, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        ''')
        print('\nCategory distribution:')
        for r in result:
            print(f'  {r["category"]}: {r["count"]} drugs')
    
    driver.close()
    print('\n✅ Enrichment complete!')


if __name__ == '__main__':
    main()
