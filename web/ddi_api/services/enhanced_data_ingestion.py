"""
Enhanced Data Ingestion Pipeline for DDI Knowledge Graph

Extends the base data ingestion with additional data sources:
1. DrugBank API integration
2. PubChem compound data
3. ChEMBL bioactivity data
4. FDA drug labels
5. Food-drug interactions
6. Herbal supplement interactions

Reference:
- DrugBank: https://go.drugbank.com/
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
- ChEMBL: https://www.ebi.ac.uk/chembl/
"""

import logging
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / 'data'


@dataclass
class FoodDrugInteraction:
    """Represents a food-drug interaction."""
    food: str
    drug: str
    mechanism: str
    severity: str
    effect: str
    recommendation: str


@dataclass
class HerbalInteraction:
    """Represents an herbal supplement-drug interaction."""
    herb: str
    drug: str
    mechanism: str
    severity: str
    effect: str
    recommendation: str


# Curated food-drug interactions database
FOOD_DRUG_INTERACTIONS = [
    # Grapefruit interactions (CYP3A4 inhibition)
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="simvastatin",
        mechanism="Grapefruit inhibits intestinal CYP3A4, dramatically increasing statin absorption",
        severity="severe",
        effect="3-16x increase in statin levels, risk of rhabdomyolysis",
        recommendation="Avoid grapefruit juice with simvastatin. Consider pravastatin or rosuvastatin."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="atorvastatin",
        mechanism="Grapefruit inhibits intestinal CYP3A4, increasing statin levels",
        severity="major",
        effect="2-3x increase in atorvastatin levels",
        recommendation="Limit grapefruit juice intake or use alternative statin."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="lovastatin",
        mechanism="Grapefruit inhibits intestinal CYP3A4",
        severity="severe",
        effect="Significant increase in lovastatin levels, myopathy risk",
        recommendation="Avoid grapefruit with lovastatin."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="cyclosporine",
        mechanism="CYP3A4 inhibition increases cyclosporine absorption",
        severity="major",
        effect="Increased cyclosporine levels and potential toxicity",
        recommendation="Avoid grapefruit; monitor cyclosporine levels."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="nifedipine",
        mechanism="CYP3A4 inhibition enhances calcium channel blocker effect",
        severity="moderate",
        effect="Increased hypotensive effect, flushing, headache",
        recommendation="Avoid or limit grapefruit with calcium channel blockers."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="felodipine",
        mechanism="CYP3A4 inhibition in gut wall",
        severity="major",
        effect="2-3x increase in felodipine levels, severe hypotension risk",
        recommendation="Avoid grapefruit juice with felodipine."
    ),
    FoodDrugInteraction(
        food="Grapefruit juice",
        drug="midazolam",
        mechanism="CYP3A4 inhibition increases benzodiazepine levels",
        severity="major",
        effect="Increased sedation and respiratory depression risk",
        recommendation="Avoid grapefruit with oral midazolam."
    ),
    
    # Vitamin K interactions with warfarin
    FoodDrugInteraction(
        food="Leafy green vegetables",
        drug="warfarin",
        mechanism="Vitamin K antagonizes warfarin's anticoagulant effect",
        severity="major",
        effect="Decreased INR, reduced anticoagulation, clot risk",
        recommendation="Maintain consistent vitamin K intake; monitor INR with diet changes."
    ),
    FoodDrugInteraction(
        food="Kale",
        drug="warfarin",
        mechanism="High vitamin K content counteracts warfarin",
        severity="major",
        effect="Reduced anticoagulation effect",
        recommendation="Keep vitamin K intake consistent; avoid sudden large increases."
    ),
    FoodDrugInteraction(
        food="Spinach",
        drug="warfarin",
        mechanism="High vitamin K content",
        severity="moderate",
        effect="May reduce INR",
        recommendation="Eat consistent amounts; monitor INR."
    ),
    
    # Dairy and antibiotics
    FoodDrugInteraction(
        food="Dairy products",
        drug="ciprofloxacin",
        mechanism="Calcium chelates quinolone antibiotics, reducing absorption",
        severity="major",
        effect="50-70% reduction in antibiotic absorption, treatment failure risk",
        recommendation="Take ciprofloxacin 2 hours before or 6 hours after dairy."
    ),
    FoodDrugInteraction(
        food="Dairy products",
        drug="tetracycline",
        mechanism="Calcium forms insoluble complexes with tetracycline",
        severity="major",
        effect="Significantly reduced antibiotic absorption",
        recommendation="Avoid dairy 2 hours before and after tetracycline."
    ),
    FoodDrugInteraction(
        food="Dairy products",
        drug="doxycycline",
        mechanism="Calcium reduces absorption (less affected than tetracycline)",
        severity="moderate",
        effect="20-30% reduction in absorption",
        recommendation="Take doxycycline separately from dairy if possible."
    ),
    
    # Tyramine-rich foods and MAOIs
    FoodDrugInteraction(
        food="Aged cheese",
        drug="phenelzine",
        mechanism="Tyramine accumulation due to MAO inhibition",
        severity="severe",
        effect="Hypertensive crisis, potentially fatal",
        recommendation="STRICT avoidance of aged cheese and tyramine-rich foods."
    ),
    FoodDrugInteraction(
        food="Aged cheese",
        drug="tranylcypromine",
        mechanism="MAO inhibition prevents tyramine breakdown",
        severity="severe",
        effect="Hypertensive crisis",
        recommendation="Avoid all tyramine-rich foods during and 2 weeks after stopping MAOI."
    ),
    FoodDrugInteraction(
        food="Red wine",
        drug="phenelzine",
        mechanism="Tyramine content in some wines",
        severity="major",
        effect="Hypertensive reaction",
        recommendation="Avoid or strictly limit wine with MAOIs."
    ),
    
    # Alcohol interactions
    FoodDrugInteraction(
        food="Alcohol",
        drug="metronidazole",
        mechanism="Disulfiram-like reaction",
        severity="major",
        effect="Nausea, vomiting, flushing, tachycardia",
        recommendation="Avoid alcohol during and 3 days after metronidazole."
    ),
    FoodDrugInteraction(
        food="Alcohol",
        drug="metformin",
        mechanism="Both increase lactic acid; alcohol impairs gluconeogenesis",
        severity="major",
        effect="Increased lactic acidosis risk, hypoglycemia",
        recommendation="Limit alcohol intake with metformin."
    ),
    FoodDrugInteraction(
        food="Alcohol",
        drug="acetaminophen",
        mechanism="Alcohol induces CYP2E1, increasing toxic metabolite formation",
        severity="major",
        effect="Increased hepatotoxicity risk, even at therapeutic doses",
        recommendation="Avoid chronic alcohol use with regular acetaminophen."
    ),
    FoodDrugInteraction(
        food="Alcohol",
        drug="benzodiazepines",
        mechanism="Additive CNS depression",
        severity="severe",
        effect="Profound sedation, respiratory depression, coma risk",
        recommendation="Avoid alcohol with all benzodiazepines."
    ),
    FoodDrugInteraction(
        food="Alcohol",
        drug="opioids",
        mechanism="Additive CNS and respiratory depression",
        severity="severe",
        effect="Respiratory depression, overdose death risk",
        recommendation="AVOID alcohol with opioids - life-threatening combination."
    ),
    
    # Caffeine interactions
    FoodDrugInteraction(
        food="Coffee/Caffeine",
        drug="theophylline",
        mechanism="Both are xanthine derivatives metabolized by CYP1A2",
        severity="moderate",
        effect="Additive CNS stimulation, increased side effects",
        recommendation="Limit caffeine intake with theophylline."
    ),
    FoodDrugInteraction(
        food="Coffee/Caffeine",
        drug="ciprofloxacin",
        mechanism="Ciprofloxacin inhibits CYP1A2, reducing caffeine clearance",
        severity="moderate",
        effect="Increased caffeine effects: anxiety, insomnia, palpitations",
        recommendation="Reduce caffeine intake during ciprofloxacin therapy."
    ),
    
    # High-fiber foods
    FoodDrugInteraction(
        food="High-fiber foods",
        drug="digoxin",
        mechanism="Fiber may bind digoxin, reducing absorption",
        severity="moderate",
        effect="Reduced digoxin levels",
        recommendation="Take digoxin at consistent times relative to high-fiber meals."
    ),
    FoodDrugInteraction(
        food="High-fiber foods",
        drug="levothyroxine",
        mechanism="Fiber reduces thyroid hormone absorption",
        severity="moderate",
        effect="Subtherapeutic thyroid levels",
        recommendation="Take levothyroxine on empty stomach, separate from fiber."
    ),
    
    # Potassium-rich foods
    FoodDrugInteraction(
        food="Potassium-rich foods (bananas, oranges)",
        drug="spironolactone",
        mechanism="Spironolactone is potassium-sparing; high K+ intake adds to effect",
        severity="major",
        effect="Hyperkalemia risk - cardiac arrhythmias",
        recommendation="Monitor potassium; avoid potassium supplements and salt substitutes."
    ),
    FoodDrugInteraction(
        food="Potassium-rich foods",
        drug="lisinopril",
        mechanism="ACE inhibitors increase potassium retention",
        severity="moderate",
        effect="Hyperkalemia risk, especially in renal impairment",
        recommendation="Monitor potassium levels; moderate K+-rich food intake."
    ),
]


# Curated herbal supplement interactions database
HERBAL_DRUG_INTERACTIONS = [
    # St. John's Wort (potent CYP3A4 inducer)
    HerbalInteraction(
        herb="St. John's Wort",
        drug="cyclosporine",
        mechanism="St. John's Wort induces CYP3A4 and P-glycoprotein",
        severity="severe",
        effect="50%+ decrease in cyclosporine levels - organ rejection risk",
        recommendation="AVOID combination. Use contraindicated."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="tacrolimus",
        mechanism="CYP3A4 and P-gp induction",
        severity="severe",
        effect="Dramatically reduced tacrolimus levels",
        recommendation="Contraindicated - risk of transplant rejection."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="oral contraceptives",
        mechanism="CYP3A4 induction increases hormone metabolism",
        severity="major",
        effect="Reduced contraceptive efficacy, breakthrough bleeding, pregnancy risk",
        recommendation="Use alternative contraception; avoid St. John's Wort."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="warfarin",
        mechanism="Induces CYP enzymes and P-gp",
        severity="major",
        effect="Reduced INR, decreased anticoagulation",
        recommendation="Avoid; if used, monitor INR closely."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="simvastatin",
        mechanism="CYP3A4 induction",
        severity="major",
        effect="Reduced statin levels and efficacy",
        recommendation="Avoid combination; use alternative antidepressant."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="SSRIs",
        mechanism="Both affect serotonin pathways",
        severity="major",
        effect="Serotonin syndrome risk",
        recommendation="Avoid combining; serious adverse effects possible."
    ),
    HerbalInteraction(
        herb="St. John's Wort",
        drug="HIV protease inhibitors",
        mechanism="CYP3A4 induction reduces protease inhibitor levels",
        severity="severe",
        effect="Treatment failure, viral resistance",
        recommendation="CONTRAINDICATED - do not use together."
    ),
    
    # Ginkgo biloba
    HerbalInteraction(
        herb="Ginkgo biloba",
        drug="warfarin",
        mechanism="Ginkgo has antiplatelet effects",
        severity="major",
        effect="Increased bleeding risk, including intracranial hemorrhage",
        recommendation="Avoid combination; increased bleeding risk."
    ),
    HerbalInteraction(
        herb="Ginkgo biloba",
        drug="aspirin",
        mechanism="Additive antiplatelet effects",
        severity="major",
        effect="Increased bleeding risk",
        recommendation="Use caution; monitor for bleeding."
    ),
    HerbalInteraction(
        herb="Ginkgo biloba",
        drug="ibuprofen",
        mechanism="Additive effects on platelet function",
        severity="moderate",
        effect="Increased GI bleeding risk",
        recommendation="Use caution with NSAIDs."
    ),
    
    # Kava
    HerbalInteraction(
        herb="Kava",
        drug="alprazolam",
        mechanism="Additive CNS depression",
        severity="major",
        effect="Excessive sedation, respiratory depression",
        recommendation="Avoid combination."
    ),
    HerbalInteraction(
        herb="Kava",
        drug="levodopa",
        mechanism="Kava may have dopamine-blocking effects",
        severity="moderate",
        effect="Reduced levodopa efficacy, worsened Parkinson's symptoms",
        recommendation="Avoid in Parkinson's disease."
    ),
    HerbalInteraction(
        herb="Kava",
        drug="hepatotoxic drugs",
        mechanism="Kava is hepatotoxic",
        severity="major",
        effect="Additive liver damage risk",
        recommendation="Avoid with any hepatotoxic medication."
    ),
    
    # Garlic
    HerbalInteraction(
        herb="Garlic supplements",
        drug="warfarin",
        mechanism="Garlic has antiplatelet activity",
        severity="moderate",
        effect="Increased bleeding risk",
        recommendation="Use caution; monitor INR."
    ),
    HerbalInteraction(
        herb="Garlic supplements",
        drug="saquinavir",
        mechanism="Garlic may induce CYP3A4",
        severity="major",
        effect="Reduced saquinavir levels by 50%",
        recommendation="Avoid garlic supplements with protease inhibitors."
    ),
    
    # Ginseng
    HerbalInteraction(
        herb="Ginseng",
        drug="warfarin",
        mechanism="Ginseng may reduce warfarin's anticoagulant effect",
        severity="moderate",
        effect="Decreased INR",
        recommendation="Monitor INR if ginseng used."
    ),
    HerbalInteraction(
        herb="Ginseng",
        drug="insulin",
        mechanism="Ginseng may have hypoglycemic effects",
        severity="moderate",
        effect="Additive blood glucose lowering",
        recommendation="Monitor blood glucose closely."
    ),
    HerbalInteraction(
        herb="Ginseng",
        drug="MAO inhibitors",
        mechanism="Potential stimulant interaction",
        severity="major",
        effect="Headache, tremor, manic episodes reported",
        recommendation="Avoid combination."
    ),
    
    # Echinacea
    HerbalInteraction(
        herb="Echinacea",
        drug="immunosuppressants",
        mechanism="Echinacea stimulates immune function",
        severity="major",
        effect="May counteract immunosuppression",
        recommendation="Avoid in transplant patients and autoimmune disease."
    ),
    
    # Valerian
    HerbalInteraction(
        herb="Valerian",
        drug="benzodiazepines",
        mechanism="Additive CNS depression",
        severity="moderate",
        effect="Increased sedation",
        recommendation="Use caution; consider dose reduction."
    ),
    HerbalInteraction(
        herb="Valerian",
        drug="alcohol",
        mechanism="Additive sedative effects",
        severity="moderate",
        effect="Excessive drowsiness, impaired coordination",
        recommendation="Avoid combining with alcohol."
    ),
    
    # Licorice
    HerbalInteraction(
        herb="Licorice root",
        drug="digoxin",
        mechanism="Licorice causes potassium loss, increasing digoxin toxicity",
        severity="major",
        effect="Hypokalemia increases digoxin sensitivity, arrhythmia risk",
        recommendation="Avoid licorice with digoxin."
    ),
    HerbalInteraction(
        herb="Licorice root",
        drug="diuretics",
        mechanism="Additive potassium loss",
        severity="major",
        effect="Severe hypokalemia risk",
        recommendation="Avoid combination."
    ),
    HerbalInteraction(
        herb="Licorice root",
        drug="antihypertensives",
        mechanism="Licorice has mineralocorticoid effects",
        severity="major",
        effect="Sodium retention, hypertension, counteracts antihypertensive effect",
        recommendation="Avoid licorice in hypertensive patients."
    ),
]


class EnhancedDataIngestion:
    """
    Enhanced data ingestion service with additional data sources.
    """
    
    def __init__(self):
        """Initialize the enhanced data ingestion service."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._food_interactions = FOOD_DRUG_INTERACTIONS
        self._herbal_interactions = HERBAL_DRUG_INTERACTIONS
    
    def get_food_drug_interactions(
        self, 
        drug_name: Optional[str] = None,
        food_name: Optional[str] = None
    ) -> List[FoodDrugInteraction]:
        """
        Get food-drug interactions, optionally filtered.
        
        Args:
            drug_name: Filter by drug name
            food_name: Filter by food name
            
        Returns:
            List of matching interactions
        """
        results = self._food_interactions
        
        if drug_name:
            drug_lower = drug_name.lower()
            results = [i for i in results if drug_lower in i.drug.lower()]
        
        if food_name:
            food_lower = food_name.lower()
            results = [i for i in results if food_lower in i.food.lower()]
        
        return results
    
    def get_herbal_drug_interactions(
        self,
        drug_name: Optional[str] = None,
        herb_name: Optional[str] = None
    ) -> List[HerbalInteraction]:
        """
        Get herbal supplement-drug interactions, optionally filtered.
        
        Args:
            drug_name: Filter by drug name
            herb_name: Filter by herb name
            
        Returns:
            List of matching interactions
        """
        results = self._herbal_interactions
        
        if drug_name:
            drug_lower = drug_name.lower()
            results = [i for i in results if drug_lower in i.drug.lower()]
        
        if herb_name:
            herb_lower = herb_name.lower()
            results = [i for i in results if herb_lower in i.herb.lower()]
        
        return results
    
    def fetch_pubchem_compound(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch compound data from PubChem.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Compound data dictionary or None
        """
        try:
            # PubChem PUG-REST API
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/JSON"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'PC_Compounds' in data and data['PC_Compounds']:
                    compound = data['PC_Compounds'][0]
                    
                    # Extract SMILES
                    smiles = None
                    if 'props' in compound:
                        for prop in compound['props']:
                            if prop.get('urn', {}).get('label') == 'SMILES' and prop.get('urn', {}).get('name') == 'Canonical':
                                smiles = prop.get('value', {}).get('sval')
                                break
                    
                    return {
                        'cid': compound.get('id', {}).get('id', {}).get('cid'),
                        'smiles': smiles,
                        'name': drug_name,
                        'source': 'PubChem'
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"PubChem fetch failed for {drug_name}: {e}")
            return None
    
    def fetch_chembl_data(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch bioactivity data from ChEMBL.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            ChEMBL data dictionary or None
        """
        try:
            # ChEMBL API
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={drug_name}&format=json"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'molecules' in data and data['molecules']:
                    mol = data['molecules'][0]
                    
                    return {
                        'chembl_id': mol.get('molecule_chembl_id'),
                        'name': mol.get('pref_name'),
                        'max_phase': mol.get('max_phase'),
                        'molecule_type': mol.get('molecule_type'),
                        'therapeutic_flag': mol.get('therapeutic_flag'),
                        'source': 'ChEMBL'
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"ChEMBL fetch failed for {drug_name}: {e}")
            return None
    
    def export_food_interactions_json(self, filepath: Optional[Path] = None) -> Path:
        """Export food-drug interactions to JSON file."""
        if filepath is None:
            filepath = DATA_DIR / 'food_drug_interactions.json'
        
        data = [
            {
                'food': i.food,
                'drug': i.drug,
                'mechanism': i.mechanism,
                'severity': i.severity,
                'effect': i.effect,
                'recommendation': i.recommendation
            }
            for i in self._food_interactions
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} food-drug interactions to {filepath}")
        return filepath
    
    def export_herbal_interactions_json(self, filepath: Optional[Path] = None) -> Path:
        """Export herbal-drug interactions to JSON file."""
        if filepath is None:
            filepath = DATA_DIR / 'herbal_drug_interactions.json'
        
        data = [
            {
                'herb': i.herb,
                'drug': i.drug,
                'mechanism': i.mechanism,
                'severity': i.severity,
                'effect': i.effect,
                'recommendation': i.recommendation
            }
            for i in self._herbal_interactions
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(data)} herbal-drug interactions to {filepath}")
        return filepath
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get statistics about available interaction data."""
        return {
            'food_drug_interactions': len(self._food_interactions),
            'herbal_drug_interactions': len(self._herbal_interactions),
            'unique_foods': len(set(i.food for i in self._food_interactions)),
            'unique_herbs': len(set(i.herb for i in self._herbal_interactions)),
            'unique_drugs_food': len(set(i.drug for i in self._food_interactions)),
            'unique_drugs_herbal': len(set(i.drug for i in self._herbal_interactions)),
            'severe_food_interactions': sum(1 for i in self._food_interactions if i.severity == 'severe'),
            'severe_herbal_interactions': sum(1 for i in self._herbal_interactions if i.severity == 'severe'),
        }


# Singleton instance
_enhanced_ingestion: Optional[EnhancedDataIngestion] = None


def get_enhanced_data_ingestion() -> EnhancedDataIngestion:
    """Get or create the enhanced data ingestion singleton."""
    global _enhanced_ingestion
    if _enhanced_ingestion is None:
        _enhanced_ingestion = EnhancedDataIngestion()
    return _enhanced_ingestion
