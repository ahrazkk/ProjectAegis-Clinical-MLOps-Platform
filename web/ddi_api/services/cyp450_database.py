"""
CYP450 Enzyme Metabolism Database

Provides comprehensive CYP450 enzyme metabolism data for drug interactions.
This includes substrates, inhibitors, and inducers for each major CYP enzyme.

Reference: 
- FDA Drug Development and Drug Interactions Table
- Flockhart Table (Indiana University)
- DrugBank metabolic data
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CYPEnzyme(str, Enum):
    """Major CYP450 enzymes involved in drug metabolism."""
    CYP1A2 = "CYP1A2"
    CYP2B6 = "CYP2B6"
    CYP2C8 = "CYP2C8"
    CYP2C9 = "CYP2C9"
    CYP2C19 = "CYP2C19"
    CYP2D6 = "CYP2D6"
    CYP2E1 = "CYP2E1"
    CYP3A4 = "CYP3A4"
    CYP3A5 = "CYP3A5"


class DrugRole(str, Enum):
    """Role of a drug in CYP metabolism."""
    SUBSTRATE = "substrate"
    INHIBITOR = "inhibitor"
    INDUCER = "inducer"


class InhibitionStrength(str, Enum):
    """Strength of CYP inhibition."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class CYPInteraction:
    """Represents a potential CYP-mediated drug interaction."""
    drug1: str
    drug2: str
    enzyme: CYPEnzyme
    mechanism: str
    severity: str
    clinical_effect: str
    management: str


# Comprehensive CYP450 drug database
# Based on FDA tables and clinical literature
CYP_DRUG_DATABASE: Dict[str, Dict[str, List[str]]] = {
    # Format: drug_name (lowercase): {enzyme: [roles]}
    
    # === CYP1A2 Substrates ===
    "caffeine": {"CYP1A2": ["substrate"]},
    "theophylline": {"CYP1A2": ["substrate"]},
    "clozapine": {"CYP1A2": ["substrate"]},
    "olanzapine": {"CYP1A2": ["substrate"]},
    "duloxetine": {"CYP1A2": ["substrate"]},
    "melatonin": {"CYP1A2": ["substrate"]},
    "propranolol": {"CYP1A2": ["substrate"], "CYP2D6": ["substrate"]},
    "ropinirole": {"CYP1A2": ["substrate"]},
    "tizanidine": {"CYP1A2": ["substrate"]},
    "ramelteon": {"CYP1A2": ["substrate"]},
    
    # === CYP1A2 Inhibitors ===
    "fluvoxamine": {"CYP1A2": ["inhibitor_strong"], "CYP2C19": ["inhibitor_strong"], "CYP3A4": ["inhibitor_moderate"]},
    "ciprofloxacin": {"CYP1A2": ["inhibitor_strong"]},
    "enoxacin": {"CYP1A2": ["inhibitor_strong"]},
    "mexiletine": {"CYP1A2": ["inhibitor_moderate"]},
    "oral contraceptives": {"CYP1A2": ["inhibitor_weak"]},
    
    # === CYP1A2 Inducers ===
    "tobacco": {"CYP1A2": ["inducer"]},
    "smoking": {"CYP1A2": ["inducer"]},
    "omeprazole": {"CYP1A2": ["inducer"]},
    "rifampin": {
        "CYP1A2": ["inducer"],
        "CYP2B6": ["inducer"],
        "CYP2C8": ["inducer"],
        "CYP2C9": ["inducer"],
        "CYP2C19": ["inducer"],
        "CYP3A4": ["inducer"]
    },
    "carbamazepine": {
        "CYP1A2": ["inducer"],
        "CYP2B6": ["inducer"],
        "CYP2C9": ["inducer"],
        "CYP3A4": ["inducer"]
    },
    "phenytoin": {
        "CYP1A2": ["inducer"],
        "CYP2B6": ["inducer"],
        "CYP2C9": ["inducer", "substrate"],
        "CYP2C19": ["inducer"],
        "CYP3A4": ["inducer"]
    },
    "phenobarbital": {
        "CYP1A2": ["inducer"],
        "CYP2B6": ["inducer"],
        "CYP2C9": ["inducer"],
        "CYP3A4": ["inducer"]
    },
    
    # === CYP2C9 Substrates ===
    "warfarin": {"CYP2C9": ["substrate"], "CYP3A4": ["substrate"]},
    "phenytoin": {"CYP2C9": ["substrate"], "CYP2C19": ["substrate"]},
    "celecoxib": {"CYP2C9": ["substrate"]},
    "fluvastatin": {"CYP2C9": ["substrate"]},
    "glipizide": {"CYP2C9": ["substrate"]},
    "glimepiride": {"CYP2C9": ["substrate"]},
    "glyburide": {"CYP2C9": ["substrate"]},
    "tolbutamide": {"CYP2C9": ["substrate"]},
    "losartan": {"CYP2C9": ["substrate"], "CYP3A4": ["substrate"]},
    "irbesartan": {"CYP2C9": ["substrate"]},
    "ibuprofen": {"CYP2C9": ["substrate"]},
    "naproxen": {"CYP2C9": ["substrate"]},
    "piroxicam": {"CYP2C9": ["substrate"]},
    "diclofenac": {"CYP2C9": ["substrate"]},
    
    # === CYP2C9 Inhibitors ===
    "fluconazole": {"CYP2C9": ["inhibitor_moderate"], "CYP2C19": ["inhibitor_moderate"], "CYP3A4": ["inhibitor_moderate"]},
    "amiodarone": {"CYP2C9": ["inhibitor_moderate"], "CYP2D6": ["inhibitor_moderate"], "CYP3A4": ["inhibitor_moderate"]},
    "miconazole": {"CYP2C9": ["inhibitor_strong"]},
    "voriconazole": {"CYP2C9": ["inhibitor_moderate"], "CYP2C19": ["inhibitor_strong"], "CYP3A4": ["inhibitor_strong"]},
    "sulfamethoxazole": {"CYP2C9": ["inhibitor_moderate"]},
    "metronidazole": {"CYP2C9": ["inhibitor_weak"]},
    
    # === CYP2C19 Substrates ===
    "clopidogrel": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},  # Prodrug activation
    "omeprazole": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    "esomeprazole": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    "lansoprazole": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    "pantoprazole": {"CYP2C19": ["substrate"]},
    "diazepam": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    "citalopram": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    "escitalopram": {"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
    
    # === CYP2C19 Inhibitors ===
    "fluoxetine": {"CYP2C19": ["inhibitor_moderate"], "CYP2D6": ["inhibitor_strong"], "CYP3A4": ["inhibitor_weak"]},
    "fluvoxamine": {"CYP2C19": ["inhibitor_strong"], "CYP1A2": ["inhibitor_strong"]},
    "ticlopidine": {"CYP2C19": ["inhibitor_strong"]},
    "esomeprazole": {"CYP2C19": ["inhibitor_moderate"]},
    
    # === CYP2D6 Substrates ===
    "codeine": {"CYP2D6": ["substrate"]},  # Prodrug - requires CYP2D6 for activation
    "tramadol": {"CYP2D6": ["substrate"]},
    "oxycodone": {"CYP2D6": ["substrate"]},
    "hydrocodone": {"CYP2D6": ["substrate"]},
    "metoprolol": {"CYP2D6": ["substrate"]},
    "carvedilol": {"CYP2D6": ["substrate"]},
    "timolol": {"CYP2D6": ["substrate"]},
    "atomoxetine": {"CYP2D6": ["substrate"]},
    "dextromethorphan": {"CYP2D6": ["substrate"]},
    "venlafaxine": {"CYP2D6": ["substrate"]},
    "duloxetine": {"CYP2D6": ["substrate"], "CYP1A2": ["substrate"]},
    "paroxetine": {"CYP2D6": ["substrate", "inhibitor_strong"]},
    "fluoxetine": {"CYP2D6": ["inhibitor_strong"]},
    "haloperidol": {"CYP2D6": ["substrate"]},
    "risperidone": {"CYP2D6": ["substrate"]},
    "aripiprazole": {"CYP2D6": ["substrate"], "CYP3A4": ["substrate"]},
    "tamoxifen": {"CYP2D6": ["substrate"]},  # Prodrug
    "nortriptyline": {"CYP2D6": ["substrate"]},
    "amitriptyline": {"CYP2D6": ["substrate"]},
    "desipramine": {"CYP2D6": ["substrate"]},
    
    # === CYP2D6 Inhibitors ===
    "quinidine": {"CYP2D6": ["inhibitor_strong"]},
    "bupropion": {"CYP2D6": ["inhibitor_strong"]},
    "cinacalcet": {"CYP2D6": ["inhibitor_strong"]},
    "sertraline": {"CYP2D6": ["inhibitor_moderate"]},
    "duloxetine": {"CYP2D6": ["inhibitor_moderate"]},
    "terbinafine": {"CYP2D6": ["inhibitor_strong"]},
    "diphenhydramine": {"CYP2D6": ["inhibitor_moderate"]},
    "cimetidine": {"CYP2D6": ["inhibitor_weak"]},
    
    # === CYP3A4 Substrates (largest group) ===
    "simvastatin": {"CYP3A4": ["substrate"]},
    "lovastatin": {"CYP3A4": ["substrate"]},
    "atorvastatin": {"CYP3A4": ["substrate"]},
    "amlodipine": {"CYP3A4": ["substrate"]},
    "nifedipine": {"CYP3A4": ["substrate"]},
    "diltiazem": {"CYP3A4": ["substrate", "inhibitor_moderate"]},
    "verapamil": {"CYP3A4": ["substrate", "inhibitor_moderate"]},
    "felodipine": {"CYP3A4": ["substrate"]},
    "cyclosporine": {"CYP3A4": ["substrate", "inhibitor_weak"]},
    "tacrolimus": {"CYP3A4": ["substrate"]},
    "sirolimus": {"CYP3A4": ["substrate"]},
    "midazolam": {"CYP3A4": ["substrate"]},
    "triazolam": {"CYP3A4": ["substrate"]},
    "alprazolam": {"CYP3A4": ["substrate"]},
    "buspirone": {"CYP3A4": ["substrate"]},
    "fentanyl": {"CYP3A4": ["substrate"]},
    "methadone": {"CYP3A4": ["substrate"]},
    "sildenafil": {"CYP3A4": ["substrate"]},
    "tadalafil": {"CYP3A4": ["substrate"]},
    "vardenafil": {"CYP3A4": ["substrate"]},
    "carbamazepine": {"CYP3A4": ["substrate", "inducer"]},
    "quetiapine": {"CYP3A4": ["substrate"]},
    "apixaban": {"CYP3A4": ["substrate"]},
    "rivaroxaban": {"CYP3A4": ["substrate"]},
    "colchicine": {"CYP3A4": ["substrate"]},
    "ergotamine": {"CYP3A4": ["substrate"]},
    
    # === CYP3A4 Strong Inhibitors ===
    "ketoconazole": {"CYP3A4": ["inhibitor_strong"]},
    "itraconazole": {"CYP3A4": ["inhibitor_strong"]},
    "posaconazole": {"CYP3A4": ["inhibitor_strong"]},
    "clarithromycin": {"CYP3A4": ["inhibitor_strong"]},
    "telithromycin": {"CYP3A4": ["inhibitor_strong"]},
    "ritonavir": {"CYP3A4": ["inhibitor_strong"], "CYP2D6": ["inhibitor_strong"]},
    "nelfinavir": {"CYP3A4": ["inhibitor_strong"]},
    "cobicistat": {"CYP3A4": ["inhibitor_strong"]},
    "grapefruit juice": {"CYP3A4": ["inhibitor_strong"]},  # Food interaction
    "grapefruit": {"CYP3A4": ["inhibitor_strong"]},
    
    # === CYP3A4 Moderate Inhibitors ===
    "erythromycin": {"CYP3A4": ["inhibitor_moderate"]},
    "fluconazole": {"CYP3A4": ["inhibitor_moderate"]},
    "aprepitant": {"CYP3A4": ["inhibitor_moderate"]},
    "verapamil": {"CYP3A4": ["inhibitor_moderate", "substrate"]},
    "diltiazem": {"CYP3A4": ["inhibitor_moderate", "substrate"]},
    "cimetidine": {"CYP3A4": ["inhibitor_weak"]},
    
    # === CYP3A4 Inducers ===
    "rifampin": {"CYP3A4": ["inducer"]},
    "rifabutin": {"CYP3A4": ["inducer"]},
    "phenytoin": {"CYP3A4": ["inducer"]},
    "carbamazepine": {"CYP3A4": ["inducer", "substrate"]},
    "phenobarbital": {"CYP3A4": ["inducer"]},
    "st. john's wort": {"CYP3A4": ["inducer"]},  # Herbal
    "efavirenz": {"CYP3A4": ["inducer"], "CYP2B6": ["substrate"]},
    "nevirapine": {"CYP3A4": ["inducer"]},
    "modafinil": {"CYP3A4": ["inducer"]},
    "bosentan": {"CYP3A4": ["inducer", "substrate"]},
    
    # === Additional commonly used drugs ===
    "aspirin": {},  # Not primarily CYP metabolized
    "digoxin": {},  # P-gp substrate, minimal CYP
    "metformin": {},  # Not CYP metabolized
    "lisinopril": {},  # Not metabolized
    "amlodipine": {"CYP3A4": ["substrate"]},
    "gabapentin": {},  # Not metabolized
    "pregabalin": {},  # Not metabolized
    "levothyroxine": {},  # Not CYP metabolized
}


class CYP450Database:
    """
    Database for CYP450 enzyme-drug interactions.
    
    Provides methods to query drug metabolism pathways and predict
    pharmacokinetic drug-drug interactions.
    """
    
    def __init__(self):
        """Initialize the CYP450 database."""
        self._drug_data = CYP_DRUG_DATABASE
        self._build_indices()
    
    def _build_indices(self):
        """Build reverse indices for efficient lookups."""
        # Index: enzyme -> {substrates, inhibitors, inducers}
        self._enzyme_index: Dict[str, Dict[str, Set[str]]] = {}
        
        for enzyme in CYPEnzyme:
            self._enzyme_index[enzyme.value] = {
                'substrates': set(),
                'inhibitors_strong': set(),
                'inhibitors_moderate': set(),
                'inhibitors_weak': set(),
                'inducers': set()
            }
        
        for drug, enzymes in self._drug_data.items():
            for enzyme, roles in enzymes.items():
                for role in roles:
                    if role == 'substrate':
                        self._enzyme_index[enzyme]['substrates'].add(drug)
                    elif role == 'inhibitor_strong':
                        self._enzyme_index[enzyme]['inhibitors_strong'].add(drug)
                    elif role == 'inhibitor_moderate':
                        self._enzyme_index[enzyme]['inhibitors_moderate'].add(drug)
                    elif role == 'inhibitor_weak':
                        self._enzyme_index[enzyme]['inhibitors_weak'].add(drug)
                    elif role == 'inducer':
                        self._enzyme_index[enzyme]['inducers'].add(drug)
                    elif role.startswith('inhibitor'):
                        # Handle simple 'inhibitor' without strength
                        self._enzyme_index[enzyme]['inhibitors_moderate'].add(drug)
    
    def get_drug_cyp_profile(self, drug_name: str) -> Dict[str, List[str]]:
        """
        Get the CYP450 profile for a drug.
        
        Args:
            drug_name: Name of the drug (case-insensitive)
            
        Returns:
            Dictionary mapping enzymes to roles
        """
        drug_lower = drug_name.lower().strip()
        return self._drug_data.get(drug_lower, {})
    
    def get_substrates(self, enzyme: str) -> Set[str]:
        """Get all substrates for a CYP enzyme."""
        return self._enzyme_index.get(enzyme, {}).get('substrates', set())
    
    def get_inhibitors(self, enzyme: str, strength: Optional[str] = None) -> Set[str]:
        """
        Get inhibitors for a CYP enzyme.
        
        Args:
            enzyme: CYP enzyme name (e.g., 'CYP3A4')
            strength: Optional filter by strength ('strong', 'moderate', 'weak')
            
        Returns:
            Set of drug names
        """
        enzyme_data = self._enzyme_index.get(enzyme, {})
        
        if strength:
            return enzyme_data.get(f'inhibitors_{strength}', set())
        else:
            # Return all inhibitors
            result = set()
            result.update(enzyme_data.get('inhibitors_strong', set()))
            result.update(enzyme_data.get('inhibitors_moderate', set()))
            result.update(enzyme_data.get('inhibitors_weak', set()))
            return result
    
    def get_inducers(self, enzyme: str) -> Set[str]:
        """Get all inducers for a CYP enzyme."""
        return self._enzyme_index.get(enzyme, {}).get('inducers', set())
    
    def check_cyp_interaction(
        self,
        drug1: str,
        drug2: str
    ) -> List[CYPInteraction]:
        """
        Check for CYP-mediated interactions between two drugs.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            List of potential CYP interactions
        """
        interactions = []
        
        profile1 = self.get_drug_cyp_profile(drug1)
        profile2 = self.get_drug_cyp_profile(drug2)
        
        if not profile1 and not profile2:
            return interactions
        
        # Check each enzyme
        for enzyme in CYPEnzyme:
            enzyme_name = enzyme.value
            
            roles1 = profile1.get(enzyme_name, [])
            roles2 = profile2.get(enzyme_name, [])
            
            # Check if drug1 inhibits drug2's metabolism
            for role1 in roles1:
                for role2 in roles2:
                    interaction = self._evaluate_role_interaction(
                        drug1, role1, drug2, role2, enzyme
                    )
                    if interaction:
                        interactions.append(interaction)
            
            # Check if drug2 inhibits drug1's metabolism
            for role2 in roles2:
                for role1 in roles1:
                    interaction = self._evaluate_role_interaction(
                        drug2, role2, drug1, role1, enzyme
                    )
                    if interaction:
                        # Avoid duplicates
                        if not any(
                            i.drug1 == interaction.drug1 and 
                            i.drug2 == interaction.drug2 and 
                            i.enzyme == interaction.enzyme 
                            for i in interactions
                        ):
                            interactions.append(interaction)
        
        return interactions
    
    def _evaluate_role_interaction(
        self,
        perpetrator: str,
        perp_role: str,
        victim: str,
        victim_role: str,
        enzyme: CYPEnzyme
    ) -> Optional[CYPInteraction]:
        """
        Evaluate if two drug roles create an interaction.
        
        Args:
            perpetrator: Drug causing the interaction
            perp_role: Role of perpetrator (inhibitor/inducer)
            victim: Drug affected by interaction
            victim_role: Role of victim (usually substrate)
            enzyme: CYP enzyme involved
            
        Returns:
            CYPInteraction if interaction exists, None otherwise
        """
        # Inhibitor + Substrate = Increased substrate levels
        if 'inhibitor' in perp_role and victim_role == 'substrate':
            strength = 'strong' if 'strong' in perp_role else (
                'moderate' if 'moderate' in perp_role else 'weak'
            )
            
            severity_map = {'strong': 'severe', 'moderate': 'major', 'weak': 'moderate'}
            severity = severity_map.get(strength, 'moderate')
            
            return CYPInteraction(
                drug1=perpetrator,
                drug2=victim,
                enzyme=enzyme,
                mechanism=f"{perpetrator} is a {strength} inhibitor of {enzyme.value}, which metabolizes {victim}. This can significantly increase {victim} blood levels.",
                severity=severity,
                clinical_effect=f"Increased {victim} plasma concentrations, potentially leading to toxicity",
                management=f"Consider dose reduction of {victim} or use alternative therapy. Monitor for {victim} toxicity."
            )
        
        # Inducer + Substrate = Decreased substrate levels
        if perp_role == 'inducer' and victim_role == 'substrate':
            return CYPInteraction(
                drug1=perpetrator,
                drug2=victim,
                enzyme=enzyme,
                mechanism=f"{perpetrator} induces {enzyme.value}, which metabolizes {victim}. This can significantly decrease {victim} blood levels.",
                severity='major',
                clinical_effect=f"Decreased {victim} plasma concentrations, potentially leading to therapeutic failure",
                management=f"Consider dose increase of {victim} or use alternative therapy. Monitor for reduced efficacy."
            )
        
        return None
    
    def get_high_risk_combinations(self) -> List[Dict]:
        """
        Get list of known high-risk drug combinations based on CYP interactions.
        
        Returns:
            List of high-risk drug pair dictionaries
        """
        high_risk = []
        
        # CYP3A4 strong inhibitors with sensitive substrates
        cyp3a4_strong_inhibitors = self.get_inhibitors('CYP3A4', 'strong')
        cyp3a4_sensitive_substrates = {
            'simvastatin', 'lovastatin', 'midazolam', 'triazolam',
            'fentanyl', 'ergotamine', 'colchicine', 'tacrolimus'
        }
        
        for inhibitor in cyp3a4_strong_inhibitors:
            for substrate in cyp3a4_sensitive_substrates:
                if inhibitor != substrate:
                    high_risk.append({
                        'drug1': inhibitor,
                        'drug2': substrate,
                        'enzyme': 'CYP3A4',
                        'severity': 'severe',
                        'mechanism': f'{inhibitor} (strong CYP3A4 inhibitor) + {substrate} (sensitive substrate)',
                        'recommendation': 'Avoid combination or use alternative therapy'
                    })
        
        # CYP2D6 inhibitors + codeine/tamoxifen (prodrugs)
        cyp2d6_inhibitors = self.get_inhibitors('CYP2D6', 'strong')
        prodrugs = {'codeine', 'tramadol', 'tamoxifen'}
        
        for inhibitor in cyp2d6_inhibitors:
            for prodrug in prodrugs:
                if inhibitor != prodrug:
                    high_risk.append({
                        'drug1': inhibitor,
                        'drug2': prodrug,
                        'enzyme': 'CYP2D6',
                        'severity': 'major',
                        'mechanism': f'{inhibitor} (CYP2D6 inhibitor) blocks activation of {prodrug} (prodrug)',
                        'recommendation': 'Reduced therapeutic effect - consider alternative analgesic'
                    })
        
        return high_risk


# Singleton instance
_cyp450_db: Optional[CYP450Database] = None


def get_cyp450_database() -> CYP450Database:
    """Get or create the CYP450 database singleton."""
    global _cyp450_db
    if _cyp450_db is None:
        _cyp450_db = CYP450Database()
    return _cyp450_db
