"""
Offline Training Data for DDI Model

This module provides a comprehensive, self-contained dataset for training
drug-drug interaction models WITHOUT requiring external APIs like DrugBank.

Data is curated from:
- FDA Drug Interaction Tables
- Published clinical literature
- DrugBank open data (static, bundled)
- DDI Corpus annotations

This allows training models locally without API dependencies.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Output directory for generated training data
DATA_DIR = Path(__file__).parent.parent.parent / 'data'


@dataclass
class DrugInfo:
    """Complete drug information for training."""
    drugbank_id: str
    name: str
    smiles: str
    category: str
    targets: List[str] = field(default_factory=list)
    enzymes: Dict[str, List[str]] = field(default_factory=dict)  # CYP enzymes
    description: str = ""


@dataclass
class InteractionRecord:
    """Drug-drug interaction training record."""
    drug1_id: str
    drug2_id: str
    drug1_name: str
    drug2_name: str
    severity: str  # severe, major, moderate, minor, none
    interaction_type: str  # mechanism, effect, advise, int
    mechanism: str
    clinical_effect: str
    management: str


# ============================================================================
# Comprehensive Drug Database (No API Required)
# SMILES structures from PubChem (public domain)
# ============================================================================

DRUG_DATABASE: List[DrugInfo] = [
    # === Anticoagulants ===
    DrugInfo(
        drugbank_id="DB00682",
        name="Warfarin",
        smiles="CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
        category="Anticoagulant",
        enzymes={"CYP2C9": ["substrate"], "CYP3A4": ["substrate"]},
        description="Vitamin K antagonist anticoagulant"
    ),
    DrugInfo(
        drugbank_id="DB01418",
        name="Acenocoumarol",
        smiles="CC(=O)CC(C1=CC=C(C=C1)[N+](=O)[O-])C2=C(C3=CC=CC=C3OC2=O)O",
        category="Anticoagulant",
        enzymes={"CYP2C9": ["substrate"]},
        description="Vitamin K antagonist anticoagulant"
    ),
    
    # === NSAIDs ===
    DrugInfo(
        drugbank_id="DB00945",
        name="Aspirin",
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        category="NSAID",
        enzymes={},
        description="Non-steroidal anti-inflammatory drug, COX inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01050",
        name="Ibuprofen",
        smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        category="NSAID",
        enzymes={"CYP2C9": ["substrate"]},
        description="Propionic acid NSAID"
    ),
    DrugInfo(
        drugbank_id="DB00788",
        name="Naproxen",
        smiles="COC1=CC2=CC(C(C)C(=O)O)=CC=C2C=C1",
        category="NSAID",
        enzymes={"CYP2C9": ["substrate"]},
        description="Propionic acid NSAID"
    ),
    DrugInfo(
        drugbank_id="DB00586",
        name="Diclofenac",
        smiles="OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl",
        category="NSAID",
        enzymes={"CYP2C9": ["substrate"]},
        description="Phenylacetic acid NSAID"
    ),
    DrugInfo(
        drugbank_id="DB00465",
        name="Ketorolac",
        smiles="OC(=O)C1CCN2C1=CC=C2C(=O)C3=CC=CC=C3",
        category="NSAID",
        enzymes={"CYP2C9": ["substrate"]},
        description="Pyrrolizine NSAID"
    ),
    DrugInfo(
        drugbank_id="DB00328",
        name="Indomethacin",
        smiles="COC1=CC2=C(C=C1)C(C)(CC(=O)O)C(=O)N2C3=CC=C(Cl)C=C3",
        category="NSAID",
        enzymes={"CYP2C9": ["substrate"]},
        description="Indoleacetic acid NSAID"
    ),
    
    # === Statins ===
    DrugInfo(
        drugbank_id="DB00641",
        name="Simvastatin",
        smiles="CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
        category="Statin",
        enzymes={"CYP3A4": ["substrate"]},
        description="HMG-CoA reductase inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01076",
        name="Atorvastatin",
        smiles="CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CCC(CC(CC(=O)O)O)O)C3=CC=C(F)C=C3)C4=CC=CC=C4",
        category="Statin",
        enzymes={"CYP3A4": ["substrate"]},
        description="HMG-CoA reductase inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00175",
        name="Pravastatin",
        smiles="CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C12",
        category="Statin",
        enzymes={},  # Not CYP metabolized
        description="HMG-CoA reductase inhibitor (hydrophilic)"
    ),
    DrugInfo(
        drugbank_id="DB01098",
        name="Rosuvastatin",
        smiles="CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(F)C=C2)N(C)S(=O)(=O)C",
        category="Statin",
        enzymes={"CYP2C9": ["substrate"]},
        description="HMG-CoA reductase inhibitor (hydrophilic)"
    ),
    DrugInfo(
        drugbank_id="DB00227",
        name="Lovastatin",
        smiles="CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
        category="Statin",
        enzymes={"CYP3A4": ["substrate"]},
        description="HMG-CoA reductase inhibitor"
    ),
    
    # === Antifungals (CYP inhibitors) ===
    DrugInfo(
        drugbank_id="DB01026",
        name="Ketoconazole",
        smiles="CC(=O)N1CCN(CC1)C2=CC=C(OCC3COC(CN4C=NC=N4)(O3)C5=CC=C(Cl)C=C5Cl)C=C2",
        category="Antifungal",
        enzymes={"CYP3A4": ["inhibitor_strong"]},
        description="Imidazole antifungal, potent CYP3A4 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01167",
        name="Itraconazole",
        smiles="CCC(C)N1N=CN(C1=O)C2=CC=C(OCC3COC(CN4C=NC=N4)(O3)C5=CC=C(Cl)C=C5Cl)C=C2",
        category="Antifungal",
        enzymes={"CYP3A4": ["inhibitor_strong"]},
        description="Triazole antifungal, potent CYP3A4 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00196",
        name="Fluconazole",
        smiles="OC(CN1C=NC=N1)(CN2C=NC=N2)C3=CC=C(F)C=C3F",
        category="Antifungal",
        enzymes={"CYP2C9": ["inhibitor_moderate"], "CYP2C19": ["inhibitor_moderate"], "CYP3A4": ["inhibitor_moderate"]},
        description="Triazole antifungal, moderate CYP inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00582",
        name="Voriconazole",
        smiles="CC(C1=NC=NC=C1F)C(O)(CN2C=NC=N2)C3=CC=C(F)C=C3",
        category="Antifungal",
        enzymes={"CYP2C9": ["inhibitor_moderate"], "CYP2C19": ["inhibitor_strong"], "CYP3A4": ["inhibitor_strong"]},
        description="Triazole antifungal, potent CYP inhibitor"
    ),
    
    # === Antibiotics ===
    DrugInfo(
        drugbank_id="DB00537",
        name="Ciprofloxacin",
        smiles="OC(=O)C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O",
        category="Antibiotic",
        enzymes={"CYP1A2": ["inhibitor_strong"]},
        description="Fluoroquinolone antibiotic, CYP1A2 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01211",
        name="Clarithromycin",
        smiles="CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(N(C)C)C3O)C(C)(CC(C)C(=O)C(C)C(O)C1(C)O)OC",
        category="Antibiotic",
        enzymes={"CYP3A4": ["inhibitor_strong", "substrate"]},
        description="Macrolide antibiotic, potent CYP3A4 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00199",
        name="Erythromycin",
        smiles="CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(N(C)C)C3O)C(C)(CC(C)C(=O)C(C)C(O)C1(C)O)O",
        category="Antibiotic",
        enzymes={"CYP3A4": ["inhibitor_moderate", "substrate"]},
        description="Macrolide antibiotic, moderate CYP3A4 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01369",
        name="Quinidine",
        smiles="COC1=CC2=C(C=CN=C2C=C1)C(O)C3CC4CCN3CC4C=C",
        category="Antiarrhythmic",
        enzymes={"CYP2D6": ["inhibitor_strong"], "CYP3A4": ["substrate"]},
        description="Class IA antiarrhythmic, potent CYP2D6 inhibitor"
    ),
    
    # === Cardiac Drugs ===
    DrugInfo(
        drugbank_id="DB00390",
        name="Digoxin",
        smiles="CC1OC(OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC7(C)C(C8=CC(=O)OC8)CCC67)C4)OC3C)OC2C)C(O)C(O)C1O",
        category="Cardiac Glycoside",
        enzymes={},  # P-gp substrate, minimal CYP
        description="Cardiac glycoside, P-glycoprotein substrate"
    ),
    DrugInfo(
        drugbank_id="DB01118",
        name="Amiodarone",
        smiles="CCCCC1=C(C2=CC=C(OCCN(CC)CC)C=C2)C3=CC(I)=C(OCCC)C(I)=C3O1",
        category="Antiarrhythmic",
        enzymes={"CYP2C9": ["inhibitor_moderate"], "CYP2D6": ["inhibitor_moderate"], "CYP3A4": ["inhibitor_moderate"]},
        description="Class III antiarrhythmic, multiple CYP inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00661",
        name="Verapamil",
        smiles="COC1=CC=C(CCN(C)CCCC(C#N)(C(C)C)C2=CC(OC)=C(OC)C=C2)C=C1OC",
        category="Calcium Channel Blocker",
        enzymes={"CYP3A4": ["inhibitor_moderate", "substrate"]},
        description="Calcium channel blocker, CYP3A4 inhibitor/substrate"
    ),
    DrugInfo(
        drugbank_id="DB00343",
        name="Diltiazem",
        smiles="COC1=CC=C(C=C1)C2SC3=CC=CC=C3N(CCN(C)C)C(=O)C2OC(C)=O",
        category="Calcium Channel Blocker",
        enzymes={"CYP3A4": ["inhibitor_moderate", "substrate"]},
        description="Calcium channel blocker, CYP3A4 inhibitor/substrate"
    ),
    
    # === Beta Blockers ===
    DrugInfo(
        drugbank_id="DB00571",
        name="Propranolol",
        smiles="CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12",
        category="Beta Blocker",
        enzymes={"CYP1A2": ["substrate"], "CYP2D6": ["substrate"]},
        description="Non-selective beta blocker"
    ),
    DrugInfo(
        drugbank_id="DB00264",
        name="Metoprolol",
        smiles="COCCC1=CC=C(OCC(O)CNC(C)C)C=C1",
        category="Beta Blocker",
        enzymes={"CYP2D6": ["substrate"]},
        description="Cardioselective beta blocker"
    ),
    DrugInfo(
        drugbank_id="DB01193",
        name="Acebutolol",
        smiles="CCCC(=O)NC1=CC(OCC(O)CNC(C)C)=C(C=C1)C(C)=O",
        category="Beta Blocker",
        enzymes={"CYP2D6": ["substrate"]},
        description="Cardioselective beta blocker with ISA"
    ),
    
    # === SSRIs ===
    DrugInfo(
        drugbank_id="DB00472",
        name="Fluoxetine",
        smiles="CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C2=CC=CC=C2",
        category="SSRI",
        enzymes={"CYP2C19": ["inhibitor_moderate"], "CYP2D6": ["inhibitor_strong"], "CYP3A4": ["inhibitor_weak"]},
        description="SSRI antidepressant, potent CYP2D6 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00715",
        name="Paroxetine",
        smiles="FC1=CC=C(C2CCNCC2COC3=CC4=C(OCO4)C=C3)C=C1",
        category="SSRI",
        enzymes={"CYP2D6": ["inhibitor_strong", "substrate"]},
        description="SSRI antidepressant, potent CYP2D6 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00176",
        name="Fluvoxamine",
        smiles="COCCCC/C(=N\\OCCN)C1=CC=C(C=C1)C(F)(F)F",
        category="SSRI",
        enzymes={"CYP1A2": ["inhibitor_strong"], "CYP2C19": ["inhibitor_strong"], "CYP3A4": ["inhibitor_moderate"]},
        description="SSRI antidepressant, potent CYP1A2/2C19 inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01104",
        name="Sertraline",
        smiles="CNC1CCC(C2=CC=C(Cl)C(Cl)=C2)C3=CC=CC=C13",
        category="SSRI",
        enzymes={"CYP2D6": ["inhibitor_moderate"]},
        description="SSRI antidepressant, moderate CYP2D6 inhibitor"
    ),
    
    # === Opioids ===
    DrugInfo(
        drugbank_id="DB00318",
        name="Codeine",
        smiles="COC1=CC=C2C3=C1OC4C5C=CC(O)C3C5CCN4C2",
        category="Opioid",
        enzymes={"CYP2D6": ["substrate"]},  # Prodrug activation
        description="Opioid analgesic, CYP2D6 converts to morphine"
    ),
    DrugInfo(
        drugbank_id="DB00497",
        name="Oxycodone",
        smiles="CN1CCC23C4OC5=C(O)C=CC(=C25)CC1C3C=CC4O",
        category="Opioid",
        enzymes={"CYP2D6": ["substrate"], "CYP3A4": ["substrate"]},
        description="Opioid analgesic"
    ),
    DrugInfo(
        drugbank_id="DB00327",
        name="Tramadol",
        smiles="COC1=CC=CC(C2(O)CCCCC2CN(C)C)=C1",
        category="Opioid",
        enzymes={"CYP2D6": ["substrate"], "CYP3A4": ["substrate"]},
        description="Atypical opioid analgesic"
    ),
    DrugInfo(
        drugbank_id="DB00813",
        name="Fentanyl",
        smiles="CCC(=O)N(C1CCN(CCC2=CC=CC=C2)CC1)C3=CC=CC=C3",
        category="Opioid",
        enzymes={"CYP3A4": ["substrate"]},
        description="Potent synthetic opioid"
    ),
    
    # === Anticonvulsants ===
    DrugInfo(
        drugbank_id="DB00564",
        name="Carbamazepine",
        smiles="NC(=O)N1C2=CC=CC=C2C=CC3=CC=CC=C13",
        category="Anticonvulsant",
        enzymes={"CYP3A4": ["inducer", "substrate"]},
        description="Anticonvulsant, potent CYP3A4 inducer"
    ),
    DrugInfo(
        drugbank_id="DB00252",
        name="Phenytoin",
        smiles="C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3",
        category="Anticonvulsant",
        enzymes={"CYP2C9": ["inducer", "substrate"], "CYP2C19": ["inducer", "substrate"], "CYP3A4": ["inducer"]},
        description="Anticonvulsant, potent CYP inducer"
    ),
    DrugInfo(
        drugbank_id="DB01174",
        name="Phenobarbital",
        smiles="CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2",
        category="Barbiturate",
        enzymes={"CYP1A2": ["inducer"], "CYP2B6": ["inducer"], "CYP2C9": ["inducer"], "CYP3A4": ["inducer"]},
        description="Barbiturate, potent pan-CYP inducer"
    ),
    
    # === Rifamycins ===
    DrugInfo(
        drugbank_id="DB01045",
        name="Rifampin",
        smiles="COC1=C2C3=C(C(=C1C)O)C(=O)C(=C(NC(=O)C(C)=CC=CC(C)C(O)C(C)C(O)C(C)C(OC(C)=O)C4OC(C)(O4)CC(C)=O)C=N3)CC2",
        category="Antibiotic",
        enzymes={"CYP1A2": ["inducer"], "CYP2B6": ["inducer"], "CYP2C8": ["inducer"], "CYP2C9": ["inducer"], "CYP2C19": ["inducer"], "CYP3A4": ["inducer"]},
        description="Rifamycin antibiotic, most potent CYP inducer"
    ),
    
    # === Immunosuppressants ===
    DrugInfo(
        drugbank_id="DB00864",
        name="Tacrolimus",
        smiles="COC1CC(CCC1OC)CC(C)C2CC(=O)C(C(C(CC(C(C(C(=O)C(C(=O)OC(C(C(CC(=O)C(C=C2C)C)O)OC)C(C)CC=CC)C)O)OC)C)C)O)C",
        category="Immunosuppressant",
        enzymes={"CYP3A4": ["substrate"]},
        description="Calcineurin inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB00091",
        name="Cyclosporine",
        smiles="CCC1NC(=O)C(C(O)C(C)CC=CCCC(C)CC2N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)OC(=O)C(NC(=O)C(NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C(=O)C(C)NC1=O)C(C)C)C(C)C)CC2=O",
        category="Immunosuppressant",
        enzymes={"CYP3A4": ["substrate", "inhibitor_weak"]},
        description="Calcineurin inhibitor"
    ),
    
    # === PPIs ===
    DrugInfo(
        drugbank_id="DB00338",
        name="Omeprazole",
        smiles="COC1=CC2=NC(CS(=O)C3=NC4=CC=CC=C4N3C)=NC(OC)=C2C=C1",
        category="PPI",
        enzymes={"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
        description="Proton pump inhibitor"
    ),
    DrugInfo(
        drugbank_id="DB01129",
        name="Rabeprazole",
        smiles="CCCOC1=CC=NC(CS(=O)C2=NC3=CC=CC=C3N2)=C1C",
        category="PPI",
        enzymes={"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
        description="Proton pump inhibitor"
    ),
    
    # === Benzodiazepines ===
    DrugInfo(
        drugbank_id="DB00683",
        name="Midazolam",
        smiles="CC1=NC=C2N1C3=CC=C(F)C=C3C(=NC2)C4=CC=CC=C4F",
        category="Benzodiazepine",
        enzymes={"CYP3A4": ["substrate"]},
        description="Short-acting benzodiazepine, CYP3A4 probe substrate"
    ),
    DrugInfo(
        drugbank_id="DB00404",
        name="Alprazolam",
        smiles="CC1=NN=C2CN=C(C3=CC=CC=C3F)C4=CC(Cl)=CC=C4N12",
        category="Benzodiazepine",
        enzymes={"CYP3A4": ["substrate"]},
        description="Triazolobenzodiazepine"
    ),
    DrugInfo(
        drugbank_id="DB00829",
        name="Diazepam",
        smiles="CN1C(=O)CN=C(C2=CC=CC=C2)C3=CC(Cl)=CC=C13",
        category="Benzodiazepine",
        enzymes={"CYP2C19": ["substrate"], "CYP3A4": ["substrate"]},
        description="Long-acting benzodiazepine"
    ),
    
    # === Additional important drugs ===
    DrugInfo(
        drugbank_id="DB00316",
        name="Acetaminophen",
        smiles="CC(=O)NC1=CC=C(C=C1)O",
        category="Analgesic",
        enzymes={"CYP2E1": ["substrate"]},
        description="Non-opioid analgesic/antipyretic"
    ),
    DrugInfo(
        drugbank_id="DB00563",
        name="Methotrexate",
        smiles="CN(CC1=CN=C2N=C(N)N=C(N)C2=N1)C3=CC=C(C(=O)NC(CCC(=O)O)C(=O)O)C=C3",
        category="Antimetabolite",
        enzymes={},  # Not CYP metabolized
        description="Folate antagonist"
    ),
    DrugInfo(
        drugbank_id="DB00959",
        name="Methylprednisolone",
        smiles="CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO",
        category="Corticosteroid",
        enzymes={"CYP3A4": ["substrate"]},
        description="Synthetic corticosteroid"
    ),
    DrugInfo(
        drugbank_id="DB00999",
        name="Hydrochlorothiazide",
        smiles="NS(=O)(=O)C1=CC2=C(NCNS2(=O)=O)C=C1Cl",
        category="Diuretic",
        enzymes={},  # Not CYP metabolized
        description="Thiazide diuretic"
    ),
    DrugInfo(
        drugbank_id="DB00668",
        name="Sildenafil",
        smiles="CCCC1=NN(C)C2=C1N=C(NC2=O)C3=CC(S(=O)(=O)N4CCN(C)CC4)=CC=C3OCC",
        category="PDE5 Inhibitor",
        enzymes={"CYP3A4": ["substrate"]},
        description="Phosphodiesterase-5 inhibitor"
    ),
]


# ============================================================================
# Drug-Drug Interaction Database
# Curated from clinical literature and FDA tables
# ============================================================================

INTERACTION_DATABASE: List[InteractionRecord] = [
    # === Warfarin Interactions (High Risk) ===
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB00945",
        drug1_name="Warfarin", drug2_name="Aspirin",
        severity="severe",
        interaction_type="effect",
        mechanism="Aspirin inhibits platelet aggregation and may displace warfarin from plasma proteins",
        clinical_effect="Significantly increased bleeding risk, including GI and intracranial hemorrhage",
        management="Avoid combination if possible. If necessary, use lowest aspirin dose and monitor INR closely"
    ),
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB01050",
        drug1_name="Warfarin", drug2_name="Ibuprofen",
        severity="severe",
        interaction_type="effect",
        mechanism="NSAIDs inhibit platelet function and may displace warfarin from protein binding",
        clinical_effect="Increased bleeding risk, particularly GI bleeding",
        management="Avoid combination. If NSAID needed, consider acetaminophen or use shortest duration possible"
    ),
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB00196",
        drug1_name="Warfarin", drug2_name="Fluconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Fluconazole inhibits CYP2C9, the primary enzyme metabolizing S-warfarin",
        clinical_effect="Markedly elevated INR, high bleeding risk",
        management="Reduce warfarin dose by 25-50%. Monitor INR within 3-5 days"
    ),
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB01045",
        drug1_name="Warfarin", drug2_name="Rifampin",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Rifampin induces CYP2C9 and CYP3A4, dramatically increasing warfarin metabolism",
        clinical_effect="Markedly decreased INR, loss of anticoagulation, thrombotic risk",
        management="May need 2-5x warfarin dose increase. Monitor INR frequently. Consider alternative antibiotic"
    ),
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB01118",
        drug1_name="Warfarin", drug2_name="Amiodarone",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Amiodarone inhibits CYP2C9 and CYP3A4, reducing warfarin clearance",
        clinical_effect="Significantly elevated INR, high bleeding risk. Effect may persist weeks after amiodarone stopped",
        management="Reduce warfarin dose by 30-50%. Effect develops over 1-2 weeks and persists for months"
    ),
    InteractionRecord(
        drug1_id="DB00682", drug2_id="DB00564",
        drug1_name="Warfarin", drug2_name="Carbamazepine",
        severity="major",
        interaction_type="mechanism",
        mechanism="Carbamazepine induces CYP2C9 and CYP3A4, increasing warfarin metabolism",
        clinical_effect="Decreased INR, reduced anticoagulation",
        management="Monitor INR closely when starting/stopping carbamazepine. May need warfarin dose increase"
    ),
    
    # === Statin Interactions ===
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB01026",
        drug1_name="Simvastatin", drug2_name="Ketoconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Ketoconazole is a potent CYP3A4 inhibitor, dramatically reducing simvastatin metabolism",
        clinical_effect="10-20x increase in simvastatin levels, high risk of rhabdomyolysis and hepatotoxicity",
        management="CONTRAINDICATED. Use pravastatin or rosuvastatin instead"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB01167",
        drug1_name="Simvastatin", drug2_name="Itraconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Itraconazole is a potent CYP3A4 inhibitor",
        clinical_effect="Markedly elevated simvastatin levels, rhabdomyolysis risk",
        management="CONTRAINDICATED. Use pravastatin or rosuvastatin"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB01211",
        drug1_name="Simvastatin", drug2_name="Clarithromycin",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Clarithromycin inhibits CYP3A4, reducing simvastatin metabolism",
        clinical_effect="Significantly elevated statin levels, myopathy/rhabdomyolysis risk",
        management="Avoid combination. Use azithromycin or switch to pravastatin"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB01118",
        drug1_name="Simvastatin", drug2_name="Amiodarone",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Amiodarone inhibits CYP3A4 and P-glycoprotein",
        clinical_effect="Elevated simvastatin levels, myopathy risk",
        management="Limit simvastatin to 20mg/day with amiodarone"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB00661",
        drug1_name="Simvastatin", drug2_name="Verapamil",
        severity="major",
        interaction_type="mechanism",
        mechanism="Verapamil inhibits CYP3A4, reducing simvastatin metabolism",
        clinical_effect="2-4x increase in simvastatin levels, increased myopathy risk",
        management="Limit simvastatin to 10mg/day with verapamil"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB00343",
        drug1_name="Simvastatin", drug2_name="Diltiazem",
        severity="major",
        interaction_type="mechanism",
        mechanism="Diltiazem inhibits CYP3A4",
        clinical_effect="Elevated simvastatin levels, myopathy risk",
        management="Limit simvastatin to 10mg/day with diltiazem"
    ),
    InteractionRecord(
        drug1_id="DB00641", drug2_id="DB00176",
        drug1_name="Simvastatin", drug2_name="Fluvoxamine",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Fluvoxamine inhibits CYP3A4, reducing simvastatin metabolism",
        clinical_effect="Significantly elevated statin levels, rhabdomyolysis risk",
        management="Avoid combination or use pravastatin"
    ),
    
    # === Digoxin Interactions ===
    InteractionRecord(
        drug1_id="DB00390", drug2_id="DB01118",
        drug1_name="Digoxin", drug2_name="Amiodarone",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Amiodarone inhibits P-glycoprotein, reducing digoxin renal and nonrenal clearance",
        clinical_effect="70-100% increase in digoxin levels, toxicity risk (arrhythmias, nausea, visual disturbances)",
        management="Reduce digoxin dose by 50% when starting amiodarone. Monitor digoxin levels"
    ),
    InteractionRecord(
        drug1_id="DB00390", drug2_id="DB00661",
        drug1_name="Digoxin", drug2_name="Verapamil",
        severity="major",
        interaction_type="mechanism",
        mechanism="Verapamil inhibits P-glycoprotein and reduces renal/nonrenal digoxin clearance",
        clinical_effect="50-70% increase in digoxin levels. Additive AV node depression",
        management="Reduce digoxin dose by 33-50%. Monitor heart rate and digoxin levels"
    ),
    InteractionRecord(
        drug1_id="DB00390", drug2_id="DB00343",
        drug1_name="Digoxin", drug2_name="Diltiazem",
        severity="major",
        interaction_type="mechanism",
        mechanism="Diltiazem reduces digoxin clearance and has additive AV nodal effects",
        clinical_effect="22-35% increase in digoxin levels. Risk of bradycardia/AV block",
        management="Monitor heart rate and digoxin levels. Consider dose reduction"
    ),
    InteractionRecord(
        drug1_id="DB00390", drug2_id="DB01369",
        drug1_name="Digoxin", drug2_name="Quinidine",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Quinidine inhibits P-glycoprotein and renal tubular secretion of digoxin",
        clinical_effect="Doubles digoxin levels, high toxicity risk",
        management="Reduce digoxin dose by 50% when adding quinidine"
    ),
    
    # === SSRI Interactions ===
    InteractionRecord(
        drug1_id="DB00472", drug2_id="DB00318",
        drug1_name="Fluoxetine", drug2_name="Codeine",
        severity="major",
        interaction_type="mechanism",
        mechanism="Fluoxetine inhibits CYP2D6, blocking conversion of codeine to morphine",
        clinical_effect="Reduced analgesic effect of codeine",
        management="Consider alternative analgesic (morphine, hydromorphone) or different antidepressant"
    ),
    InteractionRecord(
        drug1_id="DB00472", drug2_id="DB00327",
        drug1_name="Fluoxetine", drug2_name="Tramadol",
        severity="major",
        interaction_type="mechanism",
        mechanism="Fluoxetine inhibits CYP2D6 (reduces tramadol activation) and increases serotonin (additive effects)",
        clinical_effect="Reduced analgesia AND increased serotonin syndrome risk",
        management="Avoid combination. Choose alternative analgesic or antidepressant"
    ),
    InteractionRecord(
        drug1_id="DB00715", drug2_id="DB00264",
        drug1_name="Paroxetine", drug2_name="Metoprolol",
        severity="major",
        interaction_type="mechanism",
        mechanism="Paroxetine is a potent CYP2D6 inhibitor, reducing metoprolol metabolism",
        clinical_effect="2-5x increase in metoprolol levels, risk of severe bradycardia and hypotension",
        management="Monitor heart rate/BP. Consider reducing metoprolol dose or using atenolol (not CYP2D6 substrate)"
    ),
    
    # === Proton Pump Inhibitor Interactions ===
    InteractionRecord(
        drug1_id="DB00338", drug2_id="DB00563",
        drug1_name="Omeprazole", drug2_name="Methotrexate",
        severity="major",
        interaction_type="mechanism",
        mechanism="PPIs inhibit renal elimination of methotrexate",
        clinical_effect="Delayed methotrexate clearance, increased toxicity risk",
        management="Consider holding PPI during high-dose methotrexate. Monitor methotrexate levels"
    ),
    
    # === Benzodiazepine Interactions ===
    InteractionRecord(
        drug1_id="DB00683", drug2_id="DB01026",
        drug1_name="Midazolam", drug2_name="Ketoconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Ketoconazole inhibits CYP3A4, the sole metabolic pathway for midazolam",
        clinical_effect="10-15x increase in midazolam AUC, profound and prolonged sedation, respiratory depression",
        management="AVOID combination. If unavoidable, reduce midazolam dose by 75%+"
    ),
    InteractionRecord(
        drug1_id="DB00683", drug2_id="DB01167",
        drug1_name="Midazolam", drug2_name="Itraconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Itraconazole strongly inhibits CYP3A4",
        clinical_effect="Markedly prolonged sedation, respiratory depression risk",
        management="AVOID oral midazolam. Reduce IV dose significantly if necessary"
    ),
    InteractionRecord(
        drug1_id="DB00683", drug2_id="DB01045",
        drug1_name="Midazolam", drug2_name="Rifampin",
        severity="major",
        interaction_type="mechanism",
        mechanism="Rifampin induces CYP3A4, dramatically increasing midazolam metabolism",
        clinical_effect="90%+ reduction in midazolam levels, potential loss of sedation/anxiolysis",
        management="May need markedly increased midazolam dose or alternative benzodiazepine (lorazepam)"
    ),
    
    # === Opioid Interactions ===
    InteractionRecord(
        drug1_id="DB00813", drug2_id="DB01026",
        drug1_name="Fentanyl", drug2_name="Ketoconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Ketoconazole inhibits CYP3A4, the primary enzyme metabolizing fentanyl",
        clinical_effect="Markedly increased fentanyl levels, risk of fatal respiratory depression",
        management="AVOID combination. If necessary, reduce fentanyl dose significantly and monitor closely"
    ),
    InteractionRecord(
        drug1_id="DB00813", drug2_id="DB01045",
        drug1_name="Fentanyl", drug2_name="Rifampin",
        severity="major",
        interaction_type="mechanism",
        mechanism="Rifampin induces CYP3A4, increasing fentanyl metabolism",
        clinical_effect="Reduced fentanyl levels, potential opioid withdrawal or inadequate analgesia",
        management="May need fentanyl dose increase. Monitor for withdrawal symptoms"
    ),
    
    # === Immunosuppressant Interactions ===
    InteractionRecord(
        drug1_id="DB00864", drug2_id="DB01026",
        drug1_name="Tacrolimus", drug2_name="Ketoconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Ketoconazole inhibits CYP3A4 and P-glycoprotein",
        clinical_effect="3-5x increase in tacrolimus levels, nephrotoxicity and neurotoxicity risk",
        management="Reduce tacrolimus dose by 50-75%. Monitor levels within 3-5 days"
    ),
    InteractionRecord(
        drug1_id="DB00864", drug2_id="DB01045",
        drug1_name="Tacrolimus", drug2_name="Rifampin",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Rifampin induces CYP3A4 and P-glycoprotein",
        clinical_effect="80-90% reduction in tacrolimus levels, risk of organ rejection",
        management="Avoid if possible. May need 3-5x tacrolimus dose increase. Monitor levels closely"
    ),
    InteractionRecord(
        drug1_id="DB00864", drug2_id="DB00176",
        drug1_name="Tacrolimus", drug2_name="Fluvoxamine",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Fluvoxamine inhibits CYP3A4, reducing tacrolimus metabolism",
        clinical_effect="Significantly elevated tacrolimus levels, nephrotoxicity risk",
        management="Avoid combination. If necessary, reduce tacrolimus dose and monitor levels"
    ),
    InteractionRecord(
        drug1_id="DB00091", drug2_id="DB01026",
        drug1_name="Cyclosporine", drug2_name="Ketoconazole",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Ketoconazole inhibits CYP3A4 and P-glycoprotein",
        clinical_effect="2-3x increase in cyclosporine levels, nephrotoxicity risk",
        management="Reduce cyclosporine dose by 50%. Monitor levels"
    ),
    InteractionRecord(
        drug1_id="DB00091", drug2_id="DB01045",
        drug1_name="Cyclosporine", drug2_name="Rifampin",
        severity="severe",
        interaction_type="mechanism",
        mechanism="Rifampin induces CYP3A4 and P-glycoprotein",
        clinical_effect="50-70% reduction in cyclosporine levels, transplant rejection risk",
        management="AVOID. If necessary, may need 2-3x cyclosporine dose. Monitor levels closely"
    ),
    
    # === Additional high-risk interactions ===
    InteractionRecord(
        drug1_id="DB00668", drug2_id="DB01026",
        drug1_name="Sildenafil", drug2_name="Ketoconazole",
        severity="major",
        interaction_type="mechanism",
        mechanism="Ketoconazole inhibits CYP3A4, reducing sildenafil metabolism",
        clinical_effect="Markedly increased sildenafil levels, risk of severe hypotension, priapism",
        management="Start with lowest sildenafil dose (25mg) and avoid higher doses"
    ),
    InteractionRecord(
        drug1_id="DB00571", drug2_id="DB00661",
        drug1_name="Propranolol", drug2_name="Verapamil",
        severity="severe",
        interaction_type="effect",
        mechanism="Both drugs slow cardiac conduction and reduce contractility",
        clinical_effect="Severe bradycardia, heart block, hypotension, heart failure",
        management="AVOID IV combination. Oral combination with extreme caution only if necessary"
    ),
    InteractionRecord(
        drug1_id="DB00264", drug2_id="DB00661",
        drug1_name="Metoprolol", drug2_name="Verapamil",
        severity="severe",
        interaction_type="effect",
        mechanism="Additive negative chronotropic and inotropic effects",
        clinical_effect="Severe bradycardia, AV block, hypotension",
        management="Use with extreme caution. Start with lowest doses and monitor closely"
    ),
]


def get_drug_by_id(drug_id: str) -> Optional[DrugInfo]:
    """Get drug information by DrugBank ID."""
    for drug in DRUG_DATABASE:
        if drug.drugbank_id == drug_id:
            return drug
    return None


def get_drug_by_name(name: str) -> Optional[DrugInfo]:
    """Get drug information by name (case-insensitive)."""
    name_lower = name.lower()
    for drug in DRUG_DATABASE:
        if drug.name.lower() == name_lower:
            return drug
    return None


def get_interaction(drug1_id: str, drug2_id: str) -> Optional[InteractionRecord]:
    """Get interaction record between two drugs."""
    for interaction in INTERACTION_DATABASE:
        if (interaction.drug1_id == drug1_id and interaction.drug2_id == drug2_id) or \
           (interaction.drug1_id == drug2_id and interaction.drug2_id == drug1_id):
            return interaction
    return None


def get_all_drugs() -> List[DrugInfo]:
    """Get all drugs in the database."""
    return DRUG_DATABASE.copy()


def get_all_interactions() -> List[InteractionRecord]:
    """Get all interactions in the database."""
    return INTERACTION_DATABASE.copy()


def export_training_data(output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Export training data as JSON files for model training.
    
    Returns:
        Tuple of (drugs_file_path, interactions_file_path)
    """
    if output_dir is None:
        output_dir = DATA_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export drugs
    drugs_data = []
    for drug in DRUG_DATABASE:
        drugs_data.append({
            'drugbank_id': drug.drugbank_id,
            'name': drug.name,
            'smiles': drug.smiles,
            'category': drug.category,
            'enzymes': drug.enzymes,
            'description': drug.description
        })
    
    drugs_path = output_dir / 'training_drugs.json'
    with open(drugs_path, 'w', encoding='utf-8') as f:
        json.dump(drugs_data, f, indent=2, ensure_ascii=False)
    
    # Export interactions
    interactions_data = []
    for inter in INTERACTION_DATABASE:
        interactions_data.append({
            'drug1_id': inter.drug1_id,
            'drug2_id': inter.drug2_id,
            'drug1_name': inter.drug1_name,
            'drug2_name': inter.drug2_name,
            'severity': inter.severity,
            'interaction_type': inter.interaction_type,
            'mechanism': inter.mechanism,
            'clinical_effect': inter.clinical_effect,
            'management': inter.management
        })
    
    interactions_path = output_dir / 'training_interactions.json'
    with open(interactions_path, 'w', encoding='utf-8') as f:
        json.dump(interactions_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(drugs_data)} drugs to {drugs_path}")
    logger.info(f"Exported {len(interactions_data)} interactions to {interactions_path}")
    
    return drugs_path, interactions_path


def get_training_statistics() -> Dict:
    """Get statistics about the training data."""
    severity_counts = {}
    type_counts = {}
    
    for inter in INTERACTION_DATABASE:
        severity_counts[inter.severity] = severity_counts.get(inter.severity, 0) + 1
        type_counts[inter.interaction_type] = type_counts.get(inter.interaction_type, 0) + 1
    
    return {
        'total_drugs': len(DRUG_DATABASE),
        'total_interactions': len(INTERACTION_DATABASE),
        'drugs_with_smiles': sum(1 for d in DRUG_DATABASE if d.smiles),
        'severity_distribution': severity_counts,
        'type_distribution': type_counts,
        'categories': list(set(d.category for d in DRUG_DATABASE))
    }


if __name__ == '__main__':
    # Export training data when run directly
    logging.basicConfig(level=logging.INFO)
    
    drugs_path, interactions_path = export_training_data()
    stats = get_training_statistics()
    
    print("\n=== Training Data Statistics ===")
    print(f"Total drugs: {stats['total_drugs']}")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Drugs with SMILES: {stats['drugs_with_smiles']}")
    print(f"\nSeverity distribution: {stats['severity_distribution']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print(f"Drug categories: {stats['categories']}")
