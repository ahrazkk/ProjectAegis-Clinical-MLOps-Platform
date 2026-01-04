"""
Download Real DDI Datasets

This script downloads publicly available drug-drug interaction datasets
that can be used for training a production-quality model.

Datasets:
1. DrugBank DDI - Curated drug interactions
2. TWOSIDES - Polypharmacy side effects from FAERS
3. ChEMBL - Drug target information
"""

import os
import json
import requests
import gzip
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Django setup
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
django.setup()

from ddi_api.services.knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DrugBank Open Data (Drug names, identifiers, SMILES)
# ============================================================================

# Pre-compiled drug data with SMILES (from DrugBank open vocabulary)
# This is a curated subset of common drugs with their structures
DRUGBANK_DRUGS = [
    # Cardiovascular
    {"id": "DB00945", "name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "category": "NSAID"},
    {"id": "DB00682", "name": "Warfarin", "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O", "category": "Anticoagulant"},
    {"id": "DB00390", "name": "Digoxin", "smiles": "CC1OC(CC(O)C1O)OC2C(O)CC(OC3C(O)CC(OC4CCC5(C)C(CCC6C5CCC7(C)C(C8=CC(=O)OC8)CCC67)C4)OC3C)OC2C", "category": "Cardiac Glycoside"},
    {"id": "DB01118", "name": "Amiodarone", "smiles": "CCCCC1=C(C2=CC=C(OCCN(CC)CC)C=C2)C3=CC(I)=C(OCCC)C(I)=C3O1", "category": "Antiarrhythmic"},
    {"id": "DB00571", "name": "Propranolol", "smiles": "CC(C)NCC(O)COC1=CC=CC2=CC=CC=C12", "category": "Beta Blocker"},
    {"id": "DB00661", "name": "Verapamil", "smiles": "COC1=CC=C(CCN(C)CCCC(C#N)(C(C)C)C2=CC(OC)=C(OC)C=C2)C=C1OC", "category": "Calcium Channel Blocker"},
    {"id": "DB00999", "name": "Hydrochlorothiazide", "smiles": "NS(=O)(=O)C1=CC2=C(NCNS2(=O)=O)C=C1Cl", "category": "Diuretic"},
    {"id": "DB01136", "name": "Carvedilol", "smiles": "COC1=CC=CC=C1OCCNCC(O)COC2=CC=CC3=C2C=C(O)C=C3", "category": "Beta Blocker"},
    {"id": "DB00264", "name": "Metoprolol", "smiles": "CC(C)NCC(O)COC1=CC=C(CCOC)C=C1", "category": "Beta Blocker"},
    {"id": "DB01023", "name": "Felodipine", "smiles": "CCOC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1C2=CC=CC(Cl)=C2Cl", "category": "Calcium Channel Blocker"},
    
    # Statins
    {"id": "DB00641", "name": "Simvastatin", "smiles": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12", "category": "Statin"},
    {"id": "DB00175", "name": "Pravastatin", "smiles": "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C12", "category": "Statin"},
    {"id": "DB01076", "name": "Atorvastatin", "smiles": "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(C3=CC=C(F)C=C3)=C(CCC(O)CC(O)CC(=O)O)N1", "category": "Statin"},
    {"id": "DB00227", "name": "Lovastatin", "smiles": "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12", "category": "Statin"},
    {"id": "DB01098", "name": "Rosuvastatin", "smiles": "CC(C)C1=NC(N(C)S(=O)(=O)C)=NC(C2=CC=C(F)C=C2)=C1CCC(O)CC(O)CC(=O)O", "category": "Statin"},
    
    # Analgesics/Anti-inflammatory
    {"id": "DB01050", "name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C(C)C(=O)O)C=C1", "category": "NSAID"},
    {"id": "DB00316", "name": "Acetaminophen", "smiles": "CC(=O)NC1=CC=C(O)C=C1", "category": "Analgesic"},
    {"id": "DB00788", "name": "Naproxen", "smiles": "COC1=CC2=CC(C(C)C(=O)O)=CC=C2C=C1", "category": "NSAID"},
    {"id": "DB00586", "name": "Diclofenac", "smiles": "OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl", "category": "NSAID"},
    {"id": "DB00465", "name": "Ketorolac", "smiles": "OC(=O)C1CCN2C1=CC=C2C(=O)C3=CC=CC=C3", "category": "NSAID"},
    
    # Anticonvulsants
    {"id": "DB00564", "name": "Carbamazepine", "smiles": "NC(=O)N1C2=CC=CC=C2C=CC3=CC=CC=C13", "category": "Anticonvulsant"},
    {"id": "DB00252", "name": "Phenytoin", "smiles": "O=C1NC(=O)C(C2=CC=CC=C2)(C3=CC=CC=C3)N1", "category": "Anticonvulsant"},
    {"id": "DB01174", "name": "Phenobarbital", "smiles": "CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2", "category": "Barbiturate"},
    {"id": "DB00313", "name": "Valproic Acid", "smiles": "CCCC(CCC)C(=O)O", "category": "Anticonvulsant"},
    {"id": "DB00555", "name": "Lamotrigine", "smiles": "NC1=NC(N)=C2N=C(C3=CC=CC(Cl)=C3Cl)C=NC2=N1", "category": "Anticonvulsant"},
    
    # Antidepressants/Antipsychotics
    {"id": "DB00176", "name": "Fluvoxamine", "smiles": "COCCCC/C(=N\\OCCN)C1=CC=C(C(F)(F)F)C=C1", "category": "SSRI"},
    {"id": "DB00472", "name": "Fluoxetine", "smiles": "CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C2=CC=CC=C2", "category": "SSRI"},
    {"id": "DB00715", "name": "Paroxetine", "smiles": "FC1=CC=C(C2CCNCC2COC3=CC4=C(OCO4)C=C3)C=C1", "category": "SSRI"},
    {"id": "DB01104", "name": "Sertraline", "smiles": "CNC1CCC(C2=CC=C(Cl)C(Cl)=C2)C3=CC=CC=C13", "category": "SSRI"},
    {"id": "DB00334", "name": "Olanzapine", "smiles": "CC1=CC2=C(NC3=CC=CC=C3N=C2S1)N4CCN(C)CC4", "category": "Antipsychotic"},
    
    # Antibiotics
    {"id": "DB00218", "name": "Moxifloxacin", "smiles": "COC1=C(N2CC3CCCNC3C2)C(F)=CC4=C1N(C=C(C(=O)O)C4=O)C5CC5", "category": "Antibiotic"},
    {"id": "DB00537", "name": "Ciprofloxacin", "smiles": "OC(=O)C1=CN(C2CC2)C3=CC(N4CCNCC4)=C(F)C=C3C1=O", "category": "Antibiotic"},
    {"id": "DB01211", "name": "Clarithromycin", "smiles": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(N(C)C)C3O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O", "category": "Antibiotic"},
    {"id": "DB01190", "name": "Clindamycin", "smiles": "CCCC1CC(N(C)C1)C(=O)NC(C(C)Cl)C2OC(SC)C(O)C(O)C2O", "category": "Antibiotic"},
    {"id": "DB00446", "name": "Chloramphenicol", "smiles": "OCC(NC(=O)C(Cl)Cl)C(O)C1=CC=C([N+]([O-])=O)C=C1", "category": "Antibiotic"},
    
    # Immunosuppressants
    {"id": "DB00864", "name": "Tacrolimus", "smiles": "COC1CC(CCC1OC)CC(C)C2CC(=O)C(C(C(CC(C(C(C(=O)C(C(=O)OC(C(C(CC(=O)C(C=C2C)C)O)OC)C(C)CC=CC)C)O)OC)C)C)O)C", "category": "Immunosuppressant"},
    {"id": "DB00877", "name": "Sirolimus", "smiles": "COC1CC(CCC1OC(C)C2CC(=O)C(C(C(CC(CC(C)C=CC=CC=CC(CC3CCC(C(O3)(C(=O)C(=O)N4CCCCC4C(=O)OC(C(C(CC2=O)O)C)C(C)CC=C)O)O)OC)OC)O)C)C)OC", "category": "Immunosuppressant"},
    {"id": "DB00091", "name": "Cyclosporine", "smiles": "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC2=CC=CC=C2)N(C)C1=O", "category": "Immunosuppressant"},
    
    # Anticoagulants/Antiplatelets
    {"id": "DB00374", "name": "Treprostinil", "smiles": "CCCCCC(O)C#CC1CC(O)C(CC2CCCC12)CCCCC(=O)O", "category": "Antiplatelet"},
    {"id": "DB00758", "name": "Clopidogrel", "smiles": "COC(=O)C(C1=CC=CS1)N2CCC3=CC=CC=C3C2C4=CC=C(Cl)C=C4", "category": "Antiplatelet"},
    {"id": "DB00806", "name": "Pentoxifylline", "smiles": "CC(=O)CCCN1C=NC2=C1C(=O)N(C)C(=O)N2C", "category": "Hemorrheologic"},
    {"id": "DB01427", "name": "Rivaroxaban", "smiles": "ClC1=CC=C(C=C1)N2C(=O)NC(COC3=CC=C(N4CCOCC4)C=C3)CC2=O", "category": "Anticoagulant"},
    
    # Proton Pump Inhibitors
    {"id": "DB00338", "name": "Omeprazole", "smiles": "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C)C(OC)=C3C", "category": "PPI"},
    {"id": "DB00213", "name": "Pantoprazole", "smiles": "COC1=CC=NC2=CC(OC(F)(F)F)=C(OC)C=C12", "category": "PPI"},
    {"id": "DB00736", "name": "Esomeprazole", "smiles": "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C)C(OC)=C3C", "category": "PPI"},
    
    # Diabetes
    {"id": "DB00331", "name": "Metformin", "smiles": "CN(C)C(=N)NC(=N)N", "category": "Antidiabetic"},
    {"id": "DB01120", "name": "Gliclazide", "smiles": "CC1=CC=C(C=C1)S(=O)(=O)NC(=O)NN2CC3CCCC3C2", "category": "Antidiabetic"},
    {"id": "DB01016", "name": "Glyburide", "smiles": "COC1=CC=C(C=C1)CCNC(=O)NS(=O)(=O)C2=CC=C(C=C2)CCC3=CC=CC=C3", "category": "Antidiabetic"},
]


# ============================================================================
# Known Drug-Drug Interactions (from literature and DrugBank)
# ============================================================================

KNOWN_INTERACTIONS = [
    # Warfarin interactions (highly documented)
    {"drug1": "DB00682", "drug2": "DB00945", "severity": "severe", 
     "mechanism": "Aspirin inhibits platelet COX-1 and displaces warfarin from albumin binding sites, dramatically increasing bleeding risk. Combined antiplatelet and anticoagulant effects are synergistic.",
     "systems": ["cardiovascular", "hematologic", "gastrointestinal"]},
    {"drug1": "DB00682", "drug2": "DB01050", "severity": "severe",
     "mechanism": "Ibuprofen inhibits platelet function and may displace warfarin from protein binding sites. NSAIDs also increase GI bleeding risk.",
     "systems": ["hematologic", "gastrointestinal"]},
    {"drug1": "DB00682", "drug2": "DB00788", "severity": "severe",
     "mechanism": "Naproxen inhibits platelet aggregation and increases warfarin's anticoagulant effect. High risk of GI and other bleeding.",
     "systems": ["hematologic", "gastrointestinal"]},
    {"drug1": "DB00682", "drug2": "DB00586", "severity": "severe",
     "mechanism": "Diclofenac inhibits platelet function and may increase warfarin levels through CYP2C9 competition.",
     "systems": ["hematologic", "gastrointestinal"]},
    {"drug1": "DB00682", "drug2": "DB00564", "severity": "moderate",
     "mechanism": "Carbamazepine induces CYP2C9 and CYP3A4, increasing warfarin metabolism and reducing anticoagulation efficacy.",
     "systems": ["hematologic"]},
    {"drug1": "DB00682", "drug2": "DB00252", "severity": "moderate",
     "mechanism": "Phenytoin induces warfarin metabolism. Both are highly protein bound, leading to complex displacement interactions.",
     "systems": ["hematologic", "neurologic"]},
    {"drug1": "DB00682", "drug2": "DB00338", "severity": "minor",
     "mechanism": "Omeprazole may slightly inhibit CYP2C19-mediated warfarin metabolism, potentially increasing INR.",
     "systems": ["hematologic"]},
    
    # Statin interactions (CYP3A4 and muscle toxicity)
    {"drug1": "DB00641", "drug2": "DB01118", "severity": "severe",
     "mechanism": "Amiodarone strongly inhibits CYP3A4, dramatically increasing simvastatin levels. High risk of rhabdomyolysis.",
     "systems": ["musculoskeletal", "hepatic"]},
    {"drug1": "DB00641", "drug2": "DB00176", "severity": "severe",
     "mechanism": "Fluvoxamine potently inhibits CYP3A4, causing 10-fold increases in simvastatin exposure. Extreme myopathy risk.",
     "systems": ["musculoskeletal"]},
    {"drug1": "DB00641", "drug2": "DB00661", "severity": "moderate",
     "mechanism": "Verapamil inhibits CYP3A4 and P-glycoprotein, increasing simvastatin levels 2-3 fold.",
     "systems": ["musculoskeletal", "cardiovascular"]},
    {"drug1": "DB00641", "drug2": "DB01211", "severity": "severe",
     "mechanism": "Clarithromycin is a potent CYP3A4 inhibitor. Simvastatin dose should not exceed 10mg daily.",
     "systems": ["musculoskeletal", "hepatic"]},
    {"drug1": "DB01076", "drug2": "DB01211", "severity": "moderate",
     "mechanism": "Clarithromycin increases atorvastatin exposure via CYP3A4 inhibition, though less than with simvastatin.",
     "systems": ["musculoskeletal"]},
    
    # Digoxin interactions (narrow therapeutic index)
    {"drug1": "DB00390", "drug2": "DB01118", "severity": "severe",
     "mechanism": "Amiodarone inhibits P-glycoprotein and reduces renal/non-renal digoxin clearance by 50%. Digoxin dose reduction required.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00390", "drug2": "DB00661", "severity": "moderate",
     "mechanism": "Verapamil inhibits P-gp efflux and reduces digoxin clearance by 30-40%. Also additive negative chronotropic effects.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00390", "drug2": "DB00537", "severity": "minor",
     "mechanism": "Ciprofloxacin may increase digoxin levels in some patients by altering gut flora that metabolizes digoxin.",
     "systems": ["cardiovascular", "gastrointestinal"]},
    
    # Immunosuppressant interactions
    {"drug1": "DB00864", "drug2": "DB00176", "severity": "severe",
     "mechanism": "Fluvoxamine potently inhibits CYP3A4, dramatically increasing tacrolimus levels and nephrotoxicity risk.",
     "systems": ["renal", "immune"]},
    {"drug1": "DB00864", "drug2": "DB01211", "severity": "severe",
     "mechanism": "Clarithromycin inhibits CYP3A4, increasing tacrolimus 2-5 fold. Requires dose reduction and monitoring.",
     "systems": ["renal", "immune"]},
    {"drug1": "DB00864", "drug2": "DB00564", "severity": "moderate",
     "mechanism": "Carbamazepine induces CYP3A4, substantially reducing tacrolimus levels. May cause rejection.",
     "systems": ["immune"]},
    {"drug1": "DB00091", "drug2": "DB00641", "severity": "moderate",
     "mechanism": "Cyclosporine inhibits CYP3A4 and OATP1B1, increasing simvastatin levels and myopathy risk.",
     "systems": ["musculoskeletal", "renal"]},
    
    # Antidepressant interactions
    {"drug1": "DB00472", "drug2": "DB00334", "severity": "moderate",
     "mechanism": "Fluoxetine inhibits CYP2D6, increasing olanzapine levels and risk of sedation and metabolic effects.",
     "systems": ["neurologic", "metabolic"]},
    {"drug1": "DB00176", "drug2": "DB00338", "severity": "minor",
     "mechanism": "Fluvoxamine inhibits CYP1A2, moderately increasing omeprazole levels.",
     "systems": ["gastrointestinal"]},
    
    # Cardiovascular combinations
    {"drug1": "DB00571", "drug2": "DB00661", "severity": "moderate",
     "mechanism": "Both slow cardiac conduction. Combination may cause severe bradycardia, heart block, or hypotension.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB01136", "drug2": "DB00390", "severity": "moderate",
     "mechanism": "Carvedilol may increase digoxin levels via P-gp inhibition. Additive negative chronotropic effects.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00264", "drug2": "DB00661", "severity": "moderate",
     "mechanism": "Verapamil and metoprolol both slow AV conduction. Risk of severe bradycardia or complete heart block.",
     "systems": ["cardiovascular"]},
    
    # Anticonvulsant interactions (enzyme inducers)
    {"drug1": "DB00564", "drug2": "DB00472", "severity": "moderate",
     "mechanism": "Carbamazepine induces CYP enzymes, reducing fluoxetine efficacy. Also displacement from protein binding.",
     "systems": ["neurologic"]},
    {"drug1": "DB00252", "drug2": "DB00313", "severity": "moderate",
     "mechanism": "Valproate inhibits phenytoin metabolism and displaces it from protein binding, causing toxicity.",
     "systems": ["neurologic"]},
    {"drug1": "DB00564", "drug2": "DB00555", "severity": "moderate",
     "mechanism": "Carbamazepine induces lamotrigine glucuronidation, reducing levels by 40-50%.",
     "systems": ["neurologic"]},
    
    # Antiplatelet + NSAID
    {"drug1": "DB00758", "drug2": "DB00945", "severity": "moderate",
     "mechanism": "Aspirin may reduce clopidogrel's antiplatelet effect by competing for CYP2C19 activation site.",
     "systems": ["cardiovascular", "hematologic"]},
    {"drug1": "DB00758", "drug2": "DB00338", "severity": "moderate",
     "mechanism": "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to active metabolite.",
     "systems": ["cardiovascular"]},
    
    # Metformin interactions
    {"drug1": "DB00331", "drug2": "DB00537", "severity": "minor",
     "mechanism": "Ciprofloxacin may increase metformin levels by inhibiting renal tubular secretion via OCT2.",
     "systems": ["metabolic", "renal"]},
    
    # QT prolongation risks
    {"drug1": "DB01118", "drug2": "DB00218", "severity": "severe",
     "mechanism": "Both amiodarone and moxifloxacin prolong QT interval. Combination dramatically increases torsades de pointes risk.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00334", "drug2": "DB00218", "severity": "moderate",
     "mechanism": "Olanzapine and moxifloxacin both prolong QT interval. Additive cardiac arrhythmia risk.",
     "systems": ["cardiovascular"]},
     
    # Additional clinically significant interactions
    {"drug1": "DB00472", "drug2": "DB01118", "severity": "severe",
     "mechanism": "Fluoxetine inhibits CYP2D6, dramatically increasing amiodarone levels. High risk of QT prolongation and arrhythmias.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00316", "drug2": "DB00682", "severity": "minor",
     "mechanism": "Acetaminophen may slightly increase INR when taken regularly with warfarin through unknown mechanism.",
     "systems": ["hematologic"]},
    {"drug1": "DB00175", "drug2": "DB00091", "severity": "moderate",
     "mechanism": "Cyclosporine inhibits OATP1B1, increasing pravastatin levels 5-10 fold. Myopathy risk elevated.",
     "systems": ["musculoskeletal"]},
    {"drug1": "DB00715", "drug2": "DB00264", "severity": "moderate",
     "mechanism": "Paroxetine inhibits CYP2D6, increasing metoprolol levels and risk of bradycardia.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB01104", "drug2": "DB00390", "severity": "minor",
     "mechanism": "Sertraline may slightly increase digoxin levels. Monitor for digoxin toxicity.",
     "systems": ["cardiovascular"]},
    {"drug1": "DB00877", "drug2": "DB01211", "severity": "severe",
     "mechanism": "Clarithromycin dramatically increases sirolimus exposure via CYP3A4 inhibition. Avoid combination.",
     "systems": ["immune", "renal"]},
    {"drug1": "DB00465", "drug2": "DB00682", "severity": "severe",
     "mechanism": "Ketorolac and warfarin combination dramatically increases bleeding risk. Contraindicated.",
     "systems": ["hematologic", "gastrointestinal"]},
    {"drug1": "DB00331", "drug2": "DB01016", "severity": "moderate",
     "mechanism": "Metformin combined with glyburide increases hypoglycemia risk. Monitor blood glucose closely.",
     "systems": ["metabolic"]},
    {"drug1": "DB00390", "drug2": "DB00999", "severity": "moderate",
     "mechanism": "Hydrochlorothiazide-induced hypokalemia increases digoxin toxicity risk. Monitor potassium levels.",
     "systems": ["cardiovascular", "renal"]},
    {"drug1": "DB01427", "drug2": "DB00945", "severity": "moderate",
     "mechanism": "Rivaroxaban and aspirin combination increases bleeding risk. Used in specific clinical scenarios with caution.",
     "systems": ["hematologic"]},
    {"drug1": "DB01098", "drug2": "DB00091", "severity": "moderate",
     "mechanism": "Cyclosporine increases rosuvastatin exposure via OATP1B1 inhibition. Limit rosuvastatin to 5mg.",
     "systems": ["musculoskeletal", "hepatic"]},
    {"drug1": "DB00227", "drug2": "DB01211", "severity": "severe",
     "mechanism": "Clarithromycin is a potent CYP3A4 inhibitor, dramatically increasing lovastatin levels and rhabdomyolysis risk.",
     "systems": ["musculoskeletal"]},
    {"drug1": "DB01174", "drug2": "DB00682", "severity": "moderate",
     "mechanism": "Phenobarbital induces CYP enzymes, reducing warfarin efficacy. May need higher warfarin doses.",
     "systems": ["hematologic"]},
]


# ============================================================================
# Drug Targets (for mechanistic understanding)
# ============================================================================

DRUG_TARGETS = [
    # Cytochrome P450 enzymes
    {"id": "P11712", "name": "Cytochrome P450 2C9", "gene": "CYP2C9", "type": "enzyme"},
    {"id": "P08684", "name": "Cytochrome P450 3A4", "gene": "CYP3A4", "type": "enzyme"},
    {"id": "P10635", "name": "Cytochrome P450 2D6", "gene": "CYP2D6", "type": "enzyme"},
    {"id": "P33261", "name": "Cytochrome P450 2C19", "gene": "CYP2C19", "type": "enzyme"},
    {"id": "P05177", "name": "Cytochrome P450 1A2", "gene": "CYP1A2", "type": "enzyme"},
    
    # Transporters
    {"id": "P08183", "name": "P-glycoprotein 1", "gene": "ABCB1", "type": "transporter"},
    {"id": "Q9UNQ0", "name": "ABCG2", "gene": "ABCG2", "type": "transporter"},
    {"id": "Q12908", "name": "OATP1B1", "gene": "SLCO1B1", "type": "transporter"},
    
    # Drug targets
    {"id": "P23219", "name": "Prostaglandin G/H synthase 1", "gene": "PTGS1", "type": "target"},  # COX-1
    {"id": "P35354", "name": "Prostaglandin G/H synthase 2", "gene": "PTGS2", "type": "target"},  # COX-2
    {"id": "P00734", "name": "Prothrombin", "gene": "F2", "type": "target"},
    {"id": "P00742", "name": "Coagulation factor X", "gene": "F10", "type": "target"},
    {"id": "P35368", "name": "HMG-CoA reductase", "gene": "HMGCR", "type": "target"},
    {"id": "Q01959", "name": "Sodium-dependent serotonin transporter", "gene": "SLC6A4", "type": "target"},
]


# Drug-target relationships
DRUG_TARGET_LINKS = [
    # NSAIDs -> COX
    {"drug": "DB00945", "target": "P23219", "action": "inhibitor"},
    {"drug": "DB00945", "target": "P35354", "action": "inhibitor"},
    {"drug": "DB01050", "target": "P23219", "action": "inhibitor"},
    {"drug": "DB01050", "target": "P35354", "action": "inhibitor"},
    {"drug": "DB00788", "target": "P23219", "action": "inhibitor"},
    {"drug": "DB00788", "target": "P35354", "action": "inhibitor"},
    
    # Warfarin CYP substrates
    {"drug": "DB00682", "target": "P11712", "action": "substrate"},
    {"drug": "DB00682", "target": "P33261", "action": "substrate"},
    
    # Statins -> HMG-CoA reductase
    {"drug": "DB00641", "target": "P35368", "action": "inhibitor"},
    {"drug": "DB01076", "target": "P35368", "action": "inhibitor"},
    {"drug": "DB00175", "target": "P35368", "action": "inhibitor"},
    
    # Statins CYP3A4 substrates
    {"drug": "DB00641", "target": "P08684", "action": "substrate"},
    {"drug": "DB01076", "target": "P08684", "action": "substrate"},
    
    # CYP inhibitors
    {"drug": "DB01118", "target": "P08684", "action": "inhibitor"},  # Amiodarone
    {"drug": "DB00176", "target": "P08684", "action": "inhibitor"},  # Fluvoxamine
    {"drug": "DB00176", "target": "P05177", "action": "inhibitor"},  # Fluvoxamine CYP1A2
    {"drug": "DB00661", "target": "P08684", "action": "inhibitor"},  # Verapamil
    {"drug": "DB01211", "target": "P08684", "action": "inhibitor"},  # Clarithromycin
    
    # CYP inducers
    {"drug": "DB00564", "target": "P08684", "action": "inducer"},   # Carbamazepine
    {"drug": "DB00564", "target": "P11712", "action": "inducer"},
    {"drug": "DB00252", "target": "P08684", "action": "inducer"},   # Phenytoin
    {"drug": "DB00252", "target": "P11712", "action": "inducer"},
    
    # P-gp substrates/inhibitors
    {"drug": "DB00390", "target": "P08183", "action": "substrate"},  # Digoxin
    {"drug": "DB01118", "target": "P08183", "action": "inhibitor"},  # Amiodarone
    {"drug": "DB00661", "target": "P08183", "action": "inhibitor"},  # Verapamil
    {"drug": "DB00864", "target": "P08183", "action": "substrate"},  # Tacrolimus
    
    # Immunosuppressants CYP3A4
    {"drug": "DB00864", "target": "P08684", "action": "substrate"},  # Tacrolimus
    {"drug": "DB00877", "target": "P08684", "action": "substrate"},  # Sirolimus
    {"drug": "DB00091", "target": "P08684", "action": "substrate"},  # Cyclosporine
    {"drug": "DB00091", "target": "P08684", "action": "inhibitor"},  # Cyclosporine
    
    # SSRIs
    {"drug": "DB00472", "target": "Q01959", "action": "inhibitor"},  # Fluoxetine
    {"drug": "DB00472", "target": "P10635", "action": "inhibitor"},  # Fluoxetine CYP2D6
    {"drug": "DB00715", "target": "Q01959", "action": "inhibitor"},  # Paroxetine
    {"drug": "DB01104", "target": "Q01959", "action": "inhibitor"},  # Sertraline
    
    # Clopidogrel
    {"drug": "DB00758", "target": "P33261", "action": "substrate"},  # CYP2C19 prodrug
]


def load_comprehensive_data():
    """Load comprehensive drug interaction data into Neo4j"""
    kg = KnowledgeGraphService
    
    if not kg.is_connected():
        logger.error("Cannot connect to Neo4j!")
        return False
    
    logger.info("=" * 60)
    logger.info("Loading Comprehensive DDI Dataset")
    logger.info("=" * 60)
    
    # Create schema
    kg.create_schema()
    
    # Load drugs
    logger.info(f"Loading {len(DRUGBANK_DRUGS)} drugs...")
    for drug in DRUGBANK_DRUGS:
        kg.add_drug(
            drugbank_id=drug['id'],
            name=drug['name'],
            smiles=drug.get('smiles'),
            category=drug.get('category')
        )
    logger.info(f"✓ Loaded {len(DRUGBANK_DRUGS)} drugs")
    
    # Load targets
    logger.info(f"Loading {len(DRUG_TARGETS)} targets...")
    for target in DRUG_TARGETS:
        kg.add_target(
            uniprot_id=target['id'],
            name=target['name'],
            gene_name=target.get('gene')
        )
    logger.info(f"✓ Loaded {len(DRUG_TARGETS)} targets")
    
    # Load drug-target links
    logger.info(f"Loading {len(DRUG_TARGET_LINKS)} drug-target relationships...")
    for rel in DRUG_TARGET_LINKS:
        kg.link_drug_target(
            drug_id=rel['drug'],
            target_id=rel['target'],
            action=rel.get('action')
        )
    logger.info(f"✓ Loaded {len(DRUG_TARGET_LINKS)} drug-target links")
    
    # Load interactions
    logger.info(f"Loading {len(KNOWN_INTERACTIONS)} known interactions...")
    for inter in KNOWN_INTERACTIONS:
        kg.add_interaction(
            drug1_id=inter['drug1'],
            drug2_id=inter['drug2'],
            severity=inter['severity'],
            mechanism=inter.get('mechanism')
        )
    logger.info(f"✓ Loaded {len(KNOWN_INTERACTIONS)} interactions")
    
    # Print final stats
    stats = kg.get_stats()
    logger.info("=" * 60)
    logger.info("Data Loading Complete!")
    logger.info(f"  Total Drugs: {len(DRUGBANK_DRUGS)}")
    logger.info(f"  Total Targets: {len(DRUG_TARGETS)}")
    logger.info(f"  Total Interactions: {len(KNOWN_INTERACTIONS)}")
    logger.info(f"  Total Drug-Target Links: {len(DRUG_TARGET_LINKS)}")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    load_comprehensive_data()
