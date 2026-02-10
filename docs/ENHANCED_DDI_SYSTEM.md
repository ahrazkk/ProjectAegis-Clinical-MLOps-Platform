# Enhanced DDI Prediction System

## Overview

This document describes the enhanced Drug-Drug Interaction (DDI) prediction system that builds upon the original PubMedBERT-based approach with additional data sources, AI models, and clinical features.

## New Components

### 1. CYP450 Enzyme Database (`cyp450_database.py`)

A comprehensive database of cytochrome P450 enzyme-drug relationships, providing:

- **Substrates**: Drugs metabolized by each CYP enzyme
- **Inhibitors**: Drugs that inhibit CYP enzymes (strong, moderate, weak)
- **Inducers**: Drugs that induce CYP enzyme activity

#### Key Features
- Automatic detection of pharmacokinetic interactions
- Inhibitor + substrate = increased substrate levels
- Inducer + substrate = decreased substrate levels
- Pre-computed high-risk combination list

#### Example Usage
```python
from ddi_api.services.cyp450_database import get_cyp450_database

db = get_cyp450_database()

# Check if drugs interact via CYP enzymes
interactions = db.check_cyp_interaction("ketoconazole", "simvastatin")
# Returns: CYP3A4-mediated severe interaction (ketoconazole inhibits simvastatin metabolism)

# Get drug's CYP profile
profile = db.get_drug_cyp_profile("warfarin")
# Returns: {"CYP2C9": ["substrate"], "CYP3A4": ["substrate"]}
```

### 2. OpenFDA FAERS Integration (`openfda_service.py`)

Real-time integration with FDA Adverse Event Reporting System (FAERS) for evidence-based interaction assessment.

#### Features
- Query adverse event reports for drug pairs
- Estimate severity from real-world data
- Access drug label information
- Identify top adverse reactions

#### Example Usage
```python
from ddi_api.services.openfda_service import get_openfda_service

fda = get_openfda_service()

# Get adverse events for a drug pair
severity = fda.get_interaction_severity_from_faers("warfarin", "aspirin")
# Returns: {"total_reports": 150, "serious_reports": 89, "estimated_severity": "major"}

# Get drug label information
label = fda.get_drug_label("simvastatin")
# Returns: DrugLabel with warnings, contraindications, etc.
```

### 3. GNN-Based DDI Predictor (`gnn_predictor.py`)

Graph Neural Network-based prediction using molecular structure (SMILES).

#### Features
- Morgan fingerprint generation (ECFP4)
- Tanimoto similarity calculation
- Structure-based interaction prediction
- Works with any drug that has a known structure

#### Advantages Over NLP-Based Approach
- No text context required
- Works for novel drug pairs
- Captures structural similarities
- Complements text-based predictions

#### Example Usage
```python
from ddi_api.services.gnn_predictor import get_gnn_predictor

gnn = get_gnn_predictor()

# Predict interaction using SMILES
prediction = gnn.predict(
    drug1="aspirin",
    drug2="warfarin", 
    smiles1="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    smiles2="CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"  # Warfarin
)
```

### 4. Polypharmacy Risk Scorer (`polypharmacy_scorer.py`)

Comprehensive assessment for patients on multiple medications.

#### Features
- Polypharmacy classification (minor, standard, excessive)
- All-pairs interaction analysis
- High-risk medication identification
- Narrow Therapeutic Index (NTI) drug detection
- Duplicate therapy detection
- Patient-specific risk adjustment (age, renal, hepatic)
- Clinical recommendations generation

#### Example Usage
```python
from ddi_api.services.polypharmacy_scorer import (
    get_polypharmacy_scorer,
    PatientProfile
)

scorer = get_polypharmacy_scorer()

# Assess polypharmacy risk
medications = [
    "warfarin", "aspirin", "simvastatin",
    "lisinopril", "metoprolol", "omeprazole"
]

# Optional: patient profile for personalized assessment
patient = PatientProfile(
    age=72,
    creatinine_clearance=45,
    hepatic_function="mild"
)

report = scorer.assess_polypharmacy_risk(medications, patient)
# Returns comprehensive PolypharmacyRiskReport with:
# - Overall risk level
# - All detected interactions
# - High-risk medications
# - Duplicate therapies
# - Clinical recommendations
```

### 5. Ensemble DDI Predictor (`ensemble_predictor.py`)

Combines multiple prediction sources for robust, explainable predictions.

#### Prediction Sources
1. **PubMedBERT** (25% weight) - NLP-based, literature evidence
2. **CYP450 Database** (30% weight) - High reliability for PK interactions
3. **GNN/ChemicalX** (15% weight) - Structure-based predictions
4. **Knowledge Graph** (20% weight) - Curated database
5. **OpenFDA FAERS** (10% weight) - Real-world evidence

#### Features
- Weighted consensus from multiple sources
- Confidence scoring with consensus levels
- Combined mechanism explanations
- Fallback when sources unavailable
- Comprehensive recommendations

#### Example Usage
```python
from ddi_api.services.ensemble_predictor import get_ensemble_predictor

ensemble = get_ensemble_predictor()

# Get ensemble prediction
prediction = ensemble.predict("warfarin", "aspirin")

print(f"Severity: {prediction.final_severity}")
print(f"Confidence: {prediction.final_confidence:.2%}")
print(f"Consensus: {prediction.consensus_level}")
print(f"Sources used: {len([p for p in prediction.source_predictions if p.available])}")

# View individual source predictions
for source_pred in prediction.source_predictions:
    if source_pred.available:
        print(f"  {source_pred.source.value}: {source_pred.severity}")
```

### 6. Enhanced Data Ingestion (`enhanced_data_ingestion.py`)

Extended data sources for the knowledge graph.

#### New Data Types
- **Food-Drug Interactions**: Grapefruit, dairy, alcohol, vitamin K, etc.
- **Herbal Supplement Interactions**: St. John's Wort, Ginkgo, Kava, etc.
- **PubChem Integration**: Compound data and SMILES
- **ChEMBL Integration**: Bioactivity data

#### Example Usage
```python
from ddi_api.services.enhanced_data_ingestion import get_enhanced_data_ingestion

ingestion = get_enhanced_data_ingestion()

# Get food-drug interactions
grapefruit_interactions = ingestion.get_food_drug_interactions(food_name="grapefruit")

# Get herbal interactions
st_johns_interactions = ingestion.get_herbal_drug_interactions(herb_name="St. John's Wort")

# Export to JSON for knowledge graph
ingestion.export_food_interactions_json()
ingestion.export_herbal_interactions_json()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ensemble DDI Predictor                       │
│  (Combines predictions with weighted voting and consensus)      │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
    ┌─────────▼─────────┐             ┌───────────▼───────────┐
    │   AI/ML Models    │             │   Database Sources    │
    │                   │             │                       │
    │  ┌─────────────┐  │             │  ┌─────────────────┐  │
    │  │ PubMedBERT  │  │             │  │ CYP450 Database │  │
    │  │   (NLP)     │  │             │  │ (Pharmacokinetic│  │
    │  └─────────────┘  │             │  └─────────────────┘  │
    │                   │             │                       │
    │  ┌─────────────┐  │             │  ┌─────────────────┐  │
    │  │ GNN Model   │  │             │  │ Knowledge Graph │  │
    │  │ (Structure) │  │             │  │    (Neo4j)      │  │
    │  └─────────────┘  │             │  └─────────────────┘  │
    └───────────────────┘             │                       │
                                      │  ┌─────────────────┐  │
                                      │  │  OpenFDA FAERS  │  │
                                      │  │ (Real-World)    │  │
                                      │  └─────────────────┘  │
                                      └───────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Polypharmacy Risk Assessment                      │
│  (Multi-drug analysis with patient-specific adjustments)        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Enhanced Data Sources                             │
│  Food-Drug | Herbal-Drug | PubChem | ChEMBL                    │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints (Suggested)

### Ensemble Prediction
```
POST /api/v2/predict/ensemble
{
    "drug1": "warfarin",
    "drug2": "aspirin",
    "smiles1": "...",  // optional
    "smiles2": "...",  // optional
    "use_all_sources": true
}
```

### Polypharmacy Assessment
```
POST /api/v2/polypharmacy
{
    "medications": ["warfarin", "aspirin", "simvastatin", ...],
    "patient": {  // optional
        "age": 72,
        "creatinine_clearance": 45,
        "hepatic_function": "mild"
    }
}
```

### Food/Herbal Interactions
```
GET /api/v2/interactions/food?drug=warfarin
GET /api/v2/interactions/herbal?drug=cyclosporine
```

## Configuration

### Environment Variables
```bash
# OpenFDA API (optional, increases rate limits)
OPENFDA_API_KEY=your_api_key

# Neo4j Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Testing

Run the enhanced services tests:
```bash
cd tests
pytest test_enhanced_services.py -v
```

## Future Enhancements

1. **ChemicalX Integration**: Add more GNN architectures (DeepDDI, MHCADDI)
2. **Real-time FAERS Streaming**: Live adverse event monitoring
3. **Pharmacogenomics**: CYP2D6/CYP2C19 genotype-based predictions
4. **Clinical Trial Integration**: Pre-market safety signals
5. **FHIR Integration**: EHR connectivity
6. **Mobile App**: AR pill scanner with interaction checking

## References

1. ChemicalX: https://arxiv.org/abs/2202.05240
2. DrugBank: https://go.drugbank.com/
3. OpenFDA: https://open.fda.gov/
4. Flockhart Table: https://drug-interactions.medicine.iu.edu/
5. DDI Corpus: http://labda.inf.uc3m.es/ddicorpus/
