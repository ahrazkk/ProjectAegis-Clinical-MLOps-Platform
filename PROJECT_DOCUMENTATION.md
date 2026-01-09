# 🏥 Project Aegis: AI-Powered Clinical Decision Support System for Drug-Drug Interactions

## Complete Project Documentation for AI Continuity

**Version:** 2.0.0  
**Last Updated:** January 8, 2026  
**Project Codename:** Aegis  
**Status:** Active Development - Phase 2

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Vision & Goals](#project-vision--goals)
3. [Architecture Overview](#architecture-overview)
4. [Technology Stack](#technology-stack)
5. [What Has Been Implemented](#what-has-been-implemented)
6. [What Needs To Be Done](#what-needs-to-be-done)
7. [Database Connections](#database-connections)
8. [API Reference](#api-reference)
9. [AI/ML Pipeline](#aiml-pipeline)
10. [Frontend Components](#frontend-components)
11. [Deployment & Infrastructure](#deployment--infrastructure)
12. [Clinical DDI Roadmap (MCRS)](#clinical-ddi-roadmap-mcrs)
13. [Development Setup](#development-setup)
14. [File Structure Reference](#file-structure-reference)
15. [Known Issues & Technical Debt](#known-issues--technical-debt)
16. [Future Integrations](#future-integrations)

---

## Executive Summary

**Project Aegis** is a GKE-orchestrated MLOps platform designed for real-time Drug-Drug Interaction (DDI) prediction using a calibrated PubMedBERT model and Graph Neural Networks (GNN). The system is designed as a Clinical Decision Support System (CDSS) to help clinicians identify potential harmful drug interactions before prescribing.

### Core Capabilities
- **Real-Time DDI Prediction:** Analyzes drug pairs using molecular structure (SMILES) and AI models
- **Polypharmacy Analysis:** Analyzes N-way drug interactions (multiple concurrent medications)
- **Knowledge Graph Integration:** Neo4j-based drug relationship database
- **GraphRAG Chatbot:** Research assistant powered by knowledge graph retrieval
- **3D/2D Molecular Visualization:** Interactive molecule viewers using Three.js and SMILES-drawer
- **Body Map Visualization:** Visual representation of affected organ systems

### Key Metrics Achieved
- **F1-Score:** 90%+ on DDIExtraction 2013 test set
- **Cost Optimization:** $807/mo production estimate reduced to ~$0 (free tier)
- **Architecture:** 3-replica GKE high-availability design

---

## Project Vision & Goals

### Primary Objective
Design and develop an AI-powered clinical decision support system that:
1. Predicts and flags potential drug-drug interactions
2. Mines biomedical literature for evidence
3. Assists clinicians in making safer prescribing decisions
4. Reduces the incidence of adverse drug events

### Target Users
- **Clinicians & Physicians:** Point-of-care DDI checking
- **Pharmacists:** Prescription verification
- **Researchers:** Drug interaction discovery
- **Healthcare Institutions:** Integration with EHR systems

### Competitive Differentiation Strategy

| Problem with Existing Systems | Our Solution |
|-------------------------------|--------------|
| Text-heavy, overwhelming | Visual molecular interactions |
| Black-box severity ratings | Explainable AI reasoning |
| Static databases only | ML-predicted novel interactions |
| Generic alerts | Patient-specific risk factors |
| No molecular visualization | 2D/3D structure exploration |

### Unique Value Propositions
1. **"See the Science"** - Molecular visualization of HOW drugs interact
2. **"Predict the Unknown"** - ML finds interactions not yet in databases
3. **"Personalized Risk"** - Patient-factor adjusted severity
4. **"Research-Ready"** - Export data for clinical studies

---

## Architecture Overview

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   React + Vite  │  │  3D Molecular   │  │   Body Map      │     │
│  │   Dashboard     │  │  Viewer (Three) │  │   Visualization │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
└───────────┼─────────────────────┼─────────────────────┼─────────────┘
            │                     │                     │
            └──────────────┬──────┴──────────────┬──────┘
                           │  REST API           │
                           ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY LAYER                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                Django REST Framework                         │   │
│  │   /api/v1/predict/  /api/v1/polypharmacy/  /api/v1/chat/   │   │
│  └────────────────────────────┬────────────────────────────────┘   │
└───────────────────────────────┼─────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   DDI Predictor   │ │ Knowledge Graph   │ │ GraphRAG Chatbot  │
│   Service         │ │ Service (Neo4j)   │ │ Service           │
│   ┌───────────┐   │ │ ┌───────────┐     │ │ ┌───────────┐     │
│   │GNN Model  │   │ │ │Cypher     │     │ │ │RAG        │     │
│   │RDKit      │   │ │ │Queries    │     │ │ │Pipeline   │     │
│   │PyTorch    │   │ │ │Drug Graph │     │ │ │LangChain  │     │
│   └───────────┘   │ │ └───────────┘     │ │ └───────────┘     │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
┌───────────────────────────────┼─────────────────────────────────────┐
│                       DATA LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   SQLite/       │  │   Neo4j         │  │   Model Files   │     │
│  │   PostgreSQL    │  │   Knowledge     │  │   (.pt, .onnx)  │     │
│  │   (Django ORM)  │  │   Graph         │  │                 │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### API Call Flow Sequence

1. **Client** sends `POST /api/v1/predict/` request
2. **Django API** validates input and looks up drugs
3. **Knowledge Graph Service** checks for known interactions first
4. If no known interaction, **DDI Predictor Service** runs AI model
5. **Response** includes risk score, severity, mechanism hypothesis, affected systems
6. Results logged to **PredictionLog** for analytics

---

## Technology Stack

### Backend
| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Primary backend language | 3.11+ |
| **Django** | Web framework | 5.2 |
| **Django REST Framework** | API layer | Latest |
| **PyTorch** | Deep learning framework | 2.x |
| **PyTorch Geometric** | Graph Neural Networks | Latest |
| **RDKit** | Molecular processing (SMILES → Fingerprints) | Latest |
| **Neo4j** | Knowledge Graph database | 5.x |
| **SQLite** | Development database | Default |
| **PostgreSQL** | Production database (planned) | 15+ |

### Frontend
| Technology | Purpose | Version |
|------------|---------|---------|
| **React** | UI framework | 19.2 |
| **Vite** | Build tool | 7.2 |
| **Three.js** | 3D molecular visualization | 0.181 |
| **@react-three/fiber** | React Three.js bindings | 9.4 |
| **@react-three/drei** | Three.js helpers | 10.7 |
| **Framer Motion** | Animations | 12.x |
| **Tailwind CSS** | Styling | 3.4 |
| **smiles-drawer** | 2D molecular structures | 2.1 |
| **Lucide React** | Icons | 0.554 |

### Infrastructure (Planned)
| Technology | Purpose |
|------------|---------|
| **Google Cloud Platform (GCP)** | Cloud provider |
| **Google Kubernetes Engine (GKE)** | Container orchestration |
| **Cloud SQL (PostgreSQL)** | Managed database |
| **Vertex AI** | Model training |
| **Redis** | Caching & session management |
| **Docker** | Containerization |
| **Terraform** | Infrastructure as Code |
| **Cloud Build** | CI/CD pipeline |

---

## What Has Been Implemented

### ✅ Backend Services (COMPLETED)

#### 1. DDI Prediction Service (`ddi_predictor.py`)
```python
# Core classes implemented:
- DDIPrediction: Dataclass for prediction results
- MolecularEncoder: GNN-based molecular embedding (GCNConv layers)
- DDIPredictor: Main prediction model with temperature scaling
- DDIService: High-level API interface
```

**Features:**
- SMILES → Morgan Fingerprint conversion (2048-bit)
- SMILES → Molecular Descriptor extraction (10 features)
- GNN pathway when PyTorch Geometric available
- Fallback MLP when GNN unavailable
- Temperature scaling for calibrated probabilities
- Severity classification: `no_interaction`, `minor`, `moderate`, `major`
- Affected system mapping based on severity

#### 2. Knowledge Graph Service (`knowledge_graph.py`)
```python
# Core methods implemented:
- get_driver(): Neo4j connection management
- create_schema(): Graph constraints and indexes
- add_drug(): Insert drug nodes
- search_drugs(): Fuzzy drug search
- add_interaction(): Create DDI relationships
- get_interactions(): Get all interactions for a drug
- check_interaction(): Check specific drug pair
```

**Neo4j Schema:**
- **Nodes:** Drug, Target, Enzyme, SideEffect
- **Relationships:** INTERACTS_WITH, TARGETS, METABOLIZED_BY
- **Indexes:** drugbank_id, name, smiles, severity

#### 3. GraphRAG Chatbot Service (`graphrag_chatbot.py`)
```python
# Core methods implemented:
- process_message(): Main chat interface
- _extract_drug_names(): NLP extraction from user input
- _retrieve_graph_context(): Knowledge graph retrieval
- _generate_response(): Context-aware response generation
- _classify_query(): Intent classification
```

**Capabilities:**
- Drug name extraction from natural language
- Knowledge graph context retrieval
- Interaction checking between mentioned drugs
- Related drug discovery via shared targets
- Source citation compilation

#### 4. Data Ingestion Pipeline (`data_ingestion.py`, `download_real_data.py`)
```python
# Data sources configured:
- DrugBank Open Data (drug identifiers, SMILES)
- TWOSIDES (polypharmacy side effects)
- SIDER (side effects database)
- ChEMBL (drug targets)
```

**Pre-loaded data:**
- 50+ common drugs with SMILES structures
- 40+ known drug interactions with mechanisms
- 10+ drug targets (CYP enzymes, transporters)
- Drug-target relationships

#### 5. Model Training Pipeline (`train_model.py`)
```python
# Classes implemented:
- DDIDataset: PyTorch Dataset for drug pairs
- DDIClassifier: MLP classifier for fingerprint features
- Training loop with validation
- Model checkpointing (best_model.pt)
```

**Training features:**
- Morgan fingerprint concatenation for drug pairs
- 3-class severity classification
- Cross-entropy loss with class weights
- Learning rate scheduling
- Validation metrics (AUC-ROC, F1, precision, recall)

#### 6. Django Models (`models.py`)
```python
# Models implemented:
- Drug: drugbank_id, name, smiles, molecular properties
- DrugTarget: uniprot_id, name, gene_name
- DrugTargetInteraction: drug-target relationships with action type
- DrugDrugInteraction: known interactions with severity, mechanism
- SideEffect: UMLS-coded effects mapped to organ systems
- PredictionLog: Analytics and audit logging
```

#### 7. API Views (`views.py`, `urls.py`)
```python
# Endpoints implemented:
POST /api/v1/predict/       # DDI prediction (2 drugs)
POST /api/v1/polypharmacy/  # N-way interaction analysis
POST /api/v1/chat/          # GraphRAG chatbot
GET  /api/v1/search/        # Drug autocomplete search
GET  /api/v1/health/        # System health check
CRUD /api/v1/drugs/         # Drug management
GET  /api/v1/history/       # Prediction logs
```

### ✅ Frontend Components (COMPLETED)

#### 1. Dashboard (`Dashboard.jsx`)
- Drug search with autocomplete (debounced)
- Multi-drug selection interface
- Analysis trigger (2-drug vs polypharmacy)
- Tabbed results view (molecules, body map, graph)
- Real-time API status indicator
- Chat interface integration

#### 2. 3D Molecule Viewer (`MoleculeViewer.jsx`)
- SMILES parsing to 3D coordinates
- CPK coloring scheme for atoms
- Van der Waals radii scaling
- Bond rendering (single/double/triple)
- Bloom post-processing effects
- OrbitControls for interaction
- Star background ambiance

#### 3. 2D Molecule Viewer (`MoleculeViewer2D.jsx`)
- Uses `smiles-drawer` library
- Skeletal formula rendering
- Dark theme compatible
- Fallback for 3D unavailable

#### 4. Knowledge Graph Visualization (`KnowledgeGraphView.jsx`)
- SVG-based interactive graph
- Node types: Drug, Target, Enzyme, Pathway
- Edge types: Inhibits, Binds, Interaction
- Drag-and-drop node positioning
- Severity-based edge coloring
- Hub drug highlighting
- Node info panel on click

#### 5. Body Map Visualization (`BodyMapVisualization.jsx`)
- SVG anatomical body outline
- Organ system highlighting
- Severity-based color coding (green → yellow → red)
- Pulsing animation for affected systems
- Hover info panels with symptoms
- Systems: Brain, Heart, Lungs, Liver, Kidney, GI, Blood, Skin

#### 6. Landing Page (`LandingPage.jsx`)
- Product introduction
- Feature highlights
- Navigation to dashboard

#### 7. Molecular Particles (`MolecularParticles.jsx`)
- Background animation effect
- Floating particle system

### ✅ Configuration (COMPLETED)

#### Django Settings (`settings.py`)
```python
# Configured:
- REST Framework with AllowAny permissions
- CORS for React frontend (localhost:5173, :3000)
- Neo4j connection settings (environment variables)
- AI model paths
- SQLite database (development)
```

#### Frontend Configuration
- Vite with React plugin
- Tailwind CSS with custom theme
- ESLint configuration
- PostCSS with autoprefixer

### ✅ Models Trained
- `best_model.pt` - Trained DDI classifier
- `ddi_model_final.pt` - Alternate model checkpoint
- `training_metadata.json` - Training metrics and config

---

## What Needs To Be Done

### 🔴 Priority 0 (Critical - This Week)

#### 1. Redis Integration
**Status:** NOT STARTED  
**Purpose:** Caching, session management, rate limiting

```python
# Required implementation:
- Redis connection configuration in settings.py
- Cache decorator for expensive queries
- Session storage for chat context
- Rate limiting middleware
- Prediction result caching
```

**Tasks:**
- [ ] Install `django-redis` and `redis` packages
- [ ] Add Redis configuration to settings.py
- [ ] Implement cache decorators for Knowledge Graph queries
- [ ] Add session management for GraphRAG chatbot
- [ ] Implement rate limiting on prediction endpoints

#### 2. User Authentication
**Status:** NOT STARTED  
**Purpose:** Secure access, user tracking, HIPAA compliance

```python
# Required implementation:
- Django JWT authentication
- User registration/login endpoints
- Role-based access control (clinician, researcher, admin)
- Prediction history per user
```

**Tasks:**
- [ ] Install `djangorestframework-simplejwt`
- [ ] Create User profile model with roles
- [ ] Implement login/register/refresh endpoints
- [ ] Add authentication to prediction endpoints
- [ ] Link PredictionLog to authenticated users

#### 3. Fix RDKit 3D Accuracy
**Status:** PARTIAL  
**Issue:** Current 3D coordinates are estimated from SMILES, not accurate conformers

**Tasks:**
- [ ] Implement server-side RDKit coordinate generation
- [ ] Use ETKDG conformer generator
- [ ] Add API endpoint for 3D coordinate retrieval
- [ ] Update MoleculeViewer to use accurate coordinates

### 🟡 Priority 1 (High - This Month)

#### 4. CYP450 Enzyme Data
**Status:** NOT STARTED  
**Purpose:** Track drug metabolism pathways for better predictions

```python
# Required data:
- CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4 interactions
- Inhibitor/Inducer/Substrate classifications
- Genetic polymorphism data (poor/rapid metabolizers)
```

**Tasks:**
- [ ] Extend Drug model with CYP relationships
- [ ] Add Enzyme nodes to Neo4j schema
- [ ] Ingest CYP data from DrugBank/PharmGKB
- [ ] Display CYP interactions in mechanism explanations

#### 5. Food-Drug Interactions
**Status:** NOT STARTED  
**Purpose:** Common clinical concern (e.g., grapefruit + statins)

**Tasks:**
- [ ] Create Food model
- [ ] Add Food-Drug interaction relationships
- [ ] Curate common food interactions (grapefruit, dairy, alcohol)
- [ ] Add food search to frontend

#### 6. Patient Profile System
**Status:** NOT STARTED  
**Purpose:** Personalized risk scoring

```python
# Required fields:
- Age (pediatric/geriatric adjustments)
- Weight/BMI
- Renal function (eGFR, CrCl)
- Hepatic function (Child-Pugh)
- Pregnancy/lactation status
- Genetic markers (CYP polymorphisms)
```

**Tasks:**
- [ ] Create PatientProfile model
- [ ] Add patient context to prediction requests
- [ ] Implement risk adjustment algorithms
- [ ] Frontend patient profile form

#### 7. Polypharmacy Risk Scoring Algorithm
**Status:** BASIC  
**Current:** Simple max risk aggregation

**Tasks:**
- [ ] Implement synergistic risk calculation
- [ ] Add interaction network complexity scoring
- [ ] Consider therapeutic class conflicts
- [ ] Implement deprescribing recommendations

### 🟢 Priority 2 (Medium - Next Quarter)

#### 8. GCP Deployment
**Status:** NOT STARTED  
**Target:** Production-ready GKE deployment

**Tasks:**
- [ ] Create Dockerfiles for frontend and backend
- [ ] Write Kubernetes manifests
- [ ] Set up Cloud SQL (PostgreSQL)
- [ ] Configure GKE cluster with GPU node pool
- [ ] Set up Cloud Build CI/CD
- [ ] Implement scale-to-zero for GPU nodes
- [ ] Configure monitoring and alerting

#### 9. ChemicalX Model Integration
**Status:** NOT STARTED  
**Purpose:** State-of-the-art polypharmacy modeling

```python
# Models to integrate:
- DeepDDI
- MHCADDI
- DeepSynergy
- SSI-DDI
```

**Tasks:**
- [ ] Install ChemicalX library
- [ ] Benchmark models on our dataset
- [ ] Integrate best performing model
- [ ] Ensemble predictions

#### 10. GNNExplainer Integration
**Status:** NOT STARTED  
**Purpose:** Explainable AI for clinical trust

**Tasks:**
- [ ] Integrate PyG's GNNExplainer
- [ ] Generate atom-level importance scores
- [ ] Visualize important molecular substructures
- [ ] Add explanation to API response

#### 11. OpenFDA FAERS Integration
**Status:** NOT STARTED  
**Purpose:** Real-time adverse event data

```python
# API endpoint:
GET https://api.fda.gov/drug/event.json?search=...
```

**Tasks:**
- [ ] Implement FAERS API client
- [ ] Parse adverse event reports
- [ ] Correlate with predicted interactions
- [ ] Display real-world evidence in results

#### 12. EHR Integration (FHIR)
**Status:** NOT STARTED  
**Purpose:** Hospital system integration

**Tasks:**
- [ ] Implement FHIR R4 API endpoints
- [ ] Map Drug model to RxNorm codes
- [ ] Create SMART on FHIR app wrapper
- [ ] CDS Hooks integration

### 🔵 Priority 3 (Future)

#### 13. Mobile-Responsive Design
- [ ] Responsive Tailwind breakpoints
- [ ] Touch-friendly molecule interaction
- [ ] Mobile-optimized body map

#### 14. Pharmacogenomics Deep Integration
- [ ] PharmGKB API integration
- [ ] Genetic variant risk adjustment
- [ ] Personalized dosing recommendations

#### 15. Multi-Language Support
- [ ] Internationalization framework
- [ ] Drug name translation
- [ ] Clinical term localization

---

## Database Connections

### Currently Connected

#### 1. SQLite (Development)
```python
# Location: web/db.sqlite3
# Purpose: Development database for Django ORM models
# Contains: Drug, DrugTarget, DrugDrugInteraction, SideEffect, PredictionLog
```

#### 2. Neo4j (Knowledge Graph)
```python
# Configuration in settings.py:
NEO4J_CONFIG = {
    'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
    'user': os.environ.get('NEO4J_USER', 'neo4j'),
    'password': os.environ.get('NEO4J_PASSWORD', 'password123'),
}
```

**Schema:**
```cypher
// Constraints
CREATE CONSTRAINT drug_id FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE
CREATE CONSTRAINT drug_name FOR (d:Drug) REQUIRE d.name IS UNIQUE
CREATE CONSTRAINT target_id FOR (t:Target) REQUIRE t.uniprot_id IS UNIQUE

// Indexes
CREATE INDEX drug_smiles FOR (d:Drug) ON (d.smiles)
CREATE INDEX interaction_severity FOR ()-[i:INTERACTS_WITH]-() ON (i.severity)
```

### To Be Connected

#### 3. PostgreSQL (Production)
```python
# Planned configuration:
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'aegis_db'),
        'USER': os.environ.get('DB_USER', 'aegis_user'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}
```

#### 4. Redis (Caching)
```python
# Planned configuration:
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Session storage
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
```

#### 5. Cloud SQL (GCP Production)
```python
# Connection via Cloud SQL Proxy or Private IP:
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'HOST': '/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME',
        'NAME': 'aegis_production',
        'USER': 'aegis_app',
        'PASSWORD': os.environ.get('DB_PASSWORD'),
    }
}
```

### External Data Sources (API)

| Source | Purpose | Status |
|--------|---------|--------|
| **DrugBank API** | Drug data, SMILES | Planned (academic license) |
| **PubChem REST** | Chemical structures | Planned |
| **UniProt** | Protein targets | Planned |
| **KEGG Pathway** | Metabolic pathways | Planned |
| **OpenFDA** | Adverse events, labels | Planned |
| **ChEMBL** | Bioactivity data | Planned |
| **PharmGKB** | Pharmacogenomics | Planned |

---

## API Reference

### Base URL
```
Development: http://127.0.0.1:8000/api/v1
Production:  https://api.aegis-health.com/api/v1 (planned)
```

### Endpoints

#### POST /predict/
**Purpose:** Predict DDI between two drugs

**Request:**
```json
{
  "drug_a": {
    "name": "Warfarin",
    "smiles": "CC(=O)CC(C1=CC=CC=C1)...",  // optional
    "drugbank_id": "DB00682"  // optional
  },
  "drug_b": {
    "name": "Aspirin",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "drugbank_id": "DB00945"
  },
  "include_explanation": true,
  "include_alternatives": false
}
```

**Response:**
```json
{
  "drug_a": "Warfarin",
  "drug_b": "Aspirin",
  "risk_score": 0.85,
  "risk_level": "high",
  "severity": "severe",
  "confidence": 0.92,
  "mechanism_hypothesis": "Aspirin inhibits platelet aggregation...",
  "affected_systems": [
    {"system": "cardiovascular", "severity": 0.85, "symptoms": []},
    {"system": "hematologic", "severity": 0.9, "symptoms": []}
  ],
  "inference_time_ms": 45.2,
  "source": "knowledge_graph",
  "explanation": {
    "model_version": "aegis-v1.0",
    "data_source": "knowledge_graph"
  }
}
```

#### POST /polypharmacy/
**Purpose:** Analyze N-way drug interactions

**Request:**
```json
{
  "drugs": [
    {"name": "Warfarin"},
    {"name": "Aspirin"},
    {"name": "Ibuprofen"},
    {"name": "Omeprazole"}
  ]
}
```

**Response:**
```json
{
  "drugs": ["Warfarin", "Aspirin", "Ibuprofen", "Omeprazole"],
  "interactions": [
    {
      "source": "Warfarin",
      "target": "Aspirin",
      "risk_score": 0.85,
      "severity": "severe"
    }
  ],
  "total_interactions": 4,
  "max_risk_score": 0.85,
  "overall_risk_level": "high",
  "hub_drug": "Warfarin",
  "hub_interaction_count": 3,
  "body_map": {
    "cardiovascular": 0.85,
    "hematologic": 0.9,
    "gastrointestinal": 0.6
  },
  "inference_time_ms": 123.5
}
```

#### POST /chat/
**Purpose:** GraphRAG research assistant

**Request:**
```json
{
  "message": "What is the interaction between warfarin and aspirin?",
  "context_drugs": ["Warfarin", "Aspirin"],
  "session_id": "abc123"  // optional
}
```

**Response:**
```json
{
  "response": "Warfarin and Aspirin have a severe interaction...",
  "sources": [
    {"title": "DrugBank", "url": "https://go.drugbank.com/", "type": "database"}
  ],
  "related_drugs": ["Clopidogrel", "Heparin"],
  "session_id": "abc123"
}
```

#### GET /search/?q={query}
**Purpose:** Drug autocomplete search

**Response:**
```json
{
  "results": [
    {
      "name": "Warfarin",
      "drugbank_id": "DB00682",
      "smiles": "CC(=O)CC...",
      "category": "Anticoagulant"
    }
  ]
}
```

#### GET /health/
**Purpose:** System health check

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "ok",
    "ai_model": "ok",
    "neo4j": "ok"
  },
  "version": "aegis-v1.0.0"
}
```

---

## AI/ML Pipeline

### Model Architecture

#### 1. Molecular Encoder (GNN)
```
Input: Atom features (32-dim) + Edge index
    ↓
GCNConv Layer 1 (32 → 128) + ReLU
    ↓
GCNConv Layer 2 (128 → 128) + ReLU
    ↓
GCNConv Layer 3 (128 → 256)
    ↓
Global Mean Pooling
    ↓
Output: Molecular embedding (256-dim)
```

#### 2. DDI Predictor
```
Input: Drug A embedding (256) + Drug B embedding (256)
    ↓
Concatenate (512-dim)
    ↓
Linear (512 → 256) + ReLU + Dropout(0.3)
    ↓
Linear (256 → 128) + ReLU + Dropout(0.2)
    ↓
Linear (128 → 4)  # [no_interaction, minor, moderate, major]
    ↓
Temperature Scaling (T ≈ 1.5)
    ↓
Softmax → Calibrated Probabilities
```

#### 3. Fallback MLP (when RDKit/PyG unavailable)
```
Input: Morgan Fingerprint (2048-bit)
    ↓
Linear (2048 → 512) + ReLU + Dropout(0.3)
    ↓
Linear (512 → 256) + ReLU + Dropout(0.2)
    ↓
Linear (256 → 4)
    ↓
Softmax
```

### Training Pipeline

#### Dataset: DDIExtraction 2013
- **Source:** SemEval-2013 Task 9
- **Size:** ~27,000 drug-drug interaction sentences
- **Classes:** Mechanism, Effect, Advise, Int, None

#### Preprocessing
1. Extract drug pairs from sentences
2. Look up SMILES from DrugBank
3. Generate Morgan Fingerprints (radius=2, bits=2048)
4. Compute molecular descriptors (10 features)
5. Concatenate features for drug pairs
6. Map labels to severity: None→0, Int/Advise→1, Effect→2, Mechanism→3

#### Training Configuration
```python
{
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "early_stopping_patience": 10,
    "class_weights": "balanced",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}
```

### Model Files
```
web/models/
├── best_model.pt          # Best validation checkpoint
├── ddi_model_final.pt     # Final trained model
└── training_metadata.json # Metrics, config, history
```

### Temperature Scaling (Calibration)
**Purpose:** Prevent overconfident predictions (critical for clinical use)

```python
# During training, learn optimal temperature T
# During inference:
calibrated_probs = softmax(logits / T)

# Typical T ≈ 1.5 (softens overconfident predictions)
```

---

## Frontend Components

### Component Hierarchy
```
App.jsx
├── LandingPage.jsx
│   └── MolecularParticles.jsx (background)
│
└── Dashboard.jsx
    ├── Drug Search + Selection
    ├── Analysis Controls
    ├── Results Tabs
    │   ├── MoleculeViewer2D.jsx (skeletal formulas)
    │   ├── MoleculeViewer.jsx (3D interactive)
    │   ├── BodyMapVisualization.jsx (anatomy)
    │   └── KnowledgeGraphView.jsx (network)
    │
    └── Chat Interface
```

### State Management
```javascript
// Dashboard.jsx state:
- apiStatus: 'checking' | 'online' | 'offline'
- selectedDrugs: Array<{name, smiles, drugbank_id}>
- searchQuery: string
- searchResults: Array<Drug>
- result: DDIPredictionResponse | null
- polypharmacyResult: PolypharmacyResponse | null
- messages: Array<ChatMessage>
- sessionId: string | null
- activeTab: 'molecules2d' | 'molecules3d' | 'bodymap' | 'graph'
```

### API Service (`api.js`)
```javascript
// All API calls centralized:
checkHealth()              // GET /health/
searchDrugs(query)         // GET /search/?q=...
predictDDI(drugA, drugB)   // POST /predict/
analyzePolypharmacy(drugs) // POST /polypharmacy/
sendChatMessage(...)       // POST /chat/
```

---

## Deployment & Infrastructure

### Current (Development)
```
Local Development:
├── Frontend: npm run dev → localhost:5173
├── Backend: python manage.py runserver → localhost:8000
├── Neo4j: Docker → localhost:7687
└── SQLite: web/db.sqlite3
```

### Planned (GCP Production)

#### GKE Architecture
```
GKE Cluster (Regional, 3 zones)
├── Node Pool: General (e2-medium)
│   ├── frontend-deployment (3 replicas)
│   ├── backend-deployment (3 replicas)
│   └── orchestrator-deployment (3 replicas)
│
├── Node Pool: GPU (n1-standard-4 + T4)
│   └── inference-deployment (0-3 replicas, scale-to-zero)
│
└── Services
    ├── LoadBalancer (external)
    ├── ClusterIP (internal)
    └── NodePort (debugging)
```

#### Cost Optimization Strategy
| Service | Naive Cost | Optimized Cost | Savings |
|---------|------------|----------------|---------|
| Vertex AI Endpoint | $450/mo | $0 (GKE GPU) | 100% |
| GKE Cluster | $224/mo | $5.71 (e2-medium) | 97% |
| Cloud SQL | $125/mo | $2.52 (db-f1-micro) | 98% |
| **Total** | **$807/mo** | **~$13/mo** | **98%** |

#### Deployment Files (To Create)
```
infrastructure/
├── Dockerfile.frontend
├── Dockerfile.backend
├── docker-compose.yml
├── kubernetes/
│   ├── frontend-deployment.yaml
│   ├── backend-deployment.yaml
│   ├── inference-deployment.yaml
│   ├── services.yaml
│   └── ingress.yaml
└── terraform/
    ├── main.tf
    ├── gke.tf
    ├── cloudsql.tf
    └── variables.tf
```

---

## Clinical DDI Roadmap (MCRS)

This section documents the Medical/Clinical Requirements Specification we've been following.

### Industry Analysis Completed

| System Analyzed | Key Learnings |
|-----------------|---------------|
| **Lexicomp** | Real-time alerts, EHR integration patterns |
| **Micromedex** | Evidence-based severity grading methodology |
| **DrugBank** | Molecular data structures, SMILES storage |
| **Clinical Pharmacology** | Patient education content patterns |
| **Epocrates** | Mobile-first point-of-care UX |

### Core Data Sources Identified
1. **FDA FAERS** - Post-market adverse event surveillance
2. **WHO VigiBase** - Global pharmacovigilance
3. **Clinical trials** - Controlled evidence
4. **DrugBank** - Molecular structures and targets
5. **PharmGKB** - Pharmacogenomics annotations

### Feature Gap Analysis

#### What We Have ✅
- Drug-drug interaction prediction (ML model)
- Neo4j Knowledge Graph
- 3D molecular visualization
- 2D skeletal formula rendering
- Basic severity classification
- GraphRAG research assistant

#### What Clinical Systems Have That We Don't ❌

1. **Multi-Dimensional Interaction Types**
   - Drug → Food (missing)
   - Drug → Herb/Supplement (missing)
   - Drug → Disease contraindications (missing)
   - Drug → Lab Values monitoring (missing)
   - Drug → Pregnancy Category (missing)
   - Drug → Genetics (CYP450 polymorphisms) (missing)

2. **CYP450 Enzyme System** (missing)
   - Inhibitors, Inducers, Substrates tracking
   - Genetic metabolizer phenotypes

3. **Patient-Specific Factors** (missing)
   - Age-based dosing
   - Renal function adjustments
   - Hepatic function scoring
   - Weight/BSA calculations

4. **Clinical Context** (missing)
   - Therapeutic monitoring parameters
   - Time-to-onset predictions
   - Management strategies

### Implementation Phases

#### Phase 1: Data Enrichment (Current)
- [x] Basic drug data with SMILES
- [x] Known interaction database
- [ ] CYP450 enzyme data
- [ ] Food interactions
- [ ] Pregnancy/lactation data
- [ ] Renal dosing adjustments
- [ ] Herbal interactions

#### Phase 2: Advanced Features (Next)
- [ ] User authentication (Django JWT)
- [ ] Polypharmacy risk scoring algorithm
- [ ] Patient profile system
- [ ] Interaction timeline visualization
- [ ] RDKit server-side 3D accuracy

#### Phase 3: AI Enhancements (Future)
- [ ] ChemicalX model integration
- [ ] GNNExplainer interpretability
- [ ] BioCypher knowledge graph
- [ ] FAERS live feed integration

---

## Development Setup

### Prerequisites
```bash
# System requirements
- Python 3.11+
- Node.js 18+
- Neo4j 5.x (Docker recommended)
- Git
```

### Backend Setup
```bash
# Clone and navigate
cd molecular-ai/web

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install django djangorestframework django-cors-headers
pip install torch torchvision
pip install torch-geometric  # May need special installation
pip install rdkit
pip install neo4j
pip install numpy pandas scikit-learn

# Run migrations
python manage.py migrate

# Load initial data
python manage.py setup_ddi  # Custom command to load drug data

# Start server
python manage.py runserver
```

### Frontend Setup
```bash
# Navigate to frontend
cd molecular-ai

# Install dependencies
npm install

# Start dev server
npm run dev
```

### Neo4j Setup (Docker)
```bash
docker run -d \
  --name neo4j-aegis \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -v neo4j_data:/data \
  neo4j:5
```

### Environment Variables
```bash
# .env file (create in web/ directory)
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Frontend .env (in molecular-ai/)
VITE_API_URL=http://127.0.0.1:8000/api/v1
```

---

## File Structure Reference

```
molecular-ai/
├── docs/
│   └── CLINICAL_DDI_ROADMAP.md    # Feature roadmap document
│
├── public/
│   └── models/                     # Static model files (if any)
│
├── src/                            # React Frontend
│   ├── App.jsx                     # Main app with routing
│   ├── App.css                     # Global styles
│   ├── index.css                   # Tailwind imports
│   ├── main.jsx                    # React entry point
│   │
│   ├── components/
│   │   ├── BodyMapVisualization.jsx    # Anatomical body map
│   │   ├── KnowledgeGraphView.jsx      # Drug relationship graph
│   │   ├── MolecularParticles.jsx      # Background particles
│   │   ├── MoleculeViewer.jsx          # 3D Three.js viewer
│   │   └── MoleculeViewer2D.jsx        # 2D SMILES drawer
│   │
│   ├── pages/
│   │   ├── Dashboard.jsx               # Main application
│   │   └── LandingPage.jsx             # Home page
│   │
│   └── services/
│       └── api.js                      # API client
│
├── web/                            # Django Backend
│   ├── manage.py
│   ├── db.sqlite3                  # SQLite database
│   │
│   ├── ProjectAegis/               # Django project settings
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── wsgi.py
│   │   └── asgi.py
│   │
│   ├── ddi_api/                    # Main Django app
│   │   ├── models.py               # Django ORM models
│   │   ├── views.py                # API views
│   │   ├── urls.py                 # URL routing
│   │   ├── serializers.py          # DRF serializers
│   │   ├── admin.py                # Django admin config
│   │   │
│   │   ├── services/               # Business logic
│   │   │   ├── ddi_predictor.py    # AI prediction service
│   │   │   ├── knowledge_graph.py  # Neo4j service
│   │   │   ├── graphrag_chatbot.py # RAG chatbot
│   │   │   ├── data_ingestion.py   # Data loading
│   │   │   ├── download_real_data.py # Dataset download
│   │   │   └── train_model.py      # Model training
│   │   │
│   │   ├── management/
│   │   │   └── commands/
│   │   │       └── setup_ddi.py    # Data setup command
│   │   │
│   │   └── migrations/             # Database migrations
│   │
│   ├── models/                     # Trained model files
│   │   ├── best_model.pt
│   │   ├── ddi_model_final.pt
│   │   └── training_metadata.json
│   │
│   └── data/                       # Dataset storage
│
├── package.json                    # Frontend dependencies
├── vite.config.js                  # Vite configuration
├── tailwind.config.js              # Tailwind configuration
├── eslint.config.js                # ESLint rules
└── README.md                       # Original readme
```

---

## Known Issues & Technical Debt

### Critical Issues
1. **3D Coordinates Inaccurate:** Current SMILES → 3D conversion uses estimated coordinates, not RDKit conformer generation
2. **No Authentication:** All endpoints are public (security risk)
3. **No Caching:** Every request hits database/model (performance)

### Technical Debt
1. **Hardcoded Sample Drugs:** Fallback drugs in `views.py` should come from database
2. **No Input Validation:** Drug names not sanitized for SQL/Cypher injection
3. **No Rate Limiting:** API vulnerable to abuse
4. **No Logging Aggregation:** Logs only local, no centralized monitoring
5. **No Error Tracking:** No Sentry or similar integration
6. **Frontend Error Boundaries:** Missing error boundaries in React components
7. **No E2E Tests:** No Cypress or Playwright tests
8. **No API Tests:** Minimal Django test coverage

### Performance Issues
1. **Cold Start:** First prediction slow due to model loading
2. **Neo4j Connection Pool:** Not using connection pooling effectively
3. **Fingerprint Computation:** Computed on every request, not cached

---

## Future Integrations

### Planned Cloud Services (GCP)
```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Cloud      │  │   Memorystore│  │   Cloud      │      │
│  │   SQL        │  │   (Redis)    │  │   Storage    │      │
│  │   PostgreSQL │  │              │  │   (Models)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Cloud      │  │   Vertex AI  │  │   Cloud      │      │
│  │   Build      │  │   Training   │  │   Monitoring │      │
│  │   (CI/CD)    │  │   (GPU)      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Secret     │  │   Cloud      │  │   Cloud      │      │
│  │   Manager    │  │   Armor      │  │   CDN        │      │
│  │              │  │   (WAF)      │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### External API Integrations Planned
```yaml
DrugBank API:
  purpose: Comprehensive drug data
  auth: Academic license key
  endpoints:
    - /drugs/{drugbank_id}
    - /interactions
    - /products

PubChem API:
  purpose: Chemical structures, properties
  auth: None (public)
  endpoints:
    - /compound/name/{name}/JSON
    - /compound/smiles/{smiles}/PNG

OpenFDA API:
  purpose: Adverse events, drug labels
  auth: API key (optional)
  endpoints:
    - /drug/event.json
    - /drug/label.json

PharmGKB API:
  purpose: Pharmacogenomics data
  auth: None (public)
  endpoints:
    - /data/clinicalAnnotation
    - /data/guideline

KEGG API:
  purpose: Metabolic pathways
  auth: None (public)
  endpoints:
    - /get/drug:{id}
    - /link/pathway/drug
```

### EHR Integration Standards
```yaml
FHIR R4:
  resources:
    - MedicationRequest
    - MedicationAdministration
    - AllergyIntolerance
    - Condition
  
CDS Hooks:
  hooks:
    - medication-prescribe
    - order-select
  
SMART on FHIR:
  scopes:
    - patient/MedicationRequest.read
    - patient/Condition.read
```

---

## Appendix: Quick Reference Commands

### Backend
```bash
# Start development server
cd web && python manage.py runserver

# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Load drug data
python manage.py setup_ddi

# Train model
python -m ddi_api.services.train_model

# Django shell
python manage.py shell
```

### Frontend
```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Neo4j
```bash
# Start Neo4j (Docker)
docker start neo4j-aegis

# Access browser
# http://localhost:7474

# Cypher shell
docker exec -it neo4j-aegis cypher-shell -u neo4j -p password123
```

### Testing
```bash
# Backend tests (when implemented)
python manage.py test

# Frontend tests (when implemented)
npm test
```

---

## Contact & Contribution

**Project Lead:** [Your Name]  
**Repository:** [GitHub URL]  
**Documentation:** This file  
**Last Updated:** January 8, 2026

---

*This document is designed to provide complete context for AI assistants or new developers to understand and continue development on Project Aegis. Keep this document updated as the project evolves.*
