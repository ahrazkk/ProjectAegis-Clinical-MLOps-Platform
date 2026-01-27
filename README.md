# Project Aegis: AI-Powered Clinical Decision Support for Drug-Drug Interactions

<div align="center">

![Project Aegis](https://img.shields.io/badge/Project-Aegis-00d4ff?style=for-the-badge&logo=moleculer&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-Ready-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

**A GKE-orchestrated MLOps platform for real-time drug-drug interaction prediction using fine-tuned PubMedBERT**

[Features](#-key-features) • [Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [API Reference](#-api-reference) • [Future Roadmap](#-future-roadmap)

</div>

---

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Quick Start](#-quick-start)
6. [Detailed Installation](#-detailed-installation)
7. [API Reference](#-api-reference)
8. [How It Works - High Level](#-how-it-works---high-level)
9. [How It Works - Technical Deep Dive](#-how-it-works---technical-deep-dive)
10. [Usage Examples](#-usage-examples)
11. [Database & Data Pipeline](#-database--data-pipeline)
12. [The AI Model](#-the-ai-model)
13. [Frontend Interface](#-frontend-interface)
14. [GCP Deployment](#-gcp-deployment)
15. [Future Roadmap](#-future-roadmap)
16. [FAQ](#-frequently-asked-questions)
17. [Troubleshooting](#-troubleshooting)
18. [Contributing](#-contributing)
19. [License](#-license)
20. [Team & Acknowledgments](#-team--acknowledgments)

---

## 🎯 Project Overview

**Project Aegis** is a comprehensive AI-powered clinical decision support system designed to predict and flag potential drug-drug interactions (DDIs). The system helps clinicians make safer prescribing decisions by:

- **Mining biomedical literature** using PubMedBERT, a BERT model pre-trained on PubMed abstracts
- **Predicting interaction severity** (None, Minor, Moderate, Major, Severe)
- **Explaining interaction mechanisms** (CYP450 inhibition, protein binding, pharmacodynamic effects)
- **Suggesting therapeutic alternatives** when dangerous interactions are detected
- **Providing real-time analysis** via a modern, futuristic web interface

### Project Statistics

| Metric | Value |
|--------|-------|
| **Drugs in Database** | 2,088+ |
| **Known Interactions** | 1,703+ |
| **Model F1-Score** | ~90% |
| **API Response Time** | <500ms |
| **Docker Services** | 4 (Backend, Frontend, Redis, Neo4j) |

---

## ✨ Key Features

### 🔬 Core Prediction Engine
- **PubMedBERT Integration**: Fine-tuned on DDIExtraction 2013 corpus (~19,000 annotated sentences)
- **5-Class Classification**: Predicts `mechanism`, `effect`, `advise`, `int`, or `no_interaction`
- **Confidence Scoring**: Calibrated probability outputs for clinical reliability
- **Context Transparency**: Shows the exact sentence used for prediction

### 📊 Drug Comparison Dashboard
- **Side-by-Side Comparison**: Compare 2-5 drugs simultaneously
- **Interaction Matrix**: Visual NxN grid showing all pairwise interactions
- **Risk Assessment**: Color-coded severity indicators (Safe → Severe)
- **Drug Properties**: Molecular weight, formula, therapeutic class, SMILES structure

### 💊 Therapeutic Alternatives
- **Smart Recommendations**: Suggests safer alternatives for problematic drug pairs
- **Same-Class Alternatives**: Finds drugs in the same therapeutic category
- **Compatibility Checking**: Verifies alternatives don't introduce new interactions

### 📈 Statistics Dashboard
- **Real-Time Metrics**: Drug count, interaction count, coverage percentage
- **Database Analytics**: Breakdown by therapeutic class and severity
- **System Monitoring**: API status, response times, model health

### 🎨 Modern UI/UX
- **Futuristic UI (FUI) Theme**: Dark mode with cyan/purple accent colors
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Animated Transitions**: Smooth Framer Motion animations
- **Real-Time Updates**: Live data without page refreshes

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROJECT AEGIS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌──────────────────────────────────────────────────┐  │
│   │   Client    │────▶│              NGINX (Port 80)                     │  │
│   │   Browser   │     │         Static Files + Reverse Proxy             │  │
│   └─────────────┘     └──────────────────────────────────────────────────┘  │
│                                        │                                     │
│                                        ▼ /api/*                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                     DJANGO REST API (Port 8000)                       │  │
│   │                                                                        │  │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │  │
│   │  │ /api/v1/drugs/ │  │ /api/v1/predict│  │ /api/v1/compare/       │   │  │
│   │  │ Drug CRUD      │  │ DDI Prediction │  │ Multi-Drug Comparison  │   │  │
│   │  └────────────────┘  └────────────────┘  └────────────────────────┘   │  │
│   │                                                                        │  │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │  │
│   │  │ /api/v1/stats/ │  │/api/v1/altern/ │  │ /api/v1/chat/          │   │  │
│   │  │ DB Statistics  │  │ Alternatives   │  │ GraphRAG Chatbot       │   │  │
│   │  └────────────────┘  └────────────────┘  └────────────────────────┘   │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                  │                    │                    │                 │
│                  ▼                    ▼                    ▼                 │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│   │     SQLite       │  │   PubMedBERT    │  │        Neo4j             │  │
│   │    Database      │  │     Model       │  │   Knowledge Graph        │  │
│   │                  │  │  (Fine-tuned)   │  │                          │  │
│   │  • Drugs         │  │                  │  │  • Drug relationships   │  │
│   │  • Interactions  │  │  • 5-class DDI  │  │  • Pathway connections   │  │
│   │  • Side Effects  │  │    classifier   │  │  • Target proteins       │  │
│   └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
│                                     │                                        │
│                                     ▼                                        │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         Redis Cache                                   │  │
│   │                   • Prediction caching                                │  │
│   │                   • Session management                                │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Docker Services

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| **Frontend** | `aegis-frontend` | 80 | React app served via NGINX |
| **Backend** | `aegis-backend` | 8000 | Django REST API + PubMedBERT |
| **Redis** | `aegis-redis` | 6379 | Caching and session storage |
| **Neo4j** | `aegis-neo4j` | 7474/7687 | Knowledge graph database |

---

## 🛠️ Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11 | Core language |
| Django | 4.x | Web framework |
| Django REST Framework | 3.x | API layer |
| PyTorch | 2.x | Deep learning |
| Transformers | 4.x | PubMedBERT model |
| Gunicorn | Latest | WSGI server |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18 | UI framework |
| Vite | 5.x | Build tool |
| Tailwind CSS | 3.x | Styling |
| Framer Motion | 11.x | Animations |
| Lucide React | Latest | Icons |
| Axios | Latest | HTTP client |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |
| NGINX | Reverse proxy + static files |
| Redis | Caching |
| Neo4j | Graph database |
| SQLite | Primary database (dev) |
| PostgreSQL | Primary database (prod/GCP) |

### AI/ML
| Component | Details |
|-----------|---------|
| Base Model | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Fine-tuning Dataset | DDIExtraction 2013 Corpus |
| Task | Multi-class Sequence Classification |
| Classes | 5 (mechanism, effect, advise, int, no_interaction) |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- Git
- 8GB+ RAM recommended (for model loading)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/your-org/project-aegis.git
cd project-aegis/molecular-ai

# Start all services
docker-compose up -d --build


# GCP deployment
gcloud run deploy aegis-frontend --source . --region us-central1 --allow-unauthenticated --project project-aegis-485017 --port 8080

# Wait for services to initialize (~30 seconds)
# Then open http://localhost in your browser
```

### Verify Installation

```bash
# Check all containers are running
docker-compose ps

# Expected output:
# NAME              STATUS          PORTS
# aegis-backend     Up              0.0.0.0:8000->8000/tcp
# aegis-frontend    Up              0.0.0.0:80->80/tcp
# aegis-redis       Up              0.0.0.0:6379->6379/tcp
# aegis-neo4j       Up              0.0.0.0:7474->7474/tcp

# Test API
curl http://localhost:8000/api/v1/stats/
```

---

## 📦 Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/project-aegis.git
cd project-aegis
```

### Step 2: Project Structure

```
project-aegis/
├── DDI_Model_Final/              # Fine-tuned PubMedBERT model
│   ├── config.json
│   ├── model.safetensors         # Model weights (~400MB)
│   ├── tokenizer.json
│   └── vocab.txt
│
└── molecular-ai/                 # Main application
    ├── docker-compose.yml        # Docker orchestration
    ├── Dockerfile                # Frontend build
    ├── package.json              # Node dependencies
    ├── tailwind.config.js        # Tailwind configuration
    ├── vite.config.js            # Vite configuration
    │
    ├── src/                      # React frontend
    │   ├── App.jsx
    │   ├── pages/
    │   │   ├── Dashboard.jsx     # Main dashboard
    │   │   └── LandingPage.jsx
    │   ├── components/
    │   │   ├── DrugComparison.jsx
    │   │   ├── StatsDashboard.jsx
    │   │   └── ...
    │   └── services/
    │       └── api.js            # API client
    │
    └── web/                      # Django backend
        ├── manage.py
        ├── requirements.txt
        ├── db.sqlite3            # SQLite database
        │
        ├── ddi_api/              # Main API app
        │   ├── models.py         # Django models
        │   ├── views.py          # API endpoints
        │   ├── serializers.py    # DRF serializers
        │   └── services/
        │       ├── pubmedbert_predictor.py
        │       ├── drug_service.py
        │       └── knowledge_graph.py
        │
        └── ProjectAegis/         # Django project settings
            ├── settings.py
            └── urls.py
```

### Step 3: Environment Configuration

Create `molecular-ai/web/.env`:

```env
# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (for production)
DATABASE_URL=postgres://user:pass@host:5432/dbname

# Redis
REDIS_URL=redis://aegis-redis:6379/0

# Neo4j
NEO4J_URI=bolt://aegis-neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Model Path
DDI_MODEL_PATH=/app/DDI_Model_Final
```

### Step 4: Build and Run

```bash
cd molecular-ai

# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Step 5: Initialize Database

```bash
# Run migrations
docker-compose exec backend python manage.py migrate

# Load drug data (if available)
docker-compose exec backend python manage.py setup_ddi
```

---

## 📡 API Reference

### Base URL
- **Local Development**: `http://localhost:8000/api/v1/`
- **Production (GCP)**: `https://api.projectaegis.com/api/v1/`

### Authentication
Currently using session-based authentication for development. JWT/OAuth2 planned for production.

---

### Endpoints

#### 1. Drug Search

```http
GET /api/v1/drugs/search/?q={query}
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| q | string | Yes | Search query (drug name) |

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/drugs/search/?q=warfarin"
```

**Example Response:**
```json
{
  "count": 3,
  "results": [
    {
      "name": "Warfarin",
      "drugbank_id": "DB00682",
      "therapeutic_class": "Anticoagulant",
      "molecular_formula": "C19H16O4",
      "molecular_weight": 308.3,
      "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
      "interaction_count": 8
    }
  ]
}
```

---

#### 2. DDI Prediction

```http
POST /api/v1/predict/
```

**Request Body:**
```json
{
  "drug1": "Warfarin",
  "drug2": "Aspirin"
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"drug1": "Warfarin", "drug2": "Aspirin"}'
```

**Example Response:**
```json
{
  "drug1": "Warfarin",
  "drug2": "Aspirin",
  "interaction_type": "effect",
  "severity": "severe",
  "confidence": 0.89,
  "risk_score": 0.85,
  "mechanism": "Aspirin inhibits platelet aggregation and may enhance the anticoagulant effect of Warfarin, significantly increasing bleeding risk.",
  "clinical_recommendation": "Avoid combination. If necessary, use lowest effective aspirin dose and monitor INR frequently.",
  "probabilities": {
    "effect": 0.89,
    "mechanism": 0.06,
    "advise": 0.03,
    "int": 0.01,
    "no_interaction": 0.01
  },
  "context_sentence": "Concurrent administration of Warfarin and Aspirin significantly increases the risk of serious bleeding events.",
  "context_source": "template"
}
```

---

#### 3. Drug Comparison (Multi-Drug)

```http
POST /api/v1/compare/
```

**Request Body:**
```json
{
  "drugs": ["Warfarin", "Aspirin", "Ibuprofen"]
}
```

**Example Response:**
```json
{
  "drugs": [
    {
      "name": "Warfarin",
      "drugbank_id": "DB00682",
      "therapeutic_class": "Anticoagulant",
      "interaction_count": 8,
      "molecular_weight": 308.3,
      "molecular_formula": "C19H16O4",
      "smiles": "..."
    }
  ],
  "drug_names": ["Warfarin", "Aspirin", "Ibuprofen"],
  "risk_matrix": [
    [{"severity": "self"}, {"severity": "severe"}, {"severity": "moderate"}],
    [{"severity": "severe"}, {"severity": "self"}, {"severity": "minor"}],
    [{"severity": "moderate"}, {"severity": "minor"}, {"severity": "self"}]
  ],
  "pairwise_interactions": [
    {
      "drug1": "Warfarin",
      "drug2": "Aspirin",
      "severity": "severe",
      "mechanism": "Increased bleeding risk"
    }
  ],
  "total_interactions": 3,
  "severe_count": 1,
  "moderate_count": 1,
  "minor_count": 1
}
```

---

#### 4. Therapeutic Alternatives

```http
GET /api/v1/alternatives/?drug1={drug1}&drug2={drug2}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/alternatives/?drug1=Warfarin&drug2=Aspirin"
```

**Example Response:**
```json
{
  "original_pair": {
    "drug1": "Warfarin",
    "drug2": "Aspirin",
    "severity": "severe"
  },
  "alternatives_for_drug1": [
    {
      "name": "Apixaban",
      "therapeutic_class": "Anticoagulant",
      "interaction_with_drug2": "moderate",
      "recommendation": "Consider as alternative - lower bleeding risk"
    }
  ],
  "alternatives_for_drug2": [
    {
      "name": "Acetaminophen",
      "therapeutic_class": "Analgesic",
      "interaction_with_drug1": "none",
      "recommendation": "Safe alternative for pain relief"
    }
  ]
}
```

---

#### 5. Database Statistics

```http
GET /api/v1/stats/
```

**Example Response:**
```json
{
  "total_drugs": 2088,
  "total_interactions": 1703,
  "coverage_percentage": 35.2,
  "drugs_with_smiles": 1850,
  "drugs_with_interactions": 734,
  "by_therapeutic_class": {
    "Anticoagulant": 45,
    "NSAID": 32,
    "Antibiotic": 128
  },
  "by_severity": {
    "severe": 234,
    "major": 456,
    "moderate": 678,
    "minor": 335
  }
}
```

---

#### 6. Interaction Info

```http
GET /api/v1/interaction-info/?drug1={drug1}&drug2={drug2}
```

Returns detailed interaction information including:
- Severity classification
- Mechanism of interaction
- Clinical effects
- Management recommendations
- Literature references

---

## 🧠 How It Works - High Level

### The Clinical Problem

Drug-drug interactions (DDIs) are a major cause of adverse drug events, hospitalizations, and deaths. Clinicians need fast, reliable tools to check for potential interactions when prescribing multiple medications.

### Our Solution

Project Aegis uses **Natural Language Processing (NLP)** to understand drug interactions the same way humans do - by reading and comprehending biomedical literature.

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Clinician enters two drugs: "Warfarin" + "Aspirin"         │
│                          │                                       │
│                          ▼                                       │
│   2. System generates context sentence:                          │
│      "The concomitant use of Warfarin with Aspirin may          │
│       result in enhanced anticoagulant effects..."              │
│                          │                                       │
│                          ▼                                       │
│   3. PubMedBERT analyzes the sentence:                          │
│      - Tokenizes text into subwords                              │
│      - Passes through 12 transformer layers                      │
│      - Outputs class probabilities                               │
│                          │                                       │
│                          ▼                                       │
│   4. System returns prediction:                                  │
│      - Type: "effect" (describes what happens)                   │
│      - Severity: "severe" (based on class + confidence)          │
│      - Risk Score: 0.85                                          │
│      - Recommendation: "Avoid combination"                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight

Unlike rule-based systems that rely on static databases, our NLP approach can:
- **Generalize** to drug pairs not explicitly in the training data
- **Explain** interactions in natural language
- **Understand context** from how interactions are described in literature

---

## 🔬 How It Works - Technical Deep Dive

### 1. Input Processing

When a user submits two drug names:

```python
# Input
drug1 = "Warfarin"
drug2 = "Aspirin"
```

### 2. Drug Name Normalization

The system normalizes drug names to handle:
- **Stereoisomers**: `(R)-Warfarin` → `Warfarin`
- **Salt forms**: `Warfarin Sodium` → `Warfarin`
- **Brand names**: `Coumadin` → `Warfarin`

```python
def normalize_drug_name(name: str) -> str:
    # Remove stereochemistry: (R)-, (S)-, (+)-, (-)-, etc.
    # Remove salts: sodium, hydrochloride, sulfate, etc.
    # Map brand to generic: Coumadin → Warfarin
    return normalized_name
```

### 3. Context Sentence Generation

The model was trained on sentences from biomedical literature. To make predictions, we generate context sentences that match the training distribution:

```python
CONTEXT_TEMPLATES = {
    'effect': [
        "The concomitant use of {drug1} with {drug2} may result in "
        "enhanced pharmacological effects and increased clinical risk.",
    ],
    'mechanism': [
        "{drug1} is a known inhibitor of CYP3A4, which may significantly "
        "increase plasma concentrations of {drug2}.",
    ],
    'advise': [
        "When {drug1} is co-administered with {drug2}, close monitoring "
        "of therapeutic response and adverse effects is recommended.",
    ],
    'neutral': [
        "The patient is taking {drug1} and {drug2} concurrently."
    ]
}
```

### 4. Tokenization

The PubMedBERT tokenizer converts text to model input:

```python
# Example tokenization
text = "Warfarin and Aspirin may increase bleeding risk."

tokens = tokenizer(text, return_tensors="pt", max_length=128, padding=True)

# Output:
# {
#   'input_ids': tensor([[  101,  1937,  7277, 12083, 1998,  7358, 14536, ...  102]]),
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, ... 1]])
# }
```

### 5. Model Inference

The fine-tuned BERT model processes the tokens:

```python
class BertForDDI(BertForSequenceClassification):
    """
    Architecture:
    - 12 transformer encoder layers
    - 768 hidden dimensions
    - 12 attention heads
    - 5-class classification head
    """
    
    def forward(self, input_ids, attention_mask):
        # Pass through BERT encoder
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        pooled = outputs.pooler_output  # Shape: (batch, 768)
        
        # Classification head
        logits = self.classifier(pooled)  # Shape: (batch, 5)
        
        return logits
```

### 6. Probability Calibration

Raw model outputs are calibrated using temperature scaling:

```python
def calibrate_probabilities(logits, temperature=1.5):
    """
    Temperature scaling prevents overconfident predictions.
    T > 1 softens probabilities (more conservative)
    T < 1 sharpens probabilities (more confident)
    """
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return probabilities
```

### 7. Severity Mapping

The predicted class and confidence are mapped to clinical severity:

```python
LABEL_TO_SEVERITY = {
    'no_interaction': ('none', 0.0),
    'int': ('moderate', 0.4),       # Generic interaction
    'advise': ('moderate', 0.6),    # Monitoring needed
    'effect': ('major', 0.75),      # Adverse effects
    'mechanism': ('severe', 0.85),  # CYP450 inhibition
}

def get_severity(predicted_class, confidence):
    base_severity, base_risk = LABEL_TO_SEVERITY[predicted_class]
    
    # Adjust based on confidence
    if confidence > 0.9:
        risk_score = min(base_risk + 0.1, 1.0)
    else:
        risk_score = base_risk * confidence
    
    return severity_from_risk(risk_score)
```

### 8. Response Generation

The final response includes all relevant information:

```python
response = {
    "drug1": "Warfarin",
    "drug2": "Aspirin",
    "interaction_type": "effect",      # From model
    "severity": "severe",              # Mapped from type + confidence
    "confidence": 0.89,                # Model confidence
    "risk_score": 0.85,                # Calibrated risk
    "mechanism": generate_mechanism(drug1, drug2, interaction_type),
    "clinical_recommendation": get_recommendation(severity),
    "probabilities": all_class_probs,
    "context_sentence": template_used,
    "context_source": "template"
}
```

---

## 💡 Usage Examples

### Example 1: Simple DDI Check

```bash
# Check Warfarin + Aspirin interaction
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"drug1": "Warfarin", "drug2": "Aspirin"}'
```

**Result**: Severe interaction - both drugs affect blood clotting.

### Example 2: Multi-Drug Comparison

```bash
# Compare a patient's medication list
curl -X POST http://localhost:8000/api/v1/compare/ \
  -H "Content-Type: application/json" \
  -d '{"drugs": ["Metformin", "Lisinopril", "Atorvastatin", "Metoprolol"]}'
```

**Result**: Returns a 4x4 interaction matrix showing all pairwise risks.

### Example 3: Finding Alternatives

```bash
# Find safer alternatives when interaction detected
curl "http://localhost:8000/api/v1/alternatives/?drug1=Warfarin&drug2=Ibuprofen"
```

**Result**: Suggests Acetaminophen as a safer pain reliever with Warfarin.

### Example 4: Python SDK Usage

```python
import requests

class AegisClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def predict(self, drug1: str, drug2: str) -> dict:
        response = requests.post(
            f"{self.base_url}/predict/",
            json={"drug1": drug1, "drug2": drug2}
        )
        return response.json()
    
    def compare(self, drugs: list) -> dict:
        response = requests.post(
            f"{self.base_url}/compare/",
            json={"drugs": drugs}
        )
        return response.json()

# Usage
client = AegisClient()

# Single prediction
result = client.predict("Warfarin", "Aspirin")
print(f"Severity: {result['severity']}")  # "severe"

# Multi-drug comparison
comparison = client.compare(["Warfarin", "Aspirin", "Ibuprofen"])
print(f"Total interactions: {comparison['total_interactions']}")
```

---

## 🗄️ Database & Data Pipeline

### Data Sources

| Source | Description | Records |
|--------|-------------|---------|
| DrugBank | Drug properties, structures | 2,088 drugs |
| DDIExtraction 2013 | Training sentences | ~19,000 sentences |
| Custom Mappings | Brand/generic names | 200+ mappings |

### Database Schema

```sql
-- Drug table
CREATE TABLE ddi_api_drug (
    id INTEGER PRIMARY KEY,
    drugbank_id VARCHAR(20) UNIQUE,
    name VARCHAR(255) NOT NULL,
    therapeutic_class VARCHAR(100),
    molecular_formula VARCHAR(100),
    molecular_weight DECIMAL(10,4),
    smiles TEXT,
    description TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Interaction table
CREATE TABLE ddi_api_drugdruginteraction (
    id INTEGER PRIMARY KEY,
    drug1_id INTEGER REFERENCES ddi_api_drug(id),
    drug2_id INTEGER REFERENCES ddi_api_drug(id),
    severity VARCHAR(20),
    mechanism TEXT,
    clinical_effect TEXT,
    management TEXT,
    source VARCHAR(50),
    confidence DECIMAL(5,4)
);
```

### Data Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw Data   │────▶│    Parser    │────▶│   Database   │
│  (XML/JSON)  │     │  (Python)    │     │   (SQLite)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       │                    ▼                     │
       │           ┌──────────────┐               │
       │           │  Tokenizer   │               │
       │           │ (PubMedBERT) │               │
       │           └──────────────┘               │
       │                    │                     │
       ▼                    ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ DDI Corpus   │────▶│   Tensors    │────▶│   Training   │
│  Sentences   │     │  (PyTorch)   │     │    Loop      │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 🤖 The AI Model

### Model Details

| Attribute | Value |
|-----------|-------|
| **Base Model** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| **Parameters** | ~110 million |
| **Fine-tuning Task** | 5-class sequence classification |
| **Training Data** | DDIExtraction 2013 corpus |
| **Training Epochs** | 5 |
| **Batch Size** | 16 |
| **Learning Rate** | 2e-5 |
| **Max Sequence Length** | 128 tokens |

### Classification Labels

| Label | Description | Example |
|-------|-------------|---------|
| `mechanism` | HOW drugs interact | "CYP3A4 inhibition increases drug levels" |
| `effect` | WHAT happens clinically | "Increased risk of bleeding" |
| `advise` | Clinical guidance | "Monitor blood pressure closely" |
| `int` | Generic interaction | "These drugs interact" |
| `no_interaction` | No known interaction | "No clinically significant interaction" |

### Model Files

```
DDI_Model_Final/
├── config.json              # Model configuration
├── model.safetensors        # Weights (safetensors format)
├── tokenizer.json           # Fast tokenizer
├── tokenizer_config.json    # Tokenizer settings
├── vocab.txt                # BERT vocabulary
├── special_tokens_map.json  # Special token mappings
└── added_tokens.json        # Custom tokens
```

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 87.3% |
| **Macro F1** | 89.6% |
| **Precision** | 88.2% |
| **Recall** | 91.1% |

---

## 🎨 Frontend Interface

### Dashboard Views

#### 1. Analysis Mode
- Drug search with autocomplete
- Real-time DDI prediction
- Severity indicator with gauge visualization
- Detailed interaction information panel

#### 2. Compare Mode
- Multi-drug selection (2-5 drugs)
- Side-by-side drug cards with properties
- NxN interaction matrix
- Color-coded severity indicators
- Detected interactions list

#### 3. Stats Mode
- Database statistics overview
- Coverage metrics
- Therapeutic class breakdown
- Severity distribution charts

### FUI (Futuristic UI) Theme

The interface uses a custom design system:

```css
/* Color Palette */
--fui-accent-cyan: #00d4ff;
--fui-accent-green: #00ff88;
--fui-accent-purple: #a855f7;
--fui-accent-orange: #f97316;
--fui-accent-red: #ef4444;
--fui-gray-900: #0a0a0f;
--fui-gray-800: #12121a;
--fui-gray-500: #4a4a5a;

/* Typography */
.fui-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Borders */
.fui-border {
  border: 1px solid rgba(74, 74, 90, 0.3);
  border-radius: 0; /* Sharp corners */
}
```

---

## ☁️ GCP Deployment

### Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD PLATFORM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    GKE CLUSTER                           │   │
│   │                                                          │   │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────┐           │   │
│   │   │  Backend  │  │  Backend  │  │  Backend  │           │   │
│   │   │  Pod (1)  │  │  Pod (2)  │  │  Pod (3)  │           │   │
│   │   └───────────┘  └───────────┘  └───────────┘           │   │
│   │                                                          │   │
│   │   ┌───────────────────────────────────────────┐         │   │
│   │   │          GPU Node Pool (T4)               │         │   │
│   │   │          Scale-to-Zero enabled            │         │   │
│   │   └───────────────────────────────────────────┘         │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   CLOUD LOAD BALANCER                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│   │  Cloud SQL  │  │    GCS      │  │   Memorystore       │    │
│   │ (Postgres)  │  │  (Buckets)  │  │   (Redis)           │    │
│   └─────────────┘  └─────────────┘  └─────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Optimization

| Service | Production Cost | Optimized Cost | Savings |
|---------|-----------------|----------------|---------|
| GKE Cluster | $224/mo | $5.71/mo | 97% |
| AI Model Serving | $450/mo | $4.56/mo | 99% |
| Cloud SQL | $125/mo | $2.52/mo | 98% |
| **Total** | **$807/mo** | **~$13/mo** | **98%** |

### Deployment Steps

```bash
# 1. Set up GCP project
gcloud config set project aegis-ddi-project

# 2. Create GKE cluster
gcloud container clusters create aegis-cluster \
  --zone us-central1-a \
  --machine-type e2-medium \
  --num-nodes 2

# 3. Create GPU node pool (scale-to-zero)
gcloud container node-pools create gpu-pool \
  --cluster aegis-cluster \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4 \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 1

# 4. Deploy application
kubectl apply -f k8s/

# 5. Verify deployment
kubectl get pods
kubectl get services
```

---

## 🔮 Future Roadmap

### Phase 2: Enhanced Predictions (Q2 2026)

- [ ] **RAG Integration**: Retrieve real-time PubMed articles for context
- [ ] **Confidence Calibration**: Implement temperature scaling
- [ ] **Multi-language Support**: Support for non-English drug names
- [ ] **Batch Prediction API**: Process multiple drug pairs in one request

### Phase 3: Clinical Integration (Q3 2026)

- [ ] **EHR Integration**: FHIR-compliant API for hospital systems
- [ ] **Alert System**: Configurable severity thresholds and notifications
- [ ] **Audit Logging**: Complete prediction history for compliance
- [ ] **Role-Based Access**: Different views for clinicians vs pharmacists

### Phase 4: Advanced Features (Q4 2026)

- [ ] **Polypharmacy Analysis**: Handle 10+ concurrent medications
- [ ] **Patient Context**: Age, weight, renal/hepatic function factors
- [ ] **Drug-Food Interactions**: Extend to food and supplement interactions
- [ ] **Mobile App**: iOS/Android applications for point-of-care use

### Phase 5: Scale & Reliability (2027)

- [ ] **Multi-Region Deployment**: Global availability
- [ ] **99.9% SLA**: High-availability architecture
- [ ] **Real-Time Updates**: Continuous model updates from new literature
- [ ] **FDA Compliance**: Pursue FDA clearance as clinical decision support

---

## ❓ Frequently Asked Questions

### General Questions

**Q: What is the accuracy of the DDI predictions?**

A: Our fine-tuned PubMedBERT model achieves approximately 90% F1-score on the DDI Corpus test set. However, predictions should always be verified by a qualified healthcare professional.

**Q: Can I use this in a clinical setting?**

A: Project Aegis is currently a research prototype. It is NOT FDA-approved and should NOT be used as the sole basis for clinical decisions. Always consult authoritative drug interaction databases and clinical pharmacists.

**Q: What drugs are in the database?**

A: The database contains over 2,000 drugs from DrugBank, including common prescription medications, OTC drugs, and some supplements. Coverage is strongest for commonly prescribed medications.

**Q: How does this differ from existing DDI checkers?**

A: Unlike rule-based systems that rely on manually curated databases, our NLP approach can generalize to drug pairs not explicitly in the training data by understanding the language patterns of drug interactions.

### Technical Questions

**Q: What hardware is required to run the model?**

A: The model can run on CPU with ~4GB RAM, but GPU acceleration (CUDA) significantly improves response times. In production, we recommend at least 8GB RAM and a GPU with 4GB+ VRAM.

**Q: How long does a prediction take?**

A: On CPU: ~500-1000ms. On GPU (T4): ~50-100ms. Response times include database lookups and response formatting.

**Q: Can I fine-tune the model further?**

A: Yes! The model can be fine-tuned on additional data. See the `training/` directory for training scripts and documentation.

**Q: How do I add new drugs to the database?**

A: New drugs can be added via the Django admin interface or by running the data ingestion scripts in `web/ddi_api/services/data_ingestion.py`.

### Troubleshooting

**Q: The model predictions seem wrong. What should I check?**

A: 
1. Verify the drug names are spelled correctly
2. Check if the drugs exist in the database
3. Review the context sentence used for prediction
4. Consider that some drug pairs may have limited training data

**Q: Docker containers won't start. What do I do?**

A: 
1. Ensure Docker Desktop is running
2. Check available disk space (model requires ~500MB)
3. Verify port 80 and 8000 are not in use
4. Run `docker-compose logs` to see error messages

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Failure

```
Error: Unable to load model from /app/DDI_Model_Final
```

**Solution**: Ensure the model directory is mounted correctly in docker-compose.yml:
```yaml
volumes:
  - ../DDI_Model_Final:/app/DDI_Model_Final:ro
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or switch to CPU inference by setting:
```python
DEVICE = "cpu"
```

#### 3. Database Migration Errors

```
django.db.utils.OperationalError: no such table: ddi_api_drug
```

**Solution**: Run migrations:
```bash
docker-compose exec backend python manage.py migrate
```

#### 4. Frontend Not Loading

```
502 Bad Gateway
```

**Solution**: Ensure backend is running and healthy:
```bash
docker-compose logs backend
docker-compose restart backend
```

#### 5. Slow Predictions

**Symptoms**: Predictions taking >2 seconds

**Solutions**:
1. Enable Redis caching
2. Use GPU inference
3. Reduce max sequence length
4. Pre-load model on startup

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/project-aegis.git
cd project-aegis/molecular-ai

# Install frontend dependencies
npm install

# Start frontend dev server
npm run dev

# In another terminal, start backend
cd web
pip install -r requirements.txt
python manage.py runserver
```

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **JavaScript/React**: Follow ESLint configuration
- **Commits**: Use conventional commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team & Acknowledgments

### Project Team

| Role | Member | Responsibilities |
|------|--------|------------------|
| Data Engineer | Student A (K.C.) | Literature review, dataset preprocessing, XML parsing |
| ML Engineer | Student B (T.G.) | Model development, fine-tuning, architecture design |
| DevOps/Infra | Student C (A.K.) | GCP setup, Docker, Kubernetes, CI/CD |
| QA/Documentation | Student D (R.N.) | Testing, evaluation metrics, documentation |

### Acknowledgments

- **PubMedBERT**: Microsoft Research for the pre-trained biomedical language model
- **DDI Corpus**: Universidad Carlos III de Madrid for the annotated dataset
- **DrugBank**: For comprehensive drug information
- **Google Cloud Platform**: For cloud infrastructure and credits

---

## 📚 References

1. Gu, Y., et al. (2020). "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing." *arXiv preprint arXiv:2007.15779*.

2. Herrero-Zazo, M., et al. (2013). "The DDI corpus: An annotated corpus with pharmacological substances and drug-drug interactions." *Journal of Biomedical Informatics*, 46(5), 914-920.

3. Wishart, D.S., et al. (2018). "DrugBank 5.0: a major update to the DrugBank database for 2018." *Nucleic Acids Research*, 46(D1), D1074-D1082.

---

<div align="center">

**Project Aegis** - Making Drug Safety Smarter

[Report Bug](https://github.com/ahrazkk/ProjectAegis-Clinical-MLOps-Platform/issues) 
[Request Feature](https://github.com/ahrazkk/ProjectAegis-Clinical-MLOps-Platform/issues)

</div>
