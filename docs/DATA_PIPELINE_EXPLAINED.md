# Project Aegis: Complete Data Pipeline Documentation

> **Purpose**: This document explains exactly what happens when a user enters drug names into Project Aegis. It covers every step from user input to prediction output, including all databases, AI models, and future plans.

---

## 🎯 Quick Reference: What Data Source Answers Your Question?

| Your Question | Data Source Used | Notes |
|---------------|------------------|-------|
| "Do these drugs interact?" | **Neo4j** (checked FIRST) | If we have this pair stored, instant answer at 95% confidence |
| "Predict unknown interaction" | **PubMedBERT AI** | Uses DDI Sentence Database or templates for context |
| "What sentences mention this pair?" | **DDI Sentence Database** | ~19,000 curated sentences from DDI Corpus |
| "Get live research papers" | **⚠️ PubMed API (NOT CONNECTED)** | Code exists but is NOT integrated yet |
| "What drugs share enzymes?" | **Neo4j Graph Queries** | Traverses relationships between drugs |

### The Critical Decision Flow (Memorize This!)

```
User enters: "Warfarin" + "Aspirin"
                │
                ▼
    ┌───────────────────────┐
    │ 1. Check Neo4j        │
    │    "Known interaction?"│
    └───────────┬───────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
   FOUND               NOT FOUND
   (95% conf)          (Use AI)
       │                 │
       ▼                 ▼
   Return data    ┌─────────────────┐
   immediately    │ 2. Find Context │
                  │    for AI       │
                  └────────┬────────┘
                           │
                   ┌───────┴───────┐
                   ▼               ▼
              DDI Sentence    Templates
              Database        (fallback)
              (~19K sentences)
                   │               │
                   └───────┬───────┘
                           ▼
                  ┌─────────────────┐
                  │ 3. PubMedBERT   │
                  │    Prediction   │
                  └─────────────────┘
```

---

## 📚 Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [The Complete Journey: What Happens When You Search](#3-the-complete-journey-what-happens-when-you-search)
4. [Database Deep Dive](#4-database-deep-dive)
5. [AI Models Explained](#5-ai-models-explained)
6. [The RAG (Retrieval-Augmented Generation) System](#6-the-rag-system)
7. [API Endpoints](#7-api-endpoints)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [Current Limitations](#9-current-limitations)
10. [Future Improvements & Roadmap](#10-future-improvements--roadmap)
11. [Glossary](#11-glossary)

---

## 1. High-Level Overview

### What is Project Aegis?

Project Aegis is a **Drug-Drug Interaction (DDI) Prediction System** that helps healthcare professionals and patients understand if two or more medications might interact dangerously.

### The Simple Version

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │ --> │   Django    │ --> │   AI        │ --> │   Result    │
│   Input     │     │   Backend   │     │   Models    │     │   Display   │
│ "Warfarin   │     │   API       │     │   +         │     │   Risk:     │
│  + Aspirin" │     │             │     │   Databases │     │   SEVERE    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React + Three.js | User interface with 3D visualizations |
| **Backend API** | Django REST Framework | Handles requests, orchestrates predictions |
| **AI Model** | PubMedBERT (Transformer) | Classifies drug interactions from text |
| **Knowledge Graph** | Neo4j | Stores drug relationships and properties |
| **Cache** | Redis | Speeds up repeated queries |
| **Literature Search** | PubMed API | Retrieves real medical literature |

---

## 2. System Architecture Diagram

```
                                    ┌──────────────────────────────────────────┐
                                    │              USER INTERFACE               │
                                    │         (React + Vite + Three.js)         │
                                    │                                           │
                                    │  ┌─────────────────────────────────────┐ │
                                    │  │   Drug Search Box                   │ │
                                    │  │   "Warfarin" + "Aspirin"            │ │
                                    │  └─────────────────────────────────────┘ │
                                    └─────────────────┬─────────────────────────┘
                                                      │
                                                      ▼ HTTP POST
                                    ┌──────────────────────────────────────────┐
                                    │           NGINX REVERSE PROXY             │
                                    │              (Port 80)                    │
                                    └─────────────────┬─────────────────────────┘
                                                      │
                          ┌───────────────────────────┼───────────────────────────┐
                          │                           │                           │
                          ▼                           ▼                           ▼
              ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
              │   /api/v1/...     │      │   Static Files    │      │   Other Routes    │
              │   Django Backend  │      │   React Build     │      │                   │
              └─────────┬─────────┘      └───────────────────┘      └───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DJANGO REST FRAMEWORK                                        │
│                                   (Port 8000)                                            │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                              VIEWS.PY (API Layer)                                   │ │
│  │                                                                                     │ │
│  │   /api/v1/predict/              → DDIPredictionView                                 │ │
│  │   /api/v1/interaction-info/     → InteractionInfoView                               │ │
│  │   /api/v1/chat/                 → ChatbotView (GraphRAG)                            │ │
│  │   /api/v1/drugs/search/         → DrugSearchView                                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                               │
│                                          ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           SERVICES LAYER                                            │ │
│  │                                                                                     │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                  │ │
│  │  │ pubmedbert_      │  │ knowledge_       │  │ pubmed_          │                  │ │
│  │  │ predictor.py     │  │ graph.py         │  │ retriever.py     │                  │ │
│  │  │                  │  │                  │  │                  │                  │ │
│  │  │ • Load Model     │  │ • Neo4j Queries  │  │ • PubMed API     │                  │ │
│  │  │ • Classify DDI   │  │ • Drug Lookup    │  │ • Literature     │                  │ │
│  │  │ • Risk Score     │  │ • Interactions   │  │ • Sentences      │                  │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘                  │ │
│  │           │                     │                     │                            │ │
│  └───────────┼─────────────────────┼─────────────────────┼────────────────────────────┘ │
│              │                     │                     │                              │
└──────────────┼─────────────────────┼─────────────────────┼──────────────────────────────┘
               │                     │                     │
               ▼                     ▼                     ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   DDI_Model_Final/   │  │   NEO4J          │  │   NCBI PubMed        │
│   (PubMedBERT)       │  │   (Knowledge     │  │   E-utilities API    │
│                      │  │    Graph)        │  │                      │
│   • model.safetensors│  │   Port 7687      │  │   esearch + efetch   │
│   • tokenizer.json   │  │                  │  │                      │
│   • config.json      │  │   Nodes:         │  │   Rate Limited:      │
│                      │  │   - Drug         │  │   3 req/sec          │
│   5 Labels:          │  │   - Target       │  │                      │
│   - no_interaction   │  │   - Enzyme       │  │                      │
│   - mechanism        │  │   - SideEffect   │  │                      │
│   - effect           │  │                  │  │                      │
│   - advise           │  │   Edges:         │  │                      │
│   - int              │  │   - INTERACTS    │  │                      │
│                      │  │   - TARGETS      │  │                      │
│                      │  │   - METABOLIZED  │  │                      │
└──────────────────────┘  └──────────────────┘  └──────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │   REDIS          │
                          │   (Cache)        │
                          │   Port 6379      │
                          │                  │
                          │   Caches:        │
                          │   - PubMed calls │
                          │   - Drug lookups │
                          │   - Predictions  │
                          └──────────────────┘
```

---

## 3. The Complete Journey: What Happens When You Search

### Example: User searches "Warfarin" and "Aspirin"

Let's follow this query through every step of the system.

---

### Step 1: User Input (Frontend)

```javascript
// User types in the Dashboard.jsx drug search boxes
Drug 1: "Warfarin"
Drug 2: "Aspirin"

// Frontend sends HTTP request
POST /api/v1/predict/
Body: {
  "drug1": "Warfarin",
  "drug2": "Aspirin"
}
```

---

### Step 2: API Receives Request (views.py)

```python
# views.py - DDIPredictionView
class DDIPredictionView(APIView):
    def post(self, request):
        drug1 = request.data.get('drug1')  # "Warfarin"
        drug2 = request.data.get('drug2')  # "Aspirin"
        
        # Step 2a: Normalize drug names
        normalized_drug1 = normalize_drug_name(drug1)
        # Handles: "Coumadin" → "Warfarin", "warfarin sodium" → "warfarin"
```

**What is Drug Name Normalization?**

The system handles many variations of drug names:

| User Input | Normalized To | Why? |
|------------|---------------|------|
| `Coumadin` | `warfarin` | Brand name → Generic |
| `Warfarin Sodium` | `warfarin` | Remove salt form |
| `(R)-Warfarin` | `warfarin` | Remove stereochemistry |
| `WARFARIN` | `warfarin` | Case normalization |
| `Tylenol` | `acetaminophen` | Brand → Generic |

```python
# Brand name mappings (partial list)
brand_to_generic = {
    'tylenol': 'acetaminophen',
    'advil': 'ibuprofen',
    'coumadin': 'warfarin',
    'lipitor': 'atorvastatin',
    'zocor': 'simvastatin',
    # ... 50+ more mappings
}
```

---

### Step 3: Knowledge Graph Lookup (knowledge_graph.py)

The system queries Neo4j to find drug information:

```cypher
// Cypher Query to find drug
MATCH (d:Drug)
WHERE toLower(d.name) CONTAINS toLower($name)
RETURN d.drugbank_id as id,
       d.name as name,
       d.smiles as smiles,
       d.category as category
LIMIT 1
```

**What we get from Neo4j:**

```json
{
  "id": "DB00682",
  "name": "Warfarin",
  "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
  "category": "Anticoagulant",
  "targets": ["CYP2C9", "VKORC1"],
  "enzymes": ["CYP2C9", "CYP3A4", "CYP1A2"]
}
```

**Check for Known Interactions:**

```cypher
// Query for existing interaction data
MATCH (d1:Drug {drugbank_id: $drug1_id})-[r:INTERACTS_WITH]-(d2:Drug {drugbank_id: $drug2_id})
RETURN r.severity as severity,
       r.mechanism as mechanism,
       r.affected_systems as affected_systems
```

---

### Step 4: The Decision Point - Where Does Data Come From?

**🔑 THIS IS THE CRITICAL DECISION POINT**

After looking up the drugs, the system decides which path to take:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECISION FLOW DIAGRAM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User enters: "Warfarin" + "Aspirin"                                        │
│                        │                                                     │
│                        ▼                                                     │
│   ┌────────────────────────────────────────┐                                │
│   │  STEP 1: Neo4j Lookup                  │                                │
│   │  "Do we ALREADY know this interaction?"│                                │
│   └────────────────────┬───────────────────┘                                │
│                        │                                                     │
│            ┌───────────┴───────────┐                                        │
│            ▼                       ▼                                        │
│   ┌─────────────────┐    ┌─────────────────┐                                │
│   │   YES - Found   │    │   NO - Unknown  │                                │
│   │   in Neo4j!     │    │   drug pair     │                                │
│   └────────┬────────┘    └────────┬────────┘                                │
│            │                      │                                          │
│            ▼                      ▼                                          │
│   ┌─────────────────┐    ┌─────────────────────────────────┐                │
│   │ Return stored   │    │  STEP 2: Use PubMedBERT AI      │                │
│   │ severity &      │    │  to PREDICT the interaction     │                │
│   │ mechanism       │    └────────────────┬────────────────┘                │
│   │                 │                     │                                  │
│   │ Confidence: 95% │                     ▼                                  │
│   │ Source: Neo4j   │    ┌─────────────────────────────────┐                │
│   └─────────────────┘    │  WHERE DOES CONTEXT COME FROM?  │                │
│                          │                                  │                │
│                          │  Try in order:                   │                │
│                          │  1. DDI Sentence Database ✓      │                │
│                          │     (~19,000 curated sentences)  │                │
│                          │                                  │                │
│                          │  2. Templates (fallback) ✓       │                │
│                          │     "The concomitant use of..."  │                │
│                          │                                  │                │
│                          │  ⚠️ PubMed API is NOT currently │                │
│                          │     integrated! (future work)    │                │
│                          └─────────────────────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Understanding Each Data Source

| Data Source | What It Contains | When Used | Confidence |
|-------------|------------------|-----------|------------|
| **Neo4j Knowledge Graph** | Pre-loaded known interactions with severity & mechanism | Checked FIRST for every query | 95% (trusted curated data) |
| **DDI Sentence Database** | ~19,000 real sentences from DDI Corpus (training data) | If Neo4j has no interaction | High (real medical text) |
| **Context Templates** | Generic phrases like "may interact" | Fallback if no real sentence | Lower (generic) |
| **PubMed API** | Live medical literature search | ⚠️ **NOT CURRENTLY USED** | N/A (future feature) |

### Path A: Known Interaction (Neo4j Has It)

```python
# views.py - What happens when Neo4j knows the interaction
known_interaction = get_interaction_from_kg(drug_a, drug_b)

if known_interaction:
    # USE NEO4J DATA DIRECTLY - No AI needed!
    return {
        'severity': known_interaction['severity'],      # From Neo4j
        'mechanism': known_interaction['mechanism'],    # From Neo4j
        'confidence': 0.95,  # High because it's curated data
        'source': 'knowledge_graph'
    }
```

**Result:** Fast, high confidence, uses pre-stored data.

### Path B: Unknown Interaction (Need AI Prediction)

```python
# views.py - What happens when Neo4j doesn't know
else:
    # Use PubMedBERT to PREDICT the interaction
    prediction = pubmedbert.predict(drug_a['name'], drug_b['name'])
    # Note: No context is passed! The predictor finds its own context.
```

**Inside the predictor, context is found from:**

```python
# pubmedbert_predictor.py - Context discovery
def predict(self, drug1, drug2, context=None):
    
    # Priority 1: Check DDI Sentence Database for real sentences
    if DDI_SENTENCE_DB_AVAILABLE:
        sentence = ddi_db.find_sentence(drug1, drug2)
        if sentence:
            # Found a real medical sentence!
            context = sentence.sentence
            # Example: "Warfarin plasma levels may be increased by..."
    
    # Priority 2: Fallback to templates
    if not context:
        context = "The concomitant use of {drug1} with {drug2} may result in enhanced effects."
    
    # Now classify with this context
    formatted = f"<e1>{drug1}</e1> and <e2>{drug2}</e2>. {context}"
    return self.model.classify(formatted)
```

---

### ⚠️ Important: PubMed API Status

**The `pubmed_retriever.py` file exists but is NOT connected to the prediction flow!**

The code was built for future RAG (Retrieval-Augmented Generation) but is not currently integrated:

```python
# This code EXISTS but is NOT CALLED during predictions
class PubMedRetriever:
    def search(self, drug1: str, drug2: str) -> List[RetrievedContext]:
        # Would search PubMed for: "Warfarin" AND "Aspirin" AND "interaction"
        # Would return real-time sentences from medical literature
        # BUT... this is never called from views.py!
        pass
```

**Future Plan:** Connect PubMed retrieval → pass to PubMedBERT → true RAG system

---

### Step 5: AI Prediction (pubmedbert_predictor.py)

When Neo4j doesn't have a known interaction, the AI model predicts it.

**The Full Context Discovery + Prediction Flow:**

```python
class PubMedBERTPredictor:
    def predict(self, drug1: str, drug2: str, context: str = None):
        
        # ===============================================================
        # STEP 5a: Find Context (if not provided)
        # ===============================================================
        
        if context is None:
            # Try DDI Sentence Database first (best quality)
            if DDI_SENTENCE_DB_AVAILABLE:
                ddi_sentence = ddi_db.find_sentence(drug1, drug2)
                if ddi_sentence:
                    context = ddi_sentence.sentence
                    # Example: "Aspirin inhibits platelet aggregation and 
                    #          may displace warfarin from protein binding."
            
            # Fallback to templates
            if context is None:
                context = self.CONTEXT_TEMPLATES['effect'][0].format(
                    drug1=drug1, drug2=drug2
                )
                # Example: "The concomitant use of Warfarin with Aspirin 
                #          may result in enhanced pharmacological effects."
        
        # ===============================================================
        # STEP 5b: Format input with entity markers
        # ===============================================================
        
        formatted_input = f"<e1>{drug1}</e1> and <e2>{drug2}</e2>. {context}"
        # Result: "<e1>Warfarin</e1> and <e2>Aspirin</e2>. Aspirin inhibits..."
        
        # ===============================================================
        # STEP 5c: Tokenize for BERT
        # ===============================================================
        
        tokens = self.tokenizer(
            formatted_input,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        # ===============================================================
        # STEP 5d: Run through PubMedBERT model
        # ===============================================================
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            probabilities = F.softmax(outputs.logits, dim=-1)
        
        # ===============================================================
        # STEP 5e: Get prediction class and confidence
        # ===============================================================
        
        predicted_class = probabilities.argmax().item()  # e.g., "mechanism"
        confidence = probabilities.max().item()          # e.g., 0.85
```

**Visual Flow:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI PREDICTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: drug1="Warfarin", drug2="Aspirin", context=None                     │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────┐                  │
│  │ 1. CONTEXT DISCOVERY                                   │                  │
│  │                                                        │                  │
│  │    DDI Sentence DB ──► Found? ──► Use real sentence   │                  │
│  │           │                                            │                  │
│  │           ▼ Not found                                  │                  │
│  │    Templates ──► Generate: "The concomitant use..."   │                  │
│  └───────────────────────────────────────────────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────┐                  │
│  │ 2. FORMAT WITH ENTITY MARKERS                          │                  │
│  │                                                        │                  │
│  │    "<e1>Warfarin</e1> and <e2>Aspirin</e2>. {context}"│                  │
│  └───────────────────────────────────────────────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────┐                  │
│  │ 3. TOKENIZE                                            │                  │
│  │                                                        │                  │
│  │    Text → [101, 1026, 1041, 1028, 2227, ...]          │                  │
│  └───────────────────────────────────────────────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────┐                  │
│  │ 4. PUBMEDBERT INFERENCE                                │                  │
│  │                                                        │                  │
│  │    ┌──────────────────────────────┐                   │                  │
│  │    │     12-Layer BERT Encoder    │                   │                  │
│  │    │     (768 hidden dimensions)  │                   │                  │
│  │    └──────────────┬───────────────┘                   │                  │
│  │                   │                                    │                  │
│  │                   ▼                                    │                  │
│  │    ┌──────────────────────────────┐                   │                  │
│  │    │   Classification Head (5)    │                   │                  │
│  │    └──────────────────────────────┘                   │                  │
│  └───────────────────────────────────────────────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────┐                  │
│  │ 5. OUTPUT PROBABILITIES                                │                  │
│  │                                                        │                  │
│  │    no_interaction: 0.02                                │                  │
│  │    int:            0.05                                │                  │
│  │    advise:         0.08                                │                  │
│  │    effect:         0.35                                │                  │
│  │    mechanism:      0.50  <-- PREDICTED CLASS          │                  │
│  └───────────────────────────────────────────────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The 5 Interaction Classes:**

| Class | Meaning | Risk Level | Example |
|-------|---------|------------|---------|
| `no_interaction` | No known interaction | None (0.0) | Tylenol + Vitamin D |
| `int` | Generic interaction | Moderate (0.4) | Some effect noted |
| `advise` | Clinical guidance needed | Moderate (0.6) | Monitor patient closely |
| `effect` | Adverse effect described | Major (0.75) | Causes bleeding |
| `mechanism` | Explains HOW interaction works | Severe (0.85) | Inhibits CYP450 enzyme |

**For Warfarin + Aspirin:**

```python
# Model output
all_probabilities = {
    'no_interaction': 0.02,
    'int': 0.05,
    'advise': 0.08,
    'effect': 0.35,      # High - describes adverse effects
    'mechanism': 0.50    # Highest - explains CYP interaction
}

predicted_class = 'mechanism'
confidence = 0.50
risk_score = 0.85  # Mapped from severity
severity = 'severe'
```

---

### Step 6: Response Assembly (views.py)

The API assembles all the information:

```python
response = {
    "drug1": {
        "name": "Warfarin",
        "drugbank_id": "DB00682",
        "category": "Anticoagulant"
    },
    "drug2": {
        "name": "Aspirin", 
        "drugbank_id": "DB00945",
        "category": "NSAID"
    },
    "interaction": {
        "exists": True,
        "severity": "severe",
        "risk_score": 0.85,
        "confidence": 0.50,
        "interaction_type": "mechanism",
        "mechanism": "Aspirin inhibits platelet aggregation and may displace warfarin from protein binding sites, significantly increasing bleeding risk",
        "affected_systems": ["cardiovascular", "hematologic", "gastrointestinal"],
        "clinical_advice": "Avoid combination if possible. If necessary, use lowest effective aspirin dose and monitor INR closely."
    },
    "sources": [
        {
            "type": "pubmed",
            "pmid": "38291045",
            "sentence": "Concurrent use of warfarin and aspirin..."
        }
    ]
}
```

---

### Step 7: Frontend Display (Dashboard.jsx)

```jsx
// The frontend receives the response and displays it
<RiskGauge severity="severe" score={0.85} />

<div className="interaction-details">
  <h3>⚠️ SEVERE Interaction Detected</h3>
  <p><strong>Risk Score:</strong> 85%</p>
  <p><strong>Type:</strong> Pharmacokinetic (Mechanism)</p>
  <p><strong>Affected Systems:</strong> Cardiovascular, Blood, GI</p>
  <p><strong>Recommendation:</strong> Avoid if possible...</p>
</div>
```

---

## 4. Database Deep Dive

### 4.1 Neo4j Knowledge Graph

**What is Neo4j?**
Neo4j is a graph database - instead of tables with rows and columns, it stores data as **nodes** (things) connected by **relationships** (edges).

```
         ┌───────────────┐
         │   Drug:       │
         │   Warfarin    │
         │   DB00682     │
         └───────┬───────┘
                 │
         ┌───────┴───────┐
         │ INTERACTS_WITH│
         │ severity:     │
         │ "severe"      │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │   Drug:       │
         │   Aspirin     │
         │   DB00945     │
         └───────────────┘
```

**Node Types (Labels):**

| Node Type | Description | Properties |
|-----------|-------------|------------|
| `Drug` | A medication | name, drugbank_id, smiles, category, molecular_weight |
| `Target` | Protein the drug binds to | uniprot_id, name, gene_name |
| `Enzyme` | Metabolizes the drug | uniprot_id, name (CYP2C9, CYP3A4, etc.) |
| `SideEffect` | Adverse effects | umls_id, name |

**Relationship Types (Edges):**

| Relationship | Meaning | Properties |
|--------------|---------|------------|
| `INTERACTS_WITH` | Two drugs interact | severity, mechanism, affected_systems |
| `TARGETS` | Drug binds to protein | action_type (inhibitor, agonist, etc.) |
| `METABOLIZED_BY` | Drug processed by enzyme | - |
| `CAUSES` | Drug causes side effect | frequency |

**Example Cypher Query - Find All Drugs That Interact with Warfarin:**

```cypher
MATCH (warfarin:Drug {name: "Warfarin"})-[i:INTERACTS_WITH]-(other:Drug)
RETURN other.name as drug, 
       i.severity as severity,
       i.mechanism as mechanism
ORDER BY 
  CASE i.severity 
    WHEN 'severe' THEN 1 
    WHEN 'major' THEN 2 
    WHEN 'moderate' THEN 3 
    ELSE 4 
  END
```

---

### 4.2 SQLite Database (Django Models)

For local development and as a backup, we use Django's ORM with SQLite:

```python
# models.py

class Drug(models.Model):
    drugbank_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255)
    smiles = models.TextField(blank=True)  # Molecular structure
    molecular_weight = models.FloatField(null=True)
    drug_class = models.CharField(max_length=255)

class DrugDrugInteraction(models.Model):
    drug_a = models.ForeignKey(Drug, related_name='interactions_as_a')
    drug_b = models.ForeignKey(Drug, related_name='interactions_as_b')
    severity = models.CharField(choices=[
        ('minor', 'Minor'),
        ('moderate', 'Moderate'),
        ('major', 'Major'),
        ('contraindicated', 'Contraindicated')
    ])
    mechanism = models.TextField()
    
class PredictionLog(models.Model):
    # Logs every prediction for analytics
    drug_a_name = models.CharField(max_length=255)
    drug_b_name = models.CharField(max_length=255)
    predicted_severity = models.CharField(max_length=50)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
```

---

### 4.3 Redis Cache

Redis stores frequently accessed data to speed up responses:

```python
# What we cache:

# 1. PubMed API responses (expensive external calls)
cache_key = f"pubmed:{drug1}:{drug2}"
cache.set(cache_key, sentences, ttl=86400)  # 24 hours

# 2. Drug lookups
cache_key = f"drug:{normalized_name}"
cache.set(cache_key, drug_info, ttl=3600)  # 1 hour

# 3. Prediction results
cache_key = f"prediction:{drug1}:{drug2}"
cache.set(cache_key, prediction, ttl=1800)  # 30 minutes
```

---

### 4.4 JSON Drug Database (Fallback)

For simplicity, we also maintain a JSON file with drug data:

```json
// web/data/drug_db.json
[
  {
    "drugbank_id": "DB00682",
    "name": "Warfarin",
    "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "category": "Anticoagulant",
    "molecular_weight": 308.33
  },
  {
    "drugbank_id": "DB00945",
    "name": "Aspirin",
    "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "category": "NSAID",
    "molecular_weight": 180.16
  }
  // ... more drugs
]
```

---

## 5. AI Models Explained

### 5.1 PubMedBERT: The Brain of Our System

**What is BERT?**
BERT (Bidirectional Encoder Representations from Transformers) is a language model that understands context by reading text in both directions.

**What is PubMedBERT?**
PubMedBERT is BERT pre-trained on 14 million PubMed abstracts (medical literature). It understands medical terminology better than generic BERT.

**Our Fine-Tuning:**
We further trained PubMedBERT on the **DDI Corpus** - a dataset of ~19,000 sentences about drug interactions, labeled by experts.

```
Training Data Example:
─────────────────────
Input: "@DRUG$Warfarin@DRUG$ plasma levels may be increased by @DRUG$Fluconazole@DRUG$ 
        through CYP2C9 inhibition."
Label: mechanism

Input: "Patients should avoid taking @DRUG$Aspirin@DRUG$ with @DRUG$Warfarin@DRUG$."
Label: advise

Input: "No interaction was found between @DRUG$Vitamin D@DRUG$ and @DRUG$Calcium@DRUG$."
Label: no_interaction
```

**Model Architecture:**

```
Input Text
    │
    ▼
┌────────────────────────────┐
│   Tokenizer                │
│   (WordPiece, 30k vocab)   │
│                            │
│   "Warfarin" → [101, 2227, │
│                 4558, 102] │
└────────────────────────────┘
    │
    ▼
┌────────────────────────────┐
│   BERT Encoder             │
│   (12 layers, 768 hidden)  │
│                            │
│   Self-Attention +         │
│   Feed-Forward Networks    │
└────────────────────────────┘
    │
    ▼
┌────────────────────────────┐
│   Classification Head      │
│   (Linear: 768 → 5)        │
│                            │
│   Softmax → Probabilities  │
└────────────────────────────┘
    │
    ▼
[0.02, 0.05, 0.08, 0.35, 0.50]
  ↑     ↑     ↑     ↑     ↑
 no_  int  advise effect mech
 int
```

**Model Files:**

| File | Purpose | Size |
|------|---------|------|
| `model.safetensors` | Model weights | ~440 MB |
| `config.json` | Architecture config | 1 KB |
| `tokenizer.json` | Vocabulary | 711 KB |
| `vocab.txt` | Word list | 232 KB |

---

### 5.2 Why We Chose Text-Based Over Molecular Structure

We considered two approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **Molecular (GNN)** | Uses actual chemistry, can predict novel interactions | Needs SMILES structures, black-box predictions |
| **Text (PubMedBERT)** | Interpretable, leverages medical literature | Only knows what's been published |

**We chose PubMedBERT because:**
1. **Interpretability**: We can explain WHY two drugs interact (mechanism)
2. **Clinical Relevance**: Predictions align with how doctors think
3. **Data Availability**: Medical literature is abundant
4. **Speed**: Text classification is faster than graph neural networks

---

## 6. The RAG System

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of relying only on what the model memorized during training, we **retrieve** relevant information at query time and **augment** the model's input with it.

```
Traditional Approach:
────────────────────
User Query → Model → Answer
              ↑
        (only training data)

RAG Approach:
────────────────────
User Query → Retriever → Retrieved Context
                              ↓
                    Model + Context → Better Answer
                              ↑
                    (real-time information)
```

---

### ⚠️ IMPORTANT: Current State vs. Future Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HONEST ASSESSMENT OF RAG STATUS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CURRENT STATE (What we actually do):                                        │
│  ─────────────────────────────────────                                       │
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐           │
│  │    Neo4j       │     │  DDI Sentence  │     │   Templates    │           │
│  │   (Primary)    │ --> │   Database     │ --> │   (Fallback)   │           │
│  │                │     │                │     │                │           │
│  │ Known drugs &  │     │ ~19,000 real   │     │ Generic        │           │
│  │ interactions   │     │ sentences from │     │ phrases like   │           │
│  │ we've loaded   │     │ DDI Corpus     │     │ "may interact" │           │
│  └────────────────┘     └────────────────┘     └────────────────┘           │
│          ↓                      ↓                      ↓                    │
│     If found: DONE        Context for AI          Last resort               │
│     (no AI needed)                                                          │
│                                                                              │
│  ⚠️ PubMed API exists in code but is NOT connected!                        │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FUTURE VISION (What we plan to build):                                      │
│  ──────────────────────────────────────                                      │
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐           │
│  │   Neo4j KG     │     │   PubMed API   │     │   DrugBank     │           │
│  │   (Facts)      │     │   (Real-time)  │     │   (Details)    │           │
│  └───────┬────────┘     └───────┬────────┘     └───────┬────────┘           │
│          │                      │                      │                    │
│          └──────────────────────┼──────────────────────┘                    │
│                                 ▼                                            │
│                    ┌────────────────────────┐                               │
│                    │   Combined Context     │                               │
│                    │   + Vector Similarity  │                               │
│                    └───────────┬────────────┘                               │
│                                │                                             │
│                                ▼                                             │
│                    ┌────────────────────────┐                               │
│                    │   LLM (GPT-4/Claude)   │                               │
│                    │   Natural Language     │                               │
│                    │   Explanations         │                               │
│                    └────────────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### What We Currently Have (Simplified RAG)

```
Step 1: Query Understanding
───────────────────────────
User: "What happens if I take Warfarin with Aspirin?"

Extracted: drug1="Warfarin", drug2="Aspirin"


Step 2: Check Neo4j First
─────────────────────────
┌─────────────────┐
│   Neo4j KG      │ → Do we have this interaction stored?
└─────────────────┘
        │
        ├── YES → Return stored data (95% confidence)
        │
        └── NO → Continue to Step 3


Step 3: Find Context for AI
───────────────────────────
┌─────────────────┐
│   DDI Sentence  │ → Try to find a real sentence about these drugs
│   Database      │
└─────────────────┘
        │
        ├── Found → Use real medical sentence
        │
        └── Not Found → Use template: "The concomitant use of..."


Step 4: Model Inference
───────────────────────
PubMedBERT receives:
"<e1>Warfarin</e1> and <e2>Aspirin</e2>. [Context from Step 3]"

→ Predicts: "mechanism" with 0.50 confidence


Step 5: Response Generation
───────────────────────────
Final Response includes:
- Prediction (mechanism → severe)
- Confidence score
- Context source (ddi_corpus or template)
- Mechanism description (generated from type)
```

---

### Current vs. Future RAG Comparison

| Aspect | Current State | Future Plan |
|--------|---------------|-------------|
| **Knowledge Graph** | Neo4j ✅ (working) | Neo4j + more data |
| **Real-time Search** | ❌ PubMed NOT connected | PubMed + DrugBank APIs |
| **Context Source** | DDI Sentence DB + Templates | Multiple sources merged |
| **Classifier** | PubMedBERT (5 classes) | Multi-task + severity |
| **Generator** | Template-based explanations | GPT-4/Claude natural language |
| **Vector Search** | ❌ Not implemented | Pinecone/Weaviate embeddings |

---

## 7. API Endpoints

### Main Endpoints

```
BASE URL: http://localhost:8000/api/v1/
```

#### 1. Drug-Drug Interaction Prediction

```http
POST /api/v1/predict/
Content-Type: application/json

{
  "drug1": "Warfarin",
  "drug2": "Aspirin"
}

Response:
{
  "interaction": {
    "severity": "severe",
    "risk_score": 0.85,
    "confidence": 0.50,
    "interaction_type": "mechanism",
    "mechanism": "...",
    "affected_systems": ["cardiovascular", "hematologic"]
  }
}
```

#### 2. Polypharmacy Analysis (Multiple Drugs)

```http
POST /api/v1/polypharmacy/
Content-Type: application/json

{
  "drugs": ["Warfarin", "Aspirin", "Ibuprofen", "Omeprazole"]
}

Response:
{
  "interactions": [
    { "drug_pair": ["Warfarin", "Aspirin"], "severity": "severe" },
    { "drug_pair": ["Warfarin", "Ibuprofen"], "severity": "severe" },
    { "drug_pair": ["Warfarin", "Omeprazole"], "severity": "moderate" }
  ],
  "overall_risk": "high",
  "recommendations": [...]
}
```

#### 3. Drug Search

```http
GET /api/v1/drugs/search/?q=warfa

Response:
{
  "results": [
    { "name": "Warfarin", "drugbank_id": "DB00682", "category": "Anticoagulant" }
  ]
}
```

#### 4. Chat (GraphRAG)

```http
POST /api/v1/chat/
Content-Type: application/json

{
  "message": "Why is it dangerous to take blood thinners with NSAIDs?",
  "context_drugs": ["Warfarin"]
}

Response:
{
  "response": "Blood thinners like Warfarin work by inhibiting...",
  "sources": [...],
  "related_drugs": ["Aspirin", "Ibuprofen", "Naproxen"]
}
```

---

## 8. Data Flow Diagrams

### Prediction Flow (Sequence Diagram)

```
User          Frontend       Django API      PubMedBERT      Neo4j        PubMed
  │               │               │               │            │            │
  │──"Warfarin    │               │               │            │            │
  │  + Aspirin"──>│               │               │            │            │
  │               │──POST ───────>│               │            │            │
  │               │               │               │            │            │
  │               │               │──Lookup ────────────────────────────────>│
  │               │               │<─Drug Info ──────────────────────────────│
  │               │               │               │            │            │
  │               │               │──Query ─────────────────────>│            │
  │               │               │<─Graph Data ─────────────────│            │
  │               │               │               │            │            │
  │               │               │──Search ──────────────────────────────────>│
  │               │               │<─Sentences ────────────────────────────────│
  │               │               │               │            │            │
  │               │               │──Classify ─────>│            │            │
  │               │               │<─Prediction ────│            │            │
  │               │               │               │            │            │
  │               │<─JSON ────────│               │            │            │
  │<─Display ─────│               │               │            │            │
  │               │               │               │            │            │
```

### Data Sources Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA SOURCES                               │
├──────────────────┬──────────────────┬────────────────────┬───────────────────┤
│   DrugBank       │   PubMed         │   SIDER            │   TWOSIDES        │
│   (Structures,   │   (Literature)   │   (Side Effects)   │   (Interactions)  │
│    Properties)   │                  │                    │                   │
└────────┬─────────┴────────┬─────────┴──────────┬─────────┴─────────┬─────────┘
         │                  │                    │                   │
         ▼                  ▼                    ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                  │
│                         (data_ingestion.py)                                   │
│                                                                               │
│   • Download datasets    • Parse formats (CSV, XML, JSON)                     │
│   • Clean & normalize    • Deduplicate entries                                │
│   • Map identifiers      • Validate data quality                              │
└──────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                                         │
├──────────────────────────┬────────────────────────┬──────────────────────────┤
│       Neo4j              │       SQLite           │       Redis              │
│   Knowledge Graph        │   Django Models        │   Cache                  │
│                          │                        │                          │
│   - Drug nodes           │   - Drug table         │   - Query cache          │
│   - Interaction edges    │   - Interaction table  │   - API response cache   │
│   - Target nodes         │   - PredictionLog      │   - PubMed cache         │
└──────────────────────────┴────────────────────────┴──────────────────────────┘
```

---

## 9. Current Limitations

### What We Can't Do (Yet)

| Limitation | Impact | Planned Solution |
|------------|--------|------------------|
| **Limited Drug Database** | ~500 drugs vs. ~10,000+ FDA approved | Import full DrugBank dataset |
| **English Only** | Can't process non-English literature | Multilingual model fine-tuning |
| **No Dosage Consideration** | "50mg" vs "100mg" treated the same | Add dosage as model input |
| **Binary Predictions** | "Interacts" vs "Doesn't" only | Multi-level severity scores |
| **No Patient Context** | Age, weight, genetics not considered | Personalized medicine module |
| **Template Dependency** | Predictions vary with input template | Ensemble of templates |

### Data Coverage Gaps

```
Current Coverage:
─────────────────
Drugs in database:        ~500
Known interactions:       ~2,000
Coverage of FDA drugs:    ~5%

Target Coverage:
─────────────────
FDA approved drugs:       ~10,000
DrugBank entries:         ~15,000
Known interactions:       ~400,000
```

---

## 10. Future Improvements & Roadmap

### Phase 1: Data Enhancement (Q1-Q2)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ENHANCEMENT ROADMAP                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CURRENT STATE              IMPROVED STATE                      │
│   ─────────────              ──────────────                      │
│                                                                  │
│   500 Drugs          ───►    15,000+ Drugs                      │
│   (Manual curation)          (Full DrugBank import)              │
│                                                                  │
│   2,000 Interactions ───►    400,000+ Interactions              │
│   (Sample data)              (DrugBank + TWOSIDES + FDA)         │
│                                                                  │
│   5 Side Effects     ───►    5,000+ Side Effects                │
│   (Basic)                    (Full SIDER integration)            │
│                                                                  │
│   0 Genetic Markers  ───►    PharmGKB Integration               │
│                              (Pharmacogenomics)                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**How to Get More Data:**

1. **DrugBank Academic License** (Free for research)
   - 15,000 drugs with structures, targets, interactions
   - Download: https://go.drugbank.com/releases/latest

2. **TWOSIDES Database** (Public)
   - 400,000+ drug-drug-side effect relationships
   - Mined from FDA Adverse Event Reporting System

3. **FDA Orange Book** (Public)
   - All FDA-approved drugs and generics
   - https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files

4. **PubChem** (Public)
   - Chemical structures and bioactivity data
   - API: https://pubchem.ncbi.nlm.nih.gov/rest/pug/

### Phase 2: Enhanced AI (Q2-Q3)

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI ROADMAP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CURRENT                                                        │
│   ───────                                                        │
│   PubMedBERT Classification                                      │
│   ↓                                                              │
│   5 categories (mechanism, effect, advise, int, no_interaction)  │
│                                                                  │
│   PHASE 2A: Multi-Task Learning                                  │
│   ────────────────────────────                                   │
│   • Severity prediction (continuous 0-1)                         │
│   • Interaction type (pharmacokinetic vs pharmacodynamic)        │
│   • Affected organ prediction (multi-label)                      │
│                                                                  │
│   PHASE 2B: Generative AI Integration                            │
│   ───────────────────────────────────                            │
│   • GPT-4 / Claude for explanation generation                    │
│   • Natural language clinical recommendations                    │
│   • Patient-friendly summaries                                   │
│                                                                  │
│   PHASE 2C: Molecular Hybrid Model                               │
│   ────────────────────────────────                               │
│   • Combine PubMedBERT (text) + GNN (structure)                  │
│   • Predict novel interactions not in literature                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Advanced RAG (Q3-Q4)

```
Current RAG:                    Advanced RAG:
────────────                    ─────────────

Keyword Search                  Dense Retrieval
    ↓                               ↓
PubMed API                      Vector Database (Pinecone/Weaviate)
    ↓                               ↓
Sentence Extraction             Semantic Search
    ↓                               ↓
Template Classification         LLM Generation

                                + Agentic RAG:
                                ─────────────
                                AI Agent that:
                                • Decides what to search
                                • Queries multiple sources
                                • Synthesizes findings
                                • Generates citations
```

### Phase 4: Clinical Integration (Q4+)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CLINICAL DECISION SUPPORT                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   EHR Integration (Epic, Cerner)                                            │
│   ─────────────────────────────                                             │
│   • FHIR API compatibility                                                  │
│   • Real-time alerts in clinical workflow                                   │
│   • Automatic medication list import                                        │
│                                                                             │
│   Pharmacist Dashboard                                                      │
│   ────────────────────                                                      │
│   • Bulk medication review                                                  │
│   • Patient-specific risk reports                                           │
│   • Audit trail for compliance                                              │
│                                                                             │
│   Patient Portal                                                            │
│   ──────────────                                                            │
│   • Simple interaction checker                                              │
│   • OTC + Prescription combined analysis                                    │
│   • Supplement interactions (St. John's Wort, etc.)                         │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **DDI** | Drug-Drug Interaction - when one drug affects how another works |
| **SMILES** | Simplified Molecular Input Line Entry System - text representation of molecular structure |
| **Neo4j** | A graph database that stores data as nodes and relationships |
| **Cypher** | Query language for Neo4j (like SQL for graph databases) |
| **PubMedBERT** | BERT language model pre-trained on medical literature |
| **RAG** | Retrieval-Augmented Generation - enhancing AI with retrieved information |
| **CYP450** | Cytochrome P450 enzymes - metabolize most drugs in the liver |
| **Pharmacokinetics** | How the body affects a drug (absorption, metabolism, etc.) |
| **Pharmacodynamics** | How a drug affects the body (mechanism of action) |
| **FHIR** | Fast Healthcare Interoperability Resources - healthcare data standard |
| **DrugBank ID** | Unique identifier for drugs (e.g., DB00682 for Warfarin) |
| **PMID** | PubMed ID - unique identifier for research papers |

---

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────────┐
│                    PROJECT AEGIS QUICK REFERENCE                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SERVICES:                                                          │
│  ─────────                                                          │
│  Frontend:    http://localhost:80     (React)                       │
│  Backend:     http://localhost:8000   (Django)                      │
│  Neo4j:       http://localhost:7475   (Browser)                     │
│               bolt://localhost:7688   (Driver)                      │
│  Redis:       localhost:6379                                        │
│                                                                     │
│  KEY API ENDPOINTS:                                                 │
│  ─────────────────                                                  │
│  POST /api/v1/predict/           Predict DDI                        │
│  POST /api/v1/polypharmacy/      Multiple drugs                     │
│  GET  /api/v1/drugs/search/      Search drugs                       │
│  POST /api/v1/chat/              GraphRAG chatbot                   │
│                                                                     │
│  DOCKER COMMANDS:                                                   │
│  ───────────────                                                    │
│  docker-compose up --build -d    Start all services                 │
│  docker-compose down             Stop all services                  │
│  docker logs aegis-backend       View backend logs                  │
│                                                                     │
│  DATA FILES:                                                        │
│  ──────────                                                         │
│  DDI_Model_Final/                PubMedBERT model                   │
│  web/data/drug_db.json           Drug database (JSON)               │
│  web/db.sqlite3                  Django database                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-23 | Aegis Team | Initial comprehensive documentation |

---

*This document is designed for NotebookLM import and slideshow generation. Each section is self-contained for easy extraction into presentation slides.*
