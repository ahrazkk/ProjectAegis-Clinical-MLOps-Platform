# Project Aegis - DDI Prediction Platform
## Guidelines & Architecture Documentation (Pre-MCR1)

> **Document Purpose:** Complete technical reference for anyone reviewing this project. Explains what was built, why, and where it's headed.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Architecture](#current-architecture)
3. [Technology Stack](#technology-stack)
4. [AI/ML Pipeline](#aiml-pipeline)
5. [Database Architecture (Polyglot Persistence)](#database-architecture)
6. [What We Changed & Why](#what-we-changed--why)
7. [Current Limitations](#current-limitations)
8. [Cloud Deployment Strategy](#cloud-deployment-strategy)
9. [Future Roadmap](#future-roadmap)
10. [Optimistic Best-Case Scenario](#optimistic-best-case-scenario)
11. [FAQ / Key Questions](#faq--key-questions)

---

## Executive Summary

**Project Aegis** is a Drug-Drug Interaction (DDI) prediction platform that demonstrates:
- Deep learning on biomedical text (PubMedBERT)
- Graph-based molecular analysis (GNN architecture)
- Polyglot persistence (Neo4j + Redis + SQLite)
- Containerized microservices (Docker)
- Modern React frontend with glassmorphic UI

**Current Status:** MVP complete, running locally via Docker. AI predictions functional but model accuracy needs improvement with clinical data.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER NETWORK                                      │
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   FRONTEND       │    │    BACKEND       │    │    DATABASES     │          │
│  │   (Nginx)        │───▶│    (Django)      │───▶│                  │          │
│  │   Port 80        │    │    Port 8000     │    │  ┌────────────┐  │          │
│  │                  │    │                  │    │  │   Neo4j    │  │          │
│  │  React + Vite    │    │  Gunicorn        │    │  │  Port 7475 │  │          │
│  │  TailwindCSS     │    │  REST API        │    │  └────────────┘  │          │
│  │  Three.js        │    │  PubMedBERT      │    │  ┌────────────┐  │          │
│  │  Framer Motion   │    │  GNN (disabled)  │    │  │   Redis    │  │          │
│  └──────────────────┘    │  RDKit           │    │  │  Port 6379 │  │          │
│                          └──────────────────┘    │  └────────────┘  │          │
│                                                   │  ┌────────────┐  │          │
│                                                   │  │  SQLite    │  │          │
│                                                   │  │  (Django)  │  │          │
│                                                   │  └────────────┘  │          │
│                                                   └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Frontend
| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **React 18** | UI Framework | Industry standard, component-based |
| **Vite** | Build tool | Fast HMR, modern ESM bundling |
| **TailwindCSS** | Styling | Utility-first, rapid prototyping |
| **Three.js** | 3D Molecules | WebGL-based 3D visualization |
| **Framer Motion** | Animations | Declarative, React-native animations |
| **Lucide Icons** | Icon library | Lightweight, consistent iconography |

### Backend
| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Django 5.2** | Web framework | Robust, batteries-included Python framework |
| **Django REST Framework** | API layer | Standardized REST endpoints |
| **Gunicorn** | WSGI server | Production-ready Python server |
| **Transformers (HuggingFace)** | PubMedBERT | State-of-the-art NLP models |
| **PyTorch** | Deep learning | Flexible, research-friendly ML framework |
| **RDKit** | Cheminformatics | Molecular fingerprints, SMILES parsing |
| **Neo4j Driver** | Graph queries | Cypher query execution |
| **django-redis** | Cache backend | High-speed session/cache storage |

### Infrastructure
| Technology | Purpose | Why Chosen |
|------------|---------|------------|
| **Docker** | Containerization | Reproducible environments |
| **Docker Compose** | Orchestration | Multi-container coordination |
| **Nginx** | Reverse proxy | Static file serving, API proxying |
| **Neo4j** | Graph database | Relationship-first DDI storage |
| **Redis** | Cache | Low-latency data retrieval |

---

## AI/ML Pipeline

### Current Flow (What Actually Runs)

```
User Input: "Aspirin" + "Warfarin"
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 1: Drug Resolution                                        │
│ ┌─────────┐   ┌─────────────┐   ┌────────┐                    │
│ │ Neo4j   │ → │ JSON DB     │ → │ Django │                    │
│ │ (empty) │   │ (50+ drugs) │   │ ORM    │                    │
│ │   ❌    │   │     ✅      │   │        │                    │
│ └─────────┘   └─────────────┘   └────────┘                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 2: Known Interaction Check                                │
│ ┌─────────────────────────────────────────┐                   │
│ │ Neo4j: MATCH (Drug)-[INTERACTS]-(Drug)  │ → (empty, skip)   │
│ └─────────────────────────────────────────┘                   │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ STEP 3: AI Prediction (Fallback Chain)                         │
│                                                                │
│ ┌────────────────┐                                            │
│ │  PubMedBERT    │  ← CURRENTLY ACTIVE                        │
│ │  (Text-based)  │                                            │
│ │  DDI_Model_Final                                             │
│ │  ~10 second inference                                        │
│ └────────────────┘                                            │
│         │                                                      │
│         ▼ (if confidence < threshold)                         │
│ ┌────────────────┐                                            │
│ │  GNN Model     │  ← DISABLED (PyTorch Geometric missing)    │
│ │  (Structure)   │                                            │
│ └────────────────┘                                            │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Response: { risk_score: 0.19, severity: "low", mechanism: "..." }
```

### Model Comparison

| Model | Input Type | Strengths | Weaknesses | Status |
|-------|------------|-----------|------------|--------|
| **PubMedBERT** | Drug names (text) | Literature-grounded, interpretable | Can't handle novel drugs without papers | ✅ Active |
| **GNN** | SMILES (molecular graph) | Structure-aware, novel drug prediction | Requires heavy dependencies | ❌ Disabled |
| **Neo4j Lookup** | Known pairs | Instant (1ms), 100% accurate for known | Can't predict unknown pairs | ⚠️ Empty |

---

## Database Architecture

### Polyglot Persistence Strategy

**Why Multiple Databases?**
Different data types benefit from different storage paradigms.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER ARCHITECTURE                      │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │     NEO4J       │  │     REDIS       │  │     SQLITE      │ │
│  │   (Graph DB)    │  │   (Cache)       │  │   (Relational)  │ │
│  │                 │  │                 │  │                 │ │
│  │ • Drug nodes    │  │ • Session data  │  │ • User accounts │ │
│  │ • Interactions  │  │ • API cache     │  │ • Predictions   │ │
│  │ • Pathways      │  │ • Search cache  │  │ • Audit logs    │ │
│  │ • Targets       │  │                 │  │                 │ │
│  │                 │  │                 │  │                 │ │
│  │ Latency: 1-5ms  │  │ Latency: <1ms   │  │ Latency: 5-20ms │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Current State

| Database | Status | Data Loaded | Used For |
|----------|--------|-------------|----------|
| **Neo4j** | Running | ❌ Empty | (Intended for DDI knowledge graph) |
| **Redis** | Running | ✅ Active | Django cache backend |
| **SQLite** | Running | ✅ Active | Django ORM, user data |
| **JSON DB** | File | ✅ 50+ drugs | Drug info fallback |

---

## What We Changed & Why

### Change 1: Removed Blue Theme → Pure Black
**Before:** Dark blue-tinted backgrounds (#030712, #0b101b)
**After:** Pure black (#000000, #050505)
**Why:** User preference for more professional, modern aesthetic

### Change 2: Added Grid Pattern Overlay
**Before:** Solid dark background
**After:** Subtle 40px grid lines with gaussian splat-style ambient glows
**Why:** Requested "gaussian splat" aesthetic for premium feel

### Change 3: Expanded System Monitor
**Before:** 3 status badges (API, Neo4j, AI)
**After:** 7 status badges across Infrastructure and AI Models sections
**Why:** Better demonstrates the full pipeline to reviewers

### Change 4: Changed Neo4j Image
**Before:** `neo4j:5.11` (corrupted on user's system)
**After:** `neo4j:community` (fresh pull)
**Why:** `input/output error` caused by corrupted Docker image cache

### Change 5: Added Nginx Proxy for API
**Before:** Frontend called `http://127.0.0.1:8000` directly
**After:** Frontend calls `/api/v1` (relative), Nginx proxies to backend
**Why:** Docker networking requires container-to-container communication via service names

### Change 6: Volume Mount for AI Model
**Before:** Model not included in Docker image (0% risk predictions)
**After:** Model mounted as read-only volume from host
**Why:** Keeps Docker image small, allows model swapping without rebuild

---

## Current Limitations

| Limitation | Impact | Resolution |
|------------|--------|------------|
| **Neo4j Empty** | No known interactions, always falls back to AI | Load DrugBank/TwoSides data |
| **GNN Disabled** | No structure-based prediction | Add `torch-geometric` to requirements |
| **PubMedBERT Accuracy** | Aspirin+Warfarin shows 19% (should be high) | Fine-tune with clinical severity labels |
| **No Real-time Status** | System Monitor shows hardcoded statuses | Connect to actual health endpoints |
| **Single Worker** | Gunicorn runs 1 worker, can timeout | Add more workers, async endpoints |

---

## Cloud Deployment Strategy

### Target Architecture (GCP)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GOOGLE CLOUD PLATFORM                                   │
│                                                                                  │
│  ┌─────────────────┐                                                            │
│  │   Cloud CDN     │ ← Static assets (JS, CSS, images)                          │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐   │
│  │   Cloud Run     │         │   Cloud Run     │         │   Cloud SQL     │   │
│  │   (Frontend)    │────────▶│   (Backend)     │────────▶│   (PostgreSQL)  │   │
│  │   Nginx/Static  │         │   Django API    │         │                 │   │
│  └─────────────────┘         └────────┬────────┘         └─────────────────┘   │
│                                       │                                         │
│                              ┌────────┴────────┐                                │
│                              ▼                 ▼                                │
│                    ┌─────────────────┐  ┌─────────────────┐                    │
│                    │   Memorystore   │  │   Aura DB       │                    │
│                    │   (Redis)       │  │   (Neo4j Cloud) │                    │
│                    └─────────────────┘  └─────────────────┘                    │
│                                                                                  │
│  ┌─────────────────┐         ┌─────────────────┐                                │
│  │ Cloud Storage   │         │ Vertex AI       │                                │
│  │ (Model files)   │         │ (Model serving) │                                │
│  └─────────────────┘         └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Deployment Steps (When Ready)

1. **Build container images:** `docker build -t gcr.io/PROJECT/aegis-backend`
2. **Push to Container Registry:** `docker push gcr.io/PROJECT/aegis-backend`
3. **Deploy to Cloud Run:** `gcloud run deploy aegis-backend --image gcr.io/PROJECT/aegis-backend`
4. **Configure secrets:** API keys, database URLs via Secret Manager
5. **Set up Neo4j Aura:** Free tier, import drug data
6. **Configure Memorystore:** Redis cache for production

---

## Future Roadmap

### Phase 1: Data Quality (Priority: HIGH)
- [ ] Load DrugBank interaction data into Neo4j
- [ ] Add "known dangerous pairs" lookup table
- [ ] Improve PubMedBERT accuracy with clinical labels

### Phase 2: Model Enhancement (Priority: MEDIUM)
- [ ] Enable GNN model (add PyTorch Geometric)
- [ ] Implement ensemble scoring (PubMedBERT + GNN)
- [ ] Add model confidence calibration

### Phase 3: Production Readiness (Priority: MEDIUM)
- [ ] Deploy to GCP Cloud Run
- [ ] Add authentication (OAuth/JWT)
- [ ] Implement rate limiting
- [ ] Add monitoring/alerting (Cloud Monitoring)

### Phase 4: Features (Priority: LOW)
- [ ] PDF report generation
- [ ] Multi-language support
- [ ] Mobile-responsive design refinement
- [ ] Browser extension for quick lookups

---

## Optimistic Best-Case Scenario

**If everything goes perfectly, here's what Project Aegis becomes:**

### Year 1: Academic Success
- Full DrugBank integration (10,000+ interactions)
- GNN + PubMedBERT ensemble achieving 92% accuracy
- Published paper at biomedical informatics conference
- 1,000+ monthly active users from pharmacy schools

### Year 2: Clinical Pilot
- Partnership with hospital pharmacy department
- EHR integration (SMART on FHIR)
- Real-time alerting when prescriptions are entered
- FDA registration exploration

### Year 3: Commercial Product
- SaaS model for pharmacies and hospitals
- API access for EHR vendors
- Mobile app for pharmacists
- Regulatory approval for clinical decision support

### Technical Vision (Best Case)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PROJECT AEGIS - VISION 2027                                │
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   EHR SYSTEMS   │───▶│   AEGIS API     │───▶│   PHARMACIST    │             │
│  │   Epic, Cerner  │    │   (Real-time)   │    │   Mobile App    │             │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘             │
│                                  │                                               │
│                                  ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                        AI ENSEMBLE (99.2% Accuracy)                          ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           ││
│  │  │PubMed   │  │  GNN    │  │  Neo4j  │  │  LLM    │  │ Clinical│           ││
│  │  │BERT v3  │  │Molecular│  │Knowledge│  │ Explain │  │ Rules   │           ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘           ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  Data Sources: DrugBank, TwoSides, FDA FAERS, PubMed, Clinical Trials           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## FAQ / Key Questions

### Q: Why isn't Neo4j showing any data?
**A:** Neo4j starts empty. It needs a data ingestion script to load drug interaction data from sources like DrugBank or TwoSides.

### Q: Why does Aspirin + Warfarin show only 19% risk?
**A:** The PubMedBERT model was trained on DDI Corpus text classification, not clinical severity. It needs fine-tuning with severity labels, or a known-interactions lookup should override it.

### Q: Why is GNN disabled?
**A:** PyTorch Geometric requires CUDA-specific builds and adds significant complexity. It's a "nice to have" for novel drug prediction but not essential for the MVP.

### Q: Why use Docker?
**A:** Reproducibility. Anyone can run `docker-compose up` and get the exact same environment. Essential for collaboration and deployment.

### Q: Why Neo4j + Redis + SQLite? Isn't that overkill?
**A:** It's "Polyglot Persistence" - using the right database for the right job. Neo4j for relationships, Redis for speed, SQLite for relational data. Also validates the resume claim.

### Q: What's the biggest technical debt?
**A:** Hardcoded status badges in System Monitor. They show "online" but aren't connected to actual health checks. Should call `/api/v1/health/` endpoints.

### Q: Can this run without Docker?
**A:** Yes, but you'd need to install Python, Node.js, Neo4j, and Redis manually. Docker simplifies this to one command.

### Q: What would break if deployed to production today?
**A:** 1) SQLite doesn't scale (need PostgreSQL), 2) No authentication, 3) Single Gunicorn worker times out, 4) No HTTPS configured.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-22 | Claude/Gemini | Initial MCR1 pre-submission doc |

---

*Generated for Project Aegis MCR1 submission review.*
