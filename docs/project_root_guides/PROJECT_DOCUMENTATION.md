# Project Aegis: Clinical AI Platform Whitepaper

## Executive Summary
**Project Aegis** is an advanced AI platform designed to predict and visualize **Drug-Drug Interactions (DDIs)**. Unlike traditional symptom checkers, Aegis uses state-of-the-art NLP (PubMedBERT) to understand *mechanisms* from medical literature and uses 3D molecular rendering to visualize them.

The goal is to move beyond simple "list checking" and create a **visual, intelligent feedback system** for researchers and clinicians.

---

## 1. How It Works (End-to-End Workflow)

### Step 1: User Input
- **Action**: User navigates to Dashboard and selects 2 drugs (e.g., *Warfarin* and *Aspirin*).
- **Technology**: React Frontend with Glassmorphism UI.
- **Search Optimization**: Currently searches a small local cache. *To make it faster/bigger, we need a vector database (see Section 4).*

### Step 2: System Processing
1.  **Check Knowledge Graph**: The backend first checks Neo4j (Graph DB) for a *known* clinical fact. "Do we already know these interact?"
2.  **AI Prediction (The "Brain")**: If no known fact exists, the AI models kick in.
    *   **Primary Model (NLP)**: **PubMedBERT** (fine-tuned on 19k sentences). It reads a generated clinical sentence about the drugs.
    *   **Secondary Model (GNN)**: Graph Neural Network. It looks at the *molecular structure* (SMILES) to predict chemical reactivity.

### Step 3: Visualization & Result
- **Result**: The system returns a Risk Score (0-100%), Severity Level (High/Medium/Low), and Mechanism.
- **Visuals**:
    - **3D Viewer**: WebGL renders the actual drug molecules.
    - **Knowledge Graph**: A dynamic network shows how the drugs connect to protein targets (mechanisms like COX-1 or enzymes like CYP3A4).
    - **Body Map**: Highlights organs at risk (e.g., Stomach/Liver).

---

## 2. The Artificial Intelligence (AI) Stack

### Q: "Are we using PubMedBERT or GNN?"
**Answer: Both, but PubMedBERT is the leader.**

1.  **PubMedBERT (Text-Based)**
    *   **Why?** Doctors trust *literature*. PubMedBERT was pre-trained on millions of biomedical abstracts. It "reads" text to find interactions.
    *   **Why no sentence input?** To make the tool easy to use, we **auto-generate** the context.
        *   *User sees:* Selects "Aspirin" + "Warfarin".
        *   *System Generates:* "The co-administration of Aspirin and Warfarin may lead to adverse effects..."
        *   *Model Reads:* The sentence and predicts "Bleeding Risk" (Mechanism).
    *   **Improvement**: We *could* allow advanced users to input specific patient notes (e.g., "Patient is 75, on Warfarin...") for custom predictions.

2.  **GNN (Graph Neural Network)**
    *   **Role**: Fallback. If we don't have text data, we look at the raw atoms. "Do these chemical structures clash?"

### Q: "Is our prediction correct?"
**Status**: It is decent but limited by training data.
- **Accuracy**: ~85-90% on the DDI Corpus.
- **Problem**: It only knows what it was trained on (5k sentences).
- **Solution**: Connect real databases (DrugBank) to validate its guesses.

---

## 3. Current Status: What is Implemented?

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Visual Interface** | ✅ **100% Done** | Premium "Apple-style" dark mode, 3D/2D viewers, Risk Gauge. |
| **Frontend Logic** | ✅ **100% Done** | React, State management, Dynamic Knowledge Graph. |
| **Backend API** | ⚠️ **80% Done** | Django is setup, but running on "Mock Data" or limited caches often. |
| **AI Models** | ⚠️ **70% Done** | Models exist but need massive real-world data injection. |
| **Database** | ❌ **Missing** | No rich SQL/Neo4j database connected yet. |
| **Deployment** | ❌ **Missing** | Running on localhost only. |

---

## 4. How to Make It "A Lot Better" (Optimization Roadmap)

### A. Faster Search & More Drgus
*   **Problem**: Currently searches a small hardcoded list or basic DB.
*   **Solution**: Implement **Redis** or **ElasticSearch**.
    *   *Result*: Instant autocomplete for 100,000+ drugs.
*   **Data Source**: Import **DrugBank** and **ChEMBL** (open databases).

### B. Custom Testing & "Samples"
*   **Idea**: Pre-set "Test Scenarios".
*   **Implementation**: Create a "Senarios" dropdown in the UI.
    *   *Scenario 1*: "The Cardiac Patient" (Warfarin + Amiodarone).
    *   *Scenario 2*: "pain Management" (Ibuprofen + Aspirin).
    *   User clicks -> System auto-loads drugs + patient context -> AI predicts.

### C. Improving the AI (Overhaul)
1.  **Fine-tuning**: Train PubMedBERT on **TwoSides** (dataset of 4 million drug pairs).
2.  **RAG (Retrieval Augmented Generation)**:
    *   Instead of *guessing*, let the AI **search PubMed in real-time** (using API) and summarize the latest 2025 papers on the interaction.

---

## 5. Future Systems & Deployment

### What Pages are Left?
1.  **Login/Auth**: To save user history.
2.  **History/Logs**: "What did I search last Tuesday?"
3.  **Detailed Report**: A PDF export page for doctors.

### Hosting (GCP/AWS)
*   **Why GCP?** Google Cloud Platform is great for AI/Containers.
*   **How?**
    1.  **Dockerize**: Package Frontend, Backend, and AI Model into containers.
    2.  **Cloud Run**: Deploy containers to serverless URL (auto-scales).
    3.  **SQL**: Managed Cloud SQL for the drug database.

---

## 6. Project Value: "What is this accomplishing?"

1.  **Educational Tool**: Visualizing *invisible* chemical interactions for students.
2.  **Safety Check**: A "Second Opinion" for doctors. Humans miss things; AI catches patterns.
3.  **Research Accelerator**: finding *new* interactions that aren't in textbooks yet (using the GNN).

## Summary
The **Visuals are done and stunning**. The **Engine (AI)** is built but needs "Fuel" (Data). The next step is **Data Integration** (DrugBank) and **Deployment**.
