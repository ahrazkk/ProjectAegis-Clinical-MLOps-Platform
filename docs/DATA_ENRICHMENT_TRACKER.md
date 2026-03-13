# Data Enrichment Tracker & Plan

## Current Status
**Date:** March 13, 2026
**Goal:** Enrich the Neo4j drug database to convert it into a heterogeneous graph, enabling GNN models to make significantly better Drug-Drug Interaction (DDI) predictions.

### Existing Database Schema Overview
*   **Target Node:** `Drug`
*   **Existing Properties Found:** 
    *   `drugbank_id`
    *   `name`
    *   `smiles` (Some nodes have this, some don't)
    *   `molecular_weight`
    *   `molecular_formula`
    *   `iupac_name`
    *   `pubchem_cid`
    *   `category`
    *   `therapeutic_class`
    *   `mechanism`
    *   `description`
    *   `updated_at`
*   **Existing Relationships:**
    *   `INTERACTS_WITH` (between Drugs)
    *   `AFFECTS_SYSTEM` 
    *   `INTERACTS_WITH_TARGET`

---

## Planned Phases for Data Enrichment

### Phase 1: Chemical Completeness (Immediate Action)
*   **Goal:** Ensure every drug has fundamental chemical descriptors. 
*   **Rule:** **DO NOT DUPLICATE COLUMNS.** We will only update nodes where `smiles` is `NULL` or missing.
*   **Data to Fetch (from PubChem API):**
    *   `smiles`
    *   `molecular_weight` 
    *   `molecular_formula`
    *   `iupac_name`
    *   `pubchem_cid`

### Phase 2: Biological & Therapeutic Classification 
*   **Goal:** Add functional groups to help the GNN understand *what* the drug does.
*   **Data to Fetch/Connect:**
    *   `therapeutic_class` (e.g., "Anticoagulant", "NSAID")
    *   `mechanism` (What pathway it uses)
    *   These should ideally eventually become their own *Nodes* (e.g., `(Drug)-[:BELONGS_TO_CLASS]->(Class)`) because Graph Neural Networks learn better using structural connections rather than pure text properties.

### Phase 3: Advanced Precision Data (Future Scope)
*   *Note: Currently tabled until core graph is stable to avoid overwhelming the model/database.*
*   **Demographics:** How it interacts differently based on genetic markers, age groups, or race.
*   **Dosage Thresholds:** LD50 (Lethal Dose), max daily doses.
*   **Contraindications:** Interactions based on pre-existing user conditions (e.g., Liver damage + Medicine X).

---

## Change Log
- **[March 13, 2026]:** Successfully connected to Neo4j Aura Database. Analyzed existing node properties to prevent duplicates. Prepared Phase 1 scripting. 
- **[March 13, 2026]:** Enriched 1,350 valid molecular drugs with chemical properties (`pubchem_cid`, `molecular_weight`, `molecular_formula`) from the PubChem API. Left "class" nodes (e.g., "Statins") untouched.
- **[March 13, 2026]:** Launched Phase 2 scripts in background: 
  1. `enrich_interactions.py` calling NIH RxNav to scrape thousands of new precise drug-drug interaction edges.
  2. `enrich_classes_enhanced.py` running in background to label all drugs with biological/therapeutic properties.
- **[March 13, 2026]:** Wrote `extract_graph_dataset.py` to seamlessly port the enriched Neo4j nodes (Chemical Fingerprints + Class One-hot Encodings) into a PyTorch Geometric (PyG) Tensor framework.
- **[March 13, 2026]:** Shifted GNN Architecture from microscopic (atom-level) GIN to macroscopic (drug-level interactome) GraphSAGE (`macroscopic_ddi_gnn.py`).
- **[March 13, 2026]:** Identified extreme graph sparsity metric (Average Degree: 1.08). Extracted 53,493 known interactions directly from the compressed TWOSIDES `.csv.gz` polypharmacy dataset using pandas, bypassing OS memory issues and API rate limits.
- **[March 13, 2026]:** Batch `UNWIND` ingested 53k+ edges into Neo4j Aura Cloud DB, scaling Average Node Degree to 79.25 (a 7,237% density increase).
- **[March 13, 2026]:** Re-extracted PyG mapping and executed a 150-epoch training run. Model achieved robust convergence: **AUC: 0.9867 (98.67%) | AP: 0.9865 (98.65%)**. Weights successfully saved to `macroscopic_gnn_weights.pth`.

### Phase 4: API Deployment & Inference Integration (Current Action)
*   **Goal:** Construct a real-time predictive script capable of accepting an arbitrary pair of drug targets, dynamically generating their Morgan fingerprints + biologic tensors, mapping them to the GraphSAGE network, and yielding a probability score natively in the Django-driven backend.
*   **Target Components:**
    *   Build `predict_macroscopic.py` (or equivalent `macroscopic_predictor.py` service) to load `macroscopic_gnn_weights.pth`.
    *   Integrate directly over the existing Vite/React frontend interface to replace or upgrade `gnn_predictor.py`.
    *   Map results to the `YOLOv8` pill-recognition backend arrays.

