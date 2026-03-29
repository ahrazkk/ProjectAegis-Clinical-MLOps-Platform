# GNN Galaxy Viewer: Implementation & Architecture

## Overview
The **GNN Galaxy Viewer** is a 3D visualization interface designed to explore the latent space embeddings of our Graph Neural Network (GNN). Originally, the system relied on procedural dummy data generation. We upgraded the system to ingest **mathematically authentic dimensional embeddings** directly from our PyTorch GNN, projecting physical coordinates via T-SNE dimensionality reduction.

## What We Did
1. **Removed Procedural Generation:** Handled the removal of the simulated `generateHops` algorithms previously used to fake the visual data.
2. **Built an ML-to-UI Bridge Pipeline:** Created a dedicated Python script to bridge the PyTorch dataset capabilities cleanly into web-safe `.json` format using Scikit-learn's `TSNE`.
3. **Structured Frontend Rendering:** Modified the `GNNGalaxyViewer.jsx` React-Three-Fiber UI to dynamically trace structural graph logic (`nodes` positioning + `adj` dictionary linkages) using a custom Breadth-First-Search (BFS) method, reflecting true biological interactome hops up to 3 links away.
4. **Optimized for Web Performance:** Rather than conducting T-SNE inferences live on the browser (which would crash client hardware), we implemented an MLOps industry standard: compute heavy vectors mathematically on the backend pipeline, bake them to static JSON artifacts, and deploy the artifact to be read client-side. 

---

## How It Works
1. **Pipeline Execution:** `generate_tsne.py` pulls from our PyTorch model artifacts (`neo4j_gnn_dataset.pt` and `node_mapping.csv`). It extracts node embeddings (e.g., 64-dimensional vectors). 
2. **T-SNE Projection:** It applies a 3-component T-SNE mapping to reduce those dimensions down to 3D Cartesian coordinates ($x, y, z$).
3. **JSON Construction:** It processes and trims the nodes (subsampling to ~1,350 to enforce optimal WebGL frame-rates) and pairs them with their respective connections (edges) stored natively as an `adj` mapping dictionary.
4. **React Canvas Render:** `GNNGalaxyViewer.jsx` imports `gnn_real_data.json` directly into the component. On load, it renders standard spheres for nodes, and renders the edges using line primitives handled by `@react-three/drei` and `@react-three/fiber`.

---

## Core Files Involved

### 1. Frontend Components
*   **File:** [`molecular-ai/src/components/GNNGalaxyViewer.jsx`](../src/components/GNNGalaxyViewer.jsx)
*   **Purpose:** The primary React application viewport rendering the 3D interactome.
*   **Mechanics:** Uses state changes to isolate "Drug A" and "Drug B", tracking graph neighbors utilizing an internal BFS structure to highlight clusters and bridges based on the `adj` lists.
*   **To Modify:** Update here for any visual UI aesthetics: glowing effects, sizes of spheres, interaction logic, and Three.js camera mechanics.

### 2. The Data Generator Script
*   **File:** [`molecular-ai/generate_tsne.py`](../generate_tsne.py)
*   **Purpose:** The script responsible for converting the PyTorch outputs into web UI inputs. 
*   **To Modify:** Modify this file if the backend models change shapes, if you wish to adjust the total number of subsampled nodes rendered (`max_nodes` variable), or if you want to recalculate edges and structural relationships.

### 3. The Static JSON Artifact
*   **File:** [`molecular-ai/src/assets/gnn_real_data.json`](../src/assets/gnn_real_data.json)
*   **Purpose:** The single source of truth representation mapping consumed by React. 
*   **Expected Schema:** 
    ```javascript
    {
      "nodes": [ { "id": "0", "name": "Acetaminophen", "type": "Drug", "pos": [0.5, 1.2, -0.4] }, ... ],
      "adj": { "0": ["12", "40", "88"], "12": ["0", "40"], ... }
    }
    ```

### 4. ML Model References
*   **Dataset:** [`web/models/neo4j_gnn_dataset.pt`](../../web/models/neo4j_gnn_dataset.pt)
*   **Mappings:** [`web/models/node_mapping.csv`](../../web/models/node_mapping.csv)
*   **Trained Weights:** [`DDI_Model_Final/sage_classifier.pth`](../../DDI_Model_Final/sage_classifier.pth)

---

## Issues Encountered & Resolved

### 1. UI Import Resolution Failures
*   **Issue:** Vite raised a runtime compilation error missing `gnn_3d_data.json`.
*   **Resolution:** Identified correct nomenclature mismatch; updated `GNNGalaxyViewer.jsx` to load the physically present `gnn_real_data.json`.

### 2. Blank 3D Canvas / Structure Mismatches
*   **Issue:** Model loaded but output an entirely pure black screen. The Javascript BFS iterator was quietly parsing invalid arrays. 
*   **Resolution:**
  *   The original Python payload distributed coordinates as singular `x, y, z` key strings. We merged this physically in `generate_tsne.py` to spit out an array mapping: `pos: [x,y,z]`, matching the UI `n.pos[0]` iteration logic. 
  *   The `links` edge network originally deployed from python was a flat list array `[ {source: 1, target: 2} ]`. We reformatted the PyTorch matrix structure to export exactly as an adjacency mapping object (`"adj": {"1": [2]}`), aligning mathematically with the React BFS loops.

### 3. PyTorch Version Limitations
*   **Issue:** Loading older PyTorch `.pt` datasets generated `weights_only=False` object warning failures in Python via pickling.
*   **Resolution:** Configured `torch.load` to bypass pickling deprecations dynamically inside the `generate_tsne.py` payload runner.

### 4. Browser/Cloud Restraints
*   **Issue/Inquiry:** Will this stay hardcoded upon cloud deployment? Can we query 3D data dynamically in the cloud?
*   **Resolution:** Clarified that using static `gnn_real_data.json` is paradoxically the *most* scaleable MLOps setup. T-SNE recalculations of dimensions spanning +50,000 floats require high CPU bandwidth and cause intense memory lock. To save GCP/AWS backend costs & browser crash loops, exporting a finalized JSON artifact pre-deployment and providing it to the React application represents peak deployment efficiency.