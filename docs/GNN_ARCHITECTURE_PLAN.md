# 🧠 Hybrid GNN-BERT Architecture for DDI Prediction
*Feature Plan for Project Aegis Capstone*

## 1. Executive Summary
To achieve state-of-the-art (SOTA) performance in Drug-Drug Interaction (DDI) prediction, we propose a **Multi-Modal Hybrid Model** that combines:
1.  **Textual Evidence (PubMedBERT):** analyzing medical literature and interaction descriptions.
2.  **Molecular Graph Topology (GNN):** analyzing the structural relationships between drugs, targets, and enzymes in a Knowledge Graph.

This approach goes beyond simple classification by understanding the *biological context* of an interaction.

## 2. Theoretical Framework

### 2.1 The Knowledge Graph (KG)
We construct a heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where:
*   **Nodes $\mathcal{V}$:**
    *   $v_{drug}$: Drugs (Nodes with SMILES features)
    *   $v_{target}$: Proteins/Genes (UniProt features)
    *   $v_{enzyme}$: Metabolizing enzymes (CYP450 variants)
*   **Edges $\mathcal{E}$:**
    *   $e_{int}$: Known DDI (Drug $\leftrightarrow$ Drug)
    *   $e_{target}$: Drug-Target binding (Drug $\rightarrow$ Target)
    *   $e_{met}$: Metabolism (Enzyme $\rightarrow$ Drug)

### 2.2 Graph Neural Network (GNN) Encoder
We will use **GraphSAGE (Graph Sample and Aggregate)** or **GAT (Graph Attention Network)**.
*   **Input:** Initial node features $x_v$ (generated from Morgan Fingerprints of molecular SMILES).
*   **Message Passing:**
    $$ h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{AGG} \left( \{ h_u^{(k-1)}, \forall u \in \mathcal{N}(v) \} \right) \right) $$
*   **Output:** A dense vector embedding $z_{graph}$ for each drug that captures its biological neighborhood.

### 2.3 Hybrid Fusion
The final risk score $y$ is predicted by fusing the graph embedding with the text embedding:
1.  **Text Embedding ($z_{text}$):** `PubMedBERT(mechanism_text)`
2.  **Graph Embedding ($z_{graph}$):** `GNN(drug_node)`
3.  **Fusion Layer:**
    $$ z_{fused} = \text{Concat}(z_{text}, z_{graph}) $$
    $$ y = \text{Softmax}(\text{MLP}(z_{fused})) $$

## 3. Implementation Plan

### Phase 1: Graph Construction (Current)
- **Tool:** Neo4j (via `neo4j` Python driver).
- **Status:** Basic schema exists.
- **Task:** Populate with aggregated data (RxNorm/DrugBank).

### Phase 2: PyTorch Geometric (PyG) Integration
- **Library:** `torch_geometric`
- **Data Loader:** Custom loader to pull subgraphs from Neo4j into PyG tensors.
- **Model:** 2-layer GATv2Conv.

### Phase 3: Training Loop
- **Loss Function:** Binary Cross Entropy (for interaction existence) or Cross Entropy (for severity class).
- **Optimization:** Joint training of GNN and BERT heads.

## 4. Expected Impact
- **Novelty:** Most student projects use *either* NLP *or* tabular data. This uses both + graph topology.
- **Performance:** Expected F1-score increase of 5-8% over baseline BERT.
- **Award Potential:** High. Demonstrates "Deep Tech" AI engineering.

## 5. References
- *Zitnik et al. (2018)*: Modeling polypharmacy side effects with graph convolutional networks.
- *Gu et al. (2021)*: BERT-based DDI extraction.
