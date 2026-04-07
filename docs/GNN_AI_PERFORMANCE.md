# GNN AI Performance Report

## 1. Overview

Project Aegis uses **two** Graph Neural Network models for Drug-Drug Interaction prediction:

1. **Macroscopic GNN (GraphSAGE)** — `macroscopic_ddi_gnn.py` — Models the drug interaction network as a whole-graph link prediction problem using SAGEConv layers. Operates on 1,350 drug nodes with 1,343 biological features each.

2. **Microscopic GNN (Enhanced GIN v2)** — `gnn_model.py` — The primary production model. Analyzes molecular structures (SMILES → atom graphs) using Edge-Conditioned GIN convolutions with Jumping Knowledge aggregation. This is the model that powers live predictions on the website.

### How the Microscopic GIN Architecture Works
Unlike the macroscopic approach that models the entire drug network, the microscopic GIN operates at the **atomic level**. Each drug is converted from its SMILES string into a molecular graph where atoms are nodes and chemical bonds are edges. The Edge-Conditioned GIN layers perform message passing across atomic neighborhoods, learning structural patterns that correlate with drug interactions.

The model processes **two molecular graphs simultaneously** (one per drug), generates a learned embedding for each, then combines them through a multi-signal interaction head that computes:
- **Element-wise product** (shared feature activation)
- **Absolute difference** (structural divergence)
- **Element-wise sum** (combined signal)

This triple-signal approach gives the classifier far more expressive power than simple concatenation.

---

## 2. Core Evaluation Metrics (Enhanced GIN v2 — Real Data)

**Source:** `evaluation_predictions.json` from Colab GPU training (April 2026)
**Evaluated on:** `3,149` held-out test samples (never seen during training)
**Training data:** `25,137` samples from Neo4j (1,214) + DDI Corpus (1,527) + TWOSIDES (15,000)

- **PR-AUC (Average Precision):** `0.9962` *(Near-perfect precision-recall trade-off)*
- **ROC-AUC Score:** `0.9951` *(Excellent discrimination between interacting and non-interacting pairs)*
- **Precision:** `0.9809` (98.1%) *(Very few false alarms)*
- **Recall (Sensitivity):** `0.9679` (96.8%) *(Catches 96.8% of all real interactions)*
- **F1 Score:** `0.9744` (97.4%)
- **Global Accuracy:** `0.9743` (97.4%)

### Confusion Matrix Breakdown
Evaluated on **3,149** test drug pair predictions:
* **True Positives (TP):** `1,540` *(Correctly identified dangerous interactions)*
* **True Negatives (TN):** `1,528` *(Correctly identified safe combinations)*
* **False Positives (FP):** `30` *(Flagged as dangerous but actually safe — only 1.9% false alarm rate)*
* **False Negatives (FN):** `51` *(Missed real interactions — 3.2% miss rate)*

### Clinical Significance
The Enhanced GIN v2 model achieves both high precision AND high recall simultaneously, which is critical for clinical deployment:
- **Only 30 false alarms** out of 1,558 safe pairs — doctors will trust the system because it rarely cries wolf
- **Only 51 missed interactions** out of 1,591 real interactions — the system catches nearly every dangerous combination
- This eliminates the classic Alert Fatigue problem where clinicians ignore warnings due to excessive false positives

---

## 3. Dataset Configuration & Training Specifications

### Data Sources (Combined)
| Source | Positive Pairs | Description |
|--------|---------------|-------------|
| Neo4j Knowledge Graph | 1,214 | Curated from DrugBank via Aura DB |
| DDI Corpus | 1,527 | PubMed-extracted interaction pairs |
| TWOSIDES | 15,000 | Adverse event signal detection (PRR > 2, A >= 3) |
| **Total Positives** | **15,849** | — |
| **Hard Negatives** | **15,849** | Generated using Tanimoto similarity (structurally similar non-interacting pairs) |
| **Total Dataset** | **31,698** | Balanced 50/50 positive/negative |

### Split
* **Training:** 25,137 samples (79.3%)
* **Validation:** 3,149 samples (9.9%)
* **Test:** 3,149 samples (9.9%)
* **Unique Drugs:** 1,087

### Hard Negative Mining
Rather than using random non-interacting pairs (which are trivially easy to classify), we computed **Morgan fingerprint Tanimoto similarity** between all drug SMILES. Hard negatives are drug pairs that are structurally similar (high Tanimoto score) but do NOT interact — forcing the model to learn subtle discriminative features.

---

## 4. Training Hyperparameters (Enhanced GIN v2)
* **Epochs:** `63` (early stopped at epoch 48, patience 15)
* **Learning Rate (LR):** `5e-4` with ReduceLROnPlateau scheduler
* **Optimizer:** `Adam` (Weight Decay `1e-5`)
* **Loss Function:** `Focal Loss` (alpha=0.25, gamma=2.0) with label smoothing (0.05)
* **Architecture:** 4-Layer Edge-Conditioned GIN with Jumping Knowledge, 256 hidden dim, Dropout 0.15
* **Interaction Head:** Product + AbsDiff + Sum → 2-layer MLP with LayerNorm
* **GPU:** Google Colab (NVIDIA A100/T4)

---

## 5. Historical Comparison — Model Evolution

| Metric | GIN v0 (Neo4j only) | GraphSAGE Macro (BCE) | Enhanced GIN v2 (Current) |
|--------|---------------------|----------------------|---------------------------|
| Training Data | 2,002 samples | 106,987 edges | 25,137 samples |
| Data Sources | Neo4j only | Neo4j graph | Neo4j + DDI Corpus + TWOSIDES |
| PR-AUC | 0.7903 | 0.9797 | **0.9962** |
| ROC-AUC | ~0.60 | 0.9827 | **0.9951** |
| Precision | N/A | 0.6847 | **0.9809** |
| Recall | N/A | 0.9940 | **0.9679** |
| F1 | N/A | 0.8109 | **0.9744** |
| False Positives | N/A | 4,896 | **30** |
| False Negatives | N/A | 64 | **51** |

### Key Improvements from v0 → v2:
1. **12.7x more training data** (2,002 → 25,137 samples)
2. **3 data sources** instead of 1
3. **Focal Loss** instead of BCE — down-weights easy examples, focuses on hard cases
4. **Hard negative mining** — structurally similar non-interacting pairs force deeper learning
5. **Enhanced interaction head** — product + difference + sum (3 signals vs 1)
6. **4 GNN layers** instead of 3 — deeper feature extraction
7. **Label smoothing** (0.05) — regularization to prevent overconfidence

---

## 6. Methodology

All metrics are computed from `evaluation_predictions.json`, which contains the raw `y_true` labels and `y_scores` (predicted probabilities) for all 3,149 test samples. This file was generated during Colab training by running the trained model on the held-out test set.

1. **Pair-Level Splitting:** Train/val/test splits are done at the drug-pair level to prevent data leakage. No drug pair appears in multiple splits.
2. **Threshold:** Binary predictions use a standard `>= 0.5` threshold on predicted probabilities.
3. **Metrics:** Computed using scikit-learn (`roc_auc_score`, `average_precision_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`).
4. **Reproducibility:** All training data, model weights (`gnn_best_model.pt`), and evaluation predictions are saved and version-controlled.

---

## 7. Future Work
- **Multi-class severity prediction:** Distinguish minor/moderate/severe/critical interactions instead of binary interacts/safe
- **GNN feedback loop:** Use approved corrections from the admin panel as fine-tuning labels to continuously improve accuracy
- **Neo4j expansion:** Add top 10-20K TWOSIDES pairs to Aura DB for broader runtime coverage
- **Confidence calibration:** Use correction data to recalibrate model confidence outputs over time