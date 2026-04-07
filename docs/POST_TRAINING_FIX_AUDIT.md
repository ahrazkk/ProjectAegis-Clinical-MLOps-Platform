# Post-Training Fix Audit — Project Aegis

## Context

The GNN was retrained on 25K+ samples (Neo4j + DDI Corpus + TWOSIDES) with Focal Loss,
4 GNN layers, and hard negatives. After training completes on Colab, the files below need
updating with REAL metrics from `evaluation_predictions.json` and `training_results.json`.

**Old fake metrics:** PR-AUC 0.9809, Accuracy 94.6%, Precision 95.7%, Recall 93.4%,
Confusion Matrix [10247/451/703/9995], Test Set 21,396 pairs

**New real metrics:** Will come from Colab output files.

---

## PRIORITY 1: Files with Hardcoded Fake Metrics (MUST FIX)

### 1.1 — `src/pages/ResearchPage.jsx` (Lines 688, 705)

**What's wrong:** Hardcoded "AUC: 0.9809" displayed in the Research page UI.
```jsx
// Line 688
<div ...>PRECISION-RECALL [AUC: 0.9809]</div>
// Line 705
<Area ... name="PR (AUC = 0.9809)" .../>
```

**How to fix:** Replace hardcoded values with dynamic values from the API/model response,
OR update to the real PR-AUC from training. If the new model achieves ~0.99, the number
is similar but must come from real data.

**Difficulty:** Easy (2 string replacements)

---

### 1.2 — `docs/poster_visuals/generate_visuals.py` (ENTIRE FILE)

**What's wrong:** All 8 charts are generated from hardcoded fake data. Every single metric
is typed in manually — fake ROC curve, fake confusion matrix, fake t-SNE, etc.

**How to fix:** DELETE this folder entirely. Replace with `docs/poster_visuals_real/` which
already has 9 charts from real data. After Colab training, regenerate the remaining charts
(ROC, PR, confusion matrix, probability distribution) from `evaluation_predictions.json`.

**Difficulty:** Easy (delete old, generate new from real predictions)

---

### 1.3 — `docs/GNN_AI_PERFORMANCE.md` (Lines 38, 48, 109, 143, 149)

**What's wrong:** Claims "21,396 test pairs", "99.4% Recall", "64 False Negatives",
describes metrics from the macroscopic GNN as if they're verified. These numbers came
from running `evaluate_gnn_metrics.py` on the macroscopic model, but the results were
never saved — they were manually typed into this document.

**How to fix:** After Colab training, update ALL metrics in this document with the
real values from `training_results.json`. Update test set size, confusion matrix,
precision, recall, etc.

**Difficulty:** Medium (need to rewrite ~5 sections with real numbers)

---

### 1.4 — `PROJECT_AEGIS_FINAL_COMPREHENSIVE_REPORT.md` (Line 1702)

**What's wrong:** "Confusion Matrix (21,396 test pairs)" — references the fake test set size
and likely has fake confusion matrix values below it.

**How to fix:** Update with real confusion matrix from `evaluation_predictions.json`.

**Difficulty:** Easy (update one section)

---

### 1.5 — `web/evaluate_gnn_metrics.py` (Line 40)

**What's wrong:** Comment claims: "This solves Alert Fatigue by increasing Precision to
95%+ while maintaining 93%+ Recall" — these percentages were aspirational, not measured.

**How to fix:** Update comment with actual measured values after training, or remove
the specific percentage claims.

**Difficulty:** Easy (1 comment)

---

## PRIORITY 2: Architecture Mislabeling

### 2.1 — Microscopic GNN is GIN, not GraphSAGE

**What's wrong:** The project has TWO GNN models:
- **Macroscopic** (`macroscopic_ddi_gnn.py`) — uses GraphSAGE (SAGEConv) ← CORRECT
- **Microscopic** (`gnn_model.py`) — uses GIN (EdgeConditionedGINConv)

The README.md and other docs correctly describe the macroscopic model as GraphSAGE.
But make sure no documentation claims the microscopic/structure-based model is GraphSAGE.

**Files to check:**
- `README.md` — Lines 46, 77, 83, 90, 94, 134, 271, 505 reference GraphSAGE
  (these are all about the MACROSCOPIC model, so they're CORRECT)
- `docs/GNN_AI_PERFORMANCE.md` — Describes GraphSAGE (macroscopic model — CORRECT)

**How to fix:** These references are actually correct because they describe the macroscopic
model. But after retraining, the microscopic GIN model is the one being improved. Add a
note clarifying both models exist and which one was retrained.

**Difficulty:** Medium

---

## PRIORITY 3: Outdated Dataset Numbers

### 3.1 — Old training data stats vs new

**Old numbers (in docs/code):**
- 554 unique drugs, 2,002 train samples, 368 val samples
- PR-AUC 0.7903
- Source: Neo4j only

**New numbers (from enhanced training):**
- 1,087 unique drugs, 25,360 train samples, 3,169 val samples, 3,169 test samples
- Sources: Neo4j (1,214) + DDI Corpus (1,527) + TWOSIDES (15,000)
- Hard negative sampling with Tanimoto similarity
- PR-AUC: TBD from Colab (looks like ~0.99+)

**Files to update:**
- `web/models/gnn/training_results.json` — overwrite with Colab output
- `web/data/gnn_training/metadata.json` — keep as-is (old data reference)
- `README.md` — update model architecture section with new training stats
- `PROJECT_AEGIS_FEATURES_DEEP_DIVE.md` — update if it references old data sizes

**Difficulty:** Easy (copy files from Colab, update docs)

---

## PRIORITY 4: Generate Real Charts After Training

### 4.1 — Charts that need real data from Colab

After you download `evaluation_predictions.json` from Colab, generate these charts
from REAL model predictions:

| Chart | Source Data | Script |
|-------|------------|--------|
| ROC Curve | `evaluation_predictions.json` → fpr/tpr | New script needed |
| PR Curve | `evaluation_predictions.json` → precision/recall | New script needed |
| Confusion Matrix | `evaluation_predictions.json` → y_true/y_pred | New script needed |
| Probability Distribution | `evaluation_predictions.json` → y_scores | New script needed |
| Training Loss Curves | `training_history.json` → train_loss/val_loss per epoch | New script needed |
| Metrics Card | `training_results.json` → real metrics | Update existing |

### 4.2 — Charts already generated from real data (in `docs/poster_visuals_real/`)

These are already correct and don't need updating:
- `1_real_tsne_embeddings.png` — real 1,350 node positions from gnn_real_data.json
- `2_real_data_distribution.png` — real severity/class balance from train.json
- `5_real_yolo_training.png` — real YOLO training curves from CSV
- `6_real_evidence_weights.png` — real design constants from source code
- `7_real_digital_twin_factors.png` — real design constants from source code
- `8_real_drug_types.png` — real drug class distribution
- `9_real_dataset_composition.png` — real dataset stats

**NOTE:** The t-SNE chart (1) uses positions from the OLD macroscopic model embeddings.
After retraining the microscopic model, we could regenerate t-SNE from the new model's
drug embeddings for better clustering. This requires running inference on all drugs and
extracting embeddings — can be done in Colab.

---

## PRIORITY 5: Neo4j Knowledge Graph Expansion (OPTIONAL)

### 5.1 — Add TWOSIDES pairs to Neo4j Aura

**What it helps:** The app's Knowledge Graph Explorer, Evidence Chain, Report Generator,
and Drug Scanner all query Neo4j at runtime. More data = better coverage.

**What it does NOT help:** The GNN model (already trained offline on SMILES pairs).

**Recommendation:** Add top 10-20K TWOSIDES pairs with `source: "twosides"` label.
The evidence weighting system already assigns TWOSIDES weight = 0.78.

**How to do it:** Write an ingestion script that:
1. Loads `web/data/twosides_significant_pairs.json` (154K pairs)
2. Matches drug names to existing Neo4j drug nodes
3. Creates INTERACTS_WITH relationships with properties:
   - `source: "twosides"`
   - `severity: "moderate"/"severe"` (based on sig_conditions count)
   - `confidence: 0.78`
   - `sig_conditions: <count>`
4. Run locally — just API calls, no GPU needed

**Risk:** Aura free tier has limits. Check usage before bulk insert.

---

## PRIORITY 6: Model Integration

### 6.1 — Replace old model weights

After Colab:
1. Copy `gnn_best_model.pt` → `web/models/gnn/gnn_best_model.pt`
2. Copy `training_results.json` → `web/models/gnn/training_results.json`
3. Copy `evaluation_predictions.json` → `web/models/gnn/evaluation_predictions.json`
4. Copy `training_history.json` → `web/models/gnn/training_history.json`

### 6.2 — Verify model loads in Django backend

The model config changed (4 layers instead of 3, new interaction head).
Check that `web/ddi_api/services/` loads the new checkpoint correctly.
The checkpoint contains the config dict — the loading code should use it.

**Files to check:**
- `web/ddi_api/services/gnn_service.py` or wherever the model is loaded for inference
- Make sure it reads `hidden_dim`, `num_gnn_layers`, etc. from the checkpoint

---

## CHECKLIST (copy this)

After Colab training completes:

- [x] Download 4 files from Colab
- [x] Copy files to `web/models/gnn/`
- [x] Generate real charts from `evaluation_predictions.json` (8 charts in `docs/poster_visuals_real/`)
- [x] Update `src/pages/ResearchPage.jsx` with real PR-AUC (0.9962)
- [x] Update `docs/GNN_AI_PERFORMANCE.md` with real metrics (complete rewrite)
- [x] Update `PROJECT_AEGIS_FINAL_COMPREHENSIVE_REPORT.md` confusion matrix (TN=1528/FP=30/FN=51/TP=1540)
- [x] Update `web/evaluate_gnn_metrics.py` comment (removed fake percentage claims)
- [x] Delete `docs/poster_visuals/` (fake charts removed)
- [x] Verify model loads in Django backend (checkpoint has config with num_gnn_layers=4, gnn_predictor.py reads it)
- [x] Update README.md with new training data size and sources (Enhanced GIN v2 as primary, real metrics)
- [ ] (Optional) Add TWOSIDES to Neo4j
- [ ] (Optional) Regenerate t-SNE from new model embeddings
