# GNN Training Process for DDI Prediction

## Overview

ProjectAegis uses a **Graph Neural Network (GNN)** to predict Drug-Drug Interactions (DDIs) from molecular structure. Instead of relying on text-based approaches, the GNN operates directly on molecular graphs derived from SMILES strings, enabling predictions for novel drugs that may lack published literature.

---

## Architecture

```
SMILES String                SMILES String
     |                            |
     v                            v
 +-----------+              +-----------+
 | RDKit     |              | RDKit     |
 | Parse     |              | Parse     |
 +-----------+              +-----------+
     |                            |
     v                            v
 +-----------+              +-----------+
 | Atom      |              | Atom      |
 | Features  |              | Features  |
 | (40-dim)  |              | (40-dim)  |
 +-----------+              +-----------+
     |                            |
     v                            v
 +----------------------------------+
 |    Shared MolecularGNNEncoder    |
 |  (EdgeConditionedGINConv x 3)    |
 |  + Jumping Knowledge             |
 |  + Global Mean+Max Pooling       |
 +----------------------------------+
     |                            |
     v                            v
  drug1_emb                  drug2_emb
     |                            |
     +---------> concat <---------+
                   |
                   v
          +------------------+
          | DDIInteraction   |
          | Head             |
          | Dense->GELU->    |
          | Dropout->LN->Out |
          +------------------+
                   |
                   v
            interaction logit
```

### Components

**1. MolecularGraphFeaturizer** (`gnn_featurizer.py`)
Converts SMILES to graph tensors using RDKit:
- **Node features (40-dim per atom):** atom symbol (one-hot, 13 types), degree (0-5), formal charge (-2 to +2), hydrogen count (0-4), hybridization (SP/SP2/SP3/SP3D/SP3D2), aromaticity flag, ring membership flag
- **Edge features (8-dim per bond):** bond type (single/double/triple/aromatic), conjugation flag, ring flag, stereo flag
- **Adjacency matrix:** binary undirected graph with self-loops
- Molecules are padded to `max_atoms=128` with a node mask

**2. EdgeConditionedGINConv** (`gnn_model.py`)
Graph Isomorphism Network layer with edge conditioning:
```
h_i' = MLP( (1 + eps) * h_i + SUM_j( h_j * sigmoid(W_edge * e_ij) ) )
```
- `eps` is a learnable parameter controlling self-loop importance
- Edge features gate neighbor messages via a learned linear transform
- 2-layer MLP with BatchNorm and ReLU for node updates
- Masked operation ensures padding nodes remain zero

**3. MolecularGNNEncoder** (`gnn_model.py`)
Stacks multiple GINConv layers with:
- **Jumping Knowledge (JK):** concatenates outputs from all layers, then projects back to `hidden_dim`. Captures both local (early layers) and global (later layers) molecular substructures
- **Global Readout:** masked mean pooling + masked max pooling, concatenated to produce a fixed-size molecular fingerprint (`hidden_dim * 2`)

**4. DDIGraphModel** (`gnn_model.py`)
Complete model using a **shared encoder** for both drugs:
- Drug 1 graph -> encoder -> drug1_embedding
- Drug 2 graph -> encoder -> drug2_embedding
- Concatenate -> DDIInteractionHead -> logit

**5. DDIInteractionHead** (`gnn_model.py`)
Prediction head following the MCR III RelationHead pattern:
- Dense (hidden_dim) -> GELU -> Dropout -> LayerNorm -> Output
- Binary mode: 1 output, sigmoid activation
- Multi-class mode: k outputs, softmax activation

---

## Training Pipeline

### Step 1: Prepare Data

Training data is a JSON file where each sample contains SMILES strings for two drugs and their interaction label:

```json
[
  {
    "drug1_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "drug2_smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "drug1_name": "aspirin",
    "drug2_name": "warfarin",
    "interaction_type": 3,
    "has_interaction": 1
  }
]
```

**Fields:**
| Field | Required | Description |
|-------|----------|-------------|
| `drug1_smiles` | Yes | SMILES string for drug 1 |
| `drug2_smiles` | Yes | SMILES string for drug 2 |
| `has_interaction` | Yes (binary) | 1 = interacts, 0 = no interaction |
| `interaction_type` | Yes (multi-class) | 0=none, 1=advice(minor), 2=effect(moderate), 3=mechanism(major PK), 4=int(major PD) |
| `drug1_name` | No | Human-readable drug name |
| `drug2_name` | No | Human-readable drug name |

SMILES can be obtained from public databases: DrugBank, ChEMBL, or PubChem.

### Step 2: Configure Training

All hyperparameters are set via `GNNTrainingConfig`:

```python
from src.model import GNNTrainingConfig

config = GNNTrainingConfig(
    # Architecture
    hidden_dim=256,           # GNN hidden layer dimension
    num_gnn_layers=3,         # Number of message-passing layers
    use_jumping_knowledge=True,  # Concatenate all layer outputs
    max_atoms=128,            # Max atoms per molecule (padding size)

    # Optimization
    learning_rate=1e-3,       # AdamW learning rate
    batch_size=32,            # Training batch size
    weight_decay=1e-4,        # L2 regularization
    dropout_rate=0.1,         # Dropout in prediction head

    # Training
    num_epochs=50,            # Maximum training epochs
    max_grad_norm=1.0,        # Gradient clipping threshold
    early_stopping_patience=10,  # Stop after N epochs without improvement("patience")
    use_binary=True,          # True=binary DDI, False=multi-class
)
```

### Step 3: Build Dataset

The `DDIGraphDataset` pre-processes all SMILES into graph tensors at initialization:

```python
from src.model import MolecularGraphFeaturizer, DDIGraphDataset, create_graph_data_loaders

featurizer = MolecularGraphFeaturizer(max_atoms=128)

train_dataset = DDIGraphDataset.from_json("data/train.json", featurizer)
val_dataset   = DDIGraphDataset.from_json("data/val.json", featurizer)

train_loader, val_loader = create_graph_data_loaders(
    train_dataset, val_dataset, batch_size=32
)
```

During preprocessing, each SMILES string is:
1. Parsed by RDKit into a molecule object
2. Atom features extracted for each atom (40-dim vector)
3. Bond features extracted for each bond (8-dim vector)
4. Padded/truncated to `max_atoms` with a binary node mask
5. Invalid SMILES are skipped with a warning

### Step 4: Train

```python
from src.model import DDIGraphModel, GNNTrainer

model = DDIGraphModel(
    hidden_dim=config.hidden_dim,
    num_gnn_layers=config.num_gnn_layers,
    dropout_rate=config.dropout_rate,
    use_binary=config.use_binary,
)

trainer = GNNTrainer(model, config, output_dir="./checkpoints/gnn")
results = trainer.train(train_loader, val_loader)
```

**What happens during training:**

1. **Optimizer:** AdamW with configurable weight decay
2. **Scheduler:** Cosine annealing learning rate schedule over total training steps
3. **Each epoch:**
   - Forward pass: featurized drug pairs -> GNN encoder -> interaction head -> logits
   - Loss: `BCEWithLogitsLoss` (binary) or `CrossEntropyLoss` (multi-class)
   - Backward pass with gradient clipping (`max_grad_norm=1.0`)
   - Learning rate step
4. **After each epoch:**
   - Evaluate on validation set
   - Compute metrics (PR-AUC, ROC-AUC, precision, recall, F1)
   - If PR-AUC improves: save checkpoint as `gnn_best_model.pt`
   - If no improvement for `early_stopping_patience` epochs: stop
5. **After training:**
   - Save final model as `gnn_final_model.pt`
   - Calibrate temperature scaling on validation set

### Step 5: Inference

```python
from src.model import GNNPredictor

predictor = GNNPredictor(model_path="./checkpoints/gnn/gnn_best_model.pt")

# Predict by SMILES
pred = predictor.predict_from_smiles(
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"  # warfarin
)

# Or by drug name (uses built-in SMILES lookup for 20+ common drugs)
pred = predictor.predict_from_names("aspirin", "warfarin")

# Result fields
pred.has_interaction      # bool
pred.interaction_type     # str or None
pred.raw_probability      # float [0, 1]
pred.calibrated_probability  # float [0, 1] (temperature-scaled)
pred.risk_score           # float [0, 1]
pred.risk_category        # "low" | "moderate" | "high"
pred.confidence           # float [0, 1]

# Human-readable explanation
print(predictor.get_risk_explanation(pred))
```

---

## File Map

```
src/model/
  gnn_featurizer.py      SMILES -> molecular graph tensors (RDKit)
  gnn_model.py           GNN architecture (GINConv, Encoder, DDIHead)
  gnn_dataset.py         Dataset class + DataLoader creation
  gnn_trainer.py         Training loop, checkpointing, early stopping
  gnn_inference.py       High-level prediction API + DDIPrediction dataclass
  evaluation.py          Shared metrics (PR-AUC, ROC-AUC, F1, error analysis)
  risk_scorer.py         Risk scoring + temperature scaling calibration
  hyperparameter_config.py  Vertex AI Vizier HPO configuration
  __init__.py            Package exports
```

---

## Dependencies

```
torch>=2.0.0           # Core ML framework
rdkit-pypi>=2022.9.1   # Molecular graph featurization
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Evaluation metrics
pandas>=2.0.0          # Data loading
tqdm>=4.65.0           # Progress bars
```

No `torch_geometric` is required. The GNN is implemented in pure PyTorch.

---

## Running Tests

```bash
pytest tests/test_gnn.py -v
```

The test suite covers featurization, model forward passes, dataset loading, training loops, checkpoint save/load, and the inference API with real molecule SMILES.
