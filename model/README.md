# DDI Relation Extraction Model

PyTorch implementation of the Drug-Drug Interaction (DDI) Relation Extraction model based on PubMedBERT, as specified in the AI Model Specification (`.github/prompts/plan-aiModelSpecification.prompt.md`).

## Overview

This implementation follows the complete specification including:

1. **Model Architecture** (Section 1)
   - PubMedBERT encoder with entity marker tokens
   - Relation Head for DDI classification
   - Auxiliary Head for NER task

2. **Hyperparameter Tuning Strategy** (Section 2)
   - Bayesian Optimization configuration
   - PR-AUC optimization objective

3. **Risk Scoring Logic** (Section 3)
   - Temperature Scaling for calibration
   - Weighted risk score calculation
   - Risk categorization (Low/Moderate/High)

4. **Evaluation Protocol** (Section 4)
   - PR-AUC as primary metric
   - Stratified 10-fold Cross-Validation
   - Error analysis framework

## Components

### Core Model Files

- **`ddi_model.py`**: Main model architecture
  - `DDIRelationModel`: PubMedBERT-based relation extraction
  - `ModelWithTemperature`: Temperature scaling wrapper for calibration

- **`risk_scorer.py`**: Clinical risk scoring service
  - `RiskScorer`: Risk score calculation and categorization
  - `MultiDrugRiskScorer`: Polypharmacy risk assessment

- **`data_preprocessor.py`**: Data preprocessing and tokenization
  - `DDIDataPreprocessor`: Entity marker injection and tokenization
  - `DDIDataset`: PyTorch Dataset for training/evaluation

- **`trainer.py`**: Training pipeline
  - `DDITrainer`: Complete training loop with hyperparameter config

- **`evaluator.py`**: Evaluation and metrics
  - `DDIEvaluator`: PR-AUC computation and stratified k-fold CV

### Utility Files

- **`example_usage.py`**: Example code demonstrating all components
- **`requirements.txt`**: Python dependencies

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Data Preprocessing

```python
from model import DDIDataPreprocessor

# Initialize preprocessor
preprocessor = DDIDataPreprocessor()

# Inject entity markers and tokenize
text = "The interaction between aspirin and warfarin may increase bleeding risk."
encoding = preprocessor.encode_single(
    text=text,
    drug1_span=(26, 33),  # "aspirin"
    drug2_span=(38, 46),  # "warfarin"
)
```

### 2. Model Inference

```python
from model import DDIRelationModel
import torch

# Initialize model
model = DDIRelationModel(
    num_relation_classes=5,  # None, Mechanism, Effect, Advise, Int
    num_ner_classes=5,       # O, B-DRUG, I-DRUG, B-BRAND, I-BRAND
)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model.predict(
        input_ids=encoding['input_ids'].unsqueeze(0),
        attention_mask=encoding['attention_mask'].unsqueeze(0),
        drug1_mask=encoding['drug1_mask'].unsqueeze(0),
        drug2_mask=encoding['drug2_mask'].unsqueeze(0),
    )

print(f"Predicted class: {predictions['relation_pred'].item()}")
print(f"Probabilities: {predictions['relation_probs'][0]}")
```

### 3. Temperature Scaling (Calibration)

```python
from model import ModelWithTemperature

# Wrap model with temperature scaling
calibrated_model = ModelWithTemperature(model, temperature=1.5)

# Get calibrated predictions
calibrated_pred = calibrated_model.predict_calibrated(
    input_ids=encoding['input_ids'].unsqueeze(0),
    attention_mask=encoding['attention_mask'].unsqueeze(0),
    drug1_mask=encoding['drug1_mask'].unsqueeze(0),
    drug2_mask=encoding['drug2_mask'].unsqueeze(0),
)

print(f"Calibrated probabilities: {calibrated_pred['relation_probs_calibrated'][0]}")
```

### 4. Clinical Risk Scoring

```python
from model import RiskScorer

# Initialize risk scorer
risk_scorer = RiskScorer(use_calibrated_probs=True)

# Compute risk profile
risk_profiles = risk_scorer.compute_full_risk_profile(
    relation_probs=calibrated_pred['relation_probs_calibrated']
)

for profile in risk_profiles:
    print(f"Risk Score: {profile['risk_score']:.3f}")
    print(f"Risk Level: {profile['risk_level']}")
    print(f"Predicted Interaction: {profile['predicted_interaction']}")
```

### 5. Model Training

```python
from model import DDITrainer, DDIDataset
from torch.utils.data import DataLoader

# Create dataset
examples = [
    {
        "text": "...",
        "drug1_span": (start1, end1),
        "drug2_span": (start2, end2),
        "relation_label": 2,  # Effect
        "ner_labels": [0, 0, 1, 2, 2, ...],
    },
    # ... more examples
]

dataset = DDIDataset(examples, preprocessor, include_ner=True)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize trainer
trainer = DDITrainer(
    model=model,
    learning_rate=2e-5,      # Section 2: [1e-6, 5e-5]
    weight_decay=0.01,       # Section 2: [0.0, 0.1]
    num_warmup_steps=500,    # Section 2: [100, 1000]
    aux_loss_weight=0.5,     # Section 2: [0.2, 0.8]
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    save_dir="checkpoints/",
)
```

### 6. Model Evaluation

```python
from model import DDIEvaluator

# Initialize evaluator
evaluator = DDIEvaluator(model=model)

# Evaluate on test set
class_names = ["None", "Mechanism", "Effect", "Advise", "Int"]
metrics = evaluator.evaluate_model(test_loader, class_names)

print(f"PR-AUC: {metrics['pr_auc']:.4f}")  # Primary metric
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(metrics['classification_report'])

# Stratified 10-fold Cross-Validation
cv_results = evaluator.stratified_k_fold_cv(
    dataset=full_dataset,
    k=10,
    class_names=class_names,
)

print(f"Mean PR-AUC: {cv_results['mean_pr_auc']:.4f} ± {cv_results['std_pr_auc']:.4f}")
```

## Model Architecture Details

### Encoder Configuration (Section 1.1)

- **Base Model**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Entity Markers**: `[DRUG1]`, `[/DRUG1]`, `[DRUG2]`, `[/DRUG2]`
- **Pooling Strategy**: Mean pooling of marker token hidden states

### Relation Head (Section 1.2)

```
Input: [drug1_vec; drug2_vec] (1536 dim)
  ↓
Dense(768) + GELU + Dropout(0.1) + LayerNorm
  ↓
Dense(num_classes)
```

### Auxiliary Head (Section 1.3)

```
Input: sequence_output (768 dim per token)
  ↓
Dropout(0.1)
  ↓
Dense(num_ner_classes)
```

## Hyperparameter Tuning (Section 2)

The model is designed to work with Bayesian Optimization (e.g., Vertex AI Vizier):

| Parameter | Type | Scaling | Range | Current |
|-----------|------|---------|-------|---------|
| `learning_rate` | DOUBLE | LOG | [1e-6, 5e-5] | 2e-5 |
| `batch_size` | CATEGORICAL | - | [8, 16, 32] | 16 |
| `weight_decay` | DOUBLE | LINEAR | [0.0, 0.1] | 0.01 |
| `num_warmup_steps` | INTEGER | LINEAR | [100, 1000] | 500 |
| `head_dropout_rate` | DOUBLE | LINEAR | [0.1, 0.3] | 0.1 |
| `aux_loss_weight` | DOUBLE | LINEAR | [0.2, 0.8] | 0.5 |

**Objective**: Maximize Validation **PR-AUC**

## Risk Scoring (Section 3)

### Severity Weights

- **Minor** (Advise): 0.2
- **Moderate** (Effect): 0.6
- **Major** (Mechanism): 0.9

### Risk Formula

```
R = Σ(P_calibrated_i × W_i)
```

### Risk Thresholds

- **Low Risk**: R < 0.3
- **Moderate Risk**: 0.3 ≤ R < 0.7
- **High Risk**: R ≥ 0.7

## Evaluation (Section 4)

### Primary Metric

**PR-AUC** (Precision-Recall Area Under Curve)
- Why? Handles class imbalance better than accuracy
- Computed per-class and macro-averaged

### Validation Method

**Stratified 10-Fold Cross-Validation**
- Preserves rare interaction class distributions
- Reports mean ± std PR-AUC

### Error Analysis

Framework for classifying errors by type:
- Negation Misinterpretation
- Implicit Interaction
- Entity Boundary Errors
- Complex Sentence Structure

## Expected Performance

Target metrics on DDIExtraction 2013 test set:
- **F1-Score**: >90%
- **PR-AUC**: >0.85 (macro-averaged)

## Integration with Project Aegis

This model can be integrated into the Project Aegis microservice architecture:

1. **NLP Inference Service**: Deploy as gRPC endpoint
2. **Risk Scoring Service**: Use `RiskScorer` for clinical decision support
3. **Calibration**: Use `ModelWithTemperature` for reliable probabilities

## References

1. **DDIExtraction 2013 Corpus**: SemEval-2013 Task 9
2. **PubMedBERT**: Gu et al., "Domain-Specific Language Model Pretraining for Biomedical NLP"
3. **Temperature Scaling**: Guo et al., "On Calibration of Modern Neural Networks"

## Citation

```bibtex
@misc{projectaegis2024,
  title={Project Aegis: AI-Powered Clinical Decision Support for Drug-Drug Interactions},
  author={Project Aegis Team},
  year={2024},
  note={MZ02 Capstone Project}
}
```

## License

This implementation is part of Project Aegis (MZ02 Capstone Project).

## Support

For issues or questions, please refer to the main project documentation in `/PROJECT_DOCUMENTATION.md`.
