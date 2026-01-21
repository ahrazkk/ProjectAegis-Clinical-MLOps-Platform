This section defines the precise layers and configurations for the model heads built on top of the PubMedBERT encoder.

## 1.1. Encoder Configuration

**Base Model:** microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (Gu et al., 2020)

**Output:** Last hidden state (sequence_output).

**Pooling Strategy:** Mean pooling of the hidden states corresponding to the [DRUG1] and [/DRUG1] (and [DRUG2], [/DRUG2]) marker tokens to create two fixed-size vectors: drug1_vec and drug2_vec. (Baldini Soares et al., 2019)

## 1.2. Relation Head (Primary Task)

This head determines the presence and type of interaction.

**Input:** Concatenated feature vector: [drug1_vec; drug2_vec].

**Input Dimension:** $768 + 768 = 1536$

**Architecture (Sequential):**

- **Layer 1 (Dense):** units=768, activation='gelu' (GELU, to match BERT's internal activation). (Devlin et al., 2018)
- **Layer 2 (Dropout):** rate=0.1 (Provides regularization).
- **Layer 3 (LayerNorm):** Epsilon 1e-12 (For stabilization, mimics BERT's normalization).
- **Layer 4 (Output):**
  - For Binary (Interaction vs. None): units=1, activation='sigmoid'.
  - For Multi-class (Types): units=k (where $k$ = number of interaction types + "None"), activation='softmax'.

## 1.3. Auxiliary Head: Token Classification (NER)

This head helps ground the model by forcing it to recognize entities.

**Input:** The entire last hidden state from PubMedBERT (not pooled).

**Input Dimension (per token):** $768$

**Architecture (Applied to each token):**

- **Layer 1 (Dropout):** rate=0.1
- **Layer 2 (Output):** Dense, units=m (where $m$ = number of token classes, e.g., O, B-DRUG, I-DRUG), activation='softmax'.

## 2. Hyperparameter Tuning Strategy (Vertex AI Vizier)

This section defines the search space, algorithm, and objective for the hyperparameter optimization trials.

**Optimization Objective:** Maximize Validation PR-AUC (Precision-Recall Area Under Curve), as it is the most informative metric for this imbalanced dataset.

**Optimization Algorithm:** Bayesian Optimization (Vertex AI Vizier's default is recommended).

| Parameter Name | Type | Scaling | Feasible Values/Range | Purpose |
|----------------|------|---------|----------------------|---------|
| learning_rate | DOUBLE | LOG_SCALE | [1e-6, 5e-5] | Controls the step size for model weight updates. Transformers require small rates.(Devlin et al., 2018) |
| batch_size | CATEGORICAL | NONE | [8, 16, 32] | Number of samples per gradient update. Limited by VRAM; larger sizes offer more stable gradients. |
| weight_decay | DOUBLE | LINEAR_SCALE | [0.0, 0.1] | The 'W' in AdamW. A key L2 regularization parameter for Transformers.(Loshchilov & Hutter, 2019) |
| num_warmup_steps | INTEGER | LINEAR_SCALE | [100, 1000] | Number of initial steps with a low learning rate to prevent early-stage instability.(Devlin et al., 2018) |
| head_dropout_rate | DOUBLE | LINEAR_SCALE | [0.1, 0.3] | The dropout rate to use in the custom relation head (Layer 2). |
| aux_loss_weight | DOUBLE | LINEAR_SCALE | [0.2, 0.8] | The multiplier for the NER auxiliary loss. Balances the primary vs. auxiliary tasks. |

## 3. Risk-Scoring Algorithm Design

The risk-scoring algorithm provides a clinical severity assessment for detected drug-drug interactions.

### 3.1. Core Formula

The algorithm is defined as a weighted sum:

$$R = \sum P_{\text{calibrated}_i} \times W_i$$

Where:
- $R$ = Overall risk score
- $P_{\text{calibrated}_i}$ = Calibrated probability for interaction type $i$
- $W_i$ = Clinical severity weight for interaction type $i$

### 3.2. Severity Weights

The clinical severity weights ($W_i$) are grounded in DrugBank's three-tiered standard (DrugBank, n.d.):

| Severity Level | Examples | Weight ($W_i$) |
|----------------|----------|----------------|
| Minor | "Advice" | 0.2 |
| Moderate | "Effect" | 0.6 |
| Major | "Mechanism-Pharmacokinetic" | 0.9 |

### 3.3. Probability Calibration

Modern neural networks are "poorly calibrated" and "overconfident" (Guo et al., 2017). To address this critical safety issue:

**Method:** Temperature Scaling must be implemented as a post-processing step.

**Purpose:** Ensures the model's probability scores are reliable and suitable for clinical decision-making (Guo et al., 2017; Pleiss, 2017).

**Implementation:** Apply temperature parameter $T$ to softmax outputs: $P_{\text{calibrated}} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$

## 4. Formal Evaluation Methodology

This section defines the complete testing protocol for model validation and assessment.

### 4.1. Primary Metric

**Metric:** PR-AUC (Precision-Recall Area Under Curve)

**Rationale:** Accuracy is a "dangerously misleading metric" for highly imbalanced datasets, whereas PR-AUC is more robust and informative (Google, 2025; Saito & Rehmsmeier, 2015). PR-AUC focuses on the positive class performance, which is critical for rare drug-drug interactions.

### 4.2. Validation Strategy

**Method:** Stratified 10-fold Cross-Validation (DigitalOcean, 2025)

**Stratification:** Essential to ensure each fold contains a representative sample of the rare, positive interaction classes.

**Benefits:**
- Provides reliable performance estimates
- Reduces variance in evaluation metrics
- Maximizes use of limited labeled data

### 4.3. Error Analysis Protocol

A formal protocol for manually reviewing and categorizing model failures to provide actionable insights for future improvements.

**Categories (based on clinical NLP taxonomies, Fu et al., 2024):**
- Negation Misinterpretation
- Implicit Interaction
- Context Window Limitations
- Entity Boundary Errors
- Rare Interaction Types
- Ambiguous Clinical Language

**Process:**
1. Sample failed predictions (false positives and false negatives)
2. Manual review by domain experts
3. Categorize errors using the taxonomy above
4. Document patterns and edge cases
5. Prioritize fixes based on clinical impact
