# AI Model Specification: DDI Prediction & Risk Scoring
[cite_start]**Project Identifier:** MZ02 [cite: 306]
[cite_start]**Status:** Finalized for Implementation Phase [cite: 271, 455]

This specification defines the architecture, training configuration, and post-processing logic for the Drug-Drug Interaction (DDI) Clinical Decision Support System.

---

## 1. Model Architecture Design
[cite_start]**Owner:** Student B (Finalized MCR III) [cite: 271]

### 1.1 Encoder Configuration
The backbone of the model is a domain-specific BERT encoder processed to extract entity-specific features.

* [cite_start]**Base Model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`[cite: 311].
* [cite_start]**Input Processing:** Tokenization with specific marker tokens (`[DRUG1]`, `[/DRUG1]`, `[DRUG2]`, `[/DRUG2]`) injected around target entities[cite: 315].
* [cite_start]**Primary Output:** `sequence_output` (Last hidden state of the transformer)[cite: 313].
* **Pooling Strategy:** Mean pooling of hidden states corresponding to the marker tokens to create two fixed-size vectors:
    * `drug1_vec`
    * [cite_start]`drug2_vec`[cite: 315].

### 1.2 Relation Head (Primary Task)
[cite_start]**Purpose:** Determines the presence and type of interaction between the two target drugs[cite: 317].

* [cite_start]**Input Vector:** Concatenation of pooled vectors: `[drug1_vec; drug2_vec]`[cite: 318].
* [cite_start]**Input Dimension:** $768 + 768 = 1536$[cite: 319].
* **Architecture (Sequential Implementation):**
    1.  [cite_start]**Dense Layer:** `units=768`, `activation='gelu'` (Matches BERT internal activation)[cite: 321].
    2.  [cite_start]**Dropout Layer:** `rate=0.1` (Regularization)[cite: 322].
    3.  [cite_start]**LayerNorm:** `epsilon=1e-12` (Stabilization)[cite: 323].
    4.  **Output Layer:**
        * [cite_start]*Configuration A (Binary):* `units=1`, `activation='sigmoid'`[cite: 327].
        * [cite_start]*Configuration B (Multi-class):* `units=k` (where $k$ = number of interaction types + "None"), `activation='softmax'`[cite: 328].

### 1.3 Auxiliary Head (NER Task)
[cite_start]**Purpose:** Regularizes the model by forcing it to learn entity boundaries[cite: 330].

* [cite_start]**Input:** `sequence_output` (Entire last hidden state, unpooled)[cite: 332].
* [cite_start]**Input Dimension:** $768$ (per token)[cite: 333].
* **Architecture (Token-wise Application):**
    1.  [cite_start]**Dropout Layer:** `rate=0.1`[cite: 335].
    2.  [cite_start]**Output Layer:** Dense, `units=m` ($m$ = token classes e.g., O, B-DRUG, I-DRUG), `activation='softmax'`[cite: 336].

---

## 2. Hyperparameter Tuning Strategy
[cite_start]**Platform:** Vertex AI Vizier [cite: 337]
[cite_start]**Objective:** Maximize Validation **PR-AUC** (Precision-Recall Area Under Curve)[cite: 339].
[cite_start]**Algorithm:** Bayesian Optimization[cite: 343].

**Search Space Configuration:**

| Parameter | Type | Scaling | Range | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `learning_rate` | DOUBLE | LOG_SCALE | `[1e-6, 5e-5]` | [cite_start]Transformer stability[cite: 344]. |
| `batch_size` | CATEGORICAL | NONE | `[8, 16, 32]` | [cite_start]VRAM constraints vs. Gradient stability[cite: 344]. |
| `weight_decay` | DOUBLE | LINEAR_SCALE | `[0.0, 0.1]` | [cite_start]AdamW L2 regularization[cite: 344]. |
| `num_warmup_steps` | INTEGER | LINEAR_SCALE | `[100, 1000]` | [cite_start]Prevent early-stage instability[cite: 348]. |
| `head_dropout_rate` | DOUBLE | LINEAR_SCALE | `[0.1, 0.3]` | [cite_start]Regularization for custom relation head[cite: 354]. |
| `aux_loss_weight` | DOUBLE | LINEAR_SCALE | `[0.2, 0.8]` | [cite_start]Balance between Primary and NER tasks[cite: 357]. |

---

## 3. Risk Scoring Logic
[cite_start]**Owner:** Student D (Finalized MCR III) [cite: 273]

The raw model probabilities must be processed into a clinically actionable risk score using the following pipeline.

### 3.1 Probability Calibration
[cite_start]**Requirement:** Raw neural network outputs are "poorly calibrated" and must not be used directly[cite: 464].
[cite_start]**Implementation:** Apply **Temperature Scaling** post-processing to softmax outputs[cite: 465].

### 3.2 Scoring Formula
[cite_start]Calculate the scalar Risk Score ($R$) using the weighted sum of calibrated probabilities ($P_{\text{calibrated}}$) and clinical severity weights ($W$)[cite: 459].

$$R = \sum (P_{\text{calibrated}_i} \times W_i)$$

**Defined Weights ($W_i$):**
* [cite_start]**Minor** (e.g., "Advice"): $0.2$[cite: 461, 462].
* [cite_start]**Moderate** (e.g., "Effect"): $0.6$[cite: 462].
* [cite_start]**Major** (e.g., "Mechanism-Pharmacokinetic"): $0.9$[cite: 462].

### 3.3 Risk Categorization Thresholds
Map the scalar $R$ to a user-facing category.

* [cite_start]**Low Risk:** $R < 0.3$[cite: 144].
* [cite_start]**Moderate Risk:** $0.3 \le R < 0.7$[cite: 145].
* [cite_start]**High Risk:** $R \ge 0.7$[cite: 146].

---

## 4. Evaluation Protocol
[cite_start]**Owner:** Student D (Finalized MCR III) [cite: 455]

To ensure clinical reliability, the model must be validated using the following strict methodology.

* **Primary Metric:** **PR-AUC**. (Accuracy is explicitly rejected due to class imbalance) [cite_start][cite: 469].
* **Validation Method:** **Stratified 10-fold Cross-Validation**. [cite_start]Stratification is mandatory to preserve rare interaction classes in every fold[cite: 471].
* [cite_start]**Error Analysis:** Manual review of failures classified by specific error types (e.g., "Negation Misinterpretation", "Implicit Interaction")[cite: 473].