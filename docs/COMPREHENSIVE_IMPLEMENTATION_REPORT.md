# Project Aegis Comprehensive Implementation Report

Date: 2026-03-12
Project: Project Aegis (DDI + Pill Scanner + Clinical Visualization Platform)
Repository: molecular-ai

## 1. Executive Summary

This document is a full technical record of what has been implemented so far across frontend, backend, AI/ML, scanner pipeline, data systems, cloud deployment, and iterative fixes. It also captures why major design decisions were made, what changed between iterations, which problems were solved, and what remains.

The project evolved from an initial PubMedBERT-driven DDI workflow and basic camera scanner into a multimodal platform with:
- Clinical DDI inference APIs
- Camera-based drug capture and pill analysis
- YOLO-assisted pill detection integrated into a multimodal ranker
- GNN pipeline enablement for structure-based prediction and polypharmacy analysis
- Deployed frontend and backend services on Google Cloud Run

This report also includes a complete student role split (Students A, B, C, D) for final report production.

## 2. Scope and System Boundaries

### 2.1 Primary system goals
- Predict and explain drug-drug interaction risk
- Support clinical and research workflows with visual explainability
- Allow medication capture from camera workflows (barcode, OCR, and pill image analysis)
- Support both text-based and structure-based ML approaches

### 2.2 Major product surfaces implemented
- Dashboard with DDI analysis workflows and scanner integration
- Scanner modal with automatic cascade (barcode -> OCR -> pill vision)
- Research page content and platform narrative
- Backend REST APIs for DDI prediction, polypharmacy, scanner, and data services
- Model training/inference pipelines for PubMedBERT, GNN support, and pill detection

## 3. Architecture Overview

### 3.1 Frontend
- React + Vite application
- Scanner orchestration in useDrugScanner hook
- Pill CV service in pillDetection service
- Visual result rendering in DetectionResults component
- Dashboard integration for adding scanned drugs directly into DDI analysis

### 3.2 Backend
- Django + Django REST Framework
- Prediction APIs in views
- Scanner APIs in views_scanner
- Service layer for model and data source abstractions
- Fallback logic for external databases when local mappings are absent

### 3.3 Data and storage
- SQLite for app persistence
- Neo4j pathway (graph-first retrieval and known interactions)
- Redis cache use in architecture documentation
- JSON fallback datasets for drug metadata and mappings

### 3.4 Cloud runtime
- Frontend deployed as Cloud Run service (aegis-frontend)
- Backend deployed as Cloud Run service (aegis-backend)
- Verified health endpoint and live URL routing

## 4. Implementation Timeline and Iteration History

## 4.1 Foundation phase
- Created baseline full-stack app with React frontend and Django backend
- Added core DDI request flow and risk scoring response format
- Introduced visualization components for risk and mechanism communication

## 4.2 AI-first MVP phase
- PubMedBERT integrated as primary DDI prediction path
- Fallback services and rule-based scaffolding added for robustness
- Logging and API response consistency improved

## 4.3 Scanner v1 (camera workflow introduced)
- Integrated scanner entry points into dashboard
- Implemented barcode pipeline using Quagga
- Implemented OCR pipeline using Tesseract
- Added initial image/pill analysis path for visual identification

## 4.4 Scanner v2 (CV feature extraction + model fallback)
- Added image preprocessing and segmentation
- Added color extraction and shape classification
- Added imprint region generation for OCR enhancement
- Added TF.js classifier integration with graceful model-load fallback
- Maintained operation when TF model is unavailable or incompatible

## 4.5 Scanner v3 (backend-assisted multimodal ranking)
- Added backend multimodal endpoint that accepts combined signals:
  - color
  - shape
  - imprint
  - model prediction labels/confidence
  - optional uploaded image
- Added scoring and ranking logic for candidate drugs
- Added match-reason metadata for explainability

## 4.6 Scanner v4 (YOLO fusion and uncertainty control)
- Added YOLO detector adapter integration in scanner backend
- Fused detector labels/confidence into multimodal scoring
- Added conservative uncertainty gating:
  - low top confidence threshold gating
  - margin gating between first and second candidates
- Added response decision metadata:
  - status (confident or uncertain)
  - confidence margin
  - user-facing decision message

## 4.7 Scanner v5 (UX/flow polish and bug fixes)
- Fixed scan flow issue where weaker fallback path could override multimodal output
- Added confidence progress bars and method badges in detection result UI
- Added YOLO top-detection display (class, confidence, normalized box)
- Added estimate labeling and uncertainty messaging for user safety

## 4.8 Data quality and detector retraining phase
- Added cleaned dataset generation script for YOLO training set quality control
- Trained tuned detector variants:
  - bootstrap detector
  - v2 tuned detector
  - v3-clean detector
- Switched runtime preference to latest tuned clean detector checkpoint

## 4.9 Data expansion reliability phase
- Expanded DailyMed image pull to larger class coverage
- Hardened downloader with retry logic and per-drug fault tolerance so one failed request does not abort entire dataset generation
- Re-ran expansion to completion and split train/val set

## 4.10 Deployment and live validation phase
- Deployed backend and frontend to Cloud Run
- Verified health endpoint response 200
- Confirmed live URLs and scanner availability for immediate website testing

## 5. Camera and Scanner Pipeline: Detailed Technical Breakdown

## 5.1 Scan strategy (auto mode)
Current order of operations:
1. Barcode scan (fastest, highest precision when readable)
2. OCR drug name scan (label and packaging text)
3. Pill visual scan (CV + ML + multimodal backend)

This ordering minimizes ambiguity and reduces unnecessary model inference cost when deterministic identifiers are available.

## 5.2 Barcode path
- Uses Quagga decodeSingle
- Supports major reader types (UPC, EAN, Code 128, etc.)
- Performs local NDC lookup first
- Falls back to OpenFDA lookup when local database misses
- Produces high-confidence deterministic identification when barcode is valid

## 5.3 OCR path
- Uses Tesseract worker
- Extracts candidate drug tokens with patterns and heuristics
- Performs enhanced backend search with exact/prefix/contains/fuzzy/brand mapping
- Assigns confidence based on OCR confidence and name extraction quality

## 5.4 Pill vision path
- Segmentation: adaptive thresholding and morphological cleanup
- Feature extraction: color, shape, contour-derived properties, bbox
- Imprint preprocessing region generation for OCR
- Optional TF.js inference if model is loaded
- Sends multimodal payload to backend ranker with optional image upload

## 5.5 Multimodal backend ranking
Scoring aggregates multiple signals and adds reasons:
- color and shape exact/partial matching
- imprint exact/contains/fuzzy matching
- model label textual similarity
- YOLO label similarity and detector confidence weighting

Candidates are sorted by confidence and filtered by thresholding.

## 5.6 Uncertainty and safety controls
- Top-score threshold check
- Confidence margin check between rank 1 and rank 2
- Uncertain decisions flagged in API and UI
- Manual verification message shown when confidence is not strong

## 5.7 Fallback hierarchy and resilience
If multimodal ranking returns no known database candidate:
- Surface YOLO top class as estimate
- Try model-label to DB name lookup
- Try feature search endpoint
- Try image upload endpoint
- Fall back to CV-only descriptor output so the scan still provides useful feedback

## 5.8 Scanner UX improvements implemented
- Detection method badges
- Confidence percentage chip
- Confidence progress bar
- YOLO detection details on each result
- Uncertain vs confident messaging
- Overlay image preview of detected pill region

## 6. DDI Modeling Evolution: PubMedBERT to GNN

## 6.1 Why PubMedBERT was used first
- Strong biomedical language prior
- Faster initial integration with existing text-centric DDI data and mechanisms
- Good explainability via interaction-type classes and mechanism narratives

## 6.2 PubMedBERT strengths and limitations observed
Strengths:
- Works well when literature coverage and naming are clean
- Produces interpretable interaction categories
- Good baseline performance in project docs

Limitations:
- Can underperform on specific clinically known high-risk pairs without enough targeted fine-tuning/context
- Sensitive to text representation quality and sentence generation path
- Not structure-first for novel compounds lacking reliable textual context

## 6.3 Why GNN was introduced/enabled
- Structure-based reasoning from SMILES and molecular graph features
- Better route for novel compounds where literature signal may be sparse
- Supports polypharmacy pairwise graph-based scoring paths

## 6.4 GNN architecture implemented
- Molecular featurization from RDKit-derived graph features
- Edge-conditioned message passing and graph encoder stack
- Shared encoder for drug pair embeddings
- Interaction prediction head with calibration support
- Training loop with PR-AUC model selection and early stopping

## 6.5 Current coexistence and switching behavior
- For pairwise API flow, known interactions are checked first via graph data source path
- PubMedBERT remains primary when available in current prediction path
- GNN/fallback services exist and are used when PubMedBERT is unavailable or in polypharmacy structure-driven pathways
- Project direction and recent work indicate stronger practical movement toward structure-aware and multimodal systems rather than text-only operation

Important note for reporting:
This is best described as a transition from a PubMedBERT-led architecture to a hybrid/structure-aware architecture, not a complete hard replacement in every API branch.

## 7. Fine-Tuning, Training, and Data Work

## 7.1 PubMedBERT side
- Fine-tuned biomedical NLP model integrated for DDI relation prediction
- Reported project metrics (see section 8) include strong macro-F1 and recall

## 7.2 GNN side
- Training scripts and pipeline implemented and iterated
- Hyperparameter explorations were run with adjusted max-atoms, dropout, hidden dimensions, epochs, and patience
- Dataset preparation and filtering scripts executed to improve training quality
- Calibration scripts and evaluation utilities exist for post-training quality controls

## 7.3 Pill detector (YOLO and CV stack)
Training progression:
- Bootstrap detector training on generated YOLO dataset
- Tuned detector v2 run
- Cleaned-data retrain v3-clean run

Data quality progression:
- Added cleaning script to remove likely noisy entries
- Generated clean dataset splits and retrained detector
- Expanded DailyMed dataset generation to improve class/image coverage
- Updated downloader reliability with retry and continuation behavior

## 7.4 Deployment and inference readiness
- Detector model path defaults updated to prioritize best recent tuned model
- Runtime detector load checks performed successfully
- Scanner APIs return detector metadata for observability and UI display

## 8. Metrics and Performance Record

## 8.1 DDI model metrics (documented in project reports)
From project documentation and report files:
- Accuracy: 87.3%
- Macro F1: 89.6%
- Precision: 88.2%
- Recall: 91.1%

Class-level F1 (reported):
- mechanism: 91.2%
- effect: 92.4%
- advise: 87.1%
- int: 78.3%
- no_interaction: 99.1%

## 8.2 PubMedBERT baseline statements in docs
- Approximate range references include around 85-90% on DDI corpus-related evaluations
- Existing narrative notes that quality depends on label/data realism and known high-risk pair calibration

## 8.3 Pill detector metrics from training iterations
Recent YOLO iteration metrics captured during runs:
- tuned-yolo-detector-v2 best validation:
  - mAP50: approximately 0.436
  - mAP50-95: approximately 0.402
- tuned-yolo-detector-v3-clean best validation:
  - mAP50: approximately 0.628
  - mAP50-95: approximately 0.589

Interpretation:
- The clean-data retrain significantly improved detection quality versus prior tuned run.
- Differences in data subsets can affect strict apples-to-apples comparison, but directional improvement is strong.

## 8.4 Expanded dataset counts (latest expansion)
DailyMed expansion and split completed with:
- 55 classes
- 805 train images
- 187 val images

## 9. Key Problems Found and How They Were Fixed

## 9.1 Scanner mislabeling and confidence reliability
Problem:
- Wrong labels could appear overly confident when weak or conflicting signals existed.

Fixes:
- Added uncertainty gating using confidence thresholds and margin checks.
- Added multimodal evidence fusion with explicit decision status.
- Added estimate flags and cautionary messaging in UI.

## 9.2 Fallback override bug in scan flow
Problem:
- A weaker fallback path could overwrite higher-quality multimodal results.

Fix:
- Updated scanner flow order and conditional behavior so multimodal results are preserved when present.

## 9.3 TF.js model compatibility fragility
Problem:
- Runtime model load compatibility issues can occur.

Fix:
- Maintained robust CV + backend fallback paths so scanner remains useful even without local TF.js model availability.

## 9.4 Data downloader crash on transient network failure
Problem:
- A single timeout/interruption could abort full data collection.

Fix:
- Added retry logic for file download operations.
- Added per-drug exception handling to continue through failures.

## 9.5 Deployment consistency and operational checks
Problem:
- Repeated deployment command variations and runtime path issues during iterative rollout.

Fix:
- Standardized working deploy flow and validated resulting service URLs and backend health endpoint status.

## 10. Research Decisions and Trade-Off Analysis

## 10.1 PubMedBERT-led strategy
Pros:
- Clinically interpretable via biomedical language and known relation taxonomy
- Strong class-level performance in documented evaluation
- Good user-facing mechanism explanations

Cons:
- Can be brittle for unseen compounds or low-coverage pair contexts
- Requires high-quality sentence/context generation and curated text data

## 10.2 GNN structure-led strategy
Pros:
- Molecule-first representation can generalize to novel compounds with valid structures
- Strong complement to text models in hybrid systems
- Useful for polypharmacy network inference and structure-aware risk reasoning

Cons:
- Heavier engineering/training/dependency complexity
- Sensitive to featurization quality and molecular graph constraints
- Requires careful calibration and robust data filtering

## 10.3 Multimodal scanner strategy
Pros:
- Better than single-signal camera pipelines
- More robust to real-world image noise and partial observations
- Can expose uncertainty instead of forcing a wrong deterministic output

Cons:
- More pipeline complexity and tuning burden
- Requires ongoing data curation and class mapping quality controls

## 11. Production and Deployment Status

## 11.1 Cloud Run services
- Frontend service deployed and active
- Backend service deployed and active
- Backend health endpoint verified returning HTTP 200

## 11.2 Runtime model status
- Detector path points to the latest tuned clean YOLO checkpoint in runtime preference
- Detector load checks completed successfully in backend shell check flow

## 11.3 User test readiness
- Live website is testable now
- Scanner and DDI flows available through deployed frontend/backend

## 12. Remaining Gaps and Next High-Impact Steps

## 12.1 Data quality and annotation depth
- Improve detector label quality with stricter curation and annotation auditing
- Expand balanced class coverage for underrepresented pill categories

## 12.2 Calibration and thresholding
- Continue confidence calibration for both DDI and scanner outputs
- Tune uncertain decision thresholds using real evaluation sets and clinical review

## 12.3 End-to-end evaluation protocol
- Build reproducible benchmark suites for:
  - known high-risk pairs
  - uncertain scanner cases
  - novel structure-only predictions

## 12.4 Explainability and audit trail
- Add persistent prediction provenance (which model path was used and why)
- Add report export support for reviewer and assessor workflows

## 13. File and Component Map of Major Work Areas

Frontend core areas:
- src/hooks/useDrugScanner.js
- src/services/pillDetection.js
- src/components/DrugScanner/DetectionResults.jsx
- src/pages/Dashboard.jsx

Backend scanner and prediction areas:
- web/ddi_api/views_scanner.py
- web/ddi_api/views.py
- web/ddi_api/pill_detector.py
- web/ddi_api/services/ddi_predictor.py
- web/ddi_api/services/gnn_predictor.py

Training and data pipeline areas:
- web/models/train_pill_yolo.py
- web/models/prepare_pill_yolo_dataset.py
- web/models/clean_pill_yolo_dataset.py
- web/models/download_pill_data.py
- src/model/train_gnn.py
- src/model/filter_training_data.py
- src/model/export_aura_data.py

Documentation and evidence sources used:
- docs/MCR_REPORT.md
- docs/GNN_TRAINING_PROCESS.md
- GuidlinesMCR1.md
- PROJECT_DOCUMENTATION.md
- README.md

## 14. Final Student Work Allocation for Report Production

The following split is designed to produce a complete, evidence-backed final report with minimal overlap and clear ownership.

## 14.1 Student A: Scanner and Computer Vision Chapter Lead
Responsibilities:
- Write full scanner lifecycle section (v1 to v5)
- Explain camera flow, barcode/OCR/visual cascade, and multimodal ranking
- Document YOLO integration and uncertainty gating
- Provide UI evidence screenshots and user journey examples

Deliverables:
- Chapter: Camera and Scanner System
- Diagram: Scanner processing pipeline
- Table: Iteration-by-iteration scanner fixes and impact

## 14.2 Student B: AI Modeling and Research Chapter Lead
Responsibilities:
- Write PubMedBERT and GNN comparative analysis
- Cover architecture rationale, strengths/limits, and transition strategy
- Document training methodology and model-selection criteria
- Explain calibration and risk scoring framework

Deliverables:
- Chapter: AI Model Architecture and Research Basis
- Table: PubMedBERT vs GNN pros/cons and use cases
- Appendix: Model assumptions and limitations

## 14.3 Student C: Data Engineering, Training Ops, and Metrics Lead
Responsibilities:
- Document dataset sources, cleaning logic, and expansion process
- Summarize YOLO training runs and metrics progression
- Summarize DDI model metrics from report artifacts
- Build reproducibility appendix (commands, parameters, checkpoints)

Deliverables:
- Chapter: Data Pipeline, Fine-Tuning, and Evaluation
- Table: Metrics dashboard (DDI + detector)
- Appendix: Training run ledger and dataset statistics

## 14.4 Student D: Backend, Cloud, and Integration Lead
Responsibilities:
- Document API architecture and service orchestration
- Explain fallback chains, resilience measures, and bug fixes
- Cover Cloud Run deployment workflow and live validation checks
- Consolidate risk register and future work roadmap

Deliverables:
- Chapter: Backend, Deployment, and Reliability
- Diagram: End-to-end production architecture
- Section: Operational issues fixed and post-deployment validation

## 14.5 Shared final integration workflow
1. Student A drafts scanner chapter with visuals.
2. Student B drafts AI/research chapter with model analysis.
3. Student C drafts data/training/metrics chapter and evidence tables.
4. Student D drafts backend/cloud/reliability chapter and deployment proof.
5. Group merge pass to unify terminology and remove contradictions.
6. Final QA pass to verify every metric and claim maps to evidence.

## 15. Suggested Final Report Structure (Team Merge Template)

1. Abstract
2. Problem Statement and Motivation
3. System Architecture Overview
4. Camera and Scanner Evolution
5. AI Modeling Evolution (PubMedBERT to GNN/Hybrid)
6. Data Pipeline and Fine-Tuning
7. Evaluation Metrics and Discussion
8. Deployment and Real-World Testing
9. Limitations, Risks, and Ethics
10. Future Work and Clinical Readiness Plan
11. Conclusion
12. Appendices (commands, configs, screenshots, API payloads)

## 16. Closing Statement

Project Aegis is no longer only a concept or a static MVP. It now has a live deployable stack, a resilient scanner pipeline, multimodal pill identification logic with uncertainty controls, and a model architecture that has evolved from text-led prediction toward stronger structure-aware and hybrid decisioning. The remaining work is primarily quality scaling: higher-quality data, stricter evaluation, and systematic clinical validation.
