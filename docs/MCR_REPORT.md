# Milestone Completion Report (MCR)
## Project Aegis: AI-Powered Clinical Decision Support for Drug-Drug Interactions

**Project Code:** MZ02  
**Reporting Period:** January 2026  
**Report Date:** January 22, 2026

---

## Team Member Reports

---

## Student A (K.C.) - Data Engineer
**Focus Area:** Literature Review, Dataset Preprocessing, XML Parsing

### Progress Made in Reporting Period

1. **Literature Review Completed**
   - Conducted comprehensive review of existing DDI prediction systems and NLP approaches in biomedical literature
   - Analyzed 15+ research papers on drug-drug interaction extraction, including seminal works by Herrero-Zazo et al. (2013) on the DDI Corpus
   - Documented comparative analysis of rule-based vs. machine learning approaches for DDI detection
   - Identified PubMedBERT as the optimal base model based on domain-specific pretraining on 21GB of PubMed abstracts

2. **DDI Corpus Dataset Acquisition and Preprocessing**
   - Successfully obtained the DDIExtraction 2013 Corpus from Universidad Carlos III de Madrid
   - Corpus contains ~19,000 annotated sentences across 792 documents (DrugBank + MEDLINE sources)
   - Implemented XML parsing pipeline to extract drug entity pairs and their interaction labels
   - Created data preprocessing scripts in `web/ddi_api/services/data_ingestion.py`

3. **XML Parsing Implementation**
   - Developed robust XML parser for DDI Corpus format handling:
     - Document-level parsing with sentence segmentation
     - Drug entity extraction with character offsets
     - Pair-level interaction label extraction (mechanism, effect, advise, int, false)
   - Handled edge cases: overlapping entities, nested drug mentions, multi-word drug names
   - Generated clean CSV/JSON training files with columns: `sentence`, `drug1`, `drug2`, `label`

4. **DrugBank Data Integration**
   - Parsed DrugBank XML (~12,000 drugs) to extract:
     - Drug names and synonyms
     - DrugBank IDs
     - Therapeutic classifications
     - Molecular formulas and weights
     - SMILES structures
   - Loaded 2,088 drugs into SQLite database with full metadata
   - Created brand-to-generic name mapping table (200+ mappings)

5. **Data Quality Assurance**
   - Implemented validation scripts to ensure data integrity
   - Verified label distribution: mechanism (1,319), effect (1,687), advise (826), int (188), false (~15,000)
   - Created train/validation/test splits (80/10/10) with stratified sampling
   - Documented data statistics in `training_metadata.json`

### Difficulties Encountered in Reporting Period

1. **XML Parsing Complexity**
   - DDI Corpus XML schema varied between DrugBank and MEDLINE subsets
   - Required developing separate parsing logic for each source
   - Solution: Created unified data model with source-agnostic output format

2. **Class Imbalance**
   - Severe imbalance with ~78% "false" (no interaction) labels
   - This skewed initial model training toward always predicting "no interaction"
   - Solution: Implemented class weighting and undersampling strategies for training

3. **Drug Name Normalization Challenges**
   - Many drug names appeared with stereochemistry prefixes: (R)-, (S)-, (+)-, (-)- 
   - Salt forms inconsistent: "Warfarin" vs "Warfarin Sodium" vs "Warfarin Potassium"
   - Solution: Built comprehensive normalization pipeline with regex patterns and synonym mappings

4. **Missing DrugBank Properties**
   - Not all drugs had complete molecular data (SMILES, formulas)
   - ~12% of drugs lacked therapeutic classification
   - Solution: Populated missing fields with "Unknown" placeholders; added null handling in API

### Tasks to Be Completed in the Next Reporting Period

1. **Expand Drug Database**
   - Integrate additional data sources (RxNorm, ChEMBL) for broader coverage
   - Add drug-food interaction data
   - Target: Increase drug count from 2,088 to 5,000+

2. **Implement RAG Data Pipeline**
   - Set up PubMed API integration for real-time article retrieval
   - Create document embedding pipeline using sentence-transformers
   - Store embeddings in vector database (Pinecone or Chroma)

3. **Knowledge Graph Population**
   - Define Neo4j schema for drug relationships
   - Load drug-target, drug-enzyme, drug-pathway data
   - Create Cypher queries for graph traversal

4. **Data Versioning**
   - Implement DVC (Data Version Control) for dataset tracking
   - Document data lineage for reproducibility

---

## Student B (T.G.) - ML Engineer
**Focus Area:** Model Development, Fine-tuning, Architecture Design

### Progress Made in Reporting Period

1. **Model Architecture Design**
   - Selected `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` as base model
   - Architecture: 12 transformer layers, 768 hidden dimensions, 12 attention heads, ~110M parameters
   - Added 5-class classification head for DDI relation extraction
   - Implemented in PyTorch using HuggingFace Transformers library

2. **Fine-tuning Pipeline Development**
   - Created training script `web/ddi_api/services/train_model.py` with:
     - DataLoader with dynamic batching and padding
     - AdamW optimizer with linear warmup scheduler
     - Cross-entropy loss with class weights for imbalance handling
     - Gradient clipping (max_norm=1.0) for stability
   - Training hyperparameters:
     - Learning rate: 2e-5
     - Batch size: 16
     - Epochs: 5
     - Max sequence length: 128 tokens
     - Warmup steps: 500

3. **Model Training Execution**
   - Successfully trained model on DDIExtraction 2013 corpus
   - Training completed in ~4 hours on NVIDIA A100 (Vertex AI)
   - Final model saved in safetensors format (~400MB)
   - Model files stored in `DDI_Model_Final/` directory

4. **Performance Metrics Achieved**
   - **Overall Accuracy:** 87.3%
   - **Macro F1-Score:** 89.6% (exceeds 80% target by 9.6%)
   - **Precision:** 88.2%
   - **Recall:** 91.1%
   - Per-class performance:
     - mechanism: F1 = 91.2%
     - effect: F1 = 92.4%
     - advise: F1 = 87.1%
     - int: F1 = 78.3%
     - no_interaction: F1 = 99.1%

5. **Inference Service Implementation**
   - Created `web/ddi_api/services/pubmedbert_predictor.py` with:
     - Singleton pattern for efficient model loading
     - Context sentence generation using biomedical templates
     - Drug name normalization (stereochemistry, salt removal)
     - Probability output for all 5 classes
     - Severity mapping from interaction type to clinical risk
   - Optimized for CPU inference with ~500ms response time

6. **Model Calibration Research**
   - Investigated temperature scaling for probability calibration
   - Implemented `ModelWithTemperature` wrapper class
   - Documented the need for post-hoc calibration to prevent overconfidence
   - Temperature parameter (T=1.5) determined through validation set optimization

### Difficulties Encountered in Reporting Period

1. **GPU Memory Constraints**
   - Initial batch size of 32 caused CUDA OOM errors on T4 GPU
   - Solution: Reduced batch size to 16 with gradient accumulation (effective batch = 32)

2. **Overfitting on Small Classes**
   - Model quickly overfit on minority classes (int, advise)
   - Early stopping triggered too early, hurting overall performance
   - Solution: Implemented focal loss and adjusted class weights

3. **Context Sentence Design**
   - Model was trained on actual PubMed sentences, but at inference we generate synthetic context
   - Distribution mismatch between training and inference contexts
   - Solution: Designed templates that closely match training distribution patterns

4. **Inference Speed Optimization**
   - Initial CPU inference was ~2 seconds per prediction
   - Solution: Implemented model quantization research, batch processing, and caching

5. **Label Interpretation Ambiguity**
   - DDI Corpus labels (mechanism, effect, advise, int) not directly mappable to clinical severity
   - Solution: Created severity mapping heuristic based on label semantics and clinical literature

### Tasks to Be Completed in the Next Reporting Period

1. **Temperature Scaling Implementation**
   - Implement proper temperature scaling calibration on held-out validation set
   - Learn optimal T parameter using NLL minimization
   - Validate calibration using reliability diagrams

2. **RAG Integration**
   - Integrate retrieved PubMed articles as context for predictions
   - Replace template-based context with real literature sentences
   - Implement sentence ranking for most relevant context selection

3. **Model Optimization**
   - Experiment with model distillation for faster inference
   - Evaluate ONNX conversion for production deployment
   - Target: Reduce inference time to <100ms on CPU

4. **Continuous Learning Pipeline**
   - Design feedback loop for model improvement
   - Implement A/B testing framework for model versions
   - Set up model versioning with MLflow

5. **Multi-GPU Training**
   - Implement distributed training for larger batch sizes
   - Experiment with larger models (PubMedBERT-large)

---

## Student C (A.K.) - DevOps/Infrastructure Engineer
**Focus Area:** GCP Setup, Docker, Kubernetes, CI/CD

### Progress Made in Reporting Period

1. **Docker Containerization**
   - Created multi-stage Dockerfile for frontend (React + NGINX):
     - Build stage: Node 18 with Vite compilation
     - Production stage: NGINX Alpine for minimal image size
     - Final image size: ~25MB
   - Created backend Dockerfile with Python 3.11:
     - Includes PyTorch and Transformers dependencies
     - Model files mounted as volume for flexibility
     - Gunicorn WSGI server for production
   - Implemented health check endpoints for container orchestration

2. **Docker Compose Orchestration**
   - Configured 4-service architecture in `docker-compose.yml`:
     - `aegis-frontend`: React app on port 80
     - `aegis-backend`: Django API on port 8000
     - `aegis-redis`: Caching layer on port 6379
     - `aegis-neo4j`: Knowledge graph on ports 7474/7687
   - Implemented service dependencies and startup ordering
   - Created volume mounts for persistent data and model files
   - Configured NGINX reverse proxy for API routing (/api/* → backend)

3. **Local Development Environment**
   - One-command setup: `docker-compose up -d --build`
   - Hot-reload enabled for frontend development
   - Database migrations run automatically on startup
   - Documented in README with troubleshooting guide

4. **GCP Project Setup**
   - Created GCP project with appropriate IAM roles
   - Enabled required APIs: Compute, GKE, Cloud SQL, Cloud Storage, Artifact Registry
   - Set up billing alerts to stay within free tier ($300 credits)
   - Configured gcloud CLI for team access

5. **GKE Cluster Design**
   - Designed cost-optimized cluster architecture:
     - Standard GKE cluster (not Autopilot) for granular control
     - Single zone (us-central1-a) to reduce costs
     - e2-medium nodes for backend services
     - Scale-to-zero GPU node pool for AI inference
   - Estimated costs:
     - GKE cluster: $5.71/mo
     - GPU node (on-demand): $4.56/mo during active use
     - Total: ~$13/mo vs $807/mo production estimate

6. **Infrastructure Documentation**
   - Created architecture diagrams (ASCII and draw.io)
   - Documented all Docker commands and configurations
   - Wrote deployment guide with step-by-step instructions
   - Added troubleshooting section for common issues

### Difficulties Encountered in Reporting Period

1. **Model File Size in Docker**
   - Model files (~400MB) bloated Docker image to >2GB
   - Slow image pulls and increased storage costs
   - Solution: Mount model as volume instead of baking into image

2. **CORS Configuration**
   - Frontend couldn't communicate with backend due to CORS errors
   - Solution: Configured Django CORS middleware and NGINX proxy

3. **Container Memory Limits**
   - Backend container crashed when loading PubMedBERT model
   - Default memory limit insufficient for PyTorch + Transformers
   - Solution: Increased memory limit to 4GB, added swap

4. **Port Conflicts**
   - Local development conflicted with other services on ports 80, 8000
   - Solution: Made ports configurable via environment variables

5. **GCP Quota Limits**
   - Hit GPU quota limits in us-central1 region
   - Solution: Requested quota increase; documented alternative regions

### Tasks to Be Completed in the Next Reporting Period

1. **Kubernetes Deployment**
   - Create K8s manifests (Deployments, Services, Ingress)
   - Implement horizontal pod autoscaling (HPA)
   - Configure GPU node pool with scale-to-zero
   - Set up Kubernetes secrets for sensitive config

2. **CI/CD Pipeline**
   - Set up GitHub Actions for automated testing
   - Configure Cloud Build for container image creation
   - Implement GitOps workflow with ArgoCD or Flux
   - Automated deployment to GKE on merge to main

3. **Monitoring and Logging**
   - Deploy Prometheus + Grafana for metrics
   - Configure Cloud Logging for centralized logs
   - Set up alerting for service health
   - Create dashboards for API performance

4. **Security Hardening**
   - Implement network policies in GKE
   - Set up Cloud Armor for DDoS protection
   - Configure SSL/TLS with Let's Encrypt
   - Implement API rate limiting

5. **Disaster Recovery**
   - Set up automated database backups
   - Document recovery procedures
   - Implement multi-region failover plan

---

## Student D (R.N.) - QA/Documentation Engineer
**Focus Area:** Testing, Evaluation Metrics, Documentation

### Progress Made in Reporting Period

1. **Test Framework Setup**
   - Configured pytest for Python backend testing
   - Set up Jest for React frontend testing
   - Implemented test coverage reporting (currently at 45%)
   - Created CI integration for automated test runs

2. **API Endpoint Testing**
   - Wrote comprehensive test suite for all API endpoints:
     - `GET /api/v1/drugs/search/` - Drug search functionality
     - `POST /api/v1/predict/` - DDI prediction endpoint
     - `POST /api/v1/compare/` - Multi-drug comparison
     - `GET /api/v1/alternatives/` - Therapeutic alternatives
     - `GET /api/v1/stats/` - Database statistics
     - `GET /api/v1/interaction-info/` - Detailed interaction info
   - Test cases cover: valid inputs, edge cases, error handling, response format

3. **Model Evaluation Metrics**
   - Calculated and documented performance metrics:
     - Accuracy: 87.3%
     - Macro F1-Score: 89.6%
     - Precision: 88.2%
     - Recall: 91.1%
   - Created confusion matrix visualization for 5-class predictions
   - Documented per-class performance breakdown
   - Compared against baseline models and published benchmarks

4. **Integration Testing**
   - Created end-to-end test scenarios:
     - Full workflow: drug search → prediction → alternatives
     - Multi-drug comparison with 2, 3, 4, 5 drugs
     - Edge cases: unknown drugs, same drug twice, special characters
   - Implemented Docker-based integration test environment

5. **Documentation Creation**
   - **README.md** - Comprehensive project documentation including:
     - Project overview and objectives
     - System architecture with ASCII diagrams
     - Technology stack breakdown
     - Quick start guide (one-command setup)
     - Detailed installation instructions
     - API reference with request/response examples
     - How it works (high-level explanation)
     - Technical deep dive with code examples
     - Usage examples (curl + Python SDK)
     - Database schema and data pipeline
     - AI model details and performance metrics
     - GCP deployment guide
     - Future roadmap (Phases 2-5)
     - FAQ and troubleshooting guide
   - Total: ~1,400 lines of documentation

6. **User Guide Development**
   - Documented each dashboard view:
     - Analysis Mode: Drug search and prediction workflow
     - Compare Mode: Multi-drug matrix comparison
     - Stats Mode: Database statistics overview
   - Created screenshots and annotated UI elements
   - Wrote step-by-step usage instructions

### Difficulties Encountered in Reporting Period

1. **Test Data Management**
   - Needed consistent test data across all environments
   - Production database had real drug data that shouldn't be modified
   - Solution: Created separate test fixtures with mock drug data

2. **Flaky Tests**
   - Some tests failed intermittently due to timing issues
   - Model loading time caused test timeouts
   - Solution: Increased timeouts, mocked model for unit tests

3. **Documentation Scope**
   - Initial documentation was too high-level for developers
   - Users requested more technical details and code examples
   - Solution: Added technical deep dive section with implementation details

4. **API Response Format Changes**
   - Backend team made response format changes mid-sprint
   - Required updating all test assertions and documentation
   - Solution: Implemented schema validation tests to catch future changes

5. **Keeping Documentation Current**
   - Documentation became outdated as features evolved
   - Solution: Added documentation review to PR checklist

### Tasks to Be Completed in the Next Reporting Period

1. **Increase Test Coverage**
   - Target: Increase coverage from 45% to 80%
   - Add unit tests for all service modules
   - Implement property-based testing for edge cases
   - Add load testing with Locust or k6

2. **Performance Benchmarking**
   - Establish baseline performance metrics
   - Document response time targets (<500ms for predictions)
   - Create performance regression tests
   - Benchmark model inference on CPU vs GPU

3. **Clinical Validation Documentation**
   - Document model limitations and appropriate use cases
   - Create clinician-facing user guide
   - Add disclaimers about non-FDA-approved status
   - Write comparison with gold-standard DDI databases

4. **API Documentation Enhancement**
   - Generate OpenAPI/Swagger specification
   - Create interactive API documentation with Swagger UI
   - Add authentication documentation (when implemented)
   - Document rate limiting and usage quotas

5. **Release Documentation**
   - Create CHANGELOG.md for version tracking
   - Write release notes template
   - Document semantic versioning strategy
   - Create deployment runbook for production releases

---

## Summary of Key Deliverables

| Student | Key Deliverable | Status |
|---------|----------------|--------|
| K.C. | DDI Corpus parsed, 2,088 drugs loaded | ✅ Complete |
| T.G. | PubMedBERT fine-tuned, 89.6% F1 achieved | ✅ Complete |
| A.K. | Docker stack operational, GKE designed | ✅ Complete |
| R.N. | Comprehensive README, test framework | ✅ Complete |

## Next Milestone Objectives

1. **RAG Integration** - Real-time PubMed article retrieval for enhanced predictions
2. **GKE Deployment** - Production deployment on Google Kubernetes Engine
3. **CI/CD Pipeline** - Automated testing and deployment workflow
4. **80% Test Coverage** - Comprehensive test suite for reliability

---

*Report prepared by Project Aegis Team*  
*For questions, contact: project-aegis@example.com*
