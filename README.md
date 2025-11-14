# Project Aegis: GCP-Based MLOps for Clinical NLP

**Project Aegis** is a GKE-orchestrated MLOps platform for real-time DDI relation extraction using a calibrated PubMedBERT model. This high-availability (3-replica) microservice platform features REST/gRPC API contracts. Its 90% F1-score model is deployed on a cost-optimized, scale-to-zero GKE GPU node pool.

This repository contains the complete system architecture, MLOps lifecycle design, cost-analysis, and research for the "MZ02 - AI-Powered Clinical Decision Support" capstone project, codenamed Project Aegis.

## 📖 Table of Contents

* [Project Objective](#-project-objective)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [The AI Model](#-the-ai-model)
* [Cloud & Cost Engineering](#-cloud--cost-engineering)
* [Technology Stack](#-technology-stack)
* [Installation & Deployment](#-installation--deployment)
* [Acknowledgments](#-acknowledgments)

## 🎯 Project Objective

The primary objective is to design and develop an AI-powered clinical decision support system that predicts and flags potential drug-drug interactions (DDIs). By mining biomedical literature, the system aims to assist clinicians in making safer prescribing decisions and reduce the incidence of adverse drug events.

## ✨ Key Features

* **Real-Time DDI Prediction:** Analyzes drug pairs and clinical text to predict interaction type and severity.
* **High-Availability Architecture:** Built on a **3-replica GKE design** to mitigate single points of failure (like the Orchestrator) and ensure resilience.
* **Clinically-Safe AI:** Employs **Temperature Scaling** to calibrate the AI's raw outputs. This prevents the "overconfidence" common in neural networks and ensures probability scores are reliable for clinical use.
* **Asynchronous Evidence Gathering:** The central orchestrator runs parallel calls to multiple microservices (NLP, Database, Literature Search) to gather a complete risk profile efficiently.
* **Extreme Cost Optimization:** Engineered to run on a **$0 budget**, slashing a projected **$807/mo** production cost.

## 🏗️ System Architecture

The entire platform is designed as a **GCP-native microservice architecture**. This decouples concerns and allows individual components (e.g., AI model, database) to be scaled and updated independently.

The core of the system is the **Orchestration Service** ("The Brain"), which manages the flow of data. When a clinician requests a DDI check, the orchestrator executes the full API call flow:

1.  **Client** sends a `POST /api/ddi-check` request to the **API Gateway**.
2.  **Gateway** forwards the authenticated request to the **Orchestrator**.
3.  **Orchestrator** triggers parallel `gRPC` and `API` calls to:
    * **Database Service:** Queries a Cloud SQL (PostgreSQL) instance for known interactions and drug synonyms.
    * **NLP Inference Service:** Sends a `gRPC` request to the PubMedBERT model for AI-based analysis.
    * **External Literature Service:** Fetches recent articles for evidence (RAG).
4.  Once evidence is gathered, the **Orchestrator** passes the AI model's raw probabilities to the dedicated **Risk Scoring Service**.
5.  The **Risk Scoring Service** applies the weighted-sum algorithm and temperature scaling logic to calculate a final, calibrated risk score.
6.  If the risk is 'High' or 'Moderate', the **Orchestrator** calls the **Drug Alternatives Service**.
7.  The final aggregated JSON response is synthesized and returned to the clinician.

### API Sequence Diagram

This diagram details the precise API call flow (REST & gRPC) between all microservices.

*(Need to upload my capstone_sequance.drawio.png` file to this repo and update the path below)*
``
`![API Call Flow Sequence Diagram](temp)`

## 🤖 The AI Model

The core of the NLP Inference Service is a fine-tuned **PubMedBERT** model (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`).

* **Task:** Relation Extraction
* **Dataset:** DDIExtraction 2013 Corpus
* **Performance:** The prototype model achieved an **F1-Score of over 90%** on the test set, exceeding the project's initial 80% target.
* **Calibration:** A `ModelWithTemperature` wrapper is used to apply **Temperature Scaling**. This critical post-processing step divides the model's raw logits by a learned `T` value, "softening" the softmax output to produce probabilities that are clinically reliable and trustworthy.

## ☁️ Cloud & Cost Engineering

A major component of this project was engineering a financial plan to make the platform viable for a student budget. The initial production-grade estimate was **$807/mo**. The final architecture costs **$0** (covered by ~$45-60 in GCP free credits).

This **99% cost reduction** was achieved through two key engineering decisions:

1.  **AI Training:** Instead of using expensive, on-demand T4 GPUs, we used high-performance **NVIDIA A100 Spot VMs** via Vertex AI Training. This provided a 3-5x training speedup and was *more* cost-effective, at only **~$2.19 per training run**.
2.  **AI Inference:** The **$450/mo** Vertex AI Endpoint (the single largest cost) was **eliminated**. It was replaced with an on-demand, **scale-to-zero GKE GPU node pool** (NVIDIA T4). The model is only loaded onto a GPU (and thus incurs cost) when a request comes in, and scales back to zero nodes immediately after.

| Service | Initial Prod. Plan (Cost/mo) | Final Student Plan (Cost/mo) |
| :--- | :---: | :---: |
| **AI Model Serving** | Vertex AI Endpoint | **$450.00** | **~$4.56** (GKE GPU Node) |
| **GKE Cluster** | Regional GKE Standard | **$224.00** | **~$5.71** (e2-medium) |
| **Database** | Cloud SQL (HA) | **$125.00** | **~$2.52** (db-f1-micro) |
| **Total (Est.)** | | **$807.20** | **~$13.29** (Demo Week) |
| **Out-of-Pocket** | | **$807.20** | **$0.00** |

## 🛠️ Technology Stack

* **Cloud:** Google Cloud Platform (GCP)
* **Orchestration:** Google Kubernetes Engine (GKE), Docker
* **AI/ML:** Python, PyTorch, PubMedBERT, Vertex AI
* **Database:** Cloud SQL (PostgreSQL)
* **CI/CD:** Google Cloud Build (planned), GitHub
* **Other:** gRPC, REST, Terraform (planned)

## 🚀 Installation & Deployment

(not done yet.)

```bash
# 1. Clone the repository
git clone [https://github.com/ahrazkk/Aegis-Clinical-MLOps-Platform.git](https://github.com/ahrazkk/Aegis-Clinical-MLOps-Platform.git)
cd Aegis-Clinical-MLOps-Platform

# 2. Set up gcloud environment
gcloud config set project [YOUR_PROJECT_ID]
gcloud config set compute/region [YOUR_REGION]

# 3. (Terraform instructions...)

# 4. (Kubernetes deployment instructions...)
kubectl apply -f k8s/
