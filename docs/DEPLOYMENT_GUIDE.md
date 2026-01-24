# Project Aegis: Cloud Deployment Guide

> **Purpose**: Step-by-step instructions to deploy Project Aegis to the cloud.

---

## 🎯 Quick Decision: Which Cloud Should I Use?

| Cloud Provider | Best For | Cost | Difficulty |
|----------------|----------|------|------------|
| **Railway** | Quick demos, school projects | Free tier available | ⭐ Easy |
| **Render** | Production-ready, simple | Free tier available | ⭐ Easy |
| **Google Cloud Run** | Serverless, scales to zero | $300 free credit | ⭐⭐ Medium |
| **DigitalOcean** | Cost-effective VMs | $6-24/month | ⭐⭐ Medium |
| **AWS ECS** | Enterprise, scalable | Pay-as-you-go | ⭐⭐⭐ Hard |
| **Azure Container Apps** | Microsoft ecosystem | Pay-as-you-go | ⭐⭐⭐ Hard |

---

## Option 1: Railway (Recommended for Students) ⭐

**Why Railway?**
- Deploys Docker Compose directly
- Free tier: 500 hours/month
- Auto-deploys from GitHub
- Built-in Redis and databases

### Step 1: Push to GitHub

```powershell
# If not already a git repo
cd c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai

git init
git add .
git commit -m "Initial commit - Project Aegis"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/project-aegis.git
git branch -M main
git push -u origin main
```

### Step 2: Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your `project-aegis` repository

### Step 3: Configure Services

Railway will detect your `docker-compose.yml`. You need to set up:

**Backend Service:**
```yaml
# Railway will auto-detect, but ensure these env vars are set:
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=*.railway.app
NEO4J_URI=bolt://neo4j:7687
REDIS_URL=redis://redis:6379
```

**Add Redis:**
1. Click **"+ New"** → **"Database"** → **"Redis"**
2. Railway provides `REDIS_URL` automatically

**Add Neo4j (via Docker):**
1. Click **"+ New"** → **"Docker Image"**
2. Enter: `neo4j:5.9.0`
3. Set environment variables:
   ```
   NEO4J_AUTH=neo4j/your_password
   NEO4J_PLUGINS=["apoc"]
   ```

### Step 4: Deploy

Railway auto-deploys on every git push!

```powershell
# Make a change, push, and watch it deploy
git add .
git commit -m "Deploy to Railway"
git push
```

### Step 5: Get Your URL

Railway gives you a URL like: `https://project-aegis-production.up.railway.app`

---

## Option 2: Render (Also Great for Students) ⭐

### Step 1: Create `render.yaml` (Blueprint)

Create this file in your project root:

```yaml
# render.yaml
services:
  # Backend API
  - type: web
    name: aegis-backend
    env: docker
    dockerfilePath: ./web/Dockerfile
    envVars:
      - key: DJANGO_SECRET_KEY
        generateValue: true
      - key: DEBUG
        value: "False"
      - key: ALLOWED_HOSTS
        value: ".onrender.com"
      - key: NEO4J_URI
        fromService:
          type: pserv
          name: aegis-neo4j
          property: host
      - key: REDIS_URL
        fromService:
          type: redis
          name: aegis-redis
          property: connectionString

  # Frontend
  - type: web
    name: aegis-frontend
    env: static
    buildCommand: npm run build
    staticPublishPath: ./dist
    envVars:
      - key: VITE_API_URL
        fromService:
          type: web
          name: aegis-backend
          property: url

  # Neo4j (Private Service)
  - type: pserv
    name: aegis-neo4j
    env: docker
    dockerfilePath: ./neo4j.Dockerfile
    disk:
      name: neo4j-data
      mountPath: /data
      sizeGB: 1

databases:
  - name: aegis-redis
    plan: free
```

### Step 2: Deploy to Render

1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click **"New"** → **"Blueprint"**
4. Select your repository
5. Render reads `render.yaml` and deploys everything

---

## Option 3: Google Cloud Platform (GCP) ⭐⭐ RECOMMENDED

**Why GCP?**
- **$300 free credit** for new accounts (90 days)
- **Cloud Run** = serverless containers (scales to zero = pay only when used)
- Professional-grade infrastructure
- Great for ML workloads

### GCP Architecture for Project Aegis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GOOGLE CLOUD PLATFORM                              │
│                                                                              │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │  Cloud DNS  │────►│            Cloud Load Balancer                   │  │
│  │  (Domain)   │     │         (HTTPS + SSL Certificate)                │  │
│  └─────────────┘     └─────────────────────┬────────────────────────────┘  │
│                                             │                                │
│                         ┌───────────────────┼───────────────────┐           │
│                         ▼                   ▼                   ▼           │
│              ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│              │   Cloud Run     │ │   Cloud Run     │ │   Cloud Run     │   │
│              │   (Frontend)    │ │   (Backend)     │ │   (Backend)     │   │
│              │   React App     │ │   Django API    │ │   Auto-scaled   │   │
│              └─────────────────┘ └────────┬────────┘ └────────┬────────┘   │
│                                           │                   │             │
│                         ┌─────────────────┼───────────────────┘             │
│                         ▼                 ▼                                  │
│              ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│              │   Memorystore   │ │   Cloud SQL     │ │   Cloud Storage │   │
│              │   (Redis)       │ │   (PostgreSQL)  │ │   (ML Model)    │   │
│              └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                              │
│              ┌─────────────────────────────────────────────────────────┐   │
│              │              Neo4j Aura (Managed Graph DB)              │   │
│              │                  OR                                      │   │
│              │              Compute Engine VM (Self-hosted Neo4j)       │   │
│              └─────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Set Up GCP Account & Project

1. **Create GCP Account** (if you don't have one):
   - Go to [console.cloud.google.com](https://console.cloud.google.com)
   - Sign in with Google account
   - You get **$300 free credit** for 90 days!

2. **Create a New Project**:
   ```
   Click "Select Project" → "New Project"
   Project Name: project-aegis
   Project ID: project-aegis-12345 (auto-generated)
   ```

3. **Enable Required APIs**:
   ```
   Go to "APIs & Services" → "Enable APIs"
   Enable:
   - Cloud Run API
   - Cloud Build API
   - Artifact Registry API
   - Secret Manager API
   - Memorystore for Redis API (optional)
   ```

---

### Step 2: Install Google Cloud CLI

```powershell
# Install gcloud CLI on Windows
# Option 1: Download installer from https://cloud.google.com/sdk/docs/install

# Option 2: Use winget
winget install Google.CloudSDK

# Initialize and login
gcloud init

# Follow prompts to:
# 1. Log in to your Google account
# 2. Select your project (project-aegis)
# 3. Set default region (e.g., us-central1)
```

**Set your project:**
```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
```

---

### Step 3: Create Artifact Registry (Docker Repository)

```powershell
# Create a Docker repository
gcloud artifacts repositories create aegis-repo `
    --repository-format=docker `
    --location=us-central1 `
    --description="Project Aegis Docker images"

# Configure Docker to use gcloud credentials
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

### Step 4: Prepare Your Dockerfiles for Cloud Run

**Update Backend Dockerfile** (`web/Dockerfile`):

```dockerfile
# web/Dockerfile - Cloud Run optimized
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application code
COPY . .

# Copy the ML model (or download from Cloud Storage - see Step 6)
# COPY ../DDI_Model_Final /app/DDI_Model_Final

# Collect static files
RUN python manage.py collectstatic --noinput

# Cloud Run uses PORT environment variable
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 0 ProjectAegis.wsgi:application
```

**Create Frontend Dockerfile** (for static hosting):

```dockerfile
# Dockerfile.frontend - For Cloud Run static serving
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Build with production API URL
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
```

**Create `nginx.conf`** in your project root:

```nginx
server {
    listen 8080;
    server_name _;
    
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        # Proxy to backend (will be configured via env var in Cloud Run)
        proxy_pass ${BACKEND_URL};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

### Step 5: Upload ML Model to Cloud Storage

Your model is ~440MB - too large to include in Docker image efficiently. Upload to Cloud Storage:

```powershell
# Create a bucket
gcloud storage buckets create gs://aegis-ml-models --location=us-central1

# Upload the model folder
gcloud storage cp -r DDI_Model_Final gs://aegis-ml-models/

# Verify upload
gcloud storage ls gs://aegis-ml-models/DDI_Model_Final/
```

**Update your predictor to download from GCS on startup:**

```python
# web/ddi_api/services/model_loader.py
import os
from google.cloud import storage

def download_model_from_gcs():
    """Download ML model from Cloud Storage if not present locally"""
    model_path = "/app/DDI_Model_Final"
    
    if os.path.exists(model_path) and os.listdir(model_path):
        print("Model already exists locally")
        return model_path
    
    print("Downloading model from Cloud Storage...")
    os.makedirs(model_path, exist_ok=True)
    
    client = storage.Client()
    bucket = client.bucket("aegis-ml-models")
    blobs = bucket.list_blobs(prefix="DDI_Model_Final/")
    
    for blob in blobs:
        local_path = os.path.join("/app", blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded: {blob.name}")
    
    print("Model download complete!")
    return model_path
```

Add to `requirements.txt`:
```
google-cloud-storage
```

---

### Step 6: Set Up Secrets in Secret Manager

```powershell
# Create secrets for sensitive data
echo "your-super-secret-django-key" | gcloud secrets create django-secret-key --data-file=-

echo "neo4j-password-here" | gcloud secrets create neo4j-password --data-file=-

# Grant Cloud Run access to secrets
gcloud secrets add-iam-policy-binding django-secret-key `
    --member="serviceAccount:YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com" `
    --role="roles/secretmanager.secretAccessor"
```

---

### Step 7: Deploy Backend to Cloud Run

```powershell
# Navigate to your project
cd c:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai

# Build and push the backend image
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-backend:latest ./web

# Deploy to Cloud Run
gcloud run deploy aegis-backend `
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-backend:latest `
    --platform managed `
    --region us-central1 `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --min-instances 0 `
    --max-instances 10 `
    --set-env-vars "DEBUG=False,ALLOWED_HOSTS=*" `
    --set-secrets "DJANGO_SECRET_KEY=django-secret-key:latest" `
    --allow-unauthenticated
```

**You'll get a URL like:** `https://aegis-backend-xxxxx-uc.a.run.app`

---

### Step 8: Deploy Frontend to Cloud Run

```powershell
# Build and push frontend image
gcloud builds submit `
    --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-frontend:latest `
    --build-arg VITE_API_URL=https://aegis-backend-xxxxx-uc.a.run.app `
    .

# Deploy frontend
gcloud run deploy aegis-frontend `
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/aegis-repo/aegis-frontend:latest `
    --platform managed `
    --region us-central1 `
    --memory 256Mi `
    --cpu 1 `
    --allow-unauthenticated
```

**Frontend URL:** `https://aegis-frontend-xxxxx-uc.a.run.app`

---

### Step 9: Set Up Redis (Memorystore)

```powershell
# Create Redis instance (this takes a few minutes)
gcloud redis instances create aegis-redis `
    --size=1 `
    --region=us-central1 `
    --redis-version=redis_7_0

# Get the Redis IP
gcloud redis instances describe aegis-redis --region=us-central1 --format="get(host)"
```

**Note:** Memorystore requires VPC connector for Cloud Run:

```powershell
# Create VPC connector
gcloud compute networks vpc-access connectors create aegis-connector `
    --network default `
    --region us-central1 `
    --range 10.8.0.0/28

# Update Cloud Run to use Redis
gcloud run services update aegis-backend `
    --vpc-connector aegis-connector `
    --set-env-vars "REDIS_URL=redis://REDIS_IP:6379"
```

---

### Step 10: Set Up Neo4j

**Option A: Neo4j Aura (Managed - Recommended)**

1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Create free instance
3. Get connection URI: `neo4j+s://xxxxx.databases.neo4j.io`
4. Update Cloud Run:

```powershell
gcloud run services update aegis-backend `
    --set-env-vars "NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io" `
    --set-secrets "NEO4J_PASSWORD=neo4j-password:latest"
```

**Option B: Self-hosted on Compute Engine**

```powershell
# Create VM for Neo4j
gcloud compute instances create neo4j-vm `
    --machine-type=e2-medium `
    --zone=us-central1-a `
    --image-family=ubuntu-2204-lts `
    --image-project=ubuntu-os-cloud `
    --boot-disk-size=50GB `
    --tags=neo4j

# Allow Neo4j ports
gcloud compute firewall-rules create allow-neo4j `
    --allow tcp:7474,tcp:7687 `
    --target-tags=neo4j

# SSH into VM and install Neo4j
gcloud compute ssh neo4j-vm --zone=us-central1-a

# On the VM:
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j -y
sudo systemctl enable neo4j
sudo systemctl start neo4j
```

---

### Step 11: Set Up Custom Domain (Optional)

```powershell
# Map custom domain to Cloud Run
gcloud run domain-mappings create `
    --service aegis-frontend `
    --domain aegis.yourdomain.com `
    --region us-central1

# Follow instructions to add DNS records at your registrar
```

---

### Complete GCP Deployment Script

Save this as `deploy-gcp.ps1`:

```powershell
# deploy-gcp.ps1 - One-click GCP deployment

$PROJECT_ID = "YOUR_PROJECT_ID"
$REGION = "us-central1"
$REPO = "aegis-repo"

# Set project
gcloud config set project $PROJECT_ID

# Build and deploy backend
Write-Host "🚀 Deploying Backend..." -ForegroundColor Cyan
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/aegis-backend:latest" ./web

$BACKEND_URL = gcloud run deploy aegis-backend `
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/aegis-backend:latest" `
    --platform managed `
    --region $REGION `
    --memory 2Gi `
    --cpu 2 `
    --allow-unauthenticated `
    --format "value(status.url)"

Write-Host "✅ Backend deployed: $BACKEND_URL" -ForegroundColor Green

# Build and deploy frontend
Write-Host "🚀 Deploying Frontend..." -ForegroundColor Cyan
gcloud builds submit `
    --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/aegis-frontend:latest" `
    --build-arg "VITE_API_URL=$BACKEND_URL" `
    .

$FRONTEND_URL = gcloud run deploy aegis-frontend `
    --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/aegis-frontend:latest" `
    --platform managed `
    --region $REGION `
    --memory 256Mi `
    --allow-unauthenticated `
    --format "value(status.url)"

Write-Host "✅ Frontend deployed: $FRONTEND_URL" -ForegroundColor Green

Write-Host ""
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "Frontend: $FRONTEND_URL"
Write-Host "Backend API: $BACKEND_URL"
```

---

### GCP Cost Estimate

| Service | Free Tier | After Free Tier |
|---------|-----------|-----------------|
| **Cloud Run** | 2M requests/month free | ~$0.00002/request |
| **Cloud Storage** | 5GB free | ~$0.02/GB/month |
| **Memorystore Redis** | None | ~$35/month (1GB) |
| **Neo4j Aura** | Free tier available | ~$65/month (pro) |
| **Artifact Registry** | 500MB free | ~$0.10/GB/month |

**Estimated monthly cost after free tier:** $50-100/month (with managed services)
**With free Neo4j Aura + no Redis:** ~$10-20/month

---

## Option 4: DigitalOcean Droplet (Best Value) ⭐⭐

**Cost:** ~$12-24/month for a capable server

### Step 1: Create a Droplet

1. Sign up at [digitalocean.com](https://digitalocean.com)
2. Create Droplet:
   - **Image:** Ubuntu 22.04
   - **Size:** 4GB RAM / 2 vCPUs ($24/mo) - needed for ML model
   - **Region:** Choose nearest to users
   - **Authentication:** SSH Key (recommended)

### Step 2: Connect to Your Server

```powershell
# From PowerShell
ssh root@YOUR_DROPLET_IP
```

### Step 3: Install Docker

```bash
# On the server
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Verify
docker --version
docker compose version
```

### Step 4: Clone Your Repository

```bash
# On the server
git clone https://github.com/YOUR_USERNAME/project-aegis.git
cd project-aegis/molecular-ai
```

### Step 5: Create Production Environment File

```bash
# Create .env.production
cat > .env.production << 'EOF'
DJANGO_SECRET_KEY=your-super-secret-key-change-this
DEBUG=False
ALLOWED_HOSTS=your-domain.com,YOUR_DROPLET_IP

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

REDIS_URL=redis://redis:6379

# For HTTPS (optional but recommended)
VIRTUAL_HOST=your-domain.com
LETSENCRYPT_HOST=your-domain.com
LETSENCRYPT_EMAIL=your-email@example.com
EOF
```

### Step 6: Create Production Docker Compose

```bash
cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  nginx-proxy:
    image: nginxproxy/nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/tmp/docker.sock:ro
      - certs:/etc/nginx/certs
      - vhost:/etc/nginx/vhost.d
      - html:/usr/share/nginx/html
    restart: always

  acme-companion:
    image: nginxproxy/acme-companion
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - certs:/etc/nginx/certs
      - vhost:/etc/nginx/vhost.d
      - html:/usr/share/nginx/html
      - acme:/etc/acme.sh
    environment:
      - DEFAULT_EMAIL=${LETSENCRYPT_EMAIL}
    depends_on:
      - nginx-proxy
    restart: always

  backend:
    build:
      context: ./web
      dockerfile: Dockerfile
    environment:
      - DJANGO_SECRET_KEY=${DJANGO_SECRET_KEY}
      - DEBUG=${DEBUG}
      - ALLOWED_HOSTS=${ALLOWED_HOSTS}
      - NEO4J_URI=${NEO4J_URI}
      - REDIS_URL=${REDIS_URL}
      - VIRTUAL_HOST=${VIRTUAL_HOST}
      - LETSENCRYPT_HOST=${LETSENCRYPT_HOST}
    volumes:
      - ./DDI_Model_Final:/app/DDI_Model_Final:ro
      - static_volume:/app/static
    depends_on:
      - neo4j
      - redis
    restart: always

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - VIRTUAL_HOST=${VIRTUAL_HOST}
      - LETSENCRYPT_HOST=${LETSENCRYPT_HOST}
    depends_on:
      - backend
    restart: always

  neo4j:
    image: neo4j:5.9.0
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: always

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: always

volumes:
  certs:
  vhost:
  html:
  acme:
  static_volume:
  neo4j_data:
  neo4j_logs:
  redis_data:
EOF
```

### Step 7: Deploy!

```bash
# Load environment variables
export $(cat .env.production | xargs)

# Build and start
docker compose -f docker-compose.prod.yml up --build -d

# Check logs
docker compose -f docker-compose.prod.yml logs -f
```

### Step 8: Set Up Domain (Optional but Recommended)

1. Buy a domain (Namecheap, GoDaddy, etc.)
2. Add DNS A record pointing to your Droplet IP
3. Update `.env.production` with your domain
4. Redeploy - Let's Encrypt will auto-generate SSL certificate

---

## Option 4: AWS (Production-Grade) ⭐⭐⭐

For enterprise deployment, use AWS ECS (Elastic Container Service).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         AWS Cloud                                │
│                                                                  │
│  ┌─────────────┐     ┌─────────────────────────────────────┐   │
│  │   Route 53  │────►│         Application Load Balancer   │   │
│  │   (DNS)     │     │              (ALB)                   │   │
│  └─────────────┘     └─────────────────┬───────────────────┘   │
│                                         │                        │
│                        ┌────────────────┼────────────────┐      │
│                        ▼                ▼                ▼      │
│              ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│              │   ECS Task  │  │   ECS Task  │  │   ECS Task  │ │
│              │   Frontend  │  │   Backend   │  │   Backend   │ │
│              └─────────────┘  └─────────────┘  └─────────────┘ │
│                                       │                         │
│                        ┌──────────────┼──────────────┐         │
│                        ▼              ▼              ▼         │
│              ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│              │ ElastiCache │  │   Neptune   │  │     S3      │ │
│              │   (Redis)   │  │  (Graph DB) │  │  (Models)   │ │
│              └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Install AWS CLI

```powershell
# Install AWS CLI
winget install Amazon.AWSCLI

# Configure
aws configure
# Enter: Access Key, Secret Key, Region (e.g., us-east-1)
```

### Step 2: Create ECR Repository

```powershell
# Create repositories for your images
aws ecr create-repository --repository-name aegis-backend
aws ecr create-repository --repository-name aegis-frontend

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
```

### Step 3: Push Images

```powershell
# Build and tag
docker build -t aegis-backend ./web
docker tag aegis-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/aegis-backend:latest

# Push
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/aegis-backend:latest
```

### Step 4: Create ECS Cluster

Use AWS Console or Terraform (recommended for reproducibility).

```hcl
# main.tf (Terraform)
resource "aws_ecs_cluster" "aegis" {
  name = "aegis-cluster"
}

resource "aws_ecs_task_definition" "backend" {
  family                   = "aegis-backend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 1024
  memory                   = 2048
  
  container_definitions = jsonencode([
    {
      name  = "backend"
      image = "${aws_ecr_repository.backend.repository_url}:latest"
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      environment = [
        { name = "DEBUG", value = "False" }
      ]
    }
  ])
}
```

---

## 🔐 Security Checklist for Production

Before deploying, ensure:

```
┌────────────────────────────────────────────────────────────────┐
│                    SECURITY CHECKLIST                          │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [ ] DEBUG = False in Django settings                          │
│  [ ] ALLOWED_HOSTS set to specific domains only                │
│  [ ] Secret keys generated and stored securely                 │
│  [ ] HTTPS enabled (Let's Encrypt or AWS Certificate Manager)  │
│  [ ] Neo4j password changed from default                       │
│  [ ] Redis password set (if exposed)                           │
│  [ ] CORS configured for specific origins                      │
│  [ ] Rate limiting enabled on API endpoints                    │
│  [ ] Database backups configured                               │
│  [ ] Logging and monitoring set up                             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Update Django Settings for Production

```python
# web/ProjectAegis/settings.py - Production additions

import os

# Security
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

# HTTPS
SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# CORS
CORS_ALLOWED_ORIGINS = [
    "https://your-domain.com",
    "https://www.your-domain.com",
]
```

---

## 📊 Monitoring Your Deployment

### Health Check Endpoint

Add to your Django API:

```python
# views.py
from django.http import JsonResponse
from django.db import connection

def health_check(request):
    """Health check endpoint for load balancers"""
    health = {
        'status': 'healthy',
        'database': 'ok',
        'neo4j': 'ok',
        'redis': 'ok',
        'model': 'ok'
    }
    
    try:
        # Check Django DB
        connection.ensure_connection()
    except Exception as e:
        health['database'] = str(e)
        health['status'] = 'unhealthy'
    
    # Add similar checks for Neo4j, Redis, Model loading
    
    status_code = 200 if health['status'] == 'healthy' else 503
    return JsonResponse(health, status=status_code)
```

```python
# urls.py
urlpatterns = [
    path('health/', health_check, name='health_check'),
    # ... other urls
]
```

---

## 🚀 Deployment Commands Cheat Sheet

```powershell
# ==========================================
# LOCAL DEVELOPMENT
# ==========================================
docker-compose up --build -d          # Start locally
docker-compose logs -f                 # View logs
docker-compose down                    # Stop

# ==========================================
# GOOGLE CLOUD (GCP)
# ==========================================
gcloud auth login                                    # Login
gcloud config set project YOUR_PROJECT_ID           # Set project
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/REPO/IMAGE ./web  # Build
gcloud run deploy SERVICE --image IMAGE_URL         # Deploy
gcloud run services list                            # List services
gcloud run services describe SERVICE                # Get URL

# ==========================================
# RAILWAY
# ==========================================
# Just push to GitHub - auto-deploys!
git push origin main

# ==========================================
# DIGITALOCEAN
# ==========================================
ssh root@YOUR_IP                       # Connect
cd ~/project-aegis/molecular-ai        # Navigate
docker compose -f docker-compose.prod.yml up -d --build  # Deploy
docker compose -f docker-compose.prod.yml logs -f        # Logs

# ==========================================
# AWS ECS
# ==========================================
aws ecr get-login-password | docker login --username AWS --password-stdin ECR_URL
docker build -t aegis-backend ./web
docker push ECR_URL/aegis-backend:latest
aws ecs update-service --cluster aegis --service backend --force-new-deployment
```

---

## 💰 Cost Comparison

| Service | Free Tier | Paid Estimate | Notes |
|---------|-----------|---------------|-------|
| **Railway** | 500 hrs/mo, 512MB | ~$5-20/mo | Great for demos |
| **Render** | 750 hrs/mo | ~$7-25/mo | Auto-sleep on free |
| **Google Cloud Run** | $300 credit + 2M req/mo | ~$20-50/mo | Scales to zero! |
| **DigitalOcean** | $200 credit (60 days) | $24/mo | Best value for ML |
| **AWS ECS** | 1 year free tier | ~$50-100/mo | Most scalable |
| **Fly.io** | 3 shared VMs free | ~$10-30/mo | Good for APIs |

---

## 🎓 For School Projects: Quick Deploy Summary

**Fastest path (5 minutes):**

1. Push code to GitHub
2. Sign up for [Railway](https://railway.app) with GitHub
3. Click "Deploy from GitHub"
4. Add Redis database
5. Set environment variables
6. Done! Get your URL

**Share with teacher:**
- Live URL: `https://your-app.railway.app`
- GitHub repo: `https://github.com/you/project-aegis`
- This documentation

---

*Last updated: January 2026*
