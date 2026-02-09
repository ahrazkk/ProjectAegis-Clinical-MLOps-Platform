#!/bin/bash
# ============================================================================
# Project Aegis - Google Cloud Deployment Script
# ============================================================================
# This script builds and deploys the application to Google Cloud Run
# 
# Prerequisites:
# - Google Cloud SDK installed and configured
# - Docker installed
# - gcloud authenticated: gcloud auth login
# - Project configured: gcloud config set project YOUR_PROJECT_ID
# ============================================================================

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
BACKEND_SERVICE="aegis-backend"
FRONTEND_SERVICE="aegis-frontend"

# Validate configuration
if [ "${PROJECT_ID}" = "your-project-id" ]; then
    echo "Error: Please set GCP_PROJECT_ID environment variable or edit PROJECT_ID in this script"
    echo "Example: export GCP_PROJECT_ID=my-gcp-project"
    exit 1
fi

# Derived values
REGISTRY="gcr.io/${PROJECT_ID}"
BACKEND_IMAGE="${REGISTRY}/${BACKEND_SERVICE}:latest"
FRONTEND_IMAGE="${REGISTRY}/${FRONTEND_SERVICE}:latest"

echo "============================================"
echo "Project Aegis - Cloud Deployment"
echo "============================================"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Check if gcloud is configured
if ! gcloud config get-value project &>/dev/null; then
    echo "Error: gcloud not configured. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# ============================================================================
# Build and Deploy Backend
# ============================================================================
echo ""
echo "=== Building Backend Docker Image ==="
cd "$(dirname "$0")/web"

docker build -f Dockerfile.cloud -t "${BACKEND_IMAGE}" .

echo ""
echo "=== Pushing Backend Image to GCR ==="
docker push "${BACKEND_IMAGE}"

echo ""
echo "=== Deploying Backend to Cloud Run ==="
gcloud run deploy "${BACKEND_SERVICE}" \
    --image "${BACKEND_IMAGE}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars="DJANGO_SETTINGS_MODULE=ProjectAegis.settings,DEBUG=False"

# Get backend URL
BACKEND_URL=$(gcloud run services describe "${BACKEND_SERVICE}" \
    --platform managed \
    --region "${REGION}" \
    --format="value(status.url)")

echo "Backend deployed at: ${BACKEND_URL}"

# ============================================================================
# Build and Deploy Frontend
# ============================================================================
echo ""
echo "=== Building Frontend Docker Image ==="
cd "$(dirname "$0")"

docker build -f Dockerfile.prod \
    -t "${FRONTEND_IMAGE}" \
    --build-arg VITE_API_URL="${BACKEND_URL}/api/v1" \
    .

echo ""
echo "=== Pushing Frontend Image to GCR ==="
docker push "${FRONTEND_IMAGE}"

echo ""
echo "=== Deploying Frontend to Cloud Run ==="
gcloud run deploy "${FRONTEND_SERVICE}" \
    --image "${FRONTEND_IMAGE}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 5

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe "${FRONTEND_SERVICE}" \
    --platform managed \
    --region "${REGION}" \
    --format="value(status.url)")

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo "Frontend: ${FRONTEND_URL}"
echo "Backend:  ${BACKEND_URL}"
echo ""
echo "To update CORS settings on backend, run:"
echo "gcloud run services update ${BACKEND_SERVICE} --update-env-vars=CORS_ALLOWED_ORIGINS=${FRONTEND_URL}"
