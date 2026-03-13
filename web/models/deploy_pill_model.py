"""
Deploy a trained pill classifier model to the frontend.

After downloading pill-classifier-tfjs.zip from Colab, run:
    python deploy_pill_model.py --zip-path ~/Downloads/pill-classifier-tfjs.zip

Or if you have the extracted folder:
    python deploy_pill_model.py --model-dir ./pill_model_output/pill-classifier

This copies model.json, weight shards, and labels.json to:
    molecular-ai/public/models/pill-classifier/
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Target directory for the TF.js model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TARGET_DIR = os.path.join(PROJECT_ROOT, "public", "models", "pill-classifier")


def deploy_from_zip(zip_path):
    """Extract and deploy from a zip file."""
    if not os.path.exists(zip_path):
        logger.error(f"Zip file not found: {zip_path}")
        return False

    # Extract to temp directory
    temp_dir = os.path.join(SCRIPT_DIR, "_temp_deploy")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        # Find the model directory (could be pill-classifier/ or root)
        model_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if "model.json" in files:
                model_dir = root
                break

        if not model_dir:
            logger.error("No model.json found in zip file")
            return False

        return deploy_from_dir(model_dir)
    finally:
        # Clean up temp
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def deploy_from_dir(model_dir):
    """Deploy model files from a directory."""
    if not os.path.exists(model_dir):
        logger.error(f"Model directory not found: {model_dir}")
        return False

    model_json = os.path.join(model_dir, "model.json")
    if not os.path.exists(model_json):
        logger.error(f"model.json not found in {model_dir}")
        return False

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Copy all model files
    copied = 0
    for filename in os.listdir(model_dir):
        src = os.path.join(model_dir, filename)
        dst = os.path.join(TARGET_DIR, filename)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
            logger.info(f"  Copied {filename} ({size/1024:.1f} KB)")
            copied += 1

    # Verify essential files
    essential = ["model.json"]
    for f in essential:
        if not os.path.exists(os.path.join(TARGET_DIR, f)):
            logger.error(f"Missing essential file: {f}")
            return False

    # Check for labels.json
    labels_path = os.path.join(TARGET_DIR, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        logger.info(f"  Labels: {len(labels)} classes")
    else:
        logger.warning("  No labels.json found — model may not map predictions to drug names")

    # Check for weight shards
    shards = [f for f in os.listdir(TARGET_DIR) if f.endswith(".bin")]
    logger.info(f"  Weight shards: {len(shards)}")

    total_size = sum(
        os.path.getsize(os.path.join(TARGET_DIR, f))
        for f in os.listdir(TARGET_DIR)
    )
    logger.info(f"\nDeployed {copied} files ({total_size/1024/1024:.1f} MB) to:")
    logger.info(f"  {TARGET_DIR}")
    logger.info("\nThe pill classifier is now available at /models/pill-classifier/model.json")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy pill classifier to frontend")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip-path", help="Path to pill-classifier-tfjs.zip from Colab")
    group.add_argument("--model-dir", help="Path to extracted model directory")
    args = parser.parse_args()

    if args.zip_path:
        success = deploy_from_zip(args.zip_path)
    else:
        success = deploy_from_dir(args.model_dir)

    if success:
        logger.info("\nDeploy successful! The scanner will now use the trained model.")
    else:
        logger.error("\nDeploy failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
