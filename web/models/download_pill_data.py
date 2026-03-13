"""
Download and Prepare Pill Image Datasets for Training

Downloads from multiple sources:
1. DailyMed API (NLM - active, maintained)
2. ePillID benchmark dataset (GitHub)
3. NIH C3PI reference images

Usage (local or Colab):
    python download_pill_data.py --output-dir ./pill_data
    python download_pill_data.py --output-dir ./pill_data --source dailymed --count 50
    python download_pill_data.py --output-dir ./pill_data --source all --split

This creates the directory structure expected by train_pill_model.py:
    pill_data/
        train/
            acetaminophen/
            ibuprofen/
            ...
        val/
            acetaminophen/
            ...
"""

import os
import json
import shutil
import hashlib
import logging
import argparse
import time
import socket
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# DailyMed API endpoints (active, maintained by NLM)
DAILYMED_SPL_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_MEDIA_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/media.json"

# Common drugs to download images for
DRUG_LIST = [
    "acetaminophen", "ibuprofen", "aspirin", "naproxen", "amoxicillin",
    "metformin", "lisinopril", "atorvastatin", "omeprazole", "amlodipine",
    "metoprolol", "losartan", "warfarin", "gabapentin", "sertraline",
    "fluoxetine", "escitalopram", "alprazolam", "levothyroxine",
    "hydrochlorothiazide", "furosemide", "prednisone", "simvastatin",
    "clopidogrel", "tramadol", "azithromycin", "ciprofloxacin",
    "montelukast", "duloxetine", "bupropion", "aripiprazole", "quetiapine",
    "lamotrigine", "oxycodone", "celecoxib", "lorazepam", "diazepam",
    "zolpidem", "trazodone", "pantoprazole", "rosuvastatin", "famotidine",
    "ondansetron", "donepezil", "sildenafil", "rivaroxaban", "apixaban",
    "diltiazem", "carvedilol", "spironolactone", "doxycycline",
    "methylphenidate", "topiramate", "pregabalin", "meloxicam",
]

# Only skip chemical structure diagrams — keep pills, packaging, boxes, labels
SKIP_PATTERNS = [
    "struct", "structure", "formula", "chemical", "molecular",
    "mechanism", "pathway", "metabolism", "pharmacokinetic",
]


def is_pill_image(filename):
    """Skip only chemical structure diagrams. Keep everything else."""
    name_lower = filename.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in name_lower:
            return False
    return True


def download_file(url, dest_path, timeout=30, retries=2):
    """Download a file with retry logic for transient network failures."""
    req = urllib.request.Request(url, headers={"User-Agent": "ProjectAegis/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(response, f)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError, socket.timeout) as e:
            if attempt >= retries:
                logger.debug(f"Failed to download {url}: {e}")
                return False
            time.sleep(0.5 * (attempt + 1))


def fetch_json(url, timeout=15):
    """Fetch JSON from a URL with error handling."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ProjectAegis/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return None


def download_dailymed_images(output_dir, drug_name, min_images=5, max_spls=20):
    """
    Download pill images from DailyMed API for a given drug.
    Uses the SPL search endpoint to find drug products, then fetches media for each.
    Returns number of images downloaded.
    """
    drug_dir = os.path.join(output_dir, drug_name)
    os.makedirs(drug_dir, exist_ok=True)

    existing = len([f for f in os.listdir(drug_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if existing >= min_images:
        logger.info(f"  {drug_name}: already has {existing} images, skipping")
        return existing

    count = existing

    # Step 1: Search DailyMed for SPLs matching this drug
    search_url = f"{DAILYMED_SPL_API}?drug_name={urllib.request.quote(drug_name)}&pagesize={max_spls}"
    data = fetch_json(search_url)
    if not data or "data" not in data:
        logger.warning(f"  {drug_name}: no SPL results from DailyMed")
        return count

    spls = data["data"]
    logger.debug(f"  {drug_name}: found {len(spls)} SPLs")

    # Step 2: For each SPL, fetch media (images)
    for spl in spls:
        setid = spl.get("setid")
        if not setid:
            continue

        media_url = DAILYMED_MEDIA_API.format(setid=setid)
        media_data = fetch_json(media_url)
        if not media_data or "data" not in media_data:
            continue

        # Media response: { "data": { "media": [ {url, mime_type, name}, ... ] } }
        media_obj = media_data["data"]
        media_list = media_obj.get("media", []) if isinstance(media_obj, dict) else []

        for media_item in media_list:
            mime = media_item.get("mime_type", "")
            media_name = media_item.get("name", "")
            media_url_dl = media_item.get("url", "")

            # Only download image files
            if not mime.startswith("image/") or not media_url_dl:
                continue

            # Skip non-pill images (chemical structures, diagrams, labels)
            if not is_pill_image(media_name):
                logger.debug(f"  Skipping non-pill image: {media_name}")
                continue

            # Determine file extension
            ext = ".jpg"
            if "png" in mime:
                ext = ".png"
            elif "gif" in mime:
                ext = ".gif"

            # Create unique filename
            url_hash = hashlib.md5(media_url_dl.encode()).hexdigest()[:10]
            safe_name = media_name.replace("/", "_").replace("\\", "_")[:30] if media_name else "img"
            filename = f"{drug_name}_{safe_name}_{url_hash}{ext}"
            filepath = os.path.join(drug_dir, filename)

            if os.path.exists(filepath):
                count += 1
                continue

            if download_file(media_url_dl, filepath):
                # Verify file is valid (at least 1KB)
                if os.path.getsize(filepath) < 1024:
                    os.remove(filepath)
                else:
                    count += 1

        if count >= min_images * 3:
            break  # Enough images for this drug

    return count


def split_train_val(raw_dir, output_dir, val_ratio=0.2, min_images=3):
    """
    Split raw images into train/val directories.
    Only includes classes with at least min_images images.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    total_train = 0
    total_val = 0
    classes = 0

    for drug_name in sorted(os.listdir(raw_dir)):
        drug_path = os.path.join(raw_dir, drug_name)
        if not os.path.isdir(drug_path):
            continue

        images = [
            f for f in os.listdir(drug_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
        ]

        if len(images) < min_images:
            logger.warning(f"  Skipping {drug_name}: only {len(images)} images (need {min_images})")
            continue

        # Deterministic shuffle based on drug name
        images.sort()

        val_count = max(1, int(len(images) * val_ratio))
        val_images = images[:val_count]
        train_images = images[val_count:]

        # Create directories
        train_drug_dir = os.path.join(train_dir, drug_name)
        val_drug_dir = os.path.join(val_dir, drug_name)
        os.makedirs(train_drug_dir, exist_ok=True)
        os.makedirs(val_drug_dir, exist_ok=True)

        for img in train_images:
            src = os.path.join(drug_path, img)
            dst = os.path.join(train_drug_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            total_train += 1

        for img in val_images:
            src = os.path.join(drug_path, img)
            dst = os.path.join(val_drug_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            total_val += 1

        classes += 1

    logger.info(f"Split complete: {classes} classes, {total_train} train, {total_val} val images")
    return classes, total_train, total_val


def download_epillid_metadata():
    """
    Download ePillID benchmark data info.
    Returns instructions for getting the full dataset.
    """
    info = """
    ePillID Dataset Download Instructions:
    =======================================
    
    The ePillID benchmark dataset contains ~13,000 pill images across 9,804 types.
    
    Option A - Direct from the paper's GitHub (recommended for Colab):
        git clone https://github.com/usuyama/ePillID-benchmark.git
        # Images are referenced from NIH Pill Image Recognition Challenge
    
    Option B - NIH C3PI Dataset (larger, reference quality):
        Download from: https://data.lhncbc.nlm.nih.gov/public/Pills/
        - PillProjectDisc/: Reference images (studio quality)
        - C3PI_Test/: Consumer-quality test images
        - C3PI_Reference/: High-resolution reference set
    
    Option C - Use the Colab notebook (RECOMMENDED):
        Open molecular-ai/web/models/pill_training_colab.ipynb
        It handles everything: download, preprocess, train, export.
    """
    return info


def main():
    parser = argparse.ArgumentParser(description="Download pill image data for training")
    parser.add_argument("--output-dir", default="./pill_data", help="Output directory")
    parser.add_argument(
        "--source", choices=["dailymed", "epillid", "all"], default="dailymed",
        help="Data source: dailymed (NLM DailyMed API), epillid (benchmark), or all"
    )
    parser.add_argument("--count", type=int, default=50, help="Number of drugs to download")
    parser.add_argument("--min-images", type=int, default=3, help="Minimum images per drug class")
    parser.add_argument("--split", action="store_true", help="Split into train/val after download")
    args = parser.parse_args()

    output_dir = args.output_dir
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if args.source in ("epillid", "all"):
        info = download_epillid_metadata()
        print(info)

    if args.source in ("dailymed", "all"):
        drugs = DRUG_LIST[:args.count]
        logger.info(f"Downloading DailyMed images for {len(drugs)} drugs...")

        total_images = 0
        for drug_name in drugs:
            logger.info(f"  Downloading {drug_name}...")
            try:
                n = download_dailymed_images(raw_dir, drug_name)
            except Exception as e:
                logger.warning(f"    -> failed ({e}), continuing")
                n = 0
            total_images += n
            logger.info(f"    -> {n} images")

        logger.info(f"Total images downloaded: {total_images}")

    if args.split:
        logger.info("Splitting into train/val...")
        split_train_val(raw_dir, output_dir, min_images=args.min_images)

    logger.info(f"Data saved to {output_dir}")
    logger.info("Next step: train with 'python train_pill_model.py --data-dir " + output_dir + "'")


if __name__ == "__main__":
    main()
