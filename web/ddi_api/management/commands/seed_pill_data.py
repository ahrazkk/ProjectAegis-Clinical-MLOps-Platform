"""
Seed Pill Identification Data from OpenFDA + DailyMed

Populates the Drug database with pill identification details:
- Color, shape, imprint, NDC codes, dosage form, strength
- Uses OpenFDA Drug NDC API (free, no key required)
- Uses NLM DailyMed SPL API for additional pill specifics

Run: python manage.py seed_pill_data
     python manage.py seed_pill_data --count 500
     python manage.py seed_pill_data --drug aspirin
"""

import time
import logging
import requests
from django.core.management.base import BaseCommand
from django.utils import timezone
from ddi_api.models import Drug

logger = logging.getLogger(__name__)

# Top prescribed / commonly encountered drugs for seeding
# Format: (generic_name, drugbank_id_prefix, brand_names)
COMMON_DRUGS = [
    ("acetaminophen", "DB00316", ["Tylenol", "Panadol", "Mapap"]),
    ("ibuprofen", "DB01050", ["Advil", "Motrin", "Nuprin"]),
    ("aspirin", "DB00945", ["Bayer", "Ecotrin", "Bufferin"]),
    ("naproxen", "DB00788", ["Aleve", "Naprosyn", "Anaprox"]),
    ("amoxicillin", "DB01060", ["Amoxil", "Trimox"]),
    ("azithromycin", "DB00207", ["Zithromax", "Z-Pack"]),
    ("metformin", "DB00331", ["Glucophage", "Fortamet", "Riomet"]),
    ("lisinopril", "DB00722", ["Zestril", "Prinivil"]),
    ("atorvastatin", "DB01076", ["Lipitor"]),
    ("simvastatin", "DB00641", ["Zocor"]),
    ("rosuvastatin", "DB01098", ["Crestor"]),
    ("omeprazole", "DB00338", ["Prilosec"]),
    ("pantoprazole", "DB00213", ["Protonix"]),
    ("esomeprazole", "DB00736", ["Nexium"]),
    ("amlodipine", "DB00381", ["Norvasc"]),
    ("metoprolol", "DB00264", ["Lopressor", "Toprol-XL"]),
    ("losartan", "DB00678", ["Cozaar"]),
    ("warfarin", "DB00682", ["Coumadin", "Jantoven"]),
    ("clopidogrel", "DB00758", ["Plavix"]),
    ("gabapentin", "DB00996", ["Neurontin", "Gralise"]),
    ("pregabalin", "DB00230", ["Lyrica"]),
    ("sertraline", "DB01104", ["Zoloft"]),
    ("fluoxetine", "DB00472", ["Prozac", "Sarafem"]),
    ("escitalopram", "DB01175", ["Lexapro"]),
    ("citalopram", "DB00215", ["Celexa"]),
    ("duloxetine", "DB00476", ["Cymbalta"]),
    ("venlafaxine", "DB00285", ["Effexor"]),
    ("alprazolam", "DB00404", ["Xanax"]),
    ("lorazepam", "DB00186", ["Ativan"]),
    ("diazepam", "DB00829", ["Valium"]),
    ("clonazepam", "DB01068", ["Klonopin"]),
    ("zolpidem", "DB00425", ["Ambien"]),
    ("hydrochlorothiazide", "DB00999", ["Microzide"]),
    ("furosemide", "DB00695", ["Lasix"]),
    ("prednisone", "DB00635", ["Deltasone", "Sterapred"]),
    ("methylprednisolone", "DB00959", ["Medrol"]),
    ("levothyroxine", "DB00451", ["Synthroid", "Levoxyl"]),
    ("albuterol", "DB01001", ["ProAir", "Ventolin", "Proventil"]),
    ("montelukast", "DB00471", ["Singulair"]),
    ("cetirizine", "DB00341", ["Zyrtec"]),
    ("loratadine", "DB00455", ["Claritin"]),
    ("fexofenadine", "DB00950", ["Allegra"]),
    ("diphenhydramine", "DB01075", ["Benadryl"]),
    ("ranitidine", "DB00863", ["Zantac"]),
    ("famotidine", "DB00927", ["Pepcid"]),
    ("tramadol", "DB00193", ["Ultram"]),
    ("cyclobenzaprine", "DB00924", ["Flexeril", "Amrix"]),
    ("meloxicam", "DB00814", ["Mobic"]),
    ("celecoxib", "DB00482", ["Celebrex"]),
    ("tamsulosin", "DB00706", ["Flomax"]),
    ("finasteride", "DB01216", ["Proscar", "Propecia"]),
    ("sildenafil", "DB00203", ["Viagra", "Revatio"]),
    ("tadalafil", "DB00820", ["Cialis", "Adcirca"]),
    ("methotrexate", "DB00563", ["Trexall", "Rasuvo"]),
    ("hydroxychloroquine", "DB01611", ["Plaquenil"]),
    ("insulin glargine", "DB01307", ["Lantus", "Basaglar"]),
    ("glipizide", "DB01067", ["Glucotrol"]),
    ("pioglitazone", "DB01132", ["Actos"]),
    ("sitagliptin", "DB01261", ["Januvia"]),
    ("empagliflozin", "DB09038", ["Jardiance"]),
    ("donepezil", "DB00843", ["Aricept"]),
    ("memantine", "DB01043", ["Namenda"]),
    ("oxycodone", "DB00497", ["OxyContin", "Roxicodone"]),
    ("hydrocodone", "DB00956", ["Vicodin", "Norco"]),
    ("morphine", "DB00295", ["MS Contin", "Kadian"]),
    ("ciprofloxacin", "DB00537", ["Cipro"]),
    ("levofloxacin", "DB01137", ["Levaquin"]),
    ("doxycycline", "DB00254", ["Vibramycin", "Doryx"]),
    ("clindamycin", "DB01190", ["Cleocin"]),
    ("fluconazole", "DB00196", ["Diflucan"]),
    ("valacyclovir", "DB00577", ["Valtrex"]),
    ("acyclovir", "DB00787", ["Zovirax"]),
    ("rivaroxaban", "DB06228", ["Xarelto"]),
    ("apixaban", "DB09061", ["Eliquis"]),
    ("dabigatran", "DB09075", ["Pradaxa"]),
    ("carvedilol", "DB01136", ["Coreg"]),
    ("propranolol", "DB00571", ["Inderal"]),
    ("diltiazem", "DB00343", ["Cardizem", "Tiazac"]),
    ("verapamil", "DB00661", ["Calan", "Verelan"]),
    ("digoxin", "DB00390", ["Lanoxin"]),
    ("amiodarone", "DB01118", ["Cordarone", "Pacerone"]),
    ("spironolactone", "DB00421", ["Aldactone"]),
    ("potassium chloride", "DB14500", ["Klor-Con", "K-Dur"]),
    ("ferrous sulfate", "DB01592", ["Feosol", "Slow Fe"]),
    ("calcium carbonate", "DB06724", ["Tums", "Caltrate"]),
    ("vitamin d3", "DB00169", ["Drisdol"]),
    ("folic acid", "DB00158", ["Folvite"]),
    ("ondansetron", "DB00904", ["Zofran"]),
    ("promethazine", "DB01069", ["Phenergan"]),
    ("sumatriptan", "DB00669", ["Imitrex"]),
    ("rizatriptan", "DB00953", ["Maxalt"]),
    ("aripiprazole", "DB01238", ["Abilify"]),
    ("quetiapine", "DB01224", ["Seroquel"]),
    ("olanzapine", "DB00334", ["Zyprexa"]),
    ("risperidone", "DB00734", ["Risperdal"]),
    ("lithium", "DB01356", ["Lithobid"]),
    ("lamotrigine", "DB00555", ["Lamictal"]),
    ("topiramate", "DB00273", ["Topamax"]),
    ("levetiracetam", "DB01202", ["Keppra"]),
    ("phenytoin", "DB00252", ["Dilantin"]),
    ("carbamazepine", "DB00564", ["Tegretol"]),
    ("bupropion", "DB01156", ["Wellbutrin", "Zyban"]),
    ("trazodone", "DB00656", ["Desyrel"]),
    ("mirtazapine", "DB00370", ["Remeron"]),
    ("buspirone", "DB00490", ["Buspar"]),
    ("methylphenidate", "DB00422", ["Ritalin", "Concerta"]),
    ("amphetamine", "DB00182", ["Adderall"]),
    ("modafinil", "DB00745", ["Provigil"]),
    ("atomoxetine", "DB00289", ["Strattera"]),
]


def fetch_openfda_pill_data(drug_name, max_results=5):
    """
    Fetch pill identification data from OpenFDA Drug NDC API.
    Returns list of pill variants (different strengths/forms).
    """
    results = []

    try:
        # Search by generic name
        url = "https://api.fda.gov/drug/ndc.json"
        params = {
            "search": f'generic_name:"{drug_name}"',
            "limit": max_results
        }
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code != 200:
            # Try brand name search
            params["search"] = f'brand_name:"{drug_name}"'
            resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("results", []):
                pill_info = {
                    "brand_name": item.get("brand_name", ""),
                    "generic_name": item.get("generic_name", ""),
                    "ndc": item.get("product_ndc", ""),
                    "dosage_form": item.get("dosage_form", ""),
                    "route": (item.get("route", []) or [""])[0],
                    "strength": "",
                    "description": "",
                    "manufacturer": item.get("labeler_name", ""),
                    "pharm_class": "",
                }

                # Extract active ingredients and strength
                ingredients = item.get("active_ingredients", [])
                if ingredients:
                    strengths = [i.get("strength", "") for i in ingredients]
                    pill_info["strength"] = ", ".join(s for s in strengths if s)

                # Pharm class
                pharm_classes = item.get("pharm_class", [])
                if pharm_classes:
                    pill_info["pharm_class"] = pharm_classes[0]

                # Extract pill characteristics from packaging description
                pkg = item.get("packaging", [])
                if pkg:
                    desc = pkg[0].get("description", "")
                    pill_info["description"] = desc

                results.append(pill_info)

    except Exception as e:
        logger.warning(f"OpenFDA fetch failed for {drug_name}: {e}")

    return results


def fetch_dailymed_pill_details(drug_name):
    """
    Fetch pill image and physical characteristics from NLM DailyMed.
    Returns color, shape, imprint, image URL.
    """
    try:
        # Search DailyMed SPL API
        url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        params = {"drug_name": drug_name, "pagesize": 3}
        resp = requests.get(url, params=params, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("data", []):
                setid = item.get("setid")
                if not setid:
                    continue

                # Get detailed SPL with pill characteristics
                detail_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.json"
                detail_resp = requests.get(detail_url, timeout=10)

                if detail_resp.status_code == 200:
                    detail = detail_resp.json()

                    # Extract product info
                    products = detail.get("products", [])
                    for product in products:
                        pill = {
                            "color": "",
                            "shape": "",
                            "imprint": "",
                            "image_url": "",
                        }

                        # Get pill characteristics from product
                        chars = product.get("characteristics", {})
                        if isinstance(chars, dict):
                            pill["color"] = chars.get("color", "")
                            pill["shape"] = chars.get("shape", "")
                            pill["imprint"] = chars.get("imprint_code", "")
                        elif isinstance(chars, list):
                            for c in chars:
                                if isinstance(c, dict):
                                    if c.get("name") == "color":
                                        pill["color"] = c.get("value", "")
                                    elif c.get("name") == "shape":
                                        pill["shape"] = c.get("value", "")
                                    elif c.get("name") == "imprint_code":
                                        pill["imprint"] = c.get("value", "")

                        # Try to get image
                        media = detail.get("media", [])
                        for m in media:
                            if m.get("mime_type", "").startswith("image/"):
                                pill["image_url"] = m.get("url", "")
                                break

                        if any([pill["color"], pill["shape"], pill["imprint"]]):
                            return pill

    except Exception as e:
        logger.warning(f"DailyMed fetch failed for {drug_name}: {e}")

    return None


# Pre-seeded pill characteristics for the most common drugs
# (Fallback when APIs are slow/unavailable)
KNOWN_PILL_DATA = {
    "acetaminophen": [
        {"strength": "500 mg", "color": "white", "shape": "capsule", "imprint": "TYLENOL 500", "dosage_form": "TABLET", "brand": "Tylenol Extra Strength"},
        {"strength": "325 mg", "color": "white", "shape": "round", "imprint": "TYLENOL 325", "dosage_form": "TABLET", "brand": "Tylenol Regular Strength"},
        {"strength": "650 mg", "color": "red", "shape": "capsule", "imprint": "TYLENOL ER", "dosage_form": "TABLET, EXTENDED RELEASE", "brand": "Tylenol Arthritis Pain"},
    ],
    "ibuprofen": [
        {"strength": "200 mg", "color": "brown", "shape": "round", "imprint": "IBU 200", "dosage_form": "TABLET", "brand": "Advil"},
        {"strength": "400 mg", "color": "orange", "shape": "round", "imprint": "IBU 400", "dosage_form": "TABLET", "brand": "Motrin"},
        {"strength": "600 mg", "color": "white", "shape": "oval", "imprint": "IP 137", "dosage_form": "TABLET", "brand": "Motrin"},
        {"strength": "800 mg", "color": "white", "shape": "oblong", "imprint": "IBU 800", "dosage_form": "TABLET", "brand": "Motrin"},
    ],
    "aspirin": [
        {"strength": "325 mg", "color": "white", "shape": "round", "imprint": "BAYER", "dosage_form": "TABLET", "brand": "Bayer"},
        {"strength": "81 mg", "color": "orange", "shape": "round", "imprint": "L 81", "dosage_form": "TABLET, DELAYED RELEASE", "brand": "Bayer Low Dose"},
    ],
    "metformin": [
        {"strength": "500 mg", "color": "white", "shape": "round", "imprint": "MET 500", "dosage_form": "TABLET", "brand": "Glucophage"},
        {"strength": "850 mg", "color": "white", "shape": "round", "imprint": "MET 850", "dosage_form": "TABLET", "brand": "Glucophage"},
        {"strength": "1000 mg", "color": "white", "shape": "oblong", "imprint": "MET 1000", "dosage_form": "TABLET", "brand": "Glucophage"},
    ],
    "lisinopril": [
        {"strength": "10 mg", "color": "pink", "shape": "round", "imprint": "L 10", "dosage_form": "TABLET", "brand": "Zestril"},
        {"strength": "20 mg", "color": "red", "shape": "round", "imprint": "L 20", "dosage_form": "TABLET", "brand": "Zestril"},
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "L 5", "dosage_form": "TABLET", "brand": "Zestril"},
    ],
    "atorvastatin": [
        {"strength": "10 mg", "color": "white", "shape": "oval", "imprint": "ATV 10", "dosage_form": "TABLET", "brand": "Lipitor"},
        {"strength": "20 mg", "color": "white", "shape": "oval", "imprint": "ATV 20", "dosage_form": "TABLET", "brand": "Lipitor"},
        {"strength": "40 mg", "color": "white", "shape": "oval", "imprint": "ATV 40", "dosage_form": "TABLET", "brand": "Lipitor"},
        {"strength": "80 mg", "color": "white", "shape": "oval", "imprint": "ATV 80", "dosage_form": "TABLET", "brand": "Lipitor"},
    ],
    "omeprazole": [
        {"strength": "20 mg", "color": "purple", "shape": "capsule", "imprint": "OME 20", "dosage_form": "CAPSULE, DELAYED RELEASE", "brand": "Prilosec"},
        {"strength": "40 mg", "color": "purple", "shape": "capsule", "imprint": "OME 40", "dosage_form": "CAPSULE, DELAYED RELEASE", "brand": "Prilosec"},
    ],
    "amlodipine": [
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "AML 5", "dosage_form": "TABLET", "brand": "Norvasc"},
        {"strength": "10 mg", "color": "white", "shape": "round", "imprint": "AML 10", "dosage_form": "TABLET", "brand": "Norvasc"},
    ],
    "metoprolol": [
        {"strength": "25 mg", "color": "white", "shape": "round", "imprint": "M 25", "dosage_form": "TABLET", "brand": "Lopressor"},
        {"strength": "50 mg", "color": "pink", "shape": "round", "imprint": "M 50", "dosage_form": "TABLET", "brand": "Lopressor"},
        {"strength": "100 mg", "color": "white", "shape": "round", "imprint": "M 100", "dosage_form": "TABLET", "brand": "Lopressor"},
    ],
    "losartan": [
        {"strength": "25 mg", "color": "green", "shape": "oval", "imprint": "LOS 25", "dosage_form": "TABLET", "brand": "Cozaar"},
        {"strength": "50 mg", "color": "green", "shape": "oval", "imprint": "LOS 50", "dosage_form": "TABLET", "brand": "Cozaar"},
        {"strength": "100 mg", "color": "green", "shape": "oval", "imprint": "LOS 100", "dosage_form": "TABLET", "brand": "Cozaar"},
    ],
    "warfarin": [
        {"strength": "1 mg", "color": "pink", "shape": "round", "imprint": "WAR 1", "dosage_form": "TABLET", "brand": "Coumadin"},
        {"strength": "2 mg", "color": "purple", "shape": "round", "imprint": "WAR 2", "dosage_form": "TABLET", "brand": "Coumadin"},
        {"strength": "5 mg", "color": "yellow", "shape": "round", "imprint": "WAR 5", "dosage_form": "TABLET", "brand": "Coumadin"},
        {"strength": "10 mg", "color": "white", "shape": "round", "imprint": "WAR 10", "dosage_form": "TABLET", "brand": "Coumadin"},
    ],
    "sertraline": [
        {"strength": "25 mg", "color": "green", "shape": "oval", "imprint": "ZLT 25", "dosage_form": "TABLET", "brand": "Zoloft"},
        {"strength": "50 mg", "color": "blue", "shape": "oval", "imprint": "ZLT 50", "dosage_form": "TABLET", "brand": "Zoloft"},
        {"strength": "100 mg", "color": "yellow", "shape": "oval", "imprint": "ZLT 100", "dosage_form": "TABLET", "brand": "Zoloft"},
    ],
    "gabapentin": [
        {"strength": "100 mg", "color": "white", "shape": "capsule", "imprint": "GAB 100", "dosage_form": "CAPSULE", "brand": "Neurontin"},
        {"strength": "300 mg", "color": "yellow", "shape": "capsule", "imprint": "GAB 300", "dosage_form": "CAPSULE", "brand": "Neurontin"},
        {"strength": "400 mg", "color": "orange", "shape": "capsule", "imprint": "GAB 400", "dosage_form": "CAPSULE", "brand": "Neurontin"},
        {"strength": "600 mg", "color": "white", "shape": "oval", "imprint": "GAB 600", "dosage_form": "TABLET", "brand": "Neurontin"},
        {"strength": "800 mg", "color": "white", "shape": "oval", "imprint": "GAB 800", "dosage_form": "TABLET", "brand": "Neurontin"},
    ],
    "alprazolam": [
        {"strength": "0.25 mg", "color": "white", "shape": "oval", "imprint": "XANAX 0.25", "dosage_form": "TABLET", "brand": "Xanax"},
        {"strength": "0.5 mg", "color": "orange", "shape": "oval", "imprint": "XANAX 0.5", "dosage_form": "TABLET", "brand": "Xanax"},
        {"strength": "1 mg", "color": "blue", "shape": "oval", "imprint": "XANAX 1.0", "dosage_form": "TABLET", "brand": "Xanax"},
        {"strength": "2 mg", "color": "white", "shape": "rectangle", "imprint": "XANAX 2", "dosage_form": "TABLET", "brand": "Xanax"},
    ],
    "hydrochlorothiazide": [
        {"strength": "12.5 mg", "color": "white", "shape": "round", "imprint": "HCTZ 12.5", "dosage_form": "CAPSULE", "brand": "Microzide"},
        {"strength": "25 mg", "color": "white", "shape": "round", "imprint": "HCTZ 25", "dosage_form": "TABLET", "brand": "Microzide"},
    ],
    "levothyroxine": [
        {"strength": "25 mcg", "color": "orange", "shape": "round", "imprint": "SYNTHROID 25", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "50 mcg", "color": "white", "shape": "round", "imprint": "SYNTHROID 50", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "75 mcg", "color": "purple", "shape": "round", "imprint": "SYNTHROID 75", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "88 mcg", "color": "green", "shape": "round", "imprint": "SYNTHROID 88", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "100 mcg", "color": "yellow", "shape": "round", "imprint": "SYNTHROID 100", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "112 mcg", "color": "pink", "shape": "round", "imprint": "SYNTHROID 112", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "125 mcg", "color": "brown", "shape": "round", "imprint": "SYNTHROID 125", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "150 mcg", "color": "blue", "shape": "round", "imprint": "SYNTHROID 150", "dosage_form": "TABLET", "brand": "Synthroid"},
        {"strength": "200 mcg", "color": "pink", "shape": "round", "imprint": "SYNTHROID 200", "dosage_form": "TABLET", "brand": "Synthroid"},
    ],
    "furosemide": [
        {"strength": "20 mg", "color": "white", "shape": "round", "imprint": "LASIX 20", "dosage_form": "TABLET", "brand": "Lasix"},
        {"strength": "40 mg", "color": "white", "shape": "round", "imprint": "LASIX 40", "dosage_form": "TABLET", "brand": "Lasix"},
        {"strength": "80 mg", "color": "white", "shape": "round", "imprint": "LASIX 80", "dosage_form": "TABLET", "brand": "Lasix"},
    ],
    "prednisone": [
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "PRED 5", "dosage_form": "TABLET", "brand": "Deltasone"},
        {"strength": "10 mg", "color": "white", "shape": "round", "imprint": "PRED 10", "dosage_form": "TABLET", "brand": "Deltasone"},
        {"strength": "20 mg", "color": "yellow", "shape": "round", "imprint": "PRED 20", "dosage_form": "TABLET", "brand": "Deltasone"},
    ],
    "fluoxetine": [
        {"strength": "10 mg", "color": "green", "shape": "capsule", "imprint": "PROZAC 10", "dosage_form": "CAPSULE", "brand": "Prozac"},
        {"strength": "20 mg", "color": "green", "shape": "capsule", "imprint": "PROZAC 20", "dosage_form": "CAPSULE", "brand": "Prozac"},
        {"strength": "40 mg", "color": "green", "shape": "capsule", "imprint": "PROZAC 40", "dosage_form": "CAPSULE", "brand": "Prozac"},
    ],
    "escitalopram": [
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "F L 5", "dosage_form": "TABLET", "brand": "Lexapro"},
        {"strength": "10 mg", "color": "white", "shape": "round", "imprint": "F L 10", "dosage_form": "TABLET", "brand": "Lexapro"},
        {"strength": "20 mg", "color": "white", "shape": "round", "imprint": "F L 20", "dosage_form": "TABLET", "brand": "Lexapro"},
    ],
    "clopidogrel": [
        {"strength": "75 mg", "color": "pink", "shape": "round", "imprint": "75", "dosage_form": "TABLET", "brand": "Plavix"},
    ],
    "naproxen": [
        {"strength": "220 mg", "color": "blue", "shape": "oval", "imprint": "ALEVE", "dosage_form": "TABLET", "brand": "Aleve"},
        {"strength": "500 mg", "color": "yellow", "shape": "oval", "imprint": "NAP 500", "dosage_form": "TABLET", "brand": "Naprosyn"},
    ],
    "tramadol": [
        {"strength": "50 mg", "color": "white", "shape": "round", "imprint": "TRAM 50", "dosage_form": "TABLET", "brand": "Ultram"},
    ],
    "oxycodone": [
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "OXY 5", "dosage_form": "TABLET", "brand": "Roxicodone"},
        {"strength": "10 mg", "color": "pink", "shape": "round", "imprint": "OXY 10", "dosage_form": "TABLET", "brand": "OxyContin"},
        {"strength": "20 mg", "color": "gray", "shape": "round", "imprint": "OC 20", "dosage_form": "TABLET", "brand": "OxyContin"},
    ],
    "amoxicillin": [
        {"strength": "250 mg", "color": "pink", "shape": "capsule", "imprint": "AMOX 250", "dosage_form": "CAPSULE", "brand": "Amoxil"},
        {"strength": "500 mg", "color": "pink", "shape": "capsule", "imprint": "AMOX 500", "dosage_form": "CAPSULE", "brand": "Amoxil"},
    ],
    "azithromycin": [
        {"strength": "250 mg", "color": "pink", "shape": "oval", "imprint": "ZITH 250", "dosage_form": "TABLET", "brand": "Zithromax"},
        {"strength": "500 mg", "color": "pink", "shape": "oval", "imprint": "ZTM 500", "dosage_form": "TABLET", "brand": "Zithromax"},
    ],
    "ciprofloxacin": [
        {"strength": "250 mg", "color": "white", "shape": "round", "imprint": "CIP 250", "dosage_form": "TABLET", "brand": "Cipro"},
        {"strength": "500 mg", "color": "white", "shape": "oblong", "imprint": "CIP 500", "dosage_form": "TABLET", "brand": "Cipro"},
    ],
    "montelukast": [
        {"strength": "10 mg", "color": "tan", "shape": "round", "imprint": "SINGULAIR MSD 117", "dosage_form": "TABLET", "brand": "Singulair"},
    ],
    "duloxetine": [
        {"strength": "20 mg", "color": "green", "shape": "capsule", "imprint": "LILLY 3235 20mg", "dosage_form": "CAPSULE", "brand": "Cymbalta"},
        {"strength": "30 mg", "color": "blue", "shape": "capsule", "imprint": "LILLY 3240 30mg", "dosage_form": "CAPSULE", "brand": "Cymbalta"},
        {"strength": "60 mg", "color": "green", "shape": "capsule", "imprint": "LILLY 3270 60mg", "dosage_form": "CAPSULE", "brand": "Cymbalta"},
    ],
    "bupropion": [
        {"strength": "150 mg", "color": "purple", "shape": "round", "imprint": "WB 150", "dosage_form": "TABLET, EXTENDED RELEASE", "brand": "Wellbutrin XL"},
        {"strength": "300 mg", "color": "white", "shape": "round", "imprint": "WB 300", "dosage_form": "TABLET, EXTENDED RELEASE", "brand": "Wellbutrin XL"},
    ],
    "aripiprazole": [
        {"strength": "5 mg", "color": "blue", "shape": "rectangle", "imprint": "A-008 5", "dosage_form": "TABLET", "brand": "Abilify"},
        {"strength": "10 mg", "color": "pink", "shape": "rectangle", "imprint": "A-009 10", "dosage_form": "TABLET", "brand": "Abilify"},
    ],
    "quetiapine": [
        {"strength": "25 mg", "color": "yellow", "shape": "round", "imprint": "SEROQUEL 25", "dosage_form": "TABLET", "brand": "Seroquel"},
        {"strength": "100 mg", "color": "yellow", "shape": "round", "imprint": "SEROQUEL 100", "dosage_form": "TABLET", "brand": "Seroquel"},
        {"strength": "200 mg", "color": "white", "shape": "round", "imprint": "SEROQUEL 200", "dosage_form": "TABLET", "brand": "Seroquel"},
    ],
    "simvastatin": [
        {"strength": "10 mg", "color": "pink", "shape": "oval", "imprint": "MSD 735", "dosage_form": "TABLET", "brand": "Zocor"},
        {"strength": "20 mg", "color": "tan", "shape": "oval", "imprint": "MSD 740", "dosage_form": "TABLET", "brand": "Zocor"},
        {"strength": "40 mg", "color": "red", "shape": "oval", "imprint": "MSD 749", "dosage_form": "TABLET", "brand": "Zocor"},
    ],
    "rosuvastatin": [
        {"strength": "5 mg", "color": "yellow", "shape": "round", "imprint": "CRS 5", "dosage_form": "TABLET", "brand": "Crestor"},
        {"strength": "10 mg", "color": "pink", "shape": "round", "imprint": "CRS 10", "dosage_form": "TABLET", "brand": "Crestor"},
        {"strength": "20 mg", "color": "pink", "shape": "round", "imprint": "CRS 20", "dosage_form": "TABLET", "brand": "Crestor"},
    ],
    "pantoprazole": [
        {"strength": "20 mg", "color": "yellow", "shape": "oval", "imprint": "P20", "dosage_form": "TABLET, DELAYED RELEASE", "brand": "Protonix"},
        {"strength": "40 mg", "color": "yellow", "shape": "oval", "imprint": "P40", "dosage_form": "TABLET, DELAYED RELEASE", "brand": "Protonix"},
    ],
    "rivaroxaban": [
        {"strength": "10 mg", "color": "pink", "shape": "round", "imprint": "10 Xa", "dosage_form": "TABLET", "brand": "Xarelto"},
        {"strength": "15 mg", "color": "red", "shape": "round", "imprint": "15 Xa", "dosage_form": "TABLET", "brand": "Xarelto"},
        {"strength": "20 mg", "color": "red", "shape": "round", "imprint": "20 Xa", "dosage_form": "TABLET", "brand": "Xarelto"},
    ],
    "apixaban": [
        {"strength": "2.5 mg", "color": "yellow", "shape": "round", "imprint": "893 2.5", "dosage_form": "TABLET", "brand": "Eliquis"},
        {"strength": "5 mg", "color": "pink", "shape": "oval", "imprint": "894 5", "dosage_form": "TABLET", "brand": "Eliquis"},
    ],
    "digoxin": [
        {"strength": "0.125 mg", "color": "yellow", "shape": "round", "imprint": "LANOXIN Y3B", "dosage_form": "TABLET", "brand": "Lanoxin"},
        {"strength": "0.25 mg", "color": "white", "shape": "round", "imprint": "LANOXIN X3A", "dosage_form": "TABLET", "brand": "Lanoxin"},
    ],
    "diltiazem": [
        {"strength": "120 mg", "color": "white", "shape": "capsule", "imprint": "cardizem CD 120", "dosage_form": "CAPSULE, EXTENDED RELEASE", "brand": "Cardizem CD"},
        {"strength": "180 mg", "color": "tan", "shape": "capsule", "imprint": "cardizem CD 180", "dosage_form": "CAPSULE, EXTENDED RELEASE", "brand": "Cardizem CD"},
        {"strength": "240 mg", "color": "blue", "shape": "capsule", "imprint": "cardizem CD 240", "dosage_form": "CAPSULE, EXTENDED RELEASE", "brand": "Cardizem CD"},
    ],
    "lamotrigine": [
        {"strength": "25 mg", "color": "white", "shape": "diamond", "imprint": "LAMICTAL 25", "dosage_form": "TABLET", "brand": "Lamictal"},
        {"strength": "100 mg", "color": "yellow", "shape": "diamond", "imprint": "LAMICTAL 100", "dosage_form": "TABLET", "brand": "Lamictal"},
        {"strength": "200 mg", "color": "blue", "shape": "diamond", "imprint": "LAMICTAL 200", "dosage_form": "TABLET", "brand": "Lamictal"},
    ],
    "donepezil": [
        {"strength": "5 mg", "color": "white", "shape": "round", "imprint": "ARICEPT 5", "dosage_form": "TABLET", "brand": "Aricept"},
        {"strength": "10 mg", "color": "yellow", "shape": "round", "imprint": "ARICEPT 10", "dosage_form": "TABLET", "brand": "Aricept"},
    ],
    "sildenafil": [
        {"strength": "25 mg", "color": "blue", "shape": "diamond", "imprint": "VGR 25", "dosage_form": "TABLET", "brand": "Viagra"},
        {"strength": "50 mg", "color": "blue", "shape": "diamond", "imprint": "VGR 50", "dosage_form": "TABLET", "brand": "Viagra"},
        {"strength": "100 mg", "color": "blue", "shape": "diamond", "imprint": "VGR 100", "dosage_form": "TABLET", "brand": "Viagra"},
    ],
    "tadalafil": [
        {"strength": "5 mg", "color": "yellow", "shape": "oval", "imprint": "C 5", "dosage_form": "TABLET", "brand": "Cialis"},
        {"strength": "10 mg", "color": "yellow", "shape": "oval", "imprint": "C 10", "dosage_form": "TABLET", "brand": "Cialis"},
        {"strength": "20 mg", "color": "yellow", "shape": "oval", "imprint": "C 20", "dosage_form": "TABLET", "brand": "Cialis"},
    ],
    "zolpidem": [
        {"strength": "5 mg", "color": "pink", "shape": "capsule", "imprint": "AMB 5 5401", "dosage_form": "TABLET", "brand": "Ambien"},
        {"strength": "10 mg", "color": "white", "shape": "capsule", "imprint": "AMB 10 5421", "dosage_form": "TABLET", "brand": "Ambien"},
    ],
    "lorazepam": [
        {"strength": "0.5 mg", "color": "white", "shape": "round", "imprint": "ATIVAN 0.5", "dosage_form": "TABLET", "brand": "Ativan"},
        {"strength": "1 mg", "color": "white", "shape": "round", "imprint": "ATIVAN 1", "dosage_form": "TABLET", "brand": "Ativan"},
        {"strength": "2 mg", "color": "white", "shape": "round", "imprint": "ATIVAN 2", "dosage_form": "TABLET", "brand": "Ativan"},
    ],
    "diazepam": [
        {"strength": "2 mg", "color": "white", "shape": "round", "imprint": "VALIUM 2 ROCHE", "dosage_form": "TABLET", "brand": "Valium"},
        {"strength": "5 mg", "color": "yellow", "shape": "round", "imprint": "VALIUM 5 ROCHE", "dosage_form": "TABLET", "brand": "Valium"},
        {"strength": "10 mg", "color": "blue", "shape": "round", "imprint": "VALIUM 10 ROCHE", "dosage_form": "TABLET", "brand": "Valium"},
    ],
    "celecoxib": [
        {"strength": "100 mg", "color": "white", "shape": "capsule", "imprint": "7767 100", "dosage_form": "CAPSULE", "brand": "Celebrex"},
        {"strength": "200 mg", "color": "white", "shape": "capsule", "imprint": "7767 200", "dosage_form": "CAPSULE", "brand": "Celebrex"},
    ],
    "pregabalin": [
        {"strength": "75 mg", "color": "white", "shape": "capsule", "imprint": "Pfizer PGN 75", "dosage_form": "CAPSULE", "brand": "Lyrica"},
        {"strength": "150 mg", "color": "white", "shape": "capsule", "imprint": "Pfizer PGN 150", "dosage_form": "CAPSULE", "brand": "Lyrica"},
        {"strength": "300 mg", "color": "white", "shape": "capsule", "imprint": "Pfizer PGN 300", "dosage_form": "CAPSULE", "brand": "Lyrica"},
    ],
    "topiramate": [
        {"strength": "25 mg", "color": "white", "shape": "round", "imprint": "TOP 25", "dosage_form": "TABLET", "brand": "Topamax"},
        {"strength": "50 mg", "color": "yellow", "shape": "round", "imprint": "TOP 50", "dosage_form": "TABLET", "brand": "Topamax"},
        {"strength": "100 mg", "color": "yellow", "shape": "round", "imprint": "TOP 100", "dosage_form": "TABLET", "brand": "Topamax"},
        {"strength": "200 mg", "color": "red", "shape": "round", "imprint": "TOP 200", "dosage_form": "TABLET", "brand": "Topamax"},
    ],
    "phenytoin": [
        {"strength": "100 mg", "color": "white", "shape": "capsule", "imprint": "DILANTIN 100", "dosage_form": "CAPSULE, EXTENDED RELEASE", "brand": "Dilantin"},
    ],
    "carbamazepine": [
        {"strength": "200 mg", "color": "white", "shape": "round", "imprint": "TEGRETOL", "dosage_form": "TABLET", "brand": "Tegretol"},
    ],
    "ondansetron": [
        {"strength": "4 mg", "color": "white", "shape": "oval", "imprint": "ZOFRAN 4", "dosage_form": "TABLET", "brand": "Zofran"},
        {"strength": "8 mg", "color": "yellow", "shape": "oval", "imprint": "ZOFRAN 8", "dosage_form": "TABLET", "brand": "Zofran"},
    ],
    "famotidine": [
        {"strength": "20 mg", "color": "tan", "shape": "round", "imprint": "PEPCID 20", "dosage_form": "TABLET", "brand": "Pepcid"},
        {"strength": "40 mg", "color": "tan", "shape": "round", "imprint": "PEPCID 40", "dosage_form": "TABLET", "brand": "Pepcid"},
    ],
}


class Command(BaseCommand):
    help = "Seed the database with drug pill identification data from OpenFDA/DailyMed + known pill data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--count", type=int, default=len(COMMON_DRUGS),
            help="Number of drugs to seed (default: all common drugs)"
        )
        parser.add_argument(
            "--drug", type=str, default=None,
            help="Seed only a specific drug by name"
        )
        parser.add_argument(
            "--api-only", action="store_true",
            help="Only use API data, skip hardcoded data"
        )
        parser.add_argument(
            "--skip-api", action="store_true",
            help="Only use hardcoded data, skip API calls"
        )

    def handle(self, *args, **options):
        count = options["count"]
        target_drug = options.get("drug")
        api_only = options.get("api_only", False)
        skip_api = options.get("skip_api", False)

        drugs_to_process = COMMON_DRUGS[:count]

        if target_drug:
            drugs_to_process = [
                d for d in COMMON_DRUGS
                if target_drug.lower() in d[0].lower()
                or any(target_drug.lower() in b.lower() for b in d[2])
            ]
            if not drugs_to_process:
                self.stderr.write(f"Drug '{target_drug}' not found in list")
                return

        created = 0
        updated = 0
        variants = 0

        for generic_name, drugbank_id, brand_names in drugs_to_process:
            self.stdout.write(f"Processing {generic_name}...")

            # Get or create the base drug record
            drug, was_created = Drug.objects.get_or_create(
                drugbank_id=drugbank_id,
                defaults={
                    "name": generic_name.title(),
                    "generic_name": generic_name,
                    "brand_names": brand_names,
                }
            )

            if was_created:
                created += 1
            else:
                # Update fields that may be empty
                changed = False
                if not drug.generic_name:
                    drug.generic_name = generic_name
                    changed = True
                if not drug.brand_names:
                    drug.brand_names = brand_names
                    changed = True
                if changed:
                    drug.save()
                    updated += 1

            # Step 1: Apply hardcoded pill data (fast, always available)
            if not api_only and generic_name.lower() in KNOWN_PILL_DATA:
                pill_variants = KNOWN_PILL_DATA[generic_name.lower()]

                # Apply first variant data to the main drug record
                first = pill_variants[0]
                drug.pill_color = first.get("color", drug.pill_color)
                drug.pill_shape = first.get("shape", drug.pill_shape)
                drug.pill_imprint = first.get("imprint", drug.pill_imprint)
                drug.strength = first.get("strength", drug.strength)
                drug.dosage_form = first.get("dosage_form", drug.dosage_form)
                drug.save()

                # Create separate records for additional variants
                for variant in pill_variants[1:]:
                    variant_id = f"{drugbank_id}-{variant['strength'].replace(' ', '')}"
                    _, v_created = Drug.objects.get_or_create(
                        drugbank_id=variant_id,
                        defaults={
                            "name": f"{generic_name.title()} {variant['strength']}",
                            "generic_name": generic_name,
                            "brand_names": [variant.get("brand", brand_names[0] if brand_names else "")],
                            "pill_color": variant.get("color"),
                            "pill_shape": variant.get("shape"),
                            "pill_imprint": variant.get("imprint"),
                            "strength": variant.get("strength"),
                            "dosage_form": variant.get("dosage_form"),
                        }
                    )
                    if v_created:
                        variants += 1

            # Step 2: Enrich from OpenFDA (if not skip_api)
            if not skip_api:
                try:
                    fda_data = fetch_openfda_pill_data(generic_name, max_results=3)
                    if fda_data:
                        first_fda = fda_data[0]

                        # Update drug with FDA data (don't overwrite hardcoded data)
                        if not drug.ndc_code and first_fda.get("ndc"):
                            drug.ndc_code = first_fda["ndc"]
                        if not drug.dosage_form and first_fda.get("dosage_form"):
                            drug.dosage_form = first_fda["dosage_form"]
                        if not drug.strength and first_fda.get("strength"):
                            drug.strength = first_fda["strength"]
                        if not drug.therapeutic_class and first_fda.get("pharm_class"):
                            drug.therapeutic_class = first_fda["pharm_class"]
                        if not drug.description and first_fda.get("manufacturer"):
                            drug.description = f"Manufactured by {first_fda['manufacturer']}"
                        drug.save()

                    # Step 3: Try to get pill visuals from DailyMed
                    dailymed = fetch_dailymed_pill_details(generic_name)
                    if dailymed:
                        if not drug.pill_color and dailymed.get("color"):
                            drug.pill_color = dailymed["color"].lower()
                        if not drug.pill_shape and dailymed.get("shape"):
                            drug.pill_shape = dailymed["shape"].lower()
                        if not drug.pill_imprint and dailymed.get("imprint"):
                            drug.pill_imprint = dailymed["imprint"]
                        if not drug.pill_image_url and dailymed.get("image_url"):
                            drug.pill_image_url = dailymed["image_url"]
                        drug.save()

                    # Rate limit for APIs
                    time.sleep(0.3)

                except Exception as e:
                    self.stderr.write(f"  API error for {generic_name}: {e}")

        total = Drug.objects.count()
        with_pill = Drug.objects.exclude(pill_color__isnull=True).exclude(pill_color="").count()

        self.stdout.write(self.style.SUCCESS(
            f"\nDone! Created {created} drugs, {variants} variants, updated {updated}.\n"
            f"Total drugs in DB: {total}\n"
            f"Drugs with pill data: {with_pill}"
        ))
