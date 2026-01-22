"""Quick test of PubMedBERT predictor"""
import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

import django
django.setup()

from ddi_api.services.pubmedbert_predictor import get_pubmedbert_predictor

print("Testing PubMedBERT DDI Predictor...")
print("=" * 50)

predictor = get_pubmedbert_predictor()
print(f"Model loaded: {predictor.is_loaded}")
print(f"Device: {predictor.device}")

if predictor.is_loaded:
    print("\nTest 1: Aspirin + Warfarin")
    result = predictor.predict("Aspirin", "Warfarin")
    print(f"  Interaction type: {result.interaction_type}")
    print(f"  Risk score: {result.risk_score:.2f}")  
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Severity: {result.severity}")
    print(f"  Probabilities: {result.all_probabilities}")
    
    print("\nTest 2: Aspirin + Metformin")
    result2 = predictor.predict("Aspirin", "Metformin")
    print(f"  Interaction type: {result2.interaction_type}")
    print(f"  Risk score: {result2.risk_score:.2f}")
    print(f"  Confidence: {result2.confidence:.2f}")
    
    print("\nTest 3: Amiodarone + Digoxin (known severe interaction)")
    result3 = predictor.predict("Amiodarone", "Digoxin") 
    print(f"  Interaction type: {result3.interaction_type}")
    print(f"  Risk score: {result3.risk_score:.2f}")
    print(f"  Confidence: {result3.confidence:.2f}")
else:
    print("ERROR: Model failed to load!")
    print("Check that DDI_Model_Final exists at the expected path.")
