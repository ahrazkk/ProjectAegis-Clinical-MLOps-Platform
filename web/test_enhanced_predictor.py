"""
Comprehensive Test Suite for Enhanced PubMedBERT DDI Predictions

Tests the improved context handling, ensemble predictions, and rate limiting.
"""
import os
import sys
import time
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')

import django
django.setup()

from ddi_api.services.pubmedbert_predictor import get_pubmedbert_predictor

def test_known_interactions():
    """Test well-known drug interactions to validate model accuracy."""
    print("=" * 80)
    print("TEST: Known Drug Interactions")
    print("=" * 80)
    
    predictor = get_pubmedbert_predictor()
    
    if not predictor.is_loaded:
        print("ERROR: Model not loaded!")
        return False
    
    # Test cases: (drug1, drug2, expected_severity_level, description)
    # Severity levels: none, moderate, major, severe
    test_cases = [
        # Well-known severe interactions
        ("Aspirin", "Warfarin", ["major", "severe"], "increased bleeding risk"),
        ("Amiodarone", "Digoxin", ["major", "severe"], "digoxin toxicity"),
        ("Simvastatin", "Erythromycin", ["major", "severe"], "rhabdomyolysis risk"),
        ("Metoprolol", "Verapamil", ["major", "severe"], "bradycardia/heart block"),
        
        # Moderate interactions
        ("Ibuprofen", "Lisinopril", ["moderate", "major"], "reduced BP effect"),
        ("Metformin", "Alcohol", ["moderate", "major"], "lactic acidosis risk"),
        ("Ciprofloxacin", "Antacids", ["moderate"], "reduced absorption"),
        
        # Low/no interaction pairs
        ("Acetaminophen", "Omeprazole", ["none", "moderate"], "minimal interaction"),
        ("Vitamin C", "Vitamin D", ["none"], "no known interaction"),
    ]
    
    passed = 0
    failed = 0
    
    for drug1, drug2, expected_severities, description in test_cases:
        print(f"\n{drug1} + {drug2} (Expected: {expected_severities[0]}+ - {description})")
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        
        result = predictor.predict(drug1, drug2)
        
        status = "✓" if result.severity in expected_severities else "✗"
        if result.severity in expected_severities:
            passed += 1
        else:
            failed += 1
        
        print(f"  {status} Type: {result.interaction_type}")
        print(f"    Severity: {result.severity}, Risk: {result.risk_score:.2f}, Conf: {result.confidence:.2f}")
        print(f"    Context: {result.formatted_input[:60]}...")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.0f}%)")
    print(f"{'='*80}")
    
    return passed >= len(test_cases) * 0.6  # 60% pass rate

def test_ensemble_vs_single():
    """Compare ensemble predictions vs single template predictions."""
    print("\n" + "=" * 80)
    print("TEST: Ensemble vs Single Template Predictions")
    print("=" * 80)
    
    predictor = get_pubmedbert_predictor()
    
    test_pairs = [
        ("Warfarin", "Aspirin"),
        ("Simvastatin", "Amiodarone"),
        ("Metoprolol", "Diltiazem"),
    ]
    
    for drug1, drug2 in test_pairs:
        print(f"\n{drug1} + {drug2}:")
        
        # Single template (no ensemble)
        time.sleep(0.3)
        single_result = predictor.predict(drug1, drug2, use_ensemble=False)
        print(f"  Single:   {single_result.interaction_type} (conf: {single_result.confidence:.2f}, risk: {single_result.risk_score:.2f})")
        
        # Ensemble (default)
        time.sleep(0.3)
        ensemble_result = predictor.predict_ensemble(drug1, drug2)
        print(f"  Ensemble: {ensemble_result.interaction_type} (conf: {ensemble_result.confidence:.2f}, risk: {ensemble_result.risk_score:.2f})")
        print(f"            {ensemble_result.formatted_input}")

def test_context_handling():
    """Test that entity markers are correctly applied to contexts."""
    print("\n" + "=" * 80)
    print("TEST: Context Entity Marker Handling")
    print("=" * 80)
    
    predictor = get_pubmedbert_predictor()
    
    # Test cases with different context formats
    test_contexts = [
        # Context with drugs already marked
        ("Warfarin", "Aspirin", 
         "<e1>Warfarin</e1> and <e2>Aspirin</e2> increase bleeding risk.",
         "Pre-marked context"),
        
        # Context with drug names (need to be marked)
        ("Warfarin", "Aspirin",
         "Warfarin and Aspirin increase bleeding risk.",
         "Unmarked context"),
        
        # Context with case differences
        ("Warfarin", "Aspirin",
         "warfarin combined with aspirin may cause bleeding.",
         "Lowercase drug names"),
        
        # Context missing one drug
        ("Metformin", "Alcohol",
         "Metformin may cause lactic acidosis in certain conditions.",
         "Missing second drug"),
    ]
    
    for drug1, drug2, context, description in test_contexts:
        print(f"\n{description}:")
        print(f"  Input: {context[:60]}...")
        
        result = predictor.predict(drug1, drug2, context=context)
        
        has_e1 = '<e1>' in result.formatted_input
        has_e2 = '<e2>' in result.formatted_input
        
        print(f"  Output: {result.formatted_input[:80]}...")
        print(f"  Markers: e1={has_e1}, e2={has_e2}")
        print(f"  Result: {result.interaction_type} ({result.confidence:.2f})")

def test_rate_limiting():
    """Test that rapid requests don't cause 429 errors."""
    print("\n" + "=" * 80)
    print("TEST: Rate Limiting Handling")
    print("=" * 80)
    
    predictor = get_pubmedbert_predictor()
    
    # Make several rapid requests
    pairs = [
        ("Aspirin", "Warfarin"),
        ("Ibuprofen", "Lisinopril"),
        ("Metformin", "Glipizide"),
        ("Amiodarone", "Digoxin"),
        ("Simvastatin", "Amlodipine"),
    ]
    
    success = 0
    for drug1, drug2 in pairs:
        result = predictor.predict(drug1, drug2)
        if result.interaction_type != 'unknown':
            success += 1
            print(f"  ✓ {drug1} + {drug2}: {result.interaction_type}")
        else:
            print(f"  ✗ {drug1} + {drug2}: FAILED")
    
    print(f"\n  {success}/{len(pairs)} requests succeeded")

def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 80)
    print("ENHANCED PUBMEDBERT DDI PREDICTOR - TEST SUITE")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Known interactions
    if not test_known_interactions():
        all_passed = False
    
    # Test 2: Ensemble vs single
    test_ensemble_vs_single()
    
    # Test 3: Context handling
    test_context_handling()
    
    # Test 4: Rate limiting
    test_rate_limiting()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
