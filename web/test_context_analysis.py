"""
Comprehensive Context Analysis Test

This script analyzes how context sentences affect PubMedBERT predictions
to understand and fix the low/inaccurate scores issue.
"""
import os
import sys
import time
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')

import django
django.setup()

from ddi_api.services.pubmedbert_predictor import get_pubmedbert_predictor, PubMedBERTPredictor
from ddi_api.services.pubmed_retriever import get_retriever

def test_context_comparison():
    """Test how different context types affect predictions."""
    predictor = get_pubmedbert_predictor()
    
    if not predictor.is_loaded:
        print("ERROR: Model not loaded!")
        return
    
    # Test cases: known drug interactions with expected severity
    test_pairs = [
        ("Aspirin", "Warfarin", "severe", "bleeding risk"),
        ("Simvastatin", "Amiodarone", "severe", "rhabdomyolysis"),
        ("Metformin", "Alcohol", "moderate", "lactic acidosis"),
        ("Ibuprofen", "Lisinopril", "moderate", "reduced antihypertensive effect"),
        ("Metoprolol", "Verapamil", "severe", "bradycardia/heart block"),
    ]
    
    print("=" * 80)
    print("CONTEXT ANALYSIS: How Context Sentences Affect PubMedBERT Predictions")
    print("=" * 80)
    
    for drug1, drug2, expected_severity, reason in test_pairs:
        print(f"\n{'='*80}")
        print(f"Drug Pair: {drug1} + {drug2}")
        print(f"Expected: {expected_severity} ({reason})")
        print("-" * 80)
        
        # Test 1: Default behavior (uses RAG or template fallback)
        print("\n1. DEFAULT (RAG + Template Fallback):")
        result1 = predictor.predict(drug1, drug2)
        print(f"   Input: {result1.formatted_input[:100]}...")
        print(f"   Type: {result1.interaction_type}, Confidence: {result1.confidence:.2f}")
        print(f"   Risk: {result1.risk_score:.2f}, Severity: {result1.severity}")
        
        time.sleep(0.4)  # Avoid rate limiting
        
        # Test 2: Generic template only
        template = "When {drug1} and {drug2} are taken together, drug interactions may occur."
        context2 = template.format(drug1=drug1, drug2=drug2)
        print(f"\n2. GENERIC TEMPLATE ONLY:")
        result2 = predictor.predict(drug1, drug2, context=context2)
        print(f"   Input: {result2.formatted_input}")
        print(f"   Type: {result2.interaction_type}, Confidence: {result2.confidence:.2f}")
        print(f"   Risk: {result2.risk_score:.2f}, Severity: {result2.severity}")
        
        # Test 3: DDI Corpus-style sentence (mechanism-focused)
        mechanism_context = f"{drug1} inhibits the metabolism of {drug2} through CYP450 enzyme interaction, increasing plasma concentrations."
        print(f"\n3. MECHANISM-FOCUSED CONTEXT:")
        result3 = predictor.predict(drug1, drug2, context=mechanism_context)
        print(f"   Input: {result3.formatted_input}")
        print(f"   Type: {result3.interaction_type}, Confidence: {result3.confidence:.2f}")
        print(f"   Risk: {result3.risk_score:.2f}, Severity: {result3.severity}")
        
        # Test 4: Effect-focused sentence
        effect_context = f"Concurrent use of {drug1} with {drug2} significantly increases the risk of serious adverse effects and toxicity."
        print(f"\n4. EFFECT-FOCUSED CONTEXT:")
        result4 = predictor.predict(drug1, drug2, context=effect_context)
        print(f"   Input: {result4.formatted_input}")
        print(f"   Type: {result4.interaction_type}, Confidence: {result4.confidence:.2f}")
        print(f"   Risk: {result4.risk_score:.2f}, Severity: {result4.severity}")
        
        # Test 5: Advise-focused sentence
        advise_context = f"Patients taking {drug1} should avoid concomitant {drug2} or require careful monitoring and dose adjustment."
        print(f"\n5. ADVISE-FOCUSED CONTEXT:")
        result5 = predictor.predict(drug1, drug2, context=advise_context)
        print(f"   Input: {result5.formatted_input}")
        print(f"   Type: {result5.interaction_type}, Confidence: {result5.confidence:.2f}")
        print(f"   Risk: {result5.risk_score:.2f}, Severity: {result5.severity}")
        
        # Test 6: Multi-template aggregation
        print(f"\n6. MULTI-TEMPLATE AGGREGATION (all 4 templates):")
        result6 = predictor.predict_with_multiple_contexts(drug1, drug2)
        print(f"   Type: {result6.interaction_type}, Confidence: {result6.confidence:.2f}")
        print(f"   Risk: {result6.risk_score:.2f}, Severity: {result6.severity}")
        print(f"   Averaged probabilities: {result6.all_probabilities}")
        
        time.sleep(0.5)  # Avoid rate limiting between pairs

def test_retriever_debug():
    """Debug the PubMed retriever to see what contexts are being fetched."""
    print("\n" + "=" * 80)
    print("PUBMED RETRIEVER DEBUG")
    print("=" * 80)
    
    retriever = get_retriever()
    
    test_pairs = [
        ("Aspirin", "Warfarin"),
        ("Amiodarone", "Digoxin"),
    ]
    
    for drug1, drug2 in test_pairs:
        print(f"\n--- {drug1} + {drug2} ---")
        time.sleep(0.5)  # Rate limiting
        
        result = retriever.retrieve(drug1, drug2)
        if result:
            print(f"Source: {result.source}")
            print(f"PMID: {result.pmid}")
            print(f"Relevance: {result.relevance_score}")
            print(f"Sentence: {result.sentence}")
        else:
            print("No context retrieved - will use template fallback")

def test_specific_known_interactions():
    """Test drug pairs with known specific contexts to validate model behavior."""
    predictor = get_pubmedbert_predictor()
    
    print("\n" + "=" * 80)
    print("KNOWN INTERACTION CONTEXTS TEST")
    print("=" * 80)
    
    # These are DDI Corpus-style sentences that the model was trained on
    known_interactions = [
        {
            "drug1": "Warfarin",
            "drug2": "Aspirin", 
            "context": "The anticoagulant effect of warfarin is significantly enhanced when aspirin is coadministered, leading to increased risk of gastrointestinal bleeding.",
            "expected": "effect"
        },
        {
            "drug1": "Simvastatin",
            "drug2": "Erythromycin",
            "context": "Erythromycin inhibits CYP3A4, which metabolizes simvastatin, leading to increased statin plasma levels and risk of rhabdomyolysis.",
            "expected": "mechanism"
        },
        {
            "drug1": "Lisinopril",
            "drug2": "Potassium",
            "context": "Concurrent use of ACE inhibitors like lisinopril with potassium supplements may result in hyperkalemia. Serum potassium should be monitored.",
            "expected": "advise"
        },
        {
            "drug1": "Fluoxetine",
            "drug2": "Tramadol",
            "context": "There is a potential interaction between fluoxetine and tramadol due to serotonergic effects.",
            "expected": "int"
        },
        {
            "drug1": "Vitamin C",
            "drug2": "Vitamin D",
            "context": "No clinically significant drug interaction has been reported between vitamin C and vitamin D.",
            "expected": "no_interaction"
        },
    ]
    
    for case in known_interactions:
        print(f"\n{case['drug1']} + {case['drug2']} (Expected: {case['expected']})")
        print(f"Context: {case['context'][:80]}...")
        
        result = predictor.predict(case['drug1'], case['drug2'], context=case['context'])
        
        match = "✓" if result.interaction_type == case['expected'] else "✗"
        print(f"Result: {result.interaction_type} ({result.confidence:.2f}) {match}")
        print(f"All probs: {result.all_probabilities}")

if __name__ == "__main__":
    print("=" * 80)
    print("PubMedBERT Context Analysis Suite")
    print("=" * 80)
    
    # Run tests
    test_specific_known_interactions()
    print("\n\n")
    test_context_comparison()
    # test_retriever_debug()  # Uncomment to test retriever (may hit rate limits)
