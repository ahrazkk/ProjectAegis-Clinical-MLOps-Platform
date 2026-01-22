"""
Debug what PubMed returns for specific drug pairs.
"""
import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')

import django
django.setup()

from ddi_api.services.pubmed_retriever import PubMedRetriever
import re

def debug_pubmed_results():
    retriever = PubMedRetriever()
    
    test_pairs = [
        ("Aspirin", "Warfarin"),
        ("Amiodarone", "Digoxin"),
    ]
    
    for drug1, drug2 in test_pairs:
        print(f"\n{'='*80}")
        print(f"Debugging: {drug1} + {drug2}")
        print("=" * 80)
        
        # Search PubMed
        pmids = retriever.search_pmids(drug1, drug2)
        print(f"Found PMIDs: {pmids}")
        
        if not pmids:
            print("No results!")
            continue
        
        # Fetch abstracts
        abstracts = retriever.fetch_abstracts(pmids[:2])  # Just first 2
        
        for ab in abstracts:
            print(f"\n--- PMID {ab['pmid']}: {ab['title'][:60]}... ---")
            text = ab['abstract']
            
            # Check for drug mentions
            d1_lower = drug1.lower()
            d2_lower = drug2.lower()
            text_lower = text.lower()
            
            has_d1 = d1_lower in text_lower
            has_d2 = d2_lower in text_lower
            print(f"{drug1} found: {has_d1}")
            print(f"{drug2} found: {has_d2}")
            
            # Split into sentences and check each
            sentences = re.split(r'(?<=[.!?])\s+', text)
            print(f"\nSentences with BOTH drugs:")
            
            found_any = False
            for i, sent in enumerate(sentences):
                sent_lower = sent.lower()
                if d1_lower in sent_lower and d2_lower in sent_lower:
                    found_any = True
                    print(f"  [{i}] {sent[:200]}...")
            
            if not found_any:
                print("  NONE - drugs are in different sentences!")
                print("\nSentences mentioning either drug:")
                for i, sent in enumerate(sentences[:5]):
                    sent_lower = sent.lower()
                    if d1_lower in sent_lower or d2_lower in sent_lower:
                        marker = "D1" if d1_lower in sent_lower else "D2"
                        print(f"  [{i}][{marker}] {sent[:150]}...")

if __name__ == "__main__":
    debug_pubmed_results()
