"""Debug script to test PubMed retrieval for Aspirin + Warfarin"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')
django.setup()

from ddi_api.services.pubmed_retriever import PubMedRetriever

def test_pubmed_retrieval():
    retriever = PubMedRetriever()
    
    drug1 = "Aspirin"
    drug2 = "Warfarin"
    
    print(f"=== Testing PubMed Retrieval for {drug1} + {drug2} ===\n")
    
    # Step 1: Search for PMIDs
    print("Step 1: Searching for PMIDs...")
    pmids = retriever.search_pmids(drug1, drug2)
    print(f"Found PMIDs: {pmids}\n")
    
    if not pmids:
        print("ERROR: No PMIDs found!")
        return
    
    # Step 2: Fetch abstracts
    print("Step 2: Fetching abstracts...")
    abstracts = retriever.fetch_abstracts(pmids)
    print(f"Fetched {len(abstracts)} abstracts\n")
    
    for i, abstract in enumerate(abstracts):
        print(f"--- Abstract {i+1} (PMID: {abstract['pmid']}) ---")
        print(f"Title: {abstract['title']}")
        print(f"Abstract preview: {abstract['abstract'][:300]}...")
        
        # Check if both drugs are mentioned
        abs_lower = abstract['abstract'].lower()
        has_aspirin = 'aspirin' in abs_lower
        has_warfarin = 'warfarin' in abs_lower
        print(f"Contains 'aspirin': {has_aspirin}")
        print(f"Contains 'warfarin': {has_warfarin}\n")
    
    # Step 3: Test full retrieval
    print("Step 3: Full retrieval test...")
    result = retriever.retrieve(drug1, drug2)
    print(f"Final result: {result}")

if __name__ == "__main__":
    test_pubmed_retrieval()
