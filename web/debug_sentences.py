"""Debug sentence extraction"""
import os, sys, re
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')
import django
django.setup()

from ddi_api.services.pubmed_retriever import PubMedRetriever

r = PubMedRetriever()
abstracts = r.fetch_abstracts(['33765420'])

if abstracts:
    text = abstracts[0]['abstract']
    print("=== Full Abstract ===")
    print(text[:1000])
    print("\n=== All sentences ===")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for i, s in enumerate(sentences):
        has_asp = 'aspirin' in s.lower()
        has_war = 'warfarin' in s.lower()
        marker = "***BOTH***" if (has_asp and has_war) else ""
        if has_asp or has_war:
            print(f"\n[{i}] {marker}")
            print(s[:200])
    
    print("\n=== Calling extract_relevant_sentence ===")
    result = r.extract_relevant_sentence(text, "Aspirin", "Warfarin")
    print(f"Result: {result}")
