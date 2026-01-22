"""Simple debug script - check if both drugs are in abstracts"""
import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
sys.path.insert(0, '/app')

import django
django.setup()

from ddi_api.services.pubmed_retriever import PubMedRetriever

r = PubMedRetriever()
pmids = r.search_pmids("Aspirin", "Warfarin")
print(f"PMIDs: {pmids}")

abstracts = r.fetch_abstracts(pmids[:2])  # Just first 2
for ab in abstracts:
    text = ab['abstract']
    has_asp = 'aspirin' in text.lower()
    has_war = 'warfarin' in text.lower()
    print(f"\nPMID {ab['pmid']}:")
    print(f"  aspirin: {has_asp}, warfarin: {has_war}")
    if has_asp and has_war:
        print(f"  BOTH FOUND!")
    else:
        print(f"  Synonyms check - 'anticoagulant': {'anticoagulant' in text.lower()}")
        print(f"  Synonyms check - 'coumarin': {'coumarin' in text.lower()}")
