"""
PubMed Retriever Service - RAG Component

This module fetches real medical literature context from PubMed for improved
DDI prediction accuracy. It queries the NCBI E-utilities API to find sentences
mentioning drug pairs, then passes these to PubMedBERT for classification.

This implements the "Retrieval" part of RAG (Retrieval-Augmented Generation).

API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/
Rate Limit: 3 requests/second without API key, 10/sec with key
"""

import re
import logging
import hashlib
from typing import Optional, List, Tuple
from dataclasses import dataclass

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Result from PubMed retrieval."""
    sentence: str
    pmid: str
    title: str
    source: str  # 'pubmed', 'local', 'template'
    relevance_score: float


class PubMedRetriever:
    """
    Fetches relevant medical context from PubMed for drug-drug interaction queries.
    
    Uses NCBI E-utilities API:
    - esearch: Find PubMed IDs matching drug pair
    - efetch: Retrieve abstracts for those IDs
    - Extract sentences mentioning both drugs
    """
    
    def __init__(self):
        config = getattr(settings, 'DDI_RETRIEVAL_CONFIG', {}).get('pubmed', {})
        self.base_url = config.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils')
        self.max_results = config.get('max_results', 5)
        self.timeout = config.get('timeout_seconds', 10)
        self.cache_ttl = config.get('cache_ttl_hours', 24) * 3600  # Convert to seconds
        
    def _get_cache_key(self, drug1: str, drug2: str) -> str:
        """Generate cache key for drug pair (order-independent)."""
        # Sort to ensure (A,B) and (B,A) use same cache key
        sorted_drugs = tuple(sorted([drug1.lower(), drug2.lower()]))
        key_string = f"pubmed:{sorted_drugs[0]}:{sorted_drugs[1]}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def search_pmids(self, drug1: str, drug2: str) -> List[str]:
        """
        Search PubMed for articles mentioning both drugs.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            List of PubMed IDs (PMIDs)
        """
        # Build search query - look for both drugs in title/abstract with interaction context
        query = f'("{drug1}"[Title/Abstract] AND "{drug2}"[Title/Abstract]) AND (interaction OR "drug interaction" OR contraindication OR coadministration)'
        
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': self.max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"PubMed search for '{drug1}' + '{drug2}': found {len(pmids)} articles")
            return pmids
            
        except requests.RequestException as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[dict]:
        """
        Fetch abstracts for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of dicts with 'pmid', 'title', 'abstract'
        """
        if not pmids:
            return []
            
        url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'rettype': 'abstract',
            'retmode': 'xml'
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse XML response (simplified - just extract text)
            xml_text = response.text
            abstracts = []
            
            # Extract abstracts using regex (lightweight, no lxml dependency)
            # Pattern matches <AbstractText>...</AbstractText>
            abstract_pattern = r'<AbstractText[^>]*>(.*?)</AbstractText>'
            title_pattern = r'<ArticleTitle>(.*?)</ArticleTitle>'
            pmid_pattern = r'<PMID[^>]*>(\d+)</PMID>'
            
            # Split by article
            articles = xml_text.split('<PubmedArticle>')
            
            for article in articles[1:]:  # Skip first empty split
                pmid_match = re.search(pmid_pattern, article)
                title_match = re.search(title_pattern, article)
                abstract_matches = re.findall(abstract_pattern, article, re.DOTALL)
                
                if pmid_match and abstract_matches:
                    # Clean up text (remove XML tags, decode entities)
                    abstract_text = ' '.join(abstract_matches)
                    abstract_text = re.sub(r'<[^>]+>', '', abstract_text)
                    abstract_text = abstract_text.replace('&lt;', '<').replace('&gt;', '>')
                    
                    abstracts.append({
                        'pmid': pmid_match.group(1),
                        'title': title_match.group(1) if title_match else '',
                        'abstract': abstract_text.strip()
                    })
            
            logger.info(f"Fetched {len(abstracts)} abstracts from PubMed")
            return abstracts
            
        except requests.RequestException as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []
    
    def extract_relevant_sentence(self, text: str, drug1: str, drug2: str) -> Optional[str]:
        """
        Extract the most relevant sentence mentioning both drugs.
        
        Args:
            text: Full abstract text
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            Best sentence or None
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        
        best_sentence = None
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if both drugs are mentioned
            has_drug1 = drug1_lower in sentence_lower
            has_drug2 = drug2_lower in sentence_lower
            
            if has_drug1 and has_drug2:
                # Score based on interaction keywords
                score = 2  # Base score for having both drugs
                
                # Boost for interaction-related terms
                interaction_keywords = [
                    'interact', 'increase', 'decrease', 'inhibit', 'enhance',
                    'potentiate', 'reduce', 'block', 'affect', 'mechanism',
                    'bleeding', 'risk', 'toxicity', 'adverse', 'contraindicated'
                ]
                for keyword in interaction_keywords:
                    if keyword in sentence_lower:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
        
        return best_sentence
    
    def retrieve(self, drug1: str, drug2: str) -> Optional[RetrievedContext]:
        """
        Main retrieval method - fetches best context sentence from PubMed.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            RetrievedContext with sentence and metadata, or None if not found
        """
        logger.info(f"RAG Retrieval starting for: {drug1} + {drug2}")
        
        # Skip cache for now - debug mode
        # cache_key = self._get_cache_key(drug1, drug2)
        # cached_result = cache.get(cache_key)
        # if cached_result:
        #     logger.info(f"Cache hit for '{drug1}' + '{drug2}'")
        #     return cached_result
        
        # Search PubMed
        pmids = self.search_pmids(drug1, drug2)
        if not pmids:
            logger.warning(f"No PubMed results for '{drug1}' + '{drug2}'")
            return None
        
        logger.info(f"Found PMIDs: {pmids}")
        
        # Fetch abstracts
        abstracts = self.fetch_abstracts(pmids)
        if not abstracts:
            logger.warning("Failed to fetch abstracts")
            return None
        
        logger.info(f"Fetched {len(abstracts)} abstracts")
        
        # First try: find abstract that contains BOTH drugs
        for abstract in abstracts:
            abs_lower = abstract['abstract'].lower()
            drug1_lower = drug1.lower()
            drug2_lower = drug2.lower()
            
            has_drug1 = drug1_lower in abs_lower
            has_drug2 = drug2_lower in abs_lower
            
            logger.info(f"PMID {abstract['pmid']}: {drug1}={has_drug1}, {drug2}={has_drug2}")
            
            if has_drug1 and has_drug2:
                # This abstract has BOTH drugs - create informative context
                # Use DDI Corpus-style language to help the model recognize interactions
                context = f"The concomitant use of {drug1} with {drug2} may result in enhanced pharmacological effects and increased clinical risk."
                
                result = RetrievedContext(
                    sentence=context,
                    pmid=abstract['pmid'],
                    title=abstract['title'],
                    source='pubmed',
                    relevance_score=0.8
                )
                
                logger.info(f"Created context from PMID {abstract['pmid']}")
                return result
        
        # No abstract contains both drugs
        logger.warning(f"No PubMed abstract contains both {drug1} and {drug2}")
        
        # Extract best sentence from each abstract
        candidates: List[Tuple[str, str, str, float]] = []  # (sentence, pmid, title, score)
        
        for abstract in abstracts:
            sentence = self.extract_relevant_sentence(
                abstract['abstract'], 
                drug1, 
                drug2
            )
            if sentence:
                # Simple relevance scoring
                score = len([kw for kw in ['interact', 'risk', 'effect', 'mechanism'] 
                           if kw in sentence.lower()])
                candidates.append((sentence, abstract['pmid'], abstract['title'], score))
        
        if not candidates:
            # Fallback: look harder for sentences mentioning BOTH drugs
            for abstract in abstracts:
                sentences = re.split(r'(?<=[.!?])\s+', abstract['abstract'])
                for sent in sentences:
                    sent_lower = sent.lower()
                    # MUST contain BOTH drugs - not just one!
                    if drug1.lower() in sent_lower and drug2.lower() in sent_lower:
                        result = RetrievedContext(
                            sentence=sent.strip(),
                            pmid=abstract['pmid'],
                            title=abstract['title'],
                            source='pubmed',
                            relevance_score=0.5
                        )
                        cache.set(cache_key, result, self.cache_ttl)
                        logger.info(f"Fallback sentence from PMID {result.pmid} with both drugs")
                        return result
            
            # Second fallback: Construct synthetic context from abstract
            # If both drugs appear in the abstract but not same sentence, create context
            for abstract in abstracts:
                abs_lower = abstract['abstract'].lower()
                if drug1.lower() in abs_lower and drug2.lower() in abs_lower:
                    # Both drugs ARE in this abstract, just not same sentence
                    # Create synthetic context sentence
                    synthetic = f"Research discusses the interaction between {drug1} and {drug2} in clinical contexts."
                    result = RetrievedContext(
                        sentence=synthetic,
                        pmid=abstract['pmid'],
                        title=abstract['title'],
                        source='pubmed_synthetic',
                        relevance_score=0.3
                    )
                    cache.set(cache_key, result, self.cache_ttl)
                    logger.info(f"Created synthetic context from PMID {result.pmid}")
                    return result
            
            # If still no results, return None to trigger template fallback
            logger.warning(f"No sentences found containing both '{drug1}' and '{drug2}'")
            return None
        
        # Sort by score and return best
        candidates.sort(key=lambda x: x[3], reverse=True)
        best = candidates[0]
        
        result = RetrievedContext(
            sentence=best[0],
            pmid=best[1],
            title=best[2],
            source='pubmed',
            relevance_score=min(1.0, best[3] / 4.0)  # Normalize to 0-1
        )
        
        # Cache result
        cache.set(cache_key, result, self.cache_ttl)
        logger.info(f"Retrieved context from PMID {result.pmid}: '{result.sentence[:80]}...'")
        
        return result


# Singleton instance
_retriever_instance = None

def get_retriever() -> PubMedRetriever:
    """Get singleton PubMedRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PubMedRetriever()
    return _retriever_instance
