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
import time


class PubMedRetriever:
    """
    Fetches relevant medical context from PubMed for drug-drug interaction queries.
    
    Uses NCBI E-utilities API:
    - esearch: Find PubMed IDs matching drug pair
    - efetch: Retrieve abstracts for those IDs
    - Extract sentences mentioning both drugs
    
    Includes rate limiting handling with exponential backoff for 429 errors.
    """
    
    # Rate limiting - NCBI allows 3 requests/sec without API key
    _last_request_time = 0
    _min_request_interval = 0.4  # 400ms between requests to stay under limit
    
    def __init__(self):
        config = getattr(settings, 'DDI_RETRIEVAL_CONFIG', {}).get('pubmed', {})
        self.base_url = config.get('base_url', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils')
        self.max_results = config.get('max_results', 5)
        self.timeout = config.get('timeout_seconds', 10)
        self.cache_ttl = config.get('cache_ttl_hours', 24) * 3600  # Convert to seconds
        self.max_retries = 3
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - PubMedRetriever._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        PubMedRetriever._last_request_time = time.time()
    
    def _make_request_with_retry(self, url: str, params: dict) -> Optional[requests.Response]:
        """
        Make HTTP request with exponential backoff retry for rate limiting.
        
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(self.max_retries):
            self._rate_limit_wait()
            
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logger.warning(f"PubMed rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    logger.warning(f"PubMed request failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"PubMed request failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None
        
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
        
        response = self._make_request_with_retry(url, params)
        if response is None:
            return []
        
        try:
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"PubMed search for '{drug1}' + '{drug2}': found {len(pmids)} articles")
            return pmids
        except Exception as e:
            logger.error(f"Failed to parse PubMed search response: {e}")
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
        
        response = self._make_request_with_retry(url, params)
        if response is None:
            return []
        
        try:
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
            
        except Exception as e:
            logger.error(f"PubMed fetch parsing failed: {e}")
            return []
    
    def extract_relevant_sentence(self, text: str, drug1: str, drug2: str) -> Optional[str]:
        """
        Extract the most relevant sentence mentioning both drugs.
        
        Args:
            text: Full abstract text
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            Tuple of (best_sentence, score) or (None, 0) if not found
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        
        best_sentence = None
        best_score = 0
        
        # Strong interaction keywords that indicate actual DDI
        strong_keywords = [
            'interact', 'interaction', 'potentiate', 'inhibit', 'enhance',
            'contraindicated', 'concurrent', 'concomitant', 'coadminister',
            'combination', 'combined', 'adverse', 'toxicity', 'bleeding',
            'increase', 'decrease', 'reduce', 'elevate', 'affect', 'alter'
        ]
        
        # Weak keywords - indicate possible clinical relevance
        weak_keywords = [
            'risk', 'effect', 'mechanism', 'pharmacokinetic', 'pharmacodynamic',
            'metabolism', 'clearance', 'concentration', 'plasma', 'serum'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if both drugs are mentioned
            has_drug1 = drug1_lower in sentence_lower
            has_drug2 = drug2_lower in sentence_lower
            
            if has_drug1 and has_drug2:
                # Score based on interaction keywords
                score = 2  # Base score for having both drugs
                
                # Strong keywords add more weight
                for keyword in strong_keywords:
                    if keyword in sentence_lower:
                        score += 2
                
                # Weak keywords add less weight
                for keyword in weak_keywords:
                    if keyword in sentence_lower:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
        
        return (best_sentence, best_score) if best_sentence else (None, 0)
    
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
        
        # Score threshold - sentences must have at least this score to be considered
        # Score of 4+ means: both drugs (2) + at least one strong keyword (2)
        MIN_INTERACTION_SCORE = 4
        
        # Collect all candidate sentences from all abstracts
        all_candidates = []
        
        for abstract in abstracts:
            abs_lower = abstract['abstract'].lower()
            drug1_lower = drug1.lower()
            drug2_lower = drug2.lower()
            
            has_drug1 = drug1_lower in abs_lower
            has_drug2 = drug2_lower in abs_lower
            
            logger.info(f"PMID {abstract['pmid']}: {drug1}={has_drug1}, {drug2}={has_drug2}")
            
            if has_drug1 and has_drug2:
                # Extract the best sentence from this abstract
                best_sentence, score = self.extract_relevant_sentence(
                    abstract['abstract'], 
                    drug1, 
                    drug2
                )
                
                if best_sentence and score >= MIN_INTERACTION_SCORE:
                    all_candidates.append({
                        'sentence': best_sentence,
                        'score': score,
                        'pmid': abstract['pmid'],
                        'title': abstract['title']
                    })
                    logger.info(f"PMID {abstract['pmid']}: Found sentence with score {score}")
        
        # Sort by score and return the best one
        if all_candidates:
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            best = all_candidates[0]
            
            # Normalize score to relevance (score of 10+ = 1.0)
            relevance = min(1.0, best['score'] / 10.0)
            
            result = RetrievedContext(
                sentence=best['sentence'],
                pmid=best['pmid'],
                title=best['title'],
                source='pubmed',
                relevance_score=relevance
            )
            logger.info(f"Best sentence from PMID {best['pmid']} (score {best['score']}): '{best['sentence'][:80]}...'")
            return result
        
        # No high-quality interaction sentences found
        # Return None to trigger template-based prediction
        logger.info(f"No high-quality interaction sentences found for '{drug1}' + '{drug2}', using template fallback")
        return None


# Singleton instance
_retriever_instance = None

def get_retriever() -> PubMedRetriever:
    """Get singleton PubMedRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PubMedRetriever()
    return _retriever_instance
