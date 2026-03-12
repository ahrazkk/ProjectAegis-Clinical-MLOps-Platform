"""
RxNorm API Client for Project Aegis

RxNorm is the NIH's standardized nomenclature for clinical drugs.
Provides: Drug names, NDC codes, ingredient relationships, drug classes.

API Documentation: https://lhncbc.nlm.nih.gov/RxNav/APIs/
No API key required. Rate limit: ~20 requests/second.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


@dataclass
class RxNormDrug:
    """Standardized drug data from RxNorm."""
    rxcui: str
    name: str
    synonym: Optional[str] = None
    tty: Optional[str] = None  # Term type (SCD, SBD, IN, etc.)
    language: str = "ENG"
    suppress: str = "N"
    
    # Extended properties (fetched separately)
    ndc_codes: List[str] = None
    ingredients: List[Dict] = None
    drug_classes: List[Dict] = None
    interactions: List[Dict] = None
    
    def __post_init__(self):
        if self.ndc_codes is None:
            self.ndc_codes = []
        if self.ingredients is None:
            self.ingredients = []
        if self.drug_classes is None:
            self.drug_classes = []
        if self.interactions is None:
            self.interactions = []


class RxNormClient:
    """
    Client for the RxNorm REST API.
    
    Usage:
        client = RxNormClient()
        
        # Search for drugs
        results = client.search("metformin")
        
        # Get drug details
        drug = client.get_drug_by_rxcui("860975")
        
        # Get interactions
        interactions = client.get_interactions("860975")
        
        # Batch fetch all data
        drugs = client.fetch_all_drugs(limit=1000)
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.session = requests.Session()
        self.timeout = timeout
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 20 requests/second max
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a GET request to RxNorm API."""
        self._rate_limit()
        
        url = f"{BASE_URL}/{endpoint}"
        params = params or {}
        params['format'] = 'json'
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"RxNorm API error: {e}")
            raise
    
    # ==================== Search ====================
    
    def search(self, term: str, search_type: int = 0) -> List[RxNormDrug]:
        """
        Search for drugs by name.
        
        Args:
            term: Drug name to search
            search_type: 0=exact, 1=normalized, 2=contains
            
        Returns:
            List of matching drugs
        """
        data = self._get("drugs.json", {"name": term, "search": search_type})
        
        drugs = []
        drug_group = data.get("drugGroup", {})
        
        for concept_group in drug_group.get("conceptGroup", []):
            for concept in concept_group.get("conceptProperties", []):
                drugs.append(RxNormDrug(
                    rxcui=concept.get("rxcui"),
                    name=concept.get("name"),
                    synonym=concept.get("synonym"),
                    tty=concept.get("tty"),
                    language=concept.get("language", "ENG"),
                    suppress=concept.get("suppress", "N")
                ))
        
        return drugs
    
    def approximate_search(self, term: str, max_entries: int = 20) -> List[RxNormDrug]:
        """
        Approximate/fuzzy search for drugs.
        Good for handling typos and variations.
        """
        data = self._get("approximateTerm.json", {
            "term": term,
            "maxEntries": max_entries
        })
        
        drugs = []
        for candidate in data.get("approximateGroup", {}).get("candidate", []):
            drugs.append(RxNormDrug(
                rxcui=candidate.get("rxcui"),
                name=candidate.get("name", ""),
                tty=candidate.get("tty")
            ))
        
        return drugs
    
    # ==================== Drug Details ====================
    
    def get_drug_by_rxcui(self, rxcui: str, include_related: bool = True) -> Optional[RxNormDrug]:
        """
        Get detailed drug information by RxCUI.
        
        Args:
            rxcui: RxNorm Concept Unique Identifier
            include_related: Also fetch NDCs, classes, ingredients
        """
        data = self._get(f"rxcui/{rxcui}/properties.json")
        
        props = data.get("properties")
        if not props:
            return None
        
        drug = RxNormDrug(
            rxcui=props.get("rxcui"),
            name=props.get("name"),
            synonym=props.get("synonym"),
            tty=props.get("tty"),
            language=props.get("language", "ENG"),
            suppress=props.get("suppress", "N")
        )
        
        if include_related:
            drug.ndc_codes = self.get_ndcs(rxcui)
            drug.ingredients = self.get_ingredients(rxcui)
            drug.drug_classes = self.get_drug_classes(rxcui)
        
        return drug
    
    def get_ndcs(self, rxcui: str) -> List[str]:
        """Get NDC codes for a drug."""
        data = self._get(f"rxcui/{rxcui}/ndcs.json")
        ndc_group = data.get("ndcGroup", {})
        return ndc_group.get("ndcList", {}).get("ndc", [])
    
    def get_ingredients(self, rxcui: str) -> List[Dict]:
        """Get active ingredients for a drug."""
        data = self._get(f"rxcui/{rxcui}/related.json", {"tty": "IN"})
        
        ingredients = []
        for group in data.get("relatedGroup", {}).get("conceptGroup", []):
            for concept in group.get("conceptProperties", []):
                ingredients.append({
                    "rxcui": concept.get("rxcui"),
                    "name": concept.get("name"),
                    "tty": concept.get("tty")
                })
        
        return ingredients
    
    def get_drug_classes(self, rxcui: str) -> List[Dict]:
        """Get drug classes (ATC, VA, etc.)."""
        data = self._get(f"rxclass/class/byRxcui.json", {"rxcui": rxcui})
        
        classes = []
        for entry in data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
            class_info = entry.get("rxclassMinConceptItem", {})
            classes.append({
                "id": class_info.get("classId"),
                "name": class_info.get("className"),
                "type": class_info.get("classType")
            })
        
        return classes
    
    # ==================== Interactions ====================
    
    def get_interactions(self, rxcui: str) -> List[Dict]:
        """
        Get drug-drug interactions for a drug.
        Uses the RxNorm interaction API.
        """
        data = self._get("interaction/interaction.json", {"rxcui": rxcui})
        
        interactions = []
        for group in data.get("interactionTypeGroup", []):
            source = group.get("sourceName", "Unknown")
            
            for interaction_type in group.get("interactionType", []):
                for pair in interaction_type.get("interactionPair", []):
                    # Get the other drug in the interaction
                    concepts = pair.get("interactionConcept", [])
                    if len(concepts) >= 2:
                        other_drug = None
                        for concept in concepts:
                            if concept.get("minConceptItem", {}).get("rxcui") != rxcui:
                                other_drug = concept.get("minConceptItem", {})
                                break
                        
                        if other_drug:
                            interactions.append({
                                "drug_rxcui": other_drug.get("rxcui"),
                                "drug_name": other_drug.get("name"),
                                "severity": pair.get("severity"),
                                "description": pair.get("description"),
                                "source": source
                            })
        
        return interactions
    
    def get_interaction_between(self, rxcui1: str, rxcui2: str) -> List[Dict]:
        """Check for interactions between two specific drugs."""
        data = self._get("interaction/list.json", {
            "rxcuis": f"{rxcui1}+{rxcui2}"
        })
        
        interactions = []
        for group in data.get("fullInteractionTypeGroup", []):
            for interaction_type in group.get("fullInteractionType", []):
                for pair in interaction_type.get("interactionPair", []):
                    interactions.append({
                        "severity": pair.get("severity"),
                        "description": pair.get("description"),
                        "source": group.get("sourceName")
                    })
        
        return interactions
    
    # ==================== Bulk Operations ====================
    
    def get_all_rxcuis(self, tty: str = "SCD") -> List[str]:
        """
        Get all RxCUIs of a specific term type.
        
        Term types:
        - IN: Ingredient
        - SCD: Semantic Clinical Drug
        - SBD: Semantic Branded Drug
        - GPCK: Generic Pack
        - BPCK: Branded Pack
        """
        data = self._get("allconcepts.json", {"tty": tty})
        
        rxcuis = []
        for group in data.get("minConceptGroup", {}).get("minConcept", []):
            rxcuis.append(group.get("rxcui"))
        
        return rxcuis
    
    def fetch_all_drugs(
        self, 
        tty: str = "SCD", 
        limit: int = None,
        include_interactions: bool = False,
        progress_callback: callable = None
    ) -> List[RxNormDrug]:
        """
        Fetch all drugs of a specific type with full details.
        
        Args:
            tty: Term type to fetch
            limit: Maximum number of drugs to fetch
            include_interactions: Also fetch interactions (slow)
            progress_callback: Function called with (current, total)
        """
        logger.info(f"Fetching all RxCUIs with tty={tty}...")
        rxcuis = self.get_all_rxcuis(tty)
        
        if limit:
            rxcuis = rxcuis[:limit]
        
        total = len(rxcuis)
        logger.info(f"Found {total} RxCUIs. Fetching details...")
        
        drugs = []
        for i, rxcui in enumerate(rxcuis):
            try:
                drug = self.get_drug_by_rxcui(rxcui, include_related=True)
                if drug:
                    if include_interactions:
                        drug.interactions = self.get_interactions(rxcui)
                    drugs.append(drug)
                
                if progress_callback:
                    progress_callback(i + 1, total)
                    
            except Exception as e:
                logger.warning(f"Error fetching drug {rxcui}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(drugs)} drugs")
        return drugs
    
    # ==================== Utility ====================
    
    def rxcui_to_ndc(self, rxcui: str) -> List[str]:
        """Convert RxCUI to NDC codes."""
        return self.get_ndcs(rxcui)
    
    def ndc_to_rxcui(self, ndc: str) -> Optional[str]:
        """Convert NDC code to RxCUI."""
        # Remove dashes from NDC
        ndc_clean = ndc.replace("-", "")
        
        data = self._get("ndcstatus.json", {"ndc": ndc_clean})
        status = data.get("ndcStatus", {})
        return status.get("rxcui")
    
    def get_spelling_suggestions(self, term: str) -> List[str]:
        """Get spelling suggestions for a misspelled drug name."""
        data = self._get("spellingsuggestions.json", {"name": term})
        return data.get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion", [])
