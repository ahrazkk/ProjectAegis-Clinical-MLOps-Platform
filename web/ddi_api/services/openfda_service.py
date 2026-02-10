"""
OpenFDA API Integration Service

Provides real-time access to FDA adverse event reports (FAERS),
drug labels, and other FDA data for enhanced DDI prediction.

Reference: https://open.fda.gov/apis/
"""

import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# OpenFDA API endpoints
OPENFDA_BASE_URL = "https://api.fda.gov"
DRUG_EVENT_ENDPOINT = f"{OPENFDA_BASE_URL}/drug/event.json"
DRUG_LABEL_ENDPOINT = f"{OPENFDA_BASE_URL}/drug/label.json"
DRUG_NDC_ENDPOINT = f"{OPENFDA_BASE_URL}/drug/ndc.json"

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 40  # OpenFDA limit is 40/minute without API key
REQUEST_TIMEOUT = 30


@dataclass
class AdverseEvent:
    """Represents an FDA adverse event report."""
    report_id: str
    drug_names: List[str]
    reactions: List[str]
    outcome: Optional[str]
    severity: str
    report_date: Optional[str]
    patient_age: Optional[float]
    patient_sex: Optional[str]
    source_country: Optional[str]


@dataclass
class DrugLabel:
    """Represents FDA drug label information."""
    drug_name: str
    generic_name: Optional[str]
    brand_name: Optional[str]
    manufacturer: Optional[str]
    route: Optional[str]
    indications: Optional[str]
    warnings: Optional[str]
    contraindications: Optional[str]
    drug_interactions: Optional[str]
    pregnancy_category: Optional[str]
    boxed_warning: Optional[str]


class OpenFDAService:
    """
    Service for interacting with the OpenFDA API.
    
    Provides access to:
    - Adverse Event Reports (FAERS)
    - Drug Labels
    - Drug-Drug Interaction Information
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenFDA service.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self._last_request_time = 0
        self._request_count = 0
    
    def _rate_limit(self):
        """Apply rate limiting to avoid API throttling."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self._last_request_time > 60:
            self._request_count = 0
            self._last_request_time = current_time
        
        # Wait if approaching limit
        if self._request_count >= MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self._last_request_time)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._last_request_time = time.time()
        
        self._request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Make a request to the OpenFDA API.
        
        Args:
            endpoint: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        self._rate_limit()
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(
                endpoint,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug(f"No results found for query: {params}")
                return None
            elif response.status_code == 429:
                logger.warning("OpenFDA rate limit exceeded")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"OpenFDA API error: {response.status_code}")
                return None
                
        except requests.Timeout:
            logger.error("OpenFDA request timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"OpenFDA request failed: {e}")
            return None
    
    def get_drug_interactions_from_label(
        self, 
        drug_name: str
    ) -> Optional[str]:
        """
        Get drug interaction information from FDA drug labels.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Drug interaction section text or None
        """
        params = {
            'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            'limit': 1
        }
        
        result = self._make_request(DRUG_LABEL_ENDPOINT, params)
        
        if result and result.get('results'):
            label = result['results'][0]
            interactions = label.get('drug_interactions', [])
            if interactions:
                return interactions[0] if isinstance(interactions, list) else interactions
        
        return None
    
    def get_adverse_events(
        self,
        drug_name: str,
        limit: int = 100
    ) -> List[AdverseEvent]:
        """
        Get adverse event reports for a specific drug.
        
        Args:
            drug_name: Name of the drug
            limit: Maximum number of results
            
        Returns:
            List of AdverseEvent objects
        """
        params = {
            'search': f'patient.drug.openfda.brand_name:"{drug_name}" OR patient.drug.openfda.generic_name:"{drug_name}"',
            'limit': min(limit, 100)
        }
        
        result = self._make_request(DRUG_EVENT_ENDPOINT, params)
        events = []
        
        if result and result.get('results'):
            for record in result['results']:
                try:
                    # Extract drug names
                    drugs = record.get('patient', {}).get('drug', [])
                    drug_names = []
                    for drug in drugs:
                        name = drug.get('medicinalproduct', '')
                        if name:
                            drug_names.append(name)
                    
                    # Extract reactions
                    reactions_list = record.get('patient', {}).get('reaction', [])
                    reactions = [r.get('reactionmeddrapt', '') for r in reactions_list if r.get('reactionmeddrapt')]
                    
                    # Determine severity
                    serious = record.get('serious', 0)
                    outcome = record.get('patient', {}).get('patientonsetageunit')
                    severity = 'serious' if serious == 1 else 'non-serious'
                    
                    event = AdverseEvent(
                        report_id=record.get('safetyreportid', ''),
                        drug_names=drug_names,
                        reactions=reactions,
                        outcome=record.get('patient', {}).get('outcome'),
                        severity=severity,
                        report_date=record.get('receiptdate'),
                        patient_age=record.get('patient', {}).get('patientonsetage'),
                        patient_sex=record.get('patient', {}).get('patientsex'),
                        source_country=record.get('occurcountry')
                    )
                    events.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error parsing adverse event: {e}")
                    continue
        
        return events
    
    def get_drug_pair_adverse_events(
        self,
        drug1: str,
        drug2: str,
        limit: int = 100
    ) -> List[AdverseEvent]:
        """
        Get adverse events reported when two drugs are taken together.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            limit: Maximum number of results
            
        Returns:
            List of AdverseEvent objects for the drug pair
        """
        # Search for reports containing both drugs
        search_query = (
            f'(patient.drug.openfda.brand_name:"{drug1}" OR patient.drug.openfda.generic_name:"{drug1}") '
            f'AND (patient.drug.openfda.brand_name:"{drug2}" OR patient.drug.openfda.generic_name:"{drug2}")'
        )
        
        params = {
            'search': search_query,
            'limit': min(limit, 100)
        }
        
        result = self._make_request(DRUG_EVENT_ENDPOINT, params)
        events = []
        
        if result and result.get('results'):
            for record in result['results']:
                try:
                    drugs = record.get('patient', {}).get('drug', [])
                    drug_names = [d.get('medicinalproduct', '') for d in drugs if d.get('medicinalproduct')]
                    
                    reactions_list = record.get('patient', {}).get('reaction', [])
                    reactions = [r.get('reactionmeddrapt', '') for r in reactions_list if r.get('reactionmeddrapt')]
                    
                    serious = record.get('serious', 0)
                    severity = 'serious' if serious == 1 else 'non-serious'
                    
                    event = AdverseEvent(
                        report_id=record.get('safetyreportid', ''),
                        drug_names=drug_names,
                        reactions=reactions,
                        outcome=record.get('patient', {}).get('outcome'),
                        severity=severity,
                        report_date=record.get('receiptdate'),
                        patient_age=record.get('patient', {}).get('patientonsetage'),
                        patient_sex=record.get('patient', {}).get('patientsex'),
                        source_country=record.get('occurcountry')
                    )
                    events.append(event)
                    
                except Exception as e:
                    logger.warning(f"Error parsing adverse event: {e}")
                    continue
        
        return events
    
    def get_drug_label(self, drug_name: str) -> Optional[DrugLabel]:
        """
        Get comprehensive drug label information.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            DrugLabel object or None
        """
        params = {
            'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
            'limit': 1
        }
        
        result = self._make_request(DRUG_LABEL_ENDPOINT, params)
        
        if result and result.get('results'):
            label = result['results'][0]
            openfda = label.get('openfda', {})
            
            return DrugLabel(
                drug_name=drug_name,
                generic_name=openfda.get('generic_name', [None])[0] if openfda.get('generic_name') else None,
                brand_name=openfda.get('brand_name', [None])[0] if openfda.get('brand_name') else None,
                manufacturer=openfda.get('manufacturer_name', [None])[0] if openfda.get('manufacturer_name') else None,
                route=openfda.get('route', [None])[0] if openfda.get('route') else None,
                indications=label.get('indications_and_usage', [None])[0] if label.get('indications_and_usage') else None,
                warnings=label.get('warnings', [None])[0] if label.get('warnings') else None,
                contraindications=label.get('contraindications', [None])[0] if label.get('contraindications') else None,
                drug_interactions=label.get('drug_interactions', [None])[0] if label.get('drug_interactions') else None,
                pregnancy_category=label.get('pregnancy', [None])[0] if label.get('pregnancy') else None,
                boxed_warning=label.get('boxed_warning', [None])[0] if label.get('boxed_warning') else None
            )
        
        return None
    
    def count_adverse_events_by_reaction(
        self,
        drug_name: str
    ) -> Dict[str, int]:
        """
        Count adverse events grouped by reaction type.
        
        Args:
            drug_name: Name of the drug
            
        Returns:
            Dictionary of reaction -> count
        """
        params = {
            'search': f'patient.drug.openfda.brand_name:"{drug_name}" OR patient.drug.openfda.generic_name:"{drug_name}"',
            'count': 'patient.reaction.reactionmeddrapt.exact',
            'limit': 100
        }
        
        result = self._make_request(DRUG_EVENT_ENDPOINT, params)
        
        if result and result.get('results'):
            return {item['term']: item['count'] for item in result['results']}
        
        return {}
    
    def get_interaction_severity_from_faers(
        self,
        drug1: str,
        drug2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate interaction severity based on FAERS adverse event data.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            
        Returns:
            Severity assessment based on adverse event data
        """
        events = self.get_drug_pair_adverse_events(drug1, drug2, limit=100)
        
        if not events:
            return None
        
        # Count serious vs non-serious events
        serious_count = sum(1 for e in events if e.severity == 'serious')
        total_count = len(events)
        
        # Get common reactions
        all_reactions = []
        for event in events:
            all_reactions.extend(event.reactions)
        
        reaction_counts = {}
        for reaction in all_reactions:
            reaction_counts[reaction] = reaction_counts.get(reaction, 0) + 1
        
        # Sort by frequency
        top_reactions = sorted(reaction_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Estimate severity
        serious_ratio = serious_count / total_count if total_count > 0 else 0
        
        if serious_ratio > 0.5:
            estimated_severity = 'severe'
        elif serious_ratio > 0.2:
            estimated_severity = 'major'
        elif serious_ratio > 0.05:
            estimated_severity = 'moderate'
        else:
            estimated_severity = 'minor'
        
        return {
            'drug1': drug1,
            'drug2': drug2,
            'total_reports': total_count,
            'serious_reports': serious_count,
            'serious_ratio': serious_ratio,
            'estimated_severity': estimated_severity,
            'top_reactions': top_reactions,
            'source': 'FDA FAERS'
        }


# Singleton instance
_openfda_service: Optional[OpenFDAService] = None


def get_openfda_service(api_key: Optional[str] = None) -> OpenFDAService:
    """Get or create the OpenFDA service singleton."""
    global _openfda_service
    if _openfda_service is None:
        _openfda_service = OpenFDAService(api_key)
    return _openfda_service
