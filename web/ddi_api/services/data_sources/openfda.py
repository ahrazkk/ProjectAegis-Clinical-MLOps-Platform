"""
OpenFDA API Client for Project Aegis

OpenFDA provides access to FDA drug data including:
- Drug labels (SPL)
- NDC directory
- Adverse events (FAERS)
- Drug recalls

API Documentation: https://open.fda.gov/apis/
No API key required for basic access. Rate limit: 240 requests/minute.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.fda.gov"


@dataclass
class OpenFDADrug:
    """Drug data from OpenFDA."""
    brand_name: str
    generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    ndc_codes: List[str] = field(default_factory=list)
    route: Optional[str] = None
    dosage_form: Optional[str] = None
    active_ingredients: List[Dict] = field(default_factory=list)
    warnings: Optional[str] = None
    indications: Optional[str] = None
    contraindications: Optional[str] = None
    drug_interactions: Optional[str] = None
    adverse_reactions: Optional[str] = None
    spl_id: Optional[str] = None
    application_number: Optional[str] = None


@dataclass
class AdverseEvent:
    """Adverse event report from FAERS."""
    report_id: str
    receive_date: str
    drugs: List[Dict] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)
    serious: bool = False
    outcome: Optional[str] = None
    patient_age: Optional[float] = None
    patient_sex: Optional[str] = None


class OpenFDAClient:
    """
    Client for the OpenFDA API.
    
    Usage:
        client = OpenFDAClient()
        
        # Search drug labels
        labels = client.search_drug_labels("metformin")
        
        # Get adverse events
        events = client.get_adverse_events("metformin", limit=100)
        
        # Get NDC info
        ndc = client.get_ndc_info("0093-7146")
    """
    
    def __init__(self, api_key: str = None, timeout: int = 30, max_retries: int = 3):
        self.session = requests.Session()
        self.timeout = timeout
        self.api_key = api_key
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Rate limiting (240/min = 4/sec)
        self._last_request_time = 0
        self._min_request_interval = 0.25
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a GET request to OpenFDA API."""
        self._rate_limit()
        
        url = f"{BASE_URL}/{endpoint}"
        params = params or {}
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OpenFDA API error: {e}")
            raise
    
    # ==================== Drug Labels ====================
    
    def search_drug_labels(
        self, 
        query: str, 
        field: str = "openfda.brand_name",
        limit: int = 10,
        skip: int = 0
    ) -> List[OpenFDADrug]:
        """
        Search drug labels (SPL data).
        
        Args:
            query: Search term
            field: Field to search (brand_name, generic_name, substance_name)
            limit: Number of results
            skip: Results to skip (pagination)
        """
        params = {
            "search": f'{field}:"{query}"',
            "limit": limit,
            "skip": skip
        }
        
        data = self._get("drug/label.json", params)
        
        drugs = []
        for result in data.get("results", []):
            openfda = result.get("openfda", {})
            
            drug = OpenFDADrug(
                brand_name=self._first_or_none(openfda.get("brand_name")),
                generic_name=self._first_or_none(openfda.get("generic_name")),
                manufacturer=self._first_or_none(openfda.get("manufacturer_name")),
                ndc_codes=openfda.get("package_ndc", []),
                route=self._first_or_none(openfda.get("route")),
                dosage_form=self._first_or_none(result.get("dosage_form")),
                active_ingredients=self._parse_ingredients(result),
                warnings=self._first_or_none(result.get("warnings")),
                indications=self._first_or_none(result.get("indications_and_usage")),
                contraindications=self._first_or_none(result.get("contraindications")),
                drug_interactions=self._first_or_none(result.get("drug_interactions")),
                adverse_reactions=self._first_or_none(result.get("adverse_reactions")),
                spl_id=result.get("id"),
                application_number=self._first_or_none(openfda.get("application_number"))
            )
            drugs.append(drug)
        
        return drugs
    
    def get_drug_label_by_ndc(self, ndc: str) -> Optional[OpenFDADrug]:
        """Get drug label by NDC code."""
        # Normalize NDC (remove dashes)
        ndc_clean = ndc.replace("-", "")
        
        params = {
            "search": f'openfda.package_ndc:"{ndc}" OR openfda.product_ndc:"{ndc_clean}"',
            "limit": 1
        }
        
        data = self._get("drug/label.json", params)
        results = data.get("results", [])
        
        if results:
            return self._parse_drug_label(results[0])
        return None
    
    def _parse_drug_label(self, result: Dict) -> OpenFDADrug:
        """Parse a single drug label result."""
        openfda = result.get("openfda", {})
        
        return OpenFDADrug(
            brand_name=self._first_or_none(openfda.get("brand_name")),
            generic_name=self._first_or_none(openfda.get("generic_name")),
            manufacturer=self._first_or_none(openfda.get("manufacturer_name")),
            ndc_codes=openfda.get("package_ndc", []),
            route=self._first_or_none(openfda.get("route")),
            active_ingredients=self._parse_ingredients(result),
            warnings=self._first_or_none(result.get("warnings")),
            indications=self._first_or_none(result.get("indications_and_usage")),
            contraindications=self._first_or_none(result.get("contraindications")),
            drug_interactions=self._first_or_none(result.get("drug_interactions")),
            adverse_reactions=self._first_or_none(result.get("adverse_reactions")),
            spl_id=result.get("id")
        )
    
    def _parse_ingredients(self, result: Dict) -> List[Dict]:
        """Parse active ingredients from label."""
        ingredients = []
        
        # Try active_ingredient field first
        for item in result.get("active_ingredient", []):
            if isinstance(item, str):
                ingredients.append({"name": item})
            elif isinstance(item, dict):
                ingredients.append(item)
        
        # Fall back to openfda.substance_name
        if not ingredients:
            openfda = result.get("openfda", {})
            for name in openfda.get("substance_name", []):
                ingredients.append({"name": name})
        
        return ingredients
    
    # ==================== Adverse Events (FAERS) ====================
    
    def get_adverse_events(
        self,
        drug_name: str,
        limit: int = 100,
        skip: int = 0,
        serious_only: bool = False
    ) -> List[AdverseEvent]:
        """
        Get adverse event reports for a drug.
        
        Args:
            drug_name: Drug name to search
            limit: Number of results (max 1000)
            skip: Results to skip
            serious_only: Only return serious events
        """
        search = f'patient.drug.medicinalproduct:"{drug_name}"'
        if serious_only:
            search += " AND serious:1"
        
        params = {
            "search": search,
            "limit": min(limit, 1000),
            "skip": skip
        }
        
        data = self._get("drug/event.json", params)
        
        events = []
        for result in data.get("results", []):
            # Extract drugs from the report
            drugs = []
            for drug in result.get("patient", {}).get("drug", []):
                drugs.append({
                    "name": drug.get("medicinalproduct"),
                    "indication": drug.get("drugindication"),
                    "role": drug.get("drugcharacterization")  # 1=suspect, 2=concomitant, 3=interacting
                })
            
            # Extract reactions
            reactions = [
                r.get("reactionmeddrapt") 
                for r in result.get("patient", {}).get("reaction", [])
                if r.get("reactionmeddrapt")
            ]
            
            event = AdverseEvent(
                report_id=result.get("safetyreportid", ""),
                receive_date=result.get("receivedate", ""),
                drugs=drugs,
                reactions=reactions,
                serious=result.get("serious") == "1",
                outcome=self._map_outcome(result.get("seriousnessoutcome")),
                patient_age=self._parse_age(result.get("patient", {})),
                patient_sex=self._map_sex(result.get("patient", {}).get("patientsex"))
            )
            events.append(event)
        
        return events
    
    def get_adverse_event_counts(
        self,
        drug_name: str,
        group_by: str = "patient.reaction.reactionmeddrapt.exact"
    ) -> Dict[str, int]:
        """
        Get counts of adverse events grouped by reaction type.
        
        Args:
            drug_name: Drug to search
            group_by: Field to group by
        """
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "count": group_by
        }
        
        data = self._get("drug/event.json", params)
        
        return {
            item.get("term"): item.get("count")
            for item in data.get("results", [])
        }
    
    # ==================== NDC Directory ====================
    
    def get_ndc_info(self, ndc: str) -> Optional[Dict]:
        """Get NDC directory information."""
        ndc_clean = ndc.replace("-", "")
        
        params = {
            "search": f'product_ndc:"{ndc_clean}" OR package_ndc:"{ndc_clean}"',
            "limit": 1
        }
        
        data = self._get("drug/ndc.json", params)
        results = data.get("results", [])
        
        if results:
            return results[0]
        return None
    
    def search_ndc(
        self,
        query: str,
        field: str = "brand_name",
        limit: int = 10
    ) -> List[Dict]:
        """Search NDC directory."""
        params = {
            "search": f'{field}:"{query}"',
            "limit": limit
        }
        
        data = self._get("drug/ndc.json", params)
        return data.get("results", [])
    
    # ==================== Drug Recalls ====================
    
    def get_drug_recalls(
        self,
        drug_name: str = None,
        status: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get drug recall information.
        
        Args:
            drug_name: Filter by drug name
            status: Filter by status (Ongoing, Completed, Terminated)
            limit: Number of results
        """
        search_parts = []
        
        if drug_name:
            search_parts.append(f'product_description:"{drug_name}"')
        if status:
            search_parts.append(f'status:"{status}"')
        
        params = {"limit": limit}
        if search_parts:
            params["search"] = " AND ".join(search_parts)
        
        data = self._get("drug/enforcement.json", params)
        return data.get("results", [])
    
    # ==================== Utilities ====================
    
    @staticmethod
    def _first_or_none(lst: Optional[List]) -> Optional[str]:
        """Get first element of list or None."""
        if lst and len(lst) > 0:
            return lst[0]
        return None
    
    @staticmethod
    def _map_outcome(code: Optional[str]) -> Optional[str]:
        """Map outcome code to description."""
        outcomes = {
            "1": "Recovered/Resolved",
            "2": "Recovering/Resolving", 
            "3": "Not Recovered/Not Resolved",
            "4": "Recovered/Resolved with Sequelae",
            "5": "Fatal",
            "6": "Unknown"
        }
        return outcomes.get(code)
    
    @staticmethod
    def _map_sex(code: Optional[str]) -> Optional[str]:
        """Map sex code to description."""
        sex_map = {"0": "Unknown", "1": "Male", "2": "Female"}
        return sex_map.get(code)
    
    @staticmethod
    def _parse_age(patient: Dict) -> Optional[float]:
        """Parse patient age from various formats."""
        age = patient.get("patientonsetage")
        unit = patient.get("patientonsetageunit")
        
        if age is None:
            return None
        
        try:
            age = float(age)
            # Convert to years
            if unit == "800":  # Decade
                age *= 10
            elif unit == "801":  # Year
                pass
            elif unit == "802":  # Month
                age /= 12
            elif unit == "803":  # Week
                age /= 52
            elif unit == "804":  # Day
                age /= 365
            elif unit == "805":  # Hour
                age /= 8760
            return age
        except (ValueError, TypeError):
            return None
