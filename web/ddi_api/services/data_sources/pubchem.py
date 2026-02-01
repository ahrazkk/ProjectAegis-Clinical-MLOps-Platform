"""
PubChem API Client for Project Aegis

PubChem is the world's largest collection of freely accessible chemical information.
Provides: Molecular structures, properties, bioactivities, safety data.

API Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
No API key required. Rate limit: 5 requests/second.
"""

import time
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
BASE_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


@dataclass
class PubChemCompound:
    """Compound data from PubChem."""
    cid: int
    name: str
    iupac_name: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    canonical_smiles: Optional[str] = None
    isomeric_smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    
    # Physical properties
    xlogp: Optional[float] = None
    exact_mass: Optional[float] = None
    monoisotopic_mass: Optional[float] = None
    tpsa: Optional[float] = None  # Topological polar surface area
    complexity: Optional[float] = None
    charge: Optional[int] = None
    h_bond_donor_count: Optional[int] = None
    h_bond_acceptor_count: Optional[int] = None
    rotatable_bond_count: Optional[int] = None
    heavy_atom_count: Optional[int] = None
    
    # Identifiers
    synonyms: List[str] = field(default_factory=list)
    cas_numbers: List[str] = field(default_factory=list)
    
    # Safety & bioactivity
    ghs_hazards: List[str] = field(default_factory=list)
    pharmacology: Optional[str] = None


class PubChemClient:
    """
    Client for the PubChem PUG REST API.
    
    Usage:
        client = PubChemClient()
        
        # Search by name
        compounds = client.search("aspirin")
        
        # Get compound by CID
        compound = client.get_compound(2244)  # Aspirin
        
        # Get SMILES
        smiles = client.get_smiles("aspirin")
        
        # Get similar compounds
        similar = client.get_similar_compounds(2244, threshold=90)
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
        
        # Rate limiting (5 req/sec)
        self._last_request_time = 0
        self._min_request_interval = 0.2
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get(self, endpoint: str, output: str = "JSON") -> Dict:
        """Make a GET request to PubChem API."""
        self._rate_limit()
        
        url = f"{BASE_URL}/{endpoint}/{output}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"PubChem API error: {e}")
            raise
    
    def _get_view(self, endpoint: str) -> Dict:
        """Make a request to PUG View API (for annotations)."""
        self._rate_limit()
        
        url = f"{BASE_VIEW_URL}/{endpoint}/JSON"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"PubChem View API error: {e}")
            raise
    
    # ==================== Search ====================
    
    def search(self, name: str, max_results: int = 10) -> List[PubChemCompound]:
        """
        Search compounds by name.
        
        Args:
            name: Compound name or synonym
            max_results: Maximum number of results
        """
        # Get CIDs matching the name
        data = self._get(f"compound/name/{name}/cids")
        
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return []
        
        # Limit results
        cids = cids[:max_results]
        
        # Get compound details
        return self.get_compounds_by_cids(cids)
    
    def search_by_smiles(self, smiles: str) -> Optional[PubChemCompound]:
        """Search compound by SMILES string."""
        try:
            data = self._get(f"compound/smiles/{smiles}/cids")
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if cids:
                return self.get_compound(cids[0])
        except requests.HTTPError:
            pass
        
        return None
    
    def search_by_inchi(self, inchi: str) -> Optional[PubChemCompound]:
        """Search compound by InChI."""
        try:
            # InChI needs to be URL-encoded, use POST instead
            url = f"{BASE_URL}/compound/inchi/cids/JSON"
            response = self.session.post(
                url, 
                data={"inchi": inchi},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return self.get_compound(cids[0])
        except requests.RequestException:
            pass
        
        return None
    
    # ==================== Get Compound Data ====================
    
    def get_compound(self, cid: int, include_synonyms: bool = True) -> Optional[PubChemCompound]:
        """
        Get detailed compound information by CID.
        
        Args:
            cid: PubChem Compound ID
            include_synonyms: Also fetch synonyms (extra API call)
        """
        # Get properties
        properties = [
            "IUPACName", "MolecularFormula", "MolecularWeight",
            "CanonicalSMILES", "IsomericSMILES", "InChI", "InChIKey",
            "XLogP", "ExactMass", "MonoisotopicMass", "TPSA",
            "Complexity", "Charge", "HBondDonorCount", "HBondAcceptorCount",
            "RotatableBondCount", "HeavyAtomCount"
        ]
        
        prop_str = ",".join(properties)
        data = self._get(f"compound/cid/{cid}/property/{prop_str}")
        
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        
        p = props[0]
        
        compound = PubChemCompound(
            cid=cid,
            name="",  # Will be set from synonyms
            iupac_name=p.get("IUPACName"),
            molecular_formula=p.get("MolecularFormula"),
            molecular_weight=p.get("MolecularWeight"),
            canonical_smiles=p.get("CanonicalSMILES"),
            isomeric_smiles=p.get("IsomericSMILES"),
            inchi=p.get("InChI"),
            inchi_key=p.get("InChIKey"),
            xlogp=p.get("XLogP"),
            exact_mass=p.get("ExactMass"),
            monoisotopic_mass=p.get("MonoisotopicMass"),
            tpsa=p.get("TPSA"),
            complexity=p.get("Complexity"),
            charge=p.get("Charge"),
            h_bond_donor_count=p.get("HBondDonorCount"),
            h_bond_acceptor_count=p.get("HBondAcceptorCount"),
            rotatable_bond_count=p.get("RotatableBondCount"),
            heavy_atom_count=p.get("HeavyAtomCount")
        )
        
        if include_synonyms:
            synonyms = self.get_synonyms(cid)
            compound.synonyms = synonyms
            if synonyms:
                compound.name = synonyms[0]
            
            # Extract CAS numbers from synonyms
            compound.cas_numbers = [
                s for s in synonyms 
                if self._is_cas_number(s)
            ]
        
        return compound
    
    def get_compounds_by_cids(self, cids: List[int]) -> List[PubChemCompound]:
        """Get multiple compounds by their CIDs."""
        if not cids:
            return []
        
        # PubChem allows comma-separated CIDs
        cid_str = ",".join(str(c) for c in cids)
        
        properties = [
            "IUPACName", "MolecularFormula", "MolecularWeight",
            "CanonicalSMILES", "IsomericSMILES", "InChI", "InChIKey",
            "XLogP", "TPSA", "Complexity"
        ]
        
        prop_str = ",".join(properties)
        data = self._get(f"compound/cid/{cid_str}/property/{prop_str}")
        
        compounds = []
        for p in data.get("PropertyTable", {}).get("Properties", []):
            compound = PubChemCompound(
                cid=p.get("CID"),
                name=p.get("IUPACName", ""),
                iupac_name=p.get("IUPACName"),
                molecular_formula=p.get("MolecularFormula"),
                molecular_weight=p.get("MolecularWeight"),
                canonical_smiles=p.get("CanonicalSMILES"),
                isomeric_smiles=p.get("IsomericSMILES"),
                inchi=p.get("InChI"),
                inchi_key=p.get("InChIKey"),
                xlogp=p.get("XLogP"),
                tpsa=p.get("TPSA"),
                complexity=p.get("Complexity")
            )
            compounds.append(compound)
        
        return compounds
    
    def get_synonyms(self, cid: int) -> List[str]:
        """Get all synonyms for a compound."""
        try:
            data = self._get(f"compound/cid/{cid}/synonyms")
            return data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        except (requests.HTTPError, IndexError):
            return []
    
    def get_smiles(self, name_or_cid: Union[str, int]) -> Optional[str]:
        """Get canonical SMILES for a compound by name or CID."""
        try:
            if isinstance(name_or_cid, int):
                endpoint = f"compound/cid/{name_or_cid}/property/CanonicalSMILES"
            else:
                endpoint = f"compound/name/{name_or_cid}/property/CanonicalSMILES"
            
            data = self._get(endpoint)
            props = data.get("PropertyTable", {}).get("Properties", [])
            
            if props:
                return props[0].get("CanonicalSMILES")
        except requests.HTTPError:
            pass
        
        return None
    
    # ==================== Structure Search ====================
    
    def get_similar_compounds(
        self, 
        cid: int, 
        threshold: int = 90,
        max_results: int = 100
    ) -> List[int]:
        """
        Find structurally similar compounds.
        
        Args:
            cid: Reference compound CID
            threshold: Similarity threshold (0-100)
            max_results: Maximum results to return
        """
        try:
            data = self._get(
                f"compound/fastsimilarity_2d/cid/{cid}/cids",
                output=f"JSON?Threshold={threshold}&MaxRecords={max_results}"
            )
            return data.get("IdentifierList", {}).get("CID", [])
        except requests.HTTPError:
            return []
    
    def get_substructure_matches(
        self, 
        smiles: str,
        max_results: int = 100
    ) -> List[int]:
        """Find compounds containing a substructure."""
        try:
            # URL encode SMILES properly
            import urllib.parse
            encoded_smiles = urllib.parse.quote(smiles, safe='')
            
            data = self._get(
                f"compound/fastsubstructure/smiles/{encoded_smiles}/cids",
                output=f"JSON?MaxRecords={max_results}"
            )
            return data.get("IdentifierList", {}).get("CID", [])
        except requests.HTTPError:
            return []
    
    # ==================== Safety & Bioactivity ====================
    
    def get_ghs_classification(self, cid: int) -> List[Dict]:
        """Get GHS hazard classification for a compound."""
        try:
            data = self._get_view(f"data/compound/{cid}/JSON?heading=GHS+Classification")
            
            hazards = []
            # Parse the nested structure
            sections = data.get("Record", {}).get("Section", [])
            for section in sections:
                if "GHS" in section.get("TOCHeading", ""):
                    for subsection in section.get("Section", []):
                        for info in subsection.get("Information", []):
                            value = info.get("Value", {})
                            if "StringWithMarkup" in value:
                                for item in value["StringWithMarkup"]:
                                    hazards.append({
                                        "category": subsection.get("TOCHeading"),
                                        "statement": item.get("String")
                                    })
            
            return hazards
        except requests.HTTPError:
            return []
    
    def get_pharmacology(self, cid: int) -> Optional[str]:
        """Get pharmacology description for a drug compound."""
        try:
            data = self._get_view(f"data/compound/{cid}/JSON?heading=Pharmacology")
            
            sections = data.get("Record", {}).get("Section", [])
            for section in sections:
                if "Pharmacology" in section.get("TOCHeading", ""):
                    for info in section.get("Information", []):
                        value = info.get("Value", {})
                        if "StringWithMarkup" in value:
                            texts = [
                                item.get("String", "")
                                for item in value["StringWithMarkup"]
                            ]
                            return " ".join(texts)
            
            return None
        except requests.HTTPError:
            return None
    
    # ==================== Drug Info ====================
    
    def get_drug_info(self, cid: int) -> Dict:
        """Get drug-specific information (targets, pathways, etc.)."""
        try:
            data = self._get_view(f"data/compound/{cid}/JSON?heading=Drug+and+Medication+Information")
            
            info = {}
            sections = data.get("Record", {}).get("Section", [])
            
            for section in sections:
                heading = section.get("TOCHeading", "")
                if "Drug" in heading or "Medication" in heading:
                    for subsection in section.get("Section", []):
                        sub_heading = subsection.get("TOCHeading", "")
                        values = []
                        
                        for item in subsection.get("Information", []):
                            value = item.get("Value", {})
                            if "StringWithMarkup" in value:
                                for markup in value["StringWithMarkup"]:
                                    values.append(markup.get("String", ""))
                        
                        if values:
                            info[sub_heading] = values
            
            return info
        except requests.HTTPError:
            return {}
    
    # ==================== Utilities ====================
    
    @staticmethod
    def _is_cas_number(s: str) -> bool:
        """Check if a string looks like a CAS Registry Number (format: XXXXXX-XX-X)."""
        import re
        return bool(re.match(r'^\d{2,7}-\d{2}-\d$', s))
    
    def name_to_cid(self, name: str) -> Optional[int]:
        """Convert drug name to PubChem CID."""
        try:
            data = self._get(f"compound/name/{name}/cids")
            cids = data.get("IdentifierList", {}).get("CID", [])
            return cids[0] if cids else None
        except requests.HTTPError:
            return None
    
    def cid_to_name(self, cid: int) -> Optional[str]:
        """Get the primary name for a CID."""
        synonyms = self.get_synonyms(cid)
        return synonyms[0] if synonyms else None
