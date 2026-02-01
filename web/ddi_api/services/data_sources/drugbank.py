"""
DrugBank Parser for Project Aegis

DrugBank is a comprehensive database of drug and drug target information.
The full database requires a license, but DrugBank Open Data is freely available.

Open Data Download: https://go.drugbank.com/releases/latest#open-data
This module parses the XML files from DrugBank Open Data.

Note: This is a PARSER for local XML files, not an API client.
Download the DrugBank Open Data XML first.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Generator, Set
from dataclasses import dataclass, field
from pathlib import Path
import gzip

logger = logging.getLogger(__name__)

# DrugBank XML namespace
NS = {'db': 'http://www.drugbank.ca'}


@dataclass
class DrugBankDrug:
    """Drug data from DrugBank."""
    drugbank_id: str
    name: str
    description: Optional[str] = None
    cas_number: Optional[str] = None
    unii: Optional[str] = None
    state: Optional[str] = None  # solid, liquid, gas
    
    # Classification
    drug_type: Optional[str] = None  # small molecule, biotech
    groups: List[str] = field(default_factory=list)  # approved, investigational, etc.
    categories: List[str] = field(default_factory=list)
    
    # Chemical properties
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    
    # Pharmacology
    indication: Optional[str] = None
    pharmacodynamics: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    toxicity: Optional[str] = None
    metabolism: Optional[str] = None
    half_life: Optional[str] = None
    route_of_elimination: Optional[str] = None
    
    # External IDs
    synonyms: List[str] = field(default_factory=list)
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    # Interactions
    drug_interactions: List[Dict] = field(default_factory=list)
    food_interactions: List[str] = field(default_factory=list)
    
    # Targets, enzymes, transporters
    targets: List[Dict] = field(default_factory=list)
    enzymes: List[Dict] = field(default_factory=list)
    transporters: List[Dict] = field(default_factory=list)


class DrugBankParser:
    """
    Parser for DrugBank Open Data XML files.
    
    Usage:
        parser = DrugBankParser("drugbank_vocabulary.xml")
        
        # Iterate through all drugs
        for drug in parser.iter_drugs():
            print(drug.name, drug.drugbank_id)
        
        # Get specific drug
        drug = parser.get_drug_by_id("DB00945")  # Aspirin
        
        # Search by name
        drugs = parser.search("aspirin")
    """
    
    def __init__(self, xml_path: str):
        """
        Initialize parser with path to DrugBank XML file.
        
        Args:
            xml_path: Path to DrugBank XML (can be .gz compressed)
        """
        self.xml_path = Path(xml_path)
        self._drug_index: Dict[str, int] = {}  # id -> file position
        self._name_index: Dict[str, List[str]] = {}  # lowercase name -> ids
        self._indexed = False
    
    def _open_file(self):
        """Open XML file (handles .gz compression)."""
        if self.xml_path.suffix == '.gz':
            return gzip.open(self.xml_path, 'rt', encoding='utf-8')
        return open(self.xml_path, 'r', encoding='utf-8')
    
    def build_index(self):
        """Build index of drug IDs and names for fast lookup."""
        if self._indexed:
            return
        
        logger.info(f"Building index for {self.xml_path}...")
        
        with self._open_file() as f:
            for event, elem in ET.iterparse(f, events=['start', 'end']):
                if event == 'end' and elem.tag == f"{{{NS['db']}}}drug":
                    # Get drug ID
                    id_elem = elem.find('db:drugbank-id[@primary="true"]', NS)
                    if id_elem is None:
                        id_elem = elem.find('db:drugbank-id', NS)
                    
                    if id_elem is not None and id_elem.text:
                        drug_id = id_elem.text
                        
                        # Get name
                        name_elem = elem.find('db:name', NS)
                        name = name_elem.text if name_elem is not None else ""
                        
                        # Index by ID
                        self._drug_index[drug_id] = True
                        
                        # Index by name
                        if name:
                            name_lower = name.lower()
                            if name_lower not in self._name_index:
                                self._name_index[name_lower] = []
                            self._name_index[name_lower].append(drug_id)
                        
                        # Index synonyms
                        for syn_elem in elem.findall('.//db:synonym', NS):
                            if syn_elem.text:
                                syn_lower = syn_elem.text.lower()
                                if syn_lower not in self._name_index:
                                    self._name_index[syn_lower] = []
                                self._name_index[syn_lower].append(drug_id)
                    
                    elem.clear()
        
        self._indexed = True
        logger.info(f"Indexed {len(self._drug_index)} drugs")
    
    def iter_drugs(self, limit: int = None) -> Generator[DrugBankDrug, None, None]:
        """
        Iterate through all drugs in the XML file.
        
        Args:
            limit: Maximum number of drugs to yield
        """
        count = 0
        
        with self._open_file() as f:
            for event, elem in ET.iterparse(f, events=['end']):
                if elem.tag == f"{{{NS['db']}}}drug":
                    drug = self._parse_drug_element(elem)
                    if drug:
                        yield drug
                        count += 1
                        
                        if limit and count >= limit:
                            break
                    
                    elem.clear()
    
    def get_all_drugs(self, limit: int = None) -> List[DrugBankDrug]:
        """Get all drugs as a list."""
        return list(self.iter_drugs(limit=limit))
    
    def get_drug_by_id(self, drugbank_id: str) -> Optional[DrugBankDrug]:
        """
        Get a specific drug by DrugBank ID.
        
        Args:
            drugbank_id: DrugBank ID (e.g., "DB00945")
        """
        with self._open_file() as f:
            for event, elem in ET.iterparse(f, events=['end']):
                if elem.tag == f"{{{NS['db']}}}drug":
                    id_elem = elem.find('db:drugbank-id[@primary="true"]', NS)
                    if id_elem is None:
                        id_elem = elem.find('db:drugbank-id', NS)
                    
                    if id_elem is not None and id_elem.text == drugbank_id:
                        return self._parse_drug_element(elem)
                    
                    elem.clear()
        
        return None
    
    def search(self, query: str, max_results: int = 10) -> List[DrugBankDrug]:
        """
        Search for drugs by name or synonym.
        
        Args:
            query: Search term
            max_results: Maximum results to return
        """
        self.build_index()
        
        query_lower = query.lower()
        matching_ids: Set[str] = set()
        
        # Exact match first
        if query_lower in self._name_index:
            matching_ids.update(self._name_index[query_lower])
        
        # Partial match
        for name, ids in self._name_index.items():
            if query_lower in name and len(matching_ids) < max_results * 2:
                matching_ids.update(ids)
        
        # Fetch the actual drug data
        drugs = []
        for drug_id in list(matching_ids)[:max_results]:
            drug = self.get_drug_by_id(drug_id)
            if drug:
                drugs.append(drug)
        
        return drugs
    
    def _parse_drug_element(self, elem) -> Optional[DrugBankDrug]:
        """Parse a drug element from XML."""
        try:
            # Basic info
            id_elem = elem.find('db:drugbank-id[@primary="true"]', NS)
            if id_elem is None:
                id_elem = elem.find('db:drugbank-id', NS)
            
            if id_elem is None or not id_elem.text:
                return None
            
            drug = DrugBankDrug(
                drugbank_id=id_elem.text,
                name=self._get_text(elem, 'db:name'),
                description=self._get_text(elem, 'db:description'),
                cas_number=self._get_text(elem, 'db:cas-number'),
                unii=self._get_text(elem, 'db:unii'),
                state=self._get_text(elem, 'db:state'),
                drug_type=elem.get('type'),
            )
            
            # Groups (approved, investigational, etc.)
            for group in elem.findall('.//db:group', NS):
                if group.text:
                    drug.groups.append(group.text)
            
            # Categories
            for cat in elem.findall('.//db:category/db:category', NS):
                if cat.text:
                    drug.categories.append(cat.text)
            
            # Chemical properties
            drug.molecular_formula = self._get_property(elem, 'Molecular Formula')
            
            weight_str = self._get_property(elem, 'Molecular Weight')
            if weight_str:
                try:
                    drug.molecular_weight = float(weight_str)
                except ValueError:
                    pass
            
            drug.smiles = self._get_property(elem, 'SMILES')
            drug.inchi = self._get_property(elem, 'InChI')
            drug.inchi_key = self._get_property(elem, 'InChIKey')
            
            # Pharmacology
            drug.indication = self._get_text(elem, 'db:indication')
            drug.pharmacodynamics = self._get_text(elem, 'db:pharmacodynamics')
            drug.mechanism_of_action = self._get_text(elem, 'db:mechanism-of-action')
            drug.toxicity = self._get_text(elem, 'db:toxicity')
            drug.metabolism = self._get_text(elem, 'db:metabolism')
            drug.half_life = self._get_text(elem, 'db:half-life')
            drug.route_of_elimination = self._get_text(elem, 'db:route-of-elimination')
            
            # Synonyms
            for syn in elem.findall('.//db:synonyms/db:synonym', NS):
                if syn.text:
                    drug.synonyms.append(syn.text)
            
            # External identifiers
            for ext_id in elem.findall('.//db:external-identifiers/db:external-identifier', NS):
                resource = self._get_text(ext_id, 'db:resource')
                identifier = self._get_text(ext_id, 'db:identifier')
                if resource and identifier:
                    drug.external_ids[resource] = identifier
            
            # Drug interactions
            for interaction in elem.findall('.//db:drug-interactions/db:drug-interaction', NS):
                drug.drug_interactions.append({
                    'drugbank_id': self._get_text(interaction, 'db:drugbank-id'),
                    'name': self._get_text(interaction, 'db:name'),
                    'description': self._get_text(interaction, 'db:description')
                })
            
            # Food interactions
            for food in elem.findall('.//db:food-interactions/db:food-interaction', NS):
                if food.text:
                    drug.food_interactions.append(food.text)
            
            # Targets
            for target in elem.findall('.//db:targets/db:target', NS):
                drug.targets.append(self._parse_target(target))
            
            # Enzymes
            for enzyme in elem.findall('.//db:enzymes/db:enzyme', NS):
                drug.enzymes.append(self._parse_target(enzyme))
            
            # Transporters
            for transporter in elem.findall('.//db:transporters/db:transporter', NS):
                drug.transporters.append(self._parse_target(transporter))
            
            return drug
            
        except Exception as e:
            logger.error(f"Error parsing drug element: {e}")
            return None
    
    def _get_text(self, elem, path: str) -> Optional[str]:
        """Get text content of a child element."""
        child = elem.find(path, NS)
        return child.text if child is not None else None
    
    def _get_property(self, elem, name: str) -> Optional[str]:
        """Get a calculated property by name."""
        for prop in elem.findall('.//db:calculated-properties/db:property', NS):
            kind = self._get_text(prop, 'db:kind')
            if kind == name:
                return self._get_text(prop, 'db:value')
        
        # Also check experimental properties
        for prop in elem.findall('.//db:experimental-properties/db:property', NS):
            kind = self._get_text(prop, 'db:kind')
            if kind == name:
                return self._get_text(prop, 'db:value')
        
        return None
    
    def _parse_target(self, elem) -> Dict:
        """Parse a target/enzyme/transporter element."""
        return {
            'id': self._get_text(elem, 'db:id'),
            'name': self._get_text(elem, 'db:name'),
            'organism': self._get_text(elem, 'db:organism'),
            'known_action': self._get_text(elem, 'db:known-action'),
            'actions': [
                action.text for action in elem.findall('.//db:actions/db:action', NS)
                if action.text
            ],
            'uniprot_id': self._get_polypeptide_id(elem, 'UniProtKB'),
            'gene_name': self._get_text(elem, './/db:polypeptide/db:gene-name'),
        }
    
    def _get_polypeptide_id(self, elem, source: str) -> Optional[str]:
        """Get external ID from polypeptide."""
        for ext_id in elem.findall('.//db:polypeptide/db:external-identifiers/db:external-identifier', NS):
            resource = self._get_text(ext_id, 'db:resource')
            if resource == source:
                return self._get_text(ext_id, 'db:identifier')
        return None
    
    # ==================== Convenience Methods ====================
    
    def get_interactions_for_drug(self, drugbank_id: str) -> List[Dict]:
        """Get all drug-drug interactions for a specific drug."""
        drug = self.get_drug_by_id(drugbank_id)
        if drug:
            return drug.drug_interactions
        return []
    
    def get_all_interactions(self) -> Generator[Dict, None, None]:
        """
        Yield all drug-drug interactions in the database.
        
        Yields dicts with: drug1_id, drug1_name, drug2_id, drug2_name, description
        """
        for drug in self.iter_drugs():
            for interaction in drug.drug_interactions:
                yield {
                    'drug1_id': drug.drugbank_id,
                    'drug1_name': drug.name,
                    'drug2_id': interaction.get('drugbank_id'),
                    'drug2_name': interaction.get('name'),
                    'description': interaction.get('description')
                }
    
    def get_drugs_with_smiles(self) -> Generator[DrugBankDrug, None, None]:
        """Yield only drugs that have SMILES structures."""
        for drug in self.iter_drugs():
            if drug.smiles:
                yield drug
    
    def get_approved_drugs(self) -> Generator[DrugBankDrug, None, None]:
        """Yield only approved drugs."""
        for drug in self.iter_drugs():
            if 'approved' in drug.groups:
                yield drug
