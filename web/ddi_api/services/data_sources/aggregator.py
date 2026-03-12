"""
Drug Data Aggregator for Project Aegis

Combines data from multiple pharmaceutical databases into a unified format.
Handles deduplication, normalization, and enrichment of drug records.

Data Sources:
- RxNorm (NIH) - Drug names, NDC codes, interactions
- OpenFDA - Drug labels, adverse events
- PubChem - Molecular structures, properties
- DrugBank - Pharmacology, targets, interactions
"""

import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDrug:
    """
    Unified drug record combining data from multiple sources.
    """
    # Primary identifiers
    name: str
    drugbank_id: Optional[str] = None
    rxcui: Optional[str] = None
    pubchem_cid: Optional[int] = None
    
    # Alternative names
    brand_names: List[str] = field(default_factory=list)
    generic_names: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    
    # Codes
    ndc_codes: List[str] = field(default_factory=list)
    cas_numbers: List[str] = field(default_factory=list)
    unii: Optional[str] = None
    
    # Classification
    therapeutic_classes: List[str] = field(default_factory=list)
    drug_classes: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Chemical properties
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    smiles: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    
    # Physical properties
    xlogp: Optional[float] = None
    tpsa: Optional[float] = None
    h_bond_donors: Optional[int] = None
    h_bond_acceptors: Optional[int] = None
    
    # Pharmacology
    description: Optional[str] = None
    indication: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    pharmacodynamics: Optional[str] = None
    metabolism: Optional[str] = None
    half_life: Optional[str] = None
    
    # Safety
    warnings: Optional[str] = None
    contraindications: Optional[str] = None
    adverse_reactions: Optional[str] = None
    
    # Interactions
    drug_interactions: List[Dict] = field(default_factory=list)
    food_interactions: List[str] = field(default_factory=list)
    
    # Targets
    targets: List[Dict] = field(default_factory=list)
    enzymes: List[Dict] = field(default_factory=list)
    
    # Metadata
    sources: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database insertion."""
        return {
            'name': self.name,
            'drugbank_id': self.drugbank_id,
            'rxcui': self.rxcui,
            'pubchem_cid': self.pubchem_cid,
            'brand_names': self.brand_names,
            'generic_names': self.generic_names,
            'synonyms': self.synonyms,
            'ndc_codes': self.ndc_codes,
            'cas_numbers': self.cas_numbers,
            'unii': self.unii,
            'therapeutic_classes': self.therapeutic_classes,
            'drug_classes': self.drug_classes,
            'categories': self.categories,
            'molecular_formula': self.molecular_formula,
            'molecular_weight': self.molecular_weight,
            'smiles': self.smiles,
            'inchi': self.inchi,
            'inchi_key': self.inchi_key,
            'xlogp': self.xlogp,
            'tpsa': self.tpsa,
            'h_bond_donors': self.h_bond_donors,
            'h_bond_acceptors': self.h_bond_acceptors,
            'description': self.description,
            'indication': self.indication,
            'mechanism_of_action': self.mechanism_of_action,
            'pharmacodynamics': self.pharmacodynamics,
            'metabolism': self.metabolism,
            'half_life': self.half_life,
            'warnings': self.warnings,
            'contraindications': self.contraindications,
            'adverse_reactions': self.adverse_reactions,
            'drug_interactions': self.drug_interactions,
            'food_interactions': self.food_interactions,
            'targets': self.targets,
            'enzymes': self.enzymes,
            'sources': self.sources,
        }


class DrugDataAggregator:
    """
    Aggregates drug data from multiple sources.
    
    Usage:
        from data_sources import RxNormClient, OpenFDAClient, PubChemClient, DrugBankParser
        
        aggregator = DrugDataAggregator(
            rxnorm=RxNormClient(),
            openfda=OpenFDAClient(),
            pubchem=PubChemClient(),
            drugbank=DrugBankParser("drugbank.xml")  # Optional
        )
        
        # Fetch and merge data for a single drug
        drug = aggregator.fetch_drug("metformin")
        
        # Batch fetch multiple drugs
        drugs = aggregator.fetch_drugs(["metformin", "aspirin", "lisinopril"])
        
        # Export unified dataset
        aggregator.export_to_json("drugs.json")
    """
    
    def __init__(
        self,
        rxnorm=None,
        openfda=None,
        pubchem=None,
        drugbank=None
    ):
        self.rxnorm = rxnorm
        self.openfda = openfda
        self.pubchem = pubchem
        self.drugbank = drugbank
        
        # Cache for resolved identifiers
        self._id_cache: Dict[str, Dict[str, Any]] = {}
    
    def fetch_drug(
        self,
        name: str,
        include_interactions: bool = True,
        include_adverse_events: bool = False
    ) -> Optional[UnifiedDrug]:
        """
        Fetch and aggregate drug data from all available sources.
        
        Args:
            name: Drug name to search
            include_interactions: Include drug-drug interactions
            include_adverse_events: Include adverse event data (slow)
        """
        logger.info(f"Fetching drug data for: {name}")
        
        drug = UnifiedDrug(name=name)
        
        # 1. Try RxNorm first (good for names and NDC codes)
        if self.rxnorm:
            try:
                rxnorm_data = self._fetch_from_rxnorm(name)
                if rxnorm_data:
                    self._merge_rxnorm_data(drug, rxnorm_data)
            except Exception as e:
                logger.warning(f"RxNorm fetch failed for {name}: {e}")
        
        # 2. Get OpenFDA label data
        if self.openfda:
            try:
                openfda_data = self._fetch_from_openfda(name)
                if openfda_data:
                    self._merge_openfda_data(drug, openfda_data)
            except Exception as e:
                logger.warning(f"OpenFDA fetch failed for {name}: {e}")
        
        # 3. Get PubChem molecular data
        if self.pubchem:
            try:
                pubchem_data = self._fetch_from_pubchem(name)
                if pubchem_data:
                    self._merge_pubchem_data(drug, pubchem_data)
            except Exception as e:
                logger.warning(f"PubChem fetch failed for {name}: {e}")
        
        # 4. Get DrugBank data (if available)
        if self.drugbank:
            try:
                drugbank_data = self._fetch_from_drugbank(name)
                if drugbank_data:
                    self._merge_drugbank_data(drug, drugbank_data)
            except Exception as e:
                logger.warning(f"DrugBank fetch failed for {name}: {e}")
        
        # 5. Fetch interactions
        if include_interactions:
            drug.drug_interactions = self._fetch_interactions(drug)
        
        # 6. Fetch adverse events (optional, slow)
        if include_adverse_events and self.openfda:
            try:
                events = self.openfda.get_adverse_event_counts(name)
                drug.adverse_reactions = self._format_adverse_events(events)
            except Exception as e:
                logger.warning(f"Adverse event fetch failed for {name}: {e}")
        
        return drug if drug.sources else None
    
    def fetch_drugs(
        self,
        names: List[str],
        max_workers: int = 4,
        **kwargs
    ) -> List[UnifiedDrug]:
        """
        Fetch multiple drugs in parallel.
        
        Args:
            names: List of drug names
            max_workers: Number of parallel threads
            **kwargs: Additional arguments passed to fetch_drug
        """
        drugs = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.fetch_drug, name, **kwargs): name
                for name in names
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    drug = future.result()
                    if drug:
                        drugs.append(drug)
                        logger.info(f"Fetched: {name}")
                except Exception as e:
                    logger.error(f"Failed to fetch {name}: {e}")
        
        return drugs
    
    # ==================== Source-Specific Fetchers ====================
    
    def _fetch_from_rxnorm(self, name: str) -> Optional[Dict]:
        """Fetch data from RxNorm."""
        results = self.rxnorm.search(name)
        if not results:
            results = self.rxnorm.approximate_search(name, max_entries=5)
        
        if results:
            drug = results[0]
            # Get full details
            full_drug = self.rxnorm.get_drug_by_rxcui(drug.rxcui)
            if full_drug:
                full_drug.interactions = self.rxnorm.get_interactions(drug.rxcui)
                return full_drug
        
        return None
    
    def _fetch_from_openfda(self, name: str):
        """Fetch data from OpenFDA."""
        results = self.openfda.search_drug_labels(name)
        return results[0] if results else None
    
    def _fetch_from_pubchem(self, name: str):
        """Fetch data from PubChem."""
        results = self.pubchem.search(name, max_results=1)
        return results[0] if results else None
    
    def _fetch_from_drugbank(self, name: str):
        """Fetch data from DrugBank."""
        results = self.drugbank.search(name, max_results=1)
        return results[0] if results else None
    
    # ==================== Data Merging ====================
    
    def _merge_rxnorm_data(self, drug: UnifiedDrug, data) -> None:
        """Merge RxNorm data into unified drug."""
        drug.rxcui = data.rxcui
        drug.sources.append('RxNorm')
        
        if not drug.name:
            drug.name = data.name
        
        # NDC codes
        drug.ndc_codes.extend(data.ndc_codes or [])
        
        # Drug classes
        for dc in data.drug_classes or []:
            drug.drug_classes.append(dc.get('name', ''))
        
        # Ingredients as synonyms
        for ing in data.ingredients or []:
            if ing.get('name') and ing['name'] not in drug.synonyms:
                drug.synonyms.append(ing['name'])
        
        # RxNorm interactions
        for interaction in data.interactions or []:
            drug.drug_interactions.append({
                'drug_name': interaction.get('drug_name'),
                'rxcui': interaction.get('drug_rxcui'),
                'severity': interaction.get('severity'),
                'description': interaction.get('description'),
                'source': 'RxNorm'
            })
    
    def _merge_openfda_data(self, drug: UnifiedDrug, data) -> None:
        """Merge OpenFDA data into unified drug."""
        drug.sources.append('OpenFDA')
        
        if data.brand_name:
            if data.brand_name not in drug.brand_names:
                drug.brand_names.append(data.brand_name)
        
        if data.generic_name:
            if data.generic_name not in drug.generic_names:
                drug.generic_names.append(data.generic_name)
        
        drug.ndc_codes.extend([
            ndc for ndc in data.ndc_codes
            if ndc not in drug.ndc_codes
        ])
        
        # Pharmacology from label
        if data.indications and not drug.indication:
            drug.indication = data.indications
        
        if data.warnings and not drug.warnings:
            drug.warnings = data.warnings
        
        if data.contraindications and not drug.contraindications:
            drug.contraindications = data.contraindications
        
        if data.drug_interactions and not drug.adverse_reactions:
            # Store raw interaction text from label
            drug.adverse_reactions = data.drug_interactions
        
        # Active ingredients
        for ing in data.active_ingredients:
            name = ing.get('name', '')
            if name and name not in drug.synonyms:
                drug.synonyms.append(name)
    
    def _merge_pubchem_data(self, drug: UnifiedDrug, data) -> None:
        """Merge PubChem data into unified drug."""
        drug.pubchem_cid = data.cid
        drug.sources.append('PubChem')
        
        # Chemical structure
        if not drug.smiles:
            drug.smiles = data.canonical_smiles
        if not drug.inchi:
            drug.inchi = data.inchi
        if not drug.inchi_key:
            drug.inchi_key = data.inchi_key
        
        # Molecular properties
        if not drug.molecular_formula:
            drug.molecular_formula = data.molecular_formula
        if not drug.molecular_weight:
            drug.molecular_weight = data.molecular_weight
        
        # Physical properties
        if data.xlogp is not None:
            drug.xlogp = data.xlogp
        if data.tpsa is not None:
            drug.tpsa = data.tpsa
        if data.h_bond_donor_count is not None:
            drug.h_bond_donors = data.h_bond_donor_count
        if data.h_bond_acceptor_count is not None:
            drug.h_bond_acceptors = data.h_bond_acceptor_count
        
        # Synonyms and CAS numbers
        for syn in data.synonyms:
            if syn not in drug.synonyms:
                drug.synonyms.append(syn)
        
        for cas in data.cas_numbers:
            if cas not in drug.cas_numbers:
                drug.cas_numbers.append(cas)
    
    def _merge_drugbank_data(self, drug: UnifiedDrug, data) -> None:
        """Merge DrugBank data into unified drug."""
        drug.drugbank_id = data.drugbank_id
        drug.sources.append('DrugBank')
        
        # Basic info
        if not drug.description:
            drug.description = data.description
        if data.cas_number and data.cas_number not in drug.cas_numbers:
            drug.cas_numbers.append(data.cas_number)
        if data.unii:
            drug.unii = data.unii
        
        # Chemical properties (if not from PubChem)
        if not drug.smiles:
            drug.smiles = data.smiles
        if not drug.inchi:
            drug.inchi = data.inchi
        if not drug.inchi_key:
            drug.inchi_key = data.inchi_key
        if not drug.molecular_formula:
            drug.molecular_formula = data.molecular_formula
        if not drug.molecular_weight:
            drug.molecular_weight = data.molecular_weight
        
        # Pharmacology
        if not drug.indication:
            drug.indication = data.indication
        if not drug.mechanism_of_action:
            drug.mechanism_of_action = data.mechanism_of_action
        if not drug.pharmacodynamics:
            drug.pharmacodynamics = data.pharmacodynamics
        if not drug.metabolism:
            drug.metabolism = data.metabolism
        if not drug.half_life:
            drug.half_life = data.half_life
        
        # Classifications
        drug.categories.extend([
            cat for cat in data.categories
            if cat not in drug.categories
        ])
        
        # Synonyms
        for syn in data.synonyms:
            if syn not in drug.synonyms:
                drug.synonyms.append(syn)
        
        # DrugBank interactions
        for interaction in data.drug_interactions:
            drug.drug_interactions.append({
                'drug_name': interaction.get('name'),
                'drugbank_id': interaction.get('drugbank_id'),
                'description': interaction.get('description'),
                'source': 'DrugBank'
            })
        
        # Food interactions
        drug.food_interactions.extend(data.food_interactions)
        
        # Targets and enzymes
        drug.targets.extend(data.targets)
        drug.enzymes.extend(data.enzymes)
    
    # ==================== Interaction Aggregation ====================
    
    def _fetch_interactions(self, drug: UnifiedDrug) -> List[Dict]:
        """
        Fetch and deduplicate interactions from all sources.
        """
        interactions = []
        seen: Set[str] = set()
        
        # Existing interactions already merged
        for interaction in drug.drug_interactions:
            key = interaction.get('drug_name', '').lower()
            if key and key not in seen:
                seen.add(key)
                interactions.append(interaction)
        
        return interactions
    
    def _format_adverse_events(self, events: Dict[str, int]) -> str:
        """Format adverse event counts as readable text."""
        if not events:
            return None
        
        # Sort by count and take top 10
        sorted_events = sorted(events.items(), key=lambda x: x[1], reverse=True)[:10]
        
        lines = ["Most commonly reported adverse events:"]
        for event, count in sorted_events:
            lines.append(f"- {event}: {count} reports")
        
        return "\n".join(lines)
    
    # ==================== Export ====================
    
    def export_to_json(self, filepath: str, drugs: List[UnifiedDrug] = None) -> None:
        """Export drugs to JSON file."""
        import json
        
        data = [drug.to_dict() for drug in (drugs or [])]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(data)} drugs to {filepath}")
    
    def export_to_csv(self, filepath: str, drugs: List[UnifiedDrug] = None) -> None:
        """Export drugs to CSV file (flattened)."""
        import csv
        
        if not drugs:
            return
        
        # Flatten fields for CSV
        fieldnames = [
            'name', 'drugbank_id', 'rxcui', 'pubchem_cid',
            'molecular_formula', 'molecular_weight', 'smiles',
            'therapeutic_classes', 'indication', 'mechanism_of_action',
            'sources'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for drug in drugs:
                row = {
                    'name': drug.name,
                    'drugbank_id': drug.drugbank_id,
                    'rxcui': drug.rxcui,
                    'pubchem_cid': drug.pubchem_cid,
                    'molecular_formula': drug.molecular_formula,
                    'molecular_weight': drug.molecular_weight,
                    'smiles': drug.smiles,
                    'therapeutic_classes': ', '.join(drug.therapeutic_classes),
                    'indication': (drug.indication or '')[:500],
                    'mechanism_of_action': (drug.mechanism_of_action or '')[:500],
                    'sources': ', '.join(drug.sources)
                }
                writer.writerow(row)
        
        logger.info(f"Exported {len(drugs)} drugs to {filepath}")


# ==================== Convenience Functions ====================

def create_aggregator(
    include_drugbank: bool = False,
    drugbank_path: str = None
) -> DrugDataAggregator:
    """
    Create an aggregator with default clients.
    
    Args:
        include_drugbank: Include DrugBank parser (requires XML file)
        drugbank_path: Path to DrugBank XML file
    """
    from .rxnorm import RxNormClient
    from .openfda import OpenFDAClient
    from .pubchem import PubChemClient
    
    drugbank = None
    if include_drugbank and drugbank_path:
        from .drugbank import DrugBankParser
        drugbank = DrugBankParser(drugbank_path)
    
    return DrugDataAggregator(
        rxnorm=RxNormClient(),
        openfda=OpenFDAClient(),
        pubchem=PubChemClient(),
        drugbank=drugbank
    )
