"""
Data Sources for Project Aegis Drug Database Expansion

This module provides data ingestion from multiple pharmaceutical databases:
- RxNorm (NIH) - Drug names, NDC codes, relationships
- OpenFDA - Drug labels, adverse events
- DrugBank Open - Drug properties, interactions (requires XML download)
- PubChem - Molecular structures, properties

Usage:
    from ddi_api.services.data_sources import (
        RxNormClient,
        OpenFDAClient,
        PubChemClient,
        DrugBankParser,
        DrugDataAggregator,
        create_aggregator
    )
    
    # Create individual clients
    rxnorm = RxNormClient()
    drugs = rxnorm.search("metformin")
    
    # Or use the aggregator for unified data
    aggregator = create_aggregator()
    drug = aggregator.fetch_drug("aspirin")
"""

from .rxnorm import RxNormClient, RxNormDrug
from .openfda import OpenFDAClient, OpenFDADrug, AdverseEvent
from .pubchem import PubChemClient, PubChemCompound
from .drugbank import DrugBankParser, DrugBankDrug
from .aggregator import DrugDataAggregator, UnifiedDrug, create_aggregator

__all__ = [
    # Clients
    'RxNormClient',
    'OpenFDAClient', 
    'PubChemClient',
    'DrugBankParser',
    'DrugDataAggregator',
    
    # Data classes
    'RxNormDrug',
    'OpenFDADrug',
    'AdverseEvent',
    'PubChemCompound',
    'DrugBankDrug',
    'UnifiedDrug',
    
    # Utilities
    'create_aggregator',
]
