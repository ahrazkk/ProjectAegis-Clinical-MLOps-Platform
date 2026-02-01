"""
Data Sources for Project Aegis Drug Database Expansion

This module provides data ingestion from multiple pharmaceutical databases:
- RxNorm (NIH) - Drug names, NDC codes, relationships
- OpenFDA - Drug labels, NDC info, adverse events
- DrugBank Open - Drug properties, interactions
- PubChem - Molecular structures, properties
- KEGG Drug - Pathways, targets
"""

from .rxnorm import RxNormClient
from .openfda import OpenFDAClient
from .drugbank import DrugBankParser
from .pubchem import PubChemClient
from .aggregator import DrugDataAggregator

__all__ = [
    'RxNormClient',
    'OpenFDAClient', 
    'DrugBankParser',
    'PubChemClient',
    'DrugDataAggregator'
]
