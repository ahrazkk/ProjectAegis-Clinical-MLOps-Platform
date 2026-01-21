import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from difflib import get_close_matches

logger = logging.getLogger(__name__)

class DrugService:
    """
    Service to handle drug data operations using a local JSON database
    as a lightweight alternative to a full SQL/Neo4j setup for now.
    """
    
    _instance = None
    _drugs_db = []
    _drugs_map = {} # Quick lookup by name and ID
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DrugService, cls).__new__(cls)
            cls._instance._load_db()
        return cls._instance

    def _load_db(self):
        """Load drug database from JSON file."""
        try:
            # Path to web/data/drug_db.json
            base_dir = Path(__file__).parent.parent.parent # web/
            data_file = base_dir / 'data' / 'drug_db.json'
            
            if not data_file.exists():
                logger.warning(f"Drug DB not found at {data_file}. Using empty list.")
                self._drugs_db = []
                return

            with open(data_file, 'r', encoding='utf-8') as f:
                self._drugs_db = json.load(f)
                
            # Create indexing maps
            self._drugs_map = {}
            for drug in self._drugs_db:
                self._drugs_map[drug['name'].lower()] = drug
                if drug.get('drugbank_id'):
                    self._drugs_map[drug['drugbank_id'].lower()] = drug
                    
            logger.info(f"Loaded {len(self._drugs_db)} drugs from local DB.")
            
        except Exception as e:
            logger.error(f"Failed to load drug DB: {e}")
            self._drugs_db = []

    def search_drugs(self, query: str, limit: int = 10) -> List[Dict]:
        """Search drugs by name or ID."""
        query = query.lower().strip()
        if not query:
            return []

        # 1. Exact match
        if query in self._drugs_map:
            return [self._drugs_map[query]]

        # 2. Substring match (Filter)
        results = [
            d for d in self._drugs_db 
            if query in d['name'].lower() or query in (d.get('drugbank_id') or '').lower()
        ]
        
        # 3. Sort by relevance (exact starts-with first)
        results.sort(key=lambda x: 0 if x['name'].lower().startswith(query) else 1)
        
        return results[:limit]

    def get_drug(self, identifier: str) -> Optional[Dict]:
        """Get single drug by name or ID (case-insensitive)."""
        return self._drugs_map.get(identifier.lower().strip())

    def reload(self):
        """Force reload of the database."""
        self._load_db()

# Singleton accessor
def get_drug_service():
    return DrugService()
