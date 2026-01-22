"""
External Data Importers for DDI Enhancement

This module provides importers for various open pharmaceutical datasets:
1. SIDER - Side Effect Resource (drug side effects)
2. TWOSIDES - Polypharmacy side effects  
3. OpenFDA FAERS - Adverse event reports
4. DrugCentral - Open drug data

Run individual importers:
    python manage.py import_external_data --source sider
    python manage.py import_external_data --source twosides
    python manage.py import_external_data --source openfda
    python manage.py import_external_data --source all
"""

import os
import csv
import gzip
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from io import StringIO

from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)


# ============================================================================
# DATA SOURCE URLS (All freely available)
# ============================================================================

DATA_SOURCES = {
    'sider': {
        'side_effects': 'http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz',
        'drug_names': 'http://sideeffects.embl.de/media/download/drug_names.tsv',
        'description': 'Side Effect Resource - Drug side effects from package inserts'
    },
    'twosides': {
        'interactions': 'http://tatonettilab.org/resources/nsides/TWOSIDES.csv.gz',
        'description': 'Polypharmacy side effects from FAERS mining'
    },
    'openfda': {
        'base_url': 'https://api.fda.gov/drug/event.json',
        'description': 'FDA Adverse Event Reporting System'
    },
    'drugcentral': {
        'interactions': 'https://unmtid-shinyapps.net/download/DrugCentral/drugcentral_interactions.tsv',
        'drugs': 'https://unmtid-shinyapps.net/download/DrugCentral/drugcentral_drugs.tsv',
        'description': 'Open drug database with DDI data'
    }
}


class SIDERImporter:
    """
    Import side effects from SIDER database.
    
    SIDER contains side effects extracted from drug package inserts.
    ~140,000 drug-side effect pairs covering ~1,400 drugs.
    """
    
    def __init__(self):
        self.drug_names = {}  # CID -> name mapping
        self.side_effects = defaultdict(list)  # drug_name (lowercase) -> [side effects]
        self.stats = defaultdict(int)
    
    def download_and_parse(self, data_dir: Path) -> bool:
        """Download and parse SIDER data."""
        try:
            # First, build CID -> name mapping from drug_names file
            logger.info("Downloading SIDER drug names...")
            response = requests.get(DATA_SOURCES['sider']['drug_names'], timeout=60)
            cid_to_name = {}
            if response.status_code == 200:
                for line in response.text.strip().split('\n'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        full_cid = parts[0]  # e.g., CID100002244
                        name = parts[1].lower()
                        # Store multiple formats of CID for matching
                        cid_to_name[full_cid] = name
                        # Also store without leading zeros: CID100002244 -> 2244
                        short_cid = full_cid.replace('CID1', '').lstrip('0')
                        cid_to_name[short_cid] = name
                        cid_to_name[f"CID{short_cid}"] = name
                        self.drug_names[full_cid] = name
                        self.stats['drugs'] += 1
            
            logger.info(f"Loaded {len(cid_to_name)} CID mappings")
            
            # Download side effects (gzipped)
            logger.info("Downloading SIDER side effects...")
            response = requests.get(DATA_SOURCES['sider']['side_effects'], timeout=120)
            if response.status_code == 200:
                import gzip
                from io import BytesIO
                
                with gzip.open(BytesIO(response.content), 'rt') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 6:
                            stitch_id = parts[1]  # STITCH compound ID (e.g., CID000002244)
                            side_effect = parts[5]  # MedDRA term
                            
                            # Try to find drug name from various CID formats
                            drug_name = None
                            
                            # Try direct lookup
                            if stitch_id in cid_to_name:
                                drug_name = cid_to_name[stitch_id]
                            else:
                                # Try stripping CID prefix and leading zeros
                                short_id = stitch_id.replace('CID', '').lstrip('0')
                                if short_id in cid_to_name:
                                    drug_name = cid_to_name[short_id]
                                elif f"CID{short_id}" in cid_to_name:
                                    drug_name = cid_to_name[f"CID{short_id}"]
                            
                            if drug_name:
                                self.side_effects[drug_name].append(side_effect)
                                self.stats['side_effects'] += 1
                            else:
                                # Use the raw ID as fallback
                                self.side_effects[stitch_id.lower()].append(side_effect)
                                self.stats['unmapped'] += 1
                
                logger.info(f"Mapped {self.stats['side_effects']} effects, {self.stats.get('unmapped', 0)} unmapped")
                
                # Save to local cache with proper name keys
                cache_file = data_dir / 'sider_cache.json'
                with open(cache_file, 'w') as f:
                    json.dump({
                        'drug_names': self.drug_names,
                        'side_effects': dict(self.side_effects)
                    }, f)
                logger.info(f"Cached SIDER data to {cache_file}")
                
                return True
                
        except Exception as e:
            logger.error(f"SIDER download failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_from_cache(self, data_dir: Path) -> bool:
        """Load from local cache if available."""
        cache_file = data_dir / 'sider_cache.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                self.drug_names = data.get('drug_names', {})
                self.side_effects = defaultdict(list, data.get('side_effects', {}))
                self.stats['drugs'] = len(self.drug_names)
                self.stats['side_effects'] = sum(len(v) for v in self.side_effects.values())
                return True
        return False
    
    def get_side_effects(self, drug_name: str, limit: int = 20) -> List[str]:
        """Get side effects for a drug."""
        effects = self.side_effects.get(drug_name.lower(), [])
        # Return unique effects, sorted by frequency
        from collections import Counter
        counts = Counter(effects)
        return [effect for effect, _ in counts.most_common(limit)]
    
    def import_to_db(self):
        """Import side effects to the database."""
        # This will be implemented to store in a new table
        pass


class TWOSIDESImporter:
    """
    Import polypharmacy interactions from TWOSIDES.
    
    TWOSIDES contains drug-drug-side effect associations mined from FAERS.
    ~63,000 significant drug pair interactions.
    """
    
    def __init__(self):
        self.interactions = []  # (drug1, drug2, side_effect, score)
        self.stats = defaultdict(int)
    
    def download_and_parse(self, data_dir: Path, limit: int = None) -> bool:
        """Download and parse TWOSIDES data."""
        try:
            logger.info("Downloading TWOSIDES data (this may take a while)...")
            
            # TWOSIDES is large (~500MB), so we'll stream it
            response = requests.get(
                DATA_SOURCES['twosides']['interactions'], 
                stream=True, 
                timeout=300
            )
            
            if response.status_code == 200:
                import gzip
                from io import BytesIO
                
                # Decompress in memory
                buffer = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    buffer.write(chunk)
                buffer.seek(0)
                
                with gzip.open(buffer, 'rt') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if limit and i >= limit:
                            break
                        
                        drug1 = row.get('drug1_name', row.get('stitch_id1', ''))
                        drug2 = row.get('drug2_name', row.get('stitch_id2', ''))
                        side_effect = row.get('event_name', row.get('side_effect', ''))
                        score = float(row.get('score', row.get('PRR', 0)))
                        
                        if drug1 and drug2 and side_effect:
                            self.interactions.append({
                                'drug1': drug1,
                                'drug2': drug2,
                                'side_effect': side_effect,
                                'score': score
                            })
                            self.stats['interactions'] += 1
                
                # Cache locally
                cache_file = data_dir / 'twosides_cache.json'
                with open(cache_file, 'w') as f:
                    json.dump(self.interactions[:100000], f)  # Limit cache size
                
                return True
                
        except Exception as e:
            logger.error(f"TWOSIDES download failed: {e}")
            return False
    
    def load_from_cache(self, data_dir: Path) -> bool:
        """Load from local cache."""
        cache_file = data_dir / 'twosides_cache.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.interactions = json.load(f)
                self.stats['interactions'] = len(self.interactions)
                return True
        return False
    
    def get_pair_effects(self, drug1: str, drug2: str, limit: int = 10) -> List[Dict]:
        """Get side effects for a drug pair."""
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        
        effects = []
        for interaction in self.interactions:
            d1 = interaction['drug1'].lower()
            d2 = interaction['drug2'].lower()
            if (d1 == drug1_lower and d2 == drug2_lower) or \
               (d1 == drug2_lower and d2 == drug1_lower):
                effects.append({
                    'side_effect': interaction['side_effect'],
                    'score': interaction['score']
                })
        
        # Sort by score and return top
        effects.sort(key=lambda x: x['score'], reverse=True)
        return effects[:limit]


class OpenFDAImporter:
    """
    Query OpenFDA FAERS for real-world adverse events.
    
    OpenFDA provides free API access to FDA Adverse Event Reporting System.
    Millions of adverse event reports available.
    """
    
    BASE_URL = 'https://api.fda.gov/drug/event.json'
    
    def __init__(self):
        self.cache = {}
        self.stats = defaultdict(int)
    
    def get_adverse_events(self, drug_name: str, limit: int = 10) -> Dict:
        """Get adverse event statistics for a drug."""
        try:
            # Query OpenFDA
            params = {
                'search': f'patient.drug.medicinalproduct:"{drug_name}"',
                'count': 'patient.reaction.reactionmeddrapt.exact',
                'limit': limit
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                return {
                    'drug': drug_name,
                    'total_reports': data.get('meta', {}).get('results', {}).get('total', 0),
                    'top_reactions': [
                        {'reaction': r['term'], 'count': r['count']}
                        for r in results
                    ]
                }
            else:
                return {'drug': drug_name, 'error': f'API returned {response.status_code}'}
                
        except Exception as e:
            logger.error(f"OpenFDA query failed: {e}")
            return {'drug': drug_name, 'error': str(e)}
    
    def get_pair_reports(self, drug1: str, drug2: str) -> Dict:
        """Get adverse event reports mentioning both drugs."""
        try:
            # Query for reports containing both drugs
            params = {
                'search': f'patient.drug.medicinalproduct:"{drug1}" AND patient.drug.medicinalproduct:"{drug2}"',
                'count': 'patient.reaction.reactionmeddrapt.exact',
                'limit': 10
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                return {
                    'drug1': drug1,
                    'drug2': drug2,
                    'total_reports': data.get('meta', {}).get('results', {}).get('total', 0),
                    'top_reactions': [
                        {'reaction': r['term'], 'count': r['count']}
                        for r in results
                    ]
                }
            else:
                return {'drug1': drug1, 'drug2': drug2, 'total_reports': 0, 'top_reactions': []}
                
        except Exception as e:
            logger.error(f"OpenFDA pair query failed: {e}")
            return {'drug1': drug1, 'drug2': drug2, 'error': str(e)}


class Command(BaseCommand):
    help = 'Import external pharmaceutical data sources'

    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            choices=['sider', 'twosides', 'openfda', 'drugcentral', 'all'],
            default='all',
            help='Data source to import'
        )
        parser.add_argument(
            '--refresh',
            action='store_true',
            help='Force re-download even if cache exists'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of records (for testing)'
        )

    def handle(self, *args, **options):
        data_dir = Path(settings.BASE_DIR) / 'data' / 'external'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        source = options['source']
        refresh = options['refresh']
        limit = options['limit']
        
        if source in ['sider', 'all']:
            self.import_sider(data_dir, refresh)
        
        if source in ['twosides', 'all']:
            self.import_twosides(data_dir, refresh, limit)
        
        if source in ['openfda', 'all']:
            self.test_openfda()
    
    def import_sider(self, data_dir: Path, refresh: bool):
        """Import SIDER side effects."""
        self.stdout.write('\n=== Importing SIDER Side Effects ===')
        
        importer = SIDERImporter()
        
        if not refresh and importer.load_from_cache(data_dir):
            self.stdout.write(self.style.SUCCESS(f'  Loaded from cache'))
        else:
            self.stdout.write('  Downloading from SIDER...')
            if importer.download_and_parse(data_dir):
                self.stdout.write(self.style.SUCCESS('  Download complete'))
            else:
                self.stdout.write(self.style.ERROR('  Download failed'))
                return
        
        self.stdout.write(f"  Drugs: {importer.stats['drugs']}")
        self.stdout.write(f"  Side effect associations: {importer.stats['side_effects']}")
        
        # Test lookup
        test_drug = 'aspirin'
        effects = importer.get_side_effects(test_drug, limit=5)
        if effects:
            self.stdout.write(f"  Sample - {test_drug}: {', '.join(effects[:3])}...")
    
    def import_twosides(self, data_dir: Path, refresh: bool, limit: int):
        """Import TWOSIDES polypharmacy data."""
        self.stdout.write('\n=== Importing TWOSIDES Polypharmacy Data ===')
        self.stdout.write('  Note: TWOSIDES is ~500MB, this may take several minutes')
        
        importer = TWOSIDESImporter()
        
        if not refresh and importer.load_from_cache(data_dir):
            self.stdout.write(self.style.SUCCESS('  Loaded from cache'))
        else:
            self.stdout.write('  Downloading from Tatonetti Lab...')
            # Use a smaller limit for first download
            if importer.download_and_parse(data_dir, limit=limit or 50000):
                self.stdout.write(self.style.SUCCESS('  Download complete'))
            else:
                self.stdout.write(self.style.WARNING('  Download failed or file not available'))
                self.stdout.write('  TWOSIDES may require manual download from:')
                self.stdout.write('  http://tatonettilab.org/resources/nsides/')
                return
        
        self.stdout.write(f"  Interactions loaded: {importer.stats['interactions']}")
    
    def test_openfda(self):
        """Test OpenFDA API."""
        self.stdout.write('\n=== Testing OpenFDA FAERS API ===')
        
        importer = OpenFDAImporter()
        
        # Test single drug query
        result = importer.get_adverse_events('aspirin', limit=5)
        if 'error' not in result:
            self.stdout.write(self.style.SUCCESS(f"  Aspirin: {result['total_reports']} reports"))
            if result['top_reactions']:
                top = result['top_reactions'][0]
                self.stdout.write(f"  Top reaction: {top['reaction']} ({top['count']} reports)")
        else:
            self.stdout.write(self.style.WARNING(f"  API test failed: {result.get('error')}"))
        
        # Test pair query
        pair_result = importer.get_pair_reports('warfarin', 'aspirin')
        if 'error' not in pair_result:
            self.stdout.write(self.style.SUCCESS(
                f"  Warfarin+Aspirin: {pair_result['total_reports']} co-reported events"
            ))


# ============================================================================
# Global instances for use in views
# ============================================================================

_sider_importer = None
_twosides_importer = None
_openfda_importer = None


def get_sider_importer() -> SIDERImporter:
    """Get or create SIDER importer instance."""
    global _sider_importer
    if _sider_importer is None:
        _sider_importer = SIDERImporter()
        data_dir = Path(settings.BASE_DIR) / 'data' / 'external'
        _sider_importer.load_from_cache(data_dir)
    return _sider_importer


def get_twosides_importer() -> TWOSIDESImporter:
    """Get or create TWOSIDES importer instance."""
    global _twosides_importer
    if _twosides_importer is None:
        _twosides_importer = TWOSIDESImporter()
        data_dir = Path(settings.BASE_DIR) / 'data' / 'external'
        _twosides_importer.load_from_cache(data_dir)
    return _twosides_importer


def get_openfda_importer() -> OpenFDAImporter:
    """Get or create OpenFDA importer instance."""
    global _openfda_importer
    if _openfda_importer is None:
        _openfda_importer = OpenFDAImporter()
    return _openfda_importer
