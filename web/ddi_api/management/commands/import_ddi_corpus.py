"""
DDI Corpus 2013 Importer

This script parses the DDI Corpus 2013 XML files and loads:
1. All sentences with DDI pairs into the DDI Sentence Database
2. Unique drug pairs into the Knowledge Graph (Neo4j)

DDI Corpus 2013 Structure:
- Train/Test folders with XML files
- Each file contains <sentence> elements with <entity> (drugs) and <pair> (interactions)
- Interaction types: mechanism, effect, advise, int
- ddi="true" means there's an interaction, ddi="false" means no interaction

Usage:
    python manage.py import_ddi_corpus --path /path/to/DDI_Corpus
    python manage.py import_ddi_corpus --path /path/to/DDI_Corpus --test-only
    python manage.py import_ddi_corpus --path /path/to/DDI_Corpus --stats-only

The corpus contains ~27,000 sentences with ~5,000 positive DDI pairs.
"""

import os
import re
import logging
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class DDIEntity:
    """A drug entity in a sentence."""
    id: str
    text: str
    char_offset: str
    entity_type: str  # drug, brand, group, drug_n


@dataclass
class DDIPair:
    """A drug-drug pair with interaction info."""
    id: str
    e1_id: str
    e2_id: str
    ddi: bool  # True if interaction exists
    ddi_type: Optional[str] = None  # mechanism, effect, advise, int


@dataclass
class DDISentence:
    """A sentence from the DDI Corpus."""
    id: str
    text: str
    entities: List[DDIEntity] = field(default_factory=list)
    pairs: List[DDIPair] = field(default_factory=list)


@dataclass 
class DDIDocument:
    """A document from the DDI Corpus."""
    id: str
    sentences: List[DDISentence] = field(default_factory=list)


class DDICorpusParser:
    """Parser for DDI Corpus 2013 XML files."""
    
    def __init__(self):
        self.documents = []
        self.stats = defaultdict(int)
        self.unique_drugs: Set[str] = set()
        self.unique_pairs: Set[Tuple[str, str]] = set()
        self.interaction_sentences = []
    
    def parse_file(self, file_path: Path) -> Optional[DDIDocument]:
        """Parse a single XML file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            doc = DDIDocument(id=root.get('id', ''))
            
            for sent_elem in root.findall('.//sentence'):
                sentence = self._parse_sentence(sent_elem)
                if sentence:
                    doc.sentences.append(sentence)
                    self.stats['total_sentences'] += 1
                    
                    # Track entities
                    for entity in sentence.entities:
                        self.unique_drugs.add(entity.text.lower())
                    
                    # Track pairs
                    for pair in sentence.pairs:
                        if pair.ddi:
                            self.stats['positive_pairs'] += 1
                            self.stats[f'type_{pair.ddi_type or "unknown"}'] += 1
                            
                            # Get entity texts for the pair
                            e1_text = next((e.text for e in sentence.entities if e.id == pair.e1_id), None)
                            e2_text = next((e.text for e in sentence.entities if e.id == pair.e2_id), None)
                            
                            if e1_text and e2_text:
                                # Normalize and add to unique pairs
                                pair_key = tuple(sorted([e1_text.lower(), e2_text.lower()]))
                                self.unique_pairs.add(pair_key)
                                
                                # Store for database import
                                self.interaction_sentences.append({
                                    'drug1': e1_text,
                                    'drug2': e2_text,
                                    'sentence': sentence.text,
                                    'ddi_type': pair.ddi_type or 'int',
                                    'source': 'ddi_corpus',
                                    'doc_id': doc.id,
                                    'sent_id': sentence.id
                                })
                        else:
                            self.stats['negative_pairs'] += 1
            
            self.documents.append(doc)
            return doc
            
        except ET.ParseError as e:
            logger.error(f"XML parse error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_sentence(self, sent_elem) -> Optional[DDISentence]:
        """Parse a sentence element."""
        sentence = DDISentence(
            id=sent_elem.get('id', ''),
            text=sent_elem.get('text', '')
        )
        
        # Parse entities
        for entity_elem in sent_elem.findall('entity'):
            entity = DDIEntity(
                id=entity_elem.get('id', ''),
                text=entity_elem.get('text', ''),
                char_offset=entity_elem.get('charOffset', ''),
                entity_type=entity_elem.get('type', 'drug')
            )
            sentence.entities.append(entity)
        
        # Parse pairs
        for pair_elem in sent_elem.findall('pair'):
            ddi_str = pair_elem.get('ddi', 'false')
            pair = DDIPair(
                id=pair_elem.get('id', ''),
                e1_id=pair_elem.get('e1', ''),
                e2_id=pair_elem.get('e2', ''),
                ddi=ddi_str.lower() == 'true',
                ddi_type=pair_elem.get('type')
            )
            sentence.pairs.append(pair)
        
        return sentence
    
    def parse_directory(self, dir_path: Path, file_pattern: str = '*.xml'):
        """Parse all XML files in a directory."""
        xml_files = list(dir_path.glob(file_pattern))
        self.stats['total_files'] += len(xml_files)
        
        for xml_file in xml_files:
            self.parse_file(xml_file)
    
    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return {
            **dict(self.stats),
            'unique_drugs': len(self.unique_drugs),
            'unique_drug_pairs': len(self.unique_pairs),
            'interaction_sentences': len(self.interaction_sentences)
        }


class Command(BaseCommand):
    help = 'Import DDI Corpus 2013 data into the sentence database and knowledge graph'

    def add_arguments(self, parser):
        parser.add_argument(
            '--path',
            type=str,
            required=True,
            help='Path to DDI Corpus directory (containing Train/Test folders or XML files)'
        )
        parser.add_argument(
            '--train-only',
            action='store_true',
            help='Only import training data'
        )
        parser.add_argument(
            '--test-only',
            action='store_true',
            help='Only import test data'
        )
        parser.add_argument(
            '--stats-only',
            action='store_true',
            help='Only show statistics, do not import'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing DDI Corpus data before importing'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of sentences to import (for testing)'
        )
        parser.add_argument(
            '--positive-only',
            action='store_true',
            default=True,
            help='Only import sentences with positive DDI pairs (default: True)'
        )
        parser.add_argument(
            '--include-negative',
            action='store_true',
            help='Also import negative (no interaction) pairs'
        )

    def handle(self, *args, **options):
        corpus_path = Path(options['path'])
        
        if not corpus_path.exists():
            self.stdout.write(self.style.ERROR(f'Path not found: {corpus_path}'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Scanning DDI Corpus at: {corpus_path}'))
        
        # Initialize parser
        parser = DDICorpusParser()
        
        # Find directories to parse
        dirs_to_parse = []
        
        # Check for standard DDI Corpus structure
        train_dir = corpus_path / 'Train'
        test_dir = corpus_path / 'Test'
        
        # Also check for DrugBank/MedLine subdirectories
        train_drugbank = corpus_path / 'Train' / 'DrugBank'
        train_medline = corpus_path / 'Train' / 'MedLine'
        test_drugbank = corpus_path / 'Test' / 'DrugBank'
        test_medline = corpus_path / 'Test' / 'MedLine'
        
        if not options['test_only']:
            if train_drugbank.exists():
                dirs_to_parse.append(('Train/DrugBank', train_drugbank))
            if train_medline.exists():
                dirs_to_parse.append(('Train/MedLine', train_medline))
            if train_dir.exists() and not train_drugbank.exists():
                dirs_to_parse.append(('Train', train_dir))
        
        if not options['train_only']:
            if test_drugbank.exists():
                dirs_to_parse.append(('Test/DrugBank', test_drugbank))
            if test_medline.exists():
                dirs_to_parse.append(('Test/MedLine', test_medline))
            if test_dir.exists() and not test_drugbank.exists():
                dirs_to_parse.append(('Test', test_dir))
        
        # If no standard structure, try parsing the directory directly
        if not dirs_to_parse:
            xml_files = list(corpus_path.glob('*.xml'))
            if xml_files:
                dirs_to_parse.append(('Root', corpus_path))
            else:
                # Try recursive search
                xml_files = list(corpus_path.rglob('*.xml'))
                if xml_files:
                    self.stdout.write(f'Found {len(xml_files)} XML files recursively')
                    for xml_file in xml_files:
                        parser.parse_file(xml_file)
        
        # Parse directories
        for name, dir_path in dirs_to_parse:
            self.stdout.write(f'Parsing {name}...')
            parser.parse_directory(dir_path)
        
        # Show statistics
        stats = parser.get_stats()
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=== DDI Corpus Statistics ==='))
        self.stdout.write(f"  Files parsed: {stats.get('total_files', 0)}")
        self.stdout.write(f"  Total sentences: {stats.get('total_sentences', 0)}")
        self.stdout.write(f"  Positive DDI pairs: {stats.get('positive_pairs', 0)}")
        self.stdout.write(f"  Negative DDI pairs: {stats.get('negative_pairs', 0)}")
        self.stdout.write(f"  Unique drugs: {stats.get('unique_drugs', 0)}")
        self.stdout.write(f"  Unique drug pairs: {stats.get('unique_drug_pairs', 0)}")
        self.stdout.write(f"  Interaction sentences: {stats.get('interaction_sentences', 0)}")
        self.stdout.write('')
        self.stdout.write('  By interaction type:')
        for key, value in stats.items():
            if key.startswith('type_'):
                self.stdout.write(f"    {key[5:]}: {value}")
        
        if options['stats_only']:
            return
        
        # Import to databases
        self.stdout.write('')
        self.stdout.write('Importing to databases...')
        
        sentences_to_import = parser.interaction_sentences
        
        if options['limit']:
            sentences_to_import = sentences_to_import[:options['limit']]
        
        # Import to DDI Sentence Database
        self.import_to_sentence_db(sentences_to_import, options['clear'])
        
        # Import to Knowledge Graph
        self.import_to_knowledge_graph(parser.unique_drugs, parser.unique_pairs, options['clear'])
        
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('Import complete!'))

    def import_to_sentence_db(self, sentences: List[Dict], clear: bool = False):
        """Import sentences to the DDI Sentence Database."""
        try:
            from ddi_api.services.ddi_sentence_db import get_ddi_sentence_db
            
            db = get_ddi_sentence_db()
            
            if clear:
                # Clear existing DDI Corpus entries
                cursor = db.connection.cursor()
                cursor.execute("DELETE FROM ddi_sentences WHERE source = 'ddi_corpus'")
                db.connection.commit()
                self.stdout.write('  Cleared existing DDI Corpus sentences')
            
            # Bulk insert
            added = 0
            skipped = 0
            
            for sent in sentences:
                try:
                    # Add entity markers to sentence for model input
                    marked_sentence = self._add_entity_markers(
                        sent['sentence'],
                        sent['drug1'],
                        sent['drug2']
                    )
                    
                    db.add_sentence(
                        drug1=sent['drug1'],
                        drug2=sent['drug2'],
                        sentence=marked_sentence,
                        interaction_type=sent['ddi_type'],
                        source='ddi_corpus',
                        confidence=0.95,  # High confidence for corpus sentences
                        pmid=sent.get('doc_id', '')
                    )
                    added += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        logger.debug(f"Skipped sentence: {e}")
            
            self.stdout.write(self.style.SUCCESS(f'  DDI Sentence DB: Added {added} sentences ({skipped} skipped)'))
            
            # Show updated stats
            stats = db.get_stats()
            self.stdout.write(f"  Total sentences now: {stats.get('total_sentences', 0)}")
            
        except ImportError:
            self.stdout.write(self.style.WARNING('  DDI Sentence DB not available, skipping'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  DDI Sentence DB import failed: {e}'))

    def import_to_knowledge_graph(self, drugs: Set[str], pairs: Set[Tuple[str, str]], 
                                   clear: bool = False):
        """Import drugs and interactions to Neo4j Knowledge Graph."""
        try:
            from ddi_api.services.knowledge_graph import KnowledgeGraphService as KG
            
            if not KG.is_connected():
                self.stdout.write(self.style.WARNING('  Neo4j not connected, skipping Knowledge Graph import'))
                return
            
            if clear:
                # Clear DDI Corpus sourced interactions (keep curated ones)
                KG.run_query("MATCH ()-[r:INTERACTS_WITH {evidence_level: 'ddi_corpus'}]-() DELETE r")
                self.stdout.write('  Cleared existing DDI Corpus interactions')
            
            # Add drugs
            drug_count = 0
            for drug_name in drugs:
                if len(drug_name) < 2:  # Skip single chars
                    continue
                    
                # Generate a simple ID
                drug_id = f"DDI_{drug_name.upper().replace(' ', '_')[:20]}"
                
                existing = KG.get_drug(name=drug_name)
                if not existing:
                    KG.add_drug(
                        drugbank_id=drug_id,
                        name=drug_name.title(),
                        category='ddi_corpus'
                    )
                    drug_count += 1
            
            self.stdout.write(f'  Knowledge Graph: Added {drug_count} new drugs')
            
            # Add interactions
            interaction_count = 0
            for drug1, drug2 in pairs:
                if len(drug1) < 2 or len(drug2) < 2:
                    continue
                
                drug1_id = f"DDI_{drug1.upper().replace(' ', '_')[:20]}"
                drug2_id = f"DDI_{drug2.upper().replace(' ', '_')[:20]}"
                
                KG.add_interaction(
                    drug1_id=drug1_id,
                    drug2_id=drug2_id,
                    severity='moderate',  # Default severity
                    mechanism=f'Interaction between {drug1.title()} and {drug2.title()}',
                    description='From DDI Corpus 2013',
                    evidence_level='ddi_corpus'
                )
                interaction_count += 1
            
            self.stdout.write(self.style.SUCCESS(f'  Knowledge Graph: Added {interaction_count} interactions'))
            
            # Show totals
            drug_total = KG.run_query('MATCH (d:Drug) RETURN count(d) as count')
            int_total = KG.run_query('MATCH ()-[r:INTERACTS_WITH]->() RETURN count(r) as count')
            self.stdout.write(f"  Total drugs now: {drug_total[0]['count'] if drug_total else 0}")
            self.stdout.write(f"  Total interactions now: {int_total[0]['count'] if int_total else 0}")
            
        except ImportError:
            self.stdout.write(self.style.WARNING('  Knowledge Graph service not available'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'  Knowledge Graph import failed: {e}'))

    def _add_entity_markers(self, sentence: str, drug1: str, drug2: str) -> str:
        """Add entity markers (<e1>, <e2>) to sentence for model input."""
        # Case-insensitive replacement
        result = sentence
        
        # Find positions of both drugs
        drug1_lower = drug1.lower()
        drug2_lower = drug2.lower()
        sent_lower = sentence.lower()
        
        pos1 = sent_lower.find(drug1_lower)
        pos2 = sent_lower.find(drug2_lower)
        
        if pos1 == -1 or pos2 == -1:
            # If drugs not found literally, return sentence as-is
            return sentence
        
        # Ensure we mark them in order of appearance
        if pos1 < pos2:
            # Mark drug1 first, then drug2
            result = (
                sentence[:pos1] + 
                f"<e1>{sentence[pos1:pos1+len(drug1)]}</e1>" +
                sentence[pos1+len(drug1):]
            )
            # Recalculate pos2 after first insertion
            offset = len('<e1></e1>')
            new_pos2 = pos2 + offset
            result = (
                result[:new_pos2] + 
                f"<e2>{result[new_pos2:new_pos2+len(drug2)]}</e2>" +
                result[new_pos2+len(drug2):]
            )
        else:
            # Mark drug2 first, then drug1
            result = (
                sentence[:pos2] + 
                f"<e2>{sentence[pos2:pos2+len(drug2)]}</e2>" +
                sentence[pos2+len(drug2):]
            )
            offset = len('<e2></e2>')
            new_pos1 = pos1 + offset
            result = (
                result[:new_pos1] + 
                f"<e1>{result[new_pos1:new_pos1+len(drug1)]}</e1>" +
                result[new_pos1+len(drug1):]
            )
        
        return result
