"""
DDI Sentence Database Service

This module provides a SQLite-based database of real DDI (Drug-Drug Interaction) 
sentences from the DDI Corpus and other sources. These sentences are used for
more accurate predictions with PubMedBERT.

The DDI Corpus contains ~19,000 annotated sentences describing drug interactions.
Using real sentences instead of templates dramatically improves prediction accuracy.
"""

import os
import re
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class DDISentence:
    """A DDI sentence from the database."""
    drug1: str
    drug2: str
    sentence: str
    interaction_type: str  # mechanism, effect, advise, int, no_interaction
    source: str  # ddi_corpus, drugbank, pubmed, manual
    confidence: float  # How reliable is this sentence (0-1)
    pmid: Optional[str] = None  # PubMed ID if available


class DDISentenceDB:
    """
    SQLite-based database of DDI sentences.
    
    Provides fast lookup of real interaction sentences for drug pairs,
    dramatically improving prediction accuracy over template-based approaches.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the DDI Sentence Database."""
        if db_path is None:
            # Default path in the data directory
            data_dir = Path(settings.BASE_DIR) / 'data'
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / 'ddi_sentences.db')
        
        self.db_path = db_path
        self._connection = None
        self._init_db()
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def _init_db(self):
        """Initialize the database schema."""
        cursor = self.connection.cursor()
        
        # Main sentences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ddi_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug1 TEXT NOT NULL,
                drug2 TEXT NOT NULL,
                drug1_normalized TEXT NOT NULL,
                drug2_normalized TEXT NOT NULL,
                sentence TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                pmid TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(drug1_normalized, drug2_normalized, sentence)
            )
        ''')
        
        # Indexes for fast lookup
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drug_pair 
            ON ddi_sentences(drug1_normalized, drug2_normalized)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drug1 
            ON ddi_sentences(drug1_normalized)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_drug2 
            ON ddi_sentences(drug2_normalized)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_interaction_type 
            ON ddi_sentences(interaction_type)
        ''')
        
        # Drug name aliases table (for synonyms, brand names)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drug_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drug_name TEXT NOT NULL,
                alias TEXT NOT NULL,
                alias_normalized TEXT NOT NULL,
                UNIQUE(drug_name, alias_normalized)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_alias 
            ON drug_aliases(alias_normalized)
        ''')
        
        self.connection.commit()
        logger.info(f"DDI Sentence DB initialized at {self.db_path}")
    
    @staticmethod
    def normalize_drug_name(name: str) -> str:
        """Normalize drug name for consistent lookup."""
        # Lowercase, remove special chars, collapse whitespace
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def add_sentence(self, drug1: str, drug2: str, sentence: str, 
                     interaction_type: str, source: str = 'manual',
                     confidence: float = 1.0, pmid: Optional[str] = None) -> bool:
        """
        Add a DDI sentence to the database.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            sentence: The sentence describing the interaction
            interaction_type: One of: mechanism, effect, advise, int, no_interaction
            source: Data source (ddi_corpus, drugbank, pubmed, manual)
            confidence: Reliability score 0-1
            pmid: Optional PubMed ID
            
        Returns:
            True if added successfully, False if duplicate
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO ddi_sentences 
                (drug1, drug2, drug1_normalized, drug2_normalized, 
                 sentence, interaction_type, source, confidence, pmid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                drug1, drug2,
                self.normalize_drug_name(drug1),
                self.normalize_drug_name(drug2),
                sentence, interaction_type, source, confidence, pmid
            ))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error adding sentence: {e}")
            return False
    
    def add_sentences_bulk(self, sentences: List[Dict]) -> int:
        """
        Add multiple sentences in bulk.
        
        Args:
            sentences: List of dicts with keys: drug1, drug2, sentence, 
                      interaction_type, source, confidence, pmid
                      
        Returns:
            Number of sentences added
        """
        cursor = self.connection.cursor()
        added = 0
        
        for s in sentences:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO ddi_sentences 
                    (drug1, drug2, drug1_normalized, drug2_normalized, 
                     sentence, interaction_type, source, confidence, pmid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    s['drug1'], s['drug2'],
                    self.normalize_drug_name(s['drug1']),
                    self.normalize_drug_name(s['drug2']),
                    s['sentence'], s['interaction_type'], 
                    s.get('source', 'bulk'), 
                    s.get('confidence', 1.0),
                    s.get('pmid')
                ))
                added += cursor.rowcount
            except Exception as e:
                logger.warning(f"Error adding sentence: {e}")
                continue
        
        self.connection.commit()
        logger.info(f"Added {added} sentences in bulk")
        return added
    
    def find_sentence(self, drug1: str, drug2: str, 
                      interaction_type: Optional[str] = None) -> Optional[DDISentence]:
        """
        Find the best sentence for a drug pair.
        
        Searches both orderings (drug1-drug2 and drug2-drug1).
        Returns the highest confidence match.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            interaction_type: Optional filter by type
            
        Returns:
            DDISentence or None if not found
        """
        d1_norm = self.normalize_drug_name(drug1)
        d2_norm = self.normalize_drug_name(drug2)
        
        cursor = self.connection.cursor()
        
        # Build query - search both orderings
        query = '''
            SELECT drug1, drug2, sentence, interaction_type, source, confidence, pmid
            FROM ddi_sentences
            WHERE (
                (drug1_normalized = ? AND drug2_normalized = ?)
                OR (drug1_normalized = ? AND drug2_normalized = ?)
            )
        '''
        params = [d1_norm, d2_norm, d2_norm, d1_norm]
        
        if interaction_type:
            query += ' AND interaction_type = ?'
            params.append(interaction_type)
        
        query += ' ORDER BY confidence DESC LIMIT 1'
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row:
            return DDISentence(
                drug1=row['drug1'],
                drug2=row['drug2'],
                sentence=row['sentence'],
                interaction_type=row['interaction_type'],
                source=row['source'],
                confidence=row['confidence'],
                pmid=row['pmid']
            )
        
        return None
    
    def find_all_sentences(self, drug1: str, drug2: str) -> List[DDISentence]:
        """Find all sentences for a drug pair."""
        d1_norm = self.normalize_drug_name(drug1)
        d2_norm = self.normalize_drug_name(drug2)
        
        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT drug1, drug2, sentence, interaction_type, source, confidence, pmid
            FROM ddi_sentences
            WHERE (
                (drug1_normalized = ? AND drug2_normalized = ?)
                OR (drug1_normalized = ? AND drug2_normalized = ?)
            )
            ORDER BY confidence DESC
        ''', (d1_norm, d2_norm, d2_norm, d1_norm))
        
        return [
            DDISentence(
                drug1=row['drug1'],
                drug2=row['drug2'],
                sentence=row['sentence'],
                interaction_type=row['interaction_type'],
                source=row['source'],
                confidence=row['confidence'],
                pmid=row['pmid']
            )
            for row in cursor.fetchall()
        ]
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.connection.cursor()
        
        cursor.execute('SELECT COUNT(*) as total FROM ddi_sentences')
        total = cursor.fetchone()['total']
        
        cursor.execute('''
            SELECT interaction_type, COUNT(*) as count 
            FROM ddi_sentences 
            GROUP BY interaction_type
        ''')
        by_type = {row['interaction_type']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT source, COUNT(*) as count 
            FROM ddi_sentences 
            GROUP BY source
        ''')
        by_source = {row['source']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT COUNT(DISTINCT drug1_normalized || drug2_normalized) as pairs
            FROM ddi_sentences
        ''')
        unique_pairs = cursor.fetchone()['pairs']
        
        return {
            'total_sentences': total,
            'unique_drug_pairs': unique_pairs,
            'by_interaction_type': by_type,
            'by_source': by_source
        }
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# Singleton instance
_ddi_sentence_db: Optional[DDISentenceDB] = None


def get_ddi_sentence_db() -> DDISentenceDB:
    """Get or create the DDI Sentence DB singleton."""
    global _ddi_sentence_db
    if _ddi_sentence_db is None:
        _ddi_sentence_db = DDISentenceDB()
    return _ddi_sentence_db
