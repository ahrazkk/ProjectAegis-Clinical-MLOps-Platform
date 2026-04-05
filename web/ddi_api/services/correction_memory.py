"""
Correction Memory Service

Manages `:Correction` nodes in Neo4j that record user-submitted corrections
to GNN predictions.  Each correction links two drugs, captures the original
GNN prediction, the user's assessed severity, evidence, and an approval status.

Node schema:
  (:Correction {
      id: str (UUID),
      drug_a: str,
      drug_b: str,
      gnn_severity: str,
      gnn_risk_score: float,
      gnn_confidence: float,
      corrected_severity: str,       # user's assessment
      evidence_text: str,            # free-text evidence / rationale
      evidence_source: str,          # e.g. "PMID:12345", "Clinical", "Literature"
      status: str,                   # pending | approved | rejected
      created_at: str,               # ISO timestamp
      reviewed_at: str | null,       # ISO timestamp of last review
  })

Relationships:
  (:Drug)-[:HAS_CORRECTION]->(:Correction)<-[:HAS_CORRECTION]-(:Drug)
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)


class CorrectionMemory:
    """CRUD operations for Correction nodes in Neo4j."""

    VALID_STATUSES = ('pending', 'approved', 'rejected')
    VALID_SEVERITIES = ('none', 'minor', 'moderate', 'severe', 'critical')

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    def create_correction(
        self,
        drug_a: str,
        drug_b: str,
        gnn_severity: str,
        gnn_risk_score: float,
        gnn_confidence: float,
        corrected_severity: str,
        evidence_text: str = '',
        evidence_source: str = '',
    ) -> Optional[Dict]:
        """
        Store a new correction linking two drugs.

        Drugs are normalized to title-case and sorted alphabetically so
        (Aspirin, Warfarin) == (Warfarin, Aspirin).
        """
        drug_a, drug_b = sorted([drug_a.strip().title(), drug_b.strip().title()])

        # Deduplicate: skip if a pending correction already exists for this pair
        existing = KnowledgeGraphService.run_query(
            "MATCH (c:Correction {drug_a: $drug_a, drug_b: $drug_b, status: 'pending'}) RETURN c.id AS id LIMIT 1",
            {'drug_a': drug_a, 'drug_b': drug_b},
        )
        if existing:
            logger.info("Pending correction already exists for %s <-> %s, skipping", drug_a, drug_b)
            return existing[0].get('id')  # return existing id so caller knows it wasn't lost

        correction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (da:Drug {name: $drug_a})
        MERGE (db:Drug {name: $drug_b})
        CREATE (c:Correction {
            id: $id,
            drug_a: $drug_a,
            drug_b: $drug_b,
            gnn_severity: $gnn_severity,
            gnn_risk_score: $gnn_risk_score,
            gnn_confidence: $gnn_confidence,
            corrected_severity: $corrected_severity,
            evidence_text: $evidence_text,
            evidence_source: $evidence_source,
            status: 'pending',
            created_at: $created_at,
            reviewed_at: null
        })
        MERGE (da)-[:HAS_CORRECTION]->(c)
        MERGE (db)-[:HAS_CORRECTION]->(c)
        RETURN c {.*} AS correction
        """
        results = KnowledgeGraphService.run_query(query, {
            'id': correction_id,
            'drug_a': drug_a,
            'drug_b': drug_b,
            'gnn_severity': gnn_severity,
            'gnn_risk_score': gnn_risk_score,
            'gnn_confidence': gnn_confidence,
            'corrected_severity': corrected_severity,
            'evidence_text': evidence_text,
            'evidence_source': evidence_source,
            'created_at': now,
        })
        if results:
            logger.info("Created correction %s for %s <-> %s", correction_id, drug_a, drug_b)
            return results[0].get('correction')
        logger.error("Failed to create correction for %s <-> %s", drug_a, drug_b)
        return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_corrections(
        self,
        status: Optional[str] = None,
        drug_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """List corrections, optionally filtered by status or drug."""
        where_clauses = []
        params: Dict = {'limit': limit}

        if status and status in self.VALID_STATUSES:
            where_clauses.append('c.status = $status')
            params['status'] = status
        if drug_name:
            where_clauses.append('(c.drug_a = $drug_name OR c.drug_b = $drug_name)')
            params['drug_name'] = drug_name.strip().title()

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ''

        query = f"""
        MATCH (c:Correction)
        {where}
        RETURN c {{.*}} AS correction
        ORDER BY c.created_at DESC
        LIMIT $limit
        """
        results = KnowledgeGraphService.run_query(query, params)
        return [r['correction'] for r in results] if results else []

    def get_correction_by_id(self, correction_id: str) -> Optional[Dict]:
        """Fetch a single correction by ID."""
        query = """
        MATCH (c:Correction {id: $id})
        RETURN c {.*} AS correction
        """
        results = KnowledgeGraphService.run_query(query, {'id': correction_id})
        return results[0]['correction'] if results else None

    def get_approved_correction(self, drug_a: str, drug_b: str) -> Optional[Dict]:
        """
        Get the most recent approved correction for a drug pair.
        Used by the chatbot to overlay corrections on GNN predictions.
        """
        a, b = sorted([drug_a.strip().title(), drug_b.strip().title()])
        query = """
        MATCH (c:Correction {drug_a: $drug_a, drug_b: $drug_b, status: 'approved'})
        RETURN c {.*} AS correction
        ORDER BY c.reviewed_at DESC
        LIMIT 1
        """
        results = KnowledgeGraphService.run_query(query, {'drug_a': a, 'drug_b': b})
        return results[0]['correction'] if results else None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def review_correction(
        self,
        correction_id: str,
        new_status: str,
        corrected_severity: str = '',
        evidence_text: str = '',
        evidence_source: str = '',
    ) -> Optional[Dict]:
        """Approve or reject a correction, optionally updating fields."""
        if new_status not in ('approved', 'rejected'):
            return None

        now = datetime.now(timezone.utc).isoformat()

        # Build dynamic SET clause for optional field updates
        set_parts = ['c.status = $status', 'c.reviewed_at = $reviewed_at']
        params: Dict = {'id': correction_id, 'status': new_status, 'reviewed_at': now}

        if corrected_severity:
            set_parts.append('c.corrected_severity = $corrected_severity')
            params['corrected_severity'] = corrected_severity
        if evidence_text:
            set_parts.append('c.evidence_text = $evidence_text')
            params['evidence_text'] = evidence_text
        if evidence_source:
            set_parts.append('c.evidence_source = $evidence_source')
            params['evidence_source'] = evidence_source

        query = f"""
        MATCH (c:Correction {{id: $id}})
        SET {', '.join(set_parts)}
        RETURN c {{.*}} AS correction
        """
        results = KnowledgeGraphService.run_query(query, params)
        if results:
            logger.info("Correction %s set to %s", correction_id, new_status)
            return results[0]['correction']
        return None

    def count_by_status(self) -> Dict[str, int]:
        """Return counts of corrections grouped by status."""
        query = """
        MATCH (c:Correction)
        RETURN c.status AS status, count(c) AS count
        """
        results = KnowledgeGraphService.run_query(query)
        counts = {'pending': 0, 'approved': 0, 'rejected': 0}
        for r in (results or []):
            s = r.get('status', '')
            if s in counts:
                counts[s] = r.get('count', 0)
        return counts

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_correction(self, correction_id: str) -> bool:
        """Remove a correction and its relationships."""
        query = """
        MATCH (c:Correction {id: $id})
        DETACH DELETE c
        RETURN count(c) AS deleted
        """
        results = KnowledgeGraphService.run_query(query, {'id': correction_id})
        deleted = results[0].get('deleted', 0) if results else 0
        return deleted > 0

    # ------------------------------------------------------------------
    # Export for GNN retraining
    # ------------------------------------------------------------------
    def export_training_data(self) -> List[Dict]:
        """
        Export all approved corrections as training data for GNN retraining.

        Returns a list of dicts with fields suitable for training:
        drug_a, drug_b, original_severity, original_risk_score, original_confidence,
        corrected_severity, evidence_text, evidence_source, reviewed_at
        """
        query = """
        MATCH (c:Correction {status: 'approved'})
        WHERE c.corrected_severity IS NOT NULL AND c.corrected_severity <> ''
        RETURN c.drug_a AS drug_a,
               c.drug_b AS drug_b,
               c.gnn_severity AS original_severity,
               c.gnn_risk_score AS original_risk_score,
               c.gnn_confidence AS original_confidence,
               c.corrected_severity AS corrected_severity,
               c.evidence_text AS evidence_text,
               c.evidence_source AS evidence_source,
               c.reviewed_at AS reviewed_at
        ORDER BY c.reviewed_at DESC
        """
        results = KnowledgeGraphService.run_query(query)
        return results or []


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: Optional[CorrectionMemory] = None


def get_correction_memory() -> CorrectionMemory:
    global _instance
    if _instance is None:
        _instance = CorrectionMemory()
    return _instance
