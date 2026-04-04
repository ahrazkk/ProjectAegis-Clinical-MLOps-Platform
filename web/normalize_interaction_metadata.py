#!/usr/bin/env python
"""Normalize INTERACTS_WITH metadata into query-friendly canonical fields."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


PRECHECK_QUERY = """
MATCH ()-[r:INTERACTS_WITH]->()
RETURN
  count(r) AS total,
  sum(CASE WHEN coalesce(trim(r.severity), '') <> '' THEN 1 ELSE 0 END) AS with_severity_raw,
  sum(CASE WHEN coalesce(trim(r.evidence_level), '') <> '' THEN 1 ELSE 0 END) AS with_evidence_raw,
  sum(CASE WHEN coalesce(trim(r.severity_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_severity_normalized,
  sum(CASE WHEN coalesce(trim(r.evidence_level_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_evidence_normalized
"""

NORMALIZE_QUERY = """
MATCH ()-[r:INTERACTS_WITH]->()
WITH
  r,
  trim(coalesce(r.severity, '')) AS severity_raw,
  trim(coalesce(r.evidence_level, '')) AS evidence_raw,
  trim(coalesce(r.description, '')) AS description_trimmed,
  trim(coalesce(r.mechanism, '')) AS mechanism_trimmed
SET
  r.severity_raw = CASE
    WHEN coalesce(trim(r.severity_raw), '') = '' AND severity_raw <> '' THEN severity_raw
    ELSE r.severity_raw
  END,
  r.evidence_level_raw = CASE
    WHEN coalesce(trim(r.evidence_level_raw), '') = '' AND evidence_raw <> '' THEN evidence_raw
    ELSE r.evidence_level_raw
  END,
  r.severity_normalized = CASE
    WHEN toLower(replace(severity_raw, '-', '_')) IN ['none', 'no_interaction', 'no interaction'] THEN 'no_interaction'
    WHEN toLower(severity_raw) IN ['critical'] THEN 'critical'
    WHEN toLower(severity_raw) IN ['major', 'high', 'severe'] THEN 'severe'
    WHEN toLower(severity_raw) IN ['moderate', 'medium'] THEN 'moderate'
    WHEN toLower(severity_raw) IN ['minor', 'low'] THEN 'minor'
    WHEN severity_raw = '' THEN 'unknown'
    ELSE 'unknown'
  END,
  r.severity_bucket = CASE
    WHEN toLower(replace(severity_raw, '-', '_')) IN ['none', 'no_interaction', 'no interaction'] THEN 'low'
    WHEN toLower(severity_raw) IN ['minor', 'low'] THEN 'low'
    WHEN toLower(severity_raw) IN ['moderate', 'medium'] THEN 'medium'
    WHEN toLower(severity_raw) IN ['critical', 'major', 'high', 'severe'] THEN 'high'
    WHEN severity_raw = '' THEN 'unknown'
    ELSE 'unknown'
  END,
  r.evidence_level_normalized = CASE
    WHEN evidence_raw = '' THEN 'unknown'
    ELSE toLower(evidence_raw)
  END,
  r.description = CASE
    WHEN description_trimmed = '' THEN r.description
    ELSE description_trimmed
  END,
  r.mechanism = CASE
    WHEN mechanism_trimmed = '' THEN r.mechanism
    ELSE mechanism_trimmed
  END,
  r.metadata_version = 'interaction_metadata_norm_v1',
  r.metadata_updated_at = datetime()
RETURN count(r) AS updated
"""

POSTCHECK_QUERY = """
MATCH ()-[r:INTERACTS_WITH]->()
RETURN
  count(r) AS total,
  sum(CASE WHEN coalesce(trim(r.severity_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_severity_normalized,
  sum(CASE WHEN r.severity_normalized = 'unknown' THEN 1 ELSE 0 END) AS severity_unknown,
  sum(CASE WHEN coalesce(trim(r.evidence_level_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_evidence_normalized,
  sum(CASE WHEN r.evidence_level_normalized = 'unknown' THEN 1 ELSE 0 END) AS evidence_unknown
"""


def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize INTERACTS_WITH metadata fields in Neo4j.')
    parser.add_argument('--apply', action='store_true', help='Write normalized metadata fields.')
    args = parser.parse_args()

    uri = settings.NEO4J_CONFIG.get('uri')
    user = settings.NEO4J_CONFIG.get('user')
    password = settings.NEO4J_CONFIG.get('password')

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    with driver.session() as session:
        pre = session.run(PRECHECK_QUERY).single().data()

        print('=== Interaction Metadata Normalization ===')
        print('Pre-check:')
        print(pre)

        if args.apply:
            result = session.run(NORMALIZE_QUERY).single().data()
            print('Update result:')
            print(result)
            post = session.run(POSTCHECK_QUERY).single().data()
            print('Post-check:')
            print(post)
        else:
            print('Dry-run only. Re-run with --apply to persist normalized fields.')

    driver.close()


if __name__ == '__main__':
    main()
