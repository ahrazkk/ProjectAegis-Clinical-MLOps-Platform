#!/usr/bin/env python
"""Neo4j graph data quality audit for Drug/Interaction/System coverage."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


QUERIES = {
    'node_labels': """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
    """,
    'relationship_types': """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
    """,
    'drug_core': """
        MATCH (d:Drug)
        RETURN
            count(d) AS drugs,
            sum(CASE WHEN coalesce(trim(d.smiles), '') <> '' THEN 1 ELSE 0 END) AS with_smiles,
            sum(CASE WHEN coalesce(trim(d.category), '') <> '' THEN 1 ELSE 0 END) AS with_category,
            sum(CASE WHEN coalesce(trim(d.therapeutic_class), '') <> '' THEN 1 ELSE 0 END) AS with_therapeutic_class
    """,
    'drug_interaction_coverage': """
        MATCH (d:Drug)
        RETURN
            count(d) AS drugs,
            sum(CASE WHEN EXISTS { MATCH (d)-[:INTERACTS_WITH]-(:Drug) } THEN 1 ELSE 0 END) AS drugs_with_interactions
    """,
    'drug_system_coverage_any_direction': """
        MATCH (d:Drug)
        RETURN
            count(d) AS drugs,
            sum(CASE WHEN EXISTS { MATCH (d)-[:AFFECTS_SYSTEM]-(:BodySystem) } THEN 1 ELSE 0 END) AS drugs_with_system
    """,
    'drug_system_coverage_directional': """
        MATCH (d:Drug)
        RETURN
            count(d) AS drugs,
            sum(CASE WHEN EXISTS { MATCH (d)-[:AFFECTS_SYSTEM]->(:BodySystem) } THEN 1 ELSE 0 END) AS drugs_with_system_outgoing,
            sum(CASE WHEN EXISTS { MATCH (:BodySystem)-[:AFFECTS_SYSTEM]->(d) } THEN 1 ELSE 0 END) AS drugs_with_system_incoming
    """,
    'body_system_nodes': """
        MATCH (s:BodySystem)
        RETURN count(s) AS body_systems
    """,
    'affects_system_relationships': """
        MATCH ()-[r:AFFECTS_SYSTEM]->()
        RETURN count(r) AS affects_system_relationships
    """,
    'affects_system_direction_breakdown': """
        MATCH (a)-[r:AFFECTS_SYSTEM]->(b)
        RETURN labels(a) AS from_labels, labels(b) AS to_labels, count(r) AS count
        ORDER BY count DESC
    """,
    'systems_per_drug_distribution': """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[:AFFECTS_SYSTEM]-(:BodySystem)
        WITH d, count(*) AS system_count
        RETURN
            avg(system_count) AS avg_systems_per_drug,
            min(system_count) AS min_systems,
            max(system_count) AS max_systems,
            percentileDisc(system_count, 0.5) AS p50,
            percentileDisc(system_count, 0.9) AS p90
    """,
    'interacts_with_quality': """
        MATCH ()-[r:INTERACTS_WITH]->()
        RETURN
            count(r) AS relationships,
            sum(CASE WHEN coalesce(trim(r.severity), '') <> '' THEN 1 ELSE 0 END) AS with_severity,
            sum(CASE WHEN coalesce(trim(r.severity_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_severity_normalized,
            sum(CASE WHEN r.severity_normalized = 'unknown' THEN 1 ELSE 0 END) AS severity_normalized_unknown,
            sum(CASE WHEN coalesce(trim(r.description), '') <> '' THEN 1 ELSE 0 END) AS with_description,
            sum(CASE WHEN coalesce(trim(r.mechanism), '') <> '' THEN 1 ELSE 0 END) AS with_mechanism,
            sum(CASE WHEN coalesce(trim(r.evidence_level), '') <> '' THEN 1 ELSE 0 END) AS with_evidence,
            sum(CASE WHEN coalesce(trim(r.evidence_level_normalized), '') <> '' THEN 1 ELSE 0 END) AS with_evidence_normalized,
            sum(CASE WHEN r.evidence_level_normalized = 'unknown' THEN 1 ELSE 0 END) AS evidence_normalized_unknown
    """,
    'interacts_with_severity_distribution': """
        MATCH ()-[r:INTERACTS_WITH]->()
        RETURN coalesce(r.severity, '<null>') AS severity, count(*) AS count
        ORDER BY count DESC
    """,
    'interacts_with_evidence_distribution': """
        MATCH ()-[r:INTERACTS_WITH]->()
        RETURN coalesce(r.evidence_level, '<null>') AS evidence_level, count(*) AS count
        ORDER BY count DESC
    """,
    'interaction_pair_symmetry': """
        MATCH (a:Drug)-[:INTERACTS_WITH]->(b:Drug)
        WHERE elementId(a) < elementId(b)
        WITH a, b, size([(b)-[:INTERACTS_WITH]->(a) | 1]) AS reverse_edges
        RETURN
            count(*) AS unique_undirected_pairs,
            sum(CASE WHEN reverse_edges > 0 THEN 1 ELSE 0 END) AS reciprocal_pairs
    """,
    'top_high_degree_drugs_missing_system': """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[r:INTERACTS_WITH]-(:Drug)
        WITH d, count(r) AS interaction_degree
        WHERE interaction_degree > 0
          AND NOT EXISTS { MATCH (d)-[:AFFECTS_SYSTEM]-(:BodySystem) }
        RETURN d.name AS drug_name, d.drugbank_id AS drugbank_id, interaction_degree
        ORDER BY interaction_degree DESC
        LIMIT 50
    """,
    'missing_system_by_category': """
        MATCH (d:Drug)
        WHERE NOT EXISTS { MATCH (d)-[:AFFECTS_SYSTEM]-(:BodySystem) }
        RETURN coalesce(d.category, '<missing>') AS category, count(*) AS count
        ORDER BY count DESC
        LIMIT 30
    """,
}


def run_query(driver, query):
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def as_pct(part, whole):
    if not whole:
        return 0.0
    return 100.0 * float(part) / float(whole)


def main():
    parser = argparse.ArgumentParser(description='Audit Neo4j graph data quality for DDI project.')
    parser.add_argument('--output-json', default='')
    args = parser.parse_args()

    uri = settings.NEO4J_CONFIG.get('uri')
    user = settings.NEO4J_CONFIG.get('user')
    password = settings.NEO4J_CONFIG.get('password')

    if not uri or not user or not password:
        raise RuntimeError('Missing Neo4j settings. Check ProjectAegis settings/.env loading.')

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    results = {}
    for name, query in QUERIES.items():
        results[name] = run_query(driver, query)

    driver.close()

    drug_core = (results['drug_core'][0] if results['drug_core'] else {})
    interaction_cov = (results['drug_interaction_coverage'][0] if results['drug_interaction_coverage'] else {})
    system_cov_any = (results['drug_system_coverage_any_direction'][0] if results['drug_system_coverage_any_direction'] else {})
    system_cov_dir = (results['drug_system_coverage_directional'][0] if results['drug_system_coverage_directional'] else {})
    quality = (results['interacts_with_quality'][0] if results['interacts_with_quality'] else {})
    symmetry = (results['interaction_pair_symmetry'][0] if results['interaction_pair_symmetry'] else {})

    total_drugs = int(drug_core.get('drugs') or 0)
    with_system = int(system_cov_any.get('drugs_with_system') or 0)
    without_system = max(0, total_drugs - with_system)

    print('=== Graph Data Quality Audit ===')
    print(f"Total drugs: {total_drugs}")
    print(f"Drugs with SMILES: {drug_core.get('with_smiles', 0)} ({as_pct(drug_core.get('with_smiles', 0), total_drugs):.2f}%)")
    print(f"Drugs with category: {drug_core.get('with_category', 0)} ({as_pct(drug_core.get('with_category', 0), total_drugs):.2f}%)")
    print(f"Drugs with therapeutic_class: {drug_core.get('with_therapeutic_class', 0)} ({as_pct(drug_core.get('with_therapeutic_class', 0), total_drugs):.2f}%)")

    print('\n--- Interaction Coverage ---')
    print(
        f"Drugs with at least one INTERACTS_WITH: {interaction_cov.get('drugs_with_interactions', 0)} "
        f"({as_pct(interaction_cov.get('drugs_with_interactions', 0), total_drugs):.2f}%)"
    )
    rels = int(quality.get('relationships') or 0)
    print(f"INTERACTS_WITH relationships: {rels}")
    print(f"With severity: {quality.get('with_severity', 0)} ({as_pct(quality.get('with_severity', 0), rels):.2f}%)")
    print(
        f"With severity_normalized: {quality.get('with_severity_normalized', 0)} "
        f"({as_pct(quality.get('with_severity_normalized', 0), rels):.2f}%)"
    )
    print(
        f"severity_normalized='unknown': {quality.get('severity_normalized_unknown', 0)} "
        f"({as_pct(quality.get('severity_normalized_unknown', 0), rels):.2f}%)"
    )
    print(f"With description: {quality.get('with_description', 0)} ({as_pct(quality.get('with_description', 0), rels):.2f}%)")
    print(f"With mechanism: {quality.get('with_mechanism', 0)} ({as_pct(quality.get('with_mechanism', 0), rels):.2f}%)")
    print(f"With evidence_level: {quality.get('with_evidence', 0)} ({as_pct(quality.get('with_evidence', 0), rels):.2f}%)")
    print(
        f"With evidence_level_normalized: {quality.get('with_evidence_normalized', 0)} "
        f"({as_pct(quality.get('with_evidence_normalized', 0), rels):.2f}%)"
    )
    print(
        f"evidence_level_normalized='unknown': {quality.get('evidence_normalized_unknown', 0)} "
        f"({as_pct(quality.get('evidence_normalized_unknown', 0), rels):.2f}%)"
    )

    unique_pairs = int(symmetry.get('unique_undirected_pairs') or 0)
    reciprocal_pairs = int(symmetry.get('reciprocal_pairs') or 0)
    print(f"Reciprocal pair coverage: {reciprocal_pairs}/{unique_pairs} ({as_pct(reciprocal_pairs, unique_pairs):.2f}%)")

    print('\n--- Body System Coverage ---')
    print(f"Drugs linked to BodySystem (any direction): {with_system} ({as_pct(with_system, total_drugs):.2f}%)")
    print(f"Drugs missing BodySystem links: {without_system} ({as_pct(without_system, total_drugs):.2f}%)")
    print(f"Drugs with outgoing AFFECTS_SYSTEM: {system_cov_dir.get('drugs_with_system_outgoing', 0)}")
    print(f"Drugs with incoming AFFECTS_SYSTEM: {system_cov_dir.get('drugs_with_system_incoming', 0)}")

    if results.get('body_system_nodes'):
        print(f"BodySystem nodes: {results['body_system_nodes'][0].get('body_systems', 0)}")
    if results.get('affects_system_relationships'):
        print(f"AFFECTS_SYSTEM relationships: {results['affects_system_relationships'][0].get('affects_system_relationships', 0)}")

    print('\nTop missing-system high-degree drugs (first 10):')
    for row in results['top_high_degree_drugs_missing_system'][:10]:
        print(
            f"- {row.get('drug_name')} ({row.get('drugbank_id')}): "
            f"interaction_degree={row.get('interaction_degree')}"
        )

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote JSON report: {args.output_json}")


if __name__ == '__main__':
    main()
