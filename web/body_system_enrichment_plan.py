#!/usr/bin/env python
"""Plan and optionally apply Drug -> BodySystem enrichment in Neo4j."""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


CANONICAL_SYSTEMS = [
    'Cardiovascular',
    'Central Nervous System',
    'Gastrointestinal',
    'Endocrine',
    'Respiratory',
    'Renal',
    'Hepatic',
]

EXACT_CATEGORY_MAP = {
    'cardiovascular': ('Cardiovascular', 3.0),
    'antiarrhythmics': ('Cardiovascular', 2.6),
    'antihypertensives': ('Cardiovascular', 2.6),
    'antilipemic agents': ('Cardiovascular', 2.6),
    'hematology': ('Cardiovascular', 2.2),
    'pain/cardiovascular': ('Cardiovascular', 2.4),
    'diabetes': ('Endocrine', 3.0),
    'endocrine': ('Endocrine', 3.0),
    'thyroid supplements': ('Endocrine', 2.8),
    'gastrointestinal': ('Gastrointestinal', 3.0),
    'antidiarrheal agents': ('Gastrointestinal', 2.6),
    'calcium containing antacids': ('Gastrointestinal', 2.8),
    'hyperosmotic laxatives': ('Gastrointestinal', 2.8),
    'respiratory': ('Respiratory', 3.0),
    'urology': ('Renal', 2.8),
    'cns': ('Central Nervous System', 3.0),
    'psychiatry': ('Central Nervous System', 3.0),
    'neurology': ('Central Nervous System', 3.0),
    'anticonvulsants': ('Central Nervous System', 3.0),
    'antiparkinson agents': ('Central Nervous System', 3.0),
    'antimigraine agents': ('Central Nervous System', 3.0),
    'anesthesia': ('Central Nervous System', 2.2),
}

THERAPEUTIC_PHRASE_MAP = {
    'Cardiovascular': {
        'anticoagulant': 2.8,
        'antiplatelet': 2.8,
        'decreased platelet aggregation': 3.1,
        'decreased coagulation factor activity': 3.1,
        'beta blocker': 3.0,
        'ace inhibitor': 3.0,
        'arb': 3.0,
        'calcium channel blocker': 3.0,
        'statin': 3.0,
        'antiarrhythmic': 3.0,
        'vasodil': 2.2,
    },
    'Central Nervous System': {
        'benzodiazepine': 3.0,
        'antidepressant': 2.8,
        'antipsychotic': 2.8,
        'anticonvulsant': 3.0,
        'decreased central nervous system disorganized electrical activity': 3.3,
        'decreased central nervous system organized electrical activity': 3.3,
        'cerebral arterial vasoconstriction': 3.1,
        'anxiolytic': 2.8,
        'sedative': 2.6,
        'hypnotic': 2.6,
        'analgesic': 2.1,
        'opioid': 2.5,
    },
    'Gastrointestinal': {
        'proton pump inhibitor': 3.0,
        'h2 blocker': 3.0,
        'antiemetic': 2.8,
        'prokinetic': 2.5,
        'laxative': 2.6,
    },
    'Endocrine': {
        'corticosteroid': 2.8,
        'insulin': 3.0,
        'biguanide': 3.0,
        'sulfonylurea': 3.0,
        'thyroid': 3.0,
        'sglt': 2.8,
        'dpp': 2.8,
        'glp': 2.8,
    },
    'Respiratory': {
        'bronchodilation': 3.0,
        'beta agonist': 3.0,
        'leukotriene': 2.8,
        'inhaled': 2.6,
        'antihistamine': 2.2,
    },
    'Renal': {
        'diuretic': 3.0,
        'urolog': 2.8,
        'renal': 2.8,
        'nephro': 2.4,
    },
    'Hepatic': {
        'hepatic': 2.8,
        'liver': 2.8,
        'hepat': 2.6,
    },
}

STEROID_NAME_HINTS = [
    'predni', 'predniso', 'dexameth', 'methylpred', 'hydrocort',
    'fludrocort', 'cortis', 'cortic', 'betameth', 'triamcin',
]

SYSTEM_KEYWORDS = {
    'Cardiovascular': {
        'category': [
            'cardio', 'arrhythm', 'hypertens', 'lipidem', 'statin',
            'anticoagul', 'antiplatelet', 'beta blocker', 'ace inhibitor',
            'arb', 'antiarrhythmic',
        ],
        'therapeutic': [
            'cardio', 'arrhythm', 'anticoagul', 'antiplatelet', 'statin',
            'beta blocker', 'ace inhibitor', 'arb', 'vasodil',
        ],
        'name': ['pril', 'sartan', 'olol', 'dipine', 'statin'],
    },
    'Central Nervous System': {
        'category': [
            'cns', 'psych', 'neuro', 'pain', 'anesthesia', 'opioid',
        ],
        'therapeutic': [
            'benzodiazep', 'antidepress', 'antipsych', 'anxiolytic', 'sedative',
            'hypnotic', 'anticonvuls', 'triptan', 'analgesic', 'opioid', 'cns', 'neuro',
        ],
        'name': ['pam', 'lam', 'zepam', 'triptan', 'codone', 'morphine', 'gabap', 'pregabal'],
    },
    'Gastrointestinal': {
        'category': ['gastro', 'antiemetic'],
        'therapeutic': ['proton pump inhibitor', 'h2 blocker', 'antiemetic', 'gastro', 'prokinetic'],
        'name': ['prazole', 'tidine', 'setron'],
    },
    'Endocrine': {
        'category': ['diabetes', 'endocrine', 'thyroid'],
        'therapeutic': ['insulin', 'biguanide', 'sulfonylurea', 'thyroid', 'sglt', 'dpp', 'glp'],
        'name': ['gliflozin', 'gliptin', 'glitazone', 'metformin', 'levothyroxine'],
    },
    'Respiratory': {
        'category': ['respir'],
        'therapeutic': ['bronchodil', 'beta agonist', 'leukotriene', 'inhaled'],
        'name': ['terol', 'lukast', 'tropium'],
    },
    'Renal': {
        'category': ['urology', 'renal', 'urinary'],
        'therapeutic': ['diuretic', 'urolog', 'renal', 'nephro'],
        'name': ['steride', 'osin'],
    },
    'Hepatic': {
        'category': ['hepatic', 'liver'],
        'therapeutic': ['hepatic', 'liver', 'hepat'],
        'name': ['azole'],
    },
}


@dataclass
class Suggestion:
    node_id: str
    drug_name: str
    drugbank_id: str
    interaction_degree: int
    category: str
    therapeutic_class: str
    suggested_system: str
    confidence_score: float
    confidence_level: str
    rationale: str


def confidence_level(score: float) -> str:
    if score >= 3.0:
        return 'high'
    if score >= 2.0:
        return 'medium'
    if score >= 1.0:
        return 'low'
    return 'none'


def normalize(text: Optional[str]) -> str:
    return (text or '').strip().lower()


def score_drug(category: str, therapeutic_class: str, drug_name: str) -> Tuple[Optional[str], float, List[str]]:
    category_n = normalize(category)
    therapeutic_n = normalize(therapeutic_class)
    name_n = normalize(drug_name)

    reasons: List[str] = []
    system_scores = defaultdict(float)

    steroid_signal = (
        any(token in name_n for token in STEROID_NAME_HINTS)
        or 'steroid' in category_n
        or 'glucocorticoid' in category_n
        or 'glucocorticoid' in therapeutic_n
    )

    # Exact category mapping contributes strong prior evidence.
    if category_n in EXACT_CATEGORY_MAP:
        system, score = EXACT_CATEGORY_MAP[category_n]
        system_scores[system] += score
        reasons.append(f'exact_category:{category_n}')

    # Therapeutic phrase mapping captures high-signal class semantics.
    for system, phrase_scores in THERAPEUTIC_PHRASE_MAP.items():
        for phrase, weight in phrase_scores.items():
            if phrase in therapeutic_n:
                adjusted_weight = weight

                # Guardrail: some rows are mislabeled as corticosteroid.
                if phrase == 'corticosteroid' and not steroid_signal:
                    adjusted_weight = 0.4
                    reasons.append('corticosteroid_signal_downgraded')

                system_scores[system] += adjusted_weight
                reasons.append(f'therapeutic_phrase:{system}:{phrase}')

    for system, bucket in SYSTEM_KEYWORDS.items():
        for token in bucket['category']:
            if token in category_n:
                system_scores[system] += 1.6
                reasons.append(f'category_keyword:{system}:{token}')

        for token in bucket['therapeutic']:
            if token in therapeutic_n:
                system_scores[system] += 1.2
                reasons.append(f'therapeutic_keyword:{system}:{token}')

        for token in bucket['name']:
            if token in name_n:
                system_scores[system] += 0.6
                reasons.append(f'name_keyword:{system}:{token}')

    if not system_scores:
        return None, 0.0, []

    system, score = max(system_scores.items(), key=lambda kv: kv[1])
    return system, score, reasons


def fetch_missing_system_drugs(driver) -> List[Dict]:
    query = """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[r:INTERACTS_WITH]-(:Drug)
        WITH d, count(r) AS interaction_degree
        WHERE NOT EXISTS { MATCH (d)-[:AFFECTS_SYSTEM]-(:BodySystem) }
        RETURN
            elementId(d) AS node_id,
            d.name AS drug_name,
            d.drugbank_id AS drugbank_id,
            interaction_degree,
            coalesce(d.category, '') AS category,
            coalesce(d.therapeutic_class, '') AS therapeutic_class
    """
    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def ensure_system_nodes(driver) -> None:
    with driver.session() as session:
        for name in CANONICAL_SYSTEMS:
            session.run(
                """
                MERGE (s:BodySystem {name: $name})
                """,
                {'name': name},
            )


def apply_suggestions(driver, suggestions: List[Suggestion], min_confidence: str) -> int:
    allowed = {'high', 'medium', 'low'}
    if min_confidence not in allowed:
        raise ValueError(f'Unsupported min confidence: {min_confidence}')

    threshold_rank = {'low': 1, 'medium': 2, 'high': 3}
    applied = 0

    with driver.session() as session:
        for item in suggestions:
            if threshold_rank[item.confidence_level] < threshold_rank[min_confidence]:
                continue

            session.run(
                """
                MATCH (d)
                WHERE elementId(d) = $node_id
                MERGE (s:BodySystem {name: $system_name})
                MERGE (d)-[r:AFFECTS_SYSTEM]->(s)
                ON CREATE SET
                    r.source = 'heuristic_enrichment_v1',
                    r.confidence = $confidence,
                    r.score = $score,
                    r.created_at = datetime(),
                    r.rationale = $rationale
                ON MATCH SET
                    r.last_reviewed_at = datetime()
                """,
                {
                    'node_id': item.node_id,
                    'system_name': item.suggested_system,
                    'confidence': item.confidence_level,
                    'score': float(item.confidence_score),
                    'rationale': item.rationale,
                },
            )
            applied += 1

    return applied


def write_csv(path: str, suggestions: List[Suggestion]) -> None:
    fieldnames = [
        'node_id',
        'drug_name',
        'drugbank_id',
        'interaction_degree',
        'category',
        'therapeutic_class',
        'suggested_system',
        'confidence_score',
        'confidence_level',
        'rationale',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in suggestions:
            writer.writerow({
                'node_id': row.node_id,
                'drug_name': row.drug_name,
                'drugbank_id': row.drugbank_id,
                'interaction_degree': row.interaction_degree,
                'category': row.category,
                'therapeutic_class': row.therapeutic_class,
                'suggested_system': row.suggested_system,
                'confidence_score': f'{row.confidence_score:.2f}',
                'confidence_level': row.confidence_level,
                'rationale': row.rationale,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description='Plan/apply BodySystem enrichment for drugs missing AFFECTS_SYSTEM.')
    parser.add_argument('--apply', action='store_true', help='Apply suggested relationships to Neo4j (default: dry-run only).')
    parser.add_argument('--min-confidence', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--output-csv', default='')
    parser.add_argument('--top-unmapped', type=int, default=25)
    args = parser.parse_args()

    uri = settings.NEO4J_CONFIG.get('uri')
    user = settings.NEO4J_CONFIG.get('user')
    password = settings.NEO4J_CONFIG.get('password')

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    rows = fetch_missing_system_drugs(driver)

    suggestions: List[Suggestion] = []
    unmapped = []

    for row in rows:
        system, score, reasons = score_drug(
            category=row.get('category', ''),
            therapeutic_class=row.get('therapeutic_class', ''),
            drug_name=row.get('drug_name', ''),
        )

        if not system or score < 1.0:
            unmapped.append(row)
            continue

        suggestions.append(
            Suggestion(
                node_id=row.get('node_id', ''),
                drug_name=row.get('drug_name', ''),
                drugbank_id=row.get('drugbank_id', ''),
                interaction_degree=int(row.get('interaction_degree') or 0),
                category=row.get('category', ''),
                therapeutic_class=row.get('therapeutic_class', ''),
                suggested_system=system,
                confidence_score=score,
                confidence_level=confidence_level(score),
                rationale=';'.join(reasons[:6]),
            )
        )

    by_conf = Counter(item.confidence_level for item in suggestions)
    by_system = Counter(item.suggested_system for item in suggestions)

    print('=== Body System Enrichment Plan ===')
    print(f"Missing-system drugs scanned: {len(rows)}")
    print(f"Suggested mappings (score >= 1.0): {len(suggestions)}")
    print(f"Unmapped after heuristics: {len(unmapped)}")
    print('')
    print('By confidence:')
    for level in ['high', 'medium', 'low']:
        print(f"- {level}: {by_conf.get(level, 0)}")

    print('')
    print('By suggested system:')
    for system, count in by_system.most_common():
        print(f"- {system}: {count}")

    if unmapped:
        cat_counts = Counter((row.get('category') or '<missing>') for row in unmapped)
        tc_counts = Counter((row.get('therapeutic_class') or '<missing>') for row in unmapped)
        high_degree_unmapped = sorted(
            unmapped,
            key=lambda row: int(row.get('interaction_degree') or 0),
            reverse=True,
        )
        print('')
        print('Top unmapped categories:')
        for category, count in cat_counts.most_common(args.top_unmapped):
            print(f"- {category}: {count}")
        print('')
        print('Top unmapped therapeutic_class values:')
        for tc, count in tc_counts.most_common(args.top_unmapped):
            print(f"- {tc}: {count}")
        print('')
        print('Top unmapped high-degree drugs:')
        for row in high_degree_unmapped[:args.top_unmapped]:
            print(
                f"- {row.get('drug_name')} ({row.get('drugbank_id')}) | "
                f"deg={int(row.get('interaction_degree') or 0)} | "
                f"cat={row.get('category') or '<missing>'} | "
                f"tc={row.get('therapeutic_class') or '<missing>'}"
            )

    if args.output_csv:
        write_csv(args.output_csv, suggestions)
        print('')
        print(f"Wrote suggestions CSV: {args.output_csv}")

    if args.apply:
        ensure_system_nodes(driver)
        applied = apply_suggestions(driver, suggestions, args.min_confidence)
        print('')
        print(
            f"Applied AFFECTS_SYSTEM edges: {applied} "
            f"(min_confidence={args.min_confidence})"
        )
    else:
        print('')
        print('Dry-run complete. Re-run with --apply to write relationships.')

    driver.close()


if __name__ == '__main__':
    main()
