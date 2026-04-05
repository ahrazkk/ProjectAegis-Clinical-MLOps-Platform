"""
Drug Class Service for Project Aegis
Groups drugs by therapeutic class and identifies class-level interaction patterns.
"""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DrugClassService:
    """Provides therapeutic class grouping and class-level interaction analysis."""

    @staticmethod
    def get_class_for_drug(drug_name: str) -> Optional[str]:
        """Get the therapeutic class for a single drug from Neo4j."""
        from .knowledge_graph import KnowledgeGraphService
        results = KnowledgeGraphService.run_query(
            "MATCH (d:Drug) WHERE toLower(d.name) = $name "
            "RETURN d.category AS category, d.therapeutic_class AS therapeutic_class",
            {'name': drug_name.strip().lower()}
        )
        if results:
            return results[0].get('category') or results[0].get('therapeutic_class') or 'Unknown'
        return 'Unknown'

    @staticmethod
    def group_by_class(drug_names: List[str]) -> Dict[str, List[str]]:
        """Group a list of drugs by their therapeutic class."""
        from .knowledge_graph import KnowledgeGraphService
        groups = defaultdict(list)

        # Batch query for efficiency
        results = KnowledgeGraphService.run_query(
            "UNWIND $names AS drug_name "
            "MATCH (d:Drug) WHERE toLower(d.name) = toLower(drug_name) "
            "RETURN d.name AS name, "
            "COALESCE(d.category, d.therapeutic_class, 'Unknown') AS drug_class",
            {'names': drug_names}
        )

        found = set()
        for r in results:
            name = r.get('name', '')
            cls = r.get('drug_class', 'Unknown')
            groups[cls].append(name)
            found.add(name.lower())

        # Add any not found
        for d in drug_names:
            if d.lower() not in found:
                groups['Unknown'].append(d)

        return dict(groups)

    @staticmethod
    def get_class_interactions(class_a: str, class_b: str) -> Dict:
        """Get aggregate interaction stats between two drug classes."""
        from .knowledge_graph import KnowledgeGraphService
        results = KnowledgeGraphService.run_query(
            "MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug) "
            "WHERE COALESCE(d1.category, d1.therapeutic_class) = $class_a "
            "AND COALESCE(d2.category, d2.therapeutic_class) = $class_b "
            "RETURN i.severity AS severity, count(*) AS cnt",
            {'class_a': class_a, 'class_b': class_b}
        )
        severity_counts = {r['severity']: r['cnt'] for r in results}
        total = sum(severity_counts.values())
        return {
            'class_a': class_a,
            'class_b': class_b,
            'total_interactions': total,
            'severity_distribution': severity_counts,
        }

    @staticmethod
    def check_class_warnings(drug_names: List[str]) -> List[Dict]:
        """Check for class-level warnings: duplicate therapy and class-class patterns."""
        svc = DrugClassService
        groups = svc.group_by_class(drug_names)
        warnings = []

        # Warning 1: Duplicate therapy (multiple drugs in same class)
        for cls, drugs in groups.items():
            if cls != 'Unknown' and len(drugs) > 1:
                warnings.append({
                    'type': 'duplicate_therapy',
                    'severity': 'moderate',
                    'drug_class': cls,
                    'drugs': drugs,
                    'message': f'Multiple {cls} agents detected: {", ".join(drugs)}. Consider if duplicate therapy is intended.',
                })

        # Warning 2: Known high-risk class combinations
        HIGH_RISK_COMBOS = [
            ('Anticoagulants', 'NSAIDs', 'Increased bleeding risk with anticoagulant + NSAID combination'),
            ('Anticoagulants', 'Antiplatelet Agents', 'Triple antithrombotic risk: major bleeding'),
            ('SSRIs', 'MAO Inhibitors', 'Serotonin syndrome risk: potentially fatal'),
            ('ACE Inhibitors', 'Potassium-Sparing Diuretics', 'Hyperkalemia risk'),
            ('Statins', 'CYP3A4 Inhibitors', 'Increased statin exposure: rhabdomyolysis risk'),
        ]

        class_set = set(groups.keys())
        for combo_a, combo_b, msg in HIGH_RISK_COMBOS:
            if combo_a in class_set and combo_b in class_set:
                warnings.append({
                    'type': 'class_interaction',
                    'severity': 'high',
                    'classes': [combo_a, combo_b],
                    'drugs_a': groups[combo_a],
                    'drugs_b': groups[combo_b],
                    'message': msg,
                })

        return warnings
