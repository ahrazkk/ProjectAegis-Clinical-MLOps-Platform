"""Polypharmacy Digital Twin scoring service.

This service extends pairwise interaction analysis with transparent N-order
factors to support polypharmacy decision support:
- Pairwise baseline tension
- Enzyme competition load (CYP450)
- Shared target overlap (Neo4j TARGETS)
- Organ burden aggregation
- Network stress metrics
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from .cyp450_database import get_cyp450_database
from .gnn_predictor import get_gnn_predictor
from .knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)


class PolypharmacyDigitalTwinService:
    """Compute an explainable N-order toxicity profile for a drug set."""

    FACTOR_WEIGHTS = {
        "pairwise_baseline": 0.40,
        "enzyme_competition": 0.25,
        "target_overlap": 0.15,
        "organ_burden": 0.10,
        "network_stress": 0.10,
    }

    INTERACTION_EDGE_THRESHOLD = 0.20
    HIGH_RISK_THRESHOLD = 0.60

    def __init__(self):
        self._gnn_predictor = None
        self._cyp_db = None

    def _get_predictor(self):
        if self._gnn_predictor is None:
            self._gnn_predictor = get_gnn_predictor()
        return self._gnn_predictor

    def _get_cyp_db(self):
        if self._cyp_db is None:
            self._cyp_db = get_cyp450_database()
        return self._cyp_db

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.80:
            return "critical"
        if score >= 0.60:
            return "high"
        if score >= 0.30:
            return "medium"
        return "low"

    @staticmethod
    def _interaction_systems(interaction_type: str) -> List[str]:
        systems = {
            "mechanism": ["Metabolic/CYP450"],
            "effect": ["Pharmacodynamic"],
            "advise": ["Systemic"],
            "int": ["Systemic"],
        }
        return systems.get(interaction_type, ["Systemic"])

    @staticmethod
    def _normalize_drug_inputs(drugs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized = []
        for drug in drugs:
            name = (drug.get("name") or "Unknown").strip()
            normalized.append(
                {
                    "name": name,
                    "smiles": drug.get("smiles", "") or "",
                    "drugbank_id": drug.get("drugbank_id", "") or "",
                }
            )
        return normalized

    def _build_pairwise_tension(self, drugs: List[Dict[str, str]]) -> Dict[str, Any]:
        predictor = self._get_predictor()

        edges: List[Dict[str, Any]] = []
        total_pairs = len(drugs) * (len(drugs) - 1) // 2
        max_risk = 0.0
        risk_sum = 0.0
        considered_pairs = 0

        for left, right in combinations(drugs, 2):
            pred = predictor.predict(
                left["name"],
                right["name"],
                left.get("smiles", ""),
                right.get("smiles", ""),
            )

            risk_score = float(pred.interaction_probability)
            max_risk = max(max_risk, risk_score)
            risk_sum += risk_score
            considered_pairs += 1

            if risk_score < self.INTERACTION_EDGE_THRESHOLD:
                continue

            edges.append(
                {
                    "source": pred.drug1,
                    "target": pred.drug2,
                    "risk_score": risk_score,
                    "risk_level": self._risk_level(risk_score),
                    "severity": pred.severity,
                    "mechanism": pred.mechanism_hypothesis,
                    "affected_systems": self._interaction_systems(pred.interaction_type),
                }
            )

        edges.sort(key=lambda edge: edge["risk_score"], reverse=True)

        avg_risk = (risk_sum / considered_pairs) if considered_pairs else 0.0
        high_risk_pairs = sum(1 for edge in edges if edge["risk_score"] >= self.HIGH_RISK_THRESHOLD)
        baseline_score = min(1.0, (0.70 * max_risk) + (0.30 * avg_risk))

        return {
            "score": baseline_score,
            "max_pair_risk": max_risk,
            "average_pair_risk": avg_risk,
            "total_pairs": total_pairs,
            "considered_pairs": considered_pairs,
            "high_risk_pairs": high_risk_pairs,
            "edge_count": len(edges),
            "edges": edges,
        }

    @staticmethod
    def _inhibitor_strength(roles: List[str]) -> float:
        strength = 0.0
        for role in roles:
            if "inhibitor_strong" in role:
                strength = max(strength, 1.0)
            elif "inhibitor_moderate" in role:
                strength = max(strength, 0.7)
            elif "inhibitor_weak" in role or role == "inhibitor":
                strength = max(strength, 0.4)
        return strength

    def _compute_enzyme_competition(self, drugs: List[Dict[str, str]]) -> Dict[str, Any]:
        cyp_db = self._get_cyp_db()

        per_enzyme = defaultdict(lambda: {"substrates": 0, "inhibitor_strength": 0.0, "inducers": 0, "drugs": set()})

        for drug in drugs:
            profile = cyp_db.get_drug_cyp_profile(drug["name"])
            for enzyme, roles in profile.items():
                per_enzyme[enzyme]["drugs"].add(drug["name"])
                if "substrate" in roles:
                    per_enzyme[enzyme]["substrates"] += 1
                if "inducer" in roles:
                    per_enzyme[enzyme]["inducers"] += 1
                per_enzyme[enzyme]["inhibitor_strength"] = max(
                    per_enzyme[enzyme]["inhibitor_strength"],
                    self._inhibitor_strength(roles),
                )

        conflicts = []
        for enzyme, stats in sorted(per_enzyme.items()):
            substrates = stats["substrates"]
            inhibitor_strength = stats["inhibitor_strength"]
            inducers = stats["inducers"]

            conflict_score = 0.0
            if substrates >= 2:
                conflict_score += 0.25 + min(0.15, 0.05 * (substrates - 2))
            if inhibitor_strength > 0 and substrates > 0:
                conflict_score += 0.45 * inhibitor_strength
            if inducers > 0 and substrates > 0:
                conflict_score += 0.25 + min(0.10, 0.05 * (inducers - 1))
            if inhibitor_strength > 0 and inducers > 0:
                conflict_score += 0.10

            conflict_score = min(1.0, conflict_score)
            if conflict_score <= 0:
                continue

            conflicts.append(
                {
                    "enzyme": enzyme,
                    "score": conflict_score,
                    "substrate_count": substrates,
                    "inducer_count": inducers,
                    "inhibitor_strength": inhibitor_strength,
                    "involved_drugs": sorted(stats["drugs"]),
                }
            )

        conflicts.sort(key=lambda item: item["score"], reverse=True)
        score = sum(item["score"] for item in conflicts) / len(conflicts) if conflicts else 0.0

        return {
            "score": score,
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:10],
        }

    @staticmethod
    def _resolve_drugbank_id(drug: Dict[str, str]) -> str:
        if drug.get("drugbank_id"):
            return drug["drugbank_id"]

        if not KnowledgeGraphService.is_connected():
            return ""

        try:
            results = KnowledgeGraphService.search_drugs(drug["name"], limit=1)
            if not results:
                return ""
            return results[0].get("id", "")
        except Exception as exc:
            logger.warning("Failed to resolve drug ID for %s: %s", drug["name"], exc)
            return ""

    def _fetch_targets_for_drug(self, drugbank_id: str) -> Set[str]:
        if not drugbank_id:
            return set()

        query = """
            MATCH (d:Drug {drugbank_id: $drug_id})-[:TARGETS]->(t:Target)
            RETURN coalesce(t.gene_name, t.name, t.uniprot_id) as target
        """
        rows = KnowledgeGraphService.run_query(query, {"drug_id": drugbank_id})
        return {row.get("target") for row in rows if row.get("target")}

    def _compute_target_overlap(self, drugs: List[Dict[str, str]]) -> Dict[str, Any]:
        if not KnowledgeGraphService.is_connected():
            return {
                "score": 0.0,
                "available": False,
                "pair_count_with_overlap": 0,
                "conflicts": [],
            }

        targets_by_drug: Dict[str, Set[str]] = {}
        for drug in drugs:
            drug_id = self._resolve_drugbank_id(drug)
            targets_by_drug[drug["name"]] = self._fetch_targets_for_drug(drug_id)

        total_pairs = len(drugs) * (len(drugs) - 1) // 2
        conflicts = []

        for left, right in combinations(drugs, 2):
            left_targets = targets_by_drug.get(left["name"], set())
            right_targets = targets_by_drug.get(right["name"], set())
            overlap = sorted(left_targets.intersection(right_targets))
            if not overlap:
                continue
            conflicts.append(
                {
                    "drug_a": left["name"],
                    "drug_b": right["name"],
                    "shared_target_count": len(overlap),
                    "targets": overlap[:8],
                }
            )

        pair_overlap_count = len(conflicts)
        avg_overlap = (
            sum(item["shared_target_count"] for item in conflicts) / pair_overlap_count
            if pair_overlap_count
            else 0.0
        )

        if total_pairs == 0:
            score = 0.0
        else:
            score = min(
                1.0,
                (0.70 * (pair_overlap_count / total_pairs)) +
                (0.30 * min(avg_overlap / 3.0, 1.0)),
            )

        conflicts.sort(key=lambda item: item["shared_target_count"], reverse=True)

        return {
            "score": score,
            "available": True,
            "pair_count_with_overlap": pair_overlap_count,
            "conflicts": conflicts[:10],
        }

    def _compute_organ_burden(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        systems: Dict[str, float] = {}

        for edge in edges:
            risk = float(edge.get("risk_score", 0.0))
            for system in edge.get("affected_systems", []):
                systems[system] = max(systems.get(system, 0.0), risk)

        if not systems and edges:
            systems["Systemic"] = max(float(edge.get("risk_score", 0.0)) for edge in edges)

        ranked = sorted(systems.items(), key=lambda item: item[1], reverse=True)
        top_risks = [risk for _, risk in ranked[:3]]
        score = sum(top_risks) / len(top_risks) if top_risks else 0.0

        return {
            "score": score,
            "systems": {name: risk for name, risk in ranked},
            "top_systems": [name for name, _ in ranked[:3]],
        }

    def _compute_network_stress(self, drugs: List[Dict[str, str]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_pairs = len(drugs) * (len(drugs) - 1) // 2
        if total_pairs <= 0:
            return {
                "score": 0.0,
                "hub_drug": None,
                "hub_degree": 0,
                "edge_density": 0.0,
                "high_risk_density": 0.0,
            }

        degree = defaultdict(int)
        high_risk_edges = 0

        for edge in edges:
            degree[edge["source"]] += 1
            degree[edge["target"]] += 1
            if float(edge.get("risk_score", 0.0)) >= self.HIGH_RISK_THRESHOLD:
                high_risk_edges += 1

        hub_drug: Optional[str] = None
        hub_degree = 0
        if degree:
            hub_drug = max(degree, key=degree.get)
            hub_degree = degree[hub_drug]

        edge_density = len(edges) / total_pairs
        high_risk_density = high_risk_edges / total_pairs
        hub_pressure = (hub_degree / max(1, len(drugs) - 1)) if hub_drug else 0.0

        score = min(
            1.0,
            (0.40 * edge_density) +
            (0.35 * high_risk_density) +
            (0.25 * hub_pressure),
        )

        return {
            "score": score,
            "hub_drug": hub_drug,
            "hub_degree": hub_degree,
            "edge_density": edge_density,
            "high_risk_density": high_risk_density,
        }

    @staticmethod
    def _confidence_tier(pairwise: Dict[str, Any], enzyme: Dict[str, Any], target: Dict[str, Any]) -> str:
        if target.get("available") and target.get("pair_count_with_overlap", 0) > 0 and enzyme.get("conflict_count", 0) > 0:
            return "evidence-backed"
        if enzyme.get("conflict_count", 0) > 0 or pairwise.get("high_risk_pairs", 0) > 0:
            return "graph-supported"
        return "heuristic"

    def _build_recommendations(
        self,
        summary_score: float,
        enzyme: Dict[str, Any],
        target: Dict[str, Any],
        network: Dict[str, Any],
    ) -> List[str]:
        recommendations: List[str] = []

        if summary_score >= 0.80:
            recommendations.append(
                "Critical cumulative toxicity signal detected. Immediate pharmacist or physician review is recommended before co-administration."
            )
        elif summary_score >= 0.60:
            recommendations.append(
                "High systemic interaction load detected. Consider dose adjustments, alternative agents, and enhanced monitoring."
            )

        if enzyme.get("conflict_count", 0) > 0:
            top = enzyme["conflicts"][0]
            recommendations.append(
                f"Metabolic bottleneck risk detected on {top['enzyme']} with {top['substrate_count']} substrate-linked drugs."
            )

        if target.get("pair_count_with_overlap", 0) > 0:
            recommendations.append(
                "Shared target overlap detected across multiple drug pairs, indicating additive or competing pharmacodynamic pressure."
            )

        if network.get("hub_drug"):
            recommendations.append(
                f"{network['hub_drug']} appears as a network hub and should be prioritized in medication review."
            )

        if not recommendations:
            recommendations.append("No major polypharmacy stressors detected in current combination; continue routine monitoring.")

        return recommendations

    def analyze(self, drugs: List[Dict[str, str]]) -> Dict[str, Any]:
        normalized_drugs = self._normalize_drug_inputs(drugs)

        pairwise = self._build_pairwise_tension(normalized_drugs)
        enzyme = self._compute_enzyme_competition(normalized_drugs)
        target = self._compute_target_overlap(normalized_drugs)
        organ = self._compute_organ_burden(pairwise["edges"])
        network = self._compute_network_stress(normalized_drugs, pairwise["edges"])

        factors = {
            "pairwise_baseline": pairwise,
            "enzyme_competition": enzyme,
            "target_overlap": target,
            "organ_burden": organ,
            "network_stress": network,
        }

        weighted_contributions = {
            name: factors[name]["score"] * self.FACTOR_WEIGHTS[name]
            for name in self.FACTOR_WEIGHTS
        }

        toxicity_score = sum(
            factors[name]["score"] * weight
            for name, weight in self.FACTOR_WEIGHTS.items()
        )
        toxicity_score = min(1.0, toxicity_score)

        confidence_tier = self._confidence_tier(pairwise, enzyme, target)
        recommendations = self._build_recommendations(toxicity_score, enzyme, target, network)

        nodes = [{"id": drug["name"], "type": "drug"} for drug in normalized_drugs]

        return {
            "drugs": [drug["name"] for drug in normalized_drugs],
            "summary": {
                "toxicity_score": toxicity_score,
                "risk_level": self._risk_level(toxicity_score),
                "confidence_tier": confidence_tier,
            },
            "factors": {
                "pairwise_baseline": {
                    "score": pairwise["score"],
                    "max_pair_risk": pairwise["max_pair_risk"],
                    "average_pair_risk": pairwise["average_pair_risk"],
                    "total_pairs": pairwise["total_pairs"],
                    "high_risk_pairs": pairwise["high_risk_pairs"],
                },
                "enzyme_competition": {
                    "score": enzyme["score"],
                    "conflict_count": enzyme["conflict_count"],
                    "conflicts": enzyme["conflicts"],
                },
                "target_overlap": {
                    "score": target["score"],
                    "available": target["available"],
                    "pair_count_with_overlap": target["pair_count_with_overlap"],
                    "conflicts": target["conflicts"],
                },
                "organ_burden": {
                    "score": organ["score"],
                    "systems": organ["systems"],
                    "top_systems": organ["top_systems"],
                },
                "network_stress": {
                    "score": network["score"],
                    "hub_drug": network["hub_drug"],
                    "hub_degree": network["hub_degree"],
                    "edge_density": network["edge_density"],
                    "high_risk_density": network["high_risk_density"],
                },
            },
            "tension_network": {
                "nodes": nodes,
                "edges": pairwise["edges"],
                "hub_drug": network["hub_drug"],
            },
            "explainability": {
                "weights": self.FACTOR_WEIGHTS,
                "weighted_contributions": weighted_contributions,
                "composite_formula": "toxicity_score = sum(weight_i * factor_score_i)",
            },
            "body_map": organ["systems"],
            "recommendations": recommendations,
        }


_polypharmacy_digital_twin_service: Optional[PolypharmacyDigitalTwinService] = None


def get_polypharmacy_digital_twin_service() -> PolypharmacyDigitalTwinService:
    """Get or create the Polypharmacy Digital Twin singleton service."""
    global _polypharmacy_digital_twin_service
    if _polypharmacy_digital_twin_service is None:
        _polypharmacy_digital_twin_service = PolypharmacyDigitalTwinService()
    return _polypharmacy_digital_twin_service
