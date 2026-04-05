"""
Slash Command Router for Aegis Research Assistant

Parses /command syntax and routes to existing backend services.
Commands call Python service functions directly (no internal HTTP).
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedCommand:
    """Result of parsing a chat message for slash commands."""
    command: Optional[str]   # 'test', 'poly', 'compare', 'alt', 'evidence', or None
    args: List[str]          # Drug names extracted from the command
    raw: str                 # Original message text
    is_command: bool         # True if message starts with /


# Command definitions: name -> {min_args, max_args, desc, usage}
COMMANDS = {
    'test': {
        'min_args': 2, 'max_args': 2,
        'desc': 'Test interaction between two drugs',
        'usage': '/test warfarin aspirin',
    },
    'poly': {
        'min_args': 2, 'max_args': 10,
        'desc': 'Analyze polypharmacy risk for multiple drugs',
        'usage': '/poly warfarin,aspirin,ibuprofen',
    },
    'compare': {
        'min_args': 2, 'max_args': 5,
        'desc': 'Compare drugs side-by-side',
        'usage': '/compare metformin glipizide',
    },
    'alt': {
        'min_args': 1, 'max_args': 2,
        'desc': 'Find safer therapeutic alternatives',
        'usage': '/alt ibuprofen',
    },
    'evidence': {
        'min_args': 2, 'max_args': 2,
        'desc': 'Show full evidence chain for a drug pair',
        'usage': '/evidence warfarin aspirin',
    },
    'research': {
        'min_args': 1, 'max_args': 2,
        'desc': 'Deep research a drug or pair via KG + PubMed + FAERS',
        'usage': '/research warfarin aspirin',
    },
    'mutate': {
        'min_args': 3, 'max_args': 10,
        'desc': 'Test removing a drug from a regimen to find the safest option',
        'usage': '/mutate warfarin aspirin ibuprofen',
    },
    'current': {
        'min_args': 0, 'max_args': 0,
        'desc': 'Analyze drugs currently selected in the sidebar',
        'usage': '/current',
        'uses_context': True,
    },
    'demo': {
        'min_args': 0, 'max_args': 1,
        'desc': 'Run a pre-built demo case (list with /demo)',
        'usage': '/demo cardiac',
    },
    'class': {
        'min_args': 1, 'max_args': 10,
        'desc': 'Group drugs by class and check class-level interactions',
        'usage': '/class warfarin aspirin ibuprofen',
    },
}


class CommandRouter:
    """Parses slash commands and routes to existing backend services."""

    def parse(self, message: str) -> ParsedCommand:
        """
        Parse a chat message for slash commands.

        Supports:
          /test warfarin aspirin
          /poly warfarin,aspirin,ibuprofen
          /poly warfarin aspirin ibuprofen
        """
        message = message.strip()
        if not message.startswith('/'):
            return ParsedCommand(command=None, args=[], raw=message, is_command=False)

        parts = message.split(None, 1)  # Split on first whitespace
        cmd = parts[0][1:].lower()       # Remove '/' prefix
        arg_str = parts[1] if len(parts) > 1 else ''

        # Parse args: support both space-separated and comma-separated
        args = [a.strip().lower() for a in re.split(r'[,\s]+', arg_str) if a.strip()]

        return ParsedCommand(command=cmd, args=args, raw=message, is_command=True)

    def execute(self, parsed: ParsedCommand, context_drugs: List[str] = None) -> Dict[str, Any]:
        """
        Route parsed command to the appropriate handler.

        Returns a structured dict with results or error information.
        """
        if parsed.command not in COMMANDS:
            return {
                'error': f'Unknown command: /{parsed.command}',
                'available_commands': [
                    {'command': f'/{k}', 'description': v['desc'], 'usage': v['usage']}
                    for k, v in COMMANDS.items()
                ],
            }

        spec = COMMANDS[parsed.command]

        # Commands that use context_drugs instead of args
        if spec.get('uses_context'):
            handler = getattr(self, f'_handle_{parsed.command}', None)
            if not handler:
                return {'error': f'Handler not implemented for /{parsed.command}'}
            try:
                return handler(parsed.args, context_drugs=context_drugs or [])
            except Exception as e:
                logger.error("Command /%s failed: %s", parsed.command, e)
                return {'error': f'Command /{parsed.command} failed: {str(e)[:300]}'}

        if len(parsed.args) < spec['min_args']:
            return {
                'error': f'/{parsed.command} requires at least {spec["min_args"]} drug name(s)',
                'usage': spec['usage'],
            }

        # Trim to max_args
        args = parsed.args[:spec['max_args']]

        handler = getattr(self, f'_handle_{parsed.command}', None)
        if not handler:
            return {'error': f'Handler not implemented for /{parsed.command}'}

        try:
            return handler(args)
        except Exception as e:
            logger.error("Command /%s failed: %s", parsed.command, e)
            return {'error': f'Command /{parsed.command} failed: {str(e)[:300]}'}

    def _handle_test(self, drugs: List[str]) -> Dict[str, Any]:
        """Run GNN prediction for a drug pair."""
        from .gnn_predictor import get_gnn_predictor

        predictor = get_gnn_predictor()
        result = predictor.predict(drugs[0], drugs[1])

        return {
            'command': 'test',
            'drug_a': drugs[0],
            'drug_b': drugs[1],
            'risk_score': result.interaction_probability,
            'confidence': result.confidence,
            'severity': result.severity,
            'interaction_type': result.interaction_type,
            'mechanism': result.mechanism_hypothesis,
            'model_used': result.model_used,
            'calibration_method': result.calibration_method,
            'fingerprint_similarity': result.fingerprint_similarity,
            'raw_probability': result.raw_interaction_probability,
        }

    def _handle_poly(self, drugs: List[str]) -> Dict[str, Any]:
        """Run polypharmacy analysis for multiple drugs."""
        from .gnn_predictor import get_gnn_predictor

        predictor = get_gnn_predictor()
        drug_dicts = [{'name': d} for d in drugs]
        result = predictor.predict_polypharmacy(drug_dicts)

        return {
            'command': 'poly',
            'drugs': drugs,
            'result': result,
        }

    def _handle_compare(self, drugs: List[str]) -> Dict[str, Any]:
        """Compare drugs using enhanced drug service."""
        from .enhanced_drug_service import get_enhanced_drug_service

        svc = get_enhanced_drug_service()
        drug_infos = []
        for d in drugs:
            try:
                info = svc.get_drug_info(d)
                if info:
                    drug_infos.append({
                        'name': d,
                        'drugbank_id': getattr(info, 'drugbank_id', None),
                        'category': getattr(info, 'category', None),
                        'description': getattr(info, 'description', None),
                        'interaction_count': getattr(info, 'interaction_count', 0),
                        'side_effects_count': len(getattr(info, 'side_effects', [])),
                    })
                else:
                    drug_infos.append({'name': d, 'error': 'Drug not found'})
            except Exception as e:
                drug_infos.append({'name': d, 'error': str(e)[:200]})

        return {
            'command': 'compare',
            'drugs': drugs,
            'results': drug_infos,
        }

    def _handle_alt(self, drugs: List[str]) -> Dict[str, Any]:
        """Find therapeutic alternatives via Knowledge Graph shared targets."""
        from .knowledge_graph import KnowledgeGraphService

        drug_name = drugs[0]
        interacting_drug = drugs[1] if len(drugs) > 1 else None

        # Find drugs that share protein targets (same therapeutic class)
        results = KnowledgeGraphService.run_query(
            "MATCH (d:Drug)-[:TARGETS]->(t)<-[:TARGETS]-(alt:Drug) "
            "WHERE toLower(d.name) = $name AND alt.name <> d.name "
            "RETURN DISTINCT alt.name as name, alt.drugbank_id as drugbank_id, "
            "collect(DISTINCT t.name) as shared_targets "
            "LIMIT 10",
            {'name': drug_name.lower()}
        )

        alternatives = []
        for r in results:
            alt = {
                'name': r.get('name', ''),
                'drugbank_id': r.get('drugbank_id', ''),
                'shared_targets': r.get('shared_targets', []),
            }
            # If an interacting drug is specified, check if alternative also interacts
            if interacting_drug:
                conflicts = KnowledgeGraphService.run_query(
                    "MATCH (a:Drug)-[i:INTERACTS_WITH]-(b:Drug) "
                    "WHERE toLower(a.name) = $alt AND toLower(b.name) = $inter "
                    "RETURN i.severity as severity",
                    {'alt': r.get('name', '').lower(), 'inter': interacting_drug.lower()}
                )
                alt['interacts_with_target'] = len(conflicts) > 0
                if conflicts:
                    alt['interaction_severity'] = conflicts[0].get('severity', 'unknown')
            alternatives.append(alt)

        return {
            'command': 'alt',
            'drug': drug_name,
            'interacting_with': interacting_drug,
            'alternatives': alternatives,
        }

    def _handle_evidence(self, drugs: List[str]) -> Dict[str, Any]:
        """Get full evidence chain from EnhancedDrugService."""
        from .enhanced_drug_service import get_enhanced_drug_service

        svc = get_enhanced_drug_service()
        try:
            info = svc.get_interaction_info(drugs[0], drugs[1])
            if info:
                return {
                    'command': 'evidence',
                    'drug_a': drugs[0],
                    'drug_b': drugs[1],
                    'severity': info.severity,
                    'risk_score': info.risk_score,
                    'mechanism': info.mechanism,
                    'evidence_chain': info.evidence_chain,
                    'evidence_summary': info.evidence_summary,
                    'faers_reports': info.faers_reports,
                    'faers_caveats': info.faers_caveats,
                    'polypharmacy_effects_count': len(info.polypharmacy_effects) if info.polypharmacy_effects else 0,
                }
            return {
                'command': 'evidence',
                'drug_a': drugs[0],
                'drug_b': drugs[1],
                'error': 'No interaction data found for this drug pair',
            }
        except Exception as e:
            return {
                'command': 'evidence',
                'drug_a': drugs[0],
                'drug_b': drugs[1],
                'error': str(e)[:300],
            }


    # ------------------------------------------------------------------
    # Demo case definitions
    # ------------------------------------------------------------------
    DEMO_CASES = {
        'cardiac': {
            'name': 'Cardiac High-Risk Polypharmacy',
            'desc': 'Heart failure patient on anticoagulant + antiarrhythmic + antiplatelet — high bleed risk',
            'drugs': ['warfarin', 'amiodarone', 'aspirin'],
            'action': 'poly',
        },
        'psych': {
            'name': 'Psychiatric Serotonin Syndrome Risk',
            'desc': 'SSRI + MAOI combination — risk of serotonin syndrome',
            'drugs': ['fluoxetine', 'phenelzine'],
            'action': 'test',
        },
        'pain': {
            'name': 'Chronic Pain + Blood Thinner',
            'desc': 'NSAID + anticoagulant — GI bleed risk with mutation scan',
            'drugs': ['ibuprofen', 'warfarin', 'acetaminophen'],
            'action': 'mutate',
        },
        'elderly': {
            'name': 'Elderly Polypharmacy (5 drugs)',
            'desc': 'Common elderly regimen: statin + ACE inhibitor + metformin + aspirin + omeprazole',
            'drugs': ['atorvastatin', 'lisinopril', 'metformin', 'aspirin', 'omeprazole'],
            'action': 'poly',
        },
        'onc': {
            'name': 'Oncology Interaction Check',
            'desc': 'Chemotherapy agent + common supportive care — check for efficacy reduction',
            'drugs': ['methotrexate', 'ibuprofen'],
            'action': 'test',
        },
        'qtwarn': {
            'name': 'QT Prolongation Risk',
            'desc': 'Two QT-prolonging agents — risk of cardiac arrhythmia',
            'drugs': ['erbitux', 'amitriptyline'],
            'action': 'test',
        },
        'transplant': {
            'name': 'Transplant Drug Interactions',
            'desc': 'Immunosuppressant + antifungal — CYP3A4 inhibition risk',
            'drugs': ['cyclosporine', 'ketoconazole'],
            'action': 'test',
        },
    }

    def _handle_research(self, drugs: List[str]) -> Dict[str, Any]:
        """Deep research a drug or pair: KG data + evidence + alternatives."""
        from .enhanced_drug_service import get_enhanced_drug_service
        from .knowledge_graph import KnowledgeGraphService

        svc = get_enhanced_drug_service()
        result = {'command': 'research', 'drugs': drugs}

        # Get drug info for each
        drug_profiles = []
        for d in drugs:
            try:
                info = svc.get_drug_info(d)
                if info:
                    drug_profiles.append({
                        'name': d,
                        'drugbank_id': getattr(info, 'drugbank_id', None),
                        'category': getattr(info, 'category', None),
                        'description': getattr(info, 'description', None),
                        'targets': getattr(info, 'targets', []),
                        'side_effects': (getattr(info, 'side_effects', []) or [])[:10],
                    })
                else:
                    drug_profiles.append({'name': d, 'error': 'Not found in KG'})
            except Exception as e:
                drug_profiles.append({'name': d, 'error': str(e)[:200]})
        result['profiles'] = drug_profiles

        # If pair, get interaction evidence
        if len(drugs) == 2:
            try:
                info = svc.get_interaction_info(drugs[0], drugs[1])
                if info:
                    result['interaction'] = {
                        'severity': info.severity,
                        'risk_score': info.risk_score,
                        'mechanism': info.mechanism,
                        'evidence_summary': info.evidence_summary,
                        'faers_reports': info.faers_reports,
                        'evidence_chain': info.evidence_chain,
                    }
            except Exception:
                pass

            # Also run GNN prediction
            try:
                from .gnn_predictor import get_gnn_predictor
                predictor = get_gnn_predictor()
                pred = predictor.predict(drugs[0], drugs[1])
                result['gnn_prediction'] = {
                    'risk_score': pred.interaction_probability,
                    'confidence': pred.confidence,
                    'severity': pred.severity,
                    'mechanism': pred.mechanism_hypothesis,
                }
            except Exception:
                pass

            # Check for approved corrections
            try:
                from .correction_memory import get_correction_memory
                mem = get_correction_memory()
                correction = mem.get_approved_correction(drugs[0], drugs[1])
                if correction:
                    result['correction_overlay'] = correction
            except Exception:
                pass

        return result

    def _handle_mutate(self, drugs: List[str]) -> Dict[str, Any]:
        """
        Mutation scan: test removing each drug from the regimen
        and report which removal reduces risk the most.
        """
        from .gnn_predictor import get_gnn_predictor
        import itertools

        predictor = get_gnn_predictor()

        # First, get baseline poly score with all drugs
        drug_dicts = [{'name': d} for d in drugs]
        try:
            baseline = predictor.predict_polypharmacy(drug_dicts)
            baseline_risk = float(baseline.get('composite_risk', baseline.get('overall_risk', 0)))
        except Exception:
            baseline_risk = 0.0
            baseline = {}

        # Test removing each drug
        mutations = []
        for i, removed_drug in enumerate(drugs):
            remaining = [d for j, d in enumerate(drugs) if j != i]
            if len(remaining) < 2:
                # Can't run poly with < 2, run pairwise instead
                if len(remaining) == 1:
                    mutations.append({
                        'removed': removed_drug,
                        'remaining': remaining,
                        'risk': 0.0,
                        'risk_reduction': baseline_risk,
                        'note': 'Only 1 drug remaining — no interaction risk',
                    })
                continue

            try:
                rem_dicts = [{'name': d} for d in remaining]
                mut_result = predictor.predict_polypharmacy(rem_dicts)
                mut_risk = float(mut_result.get('composite_risk', mut_result.get('overall_risk', 0)))
                mutations.append({
                    'removed': removed_drug,
                    'remaining': remaining,
                    'risk': mut_risk,
                    'risk_reduction': baseline_risk - mut_risk,
                    'interactions_remaining': mut_result.get('total_interactions', 0),
                })
            except Exception as e:
                mutations.append({
                    'removed': removed_drug,
                    'remaining': remaining,
                    'error': str(e)[:200],
                })

        # Sort by risk reduction (best removal first)
        mutations.sort(key=lambda m: m.get('risk_reduction', 0), reverse=True)

        return {
            'command': 'mutate',
            'drugs': drugs,
            'baseline_risk': baseline_risk,
            'baseline_interactions': baseline.get('total_interactions', 0),
            'mutations': mutations,
            'best_removal': mutations[0]['removed'] if mutations else None,
        }

    def _handle_current(self, drugs: List[str], context_drugs: List[str] = None) -> Dict[str, Any]:
        """Analyze currently selected sidebar drugs."""
        if not context_drugs or len(context_drugs) == 0:
            return {
                'command': 'current',
                'error': 'No drugs currently selected in the sidebar. Add drugs first, then use /current.',
                'usage': '/current',
            }

        # Route to appropriate handler based on count
        if len(context_drugs) == 2:
            return self._handle_test(context_drugs)
        else:
            return self._handle_poly(context_drugs)

    def _handle_demo(self, args: List[str]) -> Dict[str, Any]:
        """Run a demo case or list available demos."""
        if not args or args[0] == 'list':
            # List all demo cases
            demos = []
            for key, case in self.DEMO_CASES.items():
                demos.append({
                    'key': key,
                    'name': case['name'],
                    'desc': case['desc'],
                    'drugs': case['drugs'],
                    'action': case['action'],
                })
            return {
                'command': 'demo',
                'action': 'list',
                'demos': demos,
            }

        case_key = args[0].lower()
        if case_key not in self.DEMO_CASES:
            return {
                'command': 'demo',
                'error': f'Unknown demo: {case_key}',
                'available': list(self.DEMO_CASES.keys()),
                'usage': '/demo cardiac',
            }

        case = self.DEMO_CASES[case_key]
        handler = getattr(self, f'_handle_{case["action"]}', None)
        if not handler:
            return {'error': f'Demo action handler not found: {case["action"]}'}

        result = handler(case['drugs'])
        result['demo_name'] = case['name']
        result['demo_desc'] = case['desc']
        return result

    def _handle_class(self, drugs: List[str]) -> Dict[str, Any]:
        """Group drugs by therapeutic class and check for warnings."""
        from .drug_class_service import DrugClassService
        svc = DrugClassService
        groups = svc.group_by_class(drugs)
        warnings = svc.check_class_warnings(drugs)
        return {
            'command': 'class',
            'drugs': drugs,
            'groups': groups,
            'warnings': warnings,
        }


def get_command_list() -> List[Dict[str, str]]:
    """Return command list for frontend autocomplete."""
    return [
        {'command': f'/{k}', 'description': v['desc'], 'usage': v['usage']}
        for k, v in COMMANDS.items()
    ]
