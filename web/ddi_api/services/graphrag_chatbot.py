"""
GraphRAG Chatbot Service

Implements a Retrieval-Augmented Generation (RAG) system using:
1. Neo4j Knowledge Graph for drug/interaction data retrieval
2. Google Gemini LLM for intelligent response generation
3. PubMed literature retrieval for evidence grounding
4. Slash command routing for deterministic tool workflows
5. Template-based fallback when LLM is unavailable
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from django.conf import settings

from .knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from the GraphRAG chatbot."""
    response: str
    sources: List[Dict]
    related_drugs: List[str]
    graph_context: Optional[Dict] = None
    citations: List[Dict] = field(default_factory=list)
    token_usage: Optional[Dict] = None


class GraphRAGChatbot:
    """
    GraphRAG-powered research assistant for drug interactions.

    Uses Neo4j Knowledge Graph to retrieve relevant drug information
    and generates contextual responses. When LLM mode is enabled,
    uses Google Gemini with RAG context for evidence-based answers.
    """

    def __init__(self):
        self.kg = KnowledgeGraphService
        self.use_llm = False  # Set per-request by ChatView based on access control

        # Lazy-init command router
        self._command_router = None

    @property
    def command_router(self):
        if self._command_router is None:
            from .command_router import CommandRouter
            self._command_router = CommandRouter()
        return self._command_router

    def process_message(
        self,
        message: str,
        context_drugs: List[str] = None,
        session_id: str = None
    ) -> ChatResponse:
        """
        Process a user message and generate a response.

        Args:
            message: User's question or slash command
            context_drugs: List of drug names in current context
            session_id: Chat session identifier

        Returns:
            ChatResponse with answer, sources, related drugs, and citations
        """
        context_drugs = context_drugs or []

        # --- Slash command routing (Phase 2) ---
        parsed = self.command_router.parse(message)
        if parsed.is_command:
            return self._handle_command(parsed, context_drugs)

        # --- Natural language flow ---
        # Step 1: Extract drug names from the message
        mentioned_drugs = self._extract_drug_names(message)
        all_drugs = list(set(context_drugs + mentioned_drugs))

        # Step 2: Retrieve relevant context from Knowledge Graph
        graph_context = self._retrieve_graph_context(all_drugs, message)

        # Step 3: Generate response (LLM or template)
        citations = []
        token_usage = None
        if self.use_llm:
            # Retrieve PubMed context for drug pairs
            pubmed_results = self._retrieve_pubmed_context(all_drugs)
            response_text, citations, token_usage = self._generate_llm_response(
                message, graph_context, pubmed_results, all_drugs
            )
            # Fall back to templates if LLM fails
            if response_text is None:
                response_text = self._generate_response(message, graph_context, all_drugs)
        else:
            response_text = self._generate_response(message, graph_context, all_drugs)

        # Step 4: Compile sources
        sources = self._compile_sources(graph_context)

        return ChatResponse(
            response=response_text,
            sources=sources,
            related_drugs=graph_context.get('related_drugs', []),
            graph_context=graph_context,
            citations=citations,
            token_usage=token_usage,
        )

    # =========================================================================
    # Slash Command Handling (Phase 2)
    # =========================================================================

    def _handle_command(self, parsed, context_drugs: List[str]) -> ChatResponse:
        """Execute a slash command and return formatted response."""
        tool_result = self.command_router.execute(parsed, context_drugs=context_drugs)

        if 'error' in tool_result and 'available_commands' in tool_result:
            # Unknown command -- show help
            help_lines = [f"**Unknown command:** `/{parsed.command}`\n\nAvailable commands:"]
            for cmd in tool_result['available_commands']:
                help_lines.append(f"- `{cmd['command']}` -- {cmd['description']}")
                help_lines.append(f"  Usage: `{cmd['usage']}`")
            return ChatResponse(
                response='\n'.join(help_lines),
                sources=[],
                related_drugs=parsed.args,
            )

        if 'error' in tool_result:
            return ChatResponse(
                response=f"**Command error:** {tool_result['error']}\n\nUsage: `{tool_result.get('usage', '')}`",
                sources=[],
                related_drugs=parsed.args,
            )

        # Enrich tool result with correction memory if available
        correction_context = self._lookup_corrections(parsed.args)
        if correction_context:
            tool_result['_correction_overlay'] = correction_context

        # Flag low confidence for /test results and auto-capture as pending correction
        if parsed.command == 'test' and 'confidence' in tool_result:
            if tool_result['confidence'] < 0.5:
                tool_result['_low_confidence_flag'] = True
                # Auto-capture low-confidence prediction to corrections queue
                if len(parsed.args) == 2:
                    try:
                        from .correction_memory import get_correction_memory
                        mem = get_correction_memory()
                        mem.create_correction(
                            drug_a=parsed.args[0],
                            drug_b=parsed.args[1],
                            gnn_severity=str(tool_result.get('severity', tool_result.get('risk_level', ''))),
                            gnn_risk_score=float(tool_result.get('risk_score', 0)),
                            gnn_confidence=float(tool_result.get('confidence', 0)),
                            corrected_severity='',
                            evidence_source='auto-capture:low-confidence',
                        )
                        logger.info("Auto-captured low-confidence /test result for %s + %s", parsed.args[0], parsed.args[1])
                    except Exception as e:
                        logger.warning("Auto-capture correction failed: %s", e)

        # Also auto-capture low-confidence polypharmacy interactions
        if parsed.command == 'poly' and 'interactions' in tool_result:
            try:
                from .correction_memory import get_correction_memory
                mem = get_correction_memory()
                for interaction in tool_result.get('interactions', []):
                    conf = interaction.get('confidence', 1.0)
                    if conf < 0.5:
                        drug_a = interaction.get('drug_a', interaction.get('source', ''))
                        drug_b = interaction.get('drug_b', interaction.get('target', ''))
                        if drug_a and drug_b:
                            mem.create_correction(
                                drug_a=drug_a,
                                drug_b=drug_b,
                                gnn_severity=str(interaction.get('severity', '')),
                                gnn_risk_score=float(interaction.get('risk_score', 0)),
                                gnn_confidence=float(conf),
                                corrected_severity='',
                                evidence_source='auto-capture:low-confidence',
                            )
            except Exception as e:
                logger.warning("Poly auto-capture failed: %s", e)

        # If LLM is enabled, let Gemini interpret the tool results
        citations = []
        if self.use_llm:
            response_text, citations, token_usage = self._interpret_tool_result(parsed.raw, tool_result)
            if response_text:
                sources = [{'title': 'Project Aegis Engine', 'type': 'internal', 'url': '#'}]
                return ChatResponse(
                    response=response_text,
                    sources=sources,
                    related_drugs=parsed.args,
                    citations=citations,
                    token_usage=token_usage,
                )

        # Fallback: format tool result as structured text
        response_text = self._format_tool_result(tool_result)
        sources = [{'title': 'Project Aegis Engine', 'type': 'internal', 'url': '#'}]
        return ChatResponse(
            response=response_text,
            sources=sources,
            related_drugs=parsed.args,
        )

    def _lookup_corrections(self, drug_args: List[str]) -> Optional[Dict]:
        """Check if an approved correction exists for the given drug pair."""
        if len(drug_args) != 2:
            return None
        try:
            from .correction_memory import get_correction_memory
            mem = get_correction_memory()
            correction = mem.get_approved_correction(drug_args[0], drug_args[1])
            if correction:
                return {
                    'corrected_severity': correction.get('corrected_severity'),
                    'evidence_text': correction.get('evidence_text', ''),
                    'evidence_source': correction.get('evidence_source', ''),
                    'gnn_severity': correction.get('gnn_severity'),
                }
        except Exception as e:
            logger.warning("Correction lookup failed: %s", e)
        return None

    def _interpret_tool_result(self, original_message: str, tool_result: dict) -> Tuple[Optional[str], List[Dict], Optional[Dict]]:
        """Send tool output to Gemini for clinical interpretation."""
        from .gemini_client import get_gemini_client

        client = get_gemini_client()
        if not client.is_available():
            return None, [], None

        # Build correction overlay text
        correction_note = ''
        if '_correction_overlay' in tool_result:
            c = tool_result.pop('_correction_overlay')
            correction_note = (
                f"\n\n=== APPROVED CORRECTION ===\n"
                f"A previous expert review corrected this prediction.\n"
                f"GNN predicted: {c.get('gnn_severity')} | Corrected to: {c.get('corrected_severity')}\n"
                f"Evidence: {c.get('evidence_text', 'N/A')}\n"
                f"Source: {c.get('evidence_source', 'N/A')}\n"
                f"IMPORTANT: Mention this correction in your response and note the discrepancy."
            )

        low_conf_note = ''
        if tool_result.pop('_low_confidence_flag', False):
            low_conf_note = (
                "\n\nWARNING: This prediction has LOW CONFIDENCE (< 0.5). "
                "You MUST prominently flag this in your response and suggest "
                "the user submit a correction if they have better evidence."
            )

        # Serialize tool result as context
        context = f"""=== TOOL EXECUTION RESULT ===
Command: {original_message}
Result:
{json.dumps(tool_result, indent=2, default=str)[:3000]}
{correction_note}{low_conf_note}"""

        result = client.generate(
            f"The user ran the command: `{original_message}`\n"
            f"Interpret these results clinically. Highlight key risks, "
            f"explain the mechanism, and note if confidence is low. "
            f"If the command was /test, comment on whether the severity seems appropriate based on the evidence."
            f"{' Flag the low confidence prominently.' if low_conf_note else ''}"
            f"{' Reference the approved correction.' if correction_note else ''}",
            context
        )
        token_usage = {
            'input_tokens': result.input_tokens,
            'output_tokens': result.output_tokens,
            'estimated_cost_usd': result.estimated_cost_usd,
        }
        return result.text, result.citations, token_usage

    def _format_tool_result(self, result: dict) -> str:
        """Format tool result as readable text (non-LLM fallback)."""
        command = result.get('command', '')

        if command == 'test':
            severity = result.get('severity', 'unknown')
            emoji = {'severe': '🔴', 'major': '🔴', 'moderate': '🟡', 'minor': '🟢', 'none': '✅'}.get(severity, '⚪')
            return (
                f"**Interaction Test: {result.get('drug_a', '')} + {result.get('drug_b', '')}**\n\n"
                f"{emoji} **Severity:** {severity.upper()}\n"
                f"- Risk Score: {result.get('risk_score', 0):.3f}\n"
                f"- Confidence: {result.get('confidence', 0):.3f}\n"
                f"- Type: {result.get('interaction_type', 'unknown')}\n"
                f"- Model: {result.get('model_used', 'unknown')}\n\n"
                f"**Mechanism:** {result.get('mechanism', 'Under investigation')}"
            )

        if command == 'evidence':
            summary = result.get('evidence_summary', {})
            chain = result.get('evidence_chain', [])
            return (
                f"**Evidence Chain: {result.get('drug_a', '')} + {result.get('drug_b', '')}**\n\n"
                f"- Severity: {result.get('severity', 'unknown')}\n"
                f"- Risk Score: {result.get('risk_score', 0):.3f}\n"
                f"- Evidence sources: {len(chain)} entries\n"
                f"- Confidence band: {summary.get('confidence_band', 'unknown')}\n"
                f"- FAERS reports: {result.get('faers_reports', 0)}\n\n"
                f"Run with LLM mode for detailed clinical interpretation."
            )

        if command == 'poly':
            drugs = result.get('drugs', [])
            return (
                f"**Polypharmacy Analysis: {', '.join(drugs)}**\n\n"
                f"```json\n{json.dumps(result.get('result', {}), indent=2, default=str)[:2000]}\n```\n\n"
                f"Run with LLM mode for clinical interpretation."
            )

        if command == 'compare':
            results = result.get('results', [])
            parts = [f"**Drug Comparison: {', '.join(result.get('drugs', []))}**\n"]
            for info in results:
                if 'error' in info:
                    parts.append(f"\n**{info['name']}:** {info['error']}")
                else:
                    parts.append(
                        f"\n**{info['name']}** (DrugBank: {info.get('drugbank_id', 'N/A')})\n"
                        f"- Category: {info.get('category', 'N/A')}\n"
                        f"- Interactions: {info.get('interaction_count', 0)}\n"
                        f"- Side effects: {info.get('side_effects_count', 0)}"
                    )
            return '\n'.join(parts)

        if command == 'alt':
            alts = result.get('alternatives', [])
            parts = [f"**Alternatives for {result.get('drug', '')}**"]
            if result.get('interacting_with'):
                parts[0] += f" (avoiding interaction with {result['interacting_with']})"
            parts.append('')
            if not alts:
                parts.append("No alternatives found sharing targets in the Knowledge Graph.")
            for alt in alts:
                safe_marker = ''
                if 'interacts_with_target' in alt:
                    safe_marker = ' ⚠️ also interacts' if alt['interacts_with_target'] else ' ✅ no known interaction'
                parts.append(f"- **{alt.get('name', '')}** -- shared targets: {', '.join(alt.get('shared_targets', []))}{safe_marker}")
            return '\n'.join(parts)

        # Generic fallback
        return f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"

    # =========================================================================
    # LLM Response Generation (Phase 1)
    # =========================================================================

    def _retrieve_pubmed_context(self, drugs: List[str]) -> list:
        """Fetch PubMed context for all drug pairs using multi-result retrieval."""
        try:
            from .pubmed_retriever import get_retriever
            retriever = get_retriever()
        except Exception as e:
            logger.warning("PubMed retriever unavailable: %s", e)
            return []

        max_results = getattr(settings, 'ASSISTANT_CONFIG', {}).get('max_pubmed_results', 3)
        results = []

        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                try:
                    contexts = retriever.retrieve_multiple(
                        drugs[i], drugs[j], max_results=max_results
                    )
                    results.extend(contexts)
                except Exception as e:
                    logger.warning("PubMed retrieval failed for %s + %s: %s", drugs[i], drugs[j], e)

        # Sort all results by relevance and cap at max_results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]

    def _build_rag_context(self, graph_context: dict, pubmed_results: list, drugs: list) -> str:
        """Serialize all retrieved data into structured text for LLM prompt."""
        sections = []

        # Knowledge Graph data
        sections.append("=== KNOWLEDGE GRAPH DATA ===")
        if graph_context.get('drugs'):
            for d in graph_context['drugs']:
                sections.append(
                    f"Drug: {d.get('name', 'Unknown')} "
                    f"(DrugBank: {d.get('id', 'N/A')}, "
                    f"SMILES: {(d.get('smiles', '') or '')[:60]})"
                )
        else:
            sections.append("No drugs found in Knowledge Graph for this query.")

        if graph_context.get('interactions'):
            sections.append("\n--- Known Interactions ---")
            for ix in graph_context['interactions']:
                source = ix.get('source', 'knowledge_graph')
                line = (
                    f"  {ix.get('drug_a', '?')} + {ix.get('drug_b', '?')}: "
                    f"severity={ix.get('severity', 'unknown')}, "
                    f"mechanism={ix.get('mechanism', ix.get('description', 'unknown'))}"
                )
                if ix.get('risk_score'):
                    line += f", risk_score={ix['risk_score']:.2f}"
                if ix.get('confidence'):
                    line += f", confidence={ix['confidence']:.2f}"
                if ix.get('model_version'):
                    line += f", model={ix['model_version']}"
                line += f" [source: {source}]"
                sections.append(line)
        else:
            sections.append("No known interactions found in Knowledge Graph.")

        if graph_context.get('targets'):
            sections.append("\n--- Drug Targets ---")
            for t in graph_context['targets'][:10]:
                sections.append(
                    f"  Target: {t.get('name', 'Unknown')} "
                    f"(action: {t.get('action', 'unknown')})"
                )

        # PubMed evidence
        if pubmed_results:
            sections.append("\n=== PUBMED EVIDENCE ===")
            sections.append("PubMed Evidence:")
            for idx, ctx in enumerate(pubmed_results, 1):
                pmid = getattr(ctx, 'pmid', 'unknown')
                title = getattr(ctx, 'title', 'Untitled')
                sentence = getattr(ctx, 'sentence', '')
                snippet = getattr(ctx, 'abstract_snippet', '')
                score = getattr(ctx, 'relevance_score', 0)
                line = f'[{idx}] PMID:{pmid} - "{title}" - {sentence}'
                if snippet:
                    line += f" Abstract: {snippet}..."
                sections.append(line)
                sections.append(f"  Relevance score: {score:.2f}")
        else:
            sections.append("\n=== PUBMED EVIDENCE ===")
            sections.append("No PubMed articles retrieved for this query.")

        return "\n".join(sections)

    def _generate_llm_response(
        self, message: str, graph_context: dict, pubmed_results: list, drugs: list
    ) -> Tuple[Optional[str], List[Dict], Optional[Dict]]:
        """Generate response using Gemini with RAG context."""
        from .gemini_client import get_gemini_client

        client = get_gemini_client()
        if not client.is_available():
            logger.warning("Gemini client not available, falling back to templates")
            return None, [], None

        context_text = self._build_rag_context(graph_context, pubmed_results, drugs)

        try:
            result = client.generate(message, context_text)
            if result.text and not result.text.startswith("I encountered an error"):
                token_usage = {
                    'input_tokens': result.input_tokens,
                    'output_tokens': result.output_tokens,
                    'estimated_cost_usd': result.estimated_cost_usd,
                }
                return result.text, result.citations, token_usage
            else:
                logger.warning("Gemini returned error response, falling back to templates")
                return None, [], None
        except Exception as e:
            logger.error("LLM response generation failed: %s", e)
            return None, [], None

    # =========================================================================
    # Template-Based Response Generation (Original Logic - Fallback)
    # =========================================================================

    def _extract_drug_names(self, message: str) -> List[str]:
        """Extract potential drug names from user message."""
        # Common drug names we know about
        known_drugs = [
            'warfarin', 'aspirin', 'ibuprofen', 'acetaminophen', 'metformin',
            'lisinopril', 'atorvastatin', 'simvastatin', 'omeprazole', 'metoprolol',
            'amlodipine', 'methotrexate', 'fluoxetine', 'sertraline', 'clopidogrel',
            'digoxin', 'phenytoin', 'carbamazepine', 'rifampin', 'ketoconazole',
            'erythromycin', 'clarithromycin', 'ciprofloxacin', 'amiodarone',
            'lithium', 'cyclosporine', 'tacrolimus', 'ritonavir', 'sildenafil'
        ]

        message_lower = message.lower()
        found = []

        for drug in known_drugs:
            if drug in message_lower:
                found.append(drug.capitalize())

        # Also search Knowledge Graph for any mentioned drugs
        words = re.findall(r'\b[a-zA-Z]{4,}\b', message)
        for word in words:
            if word.lower() not in [d.lower() for d in found]:
                try:
                    results = self.kg.search_drugs(word, limit=1)
                    if results and results[0].get('name', '').lower() == word.lower():
                        found.append(results[0]['name'])
                except Exception:
                    pass

        return found

    def _retrieve_graph_context(self, drugs: List[str], message: str) -> Dict:
        """Retrieve relevant context from the Knowledge Graph."""
        context = {
            'drugs': [],
            'interactions': [],
            'targets': [],
            'related_drugs': [],
            'query_type': self._classify_query(message)
        }

        if not self.kg.is_connected():
            logger.warning("Knowledge Graph not connected")
            return context

        # Retrieve drug information and build a map of name -> drugbank_id
        drug_id_map = {}
        for drug_name in drugs:
            try:
                results = self.kg.search_drugs(drug_name, limit=1)
                if results:
                    drug_info = results[0]
                    context['drugs'].append(drug_info)
                    drug_id_map[drug_name] = drug_info.get('id', '')

                    # Get drug targets
                    targets = self._get_drug_targets(drug_info.get('id', drug_name))
                    if targets:
                        context['targets'].extend(targets)
            except Exception as e:
                logger.warning(f"Failed to retrieve drug {drug_name}: {e}")

        # Check for interactions between drugs using drugbank IDs + categorical rules + prediction engine
        if len(drugs) >= 2:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    interaction_found = False
                    try:
                        # 1. Try Knowledge Graph edges first
                        drug_id_a = drug_id_map.get(drugs[i], '')
                        drug_id_b = drug_id_map.get(drugs[j], '')
                        if drug_id_a and drug_id_b:
                            interaction = self.kg.check_interaction(drug_id_a, drug_id_b)
                            if interaction:
                                interaction['drug_a'] = drugs[i]
                                interaction['drug_b'] = drugs[j]
                                interaction['source'] = 'knowledge_graph'
                                context['interactions'].append(interaction)
                                interaction_found = True
                    except Exception as e:
                        logger.warning(f"Failed to check KG interaction: {e}")

                    # 2. If no KG edge found, try the unified prediction pipeline
                    #    (categorical rules + GNN) for richer context
                    if not interaction_found:
                        try:
                            from ..views import lookup_drug, build_pair_prediction_response
                            from .gnn_predictor import get_gnn_predictor
                            drug_a_info = lookup_drug({'name': drugs[i]})
                            drug_b_info = lookup_drug({'name': drugs[j]})
                            pred = build_pair_prediction_response(
                                drug_a=drug_a_info,
                                drug_b=drug_b_info,
                                original_name_a=drugs[i],
                                original_name_b=drugs[j],
                                gnn_service=get_gnn_predictor(),
                            )
                            if pred:
                                context['interactions'].append({
                                    'drug_a': drugs[i],
                                    'drug_b': drugs[j],
                                    'severity': pred.get('severity', 'unknown'),
                                    'mechanism': pred.get('mechanism_hypothesis', ''),
                                    'risk_score': pred.get('risk_score', 0),
                                    'confidence': pred.get('confidence', 0),
                                    'source': pred.get('source', 'prediction_engine'),
                                    'model_version': pred.get('provenance', {}).get('model_version', ''),
                                })
                        except Exception as e:
                            logger.warning(f"Failed prediction fallback for {drugs[i]}+{drugs[j]}: {e}")

        # Find related drugs (drugs that share targets)
        if context['targets']:
            target_names = [t.get('name', '') for t in context['targets']]
            related = self._find_drugs_by_targets(target_names, exclude=drugs)
            context['related_drugs'] = related[:5]

        return context

    def _get_drug_targets(self, drug_id: str) -> List[Dict]:
        """Get targets for a drug from the Knowledge Graph."""
        try:
            query = """
            MATCH (d:Drug)-[r:TARGETS]->(t:Target)
            WHERE d.id = $drug_id OR d.name = $drug_id
            RETURN t.name as name, t.id as id, r.action as action
            LIMIT 10
            """
            results = self.kg.run_query(query, {'drug_id': drug_id})
            return results
        except Exception:
            return []

    def _find_drugs_by_targets(self, target_names: List[str], exclude: List[str]) -> List[str]:
        """Find drugs that target the same proteins."""
        try:
            query = """
            MATCH (d:Drug)-[:TARGETS]->(t:Target)
            WHERE t.name IN $targets AND NOT d.name IN $exclude
            RETURN DISTINCT d.name as name
            LIMIT 10
            """
            results = self.kg.run_query(query, {
                'targets': target_names,
                'exclude': [e.lower() for e in exclude]
            })
            return [r['name'] for r in results if r.get('name')]
        except Exception:
            return []

    def _classify_query(self, message: str) -> str:
        """Classify the type of query for response generation."""
        message_lower = message.lower()

        if any(word in message_lower for word in ['interact', 'combination', 'together', 'mix']):
            return 'interaction'
        elif any(word in message_lower for word in ['mechanism', 'how', 'why', 'work']):
            return 'mechanism'
        elif any(word in message_lower for word in ['side effect', 'adverse', 'risk', 'danger']):
            return 'safety'
        elif any(word in message_lower for word in ['dose', 'dosage', 'how much']):
            return 'dosing'
        elif any(word in message_lower for word in ['alternative', 'substitute', 'instead']):
            return 'alternatives'
        elif any(word in message_lower for word in ['cyp', 'enzyme', 'metabol']):
            return 'metabolism'
        elif any(word in message_lower for word in ['target', 'receptor', 'protein']):
            return 'targets'
        else:
            return 'general'

    def _generate_response(self, message: str, context: Dict, drugs: List[str]) -> str:
        """Generate a contextual response based on graph data (template fallback)."""
        query_type = context.get('query_type', 'general')

        # Interaction queries
        if query_type == 'interaction' and context.get('interactions'):
            return self._format_interaction_response(context)

        # Mechanism/target queries
        if query_type in ('mechanism', 'targets') and context.get('targets'):
            return self._format_mechanism_response(context, drugs)

        # Metabolism queries
        if query_type == 'metabolism':
            return self._format_metabolism_response(context, drugs)

        # Alternative drug queries
        if query_type == 'alternatives' and context.get('related_drugs'):
            return self._format_alternatives_response(context, drugs)

        # General drug info
        if context.get('drugs'):
            return self._format_drug_info_response(context)

        # Fallback response
        return self._fallback_response(message)

    def _format_interaction_response(self, context: Dict) -> str:
        """Format response about drug interactions."""
        interactions = context.get('interactions', [])
        if not interactions:
            return "I couldn't find any known interactions between these drugs in our database."

        response_parts = ["Based on our Knowledge Graph, here's what I found:\n"]

        for interaction in interactions:
            drug_a = interaction.get('drug_a', 'Drug A')
            drug_b = interaction.get('drug_b', 'Drug B')
            severity = interaction.get('severity', 'unknown')
            # Use mechanism if description is not available
            description = interaction.get('description') or interaction.get('mechanism') or 'Interaction mechanism under investigation'

            severity_emoji = {
                'severe': '🔴',
                'major': '🔴',
                'moderate': '🟡',
                'minor': '🟢'
            }.get(severity.lower(), '⚪')

            response_parts.append(
                f"\n**{drug_a} + {drug_b}** {severity_emoji} {severity.upper()}\n"
                f"{description}\n"
            )

        # Add clinical recommendation
        if any(i.get('severity', '').lower() in ('major', 'severe') for i in interactions):
            response_parts.append(
                "\n**Clinical Recommendation**: Severe interaction detected. "
                "Consider alternative therapy or close monitoring if co-administration is necessary."
            )

        return ''.join(response_parts)

    def _format_mechanism_response(self, context: Dict, drugs: List[str]) -> str:
        """Format response about drug mechanisms."""
        targets = context.get('targets', [])
        drug_info = context.get('drugs', [])

        response_parts = ["Here's what I found about the mechanism of action:\n"]

        for drug in drug_info:
            drug_name = drug.get('name', 'Drug')
            drug_targets = [t for t in targets if any(
                drug.get('id', '').lower() in str(t).lower() or
                drug_name.lower() in str(t).lower()
                for _ in [1]
            )]

            response_parts.append(f"\n**{drug_name}**\n")

            if drug_targets:
                response_parts.append("Targets:\n")
                for target in drug_targets[:3]:
                    target_name = target.get('name', 'Unknown target')
                    action = target.get('action', 'binds to')
                    response_parts.append(f"  - {action} {target_name}\n")
            else:
                response_parts.append("  - Target information being updated in our database\n")

        return ''.join(response_parts)

    def _format_metabolism_response(self, context: Dict, drugs: List[str]) -> str:
        """Format response about drug metabolism."""
        response = (
            "**Drug Metabolism via CYP450 Enzymes**\n\n"
            "The cytochrome P450 system metabolizes ~75% of drugs. Key interactions occur when:\n\n"
            "- **Inhibitors** (e.g., ketoconazole, ritonavir) increase drug levels\n"
            "- **Inducers** (e.g., rifampin, carbamazepine) decrease drug levels\n\n"
        )

        if drugs:
            drug_metabolism = {
                'warfarin': 'CYP2C9, CYP3A4',
                'simvastatin': 'CYP3A4',
                'atorvastatin': 'CYP3A4',
                'omeprazole': 'CYP2C19, CYP3A4',
                'clopidogrel': 'CYP2C19 (prodrug activation)',
                'fluoxetine': 'CYP2D6 (inhibitor)',
                'metoprolol': 'CYP2D6',
                'carbamazepine': 'CYP3A4 (inducer)',
                'phenytoin': 'CYP2C9, CYP2C19 (inducer)',
            }

            response += "**For your selected drugs:**\n"
            for drug in drugs:
                enzyme = drug_metabolism.get(drug.lower(), 'Various CYP enzymes')
                response += f"- {drug}: Metabolized by {enzyme}\n"

        return response

    def _format_alternatives_response(self, context: Dict, drugs: List[str]) -> str:
        """Format response about alternative drugs."""
        related = context.get('related_drugs', [])

        response = "**Alternative drugs targeting similar pathways:**\n\n"

        if related:
            for drug in related:
                response += f"- {drug}\n"
            response += (
                "\n*Note: These alternatives share similar targets but may have different "
                "interaction profiles. Always consult clinical guidelines.*"
            )
        else:
            response += (
                "I couldn't find specific alternatives in our database. "
                "Consider consulting drug formularies for therapeutic substitutes."
            )

        return response

    def _format_drug_info_response(self, context: Dict) -> str:
        """Format general drug information response."""
        drugs = context.get('drugs', [])

        response_parts = ["Here's information from our Knowledge Graph:\n"]

        for drug in drugs:
            name = drug.get('name', 'Drug')
            drugbank_id = drug.get('id', 'Unknown')
            smiles = drug.get('smiles', '')

            response_parts.append(f"\n**{name}** (DrugBank: {drugbank_id})\n")
            if smiles:
                response_parts.append(f"SMILES: `{smiles[:50]}{'...' if len(smiles) > 50 else ''}`\n")

        # Add interaction count if available
        interactions = context.get('interactions', [])
        if len(drugs) >= 2:
            if interactions:
                response_parts.append(
                    f"\nFound {len(interactions)} known interaction(s) between these drugs."
                )
            else:
                response_parts.append(
                    "\nNo known interactions found between these drugs in our database."
                )

        return ''.join(response_parts)

    def _fallback_response(self, message: str) -> str:
        """Generate fallback response when no specific context is available."""
        return (
            "I'm the **Aegis Research Assistant**, powered by a Knowledge Graph of drug interactions.\n\n"
            "I can help you with:\n"
            "- **Drug interactions** - Ask about specific drug combinations\n"
            "- **Mechanisms** - How drugs work at the molecular level\n"
            "- **Targets** - What proteins/receptors drugs affect\n"
            "- **Metabolism** - CYP450 enzymes and drug processing\n"
            "- **Alternatives** - Drugs with similar mechanisms\n\n"
            "**Slash commands** (type `/` to see all):\n"
            "- `/test warfarin aspirin` - Test a drug pair interaction\n"
            "- `/poly warfarin,aspirin,ibuprofen` - Polypharmacy analysis\n"
            "- `/evidence warfarin aspirin` - Full evidence chain\n"
            "- `/compare metformin glipizide` - Side-by-side comparison\n"
            "- `/alt ibuprofen` - Find safer alternatives\n\n"
            "Try asking: *'What is the interaction between Warfarin and Aspirin?'*"
        )

    def _compile_sources(self, context: Dict) -> List[Dict]:
        """Compile list of sources used for the response."""
        sources = []

        # Add DrugBank as primary source
        sources.append({
            'title': 'DrugBank Database',
            'url': 'https://go.drugbank.com/',
            'type': 'database'
        })

        # Add sources based on what context was used
        if context.get('interactions'):
            sources.append({
                'title': 'Project Aegis Knowledge Graph',
                'url': '#knowledge-graph',
                'type': 'knowledge_graph'
            })

        if context.get('targets'):
            sources.append({
                'title': 'UniProt Protein Database',
                'url': 'https://www.uniprot.org/',
                'type': 'database'
            })

        return sources


# Singleton instance
_chatbot_instance = None

def get_chatbot() -> GraphRAGChatbot:
    """Get the singleton chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = GraphRAGChatbot()
    return _chatbot_instance
