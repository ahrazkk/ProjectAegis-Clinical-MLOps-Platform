"""
GraphRAG Chatbot Service

Implements a Retrieval-Augmented Generation (RAG) system using:
1. Neo4j Knowledge Graph for drug/interaction data retrieval
2. LangChain for orchestration and prompt management
3. Context-aware responses about drug interactions
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response from the GraphRAG chatbot."""
    response: str
    sources: List[Dict]
    related_drugs: List[str]
    graph_context: Optional[Dict] = None


class GraphRAGChatbot:
    """
    GraphRAG-powered research assistant for drug interactions.
    
    Uses Neo4j Knowledge Graph to retrieve relevant drug information
    and generates contextual responses.
    """
    
    def __init__(self):
        self.kg = KnowledgeGraphService
        
    def process_message(
        self, 
        message: str, 
        context_drugs: List[str] = None,
        session_id: str = None
    ) -> ChatResponse:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's question
            context_drugs: List of drug names in current context
            session_id: Chat session identifier
            
        Returns:
            ChatResponse with answer, sources, and related drugs
        """
        context_drugs = context_drugs or []
        
        # Step 1: Extract drug names from the message
        mentioned_drugs = self._extract_drug_names(message)
        all_drugs = list(set(context_drugs + mentioned_drugs))
        
        # Step 2: Retrieve relevant context from Knowledge Graph
        graph_context = self._retrieve_graph_context(all_drugs, message)
        
        # Step 3: Generate response based on context
        response_text = self._generate_response(message, graph_context, all_drugs)
        
        # Step 4: Compile sources
        sources = self._compile_sources(graph_context)
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            related_drugs=graph_context.get('related_drugs', []),
            graph_context=graph_context
        )
    
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
        
        # Check for interactions between drugs using drugbank IDs
        if len(drugs) >= 2:
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    try:
                        # Use drugbank IDs instead of names for interaction lookup
                        drug_id_a = drug_id_map.get(drugs[i], '')
                        drug_id_b = drug_id_map.get(drugs[j], '')
                        if drug_id_a and drug_id_b:
                            interaction = self.kg.check_interaction(drug_id_a, drug_id_b)
                            if interaction:
                                # Add drug names to the interaction for display
                                interaction['drug_a'] = drugs[i]
                                interaction['drug_b'] = drugs[j]
                                context['interactions'].append(interaction)
                    except Exception as e:
                        logger.warning(f"Failed to check interaction: {e}")
        
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
        """Generate a contextual response based on graph data."""
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
                "\n⚠️ **Clinical Recommendation**: Severe interaction detected. "
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
                    response_parts.append(f"  • {action} {target_name}\n")
            else:
                response_parts.append("  • Target information being updated in our database\n")
        
        return ''.join(response_parts)
    
    def _format_metabolism_response(self, context: Dict, drugs: List[str]) -> str:
        """Format response about drug metabolism."""
        response = (
            "**Drug Metabolism via CYP450 Enzymes**\n\n"
            "The cytochrome P450 system metabolizes ~75% of drugs. Key interactions occur when:\n\n"
            "• **Inhibitors** (e.g., ketoconazole, ritonavir) ↑ increase drug levels\n"
            "• **Inducers** (e.g., rifampin, carbamazepine) ↓ decrease drug levels\n\n"
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
                response += f"• {drug}: Metabolized by {enzyme}\n"
        
        return response
    
    def _format_alternatives_response(self, context: Dict, drugs: List[str]) -> str:
        """Format response about alternative drugs."""
        related = context.get('related_drugs', [])
        
        response = "**Alternative drugs targeting similar pathways:**\n\n"
        
        if related:
            for drug in related:
                response += f"• {drug}\n"
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
                    f"\n📊 Found {len(interactions)} known interaction(s) between these drugs."
                )
            else:
                response_parts.append(
                    "\n✅ No known interactions found between these drugs in our database."
                )
        
        return ''.join(response_parts)
    
    def _fallback_response(self, message: str) -> str:
        """Generate fallback response when no specific context is available."""
        return (
            "I'm the **Aegis Research Assistant**, powered by a Knowledge Graph of drug interactions.\n\n"
            "I can help you with:\n"
            "• 💊 **Drug interactions** - Ask about specific drug combinations\n"
            "• 🔬 **Mechanisms** - How drugs work at the molecular level\n"
            "• 🧬 **Targets** - What proteins/receptors drugs affect\n"
            "• ⚗️ **Metabolism** - CYP450 enzymes and drug processing\n"
            "• 🔄 **Alternatives** - Drugs with similar mechanisms\n\n"
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
