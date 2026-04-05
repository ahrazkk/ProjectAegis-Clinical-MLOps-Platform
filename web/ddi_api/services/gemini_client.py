"""
Gemini LLM Client for Aegis Research Assistant

Thin wrapper around Google's generativeai SDK.
Provides pharmacology-specialized system prompt and citation extraction.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from django.conf import settings

logger = logging.getLogger(__name__)

PHARMACOLOGY_SYSTEM_PROMPT = """You are the Aegis Research Assistant, a clinical pharmacology expert embedded in a drug-drug interaction prediction platform called Project Aegis.

Your knowledge domain:
- Drug-drug interactions (DDIs), mechanisms of action, CYP450 metabolism
- Polypharmacy risks, organ system toxicity, therapeutic alternatives
- Clinical pharmacology, pharmacokinetics, pharmacodynamics
- FDA adverse event reporting (FAERS), TWOSIDES polypharmacy data

Your behavior rules:
1. ONLY reference data provided in the context block. Never invent interactions, PMIDs, or statistics.
2. Every clinical claim must cite its source using these tags:
   - [KG: DrugBank] for Knowledge Graph data
   - [PMID:12345678] for PubMed literature (use the actual PMID from context)
   - [TWOSIDES] for polypharmacy signal data
   - [FAERS] for FDA adverse event reports
   - [DDI-Corpus] for literature corpus matches
   - [GNN-Model] for model prediction data
3. When evidence is insufficient, say: "Insufficient evidence for a definitive assessment. Manual review recommended."
4. When sources disagree, explicitly note the disagreement and which sources support each position.
5. Lead with the most clinically relevant finding.
6. Format responses in markdown. Use bullet points for multi-item lists.
7. When analyzing a GNN prediction, evaluate whether the predicted severity and confidence align with the literature evidence. Flag discrepancies explicitly.
8. Keep responses concise but thorough. Target 150-300 words unless the user asks for detail.
9. If a prediction has low confidence (below 0.5), prominently flag this and explain what the literature suggests instead.
10. Never provide dosing advice or treatment recommendations. You analyze interactions, not prescribe."""


@dataclass
class GeminiResponse:
    """Response from Gemini with metadata."""
    text: str
    citations: List[Dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ''
    estimated_cost_usd: float = 0.0


class GeminiClient:
    """Wrapper around Google Generative AI SDK for the Aegis Research Assistant."""

    def __init__(self):
        config = getattr(settings, 'GEMINI_CONFIG', {})
        self._api_key = config.get('api_key', '')
        self._model_name = config.get('model', 'gemini-2.5-flash')
        self._max_tokens = config.get('max_output_tokens', 1024)
        self._temperature = config.get('temperature', 0.3)
        self._top_p = config.get('top_p', 0.9)
        self._model = None

        if self._api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=PHARMACOLOGY_SYSTEM_PROMPT,
                    generation_config={
                        'max_output_tokens': self._max_tokens,
                        'temperature': self._temperature,
                        'top_p': self._top_p,
                    },
                )
                logger.info("Gemini client initialized with model: %s", self._model_name)
            except Exception as e:
                logger.error("Failed to initialize Gemini client: %s", e)
                self._model = None
        else:
            logger.warning("Gemini API key not configured. LLM features disabled.")

    def is_available(self) -> bool:
        """True if Gemini model is configured and ready."""
        return self._model is not None

    def generate(self, user_message: str, context: str) -> GeminiResponse:
        """
        Send prompt with RAG context to Gemini.

        Args:
            user_message: The user's question or command interpretation request
            context: Structured RAG context (KG data, PubMed, evidence chains)

        Returns:
            GeminiResponse with text, citations, and token usage
        """
        if not self.is_available():
            return GeminiResponse(
                text="LLM assistant is not available. Please check API key configuration.",
                model=self._model_name,
            )

        prompt = f"""{context}

=== USER QUESTION ===
{user_message}

Provide an evidence-based response with citations from the context above."""

        try:
            response = self._model.generate_content(prompt)

            # Extract token usage
            usage = getattr(response, 'usage_metadata', None)
            input_tokens = getattr(usage, 'prompt_token_count', 0) if usage else 0
            output_tokens = getattr(usage, 'candidates_token_count', 0) if usage else 0

            # Gemini 2.5 Flash pricing: ~$0.15/1M input, ~$0.60/1M output
            cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)

            text = response.text if response.text else "No response generated."
            citations = self._extract_citations(text)

            logger.info(
                "Gemini response: %d in / %d out tokens, $%.6f, %d citations",
                input_tokens, output_tokens, cost, len(citations),
            )

            return GeminiResponse(
                text=text,
                citations=citations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self._model_name,
                estimated_cost_usd=round(cost, 6),
            )

        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            return GeminiResponse(
                text=f"I encountered an error processing your request. The assistant will retry with the template engine.\n\nError: {str(e)[:200]}",
                model=self._model_name,
            )

    def _extract_citations(self, text: str) -> List[Dict]:
        """Parse citation tags from response text into structured objects."""
        citations = []
        seen = set()

        # PubMed citations: [PMID:12345678]
        for match in re.finditer(r'\[PMID:(\d+)\]', text):
            pmid = match.group(1)
            if pmid not in seen:
                seen.add(pmid)
                citations.append({
                    'type': 'pubmed',
                    'pmid': pmid,
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                    'label': f'PMID:{pmid}',
                })

        # Knowledge Graph: [KG: DrugBank] or [KG: source]
        for match in re.finditer(r'\[KG:\s*([^\]]+)\]', text):
            source = match.group(1).strip()
            key = f'kg:{source}'
            if key not in seen:
                seen.add(key)
                citations.append({
                    'type': 'knowledge_graph',
                    'source': source,
                    'label': f'KG: {source}',
                })

        # TWOSIDES
        if '[TWOSIDES]' in text and 'twosides' not in seen:
            seen.add('twosides')
            citations.append({
                'type': 'twosides',
                'source': 'TWOSIDES Database',
                'label': 'TWOSIDES',
            })

        # FAERS
        if '[FAERS]' in text and 'faers' not in seen:
            seen.add('faers')
            citations.append({
                'type': 'faers',
                'source': 'FDA FAERS',
                'label': 'FAERS',
            })

        # DDI-Corpus
        if '[DDI-Corpus]' in text and 'ddi_corpus' not in seen:
            seen.add('ddi_corpus')
            citations.append({
                'type': 'ddi_corpus',
                'source': 'DDI Literature Corpus',
                'label': 'DDI-Corpus',
            })

        # GNN-Model
        if '[GNN-Model]' in text and 'gnn_model' not in seen:
            seen.add('gnn_model')
            citations.append({
                'type': 'gnn_model',
                'source': 'Aegis GNN Predictor',
                'label': 'GNN-Model',
            })

        return citations


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Return module-level singleton GeminiClient."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
