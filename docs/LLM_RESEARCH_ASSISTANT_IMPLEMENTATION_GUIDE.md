# LLM Research Assistant - Implementation Guide

## Date: 2026-04-05
## Status: Phases 1 & 2 Complete (Core Gemini + Slash Commands)

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [Architecture Overview](#2-architecture-overview)
3. [File-by-File Change Log](#3-file-by-file-change-log)
4. [How It Works - Data Flow](#4-how-it-works---data-flow)
5. [System Prompt & RAG Strategy](#5-system-prompt--rag-strategy)
6. [Slash Commands](#6-slash-commands)
7. [Access Control](#7-access-control)
8. [Citation System](#8-citation-system)
9. [Configuration Reference](#9-configuration-reference)
10. [Setup & Activation Guide](#10-setup--activation-guide)
11. [Cost Tracking](#11-cost-tracking)
12. [Frontend Changes](#12-frontend-changes)
13. [Backward Compatibility](#13-backward-compatibility)
14. [Known Limitations](#14-known-limitations)
15. [Future Phases (Not Yet Built)](#15-future-phases-not-yet-built)

---

## 1. What Was Built

A Google Gemini-powered Research Assistant integrated into the existing Project Aegis chat system. The assistant:

- **Answers drug interaction questions** using RAG (Retrieval-Augmented Generation) grounded in your Neo4j Knowledge Graph and PubMed literature
- **Provides citations** for every clinical claim ([PMID:xxx], [KG: DrugBank], [TWOSIDES], [FAERS])
- **Executes slash commands** (`/test`, `/poly`, `/compare`, `/alt`, `/evidence`) that call your existing prediction and analysis services, then interprets results clinically
- **Falls back gracefully** to the original template engine if Gemini is unavailable or the user hasn't authenticated
- **Is password-gated** so only you (the owner) can access LLM features
- **Tracks token usage and cost** per request for budget awareness

### What Was NOT Changed

- The GNN prediction pipeline (`gnn_predictor.py`) is untouched
- The polypharmacy scorer is untouched
- The enhanced drug service is untouched (we read from it, never write)
- The Knowledge Graph schema has no new nodes yet (corrections come in Phase 3)
- All existing API contracts are backward-compatible -- old frontends still work
- No existing tests were broken

---

## 2. Architecture Overview

```
User types in Chat
        |
        v
  Dashboard.jsx
  (reads assistant prefs from localStorage)
        |
        v
  api.js sendChatMessage()
  (adds assistant_mode + access_token to request)
        |
        v
  POST /api/v1/chat/
        |
        v
  ChatView (views.py)
  - Validates access_token against AEGIS_ASSISTANT_PASSWORD
  - Sets chatbot.use_llm = true/false
        |
        v
  GraphRAGChatbot.process_message()
        |
        +-- Is it a slash command? (/test, /poly, etc.)
        |       |
        |       v
        |   CommandRouter.parse() -> CommandRouter.execute()
        |   (calls GNNPredictor, EnhancedDrugService, etc. directly)
        |       |
        |       v
        |   If LLM enabled: GeminiClient interprets tool results
        |   If LLM disabled: Format as structured text
        |
        +-- Is it natural language?
                |
                v
            Step 1: Extract drug names (keyword + KG search)
            Step 2: Retrieve KG context (drugs, interactions, targets)
            Step 3: Retrieve PubMed context (abstracts, scored sentences)
            Step 4: If LLM enabled:
                      - Build RAG context string
                      - Send to Gemini with pharmacology system prompt
                      - Extract citations from response
                    If LLM disabled:
                      - Use template response (original logic)
            Step 5: Return response + citations + sources
```

---

## 3. File-by-File Change Log

### New Files Created

#### `web/ddi_api/services/gemini_client.py`
- **Purpose:** Thin wrapper around Google's `google-generativeai` SDK
- **Key class:** `GeminiClient`
  - `__init__()` - Configures API key, model, temperature from Django settings
  - `generate(user_message, context)` - Sends prompt with RAG context to Gemini
  - `is_available()` - Returns True if API key is configured
  - `_extract_citations(text)` - Parses `[PMID:xxx]`, `[KG: xxx]`, `[TWOSIDES]`, `[FAERS]`, `[DDI-Corpus]`, `[GNN-Model]` tags from response
- **Key constant:** `PHARMACOLOGY_SYSTEM_PROMPT` - 10-rule system instruction that makes Gemini behave as a clinical pharmacology expert
- **Singleton:** `get_gemini_client()` returns module-level instance
- **Returns:** `GeminiResponse` dataclass with `text`, `citations`, `input_tokens`, `output_tokens`, `model`, `estimated_cost_usd`

#### `web/ddi_api/services/command_router.py`
- **Purpose:** Parses slash commands and routes to existing backend services
- **Key class:** `CommandRouter`
  - `parse(message)` - Detects `/command arg1 arg2` or `/command arg1,arg2` syntax
  - `execute(parsed)` - Routes to handler method, returns structured dict
  - `_handle_test(drugs)` - Calls `GNNPredictor.predict()` for pairwise DDI
  - `_handle_poly(drugs)` - Calls `GNNPredictor.predict_polypharmacy()`
  - `_handle_compare(drugs)` - Calls `EnhancedDrugService.get_drug_info()` for each drug
  - `_handle_alt(drugs)` - Queries KG for drugs sharing protein targets, checks if alternatives also interact
  - `_handle_evidence(drugs)` - Calls `EnhancedDrugService.get_interaction_info()` for full evidence chain
- **Key constant:** `COMMANDS` dict defining min/max args and descriptions per command
- **Helper:** `get_command_list()` returns command metadata for frontend autocomplete

#### `src/components/ChatCommandAutocomplete.jsx`
- **Purpose:** Dropdown autocomplete for slash commands and drug names
- **Behavior:**
  - When input is `/` or `/partial`: Shows filtered command list with descriptions and usage examples
  - When input is `/command drugnam`: Searches drugs via existing `/api/v1/search/` endpoint (debounced 200ms)
  - Keyboard navigation: Arrow keys, Tab/Enter to select, Escape to dismiss
  - Mouse: Click to select, hover to highlight
- **Props:** `inputValue`, `onSelect(text, type)`, `visible`
- **Hardcoded commands list** (5 commands with descriptions and usage examples)

### Modified Files

#### `web/requirements.txt`
- **Added:** `google-generativeai>=0.8.0`

#### `web/.env`
- **Added:**
  ```
  GEMINI_API_KEY=AIza...  (your API key)
  AEGIS_ASSISTANT_ENABLED=true
  AEGIS_ASSISTANT_PASSWORD=aegis-owner-2026
  ```

#### `web/ProjectAegis/settings.py`
- **Added after `AI_MODEL_CONFIG`:**
  - `GEMINI_CONFIG` dict: `api_key`, `model` (gemini-2.5-flash), `max_output_tokens` (1024), `temperature` (0.3), `top_p` (0.9)
  - `ASSISTANT_CONFIG` dict: `enabled` (from env), `access_password` (from env), `max_context_tokens` (4000), `max_pubmed_results` (3)

#### `web/ddi_api/serializers.py`
- **ChatRequestSerializer:** Added optional fields `assistant_mode` (choices: auto/llm/legacy, default: auto) and `access_token` (string, default: '')
- **ChatResponseSerializer:** Added optional fields `citations` (list of dicts), `assistant_mode` (string), `model_used` (string)
- All new fields have defaults, so existing clients work unchanged

#### `web/ddi_api/views.py`
- **ChatView.post():** Complete rewrite of the method:
  - Reads `assistant_mode` and `access_token` from request
  - Checks `ASSISTANT_CONFIG['enabled']` and validates password
  - Sets `chatbot.use_llm = True/False` based on access check
  - If `assistant_mode='llm'` and password is wrong: returns 403
  - If `assistant_mode='auto'` and password is wrong: silently falls back to templates
  - Response now includes `citations`, `assistant_mode`, `model_used` fields
- **New class: `AssistantCommandsView`:**
  - `GET /api/v1/assistant/commands/` - Returns list of available slash commands for frontend autocomplete

#### `web/ddi_api/urls.py`
- **Added import:** `AssistantCommandsView`
- **Added route:** `path('assistant/commands/', AssistantCommandsView.as_view())`

#### `web/ddi_api/services/graphrag_chatbot.py`
- **Full rewrite** preserving all original template logic. Changes:
  - `ChatResponse` dataclass: Added `citations` field (list of dicts)
  - `__init__()`: Added `self.use_llm = False` flag and lazy `_command_router`
  - `process_message()`: Now checks for slash commands first, then routes to LLM or template path
  - **New methods for LLM path:**
    - `_retrieve_pubmed_context(drugs)` - Fetches PubMed abstracts for all drug pairs using existing `PubMedRetriever`
    - `_build_rag_context(graph_context, pubmed_results, drugs)` - Serializes KG data + PubMed results into structured text for Gemini prompt
    - `_generate_llm_response(message, graph_context, pubmed_results, drugs)` - Calls `GeminiClient.generate()` with RAG context
  - **New methods for command handling:**
    - `_handle_command(parsed, context_drugs)` - Executes command, optionally interprets with LLM
    - `_interpret_tool_result(original_message, tool_result)` - Sends tool output to Gemini for clinical interpretation
    - `_format_tool_result(result)` - Formats tool output as readable text (non-LLM fallback)
  - **All original template methods preserved unchanged:**
    - `_extract_drug_names()`, `_retrieve_graph_context()`, `_classify_query()`, `_generate_response()`, `_format_interaction_response()`, `_format_mechanism_response()`, `_format_metabolism_response()`, `_format_alternatives_response()`, `_format_drug_info_response()`, `_fallback_response()`, `_compile_sources()`

#### `src/services/api.js`
- **`sendChatMessage()`:** Added 4th parameter `options = {}` with `assistantMode` and `accessToken`. These are sent as `assistant_mode` and `access_token` in the request body. Existing callers without options still work (defaults to auto mode, empty token).
- **New function:** `getAssistantCommands()` - `GET /assistant/commands/`
- **Updated type definition:** `ChatResponse` typedef now includes `citations`, `assistant_mode`, `model_used`
- **Updated default export** to include `getAssistantCommands`

#### `src/pages/Dashboard.jsx`
- **Import:** Added `ChatCommandAutocomplete` component
- **State:** Added `showAutocomplete` state variable
- **`handleChatSubmit()`:**
  - Reads assistant preferences from `localStorage` key `aegis:assistant-prefs:v1`
  - Passes `assistantMode` and `accessToken` to `sendChatMessage()`
  - Stores `citations` and `assistantMode` in message objects
  - Log message shows "Gemini LLM" or "GraphRAG" based on response mode
- **Desktop chat panel:**
  - Removed "Under Construction" badge
  - Added LLM mode indicator badge ("LLM Active" green or "Template" blue)
  - Added "Type / for commands" hint in empty state
  - Added citation badges below assistant messages (clickable links to PubMed, KG sources)
  - Added legacy sources fallback for non-LLM responses
  - Added "via Gemini LLM" subtle indicator on LLM messages
  - Added `ChatCommandAutocomplete` component above input field
  - Input `onChange` triggers autocomplete when text starts with `/`
  - Input `onBlur` dismisses autocomplete after 200ms delay (allows click)
  - Increased panel height from `h-80` to `h-96`
  - Updated placeholder text: "Ask about interactions or type / for commands..."
- **Mobile chat panel:**
  - Removed "Under Construction" banner, replaced with mode indicator bar
  - Added citation badges and "via Gemini LLM" indicator (same as desktop)
  - Added "Type / for commands" hint in empty state

#### `src/pages/SettingsPage.jsx`
- **Import:** Added `Sparkles` icon
- **State:** Added `assistantMode` and `assistantToken` state variables (read from `localStorage`)
- **Sidebar nav:** Added 4th tab button "Research Assistant" with Sparkles icon
- **`handleSaveConfiguration()`:** Now also saves assistant preferences to `localStorage` key `aegis:assistant-prefs:v1`
- **New tab content (`activeTab === 'assistant'`):**
  - Mode selector: 3-button toggle for Auto / LLM Only / Template
  - Access token: Password input field with green "Token configured" indicator
  - Info box: Explains how the feature works, what slash commands exist, and how fallback works

---

## 4. How It Works - Data Flow

### Natural Language Query (e.g., "What is the interaction between warfarin and aspirin?")

```
1. Frontend reads localStorage for assistant prefs (mode, accessToken)
2. POST /api/v1/chat/ with {message, context_drugs, assistant_mode, access_token}
3. ChatView validates access_token against AEGIS_ASSISTANT_PASSWORD env var
4. If valid: chatbot.use_llm = True
5. GraphRAGChatbot.process_message():
   a. _extract_drug_names("warfarin", "aspirin") via keyword matching + KG search
   b. _retrieve_graph_context() - queries Neo4j for:
      - Drug info (DrugBank IDs, SMILES)
      - Known interactions (severity, mechanism)
      - Drug targets (proteins, receptors)
      - Related drugs (shared targets)
   c. _retrieve_pubmed_context() - for each drug pair:
      - PubMedRetriever searches NCBI for abstracts
      - Extracts and scores relevant sentences
      - Returns top 3 results with PMIDs
   d. _build_rag_context() - formats everything as structured text:
      === KNOWLEDGE GRAPH DATA ===
      Drug: Warfarin (DrugBank: DB00682)
      Interaction: severity=major, mechanism=...
      === PUBMED EVIDENCE ===
      [PMID:12345678] "Title..." Key finding: "sentence..."
   e. GeminiClient.generate() - sends RAG context + question to Gemini
   f. Gemini responds with citations like [PMID:12345678], [KG: DrugBank]
   g. _extract_citations() parses citation tags into structured objects
6. Response returned: {response, sources, citations, assistant_mode: 'llm', model_used: 'gemini-2.5-flash'}
7. Frontend renders markdown response + clickable citation badges
```

### Slash Command (e.g., "/test warfarin aspirin")

```
1. CommandRouter.parse() detects slash command
2. CommandRouter.execute() calls _handle_test(['warfarin', 'aspirin'])
3. _handle_test imports GNNPredictor and calls predict('warfarin', 'aspirin')
4. Returns structured result: {risk_score, confidence, severity, mechanism, model_used, ...}
5. If LLM enabled:
   - _interpret_tool_result() sends result JSON to Gemini
   - Gemini provides clinical interpretation with citations
6. If LLM disabled:
   - _format_tool_result() formats as readable markdown text
7. Frontend renders response (with citations if LLM was used)
```

### Fallback Behavior

```
Scenario 1: No access token provided, mode='auto'
  -> chatbot.use_llm = False
  -> Uses original template engine (unchanged behavior)

Scenario 2: Wrong access token, mode='auto'
  -> chatbot.use_llm = False
  -> Silent fallback to templates

Scenario 3: Wrong access token, mode='llm'
  -> Returns HTTP 403 Forbidden

Scenario 4: Gemini API error (rate limit, network, etc.)
  -> _generate_llm_response returns None
  -> Falls back to template engine
  -> Logs error for debugging

Scenario 5: AEGIS_ASSISTANT_ENABLED=false
  -> LLM path is never activated regardless of token
  -> All requests use template engine
```

---

## 5. System Prompt & RAG Strategy

### Why RAG Instead of Fine-Tuning

Fine-tuning a model on pharmacology data sounds appealing but has major downsides:
- **Expensive** to create training datasets and train
- **Stale** the moment new drug data enters your KG
- **Hallucination risk** increases when the model "knows" things vs. referencing context
- **Not needed** when your data sources (KG, PubMed, FAERS) are comprehensive

Our RAG approach:
1. Retrieve relevant facts from your existing data sources
2. Inject them as context into the prompt
3. Instruct the model to ONLY reference provided context
4. Extract citations to verify grounding

This gives you **up-to-date, citation-backed answers** without any training cost.

### System Prompt Design

The system prompt (in `gemini_client.py`) has 10 rules:

1. **ONLY reference provided context** - prevents hallucination
2. **Every clinical claim must cite** - enforces accountability
3. **"Insufficient evidence" when unsure** - prevents confident wrong answers
4. **Note source disagreements** - transparency when data conflicts
5. **Lead with most relevant finding** - clinical utility
6. **Markdown formatting** - readability
7. **Flag GNN discrepancies** - catches low-confidence predictions
8. **150-300 word target** - concise but thorough
9. **Flag low confidence (<0.5)** - highlights unreliable predictions
10. **Never provide dosing advice** - safety guardrail

### Context Budget

- `max_context_tokens: 4000` in settings
- This fits: ~3 drug descriptions + ~3 PubMed abstracts + interaction data
- Gemini 2.5 Flash context window is 1M tokens, so 4K is a tiny fraction
- Keeps cost per query very low (~$0.001-0.003)

---

## 6. Slash Commands

| Command | Args | What It Does | Service Called |
|---------|------|-------------|----------------|
| `/test warfarin aspirin` | 2 drugs | Pairwise DDI prediction | `GNNPredictor.predict()` |
| `/poly warfarin,aspirin,ibuprofen` | 2-10 drugs | N-way polypharmacy risk | `GNNPredictor.predict_polypharmacy()` |
| `/compare metformin glipizide` | 2-5 drugs | Side-by-side drug info | `EnhancedDrugService.get_drug_info()` |
| `/alt ibuprofen` | 1-2 drugs | Safer alternatives via shared targets | `KnowledgeGraphService` Cypher query |
| `/evidence warfarin aspirin` | 2 drugs | Full evidence chain with sources | `EnhancedDrugService.get_interaction_info()` |

### Command Parsing

- Commands must start with `/`
- Args can be space-separated (`/test warfarin aspirin`) or comma-separated (`/poly warfarin,aspirin,ibuprofen`)
- Drug names are lowercased for matching
- Unknown commands return a help message listing all available commands
- Too few args returns an error with usage example

### With LLM vs Without LLM

- **With LLM:** Tool result is sent to Gemini for clinical interpretation. Response includes natural language analysis + citations.
- **Without LLM:** Tool result is formatted as structured markdown (severity, confidence, risk score, etc.).

---

## 7. Access Control

### How It Works

1. Password is stored in `web/.env` as `AEGIS_ASSISTANT_PASSWORD=aegis-owner-2026`
2. Django reads it into `settings.ASSISTANT_CONFIG['access_password']`
3. User enters the password once in Settings > Research Assistant > Access Token
4. Frontend stores it in `localStorage` key `aegis:assistant-prefs:v1`
5. Every chat request sends `access_token` in the POST body
6. `ChatView` compares `access_token == ASSISTANT_CONFIG['access_password']`
7. If match: `chatbot.use_llm = True`; if not: fallback to templates (or 403 if mode='llm')

### Security Notes

- Password is stored in `.env` (server-side, never committed to git)
- Password is stored in `localStorage` (client-side, browser only)
- Password travels in POST body over HTTPS
- This is adequate for a single-owner system
- For multi-user production: upgrade to IAP or OAuth (Phase 3+)

### Changing the Password

1. Edit `web/.env`: change `AEGIS_ASSISTANT_PASSWORD=your-new-password`
2. Restart Django server
3. Update the token in Settings > Research Assistant > Access Token
4. Click Save Configuration

### Kill Switch

Set `AEGIS_ASSISTANT_ENABLED=false` in `.env` and restart. All LLM features are disabled regardless of token.

---

## 8. Citation System

### Citation Tags

The system prompt instructs Gemini to use these citation tags:

| Tag | Source | Example |
|-----|--------|---------|
| `[PMID:12345678]` | PubMed article | Links to pubmed.ncbi.nlm.nih.gov |
| `[KG: DrugBank]` | Knowledge Graph | Your Neo4j data |
| `[TWOSIDES]` | TWOSIDES database | Polypharmacy signals |
| `[FAERS]` | FDA FAERS | Adverse event reports |
| `[DDI-Corpus]` | DDI literature corpus | Literature matches |
| `[GNN-Model]` | Aegis GNN predictor | Model prediction data |

### Citation Extraction

`GeminiClient._extract_citations()` parses the response text using regex:
- Extracts PMIDs and creates clickable PubMed URLs
- Deduplicates citations (each source appears once)
- Returns structured objects: `{type, pmid/source, url, label}`

### Frontend Display

Citations appear as clickable badges below each assistant message:
- PubMed citations: blue badges linking to pubmed.ncbi.nlm.nih.gov
- Other sources: blue badges with source label
- Legacy sources (template mode): shown as text list

---

## 9. Configuration Reference

### Environment Variables (`web/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (empty) | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model name override |
| `AEGIS_ASSISTANT_ENABLED` | `false` | Master switch for LLM features |
| `AEGIS_ASSISTANT_PASSWORD` | (empty) | Access password for LLM mode |

### Django Settings (`settings.py`)

```python
GEMINI_CONFIG = {
    'api_key': '...',          # From env
    'model': 'gemini-2.5-flash',
    'max_output_tokens': 1024, # Max response length
    'temperature': 0.3,        # Low for clinical accuracy
    'top_p': 0.9,
}

ASSISTANT_CONFIG = {
    'enabled': True/False,     # From env
    'access_password': '...',  # From env
    'max_context_tokens': 4000,# Budget control for RAG context
    'max_pubmed_results': 3,   # Max PubMed abstracts per query
}
```

### Frontend LocalStorage

| Key | Format | Description |
|-----|--------|-------------|
| `aegis:assistant-prefs:v1` | `{mode, accessToken, updatedAt}` | Assistant preferences set in Settings page |

---

## 10. Setup & Activation Guide

### Step 1: Install Python Dependency

```bash
cd web
pip install google-generativeai>=0.8.0
```

### Step 2: Get a Gemini API Key (if you don't have one)

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

### Step 3: Configure Environment

Edit `web/.env`:
```
GEMINI_API_KEY=AIza...your-key
AEGIS_ASSISTANT_ENABLED=true
AEGIS_ASSISTANT_PASSWORD=your-secret-password
```

### Step 4: Restart Django Server

```bash
cd web
python manage.py runserver
```

### Step 5: Configure Frontend

1. Open the app in your browser
2. Go to Settings > Research Assistant
3. Select "Auto" mode
4. Enter the password you set in Step 3
5. Click "Save Configuration"

### Step 6: Test It

1. Go to Dashboard
2. You should see "LLM Active" badge on the Research Assistant panel
3. Type: "What is the interaction between warfarin and aspirin?"
4. You should get a Gemini-powered response with citations
5. Type: `/test warfarin aspirin`
6. You should get a GNN prediction interpreted by Gemini

---

## 11. Cost Tracking

### Per-Request Logging

Every Gemini call logs:
```
Gemini response: 2340 in / 456 out tokens, $0.000625, 3 citations
```

### Estimated Costs (Gemini 2.5 Flash)

| Query Type | Input Tokens | Output Tokens | Cost |
|-----------|-------------|--------------|------|
| Simple question | ~1,500 | ~300 | ~$0.0004 |
| With PubMed context | ~3,000 | ~500 | ~$0.0008 |
| Command interpretation | ~2,000 | ~400 | ~$0.0005 |

At $0.001/query average, your $300 GCP credits support **~300,000 queries**.

### Free Tier Limits

Google AI Studio free tier provides:
- 15 requests per minute
- 1 million tokens per day
- More than enough for development and personal use

---

## 12. Frontend Changes

### Desktop Chat Panel

| Before | After |
|--------|-------|
| "Under Construction" badge | LLM mode indicator (green "LLM Active" or blue "Template") |
| Plain text messages | Markdown-rendered messages with citation badges |
| No command support | Autocomplete dropdown on `/` keystroke |
| Sources as text | Clickable citation links to PubMed |
| Height: h-80 | Height: h-96 |
| Generic placeholder | "Ask about interactions or type / for commands..." |

### Mobile Chat Panel

| Before | After |
|--------|-------|
| "Under Construction" warning banner | Compact mode indicator bar |
| Plain text messages | Messages with citation badges |
| No "via LLM" indicator | "via Gemini LLM" label on LLM responses |

### Settings Page

| Before | After |
|--------|-------|
| 3 tabs (Appearance, Engine, Privacy) | 4 tabs (+ Research Assistant) |
| No assistant config | Mode selector, password input, info panel |

---

## 13. Backward Compatibility

### API Compatibility

The `/api/v1/chat/` endpoint is fully backward-compatible:

```javascript
// Old call (still works):
sendChatMessage("hello", ["warfarin"], sessionId)

// New call (with assistant options):
sendChatMessage("hello", ["warfarin"], sessionId, {
    assistantMode: 'auto',
    accessToken: 'password'
})
```

New response fields (`citations`, `assistant_mode`, `model_used`) are added alongside existing fields. Old frontends ignore them.

### Feature Flag

Set `AEGIS_ASSISTANT_ENABLED=false` in `.env` to completely disable LLM features. The system behaves exactly as before.

---

## 14. Known Limitations

1. **No conversation memory** - Each query is independent. Gemini doesn't remember previous messages in the session. (Fix: add conversation history to context in Phase 4)

2. **Citation accuracy** - Gemini is instructed to only cite provided PMIDs, but may occasionally format citations incorrectly. The citation extractor validates format but doesn't verify PMID existence.

3. **Drug name extraction** - Uses a hardcoded list of 29 common drugs + KG search fallback. Rare or brand-name drugs may not be extracted from natural language (but work fine in slash commands).

4. **No streaming** - Responses arrive all at once after Gemini finishes generating. No "typing" indicator beyond the spinner.

5. **PubMed rate limiting** - NCBI allows 3 requests/sec without API key. With multiple drug pairs, PubMed retrieval can take 2-5 seconds.

6. **Single-user access** - Password is shared, not per-user. Adequate for owner-only access.

---

## 15. Future Phases (Not Yet Built)

### Phase 3: Correction Memory + Low-Confidence Flagging
- Neo4j `:Correction` nodes for storing LLM-proposed corrections
- Auto-flag when GNN confidence < 0.5
- Approve/reject corrections from chat UI
- Approved corrections surface in future queries

### Phase 4: Settings Polish + Session Persistence
- Chat session persistence (survives page navigation)
- Cumulative cost tracker in header
- Conversation history in Gemini context
- Clear chat history from Settings

### Phase 5+: Advanced Features
- `/gnngalaxy`, `/knowledgegraph`, `/bodymap` commands
- Section-native critique/fix/approve workflows
- Verified response caching with invalidation
- SSE streaming for real-time response display
- Multi-reviewer correction workflows
