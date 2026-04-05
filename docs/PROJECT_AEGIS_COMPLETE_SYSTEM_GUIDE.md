# Project Aegis - Complete System Guide

## Date: 2026-04-05
## Covers: Phases 1-4 (Gemini LLM + Slash Commands + Correction Memory + Admin Page)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Phase 1: Gemini LLM Integration](#3-phase-1-gemini-llm-integration)
4. [Phase 2: Slash Commands](#4-phase-2-slash-commands)
5. [Phase 3: Correction Memory + Low-Confidence Flagging](#5-phase-3-correction-memory--low-confidence-flagging)
6. [Phase 4: Corrections Admin Page + Auto-Capture + AI Review](#6-phase-4-corrections-admin-page)
7. [Token Usage Tracking](#7-token-usage-tracking)
8. [GNN Feedback Loop](#8-gnn-feedback-loop)
9. [Access Control & Passwords](#9-access-control--passwords)
10. [Complete File Reference](#10-complete-file-reference)
11. [API Endpoints Reference](#11-api-endpoints-reference)
12. [Configuration Reference](#12-configuration-reference)
13. [Setup & Activation Guide](#13-setup--activation-guide)
14. [Cost Analysis](#14-cost-analysis)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)

---

## 1. System Overview

Project Aegis is a Drug-Drug Interaction (DDI) prediction platform. The LLM Research Assistant adds an intelligent layer on top of the existing GNN prediction engine:

- **Gemini 2.5 Flash** powers a pharmacology-specialized chatbot
- **Citation-first**: Every clinical claim must reference its source (KG, PubMed, FAERS, etc.)
- **Slash commands**: `/test`, `/poly`, `/compare`, `/alt`, `/evidence` for deterministic tool workflows
- **Correction Memory**: Low-confidence predictions are auto-captured and stored in Neo4j for review
- **Admin Page**: Password-protected page at `/corrections` to review, approve/reject, and AI-review corrections
- **GNN Feedback Loop**: Approved corrections can be exported as training data to improve the GNN over time

### How the Pieces Connect

```
User Query
    |
    v
[Dashboard Chat Input]
    |
    +--> Slash command? --> [CommandRouter] --> [GNN/KG Services] --> [Gemini interprets results]
    |                                                                       |
    +--> Natural language? --> [Extract drugs] --> [KG Context + PubMed] --> [Gemini generates response]
    |                                                                       |
    v                                                                       v
[Response with citations]  <--  [Check for approved corrections]  <--  [Neo4j :Correction nodes]
    |
    v
[Token usage tracked (local + global)]
    |
    v
[Low confidence? Auto-capture to corrections queue]
```

---

## 2. Architecture

### Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | React + Tailwind CSS + Framer Motion |
| LLM | Google Gemini 2.5 Flash via `google-generativeai` SDK |
| Backend | Django REST Framework |
| Knowledge Graph | Neo4j Aura (cloud) |
| Database | SQLite (Django models for stats/logs) |
| Deployment | Docker + docker-compose |

### Key Design Principles
1. **Additive, not destructive**: Corrections never modify original drug data in Neo4j. They are a separate `:Correction` node that overlays the GNN prediction.
2. **Backward compatible**: All new API fields have defaults. Existing frontends work unchanged.
3. **Budget conscious**: Gemini 2.5 Flash at $0.15/1M input tokens. $300 credits support ~300k+ queries.
4. **Single-user optimized**: Simple password-based access control (no role-based auth).
5. **Fail-safe**: If Gemini is unavailable, the system falls back to the original template-based responses.

---

## 3. Phase 1: Gemini LLM Integration

### What It Does
Replaces the hardcoded template-based chat responses with Gemini-powered, evidence-grounded answers. The LLM receives RAG context (Knowledge Graph data + PubMed results) and generates clinical assessments with citations.

### System Prompt
Located in `web/ddi_api/services/gemini_client.py`, the `PHARMACOLOGY_SYSTEM_PROMPT` defines 10 behavioral rules:

1. Only reference data from the context block (no hallucination)
2. Every clinical claim must cite its source with specific tags
3. When evidence is insufficient, say so explicitly
4. When sources disagree, note the disagreement
5. Lead with the most clinically relevant finding
6. Format responses in markdown
7. Evaluate whether GNN predictions align with literature
8. Target 150-300 words unless detail is requested
9. Flag low-confidence predictions (< 0.5) prominently
10. Never provide dosing advice

### Citation Tags
| Tag | Source |
|-----|--------|
| `[KG: DrugBank]` | Neo4j Knowledge Graph |
| `[PMID:12345678]` | PubMed literature |
| `[TWOSIDES]` | TWOSIDES polypharmacy database |
| `[FAERS]` | FDA adverse event reports |
| `[DDI-Corpus]` | DDI literature corpus |
| `[GNN-Model]` | Aegis GNN predictor |

### Data Flow for Natural Language Queries
1. User sends message via chat
2. `GraphRAGChatbot.process_message()` extracts drug names
3. KG context retrieved from Neo4j (drug info, interactions, targets)
4. PubMed context retrieved for relevant drug pairs
5. RAG context assembled and sent to Gemini with the system prompt
6. Gemini response parsed for citations
7. Response returned with text + citations + token usage

### Key File: `web/ddi_api/services/gemini_client.py`
- `GeminiClient` class: wraps the Google generativeai SDK
- `generate(user_message, context)` -> `GeminiResponse`
- `_extract_citations(text)` -> parses citation tags from response
- `GeminiResponse` dataclass: `text, citations, input_tokens, output_tokens, model, estimated_cost_usd`
- Singleton via `get_gemini_client()`
- Model: `gemini-2.5-flash`, temperature: 0.3, max_output_tokens: 4096

---

## 4. Phase 2: Slash Commands

### Available Commands

| Command | Description | Example |
|---------|------------|---------|
| `/test drug1 drug2` | Predict interaction between 2 drugs | `/test Warfarin Aspirin` |
| `/poly drug1,drug2,drug3` | N-way polypharmacy analysis | `/poly warfarin,aspirin,ibuprofen` |
| `/compare drug1 drug2` | Side-by-side drug comparison | `/compare metformin glipizide` |
| `/alt drug` | Find drugs sharing same targets | `/alt ibuprofen` |
| `/evidence drug1 drug2` | Full evidence chain (FAERS, severity) | `/evidence warfarin aspirin` |

### How Commands Work
1. `CommandRouter.parse(message)` checks if input starts with `/`
2. Parses command name and arguments (space or comma separated)
3. `CommandRouter.execute(parsed)` calls the appropriate Python service directly (no internal HTTP)
4. Tool result is either:
   - Sent to Gemini for clinical interpretation (if LLM mode is active)
   - Formatted as structured text (template fallback)

### Key File: `web/ddi_api/services/command_router.py`
- `ParsedCommand` dataclass: `command, args, raw, is_command`
- 5 command handlers that call existing services (GNNPredictor, EnhancedDrugService, KnowledgeGraphService)
- `get_command_list()` helper for frontend autocomplete API

### Frontend Autocomplete: `src/components/ChatCommandAutocomplete.jsx`
- Dropdown appears when input starts with `/`
- Two modes: command list (filtered by typing) and drug search (200ms debounce via API)
- Keyboard navigation: arrow keys, Tab/Enter to select, Escape to dismiss

---

## 5. Phase 3: Correction Memory + Low-Confidence Flagging

### What It Does
When the GNN predicts an interaction with low confidence (< 0.5), or when a user manually submits a correction, a `:Correction` node is created in Neo4j. On future queries for the same drug pair, the chatbot checks for approved corrections and presents both the GNN prediction and the correction.

### Neo4j Correction Node Schema
```
(:Correction {
    id: UUID,
    drug_a: string,           // Alphabetically sorted
    drug_b: string,
    gnn_severity: string,     // Original GNN prediction
    gnn_risk_score: float,
    gnn_confidence: float,
    corrected_severity: string, // User/AI assessment (none|minor|moderate|severe|critical)
    evidence_text: string,    // Rationale or evidence
    evidence_source: string,  // e.g. "PMID:12345", "auto-capture:low-confidence"
    status: string,           // pending | approved | rejected
    created_at: ISO datetime,
    reviewed_at: ISO datetime | null
})
```

### Relationships
```
(:Drug)-[:HAS_CORRECTION]->(:Correction)<-[:HAS_CORRECTION]-(:Drug)
```

### How Corrections Overlay Predictions
When `/test drug1 drug2` is run:
1. `_handle_command()` executes the GNN prediction
2. `_lookup_corrections(drug_args)` checks Neo4j for approved corrections for that pair
3. If found, the correction context is injected into the Gemini prompt:
   ```
   === APPROVED CORRECTION ===
   A previous expert review corrected this prediction.
   GNN predicted: moderate | Corrected to: severe
   Evidence: CYP2C9 inhibition is well-documented...
   IMPORTANT: Mention this correction and note the discrepancy.
   ```
4. Gemini references both the GNN prediction and the correction in its response

### Low-Confidence Flagging
When confidence < 0.5, the Gemini prompt includes:
```
WARNING: This prediction has LOW CONFIDENCE (< 0.5).
You MUST prominently flag this in your response and suggest
the user submit a correction if they have better evidence.
```

### Manual Correction (Chat UI)
- `[Correct]` button appears on every LLM assistant message
- Clicking opens an inline form: severity dropdown, evidence textarea, source input
- Submits via `POST /api/v1/corrections/`

### Key File: `web/ddi_api/services/correction_memory.py`
- `CorrectionMemory` class with CRUD operations
- `create_correction()` — stores new correction linked to drug nodes
- `get_approved_correction(drug_a, drug_b)` — used by chatbot for overlay
- `review_correction(id, status, ...)` — approve/reject with optional field updates
- `count_by_status()` — counts for stats bar
- `export_training_data()` — exports approved corrections for GNN retraining
- `delete_correction(id)` — removes correction and relationships
- Drug names are normalized (title-case, alphabetically sorted) so (Aspirin, Warfarin) == (Warfarin, Aspirin)

---

## 6. Phase 4: Corrections Admin Page

### URL: `/corrections`

### Password Protection
- Uses `sessionStorage` — password must be re-entered each browser session
- Password is the same access token configured in Settings > Research Assistant
- Default password: `aegis-owner-2026` (set in `web/.env` as `AEGIS_ASSISTANT_PASSWORD`)

### Page Layout

**Stats Bar** (top):
- 3 metric boxes: Pending (amber), Approved (green), Rejected (red)
- Counts fetched from `GET /api/v1/corrections/stats/`

**Filter Tabs**:
- All | Pending | Approved | Rejected
- Refresh button

**Action Buttons**:
- **AI Review All Pending** — batch-sends all pending corrections to Gemini for assessment
- **Export Training Data** — downloads all approved corrections as JSON (only visible when approved count > 0)

**Corrections List** (card-based):
Each card shows:
- Drug A + Drug B names (bold, uppercase)
- GNN prediction: severity, risk score, confidence %
- Corrected severity (or "Awaiting Review")
- Status badge (pending/approved/rejected)
- "Auto" badge if auto-captured from low-confidence prediction
- Expandable with click

**Expanded Review Panel**:
- AI Assessment section (if AI Review was run) — shows Gemini's analysis with citations
- Editable fields: Corrected Severity dropdown, Evidence Source input, Evidence/Rationale textarea
- Timestamp of creation
- Action buttons: AI Review, Approve, Reject, Delete

### AI Review Feature
When you click "AI Review" on a correction:
1. A prompt is built: "Drug A and Drug B — GNN predicts {severity} with {confidence}% confidence. Assess this interaction using clinical literature. Is the predicted severity accurate? Cite your sources."
2. Sent to the existing chat endpoint (`sendChatMessage`) with the drug pair as context
3. Gemini's response (including citations) is displayed inline on the correction card
4. Admin can then approve/reject with evidence auto-informed by Gemini's analysis

### Navbar Integration
- The old compact StatsDashboard next to CONNECTED has been replaced with a **CORRECTIONS** button (Shield icon)
- Shows a badge with pending count (amber circle, updates every 60 seconds)
- Clicking navigates to `/corrections`

### Key File: `src/pages/CorrectionsPage.jsx`
- Password gate with sessionStorage
- Full CRUD operations using existing API functions
- AI review integration via `sendChatMessage()`
- Training data export as JSON download
- GNN Feedback Loop info panel at the bottom

---

## 7. Token Usage Tracking

### Navbar Indicator
Located in the top-right of the Dashboard navbar, showing:
- Total tokens used (e.g. "2.1k TOKENS")
- Total cost (e.g. "$0.0003")
- Hover tooltip: full breakdown (in/out tokens, queries, model pricing)

### Local Tracking (per-user)
- Stored in `localStorage` key `aegis:token-usage`
- Accumulated on each LLM response
- Survives page refreshes

### Global Tracking (all users)
- `SystemStats` model in SQLite has fields: `llm_input_tokens`, `llm_output_tokens`, `llm_queries`, `llm_cost_usd`
- Updated atomically on every LLM response in `ChatView.post()`
- Response includes `global_usage` — frontend syncs to server totals when available
- Migration: `0005_systemstats_llm_tracking.py`

### How Cost Is Calculated
Gemini returns exact token counts per response. Cost is computed at Flash pricing:
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens
- Formula: `(input_tokens * 0.00000015) + (output_tokens * 0.0000006)`

---

## 8. GNN Feedback Loop

### The Virtuous Cycle
```
Low-confidence GNN prediction
    --> Auto-captured as pending correction
    --> Admin reviews (manual or AI-assisted)
    --> Approved correction stored in Neo4j
    --> Future queries show correction overlay
    --> Export approved corrections as training data
    --> Retrain GNN with corrected labels
    --> Better predictions, fewer low-confidence flags
    --> Fewer corrections needed
```

### Training Data Export
`GET /api/v1/corrections/export/?access_token=...` returns:
```json
{
  "training_data": [
    {
      "drug_a": "Warfarin",
      "drug_b": "Aspirin",
      "original_severity": "moderate",
      "original_risk_score": 0.72,
      "original_confidence": 0.35,
      "corrected_severity": "severe",
      "evidence_text": "CYP2C9 inhibition well-documented",
      "evidence_source": "PMID:12345678",
      "reviewed_at": "2026-04-05T10:30:00Z"
    }
  ],
  "count": 1
}
```

### How to Use for GNN Retraining (Future)
1. Export training data JSON from the Corrections page
2. Each entry provides: original prediction + corrected ground truth + confidence gap
3. Use `corrected_severity` as ground-truth labels for the drug pairs the GNN got wrong
4. Add these to the training set with higher sample weight
5. The `confidence_gap` (how confident the model was when wrong) helps recalibrate outputs
6. Retrain the PyTorch model offline, save as new version
7. Deploy updated model — predictions improve, corrections decrease

---

## 9. Access Control & Passwords

### Single Password System
The entire assistant/corrections system uses ONE password, configured in `web/.env`:

```env
AEGIS_ASSISTANT_PASSWORD=aegis-owner-2026
```

This password is used for:
- **LLM mode access** (Settings > Research Assistant > Access Token)
- **Corrections page authentication** (`/corrections` password gate)
- **Correction CRUD operations** (create, review, delete via API)
- **Training data export** (export endpoint)

### Where the password is stored
- **Server-side**: `web/.env` → loaded into `settings.ASSISTANT_CONFIG['access_password']`
- **Client-side**: User enters it in Settings > Research Assistant, saved to `localStorage` key `aegis:assistant-prefs:v1` as `{ accessToken: "..." }`
- **Corrections session**: After login, `sessionStorage` key `aegis:corrections-authed` is set to `"true"` (expires on tab close)

### Feature Flag
`AEGIS_ASSISTANT_ENABLED=true` in `.env` — set to `false` to disable all LLM features. The system falls back to template-based responses.

---

## 10. Complete File Reference

### New Files Created
| File | Phase | Purpose |
|------|-------|---------|
| `web/ddi_api/services/gemini_client.py` | 1 | Gemini SDK wrapper, system prompt, citation extraction |
| `web/ddi_api/services/command_router.py` | 2 | Slash command parsing and execution |
| `web/ddi_api/services/correction_memory.py` | 3 | Neo4j CRUD for Correction nodes |
| `src/components/ChatCommandAutocomplete.jsx` | 2 | Command/drug autocomplete dropdown |
| `src/pages/CorrectionsPage.jsx` | 4 | Admin page for correction review |
| `web/ddi_api/migrations/0005_systemstats_llm_tracking.py` | 3 | Migration for LLM token tracking fields |
| `docs/LLM_RESEARCH_ASSISTANT_IMPLEMENTATION_GUIDE.md` | 2 | Phase 1+2 implementation guide |
| `docs/PROJECT_AEGIS_COMPLETE_SYSTEM_GUIDE.md` | 4 | This document |

### Modified Files
| File | Changes |
|------|---------|
| `web/ProjectAegis/settings.py` | Added `GEMINI_CONFIG` and `ASSISTANT_CONFIG` dicts |
| `web/requirements.txt` | Added `google-generativeai>=0.8.0` |
| `web/.env` | Added `GEMINI_API_KEY`, `AEGIS_ASSISTANT_ENABLED`, `AEGIS_ASSISTANT_PASSWORD` |
| `web/ddi_api/models.py` | Added LLM tracking fields to `SystemStats` |
| `web/ddi_api/serializers.py` | Added `ChatRequest` assistant fields, `CorrectionCreate/Review` serializers |
| `web/ddi_api/views.py` | Rewrote `ChatView` with LLM support, added `AssistantCommandsView`, `CorrectionListCreateView`, `CorrectionDetailView`, `CorrectionStatsView`, `CorrectionExportView` |
| `web/ddi_api/urls.py` | Added routes for assistant commands, corrections CRUD, stats, export |
| `web/ddi_api/services/graphrag_chatbot.py` | Full rewrite: LLM path, command routing, correction overlay, low-confidence flagging, token usage tracking |
| `web/ddi_api/services/knowledge_graph.py` | Added Correction constraint + indexes to schema |
| `src/App.jsx` | Added `/corrections` route |
| `src/services/api.js` | Added assistant mode params to `sendChatMessage`, correction API functions, stats/export functions |
| `src/pages/Dashboard.jsx` | Chat UI with citations, LLM mode badge, correction form, token usage navbar, corrections indicator button, auto-capture logic |
| `src/pages/SettingsPage.jsx` | Added Research Assistant tab (mode selector, access token) |

---

## 11. API Endpoints Reference

### Existing (Modified)
| Method | Endpoint | Changes |
|--------|----------|---------|
| `POST` | `/api/v1/chat/` | Added `assistant_mode`, `access_token` request fields; `citations`, `assistant_mode`, `model_used`, `token_usage`, `global_usage` response fields |

### New Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/assistant/commands/` | List available slash commands for autocomplete |
| `GET` | `/api/v1/corrections/` | List corrections (optional `?status=pending&drug=Warfarin&limit=50`) |
| `POST` | `/api/v1/corrections/` | Create a new correction (requires `access_token`) |
| `GET` | `/api/v1/corrections/stats/` | Get counts by status (`{pending: N, approved: N, rejected: N}`) |
| `GET` | `/api/v1/corrections/export/` | Export approved corrections as training data (requires `access_token`) |
| `GET` | `/api/v1/corrections/<id>/` | Get single correction by ID |
| `PATCH` | `/api/v1/corrections/<id>/` | Approve/reject + update fields (requires `access_token`) |
| `DELETE` | `/api/v1/corrections/<id>/` | Delete correction (requires `access_token`) |

---

## 12. Configuration Reference

### `web/ProjectAegis/settings.py`

```python
GEMINI_CONFIG = {
    'api_key': os.environ.get('GEMINI_API_KEY', ''),
    'model': os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'),
    'max_output_tokens': 4096,
    'temperature': 0.3,       # Low for clinical accuracy
    'top_p': 0.9,
}

ASSISTANT_CONFIG = {
    'enabled': _env_bool('AEGIS_ASSISTANT_ENABLED', False),
    'access_password': os.environ.get('AEGIS_ASSISTANT_PASSWORD', ''),
    'max_context_tokens': 4000,
    'max_pubmed_results': 3,
}
```

### `web/.env`
```env
GEMINI_API_KEY=AIzaSy...your-key-here
AEGIS_ASSISTANT_ENABLED=true
AEGIS_ASSISTANT_PASSWORD=aegis-owner-2026
```

### Frontend localStorage Keys
| Key | Purpose |
|-----|---------|
| `aegis:assistant-prefs:v1` | `{ mode: "auto"|"llm"|"legacy", accessToken: "..." }` |
| `aegis:token-usage` | `{ totalIn, totalOut, totalCost, queries }` |
| `aegis:corrections-authed` (sessionStorage) | `"true"` when corrections page is unlocked |

---

## 13. Setup & Activation Guide

### First-Time Setup
1. **Install the Gemini SDK** in your backend environment:
   ```bash
   pip install google-generativeai>=0.8.0
   ```

2. **Set environment variables** in `web/.env`:
   ```env
   GEMINI_API_KEY=your-gemini-api-key
   AEGIS_ASSISTANT_ENABLED=true
   AEGIS_ASSISTANT_PASSWORD=your-chosen-password
   ```

3. **Run migrations** (happens automatically in Docker):
   ```bash
   python manage.py migrate
   ```

4. **Configure in the frontend**:
   - Go to Settings > Research Assistant
   - Set mode to "Auto" or "LLM Only"
   - Enter your access token (the password from `.env`)
   - Save

5. **Test it**:
   - Go to Dashboard, type `/test Warfarin Aspirin` in the chat
   - You should see a Gemini-powered clinical assessment with citations
   - Check the token usage indicator in the navbar

### Docker Setup
The Docker configuration already handles migrations on startup. Just ensure the `.env` file has the correct values and rebuild:
```bash
docker-compose up --build
```

---

## 14. Cost Analysis

### Gemini 2.5 Flash Pricing
| Type | Price |
|------|-------|
| Input tokens | $0.15 / 1M tokens |
| Output tokens | $0.60 / 1M tokens |

### Typical Query Costs
| Query Type | Approx Input | Approx Output | Cost |
|-----------|-------------|---------------|------|
| `/test drug1 drug2` | ~1,500 tokens | ~500 tokens | ~$0.0005 |
| Natural language question | ~2,000 tokens | ~400 tokens | ~$0.0005 |
| `/poly` with 5 drugs | ~3,000 tokens | ~800 tokens | ~$0.001 |
| AI Review (correction) | ~2,000 tokens | ~600 tokens | ~$0.0007 |

### Budget Projection
With $300 in GCP credits:
- At ~$0.0005-0.001 per query
- Supports approximately **300,000 - 600,000 queries**
- At 100 queries/day = **8-16 years** of usage

---

## 15. Known Limitations & Future Work

### Current Limitations
1. **No live PubMed fetch**: The chat system uses KG data and cached PubMed results, not live PubMed API calls
2. **No conversation memory**: Each chat message is independent (no multi-turn context)
3. **Polypharmacy confidence is partially hardcoded**: Some confidence values default to 0.85-0.9 rather than coming from the model
4. **Single-user auth**: Simple password, no user accounts or role-based access
5. **No GNN retraining pipeline**: Training data export is ready but the actual retraining script is not built yet

### Future Work (Phase 5+)
1. **Chat session persistence**: Store conversation history in Neo4j for multi-turn context
2. **Live PubMed integration**: Fetch real-time literature during chat queries
3. **GNN retraining script**: Python script that loads corrections + training data, retrains the model
4. **Model versioning**: Save each retrained model with version tags, A/B comparison
5. **Cumulative cost dashboard**: Persistent cost tracking in Settings page
6. **Batch correction import**: Upload CSV of known interactions to seed corrections
7. **Scheduled AI review**: Cron job that auto-reviews pending corrections using Gemini
