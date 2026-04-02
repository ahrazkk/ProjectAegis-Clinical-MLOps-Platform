# GNN Galaxy Viewer Final Features and Technical Handoff

## 1. Purpose and Scope

This document is the full knowledge transfer artifact for the GNN Galaxy Viewer implementation inside the molecular-ai frontend. It is written for both:

- Human maintainers who need to evolve, debug, or validate behavior.
- AI assistants that need precise implementation context without reverse engineering every file.

The goal of the Galaxy Viewer is not decorative visualization. It is a research-facing, interaction-aware 3D analysis workspace for:

- Multi-drug regimen exploration.
- Interaction topology interpretation.
- Embedding-space neighborhood analysis.
- Evidence-oriented drilldown and export.

This document describes what the viewer does, how it does it, why specific choices were made, and how to safely modify it.

---

## 2. Where It Lives

### 2.1 Primary frontend module

- Main orchestrator: src/components/GalaxyViewer/index.jsx
- State store: src/components/GalaxyViewer/store.jsx
- Graph computation core: src/components/GalaxyViewer/graphEngine.js
- Data loading and coordinate integrity pipeline: src/components/GalaxyViewer/graphDataService.js

### 2.2 Rendering modules

- Node renderer: src/components/GalaxyViewer/InstancedNodes.jsx
- Edge renderer: src/components/GalaxyViewer/InstancedEdges.jsx
- Camera controls: src/components/GalaxyViewer/CameraController.jsx
- Path animation: src/components/GalaxyViewer/PathParticles.jsx
- Hop shell overlays: src/components/GalaxyViewer/HopShells.jsx
- Cluster visual overlays: src/components/GalaxyViewer/ClusterBubbles.jsx

### 2.3 Layout engines

- Radial layout: src/components/GalaxyViewer/layouts/RadialLayout.js
- Cluster layout: src/components/GalaxyViewer/layouts/ClusterLayout.js
- Path layout: src/components/GalaxyViewer/layouts/PathLayout.js

### 2.4 UI overlays

- Toolbar: src/components/GalaxyViewer/overlays/Toolbar.jsx
- Search: src/components/GalaxyViewer/overlays/SearchBar.jsx
- Filters: src/components/GalaxyViewer/overlays/FilterPanel.jsx
- HUD telemetry: src/components/GalaxyViewer/overlays/HUD.jsx
- Legend and semantic explanation: src/components/GalaxyViewer/overlays/Legend.jsx
- Node detail: src/components/GalaxyViewer/overlays/NodeDetailPanel.jsx
- Edge detail: src/components/GalaxyViewer/overlays/EdgeDetailPanel.jsx
- Drug comparison strip: src/components/GalaxyViewer/overlays/DrugComparisonPanel.jsx
- Hop depth control: src/components/GalaxyViewer/overlays/HopSlider.jsx
- Embedding insights analytics panel: src/components/GalaxyViewer/overlays/EmbeddingInsightsPanel.jsx

### 2.5 Backend API contract used by Galaxy

- Routes: web/ddi_api/urls.py
- Graph views: web/ddi_api/views.py
  - GraphNodesView
  - GraphEdgesView
  - GraphNeighborhoodView

### 2.6 Dashboard integration points

- Main usage (desktop and mobile): src/pages/Dashboard.jsx

---

## 3. Runtime Data Flow

## 3.1 End-to-end sequence

1. Dashboard renders GNNGalaxyViewer with selected drugs, result, and polypharmacyResult props.
2. Galaxy Viewer mounts and calls loadGraphData from graphDataService.
3. Data service attempts live API retrieval:
   - Graph nodes from /api/v1/graph/nodes/
   - Graph edges from /api/v1/graph/edges/
4. Data service builds adjacency and assigns meaningful coordinates through a staged provenance pipeline.
5. graphEngine receives normalized graph via setGraphData, then builds edge metadata index.
6. Viewer computes subgraph state via computeSubgraph based on:
   - Selected drugs and selected regimen set.
   - Hops.
   - View mode.
   - Filters.
7. applyFilters derives nodeVisibility and filteredEdges for render.
8. Scene renders instanced nodes and edges plus mode-specific overlays.
9. On interaction, enrichment calls are dispatched for drug and pair evidence.
10. Embedding Insights panel performs local metric analysis and optional CSV export.

## 3.2 Live vs static fallback behavior

- Primary source is live API.
- If API fails or returns unusable data, static fallback imports gnn_real_data.json.
- DataSourceBadge explicitly indicates LIVE or STATIC and includes node/edge counts.
- The design prevents silent behavior changes when backend is unavailable.

---

## 4. Coordinate Integrity Model

This is one of the most important trustworthiness features.

Previous synthetic placement behavior was intentionally replaced. Current logic preserves truthful placement as much as possible.

## 4.1 Coordinate source priority

For each node, graphDataService resolves position in strict order:

1. API-native coordinates:
   - pos
   - x, y, z
   - tsne_x, tsne_y, tsne_z
   - embedding_x, embedding_y, embedding_z
2. Model atlas lookup by normalized/aliased drug name from gnn_real_data.json
3. Topology interpolation from neighbors with known coordinates
4. Deterministic topology-derived component layout for unresolved components

## 4.2 Why this matters

- Prevents fabricated geometry from being mistaken as learned model structure.
- Maximizes geometric continuity when data is partially available.
- Keeps unmatched nodes visible in deterministic, reproducible regions.
- Enables provenance-aware interpretation in research settings.

## 4.3 Provenance visibility

DataSourceBadge displays compressed coordinate source counts, for example:

- api-coordinate
- model-tsne
- topology-interpolated
- topology-derived

This gives immediate transparency into geometry quality.

---

## 5. Graph Model and Classification

## 5.1 Node model (frontend normalized)

Each node in graphEngine carries:

- id, name, type, category, pos
- index
- selection flags: isA, isB, isSelected
- focus flag: isFocusNode
- hop distances: hopA, hopB, hopAny

## 5.2 Therapeutic category normalization

graphEngine maps heterogeneous class labels into stable meta-categories using CLASS_MAP, then colors by CATEGORY_COLORS.

This is used by:

- Node coloring in non-selected views.
- Filtering by class.
- Cluster layout grouping.
- Embedding purity metrics.

---

## 6. View Modes and Their Semantics

The viewer has six modes. They are not visual skins; each mode changes the analytical semantics of rendering and edge selection.

## 6.1 Galaxy mode

- Default mode.
- Uses baseline coordinates from data pipeline.
- Highlights selected-drug neighborhoods and interaction structures over full atlas.
- HopShells enabled.

## 6.2 Embedding mode

- Explicit all-drug latent atlas view.
- Full node set remains visible independent of hop shell semantics.
- Hop slider intentionally hidden to avoid semantic mismatch.
- Edge model can switch between:
  - KNN manifold edges
  - Known graph topology edges
- Embedding Insights panel is enabled only in this mode.

## 6.3 Radial mode

- Centered, concentric hop-based arrangement.
- Single drug: center plus hop rings.
- Two drugs: dual-center split with bridge region.
- Good for quick hop-depth comprehension.

## 6.4 Cluster mode

- Groups nodes by therapeutic category.
- Cluster centers distributed on a sphere.
- In-cluster repulsion avoids overlap.
- ClusterBubbles provide soft enclosing context with labels.

## 6.5 Path mode

- Uses shortest path as primary spine.
- Path nodes laid out linearly.
- Non-path neighbors fan vertically (subway-map style).
- PathParticles emphasize directional flow and continuity.

## 6.6 Focus mode

- Minimal connector network across selected regimen drugs.
- Keeps direct selected-selected edges.
- Builds diversified least-hop connectors with reuse penalties.
- Designed to reduce clutter while preserving critical routes.

---

## 7. Selection and Multi-Drug Logic

## 7.1 Selection model

- drugA and drugB are primary anchors.
- selectedDrugIds also includes additional regimen drugs from Dashboard input.
- interactionPairs are derived from polypharmacyResult.interactions and overlaid.

## 7.2 Hop behavior

- hopA and hopB are BFS distances from anchor drugs.
- hopAny is BFS distance from union of selected regimen anchors.
- This supports proper multi-drug union neighborhoods.

## 7.3 Shortest path behavior

- shortestPath computed between A and B via BFS.
- Path is used for:
  - path mode layout
  - edge highlighting
  - path particles
  - panel metrics

---

## 8. Focus Mode Connector Algorithm

Focus mode uses a dedicated strategy in graphEngine:

1. Build initial selected set.
2. Keep all direct selected-selected graph edges.
3. Compute connected components over selected set.
4. Repeatedly connect components using preferred path candidate scoring.

Path preference combines:

- Shortest hops priority.
- Reuse penalty for intermediate connector node reuse.
- Reuse penalty for repeated connector edges.
- Optional one-hop longer alternative if it avoids repeated hubs.

This creates more diverse and readable connector routes when equivalent shortest paths exist.

---

## 9. Edge Construction Rules

## 9.1 Normal graph-based edge role assignment

Roles include:

- path
- interaction-pair
- selected-pair
- focus
- direct
- bridge
- hopA
- hopB
- multi-hop
- background

Each role has default color, opacity, and width behavior.

## 9.2 Severity-aware tinting

For active graph edges, severity from metadata may tint colors:

- critical -> red
- major -> orange family
- moderate -> amber family

Severity normalization supports synonyms like severe/high/none.

## 9.3 Embedding edge models

### KNN mode

- Builds symmetric sparse local manifold approximations from node distances.
- k is clamped to [2, 20] in UI and [1, N-1] internally.
- KNN background edges are low-opacity context.
- Important overlays are upserted with higher priority:
  - selected pair edges
  - interaction pairs
  - path edges

### Graph mode

- Uses adjacency topology directly.
- Samples background edges deterministically to avoid hairball clutter.
- Preserves highlighted selected and path semantics.

---

## 10. Filtering Pipeline

Filtering happens after subgraph generation in applyFilters.

## 10.1 Node filters

- Therapeutic class visibility
- Minimum degree
- Focus mode suppression of non-connector nodes
- Selected drugs always forced visible

## 10.2 Edge filters

- Node visibility gating for non-important edges
- Severity inclusion set
- Edge density deterministic downsampling
- Important roles always preserved regardless of density:
  - path
  - direct
  - selected-pair
  - interaction-pair
  - focus

This ensures readability controls do not hide critical interpretation edges.

---

## 11. Rendering Architecture

## 11.1 Instanced node rendering

InstancedNodes uses a single InstancedMesh with per-instance attributes:

- instanceOpacity
- instanceGlow
- instanceColor

Key features:

- Custom shader with diffuse + rim/fresnel emphasis.
- Entrance cascade animation by radius/distance.
- Smooth lerp transitions for color/scale/position.
- Hover boost behavior.
- Selection breathing animation.
- Subtle floating offset for selected regimen nodes.
- Optional label rendering based on mode and cap logic.

## 11.2 Instanced edge rendering

InstancedEdges splits edges into:

- Background lineSegments for lightweight context.
- Active instanced cylinders with shader-driven flow particles.

Active edge attributes:

- instanceColorAttr
- instanceOpacityAttr
- instanceFlowSpeed

This gives performant emphasis of active relationships while keeping context cost low.

---

## 12. Layout Engines in Detail

## 12.1 Radial layout

- Single anchor mode:
  - center at origin
  - hop buckets mapped to fixed radii
  - optional dual sub-rings for dense rings
  - category ordering for sector coherence
- Dual anchor mode:
  - A left, B right
  - bridge nodes in center band
  - per-side neighborhood arcs
  - outer shell background nodes

## 12.2 Cluster layout

- Group by category.
- Category centers on spherical Fibonacci distribution.
- In-cluster Fibonacci seed plus iterative repulsion for overlap control.

## 12.3 Path layout

- Path nodes on X-axis with fixed spacing.
- Non-path neighbors split top/bottom branches with capped count.
- Remaining nodes pushed to distant background sphere.

---

## 13. Camera and Motion Model

CameraController uses OrbitControls and two automatic movement behaviors:

- Fly-to target when a node is selected as A or B.
- Zoom-to-fit center framing when both A and B are selected.

Additional behavior:

- Auto-rotate enabled by default.
- User interaction pauses auto-rotate.
- Auto-rotate resumes after idle timeout.

This balances cinematic readability and user control.

---

## 14. Overlay System: Complete Behavior

## 14.1 Toolbar

Functions:

- Mode switching (galaxy, embedding, radial, cluster, path, focus)
- Filter panel toggle
- Label cycle toggle
- Edge visibility toggle
- Screenshot export
- Reset selection
- Fullscreen

Path and focus buttons are context-aware and can be disabled.

## 14.2 SearchBar

Supports two workflows:

- Normal drug selection mode.
- Path search mode when an anchor exists.

Path search computes shortest path and switches to path mode.

## 14.3 FilterPanel

Controls:

- Severity set
- Drug class inclusion
- Min degree
- Edge density
- Embedding edge model selector
- Embedding k slider (for KNN)

Includes filter count badge and one-click reset.

## 14.4 HUD

Displays runtime telemetry:

- Total/visible nodes and edges
- Hop depth
- Path length
- Shared bridges
- Selected count
- Focus edge/path stats
- Embedding model status when applicable

## 14.5 Legend

Shows semantic color coding and contextual explanation text.

- In embedding mode, text explains full atlas and highlighted links.
- In non-embedding modes, text explains hop aggregation and focus routing semantics.

## 14.6 HopSlider

- Only shown when a drug is selected and mode is hop-aware.
- Hidden in embedding mode by design.

## 14.7 NodeDetailPanel

Provides selected node data:

- Type and connection count
- Hop distances from A/B
- Enrichment snippets (CYP, side effects, FAERS)
- Quick actions to set node as drug A or B
- Neighbor click-through list

## 14.8 EdgeDetailPanel

On edge click:

- Fetches enriched interaction data for start/end drug names
- Displays risk gauge, severity, confidence, mechanism, systems, evidence tags
- Falls back to graph role/type text if enrichment unavailable

## 14.9 DrugComparisonPanel

When A and B are both selected:

- Shows names, categories, degrees
- Visual path indicator and hop count
- Shared/edge stats chips

## 14.10 EmbeddingInsightsPanel

This is the research analysis panel and supports:

- Normal and maximized modes
- Regimen tab buttons for A/B/selected/clicked anchors
- Atlas-wide search to analyze any drug
- Nearest-neighbor table
- Metric chips
- Formula references
- Regimen comparison rows
- CSV exports:
  - nearest neighbors
  - regimen summary

Panel layout is constrained to avoid viewport overflow:

- Normal mode top-right with max-height and internal scroll.
- Expanded mode inset layout with independent scroll regions.

---

## 15. Embedding Insights Metrics

For an anchor drug x and local neighborhood size k (analysisK):

- Mean kNN distance
- Standard deviation of kNN distances
- Local density proxy rho_k(x) = 1 / mean_kNN_distance
- Class purity at k
- Graph overlap at k
- Expected overlap under random baseline
- Overlap enrichment = observed / expected
- Local silhouette estimate
- Mean rank position of graph-neighbor nodes in latent distance ordering
- Degree and degree percentile

## 15.1 Interpretation guidance

High purity and positive local silhouette generally imply coherent therapeutic neighborhoods.

High overlap and high enrichment indicate agreement between latent geometry and known graph topology.

Low overlap with high local density can indicate:

- Potentially novel latent associations.
- Data incompleteness in known interactions.
- Model clustering around shared chemistry that is not explicit in graph edges.

Use these as hypotheses, not clinical conclusions.

---

## 16. API Contracts Used by Galaxy

## 16.1 Graph endpoints

- GET /api/v1/graph/nodes/
  - includes id, name, category, therapeutic_class, degree, and multiple coordinate fields
- GET /api/v1/graph/edges/
  - paginated source-target-severity edge list
- GET /api/v1/graph/neighborhood/
  - optional neighborhood query for targeted contexts

## 16.2 Enrichment endpoints used by viewer detail panels

- GET /api/v1/drug-info/
- GET /api/v1/interaction-info/
- GET /api/v1/real-world-evidence/
- POST /api/v1/predict/

Frontend wrapper:

- src/components/GalaxyViewer/dataEnrichment.js using src/services/api.js

---

## 17. Performance Strategy

## 17.1 Data-side

- In-memory cache for nodes, edges, and neighborhoods with TTL.
- Parallel fetch of nodes and edges.
- Edges fetched in pages for large datasets.

## 17.2 Compute-side

- Pure graph computations centralized in graphEngine.
- Extensive useMemo usage in main orchestrator and render modules.
- Deterministic edge sampling for high-density context edges.

## 17.3 Render-side

- Instanced mesh for nodes.
- Instanced cylinders for active edges.
- Lightweight lineSegments for background edges.
- Additive blending and low depth-write where appropriate.

## 17.4 UX-side

- Filter panel controls for edge density and min-degree to tame complexity.
- Focus mode for structural simplification.
- Embedding edge model switch for semantic clarity.

---

## 18. Keyboard Shortcuts

Defined in index.jsx:

- 1 -> galaxy
- 2 -> embedding
- 3 -> radial
- 4 -> cluster
- 5 -> path
- 6 -> focus
- E -> toggle edges
- L -> cycle labels
- F -> toggle filters
- R -> clear graph cache and reload data
- Escape -> clear selection and close filters

Input fields are excluded from shortcut handling.

---

## 19. State Machine Summary

Key store actions in store.jsx:

- SELECT_DRUG_A / SELECT_DRUG_B
- CLEAR_SELECTION
- SET_HOVERED
- SET_SELECTED_NODE
- SET_SELECTED_EDGE
- SET_MAX_HOPS
- SET_SHORTEST_PATH
- SET_VIEW_MODE
- TOGGLE_LABELS
- TOGGLE_EDGES
- SET_FILTERS
- SET_STATS
- SET_ENRICHED_DRUG_A / SET_ENRICHED_DRUG_B / SET_ENRICHED_INTERACTION
- SET_ENRICHMENT_LOADING
- SET_PREDICTION_RESULT
- SET_PATH_TARGET
- SET_FOCUSED_CLUSTER

State currently tracks:

- selection anchors
- hover/click details
- view configuration
- filters
- enrichment payloads and loading flags
- analytics stats

---

## 20. Research Workflow Playbook

Suggested workflow for research-oriented analysis:

1. Open embedding mode and set edge model to KNN.
2. Inspect anchor metrics for A and B, then additional regimen drugs.
3. Use overlap enrichment to evaluate latent-vs-graph consistency.
4. Switch to graph edge model to compare neighborhood semantics.
5. Use focus mode to inspect minimal connector routes for the full regimen.
6. Export nearest-neighbor and regimen summary CSV for reporting.
7. Capture screenshot states for figures.
8. Cross-check edge-level enrichment for mechanistic plausibility.

This creates a repeatable evidence trail from atlas geometry to interaction rationale.

---

## 21. Troubleshooting Guide

## 21.1 Viewer shows no data

Checklist:

- Verify backend health endpoint.
- Verify graph endpoints respond with non-empty nodes and edges.
- Confirm API base path settings in frontend environment.
- Inspect DataSourceBadge for STATIC fallback state.

## 21.2 Embedding view looks cluttered

Checklist:

- Set edge model to KNN.
- Reduce k.
- Reduce edge density in filters.
- Use focus mode for connector-only analysis if regimen-specific.

## 21.3 Insights panel overlaps content

Current implementation uses bounded panel containers with internal scroll.
If future style changes regress this:

- Re-check panel container max-height rules.
- Re-check z-index of search dropdown.
- Ensure expanded mode uses min-h-0 and dedicated scroll regions.

## 21.4 Slow interaction on large graphs

Checklist:

- Lower maxEdges in performance limits if needed.
- Decrease edge density filter.
- Increase min degree threshold.
- Prefer focus mode for targeted connector analysis.

## 21.5 Unexpected positioning anomalies

Checklist:

- Confirm coordinate fields on backend nodes.
- Confirm fallback atlas import integrity.
- Inspect coordinate source counts in badge.
- Verify no synthetic placement code is reintroduced.

---

## 22. Validation Checklist for Safe Changes

When modifying Galaxy behavior, validate all of the following:

- Live data load still succeeds.
- Static fallback still functions when API unavailable.
- DataSourceBadge still displays source and counts.
- All six view modes render and switch correctly.
- Hop slider hidden in embedding mode.
- Edge model controls appear only in embedding mode.
- KNN k slider only appears in KNN mode.
- Focus mode preserves direct selected-selected edges.
- Path mode still receives shortest path and particles.
- Node click opens detail panel.
- Edge click opens enriched detail panel.
- Embedding insights search can switch to non-selected drugs.
- CSV exports download correctly.
- Overlay layout remains bounded on smaller viewport heights.
- Keyboard shortcuts still work and do not trigger while typing in inputs.

---

## 23. Extension Points

## 23.1 Add new metrics to Embedding Insights

- Extend analysisById computation in EmbeddingInsightsPanel.jsx.
- Add chip/UI cells and export columns.
- Keep interpretation text updated with formula additions.

## 23.2 Add a new view mode

- Add mode in Toolbar viewModes list.
- Add key binding in index.jsx shortcut map.
- Add layout logic in layoutPositions switch.
- Extend legend/HUD semantics if mode-specific.

## 23.3 Add new edge role semantics

- Add role assignment in computeSubgraph.
- Add filtering and importance behavior in applyFilters.
- Add color/line style policy and optional legend text.
- Update edge detail role-to-text mapping.

## 23.4 Introduce backend graph attributes

- Include fields in GraphNodesView and/or GraphEdgesView.
- Parse and normalize in graphDataService.
- Surface in node/edge detail overlays or metrics panels.

---

## 24. Scientific Communication Notes

This viewer can support research communication, but claims should remain calibrated:

- Latent geometry is model-derived, not causal proof.
- Graph overlaps are concordance indicators, not outcome validation.
- Severity colors can communicate risk prioritization, not certainty.
- Enrichment data should be described as supporting evidence.

Recommended wording for publications and presentations:

- Use terms like neighborhood coherence, topological concordance, hypothesis-generating signals.
- Avoid terms that imply confirmed clinical outcomes without external validation.

---

## 25. Operational Notes for Human and AI Handoff

When handing this module to a new maintainer or AI agent, provide:

- This document.
- Current graph API sample responses.
- A known regimen scenario with expected mode behaviors.
- Any active performance constraints or environment-specific API path settings.

For AI assistants specifically, indicate:

- Do not reintroduce synthetic random placement for live mode.
- Preserve semantic distinction between embedding and hop-based modes.
- Keep critical edges immune to density filtering.
- Validate overlay bounds after UI edits.

---

## 26. Quick File Map by Responsibility

- Orchestration and lifecycle: src/components/GalaxyViewer/index.jsx
- State and actions: src/components/GalaxyViewer/store.jsx
- Graph algorithms and semantics: src/components/GalaxyViewer/graphEngine.js
- Data acquisition and coordinate provenance: src/components/GalaxyViewer/graphDataService.js
- Node rendering: src/components/GalaxyViewer/InstancedNodes.jsx
- Edge rendering: src/components/GalaxyViewer/InstancedEdges.jsx
- Camera: src/components/GalaxyViewer/CameraController.jsx
- Path animation: src/components/GalaxyViewer/PathParticles.jsx
- Layout systems: src/components/GalaxyViewer/layouts/*.js
- Interactive overlays: src/components/GalaxyViewer/overlays/*.jsx
- Dashboard host: src/pages/Dashboard.jsx
- Backend graph routes: web/ddi_api/urls.py
- Backend graph payload logic: web/ddi_api/views.py

---

## 27. Final Summary

The current Galaxy Viewer is a multi-layer analytical system with:

- Truthful coordinate provenance handling.
- Explicit embedding-mode semantics with KNN or graph edge models.
- Multi-drug connector-aware focus routing.
- Rich drilldown and evidence panels.
- Quantitative embedding metrics and exports.
- Production-ready controls for complexity management.

It is suitable as a research exploration and communication interface when interpreted with proper scientific caution and supported by downstream validation workflows.
