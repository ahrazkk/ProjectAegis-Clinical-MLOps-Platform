// store.js — Viewer state management via useReducer + Context
import React, { createContext, useContext, useReducer, useMemo } from 'react';

const GalaxyContext = createContext(null);
const GalaxyDispatchContext = createContext(null);

const initialState = {
  drugA: null,           // { id, name, type, pos }
  drugB: null,
  hoveredNode: null,     // instanceId (number)
  selectedNode: null,    // node object for detail panel
  maxHops: 3,            // 1-3 slider
  showShortestPath: true,
  shortestPath: [],      // array of node IDs
  viewMode: 'galaxy',   // 'galaxy' | 'radial' | 'cluster' | 'path' | 'focus'
  cameraTarget: null,    // [x, y, z] to fly to
  cameraFit: null,       // { center, radius } for zoom-to-fit
  showLabels: 'selected', // 'all' | 'selected' | 'none'
  showEdges: true,
  showFilters: false,
  filters: {
    severityLevels: ['minor', 'moderate', 'major', 'critical', 'unknown'],
    visibleClasses: null, // null = all visible
    minDegree: 0,
    edgeDensity: 1.0,
  },
  selectedEdge: null,       // clicked edge object
  enrichedDrugA: null,      // API-fetched drug info
  enrichedDrugB: null,
  enrichedInteraction: null,
  predictionResult: null,   // result prop from Dashboard
  enrichmentLoading: { drugA: false, drugB: false, interaction: false },
  pathTarget: null,         // for path querying
  focusedCluster: null,     // for cluster zoom
  stats: {
    totalNodes: 0,
    totalEdges: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    pathLength: -1,
    sharedNeighbors: 0,
    selectedCount: 0,
    interactionPairCount: 0,
    focusConnectorEdges: 0,
    focusPathCount: 0,
    focusMode: false,
  },
};

function galaxyReducer(state, action) {
  switch (action.type) {
    case 'SELECT_DRUG_A':
      return {
        ...state,
        drugA: action.payload,
        selectedNode: null,
        viewMode: state.viewMode,
        cameraTarget: action.payload?.pos || null,
      };
    case 'SELECT_DRUG_B':
      return {
        ...state,
        drugB: action.payload,
        selectedNode: null,
        viewMode: state.viewMode,
        cameraTarget: action.payload?.pos || null,
      };
    case 'CLEAR_SELECTION':
      return {
        ...state,
        drugA: null,
        drugB: null,
        selectedNode: null,
        hoveredNode: null,
        shortestPath: [],
        viewMode: 'galaxy',
        cameraTarget: null,
        cameraFit: null,
      };
    case 'SET_HOVERED':
      return { ...state, hoveredNode: action.payload };
    case 'SET_SELECTED_NODE':
      return { ...state, selectedNode: action.payload };
    case 'SET_MAX_HOPS':
      return { ...state, maxHops: action.payload };
    case 'SET_SHORTEST_PATH':
      return { ...state, shortestPath: action.payload };
    case 'SET_STATS':
      return { ...state, stats: action.payload };
    case 'SET_CAMERA_TARGET':
      return { ...state, cameraTarget: action.payload };
    case 'SET_CAMERA_FIT':
      return { ...state, cameraFit: action.payload };
    case 'SET_VIEW_MODE':
      return { ...state, viewMode: action.payload };
    case 'TOGGLE_LABELS': {
      const cycle = { 'selected': 'all', 'all': 'none', 'none': 'selected' };
      return { ...state, showLabels: cycle[state.showLabels] || 'selected' };
    }
    case 'TOGGLE_EDGES':
      return { ...state, showEdges: !state.showEdges };
    case 'SET_SHOW_FILTERS':
      return { ...state, showFilters: action.payload };
    case 'SET_FILTERS':
      return { ...state, filters: { ...state.filters, ...action.payload } };
    case 'SET_ENRICHED_DRUG_A':
      return { ...state, enrichedDrugA: action.payload, enrichmentLoading: { ...state.enrichmentLoading, drugA: false } };
    case 'SET_ENRICHED_DRUG_B':
      return { ...state, enrichedDrugB: action.payload, enrichmentLoading: { ...state.enrichmentLoading, drugB: false } };
    case 'SET_ENRICHED_INTERACTION':
      return { ...state, enrichedInteraction: action.payload, enrichmentLoading: { ...state.enrichmentLoading, interaction: false } };
    case 'SET_ENRICHMENT_LOADING':
      return { ...state, enrichmentLoading: { ...state.enrichmentLoading, ...action.payload } };
    case 'SET_PREDICTION_RESULT':
      return { ...state, predictionResult: action.payload };
    case 'SET_SELECTED_EDGE':
      return { ...state, selectedEdge: action.payload, selectedNode: null };
    case 'SET_PATH_TARGET':
      return { ...state, pathTarget: action.payload };
    case 'SET_FOCUSED_CLUSTER':
      return { ...state, focusedCluster: action.payload };
    default:
      return state;
  }
}

export function GalaxyProvider({ children, initialDrugs }) {
  const [state, dispatch] = useReducer(galaxyReducer, {
    ...initialState,
    // Will be set via effect in index.jsx when drugs prop changes
  });

  return (
    <GalaxyContext.Provider value={state}>
      <GalaxyDispatchContext.Provider value={dispatch}>
        {children}
      </GalaxyDispatchContext.Provider>
    </GalaxyContext.Provider>
  );
}

export function useGalaxy() {
  const ctx = useContext(GalaxyContext);
  if (!ctx) throw new Error('useGalaxy must be used within GalaxyProvider');
  return ctx;
}

export function useGalaxyDispatch() {
  const ctx = useContext(GalaxyDispatchContext);
  if (!ctx) throw new Error('useGalaxyDispatch must be used within GalaxyProvider');
  return ctx;
}
