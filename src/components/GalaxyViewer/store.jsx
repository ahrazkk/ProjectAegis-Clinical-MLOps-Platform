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
  viewMode: 'galaxy',   // 'galaxy' | 'neighborhood' | 'path'
  cameraTarget: null,    // [x, y, z] to fly to
  cameraFit: null,       // { center, radius } for zoom-to-fit
  stats: {
    totalNodes: 0,
    totalEdges: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    pathLength: -1,
    sharedNeighbors: 0,
  },
};

function galaxyReducer(state, action) {
  switch (action.type) {
    case 'SELECT_DRUG_A':
      return {
        ...state,
        drugA: action.payload,
        selectedNode: null,
        viewMode: action.payload ? 'neighborhood' : (state.drugB ? 'neighborhood' : 'galaxy'),
        cameraTarget: action.payload?.pos || null,
      };
    case 'SELECT_DRUG_B':
      return {
        ...state,
        drugB: action.payload,
        selectedNode: null,
        viewMode: action.payload ? 'neighborhood' : (state.drugA ? 'neighborhood' : 'galaxy'),
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
