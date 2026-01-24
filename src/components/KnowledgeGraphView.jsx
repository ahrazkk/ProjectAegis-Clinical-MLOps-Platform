import React, { useMemo, useRef, useCallback, useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Target, Activity, Pill, AlertTriangle, X, Info } from 'lucide-react';

// Node types with their colors
const nodeTypes = {
  drug: { color: '#3B82F6', bgColor: 'bg-blue-500/20', borderColor: 'border-blue-500/30', icon: Pill },
  target: { color: '#10B981', bgColor: 'bg-emerald-500/20', borderColor: 'border-emerald-500/30', icon: Target },
  enzyme: { color: '#A855F7', bgColor: 'bg-purple-500/20', borderColor: 'border-purple-500/30', icon: Activity },
  pathway: { color: '#F59E0B', bgColor: 'bg-amber-500/20', borderColor: 'border-amber-500/30', icon: Activity },
  interaction: { color: '#EF4444', bgColor: 'bg-red-500/20', borderColor: 'border-red-500/30', icon: AlertTriangle }
};

// Common drug targets and enzymes for visualization
const commonTargets = ['CYP3A4', 'CYP2D6', 'CYP2C9', 'P-gp', 'OATP1B1'];
const commonEnzymes = ['CYP3A4', 'CYP2D6', 'CYP2C9', 'CYP2C19', 'CYP1A2'];

// Helper to extract entities from text
function extractEntities(text) {
  if (!text) return [];

  const entities = new Set();

  // Common biological entities to look for
  const patterns = [
    /CYP\d+[A-Z]\d+/gi, // CYP3A4, CYP2D6
    /VKORC1/gi,
    /COX-[12]/gi,
    /P-gp/gi,
    /OATP\w+/gi,
    /Platelet/gi,
    /Serotonin/gi,
    /Dopamine/gi,
    /ACE/gi
  ];

  patterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(m => entities.add(m));
    }
  });

  return Array.from(entities).map((name, i) => ({
    id: `entity-${i}`,
    label: name,
    type: name.toUpperCase().startsWith('CYP') ? 'enzyme' : 'target'
  }));
}

function generateGraphData(drugs, result, polypharmacyResult) {
  const nodes = [];
  const edges = [];

  if (drugs.length === 0) return { nodes, edges };

  // Add drug nodes in a circle
  const centerX = 400;
  const centerY = 300;
  // Dynamic radius based on count
  const drugRadius = drugs.length === 2 ? 200 : 150;

  drugs.forEach((drug, i) => {
    // For 2 drugs, place them Left and Right
    let angle;
    if (drugs.length === 2) {
      angle = i === 0 ? Math.PI : 0; // Left (180deg) and Right (0deg)
    } else {
      angle = (i / drugs.length) * Math.PI * 2 - Math.PI / 2;
    }

    nodes.push({
      id: drug.drugbank_id || drug.name,
      label: drug.name,
      type: 'drug',
      x: centerX + Math.cos(angle) * drugRadius,
      y: centerY + Math.sin(angle) * drugRadius,
      isHub: polypharmacyResult?.hub_drug === drug.name
    });
  });

  // Dynamic Entity Extraction
  if (result && drugs.length >= 2) {
    const mechanismText = result.mechanism_hypothesis || '';
    const extractedEntities = extractEntities(mechanismText);

    // Position extracted entities in the center
    const totalEntities = extractedEntities.length;
    extractedEntities.forEach((entity, i) => {
      // Stack vertically in center
      const yOffset = totalEntities > 1 ? (i - (totalEntities - 1) / 2) * 80 : 0;

      nodes.push({
        ...entity,
        x: centerX,
        y: centerY + yOffset
      });

      // Connect drugs to this entity
      drugs.forEach(drug => {
        edges.push({
          source: drug.drugbank_id || drug.name,
          target: entity.id,
          type: 'mechanism',
          strength: 0.6
        });
      });
    });

    // Add direct interaction edge if no entities found OR explicit high risk
    if (extractedEntities.length === 0 || result.risk_score > 0.7) {
      edges.push({
        source: drugs[0].drugbank_id || drugs[0].name,
        target: drugs[1].drugbank_id || drugs[1].name,
        type: 'interaction',
        severity: result.risk_level || 'medium',
        strength: result.risk_score || 0.5
      });
    }
  }

  // Handle polypharmacy results
  if (polypharmacyResult?.interactions) {
    polypharmacyResult.interactions.forEach((interaction, i) => {
      const sourceNode = nodes.find(n => n.label === interaction.source);
      const targetNode = nodes.find(n => n.label === interaction.target);

      if (sourceNode && targetNode) {
        edges.push({
          source: sourceNode.id,
          target: targetNode.id,
          type: 'interaction',
          severity: interaction.severity,
          strength: interaction.risk_score || 0.5
        });
      }
    });
  }

  return { nodes, edges };
}

function GraphNode({ node, isSelected, onSelect, onDrag }) {
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: node.x, y: node.y });
  const nodeRef = useRef(null);

  const typeConfig = nodeTypes[node.type] || nodeTypes.drug;
  const Icon = typeConfig.icon;

  const handleMouseDown = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return;
      const rect = nodeRef.current?.parentElement?.getBoundingClientRect();
      if (rect) {
        const newX = e.clientX - rect.left;
        const newY = e.clientY - rect.top;
        setPosition({ x: newX, y: newY });
        onDrag?.(node.id, newX, newY);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, node.id, onDrag]);

  return (
    <motion.g
      ref={nodeRef}
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1, x: position.x - node.x, y: position.y - node.y }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      onMouseDown={handleMouseDown}
      onClick={() => onSelect(node)}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
    >
      {/* Glow effect for hub drugs */}
      {node.isHub && (
        <motion.circle
          cx={node.x}
          cy={node.y}
          r={45}
          fill="none"
          stroke={typeConfig.color}
          strokeWidth={2}
          opacity={0.3}
          animate={{ r: [45, 55, 45], opacity: [0.3, 0.1, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}

      {/* Node background */}
      <circle
        cx={node.x}
        cy={node.y}
        r={node.type === 'drug' ? 35 : 25}
        fill={`${typeConfig.color}20`}
        stroke={isSelected ? '#ffffff' : typeConfig.color}
        strokeWidth={isSelected ? 3 : 2}
        className="transition-all duration-200"
      />

      {/* Icon */}
      <foreignObject
        x={node.x - 10}
        y={node.y - 10}
        width={20}
        height={20}
      >
        <Icon className="w-5 h-5" style={{ color: typeConfig.color }} />
      </foreignObject>

      {/* Label */}
      <text
        x={node.x}
        y={node.y + (node.type === 'drug' ? 50 : 40)}
        textAnchor="middle"
        fill="#94A3B8"
        fontSize={node.type === 'drug' ? 12 : 10}
        fontWeight={node.type === 'drug' ? 600 : 400}
      >
        {node.label}
      </text>

      {/* Hub indicator */}
      {node.isHub && (
        <text
          x={node.x}
          y={node.y + 65}
          textAnchor="middle"
          fill="#F59E0B"
          fontSize={9}
          fontWeight={600}
        >
          HUB DRUG
        </text>
      )}
    </motion.g>
  );
}

function GraphEdge({ edge, nodes }) {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  if (!sourceNode || !targetNode) return null;

  const isInteraction = edge.type === 'interaction';

  const getEdgeColor = () => {
    if (!isInteraction) return '#475569';
    switch (edge.severity) {
      case 'critical': return '#EF4444';
      case 'high': case 'major': return '#F97316';
      case 'medium': case 'moderate': return '#EAB308';
      default: return '#22C55E';
    }
  };

  const color = getEdgeColor();
  const strokeWidth = isInteraction ? 2 + (edge.strength || 0.5) * 2 : 1.5;

  // Calculate path with slight curve
  const dx = targetNode.x - sourceNode.x;
  const dy = targetNode.y - sourceNode.y;
  const dr = Math.sqrt(dx * dx + dy * dy) * 0.5;

  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
    >
      <path
        d={`M ${sourceNode.x} ${sourceNode.y} Q ${(sourceNode.x + targetNode.x) / 2 + dy * 0.1} ${(sourceNode.y + targetNode.y) / 2 - dx * 0.1} ${targetNode.x} ${targetNode.y}`}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={isInteraction ? 'none' : '5,5'}
        opacity={isInteraction ? 0.8 : 0.4}
        markerEnd={isInteraction ? 'none' : 'url(#arrowhead)'}
      />

      {/* Animated pulse for interactions */}
      {isInteraction && (
        <motion.circle
          r={4}
          fill={color}
          animate={{
            cx: [sourceNode.x, targetNode.x],
            cy: [sourceNode.y, targetNode.y]
          }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        />
      )}
    </motion.g>
  );
}

function NodeInfoPanel({ node, onClose }) {
  if (!node) return null;

  const typeConfig = nodeTypes[node.type] || nodeTypes.drug;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="absolute top-4 right-4 w-72 bg-theme-panel backdrop-blur-xl border border-theme rounded-2xl p-4 shadow-2xl"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl ${typeConfig.bgColor} ${typeConfig.borderColor} border flex items-center justify-center`}>
            <typeConfig.icon className="w-5 h-5" style={{ color: typeConfig.color }} />
          </div>
          <div>
            <h3 className="font-semibold text-theme-primary">{node.label}</h3>
            <span className="text-xs text-theme-muted capitalize">{node.type}</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-white/10 rounded-lg transition-colors"
        >
          <X className="w-4 h-4 text-slate-400" />
        </button>
      </div>

      {node.type === 'drug' && (
        <div className="space-y-2 text-sm">
          <div className="flex justify-between py-2 border-b border-white/5">
            <span className="text-theme-muted">ID</span>
            <span className="text-theme-secondary font-mono text-xs">{node.id}</span>
          </div>
          {node.isHub && (
            <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <p className="text-xs text-amber-400">
                <strong>Hub Drug:</strong> This drug has the most interactions in the current regimen.
              </p>
            </div>
          )}
        </div>
      )}

      {node.type === 'enzyme' && (
        <p className="text-xs text-slate-400 mt-2">
          Cytochrome P450 enzyme involved in drug metabolism. Inhibition or induction of this enzyme can significantly alter drug levels.
        </p>
      )}

      {node.type === 'target' && (
        <p className="text-xs text-slate-400 mt-2">
          Pharmacological target protein. Multiple drugs binding to the same target can lead to synergistic or antagonistic effects.
        </p>
      )}
    </motion.div>
  );
}

export default function KnowledgeGraphView({ drugs = [], result, polypharmacyResult }) {
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodePositions, setNodePositions] = useState({});
  const svgRef = useRef(null);

  const { nodes, edges } = useMemo(
    () => generateGraphData(drugs, result, polypharmacyResult),
    [drugs, result, polypharmacyResult]
  );

  // Update positions when nodes are dragged
  const handleNodeDrag = useCallback((nodeId, x, y) => {
    setNodePositions(prev => ({
      ...prev,
      [nodeId]: { x, y }
    }));
  }, []);

  // Apply dragged positions to nodes
  const positionedNodes = useMemo(() => {
    return nodes.map(node => ({
      ...node,
      x: nodePositions[node.id]?.x ?? node.x,
      y: nodePositions[node.id]?.y ?? node.y
    }));
  }, [nodes, nodePositions]);

  if (drugs.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 rounded-3xl bg-theme-secondary flex items-center justify-center mx-auto mb-4">
            <Activity className="w-10 h-10 text-theme-dim" />
          </div>
          <p className="text-sm text-theme-muted">Add drugs to visualize the knowledge graph</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative overflow-hidden">
      {/* Background grid */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: 'radial-gradient(circle, #3B82F6 1px, transparent 1px)',
          backgroundSize: '30px 30px'
        }}
      />

      <svg
        ref={svgRef}
        className="w-full h-full"
        viewBox="0 0 800 600"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Defs for markers */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="10"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
          </marker>
        </defs>

        {/* Edges */}
        {edges.map((edge, i) => (
          <GraphEdge key={i} edge={edge} nodes={positionedNodes} />
        ))}

        {/* Nodes */}
        {positionedNodes.map((node) => (
          <GraphNode
            key={node.id}
            node={node}
            isSelected={selectedNode?.id === node.id}
            onSelect={setSelectedNode}
            onDrag={handleNodeDrag}
          />
        ))}
      </svg>

      {/* Node info panel */}
      <AnimatePresence>
        {selectedNode && (
          <NodeInfoPanel
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </AnimatePresence>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex items-center gap-4">
        <div className="flex items-center gap-3 px-4 py-2 bg-theme-panel backdrop-blur-sm rounded-xl border border-theme">
          {Object.entries(nodeTypes).slice(0, 4).map(([type, config]) => (
            <div key={type} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: config.color }}
              />
              <span className="text-xs text-theme-muted capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="absolute top-4 left-4 px-4 py-2 bg-theme-panel backdrop-blur-sm rounded-lg border border-theme text-xs text-theme-muted">
        Click nodes for details • Drag to reposition
      </div>

      {/* Interaction summary */}
      {result && result.severity !== 'no_interaction' && (
        <div className="absolute bottom-4 right-4 px-4 py-3 bg-theme-panel backdrop-blur-sm rounded-xl border border-theme">
          <div className="flex items-center gap-2 mb-1">
            <AlertTriangle className="w-4 h-4 text-orange-400" />
            <span className="text-sm font-medium text-theme-primary">Interaction Detected</span>
          </div>
          <p className="text-xs text-theme-muted">
            Risk Level: <span className="text-orange-400 capitalize">{result.risk_level || result.severity}</span>
          </p>
        </div>
      )}
    </div>
  );
}
