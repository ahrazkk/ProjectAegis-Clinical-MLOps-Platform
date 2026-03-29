// SearchBar.jsx — Drug search within the viewer
import React, { useState, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { rawGnnData } from '../graphEngine';
import { useGalaxy, useGalaxyDispatch } from '../store';

export default function SearchBar({ nodeDict }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const inputRef = useRef(null);
  const { drugA, drugB } = useGalaxy();
  const dispatch = useGalaxyDispatch();

  useEffect(() => {
    if (!query || query.length < 2) {
      setResults([]);
      return;
    }
    const lower = query.toLowerCase();
    const matches = rawGnnData.nodes
      .filter(n => n.name.toLowerCase().includes(lower))
      .slice(0, 8);
    setResults(matches);
  }, [query]);

  const handleSelect = (node) => {
    const fullNode = nodeDict[node.id];
    if (!fullNode) return;

    if (!drugA) {
      dispatch({ type: 'SELECT_DRUG_A', payload: fullNode });
    } else if (!drugB) {
      dispatch({ type: 'SELECT_DRUG_B', payload: fullNode });
    } else {
      // Replace drug A
      dispatch({ type: 'SELECT_DRUG_A', payload: fullNode });
    }

    setQuery('');
    setResults([]);
    setIsOpen(false);
  };

  const clearAll = () => {
    dispatch({ type: 'CLEAR_SELECTION' });
    setQuery('');
  };

  return (
    <div className="absolute top-12 left-1/2 -translate-x-1/2 z-20 w-64 sm:w-80">
      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-white/30" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => { setQuery(e.target.value); setIsOpen(true); }}
          onFocus={() => setIsOpen(true)}
          placeholder={!drugA ? 'Search Drug A...' : !drugB ? 'Search Drug B...' : 'Replace drug...'}
          className="w-full bg-black/70 backdrop-blur-md border border-white/10 rounded-md pl-8 pr-8 py-1.5 text-[11px] text-white placeholder:text-white/25 font-mono focus:outline-none focus:border-purple-500/40 transition-colors"
        />
        {(drugA || drugB) && (
          <button onClick={clearAll} className="absolute right-2 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Selected drugs chips */}
      {(drugA || drugB) && (
        <div className="flex gap-1.5 mt-1.5">
          {drugA && (
            <div className="bg-[#00d2ff]/10 border border-[#00d2ff]/30 text-[#00d2ff] px-2 py-0.5 rounded text-[9px] font-mono flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00d2ff]" />
              {drugA.name}
              <button onClick={() => dispatch({ type: 'SELECT_DRUG_A', payload: null })} className="ml-1 opacity-50 hover:opacity-100">
                <X className="w-2.5 h-2.5" />
              </button>
            </div>
          )}
          {drugB && (
            <div className="bg-[#ff8c00]/10 border border-[#ff8c00]/30 text-[#ff8c00] px-2 py-0.5 rounded text-[9px] font-mono flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#ff8c00]" />
              {drugB.name}
              <button onClick={() => dispatch({ type: 'SELECT_DRUG_B', payload: null })} className="ml-1 opacity-50 hover:opacity-100">
                <X className="w-2.5 h-2.5" />
              </button>
            </div>
          )}
        </div>
      )}

      {/* Dropdown results */}
      {isOpen && results.length > 0 && (
        <div className="mt-1 bg-black/80 backdrop-blur-md border border-white/10 rounded-md overflow-hidden shadow-xl">
          {results.map(node => (
            <button
              key={node.id}
              onClick={() => handleSelect(node)}
              className="w-full text-left px-3 py-2 text-[11px] text-white/70 hover:bg-white/5 hover:text-white transition-colors flex justify-between items-center border-b border-white/5 last:border-0"
            >
              <span className="font-medium">{node.name}</span>
              <span className="text-[8px] text-white/25 font-mono">{node.type || 'Unknown'}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
