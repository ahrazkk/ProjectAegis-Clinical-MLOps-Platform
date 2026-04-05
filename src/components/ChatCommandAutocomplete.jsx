import { useState, useEffect, useRef, useCallback } from 'react';
import { searchDrugs } from '../services/api';

const COMMANDS = [
    { command: '/test', description: 'Test interaction between two drugs', usage: '/test warfarin aspirin' },
    { command: '/poly', description: 'Analyze polypharmacy risk', usage: '/poly warfarin,aspirin,ibuprofen' },
    { command: '/compare', description: 'Compare drugs side-by-side', usage: '/compare metformin glipizide' },
    { command: '/alt', description: 'Find safer alternatives', usage: '/alt ibuprofen' },
    { command: '/evidence', description: 'Show evidence chain for drug pair', usage: '/evidence warfarin aspirin' },
    { command: '/research', description: 'Deep research a drug or pair (KG + FAERS)', usage: '/research warfarin aspirin' },
    { command: '/mutate', description: 'Find safest drug to remove from regimen', usage: '/mutate warfarin aspirin ibuprofen' },
    { command: '/current', description: 'Analyze drugs selected in sidebar', usage: '/current' },
    { command: '/demo', description: 'Run a pre-built demo case', usage: '/demo cardiac' },
    { command: '/class', description: 'Group drugs by class & check warnings', usage: '/class warfarin aspirin ibuprofen' },
];

/**
 * Autocomplete dropdown for slash commands and drug names in chat input.
 *
 * Shows command list when input starts with '/'.
 * Shows drug suggestions when typing after a command.
 */
export default function ChatCommandAutocomplete({ inputValue, onSelect, visible }) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [drugSuggestions, setDrugSuggestions] = useState([]);
    const [mode, setMode] = useState('none'); // 'command' | 'drug' | 'none'
    const listRef = useRef(null);
    const debounceRef = useRef(null);

    // Determine what to show based on input
    useEffect(() => {
        if (!inputValue || !visible) {
            setMode('none');
            setDrugSuggestions([]);
            return;
        }

        const trimmed = inputValue.trim();

        if (trimmed === '/') {
            // Show all commands
            setMode('command');
            setSelectedIndex(0);
            return;
        }

        if (trimmed.startsWith('/') && !trimmed.includes(' ')) {
            // Filtering commands by partial match
            setMode('command');
            setSelectedIndex(0);
            return;
        }

        if (trimmed.startsWith('/') && trimmed.includes(' ')) {
            // After command + space, suggest drugs
            const parts = trimmed.split(/\s+/);
            const lastPart = parts[parts.length - 1];
            // Also handle comma-separated args
            const lastArg = lastPart.split(',').pop().trim();

            if (lastArg && lastArg.length >= 2 && !lastArg.startsWith('/')) {
                setMode('drug');
                // Debounce drug search
                clearTimeout(debounceRef.current);
                debounceRef.current = setTimeout(async () => {
                    try {
                        const result = await searchDrugs(lastArg);
                        const suggestions = (result.results || result || []).slice(0, 6);
                        setDrugSuggestions(suggestions);
                        setSelectedIndex(0);
                    } catch {
                        setDrugSuggestions([]);
                    }
                }, 200);
            } else {
                setMode('none');
                setDrugSuggestions([]);
            }
            return;
        }

        setMode('none');
        setDrugSuggestions([]);
    }, [inputValue, visible]);

    // Get filtered items based on mode
    const getItems = useCallback(() => {
        if (mode === 'command') {
            const partial = inputValue.trim().slice(1).toLowerCase();
            if (!partial) return COMMANDS;
            return COMMANDS.filter(c => c.command.slice(1).startsWith(partial));
        }
        if (mode === 'drug') {
            return drugSuggestions;
        }
        return [];
    }, [mode, inputValue, drugSuggestions]);

    const items = getItems();

    // Keyboard navigation
    const handleKeyDown = useCallback((e) => {
        if (mode === 'none' || items.length === 0) return false;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIndex(prev => Math.min(prev + 1, items.length - 1));
            return true;
        }
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIndex(prev => Math.max(prev - 1, 0));
            return true;
        }
        if (e.key === 'Tab' || e.key === 'Enter') {
            if (items.length > 0 && selectedIndex < items.length) {
                e.preventDefault();
                handleSelect(items[selectedIndex]);
                return true;
            }
        }
        if (e.key === 'Escape') {
            setMode('none');
            return true;
        }
        return false;
    }, [mode, items, selectedIndex]);

    const handleSelect = (item) => {
        if (mode === 'command') {
            // Insert command with trailing space
            onSelect(item.command + ' ', 'command');
        } else if (mode === 'drug') {
            // Replace the last partial drug name with the selected one
            const name = typeof item === 'string' ? item : (item.name || item);
            const parts = inputValue.trim().split(/\s+/);
            const lastPart = parts[parts.length - 1];

            // Handle comma-separated
            if (lastPart.includes(',')) {
                const commaArgs = lastPart.split(',');
                commaArgs[commaArgs.length - 1] = name;
                parts[parts.length - 1] = commaArgs.join(',');
            } else {
                parts[parts.length - 1] = name;
            }

            onSelect(parts.join(' ') + ' ', 'drug');
        }
        setMode('none');
    };

    // Expose keyboard handler for parent to wire into input
    useEffect(() => {
        // Attach to window for the chat input -- parent should call handleKeyDown
        // We expose it via the ref pattern or the parent can check mode
    }, []);

    if (mode === 'none' || items.length === 0 || !visible) return null;

    return (
        <div
            ref={listRef}
            className="absolute bottom-full left-0 right-0 mb-1 bg-gray-900 border border-gray-700 rounded-lg shadow-xl z-50 max-h-48 overflow-y-auto"
            onKeyDown={handleKeyDown}
        >
            {mode === 'command' && items.map((cmd, i) => (
                <button
                    key={cmd.command}
                    className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                        i === selectedIndex
                            ? 'bg-cyan-900/40 text-cyan-300'
                            : 'text-gray-300 hover:bg-gray-800'
                    }`}
                    onClick={() => handleSelect(cmd)}
                    onMouseEnter={() => setSelectedIndex(i)}
                >
                    <span className="font-mono text-cyan-400">{cmd.command}</span>
                    <span className="ml-2 text-gray-400">{cmd.description}</span>
                    <span className="block text-xs text-gray-600 mt-0.5">{cmd.usage}</span>
                </button>
            ))}

            {mode === 'drug' && items.map((drug, i) => {
                const name = typeof drug === 'string' ? drug : (drug.name || drug);
                const extra = typeof drug === 'object' ? (drug.drugbank_id || drug.id || '') : '';
                return (
                    <button
                        key={name + i}
                        className={`w-full text-left px-3 py-1.5 text-sm transition-colors ${
                            i === selectedIndex
                                ? 'bg-cyan-900/40 text-cyan-300'
                                : 'text-gray-300 hover:bg-gray-800'
                        }`}
                        onClick={() => handleSelect(drug)}
                        onMouseEnter={() => setSelectedIndex(i)}
                    >
                        <span className="font-medium">{name}</span>
                        {extra && <span className="ml-2 text-xs text-gray-500">{extra}</span>}
                    </button>
                );
            })}
        </div>
    );
}

// Export the keyboard handler hook for parent component use
export function useChatAutocompleteKeyboard(autocompleteRef) {
    return useCallback((e) => {
        if (autocompleteRef.current?.handleKeyDown) {
            return autocompleteRef.current.handleKeyDown(e);
        }
        return false;
    }, [autocompleteRef]);
}
