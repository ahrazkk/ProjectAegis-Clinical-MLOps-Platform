import React, { createContext, useContext, useState, useCallback } from 'react';

const SystemLogsContext = createContext();

export function SystemLogsProvider({ children }) {
    const [logs, setLogs] = useState([]);
    const [isOpen, setIsOpen] = useState(false);

    const addLog = useCallback((message, type = 'info', source = 'SYSTEM') => {
        const timestamp = new Date().toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            fractionalSecondDigits: 3
        });

        setLogs(prev => [...prev, {
            id: Date.now() + Math.random(),
            timestamp,
            message,
            type, // 'info', 'success', 'warning', 'error'
            source
        }].slice(-100)); // Keep last 100 logs
    }, []);

    const toggleMonitor = useCallback(() => setIsOpen(prev => !prev), []);
    const clearLogs = useCallback(() => setLogs([]), []);

    return (
        <SystemLogsContext.Provider value={{ logs, addLog, isOpen, toggleMonitor, clearLogs }}>
            {children}
        </SystemLogsContext.Provider>
    );
}

export function useSystemLogs() {
    return useContext(SystemLogsContext);
}
