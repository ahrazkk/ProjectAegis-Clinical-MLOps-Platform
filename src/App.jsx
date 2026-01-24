import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPageV2';
import Dashboard from './pages/Dashboard';
import { SystemLogsProvider } from './hooks/useSystemLogs';
import { ThemeProvider } from './hooks/useTheme';
import SystemMonitor from './components/SystemMonitor';

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <SystemLogsProvider>
          <SystemMonitor />
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </SystemLogsProvider>
      </Router>
    </ThemeProvider>
  );
}