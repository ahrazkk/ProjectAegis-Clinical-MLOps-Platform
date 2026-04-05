import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPageV2';
import Dashboard from './pages/Dashboard';
import ResearchPage from './pages/ResearchPage';
import SettingsPage from './pages/SettingsPage';
import CorrectionsPage from './pages/CorrectionsPage';
import { SystemLogsProvider } from './hooks/useSystemLogs';
import { ThemeProvider } from './hooks/useTheme';
export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <SystemLogsProvider>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/research" element={<ResearchPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/corrections" element={<CorrectionsPage />} />
          </Routes>
        </SystemLogsProvider>
      </Router>
    </ThemeProvider>
  );
}