import React, { useState } from 'react';
import Dashboard from './Dashboard';
import LandingPage from './LandingPage.jsx';

export default function App() {
  const [view, setView] = useState('landing'); // 'landing' or 'dashboard'

  return (
    <>
      {view === 'landing' ? (
        <LandingPage onEnter={() => setView('dashboard')} />
      ) : (
        <Dashboard />
      )}
    </>
  );
}