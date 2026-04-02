import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ChevronLeft, Settings, Database, Brain, 
  Trash2, Shield, Activity, Save, AlertTriangle
} from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import {
  DEMO_DRUG_GROUPS,
  DEMO_MODE_SETTING_KEY,
  USER_PREFERENCES_STORAGE_KEY,
  readDemoModeSetting,
  readUserPreferences,
  writeDemoModeSetting,
  writeUserPreferences,
} from '../config/demoMode';

const SettingsPage = () => {
  const navigate = useNavigate();
  const { theme, toggleTheme, setTheme } = useTheme();

  const savedPreferences = useMemo(() => readUserPreferences() || {}, []);
  
  const [activeTab, setActiveTab] = useState('appearance');
  const [inferenceMode, setInferenceMode] = useState(() =>
    typeof savedPreferences.inferenceMode === 'string' ? savedPreferences.inferenceMode : 'graphsage'
  );
  const [threshold, setThreshold] = useState(() => {
    const value = Number(savedPreferences.threshold);
    if (!Number.isFinite(value)) return 85;
    return Math.min(99, Math.max(50, value));
  });
  const [privacyMode, setPrivacyMode] = useState(() => Boolean(savedPreferences.privacyMode));
  const [demoModeEnabled, setDemoModeEnabled] = useState(() => readDemoModeSetting());

  const handleClearCache = () => {
    // Basic cache clearing for demo purposes
    localStorage.removeItem('ddi-theme'); // Clear theme mapping as a test
    localStorage.removeItem('ddi-dashboard-state'); // Assuming we add this later
    localStorage.removeItem(DEMO_MODE_SETTING_KEY);
    localStorage.removeItem(USER_PREFERENCES_STORAGE_KEY);
    setInferenceMode('graphsage');
    setThreshold(85);
    setPrivacyMode(false);
    setDemoModeEnabled(false);
    alert('System Cache Cleared. Temporary storage purged.');
  };

  const handleSaveConfiguration = () => {
    writeDemoModeSetting(demoModeEnabled);
    writeUserPreferences({
      inferenceMode,
      threshold: Number(threshold),
      privacyMode,
      demoModeEnabled,
      updatedAt: new Date().toISOString(),
    });
    alert('Settings saved locally.');
  };

  const handleDemoModeToggle = () => {
    setDemoModeEnabled((prev) => {
      const next = !prev;
      writeDemoModeSetting(next);
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-theme-primary text-theme-primary font-mono relative overflow-hidden">
      {/* Blueprint Grid Background */}
      <div className="fixed inset-0 pointer-events-none opacity-20">
        <div className="absolute inset-0 bg-grid-blueprint bg-grid-40" />
        <div className="absolute inset-0 bg-grid-fine bg-grid-8" />
      </div>

      <header className="h-14 border-b border-theme bg-theme-panel sticky top-0 z-50">
        <div className="max-w-7xl mx-auto h-full flex items-center px-4 md:px-6">
          <button 
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 text-theme-muted hover:text-theme-secondary transition-colors group"
          >
            <ChevronLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
            <span className="text-xs tracking-widest uppercase">Back to System</span>
          </button>
          
          <div className="ml-auto flex items-center gap-4">
            <h1 className="text-xs uppercase tracking-widest text-theme-secondary flex items-center gap-2">
              <Settings className="w-4 h-4" /> User Preferences
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8 relative z-10">
        <div className="flex flex-col md:flex-row gap-8">
          
          {/* Sidebar */}
          <aside className="w-full md:w-64 shrink-0">
            <div className="border border-theme bg-theme-panel">
              <nav className="flex flex-col">
                <button 
                  onClick={() => setActiveTab('appearance')}
                  className={`flex items-center gap-3 p-4 text-left text-sm transition-colors border-b border-theme ${activeTab === 'appearance' ? 'bg-theme-accent/5 text-theme-accent' : 'text-theme-muted hover:text-theme-secondary hover:bg-theme-panel-elevated'}`}
                >
                  <Activity className="w-4 h-4" /> Appearance
                </button>
                <button 
                  onClick={() => setActiveTab('engine')}
                  className={`flex items-center gap-3 p-4 text-left text-sm transition-colors border-b border-theme ${activeTab === 'engine' ? 'bg-theme-accent/5 text-theme-accent' : 'text-theme-muted hover:text-theme-secondary hover:bg-theme-panel-elevated'}`}
                >
                  <Brain className="w-4 h-4" /> Inference Engine
                </button>
                <button 
                  onClick={() => setActiveTab('privacy')}
                  className={`flex items-center gap-3 p-4 text-left text-sm transition-colors ${activeTab === 'privacy' ? 'bg-theme-accent/5 text-theme-accent' : 'text-theme-muted hover:text-theme-secondary hover:bg-theme-panel-elevated'}`}
                >
                  <Shield className="w-4 h-4" /> Privacy & Data
                </button>
              </nav>
            </div>
          </aside>

          {/* Settings Content */}
          <section className="flex-1 space-y-6">
            
            {activeTab === 'appearance' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <div className="border border-theme bg-theme-panel p-6">
                  <h2 className="text-sm uppercase tracking-widest text-theme-secondary mb-6 flex items-center gap-2 border-b border-theme pb-4">
                    Visual Styling
                  </h2>
                  
                  <div className="space-y-6">
                    <div>
                      <label className="block text-xs uppercase tracking-widest text-theme-muted mb-3">System Theme</label>
                      <div className="flex gap-4">
                        <button 
                          onClick={() => setTheme('dark')}
                          className={`flex-1 py-3 border transition-colors ${theme === 'dark' ? 'border-theme-accent text-theme-accent bg-theme-accent/5' : 'border-theme text-theme-muted hover:border-theme-highlight'}`}
                        >
                          Dark Tactical
                        </button>
                        <button 
                          onClick={() => setTheme('light')}
                          className={`flex-1 py-3 border transition-colors ${theme === 'light' ? 'border-theme-accent text-theme-accent bg-theme-accent/5' : 'border-theme text-theme-muted hover:border-theme-highlight'}`}
                        >
                          Light Clinical
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'engine' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <div className="border border-theme bg-theme-panel p-6">
                  <h2 className="text-sm uppercase tracking-widest text-theme-secondary mb-6 flex items-center gap-2 border-b border-theme pb-4">
                    AI Model & Inference
                  </h2>
                  
                  <div className="space-y-8">
                    <div>
                      <label className="block text-xs uppercase tracking-widest text-theme-muted mb-3">Default Model Engine</label>
                      <div className="space-y-3">
                        <label className={`flex items-start gap-3 p-4 border transition-colors cursor-pointer ${inferenceMode === 'graphsage' ? 'border-theme-accent bg-theme-accent/5' : 'border-theme hover:border-theme-highlight'}`}>
                          <input 
                            type="radio" 
                            name="engine" 
                            checked={inferenceMode === 'graphsage'} 
                            onChange={() => setInferenceMode('graphsage')}
                            className="mt-1"
                          />
                          <div>
                            <div className={`text-sm ${inferenceMode === 'graphsage' ? 'text-theme-accent' : 'text-theme-secondary'}`}>Macroscopic GraphSAGE</div>
                            <div className="text-xs text-theme-muted mt-1 leading-relaxed">Fastest inference. Treats whole drugs as dense biological node vectors via TWOSIDES polypharmacy expansion. Highest benchmark AUC (98.67%).</div>
                          </div>
                        </label>

                        <label className={`flex items-start gap-3 p-4 border transition-colors cursor-pointer ${inferenceMode === 'pubmedbert' ? 'border-theme-accent bg-theme-accent/5' : 'border-theme hover:border-theme-highlight'}`}>
                          <input 
                            type="radio" 
                            name="engine" 
                            checked={inferenceMode === 'pubmedbert'} 
                            onChange={() => setInferenceMode('pubmedbert')}
                            className="mt-1"
                          />
                          <div>
                            <div className={`text-sm ${inferenceMode === 'pubmedbert' ? 'text-theme-accent' : 'text-theme-secondary'}`}>PubMedBERT NLP (Legacy)</div>
                            <div className="text-xs text-theme-muted mt-1 leading-relaxed">Text-Based Inference. Cross-references 27,792 medical literature text pairs. May result in high API latency but useful for semantic cross-checks.</div>
                          </div>
                        </label>
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs uppercase tracking-widest text-theme-muted mb-3">
                        Confidence Threshold ({threshold}%)
                      </label>
                      <p className="text-xs text-theme-muted mb-4">
                        Any drug-drug interaction prediction below this probability will not be flagged as a critical risk.
                      </p>
                      <input 
                        type="range" 
                        min="50" max="99" 
                        value={threshold} 
                        onChange={(e) => setThreshold(e.target.value)}
                        className="w-full accent-theme-accent"
                      />
                    </div>

                    <div className="border border-theme bg-theme-panel-elevated p-4 space-y-3">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="flex items-center gap-2 text-sm text-theme-secondary">
                            <Database className="w-4 h-4 text-theme-accent" /> Demo Mode: Bulk Candidate Groups
                          </div>
                          <p className="text-xs text-theme-muted mt-1 leading-relaxed max-w-xl">
                            When enabled, Dashboard shows curated grouped drug categories for one-click mass selection during demos.
                            Default is OFF so normal workflows remain unchanged.
                          </p>
                        </div>

                        <button
                          onClick={handleDemoModeToggle}
                          className={`w-12 h-6 border relative transition-colors ${demoModeEnabled ? 'bg-theme-accent border-theme-accent' : 'bg-transparent border-theme'}`}
                          aria-label="Toggle demo mode"
                        >
                          <div className={`absolute top-1 w-4 h-4 transition-transform ${demoModeEnabled ? 'left-7 bg-black' : 'left-1 bg-theme-muted'}`} />
                        </button>
                      </div>

                      <div className="pt-2 border-t border-theme/40">
                        <p className="text-[10px] text-theme-muted uppercase tracking-wider mb-2">Available Demo Groups</p>
                        <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
                          {DEMO_DRUG_GROUPS.map((group) => (
                            <div key={group.id} className="p-2 border border-theme/50 bg-theme-panel/60">
                              <p className="text-[10px] text-theme-secondary uppercase tracking-wider">{group.title}</p>
                              <p className="text-[10px] text-theme-dim mt-1 leading-relaxed">{group.reason}</p>
                              <p className="text-[9px] text-theme-muted mt-1 uppercase tracking-wider">
                                {group.drugs.join(' • ')}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'privacy' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
                <div className="border border-theme bg-theme-panel p-6">
                  <h2 className="text-sm uppercase tracking-widest text-theme-secondary mb-6 flex items-center gap-2 border-b border-theme pb-4">
                    Data & Privacy
                  </h2>
                  
                  <div className="space-y-8">
                    <div className="flex items-start justify-between border border-theme p-4 bg-theme-panel-elevated">
                      <div>
                        <div className="flex items-center gap-2 text-sm text-theme-secondary">
                          <Shield className="w-4 h-4 text-theme-accent" /> Clinical Privacy Mode
                        </div>
                        <p className="text-xs text-theme-muted mt-1 leading-relaxed max-w-md">
                          When enabled, zeroes out all interactions, search history, and prediction caches immediately upon tab close to ensure HIPAA-level compliance.
                        </p>
                      </div>
                      <button 
                        onClick={() => setPrivacyMode(!privacyMode)}
                        className={`w-12 h-6 border ${privacyMode ? 'bg-theme-accent border-theme-accent' : 'bg-transparent border-theme'} relative transition-colors`}
                      >
                        <div className={`absolute top-1 w-4 h-4 bg-white transition-transform ${privacyMode ? 'left-7 bg-black' : 'left-1 bg-theme-muted'}`} />
                      </button>
                    </div>

                    <div className="border border-red-500/30 bg-red-500/5 p-4 space-y-4">
                      <div className="flex items-center gap-2 text-sm text-red-500">
                        <AlertTriangle className="w-4 h-4" /> Danger Zone
                      </div>
                      <p className="text-xs text-red-400/80 leading-relaxed max-w-md">
                        This action will clear your entire local system cache including preferences, themes, and pending visual pill scans.
                      </p>
                      <button 
                        onClick={handleClearCache}
                        className="flex items-center gap-2 px-4 py-2 border border-red-500/50 text-red-500 text-xs hover:bg-red-500/10 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" /> Wipe Cache History
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            <div className="flex justify-end pt-4">
              <button 
                onClick={handleSaveConfiguration}
                className="flex items-center gap-2 px-6 py-2 border border-theme-accent bg-theme-accent text-black text-xs uppercase tracking-widest font-bold hover:bg-theme-accent/90 transition-colors"
              >
                <Save className="w-4 h-4" /> Save Configuration
              </button>
            </div>

          </section>
        </div>
      </main>
    </div>
  );
};

export default SettingsPage;