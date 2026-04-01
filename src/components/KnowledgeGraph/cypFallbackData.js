// cypFallbackData.js — Static CYP450 data for offline mode
// Extracted from backend cyp450_database.py. Used only when API is unavailable.
// CYP enzyme profiles are stable biochemical facts — this is safe to hardcode.

// Format: drug_name (lowercase) → { enzyme: [roles] }
export const CYP_DRUG_DATA = {
  // CYP1A2
  caffeine: { CYP1A2: ['substrate'] },
  theophylline: { CYP1A2: ['substrate'] },
  clozapine: { CYP1A2: ['substrate'] },
  olanzapine: { CYP1A2: ['substrate'] },
  fluvoxamine: { CYP1A2: ['inhibitor_strong'], CYP2C19: ['inhibitor_strong'], CYP3A4: ['inhibitor_moderate'] },
  ciprofloxacin: { CYP1A2: ['inhibitor_strong'] },

  // CYP2C9
  warfarin: { CYP2C9: ['substrate'], CYP3A4: ['substrate'] },
  celecoxib: { CYP2C9: ['substrate'] },
  ibuprofen: { CYP2C9: ['substrate'] },
  naproxen: { CYP2C9: ['substrate'] },
  diclofenac: { CYP2C9: ['substrate'] },
  losartan: { CYP2C9: ['substrate'], CYP3A4: ['substrate'] },
  glipizide: { CYP2C9: ['substrate'] },
  fluconazole: { CYP2C9: ['inhibitor_moderate'], CYP2C19: ['inhibitor_moderate'], CYP3A4: ['inhibitor_moderate'] },
  amiodarone: { CYP2C9: ['inhibitor_moderate'], CYP2D6: ['inhibitor_moderate'], CYP3A4: ['inhibitor_moderate'] },
  voriconazole: { CYP2C9: ['inhibitor_moderate'], CYP2C19: ['inhibitor_strong'], CYP3A4: ['inhibitor_strong'] },

  // CYP2C19
  clopidogrel: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  omeprazole: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  esomeprazole: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  lansoprazole: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  citalopram: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  escitalopram: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },
  diazepam: { CYP2C19: ['substrate'], CYP3A4: ['substrate'] },

  // CYP2D6
  codeine: { CYP2D6: ['substrate'] },
  tramadol: { CYP2D6: ['substrate'] },
  oxycodone: { CYP2D6: ['substrate'] },
  metoprolol: { CYP2D6: ['substrate'] },
  carvedilol: { CYP2D6: ['substrate'] },
  venlafaxine: { CYP2D6: ['substrate'] },
  paroxetine: { CYP2D6: ['substrate', 'inhibitor_strong'] },
  fluoxetine: { CYP2C19: ['inhibitor_moderate'], CYP2D6: ['inhibitor_strong'], CYP3A4: ['inhibitor_weak'] },
  haloperidol: { CYP2D6: ['substrate'] },
  risperidone: { CYP2D6: ['substrate'] },
  aripiprazole: { CYP2D6: ['substrate'], CYP3A4: ['substrate'] },
  tamoxifen: { CYP2D6: ['substrate'] },
  bupropion: { CYP2D6: ['inhibitor_strong'] },
  quinidine: { CYP2D6: ['inhibitor_strong'] },
  sertraline: { CYP2D6: ['inhibitor_moderate'] },
  duloxetine: { CYP1A2: ['substrate'], CYP2D6: ['substrate', 'inhibitor_moderate'] },

  // CYP3A4
  simvastatin: { CYP3A4: ['substrate'] },
  lovastatin: { CYP3A4: ['substrate'] },
  atorvastatin: { CYP3A4: ['substrate'] },
  amlodipine: { CYP3A4: ['substrate'] },
  nifedipine: { CYP3A4: ['substrate'] },
  diltiazem: { CYP3A4: ['substrate', 'inhibitor_moderate'] },
  verapamil: { CYP3A4: ['substrate', 'inhibitor_moderate'] },
  cyclosporine: { CYP3A4: ['substrate', 'inhibitor_weak'] },
  tacrolimus: { CYP3A4: ['substrate'] },
  midazolam: { CYP3A4: ['substrate'] },
  alprazolam: { CYP3A4: ['substrate'] },
  fentanyl: { CYP3A4: ['substrate'] },
  sildenafil: { CYP3A4: ['substrate'] },
  apixaban: { CYP3A4: ['substrate'] },
  rivaroxaban: { CYP3A4: ['substrate'] },
  ketoconazole: { CYP3A4: ['inhibitor_strong'] },
  itraconazole: { CYP3A4: ['inhibitor_strong'] },
  clarithromycin: { CYP3A4: ['inhibitor_strong'] },
  erythromycin: { CYP3A4: ['inhibitor_moderate'] },
  ritonavir: { CYP3A4: ['inhibitor_strong'], CYP2D6: ['inhibitor_strong'] },

  // Inducers (multi-CYP)
  rifampin: { CYP1A2: ['inducer'], CYP2B6: ['inducer'], CYP2C8: ['inducer'], CYP2C9: ['inducer'], CYP2C19: ['inducer'], CYP3A4: ['inducer'] },
  carbamazepine: { CYP1A2: ['inducer'], CYP2B6: ['inducer'], CYP2C9: ['inducer'], CYP3A4: ['substrate', 'inducer'] },
  phenytoin: { CYP1A2: ['inducer'], CYP2B6: ['inducer'], CYP2C9: ['inducer', 'substrate'], CYP2C19: ['inducer', 'substrate'], CYP3A4: ['inducer'] },
  phenobarbital: { CYP1A2: ['inducer'], CYP2B6: ['inducer'], CYP2C9: ['inducer'], CYP3A4: ['inducer'] },

  // Not CYP metabolized (included for completeness)
  aspirin: {},
  metformin: {},
  lisinopril: {},
  gabapentin: {},
  levothyroxine: {},
  digoxin: {},
};


// ─── Helper: get CYP profile for a drug (offline) ──────────────────────────
export function getOfflineCYPProfile(drugName) {
  return CYP_DRUG_DATA[drugName.toLowerCase().trim()] || null;
}

// ─── Helper: build offline biology response ─────────────────────────────────
// Returns same shape as fetchDrugBiology() API response.
// Offline mode only has CYP data — targets and side effects come from the API.
export function buildOfflineDrugBiology(drugName) {
  const profile = getOfflineCYPProfile(drugName);
  if (!profile) return null;

  const cyp_metabolism = { substrates: [], inhibitors: [], inducers: [] };
  for (const [enzyme, roles] of Object.entries(profile)) {
    for (const role of roles) {
      if (role === 'substrate') {
        cyp_metabolism.substrates.push(enzyme);
      } else if (role.includes('inhibitor')) {
        const strength = role.includes('strong') ? 'strong' : role.includes('moderate') ? 'moderate' : 'weak';
        cyp_metabolism.inhibitors.push({ enzyme, strength });
      } else if (role === 'inducer') {
        cyp_metabolism.inducers.push(enzyme);
      }
    }
  }

  return {
    drug: { id: '', name: drugName, therapeutic_class: '' },
    cyp_metabolism,
    targets: [],
    side_effects: [],
    interaction_count: 0,
    source: 'offline',
  };
}

// ─── Helper: build offline mechanism map ────────────────────────────────────
export function buildOfflineMechanismMap(drug1Name, drug2Name) {
  const p1 = getOfflineCYPProfile(drug1Name);
  const p2 = getOfflineCYPProfile(drug2Name);
  if (!p1 && !p2) return null;

  const profile1 = p1 || {};
  const profile2 = p2 || {};

  const commonEnzymes = new Set([
    ...Object.keys(profile1),
    ...Object.keys(profile2),
  ].filter(e => e in profile1 && e in profile2));

  const shared_enzymes = [];
  for (const enzyme of [...commonEnzymes].sort()) {
    const roles1 = profile1[enzyme] || [];
    const roles2 = profile2[enzyme] || [];

    const sub1 = roles1.includes('substrate');
    const sub2 = roles2.includes('substrate');
    const inh1 = roles1.some(r => r.includes('inhibitor'));
    const inh2 = roles2.some(r => r.includes('inhibitor'));
    const ind1 = roles1.includes('inducer');
    const ind2 = roles2.includes('inducer');

    let risk = '';
    let risk_level = 'low';
    if (sub1 && inh2) { risk = `${drug2Name} inhibits ${drug1Name} metabolism via ${enzyme}`; risk_level = 'high'; }
    else if (sub2 && inh1) { risk = `${drug1Name} inhibits ${drug2Name} metabolism via ${enzyme}`; risk_level = 'high'; }
    else if (sub1 && ind2) { risk = `${drug2Name} induces ${enzyme}, reducing ${drug1Name} levels`; risk_level = 'high'; }
    else if (sub2 && ind1) { risk = `${drug1Name} induces ${enzyme}, reducing ${drug2Name} levels`; risk_level = 'high'; }
    else if (sub1 && sub2) { risk = `Both compete for ${enzyme}`; risk_level = 'moderate'; }

    shared_enzymes.push({
      enzyme,
      drug1_role: roles1.join(', '),
      drug2_role: roles2.join(', '),
      risk,
      risk_level,
    });
  }

  const cyp_conflicts = shared_enzymes.filter(e => e.risk_level !== 'low').length;

  return {
    drugs: [
      { id: '', name: drug1Name, therapeutic_class: '' },
      { id: '', name: drug2Name, therapeutic_class: '' },
    ],
    shared_enzymes,
    cyp_interactions: [],
    shared_targets: [],
    shared_side_effects: [],
    interaction: {},
    affected_systems: [],
    conflict_summary: {
      cyp_conflicts,
      target_overlaps: 0,
      shared_side_effects: 0,
      overall_risk: cyp_conflicts >= 2 ? 'high' : cyp_conflicts >= 1 ? 'moderate' : 'low',
    },
    source: 'offline',
  };
}
