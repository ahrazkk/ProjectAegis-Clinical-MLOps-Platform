export const DEMO_MODE_SETTING_KEY = 'aegis:demo-mode:v1';
export const USER_PREFERENCES_STORAGE_KEY = 'aegis:user-preferences:v1';

export const DEMO_DRUG_GROUPS = [
  {
    id: 'cardio-bleed-risk',
    title: 'Cardio + Bleed Risk Demo',
    reason: 'Demonstrates antiplatelet/anticoagulant risk stacking and cardiovascular overlap.',
    drugs: ['Aspirin', 'Clopidogrel', 'Atorvastatin', 'Metoprolol'],
  },
  {
    id: 'hypertension-stack',
    title: 'Hypertension Multi-Drug Stack',
    reason: 'Good for showing polypharmacy interactions in blood-pressure management combinations.',
    drugs: ['Lisinopril', 'Amlodipine', 'Losartan', 'Metoprolol'],
  },
  {
    id: 'metabolic-syndrome',
    title: 'Metabolic Syndrome Candidate Set',
    reason: 'Useful for demoing diabetic/cardiometabolic regimen balancing and risk tradeoffs.',
    drugs: ['Metformin', 'Atorvastatin', 'Losartan', 'Aspirin'],
  },
  {
    id: 'pain-otc-load',
    title: 'Pain + OTC Interaction Load',
    reason: 'Highlights common over-the-counter NSAID combinations and additive adverse risk patterns.',
    drugs: ['Ibuprofen', 'Naproxen', 'Acetaminophen', 'Aspirin'],
  },
  {
    id: 'infection-support',
    title: 'Infection + Symptom Support',
    reason: 'Represents a realistic short-term treatment bundle for demoing supportive co-medications.',
    drugs: ['Amoxicillin', 'Acetaminophen', 'Ibuprofen', 'Omeprazole'],
  },
];

export function readDemoModeSetting() {
  if (typeof window === 'undefined') return false;

  const raw = window.localStorage.getItem(DEMO_MODE_SETTING_KEY);
  if (raw === null || raw === undefined) return false;

  if (raw === '1' || raw === 'true') return true;
  if (raw === '0' || raw === 'false') return false;

  try {
    const parsed = JSON.parse(raw);
    return Boolean(parsed);
  } catch {
    return false;
  }
}

export function writeDemoModeSetting(enabled) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(DEMO_MODE_SETTING_KEY, JSON.stringify(Boolean(enabled)));
}

export function readUserPreferences() {
  if (typeof window === 'undefined') return null;

  try {
    const raw = window.localStorage.getItem(USER_PREFERENCES_STORAGE_KEY);
    if (!raw) return null;

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return null;
    return parsed;
  } catch {
    return null;
  }
}

export function writeUserPreferences(preferences) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(USER_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences));
}
