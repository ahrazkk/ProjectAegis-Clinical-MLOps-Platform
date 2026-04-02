// organRegistry.js — Anatomical data: SVG paths, organ centers, circulatory routes
// All coordinates in a 400x900 viewBox. NO symptom/side-effect data here (comes from API).

// ─── Organ system definitions ───────────────────────────────────────────────
export const ORGAN_SYSTEMS = {
  brain: {
    id: 'brain',
    name: 'Central Nervous System',
    shortName: 'Brain',
    center: { x: 200, y: 72 },
    paths: [
      // Cerebrum
      'M 200,30 C 165,30 140,48 138,72 C 136,96 155,115 175,118 C 185,120 195,120 200,118 C 205,120 215,120 225,118 C 245,115 264,96 262,72 C 260,48 235,30 200,30 Z',
      // Brain stem
      'M 192,118 L 192,135 Q 196,142 200,142 Q 204,142 208,135 L 208,118',
    ],
  },
  heart: {
    id: 'heart',
    name: 'Cardiovascular System',
    shortName: 'Heart',
    center: { x: 215, y: 260 },
    paths: [
      // Anatomical heart shape (not valentine)
      'M 205,240 C 200,235 190,232 185,238 C 178,246 180,258 188,268 C 195,276 205,285 210,290 L 215,295 L 220,290 C 225,285 235,276 242,268 C 250,258 252,246 245,238 C 240,232 230,235 225,240 L 215,252 Z',
      // Aortic arch suggestion
      'M 215,240 C 215,230 222,222 232,222 C 240,222 245,228 245,235',
    ],
  },
  lungs: {
    id: 'lungs',
    name: 'Respiratory System',
    shortName: 'Lungs',
    center: { x: 200, y: 255 },
    paths: [
      // Left lung
      'M 165,210 C 148,218 135,240 135,270 C 135,300 145,318 158,322 C 172,326 182,318 185,305 C 188,290 188,260 186,240 C 184,220 175,208 165,210 Z',
      // Right lung
      'M 235,210 C 252,218 265,240 265,270 C 265,300 255,318 242,322 C 228,326 218,318 215,305 C 212,290 212,260 214,240 C 216,220 225,208 235,210 Z',
      // Trachea
      'M 196,148 L 196,215 M 204,148 L 204,215',
      // Bronchi
      'M 196,215 C 190,220 180,218 175,215 M 204,215 C 210,220 220,218 225,215',
    ],
  },
  liver: {
    id: 'liver',
    name: 'Hepatic System',
    shortName: 'Liver',
    center: { x: 175, y: 330 },
    paths: [
      // Main liver body (right side of body, left in anterior view)
      'M 145,310 C 135,315 128,330 132,348 C 136,365 155,375 180,372 C 205,369 225,360 235,348 C 242,340 240,325 232,318 C 222,310 200,305 175,305 C 158,305 148,307 145,310 Z',
      // Gallbladder
      'M 225,340 C 228,345 232,355 228,360 C 224,365 220,358 222,350 Z',
    ],
  },
  kidney: {
    id: 'kidney',
    name: 'Renal System',
    shortName: 'Kidneys',
    center: { x: 200, y: 380 },
    paths: [
      // Left kidney
      'M 150,365 C 142,370 138,385 142,400 C 146,412 158,415 165,408 C 172,400 170,378 162,368 C 158,364 154,363 150,365 Z',
      // Right kidney
      'M 250,365 C 258,370 262,385 258,400 C 254,412 242,415 235,408 C 228,400 230,378 238,368 C 242,364 246,363 250,365 Z',
      // Ureters
      'M 156,412 C 158,430 170,445 185,455 M 244,412 C 242,430 230,445 215,455',
      // Bladder
      'M 185,455 C 185,465 192,472 200,472 C 208,472 215,465 215,455',
    ],
  },
  gi: {
    id: 'gi',
    name: 'Gastrointestinal System',
    shortName: 'GI Tract',
    center: { x: 200, y: 430 },
    paths: [
      // Stomach
      'M 185,305 C 175,310 168,325 172,340 C 176,352 190,355 200,348 C 208,342 210,330 205,318 C 202,310 192,302 185,305 Z',
      // Small intestine (coiled)
      'M 180,380 C 175,390 178,400 185,405 C 192,410 200,408 205,400 C 210,392 208,385 200,382 C 192,380 185,385 183,392 C 181,398 186,405 195,408 C 204,410 210,405 212,396',
      // Large intestine
      'M 155,378 L 155,440 C 155,455 168,465 185,465 L 215,465 C 232,465 245,455 245,440 L 245,378 C 245,368 238,362 228,362 L 228,430 C 228,438 222,445 215,445 L 185,445 C 178,445 172,438 172,430 L 172,362 C 162,362 155,368 155,378 Z',
    ],
  },
  blood: {
    id: 'blood',
    name: 'Hematological System',
    shortName: 'Blood',
    center: { x: 200, y: 350 },
    isOverlay: true, // Rendered as a body-wide overlay
  },
  skin: {
    id: 'skin',
    name: 'Dermatological',
    shortName: 'Skin',
    center: { x: 200, y: 400 },
    isOverlay: true,
  },
  musculoskeletal: {
    id: 'musculoskeletal',
    name: 'Musculoskeletal',
    shortName: 'MSK',
    center: { x: 200, y: 500 },
    paths: [
      // Simplified spine
      'M 200,148 L 200,490',
      // Pelvis outline
      'M 160,470 C 165,485 180,495 200,498 C 220,495 235,485 240,470',
    ],
    isSecondary: true, // Rendered lighter, as structural reference
  },
  endocrine: {
    id: 'endocrine',
    name: 'Endocrine System',
    shortName: 'Endocrine',
    center: { x: 200, y: 175 },
    paths: [
      // Thyroid gland
      'M 190,155 C 185,160 185,170 192,172 C 196,173 200,168 200,168 C 200,168 204,173 208,172 C 215,170 215,160 210,155',
      // Adrenal glands (atop kidneys)
      'M 148,358 C 150,354 156,354 158,358 M 242,358 C 244,354 250,354 252,358',
    ],
    isSecondary: true,
  },
};

// ─── Body outline paths ─────────────────────────────────────────────────────
export const BODY_OUTLINE = {
  // Head and neck
  head: 'M 200,28 C 182,28 170,42 170,62 C 170,80 178,94 188,100 C 188,110 188,120 192,126 L 208,126 C 212,120 212,110 212,100 C 222,94 230,80 230,62 C 230,42 218,28 200,28 Z',
  // Torso
  torso: 'M 186,126 C 170,132 156,142 146,156 C 135,172 130,192 131,214 C 132,238 139,260 144,282 C 148,300 147,322 141,341 C 136,357 132,374 133,392 C 134,420 146,445 162,466 C 170,477 174,492 174,508 L 174,608 C 174,641 165,681 158,721 C 154,742 157,759 171,760 L 183,760 C 192,759 196,745 195,728 L 194,640 C 194,618 196,595 200,573 C 204,595 206,618 206,640 L 205,728 C 204,745 208,759 217,760 L 229,760 C 243,759 246,742 242,721 C 235,681 226,641 226,608 L 226,508 C 226,492 230,477 238,466 C 254,445 266,420 267,392 C 268,374 264,357 259,341 C 253,322 252,300 256,282 C 261,260 268,238 269,214 C 270,192 265,172 254,156 C 244,142 230,132 214,126 Z',
  // Left arm
  leftArm: 'M 146,158 C 132,168 121,183 114,201 C 105,223 100,247 96,273 C 92,300 88,327 85,354 C 83,372 83,390 90,404 C 94,412 102,416 109,414 C 114,412 116,406 116,401 C 116,394 115,386 115,379 C 116,386 117,393 118,400 C 119,407 122,412 127,412 C 132,411 134,406 134,400 C 133,392 133,384 134,376 C 136,385 138,394 140,402 C 142,409 145,413 150,412 C 155,410 156,404 154,397 C 152,385 152,373 154,361 C 156,344 160,327 162,310 C 165,287 166,264 164,241 C 162,212 156,184 148,164',
  // Right arm
  rightArm: 'M 254,158 C 268,168 279,183 286,201 C 295,223 300,247 304,273 C 308,300 312,327 315,354 C 317,372 317,390 310,404 C 306,412 298,416 291,414 C 286,412 284,406 284,401 C 284,394 285,386 285,379 C 284,386 283,393 282,400 C 281,407 278,412 273,412 C 268,411 266,406 266,400 C 267,392 267,384 266,376 C 264,385 262,394 260,402 C 258,409 255,413 250,412 C 245,410 244,404 246,397 C 248,385 248,373 246,361 C 244,344 240,327 238,310 C 235,287 234,264 236,241 C 238,212 244,184 252,164',
};

// ─── Skeletal hints (dashed overlay) ────────────────────────────────────────
export const SKELETAL_HINTS = {
  // Ribcage
  ribs: [
    'M 155,200 C 170,195 200,190 230,195 C 245,198 250,200 245,205',
    'M 150,220 C 165,215 200,210 235,215 C 248,218 252,222 248,226',
    'M 148,240 C 163,235 200,230 237,235 C 250,238 254,242 250,246',
    'M 147,260 C 162,255 200,250 238,255 C 252,258 255,262 252,266',
    'M 148,280 C 162,275 200,270 238,275 C 252,278 254,282 250,286',
  ],
  // Spine
  spine: 'M 200,125 L 200,480',
  // Pelvis
  pelvis: 'M 155,465 C 155,480 172,498 200,502 C 228,498 245,480 245,465 C 245,455 235,448 220,445 L 200,445 L 180,445 C 165,448 155,455 155,465 Z',
  // Skull outline
  skull: 'M 200,22 C 172,22 155,42 155,65 C 155,88 170,105 190,112 L 210,112 C 230,105 245,88 245,65 C 245,42 228,22 200,22 Z',
};

// ─── Soft tissue landmarks for visual realism ──────────────────────────────
export const ANATOMY_LANDMARKS = {
  clavicle: 'M 162,162 C 178,156 192,154 200,156 C 208,154 222,156 238,162',
  sternum: 'M 200,170 L 200,252',
  abdominalMidline: 'M 200,285 L 200,455',
  iliacCrest: 'M 160,468 C 174,462 190,460 200,462 C 210,460 226,462 240,468',
  scapulaLeft: 'M 164,212 C 154,224 154,246 164,258 C 174,270 188,266 192,252 C 196,238 190,220 178,212 Z',
  scapulaRight: 'M 236,212 C 246,224 246,246 236,258 C 226,270 212,266 208,252 C 204,238 210,220 222,212 Z',
};

// ─── Circulatory system paths ───────────────────────────────────────────────
export const CIRCULATORY = {
  // Aorta (heart → body)
  aorta: 'M 215,245 C 215,235 225,225 235,225 C 245,225 248,230 248,235 L 248,350 C 248,370 240,400 230,430 L 220,470 L 210,520 L 205,580',
  // Vena cava (body → heart)
  venaCava: 'M 210,260 L 210,320 C 210,350 200,380 195,410 L 190,470 L 185,520 L 180,580',
  // Hepatic portal vein (GI → liver)
  hepaticPortal: 'M 200,370 C 195,360 185,355 178,345',
  // Renal arteries
  renalLeft: 'M 200,385 C 190,382 170,378 158,380',
  renalRight: 'M 200,385 C 210,382 230,378 242,380',
  // Carotid arteries (to brain)
  carotidLeft: 'M 195,160 L 190,140 L 188,120 L 190,90 L 195,72',
  carotidRight: 'M 205,160 L 210,140 L 212,120 L 210,90 L 205,72',
  // Pulmonary
  pulmonaryLeft: 'M 210,248 C 200,240 185,235 170,250',
  pulmonaryRight: 'M 220,248 C 230,240 245,235 255,250',
};

// ─── Drug pathway route points ──────────────────────────────────────────────
// For particles: oral → GI → liver (first-pass) → heart → aorta → target organs
export const DRUG_PATHWAY = {
  oral: [
    { x: 200, y: 100 },  // Mouth
    { x: 200, y: 148 },  // Esophagus
    { x: 195, y: 200 },  // Esophagus lower
    { x: 192, y: 310 },  // Stomach
    { x: 195, y: 350 },  // GI absorption
    { x: 178, y: 340 },  // Hepatic portal → liver
    { x: 175, y: 330 },  // Liver (first-pass metabolism)
    { x: 200, y: 310 },  // Hepatic vein → heart
    { x: 215, y: 260 },  // Heart
    { x: 235, y: 225 },  // Aortic arch
    { x: 248, y: 235 },  // Descending aorta
  ],
  // Branch points from aorta to each organ
  branches: {
    brain: [
      { x: 235, y: 225 },
      { x: 215, y: 180 },
      { x: 205, y: 140 },
      { x: 200, y: 100 },
      { x: 200, y: 72 },
    ],
    lungs: [
      { x: 215, y: 260 },
      { x: 200, y: 250 },
      { x: 175, y: 255 },
    ],
    kidney: [
      { x: 248, y: 380 },
      { x: 230, y: 378 },
      { x: 250, y: 385 },
    ],
    gi: [
      { x: 248, y: 350 },
      { x: 230, y: 380 },
      { x: 200, y: 430 },
    ],
  },
};

// ─── Severity color scale ───────────────────────────────────────────────────
export function getSeverityColor(severity) {
  if (severity > 0.7) return { fill: '#ef4444', glow: 'rgba(239, 68, 68, 0.6)', label: 'Severe', class: 'text-red-400' };
  if (severity > 0.4) return { fill: '#f97316', glow: 'rgba(249, 115, 22, 0.5)', label: 'Moderate', class: 'text-orange-400' };
  if (severity > 0)   return { fill: '#eab308', glow: 'rgba(234, 179, 8, 0.4)', label: 'Mild', class: 'text-yellow-400' };
  return { fill: '#1e293b', glow: 'none', label: 'Normal', class: 'text-slate-500' };
}

// ─── Map backend organ_system values to our organ IDs ───────────────────────
export const ORGAN_SYSTEM_MAP = {
  'brain': 'brain',
  'central nervous system': 'brain',
  'cns': 'brain',
  'neurological': 'brain',
  'heart': 'heart',
  'cardiovascular': 'heart',
  'cardiac': 'heart',
  'lungs': 'lungs',
  'respiratory': 'lungs',
  'pulmonary': 'lungs',
  'liver': 'liver',
  'hepatic': 'liver',
  'hepatobiliary': 'liver',
  'kidney': 'kidney',
  'renal': 'kidney',
  'gi': 'gi',
  'gastrointestinal': 'gi',
  'digestive': 'gi',
  'blood': 'blood',
  'hematologic': 'blood',
  'hematological': 'blood',
  'hematopoietic': 'blood',
  'skin': 'skin',
  'dermatologic': 'skin',
  'dermatological': 'skin',
  'cutaneous': 'skin',
  'musculoskeletal': 'musculoskeletal',
  'muscular': 'musculoskeletal',
  'skeletal': 'musculoskeletal',
  'endocrine': 'endocrine',
  'hormonal': 'endocrine',
  'metabolic': 'endocrine',
  'other': null,
};

// Normalize any organ system string to our ID
export function normalizeOrganSystem(raw) {
  if (!raw) return null;
  return ORGAN_SYSTEM_MAP[raw.toLowerCase().trim()] || null;
}
