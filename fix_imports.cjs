const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let c = fs.readFileSync(path, 'utf8');

const missingImports = `import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { motion, AnimatePresence, useScroll, useTransform, useInView } from 'framer-motion';`;

c = c.replace("import React, { useState, useEffect, useRef, useMemo } from 'react';", "import React, { useState, useEffect, useRef, useMemo } from 'react';\n" + missingImports);

fs.writeFileSync(path, c);
console.log("Fixed!");
