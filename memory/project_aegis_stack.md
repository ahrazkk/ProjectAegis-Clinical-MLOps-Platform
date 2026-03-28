---
name: Project Aegis Stack
description: Tech stack and key files for the DDI AI frontend project
type: project
---

Project Aegis is an AI-powered Drug-Drug Interaction (DDI) clinical decision support system.

- Stack: React 19, Vite, Tailwind CSS, Framer Motion 12, Three.js/R3F, Recharts, Lucide React
- Pages: LandingPageV2 (/), Dashboard (/dashboard), ResearchPage (/research)
- Key styling files: src/index.css (950+ lines, CSS variables + component classes), tailwind.config.js
- Theme: Dark (GitHub dark #0D1117) + Light toggle, CSS custom properties pattern
- Existing animation libs: Framer Motion already used extensively throughout
- Backend: Django REST + SQLite + Neo4j + Redis on port 8000

**Why:** Need to know stack before suggesting changes.
**How to apply:** Always use Framer Motion for animations (not CSS transitions), Tailwind for layout, CSS variables for colors.
