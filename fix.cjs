const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

const anchor = '{/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}';
const start = content.indexOf(anchor);

// Find the left block start
const leftBlock = '<div className="col-span-1 xl:col-span-5 p-8 lg:p-14';
const mid = content.indexOf(leftBlock, start);

if (start !== -1 && mid !== -1) {
  const prefix = content.slice(0, start);
  const postfix = content.slice(mid);
  
  const newMiddle = `      {/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}
      {/* Seamless cross-fade gradient pushing up from the dark block */}
      <div className="w-full h-40 bg-gradient-to-b from-transparent to-black pointer-events-none -mt-20 z-0 relative" />
      
      <section className="relative pb-32 pt-16 z-10 bg-black">
        {/* Horizontal seamless layout bleeding edge to edge visually */}
        <div className="w-full border-y border-white/5 bg-[#0a0a0a]/50 backdrop-blur-3xl shadow-[0_0_80px_rgba(0,0,0,0.8)] z-20">
          
          <div className="max-w-[1600px] mx-auto w-full">
            <div className="grid grid-cols-1 xl:grid-cols-12 w-full">
              
              `;

  content = prefix + newMiddle + postfix;
  
  // Also remove "overflow-hidden" from the right block so our tooltips don't clip at the edges.
  // Wait, if we use substring we can just replace.
  const rightBlockOld = 'col-span-1 xl:col-span-7 relative overflow-hidden bg-black/40 min-h-[500px]';
  const rightBlockNew = 'col-span-1 xl:col-span-7 relative bg-black/40 min-h-[500px] overflow-visible';
  content = content.replace(rightBlockOld, rightBlockNew);

  // One more thing: We want the main Bento section right above to slightly shrink or parallax if possible, 
  // but a CSS gradient blend is usually enough.

  fs.writeFileSync(path, content, 'utf8');
  console.log("Success slicing");
} else {
  console.log("Could not find anchors", { start, mid });
}
