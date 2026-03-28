const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

const targetSectionOld = `      {/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}
      <section className="relative py-32 z-10">
        {/* Continuous Pipeline Connector Gradient Behind */}
        <div className="absolute top-[-20%] left-[10%] w-[1px] h-[140%] bg-gradient-to-b from-purple-500/0 via-pink-500/30 to-purple-500/0 pointer-events-none -z-10" />
        <div className="absolute top-[-20%] right-[10%] w-[1px] h-[140%] bg-gradient-to-b from-cyan-500/0 via-blue-500/30 to-cyan-500/0 pointer-events-none -z-10" />

        <div className="w-full px-4 sm:px-8 relative">

          <div className="grid grid-cols-1 xl:grid-cols-12 border border-white/5 bg-white/[0.01] backdrop-blur-2xl overflow-hidden rounded-2xl w-full">
            {/* Left Block: Data Metrics & Density Graph (Col span 5) */}
            <div className="col-span-1 xl:col-span-5 p-8 lg:p-14 flex flex-col relative bg-gradient-to-b from-black/80 to-black/40 z-10 border-r border-white/5">`;

const targetSectionNew = `      {/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}
      {/* Seamless cross-fade gradient pushing up from the dark block */}
      <div className="w-full h-40 bg-gradient-to-b from-transparent to-black pointer-events-none -mt-20 z-0 relative" />
      
      <section className="relative pb-32 pt-16 z-10 bg-black">
        {/* Horizontal seamless layout bleeding edge to edge visually */}
        <div className="w-full border-y border-white/5 bg-[#0a0a0a]/50 backdrop-blur-3xl shadow-[0_0_80px_rgba(0,0,0,0.8)] z-20">
          
          <div className="max-w-[1600px] mx-auto w-full">
            <div className="grid grid-cols-1 xl:grid-cols-12 border-x border-white/5 w-full">
              
              {/* Left Block: Data Metrics & Density Graph (Col span 5) */}
              <div className="col-span-1 xl:col-span-5 p-8 lg:p-14 flex flex-col relative bg-gradient-to-br from-black/80 to-black/40 z-10 border-r border-white/5">`;

if(content.includes(targetSectionOld)) {
  content = content.replace(targetSectionOld, targetSectionNew);
  // Remove "overflow-hidden" from the Right block so tooltip hovers can escape if needed
  content = content.replace('col-span-1 xl:col-span-7 relative overflow-hidden bg-black/40 min-h-[500px]', 'col-span-1 xl:col-span-7 relative bg-black/40 min-h-[500px]');
  fs.writeFileSync(path, content, 'utf8');
  console.log('Stretched container applied!');
} else {
  console.log('Could not find target block!');
}
