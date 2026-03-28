{/* ═══════════ GRAPH STATS, DISTRIBUTION & MOCK (FULL WIDTH) ═══════════ */}
      <section className="relative py-32 z-10">
        {/* Continuous Pipeline Connector Gradient Behind */}
        <div className="absolute top-[-20%] left-[10%] w-[1px] h-[140%] bg-gradient-to-b from-purple-500/0 via-pink-500/30 to-purple-500/0 pointer-events-none -z-10" />
        <div className="absolute top-[-20%] right-[10%] w-[1px] h-[140%] bg-gradient-to-b from-cyan-500/0 via-blue-500/30 to-cyan-500/0 pointer-events-none -z-10" />

        <div className="w-full px-4 sm:px-8 relative">
          
          <div className="grid grid-cols-1 xl:grid-cols-12 border border-white/5 bg-white/[0.01] backdrop-blur-2xl overflow-hidden rounded-2xl w-full">
            
            {/* Left Block: Data Metrics & Density Graph (Col span 5) */}
            <div className="col-span-1 xl:col-span-5 p-8 lg:p-14 flex flex-col relative bg-gradient-to-b from-black/80 to-black/40 z-10 border-r border-white/5">
               <Reveal>
                  <h3 className="text-xl lg:text-3xl font-heading font-semibold tracking-tight text-white mb-3">Model Accuracy & Scale</h3>
                  <p className="text-sm text-white/40 font-body mb-10 pb-8 border-b border-white/[0.03]">We process massive graph embeddings at 98.6% real-world inference accuracy. *Evaluated on validated DDI benchmarks.</p>
                  
                  <div className="grid grid-cols-2 gap-x-6 gap-y-10 mb-12">
                     {[
                       { label: 'Macro F1', value: '0.968', desc: 'Predictive precision across 86 classes' },
                       { label: 'ROC-AUC', value: '98.67%', desc: 'High true-positive rate bounding' },
                       { label: 'Nodes', value: '14,200+', desc: 'Isolated graph embeddings' },
                       { label: 'Edges', value: '2.4M+', desc: 'Metabolic & interaction pathways' }
                     ].map((stat, i) => (
                        <div key={i} className="group cursor-default">
                           <div className="text-[10px] font-mono uppercase tracking-widest text-[#ec4899] mb-1">{stat.label}</div>
                           <div className="text-3xl font-heading font-light text-white mb-1 group-hover:text-purple-400 transition-colors">{stat.value}</div>
                           <div className="text-[10px] text-white/30 font-body leading-relaxed max-w-[140px]">{stat.desc}</div>
                        </div>
                     ))}
                  </div>

                  {/* Probability Density Chart */}
                  <div className="mt-auto relative">
                     <div className="text-[10px] uppercase font-mono tracking-widest text-white/50 mb-4 flex items-center justify-between">
                       <span>Probability Density</span>
                       <span className="flex items-center gap-2"><div className="w-1.5 h-1.5 bg-[#facc15] rounded-full animate-pulse"/> Optimum: 0.56</span>
                     </div>
                     <div className="h-[200px] w-full relative">
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={gnnData?.probability || []} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                            <defs>
                              <linearGradient id="colorNegCard" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                              </linearGradient>
                              <linearGradient id="colorPosCard" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ec4899" stopOpacity={0.4}/>
                                <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                              </linearGradient>
                            </defs>
                            <XAxis dataKey="prob" type="number" domain={[0, 1]} stroke="#ffffff15" tick={{ fill: '#ffffff40', fontSize: 10 }} tickCount={6} />
                            <YAxis stroke="#ffffff15" tick={{ fill: '#ffffff40', fontSize: 10 }} opacity={0.5} />
                            <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', borderColor: '#333', fontSize: '11px', backdropFilter: 'blur(10px)' }} itemStyle={{ color: '#fff' }} />
                            <ReferenceLine x={0.56} stroke="#facc15" strokeDasharray="3 3" strokeWidth={1} opacity={0.6} />
                            <Area type="monotone" dataKey="neg" name="True N" stroke="#a855f7" strokeWidth={2} fill="url(#colorNegCard)" />
                            <Area type="monotone" dataKey="pos" name="True P" stroke="#ec4899" strokeWidth={2} fill="url(#colorPosCard)" />
                          </AreaChart>
                        </ResponsiveContainer>
                     </div>
                  </div>
               </Reveal>
            </div>

            {/* Right Block: Interactive Node Mock (Col span 7) */}
            <div className="col-span-1 xl:col-span-7 relative overflow-hidden bg-black/40 min-h-[500px] lg:min-h-[700px] flex items-center justify-center">
                {/* Background Grid & Blurs */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff05_1px,transparent_1px),linear-gradient(to_bottom,#ffffff05_1px,transparent_1px)] bg-[size:30px_30px] [mask-image:radial-gradient(ellipse_70%_70%_at_50%_50%,#000_100%,transparent_100%)] pointer-events-none" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40rem] h-[40rem] bg-purple-500/10 rounded-full blur-[120px] mix-blend-screen pointer-events-none" />
                
                {/* HUD Elements */}
                <div className="absolute top-8 left-8 z-30 pointer-events-none hidden sm:block">
                  <div className="flex items-center gap-2 mb-2">
                     <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-pink-500" />
                     </span>
                     <span className="text-[10px] font-mono text-white/50 tracking-widest uppercase">GNN Live State</span>
                  </div>
                  <div className="font-mono text-[9px] text-white/30 whitespace-pre">
                     false: 256\nfalse: 4\nfalse: 0.1
                  </div>
                </div>

                {/* Graph Visualization Container */}
                <div className="relative w-full h-full max-w-[700px] max-h-[700px]">
                    
                    {/* Interactive Edges */}
                    <GNNEdge x1="50%" y1="50%" x2="25%" y2="20%" interaction="CYP2C9 Inhibition (Severe Bleeding)" probability="0.984" 
                       severityColor={{text: "text-pink-400", bg: "bg-pink-500"}} 
                       lineProps={{stroke: "rgba(236,72,153,0.4)", dash: "4 4", width: 2}} delay={0.3} />
                       
                    <GNNEdge x1="50%" y1="50%" x2="75%" y2="25%" interaction="Unknown Topology Effect" probability="0.121" 
                       severityColor={{text: "text-white/80", bg: "bg-white/40"}} 
                       lineProps={{stroke: "rgba(255,255,255,0.1)", dash: "", width: 1.5}} delay={0.5} />
                       
                    <GNNEdge x1="50%" y1="50%" x2="80%" y2="70%" interaction="Decreased Renal Clearance" probability="0.835" 
                       severityColor={{text: "text-yellow-400", bg: "bg-yellow-400"}} 
                       lineProps={{stroke: "rgba(234,179,8,0.4)", dash: "", width: 2}} delay={0.7} />
                       
                    <GNNEdge x1="50%" y1="50%" x2="30%" y2="75%" interaction="Synergistic GI Hemorrhage" probability="0.923" 
                       severityColor={{text: "text-purple-400", bg: "bg-purple-500"}} 
                       lineProps={{stroke: "rgba(168,85,247,0.4)", dash: "2 6", width: 2}} delay={0.9} />

                    {/* Surrounding Nodes */}
                    <div className="absolute inset-0 pointer-events-none">
                       <div className="pointer-events-none w-full h-full relative">
                         <GNNNode x="25%" y="20%" label="Warfarin" color="border-pink-500/60" glow="bg-pink-500/20" delay={0.4} />
                         <GNNNode x="75%" y="25%" label="Clopidogrel" color="border-white/20" glow="bg-white/5" delay={0.6} />
                         <GNNNode x="80%" y="70%" label="Ibuprofen" color="border-yellow-500/50" glow="bg-yellow-500/10" delay={0.8} />
                         <GNNNode x="30%" y="75%" label="Apixaban" color="border-purple-500/60" glow="bg-purple-500/20" delay={1.0} />
                       </div>
                    </div>

                    {/* Center Node */}
                    <motion.div 
                      className="absolute z-30 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none flex flex-col items-center"
                      initial={{ scale: 0, opacity: 0 }}
                      whileInView={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", stiffness: 100, damping: 15 }}
                    >
                      <div className="text-[10px] font-mono text-[#a78bfa] mb-3 bg-black/60 px-2 py-0.5 rounded border border-purple-500/20">Target Input</div>
                      <div className="w-28 h-28 rounded-full bg-black/80 backdrop-blur-md border border-white flex flex-col items-center justify-center shadow-[0_0_40px_rgba(255,255,255,0.15)] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-white/10 to-transparent">
                        <span className="font-heading font-medium tracking-wide text-white">Aspirin</span>
                        <span className="text-[9px] text-white/40 font-mono mt-1">CHEMBL123</span>
                      </div>
                      <div className="absolute top-[32px] bottom-0 -inset-8 bg-white/5 rounded-full blur-[20px] -z-10 animate-pulse pointer-events-none" />
                    </motion.div>
                </div>

                {/* Legend */}
                <div className="absolute bottom-6 left-6 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30 border border-white/5 bg-black/40 px-4 py-2 rounded-full backdrop-blur-md">
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500/80 shadow-[0_0_10px_rgba(236,72,153,0.5)]" /> Severe</div>
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500/80 shadow-[0_0_10px_rgba(234,179,8,0.5)]" /> Moderate</div>
                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/20" /> Benign / Unknown</div>
                </div>

            </div>
          </div>
        </div>
      </section>

      