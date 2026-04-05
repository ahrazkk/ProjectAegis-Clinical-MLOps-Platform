const fs = require('fs');
const path = 'src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

const targetStr = `                  {/* Legend */}
                  <div className="absolute bottom-6 left-6 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30 border border-white/5 bg-black/40 px-4 py-2 rounded-full backdrop-blur-md">
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500" /> Severe</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500" /> Moderate</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/30" /> Benign</div>
                  </div>
              </div>
          </div>
        </div>
      </section>`;

const replacementStr = `                  {/* Legend */}
                  <div className="absolute bottom-6 left-6 flex flex-wrap items-center gap-6 text-[10px] uppercase font-mono tracking-widest text-white/40 z-30 border border-white/5 bg-black/40 px-4 py-2 rounded-full backdrop-blur-md">
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-pink-500" /> Severe</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-yellow-500" /> Moderate</div>
                     <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/30" /> Benign</div>
                  </div>
              </div>
            </div>
          </div>
        </div>
      </section>`;

content = content.replace(targetStr, replacementStr);
fs.writeFileSync(path, content, 'utf8');
console.log("Added 1 div");
