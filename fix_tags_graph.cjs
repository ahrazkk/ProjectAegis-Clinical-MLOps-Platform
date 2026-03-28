const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

const targetStr = `                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/20" /> Benign / Unknown</div>
                </div>

            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ ARCHITECTURE SCROLL PARALLAX`;

const replacement = `                   <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-white/20" /> Benign / Unknown</div>
                </div>

              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════ ARCHITECTURE SCROLL PARALLAX`;

if (content.includes(targetStr)) {
  content = content.replace(targetStr, replacement);
  fs.writeFileSync(path, content, 'utf8');
  console.log("Added the missing </div> to Graph Stats block.");
} else {
  console.log("Could not find the target string to replace.");
}
