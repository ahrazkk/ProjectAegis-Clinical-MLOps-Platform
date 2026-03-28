const fs = require('fs');
const path = 'c:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/pages/LandingPageV2.jsx';
let content = fs.readFileSync(path, 'utf8');

const targetStr = `              </div>

            </div>
          </div>
        </section>

        {/* ═══════════ CTA SECTION`;

const newStr = `              </div>

            </div>
          </div>
          </div>
        </section>

        {/* ═══════════ CTA SECTION`;

content = content.replace(targetStr, newStr);

fs.writeFileSync(path, content, 'utf8');
console.log("Fixed closing tags");