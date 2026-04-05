const { parse } = require('@babel/parser');
const fs = require('fs');

try {
  const code = fs.readFileSync('src/pages/LandingPageV2.jsx', 'utf8');
  parse(code, {
    sourceType: 'module',
    plugins: ['jsx']
  });
  console.log("No syntax errors! The JSX is completely valid.");
} catch(err) {
  console.log("Syntax error at Line " + err.loc.line + " Col " + err.loc.column);
  console.log(err.message);
}
