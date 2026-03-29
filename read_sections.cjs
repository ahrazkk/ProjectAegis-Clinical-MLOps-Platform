const fs = require('fs');
const content = fs.readFileSync('src/pages/ResearchPage.jsx', 'utf8');

const t1 = content.split("activeTab === 'overview'")[1].substring(0, 1500);
console.log("--------------- OVERVIEW ---------------");
console.log(t1);

const t2 = content.split("activeTab === 'evolution'")[1].substring(0, 2000);
console.log("--------------- EVOLUTION ---------------");
console.log(t2);
