export const generateDetailedReport = (data) => {
  const {
    drugs = [],
    result = null,
    polypharmacyResult = null,
    digitalTwinResult = null,
    bodyMapData = null,
    evidence = null,
    timestamp = new Date().toISOString()
  } = data;

  let md = "# Project Aegis V2.0 - Comprehensive Drug Interaction Analysis Report\n\n";
  md += `**Date:** ${new Date().toLocaleString()}\n`;
  md += `**Total Drugs Analyzed:** ${drugs.length}\n\n`;

  // 1. DRUGS OVERVIEW
  md += "## 1. Analyzed Regimen (Input Drugs)\n";
  drugs.forEach((d, i) => {
    md += `### ${i + 1}. ${d.name}\n`;
    if (d.smiles) md += `- **SMILES:** \`${d.smiles}\`\n`;
    if (d.features) {
      if (d.features.weight) md += `- **Weight:** ${d.features.weight}\n`;
      if (d.features.logp) md += `- **LogP:** ${d.features.logp}\n`;
    }
  });
  md += "\n";

  // 2. POLYPHARMACY / GNN GALAXY INSIGHTS
  if (polypharmacyResult) {
    const riskLevel = typeof polypharmacyResult.overall_risk_level === 'string' 
      ? polypharmacyResult.overall_risk_level.toUpperCase() 
      : 'UNKNOWN';

    md += "## 2. Polypharmacy & AI Predicted Interactions (GNN Galaxy)\n";
    md += `- **Overall Risk Level:** ${riskLevel}\n`;
    const score = polypharmacyResult.regimen_risk_score || polypharmacyResult.max_risk_score;
    if (score) md += `- **Regimen Risk Score:** ${(score * 100).toFixed(2)}%\n`;
    if (polypharmacyResult.total_interactions) {
      md += `- **Total Interactions Detected:** ${polypharmacyResult.total_interactions}\n`;
    }
    if (polypharmacyResult.hub_drug) {
      md += `- **Primary Hub Drug:** ${polypharmacyResult.hub_drug} (Involved in ${polypharmacyResult.hub_interaction_count} pathways)\n`;
    }
    md += "\n";

    if (polypharmacyResult.interactions && polypharmacyResult.interactions.length > 0) {
      md += "### Key Identified Pairwise Interactions\n";
      polypharmacyResult.interactions.forEach((interaction, idx) => {
        md += `**Pair ${idx + 1}: ${interaction.source || 'Drug A'} + ${interaction.target || 'Drug B'}**\n`;
        md += `- **Risk Level:** ${interaction.severity || interaction.risk_level || 'Unknown'}\n`;
        if (interaction.risk_score) md += `- **Confidence/Risk Score:** ${(interaction.risk_score * 100).toFixed(2)}%\n`;
        if (interaction.mechanism) md += `- **Mechanism:** ${interaction.mechanism}\n`;
        md += "\n";
      });
    }

    if (polypharmacyResult.recommendations && polypharmacyResult.recommendations.length > 0) {
      md += "### Clinical Recommendations\n";
      polypharmacyResult.recommendations.forEach(rec => {
        md += `- ${rec}\n`;
      });
      md += "\n";
    }
  } else if (result) {
    // Basic pair fallback
    md += "## 2. Macroscopic AI Pair Prediction\n";
    md += `- **Interacting Pair:** ${result.drug_a || drugs[0]?.name} & ${result.drug_b || drugs[1]?.name}\n`;
    md += `- **Risk Level:** ${result.risk_level}\n`;
    if (result.risk_score) md += `- **Confidence Score:** ${(result.risk_score * 100).toFixed(2)}%\n`;
    md += "\n";
  }

  // 3. BODY MAP (SYSTEMIC IMPACT)
  if (bodyMapData && Object.keys(bodyMapData).length > 0) {
    md += "## 3. Systemic Impact Profile (Body Map)\n";
    Object.entries(bodyMapData).forEach(([system, data]) => {
      // data might contain score, desc, severity
      md += `### ${system.replace(/_/g, ' ').toUpperCase()}\n`;
      if (data.severity) md += `- **Severity Impact:** ${data.severity}\n`;
      if (data.score) md += `- **System Risk Score:** ${(data.score * 100).toFixed(1)}%\n`;
      if (data.description) md += `- **Observation:** ${data.description}\n`;
      if (data.drugs && data.drugs.length > 0) md += `- **Implicated Drugs:** ${data.drugs.join(', ')}\n`;
      md += "\n";
    });
  } else if (polypharmacyResult && polypharmacyResult.affected_systems) {
    md += "## 3. Systemic Impact Profile (Body Map)\n";
    const systems = polypharmacyResult.affected_systems;
    if (Array.isArray(systems)) {
      systems.forEach(sys => {
        md += `- **${sys.system || sys}**: Severity: ${sys.severity || 'N/A'}\n`;
        if (sys.description) md += `  - *Details:* ${sys.description}\n`;
      });
    } else if (typeof systems === 'object') {
      Object.keys(systems).forEach(k => {
        md += `- **${k}**: ${systems[k]}\n`;
      });
    }
    md += "\n";
  }

  // 4. DIGITAL TWIN (PATIENT-LEVEL TOXICITY/METABOLISM)
  if (digitalTwinResult) {
    md += "## 4. Patient Protocol Simulation (Poly Twin)\n";
    if (digitalTwinResult.summary) {
      const ts = digitalTwinResult.summary.toxicity_score || 0;
      const ms = digitalTwinResult.summary.metabolic_burden || 0;
      md += `- **Predicted Toxicity Score:** ${(ts * 100).toFixed(1)}%\n`;
      md += `- **Estimated Metabolic Burden:** ${(ms * 100).toFixed(1)}%\n`;
    }
    if (digitalTwinResult.alerts && digitalTwinResult.alerts.length > 0) {
      md += "### Simulation Alerts\n";
      digitalTwinResult.alerts.forEach(alert => {
        md += `- ${alert.message || alert}\n`;
      });
    }
    md += "\n";
  }

  // 5. EVIDENCE (REAL WORLD EVIDENCE / WHAT-IF KNOWLEDGE)
  if (evidence) {
    md += "## 5. Real-World Evidence & Database Findings\n";
    if (evidence.faers_data) {
      md += "### FDA FAERS Data\n";
      md += `- **Total Reports:** ${evidence.faers_data.total_reports || 0}\n`;
      if (evidence.faers_data.top_reactions && Array.isArray(evidence.faers_data.top_reactions)) {
        md += `- **Top Reactions:** ${evidence.faers_data.top_reactions.join(', ')}\n`;
      }
    }
    if (evidence.drugbank_data || evidence.interaction_desc) {
      md += "### DrugBank Knowledge\n";
      md += `${evidence.interaction_desc || "Database records found detailing historical interaction profiles."}\n`;
    }
    md += "\n";
  }

  // Inject Raw Data for AI Consumption
  md += "---\n## Appendix: Raw Structured Data for AI Analysis\n";
  md += "The following JSON block contains the complete unformatted data structure for this session to enable perfectly precise AI context mapping.\n\n";
  md += "```json\n";
  try {
    const rawData = {
      drugs,
      result,
      polypharmacyResult,
      digitalTwinResult,
      bodyMapData,
      evidence,
      exportTimestamp: timestamp
    };
    md += JSON.stringify(rawData, null, 2);
  } catch(e) {
    md += '{ "error": "Serialization failed." }';
  }
  md += "\n```\n\n";

  md += "---\n";
  md += "*Generated securely via Project Aegis AI Clinical Assistant*\n";

  // Create downloadable file
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = `Aegis_Research_Report_${Date.now()}.md`;
  document.body.appendChild(link);
  link.click();
  
  // Cleanup
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
