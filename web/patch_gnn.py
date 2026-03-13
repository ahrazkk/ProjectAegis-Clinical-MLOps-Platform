
import re

with open("../web/ddi_api/services/gnn_predictor.py", "r", encoding="utf-8") as f:
    content = f.read()

polypharmacy_method = """    def predict_polypharmacy(self, drugs: List[Dict[str, str]]) -> Dict:
        \"\"\"
        Predict interactions for multiple drugs (N-way) natively on Macroscopic model.
        \"\"\"
        n = len(drugs)
        interactions = []
        max_risk = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                pred = self.predict(
                    drugs[i]["name"],
                    drugs[j]["name"],
                    drugs[i].get("smiles", ""),
                    drugs[j].get("smiles", "")
                )
                
                if pred.risk_score > 0.3:
                    interactions.append({
                        "drug_a": pred.drug_a,
                        "drug_b": pred.drug_b,
                        "risk_score": pred.risk_score,
                        "risk_level": pred.risk_level,
                        "severity": pred.severity,
                        "mechanism": pred.mechanism_hypothesis,
                        "affected_systems": list(pred.affected_systems)
                    })
                    max_risk = max(max_risk, pred.risk_score)
                    
        return {
            "max_risk_score": max_risk,
            "risk_level": "severe" if max_risk > 0.7 else "moderate" if max_risk > 0.4 else "minor" if max_risk > 0.2 else "none",
            "interactions": sorted(interactions, key=lambda x: x["risk_score"], reverse=True),
            "drugs_analyzed": n
        }

"""

if "def predict_polypharmacy" not in content:
    # Insert right before def batch_predict
    content = content.replace("    def batch_predict(", polypharmacy_method + "\n    def batch_predict(")
    with open("../web/ddi_api/services/gnn_predictor.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected predict_polypharmacy")
else:
    print("Already there")

