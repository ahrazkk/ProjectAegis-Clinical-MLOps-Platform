
import re

with open("../web/ddi_api/views.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the pubmedbert section in _internal_post_logic
old_section = r"""            # Use PubMedBERT model for text-based DDI prediction
            # This model was trained on ~19,000 DDI Corpus sentences
            pubmedbert = get_pubmedbert_predictor\(\)

            if pubmedbert\.is_loaded:
                # Use PubMedBERT for prediction \(primary method\)
                prediction = pubmedbert\.predict\(drug_a\[.name.\], drug_b\[.name.\]\)

                # Map interaction type to affected systems
                affected_systems_map = \{
                    .mechanism.: \[.liver., .metabolic.\],
                    .effect.: \[.cardiovascular., .hematologic.\],
                    .advise.: \[.general.\],
                    .int.: \[.general.\],
                    .no_interaction.: \[\]
                \}
                affected = affected_systems_map\.get\(prediction\.interaction_type, \[\]\)

                response_data = \{
                    .drug_a.: prediction\.drug_a,
                    .drug_b.: prediction\.drug_b,
                    .risk_score.: prediction\.risk_score,
                    .risk_level.: get_risk_level\(prediction\.risk_score\),
                    .severity.: prediction\.severity,
                    .confidence.: prediction\.confidence,
                    .mechanism_hypothesis.: pubmedbert\.get_mechanism_description\(
                        prediction\.interaction_type,
                        prediction\.drug_a,
                        prediction\.drug_b,
                        prediction\.confidence
                    \),
                    .affected_systems.: \[
                        \{.system.: sys, .severity.: prediction\.risk_score, .symptoms.: \[\]\}
                        for sys in affected
                    \],
                    .inference_time_ms.: \(time\.time\(\) - start_time\) \* 1000,
                    .source.: .pubmedbert.
                \}
            else:
                # Fallback to molecular structure-based model
                logger\.warning\(.PubMedBERT not loaded, falling back to GNN model.\)
                service = get_ddi_service\(\)
                prediction = service\.predict\(
                    drug_a\.get\(.smiles., ..\),
                    drug_b\.get\(.smiles., ..\),
                    drug_a\[.name.\],
                    drug_b\[.name.\]
                \)

                response_data = \{
                    .drug_a.: prediction\.drug_a,
                    .drug_b.: prediction\.drug_b,
                    .risk_score.: prediction\.risk_score,
                    .risk_level.: prediction\.risk_level,
                    .severity.: prediction\.severity,
                    .confidence.: prediction\.confidence,
                    .mechanism_hypothesis.: prediction\.mechanism_hypothesis,
                    .affected_systems.: \[
                        \{.system.: sys, .severity.: prediction\.risk_score, .symptoms.: \[\]\}
                        for sys in prediction\.affected_systems
                    \],
                    .inference_time_ms.: \(time\.time\(\) - start_time\) \* 1000,
                    .source.: .gnn.
                \}"""

new_section = r"""            # Use Macroscopic GraphSAGE model
            logger.info("Using Macroscopic GraphSAGE Model for DDI prediction")
            gnn_service = get_gnn_predictor()
            prediction = gnn_service.predict(
                drug_a["name"],
                drug_b["name"],
                drug_a.get("smiles", ""),
                drug_b.get("smiles", "")
            )
            
            response_data = {
                "drug_a": prediction.drug_a,
                "drug_b": prediction.drug_b,
                "risk_score": prediction.risk_score,
                "risk_level": prediction.risk_level,
                "severity": prediction.severity,
                "confidence": prediction.confidence,
                "mechanism_hypothesis": prediction.mechanism_hypothesis,
                "affected_systems": [
                    {"system": sys, "severity": prediction.risk_score, "symptoms": []}
                    for sys in prediction.affected_systems
                ],
                "inference_time_ms": (time.time() - start_time) * 1000,
                "source": "macroscopic_gnn"
            }"""

# using simple replace
content = re.sub(old_section, new_section, content)

with open("../web/ddi_api/views.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated views.py!")

