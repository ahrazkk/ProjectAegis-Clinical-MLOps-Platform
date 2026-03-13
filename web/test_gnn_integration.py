"""Quick integration test for trained GNN in the ensemble predictor."""
import sys, os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
# Add model source code to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import django
django.setup()

print("=" * 60)
print("GNN PREDICTOR (STANDALONE)")
print("=" * 60)

from ddi_api.services.gnn_predictor import get_gnn_predictor

predictor = get_gnn_predictor()
print(f"Model type: {predictor.model_type.value}")
print(f"Loaded: {predictor.is_loaded}")
print()

tests = [
    ("Warfarin", "Aspirin",
     "CC(=O)CC(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
     "CC(=O)OC1=CC=CC=C1C(=O)O",
     "Known severe"),
    ("Simvastatin", "Amiodarone",
     "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
     "CCCCC1=C(C2=CC=C(OCCN(CC)CC)C=C2)C3=CC(I)=C(OCCC)C(I)=C3O1",
     "Known severe"),
    ("Metformin", "Omeprazole",
     "CN(C)C(=N)NC(=N)N",
     "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC3=NC=C(C)C(OC)=C3C",
     "Minimal interaction"),
]

for drug1, drug2, s1, s2, expected in tests:
    pred = predictor.predict(drug1, drug2, smiles1=s1, smiles2=s2)
    print(f"{drug1} + {drug2} ({expected}):")
    print(f"  prob={pred.interaction_probability:.3f}, severity={pred.severity}, "
          f"type={pred.interaction_type}, model={pred.model_used}")
    print()

print("=" * 60)
print("ENSEMBLE PREDICTOR")
print("=" * 60)

from ddi_api.services.ensemble_predictor import EnsembleDDIPredictor

ep = EnsembleDDIPredictor()
print(f"Sources: {list(ep.SOURCE_WEIGHTS.keys())}")
print()

result = ep.predict("Warfarin", "Aspirin")
print(f"Warfarin + Aspirin ensemble result:")
print(f"  Risk score: {result.final_risk_score}")
print(f"  Severity: {result.final_severity}")
print(f"  Interaction type: {result.final_interaction_type}")
print(f"  Consensus: {result.consensus_level}")
for sp in result.source_predictions:
    print(f"  [{sp.source.value}] severity={sp.severity}, conf={sp.confidence:.3f}, "
          f"risk={sp.risk_score:.3f}, available={sp.available}")

print("\nDone!")
