import os
import sys
import django
import json

# Setup Django environment
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
django.setup()

from django.test import RequestFactory
from ddi_api.views import DDIPredictionView

def test_predict():
    payload = {
        "drug_a": {
            "name": "Warfarin",
            "smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"
        },
        "drug_b": {
            "name": "Agenerase",
            "smiles": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)O[C@H]2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N"
        },
        "include_alternatives": False,
        "include_explanation": True
    }

    factory = RequestFactory()
    request = factory.post('/api/v1/predict/', data=json.dumps(payload), content_type='application/json')
    
    view = DDIPredictionView.as_view()
    response = view(request)
    
    print(f"Status Code: {response.status_code}")
    print("Response Data:")
    print(response.data)

if __name__ == '__main__':
    test_predict()
