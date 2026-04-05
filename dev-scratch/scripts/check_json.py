import os
import json

path = 'C:/Users/1kibr/Documents/WebDevelopment/DDI_PROJECTV2-FRONTEND/molecular-ai/src/assets/gnn_real_data.json'
try:
    s = os.path.getsize(path)
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"SIZE: {s}, NODES: {len(data.get('nodes', []))}")
except Exception as e:
    print(f"Error: {e}")
