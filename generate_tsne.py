import torch
import pandas as pd
import json
import numpy as np
import os
import sys

print('Starting script...')

from sklearn.manifold import TSNE

print('Imports successful.')

try:
    pt_path = 'molecular-ai/web/data/neo4j_gnn_dataset.pt'
    weights_path = 'molecular-ai/web/data/macroscopic_gnn_weights.pth'
    csv_path = 'molecular-ai/web/data/node_mapping.csv'
    out_path = 'molecular-ai/src/assets/gnn_real_data.json'

    # fallback paths if running in web/
    if not os.path.exists(pt_path):
        pt_path = 'web/data/neo4j_gnn_dataset.pt'
        csv_path = 'web/data/node_mapping.csv'
        out_path = 'src/assets/gnn_real_data.json'
        
    print("Loading PT Data...")
    data = torch.load(pt_path, weights_only=False)
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()

    print("Loading CSV Data...")
    df = pd.read_csv(csv_path)
    names = df['name'].tolist()
    types = df['t_class'].tolist()
    
    print(f'Starting TSNE reduction for {len(x)} nodes...')
    max_nodes = min(2000, len(x))
    indices = np.random.choice(len(x), max_nodes, replace=False)

    x_sub = x[indices]
    names_sub = [names[i] for i in indices]
    types_sub = [types[i] for i in indices]

    print("Running TSNE... this will take a while")
    tsne = TSNE(n_components=3, random_state=42)
    x_3d = tsne.fit_transform(x_sub)

    nodes = []
    print("Formatting nodes...")
    for i in range(max_nodes):
        nodes.append({
            'id': str(indices[i]),
            'name': str(names_sub[i]),
            'type': str(types_sub[i]),
            'pos': [float(x_3d[i, 0]), float(x_3d[i, 1]), float(x_3d[i, 2])]
        })

    adj = {}
    for i in range(max_nodes):
        adj[str(indices[i])] = []

    print("Formatting adj...")
    edges_added = 0
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u in indices and v in indices:
            if str(v) not in adj[str(u)]:
                adj[str(u)].append(str(v))
            if str(u) not in adj[str(v)]:
                adj[str(v)].append(str(u))
            edges_added += 1
            if edges_added > 500:
                break

    out_data = {'nodes': nodes, 'adj': adj}
    print(f"Writing to {out_path}...")
    
    with open(out_path, 'w') as f:
        json.dump(out_data, f)
        
    print(f'COMPLETED! Wrote {len(nodes)} nodes to {out_path}')
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
