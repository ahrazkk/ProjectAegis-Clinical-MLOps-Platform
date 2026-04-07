"""
Update gnn_real_data.json with embeddings from the NEW Enhanced GIN v2 model.
Produces 3D t-SNE positions from 512-dim model embeddings.
Preserves the existing JSON structure (nodes + adj) for the Galaxy viewer.
"""
import json
import sys
import csv
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / 'web' / 'models' / 'gnn' / 'gnn_best_model.pt'
NODE_MAPPING = PROJECT_ROOT / 'web' / 'data' / 'node_mapping.csv'
OLD_DATA = PROJECT_ROOT / 'src' / 'assets' / 'gnn_real_data.json'
OUTPUT = PROJECT_ROOT / 'src' / 'assets' / 'gnn_real_data.json'

sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'model'))

# ── Import model (handle relative imports) ───────────────────────────
import importlib, types

spec_feat = importlib.util.spec_from_file_location("gnn_featurizer",
    str(PROJECT_ROOT / 'src' / 'model' / 'gnn_featurizer.py'))
gnn_featurizer = importlib.util.module_from_spec(spec_feat)
sys.modules['gnn_featurizer'] = gnn_featurizer
spec_feat.loader.exec_module(gnn_featurizer)

model_pkg = types.ModuleType('model')
model_pkg.__path__ = [str(PROJECT_ROOT / 'src' / 'model')]
model_pkg.gnn_featurizer = gnn_featurizer
sys.modules['model'] = model_pkg
sys.modules['model.gnn_featurizer'] = gnn_featurizer

spec_model = importlib.util.spec_from_file_location("model.gnn_model",
    str(PROJECT_ROOT / 'src' / 'model' / 'gnn_model.py'),
    submodule_search_locations=[])
gnn_model_mod = importlib.util.module_from_spec(spec_model)
gnn_model_mod.__package__ = 'model'
sys.modules['model.gnn_model'] = gnn_model_mod
spec_model.loader.exec_module(gnn_model_mod)

DDIGraphModel = gnn_model_mod.DDIGraphModel
MolecularGraphFeaturizer = gnn_featurizer.MolecularGraphFeaturizer
ATOM_FEATURE_DIM = gnn_featurizer.ATOM_FEATURE_DIM
EDGE_FEATURE_DIM = gnn_featurizer.EDGE_FEATURE_DIM

# ── Load existing data (preserve adjacency) ──────────────────────────
print("Loading existing gnn_real_data.json...")
with open(OLD_DATA) as f:
    old_data = json.load(f)

old_adj = old_data.get('adj', {})
old_nodes_by_name = {n['name']: n for n in old_data['nodes']}
print(f"  {len(old_data['nodes'])} existing nodes, {len(old_adj)} adjacency entries")

# ── Load model ───────────────────────────────────────────────────────
print("Loading Enhanced GIN v2 model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
config = checkpoint.get('config', {})

model = DDIGraphModel(
    atom_feature_dim=ATOM_FEATURE_DIM,
    edge_feature_dim=EDGE_FEATURE_DIM,
    hidden_dim=config.get('hidden_dim', 256),
    num_gnn_layers=config.get('num_gnn_layers', 4),
    num_relation_classes=config.get('num_relation_classes', 1),
    dropout_rate=config.get('dropout_rate', 0.15),
    use_binary=config.get('use_binary', True),
    use_jumping_knowledge=config.get('use_jumping_knowledge', True),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded: {config.get('num_gnn_layers')} layers, {config.get('hidden_dim')} hidden")

featurizer = MolecularGraphFeaturizer(max_atoms=config.get('max_atoms', 128))

# ── Load drug data ───────────────────────────────────────────────────
print("Loading drugs from node_mapping.csv...")
drugs = []
with open(NODE_MAPPING, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        drugs.append({
            'pyg_id': row['pyg_id'],
            'name': row['name'],
            'smiles': row['smiles'],
            'type': row.get('t_class', 'Unknown'),
        })
print(f"  {len(drugs)} drugs")

# ── Extract embeddings ───────────────────────────────────────────────
print("Extracting 512-dim embeddings from new model...")
embeddings = []
valid_drugs = []
failed = 0

with torch.no_grad():
    for i, drug in enumerate(drugs):
        graph = featurizer.smiles_to_graph(drug['smiles'])
        if graph is None:
            failed += 1
            continue

        nf = graph['node_features'].unsqueeze(0)
        adj = graph['adjacency'].unsqueeze(0)
        ef = graph['edge_features'].unsqueeze(0)
        nm = graph['node_mask'].unsqueeze(0)

        emb = model.encoder(nf, adj, ef, nm)
        embeddings.append(emb.squeeze(0).numpy())
        valid_drugs.append(drug)

        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(drugs)}...")

embeddings = np.array(embeddings)
print(f"  {len(embeddings)} embeddings (dim={embeddings.shape[1]}), {failed} failed")

# ── 3D t-SNE ─────────────────────────────────────────────────────────
print("Computing 3D t-SNE from new embeddings...")
tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1500, learning_rate='auto')
positions_3d = tsne.fit_transform(embeddings)
print(f"  3D positions computed: shape {positions_3d.shape}")

# ── Build new gnn_real_data.json ─────────────────────────────────────
print("Building updated gnn_real_data.json...")
new_nodes = []
new_adj = {}

for i, drug in enumerate(valid_drugs):
    node_id = drug['pyg_id']
    pos = positions_3d[i].tolist()

    new_nodes.append({
        'id': node_id,
        'name': drug['name'],
        'type': drug['type'],
        'pos': pos,
    })

    # Preserve adjacency from old data
    if node_id in old_adj:
        new_adj[node_id] = old_adj[node_id]
    elif drug['name'] in old_nodes_by_name:
        old_id = old_nodes_by_name[drug['name']]['id']
        if old_id in old_adj:
            new_adj[node_id] = old_adj[old_id]
        else:
            new_adj[node_id] = []
    else:
        new_adj[node_id] = []

output_data = {
    'nodes': new_nodes,
    'adj': new_adj,
}

with open(OUTPUT, 'w') as f:
    json.dump(output_data, f)

file_size = OUTPUT.stat().st_size / (1024 * 1024)
print(f"  Saved: {OUTPUT}")
print(f"  Size: {file_size:.1f} MB")
print(f"  Nodes: {len(new_nodes)}")
print(f"  Adj entries: {len(new_adj)}")

print(f"\nGalaxy viewer data updated with Enhanced GIN v2 embeddings!")
