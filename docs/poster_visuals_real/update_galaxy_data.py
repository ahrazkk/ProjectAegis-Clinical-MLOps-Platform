"""
Update gnn_real_data.json with embeddings from the NEW Enhanced GIN v2 model.
Produces 3D t-SNE positions from 512-dim model embeddings.
Rebuilds adjacency from neo4j_gnn_dataset.pt so edge counts stay current.
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
DATASET_PT = PROJECT_ROOT / 'web' / 'data' / 'neo4j_gnn_dataset.pt'
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


def load_dataset_edges(dataset_path: Path, pyg_id_by_index: list[str]) -> set[tuple[str, str]]:
    """Load undirected interaction edges from neo4j_gnn_dataset.pt."""
    if not dataset_path.exists():
        print(f"WARNING: Dataset not found at {dataset_path}. Falling back to legacy adjacency.")
        return set()

    print("Loading adjacency from neo4j_gnn_dataset.pt...")

    try:
        graph_data = torch.load(dataset_path, map_location='cpu', weights_only=False)
    except TypeError:
        graph_data = torch.load(dataset_path, map_location='cpu')

    edge_index = getattr(graph_data, 'edge_index', None)
    if edge_index is None and isinstance(graph_data, dict):
        edge_index = graph_data.get('edge_index')

    if edge_index is None:
        print("WARNING: edge_index missing in dataset. Falling back to legacy adjacency.")
        return set()

    if hasattr(edge_index, 'cpu'):
        edge_index = edge_index.cpu()

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        print(f"WARNING: Unexpected edge_index shape: {tuple(edge_index.shape)}. Falling back to legacy adjacency.")
        return set()

    max_valid_index = len(pyg_id_by_index) - 1
    unique_edges: set[tuple[str, str]] = set()

    for idx in range(edge_index.shape[1]):
        src_idx = int(edge_index[0, idx])
        dst_idx = int(edge_index[1, idx])

        if src_idx < 0 or dst_idx < 0 or src_idx > max_valid_index or dst_idx > max_valid_index:
            continue

        src = pyg_id_by_index[src_idx]
        dst = pyg_id_by_index[dst_idx]
        if src == dst:
            continue

        a, b = (src, dst) if src < dst else (dst, src)
        unique_edges.add((a, b))

    print(f"  Loaded {len(unique_edges)} unique undirected edges from dataset")
    return unique_edges

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
pyg_id_by_index: list[str] = []
with open(NODE_MAPPING, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pyg_id = str(row['pyg_id'])
        drugs.append({
            'pyg_id': pyg_id,
            'name': row['name'],
            'smiles': row['smiles'],
            'type': row.get('t_class', 'Unknown'),
        })
        pyg_id_by_index.append(pyg_id)
print(f"  {len(drugs)} drugs")

dataset_edges = load_dataset_edges(DATASET_PT, pyg_id_by_index)

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
new_adj: dict[str, list[str]] = {}
valid_ids: set[str] = set()

for i, drug in enumerate(valid_drugs):
    node_id = drug['pyg_id']
    pos = positions_3d[i].tolist()

    new_nodes.append({
        'id': node_id,
        'name': drug['name'],
        'type': drug['type'],
        'pos': pos,
    })
    valid_ids.add(node_id)
    new_adj[node_id] = []

if dataset_edges:
    kept = 0
    for u, v in dataset_edges:
        if u in valid_ids and v in valid_ids:
            new_adj[u].append(v)
            new_adj[v].append(u)
            kept += 1

    for node_id in new_adj:
        new_adj[node_id] = sorted(set(new_adj[node_id]))

    print(f"  Kept {kept} undirected edges after filtering to embedded drugs")
else:
    # Fallback if the tensor dataset is unavailable.
    for node in new_nodes:
        node_id = node['id']
        node_name = node['name']

        if node_id in old_adj:
            new_adj[node_id] = old_adj[node_id]
        elif node_name in old_nodes_by_name:
            old_id = old_nodes_by_name[node_name]['id']
            new_adj[node_id] = old_adj.get(old_id, [])

    print("  Using legacy adjacency fallback from previous gnn_real_data.json")

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
print(f"  Undirected edges: {sum(len(v) for v in new_adj.values()) // 2}")

print(f"\nGalaxy viewer data updated with Enhanced GIN v2 embeddings!")
