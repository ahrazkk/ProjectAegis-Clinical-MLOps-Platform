"""
Extract drug embeddings from the NEW Enhanced GIN v2 model (trained on Colab).
Then generate UMAP + t-SNE visualizations with coarse pharmacological categories.

All embeddings come from the actual production model (gnn_best_model.pt).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import csv
import sys
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from collections import Counter

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, skipping UMAP charts")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT = str(SCRIPT_DIR)

# Add src/model to path for imports
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'model'))

MODEL_PATH = PROJECT_ROOT / 'web' / 'models' / 'gnn' / 'gnn_best_model.pt'
NODE_MAPPING = PROJECT_ROOT / 'web' / 'data' / 'node_mapping.csv'

# ── Coarse category mapping ──────────────────────────────────────────
COARSE_MAP = {
    'Benzodiazepine': 'CNS / Neuro', 'Atypical Antipsychotic': 'CNS / Neuro',
    'Antidepressant': 'CNS / Neuro', 'Opioid': 'CNS / Neuro',
    'Decreased Central Nervous System Organized Electrical Activity': 'CNS / Neuro',
    'Decreased Central Nervous System Disorganized Electrical Activity': 'CNS / Neuro',
    'Decreased Organized Electrical Activity': 'CNS / Neuro',
    'Anesthetic': 'CNS / Neuro', 'Barbiturate': 'CNS / Neuro',
    'Anticonvulsant': 'CNS / Neuro', 'SSRI': 'CNS / Neuro', 'SNRI': 'CNS / Neuro',
    'Anxiolytic': 'CNS / Neuro', 'Sedative': 'CNS / Neuro', 'Hypnotic': 'CNS / Neuro',
    'Antipsychotic': 'CNS / Neuro', 'Typical Antipsychotic': 'CNS / Neuro',
    'Mood Stabilizer': 'CNS / Neuro', 'Dopamine Agonist': 'CNS / Neuro',
    'Dopamine Antagonist': 'CNS / Neuro', 'Cholinesterase Inhibitor': 'CNS / Neuro',
    'Muscle Relaxant': 'CNS / Neuro', 'Neuromuscular Blocker': 'CNS / Neuro',

    'Antibiotic': 'Anti-infectives', 'Antiviral': 'Anti-infectives',
    'Antifungal': 'Anti-infectives', 'Penicillin Antibiotic': 'Anti-infectives',
    'Cephalosporin': 'Anti-infectives', 'Fluoroquinolone': 'Anti-infectives',
    'Macrolide': 'Anti-infectives', 'Aminoglycoside': 'Anti-infectives',
    'Tetracycline': 'Anti-infectives', 'Sulfonamide': 'Anti-infectives',
    'Antiparasitic': 'Anti-infectives', 'Antimalarial': 'Anti-infectives',
    'Antitubercular': 'Anti-infectives', 'Antiretroviral': 'Anti-infectives',
    'Protease Inhibitor': 'Anti-infectives',

    'Alpha Blocker': 'Cardiovascular', 'Beta Blocker': 'Cardiovascular',
    'ACE Inhibitor': 'Cardiovascular', 'ARB': 'Cardiovascular',
    'Calcium Channel Blocker': 'Cardiovascular', 'Statin': 'Cardiovascular',
    'Antiarrhythmic': 'Cardiovascular', 'Vasodilator': 'Cardiovascular',
    'Anticoagulant': 'Cardiovascular', 'Antiplatelet': 'Cardiovascular',
    'Diuretic': 'Cardiovascular', 'Loop Diuretic': 'Cardiovascular',
    'Thiazide': 'Cardiovascular', 'Cardiac Glycoside': 'Cardiovascular',
    'Nitrate': 'Cardiovascular', 'Fibrate': 'Cardiovascular',

    'Decreased DNA Integrity': 'Oncology', 'Cellular Activity Alteration': 'Oncology',
    'Antineoplastic': 'Oncology', 'Alkylating Agent': 'Oncology',
    'Antimetabolite': 'Oncology', 'Topoisomerase Inhibitor': 'Oncology',
    'Kinase Inhibitor': 'Oncology', 'Monoclonal Antibody': 'Oncology',

    'Decreased Prostaglandin Production': 'Anti-inflammatory',
    'NSAID': 'Anti-inflammatory', 'COX-2 Inhibitor': 'Anti-inflammatory',
    'Analgesic': 'Anti-inflammatory',

    'Corticosteroid': 'Immunology', 'Glucocorticoid': 'Immunology',
    'Immunosuppressant': 'Immunology', 'DMARD': 'Immunology',

    'Bronchodilation': 'Respiratory', 'Decreased Histamine Activity': 'Respiratory',
    'Antihistamine': 'Respiratory', 'Leukotriene Antagonist': 'Respiratory',
    'Decongestant': 'Respiratory',

    'Decreased Cell Wall Synthesis & Repair': 'Metabolic',
    'Proton Pump Inhibitor': 'GI / Metabolic', 'H2 Blocker': 'GI / Metabolic',
    'Laxative': 'GI / Metabolic', 'Antiemetic': 'GI / Metabolic',

    'Insulin': 'Endocrine', 'Sulfonylurea': 'Endocrine', 'Thyroid': 'Endocrine',
    'Estrogen': 'Endocrine', 'Androgen': 'Endocrine', 'Bisphosphonate': 'Endocrine',
}

CATEGORY_COLORS = {
    'CNS / Neuro':       '#6366f1',
    'Anti-infectives':   '#06b6d4',
    'Cardiovascular':    '#ef4444',
    'Oncology':          '#f97316',
    'Anti-inflammatory': '#eab308',
    'Respiratory':       '#22c55e',
    'Immunology':        '#a855f7',
    'Endocrine':         '#ec4899',
    'GI / Metabolic':    '#14b8a6',
    'Metabolic':         '#f59e0b',
    'Other':             '#94a3b8',
    'Unknown':           '#cbd5e1',
}


def map_to_coarse(fine_type):
    if fine_type == 'Unknown':
        return 'Unknown'
    if fine_type in COARSE_MAP:
        return COARSE_MAP[fine_type]
    fl = fine_type.lower()
    if any(k in fl for k in ['neuro', 'cns', 'brain', 'seizure', 'psycho', 'anxiety', 'depress']):
        return 'CNS / Neuro'
    if any(k in fl for k in ['antibiotic', 'antiviral', 'antifungal', 'anti-infect', 'bacterio']):
        return 'Anti-infectives'
    if any(k in fl for k in ['cardio', 'heart', 'vascular', 'blood pressure', 'coagul', 'platelet']):
        return 'Cardiovascular'
    if any(k in fl for k in ['cancer', 'tumor', 'dna', 'cellular', 'neoplas', 'cytotox']):
        return 'Oncology'
    if any(k in fl for k in ['inflam', 'pain', 'prostaglandin', 'cox']):
        return 'Anti-inflammatory'
    if any(k in fl for k in ['lung', 'bronch', 'histamine', 'respiratory', 'asthma']):
        return 'Respiratory'
    if any(k in fl for k in ['immune', 'steroid', 'cortiso']):
        return 'Immunology'
    if any(k in fl for k in ['hormone', 'insulin', 'thyroid', 'estrogen', 'endocrin']):
        return 'Endocrine'
    if any(k in fl for k in ['gastro', 'stomach', 'acid', 'digest', 'metabol']):
        return 'GI / Metabolic'
    return 'Other'


def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#fafbfc')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#0f172a', pad=12)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color='#334155')
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color='#334155')
    ax.tick_params(colors='#64748b', labelsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#cbd5e1')


def plot_embedding(coords_2d, drug_types, title, filename, method_label):
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor('white')

    cat_counts = Counter(drug_types)
    show_cats = [c for c, _ in cat_counts.most_common() if c not in ('Unknown', 'Other')]

    mask_unk = np.array([t == 'Unknown' for t in drug_types])
    if mask_unk.any():
        ax.scatter(coords_2d[mask_unk, 0], coords_2d[mask_unk, 1],
                   c='#e2e8f0', s=8, alpha=0.25, edgecolors='none', zorder=1)

    mask_other = np.array([t == 'Other' for t in drug_types])
    if mask_other.any():
        ax.scatter(coords_2d[mask_other, 0], coords_2d[mask_other, 1],
                   c='#94a3b8', s=12, alpha=0.35, edgecolors='none', zorder=2)

    legend_handles = []
    for i, cat in enumerate(show_cats):
        mask = np.array([t == cat for t in drug_types])
        if not mask.any():
            continue
        color = CATEGORY_COLORS.get(cat, '#94a3b8')
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=color, s=25, alpha=0.75, edgecolors='white', linewidths=0.3, zorder=3 + i)
        legend_handles.append(mpatches.Patch(color=color, label=f'{cat} ({mask.sum()})'))

    legend_handles.append(mpatches.Patch(color='#94a3b8', label=f'Other ({mask_other.sum()})'))
    legend_handles.append(mpatches.Patch(color='#e2e8f0', label=f'Unknown ({mask_unk.sum()})'))

    style_ax(ax, title, f'{method_label} Dimension 1', f'{method_label} Dimension 2')
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right',
              framealpha=0.9, edgecolor='#cbd5e1', ncol=1, borderpad=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(f'{OUT}/{filename}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


# ── Load model and extract embeddings ────────────────────────────────
print("Loading Enhanced GIN v2 model...")

# Patch relative imports before loading
import importlib
import types

# Load featurizer first (no relative imports of its own)
spec_feat = importlib.util.spec_from_file_location("gnn_featurizer",
    str(PROJECT_ROOT / 'src' / 'model' / 'gnn_featurizer.py'))
gnn_featurizer = importlib.util.module_from_spec(spec_feat)
sys.modules['gnn_featurizer'] = gnn_featurizer
spec_feat.loader.exec_module(gnn_featurizer)

ATOM_FEATURE_DIM = gnn_featurizer.ATOM_FEATURE_DIM
EDGE_FEATURE_DIM = gnn_featurizer.EDGE_FEATURE_DIM
MolecularGraphFeaturizer = gnn_featurizer.MolecularGraphFeaturizer

# Create a fake parent package so relative imports work in gnn_model
model_pkg = types.ModuleType('model')
model_pkg.__path__ = [str(PROJECT_ROOT / 'src' / 'model')]
model_pkg.gnn_featurizer = gnn_featurizer
sys.modules['model'] = model_pkg
sys.modules['model.gnn_featurizer'] = gnn_featurizer

# Now load gnn_model with its parent package context
spec_model = importlib.util.spec_from_file_location("model.gnn_model",
    str(PROJECT_ROOT / 'src' / 'model' / 'gnn_model.py'),
    submodule_search_locations=[])
gnn_model_mod = importlib.util.module_from_spec(spec_model)
gnn_model_mod.__package__ = 'model'
sys.modules['model.gnn_model'] = gnn_model_mod
spec_model.loader.exec_module(gnn_model_mod)

DDIGraphModel = gnn_model_mod.DDIGraphModel

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
print(f"  Model loaded: {config.get('num_gnn_layers', '?')} layers, {config.get('hidden_dim', '?')} hidden dim")

featurizer = MolecularGraphFeaturizer(max_atoms=config.get('max_atoms', 128))

# Load drug names, SMILES, and types
print("Loading drug data from node_mapping.csv...")
drugs = []
with open(NODE_MAPPING, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        drugs.append({
            'name': row['name'],
            'smiles': row['smiles'],
            'type_fine': row.get('t_class', 'Unknown'),
            'type_coarse': map_to_coarse(row.get('t_class', 'Unknown')),
        })
print(f"  {len(drugs)} drugs loaded")

# Featurize and extract embeddings
print("Extracting embeddings from new model (this may take a minute)...")
embeddings = []
valid_drugs = []
failed = 0

with torch.no_grad():
    for i, drug in enumerate(drugs):
        graph = featurizer.smiles_to_graph(drug['smiles'])
        if graph is None:
            failed += 1
            continue

        # Add batch dimension
        nf = graph['node_features'].unsqueeze(0)
        adj = graph['adjacency'].unsqueeze(0)
        ef = graph['edge_features'].unsqueeze(0)
        nm = graph['node_mask'].unsqueeze(0)

        emb = model.encoder(nf, adj, ef, nm)  # [1, readout_dim]
        embeddings.append(emb.squeeze(0).numpy())
        valid_drugs.append(drug)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(drugs)} drugs...")

embeddings = np.array(embeddings)
print(f"  Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]}), {failed} failed SMILES")

drug_types_coarse = [d['type_coarse'] for d in valid_drugs]
coarse_counts = Counter(drug_types_coarse)
print(f"  Coarse categories: {len(set(drug_types_coarse))}")
for cat, cnt in coarse_counts.most_common():
    print(f"    {cat}: {cnt}")

# ── Save embeddings for future use ───────────────────────────────────
emb_data = {
    'model': 'Enhanced GIN v2',
    'config': config,
    'embedding_dim': int(embeddings.shape[1]),
    'num_drugs': len(valid_drugs),
    'drugs': [
        {'name': d['name'], 'type': d['type_fine'], 'category': d['type_coarse'],
         'embedding': emb.tolist()}
        for d, emb in zip(valid_drugs, embeddings)
    ]
}
emb_path = PROJECT_ROOT / 'web' / 'models' / 'gnn' / 'drug_embeddings_v2.json'
with open(emb_path, 'w') as f:
    json.dump(emb_data, f)
print(f"  Saved embeddings to {emb_path}")

# ── t-SNE from NEW model embeddings ──────────────────────────────────
print("\nGenerating t-SNE from new model embeddings...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1500, learning_rate='auto')
coords_tsne = tsne.fit_transform(embeddings)
plot_embedding(
    coords_tsne, drug_types_coarse,
    f'Enhanced GIN v2 Embedding Space - t-SNE (n={len(valid_drugs):,} drugs)',
    '18_tsne_coarse_categories.png', 't-SNE'
)

# ── UMAP from NEW model embeddings ───────────────────────────────────
if HAS_UMAP:
    print("\nGenerating UMAP from new model embeddings...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
    coords_umap = reducer.fit_transform(embeddings)
    plot_embedding(
        coords_umap, drug_types_coarse,
        f'Enhanced GIN v2 Embedding Space - UMAP (n={len(valid_drugs):,} drugs)',
        '19_umap_coarse_categories.png', 'UMAP'
    )

    # ── Combined side-by-side ─────────────────────────────────────────
    print("\nGenerating combined t-SNE + UMAP...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    fig.patch.set_facecolor('white')

    for ax, coords, method in [(ax1, coords_tsne, 't-SNE'), (ax2, coords_umap, 'UMAP')]:
        show_cats = [c for c, _ in coarse_counts.most_common() if c not in ('Unknown', 'Other')]
        mask_unk = np.array([t == 'Unknown' for t in drug_types_coarse])
        mask_other = np.array([t == 'Other' for t in drug_types_coarse])

        if mask_unk.any():
            ax.scatter(coords[mask_unk, 0], coords[mask_unk, 1],
                       c='#e2e8f0', s=6, alpha=0.2, edgecolors='none', zorder=1)
        if mask_other.any():
            ax.scatter(coords[mask_other, 0], coords[mask_other, 1],
                       c='#94a3b8', s=8, alpha=0.3, edgecolors='none', zorder=2)

        for i, cat in enumerate(show_cats):
            mask = np.array([t == cat for t in drug_types_coarse])
            if not mask.any():
                continue
            color = CATEGORY_COLORS.get(cat, '#94a3b8')
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=color, s=18, alpha=0.7, edgecolors='white', linewidths=0.2, zorder=3 + i)

        style_ax(ax, f'{method} Projection', f'{method} Dim 1', f'{method} Dim 2')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    legend_handles = []
    for cat in [c for c, _ in coarse_counts.most_common() if c not in ('Unknown', 'Other')]:
        legend_handles.append(mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#94a3b8'),
                                             label=f'{cat} ({coarse_counts[cat]})'))
    legend_handles.append(mpatches.Patch(color='#94a3b8', label=f'Other ({coarse_counts.get("Other", 0)})'))
    legend_handles.append(mpatches.Patch(color='#e2e8f0', label=f'Unknown ({coarse_counts.get("Unknown", 0)})'))

    fig.legend(handles=legend_handles, fontsize=7, loc='lower center',
               ncol=6, framealpha=0.9, edgecolor='#cbd5e1', bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Enhanced GIN v2 Drug Embeddings - {len(valid_drugs):,} Drugs, 12 Pharmacological Categories',
                 fontsize=13, fontweight='bold', color='#0f172a', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/20_tsne_umap_combined.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: 20_tsne_umap_combined.png")


# ── Category distribution bar chart ──────────────────────────────────
print("\nGenerating category distribution chart...")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor('white')

sorted_cats = sorted(coarse_counts.items(), key=lambda x: -x[1])
cats = [c for c, _ in sorted_cats]
counts = [n for _, n in sorted_cats]
colors = [CATEGORY_COLORS.get(c, '#94a3b8') for c in cats]

bars = ax.barh(range(len(cats)), counts, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(cats, fontsize=8)
ax.invert_yaxis()
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            str(count), va='center', fontsize=7, color='#334155', fontweight='bold')
style_ax(ax, f'Drug Distribution by Pharmacological Category (n={sum(counts):,})', 'Number of Drugs')
plt.tight_layout()
plt.savefig(f'{OUT}/21_category_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 21_category_distribution.png")

print(f"\n{'='*60}")
print("ALL VISUALIZATIONS GENERATED FROM NEW MODEL EMBEDDINGS")
print(f"Model: Enhanced GIN v2 ({config.get('num_gnn_layers')} layers, {config.get('hidden_dim')} hidden)")
print(f"Drugs: {len(valid_drugs)} successfully embedded ({failed} failed)")
print(f"Embedding dim: {embeddings.shape[1]}")
print(f"{'='*60}")
