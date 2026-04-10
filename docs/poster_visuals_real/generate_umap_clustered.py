"""
Generate UMAP + t-SNE visualizations with COARSE pharmacological groupings.
Maps 319 fine-grained drug types → 12 broad categories for visual clarity.
All data from real model embeddings (neo4j_gnn_dataset.pt + node_mapping.csv).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import csv
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

NODE_MAPPING = PROJECT_ROOT / 'web' / 'data' / 'node_mapping.csv'
DATASET_PT = PROJECT_ROOT / 'web' / 'data' / 'neo4j_gnn_dataset.pt'
GNN_REAL_DATA = PROJECT_ROOT / 'src' / 'assets' / 'gnn_real_data.json'

# ── Coarse category mapping ──────────────────────────────────────────
# Groups 319 fine types into ~12 broad pharmacological categories
COARSE_MAP = {
    # CNS / Neuro
    'Benzodiazepine': 'CNS / Neuro',
    'Atypical Antipsychotic': 'CNS / Neuro',
    'Antidepressant': 'CNS / Neuro',
    'Decreased Central Nervous System Organized Electrical Activity': 'CNS / Neuro',
    'Decreased Central Nervous System Disorganized Electrical Activity': 'CNS / Neuro',
    'Decreased Organized Electrical Activity': 'CNS / Neuro',
    'Opioid': 'CNS / Neuro',
    'Anesthetic': 'CNS / Neuro',
    'Barbiturate': 'CNS / Neuro',
    'Anticonvulsant': 'CNS / Neuro',
    'SSRI': 'CNS / Neuro',
    'SNRI': 'CNS / Neuro',
    'Anxiolytic': 'CNS / Neuro',
    'Sedative': 'CNS / Neuro',
    'Hypnotic': 'CNS / Neuro',
    'Antipsychotic': 'CNS / Neuro',
    'Typical Antipsychotic': 'CNS / Neuro',
    'Mood Stabilizer': 'CNS / Neuro',
    'Dopamine Agonist': 'CNS / Neuro',
    'Dopamine Antagonist': 'CNS / Neuro',
    'Cholinesterase Inhibitor': 'CNS / Neuro',
    'Muscle Relaxant': 'CNS / Neuro',
    'Neuromuscular Blocker': 'CNS / Neuro',

    # Anti-infectives
    'Antibiotic': 'Anti-infectives',
    'Antiviral': 'Anti-infectives',
    'Antifungal': 'Anti-infectives',
    'Penicillin Antibiotic': 'Anti-infectives',
    'Cephalosporin': 'Anti-infectives',
    'Fluoroquinolone': 'Anti-infectives',
    'Macrolide': 'Anti-infectives',
    'Aminoglycoside': 'Anti-infectives',
    'Tetracycline': 'Anti-infectives',
    'Sulfonamide': 'Anti-infectives',
    'Antiparasitic': 'Anti-infectives',
    'Antimalarial': 'Anti-infectives',
    'Antitubercular': 'Anti-infectives',
    'Antiretroviral': 'Anti-infectives',
    'Protease Inhibitor': 'Anti-infectives',

    # Cardiovascular
    'Alpha Blocker': 'Cardiovascular',
    'Beta Blocker': 'Cardiovascular',
    'ACE Inhibitor': 'Cardiovascular',
    'ARB': 'Cardiovascular',
    'Calcium Channel Blocker': 'Cardiovascular',
    'Statin': 'Cardiovascular',
    'Antiarrhythmic': 'Cardiovascular',
    'Vasodilator': 'Cardiovascular',
    'Anticoagulant': 'Cardiovascular',
    'Antiplatelet': 'Cardiovascular',
    'Diuretic': 'Cardiovascular',
    'Loop Diuretic': 'Cardiovascular',
    'Thiazide': 'Cardiovascular',
    'Cardiac Glycoside': 'Cardiovascular',
    'Nitrate': 'Cardiovascular',
    'Fibrate': 'Cardiovascular',

    # Oncology / Cytotoxic
    'Decreased DNA Integrity': 'Oncology',
    'Cellular Activity Alteration': 'Oncology',
    'Antineoplastic': 'Oncology',
    'Alkylating Agent': 'Oncology',
    'Antimetabolite': 'Oncology',
    'Topoisomerase Inhibitor': 'Oncology',
    'Kinase Inhibitor': 'Oncology',
    'Monoclonal Antibody': 'Oncology',

    # Anti-inflammatory / Pain
    'Decreased Prostaglandin Production': 'Anti-inflammatory',
    'NSAID': 'Anti-inflammatory',
    'COX-2 Inhibitor': 'Anti-inflammatory',
    'Analgesic': 'Anti-inflammatory',
    'Corticosteroid': 'Immunology',
    'Glucocorticoid': 'Immunology',

    # Respiratory
    'Bronchodilation': 'Respiratory',
    'Decreased Histamine Activity': 'Respiratory',
    'Antihistamine': 'Respiratory',
    'Leukotriene Antagonist': 'Respiratory',
    'Decongestant': 'Respiratory',

    # Immunology
    'Immunosuppressant': 'Immunology',
    'DMARD': 'Immunology',

    # Endocrine / Metabolic
    'Decreased Cell Wall Synthesis & Repair': 'Metabolic',
    'Proton Pump Inhibitor': 'GI / Metabolic',
    'H2 Blocker': 'GI / Metabolic',
    'Laxative': 'GI / Metabolic',
    'Antiemetic': 'GI / Metabolic',
    'Insulin': 'Endocrine',
    'Sulfonylurea': 'Endocrine',
    'Thyroid': 'Endocrine',
    'Estrogen': 'Endocrine',
    'Androgen': 'Endocrine',
    'Bisphosphonate': 'Endocrine',
}

# Colors for coarse categories — distinct, poster-worthy palette
CATEGORY_COLORS = {
    'CNS / Neuro':       '#6366f1',   # Indigo
    'Anti-infectives':   '#06b6d4',   # Cyan
    'Cardiovascular':    '#ef4444',   # Red
    'Oncology':          '#f97316',   # Orange
    'Anti-inflammatory': '#eab308',   # Yellow
    'Respiratory':       '#22c55e',   # Green
    'Immunology':        '#a855f7',   # Purple
    'Endocrine':         '#ec4899',   # Pink
    'GI / Metabolic':    '#14b8a6',   # Teal
    'Metabolic':         '#f59e0b',   # Amber
    'Other':             '#94a3b8',   # Slate (muted)
    'Unknown':           '#cbd5e1',   # Light grey
}


def map_to_coarse(fine_type):
    """Map a fine-grained drug type to a coarse category."""
    if fine_type == 'Unknown':
        return 'Unknown'
    if fine_type in COARSE_MAP:
        return COARSE_MAP[fine_type]
    # Keyword fallback
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


# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")

# Load gnn_real_data.json (has 3D t-SNE positions from real model embeddings)
with open(GNN_REAL_DATA) as f:
    gnn_data = json.load(f)

nodes = gnn_data['nodes']
# Extract existing 3D t-SNE positions (real, from model embeddings)
positions_3d = np.array([n['pos'] for n in nodes])
print(f"  Loaded {len(nodes)} nodes with 3D t-SNE positions")

# Load drug names and fine-grained types from node_mapping
drug_names = []
drug_types_fine = []
name_to_fine_type = {}
with open(NODE_MAPPING, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        drug_names.append(row['name'])
        t = row.get('t_class', 'Unknown')
        drug_types_fine.append(t)
        name_to_fine_type[row['name']] = t

# Match node order from gnn_real_data to node_mapping
# gnn_real_data nodes may be in different order, so map by name
drug_types_fine_ordered = []
for n in nodes:
    drug_types_fine_ordered.append(name_to_fine_type.get(n['name'], n.get('type', 'Unknown')))
drug_types_fine = drug_types_fine_ordered

# Map to coarse categories
drug_types_coarse = [map_to_coarse(t) for t in drug_types_fine]
coarse_counts = Counter(drug_types_coarse)
print(f"  Coarse categories: {len(set(drug_types_coarse))}")
for cat, cnt in coarse_counts.most_common():
    print(f"    {cat}: {cnt}")


# ── Helper: plot embedding scatter ───────────────────────────────────
def plot_embedding(coords_2d, drug_types, title, filename, method_label):
    """Create a publication-quality scatter plot of drug embeddings."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
    fig.patch.set_facecolor('white')

    # Determine which categories to show (exclude Unknown, sort by count)
    cat_counts = Counter(drug_types)
    show_cats = [c for c, _ in cat_counts.most_common() if c not in ('Unknown', 'Other')]

    # Plot Unknown first (background, very faint)
    mask_unk = np.array([t == 'Unknown' for t in drug_types])
    if mask_unk.any():
        ax.scatter(coords_2d[mask_unk, 0], coords_2d[mask_unk, 1],
                   c='#e2e8f0', s=8, alpha=0.25, edgecolors='none', zorder=1)

    # Plot Other next
    mask_other = np.array([t == 'Other' for t in drug_types])
    if mask_other.any():
        ax.scatter(coords_2d[mask_other, 0], coords_2d[mask_other, 1],
                   c='#94a3b8', s=12, alpha=0.35, edgecolors='none', zorder=2)

    # Plot named categories with distinct colors
    legend_handles = []
    for i, cat in enumerate(show_cats):
        mask = np.array([t == cat for t in drug_types])
        if not mask.any():
            continue
        color = CATEGORY_COLORS.get(cat, '#94a3b8')
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=color, s=25, alpha=0.75, edgecolors='white', linewidths=0.3, zorder=3 + i)
        legend_handles.append(
            mpatches.Patch(color=color, label=f'{cat} ({mask.sum()})')
        )

    # Add Unknown + Other to legend
    legend_handles.append(mpatches.Patch(color='#94a3b8', label=f'Other ({mask_other.sum()})'))
    legend_handles.append(mpatches.Patch(color='#e2e8f0', label=f'Unknown ({mask_unk.sum()})'))

    style_ax(ax, title, f'{method_label} Dimension 1', f'{method_label} Dimension 2')
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right',
              framealpha=0.9, edgecolor='#cbd5e1', ncol=1, borderpad=0.8)

    # Remove tick labels (dimensionless)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(f'{OUT}/{filename}', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


# ── Generate t-SNE (2D projection of existing 3D t-SNE positions) ────
print("\nProjecting 3D t-SNE to 2D...")
# Use the first 2 dimensions of the existing 3D t-SNE (real model embeddings)
coords_tsne = positions_3d[:, :2]
plot_embedding(
    coords_tsne, drug_types_coarse,
    f'GNN Drug Embedding Space — t-SNE (n={len(nodes):,} drugs)',
    '18_tsne_coarse_categories.png', 't-SNE'
)


# ── Generate UMAP (from 3D embedding positions) ──────────────────────
if HAS_UMAP:
    print("\nGenerating UMAP from 3D embedding positions...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.15, metric='euclidean')
    coords_umap = reducer.fit_transform(positions_3d)
    plot_embedding(
        coords_umap, drug_types_coarse,
        f'Drug Embedding Space — UMAP (n={len(drug_names):,} drugs, 12 pharmacological categories)',
        '19_umap_coarse_categories.png', 'UMAP'
    )

    # ── Combined side-by-side ─────────────────────────────────────────
    print("\nGenerating combined t-SNE + UMAP...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)
    fig.patch.set_facecolor('white')

    for ax, coords, method in [(ax1, coords_tsne, 't-SNE'), (ax2, coords_umap, 'UMAP')]:
        cat_counts = Counter(drug_types_coarse)
        show_cats = [c for c, _ in cat_counts.most_common() if c not in ('Unknown', 'Other')]

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

    # Shared legend
    legend_handles = []
    for cat in [c for c, _ in Counter(drug_types_coarse).most_common() if c not in ('Unknown', 'Other')]:
        cnt = sum(1 for t in drug_types_coarse if t == cat)
        legend_handles.append(mpatches.Patch(color=CATEGORY_COLORS.get(cat, '#94a3b8'), label=f'{cat} ({cnt})'))
    legend_handles.append(mpatches.Patch(color='#94a3b8', label=f'Other ({sum(1 for t in drug_types_coarse if t == "Other")})'))
    legend_handles.append(mpatches.Patch(color='#e2e8f0', label=f'Unknown ({sum(1 for t in drug_types_coarse if t == "Unknown")})'))

    fig.legend(handles=legend_handles, fontsize=7, loc='lower center',
               ncol=6, framealpha=0.9, edgecolor='#cbd5e1', bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f'GNN Drug Embedding Space — {len(nodes):,} Drugs Across {len(set(drug_types_coarse))} Pharmacological Categories',
        fontsize=13,
        fontweight='bold',
        color='#0f172a',
        y=1.01,
    )
    fig.subplots_adjust(bottom=0.16)
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
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

print(f"\n{'='*50}")
print("ALL EMBEDDING VISUALIZATIONS GENERATED")
print(f"{'='*50}")
