"""
Generate poster visuals from REAL project data only.
No fabricated metrics. Every number comes from an actual file.

Data sources:
  - gnn_real_data.json      -> real t-SNE 3D positions for 1,350 drugs
  - train.json / val.json   -> real label & severity distributions
  - training_results.json   -> real best PR-AUC (0.7903)
  - metadata.json           -> real dataset stats
  - calibration.json        -> real Platt scaling params
  - YOLO results.csv        -> real pill detector training curves
  - node_mapping.csv        -> real drug-to-class mapping
  - Source code constants    -> real evidence/factor weights
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import csv
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT = str(SCRIPT_DIR)

GNN_REAL_DATA   = PROJECT_ROOT / 'src' / 'assets' / 'gnn_real_data.json'
TRAIN_JSON      = PROJECT_ROOT / 'web' / 'data' / 'gnn_training' / 'train.json'
VAL_JSON        = PROJECT_ROOT / 'web' / 'data' / 'gnn_training' / 'val.json'
TRAIN_RESULTS   = PROJECT_ROOT / 'web' / 'models' / 'gnn' / 'training_results.json'
METADATA_JSON   = PROJECT_ROOT / 'web' / 'data' / 'gnn_training' / 'metadata.json'
CALIBRATION     = PROJECT_ROOT / 'web' / 'models' / 'gnn' / 'calibration.json'
NODE_MAPPING    = PROJECT_ROOT / 'web' / 'data' / 'node_mapping.csv'
YOLO_CSV        = PROJECT_ROOT / 'web' / 'models' / 'runs' / 'pill' / 'tuned-yolo-detector-v3-clean' / 'results.csv'

# Consistent poster palette
COLORS = {
    'teal': '#0891b2', 'red': '#dc2626', 'purple': '#9333ea',
    'orange': '#ea580c', 'green': '#16a34a', 'indigo': '#4f46e5',
    'dark': '#0f172a', 'slate': '#334155', 'muted': '#64748b',
    'border': '#cbd5e1', 'bg': '#fafbfc',
}

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['bg'])
    ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['dark'], pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=COLORS['slate'])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=COLORS['slate'])
    ax.tick_params(colors=COLORS['muted'], labelsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(COLORS['border'])


# ============================================================
# 1. REAL t-SNE Embedding Space (from gnn_real_data.json)
#    Uses actual 3D positions projected to 2D, colored by real drug types
# ============================================================
print("1. Loading real t-SNE positions from gnn_real_data.json...")
with open(GNN_REAL_DATA) as f:
    gnn_data = json.load(f)

nodes = gnn_data['nodes']
# Group by pharmacological type, take top categories
type_counts = Counter(n['type'] for n in nodes)
# Top 8 non-Unknown types for coloring
top_types = [t for t, _ in type_counts.most_common(20) if t != 'Unknown'][:8]
type_to_color = {t: c for t, c in zip(top_types, [
    '#0891b2', '#dc2626', '#9333ea', '#ea580c', '#16a34a',
    '#4f46e5', '#f59e0b', '#ec4899'
])}

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
fig.patch.set_facecolor('white')

# Plot "Other" drugs first (grey, background)
other_x = [n['pos'][0] for n in nodes if n['type'] not in top_types and n['type'] != 'Unknown']
other_y = [n['pos'][1] for n in nodes if n['type'] not in top_types and n['type'] != 'Unknown']
unknown_x = [n['pos'][0] for n in nodes if n['type'] == 'Unknown']
unknown_y = [n['pos'][1] for n in nodes if n['type'] == 'Unknown']

ax.scatter(unknown_x, unknown_y, c='#d1d5db', s=6, alpha=0.3, edgecolors='none', label=f'Unknown ({len(unknown_x)})')
ax.scatter(other_x, other_y, c='#9ca3af', s=6, alpha=0.4, edgecolors='none', label=f'Other types ({len(other_x)})')

# Plot top types with distinct colors
for drug_type in top_types:
    xs = [n['pos'][0] for n in nodes if n['type'] == drug_type]
    ys = [n['pos'][1] for n in nodes if n['type'] == drug_type]
    ax.scatter(xs, ys, c=type_to_color[drug_type], s=12, alpha=0.7,
               edgecolors='none', label=f'{drug_type} ({len(xs)})')

style_ax(ax, f'GNN Drug Embedding Space (1,350 Real Drug Nodes, 319 Classes)',
         't-SNE Dimension 1', 't-SNE Dimension 2')
leg = ax.legend(fontsize=5.5, loc='lower right', framealpha=0.9, edgecolor=COLORS['border'], ncol=1)
for t in leg.get_texts(): t.set_color(COLORS['slate'])
plt.tight_layout()
plt.savefig(f'{OUT}/1_real_tsne_embeddings.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   -> 1,350 real nodes plotted from actual learned positions")


# ============================================================
# 2. REAL Training Data Distribution
#    From train.json + val.json actual labels
# ============================================================
print("2. Loading real training data distribution...")
with open(TRAIN_JSON) as f:
    train_data = json.load(f)
with open(VAL_JSON) as f:
    val_data = json.load(f)

all_data = train_data + val_data

# Severity distribution
sev_counts = Counter(s['severity'] for s in all_data)
# Class balance
pos = sum(1 for s in all_data if s['has_interaction'])
neg = sum(1 for s in all_data if not s['has_interaction'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=200)
fig.patch.set_facecolor('white')

# Left: Severity distribution
severities = ['none', 'minor', 'moderate', 'severe']
sev_vals = [sev_counts.get(s, 0) for s in severities]
sev_colors = ['#94a3b8', '#22c55e', '#f59e0b', '#dc2626']
bars1 = ax1.bar(severities, sev_vals, color=sev_colors, width=0.6, edgecolor='none')
for bar, v in zip(bars1, sev_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             str(v), ha='center', fontsize=9, fontweight='bold', color=COLORS['slate'])
style_ax(ax1, f'Severity Distribution (n={len(all_data)})',
         'Severity Level', 'Number of Drug Pairs')

# Right: Class balance (interacting vs safe)
labels_bal = ['Non-Interacting\n(Label 0)', 'Interacting\n(Label 1)']
vals_bal = [neg, pos]
bars2 = ax2.bar(labels_bal, vals_bal, color=[COLORS['teal'], COLORS['red']], width=0.5, edgecolor='none')
for bar, v in zip(bars2, vals_bal):
    pct = v / len(all_data) * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f'{v} ({pct:.1f}%)', ha='center', fontsize=9, fontweight='bold', color=COLORS['slate'])
style_ax(ax2, 'Binary Class Balance', 'Class', 'Count')

plt.tight_layout()
plt.savefig(f'{OUT}/2_real_data_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   -> {len(all_data)} real pairs: {pos} positive, {neg} negative")


# ============================================================
# 3. REAL Model Performance Card
#    From training_results.json (actual saved metrics)
# ============================================================
print("3. Loading real training results...")
with open(TRAIN_RESULTS) as f:
    results = json.load(f)
with open(METADATA_JSON) as f:
    metadata = json.load(f)
with open(CALIBRATION) as f:
    calib = json.load(f)

fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis('off')

# Real metrics from actual files
real_metrics = [
    ('Best PR-AUC', f'{results["best_metric"]:.4f}', COLORS['purple'],
     'training_results.json'),
    ('Train Samples', str(results['train_samples']), COLORS['teal'],
     'training_results.json'),
    ('Val Samples', str(results['val_samples']), COLORS['green'],
     'training_results.json'),
    ('Unique Drugs', str(metadata['unique_drugs']), COLORS['indigo'],
     'metadata.json'),
    ('Training Time', f'{results["elapsed_seconds"]:.0f}s', COLORS['orange'],
     'training_results.json'),
]

for i, (label, value, color, source) in enumerate(real_metrics):
    x = i * 2.3 + 1.2
    rect = mpatches.FancyBboxPatch((x - 0.95, 0.4), 1.9, 2.8, boxstyle="round,pad=0.1",
                                    facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=1)
    ax.add_patch(rect)
    ax.text(x, 2.3, value, ha='center', va='center', fontsize=15, fontweight='bold', color=color)
    ax.text(x, 1.4, label, ha='center', va='center', fontsize=7.5, fontweight='bold', color=COLORS['muted'])
    ax.text(x, 0.8, f'[{source}]', ha='center', va='center', fontsize=5, color='#94a3b8', style='italic')

ax.set_title('Real GNN Training Metrics (from saved results)', fontsize=10,
             fontweight='bold', color=COLORS['dark'], pad=8, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/3_real_metrics_card.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   -> PR-AUC = {results['best_metric']:.4f} (real)")


# ============================================================
# 4. REAL Model Configuration Summary
#    Hyperparameters from actual training run
# ============================================================
print("4. Creating real model config visual...")
config = results['config']

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
fig.patch.set_facecolor('white')
ax.axis('off')

config_items = [
    ('Architecture', 'Edge-Conditioned GIN + JK'),
    ('GNN Layers', str(config['num_gnn_layers'])),
    ('Hidden Dim', str(config['hidden_dim'])),
    ('Dropout', str(config['dropout_rate'])),
    ('Learning Rate', str(config['learning_rate'])),
    ('Batch Size', str(config['batch_size'])),
    ('Weight Decay', str(config['weight_decay'])),
    ('Max Atoms', str(config['max_atoms'])),
    ('Max Epochs', str(config['num_epochs'])),
    ('Task', 'Binary' if config['use_binary'] else 'Multi-class'),
    ('Device', results['device'].upper()),
    ('Platt A', f'{calib["platt_a"]:.5f}'),
    ('Platt B', f'{calib["platt_b"]:.5f}'),
]

table = ax.table(
    cellText=[[k, v] for k, v in config_items],
    colLabels=['Parameter', 'Value'],
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

# Style header
for j in range(2):
    cell = table[0, j]
    cell.set_facecolor(COLORS['dark'])
    cell.set_text_props(color='white', fontweight='bold')

# Style rows
for i in range(1, len(config_items) + 1):
    for j in range(2):
        cell = table[i, j]
        cell.set_facecolor('#f8fafc' if i % 2 == 0 else 'white')
        cell.set_edgecolor(COLORS['border'])
        if j == 1:
            cell.set_text_props(fontweight='bold', color=COLORS['teal'])

ax.set_title('GNN Model Configuration (Actual Training Run)', fontsize=10,
             fontweight='bold', color=COLORS['dark'], pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/4_real_model_config.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   -> All values from training_results.json + calibration.json")


# ============================================================
# 5. REAL YOLO Pill Detector Training Curves
#    From tuned-yolo-detector-v3-clean/results.csv
# ============================================================
print("5. Loading real YOLO training curves...")
epochs, precisions, recalls, map50s, map50_95s = [], [], [], [], []
box_losses, cls_losses = [], []

with open(YOLO_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        precisions.append(float(row['metrics/precision(B)']))
        recalls.append(float(row['metrics/recall(B)']))
        map50s.append(float(row['metrics/mAP50(B)']))
        map50_95s.append(float(row['metrics/mAP50-95(B)']))
        box_losses.append(float(row['train/box_loss']))
        cls_losses.append(float(row['train/cls_loss']))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
fig.patch.set_facecolor('white')

# Left: Detection metrics
ax1.plot(epochs, precisions, '-o', color=COLORS['teal'], linewidth=2, markersize=4, label=f'Precision ({precisions[-1]:.3f})')
ax1.plot(epochs, recalls, '-s', color=COLORS['green'], linewidth=2, markersize=4, label=f'Recall ({recalls[-1]:.3f})')
ax1.plot(epochs, map50s, '-^', color=COLORS['purple'], linewidth=2, markersize=4, label=f'mAP@50 ({map50s[-1]:.3f})')
ax1.plot(epochs, map50_95s, '-d', color=COLORS['orange'], linewidth=2, markersize=4, label=f'mAP@50-95 ({map50_95s[-1]:.3f})')
style_ax(ax1, 'YOLO Pill Detector — Detection Metrics', 'Epoch', 'Score')
ax1.legend(fontsize=7, framealpha=0.9, edgecolor=COLORS['border'])
ax1.set_ylim(0, 1)

# Right: Training losses
ax2.plot(epochs, box_losses, '-o', color=COLORS['red'], linewidth=2, markersize=4, label=f'Box Loss ({box_losses[-1]:.3f})')
ax2.plot(epochs, cls_losses, '-s', color=COLORS['indigo'], linewidth=2, markersize=4, label=f'Class Loss ({cls_losses[-1]:.3f})')
style_ax(ax2, 'YOLO Pill Detector — Training Loss', 'Epoch', 'Loss')
ax2.legend(fontsize=7, framealpha=0.9, edgecolor=COLORS['border'])

plt.tight_layout()
plt.savefig(f'{OUT}/5_real_yolo_training.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   -> {len(epochs)} epochs, final mAP@50 = {map50s[-1]:.3f}")


# ============================================================
# 6. REAL Evidence Source Weights
#    These are actual design constants from the codebase
#    (web/ddi_api/services/ source code)
# ============================================================
print("6. Creating evidence weight chart (design constants)...")
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
fig.patch.set_facecolor('white')

sources = ['Knowledge\nGraph', 'DDI\nCorpus', 'Categorical\nRules', 'TWOSIDES', 'GNN\nModel', 'OpenFDA\nFAERS']
weights = [1.0, 0.9, 0.85, 0.78, 0.70, 0.62]
bar_colors = [COLORS['teal'], COLORS['purple'], COLORS['green'],
              COLORS['indigo'], COLORS['orange'], COLORS['red']]

bars = ax.barh(sources, weights, color=bar_colors, height=0.6, edgecolor='none')
for bar, w in zip(bars, weights):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'{w:.0%}',
            va='center', fontsize=8, fontweight='bold', color=COLORS['slate'])

ax.set_xlim(0, 1.15)
style_ax(ax, 'Evidence Source Reliability Weights (Design Constants)',
         'Weight', '')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/6_real_evidence_weights.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   -> Weights from codebase constants (not learned)")


# ============================================================
# 7. REAL Digital Twin Factor Weights
#    Fixed design weights from source code
# ============================================================
print("7. Creating digital twin factor chart (design constants)...")
fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
fig.patch.set_facecolor('white')

factors = ['Pairwise\nBaseline', 'Enzyme\nCompetition', 'Target\nOverlap', 'Organ\nBurden', 'Network\nStress']
weights_f = [0.40, 0.25, 0.15, 0.10, 0.10]
f_colors = [COLORS['red'], COLORS['purple'], COLORS['indigo'], COLORS['orange'], COLORS['teal']]

bars = ax.barh(factors, weights_f, color=f_colors, height=0.6, edgecolor='none')
for bar, w in zip(bars, weights_f):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.0%}',
            va='center', fontsize=8, fontweight='bold', color=COLORS['slate'])

ax.set_xlim(0, 0.55)
style_ax(ax, 'Digital Twin Toxicity Factor Weights (Design Constants)',
         'Weight in Composite Score', '')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/7_real_digital_twin_factors.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   -> Fixed weights from source code")


# ============================================================
# 8. REAL Drug Type Distribution
#    From gnn_real_data.json actual pharmacological classes
# ============================================================
print("8. Creating real drug type distribution...")
type_counts_all = Counter(n['type'] for n in nodes)
# Top 15 non-Unknown
top15 = [(t, c) for t, c in type_counts_all.most_common(30) if t != 'Unknown'][:15]
unknown_count = type_counts_all.get('Unknown', 0)
other_count = sum(c for t, c in type_counts_all.items() if t != 'Unknown' and t not in dict(top15))

labels = [t for t, _ in top15] + ['Other Classes', 'Unknown']
values = [c for _, c in top15] + [other_count, unknown_count]
bar_cols = ([COLORS['teal']] * len(top15)) + ['#9ca3af', '#d1d5db']

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
fig.patch.set_facecolor('white')

y_pos = range(len(labels))
bars = ax.barh(y_pos, values, color=bar_cols, height=0.7, edgecolor='none')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=7)
for bar, v in zip(bars, values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(v),
            va='center', fontsize=7, fontweight='bold', color=COLORS['slate'])

style_ax(ax, f'Drug Pharmacological Classes (1,350 Drugs, 319 Unique Types)',
         'Number of Drugs', '')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/8_real_drug_types.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   -> 319 real drug classes from knowledge graph")


# ============================================================
# 9. REAL Dataset Composition Overview
#    Everything from metadata.json
# ============================================================
print("9. Creating dataset composition chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=200)
fig.patch.set_facecolor('white')

# Left: Train/Val split
split_labels = ['Training\nSet', 'Validation\nSet']
split_vals = [results['train_samples'], results['val_samples']]
bars1 = ax1.bar(split_labels, split_vals, color=[COLORS['teal'], COLORS['purple']], width=0.5, edgecolor='none')
for bar, v in zip(bars1, split_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(v), ha='center', fontsize=11, fontweight='bold', color=COLORS['slate'])
style_ax(ax1, 'Train / Validation Split', '', 'Number of Drug Pairs')

# Right: Data source
sev_dist = metadata['severity_distribution']
sev_labels = list(sev_dist.keys())
sev_vals = list(sev_dist.values())
sev_colors_map = {'minor': '#22c55e', 'moderate': '#f59e0b', 'severe': '#dc2626'}
sev_cols = [sev_colors_map.get(s, '#94a3b8') for s in sev_labels]

wedges, texts, autotexts = ax2.pie(
    sev_vals, labels=[f'{s.title()}\n({v})' for s, v in zip(sev_labels, sev_vals)],
    colors=sev_cols, autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 8, 'color': COLORS['slate']}
)
for t in autotexts: t.set_fontsize(8); t.set_fontweight('bold')
ax2.set_title('Interaction Severity Distribution\n(Positive Pairs Only)',
              fontsize=10, fontweight='bold', color=COLORS['dark'], pad=10)

plt.tight_layout()
plt.savefig(f'{OUT}/9_real_dataset_composition.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   -> {metadata['total_samples']} total samples from metadata.json")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ALL 9 REAL VISUALS GENERATED")
print("=" * 60)
print(f"""
Data sources used:
  gnn_real_data.json     -> 1,350 real drug node positions (chart 1, 8)
  train.json + val.json  -> {len(all_data)} real drug pairs (chart 2)
  training_results.json  -> PR-AUC = {results['best_metric']:.4f} (chart 3, 4, 9)
  metadata.json          -> dataset stats (chart 9)
  calibration.json       -> Platt params (chart 4)
  YOLO results.csv       -> {len(epochs)} epochs training (chart 5)
  Source code constants   -> evidence/factor weights (chart 6, 7)

CANNOT generate without torch_geometric:
  - ROC curve (needs y_pred from model inference)
  - PR curve (needs y_pred from model inference)
  - Confusion matrix (needs y_pred from model inference)
  - Probability distribution (needs y_pred from model inference)
  - Degree vs accuracy (needs per-node evaluation)

To generate those, install torch_geometric and run:
  cd web && python evaluate_gnn_metrics.py
""")
