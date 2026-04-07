"""
Generate REAL charts from actual model evaluation predictions.
Every number comes from evaluation_predictions.json (Colab output).
Zero fake data.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent.parent / 'web' / 'models' / 'gnn'
OUT = str(SCRIPT_DIR)

# Load real data
with open(MODEL_DIR / 'evaluation_predictions.json') as f:
    eval_data = json.load(f)
with open(MODEL_DIR / 'training_history.json') as f:
    history = json.load(f)
with open(MODEL_DIR / 'training_results.json') as f:
    results = json.load(f)

y_true = np.array(eval_data['y_true'])
y_scores = np.array(eval_data['y_scores'])
y_pred = (y_scores >= 0.5).astype(int)
metrics = results['final_metrics']

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

print(f"Loaded {len(y_true)} real predictions")
print(f"Real metrics: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

# ============================================================
# 1. REAL ROC Curve
# ============================================================
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc_val = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
fig.patch.set_facecolor('white')
ax.plot(fpr, tpr, color=COLORS['teal'], linewidth=2.5, label=f'GNN ROC (AUC = {roc_auc_val:.4f})')
ax.plot([0, 1], [0, 1], '--', color=COLORS['muted'], linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['teal'])
style_ax(ax, f'ROC Curve (n={len(y_true)} real predictions)', 'False Positive Rate', 'True Positive Rate')
ax.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor=COLORS['border'])
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/10_real_roc_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  1. ROC curve saved")

# ============================================================
# 2. REAL PR Curve
# ============================================================
precision_c, recall_c, _ = precision_recall_curve(y_true, y_scores)
pr_auc_val = auc(recall_c, precision_c)

fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
fig.patch.set_facecolor('white')
ax.plot(recall_c, precision_c, color=COLORS['purple'], linewidth=2.5, label=f'PR Curve (AUC = {pr_auc_val:.4f})')
ax.fill_between(recall_c, precision_c, alpha=0.1, color=COLORS['purple'])
baseline = y_true.sum() / len(y_true)
ax.axhline(y=baseline, linestyle='--', color=COLORS['muted'], linewidth=1, alpha=0.5, label=f'Baseline ({baseline:.2f})')
style_ax(ax, f'Precision-Recall Curve (n={len(y_true)})', 'Recall', 'Precision')
ax.legend(fontsize=8, loc='lower left', framealpha=0.9, edgecolor=COLORS['border'])
ax.set_xlim(-0.02, 1.02); ax.set_ylim(max(0, baseline - 0.1), 1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/11_real_pr_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  2. PR curve saved")

# ============================================================
# 3. REAL ROC + PR Combined (poster-ready)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
fig.patch.set_facecolor('white')

ax1.plot(fpr, tpr, color=COLORS['teal'], linewidth=2.5, label=f'ROC (AUC = {roc_auc_val:.4f})')
ax1.plot([0, 1], [0, 1], '--', color=COLORS['muted'], linewidth=1, alpha=0.5)
ax1.fill_between(fpr, tpr, alpha=0.08, color=COLORS['teal'])
style_ax(ax1, 'ROC Curve', 'False Positive Rate', 'True Positive Rate')
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor=COLORS['border'])

ax2.plot(recall_c, precision_c, color=COLORS['purple'], linewidth=2.5, label=f'PR (AUC = {pr_auc_val:.4f})')
ax2.fill_between(recall_c, precision_c, alpha=0.08, color=COLORS['purple'])
style_ax(ax2, 'Precision-Recall Curve', 'Recall', 'Precision')
ax2.legend(fontsize=8, loc='lower left', framealpha=0.9, edgecolor=COLORS['border'])
ax2.set_ylim(0.9, 1.005)

plt.tight_layout()
plt.savefig(f'{OUT}/12_real_roc_pr_combined.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  3. Combined ROC+PR saved")

# ============================================================
# 4. REAL Confusion Matrix
# ============================================================
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
fig.patch.set_facecolor('white')

cm_colors = [[COLORS['teal'], '#fca5a5'], ['#fca5a5', COLORS['green']]]
labels = [['True Neg', 'False Pos'], ['False Neg', 'True Pos']]
values = [[tn, fp], [fn, tp]]

for i in range(2):
    for j in range(2):
        color = cm_colors[i][j]
        ax.add_patch(plt.Rectangle((j, 1-i), 0.95, 0.95, facecolor=color, alpha=0.3, edgecolor=COLORS['border']))
        ax.text(j + 0.475, 1.55 - i, str(values[i][j]), ha='center', va='center',
                fontsize=22, fontweight='bold', color=COLORS['dark'])
        ax.text(j + 0.475, 1.25 - i, labels[i][j], ha='center', va='center',
                fontsize=8, color=COLORS['muted'])

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.475, 1.475]); ax.set_xticklabels(['Safe (0)', 'Interacting (1)'], fontsize=8, color=COLORS['slate'])
ax.set_yticks([0.475, 1.475]); ax.set_yticklabels(['Interacting (1)', 'Safe (0)'], fontsize=8, color=COLORS['slate'])
ax.set_xlabel('Predicted', fontsize=9, color=COLORS['slate'])
ax.set_ylabel('Actual', fontsize=9, color=COLORS['slate'])
ax.set_title(f'Confusion Matrix (n={len(y_true):,} real predictions)', fontsize=11,
             fontweight='bold', color=COLORS['dark'], pad=10)
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/13_real_confusion_matrix.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  4. Confusion matrix saved (TN={tn} FP={fp} FN={fn} TP={tp})")

# ============================================================
# 5. REAL Probability Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
fig.patch.set_facecolor('white')

safe_scores = y_scores[y_true == 0]
inter_scores = y_scores[y_true == 1]

ax.hist(safe_scores, bins=50, alpha=0.6, color=COLORS['teal'], label=f'Safe pairs (n={len(safe_scores)})', density=True, edgecolor='none')
ax.hist(inter_scores, bins=50, alpha=0.6, color=COLORS['red'], label=f'Interacting pairs (n={len(inter_scores)})', density=True, edgecolor='none')
ax.axvline(x=0.5, color=COLORS['dark'], linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.5)')
style_ax(ax, f'Prediction Probability Distribution (n={len(y_true):,})',
         'Predicted Probability', 'Density')
ax.legend(fontsize=8, framealpha=0.9, edgecolor=COLORS['border'])
plt.tight_layout()
plt.savefig(f'{OUT}/14_real_probability_distribution.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  5. Probability distribution saved")

# ============================================================
# 6. REAL Training Loss Curves
# ============================================================
epochs = [h['epoch'] for h in history]
train_losses = [h['train_loss'] for h in history]
val_losses = [h['val_loss'] for h in history]
pr_aucs = [h['pr_auc'] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
fig.patch.set_facecolor('white')

ax1.plot(epochs, train_losses, color=COLORS['teal'], linewidth=2, label='Train Loss')
ax1.plot(epochs, val_losses, color=COLORS['red'], linewidth=2, label='Val Loss')
style_ax(ax1, 'Training & Validation Loss', 'Epoch', 'Focal Loss')
ax1.legend(fontsize=8, framealpha=0.9, edgecolor=COLORS['border'])

ax2.plot(epochs, pr_aucs, color=COLORS['purple'], linewidth=2, label='Val PR-AUC')
best_epoch = max(range(len(pr_aucs)), key=lambda i: pr_aucs[i])
ax2.scatter([epochs[best_epoch]], [pr_aucs[best_epoch]], color=COLORS['red'], s=60, zorder=5,
            label=f'Best: {pr_aucs[best_epoch]:.4f} (epoch {epochs[best_epoch]})')
style_ax(ax2, 'Validation PR-AUC', 'Epoch', 'PR-AUC')
ax2.legend(fontsize=8, framealpha=0.9, edgecolor=COLORS['border'])
ax2.set_ylim(0.96, 1.0)

plt.tight_layout()
plt.savefig(f'{OUT}/15_real_training_curves.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  6. Training curves saved")

# ============================================================
# 7. REAL Metrics Card (from actual training)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 3), dpi=200)
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

real_metrics = [
    ('PR-AUC', f'{metrics["pr_auc"]:.4f}', COLORS['purple']),
    ('ROC-AUC', f'{metrics["roc_auc"]:.4f}', COLORS['teal']),
    ('Accuracy', f'{metrics["accuracy"]*100:.1f}%', COLORS['green']),
    ('Precision', f'{metrics["precision"]*100:.1f}%', COLORS['orange']),
    ('Recall', f'{metrics["recall"]*100:.1f}%', COLORS['red']),
    ('F1 Score', f'{metrics["f1"]:.4f}', COLORS['indigo']),
]

for i, (label, value, color) in enumerate(real_metrics):
    x = i * 2.2 + 1.2
    rect = mpatches.FancyBboxPatch((x - 0.95, 0.4), 1.9, 2.8, boxstyle="round,pad=0.1",
                                    facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=1)
    ax.add_patch(rect)
    ax.text(x, 2.3, value, ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    ax.text(x, 1.2, label, ha='center', va='center', fontsize=8, fontweight='bold', color=COLORS['muted'])

ax.set_title(f'Enhanced GNN v2 — Real Metrics ({results["train_samples"]:,} train, {results["val_samples"]:,} val)',
             fontsize=10, fontweight='bold', color=COLORS['dark'], pad=8, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/16_real_metrics_card.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  7. Metrics card saved")

# ============================================================
# 8. REAL F1/Precision/Recall over epochs
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
fig.patch.set_facecolor('white')

ax.plot(epochs, [h['precision'] for h in history], color=COLORS['orange'], linewidth=2, label='Precision')
ax.plot(epochs, [h['recall'] for h in history], color=COLORS['red'], linewidth=2, label='Recall')
ax.plot(epochs, [h['f1'] for h in history], color=COLORS['green'], linewidth=2, label='F1')
ax.plot(epochs, [h['accuracy'] for h in history], color=COLORS['teal'], linewidth=2, label='Accuracy', linestyle='--')
style_ax(ax, 'Precision, Recall, F1, Accuracy over Training', 'Epoch', 'Score')
ax.legend(fontsize=8, framealpha=0.9, edgecolor=COLORS['border'])
ax.set_ylim(0.75, 1.0)
plt.tight_layout()
plt.savefig(f'{OUT}/17_real_metric_curves.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  8. Metric curves saved")

print(f"\n{'='*60}")
print(f"ALL 8 REAL CHARTS GENERATED FROM ACTUAL MODEL PREDICTIONS")
print(f"{'='*60}")
print(f"Source: evaluation_predictions.json ({len(y_true)} samples)")
print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
print(f"Accuracy:  {metrics['accuracy']*100:.1f}%")
print(f"Precision: {metrics['precision']*100:.1f}%")
print(f"Recall:    {metrics['recall']*100:.1f}%")
print(f"F1:        {metrics['f1']:.4f}")
print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
