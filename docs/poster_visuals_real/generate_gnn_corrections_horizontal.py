import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass
plt.rcParams['figure.facecolor'] = 'white'

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.set_xlim(-0.5, 12.5)
ax.set_ylim(-0.5, 7.5)
ax.set_axis_off()

# Define boxes: x, y, width, height, color, text
# Horizontal layout, 3 across the top, 1 centering the bottom to form a loop
boxes = {
    "GNN": (0.5, 4.5, 2.8, 1.5, "#deebf7", "1. GNN Inference\n(Predicts DDI Alerts)"),
    "UI": (4.6, 4.5, 2.8, 1.5, "#fbe5d6", "2. Expert Dashboard\n(Clinician Review)"),
    "DB": (8.7, 4.5, 2.8, 1.5, "#f8cbad", "3. Verified DB (Neo4j)\n(Stores Ground Truth)"),
    "Retrain": (4.6, 1.5, 2.8, 1.5, "#e1d5e7", "4. Active Retraining\n(Recalibrates Weights)"),
}

centers = {}

# Draw boxes
for k, v in boxes.items():
    x, y, w, h, c, t = v
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                  edgecolor='#666666', facecolor=c, linewidth=2, zorder=2)
    ax.add_patch(box)
    cx = x + w/2
    cy = y + h/2
    centers[k] = (cx, cy)
    ax.text(cx, cy, t, ha='center', va='center', fontsize=12, fontweight='bold', color='#222222', zorder=3)

def draw_arrow(k1, k2, text="", rad="0.0", x_offset=0, y_offset=0, align_x=None, align_y=None):
    xy1 = centers[k1]
    xy2 = centers[k2]
    
    # Calculate midpoint for text
    mid_x = (xy1[0] + xy2[0]) / 2 + x_offset
    mid_y = (xy1[1] + xy2[1]) / 2 + y_offset
    
    if align_x is not None: mid_x = align_x
    if align_y is not None: mid_y = align_y

    ax.annotate("", xy=xy2, xycoords='data',
                xytext=xy1, textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.8", color="#555555",
                                connectionstyle=f"arc3,rad={rad}", lw=2.5, shrinkA=45, shrinkB=45),
                zorder=1)
    
    if text:
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, color='#333333',
                fontstyle='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8), zorder=4)

# Draw precise arrows for the horizontal loop
draw_arrow("GNN", "UI", "Outputs Alert")
draw_arrow("UI", "DB", "Logs Correction\n(Overrides)")

# Curved bottom arrows to form the loop
draw_arrow("DB", "Retrain", "Extracts Corrected Batch", rad="0.2", x_offset=1.5, y_offset=0.5)
draw_arrow("Retrain", "GNN", "Deploys Updated Model\n(Higher Accuracy)", rad="0.2", x_offset=-1.5, y_offset=0.5)

# Background groupings
rect_system = mpatches.Rectangle((0.1, 1.0), 11.8, 5.8, fill=False, edgecolor='#3182bd', linewidth=2, linestyle='--', zorder=0)
ax.add_patch(rect_system)
ax.text(0.3, 7.0, "Continuous Corrections & Retraining System", color='#3182bd', fontsize=14, fontweight='bold', zorder=1)

plt.title('Expert Feedback Loop For GNN Accuracy Enhancement', fontsize=20, fontweight='bold', color='#111111', pad=15)
plt.tight_layout()

output_path = 'C:\\Users\\1kibr\\Documents\\WebDevelopment\\DDI_PROJECTV2-FRONTEND\\molecular-ai\\docs\\poster_visuals_real\\25_gnn_corrections_horizontal.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved {output_path}")
