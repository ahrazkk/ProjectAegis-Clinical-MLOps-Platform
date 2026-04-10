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

fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(-0.5, 10.5)
ax.set_axis_off()

# Define boxes: x, y, width, height, color, text
boxes = {
    "Data": (0, 8, 3, 1.4, "#e2f0d9", "1. Raw Drug Data\n(SMILES, Medical Records)"),
    "Init": (4, 8, 3, 1.4, "#deebf7", "2. Feature Tensor Init\n(Chemical Signatures)"),
    "Conv": (8, 8, 3, 1.4, "#deebf7", "3. Message Passing Layers\n(Neighborhood Aggregation)"),
    "Embed": (8, 5, 3, 1.4, "#deebf7", "4. Latent Embeddings\n(The 'GNN Galaxy')"),
    "Predict": (8, 2, 3, 1.4, "#fff2cc", "5. Probability Classifier\n(Predicts DDI Likelihood)"),
    "UI": (4, 2, 3, 1.4, "#fbe5d6", "6. Clinician Dashboard\n(Human-in-the-Loop Review)"),
    "DB": (0, 2, 3, 1.4, "#f8cbad", "7. Neo4j Knowledge Graph\n(Validates & Flags Edges)"),
    "Retrain": (0, 5, 3, 1.4, "#e1d5e7", "8. Active Recalibration\n(Model Weight Updates)"),
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

def draw_arrow(k1, k2, text="", rad="0.0", x_offset=0, y_offset=0):
    xy1 = centers[k1]
    xy2 = centers[k2]
    
    # Calculate midpoint for text
    mid_x = (xy1[0] + xy2[0]) / 2 + x_offset
    mid_y = (xy1[1] + xy2[1]) / 2 + y_offset
    
    ax.annotate("", xy=xy2, xycoords='data',
                xytext=xy1, textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=0.8", color="#555555",
                                connectionstyle=f"arc3,rad={rad}", lw=2.5, shrinkA=45, shrinkB=45),
                zorder=1)
    
    if text:
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, color='#333333',
                fontstyle='italic', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8), zorder=4)

# Draw arrows linking the architecture
draw_arrow("Data", "Init", "Parses structures")
draw_arrow("Init", "Conv", "Graph Formulation")
draw_arrow("Conv", "Embed", "Dimensionality Reduction")
draw_arrow("Embed", "Predict", "Link Prediction (Sigmoid)")
draw_arrow("Predict", "UI", "Outputs High Risk Alerts")
draw_arrow("UI", "DB", "Expert Flags False Positives")
draw_arrow("DB", "Retrain", "Triggers Graph Update")

# Arrow from Retrain back to Conv for the continuous learning loop
draw_arrow("Retrain", "Conv", "Propagates Error Gradients", rad="0.2", x_offset=-1)

# Groupings/Legends
rect_gnn = mpatches.Rectangle((3.5, 4.5), 7.8, 5.3, fill=False, edgecolor='#3182bd', linewidth=2, linestyle='--', zorder=0)
ax.add_patch(rect_gnn)
ax.text(3.6, 9.5, "Graph Neural Network (GNN) Core", color='#3182bd', fontsize=14, fontweight='bold', zorder=1)

rect_human = mpatches.Rectangle((-0.5, 1.5), 7.8, 5.3, fill=False, edgecolor='#d95f02', linewidth=2, linestyle='--', zorder=0)
ax.add_patch(rect_human)
ax.text(-0.4, 1.6, "Expert Corrections System", color='#d95f02', fontsize=14, fontweight='bold', zorder=1)

plt.title('GNN Architecture & Continuous Corrections Loop', fontsize=22, fontweight='bold', color='#111111', pad=20)
plt.tight_layout()

output_path = 'C:\\Users\\1kibr\\Documents\\WebDevelopment\\DDI_PROJECTV2-FRONTEND\\molecular-ai\\docs\\poster_visuals_real\\24_gnn_architecture_flow.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved {output_path}")
