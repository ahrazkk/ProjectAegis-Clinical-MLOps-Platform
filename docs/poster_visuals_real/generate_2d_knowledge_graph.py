import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Nodes
# Format: {ID: [x, y, radius, bg_color, label, text_color]}
nodes = {
    "Drug1": [2.5, 5, 0.9, "#e0f3db", "Atorvastatin\n(Statin)", "#2b8cbe"],
    "Drug2": [7.5, 5, 0.9, "#e0f3db", "Amiodarone\n(Antiarrhythmic)", "#2b8cbe"],
    "Target": [5, 7.5, 0.7, "#fee0d2", "CYP3A4\n(Liver Enzyme)", "#de2d26"],
    "AE": [5, 2.5, 0.8, "#fdbb84", "Myopathy /\nRhabdomyolysis", "#c51b8a"],
    "Lit": [5, 9.5, 0.5, "#ece7f2", "PubMed ID:\n153...", "#807dba"]
}

# Edges
# Format: [(n1_id, n2_id, label, x_offset, y_offset)]
edges = [
    ("Drug1", "Target", "metabolized_by", -0.9, 0.3),
    ("Drug2", "Target", "inhibits", 0.9, 0.3),
    ("Drug1", "AE", "increases_risk_of", -1.0, -0.2),
    ("Drug2", "AE", "increases_risk_of", 1.0, -0.2),
    ("Lit", "Target", "documents", 0, 0),
    ("Drug1", "Drug2", "AI DDI Alert: Severe", 0, -0.2)
]

# Draw Edges
for n1, n2, label, ox, oy in edges:
    x1, y1 = nodes[n1][0], nodes[n1][1]
    x2, y2 = nodes[n2][0], nodes[n2][1]
    
    color = "#e51c23" if "Alert" in label else "#999999"
    style = "dashed" if "Alert" in label else "solid"
    lw = 2.5 if "Alert" in label else 1.5
    
    ax.annotate("",
                xy=(x2, y2), xycoords='data',
                xytext=(x1, y1), textcoords='data',
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.6", color=color, ls=style, lw=lw, shrinkA=45, shrinkB=45,
                                connectionstyle="arc3,rad=0.1"))
    
    # Midpoint for text
    mx, my = (x1 + x2)/2 + ox, (y1 + y2)/2 + oy
    
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    if "Alert" in label:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#e51c23", alpha=1)
        
    ax.text(mx, my, label.replace("_", " "), ha="center", va="center", 
            fontsize=10, color=color, fontweight="bold", bbox=bbox_props, zorder=3)

# Draw Nodes
for key, data in nodes.items():
    x, y, r, c, label, tc = data
    circle = mpatches.Circle((x, y), r, facecolor=c, edgecolor=tc, linewidth=2, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=11, fontweight="bold", color=tc, zorder=5)

plt.title("Aura 2D Knowledge Graph:\nExtracting Semantic Clinical Evidence", fontsize=18, fontweight="bold", color="#111111", pad=20)
plt.tight_layout()

# Save
output_path = 'C:\\Users\\1kibr\\Documents\\WebDevelopment\\DDI_PROJECTV2-FRONTEND\\molecular-ai\\docs\\poster_visuals_real\\27_knowledge_graph_2d.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved {output_path}")
