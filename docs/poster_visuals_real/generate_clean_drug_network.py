import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

output_path = r'C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\docs\poster_visuals_real\30_clean_drug_network.png'

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(-6, 6)
ax.set_ylim(-5, 6)
ax.axis('off')

def draw_hex_node(x, y, main_text, sub_text, color_theme, is_enzyme=False):
    # color_theme = (fill_color, edge_color, text_color)
    fc, ec, tc = color_theme
    
    # Subtle Drop Shadow
    shadow = mpatches.RegularPolygon((x+0.1, y-0.1), numVertices=6, radius=1.4,
                                     orientation=0, facecolor='#e9ecef', edgecolor='none', alpha=0.8, zorder=1)
    ax.add_patch(shadow)
    
    # Main Hexagon Node
    hex_patch = mpatches.RegularPolygon((x, y), numVertices=6, radius=1.4,
                                        orientation=0, facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=2)
    ax.add_patch(hex_patch)
    
    # Text Placement
    ax.text(x, y + 0.15, main_text, color=tc, fontsize=16, fontweight='bold', ha='center', va='center', fontfamily='sans-serif', zorder=3)
    ax.text(x, y - 0.4, sub_text, color=ec, fontsize=11, fontweight='bold', ha='center', va='center', fontfamily='sans-serif', zorder=3)
    
    # Little badge
    badge_y = y + 1.4 if is_enzyme else y - 1.4
    badge_label = "ENZYME TARGET" if is_enzyme else "PHARMACEUTICAL"
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec=ec, lw=1.5)
    ax.text(x, badge_y, badge_label, color=ec, fontsize=8, fontweight='bold', ha='center', va='center', bbox=bbox_props, zorder=4)

# Define clean color themes
enzyme_theme = ('#f0f8ff', '#0288d1', '#01579b') # Light blue fill, bold blue text
drug_theme = ('#fff5f5', '#d32f2f', '#b71c1c')    # Light red fill, bold red text

# --- Draw Edges First (so they appear underneath nodes) ---
arrow_props = dict(arrowstyle="-|>", lw=2.5, color="#adb5bd", shrinkA=45, shrinkB=45)
conflict_arrow_props = dict(arrowstyle="-|>", lw=3.0, color="#d63384", shrinkA=45, shrinkB=45) # Pink/Red for conflict

# Aspirin -> CYP2C9
ax.annotate("", xy=(-3, 2.5), xytext=(-3, -2.5), arrowprops=arrow_props, zorder=1)
ax.text(-3.2, 0, "Metabolized Substrate", color="#6c757d", fontsize=10, ha='right', va='center', fontweight='bold', rotation=90)

# Warfarin -> CYP3A4
ax.annotate("", xy=(3, 2.5), xytext=(3, -2.5), arrowprops=arrow_props, zorder=1)
ax.text(3.2, 0, "Secondary Substrate", color="#6c757d", fontsize=10, ha='left', va='center', fontweight='bold', rotation=-90)

# Warfarin -> CYP2C9 (The Conflict)
ax.annotate("", xy=(-3, 2.5), xytext=(3, -2.5), arrowprops=conflict_arrow_props, zorder=1)
# Add a background box for the conflict text so it's readable over the line
bbox_text = dict(boxstyle="round,pad=0.3", fc="white", ec="#d63384", lw=1, alpha=0.9)
ax.text(0, 0, "Primary Substrate\n(Competitive Inhibition Risk)", color="#d63384", fontsize=10, 
        ha='center', va='center', fontweight='bold', rotation=-39, bbox=bbox_text, zorder=5)

# --- Draw Nodes ---
# Enzymes (Top row)
draw_hex_node(-3, 2.5, "CYP2C9", "Liver CYP450", enzyme_theme, is_enzyme=True)
draw_hex_node(3, 2.5, "CYP3A4", "Liver CYP450", enzyme_theme, is_enzyme=True)

# Drugs (Bottom row)
draw_hex_node(-3, -2.5, "Aspirin", "NSAID", drug_theme, is_enzyme=False)
draw_hex_node(3, -2.5, "Warfarin", "Anticoagulant", drug_theme, is_enzyme=False)

# --- Titles ---
plt.suptitle("Pharmacokinetic Network Graph", fontsize=22, fontweight='bold', color="#212529", y=0.95)
plt.title("Visualizing shared metabolic pathways and competitive inhibition", fontsize=14, color="#6c757d", pad=15)

plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"Clean white diagram saved to {output_path}")
