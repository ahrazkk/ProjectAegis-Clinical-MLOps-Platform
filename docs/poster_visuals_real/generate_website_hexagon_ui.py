import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set absolute path for saving
output_path = r'C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\docs\poster_visuals_real\29_website_hexagon_ui.png'

# Create canvas
fig, ax = plt.subplots(figsize=(10, 13), dpi=300)
fig.patch.set_facecolor('#e3e8ee')  # Very light blue-grey matching the image background
ax.set_facecolor('#e3e8ee')
ax.set_xlim(-8, 8)
ax.set_ylim(-7, 10)
ax.axis('off')

# 1. Background Hexagon Grid
R_bg = 1.0
dx = R_bg * 1.5
dy = R_bg * np.sqrt(3)

# Create a light grid pattern
for i in range(-15, 15):
    for j in range(-15, 15):
        x = i * dx
        y = j * dy + (i % 2) * (dy / 2)
        hex_patch = mpatches.RegularPolygon((x, y), numVertices=6, radius=R_bg,
                                      orientation=0, facecolor='none',
                                      edgecolor='#cfd8e1', linewidth=0.5, alpha=0.7)
        ax.add_patch(hex_patch)

# 2. Draw Connection Edges (Substrate lines)
# Aspirin (Left bottom) connected to CYP2C9 (Left top)
ax.plot([-4, -4], [-3.5, 3], color="#4b8bf4", lw=1.5, zorder=2)
ax.text(-3.8, -0.5, "substrate 92%", color="#659af5", fontsize=9, va='center', fontfamily='monospace', weight='bold')

# Warfarin (Right bottom) connected to CYP3A4 (Right top)
ax.plot([4, 4], [-3.5, 3], color="#4b8bf4", lw=1.5, zorder=2)
ax.text(4.2, -0.5, "substrate 92%", color="#659af5", fontsize=9, va='center', fontfamily='monospace', weight='bold')

# Warfarin (Right bottom) connected to CYP2C9 (Left top)
# Steps: 1) Left from Warfarin 2) Up the middle 3) Left into CYP2C9
ax.plot([2.4, 0, 0, -2.4], [-3.5, -3.5, 3, 3], color="#4b8bf4", lw=1.5, zorder=2)

# 3. Draw Nodes Logic
def draw_node(x, y, main_text, sub_text, color, is_drug=False):
    # Outer thin container hexagon
    outer = mpatches.RegularPolygon((x, y), numVertices=6, radius=2.2,
                                    orientation=0, facecolor='none',
                                    edgecolor='#aab4c2', linewidth=1, zorder=3)
    ax.add_patch(outer)
    
    # Inner colored hexagon
    inner_fill = mpatches.RegularPolygon((x, y), numVertices=6, radius=1.6,
                                    orientation=0, facecolor=color, alpha=0.08, zorder=3)
    ax.add_patch(inner_fill)
    
    inner_edge = mpatches.RegularPolygon((x, y), numVertices=6, radius=1.6,
                                    orientation=0, facecolor='none', edgecolor=color, linewidth=2, zorder=4)
    ax.add_patch(inner_edge)
    
    # Center text formatting
    if is_drug:
        ax.text(x, y + 0.1, main_text, color=color, fontsize=18, fontweight='bold', ha='center', va='center', fontfamily='monospace', zorder=5)
        ax.text(x, y - 0.6, sub_text, color='#b2b9c2', fontsize=11, fontweight='bold', ha='center', va='center', fontfamily='sans-serif', zorder=5)
    else:
        ax.text(x, y, main_text, color=color, fontsize=16, fontweight='bold', ha='center', va='center', fontfamily='monospace', zorder=5)
        
        # Small orange badge at the bottom of Targets (like CYP)
        badge_y = y - 1.62
        bbox_props = dict(boxstyle="round,pad=0.2", fc="#e3e8ee", ec="#fd7e14", lw=0.8)
        ax.text(x, badge_y, " SUD ", color="#fd7e14", fontsize=7, fontweight='bold', ha='center', va='center', fontfamily='monospace', bbox=bbox_props, zorder=5)

# Place the 4 exact nodes from the image
draw_node(-4, 3, "CYP2C9", "", "#4b8bf4", is_drug=False)
draw_node(4, 3, "CYP3A4", "", "#4b8bf4", is_drug=False)
draw_node(-4, -3.5, "Aspirin", "Pain/Cardiovascular", "#0dcaf0", is_drug=True)
draw_node(4, -3.5, "Warfarin", "Hematology", "#fd7e14", is_drug=True)

# 4. Top Dark Alert Widget (Exactly replicating the UI)
box_y = 6.4
box_h = 2.4
alert_box = mpatches.FancyBboxPatch((-7.5, box_y), 15, box_h, boxstyle="round,pad=0.3,rounding_size=0.4",
                                fc="#2d2d2d", ec="#383838", lw=1, zorder=10)
ax.add_patch(alert_box)

# Header Row
ax.text(-7, box_y + 1.6, "⚠  SEVERE INTERACTION", color="#fc5555", fontsize=14, fontweight='bold', fontfamily='monospace', zorder=11)
# Subtitle Row
ax.text(-7, box_y + 1.0, "Increased risk of bleeding due to additive anticoagulant effects", color="#9fa5aa", fontsize=11, fontweight='bold', fontfamily='monospace', zorder=11)

# Stats Line Separator
ax.plot([-7, 7.0], [box_y + 0.5, box_y + 0.5], color="#444444", lw=1.2, zorder=11)

# Stats Text (Cyan and Maroon like the screenshot)
stat_y = box_y + 0.15
ax.text(-7, stat_y, "0", color="#0dcaf0", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)
ax.text(-6.6, stat_y, "CYP conflicts      ", color="#a35f79", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)

ax.text(-3.3, stat_y, "0", color="#0dcaf0", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)
ax.text(-2.9, stat_y, "shared targets     ", color="#a35f79", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)

ax.text(0.7, stat_y, "0", color="#0dcaf0", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)
ax.text(1.1, stat_y, "shared effects", color="#a35f79", fontsize=10, fontweight='bold', fontfamily='monospace', zorder=11)

# Save result
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"Hexagon UI exactly matched and saved to {output_path}")