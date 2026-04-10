import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

output_path = r'C:\Users\1kibr\Documents\WebDevelopment\DDI_PROJECTV2-FRONTEND\molecular-ai\docs\poster_visuals_real\31_elegant_drug_network.png'

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')
ax.set_xlim(-6, 6)
ax.set_ylim(-4.5, 4.5)
ax.axis('off')

# Palette - Elegant, highly modern muted corporate colors
border_gray = '#cbd5e1'
line_gray = '#94a3b8'
text_dark = '#0f172a'
text_muted = '#64748b'
enzyme_color = '#0284c7'  # Clean, professional blue
safe_drug_color = '#64748b' # Sophisticated slate
conflict_color = '#e11d48' # Sharp, non-neon rose red

# Calculate hex edge offset for perfectly flush arrows
# Radius is 1.05. Using orientation pi/2 (flat tops/bottoms)
R = 1.05
edge_offset = R * np.cos(np.pi / 6)

def draw_elegant_hex(x, y, title, subtitle, category_label, theme_color):
    # Ultra-soft elegant shadow beneath the line
    shadow = mpatches.RegularPolygon((x, y-0.08), numVertices=6, radius=R,
                                     orientation=np.pi/2, facecolor='#0f172a', alpha=0.03, edgecolor='none', zorder=1)
    ax.add_patch(shadow)

    # Main Hex - White fill with a crisp, beautifully colored border
    hex_patch = mpatches.RegularPolygon((x, y), numVertices=6, radius=R,
                                        orientation=np.pi/2, facecolor='#ffffff', edgecolor=theme_color, linewidth=1.8, zorder=3)
    ax.add_patch(hex_patch)

    # Texts inside hex
    ax.text(x, y + 0.15, title, fontsize=15, fontweight='bold', color=text_dark, ha='center', va='center', zorder=5)
    ax.text(x, y - 0.25, subtitle, fontsize=9, color=text_muted, ha='center', va='center', zorder=5)

    # Floating Category Label (clean text, no heavy badges)
    badge_y = y + 1.25 if 'ENZYME' in category_label else y - 1.25
    ax.text(x, badge_y, category_label, fontsize=8, fontweight='bold', color=theme_color, ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="none"), zorder=4)

# Grid Layout Coordinates
x_left = -2.5
x_right = 2.5
y_top = 1.8
y_bot = -1.8

# --- 1. DRAW EDGES FIRST (Z-order 1-2) ---
bbox_white = dict(boxstyle="square,pad=0.5", fc="white", ec="none", alpha=0.9)

# Aspirin -> CYP2C9 (Normal Path)
ax.annotate("", xy=(x_left, y_top - edge_offset - 0.05), xytext=(x_left, y_bot + edge_offset + 0.05),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=line_gray, shrinkA=0, shrinkB=0), zorder=1)
ax.text(x_left, 0, "Metabolized Substrate", color=text_muted, fontsize=9, fontweight='bold', ha='center', va='center', bbox=bbox_white, zorder=2)

# Warfarin -> CYP3A4 (Normal Path)
ax.annotate("", xy=(x_right, y_top - edge_offset - 0.05), xytext=(x_right, y_bot + edge_offset + 0.05),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=line_gray, shrinkA=0, shrinkB=0), zorder=1)
ax.text(x_right, 0, "Secondary Substrate", color=text_muted, fontsize=9, fontweight='bold', ha='center', va='center', bbox=bbox_white, zorder=2)

# Warfarin -> CYP2C9 (The Conflict - Sweeping Elegant Curve)
# Curve leftward from Warfarin to CYP2C9
ax.annotate("", xy=(x_left + 0.6, y_top - edge_offset - 0.1), xytext=(x_right - 0.6, y_bot + edge_offset + 0.1),
            arrowprops=dict(arrowstyle="-|>", lw=2.2, color=conflict_color, 
                            connectionstyle="arc3,rad=-0.15", shrinkA=0, shrinkB=0), zorder=1)

# Text floating over the curve with an elegant, very soft pink rounded box
conflict_box = dict(boxstyle="round,pad=0.6,rounding_size=0.3", fc="#fff1f2", ec=conflict_color, lw=1.2)
ax.text(0, -0.4, "Primary Substrate\nCompetitive Inhibition Risk", color=conflict_color, fontsize=10, 
        fontweight='bold', ha='center', va='center', bbox=conflict_box, zorder=2)


# --- 2. DRAW NODES OVER EDGES (Z-order 3-5) ---
draw_elegant_hex(x_left, y_top, "CYP2C9", "Liver CYP450", "ENZYME TARGET", enzyme_color)
draw_elegant_hex(x_right, y_top, "CYP3A4", "Liver CYP450", "ENZYME TARGET", enzyme_color)

draw_elegant_hex(x_left, y_bot, "Aspirin", "NSAID", "PHARMACEUTICAL", safe_drug_color)
draw_elegant_hex(x_right, y_bot, "Warfarin", "Anticoagulant", "PHARMACEUTICAL", conflict_color)


# --- 3. ELEGANT EDITORIAL TITLES ---
ax.text(-5.5, 4.0, "DDI Pathway Graph", fontsize=22, fontweight='bold', color=text_dark, ha='left')
ax.text(-5.5, 3.6, "Visualizing molecular network conflicts and enzyme inhibition.", fontsize=12, color=text_muted, ha='left')

plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"Elegant minimalist diagram saved to {output_path}")
