import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass

# Categories for the radar chart
categories = ['Predictive\nAccuracy', 'Alert Relevance\n(Low False Positives)', 
              'Model\nExplainability', 'Real-Time\nAdaptability', 'Novel Drug\nHandling']
N = len(categories)

# Data values
values_ai = [98, 95, 90, 92, 88]
values_traditional = [65, 30, 40, 20, 10]

# Repeat the first value to close the circular graph
values_ai += values_ai[:1]
values_traditional += values_traditional[:1]

# Calculate angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize spider plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories, size=12, fontweight='bold', color='#333333')

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=10)
plt.ylim(0, 100)

# Plot data
# Traditional System
ax.plot(angles, values_traditional, linewidth=2, linestyle='solid', label='Traditional EHR Alerts', color='#fc8d62')
ax.fill(angles, values_traditional, '#fc8d62', alpha=0.25)

# Our AI System
ax.plot(angles, values_ai, linewidth=2, linestyle='solid', label='GNN + NLP Fusion Platform', color='#3182bd')
ax.fill(angles, values_ai, '#3182bd', alpha=0.4)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.title('System Capability Comparison', size=18, fontweight='bold', y=1.1)

plt.tight_layout()
output_path = 'C:\\Users\\1kibr\\Documents\\WebDevelopment\\DDI_PROJECTV2-FRONTEND\\molecular-ai\\docs\\poster_visuals_real\\26_system_capability_radar.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved {output_path}")
