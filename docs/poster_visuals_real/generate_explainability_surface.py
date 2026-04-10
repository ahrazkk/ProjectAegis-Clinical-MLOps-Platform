import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# Generate data for the surface
X = np.linspace(0, 1, 100) # GNN Structural Confidence
Y = np.linspace(0, 1, 100) # Literature/NLP Evidence Score
X, Y = np.meshgrid(X, Y)

# A non-linear function simulating an ensemble neural network's decision surface
# Enhances the probability strongly when both signals are high
alpha = 10
beta_x = 0.6
beta_y = 0.6
bias = -0.7
Z = 1.0 / (1.0 + np.exp(-alpha * (beta_x*X + beta_y*Y + bias)))

# Plot the upper surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.85)

# Add a contour project at the floor Level
cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.5)

# Add some simulated "clinical drug queries" (scatter points) resting on/near the surface
np.random.seed(42)
num_points = 50
px = np.random.uniform(0, 1, num_points)
py = np.random.uniform(0, 1, num_points)

# Simulate slight variation from standard expected model output (e.g. unique edge cases)
pz = 1.0 / (1.0 + np.exp(-alpha * (beta_x*px + beta_y*py + bias))) 
noise = np.random.normal(0, 0.03, num_points)
pz = np.clip(pz + noise, 0, 1)

# Color points by their output probability
ax.scatter(px, py, pz, c=pz, cmap=cm.coolwarm, s=35, edgecolor='k', linewidth=0.5, depthshade=True, zorder=5)

# Customize the axes
ax.set_xlabel('GNN Structural Confidence\n(Molecular Latent Space)', fontsize=11, labelpad=15, fontweight='bold', color='#333333')
ax.set_ylabel('NLP/Literature Evidence\n(Knowledge Graph Weight)', fontsize=11, labelpad=15, fontweight='bold', color='#333333')
ax.set_zlabel('Final Assessed Probability Level', fontsize=11, labelpad=15, fontweight='bold', color='#333333')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Modify tick styles
ax.tick_params(axis='x', colors='#555555')
ax.tick_params(axis='y', colors='#555555')
ax.tick_params(axis='z', colors='#555555')

# Enhance background panes
ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))

# Make grids fainter
ax.xaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 1)})
ax.yaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 1)})
ax.zaxis._axinfo["grid"].update({"color": (0.8, 0.8, 0.8, 1)})

# Viewing angle
ax.view_init(elev=20, azim=-50)

plt.title('Fusion Decision Surface:\nMapping Modalities to Confidence', fontsize=16, fontweight='bold', color='#222222', pad=20)
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('DDI Likelihood Index', fontsize=10, fontweight='bold', color='#333333')

plt.tight_layout()
plt.savefig('23_explainability_decision_surface.png', bbox_inches='tight')
print("Saved 23_explainability_decision_surface.png")
