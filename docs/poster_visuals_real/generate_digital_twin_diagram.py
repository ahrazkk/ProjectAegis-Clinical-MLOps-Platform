import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Arial']
except:
    pass

fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Box 1: Patient Baseline
ax.add_patch(mpatches.FancyBboxPatch((0.5, 1.5), 3.5, 3, boxstyle="round,pad=0.2", fc="#f8f9fa", ec="#dee2e6", lw=2))
ax.text(2.25, 4.0, "Clinical Digital Twin\n(Patient Profile)", ha="center", va="center", fontsize=14, fontweight="bold", color="#212529")
ax.text(0.8, 3.2, "• Age: 68\n• Med Hx: HTN, Osteoarthritis\n• Bio: eGFR 45 mL/min", va="top", fontsize=11, color="#495057")

# Current Rx within Box 1
ax.add_patch(mpatches.FancyBboxPatch((0.8, 1.7), 2.9, 0.8, boxstyle="round,pad=0.1", fc="#e0f3db", ec="#a8ddb5", lw=1.5))
ax.text(2.25, 2.1, "Current Regimen:\nLisinopril + Amlodipine", ha="center", va="center", fontsize=10, fontweight="bold", color="#212529")

# Arrow 1
ax.annotate("", xy=(4.6, 3), xytext=(4.0, 3), arrowprops=dict(arrowstyle="->", lw=4, color="#adb5bd"))
ax.text(4.3, 3.3, "Simulate\nRx", ha="center", fontsize=10, fontweight="bold", color="#6c757d")

# Box 2: Proposal
ax.add_patch(mpatches.FancyBboxPatch((4.8, 2.3), 3, 1.4, boxstyle="round,pad=0.2", fc="#fff3cd", ec="#ffc107", lw=2))
ax.text(6.3, 3.0, "Proposed Addition:\nIbuprofen (NSAID)", ha="center", va="center", fontsize=13, fontweight="bold", color="#856404")

# Arrow 2
ax.annotate("", xy=(8.4, 3), xytext=(7.8, 3), arrowprops=dict(arrowstyle="->", lw=4, color="#adb5bd"))
ax.text(8.1, 3.3, "Analyze", ha="center", fontsize=10, fontweight="bold", color="#6c757d")

# Box 3: Organ System Body Map Prediction
ax.add_patch(mpatches.FancyBboxPatch((8.6, 1.2), 4.5, 3.6, boxstyle="round,pad=0.2", fc="#f8f9fa", ec="#dee2e6", lw=2))
ax.text(10.85, 4.4, "Poly Twin Toxicity Projection\n(Organ Body Map)", ha="center", va="center", fontsize=14, fontweight="bold", color="#212529")

# Organs
organs = [
    (10.85, 3.5, "Liver / Hepatic", "Safe", "#d1e7dd", "#0f5132", "Metabolism Cleared"),
    (10.85, 2.7, "Heart / Cardio", "Elevated Risk", "#fff3cd", "#856404", "BP Interference"),
    (10.85, 1.8, "Kidneys / Renal", "CRITICAL FAILURE", "#f8d7da", "#842029", "Triple Whammy Effect!"),
]

for x, y, name, status, bg, tc, desc in organs:
    ax.add_patch(mpatches.FancyBboxPatch((8.9, y-0.35), 3.9, 0.7, boxstyle="round,pad=0.05", fc=bg, ec=tc, lw=1.5))
    ax.text(9.1, y, f"{name}:", ha="left", va="center", fontsize=10, fontweight="bold", color=tc)
    ax.text(12.6, y, status, ha="right", va="center", fontsize=10, fontweight="bold", color=tc)
    ax.text(10.85, y-0.2, desc, ha="center", va="center", fontsize=8, fontstyle="italic", color=tc)

plt.title("The Clinical Digital Twin: Real-Time Prescription Simulation", fontsize=18, fontweight="bold", pad=20, color="#111111")
plt.tight_layout()

# Save
output_path = 'C:\\Users\\1kibr\\Documents\\WebDevelopment\\DDI_PROJECTV2-FRONTEND\\molecular-ai\\docs\\poster_visuals_real\\28_digital_twin_bodymap.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved {output_path}")
