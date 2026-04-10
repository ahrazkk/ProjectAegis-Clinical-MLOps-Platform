"""
Generate an improved probability distribution chart with smooth, parabolic-style density curves.
Uses real evaluation outputs from evaluation_predictions.json.
"""

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent.parent / "web" / "models" / "gnn"
OUT_FILE = SCRIPT_DIR / "14_real_probability_distribution_parabolic.png"

COLORS = {
    "safe": "#0891b2",
    "inter": "#dc2626",
    "dark": "#0f172a",
    "muted": "#64748b",
    "border": "#cbd5e1",
    "bg": "#fafbfc",
}


def kde_curve(values: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a KDE and return a smooth density curve on [0, 1]."""
    clipped = np.clip(values, 1e-4, 1 - 1e-4).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(clipped)

    x = np.linspace(0.0, 1.0, 700)
    log_density = kde.score_samples(x.reshape(-1, 1))
    density = np.exp(log_density)

    area = np.trapezoid(density, x)
    if area > 0:
        density = density / area

    return x, density


def fit_beta_moments(values: np.ndarray) -> tuple[float, float]:
    """Fit beta distribution parameters using method of moments."""
    eps = 1e-4
    v = np.clip(values, eps, 1 - eps)
    mu = float(np.mean(v))
    var = float(np.var(v))

    # Keep variance in a stable range for sharp but smooth curves.
    max_var = max(mu * (1 - mu) - 1e-5, 1e-6)
    var = min(max(var, 1e-6), max_var)

    common = (mu * (1 - mu) / var) - 1.0
    alpha = max(mu * common, 1.01)
    beta = max((1 - mu) * common, 1.01)
    return alpha, beta


def beta_pdf_curve(alpha: float, beta: float, points: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Compute beta pdf curve without scipy."""
    x = np.linspace(1e-4, 1 - 1e-4, points)
    log_norm = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_pdf = (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1 - x) - log_norm
    pdf = np.exp(log_pdf)
    area = np.trapezoid(pdf, x)
    if area > 0:
        pdf = pdf / area
    return x, pdf


def main() -> None:
    eval_path = MODEL_DIR / "evaluation_predictions.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {eval_path}")

    with eval_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    y_true = np.asarray(eval_data["y_true"])
    y_scores = np.asarray(eval_data["y_scores"])

    safe_scores = y_scores[y_true == 0]
    inter_scores = y_scores[y_true == 1]

    x_safe, d_safe = kde_curve(safe_scores, bandwidth=0.045)
    x_inter, d_inter = kde_curve(inter_scores, bandwidth=0.045)

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=250)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["bg"])

    # Keep faint histograms as empirical reference.
    bins = np.linspace(0, 1, 55)
    ax.hist(
        safe_scores,
        bins=bins,
        density=True,
        alpha=0.13,
        color=COLORS["safe"],
        edgecolor="none",
    )
    ax.hist(
        inter_scores,
        bins=bins,
        density=True,
        alpha=0.13,
        color=COLORS["inter"],
        edgecolor="none",
    )

    # Smooth, parabolic-style curves.
    ax.plot(x_safe, d_safe, color=COLORS["safe"], linewidth=3.0, label=f"Safe pairs (n={len(safe_scores)})")
    ax.fill_between(x_safe, d_safe, color=COLORS["safe"], alpha=0.22)

    ax.plot(x_inter, d_inter, color=COLORS["inter"], linewidth=3.0, label=f"Interacting pairs (n={len(inter_scores)})")
    ax.fill_between(x_inter, d_inter, color=COLORS["inter"], alpha=0.22)

    # Classification threshold.
    ax.axvline(0.5, color=COLORS["dark"], linestyle="--", linewidth=1.8, alpha=0.75, label="Threshold (0.5)")

    ax.set_title(
        f"Prediction Probability Distribution (Smoothed, n={len(y_true):,})",
        fontsize=15,
        fontweight="bold",
        color=COLORS["dark"],
        pad=10,
    )
    ax.set_xlabel("Predicted Probability", fontsize=11, color=COLORS["dark"])
    ax.set_ylabel("Density", fontsize=11, color=COLORS["dark"])

    ax.tick_params(colors=COLORS["muted"], labelsize=10)
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.6)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])

    ax.set_xlim(-0.01, 1.01)

    legend = ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    legend.get_frame().set_edgecolor(COLORS["border"])

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUT_FILE}")

    # Optional second variant: explicit parabolic-style beta fit.
    a_safe, b_safe = fit_beta_moments(safe_scores)
    a_inter, b_inter = fit_beta_moments(inter_scores)
    xb_safe, db_safe = beta_pdf_curve(a_safe, b_safe)
    xb_inter, db_inter = beta_pdf_curve(a_inter, b_inter)

    beta_out = SCRIPT_DIR / "14_real_probability_distribution_parabolic_beta.png"

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=250)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(COLORS["bg"])

    ax.plot(xb_safe, db_safe, color=COLORS["safe"], linewidth=3.2, label=f"Safe pairs (beta fit)")
    ax.fill_between(xb_safe, db_safe, color=COLORS["safe"], alpha=0.22)

    ax.plot(xb_inter, db_inter, color=COLORS["inter"], linewidth=3.2, label=f"Interacting pairs (beta fit)")
    ax.fill_between(xb_inter, db_inter, color=COLORS["inter"], alpha=0.22)

    ax.axvline(0.5, color=COLORS["dark"], linestyle="--", linewidth=1.8, alpha=0.75, label="Threshold (0.5)")

    ax.set_title(
        f"Prediction Probability Distribution (Parabolic Beta Fit, n={len(y_true):,})",
        fontsize=15,
        fontweight="bold",
        color=COLORS["dark"],
        pad=10,
    )
    ax.set_xlabel("Predicted Probability", fontsize=11, color=COLORS["dark"])
    ax.set_ylabel("Density", fontsize=11, color=COLORS["dark"])

    ax.tick_params(colors=COLORS["muted"], labelsize=10)
    ax.grid(axis="y", alpha=0.18, linestyle="-", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["border"])
    ax.spines["bottom"].set_color(COLORS["border"])
    ax.set_xlim(-0.01, 1.01)

    legend = ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    legend.get_frame().set_edgecolor(COLORS["border"])

    plt.tight_layout()
    plt.savefig(beta_out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {beta_out}")


if __name__ == "__main__":
    main()
