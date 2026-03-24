#!/usr/bin/env python3
"""Generate report-quality plots from P2/P3 grid-search CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR.parent.parent / "cw" / "gl425" / "YOLO_logitKD_experiments"
OUT_DIR = SCRIPT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

P2 = pd.read_csv(CSV_DIR / "P2_grid_search_results.csv")
P3 = pd.read_csv(CSV_DIR / "P3_grid_search_results.csv")

# Separate baseline (exp_00) from KD runs
P2_base = P2[P2["notebook"] == "exp_00"].iloc[0]
P3_base = P3[P3["notebook"] == "exp_00"].iloc[0]
P2_kd = P2[P2["notebook"] != "exp_00"].copy()
P3_kd = P3[P3["notebook"] != "exp_00"].copy()

ALPHAS = sorted(P2_kd["alpha"].unique())
TEMPS = sorted(P2_kd["temperature"].unique())

# ── Shared style ───────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
})


def _pivot(df, metric):
    """Pivot KD results into a (alpha × temperature) matrix."""
    return df.pivot_table(index="alpha", columns="temperature", values=metric, aggfunc="first")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — KD Gain vs CE-only heatmaps (α × T)
# ══════════════════════════════════════════════════════════════════════════════

def plot_kd_gain_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for ax, df_kd, base, title in [
        (axes[0], P2_kd, P2_base, "CIFAR-100 (P2)"),
        (axes[1], P3_kd, P3_base, "CIFAR-10 (P3)"),
    ]:
        mat = _pivot(df_kd, "kd_gain_vs_ce")
        vmax = max(abs(mat.values.max()), abs(mat.values.min()))
        im = ax.imshow(
            mat.values, cmap="RdYlGn", aspect="auto",
            vmin=-vmax, vmax=vmax,
        )
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                colour = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=colour)

        ax.set_xticks(range(len(TEMPS)))
        ax.set_xticklabels([f"{t:g}" for t in TEMPS])
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_yticklabels([f"{a}" for a in ALPHAS])
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Soft-loss weight $\\alpha$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="KD gain vs CE-only (pp)", shrink=0.85)

    fig.suptitle("Top-1 Accuracy Gain of KD over CE-only Fine-tuning", fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "heatmap_kd_gain.png", bbox_inches="tight")
    print(f"Saved: heatmap_kd_gain.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — 4-condition bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_condition_bars():
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    datasets = ["CIFAR-100\n(P2)", "CIFAR-10\n(P3)"]
    conditions = ["Teacher", "Untrained\nStudent", "CE-only\nFine-tuned", "Best KD"]
    colours = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50"]

    best_p2 = P2_kd.loc[P2_kd["kd_top1"].idxmax()]
    best_p3 = P3_kd.loc[P3_kd["kd_top1"].idxmax()]

    values = np.array([
        [P2_base["teacher_top1"], P2_base["pruned_top1"], P2_base["ce_only_top1"], best_p2["kd_top1"]],
        [P3_base["teacher_top1"], P3_base["pruned_top1"], P3_base["ce_only_top1"], best_p3["kd_top1"]],
    ])

    x = np.arange(len(datasets))
    width = 0.18
    for i, (cond, colour) in enumerate(zip(conditions, colours)):
        bars = ax.bar(x + (i - 1.5) * width, values[:, i], width,
                      label=cond, color=colour, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values[:, i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

    # Annotate KD gain
    for idx, (base_ce, best_kd) in enumerate([
        (P2_base["ce_only_top1"], best_p2["kd_top1"]),
        (P3_base["ce_only_top1"], best_p3["kd_top1"]),
    ]):
        gain = best_kd - base_ce
        bar_x = x[idx] + 1.5 * width
        ax.annotate(
            f"$\\Delta$={gain:+.1f}pp",
            xy=(bar_x, best_kd + 0.8), xytext=(bar_x + 0.15, best_kd + 5),
            fontsize=8, fontweight="bold", color="#4CAF50",
            arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1.2),
            ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Model Comparison: Teacher / Untrained / CE-only / Best KD")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add best-config annotation
    ax.text(0.98, 0.02,
            f"Best P2: α={best_p2['alpha']}, T={best_p2['temperature']:g}\n"
            f"Best P3: α={best_p3['alpha']}, T={best_p3['temperature']:g}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    fig.savefig(OUT_DIR / "bar_4condition.png", bbox_inches="tight")
    print(f"Saved: bar_4condition.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — KD Top-1 vs Temperature line plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_top1_vs_temperature():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    markers = ["o", "s", "D", "^", "v"]

    for ax, df_kd, base, title in [
        (axes[0], P2_kd, P2_base, "CIFAR-100 (P2)"),
        (axes[1], P3_kd, P3_base, "CIFAR-10 (P3)"),
    ]:
        for i, alpha in enumerate(ALPHAS):
            subset = df_kd[df_kd["alpha"] == alpha].sort_values("temperature")
            ax.plot(subset["temperature"], subset["kd_top1"],
                    marker=markers[i], markersize=5, linewidth=1.5,
                    label=f"α={alpha}")

        # Reference lines
        ax.axhline(base["teacher_top1"], color="blue", linestyle="--",
                    linewidth=1, alpha=0.6, label="Teacher")
        ax.axhline(base["ce_only_top1"], color="orange", linestyle="--",
                    linewidth=1, alpha=0.6, label="CE-only")

        ax.set_xscale("log", base=2)
        ax.set_xticks(TEMPS)
        ax.set_xticklabels([f"{t:g}" for t in TEMPS])
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("KD Top-1 Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle("KD Top-1 Accuracy vs Temperature (by $\\alpha$)", fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "line_top1_vs_temp.png", bbox_inches="tight")
    print(f"Saved: line_top1_vs_temp.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — CE Loss heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def plot_ce_loss_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for ax, df_kd, base, title in [
        (axes[0], P2_kd, P2_base, "CIFAR-100 (P2)"),
        (axes[1], P3_kd, P3_base, "CIFAR-10 (P3)"),
    ]:
        mat = _pivot(df_kd, "kd_ce_loss")
        ce_only = base["ce_only_ce_loss"]
        # Show improvement: negative means KD has lower loss (better)
        delta_mat = mat - ce_only

        vmax = max(abs(delta_mat.values.max()), abs(delta_mat.values.min()))
        im = ax.imshow(
            delta_mat.values, cmap="RdYlGn_r", aspect="auto",
            vmin=-vmax, vmax=vmax,
        )
        for i in range(delta_mat.shape[0]):
            for j in range(delta_mat.shape[1]):
                val = delta_mat.values[i, j]
                raw = mat.values[i, j]
                colour = "white" if abs(val) > 0.6 * vmax else "black"
                ax.text(j, i, f"{raw:.2f}", ha="center", va="center",
                        fontsize=7.5, color=colour)

        ax.set_xticks(range(len(TEMPS)))
        ax.set_xticklabels([f"{t:g}" for t in TEMPS])
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_yticklabels([f"{a}" for a in ALPHAS])
        ax.set_xlabel("Temperature $T$")
        ax.set_ylabel("Soft-loss weight $\\alpha$")
        ax.set_title(f"{title}  (CE-only baseline: {ce_only:.3f})")
        fig.colorbar(im, ax=ax, label="ΔCE vs CE-only", shrink=0.85)

    fig.suptitle("Validation CE Loss: KD vs CE-only Fine-tuning", fontsize=12, y=1.02)
    fig.savefig(OUT_DIR / "heatmap_ce_loss.png", bbox_inches="tight")
    print(f"Saved: heatmap_ce_loss.png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    plot_kd_gain_heatmaps()
    plot_condition_bars()
    plot_top1_vs_temperature()
    plot_ce_loss_heatmaps()
    print(f"\nAll figures saved to {OUT_DIR}/")
