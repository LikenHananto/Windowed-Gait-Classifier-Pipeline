"""
Two-panel DS1 LOSO comparison plot for any fold metric.

  Left:  bar chart of mean metric per model (with std error bars), sorted descending.
  Right: heatmap of per-subject metric for the top-10 models.

Usage:
    python plot_ds1_loso.py                    # walking_precision (default)
    python plot_ds1_loso.py walking_recall     # any column in fold_metrics
    python plot_ds1_loso.py walking_f1
    python plot_ds1_loso.py accuracy
    python plot_ds1_loso.py macro_f1

Output: Results_Plots/DS1_LOSO/loso_<metric>.png
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


FOLD_CSV = "Results_Plots/DS1_LOSO/fold_metrics.csv"


def main():
    metric = sys.argv[1] if len(sys.argv) > 1 else "walking_precision"
    pretty = metric.replace("_", " ").title()
    out_png = f"Results_Plots/DS1_LOSO/loso_{metric}.png"

    df = pd.read_csv(FOLD_CSV)
    if metric not in df.columns:
        raise SystemExit(f"Column {metric!r} not in {FOLD_CSV} (have: {list(df.columns)})")

    summary = (df.groupby("model")[metric]
                 .agg(mean="mean", std="std")
                 .reset_index()
                 .sort_values("mean", ascending=False))

    mat = (df.pivot(index="model", columns="held_out", values=metric)
             .reindex(summary["model"].tolist()))
    mat.columns = [f"Subj {c}" for c in mat.columns]

    fig, axes = plt.subplots(1, 2, figsize=(26, 10),
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ── Left: horizontal bar with std error bars (top model at top) ──
    left = summary.iloc[::-1]
    ax = axes[0]
    bars = ax.barh(left["model"], left["mean"],
                   xerr=left["std"], color="#1D9E75", alpha=0.85,
                   error_kw={"elinewidth": 1, "capsize": 4})
    for bar, val in zip(bars, left["mean"]):
        if not np.isnan(val):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel(f"{pretty} (mean ± std)")
    ax.set_title(f"{pretty} — LOSO mean", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Right: heatmap of top-10 models ──
    top10 = summary.head(10)["model"].tolist()
    mat_top = mat.loc[top10]
    ax = axes[1]
    sns.heatmap(mat_top, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.70, vmax=1.00, ax=ax, cbar=True,
                linewidths=0.4, linecolor="white",
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title(f"{pretty} per subject — top 10", fontweight="bold")
    ax.set_xlabel("Test subject")
    ax.set_ylabel("Model")

    plt.suptitle(f"Leave-One-Subject-Out Evaluation — primary metric: {pretty}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"✅ saved: {out_png}")


if __name__ == "__main__":
    main()
