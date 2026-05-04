"""
Regenerate every plot in `Results_Plots/` in a horizontally compact form.

Output goes to `Results_Plots/compact/`, mirroring the directory layout but
using stacked / 2×2 panel arrangements instead of the original wide 1×N rows.
Reads the CSVs that the run scripts already produced — does not retrain.

Usage:
    python plot_compact.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT      = Path("Results_Plots")
COMPACT   = ROOT / "compact"
METRICS   = ["walking_precision", "walking_recall", "walking_f1", "accuracy"]
TITLES    = ["Walking Precision", "Walking Recall", "Walking F1", "Accuracy"]
COLORS_4  = ["#1D9E75", "#D85A30", "#378ADD", "#888780"]


# ─────────────────────────────────────────────────────────────────────────────
# Generic 2×2 classifier comparison (used for all four run_*.py outputs)
# ─────────────────────────────────────────────────────────────────────────────
def compact_classifier_comparison(summary_csv: Path, out_png: Path, suptitle: str):
    df = pd.read_csv(summary_csv)
    n  = len(df)
    fig, axes = plt.subplots(2, 2, figsize=(14, max(8, 0.32 * n + 1)))
    for ax, metric, title, color in zip(axes.ravel(), METRICS, TITLES, COLORS_4):
        s = df.sort_values(metric, ascending=True)
        bars = ax.barh(s["model"], s[metric], color=color, alpha=0.85)
        for bar, val in zip(bars, s[metric]):
            if not np.isnan(val):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=7)
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# DS1 LOSO per-metric: stacked bar+heatmap (replaces plot_ds1_loso.py output)
# ─────────────────────────────────────────────────────────────────────────────
def compact_loso_metric(fold_csv: Path, out_png: Path, metric: str):
    pretty = metric.replace("_", " ").title()
    df = pd.read_csv(fold_csv)
    if metric not in df.columns:
        return
    summary = (df.groupby("model")[metric]
                 .agg(mean="mean", std="std")
                 .reset_index()
                 .sort_values("mean", ascending=False))
    mat = (df.pivot(index="model", columns="held_out", values=metric)
             .reindex(summary["model"].tolist()))
    mat.columns = [f"Subj {c}" for c in mat.columns]

    n_models = len(summary)
    fig, axes = plt.subplots(2, 1, figsize=(12, max(10, 0.36 * n_models + 4)),
                             gridspec_kw={"height_ratios": [1.6, 1.0]})

    # ── Top: horizontal bar with std error bars ──
    left = summary.iloc[::-1]
    ax = axes[0]
    bars = ax.barh(left["model"], left["mean"],
                   xerr=left["std"], color="#1D9E75", alpha=0.85,
                   error_kw={"elinewidth": 1, "capsize": 3})
    for bar, val in zip(bars, left["mean"]):
        if not np.isnan(val):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel(f"{pretty} (mean ± std)", fontsize=10)
    ax.set_title(f"{pretty} — LOSO mean", fontweight="bold", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # ── Bottom: heatmap of top-10 models ──
    top10 = summary.head(10)["model"].tolist()
    mat_top = mat.loc[top10]
    ax = axes[1]
    sns.heatmap(mat_top, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.70, vmax=1.00, ax=ax, cbar=True,
                linewidths=0.4, linecolor="white",
                annot_kws={"size": 9, "weight": "bold"})
    ax.set_title(f"{pretty} per subject — top 10", fontweight="bold", fontsize=11)
    ax.set_xlabel("Test subject", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.tick_params(labelsize=8)

    plt.suptitle(f"DS1 LOSO — {pretty}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation strategy comparison — 2×1 stacked
# ─────────────────────────────────────────────────────────────────────────────
def compact_strategy_comparison(comp_csv: Path, out_png: Path):
    pivot = pd.read_csv(comp_csv).sort_values(
        "walking_precision_aggregated", ascending=False
    ).reset_index(drop=True)
    sorted_models = pivot.sort_values("walking_precision_aggregated",
                                      ascending=True)["model"].tolist()
    y_pos = np.arange(len(sorted_models))
    height = 0.27

    strat_colors = {
        "aggregated":         "#1D9E75",
        "aggregated_matched": "#378ADD",
        "kfold":              "#D85A30",
    }
    strat_labels = {
        "aggregated":         "aggregated (80% of DS2)",
        "aggregated_matched": "aggregated_matched (~67%)",
        "kfold":              "kfold avg of 3 folds (~67%/fold)",
    }
    metrics = [("walking_precision", "Walking Precision"),
               ("accuracy", "Accuracy")]

    fig, axes = plt.subplots(2, 1, figsize=(12, max(12, 0.5 * len(sorted_models) + 2)))
    for ax, (metric, title) in zip(axes, metrics):
        for offset, strat in zip([-height, 0, height],
                                 ["aggregated", "aggregated_matched", "kfold"]):
            vals = [pivot.loc[pivot["model"] == m,
                              f"{metric}_{strat}"].values[0] for m in sorted_models]
            ax.barh(y_pos + offset, vals, height, color=strat_colors[strat],
                    alpha=0.85, label=strat_labels[strat])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_models, fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8)
    plt.suptitle("Validation_Data: aggregated vs aggregated_matched vs kfold",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrices — copy from existing folder, no resize needed (already small)
# ─────────────────────────────────────────────────────────────────────────────
def copy_cms(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("cm_*.png")):
        # copy bytes through (delete may be blocked but write is fine)
        with open(f, "rb") as fin, open(dst_dir / f.name, "wb") as fout:
            fout.write(fin.read())
    print(f"  copied {len(list(src_dir.glob('cm_*.png')))} confusion matrices "
          f"from {src_dir.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\nWriting compact plots to {COMPACT}/\n")

    # 1. Four training analyses — compact 2×2 classifier_comparison
    for tag, suptitle in [
        ("DS1_LOSO",        "DS1 — LOSO (3 folds)"),
        ("DS1_aggregated",  "DS1 — aggregated (random 70/30)"),
        ("DS2_kfold",       "DS2 — GroupKFold k=3"),
        ("DS2_aggregated",  "DS2 — aggregated (random 70/30)"),
    ]:
        src_csv = ROOT / tag / "summary_metrics.csv"
        if src_csv.exists():
            print(f"[{tag}]")
            compact_classifier_comparison(
                src_csv, COMPACT / tag / "classifier_comparison.png", suptitle,
            )
            copy_cms(ROOT / tag, COMPACT / tag)

    # 2. DS1 LOSO per-metric stacked plots (precision, recall, f1, accuracy)
    fold_csv = ROOT / "DS1_LOSO" / "fold_metrics.csv"
    if fold_csv.exists():
        print("[DS1_LOSO per-metric]")
        for m in ("walking_precision", "walking_recall", "walking_f1",
                  "accuracy", "macro_f1"):
            compact_loso_metric(
                fold_csv, COMPACT / "DS1_LOSO" / f"loso_{m}.png", m,
            )

    # 3. Validation strategy comparison (stacked)
    comp_csv = ROOT / "Validation_DS2" / "comparison_pooled.csv"
    if comp_csv.exists():
        print("[Validation_DS2]")
        compact_strategy_comparison(
            comp_csv, COMPACT / "Validation_DS2" / "strategy_comparison.png",
        )
        copy_cms(ROOT / "Validation_DS2", COMPACT / "Validation_DS2")

    print(f"\n✅ Done. All compact plots are under {COMPACT}/")


if __name__ == "__main__":
    main()
