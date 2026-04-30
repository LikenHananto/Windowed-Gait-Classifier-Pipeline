"""
Validate DS2-trained classifiers against held-out subjects in Validation_Data
AND compare three training strategies side by side.

Strategy A — 'aggregated':
    Train on a random 70% of all DS2 windows (same split as run_ds2_aggregated.py:
    stratified, random_state=42). Predict on Validation_Data.

Strategy B — 'kfold':
    GroupKFold(k=3) on DS2: 3 fold models per classifier, each trained on
    4 of the 6 DS2 subjects. Predict on Validation_Data with each fold model;
    metrics are averaged across the 3 fold models.

Strategy C — 'aggregated_matched':
    Same as A, but the training-set size is subsampled to match the k-fold
    AVERAGE training size. This isolates the *strategy* effect from the
    *sample-size* effect — the only difference vs k-fold is whether training
    samples come from all 6 subjects randomly (matched) or 4 specific subjects
    (kfold).

Compare A vs C to read the effect of "more data" alone.
Compare C vs B to read the effect of "subject-aware split" alone at equal n.

All three are evaluated against the same Validation_Data so the metrics are
directly comparable. Results are reported pooled over both validation
subjects and broken out per subject.

Usage:
    python validate_ds2.py            # all 22 classifiers (slow)
    python validate_ds2.py --fast     # 7-classifier smoke set
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)

import analysis

# Lazy import — only needed if processed CSVs are missing.
def _lazy_import_preprocessors():
    from preprocess_ds2 import preprocess_ds2, preprocess_validation
    return preprocess_ds2, preprocess_validation


warnings.filterwarnings("ignore")

WINDOW, STEP   = 256, 128
TRAIN_CSV      = f"Processed_Data/DS2_W{WINDOW}_S{STEP}.csv"
VAL_CSV        = f"Processed_Data/Validation_W{WINDOW}_S{STEP}.csv"
OUT_DIR        = "Results_Plots/Validation_DS2"
RANDOM_STATE   = 42
KFOLD_K        = 3
AGGREGATED_TEST_SIZE = 0.30


def _ensure_processed():
    if not os.path.exists(TRAIN_CSV):
        print("Training CSV missing — running preprocess_ds2 ...")
        preprocess_ds2, _ = _lazy_import_preprocessors()
        preprocess_ds2(window_size=WINDOW, step_size=STEP)
    if not os.path.exists(VAL_CSV):
        print("Validation CSV missing — running preprocess_validation ...")
        _, preprocess_validation = _lazy_import_preprocessors()
        preprocess_validation(window_size=WINDOW, step_size=STEP)


def _metrics(y_true, y_pred, walking_idx):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if walking_idx is not None and walking_idx in np.unique(y_true):
        out["walking_precision"] = precision_score(y_true, y_pred, labels=[walking_idx],
                                                    average=None, zero_division=0)[0]
        out["walking_recall"] = recall_score(y_true, y_pred, labels=[walking_idx],
                                              average=None, zero_division=0)[0]
        out["walking_f1"] = f1_score(y_true, y_pred, labels=[walking_idx],
                                      average=None, zero_division=0)[0]
    else:
        out["walking_precision"] = np.nan
        out["walking_recall"] = np.nan
        out["walking_f1"] = np.nan
    return out


def _scope_metrics(y_true, y_pred, val_subjects, walking_idx, prefix=""):
    """Return a list of dict rows: pooled + one row per validation subject."""
    rows = []
    rows.append({"scope": "pooled", "n": len(y_true), **_metrics(y_true, y_pred, walking_idx)})
    for s in sorted(np.unique(val_subjects)):
        mask = val_subjects == s
        rows.append({"scope": f"subject_{s}", "n": int(mask.sum()),
                     **_metrics(y_true[mask], y_pred[mask], walking_idx)})
    return rows


def _evaluate_aggregated(model_proto, scaled, X_train, y_train, X_val, y_val,
                         val_subjects, walking_idx, class_names,
                         test_size=None, random_state=RANDOM_STATE):
    """
    Aggregated random split: stratified random hold-out of DS2; train on the
    remainder; predict on val. `test_size` defaults to AGGREGATED_TEST_SIZE.
    """
    if test_size is None:
        test_size = AGGREGATED_TEST_SIZE
    idx = np.arange(len(y_train))
    tr_idx, _ = train_test_split(
        idx, test_size=test_size, stratify=y_train, random_state=random_state
    )
    Xtr_raw = X_train[tr_idx]
    ytr = y_train[tr_idx]

    if scaled:
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr = scaler.transform(Xtr_raw)
        Xva = scaler.transform(X_val)
    else:
        Xtr, Xva = Xtr_raw, X_val

    m = clone(model_proto)
    m.fit(Xtr, ytr)
    y_pred = m.predict(Xva)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(class_names))))
    return y_pred, cm, len(tr_idx)


def _evaluate_kfold(model_proto, scaled, X_train, y_train, groups,
                    X_val, y_val, val_subjects, walking_idx, class_names):
    """
    Strategy B: GroupKFold k=3 on DS2 subjects; predict on val with each fold model.
    Returns per-fold predictions plus averaged confusion matrix.
    """
    splitter = GroupKFold(n_splits=KFOLD_K)
    fold_preds = []
    cm_total = np.zeros((len(class_names), len(class_names)), dtype=int)

    for fold_id, (tr_idx, _) in enumerate(splitter.split(X_train, y_train, groups)):
        Xtr_raw = X_train[tr_idx]
        ytr = y_train[tr_idx]
        held = "_".join(map(str, sorted(np.unique(groups[~np.isin(np.arange(len(groups)), tr_idx)]))))

        if scaled:
            scaler = StandardScaler().fit(Xtr_raw)
            Xtr = scaler.transform(Xtr_raw)
            Xva = scaler.transform(X_val)
        else:
            Xtr, Xva = Xtr_raw, X_val

        m = clone(model_proto)
        m.fit(Xtr, ytr)
        y_pred = m.predict(Xva)
        fold_preds.append({"fold": fold_id, "held_train_subjects": held, "y_pred": y_pred})
        cm_total += confusion_matrix(y_val, y_pred, labels=list(range(len(class_names))))

    return fold_preds, cm_total


def main():
    fast = "--fast" in sys.argv
    _ensure_processed()

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # Drop NaN rows defensively
    n_train_nan = train_df.isna().any(axis=1).sum()
    n_val_nan   = val_df.isna().any(axis=1).sum()
    if n_train_nan:
        print(f"  dropping {n_train_nan} train rows containing NaN")
        train_df = train_df.dropna().reset_index(drop=True)
    if n_val_nan:
        print(f"  dropping {n_val_nan} val rows containing NaN")
        val_df = val_df.dropna().reset_index(drop=True)

    print(f"\nTrain (DS2):     shape={train_df.shape}  subjects={sorted(train_df['subject_id'].unique())}")
    print(f"Validation:      shape={val_df.shape}    subjects={sorted(val_df['subject_id'].unique())}")

    # ── Align feature columns ───────────────────────────────────────────────
    feature_cols = [c for c in train_df.columns if c not in ("subject_id", "is_walking")]
    missing = set(feature_cols) - set(val_df.columns)
    if missing:
        raise RuntimeError(f"Validation set is missing feature columns: {sorted(missing)}")
    X_train = train_df[feature_cols].to_numpy()
    X_val   = val_df[feature_cols].to_numpy()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["is_walking"].values)
    y_val   = le.transform(val_df["is_walking"].values)
    class_names = list(le.classes_)
    walking_idx = class_names.index(1) if 1 in class_names else None
    val_subjects = val_df["subject_id"].astype(str).values
    groups = train_df["subject_id"].astype(str).values

    # ── Compute kfold's average training size so we can size-match aggregated ─
    splitter = GroupKFold(n_splits=KFOLD_K)
    fold_train_sizes = []
    for tr_idx, _ in splitter.split(X_train, y_train, groups):
        fold_train_sizes.append(len(tr_idx))
    avg_kf_train = int(np.mean(fold_train_sizes))
    matched_test_size = 1.0 - (avg_kf_train / len(y_train))

    print(f"\nStrategy training sizes:")
    print(f"  aggregated         (test_size={AGGREGATED_TEST_SIZE:.2f}):  "
          f"{int(len(y_train) * (1-AGGREGATED_TEST_SIZE))} windows")
    print(f"  kfold (k={KFOLD_K})        avg per fold: {avg_kf_train} "
          f"({['{:,}'.format(s) for s in fold_train_sizes]})")
    print(f"  aggregated_matched (test_size={matched_test_size:.3f}):  "
          f"{avg_kf_train} windows  ← same n as kfold avg")

    # ── Run all classifiers under all three strategies ──────────────────────
    classifiers = analysis.build_classifiers_fast() if fast else analysis.build_classifiers()
    print(f"\nRunning {len(classifiers)} classifiers × 3 strategies\n")

    rows = []                                # long format: model, strategy, scope, metrics
    cms_agg, cms_kf, cms_match = {}, {}, {}  # confusion matrices for plotting

    for name, model_proto, scaled in classifiers:
        try:
            # ── Strategy A: aggregated (70%) ──
            y_pred_agg, cm_agg, _ = _evaluate_aggregated(
                model_proto, scaled, X_train, y_train, X_val, y_val,
                val_subjects, walking_idx, class_names,
                test_size=AGGREGATED_TEST_SIZE,
            )
            cms_agg[name] = cm_agg
            for r in _scope_metrics(y_val, y_pred_agg, val_subjects, walking_idx):
                rows.append({"model": name, "strategy": "aggregated", **r})

            # ── Strategy C: aggregated, sample-size-matched to kfold ──
            y_pred_match, cm_match, _ = _evaluate_aggregated(
                model_proto, scaled, X_train, y_train, X_val, y_val,
                val_subjects, walking_idx, class_names,
                test_size=matched_test_size,
            )
            cms_match[name] = cm_match
            for r in _scope_metrics(y_val, y_pred_match, val_subjects, walking_idx):
                rows.append({"model": name, "strategy": "aggregated_matched", **r})

            # ── Strategy B: kfold (3 fold models, metrics averaged) ──
            fold_preds, cm_kf = _evaluate_kfold(
                model_proto, scaled, X_train, y_train, groups,
                X_val, y_val, val_subjects, walking_idx, class_names,
            )
            cms_kf[name] = cm_kf

            per_scope = {}
            for fp in fold_preds:
                for r in _scope_metrics(y_val, fp["y_pred"], val_subjects, walking_idx):
                    per_scope.setdefault(r["scope"], []).append(r)
            for scope, items in per_scope.items():
                avg = {"scope": scope, "n": items[0]["n"]}
                for k in ("accuracy", "macro_f1", "walking_precision",
                          "walking_recall", "walking_f1"):
                    avg[k] = float(np.nanmean([x[k] for x in items]))
                rows.append({"model": name, "strategy": "kfold", **avg})

            # console summary
            def _pick(strat):
                return next(r for r in rows
                            if r["model"] == name and r["strategy"] == strat
                            and r["scope"] == "pooled")
            a, c, k = _pick("aggregated"), _pick("aggregated_matched"), _pick("kfold")
            print(f"  {name:24s} "
                  f"agg: walk_p={a['walking_precision']:.3f}  "
                  f"matched: walk_p={c['walking_precision']:.3f}  "
                  f"kfold: walk_p={k['walking_precision']:.3f}")

        except Exception as e:
            print(f"  {name:24s} ⚠️  skipped — {e}")

    long_df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    long_df.to_csv(os.path.join(OUT_DIR, "validation_long.csv"), index=False)

    # ── Side-by-side comparison table (pooled only) ─────────────────────────
    pooled = long_df[long_df["scope"] == "pooled"].copy()
    pivot = pooled.pivot(index="model", columns="strategy",
                         values=["walking_precision", "walking_recall",
                                 "walking_f1", "accuracy"])
    pivot.columns = [f"{m}_{s}" for m, s in pivot.columns]
    pivot = pivot.reset_index()

    # Two key deltas:
    #   sample_size_effect = aggregated_70 − aggregated_matched
    #   strategy_effect    = aggregated_matched − kfold       (apples-to-apples at equal n)
    pivot["sample_size_effect"] = (
        pivot["walking_precision_aggregated"] - pivot["walking_precision_aggregated_matched"]
    )
    pivot["strategy_effect"] = (
        pivot["walking_precision_aggregated_matched"] - pivot["walking_precision_kfold"]
    )
    pivot = pivot.sort_values("walking_precision_aggregated", ascending=False).reset_index(drop=True)
    pivot.to_csv(os.path.join(OUT_DIR, "comparison_pooled.csv"), index=False)

    print("\n" + "=" * 110)
    print("📊 Strategy comparison — pooled validation (sorted by aggregated walking precision)")
    print("=" * 110)
    cols = ["model",
            "walking_precision_aggregated",
            "walking_precision_aggregated_matched",
            "walking_precision_kfold",
            "sample_size_effect", "strategy_effect"]
    print(pivot[cols].to_string(index=False, float_format="{:.4f}".format))
    print()
    print("  sample_size_effect = aggregated(70%) − aggregated_matched   (extra-data benefit)")
    print("  strategy_effect    = aggregated_matched − kfold             (subject-shuffling benefit at equal n)")

    # ── Per-subject side-by-side (walking precision only) ───────────────────
    sub_long = long_df[long_df["scope"].str.startswith("subject_")].copy()
    sub_pivot = sub_long.pivot_table(
        index="model",
        columns=["strategy", "scope"],
        values="walking_precision",
    )
    sub_pivot.columns = [f"{strat}_{scope}" for strat, scope in sub_pivot.columns]
    sub_pivot = sub_pivot.reset_index()
    sub_pivot.to_csv(os.path.join(OUT_DIR, "comparison_per_subject.csv"), index=False)

    print("\n" + "=" * 100)
    print("📊 Walking precision by validation subject (aggregated vs kfold)")
    print("=" * 100)
    print(sub_pivot.to_string(index=False, float_format="{:.4f}".format))

    # ── Plots: 3-bar group per model for walking precision and accuracy ────
    sorted_models = pivot.sort_values("walking_precision_aggregated", ascending=True)["model"].tolist()
    y_pos = np.arange(len(sorted_models))
    height = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(22, max(8, 0.45 * len(sorted_models))))
    strat_colors = {
        "aggregated":         "#1D9E75",
        "aggregated_matched": "#378ADD",
        "kfold":              "#D85A30",
    }
    strat_labels = {
        "aggregated":         f"aggregated (70% of DS2, n≈{int(len(y_train)*(1-AGGREGATED_TEST_SIZE))})",
        "aggregated_matched": f"aggregated_matched (n≈{avg_kf_train})",
        "kfold":              f"kfold avg of {KFOLD_K} folds (n≈{avg_kf_train}/fold)",
    }
    for ax, metric, title in zip(
        axes,
        ["walking_precision", "accuracy"],
        ["Walking Precision", "Accuracy"],
    ):
        for offset, strat in zip([-height, 0, height], ["aggregated", "aggregated_matched", "kfold"]):
            vals = [pivot.loc[pivot["model"] == m, f"{metric}_{strat}"].values[0] for m in sorted_models]
            ax.barh(y_pos + offset, vals, height, color=strat_colors[strat],
                    alpha=0.85, label=strat_labels[strat])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_models)
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel(metric)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8)
    plt.suptitle("Validation_Data: aggregated vs aggregated_matched vs kfold",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "strategy_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Confusion matrices for top-3 models under each strategy ─────────────
    top3 = pivot.head(3)["model"].tolist()
    for name in top3:
        for tag, cm_dict in (("aggregated", cms_agg),
                             ("aggregated_matched", cms_match),
                             ("kfold", cms_kf)):
            cm = cm_dict.get(name)
            if cm is None:
                continue
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f"{name} — validation CM ({tag})")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout()
            safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            plt.savefig(os.path.join(OUT_DIR, f"cm_{safe}_{tag}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"\n✅ Outputs written to {OUT_DIR}/")
    print("   - validation_long.csv         (model × strategy × scope)")
    print("   - comparison_pooled.csv       (side-by-side pooled metrics, with effect deltas)")
    print("   - comparison_per_subject.csv  (per-subject walking precision)")
    print("   - strategy_comparison.png     (3-strategy bar chart)")
    print("   - cm_<top3>_{aggregated,aggregated_matched,kfold}.png")


if __name__ == "__main__":
    main()
