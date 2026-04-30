"""
Generic analysis runner.

Drives the full classifier-comparison pipeline against any preprocessed
dataframe under any split scheme:

    run_analysis(df, label_col, split='loso',        ...)   # leave-one-subject-out
    run_analysis(df, label_col, split='groupkfold',  k=3)   # subject-grouped k-fold
    run_analysis(df, label_col, split='aggregated',  test_size=0.3)  # random split

For fold-based schemes, per-fold metrics are averaged; per-fold confusion
matrices are summed; the consolidated metrics CSV and bar chart are written
to `save_path`. For 'aggregated' the run is single-shot.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

# ── Linear models ─────────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    SGDClassifier, PassiveAggressiveClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Classifier registry — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
def build_classifiers(random_state: int = 42):
    clfs = [
        ("Logistic Regression",    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),         True),
        ("Ridge Classifier",       RidgeClassifier(class_weight="balanced"),                                                       True),
        ("SGD Classifier",         SGDClassifier(max_iter=2000, class_weight="balanced", random_state=random_state),               True),
        ("Passive Aggressive",     PassiveAggressiveClassifier(class_weight="balanced", random_state=random_state),                True),
        ("Linear Discriminant",    LinearDiscriminantAnalysis(),                                                                   True),
        ("Quadratic Discriminant", QuadraticDiscriminantAnalysis(),                                                                True),

        ("SVM (RBF)",              SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state),        True),
        ("SVM (Linear)",           LinearSVC(class_weight="balanced", max_iter=5000, random_state=random_state),                   True),
        ("Nu-SVC",                 NuSVC(probability=True, random_state=random_state),                                             True),

        ("KNN (k=5)",              KNeighborsClassifier(n_neighbors=5),                                                            True),
        ("KNN (k=11)",             KNeighborsClassifier(n_neighbors=11),                                                           True),

        ("Decision Tree",          DecisionTreeClassifier(class_weight="balanced", random_state=random_state),                     False),
        ("Extra Tree",             ExtraTreeClassifier(class_weight="balanced", random_state=random_state),                        False),

        ("Random Forest",          RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                          class_weight="balanced", random_state=random_state),                    False),
        ("Extra Trees",            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                        class_weight="balanced", random_state=random_state),                      False),
        ("Gradient Boosting",      GradientBoostingClassifier(n_estimators=300, random_state=random_state),                        False),
        ("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=random_state),                                      False),
        ("AdaBoost",               AdaBoostClassifier(n_estimators=200, random_state=random_state),                                False),
        ("Bagging",                BaggingClassifier(n_estimators=100, random_state=random_state),                                 False),

        ("Gaussian NB",            GaussianNB(),                                                                                   False),

        ("MLP",                    MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=random_state),     True),
    ]
    if _HAS_XGB:
        clfs.append(("XGBoost", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              random_state=random_state, verbosity=0), False))
    return clfs


def build_classifiers_fast(random_state: int = 42):
    """Smaller, faster model set — useful for smoke testing."""
    return [
        ("Logistic Regression",    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),         True),
        ("Linear Discriminant",    LinearDiscriminantAnalysis(),                                                                  True),
        ("KNN (k=5)",              KNeighborsClassifier(n_neighbors=5),                                                           True),
        ("Decision Tree",          DecisionTreeClassifier(class_weight="balanced", random_state=random_state),                    False),
        ("Random Forest",          RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                          class_weight="balanced", random_state=random_state),                   False),
        ("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=random_state),                                     False),
        ("Gaussian NB",            GaussianNB(),                                                                                  False),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold trainer
# ─────────────────────────────────────────────────────────────────────────────
def _train_one(model, X_tr, y_tr, X_te, y_te, walking_idx, class_names):
    name = model.__class__.__name__
    if name == "XGBClassifier":
        # XGB needs labels in [0..K-1]; handled because LabelEncoder already does this.
        pass
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
    }
    if walking_idx is not None and walking_idx in y_te:
        metrics["walking_precision"] = precision_score(y_te, y_pred, labels=[walking_idx], average=None, zero_division=0)[0]
        metrics["walking_recall"]    = recall_score(y_te, y_pred,    labels=[walking_idx], average=None, zero_division=0)[0]
        metrics["walking_f1"]        = f1_score(y_te, y_pred,        labels=[walking_idx], average=None, zero_division=0)[0]
    else:
        metrics["walking_precision"] = np.nan
        metrics["walking_recall"]    = np.nan
        metrics["walking_f1"]        = np.nan

    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(class_names))))
    return metrics, cm, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_walking_idx(class_names, label_col):
    """
    Identify the 'walking-positive' class index for the per-class metrics.
    DS1 uses 'walking' string label; DS2 uses is_walking==1.
    """
    candidates = []
    if label_col == "is_walking":
        if 1 in class_names:
            candidates.append(1)
    else:
        for name in ("walking", "Walking"):
            if name in class_names:
                candidates.append(name)

    if not candidates:
        return None
    # class_names came from LabelEncoder, so position is the encoded int label.
    return list(class_names).index(candidates[0])


def _split_iter(X, y, groups, scheme, k=3, test_size=0.3, random_state=42):
    """Yield (fold_id, train_idx, test_idx) tuples."""
    if scheme == "loso":
        splitter = LeaveOneGroupOut()
        for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups)):
            yield fold_id, tr, te, str(np.unique(groups[te])[0])
    elif scheme == "groupkfold":
        splitter = GroupKFold(n_splits=k)
        for fold_id, (tr, te) in enumerate(splitter.split(X, y, groups)):
            held = "_".join(map(str, sorted(np.unique(groups[te]))))
            yield fold_id, tr, te, held
    elif scheme == "aggregated":
        idx = np.arange(len(y))
        try:
            tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=random_state)
        except ValueError:
            tr, te = train_test_split(idx, test_size=test_size, random_state=random_state)
        yield 0, tr, te, "random"
    else:
        raise ValueError(f"Unknown split scheme: {scheme}")


def run_analysis(
    df: pd.DataFrame,
    label_col: str,
    save_path: str,
    split: str = "loso",
    k: int = 3,
    test_size: float = 0.3,
    random_state: int = 42,
    classifiers=None,
    title_suffix: str = "",
):
    """Run the full classifier comparison pipeline."""
    os.makedirs(save_path, exist_ok=True)

    if classifiers is None:
        classifiers = build_classifiers(random_state=random_state)

    # ── Encode labels ────────────────────────────────────────────────────────
    y_raw = df[label_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    walking_idx = _resolve_walking_idx(class_names, label_col)

    print(f"\n📚 Label classes: {class_names}")
    if walking_idx is not None:
        print(f"   walking-positive class index: {walking_idx} ({class_names[walking_idx]})")

    # Drop any rows with NaN features (rare; happens when a window has
    # zero variance in some signal column and std/dominant-freq become NaN).
    n_nan = df.isna().any(axis=1).sum()
    if n_nan:
        print(f"   dropping {n_nan} rows containing NaN feature values")
        df = df.dropna().reset_index(drop=True)
        y = LabelEncoder().fit(y_raw).transform(df[label_col].values)

    feature_cols = [c for c in df.columns if c not in (label_col, "subject_id")]
    X = df[feature_cols].to_numpy()
    groups = df["subject_id"].values if "subject_id" in df.columns else None

    if split in ("loso", "groupkfold") and groups is None:
        raise ValueError("subject_id column required for loso/groupkfold splits")

    # ── Iterate folds ────────────────────────────────────────────────────────
    fold_records = []   # per fold per model
    aggregate_cm = {n: np.zeros((len(class_names), len(class_names)), dtype=int)
                    for (n, _, _) in classifiers}

    for fold_id, tr_idx, te_idx, held in _split_iter(
        X, y, groups, split, k=k, test_size=test_size, random_state=random_state
    ):
        n_tr, n_te = len(tr_idx), len(te_idx)
        print(f"\n──── Fold {fold_id} (held out: {held}) | train={n_tr} test={n_te} ────")
        if n_te == 0 or n_tr == 0:
            print("    (empty fold — skipping)")
            continue

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X[tr_idx])
        X_te_sc = scaler.transform(X[te_idx])
        X_tr_raw = X[tr_idx]
        X_te_raw = X[te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        for name, model_proto, scaled in classifiers:
            X_tr = X_tr_sc if scaled else X_tr_raw
            X_te = X_te_sc if scaled else X_te_raw
            try:
                model = clone(model_proto)   # fresh estimator per fold
                metrics, cm, _ = _train_one(model, X_tr, y_tr, X_te, y_te, walking_idx, class_names)
                aggregate_cm[name] += cm
                rec = {"fold": fold_id, "held_out": held, "model": name, **metrics}
                fold_records.append(rec)
                wp = metrics.get("walking_precision", float("nan"))
                wr = metrics.get("walking_recall", float("nan"))
                wf = metrics.get("walking_f1", float("nan"))
                print(f"    {name:24s} acc={metrics['accuracy']:.3f}  "
                      f"walk_p={wp:.3f}  walk_r={wr:.3f}  f1={wf:.3f}")
            except Exception as e:
                print(f"    {name:24s} ⚠️  skipped — {e}")
                fold_records.append({"fold": fold_id, "held_out": held, "model": name,
                                     "accuracy": np.nan, "macro_f1": np.nan,
                                     "walking_precision": np.nan, "walking_recall": np.nan,
                                     "walking_f1": np.nan})

    # ── Aggregate ────────────────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv(os.path.join(save_path, "fold_metrics.csv"), index=False)

    summary = (
        fold_df.groupby("model")
        .agg(
            accuracy=("accuracy", "mean"),
            walking_precision=("walking_precision", "mean"),
            walking_recall=("walking_recall", "mean"),
            walking_f1=("walking_f1", "mean"),
            macro_f1=("macro_f1", "mean"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )

    sort_key = "walking_precision" if walking_idx is not None else "accuracy"
    summary = summary.sort_values(sort_key, ascending=False).reset_index(drop=True)
    summary.to_csv(os.path.join(save_path, "summary_metrics.csv"), index=False)

    print("\n" + "=" * 72)
    print(f"📊 Summary  ({split} {title_suffix})  — sorted by {sort_key}")
    print("=" * 72)
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    # ── Bar chart ────────────────────────────────────────────────────────────
    metrics_to_plot = ["walking_precision", "walking_recall", "walking_f1", "accuracy"]
    titles = ["Walking Precision", "Walking Recall", "Walking F1", "Accuracy"]
    colors = ["#1D9E75", "#D85A30", "#378ADD", "#888780"]

    fig, axes = plt.subplots(1, 4, figsize=(28, max(7, 0.35 * len(summary))))
    for ax, metric, title, color in zip(axes, metrics_to_plot, titles, colors):
        s = summary.sort_values(metric, ascending=True)
        bars = ax.barh(s["model"], s[metric], color=color, alpha=0.8)
        for bar, val in zip(bars, s[metric]):
            if not np.isnan(val):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(metric)
        ax.set_title(title, fontweight="bold")
        ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"Classifier Comparison — {split} {title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "classifier_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Aggregated confusion matrices for top 5 models ──────────────────────
    top5 = summary.head(5)["model"].tolist()
    for name in top5:
        cm = aggregate_cm[name]
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{name} — aggregated CM ({split})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plt.savefig(os.path.join(save_path, f"cm_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n✅ Outputs written to {save_path}")
    return summary, fold_df
