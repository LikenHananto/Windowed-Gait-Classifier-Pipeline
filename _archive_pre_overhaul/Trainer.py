import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_score, recall_score
)


def train_and_evaluate_model(
    model, X_train, y_train, X_test, y_test,
    evaluate=True, save_path=None, remap=False,
    class_names=None, walking_idx=None
):
    model_name = model.__class__.__name__

    print("==========================")
    print(f"⚙️  Training {model_name}...")
    print("==========================")

    # XGBoost label remap (only when remap=True)
    if model_name == "XGBClassifier" and remap:
        y_train = y_train - 1
        y_test  = y_test  - 1

    model.fit(X_train, y_train)

    if not evaluate:
        return model

    y_pred = model.predict(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy   = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average='macro')

    walking_precision = (
        precision_score(y_test, y_pred, labels=[walking_idx], average=None)[0]
        if walking_idx is not None else None
    )
    walking_recall = (
        recall_score(y_test, y_pred, labels=[walking_idx], average=None)[0]
        if walking_idx is not None else None
    )
    walking_f1 = (
        f1_score(y_test, y_pred, labels=[walking_idx], average=None)[0]
        if walking_idx is not None else None
    )

    # ── Classification report ─────────────────────────────────────────────────
    print(classification_report(
        y_test, y_pred,
        target_names=class_names if class_names is not None else None
    ))
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    if walking_idx is not None:
        print(f"  Walking Precision: {walking_precision:.4f}  ← primary metric")
        print(f"  Walking Recall:    {walking_recall:.4f}")
        print(f"  Walking F1:        {walking_f1:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"confusion_matrix_{model_name}.png"),
            dpi=150, bbox_inches='tight'
        )
    plt.clf()
    plt.close()

    return {
        'accuracy':          accuracy,
        'macro_f1':          macro_f1,
        'walking_precision': walking_precision,
        'walking_recall':    walking_recall,
        'walking_f1':        walking_f1,
    }


def tune_walking_threshold(model, X_test, y_test, walking_idx, save_path=None):
    """
    Sweeps the walking prediction threshold and plots the precision/recall
    tradeoff curve. Returns the threshold that maximises precision while
    keeping recall above 0.5.

    Only works for models that support predict_proba.
    """
    if not hasattr(model, 'predict_proba'):
        print(f"  ⚠️  {model.__class__.__name__} does not support predict_proba — skipping threshold tuning.")
        return None

    y_proba = model.predict_proba(X_test)
    thresholds = np.arange(0.30, 0.96, 0.01)
    threshold_results = []

    for thresh in thresholds:
        y_pred_custom = np.argmax(y_proba, axis=1).copy()

        # Where model predicts walking but confidence is below threshold,
        # fall back to the next most probable class
        low_conf = (y_pred_custom == walking_idx) & (y_proba[:, walking_idx] < thresh)
        if low_conf.any():
            y_pred_custom[low_conf] = np.argsort(
                y_proba[low_conf], axis=1
            )[:, -2]

        prec = precision_score(y_test, y_pred_custom, labels=[walking_idx], average=None, zero_division=0)[0]
        rec  = recall_score(y_test, y_pred_custom, labels=[walking_idx], average=None, zero_division=0)[0]
        f1   = f1_score(y_test, y_pred_custom, labels=[walking_idx], average=None, zero_division=0)[0]
        threshold_results.append({
            'threshold': thresh,
            'precision': prec,
            'recall':    rec,
            'f1':        f1,
        })

    thresh_df = pd.DataFrame(threshold_results)

    # Best threshold: highest precision where recall >= 0.5
    viable = thresh_df[thresh_df['recall'] >= 0.50]
    best_row = viable.loc[viable['precision'].idxmax()] if not viable.empty else thresh_df.iloc[0]

    print(f"\n  Best threshold: {best_row['threshold']:.2f}")
    print(f"    Walking Precision: {best_row['precision']:.4f}")
    print(f"    Walking Recall:    {best_row['recall']:.4f}")
    print(f"    Walking F1:        {best_row['f1']:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(thresh_df['threshold'], thresh_df['precision'],
             label='Walking Precision', color='#1D9E75', linewidth=2)
    plt.plot(thresh_df['threshold'], thresh_df['recall'],
             label='Walking Recall', color='#D85A30', linewidth=2)
    plt.plot(thresh_df['threshold'], thresh_df['f1'],
             label='Walking F1', color='#378ADD', linewidth=1.5, linestyle='--')
    plt.axvline(x=best_row['threshold'], color='gray',
                linestyle=':', linewidth=1.5, label=f"Best threshold ({best_row['threshold']:.2f})")
    plt.xlabel('Walking prediction threshold')
    plt.ylabel('Score')
    plt.title(f"Precision / Recall tradeoff — {model.__class__.__name__}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"threshold_tuning_{model.__class__.__name__}.png"),
            dpi=150, bbox_inches='tight'
        )
    plt.clf()
    plt.close()

    return best_row['threshold']