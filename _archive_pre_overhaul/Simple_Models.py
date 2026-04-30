import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Linear models ─────────────────────────────────────────────────────────────
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier,
    SGDClassifier, PassiveAggressiveClassifier
)
# ── Discriminant analysis ─────────────────────────────────────────────────────
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
# ── SVM ───────────────────────────────────────────────────────────────────────
from sklearn.svm import SVC, LinearSVC, NuSVC
# ── Neighbors ─────────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
# ── Trees ─────────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# ── Ensembles ─────────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    BaggingClassifier, HistGradientBoostingClassifier
)
# ── Naive Bayes ───────────────────────────────────────────────────────────────
from sklearn.naive_bayes import GaussianNB
# ── Neural network ────────────────────────────────────────────────────────────
from sklearn.neural_network import MLPClassifier
# ── Boosting libraries ────────────────────────────────────────────────────────
from xgboost import XGBClassifier

from Trainer import train_and_evaluate_model, tune_walking_threshold

save_path = "Results_Plots"

# ── Load and prepare data ─────────────────────────────────────────────────────
df = pd.read_csv("Processed_Data/lab_W256_S128.csv")
Y  = df['Task']
X  = df.drop(columns=['Task'])

le = LabelEncoder()
Y  = le.fit_transform(Y)

class_names = list(le.classes_)
walking_idx = class_names.index('walking')

print("📚 Label mapping:")
for i, label in enumerate(class_names):
    print(f"  {i}: {label}")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, random_state=42
)
X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()

# Scaled versions for distance/margin/linear models
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Classifier registry ───────────────────────────────────────────────────────
# (display_name, model, needs_scaling)
classifiers = [
    # Linear
    ("Logistic Regression",    LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),          True),
    ("Ridge Classifier",       RidgeClassifier(class_weight='balanced'),                                             True),
    ("SGD Classifier",         SGDClassifier(max_iter=2000, class_weight='balanced', random_state=42),               True),
    ("Passive Aggressive",     PassiveAggressiveClassifier(class_weight='balanced', random_state=42),                True),
    ("Linear Discriminant",    LinearDiscriminantAnalysis(),                                                         True),
    ("Quadratic Discriminant", QuadraticDiscriminantAnalysis(),                                                      True),

    # SVM
    ("SVM (RBF)",              SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),        True),
    ("SVM (Linear)",           LinearSVC(class_weight='balanced', max_iter=5000, random_state=42),                   True),
    ("Nu-SVC",                 NuSVC(probability=True, random_state=42),                                             True),

    # Neighbors
    ("KNN (k=5)",              KNeighborsClassifier(n_neighbors=5),                                                  True),
    ("KNN (k=11)",             KNeighborsClassifier(n_neighbors=11),                                                 True),

    # Trees
    ("Decision Tree",          DecisionTreeClassifier(class_weight='balanced', random_state=42),                     False),
    ("Extra Tree",             ExtraTreeClassifier(class_weight='balanced', random_state=42),                        False),

    # Ensembles
    ("Random Forest",          RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                      class_weight='balanced', random_state=42),                     False),
    ("Extra Trees",            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                    class_weight='balanced', random_state=42),                       False),
    ("Gradient Boosting",      GradientBoostingClassifier(n_estimators=300, random_state=42),                        False),
    ("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=42),                                      False),
    ("AdaBoost",               AdaBoostClassifier(n_estimators=200, random_state=42),                                False),
    ("Bagging",                BaggingClassifier(n_estimators=100, random_state=42),                                 False),

    # Naive Bayes
    ("Gaussian NB",            GaussianNB(),                                                                         False),

    # Neural network
    ("MLP",                    MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                             max_iter=500, random_state=42),                                         True),

    # Boosting
    ("XGBoost",                XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             random_state=42, verbosity=0),                                          False),
]

# ── Training loop ─────────────────────────────────────────────────────────────
results  = []
trained  = {}   # store fitted models for threshold tuning later

for name, model, scaled in classifiers:
    X_tr = X_train_sc if scaled else X_train
    X_te = X_test_sc  if scaled else X_test

    try:
        metrics = train_and_evaluate_model(
            model, X_tr, y_train, X_te, y_test,
            evaluate=True,
            save_path=save_path,
            remap=False,
            class_names=class_names,
            walking_idx=walking_idx
        )
        results.append({
            'Model':             name,
            'Accuracy':          metrics['accuracy'],
            'Walking Precision': metrics['walking_precision'],
            'Walking Recall':    metrics['walking_recall'],
            'Walking F1':        metrics['walking_f1'],
            'Macro F1':          metrics['macro_f1'],
        })
        trained[name] = (model, scaled)

    except Exception as e:
        print(f"  ⚠️  {name} skipped — {e}")

# ── Results table ─────────────────────────────────────────────────────────────
results_df = (pd.DataFrame(results)
              .sort_values('Walking Precision', ascending=False)
              .reset_index(drop=True))

print("\n" + "="*70)
print("📊 Final Results — sorted by Walking Precision")
print("="*70)
print(results_df.to_string(index=False, float_format='{:.4f}'.format))

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(32, 9))
metrics_to_plot = ['Walking Precision', 'Walking Recall', 'Walking F1', 'Accuracy']
colors = ['#1D9E75', '#D85A30', '#378ADD', '#888780']

for ax, metric, color in zip(axes, metrics_to_plot, colors):
    sorted_df = results_df.sort_values(metric, ascending=True)
    bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.8)

    for bar, val in zip(bars, sorted_df[metric]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric)
    ax.set_title(metric, fontweight='bold')
    ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle('Classifier Comparison — primary metric: Walking Precision',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{save_path}/classifier_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"\nVisualization saved as '{save_path}/classifier_comparison.png'")

# ── Threshold tuning on top 3 models by walking precision ────────────────────
print("\n" + "="*70)
print("🎯 Threshold tuning — top 3 models by Walking Precision")
print("="*70)

top3 = results_df.head(3)['Model'].tolist()
for name in top3:
    model, scaled = trained[name]
    X_te = X_test_sc if scaled else X_test
    print(f"\n── {name}")
    tune_walking_threshold(
        model, X_te, y_test,
        walking_idx=walking_idx,
        save_path=save_path
    )