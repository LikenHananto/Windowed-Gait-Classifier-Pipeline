# Gait Classifier Project — Context for Discussion

## 1. Goal

Binary "walking vs not-walking" classification from wearable IMU (accelerometer + gyroscope) data, comparing how different cross-validation strategies estimate true generalization to unseen subjects.

## 2. Data

Two training datasets and one held-out validation set, all sampled at ~100 Hz.

### DS1 — dual-IMU (lab data)
- **Sensors**: 2 IMUs per subject — `_lower` (waist/leg) and `_upper` (torso/wrist)
- **Subjects**: 3 — IDs 01, 05, 07
- **Labels**: `walking`, `bending`, plus `Unknown` (dropped)
- **File schema**: `Var1` (timestamp), `Var2..Var4` (acc xyz), `Var5..Var7` (gyro xyz), `Task`, `time`
- After windowing (W=256, S=128): **1,992 windows total**, **120 predictors per window**

### DS2 — single-IMU (free-living data)
- **Sensors**: 1 IMU per subject
- **Subjects**: 6 — MCM-001, 002, 003, 010, 011, 012
- **Labels**: 8 activities (`bending_squatting`, `lying`, `sitting`, `stairs`, `standing`, `walking_indoor`, `walking_outdoor`, `walking_phone`) + `unlabeled` (dropped). **Target column: `is_walking` (binary, derived)**.
- **File schema**: `timestamp, acc_x_g, acc_y_g, acc_z_g, gyro_x_deg_s, gyro_y_deg_s, gyro_z_deg_s, activity, is_walking`
- Note: DS2 timestamps are minute-resolution only; rows are treated as already-ordered at fs=100 Hz
- After windowing: **7,084 windows total** (1 NaN row dropped → 7,076 used), **60 predictors per window**

### Validation_Data — held-out unseen subjects
- Same schema as DS2; subjects MCM-004, MCM-007 (both completely unseen during DS2 training)
- After windowing: **2,418 windows** (1316 from MCM-004, 1102 from MCM-007), 60 predictors

## 3. Pipeline

12 stages, identical primitives applied to DS1 and DS2 (only the channel set differs):

1. **Load + label cleanup** — drop "Unknown"/`unlabeled` rows
2. **Sensor alignment** (DS1 only) — resample upper+lower onto a 10 ms grid
3. **Denoise** — 4th-order Butterworth low-pass at 20 Hz (`filtfilt`, zero-phase)
4. **Component separation** — 0.3 Hz low-pass to split each channel into gravity/body (acc) and static/dynamic (gyro)
5. **Jerk** — `diff(signal) * fs` on body-acc and dynamic-gyro
6. **Sliding window** — W=256 samples, S=128 step (~2.56 s, 50% overlap); mixed-label windows dropped
7. **Feature extraction** — per window:
   - Time domain: mean + std for every signal column (raw, gravity/static, body/dynamic, jerks)
   - Frequency domain: dominant FFT frequency for body-acc, dynamic-gyro, and their jerks
   - DS1: 48 means + 48 stds + 24 freqs = **120 predictors** (suffixed `_lower` / `_upper`)
   - DS2: 24 means + 24 stds + 12 freqs = **60 predictors**
8. **Standardization** — `StandardScaler` fit on train fold only, applied to test/validation; tree-based models see raw features
9. **Classifier registry** — 22 sklearn classifiers (LR, Ridge, SGD, PA, LDA, QDA, SVM-RBF/Linear, Nu-SVC, KNN-5/11, DT, Extra Tree, RF, Extra Trees, GB, HGB, AdaBoost, Bagging, GaussianNB, MLP, XGBoost). Cloned per fold via `sklearn.base.clone`.
10. **Split schemes** (pluggable):
    - `loso` — Leave-One-Subject-Out (DS1, 3 folds)
    - `groupkfold` — k=3 by subject (DS2, 2 subjects held out per fold)
    - `aggregated` — random stratified train/test split (no subject separation)
11. **Evaluation** — per fold: accuracy, macro-F1, walking precision/recall/F1, confusion matrix. Across folds: mean per metric, summed CMs.
12. **External validation** — `validate_ds2.py` retrains on full DS2 and predicts on `Validation_Data/` under three strategies (see §5)

## 4. Training results (top 5 by walking precision)

### DS1 LOSO (3 folds: hold out 01, 05, 07 in turn)
```
              model  accuracy  walking_precision  walking_recall  walking_f1  macro_f1
   Ridge Classifier    0.7743             0.9639          0.6926      0.7072    0.7446
Linear Discriminant    0.7775             0.9506          0.6997      0.7171    0.7506
         Extra Tree    0.7532             0.8999          0.7078      0.7174    0.7234
         KNN (k=11)    0.7595             0.8992          0.6670      0.6539    0.7198
          KNN (k=5)    0.7682             0.8936          0.6662      0.6582    0.7323
```
- Subject 7 collapses every model toward majority-class prediction when held out (~36% accuracy on that fold).
- The ~75% mean accuracy reflects this drag — it's two strong folds (subjects 1 and 5) and one near-failure (subject 7).

### DS1 Aggregated (random 70/30 stratified)
```
           model  accuracy  walking_precision  walking_recall  walking_f1  macro_f1
Ridge Classifier    0.9933             0.9975          0.9926      0.9950    0.9924
     Extra Trees    0.9900             0.9950          0.9901      0.9926    0.9886
   Random Forest    0.9900             0.9950          0.9901      0.9926    0.9886
       SVM (RBF)    0.9900             0.9950          0.9901      0.9926    0.9886
       KNN (k=5)    0.9916             0.9926          0.9951      0.9938    0.9904
```
- ~99% across the board — but this is leakage: windows from the same subject are in both train and test sets. Treat LOSO as the honest estimate.

### DS2 GroupKFold k=3 (2 subjects held out per fold)
```
        model  accuracy  walking_precision  walking_recall  walking_f1  macro_f1
    SVM (RBF)    0.8078             0.9182          0.6647      0.7588    0.7986
  Extra Trees    0.7955             0.9128          0.6414      0.7337    0.7821
Random Forest    0.7621             0.9088          0.5657      0.6706    0.7410
       Nu-SVC    0.8025             0.9065          0.6631      0.7504    0.7919
     AdaBoost    0.7816             0.9063          0.6149      0.7130    0.7670
```
- Walking precision ~0.91 but recall is much lower (~0.60–0.70). The models tend to be conservative when predicting walking on unseen subjects.

### DS2 Aggregated (random 70/30 stratified)
```
                 model  accuracy  walking_precision  walking_recall  walking_f1  macro_f1
         Random Forest    0.9370             0.9492          0.9198      0.9342    0.9369
               XGBoost    0.9445             0.9482          0.9372      0.9427    0.9444
     Gradient Boosting    0.9403             0.9477          0.9285      0.9380    0.9402
Hist Gradient Boosting    0.9450             0.9456          0.9411      0.9433    0.9449
               Bagging    0.9389             0.9449          0.9285      0.9366    0.9388
```
- ~94% across both precision and recall (vs k-fold's 0.91 / 0.65).
- The accuracy gap between aggregated and group k-fold (0.94 vs 0.80) is the LOSO/aggregated story repeated on DS2 — random splitting flatters the model.

## 5. External validation — comparing strategies on Validation_Data

Trained on all 6 DS2 subjects, predicted on the 2 unseen MCM subjects (004, 007).
**Three strategies, with `aggregated` widened to 80/20 to deliberately exaggerate any sample-size advantage:**

- **aggregated** — 80% random stratified split of DS2 → train, predict on val. n=5,667 train windows.
- **aggregated_matched** — same as aggregated but subsampled so n equals the k-fold per-fold average. n=4,722 windows. **Isolates the strategy effect from the sample-size effect.**
- **kfold** — GroupKFold k=3 by subject; 3 fold models trained on 4 subjects each (n≈4,722/fold), predictions on val averaged across folds.

Two derived columns:
- `sample_size_effect` = aggregated(80%) − aggregated_matched (extra-data benefit at the same strategy)
- `strategy_effect` = aggregated_matched − kfold (subject-shuffled benefit at equal n)

### Pooled validation walking precision — full 22 classifiers (sorted by aggregated)

```
                 model  agg(80%)  matched   kfold     size_eff   strat_eff
         Random Forest  0.9321    0.9338    0.9283    -0.0016    +0.0054
           Extra Trees  0.9275    0.9266    0.9297    +0.0009    -0.0032
             SVM (RBF)  0.9270    0.9276    0.9269    -0.0006    +0.0007
Quadratic Discriminant  0.9229    0.8261    0.8845    +0.0968    -0.0584
     Gradient Boosting  0.9199    0.9206    0.9161    -0.0007    +0.0045
   Logistic Regression  0.9191    0.9214    0.9179    -0.0023    +0.0035
          SVM (Linear)  0.9187    0.9203    0.9188    -0.0016    +0.0015
                Nu-SVC  0.9173    0.9173    0.9172    +0.0000    +0.0002
    Passive Aggressive  0.9144    0.8657    0.8651    +0.0488    +0.0006
   Linear Discriminant  0.9141    0.9125    0.9134    +0.0016    -0.0009
      Ridge Classifier  0.9135    0.9135    0.9114    +0.0001    +0.0021
        SGD Classifier  0.9120    0.9021    0.9121    +0.0099    -0.0101
              AdaBoost  0.9117    0.9182    0.9054    -0.0065    +0.0129
               Bagging  0.9111    0.9121    0.8826    -0.0010    +0.0294
            KNN (k=11)  0.9109    0.9116    0.9124    -0.0006    -0.0008
Hist Gradient Boosting  0.9057    0.9085    0.9000    -0.0029    +0.0085
                   MLP  0.9050    0.9065    0.8913    -0.0015    +0.0152
             KNN (k=5)  0.9031    0.9039    0.9007    -0.0008    +0.0032
               XGBoost  0.8978    0.9018    0.9031    -0.0040    -0.0014
           Gaussian NB  0.8932    0.8963    0.8987    -0.0031    -0.0024
         Decision Tree  0.8595    0.8460    0.8138    +0.0135    +0.0322
            Extra Tree  0.8464    0.8655    0.7931    -0.0191    +0.0724
```

### Aggregate stats

- Mean **sample_size_effect**: +0.0057 (median -0.0007). Heavily driven by 2 outliers (QDA +0.097, PA +0.049). For the other 20 models the median is essentially zero — most are slightly negative.
- Mean **strategy_effect**: +0.0052 (median +0.0018). Small positive on average; largest positive contributions are from Extra Tree (+0.072), Decision Tree (+0.032), Bagging (+0.029); largest negative from QDA (-0.058) and SGD (-0.010).

### Per-validation-subject walking precision (aggregated strategy, full registry, top of table)
- MCM-004 (n=1,316 windows): top models hit ~0.93–0.94
- MCM-007 (n=1,102 windows): top models hit ~0.92–0.93
- The two unseen subjects perform comparably; no strong subject-by-subject divergence (unlike DS1's subject 7).

## 6. Key findings

### A. Subject-aware splitting matters dramatically more than sample size
- DS1 LOSO walking precision averages ~0.86 across the registry; DS1 aggregated averages ~0.99. That ~13-point gap is **purely** the subject-leakage effect — random splitting puts windows from the same subject in both train and test.
- DS2 mirrors this: GroupKFold ~0.80 accuracy / 0.91 walking precision vs aggregated ~0.94 / 0.95.
- The honest estimate of generalization to unseen subjects is the subject-grouped one. Aggregated random splits should be reported only with a caveat.

### B. The "aggregated wins" effect on Validation_Data is real but tiny
- For the top performers (RF, Extra Trees, SVM-RBF, GB, LR), the gap between aggregated(80%) and kfold on validation is ~0.003–0.006 walking precision — well within fold-level noise.
- When you control for sample size (aggregated_matched vs kfold), the gap shrinks further or flips sign for several models (LDA, ExtraTrees, NB, XGB, KNN-11).
- Conclusion: **for deployable well-tuned models, the choice between random and subject-grouped splitting barely affects out-of-sample performance once you control for n**.

### C. The exception: simple/fragile tree models
- Decision Tree, Extra Tree, Bagging show meaningful strategy effects (+0.03 to +0.07): they really do prefer random within-subject sampling. They overfit subject-specific quirks, so subject-grouped training data hurts them more.
- These models also tend to be lower-performing overall — they're not what you'd ship.

### D. The QDA outlier
- QDA shows the largest sample-size effect (+0.097): more data dramatically helps because its covariance estimate is unstable at small n.
- But its strategy effect is strongly negative (-0.058): kfold beats matched-aggregated by 6 points at equal n.
- Practical read: QDA is data-hungry and split-sensitive; not a reliable deployment choice for this dataset.

### E. The Passive Aggressive anomaly
- Sample-size effect is +0.049 (more data helped) but the matched version actually has lower walking precision (0.866) than the kfold version (0.865). PA's online-update sensitivity to sample order makes it unstable — this should not be read as evidence that "less data helps PA"; it's noise in PA's training trajectory.

### F. Subject 7 (DS1) is a real outlier
- Across nearly every model in DS1 LOSO, walking precision is ~0.0–0.3 when subject 7 is held out, while subjects 1 and 5 sit at ~0.7–1.0.
- This drags the LOSO mean down by ~10–15 percentage points on most models compared to a 2-subject-only LOSO would.
- Either the IMU mounting differed for subject 7, the activity patterns are unusually different, or there's a sensor/orientation issue. **Worth investigating before drawing strong conclusions from the DS1 LOSO numbers.**

### G. DS2 has lower recall than precision under group k-fold
- Walking precision ~0.91 but recall ~0.60–0.70.
- Models are biased toward not predicting walking on unseen subjects — they need to see enough subject variation to learn what walking looks like across people.
- Aggregated DS2 doesn't show this gap (precision and recall both ~0.94), confirming that subject-level variation is the missing signal under k-fold.

## 7. Suggested discussion / write-up angles

1. **Train/test discipline matters more than model choice.** The DS1 LOSO-vs-aggregated gap (~13 points) and the DS2 k-fold-vs-aggregated gap (~13 points accuracy) dwarf the model-vs-model differences within either regime (~5 points top to bottom).
2. **At equal training size, the aggregated strategy isn't actually better.** Once we matched the sample sizes (80/20 vs k-fold ~67%), the strategy effect disappeared for the deployable models. The "aggregated wins" reading is mostly a sample-size artifact.
3. **Subject-grouped CV is genuinely pessimistic only because it's correct.** It estimates what you get when you deploy on a new person. The aggregated number is overconfident.
4. **Recommendation for production**: use Random Forest or SVM-RBF, train on as much subject-diverse data as possible, and report performance using the GroupKFold-style estimate (~0.93 walking precision, ~0.91 accuracy on this validation set).
5. **DS1 has too few subjects for stable LOSO.** With n=3 subjects, one outlier (subject 7) tanks the mean. Adding more subjects is the highest-leverage improvement.
6. **DS2 generalizes well to unseen subjects** (validation walking precision 0.93 for the top model, on subjects never seen in any form during training). This is the most important quantitative result of the project.
7. **Open question**: does the DS1 multi-IMU setup buy you anything over DS2's single IMU? They're not directly comparable (different label sets, sample sizes, subject pools), but a meaningful next step would be to drop one IMU from DS1 and re-run, to measure the per-IMU contribution.

## 8. Files & artifacts

- Code: `preprocessing_utils.py`, `preprocess_ds1.py`, `preprocess_ds2.py`, `analysis.py`, `run_ds1_loso.py`, `run_ds1_aggregated.py`, `run_ds2_kfold.py`, `run_ds2_aggregated.py`, `run_all.py`, `validate_ds2.py`
- Processed data: `Processed_Data/DS1_W256_S128.csv`, `DS2_W256_S128.csv`, `Validation_W256_S128.csv`
- Results: `Results_Plots/{DS1_LOSO, DS1_aggregated, DS2_kfold, DS2_aggregated, Validation_DS2}/` — each contains `summary_metrics.csv`, `fold_metrics.csv` (where applicable), bar charts, and top-5 confusion matrices. Validation_DS2 additionally has `comparison_pooled.csv`, `comparison_per_subject.csv`, `strategy_comparison.png`.
