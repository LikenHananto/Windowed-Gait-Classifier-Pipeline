# Human Activity Recognition — Gait Classifier

Two datasets, four analyses. Both datasets target a binary "walking" classification.

## Data layout

```
Raw_Data/
  DS1/   — multi-IMU (upper + lower), 3 subjects (01, 05, 07)
           NN_train_lower_labeled.csv
           NN_train_upper_labeled.csv
           Cols: Var1, Var2..Var7, Task, time
           Labels: walking, bending, (Unknown)
  DS2/   — single IMU, 6 subjects (MCM-001..012)
           MCM-NNN-HOP-{F,M}_Training_labeled.csv
           Cols: timestamp, acc_x_g..z, gyro_x_deg_s..z, activity, is_walking
           Labels: 8 activities + 'unlabeled' (dropped); target is `is_walking`
```

Both datasets sample at ~100 Hz.

## Pipeline

1. `preprocessing_utils.py` — shared signal-processing primitives:
   - 20 Hz Butterworth low-pass denoise
   - 0.3 Hz low-pass for gravity/body and static/dynamic separation
   - Jerk = diff(signal) * fs on body-acc / dynamic-gyro
   - Sliding-window features: mean, std, dominant FFT freq

2. `preprocess_ds1.py` — DS1: align upper/lower IMU pair on a 10ms grid, then
   apply the shared pipeline with `_lower` / `_upper` suffixes. Drops "Unknown"
   rows. Output: `Processed_Data/DS1_W{w}_S{s}.csv` with columns
   `subject_id, Task, <features>` (122 feature cols at W=256, S=128).

3. `preprocess_ds2.py` — DS2: drops `activity == 'unlabeled'`, then applies
   the shared pipeline (no alignment, single IMU). Output:
   `Processed_Data/DS2_W{w}_S{s}.csv` with columns
   `subject_id, is_walking, <features>` (62 feature cols at W=256, S=128).

4. `analysis.py` — generic runner. Splits supported:
   - `loso` — Leave-One-Subject-Out
   - `groupkfold` — k folds, subjects as groups
   - `aggregated` — random stratified train/test split (no subject separation)

   For fold-based splits, per-fold metrics are averaged and confusion matrices
   are summed. Walking-positive class is auto-detected (`'walking'` for DS1,
   `1` for DS2).

5. `Trainer.py` — kept from the original codebase (per-model train/eval helper
   + walking-threshold sweep). The new `analysis.py` does not currently call
   threshold tuning, but the helper is still useable standalone.

## Run scripts

| Script                    | Dataset | Split scheme                  |
| ------------------------- | ------- | ----------------------------- |
| `run_ds1_loso.py`         | DS1     | LOSO (3 folds: 01, 05, 07)    |
| `run_ds1_aggregated.py`   | DS1     | Random 70/30 stratified       |
| `run_ds2_kfold.py`        | DS2     | GroupKFold k=3 (2 subj/fold)  |
| `run_ds2_aggregated.py`   | DS2     | Random 70/30 stratified       |
| `run_all.py`              | both    | All four, plus combined `all_summaries.csv` |
| `validate_ds2.py`         | DS2     | Train on full DS2 (all 6 subjects), predict on `Validation_Data/` (subjects 004, 007) |

Each accepts `--fast` to use a 7-classifier smoke set instead of the full
22-classifier registry. Without `--fast` the runs can take a long time on
slower machines (Nu-SVC, GradientBoosting and MLP dominate).

```
python DPP.py                 # preprocess both datasets
python run_all.py             # full registry (slow)
python run_all.py --fast      # smoke set (~minute total)
```

## Outputs

```
Results_Plots/
  DS1_LOSO/         summary_metrics.csv, fold_metrics.csv, classifier_comparison.png, cm_<top5>.png
  DS1_aggregated/   same set
  DS2_kfold/        same set
  DS2_aggregated/   same set
  all_summaries.csv (combined table from run_all.py)
```

`fold_metrics.csv` is long-format (one row per (fold, model)); `summary_metrics.csv`
is the per-model average across folds.

## Notes

- **Subject 7 is an outlier under LOSO**: when held out, every model collapses
  toward majority-class prediction (~36% accuracy). This drags the LOSO mean
  metrics down significantly compared to aggregated. Worth investigating
  whether subject 7's recording differs in mounting orientation or sensor.
- The **aggregated DS1** number (~99% accuracy) is optimistic — it leaks
  windows from the same subject across train and test. Treat LOSO as the
  honest cross-subject estimate.
- DS2 GroupKFold with k=3 means 2 subjects are held out per fold; the
  combined train set per fold is ~4700 windows, validation ~2300.
- The original code is preserved under `_archive_pre_overhaul/`.
