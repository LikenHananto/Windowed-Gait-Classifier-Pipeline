"""
DS1 preprocessor — multi-IMU (upper + lower), 3 subjects.

Input layout (Raw_Data/DS1/):
    NN_train_lower_labeled.csv
    NN_train_upper_labeled.csv
where NN ∈ {01, 05, 07}.

Each CSV has columns: Var1, Var2..Var7, Task, time
    Var2..Var4 = accelerometer xyz
    Var5..Var7 = gyroscope xyz

Output: Processed_Data/DS1_W{w}_S{s}.csv with
    subject_id  Task  <feature columns ...>

Each row is one sliding window over a (lower, upper) sensor pair for one
subject. Lower features get '_lower' suffix; upper features get '_upper'.
"""

from __future__ import annotations

import os
import re
import pandas as pd

import preprocessing_utils as pp


ACC_COLS = ["Var2", "Var3", "Var4"]
GYRO_COLS = ["Var5", "Var6", "Var7"]
DS1_SUBJECTS = ["01", "05", "07"]


def _process_one_pair(df_lower: pd.DataFrame, df_upper: pd.DataFrame, fs: int):
    df_lower, df_upper = pp.align_two_sensors(df_lower, df_upper, time_col="time", freq="10ms")

    df_lower = pp.apply_noise_filter(df_lower, ACC_COLS + GYRO_COLS, fs=fs, cutoff=20)
    df_upper = pp.apply_noise_filter(df_upper, ACC_COLS + GYRO_COLS, fs=fs, cutoff=20)

    df_lower = pp.separate_gravity_body(df_lower, ACC_COLS, GYRO_COLS, fs=fs, suffix="_lower")
    df_upper = pp.separate_gravity_body(df_upper, ACC_COLS, GYRO_COLS, fs=fs, suffix="_upper")

    df_lower = pp.compute_jerk(df_lower, suffix="_lower", fs=fs)
    df_upper = pp.compute_jerk(df_upper, suffix="_upper", fs=fs)

    # rename raw columns so suffix is consistent
    df_lower = df_lower.rename(columns={c: f"{c}_lower" for c in ACC_COLS + GYRO_COLS})
    df_upper = df_upper.rename(columns={c: f"{c}_upper" for c in ACC_COLS + GYRO_COLS})

    # combine into one wide dataframe — both have the same DatetimeIndex
    combined = pd.concat([df_lower, df_upper], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def preprocess_ds1(
    base_path: str = "Raw_Data/DS1",
    save_path: str = "Processed_Data",
    window_size: int = 256,
    step_size: int = 128,
    fs: int = 100,
) -> pd.DataFrame:
    files = sorted(os.listdir(base_path))
    pairs = {}
    for fn in files:
        m = re.match(r"(\d+)_train_(lower|upper)_labeled\.csv", fn)
        if not m:
            continue
        subj, side = m.group(1), m.group(2)
        if subj not in DS1_SUBJECTS:
            continue
        pairs.setdefault(subj, {})[side] = fn

    all_windowed = []
    for subj in DS1_SUBJECTS:
        if subj not in pairs or "lower" not in pairs[subj] or "upper" not in pairs[subj]:
            print(f"  ⚠️  Subject {subj}: missing files, skipping")
            continue
        print(f"\n[DS1] Subject {subj}")
        df_lower = pd.read_csv(os.path.join(base_path, pairs[subj]["lower"]), low_memory=False)
        df_upper = pd.read_csv(os.path.join(base_path, pairs[subj]["upper"]), low_memory=False)

        for df in (df_lower, df_upper):
            if "Var1" in df.columns:
                df.drop(columns=["Var1"], inplace=True)
            df["Task"] = df["Task"].fillna("Unknown")

        combined = _process_one_pair(df_lower, df_upper, fs=fs)

        # Drop "Unknown" rows so unlabeled time doesn't contaminate windows
        combined = combined[combined["Task"] != "Unknown"]

        windowed = pp.sliding_window_features(
            combined,
            label_col="Task",
            window_size=window_size,
            step_size=step_size,
            fs=fs,
            fft_suffix_filter=["_lower", "_upper"],
        )
        windowed["subject_id"] = subj
        all_windowed.append(windowed)
        print(f"  Subject {subj}: {len(windowed)} windows")

    out = pd.concat(all_windowed, axis=0).reset_index(drop=True)

    # tidy column order: subject_id, Task, <features>
    feature_cols = [c for c in out.columns if c not in ("subject_id", "Task")]
    out = out[["subject_id", "Task"] + feature_cols]

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, f"DS1_W{window_size}_S{step_size}.csv")
        out.to_csv(out_file, index=False)
        print(f"\n✅ DS1 saved to {out_file}  shape={out.shape}")
    return out


if __name__ == "__main__":
    preprocess_ds1()
