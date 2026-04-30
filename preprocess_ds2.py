"""
DS2-style preprocessor — single IMU, 6-axis (acc + gyro).

Used for two flows:
  - Training:   Raw_Data/DS2/MCM-NNN-HOP-{F,M}_Training_labeled.csv
                (subjects 001, 002, 003, 010, 011, 012)
  - Validation: Validation_Data/MCM-NNN-HOP-{F,M}_Training_labeled.csv
                (subjects auto-discovered, currently 004 + 007)

Each CSV has columns:
    timestamp, acc_x_g, acc_y_g, acc_z_g,
    gyro_x_deg_s, gyro_y_deg_s, gyro_z_deg_s,
    activity, is_walking

Notes:
  - timestamp is only minute-resolution and is NOT trustworthy for alignment;
    we treat the rows as already-ordered at fs=100 Hz (~5923 rows/min observed).
  - We DROP rows with activity == 'unlabeled' before windowing, per user direction.
  - Target column is `is_walking` (binary).
"""

from __future__ import annotations

import os
import re
import pandas as pd

import preprocessing_utils as pp


ACC_COLS = ["acc_x_g", "acc_y_g", "acc_z_g"]
GYRO_COLS = ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]
DS2_SUBJECTS = ["001", "002", "003", "010", "011", "012"]


def _process_one_subject(df: pd.DataFrame, fs: int) -> pd.DataFrame:
    df = df.drop(columns=[c for c in ("timestamp", "activity") if c in df.columns])
    df = pp.apply_noise_filter(df, ACC_COLS + GYRO_COLS, fs=fs, cutoff=20)
    df = pp.separate_gravity_body(df, ACC_COLS, GYRO_COLS, fs=fs, suffix="")
    df = pp.compute_jerk(df, suffix="", fs=fs)
    return df


def preprocess_ds2_like(
    base_path: str,
    save_path: str = "Processed_Data",
    out_filename: str = "DS2_W{w}_S{s}.csv",
    subjects=None,
    tag: str = "DS2",
    window_size: int = 256,
    step_size: int = 128,
    fs: int = 100,
) -> pd.DataFrame:
    """
    Generic single-IMU MCM-* preprocessor used for both DS2 training
    (Raw_Data/DS2) and held-out validation (Validation_Data).

    `subjects` restricts which MCM-NNN files to keep; if None, every
    matching file in `base_path` is included.
    """
    files = sorted(os.listdir(base_path))
    subj_files = {}
    for fn in files:
        m = re.match(r"MCM-(\d+)-HOP-[FM]_Training_labeled\.csv", fn)
        if m:
            subj_files[m.group(1)] = fn

    if subjects is None:
        subjects = sorted(subj_files.keys())

    all_windowed = []
    for subj in subjects:
        if subj not in subj_files:
            print(f"  ⚠️  Subject {subj}: file missing, skipping")
            continue
        print(f"\n[{tag}] Subject {subj}")
        df = pd.read_csv(os.path.join(base_path, subj_files[subj]), low_memory=False)
        before = len(df)

        df = df[df["activity"] != "unlabeled"].reset_index(drop=True)
        print(f"  Rows: {before:,} -> {len(df):,} after dropping unlabeled")

        if len(df) == 0:
            print(f"  ⚠️  No labeled rows for {subj}, skipping")
            continue

        df = _process_one_subject(df, fs=fs)

        windowed = pp.sliding_window_features(
            df,
            label_col="is_walking",
            window_size=window_size,
            step_size=step_size,
            fs=fs,
            fft_suffix_filter=[""],
        )
        windowed["subject_id"] = subj
        all_windowed.append(windowed)
        print(f"  Subject {subj}: {len(windowed)} windows")

    if not all_windowed:
        raise RuntimeError(f"No windows produced from {base_path}")

    out = pd.concat(all_windowed, axis=0).reset_index(drop=True)
    feature_cols = [c for c in out.columns if c not in ("subject_id", "is_walking")]
    out = out[["subject_id", "is_walking"] + feature_cols]
    out["is_walking"] = out["is_walking"].astype(int)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, out_filename.format(w=window_size, s=step_size))
        out.to_csv(out_file, index=False)
        print(f"\n✅ {tag} saved to {out_file}  shape={out.shape}")
    return out


def preprocess_ds2(
    base_path: str = "Raw_Data/DS2",
    save_path: str = "Processed_Data",
    window_size: int = 256,
    step_size: int = 128,
    fs: int = 100,
) -> pd.DataFrame:
    """Backwards-compatible wrapper: DS2 training set (subjects 001/002/003/010/011/012)."""
    return preprocess_ds2_like(
        base_path=base_path,
        save_path=save_path,
        out_filename="DS2_W{w}_S{s}.csv",
        subjects=DS2_SUBJECTS,
        tag="DS2",
        window_size=window_size,
        step_size=step_size,
        fs=fs,
    )


def preprocess_validation(
    base_path: str = "Validation_Data",
    save_path: str = "Processed_Data",
    window_size: int = 256,
    step_size: int = 128,
    fs: int = 100,
) -> pd.DataFrame:
    """
    Validation_Data uses the same DS2 schema; subjects are auto-discovered.
    Output: Processed_Data/Validation_W{w}_S{s}.csv
    """
    return preprocess_ds2_like(
        base_path=base_path,
        save_path=save_path,
        out_filename="Validation_W{w}_S{s}.csv",
        subjects=None,
        tag="Validation",
        window_size=window_size,
        step_size=step_size,
        fs=fs,
    )


if __name__ == "__main__":
    preprocess_ds2()
