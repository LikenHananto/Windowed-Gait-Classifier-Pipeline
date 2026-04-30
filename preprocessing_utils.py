"""
Shared signal-processing primitives used by both DS1 and DS2 preprocessing.

Pipeline stages (each is a small, composable function):
    Stage 1: 20 Hz Butterworth low-pass denoise
    Stage 2: 0.3 Hz low-pass to separate gravity/body (acc) and static/dynamic (gyro)
    Stage 3: jerk = diff(signal) * fs   on body-acc and dynamic-gyro
    Stage 4: sliding-window feature extraction (mean, std, dominant FFT freq)

Compared to the original Data_Pre_Processor.py these helpers are dataset-agnostic:
the caller passes in the raw acc / gyro column names so the same code runs on
DS1 (Var2..Var7) and DS2 (acc_*_g / gyro_*_deg_s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — denoise
# ─────────────────────────────────────────────────────────────────────────────
def apply_noise_filter(df: pd.DataFrame, cols, fs: int = 100, cutoff: float = 20.0) -> pd.DataFrame:
    """20 Hz low-pass Butterworth filter applied in-place to `cols`."""
    df = df.copy()
    nyquist = fs / 2
    b, a = butter(N=4, Wn=cutoff / nyquist, btype="low", analog=False)
    for col in cols:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col].values)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — gravity/body and static/dynamic separation
# ─────────────────────────────────────────────────────────────────────────────
def separate_gravity_body(
    df: pd.DataFrame,
    acc_cols,
    gyro_cols,
    fs: int = 100,
    cutoff: float = 0.3,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Apply a 0.3 Hz low-pass to extract:
        acc_<axis>_gravity / acc_<axis>_body
        gyro_<axis>_static / gyro_<axis>_dynamic

    `suffix` is optional and gets appended (e.g. "_lower") so multi-IMU
    setups don't collide on column names.
    """
    df = df.copy()
    nyquist = fs / 2
    b, a = butter(N=4, Wn=cutoff / nyquist, btype="low", analog=False)

    acc_axes = ["x", "y", "z"]
    gyro_axes = ["x", "y", "z"]

    for col, ax in zip(acc_cols, acc_axes):
        gravity = filtfilt(b, a, df[col].values)
        df[f"acc_{ax}_gravity{suffix}"] = gravity
        df[f"acc_{ax}_body{suffix}"] = df[col].values - gravity

    for col, ax in zip(gyro_cols, gyro_axes):
        static = filtfilt(b, a, df[col].values)
        df[f"gyro_{ax}_static{suffix}"] = static
        df[f"gyro_{ax}_dynamic{suffix}"] = df[col].values - static

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — jerk
# ─────────────────────────────────────────────────────────────────────────────
def compute_jerk(df: pd.DataFrame, suffix: str = "", fs: int = 100) -> pd.DataFrame:
    """
    Jerk = diff(signal) * fs
    Applied to body-acc and dynamic-gyro. Adds `<col>_jerk` columns.
    """
    df = df.copy()
    sources = (
        [f"acc_{ax}_body{suffix}" for ax in ["x", "y", "z"]]
        + [f"gyro_{ax}_dynamic{suffix}" for ax in ["x", "y", "z"]]
    )
    sources = [c for c in sources if c in df.columns]
    for col in sources:
        df[f"{col}_jerk"] = np.diff(df[col].values, prepend=df[col].values[0]) * fs
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — sliding-window feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window_features(
    data: pd.DataFrame,
    label_col: str,
    window_size: int = 256,
    step_size: int = 128,
    fs: int = 100,
    fft_suffix_filter=None,
) -> pd.DataFrame:
    """
    Slide a (window_size, step_size) window over `data` and emit one feature
    row per window. Mixed-label windows are dropped.

    Per window:
      - mean, std for every numeric column except `label_col`
      - dominant FFT frequency for body-acc, dynamic-gyro and their jerks

    `fft_suffix_filter`: optional list of suffix strings (e.g. ['_lower', '_upper'])
    to enumerate FFT-eligible columns across multiple IMUs. Pass [''] (default)
    for a single IMU.
    """
    if fft_suffix_filter is None:
        fft_suffix_filter = [""]

    feature_cols = [c for c in data.columns if c != label_col]

    fft_cols = []
    for sfx in fft_suffix_filter:
        fft_cols += [
            f"acc_{ax}_body{sfx}" for ax in ["x", "y", "z"]
        ] + [
            f"gyro_{ax}_dynamic{sfx}" for ax in ["x", "y", "z"]
        ] + [
            f"acc_{ax}_body{sfx}_jerk" for ax in ["x", "y", "z"]
        ] + [
            f"gyro_{ax}_dynamic{sfx}_jerk" for ax in ["x", "y", "z"]
        ]
    fft_cols = [c for c in fft_cols if c in feature_cols]

    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    rows = []
    dropped = 0

    label_values = data[label_col].values

    for i in range(0, len(data) - window_size + 1, step_size):
        window_labels = label_values[i : i + window_size]
        unique_labels = pd.unique(window_labels)
        if len(unique_labels) > 1:
            dropped += 1
            continue
        task = unique_labels[0]

        window = data.iloc[i : i + window_size]
        window_features = window[feature_cols]
        means = window_features.mean()
        stds = window_features.std()

        row_dict = {}
        for col in feature_cols:
            row_dict[f"{col}_mean"] = means[col]
            row_dict[f"{col}_std"] = stds[col]

        for col in fft_cols:
            signal = window[col].values
            mag = np.abs(np.fft.rfft(signal))
            mag[0] = 0  # zero DC
            row_dict[f"{col}_dominant_freq"] = freqs[np.argmax(mag)]

        row_dict[label_col] = task
        rows.append(row_dict)

    print(f"  Sliding window: kept {len(rows)} windows, dropped {dropped} mixed")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DS1-specific helper — temporal alignment of paired (lower, upper) IMUs
# ─────────────────────────────────────────────────────────────────────────────
def align_two_sensors(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    time_col: str = "time",
    freq: str = "10ms",
):
    """
    Resample two sensor streams onto a common time grid by linear interpolation.
    Returns (df_a_aligned, df_b_aligned) on the same DatetimeIndex.
    """
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a.index = pd.to_datetime(df_a[time_col])
    df_b.index = pd.to_datetime(df_b[time_col])
    df_a.drop(columns=[time_col], inplace=True)
    df_b.drop(columns=[time_col], inplace=True)
    df_a = df_a[~df_a.index.duplicated(keep="first")]
    df_b = df_b[~df_b.index.duplicated(keep="first")]

    num_a = df_a.select_dtypes(include="number").columns
    num_b = df_b.select_dtypes(include="number").columns

    start = max(df_a.index[0], df_b.index[0])
    end = min(df_a.index[-1], df_b.index[-1])
    grid = pd.date_range(start=start, end=end, freq=freq)

    def to_grid(df, num_cols):
        df_num = (
            df[num_cols]
            .reindex(df.index.union(grid))
            .interpolate(method="time")
            .reindex(grid)
        )
        non_num = [c for c in df.columns if c not in num_cols]
        if non_num:
            df_cat = (
                df[non_num]
                .reindex(df.index.union(grid))
                .ffill()
                .bfill()
                .reindex(grid)
            )
            return pd.concat([df_num, df_cat], axis=1)
        return df_num

    return to_grid(df_a, num_a), to_grid(df_b, num_b)
