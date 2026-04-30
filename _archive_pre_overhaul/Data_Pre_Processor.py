import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

ACC_COLS  = ['Var2', 'Var3', 'Var4']
GYRO_COLS = ['Var5', 'Var6', 'Var7']


def apply_noise_filter(df, fs=100, cutoff=20, cols=None):
    """
    Stage 1: 20 Hz low-pass Butterworth filter to remove noise.
    Replaces raw signal in-place with the filtered version.
    """
    df = df.copy()
    if cols is None:
        cols = ACC_COLS + GYRO_COLS

    nyquist = fs / 2
    b, a = butter(N=4, Wn=cutoff / nyquist, btype='low', analog=False)

    print(f"  Applying 20 Hz noise filter to: {cols}")
    for col in cols:
        if col in df.columns:
            df[col] = filtfilt(b, a, df[col].values)
    return df


def separate_gravity_body(df, fs=100, cutoff=0.3, acc_cols=None, gyro_cols=None):
    """
    Stage 2: 0.3 Hz low-pass filter to separate gravity/body and static/dynamic.

    For accelerometers:
        acc_x/y/z_gravity = 0.3 Hz low-pass
        acc_x/y/z_body    = filtered - gravity
    For gyroscopes:
        gyro_x/y/z_static  = 0.3 Hz low-pass
        gyro_x/y/z_dynamic = filtered - static
    """
    df = df.copy()
    if acc_cols is None:
        acc_cols  = ACC_COLS
    if gyro_cols is None:
        gyro_cols = GYRO_COLS

    nyquist = fs / 2
    b, a = butter(N=4, Wn=cutoff / nyquist, btype='low', analog=False)

    acc_axis_map  = {col: ax for col, ax in zip(acc_cols,  ['x', 'y', 'z'])}
    gyro_axis_map = {col: ax for col, ax in zip(gyro_cols, ['x', 'y', 'z'])}

    print(f"  Separating gravity/body for acc:  {acc_cols}")
    for col in acc_cols:
        ax = acc_axis_map[col]
        df[f"acc_{ax}_gravity"] = filtfilt(b, a, df[col].values)
        df[f"acc_{ax}_body"]    = df[col].values - df[f"acc_{ax}_gravity"].values

    print(f"  Separating static/dynamic for gyro: {gyro_cols}")
    for col in gyro_cols:
        ax = gyro_axis_map[col]
        df[f"gyro_{ax}_static"]  = filtfilt(b, a, df[col].values)
        df[f"gyro_{ax}_dynamic"] = df[col].values - df[f"gyro_{ax}_static"].values

    return df


def compute_jerk(df, fs=100):
    """
    Stage 3: Compute jerk (rate of change) for body acc and gyro dynamic columns.

    Jerk = diff(signal) * fs
    Produces new columns:
        acc_x/y/z_body_jerk
        gyro_x/y/z_dynamic_jerk

    The first sample is set to 0 (no previous sample to diff against).
    """
    df = df.copy()

    jerk_source_cols = (
        [f"acc_{ax}_body"     for ax in ['x', 'y', 'z']] +
        [f"gyro_{ax}_dynamic" for ax in ['x', 'y', 'z']]
    )

    # Only process columns that actually exist
    jerk_source_cols = [c for c in jerk_source_cols if c in df.columns]

    print(f"  Computing jerk for: {jerk_source_cols}")
    for col in jerk_source_cols:
        jerk = np.diff(df[col].values, prepend=df[col].values[0]) * fs
        df[f"{col}_jerk"] = jerk

    return df


def align_sensors(df_lower, df_upper, time_col='time', freq='10ms'):
    df_lower = df_lower.copy()
    df_upper = df_upper.copy()

    df_lower.index = pd.to_datetime(df_lower[time_col])
    df_upper.index = pd.to_datetime(df_upper[time_col])

    df_lower.drop(columns=[time_col], inplace=True)
    df_upper.drop(columns=[time_col], inplace=True)

    df_lower = df_lower[~df_lower.index.duplicated(keep='first')]
    df_upper = df_upper[~df_upper.index.duplicated(keep='first')]

    numeric_cols_lower = df_lower.select_dtypes(include='number').columns
    numeric_cols_upper = df_upper.select_dtypes(include='number').columns

    start = max(df_lower.index[0], df_upper.index[0])
    end   = min(df_lower.index[-1], df_upper.index[-1])
    common_grid = pd.date_range(start=start, end=end, freq=freq)

    def resample_to_grid(df, numeric_cols, grid):
        df_num = (df[numeric_cols]
                  .reindex(df.index.union(grid))
                  .interpolate(method='time')
                  .reindex(grid))
        non_numeric = [c for c in df.columns if c not in numeric_cols]
        if non_numeric:
            df_cat = (df[non_numeric]
                      .reindex(df.index.union(grid))
                      .ffill()
                      .bfill()
                      .reindex(grid))
            return pd.concat([df_num, df_cat], axis=1)
        return df_num

    df_lower = resample_to_grid(df_lower, numeric_cols_lower, common_grid)
    df_upper = resample_to_grid(df_upper, numeric_cols_upper, common_grid)

    return df_lower, df_upper


def sliding_window(data, window_size=128, step_size=64, label_col='Task', fs=100):
    """
    Extracts per-window features:
      - Mean and std for ALL columns
      - Dominant FFT frequency for body acc, gyro dynamic, and their jerks
      - Mixed windows (more than one unique label) are dropped
    """
    feature_cols = [c for c in data.columns if c != label_col]

    # Columns eligible for dominant frequency extraction
    fft_cols = (
        [f"acc_{ax}_body"          for ax in ['x', 'y', 'z']] +
        [f"gyro_{ax}_dynamic"      for ax in ['x', 'y', 'z']] +
        [f"acc_{ax}_body_jerk"     for ax in ['x', 'y', 'z']] +
        [f"gyro_{ax}_dynamic_jerk" for ax in ['x', 'y', 'z']]
    )
    fft_cols = [c for c in fft_cols if c in feature_cols]

    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    rows = []
    dropped = 0

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i + window_size]

        # ── Drop mixed-label windows ─────────────────────────────────────────
        unique_labels = window[label_col].unique()
        if len(unique_labels) > 1:
            dropped += 1
            continue
        task = unique_labels[0]

        window_features = window[feature_cols]
        mean_values = window_features.mean()
        std_values  = window_features.std()

        row_dict = {}

        # ── Time-domain: mean + std for all columns ──────────────────────────
        for col in feature_cols:
            row_dict[f"{col}_mean"] = mean_values[col]
            row_dict[f"{col}_std"]  = std_values[col]

        # ── Frequency-domain: dominant freq for body, dynamic, and jerks ─────
        for col in fft_cols:
            signal = window[col].values
            fft_magnitude = np.abs(np.fft.rfft(signal))
            fft_magnitude[0] = 0  # zero DC component
            row_dict[f"{col}_dominant_freq"] = freqs[np.argmax(fft_magnitude)]

        row_dict[label_col] = task
        rows.append(row_dict)

    print(f"  Dropped {dropped} mixed-label windows")
    return pd.DataFrame(rows)


def preprocess_lab_data(base_path="Raw_Data/Labeled", window_size=128, step_size=64, save_path="Processed_Data"):

    # ── Load all CSV files ───────────────────────────────────────────────────
    dataframes = []
    files = sorted(f for f in os.listdir(base_path) if f.endswith(".csv"))
    for filename in files:
        print(f"Loading {filename}...")
        df = pd.read_csv(os.path.join(base_path, filename), low_memory=False)
        dataframes.append(df)

    # ── Drop unused columns, fill missing labels ─────────────────────────────
    for df in dataframes:
        if 'Var1' in df.columns:
            df.drop(columns=['Var1'], inplace=True)
        if 'Task' in df.columns:
            df['Task'] = df['Task'].fillna('Unknown')

    # ── Align each lower/upper sensor pair ───────────────────────────────────
    aligned_pairs = []
    for i in range(0, len(dataframes), 2):
        print(f"Aligning sensor pair {i} (lower) and {i+1} (upper)...")
        df_lower, df_upper = align_sensors(
            dataframes[i], dataframes[i + 1], time_col='time', freq='10ms'
        )
        aligned_pairs.append((df_lower, df_upper))

    # ── Stage 1: 20 Hz noise filter ──────────────────────────────────────────
    denoised_pairs = []
    for pair_idx, (df_lower, df_upper) in enumerate(aligned_pairs):
        print(f"Stage 1 — Noise filter for pair {pair_idx}...")
        df_lower = apply_noise_filter(df_lower, fs=100, cutoff=20)
        df_upper = apply_noise_filter(df_upper, fs=100, cutoff=20)
        denoised_pairs.append((df_lower, df_upper))

    # ── Stage 2: 0.3 Hz gravity/body separation ───────────────────────────────
    filtered_pairs = []
    for pair_idx, (df_lower, df_upper) in enumerate(denoised_pairs):
        print(f"Stage 2 — Gravity/body separation for pair {pair_idx}...")
        df_lower = separate_gravity_body(df_lower, fs=100, cutoff=0.3)
        df_upper = separate_gravity_body(df_upper, fs=100, cutoff=0.3)
        filtered_pairs.append((df_lower, df_upper))

    # ── Stage 3: Jerk computation ─────────────────────────────────────────────
    jerk_pairs = []
    for pair_idx, (df_lower, df_upper) in enumerate(filtered_pairs):
        print(f"Stage 3 — Jerk computation for pair {pair_idx}...")
        df_lower = compute_jerk(df_lower, fs=100)
        df_upper = compute_jerk(df_upper, fs=100)
        jerk_pairs.append((df_lower, df_upper))

    # ── Stage 4: Sliding window + suffix renaming ─────────────────────────────
    session_dfs = []
    for pair_idx, (df_lower, df_upper) in enumerate(jerk_pairs):
        pair_windowed = []
        for sensor_df, suffix in [(df_lower, '_lower'), (df_upper, '_upper')]:
            print(f"  Stage 4 — Sliding window for pair {pair_idx}, suffix {suffix}...")
            df_w = sliding_window(sensor_df, window_size=window_size,
                                  step_size=step_size, fs=100)
            new_columns = {col: f"{col}{suffix}" for col in df_w.columns if col != 'Task'}
            df_w.rename(columns=new_columns, inplace=True)
            print(f"  Shape: {df_w.shape}")
            pair_windowed.append(df_w)

        merged = pd.concat(pair_windowed, axis=1, join='inner')
        merged = merged.loc[:, ~merged.columns.duplicated()]
        session_dfs.append(merged)

    # ── Stack all sessions ────────────────────────────────────────────────────
    merged_df = pd.concat(session_dfs, axis=0).reset_index(drop=True)
    print(f"Final shape: {merged_df.shape}")

    if save_path:
        out_path = f"{save_path}/lab_W{window_size}_S{step_size}.csv"
        merged_df.to_csv(out_path, index=False)
        print(f"Saved to '{out_path}'")

    return merged_df