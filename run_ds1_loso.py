"""DS1 — Leave-One-Subject-Out (3 folds: 01, 05, 07).

Usage:
    python run_ds1_loso.py            # full 22-classifier registry (slow)
    python run_ds1_loso.py --fast     # 7-classifier smoke set (fast)
"""
import os, sys, pandas as pd
import analysis
from preprocess_ds1 import preprocess_ds1


WINDOW, STEP = 256, 128
CSV = f"Processed_Data/DS1_W{WINDOW}_S{STEP}.csv"


def main():
    fast = "--fast" in sys.argv
    if not os.path.exists(CSV):
        print(f"Processed file not found, running preprocess_ds1 ...")
        preprocess_ds1(window_size=WINDOW, step_size=STEP)
    df = pd.read_csv(CSV)
    print(f"Loaded {CSV}  shape={df.shape}  subjects={sorted(df['subject_id'].unique())}")
    clfs = analysis.build_classifiers_fast() if fast else analysis.build_classifiers()
    analysis.run_analysis(
        df,
        label_col="Task",
        save_path="Results_Plots/DS1_LOSO",
        split="loso",
        classifiers=clfs,
        title_suffix="DS1" + (" [fast]" if fast else ""),
    )


if __name__ == "__main__":
    main()
