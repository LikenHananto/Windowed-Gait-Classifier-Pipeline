"""DS2 — Group K-Fold (k=3) by subject (2 held out per fold). Target: is_walking."""
import os, sys, pandas as pd
import analysis
from preprocess_ds2 import preprocess_ds2


WINDOW, STEP = 256, 128
CSV = f"Processed_Data/DS2_W{WINDOW}_S{STEP}.csv"


def main():
    fast = "--fast" in sys.argv
    if not os.path.exists(CSV):
        print(f"Processed file not found, running preprocess_ds2 ...")
        preprocess_ds2(window_size=WINDOW, step_size=STEP)
    df = pd.read_csv(CSV)
    print(f"Loaded {CSV}  shape={df.shape}  subjects={sorted(df['subject_id'].unique())}")
    clfs = analysis.build_classifiers_fast() if fast else analysis.build_classifiers()
    analysis.run_analysis(
        df,
        label_col="is_walking",
        save_path="Results_Plots/DS2_kfold",
        split="groupkfold",
        k=3,
        classifiers=clfs,
        title_suffix="DS2 (3-fold by subject)" + (" [fast]" if fast else ""),
    )


if __name__ == "__main__":
    main()
