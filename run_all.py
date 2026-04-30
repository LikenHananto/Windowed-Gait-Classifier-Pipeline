"""
Run all four analyses end-to-end, then write a comparison summary.

    1. DS1 LOSO              (3 folds: 01, 05, 07)
    2. DS1 aggregated        (random 70/30 stratified)
    3. DS2 group K-fold k=3  (2 subjects held out per fold)
    4. DS2 aggregated        (random 70/30 stratified)

Usage:
    python run_all.py            # full 22-classifier registry (slow, hours)
    python run_all.py --fast     # 7-classifier smoke set (fast, ~minute)
"""
import os, sys
import pandas as pd

import run_ds1_loso
import run_ds1_aggregated
import run_ds2_kfold
import run_ds2_aggregated


def main():
    print("\n" + "=" * 80 + "\n[1/4] DS1 — LOSO\n" + "=" * 80)
    run_ds1_loso.main()

    print("\n" + "=" * 80 + "\n[2/4] DS1 — aggregated\n" + "=" * 80)
    run_ds1_aggregated.main()

    print("\n" + "=" * 80 + "\n[3/4] DS2 — group K-fold (k=3)\n" + "=" * 80)
    run_ds2_kfold.main()

    print("\n" + "=" * 80 + "\n[4/4] DS2 — aggregated\n" + "=" * 80)
    run_ds2_aggregated.main()

    # ── Cross-analysis comparison ───────────────────────────────────────────
    out_root = "Results_Plots"
    parts = []
    for tag in ("DS1_LOSO", "DS1_aggregated", "DS2_kfold", "DS2_aggregated"):
        p = os.path.join(out_root, tag, "summary_metrics.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.insert(0, "analysis", tag)
            parts.append(df)
    if parts:
        big = pd.concat(parts, axis=0).reset_index(drop=True)
        big.to_csv(os.path.join(out_root, "all_summaries.csv"), index=False)
        print(f"\n✅ Cross-analysis summary -> {out_root}/all_summaries.csv")
        print(big.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
