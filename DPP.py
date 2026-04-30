"""
Driver — runs preprocessing for both datasets.

Usage:
    python DPP.py            # preprocess DS1 + DS2 with defaults (W=256, S=128)
    python DPP.py ds1        # preprocess only DS1
    python DPP.py ds2        # preprocess only DS2
"""
import sys
from preprocess_ds1 import preprocess_ds1
from preprocess_ds2 import preprocess_ds2


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if target in ("all", "ds1"):
        preprocess_ds1(window_size=256, step_size=128)
    if target in ("all", "ds2"):
        preprocess_ds2(window_size=256, step_size=128)


if __name__ == "__main__":
    main()
