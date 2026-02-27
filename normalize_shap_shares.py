"""
Normalize SHAP importances by total |SHAP| per model to compare relative feature shares
across variants (with_SP, no_SP, noAgeSP_keepPhone). Outputs a CSV in shap_normalized/.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "shap_normalized"
OUTPUT_CSV = OUTPUT_DIR / "normalized_shap_shares.csv"

VARIANTS = [
    ("with_SP", ROOT / "one_hot_encoding_groups", "one_hot_encoding_results.pkl"),
    ("no_SP", ROOT / "one_hot_encoding_groups_noAgeFirstSP", "one_hot_encoding_results_noSP.pkl"),
    (
        "noAgeSP_keepPhone",
        ROOT / "one_hot_encoding_groups_noAgeFirstSP" / "one_hot_encoding_groups_noAgeFirstSP_keepPhone",
        "one_hot_encoding_results_noAgeSP_keepPhone.pkl",
    ),
]

GROUPS = ["rural_hadza", "gm", "age_18_24"]


def load_shap_table(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res["shap_importance"]).copy()
    df = df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    total = df["Mean_SHAP"].sum()
    df["Share"] = df["Mean_SHAP"] / total if total > 0 else 0.0
    df["Rank"] = range(1, len(df) + 1)
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for variant, folder, fname in VARIANTS:
        for group in GROUPS:
            path = folder / group / fname
            if not path.exists():
                print(f"Missing file for {variant}/{group}: {path}")
                continue
            df = load_shap_table(path)
            df["Variant"] = variant
            df["Group"] = group
            rows.append(df)
    if not rows:
        print("No data found; nothing written.")
        return
    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out[["Variant", "Group", "Rank", "Feature", "Mean_SHAP", "Share"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote normalized SHAP shares to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
