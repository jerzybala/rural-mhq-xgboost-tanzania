"""
Compare SHAP mean absolute values between one-hot group runs.
Produces Excel and bar plots of pairwise SHAP mean differences.
"""

from __future__ import annotations

import itertools
import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

GROUP_DIRS: Dict[str, str] = {
    "rural_hadza": "rural_hadza",
    "gm": "gm",
    "age_18_24": "age_18_24",
}

RESULTS_FILENAME = "one_hot_encoding_results.pkl"
OUTPUT_EXCEL = "group_shap_comparison.xlsx"
PLOT_TOP_N = 20


def load_shap_mean(group_key: str, base_dir: str = ".") -> pd.DataFrame:
    path = os.path.join(base_dir, GROUP_DIRS[group_key], RESULTS_FILENAME)
    with open(path, "rb") as f:
        res = pickle.load(f)
    shap_df = pd.DataFrame(res["shap_importance"]).copy()
    shap_df = shap_df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    shap_df["Group"] = group_key
    return shap_df[["Group", "Feature", "Mean_SHAP"]]


def pairwise_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(df_a, df_b, on="Feature", how="outer", suffixes=("_a", "_b"))
    merged["Mean_SHAP_a"] = merged["Mean_SHAP_a"].fillna(0)
    merged["Mean_SHAP_b"] = merged["Mean_SHAP_b"].fillna(0)
    merged["diff"] = merged["Mean_SHAP_a"] - merged["Mean_SHAP_b"]
    merged["abs_diff"] = merged["diff"].abs()
    merged = merged.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return merged


def plot_top_diffs(df_diff: pd.DataFrame, title: str, outfile: str):
    top = df_diff.head(PLOT_TOP_N)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top["Feature"], top["diff"], color=["#4CAF50" if v > 0 else "#F44336" for v in top["diff"]])
    plt.xlabel("Mean SHAP difference (A - B)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.gca().tick_params(axis="y", labelsize=7)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = "."
    shap_frames = {k: load_shap_mean(k, base_dir) for k in GROUP_DIRS}

    # Write combined means
    with pd.ExcelWriter(os.path.join(base_dir, OUTPUT_EXCEL)) as writer:
        for k, df in shap_frames.items():
            df.sort_values("Mean_SHAP", ascending=False).to_excel(writer, sheet_name=f"means_{k}", index=False)

        for a, b in itertools.combinations(GROUP_DIRS.keys(), 2):
            diff_df = pairwise_diff(
                shap_frames[a][["Feature", "Mean_SHAP"]],
                shap_frames[b][["Feature", "Mean_SHAP"]],
            )
            diff_df.to_excel(writer, sheet_name=f"diff_{a}_vs_{b}", index=False)

            title = f"Top {PLOT_TOP_N} SHAP mean differences: {a} - {b}"
            outfile = os.path.join(base_dir, f"shap_diff_{a}_vs_{b}.png")
            plot_top_diffs(diff_df, title, outfile)

    print(f"Wrote {OUTPUT_EXCEL} and pairwise plots to {base_dir}/")


if __name__ == "__main__":
    main()
