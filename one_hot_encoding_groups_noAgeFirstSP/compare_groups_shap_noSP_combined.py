"""
Combined SHAP mean difference plot across all group pairs (no AgeOfFirstSP/Smartphone.ownership).
Generates a single PNG showing per-feature differences for all pairs.
"""

from __future__ import annotations

import itertools
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GROUP_DIRS: Dict[str, str] = {
    "rural_hadza": "rural_hadza",
    "gm": "gm",
    "age_18_24": "age_18_24",
}

RESULTS_FILENAME = "one_hot_encoding_results_noSP.pkl"
PLOT_TOP_N = 25
COMBINED_PNG = "shap_diff_combined_noSP.png"
COMBINED_CSV = "shap_diff_combined_noSP.csv"

sns.set_style("whitegrid")


def load_shap_mean(group_key: str, base_dir: str = ".") -> pd.DataFrame:
    path = os.path.join(base_dir, GROUP_DIRS[group_key], RESULTS_FILENAME)
    with open(path, "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res["shap_importance"]).copy()
    df = df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    df["Group"] = group_key
    return df[["Group", "Feature", "Mean_SHAP"]]


def pairwise_long(shap_means: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for a, b in itertools.combinations(shap_means.keys(), 2):
        left = shap_means[a][["Feature", "Mean_SHAP"]].rename(columns={"Mean_SHAP": "Mean_SHAP_a"})
        right = shap_means[b][["Feature", "Mean_SHAP"]].rename(columns={"Mean_SHAP": "Mean_SHAP_b"})
        merged = pd.merge(left, right, on="Feature", how="outer").fillna(0)
        merged["diff"] = merged["Mean_SHAP_a"] - merged["Mean_SHAP_b"]
        merged["pair"] = f"{a} - {b}"
        rows.append(merged[["Feature", "pair", "diff"]])
    return pd.concat(rows, ignore_index=True)


def plot_combined(df: pd.DataFrame, outfile: str, top_n: int):
    # Select top features by max abs diff across pairs
    agg = df.copy()
    agg["abs_diff"] = agg["diff"].abs()
    top_features = (
        agg.groupby("Feature")["abs_diff"].max().sort_values(ascending=False).head(top_n).index.tolist()
    )
    df_top = df[df["Feature"].isin(top_features)]

    plt.figure(figsize=(10, 11))
    ax = sns.barplot(data=df_top, y="Feature", x="diff", hue="pair", palette="Set2", orient="h")
    ax.axvline(0, color="black", linewidth=1)
    for i in range(len(df_top["Feature"].unique())):
        ax.axhline(i + 0.5, color="#cccccc", linewidth=0.5, zorder=0)
    plt.title(f"SHAP mean differences across groups (noSP) — top {top_n} features")
    plt.ylabel("Feature")
    plt.xlabel("Mean SHAP difference (A - B)")
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=9)
    plt.legend(title="Pair", fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = "."
    shap_means = {k: load_shap_mean(k, base_dir) for k in GROUP_DIRS}
    long_df = pairwise_long(shap_means)

    # Save combined CSV
    long_df.to_csv(os.path.join(base_dir, COMBINED_CSV), index=False)

    # Plot combined
    plot_combined(long_df, os.path.join(base_dir, COMBINED_PNG), PLOT_TOP_N)

    print(f"Wrote {COMBINED_CSV} and {COMBINED_PNG} to {base_dir}/")


if __name__ == "__main__":
    main()
