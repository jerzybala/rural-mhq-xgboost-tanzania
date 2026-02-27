"""
For each feature, find the group with the highest mean absolute SHAP (with AgeOfFirstSP/Smartphone.ownership included).
Outputs a CSV and a bar plot colored by the winning group.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

GROUP_DIRS: Dict[str, str] = {
    "rural_hadza": "rural_hadza",
    "gm": "gm",
    "age_18_24": "age_18_24",
}

RESULTS_FILENAME = "one_hot_encoding_results.pkl"
OUTPUT_CSV = "max_impact_per_feature.csv"
OUTPUT_PNG = "max_impact_per_feature.png"
TOP_N = 40

sns.set_style("whitegrid")


def load_shap_mean(group_key: str, base_dir: str = ".") -> pd.DataFrame:
    path = os.path.join(base_dir, GROUP_DIRS[group_key], RESULTS_FILENAME)
    with open(path, "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res["shap_importance"]).copy()
    df = df.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    df["Group"] = group_key
    return df[["Group", "Feature", "Mean_SHAP"]]


def build_max_table(shap_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(shap_frames.values(), axis=0, ignore_index=True)
    pivot = all_df.pivot_table(index="Feature", columns="Group", values="Mean_SHAP", fill_value=0)
    best_group = pivot.idxmax(axis=1)
    best_value = pivot.max(axis=1)
    second_value = pivot.apply(lambda r: r.nlargest(2).iloc[-1] if (r != 0).any() else 0, axis=1)
    gap = best_value - second_value
    out = pd.DataFrame({
        "Feature": pivot.index,
        "BestGroup": best_group,
        "BestMeanSHAP": best_value,
        "SecondBestMeanSHAP": second_value,
        "Gap": gap,
    }).sort_values("BestMeanSHAP", ascending=False)
    return out


def plot_max(df: pd.DataFrame, outfile: str, top_n: int):
    df_top = df.head(top_n)
    palette = {
        "rural_hadza": "#4CAF50",
        "gm": "#2196F3",
        "age_18_24": "#FF9800",
    }
    plt.figure(figsize=(10, 12))
    ax = sns.barplot(data=df_top, y="Feature", x="BestMeanSHAP", hue="BestGroup", palette=palette, orient="h")
    plt.title(f"Per-feature highest mean SHAP — top {top_n}")
    plt.xlabel("Mean SHAP (abs)")
    plt.ylabel("Feature")
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=9)
    plt.legend(title="Best group", fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = "."
    shap_frames = {k: load_shap_mean(k, base_dir) for k in GROUP_DIRS}
    max_table = build_max_table(shap_frames)
    max_table.to_csv(os.path.join(base_dir, OUTPUT_CSV), index=False)
    plot_max(max_table, os.path.join(base_dir, OUTPUT_PNG), TOP_N)
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_PNG} to {base_dir}/")


if __name__ == "__main__":
    main()
