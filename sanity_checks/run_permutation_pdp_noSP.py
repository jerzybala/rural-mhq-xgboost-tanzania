"""
Permutation importance + partial dependence for no-SP variant (excludes AgeOfFirstSP and Smartphone.ownership).
Outputs under sanity_checks/permutation_pdp/noSP/<group>/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.default"] = "regular"

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "rural_gmdata_forML.csv"
OUTPUT_ROOT = ROOT / "sanity_checks" / "permutation_pdp" / "noSP"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N = 5

# Exclude AgeOfFirstSP / AgeOfFirstSmartPhone and Smartphone.ownership
BASE_FEATURES: List[str] = [
    "Education_Years",
    "ShareHomeWith",
    "RelationWithAdultFamily",
    "UPF.Freq",
    "Exercise.Freq",
]


@dataclass
class Job:
    name: str
    group_filter: Optional[Iterable[str]]
    age_filter: Optional[Iterable[str]]


def load_and_filter(job: Job) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[(df["Overall.MHQ"] >= 100) | (df["Overall.MHQ"] < 0)].copy()
    df["Target"] = (df["Overall.MHQ"] >= 100).astype(int)
    if job.group_filter is not None:
        df = df[df["Group"].isin(job.group_filter)]
    if job.age_filter is not None:
        df = df[df["AgeGroup"].isin(job.age_filter)]
    return df


def prepare_features(df: pd.DataFrame):
    features = BASE_FEATURES.copy()
    num_features = ["Education_Years"]
    cat_features = [c for c in features if c not in num_features]
    X = df[features].copy()
    y = df["Target"].copy()
    X[num_features] = X[num_features].fillna(X[num_features].median())
    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str)
    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=False)
    return X_encoded, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=220,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def compute_perm_importance(model, X_test, y_test):
    r = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": r.importances_mean,
    }).sort_values("Importance", ascending=False)
    return df


def plot_perm(df: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(TOP_N), x="Importance", y="Feature", orient="h")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pdp(model, X_train, features: list[str], out_path: Path, title: str):
    fig, ax = plt.subplots(1, len(features), figsize=(4 * len(features), 3.5))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=features,
        grid_resolution=20,
        ax=ax,
        kind="average",
    )
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_job(job: Job):
    df = load_and_filter(job)
    if df.empty:
        print(f"{job.name}: no data after filtering; skip")
        return
    X, y = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    perm_df = compute_perm_importance(model, X_test, y_test)

    job_dir = OUTPUT_ROOT / job.name
    job_dir.mkdir(parents=True, exist_ok=True)

    perm_df.to_csv(job_dir / "permutation_importance.csv", index=False)
    plot_perm(perm_df, job_dir / "permutation_importance.png", f"Permutation importance (AUC) — {job.name} noSP")

    top_features = perm_df.head(3)["Feature"].tolist()
    plot_pdp(model, X_train, top_features, job_dir / "pdp_top3.png", f"Partial dependence — {job.name} noSP")
    print(f"{job.name}: done (perm + PDP)")


def main():
    jobs = [
        Job(name="rural_hadza", group_filter=["Rural.Hadza"], age_filter=None),
        Job(name="gm", group_filter=["GM"], age_filter=None),
        Job(name="age_18_24", group_filter=["Rural.Hadza", "GM"], age_filter=["16-20", "21-24"]),
    ]
    for job in jobs:
        run_job(job)


if __name__ == "__main__":
    main()
