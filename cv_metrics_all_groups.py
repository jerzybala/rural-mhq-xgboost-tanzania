"""
Compute cross-validation metrics (AUC, F1, Recall, Precision) for all groups
across both feature variants: with AgeOfFirstSP/Smartphone.ownership (with-SP)
and without them (no-SP). Writes an Excel file with mean/std per metric.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_PATH = "./rural_gmdata_forML.csv"
OUTPUT_EXCEL = "cv_metrics_all_groups.xlsx"
RANDOM_STATE = 42
CV_FOLDS = 5

WITH_SP_FEATURES: List[str] = [
    "Education_Years",
    "ShareHomeWith",
    "RelationWithAdultFamily",
    "UPF.Freq",
    "AgeOfFirstSP",
    "Smartphone.ownership",
    "Exercise.Freq",
]

NO_SP_FEATURES: List[str] = [
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


def prepare_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    feat = features.copy()
    # Switch to alternate column if present in the with-SP variant
    if "AgeOfFirstSP" in feat and "AgeOfFirstSmartPhone" in df.columns:
        feat = ["AgeOfFirstSmartPhone" if c == "AgeOfFirstSP" else c for c in feat]

    num_features = [c for c in feat if c == "Education_Years"]
    cat_features = [c for c in feat if c not in num_features]

    X = df[feat].copy()
    y = df["Target"].copy()

    X[num_features] = X[num_features].fillna(X[num_features].median())
    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str)

    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=False)
    return X_encoded, y


def build_model(base_scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=220,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=base_scale_pos_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def eval_job(job: Job, features: List[str], variant: str) -> dict:
    df = load_and_filter(job)
    if df.empty:
        raise ValueError(f"No data after filtering for job {job.name} ({variant})")

    X, y = prepare_features(df, features)
    # Use dataset-level ratio for scale_pos_weight; close to per-fold ratio and simpler for cross_validate
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = build_model(scale_pos_weight)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    return {
        "Variant": variant,
        "Group": job.name,
        "CV_Folds": CV_FOLDS,
        "AUC_mean": float(np.mean(res["test_auc"])),
        "AUC_std": float(np.std(res["test_auc"])),
        "Accuracy_mean": float(np.mean(res["test_accuracy"])),
        "Accuracy_std": float(np.std(res["test_accuracy"])),
        "F1_mean": float(np.mean(res["test_f1"])),
        "F1_std": float(np.std(res["test_f1"])),
        "Precision_mean": float(np.mean(res["test_precision"])),
        "Precision_std": float(np.std(res["test_precision"])),
        "Recall_mean": float(np.mean(res["test_recall"])),
        "Recall_std": float(np.std(res["test_recall"])),
        "Rows": int(len(df)),
        "Positives": int((df["Target"] == 1).sum()),
    }


def main():
    jobs = [
        Job(name="all", group_filter=None, age_filter=None),
        Job(name="rural_hadza", group_filter=["Rural.Hadza"], age_filter=None),
        Job(name="gm", group_filter=["GM"], age_filter=None),
        Job(name="age_18_24", group_filter=["Rural.Hadza", "GM"], age_filter=["16-20", "21-24"]),
    ]

    rows = []
    for variant, feats in (("with_SP", WITH_SP_FEATURES), ("no_SP", NO_SP_FEATURES)):
        for job in jobs:
            rows.append(eval_job(job, feats, variant))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "Variant",
            "Group",
            "CV_Folds",
            "Rows",
            "Positives",
            "AUC_mean",
            "AUC_std",
            "Accuracy_mean",
            "Accuracy_std",
            "F1_mean",
            "F1_std",
            "Precision_mean",
            "Precision_std",
            "Recall_mean",
            "Recall_std",
        ]
    ]

    mean_std_df = summary_df[
        [
            "Variant",
            "Group",
            "AUC_mean",
            "AUC_std",
            "Accuracy_mean",
            "Accuracy_std",
            "F1_mean",
            "F1_std",
            "Precision_mean",
            "Precision_std",
            "Recall_mean",
            "Recall_std",
        ]
    ]

    out_path = os.path.join(os.path.dirname(DATA_PATH), OUTPUT_EXCEL)
    with pd.ExcelWriter(out_path) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        mean_std_df.to_excel(writer, sheet_name="mean_std", index=False)
    print(f"Wrote CV metrics to {out_path}")


if __name__ == "__main__":
    main()
