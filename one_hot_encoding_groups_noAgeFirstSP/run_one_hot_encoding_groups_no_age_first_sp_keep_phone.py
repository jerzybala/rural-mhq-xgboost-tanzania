"""
One-hot XGBoost group runs excluding AgeOfFirstSP/AgeOfFirstSmartPhone but KEEPING Smartphone.ownership.
Outputs land in one_hot_encoding_groups_noAgeFirstSP_keepPhone/<group_name>/ with _noAgeSP_keepPhone suffixes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["text.usetex"] = False
plt.rcParams["mathtext.default"] = "regular"

DATA_PATH = "../rural_gmdata_forML.csv"
OUTPUT_ROOT = "./one_hot_encoding_groups_noAgeFirstSP_keepPhone"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Exclude AgeOfFirstSP / AgeOfFirstSmartPhone but KEEP Smartphone.ownership
BASE_FEATURES: List[str] = [
    "Education_Years",
    "ShareHomeWith",
    "RelationWithAdultFamily",
    "UPF.Freq",
    "Smartphone.ownership",
    "Exercise.Freq",
]


@dataclass
class Job:
    name: str
    group_filter: Optional[Iterable[str]]
    age_filter: Optional[Iterable[str]]


def _tags(job_name: str) -> Tuple[str, str]:
    title_tag = f"{job_name} (no AgeOfFirstSP + keep phone)"
    file_tag = f"{job_name}_noAgeSP_keepPhone".replace("/", "_").replace(" ", "_")
    return title_tag, file_tag


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
    cat_features = [col for col in features if col not in num_features]

    X = df[features].copy()
    y = df["Target"].copy()

    X[num_features] = X[num_features].fillna(X[num_features].median())
    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str)

    X_encoded = pd.get_dummies(X, columns=cat_features, drop_first=False)
    return X_encoded, y, num_features, cat_features


def train_and_evaluate(X, y):
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

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )

    metrics = {
        "roc_auc_test": roc_auc,
        "cv_mean_roc_auc": float(np.mean(cv_scores)),
        "cv_std_roc_auc": float(np.std(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_test_size": int(len(y_test)),
    }

    split = {
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cm": cm,
    }

    return model, metrics, split


def compute_feature_importance(model, X):
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    return fi


def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_SHAP": np.abs(shap_values).mean(axis=0),
    }).sort_values("Mean_SHAP", ascending=False)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 else expected_value[0]
    return explainer, shap_values, shap_importance, expected_value


def plot_and_save(job_dir, job_name, feature_importance, shap_values, shap_importance, explainer, expected_value, X_test, y_test, cm, y_prob, roc_auc):
    os.makedirs(job_dir, exist_ok=True)

    title_tag, file_tag = _tags(job_name)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"XGBoost Feature Importance (One-Hot) - {title_tag}")
    plt.gca().tick_params(axis="y", labelsize=7)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, f"xgboost_feature_importance_{file_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Skip SHAP summary and dependence plots in this quick variant to avoid slow rendering.

    # Skip per-sample force plots to avoid slow PNG rendering.

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Struggling", "Succeeding"], yticklabels=["Struggling", "Succeeding"])
    plt.title(f"Confusion Matrix (One-Hot) - {title_tag}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, f"confusion_matrix_{file_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Succeeding=1, One-Hot) - {title_tag}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, f"roc_curve_{file_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)

    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2)
    plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2)
    plt.plot(thresholds, f1, label="F1", linewidth=2)
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.ylim([0.0, 1.05])
    plt.title(f"Threshold vs Precision/Recall/F1 (Succeeding=1) - {title_tag}")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, f"threshold_metrics_{file_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(job_dir: str, model, feature_importance, shap_importance, metrics):
    import pickle

    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "xgboost_model_onehot_noAgeSP_keepPhone.pkl"), "wb") as f:
        pickle.dump(model, f)

    results = {
        "feature_importance": feature_importance.to_dict(),
        "shap_importance": shap_importance.to_dict(),
        "roc_auc": metrics["roc_auc_test"],
        "cv_scores": metrics["cv_scores"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }

    with open(os.path.join(job_dir, "one_hot_encoding_results_noAgeSP_keepPhone.pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join(job_dir, "metrics_noAgeSP_keepPhone.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def run_job(job: Job):
    print(f"\n=== Running job: {job.name} (no AgeOfFirstSP, keep phone) ===")
    df = load_and_filter(job)
    print(f"Rows after filtering: {len(df)}")
    if df.empty:
        print("No data after filtering; skipping.")
        return

    X, y, num_features, cat_features = prepare_features(df)
    print(f"Numeric features: {num_features} | One-hot categorical: {len([c for c in X.columns if c not in num_features])} encoded columns")

    model, metrics, split = train_and_evaluate(X, y)

    feature_importance = compute_feature_importance(model, X)
    explainer, shap_values, shap_importance, expected_value = compute_shap(model, split["X_test"])

    job_dir = os.path.join(OUTPUT_ROOT, job.name)
    plot_and_save(
        job_dir,
        job.name,
        feature_importance,
        shap_values,
        shap_importance,
        explainer,
        expected_value,
        split["X_test"],
        split["y_test"],
        split["cm"],
        split["y_prob"],
        metrics["roc_auc_test"],
    )

    save_outputs(job_dir, model, feature_importance, shap_importance, metrics)
    print(f"Saved outputs to {job_dir}/")
    print(f"ROC-AUC (test): {metrics['roc_auc_test']:.3f} | CV mean ROC-AUC: {metrics['cv_mean_roc_auc']:.3f}")


def main():
    jobs = [
        Job(name="rural_hadza", group_filter=["Rural.Hadza"], age_filter=None),
        Job(name="gm", group_filter=["GM"], age_filter=None),
        Job(
            name="age_18_24",
            group_filter=["Rural.Hadza", "GM"],
            age_filter=["16-20", "21-24"],
        ),
    ]

    for job in jobs:
        run_job(job)


if __name__ == "__main__":
    main()
