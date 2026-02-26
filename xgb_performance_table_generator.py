
"""
Automated XGBoost Performance Table Generator
================================================
This script runs XGBoost classification for all combinations of group and variant in the MHQ dataset.
It performs preprocessing, 5-fold cross-validation, and exports summary and detailed results to Excel files.

Outputs:
    - xgb_performance_table.xlsx: Summary metrics (mean/std) for each group/variant
    - xgb_performance_table_per_fold.xlsx: All per-fold metrics
    - xgb_confusion_matrices.xlsx: Confusion matrices for each fold
    - xgb_feature_importances.xlsx: Feature importances (mean/std across folds)

Edit VARIANTS and GROUPS below to customize analysis scope.
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Settings
#+# === Imports ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
GROUPS = ['all', 'rural_hadza', 'gm', 'age_18_24']
#+# === Settings ===
VARIANTS = ['with_SP', 'no_SP']
GROUPS = ['all', 'rural_hadza', 'gm', 'age_18_24']
DATA_PATH = 'rural_gmdata_forML.csv'

#+# === Load Data ===
print("Loading data...")
data = pd.read_csv(DATA_PATH)

def filter_data(df, variant, group):
    """
    Filters the dataframe for the given variant and group.
    - Variant: 'with_SP' includes AgeOfFirstSP, 'no_SP' excludes it.
    - Group: filters by population group or age group.
    Returns filtered dataframe and feature list.
    """
    df = df.copy()
    # Select features based on variant
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 'UPF.Freq', 'Smartphone.ownership', 'Exercise.Freq']
    if variant == 'with_SP':
        if 'AgeOfFirstSP' in df.columns:
            features.insert(4, 'AgeOfFirstSP')
    # Filter by group
    if group != 'all':
        if group == 'age_18_24':
            # Include all rows where AgeGroup contains '18', '19', '20', '21', '22', '23', or '24'
            df = df[df['AgeGroup'].astype(str).str.contains(r'18|19|20|21|22|23|24', regex=True)]
        else:
            df = df[df['Group'].str.lower().str.replace('.', '_') == group]
    # Filter for binary target (Succeeding vs. Struggling)
    df = df[(df['Overall.MHQ'] >= 100) | (df['Overall.MHQ'] < 0)].copy()
    df['Target'] = (df['Overall.MHQ'] >= 100).astype(int)
    return df, features

results = []

CV_FOLDS = 5
CV_FOLDS = 5
per_fold_rows = []
confusion_rows = []
feature_importance_rows = []
confusion_rows = []
feature_importance_rows = []

for variant in VARIANTS:
    for group in GROUPS:
        print(f"\n=== Variant: {variant}, Group: {group} ===")
        df, features = filter_data(data, variant, group)
        if len(df) < 50:
            print("  Skipped (not enough data)")
            continue
        X = df[features].copy()
        y = df['Target'].copy()
        # Preprocessing
        categorical_features = [col for col in X.columns if col != 'Education_Years']
        X['Education_Years'] = X['Education_Years'].fillna(X['Education_Years'].median())
        for col in categorical_features:
            X[col] = X[col].fillna('Missing').astype(str)
        X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        aucs, accs, f1s, precs, recalls, bal_accs = [], [], [], [], [], []
        specs, mccs, kappas = [], [], []
        fold_feature_importances = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            # Additional metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            mcc = matthews_corrcoef(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            # Feature importances
            importances = model.feature_importances_
            feature_names = X.columns
            fold_feature_importances.append(importances)
            # Save per-fold metrics
            aucs.append(auc)
            accs.append(acc)
            f1s.append(f1)
            precs.append(prec)
            recalls.append(rec)
            bal_accs.append(bal_acc)
            specs.append(specificity)
            mccs.append(mcc)
            kappas.append(kappa)
            per_fold_rows.append({
                'Variant': variant,
                'Group': group,
                'Fold': fold,
                'AUC': auc,
                'Accuracy': acc,
                'F1': f1,
                'Precision': prec,
                'Recall': rec,
                'Specificity': specificity,
                'Balanced_Accuracy': bal_acc,
                'MCC': mcc,
                'Cohen_Kappa': kappa
            })
            # Save confusion matrix
            confusion_rows.append({
                'Variant': variant,
                'Group': group,
                'Fold': fold,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp
            })
            print(f"    Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}, Spec={specificity:.4f}, MCC={mcc:.4f}, Kappa={kappa:.4f}")
        # Aggregate feature importances
        fi_arr = np.array(fold_feature_importances)
        fi_mean = np.mean(fi_arr, axis=0)
        fi_std = np.std(fi_arr, axis=0)
        for fname, mean, std in zip(feature_names, fi_mean, fi_std):
            feature_importance_rows.append({
                'Variant': variant,
                'Group': group,
                'Feature': fname,
                'Importance_mean': mean,
                'Importance_std': std
            })
        metrics = {
            'Variant': variant,
            'Group': group,
            'Rows': len(df),
            'Positives': (df['Target'] == 1).sum(),
            'CV_Folds': CV_FOLDS,
            'AUC_mean': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'Accuracy_mean': np.mean(accs),
            'Accuracy_std': np.std(accs),
            'F1_mean': np.mean(f1s),
            'F1_std': np.std(f1s),
            'Precision_mean': np.mean(precs),
            'Precision_std': np.std(precs),
            'Recall_mean': np.mean(recalls),
            'Recall_std': np.std(recalls),
            'Specificity_mean': np.mean(specs),
            'Specificity_std': np.std(specs),
            'Balanced_Accuracy_mean': np.mean(bal_accs),
            'Balanced_Accuracy_std': np.std(bal_accs),
            'MCC_mean': np.mean(mccs),
            'MCC_std': np.std(mccs),
            'Cohen_Kappa_mean': np.mean(kappas),
            'Cohen_Kappa_std': np.std(kappas)
        }
        results.append(metrics)

# === Export Results ===
# Main summary table (mean/std metrics for each group/variant)
results_df = pd.DataFrame(results)
results_df.to_excel('xgb_performance_table.xlsx', index=False)
print("\nExported results to xgb_performance_table.xlsx")

# Per-fold metrics (all metrics for each fold)
per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_excel('xgb_performance_table_per_fold.xlsx', index=False)
print("Exported per-fold metrics to xgb_performance_table_per_fold.xlsx")

# Confusion matrices (TN, FP, FN, TP for each fold)
confusion_df = pd.DataFrame(confusion_rows)
confusion_df.to_excel('xgb_confusion_matrices.xlsx', index=False)
print("Exported confusion matrices to xgb_confusion_matrices.xlsx")

# Feature importances (mean/std across folds)
feature_importance_df = pd.DataFrame(feature_importance_rows)
feature_importance_df.to_excel('xgb_feature_importances.xlsx', index=False)
print("Exported feature importances to xgb_feature_importances.xlsx")
feature_importance_df = pd.DataFrame(feature_importance_rows)
feature_importance_df.to_excel('xgb_feature_importances.xlsx', index=False)
print("Exported feature importances to xgb_feature_importances.xlsx")
