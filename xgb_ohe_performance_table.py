"""
XGBoost + SHAP Analysis for MHQ Lifestyle Drivers (NEW DATA)
This script is adapted from xgboost_shap_one_hot_encoding.py to use the new data in performace_table.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("XGBoost + SHAP Analysis: NEW DATA")
print("="*80)

# 1. Load Data
print("\n1. Loading data...")
data = pd.read_csv('rural_gmdata_forML.csv')
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# 2. Prepare Features and Target
features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
            'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']
if 'AgeOfFirstSmartPhone' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSmartPhone', 'Smartphone.ownership', 'Exercise.Freq']
elif 'AgeOfFirstSP' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']

# Filter to Succeeding and Struggling
if 'Overall.MHQ' in data.columns:
    data_filtered = data[(data['Overall.MHQ'] >= 100) | (data['Overall.MHQ'] < 0)].copy()
    data_filtered['Target'] = (data_filtered['Overall.MHQ'] >= 100).astype(int)
else:
    raise ValueError('Overall.MHQ column not found in data.')

X = data_filtered[features].copy()
y = data_filtered['Target'].copy()

# 3. Data Preprocessing - One-hot encoding
categorical_features = [col for col in X.columns if col != 'Education_Years']
X['Education_Years'] = X['Education_Years'].fillna(X['Education_Years'].median())
for col in categorical_features:
    X[col] = X[col].fillna('Missing').astype(str)
X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, drop_first=False)
X = X_encoded

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train XGBoost Model
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

# 6. Model Evaluation
print("\nModel Evaluation:")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = np.mean(y_test == y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# 7. Feature Importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nTop 10 Features:")
print(feature_importance_df.head(10))

# 8. SHAP Analysis (optional, can be commented out if not needed)
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("\nSHAP analysis completed.")
except Exception as e:
    print(f"SHAP analysis skipped: {e}")

# 9. Save results
results = {
    'roc_auc': roc_auc,
    'accuracy': accuracy,
    'f1': f1,
    'precision': precision,
    'recall': recall,
    'balanced_accuracy': balanced_acc,
    'feature_importance': feature_importance_df
}
import pickle
with open('xgb_results_newdata.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved to xgb_results_newdata.pkl")
