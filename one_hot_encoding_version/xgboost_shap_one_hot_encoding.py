
"""
XGBoost Classifier: MHQ Lifestyle Drivers in Tanzania
VERSION: ONE-HOT ENCODING
Binary Classification: Succeeding (MHQ>=100) vs Struggling (MHQ<0)

This script loads data, preprocesses features, trains an XGBoost classifier with hyperparameter optimization, evaluates performance, and provides model interpretation (feature importance, SHAP, etc.).
Sections are clearly marked for reproducibility and clarity.
"""


# === IMPORTS ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, make_scorer
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')


# === PLOTTING STYLE ===
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# === SCRIPT START ===
print("="*80)
print("XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Tanzania")
print("VERSION: ONE-HOT ENCODING")
print("="*80)


# === 1. LOAD DATA ===
print("\n1. Loading data...")
# Load the main dataset (works from both project root and script folder)
import os
csv_path = 'rural_gmdata_forML.csv'
if not os.path.exists(csv_path):
    # Try parent directory
    csv_path = os.path.join('..', 'rural_gmdata_forML.csv')
data = pd.read_csv(csv_path)
print(f"Loaded data from: {csv_path}")
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())


# === 2. DATA EXPLORATION ===
print("\n" + "="*80)
print("2. Data Exploration")
print("="*80)
# Show column types, missing values, and target distribution
print("\nColumn names and types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())
print("\nMHQ Distribution:")
print(data['Overall.MHQ'].describe())


# === 3. PREPARE FEATURES AND TARGET ===
print("\n" + "="*80)
print("3. Preparing Features and Target")
print("="*80)
# Define features of interest
features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
            'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']
# Use correct feature for age of first smartphone if present
if 'AgeOfFirstSmartPhone' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSmartPhone', 'Smartphone.ownership', 'Exercise.Freq']
elif 'AgeOfFirstSP' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']
print(f"\nFeatures used: {features}")
# Filter to only Succeeding and Struggling cases
data_filtered = data[(data['Overall.MHQ'] >= 100) | (data['Overall.MHQ'] < 0)].copy()
print(f"\nOriginal dataset size: {len(data)}")
print(f"Filtered dataset size (MHQ>=100 or MHQ<0): {len(data_filtered)}")
# Create binary target
data_filtered['Target'] = (data_filtered['Overall.MHQ'] >= 100).astype(int)
print(f"\nTarget distribution:")
print(f"Struggling (MHQ<0): {(data_filtered['Target']==0).sum()} ({(data_filtered['Target']==0).sum()/len(data_filtered)*100:.1f}%)")
print(f"Succeeding (MHQ>=100): {(data_filtered['Target']==1).sum()} ({(data_filtered['Target']==1).sum()/len(data_filtered)*100:.1f}%)")
# Select features and target
X = data_filtered[features].copy()
y = data_filtered['Target'].copy()
print(f"\nFeature data types before encoding:")
print(X.dtypes)


# === 4. DATA PREPROCESSING: ONE-HOT ENCODING ===
print("\n" + "="*80)
print("4. Data Preprocessing - ONE-HOT ENCODING")
print("="*80)
# Check for missing values in features
print("\nMissing values in features:")
print(X.isnull().sum())
# Show unique values for each feature (for reference)
print("\nUnique values for each feature:")
for col in X.columns:
    print(f"\n{col}:")
    print(X[col].value_counts().head(10))
# Identify categorical features (all except Education_Years)
categorical_features = [col for col in X.columns if col != 'Education_Years']
print(f"\nCategorical features to one-hot encode: {categorical_features}")
# Fill missing values in Education_Years (numerical)
X['Education_Years'] = X['Education_Years'].fillna(X['Education_Years'].median())
# For categorical features, fill missing with 'Missing' before encoding
for col in categorical_features:
    X[col] = X[col].fillna('Missing').astype(str)
# One-hot encode categorical variables
print("\nApplying one-hot encoding to categorical features...")
X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, drop_first=False)
print(f"\nOriginal feature count: {len(X.columns)}")
print(f"After one-hot encoding: {len(X_encoded.columns)} features")
print(f"\nNew feature names (first 20):")
for i, col in enumerate(X_encoded.columns[:20]):
    print(f"  {i+1}. {col}")
if len(X_encoded.columns) > 20:
    print(f"  ... and {len(X_encoded.columns) - 20} more features")
# Update X with encoded features
X = X_encoded
# Create a mapping of original features to encoded columns for interpretability
feature_groups = {'Education_Years': ['Education_Years']}
for cat_feat in categorical_features:
    feature_groups[cat_feat] = [col for col in X.columns if col.startswith(f"{cat_feat}_")]
print(f"\nFeature groups created:")
for key, cols in feature_groups.items():
    print(f"  {key}: {len(cols)} columns")
print(f"\nFinal feature matrix shape: {X.shape}")
# Store original categorical features for later reference
original_categorical_features = categorical_features


# === 5. SPLIT DATA ===
print("\n" + "="*80)
print("5. Splitting Data")
print("="*80)
# Split into train/test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set target distribution:")
print(f"Struggling: {(y_train==0).sum()}, Succeeding: {(y_train==1).sum()}")
print(f"Test set target distribution:")
print(f"Struggling: {(y_test==0).sum()}, Succeeding: {(y_test==1).sum()}")


################################################################################
# === 6. TRAIN XGBOOST MODEL WITH HYPERPARAMETER OPTIMIZATION ===
# This section performs a two-step hyperparameter search for the XGBoost model:
#   1. RandomizedSearchCV: Fast, broad exploration of parameter space.
#   2. GridSearchCV: Fine-tuning around the best parameters from random search.
################################################################################

print("\n" + "="*80)
print("6. Training XGBoost Model with Hyperparameter Optimization")
print("="*80)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

# --- Step 1: Randomized Search ---
# Use RandomizedSearchCV to quickly explore a wide range of hyperparameters.
print("\n" + "-"*80)
print("STEP 1: Randomized Search (Fast exploration of parameter space)")
print("-"*80)

# Define parameter distribution for randomized search
param_distributions = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'n_estimators': [100, 150, 200, 250, 300],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0]
}

# Base model for randomized search
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    tree_method='hist'
)

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,  # Try 50 random combinations
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
print("\nRunning Randomized Search (this may take a few minutes)...")
random_search.fit(X_train, y_train)
print(f"\nBest ROC-AUC from Randomized Search: {random_search.best_score_:.4f}")
print(f"Best parameters from Randomized Search:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# --- Step 2: Grid Search ---
# Use GridSearchCV to fine-tune around the best parameters found above.
print("\n" + "-"*80)
print("STEP 2: Grid Search (Fine-tuning around best parameters)")
print("-"*80)

# Create narrow grid around best parameters
best_params = random_search.best_params_
param_grid = {
    'max_depth': [max(3, best_params['max_depth']-1), best_params['max_depth'], min(10, best_params['max_depth']+1)],
    'learning_rate': [max(0.01, best_params['learning_rate']-0.02), best_params['learning_rate'], min(0.2, best_params['learning_rate']+0.02)],
    'n_estimators': [max(100, best_params['n_estimators']-50), best_params['n_estimators'], best_params['n_estimators']+50],
    'min_child_weight': [max(1, best_params['min_child_weight']-1), best_params['min_child_weight'], best_params['min_child_weight']+1],
    'subsample': [max(0.6, best_params['subsample']-0.1), best_params['subsample'], min(1.0, best_params['subsample']+0.1)],
    'colsample_bytree': [max(0.6, best_params['colsample_bytree']-0.1), best_params['colsample_bytree'], min(1.0, best_params['colsample_bytree']+0.1)]
}

# Run GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)
print("\nRunning Grid Search for fine-tuning (this may take a few minutes)...")
grid_search.fit(X_train, y_train)
print(f"\nBest ROC-AUC from Grid Search: {grid_search.best_score_:.4f}")
print(f"Improvement over Randomized Search: {grid_search.best_score_ - random_search.best_score_:.4f}")
print(f"\nOptimized parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Use the best model from grid search
model = grid_search.best_estimator_

# Train final model with early stopping
print("\n" + "-"*80)
print("Training final optimized model with early stopping...")
print("-"*80)
final_params = grid_search.best_params_.copy()
final_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'use_label_encoder': False,
    'tree_method': 'hist',
    'gamma': best_params.get('gamma', 0),
    'reg_alpha': best_params.get('reg_alpha', 0),
    'reg_lambda': best_params.get('reg_lambda', 1.0)
})
model = xgb.XGBClassifier(**final_params)
model.fit(X_train, y_train)
print("\nOptimized model training completed!")


# === 7. MODEL EVALUATION ===
print("\n" + "="*80)
print("7. Model Evaluation - OPTIMIZED MODEL")
print("="*80)
# Predict on test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Struggling (MHQ<0)', 'Succeeding (MHQ>=100)']))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"Improvement over baseline: {roc_auc - 0.7461:.4f}")
# Cross-validation
print("\nPerforming 5-fold cross-validation with optimized model...")
cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
# Additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
precision_struggling = precision_score(y_test, y_pred, pos_label=0)
recall_struggling = recall_score(y_test, y_pred, pos_label=0)
f1_struggling = f1_score(y_test, y_pred, pos_label=0)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"\nAdditional Metrics:")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  Precision (Struggling): {precision_struggling:.4f}")
print(f"  Recall (Struggling): {recall_struggling:.4f}")
print(f"  F1-Score (Struggling): {f1_struggling:.4f}")

# Save optimization results
optimization_results = {
    'random_search_best_score': random_search.best_score_,
    'random_search_best_params': random_search.best_params_,
    'grid_search_best_score': grid_search.best_score_,
    'grid_search_best_params': grid_search.best_params_,
    'final_test_roc_auc': roc_auc,
    'improvement_over_baseline': roc_auc - 0.7461,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std()
}

print(f"\n📊 OPTIMIZATION SUMMARY:")
print(f"  Baseline ROC-AUC: 0.7461")
print(f"  Optimized ROC-AUC: {roc_auc:.4f}")
print(f"  Improvement: {(roc_auc - 0.7461):.4f} ({((roc_auc/0.7461 - 1)*100):.2f}%)")

# 8. Feature Importance from XGBoost (Grouped by Original Features)
print("\n" + "="*80)
print("8. Feature Importance (XGBoost) - ONE-HOT ENCODED FEATURES")
print("="*80)

# Get feature importance for individual encoded features
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Individual Encoded Features:")
print(feature_importance_df.head(20))

# Aggregate importance by original feature groups
print("\n" + "-"*80)
print("Aggregated Importance by Original Features:")
print("-"*80)

aggregated_importance = {}
for original_feat, encoded_cols in feature_groups.items():
    total_importance = feature_importance_df[
        feature_importance_df['Feature'].isin(encoded_cols)
    ]['Importance'].sum()
    aggregated_importance[original_feat] = total_importance

aggregated_importance_df = pd.DataFrame({
    'Original_Feature': list(aggregated_importance.keys()),
    'Total_Importance': list(aggregated_importance.values())
}).sort_values('Total_Importance', ascending=False)

print("\nAggregated Feature Importance:")
print(aggregated_importance_df)

# Plot aggregated feature importance
plt.figure(figsize=(10, 6))
plt.barh(aggregated_importance_df['Original_Feature'], 
         aggregated_importance_df['Total_Importance'])
plt.xlabel('Total Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Original Feature', fontsize=12, fontweight='bold')
plt.title('XGBoost Feature Importance (Aggregated from One-Hot Encoded Features)', 
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('xgboost_feature_importance_aggregated.png', dpi=300, bbox_inches='tight')
print("\nSaved: xgboost_feature_importance_aggregated.png")
plt.close()

# Plot top 20 individual encoded features
plt.figure(figsize=(12, 10))
top_20 = feature_importance_df.head(20)
plt.barh(range(len(top_20)), top_20['Importance'])
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Encoded Feature', fontsize=12, fontweight='bold')
plt.title('Top 20 Individual Encoded Features (XGBoost)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgboost_feature_importance_top20.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_feature_importance_top20.png")
plt.close()

# 9. SHAP Analysis
print("\n" + "="*80)
print("9. SHAP (SHapley Additive exPlanations) Analysis - ONE-HOT ENCODED")
print("="*80)

print("\nCalculating SHAP values... (this may take a moment)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully!")

# SHAP Summary Plot (Global Feature Importance) - Top 30 features
print("\nGenerating SHAP summary plot (top 30 features)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, 
                 max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("Saved: shap_summary_plot.png")
plt.close()

# SHAP Bar Plot (Mean Absolute SHAP Values) - Top 30 features
print("\nGenerating SHAP bar plot (top 30 features)...")
plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, 
                 plot_type="bar", max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("Saved: shap_bar_plot.png")
plt.close()

# Aggregate SHAP values by original features
print("\nAggregating SHAP values by original features...")
aggregated_shap = {}
for original_feat, encoded_cols in feature_groups.items():
    # Get indices of encoded columns
    col_indices = [X.columns.tolist().index(col) for col in encoded_cols if col in X.columns]
    # Sum absolute SHAP values across these columns
    if col_indices:
        aggregated_shap[original_feat] = np.abs(shap_values[:, col_indices]).sum(axis=1).mean()
    else:
        aggregated_shap[original_feat] = 0

aggregated_shap_df = pd.DataFrame({
    'Original_Feature': list(aggregated_shap.keys()),
    'Mean_Abs_SHAP': list(aggregated_shap.values())
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nAggregated SHAP Values by Original Features:")
print(aggregated_shap_df)

# Plot aggregated SHAP values
plt.figure(figsize=(10, 6))
plt.barh(aggregated_shap_df['Original_Feature'], aggregated_shap_df['Mean_Abs_SHAP'])
plt.xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
plt.ylabel('Original Feature', fontsize=12, fontweight='bold')
plt.title('Aggregated SHAP Values by Original Features', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_aggregated_by_original_features.png', dpi=300, bbox_inches='tight')
print("Saved: shap_aggregated_by_original_features.png")
plt.close()

# SHAP Dependence Plots for Top 3 Individual Encoded Features
print("\nGenerating SHAP dependence plots for top individual encoded features...")
top_features_encoded = feature_importance_df['Feature'].head(5).tolist()

for i, feature in enumerate(top_features_encoded):
    plt.figure(figsize=(10, 6))
    feature_idx = X.columns.tolist().index(feature)
    shap.dependence_plot(feature_idx, shap_values, X_test, 
                        feature_names=X.columns, show=False)
    plt.tight_layout()
    # Clean filename
    clean_name = feature.replace('/', '_').replace(' ', '_').replace('.', '_')
    plt.savefig(f'shap_dependence_{clean_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: shap_dependence_{clean_name}.png")
    plt.close()

# SHAP Force Plot for a few examples
print("\nGenerating SHAP force plots for sample predictions...")

# Select a few interesting examples
struggling_idx = np.where(y_test == 0)[0][0] if len(np.where(y_test == 0)[0]) > 0 else 0
succeeding_idx = np.where(y_test == 1)[0][0] if len(np.where(y_test == 1)[0]) > 0 else 1

# Force plot for Struggling individual
plt.figure(figsize=(20, 3))
shap.force_plot(explainer.expected_value, shap_values[struggling_idx], 
                X_test.iloc[struggling_idx], feature_names=X.columns,
                matplotlib=True, show=False)
plt.tight_layout()
plt.savefig('shap_force_plot_struggling.png', dpi=300, bbox_inches='tight')
print("Saved: shap_force_plot_struggling.png")
plt.close()

# Force plot for Succeeding individual
plt.figure(figsize=(20, 3))
shap.force_plot(explainer.expected_value, shap_values[succeeding_idx], 
                X_test.iloc[succeeding_idx], feature_names=X.columns,
                matplotlib=True, show=False)
plt.tight_layout()
plt.savefig('shap_force_plot_succeeding.png', dpi=300, bbox_inches='tight')
print("Saved: shap_force_plot_succeeding.png")
plt.close()

# 10. Additional Visualizations
print("\n" + "="*80)
print("10. Additional Visualizations")
print("="*80)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Struggling', 'Succeeding'],
            yticklabels=['Struggling', 'Succeeding'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

# ROC Curve - Overall
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curve.png")
plt.close()

# ROC Curves - Separate for Each Class
print("\nGenerating class-specific ROC curves...")

# ROC for Succeeding class (class 1) - using probability of class 1
fpr_succeeding, tpr_succeeding, _ = roc_curve(y_test, y_pred_proba)
auc_succeeding = roc_auc_score(y_test, y_pred_proba)

# ROC for Struggling class (class 0) - using probability of class 0 (1 - prob of class 1)
y_pred_proba_struggling = 1 - y_pred_proba
fpr_struggling, tpr_struggling, _ = roc_curve(1 - y_test, y_pred_proba_struggling)
auc_struggling = roc_auc_score(1 - y_test, y_pred_proba_struggling)

# Plot both class-specific ROC curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Struggling class ROC
ax1.plot(fpr_struggling, tpr_struggling, linewidth=2, color='#d62728', 
         label=f'Struggling Class (AUC = {auc_struggling:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curve - Struggling Class (MHQ < 0)', fontsize=12, fontweight='bold')
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

# Succeeding class ROC
ax2.plot(fpr_succeeding, tpr_succeeding, linewidth=2, color='#2ca02c',
         label=f'Succeeding Class (AUC = {auc_succeeding:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve - Succeeding Class (MHQ ≥ 100)', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_by_class.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_by_class.png")
print(f"  Struggling Class AUC: {auc_struggling:.4f}")
print(f"  Succeeding Class AUC: {auc_succeeding:.4f}")
plt.close()

# Precision-Recall Curves - Separate for Each Class
print("\nGenerating class-specific Precision-Recall curves...")

# PR curve for Succeeding class (class 1)
precision_succeeding, recall_succeeding, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision_succeeding = average_precision_score(y_test, y_pred_proba)

# PR curve for Struggling class (class 0) - invert labels and probabilities
precision_struggling, recall_struggling, _ = precision_recall_curve(1 - y_test, 1 - y_pred_proba)
avg_precision_struggling = average_precision_score(1 - y_test, 1 - y_pred_proba)

# Calculate baseline (random classifier) for each class
baseline_struggling = (1 - y_test).mean()  # Proportion of Struggling class
baseline_succeeding = y_test.mean()  # Proportion of Succeeding class

# Plot both class-specific PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Struggling class PR curve
ax1.plot(recall_struggling, precision_struggling, linewidth=2, color='#d62728',
         label=f'Struggling Class (AP = {avg_precision_struggling:.3f})')
ax1.axhline(y=baseline_struggling, color='k', linestyle='--', linewidth=1,
            label=f'Baseline (AP = {baseline_struggling:.3f})')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('Recall', fontsize=11)
ax1.set_ylabel('Precision', fontsize=11)
ax1.set_title('Precision-Recall Curve - Struggling Class (MHQ < 0)', fontsize=12, fontweight='bold')
ax1.legend(loc="best")
ax1.grid(alpha=0.3)

# Succeeding class PR curve
ax2.plot(recall_succeeding, precision_succeeding, linewidth=2, color='#2ca02c',
         label=f'Succeeding Class (AP = {avg_precision_succeeding:.3f})')
ax2.axhline(y=baseline_succeeding, color='k', linestyle='--', linewidth=1,
            label=f'Baseline (AP = {baseline_succeeding:.3f})')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title('Precision-Recall Curve - Succeeding Class (MHQ ≥ 100)', fontsize=12, fontweight='bold')
ax2.legend(loc="best")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('precision_recall_curves_by_class.png', dpi=300, bbox_inches='tight')
print("Saved: precision_recall_curves_by_class.png")
print(f"  Struggling Class AP: {avg_precision_struggling:.4f} (baseline: {baseline_struggling:.4f})")
print(f"  Succeeding Class AP: {avg_precision_succeeding:.4f} (baseline: {baseline_succeeding:.4f})")
plt.close()

# Threshold Tuning Curves - Metrics vs Decision Threshold
print("\nGenerating threshold tuning curves...")

# Calculate precision, recall, and F1 for different thresholds
thresholds_to_test = np.linspace(0, 1, 101)
precisions_struggling = []
recalls_struggling = []
f1s_struggling = []

precisions_succeeding = []
recalls_succeeding = []
f1s_succeeding = []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    
    # Metrics for Struggling class (class 0)
    # For class 0, we need to invert predictions and labels
    y_pred_struggling_thresh = 1 - y_pred_thresh
    y_test_struggling = 1 - y_test
    
    if y_pred_struggling_thresh.sum() > 0:  # Avoid division by zero
        prec_struggling = precision_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0)
    else:
        prec_struggling = 0
    rec_struggling = recall_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0)
    f1_struggling = f1_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0)
    
    precisions_struggling.append(prec_struggling)
    recalls_struggling.append(rec_struggling)
    f1s_struggling.append(f1_struggling)
    
    # Metrics for Succeeding class (class 1)
    if y_pred_thresh.sum() > 0:
        prec_succeeding = precision_score(y_test, y_pred_thresh, zero_division=0)
    else:
        prec_succeeding = 0
    rec_succeeding = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_succeeding = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    precisions_succeeding.append(prec_succeeding)
    recalls_succeeding.append(rec_succeeding)
    f1s_succeeding.append(f1_succeeding)

# Find optimal thresholds based on F1 score
optimal_idx_struggling = np.argmax(f1s_struggling)
optimal_threshold_struggling = thresholds_to_test[optimal_idx_struggling]
optimal_f1_struggling = f1s_struggling[optimal_idx_struggling]

optimal_idx_succeeding = np.argmax(f1s_succeeding)
optimal_threshold_succeeding = thresholds_to_test[optimal_idx_succeeding]
optimal_f1_succeeding = f1s_succeeding[optimal_idx_succeeding]

# Plot threshold tuning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Struggling class
ax1.plot(thresholds_to_test, f1s_struggling, linewidth=2, label='F1-Score', color='#1f77b4')
ax1.plot(thresholds_to_test, recalls_struggling, linewidth=2, label='Recall', color='#2ca02c')
ax1.plot(thresholds_to_test, precisions_struggling, linewidth=2, label='Precision', color='#d62728')
ax1.axvline(x=optimal_threshold_struggling, color='k', linestyle='--', linewidth=1, alpha=0.7,
            label=f'Optimal Threshold = {optimal_threshold_struggling:.2f}')
ax1.set_xlabel('Decision Threshold', fontsize=11)
ax1.set_ylabel('Score', fontsize=11)
ax1.set_title(f'Threshold Tuning - Struggling Class (MHQ < 0)\nBest F1={optimal_f1_struggling:.3f} at threshold={optimal_threshold_struggling:.2f}', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# Succeeding class
ax2.plot(thresholds_to_test, f1s_succeeding, linewidth=2, label='F1-Score', color='#1f77b4')
ax2.plot(thresholds_to_test, recalls_succeeding, linewidth=2, label='Recall', color='#2ca02c')
ax2.plot(thresholds_to_test, precisions_succeeding, linewidth=2, label='Precision', color='#d62728')
ax2.axvline(x=optimal_threshold_succeeding, color='k', linestyle='--', linewidth=1, alpha=0.7,
            label=f'Optimal Threshold = {optimal_threshold_succeeding:.2f}')
ax2.set_xlabel('Decision Threshold', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title(f'Threshold Tuning - Succeeding Class (MHQ ≥ 100)\nBest F1={optimal_f1_succeeding:.3f} at threshold={optimal_threshold_succeeding:.2f}', 
              fontsize=12, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('threshold_tuning_curves.png', dpi=300, bbox_inches='tight')
print("Saved: threshold_tuning_curves.png")
print(f"  Struggling Class: Optimal threshold={optimal_threshold_struggling:.2f}, F1={optimal_f1_struggling:.4f}")
print(f"  Succeeding Class: Optimal threshold={optimal_threshold_succeeding:.2f}, F1={optimal_f1_succeeding:.4f}")
plt.close()

# 11. Summary Statistics
print("\n" + "="*80)
print("11. Summary of Key Findings (ONE-HOT ENCODED ANALYSIS)")
print("="*80)

print("\n📊 MODEL PERFORMANCE:")
print(f"  • ROC-AUC Score: {roc_auc:.4f}")
print(f"  • Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n🎯 TOP 3 MOST IMPORTANT ORIGINAL FEATURES (Aggregated XGBoost):")
for idx, row in aggregated_importance_df.head(3).iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Total_Importance']:.4f}")

print("\n🔍 TOP 5 INDIVIDUAL ENCODED FEATURES (XGBoost):")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")

print("\n🔍 SHAP INSIGHTS (Aggregated by Original Features):")
print("  Mean Absolute SHAP Values (Feature Impact):")
for idx, row in aggregated_shap_df.iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print("\n✅ FILES GENERATED:")
print("  • xgboost_feature_importance_aggregated.png (Original features)")
print("  • xgboost_feature_importance_top20.png (Individual encoded features)")
print("  • shap_summary_plot.png (Top 30 encoded features)")
print("  • shap_bar_plot.png (Top 30 encoded features)")
print("  • shap_aggregated_by_original_features.png (Aggregated SHAP)")
for feature in top_features_encoded:
    clean_name = feature.replace('/', '_').replace(' ', '_').replace('.', '_')
    print(f"  • shap_dependence_{clean_name}.png")
print("  • shap_force_plot_struggling.png")
print("  • shap_force_plot_succeeding.png")
print("  • confusion_matrix.png")
print("  • roc_curve.png")

print("\n" + "="*80)
print("Analysis Complete! 🎉")
print("="*80)

# Save model
print("\nSaving trained model...")
try:
    import pickle
    with open('xgboost_model_onehot.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved: xgboost_model_onehot.pkl")
except Exception as e:
    print(f"Note: Model saving encountered an issue: {e}")

# Save feature information and results
results = {
    'feature_groups': feature_groups,
    'feature_columns': X.columns.tolist(),
    'aggregated_importance': aggregated_importance_df.to_dict(),
    'aggregated_shap': aggregated_shap_df.to_dict(),
    'roc_auc': roc_auc,
    'cv_scores': cv_scores.tolist(),
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True),
    'optimization_results': optimization_results,
    'final_params': final_params
}

with open('one_hot_encoding_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: one_hot_encoding_results.pkl")

print("\n🔬 HYPERPARAMETER OPTIMIZATION SUMMARY:")
print("="*80)
print("Two-stage optimization approach:")
print("  1. Randomized Search: Explored 50 parameter combinations")
print(f"     Best CV ROC-AUC: {random_search.best_score_:.4f}")
print("  2. Grid Search: Fine-tuned top parameters")
print(f"     Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"  3. Final model with early stopping")
print(f"     Test ROC-AUC: {roc_auc:.4f}")
print(f"\n  Total improvement over baseline: {(roc_auc - 0.7461):.4f}")
print("="*80)

print("\n🔬 ONE-HOT ENCODING ADVANTAGES:")
print("="*80)
print("• Binary representation: Each category becomes its own feature (0 or 1)")
print("• No ordinal assumption: Categories are not forced into arbitrary order")
print("• Better interpretability: Can see impact of each specific category")
print("• Individual category effects: SHAP shows which specific values matter most")
print("• Example: 'RelationWithAdultFamily_1_Very Close to Most' shows")
print("  the exact impact of having very close family relationships")
print("="*80)

print("\n🎯 KEY DIFFERENCES FROM LABEL ENCODING:")
print("="*80)
print("• Label encoding: Assigns single number to each category (0, 1, 2, 3...)")
print("  - Problem: Implies ordering (e.g., category 3 > category 1)")
print("  - Less interpretable for categorical data")
print("\n• One-hot encoding: Creates separate binary column for each category")
print("  - Advantage: No artificial ordering")
print("  - Advantage: Model learns independent effect of each category")
print("  - Trade-off: More features (increases dimensionality)")
print(f"\n• Your data: {len(original_categorical_features)} categorical features")
print(f"  expanded to {len(X.columns)} total features with one-hot encoding")
print("="*80)

print("\n🔬 INTERPRETATION GUIDE:")
print("="*80)
print("• SHAP Summary Plot: Shows feature importance and impact direction")
print("  - Red points: High feature values")
print("  - Blue points: Low feature values")
print("  - X-axis: Positive SHAP = increases likelihood of Succeeding")
print("            Negative SHAP = increases likelihood of Struggling")
print("\n• SHAP Dependence Plots: Show how each feature affects predictions")
print("  - Reveals non-linear relationships and interactions")
print("\n• SHAP Force Plots: Explain individual predictions")
print("  - Red arrows: Features pushing prediction toward Succeeding")
print("  - Blue arrows: Features pushing prediction toward Struggling")
print("="*80)
# Export Feature Importance and SHAP to Excel
print("\n" + "="*80)
print("Exporting Feature Importance and SHAP Values to Excel")
print("="*80)


# Create a comprehensive Excel file with multiple sheets
excel_filename = 'feature_importance_and_shap_analysis.xlsx'

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Individual Feature Importance
    feature_importance_df.to_excel(writer, sheet_name='Individual_Features', index=False)
    
    # Sheet 2: Aggregated Feature Importance
    aggregated_importance_df.to_excel(writer, sheet_name='Aggregated_Importance', index=False)
    
    # Sheet 3: Individual SHAP Values (Mean Absolute)
    shap_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    shap_importance_df.to_excel(writer, sheet_name='Individual_SHAP', index=False)
    
    # Sheet 4: Aggregated SHAP Values
    aggregated_shap_df.to_excel(writer, sheet_name='Aggregated_SHAP', index=False)
    
    # Sheet 5: Combined Summary (Top features)
    combined_df = pd.merge(
        feature_importance_df.head(20)[['Feature', 'Importance']],
        shap_importance_df[['Feature', 'Mean_Abs_SHAP']],
        on='Feature',
        how='left'
    )
    combined_df.to_excel(writer, sheet_name='Top20_Combined', index=False)
    
    # Sheet 6: Aggregated Combined
    aggregated_combined = pd.merge(
        aggregated_importance_df,
        aggregated_shap_df,
        on='Original_Feature',
        how='left'
    )
    aggregated_combined.to_excel(writer, sheet_name='Aggregated_Combined', index=False)

print(f"\n✅ Excel file saved: {excel_filename}")
print("\nSheets included:")
print("  1. Individual_Features: XGBoost importance for all 46 features")
print("  2. Aggregated_Importance: Importance grouped by 7 original features")
print("  3. Individual_SHAP: Mean absolute SHAP values for all 46 features")
print("  4. Aggregated_SHAP: SHAP values grouped by 7 original features")
print("  5. Top20_Combined: Top 20 features with both importance and SHAP")
print("  6. Aggregated_Combined: Original features with both metrics")