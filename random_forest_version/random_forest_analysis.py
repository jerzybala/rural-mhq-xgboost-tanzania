"""
Random Forest + SHAP Analysis for MHQ Lifestyle Drivers in Tanzania
VERSION: Random Forest + ONE-HOT ENCODING
Binary Classification: Succeeding (MHQ>=100) vs Struggling (MHQ<0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("Random Forest + SHAP Analysis: Lifestyle Drivers of MHQ in Tanzania")
print("VERSION: Random Forest + ONE-HOT ENCODING")
print("="*80)

# 1. Load Data
print("\n1. Loading data...")
data = pd.read_csv('../rural_gmdata_forML.csv')
print(f"Dataset shape: {data.shape}")

# 2. Prepare Features and Target
print("\n" + "="*80)
print("2. Preparing Features and Target")
print("="*80)

# Define features
features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
            'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']

# Check if AgeOfFirstSP or AgeOfFirstSmartPhone exists
if 'AgeOfFirstSmartPhone' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSmartPhone', 'Smartphone.ownership', 'Exercise.Freq']
elif 'AgeOfFirstSP' in data.columns:
    features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
                'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']

print(f"Features used: {features}")

# Create binary target: Succeeding (MHQ>=100) vs Struggling (MHQ<0)
data_filtered = data[
    (data['Overall.MHQ'] >= 100) | (data['Overall.MHQ'] < 0)
].copy()

print(f"\nOriginal dataset size: {len(data)}")
print(f"Filtered dataset size (MHQ>=100 or MHQ<0): {len(data_filtered)}")

# Create binary target
data_filtered['Target'] = (data_filtered['Overall.MHQ'] >= 100).astype(int)
print(f"\nTarget distribution:")
print(f"Struggling (MHQ<0): {(data_filtered['Target']==0).sum()} ({(data_filtered['Target']==0).sum()/len(data_filtered)*100:.1f}%)")
print(f"Succeeding (MHQ>=100): {(data_filtered['Target']==1).sum()} ({(data_filtered['Target']==1).sum()/len(data_filtered)*100:.1f}%)")

# Select features
X = data_filtered[features].copy()
y = data_filtered['Target'].copy()

# 3. Data Preprocessing - ONE-HOT ENCODING
print("\n" + "="*80)
print("3. Data Preprocessing - ONE-HOT ENCODING")
print("="*80)

# Handle missing values in Education_Years (numerical)
X['Education_Years'] = X['Education_Years'].fillna(X['Education_Years'].median())

# Identify categorical features
categorical_features = [col for col in X.columns if col != 'Education_Years']

# For categorical features, fill missing with 'Missing' category before encoding
for col in categorical_features:
    X[col] = X[col].fillna('Missing').astype(str)

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, 
                           drop_first=False)

print(f"\nOriginal feature count: {len(X.columns)}")
print(f"After one-hot encoding: {len(X_encoded.columns)} features")

# Create a mapping of original features to encoded columns for interpretability
feature_groups = {'Education_Years': ['Education_Years']}
for cat_feat in categorical_features:
    feature_groups[cat_feat] = [col for col in X_encoded.columns if col.startswith(f"{cat_feat}_")]

print(f"\nFeature groups:")
for key, cols in feature_groups.items():
    print(f"  {key}: {len(cols)} columns")

X = X_encoded

# 4. Split Data
print("\n" + "="*80)
print("4. Splitting Data")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set target distribution:")
print(f"Struggling: {(y_train==0).sum()}, Succeeding: {(y_train==1).sum()}")
print(f"Test set target distribution:")
print(f"Struggling: {(y_test==0).sum()}, Succeeding: {(y_test==1).sum()}")

# 5. Train Random Forest Model
print("\n" + "="*80)
print("5. Training Random Forest Model")
print("="*80)

# Calculate class weights for imbalance
class_weight_ratio = (y_train==1).sum() / (y_train==0).sum()
print(f"\nClass imbalance ratio: {class_weight_ratio:.2f}")

print("\n🌲 WHAT IS RANDOM FOREST?")
print("-" * 80)
print("Random Forest is an ensemble of decision trees.")
print("Advantages:")
print("  • Robust to overfitting (averages many trees)")
print("  • Handles non-linear relationships well")
print("  • Works well with categorical features")
print("  • Built-in feature importance")
print("  • Less sensitive to hyperparameters than boosting")
print("  • Naturally handles class imbalance with class_weight='balanced'")
print("-" * 80)

# Train Random Forest with balanced class weights
model = RandomForestClassifier(
    n_estimators=200,        # More trees = more stable
    max_depth=10,            # Reasonable depth
    min_samples_split=10,    # Prevent overfitting
    min_samples_leaf=5,      # Prevent overfitting
    max_features='sqrt',     # Use sqrt(n_features) per tree
    class_weight='balanced', # Handle class imbalance automatically
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbose=0
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Model training completed!")

# 6. Model Evaluation
print("\n" + "="*80)
print("6. Model Evaluation - Random Forest")
print("="*80)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report:")
target_names = ['Struggling (MHQ<0)', 'Succeeding (MHQ>=100)']
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Additional metrics
balanced_acc = balanced_accuracy_score(y_test, y_pred)
precision_struggling = precision_score(y_test, y_pred, pos_label=0)
recall_struggling = recall_score(y_test, y_pred, pos_label=0)
f1_struggling = f1_score(y_test, y_pred, pos_label=0)

print(f"\nAdditional Metrics:")
print(f"  Balanced Accuracy: {balanced_acc:.4f}")
print(f"  Precision (Struggling): {precision_struggling:.4f}")
print(f"  Recall (Struggling): {recall_struggling:.4f}")
print(f"  F1-Score (Struggling): {f1_struggling:.4f}")

# 7. Feature Importance (Random Forest)
print("\n" + "="*80)
print("7. Feature Importance (Random Forest)")
print("="*80)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Individual Encoded Features:")
print(feature_importance.head(20).to_string(index=False))

# Aggregate importance by original features
aggregated_importance = {}
for orig_feat, encoded_cols in feature_groups.items():
    total_importance = feature_importance[
        feature_importance['Feature'].isin(encoded_cols)
    ]['Importance'].sum()
    aggregated_importance[orig_feat] = total_importance

aggregated_df = pd.DataFrame({
    'Original_Feature': aggregated_importance.keys(),
    'Total_Importance': aggregated_importance.values()
}).sort_values('Total_Importance', ascending=False)

print("\n" + "-"*80)
print("Aggregated Importance by Original Features:")
print("-"*80)
print("\nAggregated Feature Importance:")
print(aggregated_df.to_string(index=False))

# Visualization: Aggregated importance
plt.figure(figsize=(10, 6))
plt.barh(aggregated_df['Original_Feature'], aggregated_df['Total_Importance'])
plt.xlabel('Total Importance')
plt.title('Random Forest Feature Importance (Aggregated)')
plt.tight_layout()
plt.savefig('random_forest_feature_importance_aggregated.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: random_forest_feature_importance_aggregated.png")

# Visualization: Top 20 individual features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Individual Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('random_forest_feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: random_forest_feature_importance_top20.png")

# 8. SHAP Analysis
print("\n" + "="*80)
print("8. SHAP Analysis - Random Forest")
print("="*80)

print("\nCalculating SHAP values... (this may take a few minutes for Random Forest)")
# Use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model)
# Sample 500 test instances for faster SHAP computation
sample_size = min(500, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Random Forest returns SHAP values for both classes
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Class 1 (Succeeding)

print("SHAP values calculated successfully!")

# SHAP Summary Plot (top 30 features)
print("\nGenerating SHAP summary plot (top 30 features)...")
# Handle 3D SHAP values
if len(shap_values.shape) == 3:
    shap_values_plot = shap_values[:, :, 1]
else:
    shap_values_plot = shap_values

plt.figure()
shap.summary_plot(shap_values_plot, X_test_sample, max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_summary_plot_rf.png")

# SHAP Bar Plot
print("\nGenerating SHAP bar plot (top 30 features)...")
plt.figure()
shap.summary_plot(shap_values_plot, X_test_sample, plot_type='bar', max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_bar_plot_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_bar_plot_rf.png")

# Aggregate SHAP values by original features
print("\nAggregating SHAP values by original features...")
# Handle 3D SHAP values array for binary classification
if len(shap_values.shape) == 3:
    # Use SHAP values for class 1 (Succeeding)
    shap_values_class1 = shap_values[:, :, 1]
else:
    shap_values_class1 = shap_values

shap_df = pd.DataFrame(shap_values_class1, columns=X_test_sample.columns)
aggregated_shap = {}

for orig_feat, encoded_cols in feature_groups.items():
    relevant_cols = [col for col in encoded_cols if col in shap_df.columns]
    if relevant_cols:
        aggregated_shap[orig_feat] = shap_df[relevant_cols].abs().mean().sum()

aggregated_shap_df = pd.DataFrame({
    'Original_Feature': aggregated_shap.keys(),
    'Mean_Abs_SHAP': aggregated_shap.values()
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nAggregated SHAP Values by Original Features:")
print(aggregated_shap_df.to_string(index=False))

# Visualization: Aggregated SHAP
plt.figure(figsize=(10, 6))
plt.barh(aggregated_shap_df['Original_Feature'], aggregated_shap_df['Mean_Abs_SHAP'])
plt.xlabel('Mean Absolute SHAP Value')
plt.title('SHAP Values (Aggregated by Original Features) - Random Forest')
plt.tight_layout()
plt.savefig('shap_aggregated_by_original_features_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_aggregated_by_original_features_rf.png")

# SHAP Dependence Plots for top features
print("\nGenerating SHAP dependence plots for top individual encoded features...")
top_5_features = feature_importance.head(5)['Feature'].values

for feat in top_5_features:
    if feat in X_test_sample.columns:
        plt.figure()
        shap.dependence_plot(feat, shap_values_plot, X_test_sample, show=False)
        plt.tight_layout()
        safe_feat_name = feat.replace('/', '_').replace(' ', '_').replace('.', '_')
        plt.savefig(f'shap_dependence_{safe_feat_name}_rf.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: shap_dependence_{safe_feat_name}_rf.png")

# SHAP Force Plots
print("\nGenerating SHAP force plots for sample predictions...")
# Sample struggling case
y_test_sample = y_test.loc[X_test_sample.index]
struggling_idx = np.where(y_test_sample == 0)[0][0]
plt.figure()
expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
shap.force_plot(expected_val, 
                shap_values_plot[struggling_idx], 
                X_test_sample.iloc[struggling_idx], matplotlib=True, show=False)
plt.savefig('shap_force_plot_struggling_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_force_plot_struggling_rf.png")

# Sample succeeding case
succeeding_idx = np.where(y_test_sample == 1)[0][0]
plt.figure()
shap.force_plot(expected_val,
                shap_values_plot[succeeding_idx], 
                X_test_sample.iloc[succeeding_idx], matplotlib=True, show=False)
plt.savefig('shap_force_plot_succeeding_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_force_plot_succeeding_rf.png")
plt.close()
print("Saved: shap_force_plot_succeeding_rf.png")

# 9. Additional Visualizations
print("\n" + "="*80)
print("9. Additional Visualizations")
print("="*80)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_rf.png")

# ROC Curve - Overall
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve_rf.png")

# ROC Curves - Separate for Each Class
print("\nGenerating class-specific ROC curves...")

# ROC for Succeeding class (class 1)
fpr_succeeding, tpr_succeeding, _ = roc_curve(y_test, y_pred_proba)
auc_succeeding = roc_auc_score(y_test, y_pred_proba)

# ROC for Struggling class (class 0)
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
plt.savefig('roc_curves_by_class_rf.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_by_class_rf.png")
print(f"  Struggling Class AUC: {auc_struggling:.4f}")
print(f"  Succeeding Class AUC: {auc_succeeding:.4f}")
plt.close()

# Precision-Recall Curves - Separate for Each Class
print("\nGenerating class-specific Precision-Recall curves...")

# PR curve for Succeeding class (class 1)
precision_succeeding, recall_succeeding, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision_succeeding = average_precision_score(y_test, y_pred_proba)

# PR curve for Struggling class (class 0)
precision_struggling, recall_struggling, _ = precision_recall_curve(1 - y_test, 1 - y_pred_proba)
avg_precision_struggling = average_precision_score(1 - y_test, 1 - y_pred_proba)

# Baseline
baseline_struggling = (1 - y_test).mean()
baseline_succeeding = y_test.mean()

# Plot both class-specific PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Struggling class
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

# Succeeding class
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
plt.savefig('precision_recall_curves_by_class_rf.png', dpi=300, bbox_inches='tight')
print("Saved: precision_recall_curves_by_class_rf.png")
print(f"  Struggling Class AP: {avg_precision_struggling:.4f} (baseline: {baseline_struggling:.4f})")
print(f"  Succeeding Class AP: {avg_precision_succeeding:.4f} (baseline: {baseline_succeeding:.4f})")
plt.close()

# Threshold Tuning Curves
print("\nGenerating threshold tuning curves...")

thresholds_to_test = np.linspace(0, 1, 101)
precisions_struggling = []
recalls_struggling = []
f1s_struggling = []
precisions_succeeding = []
recalls_succeeding = []
f1s_succeeding = []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    y_pred_struggling_thresh = 1 - y_pred_thresh
    y_test_struggling = 1 - y_test
    
    prec_struggling = precision_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0) if y_pred_struggling_thresh.sum() > 0 else 0
    rec_struggling = recall_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0)
    f1_struggling = f1_score(y_test_struggling, y_pred_struggling_thresh, zero_division=0)
    precisions_struggling.append(prec_struggling)
    recalls_struggling.append(rec_struggling)
    f1s_struggling.append(f1_struggling)
    
    prec_succeeding = precision_score(y_test, y_pred_thresh, zero_division=0) if y_pred_thresh.sum() > 0 else 0
    rec_succeeding = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_succeeding = f1_score(y_test, y_pred_thresh, zero_division=0)
    precisions_succeeding.append(prec_succeeding)
    recalls_succeeding.append(rec_succeeding)
    f1s_succeeding.append(f1_succeeding)

optimal_idx_struggling = np.argmax(f1s_struggling)
optimal_threshold_struggling = thresholds_to_test[optimal_idx_struggling]
optimal_f1_struggling = f1s_struggling[optimal_idx_struggling]

optimal_idx_succeeding = np.argmax(f1s_succeeding)
optimal_threshold_succeeding = thresholds_to_test[optimal_idx_succeeding]
optimal_f1_succeeding = f1s_succeeding[optimal_idx_succeeding]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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
plt.savefig('threshold_tuning_curves_rf.png', dpi=300, bbox_inches='tight')
print("Saved: threshold_tuning_curves_rf.png")
print(f"  Struggling Class: Optimal threshold={optimal_threshold_struggling:.2f}, F1={optimal_f1_struggling:.4f}")
print(f"  Succeeding Class: Optimal threshold={optimal_threshold_succeeding:.2f}, F1={optimal_f1_succeeding:.4f}")
plt.close()

# 10. Summary
print("\n" + "="*80)
print("10. Summary of Key Findings (Random Forest)")
print("="*80)

print("\n📊 MODEL PERFORMANCE:")
print(f"  • ROC-AUC Score: {roc_auc:.4f}")
print(f"  • Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n🎯 TOP 3 MOST IMPORTANT ORIGINAL FEATURES (Aggregated):")
for idx, row in aggregated_df.head(3).iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Total_Importance']:.4f}")

print(f"\n🔍 TOP 5 INDIVIDUAL ENCODED FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")

print(f"\n🔍 SHAP INSIGHTS (Aggregated by Original Features):")
print("  Mean Absolute SHAP Values (Feature Impact):")
for idx, row in aggregated_shap_df.iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print("\n✅ FILES GENERATED:")
file_list = [
    'random_forest_feature_importance_aggregated.png',
    'random_forest_feature_importance_top20.png',
    'shap_summary_plot_rf.png',
    'shap_bar_plot_rf.png',
    'shap_aggregated_by_original_features_rf.png',
    'confusion_matrix_rf.png',
    'roc_curve_rf.png'
]
for f in file_list:
    print(f"  • {f}")
print(f"  • 5 SHAP dependence plots")
print(f"  • 2 SHAP force plots")

print("\n" + "="*80)
print("Analysis Complete! 🎉")
print("="*80)

# Save model and results
print("\nSaving trained model...")
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: random_forest_model.pkl")

results = {
    'model': model,
    'feature_importance': feature_importance,
    'aggregated_importance': aggregated_df,
    'shap_values': shap_values,
    'aggregated_shap': aggregated_shap_df,
    'roc_auc': roc_auc,
    'cv_scores': cv_scores,
    'confusion_matrix': cm,
    'feature_groups': feature_groups,
    'X_test': X_test,
    'y_test': y_test,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba
}

with open('random_forest_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: random_forest_results.pkl")

print("\n🌲 RANDOM FOREST ADVANTAGES:")
print("="*80)
print("• Ensemble learning: Averages predictions from 200 trees")
print("• Robust to overfitting: Each tree sees different data subset")
print("• Built-in class balancing: class_weight='balanced' parameter")
print("• Less sensitive to hyperparameters than gradient boosting")
print("• Handles non-linear relationships and interactions naturally")
print("="*80)
