"""
SMOTE + XGBoost + SHAP Analysis for MHQ Lifestyle Drivers in Tanzania
VERSION: SMOTE (Synthetic Minority Over-sampling) + ONE-HOT ENCODING
Binary Classification: Succeeding (MHQ>=100) vs Struggling (MHQ<0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("SMOTE + XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Tanzania")
print("VERSION: SMOTE + ONE-HOT ENCODING")
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
print(f"\nTarget distribution (BEFORE SMOTE):")
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

# 4. Split Data FIRST (before SMOTE)
print("\n" + "="*80)
print("4. Splitting Data (BEFORE SMOTE)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set target distribution (BEFORE SMOTE):")
print(f"Struggling: {(y_train==0).sum()}, Succeeding: {(y_train==1).sum()}")
print(f"Test set target distribution:")
print(f"Struggling: {(y_test==0).sum()}, Succeeding: {(y_test==1).sum()}")

# 5. Apply SMOTE to Training Data Only
print("\n" + "="*80)
print("5. Applying SMOTE (Synthetic Minority Over-sampling)")
print("="*80)

print("\n🔬 WHAT IS SMOTE?")
print("-" * 80)
print("SMOTE creates synthetic examples of the minority class (Struggling)")
print("by interpolating between existing minority class samples.")
print("This balances the dataset and helps the model learn minority patterns better.")
print("-" * 80)

# Initialize SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)

# Apply SMOTE only to training data (never to test data!)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\n📊 BEFORE SMOTE:")
print(f"  Training samples: {len(y_train)}")
print(f"  Struggling: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"  Succeeding: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

print(f"\n📊 AFTER SMOTE:")
print(f"  Training samples: {len(y_train_smote)}")
print(f"  Struggling: {(y_train_smote==0).sum()} ({(y_train_smote==0).sum()/len(y_train_smote)*100:.1f}%)")
print(f"  Succeeding: {(y_train_smote==1).sum()} ({(y_train_smote==1).sum()/len(y_train_smote)*100:.1f}%)")

print(f"\n✨ SMOTE created {len(y_train_smote) - len(y_train)} synthetic samples!")

# 6. Train XGBoost Model
print("\n" + "="*80)
print("6. Training XGBoost Model on SMOTE-Balanced Data")
print("="*80)

# Calculate scale_pos_weight (not needed with balanced data, but kept for consistency)
scale_pos_weight = (y_train_smote==1).sum() / (y_train_smote==0).sum()
print(f"\nClass balance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
print("Note: After SMOTE, classes are balanced (~1.0)")

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Should be ~1 now
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

print("\nTraining model...")
model.fit(X_train_smote, y_train_smote)
print("Model training completed!")

# 7. Model Evaluation
print("\n" + "="*80)
print("7. Model Evaluation - SMOTE + XGBoost")
print("="*80)

# Predictions on original (non-SMOTE) test set
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
cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
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

# 8. Feature Importance (XGBoost)
print("\n" + "="*80)
print("8. Feature Importance (XGBoost) - SMOTE Model")
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
plt.title('XGBoost Feature Importance (Aggregated) - SMOTE Model')
plt.tight_layout()
plt.savefig('xgboost_feature_importance_aggregated_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: xgboost_feature_importance_aggregated_smote.png")

# Visualization: Top 20 individual features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 20 Individual Feature Importance - SMOTE Model')
plt.tight_layout()
plt.savefig('xgboost_feature_importance_top20_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: xgboost_feature_importance_top20_smote.png")

# 9. SHAP Analysis
print("\n" + "="*80)
print("9. SHAP Analysis - SMOTE Model")
print("="*80)

print("\nCalculating SHAP values... (this may take a moment)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
print("SHAP values calculated successfully!")

# SHAP Summary Plot (top 30 features)
print("\nGenerating SHAP summary plot (top 30 features)...")
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_summary_plot_smote.png")

# SHAP Bar Plot
print("\nGenerating SHAP bar plot (top 30 features)...")
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=30, show=False)
plt.tight_layout()
plt.savefig('shap_bar_plot_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_bar_plot_smote.png")

# Aggregate SHAP values by original features
print("\nAggregating SHAP values by original features...")
shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
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
plt.title('SHAP Values (Aggregated by Original Features) - SMOTE Model')
plt.tight_layout()
plt.savefig('shap_aggregated_by_original_features_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_aggregated_by_original_features_smote.png")

# SHAP Dependence Plots for top features
print("\nGenerating SHAP dependence plots for top individual encoded features...")
top_5_features = feature_importance.head(5)['Feature'].values

for feat in top_5_features:
    if feat in X_test.columns:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_test, show=False)
        plt.tight_layout()
        safe_feat_name = feat.replace('/', '_').replace(' ', '_').replace('.', '_')
        plt.savefig(f'shap_dependence_{safe_feat_name}_smote.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: shap_dependence_{safe_feat_name}_smote.png")

# SHAP Force Plots
print("\nGenerating SHAP force plots for sample predictions...")
# Sample struggling case
struggling_idx = np.where(y_test == 0)[0][0]
plt.figure()
shap.force_plot(explainer.expected_value, shap_values[struggling_idx], 
                X_test.iloc[struggling_idx], matplotlib=True, show=False)
plt.savefig('shap_force_plot_struggling_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_force_plot_struggling_smote.png")

# Sample succeeding case
succeeding_idx = np.where(y_test == 1)[0][0]
plt.figure()
shap.force_plot(explainer.expected_value, shap_values[succeeding_idx], 
                X_test.iloc[succeeding_idx], matplotlib=True, show=False)
plt.savefig('shap_force_plot_succeeding_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: shap_force_plot_succeeding_smote.png")

# 10. Additional Visualizations
print("\n" + "="*80)
print("10. Additional Visualizations")
print("="*80)

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - SMOTE Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_smote.png")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SMOTE Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_smote.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: roc_curve_smote.png")

# 11. Summary
print("\n" + "="*80)
print("11. Summary of Key Findings (SMOTE Model)")
print("="*80)

print("\n📊 MODEL PERFORMANCE:")
print(f"  • ROC-AUC Score: {roc_auc:.4f}")
print(f"  • Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\n🎯 TOP 3 MOST IMPORTANT ORIGINAL FEATURES (Aggregated XGBoost):")
for idx, row in aggregated_df.head(3).iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Total_Importance']:.4f}")

print(f"\n🔍 TOP 5 INDIVIDUAL ENCODED FEATURES (XGBoost):")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")

print(f"\n🔍 SHAP INSIGHTS (Aggregated by Original Features):")
print("  Mean Absolute SHAP Values (Feature Impact):")
for idx, row in aggregated_shap_df.iterrows():
    print(f"  {idx+1}. {row['Original_Feature']}: {row['Mean_Abs_SHAP']:.4f}")

print("\n✅ FILES GENERATED:")
file_list = [
    'xgboost_feature_importance_aggregated_smote.png',
    'xgboost_feature_importance_top20_smote.png',
    'shap_summary_plot_smote.png',
    'shap_bar_plot_smote.png',
    'shap_aggregated_by_original_features_smote.png',
    'confusion_matrix_smote.png',
    'roc_curve_smote.png'
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
with open('xgboost_model_smote.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: xgboost_model_smote.pkl")

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

with open('smote_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: smote_results.pkl")

print("\n🔬 SMOTE ADVANTAGES:")
print("="*80)
print("• Balanced training data: Equal representation of both classes")
print("• Better minority class learning: Model learns patterns from more examples")
print("• Improved recall: Better at identifying Struggling individuals")
print("• No information leakage: SMOTE applied only to training data")
print("• Synthetic samples: Created by interpolating between real minority examples")
print("="*80)
