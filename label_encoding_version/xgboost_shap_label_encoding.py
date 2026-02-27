"""
XGBoost + SHAP Analysis for MHQ Lifestyle Drivers in Tanzania
VERSION: LABEL ENCODING
Binary Classification: Succeeding (MHQ>=100) vs Struggling (MHQ<0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Tanzania")
print("VERSION: LABEL ENCODING")
print("="*80)

# 1. Load Data
print("\n1. Loading data...")
data = pd.read_csv('../rural_gmdata_forML.csv')
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# 2. Data Exploration
print("\n" + "="*80)
print("2. Data Exploration")
print("="*80)

print("\nColumn names and types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nMHQ Distribution:")
print(data['Overall.MHQ'].describe())

# 3. Prepare Features and Target
print("\n" + "="*80)
print("3. Preparing Features and Target")
print("="*80)

# Define features
features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
            'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']

print(f"\nFeatures used: {features}")

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

print(f"\nFeature data types before encoding:")
print(X.dtypes)

# 4. Handle Missing Values and Encode Categorical Variables
print("\n" + "="*80)
print("4. Data Preprocessing - LABEL ENCODING")
print("="*80)

# Check for missing values in features
print("\nMissing values in features:")
print(X.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_features = []

for col in X.columns:
    if X[col].dtype == 'object' or col != 'Education_Years':
        categorical_features.append(col)
        le = LabelEncoder()
        # Handle missing values by converting to string
        X[col] = X[col].fillna('Missing').astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"\nEncoded {col}: {len(le.classes_)} unique categories")
        print(f"  Categories: {list(le.classes_)[:10]}")  # Show first 10

# Handle any remaining missing values
X = X.fillna(X.median())

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features after preprocessing: {X.columns.tolist()}")

# 5. Split Data
print("\n" + "="*80)
print("5. Splitting Data")
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

# 6. Train XGBoost Model
print("\n" + "="*80)
print("6. Training XGBoost Model")
print("="*80)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

# XGBoost parameters
params = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'scale_pos_weight': scale_pos_weight,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False
}

print(f"\nModel parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Train model
model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("\nModel training completed!")

# 7. Model Evaluation
print("\n" + "="*80)
print("7. Model Evaluation")
print("="*80)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Struggling (MHQ<0)', 'Succeeding (MHQ>=100)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), 
                            scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 8. Feature Importance from XGBoost
print("\n" + "="*80)
print("8. Feature Importance (XGBoost)")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Rankings:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('XGBoost Feature Importance (Label Encoding)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: xgboost_feature_importance.png")
plt.close()

# 9. SHAP Analysis
print("\n" + "="*80)
print("9. SHAP (SHapley Additive exPlanations) Analysis")
print("="*80)

print("\nCalculating SHAP values... (this may take a moment)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP values calculated successfully!")

# SHAP Summary Plot (Global Feature Importance)
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title('SHAP Summary Plot (Label Encoding)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("Saved: shap_summary_plot.png")
plt.close()

# SHAP Bar Plot (Mean Absolute SHAP Values)
print("\nGenerating SHAP bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, 
                 plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Label Encoding)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
print("Saved: shap_bar_plot.png")
plt.close()

# SHAP Dependence Plots for Top 3 Features
print("\nGenerating SHAP dependence plots for top features...")
top_features = feature_importance['Feature'].head(3).tolist()

for i, feature in enumerate(top_features):
    plt.figure(figsize=(10, 6))
    feature_idx = X.columns.tolist().index(feature)
    shap.dependence_plot(feature_idx, shap_values, X_test, 
                        feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot: {feature} (Label Encoding)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: shap_dependence_{feature}.png")
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
plt.title('Confusion Matrix (Label Encoding)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Label Encoding)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curve.png")
plt.close()

# 11. Summary Statistics
print("\n" + "="*80)
print("11. Summary of Key Findings (LABEL ENCODING VERSION)")
print("="*80)

print("\n📊 MODEL PERFORMANCE:")
print(f"  • ROC-AUC Score: {roc_auc:.4f}")
print(f"  • Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n🎯 TOP 3 MOST IMPORTANT FEATURES (XGBoost):")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n🔍 SHAP INSIGHTS:")
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

print("  Mean Absolute SHAP Values (Feature Impact):")
for idx, row in shap_importance.iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Mean_SHAP']:.4f}")

print("\n✅ FILES GENERATED:")
print("  • xgboost_feature_importance.png")
print("  • shap_summary_plot.png")
print("  • shap_bar_plot.png")
for feature in top_features:
    print(f"  • shap_dependence_{feature}.png")
print("  • shap_force_plot_struggling.png")
print("  • shap_force_plot_succeeding.png")
print("  • confusion_matrix.png")
print("  • roc_curve.png")

print("\n" + "="*80)
print("Label Encoding Analysis Complete! 🎉")
print("="*80)

# Save model
print("\nSaving trained model...")
import pickle
try:
    with open('xgboost_model_label.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved: xgboost_model_label.pkl")
except Exception as e:
    print(f"Note: Model saving encountered an issue: {e}")

# Save label encoders and results
results = {
    'label_encoders': label_encoders,
    'feature_importance': feature_importance.to_dict(),
    'shap_importance': shap_importance.to_dict(),
    'roc_auc': roc_auc,
    'cv_scores': cv_scores.tolist(),
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}

with open('label_encoding_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved: label_encoding_results.pkl")

print("\n💡 LABEL ENCODING CHARACTERISTICS:")
print("="*80)
print("• Compact representation: Each categorical variable → single numeric column")
print("• 7 features total (6 categorical + 1 numerical)")
print("• Faster training and prediction")
print("• Trade-off: Implies ordering of categories (may not be appropriate)")
print("="*80)
