"""
Export Feature Importance and SHAP Values to Excel
Reads from saved model and generates comprehensive Excel file
"""

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import shap

print("="*80)
print("Exporting Feature Importance and SHAP Values to Excel")
print("="*80)

# Load saved model and results
print("\nLoading saved model and results...")
with open('xgboost_model_onehot.pkl', 'rb') as f:
    model = pickle.load(f)

with open('one_hot_encoding_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("  ✓ Model loaded")
print("  ✓ Results loaded")

# Get feature names and aggregated data
feature_columns = results['feature_columns']

# Convert dict format to DataFrame
aggregated_importance = pd.DataFrame(results['aggregated_importance'])
aggregated_shap = pd.DataFrame(results['aggregated_shap'])

print(f"  ✓ {len(feature_columns)} features")
print(f"  ✓ {len(aggregated_importance)} original features")

# Get feature importance from model
importance_scores = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance_scores
}).sort_values('Importance', ascending=False)

print(f"  ✓ Extracted {len(feature_importance)} feature importances")

# Load data to calculate SHAP
print("\nLoading data for SHAP calculation...")
data = pd.read_csv('../rural_gmdata_forML.csv')

# Prepare features (same as in main script)
features = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily', 
            'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']
data_filtered = data[(data['Overall.MHQ'] >= 100) | (data['Overall.MHQ'] < 0)].copy()
X_raw = data_filtered[features].copy()

# One-hot encoding
categorical_features = X_raw.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X_raw, columns=categorical_features, dummy_na=True, 
                   dtype=int, drop_first=False)

# Fill missing numeric
X['Education_Years'] = X['Education_Years'].fillna(X['Education_Years'].median())

print(f"  ✓ Prepared {len(X)} samples with {len(X.columns)} features")

# Calculate SHAP values (use a sample for speed)
print("\nCalculating SHAP values (using sample for speed)...")
sample_size = min(500, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print(f"  ✓ SHAP calculated for {sample_size} samples")

# Create individual SHAP dataframe
shap_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_Abs_SHAP', ascending=False)

# Normalize feature importance to sum to 1
feature_importance['Importance'] = feature_importance['Importance'] / feature_importance['Importance'].sum()

# Create Excel file with multiple sheets
excel_filename = 'feature_importance_and_shap_analysis.xlsx'

print(f"\nCreating Excel file: {excel_filename}")

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # Sheet 1: Individual Feature Importance
    feature_importance.to_excel(writer, sheet_name='Individual_Features', index=False)
    print("  ✓ Sheet 1: Individual_Features")
    
    # Sheet 2: Aggregated Feature Importance
    aggregated_importance.to_excel(writer, sheet_name='Aggregated_Importance', index=False)
    print("  ✓ Sheet 2: Aggregated_Importance")
    
    # Sheet 3: Individual SHAP Values (Mean Absolute)
    shap_importance_df.to_excel(writer, sheet_name='Individual_SHAP', index=False)
    print("  ✓ Sheet 3: Individual_SHAP")
    
    # Sheet 4: Aggregated SHAP Values
    aggregated_shap.to_excel(writer, sheet_name='Aggregated_SHAP', index=False)
    print("  ✓ Sheet 4: Aggregated_SHAP")
    
    # Sheet 5: Combined Summary (Top 20 features)
    combined_df = pd.merge(
        feature_importance.head(20),
        shap_importance_df[['Feature', 'Mean_Abs_SHAP']],
        on='Feature',
        how='left'
    )
    combined_df.to_excel(writer, sheet_name='Top20_Combined', index=False)
    print("  ✓ Sheet 5: Top20_Combined")
    
    # Sheet 6: Aggregated Combined
    aggregated_combined = pd.merge(
        aggregated_importance,
        aggregated_shap,
        on='Original_Feature',
        how='left'
    ).sort_values('Total_Importance', ascending=False)
    aggregated_combined.to_excel(writer, sheet_name='Aggregated_Combined', index=False)
    print("  ✓ Sheet 6: Aggregated_Combined")

print(f"\n✅ Excel file saved: {excel_filename}")
print("\n" + "="*80)
print("KEY INSIGHTS - AGGREGATED FEATURES:")
print("="*80 + "\n")
print(aggregated_combined.to_string(index=False))

print("\n" + "="*80)
print("Export Complete! 🎉")
print("="*80)
