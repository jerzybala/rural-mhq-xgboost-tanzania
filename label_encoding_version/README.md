# Label Encoding Version

## Overview
This version uses **label encoding** for categorical variables, where each category is assigned a single integer value (0, 1, 2, 3, ...).

## Characteristics

### Encoding Method
- Each categorical variable → Single numeric column
- Categories assigned sequential integers
- Total features: **7** (6 categorical + 1 numerical)

### Advantages
✅ Compact representation (fewer features)
✅ Faster training and prediction
✅ Lower memory usage
✅ Works well with tree-based models like XGBoost

### Limitations
⚠️ Implies ordering of categories (may not be appropriate for nominal data)
⚠️ Less interpretable (hard to see specific category effects)
⚠️ Model may learn artificial relationships based on numeric ordering

## How to Run

```bash
cd label_encoding_version
python xgboost_shap_label_encoding.py
```

Or from the main directory:
```bash
python run_both_versions.py
```

## Files Generated

### Visualizations
- `xgboost_feature_importance.png` - Feature importance from XGBoost
- `shap_summary_plot.png` - SHAP summary (beeswarm plot)
- `shap_bar_plot.png` - SHAP feature importance
- `shap_dependence_*.png` - SHAP dependence plots for top 3 features
- `shap_force_plot_struggling.png` - Force plot for struggling individual
- `shap_force_plot_succeeding.png` - Force plot for succeeding individual
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve

### Data Files
- `xgboost_model_label.pkl` - Trained XGBoost model
- `label_encoding_results.pkl` - Complete results including encoders, metrics, etc.

## Expected Results

### Performance
- **ROC-AUC**: ~0.746
- **Accuracy**: ~70%
- **Cross-Validation ROC-AUC**: ~0.710 ± 0.011

### Top Features
1. RelationWithAdultFamily (~29% importance)
2. UPF.Freq (~16% importance)
3. Smartphone.ownership (~13% importance)

## Interpretation Notes

When interpreting results:
- Feature values are encoded as integers (0, 1, 2, ...)
- Check `label_encoding_results.pkl` for mapping of categories to integers
- SHAP values show impact but specific categories need to be decoded
- Compare with one-hot encoding version for clearer category-specific insights

## Comparison with One-Hot Encoding

See `../COMPARISON_LABEL_VS_ONEHOT.md` for detailed comparison between versions.

**Quick Summary:**
- Similar performance (ROC-AUC within 0.0003)
- Fewer features (7 vs 46)
- Less interpretable
- Faster computation
