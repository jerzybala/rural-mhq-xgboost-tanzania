# One-Hot Encoding Version

## Overview
This version uses **one-hot encoding** for categorical variables, where each category becomes its own binary column (0 or 1).

## Characteristics

### Encoding Method
- Each categorical variable → Multiple binary columns (one per category)
- Each column represents presence (1) or absence (0) of that category
- Total features: **46** (45 binary + 1 numerical)

### Advantages
✅ No artificial ordering of categories
✅ Each category's effect is explicit and measurable
✅ Highly interpretable (can see specific category impacts)
✅ Better for stakeholder communication
✅ More actionable for interventions

### Limitations
⚠️ Higher dimensionality (46 vs 7 features)
⚠️ Slightly slower computation
⚠️ Higher memory usage
⚠️ Can lead to sparse data with many categories

## How to Run

```bash
cd one_hot_encoding_version
python xgboost_shap_one_hot_encoding.py
```

Or from the main directory:
```bash
python run_both_versions.py
```

## Files Generated

### Visualizations
- `xgboost_feature_importance_aggregated.png` ⭐ - Aggregated by original features
- `xgboost_feature_importance_top20.png` - Top 20 individual encoded features
- `shap_aggregated_by_original_features.png` ⭐ - SHAP aggregated by original features
- `shap_summary_plot.png` - SHAP summary for top 30 encoded features
- `shap_bar_plot.png` - SHAP bar plot for top 30 encoded features
- `shap_dependence_*.png` - SHAP dependence plots for top 5 individual categories
- `shap_force_plot_struggling.png` - Force plot for struggling individual
- `shap_force_plot_succeeding.png` - Force plot for succeeding individual
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve

### Data Files
- `xgboost_model_onehot.pkl` - Trained XGBoost model
- `one_hot_encoding_results.pkl` - Complete results including feature groups, metrics, etc.

## Expected Results

### Performance
- **ROC-AUC**: ~0.746
- **Accuracy**: ~71%
- **Cross-Validation ROC-AUC**: ~0.709 ± 0.008

### Top Features (Aggregated)
1. RelationWithAdultFamily (~31% importance)
2. AgeOfFirstSP (~20% importance) ⬆️ Higher than label encoding!
3. UPF.Freq (~15% importance)

### Top Individual Categories
1. **RelationWithAdultFamily_1_Very Close to Most** - 11.9% importance
2. **UPF.Freq_several times a day** - 5.4% importance
3. **RelationWithAdultFamily_4_Dont Get Along** - 5.3% importance

## Interpretation Notes

### Key Insights
- **"Very Close to Most" family relationships** account for 12% of predictive power alone!
- **High-frequency UPF consumption** ("several times a day") is specifically harmful
- **Poor family relationships** ("Don't Get Along") strongly predict struggling MHQ
- Can see exact category effects, not just overall feature importance

### Reading Feature Names
Format: `OriginalFeature_CategoryName`

Examples:
- `RelationWithAdultFamily_1_Very Close to Most` = Being very close to most family
- `UPF.Freq_several times a day` = Consuming UPF multiple times daily
- `Exercise.Freq_once a day` = Exercising once per day

### Aggregated vs Individual Results
- **Aggregated**: Shows overall importance of original features (comparable to label encoding)
- **Individual**: Shows which specific categories within each feature drive predictions (unique to one-hot)

## Comparison with Label Encoding

See `../COMPARISON_LABEL_VS_ONEHOT.md` for detailed comparison between versions.

**Quick Summary:**
- Similar performance (ROC-AUC within 0.0003)
- More features (46 vs 7)
- Much more interpretable ⭐
- Slightly slower computation
- **Recommended for publication and presentation**

## Why Use This Version?

### Best for:
1. **Publications** - More rigorous methodology
2. **Presentations** - Clear, specific findings
3. **Interventions** - Know exactly which categories to target
4. **Stakeholder communication** - Easy to explain
5. **Policy recommendations** - Actionable, specific targets

### Example Clear Findings:
- "Being 'Very Close to Most' family members increases MHQ prediction by 11.9%"
- "Consuming UPF 'several times a day' decreases MHQ prediction by 5.4%"
- "Having poor family relationships ('Don't Get Along') decreases MHQ by 5.3%"

These specific insights are not available with label encoding!
