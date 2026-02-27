# Hyperparameter Optimization Results

## Summary

A comprehensive 2-stage hyperparameter optimization was performed on the One-Hot Encoding model to improve prediction performance.

## Optimization Strategy

### Stage 1: Randomized Search (Broad Exploration)
- **Method**: RandomizedSearchCV
- **Iterations**: 50 parameter combinations
- **Cross-validation**: 3-fold CV
- **Total fits**: 150 (50 candidates × 3 folds)
- **Best CV ROC-AUC**: 0.7146

**Best Parameters from Random Search**:
- `max_depth`: 3
- `learning_rate`: 0.07
- `n_estimators`: 100
- `min_child_weight`: 7
- `subsample`: 0.9
- `colsample_bytree`: 0.6
- `gamma`: 0.1
- `reg_alpha`: 0.5
- `reg_lambda`: 1.0

### Stage 2: Grid Search (Fine-tuning)
- **Method**: GridSearchCV
- **Cross-validation**: 5-fold CV
- **Total fits**: 3,645 (729 candidates × 5 folds)
- **Best CV ROC-AUC**: 0.7167
- **Improvement over Stage 1**: +0.0021

**Optimized Parameters**:
- `max_depth`: 3
- `learning_rate`: 0.09
- `n_estimators`: 100
- `min_child_weight`: 8
- `subsample`: 0.9
- `colsample_bytree`: 0.6

## Performance Comparison

| Metric | Baseline (No Optimization) | Optimized Model | Change |
|--------|---------------------------|-----------------|--------|
| **Test ROC-AUC** | 0.7461 | 0.7428 | -0.0033 (-0.44%) |
| **CV ROC-AUC** | Not measured | 0.7155 ± 0.0090 | N/A |
| **Accuracy** | 71% | 68% | -3% |
| **Precision (Struggling)** | 0.48 | 0.44 | -0.04 |
| **Recall (Struggling)** | 0.67 | 0.66 | -0.01 |
| **F1-Score (Struggling)** | 0.56 | 0.53 | -0.03 |
| **Balanced Accuracy** | Not measured | 0.68 | N/A |

## Key Findings

### 1. Optimization Did Not Improve Performance
The hyperparameter-optimized model actually performed **slightly worse** than the baseline model:
- Test ROC-AUC decreased by 0.0033 (0.44%)
- Precision for the Struggling class dropped from 0.48 to 0.44
- Recall remained relatively stable (0.67 → 0.66)

### 2. Why Optimization Might Not Help

Several factors could explain why optimization didn't improve performance:

**a) Data-Related Factors**:
- **Class Imbalance**: 27.2% Struggling vs 72.8% Succeeding
  - The model may already be near optimal given the imbalance
  - Further optimization may lead to overfitting the majority class
  
- **Feature Quality**: The lifestyle predictors may have limited discriminative power
  - RelationWithAdultFamily dominates (~40% importance)
  - Other features contribute relatively less
  
- **Sample Size**: 5,095 participants split into:
  - Training: 4,076 (1,109 Struggling, 2,967 Succeeding)
  - Test: 1,019 (277 Struggling, 742 Succeeding)
  - With 46 one-hot encoded features, the model may be approaching capacity

**b) Model-Related Factors**:
- **Already Near Optimal**: The baseline parameters were already reasonable
  - Default XGBoost parameters work well for many datasets
  - The search space explored may not contain significantly better configurations
  
- **Overfitting Risk**: More complex models (higher depth, more estimators) may overfit
  - Optimized model uses similar depth (3) and estimators (100) as baseline
  - This suggests the simpler model generalizes better

**c) Random Variation**:
- Different train/test splits could produce different results
- The 0.0033 difference is within typical variation range
- Cross-validation shows higher variability (0.7155 ± 0.0090)

### 3. Cross-Validation Insights
- **Mean CV ROC-AUC**: 0.7155
- **Standard deviation**: ±0.0090
- **Range**: [0.7085, 0.7299]

The test score (0.7428) is notably higher than CV mean (0.7155), suggesting:
- Some lucky variance in the test set split
- Potential overfitting concerns

### 4. Feature Importance Stability
Despite optimization, feature importance remained consistent:

**Top 3 Features (Unchanged)**:
1. RelationWithAdultFamily: ~40%
2. UPF.Freq: ~19%
3. Exercise.Freq: ~14%

This stability suggests:
- The model is capturing genuine patterns
- The relationships are robust across different hyperparameters

## Recommendations

### 1. Use Baseline Model
**Recommendation**: Keep the baseline (non-optimized) one-hot encoding model.

**Reasons**:
- Better test performance (0.7461 vs 0.7428)
- Simpler to explain and maintain
- No significant difference justifies the added complexity
- Lower risk of overfitting

### 2. Address Root Causes Instead
Rather than further hyperparameter tuning, consider:

**a) Feature Engineering**:
- Create interaction features (e.g., RelationWithAdultFamily × ShareHomeWith)
- Add domain-specific features if available
- Consider polynomial features for Education_Years

**b) Class Imbalance Techniques**:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class weights adjustment**: Increase penalty for misclassifying Struggling
- **Threshold tuning**: Adjust classification threshold to favor recall
- **Ensemble methods**: Combine predictions from multiple models

**c) Alternative Models**:
- Try ensemble methods (Random Forest, LightGBM)
- Consider calibrated classifiers
- Test neural networks if sufficient data exists

**d) Data Collection**:
- Increase sample size, especially for Struggling class
- Collect additional features that may be more predictive
- Consider longitudinal data if temporal patterns exist

### 3. Performance Context
ROC-AUC of 0.746 is **moderate** for this type of problem:
- **Good**: 0.8-0.9
- **Fair**: 0.7-0.8 ← Your model is here
- **Poor**: < 0.7

For mental health prediction with lifestyle factors:
- This performance is reasonable given feature limitations
- Social/behavioral factors alone may not fully predict mental health
- Consider adding clinical, genetic, or environmental features

## Computational Cost

The optimization process required:
- **Stage 1**: 150 model fits (~5 minutes)
- **Stage 2**: 3,645 model fits (~45 minutes)
- **Total time**: ~50 minutes

**Cost-benefit analysis**: 50 minutes of computation resulted in -0.44% performance change, indicating optimization was not worthwhile for this dataset.

## Conclusion

1. ✅ **Optimization completed successfully**: Explored 779 parameter combinations
2. ❌ **No performance gain**: ROC-AUC decreased by 0.0033
3. 📊 **Baseline model recommended**: Simpler and performs better
4. 🔍 **Focus on other improvements**: Feature engineering, class imbalance handling, or collecting better data would be more beneficial than further hyperparameter tuning

---

## Files Generated During Optimization

All output files are in `one_hot_encoding_version/`:
- Model: `xgboost_model_onehot.pkl`
- Results: `one_hot_encoding_results.pkl`
- 14 visualization PNG files
- This summary: `OPTIMIZATION_RESULTS.md`
