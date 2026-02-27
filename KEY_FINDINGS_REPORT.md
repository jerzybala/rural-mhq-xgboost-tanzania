# Key Findings Report: Machine Learning Analysis of Lifestyle Drivers of Mental Health in Rural Tanzania

**Date:** January 2, 2026  
**Dataset:** 5,095 participants from rural Tanzania  
**Outcome:** Binary classification — Struggling (MHQ < 0) vs Succeeding (MHQ ≥ 100)

---

## Executive Summary

This analysis used machine learning (XGBoost with SHAP explainability) to identify lifestyle factors associated with mental health outcomes in rural Tanzanian populations (n=5,095). While predictive performance was moderate (ROC-AUC = 0.74, minority class F1 = 0.54), reflecting the complexity of predicting mental health from limited lifestyle factors, the consistency of feature importance rankings across four machine learning algorithms (XGBoost, LightGBM, Random Forest, SMOTE+XGBoost) provides confidence in the identified associations. Having "Very Close" relationships with most adult family members emerged as the single strongest predictor of mental health success, accounting for ~40% of the model's predictive power, followed by dietary habits (lower ultra-processed food consumption) and regular exercise.

---

## 1. Top Predictors of Mental Health Status

### Aggregated Feature Importance (7 Original Features)

| Rank | Feature | XGBoost Importance | Mean Abs SHAP | Interpretation |
|------|---------|-------------------|---------------|----------------|
| 1 | **Relationship with Adult Family** | 40.0% | 0.537 | Strongest predictor |
| 2 | **Ultra-Processed Food Frequency** | 19.2% | 0.331 | Diet quality matters |
| 3 | **Exercise Frequency** | 13.8% | 0.265 | Physical activity protective |
| 4 | ShareHomeWith | 10.2% | 0.158 | Living situation |
| 5 | Age of First Smartphone | 8.2% | 0.179 | Technology exposure timing |
| 6 | Smartphone Ownership | 6.0% | 0.085 | Technology access |
| 7 | Education Years | 2.7% | 0.133 | Educational attainment |

### Key Insight
Having **"Very Close" relationships with most adult family members** is the single strongest predictor of mental health success, accounting for ~40% of the model's predictive power. This is followed by **dietary habits** (lower ultra-processed food consumption) and **regular exercise**.

---

## 2. Model Performance Comparison

### Algorithm Comparison

| Model | ROC-AUC | CV ROC-AUC | Struggling F1 | Succeeding F1 |
|-------|---------|------------|---------------|---------------|
| **XGBoost (Baseline)** | **0.746** | 0.716 | 0.53 | 0.76 |
| Random Forest | 0.744 | 0.717 | 0.54 | 0.78 |
| LightGBM | 0.742 | 0.707 | 0.55 | 0.78 |
| SMOTE + XGBoost | 0.738 | 0.715 | 0.52 | 0.76 |

### Observations
- All four algorithms produced **consistent results** (ROC-AUC range: 0.738–0.746)
- XGBoost and Random Forest performed best
- SMOTE oversampling did not improve performance
- Consistency across algorithms **strengthens confidence** in identified predictors

---

## 3. Class-Specific Performance

### Class Distribution
- **Struggling (MHQ < 0):** 1,386 (27.2%) — Minority class
- **Succeeding (MHQ ≥ 100):** 3,709 (72.8%) — Majority class

### Precision-Recall Analysis

| Class | Average Precision | Baseline (Random) | Improvement |
|-------|------------------|-------------------|-------------|
| Struggling | 0.57 | 0.27 | **+111% above baseline** |
| Succeeding | 0.87 | 0.73 | +19% above baseline |

### Threshold Optimization
- **Optimal threshold for Struggling class:** 0.52 (F1 = 0.54)
- **Optimal threshold for Succeeding class:** 0.30 (F1 = 0.86)

---

## 4. Interpreting Moderate Predictive Performance

### Is ROC-AUC = 0.74 Good Enough?

| Context | ROC-AUC Interpretation |
|---------|----------------------|
| < 0.50 | Worse than random |
| 0.50–0.60 | Poor |
| 0.60–0.70 | Fair |
| **0.70–0.80** | **Acceptable/Good** ✓ |
| 0.80–0.90 | Very Good |
| > 0.90 | Excellent (rare for behavioral data) |

**Your ROC-AUC of 0.74 falls in the "Acceptable/Good" range**, which is typical for behavioral and social science research where outcomes are influenced by many unmeasured factors.

### Why F1 = 0.54 for Struggling Class is Reasonable

1. **Class imbalance:** Struggling class is only 27% of the data
2. **Complex outcome:** Mental health is influenced by genetics, trauma, economic factors, and other unmeasured variables
3. **Limited features:** Only 7 lifestyle factors available; many important predictors are missing
4. **Baseline comparison:** Model performs 2× better than random guessing for minority class

---

## 5. Validity of SHAP Analysis with Moderate Performance

### SHAP Answers a Different Question Than Prediction

| Goal | Question | Metric | Your Result |
|------|----------|--------|-------------|
| **Prediction** | "Can we classify individuals accurately?" | F1, ROC-AUC | Moderate |
| **Explanation** | "What factors are associated with the outcome?" | SHAP, Importance | **Valid ✓** |

### When Would SHAP Be Unreliable?

| Condition | Status in Your Analysis |
|-----------|------------------------|
| ROC-AUC ≈ 0.50 (random guessing) | ❌ No — Your AUC = 0.74 |
| Severe overfitting | ❌ No — CV and test scores are similar |
| Data quality issues | ❌ No — Data is clean |
| Inconsistent results across algorithms | ❌ No — All 4 algorithms agree |

### Conclusion on SHAP Validity

> **The SHAP analysis is valid and scientifically meaningful.** The moderate predictive performance reflects the inherent complexity of predicting mental health outcomes, not a failure of the analysis. The consistency of feature importance rankings across four different algorithms provides strong confidence in the identified predictors.

---

## 6. Scientific Interpretation

### What the Model Learned

The model identified three primary lifestyle domains associated with mental health outcomes:

1. **Social Connectedness** (RelationWithAdultFamily, ShareHomeWith)
   - Having very close family relationships strongly predicts better mental health
   - This aligns with extensive literature on social support and mental health

2. **Diet Quality** (UPF.Freq)
   - Higher ultra-processed food consumption is associated with struggling mental health
   - Consistent with emerging research on diet-mental health connections

3. **Physical Activity** (Exercise.Freq)
   - Regular exercise is associated with better mental health outcomes
   - Well-established in mental health literature

### What the Model Cannot Tell Us

- **Causation:** Associations do not prove causality
- **Unmeasured confounders:** Income, employment, life events, genetics not captured
- **Generalizability:** Results specific to rural Tanzania population
- **Individual prediction:** Model not suitable for clinical decision-making at individual level

---

## 7. Recommendations

### For This Research
1. Report SHAP findings as **associations**, not causal effects
2. Emphasize **consistency across algorithms** as evidence of robust patterns
3. Acknowledge limitations of moderate predictive performance
4. Consider these findings as **hypothesis-generating** for future research

### Suggested Wording for Publications

> *"Using XGBoost with SHAP explainability, we identified lifestyle factors associated with mental health outcomes in rural Tanzania (n=5,095). While predictive performance was moderate (ROC-AUC = 0.74, minority class F1 = 0.54), reflecting the complexity of predicting mental health from limited lifestyle factors, the consistency of feature importance rankings across four machine learning algorithms (XGBoost, LightGBM, Random Forest, SMOTE+XGBoost) provides confidence in the identified associations. Family relationship quality emerged as the strongest predictor, followed by dietary habits and exercise frequency."*

---

## 8. Files Generated

### Visualizations
- `roc_curve.png` — Overall ROC curve
- `roc_curves_by_class.png` — Class-specific ROC curves
- `precision_recall_curves_by_class.png` — Precision-recall by class
- `threshold_tuning_curves.png` — F1/Precision/Recall vs threshold
- `shap_summary_plot.png` — SHAP beeswarm plot (top 30 features)
- `shap_bar_plot.png` — SHAP importance bar chart
- `shap_aggregated_by_original_features.png` — Aggregated SHAP by original features
- `xgboost_feature_importance_aggregated.png` — Feature importance by original features
- `confusion_matrix.png` — Confusion matrix heatmap

### Data Exports
- `feature_importance_and_shap_analysis.xlsx` — Excel file with 6 sheets:
  1. Individual_Features (46 one-hot encoded features)
  2. Aggregated_Importance (7 original features)
  3. Individual_SHAP (46 features)
  4. Aggregated_SHAP (7 features)
  5. Top20_Combined (top 20 with both metrics)
  6. Aggregated_Combined (summary table)

---

## 9. Technical Details

### Model Configuration
- **Algorithm:** XGBoost Classifier with hyperparameter optimization
- **Encoding:** One-hot encoding (7 features → 46 features)
- **Class imbalance handling:** scale_pos_weight parameter
- **Validation:** 5-fold stratified cross-validation
- **Train/Test split:** 80/20 stratified

### Optimized Hyperparameters
- max_depth: 3
- learning_rate: 0.09
- n_estimators: 100
- min_child_weight: 8
- subsample: 0.9
- colsample_bytree: 0.6

---

**Report prepared using XGBoost + SHAP analysis pipeline**  
**Analysis location:** `/Users/jerzybala/Desktop/Rural_gmdata_for_ML/one_hot_encoding_version/`
