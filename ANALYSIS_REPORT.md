# XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Tanzania

## Executive Summary

This analysis uses XGBoost machine learning with SHAP (SHapley Additive exPlanations) to understand which lifestyle factors predict mental health outcomes (MHQ) in Tanzania's rural population.

**Classification Task**: Binary classification to distinguish between:
- **Succeeding** (MHQ ≥ 100): Good mental health
- **Struggling** (MHQ < 0): Poor mental health

---

## Dataset Overview

- **Total participants**: 5,095
- **Struggling (MHQ < 0)**: 1,386 participants (27.2%)
- **Succeeding (MHQ ≥ 100)**: 3,709 participants (72.8%)
- **Training set**: 4,076 samples
- **Test set**: 1,019 samples

---

## Features Analyzed

1. **Education_Years** (continuous) - Years of formal education
2. **ShareHomeWith** (categorical) - Number of people sharing home
3. **RelationWithAdultFamily** (categorical) - Quality of family relationships
4. **UPF.Freq** (categorical) - Ultra-processed food consumption frequency
5. **AgeOfFirstSP** (categorical) - Age when first owned a smartphone
6. **Smartphone.ownership** (categorical) - Current smartphone ownership status
7. **Exercise.Freq** (categorical) - Exercise frequency

---

## Model Performance

### Overall Metrics
- **ROC-AUC Score**: 0.7464 (Test set)
- **Cross-Validation ROC-AUC**: 0.7097 ± 0.0109
- **Accuracy**: 70%

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Struggling (MHQ<0) | 0.47 | 0.64 | 0.54 | 277 |
| Succeeding (MHQ≥100) | 0.84 | 0.73 | 0.78 | 742 |

### Confusion Matrix
- True Negatives (Correctly predicted Struggling): 177
- False Positives (Struggling predicted as Succeeding): 100
- False Negatives (Succeeding predicted as Struggling): 201
- True Positives (Correctly predicted Succeeding): 541

---

## Key Findings

### 🎯 Top 3 Most Important Features (XGBoost Feature Importance)

1. **RelationWithAdultFamily** (29.1% importance)
   - The quality of relationships with adult family members is the strongest predictor
   - Being "Very Close to Most" family members is associated with better MHQ
   
2. **UPF.Freq** (15.9% importance)
   - Ultra-processed food consumption frequency is the second most important factor
   - Lower UPF consumption ("rarely/never") correlates with better mental health

3. **Smartphone.ownership** (13.2% importance)
   - Smartphone ownership status significantly impacts mental health outcomes
   
### 🔍 SHAP Analysis Insights (Mean Absolute Impact)

SHAP values reveal the actual impact and direction of each feature:

1. **RelationWithAdultFamily** (0.4805) - Highest impact
2. **UPF.Freq** (0.2706) - Second highest impact
3. **Education_Years** (0.1999) - Moderate impact
4. **Exercise.Freq** (0.1900) - Moderate impact
5. **AgeOfFirstSP** (0.1555) - Lower impact
6. **ShareHomeWith** (0.1495) - Lower impact
7. **Smartphone.ownership** (0.0928) - Lowest impact

**Key Insight**: While XGBoost and SHAP rankings are similar, SHAP reveals that Education_Years has more actual predictive impact than Smartphone.ownership, despite the latter having higher XGBoost feature importance.

---

## Detailed Interpretations

### 1. RelationWithAdultFamily (Most Important Factor)
- **Strong family relationships are the #1 predictor of mental health success**
- Participants who are "Very Close to Most" family members show significantly higher MHQ scores
- Social disconnection ("Not Close" or "Don't Get Along") strongly predicts struggling mental health
- **Policy Implication**: Community programs that strengthen family bonds could have major mental health benefits

### 2. UPF.Freq (Ultra-Processed Food Frequency)
- **Higher UPF consumption is associated with poorer mental health**
- "Rarely/never" consuming UPF predicts better MHQ outcomes
- The relationship appears dose-dependent (more frequent = worse outcomes)
- **Policy Implication**: Nutrition education and access to whole foods may improve mental health

### 3. Education_Years
- **More years of education correlate with better mental health outcomes**
- Education may provide:
  - Better coping mechanisms
  - Improved economic opportunities
  - Enhanced social connections
  - Greater health literacy
- **Policy Implication**: Expanding educational access could have mental health benefits

### 4. Exercise.Freq
- **Regular exercise is protective for mental health**
- Daily or frequent exercise shows positive associations with succeeding MHQ
- Physical activity may serve as a buffer against mental health struggles
- **Policy Implication**: Promoting physical activity programs in rural communities

### 5. Smartphone-Related Factors
- The relationship between smartphone ownership/age of first smartphone and MHQ is complex
- May reflect:
  - Economic status (ability to afford smartphones)
  - Social connectivity (staying in touch with others)
  - Digital stress (excessive screen time)
- Requires further investigation to understand mechanisms

### 6. ShareHomeWith
- Living arrangements show moderate impact on mental health
- May interact with family relationship quality

---

## Visualization Files Generated

1. **xgboost_feature_importance.png** - Bar chart of XGBoost feature importances
2. **shap_summary_plot.png** - Comprehensive SHAP summary showing feature impacts
3. **shap_bar_plot.png** - Mean absolute SHAP values
4. **shap_dependence_RelationWithAdultFamily.png** - How family relationships affect predictions
5. **shap_dependence_UPF.Freq.png** - How UPF consumption affects predictions
6. **shap_dependence_Smartphone.ownership.png** - How smartphone ownership affects predictions
7. **shap_force_plot_struggling.png** - Explanation of a struggling individual's prediction
8. **shap_force_plot_succeeding.png** - Explanation of a succeeding individual's prediction
9. **confusion_matrix.png** - Heatmap of classification results
10. **roc_curve.png** - ROC curve showing model discrimination ability

---

## How to Read SHAP Plots

### SHAP Summary Plot
- **Y-axis**: Features ranked by importance
- **X-axis**: SHAP value (impact on prediction)
  - Positive values → Increases likelihood of "Succeeding"
  - Negative values → Increases likelihood of "Struggling"
- **Color**: Feature value
  - Red/Pink → High feature values
  - Blue → Low feature values

### SHAP Dependence Plots
- Show the relationship between feature values and their SHAP values
- Reveal non-linear relationships and interactions with other features
- Help understand how different levels of a feature affect predictions

### SHAP Force Plots
- Explain individual predictions
- Red arrows → Push prediction toward "Succeeding"
- Blue arrows → Push prediction toward "Struggling"
- Arrow length → Strength of impact

---

## Model Strengths & Limitations

### Strengths
✓ Good overall discrimination (ROC-AUC = 0.75)
✓ High precision for identifying "Succeeding" individuals (84%)
✓ Interpretable results through SHAP analysis
✓ Identifies clear actionable factors
✓ Cross-validation confirms model stability

### Limitations
✗ Moderate recall for "Struggling" class (64%)
✗ Lower precision for "Struggling" class (47%)
✗ Class imbalance (more succeeding than struggling)
✗ Causality cannot be inferred (this is observational data)
✗ Some features have missing values

---

## Recommendations

### For Researchers
1. Conduct longitudinal studies to establish causality
2. Investigate mechanisms linking UPF consumption to mental health
3. Explore family intervention programs
4. Study the role of digital technology more deeply

### For Policy Makers
1. **Prioritize family support programs** - The strongest factor
2. **Improve nutrition access** - Reduce UPF availability, increase whole foods
3. **Expand educational opportunities** - Long-term mental health benefits
4. **Promote physical activity** - Community exercise programs
5. **Address social isolation** - Programs for those with weak family ties

### For Healthcare Providers
1. Screen for family relationship quality in mental health assessments
2. Include nutrition counseling in mental health treatment
3. Recommend physical activity as part of mental health care
4. Consider social support interventions

---

## Technical Details

### Model Configuration
- **Algorithm**: XGBoost Classifier
- **Max Depth**: 5
- **Learning Rate**: 0.05
- **N Estimators**: 200
- **Scale Pos Weight**: 0.37 (to handle class imbalance)
- **Subsample**: 0.8
- **Colsample By Tree**: 0.8

### Data Preprocessing
- Categorical variables encoded using Label Encoding
- Missing values imputed (median for numerical, "Missing" category for categorical)
- 80-20 train-test split with stratification
- No feature scaling (not required for tree-based models)

---

## Files Saved

- `xgboost_mhq_model.pkl` - Trained XGBoost model
- `label_encoders.pkl` - Label encoders for categorical variables
- All visualization PNG files (10 files)

---

## Conclusion

This analysis reveals that **social relationships, particularly with family, are the most important lifestyle factor** influencing mental health in rural Tanzania. Combined with **nutrition (UPF consumption), education, and exercise**, these factors create a comprehensive picture of modifiable lifestyle determinants of mental well-being.

The XGBoost model achieves reasonable predictive performance (ROC-AUC = 0.75), and SHAP analysis provides clear, interpretable insights into which factors matter most and how they influence mental health outcomes.

**Most Actionable Finding**: Interventions targeting family relationship quality and reducing ultra-processed food consumption could have the largest positive impact on mental health in this population.

---

*Analysis completed: December 31, 2025*
*Model: XGBoost with SHAP interpretability*
*Dataset: Rural Tanzania Global Mind Data (n=5,095)*
