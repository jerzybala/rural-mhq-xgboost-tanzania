<<<<<<< HEAD
# XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Rural Tanzania

## 📁 Project Overview

This project analyzes lifestyle factors that predict mental health outcomes (MHQ scores) in rural Tanzania using XGBoost machine learning with SHAP (SHapley Additive exPlanations) interpretability.

**Research Question**: What lifestyle factors best predict whether individuals in rural Tanzania will have good mental health (Succeeding: MHQ ≥ 100) vs poor mental health (Struggling: MHQ < 0)?


## 🎯 Quick Start

### View Results (Start Here!)
1. **SUMMARY_INFOGRAPHIC.png** - Visual overview of all key findings (best for presentations)
2. **ANALYSIS_REPORT.md** - Comprehensive written report with interpretations
3. **QUICK_GUIDE.txt** - Quick reference for interpreting visualizations

### Run Analysis
```bash
cd /Users/jerzybala/Desktop/Rural_gmdata_for_ML
.venv/bin/python xgboost_shap_analysis.py
```


## 📊 Key Findings Summary

### Top 3 Predictors of Good Mental Health:

1. **Family Relationships** (29.1% importance)
   - Being "Very Close to Most" adult family members
   - Strongest single predictor
   - Actionable through family therapy/support programs

2. **UPF Consumption** (15.9% importance)
   - Lower ultra-processed food consumption
   - "Rarely/never" eating UPF correlates with better MHQ
   - Actionable through nutrition education

3. **Education** (20% actual SHAP impact)
   - More years of formal education
   - Protective through multiple pathways
   - Actionable through expanding educational access

### Model Performance:


## 📂 File Structure

### 📄 Documentation

### 🐍 Python Scripts

### 📊 Visualizations (10 files)

#### Model Performance

#### SHAP Analysis (Interpretability)

### 📁 Data


## 🔬 Methodology

### Dataset
  - Succeeding: MHQ ≥ 100 (n=3,709, 72.8%)
  - Struggling: MHQ < 0 (n=1,386, 27.2%)

### Features Analyzed (7 total)
1. **Education_Years** (continuous) - Years of formal education
2. **ShareHomeWith** (categorical) - Number of people in household
3. **RelationWithAdultFamily** (categorical) - Quality of family relationships
4. **UPF.Freq** (categorical) - Ultra-processed food consumption frequency
5. **AgeOfFirstSP** (categorical) - Age of first smartphone ownership
6. **Smartphone.ownership** (categorical) - Current smartphone ownership
7. **Exercise.Freq** (categorical) - Exercise frequency

### Model
  - Max depth: 5
  - Learning rate: 0.05
  - N estimators: 200
  - Scale pos weight: 0.37 (to handle class imbalance)

### Interpretability
  - Game-theory based approach to explain predictions
  - Shows both feature importance and direction of impact
  - Provides individual-level explanations


## 💡 Main Insights

### 1. Social Factors Dominate
**Family relationships** are the #1 predictor, accounting for 29% of model importance and having the highest SHAP impact (0.48). This suggests:

### 2. Nutrition-Mental Health Link
**Ultra-processed food consumption** is the #2 predictor (16% importance). Lower UPF consumption correlates with better mental health, suggesting:

### 3. Education as Protection
**Years of education** shows 20% actual SHAP impact. More education predicts better mental health through:

### 4. Lifestyle is Modifiable
**Exercise frequency** shows moderate positive impact. Combined with nutrition and social factors, suggests:

### 5. Complex Smartphone Relationship
Smartphone ownership shows moderate but complex relationship with MHQ:


## 🎨 How to Read the Visualizations

### SHAP Summary Plot (shap_summary_plot.png) ⭐ MOST IMPORTANT
  - Left (negative) = Pushes toward "Struggling"
  - Right (positive) = Pushes toward "Succeeding"
  - Red/Pink = High value
  - Blue = Low value

**Example**: For RelationWithAdultFamily, red dots (good relationships) are mostly on the right (predicting Succeeding), while blue dots (poor relationships) are mostly on the left (predicting Struggling).

### SHAP Dependence Plots

### SHAP Force Plots


## 📈 Model Performance Details

### Classification Metrics
|  | Precision | Recall | F1-Score | Support |
|---|-----------|--------|----------|---------|
| Struggling (MHQ<0) | 0.47 | 0.64 | 0.54 | 277 |
| Succeeding (MHQ≥100) | 0.84 | 0.73 | 0.78 | 742 |
| **Overall** | **0.74** | **0.70** | **0.72** | **1,019** |

### ROC-AUC

### Confusion Matrix
|  | Predicted Struggling | Predicted Succeeding |
|---|---------------------|---------------------|
| **Actual Struggling** | 177 (TP) | 100 (FN) |
| **Actual Succeeding** | 201 (FP) | 541 (TN) |


## 🚀 Recommendations

### For Policy Makers
1. **Priority**: Family support and community bonding programs
2. Improve nutrition access (reduce UPF, increase whole foods)
3. Expand educational opportunities in rural areas
4. Promote physical activity through community programs
5. Address social isolation systematically

### For Healthcare Providers
1. Screen for family relationship quality in mental health assessments
2. Include nutrition counseling in mental health treatment
3. Recommend physical activity as part of mental health care
4. Consider social support interventions for isolated individuals
5. Take holistic approach considering multiple lifestyle factors

### For Researchers
1. Conduct longitudinal studies to establish causality
2. Investigate mechanisms linking UPF to mental health
3. Test family intervention programs in RCTs
4. Explore interaction effects between factors
5. Study digital technology's role more deeply


## 🔧 Technical Requirements

### Python Packages

### Installation
```bash
# Create virtual environment (already done)
python -m venv .venv

# Activate environment
source .venv/bin/activate  # macOS/Linux

# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```


## 📝 Citation

If you use this analysis or findings, please cite:

```
XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Rural Tanzania
Dataset: Rural Tanzania Global Mind Data (n=5,095)
Model: XGBoost Classifier with SHAP interpretability
Date: December 31, 2025
```


## 🤝 Next Steps

1. **Review Findings**
   - Start with SUMMARY_INFOGRAPHIC.png
   - Read ANALYSIS_REPORT.md for details
   - Examine SHAP visualizations

2. **Share Results**
   - Present infographic to stakeholders
   - Discuss policy implications
   - Plan intervention studies

3. **Further Analysis**
   - Longitudinal follow-up study
   - RCTs for top interventions
   - Mechanism investigations

4. **Implementation**
   - Pilot family support programs
   - Nutrition education initiatives
   - Educational expansion
   - Community exercise programs


## ❓ Questions or Issues?

For questions about:


## 📜 License & Acknowledgments

This analysis was conducted to understand lifestyle drivers of mental health in rural Tanzania using machine learning and interpretable AI techniques.

**Key Strengths**:
✓ Large sample size (n=5,095)
✓ Interpretable machine learning (SHAP)
✓ Clear actionable findings
✓ Multiple modifiable factors identified

**Limitations**:
⚠ Observational data (cannot prove causality)
⚠ Cross-sectional design
⚠ Class imbalance
⚠ Some missing data


*Analysis completed: December 31, 2025*

*This project demonstrates the power of combining machine learning (XGBoost) with interpretability tools (SHAP) to generate actionable insights from health data.*
=======

# XGBoost Analysis of Tanzanian MHQ Data


## Project Background
This project investigates the lifestyle drivers of Mental Health Quotient (MHQ) in rural Tanzania using machine learning. 




This project provides automated, reproducible performance analysis of XGBoost classifiers for mental health questionnaire (MHQ) data, supporting group and variant comparisons.

## Main Script

- **xgb_performance_table_generator.py**
  - Loops through all group/variant combinations
  - Applies one-hot encoding and preprocessing
  - Runs 5-fold cross-validation for each combination
  - Computes and exports:
    - Summary metrics (mean/std for AUC, accuracy, F1, precision, recall, specificity, balanced accuracy, MCC, Cohen’s kappa) to `xgb_performance_table.xlsx`
    - Per-fold metrics to `xgb_performance_table_per_fold.xlsx`
    - Confusion matrices for each fold to `xgb_confusion_matrices.xlsx`
    - Feature importances (mean/std across folds) to `xgb_feature_importances.xlsx`

## Data
- Input: `rural_gmdata_forML.csv`
- Target: Binary classification (Succeeding vs. Struggling, based on Overall.MHQ)

## Usage
1. Activate your Python environment:
   ```bash
   source /path/to/venv/bin/activate
   ```
2. Run the main script:
   ```bash
   python xgb_performance_table_generator.py
   ```
3. Results will be saved as Excel files in the same directory.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, openpyxl

Install requirements (if needed):
```bash
pip install pandas numpy scikit-learn xgboost openpyxl
```

## Output Files
- `xgb_performance_table.xlsx`: Summary metrics (mean/std) for each group/variant
- `xgb_performance_table_per_fold.xlsx`: All per-fold metrics
- `xgb_confusion_matrices.xlsx`: Confusion matrices for each fold
- `xgb_feature_importances.xlsx`: Feature importances (mean/std across folds)

## Customization
- Edit `VARIANTS` and `GROUPS` in the script to change analysis scope.
- Adjust model parameters or metrics as needed.

## Contributors
- Jerzy Bala (Project Maintainer, jerzy@sapienlabs.org)
- Tara Thiagarajan
- Dhanya Parameshwaran

For questions or contributions, contact the project maintainer.
>>>>>>> 167d686ebb2cd62b0b7404cea62ce95c5635d7be
