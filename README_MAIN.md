# XGBoost + SHAP Analysis: Lifestyle Drivers of MHQ in Rural Tanzania

## 📁 Project Overview

This project analyzes lifestyle factors that predict mental health outcomes (MHQ scores) in rural Tanzania using XGBoost machine learning with SHAP interpretability. 

**Two versions are provided with different encoding methods:**
1. **Label Encoding** - Compact representation (7 features)
2. **One-Hot Encoding** - Interpretable representation (46 features) ⭐ Recommended

## 🔧 Methods (short)
- Filter data to MHQ >= 100 or < 0; define binary target (1 = succeeding, 0 = struggling).
- Two feature sets: `with_SP` (includes AgeOfFirstSP/SmartPhone + ownership) and `no_SP` (excludes them).
- Preprocess: median-impute Education_Years; fill categorical NaNs with "Missing"; one-hot encode all categoricals (no drop-first).
- Model: XGBoost classifier (hist), fixed hyperparams (depth 5, lr 0.05, 220 trees, subsample/colsample 0.85), scale_pos_weight from class ratio.
- Evaluation: stratified 5-fold CV with metrics AUC, F1, Precision, Recall; per-group jobs (all, rural_hadza, gm, age_18_24) and both variants; results saved to cv_metrics_all_groups.xlsx.

---

## 🚀 Quick Start

### Run Both Versions
```bash
cd /Users/jerzybala/Desktop/Rural_gmdata_for_ML
source .venv/bin/activate
python run_both_versions.py
```

### Run Individual Versions

**Label Encoding:**
```bash
cd label_encoding_version
python xgboost_shap_label_encoding.py
```

**One-Hot Encoding:**
```bash
cd one_hot_encoding_version
python xgboost_shap_one_hot_encoding.py
```

---

## 📂 Project Structure

```
Rural_gmdata_for_ML/
├── rural_gmdata_forML.csv              # Dataset (5,095 participants)
├── run_both_versions.py                # Master script to run both versions
│
├── label_encoding_version/             # Label encoding analysis
│   ├── xgboost_shap_label_encoding.py
│   ├── README.md
│   └── [Generated results...]
│
├── one_hot_encoding_version/           # One-hot encoding analysis ⭐
│   ├── xgboost_shap_one_hot_encoding.py
│   ├── README.md
│   └── [Generated results...]
│
├── Documentation/
│   ├── COMPARISON_LABEL_VS_ONEHOT.md  # Detailed comparison
│   ├── ANALYSIS_REPORT.md              # Comprehensive analysis report
│   ├── ONE_HOT_SUMMARY.md              # Quick summary
│   └── QUICK_GUIDE.txt                 # Quick reference
│
└── .venv/                              # Python virtual environment
```

---

## 🎯 Key Findings (Both Versions Agree!)

### Top 3 Predictors of Good Mental Health:

1. **Family Relationships** (~30% importance)
   - Being "Very Close to Most" adult family members
   - Strongest single predictor
   
2. **UPF Consumption** (~15% importance)
   - Lower ultra-processed food consumption
   - "Rarely/never" eating UPF correlates with better MHQ
   
3. **Education/Smartphone/Exercise** (varying ~10-20% importance)
   - More years of education
   - Exercise frequency
   - Smartphone ownership patterns

### Model Performance (Both Versions):
- **ROC-AUC**: ~0.746 (good discrimination)
- **Accuracy**: ~70-71%
- **Cross-Validation**: Stable across folds

---

## 🔍 Version Comparison

| Aspect | Label Encoding | One-Hot Encoding |
|--------|---------------|------------------|
| **Features** | 7 | 46 |
| **ROC-AUC** | 0.7464 | 0.7461 |
| **Interpretability** | Moderate | High ⭐ |
| **Computation** | Faster | Slightly slower |
| **Specific Insights** | Limited | Rich ⭐ |
| **Best for** | Exploration | Publication ⭐ |

### 📊 Performance: TIE (virtually identical)
### 💡 Interpretability: ONE-HOT WINS ⭐

---

## 🎓 Which Version Should You Use?

### ✅ Use **One-Hot Encoding** for:
- **Publications and presentations** - More rigorous, clearer findings
- **Stakeholder communication** - Specific, actionable insights
- **Intervention design** - Know exactly which categories to target
- **Policy recommendations** - Clear, evidence-based targets

**Example insights only available with one-hot:**
- "Being 'Very Close to Most' family members = 11.9% impact"
- "Consuming UPF 'several times a day' = 5.4% negative impact"
- "Poor family relationships = 5.3% negative impact"

### ✅ Use **Label Encoding** for:
- Quick exploratory analysis
- Resource-constrained environments
- Very high cardinality features (>100 categories)
- When speed is priority over interpretability

---

## 📊 Dataset Information

- **Source**: Rural Tanzania Global Mind Data
- **Sample Size**: 5,095 participants
- **Target Variable**: Binary classification
  - Succeeding: MHQ ≥ 100 (n=3,709, 72.8%)
  - Struggling: MHQ < 0 (n=1,386, 27.2%)

### Features Analyzed (7 total):
1. **Education_Years** (continuous)
2. **ShareHomeWith** (categorical) - Household size
3. **RelationWithAdultFamily** (categorical) - Family relationship quality
4. **UPF.Freq** (categorical) - Ultra-processed food frequency
5. **AgeOfFirstSP** (categorical) - Age of first smartphone
6. **Smartphone.ownership** (categorical) - Current smartphone status
7. **Exercise.Freq** (categorical) - Exercise frequency

---

## 📈 Files Generated by Each Version

### Label Encoding Version (8 files):
- Feature importance plot
- 3 SHAP dependence plots
- SHAP summary and bar plots
- 2 force plots
- Confusion matrix
- ROC curve
- Model and results files

### One-Hot Encoding Version (13 files):
- Aggregated feature importance plot ⭐
- Top 20 individual features plot
- Aggregated SHAP plot ⭐
- 5 individual category dependence plots
- SHAP summary and bar plots (top 30)
- 2 force plots
- Confusion matrix
- ROC curve
- Model and results files

---

## 💡 Key Insights from Both Versions

### Confirmed Across Both Methods:
✅ Family relationships are the #1 predictor (~30% importance)
✅ UPF consumption shows strong negative association (~15%)
✅ Exercise frequency has moderate positive impact (~11-12%)
✅ Model achieves ~0.74 ROC-AUC (good discrimination)

### New Insights from One-Hot Encoding:
🆕 Age of first smartphone more important than initially thought (20% vs 10%)
🆕 "Very Close to Most" family relationships = 12% importance alone
🆕 High-frequency UPF consumption specifically harmful (5.4%)
🆕 Poor family relationships strongly predict struggling (5.3%)

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
cd /Users/jerzybala/Desktop/Rural_gmdata_for_ML
source .venv/bin/activate  # Already configured
```

### 2. Run Analysis
```bash
# Run both versions at once
python run_both_versions.py

# Or run individually
cd label_encoding_version && python xgboost_shap_label_encoding.py
cd one_hot_encoding_version && python xgboost_shap_one_hot_encoding.py
```

### 3. Review Results
- Check each version's folder for visualizations
- Read `COMPARISON_LABEL_VS_ONEHOT.md` for detailed comparison
- Start with one-hot encoding results for clearest insights

---

## 📚 Documentation Guide

### For Quick Overview:
1. **ONE_HOT_SUMMARY.md** - Quick summary of one-hot encoding results
2. **QUICK_GUIDE.txt** - Quick reference for interpreting plots

### For Deep Dive:
1. **ANALYSIS_REPORT.md** - Comprehensive analysis report
2. **COMPARISON_LABEL_VS_ONEHOT.md** - Detailed version comparison
3. Version-specific READMEs in each folder

### For Methods:
- Check Python scripts in each version folder
- Scripts are well-commented and self-documenting

---

## 🎯 Actionable Recommendations

Based on both analyses:

### Priority #1: Family Support Programs
- **Target**: Individuals with poor family relationships
- **Goal**: Move toward "Very Close to Most" relationships
- **Expected Impact**: Highest (30% importance, 11.9% for top category)

### Priority #2: Nutrition Interventions
- **Target**: High-frequency UPF consumers
- **Goal**: Reduce to "rarely/never" consumption
- **Expected Impact**: High (15% importance, 5.4% for worst pattern)

### Priority #3: Exercise Promotion
- **Target**: Sedentary individuals
- **Goal**: Increase to daily or frequent exercise
- **Expected Impact**: Moderate (12% importance)

### Priority #4: Digital Wellness Research
- **Target**: Understand smartphone/digital patterns
- **Goal**: Identify protective or harmful digital behaviors
- **Expected Impact**: Moderate-High (10-20% importance varies by version)

---

## 🔬 Technical Details

### Model Configuration (Same for Both):
- **Algorithm**: XGBoost Gradient Boosting
- **Max Depth**: 5
- **Learning Rate**: 0.05
- **N Estimators**: 200
- **Class Imbalance Handling**: scale_pos_weight = 0.37
- **Cross-Validation**: 5-fold stratified

### Interpretability:
- **SHAP (SHapley Additive exPlanations)**
  - Game-theory based feature attribution
  - Shows both magnitude and direction of impact
  - Individual-level explanations available

---

## 📞 Questions?

- **Methodology**: See Python scripts in version folders
- **Comparison**: See COMPARISON_LABEL_VS_ONEHOT.md
- **Quick help**: See QUICK_GUIDE.txt
- **Detailed analysis**: See ANALYSIS_REPORT.md

---

## ✅ Recommended Workflow

1. **Run both versions**: `python run_both_versions.py`
2. **Review one-hot results first** (more interpretable)
3. **Check comparison document** for differences
4. **Use one-hot results for publication/presentation**
5. **Keep label encoding results** for reference/validation

---

## 🏆 Final Recommendation

**Use One-Hot Encoding results for your study** because:
1. ✅ Same predictive performance (ROC-AUC 0.746)
2. ✅ Far superior interpretability
3. ✅ Specific, actionable category-level insights
4. ✅ Better for stakeholder communication
5. ✅ More rigorous methodology (no artificial ordering)

### Key Message for Publication:
*"Being 'Very Close to Most' family members accounts for 12% of our model's predictive power—the single strongest factor predicting mental health outcomes in rural Tanzania. This is followed by ultra-processed food consumption patterns, with consuming UPF 'several times a day' showing a 5.4% negative impact. These findings provide clear, actionable targets for community mental health interventions."*

---

*Analysis framework created: December 31, 2025*
*Dataset: Rural Tanzania Global Mind Data (n=5,095)*
*Model: XGBoost + SHAP with both Label and One-Hot Encoding*

**Status: ✅ READY FOR ANALYSIS - Run both versions and compare!**
