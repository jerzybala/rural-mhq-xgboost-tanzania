# Comparison: SMOTE vs LightGBM vs Baseline XGBoost

## Executive Summary

Three different approaches were tested to improve mental health prediction:
1. **Baseline XGBoost** (one-hot encoding)
2. **SMOTE + XGBoost** (class balancing via synthetic oversampling)
3. **LightGBM** (modern gradient boosting algorithm)

## 📊 Performance Comparison

| Metric | Baseline XGBoost | SMOTE + XGBoost | LightGBM | Best |
|--------|------------------|-----------------|----------|------|
| **Test ROC-AUC** | **0.7461** | 0.7384 | 0.7423 | ✅ Baseline |
| **CV ROC-AUC** | 0.7155 ± 0.009 | **0.8638 ± 0.095** | 0.7072 ± 0.012 | ✅ SMOTE |
| **Accuracy** | 68% | **74%** | 70% | ✅ SMOTE |
| **Precision (Struggling)** | 0.44 | **0.53** | 0.47 | ✅ SMOTE |
| **Recall (Struggling)** | 0.66 | 0.44 | **0.66** | ✅ Baseline/LightGBM |
| **F1-Score (Struggling)** | 0.53 | 0.48 | **0.55** | ✅ LightGBM |
| **Balanced Accuracy** | 0.68 | 0.65 | **0.69** | ✅ LightGBM |

### 🏆 Winner by Category

- **Best Overall ROC-AUC**: Baseline XGBoost (0.7461)
- **Best Precision**: SMOTE + XGBoost (0.53)
- **Best Recall**: Baseline XGBoost & LightGBM (0.66)
- **Best F1-Score**: LightGBM (0.55)
- **Most Stable CV**: LightGBM (lowest std: 0.012)

## 🔍 Detailed Analysis

### 1. Baseline XGBoost (One-Hot Encoding)
**Folder**: `one_hot_encoding_version/`

**Performance**:
- ROC-AUC: 0.7461
- Precision (Struggling): 0.44
- Recall (Struggling): 0.66

**Strengths**:
- ✅ Highest test ROC-AUC
- ✅ High recall (catches 66% of struggling individuals)
- ✅ Simple and interpretable

**Weaknesses**:
- ❌ Low precision (many false positives)
- ❌ Class imbalance not addressed

**Top Features**:
1. RelationWithAdultFamily (40%)
2. UPF.Freq (19%)
3. Exercise.Freq (14%)

---

### 2. SMOTE + XGBoost
**Folder**: `smote_xgboost_version/`

**Performance**:
- ROC-AUC: 0.7384 (↓0.0077 from baseline)
- Precision (Struggling): 0.53 (↑0.09 from baseline)
- Recall (Struggling): 0.44 (↓0.22 from baseline)

**Strengths**:
- ✅ **Best precision** (0.53) - fewer false positives
- ✅ **Highest CV score** (0.86) - BUT high variance (±0.095)
- ✅ Balanced training data (50-50 split)
- ✅ Better accuracy (74%)

**Weaknesses**:
- ❌ **Much lower recall** (44% vs 66%) - misses more struggling individuals
- ❌ High CV variance suggests overfitting to synthetic data
- ❌ Test performance worse than CV (red flag)

**SMOTE Effect**:
- Created 1,858 synthetic samples
- Training set: 1,109 → 2,967 struggling samples
- Balanced classes improved precision but hurt recall

**Top Features** (Different ranking!):
1. RelationWithAdultFamily (23%)
2. ShareHomeWith (17%)
3. AgeOfFirstSP (17%)

**⚠️ Key Issue**: The huge gap between CV (0.86) and test (0.74) ROC-AUC suggests SMOTE may be causing overfitting. The synthetic samples don't fully represent real struggling cases.

---

### 3. LightGBM
**Folder**: `lightgbm_version/`

**Performance**:
- ROC-AUC: 0.7423 (↓0.0038 from baseline)
- Precision (Struggling): 0.47 (↑0.03 from baseline)
- Recall (Struggling): 0.66 (same as baseline)

**Strengths**:
- ✅ **Best F1-score** (0.55) - good balance
- ✅ **Best balanced accuracy** (0.69)
- ✅ **Most stable CV** (±0.012 std)
- ✅ Built-in imbalance handling
- ✅ Matches baseline recall while improving precision

**Weaknesses**:
- ❌ Slightly lower ROC-AUC than baseline
- ❌ Lower CV mean (0.71 vs 0.72 baseline)

**Top Features** (Very different!):
1. Education_Years (20%)
2. ShareHomeWith (19%)
3. Exercise.Freq (18%)

**Note**: LightGBM found Education_Years most important, while XGBoost ranked it least important. This suggests different feature interaction patterns.

---

## 🎯 Recommendation

### Best Choice: **Baseline XGBoost** (one_hot_encoding_version)

**Reasons**:
1. **Highest test ROC-AUC** (0.7461)
2. **Highest recall** (0.66) - critical for identifying struggling individuals
3. **Simplest approach** - no synthetic data or complex preprocessing
4. **Most reliable** - no overfitting concerns

### When to Use Alternatives:

**Use SMOTE + XGBoost if**:
- False positives are very costly
- You need higher precision (53% vs 44%)
- You can afford to miss more struggling cases (44% recall)
- You have validation data to confirm synthetic samples are realistic

**Use LightGBM if**:
- You need the most balanced performance (F1 = 0.55)
- Training speed matters (faster than XGBoost)
- You want the most stable cross-validation
- You prefer built-in class imbalance handling

## 📈 Performance Trade-offs

### Precision vs Recall Trade-off

| Model | Precision | Recall | Interpretation |
|-------|-----------|--------|----------------|
| Baseline | 0.44 | **0.66** | Catches more struggling people, but more false alarms |
| SMOTE | **0.53** | 0.44 | Fewer false alarms, but misses more struggling people |
| LightGBM | 0.47 | **0.66** | Good balance - catches as many as baseline with better precision |

**For mental health screening**: High recall is typically more important than precision. It's better to flag more people for follow-up (even with false positives) than to miss struggling individuals.

## 🔬 Feature Importance Comparison

### Top 3 Features by Model:

**Baseline XGBoost**:
1. RelationWithAdultFamily (40%)
2. UPF.Freq (19%)
3. Exercise.Freq (14%)

**SMOTE + XGBoost**:
1. RelationWithAdultFamily (23%)
2. ShareHomeWith (17%)
3. AgeOfFirstSP (17%)

**LightGBM**:
1. Education_Years (20%)
2. ShareHomeWith (19%)
3. Exercise.Freq (18%)

### Key Insight:
- **RelationWithAdultFamily** is consistently important (top 1 or 2)
- **SMOTE redistributed importance** more evenly (23% vs 40% for top feature)
- **LightGBM found Education_Years critical**, while XGBoost models ranked it last

This suggests:
- Family relationships are universally important for mental health
- SMOTE may dilute feature importance by spreading signal across synthetic samples
- Different algorithms capture different feature interactions

## 💡 Why Baseline Won

Despite trying two improvement strategies:

1. **SMOTE created unrealistic patterns**:
   - CV score (0.86) much higher than test (0.74)
   - Synthetic samples don't capture true struggling cases
   - Overfitting to interpolated data

2. **LightGBM's advantages didn't materialize**:
   - Only marginal precision gain (+0.03)
   - Lower overall ROC-AUC
   - Dataset may be too small for LightGBM's strengths

3. **Baseline is already near-optimal**:
   - Class imbalance handled via scale_pos_weight
   - Simple model prevents overfitting
   - One-hot encoding provides good interpretability

## 📁 Folder Structure

```
Rural_gmdata_for_ML/
├── one_hot_encoding_version/        ← ✅ RECOMMENDED
│   ├── xgboost_shap_one_hot_encoding.py
│   ├── xgboost_model_onehot.pkl
│   └── 14 visualization PNG files
│
├── smote_xgboost_version/
│   ├── smote_xgboost_analysis.py
│   ├── xgboost_model_smote.pkl
│   └── 14 visualization PNG files
│
└── lightgbm_version/
    ├── lightgbm_analysis.py
    ├── lightgbm_model.pkl
    └── 14 visualization PNG files
```

## 🎓 Lessons Learned

1. **Simple is often better**: Baseline XGBoost outperformed sophisticated approaches
2. **SMOTE isn't always helpful**: Can create unrealistic synthetic patterns
3. **Different algorithms, different stories**: Feature importance varies significantly
4. **Class imbalance handling**: Built-in `scale_pos_weight` was sufficient
5. **Dataset size matters**: 5,095 samples may be too small for complex methods to shine

## 🔮 Future Improvements

Since current approaches hit a ceiling (~0.74 ROC-AUC), consider:

1. **Feature engineering**:
   - Interaction terms (RelationWithAdultFamily × ShareHomeWith)
   - Polynomial features for Education_Years
   - Domain-specific combinations

2. **Ensemble methods**:
   - Stack baseline + LightGBM predictions
   - Voting classifier across multiple models

3. **Data collection**:
   - More samples, especially struggling cases
   - Additional predictive features (sleep, diet, stress)
   - Longitudinal data (track changes over time)

4. **Threshold optimization**:
   - Current threshold: 0.5
   - Optimize for specific precision/recall balance
   - ROC curve analysis for optimal cutoff

---

## Summary Table: Quick Reference

| Aspect | Baseline | SMOTE | LightGBM |
|--------|----------|-------|----------|
| ROC-AUC | **0.7461** ⭐ | 0.7384 | 0.7423 |
| Best for | **Overall performance** | Precision | F1/Balance |
| Training time | Medium | Slow | **Fast** |
| Interpretability | ✅ High | ⚠️ Medium | ✅ High |
| Overfitting risk | ✅ Low | ❌ High | ✅ Low |
| Recommendation | **USE THIS** | Special cases | Alternative |

---

**Final Verdict**: Stick with **Baseline XGBoost** (`one_hot_encoding_version`) for production use. It provides the best overall performance with the least complexity.
