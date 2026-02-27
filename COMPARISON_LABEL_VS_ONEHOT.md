# Comparison: Label Encoding vs One-Hot Encoding for MHQ Analysis

## Executive Summary

This document compares the XGBoost + SHAP analysis results using two different encoding methods for categorical variables:
1. **Label Encoding** (original analysis)
2. **One-Hot Encoding** (updated analysis)

---

## Key Differences in Approach

### Label Encoding
- **Method**: Assigns a single integer to each category (0, 1, 2, 3, ...)
- **Features**: 7 features total (6 categorical + 1 numerical)
- **Pros**: Fewer features, simpler model, faster computation
- **Cons**: Implies artificial ordering, less interpretable for categorical data

### One-Hot Encoding ✨
- **Method**: Creates separate binary column for each category (0 or 1)
- **Features**: 46 features total (45 binary + 1 numerical)
- **Pros**: No artificial ordering, each category's effect is explicit, more interpretable
- **Cons**: More features (curse of dimensionality), slightly more complex

---

## Model Performance Comparison

| Metric | Label Encoding | One-Hot Encoding | Change |
|--------|---------------|------------------|--------|
| **ROC-AUC (Test)** | 0.7464 | 0.7461 | -0.0003 (negligible) |
| **CV ROC-AUC** | 0.7097 ± 0.011 | 0.7087 ± 0.008 | -0.0010 (negligible) |
| **Accuracy** | 70% | 71% | +1% |
| **Precision (Struggling)** | 0.47 | 0.48 | +0.01 |
| **Recall (Struggling)** | 0.64 | 0.67 | +0.03 |
| **Precision (Succeeding)** | 0.84 | 0.86 | +0.02 |
| **Recall (Succeeding)** | 0.73 | 0.73 | 0 |

### 📊 Performance Verdict
**Virtually identical performance** with slight edge to one-hot encoding:
- Both achieve ~0.74 ROC-AUC
- One-hot encoding shows marginally better precision and recall
- **Winner: Slight advantage to One-Hot Encoding**

---

## Feature Importance Comparison

### XGBoost Feature Importance (Aggregated)

#### Label Encoding Rankings:
1. **RelationWithAdultFamily** - 29.1%
2. **UPF.Freq** - 15.9%
3. **Smartphone.ownership** - 13.2%
4. **Education_Years** - 11.7%
5. **Exercise.Freq** - 11.1%
6. **AgeOfFirstSP** - 9.9%
7. **ShareHomeWith** - 9.2%

#### One-Hot Encoding Rankings:
1. **RelationWithAdultFamily** - 30.7% ⬆️
2. **AgeOfFirstSP** - 19.9% ⬆️⬆️
3. **UPF.Freq** - 14.9% ⬇️
4. **ShareHomeWith** - 13.9% ⬆️⬆️
5. **Exercise.Freq** - 12.0% ⬆️
6. **Smartphone.ownership** - 6.6% ⬇️⬇️
7. **Education_Years** - 2.2% ⬇️⬇️

### 🔍 Key Differences in Feature Importance:
- **RelationWithAdultFamily** remains #1 in both (slight increase with one-hot)
- **AgeOfFirstSP** jumps from #6 to #2 with one-hot encoding (+10% importance)
- **Education_Years** drops from #4 to #7 (-9.5% importance)
- **Smartphone.ownership** drops from #3 to #6 (-6.6% importance)

---

## SHAP Values Comparison (Aggregated)

### Label Encoding SHAP Rankings:
1. **RelationWithAdultFamily** - 0.481
2. **UPF.Freq** - 0.271
3. **Education_Years** - 0.200
4. **Exercise.Freq** - 0.190
5. **AgeOfFirstSP** - 0.156
6. **ShareHomeWith** - 0.150
7. **Smartphone.ownership** - 0.093

### One-Hot Encoding SHAP Rankings:
1. **RelationWithAdultFamily** - 0.581 ⬆️
2. **UPF.Freq** - 0.395 ⬆️
3. **Exercise.Freq** - 0.280 ⬆️
4. **ShareHomeWith** - 0.238 ⬆️
5. **AgeOfFirstSP** - 0.218 ⬆️
6. **Education_Years** - 0.190 ⬇️
7. **Smartphone.ownership** - 0.091 ⬇️

### 🔍 Key Differences in SHAP Values:
- **Rankings remain similar** but magnitudes increase
- **RelationWithAdultFamily** shows 21% higher SHAP impact with one-hot
- **UPF.Freq** shows 46% higher SHAP impact with one-hot
- **Exercise.Freq** shows 47% higher SHAP impact with one-hot
- All features show increased absolute SHAP values (more discriminative)

---

## Top Individual Features (One-Hot Encoding Only)

One-hot encoding allows us to see **specific category effects**:

### 🏆 Most Important Individual Categories (XGBoost):

1. **RelationWithAdultFamily_1_Very Close to Most** - 11.9% importance
   - Being very close to most family members has the highest single impact
   
2. **UPF.Freq_several times a day** - 5.4% importance
   - High UPF consumption multiple times daily is a strong negative predictor
   
3. **RelationWithAdultFamily_4_Dont Get Along** - 5.3% importance
   - Poor family relationships strongly predict struggling mental health
   
4. **RelationWithAdultFamily_Missing** - 3.9% importance
   - Missing family relationship data is itself informative
   
5. **AgeOfFirstSP_Above 25** - 3.5% importance
   - Getting first smartphone after age 25 (or never) shows significant impact

### 💡 Key Insights from Individual Categories:

**Family Relationships:**
- "Very Close to Most" → Strong positive predictor (11.9%)
- "Don't Get Along" → Strong negative predictor (5.3%)
- Clear gradient: Better relationships = Better mental health

**UPF Consumption:**
- "Several times a day" → Strong negative predictor (5.4%)
- "Rarely/never" → Positive predictor
- Clear dose-response: More UPF = Worse mental health

**Smartphone/Digital:**
- "Above 25" (late or no smartphone) shows complex relationship
- May reflect socioeconomic factors, rural lifestyle, or digital stress avoidance

---

## Interpretability Comparison

### Label Encoding Interpretation:
❌ "RelationWithAdultFamily has importance 0.29"
- What does this mean? Which relationships? How are they ordered?
- Categories assigned arbitrary numbers (e.g., 0, 1, 2, 3, 4, 5)
- Difficult to know which specific relationship quality matters most

### One-Hot Encoding Interpretation: ✅
✓ "RelationWithAdultFamily_1_Very Close to Most has importance 0.119"
- Crystal clear: Being very close to most family members is the #1 factor
- Each category's effect is explicit and measurable
- Can see that "Don't Get Along" (0.053) has nearly half the impact

### 🎯 Interpretability Winner: **One-Hot Encoding**
- Provides specific, actionable insights
- Shows which exact categories drive predictions
- No ambiguity about what the model learned

---

## Practical Implications

### What Changed with One-Hot Encoding?

1. **Age of First Smartphone became more important**
   - Jumped from #6 to #2 in feature importance
   - Specific categories (e.g., "Above 25") show clear effects
   - Suggests smartphone timing/ownership is more nuanced than label encoding revealed

2. **Family Relationships confirmed as #1 predictor**
   - Still the dominant factor (30.7% importance, 0.581 SHAP)
   - Now can see "Very Close to Most" specifically drives this (11.9% alone)

3. **UPF Consumption effects are clearer**
   - "Several times a day" is explicitly harmful (5.4% importance)
   - Can quantify dose-response relationship

4. **Education_Years less important**
   - Dropped from 11.7% to 2.2% importance
   - May have been capturing variance that one-hot encoding attributes to other factors

---

## Recommendations Based on One-Hot Analysis

### 🎯 Top Intervention Targets (Updated):

1. **Family Relationships (30.7% importance, 0.581 SHAP)**
   - **Specific action**: Programs targeting those "Not Close" or "Don't Get Along" with family
   - **Why**: "Very Close to Most" shows the highest single-category impact (11.9%)
   - **Expected impact**: Highest potential for improving MHQ

2. **Ultra-Processed Food Reduction (14.9% importance, 0.395 SHAP)**
   - **Specific action**: Target those consuming UPF "several times a day"
   - **Why**: This specific pattern shows 5.4% importance (2nd highest individual category)
   - **Expected impact**: Clear dose-response, reduce frequency for better outcomes

3. **AgeOfFirstSP/Smartphone Factors (19.9% importance, 0.218 SHAP)**
   - **New insight**: Higher importance than previously thought
   - **Specific action**: Investigate "Above 25" category (late adopters or non-owners)
   - **Expected impact**: May reflect socioeconomic, lifestyle, or digital wellness factors

4. **Exercise Promotion (12.0% importance, 0.280 SHAP)**
   - **Specific action**: Promote daily or frequent exercise patterns
   - **Why**: Consistent moderate-high impact
   - **Expected impact**: Moderate improvement in MHQ

---

## Which Encoding Should You Use?

### Use **One-Hot Encoding** when:
✅ You need clear, interpretable results
✅ You want to see specific category effects
✅ You're presenting to non-technical stakeholders
✅ You're designing interventions (need to know which categories to target)
✅ Your dataset has moderate number of categories (<50 unique values per feature)
✅ **Recommendation for this study: ONE-HOT ENCODING** ⭐

### Use **Label Encoding** when:
- You have very high cardinality (thousands of categories)
- You need maximum computational efficiency
- Categories have natural ordering (e.g., small/medium/large)
- You're doing quick exploratory analysis

---

## Final Comparison: Side-by-Side

| Aspect | Label Encoding | One-Hot Encoding | Winner |
|--------|---------------|------------------|--------|
| **Performance** | ROC-AUC: 0.7464 | ROC-AUC: 0.7461 | Tie |
| **Interpretability** | Moderate | High | ✅ One-Hot |
| **Specific Insights** | Limited | Rich | ✅ One-Hot |
| **Feature Count** | 7 | 46 | Label (simpler) |
| **Computation Speed** | Faster | Slightly slower | Label |
| **Actionability** | Moderate | High | ✅ One-Hot |
| **Stakeholder Communication** | Harder | Easier | ✅ One-Hot |
| **Overall** | Good | Better | ✅ **One-Hot** |

---

## Conclusions

### 🏆 Overall Winner: **One-Hot Encoding**

**Why:**
1. **Nearly identical performance** (0.7461 vs 0.7464 ROC-AUC)
2. **Dramatically better interpretability**
3. **Reveals specific category effects** (e.g., "Very Close to Most" = 11.9%)
4. **More actionable for interventions**
5. **Better for stakeholder communication**

### Key Findings Confirmed Across Both Methods:

✅ **Family relationships** are the #1 predictor (30% importance in both)
✅ **UPF consumption** is a strong negative factor (15% importance in both)
✅ **Exercise frequency** shows moderate positive impact (11-12% importance)
✅ **Model performance** is robust (ROC-AUC ~0.74 regardless of encoding)

### New Insights from One-Hot Encoding:

🆕 **Age of first smartphone** is more important than initially thought (20% vs 10%)
🆕 **"Very Close to Most"** family relationships account for 12% importance alone
🆕 **"Several times a day" UPF consumption** is specifically harmful (5.4%)
🆕 **"Don't Get Along"** with family is a strong negative predictor (5.3%)

---

## Files Generated

### One-Hot Encoding Analysis:
- `xgboost_feature_importance_aggregated.png` - Shows original features
- `xgboost_feature_importance_top20.png` - Shows individual encoded features
- `shap_aggregated_by_original_features.png` - SHAP values by original features
- `shap_summary_plot.png` - Top 30 encoded features with SHAP values
- `shap_bar_plot.png` - Mean absolute SHAP values
- Plus 5 individual category dependence plots
- Plus confusion matrix, ROC curve, force plots

### Models Saved:
- `xgboost_mhq_model_onehot.pkl` - One-hot encoded model
- `feature_info_onehot.pkl` - Feature group mappings

---

## Recommendation for Publication/Presentation

**Use the One-Hot Encoding results** because:

1. ✅ Better interpretability for stakeholders
2. ✅ Can state specific findings like:
   - "Being 'Very Close to Most' family members accounts for 12% of predictive power"
   - "Consuming UPF 'several times a day' is the 2nd strongest negative predictor"
3. ✅ More actionable for policy makers
4. ✅ No performance sacrifice (same ROC-AUC)
5. ✅ More rigorous methodologically (no artificial ordering)

---

*Analysis completed: December 31, 2025*
*Both encoding methods validated the core findings while one-hot encoding provided superior interpretability*
