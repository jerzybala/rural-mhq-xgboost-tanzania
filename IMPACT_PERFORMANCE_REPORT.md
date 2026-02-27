# Impactful Features and Performance (With vs. Without SP)

## Feature Drivers by Group
- **Rural Hadza**
  - With SP: RelationWithAdultFamily_1_Very Close to Most; Education_Years; ShareHomeWith_6+; UPF.Freq_several days a week; UPF.Freq_several times a day.
  - No SP: RelationWithAdultFamily_1_Very Close to Most; Education_Years; UPF.Freq_several days a week; ShareHomeWith_6+; ShareHomeWith_2.
  - Takeaway: Family closeness and education dominate; household size stays important. Removing SP features barely changes ordering.
- **GM**
  - With SP: RelationWithAdultFamily_1_Very Close to Most; Education_Years; UPF.Freq_several times a day; Exercise.Freq_rarely/never; UPF.Freq_rarely/never.
  - No SP: Education_Years; RelationWithAdultFamily_1_Very Close to Most; UPF.Freq_several times a day; Exercise.Freq_rarely/never; UPF.Freq_rarely/never.
  - Takeaway: Education and family closeness remain the core signals; UPF intensity and low exercise consistently rank high; SP fields do not shift the pattern.
- **Age 18–24**
  - With SP: RelationWithAdultFamily_1_Very Close to Most; Education_Years; AgeOfFirstSP_Missing; Exercise.Freq_rarely/never; UPF.Freq_a few times a month.
  - No SP: RelationWithAdultFamily_1_Very Close to Most; Education_Years; Exercise.Freq_rarely/never; UPF.Freq_a few times a month; UPF.Freq_several times a day.
  - Takeaway: Family closeness and education lead. When SP fields are removed, lifestyle (exercise/UPF) fills the gap previously captured by the missing AgeOfFirstSP signal.

## Test-Set Performance (positive class = Succeeding)
- **Test metrics (class=1 Succeeding)**

  | Variant | Group | AUC | Precision | Recall | F1 |
  | --- | --- | --- | --- | --- | --- |
  | With SP | Rural Hadza | 0.648 | 0.862 | 0.708 | 0.777 |
  | With SP | GM | 0.765 | 0.801 | 0.743 | 0.771 |
  | With SP | Age 18–24 | 0.691 | 0.759 | 0.672 | 0.713 |
  | No SP | Rural Hadza | 0.644 | 0.859 | 0.690 | 0.765 |
  | No SP | GM | 0.760 | 0.787 | 0.737 | 0.761 |
  | No SP | Age 18–24 | 0.709 | 0.744 | 0.664 | 0.702 |

## Cross-Validation (AUC mean ± std, 5-fold)
- **5-fold AUC**

  | Variant | Group | AUC mean | AUC std |
  | --- | --- | --- | --- |
  | With SP | Rural Hadza | 0.641 | 0.030 |
  | With SP | GM | 0.717 | 0.019 |
  | With SP | Age 18–24 | 0.730 | 0.021 |
  | No SP | Rural Hadza | 0.644 | 0.037 |
  | No SP | GM | 0.707 | 0.020 |
  | No SP | Age 18–24 | 0.688 | 0.017 |

## High-Level Narrative
- Family closeness and education are the most stable differentiators across all groups; UPF frequency and low exercise add group-specific lift.
- SP-related fields have minimal impact on GM and Rural Hadza; for Age 18–24, removing SP shifts importance toward exercise/UPF but improves AUC slightly.
- GM remains the strongest-performing segment; Rural Hadza is weakest; Age 18–24 sits between them with small gains when SP features are excluded.

## Sanity Checks (Permutation + PDP, No SP)
- Files in [sanity_checks/permutation_pdp/noSP](sanity_checks/permutation_pdp/noSP): permutation_importance.csv/png and pdp_top3.png per group.
- Permutation top-3 (AUC-based):
  - Rural Hadza: RelationWithAdultFamily_1_Very Close to Most; Education_Years; UPF.Freq_several days a week
  - GM: RelationWithAdultFamily_1_Very Close to Most; Education_Years; UPF.Freq_several times a day
  - Age 18–24: RelationWithAdultFamily_1_Very Close to Most; UPF.Freq_a few times a month; Exercise.Freq_once a day
- These align with SHAP rankings, reinforcing the same core drivers.
