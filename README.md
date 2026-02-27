
# XGBoost Analysis of Tanzanian MHQ Data


## Project Background
This project investigates the lifestyle drivers of Mental Health Quotient (MHQ) in rural Tanzania using machine learning. We apply XGBoost and SHAP analysis to identify and interpret key predictors of mental health outcomes.


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
