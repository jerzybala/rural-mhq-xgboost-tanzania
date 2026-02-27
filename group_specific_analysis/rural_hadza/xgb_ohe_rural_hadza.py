import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '../../rural_gmdata_forML.csv'
OUTPUT_DIR = '.'
GROUP_FILTER = 'Rural.Hadza'
AGE_FILTER = None  # use None here; age filter handled in specific script variant
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Features as in main OHE pipeline
BASE_FEATURES = ['Education_Years', 'ShareHomeWith', 'RelationWithAdultFamily',
                 'UPF.Freq', 'AgeOfFirstSP', 'Smartphone.ownership', 'Exercise.Freq']


def load_and_filter():
    df = pd.read_csv(DATA_PATH)
    df = df[(df['Overall.MHQ'] >= 100) | (df['Overall.MHQ'] < 0)]
    df['Target'] = (df['Overall.MHQ'] >= 100).astype(int)
    df = df[df['Group'] == GROUP_FILTER]
    if AGE_FILTER:
        df = df[df['AgeGroup'].isin(AGE_FILTER)]
    return df


def build_pipeline(df):
    # decide which age column exists
    features = BASE_FEATURES.copy()
    if 'AgeOfFirstSmartPhone' in df.columns:
        features[4] = 'AgeOfFirstSmartPhone'
    num_features = ['Education_Years']
    cat_features = [col for col in features if col not in num_features]

    X = df[features].copy()
    y = df['Target'].copy()

    # fill missing
    X[num_features] = X[num_features].fillna(X[num_features].median())
    for col in cat_features:
        X[col] = X[col].fillna('Missing').astype(str)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_features),
        ('cat', ohe, cat_features)
    ])

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE,
        tree_method='hist'
    )

    clf = Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])
    return clf, X, y, cat_features, num_features, ohe


def tune_and_train(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    param_distributions = {
        'model__max_depth': [3, 4, 5, 6],
        'model__learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'model__n_estimators': [120, 160, 200, 260],
        'model__min_child_weight': [1, 3, 5],
        'model__subsample': [0.7, 0.85, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__gamma': [0, 0.1, 0.2],
        'model__reg_alpha': [0, 0.05, 0.1],
        'model__reg_lambda': [0.8, 1.0, 1.5],
        'model__scale_pos_weight': [scale_pos_weight]
    }

    rand = RandomizedSearchCV(clf, param_distributions, n_iter=25, cv=3,
                              scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    rand.fit(X_train, y_train)

    best_params = rand.best_params_

    grid_space = {
        'model__max_depth': [max(3, best_params['model__max_depth'] - 1), best_params['model__max_depth'], min(8, best_params['model__max_depth'] + 1)],
        'model__learning_rate': [max(0.01, best_params['model__learning_rate'] - 0.02), best_params['model__learning_rate'], min(0.15, best_params['model__learning_rate'] + 0.02)],
        'model__n_estimators': [max(100, best_params['model__n_estimators'] - 40), best_params['model__n_estimators'], best_params['model__n_estimators'] + 40],
        'model__min_child_weight': [max(1, best_params['model__min_child_weight'] - 1), best_params['model__min_child_weight'], best_params['model__min_child_weight'] + 1],
        'model__subsample': [max(0.6, best_params['model__subsample'] - 0.1), best_params['model__subsample'], min(1.0, best_params['model__subsample'] + 0.1)],
        'model__colsample_bytree': [max(0.6, best_params['model__colsample_bytree'] - 0.1), best_params['model__colsample_bytree'], min(1.0, best_params['model__colsample_bytree'] + 0.1)],
        'model__gamma': [max(0, best_params['model__gamma'] - 0.1), best_params['model__gamma'], best_params['model__gamma'] + 0.1],
        'model__reg_alpha': [max(0, best_params['model__reg_alpha'] - 0.05), best_params['model__reg_alpha'], best_params['model__reg_alpha'] + 0.05],
        'model__reg_lambda': [max(0.5, best_params['model__reg_lambda'] - 0.2), best_params['model__reg_lambda'], best_params['model__reg_lambda'] + 0.2],
        'model__scale_pos_weight': [scale_pos_weight]
    }

    grid = GridSearchCV(rand.best_estimator_, grid_space, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision_struggling': precision_score(y_test, y_pred, pos_label=0),
        'recall_struggling': recall_score(y_test, y_pred, pos_label=0),
        'f1_struggling': f1_score(y_test, y_pred, pos_label=0),
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)

    results = {
        'best_params_random': rand.best_params_,
        'best_params_grid': grid.best_params_,
        'roc_auc_test': metrics['roc_auc'],
        'cv_scores': cv_scores,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    return model, results, X_test, y_test, y_prob


def save_outputs(model, results, ohe, cat_features, num_features):
    import pickle
    with open(f"{OUTPUT_DIR}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{OUTPUT_DIR}/results.pkl", 'wb') as f:
        pickle.dump(results, f)


def main():
    print(f"Running OHE XGBoost for group: {GROUP_FILTER} | age filter: {AGE_FILTER}")
    df = load_and_filter()
    if df.empty:
        print('No data after filtering. Exiting.')
        return
    clf, X, y, cat_features, num_features, ohe = build_pipeline(df)
    model, results, X_test, y_test, y_prob = tune_and_train(clf, X, y)
    save_outputs(model, results, ohe, cat_features, num_features)
    print('Done. Key metrics:')
    print(results)


if __name__ == '__main__':
    main()
