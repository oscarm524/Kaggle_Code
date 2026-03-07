import warnings

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import (count_services,
                                 add_tenure_features,
                                 add_risk_indicator_features,
                                 add_numeric_features)

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype(str)

# Create a feature for total number of services
service_cols = [
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'StreamingTV',
    'StreamingMovies'
]

df['TotalServices'] = df.apply(count_services, axis=1, args=(service_cols,))
df = add_tenure_features(df)
df = add_risk_indicator_features(df)
df = add_numeric_features(df)

test['TotalServices'] = test.apply(
    count_services, axis=1, args=(service_cols,)
)
test = add_tenure_features(test)
test = add_risk_indicator_features(test)
test = add_numeric_features(test)

X = df.drop(columns=["Churn"])
y = df["Churn"]

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
        test[col] = test[col].astype(str)

cat_params = {
    'iterations': 10000,
    'max_depth': 5,
    'eval_metric': 'AUC',
    'l2_leaf_reg': 3,
    'task_type': 'GPU',
    'verbose': 0
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, test_preds = [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    model_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    eval_pool = Pool(data=X_val, label=y_val, cat_features=cat_cols)

    md = CatBoostClassifier(**cat_params).fit(
        model_pool,
        eval_set=eval_pool,
        verbose=0,
        early_stopping_rounds=1000
    )
    md_pred = md.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, md_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

    test_pred = md.predict_proba(test)[:, 1]
    test_preds.append(test_pred)

print(f"Mean AUC: {np.mean(scores)}")

# Default metric period is 5 because AUC is/are not implemented for GPU
# Fold 1 AUC: 0.917435395238766
# Default metric period is 5 because AUC is/are not implemented for GPU
# Fold 2 AUC: 0.9182512561810444
# Default metric period is 5 because AUC is/are not implemented for GPU
# Fold 3 AUC: 0.917735055129148
# Default metric period is 5 because AUC is/are not implemented for GPU
# Fold 4 AUC: 0.9188092961099265
# Default metric period is 5 because AUC is/are not implemented for GPU
# Fold 5 AUC: 0.9159772132836163
# Mean AUC: 0.9176416431885002
# Mean AUC: 0.9176416431885002
