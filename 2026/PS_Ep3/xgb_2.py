import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype("category")

# Convert object columns to category dtype
object_cols = df.select_dtypes(include="object").columns.to_list()
if "Churn" in object_cols:
    object_cols.remove("Churn")

for col in object_cols:
    df[col] = df[col].astype("category")
    test[col] = test[col].astype("category")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Create DMatrix for test data
dtest = xgb.DMatrix(test, enable_categorical=True)

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 2,
    "reg_lambda": 10,
    "max_bin": 16000,
    "device": "cuda",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = X.select_dtypes(include="category").columns.tolist()

aucs, test_preds = [], []
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=300,
        verbose_eval=False,
    )

    y_pred = model.predict(dvalid)
    auc = roc_auc_score(y_valid, y_pred)
    aucs.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.4f}")

    test_pred = model.predict(dtest)
    test_preds.append(test_pred)

print(f"Mean AUC: {np.mean(aucs):.4f}")

# Fold 1 AUC: 0.9181
# Fold 2 AUC: 0.9187
# Fold 3 AUC: 0.9182
# Fold 4 AUC: 0.9190
# Fold 5 AUC: 0.9166
# Mean AUC: 0.9181