import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype("category")

# Convert object columns with 3 or more unique labels to category dtype
object_cols = df.select_dtypes(include="object").columns.to_list()
if "Churn" in object_cols:
    object_cols.remove("Churn")

for col in object_cols:
    df[col] = df[col].astype("category")
    test[col] = test[col].astype("category")

X = df.drop(columns=["Churn"])
y = df["Churn"]


lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 2,
    "reg_lambda": 10,
    "max_bin": 20000,
    "early_stopping_rounds": 300,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = X.select_dtypes(include="category").columns.tolist()

scores, test_preds = [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(**lgb_params, n_estimators=10000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_features,
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred)
    scores.append(auc_score)

    print(f"Fold {fold} AUC: {auc_score:.4f}")
    
    test_pred = model.predict_proba(test)[:, 1]
    test_preds.append(test_pred)

print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Fold 0 AUC: 0.9181
# Fold 1 AUC: 0.9188
# Fold 2 AUC: 0.9180
# Fold 3 AUC: 0.9193
# Fold 4 AUC: 0.9166
# Mean AUC: 0.9182 ± 0.0009
