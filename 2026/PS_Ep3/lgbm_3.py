import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from feature_engineering import add_numeric_features

import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("train.csv", index_col="id")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

test = pd.read_csv("test.csv", index_col="id")
test["SeniorCitizen"] = test["SeniorCitizen"].astype("category")

df = add_numeric_features(df)
test = add_numeric_features(test)

# Convert object columns with 3 or more unique labels to category dtype
object_cols = df.select_dtypes(include='object').columns.to_list()
# Exclude target variable if it exists in object columns
if "Churn" in object_cols:
    object_cols.remove("Churn")

for col in object_cols:
    df[col] = df[col].astype('category')
    test[col] = test[col].astype('category')

X = df.drop(columns=["Churn"])
y = df["Churn"]

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.11477585448675186,
    'num_leaves': 55,
    'max_depth': 3,
    'max_bin': 1000,
    'min_child_samples': 25,
    'min_child_weight': 0.903644027293575,
    'subsample': 0.7103174807161573,
    'colsample_bytree': 0.6093623922638883,
    'reg_alpha': 0.9415052421195693,
    'reg_lambda': 0.2594253930646279,
    'device': 'cpu',
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_features = X.select_dtypes(include='category').columns.tolist()

scores, test_preds = [], []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features
    )
    dval = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_features
    )

    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False)
        ]
    )

    y_pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val, y_pred)
    scores.append(auc_score)

    print(f"Fold {fold} AUC: {auc_score:.4f}")

    test_pred = model.predict(test)
    test_preds.append(test_pred)

print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Fold 0 AUC: 0.9177
# Fold 1 AUC: 0.9186
# Fold 2 AUC: 0.9181
# Fold 3 AUC: 0.9192
# Fold 4 AUC: 0.9164
# Mean AUC: 0.9180 ± 0.0009