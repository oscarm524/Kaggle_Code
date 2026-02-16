import gc
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from target_encode import target_encode

warnings.filterwarnings("ignore")
    
df = pd.read_csv("train.csv", index_col="id")
X = df.drop(columns=["Heart Disease"])
y = df["Heart Disease"].map({"Absence": 0, "Presence": 1})

potential_cat = ["Chest pain type", "EKG results", "Slope of ST",
                 "Number of vessels fluro", "Thallium"]

cat_params = {
    'max_depth': 3,
    'verbose': 0
}

del df
gc.collect()

original = pd.read_csv("Heart_Disease_Prediction.csv")
original["Heart Disease"] = original["Heart Disease"].map(
    {"Absence": 0, "Presence": 1}
)

potential_cat_1 = X.columns.tolist()

for col in potential_cat_1:

    mean_col = pd.DataFrame(
        original.groupby(col)["Heart Disease"]
        .mean()).reset_index()
    mean_col = mean_col.rename(columns={"Heart Disease": f"{col}_mean"})
    X = X.merge(mean_col, on=col, how="left")

    count_col = pd.DataFrame(
        original.groupby(col)["Heart Disease"]
        .size()).reset_index()
    count_col = count_col.rename(columns={"Heart Disease": f"{col}_count"})
    X = X.merge(count_col, on=col, how="left")

del original
gc.collect()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
   
    # Apply target encoding within the fold
    X_train_encoded, X_val_encoded = target_encode(
        X_train, X_val, y_train,
        potential_cat,
    )

    # Drop original categorical columns and keep encoded versions
    cols_to_drop = potential_cat
    X_train_final = X_train_encoded.drop(columns=cols_to_drop)
    X_val_final = X_val_encoded.drop(columns=cols_to_drop)

    model_pool = Pool(data=X_train_final, label=y_train)
    eval_pool = Pool(data=X_val_final, label=y_val)

    md = CatBoostClassifier(**cat_params).fit(
        model_pool, eval_set=eval_pool, verbose=0, early_stopping_rounds=100
    )
    md_pred = md.predict_proba(X_val_final)[:, 1]

    score = roc_auc_score(y_val, md_pred)
    scores.append(score)

    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")
# Fold 1 AUC: 0.9559304864830427
# Fold 2 AUC: 0.9548542174283136
# Fold 3 AUC: 0.9556239305801022
# Fold 4 AUC: 0.9552927204754584
# Fold 5 AUC: 0.9561213824134085
# Mean AUC: 0.955564547476065

df = pd.read_csv("train.csv", index_col="id")
X = df.drop(columns=["Heart Disease"])
y = df["Heart Disease"].map({"Absence": 0, "Presence": 1})

test_df = pd.read_csv("test.csv", index_col="id")

potential_cat = ["Chest pain type", "EKG results", "Slope of ST",
                 "Number of vessels fluro", "Thallium"]

original = pd.read_csv("Heart_Disease_Prediction.csv")
original["Heart Disease"] = original["Heart Disease"].map(
    {"Absence": 0, "Presence": 1}
)

potential_cat_1 = X.columns.tolist()

for col in potential_cat_1:

    mean_col = pd.DataFrame(
        original.groupby(col)["Heart Disease"]
        .mean()).reset_index()
    mean_col = mean_col.rename(columns={"Heart Disease": f"{col}_mean"})
    X = X.merge(mean_col, on=col, how="left")
    test_df = test_df.merge(mean_col, on=col, how="left")

    count_col = pd.DataFrame(
        original.groupby(col)["Heart Disease"]
        .size()).reset_index()
    count_col = count_col.rename(columns={"Heart Disease": f"{col}_count"})
    X = X.merge(count_col, on=col, how="left")
    test_df = test_df.merge(count_col, on=col, how="left")

del original
gc.collect()

cat_params = {
    'max_depth': 3,
    'verbose': 0
}

X_encoded, test_encoded = target_encode(
    X, test_df, y,
    potential_cat,
)

model_pool = Pool(data=X_encoded, label=y)
test_pool = Pool(data=test_encoded)

model = CatBoostClassifier(**cat_params).fit(model_pool, verbose=0)
test_pred = model.predict_proba(test_pool)[:, 1]

submission = pd.read_csv("sample_submission.csv")
submission["Heart Disease"] = test_pred
submission.to_csv("cat_sub_3.csv", index=False)
