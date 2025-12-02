import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()
df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)

test = pd.read_csv("test.csv", index_col="id")
original = pd.read_csv("diabetes_dataset.csv")

for col in cat_cols:

    mean_col = pd.DataFrame(
        original.groupby(col)["diagnosed_diabetes"]
        .mean()).reset_index()
    mean_col = mean_col.rename(columns={"diagnosed_diabetes": f"{col}_mean"})
    df = df.merge(mean_col, on=col, how="left")
    test = test.merge(mean_col, on=col, how="left")

encoder = OrdinalEncoder()
df[cat_cols] = encoder.fit_transform(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(int)
df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)

test[cat_cols] = encoder.transform(test[cat_cols])
test[cat_cols] = test[cat_cols].astype(int)

X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

del df, original
gc.collect()

xgb_params = {
    'device': 'cuda',
    'max_depth': 8,
    'learning_rate': 0.01743061164700545,
    'gamma': 0.05367436158322553,
    'min_child_weight': 39,
    'colsample_bytree': 0.7,
    'reg_lambda': 9.852809220595176,
    'reg_alpha': 4.347223030038844,
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
   
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    xgb_md = xgb.train(xgb_params,
                       dtrain=dtrain,
                       num_boost_round=2000,
                       evals=[(dvalid, 'validation')],
                       early_stopping_rounds=100,
                       verbose_eval=False)
    xgb_pred = xgb_md.predict(dvalid)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["xgb_pred_3"] = xgb_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = xgb_md.predict(xgb.DMatrix(test))
    test_preds = pd.DataFrame()
    test_preds["xgb_pred_3"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = roc_auc_score(y_val, xgb_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["xgb_pred_3"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = preds
submission.to_csv("xgb_sub_3.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_xgb_3.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_xgb_3.csv", index=False)
print("Done!")
