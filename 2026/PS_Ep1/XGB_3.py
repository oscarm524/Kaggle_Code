import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()

test = pd.read_csv("test.csv", index_col="id")

for col in cat_cols:
    df[col] = df[col].astype("category")
    test[col] = test[col].astype("category")

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

del df
gc.collect()

xgb_params = {
 'device': 'cuda',
 'max_depth': 7,
 'learning_rate': 0.04541499486581436,
 'gamma': 2.6546486024073404,
 'min_child_weight': 90,
 'colsample_bytree': 0.8,
 'reg_lambda': 7.2914042086797615,
 'reg_alpha': 9.280202100596322,
 'n_jobs': -1
 }

skf = KFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    xgb_md = xgb.train(xgb_params,
                       dtrain=dtrain,
                       num_boost_round=1000,
                       evals=[(dvalid, 'validation')],
                       early_stopping_rounds=100,
                       verbose_eval=False)
    xgb_pred = xgb_md.predict(dvalid)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["xgb_pred_3"] = xgb_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = xgb_md.predict(xgb.DMatrix(test, enable_categorical=True))
    test_preds = pd.DataFrame()
    test_preds["xgb_pred_3"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = root_mean_squared_error(y_val, xgb_pred)
    scores.append(score)
    print(f"Fold {i+1} RMSE: {score}")

print(f"Mean RMSE: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["xgb_pred_3"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["exam_score"] = preds
submission.to_csv("xgb_sub_3.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_xgb_3.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_xgb_3.csv", index=False)
print("Done!")

# Fold 1 RMSE: 8.74522658977792
# Fold 2 RMSE: 8.747028684677758
# Fold 3 RMSE: 8.734350514066907
# Fold 4 RMSE: 8.75517864045366
# Fold 5 RMSE: 8.77082318126292
# Mean RMSE: 8.750521522047833