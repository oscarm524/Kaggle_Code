import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from lightgbm import Dataset, early_stopping

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
test = pd.read_csv("test.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()

for col in cat_cols:
    df[col] = df[col].astype('category')
    test[col] = test[col].astype('category')

df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)

X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

del df
gc.collect()

lgb_params = {
    'learning_rate': 0.09998487056450626,
    'max_depth': 5,
    'reg_alpha': 5.876801609796171,
    'reg_lambda': 0.03194501388706304,
    'num_leaves': 24,
    'colsample_bytree': 0.6397508847379295,
    'verbose': -1,
    'n_jobs': -1,
    'device': 'gpu'
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    dtrain = Dataset(X_train, label=y_train)
    dtest = Dataset(X_val, label=y_val, reference=dtrain)

    md = lgb.train(params=lgb_params,
                   train_set=dtrain,
                   num_boost_round=1000,
                   valid_sets=[dtest],
                   callbacks=[early_stopping(stopping_rounds=100,
                                             verbose=None)])

    md_pred = md.predict(X_val)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["lgb_pred_3"] = md_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = md.predict(test)
    test_preds = pd.DataFrame()
    test_preds["lgb_pred_3"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = roc_auc_score(y_val, md_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")

# average the per-fold test predictions properly
preds_per_fold = [df["lgb_pred_3"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = preds
submission.to_csv("lgb_sub_3.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_lgb_3.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_lgb_3.csv", index=False)
print("Done!")
