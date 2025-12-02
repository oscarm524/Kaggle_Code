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
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(int)

df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)

X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

del df
gc.collect()

test = pd.read_csv("test.csv", index_col="id")
test[cat_cols] = encoder.transform(test[cat_cols])
test[cat_cols] = test[cat_cols].astype(int)

xgb_params = {
    'device': 'cuda',
    'max_depth': 11,
    'learning_rate': 0.014888815587019282,
    'gamma': 0.062487392993290314,
    'min_child_weight': 34,
    'colsample_bytree': 0.6,
    'reg_lambda': 2.976059681891355,
    'reg_alpha': 7.40850302183151,
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
                       early_stopping_rounds=150,
                       verbose_eval=False)
    xgb_pred = xgb_md.predict(dvalid)

    oof_preds = pd.DataFrame()
    oof_preds["id"] = X_val.index
    oof_preds["xgb_pred_2"] = xgb_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = xgb_md.predict(xgb.DMatrix(test))
    test_preds = pd.DataFrame()
    test_preds["xgb_pred_2"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = roc_auc_score(y_val, xgb_pred)
    scores.append(score)
    print(f"Fold {i+1} AUC: {score}")

print(f"Mean AUC: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["xgb_pred_2"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = preds
submission.to_csv("xgb_sub_2.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_xgb_2.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_xgb_2.csv", index=False)
print("Done!")
