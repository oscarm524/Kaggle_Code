import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import lightgbm as lgb
from lightgbm import Dataset, early_stopping

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(int)

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

del df
gc.collect()

test = pd.read_csv("test.csv", index_col="id")
test[cat_cols] = encoder.transform(test[cat_cols])
test[cat_cols] = test[cat_cols].astype(int)

lgb_params = {
 'learning_rate': 0.09257679474290241,
 'max_depth': 5,
 'reg_alpha': 0.02416818671268253,
 'reg_lambda': 0.6930968146965372,
 'num_leaves': 22,
 'colsample_bytree': 0.8424807137427762,
 'verbose': -1,
 'n_jobs': -1,
 'device': 'gpu'
 }

skf = KFold(n_splits=5, shuffle=True, random_state=42)
scores, oof_preds_df, test_preds_df = [], [], []
for i, (train_index, test_index) in enumerate(skf.split(X)):

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

    score = root_mean_squared_error(y_val, md_pred)
    scores.append(score)
    print(f"Fold {i+1} RMSE: {score}")

print(f"Mean RMSE: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["lgb_pred_3"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["exam_score"] = preds
submission.to_csv("lgb_sub_3.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_lgb_3.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_lgb_3.csv", index=False)
print("Done!")

# Fold 1 RMSE: 8.747960663470717
# Fold 2 RMSE: 8.752077054836557
# Fold 3 RMSE: 8.747806302094348
# Fold 4 RMSE: 8.766660963156239
# Fold 5 RMSE: 8.781831995362301
# Mean RMSE: 8.759267395784033