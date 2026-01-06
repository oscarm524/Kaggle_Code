import pandas as pd
import numpy as np

import gc
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv", index_col="id")
cat_cols = df.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = encoder.fit_transform(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(int)

test = pd.read_csv("test.csv", index_col="id")
test[cat_cols] = encoder.transform(test[cat_cols])
test[cat_cols] = test[cat_cols].astype(int)

num_cols = ['study_hours', 'class_attendance', 'sleep_hours']

def fe(df):

    df_temp = df.copy()
    # Thanks to Vladimir Demidov for the following feature
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32') 

    for col in num_cols:
        if col in df_temp.columns:
            df_temp[f'log_{col}'] = np.log1p(df_temp[col])
            df_temp[f'{col}_sq'] = df_temp[col] ** 2

    # # Thanks to Spiritmilk for the following feature      
    # df_temp['feature_formula'] = (
    #     5.9051154511950499 * df_temp['study_hours'] + 
    #     0.34540967058057986 * df_temp['class_attendance'] + 
    #     1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    # )

    return df_temp

df = fe(df)
test = fe(test)

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

del df
gc.collect()

xgb_params = {
 'device': 'cuda',
 'max_depth': 6,
 'learning_rate': 0.072739718403206,
 'gamma': 1.7888938355292365,
 'min_child_weight': 89,
 'colsample_bytree': 0.6,
 'reg_lambda': 2.2420868353220964,
 'reg_alpha': 2.4916117766200605,
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
    oof_preds["xgb_pred_4"] = xgb_pred
    oof_preds["y"] = y_val.values
    oof_preds_df.append(oof_preds)

    test_pred = xgb_md.predict(xgb.DMatrix(test, enable_categorical=True))
    test_preds = pd.DataFrame()
    test_preds["xgb_pred_4"] = test_pred
    test_preds["fold"] = i
    test_preds_df.append(test_preds)

    score = root_mean_squared_error(y_val, xgb_pred)
    scores.append(score)
    print(f"Fold {i+1} RMSE: {score}")

print(f"Mean RMSE: {np.mean(scores)}")
# average the per-fold test predictions properly
preds_per_fold = [df["xgb_pred_4"].values for df in test_preds_df]
final_test_pred = np.mean(preds_per_fold, axis=0)
# use averaged fold predictions for submission
# (or retrain on full data if desired)
preds = final_test_pred

submission = pd.read_csv("sample_submission.csv")
submission["exam_score"] = preds
submission.to_csv("xgb_sub_4.csv", index=False)

print("Saving OOF predictions...")
oof_preds_df = pd.concat(oof_preds_df)
oof_preds_df.to_csv("oof_xgb_4.csv", index=False)

print("Saving test predictions...")
test_preds_df = pd.concat(test_preds_df)
test_preds_df.to_csv("test_xgb_4.csv", index=False)
print("Done!")

# Fold 1 RMSE: 8.712291012400538
# Fold 2 RMSE: 8.716743794705224
# Fold 3 RMSE: 8.700470612156051
# Fold 4 RMSE: 8.723637086926997
# Fold 5 RMSE: 8.740424410490801
# Mean RMSE: 8.718713383335922
