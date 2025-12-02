import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dat_1 = pd.read_csv("oof_xgb_1.csv")
dat_1.columns = ["id", "xgb_pred_1", "y"]

dat_2 = pd.read_csv("oof_xgb_2.csv")
dat_2 = dat_2.drop(columns=["y"])
dat_2.columns = ["id", "xgb_pred_2"]

dat_3 = pd.read_csv("oof_xgb_3.csv")
dat_3 = dat_3.drop(columns=["y"])
dat_3.columns = ["id", "xgb_pred_3"]

dat_4 = pd.read_csv("oof_xgb_4.csv")
dat_4 = dat_4.drop(columns=["y"])
dat_4.columns = ["id", "xgb_pred_4"]

dat = pd.merge(dat_1, dat_2, on=["id"])
dat = pd.merge(dat, dat_3, on=["id"])
dat = pd.merge(dat, dat_4, on=["id"])

X = dat.drop(columns=["id", "y"])
y = dat["y"]

model = LinearDiscriminantAnalysis()
model.fit(X, y)

test_1 = pd.read_csv("test_xgb_1.csv")
test_1.columns = ["xgb_pred_1", "fold"]

test_2 = pd.read_csv("test_xgb_2.csv")
test_2 = test_2.drop(columns=["fold"])
test_2.columns = ["xgb_pred_2"]   

test_3 = pd.read_csv("test_xgb_3.csv")
test_3 = test_3.drop(columns=["fold"])
test_3.columns = ["xgb_pred_3"]

test_4 = pd.read_csv("test_xgb_4.csv")
test_4 = test_4.drop(columns=["fold"])
test_4.columns = ["xgb_pred_4"]

test = pd.concat(
    [
        test_1, test_2, test_3, test_4
    ],
    axis=1
)

test_preds = []
for i in range(0, 5):
    X_test = test[test["fold"] == i].drop(columns=["fold"], axis=1)
    preds = model.predict_proba(X_test)[:, 1]
    test_preds.append(preds)

final_test_preds = np.mean(test_preds, axis=0)

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = final_test_preds
submission.to_csv("ens_sub_2.csv", index=False)
print("Ensemble submission saved as ens_sub_2.csv")
print("Done!")
