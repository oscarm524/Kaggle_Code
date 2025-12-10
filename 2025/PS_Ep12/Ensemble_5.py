import pandas as pd
import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

dat_1 = pd.read_csv("oof_xgb_5.csv")
dat_1.columns = ["id", "xgb_pred_5", "y"]

dat_2 = pd.read_csv("oof_lgb_7.csv")
dat_2 = dat_2.drop(columns=["y"])
dat_2.columns = ["id", "lgb_pred_7"]

dat_3 = pd.read_csv("oof_ydf_3.csv")
dat_3 = dat_3.drop(columns=["y"])
dat_3.columns = ["id", "ydf_pred_3"]

dat = pd.merge(dat_1, dat_2, on=["id"])
dat = pd.merge(dat, dat_3, on=["id"])

X = dat.drop(columns=["id", "y"])
y = dat["y"]

model_1 = LogisticRegression(solver="liblinear", random_state=42)
model_2 = LinearDiscriminantAnalysis()
model_3 = VotingClassifier(
    estimators=[
        ("lr", model_1),
        ("lda", model_2)
    ],
    voting="soft",
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv_1 = cross_val_predict(
    model_1, X, y, cv=skf, method="predict_proba", n_jobs=-1
)[:, 1]
print("Logistic OOF AUC:", roc_auc_score(y, model_cv_1))

model_cv_2 = cross_val_predict(
    model_2, X, y, cv=skf, method="predict_proba", n_jobs=-1
)[:, 1]
print("LDA OOF AUC:", roc_auc_score(y, model_cv_2))

model_cv_3 = cross_val_predict(
    model_3, X, y, cv=skf, method="predict_proba", n_jobs=-1
)[:, 1]
print("Voting OOF AUC:", roc_auc_score(y, model_cv_3))

# model_1.fit(X, y)
# model_2.fit(X, y)
model_3.fit(X, y)

test_1 = pd.read_csv("test_xgb_5.csv")
test_1.columns = ["xgb_pred_5", "fold"]

test_2 = pd.read_csv("test_lgb_7.csv")
test_2 = test_2.drop(columns=["fold"])
test_2.columns = ["lgb_pred_7"]

test_3 = pd.read_csv("test_ydf_3.csv")
test_3 = test_3.drop(columns=["fold"])
test_3.columns = ["ydf_pred_3"]

test = pd.concat(
    [
        test_1, test_2, test_3
    ],
    axis=1
)

# test_preds_1, test_preds_2, test_preds_3 = [], [], []
test_preds_3 = []
for i in range(0, 5):
    X_test = test[test["fold"] == i].drop(columns=["fold"], axis=1)
    # preds_1 = model_1.predict_proba(X_test)[:, 1]
    # preds_2 = model_2.predict_proba(X_test)[:, 1]
    preds_3 = model_3.predict_proba(X_test)[:, 1]

    # test_preds_1.append(preds_1)
    # test_preds_2.append(preds_2)
    test_preds_3.append(preds_3)

# test_preds_1 = np.mean(test_preds_1, axis=0)
# test_preds_2 = np.mean(test_preds_2, axis=0)
test_preds_3 = np.mean(test_preds_3, axis=0)
# final_test_preds = np.mean([test_preds_1, test_preds_2, test_preds_3], axis=0)

submission = pd.read_csv("sample_submission.csv")
submission["diagnosed_diabetes"] = test_preds_3
submission.to_csv("ens_sub_5.csv", index=False)
print("Ensemble submission saved as ens_sub_5.csv")
print("Done!")

# Logistic OOF AUC: 0.7304714284865119
# LDA OOF AUC: 0.7304898086929404
# Voting OOF AUC: 0.7305151414707725
