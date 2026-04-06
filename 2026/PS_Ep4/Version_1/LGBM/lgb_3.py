"""LightGBM model with best hyperparameters from fourth Optuna HPO.

Trains a 5-fold stratified cross-validation LightGBM model using the
best hyperparameter combination found during the fourth tuning run
(with balanced sample weights and a logistic regression meta-feature).
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from lightgbm import Dataset
import lightgbm as lgb

# -- Paths -------------------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/LGBM/lgb_3.csv"
OOF_PATH = f"{BASE_DIR}/Models/LGBM/lgb_3_oof.csv"

# -- Load Data ---------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

# -- Build logistic-regression meta-feature ----------------------------------
X_org = train.drop(columns=["Irrigation_Need"])
y_org = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

cat_cols_org = X_org.select_dtypes(include=["object"]).columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = ohe.fit_transform(X_org[cat_cols_org])
X_num = X_org.drop(columns=cat_cols_org).values
X_ohe = np.hstack([X_num, X_cat])

base_logit = make_pipeline(
    RobustScaler(),
    LogisticRegression(
        C=0.1,
        class_weight="balanced",
        multi_class="multinomial",
        max_iter=1000,
        random_state=42,
    ),
)
base_logit.fit(X_ohe, y_org)

# Train meta-feature
label_map = {0: "Low", 1: "Medium", 2: "High"}
train_logit_pred = base_logit.predict(X_ohe)
train_logit_label = [label_map[i] for i in train_logit_pred]
print(
    "Base Logistic Regression Balanced Accuracy:",
    balanced_accuracy_score(y_org, train_logit_pred),
)

# Test meta-feature
X_test_cat = ohe.transform(test[cat_cols_org])
X_test_num = test.drop(columns=cat_cols_org).values
X_test_ohe = np.hstack([X_test_num, X_test_cat])
test_logit_pred = base_logit.predict(X_test_ohe)
test_logit_label = [label_map[i] for i in test_logit_pred]

del X_cat, X_num, X_ohe, X_test_cat, X_test_num, X_test_ohe
gc.collect()

# -- Prepare features --------------------------------------------------------
X = train.drop(columns=["Irrigation_Need"])
X["original_pred"] = train_logit_label
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()
X_test["original_pred"] = test_logit_label

cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# -- Best Hyper-parameters from Optuna (4th run) -----------------------------
param = {
    "device": "gpu",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "max_depth": 3,
    "num_leaves": 7,
    "learning_rate": 0.18369408096746126,
    "subsample": 0.7399791753467232,
    "colsample_bytree": 0.5897101759600512,
    "min_child_samples": 100,
    "reg_alpha": 9.698011919178818,
    "reg_lambda": 0.19805444312433593,
    "random_state": 42,
}

# -- 5-Fold CV Training & Test Prediction ------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1} / 5")

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    w_train = compute_sample_weight("balanced", y_train)
    w_valid = compute_sample_weight("balanced", y_valid)

    dtrain = Dataset(
        X_train, label=y_train, weight=w_train,
        categorical_feature=cat_cols,
    )
    dvalid = Dataset(
        X_valid, label=y_valid, weight=w_valid,
        reference=dtrain, categorical_feature=cat_cols,
    )

    model = lgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # OOF predictions
    oof_probs[valid_index] = model.predict(X_valid)
    oof_preds[valid_index] = oof_probs[valid_index].argmax(axis=1)
    fold_score = balanced_accuracy_score(
        y_valid, oof_preds[valid_index]
    )
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # Test predictions (average probabilities across folds)
    test_preds += model.predict(X_test) / 5

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# -- Save OOF Predictions ----------------------------------------------------
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Generate Submission ------------------------------------------------------
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.104283
# [200]   valid_0's multi_logloss: 0.0914242
# [300]   valid_0's multi_logloss: 0.0879423
# [400]   valid_0's multi_logloss: 0.0861516
# [500]   valid_0's multi_logloss: 0.0854135
# [600]   valid_0's multi_logloss: 0.0850617
# [700]   valid_0's multi_logloss: 0.0849842
# Early stopping, best iteration is:
# [715]   valid_0's multi_logloss: 0.0849173
#   Fold 1 Balanced Accuracy: 0.970654

# Fold 2 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.102606
# [200]   valid_0's multi_logloss: 0.089494
# [300]   valid_0's multi_logloss: 0.0858463
# [400]   valid_0's multi_logloss: 0.08412
# [500]   valid_0's multi_logloss: 0.083176
# [600]   valid_0's multi_logloss: 0.082766
# [700]   valid_0's multi_logloss: 0.0826174
# Early stopping, best iteration is:
# [723]   valid_0's multi_logloss: 0.0825217
#   Fold 2 Balanced Accuracy: 0.972345

# Fold 3 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.101007
# [200]   valid_0's multi_logloss: 0.0881473
# [300]   valid_0's multi_logloss: 0.0841845
# [400]   valid_0's multi_logloss: 0.0823726
# [500]   valid_0's multi_logloss: 0.0813644
# [600]   valid_0's multi_logloss: 0.0806735
# [700]   valid_0's multi_logloss: 0.0803018
# [800]   valid_0's multi_logloss: 0.0801132
# Early stopping, best iteration is:
# [800]   valid_0's multi_logloss: 0.0801132
#   Fold 3 Balanced Accuracy: 0.973759

# Fold 4 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.10073
# [200]   valid_0's multi_logloss: 0.0881378
# [300]   valid_0's multi_logloss: 0.0846165
# [400]   valid_0's multi_logloss: 0.0828255
# [500]   valid_0's multi_logloss: 0.0820769
# [600]   valid_0's multi_logloss: 0.0817494
# [700]   valid_0's multi_logloss: 0.0812997
# Early stopping, best iteration is:
# [710]   valid_0's multi_logloss: 0.081268
#   Fold 4 Balanced Accuracy: 0.971346

# Fold 5 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.101175
# [200]   valid_0's multi_logloss: 0.0877694
# [300]   valid_0's multi_logloss: 0.0836048
# [400]   valid_0's multi_logloss: 0.0818765
# [500]   valid_0's multi_logloss: 0.0810134
# [600]   valid_0's multi_logloss: 0.0803336
# Early stopping, best iteration is:
# [616]   valid_0's multi_logloss: 0.0802529
#   Fold 5 Balanced Accuracy: 0.971836

# Overall OOF Balanced Accuracy: 0.971988