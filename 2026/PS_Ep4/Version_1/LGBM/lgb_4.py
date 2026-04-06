"""LightGBM model with best hyperparameters from fifth Optuna HPO.

Trains a 5-fold stratified cross-validation LightGBM model using the
best hyperparameter combination found during the fifth tuning run
(with balanced sample weights, a logistic regression meta-feature,
max_bin=1000, and extended boosting rounds).  Saves out-of-fold
predicted probabilities (likelihoods) and a test submission to CSV.
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
OUTPUT_PATH = f"{BASE_DIR}/Models/LGBM/lgb_4.csv"
OOF_PATH = f"{BASE_DIR}/Models/LGBM/lgb_4_oof.csv"

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

# -- Best Hyper-parameters from Optuna (5th run) -----------------------------
param = {
    # "device": "gpu",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "max_depth": 3,
    "num_leaves": 7,
    "learning_rate": 0.19809845397065912,
    "subsample": 0.6472797677041936,
    "colsample_bytree": 0.5087483649032954,
    "min_child_samples": 61,
    "reg_alpha": 6.468254803149446,
    "reg_lambda": 1.4944925916279808e-05,
    "max_bin": 1000,
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
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
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
# Training until validation scores don't improve for 100 rounds
# [100]   valid_0's multi_logloss: 0.102025
# [200]   valid_0's multi_logloss: 0.0884962
# [300]   valid_0's multi_logloss: 0.0841091
# [400]   valid_0's multi_logloss: 0.0820215
# [500]   valid_0's multi_logloss: 0.0811341
# [600]   valid_0's multi_logloss: 0.0806011
# [700]   valid_0's multi_logloss: 0.0801321
# Early stopping, best iteration is:
# [658]   valid_0's multi_logloss: 0.0800713
#   Fold 1 Balanced Accuracy: 0.972761

# Fold 2 / 5
# Training until validation scores don't improve for 100 rounds
# [100]   valid_0's multi_logloss: 0.0996733
# [200]   valid_0's multi_logloss: 0.0859284
# [300]   valid_0's multi_logloss: 0.081688
# [400]   valid_0's multi_logloss: 0.0799022
# [500]   valid_0's multi_logloss: 0.0791534
# [600]   valid_0's multi_logloss: 0.0786221
# [700]   valid_0's multi_logloss: 0.0782334
# [800]   valid_0's multi_logloss: 0.0783065
# Early stopping, best iteration is:
# [726]   valid_0's multi_logloss: 0.0781693
#   Fold 2 Balanced Accuracy: 0.973853

# Fold 3 / 5
# Training until validation scores don't improve for 100 rounds
# [100]   valid_0's multi_logloss: 0.0997734
# [200]   valid_0's multi_logloss: 0.0864187
# [300]   valid_0's multi_logloss: 0.0822869
# [400]   valid_0's multi_logloss: 0.0800035
# [500]   valid_0's multi_logloss: 0.0788076
# [600]   valid_0's multi_logloss: 0.0780707
# [700]   valid_0's multi_logloss: 0.0777577
# [800]   valid_0's multi_logloss: 0.0776213
# [900]   valid_0's multi_logloss: 0.0777762
# Early stopping, best iteration is:
# [817]   valid_0's multi_logloss: 0.0775832
#   Fold 3 Balanced Accuracy: 0.974373

# Fold 4 / 5
# Training until validation scores don't improve for 100 rounds
# [100]   valid_0's multi_logloss: 0.0979842
# [200]   valid_0's multi_logloss: 0.0845663
# [300]   valid_0's multi_logloss: 0.0802716
# [400]   valid_0's multi_logloss: 0.0786164
# [500]   valid_0's multi_logloss: 0.0778476
# [600]   valid_0's multi_logloss: 0.0776629
# [700]   valid_0's multi_logloss: 0.0773672
# [800]   valid_0's multi_logloss: 0.0774836
# Early stopping, best iteration is:
# [711]   valid_0's multi_logloss: 0.077295
#   Fold 4 Balanced Accuracy: 0.973061

# Fold 5 / 5
# Training until validation scores don't improve for 100 rounds
# [100]   valid_0's multi_logloss: 0.0997161
# [200]   valid_0's multi_logloss: 0.0853235
# [300]   valid_0's multi_logloss: 0.0808113
# [400]   valid_0's multi_logloss: 0.078715
# [500]   valid_0's multi_logloss: 0.0775607
# [600]   valid_0's multi_logloss: 0.0769783
# [700]   valid_0's multi_logloss: 0.076494
# [800]   valid_0's multi_logloss: 0.0762587
# Early stopping, best iteration is:
# [780]   valid_0's multi_logloss: 0.0761298
#   Fold 5 Balanced Accuracy: 0.972870

# Overall OOF Balanced Accuracy: 0.973384