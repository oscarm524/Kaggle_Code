"""LightGBM model with best hyperparameters from second Optuna HPO.

Trains a 5-fold stratified cross-validation LightGBM model using the
best hyperparameter combination found during the second tuning run
(with balanced sample weights).  Saves out-of-fold predicted
probabilities (likelihoods) and a test submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import Dataset
import lightgbm as lgb

# -- Paths -------------------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/LGBM/lgb_2.csv"

# -- Load Data ---------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

# Convert object columns to category
cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# -- Best Hyper-parameters from Optuna (2nd run, with sample weights) --------
param = {
    "device": "gpu",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "max_depth": 5,
    "num_leaves": 21,
    "learning_rate": 0.12213408531582877,
    "subsample": 0.9987040598624739,
    "colsample_bytree": 0.5061118573458931,
    "min_child_samples": 83,
    "reg_alpha": 0.24729671188092317,
    "reg_lambda": 0.0002351201095720575,
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
OOF_PATH = f"{BASE_DIR}/Models/LGBM/lgb_2_oof.csv"
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Generate Submission ------------------------------------------------------
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0992022
# [200]   valid_0's multi_logloss: 0.0881318
# Early stopping, best iteration is:
# [242]   valid_0's multi_logloss: 0.0870518
#   Fold 1 Balanced Accuracy: 0.969429

# Fold 2 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.096356
# [200]   valid_0's multi_logloss: 0.0853368
# [300]   valid_0's multi_logloss: 0.0841697
# Early stopping, best iteration is:
# [269]   valid_0's multi_logloss: 0.0839529
#   Fold 2 Balanced Accuracy: 0.971983

# Fold 3 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0954311
# [200]   valid_0's multi_logloss: 0.0836672
# [300]   valid_0's multi_logloss: 0.081546
# Early stopping, best iteration is:
# [293]   valid_0's multi_logloss: 0.0815288
#   Fold 3 Balanced Accuracy: 0.972314

# Fold 4 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0960502
# [200]   valid_0's multi_logloss: 0.0845415
# [300]   valid_0's multi_logloss: 0.083247
# Early stopping, best iteration is:
# [278]   valid_0's multi_logloss: 0.0831424
#   Fold 4 Balanced Accuracy: 0.970136

# Fold 5 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0958044
# [200]   valid_0's multi_logloss: 0.0842468
# [300]   valid_0's multi_logloss: 0.0827182
# Early stopping, best iteration is:
# [275]   valid_0's multi_logloss: 0.0825206
#   Fold 5 Balanced Accuracy: 0.970344

# Overall OOF Balanced Accuracy: 0.970841