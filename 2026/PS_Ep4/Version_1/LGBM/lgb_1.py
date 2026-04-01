"""LightGBM model with best hyperparameters from first Optuna HPO.

Trains a 5-fold stratified cross-validation LightGBM model using the
best hyperparameter combination found during the first tuning run.
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from lightgbm import Dataset
import lightgbm as lgb

# -- Paths -------------------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/LGBM/lgb_1.csv"

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

# -- Best Hyper-parameters from Optuna ---------------------------------------
param = {
    "device": "gpu",
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbosity": -1,
    "max_depth": 5,
    "num_leaves": 24,
    "learning_rate": 0.07569094658099462,
    "subsample": 0.808689869335482,
    "colsample_bytree": 0.9288709861273847,
    "min_child_samples": 31,
    "reg_alpha": 3.920993953631824e-08,
    "reg_lambda": 8.112880182665321e-07,
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

    dtrain = Dataset(
        X_train, label=y_train, categorical_feature=cat_cols
    )
    dvalid = Dataset(
        X_valid,
        label=y_valid,
        reference=dtrain,
        categorical_feature=cat_cols,
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
OOF_PATH = f"{BASE_DIR}/Models/LGBM/lgb_1_oof.csv"
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
# [100]   valid_0's multi_logloss: 0.0621541
# [200]   valid_0's multi_logloss: 0.0591219
# [300]   valid_0's multi_logloss: 0.0580065
# [400]   valid_0's multi_logloss: 0.0574808
# [500]   valid_0's multi_logloss: 0.0572867
# [600]   valid_0's multi_logloss: 0.0570996
# [700]   valid_0's multi_logloss: 0.0570443
# Early stopping, best iteration is:
# [736]   valid_0's multi_logloss: 0.0570026
#   Fold 1 Balanced Accuracy: 0.961755

# Fold 2 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0626493
# [200]   valid_0's multi_logloss: 0.0598648
# [300]   valid_0's multi_logloss: 0.0586894
# [400]   valid_0's multi_logloss: 0.0581795
# [500]   valid_0's multi_logloss: 0.0579766
# [600]   valid_0's multi_logloss: 0.0578628
# [700]   valid_0's multi_logloss: 0.0578493
# Early stopping, best iteration is:
# [660]   valid_0's multi_logloss: 0.0578253
#   Fold 2 Balanced Accuracy: 0.963514

# Fold 3 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0624218
# [200]   valid_0's multi_logloss: 0.0594537
# [300]   valid_0's multi_logloss: 0.0582853
# [400]   valid_0's multi_logloss: 0.0577774
# [500]   valid_0's multi_logloss: 0.0575182
# [600]   valid_0's multi_logloss: 0.0573889
# [700]   valid_0's multi_logloss: 0.0573644
# Early stopping, best iteration is:
# [662]   valid_0's multi_logloss: 0.0573415
#   Fold 3 Balanced Accuracy: 0.963709

# Fold 4 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0616351
# [200]   valid_0's multi_logloss: 0.0585912
# [300]   valid_0's multi_logloss: 0.0575006
# [400]   valid_0's multi_logloss: 0.0570218
# [500]   valid_0's multi_logloss: 0.0568342
# [600]   valid_0's multi_logloss: 0.0567622
# [700]   valid_0's multi_logloss: 0.0567404
# Early stopping, best iteration is:
# [698]   valid_0's multi_logloss: 0.0567337
#   Fold 4 Balanced Accuracy: 0.961221

# Fold 5 / 5
# Training until validation scores don't improve for 50 rounds
# [100]   valid_0's multi_logloss: 0.0608571
# [200]   valid_0's multi_logloss: 0.0576727
# [300]   valid_0's multi_logloss: 0.0566371
# [400]   valid_0's multi_logloss: 0.0560968
# [500]   valid_0's multi_logloss: 0.0558464
# [600]   valid_0's multi_logloss: 0.0557279
# [700]   valid_0's multi_logloss: 0.0556435
# Early stopping, best iteration is:
# [724]   valid_0's multi_logloss: 0.0556304
#   Fold 5 Balanced Accuracy: 0.962103

# Overall OOF Balanced Accuracy: 0.962461