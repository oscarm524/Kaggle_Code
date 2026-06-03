"""XGBoost model with best hyperparameters from first Optuna HPO.

Trains a 5-fold stratified cross-validation XGBoost model using the
best hyperparameter combination found during the first tuning run.
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/DATA"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/MODELING_V1/XGB/xgb_2.csv"

# ── Load Data ────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["class"], axis=1)
y = train["class"].map({"GALAXY": 0, "QSO": 1, "STAR": 2})

X_test = test.copy()

# Convert object columns to category
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# ── Best Hyper-parameters from Optuna ────────────────────────────────────
param = {
    "device": "cuda",
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 7,
    "learning_rate": 0.07854200743388179,
    "subsample": 0.9060669365605736,
    "colsample_bytree": 0.742814861163083,
    "random_state": 42,
}

# ── 5-Fold CV Training & Test Prediction ─────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dtest = xgb.DMatrix(X_test, enable_categorical=True)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))  # OOF softprob likelihoods
test_preds = np.zeros((len(X_test), 3))  # accumulate softprob per fold
fold_scores = []

# Use softprob for test averaging, softmax for OOF scoring
param_prob = param.copy()
param_prob["objective"] = "multi:softprob"

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1} / 5")

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Compute sample weights based on class distribution
    w_train = compute_sample_weight('balanced', y_train)
    w_valid = compute_sample_weight('balanced', y_valid)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, weight=w_valid, enable_categorical=True)
    
    model = xgb.train(
        param,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    # OOF predictions
    oof_preds[valid_index] = model.predict(dvalid)
    fold_score = balanced_accuracy_score(y_valid, oof_preds[valid_index])
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # Test predictions (use softprob model for averaging)
    model_prob = xgb.train(
        param_prob,
        dtrain,
        num_boost_round=model.best_iteration + 1,
        verbose_eval=False,
    )
    test_preds += model_prob.predict(dtest) / 5

    # OOF likelihoods
    dvalid_prob = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    oof_probs[valid_index] = model_prob.predict(dvalid_prob)

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# ── Save OOF Predictions ─────────────────────────────────────────────────
OOF_PATH = f"/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/MODELING_V1/XGB/xgb_2_oof.csv"
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_GALAXY": oof_probs[:, 0],
    "prob_QSO": oof_probs[:, 1],
    "prob_STAR": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# ── Generate Submission ──────────────────────────────────────────────────
label_map = {0: "GALAXY", 1: "QSO", 2: "STAR"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["class"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# [0]     validation-mlogloss:1.00200
# [100]   validation-mlogloss:0.11466
# [200]   validation-mlogloss:0.10294
# [300]   validation-mlogloss:0.09894
# [400]   validation-mlogloss:0.09730
# [500]   validation-mlogloss:0.09671
# [600]   validation-mlogloss:0.09641
# [653]   validation-mlogloss:0.09647
#   Fold 1 Balanced Accuracy: 0.965961

# Fold 2 / 5
# [0]     validation-mlogloss:1.00199
# [100]   validation-mlogloss:0.11638
# [200]   validation-mlogloss:0.10425
# [300]   validation-mlogloss:0.10044
# [400]   validation-mlogloss:0.09877
# [500]   validation-mlogloss:0.09789
# [586]   validation-mlogloss:0.09784
#   Fold 2 Balanced Accuracy: 0.965362

# Fold 3 / 5
# [0]     validation-mlogloss:1.00206
# [100]   validation-mlogloss:0.11809
# [200]   validation-mlogloss:0.10614
# [300]   validation-mlogloss:0.10226
# [400]   validation-mlogloss:0.10035
# [500]   validation-mlogloss:0.09968
# [600]   validation-mlogloss:0.09961
# [632]   validation-mlogloss:0.09964
#   Fold 3 Balanced Accuracy: 0.965167

# Fold 4 / 5
# [0]     validation-mlogloss:1.00197
# [100]   validation-mlogloss:0.11587
# [200]   validation-mlogloss:0.10396
# [300]   validation-mlogloss:0.10066
# [400]   validation-mlogloss:0.09897
# [500]   validation-mlogloss:0.09846
# [600]   validation-mlogloss:0.09838
# [651]   validation-mlogloss:0.09851
#   Fold 4 Balanced Accuracy: 0.965104

# Fold 5 / 5
# [0]     validation-mlogloss:1.00219
# [100]   validation-mlogloss:0.11653
# [200]   validation-mlogloss:0.10509
# [300]   validation-mlogloss:0.10107
# [400]   validation-mlogloss:0.09928
# [500]   validation-mlogloss:0.09859
# [600]   validation-mlogloss:0.09839
# [633]   validation-mlogloss:0.09838
#   Fold 5 Balanced Accuracy: 0.965170

# Overall OOF Balanced Accuracy: 0.965353