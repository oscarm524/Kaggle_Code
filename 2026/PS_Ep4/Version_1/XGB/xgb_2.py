"""XGBoost model with best hyperparameters from second Optuna HPO.

Trains a 5-fold stratified cross-validation XGBoost model using the
best hyperparameter combination found during the second tuning run
(with balanced sample weights). Saves out-of-fold predicted
probabilities (likelihoods) and a test submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# -- Paths ---------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/XGB/xgb_2.csv"

# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

# Convert object columns to category
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# -- Best Hyper-parameters from Optuna (2nd run w/ sample_weight) --
param = {
    "device": "cuda",
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.11064727109405381,
    "subsample": 0.752231741417937,
    "colsample_bytree": 0.8704202245249855,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

# -- 5-Fold CV Training & Test Prediction --------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dtest = xgb.DMatrix(X_test, enable_categorical=True)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

# Use softprob for test averaging, softmax for OOF scoring
param_prob = param.copy()
param_prob["objective"] = "multi:softprob"

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1} / 5")

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Compute sample weights based on class distribution
    w_train = compute_sample_weight("balanced", y_train)
    w_valid = compute_sample_weight("balanced", y_valid)

    dtrain = xgb.DMatrix(
        X_train, label=y_train, weight=w_train,
        enable_categorical=True,
    )
    dvalid = xgb.DMatrix(
        X_valid, label=y_valid, weight=w_valid,
        enable_categorical=True,
    )

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
    fold_score = balanced_accuracy_score(
        y_valid, oof_preds[valid_index],
    )
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # Test predictions (use softprob model for averaging)
    dtrain_prob = xgb.DMatrix(
        X_train, label=y_train, weight=w_train,
        enable_categorical=True,
    )
    model_prob = xgb.train(
        param_prob,
        dtrain_prob,
        num_boost_round=model.best_iteration + 1,
        verbose_eval=False,
    )
    test_preds += model_prob.predict(dtest) / 5

    # OOF likelihoods
    dvalid_prob = xgb.DMatrix(
        X_valid, label=y_valid, weight=w_valid,
        enable_categorical=True,
    )
    oof_probs[valid_index] = model_prob.predict(dvalid_prob)

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# -- Save OOF Predictions ------------------------------------------
OOF_PATH = f"{BASE_DIR}/Models/XGB/xgb_2_oof.csv"
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Generate Submission --------------------------------------------
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# [0]     validation-mlogloss:1.01685
# [100]   validation-mlogloss:0.12703
# [200]   validation-mlogloss:0.09812
# [300]   validation-mlogloss:0.09024
# [400]   validation-mlogloss:0.08674
# [500]   validation-mlogloss:0.08475
# [600]   validation-mlogloss:0.08346
# [700]   validation-mlogloss:0.08252
# [800]   validation-mlogloss:0.08178
# [900]   validation-mlogloss:0.08138
# [999]   validation-mlogloss:0.08121
#   Fold 1 Balanced Accuracy: 0.971136

# Fold 2 / 5
# [0]     validation-mlogloss:1.01670
# [100]   validation-mlogloss:0.12460
# [200]   validation-mlogloss:0.09642
# [300]   validation-mlogloss:0.08848
# [400]   validation-mlogloss:0.08485
# [500]   validation-mlogloss:0.08286
# [600]   validation-mlogloss:0.08137
# [700]   validation-mlogloss:0.08057
# [800]   validation-mlogloss:0.08014
# [900]   validation-mlogloss:0.07973
# [995]   validation-mlogloss:0.07970
#   Fold 2 Balanced Accuracy: 0.973418

# Fold 3 / 5
# [0]     validation-mlogloss:1.01687
# [100]   validation-mlogloss:0.12629
# [200]   validation-mlogloss:0.09735
# [300]   validation-mlogloss:0.08965
# [400]   validation-mlogloss:0.08567
# [500]   validation-mlogloss:0.08349
# [600]   validation-mlogloss:0.08197
# [700]   validation-mlogloss:0.08101
# [800]   validation-mlogloss:0.08024
# [900]   validation-mlogloss:0.07974
# [999]   validation-mlogloss:0.07937
#   Fold 3 Balanced Accuracy: 0.973412

# Fold 4 / 5
# [0]     validation-mlogloss:1.01684
# [100]   validation-mlogloss:0.12434
# [200]   validation-mlogloss:0.09487
# [300]   validation-mlogloss:0.08724
# [400]   validation-mlogloss:0.08346
# [500]   validation-mlogloss:0.08120
# [600]   validation-mlogloss:0.07990
# [700]   validation-mlogloss:0.07903
# [800]   validation-mlogloss:0.07827
# [900]   validation-mlogloss:0.07805
# [999]   validation-mlogloss:0.07786
#   Fold 4 Balanced Accuracy: 0.972494

# Fold 5 / 5
# [0]     validation-mlogloss:1.01710
# [100]   validation-mlogloss:0.12510
# [200]   validation-mlogloss:0.09507
# [300]   validation-mlogloss:0.08729
# [400]   validation-mlogloss:0.08340
# [500]   validation-mlogloss:0.08138
# [600]   validation-mlogloss:0.07998
# [700]   validation-mlogloss:0.07915
# [800]   validation-mlogloss:0.07861
# [900]   validation-mlogloss:0.07804
# [999]   validation-mlogloss:0.07767
#   Fold 5 Balanced Accuracy: 0.972849

# Overall OOF Balanced Accuracy: 0.972662