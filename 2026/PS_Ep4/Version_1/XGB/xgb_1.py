import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/XGB/xgb_1.csv"

# ── Load Data ────────────────────────────────────────────────────────────
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

# ── Best Hyper-parameters from Optuna ────────────────────────────────────
param = {
    "device": "cuda",
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 3,
    "learning_rate": 0.18937397370471612,
    "subsample": 0.8261769234779542,
    "colsample_bytree": 0.9894267398014059,
    "eval_metric": "mlogloss",
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

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

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
OOF_PATH = f"{BASE_DIR}/Models/XGB/xgb_1_oof.csv"
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# ── Generate Submission ──────────────────────────────────────────────────
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# [0]     validation-mlogloss:0.91556
# [100]   validation-mlogloss:0.06342
# [200]   validation-mlogloss:0.05826
# [300]   validation-mlogloss:0.05648
# [400]   validation-mlogloss:0.05561
# [500]   validation-mlogloss:0.05518
# [600]   validation-mlogloss:0.05483
# [700]   validation-mlogloss:0.05457
# [800]   validation-mlogloss:0.05446
# [900]   validation-mlogloss:0.05441
# [980]   validation-mlogloss:0.05439
#   Fold 1 Balanced Accuracy: 0.963068

# Fold 2 / 5
# [0]     validation-mlogloss:0.91596
# [100]   validation-mlogloss:0.06454
# [200]   validation-mlogloss:0.05943
# [300]   validation-mlogloss:0.05774
# [400]   validation-mlogloss:0.05681
# [500]   validation-mlogloss:0.05630
# [600]   validation-mlogloss:0.05601
# [700]   validation-mlogloss:0.05586
# [800]   validation-mlogloss:0.05578
# [900]   validation-mlogloss:0.05572
# [999]   validation-mlogloss:0.05568
#   Fold 2 Balanced Accuracy: 0.964665

# Fold 3 / 5
# [0]     validation-mlogloss:0.91649
# [100]   validation-mlogloss:0.06608
# [200]   validation-mlogloss:0.06095
# [300]   validation-mlogloss:0.05918
# [400]   validation-mlogloss:0.05823
# [500]   validation-mlogloss:0.05776
# [600]   validation-mlogloss:0.05738
# [700]   validation-mlogloss:0.05722
# [800]   validation-mlogloss:0.05713
# [900]   validation-mlogloss:0.05708
# [939]   validation-mlogloss:0.05707
#   Fold 3 Balanced Accuracy: 0.963481

# Fold 4 / 5
# [0]     validation-mlogloss:0.91615
# [100]   validation-mlogloss:0.06470
# [200]   validation-mlogloss:0.05927
# [300]   validation-mlogloss:0.05756
# [400]   validation-mlogloss:0.05662
# [500]   validation-mlogloss:0.05610
# [600]   validation-mlogloss:0.05579
# [700]   validation-mlogloss:0.05562
# [800]   validation-mlogloss:0.05555
# [900]   validation-mlogloss:0.05545
# [999]   validation-mlogloss:0.05543
#   Fold 4 Balanced Accuracy: 0.962203

# Fold 5 / 5
# [0]     validation-mlogloss:0.91530
# [100]   validation-mlogloss:0.06338
# [200]   validation-mlogloss:0.05814
# [300]   validation-mlogloss:0.05631
# [400]   validation-mlogloss:0.05533
# [500]   validation-mlogloss:0.05486
# [600]   validation-mlogloss:0.05454
# [700]   validation-mlogloss:0.05434
# [800]   validation-mlogloss:0.05420
# [900]   validation-mlogloss:0.05406
# [999]   validation-mlogloss:0.05396
#   Fold 5 Balanced Accuracy: 0.963602

# Overall OOF Balanced Accuracy: 0.963404