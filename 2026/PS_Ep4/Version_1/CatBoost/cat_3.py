"""CatBoost model (v3) -- plain features, deeper iterations.

Trains a 5-fold stratified cross-validation CatBoost model without the
logistic-regression stacking feature used in v2.  Uses a higher iteration
count (5000) with a larger border_count (2000) and extended
early-stopping patience (300 rounds).
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool

# -- Paths ---------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/CatBoost/cat_3.csv"
OOF_PATH = f"{BASE_DIR}/Models/CatBoost/cat_3_oof.csv"

# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()

label_map = {0: "Low", 1: "Medium", 2: "High"}

# -- Hyper-parameters -------------------
param = {
    "loss_function": "MultiClass",
    "classes_count": 3,
    "depth": 3,
    "iterations": 5000,
    "border_count": 2000,
    "early_stopping_rounds": 300,
    "random_seed": 42,
    "verbose": 100,
}

# -- 5-Fold CV Training & Test Prediction --------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
fold_scores = []

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1} / 5")

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Compute class weights from training fold
    classes = np.sort(y_train.unique())
    fold_class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train,
    )
    fold_param = param.copy()
    fold_param["class_weights"] = list(fold_class_weights)

    train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**fold_param)
    model.fit(train_pool, eval_set=valid_pool)

    # OOF predictions
    oof_preds[valid_index] = model.predict(valid_pool).flatten().astype(int)
    fold_score = balanced_accuracy_score(
        y_valid, oof_preds[valid_index],
    )
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # OOF likelihoods
    oof_probs[valid_index] = model.predict_proba(valid_pool)

    # Test predictions (average probabilities across folds)
    test_preds += model.predict_proba(test_pool) / 5

print(f"Overall OOF Balanced Accuracy: {np.mean(fold_scores):.6f}")

# -- Save OOF Predictions ------------------------------------------
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# -- Generate Submission --------------------------------------------
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 Balanced Accuracy: 0.97147
# Fold 2 Balanced Accuracy: 0.97261
# Fold 3 Balanced Accuracy: 0.97395
# Fold 4 Balanced Accuracy: 0.97281
# Fold 5 Balanced Accuracy: 0.97355
# Final Balanced Accuracy: 0.9728781019553552
