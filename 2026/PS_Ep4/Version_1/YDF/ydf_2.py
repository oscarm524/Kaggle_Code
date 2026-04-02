"""YDF Gradient Boosted Trees model with balanced class weights.

Trains a 5-fold stratified cross-validation YDF GBT model using
balanced class weights computed per fold. Saves out-of-fold predicted
probabilities (likelihoods) and a test submission to CSV.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import ydf
from ydf import GradientBoostedTreesLearner

print(f"YDF version: {ydf.__version__}")

# -- Paths ---------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/YDF/ydf_2.csv"
OOF_PATH = f"{BASE_DIR}/Models/YDF/ydf_2_oof.csv"

# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

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

    # Compute class weights based on class distribution in the training fold
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {str(int(c)): w for c, w in zip(classes, weights)}

    ydf_params = dict(
        task=ydf.Task.CLASSIFICATION,
        label="Irrigation_Need",
        class_weights=class_weights,
        num_threads=10,
        shrinkage=0.1,
        num_trees=10000,
        early_stopping_num_trees_look_ahead=150,
        max_depth=2,
        growing_strategy="BEST_FIRST_GLOBAL",
        categorical_algorithm="RANDOM",
    )

    train_df = X_train.copy()
    train_df["Irrigation_Need"] = y_train

    val_df = X_valid.copy()
    val_df["Irrigation_Need"] = y_valid

    model = GradientBoostedTreesLearner(**ydf_params).train(train_df)

    # OOF predictions (probabilities)
    val_probs = model.predict(val_df)
    oof_probs[valid_index] = val_probs
    oof_preds[valid_index] = np.argmax(val_probs, axis=1)

    fold_score = balanced_accuracy_score(y_valid, oof_preds[valid_index])
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}\n")

    # Test predictions (average probabilities across folds)
    test_df = X_test.copy()
    test_preds += model.predict(test_df) / 5

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
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# Train model on 504000 examples
# Model trained in 0:02:07.306666
#   Fold 1 Balanced Accuracy: 0.974137

# Fold 2 / 5
# Train model on 504000 examples
# Model trained in 0:02:17.554652
#   Fold 2 Balanced Accuracy: 0.975397

# Fold 3 / 5
# Train model on 504000 examples
# Model trained in 0:01:41.298702
#   Fold 3 Balanced Accuracy: 0.975660

# Fold 4 / 5
# Train model on 504000 examples
# Model trained in 0:02:06.722348
#   Fold 4 Balanced Accuracy: 0.975107

# Fold 5 / 5
# Train model on 504000 examples
# Model trained in 0:02:10.315920
#   Fold 5 Balanced Accuracy: 0.975521

# Overall OOF Balanced Accuracy: 0.975164