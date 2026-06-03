"""XGBoost model with best hyperparameters from first Optuna HPO.

Trains a 5-fold stratified cross-validation XGBoost model using the
best hyperparameter combination found during the first tuning run.
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# -- Reproducibility ------------------------------------------------
SEED = 2026


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


seed_everything(SEED)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/DATA"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/MODELING_V1/XGB/xgb_3.csv"

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

# -- Custom Metric --------------------------------------------------
def balanced_acc(y_true, y_pred):
    """Balanced accuracy (macro-averaged per-class accuracy)."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

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

# -- Optuna Class-Weight Post-Processing ----------------------------
print("\nRunning Optuna class-weight optimisation...")

def objective(trial):
    cw1 = trial.suggest_float("cw1", 0.5, 3.0)
    cw2 = trial.suggest_float("cw2", 0.5, 3.0)
    cw3 = trial.suggest_float("cw3", 0.5, 3.0)

    class_weights = np.array([cw1, cw2, cw3])
    adjusted_probs = oof_probs * class_weights
    adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)

    return balanced_acc(y.values, np.argmax(adjusted_probs, axis=1))


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name="class_weight_optimization",
)
study.optimize(objective, n_trials=3000, show_progress_bar=True)

print(f"\nBest Optuna Balanced Accuracy: {study.best_value:.6f}")
print(f"  cw_GALAXY    = {study.best_params['cw1']:.4f}")
print(f"  cw_QSO = {study.best_params['cw2']:.4f}")
print(f"  cw_STAR   = {study.best_params['cw3']:.4f}")

# Apply best class weights to test predictions
best_cw = np.array([
    study.best_params["cw1"],
    study.best_params["cw2"],
    study.best_params["cw3"],
])

final_oof_probs = oof_probs * best_cw
final_oof_probs = final_oof_probs / final_oof_probs.sum(axis=1, keepdims=True)


final_test_probs = test_preds * best_cw
final_test_probs = (
    final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
)

# ── Save OOF Predictions ─────────────────────────────────────────────────
OOF_PATH = f"/media/oscarm524/Expansion/Kaggle/2026/PS_Ep6/MODELING_V1/XGB/xgb_3_oof.csv"
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_GALAXY": final_oof_probs[:, 0],
    "prob_QSO": final_oof_probs[:, 1],
    "prob_STAR": final_oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"OOF predictions saved to {OOF_PATH}")

# ── Generate Submission ──────────────────────────────────────────────────
label_map = {0: "GALAXY", 1: "QSO", 2: "STAR"}
test_classes = np.argmax(final_test_probs, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["class"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Best trial: 1308. Best value: 0.965469: 100%|████████████████████████| 3000/3000 [01:37<00:00, 30.79it/s]

# Best Optuna Balanced Accuracy: 0.965469
#   cw_GALAXY    = 2.3159
#   cw_QSO = 2.5311
#   cw_STAR   = 2.9420