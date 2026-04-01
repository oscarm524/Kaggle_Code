"""Optuna-based ensemble of XGBoost, LightGBM, and CatBoost OOF predictions.

Uses Optuna to find the optimal positive weights (summing to 1) for
blending out-of-fold predicted probabilities from three gradient-boosted
tree models.  The objective maximises balanced accuracy.  Once the best
weights are found, they are applied to the corresponding test-set
predicted likelihoods to produce a final submission CSV.
"""

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import balanced_accuracy_score

# -- Paths -----------------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
ENSEMBLE_DIR = f"{BASE_DIR}/Models/Optuna_Ensemble"

OOF_XGB_PATH = f"{ENSEMBLE_DIR}/xgb_2_oof.csv"
OOF_LGB_PATH = f"{ENSEMBLE_DIR}/lgb_2_oof.csv"
OOF_CAT_PATH = f"{ENSEMBLE_DIR}/cat_1_oof.csv"

TEST_XGB_PATH = f"{ENSEMBLE_DIR}/xgb_2_test.csv"
TEST_LGB_PATH = f"{ENSEMBLE_DIR}/lgb_2_test.csv"
TEST_CAT_PATH = f"{ENSEMBLE_DIR}/cat_1_test.csv"

TRAIN_PATH = f"{BASE_DIR}/train.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{ENSEMBLE_DIR}/optuna_ens_sub_1.csv"

PROB_COLS = ["prob_Low", "prob_Medium", "prob_High"]

# -- Load OOF Predictions --------------------------------------------------
oof_xgb = pd.read_csv(OOF_XGB_PATH, index_col="id")
oof_lgb = pd.read_csv(OOF_LGB_PATH, index_col="id")
oof_cat = pd.read_csv(OOF_CAT_PATH, index_col="id")

probs_1 = oof_xgb[PROB_COLS].values
probs_2 = oof_lgb[PROB_COLS].values
probs_3 = oof_cat[PROB_COLS].values

# -- Load Target -----------------------------------------------------------
y = (
    pd.read_csv(TRAIN_PATH)["Irrigation_Need"]
    .map({"Low": 0, "Medium": 1, "High": 2})
)


# -- Optuna Objective ------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Return balanced accuracy for a weighted blend of three models."""
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = trial.suggest_float("w2", 0.0, 1.0 - w1)
    w3 = 1.0 - w1 - w2

    blended = w1 * probs_1 + w2 * probs_2 + w3 * probs_3
    preds = blended.argmax(axis=1)

    return balanced_accuracy_score(y, preds)


# -- Run Optimisation ------------------------------------------------------
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
)
study.optimize(objective, n_trials=1000, show_progress_bar=True)

best_w1 = study.best_params["w1"]
best_w2 = study.best_params["w2"]
best_w3 = 1.0 - best_w1 - best_w2

print(f"\nBest balanced accuracy: {study.best_value:.6f}")
print(
    f"Best weights: w1={best_w1:.4f}, w2={best_w2:.4f}, w3={best_w3:.4f}"
)

# -- Generate Test Submission -----------------------------------------------
test_probs_1 = pd.read_csv(TEST_XGB_PATH, index_col="id")[PROB_COLS].values
test_probs_2 = pd.read_csv(TEST_LGB_PATH, index_col="id")[PROB_COLS].values
test_probs_3 = pd.read_csv(TEST_CAT_PATH, index_col="id")[PROB_COLS].values

blended_test = (
    best_w1 * test_probs_1
    + best_w2 * test_probs_2
    + best_w3 * test_probs_3
)
test_preds = blended_test.argmax(axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = pd.Categorical.from_codes(
    test_preds, categories=["Low", "Medium", "High"]
)

print(submission.head())
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nEnsemble submission saved to {OUTPUT_PATH}")
