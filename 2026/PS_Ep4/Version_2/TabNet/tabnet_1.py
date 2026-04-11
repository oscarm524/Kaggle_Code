"""TabNet model (v1).

Trains a 5-fold stratified cross-validation TabNetClassifier based on the
Kaggle notebook "pss6e4-lgb-baselinecv-0-97943" by yunsuxiaozi.
Uses digit-extraction feature engineering, target encoding, and frequency
encoding.  Includes Optuna class-weight post-processing optimisation.
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import pandas as pd
import numpy as np
import random
import warnings

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -- Reproducibility ------------------------------------------------
SEED = 2026


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


seed_everything(SEED)

# -- Paths ----------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
SAVE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_2/TabNet"
OUTPUT_PATH = f"{SAVE_DIR}/tabnet_1.csv"
OOF_PATH = f"{SAVE_DIR}/tabnet_1_oof.csv"
TEST_PROBS_PATH = f"{SAVE_DIR}/tabnet_1_test.csv"

# -- Load Data ------------------------------------------------------
TARGET_COL = "Irrigation_Need"
NUM_CLASSES = 3

train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL].map({"Low": 0, "Medium": 1, "High": 2})
X_test = test.copy()

label_map = {0: "Low", 1: "Medium", 2: "High"}

# -- Detect column types -------------------------------------------
CAT_COLS = [c for c in X_test.columns if X[c].dtype == object]
NUM_COLS = [c for c in X_test.columns if c not in CAT_COLS]

# -- Feature Engineering --------------------------------------------
M = X[NUM_COLS].max()


def feature_engineering(df):
    """Digit extraction + rounding per the Kaggle notebook."""
    for c in NUM_COLS:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10 ** k) % 10).astype("int8")

        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    return df


X = feature_engineering(X)
X_test = feature_engineering(X_test)

# Drop constant columns
DROP = [c for c in X_test.columns if X_test[c].nunique() == 1]
if DROP:
    print(f"Dropping constant columns: {DROP}")
    X.drop(DROP, axis=1, inplace=True)
    X_test.drop(DROP, axis=1, inplace=True)

# Frequency-encode categoricals + digit columns
CATEGORY = CAT_COLS + [c for c in X_test.columns if "digit" in c]
for c in CATEGORY:
    freq = X[c].value_counts()
    mapping = {
        val: idx
        for idx, (val, count) in enumerate(freq[freq >= 5].items())
    }
    mapping_default = len(mapping)
    X[c] = X[c].map(lambda x, m=mapping, d=mapping_default: m.get(x, d))
    X_test[c] = X_test[c].map(lambda x, m=mapping, d=mapping_default: m.get(x, d))

FEATURES = CATEGORY + NUM_COLS

# -- Custom Metric --------------------------------------------------


def balanced_acc(y_true, y_pred):
    """Balanced accuracy (macro-averaged per-class accuracy)."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = NUM_CLASSES
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc


# -- Hyper-parameters -----------------------------------------------
tabnet_params = {
    "seed": SEED,
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-4,
    "optimizer_params": {"lr": 2e-2},
    "scheduler_params": {"step_size": 10, "gamma": 0.9},
    "scheduler_fn": __import__("torch.optim.lr_scheduler", fromlist=["StepLR"]).StepLR,
    "verbose": 1,
    "device_name": "cuda",
}

# -- 5-Fold CV Training & Test Prediction ---------------------------
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
oof_probs = np.zeros((len(X), NUM_CLASSES))
test_preds = np.zeros((len(X_test), NUM_CLASSES))
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1} / {N_FOLDS}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Target encoding
    te = TargetEncoder(
        target_type="multiclass", smooth="auto", cv=5, random_state=42,
    )
    X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
    X_valid_enc = te.transform(X_valid[FEATURES])
    X_test_enc = te.transform(X_test[FEATURES])

    X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
    X_valid_enc = pd.DataFrame(X_valid_enc, index=X_valid.index)
    X_test_enc = pd.DataFrame(X_test_enc, index=X_test.index)

    X_train_fold = pd.concat([X_train, X_train_enc], axis=1)
    X_valid_fold = pd.concat([X_valid, X_valid_enc], axis=1)
    X_test_fold = pd.concat([X_test, X_test_enc], axis=1)

    # Drop original categorical columns (now target-encoded)
    X_train_fold = X_train_fold.drop(CAT_COLS, axis=1)
    X_valid_fold = X_valid_fold.drop(CAT_COLS, axis=1)
    X_test_fold = X_test_fold.drop(CAT_COLS, axis=1)

    # Deduplicate column names introduced by target-encoding concatenation
    X_train_fold.columns = [str(c) for c in range(len(X_train_fold.columns))]
    X_valid_fold.columns = [str(c) for c in range(len(X_valid_fold.columns))]
    X_test_fold.columns = [str(c) for c in range(len(X_test_fold.columns))]

    model = TabNetClassifier(**tabnet_params)
    model.fit(
        X_train_fold.values, y_train.values,
        eval_set=[(X_valid_fold.values, y_valid.values)],
        eval_metric=["balanced_accuracy"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
    )

    # OOF predictions
    oof_probs[valid_idx] = model.predict_proba(X_valid_fold.values)
    oof_preds[valid_idx] = np.argmax(oof_probs[valid_idx], axis=1)

    fold_score = balanced_accuracy_score(y_valid, oof_preds[valid_idx])
    fold_scores.append(fold_score)
    print(f"  Fold {fold + 1} Balanced Accuracy: {fold_score:.6f}")

    # Test predictions (average probabilities across folds)
    test_preds += model.predict_proba(X_test_fold.values) / N_FOLDS

overall_score = np.mean(fold_scores)
print(f"\nOverall OOF Balanced Accuracy: {overall_score:.6f}")

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
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(f"\nBest Optuna Balanced Accuracy: {study.best_value:.6f}")
print(f"  cw_Low    = {study.best_params['cw1']:.4f}")
print(f"  cw_Medium = {study.best_params['cw2']:.4f}")
print(f"  cw_High   = {study.best_params['cw3']:.4f}")

# Apply best class weights to test predictions
best_cw = np.array([
    study.best_params["cw1"],
    study.best_params["cw2"],
    study.best_params["cw3"],
])
final_test_probs = test_preds * best_cw
final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)

# -- Save OOF Predictions ------------------------------------------
oof_df = pd.DataFrame({
    "id": X.index,
    "prob_Low": oof_probs[:, 0],
    "prob_Medium": oof_probs[:, 1],
    "prob_High": oof_probs[:, 2],
})
oof_df.to_csv(OOF_PATH, index=False)
print(f"\nOOF predictions saved to {OOF_PATH}")

# -- Save Test Predicted Likelihoods --------------------------------
test_probs_df = pd.DataFrame({
    "id": X_test.index,
    "prob_Low": final_test_probs[:, 0],
    "prob_Medium": final_test_probs[:, 1],
    "prob_High": final_test_probs[:, 2],
})
test_probs_df.to_csv(TEST_PROBS_PATH, index=False)
print(f"Test predicted likelihoods saved to {TEST_PROBS_PATH}")

# -- Generate Submission --------------------------------------------
test_classes = np.argmax(final_test_probs, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission[TARGET_COL] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Overall OOF Balanced Accuracy: 0.974085

# Running Optuna class-weight optimisation...
# Best trial: 183. Best value: 0.976985: 100%|████████████████████████████████████████████████████████████████████████| 200/200 [00:04<00:00, 43.61it/s]

# Best Optuna Balanced Accuracy: 0.976985
#   cw_Low    = 0.5536
#   cw_Medium = 0.6399
#   cw_High   = 2.9554