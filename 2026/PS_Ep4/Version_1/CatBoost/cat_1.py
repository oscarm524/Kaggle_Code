"""CatBoost model with best hyperparameters from Optuna HPO.

Trains a 5-fold stratified cross-validation CatBoost model using the
best hyperparameter combination found during tuning (with balanced
class weights). Saves out-of-fold predicted probabilities (likelihoods)
and a test submission to CSV.
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
OUTPUT_PATH = f"{BASE_DIR}/Models/CatBoost/cat_1.csv"
OOF_PATH = f"{BASE_DIR}/Models/CatBoost/cat_1_oof.csv"

# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test = test.copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()

# -- Best Hyper-parameters from Optuna -----------------------------
param = {
    "task_type": "GPU",
    "loss_function": "MultiClass",
    "classes_count": 3,
    "depth": 3,
    "learning_rate": 0.16774868966077647,
    "l2_leaf_reg": 0.014778950294680678,
    "iterations": 1000,
    "early_stopping_rounds": 50,
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
label_map = {0: "Low", 1: "Medium", 2: "High"}
test_classes = np.argmax(test_preds, axis=1)

submission = pd.read_csv(SUBMISSION_PATH)
submission["Irrigation_Need"] = [label_map[c] for c in test_classes]
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved to {OUTPUT_PATH}")

# Fold 1 / 5
# 0:      learn: 0.9511192        test: 0.9507339 best: 0.9507339 (0)     total: 6.39ms   remaining: 6.39s
# 100:    learn: 0.1082296        test: 0.1120490 best: 0.1120490 (100)   total: 330ms    remaining: 2.94s
# 200:    learn: 0.0966764        test: 0.1024503 best: 0.1024503 (200)   total: 647ms    remaining: 2.57s
# 300:    learn: 0.0912551        test: 0.0985936 best: 0.0985936 (300)   total: 959ms    remaining: 2.23s
# 400:    learn: 0.0880551        test: 0.0966041 best: 0.0966041 (400)   total: 1.28s    remaining: 1.91s
# 500:    learn: 0.0857035        test: 0.0955285 best: 0.0955285 (500)   total: 1.59s    remaining: 1.59s
# 600:    learn: 0.0835791        test: 0.0945258 best: 0.0945244 (592)   total: 1.91s    remaining: 1.27s
# 700:    learn: 0.0818816        test: 0.0940677 best: 0.0940677 (700)   total: 2.23s    remaining: 953ms
# 800:    learn: 0.0800850        test: 0.0933308 best: 0.0933308 (800)   total: 2.56s    remaining: 636ms
# 900:    learn: 0.0786113        test: 0.0931680 best: 0.0931671 (899)   total: 2.88s    remaining: 316ms
# 999:    learn: 0.0773890        test: 0.0929842 best: 0.0929781 (992)   total: 3.19s    remaining: 0us
# bestTest = 0.09297805535
# bestIteration = 992
# Shrink model to first 993 iterations.
#   Fold 1 Balanced Accuracy: 0.967469

# Fold 2 / 5
# 0:      learn: 0.9506054        test: 0.9497291 best: 0.9497291 (0)     total: 3.61ms   remaining: 3.61s
# 100:    learn: 0.1074653        test: 0.1085405 best: 0.1085405 (100)   total: 336ms    remaining: 2.99s
# 200:    learn: 0.0971623        test: 0.0998053 best: 0.0998053 (200)   total: 673ms    remaining: 2.67s
# 300:    learn: 0.0918251        test: 0.0960001 best: 0.0960001 (300)   total: 1s       remaining: 2.33s
# 400:    learn: 0.0881079        test: 0.0937776 best: 0.0937776 (400)   total: 1.33s    remaining: 1.99s
# 500:    learn: 0.0854929        test: 0.0925316 best: 0.0925226 (499)   total: 1.66s    remaining: 1.65s
# 600:    learn: 0.0833477        test: 0.0916063 best: 0.0915995 (599)   total: 1.98s    remaining: 1.32s
# 700:    learn: 0.0812898        test: 0.0908431 best: 0.0908272 (699)   total: 2.31s    remaining: 988ms
# 800:    learn: 0.0795592        test: 0.0905267 best: 0.0905073 (778)   total: 2.64s    remaining: 657ms
# bestTest = 0.09050731102
# bestIteration = 778
# Shrink model to first 779 iterations.
#   Fold 2 Balanced Accuracy: 0.969045

# Fold 3 / 5
# 0:      learn: 0.9501572        test: 0.9511208 best: 0.9511208 (0)     total: 4.07ms   remaining: 4.07s
# 100:    learn: 0.1075179        test: 0.1075807 best: 0.1075807 (100)   total: 340ms    remaining: 3.03s
# 200:    learn: 0.0960236        test: 0.0976092 best: 0.0976092 (200)   total: 667ms    remaining: 2.65s
# 300:    learn: 0.0914030        test: 0.0944130 best: 0.0944069 (299)   total: 987ms    remaining: 2.29s
# 400:    learn: 0.0884044        test: 0.0925608 best: 0.0925608 (400)   total: 1.32s    remaining: 1.97s
# 500:    learn: 0.0859818        test: 0.0912554 best: 0.0912554 (500)   total: 1.64s    remaining: 1.63s
# 600:    learn: 0.0836995        test: 0.0899309 best: 0.0899309 (600)   total: 1.96s    remaining: 1.3s
# 700:    learn: 0.0816940        test: 0.0890915 best: 0.0890875 (695)   total: 2.29s    remaining: 976ms
# 800:    learn: 0.0800525        test: 0.0883555 best: 0.0883488 (799)   total: 2.62s    remaining: 652ms
# 900:    learn: 0.0786427        test: 0.0879439 best: 0.0879365 (897)   total: 2.94s    remaining: 323ms
# 999:    learn: 0.0773679        test: 0.0875500 best: 0.0875186 (989)   total: 3.26s    remaining: 0us
# bestTest = 0.08751864423
# bestIteration = 989
# Shrink model to first 990 iterations.
#   Fold 3 Balanced Accuracy: 0.970894

# Fold 4 / 5
# 0:      learn: 0.9507301        test: 0.9504566 best: 0.9504566 (0)     total: 4.03ms   remaining: 4.02s
# 100:    learn: 0.1081346        test: 0.1082646 best: 0.1082646 (100)   total: 336ms    remaining: 2.99s
# 200:    learn: 0.0973959        test: 0.0988284 best: 0.0988284 (200)   total: 662ms    remaining: 2.63s
# 300:    learn: 0.0923225        test: 0.0951489 best: 0.0951489 (300)   total: 989ms    remaining: 2.3s
# 400:    learn: 0.0890475        test: 0.0930666 best: 0.0930630 (398)   total: 1.31s    remaining: 1.96s
# 500:    learn: 0.0860963        test: 0.0916053 best: 0.0915953 (497)   total: 1.63s    remaining: 1.63s
# 600:    learn: 0.0840812        test: 0.0906027 best: 0.0906027 (599)   total: 1.96s    remaining: 1.3s
# 700:    learn: 0.0822910        test: 0.0902375 best: 0.0902189 (688)   total: 2.29s    remaining: 978ms
# 800:    learn: 0.0806323        test: 0.0895652 best: 0.0895432 (796)   total: 2.62s    remaining: 650ms
# 900:    learn: 0.0791375        test: 0.0890494 best: 0.0890293 (896)   total: 2.94s    remaining: 324ms
# 999:    learn: 0.0777439        test: 0.0886584 best: 0.0886425 (987)   total: 3.26s    remaining: 0us
# bestTest = 0.08864248091
# bestIteration = 987
# Shrink model to first 988 iterations.
#   Fold 4 Balanced Accuracy: 0.969183

# Fold 5 / 5
# 0:      learn: 0.9512219        test: 0.9519321 best: 0.9519321 (0)     total: 3.19ms   remaining: 3.19s
# 100:    learn: 0.1087457        test: 0.1083928 best: 0.1083928 (100)   total: 331ms    remaining: 2.94s
# 200:    learn: 0.0982027        test: 0.0991071 best: 0.0991071 (200)   total: 662ms    remaining: 2.63s
# 300:    learn: 0.0931707        test: 0.0953827 best: 0.0953827 (300)   total: 994ms    remaining: 2.31s
# 400:    learn: 0.0898135        test: 0.0933795 best: 0.0933795 (400)   total: 1.31s    remaining: 1.97s
# 500:    learn: 0.0869419        test: 0.0918320 best: 0.0918071 (496)   total: 1.65s    remaining: 1.64s
# 600:    learn: 0.0848298        test: 0.0908521 best: 0.0908521 (600)   total: 1.97s    remaining: 1.31s
# 700:    learn: 0.0830455        test: 0.0900854 best: 0.0900800 (691)   total: 2.31s    remaining: 984ms
# 800:    learn: 0.0813494        test: 0.0894718 best: 0.0894516 (796)   total: 2.64s    remaining: 655ms
# 900:    learn: 0.0798986        test: 0.0889810 best: 0.0889809 (884)   total: 2.98s    remaining: 327ms
# 999:    learn: 0.0785562        test: 0.0886279 best: 0.0885677 (979)   total: 3.3s     remaining: 0us
# bestTest = 0.08856765486
# bestIteration = 979
# Shrink model to first 980 iterations.
#   Fold 5 Balanced Accuracy: 0.969070

# Overall OOF Balanced Accuracy: 0.969132