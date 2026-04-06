"""CatBoost model (v2) with logistic-regression stacking feature.

Trains a 5-fold stratified cross-validation CatBoost model using the
best hyperparameter combination found during the second Optuna tuning
round.  A base logistic-regression prediction is added as an extra
categorical feature ("original_pred") before training CatBoost.
Saves out-of-fold predicted probabilities (likelihoods) and a test
submission to CSV.
"""

import gc

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from catboost import CatBoostClassifier, Pool

# -- Paths ---------------------------------------------------------
BASE_DIR = "/media/oscarm524/Expansion/Kaggle/2026/PS_Ep4/Version_1"
TRAIN_PATH = f"{BASE_DIR}/train.csv"
TEST_PATH = f"{BASE_DIR}/test.csv"
SUBMISSION_PATH = f"{BASE_DIR}/sample_submission.csv"
OUTPUT_PATH = f"{BASE_DIR}/Models/CatBoost/cat_2.csv"
OOF_PATH = f"{BASE_DIR}/Models/CatBoost/cat_2_oof.csv"

# -- Load Data -----------------------------------------------------
train = pd.read_csv(TRAIN_PATH, index_col="id")
test = pd.read_csv(TEST_PATH, index_col="id")

X_raw = train.drop(columns=["Irrigation_Need"])
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2})

X_test_raw = test.copy()

# -- Base Logistic Regression (stacking feature) -------------------
cat_cols_lr = X_raw.select_dtypes(include=["object"]).columns
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Prepare numeric representation for logistic regression
X_cat = ohe.fit_transform(X_raw[cat_cols_lr])
X_num = X_raw.drop(columns=cat_cols_lr).values
X_lr = np.hstack([X_num, X_cat])

base_logit = make_pipeline(
    RobustScaler(),
    LogisticRegression(
        C=0.1,
        class_weight="balanced",
        multi_class="multinomial",
        max_iter=1000,
        random_state=42,
    ),
)
base_logit.fit(X_lr, y)

label_map = {0: "Low", 1: "Medium", 2: "High"}

# Train stacking feature
train_lr_pred = base_logit.predict(X_lr)
train_lr_label = [label_map[i] for i in train_lr_pred]
print(
    "Base Logistic Regression Balanced Accuracy:",
    balanced_accuracy_score(y, train_lr_pred),
)

# Test stacking feature
X_test_cat = ohe.transform(X_test_raw[cat_cols_lr])
X_test_num = X_test_raw.drop(columns=cat_cols_lr).values
X_test_lr = np.hstack([X_test_num, X_test_cat])
test_lr_pred = base_logit.predict(X_test_lr)
test_lr_label = [label_map[i] for i in test_lr_pred]

del X_cat, X_num, X_lr, X_test_cat, X_test_num, X_test_lr
gc.collect()

# -- Build final feature sets with stacking feature ----------------
X = train.drop(columns=["Irrigation_Need"])
X["original_pred"] = train_lr_label

X_test = test.copy()
X_test["original_pred"] = test_lr_label

cat_cols = X.select_dtypes(include="object").columns.tolist()

# -- Best Hyper-parameters from Optuna (run 2) ---------------------
param = {
    "loss_function": "MultiClass",
    "classes_count": 3,
    "depth": 3,
    "learning_rate": 0.18546824346860405,
    "l2_leaf_reg": 0.19531502975724344,
    "iterations": 2000,
    "border_count": 1000,
    "early_stopping_rounds": 100,
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

# Fold 1 / 5
# 0:      learn: 0.9027599        test: 0.9026735 best: 0.9026735 (0)     total: 140ms    remaining: 4m 39s
# 100:    learn: 0.1124288        test: 0.1158367 best: 0.1158367 (100)   total: 7.26s    remaining: 2m 16s
# 200:    learn: 0.0953206        test: 0.1004628 best: 0.1004628 (200)   total: 14.9s    remaining: 2m 13s
# 300:    learn: 0.0888194        test: 0.0956294 best: 0.0956294 (300)   total: 22.7s    remaining: 2m 8s
# 400:    learn: 0.0846360        test: 0.0932697 best: 0.0932697 (400)   total: 30.4s    remaining: 2m 1s
# 500:    learn: 0.0815373        test: 0.0920301 best: 0.0920301 (500)   total: 38.1s    remaining: 1m 54s
# 600:    learn: 0.0791046        test: 0.0910094 best: 0.0910094 (600)   total: 45.8s    remaining: 1m 46s
# 700:    learn: 0.0769806        test: 0.0900741 best: 0.0900741 (700)   total: 53.4s    remaining: 1m 38s
# 800:    learn: 0.0750993        test: 0.0891887 best: 0.0891887 (800)   total: 1m 1s    remaining: 1m 31s
# 900:    learn: 0.0732891        test: 0.0883248 best: 0.0883248 (900)   total: 1m 8s    remaining: 1m 23s
# 1000:   learn: 0.0717304        test: 0.0876822 best: 0.0876822 (1000)  total: 1m 16s   remaining: 1m 16s
# 1100:   learn: 0.0702465        test: 0.0874969 best: 0.0874783 (1083)  total: 1m 23s   remaining: 1m 8s
# 1200:   learn: 0.0688238        test: 0.0869292 best: 0.0868948 (1187)  total: 1m 31s   remaining: 1m 1s
# 1300:   learn: 0.0676181        test: 0.0866699 best: 0.0866348 (1293)  total: 1m 39s   remaining: 53.7s
# 1400:   learn: 0.0662044        test: 0.0862855 best: 0.0862837 (1399)  total: 1m 47s   remaining: 45.9s
# 1500:   learn: 0.0650542        test: 0.0861104 best: 0.0860823 (1488)  total: 1m 54s   remaining: 38.2s
# 1600:   learn: 0.0640305        test: 0.0857914 best: 0.0857892 (1599)  total: 2m 2s    remaining: 30.5s
# 1700:   learn: 0.0631280        test: 0.0856109 best: 0.0856102 (1698)  total: 2m 10s   remaining: 22.9s
# 1800:   learn: 0.0621801        test: 0.0854417 best: 0.0853886 (1772)  total: 2m 18s   remaining: 15.3s
# 1900:   learn: 0.0613253        test: 0.0853681 best: 0.0853358 (1893)  total: 2m 25s   remaining: 7.59s
# 1999:   learn: 0.0604687        test: 0.0852749 best: 0.0852749 (1999)  total: 2m 33s   remaining: 0us

# bestTest = 0.08527494926
# bestIteration = 1999

#   Fold 1 Balanced Accuracy: 0.970865

# Fold 2 / 5
# 0:      learn: 0.9033390        test: 0.9036632 best: 0.9036632 (0)     total: 98.2ms   remaining: 3m 16s
# 100:    learn: 0.1120409        test: 0.1128353 best: 0.1128353 (100)   total: 7.33s    remaining: 2m 17s
# 200:    learn: 0.0966400        test: 0.0986547 best: 0.0986547 (200)   total: 15s      remaining: 2m 14s
# 300:    learn: 0.0904287        test: 0.0938016 best: 0.0938016 (300)   total: 22.7s    remaining: 2m 8s
# 400:    learn: 0.0859109        test: 0.0908832 best: 0.0908832 (400)   total: 30.5s    remaining: 2m 1s
# 500:    learn: 0.0822571        test: 0.0886475 best: 0.0886475 (500)   total: 38.2s    remaining: 1m 54s
# 600:    learn: 0.0796163        test: 0.0874630 best: 0.0874630 (600)   total: 46s      remaining: 1m 47s
# 700:    learn: 0.0771323        test: 0.0861062 best: 0.0861062 (700)   total: 53.8s    remaining: 1m 39s
# 800:    learn: 0.0751547        test: 0.0853222 best: 0.0853222 (800)   total: 1m 1s    remaining: 1m 32s
# 900:    learn: 0.0734453        test: 0.0847458 best: 0.0847458 (900)   total: 1m 9s    remaining: 1m 24s
# 1000:   learn: 0.0718234        test: 0.0842218 best: 0.0842218 (1000)  total: 1m 16s   remaining: 1m 16s
# 1100:   learn: 0.0703922        test: 0.0838638 best: 0.0837594 (1061)  total: 1m 24s   remaining: 1m 9s
# 1200:   learn: 0.0689677        test: 0.0836523 best: 0.0835788 (1195)  total: 1m 32s   remaining: 1m 1s
# 1300:   learn: 0.0677775        test: 0.0834602 best: 0.0834473 (1298)  total: 1m 40s   remaining: 53.7s
# 1400:   learn: 0.0667298        test: 0.0832940 best: 0.0832369 (1394)  total: 1m 47s   remaining: 46s
# 1500:   learn: 0.0655962        test: 0.0832202 best: 0.0832106 (1499)  total: 1m 55s   remaining: 38.4s
# 1600:   learn: 0.0644409        test: 0.0828576 best: 0.0828503 (1599)  total: 2m 3s    remaining: 30.7s
# 1700:   learn: 0.0634598        test: 0.0826634 best: 0.0826298 (1673)  total: 2m 10s   remaining: 23s
# 1800:   learn: 0.0624789        test: 0.0826435 best: 0.0825886 (1713)  total: 2m 18s   remaining: 15.3s
# Stopped by overfitting detector  (100 iterations wait)

# bestTest = 0.08258858897
# bestIteration = 1713

# Shrink model to first 1714 iterations.
#   Fold 2 Balanced Accuracy: 0.972193

# Fold 3 / 5
# 0:      learn: 0.9030457        test: 0.9037589 best: 0.9037589 (0)     total: 115ms    remaining: 3m 49s
# 100:    learn: 0.1098440        test: 0.1097619 best: 0.1097619 (100)   total: 7.48s    remaining: 2m 20s
# 200:    learn: 0.0958021        test: 0.0982167 best: 0.0982167 (200)   total: 15.2s    remaining: 2m 16s
# 300:    learn: 0.0890626        test: 0.0927782 best: 0.0927782 (300)   total: 22.9s    remaining: 2m 9s
# 400:    learn: 0.0849847        test: 0.0898774 best: 0.0898774 (400)   total: 30.7s    remaining: 2m 2s
# 500:    learn: 0.0820127        test: 0.0880300 best: 0.0880230 (499)   total: 38.5s    remaining: 1m 55s
# 600:    learn: 0.0793724        test: 0.0866371 best: 0.0866331 (599)   total: 46.2s    remaining: 1m 47s
# 700:    learn: 0.0774670        test: 0.0858228 best: 0.0858078 (696)   total: 53.9s    remaining: 1m 39s
# 800:    learn: 0.0754062        test: 0.0847106 best: 0.0846971 (798)   total: 1m 1s    remaining: 1m 32s
# 900:    learn: 0.0735985        test: 0.0841886 best: 0.0841817 (897)   total: 1m 9s    remaining: 1m 24s
# 1000:   learn: 0.0721384        test: 0.0835779 best: 0.0835729 (998)   total: 1m 16s   remaining: 1m 16s
# 1100:   learn: 0.0706867        test: 0.0831667 best: 0.0831363 (1092)  total: 1m 24s   remaining: 1m 8s
# 1200:   learn: 0.0694017        test: 0.0827033 best: 0.0826776 (1195)  total: 1m 32s   remaining: 1m 1s
# 1300:   learn: 0.0682595        test: 0.0825148 best: 0.0824873 (1277)  total: 1m 39s   remaining: 53.6s
# 1400:   learn: 0.0670054        test: 0.0820446 best: 0.0820304 (1395)  total: 1m 47s   remaining: 45.9s
# 1500:   learn: 0.0657601        test: 0.0818161 best: 0.0818122 (1490)  total: 1m 55s   remaining: 38.2s
# 1600:   learn: 0.0648175        test: 0.0814158 best: 0.0814139 (1599)  total: 2m 2s    remaining: 30.6s
# 1700:   learn: 0.0637765        test: 0.0811740 best: 0.0811565 (1692)  total: 2m 10s   remaining: 22.9s
# 1800:   learn: 0.0628101        test: 0.0810487 best: 0.0810269 (1763)  total: 2m 17s   remaining: 15.2s
# 1900:   learn: 0.0618193        test: 0.0809592 best: 0.0809352 (1889)  total: 2m 25s   remaining: 7.58s
# 1999:   learn: 0.0609554        test: 0.0807045 best: 0.0806902 (1997)  total: 2m 33s   remaining: 0us

# bestTest = 0.08069019345
# bestIteration = 1997

# Shrink model to first 1998 iterations.
#   Fold 3 Balanced Accuracy: 0.973724

# Fold 4 / 5
# 0:      learn: 0.9029864        test: 0.9023233 best: 0.9023233 (0)     total: 98.5ms   remaining: 3m 16s
# 100:    learn: 0.1115334        test: 0.1109756 best: 0.1109756 (100)   total: 7.22s    remaining: 2m 15s
# 200:    learn: 0.0966041        test: 0.0975133 best: 0.0975133 (200)   total: 14.9s    remaining: 2m 13s
# 300:    learn: 0.0897467        test: 0.0919548 best: 0.0919336 (298)   total: 22.4s    remaining: 2m 6s
# 400:    learn: 0.0857402        test: 0.0893791 best: 0.0893791 (400)   total: 30.1s    remaining: 2m
# 500:    learn: 0.0825568        test: 0.0876016 best: 0.0875661 (497)   total: 37.9s    remaining: 1m 53s
# 600:    learn: 0.0801721        test: 0.0865177 best: 0.0865068 (597)   total: 45.5s    remaining: 1m 45s
# 700:    learn: 0.0778345        test: 0.0852031 best: 0.0852031 (700)   total: 53.1s    remaining: 1m 38s
# 800:    learn: 0.0761739        test: 0.0847537 best: 0.0847314 (793)   total: 1m       remaining: 1m 31s
# 900:    learn: 0.0744235        test: 0.0841648 best: 0.0841544 (897)   total: 1m 8s    remaining: 1m 23s
# 1000:   learn: 0.0727697        test: 0.0835402 best: 0.0835402 (1000)  total: 1m 16s   remaining: 1m 15s
# 1100:   learn: 0.0714043        test: 0.0832652 best: 0.0832303 (1083)  total: 1m 23s   remaining: 1m 8s
# 1200:   learn: 0.0699180        test: 0.0825251 best: 0.0825251 (1200)  total: 1m 31s   remaining: 1m
# 1300:   learn: 0.0684867        test: 0.0820512 best: 0.0820374 (1299)  total: 1m 39s   remaining: 53.3s
# 1400:   learn: 0.0672074        test: 0.0817183 best: 0.0816917 (1393)  total: 1m 46s   remaining: 45.7s
# Stopped by overfitting detector  (100 iterations wait)

# bestTest = 0.08169171525
# bestIteration = 1393

# Shrink model to first 1394 iterations.
#   Fold 4 Balanced Accuracy: 0.971917

# Fold 5 / 5
# 0:      learn: 0.9038606        test: 0.9035776 best: 0.9035776 (0)     total: 92.4ms   remaining: 3m 4s
# 100:    learn: 0.1110874        test: 0.1113248 best: 0.1113248 (100)   total: 7.18s    remaining: 2m 15s
# 200:    learn: 0.0964014        test: 0.0979739 best: 0.0979739 (200)   total: 14.8s    remaining: 2m 12s
# 300:    learn: 0.0905102        test: 0.0931786 best: 0.0931725 (297)   total: 22.4s    remaining: 2m 6s
# 400:    learn: 0.0858195        test: 0.0899169 best: 0.0899169 (400)   total: 30s      remaining: 1m 59s
# 500:    learn: 0.0825985        test: 0.0879768 best: 0.0879768 (500)   total: 37.7s    remaining: 1m 52s
# 600:    learn: 0.0797487        test: 0.0862286 best: 0.0862286 (600)   total: 45.3s    remaining: 1m 45s
# 700:    learn: 0.0774098        test: 0.0854602 best: 0.0854602 (700)   total: 53s      remaining: 1m 38s
# 800:    learn: 0.0754776        test: 0.0846493 best: 0.0846493 (800)   total: 1m       remaining: 1m 30s
# 900:    learn: 0.0736877        test: 0.0840329 best: 0.0840236 (899)   total: 1m 8s    remaining: 1m 23s
# 1000:   learn: 0.0720000        test: 0.0834892 best: 0.0834892 (1000)  total: 1m 15s   remaining: 1m 15s
# 1100:   learn: 0.0705498        test: 0.0830162 best: 0.0830162 (1100)  total: 1m 23s   remaining: 1m 8s
# 1200:   learn: 0.0690638        test: 0.0825410 best: 0.0825273 (1198)  total: 1m 31s   remaining: 1m
# 1300:   learn: 0.0677331        test: 0.0822357 best: 0.0822147 (1275)  total: 1m 39s   remaining: 53.3s
# 1400:   learn: 0.0665126        test: 0.0818776 best: 0.0818655 (1373)  total: 1m 46s   remaining: 45.7s
# 1500:   learn: 0.0653443        test: 0.0816004 best: 0.0815975 (1499)  total: 1m 54s   remaining: 38.1s
# 1600:   learn: 0.0642865        test: 0.0814549 best: 0.0814466 (1598)  total: 2m 2s    remaining: 30.5s
# 1700:   learn: 0.0632648        test: 0.0813676 best: 0.0813181 (1672)  total: 2m 9s    remaining: 22.8s
# 1800:   learn: 0.0623967        test: 0.0813632 best: 0.0812877 (1737)  total: 2m 17s   remaining: 15.2s
# Stopped by overfitting detector  (100 iterations wait)

# bestTest = 0.08128768636
# bestIteration = 1737

# Shrink model to first 1738 iterations.
#   Fold 5 Balanced Accuracy: 0.972847

# Overall OOF Balanced Accuracy: 0.972309